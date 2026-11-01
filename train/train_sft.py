"""SFT：LoRA 微调客服对话数据。

用法
  python -m train.train_sft --config train/configs/sft.yaml --dry-run   # 只校验数据与分词
  python -m train.train_sft --config train/configs/sft.yaml
"""
import argparse
import json
import os
import re
from pathlib import Path

import yaml

from app import config


def load_config(path: str) -> dict:
    """读 YAML，支持 ${VAR} 引用环境变量（模型路径按环境切换）。

    取值顺序：环境变量 → app.config 里的同名配置 → 报错。
    找不到时明确报错，避免把 ${MODEL_DIR} 原样传给 transformers。
    """

    def substitute(match):
        key = match.group(1)
        value = os.getenv(key) or getattr(config, key, None)
        if value is None:
            raise ValueError(f"配置里的 ${{{key}}} 找不到对应值，请在 .env 里设置 {key}")
        return str(value)

    raw = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(re.sub(r"\$\{(\w+)\}", substitute, raw))


def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def to_messages(row: dict) -> dict:
    """本地 ShareGPT 格式转 TRL 的 messages 格式。"""
    return {"messages": [{"role": t["from"], "content": t["value"]} for t in row["conversations"]]}


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else config.BASE_DIR / p


def build_dataset(cfg: dict, split: str):
    from datasets import Dataset

    rows = load_jsonl(resolve_path(cfg["data"][split]))
    return Dataset.from_list([to_messages(r) for r in rows])


def dry_run(cfg: dict) -> None:
    """不训练：检查数据格式、chat template、token 长度分布。"""
    model_path = cfg["model"]["name"]
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型目录不存在: {model_path}\n请检查 .env 里的 MODEL_DIR 是否指向模型所在目录")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    train_rows = [to_messages(r) for r in load_jsonl(resolve_path(cfg["data"]["train"]))]
    val_rows = [to_messages(r) for r in load_jsonl(resolve_path(cfg["data"]["val"]))]

    # 先渲染成文本再分词：tokenize=True 在新版 transformers 返回 dict，len() 会得到 2
    def token_len(messages: list) -> int:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=False, enable_thinking=False)
        return len(tok(text)["input_ids"])

    lengths = sorted(token_len(r["messages"]) for r in train_rows[:200])
    sample = tok.apply_chat_template(train_rows[0]["messages"], tokenize=False,
                                     add_generation_prompt=False, enable_thinking=False)

    print(json.dumps({
        "train": len(train_rows),
        "val": len(val_rows),
        "tokens_min": lengths[0],
        "tokens_p50": lengths[len(lengths) // 2],
        "tokens_max": lengths[-1],
        "max_length": cfg["model"]["max_length"],
        "over_limit": sum(1 for n in lengths if n > cfg["model"]["max_length"]),
        "has_system": train_rows[0]["messages"][0]["role"] == "system",
        "has_think": "<think>" in sample,
        "sample_prompt": sample[:300],
    }, ensure_ascii=False, indent=2))


def pick_precision() -> dict:
    """有 bf16 用 bf16，否则 fp16，都没有就 fp32。"""
    import torch

    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False, "dtype": torch.float32}
    if torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False, "dtype": torch.bfloat16}
    return {"bf16": False, "fp16": True, "dtype": torch.float16}


def build_trainer(cfg: dict, out_dir: Path):
    """按 YAML 组装 SFTTrainer。参数用签名过滤，规避 TRL 版本差异。"""
    import inspect

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    precision = pick_precision()
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], torch_dtype=precision["dtype"],
        device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    t, lora = cfg["training"], cfg["lora"]
    raw = dict(
        output_dir=str(out_dir), do_train=True, do_eval=True,
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["lr"], warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type="cosine", logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"], eval_strategy=t["save_strategy"],
        save_total_limit=2, report_to="none", gradient_checkpointing=True,
        max_length=cfg["model"]["max_length"],
        bf16=precision["bf16"], fp16=precision["fp16"],
        remove_unused_columns=False,
        # 数据里没有推理过程，关掉 Qwen3 的思考块，避免训练出空的 <think>
        chat_template_kwargs={"enable_thinking": False},
    )
    allowed = set(inspect.signature(SFTConfig.__init__).parameters)
    args = SFTConfig(**{k: v for k, v in raw.items() if k in allowed})

    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM", r=lora["rank"], lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"], target_modules=lora["targets"],
    )
    trainer = SFTTrainer(
        model=model, args=args,
        train_dataset=build_dataset(cfg, "train"),
        eval_dataset=build_dataset(cfg, "val"),
        processing_class=tok, peft_config=peft_cfg,
    )
    return trainer, tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/configs/sft.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dry_run:
        dry_run(cfg)
        return

    import torch

    out_dir = resolve_path(cfg["output"]["dir"])
    art_dir = resolve_path(cfg["output"]["artifact_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    trainer, tok = build_trainer(cfg, out_dir)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    result = trainer.train()
    trainer.save_model()
    trainer.save_state()
    tok.save_pretrained(out_dir)

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    summary = {
        "train_samples": len(trainer.train_dataset),
        "eval_samples": len(trainer.eval_dataset),
        "global_step": trainer.state.global_step,
        "final_loss": result.metrics.get("train_loss"),
        "trainable_params": trainable,
        "trainable_percent": round(trainable / total * 100, 4) if total else 0.0,
        "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        if torch.cuda.is_available() else None,
    }
    (art_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (art_dir / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
