"""DPO：在 SFT adapter 上做偏好对齐。

参考策略必须是「冻结的 SFT 模型」，不是 base 模型。部分 TRL 版本在 PEFT
场景下会退化成 base，所以这里复制一份 adapter 当 reference 并冻结。

用法
  python -m train.train_dpo --config train/configs/dpo.yaml --dry-run
  python -m train.train_dpo --config train/configs/dpo.yaml
"""
import argparse
import json
from pathlib import Path

from app import config
from train.train_sft import load_config, pick_precision, resolve_path


def load_pairs(path: Path) -> list:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def build_dataset(path: Path):
    from datasets import Dataset

    return Dataset.from_list(load_pairs(path))


def clone_adapter(model, src: str, dst: str) -> None:
    """复制并冻结一份 adapter，作为 DPO 的参考策略。"""
    import torch

    if dst in model.peft_config:
        return
    model.add_adapter(dst, model.peft_config[src])
    src_marker, dst_marker = f".{src}.", f".{dst}."
    with torch.no_grad():
        for name, param in model.named_parameters():
            if src_marker in name:
                model.get_parameter(name.replace(src_marker, dst_marker)).data.copy_(param.data)
    for name, param in model.named_parameters():
        if dst_marker in name:
            param.requires_grad_(False)


def build_trainer(cfg: dict, out_dir: Path):
    import inspect

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    precision = pick_precision()
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], torch_dtype=precision["dtype"],
        device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(resolve_path(cfg["model"]["sft_adapter"])),
                                      adapter_name="default", is_trainable=True)
    clone_adapter(model, "default", "ref")
    model.set_adapter("default")
    model.config.use_cache = False

    t, d = cfg["training"], cfg["dpo"]
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
        beta=d["beta"], loss_type=d["loss_type"],
        max_length=cfg["model"]["max_length"],
        bf16=precision["bf16"], fp16=precision["fp16"],
        remove_unused_columns=False,
        ref_adapter_name="ref",
    )
    allowed = set(inspect.signature(DPOConfig.__init__).parameters)
    args = DPOConfig(**{k: v for k, v in raw.items() if k in allowed})

    trainer = DPOTrainer(
        model=model, ref_model=None, args=args,
        train_dataset=build_dataset(resolve_path(cfg["data"]["train"])),
        eval_dataset=build_dataset(resolve_path(cfg["data"]["val"])),
        processing_class=tok,
    )
    return trainer, tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/configs/dpo.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.dry_run:
        pairs = load_pairs(resolve_path(cfg["data"]["train"]))
        first = pairs[0] if pairs else {}
        print(json.dumps({
            "pairs": len(pairs),
            "has_prompt": bool(first.get("prompt")),
            "chosen_turns": len(first.get("chosen", [])),
            "rejected_turns": len(first.get("rejected", [])),
            "sample_prompt": first.get("prompt", [{}])[0].get("content", "")[:80],
            "beta": cfg["dpo"]["beta"],
        }, ensure_ascii=False, indent=2))
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

    metrics = {k: v for k, v in result.metrics.items() if "rewards" in k or "loss" in k}
    summary = {
        "pairs": len(trainer.train_dataset),
        "global_step": trainer.state.global_step,
        "metrics": {k: round(float(v), 4) for k, v in metrics.items() if isinstance(v, (int, float))},
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
