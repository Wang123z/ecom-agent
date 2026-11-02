"""把项目数据转成 LLaMA-Factory 的 sharegpt 格式。

SFT  data/sft_train.jsonl → data/factory/sft.jsonl
DPO  data/preference.jsonl → data/factory/dpo.jsonl

LLaMA-Factory 用 human/gpt 作为角色标签，本项目用 user/assistant，需转换。
用法  python -m train.to_factory
"""
import json
from pathlib import Path

from app import config

ROLE = {"user": "human", "assistant": "gpt", "system": "system"}


def convert_sft(src: Path, dst: Path) -> int:
    rows = []
    for line in src.open(encoding="utf-8"):
        if not line.strip():
            continue
        item = json.loads(line)
        rows.append({"conversations": [
            {"from": ROLE.get(m["from"], m["from"]), "value": m["value"]}
            for m in item["conversations"]
        ]})
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def convert_dpo(src: Path, dst: Path) -> int:
    rows = []
    for line in src.open(encoding="utf-8"):
        if not line.strip():
            continue
        item = json.loads(line)
        rows.append({
            "conversations": [{"from": "human", "value": item["prompt"]}],
            "chosen": {"from": "gpt", "value": item["chosen"]},
            "rejected": {"from": "gpt", "value": item["rejected"]},
        })
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    out = config.DATA_DIR / "factory"
    sft = convert_sft(config.DATA_DIR / "sft_train.jsonl", out / "sft.jsonl")
    dpo = convert_dpo(config.DATA_DIR / "preference.jsonl", out / "dpo.jsonl")
    print(json.dumps({"sft": sft, "dpo": dpo, "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
