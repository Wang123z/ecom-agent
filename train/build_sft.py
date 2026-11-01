"""SFT 数据构造：单轮 QA 与同簇多轮拼接，统一转成 ShareGPT 格式。

输入  data/train.jsonl  data/clusters.json（可选，缺失则全为单轮）
输出  data/sft_train.jsonl  data/sft_val.jsonl
格式  {"id":"s_001","conversations":[{"from":"user","value":...},{"from":"assistant","value":...}]}
用法  python -m train.build_sft --multi-ratio 0.3
"""
import argparse
import json
import random
from pathlib import Path

from app import config

SYSTEM_PROMPT = "你是电商售后客服，回答要准确、简洁、给出可执行的下一步操作，不编造订单状态与平台政策。"


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def to_sample(sid: str, turns: list) -> dict:
    """turns 是 [(question, answer), ...]，拼成带 system 的 ShareGPT 样本。"""
    conversations = [{"from": "system", "value": SYSTEM_PROMPT}]
    for question, answer in turns:
        conversations.append({"from": "user", "value": question})
        conversations.append({"from": "assistant", "value": answer})
    return {"id": sid, "conversations": conversations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.DATA_DIR / "train.jsonl"))
    parser.add_argument("--clusters", default=str(config.DATA_DIR / "clusters.json"))
    parser.add_argument("--output-dir", default=str(config.DATA_DIR))
    parser.add_argument("--multi-ratio", type=float, default=0.3, help="多轮样本占比")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_jsonl(Path(args.input))
    clusters = json.loads(Path(args.clusters).read_text(encoding="utf-8")) if Path(args.clusters).exists() else []

    samples = [to_sample(f"s_{i:05d}", [(r["question"], r["answer"])]) for i, r in enumerate(rows)]

    # 多轮：只取同簇内答案一致的簇，避免模型学到矛盾口径
    usable = [c for c in clusters if c["size"] >= 2 and not c.get("conflict")]
    target = int(len(samples) * args.multi_ratio / max(1, args.max_turns - 1))
    for c in usable[:target]:
        turns = list(zip(c["questions"], c["answers"]))[:args.max_turns]
        samples.append(to_sample(f"m_{c['cluster_id']}", turns))

    rng.shuffle(samples)
    cut = int(len(samples) * (1 - args.val_ratio))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in (("sft_train.jsonl", samples[:cut]), ("sft_val.jsonl", samples[cut:])):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    multi = sum(1 for s in samples if len(s["conversations"]) > 3)
    print(json.dumps({
        "single_turn": len(samples) - multi,
        "multi_turn": multi,
        "train": cut,
        "val": len(samples) - cut,
        "usable_clusters": len(usable),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
