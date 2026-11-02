"""按品类配对生成两轮对话，把多轮样本比例拉到目标值。

真实多轮数据没有，只能规则配对：同品类的两条问题拼成两轮，
第二问加承接词（「那...呢」），保证领域一致、语气连贯。

用法  python -m scripts.build_multiturn --ratio 0.3
输出  data/multi_turn_gen.jsonl（build_sft 会自动合并）
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from app import config

# 第二问的承接句式，按第一问的结尾语气选
BRIDGES = ("那{core}呢", "如果是{core}的话呢", "{core}这种情况怎么算", "还有{core}吗")


def core_of(question: str) -> str:
    """截掉疑问词，留核心名词短语。"""
    for suffix in ("怎么办", "怎么处理", "能退吗", "可以退吗", "是什么", "吗", "呢", "？", "?"):
        if question.endswith(suffix):
            return question[: -len(suffix)].strip()
    return question[:8]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.DATA_DIR / "train.jsonl"))
    parser.add_argument("--output", default=str(config.DATA_DIR / "multi_turn_gen.jsonl"))
    parser.add_argument("--ratio", type=float, default=0.3, help="多轮样本占单轮样本的比例")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).open(encoding="utf-8") if line.strip()]
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row.get("category") or "通用"].append(row)

    target = int(len(rows) * args.ratio / 2)      # 每个多轮样本消耗 2 条单轮
    rng = random.Random(args.seed)
    out = []

    cats = [c for c, items in by_cat.items() if len(items) >= 4]
    rng.shuffle(cats)
    for cat in cats:
        items = by_cat[cat][:]
        rng.shuffle(items)
        for i in range(0, len(items) - 1, 2):
            if len(out) >= target:
                break
            first, second = items[i], items[i + 1]
            bridge = rng.choice(BRIDGES).format(core=core_of(second["question"]))
            out.append({
                "id": f"mt_{first['id']}_{second['id']}",
                "turns": [
                    {"user": first["question"], "assistant": first["answer"]},
                    {"user": bridge, "assistant": second["answer"]},
                ],
                "category": cat,
            })
        if len(out) >= target:
            break

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"single": len(rows), "generated": len(out),
                      "ratio": round(len(out) * 2 / len(rows), 3) if rows else 0,
                      "output": str(path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
