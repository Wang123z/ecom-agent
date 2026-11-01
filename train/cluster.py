"""问题聚类：把问法不同但语义相近的问题合并成一簇。

用途有两个：构造多轮对话（同簇问题可串成追问）、发现答案冲突（同簇答案矛盾要治理）。
纯 Python 字符 bigram + 余弦，不依赖向量模型，本地可直接跑。

输入  data/train.jsonl
输出  data/clusters.json
用法  python -m train.cluster --threshold 0.75
"""
import argparse
import json
from pathlib import Path

from app import config
from train.sim import cos, encode_all


def find(parent: list, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def union(parent: list, i: int, j: int) -> None:
    ri, rj = find(parent, i), find(parent, j)
    if ri != rj:
        parent[rj] = ri


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.DATA_DIR / "train.jsonl"))
    parser.add_argument("--output", default=str(config.DATA_DIR / "clusters.json"))
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--backend", choices=["bigram", "bge"], default="bigram")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).open(encoding="utf-8") if line.strip()]
    vectors = encode_all([r["question"] for r in rows], args.backend)

    parent = list(range(len(rows)))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if cos(vectors[i], vectors[j], args.backend) >= args.threshold:
                union(parent, i, j)

    groups: dict = {}
    for idx, row in enumerate(rows):
        groups.setdefault(find(parent, idx), []).append(row)

    clusters = []
    for cid, items in groups.items():
        clusters.append({
            "cluster_id": f"c{len(clusters):04d}",
            "size": len(items),
            "questions": [it["question"] for it in items],
            "answers": [it["answer"] for it in items],
            "conflict": len({it["answer"] for it in items}) > 1,
        })
    clusters.sort(key=lambda c: -c["size"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(clusters, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "questions": len(rows),
        "clusters": len(clusters),
        "singletons": sum(1 for c in clusters if c["size"] == 1),
        "largest": clusters[0]["size"] if clusters else 0,
        "conflicts": sum(1 for c in clusters if c["conflict"]),
        "output": args.output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
