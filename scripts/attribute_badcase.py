"""bad case 归因：把失败样本分到四类，输出清单供针对性修复。

四类
  检索没召回  citations 为空
  召回但答错  citations 非空但答案不含期望要素
  分支走错    intent 与预期不符
  工具失败    调用抛异常

用法  python -m scripts.attribute_badcase
输出  reports/badcase.json
"""
import json
from datetime import date
from pathlib import Path

from app import config
from eval.run_all import load


def classify(row: dict, out: dict) -> str:
    if out.get("error"):
        return "工具失败"
    if out.get("intent") != row["expect_intent"]:
        return "分支走错"
    answer = out.get("answer", "") or ""
    if row.get("expect_contains") and row["expect_contains"] not in answer:
        return "召回但答错" if out.get("citations") else "检索没召回"
    return ""


def main() -> None:
    from app.core import graph

    rows = load("regression")
    buckets = {"检索没召回": [], "召回但答错": [], "分支走错": [], "工具失败": []}

    for row in rows:
        try:
            out = graph.chat(row["query"], "eval", f"bc-{row['id']}")
        except Exception as exc:
            out = {"error": str(exc)[:120]}
        kind = classify(row, out)
        if kind:
            buckets[kind].append({
                "id": row["id"], "query": row["query"],
                "expect": row["expect_intent"], "got": out.get("intent"),
                "answer": (out.get("answer") or "")[:100],
            })

    out_path = config.BASE_DIR / "reports" / "badcase.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "date": str(date.today()),
        "total": len(rows),
        "failed": sum(len(v) for v in buckets.values()),
        "buckets": {k: {"count": len(v), "items": v} for k, v in buckets.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, items in buckets.items():
        print(f"{name}: {len(items)} 条")
        for it in items[:3]:
            print(f"  - {it['query']}（期望 {it['expect']}，实际 {it['got']}）")
    print(f"\n明细：{out_path}")


if __name__ == "__main__":
    main()
