"""一条命令跑评测，输出 JSON 报告并与上一次对比。

用法
  python -m eval.run_all --suite business          # 纯规则，无需任何外部服务
  python -m eval.run_all --suite retrieval         # 需要 bge-m3 + Milvus
  python -m eval.run_all --suite generation        # 需要 LLM
  python -m eval.run_all --suite all

报告落在 reports/YYYY-MM-DD.json，同时打印与上一份的差值。
"""
import argparse
import json
from datetime import date
from pathlib import Path

from app import config
from eval import metrics

DATASET_DIR = Path(__file__).resolve().parent / "datasets"


def load(name: str) -> list:
    path = DATASET_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def run_retrieval() -> dict:
    from app.core.rag import retriever

    rows = load("retrieval")
    return metrics.evaluate(rows, lambda q: retriever.recall(q, config.RECALL_TOP_K))


def run_generation() -> dict:
    from app.core import graph
    from eval import judge

    rows = load("generation")
    scored, failed = [], []
    for row in rows:
        try:
            out = graph.chat(row["query"], "eval", f"eval-{row['query'][:8]}")
            scored.append(judge.judge_one(row["query"], out.get("answer", ""), row["reference"]))
        except Exception as exc:
            failed.append({"query": row["query"], "error": str(exc)[:120]})
    result = judge.summarize(scored)
    if failed:
        result["failed"] = failed
    return result


def run_business() -> dict:
    """只跑意图路由：规则判断，不调模型，任何时候都能跑。"""
    from app.core import graph as g

    rows = load("business")
    correct, detail = 0, []
    for row in rows:
        state = {}
        for turn in row["turns"]:
            state.update(g.node_route({**state, "query": turn}))
        ok = state.get("intent") == row["expect_intent"]
        correct += ok
        detail.append({"turns": row["turns"], "expect": row["expect_intent"],
                       "got": state.get("intent"), "ok": ok})
    return {"total": len(rows), "accuracy": round(correct / len(rows), 4) if rows else 0.0,
            "detail": [d for d in detail if not d["ok"]]}


def run_regression() -> dict:
    """回归集：每次改动必跑，同时校验意图与答案关键要素。"""
    from app.core import graph

    rows = load("regression")
    failed = []
    for row in rows:
        out = graph.chat(row["query"], "eval", f"rg-{row['id']}")
        answer = out.get("answer", "") or ""
        ok_intent = out.get("intent") == row["expect_intent"]
        ok_text = not row.get("expect_contains") or row["expect_contains"] in answer
        if not (ok_intent and ok_text):
            failed.append({
                "id": row["id"], "query": row["query"],
                "expect_intent": row["expect_intent"], "got_intent": out.get("intent"),
                "missing": "" if ok_text else row["expect_contains"],
            })
    total = len(rows)
    return {"total": total, "passed": total - len(failed),
            "pass_rate": round((total - len(failed)) / total, 4) if total else 0.0,
            "failed": failed}


SUITES = {"retrieval": run_retrieval, "generation": run_generation,
          "business": run_business, "regression": run_regression}


def previous_report(out_dir: Path) -> dict:
    files = sorted(out_dir.glob("*.json"))
    if not files:
        return {}
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    names = list(SUITES) if args.suite == "all" else [args.suite]
    report, errors = {}, {}
    for name in names:
        try:
            report[name] = SUITES[name]()
        except Exception as exc:
            errors[name] = str(exc)[:200]

    out = Path(args.output) if args.output else config.BASE_DIR / "reports" / f"{date.today()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = previous_report(out.parent)
    out.write_text(json.dumps({"suites": report, "errors": errors},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "report": str(out),
        "results": report,
        "errors": errors,
        "diff_vs_last": {k: metrics.compare(v, prev.get("suites", {}).get(k))
                         for k, v in report.items() if k in ("retrieval",)},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
