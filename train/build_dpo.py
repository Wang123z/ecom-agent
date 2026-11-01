"""DPO 偏好对构造：模型生成多候选 → 裁判打分 → 选 chosen / rejected。

用法
  python -m train.build_dpo --limit 50 --dry-run      # 只看前几条，不调模型
  python -m train.build_dpo --limit 3000

要点：裁判只打分排序，不生成答案；rejected 优先选「安全但没用」的候选，
而不是明显错误的——差太多的负样本模型早就知道该避开，学不到东西。
"""
import argparse
import json
import random
from pathlib import Path

from app import config

JUDGE_PROMPT = """你是电商客服质检员。给下面这条回复打分，只输出 JSON。

用户问题：{question}
客服回复：{answer}

评分字段：
- accuracy: 0 或 1，是否准确回答且没有编造订单状态或平台政策
- auto_resolve: 0 或 1，是否给出明确下一步动作或操作路径
- closure: 0 或 1，是否形成闭环，用户不需要再追问
- hallucination: 0 或 1，是否虚构了订单状态、物流节点、退款进度

只输出：{{"accuracy":0,"auto_resolve":0,"closure":0,"hallucination":0}}"""


def load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def parse_json(text: str) -> dict:
    """模型输出可能带代码块或前后解释，抓第一个 JSON 对象。"""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def generate_candidates(question: str, n: int) -> list:
    from app.infra import llm

    out = []
    for _ in range(n):
        try:
            reply = llm.chat([{"role": "user", "content": question}], temperature=0.9)
        except Exception as exc:
            print(f"[warn] 生成失败: {exc}")
            continue
        if reply and reply not in out:
            out.append(reply)
    return out


def judge(question: str, answer: str) -> dict:
    from app.infra import llm

    raw = llm.chat([{"role": "user", "content": JUDGE_PROMPT.format(question=question, answer=answer)}],
                   temperature=0.0, max_tokens=80)
    scores = parse_json(raw)
    return {k: int(scores.get(k, 0)) for k in ("accuracy", "auto_resolve", "closure", "hallucination")}


def rank_key(scores: dict) -> tuple:
    """排序用：先排除幻觉，再比三维总分。"""
    return (1 - scores["hallucination"], scores["accuracy"] + scores["auto_resolve"] + scores["closure"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.DATA_DIR / "sft_train.jsonl"))
    parser.add_argument("--output", default=str(config.DATA_DIR / "preference.jsonl"))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="不调模型，只校验数据")
    args = parser.parse_args()

    rows = [r for r in load_jsonl(Path(args.input)) if len(r["conversations"]) == 3]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[:args.limit]

    if args.dry_run:
        print(json.dumps({"single_turn_samples": len(rows),
                          "first_question": rows[0]["conversations"][1]["value"] if rows else ""},
                         ensure_ascii=False, indent=2))
        return

    pairs, skipped = [], 0
    for i, row in enumerate(rows, start=1):
        question = row["conversations"][1]["value"]
        candidates = generate_candidates(question, args.candidates)
        if len(candidates) < 2:
            skipped += 1
            continue

        scored = [(c, judge(question, c)) for c in candidates]
        scored.sort(key=lambda x: rank_key(x[1]), reverse=True)

        chosen, chosen_score = scored[0]
        if chosen_score["hallucination"]:
            skipped += 1
            continue
        # rejected 取分数最低的，但跳过同样有幻觉的（负样本要「平庸」不要「离谱」）
        rejected = next((c for c, s in reversed(scored) if not s["hallucination"]), None)
        if not rejected or rejected == chosen:
            skipped += 1
            continue

        pairs.append({
            "prompt": [{"role": "user", "content": question}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
        if i % 20 == 0:
            print(f"[{i}/{len(rows)}] 已生成 {len(pairs)} 对")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps({"pairs": len(pairs), "skipped": skipped, "output": str(out)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
