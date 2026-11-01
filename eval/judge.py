"""LLM 裁判：四维打分。

rubric 里明确写了「允许什么、禁止什么」，比只给「1-5 分」稳定得多。
裁判只输出 JSON，解析失败按 0 分处理并记入 failed 列表。
"""
import json

from app.infra import llm

JUDGE_PROMPT = """你是电商客服质检评审员。评估下面这条客服回复是否适合真实电商场景。

【用户问题】{query}
【参考答案】{reference}
【客服回复】{answer}

评分字段（只输出 JSON，不要解释）：
- accuracy：0 或 1，是否准确回答，无答非所问、无虚构订单状态、无明显错误政策
- auto_resolve：0 或 1，是否给出明确下一步动作或操作路径
- closure：0 或 1，是否形成闭环，用户不需要再追问
- satisfaction：1-5 整数，5=准确自然可执行，1=答非所问或严重虚构
- hallucination：0 或 1，是否虚构了订单状态、物流节点、退款进度

允许：合理的行业常见推断、常见物流时效表达、常见平台操作建议
禁止：虚构订单状态、虚构系统已操作完成、违背上下文

输出示例：{{"accuracy":1,"auto_resolve":1,"closure":1,"satisfaction":5,"hallucination":0}}"""

FIELDS = ("accuracy", "auto_resolve", "closure", "hallucination")


def parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def normalize(raw: dict) -> dict:
    out = {k: int(raw.get(k, 0)) for k in FIELDS}
    try:
        out["satisfaction"] = max(1, min(5, int(raw.get("satisfaction", 1))))
    except (TypeError, ValueError):
        out["satisfaction"] = 1
    return out


def judge_one(query: str, answer: str, reference: str = "") -> dict:
    prompt = JUDGE_PROMPT.format(query=query, answer=answer, reference=reference or "无")
    raw = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=120)
    return normalize(parse_json(raw))


def summarize(results: list) -> dict:
    """把逐条打分汇总成比率。"""
    n = len(results)
    if not n:
        return {"total": 0}
    return {
        "total": n,
        "accuracy": round(sum(r["accuracy"] for r in results) / n, 4),
        "auto_resolve": round(sum(r["auto_resolve"] for r in results) / n, 4),
        "closure": round(sum(r["closure"] for r in results) / n, 4),
        "hallucination_rate": round(sum(r["hallucination"] for r in results) / n, 4),
        "satisfaction_avg": round(sum(r["satisfaction"] for r in results) / n, 2),
    }
