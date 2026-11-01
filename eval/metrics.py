"""检索指标：Hit@K / Recall@K / MRR。

金标用「文档名 + 章节名」匹配，不依赖 chunk id——
重建索引后 chunk 会变，文档和章节相对稳定。
"""


def _is_gold(hit: dict, gold_docs: list, gold_sections: list) -> bool:
    if gold_sections:
        return hit.get("doc") in gold_docs and hit.get("section") in gold_sections
    return hit.get("doc") in gold_docs


def first_hit_rank(hits: list, gold_docs: list, gold_sections: list = None) -> int:
    """第一个命中金标的名次，从 1 开始；没命中返回 0。"""
    for rank, hit in enumerate(hits, start=1):
        if _is_gold(hit, gold_docs, gold_sections or []):
            return rank
    return 0


def evaluate(rows: list, retrieve_fn, ks=(1, 5, 10)) -> dict:
    """rows: [{"query","gold_docs","gold_sections"?}]，retrieve_fn(query) -> hits。"""
    result = {f"hit@{k}": 0.0 for k in ks}
    result.update({f"recall@{k}": 0.0 for k in ks})
    result["mrr"] = 0.0
    result["total"] = len(rows)

    if not rows:
        return result

    for row in rows:
        hits = retrieve_fn(row["query"]) or []
        rank = first_hit_rank(hits, row["gold_docs"], row.get("gold_sections"))
        if not rank:
            continue
        for k in ks:
            if rank <= k:
                result[f"hit@{k}"] += 1
                result[f"recall@{k}"] += 1
        result["mrr"] += 1.0 / rank

    for key in result:
        if key != "total":
            result[key] = round(result[key] / len(rows), 4)
    return result


def compare(current: dict, previous: dict = None) -> dict:
    """与上次结果做差，用来判断改动是变好还是变坏。"""
    if not previous:
        return {}
    return {k: round(current[k] - previous.get(k, 0.0), 4)
            for k in current if k != "total" and k in previous}
