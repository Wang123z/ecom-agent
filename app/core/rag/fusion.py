"""加权 RRF：只用名次，规避余弦与稀疏内积的量纲差异。

score(d) = Σ w_i / (k + rank_i(d))
"""
from app import config


def rrf(rank_lists: list, k: int = None, weights: list = None) -> list:
    """rank_lists 是若干已按相关度降序的命中列表，每项需有 id 字段。"""
    k = k or config.RRF_K
    weights = weights or config.RRF_WEIGHTS

    scores, items = {}, {}
    for i, hits in enumerate(rank_lists):
        weight = weights[i] if i < len(weights) else 1.0
        for rank, hit in enumerate(hits, start=1):
            key = hit["id"]
            items[key] = hit
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
    return sorted(items.values(), key=lambda h: scores[h["id"]], reverse=True)


def take_top(hits: list, n: int) -> list:
    return hits[:n] if n and n > 0 else hits
