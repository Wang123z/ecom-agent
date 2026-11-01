"""精排：bge-reranker-v2-m3 本地 CrossEncoder。"""
from app import config

_model = None


def model():
    global _model
    if _model is None:
        if not config.RERANK_MODEL:
            raise RuntimeError("RERANK_MODEL 未配置，请指向 bge-reranker-v2-m3 本地目录")
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(config.RERANK_MODEL)
    return _model


def score(query: str, docs: list) -> list:
    """返回每个文档与 query 的相关性分数，未归一化。"""
    if not docs:
        return []
    return [float(s) for s in model().predict([[query, d] for d in docs])]


def rerank(query: str, hits: list, top_k: int = None, text_key: str = "text") -> list:
    """给命中打上 rerank_score 并按分数降序，可选截断。"""
    if not hits:
        return []
    scores = score(query, [h.get(text_key, "") for h in hits])
    for hit, s in zip(hits, scores):
        hit["rerank_score"] = s
    hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return hits[:top_k] if top_k else hits
