"""三路召回：FAQ 精确匹配 + 稠密向量 + 稀疏向量，加权 RRF 融合。

第一路是精确匹配：客服场景里大量问题是 FAQ 原句或近似原句，
直接命中标准答案比向量检索更准也更快，还能压住长尾误召回。
"""
import json
import re
from pathlib import Path

from app import config
from app.core.rag.fusion import rrf
from app.infra import embed, milvus


def normalize(text: str) -> str:
    """精确匹配键：去掉标点空白并小写。"""
    return re.sub(r"[\W_]+", "", (text or "").lower())


def load_faq_index(path: Path = None) -> dict:
    path = Path(path) if path else (config.DATA_DIR / "faq_index.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def exact_route(query: str, index: dict) -> list:
    hit = index.get(normalize(query))
    if not hit:
        return []
    return [{"id": f"faq_{hit['id']}", "score": 1.0, "text": hit["answer"],
             "doc": hit.get("doc", "FAQ"), "section": hit.get("category", "")}]


def recall(query: str, top_k: int = None, index: dict = None) -> list:
    """三路并行召回后融合。index 传入可避免每次读盘。"""
    top_k = top_k or config.RECALL_TOP_K
    index = load_faq_index() if index is None else index

    exact_hits = exact_route(query, index)
    dense_vec, sparse_vec = embed.encode_query(query)
    dense_hits = milvus.search_dense(dense_vec, top_k)
    sparse_hits = milvus.search_sparse(sparse_vec, top_k)

    return rrf([exact_hits, dense_hits, sparse_hits])


def retrieve(query: str, top_k: int = None, index: dict = None) -> list:
    """召回 + 精排。rerank 未配置模型时会抛错，调用方按需选择是否走这一步。"""
    from app.infra.rerank import rerank

    hits = recall(query, top_k, index)
    return rerank(query, hits, config.FINAL_TOP_K)
