"""向量编码：bge-m3 本地模型，同时产出稠密与稀疏向量。

不做 API 兜底：本项目是私有化部署，向量化也应在本地完成。
EMBEDDING_MODEL 未配置时直接报错，避免悄悄降级成别的语义空间。
"""
from app import config

_model = None


def model():
    global _model
    if _model is None:
        if not config.EMBEDDING_MODEL:
            raise RuntimeError("EMBEDDING_MODEL 未配置，请指向 bge-m3 本地目录")
        from FlagEmbedding import BGEM3FlagModel

        _model = BGEM3FlagModel(config.EMBEDDING_MODEL, use_fp16=True)
    return _model


def encode(texts: list, with_sparse: bool = False):
    """返回 (dense, sparse)。dense 是 list[list[float]]；未请求稀疏时 sparse 为 None。"""
    out = model().encode(
        list(texts), return_dense=True, return_sparse=with_sparse, return_colbert_vecs=False
    )
    dense = [v.tolist() for v in out["dense_vecs"]]
    if not with_sparse:
        return dense, None
    return dense, [dict(w) for w in out["lexical_weights"]]


def encode_query(text: str):
    """单条查询，同时返回稠密与稀疏，供两路检索复用一次编码。"""
    dense, sparse = encode([text], with_sparse=True)
    return dense[0], sparse[0]
