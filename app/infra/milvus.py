"""Milvus 集合管理：稠密 + 稀疏双向量字段。

VARCHAR 上限按字节算（中文 3 字节/字），写入前一律按字节截断。
"""
from pymilvus import DataType, MilvusClient

from app import config

TEXT_MAX_BYTES = 8192
NAME_MAX_BYTES = 512
OUTPUT_FIELDS = ["text", "doc", "section"]

_client = None


def client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=config.MILVUS_URI)
    return _client


def fit_bytes(value: str, limit: int, suffix: str = "…") -> str:
    """按 UTF-8 字节截断，保证不超出 Milvus 的 VARCHAR 上限。"""
    raw = (value or "").encode("utf-8")
    if len(raw) <= limit:
        return value or ""
    keep = limit - len(suffix.encode("utf-8"))
    return raw[:max(keep, 0)].decode("utf-8", "ignore") + suffix


def build_schema(dim: int):
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=TEXT_MAX_BYTES)
    schema.add_field("doc", DataType.VARCHAR, max_length=NAME_MAX_BYTES)
    schema.add_field("section", DataType.VARCHAR, max_length=NAME_MAX_BYTES)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    return schema


def build_index_params():
    params = MilvusClient.prepare_index_params()
    params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
    params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    return params


def ensure_collection(dim: int = None) -> None:
    dim = dim or config.EMBEDDING_DIM
    c = client()
    if c.has_collection(config.MILVUS_COLLECTION):
        return
    c.create_collection(
        config.MILVUS_COLLECTION, schema=build_schema(dim), index_params=build_index_params()
    )


def drop_collection() -> None:
    c = client()
    if c.has_collection(config.MILVUS_COLLECTION):
        c.drop_collection(config.MILVUS_COLLECTION)


def insert(rows: list) -> int:
    if not rows:
        return 0
    client().insert(config.MILVUS_COLLECTION, rows)
    return len(rows)


def search_dense(vector: list, top_k: int, expr: str = "") -> list:
    res = client().search(
        config.MILVUS_COLLECTION, data=[vector], anns_field="dense",
        limit=top_k, filter=expr or "", output_fields=OUTPUT_FIELDS,
    )
    return _flatten(res)


def search_sparse(weights: dict, top_k: int, expr: str = "") -> list:
    res = client().search(
        config.MILVUS_COLLECTION, data=[weights], anns_field="sparse",
        limit=top_k, filter=expr or "", output_fields=OUTPUT_FIELDS,
    )
    return _flatten(res)


def _flatten(res) -> list:
    """把 Milvus 的 [[{id, distance, entity}]] 拍平成统一结构。"""
    hits = []
    for item in (res[0] if res else []):
        entity = item.get("entity") or {}
        hits.append({
            "id": str(item.get("id")),
            "score": float(item.get("distance", 0.0)),
            "text": entity.get("text", ""),
            "doc": entity.get("doc", ""),
            "section": entity.get("section", ""),
        })
    return hits
