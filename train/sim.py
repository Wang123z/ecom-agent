"""相似度计算：bigram 兜底，bge 语义。

bigram 只能合并字面相近的问题（同义不同词抓不到），正式跑数据用 bge。
本地没装 sentence-transformers 时用 bigram，云端用 bge。
"""
import math
import re
from collections import Counter

from app import config


def bigrams(text: str) -> set:
    s = re.sub(r"[\W_]+", "", text)
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def encode_all(texts: list, backend: str = "bigram") -> list:
    """一次性编码，避免两两比较时重复计算。"""
    if backend == "bigram":
        return [bigrams(t) for t in texts]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBEDDING_MODEL or "BAAI/bge-m3")
    return [v.tolist() for v in model.encode(texts, normalize_embeddings=True, show_progress_bar=False)]


def cos(a, b, backend: str = "bigram") -> float:
    if backend == "bigram":
        return len(a & b) / len(a | b) if a and b else 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def jaccard_text(a: str, b: str) -> float:
    """单对文本的 bigram 相似度，供单元测试与轻量场景使用。"""
    sa, sb = bigrams(a), bigrams(b)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0
