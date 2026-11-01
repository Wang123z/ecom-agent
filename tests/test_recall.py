"""三路召回的接线测试：用 monkeypatch 替掉模型与 Milvus，只验证融合逻辑。"""
from app.core.rag import retriever
from app.infra import embed, milvus


def test_recall_puts_exact_hit_first(monkeypatch):
    monkeypatch.setattr(embed, "encode_query", lambda q: ([0.1, 0.2], {"1": 0.5}))
    monkeypatch.setattr(milvus, "search_dense", lambda v, k: [
        {"id": "d1", "score": 0.9, "text": "稠密命中", "doc": "政策", "section": "时效"}])
    monkeypatch.setattr(milvus, "search_sparse", lambda w, k: [
        {"id": "s1", "score": 0.7, "text": "稀疏命中", "doc": "政策", "section": "运费"}])

    index = {retriever.normalize("退款多久到账"): {
        "id": "q1", "answer": "1-3 个工作日", "doc": "FAQ", "category": "退款"}}
    hits = retriever.recall("退款多久到账", index=index)

    assert hits[0]["id"] == "faq_q1"
    assert hits[0]["text"] == "1-3 个工作日"
    assert {h["id"] for h in hits} == {"faq_q1", "d1", "s1"}


def test_recall_without_exact_hit_still_fuses(monkeypatch):
    monkeypatch.setattr(embed, "encode_query", lambda q: ([0.1], {"1": 0.5}))
    monkeypatch.setattr(milvus, "search_dense", lambda v, k: [
        {"id": "d1", "score": 0.9, "text": "A", "doc": "x", "section": "s"}])
    monkeypatch.setattr(milvus, "search_sparse", lambda w, k: [
        {"id": "d1", "score": 0.6, "text": "A", "doc": "x", "section": "s"}])

    hits = retriever.recall("发票怎么开", index={})
    assert len(hits) == 1
    assert hits[0]["id"] == "d1"


def test_recall_empty(monkeypatch):
    monkeypatch.setattr(embed, "encode_query", lambda q: ([0.1], {"1": 0.5}))
    monkeypatch.setattr(milvus, "search_dense", lambda v, k: [])
    monkeypatch.setattr(milvus, "search_sparse", lambda w, k: [])
    assert retriever.recall("随便问问", index={}) == []
