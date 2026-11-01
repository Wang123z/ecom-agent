"""检索层纯函数测试：归一化、精确匹配、政策分块。不依赖 Milvus 与模型。"""
import json

from app.core.rag.retriever import exact_route, load_faq_index, normalize
from scripts.build_index import load_faq, split_paragraphs, split_sections


def test_normalize_strips_punctuation_and_case():
    assert normalize("退款，多久到账？") == normalize("退款多久到账")
    assert normalize("ABC-123") == "abc123"


def test_exact_route_hits():
    index = {normalize("退款多久到账"): {"id": "q1", "answer": "1-3 个工作日",
                                        "doc": "FAQ", "category": "退款"}}
    hits = exact_route("退款多久到账？", index)
    assert len(hits) == 1
    assert hits[0]["id"] == "faq_q1"
    assert hits[0]["text"] == "1-3 个工作日"


def test_exact_route_misses():
    assert exact_route("发票怎么开", {}) == []


def test_load_faq_index_from_file(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text(json.dumps({"id": "q1", "question": "退款多久到账", "answer": "1-3 天",
                             "category": "退款"}, ensure_ascii=False) + "\n", encoding="utf-8")
    index = load_faq(p)
    assert len(index) == 1
    assert index[normalize("退款多久到账")]["answer"] == "1-3 天"


def test_load_faq_index_missing_file(tmp_path):
    assert load_faq_index(tmp_path / "nope.json") == {}


def test_split_paragraphs_respects_limit():
    para = "这是一段用于测试分块逻辑的客服政策说明文字。" * 8   # 约 200 字
    chunks = split_paragraphs("\n\n".join([para] * 3))
    assert len(chunks) >= 2
    assert all(len(c) <= 400 + 50 for c in chunks)


def test_split_paragraphs_single_para():
    assert split_paragraphs("只有一段话。") == ["只有一段话。"]


def test_split_sections_uses_nearest_heading():
    """空正文的标题会被丢掉，只保留有内容的节。"""
    text = "# 退款\n## 时效\n三天到账。\n## 运费\n商家承担。"
    sections = split_sections(text)
    assert [s[0] for s in sections] == ["时效", "运费"]
    assert sections[0][1] == "三天到账。"
