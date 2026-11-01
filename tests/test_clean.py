"""数据清洗的纯函数测试，不依赖任何外部服务。"""
from train.clean import clean_text, dedup_key, quality, similarity


def test_clean_text_strips_emoji_and_spaces():
    assert clean_text("  退货怎么办 😊  ") == "退货怎么办"


def test_clean_text_keeps_chinese_punctuation():
    """中文标点与【】是客服话术的正常写法，清洗不能转成半角。"""
    assert clean_text("可以，但是（需要）凭证") == "可以，但是（需要）凭证"
    assert clean_text("在【我的订单】里申请") == "在【我的订单】里申请"


def test_clean_text_normalizes_date():
    assert clean_text("2026年1月5日发货") == "2026-1-5发货"
    assert clean_text("2026/1/5发货") == "2026-1-5发货"


def test_dedup_key_ignores_punctuation_and_case():
    assert dedup_key("退货,怎么办?") == dedup_key("退货怎么办")


def test_similarity_identical_is_one():
    assert similarity("退款多久到账", "退款多久到账") == 1.0


def test_similarity_near_identical_is_high():
    assert similarity("退款多久到账", "退款多久能到账") > 0.5


def test_similarity_synonym_is_low():
    """已知局限：bigram 抓不到同义词，同义不同词只能靠向量聚类。"""
    assert similarity("退款多久能到账", "退款几天到账") < 0.3


def test_similarity_unrelated_is_low():
    assert similarity("退款多久到账", "发票怎么开") < 0.3


def test_quality_prefers_concrete_answer():
    concrete = {"question": "退款多久到账", "answer": "审核通过后1-3个工作日退回原支付渠道。"}
    vague = {"question": "退款多久到账", "answer": "很快的,请耐心等待就好啦。"}
    assert quality(concrete) > quality(vague)
