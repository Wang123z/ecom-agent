"""偏好数据构造的纯函数测试：JSON 容错解析与排序键。"""
from train.build_dpo import parse_json, rank_key


def test_parse_json_plain():
    assert parse_json('{"accuracy":1}') == {"accuracy": 1}


def test_parse_json_with_code_fence():
    text = '```json\n{"accuracy":1,"hallucination":0}\n```'
    assert parse_json(text)["accuracy"] == 1


def test_parse_json_with_chatter():
    text = '评分如下：{"accuracy":0,"closure":1} 以上。'
    assert parse_json(text)["closure"] == 1


def test_parse_json_invalid():
    assert parse_json("没有 JSON") == {}


def test_rank_key_prefers_no_hallucination():
    clean = {"accuracy": 1, "auto_resolve": 1, "closure": 1, "hallucination": 0}
    hallucinated = {"accuracy": 1, "auto_resolve": 1, "closure": 1, "hallucination": 1}
    assert rank_key(clean) > rank_key(hallucinated)


def test_rank_key_sums_three_dims():
    low = {"accuracy": 1, "auto_resolve": 0, "closure": 0, "hallucination": 0}
    high = {"accuracy": 1, "auto_resolve": 1, "closure": 1, "hallucination": 0}
    assert rank_key(high) > rank_key(low)
