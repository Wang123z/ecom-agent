"""RRF 融合测试：只看名次，不看分数，权重可调。"""
from app.core.rag.fusion import rrf, take_top


def hit(i):
    return {"id": i}


def test_single_route_keeps_order():
    out = rrf([[hit("a"), hit("b"), hit("c")]], k=60, weights=[1.0])
    assert [h["id"] for h in out] == ["a", "b", "c"]


def test_shared_hit_ranks_first():
    out = rrf([[hit("a"), hit("b")], [hit("b"), hit("c")]], k=60, weights=[1.0, 1.0])
    assert out[0]["id"] == "b"


def test_weight_decides_tie():
    # a 只出现在高权重路，c 只出现在低权重路，名次相同
    out = rrf([[hit("a")], [hit("c")]], k=60, weights=[2.0, 0.5])
    assert [h["id"] for h in out] == ["a", "c"]


def test_empty_lists():
    assert rrf([[], []], k=60, weights=[1.0, 1.0]) == []


def test_take_top():
    hits = [hit("a"), hit("b"), hit("c")]
    assert len(take_top(hits, 2)) == 2
    assert take_top(hits, 0) == hits
