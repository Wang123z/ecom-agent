"""评测层测试：指标计算、裁判解析、业务集路由。全部不依赖外部服务。"""
from eval import judge, metrics


def hit(doc, section=""):
    return {"doc": doc, "section": section, "text": "x"}


# ---------- 检索指标 ----------
def test_first_hit_rank_found():
    hits = [hit("其他"), hit("退款政策", "价保规则"), hit("退款政策", "退款到账时效")]
    assert metrics.first_hit_rank(hits, ["退款政策"], ["退款到账时效"]) == 3


def test_first_hit_rank_not_found():
    assert metrics.first_hit_rank([hit("其他")], ["退款政策"]) == 0


def test_first_hit_rank_matches_doc_only():
    assert metrics.first_hit_rank([hit("物流与发货", "发货时效")], ["物流与发货"]) == 1


def test_evaluate_basic():
    rows = [
        {"query": "q1", "gold_docs": ["A"]},
        {"query": "q2", "gold_docs": ["B"]},
    ]
    index = {"q1": [hit("A"), hit("X")], "q2": [hit("X"), hit("Y"), hit("B")]}
    result = metrics.evaluate(rows, lambda q: index[q], ks=(1, 3))
    assert result["hit@1"] == 0.5
    assert result["hit@3"] == 1.0
    assert result["mrr"] == round((1.0 + 1 / 3) / 2, 4)


def test_evaluate_empty():
    assert metrics.evaluate([], lambda q: [])["total"] == 0


def test_compare():
    current = {"hit@1": 0.8, "mrr": 0.7, "total": 10}
    previous = {"hit@1": 0.6, "mrr": 0.7, "total": 10}
    assert metrics.compare(current, previous) == {"hit@1": 0.2, "mrr": 0.0}


# ---------- 裁判 ----------
def test_judge_parse_plain():
    assert judge.parse_json('{"accuracy":1}')["accuracy"] == 1


def test_judge_parse_with_chatter():
    assert judge.parse_json('结果如下：{"closure":1} 完毕')["closure"] == 1


def test_judge_parse_invalid():
    assert judge.parse_json("无法评分") == {}


def test_judge_normalize_clamps_satisfaction():
    assert judge.normalize({"satisfaction": 9})["satisfaction"] == 5
    assert judge.normalize({"satisfaction": 0})["satisfaction"] == 1


def test_judge_normalize_missing_fields():
    out = judge.normalize({"accuracy": 1})
    assert out["accuracy"] == 1
    assert out["hallucination"] == 0


def test_judge_summarize():
    results = [
        {"accuracy": 1, "auto_resolve": 1, "closure": 1, "satisfaction": 5, "hallucination": 0},
        {"accuracy": 0, "auto_resolve": 0, "closure": 1, "satisfaction": 3, "hallucination": 1},
    ]
    summary = judge.summarize(results)
    assert summary["accuracy"] == 0.5
    assert summary["hallucination_rate"] == 0.5
    assert summary["satisfaction_avg"] == 4.0


# ---------- 业务集 ----------
def test_business_suite_all_pass():
    from eval.run_all import run_business

    result = run_business()
    assert result["total"] > 0
    assert result["accuracy"] == 1.0
    assert result["detail"] == []
