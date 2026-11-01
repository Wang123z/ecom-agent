"""编排测试：意图路由与五分支，模型与检索全部替掉，不依赖外部服务。"""
import pytest

from app.core import graph as g
from app.core.rag import retriever
from app.infra import llm


@pytest.fixture
def no_model(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda messages, **kw: "好的，已为您处理。")
    return None


def run(query, user_id="u1", **extra):
    return g.build_graph().invoke({"query": query, "user_id": user_id, **extra})


def test_extract_order_id_variants():
    assert g.extract_order_id("我的订单 O10001 到哪了") == "O10001"
    assert g.extract_order_id("订单号：O10002") == "O10002"
    assert g.extract_order_id("帮我查查") == ""


@pytest.mark.parametrize("query,expected", [
    ("我要投诉你们", "handoff"),
    ("我要转人工", "handoff"),
    ("我要退款", "refund"),
    ("申请退货", "refund"),
    ("这个能退款吗", "rag"),
    ("退货要几天到账", "rag"),
    ("我的快递到哪了", "order"),
    ("你好", "chat"),
])
def test_classify(query, expected):
    assert g.classify(query) == expected


def test_route_asks_for_order_id():
    state = g.node_route({"query": "我要退款"})
    assert state["intent"] == "refund"
    assert state["missing"] == "order_id"


def test_route_keeps_intent_when_order_id_follows():
    """上一轮在等订单号，用户这一轮只回了订单号，意图要沿用。"""
    state = g.node_route({"query": "O10001", "intent": "refund", "missing": "order_id"})
    assert state["intent"] == "refund"
    assert state["order_id"] == "O10001"
    assert state["missing"] == ""


def test_chat_branch(no_model):
    out = run("你好")
    assert out["intent"] == "chat"
    assert out["answer"] == "好的，已为您处理。"


def test_rag_branch(no_model, monkeypatch):
    monkeypatch.setattr(retriever, "recall", lambda q, k: [
        {"id": "p1", "text": "签收后 7 天内可申请退货", "doc": "退款政策", "section": "七天无理由退货"}])
    out = run("退货要几天到账")
    assert out["intent"] == "rag"
    assert out["citations"][0]["doc"] == "退款政策"


def test_rag_branch_without_hits(no_model, monkeypatch):
    monkeypatch.setattr(retriever, "recall", lambda q, k: [])
    out = run("完全不相关的问题")
    assert out["no_answer"] is True


def test_order_branch(no_model, monkeypatch):
    monkeypatch.setattr(g.tools, "get_order", lambda u, o: {
        "ok": True, "order": {"id": o, "product": "跑鞋", "status": "delivered", "amount": 199.0}})
    out = run("查一下我的订单 O10001")
    assert out["intent"] == "order"
    assert "跑鞋" in out["answer"]


def test_order_branch_missing_order_id(no_model):
    out = run("我的快递到哪了")
    assert out["intent"] == "order"
    assert "订单号" in out["answer"]


def test_refund_branch(no_model, monkeypatch):
    monkeypatch.setattr(g.tools, "submit_refund", lambda u, o, r: {
        "ok": True, "ticket_id": "T123", "msg": "退款申请已提交，工单号 T123"})
    out = run("我要退款 O10001")
    assert out["intent"] == "refund"
    assert "T123" in out["answer"]


def test_handoff_branch(no_model, monkeypatch):
    monkeypatch.setattr(g.tools, "create_ticket", lambda u, o, r: {
        "ok": True, "ticket_id": "T999", "msg": "已转人工，工单号 T999"})
    out = run("我要投诉")
    assert out["intent"] == "handoff"
    assert "T999" in out["answer"]


def test_multi_turn_keeps_intent_with_checkpointer(no_model, monkeypatch):
    """多轮补参：第一轮缺订单号，第二轮只回订单号，仍走退款分支。"""
    from langgraph.checkpoint.memory import MemorySaver

    monkeypatch.setattr(g.tools, "submit_refund", lambda u, o, r: {
        "ok": True, "ticket_id": "T777", "msg": f"退款申请已提交，订单 {o}"})
    graph = g.build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "sess-1"}}

    first = graph.invoke({"query": "我要退款", "user_id": "u1", "session_id": "sess-1"}, config)
    assert first["missing"] == "order_id"
    assert "订单号" in first["answer"]

    second = graph.invoke({"query": "O10001", "user_id": "u1", "session_id": "sess-1"}, config)
    assert second["intent"] == "refund"
    assert "O10001" in second["answer"]
