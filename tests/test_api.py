"""接口测试：用内存 SQLite + 内存图，验证鉴权、越权拦截与 SSE 协议。"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import guard
from app.api import auth, tickets
from app.core import graph, tools
from app.infra import db


@pytest.fixture
def client(monkeypatch):
    # StaticPool 让所有连接共用同一个内存库，否则 TestClient 换线程会看到空库
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    db.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    for module in (auth, tickets, tools):
        monkeypatch.setattr(module, "SessionLocal", Session)
    monkeypatch.setattr(guard, "allow", lambda user_id: True)

    with Session() as s:
        s.add(db.User(id="u1", phone="13800000000", name="张三"))
        s.add(db.User(id="u2", phone="13900000000", name="李四"))
        s.add(db.Order(id="O10001", user_id="u1", status="delivered", amount=199.0,
                       product="跑鞋", logistics="已签收"))
        s.add(db.Order(id="O20001", user_id="u2", status="delivered", amount=299.0,
                       product="背包", logistics="运输中"))
        s.commit()

    from app.main import app

    monkeypatch.setattr("app.main.init_db", lambda: None)
    return TestClient(app)


def login(client, phone="13800000000", code="1234"):
    resp = client.post("/api/login", json={"phone": phone, "code": code})
    return resp


def auth_header(client, phone="13800000000"):
    token = login(client, phone).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_login_ok(client):
    resp = login(client)
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "u1"
    assert resp.json()["token"]


def test_login_wrong_code(client):
    assert login(client, code="0000").status_code == 401


def test_login_unknown_phone(client):
    assert login(client, phone="13700000000").status_code == 404


def test_order_requires_token(client):
    assert client.get("/api/orders/O10001").status_code == 401


def test_order_own(client):
    resp = client.get("/api/orders/O10001", headers=auth_header(client))
    assert resp.status_code == 200
    assert resp.json()["product"] == "跑鞋"


def test_order_of_other_user_blocked(client):
    """越权返回 404，不暴露「这个订单存在但不属于你」。"""
    resp = client.get("/api/orders/O20001", headers=auth_header(client))
    assert resp.status_code == 404


def test_tickets_only_own(client, monkeypatch):
    monkeypatch.setattr(tools, "submit_refund", lambda u, o, r: {
        "ok": True, "ticket_id": "T1", "msg": "ok"})
    tools.create_ticket("u1", "O10001", "投诉")
    tools.create_ticket("u2", "O20001", "投诉")

    resp = client.get("/api/tickets", headers=auth_header(client))
    assert resp.status_code == 200
    assert [t["order_id"] for t in resp.json()] == ["O10001"]


def test_chat_sse(client, monkeypatch):
    monkeypatch.setattr(graph, "chat", lambda q, u, s: {
        "answer": "退款 1-3 个工作日到账", "intent": "rag",
        "citations": [{"doc": "退款政策"}], "no_answer": False})

    resp = client.post("/api/chat", json={"query": "退款多久到账", "session_id": "s1"},
                       headers=auth_header(client))
    assert resp.status_code == 200
    body = resp.text
    assert '"type": "token"' in body
    assert '"type": "done"' in body
    assert body.strip().endswith("[DONE]")


def test_chat_requires_token(client):
    assert client.post("/api/chat", json={"query": "hi", "session_id": "s1"}).status_code == 401
