"""业务工具测试：用内存 SQLite 替掉 MySQL，重点验证越权与状态判断。"""
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core import tools
from app.infra import db


@pytest.fixture
def mem_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    db.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(tools, "SessionLocal", Session)

    with Session() as s:
        s.add(db.User(id="u1", phone="13800000000", name="张三"))
        s.add(db.User(id="u2", phone="13900000000", name="李四"))
        s.add(db.Order(id="O10001", user_id="u1", status="delivered", amount=199.0,
                       product="跑鞋", logistics="已签收", created_at=datetime.now()))
        s.add(db.Order(id="O10002", user_id="u1", status="paid", amount=59.0,
                       product="袜子", logistics="", created_at=datetime.now()))
        s.add(db.Order(id="O10003", user_id="u1", status="delivered", amount=99.0,
                       product="帽子", logistics="已签收",
                       created_at=datetime.now() - timedelta(days=10)))
        s.add(db.Order(id="O20001", user_id="u2", status="delivered", amount=299.0,
                       product="背包", logistics="运输中", created_at=datetime.now()))
        s.commit()
    return Session


def test_get_order_owner_ok(mem_db):
    result = tools.get_order("u1", "O10001")
    assert result["ok"] is True
    assert result["order"]["product"] == "跑鞋"


def test_get_order_other_user_is_invisible(mem_db):
    """越权不报「无权」，统一当「不存在」，避免探测他人订单号。"""
    result = tools.get_order("u1", "O20001")
    assert result["ok"] is False
    assert "没有找到" in result["msg"]


def test_logistics_not_shipped(mem_db):
    result = tools.get_logistics("u1", "O10002")
    assert result["ok"] is True
    assert "还没发货" in result["msg"]


def test_logistics_shipped(mem_db):
    result = tools.get_logistics("u1", "O10001")
    assert result["ok"] is True
    assert result["logistics"] == "已签收"


def test_refund_success(mem_db):
    result = tools.submit_refund("u1", "O10001", "尺码不合适")
    assert result["ok"] is True
    assert result["ticket_id"].startswith("T")
    assert tools.get_order("u1", "O10001")["order"]["status"] == "refunding"


def test_refund_twice_rejected(mem_db):
    tools.submit_refund("u1", "O10001")
    second = tools.submit_refund("u1", "O10001")
    assert second["ok"] is False
    assert "重复" in second["msg"]


def test_refund_over_deadline(mem_db):
    result = tools.submit_refund("u1", "O10003")
    assert result["ok"] is False
    assert "超过" in result["msg"]


def test_refund_other_user_rejected(mem_db):
    assert tools.submit_refund("u1", "O20001")["ok"] is False


def test_create_ticket(mem_db):
    result = tools.create_ticket("u1", "O10001", "要投诉")
    assert result["ok"] is True
    assert "24 小时" in result["msg"]
