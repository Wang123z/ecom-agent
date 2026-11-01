"""四个业务工具：查订单、查物流、提交退款、建工单。

每个函数都要求 user_id 并校验订单归属——越权一律当「订单不存在」，
不区分「没这个单」和「不是你的单」，避免通过报错探测他人订单。
"""
import uuid
from datetime import datetime, timedelta

from app.infra.db import Order, SessionLocal, Ticket

REFUND_DAYS = 7
ACTIVE_STATUS = ("paid", "shipped", "delivered")


def _order_dict(user_id: str, order_id: str):
    """取订单并校验归属，返回普通 dict，避免会话关闭后访问 ORM 属性。"""
    with SessionLocal() as s:
        order = s.get(Order, order_id)
        if not order or order.user_id != user_id:
            return None
        return {
            "id": order.id, "status": order.status, "amount": order.amount,
            "product": order.product, "logistics": order.logistics,
            "created_at": order.created_at,
        }


def get_order(user_id: str, order_id: str) -> dict:
    order = _order_dict(user_id, order_id)
    if not order:
        return {"ok": False, "msg": "没有找到这个订单，请核对订单号"}
    return {"ok": True, "order": order}


def get_logistics(user_id: str, order_id: str) -> dict:
    order = _order_dict(user_id, order_id)
    if not order:
        return {"ok": False, "msg": "没有找到这个订单，请核对订单号"}
    if order["status"] == "paid":
        return {"ok": True, "msg": "订单还没发货，发货后可在【我的订单】查看物流"}
    return {"ok": True, "logistics": order["logistics"], "status": order["status"]}


def submit_refund(user_id: str, order_id: str, reason: str = "") -> dict:
    order = _order_dict(user_id, order_id)
    if not order:
        return {"ok": False, "msg": "没有找到这个订单，请核对订单号"}
    if order["status"] == "refunding":
        return {"ok": False, "msg": "这笔订单已经在退款流程中，无需重复提交"}
    if order["status"] not in ACTIVE_STATUS:
        return {"ok": False, "msg": f"当前订单状态为 {order['status']}，不支持申请退款"}
    if datetime.now() - order["created_at"] > timedelta(days=REFUND_DAYS):
        return {"ok": False, "msg": f"已超过签收后 {REFUND_DAYS} 天，无法自助申请退款，建议转人工处理"}

    ticket_id = f"T{uuid.uuid4().hex[:8].upper()}"
    with SessionLocal() as s:
        s.add(Ticket(id=ticket_id, user_id=user_id, order_id=order_id,
                     kind="refund", reason=reason or "用户申请退款"))
        s.get(Order, order_id).status = "refunding"
        s.commit()
    return {"ok": True, "ticket_id": ticket_id,
            "msg": f"退款申请已提交，工单号 {ticket_id}，1-3 个工作日原路退回"}


def create_ticket(user_id: str, order_id: str = "", reason: str = "") -> dict:
    ticket_id = f"T{uuid.uuid4().hex[:8].upper()}"
    with SessionLocal() as s:
        s.add(Ticket(id=ticket_id, user_id=user_id, order_id=order_id or None,
                     kind="handoff", reason=reason or "用户要求转人工"))
        s.commit()
    return {"ok": True, "ticket_id": ticket_id, "msg": f"已转人工，工单号 {ticket_id}，客服会在 24 小时内联系您"}
