"""工单列表：只返回当前用户的工单。"""
from fastapi import APIRouter, Depends

from app import guard
from app.infra.db import SessionLocal, Ticket

router = APIRouter()


@router.get("/tickets")
def list_tickets(user_id: str = Depends(guard.current_user)) -> list:
    with SessionLocal() as s:
        rows = (s.query(Ticket)
                .filter(Ticket.user_id == user_id)
                .order_by(Ticket.created_at.desc())
                .limit(50)
                .all())
        return [{"id": r.id, "kind": r.kind, "reason": r.reason,
                 "status": r.status, "order_id": r.order_id} for r in rows]
