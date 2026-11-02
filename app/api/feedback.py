"""用户反馈：记录满意与否，供后续回流做偏好数据。"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app import guard
from app.infra.db import Feedback, SessionLocal

router = APIRouter()


class FeedbackBody(BaseModel):
    query: str = ""
    answer: str = ""
    satisfied: bool = False
    trace_id: str = ""


@router.post("/feedback")
def submit(body: FeedbackBody, user_id: str = Depends(guard.current_user)) -> dict:
    with SessionLocal() as s:
        s.add(Feedback(user_id=user_id, query=body.query, answer=body.answer,
                       satisfied=body.satisfied, trace_id=body.trace_id))
        s.commit()
    return {"ok": True}
