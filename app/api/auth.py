"""登录：手机号 + 固定验证码换 JWT。演示用，不接短信服务。"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import guard
from app.infra.db import SessionLocal, User

router = APIRouter()
DEMO_CODE = "1234"


class LoginRequest(BaseModel):
    phone: str
    code: str


@router.post("/login")
def login(body: LoginRequest) -> dict:
    if body.code != DEMO_CODE:
        raise HTTPException(401, "验证码错误")
    with SessionLocal() as s:
        user = s.query(User).filter(User.phone == body.phone).first()
        if not user:
            raise HTTPException(404, "手机号未注册")
        return {"token": guard.create_token(user.id), "user_id": user.id, "name": user.name}
