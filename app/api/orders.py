"""订单查询：数据权限在这里生效，只返回当前 token 对应用户的订单。"""
from fastapi import APIRouter, Depends, HTTPException

from app import guard
from app.core import tools

router = APIRouter()


@router.get("/orders/{order_id}")
def get_order(order_id: str, user_id: str = Depends(guard.current_user)) -> dict:
    result = tools.get_order(user_id, order_id)
    if not result["ok"]:
        raise HTTPException(404, result["msg"])
    return result["order"]
