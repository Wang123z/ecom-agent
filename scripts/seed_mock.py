"""灌入演示数据：2 个用户 + 4 个订单。可重复执行，已存在则跳过。

用法：python -m scripts.seed_mock
"""
from datetime import datetime, timedelta

from app.infra.db import Order, SessionLocal, User, init_db

USERS = [
    {"id": "u1", "phone": "13800000000", "name": "张三"},
    {"id": "u2", "phone": "13900000000", "name": "李四"},
]

ORDERS = [
    {"id": "O10001", "user_id": "u1", "status": "delivered", "amount": 199.0,
     "product": "轻量跑鞋 42 码", "logistics": "已签收，杭州转运中心 10-30 14:20"},
    {"id": "O10002", "user_id": "u1", "status": "paid", "amount": 59.0,
     "product": "纯棉短袜 3 双", "logistics": "", "days_ago": 1},
    {"id": "O10003", "user_id": "u1", "status": "delivered", "amount": 99.0,
     "product": "棒球帽", "logistics": "已签收，上海分拨中心", "days_ago": 12},
    {"id": "O20001", "user_id": "u2", "status": "delivered", "amount": 299.0,
     "product": "双肩背包", "logistics": "运输中，广州转运中心", "days_ago": 2},
]


def main() -> None:
    init_db()
    with SessionLocal() as s:
        for u in USERS:
            if not s.get(User, u["id"]):
                s.add(User(**u))

        for item in ORDERS:
            if s.get(Order, item["id"]):
                continue
            row = dict(item)
            row["created_at"] = datetime.now() - timedelta(days=row.pop("days_ago", 0))
            s.add(Order(**row))
        s.commit()

    print(f"用户 {len(USERS)} 个，订单 {len(ORDERS)} 个已就绪")
    print("登录示例：phone=13800000000, code=1234")


if __name__ == "__main__":
    main()
