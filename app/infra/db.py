"""业务库：用户、订单、工单、反馈。SQLAlchemy 同步会话。

订单查询必须带 user_id 过滤——这是 JWT 数据权限的落点，
不靠调用方自觉，写在仓储函数里。
"""
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app import config

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(String(32), primary_key=True)
    phone = Column(String(20), nullable=False, unique=True)
    name = Column(String(64), nullable=False, default="")


class Order(Base):
    __tablename__ = "orders"

    id = Column(String(32), primary_key=True)
    user_id = Column(String(32), nullable=False, index=True)
    status = Column(String(16), nullable=False)          # paid / shipped / delivered / refunding
    amount = Column(Float, nullable=False, default=0.0)
    product = Column(String(128), nullable=False, default="")
    logistics = Column(String(255), nullable=False, default="")
    created_at = Column(DateTime, default=datetime.now)


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(String(32), primary_key=True)
    user_id = Column(String(32), nullable=False, index=True)
    order_id = Column(String(32), nullable=True)
    kind = Column(String(16), nullable=False)            # refund / handoff
    reason = Column(String(255), nullable=False, default="")
    status = Column(String(16), nullable=False, default="open")
    created_at = Column(DateTime, default=datetime.now)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(32), nullable=True)
    query = Column(Text, nullable=False, default="")
    answer = Column(Text, nullable=False, default="")
    satisfied = Column(Boolean, nullable=False, default=False)
    trace_id = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=datetime.now)


engine = create_engine(config.MYSQL_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
