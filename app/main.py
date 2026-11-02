"""FastAPI 入口：中间件 + 路由 + 建表。"""
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app import config
from app.api import auth, chat, feedback, orders, tickets
from app.infra.db import init_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        init_db()
    except Exception as exc:
        logger.warning("建表失败，业务接口可能不可用: {}", exc)
    yield


app = FastAPI(title="电商售后智能客服平台", version="0.1.0", lifespan=lifespan)


class RequestContext:
    """纯 ASGI 中间件：加 request_id 与耗时日志。

    不用 BaseHTTPMiddleware——它会给请求套一层额外 task，
    导致 SSE 里 request.is_disconnected() 失效。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        rid = uuid.uuid4().hex[:12]
        start = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"x-request-id", rid.encode()))
            await send(message)

        await self.app(scope, receive, send_wrapper)
        logger.info("{} {} {} {}ms", rid, scope["method"], scope["path"],
                    int((time.perf_counter() - start) * 1000))


app.add_middleware(RequestContext)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("未捕获异常 {} {}", request.method, request.url.path)
    return JSONResponse({"code": 500, "msg": "服务异常，请稍后重试", "data": None}, status_code=500)


app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(orders.router, prefix="/api", tags=["orders"])
app.include_router(tickets.router, prefix="/api", tags=["tickets"])
app.include_router(feedback.router, prefix="/api", tags=["feedback"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm": config.LLM_MODEL}
