"""FastAPI 入口。后续批次在这里挂路由与中间件。"""
from fastapi import FastAPI

from app import config

app = FastAPI(title="电商售后智能客服平台", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm": config.LLM_MODEL}
