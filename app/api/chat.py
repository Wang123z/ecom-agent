"""对话接口：SSE 流式返回。

生成本身是整段的（图里节点同步跑完），这里按 24 字切片推给前端，
保证前端逐字渲染的协议一致；后续把 generate 节点改成流式即可无缝替换。
"""
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import guard
from app.core import graph

router = APIRouter()
CHUNK = 24


class ChatRequest(BaseModel):
    query: str
    session_id: str


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat")
def chat(body: ChatRequest, user_id: str = Depends(guard.rate_limit)):
    def gen():
        try:
            out = graph.chat(body.query, user_id, body.session_id)
        except Exception as exc:
            yield _frame({"type": "error", "text": str(exc)})
            yield "data: [DONE]\n\n"
            return

        answer = out.get("answer", "")
        for i in range(0, len(answer), CHUNK):
            yield _frame({"type": "token", "text": answer[i:i + CHUNK]})
        yield _frame({
            "type": "done",
            "intent": out.get("intent", ""),
            "citations": out.get("citations") or [],
            "no_answer": bool(out.get("no_answer")),
        })
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
