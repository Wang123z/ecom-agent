"""图状态：一次对话内流转的数据。"""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    # 输入
    query: str
    user_id: str
    session_id: str
    trace_id: str

    # 路由与参数
    intent: str          # chat / rag / order / refund / handoff
    order_id: str
    reason: str
    missing: str         # 缺哪个参数，用来生成追问

    # 检索与生成
    hits: list
    answer: str
    citations: list
    no_answer: bool

    # 工具结果
    action: dict
