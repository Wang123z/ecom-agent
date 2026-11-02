"""LangGraph 五分支编排。

route 用规则判断意图（不调模型，省一次往返、也让测试不依赖模型），
只有生成环节才调 LLM。缺参数时先追问，用户补齐后同一 session 内继续。
"""
import re

from langgraph.graph import END, StateGraph

from app import config
from app.core import tools
from app.core.rag import retriever
from app.core.state import AgentState
from app.infra import llm

_ORDER_RE = re.compile(r"(?:订单号?|单号)[:：\s]*([A-Za-z0-9]{4,})|([A-Z]\d{4,})")

# 问「能不能退」「几天到账」是政策咨询，走检索；
# 只有带动作词（我要/帮我/申请）才是要办事，走工具。
_HANDOFF_WORDS = ("投诉", "人工", "举报", "差评")
# 强查件词：出现就一定是在查件，不可能是在问政策
_ORDER_STRONG = ("到哪", "查件", "物流信息", "快递单号", "物流单号")
# 弱领域词：既可能在问政策，也可能在查件，必须配动作词才判 order
_ORDER_WEAK = ("物流", "快递", "签收", "配送", "发货", "订单")
_REFUND_WORDS = ("退款", "退货", "换货", "退钱", "售后", "价保", "退差价")
_ACTION_WORDS = ("我要", "帮我", "申请", "给我", "想退", "要退", "查一下", "查查")
_CHAT_WORDS = ("你好", "在吗", "谢谢", "你是谁", "早上好", "晚上好")

INTENT_PROMPT = """判断用户意图，只输出一个词，不要解释：
rag=问政策/流程/时效，order=查订单或物流进度，refund=要申请退款退货，handoff=投诉或转人工，chat=闲聊
用户：{query}
意图："""

SYSTEM_PROMPT = (
    "你是电商售后客服。只根据【资料】回答，资料里没有的不要编造，"
    "涉及订单状态、物流节点、退款进度这类实时信息时，明确说明需要查询订单。\n"
    "回答必须包含三部分：①结论（能否办理、时效）②适用条件 ③具体操作路径。"
    "路径用【】标注入口，例如「在【我的订单】→【申请退货】提交」；"
    "资料里没有明确路径的，说明联系客服协助。"
    "回答简洁，不超过 100 字。"
)


def extract_order_id(text: str) -> str:
    m = _ORDER_RE.search(text or "")
    if not m:
        return ""
    return (m.group(1) or m.group(2) or "").upper()


def _model_intent(query: str) -> str:
    """模型兜底：只输出一个词，调用失败退回 rag。"""
    try:
        out = llm.chat([{"role": "user", "content": INTENT_PROMPT.format(query=query)}],
                       temperature=0.0, max_tokens=8).strip().lower()
    except Exception:
        return "rag"
    for name in ("handoff", "refund", "order", "chat", "rag"):
        if name in out:
            return name
    return "rag"


def classify(query: str, allow_model: bool = True) -> str:
    """规则优先，模糊时模型兜底。

    强信号（订单号、投诉词、强查件词）直接定意图；
    只命中弱领域词时规则不可靠——「物流三天没更新怎么办」问的是政策，
    「我的快递到哪了」问的是查件，同一个「物流」两种意图，只能交给模型判断。

    退款/退货优先于订单：用户说「我要退款，订单 O10001」要走退款，不能被订单抢走。
    """
    if any(w in query for w in _HANDOFF_WORDS):
        return "handoff"
    # 退款优先于订单：「我要退款 O10001」是退款，不能被订单号抢走
    if any(w in query for w in _REFUND_WORDS) and any(w in query for w in _ACTION_WORDS):
        return "refund"
    if any(w in query for w in _ORDER_STRONG):
        return "order"
    if extract_order_id(query):
        return "order"
    if any(w in query for w in _ORDER_WEAK) and any(w in query for w in _ACTION_WORDS):
        return "order"
    if any(w in query for w in _CHAT_WORDS):
        return "chat"
    return _model_intent(query) if allow_model else "rag"


# ---------- 节点 ----------
def node_route(state: AgentState) -> dict:
    query = state["query"]
    order_id = extract_order_id(query)

    # 上一轮在等订单号，这一轮用户只回了订单号：沿用上一轮的意图
    if state.get("missing") and order_id:
        return {"intent": state.get("intent", "order"), "order_id": order_id, "missing": ""}

    intent = classify(query)
    if intent in ("order", "refund") and not order_id:
        return {"intent": intent, "order_id": "", "missing": "order_id"}
    return {"intent": intent, "order_id": order_id, "missing": ""}


def node_chat(state: AgentState) -> dict:
    answer = llm.chat([
        {"role": "system", "content": "你是电商售后客服助手，回答简短友好，1-2 句。"},
        {"role": "user", "content": state["query"]},
    ], temperature=0.3)
    return {"answer": answer, "citations": []}


def node_retrieve(state: AgentState) -> dict:
    hits = retriever.recall(state["query"], config.RECALL_TOP_K)
    return {"hits": hits[:config.FINAL_TOP_K]}


def node_generate(state: AgentState) -> dict:
    hits = state.get("hits") or []
    if not hits:
        return {"answer": "资料库里没有查到相关内容，可以换个问法，或者转人工帮你处理。",
                "citations": [], "no_answer": True}

    context = "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
    answer = llm.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【资料】\n{context}\n\n【问题】{state['query']}"},
    ], temperature=0.2)
    citations = [{"doc": h.get("doc", ""), "section": h.get("section", ""),
                  "text": h["text"][:80]} for h in hits]
    return {"answer": answer, "citations": citations}


def node_tool_order(state: AgentState) -> dict:
    if state.get("missing"):
        return {"answer": "请提供一下订单号，我帮你查。订单号可以在【我的订单】里看到。"}

    query = state["query"]
    order_id = state["order_id"]
    if any(w in query for w in ("物流", "快递", "到哪", "签收", "配送")):
        result = tools.get_logistics(state["user_id"], order_id)
    else:
        result = tools.get_order(state["user_id"], order_id)

    if result["ok"]:
        order = result.get("order") or {}
        text = result.get("msg") or (
            f"订单 {order['id']}：{order['product']}，状态 {order['status']}，"
            f"金额 {order['amount']} 元"
        )
    else:
        text = result["msg"]
    return {"action": result, "answer": text, "citations": []}


def node_tool_refund(state: AgentState) -> dict:
    if state.get("missing"):
        return {"answer": "申请退款需要订单号，请提供一下，我在【我的订单】里能查到。"}

    result = tools.submit_refund(state["user_id"], state["order_id"], state.get("reason", ""))
    return {"action": result, "answer": result["msg"], "citations": []}


def node_handoff(state: AgentState) -> dict:
    result = tools.create_ticket(state["user_id"], state.get("order_id", ""), state["query"])
    return {"action": result, "answer": result["msg"], "citations": []}


# ---------- 边 ----------
def after_route(state: AgentState) -> str:
    return state.get("intent", "rag")


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("route", node_route)
    g.add_node("chat", node_chat)
    g.add_node("retrieve", node_retrieve)
    g.add_node("generate", node_generate)
    g.add_node("tool_order", node_tool_order)
    g.add_node("tool_refund", node_tool_refund)
    g.add_node("handoff", node_handoff)

    g.set_entry_point("route")
    g.add_conditional_edges("route", after_route, {
        "chat": "chat", "rag": "retrieve", "order": "tool_order",
        "refund": "tool_refund", "handoff": "handoff",
    })
    g.add_edge("retrieve", "generate")
    for node in ("chat", "generate", "tool_order", "tool_refund", "handoff"):
        g.add_edge(node, END)
    return g.compile(checkpointer=checkpointer)


_graph = None


def get_graph():
    """带 MemorySaver 的编译图：同一 session_id 的多轮共享状态（slots 不会丢）。"""
    global _graph
    if _graph is None:
        from langgraph.checkpoint.memory import MemorySaver

        _graph = build_graph(MemorySaver())
    return _graph


def chat(query: str, user_id: str, session_id: str) -> dict:
    """跑一轮对话。thread_id 用 session_id，同一会话的多轮状态自动延续。"""
    return get_graph().invoke(
        {"query": query, "user_id": user_id, "session_id": session_id},
        config={"configurable": {"thread_id": session_id}},
    )
