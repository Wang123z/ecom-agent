"""LLM 调用：OpenAI 兼容接口，指向 vLLM 或任意兼容服务。

换模型只改 .env 的 LLM_BASE_URL / LLM_MODEL，代码不动。
"""
from app import config

_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    return _client


def chat(messages: list, model: str = None, temperature: float = 0.3,
         max_tokens: int = None) -> str:
    resp = client().chat.completions.create(
        model=model or config.LLM_MODEL, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def stream(messages: list, model: str = None, temperature: float = 0.3):
    """逐 token 产出，供 SSE 使用。"""
    resp = client().chat.completions.create(
        model=model or config.LLM_MODEL, messages=messages,
        temperature=temperature, stream=True,
    )
    for chunk in resp:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
