"""压测 vLLM 的 OpenAI 兼容接口。

用法
  python -m deploy.bench --concurrency 1,5,10,20 --duration 60
  python -m deploy.bench --base-url http://localhost:6006/v1 --concurrency 10

指标：QPS、P50/P95/P99 总延迟、首 token 延迟、错误率。
压测要在模型服务同机跑，跨公网测出来的全是网络延迟。
"""
import argparse
import asyncio
import json
import statistics
import time

import httpx

PROMPT = "退货要几天到账？请给出具体时效。"


async def one_request(client: httpx.AsyncClient, url: str, model: str):
    """返回 (首token耗时, 总耗时)，失败返回 (None, None)。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.3,
        "max_tokens": 128,
        "stream": True,
    }
    start = time.perf_counter()
    first = None
    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                await resp.aread()
                return None, None
            async for line in resp.aiter_lines():
                if line.startswith("data:") and first is None:
                    first = time.perf_counter() - start
        return first, time.perf_counter() - start
    except Exception:
        return None, None


async def worker(client, url, model, deadline, sink):
    while time.perf_counter() < deadline:
        sink.append(await one_request(client, url, model))


def pct(values: list, q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * q), len(ordered) - 1)
    return round(ordered[idx] * 1000, 1)


async def run_level(url: str, model: str, concurrency: int, duration: int) -> dict:
    sink = []
    deadline = time.perf_counter() + duration
    async with httpx.AsyncClient(timeout=None) as client:
        start = time.perf_counter()
        await asyncio.gather(*[worker(client, url, model, deadline, sink) for _ in range(concurrency)])
        elapsed = time.perf_counter() - start

    ok = [t for f, t in sink if t is not None]
    ttft = [f for f, t in sink if f is not None]
    return {
        "concurrency": concurrency,
        "requests": len(sink),
        "errors": len(sink) - len(ok),
        "qps": round(len(ok) / elapsed, 2) if elapsed else 0.0,
        "ttft_ms": round(statistics.mean(ttft) * 1000, 1) if ttft else 0.0,
        "p50_ms": pct(ok, 0.50),
        "p95_ms": pct(ok, 0.95),
        "p99_ms": pct(ok, 0.99),
    }


async def main_async(args) -> None:
    url = f"{args.base_url.rstrip('/')}/chat/completions"
    levels = [int(x) for x in args.concurrency.split(",")]
    results = []
    for level in levels:
        print(f">>> 并发 {level}，持续 {args.duration}s ...", flush=True)
        results.append(await run_level(url, args.model, level, args.duration))
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--concurrency", default="1,5,10,20")
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
