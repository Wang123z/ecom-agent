# API

Base URL：`http://localhost:8080`　所有业务接口需 `Authorization: Bearer <token>`。

## POST /api/login

手机号 + 固定验证码换 JWT（演示用，不接短信服务）。

```json
{"phone": "13800000000", "code": "1234"}
```

```json
{"token": "eyJhbGciOi...", "user_id": "u1", "name": "张三"}
```

| 状态码 | 含义 |
|---|---|
| 200 | 成功 |
| 401 | 验证码错误 |
| 404 | 手机号未注册 |

## POST /api/chat

SSE 流式对话。`session_id` 相同的多轮共享上下文（LangGraph checkpointer）。

```json
{"query": "退货要几天到账", "session_id": "sess-001"}
```

返回 `text/event-stream`：

```
data: {"type":"token","text":"审核通过后1-3个工作日"}
data: {"type":"token","text":"退回原支付渠道。"}
data: {"type":"done","intent":"rag","citations":[{"doc":"退款政策","section":"退款到账时效","text":"..."}],"no_answer":false}
data: [DONE]
```

| type | 含义 |
|---|---|
| `token` | 正文片段，按 24 字切片 |
| `done` | 结束，带意图与引用来源 |
| `error` | 异常，`text` 为原因 |

## GET /api/orders/{order_id}

查订单。只返回当前 token 对应用户的订单。

```json
{"id":"O10001","status":"delivered","amount":199.0,"product":"轻量跑鞋 42 码","logistics":"已签收，杭州转运中心","created_at":"2026-11-01T11:32:40"}
```

| 状态码 | 含义 |
|---|---|
| 200 | 成功 |
| 401 | 未登录 |
| 404 | 订单不存在**或不属于当前用户** |

越权返回 404 而非 403，避免探测他人订单号。

## GET /api/tickets

当前用户的工单列表，倒序，最多 50 条。

```json
[{"id":"T3F8A21C","kind":"refund","reason":"尺码不合适","status":"open","order_id":"O10001"}]
```

## POST /api/feedback

提交满意度反馈，落库供后续回流做偏好数据。

```json
{"query": "退货要几天到账", "answer": "1-3 个工作日", "satisfied": true, "trace_id": "a1b2c3"}
```

```json
{"ok": true}
```

`trace_id` 取自响应头 `x-request-id`，用于把反馈关联到具体请求。

## GET /health

无需鉴权。

```json
{"status":"ok","llm":"qwen3-8b"}
```

---

## 模型服务（vLLM，端口 6006）

OpenAI 兼容接口，与业务服务分离部署。

```bash
curl http://localhost:6006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"cs-lora","messages":[{"role":"user","content":"吊牌剪了能退吗"}],"max_tokens":150,"temperature":0}'
```

`model` 取值：

| 值 | 含义 |
|---|---|
| `qwen3-8b` | 基座模型 |
| `cs-lora` | 挂载 SFT / DPO 后的 adapter |

## 错误格式

未捕获异常统一返回：

```json
{"code": 500, "msg": "服务异常，请稍后重试", "data": null}
```

每个响应带 `x-request-id` 头，用于日志追踪。
