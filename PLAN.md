# 电商售后智能客服平台 · 实施规划

> 本文档只讲一件事：这个项目怎么做完。每一步都有验收标准。

---

## 一、项目定位与范围

**做什么**：Qwen3-8B 经 SFT + DPO 微调、vLLM 私有化部署，结合售后政策 RAG 与订单/物流/退款工具调用，完成「问答 → 查单 → 售后 → 转人工 → 回流」闭环。

**技术目标**：

| 目标 | 量化标准 |
|---|---|
| 领域微调有效 | 业务集闭环率高于 base 模型 15 个百分点以上 |
| 部署可用 | 10 并发下 P99 ≤ 3s，显存占用可控 |
| 检索准确 | 检索集 Hit@1 ≥ 0.85 |
| 编排正确 | 业务集分支准确率 ≥ 90% |
| 数据权限 | 越权访问 100% 被拒 |

**明确不做**：Spring Boot 网关、K8s、注册与第三方登录、RBAC、多 Agent 协作、语音与图片输入、管理后台。

---

## 二、总体架构

```
React 前端（买家对话 + 工单列表）
      │  HTTP / SSE，Bearer JWT
      ▼
FastAPI 应用层
  ├─ 中间件     request_id / 审计日志 / 统一错误格式
  ├─ 鉴权       JWT 校验 → user_id（数据权限的基础）
  ├─ 限流       Redis 令牌桶：用户级 20/分，全局 10 QPS
  ├─ 编排       LangGraph 五分支
  ├─ 检索       三路召回（FAQ 精确 + bge-m3 稠密 + 稀疏）→ 加权 RRF → bge-reranker
  └─ 工具       查订单 / 查物流 / 提交退款 / 建工单（均校验归属）
      │
      ▼
vLLM(OpenAI 兼容) · Milvus · MySQL · Redis

离线链路（独立进程，产物是 adapter）：
数据治理 → SFT → DPO → Judge 评测 → 消融 → 上线
```

**一次请求的路径**：

```
提问
 → JWT 验签
 → 令牌桶
 → 意图路由
     ├─ 闲聊      直接回答
     ├─ 政策问答  召回 → RRF → 精排 → 生成（带引用）
     ├─ 查订单    缺参追问 → 工具 → 校验归属
     ├─ 售后申请  时效判断 → 工具 → 工单号
     └─ 转人工    置信度低 / 投诉情绪 → 建工单
 → SSE 流式返回 + 落库
```

---

## 三、目录结构

```
ecom-agent/
├── app/
│   ├── main.py              入口 + 中间件
│   ├── config.py            环境变量集中管理
│   ├── api/
│   │   ├── auth.py          登录
│   │   ├── chat.py          SSE 对话
│   │   ├── orders.py        订单查询（含权限校验）
│   │   └── tickets.py       工单
│   ├── core/
│   │   ├── graph.py         LangGraph 五分支
│   │   ├── state.py         状态定义
│   │   ├── rag/
│   │   │   ├── retriever.py 三路召回
│   │   │   ├── fusion.py    加权 RRF
│   │   │   ├── reranker.py  精排 + 阈值门控
│   │   │   └── prompts.py   提示词（版本化）
│   │   ├── tools/           四个工具
│   │   └── memory.py        多轮补参 + 会话窗口
│   ├── infra/               vllm / milvus / mysql / redis
│   └── guard.py             JWT + 令牌桶
├── train/
│   ├── configs/             sft.yaml / dpo.yaml
│   ├── clean.py             清洗去重
│   ├── cluster.py           问题聚类
│   ├── build_sft.py         FAQ → 多轮 ShareGPT
│   ├── build_dpo.py         多候选 → 打分 → 偏好对
│   ├── train_sft.py         LoRA
│   ├── train_dpo.py         DPO
│   └── infer.py             批量推理采样
├── eval/
│   ├── datasets/            retrieval / generation / business / regression
│   ├── judge.py             rubric + 打分
│   ├── metrics.py           检索与生成指标
│   └── run_all.py           一条命令跑全层
├── deploy/
│   ├── docker-compose.yml
│   ├── vllm.Dockerfile
│   └── bench.py             压测
├── data/
│   ├── faq_raw.xlsx         原始 FAQ
│   ├── policy/              售后政策 md
│   └── mock/                订单与物流 mock 数据
└── frontend/
```

---

## 四、模块拆解与验收标准

| 模块 | 做什么 | 做完的标志 |
|---|---|---|
| 数据治理 | 清洗、去重、聚类、质量排序 | 500–1000 条 QA，清洗前后对比表，聚类分布图 |
| SFT | LoRA 微调 | loss 曲线收敛，`run_summary.json` 有显存/参数量，推理样例可读 |
| DPO | 偏好对齐 | rewards/accuracies 上升，chosen-rejected margin 扩大 |
| 部署 | vLLM + 量化 | `/v1/chat/completions` 可调，`/health` 可用，4 档压测有数 |
| RAG | 三路召回 + 融合 + 精排 | 检索集 Hit@1 ≥ 0.85，回答带引用 |
| 编排 | LangGraph 五分支 | 业务集分支准确率 ≥ 90% |
| 工具 | 四个业务工具 | 查订单/物流/退款/工单全部可调，参数校验完整 |
| 鉴权 | JWT + 数据权限 | 越权用例被 403 |
| 限流 | 令牌桶 | 压测时超限返回 429 |
| 评测 | 四数据集 + 五层指标 | 一条命令产出 JSON 报告 + 与上次 diff |
| 部署编排 | Docker Compose | `docker compose up` 一键起全栈 |

---

## 五、数据方案

### 5.1 三类数据

| 类型 | 内容 | 来源 | 规模 |
|---|---|---|---|
| FAQ 问答对 | 问题 + 标准答案 | 公开电商 FAQ + 规则改写 | 500–1000 |
| 政策文档 | 退换货、物流、发票、售后时效 | 按真实平台公开规则整理，注明来源 | 200–300 段 |
| Mock 业务数据 | 订单、物流节点、工单 | 脚本生成（含用户归属） | 50 用户 / 200 订单 |

### 5.2 清洗流程

```
原始 FAQ
 → 文本标准化（时间格式 4 种正则、数字、emoji 剥离）
 → 精确去重（MD5）
 → 相似去重（编辑距离 > 0.9 视为重复）
 → 低价值过滤（问题 < 6 字、答案 < 15 字）
 → 问题聚类（bge-small 向量 + UnionFind，相似 > 0.85 合并）
 → 答案冲突治理（同一簇内答案矛盾 → 保留信息量最大的）
 → 质量打分排序
 → 输出 train / val 8:2
```

### 5.3 SFT 数据格式

```json
{"id": "s_001", "session_id": "sess_01",
 "conversations": [
   {"from": "user", "value": "我买的鞋子尺码不对，能换吗？"},
   {"from": "assistant", "value": "可以换。签收后 7 天内..."}
 ]}
```

多轮样本占比目标 30%。客服真实场景以多轮为主，纯单轮训练会学不会承接。

### 5.4 DPO 偏好对构造

```
1. 从 SFT train.jsonl 抽 3000 条
2. 用 SFT 模型对每条生成 4 个候选（temperature 0.8，保证多样性）
3. qwen-plus 只打分排序，不生成答案
4. 程序按分数选 chosen（排除 hallucination=1 的）
5. rejected 优先选「基本正确但不够具体」，不是「明显错误」
6. 输出 {"prompt": [...], "chosen": [...], "rejected": [...]}
```

负样本要选「安全但没用」的，明显错误的样本模型早就知道该避免，学不到新东西。标准 pair 示例：

- chosen：`您可以在【我的订单→申请退款】提交，未发货一般原路退款。`
- rejected：`请联系客服处理。`

---

## 六、训练方案

### 6.1 配置（YAML 驱动）

```yaml
# train/configs/sft.yaml
model:
  model_name_or_path: Qwen/Qwen3-8B
  max_seq_length: 2048
  eos_token: <|im_end|>
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  num_train_epochs: 3
  learning_rate: 2.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  assistant_only_loss: true
  gradient_checkpointing: true
logging:
  logging_steps: 5
  save_strategy: epoch
  eval_strategy: epoch
output:
  output_dir: outputs/sft
  artifact_dir: outputs/sft/artifacts
```

### 6.2 显存预算

| 任务 | 配置 | 显存 | 卡 |
|---|---|---|---|
| SFT 8B LoRA bf16 | bs=1, ga=8, max_len=2048 | ~22 GB | 4090 24G |
| SFT 8B QLoRA 4bit | 同上 | ~11 GB | 4080 16G |
| DPO 8B LoRA | 双 adapter（policy + ref） | ~26 GB | A100 40G 或 4090 + offload |
| vLLM 8B FP16 | max_model_len=4096 | ~18 GB | 4090 |
| vLLM 8B AWQ | 同上 | ~8 GB | 4080 |

显存不够就降级到 Qwen3-4B，流程不变，指标重测。

### 6.3 训练产物

```
outputs/sft/
├── adapter_model.safetensors
├── artifacts/
│   ├── loss_curve.png            训练/验证损失
│   ├── inference_samples.json    reference vs prediction
│   └── run_summary.json          样本数 / 显存峰值 / 可训练参数占比
```

显存与参数量在训练脚本里直接埋点：

```python
torch.cuda.reset_peak_memory_stats()
trainer.train()
summary = {
    "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    "trainable_percent": trainable / total * 100,
}
```

### 6.4 消融实验

| 实验 | 变量 | 观察 |
|---|---|---|
| 基线 | base 模型 | 起点 |
| SFT | LoRA rank = 8 / 16 / 32 | rank 与效果、显存的关系 |
| DPO | beta = 0.05 / 0.1 / 0.2 | margin 与过拟合 |
| 数据量 | 300 / 600 / 1000 条 | 数据量与收益的拐点 |

每组只改一个变量，完整配置存进 `outputs/ablation/`。

### 6.5 DPO 的 reference 处理

SFT 之后接 DPO 时，参考策略应当是冻结的 SFT 模型，而不是 base 模型。部分 TRL 版本在 PEFT 场景下会把 reference 退化成 base，导致 DPO 学错方向。做法是复制一份 SFT adapter 作为 reference 并冻结：

```python
def clone_peft_adapter(model, source, target):
    if target in model.peft_config:
        return
    model.add_adapter(target, model.peft_config[source])
    with torch.no_grad():
        for name, param in model.named_parameters():
            if f".{source}." in name:
                model.get_parameter(name.replace(f".{source}.", f".{target}.")).data.copy_(param.data)
```

---

## 七、部署方案

### 7.1 vLLM 启动

```bash
vllm serve Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --enable-lora --lora-modules cs-lora=/path/to/adapter
```

用 `--enable-lora` 动态挂载 adapter，不合并权重，方便多版本切换。

### 7.2 压测方案

```python
# deploy/bench.py：固定 prompt 集，N 并发，记录每请求耗时
并发档位：1 / 5 / 10 / 20
每档 60 秒
指标：QPS、P50/P95/P99、首 token 延迟、显存峰值、错误率
```

三组对比：

| 组 | 配置 |
|---|---|
| A | FP16，无前缀缓存 |
| B | FP16，开 prefix caching |
| C | AWQ 4bit，开 prefix caching |

产出：一张对比表 + 三条结论（量化省多少显存、损失多少精度、前缀缓存对多轮对话的提升幅度）。

---

## 八、应用层方案

### 8.1 五分支编排

```python
graph.add_node('route', node_route)             # 意图判断
graph.add_node('chat', node_chat)               # 闲聊
graph.add_node('retrieve', node_retrieve)       # 政策检索
graph.add_node('rerank', node_rerank)           # 精排 + 门控
graph.add_node('generate', node_generate)       # 带引用生成
graph.add_node('tool_order', node_tool_order)   # 查订单
graph.add_node('tool_refund', node_tool_refund) # 售后
graph.add_node('handoff', node_handoff)         # 转人工
graph.add_node('verify', node_verify)           # 落库 + 反馈
```

意图判断优先用规则 + 小模型，不用大模型，省一次往返。

### 8.2 多轮补参

查订单需要订单号，用户不会主动给。状态里维护 slots：

```python
{'order_id': None, 'phone_tail': None}
```

缺参时生成追问，用户回答后回填。这是「多轮」和「单轮问答」的分水岭，必须实现。

### 8.3 鉴权与数据权限

```python
@app.get('/api/orders/{order_id}')
async def get_order(order_id: str, user_id: str = Depends(current_user)):
    order = repo.get(order_id)
    if order.user_id != user_id:
        raise HTTPException(403, '无权查看该订单')
    return order
```

token 里的 user_id 由服务端签发，用户改不了，可以直接作为数据过滤条件。

### 8.4 限流

Redis 令牌桶，Lua 保证原子性。用户级 20/分 + 全局 10 QPS。压测时要验证超限返回 429 而不是排队。

### 8.5 SSE

生成放在独立 task 里，主循环负责取事件与发心跳；客户端断开时取消生成，避免继续消耗 GPU。

---

## 九、评测体系

### 9.1 四套数据集

| 集 | 规模 | 内容 | 标注方式 |
|---|---|---|---|
| 检索集 | 200–300 | query → 正确文档/段落 | 人工 70 + 规则扩增 |
| 生成集 | 50–80 | query → 人工参考答案 | 全人工 |
| 业务集 | 30–50 | 多轮 + 期望动作（该调哪个工具、该不该转人工） | 全人工 |
| 回归集 | 20 | 每次改动必跑，覆盖五分支 | 从上面挑 |

### 9.2 五层指标

| 层 | 指标 | 目标 |
|---|---|---|
| 检索 | Hit@1 / Recall@10 / MRR | Hit@1 ≥ 0.85 |
| 生成 | 准确率 / 幻觉率 / 引用正确率 | 幻觉率 ≤ 5% |
| 业务 | Auto-Resolve / 闭环率 / 转人工率 | 闭环率 ≥ 80% |
| 系统 | QPS / P99 / 首 token / 显存 / 每千次成本 | P99 ≤ 3s（10 并发） |
| 模型 | base vs SFT vs SFT+DPO + 消融 | SFT+DPO > SFT > base |

### 9.3 Judge rubric

```
你是电商客服质检评审员。评估模型最后一条回复是否适合真实客服场景。

四个字段：
1. accuracy: 0/1      是否准确回答，无答非所问、无虚构平台状态、无明显错误政策
2. auto_resolve: 0/1  是否给出明确下一步动作或操作路径
3. closure: 0/1       是否形成闭环，用户无需再追问
4. satisfaction: 1-5  5=准确自然可执行，1=答非所问或严重虚构

允许：合理行业推断、常见物流时效表达、常见平台操作建议
禁止：虚构订单状态、虚构系统已操作完成、违背上下文

只输出 JSON。
```

### 9.4 Judge 约束

1. 不用同一个模型当生成器和裁判
2. pairwise 优于绝对打分
3. 抽 10% 人工复核，报告 Judge 与人类的一致性

### 9.5 一条命令跑完

```bash
python -m eval.run_all --suite all --output reports/2026-11-05.json
```

输出包含每层指标 + 与上次的 diff。

### 9.6 bad case 闭环

| 归因 | 修复动作 |
|---|---|
| 检索没召回 | 调分块 / RRF 权重 / 补文档 |
| 召回但答错 | 调 prompt / 补 SFT 数据 |
| 分支走错 | 调意图判断 |
| 工具调用失败 | 修工具 / 补参数校验 |

修完的样本进回归集。

---

## 十、进度计划

### 第 1 周：数据 + 微调

| 天 | 任务 | 产出 |
|---|---|---|
| D1 | 目录骨架、config、数据 schema | 能 import 不报错 |
| D2 | 清洗脚本 + 聚类 | 清洗前后对比表、分布图 |
| D3 | build_sft + build_dpo | train/val.jsonl、preference.jsonl |
| D4 | 租卡跑 SFT（先 0.6B 验证，再 8B） | adapter + loss 曲线 |
| D5 | 跑 DPO | adapter + rewards 曲线 |
| D6 | 消融（rank / beta） | 消融表 |
| D7 | 数据与训练报告 | `reports/w1.md` |

### 第 2 周：部署

| 天 | 任务 | 产出 |
|---|---|---|
| D1 | vLLM 起服务，接 LoRA | `/v1/chat/completions` 可调 |
| D2 | AWQ 量化 + 精度对比 | 量化对比表 |
| D3 | 写 bench.py，跑 4 档并发 | 压测原始数据 |
| D4 | 压测报告 + 瓶颈分析 | `reports/w2.md` |
| D5 | Dockerfile + compose | 容器可起 |
| D6-7 | 缓冲 / 补测 | — |

### 第 3 周：应用层

| 天 | 任务 | 产出 |
|---|---|---|
| D1 | 中间件 + JWT + 限流 | 鉴权用例通过 |
| D2 | RAG 检索层 | 检索集指标 |
| D3 | LangGraph 五分支骨架 | 图能跑通 |
| D4 | 四个工具 + 数据权限 | 越权用例被拒 |
| D5 | 多轮补参 | 缺参追问可用 |
| D6 | SSE 流式 + 前端接入 | 端到端可对话 |
| D7 | 业务集联调 | 分支准确率 |

### 第 4 周：评测 + 收尾

| 天 | 任务 | 产出 |
|---|---|---|
| D1-2 | 四套数据集标注 | eval/datasets 完整 |
| D3 | judge.py + run_all.py | 一条命令出报告 |
| D4 | Judge 人工复核 10% | 一致性指标 |
| D5 | bad case 归因 + 修复 | 归因表 |
| D6 | 文档（架构/部署/评测） | README + 3 篇 docs |
| D7 | 演示录屏 | 交付 |

---

## 十一、硬件与成本

| 项 | 配置 | 时长 | 费用 |
|---|---|---|---|
| SFT 8B LoRA | 4090 24G | 6 h | ~15 元 |
| DPO 8B | A100 40G | 5 h | ~30 元 |
| 消融（3 组） | 4090 | 8 h | ~20 元 |
| vLLM 部署 + 压测 | 4090 | 10 h | ~25 元 |
| 缓冲（重跑） | — | 10 h | ~25 元 |
| **合计** | | **~39 h** | **~115 元** |

省钱要点：

1. 本机把代码全写完、用 mock 跑通再上卡
2. 先 0.6B 验证流程，再换 8B
3. `nohup` 后台跑，跑完立刻关机
4. 权重下到数据盘（`/root/autodl-tmp`）
5. 数据用网盘传，别 git

---

## 十二、风险清单

| 风险 | 概率 | 应对 |
|---|---|---|
| 显存不够跑 DPO | 中 | 换 Qwen3-4B，或上 A100 40G |
| 数据造不出 1000 条 | 中 | 降到 500 条，先把链路跑通 |
| 微调后效果不如 base | 中 | 数据质量问题，回到清洗环节查 |
| 压测时 vLLM OOM | 中 | 降 `gpu_memory_utilization` 或 `max_model_len` |
| 4 周做不完 | 高 | 砍掉前端，用 curl 演示；评测集缩到 100 条 |
| 消融结论讲不清 | 中 | 每个实验只改一个变量，完整配置存档 |

---

## 十三、验收标准

- [ ] `docker compose up` 一键起全栈（vLLM + Milvus + MySQL + Redis + app）
- [ ] 登录 → 对话 → 查订单 → 申请退款 → 转人工，五条链路可演示
- [ ] 用户 A 无法查看用户 B 的订单（越权被 403）
- [ ] `python -m eval.run_all` 一条命令产出五层指标
- [ ] 三档模型对比有数字（base / SFT / SFT+DPO）
- [ ] 压测报告有 4 个并发档位的 QPS / P99 / 首 token / 显存
- [ ] 消融表至少 2 组（rank、beta）
- [ ] Judge 与人工一致性 ≥ 80%
- [ ] README 能让别人在 30 分钟内跑起来

---

## 十四、下一步

按第 1 周 D1 开始：建目录骨架 + `config.py` + 数据 schema。

GPU 相关脚本（`train_sft.py` / `train_dpo.py` / `bench.py`）一次性写完，本机用 0.6B + mock 验证逻辑，上卡直接跑。
