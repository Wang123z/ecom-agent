# 电商售后智能客服平台

Qwen3-8B 经 SFT + DPO 微调、vLLM 私有化部署，结合售后政策 RAG 与订单/物流/退款工具调用，
完成「问答 → 查单 → 售后 → 转人工 → 回流」闭环，带登录鉴权、数据权限与四层评测。

单卡 RTX 4090 24G 全流程跑通，从数据治理到服务部署每环都有实测数字。

## 实测数据

| 环节 | 指标 | 结果 |
|---|---|---|
| **检索** | Hit@5 / MRR | **0.98 / 0.872**（50 条） |
| **生成** | 满意度 / 幻觉率 | **4.09 / 5　0.00**（40 条中 22 条完成） |
| **业务** | 意图路由准确率 | **1.00**（8 条） |
| **回归** | 通过率 | **0.80**（16/20） |
| **推理** | QPS @ 并发20 / P95 / 首 token | **7.81 / 2.5s / 151ms** |
| **工程** | 单元测试 | **86 passed** |

> 生成评测 40 条中 22 条完成——裁判 API 免费额度耗尽。22 条偏乐观（前半是基础问题，后半是难品类），不作为最终结论。

## 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 基座模型 | **Qwen3-8B** | 与 E-Commerce-Customer-Service 一致 |
| 微调 | **LoRA + TRL + PEFT** | rank 32，target 全部线性层，trainable 1.05% |
| 两阶段训练 | **SFT → DPO** | DPO 用冻结 SFT adapter 作 reference |
| 推理 | **vLLM 0.32** | OpenAI 兼容接口，LoRA 热挂载，prefix caching |
| 应用 | **FastAPI + SSE** | 流式输出，纯 ASGI 中间件 |
| 编排 | **LangGraph** | 五分支状态机 + 检查点，比 LangChain Agent 更可控 |
| 向量库 | **Milvus 2.5.4** | standalone 三进程（etcd + minio + milvus） |
| 检索 | **bge-m3 稠密 + 稀疏** + **bge-reranker-v2-m3** | 三路召回 + RRF + 精排 |
| 存储 | **MySQL + Redis** | 订单/工单/反馈；令牌桶限流 |
| 安全 | **JWT + Redis 令牌桶** | 20 次/分 + 突发 5，Redis 挂了降级放行 |
| 评测 | **qwen3.8-flash（第三方 Judge）** | 与生成模型不同源，避免自评自测 |

## 与两个参考项目的对齐情况

| 能力 | enterprise-kb-qa | E-Commerce-Customer-Service | 本项目 |
|---|---|---|---|
| 基座 | Qwen2.5-7B | Qwen3-8B | **Qwen3-8B** |
| 微调 | LLaMA-Factory | LoRA + TRL + PEFT | **LoRA + TRL + PEFT**（自写脚本） |
| 训练阶段 | SFT | SFT + DPO | **SFT + DPO** |
| 推理 | vLLM | — | **vLLM** |
| RAG | Milvus + bge-m3 + reranker | — | **三路召回（稠密+稀疏+FAQ）+ RRF + reranker** |
| 编排 | LangChain Agent | — | **LangGraph 五分支** |
| 流式 | ✗ | ✗ | **✓ SSE** |
| 鉴权限流 | ✗ | ✗ | **✓ JWT + Redis 令牌桶** |
| 业务数据 | ✗ | ✗ | **✓ MySQL 订单/工单/反馈** |
| 评测 | before/after 表（待填） | 四维 Judge | **四套数据集 + 真实数字** |
| 单元测试 | ✗ | ✗ | **✓ 86 passed** |
| 数据规模 | — | 9408 条 | **9408 条**（同源数据集） |
| K8s / Spring Boot | ✓ | ✗ | ✗（计划排除） |
| DeepSpeed 多卡 | ✗ | ✓ | ✗（单卡 4090） |
| LLaMA-Factory | ✓ | ✗ | ✗（与 vLLM 依赖冲突，见下） |

**核心能力已对齐，差异均为主动取舍。**

### 为什么不用 LLaMA-Factory

LLaMA-Factory 0.9.5 要求 `transformers<=5.6.0`，vLLM 0.32 要求 `>=5.16.1`，差 11 个小版本，无法共存。
装进主环境会把 vLLM 依赖全部降级。评估后选择自写训练脚本——消融实验只需加 3 个 CLI 参数，
且自己实现 chat template 渲染、label mask、LoRA 挂载、DPO 双 adapter 更有可讲性。

## 架构

```
用户提问
   ↓ JWT 鉴权 + Redis 令牌桶（20/分，突发 5）
   ↓ LangGraph 意图路由（规则判断，不调模型）
   ├─ chat        闲聊
   ├─ rag         三路召回 → RRF → 精排 → 生成（带引用）
   ├─ order       查订单 / 查物流（缺参追问，校验归属）
   ├─ refund      提交退款（时效判断）
   └─ handoff     转人工建工单
   ↓ SSE 流式返回 + 落库
```

**意图路由的关键设计**：强弱词分级。「物流」是领域词（可能问政策也可能查件），
「到哪」是意图词（一定是查件）。强词直接判 order，弱词必须配动作词。

## 功能清单

| # | 功能 | 关键点 |
|---|---|---|
| 1 | 登录 | 手机号 + 固定验证码 → JWT，user_id 服务端签发不可篡改 |
| 2 | 政策问答 | 三路召回 + RRF + 精排，SSE 流式，返回 5 条引用（文档/章节/原文） |
| 3 | 查订单 | 多轮补参 + 归属校验（越权返回 404 而非 403，防探测） |
| 4 | 售后申请 | 时效判断 + 工具调用 + 工单号返回 |
| 5 | 转人工 | 投诉词触发，建工单 |
| 6 | 评价回流 | `POST /api/feedback` 落库，供后续做偏好数据 |

## 目录结构

```
ecom-agent/
├── app/
│   ├── main.py          入口 + 纯 ASGI 中间件（request_id / 统一错误）
│   ├── config.py        配置集中，无硬编码路径
│   ├── guard.py         JWT + Redis 令牌桶
│   ├── api/             auth / chat / orders / tickets / feedback
│   ├── core/
│   │   ├── graph.py     LangGraph 五分支 + 意图路由
│   │   ├── rag/         三路召回 / RRF / 精排
│   │   ├── tools.py     查订单 / 查物流 / 退款 / 建工单
│   │   └── state.py     会话状态
│   └── infra/           vllm / milvus / mysql / embed
├── train/
│   ├── clean.py         清洗去重（支持 ModelScope messages 格式）
│   ├── cluster.py       问题聚类 + 答案冲突检测
│   ├── build_sft.py     单轮 + 多轮样本构建
│   ├── build_dpo.py     候选生成 + Judge 打分 + 偏好对构造
│   ├── train_sft.py     LoRA SFT（支持 --lora-rank / --targets / --epochs）
│   └── train_dpo.py     DPO（支持 --beta）
├── eval/
│   ├── datasets/        retrieval(50) / generation(40) / business(8) / regression(20)
│   ├── judge.py         四维打分（独立 JUDGE_* 配置）
│   └── run_all.py       一条命令跑四套
├── scripts/
│   ├── build_index.py       政策文档切分 + 向量化入 Milvus
│   ├── augment_closure.py   按品类补操作路径
│   ├── build_multiturn.py   按品类配对生成多轮
│   ├── attribute_badcase.py bad case 四类归因
│   ├── plot_loss.py         重绘 loss 曲线
│   └── seed_mock.py         建表 + 插模拟数据
├── deploy/              serve.sh（vLLM 启动）/ bench.py（压测）/ compose
├── data/                原始数据 + 9 篇政策文档
├── docs/                架构 / API / 训练记录
└── start_all.sh         一键起全栈
```

## 数据与训练

### 数据

| 类型 | 规模 | 来源 |
|---|---|---|
| 政策文档 | 9 篇 / 49 节 → 41 chunk | 手写 |
| 真实客服 QA | 9408 条（7526/1882） | ModelScope 电商客服数据集，24.6 万条筛选 |
| 多轮对话 | 1173 条（13.5%） | 40 条手写 + 品类配对生成 |
| 操作路径覆盖 | **98.4%** | 按品类补全（原 14.4%） |

### 训练

| 阶段 | 配置 | 结果 |
|---|---|---|
| SFT | rank 32 / alpha 64 / 全部线性层 / 1 epoch / 979 步 | train_loss 1.012，eval_loss 0.877，token 准确率 82.5%，62 分钟 |
| DPO | 300 偏好对 / beta 0.1 / 2 epoch / 68 步 | eval_loss 0.371，六条测试 SFT/DPO 对比 |

早期用自造 1640 条精编数据跑过 5 轮迭代，eval_loss 0.741 → 0.424。
换成 9408 条真实数据后准确率下降、可执行性上升——**数据规模与质量是 trade-off**，
详见 [`docs/训练记录.md`](docs/训练记录.md) 第十二节。

## 评测体系

### 四套数据集

| 集 | 规模 | 标什么 |
|---|---|---|
| 检索 | 50 | query → 正确文档/章节（覆盖 49 节） |
| 生成 | 40 | query → 人工参考答案（四维打分） |
| 业务 | 8 | 多轮对话 + 期望意图 |
| 回归 | 20 | 五分支覆盖，每次改动必跑 |

### 四层指标

| 层 | 指标 |
|---|---|
| 检索 | Hit@1/5/10、Recall、MRR |
| 生成 | accuracy、auto_resolve、closure、hallucination_rate、satisfaction |
| 业务 | 意图路由准确率 |
| 系统 | QPS、P50/P95/P99、首 token、峰值显存 |

### Judge 设计

- **与生成模型不同源**：qwen3.8-flash（第三方），避免自评自测
- **四维 rubric**：明确写「允许什么、禁止什么」，比只给 1–5 分稳定
- **只输出 JSON**：解析失败按 0 分并记入 failed
- **剥离思考块**：`parse_json` 先删 `<think>...</think>`，兼容思考型模型

> 换裁判前后的对比：自评（cs-lora）closure 0.85 → 第三方评 0.425。
> **自评会系统性高估**，这是本项目最重要的评测教训。

## 运行环境

**只有 AutoDL（Linux + CUDA）一个环境。** 本地 Windows 仅用于写代码，不参与运行。

### 首次初始化（一次性）

```bash
source /etc/network_turbo
git clone <repo> && cd ecom-agent
cp .env.example .env      # 填 MODEL_DIR / LLM_BASE_URL / JUDGE_*
bash setup.sh             # 装依赖 + 下模型 + 生成数据 + 跑测试
```

基础设施（Milvus / MySQL）在 AutoDL 上无 Docker，需二进制部署，见 `docs/训练记录.md` 第十节。

### 每次启动

```bash
bash start_all.sh
```

按依赖顺序拉起 MySQL → Redis → etcd → minio → milvus → vLLM → app，每个都等到就绪才继续。
已在跑的服务自动跳过，可重复执行。

```
==> 全栈就绪
    Milvus   localhost:19530
    vLLM     http://localhost:6006/v1
    App      http://localhost:8080
```

切换 adapter：

```bash
ADAPTER=outputs/sft_8b_9408 bash start_all.sh
```

### 评测与压测

```bash
python -m eval.run_all --suite all          # 四套评测
python -m deploy.bench --base-url http://localhost:6006/v1   # 压测
```

## 依赖分两份

| 文件 | 装在哪 | 内容 |
|---|---|---|
| `requirements-app.txt` | 本地 + 云端 | 应用层，不含 torch |
| `requirements-train.txt` | 仅云端 | 训练与推理 |

`vllm==0.32.0` 会锁定 `torch==2.13.0` 与 `transformers<5.20.0`，先装它再装其余。

## 文档

| 文件 | 内容 |
|---|---|
| `docs/架构.md` | 八节：链路、意图路由、三路召回、数据流、权限、限流、ASGI、目录 |
| `docs/API.md` | 5 个业务接口 + vLLM 接口 + SSE 帧格式 + 错误格式 |
| `docs/训练记录.md` | 十三节：五轮迭代、DPO、评测、压测、端到端、9408 实验、路由修复 |
| `PLAN.md` | 施工图：模块、数据、训练、部署、进度 |
| `ENV.md` | 环境规范：版本矩阵、模型清单、安装顺序 |

## 已知局限

| 项 | 说明 |
|---|---|
| 生成评测 22/40 | 裁判 API 免费额度耗尽，样本不全 |
| 真实数据政策不一致 | 不同商家口径不同，147 个冲突簇；「奶粉能退」被通用常识覆盖 |
| 客服口癖 | 学到「亲亲…呢」，真实数据副作用 |
| 无消融实验 | LoRA rank / DPO beta 未做系统扫描（脚本已就绪） |
| 无前端 | 仅 API，计划中的 React 前端未做 |
