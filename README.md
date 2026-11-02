# 电商售后智能客服平台

Qwen3-8B 经 SFT + DPO 微调、vLLM 私有化部署，结合售后政策 RAG 与订单/物流/退款工具调用，
完成「问答 → 查单 → 售后 → 转人工 → 回流」闭环。

## 实测数据

单卡 RTX 4090 24G 全流程跑通，以下为真实结果。

| 环节 | 指标 | 结果 |
|---|---|---|
| **检索** | Hit@5 / MRR | **0.98 / 0.872** |
| **生成** | 满意度 / 幻觉率 | **4.60 / 5　0.075** |
| **推理** | QPS @ 并发20 / P95 / 首 token | **7.81 / 2.5s / 151ms** |
| **训练** | 4 轮迭代 eval_loss | 0.741 → **0.424** |
| **工程** | 单元测试 | **85 passed** |

数据规模：1640 条 QA（每知识点 3 种问法）+ 40 条多轮对话 + 9 篇政策文档（41 个 chunk）。

训练曲线、踩坑记录与参数依据见 [`docs/训练记录.md`](docs/训练记录.md)。

## 目录

```
app/        服务层：API、编排、检索、工具、鉴权
train/      离线：数据清洗、SFT、DPO
eval/       评测：数据集、Judge、指标
deploy/     部署：compose、vLLM 启动、压测
data/       原始与处理后数据
```

## 运行环境

**只有 AutoDL（Linux + CUDA）一个环境。** 本地 Windows 仅用于写代码，不参与运行。

### 首次初始化（一次性）

```bash
source /etc/network_turbo
git clone <repo> && cd ecom-agent
cp .env.example .env      # 填 MODEL_DIR / LLM_BASE_URL
bash setup.sh             # 装依赖 + 下模型 + 生成数据 + 跑测试
```

基础设施（Milvus / MySQL）在 AutoDL 上无 Docker，需二进制部署，见 `docs/训练记录.md` 第十节。

### 每次启动

```bash
bash start_all.sh
```

按依赖顺序拉起 MySQL → etcd → minio → milvus → vLLM → app，每个都等到就绪才继续。已在跑的服务自动跳过，可重复执行。

```
==> 全栈就绪
    Milvus   localhost:19530
    vLLM     http://localhost:6006/v1
    App      http://localhost:8080
```

切换 adapter：

```bash
ADAPTER=outputs/sft_8b/checkpoint-308 bash start_all.sh
```

### 压测

```bash
python -m deploy.bench --base-url http://localhost:6006/v1
```

## 文档

| 文件 | 内容 |
|---|---|
| `docs/架构.md` | 五分支编排、三路召回、数据流、权限设计 |
| `docs/API.md` | 接口清单、请求响应示例、SSE 帧格式 |
| `docs/训练记录.md` | 四轮迭代数据、DPO 对比、评测与压测结果、踩坑记录 |
| `PLAN.md` | 施工图：模块、数据、训练、部署、进度 |
| `ENV.md` | 环境规范：版本矩阵、模型清单、安装顺序 |

## 依赖分两份

| 文件 | 装在哪 | 内容 |
|---|---|---|
| `requirements-app.txt` | 本地 + 云端 | 应用层，不含 torch |
| `requirements-train.txt` | 仅云端 | 训练与推理 |

`vllm==0.32.0` 会锁定 `torch==2.13.0` 与 `transformers<5.20.0`，先装它再装其余。
