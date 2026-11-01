# 电商售后智能客服平台

Qwen3-8B 经 SFT + DPO 微调、vLLM 私有化部署，结合售后政策 RAG 与订单/物流/退款工具调用，
完成「问答 → 查单 → 售后 → 转人工 → 回流」闭环。

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

```bash
source /etc/network_turbo
git clone <repo> && cd ecom-agent
cp .env.example .env      # 填 MODEL_DIR / LLM_BASE_URL
bash setup.sh             # 装 vllm + 训练依赖 + 检查模型
python -m pytest tests -q
```

启动基础设施（Milvus + MySQL + Redis）：

```bash
docker compose -f deploy/docker-compose.yml up -d milvus mysql redis
```

启动模型服务与压测：

```bash
PORT=6006 bash deploy/serve.sh outputs/sft_8b
python -m deploy.bench --base-url http://localhost:6006/v1
```

## 依赖分两份

| 文件 | 装在哪 | 内容 |
|---|---|---|
| `requirements-app.txt` | 本地 + 云端 | 应用层，不含 torch |
| `requirements-train.txt` | 仅云端 | 训练与推理 |

`vllm==0.32.0` 会锁定 `torch==2.13.0` 与 `transformers<5.20.0`，先装它再装其余。

## 文档

- `PLAN.md` 施工图：模块、数据、训练、部署、评测、进度
- `ENV.md` 环境规范：版本矩阵、模型清单、安装顺序
