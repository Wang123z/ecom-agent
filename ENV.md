# 环境与版本规范

> 原则：**vLLM 是版本锚点**。它 pin 死了 torch 和 transformers 的版本区间，其他包围绕它适配。
> 不要按「装最新版」的思路装，否则会冲突到跑不起来。

---

## 一、两个环境的定位

| 环境 | 用途 | Python | GPU | CUDA |
|---|---|---|---|---|
| Windows 本地 | 写代码、数据脚本、应用层、评测 | 3.11 | 无 | 无 |
| AutoDL 实例 | 训练、vLLM 部署、压测 | 3.11 | 4090 24G / A100 40G | 12.4 |

**Python 统一 3.11**，理由：

- vLLM 0.32.0 要求 `>=3.11,<3.15`
- TRL 1.17.1 要求 `>=3.11`
- 3.11 是当前生态里兼容性最好的版本

本地如果已经是 3.12/3.13，建议用 conda 建一个 3.11 环境，不要凑合。

```bash
conda create -n ecom python=3.11 -y
conda activate ecom
```

---

## 二、版本锚点：为什么不能装最新

实测当前 PyPI 最新版本：

| 包 | 最新版 | 但 vLLM 要求 |
|---|---|---|
| torch | 2.15.0 | **==2.13.0**（硬 pin） |
| transformers | 5.22.1 | **<5.20.0, >=5.16.1** |
| vllm | 0.32.0 | — |

也就是说：**装 vllm 0.32.0 会自动把 torch 降到 2.13.0、transformers 降到 5.19.x**。如果先装了 torch 2.15，再装 vllm，会来回卸载重装，甚至装崩。

正确顺序：**先装 vllm，让它决定 torch 和 transformers**，再装 trl / peft。

---

## 三、依赖清单

### 3.1 应用层（本地 + 云端都装）

`requirements-app.txt`

```
fastapi==0.162.0
uvicorn[standard]==0.54.0
httpx==0.28.1
openai==3.42.0

langgraph==1.2.19
langchain-core==1.7.3
langchain-openai==1.7.3

pymilvus==3.0.2
redis==8.1.0
sqlalchemy==2.1.6
pymysql==1.2.3

python-jose[cryptography]==3.5.0
python-multipart==0.0.20

loguru==0.7.3
python-dotenv==1.2.4

pytest==8.4.2
```

不用 `pydantic-settings`：配置只有几十个常量，`os.getenv` + `dotenv` 更直接，也少一层依赖。

**不包含** torch / sentence-transformers。本地开发和云端部署的 embedding 走两条路：

| 场景 | embedding | rerank | 需要 torch |
|---|---|---|---|
| 本地开发 | DashScope `text-embedding-v3` | DashScope `gte-rerank` | ✗ |
| 云端部署 | 本地 `bge-m3` | 本地 `bge-reranker-v2-m3` | ✓ |

接口统一成 OpenAI 兼容 + 自定义 `/rerank`，切换只改 `EMBEDDING_BASE_URL`。这样本地环境轻量（不装 torch），云端能力完整。

### 3.2 训练与推理（仅 AutoDL）

`requirements-train.txt`

```
# 先装 vllm，它会带入正确的 torch 与 transformers
vllm==0.32.0
# 装完 vllm 后再装这些
trl==1.17.1
peft==0.21.2
accelerate>=1.11.0
datasets>=4.7.0
modelscope==1.28.0
```

装完 vllm 后，torch 与 transformers 由它锁定，不要再手动 `pip install torch`。

### 3.3 云端 embedding 依赖（可选，与训练同环境）

```
sentence-transformers==6.2.0
FlagEmbedding==1.3.5
```

`sentence-transformers` 会复用 vllm 装好的 torch，不会重复下载。

### 3.4 本地如果要跑 embedding（不推荐）

```
torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
sentence-transformers==6.2.0
```

CPU 版 torch 约 800MB，跑 bge-m3 一条 query 约 200ms，能接受但没必要——本地直接用 DashScope 更快。

---

## 四、模型清单

| 模型 | 用途 | 精度 | 大小 | 部署位置 |
|---|---|---|---|---|
| `Qwen/Qwen3-8B` | 微调基座 + 推理 | bf16 | ~16.4 GB | AutoDL |
| `Qwen/Qwen3-0.6B` | 流程验证（先跑通再换 8B） | bf16 | ~1.5 GB | AutoDL |
| `Qwen/Qwen3-4B` | 显存不够时的降级方案 | bf16 | ~8.2 GB | AutoDL |
| `BAAI/bge-m3` | 稠密 + 稀疏向量（1024 维） | fp16 | ~2.3 GB | AutoDL |
| `BAAI/bge-reranker-v2-m3` | 精排 | fp16 | ~2.3 GB | AutoDL |

**存放路径（AutoDL）**：

```
/root/autodl-tmp/models/
├── Qwen3-8B/
├── Qwen3-0.6B/
├── bge-m3/
└── bge-reranker-v2-m3/
```

必须放 `autodl-tmp`（数据盘）。系统盘只有 30–50G，8B 权重就占 16G，放 `/root` 会满。

**下载命令（ModelScope，国内快）**：

```bash
pip install modelscope

modelscope download --model Qwen/Qwen3-8B \
  --local_dir /root/autodl-tmp/models/Qwen3-8B

modelscope download --model Qwen/Qwen3-0.6B \
  --local_dir /root/autodl-tmp/models/Qwen3-0.6B

modelscope download --model BAAI/bge-m3 \
  --local_dir /root/autodl-tmp/models/bge-m3

modelscope download --model BAAI/bge-reranker-v2-m3 \
  --local_dir /root/autodl-tmp/models/bge-reranker-v2-m3
```

下载用**无卡模式**（0.1 元/小时），不要开着 GPU 下 16G 模型。

---

## 五、基础设施服务

| 服务 | 版本 | 端口 | 本地 | 云端 |
|---|---|---|---|---|
| Milvus | 2.5.x standalone | 19530 | Docker Desktop | Docker |
| MySQL | 8.0 | 3306 | Docker Desktop | Docker |
| Redis | 7.4 | 6379 | Docker Desktop | Docker |

Windows 上 Milvus 必须走 Docker（没有原生 Windows 版）。Docker Desktop 需开启 WSL2 后端。

`deploy/docker-compose.dev.yml`（本地只起这三个）：

```yaml
services:
  milvus:
    image: milvusdb/milvus:v2.5.4
    command: ["milvus", "run", "standalone"]
    ports: ["19530:19530", "9091:9091"]
    volumes: ["milvus_data:/var/lib/milvus"]

  mysql:
    image: mysql:8.0
    ports: ["3306:3306"]
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: ecom
    volumes: ["mysql_data:/var/lib/mysql"]

  redis:
    image: redis:7.4-alpine
    ports: ["6379:6379"]
    command: ["redis-server", "--appendonly", "yes"]
    volumes: ["redis_data:/data"]

volumes:
  milvus_data:
  mysql_data:
  redis_data:
```

Redis 用普通 7.4 即可，不需要 Redis Stack——项目不使用 LangGraph 的 Redis checkpoint。

---

## 六、环境变量

`.env.example`

```env
# ---- 模型服务（OpenAI 兼容）----
# 本地：DashScope 或 Ollama；云端：vLLM
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=qwen-plus

# ---- 向量与重排 ----
# 本地：DashScope；云端：http://localhost:8001（本地 bge 服务）
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_DIM=1024
RERANK_BASE_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
RERANK_MODEL=gte-rerank

# ---- 存储 ----
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=policy_chunks
MYSQL_URL=mysql+pymysql://root:root@localhost:3306/ecom
REDIS_URL=redis://localhost:6379/0

# ---- 安全 ----
JWT_SECRET=change-me-in-production
JWT_EXPIRE_SECONDS=7200
RATE_LIMIT_USER_PER_MIN=20
RATE_LIMIT_GLOBAL_QPS=10

# ---- 路径（本地/云端各写一份）----
# 本地
MODEL_DIR=E:/models
DATA_DIR=E:/agent_project/ecom-agent/data
# 云端
# MODEL_DIR=/root/autodl-tmp/models
# DATA_DIR=/root/autodl-tmp/data
```

---

## 七、环境验证

装完必须跑这三步，任何一步失败都不算环境就绪。

```bash
# 1. 依赖冲突检查
pip check

# 2. 版本对齐检查
python -c "
import torch, transformers, vllm
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('vllm', vllm.__version__)
print('cuda', torch.version.cuda, torch.cuda.is_available())
"

# 3. 模型可加载（云端，无卡模式即可跑）
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen3-0.6B')
print('vocab', len(tok))
print(tok.apply_chat_template([{'role':'user','content':'你好'}], tokenize=False))
"
```

第 3 步输出要能看到 Qwen 的 `<|im_start|>` 标记，否则 chat template 有问题，训练会白跑。

---

## 八、已知坑

| 坑 | 现象 | 处理 |
|---|---|---|
| 先装 torch 再装 vllm | vllm 安装时反复卸载重装 torch，甚至失败 | 先装 vllm，让它决定 torch |
| 装最新 transformers | vllm 报版本不兼容或 import 失败 | 锁 `<5.20.0` |
| 模型下到 `/root` | 系统盘满，实例起不来 | 一律下到 `/root/autodl-tmp` |
| Windows 默认 GBK | 读中文 jsonl 乱码 | 所有 `open()` 显式 `encoding='utf-8'` |
| CRLF 换行 | 云端执行 `.sh` 报 `\r: command not found` | `.gitattributes` 配 `*.sh text eol=lf` |
| 本地装 CPU 版 torch 后上云 | 云端 pip 认为 torch 已满足，跳过 GPU 版 | 云端独立虚拟环境，不要同步 site-packages |
| `pip check` 报 torchvision 冲突 | vllm 依赖 torchvision 0.28.0，与 torch 2.13 配套 | 不要手动升级 torchvision |

---

## 九、安装顺序（照抄即可）

### 本地 Windows

```bash
conda create -n ecom python=3.11 -y
conda activate ecom
pip install -r requirements-app.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip check
```

### AutoDL

```bash
# 镜像选 PyTorch 2.13.0 + CUDA 12.4 + Python 3.11
cd /root/autodl-tmp/ecom-agent

# 1. 先装 vllm（会带入 torch 2.13.0 和 transformers 5.19.x）
pip install vllm==0.32.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 再装训练与其余依赖
pip install -r requirements-train.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements-app.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 验证
pip check
python -c "import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__)"
```

**注意**：如果 AutoDL 镜像自带的 torch 版本不是 2.13.0，选「基础镜像 + 自己装」更省事，避免镜像里的预装包干扰。
