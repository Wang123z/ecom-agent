# 环境与版本规范

> 运行环境只有一个：**AutoDL（Linux + CUDA）**。本地 Windows 仅用于写代码，不参与运行。
> 原则：**vLLM 是版本锚点**。它 pin 死了 torch 和 transformers 的版本区间，其他包围绕它适配。

---

## 一、环境

| 项 | 值 |
|---|---|
| 平台 | AutoDL 实例（Linux） |
| Python | 3.11 |
| CUDA | 12.4 |
| GPU | 4090 24G（主力）/ A100 40G（DPO） |
| 工作目录 | `/root/autodl-tmp/ecom-agent` |
| 模型目录 | `/root/autodl-tmp/models` |

**Python 必须 3.11**：vLLM 0.32.0 要求 `>=3.11,<3.15`，TRL 1.17.1 要求 `>=3.11`。

---

## 二、版本锚点：为什么不能装最新

实测当前 PyPI 最新版本：

| 包 | 最新版 | 但 vLLM 要求 |
|---|---|---|
| torch | 2.15.0 | **==2.13.0**（硬 pin） |
| transformers | 5.22.1 | **<5.20.0, >=5.16.1** |
| vllm | 0.32.0 | — |

**装 vllm 0.32.0 会自动把 torch 降到 2.13.0、transformers 降到 5.19.x。** 先装 torch 2.15 再装 vllm，会来回卸载重装，甚至装崩。

正确顺序：**先装 vllm，让它决定 torch 和 transformers**，再装 trl / peft。

---

## 三、依赖清单

### 3.1 应用层 `requirements-app.txt`

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

不含 torch。配置用 `os.getenv` + `dotenv`，不引入 `pydantic-settings`。

### 3.2 训练与推理 `requirements-train.txt`

```
trl==1.17.1
peft==0.21.2
accelerate>=1.11.0
datasets>=4.7.0
modelscope==1.28.0
sentence-transformers==6.2.0
FlagEmbedding==1.3.5
```

`vllm==0.32.0` 单独先装，它会带入匹配的 torch 与 transformers。

`sentence-transformers` / `FlagEmbedding` 复用 vllm 装好的 torch，不重复下载。

### 3.3 embedding 与 rerank 一律用本地模型

`app/infra/embed.py` 和 `app/infra/rerank.py` 直接加载本地 bge-m3 / bge-reranker-v2-m3，不做 API 兜底——私有化部署的向量化也应在本地完成。

`EMBEDDING_MODEL` / `RERANK_MODEL` 未配置时直接报错，不静默降级。

---

## 四、模型清单

| 模型 | 用途 | 精度 | 大小 |
|---|---|---|---|
| `Qwen/Qwen3-8B` | 微调基座 + 推理 | bf16 | ~16.4 GB |
| `Qwen/Qwen3-0.6B` | 流程验证（先跑通再换 8B） | bf16 | ~1.5 GB |
| `Qwen/Qwen3-4B` | 显存不够时的降级方案 | bf16 | ~8.2 GB |
| `BAAI/bge-m3` | 稠密 + 稀疏向量（1024 维） | fp16 | ~2.3 GB |
| `BAAI/bge-reranker-v2-m3` | 精排 | fp16 | ~2.3 GB |

**必须放 `/root/autodl-tmp/models/`**。系统盘只有 30–50G，8B 权重就占 16G。

下载用项目脚本（检查完整性，缺了才下）：

```bash
python scripts/download_models.py                      # 全部
python scripts/download_models.py --only Qwen3-0.6B    # 只下 0.6B
```

**下载用无卡模式**（0.1 元/小时），不要开着 GPU 下 16G 模型。

---

## 五、基础设施服务

由 `deploy/docker-compose.yml` 提供：

| 服务 | 版本 | 端口 |
|---|---|---|
| Milvus（含 etcd + minio） | 2.5.4 standalone | 19530 |
| MySQL | 8.0 | 3306 |
| Redis | 7.4 | 6379 |

```bash
docker compose -f deploy/docker-compose.yml up -d milvus mysql redis
```

Milvus 必须是完整版：项目用了稀疏向量，Milvus Lite 不支持。

Redis 用普通 7.4 即可，不需要 Redis Stack——项目不使用 LangGraph 的 Redis checkpoint。

---

## 六、环境变量

`.env` 由 `.env.example` 复制后填写。

```env
# 模型服务（vLLM 的 OpenAI 兼容接口）
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=not-needed
LLM_MODEL=qwen3-8b

# 本地模型路径（留空则检索不可用，会直接报错）
EMBEDDING_MODEL=/root/autodl-tmp/models/bge-m3
EMBEDDING_DIM=1024
RERANK_MODEL=/root/autodl-tmp/models/bge-reranker-v2-m3

# 存储
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=policy_chunks
MYSQL_URL=mysql+pymysql://root:root@localhost:3306/ecom
REDIS_URL=redis://localhost:6379/0

# 安全
JWT_SECRET=change-me-in-production
JWT_EXPIRE_SECONDS=7200
RATE_LIMIT_USER_PER_MIN=20
RATE_LIMIT_GLOBAL_QPS=10

# 路径（相对路径按项目根解析）
MODEL_DIR=/root/autodl-tmp/models
DATA_DIR=data
OUTPUT_DIR=outputs
```

---

## 七、环境验证

三步全过才算就绪。

```bash
# 1. 依赖冲突
pip check

# 2. 版本对齐
python -c "
import torch, transformers, vllm
print(torch.__version__, transformers.__version__, vllm.__version__)
print('cuda', torch.version.cuda, torch.cuda.is_available())
"

# 3. 模型可加载（无卡模式即可跑）
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen3-0.6B')
print(tok.apply_chat_template([{'role':'user','content':'你好'}], tokenize=False))
"
```

第 3 步要看到 `<|im_start|>` 标记，否则 chat template 有问题，训练会白跑。

---

## 八、已知坑

| 坑 | 现象 | 处理 |
|---|---|---|
| 先装 torch 再装 vllm | 反复卸载重装，甚至失败 | 先装 vllm |
| 装最新 transformers | vllm import 失败 | 锁 `<5.20.0` |
| 模型下到 `/root` | 系统盘满，实例起不来 | 一律下到 `/root/autodl-tmp` |
| CRLF 换行 | 执行 `.sh` 报 `\r: command not found` | `.gitattributes` 已配 `*.sh text eol=lf` |
| `pip check` 报 torchvision 冲突 | vllm 依赖 torchvision 0.28.0 | 不要手动升级 torchvision |
| `network_turbo` 开着装 pip | 国内源反而变慢 | `unset http_proxy https_proxy` 后再装 |
| GitHub pull 报 HTTP2 错误 | `curl 16 Error in the HTTP2 framing layer` | `git config --global http.version HTTP/1.1` |

---

## 九、安装顺序（照抄即可）

```bash
source /etc/network_turbo
cd /root/autodl-tmp
git clone <repo> && cd ecom-agent
cp .env.example .env

unset http_proxy https_proxy
pip install vllm==0.32.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements-train.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements-app.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

pip check
python -m pytest tests -q
```

镜像选 PyTorch 2.x + CUDA 12.4 + Python 3.11。如果镜像自带的 torch 不是 2.13.0，选「基础镜像 + 自己装」更省事。
