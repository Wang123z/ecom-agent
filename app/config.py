"""全局配置：路径、密钥、模型名集中在这里，代码里不出现绝对路径。"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _path(key: str, default: str) -> Path:
    """目录配置：相对路径按项目根解析，绝对路径原样返回。"""
    p = Path(os.getenv(key, default))
    return p if p.is_absolute() else BASE_DIR / p


# ---- 模型服务（OpenAI 兼容）----
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-8b")

# ---- 向量与重排：填本地模型路径，留空则走 API ----
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "")

# ---- 存储 ----
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "policy_chunks")
MYSQL_URL = os.getenv("MYSQL_URL", "mysql+pymysql://root:root@localhost:3306/ecom")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ---- 安全 ----
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_EXPIRE_SECONDS = int(os.getenv("JWT_EXPIRE_SECONDS", "7200"))
RATE_LIMIT_USER_PER_MIN = int(os.getenv("RATE_LIMIT_USER_PER_MIN", "20"))
RATE_LIMIT_GLOBAL_QPS = float(os.getenv("RATE_LIMIT_GLOBAL_QPS", "10"))

# ---- 路径 ----
MODEL_DIR = _path("MODEL_DIR", "models")
DATA_DIR = _path("DATA_DIR", "data")
OUTPUT_DIR = _path("OUTPUT_DIR", "outputs")

# ---- 检索 ----
RECALL_TOP_K = int(os.getenv("RECALL_TOP_K", "20"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
RRF_WEIGHTS = [float(x) for x in os.getenv("RRF_WEIGHTS", "1.2,1.0,0.8").split(",")]
RERANK_LOW = float(os.getenv("RERANK_LOW", "0.4"))
RERANK_HIGH = float(os.getenv("RERANK_HIGH", "0.7"))

# ---- 日志 ----
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
