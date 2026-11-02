#!/usr/bin/env bash
# AutoDL / Linux 一键环境初始化。可重复执行，缺什么补什么。
# 用法：bash setup.sh
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "==> Python 版本"
python -V

echo "==> 安装 vLLM（会锁定 torch 与 transformers 版本，必须最先装）"
pip install "vllm==0.32.0" -i "$PIP_INDEX"

echo "==> 安装训练依赖"
pip install -r requirements-train.txt -i "$PIP_INDEX"

echo "==> 安装应用依赖"
pip install -r requirements-app.txt -i "$PIP_INDEX"

echo "==> 检查依赖冲突"
pip check

echo "==> 检查模型（缺失则下载，已完整则跳过）"
python scripts/download_models.py --root "$MODEL_ROOT"

if [ ! -f .env ]; then
  echo "==> 未找到 .env，从 .env.example 复制，请手动填写后再启动"
  cp .env.example .env
fi

echo "==> 生成训练数据（清洗 → 聚类 → SFT 样本）"
python -m train.clean
python -m train.cluster
python -m train.build_sft

echo "==> 跑单元测试"
python -m pytest tests -q

echo "==> 完成。"
echo "    建索引：python -m scripts.build_index"
echo "    启动服务：uvicorn app.main:app --host 0.0.0.0 --port 8080"
