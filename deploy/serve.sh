#!/usr/bin/env bash
# 启动 vLLM 的 OpenAI 兼容服务。
#
# 用法
#   bash deploy/serve.sh                        # 只起基座模型
#   bash deploy/serve.sh outputs/sft_8b         # 额外挂载 LoRA adapter
#   PORT=6006 bash deploy/serve.sh              # 换端口（AutoDL 自定义服务用 6006）
set -e

MODEL_ROOT="${MODEL_DIR:-/root/autodl-tmp/models}"
MODEL="${MODEL_ROOT}/Qwen3-8B"
ADAPTER="$1"
PORT="${PORT:-8000}"

ARGS=(
  --served-model-name qwen3-8b
  --host 0.0.0.0
  --port "$PORT"
  --max-model-len 4096
  --gpu-memory-utilization 0.85
  --enable-prefix-caching
)

if [ -n "$ADAPTER" ]; then
  ARGS+=(--enable-lora --lora-modules "cs-lora=${ADAPTER}")
  echo ">>> 挂载 LoRA: $ADAPTER"
fi

echo ">>> 模型: $MODEL  端口: $PORT"
exec vllm serve "$MODEL" "${ARGS[@]}"
