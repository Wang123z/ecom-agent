#!/usr/bin/env bash
# 消融实验：SFT 5 组 + DPO beta 3 组
# 用法
#   bash scripts/ablation.sh sft     # 只跑 SFT 消融（约 2 小时）
#   bash scripts/ablation.sh dpo     # 只跑 DPO 消融（约 1 小时，需先有 SFT 产物）
#   bash scripts/ablation.sh all
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

run_sft() {
  local tag="$1" rank="$2" targets="$3"
  echo "===== SFT $tag (rank=$rank targets=$targets)"
  python -m train.train_sft --config train/configs/sft.yaml \
    --lora-rank "$rank" --lora-targets "$targets" --tag "$tag"
}

run_dpo() {
  local beta="$1"
  echo "===== DPO beta=$beta"
  python -m train.train_dpo --config train/configs/dpo.yaml \
    --beta "$beta" --tag "beta$beta"
}

case "${1:-all}" in
  sft)
    run_sft rank8_qv      8  "q_proj,v_proj"
    run_sft rank16_qv    16  "q_proj,v_proj"
    run_sft rank32_qv    32  "q_proj,v_proj"
    run_sft rank16_qvko  16  "q_proj,k_proj,v_proj,o_proj"
    run_sft rank16_all   16  "all"
    ;;
  dpo)
    for b in 0.05 0.1 0.3; do run_dpo "$b"; done
    ;;
  all)
    "$0" sft
    "$0" dpo
    ;;
  *)
    echo "用法: bash scripts/ablation.sh [sft|dpo|all]"
    exit 1
    ;;
esac

echo ""
echo "===== 跑完，画 loss 曲线"
python -m scripts.plot_loss --root outputs
