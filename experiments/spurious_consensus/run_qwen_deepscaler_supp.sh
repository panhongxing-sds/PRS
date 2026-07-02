我感觉我想zheng'shihengshi#!/usr/bin/env bash
# 仅补 Qwen deepscaler 缺口（约 800 题 supp_800）
set -euo pipefail
cd "$(dirname "$0")"
export PANDA_MODELS="${PANDA_MODELS:-/root/autodl-tmp/panda-models}"
GPU=${GPU:-0}
MODEL_TAG=qwen25_3b MODEL_PATH="$PANDA_MODELS/TFB-Qwen2.5-3B-Instruct" GPU="$GPU" \
  python sample.py --benchmark deepscaler --resume
python check_samples.py --tag qwen25_3b
