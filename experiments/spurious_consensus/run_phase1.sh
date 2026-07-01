#!/usr/bin/env bash
# Phase 1：三模型顺序全量采样（HF transformers）
set -euo pipefail
cd "$(dirname "$0")"
export PRS_ROOT="${PRS_ROOT:-/root/PRS}"
export PRS_MODELS="${PRS_MODELS:-/root/autodl-tmp/prs-models}"
GPU=${GPU:-0}
LOG_DIR="${LOG_DIR:-logs/phase1}"
mkdir -p "$LOG_DIR"

run_one() {
  local tag="$1" path="$2"
  echo ""
  echo "########## $tag ##########"
  MODEL_TAG="$tag" MODEL_PATH="$path" GPU="$GPU" \
    ./run_sampling.sh 2>&1 | tee "$LOG_DIR/${tag}.log"
}

# Qwen 已有样本，仅补 deepscaler 缺口（如需要）
# python check_samples.py --tag qwen25_3b
# MODEL_TAG=qwen25_3b MODEL_PATH=$PRS_MODELS/TFB-Qwen2.5-3B-Instruct GPU=$GPU ./run_sampling_vllm.sh

run_one llama32_1b "$PRS_MODELS/TFB-Llama-3.2-1B-Instruct"
run_one phi4_mini   "$PRS_MODELS/Phi-4-mini-instruct"
run_one gemma3_4b   "$PRS_MODELS/gemma-3-4b-it"

echo ""
echo "Phase 1 完成。运行: python check_samples.py --tag <tag>"
