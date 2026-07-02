#!/usr/bin/env bash
# Plan B scale-up: Qwen3-8B random300 PANDA (same 300 ids as Qwen2.5-3B), 6-GPU Phase A+B.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/env.sh"

export OUT_BASE="${OUT_BASE:-/root/autodl-tmp/prs-outputs/maintable_qwen3_8b_deepscaler_random300}"
export OUT_DIR="${OUT_DIR:-$OUT_BASE/seed41}"
export MODEL_PATH="${MODEL_PATH:-${PANDA_MODELS}/TFB-Qwen3-8B}"
export VARIANTS="${VARIANTS:-/root/autodl-tmp/prs-outputs/qaac_api_bench/deepscaler_random300/variants.jsonl}"
export NUM_GPUS="${NUM_GPUS:-6}"
export PANDA_SKIP_TOKUR=1
# 8B on 5090: leave headroom for KV + HF Phase B on same card (sequential phases)
export GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.78}"
export VLLM_CHUNK_SIZE="${VLLM_CHUNK_SIZE:-256}"
export PANDA_DYNAMIC_CLAIM="${PANDA_DYNAMIC_CLAIM:-1}"
export VLLM_USE_V1=0

exec bash "$ROOT/scripts/run_random300_6gpu.sh" "${1:-all}"
