#!/usr/bin/env bash
# Plan B: Llama-3.2-1B random300 PANDA (same 300 ids as Qwen), 6-GPU Phase A+B.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/env.sh"

export OUT_BASE="${OUT_BASE:-/root/autodl-tmp/panda-outputs/maintable_llama32_1b_deepscaler_random300}"
export OUT_DIR="${OUT_DIR:-$OUT_BASE/seed41}"
export MODEL_PATH="${MODEL_PATH:-${PANDA_MODELS}/TFB-Llama-3.2-1B-Instruct}"
export VARIANTS="${VARIANTS:-/root/autodl-tmp/panda-outputs/qaac_api_bench/deepscaler_random300/variants.jsonl}"
export NUM_GPUS="${NUM_GPUS:-6}"
export PANDA_SKIP_TOKUR=1
export GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
export VLLM_CHUNK_SIZE="${VLLM_CHUNK_SIZE:-512}"
export PANDA_DYNAMIC_CLAIM="${PANDA_DYNAMIC_CLAIM:-1}"

exec bash "$ROOT/scripts/run_random300_6gpu.sh" "${1:-all}"
