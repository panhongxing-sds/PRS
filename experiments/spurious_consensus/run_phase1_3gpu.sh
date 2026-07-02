#!/usr/bin/env bash
# 三卡并行：Llama/Phi 用 vLLM，Gemma 用 HF
set -euo pipefail
cd "$(dirname "$0")"
export PANDA_ROOT="${PANDA_ROOT:-/root/PANDA}"
export PANDA_MODELS="${PANDA_MODELS:-/root/autodl-tmp/panda-models}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LOG_DIR="${LOG_DIR:-logs/phase1}"
mkdir -p "$LOG_DIR"

launch() {
  local gpu="$1" tag="$2" path="$3" backend="$4" kchunk="$5" pbatch="$6"
  local log="$LOG_DIR/${tag}.log"
  echo "[GPU $gpu] $tag backend=$backend K_CHUNK=$kchunk PROMPT_BATCH=$pbatch"
  nohup env GPU="$gpu" MODEL_TAG="$tag" MODEL_PATH="$path" BACKEND="$backend" \
    K_CHUNK="$kchunk" PROMPT_BATCH="$pbatch" \
    ./run_sampling.sh >> "$log" 2>&1 &
  echo $! > "$LOG_DIR/${tag}.pid"
}

launch 0 llama32_1b "$PANDA_MODELS/TFB-Llama-3.2-1B-Instruct" vllm 16 8
launch 1 phi4_mini   "$PANDA_MODELS/Phi-4-mini-instruct"      vllm 16 4
launch 2 gemma3_4b   "$PANDA_MODELS/gemma-3-4b-it"          hf   8  2

echo "已启动 vLLM×2 + HF×1。tail -f logs/phase1/*.log"
