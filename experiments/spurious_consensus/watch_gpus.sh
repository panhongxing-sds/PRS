#!/usr/bin/env bash
# 看门狗：检测 GPU 空闲 / 进程挂掉，自动重启
set -euo pipefail
cd "$(dirname "$0")"
export PRS_ROOT="${PRS_ROOT:-/root/PRS}"
export PRS_MODELS="${PRS_MODELS:-/root/autodl-tmp/prs-models}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LOG_DIR="${LOG_DIR:-logs/phase1}"
INTERVAL="${WATCH_INTERVAL:-60}"

# tag|path|k_chunk|prompt_batch|backend
declare -A MODELS=(
  [0]="llama32_1b|$PRS_MODELS/TFB-Llama-3.2-1B-Instruct|16|8|vllm"
  [1]="phi4_mini|$PRS_MODELS/Phi-4-mini-instruct|16|4|vllm"
  [2]="gemma3_4b|$PRS_MODELS/gemma-3-4b-it|8|2|hf"
)

gpu_busy() {
  local gpu="$1"
  local util mem
  util=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  mem=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' MiB')
  [[ "${util:-0}" -ge 30 ]] || [[ "${mem:-0}" -ge 8000 ]]
}

model_running() {
  local tag="$1"
  local pidf="$LOG_DIR/${tag}.pid"
  [[ -f "$pidf" ]] || return 1
  local pid
  pid=$(cat "$pidf")
  kill -0 "$pid" 2>/dev/null || return 1
  pgrep -P "$pid" >/dev/null 2>&1 || pgrep -f "MODEL_TAG=${tag}" >/dev/null 2>&1
}

while true; do
  for gpu in 0 1 2; do
    IFS='|' read -r tag path kc pb backend <<< "${MODELS[$gpu]}"
    if model_running "$tag" && gpu_busy "$gpu"; then
      continue
    fi
    echo "[watch $(date -Iseconds)] GPU${gpu} ${tag} 需重启 (running=$(model_running "$tag" && echo yes || echo no) busy=$(gpu_busy "$gpu" && echo yes || echo no))"
    pkill -f "MODEL_TAG=${tag}" 2>/dev/null || true
    sleep 2
    ./launch_model.sh "$gpu" "$tag" "$path" "$kc" "$pb" "$backend"
  done
  sleep "$INTERVAL"
done
