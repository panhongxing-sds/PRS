#!/usr/bin/env bash
# 启动/重启单个模型采样（默认 vLLM；Gemma 用 HF）
set -euo pipefail
cd "$(dirname "$0")"
export PRS_ROOT="${PRS_ROOT:-/root/PRS}"
export PRS_MODELS="${PRS_MODELS:-/root/autodl-tmp/prs-models}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU="${1:?gpu}"
TAG="${2:?tag}"
PATH_M="${3:?model_path}"
K_CHUNK="${4:-16}"
P_BATCH="${5:-2}"
BACKEND="${6:-}"
if [[ -z "$BACKEND" ]]; then
  if [[ "$TAG" == gemma3_4b ]]; then BACKEND=hf; else BACKEND=vllm; fi
fi
LOG_DIR="${LOG_DIR:-logs/phase1}"
mkdir -p "$LOG_DIR"

if pgrep -f "MODEL_TAG=${TAG} MODEL_PATH=${PATH_M}" >/dev/null 2>&1; then
  echo "[skip] ${TAG} 已在运行"
  exit 0
fi

echo "[$(date -Iseconds)] 启动 ${TAG} @ GPU${GPU} backend=${BACKEND} K_CHUNK=${K_CHUNK} PROMPT_BATCH=${P_BATCH}"
nohup env GPU="$GPU" MODEL_TAG="$TAG" MODEL_PATH="$PATH_M" BACKEND="$BACKEND" \
  K_CHUNK="$K_CHUNK" PROMPT_BATCH="$P_BATCH" \
  ./run_sampling.sh >> "$LOG_DIR/${TAG}.log" 2>&1 &
echo $! > "$LOG_DIR/${TAG}.pid"
