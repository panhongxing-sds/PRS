#!/usr/bin/env bash
# Wait for ASE decode to finish, then run official SE + U_Ecc + U_Deg (vLLM backfill + NLI)
# for all four main-table models on math datasets (minerva, math500, gsm8k × seeds 41–43).
#
# Usage:
#   nohup bash scripts/queue_se_vllm_math.sh >> /root/autodl-tmp/logs/queue_se_vllm.log 2>&1 &
#
# Env:
#   WAIT_DECODE=1          — wait until check_math_complete + no run_panda_experiment (default 1)
#   MODELS="qwen3_8b llama31_8b"  — subset
#   NGPU=4 SE_CHUNK_8B=32 SE_CHUNK_SMALL=64
#   SKIP_VLLM=1 / SKIP_NLI=1 — partial rerun
set -euo pipefail
source "$(dirname "$0")/env.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/panda_gpu_lock.sh
source "$ROOT/scripts/panda_gpu_lock.sh"

export PANDA_OUTPUTS="${PANDA_OUTPUTS:-/root/autodl-tmp/panda-outputs}"
export PANDA_MODELS="${PANDA_MODELS:-/root/autodl-tmp/panda-models}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

LOG="${LOG:-/root/autodl-tmp/logs/queue_se_vllm.log}"
STAMP="${STAMP:-$PANDA_OUTPUTS/.se_vllm_math_complete}"
WAIT_DECODE="${WAIT_DECODE:-1}"
NGPU="${NGPU:-4}"
SE_CHUNK_8B="${SE_CHUNK_8B:-32}"
SE_CHUNK_SMALL="${SE_CHUNK_SMALL:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MODELS="${MODELS:-qwen3_8b llama31_8b qwen25_3b llama32_1b}"
DATASETS="${DATASETS:-minerva,math500,gsm8k}"
SEEDS="${SEEDS:-41,42,43}"

mkdir -p "$(dirname "$LOG")" "$PANDA_OUTPUTS"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_model() {
  local model=$1
  local chunk=$2
  log "=== SE vLLM+NLI: $model NGPU=$NGPU chunk=$chunk ==="
  acquire_gpu se_vllm "SE backfill $model"
  NGPU="$NGPU" SE_CHUNK="$chunk" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    DATASETS="$DATASETS" SEEDS="$SEEDS" \
    bash "$ROOT/scripts/run_se_vllm_nli.sh" "$model" >>"$LOG" 2>&1 \
    || { log "WARN: SE pipeline failed for $model (continuing)"; release_gpu se_vllm; return 1; }
  release_gpu se_vllm
  log "=== done $model ==="
}

log "========== queue_se_vllm_math START =========="
log "MODELS=$MODELS NGPU=$NGPU WAIT_DECODE=$WAIT_DECODE"

# --- vLLM venv must exist ---
PANDA_VLLM_VENV="${PANDA_VLLM_VENV:-$PANDA_ROOT/.vllm_venv}"
VLLM_PY="${VLLM_PY:-$PANDA_VLLM_VENV/bin/python}"
if [[ ! -x "$VLLM_PY" ]] || ! "$VLLM_PY" -c "import vllm" 2>/dev/null; then
  log "Setting up PANDA vLLM venv → $PANDA_VLLM_VENV"
  bash "$ROOT/scripts/setup_vllm_panda_venv.sh" >>"$LOG" 2>&1
fi
export PANDA_VLLM_VENV VLLM_PY
"$VLLM_PY" -c "import vllm, torch; print('vLLM OK', vllm.__version__, 'gpus', torch.cuda.device_count())" \
  | tee -a "$LOG"

if [[ "$WAIT_DECODE" == "1" ]]; then
  log "Waiting for math decode complete + GPU idle..."
  while ! bash "$ROOT/scripts/check_math_complete.sh" 2>/dev/null; do
    n=$(pgrep -cf 'run_panda_experiment' || true)
    log "  decode incomplete (run_panda_procs=$n)"
    sleep 120
  done
  wait_for_gpus_compatible
  log "Decode complete, GPUs free."
fi

for model in $MODELS; do
  case "$model" in
    qwen3_8b|llama31_8b) chunk="$SE_CHUNK_8B" ;;
    *) chunk="$SE_CHUNK_SMALL" ;;
  esac
  run_model "$model" "$chunk" || true
done

date '+%F %T' >"$STAMP"
log "========== queue_se_vllm_math DONE stamp=$STAMP =========="
