#!/usr/bin/env bash
# Official TokUR baseline for maintable runs (Wang-ML-Lab/TokUR greedy_unc + TFB).
#
# NOT the approximate post-hoc path (score_tokur_baseline.py).
#
# Requires: ASE raw_runs with question/reference (export_tokur_jsonl), TokUR venv with vLLM.
#
# Usage (called from run_maintable_vllm.sh):
#   OUT_DIR=$PANDA_OUTPUTS/maintable_qwen25_3b/seed41 \
#   PANDA_MODEL_TAG=qwen25_3b DATASET=math500 TOKUR_SEED=41 \
#   bash scripts/run_tokur_official_maintable.sh
#
# Env:
#   SKIP_TOKUR_GENERATE=1   convert existing pkl only
#   SKIP_TOKUR_EXPORT=1     skip jsonl export (pkl already built)
set -euo pipefail
source "$(dirname "$0")/env.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

OUT_DIR="${OUT_DIR:?OUT_DIR required}"
PANDA_MODEL_TAG="${PANDA_MODEL_TAG:?PANDA_MODEL_TAG required}"
DATASET="${DATASET:?DATASET required}"
TOKUR_SEED="${TOKUR_SEED:-96}"
SKIP_GENERATE="${SKIP_TOKUR_GENERATE:-0}"
SKIP_EXPORT="${SKIP_TOKUR_EXPORT:-0}"

TOKUR_VENV="${TOKUR_VENV:-${PANDA_ROOT}/.tokur_venv}"
TOKUR_PY="${TOKUR_PY:-${TOKUR_VENV}/bin/python}"

if [[ ! -x "$TOKUR_PY" ]]; then
  echo "Missing TokUR venv: $TOKUR_PY (run bash scripts/setup_after_clone.sh)" >&2
  exit 1
fi

panda_to_tokur_tag() {
  case "$1" in
    qwen25_3b) echo qwen3b ;;
    llama32_1b) echo llama1b ;;
    llama31_8b) echo llama8b ;;
    qwen3_8b) echo qwen8b ;;
    qwen3b|llama1b|llama8b|qwen8b) echo "$1" ;;
    *) echo "Unknown PANDA_MODEL_TAG=$1" >&2; return 1 ;;
  esac
}

MODEL_TAG="$(panda_to_tokur_tag "$PANDA_MODEL_TAG")"
export OUT_DIR MODEL_TAG DATASETS="$DATASET" SEEDS="$TOKUR_SEED"
export SKIP_GENERATE SKIP_EXPORT
export TOKUR_VENV TOKUR_PY
export PATH="${TOKUR_VENV}/bin:${PATH}"
export TMPDIR="${TMPDIR:-/tmp}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

# maintable out dirs never use legacy panda_{dataset} slug
export TOKUR_DS_LEGACY=0

exec bash "$ROOT/scripts/run_tokur_strict_baseline.sh"
