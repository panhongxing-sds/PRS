#!/usr/bin/env bash
# After git clone: install Python deps + optional TokUR vLLM venv.
set -euo pipefail
source "$(dirname "$0")/activate_env.sh"
cd "$PANDA_ROOT"

echo "PANDA_ROOT=$PANDA_ROOT"
pip install -e ".[dev]" --no-deps
pip install pyyaml tqdm scikit-learn pytest safetensors

if [[ ! -x "${TOKUR_VENV:-/HDDDATA/phx/tokur_venv}/bin/python" ]]; then
  echo "Creating TokUR venv at ${TOKUR_VENV:-$PANDA_ROOT/.tokur_venv} ..."
  VENV="${TOKUR_VENV:-$PANDA_ROOT/.tokur_venv}"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -U pip
  if [[ -f third_party/TokUR/requirements.txt ]]; then
    "$VENV/bin/pip" install -r third_party/TokUR/requirements.txt
  fi
  echo "Set: export TOKUR_VENV=$VENV"
fi

echo "Ready. Example:"
echo "  bash scripts/fast_four_model_tables.sh"
echo "  bash scripts/run_panda_model_pipeline.sh qwen25_3b \$PANDA_MODELS/TFB-Qwen2.5-3B-Instruct"
