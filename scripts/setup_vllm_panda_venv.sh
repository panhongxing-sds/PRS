#!/usr/bin/env bash
# PANDA Phase-A vLLM venv (standard pip vllm — NOT the TokUR fork).
# Use for SE / clean+R batched generation on 4090/A100 (or 5090 if wheel supports sm_120).
set -euo pipefail
source "$(dirname "$0")/env.sh"
VENV="${PANDA_VLLM_VENV:-$PANDA_ROOT/.vllm_venv}"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

echo "=== PANDA vLLM venv: $VENV ==="
if [[ ! -x "$PY" ]]; then
  python3 -m venv "$VENV"
  "$PIP" install -U pip setuptools wheel
fi

# Match main PANDA torch CUDA when possible; vllm pulls its own torch if needed.
"$PIP" install "vllm>=0.6.0" "numpy>=1.24" "transformers>=4.40" sentencepiece
"$PIP" install -e "$PANDA_ROOT" --no-deps
"$PIP" install pyyaml tqdm scikit-learn safetensors sympy

echo "=== verify ==="
"$PY" -c "import vllm, torch; print('OK vllm', vllm.__version__, 'torch', torch.__version__)"
echo "export PANDA_VLLM_VENV=$VENV"
echo "export VLLM_PY=$PY"
