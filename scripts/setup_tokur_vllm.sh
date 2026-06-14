#!/usr/bin/env bash
# TokUR Phase C venv for RTX 4090 / A100 (precompiled vLLM wheel path).
# See paper/EXPERIMENT_PLAN.md §14.4
set -euo pipefail
source "$(dirname "$0")/env.sh"
VENV="${TOKUR_VENV:-$PRS_ROOT/.tokur_venv}"
VLLM_COMMIT="${VLLM_COMMIT:-61c6a5a79664882a8ab1c9af3ff78677911516dc}"
VLLM_SRC="${VLLM_SRC:-/tmp/vllm-tokur}"
WHEEL="https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

echo "=== TokUR venv: $VENV ==="
python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip setuptools wheel

if [[ ! -d "$VLLM_SRC/.git" ]]; then
  git clone https://github.com/haizhou-shi/vllm.git "$VLLM_SRC"
fi
cd "$VLLM_SRC"
git fetch origin "$VLLM_COMMIT" 2>/dev/null || true
git checkout "$VLLM_COMMIT"
export VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL"
"$VENV/bin/pip" install -e .

"$VENV/bin/pip" install "transformers==4.53.3" "numpy>=1.19.2,<2"
"$VENV/bin/pip" install -e "$PRS_ROOT/third_party/TokUR/bayesian_transformer"
"$VENV/bin/pip" install -r "$PRS_ROOT/third_party/TokUR/requirements.txt" 2>/dev/null || true
"$VENV/bin/pip" install pyyaml tqdm scikit-learn sympy click latex2sympy2 regex word2number safetensors datasets pandas

echo "=== verify ==="
"$VENV/bin/python" -c "import vllm, bayesian_transformer, torch; print('OK', vllm.__version__, torch.__version__)"
echo "Done. export TOKUR_VENV=$VENV && bash scripts/smoke_tokur_vllm.sh"
