#!/usr/bin/env bash
# PRS 环境激活：复用集群 vLLM conda 包 + 本仓库 PYTHONPATH
# 用法: source scripts/activate_env.sh
set -euo pipefail

_env_file="${BASH_SOURCE[0]:-$0}"
PRS_ROOT="$(cd "$(dirname "$_env_file")/.." && pwd)"

VLLM_ENV="${VLLM_ENV:-/tmp/vllm-cu124}"
if [[ -f "${VLLM_ENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VLLM_ENV}/bin/activate"
else
  echo "[warn] vLLM env not found at ${VLLM_ENV}; using current python" >&2
fi

# shellcheck disable=SC1091
source "${PRS_ROOT}/scripts/env.sh"

# 可选：启用代理（下载 HuggingFace / ModelScope）
if [[ -f /mnt/afs/L202500372/proxy/proxy-env.sh ]]; then
  # shellcheck disable=SC1091
  source /mnt/afs/L202500372/proxy/proxy-env.sh
fi

export TOKUR_VENV="${TOKUR_VENV:-${PRS_ROOT}/.tokur_venv}"

echo "PRS env ready: PRS_ROOT=${PRS_ROOT}"
echo "  python: $(which python)"
echo "  PRS_MODELS=${PRS_MODELS}"
echo "  PRS_OUTPUTS=${PRS_OUTPUTS}"
