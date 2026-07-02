#!/usr/bin/env bash
# Source: source "$(dirname "$0")/env.sh"
_env_file="${BASH_SOURCE[0]:-$0}"
PANDA_ROOT="$(cd "$(dirname "$_env_file")/.." && pwd)"
export PANDA_ROOT
export PYTHONPATH="${PANDA_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
# HuggingFace mirror (autodl / CN networks)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export PANDA_OUTPUTS="${PANDA_OUTPUTS:-${PANDA_ROOT}/outputs}"
export PANDA_MODELS="${PANDA_MODELS:-${PANDA_ROOT}/models}"
# Prefer shared model cache on this machine when present.
if [[ -d /root/autodl-tmp/panda-models ]] && [[ ! -d "${PANDA_MODELS}/TFB-Qwen2.5-3B-Instruct" ]]; then
  export PANDA_MODELS="/root/autodl-tmp/panda-models"
fi
export TOKUR_ROOT="${TOKUR_ROOT:-${PANDA_ROOT}/third_party/TokUR}"
export TOKUR_VENV="${TOKUR_VENV:-${PANDA_ROOT}/.tokur_venv}"
export PANDA_VLLM_VENV="${PANDA_VLLM_VENV:-${PANDA_ROOT}/.vllm_venv}"
export VLLM_PY="${VLLM_PY:-${PANDA_VLLM_VENV}/bin/python}"
export MODEL_BASE_DIR="${MODEL_BASE_DIR:-${PANDA_MODELS}}"
