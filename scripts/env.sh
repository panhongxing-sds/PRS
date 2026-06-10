#!/usr/bin/env bash
# Source: source "$(dirname "$0")/env.sh"
_env_file="${BASH_SOURCE[0]:-$0}"
PRS_ROOT="$(cd "$(dirname "$_env_file")/.." && pwd)"
export PRS_ROOT
export PYTHONPATH="${PRS_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PRS_OUTPUTS="${PRS_OUTPUTS:-${PRS_ROOT}/outputs}"
export PRS_MODELS="${PRS_MODELS:-${PRS_ROOT}/models}"
export TOKUR_ROOT="${TOKUR_ROOT:-${PRS_ROOT}/third_party/TokUR}"
export TOKUR_VENV="${TOKUR_VENV:-${PRS_ROOT}/.tokur_venv}"
export MODEL_BASE_DIR="${MODEL_BASE_DIR:-${PRS_MODELS}}"
