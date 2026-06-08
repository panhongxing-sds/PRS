#!/usr/bin/env bash
# Download MATH500, GSM8K, DeepScaleR to third_party/TokUR/datasets (HF mirror)
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TOKUR_ROOT/datasets"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
python download_data.py
echo "Datasets:"
wc -l *.jsonl 2>/dev/null || true
