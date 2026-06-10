#!/usr/bin/env bash
# 下载逻辑 + 代码 benchmark（扩展 TokUR datasets/）
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TOKUR_ROOT/datasets"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
python download_data.py
echo ""
echo "=== Dataset status ==="
for f in leg-counting.jsonl zebra_puzzles.jsonl color_cube.jsonl humaneval.jsonl; do
  if [[ -f "$f" ]]; then
    echo "  OK  $f ($(wc -l <"$f") lines)"
  else
    echo "  MISSING  $f  (run with HF token or manual download)"
  fi
done
