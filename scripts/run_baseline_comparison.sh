#!/usr/bin/env bash
# Regenerate ISO baseline comparison tables for ASE outputs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/env.sh" 2>/dev/null || export PYTHONPATH="$ROOT/src"

OUT_DIR="${OUT_DIR:-$ROOT/outputs/panda_full}"
DATASETS="${DATASETS:-math500,minerva,gsm8k}"

IFS=',' read -ra DS_ARR <<< "$DATASETS"
for dataset in "${DS_ARR[@]}"; do
  dataset="$(echo "$dataset" | xargs)"
  [[ -z "$dataset" ]] && continue
  echo "=== baselines: $dataset ==="
  python3 -m panda.baselines.analyze_comparison --out-dir "$OUT_DIR" --dataset "$dataset"
done

echo "Done. See outputs/*/BASELINES_*.md"
