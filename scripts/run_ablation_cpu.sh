#!/usr/bin/env bash
# CPU-only PANDA ablation tables from existing raw_runs (no GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/env.sh" 2>/dev/null || export PYTHONPATH="$ROOT/src"

OUT_DIR="${OUT_DIR:-$PANDA_OUTPUTS:-$ROOT/outputs/panda_full}"
DATASETS="${DATASETS:-minerva,math500,gsm8k,deepscaler}"

echo "=== Table A: component ablation ==="
python3 -m panda.core.ablation_recompute \
  --out-dir "$OUT_DIR" \
  --datasets "$DATASETS" \
  --output "$OUT_DIR/ABLATION_component.md"

echo "=== Table B: perturbation budget sweep ==="
python3 -m panda.core.ablation_recompute \
  --out-dir "$OUT_DIR" \
  --datasets "$DATASETS" \
  --budget-sweep \
  --output "$OUT_DIR/ABLATION_budget_sweep.json"

echo "Done: $OUT_DIR/ABLATION_component.md"
