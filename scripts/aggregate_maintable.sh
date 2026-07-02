#!/usr/bin/env bash
# CPU-only：3-seed 聚合主表 (markdown + latex)
#
# Usage:
#   bash scripts/aggregate_maintable.sh llama32_1b
#   DATASETS=leg_counting,zebra_puzzles,color_cube,humaneval bash scripts/aggregate_maintable.sh llama32_1b
set -euo pipefail
source "$(dirname "$0")/env.sh"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL_TAG="${1:?model_tag}"
SEEDS="${SEEDS:-41,42,43}"
DATASETS="${DATASETS:-minerva,math500,gsm8k,deepscaler,leg_counting,zebra_puzzles,color_cube,humaneval}"
OUT_BASE="${OUT_BASE:-$PANDA_OUTPUTS/maintable_${MODEL_TAG}}"
PAPER_DIR="${PAPER_DIR:-$ROOT/paper/maintable/${MODEL_TAG}}"

declare -A MODEL_LABEL=(
  [qwen25_3b]="Qwen2.5-3B-Instruct"
  [llama32_1b]="Llama-3.2-1B-Instruct"
  [llama31_8b]="Llama-3.1-8B-Instruct"
  [qwen3_8b]="Qwen3-8B"
)

LABEL="${MODEL_LABEL[$MODEL_TAG]:-$MODEL_TAG}"
mkdir -p "$PAPER_DIR"

for seed in $(echo "$SEEDS" | tr ',' ' '); do
  OUT="$OUT_BASE/seed${seed}"
  if [[ -d "$OUT" ]]; then
    python3 -m panda.core.recompute_metrics \
      --out-dir "$OUT" \
      --datasets "$DATASETS" 2>/dev/null || true
  fi
done

python3 -m panda.core.analyze_maintable \
  --out-dir "$OUT_BASE" \
  --paper-dir "$PAPER_DIR" \
  --model-label "$LABEL" \
  --datasets "$DATASETS" \
  --seeds "$SEEDS" \
  --features-only \
  ${ISO_COL:+--iso-col}

echo "Wrote $PAPER_DIR/maintable.{md,tex}"
