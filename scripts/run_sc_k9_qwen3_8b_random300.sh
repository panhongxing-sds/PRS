#!/usr/bin/env bash
# Option A: GPU K=9 on deepscaler_random300 (300 ids), T=0.5 top_p=0.95 seed=41.
# No K=64 pool exists for Qwen3-8B deepscaler — fresh fair-9 decode (SC protocol).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/env.sh"
cd "$ROOT/experiments/spurious_consensus"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=0
export PANDA_MODELS="${PANDA_MODELS:-/root/autodl-tmp/panda-models}"
export FORCE_VLLM=1

NUM_GPUS="${NUM_GPUS:-6}"
CFG="${CFG:-config_qwen3_8b.yaml}"
OUT_FINAL="${OUT_FINAL:-$ROOT/experiments/spurious_consensus/data/samples/samples_qwen3_8b_seed41_deepscaler_random300_k9.jsonl}"
LOG="${LOG:-/root/autodl-tmp/prs-outputs/maintable_qwen3_8b_deepscaler_random300/logs/sc_k9_random300.log}"
mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT_FINAL")"

if [[ -f "$OUT_FINAL" ]] && [[ "$(wc -l < "$OUT_FINAL" | tr -d ' ')" -ge 300 ]]; then
  echo "[skip] SC@9 already complete: $OUT_FINAL ($(wc -l < "$OUT_FINAL") lines)"
  exit 0
fi

echo "=== SC@9 Qwen3-8B random300 K=9 GPU $(date -Iseconds) ===" | tee -a "$LOG"
PIDS=()
for shard in $(seq 0 $((NUM_GPUS - 1))); do
  slog="${LOG%.log}.shard${shard}.log"
  CUDA_VISIBLE_DEVICES=$shard /root/miniconda3/bin/python -u sample_vllm.py \
    --config "$CFG" \
    --benchmark deepscaler_random300 \
    --k 9 --temp 0.5 --top-p 0.95 --seed 41 \
    --max-tokens 2048 \
    --batch-size 4 \
    --shard-id "$shard" --num-shards "$NUM_GPUS" \
    --resume \
    >> "$slog" 2>&1 &
  PIDS+=($!)
  echo "SC_K9 shard=$shard PID=${PIDS[$(( ${#PIDS[@]} - 1 ))]}" | tee -a "$LOG"
done
fail=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || fail=1
done
if [[ "$fail" -ne 0 ]]; then
  echo "ERROR: one or more SC@9 shards failed — see ${LOG%.log}.shard*.log" | tee -a "$LOG"
  exit 1
fi

SAMPLE_DIR="$ROOT/experiments/spurious_consensus/data/samples"
TMP_MERGE="${OUT_FINAL}.merge_tmp"
: > "$TMP_MERGE"
for shard in $(seq 0 $((NUM_GPUS - 1))); do
  shard_file="$SAMPLE_DIR/samples_qwen3_8b_seed41_deepscaler_random300.shard${shard}.jsonl"
  [[ -f "$shard_file" ]] || { echo "missing $shard_file" >&2; exit 1; }
  cat "$shard_file" >> "$TMP_MERGE"
done
mv "$TMP_MERGE" "$OUT_FINAL"
lines=$(wc -l < "$OUT_FINAL" | tr -d ' ')
echo "Merged → $OUT_FINAL ($lines lines)" | tee -a "$LOG"

python3 - "$OUT_FINAL" << 'PY' | tee -a "$LOG"
import json, sys
from collections import Counter
from pathlib import Path
p = Path(sys.argv[1])
recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print(f"n={len(recs)}")
PY
