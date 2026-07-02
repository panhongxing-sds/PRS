#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Run recompute_metrics for multiple datasets in parallel (CPU-only).
#
# Usage:
#   bash scripts/recompute_metrics_parallel.sh OUT_DIR [dataset1,dataset2,...]
#   MAX_PARALLEL=4 bash scripts/recompute_metrics_parallel.sh /path/out minerva,math500,gsm8k,deepscaler
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="${1:?OUT_DIR required}"
shift || true
DATASETS_CSV="${1:-minerva,math500,gsm8k,deepscaler}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

IFS=',' read -ra DATASETS <<< "$DATASETS_CSV"
PIDS=()
RUNNING=0

for ds in "${DATASETS[@]}"; do
  ds="${ds// /}"
  [[ -n "$ds" ]] || continue
  while [[ "$RUNNING" -ge "$MAX_PARALLEL" ]]; do
    for i in "${!PIDS[@]}"; do
      if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
        wait "${PIDS[$i]}" || true
        unset 'PIDS[i]'
        RUNNING=$((RUNNING - 1))
      fi
    done
    PIDS=("${PIDS[@]+"${PIDS[@]}"}")
    [[ "$RUNNING" -lt "$MAX_PARALLEL" ]] || sleep 2
  done
  echo "[recompute_parallel] start $ds $(date)"
  python3 -m panda.core.recompute_metrics --out-dir "$OUT_DIR" --datasets "$ds" &
  PIDS+=($!)
  RUNNING=$((RUNNING + 1))
done

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=$((FAIL + 1))
done
echo "[recompute_parallel] done fail=$FAIL $(date)"
exit "$FAIL"
