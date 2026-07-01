#!/usr/bin/env bash
# Poll until 4×3×3 math raw_runs complete, then aggregate PRS v2 + refresh EXPERIMENT_PLAN.
set -uo pipefail

ROOT=/root/autodl-tmp/PRS
export PRS_OUTPUTS=/root/autodl-tmp/prs-outputs
export PYTHONPATH=$ROOT/src
LOG=/root/autodl-tmp/logs/wait_aggregate_prs_v2.log
STAMP=/root/autodl-tmp/prs-outputs/.prs_v2_aggregated_complete
INTERVAL="${INTERVAL:-300}"  # seconds between checks

mkdir -p /root/autodl-tmp/logs
exec >>"$LOG" 2>&1

echo "========== wait_and_aggregate_prs_v2 START $(date) =========="
echo "poll_interval=${INTERVAL}s  stamp=$STAMP"

while true; do
  if bash "$ROOT/scripts/check_math_complete.sh"; then
    echo "All math datasets complete at $(date)"
    break
  fi
  running=$(pgrep -f 'run_ase_experiment.*qwen3_8b' | wc -l)
  echo "--- poll $(date) decode_procs=$running ---"
  bash "$ROOT/scripts/check_math_complete.sh" || true
  if [[ "$running" -eq 0 ]]; then
    echo "WARN: no qwen3 decode procs but data incomplete — still waiting"
  fi
  sleep "$INTERVAL"
done

# --- SE baselines (vLLM high-temp + NLI): SE / U_Ecc / U_Deg ---
export PRS_OUTPUTS="${PRS_OUTPUTS:-/root/autodl-tmp/prs-outputs}"
export PRS_MODELS="${PRS_MODELS:-/root/autodl-tmp/prs-models}"
# SE vLLM backfill not deployed on this machine — skip by default (set SKIP_SE_VLLM=0 to enable).
if [[ "${SKIP_SE_VLLM:-1}" == "0" ]]; then
  echo "Running SE vLLM+NLI queue $(date)"
  WAIT_DECODE=0 bash "$ROOT/scripts/queue_se_vllm_math.sh" \
    >>/root/autodl-tmp/logs/queue_se_vllm.log 2>&1 \
    || echo "WARN: SE queue failed $(date)"
else
  echo "SKIP_SE_VLLM=1 — skipping SE backfill $(date)"
fi

echo "Running aggregate_prs_v2.py $(date)"
python3 "$ROOT/scripts/aggregate_prs_v2.py" \
  --out-root "$PRS_OUTPUTS" \
  --json-out "$ROOT/paper/maintable/prs_v2_results.json"

echo "Updating EXPERIMENT_PLAN.md $(date)"
python3 "$ROOT/scripts/update_experiment_plan_prs_v2.py"

# optional legacy maintble per-model md
for mt in qwen25_3b llama32_1b llama31_8b qwen3_8b; do
  bash "$ROOT/scripts/aggregate_maintable.sh" "$mt" 2>&1 || echo "legacy aggregate $mt failed"
done

date '+%F %T' >"$STAMP"
echo "========== DONE $(date) stamp=$STAMP =========="
