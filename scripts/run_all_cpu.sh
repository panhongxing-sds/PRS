#!/usr/bin/env bash
# Run ALL CPU-only paper analyses (no GPU, no HuggingFace download).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/env.sh"

export PANDA_OUTPUTS="${PANDA_OUTPUTS:-/root/autodl-tmp/panda-outputs}"
export PANDA_MODELS="${PANDA_MODELS:-/root/autodl-tmp/panda-models}"
LOG_DIR="$ROOT/paper/analysis/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/run_all_cpu_${STAMP}.log"

exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== PANDA CPU suite started $(date) ==="
echo "PANDA_OUTPUTS=$PANDA_OUTPUTS"

run_step() {
  local name="$1"
  shift
  echo ""
  echo ">>> [$name] $(date +%H:%M:%S)"
  local t0=$SECONDS
  "$@"
  echo "<<< [$name] done in $((SECONDS - t0))s"
}

# 1. Fast PANDA cache analyses (~5s)
run_step "panda_cpu" python3 "$ROOT/scripts/run_cpu_paper_analyses.py"

# 2. SC failure-mode analyses (~5min, data in repo)
run_step "sc_all_models" bash -c "cd $ROOT/experiments/spurious_consensus && python3 analyze_all_models.py"
run_step "sc_extras" bash -c "cd $ROOT/experiments/spurious_consensus && python3 analyze_cpu_extras.py"
run_step "sc_reasoning" bash -c "cd $ROOT/experiments/spurious_consensus && python3 analyze_scr_reasoning_t100.py"

# 3. Mechanism analysis (reads raw_runs, ~10-20min)
run_step "mechanism" python3 "$ROOT/scripts/analyze_mechanism.py" \
  --out-root "$PANDA_OUTPUTS" \
  --json-out "$ROOT/paper/tables/mechanism_results.json" \
  --md-out "$ROOT/paper/tables/table_mechanism.md"

# 4. Main-table aggregation from raw (~10-20min)
run_step "aggregate_panda_v2" python3 "$ROOT/scripts/aggregate_panda_v2.py" \
  --out-root "$PANDA_OUTPUTS" \
  --json-out "$ROOT/paper/maintable/panda_v2_results.json" \
  --workers 16

# 5. Summary manifest
cat > "$ROOT/paper/analysis/CPU_DONE.md" <<EOF
# CPU analyses complete

- **Finished:** $(date)
- **Log:** $MAIN_LOG

## Outputs

| Task | Output |
|------|--------|
| PANDA CPU | \`paper/analysis/cpu_results.{md,json}\` |
| Ablation | \`paper/maintable/ablation_macro.json\` |
| SC failure mode | \`experiments/spurious_consensus/CPU_ANALYSIS_RESULTS.md\` |
| SC results | \`experiments/spurious_consensus/results/*.json\` |
| SC figures | \`experiments/spurious_consensus/figures/*.png\` |
| Mechanism | \`paper/tables/mechanism_results.json\` |
| Main table | \`paper/maintable/panda_v2_results.json\` |

Next: \`bash scripts/run_gpu_minimal_plan.sh\`
EOF

echo ""
echo "=== ALL CPU DONE $(date) ==="
echo "Summary: paper/analysis/CPU_DONE.md"
echo "Log: $MAIN_LOG"
