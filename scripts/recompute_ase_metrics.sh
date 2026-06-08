#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# Recompute ASE/ATU from saved cache (CPU only, seconds).
# Usage: bash scripts/recompute_ase_metrics.sh [OUT_DIR] [ATU_TOP_PCT]
set -euo pipefail
OUT="${1:-/home/phx/PRS/outputs/ase_v2}"
PCT="${2:-0.10}"
python3 -m prs.ase.recompute_metrics --out-dir "$OUT" --datasets minerva,math500 --atu-top-pct "$PCT"
python3 -m prs.ase.analyze_ase --out-dir "$OUT" --datasets minerva,math500
python3 -m prs.ase.analyze_atu_extended --out-dir "$OUT" --datasets minerva,math500
echo "Done. See $OUT/FULL_ANALYSIS.md and $OUT/ATU_EXTENDED_ANALYSIS.md"
