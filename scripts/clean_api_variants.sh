#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m prs.ase.clean_variants "$@"
