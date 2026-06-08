#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# 官方 TokUR baseline（vLLM + greedy_unc_single_batch_refine）四模型队列。
# partial raw 视为满数据；默认强制重跑非 official 或过期的 baseline。
#
# Usage:
#   nohup bash scripts/launch_official_tokur_all.sh >> $PRS_OUTPUTS/strict_tokur_four.log 2>&1 &
#
# Env:
#   SKIP_IF_OFFICIAL=0   强制全部重跑（默认）
#   SKIP_IF_OFFICIAL=1   仅补缺失/非 official
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export SKIP_IF_OFFICIAL="${SKIP_IF_OFFICIAL:-0}"
export MIN_RAW="${MIN_RAW:-50}"
export PARALLEL_SHARDS="${PARALLEL_SHARDS:-0}"
export SEEDS="${SEEDS:-96}"

exec bash "$ROOT/scripts/queue_strict_tokur_four_models.sh"
