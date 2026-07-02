#!/usr/bin/env bash
# random300 PANDA fair-budget (SC@9): Phase A vLLM + Phase B HF ASE on 6 GPUs.
# Fair budget: SE_SAMPLES=0, N_REPHRASES=4, WEIGHT_SEEDS=42,43,44,45 → 9 decodes/q.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/env.sh"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=0
export PANDA_SKIP_TOKUR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="${TMPDIR:-/tmp/panda_tmp}"
mkdir -p "$TMPDIR"
export TORCH_ALLOW_TF32=1
# Phase A (vLLM): higher batch util on 5090; Phase B (HF ASE) uses ~8–9GiB/GPU — low SM% is expected (serial decodes + CPU token_trace).
VLLM_GPU_MEM="${GPU_MEMORY_UTIL:-0.90}"
VLLM_CHUNK="${VLLM_CHUNK_SIZE:-512}"
PANDA_ATTN="${PANDA_ATTN_IMPLEMENTATION:-sdpa}"
PANDA_DYNAMIC_CLAIM="${PANDA_DYNAMIC_CLAIM:-0}"
WEIGHT_SIGMA="${WEIGHT_SIGMA:-0.03}"


NUM_GPUS="${NUM_GPUS:-6}"
OUT_BASE="${OUT_BASE:-/root/autodl-tmp/panda-outputs/maintable_qwen25_7b_deepscaler_random300}"
OUT="${OUT_DIR:-$OUT_BASE/seed41}"
VARIANTS="${VARIANTS:-/root/autodl-tmp/panda-outputs/qaac_api_bench/deepscaler_random300/variants.jsonl}"
MODEL="${MODEL_PATH:-/root/autodl-tmp/panda-models/TFB-Qwen2.5-7B-Instruct}"
VLLM_MODEL="${VLLM_MODEL_PATH:-/root/autodl-tmp/panda-models/.vllm_ready/TFB-Qwen2.5-7B-Instruct}"
VLLM_PY="${VLLM_PYTHON:-/root/miniconda3/bin/python}"
TFTTCL="${TFTTCL_ROOT:-/home/phx/TF-TTCL/data/MATH}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
LOG="$OUT_BASE/logs"
mkdir -p "$LOG" "$OUT/deepscaler/raw_runs"

PHASE="${1:-all}"  # all | phase_a | phase_b

scr500_guard() {
  pkill -9 -f run_sc_k9_scr500_watchdog 2>/dev/null || true
  pkill -9 -f 'sample_vllm.py.*deepscaler_scr500' 2>/dev/null || true
  pkill -9 -f maintable_qwen25_7b_deepscaler_scr500 2>/dev/null || true
}

stop_single_gpu_random300() {
  pkill -9 -f "$OUT_BASE/logs/orchestrate_ab.sh" 2>/dev/null || true
  pkill -9 -f "$OUT_BASE/logs/launch_phase_a.sh" 2>/dev/null || true
  pkill -9 -f "$OUT_BASE/logs/phase_b_guard.sh" 2>/dev/null || true
  # Single-GPU vLLM/ASE without shard-id (avoid duplicate work with 6-way launch)
  pgrep -af 'run_vllm_phase.*deepscaler_random300' 2>/dev/null | grep -v 'shard-id' | awk '{print $1}' | xargs -r kill -TERM 2>/dev/null || true
  pgrep -af 'run_panda_experiment.*deepscaler' 2>/dev/null | grep -v 'shard-id' | awk '{print $1}' | xargs -r kill -TERM 2>/dev/null || true
  sleep 2
}

launch_phase_a_shards() {
  scr500_guard
  local PIDS=()
  echo "=== Phase A 6-GPU launch $(date -Iseconds) ===" >> "$LOG/random300_6gpu_orchestrator.log"
  for shard in $(seq 0 $((NUM_GPUS - 1))); do
    local slog="$LOG/vllm_deepscaler_random300_shard${shard}.log"
    echo "[gpu $shard] Phase A shard $shard/$NUM_GPUS → $slog" >> "$LOG/random300_6gpu_orchestrator.log"
    CUDA_VISIBLE_DEVICES=$shard env TMPDIR="$TMPDIR" "$VLLM_PY" -u -m panda.core.run_vllm_phase \
      --dataset deepscaler \
      --max-samples "$MAX_SAMPLES" \
      --n-rephrases 4 \
      --se-samples 0 \
      --weight-seeds 42,43,44,45 \
      --weight-sigma "$WEIGHT_SIGMA" --weight-rank 4 \
      --max-new-tokens 2048 \
      --topk-save 10 \
      --gpu-memory-utilization "$VLLM_GPU_MEM" \
      --chunk-size "$VLLM_CHUNK" \
      --max-model-len 8192 \
      --model-path "$VLLM_MODEL" \
      --out-dir "$OUT" \
      --variants-path "$VARIANTS" \
      --tfttcl-root "$TFTTCL" \
      --shard-id "$shard" \
      --num-shards "$NUM_GPUS" \
      --resume \
      >> "$slog" 2>&1 &
    PIDS+=($!)
    echo "PHASE_A shard=$shard PID=${PIDS[$(( ${#PIDS[@]} - 1 ))]}" >> "$LOG/random300_6gpu_pids.txt"
  done
  printf '%s
' "${PIDS[@]}"
}

wait_phase_a() {
  local fail=0
  for shard in $(seq 0 $((NUM_GPUS - 1))); do
    local slog="$LOG/vllm_deepscaler_random300_shard${shard}.log"
    while ! grep -q 'complete. written=' "$slog" 2>/dev/null; do
      if ! pgrep -f "run_vllm_phase.*deepscaler.*--shard-id $shard " >/dev/null 2>&1; then
        echo "WARN: Phase A shard $shard not running and not complete — check $slog" | tee -a "$LOG/random300_6gpu_orchestrator.log"
        fail=1
        break
      fi
      sleep 30
    done
  done
  return $fail
}

launch_phase_b_shards() {
  scr500_guard
  local PIDS=()
  echo "=== Phase B 6-GPU launch $(date -Iseconds) ===" >> "$LOG/random300_6gpu_orchestrator.log"
  for shard in $(seq 0 $((NUM_GPUS - 1))); do
    local slog="$LOG/hf_deepscaler_random300_shard${shard}.log"
    echo "[gpu $shard] Phase B shard $shard/$NUM_GPUS → $slog" >> "$LOG/random300_6gpu_orchestrator.log"
    extra_ase=()
    if [[ "$PANDA_DYNAMIC_CLAIM" == "1" ]]; then
      extra_ase+=(--dynamic-claim)
    else
      extra_ase+=(--shard-id "$shard" --num-shards "$NUM_GPUS")
    fi
    CUDA_VISIBLE_DEVICES=$shard env TMPDIR="$TMPDIR" "$VLLM_PY" -u -m panda.core.run_panda_experiment \
      --mode all \
      --dataset deepscaler \
      --max-samples "$MAX_SAMPLES" \
      --n-rephrases 4 \
      --se-samples 0 \
      --weight-seeds 42,43,44,45 \
      --weight-sigma "$WEIGHT_SIGMA" --weight-rank 4 \
      --max-new-tokens 2048 \
      --topk-save 10 \
      --model-path "$MODEL" \
      --out-dir "$OUT" \
      --variants-path "$VARIANTS" \
      --tfttcl-root "$TFTTCL" \
      --device cuda:0 \
      --attn-implementation "$PANDA_ATTN" \
      "${extra_ase[@]}" \
      --resume --fast \
      >> "$slog" 2>&1 &
    PIDS+=($!)
    echo "PHASE_B shard=$shard PID=${PIDS[$(( ${#PIDS[@]} - 1 ))]}" >> "$LOG/random300_6gpu_pids.txt"
  done
  printf '%s
' "${PIDS[@]}"
}

wait_phase_b() {
  for shard in $(seq 0 $((NUM_GPUS - 1))); do
    local slog="$LOG/hf_deepscaler_random300_shard${shard}.log"
    while pgrep -f "run_panda_experiment.*deepscaler.*--shard-id $shard " >/dev/null 2>&1; do
      sleep 60
    done
    echo "Phase B shard $shard finished $(date -Iseconds)" >> "$LOG/random300_6gpu_orchestrator.log"
  done
  "$VLLM_PY" -m panda.core.recompute_metrics --out-dir "$OUT" --datasets deepscaler || true
}

case "$PHASE" in
  phase_a)
    stop_single_gpu_random300
    launch_phase_a_shards
    ;;
  phase_b)
    scr500_guard
    launch_phase_b_shards
    ;;
  all)
    stop_single_gpu_random300
    : > "$LOG/random300_6gpu_pids.txt"
    echo "=== random300 6-GPU orchestrator $(date -Iseconds) ===" >> "$LOG/random300_6gpu_orchestrator.log"
    launch_phase_a_shards > "$LOG/phase_a.pids"
    mapfile -t PIDS_A < "$LOG/phase_a.pids"
    for pid in "${PIDS_A[@]}"; do wait "$pid" || echo "WARN Phase A pid $pid exit $?" >> "$LOG/random300_6gpu_orchestrator.log"; done
    wait_phase_a || true
    launch_phase_b_shards > "$LOG/phase_b.pids"
    mapfile -t PIDS_B < "$LOG/phase_b.pids"
    for pid in "${PIDS_B[@]}"; do wait "$pid" || echo "WARN Phase B pid $pid exit $?" >> "$LOG/random300_6gpu_orchestrator.log"; done
    wait_phase_b
    echo "=== ALL DONE $(date -Iseconds) ===" >> "$LOG/random300_6gpu_orchestrator.log"
    ;;
  *)
    echo "Usage: $0 [all|phase_a|phase_b]" >&2
    exit 1
    ;;
esac
