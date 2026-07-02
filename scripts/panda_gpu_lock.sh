#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
# GPU coordination: exclusive ASE generate vs prefix re-query pilot.
#
# Source from other scripts:
#   source "$(dirname "$0")/panda_gpu_lock.sh"
#
# Owners: prefix_requery | ase_generate | (empty)

LOCK_DIR="$PANDA_OUTPUTS/ase_locks"
GPU_OWNER_FILE="${LOCK_DIR}/gpu_owner"
GPU_LOCK_FILE="${LOCK_DIR}/gpu.lock"
GPU_LOG="${LOCK_DIR}/gpu_lock.log"

mkdir -p "$LOCK_DIR"

_ase_gpu_log() {
  echo "[$(date '+%F %T')] $*" >> "$GPU_LOG"
}

_gpu_owner_read() {
  if [[ -f "$GPU_OWNER_FILE" ]]; then
    tr -d '[:space:]' < "$GPU_OWNER_FILE"
  fi
}

_gpu_conflicts_running() {
  local owner="${1:-}"
  local cur
  cur=$(_gpu_owner_read)
  if pgrep -f "python3 -m panda.core.run_panda_experiment" >/dev/null 2>&1; then
    return 0
  fi
  if pgrep -f "greedy_unc_single_batch_refine.py" >/dev/null 2>&1; then
    return 0
  fi
  if pgrep -f "python3 -u -m panda.core.run_prefix_requery_collect" >/dev/null 2>&1 \
    || pgrep -f "python3 -m panda.core.run_prefix_requery_collect" >/dev/null 2>&1; then
    return 0
  fi
  if [[ -n "$cur" ]]; then
    if [[ -z "$owner" || "$cur" != "$owner" ]]; then
      return 0
    fi
  fi
  return 1
}

acquire_gpu() {
  local owner="${1:?owner required}"
  local reason="${2:-}"
  while true; do
    local fd cur
    exec {fd}>"$GPU_LOCK_FILE"
    flock -x "$fd"
    cur=$(_gpu_owner_read)
    if [[ -z "$cur" || "$cur" == "$owner" ]]; then
      if ! _gpu_conflicts_running "$owner"; then
        echo "$owner" > "$GPU_OWNER_FILE"
        if [[ -n "$reason" ]]; then
          echo "$reason" > "${LOCK_DIR}/gpu_reason"
        fi
        _ase_gpu_log "acquire $owner ${reason:-}"
        flock -u "$fd"
        exec {fd}>&-
        return 0
      fi
    fi
    flock -u "$fd"
    exec {fd}>&-
    _ase_gpu_log "acquire wait $owner (holder=${cur:-free} conflicts=1) ${reason:-}"
    sleep 30
  done
}

release_gpu() {
  local owner="${1:?owner required}"
  local fd cur
  exec {fd}>"$GPU_LOCK_FILE"
  flock -x "$fd"
  cur=$(_gpu_owner_read)
  if [[ -z "$cur" || "$cur" == "$owner" ]]; then
    : > "$GPU_OWNER_FILE"
    rm -f "${LOCK_DIR}/gpu_reason"
    _ase_gpu_log "release $owner"
  else
    _ase_gpu_log "release skip $owner (holder=$cur)"
  fi
  flock -u "$fd"
  exec {fd}>&-
}

wait_gpu_free() {
  while true; do
    local cur
    cur=$(_gpu_owner_read)
    if [[ -z "$cur" ]] && ! _gpu_conflicts_running ""; then
      return 0
    fi
    _ase_gpu_log "wait_gpu_free (owner=${cur:-empty})"
    sleep 30
  done
}

wait_gpu_owner() {
  local want="${1:?owner required}"
  while true; do
    local cur
    cur=$(_gpu_owner_read)
    if [[ -z "$cur" || "$cur" == "$want" ]]; then
      return 0
    fi
    _ase_gpu_log "wait_gpu_owner want=$want holder=$cur"
    sleep 30
  done
}

wait_for_gpus_compatible() {
  _ase_gpu_log "wait_for_gpus_compatible start"
  while _gpu_conflicts_running ""; do
    local cur
    cur=$(_gpu_owner_read)
    _ase_gpu_log "compatible wait owner=${cur:-empty} procs=1"
    sleep 60
  done
  wait_gpu_free
  _ase_gpu_log "wait_for_gpus_compatible done"
}
