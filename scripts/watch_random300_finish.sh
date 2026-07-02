#!/usr/bin/env bash
# Poll random300 Phase B until 300 final raw_runs, then metrics + CPU SC@9 pairing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/env.sh"

OUT_DIR="${OUT_DIR:-/root/autodl-tmp/panda-outputs/maintable_qwen25_3b_deepscaler_random300/seed41}"
RAW_RUNS="$OUT_DIR/deepscaler/raw_runs"
VARIANTS="${VARIANTS:-/root/autodl-tmp/panda-outputs/qaac_api_bench/deepscaler_random300/variants.jsonl}"
K9_JSONL="${K9_JSONL:-/root/autodl-tmp/PANDA/experiments/spurious_consensus/data/samples/samples_qwen25_3b_seed41_deepscaler_random300_k9.jsonl}"
STATUS="$ROOT/paper/analysis/random300_status.txt"
LOG="${LOG:-/root/autodl-tmp/panda-outputs/maintable_qwen25_3b_deepscaler_random300/logs/watch_random300_finish.log}"
LOCK="${LOCK:-/root/autodl-tmp/panda-outputs/maintable_qwen25_3b_deepscaler_random300/logs/watch_random300_finish.lock}"
METRICS_MARKER="$OUT_DIR/.random300_metrics_done"
PAIR_OUT="$ROOT/paper/analysis/random300_sc9_panda_join.jsonl"
FAIR_OUT="$ROOT/paper/analysis/random300_fair_baselines.json"
POLL_SEC="${POLL_SEC:-60}"
TARGET="${TARGET:-300}"

mkdir -p "$(dirname "$LOG")" "$(dirname "$LOCK")"

count_finals() {
  find "$RAW_RUNS" -maxdepth 1 -type f -name '*.json' ! -name '*partial*' 2>/dev/null | wc -l | tr -d ' '
}

shard_progress() {
  python3 - "$RAW_RUNS" "$VARIANTS" << 'PY'
import glob, json, os, sys
raw, variants = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(variants)]
finals = {
    os.path.basename(p).replace(".json", "")
    for p in glob.glob(os.path.join(raw, "*.json"))
    if "partial" not in p
}
for sid in range(6):
    owned = [r["unique_id"] for r in rows if r["idx"] % 6 == sid]
    done = sum(1 for u in owned if u in finals)
    print(f"shard{sid}={done}/{len(owned)}")
print(f"total={len(finals)}/{len(rows)}")
PY
}

append_status() {
  {
    echo ""
    echo "=== Completion watcher $(date -Iseconds) ==="
    echo "$1"
  } >> "$STATUS"
}

run_metrics() {
  if [[ -f "$METRICS_MARKER" ]]; then
    echo "[$(date -Iseconds)] metrics already done (marker $METRICS_MARKER)" | tee -a "$LOG"
    return 0
  fi
  echo "[$(date -Iseconds)] Running metrics mode (non-sharded)..." | tee -a "$LOG"
  cd "$ROOT"
  python3 -m panda.core.run_panda_experiment --mode metrics --dataset deepscaler \
    --out-dir "$OUT_DIR" \
    --variants-path "$VARIANTS" \
    2>&1 | tee -a "$LOG"
  date -Iseconds > "$METRICS_MARKER"
  append_status "Metrics complete: summary at $OUT_DIR/deepscaler/summary.jsonl"
}

run_cpu_pairing() {
  echo "[$(date -Iseconds)] CPU pairing SC@9 + PANDA summary..." | tee -a "$LOG"
  SUMMARY="$OUT_DIR/deepscaler/summary.jsonl"
  if [[ ! -f "$K9_JSONL" ]]; then
    echo "WARN: missing K9 jsonl $K9_JSONL" | tee -a "$LOG"
    return 1
  fi
  if [[ ! -f "$SUMMARY" ]]; then
    echo "WARN: missing summary $SUMMARY" | tee -a "$LOG"
    return 1
  fi

  python3 - "$K9_JSONL" "$SUMMARY" "$PAIR_OUT" << 'PY'
import json, sys
from pathlib import Path

k9_path, summ_path, out_path = map(Path, sys.argv[1:4])
sc = {}
for ln in k9_path.read_text().splitlines():
    if ln.strip():
        o = json.loads(ln)
        sc[o["id"]] = o

def p_top(rec):
    ans = [a for a, c in zip(rec.get("answers") or [], rec.get("correct") or []) if a]
    if len(ans) < 2:
        return float("nan"), float("nan")
    from collections import Counter
    cnt = Counter(ans)
    p = max(cnt.values()) / len(ans)
    return p, 1.0 - p

joined = []
for ln in summ_path.read_text().splitlines():
    if not ln.strip():
        continue
    p = json.loads(ln)
    rid = p.get("id")
    srec = sc.get(rid)
    if not srec:
        continue
    pt, u = p_top(srec)
    joined.append({
        "id": rid,
        "y": int(p.get("label_wrong") or p.get("label_wrong_clean") or 0),
        "p_top_k9": pt,
        "u_ans_k9": u,
        "bd": p.get("bd"),
        "T_ent_prox_lin": p.get("T_ent_prox_lin"),
        "DH_Score": p.get("DH_Score") or p.get("DH-Score"),
    })

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    for row in joined:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"joined n={len(joined)} -> {out_path}")
PY

  if [[ -f "$ROOT/scripts/aggregate_fair_baselines.py" ]]; then
    echo "[$(date -Iseconds)] aggregate_fair_baselines on random300 raw_runs..." | tee -a "$LOG"
    python3 - "$RAW_RUNS" "$FAIR_OUT" << 'PY' 2>&1 | tee -a "$LOG" || true
import json, sys
from pathlib import Path
import importlib.util

raw_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location(
    "afb", Path("/root/autodl-tmp/PANDA/scripts/aggregate_fair_baselines.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

summ = {}
summ_path = raw_dir.parent.parent / "summary.jsonl"
if summ_path.exists():
    for ln in summ_path.read_text().splitlines():
        if ln.strip():
            o = json.loads(ln)
            summ[o["id"]] = o
jobs = []
for rp in sorted(raw_dir.glob("*.json")):
    if "partial" in rp.name or "error" in rp.name:
        continue
    rid = rp.stem
    sr = summ.get(rid)
    if not sr:
        continue
    jobs.append((str(rp), rid, 41, "deepscaler", "qwen25_3b", sr))

rows = []
for j in jobs:
    r = mod._job(j)
    if r:
        rows.append(r)

keys = [k for k, _, _ in mod.TOKEN_KEYS] + ["SE (8-sample)", "Dissent", "F_resp (8-pert)"]
summary = {"N": len(rows), "methods": {}, "note": "random300 subset fair baselines from raw_runs"}
import numpy as np
for k in keys:
    inv = k.startswith(("SC", "DeepConf", "SAR")) or "DeepConf" in k
    summary["methods"][k] = mod.pooled_auroc(rows, k, invert=inv)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
  fi
  append_status "CPU pairing: $PAIR_OUT (fair: $FAIR_OUT if generated)"
}

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another watcher holds $LOCK; exit." | tee -a "$LOG"
  exit 0
fi

echo "watch_random300_finish started pid=$$ $(date -Iseconds)" | tee -a "$LOG"
append_status "Watcher PID $$ polling every ${POLL_SEC}s for ${TARGET} finals in $RAW_RUNS"

while true; do
  n="$(count_finals)"
  prog="$(shard_progress | tr '\n' ' ')"
  echo "[$(date -Iseconds)] Phase B finals ${n}/${TARGET} (${prog})" | tee -a "$LOG"
  if [[ "$n" -ge "$TARGET" ]]; then
    append_status "Phase B COMPLETE ${n}/${TARGET} finals @ $(date -Iseconds)"
    run_metrics
    run_cpu_pairing
    append_status "Pipeline DONE @ $(date -Iseconds). Join: $PAIR_OUT"
    echo "[$(date -Iseconds)] All done." | tee -a "$LOG"
    exit 0
  fi
  sleep "$POLL_SEC"
done
