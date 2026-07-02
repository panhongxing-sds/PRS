#!/usr/bin/env bash
# Exit 0 iff all 4 models × 3 seeds × 3 math datasets have expected raw counts.
set -uo pipefail
OUT="${PANDA_OUTPUTS:-/root/autodl-tmp/panda-outputs}"
declare -A EXPECT=( [minerva]=272 [math500]=300 [gsm8k]=300 )
MODELS=(qwen25_3b llama32_1b llama31_8b qwen3_8b)
SEEDS=(41 42 43)

count_ok() {
  local dir=$1 want=$2
  local n
  n=$(find "$dir" -maxdepth 1 -name '*.json' ! -name '*.partial.json' ! -name '*.error.json' 2>/dev/null | wc -l)
  [[ "$n" -ge "$want" ]]
}

ok=1
for mt in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for ds in minerva math500 gsm8k; do
      dir="$OUT/maintable_${mt}/seed${seed}/${ds}/raw_runs"
      want=${EXPECT[$ds]}
      if [[ ! -d $dir ]] || ! count_ok "$dir" "$want"; then
        have=$(find "$dir" -maxdepth 1 -name '*.json' ! -name '*.partial.json' ! -name '*.error.json' 2>/dev/null | wc -l)
        echo "INCOMPLETE $mt seed${seed}/$ds: ${have:-0}/$want"
        ok=0
      fi
    done
  done
done

if [[ $ok -eq 1 ]]; then
  echo "ALL_MATH_COMPLETE $(date '+%F %T')"
fi
exit $((1 - ok))
