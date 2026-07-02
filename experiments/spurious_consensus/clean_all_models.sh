#!/usr/bin/env bash
# 六模型统一 cleaned/canonicalized pipeline（逐文件，避免 OOM）
set -euo pipefail
cd "$(dirname "$0")"
export PANDA_ROOT="${PANDA_ROOT:-/root/PANDA}"

TAGS=(qwen25_05b llama32_1b qwen25_15b phi4_mini qwen25_3b qwen25_7b)
SEED=41

for tag in "${TAGS[@]}"; do
  echo "========== cleaning ${tag} =========="
  for f in data/samples/samples_${tag}_seed${SEED}_*.jsonl; do
    [[ "$f" == *.bak ]] && continue
    [[ "$f" == *.dryrun ]] && continue
    [[ "$f" == *.preshard ]] && continue
    bench=$(basename "$f" .jsonl)
    bench=${bench#samples_${tag}_seed${SEED}_}
    bench=${bench%%.shard*}
    bench=${bench%%.s3_*}
    case "$bench" in
      deepscaler|gpqa_diamond|aime_2024) ;;
      *) continue ;;
    esac
    if [[ ! -f "${f}.bak" ]]; then
      cp "$f" "${f}.bak"
      echo "  backup ${f}.bak"
    fi
    python3 clean_one_file.py "$f" "$tag"
  done
done

echo "All models cleaned."
