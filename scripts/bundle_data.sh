#!/usr/bin/env bash
# Compress large data dirs, split for GitHub 100MB file limit, write MANIFEST.json
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PANDA_ROOT"

BUNDLES_DIR="${PANDA_ROOT}/bundles"
PART_SIZE_MB=95
PART_BYTES=$((PART_SIZE_MB * 1024 * 1024))
mkdir -p "$BUNDLES_DIR"

if command -v zstd >/dev/null 2>&1; then
  COMPRESSOR=zstd
  COMPRESS_LEVEL=-19
  EXT=tar.zst
else
  COMPRESSOR=gzip
  COMPRESS_LEVEL=-9
  EXT=tar.gz
fi

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

dir_size_bytes() {
  du -sb "$1" 2>/dev/null | awk '{print $1}'
}

bundle_one() {
  local name="$1"
  local src_path="$2"
  local exclude_logs="${3:-0}"
  local archive_base="${BUNDLES_DIR}/${name}.${EXT}"

  if [[ ! -e "$src_path" ]]; then
    echo "Skip missing: $src_path"
    return 0
  fi

  local orig_bytes
  orig_bytes=$(dir_size_bytes "$src_path")
  echo "=== Bundling $name (original ~${orig_bytes} bytes) ==="

  rm -f "${archive_base}" "${BUNDLES_DIR}/${name}.${EXT}.part_"*
  local tar_opts=(-cf - -C "$(dirname "$src_path")" "$(basename "$src_path")")
  if [[ "$exclude_logs" == "1" ]]; then
    tar_opts+=(--exclude='*.log')
  fi

  if [[ "$COMPRESSOR" == zstd ]]; then
    tar "${tar_opts[@]}" | zstd $COMPRESS_LEVEL -T0 -o "$archive_base"
  else
    tar "${tar_opts[@]}" | gzip $COMPRESS_LEVEL > "$archive_base"
  fi

  local archive_bytes
  archive_bytes=$(stat -c%s "$archive_base")
  split -b "$PART_BYTES" "$archive_base" \
    "${BUNDLES_DIR}/${name}.${EXT}.part_"
  rm -f "$archive_base"

  local parts=()
  local part
  for part in "${BUNDLES_DIR}/${name}.${EXT}.part_"*; do
    [[ -f "$part" ]] || continue
    parts+=("$(basename "$part")")
  done

  python3 - "$name" "$EXT" "$orig_bytes" "$archive_bytes" "$COMPRESSOR" "${parts[@]}" << 'PY'
import json, sys, os
name, ext, orig, arch, comp = sys.argv[1:6]
parts = sys.argv[6:]
manifest_path = os.path.join(os.environ.get("PANDA_ROOT", "."), "bundles", "MANIFEST.json")
data = {}
if os.path.isfile(manifest_path):
    with open(manifest_path) as f:
        data = json.load(f)
shas = []
bundles_dir = os.path.dirname(manifest_path)
for p in parts:
    with open(os.path.join(bundles_dir, p), "rb") as f:
        import hashlib
        shas.append(hashlib.sha256(f.read()).hexdigest())
data[name] = {
    "archive_extension": ext,
    "compressor": comp,
    "original_size_bytes": int(orig),
    "compressed_size_bytes": int(arch),
    "part_size_mb": int(os.environ.get("PART_SIZE_MB", 95)),
    "parts": [{"file": p, "sha256": s} for p, s in zip(parts, shas)],
}
with open(manifest_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
print(f"Wrote {manifest_path} entry for {name} ({len(parts)} parts)")
PY

  echo "Done $name: compressed ${archive_bytes} bytes, ${#parts[@]} parts"
}

export PART_SIZE_MB
export PANDA_ROOT
PART_SIZE_MB=$PART_SIZE_MB

# outputs (~85GB) and models (~39GB) are too large for GitHub — keep local only.
# third_party/TokUR is tracked directly in git (~150MB, no file >100MB).
# Uncomment below only for offline archival:
# bundle_one outputs "$PANDA_ROOT/outputs" 1
# bundle_one tokur "$PANDA_ROOT/third_party/TokUR" 0

echo "All bundles written under $BUNDLES_DIR"
ls -lh "${BUNDLES_DIR}"/*.part_* 2>/dev/null | wc -l
