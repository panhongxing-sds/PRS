#!/usr/bin/env bash
# Reassemble split archives from bundles/ and restore outputs/, models/, third_party/TokUR/
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PRS_ROOT"
export PRS_ROOT

BUNDLES_DIR="${PRS_ROOT}/bundles"
MANIFEST="${BUNDLES_DIR}/MANIFEST.json"
[[ -f "$MANIFEST" ]] || { echo "Missing $MANIFEST"; exit 1; }

restore_one() {
  local name="$1"
  local dest_parent="$2"
  local dest_name="$3"
  python3 - "$name" "$dest_parent" "$dest_name" << 'PY'
import hashlib, json, os, subprocess, sys

name, dest_parent, dest_name = sys.argv[1:4]
prs_root = os.environ["PRS_ROOT"]
bundles_dir = os.path.join(prs_root, "bundles")
with open(os.path.join(bundles_dir, "MANIFEST.json")) as f:
    entry = json.load(f)[name]

ext = entry["archive_extension"]
archive = os.path.join(bundles_dir, f"_restore_{name}.{ext}")
with open(archive, "wb") as out:
    for part in entry["parts"]:
        path = os.path.join(bundles_dir, part["file"])
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != part["sha256"]:
            raise SystemExit(f"Checksum mismatch: {part['file']}")

dest = os.path.join(dest_parent, dest_name)
if os.path.exists(dest):
    print(f"Already exists: {dest}")
    os.remove(archive)
    sys.exit(0)

os.makedirs(dest_parent, exist_ok=True)
if ext == "tar.zst":
    p1 = subprocess.Popen(["zstd", "-d", "-c", archive], stdout=subprocess.PIPE)
    subprocess.check_call(["tar", "-xf", "-", "-C", dest_parent], stdin=p1.stdout)
    p1.wait()
elif ext == "tar.gz":
    subprocess.check_call(["tar", "-xzf", archive, "-C", dest_parent])
else:
    raise SystemExit(f"Unknown extension: {ext}")

os.remove(archive)
print(f"Restored {dest}")
PY
}

restore_one outputs "$PRS_ROOT" outputs
restore_one models "$PRS_ROOT" models
restore_one tokur "$(dirname "$PRS_ROOT/third_party/TokUR")" "$(basename "$PRS_ROOT/third_party/TokUR")"
echo "Extract complete."
