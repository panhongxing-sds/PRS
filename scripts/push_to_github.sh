#!/usr/bin/env bash
# Stage code + small data, commit, push (requires GITHUB_TOKEN env).
set -euo pipefail
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN}"

source "$(dirname "$0")/env.sh"
cd "$PRS_ROOT"

echo "=== $(date) push start ==="
rm -f .git/index.lock
rm -rf third_party/TokUR/.git 2>/dev/null || true

git add .gitattributes .gitignore pyproject.toml README.md LICENSE requirements.txt
git add configs data paper scripts src tests third_party

git status --short

git commit -m "$(cat <<'EOF'
Initial public release: PRS code, paper tables, and TokUR baseline.

Excludes large model weights (~39GB) and experiment outputs (~85GB);
those are downloaded or regenerated locally.
EOF
)" || echo "nothing to commit or commit failed"

REMOTE="https://x-access-token:${GITHUB_TOKEN}@github.com/panhongxing-sds/PRS.git"
git remote set-url origin "$REMOTE" 2>/dev/null || git remote add origin "$REMOTE"

git config http.postBuffer 524288000

echo "git push main..."
git push -u origin main

echo "=== $(date) push done ==="
