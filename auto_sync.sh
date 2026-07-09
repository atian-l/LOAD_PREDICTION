#!/usr/bin/env bash
# Periodic auto-sync: stage all changes, commit if any, push to origin.
# Safe for unattended scheduled runs; skips when there is nothing to commit.
set -u
cd "$(dirname "$0")" || exit 1
ts() { date '+%Y-%m-%d %H:%M:%S'; }

git add -A
if git diff --cached --quiet; then
  echo "[$(ts)] nothing to commit"
  exit 0
fi

if git commit -m "auto-sync $(ts)" >/dev/null 2>&1; then
  echo "[$(ts)] committed"
else
  echo "[$(ts)] commit failed"
  exit 1
fi

# post-commit hook already pushes; explicit push as backup
git push origin HEAD >/dev/null 2>&1 && echo "[$(ts)] pushed" || echo "[$(ts)] push failed"
