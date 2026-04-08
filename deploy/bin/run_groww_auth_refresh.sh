#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
LOG_DIR="$REPO/cache/logs"
mkdir -p "$LOG_DIR"

cd "$REPO"

status=0
"$REPO/.venv/bin/python" -m runtime.groww_auth_refresh >> "$LOG_DIR/groww_auth_refresh.log" 2>&1 || status=$?

if command -v node >/dev/null 2>&1; then
  node "$REPO/dashboard/scripts/sync-data.mjs" >> "$LOG_DIR/groww_auth_refresh.log" 2>&1 || {
    if [ "$status" -eq 0 ]; then
      status=1
    fi
  }
fi

exit "$status"
