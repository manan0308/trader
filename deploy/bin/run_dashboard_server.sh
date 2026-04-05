#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
LOG_DIR="$REPO/cache/logs"
HOST="${TRADER_HOST:-127.0.0.1}"
PORT="${TRADER_PORT:-8050}"
mkdir -p "$LOG_DIR"

cd "$REPO"
exec "$REPO/.venv/bin/python" -m runtime.local_api_server --host "$HOST" --port "$PORT" >> "$LOG_DIR/dashboard_server.log" 2>&1
