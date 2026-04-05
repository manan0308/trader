#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
LOG_DIR="$REPO/cache/logs"
mkdir -p "$LOG_DIR"

cd "$REPO"
exec "$REPO/.venv/bin/python" -m trader_system.runtime.daily_cycle --portfolio-file config/portfolio_state.example.json >> "$LOG_DIR/daily_cycle.log" 2>&1
