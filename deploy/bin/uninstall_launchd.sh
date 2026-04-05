#!/bin/zsh
set -euo pipefail

AGENTS_DIR="$HOME/Library/LaunchAgents"

remove_job() {
  local label="$1"
  local plist="$AGENTS_DIR/$2"
  launchctl bootout "gui/$(id -u)" "$plist" >/dev/null 2>&1 || true
  rm -f "$plist"
  echo "Removed $label"
}

remove_job "daily-cycle" "com.trader.daily-cycle.plist"
remove_job "dashboard" "com.trader.dashboard.plist"
