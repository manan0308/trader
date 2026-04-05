#!/bin/zsh
set -euo pipefail

REPO="/Users/mananagarwal/Desktop/2nd brain/plant to image/trader"
AGENTS_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$AGENTS_DIR" "$REPO/cache/logs"

copy_and_load() {
  local source_plist="$1"
  local target_plist="$AGENTS_DIR/$(basename "$source_plist")"
  cp "$source_plist" "$target_plist"
  launchctl bootout "gui/$(id -u)" "$target_plist" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$(id -u)" "$target_plist"
}

copy_and_load "$REPO/deploy/launchd/com.trader.daily-cycle.plist"
copy_and_load "$REPO/deploy/launchd/com.trader.dashboard.plist"

launchctl kickstart -k "gui/$(id -u)/com.trader.dashboard"
launchctl kickstart -k "gui/$(id -u)/com.trader.daily-cycle"

echo "Installed launchd jobs:"
launchctl list | rg 'com\.trader\.(daily-cycle|dashboard)' || true
