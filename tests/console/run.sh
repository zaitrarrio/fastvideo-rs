#!/usr/bin/env bash
# Headless browser smoke test of the fv-serve console (docs/serve/console.md).
#
#   bash tests/console/run.sh
#   FV_CONSOLE_TESTS=live_echo bash tests/console/run.sh   # one of them
#
# Builds `fv-serve --features fake,full` (the director session needs
# OpenH264: with no config file its encoder is `auto`, which is OpenH264
# without NVENC; ui_gaps publishes a native WHIP stream), then runs tests/console/smoke.cjs with
# the globally installed `playwright` npm package and the preinstalled
# Chromium (PLAYWRIGHT_BROWSERS_PATH, default /opt/pw-browsers). Never runs
# `playwright install`. Then tests/console/director_playback.cjs (the
# director page against a fake engine slower than real time) and
# tests/console/live_echo.cjs (the Live input page with a fake camera and
# microphone publishing to the loopback echo model) and
# tests/console/ui_gaps.cjs (Live stream, Native API, the causal and script
# director, tier / recipe / draft, reference limits, per-API snippets).
# scripts/serve/check.sh runs it when FV_SERVE_UI=1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

command -v node >/dev/null || { echo "console smoke: node is required" >&2; exit 1; }
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/opt/pw-browsers}"
if [[ -z "${NODE_PATH:-}" ]]; then
  NODE_PATH="$(npm root -g 2>/dev/null || true)"
  export NODE_PATH
fi
node -e "require('playwright')" 2>/dev/null || {
  echo "console smoke: the 'playwright' npm package is not installed (npm i -g playwright); NODE_PATH=$NODE_PATH" >&2
  exit 1
}

# `full`: WHIP publishing (/fv/v1/streams, ui_gaps) needs webrtc + http-client.
cargo build -p fastvideo-serve --features fake,full --bin fv-serve
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
export FV_SERVE_BIN="${FV_SERVE_BIN:-$TARGET_DIR/debug/fv-serve}"
# FV_CONSOLE_ONLY=avatar runs only the avatar page test.
if [[ "${FV_CONSOLE_ONLY:-}" == avatar ]]; then exec node tests/console/avatar.cjs; fi
# FV_CONSOLE_TESTS picks a subset (e.g. `live_echo`); default: all.
for t in ${FV_CONSOLE_TESTS:-smoke director_playback avatar live_echo ui_gaps}; do
  case "$t" in
    # Director playback when generation is slower than real time (~50 s).
    # The script avatar page (Reactor avatar mode on the fake engine, ~30 s).
    # Live input: a fake camera through WHIP ingest and the Reactor
    # runtime to the loopback echo model and back (~30 s).
    # The UI-gap pages and fields (~2 min).
    smoke|director_playback|avatar|live_echo|ui_gaps) node "tests/console/$t.cjs" ;;
    *) echo "console tests: unknown test $t" >&2; exit 2 ;;
  esac
done
