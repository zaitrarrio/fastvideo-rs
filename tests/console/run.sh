#!/usr/bin/env bash
# Headless browser smoke test of the fv-serve console (docs/serve/console.md).
#
#   bash tests/console/run.sh
#
# Builds `fv-serve --features fake`, then runs tests/console/smoke.cjs with
# the globally installed `playwright` npm package and the preinstalled
# Chromium (PLAYWRIGHT_BROWSERS_PATH, default /opt/pw-browsers). Never runs
# `playwright install`. scripts/serve/check.sh runs it when FV_SERVE_UI=1.
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

cargo build -p fastvideo-serve --features fake --bin fv-serve
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
FV_SERVE_BIN="${FV_SERVE_BIN:-$TARGET_DIR/debug/fv-serve}" node tests/console/smoke.cjs
