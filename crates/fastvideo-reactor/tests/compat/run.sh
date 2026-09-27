#!/usr/bin/env bash
# Python reactor_sdk 1.6.0 (local mode) against the Rust Reactor runtime on
# the fake engine (design §7.5): an audio+video clip model, a video-only
# clip model and the causal model.
#
#   bash crates/fastvideo-reactor/tests/compat/run.sh
#
# Needs network access to PyPI once (the venv is cached under
# $FV_COMPAT_VENV, default target/reactor-compat-venv).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
VENV="${FV_COMPAT_VENV:-$TARGET/reactor-compat-venv}"
PY="$VENV/bin/python"
if [[ ! -x "$PY" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q "reactor_sdk==1.6.0"
fi
cargo build -q -p fastvideo-reactor --example fake_runtime
BIN="$TARGET/debug/examples/fake_runtime"

status=0
port=18780
for mode in av video causal; do
  port=$((port + 1))
  "$BIN" --port "$port" --model "$mode" >/dev/null 2>&1 &
  srv=$!
  for _ in $(seq 50); do
    curl -fsS "http://127.0.0.1:$port/session" >/dev/null 2>&1 && break
    sleep 0.1
  done
  if ! timeout 120 "$PY" "$ROOT/crates/fastvideo-reactor/tests/compat/reactor_sdk_compat.py" \
      --url "http://127.0.0.1:$port" --mode "$mode"; then
    status=1
  fi
  kill "$srv" 2>/dev/null || true
  wait "$srv" 2>/dev/null || true
done
exit $status
