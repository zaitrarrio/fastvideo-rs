#!/usr/bin/env bash
# Native /fv/v1/streams end to end on the fake engine (WP-15): fake frames →
# pacer → OpenH264/Opus → WHIP to a local MediaMTX → WHEP viewer that
# decodes the H.264 and reads back the burned-in frame index.
#
#   bash scripts/serve/whip-e2e.sh            # downloads MediaMTX if needed
#   MEDIAMTX=/path/to/mediamtx bash scripts/serve/whip-e2e.sh
#
# Without MediaMTX the same test runs against the in-process WHIP endpoint
# (cargo test -p fastvideo-serve --features full --test streams_whip).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
VER="${MEDIAMTX_VERSION:-v1.15.1}"
WORK="${TMPDIR:-/tmp}/fv-whip-e2e"
mkdir -p "$WORK"
MTX="${MEDIAMTX:-$WORK/mediamtx}"
if [[ ! -x "$MTX" ]]; then
  curl -fsSL "https://github.com/bluenviron/mediamtx/releases/download/${VER}/mediamtx_${VER}_linux_amd64.tar.gz" \
    | tar xz -C "$WORK" mediamtx mediamtx.yml
  MTX="$WORK/mediamtx"
fi
[[ -f "$WORK/mediamtx.yml" ]] || "$MTX" --help >/dev/null
PORT="${MTX_PORT:-18889}"
MTX_RTSP=no MTX_RTMP=no MTX_HLS=no MTX_SRT=no MTX_API=no \
MTX_WEBRTCADDRESS="127.0.0.1:${PORT}" MTX_WEBRTCLOCALUDPADDRESS="127.0.0.1:$((PORT - 700))" \
MTX_WEBRTCADDITIONALHOSTS=127.0.0.1 \
  "$MTX" "$WORK/mediamtx.yml" >"$WORK/mediamtx.log" 2>&1 &
MTX_PID=$!
trap 'kill "$MTX_PID" 2>/dev/null || true' EXIT
sleep 1
FV_TEST_MEDIAMTX="http://127.0.0.1:${PORT}" \
  cargo test -p fastvideo-serve --features full --test streams_whip -- --nocapture
grep -E 'is (publishing|reading)' "$WORK/mediamtx.log" || true
