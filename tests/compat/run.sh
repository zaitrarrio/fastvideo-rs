#!/usr/bin/env bash
# Client-compat suites (docs/serve/design.md §7.5, WP-17): real client
# libraries against `fv-serve --features fake` (plus the streaming/HTTP
# features), each suite on a fresh server.
#
#   bash tests/compat/run.sh               # every suite
#   bash tests/compat/run.sh openai ltx    # some suites
#   bash tests/compat/run.sh --list
#
# Suites:
#   openai        openai (Python) client.videos.* against /v1/videos
#   fastwan       the FastWan Video API client flow (aiohttp)
#   minimax       MiniMax V2 documented HTTP flow (requests), callback_url
#   ltx           LTX API documented requests snippets (v2 async, v1 sync, upload)
#   fal-py        fal-client (Python): queue, status, result, subscribe, run, cancel, errors
#   fal-js        @fal-ai/client (JS): requestMiddleware + proxyUrl, SSE, storage upload
#   fal-webhook   fal webhooks verified with the JWKS + Ed25519 (PyNaCl)
#   fal-director  @fal-ai/client alpha fal.realtime.open(wma) in headless Chromium
#   reactor       reactor_sdk (Python) local mode: A/V, video-only, causal
#   console       the /console Playwright smoke (tests/console/smoke.cjs)
#
# Client versions are pinned in tests/compat/requirements.txt and
# tests/compat/package-lock.json; the venv and node_modules are cached under
# $FV_COMPAT_DIR (default $CARGO_TARGET_DIR/compat) and rebuilt when the pin
# files change. API suites run over TLS: a throwaway self-signed CA is
# exported as SSL_CERT_FILE / REQUESTS_CA_BUNDLE / NODE_EXTRA_CA_CERTS and
# tests/compat/lib/tls_proxy.py fronts each server.
#
# Environment:
#   FV_SERVE_BIN              use this fv-serve instead of building one
#   PLAYWRIGHT_BROWSERS_PATH  Chromium for the browser suites (default
#                             /opt/pw-browsers; this script never installs
#                             browsers: CI runs `playwright install` itself)
#   FV_COMPAT_KEEP_LOGS=1     keep the per-suite logs directory
#
# Needs: cargo (unless FV_SERVE_BIN), python3 (3.10+), node 18+, npm,
# openssl, curl; ffmpeg on PATH so the fake engine writes real MP4s.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/tests/compat"
ALL_SUITES=(openai fastwan minimax ltx fal-py fal-js fal-webhook fal-director reactor console)

if [[ "${1:-}" == "--list" ]]; then
  printf '%s\n' "${ALL_SUITES[@]}"
  exit 0
fi
SUITES=("$@")
[[ ${#SUITES[@]} -eq 0 ]] && SUITES=("${ALL_SUITES[@]}")
for s in "${SUITES[@]}"; do
  [[ " ${ALL_SUITES[*]} " == *" $s "* ]] || { echo "unknown suite: $s (see --list)" >&2; exit 2; }
done

TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
CACHE="${FV_COMPAT_DIR:-$TARGET/compat}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/opt/pw-browsers}"
KEY="fv-compat-key"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/fv-compat.XXXXXX")"
LOGS="$WORK/logs"
mkdir -p "$LOGS" "$CACHE"

log() { printf '\n==> %s\n' "$*" >&2; }
die() { echo "compat: $*" >&2; exit 1; }

# ---------------------------------------------------------------- processes
PIDS=()
cleanup() {
  local p
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  for p in "${PIDS[@]}"; do wait "$p" 2>/dev/null || true; done
  if [[ "${FV_COMPAT_KEEP_LOGS:-0}" == "1" ]]; then
    echo "compat: logs kept in $LOGS" >&2
  else
    rm -rf "$WORK"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'; }

stop_pids() {
  local p keep=()
  for p in "$@"; do kill "$p" 2>/dev/null || true; wait "$p" 2>/dev/null || true; done
  for p in "${PIDS[@]}"; do [[ " $* " == *" $p "* ]] || keep+=("$p"); done
  PIDS=("${keep[@]}")
}

# ------------------------------------------------------------------- setup
command -v python3 >/dev/null || die "python3 is required"
command -v node >/dev/null || die "node is required"
command -v openssl >/dev/null || die "openssl is required"
command -v ffmpeg >/dev/null || echo "compat: warning: no ffmpeg on PATH; outputs are placeholders and MP4 checks fail" >&2

if [[ -n "${FV_SERVE_BIN:-}" ]]; then
  BIN="$FV_SERVE_BIN"
else
  log "building fv-serve (fake engine, WebRTC, encoders, HTTP client)"
  (cd "$ROOT" && cargo build -p fastvideo-serve --features fake,full --bin fv-serve \
    --config 'profile.dev.package.openh264-sys2.opt-level=3' \
    --config 'profile.dev.package.openh264.opt-level=3' \
    --config 'profile.dev.package.audiopus_sys.opt-level=3' \
    --config 'profile.dev.package.libwebp-sys.opt-level=3') || die "cargo build failed"
  BIN="$TARGET/debug/fv-serve"
fi
[[ -x "$BIN" ]] || die "no fv-serve at $BIN"

hash_of() { sha256sum "$@" | sha256sum | cut -c1-16; }

VENV="$CACHE/venv"
PY="$VENV/bin/python"
stamp="$(hash_of "$HERE/requirements.txt")"
if [[ ! -x "$PY" || "$(cat "$VENV/.stamp" 2>/dev/null)" != "$stamp" ]]; then
  log "python clients (tests/compat/requirements.txt)"
  rm -rf "$VENV"
  python3 -m venv "$VENV" || die "venv"
  "$VENV/bin/pip" install -q --disable-pip-version-check -r "$HERE/requirements.txt" || die "pip install"
  echo "$stamp" > "$VENV/.stamp"
fi

NODE_DIR="$CACHE/node"
stamp="$(hash_of "$HERE/package.json" "$HERE/package-lock.json")"
if [[ ! -d "$NODE_DIR/node_modules" || "$(cat "$NODE_DIR/.stamp" 2>/dev/null)" != "$stamp" ]]; then
  log "js clients (tests/compat/package-lock.json)"
  rm -rf "$NODE_DIR"
  mkdir -p "$NODE_DIR"
  cp "$HERE/package.json" "$HERE/package-lock.json" "$NODE_DIR/"
  (cd "$NODE_DIR" && PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm ci --no-audit --no-fund --loglevel=error) || die "npm ci"
  echo "$stamp" > "$NODE_DIR/.stamp"
fi

# The throwaway CA (self-signed, CA:TRUE, SAN 127.0.0.1/localhost).
CERT="$WORK/ca.pem"
CERT_KEY="$WORK/ca.key"
openssl req -x509 -newkey rsa:2048 -nodes -days 1 -keyout "$CERT_KEY" -out "$CERT" \
  -subj "/CN=fv-compat" -addext "subjectAltName=IP:127.0.0.1,DNS:localhost" \
  -addext "basicConstraints=critical,CA:TRUE" >/dev/null 2>&1 || die "openssl"

# Local traffic never goes through an outbound proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

# ---------------------------------------------------------------- fv-serve
# start_serve <name> <protocols toml> [--tls] [ENV=VAL...]
# Sets SERVE_PID, SERVE_PORT, BASE (https when --tls) and TLS_PORT.
start_serve() {
  local name="$1" protocols="$2"
  shift 2
  local tls=0
  if [[ "${1:-}" == "--tls" ]]; then tls=1; shift; fi
  local dir="$WORK/$name"
  mkdir -p "$dir/state"
  { cat "$HERE/serve.toml"; printf '\n%s\n' "$protocols"; } > "$dir/serve.toml"
  SERVE_PORT="$(free_port)"
  TLS_PORT="$(free_port)"
  local public="http://127.0.0.1:$SERVE_PORT"
  [[ $tls == 1 ]] && public="https://127.0.0.1:$TLS_PORT"
  local keyhash
  keyhash="$(printf '%s' "$KEY" | sha256sum | cut -d' ' -f1)"
  env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy \
    FV_BIND="127.0.0.1:$SERVE_PORT" FV_PUBLIC_BASE_URL="$public" FV_STATE_DIR="$dir/state" \
    FV_API_KEYS="$keyhash" FV_URL_SIGNING_KEY="compat-signing" FV_ADMIN_TOKEN="fvadm_compat_admin_token" \
    FV_CALLBACKS_ALLOW_PRIVATE=1 RUST_LOG="${FV_COMPAT_RUST_LOG:-warn}" "$@" \
    "$BIN" --config "$dir/serve.toml" >"$LOGS/$name.serve.log" 2>&1 &
  SERVE_PID=$!
  PIDS+=("$SERVE_PID")
  local i
  for i in $(seq 300); do
    curl -fsS "http://127.0.0.1:$SERVE_PORT/health" >/dev/null 2>&1 && break
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
      tail -30 "$LOGS/$name.serve.log" >&2
      return 1
    fi
    sleep 0.1
  done
  curl -fsS "http://127.0.0.1:$SERVE_PORT/health" >/dev/null 2>&1 || { tail -30 "$LOGS/$name.serve.log" >&2; return 1; }
  BASE="http://127.0.0.1:$SERVE_PORT"
  TLS_PID=""
  if [[ $tls == 1 ]]; then
    "$PY" "$HERE/lib/tls_proxy.py" --listen "$TLS_PORT" --upstream "$SERVE_PORT" --cert "$CERT" --key "$CERT_KEY" \
      >"$LOGS/$name.tls.log" 2>&1 &
    TLS_PID=$!
    PIDS+=("$TLS_PID")
    for i in $(seq 100); do grep -q READY "$LOGS/$name.tls.log" 2>/dev/null && break; sleep 0.05; done
    BASE="https://127.0.0.1:$TLS_PORT"
  fi
}

stop_serve() { stop_pids $SERVE_PID ${TLS_PID:-}; }

BATCH='[protocols]
openai_videos = true
fastwan = true
minimax = true
fal = true
fal_director = false
ltx = true
reactor = false
native = true
fal_apps = ["minimax/h3-max", "minimax/h3-turbo", "minimax/h3-draft"]'

DIRECTOR='[protocols]
openai_videos = false
fastwan = false
minimax = false
fal = true
fal_director = true
ltx = false
reactor = false
native = true
fal_apps = ["minimax/h3-max", "minimax/h3-turbo"]'

REACTOR='[protocols]
openai_videos = false
fastwan = false
minimax = false
fal = false
fal_director = false
ltx = false
reactor = true
native = true'

# Client environment: trust the compat CA.
client_env=(SSL_CERT_FILE="$CERT" REQUESTS_CA_BUNDLE="$CERT" NODE_EXTRA_CA_CERTS="$CERT")

# py_suite <name> <script> : a Python suite from tests/compat/suites over TLS.
py_suite() {
  local name="$1" script="$2"
  start_serve "$name" "$BATCH" --tls || return 1
  env "${client_env[@]}" PYTHONPATH="$HERE/suites" timeout 900 "$PY" "$HERE/suites/$script" --base "$BASE" --key "$KEY"
  local rc=$?
  stop_serve
  return $rc
}

suite_openai() { py_suite openai openai_videos.py; }
suite_fastwan() { py_suite fastwan fastwan.py; }
suite_minimax() { py_suite minimax minimax.py; }
suite_ltx() { py_suite ltx ltx.py; }

suite_fal-py() {
  # crates/fastvideo-fal's script terminates TLS itself (its own cert) in
  # front of the plain port and needs the server's public base to be that.
  local tls
  tls="$(free_port)"
  start_serve fal-py "$BATCH" FV_PUBLIC_BASE_URL="https://127.0.0.1:$tls" || return 1
  timeout 900 "$PY" "$ROOT/crates/fastvideo-fal/tests/queue_compat/fal_client_compat.py" \
    --upstream "$SERVE_PORT" --tls-port "$tls" --key "$KEY"
  local rc=$?
  stop_serve
  return $rc
}

suite_fal-js() {
  start_serve fal-js "$BATCH" --tls || return 1
  local png="$WORK/fal-js.png"
  "$PY" -c "import sys; sys.path.insert(0, '$HERE/suites'); from common import png; open('$png','wb').write(png(96, 128))"
  env "${client_env[@]}" timeout 900 node "$ROOT/crates/fastvideo-fal/tests/queue_compat/fal_js_compat.mjs" \
    "$NODE_DIR" "$BASE" "$KEY" "$png"
  local rc=$?
  stop_serve
  return $rc
}

suite_fal-webhook() {
  start_serve fal-webhook "$BATCH" --tls || return 1
  local host="${BASE#https://}"
  env "${client_env[@]}" FAL_KEY="$KEY" FAL_QUEUE_RUN_HOST="$host" FAL_RUN_HOST="$host/run" PYTHONPATH="$HERE/suites" \
    timeout 900 "$PY" "$HERE/suites/fal_webhook.py" --base "$BASE" --key "$KEY"
  local rc=$?
  stop_serve
  return $rc
}

suite_fal-director() {
  start_serve fal-director "$DIRECTOR" || return 1
  timeout 900 node "$HERE/suites/fal_director.mjs" "$NODE_DIR" "$BASE" "$KEY"
  local rc=$?
  stop_serve
  return $rc
}

suite_reactor() {
  local rc=0 mode model
  for mode in av video causal; do
    case $mode in
      av) model=fake-h3-turbo ;;
      video) model=fake-wan ;;
      causal) model=fake-sfwan ;;
    esac
    start_serve "reactor-$mode" "$REACTOR" FV_REACTOR_MODEL="$model" || return 1
    timeout 300 "$PY" "$ROOT/crates/fastvideo-reactor/tests/compat/reactor_sdk_compat.py" --url "$BASE" --mode "$mode" || rc=1
    stop_serve
  done
  return $rc
}

suite_console() {
  # tests/console/smoke.cjs starts its own fv-serve (default config) from
  # FV_SERVE_BIN; playwright resolves from the pinned node_modules.
  FV_SERVE_BIN="$BIN" NODE_PATH="$NODE_DIR/node_modules" timeout 900 node "$ROOT/tests/console/smoke.cjs"
}

# --------------------------------------------------------------------- run
declare -A RESULT
declare -A SECS
failed=0
for s in "${SUITES[@]}"; do
  log "suite $s"
  t0=$SECONDS
  "suite_$s" 2>&1 | tee "$LOGS/$s.client.log"
  rc=${PIPESTATUS[0]}
  SECS[$s]=$((SECONDS - t0))
  if [[ $rc == 0 ]]; then
    RESULT[$s]=PASS
  else
    RESULT[$s]=FAIL
    failed=1
    for f in "$LOGS/$s"*.serve.log; do
      [[ -f "$f" ]] || continue
      echo "--- $(basename "$f") (last 40 lines) ---" >&2
      tail -40 "$f" >&2
    done
  fi
done

printf '\nclient compat summary\n'
for s in "${SUITES[@]}"; do printf '  %-13s %s  (%ss)\n' "$s" "${RESULT[$s]}" "${SECS[$s]}"; done
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "| suite | result | seconds |"
    echo "|---|---|---|"
    for s in "${SUITES[@]}"; do echo "| $s | ${RESULT[$s]} | ${SECS[$s]} |"; done
  } >> "$GITHUB_STEP_SUMMARY"
fi
exit $failed
