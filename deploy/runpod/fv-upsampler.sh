#!/bin/bash
# fv-upsampler: the prompt upsampler's local sidecar (docs/serve/upsampler-local.md). llama.cpp's llama-server
# serves a local Qwen3-VL-4B-Instruct GGUF (language model + vision projector, --mmproj) on 127.0.0.1 next to
# fv-serve, on the same GPU; fv-serve's upsampler talks to it as an OpenAI-compatible /v1/chat/completions
# (FV_UPSAMPLER_BASE_URL, FV_UPSAMPLER_MODEL; FV_UPSAMPLER_VISION=1 when the projector is loaded: image_url content
# parts work, as data: URLs or http(s) URLs llama-server downloads itself, at most 10 MB within 10 s).
#
# Shipped in the CUDA serve images as /opt/fastvideo-rs/bin/fv-upsampler, beside /opt/fastvideo-rs/bin/llama-server.
# The image entrypoint (fv-entry) and fv-control's boots run it when FV_UPSAMPLER_LOCAL=1:
#
#   fv-upsampler start   returns at once; in the background: find the GGUF tree (or fetch it from the R2 weights
#                        store, the same plan + r2-fetch.py path as the weight trees), start llama-server, wait for
#                        /health and one 1-token completion, then write $S/upsampler.env (the exports) or
#                        $S/upsampler.failed (why). llama-server is restarted if it dies (FV_UPSAMPLER_RESTARTS).
#   fv-upsampler wait    waits for start's outcome (FV_UPSAMPLER_WAIT_S) and prints `export …` lines for the boot
#                        to eval before it execs fv-serve; logs a failure and prints nothing then (fv-serve starts
#                        anyway and its upsampler falls back). Exit status 0 either way.
#   fv-upsampler run     start + wait in one call (for a shell that only wants the exports).
#
# Where the GGUF comes from, in order: FV_UPSAMPLER_GGUF (a file); a complete tree <root>/$FV_UPSAMPLER_TREE (its
# .complete, and .fv-revision = FV_UPSAMPLER_REV when that is set) under FV_UPSAMPLER_ROOT, FV_WEIGHTS,
# /workspace/weights or /runpod-volume/weights; else fetched with r2-fetch.py into FV_UPSAMPLER_ROOT (default: FV_WEIGHTS
# when the pod's weights are on its own disk, weights_source r2 / hub; else $S/upsampler-weights on the container
# disk: a mounted weights volume is never written). The fetch asks FV_WEIGHTS_PLAN_URL?set=upsampler with the pod's
# token (FV_WEIGHTS_TOKEN on serverless workers, else FV_ENDPOINT_REPORT_TOKEN): fv-control answers with the
# upsampler tree only, for any weights source.
#
# Env (defaults): FV_UPSAMPLER_PORT (18080), FV_UPSAMPLER_MODEL (qwen3-vl-4b-instruct, the server's alias),
# FV_UPSAMPLER_TREE (qwen3-vl-4b-instruct-gguf: the model is its non-mmproj *.gguf, the projector its mmproj-*.gguf),
# FV_UPSAMPLER_GGUF / FV_UPSAMPLER_MMPROJ (explicit files), FV_UPSAMPLER_VISION (1: load the projector when the tree
# has one; 0: text only), FV_UPSAMPLER_CTX (65536: the KV of all slots, split evenly, so
# 32768 tokens per slot: a director session's 50k-character premise + 12-50 prior segments + the entity registry),
# FV_UPSAMPLER_PARALLEL (2), FV_UPSAMPLER_KV_TYPE (q8_0: ~4.8 GB of KV at 65536 tokens, flash attention on),
# FV_UPSAMPLER_CACHE_REUSE (256: reuse cached KV chunks of at least that many tokens after a shifted prefix),
# FV_UPSAMPLER_CACHE_RAM_MIB (4096: host-RAM prompt cache), FV_UPSAMPLER_ARGS (extra llama-server flags),
# FV_UPSAMPLER_START_S (180: fetch + load), FV_UPSAMPLER_WAIT_S (240), FV_UPSAMPLER_RESTARTS (3),
# FV_UPSAMPLER_LOG (the launcher's log: default stderr), FV_LLAMA_SERVER (/opt/fastvideo-rs/bin/llama-server), S = FV_UPSAMPLER_STATE or FV_STATE_DIR or /fvstate.
set -u
S="${FV_UPSAMPLER_STATE:-${FV_STATE_DIR:-/fvstate}}"
BIN="${FV_LLAMA_SERVER:-/opt/fastvideo-rs/bin/llama-server}"
PORT="${FV_UPSAMPLER_PORT:-18080}"
MODEL="${FV_UPSAMPLER_MODEL:-qwen3-vl-4b-instruct}"
TREE="${FV_UPSAMPLER_TREE:-qwen3-vl-4b-instruct-gguf}"
MMPROJ=""
REV="${FV_UPSAMPLER_REV:-}"
ENVF="$S/upsampler.env"
FAILF="$S/upsampler.failed"
STEPF="$S/upsampler.steps"
LOG="$S/llama-server.log"

ts() { date -u +%H:%M:%S; }
say() { echo "[fv-upsampler] $(ts) $*" >&2; }
el() { awk -v a="$1" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }'; }
fail() {
  say "FAILED: $1"
  if [ -f "$S/llama-server.pid" ]; then kill "$(cat "$S/llama-server.pid")" 2>/dev/null || true; fi
  if [ -f "$S/llama-server.pid" ]; then kill "$(cat "$S/llama-server.pid")" 2>/dev/null || true; fi
  printf '%s\n' "$1" > "$FAILF.part" && mv -f "$FAILF.part" "$FAILF"
  exit 1
}

# A complete tree at its revision under one of the roots -> its directory.
find_tree() {
  local r d
  for r in "${FV_UPSAMPLER_ROOT:-}" "${FV_WEIGHTS:-}" /workspace/weights /runpod-volume/weights; do
    [ -n "$r" ] || continue
    d="$r/$TREE"
    [ -f "$d/.complete" ] || continue
    if [ -n "$REV" ] && [ "$(cat "$d/.fv-revision" 2>/dev/null || true)" != "$REV" ]; then continue; fi
    printf '%s\n' "$d"
    return 0
  done
  return 1
}

# The fetcher: the image's copy when it is the one fv-control pins, else fv-control's (checked the same way).
fetcher() {
  local f="${FV_R2_FETCH_IMAGE:-/opt/fastvideo-rs/bin/r2-fetch.py}"
  if [ -f "$f" ] && { [ -z "${FV_R2_FETCH_SHA256:-}" ] || echo "$FV_R2_FETCH_SHA256  $f" | sha256sum -c - >/dev/null 2>&1; }; then
    printf '%s\n' "$f"
    return 0
  fi
  [ -n "${FV_R2_FETCH_URL:-}" ] && [ -n "${FV_R2_FETCH_SHA256:-}" ] || return 1
  f="${FV_R2_TOOLS:-/opt/fv}/r2-fetch.py"
  mkdir -p "$(dirname "$f")"
  curl -fsS --connect-timeout 5 --max-time 30 --retry 4 --retry-delay 1 --retry-all-errors -o "$f" "$FV_R2_FETCH_URL" || return 1
  echo "$FV_R2_FETCH_SHA256  $f" | sha256sum -c - >/dev/null 2>&1 || return 1
  printf '%s\n' "$f"
}

fetch_tree() {
  local root tok f url sep
  root="${FV_UPSAMPLER_ROOT:-}"
  if [ -z "$root" ]; then
    case "${FV_WEIGHTS_SOURCE:-}" in
      r2 | hub) root="${FV_WEIGHTS:-$S/upsampler-weights}" ;;
      *) root="$S/upsampler-weights" ;;
    esac
  fi
  [ -n "${FV_WEIGHTS_PLAN_URL:-}" ] || fail "no $TREE tree on this host and no FV_WEIGHTS_PLAN_URL to fetch it from the R2 weights store"
  command -v python3 >/dev/null 2>&1 || fail "python3 is not in the image (r2-fetch.py)"
  f="$(fetcher)" || fail "r2-fetch.py: neither the image's copy nor fv-control's matches FV_R2_FETCH_SHA256"
  tok=FV_ENDPOINT_REPORT_TOKEN
  [ -n "${FV_WEIGHTS_TOKEN:-}" ] && tok=FV_WEIGHTS_TOKEN
  case "$FV_WEIGHTS_PLAN_URL" in *\?*) sep="&" ;; *) sep="?" ;; esac
  url="${FV_WEIGHTS_PLAN_URL}${sep}set=upsampler"
  mkdir -p "$root"
  say "fetching $TREE from the R2 weights store into $root"
  # No --report-url: fv-control's weights progress is the main fetch's.
  python3 "$f" --root "$root" --plan-url "$url" --token-env "$tok" --pod "${FV_POD_ID:-${RUNPOD_POD_ID:-}}" \
    --status "$S/upsampler-fetch.json" --concurrency "${FV_UPSAMPLER_R2_CONCURRENCY:-16}" --min-free-gb 2 \
    >"$S/upsampler-fetch.log" 2>&1 || fail "fetch: $(grep -E 'FAILED|Error|error' "$S/upsampler-fetch.log" | tail -1 | cut -c1-200)"
  [ -f "$root/$TREE/.complete" ] || fail "fetch: $root/$TREE is not complete after the fetch"
  FV_UPSAMPLER_ROOT="$root"
}

healthy() { curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

# One 1-token completion: the CUDA kernels run on this GPU (the arch is in the binary), not just the load.
probe() {
  curl -fsS --max-time 30 -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1}" \
    "http://127.0.0.1:$PORT/v1/chat/completions" >/dev/null 2>&1
}

launch() {
  local mm=()
  [ -n "$MMPROJ" ] && mm=(--mmproj "$MMPROJ")
  # shellcheck disable=SC2086 # FV_UPSAMPLER_ARGS: extra flags, split on purpose
  "$BIN" -m "$1" "${mm[@]}" --host 127.0.0.1 --port "$PORT" --alias "$MODEL" \
    -ngl 99 -c "${FV_UPSAMPLER_CTX:-65536}" -np "${FV_UPSAMPLER_PARALLEL:-2}" --no-kv-unified \
    -fa on -ctk "${FV_UPSAMPLER_KV_TYPE:-q8_0}" -ctv "${FV_UPSAMPLER_KV_TYPE:-q8_0}" \
    --cache-reuse "${FV_UPSAMPLER_CACHE_REUSE:-256}" --cache-ram "${FV_UPSAMPLER_CACHE_RAM_MIB:-4096}" \
    --fit off --no-webui ${FV_UPSAMPLER_ARGS:-} \
    >>"$LOG" 2>&1 &
  echo $! > "$S/llama-server.pid"
}

start_bg() {
  local t0 t gguf d pid n=0 restarts="${FV_UPSAMPLER_RESTARTS:-3}" deadline
  t0="$(date +%s.%N)"
  : > "$STEPF"
  [ -x "$BIN" ] || fail "$BIN is not in this image"
  t="$(date +%s.%N)"
  gguf="${FV_UPSAMPLER_GGUF:-}"
  if [ -z "$gguf" ]; then
    if ! d="$(find_tree)"; then
      fetch_tree
      d="$(find_tree)" || fail "$TREE: not found after the fetch"
      echo "fetch $(el "$t")" >> "$STEPF"
    else
      say "using $d"
    fi
    gguf="$(find "$d" -maxdepth 1 -name '*.gguf' ! -name 'mmproj*' | sort | head -n 1)"
    MMPROJ="$(find "$d" -maxdepth 1 -name 'mmproj*.gguf' | sort | grep -i f16 | head -n 1)"
    [ -n "$MMPROJ" ] || MMPROJ="$(find "$d" -maxdepth 1 -name 'mmproj*.gguf' | sort | head -n 1)"
  fi
  [ -n "${FV_UPSAMPLER_MMPROJ:-}" ] && MMPROJ="$FV_UPSAMPLER_MMPROJ"
  [ "${FV_UPSAMPLER_VISION:-1}" = 0 ] && MMPROJ=""
  [ -n "$gguf" ] && [ -f "$gguf" ] || fail "no .gguf file (FV_UPSAMPLER_GGUF or the $TREE tree)"
  t="$(date +%s.%N)"
  say "llama-server on 127.0.0.1:$PORT: $gguf${MMPROJ:+ + vision projector $MMPROJ}"
  launch "$gguf"
  deadline=$(( $(date +%s) + ${FV_UPSAMPLER_START_S:-180} ))
  pid="$(cat "$S/llama-server.pid")"
  until healthy; do
    kill -0 "$pid" 2>/dev/null || fail "llama-server exited while loading: $(tail -n 3 "$LOG" | tr '\n' ' ' | cut -c1-240)"
    [ "$(date +%s)" -lt "$deadline" ] || fail "llama-server not healthy after ${FV_UPSAMPLER_START_S:-180}s: $(tail -n 2 "$LOG" | tr '\n' ' ' | cut -c1-200)"
    sleep 0.5
  done
  probe || fail "llama-server is up but a 1-token completion failed: $(tail -n 3 "$LOG" | tr '\n' ' ' | cut -c1-240)"
  echo "load $(el "$t")" >> "$STEPF"
  [ -n "$MMPROJ" ] && echo "vision on" >> "$STEPF"
  echo "total $(el "$t0")" >> "$STEPF"
  {
    echo "export FV_UPSAMPLER_BASE_URL='http://127.0.0.1:$PORT/v1'"
    echo "export FV_UPSAMPLER_MODEL='$MODEL'"
    echo "export FV_UPSAMPLER_VISION='$([ -n "$MMPROJ" ] && echo 1 || echo 0)'"
  } > "$ENVF.part" && mv -f "$ENVF.part" "$ENVF"
  say "ready in $(el "$t0")s ($(tr '\n' ' ' < "$STEPF" | sed 's/ $//'))"
  # Keep it up: a crashed llama-server is restarted (requests in between fall back in fv-serve).
  while :; do
    wait "$pid" 2>/dev/null
    while kill -0 "$pid" 2>/dev/null; do sleep 5; done
    n=$((n + 1))
    if [ "$n" -gt "$restarts" ]; then say "llama-server exited $n times: not restarted again (upsampling falls back)"; exit 1; fi
    say "llama-server exited ($(tail -n 1 "$LOG" | cut -c1-160)): restart $n/$restarts"
    sleep 2
    launch "$gguf"
    pid="$(cat "$S/llama-server.pid")"
  done
}

cmd_start() {
  mkdir -p "$S"
  if [ -f "$S/upsampler.pid" ] && kill -0 "$(cat "$S/upsampler.pid")" 2>/dev/null; then say "already starting / running"; return 0; fi
  rm -f "$ENVF" "$FAILF" "$STEPF"
  date +%s.%N > "$S/upsampler.t0"
  # stdout -> stderr (the container log): a caller's $(fv-upsampler run) must not wait on the background launcher;
  # FV_UPSAMPLER_LOG: a file instead (a caller that waits for its stderr to close).
  if [ -n "${FV_UPSAMPLER_LOG:-}" ]; then ( start_bg ) </dev/null >>"$FV_UPSAMPLER_LOG" 2>&1 &
  else ( start_bg ) </dev/null >&2 &
  fi
  echo $! > "$S/upsampler.pid"
}

cmd_wait() {
  local t0 deadline st
  t0="$(cat "$S/upsampler.t0" 2>/dev/null || date +%s.%N)"
  deadline=$(( $(date +%s) + ${FV_UPSAMPLER_WAIT_S:-240} ))
  while [ ! -f "$ENVF" ] && [ ! -f "$FAILF" ]; do
    if ! { [ -f "$S/upsampler.pid" ] && kill -0 "$(cat "$S/upsampler.pid")" 2>/dev/null; }; then
      sleep 0.2
      [ -f "$ENVF" ] || [ -f "$FAILF" ] || printf '%s\n' "the sidecar's launcher exited without a result" > "$FAILF"
      break
    fi
    [ "$(date +%s)" -lt "$deadline" ] || break
    sleep 0.5
  done
  st="$(tr '\n' ' ' < "$STEPF" 2>/dev/null | sed 's/ $//')"
  if [ -f "$ENVF" ]; then
    echo "[fv-boot] $(ts) step upsampler: $(el "$t0")s (llama-server 127.0.0.1:$PORT, $MODEL${st:+; $st})" >&2
    cat "$ENVF"
  elif [ -f "$FAILF" ]; then
    echo "[fv-boot] $(ts) upsampler: FAILED: $(cut -c1-300 "$FAILF"): fv-serve starts without it; upsampling falls back" >&2
  else
    # Still loading after the wait: fv-serve gets the URL anyway; until llama-server answers, its requests fall back.
    echo "[fv-boot] $(ts) upsampler: not ready after ${FV_UPSAMPLER_WAIT_S:-240}s (still starting): fv-serve starts with it; upsampling falls back until it answers" >&2
    echo "export FV_UPSAMPLER_BASE_URL='http://127.0.0.1:$PORT/v1'"
    echo "export FV_UPSAMPLER_MODEL='$MODEL'"
  fi
  return 0
}

case "${1:-}" in
  start) cmd_start ;;
  wait) cmd_wait ;;
  run) cmd_start && cmd_wait ;;
  *) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
