#!/usr/bin/env bash
# CPU-only gate for the fv-serve crates (docs/serve/design.md §7): check,
# clippy (warnings are errors) and tests, with no CUDA and no GPU.
# Stages (docs/dev/testing.md):
#   1. build: check + clippy;
#   2. tests: `cargo test` in parallel. Load-tolerant: no wall-clock rate
#      or latency assertions (tests named `realtime_*` are #[ignore]d);
#   3. realtime (--realtime): the `realtime_*` tests (media delivered at
#      24 fps / 48 kHz, decode latency) and the console browser tests, each
#      test binary built first and then run with --test-threads=1, one after
#      the other, with nothing else of this script compiling meanwhile.
#
#   bash scripts/serve/check.sh            # default features (fast CI path)
#   bash scripts/serve/check.sh --realtime # ...then the realtime stage
#   bash scripts/serve/check.sh --realtime-only
#                                          # only the realtime stage
#   FV_SERVE_HEAVY=1 bash scripts/serve/check.sh
#                                          # also build str0m, OpenH264, Opus,
#                                          # reqwest and prost codegen features
#   FV_SERVE_UI=1 bash scripts/serve/check.sh
#                                          # also run the /console browser
#                                          # tests (tests/console/, headless
#                                          # Chromium), in the realtime stage
#   FV_SERVE_STAGES=lint bash scripts/serve/check.sh
#                                          # stages 1/2 one at a time: lint (check,
#                                          # clippy, wasm clippy) or test (cargo
#                                          # test, release.test.sh); default both
#                                          # (CI runs them as parallel jobs)
#   FV_SERVE_CUDA=1 bash scripts/serve/check.sh
#                                          # also type-check the engine's `cuda`
#                                          # feature (compiles, never runs CUDA)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REALTIME="${FV_SERVE_REALTIME:-0}"   # 1: run the realtime stage too; only: only it
for a in "$@"; do
  case "$a" in
    --realtime) REALTIME=1 ;;
    --realtime-only) REALTIME=only ;;
    *) echo "check.sh: unknown argument $a (--realtime | --realtime-only)" >&2; exit 2 ;;
  esac
done

CRATES=(
  fastvideo-protocol
  fastvideo-trace
  fastvideo-engine-service
  fastvideo-media
  fastvideo-webrtc
  fastvideo-serve-kit
  fastvideo-openai-videos
  fastvideo-minimax
  fastvideo-ltxapi
  fastvideo-fal
  fastvideo-reactor
  fastvideo-deploy
  fastvideo-autoscale
  fastvideo-serve
  fastvideo-dispatch-proto
  fastvideo-edge
)
PKGS=()
for c in "${CRATES[@]}"; do PKGS+=(-p "$c"); done

run() { printf '\n==> %s\n' "$*" >&2; "$@"; }

# The fal director's media tests: OpenH264, Opus and libwebp optimized so a
# debug build keeps real time.
FAL_DIRECTOR=(-p fastvideo-fal --features "director,openh264"
  --config 'profile.dev.package.openh264-sys2.opt-level=3'
  --config 'profile.dev.package.openh264.opt-level=3'
  --config 'profile.dev.package.audiopus_sys.opt-level=3'
  --config 'profile.dev.package.libwebp-sys.opt-level=3')

# Stage 3: tests that measure delivery rates or latencies (`realtime_*`,
# #[ignore]d elsewhere) and the browser tests. Built first, then run one
# binary and one test at a time.
rt_build() { run cargo test "$@" --no-run; }
rt_run() { run cargo test "$@" -- --ignored --test-threads=1 realtime_; }
realtime_stage() {
  printf '\n==> realtime stage (serialized)\n' >&2
  local step
  for step in rt_build rt_run; do
    "$step" -p fastvideo-reactor --test runtime
    "$step" "${FAL_DIRECTOR[@]}" --test director_e2e
    "$step" -p fastvideo-media --test decode_pipe
  done
  if [[ "${FV_SERVE_UI:-0}" == "1" ]]; then
    # Headless Chromium tests of the /console pages against
    # `fv-serve --features fake,encoders` (needs node + the playwright npm
    # package); the script builds fv-serve before it starts a browser.
    run bash tests/console/run.sh
  fi
}

if [[ "$REALTIME" == "only" ]]; then
  realtime_stage
  printf '\nfv-serve crates: realtime stage OK\n' >&2
  exit 0
fi

STAGES=" ${FV_SERVE_STAGES:-lint test} "
stage() { [[ "$STAGES" == *" $1 "* ]]; }
if stage lint; then
  run cargo check "${PKGS[@]}" --all-targets
  run cargo clippy "${PKGS[@]}" --all-targets --no-deps -- -D warnings
fi
stage test && run cargo test "${PKGS[@]}"
# The Cloudflare Worker / Durable Object dispatcher builds for wasm only
# (docs/serve/gateway-cloudflare.md; scripts/serve/cf-edge.sh deploys it).
stage lint && run cargo clippy -p fastvideo-dispatch-proto -p fastvideo-edge --target wasm32-unknown-unknown --no-deps -- -D warnings
# The release / deployment scripts against a mocked API (docs/serve/releases.md;
# skips without python3).
stage test && run bash scripts/serve/tests/release.test.sh
# The trace waterfall analysis (console/trace.js; docs/serve/tracing.md).
stage test && run node --test scripts/serve/tests/trace-analyze.test.mjs

if [[ "${FV_SERVE_HEAVY:-0}" == "1" ]]; then
  run cargo check -p fastvideo-serve --features full,fake --all-targets
  run cargo test -p fastvideo-reactor --features proto-codegen --lib
  run cargo clippy -p fastvideo-serve -p fastvideo-webrtc -p fastvideo-media -p fastvideo-serve-kit -p fastvideo-deploy \
    -p fastvideo-fal --features fastvideo-serve/full --all-targets --no-deps -- -D warnings
  # serve-kit's HTTP fetch / callback-receiver tests only exist with `fetch`.
  run cargo test -p fastvideo-serve-kit --features fetch
  # fal WMA director (WP-14): protocol units, WebRTC end to end against the
  # fake engine (A/V at 24 fps / 48 kHz; OpenH264 and libwebp optimized so a
  # debug build keeps real time, or ffmpeg+libx264 via FV_FFMPEG), the
  # fv-serve mount, and the Chromium `fal.realtime.open` compat test when
  # FV_FAL_JS_DIR is set (see crates/fastvideo-fal/tests/director_browser.rs).
  run cargo test "${FAL_DIRECTOR[@]}"
  run cargo test -p fastvideo-serve --features webrtc,fake --test director
  # WebRTC ingest (design §5.11): receiving answers, the bitrate cap, PLIs,
  # the decode thread into the input rings (VP8 via ffmpeg libvpx; skips
  # without it). The Reactor duplex, engine echo and native WHIP ingest
  # tests run in the default pass above.
  run cargo test -p fastvideo-webrtc --features str0m,opus --test ingest
  # fv-serve with outbound HTTP: D1 over HTTP, S3/R2 read-back for LTX v1.
  # (serve-kit's d1_live smoke test runs above when CLOUDFLARE_API_KEY or
  # FV_CF_API_TOKEN is set, and skips otherwise.)
  run cargo test -p fastvideo-serve --features http-client
  # Native /fv/v1/streams: fake engine → pacer → OpenH264/Opus → WHIP (in
  # process endpoint); scripts/serve/whip-e2e.sh adds MediaMTX + WHEP.
  run cargo test -p fastvideo-serve --features full --test streams_whip
  # The Runpod worker over reqwest against the queue simulator on TCP.
  run cargo test -p fastvideo-deploy --features runpod
  # Autoscaler: Runpod providers, D1 lease over HTTPS, fv-autoscale.
  run cargo clippy -p fastvideo-autoscale --features runpod,d1-http --all-targets --no-deps -- -D warnings
  run cargo test -p fastvideo-autoscale --features runpod,d1-http
fi

if [[ "${FV_SERVE_CUDA:-0}" == "1" ]]; then
  # Type-checking cudarc needs a target CUDA version; nothing runs on a GPU.
  CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-13000}" \
    run cargo check -p fastvideo-engine-service --features cuda
fi

# Last, after every compile above: the realtime stage (--realtime), or at
# least the browser tests when FV_SERVE_UI=1 asks for them.
if [[ "$REALTIME" == "1" ]]; then
  realtime_stage
elif [[ "${FV_SERVE_UI:-0}" == "1" ]]; then
  run bash tests/console/run.sh
fi

printf '\nfv-serve crates: OK\n' >&2
