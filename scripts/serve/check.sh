#!/usr/bin/env bash
# CPU-only gate for the fv-serve crates (docs/serve/design.md §7): check,
# clippy (warnings are errors) and tests, with no CUDA and no GPU.
#
#   bash scripts/serve/check.sh            # default features (fast CI path)
#   FV_SERVE_HEAVY=1 bash scripts/serve/check.sh
#                                          # also build str0m, OpenH264, Opus,
#                                          # reqwest and prost codegen features
#   FV_SERVE_UI=1 bash scripts/serve/check.sh
#                                          # also run the /console browser smoke
#                                          # test (tests/console/, headless Chromium)
#   FV_SERVE_CUDA=1 bash scripts/serve/check.sh
#                                          # also type-check the engine's `cuda`
#                                          # feature (compiles, never runs CUDA)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CRATES=(
  fastvideo-protocol
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

run cargo check "${PKGS[@]}" --all-targets
run cargo clippy "${PKGS[@]}" --all-targets --no-deps -- -D warnings
run cargo test "${PKGS[@]}"
# The Cloudflare Worker / Durable Object dispatcher builds for wasm only
# (docs/serve/gateway-cloudflare.md; scripts/serve/cf-edge.sh deploys it).
run cargo clippy -p fastvideo-dispatch-proto -p fastvideo-edge --target wasm32-unknown-unknown --no-deps -- -D warnings
# The release / deployment scripts against a mocked API (docs/serve/releases.md;
# skips without python3).
run bash scripts/serve/tests/release.test.sh

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
  run cargo test -p fastvideo-fal --features director,openh264 \
    --config 'profile.dev.package.openh264-sys2.opt-level=3' \
    --config 'profile.dev.package.openh264.opt-level=3' \
    --config 'profile.dev.package.audiopus_sys.opt-level=3' \
    --config 'profile.dev.package.libwebp-sys.opt-level=3'
  run cargo test -p fastvideo-serve --features webrtc,fake --test director
  # fv-serve with outbound HTTP: D1 over HTTP, S3/R2 read-back for LTX v1.
  # (serve-kit's d1_live smoke test runs above when CLOUDFLARE_API_KEY or
  # FV_CF_API_TOKEN is set, and skips otherwise.)
  run cargo test -p fastvideo-serve --features http-client
  # Native /fv/v1/streams: fake engine → pacer → OpenH264/Opus → WHIP (in
  # process endpoint); scripts/serve/whip-e2e.sh adds MediaMTX + WHEP.
  run cargo test -p fastvideo-serve --features full --test streams_whip
  # The Runpod worker over reqwest against the queue simulator on TCP.
  run cargo test -p fastvideo-deploy --features runpod
  # Autoscaler: Runpod providers, gateway poller, D1 lease over HTTPS, fv-autoscale.
  run cargo clippy -p fastvideo-autoscale --features runpod,d1-http --all-targets --no-deps -- -D warnings
  run cargo test -p fastvideo-autoscale --features runpod,d1-http
fi

if [[ "${FV_SERVE_UI:-0}" == "1" ]]; then
  # Headless Chromium smoke test of the /console pages against
  # `fv-serve --features fake` (needs node + the playwright npm package).
  run bash tests/console/run.sh
fi

if [[ "${FV_SERVE_CUDA:-0}" == "1" ]]; then
  # Type-checking cudarc needs a target CUDA version; nothing runs on a GPU.
  CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-13000}" \
    run cargo check -p fastvideo-engine-service --features cuda
fi

printf '\nfv-serve crates: OK\n' >&2
