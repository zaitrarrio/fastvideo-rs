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
  fastvideo-serve
)
PKGS=()
for c in "${CRATES[@]}"; do PKGS+=(-p "$c"); done

run() { printf '\n==> %s\n' "$*" >&2; "$@"; }

run cargo check "${PKGS[@]}" --all-targets
run cargo clippy "${PKGS[@]}" --all-targets --no-deps -- -D warnings
run cargo test "${PKGS[@]}"

if [[ "${FV_SERVE_HEAVY:-0}" == "1" ]]; then
  run cargo check -p fastvideo-serve --features full,fake --all-targets
  run cargo check -p fastvideo-reactor --features proto-codegen
  run cargo clippy -p fastvideo-serve -p fastvideo-webrtc -p fastvideo-media -p fastvideo-serve-kit -p fastvideo-deploy \
    --features fastvideo-serve/full --all-targets --no-deps -- -D warnings
  # serve-kit's HTTP fetch / callback-receiver tests only exist with `fetch`.
  run cargo test -p fastvideo-serve-kit --features fetch
  # fv-serve with outbound HTTP: D1 over HTTP, S3/R2 read-back for LTX v1.
  # (serve-kit's d1_live smoke test runs above when CLOUDFLARE_API_KEY or
  # FV_CF_API_TOKEN is set, and skips otherwise.)
  run cargo test -p fastvideo-serve --features http-client
  # The Runpod worker over reqwest against the queue simulator on TCP.
  run cargo test -p fastvideo-deploy --features runpod
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
