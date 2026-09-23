#!/usr/bin/env bash
# Build the release fv-gpucheck binary — with ahead-of-time cubins for every
# SM — on a cheap rented Linux box instead of this machine.
#
#   build-remote.sh            rent → install CUDA 13.0 nvcc + Rust → build → pull dist → destroy
#
# Why not here: the Mac has no nvcc and its Docker builds are emulated x86.
# Why not CI: CI does this too (gpucheck-runtime-image.yml, every branch), but
# a build box gives the same artifact in minutes from an uncommitted tree.
#
# The recipe is docker/cuda-builder.Dockerfile, run on a bare ubuntu:22.04
# instance, so the two stay the same by construction: NVIDIA's apt packages,
# rustup stable, CUDARC_CUDA_VERSION=13000, NVCC pointed at the toolkit.
#
# Output: artifacts/gpucheck/dist/fv-gpucheck + .build-id, the same files
# `docker.sh dist` produces, so validate.sh picks it up unchanged.
set -euo pipefail
# shellcheck source=scripts/gpu/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
# Rental helpers (offers, create_instance, wait_ready, cleanup) without dispatch.
FV_SOURCE_ONLY=1 source "$(dirname "${BASH_SOURCE[0]}")/validate.sh"
# validate.sh pulls a validation output dir after stages and on exit; a build
# box has none, so make both a no-op rather than an "artifact pull failed" line.
pull_outputs() { :; }
FV_NO_ARTIFACTS=1

DIST="$FV_ROOT/artifacts/gpucheck/dist"
IMAGE="${FV_BUILD_IMAGE:-ubuntu:22.04}"
FV_LABEL_PREFIX="${FV_LABEL_PREFIX:-fvbuild}"
REMOTE=/workspace/src
INSTANCE=""; HOST=""; PORT=""

require_tools vastai jq rsync git
vast_check_auth

build_id="$(cd "$FV_ROOT" && bash scripts/gpu/docker.sh build-id)"
if [[ -x "$DIST/fv-gpucheck" && "$(cat "$DIST/fv-gpucheck.build-id" 2>/dev/null)" == "$build_id" && -x "$DIST/hf-fm" ]]; then
  log "dist already matches build $build_id (fv-gpucheck + hf-fm)"; exit 0
fi

max_dph="${MAX_DPH:-$(tier_max_dph build)}"
offers="$(offers_json build | jq -c --argjson max "$max_dph" '[.[] | select(.dph_total <= $max)] | .[0:3]')"
[[ "$(jq length <<<"$offers")" -gt 0 ]] || die "no build offers under \$$max_dph/hr"

# Reuse validate.sh's cleanup trap so a failed build still destroys the box.
RUN_DIR="$(mktemp -d)"; T_START=$(date +%s); OWN_INSTANCE=1; KEEP=0; CURRENT_MACHINE=""; LAST_STAGE=""
trap cleanup EXIT
trap 'cleanup 130' INT

for offer in $(jq -r '.[].id' <<<"$offers"); do
  CURRENT_MACHINE="$(jq -r --arg id "$offer" '.[] | select(.id == ($id|tonumber)) | .machine_id' <<<"$offers")"
  log "offer $offer: $(jq -r --arg id "$offer" '.[] | select(.id == ($id|tonumber)) | "\(.gpu_name) $\(.dph_total)/hr \(.cpu_ram)GB ram"' <<<"$offers")"
  create_instance build "$offer" || continue
  if wait_ready; then break; fi
  bad_host_record "$CURRENT_MACHINE" "build box never booted"
  vast_destroy "$INSTANCE" || true; INSTANCE=""
done
[[ -n "$INSTANCE" && -n "$HOST" ]] || die "no build box came up"

log "installing CUDA 13.0 nvcc + Rust (same recipe as docker/cuda-builder.Dockerfile)"
fv_ssh "$HOST" "$PORT" 'set -euo pipefail; export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    build-essential pkg-config libssl-dev clang curl wget ca-certificates git rsync >/dev/null
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null && rm cuda-keyring_1.1-1_all.deb
  apt-get update -qq && apt-get install -y -qq --no-install-recommends cuda-nvcc-13-0 cuda-nvrtc-13-0 cuda-nvrtc-dev-13-0 >/dev/null
  curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
  /usr/local/cuda-13.0/bin/nvcc --version | tail -1; ~/.cargo/bin/rustc --version'

log "syncing source (git-tracked files only)"
# rsync creates the last path component only; a bare ubuntu image has no
# /workspace, and rsync reports that as a receiver file-IO error (code 11).
fv_ssh "$HOST" "$PORT" "mkdir -p $REMOTE"
# Submodule gitlinks are not file contents. Sync the repo, then the vendored
# Rust-to-PTX trees (cuda-oxide, cutile-rs) without their .git directories.
(cd "$FV_ROOT" && git ls-files -z -- . ':!third_party' | rsync -az --files-from=- --from0 -e "ssh -p $PORT -o StrictHostKeyChecking=no" . "root@$HOST:$REMOTE/") >/dev/null
if [[ -d "$FV_ROOT/third_party/cuda-oxide" ]]; then
  log "syncing vendored Rust-to-PTX toolchain"
  rsync -az --exclude '.git' -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "$FV_ROOT/third_party/" "root@$HOST:$REMOTE/third_party/" >/dev/null
fi

log "building release with ahead-of-time cubins + hf-fm"
fv_ssh "$HOST" "$PORT" "set -euo pipefail; cd $REMOTE
  export PATH=\$HOME/.cargo/bin:/usr/local/cuda-13.0/bin:\$PATH NVCC=/usr/local/cuda-13.0/bin/nvcc \
         CUDARC_CUDA_VERSION=13000 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
  cargo build --release -p fastvideo-gpucheck --features cuda 2>&1 | tee build.log | grep -E 'warning: fastvideo-cudarc|Compiling fastvideo|Finished|error' || true
  grep -q 'embedded cubins for sm' build.log || { echo 'build.rs did not embed cubins'; exit 1; }
  echo '$build_id' > target/release/fv-gpucheck.build-id
  ./target/release/fv-gpucheck --out /tmp/gate nvrtc >/dev/null && python3 -c \"import json;print('aot_sms', json.load(open('/tmp/gate/nvrtc.json'))['context'].get('aot_sms'))\"
  cargo install hf-fetch-model --features cli --root /tmp/hf-fm-root 2>&1 | tee -a build.log | grep -E 'Installed|Compiling hf|Finished|error' || true
  test -x /tmp/hf-fm-root/bin/hf-fm"

mkdir -p "$DIST"
fv_rsync_from "$HOST" "$PORT" "$REMOTE/target/release/fv-gpucheck" "$DIST/fv-gpucheck" >/dev/null
fv_rsync_from "$HOST" "$PORT" "$REMOTE/target/release/fv-gpucheck.build-id" "$DIST/fv-gpucheck.build-id" >/dev/null
fv_rsync_from "$HOST" "$PORT" "/tmp/hf-fm-root/bin/hf-fm" "$DIST/hf-fm" >/dev/null
fv_rsync_from "$HOST" "$PORT" "/tmp/hf-fm-root/bin/hf-fetch-model" "$DIST/hf-fetch-model" >/dev/null || true
chmod +x "$DIST/fv-gpucheck" "$DIST/hf-fm"
[[ -x "$DIST/hf-fetch-model" ]] && chmod +x "$DIST/hf-fetch-model"
log "dist: $DIST/fv-gpucheck ($(du -h "$DIST/fv-gpucheck" | cut -f1), build $build_id) + hf-fm in $(( $(date +%s) - T_START ))s"
