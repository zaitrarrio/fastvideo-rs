#!/usr/bin/env bash
# Build and run fv-gpucheck in Docker instead of on the host.
#
#   docker.sh builder          build the builder image (Rust + NVRTC, no GPU)
#   docker.sh test             unit tests (gpucheck + cudarc) in the builder
#   docker.sh nvrtc            compile every NVRTC kernel for sm 7.5–12.0 (no GPU)
#   docker.sh dist             release binary → artifacts/gpucheck/dist/ (shipped to rented boxes)
#   docker.sh refs [--parity]  CPU-path reference dumps → artifacts/gpucheck/refs/ (saves billed GPU idle time)
#   docker.sh image            slim runtime (Ubuntu + CUDA 13 libs + binary; no PyTorch)
#   docker.sh vast-image       Vast image FROM vastai/pytorch (runtime target)
#   docker.sh vast-oracle      Vast image FROM vastai/pytorch + transformers/diffusers
#   docker.sh gpu <args...>    run fv-gpucheck on a local NVIDIA GPU: docker run --gpus all
#
# Everything is keyed by a build id (hash of the Rust sources), so stale
# binaries and references are rebuilt instead of silently reused.
set -euo pipefail
# shellcheck source=scripts/gpu/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

BUILDER_IMAGE="${FV_BUILDER_IMAGE:-fastvideo-rs/gpucheck-builder:cu130}"
RUNTIME_IMAGE="${FV_RUNTIME_IMAGE:-fastvideo-rs-runtime:local}"
VAST_IMAGE="${FV_VAST_IMAGE:-fastvideo-rs-vast:local}"
VAST_ORACLE_IMAGE="${FV_VAST_ORACLE_IMAGE:-fastvideo-rs-vast-oracle:local}"
DOCKERFILE="$FV_ROOT/docker/gpucheck.Dockerfile"
VAST_DOCKERFILE="$FV_ROOT/docker/vast-pytorch.Dockerfile"
# Pin so local builds match CI; override with FV_VAST_PYTORCH_IMAGE.
VAST_PYTORCH_BASE="${FV_VAST_PYTORCH_IMAGE:-vastai/pytorch:cuda-13.0.3-auto}"
DIST="$FV_ROOT/artifacts/gpucheck/dist"
REFS="$FV_ROOT/artifacts/gpucheck/refs"
LOCAL_OUT="$FV_ROOT/artifacts/gpucheck/local"
PLATFORM=linux/amd64
BASE_REPO="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

require_docker() {
  require_tools docker
  docker info >/dev/null 2>&1 || die "Docker daemon not running (start Docker Desktop)"
}

# Hash of everything that affects the binary (sources + builder image/profile).
fv_build_id() {
  (cd "$FV_ROOT" && git ls-files -z -co --exclude-standard -- crates Cargo.toml Cargo.lock rust-toolchain.toml docker/gpucheck.Dockerfile docker/vast-pytorch.Dockerfile \
    | LC_ALL=C sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-16)
}

cmd_builder() {
  require_docker
  log "building $BUILDER_IMAGE"
  docker build --platform "$PLATFORM" -f "$DOCKERFILE" --target builder -t "$BUILDER_IMAGE" "$FV_ROOT/docker" >&2
}

# in_builder <bash command>: repo bind-mounted at /src (with .env masked),
# cargo registry and target dir in named volumes so rebuilds are incremental.
in_builder() {
  require_docker
  docker image inspect "$BUILDER_IMAGE" >/dev/null 2>&1 || cmd_builder
  local extra=(--platform "$PLATFORM")
  [[ -t 1 ]] && extra+=(-t)
  # Keep secrets out of the container even though the repo is mounted.
  [[ -f "$FV_ROOT/.env" ]] && extra+=(-v /dev/null:/src/.env:ro)
  extra+=(${FV_DOCKER_EXTRA[@]+"${FV_DOCKER_EXTRA[@]}"})
  docker run --rm "${extra[@]}" \
    -v "$FV_ROOT:/src" \
    -v fastvideo-rs-cargo-registry:/usr/local/cargo/registry \
    -v fastvideo-rs-cargo-git:/usr/local/cargo/git \
    -v fastvideo-rs-gpucheck-target:/target \
    -e FV_GIT_SHA="$(fv_build_id)" \
    "$BUILDER_IMAGE" bash -euo pipefail -c "$1"
}
FV_DOCKER_EXTRA=()

# shellcheck disable=SC2016  # expanded inside the container
# Subshell: `exit` must end only the build step, not the chained commands after it.
BUILD_CMD='(cargo build --release -p fastvideo-gpucheck --features cuda 2>&1 | grep -E "^(error|warning: unused)|Compiling fastvideo|Finished" ; exit "${PIPESTATUS[0]}")'

cmd_test() {
  log "unit tests in $BUILDER_IMAGE"
  # shellcheck disable=SC2016
  in_builder 'cargo test -q -p fastvideo-gpucheck -p fastvideo-cudarc --lib --bins 2>&1 | grep -E "test result|FAILED|panicked|^error" ; exit "${PIPESTATUS[0]}"'
}

cmd_nvrtc() {
  log "NVRTC compile gate in $BUILDER_IMAGE (CUDA 13.0 nvrtc + nvcc, no GPU)"
  mkdir -p "$LOCAL_OUT"
  in_builder "$BUILD_CMD && /target/release/fv-gpucheck --out /src/artifacts/gpucheck/local nvrtc"
}

dist_fresh() {
  [[ -x "$DIST/fv-gpucheck" && -f "$DIST/fv-gpucheck.build-id" && "$(cat "$DIST/fv-gpucheck.build-id")" == "$(fv_build_id)" ]]
}

cmd_dist() {
  local id; id="$(fv_build_id)"
  if dist_fresh && [[ "${1:-}" != "--force" ]]; then
    log "dist binary up to date (build $id)"
    return 0
  fi
  log "release build in $BUILDER_IMAGE (build $id)"
  mkdir -p "$DIST"
  in_builder "$BUILD_CMD && install -m 755 /target/release/fv-gpucheck /src/artifacts/gpucheck/dist/fv-gpucheck \
    && echo $id > /src/artifacts/gpucheck/dist/fv-gpucheck.build-id \
    && ldd /src/artifacts/gpucheck/dist/fv-gpucheck | grep -v -E 'linux-vdso|ld-linux' | sed 's/^/  needs /'"
  log "dist: $DIST/fv-gpucheck"
}

hf_snapshot() {
  local dir="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${1//\//--}/snapshots"
  [[ -d "$dir" ]] || return 1
  local snap; snap="$(find "$dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
  [[ -d "$snap/transformer" && -d "$snap/vae" ]] || return 1
  echo "$snap"
}

# CPU-path references: identical binary and build id to what the rented box
# runs, so the GPU stages can compare against them without spending billed
# time on the (slow) host path.
# Dump CPU-path references locally into the same keyed cache `validate.sh run`
# reads (artifacts/gpucheck/refs/<ref key>/<model|parity>/refs), so a rental
# skips those CPU stages.
cmd_refs() {
  local parity=0
  [[ "${1:-}" == "--parity" ]] && parity=1
  cmd_dist
  local key; key="$(fv_ref_key)"
  local scratch="/src/artifacts/gpucheck/refs/$key/.dump"
  FV_DOCKER_EXTRA=(-e "FV_REF_KEY=$key")
  log "model reference (random weights, CPU path) for ref key $key"
  in_builder "/src/artifacts/gpucheck/dist/fv-gpucheck --out /src/artifacts/gpucheck/local --tag ref \
    --mode exact model --device cpu --dump $scratch"
  ref_file model
  if [[ $parity -eq 1 ]]; then
    local snap mem_gb
    snap="$(hf_snapshot "$BASE_REPO")" || die "no local $BASE_REPO snapshot with transformer/ + vae/ in the HF cache"
    mem_gb=$(( $(docker info --format '{{.MemTotal}}') / 1073741824 ))
    (( mem_gb >= 28 )) || die "parity reference needs ~28GB in the Docker VM (have ${mem_gb}GB): raise Docker Desktop → Settings → Resources → Memory, or let the rented box compute it"
    local models_dir="${snap%/snapshots/*}"
    log "parity reference (real 1.3B weights, CPU path; tens of minutes)"
    FV_DOCKER_EXTRA=(-e "FV_REF_KEY=$key" -v "$models_dir:/hf/model:ro")
    in_builder "/src/artifacts/gpucheck/dist/fv-gpucheck --out /src/artifacts/gpucheck/local --tag ref \
      --mode exact parity --weights /hf/model/snapshots/$(basename "$snap") --device cpu --dump $scratch"
    ref_file parity
  fi
  FV_DOCKER_EXTRA=()
  rm -rf "$REFS/$key/.dump"
  log "references for ref key $key in $REFS/$key"
}

# ref_file <model|parity>: move a dump from the scratch dir into the cache layout.
ref_file() {
  local key; key="$(fv_ref_key)"
  local dir="$REFS/$key/$1/refs"
  mkdir -p "$dir"
  mv "$REFS/$key/.dump/$1.safetensors" "$REFS/$key/.dump/$1.json" "$dir/"
}

cmd_image() {
  cmd_dist
  require_docker
  log "building $RUNTIME_IMAGE"
  # Same Dockerfile CI publishes to GHCR; reuse the local dist binary instead of recompiling.
  docker buildx build --platform "$PLATFORM" -f "$DOCKERFILE" --target runtime \
    --build-context binary="$DIST" -t "$RUNTIME_IMAGE" --load "$FV_ROOT" >&2
}

# Vast images: FROM vastai/pytorch (host-cached) + our binary. Reuse dist when present.
cmd_vast_image() {
  local target="${1:-runtime}" tag="$VAST_IMAGE"
  [[ "$target" == "oracle" ]] && tag="$VAST_ORACLE_IMAGE"
  cmd_dist
  require_docker
  log "building $tag (FROM $VAST_PYTORCH_BASE, target=$target)"
  docker buildx build --platform "$PLATFORM" -f "$VAST_DOCKERFILE" --target "$target" \
    --build-arg "VAST_PYTORCH_IMAGE=$VAST_PYTORCH_BASE" \
    --build-arg "BUILD_ID=$(fv_build_id)" \
    --build-context binary="$DIST" \
    -t "$tag" --load "$FV_ROOT" >&2
  log "built $tag"
}

cmd_vast_oracle() {
  cmd_vast_image oracle
}

cmd_gpu() {
  require_docker
  if ! docker info --format '{{json .Runtimes}}' | grep -q nvidia \
     && ! docker info 2>/dev/null | grep -q 'nvidia.com/gpu'; then
    die "no NVIDIA container runtime on this Docker host. GPU stages need Linux + an NVIDIA GPU + nvidia-container-toolkit (Docker Desktop on macOS cannot pass a GPU through). Use: scripts/gpu/validate.sh run <tier>"
  fi
  docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1 || cmd_image
  local out="$FV_ROOT/artifacts/gpucheck/docker-gpu"
  mkdir -p "$out"
  docker run --rm --gpus all --platform "$PLATFORM" \
    -v "$out:/work/out" \
    -v "${HF_HOME:-$HOME/.cache/huggingface}:/hf:ro" \
    -e FV_GIT_SHA="$(fv_build_id)" \
    "$RUNTIME_IMAGE" /opt/fastvideo-rs/target/release/fv-gpucheck --out /work/out "$@"
}

# Tests source this file for helpers.
[[ -n "${FV_SOURCE_ONLY:-}" && "${BASH_SOURCE[0]}" != "$0" ]] && return 0

case "${1:-}" in
  builder) cmd_builder ;;
  test) cmd_test ;;
  nvrtc) cmd_nvrtc ;;
  dist) shift; cmd_dist "$@" ;;
  refs) shift; cmd_refs "$@" ;;
  image) cmd_image ;;
  vast-image) cmd_vast_image runtime ;;
  vast-oracle) cmd_vast_oracle ;;
  gpu) shift; cmd_gpu "$@" ;;
  build-id) fv_build_id ;;
  *) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
