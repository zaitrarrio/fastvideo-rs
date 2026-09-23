#!/usr/bin/env bash
# Build the vendored Rust-to-PTX packages as three Docker layers.
# No NVIDIA GPU required. Docker Desktop on macOS cross-builds linux/amd64.
#
#   fastvideo-oxide-base:cu134     CUDA 13.4 runtime
#   fastvideo-oxide-build:cu134    base + toolchain, compiles the packages
#   fastvideo-oxide-runtime:cu134  base + nightly rustc + llc-21 + artifacts
#
# Artifacts are also copied to artifacts/oxide/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
BASE_IMAGE="${OXIDE_BASE_IMAGE:-fastvideo-oxide-base:cu134}"
BUILD_IMAGE="${OXIDE_BUILD_IMAGE:-fastvideo-oxide-build:cu134}"
RUNTIME_IMAGE="${OXIDE_RUNTIME_IMAGE:-fastvideo-oxide-runtime:cu134}"
OXIDE="$ROOT/third_party/cuda-oxide"
CUTILE="$ROOT/third_party/cutile-rs"

if [[ ! -f "$OXIDE/Cargo.toml" || ! -f "$CUTILE/Cargo.toml" ]]; then
  echo "oxide: vendored toolchain is not checked out." >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi
command -v docker >/dev/null 2>&1 || { echo "oxide: docker is required" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "oxide: Docker daemon is not running" >&2; exit 1; }

build_target() {
  local target="$1" tag="$2"
  docker build \
    --platform "$PLATFORM" \
    -f "$ROOT/docker/oxide.Dockerfile" \
    --target "$target" \
    -t "$tag" \
    "$ROOT"
}

# Runtime depends on build, which depends on base. One pass fills the cache;
# the two retags then only name the earlier stages.
build_target runtime "$RUNTIME_IMAGE"
build_target base "$BASE_IMAGE"
build_target build "$BUILD_IMAGE"

rm -rf "$ROOT/artifacts/oxide"
mkdir -p "$ROOT/artifacts/oxide"
cid="$(docker create --platform "$PLATFORM" "$RUNTIME_IMAGE")"
docker cp "$cid":/opt/oxide/. "$ROOT/artifacts/oxide/"
docker rm "$cid" >/dev/null

echo "oxide: base    $BASE_IMAGE"
echo "oxide: build   $BUILD_IMAGE"
echo "oxide: runtime $RUNTIME_IMAGE"
echo "oxide: artifacts $ROOT/artifacts/oxide"
