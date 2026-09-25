#!/usr/bin/env bash
# Compile the Tile-IR NVFP4 W4A4 GEMM (crates/fastvideo-oxide-kernels) to
# cubins for sm_100 and sm_120 into artifacts/oxide/, where
# crates/fastvideo-cudarc/build.rs embeds them (manifest.tsv + *.cubin).
# No NVIDIA GPU required.
#
# The kernel is cutile-rs (vendored at third_party/cutile-rs). cutile normally
# JIT-compiles on first launch through `tileiras`; fv-oxide-aot runs cutile's
# compile-only API and tileiras ahead of time instead, so the runtime needs
# neither. CI does the same in docker/gpucheck.Dockerfile (stage `oxide`).
#
#   scripts/oxide.sh            docker buildx, stage oxide-out (any host)
#   scripts/oxide.sh --local    cargo on this host; needs CUDA >= 13.2 headers
#                               (cuda.h, curand.h) and tileiras under
#                               CUDA_TOOLKIT_PATH (or CUDA_HOME)
#
# docker/oxide.Dockerfile is separate: the cuda-oxide rustc backend (SIMT
# kernels in Rust, nightly-2026-04-03). The NVFP4 GEMM does not need it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/artifacts/oxide"
KERNELS="$ROOT/crates/fastvideo-oxide-kernels"

if [[ ! -f "$ROOT/third_party/cutile-rs/Cargo.toml" ]]; then
  echo "oxide: third_party/cutile-rs is not checked out." >&2
  echo "  git submodule update --init third_party/cutile-rs" >&2
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"
if [[ "${1:-}" == "--local" ]]; then
  target="${CARGO_TARGET_DIR:-$ROOT/target/oxide}"
  (cd "$KERNELS" && CARGO_TARGET_DIR="$target" cargo build --release --locked)
  "$target/release/fv-oxide-aot" "$OUT" --sm 100,120
else
  docker buildx build \
    --platform "${DOCKER_PLATFORM:-linux/amd64}" \
    -f "$ROOT/docker/gpucheck.Dockerfile" \
    --target oxide-out \
    --output "type=local,dest=$OUT" \
    "$ROOT"
fi
echo "oxide: cubins in $OUT"
cat "$OUT/manifest.tsv"
