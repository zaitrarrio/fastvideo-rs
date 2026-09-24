#!/usr/bin/env bash
# Build the vendored Rust-to-PTX packages as three Docker layers, then emit
# Tile-IR W4A4 cubins via `cargo oxide build --arch sm_100,sm_120`.
# No NVIDIA GPU required. Docker Desktop on macOS cross-builds linux/amd64.
#
#   fastvideo-oxide-base:cu134     CUDA 13.4 runtime
#   fastvideo-oxide-build:cu134    base + toolchain, compiles the packages
#   fastvideo-oxide-runtime:cu134  base + nightly rustc + llc-21 + artifacts
#
# Artifacts (including cubins, when cargo-oxide succeeds) go to artifacts/oxide/.
# The Tile-IR GEMM stays OFF at runtime until it beats cuBLAS bf16 on the H3
# FFN shape (K=5376, N=14336) and PSNR ≥ 30 dB vs bf16.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
BASE_IMAGE="${OXIDE_BASE_IMAGE:-fastvideo-oxide-base:cu134}"
BUILD_IMAGE="${OXIDE_BUILD_IMAGE:-fastvideo-oxide-build:cu134}"
RUNTIME_IMAGE="${OXIDE_RUNTIME_IMAGE:-fastvideo-oxide-runtime:cu134}"
OXIDE="$ROOT/third_party/cuda-oxide"
CUTILE="$ROOT/third_party/cutile-rs"
KERNELS="$ROOT/crates/fastvideo-oxide-kernels"

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

# cargo-oxide v0.2.1 `--arch` is one SM. Try the multi-arch form the contract
# names, then each Blackwell SM. Cubin emit is best-effort: Mac/host without
# tileiras still leaves the rustc backend artifacts in place.
emit_cubins() {
  echo "oxide: cargo oxide build --arch sm_100,sm_120"
  docker run --rm --platform "$PLATFORM" \
    -e CUDA_OXIDE_BACKEND=/opt/oxide/lib/librustc_codegen_cuda.so \
    -e CUDA_OXIDE_LLC=/usr/bin/llc-21 \
    -v "$KERNELS:/src/crates/fastvideo-oxide-kernels" \
    -v "$OXIDE:/src/third_party/cuda-oxide" \
    -v "$CUTILE:/src/third_party/cutile-rs" \
    -v "$ROOT/artifacts/oxide:/out" \
    "$BUILD_IMAGE" \
    bash -euo pipefail -c '
      export PATH=/usr/local/cargo/bin:/usr/lib/llvm-21/bin:$PATH
      cd /src/crates/fastvideo-oxide-kernels
      if cargo oxide build --arch sm_100,sm_120; then
        :
      else
        echo "oxide: multi-arch flag rejected; building sm_100 and sm_120"
        cargo oxide build --arch sm_100
        cargo oxide build --arch sm_120
      fi
      mkdir -p /out/cubins
      python3 - <<'"'"'PY'"'"'
import pathlib, shutil, re
roots = [pathlib.Path("/src/crates/fastvideo-oxide-kernels"), pathlib.Path("/oxide-target"), pathlib.Path("/opt/oxide")]
copied = 0
for root in roots:
    if not root.exists():
        continue
    for p in root.rglob("*.cubin"):
        m = re.search(r"sm[_]?(\d+)", p.name + str(p.parent))
        sm = m.group(1) if m else "unknown"
        dest = pathlib.Path(f"/out/nvfp4_w4a4_sm{sm}.cubin")
        shutil.copy2(p, dest)
        copied += 1
        print(f"oxide: cubin {p} -> {dest}")
print(f"oxide: copied {copied} cubin(s)")
PY
    '
}

if emit_cubins; then
  echo "oxide: cubin emit finished"
else
  echo "oxide: cubin emit skipped (cargo oxide build failed; no Blackwell cubins on this host)"
fi

echo "oxide: base    $BASE_IMAGE"
echo "oxide: build   $BUILD_IMAGE"
echo "oxide: runtime $RUNTIME_IMAGE"
echo "oxide: artifacts $ROOT/artifacts/oxide"
ls -la "$ROOT/artifacts/oxide" || true
