#!/usr/bin/env bash
# Pod half of `build-pod.sh release-artifacts <sha>` (docs/dev/build-pod.md,
# "Release artifacts"). Runs ON THE BUILD POD as an allowlisted job, in the
# agent snapshot of one commit (cwd), with the job env of build-pod-server.py
# (CARGO_TARGET_DIR, CUDA_HOME on the volume, sccache). Builds, in release
# mode, exactly what the Dockerfiles build today, and packs one tarball per
# artifact set plus manifest.json into $CARGO_TARGET_DIR/release-artifacts/<sha>/:
#
#   set            what (Dockerfile stage it replaces)                    tarball root
#   oxide          Tile-IR NVFP4 cubins, fv-oxide-aot --sm 100,120 (oxide)  out/oxide/
#   gpucheck       fv-gpucheck --features cuda + oxide embedded (binary)   fv-gpucheck, .build-id
#   gpucheck-vast  vast-pytorch.Dockerfile's fv-gpucheck: no nvcc on the
#                  builder there, so no AOT/oxide cubins (binary)          fv-gpucheck, .build-id
#   hf-fm          cargo install hf-fetch-model --features cli (hf-fm)     out/hf-fm, out/hf-fetch-model
#   serve-cuda     fv-serve --features $FV_SERVE_FEATURES (serve-build)    out/fv-serve, out/fv-serve.features
#   serve-gateway  fv-serve --features $FV_GATEWAY_FEATURES (gateway-build) out/fv-serve, out/fv-serve.features
#   serve-fake     serve-compat.yml's debug fv-serve --features fake,full  fv-serve
#   gpucheck-tests gpucheck-t0.yml's `cargo test -p fastvideo-gpucheck
#                  -p fastvideo-cudarc --lib --bins`, --no-run            tests.tsv, bin/*
#
# Each tarball root is laid out like the stage it replaces, so GitHub passes
# the extracted directory as a named build context (`--build-context
# serve-build=…`) and the Dockerfile's COPY --from lines take it unchanged.
#
# Settings match docker/gpucheck.Dockerfile's builder: CARGO_PROFILE_RELEASE_
# LTO=off, CODEGEN_UNITS=16, PANIC=unwind (the job env), CUDARC_CUDA_VERSION=
# 13000, the CUDA 13.4.92 toolkit, toolchain `stable` per rust-toolchain.toml,
# and no RUSTFLAGS: the pod's mold link-arg is dropped so the system linker
# links, as in the image. The pod is Debian bookworm (glibc 2.36) while the
# images are Ubuntu 22.04 (glibc 2.35): every ELF is checked for GLIBC_*
# symbol versions above FV_REL_MAX_GLIBC (2.35) and the build fails if one
# appears.
#
# Env (set by build-pod.sh): FV_REL_SHA, FV_GIT_SHA, FV_BUILD_TIME, FV_BUILD_ID
# (scripts/gpu/docker.sh build-id at that commit), FV_REL_RUN_ID; optional
# FV_REL_SETS (space list, default all), FV_SERVE_FEATURES (cuda,http-client),
# FV_GATEWAY_FEATURES (http-client), FV_REL_MAX_GLIBC (2.35).
set -euo pipefail

log() { printf '[rel %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 2; }
: "${FV_REL_SHA:?FV_REL_SHA}" "${FV_BUILD_ID:?FV_BUILD_ID}" "${FV_GIT_SHA:?FV_GIT_SHA}" "${CARGO_TARGET_DIR:?CARGO_TARGET_DIR}" "${CUDA_HOME:?CUDA_HOME}"
[[ "$FV_REL_SHA" =~ ^[0-9a-f]{40}$ ]] || die "FV_REL_SHA must be a full sha"
ALL_SETS="oxide gpucheck gpucheck-vast hf-fm serve-cuda serve-gateway serve-fake gpucheck-tests"
SETS=" ${FV_REL_SETS:-$ALL_SETS} "
SERVE_FEATURES="${FV_SERVE_FEATURES:-cuda,http-client}"
GATEWAY_FEATURES="${FV_GATEWAY_FEATURES:-http-client}"
MAX_GLIBC="${FV_REL_MAX_GLIBC:-2.35}"
SRC="$PWD"
T="$CARGO_TARGET_DIR"
VOL="$(dirname "$CUDA_HOME")"          # the fv-build volume root
CACHE="$VOL/release-cache"
OUT="$T/release-artifacts/$FV_REL_SHA"
STAGE="$OUT/stage"
want() { [[ "$SETS" == *" $1 "* ]]; }
t_start=$(date +%s)

# Same flags as the image builder: no RUSTFLAGS (the pod adds mold).
unset CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS RUSTFLAGS CARGO_BUILD_RUSTFLAGS
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_PROFILE_RELEASE_PANIC=unwind
export CUDARC_CUDA_VERSION=13000 NVCC="$CUDA_HOME/bin/nvcc"
# Build identity, as the image build args set it (crates/fastvideo-serve/build.rs).
export BUILD_ID="$FV_BUILD_ID" FV_BUILD_TIME="${FV_BUILD_TIME:-}"
unset GITHUB_SHA

# Older snapshots of this commit's outputs only (the target dir is reused).
rm -rf "$T/release-artifacts"
mkdir -p "$STAGE" "$CACHE"

# jq (manifest) and binutils (glibc check) are not in rust:1-bookworm by default.
need_apt=()
command -v jq >/dev/null || need_apt+=(jq)
command -v objdump >/dev/null || need_apt+=(binutils)
if (( ${#need_apt[@]} )); then
  log "apt: ${need_apt[*]}"
  DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 update -qq >/dev/null
  DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y -qq --no-install-recommends "${need_apt[@]}" >/dev/null
fi

# ---- prerequisites the image's oxide stage installs with apt ---------------
# libclang (cutile's bindgen) and the cuda.h / curand.h headers. The volume's
# toolkit (build-pod-server.py) has cudart but no cuRAND: add the libcurand
# redist headers into a private overlay of it on the container disk.
oxide_prereqs() {
  if ! ldconfig -p | grep -q 'libclang[-.0-9]*\.so'; then
    log "apt: libclang-dev"
    DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 update -qq >/dev/null
    DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y -qq --no-install-recommends libclang-dev >/dev/null
  fi
  local redist ver="${FV_BUILD_CUDA_REDIST:-13.4.2}" base="https://developer.download.nvidia.com/compute/cuda/redist/"
  redist="$CACHE/cuda-$ver-libcurand"
  if [[ ! -f "$redist/.ok" ]]; then
    log "cuda redist $ver: libcurand headers"
    local idx rel sum tmp
    tmp="$(mktemp -d)"
    idx="$(curl -fsSL "${base}redistrib_$ver.json")"
    rel="$(jq -r '.libcurand["linux-x86_64"].relative_path' <<<"$idx")"
    sum="$(jq -r '.libcurand["linux-x86_64"].sha256' <<<"$idx")"
    curl -fsSL -o "$tmp/c.tar.xz" "$base$rel"
    echo "$sum  $tmp/c.tar.xz" | sha256sum -c - >/dev/null || die "libcurand redist sha256 mismatch"
    rm -rf "$redist.part" && mkdir -p "$redist.part"
    tar -xJf "$tmp/c.tar.xz" -C "$redist.part" --strip-components=1 --wildcards '*/include/*' '*/LICENSE'
    rm -rf "$tmp"
    touch "$redist.part/.ok"
    rm -rf "$redist" && mv "$redist.part" "$redist"
  fi
  CUDA_OVERLAY="$T/cuda-overlay"
  rm -rf "$CUDA_OVERLAY"
  mkdir -p "$CUDA_OVERLAY"
  cp -as "$CUDA_HOME/." "$CUDA_OVERLAY/"
  cp -as "$redist/include/." "$CUDA_OVERLAY/include/"
  [[ -f "$CUDA_OVERLAY/include/cuda.h" && -f "$CUDA_OVERLAY/include/curand.h" ]] \
    || die "cuda.h / curand.h missing from the toolkit overlay"
  [[ -x "$CUDA_HOME/bin/tileiras" ]] || die "no tileiras in $CUDA_HOME"
}

# ---- oxide: cached on the volume by its inputs (like the image's layer cache)
build_oxide() {
  local key dir
  key="$( { find crates/fastvideo-oxide-kernels third_party/cutile-rs -type f ! -path '*/target/*' -print0 \
            | LC_ALL=C sort -z | xargs -0 sha256sum; "$CUDA_HOME/bin/tileiras" --version 2>&1 || true; } \
          | sha256sum | cut -c1-16)"
  dir="$CACHE/oxide/$key"
  if [[ -s "$dir/out/oxide/manifest.tsv" ]]; then
    log "oxide: cached ($key)"
  else
    [[ -f third_party/cutile-rs/Cargo.toml ]] || die "third_party/cutile-rs is not in the snapshot"
    oxide_prereqs
    log "oxide: cargo build fv-oxide-aot, then --sm 100,120"
    rm -rf "$dir.part" && mkdir -p "$dir.part/out"
    (cd crates/fastvideo-oxide-kernels \
      && CUDA_TOOLKIT_PATH="$CUDA_OVERLAY" CUDA_HOME="$CUDA_OVERLAY" CARGO_TARGET_DIR="$T/oxide" \
         cargo build --release --locked)
    CUTILE_TILEIRAS_PATH="$CUDA_HOME/bin/tileiras" CUDA_TOOLKIT_PATH="$CUDA_OVERLAY" \
      "$T/oxide/release/fv-oxide-aot" "$dir.part/out/oxide" --sm 100,120
    test -s "$dir.part/out/oxide/manifest.tsv"
    rm -rf "$dir" && mkdir -p "$(dirname "$dir")" && mv "$dir.part" "$dir"
  fi
  OXIDE_DIR="$dir/out/oxide"
  OXIDE_KEY="$key"
  mkdir -p "$STAGE/oxide/out"
  cp -a "$OXIDE_DIR" "$STAGE/oxide/out/oxide"
  cat "$STAGE/oxide/out/oxide/manifest.tsv" >&2
}

# The `build` stage's ENV, inherited by serve-build and gateway-build.
with_oxide() { FV_OXIDE_CUBIN_DIR="$OXIDE_DIR" FV_REQUIRE_OXIDE=100,120 "$@"; }

build_gpucheck() {
  log "gpucheck: cargo build --release -p fastvideo-gpucheck --features cuda"
  with_oxide cargo build --release -p fastvideo-gpucheck --features cuda >&2
  # The image smoke check, minus the GPU: NVRTC compile gate, AOT cubins and
  # the oxide cubins for both SMs embedded (the job env has libnvrtc).
  "$T/release/fv-gpucheck" --out "$OUT/nvrtc" nvrtc >&2
  grep -q aot_sms "$OUT/nvrtc/nvrtc.json" || die "gpucheck: no embedded AOT cubins"
  for sm in 100 120; do
    grep -q "sm$sm bf16 128x128x128" "$OUT/nvrtc/nvrtc.json" || die "gpucheck: no oxide cubin for sm_$sm"
  done
  mkdir -p "$STAGE/gpucheck"
  cp "$T/release/fv-gpucheck" "$STAGE/gpucheck/fv-gpucheck"
  echo "$FV_BUILD_ID" >"$STAGE/gpucheck/fv-gpucheck.build-id"
}

# vast-pytorch.Dockerfile's builder points NVCC at /usr/local/cuda-13.0/bin/nvcc,
# which its CUDA 13.4 packages do not provide, and sets no oxide dir: its
# fv-gpucheck NVRTC-compiles every kernel at run time. Reproduced as is
# (own target dir, so it does not flip the main build's cudarc build script).
build_gpucheck_vast() {
  log "gpucheck-vast: same build without nvcc and oxide (as vast-pytorch.Dockerfile)"
  env -u FV_OXIDE_CUBIN_DIR -u FV_REQUIRE_OXIDE NVCC=/usr/local/cuda-13.0/bin/nvcc \
    PATH="$(tr ':' '\n' <<<"$PATH" | grep -v "^$CUDA_HOME/bin$" | paste -sd:)" CARGO_TARGET_DIR="$T/vast" \
    cargo build --release -p fastvideo-gpucheck --features cuda >&2
  grep -qs "no nvcc found" "$T"/vast/release/build/fastvideo-cudarc-*/output || die "gpucheck-vast: expected the no-nvcc build"
  mkdir -p "$STAGE/gpucheck-vast"
  cp "$T/vast/release/fv-gpucheck" "$STAGE/gpucheck-vast/fv-gpucheck"
  echo "$FV_BUILD_ID" >"$STAGE/gpucheck-vast/fv-gpucheck.build-id"
}

# hf-fm stage: `cargo install hf-fetch-model --features cli` (unpinned, as in
# the image; the version lands in the manifest). The install root lives on the
# volume, so an unchanged release is not rebuilt.
build_hf_fm() {
  local root="$CACHE/hf-fm-root"
  log "hf-fm: cargo install hf-fetch-model --features cli"
  env -u CARGO_TARGET_DIR cargo install hf-fetch-model --features cli --root "$root" >&2
  mkdir -p "$STAGE/hf-fm/out"
  cp "$root/bin/hf-fm" "$root/bin/hf-fetch-model" "$STAGE/hf-fm/out/"
  HF_FM_VERSION="$(cargo install --list --root "$root" | awk '/^hf-fetch-model /{print $2}' | tr -d 'v:')"
}

build_serve() { # <set> <features>
  local set="$1" feats="$2"
  log "$set: cargo build --release -p fastvideo-serve --features $feats"
  with_oxide cargo build --release -p fastvideo-serve --features "$feats" >&2
  mkdir -p "$STAGE/$set/out"
  cp "$T/release/fv-serve" "$STAGE/$set/out/fv-serve"
  echo "$feats" >"$STAGE/$set/out/fv-serve.features"
}

# serve-compat.yml's build job (debug profile, its env and --config flags).
build_serve_fake() {
  log "serve-fake: cargo build -p fastvideo-serve --features fake,full --bin fv-serve (debug)"
  CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 \
  cargo build -p fastvideo-serve --features fake,full --bin fv-serve \
    --config 'profile.dev.package.openh264-sys2.opt-level=3' \
    --config 'profile.dev.package.openh264.opt-level=3' \
    --config 'profile.dev.package.audiopus_sys.opt-level=3' \
    --config 'profile.dev.package.libwebp-sys.opt-level=3' >&2
  mkdir -p "$STAGE/serve-fake"
  cp "$T/debug/fv-serve" "$STAGE/serve-fake/fv-serve"
}

# gpucheck-t0.yml's unit-tests job, compiled only. tests.tsv: one line per
# test binary: file, package, crate-relative manifest dir, target kind/name.
build_gpucheck_tests() {
  log "gpucheck-tests: cargo test -p fastvideo-gpucheck -p fastvideo-cudarc --lib --bins --no-run"
  mkdir -p "$STAGE/gpucheck-tests/bin"
  env -u FV_OXIDE_CUBIN_DIR -u FV_REQUIRE_OXIDE \
    cargo test -p fastvideo-gpucheck -p fastvideo-cudarc --lib --bins --no-run --message-format=json-render-diagnostics \
    >"$OUT/tests.jsonl"
  jq -r --arg src "$SRC/" 'select(.reason == "compiler-artifact" and .profile.test == true and .executable != null)
      | [.executable, (.package_id), (.manifest_path | sub("/Cargo.toml$"; "") | ltrimstr($src)), (.target.kind[0]), .target.name] | @tsv' \
    "$OUT/tests.jsonl" >"$OUT/tests.raw"
  [[ -s "$OUT/tests.raw" ]] || die "no test binaries in cargo's output"
  : >"$STAGE/gpucheck-tests/tests.tsv"
  while IFS=$'\t' read -r exe pkg dir kind name; do
    cp "$exe" "$STAGE/gpucheck-tests/bin/"
    pkg="$(basename "$dir")"   # the package ids (path+file:///…#0.1.0) omit the name
    printf '%s\t%s\t%s\t%s\t%s\n' "bin/$(basename "$exe")" "$pkg" "$dir" "$kind" "$name" >>"$STAGE/gpucheck-tests/tests.tsv"
  done <"$OUT/tests.raw"
  printf '%s\n' "$SRC" >"$STAGE/gpucheck-tests/src-root"
  cat "$STAGE/gpucheck-tests/tests.tsv" >&2
}

# x86-64 host executables / libraries (the CUDA cubins are ELF too: skipped).
is_host_elf() {
  [[ "$(head -c4 "$1" | od -An -c | tr -d ' ')" == 177ELF \
     && "$(od -An -tx1 -j18 -N2 "$1" | tr -d ' ')" == 3e00 ]]
}

# GLIBC_x.y symbol versions above the images' glibc fail the build.
check_glibc() {
  local f bad=0 v
  while IFS= read -r -d '' f; do
    is_host_elf "$f" || continue
    v="$( { objdump -T "$f" 2>/dev/null || true; } | { grep -o 'GLIBC_[0-9.]*' || true; } \
          | sed 's/GLIBC_//' | sort -uV | tail -1)"
    [[ -z "$v" ]] && continue
    if [[ "$(printf '%s\n%s\n' "$v" "$MAX_GLIBC" | sort -V | tail -1)" != "$MAX_GLIBC" ]]; then
      log "FAIL glibc: ${f#"$STAGE"/} needs GLIBC_$v (> $MAX_GLIBC)"; bad=1
    fi
  done < <(find "$STAGE" -type f -print0)
  (( bad == 0 )) || die "binaries need a newer glibc than the Ubuntu 22.04 images have"
  log "glibc: every ELF needs <= GLIBC_$MAX_GLIBC"
}

# ---- build ------------------------------------------------------------------
OXIDE_DIR=""; OXIDE_KEY=""; HF_FM_VERSION=""
if want oxide || want gpucheck || want serve-cuda || want serve-gateway; then build_oxide; fi
want gpucheck && build_gpucheck
want serve-cuda && build_serve serve-cuda "$SERVE_FEATURES"
want serve-gateway && build_serve serve-gateway "$GATEWAY_FEATURES"
want serve-fake && build_serve_fake
want gpucheck-tests && build_gpucheck_tests
want gpucheck-vast && build_gpucheck_vast
want hf-fm && build_hf_fm
want oxide || rm -rf "$STAGE/oxide"
check_glibc

# ---- pack + manifest --------------------------------------------------------
mtime="@$(date -d "${FV_BUILD_TIME:-1970-01-01T00:00:00Z}" +%s 2>/dev/null || echo 0)"
sets_json='{}'
for d in "$STAGE"/*/; do
  set="$(basename "$d")"
  tar -C "$d" --sort=name --owner=0 --group=0 --numeric-owner --mtime="$mtime" -cf - . | gzip -n -6 >"$OUT/$set.tar.gz"
  files="$(cd "$d" && find . -type f -printf '%P\n' | LC_ALL=C sort | while IFS= read -r p; do
      printf '%s\t%s\t%s\t%s\n' "$p" "$(sha256sum "$p" | cut -d' ' -f1)" "$(stat -c %s "$p")" "$(stat -c %a "$p")"; done \
    | jq -R -s 'split("\n") | map(select(length > 0) | split("\t") | {key: .[0], value: {sha256: .[1], size: (.[2]|tonumber), mode: .[3]}}) | from_entries')"
  feats=""
  case "$set" in
    serve-cuda) feats="$SERVE_FEATURES" ;; serve-gateway) feats="$GATEWAY_FEATURES" ;;
    serve-fake) feats="fake,full" ;; gpucheck|gpucheck-vast) feats="cuda" ;; hf-fm) feats="cli" ;;
  esac
  needed="$(while IFS= read -r -d '' f; do
        if is_host_elf "$f"; then readelf -d "$f" 2>/dev/null || true; fi
      done < <(find "$d" -type f -print0) \
    | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' | sort -u | jq -R -s 'split("\n") | map(select(length > 0))')"
  sets_json="$(jq -c --arg s "$set" --arg tb "$set.tar.gz" --arg sha "$(sha256sum "$OUT/$set.tar.gz" | cut -d' ' -f1)" \
    --argjson size "$(stat -c %s "$OUT/$set.tar.gz")" --arg f "$feats" --argjson files "$files" --argjson needed "$needed" \
    '. + {($s): {tarball: $tb, sha256: $sha, size: $size, features: $f, needed: $needed, files: $files}}' <<<"$sets_json")"
done

jq -n --arg sha "$FV_REL_SHA" --arg git "$FV_GIT_SHA" --arg bid "$FV_BUILD_ID" --arg btime "${FV_BUILD_TIME:-}" \
  --arg run "${FV_REL_RUN_ID:-}" --arg created "$(date -u +%FT%TZ)" --argjson secs "$(( $(date +%s) - t_start ))" \
  --arg rustc "$(rustc -vV | tr '\n' ';')" --arg cargo "$(cargo -V)" \
  --arg toolchain "$(rustup show active-toolchain 2>/dev/null | head -1)" \
  --arg nvcc "$("$CUDA_HOME/bin/nvcc" --version | tail -2 | tr '\n' ' ')" \
  --arg tileiras "$("$CUDA_HOME/bin/tileiras" --version 2>&1 | tail -1)" \
  --arg glibc "$(ldd --version | head -1)" --arg maxg "$MAX_GLIBC" --arg oxk "$OXIDE_KEY" --arg hffm "$HF_FM_VERSION" \
  --arg script "$(sha256sum "$0" | cut -c1-16)" --argjson sets "$sets_json" '{
    schema: 1, sha: $sha, git_sha: $git, build_id: $bid, build_time: $btime, run_id: $run,
    created: $created, build_seconds: $secs,
    builder: {host: "fv-build pod (Runpod CPU, rust:1-bookworm)", rustc: $rustc, cargo: $cargo,
              toolchain: $toolchain, nvcc: $nvcc, tileiras: $tileiras, glibc: $glibc,
              max_glibc_symbol: $maxg, recipe_sha256: $script},
    settings: {CARGO_PROFILE_RELEASE_LTO: "off", CARGO_PROFILE_RELEASE_CODEGEN_UNITS: "16",
               CARGO_PROFILE_RELEASE_PANIC: "unwind", RUSTFLAGS: "", linker: "cc (system default, no mold)",
               CUDARC_CUDA_VERSION: "13000", FV_REQUIRE_OXIDE: "100,120"},
    oxide_key: $oxk, hf_fetch_model_version: $hffm, sets: $sets}' >"$OUT/manifest.json"
rm -rf "$STAGE"
ls -la "$OUT" >&2
log "done in $(( $(date +%s) - t_start ))s: $OUT"
