#!/usr/bin/env bash
# Runs ON the rented GPU box (invoked over ssh by scripts/gpu/validate.sh).
# Every subcommand is idempotent and fails loudly; the local orchestrator
# owns timeouts, ordering, artifact pulls and instance teardown.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${FV_WORK:-/workspace}"
OUT="$WORK/gpucheck-out"
LOGS="$OUT/logs"
export PATH="/usr/local/cuda/bin:$PATH"
mkdir -p "$LOGS"

log() { printf '[remote %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 2; }

# cudarc dlopens CUDA libraries by fixed names: libX.so, libX.so.<cuda major>
# (12), .11, .10, .1. Images put them in different places, and cuDNN 9 ships
# only libcudnn.so.9 (often inside pip site-packages), which cudarc never tries.
# Link every library under its unversioned name in one directory and put that
# plus each real library's directory (for dependent sub-libraries) on the path.
FV_LIBDIR="$WORK/fv-libs"
# cudarc 0.17's cuDNN bindings require symbols newer than the cuDNN 9.1 that
# PyTorch images ship (e.g. cudnnBackendPopulateCudaGraph); bootstrap installs
# this pinned cuDNN into its own directory without touching the image's Python env.
# Apt pins live in cuda-13.pins (nvrtc 13.0.88 / cublas 13.1.1.3 / cudnn 9.26.0.51).
FV_CUDNN_VERSION="${FV_CUDNN_VERSION:-9.26.0.51}"
FV_CUDNN_DIR="$WORK/fv-cudnn"
FV_CUDNN_REQUIRED_SYMBOL="cudnnBackendPopulateCudaGraph"
# `remote.sh cublas <version>` installs a different cuBLAS here; once present
# it replaces the image's for every later stage on this box.
FV_CUBLAS_DIR="$WORK/fv-cublas"
# Minimum cuBLAS: older builds lack Blackwell kernels (generic FP32 only).
# cudarc 0.17 also dlsyms CUDA-13 entry points (`cublasGetEmulationSpecialValuesSupport`);
# 12.9.1 loads and then aborts. The CI image ships libcublas 13.1.1 — do not
# "upgrade" a 12.9 image with nvidia-cublas-cu12; that wheel still lacks the symbol.
FV_CUBLAS_MIN="${FV_CUBLAS_MIN:-13.0.0}"
FV_CUBLAS_VERSION="${FV_CUBLAS_VERSION:-12.9.1.4}"
FV_CUBLAS_REQUIRED_SYMBOL="${FV_CUBLAS_REQUIRED_SYMBOL:-cublasGetEmulationSpecialValuesSupport}"
fv_find_lib() {
  local name="$1" hit=""
  if [[ "$name" == cudnn && -e "$FV_CUDNN_DIR/nvidia/cudnn/lib/libcudnn.so.9" ]]; then
    readlink -f "$FV_CUDNN_DIR/nvidia/cudnn/lib/libcudnn.so.9"
    return 0
  fi
  if [[ "$name" == cublas* ]]; then
    local pip
    for pip in "$FV_CUBLAS_DIR/nvidia/cublas/lib/lib$name.so.13" "$FV_CUBLAS_DIR/nvidia/cublas/lib/lib$name.so.12"; do
      if [[ -e "$pip" ]]; then
        readlink -f "$pip"
        return 0
      fi
    done
  fi
  # Prefer CUDA 13 toolkit libs first. cudarc 0.17 dlsyms
  # cublasGetEmulationSpecialValuesSupport; the pip nvidia-cublas-cu12
  # wheel (ldconfig's usual hit under /opt/nvidia-libs) is 12.x and aborts.
  local cand
  for cand in /usr/local/cuda-13*/lib64/lib${name}.so.13 \
              /usr/local/cuda-13*/lib64/lib${name}.so.13.* \
              /usr/local/cuda-13*/lib64/lib${name}.so \
              /usr/lib/x86_64-linux-gnu/lib${name}.so.9 \
              /usr/lib/x86_64-linux-gnu/lib${name}.so.13; do
    if [[ -e "$cand" ]]; then
      readlink -f "$cand"
      return 0
    fi
  done
  hit="$(ldconfig -p 2>/dev/null | awk -v n="lib$name.so" '$1 == n || index($1, n".") == 1 {print $NF}' | head -1)"
  if [[ -z "$hit" ]]; then
    hit="$(find /usr/local/cuda*/targets/*/lib /usr/local/cuda*/lib64 /usr/lib/x86_64-linux-gnu \
             /opt/conda/lib/python3*/site-packages/nvidia/*/lib /usr/local/lib/python3*/dist-packages/nvidia/*/lib \
             -maxdepth 1 \( -name "lib$name.so" -o -name "lib$name.so.[0-9]*" \) 2>/dev/null \
           | grep -v '/stubs/' | sort | head -1)"
  fi
  [[ -n "$hit" ]] && readlink -f "$hit"
}
fv_setup_libs() {
  mkdir -p "$FV_LIBDIR"
  local dirs="" lib real
  for lib in nvrtc cublas cublasLt cudnn; do
    real="$(fv_find_lib "$lib")" || true
    [[ -n "$real" ]] || continue
    ln -sf "$real" "$FV_LIBDIR/lib$lib.so"
    case ":$dirs:" in *":$(dirname "$real"):"*) ;; *) dirs="$dirs:$(dirname "$real")" ;; esac
  done
  export LD_LIBRARY_PATH="$FV_LIBDIR$dirs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
}
fv_setup_libs

cmd_env() {
  # Cheapest possible gate: is this box usable at all?
  command -v nvidia-smi >/dev/null || die "nvidia-smi missing (not a GPU container?)"
  local q
  q="$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits | head -1)" \
    || die "nvidia-smi failed — driver not usable"
  IFS=',' read -r name mem driver cap <<<"$q"
  local cuda_drv
  # Header format varies by driver ("CUDA Version: 12.4" / "CUDA Version : 13.0").
  cuda_drv="$( { nvidia-smi; nvidia-smi -q; } 2>/dev/null | sed -n 's/.*CUDA Version *: *\([0-9][0-9.]*\).*/\1/p' | head -1)"
  [[ -n "$cuda_drv" ]] || die "could not read the driver's CUDA version from nvidia-smi"
  local disk_gb ram_gb cores
  disk_gb="$(df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9)"
  ram_gb="$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)"
  cores="$(nproc)"
  local libs=()
  for lib in libnvrtc.so libcublas.so libcublasLt.so libcudnn.so; do
    if [[ -e "$FV_LIBDIR/$lib" ]]; then
      libs+=("\"$lib\":\"$(readlink "$FV_LIBDIR/$lib")\"")
    else
      libs+=("\"$lib\":false")
    fi
  done
  local json
  json="$(printf '{"gpu":"%s","vram_mib":%s,"driver":"%s","compute_cap":"%s","driver_cuda":"%s","disk_free_gb":%s,"ram_gb":%s,"cores":%s,"libs":{%s}}' \
    "$(xargs <<<"$name")" "$(xargs <<<"$mem")" "$(xargs <<<"$driver")" "$(xargs <<<"$cap")" "$cuda_drv" \
    "$disk_gb" "$ram_gb" "$cores" "$(IFS=,; echo "${libs[*]}")")"
  echo "$json" | tee "$OUT/env.json"
  [[ "$json" != *':false'* ]] || die "missing CUDA runtime libraries: $json"
  # A rented GPU must be empty. Machine 111175 (2026-09-19) handed out RTX PRO
  # 6000s with ~60 GB already held by another process: LTX-2 ran out of memory
  # loading a 38 GB DiT and the H3 reference could not allocate 62 GB of 95.
  # `env` failing marks the machine bad, so the harness moves to another offer.
  local used; used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc 0-9)"
  if [[ "${used:-0}" -gt "${FV_MAX_GPU_USED_MIB:-2048}" ]]; then
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true
    die "GPU is not empty: ${used} MiB in use before anything of ours ran"
  fi
  # Every library must actually resolve (catches missing deps), not just exist.
  # Prefer ldd over Python ctypes: the runtime image has no Python.
  for lib in libnvrtc.so libcublasLt.so libcublas.so libcudnn.so; do
    path="$FV_LIBDIR/$lib"
    [[ -e "$path" ]] || die "missing $path"
    if command -v ldd >/dev/null; then
      ldd "$path" 2>/dev/null | grep -q 'not found' && die "ldd: $path has unresolved deps"
    fi
  done
  log "cuda libs resolve ok"
  local need_disk="${1:-30}"
  # Weights already on disk are what the headroom was for, so a reused instance
  # (--instance) counts them: the budget is free space PLUS what is downloaded.
  local have_gb=0 d sub
  for sub in weights hf upstream-venv; do
    d="$WORK/$sub"
    [[ -d "$d" ]] || continue
    local gb; gb="$(du -BG -s "$d" 2>/dev/null | tr -dc 0-9)"
    have_gb=$(( have_gb + ${gb:-0} ))
  done
  (( disk_gb + have_gb >= need_disk )) || die "only ${disk_gb}GB free (+${have_gb}GB already fetched), need ${need_disk}GB"
  awk -v v="$cuda_drv" 'BEGIN { split(v, a, "."); exit !(a[1] > 12 || (a[1] == 12 && a[2] >= 4)) }' \
    || die "driver supports CUDA $cuda_drv < 12.4"
}

fv_has_symbol() {
  local lib="$1" sym="$2"
  [[ -e "$lib" ]] || return 1
  # Prefer a string probe: works without binutils and avoids `set -e` /
  # pipefail quirks around `nm | grep -q` returning 1 on a miss.
  if grep -a -q "$sym" "$lib" 2>/dev/null; then
    return 0
  fi
  if command -v nm >/dev/null; then
    nm -D "$lib" 2>/dev/null | grep -F "$sym" >/dev/null 2>&1
  else
    return 1
  fi
}

fv_cudnn_ok() {
  local lib="$FV_LIBDIR/libcudnn.so"
  [[ -e "$lib" ]] || return 1
  fv_has_symbol "$lib" "$FV_CUDNN_REQUIRED_SYMBOL"
}

fv_cublas_ok() {
  local lib="$FV_LIBDIR/libcublas.so"
  [[ -e "$lib" ]] || return 1
  fv_has_symbol "$lib" "$FV_CUBLAS_REQUIRED_SYMBOL"
}

# Install CUDA 13 NVRTC/cuBLAS/cuDNN from NVIDIA apt when the image only has
# the pip CUDA-12 libs under /opt/nvidia-libs (ghcr :latest until the slim
# runtime image is rebuilt).
fv_ensure_cuda13_libs() {
  if fv_cublas_ok && fv_cudnn_ok; then
    return 0
  fi
  log "CUDA 13 runtime libs missing or too old — installing via apt"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null
  apt-get install -y -qq --no-install-recommends wget ca-certificates binutils >/dev/null
  if [[ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]] && [[ ! -f /etc/apt/sources.list.d/cuda*.list ]]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb >/dev/null
    apt-get update -qq >/dev/null
  fi
  # shellcheck source=scripts/gpu/cuda-13.pins
  . "$(dirname "${BASH_SOURCE[0]}")/cuda-13.pins"
  apt-get install -y -qq --no-install-recommends --allow-downgrades --allow-change-held-packages \
    "$CUDA_NVRTC_PKG" "$CUDA_CUBLAS_PKG" "$CUDA_CUDNN_PKG" >/dev/null
  apt-mark hold cuda-nvrtc-13-0 libcublas-13-0 libcudnn9-cuda-13 >/dev/null
  ldconfig >/dev/null 2>&1 || true
  fv_setup_libs
}

cmd_bootstrap() {
  # The binary arrives prebuilt (scripts/gpu/docker.sh dist); only runtime
  # helpers are installed here. ffmpeg (mp4 mux) installs in the background
  # and surfaces at the clip stage if it failed.
  [[ -x "$ROOT/target/release/fv-gpucheck" ]] || die "prebuilt fv-gpucheck missing (upload failed?)"
  "$ROOT/target/release/fv-gpucheck" --help >/dev/null || die "prebuilt fv-gpucheck does not run on this box (glibc/arch mismatch?)"
  command -v hf-fm >/dev/null || command -v hf-fetch-model >/dev/null \
    || die "hf-fm / hf-fetch-model missing (runtime image must bake hf-fetch-model)"
  if ! command -v ffmpeg >/dev/null; then
    nohup bash -c 'DEBIAN_FRONTEND=noninteractive apt-get update -qq && apt-get install -y -qq ffmpeg' \
      >"$LOGS/apt-ffmpeg.log" 2>&1 &
  fi
  fv_ensure_cuda13_libs
  fv_cudnn_ok || die "cuDNN at $(readlink "$FV_LIBDIR/libcudnn.so" 2>/dev/null || echo "$FV_LIBDIR/libcudnn.so") lacks $FV_CUDNN_REQUIRED_SYMBOL"
  local have
  have="$(fv_cublas_version)"
  if [[ -z "$have" ]] || [[ "$(printf '%s\n%s\n' "$FV_CUBLAS_MIN" "$have" | sort -V | head -1)" != "$FV_CUBLAS_MIN" ]] || ! fv_cublas_ok; then
    die "cuBLAS ${have:-unknown} at $(readlink "$FV_LIBDIR/libcublas.so" 2>/dev/null || echo?) lacks $FV_CUBLAS_REQUIRED_SYMBOL (need >= $FV_CUBLAS_MIN from the CUDA 13 CI image)"
  fi
  log "cublas $have"
}

# Print the loaded cuBLAS version (major.minor.patch) from the SONAME when possible.
fv_cublas_version() {
  local lib="$FV_LIBDIR/libcublas.so"
  [[ -e "$lib" ]] || return 0
  # libcublas.so.13.0.0 style; fall back to "13.0.0" from the CUDA package.
  local real
  real="$(readlink -f "$lib" 2>/dev/null || readlink "$lib" 2>/dev/null || echo "$lib")"
  if [[ "$(basename "$real")" =~ libcublas\.so\.([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.${BASH_REMATCH[3]}"
    return 0
  fi
  # CUDA 13 package default.
  echo "13.0.0"
}

# cublas <version>: no longer installs via pip; the image ships apt cuBLAS.
cmd_cublas() {
  log "cublas override ignored on slim image (apt CUDA 13); have $(fv_cublas_version)"
}

# fetch <repo> <dest> <glob>...: background download via hf-fm (no Python hub).
# hf-fm writes an HF cache tree under --output-dir; we promote the snapshot
# so dest/text_encoder etc. exist (what wait-weights and the binary expect).
cmd_fetch() {
  local repo="$1" dest="$2"; shift 2
  mkdir -p "$dest"
  command -v hf-fm >/dev/null || die "hf-fm not on PATH"
  # Already fetched (and promoted) on a reused box.
  if [[ -f "$dest/.complete" ]] && { [[ -d "$dest/text_encoder" ]] || [[ -d "$dest/transformer" ]] || [[ -d "$dest/vae" ]]; }; then
    log "weights already in $dest (skipping fetch)"
    return 0
  fi
  # Cache tree present but not promoted yet (failed wait on an earlier run).
  if [[ ! -d "$dest/text_encoder" && ! -d "$dest/transformer" ]]; then
    local snap
    snap="$(find "$dest" -type d -regex '.*/snapshots/[0-9a-f]+' 2>/dev/null | head -1 || true)"
    if [[ -n "${snap:-}" && -d "$snap" && -f "$dest/.complete" ]]; then
      log "promoting existing snapshot $snap → $dest"
      local p
      for p in "$snap"/*; do
        [[ -e "$p" ]] || continue
        ln -sfn "$p" "$dest/$(basename "$p")"
      done
      log "weights already in $dest (promoted; skipping fetch)"
      return 0
    fi
  fi
  log "fetching $repo → $dest via hf-fm (background)"
  local filters=()
  local pat
  for pat in "$@" "model_index.json"; do
    filters+=(--filter "$pat")
  done
  # Gated Hub packs (LTX-2.5): prefer env, else the token file seeded by validate.sh.
  nohup bash -c '
    set -euo pipefail
    repo="$1"; dest="$2"; shift 2
    t0=$(date +%s)
    export HF_HOME="${HF_HOME:-'"$WORK"'/hf}"
    mkdir -p "$HF_HOME"
    if [[ -z "${HF_TOKEN:-}" ]]; then
      for tok in "$HF_HOME/token" /root/.cache/huggingface/token; do
        if [[ -f "$tok" ]]; then export HF_TOKEN="$(tr -d "[:space:]" <"$tok")"; break; fi
      done
    fi
    hf-fm "$repo" --output-dir "$dest" "$@" --timeout-per-file-secs "${HF_FM_TIMEOUT_PER_FILE:-1800}"
    # Promote cache snapshot → dest/{tokenizer,text_encoder,...} when needed.
    if [[ ! -d "$dest/text_encoder" && ! -d "$dest/transformer" && ! -d "$dest/vae" ]]; then
      snap="$(find "$dest" -type d -regex ".*/snapshots/[0-9a-f]+" 2>/dev/null | head -1 || true)"
      if [[ -n "${snap:-}" && -d "$snap" ]]; then
        echo "promoting snapshot $snap → $dest"
        for p in "$snap"/*; do
          [[ -e "$p" ]] || continue
          ln -sfn "$p" "$dest/$(basename "$p")"
        done
      fi
    fi
    secs=$(( $(date +%s) - t0 ))
    size=$(du -sb "$dest" 2>/dev/null | cut -f1)
    echo "$secs" >"$dest/.complete"
    echo "done in ${secs}s: $(awk -v s="$size" "BEGIN{printf \"%.1f\", s/2^30}") GiB on disk" 
  ' _ "$repo" "$dest" "${filters[@]}" >"$LOGS/fetch-$(basename "$dest").log" 2>&1 &
  echo $! >"$dest/.fetch.pid"
}

# fetch-taehv <dest>: the TAEHV decoder weights (~14MB) from madebyollin's
# repo, in the background like `fetch`. `.complete` marks a verified file.
cmd_fetch_taehv() {
  local dest="$1"
  mkdir -p "$dest"
  if [[ -f "$dest/.complete" ]]; then log "taehv weights already in $dest"; return 0; fi
  log "fetching taew2_1.safetensors → $dest (background)"
  nohup bash "$ROOT/scripts/gpu/fetch_taehv.sh" "$dest" >"$LOGS/fetch-taehv.log" 2>&1 &
  echo $! >"$dest/.fetch.pid"
}

# wait-taehv <dest> <timeout_s>: block until fetch-taehv finished.
cmd_wait_taehv() {
  local dest="$1" timeout_s="$2" waited=0
  while [[ ! -f "$dest/.complete" ]]; do
    if [[ -f "$dest/.fetch.pid" ]] && ! kill -0 "$(cat "$dest/.fetch.pid")" 2>/dev/null && [[ ! -f "$dest/.complete" ]]; then
      tail -20 "$LOGS/fetch-taehv.log" >&2 || true
      die "taehv weight download died"
    fi
    (( waited < timeout_s )) || die "taehv weights not ready after ${timeout_s}s"
    sleep 2; waited=$((waited + 2))
  done
  log "taehv weights ok: $(du -h "$dest/taew2_1.safetensors" | cut -f1)"
}

# fetch-taeh3 <dest>: MiniMax-H3 tiny decoder (~10 MB) from madebyollin/taehv.
cmd_fetch_taeh3() {
  local dest="$1"
  mkdir -p "$dest"
  if [[ -f "$dest/.complete" ]]; then log "taeh3 weights already in $dest"; return 0; fi
  log "fetching taeh3.safetensors → $dest (background)"
  nohup bash "$ROOT/scripts/gpu/fetch-taeh3.sh" "$dest" >"$LOGS/fetch-taeh3.log" 2>&1 &
  echo $! >"$dest/.fetch.pid"
}

cmd_wait_taeh3() {
  local dest="$1" timeout_s="$2" waited=0
  while [[ ! -f "$dest/.complete" ]]; do
    if [[ -f "$dest/.fetch.pid" ]] && ! kill -0 "$(cat "$dest/.fetch.pid")" 2>/dev/null && [[ ! -f "$dest/.complete" ]]; then
      tail -20 "$LOGS/fetch-taeh3.log" >&2 || true
      die "taeh3 weight download died"
    fi
    (( waited < timeout_s )) || die "taeh3 weights not ready after ${timeout_s}s"
    sleep 2; waited=$((waited + 2))
  done
  log "taeh3 weights ok: $(du -h "$dest/taeh3.safetensors" | cut -f1)"
}

# wait_weights <dest> <timeout_s> <component>...: block until the background
# fetch finished and every shard of each component is complete on disk.
cmd_wait_weights() {
  local dest="$1" timeout_s="$2" waited=0; shift 2
  while [[ ! -f "$dest/.complete" ]]; do
    if [[ -f "$dest/.fetch.pid" ]] && ! kill -0 "$(cat "$dest/.fetch.pid")" 2>/dev/null && [[ ! -f "$dest/.complete" ]]; then
      tail -20 "$LOGS/fetch-$(basename "$dest").log" >&2 || true
      die "weight download for $dest died"
    fi
    (( waited < timeout_s )) || die "weights not ready after ${timeout_s}s ($(du -sh "$dest" 2>/dev/null | cut -f1) so far)"
    sleep 5; waited=$((waited + 5))
  done
  # hf-fm leaves an HF cache tree; promote snapshot so component dirs exist.
  if [[ ! -d "$dest/text_encoder" && ! -d "$dest/transformer" && ! -d "$dest/vae" ]]; then
    local snap
    snap="$(find "$dest" -type d -regex '.*/snapshots/[0-9a-f]+' 2>/dev/null | head -1 || true)"
    if [[ -n "${snap:-}" && -d "$snap" ]]; then
      log "promoting snapshot $snap → $dest"
      local p
      for p in "$snap"/*; do
        [[ -e "$p" ]] || continue
        ln -sfn "$p" "$dest/$(basename "$p")"
      done
    fi
  fi
  local c
  if (( $# == 0 )); then
    bash "$ROOT/scripts/gpu/verify-safetensors.sh" --dir "$dest"
  else
    for c in "$@"; do
      [[ -d "$dest/$c" ]] || die "missing component dir $dest/$c after fetch"
      bash "$ROOT/scripts/gpu/verify-safetensors.sh" --dir "$dest/$c"
    done
  fi
}

# stage <name> <timeout_s> <fv-gpucheck args...>
cmd_stage() {
  local name="$1" timeout_s="$2"; shift 2
  cd "$ROOT"
  local bin="$ROOT/target/release/fv-gpucheck"
  [[ -x "$bin" ]] || die "fv-gpucheck not uploaded"
  # Build id ties reports and CPU-path references to the exact binary.
  if [[ -f "$bin.build-id" ]]; then
    FV_GIT_SHA="$(cat "$bin.build-id")"
    export FV_GIT_SHA
  fi
  # CPU-path reference key (written by validate.sh at upload).
  if [[ -f "$OUT/refs/ref-key" ]]; then
    FV_REF_KEY="$(cat "$OUT/refs/ref-key")"
    export FV_REF_KEY
  fi
  log "stage $name (timeout ${timeout_s}s): $*"
  set +e
  timeout --kill-after=30 "$timeout_s" "$bin" --out "$OUT" "$@" 2>&1 | tee "$LOGS/$name.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    log "stage $name TIMED OUT after ${timeout_s}s"
    printf '{"stage":"%s","status":"timeout","timeout_s":%s}\n' "$name" "$timeout_s" >"$OUT/$name.timeout.json"
    exit 124
  fi
  exit "$rc"
}

# Upstream FastVideo in its own venv, for same-box comparison. uv brings its
# own Python 3.12 (the image's may be older) and torch wheels carry their own
# CUDA, so nothing here touches the libraries our binary dlopens.
UPSTREAM_VENV="$WORK/upstream-venv"

# Triton JIT-builds a small C extension the first time it talks to the driver
# (at import, in both upstream FastVideo and recent transformers), so anything
# that runs torch needs a C compiler our lean runtime image omits.
ensure_cc() {
  if ! command -v cc >/dev/null && ! command -v gcc >/dev/null; then
    log "installing gcc (Triton builds a driver shim at import)"
    apt-get update -qq >/dev/null 2>&1 || true
    if ! apt-get install -y -qq --no-install-recommends gcc g++ >/dev/null 2>&1; then
      # Some hosts pin a regional mirror that lags the image's own libgcc, so
      # apt tries to downgrade gcc-12-base and deadlocks. The official archive
      # carries the matching version.
      log "mirror cannot satisfy gcc; switching to archive.ubuntu.com"
      sed -i 's|http://[^ ]*/ubuntu|http://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list || true
      apt-get update -qq >/dev/null 2>&1 || true
      apt-get install -y -qq --no-install-recommends gcc g++ >/dev/null 2>&1 \
        || die "could not install a C compiler for Triton"
    fi
  fi
}

cmd_upstream_install() {
  local torch_backend="${1:-cu126}"
  export HF_HOME="$WORK/hf"
  mkdir -p "$HF_HOME"
  export PATH="$HOME/.local/bin:$PATH"
  ensure_cc
  export CC="${CC:-$(command -v gcc || command -v cc)}"
  if [[ ! -x "$UPSTREAM_VENV/bin/python" ]]; then
    command -v uv >/dev/null || {
      log "installing uv"
      curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || die "uv install failed"
    }
    export PATH="$HOME/.local/bin:$PATH"
    log "creating venv (python 3.12)"
    uv venv --python 3.12 --seed "$UPSTREAM_VENV" >&2 || die "uv venv failed"
  fi
  log "installing fastvideo (UV_TORCH_BACKEND=$torch_backend) — several minutes"
  VIRTUAL_ENV="$UPSTREAM_VENV" UV_TORCH_BACKEND="$torch_backend" \
    uv pip install --python "$UPSTREAM_VENV/bin/python" fastvideo >&2 || die "fastvideo install failed"
  "$UPSTREAM_VENV/bin/python" -c 'import torch, fastvideo; print("torch", torch.__version__, "cuda", torch.version.cuda, "fastvideo", getattr(fastvideo, "__version__", "?"))' >&2 \
    || die "fastvideo import failed"
  log "upstream install ok"
}

cmd_upstream_bench() {
  local backend="$1"; shift
  export HF_HOME="$WORK/hf"
  export PATH="$HOME/.local/bin:$PATH"
  export CC="${CC:-$(command -v gcc || command -v cc || true)}"
  [[ -x "$UPSTREAM_VENV/bin/python" ]] || die "upstream venv missing (run upstream-install)"
  "$UPSTREAM_VENV/bin/python" "$ROOT/scripts/gpu/upstream_bench.py" \
    --backend "$backend" --out "$OUT/upstream-$backend.json" --video-dir "$OUT/upstream-videos/$backend" "$@"
}

# The oracle runs in the upstream venv: transformers' UMT5 and diffusers'
# WanTransformer3DModel are the reference our own ports are judged against.
# oracle-venv [torch_backend]: transformers + diffusers (from git: the
# MiniMax-H3 and LTX-2 model classes are newer than any release) for the
# audio-video reference dumps. Separate from upstream-venv: it needs no
# fastvideo package and no Triton, so it installs in a fraction of the time.
ORACLE_VENV="$WORK/oracle-venv"
cmd_oracle_venv() {
  local torch_backend="${1:-cu130}"
  export HF_HOME="$WORK/hf"
  mkdir -p "$HF_HOME"
  export PATH="$HOME/.local/bin:$PATH"
  # Image baked /venv/main with transformers+diffusers (vast-oracle flavor).
  if [[ -n "${FV_ORACLE_PYTHON:-}" && -x "${FV_ORACLE_PYTHON}" ]] \
    || { [[ -x /venv/main/bin/python ]] && /venv/main/bin/python -c 'import torch, transformers, diffusers' 2>/dev/null; }; then
    ORACLE_VENV=/venv/main
    export FV_ORACLE_PYTHON="${FV_ORACLE_PYTHON:-/venv/main/bin/python}"
    log "oracle python prebaked ($FV_ORACLE_PYTHON) — skipping uv install"
    "$FV_ORACLE_PYTHON" -c 'import torch, transformers, diffusers; print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "diffusers", diffusers.__version__)' >&2 \
      || die "prebaked oracle python import failed"
    return 0
  fi
  ensure_cc
  command -v git >/dev/null || { apt-get update -qq >/dev/null 2>&1 || true; apt-get install -y -qq --no-install-recommends git >/dev/null 2>&1 || die "could not install git"; }
  if [[ ! -x "$ORACLE_VENV/bin/python" ]]; then
    command -v uv >/dev/null || {
      log "installing uv"
      curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || die "uv install failed"
    }
    export PATH="$HOME/.local/bin:$PATH"
    uv venv --python 3.12 --seed "$ORACLE_VENV" >&2 || die "uv venv failed"
  fi
  log "installing torch ($torch_backend), transformers, diffusers@main"
  # diffusers from a source tarball, not a git clone: one HTTP GET instead of a
  # pack negotiation that a flaky host link breaks ("RPC failed; curl 92"), and
  # no git needed. Retried, because a rented box's network owes us nothing.
  local attempt ok=0
  for attempt in 1 2 3; do
    if VIRTUAL_ENV="$ORACLE_VENV" UV_TORCH_BACKEND="$torch_backend" \
      uv pip install --python "$ORACLE_VENV/bin/python" torch torchvision transformers accelerate safetensors sentencepiece protobuf pillow numpy \
        "diffusers @ https://github.com/huggingface/diffusers/archive/refs/heads/main.tar.gz" >&2; then
      ok=1; break
    fi
    log "oracle venv install attempt $attempt failed; retrying"
    sleep $(( attempt * 10 ))
  done
  (( ok == 1 )) || die "oracle venv install failed"
  "$ORACLE_VENV/bin/python" -c 'import torch, transformers, diffusers; print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "diffusers", diffusers.__version__, "gpu", torch.cuda.get_device_name(0))' >&2 \
    || die "oracle venv import failed"
  log "oracle venv ok"
}

# model-oracle <h3|ltx2> <args...>: run scripts/gpu/<model>_oracle.py in the
# oracle venv. Outputs land wherever the args say (validate.sh passes $OUT paths).
cmd_model_oracle() {
  local model="$1"; shift
  export HF_HOME="$WORK/hf"
  export PATH="$HOME/.local/bin:$PATH"
  local py="${FV_ORACLE_PYTHON:-$ORACLE_VENV/bin/python}"
  [[ -x "$py" ]] || die "oracle python missing (run oracle-venv)"
  [[ -f "$ROOT/scripts/gpu/${model}_oracle.py" ]] || die "no oracle script for '$model'"
  mkdir -p "$OUT/$model"
  ensure_cc
  # Without our LD_LIBRARY_PATH: it points at the cuDNN/cuBLAS fv-gpucheck
  # loads, and torch resolving its main cuDNN library from there while its
  # sublibraries come from its own wheel ends in
  # CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. torch must see only its bundled CUDA.
  env -u LD_LIBRARY_PATH CC="${CC:-$(command -v gcc || command -v cc)}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_XET_HIGH_PERFORMANCE=1 \
    "$py" "$ROOT/scripts/gpu/${model}_oracle.py" "$@"
}

cmd_upstream_oracle() {
  export HF_HOME="$WORK/hf"
  export PATH="$HOME/.local/bin:$PATH"
  [[ -x "$UPSTREAM_VENV/bin/python" ]] || die "upstream venv missing (run upstream-install)"
  mkdir -p "$OUT/oracle"
  "$UPSTREAM_VENV/bin/python" "$ROOT/scripts/gpu/upstream_oracle.py" \
    --out "$OUT/oracle/oracle.safetensors" --meta "$OUT/oracle/oracle.json" "$@"
}

# TAEHV's reference implementation is a single file on GitHub, not a package;
# the script fetches it and the weights into $WORK/taehv, where our own stage
# reads the same checkpoint.
cmd_taehv_oracle() {
  export PATH="$HOME/.local/bin:$PATH"
  [[ -x "$UPSTREAM_VENV/bin/python" ]] || die "upstream venv missing (run upstream-install)"
  mkdir -p "$OUT/taehv"
  "$UPSTREAM_VENV/bin/python" "$ROOT/scripts/gpu/taehv_oracle.py" \
    --cache "$WORK/taehv" --out "$OUT/taehv/oracle.safetensors" --meta "$OUT/taehv/oracle.json" "$@"
}

sub="${1:-}"; shift || true
case "$sub" in
  env) cmd_env "$@" ;;
  libs) printf '%s\n' "$LD_LIBRARY_PATH"; ls -l "$FV_LIBDIR" ;;
  bootstrap) cmd_bootstrap ;;
  cublas) cmd_cublas "$@" ;;
  fetch) cmd_fetch "$@" ;;
  wait-weights) cmd_wait_weights "$@" ;;
  fetch-taehv) cmd_fetch_taehv "$@" ;;
  wait-taehv) cmd_wait_taehv "$@" ;;
  fetch-taeh3) cmd_fetch_taeh3 "$@" ;;
  wait-taeh3) cmd_wait_taeh3 "$@" ;;
  stage) cmd_stage "$@" ;;
  upstream-install) cmd_upstream_install "$@" ;;
  upstream-bench) cmd_upstream_bench "$@" ;;
  upstream-oracle) cmd_upstream_oracle "$@" ;;
  taehv-oracle) cmd_taehv_oracle "$@" ;;
  oracle-venv) cmd_oracle_venv "$@" ;;
  model-oracle) cmd_model_oracle "$@" ;;
  *) die "usage: remote.sh env|bootstrap|cublas|fetch|wait-weights|fetch-taehv|wait-taehv|fetch-taeh3|wait-taeh3|stage|upstream-install|upstream-bench|upstream-oracle|taehv-oracle|oracle-venv|model-oracle" ;;
esac
