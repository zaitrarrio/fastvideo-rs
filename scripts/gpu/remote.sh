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
FV_CUDNN_VERSION="${FV_CUDNN_VERSION:-9.26.0.51}"
FV_CUDNN_DIR="$WORK/fv-cudnn"
FV_CUDNN_REQUIRED_SYMBOL="cudnnBackendPopulateCudaGraph"
# `remote.sh cublas <version>` installs a different cuBLAS here; once present
# it replaces the image's for every later stage on this box.
FV_CUBLAS_DIR="$WORK/fv-cublas"
# Minimum cuBLAS: older builds lack Blackwell kernels (generic FP32 only).
FV_CUBLAS_MIN="${FV_CUBLAS_MIN:-12.9.1}"
FV_CUBLAS_VERSION="${FV_CUBLAS_VERSION:-12.9.1.4}"
fv_find_lib() {
  local name="$1" hit=""
  if [[ "$name" == cudnn && -e "$FV_CUDNN_DIR/nvidia/cudnn/lib/libcudnn.so.9" ]]; then
    readlink -f "$FV_CUDNN_DIR/nvidia/cudnn/lib/libcudnn.so.9"
    return 0
  fi
  if [[ "$name" == cublas* && -e "$FV_CUBLAS_DIR/nvidia/cublas/lib/lib$name.so.12" ]]; then
    readlink -f "$FV_CUBLAS_DIR/nvidia/cublas/lib/lib$name.so.12"
    return 0
  fi
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
  # Every library must actually load (catches wrong arch / missing deps), not just exist.
  python3 - "$FV_LIBDIR" <<'PY' || die "a CUDA runtime library failed to load"
import ctypes, os, sys
for lib in ("libnvrtc.so", "libcublasLt.so", "libcublas.so", "libcudnn.so"):
    ctypes.CDLL(os.path.join(sys.argv[1], lib))
print("cuda libs load ok")
PY
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

fv_cudnn_ok() {
  python3 - "$FV_LIBDIR/libcudnn.so" "$FV_CUDNN_REQUIRED_SYMBOL" <<'PY' 2>/dev/null
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
getattr(lib, sys.argv[2])
lib.cudnnGetVersion.restype = ctypes.c_size_t
print("cudnn", lib.cudnnGetVersion())
PY
}

cmd_bootstrap() {
  # The binary arrives prebuilt (scripts/gpu/docker.sh dist); only runtime
  # helpers are installed here. ffmpeg (mp4 mux) installs in the background
  # and surfaces at the clip stage if it failed.
  [[ -x "$ROOT/target/release/fv-gpucheck" ]] || die "prebuilt fv-gpucheck missing (upload failed?)"
  "$ROOT/target/release/fv-gpucheck" --help >/dev/null || die "prebuilt fv-gpucheck does not run on this box (glibc/arch mismatch?)"
  if ! command -v ffmpeg >/dev/null; then
    nohup bash -c 'DEBIAN_FRONTEND=noninteractive apt-get update -qq && apt-get install -y -qq ffmpeg' \
      >"$LOGS/apt-ffmpeg.log" 2>&1 &
  fi
  # Always the current downloader: huggingface_hub 1.x moves bytes through
  # hf_xet, and the image's copy is only as new as its last build.
  if ! python3 -m pip install -q -U huggingface_hub hf_xet >"$LOGS/pip-hf.log" 2>&1; then
    python3 -c 'import huggingface_hub' 2>/dev/null \
      || { tail -20 "$LOGS/pip-hf.log"; die "pip install huggingface_hub failed"; }
    log "could not upgrade huggingface_hub; using the image's copy"
  fi
  if ! fv_cudnn_ok; then
    log "image cuDNN lacks $FV_CUDNN_REQUIRED_SYMBOL; installing nvidia-cudnn-cu12==$FV_CUDNN_VERSION → $FV_CUDNN_DIR"
    python3 -m pip install -q --no-deps --target "$FV_CUDNN_DIR" "nvidia-cudnn-cu12==$FV_CUDNN_VERSION" \
      >"$LOGS/pip-cudnn.log" 2>&1 || { tail -20 "$LOGS/pip-cudnn.log"; die "cuDNN install failed"; }
    fv_setup_libs
  fi
  fv_cudnn_ok || die "cuDNN at $(readlink "$FV_LIBDIR/libcudnn.so") still lacks $FV_CUDNN_REQUIRED_SYMBOL"
  local have
  have="$(fv_cublas_version)"
  if [[ -z "$have" ]] || [[ "$(printf '%s\n%s\n' "$FV_CUBLAS_MIN" "$have" | sort -V | head -1)" != "$FV_CUBLAS_MIN" ]]; then
    log "image cuBLAS ${have:-unknown} < $FV_CUBLAS_MIN"
    cmd_cublas "$FV_CUBLAS_VERSION"
  fi
  log "cublas $(fv_cublas_version)"
}

# Print the loaded cuBLAS version (major.minor.patch).
fv_cublas_version() {
  python3 - "$FV_LIBDIR/libcublas.so" <<'PY' 2>/dev/null
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
v = ctypes.c_int()
out = []
for prop in (0, 1, 2):
    lib.cublasGetProperty(prop, ctypes.byref(v))
    out.append(str(v.value))
print(".".join(out))
PY
}

# cublas <version>: install nvidia-cublas-cu12==<version> beside the image's
# and switch the library shim to it.
cmd_cublas() {
  local version="${1:?usage: remote.sh cublas <version>}"
  log "installing nvidia-cublas-cu12==$version → $FV_CUBLAS_DIR"
  rm -rf "$FV_CUBLAS_DIR"
  python3 -m pip install -q --no-deps --target "$FV_CUBLAS_DIR" "nvidia-cublas-cu12==$version" \
    >"$LOGS/pip-cublas.log" 2>&1 || { tail -20 "$LOGS/pip-cublas.log"; die "cuBLAS install failed"; }
  fv_setup_libs
  python3 - "$FV_LIBDIR/libcublas.so" <<'PY' || die "installed cuBLAS does not load"
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
v = ctypes.c_int()
lib.cublasGetProperty(0, ctypes.byref(v)); major = v.value
lib.cublasGetProperty(1, ctypes.byref(v)); minor = v.value
lib.cublasGetProperty(2, ctypes.byref(v)); patch = v.value
print(f"cublas {major}.{minor}.{patch} at {sys.argv[1]}")
PY
}

# fetch <repo> <dest> <glob>...: background download of only the listed
# components (e.g. "transformer/*" "vae/*").
cmd_fetch() {
  local repo="$1" dest="$2"; shift 2
  mkdir -p "$dest"
  log "fetching $repo → $dest (background)"
  # HF_XET_HIGH_PERFORMANCE=1 is hf_xet's fast path: it saturates the link and
  # uses every core for chunk reassembly instead of the polite defaults (which
  # measured 45 MiB/s on a 940 Mbps, 208-core box). HF_HUB_ENABLE_HF_TRANSFER
  # is what this used to set; huggingface_hub 1.x dropped it and ignores it
  # silently, so it stays only for a box that still has a 0.x hub.
  nohup env HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    python3 - "$repo" "$dest" "$@" >"$LOGS/fetch-$(basename "$dest").log" 2>&1 <<'PY' &
import importlib.metadata as md, os, sys, time
from huggingface_hub import snapshot_download
repo, dest, patterns = sys.argv[1], sys.argv[2], sys.argv[3:]
def ver(p):
    try: return md.version(p)
    except Exception: return "absent"
print("huggingface_hub", ver("huggingface_hub"), "hf_xet", ver("hf_xet"),
      "HF_XET_HIGH_PERFORMANCE", os.environ.get("HF_XET_HIGH_PERFORMANCE"), flush=True)
t = time.time()
snapshot_download(repo, local_dir=dest, allow_patterns=patterns + ["model_index.json"], max_workers=16)
secs = time.time() - t
size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(dest) for f in fs if ".cache" not in r)
open(dest + "/.complete", "w").write(str(secs))
print(f"done in {secs:.0f} s: {size / 2**30:.1f} GiB on disk, {size / 2**20 / max(secs, 1e-9):.0f} MiB/s", flush=True)
PY
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
  python3 - "$dest" "$@" <<'PY'
import json, os, struct, sys
root, comps = sys.argv[1], sys.argv[2:]
n = 0
for comp in comps:
    d = os.path.join(root, comp)
    files = [f for f in os.listdir(d) if f.endswith(".safetensors")]
    if not files:
        sys.exit(f"no safetensors in {d}")
    for f in files:
        p = os.path.join(d, f)
        with open(p, "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(hlen))
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        if os.path.getsize(p) != 8 + hlen + end:
            sys.exit(f"truncated shard {p}")
        n += 1
print(f"weights ok: {n} shards verified under {root}")
PY
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
  VIRTUAL_ENV="$ORACLE_VENV" UV_TORCH_BACKEND="$torch_backend" \
    uv pip install --python "$ORACLE_VENV/bin/python" torch torchvision transformers accelerate safetensors sentencepiece protobuf pillow numpy \
      "diffusers @ git+https://github.com/huggingface/diffusers" >&2 || die "oracle venv install failed"
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
  [[ -x "$ORACLE_VENV/bin/python" ]] || die "oracle venv missing (run oracle-venv)"
  [[ -f "$ROOT/scripts/gpu/${model}_oracle.py" ]] || die "no oracle script for '$model'"
  mkdir -p "$OUT/$model"
  ensure_cc
  # Without our LD_LIBRARY_PATH: it points at the cuDNN/cuBLAS fv-gpucheck
  # loads, and torch resolving its main cuDNN library from there while its
  # sublibraries come from its own wheel ends in
  # CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. torch must see only its bundled CUDA.
  env -u LD_LIBRARY_PATH CC="${CC:-$(command -v gcc || command -v cc)}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_XET_HIGH_PERFORMANCE=1 \
    "$ORACLE_VENV/bin/python" "$ROOT/scripts/gpu/${model}_oracle.py" "$@"
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
  stage) cmd_stage "$@" ;;
  upstream-install) cmd_upstream_install "$@" ;;
  upstream-bench) cmd_upstream_bench "$@" ;;
  upstream-oracle) cmd_upstream_oracle "$@" ;;
  taehv-oracle) cmd_taehv_oracle "$@" ;;
  oracle-venv) cmd_oracle_venv "$@" ;;
  model-oracle) cmd_model_oracle "$@" ;;
  *) die "usage: remote.sh env|bootstrap|cublas|fetch|wait-weights|fetch-taehv|wait-taehv|stage|upstream-install|upstream-bench|upstream-oracle|taehv-oracle|oracle-venv|model-oracle" ;;
esac
