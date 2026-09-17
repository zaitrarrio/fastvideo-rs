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
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$LOGS"

log() { printf '[remote %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 2; }

cmd_env() {
  # Cheapest possible gate: is this box usable at all?
  command -v nvidia-smi >/dev/null || die "nvidia-smi missing (not a GPU container?)"
  local q
  q="$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits | head -1)" \
    || die "nvidia-smi failed — driver not usable"
  IFS=',' read -r name mem driver cap <<<"$q"
  local cuda_drv
  cuda_drv="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
  local disk_gb ram_gb cores
  disk_gb="$(df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9)"
  ram_gb="$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)"
  cores="$(nproc)"
  local libs=()
  for lib in libnvrtc.so libcublas.so libcublasLt.so libcudnn.so; do
    if ldconfig -p | grep -q "$lib" || compgen -G "/usr/local/cuda/lib64/${lib}*" >/dev/null; then
      libs+=("\"$lib\":true")
    else
      libs+=("\"$lib\":false")
    fi
  done
  local json
  json="$(printf '{"gpu":"%s","vram_mib":%s,"driver":"%s","compute_cap":"%s","driver_cuda":"%s","disk_free_gb":%s,"ram_gb":%s,"cores":%s,"libs":{%s}}' \
    "$(xargs <<<"$name")" "$(xargs <<<"$mem")" "$(xargs <<<"$driver")" "$(xargs <<<"$cap")" "$cuda_drv" \
    "$disk_gb" "$ram_gb" "$cores" "$(IFS=,; echo "${libs[*]}")")"
  echo "$json" | tee "$OUT/env.json"
  [[ "$json" != *'false'* ]] || die "missing CUDA runtime libraries: $json"
  local need_disk="${1:-30}"
  (( disk_gb >= need_disk )) || die "only ${disk_gb}GB free, need ${need_disk}GB"
  awk -v v="$cuda_drv" 'BEGIN { split(v, a, "."); exit !(a[1] > 12 || (a[1] == 12 && a[2] >= 4)) }' \
    || die "driver supports CUDA $cuda_drv < 12.4"
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
  if ! python3 -c 'import huggingface_hub, hf_transfer' 2>/dev/null; then
    python3 -m pip install -q 'huggingface_hub[hf_transfer]' >"$LOGS/pip-hf.log" 2>&1 \
      || { tail -20 "$LOGS/pip-hf.log"; die "pip install huggingface_hub failed"; }
  fi
}

# fetch <repo> <dest> <glob>...: background download of only the listed
# components (e.g. "transformer/*" "vae/*").
cmd_fetch() {
  local repo="$1" dest="$2"; shift 2
  mkdir -p "$dest"
  log "fetching $repo → $dest (background)"
  nohup env HF_HUB_ENABLE_HF_TRANSFER=1 python3 - "$repo" "$dest" "$@" >"$LOGS/fetch-$(basename "$dest").log" 2>&1 <<'PY' &
import sys, time
from huggingface_hub import snapshot_download
repo, dest, patterns = sys.argv[1], sys.argv[2], sys.argv[3:]
t = time.time()
snapshot_download(repo, local_dir=dest, allow_patterns=patterns + ["model_index.json"])
open(dest + "/.complete", "w").write(str(time.time() - t))
print("done in", time.time() - t, "s")
PY
  echo $! >"$dest/.fetch.pid"
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

sub="${1:-}"; shift || true
case "$sub" in
  env) cmd_env "$@" ;;
  bootstrap) cmd_bootstrap ;;
  fetch) cmd_fetch "$@" ;;
  wait-weights) cmd_wait_weights "$@" ;;
  stage) cmd_stage "$@" ;;
  *) die "usage: remote.sh env|bootstrap|fetch|wait-weights|stage" ;;
esac
