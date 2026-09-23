#!/usr/bin/env bash
# Tiered, fail-fast validation of the cudarc Wan backend on the cheapest
# hardware that can prove each property. See scripts/gpu/README.md.
#
#   validate.sh local                 free       preflight: unit tests + CUDA feature type-check
#   validate.sh offers <tier>                    cheapest matching offers + cost cap (no rental)
#   validate.sh run <tier> [opts]     T1-T3      rent → stages → pull artifacts → destroy
#   validate.sh reap                             destroy every fvgpu-* instance
#
# tiers: kernels (T1) | parity (T2) | clip (T3) | compare (T4, + upstream FastVideo)
#        each validation tier includes the ones before it.
#        gen — the UI path: deploy, encode one prompt, generate one clip.
#        oracle — our text encoder and one DiT step vs transformers/diffusers.
#        fp8 — the same clip with and without FP8 linears, on one box.
#        taehv — our tiny-autoencoder decoder vs madebyollin's own.
#        vaeab — the same clip decoded by the Wan VAE and by TAEHV.
set -euo pipefail
# shellcheck source=scripts/gpu/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

RUNS="${FV_RUNS:-$FV_ROOT/artifacts/gpucheck/runs}"
DOCKER_SH="$FV_ROOT/scripts/gpu/docker.sh"
DIST="$FV_ROOT/artifacts/gpucheck/dist"
LOCAL_REFS="$FV_ROOT/artifacts/gpucheck/refs"
# Slim runtime (default): .github/workflows/gpucheck-runtime-image.yml — CUDA 13
# libs + fv-gpucheck, no PyTorch. Vast pytorch images are only for tiers that
# need Python+torch; see image_flavor_for_tier / resolve_image.
IMAGE_REPO_SLIM="${VAST_IMAGE_REPO:-ghcr.io/zaitrarrio/fastvideo-rs-runtime}"
IMAGE_REPO_PYTORCH="${VAST_IMAGE_REPO_PYTORCH:-ghcr.io/zaitrarrio/fastvideo-rs-vast}"
IMAGE_REPO_ORACLE="${VAST_IMAGE_REPO_ORACLE:-ghcr.io/zaitrarrio/fastvideo-rs-vast-oracle}"
IMAGE_REPO="$IMAGE_REPO_SLIM"
IMAGE="${VAST_IMAGE:-}"
# slim | pytorch | oracle | auto (default: pick from tier). Override with VAST_IMAGE_FLAVOR.
IMAGE_FLAVOR="${VAST_IMAGE_FLAVOR:-auto}"
# The image bakes the binary and scripts here; uploads land in the same place.
FV_REMOTE_DIR="${VAST_REMOTE_DIR:-/opt/fastvideo-rs}"
# Machines that failed to boot / never accepted ssh / failed env or bootstrap.
BAD_HOSTS="${FV_BAD_HOSTS_FILE:-$FV_ROOT/artifacts/gpucheck/bad-machines.tsv}"
BAD_HOST_TTL_H="${FV_BAD_HOST_TTL_H:-24}"
CURRENT_MACHINE=""; LAST_STAGE=""; IMAGE_HAS_BUILD=0
WORK="/workspace"
OUTR="$WORK/gpucheck-out"
BASE_REPO="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
# FV_FAST_REPO swaps the distilled checkpoint without touching the tier code —
# FastWan-QAD is the same architecture, so only the weights differ.
FAST_REPO="${FV_FAST_REPO:-FastVideo/FastWan2.1-T2V-1.3B-Diffusers}"
BASE_W="$WORK/weights/wan21-1.3b"
FAST_W="$WORK/weights/fastwan21-1.3b"

# ---- tier definitions ---------------------------------------------------------
# Offer filter, $/hr ceiling, hard wall-clock cap (watchdog destroys at the cap).
# FV_GPU_RAM_MIN overrides a tier's VRAM floor (e.g. 78 to admit an 80 GB A100
# for a model that peaks well under it); FV_DISK_GB the disk rented, for a box
# that will be kept and reused by tiers needing more than the first one.
tier_query() { local q; q="$(tier_query_base "$1")" || return 1
  if [[ -n "${FV_GPU_RAM_MIN:-}" ]]; then q="$(sed -E "s/gpu_ram>=[0-9]+/gpu_ram>=${FV_GPU_RAM_MIN}/" <<<"$q")"; fi
  # Cap VRAM (e.g. H100-80: exclude NVL 94). Appended; Vast ANDs query terms.
  if [[ -n "${FV_GPU_RAM_MAX:-}" ]]; then q="$q gpu_ram<=${FV_GPU_RAM_MAX}"; fi
  echo "$q"; }
tier_query_base() {
  # FV_OFFER_QUERY_EXTRA narrows the search, e.g. "gpu_name=RTX_4070S".
  # cuda_vers>=13.0 is the driver floor for the CUDA 13.0 runtime libraries
  # the image ships (>= 580); there is no minor-version compatibility across
  # a major, so a 12.8-driver box would fail at cuBLAS load.
  local base="num_gpus=1 compute_cap>=800 cuda_vers>=13.0 reliability>0.97 rentable=true verified=true direct_port_count>=1 inet_down>=200 ${FV_OFFER_QUERY_EXTRA:-}"
  case "$1" in
    kernels) echo "$base gpu_ram>=8 disk_space>=40 cpu_ram>=16" ;;
    # cuBLAS math-mode probe: DiT linears at 8s-clip size (~2 GB of buffers).
    mathprobe) echo "$base gpu_ram>=12 disk_space>=40 cpu_ram>=16" ;;
    parity) echo "$base gpu_ram>=16 disk_space>=60 cpu_ram>=32 inet_down>=500" ;;
    # UMT5-XXL loads ~60GB of host RAM (raw bytes + F32 views) before upload.
    clip) echo "$base gpu_ram>=24 disk_space>=100 cpu_ram>=80 inet_down>=500" ;;
    # Same box runs our clip stages and upstream FastVideo: + torch wheels and
    # upstream's own copy of the weights in the HF cache.
    compare) echo "$base gpu_ram>=24 disk_space>=200 cpu_ram>=80 inet_down>=500" ;;
    # `gen` is the UI's path: deploy, encode one prompt, generate one clip.
    gen) echo "$base gpu_ram>=24 disk_space>=100 cpu_ram>=80 inet_down>=500" ;;
    # The oracle holds UMT5-XXL in torch float32 (~22GB) before the DiT loads;
    # 24GB is too tight to risk a rental on.
    oracle) echo "$base gpu_ram>=40 disk_space>=160 cpu_ram>=80 inet_down>=500" ;;
    # FP8 E4M3 tensor cores start at Ada (sm89); Ampere has none.
    fp8) echo "$base compute_cap>=890 gpu_ram>=24 disk_space>=100 cpu_ram>=80 inet_down>=500" ;;
    # TAEHV is ~10M parameters and needs no Wan weights at all — only torch,
    # which is why this tier is cheap despite installing the upstream venv.
    taehv) echo "$base gpu_ram>=12 disk_space>=60 cpu_ram>=16" ;;
    # The A/B decodes the same clip both ways, so it needs the Wan weights too.
    vaeab) echo "$base gpu_ram>=24 disk_space>=100 cpu_ram>=80 inet_down>=500" ;;
    # The audio-video ports (docs/ports/{h3,ltx2}.md). The references hold a
    # 12B-32B text encoder or a 19B-33B DiT in bf16 on the GPU, and the
    # checkpoints are 80-150 GB, so these ask for a 96 GB card, a big disk and
    # a fast link. `-text` is the first milestone: the text encoder alone.
    h3-text) echo "$base gpu_ram>=90 disk_space>=220 cpu_ram>=64 inet_down>=800" ;;
    # The decoders alone: ~10 GB of weights, held in float32 by both sides.
    h3-vae) echo "$base gpu_ram>=32 disk_space>=100 cpu_ram>=48 inet_down>=500" ;;
    ltx2-vae) echo "$base gpu_ram>=24 disk_space>=80 cpu_ram>=32 inet_down>=500" ;;
    # One DiT forward: diffusers holds the 66 GB bf16 transformer, then ours
    # holds 37 GiB of it (AdaLN precomputed) plus ~10 GiB of activations.
    h3-dit) echo "$base gpu_ram>=90 disk_space>=200 cpu_ram>=64 inet_down>=800" ;;
    # Gemma, connectors and the DiT for the reference (90 GB), plus the official
    # single-file DiT our side loads through the rename view (43 GB).
    ltx2-dit) echo "$base gpu_ram>=90 disk_space>=260 cpu_ram>=64 inet_down>=800" ;;
    # VSA-H3 on the device against f64 host loops: no weights at all, but the
    # tensor-core fine kernel wants sm80+ (the base filter) and head dim 128.
    h3-vsa) echo "$base gpu_ram>=12 disk_space>=40 cpu_ram>=16" ;;
    # Text to mp4-with-audio. H3 reads 50 GB of Qwen3-VL (shards 1-11), the
    # 70 GB DiT and both decoders; LTX-2 reads 49 GB of Gemma, the 43 GB
    # single-file DiT and its decoders. No python, no oracle: our binary only.
    h3-gen) echo "$base gpu_ram>=90 disk_space>=220 cpu_ram>=64 inet_down>=1000" ;;
    # Encoder × residency matrix: streamed 32B / recovered-8B on 80 GB cards;
    # FV_GPU_RAM_MIN=78 (A100/H100) or 140 (H200). TAEH3, no oracle.
    h3-matrix) echo "$base gpu_ram>=78 disk_space>=360 cpu_ram>=64 inet_down>=1000" ;;
    # LTX-2.5 Diffusers is ~100 GiB: advertised ≤2 Gbps hosts often crawl; require 4 Gbps+.
    ltx2-gen) echo "$base gpu_ram>=90 disk_space>=200 cpu_ram>=64 inet_down>=4000" ;;
    ltx2-text) echo "$base gpu_ram>=90 disk_space>=200 cpu_ram>=64 inet_down>=800" ;;
    # A build box: the GPU is irrelevant, so this asks for the cheapest thing
    # with cores and RAM for a release cargo build plus nvcc for 7 SMs.
    build) echo "num_gpus=1 cuda_vers>=13.0 reliability>0.97 rentable=true verified=true direct_port_count>=1 inet_down>=200 cpu_cores>=8 cpu_ram>=16 disk_space>=40 ${FV_OFFER_QUERY_EXTRA:-}" ;;
    *) die "unknown tier '$1' (mathprobe|kernels|parity|clip|compare|gen|oracle|fp8|taehv|vaeab|build|h3-text|ltx2-text|h3-vae|ltx2-vae|h3-dit|ltx2-dit|h3-vsa|h3-gen|h3-matrix|ltx2-gen)" ;;
  esac
}
tier_max_dph() { case "$1" in mathprobe) echo 0.40 ;; kernels) echo 0.25 ;; parity) echo 0.40 ;; clip) echo 0.60 ;; compare) echo 0.60 ;; gen) echo 0.80 ;; oracle) echo 1.60 ;; fp8) echo 0.80 ;; taehv) echo 0.40 ;; vaeab) echo 0.80 ;; build) echo 0.20 ;; h3-text | ltx2-text) echo 2.00 ;; h3-vae) echo 1.00 ;; ltx2-vae) echo 0.80 ;; h3-dit | ltx2-dit | h3-gen | ltx2-gen | h3-matrix) echo 2.50 ;; h3-vsa) echo 0.40 ;; esac; }
tier_max_minutes() { case "$1" in mathprobe) echo 30 ;; kernels) echo 40 ;; parity) echo 75 ;; clip) echo 180 ;; compare) echo 240 ;; gen) echo 180 ;; oracle) echo 150 ;; fp8) echo 90 ;; taehv) echo 60 ;; vaeab) echo 90 ;; build) echo 45 ;; h3-text | ltx2-text) echo 150 ;; h3-vae | ltx2-vae) echo 90 ;; h3-dit | ltx2-dit) echo 180 ;; h3-gen | ltx2-gen) echo 150 ;; h3-matrix) echo 360 ;; h3-vsa) echo 40 ;; esac; }
tier_disk() { case "$1" in mathprobe) echo 40 ;; kernels) echo 40 ;; parity) echo 60 ;; clip) echo 100 ;; compare) echo 200 ;; gen) echo 100 ;; oracle) echo 160 ;; fp8) echo 100 ;; taehv) echo 60 ;; vaeab) echo 100 ;; build) echo 40 ;; h3-text) echo 220 ;; ltx2-text) echo 200 ;; h3-vae) echo 100 ;; ltx2-vae) echo 80 ;; h3-dit) echo 200 ;; ltx2-dit) echo 260 ;; h3-vsa) echo 40 ;; h3-gen) echo 220 ;; h3-matrix) echo 360 ;; ltx2-gen) echo 200 ;; esac; }

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

# ---- bad hosts ------------------------------------------------------------------
bad_host_record() {
  local machine="$1" reason="$2"
  [[ -n "$machine" ]] || return 0
  mkdir -p "$(dirname "$BAD_HOSTS")"
  printf '%s\t%s\t%s\n' "$machine" "$(date +%s)" "$reason" >>"$BAD_HOSTS"
  log "recorded machine $machine as bad for ${BAD_HOST_TTL_H}h ($reason)"
}

# JSON array of machine ids recorded within the TTL.
bad_hosts_json() {
  local cutoff=$(( $(date +%s) - BAD_HOST_TTL_H * 3600 ))
  if [[ -f "$BAD_HOSTS" ]]; then
    awk -F'\t' -v c="$cutoff" '$2 >= c { print $1 }' "$BAD_HOSTS" | sort -u | jq -R 'tonumber? // empty' | jq -s -c .
  else
    echo '[]'
  fi
}

# ---- image ------------------------------------------------------------------------
# Slim = default (cudarc only). Vast pytorch/oracle images only when the tier
# runs Python with torch (transformers/diffusers/FastVideo).
image_flavor_for_tier() {
  case "$1" in
    # Prebaked transformers/diffusers in /venv/main.
    oracle | h3-text | ltx2-text | h3-vae | ltx2-vae | h3-dit | ltx2-dit) echo oracle ;;
    # Torch present; upstream/taehv still install their own venv on top.
    compare | taehv | vaeab) echo pytorch ;;
    # Everything else is fv-gpucheck only — stay on the slim image.
    *) echo slim ;;
  esac
}

# Prefer the CI image built from exactly these sources (binary baked in, no
# upload); otherwise :latest for the runtime libraries plus an uploaded binary.
resolve_image() {
  local build_id="$1" tier="${2:-}" flavor="$IMAGE_FLAVOR"
  if [[ -n "$IMAGE" ]]; then
    log "image $IMAGE (VAST_IMAGE override; uploading binary)"
    return 0
  fi
  if [[ "$flavor" == "auto" ]]; then
    if [[ -n "$tier" ]]; then
      flavor="$(image_flavor_for_tier "$tier")"
    else
      flavor=slim
    fi
  fi
  case "$flavor" in
    slim) IMAGE_REPO="$IMAGE_REPO_SLIM" ;;
    pytorch) IMAGE_REPO="$IMAGE_REPO_PYTORCH" ;;
    oracle) IMAGE_REPO="$IMAGE_REPO_ORACLE" ;;
    *) die "unknown VAST_IMAGE_FLAVOR='$flavor' (slim|pytorch|oracle|auto)" ;;
  esac
  IMAGE_FLAVOR="$flavor"
  if command -v docker >/dev/null 2>&1 && docker manifest inspect "$IMAGE_REPO:build-$build_id" >/dev/null 2>&1; then
    IMAGE="$IMAGE_REPO:build-$build_id"
    IMAGE_HAS_BUILD=1
    log "image $IMAGE (flavor=$flavor; CI-built from these sources; binary baked in)"
  else
    IMAGE="$IMAGE_REPO:latest"
    log "image $IMAGE (flavor=$flavor; no CI image for build $build_id yet — push to publish one; uploading local binary)"
  fi
}

# ---- preflight: local, free ------------------------------------------------------
# Compile-level gates only: catches a broken build before paying for one.

cmd_local() {
  log "preflight: shell lint"
  bash "$FV_ROOT/scripts/gpu/lint.sh" || die "shell lint failed" 1
  # Everything builds and runs in Docker (scripts/gpu/docker.sh), not on the host.
  log "preflight: unit tests (Docker)"
  "$DOCKER_SH" test || die "unit tests failed" 1
  log "preflight: NVRTC compile gate (Docker, libnvrtc only)"
  "$DOCKER_SH" nvrtc || die "NVRTC kernel compile failed — no GPU run can pass; see artifacts/gpucheck/local/nvrtc.json" 1
  log "preflight: release binary for the rented box (Docker)"
  "$DOCKER_SH" dist || die "dist build failed" 1
  log "preflight PASS"
}

# ---- offers -----------------------------------------------------------------------

offers_json() {
  local tier="$1"
  vastai search offers "$(tier_query "$tier")" -o 'dph_total' --raw 2>/dev/null \
    | jq '[.[] | {id, machine_id, gpu_name, gpu_ram: (.gpu_ram/1024|floor), dph_total, reliability: (.reliability2 // .reliability),
               inet_down, cpu_ram: ((.cpu_ram // 0)/1024|floor), compute_cap, cuda_max_good, geolocation}]'
}

cmd_offers() {
  local tier="${1:-}"; [[ -n "$tier" ]] || usage
  require_tools vastai jq
  vast_check_auth
  local max_dph="${MAX_DPH:-$(tier_max_dph "$tier")}" mins
  mins="${MAX_MINUTES:-$(tier_max_minutes "$tier")}"
  offers_json "$tier" | jq -r --argjson max "$max_dph" --argjson mins "$mins" '
    (map(select(.dph_total <= $max)) | .[0:8]) as $ok
    | "query ceiling $\($max)/hr, wall cap \($mins) min",
      (if ($ok|length)==0 then "NO OFFERS under the ceiling" else
        ($ok[] | "  offer \(.id)  \(.gpu_name) \(.gpu_ram)GB  $\(.dph_total*1000|round/1000)/hr  sm\(.compute_cap)  rel \(.reliability*1000|round/1000)  \(.inet_down|round)Mbps  \(.geolocation)  worst-case $\((.dph_total*$mins/60)*100|round/100)")
      end)'
}

# ---- run ---------------------------------------------------------------------------

REF_KEY=""
INSTANCE=""; HOST=""; PORT=""; RUN_DIR=""; KEEP=0; OWN_INSTANCE=1; WATCHDOG_PID=""; DPH=0; T_START=0

cleanup() {
  local rc=$?
  # Signal traps pass the conventional code; `$?` there is the interrupted
  # command's status (often 0), which would report a killed run as passing.
  [[ -n "${1:-}" ]] && rc="$1"
  trap - EXIT INT TERM
  # A host that can't get through env/bootstrap is a host problem, not a test result.
  if [[ $rc -ne 0 && -n "$CURRENT_MACHINE" ]]; then
    case "$LAST_STAGE" in
      env|bootstrap|upload) bad_host_record "$CURRENT_MACHINE" "$LAST_STAGE rc=$rc" ;;
    esac
  fi
  # FV_NO_ARTIFACTS=1 (build-remote.sh): the box has no validation output dir.
  if [[ -n "$HOST" && -n "$RUN_DIR" && "${FV_NO_ARTIFACTS:-0}" != 1 ]]; then
    log "pulling artifacts → $RUN_DIR"
    fv_timeout 300 rsync -az -e "ssh -i $FV_SSH_KEY -p $PORT ${FV_SSH_OPTS[*]}" "root@$HOST:$OUTR/" "$RUN_DIR/remote/" \
      || log "artifact pull failed/timed out"
    collect_clips
  fi
  if [[ -n "$WATCHDOG_PID" ]]; then kill "$WATCHDOG_PID" 2>/dev/null || true; fi
  if [[ -n "$INSTANCE" && $OWN_INSTANCE -eq 1 && $KEEP -eq 0 ]]; then
    vast_destroy "$INSTANCE" || true
  elif [[ -n "$INSTANCE" ]]; then
    log "leaving instance $INSTANCE running (--keep/--instance). Destroy: vastai destroy instance -y $INSTANCE"
  fi
  if [[ -n "$RUN_DIR" ]]; then
    local mins cost
    mins=$(( ($(date +%s) - T_START) / 60 ))
    cost="$(awk -v d="$DPH" -v s="$(( $(date +%s) - T_START ))" 'BEGIN { printf "%.3f", d * s / 3600 }')"
    jq -n --arg status "$([[ $rc -eq 0 ]] && echo pass || echo fail)" --argjson rc "$rc" \
      --arg instance "$INSTANCE" --argjson dph "$DPH" --argjson minutes "$mins" --argjson cost "$cost" \
      --slurpfile stages <(cat "$RUN_DIR/stages.jsonl" 2>/dev/null || true) \
      '{status: $status, exit_code: $rc, instance: $instance, dph: $dph, wall_minutes: $minutes, est_cost_usd: $cost, stages: $stages}' \
      >"$RUN_DIR/summary.json" 2>/dev/null || true
    log "run $([[ $rc -eq 0 ]] && echo PASSED || echo "FAILED rc=$rc") in ${mins} min, ~\$${cost} — $RUN_DIR/summary.json"
    if bash "$FV_ROOT/scripts/gpu/timings.sh" "$RUN_DIR" >"$RUN_DIR/.timings.out" 2>&1; then
      log "timings (also $RUN_DIR/timings.md):"
      cat "$RUN_DIR/.timings.out" >&2
    fi
    rm -f "$RUN_DIR/.timings.out"
  fi
  exit "$rc"
}

# remote_run <name> <timeout_s> <remote.sh args...>
# Detached on the box (survives ssh drops); polled with short ssh calls.
remote_run() {
  local name="$1" timeout_s="$2"; shift 2
  local args; args="$(printf '%q ' "$@")"
  local t0; t0=$(date +%s)
  LAST_STAGE="$name"
  log "▶ $name (timeout ${timeout_s}s)"
  # FV_STAGE_ENV ("FASTVIDEO_SDPA=flash FASTVIDEO_TEACACHE=1") reaches the stage
  # process itself; gpucheck records every FASTVIDEO_* it ran under.
  fv_ssh "$HOST" "$PORT" "mkdir -p $OUTR/rc $OUTR/logs && rm -f $OUTR/rc/$name $OUTR/rc/$name.pid && cd $FV_REMOTE_DIR && \
    { setsid nohup bash -c 'env ${FV_STAGE_ENV:-} bash scripts/gpu/remote.sh $args >$OUTR/logs/$name.driver.log 2>&1; echo \$? >$OUTR/rc/$name' \
      </dev/null >/dev/null 2>&1 & echo \$! >$OUTR/rc/$name.pid; }" || {
    # ssh worked at boot and has now stopped working: the host is the problem,
    # so record it rather than rent it again on the next run.
    bad_host_record "$CURRENT_MACHINE" "ssh died before $name"
    die "could not start $name"
  }
  local offset=0 fails=0 deadline=$(( t0 + timeout_s + 180 )) rc="" alive
  while :; do
    sleep "${FV_POLL_S:-10}"
    local chunk
    # Liveness is read *before* the exit code, so "dead and no rc" is never a race.
    if chunk="$(fv_ssh "$HOST" "$PORT" "a=0; kill -0 \$(cat $OUTR/rc/$name.pid 2>/dev/null) 2>/dev/null && a=1; \
        tail -c +$((offset + 1)) $OUTR/logs/$name.driver.log 2>/dev/null | head -c 65536; \
        printf '\n@@RC@@%s@@ALIVE@@%s' \"\$(cat $OUTR/rc/$name 2>/dev/null)\" \"\$a\"")"; then
      fails=0
      local text="${chunk%@@RC@@*}"
      local tail_part="${chunk##*@@RC@@}"
      rc="${tail_part%@@ALIVE@@*}"
      alive="${tail_part##*@@ALIVE@@}"
      text="${text%$'\n'}"
      if [[ -n "$text" ]]; then
        # Matrix cells set FV_TEST_ID so every streamed line carries the id.
        local prefix="[$name]"
        [[ -n "${FV_TEST_ID:-}" ]] && prefix="[${FV_TEST_ID}]"
        printf '%s\n' "${text%$'\n'}" | sed "s/^/  $prefix /" >&2
        offset=$(( offset + $(printf '%s' "$text" | wc -c) ))
      fi
      [[ -n "$rc" ]] && break
      if [[ "$alive" != 1 ]]; then
        rc=2
        log "$name: process exited without an exit code (crashed/killed/OOM)"
        break
      fi
    else
      fails=$((fails + 1))
      log "ssh poll failed ($fails/12)"
      (( fails < 12 )) || die "lost contact with instance during $name"
    fi
    (( $(date +%s) < deadline )) || die "$name exceeded ${timeout_s}s (+grace)" 124
  done
  local secs=$(( $(date +%s) - t0 ))
  printf '{"name":"%s","rc":%s,"seconds":%s}\n' "$name" "$rc" "$secs" >>"$RUN_DIR/stages.jsonl"
  # Reports and finished clips are small; pull after every stage so a later
  # failure (or a lost instance) keeps them.
  pull_outputs
  if [[ "$rc" != "0" ]]; then
    # An optional stage (a backend that will not build or run) is a result to
    # record, not a reason to stop the run.
    if [[ "${STAGE_OPTIONAL:-0}" == 1 ]]; then
      log "✗ $name failed rc=$rc after ${secs}s — continuing (optional)"
      return "$rc"
    fi
    log "✗ $name failed rc=$rc after ${secs}s — stopping (fail-fast). Report: $RUN_DIR/remote/"
    exit "$rc"
  fi
  log "✓ $name ${secs}s"
}

# pull_outputs: reports, latents, contact sheets and clip mp4s (not the PNG
# frames), then copy each clip into artifacts/clips/<run>/ where they collect.
pull_outputs() {
  fv_rsync_from "$HOST" "$PORT" "$OUTR/" "$RUN_DIR/remote/" \
    --include 'clips/*/frames/*.mp4' --exclude 'clips/*/frames/*' >/dev/null 2>&1 || true
  collect_clips
}

collect_clips() {
  local dest="$FV_ROOT/artifacts/clips/$(basename "$RUN_DIR")" dir name
  for dir in "$RUN_DIR"/remote/clips/*/; do
    [[ -f "$dir/frames/output.mp4" ]] || continue
    name="$(basename "$dir")"
    [[ -f "$dest/$name.mp4" ]] && continue
    mkdir -p "$dest"
    cp "$dir/frames/output.mp4" "$dest/$name.mp4"
    [[ -f "$dir/contact_sheet.png" ]] && cp "$dir/contact_sheet.png" "$dest/$name.png"
    log "saved clip → $dest/$name.mp4"
  done
  # Upstream FastVideo's own mp4s (compare tier), kept beside ours.
  local mp4
  while IFS= read -r mp4; do
    [[ -n "$mp4" ]] || continue
    name="upstream-$(basename "$(dirname "$mp4")")-$(basename "$mp4")"
    [[ -f "$dest/$name" ]] && continue
    mkdir -p "$dest"
    cp "$mp4" "$dest/$name"
    log "saved clip → $dest/$name"
  done < <(find "$RUN_DIR/remote/upstream-videos" -name '*.mp4' 2>/dev/null)
}

# ref_cache <name>: local cache dir for one reference set under the current key.
ref_cache() { printf '%s/%s/%s' "$LOCAL_REFS" "$REF_KEY" "$1"; }

# ref_restore <model|parity>: upload a cached reference (and the CPU stage's
# report + videos, so timings still list the CPU times) to the box. Returns 1
# when this key has no complete cached reference.
ref_restore() {
  local name="$1" dir; dir="$(ref_cache "$1")"
  [[ -f "$dir/refs/$name.safetensors" && -f "$dir/refs/$name.json" ]] || return 1
  fv_rsync_to "$HOST" "$PORT" "$dir/refs/" "$OUTR/refs/" >/dev/null
  [[ -d "$dir/out" ]] && fv_rsync_to "$HOST" "$PORT" "$dir/out/" "$OUTR/" >/dev/null
  printf '%s\n' "$name" >>"$RUN_DIR/cached-refs"
  log "$name-cpu-ref: cached CPU-path reference $REF_KEY (stage skipped)"
}

# ref_save <model|parity> <stage>: keep a freshly dumped reference for later runs.
ref_save() {
  local name="$1" stage="$2" dir tmp
  dir="$(ref_cache "$name")"
  tmp="$dir.tmp.$$"
  rm -rf "$tmp" && mkdir -p "$tmp/refs" "$tmp/out/videos"
  cp "$RUN_DIR/remote/refs/$name.safetensors" "$RUN_DIR/remote/refs/$name.json" "$tmp/refs/" || { rm -rf "$tmp"; log "could not cache $name reference"; return 0; }
  cp "$RUN_DIR/remote/$name-$stage.json" "$tmp/out/" 2>/dev/null || true
  [[ -d "$RUN_DIR/remote/videos/$name-$stage" ]] && cp -R "$RUN_DIR/remote/videos/$name-$stage" "$tmp/out/videos/"
  rm -rf "$dir" && mv "$tmp" "$dir"
  log "cached $name reference → $dir"
}

gpucheck_stage() {
  local tag="$1" timeout_s="$2"; shift 2
  remote_run "$tag" "$timeout_s" stage "$tag" "$timeout_s" --tag "$tag" "$@"
}

# create_instance <tier> <offer>: returns 1 (not fatal) so the caller can try another offer.
create_instance() {
  local tier="$1" offer="$2"
  local label out
  label="${FV_LABEL_PREFIX}-$(date -u +%Y%m%d%H%M%S)-$tier"
  if ! out="$(vastai create instance "$offer" --image "$IMAGE" --disk "${FV_DISK_GB:-$(tier_disk "$tier")}" --ssh --direct \
          --label "$label" --cancel-unavail --raw 2>&1)"; then
    log "create failed for offer $offer: $(head -c 300 <<<"$out")"
    return 1
  fi
  INSTANCE="$(jq -r '.new_contract // empty' <<<"$out" 2>/dev/null)"
  if [[ -z "$INSTANCE" ]]; then
    log "create returned no contract id for offer $offer: $(head -c 300 <<<"$out")"
    return 1
  fi
  log "created instance $INSTANCE ($label)"
}

# wait_ready: 0 once ssh works; 1 if the host never boots or never accepts ssh
# (a bad host, not a bad test) so the caller can destroy it and move on.
# FV_IMAGE_PULL_TIMEOUT (default 120s for h3-matrix via caller, else unset):
# fail fast while actual_status is still loading/creating.
wait_ready() {
  local t0; t0=$(date +%s)
  local status="" pull_cap="${FV_IMAGE_PULL_TIMEOUT:-0}"
  while :; do
    status="$(vastai show instance "$INSTANCE" --raw 2>/dev/null | jq -r '.actual_status // "unknown"')" || status=unknown
    case "$status" in
      running) break ;;
      exited|offline|error) log "instance $INSTANCE entered '$status'"; return 1 ;;
      loading|creating|created)
        if (( pull_cap > 0 )) && (( $(date +%s) - t0 >= pull_cap )); then
          log "instance $INSTANCE still '$status' after ${pull_cap}s (image pull timeout)"
          bad_host_record "$CURRENT_MACHINE" "image-pull-timeout"
          return 1
        fi
        ;;
    esac
    if (( $(date +%s) - t0 >= ${FV_BOOT_TIMEOUT:-600} )); then
      log "instance $INSTANCE not running after ${FV_BOOT_TIMEOUT:-600}s (status=$status)"
      return 1
    fi
    sleep 10
  done
  local t_run; t_run=$(date +%s)
  log "instance running after $(( t_run - t0 ))s; waiting for ssh (up to ${FV_SSH_TIMEOUT:-180}s)"
  while :; do
    if read -r HOST PORT < <(vast_ssh_target "$INSTANCE") && fv_ssh "$HOST" "$PORT" true 2>/dev/null; then
      log "ssh ok root@$HOST:$PORT"
      return 0
    fi
    if (( $(date +%s) - t_run >= ${FV_SSH_TIMEOUT:-180} )); then
      log "ssh to instance $INSTANCE never came up (${HOST:-?}:${PORT:-?})"
      HOST=""; PORT=""
      return 1
    fi
    sleep 10
  done
}

# ---- oracle dump cache -------------------------------------------------------
# A reference dump is a function of the oracle script, what it was asked for,
# the model revision and the prompt — not of our code. So it is produced once
# and kept: parity runs after that need no Python environment and no 12B-66B
# reference model on the box, only a few hundred MB rsynced up while the
# weights download. Local on purpose: no tokens on rented machines, and nothing
# derived from licensed weights is published anywhere.
#   FV_ORACLE_CACHE    where dumps live (default artifacts/oracle-cache)
#   FV_ORACLE_REFRESH  1 = ignore a cached dump and regenerate it
ORACLE_CACHE="${FV_ORACLE_CACHE:-$FV_ROOT/artifacts/oracle-cache}"
ORACLE_KEY=""
ORACLE_DIR=""
ORACLE_REV=""

# oracle_lookup <model> <repo> <prompt> <semantic oracle args...>
# Sets ORACLE_KEY / ORACLE_DIR / ORACLE_REV; returns 0 when a complete dump is cached.
oracle_lookup() {
  local model="$1" repo="$2" prompt="$3"; shift 3
  local script="$FV_ROOT/scripts/gpu/${model}_oracle.py" script_sha
  [[ -f "$script" ]] || die "no oracle script for '$model'"
  script_sha="$(shasum -a 256 "$script" | cut -c1-16)"
  # The model revision is part of the key. If the hub cannot be asked, the key
  # is unique to this run: a guess could serve a dump of other weights.
  ORACLE_REV="$(curl -fsSL --max-time 20 "https://huggingface.co/api/models/$repo" 2>/dev/null | jq -r '.sha // empty' 2>/dev/null || true)"
  [[ -n "$ORACLE_REV" ]] || ORACLE_REV="unresolved-$(date -u +%Y%m%dT%H%M%SZ)"
  ORACLE_KEY="$(printf '%s\n' "$model" "$repo" "$ORACLE_REV" "$script_sha" "$prompt" "$@" | shasum -a 256 | cut -c1-20)"
  ORACLE_DIR="$ORACLE_CACHE/$model/$ORACLE_KEY"
  [[ "${FV_ORACLE_REFRESH:-0}" != 1 ]] || return 1
  [[ -s "$ORACLE_DIR/oracle.safetensors" && -s "$ORACLE_DIR/oracle.json" && -s "$ORACLE_DIR/key.json" ]]
}

# oracle_push <model>: the cached dump to the box, where the stages expect it.
oracle_push() {
  local model="$1"
  fv_ssh "$HOST" "$PORT" "mkdir -p $OUTR/$model"
  fv_rsync_to "$HOST" "$PORT" "$ORACLE_DIR/" "$OUTR/$model/" --exclude key.json >/dev/null \
    || die "could not upload the cached oracle dump $ORACLE_KEY"
  log "oracle dump uploaded ($(du -sh "$ORACLE_DIR" | cut -f1))"
}

# oracle_store <model> <repo> <prompt> <semantic args...>: keep what the oracle
# just wrote. key.json is written last, so a partial pull is never a hit.
oracle_store() {
  local model="$1" repo="$2" prompt="$3"; shift 3
  mkdir -p "$ORACLE_DIR"
  rm -f "$ORACLE_DIR/key.json"
  if ! fv_rsync_from "$HOST" "$PORT" "$OUTR/$model/" "$ORACLE_DIR/" \
      --include 'oracle.safetensors' --include 'oracle.json' --include 'llm.safetensors' --exclude '*' >/dev/null; then
    log "could not pull the oracle dump into the cache (the run continues)"
    return 0
  fi
  [[ -s "$ORACLE_DIR/oracle.safetensors" && -s "$ORACLE_DIR/oracle.json" ]] || { log "oracle dump incomplete; not cached"; return 0; }
  jq -n --arg model "$model" --arg repo "$repo" --arg rev "$ORACLE_REV" --arg prompt "$prompt" \
    --arg script_sha "$(shasum -a 256 "$FV_ROOT/scripts/gpu/${model}_oracle.py" | cut -c1-16)" \
    --arg created "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg args "$*" \
    '{model: $model, repo: $repo, revision: $rev, script_sha: $script_sha, args: $args, prompt: $prompt, created: $created}' \
    >"$ORACLE_DIR/key.json"
  log "oracle dump cached as $model/$ORACLE_KEY ($(du -sh "$ORACLE_DIR" | cut -f1))"
}

cmd_run() {
  local tier="${1:-}"; [[ -n "$tier" ]] || usage; shift
  tier_query "$tier" >/dev/null
  if [[ "$tier" == h3-matrix ]]; then
    export FV_IMAGE_PULL_TIMEOUT="${FV_IMAGE_PULL_TIMEOUT:-120}"
    KEEP=1  # reuse until the SKU's cells finish; destroy in the matrix path
  fi
  local offer="" skip_local=0 budget_min="${FV_CLIP_BUDGET_MIN:-45}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --instance) INSTANCE="$2"; OWN_INSTANCE=0; shift 2 ;;
      --offer) offer="$2"; shift 2 ;;
      --keep) KEEP=1; shift ;;
      --skip-local) skip_local=1; shift ;;
      --clip-budget-min) budget_min="$2"; shift 2 ;;
      *) die "unknown option $1" ;;
    esac
  done
  require_tools vastai jq rsync ssh
  [[ -f "$FV_SSH_KEY" ]] || die "ssh key $FV_SSH_KEY missing (VAST_SSH_KEY)"
  vast_check_auth
  if [[ $skip_local -eq 0 ]]; then
    log "free local preflight first (skip with --skip-local)"
    cmd_local
  else
    "$DOCKER_SH" dist || die "dist build failed" 1
  fi
  local build_id
  build_id="$("$DOCKER_SH" build-id)"
  [[ "$(cat "$DIST/fv-gpucheck.build-id" 2>/dev/null)" == "$build_id" ]] || die "dist binary is stale; run: bash scripts/gpu/dist-ci.sh (CI-built, no GPU) — or $DOCKER_SH dist to build locally"
  resolve_image "$build_id" "$tier"

  local max_dph mins
  max_dph="${MAX_DPH:-$(tier_max_dph "$tier")}"
  mins="${MAX_MINUTES:-$(tier_max_minutes "$tier")}"
  RUN_DIR="$RUNS/$(date -u +%Y%m%dT%H%M%SZ)-$tier"
  mkdir -p "$RUN_DIR/remote"
  T_START=$(date +%s)
  trap cleanup EXIT
  trap 'cleanup 130' INT
  trap 'cleanup 143' TERM

  if [[ $OWN_INSTANCE -eq 0 ]]; then
    DPH="$(vastai show instance "$INSTANCE" --raw | jq -r '.dph_total // 0')"
    wait_ready || die "instance $INSTANCE is not reachable"
  else
    # Bad hosts (no ssh, stuck boot, offer gone) are destroyed and skipped:
    # up to FV_OFFER_RETRIES more offers, never the same machine twice.
    local tried attempt=0 max_attempts=$(( ${FV_OFFER_RETRIES:-2} + 1 )) checked_balance=0
    tried="$(bad_hosts_json)"
    [[ "$tried" == "[]" ]] || log "skipping recently bad machines: $tried"
    while :; do
      attempt=$((attempt + 1))
      (( attempt <= max_attempts )) || die "no usable host after $max_attempts offers"
      local pick
      pick="$(offers_json "$tier" | jq -c --arg id "$offer" --argjson max "$max_dph" --argjson tried "$tried" \
        'map(select((.machine_id as $m | $tried | index($m) | not)
                    and (if $id == "" then .dph_total <= $max else (.id|tostring) == $id end))) | .[0] // empty')"
      [[ -n "$pick" ]] || die "no untried offer ≤ \$$max_dph/hr for tier $tier (raise MAX_DPH or see '$0 offers $tier')"
      offer=""  # an explicit --offer applies to the first attempt only
      local offer_id; offer_id="$(jq -r .id <<<"$pick")"
      tried="$(jq -c --argjson m "$(jq .machine_id <<<"$pick")" '. + [$m]' <<<"$tried")"
      DPH="$(jq -r .dph_total <<<"$pick")"
      local cap
      cap="$(awk -v d="$DPH" -v m="$mins" 'BEGIN { printf "%.2f", d * m / 60 }')"
      log "attempt $attempt/$max_attempts offer $offer_id: $(jq -r '"\(.gpu_name) \(.gpu_ram)GB $\(.dph_total)/hr (machine \(.machine_id))"' <<<"$pick"); worst case \$$cap (${mins} min cap)"
      if [[ $checked_balance -eq 0 ]]; then
        local balance; balance="$(vast_balance)"
        log "balance \$${balance:-?}"
        if [[ -n "$balance" ]] && awk -v b="$balance" -v c="$cap" 'BEGIN { exit !(b < c) }'; then
          die "balance \$$balance below worst-case \$$cap"
        fi
        checked_balance=1
      fi
      CURRENT_MACHINE="$(jq -r .machine_id <<<"$pick")"
      create_instance "$tier" "$offer_id" || continue
      # Independent of this shell: destroys at the wall cap even if we're killed.
      nohup bash -c "sleep $((mins * 60)); vastai destroy instance -y $INSTANCE" >/dev/null 2>&1 &
      WATCHDOG_PID=$!
      disown "$WATCHDOG_PID"
      if wait_ready; then
        break
      fi
      log "host unusable — destroying $INSTANCE and trying another offer"
      bad_host_record "$CURRENT_MACHINE" "boot/ssh"
      kill "$WATCHDOG_PID" 2>/dev/null || true
      WATCHDOG_PID=""
      vast_destroy "$INSTANCE" || true
      INSTANCE=""
    done
  fi

  # No source, no compile on the box. Scripts always sync (they are not part of
  # the build id); the binary only when the image doesn't already carry this build.
  LAST_STAGE=upload
  fv_ssh "$HOST" "$PORT" "mkdir -p $FV_REMOTE_DIR/scripts $FV_REMOTE_DIR/target/release $OUTR/refs"
  fv_rsync_to "$HOST" "$PORT" "$FV_ROOT/scripts/" "$FV_REMOTE_DIR/scripts/" --delete --exclude '.env*'
  # Gated Hub repos (LTX-2.5): seed the HF token file for hf-fm without putting
  # the secret on the ssh argv. validate.sh never ships .env*; this is the opt-in path.
  local hf_tok_file="${HF_HOME:-$HOME/.cache/huggingface}/token"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    mkdir -p "$RUN_DIR/.hf"
    printf '%s\n' "$HF_TOKEN" >"$RUN_DIR/.hf/token"
    chmod 600 "$RUN_DIR/.hf/token"
    hf_tok_file="$RUN_DIR/.hf/token"
  fi
  if [[ -f "$hf_tok_file" ]]; then
    fv_ssh "$HOST" "$PORT" "mkdir -p /root/.cache/huggingface $WORK/hf"
    fv_rsync_to "$HOST" "$PORT" "$hf_tok_file" "/root/.cache/huggingface/token"
    fv_ssh "$HOST" "$PORT" "chmod 600 /root/.cache/huggingface/token; cp -f /root/.cache/huggingface/token $WORK/hf/token; chmod 600 $WORK/hf/token"
    log "seeded HF token on box for gated Hub fetches"
  fi
  # Ask the box which build it holds rather than inferring it from the image
  # we would pick today: a reused instance (--instance) was created from an
  # older image, and trusting "the image has the binary" ran a stale
  # fv-gpucheck there (it lacked the stage being asked for).
  local remote_bid
  remote_bid="$(fv_ssh "$HOST" "$PORT" "cat $FV_REMOTE_DIR/target/release/fv-gpucheck.build-id 2>/dev/null" 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ "$remote_bid" == "$build_id" ]]; then
    log "box already holds fv-gpucheck build $build_id"
  else
    log "uploading prebuilt fv-gpucheck (build $build_id; box has '${remote_bid:-none}')"
    fv_rsync_to "$HOST" "$PORT" "$DIST/" "$FV_REMOTE_DIR/target/release/"
  fi
  # Slim runtime path: bake hf-fm into PATH even when the GHCR image is still
  # the previous Python-based tag (binary upload is the matrix failsafe).
  if [[ -x "$DIST/hf-fm" ]]; then
    fv_rsync_to "$HOST" "$PORT" "$DIST/hf-fm" "$FV_REMOTE_DIR/target/release/hf-fm"
    [[ -x "$DIST/hf-fetch-model" ]] && fv_rsync_to "$HOST" "$PORT" "$DIST/hf-fetch-model" "$FV_REMOTE_DIR/target/release/hf-fetch-model"
    fv_ssh "$HOST" "$PORT" "install -m 755 $FV_REMOTE_DIR/target/release/hf-fm /usr/local/bin/hf-fm
      [[ -x $FV_REMOTE_DIR/target/release/hf-fetch-model ]] && install -m 755 $FV_REMOTE_DIR/target/release/hf-fetch-model /usr/local/bin/hf-fetch-model
      command -v hf-fm"
  elif ! fv_ssh "$HOST" "$PORT" "command -v hf-fm >/dev/null"; then
    die "hf-fm missing on the box and not in $DIST — rebuild with build-remote or push a runtime image that bakes hf-fetch-model"
  fi
  # CPU-path references are cached per ref key (scripts/gpu/lib.sh fv_ref_key):
  # a hit uploads them and skips that CPU-reference stage entirely.
  REF_KEY="$(fv_ref_key)"
  fv_ssh "$HOST" "$PORT" "echo $REF_KEY >$OUTR/refs/ref-key"
  local have_model_ref=0 have_parity_ref=0
  ref_restore model && have_model_ref=1
  ref_restore parity && have_parity_ref=1

  # A fresh box must have room for the tier's downloads. A reused one
  # (--instance) already holds them, and the tier's figure is a rental filter,
  # not a measurement: ask only for room to write outputs.
  local need_disk; need_disk=$(( $(tier_disk "$tier") - 10 ))
  if [[ $OWN_INSTANCE -eq 0 ]]; then need_disk="${FV_NEED_DISK:-20}"; fi
  remote_run env 120 env "$need_disk"
  remote_run bootstrap 600 bootstrap
  # Downloads run in the background during the weight-free stages.
  case "$tier" in
    parity) remote_run fetch-base 120 fetch "$BASE_REPO" "$BASE_W" "transformer/*" "vae/*" ;;
    clip | compare | gen | oracle | fp8 | vaeab)
      remote_run fetch-base 120 fetch "$BASE_REPO" "$BASE_W" "transformer/*" "vae/*" "text_encoder/*" "tokenizer/*"
      remote_run fetch-fast 120 fetch "$FAST_REPO" "$FAST_W" "transformer/*" "vae/*"
      ;;
  esac

  local refs="$OUTR/refs"
  # ---- audio-video parity tiers -------------------------------------------
  # One body for every tier that judges a port against the diffusers /
  # transformers reference. The reference dump is a cached artifact (see
  # oracle_lookup): on a hit the box never builds the Python environment, never
  # loads the 12B-66B reference model, and fetches only the weights OUR side
  # reads. A miss runs the oracle once and stores the dump for next time.
  if [[ "$tier" == h3-text || "$tier" == ltx2-text || "$tier" == h3-vae || "$tier" == ltx2-vae \
     || "$tier" == h3-dit || "$tier" == ltx2-dit ]]; then
    local model repo sem=() pat_ours=() pat_oracle=() comp_ours=() comp_oracle=()
    local h3_repo="${FV_H3_REPO:-FastVideo/FastVideo-FastH3-8-Step-V2}"
    # The distilled model in diffusers layout: Lightricks/LTX-2's own
    # transformer/ and connectors/ are the *dev* model. Everything that is not
    # the DiT or the connectors is byte-identical between the two repos.
    local ltx_repo="${FV_LTX2_REPO:-rootonchair/LTX-2-19b-distilled}"
    # Qwen3-VL shards 12-14 hold layers past the tap, the final norm, the LM
    # head and the vision tower: our side never reads them, transformers does.
    local h3_text_ours=("tokenizer/*" "text_encoder/*.json"
      "text_encoder/model-0000[1-9]-of-00014.safetensors" "text_encoder/model-0001[01]-of-00014.safetensors")
    # h3_oracle.py loads tokenizer/ and both scheduler configs whatever --stages says.
    local h3_always=("tokenizer/*" "scheduler/*" "audio_scheduler/*")
    case "$tier" in
      h3-text)
        model=h3 repo="$h3_repo" sem=(--stages text)
        pat_ours=("${h3_text_ours[@]}") comp_ours=(text_encoder)
        pat_oracle=("text_encoder/*" "processor/*" "${h3_always[@]}") comp_oracle=(text_encoder) ;;
      ltx2-text)
        # `model-*` is the live text-encoder shard set; `diffusion_pytorch_model-*`
        # beside it is a stale 52 GB duplicate (docs/ports/ltx2.md).
        model=ltx2 repo="$ltx_repo" sem=(--skip dit,vae,audio)
        pat_ours=("text_encoder/model-*" "text_encoder/*.json" "tokenizer/*" "connectors/*")
        comp_ours=(text_encoder connectors) ;;
      h3-vae)
        model=h3 repo="$h3_repo" sem=(--stages vae,audio)
        pat_ours=("vae/*" "audio_vae/*") comp_ours=(vae audio_vae) pat_oracle=("${h3_always[@]}") ;;
      ltx2-vae)
        model=ltx2 repo="$ltx_repo" sem=(--skip text,conn,dit)
        pat_ours=("vae/*" "audio_vae/*" "vocoder/*") comp_ours=(vae audio_vae vocoder) ;;
      h3-dit)
        model=h3 repo="$h3_repo" sem=(--stages dit,loop)
        pat_ours=("transformer/*") comp_ours=(transformer) pat_oracle=("${h3_always[@]}") ;;
      ltx2-dit)
        # `both`: a bf16 pass and a float32 pass of the same module, plus the
        # reference's own bf16-vs-f32 distance per tap — the floor ours is judged by.
        model=ltx2 repo="$ltx_repo" sem=(--skip vae,audio --sample --dit-dtype "${FV_LTX2_DIT_DTYPE:-both}")
        pat_ours=("vae/*" "audio_vae/*" "vocoder/*") comp_ours=(vae audio_vae vocoder)
        pat_oracle=("transformer/*" "connectors/*" "text_encoder/model-*" "text_encoder/*.json" "tokenizer/*")
        comp_oracle=(transformer connectors text_encoder) ;;
    esac
    local wdir="$WORK/weights/$model" odir="$OUTR/$model"
    local prompt="${FV_PROMPT:-$(jq -r '.prompts[0].prompt' "$FV_ROOT/scripts/gpu/prompts.json")}"
    jq -n --arg p "$prompt" '{negative: "", prompts: [{name: "oracle", prompt: $p}]}' >"$RUN_DIR/prompt.json"
    fv_rsync_to "$HOST" "$PORT" "$RUN_DIR/prompt.json" "$OUTR/prompt.json" >/dev/null

    local hit=0
    if oracle_lookup "$model" "$repo" "$prompt" "${sem[@]}"; then hit=1; fi
    if [[ $hit -eq 1 ]]; then
      log "oracle dump $ORACLE_KEY cached: no Python, no reference model, our weights only"
      remote_run "fetch-$model" 120 fetch "$repo" "$wdir" "${pat_ours[@]}"
      oracle_push "$model"
    else
      log "oracle dump $ORACLE_KEY not cached: running the reference once"
      remote_run "fetch-$model" 120 fetch "$repo" "$wdir" "${pat_ours[@]}" ${pat_oracle[@]+"${pat_oracle[@]}"}
    fi
    # LTX-2's production DiT is the official single file, read through the
    # key-rename view; it downloads beside everything else.
    local sdir="$WORK/weights/ltx2-single"
    if [[ "$tier" == ltx2-dit ]]; then
      remote_run fetch-ltx2-single 120 fetch "${FV_LTX2_SINGLE_REPO:-Lightricks/LTX-2}" "$sdir" "ltx-2-19b-distilled.safetensors"
    fi
    if [[ $hit -eq 0 ]]; then
      remote_run oracle-venv 1800 oracle-venv "${FV_TORCH_BACKEND:-cu130}"
      remote_run "wait-$model" 7200 wait-weights "$wdir" 7200 "${comp_ours[@]}" ${comp_oracle[@]+"${comp_oracle[@]}"}
      local oargs=(--weights "$wdir" --prompts "$OUTR/prompt.json" --out "$odir/oracle.safetensors" --meta "$odir/oracle.json")
      if [[ "$tier" == *-text ]]; then oargs+=(--llm-out "$odir/llm.safetensors"); fi
      remote_run "oracle-$model" 7200 model-oracle "$model" "${oargs[@]}" "${sem[@]}"
      oracle_store "$model" "$repo" "$prompt" "${sem[@]}"
    else
      remote_run "wait-$model" 7200 wait-weights "$wdir" 7200 "${comp_ours[@]}"
    fi

    local orc="$odir/oracle.safetensors"
    case "$tier" in
      h3-text | ltx2-text)
        local family=qwen3-vl-32b; [[ "$model" == ltx2 ]] && family=gemma3-12b
        STAGE_OPTIONAL=1 gpucheck_stage "llm-$model" 3600 --keep-going --mode fast llm --weights "$wdir/text_encoder" \
          --family "$family" --oracle "$odir/llm.safetensors" --device cuda || true
        if [[ "$model" == h3 ]]; then
          gpucheck_stage h3-text 3600 --keep-going --mode fast h3 text --weights "$wdir" --oracle "$orc" --meta "$odir/oracle.json"
          # The same gates through a resident weight-only-FP8 encoder, plus FP8
          # against our own native encoder (quantization error on its own).
          STAGE_OPTIONAL=1 gpucheck_stage h3-text-fp8 3600 --keep-going --mode fast h3 text --weights "$wdir" \
            --oracle "$orc" --meta "$odir/oracle.json" --precision fp8 || true
        else
          # Exact mode: Gemma streams a layer at a time and the connectors are
          # 5.8 GB in f32, so nothing here needs bf16 to fit.
          gpucheck_stage ltx2-text 5400 --keep-going --mode exact ltx2 text --weights "$wdir" --dit "$wdir" \
            --oracle "$orc" --meta "$odir/oracle.json"
        fi ;;
      h3-vae)
        STAGE_OPTIONAL=1 gpucheck_stage h3-audio-vae 1800 --keep-going --mode exact h3 audio-vae --weights "$wdir" --oracle "$orc" || true
        gpucheck_stage h3-vae 3600 --keep-going --mode exact h3 vae --weights "$wdir" --oracle "$orc" ;;
      ltx2-vae)
        STAGE_OPTIONAL=1 gpucheck_stage ltx2-audio 1800 --keep-going --mode exact ltx2 audio --weights "$wdir" \
          --oracle "$orc" --wav "$OUTR/ltx2/audio.wav" || true
        gpucheck_stage ltx2-vae 3600 --keep-going --mode exact ltx2 vae --weights "$wdir" --oracle "$orc" ;;
      h3-dit)
        STAGE_OPTIONAL=1 gpucheck_stage h3-dit 7200 --keep-going --mode fast h3 dit --weights "$wdir" --oracle "$orc" || true
        # The dense 8-rung ladder on the oracle's text and noise, on the default
        # (chunked cuBLAS) SDPA: 87 s per 38k-token forward on an RTX PRO 6000. The
        # tiled NVRTC "flash" kernel is ~20x SLOWER at this length (measured: 25
        # blocks in 15 minutes) — it exists for memory, not speed.
        gpucheck_stage h3-loop 3600 --keep-going --mode fast h3 loop --weights "$wdir" --oracle "$orc" ;;
      ltx2-dit)
        # Tables first: no weights, seconds, and a layout bug is named by table.
        STAGE_OPTIONAL=1 gpucheck_stage ltx2-rope 600 --keep-going --mode fast ltx2 dit --rope-only \
          --dit "$sdir/ltx-2-19b-distilled.safetensors" --oracle "$orc" || true
        # The converted transformer/ is only on the box when the reference ran
        # there; it was proved identical to the single file, to the last digit.
        if [[ $hit -eq 0 && "${FV_LTX2_CONVERTED:-0}" == 1 ]]; then
          STAGE_OPTIONAL=1 gpucheck_stage ltx2-dit-converted 5400 --keep-going --mode fast ltx2 dit \
            --dit "$wdir" --oracle "$orc" || true
        fi
        remote_run wait-ltx2-single 7200 wait-weights "$sdir" 7200
        STAGE_OPTIONAL=1 gpucheck_stage ltx2-dit-single 5400 --keep-going --mode fast ltx2 dit \
          --dit "$sdir/ltx-2-19b-distilled.safetensors" --oracle "$orc" || true
        # The 8-step trajectory on the oracle's noise, then both decoders at
        # production size on the oracle's and on our own final latents.
        gpucheck_stage ltx2-loop 7200 --keep-going --mode fast ltx2 loop \
          --dit "$sdir/ltx-2-19b-distilled.safetensors" --oracle "$orc" --weights "$wdir" ;;
    esac
    log "$tier done"
    return 0
  fi
  # VSA-H3 against its host statement: a kernel-reuse check, no weights.
  if [[ "$tier" == h3-vsa ]]; then
    gpucheck_stage h3-vsa 1800 --keep-going --mode fast h3 vsa
    log "h3-vsa done"
    return 0
  fi
  # Checkpoint × encoder × residency matrix (TAEH3, --warm, no oracle).
  if [[ "$tier" == h3-matrix ]]; then
    run_h3_matrix
    return 0
  fi
  # Generation, end to end, by our binary alone: prompt in, mp4 with an audio
  # track out. FV_PROMPT, FV_SECONDS (H3) and FV_SEED choose the clip.
  if [[ "$tier" == h3-gen || "$tier" == ltx2-gen ]]; then
    local prompt="${FV_PROMPT:-A man in his thirties talking to the camera in a bright living room, medium close-up, natural expressions and hand gestures, soft window light. He says: <d>Hello, this was generated entirely in Rust.</d>}"
    local name="${FV_CLIP_NAME:-$tier-$(date -u +%Y%m%d%H%M%S)}" clip="$OUTR/clips"
    if [[ "$tier" == h3-gen ]]; then
      # FV_H3_RECIPE=sol-h3 loads base MiniMax-H3 and fuses the 4-step LoRA.
      # The FastH3-8-Step checkpoint is already merged and is the wrong pack.
      local recipe="${FV_H3_RECIPE:-}"
      local repo="${FV_H3_REPO:-}"
      if [[ -z "$repo" ]]; then
        if [[ "$recipe" == sol-h3* ]]; then
          repo="MiniMaxAI/MiniMax-H3"
        else
          repo="FastVideo/FastVideo-FastH3-8-Step-V2"
        fi
      fi
      local wdir="$WORK/weights/h3"
      # Qwen3-VL shards 12-14 hold layers past the tap, the final norm, the LM
      # head and the vision tower: never read, so never fetched.
      # Bash 3.2 + set -u treats "${empty[@]}" as unbound; do not empty the
      # arrays and expand them — omit the official VAE glob instead.
      local fetch_h3=(fetch "$repo" "$wdir" "tokenizer/*" "text_encoder/*.json"
        "text_encoder/model-0000[1-9]-of-00014.safetensors" "text_encoder/model-0001[01]-of-00014.safetensors"
        "transformer/*" "audio_vae/*")
      local wait_h3=(wait-weights "$wdir" 7200 text_encoder transformer audio_vae)
      if [[ "${FV_TAEH3:-0}" != 1 ]]; then
        fetch_h3+=("vae/*")
        wait_h3+=(vae)
      fi
      remote_run fetch-h3 120 "${fetch_h3[@]}"
      local adapter_dir=""
      if [[ "$recipe" == sol-h3* ]]; then
        # Resolver finds this next to the snapshot: FastH3-4-step-Preview-v1-LoRA/dense-datafree/.
        adapter_dir="$wdir/FastH3-4-step-Preview-v1-LoRA"
        remote_run fetch-sol-adapter 120 fetch "FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA" "$adapter_dir" \
          "dense-datafree/adapter_model.safetensors"
      fi
      local taeh3_dir="$WORK/taeh3"
      if [[ "${FV_TAEH3:-0}" == 1 ]]; then remote_run fetch-taeh3 300 fetch-taeh3 "$taeh3_dir"; fi
      remote_run wait-h3 7200 "${wait_h3[@]}"
      if [[ -n "$adapter_dir" ]]; then
        remote_run wait-sol-adapter 1800 wait-weights "$adapter_dir" 1800
      fi
      local h3gen=(--mode fast h3 gen --weights "$wdir" --prompt "$prompt" --seconds "${FV_SECONDS:-5}"
        --seed "${FV_SEED:-1024}" --adaln-cache "$WORK/h3-adaln.cache")
      if [[ -n "$recipe" ]]; then h3gen+=(--h3-recipe "$recipe"); fi
      if [[ -n "${FV_H3_TEXT_ENCODER:-}" ]]; then h3gen+=(--text-encoder "$FV_H3_TEXT_ENCODER"); fi
      # --profile is a global flag: each DiT/VSA phase synchronizes, so this
      # run answers *where* the 94 s denoise goes, not *how long* a served
      # request takes. FV_WARM=1 times the resident pipeline (the 94 s number).
      if [[ "${FV_PROFILE:-0}" == 1 ]]; then h3gen=(--profile "${h3gen[@]}"); fi
      if [[ "${FV_WARM:-0}" == 1 ]]; then h3gen+=(--warm); fi
      if [[ "${FV_TAEH3:-0}" == 1 ]]; then
        remote_run wait-taeh3 300 wait-taeh3 "$taeh3_dir" 300
        h3gen+=(--taeh3-weights "$taeh3_dir")
      fi
      if [[ "${FV_TEXT_PLAN:-0}" == 1 ]]; then
        # What text conditioning costs by each route, same prompt and seed:
        # streamed (layers prefetched through pinned memory), resident FP8 in a
        # warm process with its drift against the streamed conditioning, and a
        # conditioning-cache hit.
        local tcache="$WORK/h3-text-cache"
        gpucheck_stage "gen-$name-streamed" 7200 "${h3gen[@]}" --clip-dir "$clip/$name-streamed/frames" \
          --text-encoder streamed --no-text-cache
        STAGE_OPTIONAL=1 gpucheck_stage "gen-$name-resident" 7200 "${h3gen[@]}" --clip-dir "$clip/$name-resident/frames" \
          --text-encoder auto --compare-text-encoders --warm --text-cache "$tcache" || true
        STAGE_OPTIONAL=1 gpucheck_stage "gen-$name-cached" 7200 "${h3gen[@]}" --clip-dir "$clip/$name-cached/frames" \
          --text-encoder auto --text-cache "$tcache" || true
      else
        if [[ -n "${FV_H3_AFFINE:-}" && "${FV_H3_AFFINE}" != "0" && "${FV_H3_AFFINE}" != "off" ]]; then
          # Same prompt/seed/cache, two gens: bf16 DiT vs MLX affine INT8/6/4.
          local bits="${FV_H3_AFFINE}"
          case "$bits" in 1|on|true|yes) bits=int8 ;; esac
          gpucheck_stage "gen-$name-bf16" 7200 "${h3gen[@]}" --clip-dir "$clip/$name-bf16/frames"
          gpucheck_stage "gen-$name-affine" 7200 --h3-affine "$bits" "${h3gen[@]}" --clip-dir "$clip/$name-affine/frames"
          local a="$clip/$name-bf16/frames/output.mp4" b="$clip/$name-affine/frames/output.mp4"
          log "affine A/B PSNR ($bits) $a vs $b"
          fv_ssh "$HOST" "$PORT" "ffmpeg -hide_banner -i '$a' -i '$b' -lavfi '[0:v][1:v]psnr' -f null - 2>&1 | tail -8" || true
        elif [[ "${FV_H3_FFN_FP8:-0}" == 1 ]]; then
          # Same prompt/seed/cache, two gens: bf16 FFN vs H3-only E4M3 FFN.
          gpucheck_stage "gen-$name-bf16" 7200 "${h3gen[@]}" --clip-dir "$clip/$name-bf16/frames"
          gpucheck_stage "gen-$name-fp8" 7200 --h3-ffn-fp8 "${h3gen[@]}" --clip-dir "$clip/$name-fp8/frames"
        else
          gpucheck_stage "gen-$name" 7200 "${h3gen[@]}" --clip-dir "$clip/$name/frames"
        fi
      fi
    else
      # FV_LTX2_VERSION=2.5 → Diffusers LTX-2.5 pack + ancestral gen (docs/ports/ltx25.md).
      # FV_LTX2_TWO_STAGE=1 → distilled two-stage (half-res → upsampler → stage-2).
      local ltx_ver="${FV_LTX2_VERSION:-2.0}"
      local repo wdir
      local ltxgen=(--mode fast ltx2 gen --prompt "$prompt" --seed "${FV_SEED:-10}")
      if [[ "$ltx_ver" == "2.5" ]]; then
        repo="${FV_LTX2_BASE_REPO:-Lightricks/LTX-2.5-Diffusers}"
        wdir="$WORK/weights/ltx2-5"
        remote_run fetch-ltx2-5 120 fetch "$repo" "$wdir" \
          "tokenizer/*" "text_encoder/model-*" "text_encoder/*.json" \
          "connectors/*" "transformer/*" "vae/*" "audio_vae/*" "vocoder/*" "latent_upsampler/*" "diffusion_decoder/*"
        ltxgen+=(--model-version 2.5 --weights "$wdir" --dit "$wdir")
        if [[ "${FV_LTX2_TWO_STAGE:-0}" == 1 ]]; then
          ltxgen+=(--two-stage)
        fi
        if [[ "${FV_LTX2_DIFF_VAE:-0}" == 1 ]]; then
          ltxgen+=(--diff-vae)
        fi
        if [[ -n "${FV_IMAGE:-}" ]]; then
          local img_remote="$WORK/i2v-first.png"
          if [[ "$FV_IMAGE" == /* && ! -f "$FV_IMAGE" ]]; then
            img_remote="$FV_IMAGE"
          else
            [[ -f "$FV_IMAGE" ]] || die "FV_IMAGE is not a file: $FV_IMAGE"
            fv_rsync_to "$HOST" "$PORT" "$FV_IMAGE" "$img_remote" >/dev/null
          fi
          ltxgen+=(--image "$img_remote")
          log "I2V first-frame encode from $img_remote"
        fi
        remote_run wait-ltx2-5 10800 wait-weights "$wdir" 10800 text_encoder connectors transformer vae audio_vae vocoder latent_upsampler diffusion_decoder
        gpucheck_stage "gen-$name" 10800 "${ltxgen[@]}" --clip "$clip/$name/frames"
      else
        repo="${FV_LTX2_BASE_REPO:-Lightricks/LTX-2}"
        wdir="$WORK/weights/ltx2"
        remote_run fetch-ltx2 120 fetch "$repo" "$wdir" "tokenizer/*" "text_encoder/model-*" "text_encoder/*.json" \
          "vae/*" "audio_vae/*" "vocoder/*" "ltx-2-19b-distilled.safetensors"
        ltxgen+=(--weights "$wdir" --dit "$wdir/ltx-2-19b-distilled.safetensors")
        if [[ "${FV_TEXT_PLAN:-0}" == 1 ]]; then
          # The slim checkpoint is CPU work: it runs as soon as the text encoder has
          # landed, while the 43 GB DiT file is still downloading.
          local slim="$WORK/ltx2-slim" tcache="$WORK/ltx2-text-cache"
          remote_run wait-ltx2-text 7200 wait-weights "$wdir" 7200 text_encoder
          gpucheck_stage ltx2-slim-text 3600 --mode fast ltx2 slim-text --weights "$wdir" --slim "$slim"
          remote_run wait-ltx2 7200 wait-weights "$wdir" 7200 text_encoder vae audio_vae vocoder
          gpucheck_stage "gen-$name-streamed" 7200 "${ltxgen[@]}" --clip "$clip/$name-streamed/frames" \
            --text streamed --no-text-cache
          STAGE_OPTIONAL=1 gpucheck_stage "gen-$name-streamed-slim" 7200 "${ltxgen[@]}" --clip "$clip/$name-streamed-slim/frames" \
            --text streamed --no-text-cache --text-weights "$slim" || true
          STAGE_OPTIONAL=1 gpucheck_stage "gen-$name-resident" 7200 "${ltxgen[@]}" --clip "$clip/$name-resident/frames" \
            --text resident --text-weights "$slim" --warm --text-cache "$tcache" || true
          STAGE_OPTIONAL=1 gpucheck_stage "gen-$name-cached" 7200 "${ltxgen[@]}" --clip "$clip/$name-cached/frames" \
            --text resident --text-weights "$slim" --text-cache "$tcache" || true
        else
          remote_run wait-ltx2 7200 wait-weights "$wdir" 7200 text_encoder vae audio_vae vocoder
          gpucheck_stage "gen-$name" 7200 "${ltxgen[@]}" --clip "$clip/$name/frames"
        fi
      fi
    fi
    log "GEN done: $name"
    return 0
  fi

  if [[ "$tier" == mathprobe ]]; then
    # Which cuBLAS math runs here: the image's cuBLAS, then a newer one.
    gpucheck_stage device 300 device
    gpucheck_stage gemm-probe-image 900 gemm-probe
    remote_run cublas-new 600 cublas "${FV_PROBE_CUBLAS_VERSION:-12.9.1.4}"
    gpucheck_stage gemm-probe-new 900 gemm-probe
    log "math probe done"
    return 0
  fi
  # `gen` is the UI path: no validation, just enough to turn a prompt into a
  # clip. FV_PROMPT carries the text; the negative prompt comes from the shared
  # prompts file so generations match the benchmark configuration.
  if [[ "$tier" == gen ]]; then
    local prompt="${FV_PROMPT:-}"
    [[ -n "$prompt" ]] || die "gen needs FV_PROMPT"
    local name="${FV_CLIP_NAME:-ui-$(date -u +%Y%m%d%H%M%S)}"
    local negative; negative="$(jq -r '.negative' "$FV_ROOT/scripts/gpu/prompts.json")"
    jq -n --arg p "$prompt" --arg n "$negative" \
      '{negative: $n, prompts: [{name: "ui", prompt: $p}]}' >"$RUN_DIR/prompt.json"
    fv_rsync_to "$HOST" "$PORT" "$RUN_DIR/prompt.json" "$OUTR/prompt.json" >/dev/null
    local embeds="$OUTR/embeds"
    # TAEHV decodes the clip unless FV_TAEHV=0: ~10x faster than the Wan VAE
    # and 34 dB against it. The weights ship separately (madebyollin/taehv),
    # so fetch them behind the text encoding.
    local taehv_dir="$WORK/taehv"
    if [[ "${FV_TAEHV:-1}" == 1 ]]; then remote_run fetch-taehv 300 fetch-taehv "$taehv_dir"; fi
    remote_run wait-text 1800 wait-weights "$BASE_W" 1800 text_encoder
    gpucheck_stage embed 1800 --mode exact embed --weights "$BASE_W" --prompts "$OUTR/prompt.json" \
      --embeds "$embeds" --device cuda
    remote_run wait-fast 1800 wait-weights "$FAST_W" 1800 transformer vae
    local gen=(--height "${FV_HEIGHT:-448}" --width "${FV_WIDTH:-832}" --frames "${FV_FRAMES:-129}"
               --steps "${FV_STEPS:-3}" --guidance 1.0 --flow-shift 8.0 --dmd --fps 16)
    if [[ "${FV_VSA:-1}" == 1 ]]; then gen+=(--vsa); fi
    if [[ -n "${FV_VAE_CHUNK:-2}" ]]; then gen+=(--vae-chunk "${FV_VAE_CHUNK:-2}"); fi
    if [[ "${FV_TAEHV:-1}" == 1 ]]; then
      remote_run wait-taehv 300 wait-taehv "$taehv_dir" 300
      gen+=(--taehv-weights "$taehv_dir")
    fi
    # FV_WARM=1: one untimed generation first, so the clip's timings are a
    # resident pipeline's rather than the first clip after load.
    if [[ "${FV_WARM:-0}" == 1 ]]; then gen+=(--warm); fi
    gpucheck_stage "clip-$name" $(( budget_min * 60 * 13 / 10 + 600 )) --mode fast clip \
      --weights "$FAST_W" --embeds "$embeds/ui.safetensors" --device cuda "${gen[@]}" \
      --name "$name" --budget-min "$budget_min"
    log "GEN done: $name"
    return 0
  fi

  # `oracle` is the only tier that judges us against something other than
  # ourselves. transformers' UMT5 and diffusers' WanTransformer3DModel run on
  # the same weights, and both sides then consume byte-identical inputs, so the
  # three checks attribute error instead of only detecting it:
  #   text — our embedding vs the reference embedding   → the UMT5 port
  #   dit  — our DiT on the *reference* embedding       → the DiT port
  #   e2e  — our DiT on *our* embedding                 → what a clip gets
  if [[ "$tier" == oracle ]]; then
    local prompt="${FV_PROMPT:-$(jq -r '.prompts[0].prompt' "$FV_ROOT/scripts/gpu/prompts.json")}"
    local negative; negative="$(jq -r '.negative' "$FV_ROOT/scripts/gpu/prompts.json")"
    jq -n --arg p "$prompt" --arg n "$negative" \
      '{negative: $n, prompts: [{name: "oracle", prompt: $p}]}' >"$RUN_DIR/prompt.json"
    fv_rsync_to "$HOST" "$PORT" "$RUN_DIR/prompt.json" "$OUTR/prompt.json" >/dev/null
    log "oracle prompt: $prompt"
    local embeds="$OUTR/embeds"
    remote_run wait-text 1800 wait-weights "$BASE_W" 1800 text_encoder
    gpucheck_stage embed 1800 --mode exact embed --weights "$BASE_W" --prompts "$OUTR/prompt.json" \
      --embeds "$embeds" --device cuda
    remote_run wait-fast 1800 wait-weights "$FAST_W" 1800 transformer vae
    remote_run upstream-install "${FV_UPSTREAM_INSTALL_TIMEOUT:-2400}" upstream-install "${FV_TORCH_BACKEND:-cu126}"
    remote_run upstream-oracle "${FV_ORACLE_TIMEOUT:-3600}" upstream-oracle \
      --base-weights "$BASE_W" --dit-weights "$FAST_W" --prompts "$OUTR/prompt.json" \
      --height "${FV_HEIGHT:-448}" --width "${FV_WIDTH:-832}" --num-frames "${FV_ORACLE_FRAMES:-33}"
    # Exact mode: the oracle is float32, and a looser judge cannot set a limit.
    gpucheck_stage oracle 1800 --keep-going --mode exact oracle --weights "$FAST_W" \
      --oracle "$OUTR/oracle/oracle.safetensors" --embeds "$embeds/oracle.safetensors" --device cuda
    log "ORACLE done"
    return 0
  fi

  # `fp8` answers two questions about the FP8 linear path that only hardware
  # can: is it faster, and does per-tensor E4M3 hold on a checkpoint distilled
  # for it. One box, one prompt, one seed, the same clip twice — so the only
  # variable is --fp8 — then `compare` diffs the two latents and frames.
  #
  # Point FV_FAST_REPO at FastVideo/FastWan-QAD-FP8-1.3B for the QAD weights.
  # Running it against stock FastWan is also informative: the difference between
  # the two is the whole claim QAD makes.
  if [[ "$tier" == fp8 ]]; then
    gpucheck_stage nvrtc 300 nvrtc
    gpucheck_stage device 300 device
    # Kernel-level FP8 checks first: no point timing a GEMM that is wrong.
    gpucheck_stage kernels-fp8 900 --keep-going --mode fast kernels
    local prompt; prompt="$(jq -r '.prompts[0].prompt' "$FV_ROOT/scripts/gpu/prompts.json")"
    local negative; negative="$(jq -r '.negative' "$FV_ROOT/scripts/gpu/prompts.json")"
    jq -n --arg p "$prompt" --arg n "$negative" \
      '{negative: $n, prompts: [{name: "fp8", prompt: $p}]}' >"$RUN_DIR/prompt.json"
    fv_rsync_to "$HOST" "$PORT" "$RUN_DIR/prompt.json" "$OUTR/prompt.json" >/dev/null
    log "fp8 tier: checkpoint $FAST_REPO"
    local embeds="$OUTR/embeds"
    remote_run wait-text 1800 wait-weights "$BASE_W" 1800 text_encoder
    gpucheck_stage embed 1800 --mode exact embed --weights "$BASE_W" --prompts "$OUTR/prompt.json" \
      --embeds "$embeds" --device cuda
    remote_run wait-fast 1800 wait-weights "$FAST_W" 1800 transformer vae
    # 2s keeps the A/B cheap; the frame count is the only thing dropped, and the
    # per-step cost is what the timing question is about.
    local ab=(--height 448 --width 832 --frames "${FV_FP8_FRAMES:-33}" --steps 3
              --guidance 1.0 --flow-shift 8.0 --dmd --fps 16)
    gpucheck_stage clip-bf16 1800 --mode fast clip --weights "$FAST_W" \
      --embeds "$embeds/fp8.safetensors" --device cuda "${ab[@]}" \
      --name dmd-bf16 --budget-min 20 --no-mp4
    gpucheck_stage clip-fp8 1800 --mode fast --fp8 clip --weights "$FAST_W" \
      --embeds "$embeds/fp8.safetensors" --device cuda "${ab[@]}" \
      --name dmd-fp8 --budget-min 20 --no-mp4
    # Limits are loose on purpose: this stage measures the cost of FP8 rather
    # than gating it. What the numbers mean is a judgement for the report, not
    # something to bake in before we have ever seen one.
    gpucheck_stage compare-fp8 300 compare --a "$OUTR/clips/dmd-bf16" --b "$OUTR/clips/dmd-fp8" \
      --max-step1-rel 1.0 --max-latent-rel 1.0 --min-psnr 0.0
    log "FP8 A/B done"
    return 0
  fi

  # `taehv` judges our tiny-autoencoder port against the implementation it was
  # read from. No Wan weights are involved: the oracle fetches taehv.py and
  # taew2_1.safetensors, and both sides then decode the identical latent.
  if [[ "$tier" == taehv ]]; then
    gpucheck_stage nvrtc 300 nvrtc
    gpucheck_stage device 300 device
    remote_run upstream-install "${FV_UPSTREAM_INSTALL_TIMEOUT:-2400}" upstream-install "${FV_TORCH_BACKEND:-cu126}"
    remote_run taehv-oracle "${FV_TAEHV_TIMEOUT:-1800}" taehv-oracle \
      --latent-frames "${FV_TAEHV_LATENT_FRAMES:-3}" --height "${FV_TAEHV_H:-56}" --width "${FV_TAEHV_W:-104}"
    # Exact mode: the reference runs float32, and a looser judge cannot set a
    # limit on it.
    gpucheck_stage taehv 900 --keep-going --mode exact taehv --weights "$WORK/taehv" \
      --oracle "$OUTR/taehv/oracle.safetensors" --device cuda
    log "TAEHV oracle done"
    return 0
  fi

  # `vaeab` decodes one clip both ways. The Wan VAE is 3.9s of a 23.7s H100
  # clip and TAEHV is ~10M parameters against its ~130M, so the time question
  # is close to settled — what this measures is the *quality* trade, which the
  # oracle tier says nothing about because it only proves our port matches
  # madebyollin's, not that TAEHV is good enough for a clip.
  if [[ "$tier" == vaeab ]]; then
    local prompt; prompt="$(jq -r '.prompts[0].prompt' "$FV_ROOT/scripts/gpu/prompts.json")"
    local negative; negative="$(jq -r '.negative' "$FV_ROOT/scripts/gpu/prompts.json")"
    jq -n --arg p "$prompt" --arg n "$negative" \
      '{negative: $n, prompts: [{name: "vaeab", prompt: $p}]}' >"$RUN_DIR/prompt.json"
    fv_rsync_to "$HOST" "$PORT" "$RUN_DIR/prompt.json" "$OUTR/prompt.json" >/dev/null
    local embeds="$OUTR/embeds"
    remote_run wait-text 1800 wait-weights "$BASE_W" 1800 text_encoder
    gpucheck_stage embed 1800 --mode exact embed --weights "$BASE_W" --prompts "$OUTR/prompt.json" \
      --embeds "$embeds" --device cuda
    remote_run wait-fast 1800 wait-weights "$FAST_W" 1800 transformer vae
    # taehv-oracle is what puts taew2_1.safetensors on the box; it needs torch,
    # which is the only reason this tier installs the upstream venv at all.
    remote_run upstream-install "${FV_UPSTREAM_INSTALL_TIMEOUT:-2400}" upstream-install "${FV_TORCH_BACKEND:-cu126}"
    remote_run taehv-oracle "${FV_TAEHV_TIMEOUT:-1800}" taehv-oracle --latent-frames 1 --height 8 --width 8
    local ab=(--height 448 --width 832 --frames "${FV_VAEAB_FRAMES:-129}" --steps 3
              --guidance 1.0 --flow-shift 8.0 --dmd --fps 16 --vsa)
    gpucheck_stage clip-wanvae "$(( budget_min * 60 * 13 / 10 + 600 ))" --mode fast clip \
      --weights "$FAST_W" --embeds "$embeds/vaeab.safetensors" --device cuda "${ab[@]}" \
      --vae-chunk 2 --name dmd-wanvae --budget-min "$budget_min"
    gpucheck_stage clip-taehv "$(( budget_min * 60 * 13 / 10 + 600 ))" --mode fast clip \
      --weights "$FAST_W" --embeds "$embeds/vaeab.safetensors" --device cuda "${ab[@]}" \
      --taehv-weights "$WORK/taehv" --name dmd-taehv --budget-min "$budget_min"
    # Wide-open limits: two different decoders will not agree numerically, and
    # the point is to see how far apart they are, not to gate on a number
    # nobody has looked at yet.
    gpucheck_stage compare-vae 300 compare --a "$OUTR/clips/dmd-wanvae" --b "$OUTR/clips/dmd-taehv" \
      --max-step1-rel 1e9 --max-latent-rel 1e9 --min-psnr 0.0
    log "VAE A/B done"
    return 0
  fi

  # `compare` benchmarks two implementations; it does not re-validate them, so
  # it skips T1/T2 and goes straight to the clip stages it times. That also
  # keeps it off exact-mode parity, whose peak does not fit a 24GB card.
  if [[ "$tier" == compare ]]; then
    gpucheck_stage nvrtc 300 nvrtc
    gpucheck_stage device 300 device
  else
  # T1: kernels vs plain-Rust math; random-weight model, GPU vs cudarc CPU path.
  gpucheck_stage nvrtc 300 nvrtc
  gpucheck_stage device 300 device
  gpucheck_stage kernels-exact 900 --keep-going --mode exact kernels
  gpucheck_stage kernels-fast 900 --keep-going --mode fast kernels
  if [[ $have_model_ref -eq 0 ]]; then
    gpucheck_stage model-cpu-ref 600 --mode exact model --device cpu --dump "$refs"
    ref_save model model-cpu-ref
  fi
  gpucheck_stage model-exact 600 --mode exact model --device cuda --reference "$refs"
  gpucheck_stage model-fast 600 --mode fast model --device cuda --reference "$refs"
  [[ "$tier" == kernels ]] && { log "T1 PASS"; return 0; }

  # T2: real 1.3B weights, GPU vs cudarc CPU path.
  remote_run wait-base 1800 wait-weights "$BASE_W" 1800 transformer vae
  if [[ $have_parity_ref -eq 0 ]]; then
    gpucheck_stage parity-cpu-ref 3600 --mode exact parity --weights "$BASE_W" --device cpu --dump "$refs"
    ref_save parity parity-cpu-ref
  fi
  gpucheck_stage parity-exact 1200 --mode exact parity --weights "$BASE_W" --device cuda --reference "$refs"
  gpucheck_stage parity-fast 1200 --mode fast parity --weights "$BASE_W" --device cuda --reference "$refs"
  [[ "$tier" == parity ]] && { log "T2 PASS"; return 0; }
  fi

  # T3: 8s clips. Prompts are encoded by cudarc UMT5 on this GPU (own process,
  # BF16 off so the XXL weights fit), then the probe must project a clip that
  # fits the time/VRAM budget before any long run starts.
  local embeds="$OUTR/embeds"
  remote_run wait-text 1800 wait-weights "$BASE_W" 1800 text_encoder
  gpucheck_stage embed 1800 --mode exact embed --weights "$BASE_W" --prompts "$FV_REMOTE_DIR/scripts/gpu/prompts.json" \
    --embeds "$embeds" --device cuda
  remote_run wait-fast 1800 wait-weights "$FAST_W" 1800 transformer vae
  local names first
  names="$(jq -r '.prompts[].name' "$FV_ROOT/scripts/gpu/prompts.json")"
  first="$(head -1 <<<"$names")"
  local clip=(--height 448 --width 832 --frames 129 --steps 3 --guidance 1.0 --flow-shift 8.0 --dmd --fps 16)
  # FV_VSA=1 runs the generation stages with video sparse attention. It is not
  # passed to the model/parity stages: those compare against dense references.
  if [[ "${FV_VSA:-0}" == 1 ]]; then
    clip+=(--vsa)
  fi
  # Same reasoning as --vsa: the generation stages only. Exact parity is at the
  # edge of a 24GB card and a larger VAE chunk pushes it over.
  if [[ -n "${FV_VAE_CHUNK:-}" ]]; then
    clip+=(--vae-chunk "$FV_VAE_CHUNK")
  fi
  gpucheck_stage probe-8s 1800 --mode fast probe --weights "$FAST_W" --embeds "$embeds/$first.safetensors" \
    --device cuda "${clip[@]}" --budget-min "$budget_min"
  local clip_timeout=$(( budget_min * 60 * 13 / 10 + 600 ))
  local name
  for name in $names; do
    gpucheck_stage "clip-8s-$name" "$clip_timeout" --mode fast clip --weights "$FAST_W" \
      --embeds "$embeds/$name.safetensors" --device cuda "${clip[@]}" \
      --name "dmd-8s-$name" --budget-min "$budget_min"
  done
  # Precision cost of the fast path on a 2s clip: same seed, exact vs fast.
  local short=(--height 448 --width 832 --frames 33 --steps 3 --guidance 1.0 --flow-shift 8.0 --dmd --fps 16)
  gpucheck_stage clip-2s-exact 1800 --mode exact clip --weights "$FAST_W" --embeds "$embeds/$first.safetensors" \
    --device cuda "${short[@]}" --name dmd-2s-exact --budget-min 20 --no-mp4
  gpucheck_stage clip-2s-fast 1800 --mode fast clip --weights "$FAST_W" --embeds "$embeds/$first.safetensors" \
    --device cuda "${short[@]}" --name dmd-2s-fast --budget-min 20 --no-mp4
  gpucheck_stage compare-2s 300 compare --a "$OUTR/clips/dmd-2s-exact" --b "$OUTR/clips/dmd-2s-fast"
  [[ "$tier" == clip ]] && { log "T3 PASS"; return 0; }

  # T4: the same 8s clip through upstream FastVideo on this same GPU. Upstream
  # pulls its own copy of the weights (its loader expects the HF cache layout),
  # so this only measures generation with the pipeline already resident.
  remote_run upstream-install "${FV_UPSTREAM_INSTALL_TIMEOUT:-2400}" upstream-install "${FV_TORCH_BACKEND:-cu126}"
  local ub=(--model-path "$FAST_REPO" --height 448 --width 832 --num-frames 129 --steps 3
            --guidance 1.0 --fps 16 --seed 0 --runs "${FV_UPSTREAM_RUNS:-2}")
  local backend
  for backend in ${FV_UPSTREAM_BACKENDS:-TORCH_SDPA VIDEO_SPARSE_ATTN}; do
    local extra=()
    if [[ "$backend" == VIDEO_SPARSE_ATTN* ]]; then
      extra=(--vsa-sparsity "${FV_VSA_SPARSITY:-0.8}")
    fi
    # A backend that will not import or run is a result, not a run failure.
    STAGE_OPTIONAL=1 remote_run "upstream-$backend" "${FV_UPSTREAM_TIMEOUT:-3600}" upstream-bench "$backend" "${ub[@]}" ${extra[@]+"${extra[@]}"} || true
  done
  log "T4 PASS"
}

# ---- h3-matrix ------------------------------------------------------------------
# Kill leftover GPU processes and require empty VRAM. Restart the instance if
# VRAM stays dirty (image already local — no pull).
h3_matrix_gpu_empty() {
  local used
  # killall matches the process name only. Do NOT use `pkill -f hf-fm` — that
  # also matches this ssh command line and kills the session (rc 255).
  fv_ssh "$HOST" "$PORT" "killall -9 fv-gpucheck hf-fm hf-fetch-model 2>/dev/null || true; sleep 2" || true
  used="$(fv_ssh "$HOST" "$PORT" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc 0-9" || echo 0)"
  if [[ "${used:-0}" -gt "${FV_MAX_GPU_USED_MIB:-2048}" ]]; then
    log "GPU still ${used} MiB used — restarting instance $INSTANCE (no re-rent)"
    # Pull timeout does not apply to a restart (image already local).
    local saved_pull="${FV_IMAGE_PULL_TIMEOUT:-}"
    unset FV_IMAGE_PULL_TIMEOUT
    vastai restart instance "$INSTANCE" >/dev/null || die "vastai restart failed"
    HOST=""; PORT=""
    wait_ready || die "instance $INSTANCE unreachable after restart"
    [[ -n "$saved_pull" ]] && export FV_IMAGE_PULL_TIMEOUT="$saved_pull"
    used="$(fv_ssh "$HOST" "$PORT" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc 0-9" || echo 0)"
    [[ "${used:-0}" -le "${FV_MAX_GPU_USED_MIB:-2048}" ]] || die "GPU still dirty after restart: ${used} MiB"
  fi
  log "GPU empty (${used:-0} MiB used)"
}

# Weight dir for a matrix recipe key (`v2` → `h3-v2`, synth siblings share 4step-vsa contract).
h3_matrix_weights() {
  case "$1" in
    v2|8step) echo "$WORK/weights/h3-v2" ;;
    4step-vsa|preview-vsa) echo "$WORK/weights/h3-4step-vsa" ;;
    4step-vsa-synth1300) echo "$WORK/weights/h3-4step-vsa-synth1300" ;;
    4step-vsa-synth1900) echo "$WORK/weights/h3-4step-vsa-synth1900" ;;
    4step-dense|preview-dense) echo "$WORK/weights/h3-4step-dense" ;;
    *) die "unknown matrix recipe $1" ;;
  esac
}

# CLI --h3-recipe for a matrix key (synth cells reuse the Preview VSA ladder).
h3_matrix_inference_recipe() {
  case "$1" in
    v2|8step) echo v2 ;;
    4step-vsa|preview-vsa|4step-vsa-synth1300|4step-vsa-synth1900) echo 4step-vsa ;;
    4step-dense|preview-dense) echo 4step-dense ;;
    *) die "unknown matrix recipe $1" ;;
  esac
}

# Clip seconds from residency (`streamed-15s` → 15, else FV_SECONDS/5).
h3_matrix_seconds() {
  local residency="$1"
  case "$residency" in
    *-15s|15s) echo 15 ;;
    *-10s|10s) echo 10 ;;
    *) echo "${FV_SECONDS:-5}" ;;
  esac
}

# Strip duration suffix for encoder wiring (`streamed-15s` → `streamed`).
h3_matrix_residency_base() {
  local r="$1"
  r="${r%-15s}"
  r="${r%-10s}"
  echo "$r"
}

# One cell: TEST_ID, GPU clear, nohup gen. Args: sku recipe encoder residency
h3_matrix_cell() {
  local sku="$1" recipe="$2" encoder="$3" residency="$4"
  local test_id prior
  local res_base; res_base="$(h3_matrix_residency_base "$residency")"
  local secs; secs="$(h3_matrix_seconds "$residency")"
  # Reattach after a mid-matrix fix: keep a prior successful clip for this cell.
  prior="$(fv_ssh "$HOST" "$PORT" "ls -1d $OUTR/clips/*-${sku}-${recipe}-${encoder}-${residency}/frames/output.mp4 2>/dev/null | head -1" || true)"
  if [[ -n "$prior" ]]; then
    test_id="$(basename "$(dirname "$(dirname "$prior")")")"
    log "skip cell ${sku}/${recipe}/${encoder}/${residency} — reusing $test_id"
    printf '%s\n' "$test_id" >>"$RUN_DIR/matrix-test-ids.txt"
    return 0
  fi
  test_id="$(date -u +%Y%m%dT%H%M%SZ)-${sku}-${recipe}-${encoder}-${residency}"
  test_id="${test_id//_/-}"
  export FV_TEST_ID="$test_id"
  log "▶ TEST_ID=$test_id"
  h3_matrix_gpu_empty
  local clip="$OUTR/clips/$test_id"
  local wdir; wdir="$(h3_matrix_weights "$recipe")"
  local irecipe; irecipe="$(h3_matrix_inference_recipe "$recipe")"
  local taeh3_dir="$WORK/taeh3"
  local tcache="$WORK/h3-text-cache-$recipe"
  local adaln="$WORK/h3-adaln-$recipe.cache"
  local stage_timeout=7200
  (( secs > 5 )) && stage_timeout=10800
  local h3gen=(--mode fast h3 gen --weights "$wdir" --prompt "$FV_PROMPT" --seconds "$secs"
    --seed "${FV_SEED:-1024}" --adaln-cache "$adaln" --warm --taeh3-weights "$taeh3_dir"
    --h3-recipe "$irecipe" --clip-dir "$clip/frames")
  case "$encoder/$res_base" in
    stock/streamed)
      # Write the conditioning cache so a later cache-hit cell can reuse it.
      h3gen+=(--text-encoder streamed --text-cache "$tcache")
      ;;
    stock/cache-hit)
      h3gen+=(--text-encoder streamed --text-cache "$tcache")
      ;;
    stock/resident-fp8|stock/resident-bf16)
      h3gen+=(--text-encoder "${res_base}" --text-cache "$tcache")
      ;;
    recovered-8b/resident-bf16)
      h3gen+=(--text-encoder recovered-8b --text-weights "$WORK/weights/recovered-8b" --text-cache "$tcache")
      ;;
    *) die "unknown matrix cell $encoder/$residency" ;;
  esac
  gpucheck_stage "$test_id" "$stage_timeout" "${h3gen[@]}"
  printf '%s\n' "$test_id" >>"$RUN_DIR/matrix-test-ids.txt"
  unset FV_TEST_ID
}

# PSNR between two clips that share this box (8B vs stock on the same checkpoint).
h3_matrix_psnr() {
  local a_id="$1" b_id="$2" label="$3"
  local a="$OUTR/clips/$a_id/frames/output.mp4" b="$OUTR/clips/$b_id/frames/output.mp4"
  log "PSNR $label: $a_id vs $b_id"
  local out
  out="$(fv_ssh "$HOST" "$PORT" "ffmpeg -hide_banner -i '$a' -i '$b' -lavfi '[0:v][1:v]psnr' -f null - 2>&1 | tail -5" || true)"
  printf '%s\n' "$out" | sed "s/^/  [psnr] /" >&2
  printf '%s\t%s\t%s\t%s\n' "$label" "$a_id" "$b_id" "$(printf '%s' "$out" | tr '\n' ' ')" >>"$RUN_DIR/matrix-psnr.tsv"
}

run_h3_matrix() {
  local prompt="${FV_PROMPT:-A man in his thirties talking to the camera in a bright living room, medium close-up, natural expressions and hand gestures, soft window light. He says: <d>Hello, this was generated entirely in Rust.</d>}"
  export FV_PROMPT="$prompt"
  local sku="${FV_MATRIX_SKU:-}"
  if [[ -z "$sku" ]]; then
    case "${FV_OFFER_QUERY_EXTRA:-}" in
      *[Hh]200*) sku=h200 ;;
      *[Hh]100*) sku=h100 ;;
      *[Aa]100*) sku=a100 ;;
      *) sku=auto ;;
    esac
  fi
  sku="$(tr '[:upper:]' '[:lower:]' <<<"$sku")"
  # full (default) | followup — skip V2 gens; synth1900 + Preview-80 + 15s on H200.
  local wave="${FV_MATRIX_WAVE:-full}"
  log "h3-matrix sku=$sku wave=$wave instance=$INSTANCE"
  mkdir -p "$RUN_DIR"
  : >"$RUN_DIR/matrix-test-ids.txt"
  : >"$RUN_DIR/matrix-psnr.tsv"
  : >"$RUN_DIR/matrix-skips.tsv"

  # --- fetches (shared on this box for every cell) ---
  local v2_repo="${FV_H3_REPO:-FastVideo/FastVideo-FastH3-8-Step-V2}"
  local v2="$WORK/weights/h3-v2"
  local preview_vsa_repo="${FV_H3_PREVIEW_VSA_REPO:-FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree}"
  local preview_dense_repo="${FV_H3_PREVIEW_DENSE_REPO:-FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree}"
  local synth1900_repo="${FV_H3_PREVIEW_SYNTH1900_REPO:-FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-Synthetic-Step1900}"
  local text_pat=("tokenizer/*" "text_encoder/*.json"
    "text_encoder/model-0000[1-9]-of-00014.safetensors" "text_encoder/model-0001[01]-of-00014.safetensors")
  remote_run fetch-v2 120 fetch "$v2_repo" "$v2" "${text_pat[@]}" "transformer/*" "audio_vae/*"
  local taeh3_dir="$WORK/taeh3"
  remote_run fetch-taeh3 300 fetch-taeh3 "$taeh3_dir"
  remote_run wait-v2 7200 wait-weights "$v2" 7200 text_encoder transformer audio_vae
  remote_run wait-taeh3 300 wait-taeh3 "$taeh3_dir" 300

  local recovered="$WORK/weights/recovered-8b"
  remote_run fetch-8b 120 fetch SearchingMan/MiniMax-H3-Text-Encoders "$recovered" \
    "text_encoders/recovered_8b/qwen3vl_8b_minimax_h3_recovered_bf16.safetensors" \
    "text_encoders/recovered_8b/ara.safetensors" \
    "text_encoders/recovered_8b/conditioning_adapter.safetensors" \
    "text_encoders/recovered_8b/minimax_h3_recovered_8b_manifest.json"
  # SearchingMan layout is nested; verify leaves, not component dirs.
  remote_run wait-8b 7200 wait-weights "$recovered" 7200

  # --- V2 cells (all SKUs) — skipped on followup wave (already signed off) ---
  local id_v2_stock id_v2_8b id_v2_cache
  if [[ "$wave" != followup ]]; then
    h3_matrix_cell "$sku" v2 stock streamed
    id_v2_stock="$(tail -1 "$RUN_DIR/matrix-test-ids.txt")"
    h3_matrix_cell "$sku" v2 recovered-8b resident-bf16
    id_v2_8b="$(tail -1 "$RUN_DIR/matrix-test-ids.txt")"
    h3_matrix_cell "$sku" v2 stock cache-hit
    id_v2_cache="$(tail -1 "$RUN_DIR/matrix-test-ids.txt")"
    h3_matrix_psnr "$id_v2_stock" "$id_v2_8b" "v2-stock-vs-8b"
  else
    log "wave=followup — skipping V2 gen cells"
    printf 'skip\tv2\twave-followup\n' >>"$RUN_DIR/matrix-skips.tsv"
  fi

  if [[ "$sku" == h200 && "$wave" != followup ]]; then
    h3_matrix_cell "$sku" v2 stock resident-fp8
    h3_matrix_cell "$sku" v2 stock resident-bf16

    # Preview VSA + Dense (H200 first). Repos may need FV_H3_PREVIEW_* overrides.
    # `fetch` only starts hf-fm in the background — wait for .complete before cells.
    local pvsa="$WORK/weights/h3-4step-vsa"
    STAGE_OPTIONAL=1 remote_run fetch-preview-vsa 120 fetch "$preview_vsa_repo" "$pvsa" \
      "transformer/*" "audio_vae/*" || true
    if STAGE_OPTIONAL=1 remote_run wait-preview-vsa 7200 wait-weights "$pvsa" 7200 transformer audio_vae; then
      fv_ssh "$HOST" "$PORT" "ln -sfn $v2/text_encoder $pvsa/text_encoder; ln -sfn $v2/tokenizer $pvsa/tokenizer"
      local id_pv_stock id_pv_8b
      h3_matrix_cell "$sku" 4step-vsa stock streamed
      id_pv_stock="$(tail -1 "$RUN_DIR/matrix-test-ids.txt")"
      h3_matrix_cell "$sku" 4step-vsa recovered-8b resident-bf16
      id_pv_8b="$(tail -1 "$RUN_DIR/matrix-test-ids.txt")"
      h3_matrix_cell "$sku" 4step-vsa stock cache-hit
      h3_matrix_psnr "$id_pv_stock" "$id_pv_8b" "4step-vsa-stock-vs-8b"
    else
      log "preview VSA fetch skipped/failed — set FV_H3_PREVIEW_VSA_REPO if the Hub id differs"
      printf 'skip\t4step-vsa\tpreview-repo-missing\n' >>"$RUN_DIR/matrix-skips.tsv"
    fi

    local pdense="$WORK/weights/h3-4step-dense"
    STAGE_OPTIONAL=1 remote_run fetch-preview-dense 120 fetch "$preview_dense_repo" "$pdense" \
      "transformer/*" "audio_vae/*" || true
    if STAGE_OPTIONAL=1 remote_run wait-preview-dense 7200 wait-weights "$pdense" 7200 transformer audio_vae; then
      fv_ssh "$HOST" "$PORT" "ln -sfn $v2/text_encoder $pdense/text_encoder; ln -sfn $v2/tokenizer $pdense/tokenizer"
      h3_matrix_cell "$sku" 4step-dense stock streamed
    else
      log "preview Dense fetch skipped/failed — set FV_H3_PREVIEW_DENSE_REPO if needed"
      printf 'skip\t4step-dense\tpreview-repo-missing\n' >>"$RUN_DIR/matrix-skips.tsv"
    fi
  fi

  # --- Followup wave (H200): DataFree baseline + Synthetic Step1900 + 15s ---
  if [[ "$sku" == h200 && "$wave" == followup ]]; then
    local pvsa="$WORK/weights/h3-4step-vsa"
    STAGE_OPTIONAL=1 remote_run fetch-preview-vsa 120 fetch "$preview_vsa_repo" "$pvsa" \
      "transformer/*" "audio_vae/*" || true
    if STAGE_OPTIONAL=1 remote_run wait-preview-vsa 7200 wait-weights "$pvsa" 7200 transformer audio_vae; then
      fv_ssh "$HOST" "$PORT" "ln -sfn $v2/text_encoder $pvsa/text_encoder; ln -sfn $v2/tokenizer $pvsa/tokenizer"
      local id_df_stock
      h3_matrix_cell "$sku" 4step-vsa stock streamed
      id_df_stock="$(tail -1 "$RUN_DIR/matrix-test-ids.txt")"
    else
      log "preview VSA DataFree fetch failed — synth PSNR baseline unavailable"
      printf 'skip\t4step-vsa\tpreview-repo-missing\n' >>"$RUN_DIR/matrix-skips.tsv"
      id_df_stock=""
    fi

    local psynth="$WORK/weights/h3-4step-vsa-synth1900"
    STAGE_OPTIONAL=1 remote_run fetch-preview-synth1900 120 fetch "$synth1900_repo" "$psynth" \
      "transformer/*" "audio_vae/*" || true
    if STAGE_OPTIONAL=1 remote_run wait-preview-synth1900 7200 wait-weights "$psynth" 7200 transformer audio_vae; then
      fv_ssh "$HOST" "$PORT" "ln -sfn $v2/text_encoder $psynth/text_encoder; ln -sfn $v2/tokenizer $psynth/tokenizer"
      local id_sy_stock
      h3_matrix_cell "$sku" 4step-vsa-synth1900 stock streamed
      id_sy_stock="$(tail -1 "$RUN_DIR/matrix-test-ids.txt")"
      if [[ -n "$id_df_stock" ]]; then
        h3_matrix_psnr "$id_df_stock" "$id_sy_stock" "4step-vsa-datafree-vs-synth1900"
      fi
    else
      log "preview VSA Synthetic-Step1900 fetch skipped/failed — set FV_H3_PREVIEW_SYNTH1900_REPO if needed"
      printf 'skip\t4step-vsa-synth1900\tpreview-repo-missing\n' >>"$RUN_DIR/matrix-skips.tsv"
    fi

    # Longer clip on recommended DataFree Preview VSA (same res ladder as seconds=15).
    if [[ -n "$id_df_stock" ]] || fv_ssh "$HOST" "$PORT" "test -f $pvsa/.complete"; then
      fv_ssh "$HOST" "$PORT" "ln -sfn $v2/text_encoder $pvsa/text_encoder; ln -sfn $v2/tokenizer $pvsa/tokenizer" || true
      h3_matrix_cell "$sku" 4step-vsa stock streamed-15s
    else
      printf 'skip\t4step-vsa-15s\tno-datafree-weights\n' >>"$RUN_DIR/matrix-skips.tsv"
    fi
  fi

  # Preview VSA on 80 GB (A100/H100). followup wave always; full wave via FV_MATRIX_PREVIEW_80=1.
  local preview80="${FV_MATRIX_PREVIEW_80:-0}"
  [[ "$wave" == followup ]] && preview80=1
  if [[ "$preview80" == 1 && "$sku" != h200 ]]; then
    local pvsa="$WORK/weights/h3-4step-vsa"
    STAGE_OPTIONAL=1 remote_run fetch-preview-vsa 120 fetch "$preview_vsa_repo" "$pvsa" \
      "transformer/*" "audio_vae/*" || true
    if STAGE_OPTIONAL=1 remote_run wait-preview-vsa 7200 wait-weights "$pvsa" 7200 transformer audio_vae; then
      fv_ssh "$HOST" "$PORT" "ln -sfn $v2/text_encoder $pvsa/text_encoder; ln -sfn $v2/tokenizer $pvsa/tokenizer"
      local id_p80_stock id_p80_8b
      h3_matrix_cell "$sku" 4step-vsa stock streamed
      id_p80_stock="$(tail -1 "$RUN_DIR/matrix-test-ids.txt")"
      h3_matrix_cell "$sku" 4step-vsa recovered-8b resident-bf16
      id_p80_8b="$(tail -1 "$RUN_DIR/matrix-test-ids.txt")"
      h3_matrix_psnr "$id_p80_stock" "$id_p80_8b" "4step-vsa-stock-vs-8b"
    else
      log "preview VSA on 80 GB fetch skipped/failed"
      printf 'skip\t4step-vsa-80\tpreview-repo-missing\n' >>"$RUN_DIR/matrix-skips.tsv"
    fi
  fi

  log "h3-matrix done for $sku; TEST_IDs:"
  cat "$RUN_DIR/matrix-test-ids.txt" >&2 || true
  # Allow cleanup to destroy this SKU's box.
  KEEP=0
}

cmd_reap() {
  require_tools vastai jq
  local ids
  ids="$(vastai show instances --raw | jq -r --arg p "$FV_LABEL_PREFIX" '.[] | select((.label // "") | startswith($p)) | .id')"
  [[ -n "$ids" ]] || { log "no ${FV_LABEL_PREFIX}* instances"; return 0; }
  for id in $ids; do vast_destroy "$id" || true; done
}

# Tests source this file to exercise helpers without dispatching.
[[ -n "${FV_SOURCE_ONLY:-}" && "${BASH_SOURCE[0]}" != "$0" ]] && return 0

case "${1:-}" in
  local) shift; cmd_local "$@" ;;
  offers) shift; cmd_offers "$@" ;;
  run) shift; cmd_run "$@" ;;
  reap) cmd_reap ;;
  *) usage ;;
esac
