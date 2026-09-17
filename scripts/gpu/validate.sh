#!/usr/bin/env bash
# Tiered, fail-fast validation of the cudarc Wan backend on the cheapest
# hardware that can prove each property. See scripts/gpu/README.md.
#
#   validate.sh local                 free       preflight: unit tests + CUDA feature type-check
#   validate.sh offers <tier>                    cheapest matching offers + cost cap (no rental)
#   validate.sh run <tier> [opts]     T1-T3      rent → stages → pull artifacts → destroy
#   validate.sh reap                             destroy every fvgpu-* instance
#
# tiers: kernels (T1) | parity (T2) | clip (T3) | compare (T4, + upstream FastVideo)\n#        each tier includes the ones before it.
set -euo pipefail
# shellcheck source=scripts/gpu/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

RUNS="${FV_RUNS:-$FV_ROOT/artifacts/gpucheck/runs}"
DOCKER_SH="$FV_ROOT/scripts/gpu/docker.sh"
DIST="$FV_ROOT/artifacts/gpucheck/dist"
LOCAL_REFS="$FV_ROOT/artifacts/gpucheck/refs"
# Small runtime image built by .github/workflows/gpucheck-runtime-image.yml:
# CUDA runtime libraries cudarc loads + fv-gpucheck + scripts (no PyTorch).
IMAGE_REPO="${VAST_IMAGE_REPO:-ghcr.io/zaitrarrio/fastvideo-rs-runtime}"
IMAGE="${VAST_IMAGE:-}"
# The image bakes the binary and scripts here; uploads land in the same place.
FV_REMOTE_DIR="${VAST_REMOTE_DIR:-/opt/fastvideo-rs}"
# Machines that failed to boot / never accepted ssh / failed env or bootstrap.
BAD_HOSTS="${FV_BAD_HOSTS_FILE:-$FV_ROOT/artifacts/gpucheck/bad-machines.tsv}"
BAD_HOST_TTL_H="${FV_BAD_HOST_TTL_H:-24}"
CURRENT_MACHINE=""; LAST_STAGE=""; IMAGE_HAS_BUILD=0
WORK="/workspace"
OUTR="$WORK/gpucheck-out"
BASE_REPO="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
FAST_REPO="FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
BASE_W="$WORK/weights/wan21-1.3b"
FAST_W="$WORK/weights/fastwan21-1.3b"

# ---- tier definitions ---------------------------------------------------------
# Offer filter, $/hr ceiling, hard wall-clock cap (watchdog destroys at the cap).
tier_query() {
  # FV_OFFER_QUERY_EXTRA narrows the search, e.g. "gpu_name=RTX_4070S".
  local base="num_gpus=1 compute_cap>=800 cuda_vers>=12.4 reliability>0.97 rentable=true verified=true direct_port_count>=1 inet_down>=200 ${FV_OFFER_QUERY_EXTRA:-}"
  case "$1" in
    kernels) echo "$base gpu_ram>=8 disk_space>=40 cpu_ram>=16" ;;
    # cuBLAS math-mode probe: DiT linears at 8s-clip size (~2 GB of buffers).
    mathprobe) echo "$base gpu_ram>=12 disk_space>=40 cpu_ram>=16" ;;
    parity) echo "$base gpu_ram>=16 disk_space>=60 cpu_ram>=32 inet_down>=500" ;;
    # UMT5-XXL loads ~60GB of host RAM (raw bytes + F32 views) before upload.
    clip) echo "$base gpu_ram>=24 disk_space>=100 cpu_ram>=80 inet_down>=500" ;;
    # Same box runs our clip stages and upstream FastVideo: + torch wheels and
    # upstream's own copy of the weights in the HF cache.
    compare) echo "$base gpu_ram>=24 disk_space>=180 cpu_ram>=80 inet_down>=500" ;;
    *) die "unknown tier '$1' (mathprobe|kernels|parity|clip|compare)" ;;
  esac
}
tier_max_dph() { case "$1" in mathprobe) echo 0.40 ;; kernels) echo 0.25 ;; parity) echo 0.40 ;; clip) echo 0.60 ;; compare) echo 0.60 ;; esac; }
tier_max_minutes() { case "$1" in mathprobe) echo 30 ;; kernels) echo 40 ;; parity) echo 75 ;; clip) echo 180 ;; compare) echo 240 ;; esac; }
tier_disk() { case "$1" in mathprobe) echo 40 ;; kernels) echo 40 ;; parity) echo 60 ;; clip) echo 100 ;; compare) echo 180 ;; esac; }

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
# Prefer the CI image built from exactly these sources (binary baked in, no
# upload); otherwise :latest for the runtime libraries plus an uploaded binary.
resolve_image() {
  local build_id="$1"
  if [[ -n "$IMAGE" ]]; then
    log "image $IMAGE (VAST_IMAGE override; uploading binary)"
    return 0
  fi
  if command -v docker >/dev/null 2>&1 && docker manifest inspect "$IMAGE_REPO:build-$build_id" >/dev/null 2>&1; then
    IMAGE="$IMAGE_REPO:build-$build_id"
    IMAGE_HAS_BUILD=1
    log "image $IMAGE (CI-built from these sources; binary baked in)"
  else
    IMAGE="$IMAGE_REPO:latest"
    log "image $IMAGE (no CI image for build $build_id yet — push to main to publish one; uploading local binary)"
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
  if [[ -n "$HOST" && -n "$RUN_DIR" ]]; then
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
        printf '%s\n' "${text%$'\n'}" | sed "s/^/  [$name] /" >&2
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
  if ! out="$(vastai create instance "$offer" --image "$IMAGE" --disk "$(tier_disk "$tier")" --ssh --direct \
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
wait_ready() {
  local t0; t0=$(date +%s)
  local status=""
  while :; do
    status="$(vastai show instance "$INSTANCE" --raw 2>/dev/null | jq -r '.actual_status // "unknown"')" || status=unknown
    case "$status" in
      running) break ;;
      exited|offline|error) log "instance $INSTANCE entered '$status'"; return 1 ;;
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

cmd_run() {
  local tier="${1:-}"; [[ -n "$tier" ]] || usage; shift
  tier_query "$tier" >/dev/null
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
  [[ "$(cat "$DIST/fv-gpucheck.build-id" 2>/dev/null)" == "$build_id" ]] || die "dist binary is stale; run: $DOCKER_SH dist"
  resolve_image "$build_id"

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
  if [[ $IMAGE_HAS_BUILD -eq 1 ]]; then
    log "binary for build $build_id is baked into $IMAGE"
  else
    log "uploading prebuilt fv-gpucheck (build $build_id)"
    fv_rsync_to "$HOST" "$PORT" "$DIST/" "$FV_REMOTE_DIR/target/release/"
  fi
  # CPU-path references are cached per ref key (scripts/gpu/lib.sh fv_ref_key):
  # a hit uploads them and skips that CPU-reference stage entirely.
  REF_KEY="$(fv_ref_key)"
  fv_ssh "$HOST" "$PORT" "echo $REF_KEY >$OUTR/refs/ref-key"
  local have_model_ref=0 have_parity_ref=0
  ref_restore model && have_model_ref=1
  ref_restore parity && have_parity_ref=1

  local need_disk; need_disk=$(( $(tier_disk "$tier") - 10 ))
  remote_run env 120 env "$need_disk"
  remote_run bootstrap 600 bootstrap
  # Downloads run in the background during the weight-free stages.
  case "$tier" in
    parity) remote_run fetch-base 120 fetch "$BASE_REPO" "$BASE_W" "transformer/*" "vae/*" ;;
    clip | compare)
      remote_run fetch-base 120 fetch "$BASE_REPO" "$BASE_W" "transformer/*" "vae/*" "text_encoder/*" "tokenizer/*"
      remote_run fetch-fast 120 fetch "$FAST_REPO" "$FAST_W" "transformer/*" "vae/*"
      ;;
  esac

  local refs="$OUTR/refs"
  if [[ "$tier" == mathprobe ]]; then
    # Which cuBLAS math runs here: the image's cuBLAS, then a newer one.
    gpucheck_stage device 300 device
    gpucheck_stage gemm-probe-image 900 gemm-probe
    remote_run cublas-new 600 cublas "${FV_PROBE_CUBLAS_VERSION:-12.9.1.4}"
    gpucheck_stage gemm-probe-new 900 gemm-probe
    log "math probe done"
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
