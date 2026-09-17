#!/usr/bin/env bash
# Tiered, fail-fast validation of the cudarc Wan backend on the cheapest
# hardware that can prove each property. See scripts/gpu/README.md.
#
#   validate.sh local                 free       preflight: unit tests + CUDA feature type-check
#   validate.sh offers <tier>                    cheapest matching offers + cost cap (no rental)
#   validate.sh run <tier> [opts]     T1-T3      rent → stages → pull artifacts → destroy
#   validate.sh reap                             destroy every fvgpu-* instance
#
# tiers: kernels (T1) | parity (T2) | clip (T3)   — each tier includes the ones before it.
set -euo pipefail
# shellcheck source=scripts/gpu/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

RUNS="${FV_RUNS:-$FV_ROOT/artifacts/gpucheck/runs}"
DOCKER_SH="$FV_ROOT/scripts/gpu/docker.sh"
DIST="$FV_ROOT/artifacts/gpucheck/dist"
LOCAL_REFS="$FV_ROOT/artifacts/gpucheck/refs"
IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}"
WORK="/workspace"
OUTR="$WORK/gpucheck-out"
BASE_REPO="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
FAST_REPO="FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
BASE_W="$WORK/weights/wan21-1.3b"
FAST_W="$WORK/weights/fastwan21-1.3b"

# ---- tier definitions ---------------------------------------------------------
# Offer filter, $/hr ceiling, hard wall-clock cap (watchdog destroys at the cap).
tier_query() {
  local base="num_gpus=1 compute_cap>=800 cuda_vers>=12.4 reliability>0.97 rentable=true verified=true direct_port_count>=1 inet_down>=200"
  case "$1" in
    kernels) echo "$base gpu_ram>=8 disk_space>=40 cpu_ram>=16" ;;
    parity) echo "$base gpu_ram>=16 disk_space>=60 cpu_ram>=32 inet_down>=500" ;;
    # UMT5-XXL loads ~60GB of host RAM (raw bytes + F32 views) before upload.
    clip) echo "$base gpu_ram>=24 disk_space>=100 cpu_ram>=80 inet_down>=500" ;;
    *) die "unknown tier '$1' (kernels|parity|clip)" ;;
  esac
}
tier_max_dph() { case "$1" in kernels) echo 0.25 ;; parity) echo 0.40 ;; clip) echo 0.60 ;; esac; }
tier_max_minutes() { case "$1" in kernels) echo 40 ;; parity) echo 75 ;; clip) echo 180 ;; esac; }
tier_disk() { case "$1" in kernels) echo 40 ;; parity) echo 60 ;; clip) echo 100 ;; esac; }

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

# ---- preflight: local, free ------------------------------------------------------
# Compile-level gates only: catches a broken build before paying for one.

cmd_local() {
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
    | jq '[.[] | {id, gpu_name, gpu_ram: (.gpu_ram/1024|floor), dph_total, reliability: (.reliability2 // .reliability),
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

INSTANCE=""; HOST=""; PORT=""; RUN_DIR=""; KEEP=0; OWN_INSTANCE=1; WATCHDOG_PID=""; DPH=0; T_START=0

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "$HOST" && -n "$RUN_DIR" ]]; then
    log "pulling artifacts → $RUN_DIR"
    fv_timeout 300 rsync -az -e "ssh -i $FV_SSH_KEY -p $PORT ${FV_SSH_OPTS[*]}" "root@$HOST:$OUTR/" "$RUN_DIR/remote/" \
      || log "artifact pull failed/timed out"
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
    log "run $([[ $rc -eq 0 ]] && echo PASSED || echo FAILED rc=$rc) in ${mins} min, ~\$${cost} — $RUN_DIR/summary.json"
  fi
  exit "$rc"
}

# remote_run <name> <timeout_s> <remote.sh args...>
# Detached on the box (survives ssh drops); polled with short ssh calls.
remote_run() {
  local name="$1" timeout_s="$2"; shift 2
  local args; args="$(printf '%q ' "$@")"
  local t0; t0=$(date +%s)
  log "▶ $name (timeout ${timeout_s}s)"
  fv_ssh "$HOST" "$PORT" "mkdir -p $OUTR/rc $OUTR/logs && rm -f $OUTR/rc/$name $OUTR/rc/$name.pid && cd $FV_REMOTE_DIR && \
    { setsid nohup bash -c 'bash scripts/gpu/remote.sh $args >$OUTR/logs/$name.driver.log 2>&1; echo \$? >$OUTR/rc/$name' \
      </dev/null >/dev/null 2>&1 & echo \$! >$OUTR/rc/$name.pid; }" || die "could not start $name"
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
  # Reports are small; pull after every stage so a later failure keeps them.
  fv_rsync_from "$HOST" "$PORT" "$OUTR/" "$RUN_DIR/remote/" --exclude 'clips/*/frames' >/dev/null 2>&1 || true
  if [[ "$rc" != "0" ]]; then
    log "✗ $name failed rc=$rc after ${secs}s — stopping (fail-fast). Report: $RUN_DIR/remote/"
    exit "$rc"
  fi
  log "✓ $name ${secs}s"
}

gpucheck_stage() {
  local tag="$1" timeout_s="$2"; shift 2
  remote_run "$tag" "$timeout_s" stage "$tag" "$timeout_s" --tag "$tag" "$@"
}

create_instance() {
  local tier="$1" offer="$2"
  local label out
  label="${FV_LABEL_PREFIX}-$(date -u +%Y%m%d%H%M%S)-$tier"
  out="$(vastai create instance "$offer" --image "$IMAGE" --disk "$(tier_disk "$tier")" --ssh --direct \
          --label "$label" --cancel-unavail --raw 2>&1)" || die "create failed: $out"
  INSTANCE="$(jq -r '.new_contract // empty' <<<"$out" 2>/dev/null)"
  [[ -n "$INSTANCE" ]] || die "create returned no contract id: $out"
  log "created instance $INSTANCE ($label)"
}

wait_ready() {
  local t0; t0=$(date +%s)
  local status=""
  while :; do
    status="$(vastai show instance "$INSTANCE" --raw 2>/dev/null | jq -r '.actual_status // "unknown"')" || status=unknown
    case "$status" in
      running) break ;;
      exited|offline|error) die "instance $INSTANCE entered '$status'" ;;
    esac
    (( $(date +%s) - t0 < ${FV_BOOT_TIMEOUT:-900} )) || die "instance not running after ${FV_BOOT_TIMEOUT:-900}s (status=$status)"
    sleep 10
  done
  log "instance running after $(( $(date +%s) - t0 ))s; waiting for ssh"
  while :; do
    if read -r HOST PORT < <(vast_ssh_target "$INSTANCE") && fv_ssh "$HOST" "$PORT" true 2>/dev/null; then
      break
    fi
    (( $(date +%s) - t0 < ${FV_BOOT_TIMEOUT:-900} + 300 )) || die "ssh never came up"
    sleep 10
  done
  log "ssh ok root@$HOST:$PORT"
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

  local max_dph mins
  max_dph="${MAX_DPH:-$(tier_max_dph "$tier")}"
  mins="${MAX_MINUTES:-$(tier_max_minutes "$tier")}"
  if [[ -z "$INSTANCE" ]]; then
    local pick
    pick="$(offers_json "$tier" | jq -c --arg id "$offer" --argjson max "$max_dph" \
      'map(select(if $id == "" then .dph_total <= $max else (.id|tostring) == $id end)) | .[0] // empty')"
    [[ -n "$pick" ]] || die "no offer ≤ \$$max_dph/hr for tier $tier (raise MAX_DPH or see '$0 offers $tier')"
    offer="$(jq -r .id <<<"$pick")"
    DPH="$(jq -r .dph_total <<<"$pick")"
    local cap balance
    cap="$(awk -v d="$DPH" -v m="$mins" 'BEGIN { printf "%.2f", d * m / 60 }')"
    balance="$(vast_balance)"
    log "offer $offer: $(jq -r '"\(.gpu_name) \(.gpu_ram)GB $\(.dph_total)/hr"' <<<"$pick"); worst case \$$cap (${mins} min cap); balance \$${balance:-?}"
    if [[ -n "$balance" ]] && awk -v b="$balance" -v c="$cap" 'BEGIN { exit !(b < c) }'; then
      die "balance \$$balance below worst-case \$$cap"
    fi
  else
    DPH="$(vastai show instance "$INSTANCE" --raw | jq -r '.dph_total // 0')"
  fi

  RUN_DIR="$RUNS/$(date -u +%Y%m%dT%H%M%SZ)-$tier"
  mkdir -p "$RUN_DIR/remote"
  T_START=$(date +%s)
  trap cleanup EXIT INT TERM
  if [[ $OWN_INSTANCE -eq 1 ]]; then
    create_instance "$tier" "$offer"
    # Independent of this shell: destroys at the wall cap even if we're killed.
    nohup bash -c "sleep $((mins * 60)); vastai destroy instance -y $INSTANCE" >/dev/null 2>&1 &
    WATCHDOG_PID=$!
  fi
  wait_ready

  # The box gets scripts + the Docker-built binary only: no source, no compile.
  log "uploading scripts and prebuilt fv-gpucheck (build $build_id)"
  fv_ssh "$HOST" "$PORT" "mkdir -p $FV_REMOTE_DIR/scripts $FV_REMOTE_DIR/target/release $OUTR/refs"
  fv_rsync_to "$HOST" "$PORT" "$FV_ROOT/scripts/" "$FV_REMOTE_DIR/scripts/" --delete --exclude '.env*'
  fv_rsync_to "$HOST" "$PORT" "$DIST/" "$FV_REMOTE_DIR/target/release/"
  local have_refs=0
  if [[ "$(cat "$LOCAL_REFS/build-id" 2>/dev/null)" == "$build_id" ]]; then
    fv_rsync_to "$HOST" "$PORT" "$LOCAL_REFS/" "$OUTR/refs/"
    have_refs=1
    log "using local CPU-path references for build $build_id"
  fi

  local need_disk; need_disk=$(( $(tier_disk "$tier") - 10 ))
  remote_run env 120 env "$need_disk"
  remote_run bootstrap 600 bootstrap
  # Downloads run in the background during the weight-free stages.
  case "$tier" in
    parity) remote_run fetch-base 120 fetch "$BASE_REPO" "$BASE_W" "transformer/*" "vae/*" ;;
    clip)
      remote_run fetch-base 120 fetch "$BASE_REPO" "$BASE_W" "transformer/*" "vae/*" "text_encoder/*" "tokenizer/*"
      remote_run fetch-fast 120 fetch "$FAST_REPO" "$FAST_W" "transformer/*" "vae/*"
      ;;
  esac

  local refs="$OUTR/refs"
  # T1: kernels vs plain-Rust math; random-weight model, GPU vs cudarc CPU path.
  gpucheck_stage nvrtc 300 nvrtc
  gpucheck_stage device 300 device
  gpucheck_stage kernels-exact 900 --keep-going --mode exact kernels
  gpucheck_stage kernels-fast 900 --keep-going --mode fast kernels
  if [[ $have_refs -eq 0 || ! -f "$LOCAL_REFS/model.safetensors" ]]; then
    gpucheck_stage model-cpu-ref 600 --mode exact model --device cpu --dump "$refs"
  fi
  gpucheck_stage model-exact 600 --mode exact model --device cuda --reference "$refs"
  gpucheck_stage model-fast 600 --mode fast model --device cuda --reference "$refs"
  [[ "$tier" == kernels ]] && { log "T1 PASS"; return 0; }

  # T2: real 1.3B weights, GPU vs cudarc CPU path.
  remote_run wait-base 1800 wait-weights "$BASE_W" 1800 transformer vae
  if [[ $have_refs -eq 0 || ! -f "$LOCAL_REFS/parity.safetensors" ]]; then
    gpucheck_stage parity-cpu-ref 3600 --mode exact parity --weights "$BASE_W" --device cpu --dump "$refs"
  fi
  gpucheck_stage parity-exact 1200 --mode exact parity --weights "$BASE_W" --device cuda --reference "$refs"
  gpucheck_stage parity-fast 1200 --mode fast parity --weights "$BASE_W" --device cuda --reference "$refs"
  [[ "$tier" == parity ]] && { log "T2 PASS"; return 0; }

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
  log "T3 PASS"
}

cmd_reap() {
  require_tools vastai jq
  local ids
  ids="$(vastai show instances --raw | jq -r --arg p "$FV_LABEL_PREFIX" '.[] | select((.label // "") | startswith($p)) | .id')"
  [[ -n "$ids" ]] || { log "no ${FV_LABEL_PREFIX}* instances"; return 0; }
  for id in $ids; do vast_destroy "$id" || true; done
}

# Tests source this file to exercise helpers without dispatching.
[[ -n "${FV_SOURCE_ONLY:-}" ]] && return 0

case "${1:-}" in
  local) shift; cmd_local "$@" ;;
  offers) shift; cmd_offers "$@" ;;
  run) shift; cmd_run "$@" ;;
  reap) cmd_reap ;;
  *) usage ;;
esac
