#!/usr/bin/env bash
# Shared CPU build pod (docs/dev/build-pod.md): one Runpod CPU pod plus the
# `fv-build` network volume, driven over the pod's HTTPS proxy (no SSH).
# Each agent gets its own worktree snapshot and CARGO_TARGET_DIR on the pod.
#
#   build-pod.sh up                  reuse the running pod, start the stopped one,
#                                    or create it; wait until the toolchain is ready
#   build-pod.sh status              pod, $/hr, setup state, jobs, disk, the
#                                    self-stop timers (idle, time to idle / cap stop),
#                                    and per agent: idle hours, target/snapshot sizes,
#                                    when eviction takes them; recent evictions
#   build-pod.sh agents [--sizes]    agent dirs on the pod
#   build-pod.sh sync <agent>        mirror this worktree (tracked + untracked,
#                                    not ignored) to worktrees/<agent>/
#   build-pod.sh run <agent> [--no-sync] -- [K=V ...] <cmd...>
#                                    sync, then run an allowlisted command
#                                    (cargo check|build|test|clippy|fmt|doc|tree|
#                                    metadata, bash scripts/serve/check.sh,
#                                    bash scripts/gpu/lint.sh); streams the log
#                                    and exits with the command's status
#   build-pod.sh log <job>           re-attach to a job's log
#   build-pod.sh cancel <job>        cancel a job (e.g. after the client died)
#   build-pod.sh fetch <agent> <path> [dest]
#                                    copy target/<agent>/<path> back (gzip in
#                                    transit), e.g. release/fv-serve
#   build-pod.sh clean <agent> [target|worktree|all]   (default all)
#   build-pod.sh seed <agent> [--force]
#                                    sync, then build the deps seed of that worktree's
#                                    Cargo.lock now (jobs also start one after their
#                                    first success on a Cargo.lock without a seed)
#   build-pod.sh evict               run the pod's eviction pass now (it also runs
#                                    every minute: dirs unused > FV_BUILD_EVICT_HOURS,
#                                    default 6, then LRU target dirs while under
#                                    FV_BUILD_EVICT_FREE_GB, default 40, free)
#   build-pod.sh stop                stop the pod (terminate if Runpod refuses a
#                                    stop); the volume and its caches persist
#   build-pod.sh down                terminate the pod (volume persists)
#   build-pod.sh release-artifacts <sha|ref> [--sets "a b"] [--force] [--no-upload] [--keep]
#                                    build that commit's release binaries on the
#                                    pod (scripts/dev/release-artifacts-pod.sh),
#                                    verify and upload them to R2 artifacts/<sha>/
#                                    for the image workflows (wakes the pod)
#   build-pod.sh volume-create       create the fv-build volume (once)
#   build-pod.sh plan                print the pod create payload (no API call)
#
# <agent> is any [A-Za-z0-9._-] name; "." means this worktree's directory name.
#
# Money guards: balance floor FV_MIN_BALANCE (default 8 $) checked by `up`;
# $/hr cap FV_BUILD_MAX_DPH (default 1.5); the pod stops itself after
# FV_BUILD_IDLE_MIN (default 20) minutes without jobs and FV_BUILD_MAX_HOURS
# (default 8) after boot (+ FV_BUILD_MAX_GRACE_MIN, default 30, for a job still
# running), enforced on the pod so it survives this container; a curl watchdog
# in the start command repeats the cap 15 min later, and fv-control's cron
# stops the pod past 9 h or 15 min past a missed idle stop (docs/dev/build-pod.md);
# ledger at $FV_BUILD_STATE/ledger.tsv.
#
# Env: RUNPOD_API_KEY; FV_BUILD_STATE (default ~/.config/fv-build: token,
# mode 600, never printed); FV_BUILD_VOLUME (default fv-build);
# FV_BUILD_FLAVORS (default "cpu5c cpu3c"); FV_BUILD_VCPUS (default 32) and
# FV_BUILD_VCPUS_FALLBACK (default 16, taken when the first has no stock);
# FV_BUILD_CONTAINER_GB (default 200: snapshots and target dirs live there);
# FV_BUILD_IMAGE (default: the pinned base image, BASE_IMAGE_TAG below);
# FV_BUILD_POD_ENV (JSON object merged into the pod env, e.g. a throwaway test
# pod's {"FV_BUILD_ROOT": "/workspace/fv-build-test"}); FV_BUILD_DC / FV_BUILD_VOLUME_GB
# for volume-create (default EU-RO-1 / 200); FV_BUILD_EVICT_HOURS (6) and
# FV_BUILD_EVICT_FREE_GB (40), sent to the pod at creation.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/lib.sh
source "$HERE/../gpu/lib.sh"

API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
STATE="${FV_BUILD_STATE:-${XDG_CONFIG_HOME:-$HOME/.config}/fv-build}"
POD_NAME="${FV_BUILD_POD_NAME:-fv-build}"
VOL_NAME="${FV_BUILD_VOLUME:-fv-build}"
FLAVORS="${FV_BUILD_FLAVORS:-cpu5c cpu3c}"
VCPUS="${FV_BUILD_VCPUS:-32}"
VCPUS_FALLBACK="${FV_BUILD_VCPUS_FALLBACK-16}"   # "" disables
# The base image (docker/build-base.Dockerfile): toolchains, CUDA, sccache, mold,
# ffmpeg, Node/Playwright/Chromium. The tag is a content hash of its inputs;
# `bash scripts/dev/build-base-tag.sh --pin` updates it, and the
# build-base-image workflow pushes the image and checks this pin.
BASE_IMAGE_TAG="bb-b49b814e45f9f7d2"
IMAGE="${FV_BUILD_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-build-base:$BASE_IMAGE_TAG}"
DISK_GB="${FV_BUILD_CONTAINER_GB:-200}"
MAX_DPH="${FV_BUILD_MAX_DPH:-1.5}"
MIN_BALANCE="${FV_MIN_BALANCE:-8}"
IDLE_MIN="${FV_BUILD_IDLE_MIN:-20}"
MAX_HOURS="${FV_BUILD_MAX_HOURS:-8}"
MAX_GRACE_MIN="${FV_BUILD_MAX_GRACE_MIN:-30}"
EVICT_HOURS="${FV_BUILD_EVICT_HOURS:-6}"
EVICT_FREE_GB="${FV_BUILD_EVICT_FREE_GB:-40}"
VOLUME_USD_GB_MONTH="0.07"
LEDGER="$STATE/ledger.tsv"
TOKEN_FILE="$STATE/token"
AUTH_FILE="$STATE/auth-header"

umask 077
mkdir -p "$STATE"
chmod 700 "$STATE"

ledger() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }
need_key() { : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"; }

# Runpod REST; the API key goes in through a header file, never argv.
rest() {
  local method="$1" path="$2" body="${3:-}"
  curl -sS --fail-with-body -X "$method" -H @<(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY") \
    -H 'content-type: application/json' ${body:+-d "$body"} "$API$path"
}
gql() {
  curl -sS -H @<(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY") -H 'content-type: application/json' \
    "$GQL" -d "$(jq -n --arg q "$1" '{query: $q}')"
}

balance() { gql '{ myself { clientBalance } }' | jq -r '.data.myself.clientBalance // empty'; }
check_balance() {
  local b
  b="$(balance)"
  [[ -n "$b" ]] || die "could not read the Runpod balance"
  awk -v b="$b" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+0)}' \
    || die "Runpod balance \$$b is below the floor \$$MIN_BALANCE"
  log "Runpod balance \$$(printf '%.2f' "$b") (floor \$$MIN_BALANCE)"
}

volume_json() { rest GET /networkvolumes | jq -c --arg n "$VOL_NAME" '[.[] | select(.name == $n)][0] // empty'; }
pod_json() {
  local id
  id="$(cat "$STATE/pod" 2>/dev/null || true)"
  if [[ -n "$id" ]]; then
    rest GET "/pods/$id" 2>/dev/null | jq -c 'select(.id != null)' && return 0
  fi
  rest GET /pods | jq -c --arg n "$POD_NAME" '[.[] | select(.name == $n)][0] // empty'
}

# FV_BUILD_URL points the service calls elsewhere (a local test server).
base_url() { echo "${FV_BUILD_URL:-https://$1-8000.proxy.runpod.net}"; }
pod_id() {
  local id
  [[ -n "${FV_BUILD_URL:-}" ]] && { echo local; return; }
  id="$(cat "$STATE/pod" 2>/dev/null || true)"
  [[ -n "$id" ]] || die "no build pod recorded in $STATE/pod; run: build-pod.sh up"
  echo "$id"
}
# Pod service call: $1 method, $2 path, rest = extra curl args. Token via header file.
svc() {
  local method="$1" path="$2"; shift 2
  [[ -s "$AUTH_FILE" ]] || die "no token in $STATE (the pod was created elsewhere?); run: build-pod.sh down && build-pod.sh up"
  curl -sS --fail-with-body --max-time "${FV_BUILD_HTTP_TIMEOUT:-90}" -X "$method" -H @"$AUTH_FILE" \
    "$@" "$(base_url "$(pod_id)")$path"
}

# svc, dying with the pod's error (e.g. 507 "build pod disk full: ...") on failure.
svc_or_die() {
  local what="$1" out rc=0
  shift
  out="$(svc "$@" 2>&1)" || rc=$?
  (( rc == 0 )) && return 0
  # The pod's JSON error body and curl's own message, in either order.
  die "$what failed: $(grep -m1 '^{' <<<"$out" | jq -r '.error // empty' 2>/dev/null || true) [$(grep -v '^{' <<<"$out" | head -c 200)]"
}

agent_name() {
  local a="$1"
  [[ "$a" == . ]] && a="$(basename "$FV_ROOT")"
  [[ "$a" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || die "bad agent name '$a'"
  echo "$a"
}

new_token() {
  local tok
  tok="$(openssl rand -hex 32)"
  printf '%s' "$tok" >"$TOKEN_FILE.new"
  printf 'Authorization: Bearer %s\n' "$tok" >"$AUTH_FILE.new"
  chmod 600 "$TOKEN_FILE.new" "$AUTH_FILE.new"
  printf '%s' "$tok" | sha256sum | cut -d' ' -f1
}
server_sha() { sha256sum "$HERE/build-pod-server.py" | cut -c1-12; }
commit_token() { mv -f "$TOKEN_FILE.new" "$TOKEN_FILE"; mv -f "$AUTH_FILE.new" "$AUTH_FILE"; }

start_cmd() {
  cat <<'EOF'
mkdir -p /opt/fvb
printf '%s' "$FV_BUILD_SERVER_B64" | base64 -d | gunzip >/opt/fvb/server.py
# Nothing is installed here: the base image (docker/build-base.Dockerfile) has
# python3, curl and every build tool. Fail loudly when it does not.
command -v python3 >/dev/null || { echo "fv-build: no python3 in image $FV_BUILD_IMAGE (not the build base image?)" >&2; sleep 600; exit 1; }
# Second wall-clock backstop, independent of the Python server (curl, its own
# process): 15 min after the server's hard stop (cap + grace), stop, else
# terminate, else GraphQL podTerminate; every 5 min until the pod is gone.
fvb_rp() { curl -sS --fail-with-body --max-time 30 -A fv-build-pod-backstop/1 -H @<(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY") -H 'content-type: application/json' "$@"; }
fvb_s=$(awk -v h="${FV_BUILD_MAX_HOURS:-8}" -v g="${FV_BUILD_MAX_GRACE_MIN:-30}" 'BEGIN{printf "%d", h*3600 + g*60 + 900}')
( sleep "$fvb_s"
  while :; do
    echo "[backstop] up ${fvb_s}s: stopping pod $RUNPOD_POD_ID" >&2
    fvb_rp -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
      || fvb_rp -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
      || fvb_rp https://api.runpod.io/graphql -d "$(printf '{"query":"mutation { podTerminate(input: {podId: \\"%s\\"}) }"}' "$RUNPOD_POD_ID")"
    echo "[backstop] exit $?" >&2
    sleep 300
  done ) &
while true; do python3 /opt/fvb/server.py; echo "server exited $?; restarting" >&2; sleep 5; done
EOF
}

# $1 volume id, $2 datacenter, $3 sha256 of the token -> PodCreateInput JSON
payload() {
  local vol="$1" dc="$2" hash="$3" flavors_json
  flavors_json="$(jq -nc --arg f "$FLAVORS" '$f | split(" ") | map(select(. != ""))')"
  jq -n --arg name "$POD_NAME" --arg image "$IMAGE" --arg vol "$vol" --arg dc "$dc" \
    --argjson flavors "$flavors_json" --arg vcpu "$VCPUS" --arg disk "$DISK_GB" --arg cmd "$(start_cmd)" \
    --arg hash "$hash" --arg srv "$(gzip -9c "$HERE/build-pod-server.py" | base64 -w0)" \
    --arg idle "$IDLE_MIN" --arg cap "$MAX_HOURS" --arg grace "$MAX_GRACE_MIN" \
    --arg evh "$EVICT_HOURS" --arg evf "$EVICT_FREE_GB" --argjson extra "${FV_BUILD_POD_ENV:-{\}}" '{
      name: $name, imageName: $image, computeType: "CPU", cloudType: "SECURE",
      cpuFlavorIds: $flavors, cpuFlavorPriority: "custom", vcpuCount: ($vcpu|tonumber),
      containerDiskInGb: ($disk|tonumber), volumeInGb: 0,
      networkVolumeId: $vol, volumeMountPath: "/workspace", dataCenterIds: [$dc],
      ports: ["8000/http"], dockerStartCmd: ["/bin/bash", "-c", $cmd],
      env: ({FV_BUILD_TOKEN_SHA256: $hash, FV_BUILD_SERVER_B64: $srv,
            FV_BUILD_IDLE_MIN: $idle, FV_BUILD_MAX_HOURS: $cap, FV_BUILD_MAX_GRACE_MIN: $grace,
            FV_BUILD_EVICT_HOURS: $evh, FV_BUILD_EVICT_FREE_GB: $evf, FV_BUILD_IMAGE: $image} + $extra)
    }'
}

create_pod() {
  local vol="$1" dc="$2" hash resp="" id dph v created=""
  hash="$(new_token)"
  log "create pod $POD_NAME: ${VCPUS} vCPU (fallback ${VCPUS_FALLBACK:-none}) [$FLAVORS] in $dc, image $IMAGE, ${DISK_GB} GB disk, volume $vol"
  # The volume pins the datacenter, and CPU stock there comes and goes (no
  # 32-vCPU cpu5c/cpu3c in EU-RO-1 for 15+ min on 2026-09-28 while 16 had
  # stock): each round tries VCPUS, then the fallback, for a while.
  local t0=$SECONDS
  while [[ -z "$created" ]]; do
    for v in $VCPUS $VCPUS_FALLBACK; do
      if resp="$(VCPUS=$v payload "$vol" "$dc" "$hash" | rest POST /pods @- 2>&1)"; then
        created=$v
        break
      fi
      grep -q "no longer any instances available" <<<"$resp" || die "pod create failed: $(head -c 400 <<<"$resp")"
    done
    [[ -n "$created" ]] && break
    (( SECONDS - t0 < ${FV_BUILD_STOCK_WAIT_S:-900} )) \
      || die "no ${VCPUS}/${VCPUS_FALLBACK:-} vCPU [$FLAVORS] stock in $dc for ${FV_BUILD_STOCK_WAIT_S:-900}s (try more FV_BUILD_FLAVORS)"
    log "no stock in $dc; retrying in 30s"
    sleep 30
  done
  [[ "$created" == "$VCPUS" ]] || log "no ${VCPUS}-vCPU stock; took ${created} vCPU"
  id="$(jq -r '.id // empty' <<<"$resp")"
  [[ -n "$id" ]] || die "pod create returned no id: $(head -c 400 <<<"$resp")"
  dph="$(jq -r '.costPerHr // 0' <<<"$resp")"
  ledger "pod-created $id flavor=$(jq -r '.machine.cpuFlavorId // .cpuFlavorId // "?"' <<<"$resp") vcpu=$created usd_per_hr=$dph dc=$dc"
  if awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 > c+0)}'; then
    rest DELETE "/pods/$id" >/dev/null || true
    ledger "pod-deleted $id over-cap"
    die "pod $id at \$$dph/hr exceeds FV_BUILD_MAX_DPH=\$$MAX_DPH; deleted"
  fi
  commit_token
  echo "$id" >"$STATE/pod"
  server_sha >"$STATE/pod-server"
  echo "$IMAGE" >"$STATE/pod-image"
  log "pod $id at \$$dph/hr (idle stop ${IDLE_MIN} min, cap ${MAX_HOURS} h + ${MAX_GRACE_MIN} min grace, pod-side)"
}

# $2: epoch the pod was (re)started at. Right after a start the proxy can
# still answer from the stopped container ("ready after 0s", then 502s), so
# only a server that booted after that counts.
wait_ready() {
  local id="$1" since="${2:-0}" t0 h phase="" last="" boot
  t0=$(date +%s)
  while :; do
    h="$(curl -sS --max-time 15 "$(base_url "$id")/healthz" 2>/dev/null || true)"
    phase="$(jq -r '.phase // empty' <<<"$h" 2>/dev/null || true)"
    boot="$(jq -r '.boot // 0' <<<"$h" 2>/dev/null || echo 0)"
    if [[ -n "$phase" ]] && (( ${boot%.*} > 0 && ${boot%.*} < since - 30 )); then
      phase=""  # the previous container
    fi
    [[ -n "$phase" && "$phase" != "$last" ]] && { log "pod setup: $phase"; last="$phase"; }
    [[ "$phase" == ready ]] && break
    [[ "$phase" == failed ]] && { svc GET /v1/status | jq -c .setup >&2; die "pod setup failed"; }
    (( $(date +%s) - t0 < ${FV_BUILD_BOOT_WAIT_S:-1500} )) || die "pod $id not ready after ${FV_BUILD_BOOT_WAIT_S:-1500}s (last: ${phase:-no answer})"
    sleep 10
  done
  log "ready after $(( $(date +%s) - t0 ))s: $(base_url "$id")"
}

cmd_up() {
  need_key
  require_tools curl jq openssl gzip base64 sha256sum
  check_balance
  local pod id status vol dc since=0
  pod="$(pod_json)"
  if [[ -n "$pod" ]]; then
    id="$(jq -r .id <<<"$pod")"
    status="$(jq -r .desiredStatus <<<"$pod")"
    echo "$id" >"$STATE/pod"
    case "$status" in
      RUNNING)
        [[ -s "$AUTH_FILE" ]] || die "pod $id is running but this container has no token; run: build-pod.sh down && build-pod.sh up"
        log "reusing running pod $id" ;;
      EXITED)
        # A stopped pod restarts with the server it was created with (it is
        # sent in the pod's env); recreate it when this checkout's differs.
        # Nothing is lost: the container disk does not survive a stop anyway.
        if [[ "$(cat "$STATE/pod-server" 2>/dev/null)" != "$(server_sha)" || "$(jq -r '.imageName // ""' <<<"$pod")" != "$IMAGE" ]]; then
          log "stopped pod $id runs an older server or image; terminating it and creating a new pod"
          rest DELETE "/pods/$id" >/dev/null || true
          ledger "pod-deleted $id server-update"
          pod=""
        elif log "starting stopped pod $id" && since=$(date +%s) && rest POST "/pods/$id/start" >/dev/null 2>&1; then
          ledger "pod-started $id"
        else
          log "start refused; terminating $id and creating a new pod"
          rest DELETE "/pods/$id" >/dev/null || true
          ledger "pod-deleted $id start-refused"
          pod=""
        fi ;;
      *) log "pod $id is $status" ;;
    esac
  fi
  if [[ -z "$pod" ]]; then
    vol="$(volume_json)"
    [[ -n "$vol" ]] || die "no network volume named $VOL_NAME; run: build-pod.sh volume-create"
    dc="$(jq -r .dataCenterId <<<"$vol")"
    since=$(date +%s)
    create_pod "$(jq -r .id <<<"$vol")" "$dc"
    id="$(cat "$STATE/pod")"
  fi
  wait_ready "$id" "$since"
  local srv_local srv_pod
  srv_local="$(server_sha)"
  srv_pod="$(svc GET /v1/status | jq -r '.server_sha // ""')"
  [[ "$srv_pod" == "$srv_local" ]] || log "note: pod runs server $srv_pod, this checkout has $srv_local (down + up to update)"
  [[ "$(jq -r '.imageName // ""' <<<"$(pod_json)")" == "$IMAGE" ]] \
    || log "note: pod runs image $(jq -r '.imageName // "?"' <<<"$(pod_json)"), this checkout pins $IMAGE (down + up to update)"
}

cmd_status() {
  need_key
  local pod vol
  pod="$(pod_json)"
  vol="$(volume_json)"
  if [[ -n "$vol" ]]; then
    jq -r --arg p "$VOLUME_USD_GB_MONTH" '"volume \(.id) \(.name) \(.size) GB in \(.dataCenterId)  ~$\((.size * ($p|tonumber)) * 100 | round / 100)/month"' <<<"$vol"
  else
    echo "volume $VOL_NAME: none"
  fi
  if [[ -z "$pod" ]]; then echo "pod $POD_NAME: none"; return 0; fi
  jq -r '"pod \(.id) \(.name) \(.desiredStatus)  $\(.costPerHr)/hr  vcpu=\(.vcpuCount // "?") mem=\(.memoryInGb // "?")GB"' <<<"$pod"
  if [[ "$(jq -r .desiredStatus <<<"$pod")" == RUNNING ]]; then
    local st
    st="$(svc GET /v1/status)" || return 0
    jq 'del(.agents)' <<<"$st"
    stop_timers <<<"$st"
    cache_summary <<<"$st"
    # A server older than eviction has no agent rows.
    if jq -e '.eviction' <<<"$st" >/dev/null; then
      jq -r '"agents on the container disk (\(.local_disk.free_gb // "?") GB free; evict after \(.eviction.idle_hours) h unused, LRU targets below \(.eviction.free_gb_floor) GB free; \(.eviction.protect // [] | join(" ")) held \(.eviction.hold_min // 0) min after use):",
        (["agent", "idle_h", "busy", "held", "target_gb", "worktree_gb", "evict_in_h"] | @tsv),
        (.agents // [] | .[] | [.agent, .idle_h, .busy, (.held // false),
           (if .target then (.target_gb // "?") else "-" end),
           (if .worktree then (.worktree_gb // "?") else "-" end), .evict_in_h] | @tsv)' <<<"$st" | tsv_table
    fi
  fi
}

# One line from /v1/status: idle time and when the pod stops itself. Servers
# before 2026-10-02 lack the *_in_s fields; derive them there.
# Aligned columns from TSV on stdin (`column` is not in every container).
tsv_table() {
  if command -v column >/dev/null; then column -t -s $'\t'; else tr '\t' ' '; fi
}

# Image, sccache hit rate, cache sizes and deps seeds, one line each.
cache_summary() {
  jq -r 'def gb: if . == null then "?" else "\(.) GB" end;
    "image: \(.image // "?")  (\(.rustc // "rustc ?"))",
    (if (.sccache | type) == "object" then
       "sccache: \(.sccache.hits) hits / \(.sccache.hits + .sccache.misses) cacheable compiles"
       + " (hit rate \(if .sccache.hit_rate == null then "-" else "\(.sccache.hit_rate * 100 | round) %" end)),"
       + " \(.sccache.not_cacheable // 0) not cacheable, \(.sccache.errors // 0) errors since boot;"
       + " cache \(.sccache.cache_size_gb | gb) of \(.sccache.max_cache_size_gb | gb)"
     else "sccache: NOT RUNNING (\(.sccache))" end),
    "caches: " + ([(.caches_gb // {}) | to_entries[] | select(.key != "t") | "\(.key) \(.value | gb)"] | join(", ")),
    "deps seeds (\(if .deps_seeds.enabled then "on" else "off" end), keep \(.deps_seeds.keep)): "
      + ([.deps_seeds.seeds[]? | "\(.key) \(.gb) GB (\(.unpacked_gb) GB unpacked, used \(.last_used_h_ago) h ago)"] | join("; "))
      + (if (.deps_seeds.building // []) | length > 0 then "; building: \(.deps_seeds.building | map(.argv[1]) | join(" "))" else "" end)' 2>/dev/null || true
}

stop_timers() {
  jq -r 'def mins: if . == null then "-" else "\((. / 60) | floor) min" end;
    (.jobs_active | length) as $n
    | (.idle_stop_in_s // (if $n > 0 then null else ([.idle_stop_s - .idle_s, 0] | max) end)) as $idle_in
    | (.max_stop_in_s // ([.max_s - .uptime_s, 0] | max)) as $max_in
    | "self-stop: up \(.uptime_s | mins), idle \(.idle_s | mins), \($n) job(s) active;"
      + " idle stop in \(if $idle_in == null then "- (jobs active)" else ($idle_in | mins) end),"
      + " cap stop in \($max_in | mins)"
      + (if (.self_stop.error // null) != null then "; LAST STOP ATTEMPT FAILED: \(.self_stop.error)"
         elif (.self_stop.ok // null) != null then "; stop accepted (\(.self_stop.ok))" else "" end)'
}

# Mirror tracked + untracked (not ignored) files of this worktree to the pod.
cmd_sync() {
  local agent tmp n_changed n_deleted t0
  agent="$(agent_name "${1:?agent}")"
  t0=$(date +%s)
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/fv-build-sync.XXXXXX")"
  # No RETURN trap: it is global, not function-local, and depending on the
  # bash version it fired again after this function (in cmd_run, or on the
  # script's last return) with the local $tmp gone, so set -u reported
  # "tmp: unbound variable" after successful runs. A global path plus an EXIT
  # trap covers die(); the normal path removes the directory itself.
  FV_SYNC_TMP="$tmp"
  trap 'rm -rf "${FV_SYNC_TMP:-}"; fv_rel_cleanup' EXIT
  svc GET "/v1/agents/$agent/manifest" >"$tmp/remote"
  {
    git -C "$FV_ROOT" ls-files -z --recurse-submodules
    git -C "$FV_ROOT" ls-files -z -o --exclude-standard
  } >"$tmp/files"
  python3 - "$FV_ROOT" "$tmp/files" "$tmp/remote" "$tmp/changed" "$tmp/deleted" <<'PY'
import os, sys
root, files, remote, changed, deleted = sys.argv[1:]
rem = {}
for line in open(remote, encoding="utf-8", errors="surrogateescape"):
    p, s, m = line.rstrip("\n").rsplit("\t", 2)
    rem[p] = (int(s), int(m))
local = {}
for p in open(files, "rb").read().decode("utf-8", "surrogateescape").split("\0"):
    if not p or p in local:
        continue
    try:
        st = os.lstat(os.path.join(root, p))
    except FileNotFoundError:
        continue
    if os.path.isdir(os.path.join(root, p)) and not os.path.islink(os.path.join(root, p)):
        continue
    local[p] = (st.st_size, int(st.st_mtime))
with open(changed, "wb") as f:
    f.write(b"".join(p.encode("utf-8", "surrogateescape") + b"\0" for p, v in sorted(local.items()) if rem.get(p) != v))
with open(deleted, "wb") as f:
    f.write(b"".join(p.encode("utf-8", "surrogateescape") + b"\0" for p in sorted(rem) if p not in local))
PY
  n_changed="$(tr -cd '\0' <"$tmp/changed" | wc -c)"
  n_deleted="$(tr -cd '\0' <"$tmp/deleted" | wc -c)"
  if (( n_changed > 0 )); then
    tar -C "$FV_ROOT" --null -T "$tmp/changed" --format=gnu -cf - | gzip -1 >"$tmp/upload.tgz"
    # A first full snapshot (~1400 files, 29 MB) takes about a minute to
    # extract onto the network volume; allow far more than the default 90 s.
    FV_BUILD_HTTP_TIMEOUT="${FV_BUILD_SYNC_TIMEOUT:-900}" svc_or_die "sync $agent" PUT "/v1/agents/$agent/files" \
      --data-binary @"$tmp/upload.tgz" -H 'content-type: application/gzip'
  fi
  if (( n_deleted > 0 )); then
    svc_or_die "sync $agent (deletions)" POST "/v1/agents/$agent/delete" --data-binary @"$tmp/deleted"
  fi
  rm -rf "$tmp"
  FV_SYNC_TMP=""
  log "sync $agent: $n_changed changed, $n_deleted deleted ($(( $(date +%s) - t0 ))s)"
}

follow() {
  local job="$1" off=0 resp fails=0 fin rc
  while :; do
    if ! resp="$(svc GET "/v1/jobs/$job?offset=$off&wait=20" 2>/dev/null)"; then
      fails=$((fails + 1))
      (( fails < 20 )) || die "lost the pod while following job $job (re-attach: build-pod.sh log $job)"
      sleep 5
      continue
    fi
    fails=0
    jq -j '.data' <<<"$resp"
    off="$(jq -r '.next' <<<"$resp")"
    fin="$(jq -r '.finished' <<<"$resp")"
    if [[ "$fin" == true ]]; then
      rc="$(jq -r '.exit // 1' <<<"$resp")"
      return "$rc"
    fi
  done
}

cmd_run() {
  local agent sync=1 env_json='{}' argv_json resp job
  agent="$(agent_name "${1:?agent}")"; shift
  [[ "${1:-}" == --no-sync ]] && { sync=0; shift; }
  [[ "${1:-}" == -- ]] && shift
  (( $# > 0 )) || die "usage: build-pod.sh run <agent> [--no-sync] -- [K=V ...] <cmd...>"
  while [[ "${1:-}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; do
    env_json="$(jq -c --arg k "${1%%=*}" --arg v "${1#*=}" '. + {($k): $v}' <<<"$env_json")"
    shift
  done
  argv_json="$(printf '%s\n' "$@" | jq -R . | jq -sc .)"
  (( sync )) && cmd_sync "$agent"
  resp="$(svc POST "/v1/agents/$agent/jobs" -H 'content-type: application/json' \
    -d "$(jq -nc --argjson a "$argv_json" --argjson e "$env_json" '{argv: $a, env: $e}')" 2>/dev/null)" \
    || die "rejected: $(jq -r '.error // .' <<<"$resp" 2>/dev/null || echo "$resp")"
  job="$(jq -r .id <<<"$resp")"
  log "job $job ($agent): $*"
  trap 'svc POST "/v1/jobs/'"$job"'/cancel" >/dev/null 2>&1; log "cancelled job '"$job"'"; exit 130' INT TERM
  local rc=0
  follow "$job" || rc=$?
  trap - INT TERM
  return "$rc"
}

cmd_fetch() {
  local agent path dest t0
  agent="$(agent_name "${1:?agent}")"; path="${2:?path under target/<agent>, e.g. release/fv-serve}"
  dest="${3:-$FV_ROOT/artifacts/build-pod/$agent/$path}"
  mkdir -p "$(dirname "$dest")"
  t0=$(date +%s)
  # Download first, so a 404 shows the pod's error instead of a gunzip one.
  if ! FV_BUILD_HTTP_TIMEOUT=900 svc GET "/v1/agents/$agent/artifact?path=$(jq -rn --arg p "$path" '$p|@uri')&gz=1" \
    -o "$dest.gz.part"; then
    local err
    err="$(jq -r '.error // empty' "$dest.gz.part" 2>/dev/null || true)"
    rm -f "$dest.gz.part"
    die "fetch $path failed${err:+: $err}"
  fi
  gunzip <"$dest.gz.part" >"$dest.part"
  rm -f "$dest.gz.part"
  mv -f "$dest.part" "$dest"
  chmod +x "$dest" 2>/dev/null || true
  log "fetched $path → $dest ($(du -h "$dest" | cut -f1), $(( $(date +%s) - t0 ))s)"
}

cmd_stop() {
  need_key
  local id
  id="$(pod_id)"
  if rest POST "/pods/$id/stop" >/dev/null 2>&1; then
    ledger "pod-stopped $id"; log "stopped $id (volume kept)"
  else
    log "stop refused (network-volume pods may only terminate); terminating $id"
    rest DELETE "/pods/$id" >/dev/null && ledger "pod-deleted $id stop-refused" && rm -f "$STATE/pod"
    log "terminated $id (volume kept)"
  fi
}

cmd_down() {
  need_key
  local id
  id="$(pod_id)"
  rest DELETE "/pods/$id" >/dev/null
  ledger "pod-deleted $id"
  rm -f "$STATE/pod"
  log "terminated $id (volume kept)"
}

# ---- release artifacts (docs/dev/build-pod.md "Release artifacts") ----------
# Build one commit's release binaries on the pod (scripts/dev/release-artifacts-pod.sh),
# fetch the tarballs + manifest.json, check their sha256s and upload them to
# R2 under artifacts/<sha>/ (manifest.json last: its presence means complete).
REL_AGENT="${FV_RELEASE_AGENT:-fv-release}"
R2_ENV_FILE="${FV_R2_ARTIFACTS_ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/fv/r2-build-artifacts-rw.env}"
# The temporary worktree of a release build; also called from cmd_sync's EXIT trap.
FV_REL_WT=""
fv_rel_cleanup() {
  [[ -n "$FV_REL_WT" ]] || return 0
  git -C "$FV_ROOT" worktree remove --force "$FV_REL_WT" >/dev/null 2>&1 || rm -rf "$FV_REL_WT"
}
r2() { FV_R2_ARTIFACTS_ENV_FILE="$R2_ENV_FILE" python3 "$HERE/r2.py" "$@"; }

cmd_release_artifacts() {
  local rev="" sets="" force=0 upload=1 keep=0
  while (( $# )); do
    case "$1" in
      --sets) sets="${2:?--sets needs a list}"; shift 2 ;;
      --force) force=1; shift ;;
      --no-upload) upload=0; shift ;;
      --keep) keep=1; shift ;;
      -*) die "unknown flag $1" ;;
      *) [[ -z "$rev" ]] || die "one revision only"; rev="$1"; shift ;;
    esac
  done
  [[ -n "$rev" ]] || die "usage: build-pod.sh release-artifacts <sha|ref> [--sets \"a b\"] [--force] [--no-upload] [--keep]"
  require_tools git jq python3 sha256sum
  local sha
  if ! sha="$(git -C "$FV_ROOT" rev-parse -q --verify "$rev^{commit}")"; then
    git -C "$FV_ROOT" fetch -q origin || true
    sha="$(git -C "$FV_ROOT" rev-parse -q --verify "$rev^{commit}" || git -C "$FV_ROOT" rev-parse -q --verify "origin/$rev^{commit}")" \
      || die "unknown revision $rev"
  fi
  if (( upload )); then
    [[ -s "$R2_ENV_FILE" ]] || die "no R2 credentials in $R2_ENV_FILE (FV_R2_ARTIFACTS_*; docs/dev/build-pod.md \"Release artifacts\"); --no-upload builds without uploading"
    local rc=0
    r2 head "artifacts/$sha/manifest.json" || rc=$?
    case "$rc" in
      0) if (( !force )); then log "artifacts/$sha already in R2 (--force rebuilds)"; return 0; fi ;;
      1) ;;
      *) die "cannot read the R2 bucket with $R2_ENV_FILE (r2.py exit $rc)" ;;
    esac
  fi
  # One release build per container at a time: they share the pod's
  # $REL_AGENT snapshot and target dir.
  exec 9>"$STATE/release.lock"
  flock -w "${FV_RELEASE_LOCK_WAIT_S:-3600}" 9 || die "another release-artifacts run holds $STATE/release.lock"

  local wt="${TMPDIR:-/tmp}/fv-release-${sha:0:12}" out="${FV_RELEASE_OUT:-$FV_ROOT/artifacts/release/$sha}"
  local build_id build_time run_id t0=$SECONDS
  git -C "$FV_ROOT" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"
  git -C "$FV_ROOT" worktree add -q --detach "$wt" "$sha"
  FV_REL_WT="$wt"
  trap fv_rel_cleanup EXIT
  # The image workflows check out submodules too; only cutile-rs is built.
  git -C "$wt" submodule update -q --init third_party/cutile-rs
  build_id="$(bash "$wt/scripts/gpu/docker.sh" build-id)"
  build_time="$(cd "$wt" && TZ=UTC git log -1 --format=%cd --date=format-local:%Y-%m-%dT%H:%M:%SZ)"
  run_id="$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 3)"
  # The recipe comes from this checkout (it may postdate <sha>); its hash is
  # in the manifest (builder.recipe_sha256).
  install -m 755 "$HERE/release-artifacts-pod.sh" "$wt/scripts/dev/release-artifacts-pod.sh"
  log "release artifacts for $sha (build id $build_id, run $run_id)"

  cmd_up
  FV_ROOT="$wt" cmd_sync "$REL_AGENT"
  local env=(FV_REL_SHA="$sha" FV_GIT_SHA="$sha" FV_BUILD_TIME="$build_time" FV_BUILD_ID="$build_id" FV_REL_RUN_ID="$run_id")
  [[ -n "$sets" ]] && env+=(FV_REL_SETS="$sets")
  cmd_run "$REL_AGENT" --no-sync -- "${env[@]}" bash scripts/dev/release-artifacts-pod.sh \
    || die "pod build failed (log above; re-attach with build-pod.sh log <job>)"

  rm -rf "$out" && mkdir -p "$out"
  cmd_fetch "$REL_AGENT" "release-artifacts/$sha/manifest.json" "$out/manifest.json"
  chmod -x "$out/manifest.json"
  [[ "$(jq -r .sha "$out/manifest.json")" == "$sha" ]] || die "manifest is for another sha"
  local tb want
  while IFS=$'\t' read -r _ tb want; do
    cmd_fetch "$REL_AGENT" "release-artifacts/$sha/$tb" "$out/$tb"
    chmod -x "$out/$tb"
    [[ "$(sha256sum "$out/$tb" | cut -d' ' -f1)" == "$want" ]] || die "$tb: sha256 differs from the manifest"
  done < <(jq -r '.sets | to_entries[] | [.key, .value.tarball, .value.sha256] | @tsv' "$out/manifest.json")
  log "fetched and verified $(jq '.sets | length' "$out/manifest.json") sets ($(du -sh "$out" | cut -f1)) into $out"

  if (( upload )); then
    while IFS=$'\t' read -r _ tb; do
      r2 put "artifacts/$sha/$tb" "$out/$tb"
    done < <(jq -r '.sets | to_entries[] | [.key, .value.tarball] | @tsv' "$out/manifest.json")
    r2 put "artifacts/$sha/manifest.json" "$out/manifest.json"
    log "uploaded to R2: artifacts/$sha/ ($(( SECONDS - t0 ))s in all)"
    (( keep )) || rm -rf "$out"
  fi
}

cmd_volume_create() {
  need_key
  [[ -z "$(volume_json)" ]] || die "volume $VOL_NAME already exists"
  local resp
  resp="$(rest POST /networkvolumes "$(jq -nc --arg n "$VOL_NAME" --arg dc "${FV_BUILD_DC:-EU-RO-1}" \
    --arg s "${FV_BUILD_VOLUME_GB:-200}" '{name: $n, size: ($s|tonumber), dataCenterId: $dc}')")"
  ledger "volume-created $(jq -r .id <<<"$resp") size=${FV_BUILD_VOLUME_GB:-200} dc=${FV_BUILD_DC:-EU-RO-1}"
  jq -c '{id, name, size, dataCenterId}' <<<"$resp"
}

case "${1:-}" in
  up) cmd_up ;;
  status) cmd_status ;;
  agents) shift; svc GET "/v1/agents$([[ "${1:-}" == --sizes ]] && echo '?sizes=1')" | jq . ;;
  sync) shift; cmd_sync "$@" ;;
  run) shift; cmd_run "$@" ;;
  log) follow "${2:?job id}" ;;
  cancel) svc POST "/v1/jobs/${2:?job id}/cancel" | jq -c '{id, agent, state}' ;;
  fetch) shift; cmd_fetch "$@" ;;
  clean) svc POST "/v1/agents/$(agent_name "${2:?agent}")/clean" -d "{\"what\":\"${3:-all}\"}" | jq -c . ;;
  seed) a="$(agent_name "${2:?agent}")"; cmd_sync "$a"
    svc POST "/v1/agents/$a/seed" -d "{\"force\":$([[ "${3:-}" == --force ]] && echo true || echo false)}" | jq -c . ;;
  evict) FV_BUILD_HTTP_TIMEOUT=900 svc POST /v1/evict | jq -c '.evicted[]' ;;
  stop) cmd_stop ;;
  down) cmd_down ;;
  release-artifacts) shift; cmd_release_artifacts "$@" ;;
  volume-create) cmd_volume_create ;;
  plan) payload "<volume id>" "${FV_BUILD_DC:-EU-RO-1}" "<sha256 of the pod token>" \
    | jq '.env.FV_BUILD_SERVER_B64 |= "<\(length) bytes: gzip+base64 of build-pod-server.py>"' ;;
  # The header comment (line 2 up to `set -euo pipefail`) is the usage.
  *) sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
