#!/usr/bin/env bash
# Shared CPU build pod (docs/dev/build-pod.md): one Runpod CPU pod plus the
# `fv-build` network volume, driven over the pod's HTTPS proxy (no SSH).
# Each agent gets its own worktree snapshot and CARGO_TARGET_DIR on the volume.
#
#   build-pod.sh up                  reuse the running pod, start the stopped one,
#                                    or create it; wait until the toolchain is ready
#   build-pod.sh status              pod, $/hr, setup state, jobs, disk
#   build-pod.sh agents [--sizes]    agent dirs on the volume
#   build-pod.sh sync <agent>        mirror this worktree (tracked + untracked,
#                                    not ignored) to worktrees/<agent>/
#   build-pod.sh run <agent> [--no-sync] -- [K=V ...] <cmd...>
#                                    sync, then run an allowlisted command
#                                    (cargo check|build|test|clippy|fmt|doc|tree|
#                                    metadata, bash scripts/serve/check.sh,
#                                    bash scripts/gpu/lint.sh); streams the log
#                                    and exits with the command's status
#   build-pod.sh log <job>           re-attach to a job's log
#   build-pod.sh fetch <agent> <path> [dest]
#                                    copy target/<agent>/<path> back (gzip in
#                                    transit), e.g. release/fv-serve
#   build-pod.sh clean <agent> [target|worktree|all]   (default all)
#   build-pod.sh stop                stop the pod (terminate if Runpod refuses a
#                                    stop); the volume and its caches persist
#   build-pod.sh down                terminate the pod (volume persists)
#   build-pod.sh volume-create       create the fv-build volume (once)
#   build-pod.sh plan                print the pod create payload (no API call)
#
# <agent> is any [A-Za-z0-9._-] name; "." means this worktree's directory name.
#
# Money guards: balance floor FV_MIN_BALANCE (default 8 $) checked by `up`;
# $/hr cap FV_BUILD_MAX_DPH (default 1.5); the pod stops itself after
# FV_BUILD_IDLE_MIN (default 20) idle minutes and FV_BUILD_MAX_HOURS (default
# 8) after boot, enforced on the pod so it survives this container; ledger at
# $FV_BUILD_STATE/ledger.tsv.
#
# Env: RUNPOD_API_KEY; FV_BUILD_STATE (default ~/.config/fv-build: token,
# mode 600, never printed); FV_BUILD_VOLUME (default fv-build);
# FV_BUILD_FLAVORS (default "cpu5c cpu3c"); FV_BUILD_VCPUS (default 32);
# FV_BUILD_IMAGE (default rust:1-bookworm); FV_BUILD_DC / FV_BUILD_VOLUME_GB
# for volume-create (default EU-RO-1 / 200).
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
IMAGE="${FV_BUILD_IMAGE:-rust:1-bookworm}"
DISK_GB="${FV_BUILD_CONTAINER_GB:-40}"
MAX_DPH="${FV_BUILD_MAX_DPH:-1.5}"
MIN_BALANCE="${FV_MIN_BALANCE:-8}"
IDLE_MIN="${FV_BUILD_IDLE_MIN:-20}"
MAX_HOURS="${FV_BUILD_MAX_HOURS:-8}"
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
commit_token() { mv -f "$TOKEN_FILE.new" "$TOKEN_FILE"; mv -f "$AUTH_FILE.new" "$AUTH_FILE"; }

start_cmd() {
  cat <<'EOF'
mkdir -p /opt/fvb
printf '%s' "$FV_BUILD_SERVER_B64" | base64 -d | gunzip >/opt/fvb/server.py
command -v python3 >/dev/null || { apt-get update -qq && apt-get install -y -qq python3; }
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
    --arg idle "$IDLE_MIN" --arg cap "$MAX_HOURS" '{
      name: $name, imageName: $image, computeType: "CPU", cloudType: "SECURE",
      cpuFlavorIds: $flavors, cpuFlavorPriority: "custom", vcpuCount: ($vcpu|tonumber),
      containerDiskInGb: ($disk|tonumber), volumeInGb: 0,
      networkVolumeId: $vol, volumeMountPath: "/workspace", dataCenterIds: [$dc],
      ports: ["8000/http"], dockerStartCmd: ["/bin/bash", "-c", $cmd],
      env: {FV_BUILD_TOKEN_SHA256: $hash, FV_BUILD_SERVER_B64: $srv,
            FV_BUILD_IDLE_MIN: $idle, FV_BUILD_MAX_HOURS: $cap}
    }'
}

create_pod() {
  local vol="$1" dc="$2" payload resp id dph
  payload="$(payload "$vol" "$dc" "$(new_token)")"
  log "create pod $POD_NAME: ${VCPUS} vCPU [$FLAVORS] in $dc, image $IMAGE, volume $vol"
  resp="$(rest POST /pods "$payload" 2>&1)" || die "pod create failed: $(head -c 400 <<<"$resp")"
  id="$(jq -r '.id // empty' <<<"$resp")"
  [[ -n "$id" ]] || die "pod create returned no id: $(head -c 400 <<<"$resp")"
  dph="$(jq -r '.costPerHr // 0' <<<"$resp")"
  ledger "pod-created $id flavor=$(jq -r '.machine.cpuFlavorId // .cpuFlavorId // "?"' <<<"$resp") vcpu=$VCPUS usd_per_hr=$dph dc=$dc"
  if awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 > c+0)}'; then
    rest DELETE "/pods/$id" >/dev/null || true
    ledger "pod-deleted $id over-cap"
    die "pod $id at \$$dph/hr exceeds FV_BUILD_MAX_DPH=\$$MAX_DPH; deleted"
  fi
  commit_token
  echo "$id" >"$STATE/pod"
  log "pod $id at \$$dph/hr (idle stop ${IDLE_MIN} min, cap ${MAX_HOURS} h, pod-side)"
}

wait_ready() {
  local id="$1" t0 h phase="" last=""
  t0=$(date +%s)
  while :; do
    h="$(curl -sS --max-time 15 "$(base_url "$id")/healthz" 2>/dev/null || true)"
    phase="$(jq -r '.phase // empty' <<<"$h" 2>/dev/null || true)"
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
  local pod id status vol dc
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
        log "starting stopped pod $id"
        if rest POST "/pods/$id/start" >/dev/null 2>&1; then
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
    create_pod "$(jq -r .id <<<"$vol")" "$dc"
    id="$(cat "$STATE/pod")"
  fi
  wait_ready "$id"
  local srv_local srv_pod
  srv_local="$(sha256sum "$HERE/build-pod-server.py" | cut -c1-12)"
  srv_pod="$(svc GET /v1/status | jq -r '.server_sha // ""')"
  [[ "$srv_pod" == "$srv_local" ]] || log "note: pod runs server $srv_pod, this checkout has $srv_local (down + up to update)"
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
    svc GET /v1/status | jq . || true
  fi
}

# Mirror tracked + untracked (not ignored) files of this worktree to the pod.
cmd_sync() {
  local agent tmp n_changed n_deleted t0
  agent="$(agent_name "${1:?agent}")"
  t0=$(date +%s)
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/fv-build-sync.XXXXXX")"
  trap 'rm -rf "$tmp"' RETURN
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
    svc PUT "/v1/agents/$agent/files" --data-binary @"$tmp/upload.tgz" -H 'content-type: application/gzip' >/dev/null
  fi
  if (( n_deleted > 0 )); then
    svc POST "/v1/agents/$agent/delete" --data-binary @"$tmp/deleted" >/dev/null
  fi
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
  FV_BUILD_HTTP_TIMEOUT=900 svc GET "/v1/agents/$agent/artifact?path=$(jq -rn --arg p "$path" '$p|@uri')&gz=1" \
    | gunzip >"$dest.part"
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
  fetch) shift; cmd_fetch "$@" ;;
  clean) svc POST "/v1/agents/$(agent_name "${2:?agent}")/clean" -d "{\"what\":\"${3:-all}\"}" | jq -c . ;;
  stop) cmd_stop ;;
  down) cmd_down ;;
  volume-create) cmd_volume_create ;;
  plan) payload "<volume id>" "${FV_BUILD_DC:-EU-RO-1}" "<sha256 of the pod token>" \
    | jq '.env.FV_BUILD_SERVER_B64 |= "<\(length) bytes: gzip+base64 of build-pod-server.py>"' ;;
  *) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
