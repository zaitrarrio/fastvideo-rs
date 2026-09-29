#!/usr/bin/env bash
# WP-18 GPU E2E pod (docs/serve/design.md §7.6): one Runpod pod running the
# serve image with scripts/serve/e2e/pod-boot.sh as its start command, i.e.
# fv-serve on :8000 plus the E2E sidecar on :8001 (webhook receiver, test
# bundle upload, command runner for the on-pod streaming clients).
#
#   pod.sh up <image> [config-in-image]   create; writes the state file
#   pod.sh down                           delete the pod in the state file
#   pod.sh wait                           wait for /ping 200 (prints timings)
#   pod.sh bundle                         upload the test bundle to /e2e
#   pod.sh run '<shell>' [timeout_s]      run on the pod via the sidecar, print output
#   pod.sh hooks                          the sidecar's recorded webhook deliveries
#
# State (pod id, API key, admin token, sidecar token) lives in
# $FV_E2E_STATE (mode 600, never printed). Guards: balance floor
# (FV_MIN_BALANCE, 8 $), $/hr cap (RUNPOD_GPU_MAX_DPH, 2.2), a detached
# wall-clock backstop (FV_POD_CAP_S, max 5400 s) that deletes the pod.
# Env: RUNPOD_API_KEY, RUNPOD_GPU_TYPES (comma list), RUNPOD_VOLUME_ID,
# FV_E2E_NAME (pod name prefix, default fv-e2e-a-), FV_E2E_ENV_JSON (extra
# pod env as a JSON object, e.g. '{"FV_REACTOR_MODE":"avatar"}').
# `up <image> <config>`: the in-image config pod-boot.sh starts from
# (default /etc/fv/runpod.toml; e.g. /etc/fv/runpod-ltx.toml).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="https://api.runpod.io/graphql"
STATE="${FV_E2E_STATE:?FV_E2E_STATE (state file path) missing}"
CAP_S="${FV_POD_CAP_S:-5400}"
(( CAP_S <= 5400 )) || { echo "FV_POD_CAP_S must be <= 5400" >&2; exit 2; }
MAX_DPH="${RUNPOD_GPU_MAX_DPH:-2.2}"
MIN_BALANCE="${FV_MIN_BALANCE:-8}"
LEDGER="${FV_E2E_LEDGER:-$ROOT/artifacts/serve/e2e/ledger.tsv}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
rest() {
  curl -sS --fail-with-body -X "$1" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${3:+-d "$3"} "$API$2"
}
ledger() { mkdir -p "$(dirname "$LEDGER")"; printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }
st() { jq -r ".$1 // empty" "$STATE"; }
side() { curl -sS --max-time "${2:-60}" -H "X-Sidecar-Token: $(st sidecar)" "https://$(st pod)-8001.proxy.runpod.net$1" "${@:3}"; }

cmd_up() {
  local image="${1:?image digest}" cfg="${2:-/etc/fv/runpod.toml}" bal gpu resp pod dph name vol='{}'
  : "${RUNPOD_API_KEY:?}"
  [[ "$image" == *@sha256:* ]] || die "pin the image by digest"
  bal="$(curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' "$GQL" \
    -d '{"query":"{ myself { clientBalance } }"}' | jq -r '.data.myself.clientBalance')"
  awk -v b="$bal" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+0)}' || die "balance $bal below floor $MIN_BALANCE"
  log "balance \$$bal"
  local key admin sidecar
  key="fvk-$(openssl rand -hex 16)"; admin="fvadm_$(openssl rand -hex 24)"; sidecar="$(openssl rand -hex 24)"
  umask 077
  jq -n --arg key "$key" --arg admin "$admin" --arg sidecar "$sidecar" --arg image "$image" \
    '{key: $key, admin: $admin, sidecar: $sidecar, image: $image}' > "$STATE"
  if [[ -n "${RUNPOD_VOLUME_ID:-}" ]]; then
    local dc
    dc="$(rest GET "/networkvolumes/$RUNPOD_VOLUME_ID" | jq -r '.dataCenterId')"
    vol="$(jq -n --arg v "$RUNPOD_VOLUME_ID" --arg dc "$dc" '{networkVolumeId: $v, volumeMountPath: "/workspace", dataCenterIds: [$dc]}')"
  fi
  name="${FV_E2E_NAME:-fv-e2e-a-}$(date -u +%m%d%H%M%S)"
  IFS=',' read -r -a types <<<"${RUNPOD_GPU_TYPES:-NVIDIA RTX PRO 6000 Blackwell Server Edition}"
  for gpu in "${types[@]}"; do
    local payload
    payload="$(jq -n --arg name "$name" --arg image "$image" --arg gpu "$gpu" --arg cfg "$cfg" \
      --arg boot "$(cat "$ROOT/scripts/serve/e2e/pod-boot.sh")" \
      --arg side "$(base64 -w0 "$ROOT/scripts/serve/e2e/sidecar.py")" \
      --arg keyhash "$(printf '%s' "$key" | sha256sum | cut -d' ' -f1)" --arg admin "$admin" --arg sidecar "$sidecar" \
      --argjson vol "$vol" --argjson extra "${FV_E2E_ENV_JSON:-{\}}" '{
        name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU",
        gpuTypeIds: [$gpu], gpuCount: 1, containerDiskInGb: 40, volumeInGb: 0,
        ports: ["8000/http", "8001/http"],
        dockerEntrypoint: ["bash", "-c"], dockerStartCmd: [$boot],
        env: ({
          FV_CF_ACCOUNT_ID: "{{ RUNPOD_SECRET_fv_cf_account_id }}",
          FV_CF_API_TOKEN: "{{ RUNPOD_SECRET_fv_cf_api_token }}",
          FV_D1_DATABASE_ID: "{{ RUNPOD_SECRET_fv_d1_database_id }}",
          FV_R2_BUCKET: "{{ RUNPOD_SECRET_fv_r2_bucket }}",
          FV_R2_ENDPOINT: "{{ RUNPOD_SECRET_fv_r2_endpoint }}",
          FV_R2_ACCESS_KEY_ID: "{{ RUNPOD_SECRET_fv_r2_access_key_id }}",
          FV_R2_SECRET_ACCESS_KEY: "{{ RUNPOD_SECRET_fv_r2_secret_access_key }}",
          FV_WEBHOOK_ED25519_KEY: "{{ RUNPOD_SECRET_fv_webhook_ed25519_key }}",
          FV_SERVE_MODE: "http", FV_STATE_DIR: "/fvstate", FV_WEIGHTS: "/workspace/weights",
          FV_API_KEYS: $keyhash, FV_ADMIN_TOKEN: $admin, FV_SIDECAR_TOKEN: $sidecar, FV_SIDECAR_B64: $side,
          FV_CALLBACKS_ALLOW_PRIVATE: "1", RUST_LOG: "info", FV_E2E_BASE_CONFIG: $cfg
        } + $extra)
      } + $vol')"
    if ! resp="$(rest POST /pods "$payload" 2>&1)"; then
      log "no pod on $gpu: $(head -c 200 <<<"$resp")"; continue
    fi
    pod="$(jq -r '.id // empty' <<<"$resp")"
    [[ -n "$pod" ]] || { log "no id: $(head -c 200 <<<"$resp")"; continue; }
    dph="$(jq -r '.costPerHr // 0' <<<"$resp")"
    ledger "pod-created $pod $name gpu=$gpu dph=$dph image=$image"
    nohup env POD_ID="$pod" CAP="$CAP_S" API="$API" bash -c \
      'sleep "$CAP"; curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/pods/$POD_ID" >/dev/null' \
      >/dev/null 2>&1 &
    jq --arg pod "$pod" --arg gpu "$gpu" --arg dph "$dph" --arg t "$(date +%s)" --arg name "$name" \
      '. + {pod: $pod, gpu: $gpu, dph: $dph, created: ($t|tonumber), name: $name}' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
    if awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 > c+0)}'; then
      log "pod $pod costs \$$dph/hr > cap: deleting"; cmd_down; exit 1
    fi
    log "pod $pod ($name) on $gpu at \$$dph/hr; backstop ${CAP_S}s"
    return 0
  done
  die "no capacity"
}

cmd_down() {
  local pod; pod="$(st pod)"; [[ -n "$pod" ]] || die "no pod in state"
  rest DELETE "/pods/$pod" >/dev/null && ledger "pod-deleted $pod" && log "deleted $pod"
  local t0; t0="$(st created)"
  [[ -n "$t0" ]] && log "lifetime $(( $(date +%s) - t0 ))s at \$$(st dph)/hr"
}

cmd_wait() {
  local pod base t0 code first="" i
  pod="$(st pod)"; base="https://$pod-8000.proxy.runpod.net"; t0="$(st created)"
  for i in $(seq 1 480); do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$base/ping" || true)"
    [[ -z "$first" && ( "$code" == 204 || "$code" == 200 ) ]] && { first=$(( $(date +%s) - t0 )); log "first /ping $code at ${first}s"; }
    [[ "$code" == 200 ]] && { log "ready at $(( $(date +%s) - t0 ))s after create"; echo "{\"first_ping_s\":${first:-null},\"ready_s\":$(( $(date +%s) - t0 ))}"; return 0; }
    (( i % 12 == 0 )) && log "ping $code ($(( $(date +%s) - t0 ))s)"
    sleep 5
  done
  die "not ready"
}

cmd_bundle() {
  local tgz; tgz="$(mktemp)"
  tar czf "$tgz" -C "$ROOT" tests/compat/package.json tests/compat/package-lock.json tests/compat/requirements.txt \
    tests/compat/suites crates/fastvideo-reactor/tests/compat/reactor_sdk_compat.py tests/console/smoke.cjs \
    tests/console/avatar.cjs scripts/gpu/lipsync_proxy.py scripts/serve/e2e
  side /bundle 120 -X PUT --data-binary "@$tgz" -H 'content-type: application/gzip'; echo
  rm -f "$tgz"
}

cmd_run() {
  local cmd="$1" to="${2:-900}" id r t0
  id="$(side /exec 30 -H 'content-type: application/json' -d "$(jq -nc --arg c "$cmd" '{cmd: $c}')" | jq -r .id)"
  [[ -n "$id" && "$id" != null ]] || die "exec refused"
  t0=$(date +%s)
  while :; do
    r="$(side "/exec/$id" 30 || echo '{}')"
    [[ "$(jq -r '.done // false' <<<"$r" 2>/dev/null)" == true ]] && break
    (( $(date +%s) - t0 < to )) || { log "timeout after ${to}s (job $id still running)"; jq -r '.out // ""' <<<"$r" | tail -40; return 124; }
    sleep 3
  done
  jq -r .out <<<"$r"
  log "rc=$(jq -r .rc <<<"$r") secs=$(jq -r .secs <<<"$r")"
  return "$(jq -r .rc <<<"$r")"
}

cmd_hooks() { side /hooks 30; }

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  down) cmd_down ;;
  wait) cmd_wait ;;
  bundle) cmd_bundle ;;
  run) shift; cmd_run "$@" ;;
  hooks) cmd_hooks ;;
  clients)
    # pod-clients.sh <sub> on the pod; `key`/`admin` in the args are replaced
    # by the state file's values on the pod side (never echoed here).
    shift; sub="$1"; shift
    args=""; for x in "$@"; do args+=" $(printf '%q' "$x")"; done
    envs="FV_KEY=$(printf '%q' "$(st key)") FV_ADMIN_TOKEN=$(printf '%q' "$(st admin)")"
    cmd_run "$envs bash /e2e/scripts/serve/e2e/pod-clients.sh $sub$args 2>&1 | tail -${FV_TAIL:-60}" "${FV_RUN_TIMEOUT:-1800}" ;;
  log) cmd_run "tail -n ${2:-40} /e2e/serve.log" 60 ;;
  restart) cmd_run "/e2e/serve.sh ${2:-/e2e/fv.toml}; sleep 1; cat /e2e/serve.pid" 120 ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
