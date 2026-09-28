#!/usr/bin/env bash
# WP-18 GPU E2E, LTX-2.5 pod (docs/serve/design.md §7.6, docs/serve/e2e/ltx.md).
#
#   ltx-pod.sh up [image]        create a pod `fv-e2e-c-<ts>` running fv-serve with
#                                the repo's configs/serve/runpod-ltx.toml (inlined,
#                                so a config change needs no image rebuild); the
#                                model is FV_LTX_RECIPE (ltx-turbo | ltx-pro).
#                                Writes pod id + per-run key to $FV_E2E_STATE (0600)
#   ltx-pod.sh switch <recipe>   restart the same pod with another recipe
#                                (PATCH: the image stays pulled)
#   ltx-pod.sh down              delete the pod and verify it is gone
#   ltx-pod.sh wait              wait for /ping 200, print seconds
#   ltx-pod.sh entry             print the container command (for a local check)
#
# Guards: RUNPOD_GPU_MAX_DPH (default 2.2), FV_POD_CAP_S (default 5400) detached
# backstop delete, FV_MIN_BALANCE (default 8). Env: RUNPOD_API_KEY,
# RUNPOD_GPU_TYPES (default RTX PRO 6000), RUNPOD_VOLUME_ID (default the EU
# weight volume, mounted at /workspace and never written: state is /fvstate).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
API="https://rest.runpod.io/v1"
GQL="https://api.runpod.io/graphql"
STATE="${FV_E2E_STATE:-$ROOT/target/e2e-ltx.state}"
GPUS="${RUNPOD_GPU_TYPES:-NVIDIA RTX PRO 6000 Blackwell Server Edition}"
MAX_DPH="${RUNPOD_GPU_MAX_DPH:-2.2}"
CAP_S="${FV_POD_CAP_S:-5400}"
MIN_BALANCE="${FV_MIN_BALANCE:-8}"
VOL="${RUNPOD_VOLUME_ID-jg48s6o1w0}"
RECIPE="${FV_LTX_RECIPE:-ltx-turbo}"
LEDGER="$ROOT/artifacts/runpod/serve/ledger.tsv"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 2; }
rest() { curl -sS --fail-with-body -X "$1" -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' ${3:+-d "$3"} "$API$2"; }
ledger() { mkdir -p "$(dirname "$LEDGER")"; printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"

# The container command: write the inlined config (recipe substituted) and exec fv-serve.
entry_cmd() {
  local b64
  b64="$(base64 -w0 "$ROOT/configs/serve/runpod-ltx.toml")"
  printf '%s' "echo $b64 | base64 -d | sed -e \"s/^recipe = \\\"ltx-turbo\\\"/recipe = \\\"\$FV_LTX_RECIPE\\\"/\" -e \"s/^id = \\\"ltx25-distill-sol\\\"/id = \\\"ltx25-\$FV_LTX_RECIPE\\\"/\" -e \"s#fastvideo/ltx-turbo#fastvideo/\$FV_LTX_RECIPE#\" > /tmp/fv-ltx.toml && cat /tmp/fv-ltx.toml | grep -E '^(id|recipe|fal_apps)' && exec /opt/fastvideo-rs/bin/fv-serve --config /tmp/fv-ltx.toml"
}

payload() { # $1 image $2 gpu $3 name $4 keyhash $5 recipe
  local plan
  plan="$(RUNPOD_VOLUME_ID="$VOL" RUNPOD_GPU_TYPES="$2" FV_SERVE_CONFIG=/tmp/fv-ltx.toml bash "$ROOT/scripts/serve/runpod-pod.sh" plan "$1")"
  jq --arg name "$3" --arg keys "$4" --arg cmd "$(entry_cmd)" --arg r "$5" '
    .name = $name | .dockerEntrypoint = ["bash", "-c", $cmd] | .dockerStartCmd = []
    | .env.FV_API_KEYS = $keys | .env.FV_LTX_RECIPE = $r' <<<"$plan"
}

balance() { curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' "$GQL" -d '{"query":"{ myself { clientBalance } }"}' | jq -r '.data.myself.clientBalance'; }

cmd_up() {
  local image="${1:-ghcr.io/zaitrarrio/fastvideo-rs-serve:sha-$(git -C "$ROOT" rev-parse --short=7 HEAD)}"
  local b key keyhash name resp pod dph gpu
  b="$(balance)"; log "balance \$$b"
  awk -v b="$b" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+2.5)}' || die "balance \$$b too close to the floor \$$MIN_BALANCE"
  [[ ! -s "$STATE" ]] || die "state $STATE exists (pod up?); run down first"
  key="fvk-$(openssl rand -hex 16)"; keyhash="$(printf '%s' "$key" | sha256sum | cut -d' ' -f1)"
  name="${FV_POD_NAME_PREFIX:-fv-e2e-c}-$(date -u +%m%d%H%M%S)"
  IFS=',' read -r -a types <<<"$GPUS"
  for gpu in "${types[@]}"; do
    if ! resp="$(rest POST /pods "$(payload "$image" "$gpu" "$name" "$keyhash" "$RECIPE")" 2>&1)"; then
      log "no pod on $gpu: $(head -c 200 <<<"$resp")"; continue
    fi
    pod="$(jq -r '.id // empty' <<<"$resp")"; [[ -n "$pod" ]] || continue
    dph="$(jq -r '.costPerHr // 0' <<<"$resp")"
    ledger "pod-created $pod $name gpu=$gpu image=$image dph=$dph"
    if awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 > c+0)}'; then
      rest DELETE "/pods/$pod" >/dev/null || true; ledger "pod-deleted $pod over-cap"; log "$gpu \$$dph/hr over cap"; continue
    fi
    # shellcheck disable=SC2016
    nohup env POD_ID="$pod" CAP="$CAP_S" API="$API" bash -c \
      'sleep "$CAP"; curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/pods/$POD_ID" >/dev/null' >/dev/null 2>&1 &
    umask 077
    printf '%s\t%s\t%s\t%s\t%s\n' "$pod" "$key" "$gpu" "$dph" "$(date +%s)" >"$STATE"
    log "pod $pod ($name) on $gpu at \$$dph/hr, backstop ${CAP_S}s, recipe $RECIPE"
    echo "$pod"; return 0
  done
  die "no GPU in [$GPUS]"
}

cmd_switch() {
  local pod key gpu recipe="${1:?recipe}" env
  IFS=$'\t' read -r pod key gpu _ <"$STATE"
  # The create env again (secret references, never values) with the new recipe.
  env="$(payload ghcr.io/x@sha256:0 "$gpu" x "$(printf '%s' "$key" | sha256sum | cut -d' ' -f1)" "$recipe" | jq -c .env)"
  rest PATCH "/pods/$pod" "$(jq -nc --argjson e "$env" '{env: $e}')" | jq -c '{id, desiredStatus}'
  ledger "pod-switch $pod recipe=$recipe"
}

cmd_wait() {
  local pod t0 code
  IFS=$'\t' read -r pod _ <"$STATE"; t0=$(date +%s)
  while :; do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "https://$pod-8000.proxy.runpod.net/ping" || true)"
    [[ "$code" == 200 ]] && { echo $(( $(date +%s) - t0 )); return 0; }
    (( $(date +%s) - t0 < ${FV_BOOT_WAIT_S:-1500} )) || die "no /ping 200 (last $code)"
    sleep 10
  done
}

cmd_down() {
  local pod
  IFS=$'\t' read -r pod _ <"$STATE"
  rest DELETE "/pods/$pod" >/dev/null || true
  if curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/pods" | jq -e --arg p "$pod" 'any(.[]; .id == $p)' >/dev/null; then
    die "pod $pod still listed"
  fi
  ledger "pod-deleted $pod"; rm -f "$STATE"; log "deleted $pod (verified)"
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  switch) shift; cmd_switch "$@" ;;
  wait) cmd_wait ;;
  down) cmd_down ;;
  entry) entry_cmd; echo ;;
  *) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
