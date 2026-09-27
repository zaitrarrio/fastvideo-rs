#!/usr/bin/env bash
# WP-19: real Runpod serverless (queue) cold starts, driven over REST.
#
#   serverless-coldstart.sh up <image> [gpu type] [volume id] [dc]
#       serverless template + queue endpoint (min 0 / max 1 workers, idle 5 s,
#       FlashBoot off) whose start command is serverless-worker.sh, inlined so
#       any runtime image works (also images built before the script existed).
#       Prints "<endpoint id> <template id>" and appends them to the ledger.
#   serverless-coldstart.sh job <endpoint> '<input json>'
#       POST /run, poll /status every 2 s; prints one JSON line: submit time,
#       first time seen IN_PROGRESS, completion time, Runpod's delayTime /
#       executionTime and the worker's output (its own timestamps).
#   serverless-coldstart.sh wait-idle <endpoint>
#       until /health reports no workers (the next job is a cold start).
#   serverless-coldstart.sh down <endpoint> <template>
#
# Ledger: artifacts/runpod/serverless/ledger.tsv (created/deleted ids).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
REST="${RUNPOD_REST:-https://rest.runpod.io/v1}"
API="${RUNPOD_QUEUE_API:-https://api.runpod.ai/v2}"
LEDGER="$ROOT/artifacts/runpod/serverless/ledger.tsv"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
mkdir -p "$(dirname "$LEDGER")"

rest() {
  curl -sS --fail-with-body -X "$1" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${3:+-d "$3"} "$REST$2"
}
queue() {
  curl -sS -X "$1" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${3:+-d "$3"} "$API/$2"
}
now() { date +%s.%N; }
ledger() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }

cmd_up() {
  local image="$1" gpu="${2:-NVIDIA H200}" vol="${3:-s2k01690bi}" dc="${4:-US-CA-2}" tag tpl ep script
  tag="fv-coldstart-$(date -u +%m%d%H%M%S)"
  script="$(cat "$HERE/serverless-worker.sh")"
  tpl="$(rest POST /templates "$(jq -n --arg name "$tag" --arg image "$image" --arg s "$script" '{
      name: $name, imageName: $image, isServerless: true, containerDiskInGb: 40,
      dockerEntrypoint: [], dockerStartCmd: ["bash", "-c", $s, "fv-worker"],
      env: {FV_W: "/runpod-volume/weights", NVIDIA_DRIVER_CAPABILITIES: "compute,utility"}
    }')" | jq -r '.id')"
  [[ -n "$tpl" && "$tpl" != null ]] || { echo "template create failed" >&2; exit 1; }
  ledger "template-created $tpl $tag $image"
  ep="$(rest POST /endpoints "$(jq -n --arg name "$tag" --arg tpl "$tpl" --arg gpu "$gpu" --arg vol "$vol" --arg dc "$dc" '{
      name: $name, templateId: $tpl, computeType: "GPU", gpuTypeIds: [$gpu], gpuCount: 1,
      networkVolumeId: $vol, dataCenterIds: [$dc], workersMin: 0, workersMax: 1,
      idleTimeout: 5, flashboot: false, executionTimeoutMs: 3600000,
      scalerType: "QUEUE_DELAY", scalerValue: 1, minCudaVersion: "13.0"
    }')" | jq -r '.id')" || { rest DELETE "/templates/$tpl" >/dev/null || true; exit 1; }
  [[ -n "$ep" && "$ep" != null ]] || { rest DELETE "/templates/$tpl" >/dev/null || true; echo "endpoint create failed" >&2; exit 1; }
  ledger "endpoint-created $ep $tag gpu=$gpu vol=$vol dc=$dc"
  echo "$ep $tpl"
}

cmd_job() {
  local ep="$1" input="$2" t_submit id st t_prog="" t_done="" last=""
  t_submit="$(now)"
  id="$(queue POST "$ep/run" "{\"input\":$input,\"policy\":{\"executionTimeout\":3600000}}" | jq -r '.id')"
  [[ -n "$id" && "$id" != null ]] || { echo "submit failed" >&2; exit 1; }
  echo "[$(date -u +%T)] submitted $id" >&2
  while :; do
    st="$(queue GET "$ep/status/$id" || true)"
    local s; s="$(jq -r '.status // "?"' <<<"$st" 2>/dev/null || echo '?')"
    [[ "$s" != "$last" ]] && { echo "[$(date -u +%T)] $s" >&2; last="$s"; }
    [[ "$s" == IN_PROGRESS && -z "$t_prog" ]] && t_prog="$(now)"
    case "$s" in
      COMPLETED | FAILED | CANCELLED | TIMED_OUT) t_done="$(now)"; break ;;
    esac
    sleep 2
  done
  jq -c --arg sub "$t_submit" --arg prog "${t_prog:-null}" --arg done "$t_done" --arg id "$id" \
    '{job: $id, t_submit: ($sub|tonumber), t_in_progress_seen: ($prog|tonumber? // null), t_done: ($done|tonumber),
      status, delayTime, executionTime, workerId, output, error}' <<<"$st"
}

cmd_wait_idle() {
  local ep="$1" h
  for _ in $(seq 1 90); do
    h="$(queue GET "$ep/health")"
    if [[ "$(jq '[.workers[]?] | add // 0' <<<"$h")" == 0 ]]; then echo "$h"; return 0; fi
    sleep 5
  done
  echo "workers still up: $h" >&2
  return 1
}

cmd_down() {
  local ep="$1" tpl="$2"
  # Scale to zero first; a delete with a running worker can be refused.
  rest PATCH "/endpoints/$ep" '{"workersMin":0,"workersMax":0}' >/dev/null 2>&1 || true
  rest DELETE "/endpoints/$ep" >/dev/null && ledger "endpoint-deleted $ep"
  rest DELETE "/templates/$tpl" >/dev/null && ledger "template-deleted $tpl"
  echo "deleted $ep $tpl"
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  job) shift; cmd_job "$@" ;;
  wait-idle) shift; cmd_wait_idle "$@" ;;
  down) shift; cmd_down "$@" ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
