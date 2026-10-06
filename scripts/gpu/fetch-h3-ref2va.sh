#!/usr/bin/env bash
# Fetch the H3 Ref2VA tree (transformer_ref + lightx2v ref2v turbo LoRAs) onto
# one weight volume, add-only, with a CPU pod driven through the Runpod REST
# API and the pod's HTTPS proxy (no SSH). Same pattern as fetch-mmaudio.sh.
#
#   fetch-h3-ref2va.sh <volume id>     jg48s6o1w0 (EU; the US volume s2k01690bi
#                                      was deleted 2026-10 and is refused)
#
# The pod runs scripts/gpu/fetch-h3-ref2va.py: it refuses when
# weights/h3-ref2va exists, downloads into weights/.h3-ref2va.partial-<stamp>,
# checks every file against the Hub (LFS SHA-256, sizes), re-reads it after an
# fsync, and only then renames the folder to weights/h3-ref2va. Nothing else on
# the volume is touched. FV_POD_CAP_S (default 5400) is a hard wall-clock cap
# enforced by a detached backstop that deletes the pod whatever happens here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source-path=SCRIPTDIR source=volumes.sh
source "$HERE/volumes.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
VOL_ID="${1:?volume id}"
fv_check_volume "$VOL_ID"
CAP_S="${FV_POD_CAP_S:-5400}"
OUT="${FETCH_OUT:-$ROOT/artifacts/runpod/fetch-h3-ref2va/$VOL_ID}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
rest() {
  local method="$1" path="$2" body="${3:-}"
  curl -sS --fail-with-body -X "$method" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${body:+-d "$body"} "$API$path"
}

dc="$(rest GET "/networkvolumes/$VOL_ID" | jq -r '.dataCenterId // empty')"
[[ -n "$dc" ]] || { echo "no volume $VOL_ID" >&2; exit 1; }
script_b64="$(base64 -w0 "$HERE/fetch-h3-ref2va.py")"
start='mkdir -p /srv && cd /srv && (python -m http.server 8000 --directory /srv >/dev/null 2>&1 &) && echo "$FETCH_PY" | base64 -d > /srv/fetch.py && pip install -q --no-cache-dir "huggingface_hub>=0.34" >>/srv/log.txt 2>&1 && python /srv/fetch.py; sleep infinity'
payload="$(jq -n --arg name "fv-ref-fetch-$(date -u +%m%d%H%M)" --arg vol "$VOL_ID" --arg dc "$dc" \
  --arg start "$start" --arg py "$script_b64" --arg hf "${HF_TOKEN:-}" '{
    name: $name, imageName: "python:3.12-slim", cloudType: "SECURE", computeType: "CPU",
    cpuFlavorIds: ["cpu3c","cpu5c","cpu3g"], cpuFlavorPriority: "availability", vcpuCount: 8,
    containerDiskInGb: 20, networkVolumeId: $vol, volumeMountPath: "/workspace",
    dataCenterIds: [$dc], ports: ["8000/http"],
    dockerStartCmd: ["/bin/bash","-lc",$start],
    env: ({FETCH_PY: $py, HF_HUB_ENABLE_HF_TRANSFER: "0", HF_HUB_DISABLE_PROGRESS_BARS: "1"}
          + (if $hf != "" then {HF_TOKEN: $hf} else {} end))
  }')"
resp="$(rest POST /pods "$payload")"
id="$(jq -r '.id // empty' <<<"$resp")"
[[ -n "$id" ]] || { echo "pod create failed: $resp" >&2; exit 1; }
mkdir -p "$OUT"; echo "$id" >"$OUT/pod-id"
log "pod $id \$$(jq -r '.costPerHr' <<<"$resp")/hr dc=$dc vol=$VOL_ID"
# Detached backstop: deletes the pod after CAP_S even if this shell dies.
nohup setsid bash -c "sleep $CAP_S; curl -sS -X DELETE -H 'Authorization: Bearer $RUNPOD_API_KEY' '$API/pods/$id' >/dev/null 2>&1" \
  >/dev/null 2>&1 &
backstop=$!
echo "$backstop" >"$OUT/backstop-pid"
log "backstop pid $backstop (cap ${CAP_S}s)"
cleanup() {
  rest DELETE "/pods/$id" >/dev/null 2>&1 || true
  sleep 5
  if rest GET "/pods/$id" >/dev/null 2>&1; then log "WARN pod $id still listed after delete"; else log "pod $id deleted (verified)"; fi
  kill "$backstop" 2>/dev/null || true
}
trap cleanup EXIT
url="https://$id-8000.proxy.runpod.net"
status=""
while :; do
  sleep 30
  status="$(curl -sS --max-time 20 --fail "$url/DONE" 2>/dev/null || true)"
  curl -sS --max-time 20 --fail "$url/log.txt" -o "$OUT/log.txt" 2>/dev/null || true
  [[ -s "$OUT/log.txt" ]] && tail -n 1 "$OUT/log.txt" >&2
  [[ -n "$status" ]] && break
done
curl -sS --max-time 60 --fail "$url/sha256.txt" -o "$OUT/sha256.txt" 2>/dev/null || true
log "DONE=$status"
[[ "$status" == 0 ]]
