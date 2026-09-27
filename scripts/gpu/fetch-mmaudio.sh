#!/usr/bin/env bash
# Fetch the MMAudio large-44k-v2 weight tree onto a network volume with a CPU
# pod, driven through the Runpod REST API and the pod's HTTPS proxy (no SSH).
#
#   fetch-mmaudio.sh                 create the pod, wait for DONE, pull the log
#                                    and key listings, delete the pod
#   RUNPOD_VOLUME_NAME=fv-weights-b200-us (default) picks the volume; its
#   datacenter picks the pod's. FV_POD_CAP_S (default 3600) is a hard wall
#   clock cap: a backstop process deletes the pod regardless.
#
# The pod runs scripts/gpu/fetch-mmaudio.py (shipped base64 in the env) and
# serves /srv (log.txt, keys-*.txt, DONE) on port 8000.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
VOL_NAME="${RUNPOD_VOLUME_NAME:-fv-weights-b200-us}"
CAP_S="${FV_POD_CAP_S:-3600}"
OUT="${FETCH_OUT:-$ROOT/artifacts/runpod/fetch-mmaudio}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
[[ "$VOL_NAME" != "fv-weights-h3-ltx-hy" ]] || { echo "refusing: the EU volume is full" >&2; exit 2; }

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
rest() {
  local method="$1" path="$2" body="${3:-}"
  curl -sS --fail-with-body -X "$method" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${body:+-d "$body"} "$API$path"
}

vol="$(rest GET /networkvolumes | jq -r --arg n "$VOL_NAME" '.[] | select(.name==$n) | "\(.id) \(.dataCenterId)"' | head -1)"
[[ -n "$vol" ]] || { echo "no volume named $VOL_NAME" >&2; exit 1; }
vol_id="${vol% *}"; dc="${vol#* }"
script_b64="$(base64 -w0 "$HERE/fetch-mmaudio.py")"
start='mkdir -p /srv && cd /srv && (python -m http.server 8000 --directory /srv >/dev/null 2>&1 &) && echo "$FETCH_PY" | base64 -d > /srv/fetch.py && pip install -q --no-cache-dir huggingface_hub safetensors numpy >>/srv/log.txt 2>&1 && pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu >>/srv/log.txt 2>&1 && python /srv/fetch.py; sleep infinity'
payload="$(jq -n --arg name "fv-fetch-mmaudio-$(date -u +%m%d%H%M)" --arg vol "$vol_id" --arg dc "$dc" \
  --arg start "$start" --arg py "$script_b64" --arg hf "${HF_TOKEN:-}" '{
    name: $name, imageName: "python:3.12-slim", cloudType: "SECURE", computeType: "CPU",
    cpuFlavorIds: ["cpu3c","cpu5c","cpu3g"], cpuFlavorPriority: "availability", vcpuCount: 8,
    containerDiskInGb: 30, networkVolumeId: $vol, volumeMountPath: "/workspace",
    dataCenterIds: [$dc], ports: ["8000/http"],
    dockerStartCmd: ["/bin/bash","-lc",$start],
    env: ({FETCH_PY: $py, HF_HUB_ENABLE_HF_TRANSFER: "0"} + (if $hf != "" then {HF_TOKEN: $hf} else {} end))
  }')"
resp="$(rest POST /pods "$payload")"
id="$(jq -r '.id // empty' <<<"$resp")"
[[ -n "$id" ]] || { echo "pod create failed: $resp" >&2; exit 1; }
log "pod $id \$$(jq -r '.costPerHr' <<<"$resp")/hr dc=$dc vol=$vol_id"
mkdir -p "$OUT"; echo "$id" >"$OUT/pod-id"
# Backstop: delete the pod after CAP_S whatever happens to this shell.
( sleep "$CAP_S"; rest DELETE "/pods/$id" >/dev/null 2>&1 || true ) &
backstop=$!
echo "$backstop" >"$OUT/backstop-pid"
log "backstop pid $backstop (cap ${CAP_S}s)"
cleanup() { rest DELETE "/pods/$id" >/dev/null 2>&1 || true; kill "$backstop" 2>/dev/null || true; log "pod $id deleted"; }
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
for f in keys-mmaudio_large_44k_v2 keys-vae_44k keys-synchformer keys-bigvgan_v2_44k keys-clip_dfn5b_h14_384; do
  curl -sS --max-time 60 --fail "$url/$f.txt" -o "$OUT/$f.txt" 2>/dev/null || true
done
log "DONE=$status"
[[ "$status" == 0 ]]
