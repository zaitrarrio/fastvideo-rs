#!/usr/bin/env bash
# Add one weights-manifest.tsv tree to a network volume with a CPU pod,
# add-only, driven through the Runpod REST API and the pod's HTTPS proxy.
#
#   fetch-hub-tree.sh <dest> <revision> [expect-sha256.txt]
#
#   <dest> is a manifest row (dest, hub_id, globs); <revision> pins the Hub
#   commit. The pod (scripts/gpu/fetch-hub-tree.py) downloads into
#   weights/.<dest>.partial-<stamp>, checks every file against the Hub
#   (LFS SHA-256, git blob SHA-1 for the rest) and, with the optional
#   sha256.txt of the other volume's copy, against that list too; only then
#   it writes .complete and renames the folder to weights/<dest>. It stops if
#   weights/<dest> already exists. Nothing else on the volume is touched.
#
#   RUNPOD_VOLUME_NAME (default fv-weights-h3-ltx-hy, EU; the US volume
#   fv-weights-b200-us was deleted 2026-10 and is refused, EU only,
#   scripts/gpu/volumes.sh) picks the volume and the pod's datacenter. FV_POD_CAP_S (default 3600)
#   is a hard wall-clock cap: a detached backstop (setsid, survives this
#   shell) deletes the pod regardless. FV_MIN_BALANCE (default 8) is the
#   Runpod balance floor. The pod log and sha256.txt land in
#   artifacts/runpod/fetch-<dest>-<volume>/ ('/' in <dest> becomes '-').
#   FV_POD_PREFIX (default fv-p0-fetch) starts the pod name; FV_FETCH_VCPU
#   (default 8) sizes the CPU pod when the datacentre is short of stock.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source-path=SCRIPTDIR source=volumes.sh
source "$HERE/volumes.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
VOL_NAME="${RUNPOD_VOLUME_NAME:-$FV_EU_VOLUME_NAME}"
fv_check_volume "$VOL_NAME"
CAP_S="${FV_POD_CAP_S:-3600}"
MIN_BALANCE="${FV_MIN_BALANCE:-8}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
dest="${1:?dest}"; rev="${2:?revision}"; expect="${3:-}"
OUT="${FETCH_OUT:-$ROOT/artifacts/runpod/fetch-${dest//\//-}-$VOL_NAME}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
rest() {
  local method="$1" path="$2" body="${3:-}"
  curl -sS --fail-with-body -X "$method" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${body:+-d "$body"} "$API$path"
}

row="$(awk -F'\t' -v d="$dest" '$1==d {print; exit}' "$HERE/weights-manifest.tsv")"
[[ -n "$row" ]] || { echo "no weights-manifest.tsv row for $dest" >&2; exit 1; }
repo="$(cut -f2 <<<"$row")"; globs="$(cut -f3 <<<"$row")"
bal="$(curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' "$GQL" \
  -d '{"query":"{ myself { clientBalance } }"}' | jq -r '.data.myself.clientBalance // empty')"
awk -v b="$bal" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+0)}' || { echo "balance \$$bal below \$$MIN_BALANCE" >&2; exit 1; }
log "balance \$$bal; $repo@$rev -> $VOL_NAME:weights/$dest"
vol="$(rest GET /networkvolumes | jq -r --arg n "$VOL_NAME" '.[] | select(.name==$n) | "\(.id) \(.dataCenterId)"' | head -1)"
[[ -n "$vol" ]] || { echo "no volume named $VOL_NAME" >&2; exit 1; }
vol_id="${vol% *}"; dc="${vol#* }"
py_b64="$(base64 -w0 "$HERE/fetch-hub-tree.py")"
exp_b64=""; [[ -n "$expect" ]] && exp_b64="$(base64 -w0 "$expect")"
start='mkdir -p /srv && cd /srv && (python -m http.server 8000 --directory /srv >/dev/null 2>&1 &) && echo "$FETCH_PY" | base64 -d > /srv/fetch.py && pip install -q --no-cache-dir "huggingface_hub>=0.34" hf_xet >>/srv/log.txt 2>&1 && python /srv/fetch.py; sleep infinity'
payload="$(jq -n --arg name "${FV_POD_PREFIX:-fv-p0-fetch}-${dest##*/}-$(date -u +%m%d%H%M)" --arg vol "$vol_id" --arg dc "$dc" \
  --arg start "$start" --arg py "$py_b64" --arg repo "$repo" --arg rev "$rev" --arg dest "$dest" \
  --arg globs "$globs" --arg vcpu "${FV_FETCH_VCPU:-8}" --arg exp "$exp_b64" --arg hf "${HF_TOKEN:-}" '{
    name: $name, imageName: "python:3.12-slim", cloudType: "SECURE", computeType: "CPU",
    cpuFlavorIds: ["cpu3c","cpu5c","cpu3g"], cpuFlavorPriority: "availability", vcpuCount: ($vcpu|tonumber),
    containerDiskInGb: 20, networkVolumeId: $vol, volumeMountPath: "/workspace",
    dataCenterIds: [$dc], ports: ["8000/http"],
    dockerStartCmd: ["/bin/bash","-lc",$start],
    env: ({FETCH_PY: $py, FETCH_REPO: $repo, FETCH_REVISION: $rev, FETCH_DEST: $dest, FETCH_GLOBS: $globs}
      + (if $exp != "" then {EXPECT_SHA256: $exp} else {} end)
      + (if $hf != "" then {HF_TOKEN: $hf} else {} end))
  }')"
resp="$(rest POST /pods "$payload")" || { echo "pod create failed: $resp" >&2; exit 1; }
id="$(jq -r '.id // empty' <<<"$resp")"
[[ -n "$id" ]] || { echo "pod create failed: $resp" >&2; exit 1; }
mkdir -p "$OUT"; echo "$id" >"$OUT/pod-id"
log "pod $id \$$(jq -r '.costPerHr' <<<"$resp")/hr dc=$dc vol=$vol_id"
# Detached backstop: deletes the pod after CAP_S even if this shell dies.
setsid nohup bash -c "sleep $CAP_S; curl -sS -X DELETE -H \"Authorization: Bearer \$RUNPOD_API_KEY\" '$API/pods/$id' >/dev/null 2>&1" \
  >/dev/null 2>&1 < /dev/null &
backstop=$!
echo "$backstop" >"$OUT/backstop-pid"
log "backstop pid $backstop (cap ${CAP_S}s)"
cleanup() {
  rest DELETE "/pods/$id" >/dev/null 2>&1 || true
  if rest GET "/pods/$id" >/dev/null 2>&1; then log "WARNING: pod $id still listed after delete"; else log "pod $id deleted (verified)"; fi
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
curl -sS --max-time 60 --fail "$url/log.txt" -o "$OUT/log.txt" 2>/dev/null || true
curl -sS --max-time 60 --fail "$url/sha256.txt" -o "$OUT/sha256.txt" 2>/dev/null || true
log "DONE=$status"
[[ "$status" == 0 ]]
