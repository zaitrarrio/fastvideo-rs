#!/usr/bin/env bash
# Fill a Runpod weights network volume from the R2 weights store, or compare one with it, with ONE CPU pod
# (docs/ops/runpod-volumes.md §5.7: the H3 + LTX volume of the 2026-10-13 owner decision).
#
#   r2-fill-volume.sh up <volume-id> [fill|compare]   create the pod (+ a detached wall-clock backstop that deletes it)
#   r2-fill-volume.sh status <pod-id>                 status.tsv, the log tails, DONE (through the Runpod proxy)
#   r2-fill-volume.sh fetch <pod-id> DIR              its logs, compare.txt, du.tsv into DIR
#   r2-fill-volume.sh down <pod-id>                   delete the pod and check it is gone
#
# fill (default): the pod mounts the volume at /workspace and runs r2-fill-volume-pod.sh: r2-fetch.py --add-only
# --full-hash writes each tree under a temp name, checks every part and file against the R2 manifest, renames it
# into place with its markers, then compare-dir reads every tree back (SHA-256 of every part). compare: compare-dir
# only; nothing is written to the volume (use it on the old volume to tie both to the same manifests).
#
# Env: RUNPOD_API_KEY; FV_FILL_TREES (default: FV_EU_VOLUME_TREES from volumes.sh); FV_POD_CAP_S (backstop,
# default 10800 = 3 h); FV_MIN_BALANCE (default 12); FV_FILL_VCPU (default 16); FV_FILL_TAG (resume an earlier
# pod's staging: fill-<its pod id>); FV_FILL_FLAVORS (CPU flavors, any
# in stock: default "cpu3c cpu5c cpu3g cpu5g cpu3m cpu5m"). Only the read-only R2 key pair
# (/root/.config/fv/r2_weights_ro_*) goes into the pod's env; nothing is printed. The scripts reach the pod as a
# tarball in the staging bucket (fv-weights-staging, tmp/), fetched through a presigned URL and checked by SHA-256.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=volumes.sh
source "$HERE/volumes.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
CFG="${FV_CONFIG_DIR:-/root/.config/fv}"
CAP_S="${FV_POD_CAP_S:-10800}"
MIN_BALANCE="${FV_MIN_BALANCE:-12}"
ACCT="${FV_CF_ACCOUNT_ID:-0cd06fd37ee4e07de370821cb3852a8a}"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
auth() { printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY"; }
rest() { # method path: a body (if any) comes on stdin, never on the command line
  local method="$1" path="$2"
  if [[ "$method" == POST ]]; then
    curl -sS --fail-with-body -X POST -H @<(auth) -H 'content-type: application/json' --data-binary @- "$API$path"
  else
    curl -sS --fail-with-body -X "$method" -H @<(auth) "$API$path"
  fi
}
balance() { curl -sS -H @<(auth) -H 'content-type: application/json' "$GQL" -d '{"query":"{ myself { clientBalance } }"}' | jq -r '.data.myself.clientBalance // empty'; }

cmd="${1:-}"
case "$cmd" in
  up)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    vol="${2:?volume id}"; mode="${3:-fill}"
    [[ "$mode" == fill || "$mode" == compare ]] || { log "mode: fill or compare"; exit 2; }
    fv_check_volume "$vol"
    trees="${FV_FILL_TREES:-$FV_EU_VOLUME_TREES}"
    vj="$(rest GET "/networkvolumes/$vol")" || { log "no volume $vol"; exit 1; }
    dc="$(jq -r '.dataCenterId' <<<"$vj")"
    bal="$(balance)"
    awk -v b="$bal" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+2)}' || { log "balance \$$bal too close to the \$$MIN_BALANCE floor"; exit 1; }
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    tar czf "$tmp/bundle.tgz" -C "$HERE" r2-weights.py r2-fetch.py r2-fill-volume-pod.sh weights-manifest.tsv weights-revisions.tsv weights-sha256.tsv weights-r2.tsv
    sha="$(sha256sum "$tmp/bundle.tgz" | cut -d' ' -f1)"
    url="$(python3 "$HERE/r2-weights.py" --bucket fv-weights-staging put-temp "$tmp/bundle.tgz" --key "tmp/r2-fill-bundle-$sha.tgz" --ttl "$CAP_S")"
    log "balance \$$bal; volume $vol ($(jq -r '.name' <<<"$vj"), $(jq -r '.size' <<<"$vj") GB, $dc); mode $mode; bundle ${sha:0:12} staged"
    # shellcheck disable=SC2016 # expanded on the pod, not here
    start='mkdir -p /srv /opt/fv && cd /srv && (python -m http.server 8000 --directory /srv >/dev/null 2>&1 &) && python -c "
import hashlib, io, os, tarfile, urllib.request
b = urllib.request.urlopen(os.environ[\"FV_BUNDLE_URL\"], timeout=120).read()
assert hashlib.sha256(b).hexdigest() == os.environ[\"FV_BUNDLE_SHA256\"], \"bundle sha256\"
tarfile.open(fileobj=io.BytesIO(b)).extractall(\"/opt/fv\")
" >>/srv/log.txt 2>&1 && FV_POD_ID="$RUNPOD_POD_ID" bash /opt/fv/r2-fill-volume-pod.sh; [ -f /srv/DONE ] || echo 1 >/srv/DONE; sleep infinity'
    # The keys and the presigned URL go into jq through its environment (env.X), never through argv.
    export FVX_URL="$url" FVX_ROID FVX_ROSEC
    FVX_ROID="$(cat "$CFG/r2_weights_ro_access_key_id")"; FVX_ROSEC="$(cat "$CFG/r2_weights_ro_secret_access_key")"
    payload="$(jq -n --arg name "fv-p0-r2fill-$mode-$(date -u +%m%d%H%M)" --arg vol "$vol" --arg dc "$dc" --arg start "$start" \
      --arg vcpu "${FV_FILL_VCPU:-16}" --arg flavors "${FV_FILL_FLAVORS:-cpu3c cpu5c cpu3g cpu5g cpu3m cpu5m}" --arg sha "$sha" --arg ep "https://$ACCT.r2.cloudflarestorage.com" --arg mode "$mode" --arg trees "$trees" --arg tag "${FV_FILL_TAG:-}" '{
        name: $name, imageName: "python:3.12-slim", cloudType: "SECURE", computeType: "CPU",
        cpuFlavorIds: ($flavors | split(" ") | map(select(. != ""))), cpuFlavorPriority: "availability", vcpuCount: ($vcpu|tonumber),
        containerDiskInGb: 20, networkVolumeId: $vol, volumeMountPath: "/workspace", dataCenterIds: [$dc], ports: ["8000/http"],
        dockerStartCmd: ["/bin/bash","-lc",$start],
        env: {FV_BUNDLE_URL: env.FVX_URL, FV_BUNDLE_SHA256: $sha, FV_WEIGHTS_R2_ENDPOINT: $ep, FV_WEIGHTS_R2_BUCKET: "fv-weights",
              FV_R2_RO_ID: env.FVX_ROID, FV_R2_RO_SECRET: env.FVX_ROSEC, FV_FILL_MODE: $mode, FV_FILL_TREES: $trees, FV_FILL_TAG: $tag}
      }')"
    unset FVX_URL FVX_ROID FVX_ROSEC
    resp="$(rest POST /pods <<<"$payload")" || { log "pod create failed: $(jq -r '.error // .' <<<"$resp" | head -c 300)"; exit 1; }
    id="$(jq -r '.id // empty' <<<"$resp")"
    [[ -n "$id" ]] || { log "pod create failed: $(head -c 300 <<<"$resp")"; exit 1; }
    setsid nohup bash -c "sleep $CAP_S; curl -sS -X DELETE -H @<(printf 'Authorization: Bearer %s\\n' \"\$RUNPOD_API_KEY\") '$API/pods/$id' >/dev/null 2>&1" >/dev/null 2>&1 </dev/null &
    log "pod $id \$$(jq -r '.costPerHr' <<<"$resp")/hr in $dc, backstop pid $! deletes it after ${CAP_S}s; logs https://$id-8000.proxy.runpod.net/"
    echo "$id"
    ;;
  status)
    id="${2:?pod id}"; u="https://$id-8000.proxy.runpod.net"
    curl -sS --max-time 20 "$u/status.tsv" 2>/dev/null || true
    curl -sS --max-time 20 "$u/log.txt" 2>/dev/null | tail -n "${FV_TAIL:-5}" || true
    curl -sS --max-time 20 "$u/fetch.log" 2>/dev/null | tail -n "${FV_TAIL:-3}" || true
    curl -sS --max-time 20 "$u/compare.txt" 2>/dev/null | tail -n "${FV_TAIL:-3}" || true
    printf 'DONE=%s\n' "$(curl -sS --max-time 20 --fail "$u/DONE" 2>/dev/null || true)"
    ;;
  fetch)
    id="${2:?pod id}"; out="${3:?dir}"; u="https://$id-8000.proxy.runpod.net"
    mkdir -p "$out"
    for f in log.txt status.tsv inventory.txt fetch.log fetch-status.json compare.txt du.tsv DONE; do
      curl -sS --max-time 60 --fail "$u/$f" -o "$out/$f" 2>/dev/null || true
    done
    log "fetched into $out"
    ;;
  down)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    id="${2:?pod id}"
    rest DELETE "/pods/$id" >/dev/null 2>&1 || true
    if rest GET "/pods/$id" >/dev/null 2>&1; then log "WARNING: pod $id still listed"; exit 1; fi
    log "pod $id deleted (GET /pods/$id: not found)"
    [[ -n "${FV_FILL_KEEP_BUNDLE:-}" ]] || python3 "$HERE/r2-weights.py" --bucket fv-weights-staging rm-temp --prefix tmp/r2-fill-bundle- || true
    ;;
  *) sed -n '2,19p' "$0"; exit 2 ;;
esac
