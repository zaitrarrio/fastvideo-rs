#!/usr/bin/env bash
# Copy the EU weight volume into the R2 weights store with ONE Runpod CPU pod (docs/ops/r2-weights.md §6).
#
#   r2-copy-volume.sh up                 create the pod (+ a detached wall-clock backstop that deletes it)
#   r2-copy-volume.sh status <pod-id>    the pod's status.tsv, log tail and DONE (through the Runpod proxy)
#   r2-copy-volume.sh fetch <pod-id> DIR its logs, listings and record.tsv into DIR
#   r2-copy-volume.sh down <pod-id>      delete the pod and check it is gone
#
# The pod (python:3.12-slim, secure CPU, EUR-IS-1) mounts jg48s6o1w0 at /workspace and only READS it; it runs
# r2-copy-volume-pod.sh: every Hub tree (checked against the Hub's listing at its pinned revision) and every
# volume-only tree via r2-weights.py publish-dir, runs/ and wan/ via `archive`, then compare-dir, a deep verify read
# with fv-control's read-only key, and `record`. Rerunning `up` after a dead pod resumes (add-only, idempotent).
#
# Env: RUNPOD_API_KEY; FV_POD_CAP_S (backstop, default 14400 = 4 h); FV_MIN_BALANCE (default 8); FV_COPY_VCPU
# (default 16); FV_COPY_JOBS / FV_COPY_FILES / FV_COPY_WINDOW / FV_COPY_ONLY / FV_COPY_REV (passed to the pod).
# Credentials are read from /root/.config/fv (r2_weights_rw_*, r2_weights_ro_*, cf_api_token) into the pod's env
# only; nothing is printed. The scripts reach the pod as a tarball in the staging bucket (fv-weights-staging,
# tmp/), fetched through a presigned URL and checked by SHA-256; `down` deletes that tarball.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=volumes.sh
source "$HERE/volumes.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
CFG="${FV_CONFIG_DIR:-/root/.config/fv}"
CAP_S="${FV_POD_CAP_S:-14400}"
MIN_BALANCE="${FV_MIN_BALANCE:-8}"
ACCT="${FV_CF_ACCOUNT_ID:-0cd06fd37ee4e07de370821cb3852a8a}"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
rest() { # method path: a body (if any) comes on stdin, never on the command line
  local method="$1" path="$2"
  if [[ "$method" == POST ]]; then
    curl -sS --fail-with-body -X POST -H @<(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY") -H 'content-type: application/json' --data-binary @- "$API$path"
  else
    curl -sS --fail-with-body -X "$method" -H @<(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY") "$API$path"
  fi
}
balance() { curl -sS -H @<(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY") -H 'content-type: application/json' "$GQL" -d '{"query":"{ myself { clientBalance } }"}' | jq -r '.data.myself.clientBalance // empty'; }

cmd="${1:-}"
case "$cmd" in
  up)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    bal="$(balance)"
    awk -v b="$bal" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+2)}' || { log "balance \$$bal too close to the \$$MIN_BALANCE floor"; exit 1; }
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    tar czf "$tmp/bundle.tgz" -C "$HERE" r2-weights.py r2-fetch.py r2-copy-volume-pod.sh weights-manifest.tsv weights-revisions.tsv weights-sha256.tsv weights-r2.tsv verify-weights.sh
    sha="$(sha256sum "$tmp/bundle.tgz" | cut -d' ' -f1)"
    url="$(python3 "$HERE/r2-weights.py" --bucket fv-weights-staging put-temp "$tmp/bundle.tgz" --key "tmp/r2-copy-bundle-$sha.tgz" --ttl "$CAP_S")"
    log "balance \$$bal; bundle ${sha:0:12} staged"
    start='mkdir -p /srv /opt/fv && cd /srv && (python -m http.server 8000 --directory /srv >/dev/null 2>&1 &) && python -c "
import hashlib, io, os, tarfile, urllib.request
b = urllib.request.urlopen(os.environ[\"FV_BUNDLE_URL\"], timeout=120).read()
assert hashlib.sha256(b).hexdigest() == os.environ[\"FV_BUNDLE_SHA256\"], \"bundle sha256\"
tarfile.open(fileobj=io.BytesIO(b)).extractall(\"/opt/fv\")
" >>/srv/log.txt 2>&1 && bash /opt/fv/r2-copy-volume-pod.sh; [ -f /srv/DONE ] || echo 1 >/srv/DONE; sleep infinity'
    # The keys and the presigned URL go into jq through its environment (env.X), never through argv.
    export FVX_URL="$url" FVX_RWID FVX_RWSEC FVX_ROID FVX_ROSEC FV_COPY_SKIP_COMPARE="${FV_COPY_SKIP_COMPARE:-0}"
    FVX_RWID="$(cat "$CFG/r2_weights_rw_access_key_id")"; FVX_RWSEC="$(cat "$CFG/r2_weights_rw_secret_access_key")"
    FVX_ROID="$(cat "$CFG/r2_weights_ro_access_key_id")"; FVX_ROSEC="$(cat "$CFG/r2_weights_ro_secret_access_key")"
    payload="$(jq -n --arg name "fv-p0-r2copy-$(date -u +%m%d%H%M)" --arg vol "$FV_EU_VOLUME_ID" --arg dc "$FV_EU_DC" --arg start "$start" \
      --arg vcpu "${FV_COPY_VCPU:-16}" --arg sha "$sha" --arg ep "https://$ACCT.r2.cloudflarestorage.com" \
      --arg rev "${FV_COPY_REV:-eu-$(date -u +%Y%m%d)}" --arg jobs "${FV_COPY_JOBS:-3}" --arg files "${FV_COPY_FILES:-6}" --arg window "${FV_COPY_WINDOW:-4}" --arg only "${FV_COPY_ONLY:-}" '{
        name: $name, imageName: "python:3.12-slim", cloudType: "SECURE", computeType: "CPU",
        cpuFlavorIds: ["cpu3c","cpu5c","cpu3g"], cpuFlavorPriority: "availability", vcpuCount: ($vcpu|tonumber),
        containerDiskInGb: 20, networkVolumeId: $vol, volumeMountPath: "/workspace", dataCenterIds: [$dc], ports: ["8000/http"],
        dockerStartCmd: ["/bin/bash","-lc",$start],
        env: {FV_BUNDLE_URL: env.FVX_URL, FV_BUNDLE_SHA256: $sha, FV_WEIGHTS_R2_ENDPOINT: $ep, FV_WEIGHTS_R2_BUCKET: "fv-weights",
              FV_WEIGHTS_R2_ACCESS_KEY_ID: env.FVX_RWID, FV_WEIGHTS_R2_SECRET_ACCESS_KEY: env.FVX_RWSEC, FV_R2_RO_ID: env.FVX_ROID, FV_R2_RO_SECRET: env.FVX_ROSEC,
              FV_COPY_REV: $rev, FV_COPY_JOBS: $jobs, FV_COPY_FILES: $files, FV_COPY_WINDOW: $window, FV_COPY_ONLY: $only, FV_COPY_SKIP_COMPARE: env.FV_COPY_SKIP_COMPARE}
      }')"
    unset FVX_URL FVX_RWID FVX_RWSEC FVX_ROID FVX_ROSEC
    resp="$(rest POST /pods <<<"$payload")" || { log "pod create failed: $(jq -r '.error // .' <<<"$resp" | head -c 300)"; exit 1; }
    id="$(jq -r '.id // empty' <<<"$resp")"
    [[ -n "$id" ]] || { log "pod create failed: $(head -c 300 <<<"$resp")"; exit 1; }
    setsid nohup bash -c "sleep $CAP_S; curl -sS -X DELETE -H @<(printf 'Authorization: Bearer %s\\n' \"\$RUNPOD_API_KEY\") '$API/pods/$id' >/dev/null 2>&1" >/dev/null 2>&1 </dev/null &
    log "pod $id \$$(jq -r '.costPerHr' <<<"$resp")/hr in $FV_EU_DC, backstop pid $! deletes it after ${CAP_S}s; logs https://$id-8000.proxy.runpod.net/"
    echo "$id"
    ;;
  status)
    id="${2:?pod id}"; u="https://$id-8000.proxy.runpod.net"
    curl -sS --max-time 20 "$u/status.tsv" 2>/dev/null || true
    curl -sS --max-time 20 "$u/log.txt" 2>/dev/null | tail -n "${FV_TAIL:-5}" || true
    printf 'DONE=%s\n' "$(curl -sS --max-time 20 --fail "$u/DONE" 2>/dev/null || true)"
    ;;
  fetch)
    id="${2:?pod id}"; out="${3:?dir}"; u="https://$id-8000.proxy.runpod.net"
    mkdir -p "$out"
    for f in log.txt status.tsv inventory.txt wan-listing.txt compare.txt verify.txt record.tsv status-final.txt archive-runs.log archive-wan.log; do
      curl -sS --max-time 60 --fail "$u/$f" -o "$out/$f" 2>/dev/null || true
    done
    for f in $(curl -sS --max-time 20 "$u/" | grep -o 'href="pub-[^"]*\.log"' | cut -d'"' -f2); do curl -sS --max-time 60 --fail "$u/$f" -o "$out/$f" || true; done
    log "fetched into $out"
    ;;
  down)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    id="${2:?pod id}"
    rest DELETE "/pods/$id" >/dev/null 2>&1 || true
    if rest GET "/pods/$id" >/dev/null 2>&1; then log "WARNING: pod $id still listed"; exit 1; fi
    log "pod $id deleted (GET /pods/$id: not found)"
    python3 "$HERE/r2-weights.py" --bucket fv-weights-staging rm-temp --prefix tmp/r2-copy-bundle- || true
    ;;
  *) sed -n '2,25p' "$0"; exit 2 ;;
esac
