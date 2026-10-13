#!/usr/bin/env bash
# Publish Hub weight trees to the R2 weights store from ONE Runpod CPU pod (docs/ops/r2-weights.md §3), so no
# weights pass through the machine that runs this script. The pod streams each tree Hub -> R2 with
# `r2-weights.py publish-hub` (add-only, every file checked against the Hub's LFS SHA-256 / git blob before its
# upload completes), re-reads every byte with `verify --deep --record`, and prints `record` rows.
#
#   r2-publish-hub.sh up DEST...        create the pod (+ a detached wall-clock backstop that deletes it)
#   r2-publish-hub.sh status <pod-id>   status.tsv, the log tail and DONE (through the Runpod proxy)
#   r2-publish-hub.sh fetch <pod-id> DIR  logs, verify.txt and record.tsv into DIR
#   r2-publish-hub.sh down <pod-id>     delete the pod, check it is gone, drop the staged bundle
#
# DEST: rows of weights-manifest.tsv + weights-revisions.tsv (fetcher hub), e.g. fasth3-trim-8step (first used
# 2026-10-13 for FastH3 Trim, docs/ports/h3-trim.md). The pod: python:3.12-slim, secure CPU, no volume, any data
# centre unless FV_PUB_DC is set. Credentials: /root/.config/fv/r2_weights_rw_* go into the pod's env only (never
# argv, never printed); HF_TOKEN from the environment for gated repos. The scripts reach the pod as a tarball in
# the staging bucket (fv-weights-staging, tmp/), fetched through a presigned URL and checked by SHA-256.
#
# Env: RUNPOD_API_KEY; FV_POD_CAP_S (backstop, default 5400); FV_MIN_BALANCE (default 12: refuse below it + 2);
# FV_PUB_VCPU (default 16); FV_PUB_DC; FV_PUB_FILES / FV_PUB_WINDOW (r2-weights.py --files / --window).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
CFG="${FV_CONFIG_DIR:-/root/.config/fv}"
CAP_S="${FV_POD_CAP_S:-5400}"
MIN_BALANCE="${FV_MIN_BALANCE:-12}"
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

# The script the pod runs (from the bundle): publish each dest, verify them deeply, record.
pod_script() {
  cat <<'PODEOF'
#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
SRV=/srv; P="python3 /opt/fv/r2-weights.py"
cd "$SRV" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >>$SRV/log.txt; }
step() { local name="$1" t0=$SECONDS rc; shift; "$@"; rc=$?; printf '%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS - t0))" >>$SRV/status.tsv; return "$rc"; }
log "start: $FV_PUB_DESTS"
for d in $FV_PUB_DESTS; do
  f="pub-$(echo "$d" | tr / -).log"
  step "publish $d" $P publish-hub "$d" --files "${FV_PUB_FILES:-4}" --window "${FV_PUB_WINDOW:-4}" >>"$SRV/$f" 2>&1 && log "published $d" || log "FAILED publish $d"
done
step "verify" $P verify --deep --record $FV_PUB_DESTS >>$SRV/verify.txt 2>&1
$P record >$SRV/record.tsv 2>>$SRV/log.txt
bad="$(awk -F'\t' '$2 != 0' $SRV/status.tsv | wc -l)"
log "end: $bad step(s) failed"
echo $(( bad > 0 ? 1 : 0 )) >$SRV/DONE
PODEOF
}

cmd="${1:-}"
case "$cmd" in
  up)
    shift
    (( $# >= 1 )) || { log "usage: $0 up DEST..."; exit 2; }
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    for d in "$@"; do
      python3 - "$HERE" "$d" <<'PYEOF' || { log "$d: no Hub row (weights-manifest.tsv + weights-revisions.tsv)"; exit 2; }
import importlib.util, sys
spec = importlib.util.spec_from_file_location("r2w", sys.argv[1] + "/r2-weights.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.exit(0 if sys.argv[2] in m.hub_rows() else 1)
PYEOF
    done
    bal="$(balance)"
    awk -v b="$bal" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+2)}' || { log "balance \$$bal too close to the \$$MIN_BALANCE floor"; exit 1; }
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    pod_script >"$tmp/r2-publish-hub-pod.sh"
    cp "$HERE"/{r2-weights.py,r2-fetch.py,weights-manifest.tsv,weights-revisions.tsv,weights-sha256.tsv,weights-r2.tsv,verify-weights.sh} "$tmp/"
    tar czf "$tmp/bundle.tgz" -C "$tmp" r2-publish-hub-pod.sh r2-weights.py r2-fetch.py weights-manifest.tsv weights-revisions.tsv weights-sha256.tsv weights-r2.tsv verify-weights.sh
    sha="$(sha256sum "$tmp/bundle.tgz" | cut -d' ' -f1)"
    url="$(python3 "$HERE/r2-weights.py" --bucket fv-weights-staging put-temp "$tmp/bundle.tgz" --key "tmp/r2-publish-bundle-$sha.tgz" --ttl "$CAP_S")"
    log "balance \$$bal; bundle ${sha:0:12} staged; dests: $*"
    start='mkdir -p /srv /opt/fv && cd /srv && (python -m http.server 8000 --directory /srv >/dev/null 2>&1 &) && python -c "
import hashlib, io, os, tarfile, urllib.request
b = urllib.request.urlopen(os.environ[\"FV_BUNDLE_URL\"], timeout=120).read()
assert hashlib.sha256(b).hexdigest() == os.environ[\"FV_BUNDLE_SHA256\"], \"bundle sha256\"
tarfile.open(fileobj=io.BytesIO(b)).extractall(\"/opt/fv\")
" >>/srv/log.txt 2>&1 && bash /opt/fv/r2-publish-hub-pod.sh; [ -f /srv/DONE ] || echo 1 >/srv/DONE; sleep infinity'
    # The keys and the presigned URL go into jq through its environment (env.X), never through argv.
    export FVX_URL="$url" FVX_RWID FVX_RWSEC FVX_HF="${HF_TOKEN:-}"
    FVX_RWID="$(cat "$CFG/r2_weights_rw_access_key_id")"; FVX_RWSEC="$(cat "$CFG/r2_weights_rw_secret_access_key")"
    payload="$(jq -n --arg name "fv-r2pub-$(date -u +%m%d%H%M)" --arg dc "${FV_PUB_DC:-}" --arg start "$start" \
      --arg vcpu "${FV_PUB_VCPU:-16}" --arg sha "$sha" --arg ep "https://$ACCT.r2.cloudflarestorage.com" --arg dests "$*" \
      --arg files "${FV_PUB_FILES:-4}" --arg window "${FV_PUB_WINDOW:-4}" '{
        name: $name, imageName: "python:3.12-slim", cloudType: "SECURE", computeType: "CPU",
        cpuFlavorIds: ["cpu3c","cpu5c","cpu3g"], cpuFlavorPriority: "availability", vcpuCount: ($vcpu|tonumber),
        containerDiskInGb: 20, ports: ["8000/http"], dockerStartCmd: ["/bin/bash","-lc",$start],
        env: {FV_BUNDLE_URL: env.FVX_URL, FV_BUNDLE_SHA256: $sha, FV_WEIGHTS_R2_ENDPOINT: $ep, FV_WEIGHTS_R2_BUCKET: "fv-weights",
              FV_WEIGHTS_R2_ACCESS_KEY_ID: env.FVX_RWID, FV_WEIGHTS_R2_SECRET_ACCESS_KEY: env.FVX_RWSEC, HF_TOKEN: env.FVX_HF,
              FV_PUB_DESTS: $dests, FV_PUB_FILES: $files, FV_PUB_WINDOW: $window}
      } + (if $dc == "" then {} else {dataCenterIds: [$dc]} end)')"
    unset FVX_URL FVX_RWID FVX_RWSEC FVX_HF
    resp="$(rest POST /pods <<<"$payload")" || { log "pod create failed: $(jq -r '.error // .' <<<"$resp" | head -c 300)"; exit 1; }
    id="$(jq -r '.id // empty' <<<"$resp")"
    [[ -n "$id" ]] || { log "pod create failed: $(head -c 300 <<<"$resp")"; exit 1; }
    setsid nohup bash -c "sleep $CAP_S; curl -sS -X DELETE -H @<(printf 'Authorization: Bearer %s\\n' \"\$RUNPOD_API_KEY\") '$API/pods/$id' >/dev/null 2>&1" >/dev/null 2>&1 </dev/null &
    log "pod $id \$$(jq -r '.costPerHr' <<<"$resp")/hr, backstop pid $! deletes it after ${CAP_S}s; logs https://$id-8000.proxy.runpod.net/"
    echo "$id"
    ;;
  status)
    id="${2:?pod id}"; u="https://$id-8000.proxy.runpod.net"
    curl -sS --max-time 20 "$u/status.tsv" 2>/dev/null || true
    curl -sS --max-time 20 "$u/log.txt" 2>/dev/null | tail -n "${FV_TAIL:-5}" || true
    for f in $(curl -sS --max-time 20 "$u/" 2>/dev/null | grep -oE 'href="pub-[^"]*\.log"' | cut -d'"' -f2); do
      printf '%s: ' "$f"; curl -sS --max-time 20 "$u/$f" 2>/dev/null | tail -n 1 || true
    done
    printf 'DONE=%s\n' "$(curl -sS --max-time 20 --fail "$u/DONE" 2>/dev/null || true)"
    ;;
  fetch)
    id="${2:?pod id}"; out="${3:?dir}"; u="https://$id-8000.proxy.runpod.net"
    mkdir -p "$out"
    for f in log.txt status.tsv verify.txt record.tsv; do curl -sS --max-time 60 --fail "$u/$f" -o "$out/$f" 2>/dev/null || true; done
    for f in $(curl -sS --max-time 20 "$u/" | grep -oE 'href="pub-[^"]*\.log"' | cut -d'"' -f2); do curl -sS --max-time 60 --fail "$u/$f" -o "$out/$f" || true; done
    log "fetched into $out"
    ;;
  down)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    id="${2:?pod id}"
    rest DELETE "/pods/$id" >/dev/null 2>&1 || true
    if rest GET "/pods/$id" >/dev/null 2>&1; then log "WARNING: pod $id still listed"; exit 1; fi
    log "pod $id deleted (GET /pods/$id: not found)"
    python3 "$HERE/r2-weights.py" --bucket fv-weights-staging rm-temp --prefix tmp/r2-publish-bundle- || true
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
