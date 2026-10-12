#!/usr/bin/env bash
# Measure R2 fetch throughput from ONE small Runpod CPU pod (docs/ops/r2-weights.md §5.4).
#
#   r2-fetch-bench.sh up [DC...]        create the pod (default: US data centres) + a detached DELETE backstop
#   r2-fetch-bench.sh rerun <pod-id>    the same pod again with the current scripts and FV_BENCH_* (PATCH: new env, the
#                                       container restarts; save the results so far with `status` first)
#   r2-fetch-bench.sh status <pod-id>   results so far (results.jsonl, log tail, DONE)
#   r2-fetch-bench.sh down <pod-id>     delete the pod and check it is gone
#
# The pod (python:3.12-slim, secure CPU, FV_BENCH_VCPU vCPUs, default 16) runs r2-fetch-bench.py with the READ-ONLY
# key pair from /root/.config/fv/r2_weights_ro_* in its env (never printed, never on a command line): it presigns
# its own GETs, reads the trees of FV_BENCH_TREES onto its container disk and never writes to R2. The scripts ride
# in the pod's env (base64 tarball, checked by SHA-256). Results are served at https://<pod>-8000.proxy.runpod.net/.
#
# Env: RUNPOD_API_KEY; FV_BENCH_CAP_S (backstop, default 2700 = 45 min); FV_MIN_BALANCE (default 20: the pod is
# refused below it + $2); FV_BENCH_TREES (default fastwan22-ti2v-5b at its recorded rev), FV_BENCH_RAW,
# FV_BENCH_RAW_S, FV_BENCH_FETCH (see r2-fetch-bench.py).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
CFG="${FV_CONFIG_DIR:-/root/.config/fv}"
CAP_S="${FV_BENCH_CAP_S:-2700}"
MIN_BALANCE="${FV_MIN_BALANCE:-20}"
ACCT="${FV_CF_ACCOUNT_ID:-0cd06fd37ee4e07de370821cb3852a8a}"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
rest() { # method path: a body (if any) comes on stdin, never on the command line
  local method="$1" path="$2"
  if [[ "$method" == POST || "$method" == PATCH ]]; then
    curl -sS --fail-with-body -X "$method" -H @<(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY") -H 'content-type: application/json' --data-binary @- "$API$path"
  else
    curl -sS --fail-with-body -X "$method" -H @<(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY") "$API$path"
  fi
}
balance() { curl -sS -H @<(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY") -H 'content-type: application/json' "$GQL" -d '{"query":"{ myself { clientBalance } }"}' | jq -r '.data.myself.clientBalance // empty'; }
rev_of() { awk -F'\t' -v d="$1" '$1 == d {print $2}' "$HERE/weights-r2.tsv"; }

# The pod's create body into $payload (the keys go into jq through its environment, never argv). Uses ${dcs[@]}.
mkpayload() {
  local trees b64 sha start
  trees="${FV_BENCH_TREES:-fastwan22-ti2v-5b@$(rev_of fastwan22-ti2v-5b)}"
  b64="$(tar czf - -C "$HERE" r2-fetch.py r2-fetch-bench.py | base64 -w0)"
  sha="$(printf %s "$b64" | base64 -d | sha256sum | cut -d' ' -f1)"
  start='mkdir -p /srv /opt/fv && (python -m http.server 8000 --directory /srv >/dev/null 2>&1 &) && python -c "
import base64, hashlib, io, os, tarfile
b = base64.b64decode(os.environ[\"FV_BENCH_B64\"])
assert hashlib.sha256(b).hexdigest() == os.environ[\"FV_BENCH_SHA256\"], \"bundle sha256\"
tarfile.open(fileobj=io.BytesIO(b)).extractall(\"/opt/fv\")
" >>/srv/log.txt 2>&1 && python /opt/fv/r2-fetch-bench.py >>/srv/log.txt 2>&1; sleep infinity'
  export FVX_ROID FVX_ROSEC FVX_B64="$b64"
  FVX_ROID="$(cat "$CFG/r2_weights_ro_access_key_id")"; FVX_ROSEC="$(cat "$CFG/r2_weights_ro_secret_access_key")"
  payload="$(jq -n --arg name "fv-r2bench-$(date -u +%m%d%H%M)" --arg start "$start" --arg vcpu "${FV_BENCH_VCPU:-16}" --arg sha "$sha" \
    --arg ep "https://$ACCT.r2.cloudflarestorage.com" --arg trees "$trees" --argjson dcs "$(printf '%s\n' "${dcs[@]}" | jq -R . | jq -s 'map(select(length > 0))')" \
    --arg raw "${FV_BENCH_RAW-32:64 64:64 128:64 64:256 128:256 64:64:h}" --arg raws "${FV_BENCH_RAW_S:-20}" \
    --arg fetch "${FV_BENCH_FETCH:-32:1 32:1:--no-full-hash 64:1:--no-full-hash 64:4:--no-full-hash 128:4:--no-full-hash 128:2:--no-full-hash 64:4}" '{
      name: $name, imageName: "python:3.12-slim", cloudType: "SECURE", computeType: "CPU",
      cpuFlavorIds: ["cpu3c","cpu5c","cpu3g"], cpuFlavorPriority: "availability", vcpuCount: ($vcpu|tonumber),
      containerDiskInGb: 80, dataCenterIds: $dcs, dataCenterPriority: "availability", ports: ["8000/http"],
      dockerStartCmd: ["/bin/bash","-lc",$start],
      env: {FV_BENCH_B64: env.FVX_B64, FV_BENCH_SHA256: $sha, FV_WEIGHTS_R2_ENDPOINT: $ep, FV_WEIGHTS_R2_BUCKET: "fv-weights",
            FV_R2_RO_ID: env.FVX_ROID, FV_R2_RO_SECRET: env.FVX_ROSEC, FV_BENCH_TREES: $trees, FV_BENCH_RAW: $raw, FV_BENCH_RAW_S: $raws, FV_BENCH_FETCH: $fetch}
    }')"
  unset FVX_ROID FVX_ROSEC FVX_B64
}

case "${1:-}" in
  up)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    shift
    dcs=("$@"); [[ ${#dcs[@]} -gt 0 ]] || dcs=(US-TX-3 US-IL-1 US-KS-2 US-GA-2 US-CA-2 US-WA-1 US-NC-1 US-DE-1 US-TX-4)
    bal="$(balance)"
    awk -v b="$bal" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+2)}' || { log "balance \$$bal too close to \$$MIN_BALANCE"; exit 1; }
    mkpayload
    resp="$(rest POST /pods <<<"$payload")" || { log "pod create failed: $(jq -r '.error // .' <<<"$resp" 2>/dev/null | head -c 300)"; exit 1; }
    id="$(jq -r '.id // empty' <<<"$resp")"
    [[ -n "$id" ]] || { log "pod create failed: $(head -c 300 <<<"$resp")"; exit 1; }
    setsid nohup bash -c "sleep $CAP_S; curl -sS -X DELETE -H @<(printf 'Authorization: Bearer %s\\n' \"\$RUNPOD_API_KEY\") '$API/pods/$id' >/dev/null 2>&1" >/dev/null 2>&1 </dev/null &
    log "pod $id \$$(jq -r '.costPerHr' <<<"$resp")/hr in $(jq -r '.machine.dataCenterId // .dataCenterId // "?"' <<<"$resp"), balance \$$bal; backstop pid $! deletes it after ${CAP_S}s; https://$id-8000.proxy.runpod.net/"
    echo "$id"
    ;;
  rerun)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    id="${2:?pod id}"; dcs=(); mkpayload
    rest PATCH "/pods/$id" <<<"$(jq -c '{env, dockerStartCmd}' <<<"$payload")" >/dev/null || { log "update of $id failed"; exit 1; }
    log "pod $id updated: the bench runs again with the current scripts"
    ;;
  status)
    id="${2:?pod id}"; u="https://$id-8000.proxy.runpod.net"
    curl -sS --max-time 20 "$u/results.jsonl" 2>/dev/null || true
    curl -sS --max-time 20 "$u/log.txt" 2>/dev/null | tail -n "${FV_TAIL:-3}" || true
    printf 'DONE=%s\n' "$(curl -sS --max-time 20 --fail "$u/DONE" 2>/dev/null || true)"
    ;;
  down)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    id="${2:?pod id}"
    rest DELETE "/pods/$id" >/dev/null 2>&1 || true
    if rest GET "/pods/$id" >/dev/null 2>&1; then log "WARNING: pod $id still listed"; exit 1; fi
    log "pod $id deleted (GET /pods/$id: not found)"
    ;;
  *) sed -n '2,19p' "$0"; exit 2 ;;
esac
