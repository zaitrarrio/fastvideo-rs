#!/usr/bin/env bash
# One real GPU worker through the Durable Object path against the gateway path
# (docs/serve/gateway-cloudflare.md §9.7): one h3-turbo worker pod on the EU
# volume, holding a socket to fv-edge-staging AND serving the gateway's
# internal routes, and a local fv-serve gateway run twice over the same pod,
# once per `dispatch` mode; the same serial jobs on both, queue times printed.
#
#   edge-gpu-test.sh up <image>          create the worker pod (price checked first),
#                                        start the backstops, wait for /ping 200
#   edge-gpu-test.sh bench <n>           for each mode: start the gateway, submit n
#                                        jobs one at a time, print queue times; stop it
#   edge-gpu-test.sh down                delete the pod and check it is gone
#
# Money guards: RTX PRO 6000 in EUR-IS-1 only if quoted ≤ FV_GPU_MAX_DPH (2.5 $/h);
# a balance of at least FV_MIN_BALANCE_START (15 $); a wall-clock cap
# FV_GPU_CAP_S (4200 s) enforced twice: on the pod (it deletes itself with its
# pod-scoped RUNPOD_API_KEY) and by a detached loop here; the pod also deletes
# itself after 10 min at 0 % GPU. State: artifacts/edge/gpu.json (no secrets).
#
# Env: RUNPOD_API_KEY; FV_SERVE_BIN (a release fv-serve with http-client, for the
# local gateway); the staging tokens from scripts/serve/cf-edge.sh
# (~/.config/fv-edge-staging); the Cloudflare token (D1 for the local gateway).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/runpod-price.sh
source "$HERE/../gpu/runpod-price.sh"

REST="https://rest.runpod.io/v1"
GQL="https://api.runpod.io/graphql"
OUT="$ROOT/artifacts/edge"
STATEF="$OUT/gpu.json"
EDGE_STATE="${FV_EDGE_STATE:-$HOME/.config/fv-edge-staging}"
GPU="NVIDIA RTX PRO 6000 Blackwell Server Edition"
DC="EUR-IS-1"
VOLUME="jg48s6o1w0"
POOL="gpu-h3"
MAX_DPH="${FV_GPU_MAX_DPH:-2.5}"
CAP_S="${FV_GPU_CAP_S:-4200}"
MIN_START="${FV_MIN_BALANCE_START:-15}"
PORT=18000
LEDGER="$ROOT/artifacts/runpod/serve/ledger.tsv"
umask 077

log() { printf '[%s] %s\n' "$(date -u +%T)" "$*" >&2; }
die() { log "error: $*"; exit 1; }
hdr() { local f; f="$(mktemp)"; printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY" >"$f"; echo "$f"; }
rest() { local h; h="$(hdr)"; curl -sS --max-time 60 -X "$1" -H @"$h" -H 'content-type: application/json' ${3:+-d "$3"} "$REST$2"; local rc=$?; rm -f "$h"; return $rc; }
balance() { local h; h="$(hdr)"; curl -sS -H @"$h" -H 'content-type: application/json' "$GQL" -d '{"query":"{ myself { clientBalance } }"}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"]["myself"]["clientBalance"])'; rm -f "$h"; }
ledger() { mkdir -p "$(dirname "$LEDGER")"; printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }
st() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$STATEF" "$1"; }

# The worker: the image's h3-turbo config, not registered with any gateway,
# its pool's DO socket from the env; a watchdog deletes the pod at the cap or
# after 10 min at 0 % GPU.
# shellcheck disable=SC2016 # runs on the pod
BOOT='set -u
mkdir -p /fvstate
cp /etc/fv/runpod.toml /fv-worker.toml
if grep -q "^\[gateway\]" /fv-worker.toml; then sed -i "/^\[gateway\]/a register = false" /fv-worker.toml; else printf "\n[gateway]\nregister = false\n" >> /fv-worker.toml; fi
export FV_WORKER_ID="${RUNPOD_POD_ID}"
export FV_PUBLIC_BASE_URL="https://${RUNPOD_POD_ID}-8000.proxy.runpod.net"
(
  command -v curl >/dev/null 2>&1 || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends curl; } >/fvstate/watchdog-apt.log 2>&1
  start=$(date +%s); idle=0
  bye() { echo "[watchdog] $1: deleting ${RUNPOD_POD_ID}" >&2; for _ in 1 2 3; do curl -sS --max-time 30 -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" && break; sleep 5; done; }
  while :; do
    sleep 30
    [ $(( $(date +%s) - start )) -ge "$FV_GPU_CAP_S" ] && bye cap
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d " ")
    if [ "${u:-0}" = "0" ]; then idle=$((idle + 30)); else idle=0; fi
    [ "$idle" -ge 600 ] && bye "10 min at 0% GPU"
  done
) &
exec /opt/fastvideo-rs/bin/fv-serve --config /fv-worker.toml'

cmd_up() {
  local image="${1:?image, e.g. ghcr.io/zaitrarrio/fastvideo-rs-serve:h3-turbo-sha-<sha>}"
  [[ ! -s "$STATEF" || -z "$(st pod)" ]] || die "a pod is already recorded in $STATEF: down first"
  local b
  b="$(balance)"
  awk -v b="$b" -v m="$MIN_START" 'BEGIN{exit !(b+0 >= m+0)}' || die "balance \$$b is below \$$MIN_START"
  fv_runpod_price_ok "$GPU" "$MAX_DPH" SECURE "$DC" || die "no $GPU at ≤ \$$MAX_DPH/h in $DC"
  local token do_url
  token="$(cat "$EDGE_STATE/internal_token")"
  do_url="$(cat "$EDGE_STATE/url")"
  local payload
  payload="$(python3 - "$image" "$GPU" "$VOLUME" "$DC" "$BOOT" "$token" "$do_url" "$POOL" "$CAP_S" <<'EOF'
import json, sys, secrets
image, gpu, vol, dc, boot, token, do_url, pool, cap = sys.argv[1:]
sec = lambda n: "{{ RUNPOD_SECRET_%s }}" % n
env = {
    "FV_CF_ACCOUNT_ID": sec("fv_cf_account_id"), "FV_CF_API_TOKEN": sec("fv_cf_api_token"),
    "FV_D1_DATABASE_ID": sec("fv_d1_database_id"), "FV_R2_BUCKET": sec("fv_r2_bucket"),
    "FV_R2_ENDPOINT": sec("fv_r2_endpoint"), "FV_R2_ACCESS_KEY_ID": sec("fv_r2_access_key_id"),
    "FV_R2_SECRET_ACCESS_KEY": sec("fv_r2_secret_access_key"),
    "FV_SERVE_MODE": "http", "FV_SERVE_ROLE": "worker", "FV_INTERNAL_TOKEN": token,
    "FV_URL_SIGNING_KEY": secrets.token_hex(16), "FV_STATE_DIR": "/fvstate", "FV_WEIGHTS": "/workspace/weights",
    "FV_JOBS_HEARTBEAT_S": "10", "RUST_LOG": "info", "FV_GATEWAY_POOL": pool, "FV_DISPATCH_DO_URL": do_url,
    "FV_DISPATCH_CAPACITY": "2", "FV_GPU_CAP_S": cap, "FV_IMAGE_REF": image,
}
print(json.dumps({
    "name": "fv-edge-gpu-h3", "imageName": image, "cloudType": "SECURE", "computeType": "GPU",
    "gpuTypeIds": [gpu], "gpuCount": 1, "containerDiskInGb": 40, "volumeInGb": 0,
    "networkVolumeId": vol, "volumeMountPath": "/workspace", "dataCenterIds": [dc],
    "ports": ["8000/http"], "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [boot], "env": env,
}))
EOF
)"
  local resp pod dph
  resp="$(rest POST /pods "$payload")" || die "create failed: $resp"
  pod="$(python3 -c 'import sys,json; print(json.loads(sys.argv[1]).get("id",""))' "$resp")"
  [[ -n "$pod" ]] || die "create returned no id: $(head -c 300 <<<"$resp")"
  dph="$(python3 -c 'import sys,json; print(json.loads(sys.argv[1]).get("costPerHr",0))' "$resp")"
  local deadline=$(( $(date +%s) + CAP_S ))
  mkdir -p "$OUT"
  python3 -c 'import json,sys; json.dump({"pod": sys.argv[1], "dph": float(sys.argv[2]), "created": int(sys.argv[3]), "deadline": int(sys.argv[4]), "image": sys.argv[5]}, open(sys.argv[6], "w"))' \
    "$pod" "$dph" "$(date +%s)" "$deadline" "$image" "$STATEF"
  ledger "pod-created $pod edge-gpu gpu=$GPU dc=$DC dph=$dph backstop=$(date -u -d "@$deadline" +%FT%TZ)"
  log "pod $pod at \$$dph/h; deadline $(date -u -d "@$deadline" +%T)"
  if awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 > c+0)}'; then
    rest DELETE "/pods/$pod" >/dev/null; ledger "pod-deleted $pod over-cap"; die "\$$dph/h is over the cap: deleted"
  fi
  # Local backstop (the pod has its own).
  setsid bash -c "while [ \$(date +%s) -lt $deadline ]; do sleep 30; done; \
    h=\$(mktemp); printf 'Authorization: Bearer %s\n' \"\$RUNPOD_API_KEY\" > \$h; \
    curl -sS -X DELETE -H @\$h $REST/pods/$pod >/dev/null; rm -f \$h; \
    printf '%s\tpod-deleted $pod edge-gpu backstop\n' \"\$(date -u +%FT%TZ)\" >> '$LEDGER'" </dev/null >/dev/null 2>&1 &
  local url="https://$pod-8000.proxy.runpod.net" t0
  t0=$(date +%s)
  until [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$url/ping")" == "200" ]]; do
    (( $(date +%s) - t0 < 2400 )) || die "no /ping 200 after 40 min"
    sleep 15
  done
  log "worker ready after $(( $(date +%s) - t0 )) s: $url"
}

gateway_toml() {
  local mode="$1" pod="$2" dir="$3"
  cat <<EOF
[server]
bind = "127.0.0.1:$PORT"
state_dir = "$dir/state"
[auth]
mode = "none"
[artifacts]
backend = "local"
local_dir = "$dir/artifacts"
[jobs]
backend = "d1"
progress_interval_ms = 1000
stale_after_s = 900
[engine]
backend = "remote"
[gateway]
tick_s = 2
watch_poll_ms = 500
[[pools]]
id = "$POOL"
kind = "pod"
urls = [$( [[ "$mode" == gateway ]] && printf '"https://%s-8000.proxy.runpod.net"' "$pod" )]
dispatch = "$mode"
$( [[ "$mode" == durable-object ]] && printf 'do_url = "%s"' "$(cat "$EDGE_STATE/url")" )
dispatch_timeout_s = 60
job_timeout_s = 1800
stale_after_s = 180
retries = 1
[[pools.models]]
id = "fasth3"
family = "h3"
recipe = "h3-turbo"
EOF
}

cmd_bench() {
  local n="${1:-5}" pod
  pod="$(st pod)"
  [[ -n "$pod" ]] || die "no pod: up first"
  [[ -x "${FV_SERVE_BIN:-}" ]] || die "FV_SERVE_BIN must be a release fv-serve with http-client"
  local cf acct dbid
  cf="$(tr -d '\n' </root/.config/fv/cf_api_token)"
  local h; h="$(mktemp)"; printf 'Authorization: Bearer %s\n' "$cf" >"$h"
  acct="$(curl -sS -H @"$h" https://api.cloudflare.com/client/v4/accounts | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"][0]["id"])')"
  dbid="$(curl -sS -H @"$h" "https://api.cloudflare.com/client/v4/accounts/$acct/d1/database?name=fv-jobs" | python3 -c 'import sys,json; print(json.load(sys.stdin)["result"][0]["uuid"])')"
  rm -f "$h"
  for mode in gateway durable-object; do
    local dir="$OUT/gpu-gw-$mode"
    rm -rf "$dir" && mkdir -p "$dir/state" "$dir/artifacts"
    gateway_toml "$mode" "$pod" "$dir" >"$dir/gateway.toml"
    FV_INTERNAL_TOKEN="$(cat "$EDGE_STATE/internal_token")" FV_CF_ACCOUNT_ID="$acct" FV_CF_API_TOKEN="$cf" FV_D1_DATABASE_ID="$dbid" \
      FV_URL_SIGNING_KEY="edge-gpu-bench" NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost" RUST_LOG=info \
      "$FV_SERVE_BIN" --config "$dir/gateway.toml" >"$dir/gateway.log" 2>&1 &
    local gpid=$!
    local t0=$(date +%s)
    until curl -s --noproxy '*' "http://127.0.0.1:$PORT/fv/v1/status" | python3 -c 'import sys,json; sys.exit(0 if any(p.get("available") for p in json.load(sys.stdin).get("pools",[])) else 1)' 2>/dev/null; do
      (( $(date +%s) - t0 < 120 )) || { kill $gpid; die "$mode: pool never available (see $dir/gateway.log)"; }
      sleep 1
    done
    log "$mode: gateway up; $n jobs one at a time (+1 warm-up)"
    python3 - "$PORT" "$n" "$mode" <<'EOF' | tee "$dir/results.txt"
import json, sys, time, urllib.request
port, n, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
base = f"http://127.0.0.1:{port}"
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def call(method, path, body=None):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body else None, headers={"content-type": "application/json"})
    with op.open(req, timeout=60) as r:
        return json.loads(r.read())
rows = []
for i in range(n + 1):
    t0 = time.time()
    j = call("POST", "/fv/v1/jobs", {"model": "fasth3", "prompt": f"a red fox running through snow, take {i}", "seed": i})
    submit = time.time() - t0
    jid = j["id"]
    while True:
        j = call("GET", f"/fv/v1/jobs/{jid}")
        if j["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.5)
    q = (j.get("metrics") or {}).get("queue_s")
    run = (j.get("metrics") or {}).get("run_s")
    print(json.dumps({"mode": mode, "i": i, "status": j["status"], "queue_s": q, "run_s": run, "submit_s": round(submit, 3)}), flush=True)
    if i > 0 and j["status"] == "succeeded":
        rows.append(q)
rows.sort()
if rows:
    print(f"SUMMARY {mode}: n={len(rows)} queue p50={rows[len(rows)//2]:.3f} mean={sum(rows)/len(rows):.3f} max={rows[-1]:.3f}", flush=True)
EOF
    kill $gpid 2>/dev/null || true
    wait $gpid 2>/dev/null || true
  done
}

cmd_down() {
  local pod
  pod="$(st pod)"
  [[ -n "$pod" ]] || { log "no pod recorded"; return; }
  rest DELETE "/pods/$pod" >/dev/null || true
  ledger "pod-deleted $pod edge-gpu"
  sleep 5
  local code
  code="$(local h; h="$(hdr)"; curl -s -o /dev/null -w '%{http_code}' -H @"$h" "$REST/pods/$pod"; rm -f "$h")"
  log "GET /pods/$pod → $code (404 = gone)"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); d["deleted"]=True; d["pod_last"]=d.pop("pod"); json.dump(d, open(sys.argv[1],"w"))' "$STATEF"
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  bench) shift; cmd_bench "$@" ;;
  down) shift; cmd_down "$@" ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
