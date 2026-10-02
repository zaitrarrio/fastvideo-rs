#!/bin/bash
# Family Durable Objects on one GPU (docs/serve/dispatch-do-family.md §14):
# runs ON the e2e pod (scripts/serve/e2e/pod.sh run 'bash /e2e/scripts/serve/e2e/do-family-pod.sh <stage>').
# One fv-serve worker serving two families (wan: FastWan2.2-TI2V-5B, sfwan:
# LongLive-1.3B) through one arbiter, and an fv-serve gateway, both on
# loopback; the family objects are the staging Worker. The binaries come in
# the bundle under /e2e/dof/ (fv-serve, director_client).
#
#   setup <do_url>                 configs (FV_INTERNAL_TOKEN must be in the env)
#   worker do|classic              (re)start the worker: `do` = family sockets,
#                                  fragmented MP4 + direct uploads; `classic` = the
#                                  gateway posts to it, faststart + its own R2 upload
#   gateway do|classic             (re)start the gateway with the matching pools
#   burst <n> [models]             n jobs per model at once; queue / result times
#   kill                           SIGKILL the worker (the drill), then `worker do`
#   director <seconds>             a LongLive director session through the gateway
#   logs                           worker/gateway log extracts
set -u
D=/e2e/dof; WK=/root/dof; W=/workspace/weights
mkdir -p $WK
log() { echo "$(date -u +%FT%TZ) | $*" | tee -a $WK/progress.log; }
WAN=fastwan22-ti2v-5b; LL=longlive-1.3b

models_toml() {
  cat <<TOML
[[models]]
id = "$WAN"
family = "wan"
recipe = "$WAN"
weights = "$W/fastwan22-ti2v-5b"
resident = true

[[models]]
id = "$LL"
family = "wan"
recipe = "sfwan21-1.3b"
weights = "$W/sfwan21-1.3b"
resident = true
longlive = "$W/longlive-1.3b-safetensors"
TOML
}

start() { # name port cfg extra-env...
  local name=$1 cfg=$2; shift 2
  [ -s $WK/$name.pid ] && kill -TERM "$(cat $WK/$name.pid)" 2>/dev/null && sleep 3 && kill -KILL "$(cat $WK/$name.pid)" 2>/dev/null
  echo "=== $(date -u +%FT%TZ) start $cfg" >> $WK/$name.log
  env -u RUNPOD_POD_ID -u RUNPOD_PUBLIC_IP FV_JOBS_HEARTBEAT_S=10 FV_WEIGHTS=$W FV_TAE_DIR=$W/auxiliary/tae FASTVIDEO_TAE_DIR=$W/auxiliary/tae \
    RUST_LOG="info,fastvideo_serve_kit::events=debug,fastvideo_serve::gate=debug" "$@" \
    $D/fv-serve --config $cfg >> $WK/$name.log 2>&1 &
  echo $! > $WK/$name.pid
}

ready() { # url tries
  local t0; t0=$(date +%s)
  for _ in $(seq 1 "${2:-120}"); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1/ping")" = 200 ] && { log "ready $1 in $(( $(date +%s) - t0 ))s"; return 0; }
    sleep 3
  done
  log "not ready: $1"; return 1
}

case "${1:?stage}" in
setup)
  DO_URL="${2:?do url}"
  [ -s /e2e/serve.pid ] && kill -TERM "$(cat /e2e/serve.pid)" 2>/dev/null
  chmod +x $D/fv-serve $D/director_client
  printf '%s' "$DO_URL" > $WK/do_url
  ls -d $W/fastwan22-ti2v-5b $W/sfwan21-1.3b $W/longlive-1.3b-safetensors $W/auxiliary/tae
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  for mode in do classic; do
    {
      cat <<TOML
[server]
bind = "127.0.0.1:8100"
state_dir = "$WK/wstate-$mode"
role = "worker"
worker_id = "gpu-$(hostname | tr -cd 'a-z0-9-' | cut -c1-20)"
public_base_url = "http://127.0.0.1:8100"
[auth]
mode = "keys"
[artifacts]
backend = "auto"
[jobs]
backend = "auto"
progress_interval_ms = 1000
[engine]
backend = "cuda"
post_encoder = "auto"
mp4_fragmented = $([ $mode = do ] && echo true || echo false)
TOML
      models_toml
      cat <<TOML
[protocols]
fal = true
fal_director = true
fal_apps = ["fastvideo/longlive"]
openai_videos = false
fastwan = false
minimax = false
ltx = false
reactor = false
native = true
[director]
encoder = "auto"
[webrtc]
udp_port = 40110
tcp_port = 40100
public_ip = "127.0.0.1"
[gateway]
register = false
TOML
      if [ $mode = do ]; then
        printf '[dispatch]\ndo_url = "%s"\nfamilies = ["wan", "sfwan"]\ncapacity = 1\nsessions = 1\nstatus_s = 5\ndirect_upload = true\nupload_part_mib = 5\n' "$DO_URL"
      fi
    } > $WK/worker-$mode.toml
    {
      cat <<TOML
[server]
bind = "127.0.0.1:8200"
state_dir = "$WK/gstate-$mode"
[auth]
mode = "keys"
[jobs]
backend = "d1"
progress_interval_ms = 500
[engine]
backend = "remote"
[protocols]
fal = true
fal_director = true
fal_apps = ["fastvideo/longlive"]
openai_videos = false
fastwan = false
minimax = false
ltx = false
reactor = false
native = true
[gateway]
tick_s = 2
watch_poll_ms = 200
TOML
      for pool in "p-wan wan $WAN" "p-ll sfwan $LL"; do
        set -- $pool
        if [ $mode = do ]; then
          printf '\n[[pools]]\nid = "%s"\nkind = "pod"\ndispatch = "durable-object"\ndo_url = "%s"\nfamily = "%s"\nretries = 1\n' "$1" "$DO_URL" "$2"
        else
          printf '\n[[pools]]\nid = "%s"\nkind = "pod"\nurls = ["http://127.0.0.1:8100"]\nretries = 1\n' "$1"
        fi
        models_toml | awk -v m="$3" 'BEGIN{RS="";ORS="\n\n"} $0 ~ "id = \""m"\"" {gsub(/\[\[models\]\]/, "[[pools.models]]"); print}'
      done
    } > $WK/gateway-$mode.toml
  done
  # A key for the clients (its hash on the gateway).
  python3 -c 'import secrets; print("fvk-" + secrets.token_hex(16), end="")' > $WK/key
  printf '%s' "$(sha256sum < $WK/key | cut -d' ' -f1)" > $WK/key.sha
  python3 - <<'PY' > $WK/prompts.txt
import json
for l in open('/workspace/weights/longlive-1.3b/prompts/interactive_example.jsonl'):
    l = l.strip()
    if not l:
        continue
    v = json.loads(l)
    for p in v.get('prompts') or v.get('prompt_list') or [v.get('prompt') or v.get('text')]:
        print(p.replace('\n', ' '))
    break
PY
  log "setup done ($DO_URL)"
  ;;
worker)
  mode=${2:-do}
  start worker $WK/worker-$mode.toml FV_SERVE_ROLE=worker
  ready http://127.0.0.1:8100 ${READY_TRIES:-200}
  ;;
gateway)
  mode=${2:-do}
  start gateway $WK/gateway-$mode.toml FV_API_KEYS="$(cat $WK/key.sha)" FV_ADMIN_TOKEN=dof-admin \
    FV_R2_BUCKET= FV_R2_ENDPOINT= FV_R2_ACCESS_KEY_ID= FV_R2_SECRET_ACCESS_KEY=
  ready http://127.0.0.1:8200 60
  ;;
burst)
  n=${2:-4}; ms=${3:-$WAN,$LL}
  FV_KEY="$(cat $WK/key)" python3 - "$n" "$ms" <<'PY' | tee -a $WK/burst.log
import json, sys, time, threading, urllib.request
from datetime import datetime
n, models = int(sys.argv[1]), sys.argv[2].split(',')
import os
key = os.environ['FV_KEY']; G = 'http://127.0.0.1:8200'
def call(m, path, body=None):
    r = urllib.request.Request(G + path, method=m, data=json.dumps(body).encode() if body else None,
                               headers={'authorization': 'Bearer ' + key, 'content-type': 'application/json'})
    with urllib.request.urlopen(r, timeout=60) as x:
        return json.load(x)
import re
def ts(s):
    if not s:
        return None
    s = re.sub(r'(\.\d{6})\d+', r'\1', s.replace('Z', '+00:00'))
    return datetime.fromisoformat(s).timestamp()
res = []
def one(model, i):
    body = {'model': model, 'prompt': f'a red fox running through snow, shot {i}', 'seed': 1000 + i}
    if 'longlive' in model:
        body.update({'size': '832x480', 'seconds': 3})
    else:
        body.update({'size': '1280x704', 'seconds': 3})
    t0 = time.time()
    try:
        j = call('POST', '/fv/v1/jobs', body)
    except Exception as e:
        res.append({'model': model, 'error': f'submit: {e}'}); return
    sub = time.time() - t0
    while True:
        v = call('GET', f"/fv/v1/jobs/{j['id']}")
        if v.get('status') in ('succeeded', 'failed', 'cancelled'):
            break
        time.sleep(0.1)
    seen = time.time() - t0
    c, s, d, f = ts(v.get('created_at')), ts(v.get('started_at')), ts(v.get('dispatched_at')), ts(v.get('completed_at'))
    art = (v.get('artifacts') or v.get('outputs') or [{}])[0] if v.get('status') == 'succeeded' else {}
    res.append({'model': model, 'id': j['id'], 'status': v.get('status'), 'submit_s': round(sub, 3), 'client_result_s': round(seen, 3),
                'queue_s': round(s - c, 3) if s and c else None, 'run_s': round(f - s, 3) if f and s else None,
                'start': s, 'end': f, 'bytes': art.get('bytes'), 'error': (v.get('error') or {}).get('message')})
th = [threading.Thread(target=one, args=(m, i)) for i in range(n) for m in models]
for t in th: t.start()
for t in th: t.join()
res.sort(key=lambda r: r.get('start') or 0)
for r in res: print(json.dumps(r))
ok = [r for r in res if r.get('status') == 'succeeded']
spans = sorted((r['start'], r['end'], r['model']) for r in ok if r.get('start') and r.get('end'))
overlap = sum(1 for a, b in zip(spans, spans[1:]) if b[0] < a[1] - 0.05)
for m in models:
    q = sorted(r['queue_s'] for r in ok if r['model'] == m and r['queue_s'] is not None)
    print(json.dumps({'model': m, 'ok': sum(1 for r in ok if r['model'] == m), 'of': n, 'queue_min': q[0] if q else None, 'queue_max': q[-1] if q else None}))
print(json.dumps({'summary': True, 'jobs': len(res), 'succeeded': len(ok), 'overlapping_runs': overlap}))
PY
  ;;
kill)
  log "SIGKILL worker $(cat $WK/worker.pid)"
  kill -KILL "$(cat $WK/worker.pid)"
  ;;
director)
  s=${2:-20}
  $D/director_client --url http://127.0.0.1:8200 --app fastvideo/longlive --key "$(cat $WK/key)" --seconds "$s" \
    --switch-at "$(( s / 2 ))" --prompts $WK/prompts.txt --out $WK/director 2>&1 | tail -25
  ls -la $WK/director | head
  ;;
logs)
  grep -hE "arbiter|session|upload: committed|took a pushed|refused|output: finalize|output stored|reconnect|connected to the dispatcher|lost|fenced" $WK/worker.log | tail -${2:-60}
  echo ---; grep -hE "admitted|family object|edge|WARN|ERROR" $WK/gateway.log | tail -30
  ;;
esac
