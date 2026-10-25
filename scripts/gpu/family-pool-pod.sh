#!/usr/bin/env bash
# Runs ON the family-pool validation pod (scripts/gpu/family-pool-validate.sh; docs/control/README.md §4 "Family
# templates"). One RTX PRO 6000 on the serve image; the fv-serve binary under test and the shipped H3 family config
# (configs/serve/runpod-family-h3.toml, unchanged) come in the bundle; the weights from the R2 weights store (plan.json,
# presigned GETs) onto the container disk:
#
#   1. fetch h3-base, the FastH3 LoRA, fasth3-v2-nvfp4 and auxiliary/tae (the trees of the variants exercised here;
#      the family's other trees are not fetched, so their models are not requested);
#   2. start ONE fv-serve process with the family config (fasth3 resident, every other variant `resident = false`,
#      `[engine] swap = true`), wait for /ping 200;
#   3. one after the other through the native API (480p, seed 1, the same prompt): fasth3 (h3-turbo, resident), sol-h3
#      (h3-max: swaps the turbo DiT out), sol-h3 again (resident: the generation alone), fasth3-v2-nvfp4 (an NVFP4
#      release without a technique profile: swaps across profiles), fasth3 again (back to turbo: same output as the
#      first run?);
#   4. each job's JSON, its MP4, the fv-serve log ("model resident" / "loading" lines) and a summary.
#
# /srv is served on :8000 (Runpod proxy): log.txt, status.tsv (step, rc, seconds), jobs.tsv, summary.json, serve.log,
# the clips and DONE (0 = every step ok). Nothing is written to /workspace.
set -u
SRV=/srv
R=/fvw/root
mkdir -p "$SRV" "$R" /fvstate
cd "$SRV" || exit 1
# The in-pod self-delete (scripts/gpu/lib/self-delete.sh, the same text; checked by
# scripts/gpu/tests/self-delete.test.sh).
fv_self_delete() {
  # GraphQL podTerminate: the pod-scoped key may not DELETE through REST (HTTP 403,
  # docs/adr/0023-lab-report-and-ci-bridge.md §4). Success: HTTP 200 and no "errors".
  # The key reaches curl on stdin (-H @-), never in its argv; the answer is logged, then removed.
  _fvsd_code=$(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY" | curl -sS --max-time 30 -o "$1.resp" \
    -w '%{http_code}' -X POST https://api.runpod.io/graphql -H 'Content-Type: application/json' -H @- \
    --data "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" 2>>"$1")
  { cat "$1.resp"; echo " http=$_fvsd_code"; } >>"$1" 2>/dev/null
  _fvsd_ok=1
  [ "$_fvsd_code" = 200 ] && ! grep -q '"errors"' "$1.resp" 2>/dev/null && _fvsd_ok=0
  rm -f "$1.resp"
  return "$_fvsd_ok"
}
log() { echo "$(date -u +%FT%TZ) $*" >>$SRV/log.txt; }
step() {
  local name="$1" t0=$SECONDS rc
  shift
  log "begin $name"
  "$@"
  rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS - t0))" >>$SRV/status.tsv
  log "end $name rc=$rc $((SECONDS - t0))s"
  return "$rc"
}
nvidia-smi >$SRV/nvidia-smi.txt 2>&1
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [[ "${used:-0}" -gt 4096 ]]; then
  log "GPU busy before start: ${used} MiB used; deleting this pod"
  echo busy >$SRV/DONE
  if [[ -n "${RUNPOD_API_KEY:-}" && -n "${RUNPOD_POD_ID:-}" ]]; then
    for _ in 1 2 3 4 5; do fv_self_delete $SRV/log.txt && break; sleep 30; done
  fi
  exit 1
fi
(while sleep 2; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits; done) >$SRV/nvsmi.csv 2>/dev/null &

fetch() { python3 /opt/fv/r2-fetch.py --root "$R" --plan /opt/fv/plan.json --status $SRV/fetch-status.json >>$SRV/fetch.log 2>&1; }
step fetch fetch || { echo 1 >$SRV/DONE; exit 1; }
du -sh "$R"/* >$SRV/trees.txt 2>&1

# The serve proxy port is 8000: fv-serve listens on 8001 here, /srv's static server on 8000.
sed 's/^bind = "0.0.0.0:8000"/bind = "127.0.0.1:8001"/; s#^state_dir = .*#state_dir = "/fvstate"#' /opt/fv/runpod-family-h3.toml >/opt/fv/family.toml
BASE=http://127.0.0.1:8001
serve() {
  chmod +x /opt/fv/fv-serve
  ldd /opt/fv/fv-serve >$SRV/ldd.txt 2>&1
  FV_WEIGHTS="$R" FV_AUTH_MODE=none FV_CONFIG=/opt/fv/family.toml RUST_LOG=info \
    /opt/fv/fv-serve --config /opt/fv/family.toml >$SRV/serve.log 2>&1 &
  echo $! >/tmp/serve.pid
  local t0=$SECONDS
  until [[ "$(curl -s -o /dev/null -w '%{http_code}' $BASE/ping)" == 200 ]]; do
    kill -0 "$(cat /tmp/serve.pid)" 2>/dev/null || { log "fv-serve exited: $(tail -5 $SRV/serve.log)"; return 1; }
    (( SECONDS - t0 < 1800 )) || { log "no /ping 200 in 30 min"; return 1; }
    sleep 3
  done
  curl -sS $BASE/health >$SRV/health-ready.json
  curl -sS $BASE/fv/v1/capabilities >$SRV/capabilities.json
}
step serve serve || { echo 1 >$SRV/DONE; exit 1; }

PROMPT="${FV_PROMPT:-A red fox trots through fresh snow at dawn, its breath steaming, soft golden light, cinematic tracking shot.}"
printf 'n\tmodel\tstatus\twall_s\tbytes\tsha256\n' >$SRV/jobs.tsv
job() {
  local n="$1" model="$2" t0 id st s bytes sha
  t0="$(date +%s.%N)"
  id="$(curl -sS -H 'content-type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "prompt": sys.argv[2], "seed": 1, "short_edge": 480, "aspect_ratio": "16:9"}))' "$model" "$PROMPT")" \
    $BASE/fv/v1/jobs | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id") or "")' 2>/dev/null)"
  [[ -n "$id" ]] || { log "job $n $model: submit failed"; return 1; }
  for _ in $(seq 1 900); do
    st="$(curl -sS $BASE/fv/v1/jobs/$id)"
    s="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("status", ""))' <<<"$st" 2>/dev/null)"
    case "$s" in succeeded | failed | cancelled) break ;; esac
    sleep 1
  done
  local wall
  wall="$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b - a}')"
  echo "$st" >"$SRV/job-$n-$model.json"
  # The output's size from the job (its /files URL is signed for the public port); the MP4 from the local store.
  bytes="$(python3 -c 'import json,sys; print((json.load(sys.stdin).get("output") or {}).get("bytes", 0))' <<<"$st" 2>/dev/null)"
  f="$(find /fvstate -name "${id#fvjob_}.mp4" 2>/dev/null | head -1)"
  [[ -n "$f" ]] && cp "$f" "$SRV/job-$n-$model.mp4"
  sha="$(sha256sum "$SRV/job-$n-$model.mp4" 2>/dev/null | cut -c1-16)"
  curl -sS $BASE/health >"$SRV/health-after-$n.json"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$model" "$s" "$wall" "$bytes" "$sha" >>$SRV/jobs.tsv
  log "job $n $model: $s in ${wall}s, $bytes B"
  [[ "$s" == succeeded ]]
}
rc=0
step job1-turbo job 1 fasth3 || rc=1
step job2-max-swap job 2 sol-h3 || rc=1
step job3-max-resident job 3 sol-h3 || rc=1
step job4-v2nvfp4-swap job 4 fasth3-v2-nvfp4 || rc=1
step job5-turbo-back job 5 fasth3 || rc=1
grep -E 'model resident|"loading"|loading|swap|unload|ERROR|error' $SRV/serve.log | grep -v ' DEBUG ' | tail -200 >$SRV/serve-loads.txt
python3 -c 'import json,sys; print(json.dumps({"jobs": open(sys.argv[1]).read(), "status": open(sys.argv[2]).read()}))' $SRV/jobs.tsv $SRV/status.tsv >$SRV/summary.json
kill "$(cat /tmp/serve.pid)" 2>/dev/null
echo $rc >$SRV/DONE
log "done rc=$rc"
