#!/usr/bin/env bash
# Minimal Runpod serverless (queue) worker for cold-start measurement (WP-19).
#
# Speaks the job-take / job-done / ping protocol of docs/serve/research-deploy.md
# §1.1 with curl, runs one fv-gpucheck generation per job (a fresh process, so
# every job pays the model load) and returns timings, never the video:
#
#   {"kind":"gen","model":"fasth3|ltx25|fastwan","prompt":"…","seed":1024,"env":"K=V …","evict":1}
#   {"kind":"quantize","family":"h3|ltx2-gemma4","root":"h3-base|ltx25"}   (E13 tree, adds files only)
#   {"kind":"io-bench","dir":"h3-base/transformer"}
#   {"kind":"ls","dir":"h3-base"}
#
# The output carries epoch timestamps: worker_start (this script started,
# i.e. after image pull and container start), job_taken, gen_start, gen_end,
# plus the generator's own load_s / timings / load/io lines. Weights come from
# FV_W (default /runpod-volume/weights); everything written goes to /tmp,
# except the quantize job's new text_encoder_fp8/ tree.
#
# Runs as the template's start command; outside Runpod (RUNPOD_WEBHOOK_GET_JOB
# unset) it runs the job JSON given as $1 once and prints the output.
set -u
WORKER_START="$(date +%s.%N)"
FV="${FV_BIN:-/opt/fastvideo-rs/target/release/fv-gpucheck}"
W="${FV_W:-/runpod-volume/weights}"
SCRATCH="${FV_SCRATCH:-/tmp/fv-worker}"
mkdir -p "$SCRATCH"
# The volume's HF-cache trees link absolutely into /workspace/weights (where a
# pod mounts it); a serverless worker mounts it at /runpod-volume.
if [[ -d /runpod-volume/weights && ! -e /workspace/weights ]]; then
  mkdir -p /workspace && ln -sfn /runpod-volume/weights /workspace/weights
fi
POD="${RUNPOD_POD_ID:-local}"
AUTH="${RUNPOD_AI_API_KEY:-}"
VERSION="fv-rs-worker/1"
BUILD="$(cat /opt/fastvideo-rs/target/release/fv-gpucheck.build-id 2>/dev/null || echo unknown)"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
now() { date +%s.%N; }
# "key":"string" or "key":number from a small flat JSON object.
field() {
  local v
  v="$(grep -o "\"$1\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" <<<"$2" | head -1 | sed 's/^[^:]*:[[:space:]]*"//; s/"$//')"
  [[ -z "$v" ]] && v="$(grep -o "\"$1\"[[:space:]]*:[[:space:]]*[0-9.]*" <<<"$2" | head -1 | sed 's/^[^:]*:[[:space:]]*//')"
  printf '%s' "$v"
}
jstr() { # JSON string literal
  local s
  s="$(printf '%s' "$1" | tr -d '\000-\010\013-\037')"
  s="${s//\\/\\\\}"; s="${s//\"/\\\"}"; s="${s//$'\t'/\\t}"; s="${s//$'\n'/\\n}"
  printf '"%s"' "$s"
}
# Last "[INFO] <tag> {json}" of a log, as raw JSON (or null).
info_json() {
  local line
  line="$(grep -F "$1" "$2" 2>/dev/null | tail -1)"
  [[ -n "$line" ]] && printf '%s' "{${line#*\{}" || printf 'null'
}
# Every "load/io <phase> {json}" line as a JSON object keyed by phase.
load_io_json() {
  local out="{" first=1 line phase
  while IFS= read -r line; do
    phase="${line#*load/io }"; phase="${phase%% \{*}"
    [[ $first == 1 ]] || out+=","
    out+="$(jstr "$phase"):{${line#*\{}"
    first=0
  done < <(grep -F 'load/io ' "$1" 2>/dev/null)
  printf '%s}' "$out"
}

run_gen() {
  local input="$1" dir="$2" model prompt seed extra rc
  model="$(field model "$input")"
  prompt="$(field prompt "$input")"
  prompt="${prompt:-a red fox trotting through fresh snow in a pine forest, golden hour, cinematic}"
  seed="$(field seed "$input")"; seed="${seed:-1024}"
  extra="$(field env "$input")"
  if [[ "$(field evict "$input")" == 1 ]]; then
    "$FV" --out "$dir/evict" evict-cache "$W/h3-base" "$W/ltx25" "$W/fastwan21-1.3b" >/dev/null 2>&1 || true
  fi
  local cmd=()
  case "$model" in
    fasth3)
      cmd=("$FV" --mode fast --out "$dir/out" h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa
        --adaln-cache "$dir/adaln.cache" --clip-dir "$dir/frames" --prompt "$prompt" --seconds 5
        --seed "$seed" --text-encoder auto --no-text-cache --text-weights "$W/h3-base") ;;
    ltx25)
      local text=streamed
      [[ " $extra " == *" FASTVIDEO_LTX2_TEXT_FP8=1 "* ]] && text=auto
      cmd=("$FV" --mode fast --out "$dir/out" ltx2 gen --model-version 2.5 --weights "$W/ltx25"
        --dit "$W/ltx25" --prompt "$prompt" --seed "$seed" --two-stage --text "$text" --no-text-cache
        --clip "$dir/frames") ;;
    fastwan)
      cmd=("$FV" --mode fast --vsa --out "$dir/out" wan gen --weights "$W/fastwan21-1.3b"
        --prompt "$prompt" --seed "$seed" --no-text-cache --clip-dir "$dir/frames") ;;
    *) printf '{"error":%s}' "$(jstr "unknown model '$model'")"; return 1 ;;
  esac
  GEN_START="$(now)"
  # shellcheck disable=SC2086
  env $extra "${cmd[@]}" >"$dir/stdout.log" 2>"$dir/stderr.log"
  rc=$?
  GEN_END="$(now)"
  local frames hash first_frame
  frames="$(find "$dir/frames" -name '*.png' 2>/dev/null | wc -l)"
  hash="$(find "$dir/frames" -name '*.png' 2>/dev/null | sort | xargs -r sha256sum | awk '{print $1}' | sha256sum | awk '{print $1}')"
  first_frame="$(find "$dir/frames" -name '*.png' 2>/dev/null | sort | head -1 | xargs -r sha256sum | awk '{print $1}')"
  local timings
  case "$model" in
    fasth3) timings="$(info_json 'h3/timings' "$dir/stderr.log")" ;;
    ltx25) timings="$(info_json 'ltx2/timings' "$dir/stderr.log")" ;;
    *) timings="$(info_json '/timings' "$dir/stderr.log")" ;;
  esac
  local bench=null
  [[ -s "$dir/benchmark.json" ]] && bench="$(cat "$dir/benchmark.json")"
  printf '{"model":%s,"rc":%s,"env":%s,"gen_start":%s,"gen_end":%s,"frames":%s,"frames_sha256":%s,"first_frame_sha256":%s,"timings":%s,"load_io":%s,"benchmark":%s,"stderr_tail":%s}' \
    "$(jstr "$model")" "$rc" "$(jstr "$extra")" "$GEN_START" "$GEN_END" "$frames" "$(jstr "$hash")" "$(jstr "$first_frame")" \
    "$timings" "$(load_io_json "$dir/stderr.log")" "$bench" "$(jstr "$(grep -v 'block [0-9]*/\|resident layer' "$dir/stderr.log" | grep -i 'load\|text\|INFO\|error\|prequant\|fp8\|mismatch\|verify' | tail -40)")"
  return $rc
}

run_job() {
  local input="$1" kind dir t0 body rc=0
  kind="$(field kind "$input")"
  dir="$SCRATCH/job-$(date +%s%N)"
  mkdir -p "$dir"
  t0="$(now)"
  case "$kind" in
    gen) body="$(run_gen "$input" "$dir")"; rc=$? ;;
    quantize)
      local fam root
      fam="$(field family "$input")"; root="$(field root "$input")"
      "$FV" --out "$dir/out" quantize-text-encoder --family "$fam" --root "$W/$root" >"$dir/q.log" 2>&1; rc=$?
      body="$(printf '{"rc":%s,"report":%s,"log":%s}' "$rc" "$(cat "$dir"/out/quantize-text-encoder.json 2>/dev/null || echo null)" "$(jstr "$(tail -40 "$dir/q.log")")")" ;;
    io-bench)
      local d t; d="$(field dir "$input")"; t="$(field threads "$input")"
      "$FV" --out "$dir/out" io-bench --dir "$W/$d" --threads "${t:-1,4,16,32}" --limit-gb 6 --evict --mmap >"$dir/io.log" 2>&1; rc=$?
      body="$(printf '{"rc":%s,"log":%s}' "$rc" "$(jstr "$(grep -F '[INFO]' "$dir/io.log" | tail -20)")")" ;;
    ls)
      local d; d="$(field dir "$input")"
      body="$(printf '{"ls":%s}' "$(jstr "$(cd "$W" && ls -la "$d" 2>&1 | head -80; du -sh "$d"/* 2>/dev/null | head -40)")")" ;;
    *) body="$(printf '{"error":%s}' "$(jstr "unknown kind '$kind'")")"; rc=1 ;;
  esac
  printf '{"worker_start":%s,"job_taken":%s,"job_end":%s,"build":%s,"gpu":%s,"host_mem_gb":%s,"result":%s}' \
    "$WORKER_START" "$t0" "$(now)" "$(jstr "$BUILD")" \
    "$(jstr "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)")" \
    "$(awk '/MemTotal/{printf "%.0f", $2/1e6}' /proc/meminfo)" "${body:-null}"
  return $rc
}

if [[ -z "${RUNPOD_WEBHOOK_GET_JOB:-}" ]]; then
  run_job "${1:?job JSON}"
  exit $?
fi

GET_URL="${RUNPOD_WEBHOOK_GET_JOB//\$ID/$POD}"
[[ "$GET_URL" == *\?* ]] || GET_URL="$GET_URL?"
PING_URL="${RUNPOD_WEBHOOK_PING//\$RUNPOD_POD_ID/$POD}"
CURRENT="$SCRATCH/current-job"
: >"$CURRENT"
# Heartbeat.
(
  while :; do
    sep='?'; [[ "$PING_URL" == *\?* ]] && sep='&'
    curl -sS -m 10 -o /dev/null -H "Authorization: $AUTH" \
      "${PING_URL}${sep}job_id=$(cat "$CURRENT" 2>/dev/null)&runpod_version=$VERSION" 2>/dev/null
    sleep $(( ${RUNPOD_PING_INTERVAL:-10000} / 1000 ))
  done
) &
log "worker $POD up (build $BUILD); polling"
busy=0
while :; do
  resp="$SCRATCH/take.json"
  code="$(curl -sS -m 95 -o "$resp" -w '%{http_code}' -H "Authorization: $AUTH" "${GET_URL}&job_in_progress=$busy" 2>/dev/null || echo 000)"
  case "$code" in
    200) ;;
    429) sleep 5; continue ;;
    *) sleep 1; continue ;;
  esac
  job="$(cat "$resp")"
  id="$(field id "$job")"
  [[ -n "$id" ]] || continue
  input="$(sed 's/.*"input"[[:space:]]*:[[:space:]]*\({[^}]*}\).*/\1/' <<<"$job")"
  log "job $id: $input"
  echo "$id" >"$CURRENT"
  busy=1
  out="$(run_job "$input")"; rc=$?
  if [[ $rc == 0 ]]; then
    body="{\"output\":$out}"
  else
    body="{\"error\":$(jstr "$out")}"
  fi
  done_url="${RUNPOD_WEBHOOK_POST_OUTPUT//\$ID/$id}"
  done_url="${done_url//\$RUNPOD_POD_ID/$POD}"
  for delay in 1 1 2; do
    if curl -sS -m 60 -o /dev/null -f -X POST -H "Authorization: $AUTH" -H "X-Request-ID: $id" \
      -H 'Content-Type: application/x-www-form-urlencoded' --data-binary "$body" \
      "${done_url}&isStream=false"; then
      break
    fi
    sleep "$delay"
  done
  log "job $id done rc=$rc"
  : >"$CURRENT"
  busy=0
done
