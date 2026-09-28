#!/usr/bin/env bash
# The fv-serve gateway on Runpod (docs/serve/gateway.md): a CPU pod running
# `fv-serve --config /etc/fv/gateway.toml` in front of serverless queue
# pools whose workers run `FV_SERVE_ROLE=worker`.
#
#   runpod-gateway.sh validate [image]   h3-turbo + wan pools (queue endpoints on the weight
#                                        volume), the gateway on a CPU pod; one job per API
#                                        through the gateway, the same request straight to
#                                        the pool's endpoint for the added latency; then
#                                        deletes everything and checks it is gone
#   runpod-gateway.sh down               deletes every fv-gw-* pod, endpoint and template
#
# Secrets: the internal token and the user API key are generated per run and
# never printed (the key's SHA-256 goes to the gateway); D1/R2 come from the
# account's Runpod secrets ({{ RUNPOD_SECRET_fv_* }}). The gateway pod gets
# RUNPOD_API_KEY (it calls /run, /status, /cancel, /health of the pools).
#
# Money guards: the balance floor (FV_MIN_BALANCE, default 8 $), workers
# max 1 / idle 30 s per endpoint, the endpoint script's wall-clock backstop,
# a pod backstop (FV_GATEWAY_CAP_S, default 3600 s), destroy-on-exit.
#
# Images: [image] / FV_SERVE_IMAGE runs everything on one image (e.g. the
# legacy all-in-one :latest); without either, each pool boots its variant's
# published image and the gateway the CPU-only `gateway` image
# (docs/serve/images.md, scripts/serve/variants.sh).
#
# Env: RUNPOD_API_KEY; FV_SERVE_IMAGE; RUNPOD_VOLUME_ID (default s2k01690bi);
# RUNPOD_GPU_TYPES (default H100/H200 list); FV_GATEWAY_CPU_FLAVOR (default cpu3c).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/lib.sh
source "$HERE/../gpu/lib.sh"
# shellcheck source-path=SCRIPTDIR source=variants.sh
source "$HERE/variants.sh"

REST="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
QUEUE="${RUNPOD_QUEUE_API:-https://api.runpod.ai/v2}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
MIN_BALANCE="${FV_MIN_BALANCE:-8}"
CAP_S="${FV_GATEWAY_CAP_S:-3600}"
CPU_FLAVOR="${FV_GATEWAY_CPU_FLAVOR:-cpu3c}"
export RUNPOD_GPU_TYPES="${RUNPOD_GPU_TYPES:-NVIDIA H100 80GB HBM3,NVIDIA H100 NVL,NVIDIA H200}"
OUT_DIR="$FV_ROOT/artifacts/serve/e2e/gateway"
LEDGER="${FV_SERVE_LEDGER:-$FV_ROOT/artifacts/runpod/serve/ledger.tsv}"

SECRET_ENV_JSON='{
  "FV_CF_ACCOUNT_ID": "{{ RUNPOD_SECRET_fv_cf_account_id }}",
  "FV_CF_API_TOKEN": "{{ RUNPOD_SECRET_fv_cf_api_token }}",
  "FV_D1_DATABASE_ID": "{{ RUNPOD_SECRET_fv_d1_database_id }}",
  "FV_R2_BUCKET": "{{ RUNPOD_SECRET_fv_r2_bucket }}",
  "FV_R2_ENDPOINT": "{{ RUNPOD_SECRET_fv_r2_endpoint }}",
  "FV_R2_ACCESS_KEY_ID": "{{ RUNPOD_SECRET_fv_r2_access_key_id }}",
  "FV_R2_SECRET_ACCESS_KEY": "{{ RUNPOD_SECRET_fv_r2_secret_access_key }}",
  "FV_WEBHOOK_ED25519_KEY": "{{ RUNPOD_SECRET_fv_webhook_ed25519_key }}"
}'

rest() {
  curl -sS --fail-with-body -X "$1" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${3:+-d "$3"} "$REST$2"
}
queue() {
  curl -sS --max-time 60 -X "$1" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${3:+-d "$3"} "$QUEUE/$2"
}
ledger() { mkdir -p "$(dirname "$LEDGER")"; printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }
now() { date +%s.%N; }

balance() {
  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' "$GQL" \
    -d '{"query":"{ myself { clientBalance } }"}' | jq -r '.data.myself.clientBalance // empty'
}

check_balance() {
  local b
  b="$(balance)"
  [[ -n "$b" ]] || die "could not read the Runpod balance"
  awk -v b="$b" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+0)}' || die "Runpod balance \$$b is below the floor \$$MIN_BALANCE"
  log "Runpod balance \$$b (floor \$$MIN_BALANCE)"
}

# Every fv-gw-* resource (pods, endpoints, templates) as "kind id name".
leftovers() {
  { rest GET /pods | jq -r '.[]? | select(.name | startswith("fv-gw")) | "pod \(.id) \(.name)"'
    rest GET /endpoints | jq -r '.[]? | select(.name | startswith("fv-gw")) | "endpoint \(.id) \(.name)"'
    rest GET /templates | jq -r '.[]? | select(.name | startswith("fv-gw")) | "template \(.id) \(.name)"'
  } 2>/dev/null || true
}

cmd_down() {
  local kind id name
  while read -r kind id name; do
    [[ -n "$id" ]] || continue
    case $kind in
      pod) rest DELETE "/pods/$id" >/dev/null 2>&1 && ledger "pod-deleted $id" ;;
      endpoint) bash "$HERE/runpod-endpoint.sh" down "$id" >/dev/null 2>&1 ;;
      template) rest DELETE "/templates/$id" >/dev/null 2>&1 && ledger "template-deleted $id" ;;
    esac
    log "deleted $kind $id ($name)"
  done < <(leftovers | sort -r)   # pods and endpoints before templates
  local left
  left="$(leftovers)"
  [[ -z "$left" ]] || die "still present: $left"
  log "no fv-gw-* pod, endpoint or template is left"
}

POD=""
EPS=()
TPLS=()
cleanup() {
  local rc=$?
  [[ -n "$POD" ]] && { rest DELETE "/pods/$POD" >/dev/null 2>&1 && ledger "pod-deleted $POD"; POD=""; }
  local i
  for i in "${!EPS[@]}"; do bash "$HERE/runpod-endpoint.sh" down "${EPS[$i]}" "${TPLS[$i]}" >/dev/null 2>&1 || true; done
  EPS=(); TPLS=()
  exit "$rc"
}

# gw <method> <path> [body]: through the gateway with the user key; prints
# "<http code> <seconds>" on stderr's last line and the body on stdout.
GW=""
USER_KEY=""
gw_call() {
  local m="$1" p="$2" body="${3:-}" auth="${4:-Bearer}"
  curl -sS --max-time 120 -X "$m" -H "Authorization: $auth $USER_KEY" -H 'content-type: application/json' \
    ${body:+-d "$body"} -w '\n%{http_code} %{time_total}' "$GW$p"
}

# split_submit <gw_call output>: sets `sub` ("<code> <seconds>") and `body`
# in the caller, logs the reply (ids and errors only), stops on a refusal.
split_submit() {
  sub="$(tail -1 <<<"$1")"
  body="$(sed '$d' <<<"$1")"
  log "submit: $sub $(jq -c '{id, request_id, task_id, error}' <<<"$body" 2>/dev/null | head -c 400)"
  [[ "${sub%% *}" == 2* ]] || die "submit refused: $sub $(head -c 400 <<<"$body")"
}

# poll_gw <status path> <jq done expr> [auth]: seconds until done, and the last body.
poll_gw() {
  local p="$1" expr="$2" auth="${3:-Bearer}" t0 out body
  t0="$(now)"
  while :; do
    out="$(gw_call GET "$p" "" "$auth" || true)"
    body="$(sed '$d' <<<"$out")"
    if jq -e "$expr" <<<"$body" >/dev/null 2>&1; then break; fi
    (( ${t0%.*} + 1500 > $(date +%s) )) || { echo "timeout"; return 1; }
    sleep 0.5
  done
  printf '%s\n%s\n' "$(awk -v a="$(now)" -v b="$t0" 'BEGIN{printf "%.2f", a-b}')" "$body"
}

# direct <endpoint> <input>: the same request straight to the pool (queue
# envelope, wait: true), polled at 0.5 s like the gateway; prints seconds.
direct() {
  local ep="$1" input="$2" t0 id st s
  t0="$(now)"
  id="$(queue POST "$ep/run" "{\"input\":$input}" | jq -r '.id // empty')"
  [[ -n "$id" ]] || die "direct submit to $ep failed"
  while :; do
    st="$(queue GET "$ep/status/$id" || echo '{}')"
    s="$(jq -r '.status // "?"' <<<"$st")"
    case "$s" in COMPLETED | FAILED | CANCELLED | TIMED_OUT) break ;; esac
    sleep 0.5
  done
  jq -c --arg w "$(awk -v a="$(now)" -v b="$t0" 'BEGIN{printf "%.2f", a-b}')" \
    '{status, wall_s: ($w|tonumber), delayTime, executionTime, job_status: .output.body.status}' <<<"$st"
}

cmd_validate() {
  require_tools curl jq openssl
  : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
  check_balance
  local b0 image gw_image token keyhash admin
  b0="$(balance)"
  image="${1:-${FV_SERVE_IMAGE:-}}"   # empty: every pool picks its variant image
  gw_image="${image:-$(fv_variant_image gateway pod)}"
  token="$(openssl rand -hex 24)"
  USER_KEY="fvk-$(openssl rand -hex 16)"
  admin="fvadm_$(openssl rand -hex 16)"
  keyhash="$(printf '%s' "$USER_KEY" | sha256sum | cut -d' ' -f1)"
  trap cleanup EXIT INT TERM
  mkdir -p "$OUT_DIR"

  # Pools: queue endpoints whose workers run behind the gateway.
  local extra ep_h3 tpl_h3 ep_wan tpl_wan line
  extra="$(jq -cn --arg t "$token" '{FV_SERVE_ROLE: "worker", FV_INTERNAL_TOKEN: $t, FV_JOBS_HEARTBEAT_S: "10"}')"
  line="$(FV_EXTRA_ENV_JSON="$extra" FV_ENDPOINT_PREFIX=fv-gw-h3 FV_SERVE_CONFIG=/etc/fv/runpod.toml FV_IDLE_TIMEOUT_S=30 \
    FV_ENDPOINT_CAP_S="$CAP_S" bash "$HERE/runpod-endpoint.sh" up ${image:+"$image"} | tail -1)"
  read -r ep_h3 tpl_h3 <<<"$line"; EPS+=("$ep_h3"); TPLS+=("$tpl_h3")
  line="$(FV_EXTRA_ENV_JSON="$extra" FV_ENDPOINT_PREFIX=fv-gw-wan FV_SERVE_CONFIG=/etc/fv/runpod-wan.toml FV_IDLE_TIMEOUT_S=30 \
    FV_ENDPOINT_CAP_S="$CAP_S" bash "$HERE/runpod-endpoint.sh" up ${image:+"$image"} | tail -1)"
  read -r ep_wan tpl_wan <<<"$line"; EPS+=("$ep_wan"); TPLS+=("$tpl_wan")
  log "pools: h3-turbo=$ep_h3 wan=$ep_wan"

  # The gateway: a CPU pod. h3-max and ltx have no endpoint here (503 path).
  local payload resp t_create t_ready code
  payload="$(jq -n --arg image "$gw_image" \
    --arg flavor "$CPU_FLAVOR" --argjson secrets "$SECRET_ENV_JSON" --arg tok "$token" --arg keys "$keyhash" \
    --arg admin "$admin" --arg rp "$RUNPOD_API_KEY" --arg h3 "$ep_h3" --arg wan "$ep_wan" --arg name "fv-gw-gateway-$(date -u +%m%d%H%M%S)" '{
      name: $name, imageName: $image, computeType: "CPU", cpuFlavorIds: [$flavor], vcpuCount: 2,
      containerDiskInGb: 20, ports: ["8000/http"],
      dockerEntrypoint: ["/opt/fastvideo-rs/bin/fv-serve"], dockerStartCmd: ["--config", "/etc/fv/gateway.toml"],
      env: ($secrets + {
        FV_SERVE_MODE: "http", FV_STATE_DIR: "/fvstate", FV_INTERNAL_TOKEN: $tok, FV_API_KEYS: $keys,
        FV_ADMIN_TOKEN: $admin, FV_RUNPOD_API_KEY: $rp, FV_POOL_H3_TURBO_ENDPOINT: $h3, FV_POOL_WAN_ENDPOINT: $wan,
        FV_POOL_H3_MAX_ENDPOINT: "fv-gw-not-deployed", FV_POOL_LTX_ENDPOINT: "fv-gw-not-deployed", RUST_LOG: "info"
      })
    }')"
  t_create="$(now)"
  resp="$(rest POST /pods "$payload")" || die "gateway pod create failed: $(jq -r ".error // .message // \"unknown error\"" <<<"$resp" 2>/dev/null | head -c 300)"
  POD="$(jq -r '.id // empty' <<<"$resp")"
  [[ -n "$POD" ]] || die "gateway pod create returned no id"
  ledger "pod-created $POD gateway cpu=$CPU_FLAVOR $(jq -r '.costPerHr // empty' <<<"$resp")/h"
  # shellcheck disable=SC2016 # expanded by the child shell
  nohup env POD_ID="$POD" CAP="$CAP_S" API="$REST" bash -c \
    'sleep "$CAP"; curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/pods/$POD_ID" >/dev/null' >/dev/null 2>&1 &
  GW="https://$POD-8000.proxy.runpod.net"
  log "gateway pod $POD ($GW); waiting for /health"
  while :; do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$GW/health" || true)"
    [[ "$code" == 200 ]] && { t_ready="$(now)"; break; }
    (( ${t_create%.*} + 1500 > $(date +%s) )) || die "the gateway never answered /health (last $code)"
    sleep 5
  done
  log "gateway ready after $(awk -v a="$t_ready" -v b="$t_create" 'BEGIN{printf "%.0f", a-b}') s"

  local caps out body sub r
  caps="$(gw_call GET /fv/v1/capabilities | sed '$d')"
  log "gateway models: $(jq -c '[.models[].caps.id]' <<<"$caps") pools: $(jq -c '[.pools[] | {id, available}]' <<<"$caps")"
  local results='[]'
  add() { results="$(jq -c --argjson r "$1" '. + [$r]' <<<"$results")"; }

  # 1. native → wan (cold pool start included).
  out="$(gw_call POST /fv/v1/jobs '{"model":"fastwan21-1.3b","prompt":"a red fox trotting through fresh snow, cinematic","seed":1}')"
  split_submit "$out"
  r="$(poll_gw "/fv/v1/jobs/$(jq -r .id <<<"$body")" '.status == "succeeded" or .status == "failed"')"
  add "$(jq -c --arg sub "$sub" --arg w "$(head -1 <<<"$r")" '{api: "native", pool: "wan", cold: true, submit: $sub, wall_s: ($w|tonumber), status: .status, url_host: (.output.url // "" | capture("^(?<h>https://[^/?]+)").h? // "")}' <<<"$(sed 1d <<<"$r")")"
  # 2. FastVideo /v1/videos → wan (warm).
  out="$(gw_call POST /v1/videos '{"model":"fastwan21-1.3b","prompt":"ocean waves at sunset","seconds":"5"}')"
  split_submit "$out"
  r="$(poll_gw "/v1/videos/$(jq -r .id <<<"$body")" '.status == "completed" or .status == "failed"')"
  add "$(jq -c --arg sub "$sub" --arg w "$(head -1 <<<"$r")" '{api: "fastvideo", pool: "wan", cold: false, submit: $sub, wall_s: ($w|tonumber), status: .status}' <<<"$(sed 1d <<<"$r")")"
  # 3. fal queue → h3-turbo (cold pool start included).
  out="$(gw_call POST /minimax/h3-turbo/text-to-video '{"prompt":"A red fox trots through fresh snow at dawn","seed":1}' Key)"
  split_submit "$out"
  r="$(poll_gw "/minimax/h3-turbo/requests/$(jq -r .request_id <<<"$body")/status" '.status == "COMPLETED"' Key)"
  add "$(jq -c --arg sub "$sub" --arg w "$(head -1 <<<"$r")" '{api: "fal", pool: "h3-turbo", cold: true, submit: $sub, wall_s: ($w|tonumber), status: .status, error: .error}' <<<"$(sed 1d <<<"$r")")"
  # 4. MiniMax → h3-turbo (warm).
  out="$(gw_call POST /v2/video_generation '{"model":"MiniMax-H3-Turbo","content":[{"type":"text","text":"a lighthouse in a storm"}],"resolution":"768P","duration":5,"ratio":"16:9"}')"
  split_submit "$out"
  r="$(poll_gw "/v2/query/video_generation/$(jq -r .task_id <<<"$body")" '.task.status == "succeeded" or .task.status == "failed"')"
  add "$(jq -c --arg sub "$sub" --arg w "$(head -1 <<<"$r")" '{api: "minimax", pool: "h3-turbo", cold: false, submit: $sub, wall_s: ($w|tonumber), status: .task.status}' <<<"$(sed 1d <<<"$r")")"
  # 5. LTX → no ltx pool deployed: 503 + Retry-After.
  out="$(gw_call POST /v2/text-to-video '{"prompt":"a red fox","model":"ltx-2-5-fast","duration":6,"resolution":"1920x1080"}')"
  add "$(jq -cn --arg sub "$(tail -1 <<<"$out")" '{api: "ltx", pool: "ltx (not deployed)", submit: $sub}')"

  # Added latency: the same warm request through the gateway and straight to the pool.
  local lat='[]' i g d
  for i in 1 2 3; do
    out="$(gw_call POST /fv/v1/jobs '{"model":"fastwan21-1.3b","prompt":"a paper boat on a stream","seed":7}')"
    split_submit "$out"
    g="$(poll_gw "/fv/v1/jobs/$(jq -r .id <<<"$body")" '.status == "succeeded" or .status == "failed"' | head -1)"
    d="$(direct "$ep_wan" '{"kind":"http","method":"POST","path":"/fv/v1/jobs","headers":{},"body":{"model":"fastwan21-1.3b","prompt":"a paper boat on a stream","seed":7},"wait":true}')"
    lat="$(jq -c --arg g "$g" --arg sub "$sub" --argjson d "$d" '. + [{gateway_wall_s: ($g|tonumber), gateway_submit: $sub, direct: $d}]' <<<"$lat")"
  done
  local pools
  pools="$(curl -sS --max-time 30 -H "Authorization: Bearer $admin" "$GW/fv/v1/gateway/pools" | jq -c '[.pools[] | {pool, queued, running, run_time, queue_wait, workers, available}]')"

  trap - EXIT INT TERM
  rest DELETE "/pods/$POD" >/dev/null 2>&1 && ledger "pod-deleted $POD"; POD=""
  for i in "${!EPS[@]}"; do bash "$HERE/runpod-endpoint.sh" down "${EPS[$i]}" "${TPLS[$i]}" >/dev/null 2>&1 || true; done
  EPS=(); TPLS=()
  sleep 10
  local left b1
  left="$(leftovers)"
  b1="$(balance)"
  jq -n --arg image "${image:-variants}" --arg ready "$(awk -v a="$t_ready" -v b="$t_create" 'BEGIN{printf "%.1f", a-b}')" \
    --argjson results "$results" --argjson lat "$lat" --argjson pools "$pools" --arg left "$left" \
    --arg b0 "$b0" --arg b1 "$b1" '{
      image: $image, gateway_pod_ready_s: ($ready|tonumber), jobs: $results, latency: $lat, pools: $pools,
      leftovers: $left, balance_before: ($b0|tonumber), balance_after: ($b1|tonumber)}' | tee "$OUT_DIR/validate-$(date -u +%m%d%H%M%S).json"
  [[ -z "$left" ]] || die "left over: $left"
}

case "${1:-}" in
  validate) shift; cmd_validate "$@" ;;
  down) shift; : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"; cmd_down ;;
  *) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
