#!/usr/bin/env bash
# fv-serve as a Runpod serverless endpoint (docs/serve/design.md §6.2, §6.4; WP-16).
#
#   runpod-endpoint.sh smoke [image]     queue endpoint with the weight volume mounted:
#                                        cold info job (NVENC encode, volume listing,
#                                        ready_after_s), /fv/v1/capabilities, one small
#                                        job (wait: true; media in R2, record in D1),
#                                        a warm info job; timings; then deletes it all
#   runpod-endpoint.sh smoke-lb [image]  the same as a load-balancer endpoint
#                                        (fv-serve's HTTP on $PORT, /ping 204→200)
#   runpod-endpoint.sh up [image]        queue endpoint; prints "<endpoint> <template>"
#   runpod-endpoint.sh up-lb [image]     LB endpoint; prints "<endpoint> -"
#   runpod-endpoint.sh job <ep> '<input json>'   POST /run, poll /status, print timings
#   runpod-endpoint.sh down <ep> [template]
#   runpod-endpoint.sh plan [image]      print both payloads (no API call)
#
# Queue workers run the Rust worker loop (FV_SERVE_MODE=runpod-queue) from
# the digest-pinned `serve` image, with the weight volume at /runpod-volume
# (read only; state on the container disk). Secrets are template references
# ({{ RUNPOD_SECRET_fv_* }} → FV_*), never values. The gateway authenticates,
# so fv-serve runs `auth.mode = trust-gateway`.
#
# Money guards: workers min 0 / max 1, short idle timeout, FlashBoot off, a
# balance floor (FV_MIN_BALANCE, default 8 $), a wall-clock cap
# (FV_ENDPOINT_CAP_S, default 1800 s) whose detached backstop deletes the
# endpoint and template, a destroy-on-exit trap, and the ledger
# (artifacts/runpod/serve/ledger.tsv).
#
# Env: RUNPOD_API_KEY; FV_SERVE_IMAGE; FV_SERVE_CONFIG (default
# /etc/fv/runpod-fake.toml); RUNPOD_VOLUME_ID (default s2k01690bi,
# fv-weights-b200-us) — its datacenter pins the endpoint; RUNPOD_GPU_TYPES
# (comma list); RUNPOD_ALLOWED_CUDA (default 13.0; empty drops the filter); FV_SMOKE_MODEL (default
# fake-wan); FV_LB_WORKERS_MAX (LB workers.max, default 1; also passed to
# fv-serve as FV_WORKERS_MAX); FV_ENDPOINT_PREFIX (endpoint/template name
# prefix, default fv-serve); FV_IDLE_TIMEOUT_S (default 5).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/lib.sh
source "$HERE/../gpu/lib.sh"

REST="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
REST2="${RUNPOD_API2_BASE:-https://api.runpod.io/v2}"
QUEUE="${RUNPOD_QUEUE_API:-https://api.runpod.ai/v2}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
IMAGE_DEFAULT="ghcr.io/zaitrarrio/fastvideo-rs-serve:latest"
CONFIG="${FV_SERVE_CONFIG:-/etc/fv/runpod-fake.toml}"
VOLUME="${RUNPOD_VOLUME_ID:-s2k01690bi}"
GPUS="${RUNPOD_GPU_TYPES:-NVIDIA RTX 4000 Ada Generation,NVIDIA RTX A5000,NVIDIA GeForce RTX 4090,NVIDIA RTX 6000 Ada Generation}"
CUDA="${RUNPOD_ALLOWED_CUDA-13.0}"
CAP_S="${FV_ENDPOINT_CAP_S:-1800}"
MIN_BALANCE="${FV_MIN_BALANCE:-8}"
EXEC_MS="${FV_EXECUTION_TIMEOUT_MS:-1800000}"
MODEL="${FV_SMOKE_MODEL:-fake-wan}"
PREFIX="${FV_ENDPOINT_PREFIX:-fv-serve}"
IDLE_S="${FV_IDLE_TIMEOUT_S:-5}"
LEDGER="${FV_SERVE_LEDGER:-$FV_ROOT/artifacts/runpod/serve/ledger.tsv}"
OUT_DIR="$FV_ROOT/artifacts/runpod/serve"

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
rest2() {
  curl -sS --fail-with-body -X "$1" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${3:+-d "$3"} "$REST2$2"
}
queue() {
  curl -sS --max-time 60 -X "$1" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${3:+-d "$3"} "$QUEUE/$2"
}
ledger() { mkdir -p "$(dirname "$LEDGER")"; printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }
now() { date +%s.%N; }

check_balance() {
  local b
  b="$(curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' "$GQL" \
    -d '{"query":"{ myself { clientBalance } }"}' | jq -r '.data.myself.clientBalance // empty')"
  [[ -n "$b" ]] || die "could not read the Runpod balance"
  awk -v b="$b" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+0)}' || die "Runpod balance \$$b is below the floor \$$MIN_BALANCE"
  log "Runpod balance \$$b (floor \$$MIN_BALANCE)"
}

resolve_digest() {
  local ref="$1" repo tag tok digest
  if [[ "$ref" == *@sha256:* ]]; then echo "$ref"; return; fi
  [[ "$ref" == ghcr.io/* ]] || die "cannot pin $ref: only ghcr.io tags are resolved; pass a digest"
  repo="${ref#ghcr.io/}"; tag="${repo##*:}"; repo="${repo%:*}"
  [[ "$tag" != "$repo" ]] || tag=latest
  tok="$(curl -sS "https://ghcr.io/token?scope=repository:$repo:pull" | jq -r '.token // empty')"
  digest="$(curl -sS -I -H "Authorization: Bearer $tok" \
    -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json' \
    "https://ghcr.io/v2/$repo/manifests/$tag" | tr -d '\r' | awk -F': ' 'tolower($1)=="docker-content-digest"{print $2}')"
  [[ "$digest" == sha256:* ]] || die "could not resolve $ref to a digest (private package, or no such tag)"
  echo "ghcr.io/$repo@$digest"
}

volume_dc() {
  if [[ -n "${RUNPOD_VOLUME_DC:-}" ]]; then echo "$RUNPOD_VOLUME_DC"; return; fi
  rest GET "/networkvolumes/$VOLUME" | jq -r '.dataCenterId'
}

gpu_json() { jq -cn --arg g "$GPUS" '$g | split(",") | map(gsub("^ +| +$"; ""))'; }

# Serverless template (REST v1). The volume's HF-cache trees link
# absolutely into /workspace/weights (where pods mount it); a serverless
# worker mounts it at /runpod-volume, so the entrypoint links
# /workspace/weights there before exec'ing fv-serve (newer fv-serve does the
# same itself; this keeps older images working).
LINK_WEIGHTS='if [ -d /runpod-volume/weights ] && [ ! -e /workspace/weights ]; then mkdir -p /workspace && ln -s /runpod-volume/weights /workspace/weights; fi; exec /opt/fastvideo-rs/bin/fv-serve "$@"'
template_payload() {
  local image="$1" name="$2"
  jq -n --arg name "$name" --arg image "$image" --arg cfg "$CONFIG" --arg link "$LINK_WEIGHTS" --argjson secrets "$SECRET_ENV_JSON" '{
    name: $name, imageName: $image, isServerless: true, containerDiskInGb: 20, volumeInGb: 0,
    dockerEntrypoint: ["/bin/sh", "-c", $link, "fv-serve"], dockerStartCmd: ["--config", $cfg],
    env: ($secrets + {
      FV_SERVE_MODE: "runpod-queue", FV_AUTH_MODE: "trust-gateway", FV_STATE_DIR: "/fvstate",
      FV_WEIGHTS: "/runpod-volume/weights", FV_CACHE_DIR: "/fvstate/cache", RUST_LOG: "info"
    })
  }'
}

# Queue endpoint (REST v1).
endpoint_payload() {
  local tpl="$1" name="$2" dc="$3"
  jq -n --arg name "$name" --arg tpl "$tpl" --argjson gpus "$(gpu_json)" --arg vol "$VOLUME" --arg dc "$dc" \
    --arg cuda "$CUDA" --arg exec "$EXEC_MS" --argjson idle "$IDLE_S" '{
    name: $name, templateId: $tpl, computeType: "GPU", gpuTypeIds: $gpus, gpuCount: 1,
    networkVolumeId: $vol, dataCenterIds: [$dc], workersMin: 0, workersMax: 1, idleTimeout: $idle,
    flashboot: false, executionTimeoutMs: ($exec|tonumber), scalerType: "QUEUE_DELAY", scalerValue: 1,
    allowedCudaVersions: ($cuda | split(" "))
  }'
}

# Load-balancer endpoint (REST v2: the only API with `type`). GPU pools come
# from the catalog for the configured types. FV_WORKERS_MAX follows
# workers.max: above 1 fv-serve serves only routes any worker can answer
# (design §6.5).
LB_WORKERS_MAX="${FV_LB_WORKERS_MAX:-1}"
[[ "$LB_WORKERS_MAX" =~ ^[1-9][0-9]*$ ]] || die "FV_LB_WORKERS_MAX must be a positive integer"
lb_payload() {
  local image="$1" name="$2" dc="$3" pools
  pools="$(curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" "$REST2/catalog/gpus" 2>/dev/null \
    | jq -c --argjson want "$(gpu_json)" '[.gpus[] | select(.id as $i | $want | index($i)) | .pool | select(. != null)] | unique' 2>/dev/null || echo '[]')"
  [[ "$pools" != "[]" && -n "$pools" ]] || pools='["ADA_24"]'
  jq -n --arg name "$name" --arg image "$image" --arg cfg "$CONFIG" --argjson pools "$pools" --arg vol "$VOLUME" \
    --arg dc "$dc" --arg cuda "$CUDA" --argjson secrets "$SECRET_ENV_JSON" --argjson max "$LB_WORKERS_MAX" --argjson idle "$IDLE_S" '{
    name: $name, type: "LOAD_BALANCER", image: $image,
    args: ("--config " + $cfg), ports: ["8000/http"], disk: 20,
    env: ($secrets + {
      FV_SERVE_MODE: "http", FV_AUTH_MODE: "trust-gateway", PORT: "8000", PORT_HEALTH: "8000",
      FV_STATE_DIR: "/fvstate", FV_WEIGHTS: "/runpod-volume/weights", RUST_LOG: "info",
      FV_WORKERS_MAX: ($max | tostring)
    }),
    gpu: ({pools: $pools, count: 1} + (if $cuda == "" then {} else {allowedCudaVersions: ($cuda | split(" "))} end)),
    workers: {min: 0, max: $max, idleTimeout: $idle},
    scaling: {type: "REQUEST_COUNT", requestCount: 1},
    networkVolumes: [$vol], dataCenterIds: [$dc], flashboot: "OFF", timeout: 330000
  }'
}

EP=""
TPL=""
down() {
  local ep="$1" tpl="${2:-}"
  if [[ -n "$ep" ]]; then
    rest PATCH "/endpoints/$ep" '{"workersMin":0,"workersMax":0}' >/dev/null 2>&1 || true
    local i
    for i in 1 2 3 4 5 6; do
      if rest DELETE "/endpoints/$ep" >/dev/null 2>&1; then ledger "endpoint-deleted $ep"; break; fi
      [[ $i == 6 ]] && log "WARNING: could not delete endpoint $ep (backstop retries at the cap)"
      sleep $((i * 5))
    done
  fi
  if [[ -n "$tpl" && "$tpl" != - ]]; then
    if rest DELETE "/templates/$tpl" >/dev/null 2>&1; then ledger "template-deleted $tpl"; else log "WARNING: template $tpl not deleted"; fi
  fi
}
cleanup() {
  local rc=$?
  if [[ -n "$EP" || -n "$TPL" ]]; then
    log "destroy-on-exit: endpoint ${EP:-none} template ${TPL:-none}"
    down "$EP" "$TPL"
    EP=""; TPL=""
  fi
  exit "$rc"
}

backstop() {
  # shellcheck disable=SC2016 # expanded by the child shell
  nohup env EP_ID="$1" TPL_ID="${2:--}" CAP="$CAP_S" API="$REST" bash -c '
    sleep "$CAP"
    h=(-H "Authorization: Bearer $RUNPOD_API_KEY" -H "content-type: application/json")
    curl -sS -X PATCH "${h[@]}" -d "{\"workersMin\":0,\"workersMax\":0}" "$API/endpoints/$EP_ID" >/dev/null
    sleep 20
    curl -sS -X DELETE "${h[@]}" "$API/endpoints/$EP_ID" >/dev/null
    [ "$TPL_ID" = - ] || curl -sS -X DELETE "${h[@]}" "$API/templates/$TPL_ID" >/dev/null' >/dev/null 2>&1 &
}

# Sets EP and TPL.
up_queue() {
  local image="$1" name dc resp
  name="$PREFIX-q-$(date -u +%m%d%H%M%S)"
  dc="$(volume_dc)"
  resp="$(rest POST /templates "$(template_payload "$image" "$name")")" || die "template create failed: $(head -c 300 <<<"$resp")"
  TPL="$(jq -r '.id // empty' <<<"$resp")"
  [[ -n "$TPL" ]] || die "template create returned no id"
  ledger "template-created $TPL $name $image"
  resp="$(rest POST /endpoints "$(endpoint_payload "$TPL" "$name" "$dc")")" || die "endpoint create failed: $(head -c 300 <<<"$resp")"
  EP="$(jq -r '.id // empty' <<<"$resp")"
  [[ -n "$EP" ]] || die "endpoint create returned no id"
  ledger "endpoint-created $EP $name volume=$VOLUME dc=$dc gpus=$GPUS"
  backstop "$EP" "$TPL"
  log "queue endpoint $EP (template $TPL, volume $VOLUME in $dc, cap ${CAP_S}s)"
}

up_lb() {
  local image="$1" name dc resp
  name="$PREFIX-lb-$(date -u +%m%d%H%M%S)"
  dc="$(volume_dc)"
  resp="$(rest2 POST /serverless "$(lb_payload "$image" "$name" "$dc")")" || die "LB endpoint create failed: $(head -c 400 <<<"$resp")"
  EP="$(jq -r '.id // empty' <<<"$resp")"
  [[ -n "$EP" ]] || die "LB endpoint create returned no id: $(head -c 300 <<<"$resp")"
  TPL="$(jq -r '.templateId // .template.id // "-"' <<<"$resp")"
  ledger "endpoint-created $EP $name type=LOAD_BALANCER volume=$VOLUME dc=$dc template=$TPL"
  backstop "$EP" "$TPL"
  log "LB endpoint $EP (https://$EP.api.runpod.ai, cap ${CAP_S}s)"
}

# job <ep> <input json> -> one JSON line with client-side timings.
job() {
  local ep="$1" input="$2" t_submit id st s last="" t_prog="" t_done
  t_submit="$(now)"
  id="$(queue POST "$ep/run" "{\"input\":$input,\"policy\":{\"executionTimeout\":$EXEC_MS}}" | jq -r '.id // empty')"
  [[ -n "$id" ]] || die "submit to $ep failed"
  while :; do
    st="$(queue GET "$ep/status/$id" || echo '{}')"
    s="$(jq -r '.status // "?"' <<<"$st" 2>/dev/null || echo '?')"
    [[ "$s" != "$last" ]] && { log "job $id: $s"; last="$s"; }
    [[ "$s" == IN_PROGRESS && -z "$t_prog" ]] && t_prog="$(now)"
    case "$s" in COMPLETED | FAILED | CANCELLED | TIMED_OUT) t_done="$(now)"; break ;; esac
    (( ${t_submit%.*} + CAP_S > $(date +%s) )) || die "job $id still $s at the cap"
    sleep 2
  done
  jq -c --arg sub "$t_submit" --arg prog "${t_prog:-}" --arg fin "$t_done" --arg id "$id" '{
    job: $id, status, delayTime, executionTime, workerId,
    client_wall_s: (($fin|tonumber) - ($sub|tonumber)),
    first_in_progress_s: (if $prog == "" then null else ($prog|tonumber) - ($sub|tonumber) end),
    output, error}' <<<"$st"
}

cmd_smoke() {
  require_tools curl jq
  : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
  check_balance
  local image cold caps gen warm media_url media=""
  image="$(resolve_digest "${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}")"
  log "image $image"
  trap cleanup EXIT INT TERM
  up_queue "$image"
  cold="$(job "$EP" '{"kind":"info","nvenc":true}')"
  caps="$(job "$EP" '{"kind":"http","method":"GET","path":"/fv/v1/capabilities"}')"
  gen="$(job "$EP" "{\"kind\":\"http\",\"method\":\"POST\",\"path\":\"/fv/v1/jobs\",\"body\":{\"model\":\"$MODEL\",\"prompt\":\"a red fox trotting through fresh snow\",\"seed\":1},\"wait\":true}")"
  warm="$(job "$EP" '{"kind":"info"}')"
  media_url="$(jq -r '.output.body.output.video.url // .output.body.output.url // empty' <<<"$gen")"
  [[ -n "$media_url" ]] && media="$(curl -sS -o /dev/null -w '%{http_code} %{size_download}' --max-time 60 "$media_url" || true)"
  mkdir -p "$OUT_DIR"
  jq -n --arg ep "$EP" --arg image "$image" --arg vol "$VOLUME" --argjson cold "$cold" --argjson caps "$caps" \
    --argjson gen "$gen" --argjson warm "$warm" --arg media "$media" \
    --arg host "$(sed -E 's#^(https://[^/?]+).*#\1#' <<<"$media_url")" '{
    target: "runpod-serverless-queue", endpoint: $ep, image: $image, volume: $vol,
    cold: ($cold | {status, delayTime, executionTime, client_wall_s, first_in_progress_s, workerId,
                    info: (.output | {gpu, ready_after_s, uptime_s, ffmpeg_h264_nvenc, nvenc_encode_ok, nvenc_probe,
                                      nvidia_driver_capabilities, jobs_backend, artifacts_backend, webhook_key_configured,
                                      weights, deploy})}),
    capabilities: ($caps | {status, delayTime, executionTime, client_wall_s, http_status: .output.status,
                            models: [.output.body.models[]?.caps.id]}),
    job: ($gen | {status, delayTime, executionTime, client_wall_s, http_status: .output.status,
                  job_status: .output.body.status, model: .output.body.model, poll_path: .output.poll_path,
                  media_host: $host, media: $media, error}),
    warm: ($warm | {status, delayTime, executionTime, client_wall_s})
  }' | tee "$OUT_DIR/endpoint-smoke-$(date -u +%m%d%H%M%S).json"
  [[ "$(jq -r .status <<<"$gen")" == COMPLETED && "$(jq -r .output.body.status <<<"$gen")" == succeeded ]] \
    || die "the generation job did not succeed"
}

cmd_smoke_lb() {
  require_tools curl jq
  : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
  check_balance
  local image base t0 t_first="" t_ready="" code caps
  image="$(resolve_digest "${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}")"
  trap cleanup EXIT INT TERM
  t0="$(now)"
  up_lb "$image"
  base="https://$EP.api.runpod.ai"
  while :; do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 150 -H "Authorization: Bearer $RUNPOD_API_KEY" "$base/ping" || true)"
    [[ "$code" == 204 && -z "$t_first" ]] && t_first="$(now)"
    [[ "$code" == 200 ]] && { t_ready="$(now)"; break; }
    (( ${t0%.*} + ${FV_BOOT_WAIT_S:-1200} > $(date +%s) )) || die "LB endpoint never answered /ping 200 (last $code)"
    sleep 5
  done
  caps="$(curl -sS --max-time 120 -H "Authorization: Bearer $RUNPOD_API_KEY" "$base/fv/v1/capabilities")"
  jq -n --arg ep "$EP" --arg t0 "$t0" --arg tf "$t_first" --arg tr "$t_ready" --argjson caps "$caps" '{
    target: "runpod-serverless-lb", endpoint: $ep,
    first_204_s: (if $tf == "" then null else ($tf|tonumber) - ($t0|tonumber) end),
    ready_s: (($tr|tonumber) - ($t0|tonumber)), models: [$caps.models[]?.caps.id]}'
}

case "${1:-}" in
  smoke) shift; cmd_smoke "$@" ;;
  smoke-lb) shift; cmd_smoke_lb "$@" ;;
  up)
    shift; : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"; check_balance
    up_queue "$(resolve_digest "${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}")"; echo "$EP $TPL" ;;
  up-lb)
    shift; : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"; check_balance
    up_lb "$(resolve_digest "${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}")"; echo "$EP $TPL" ;;
  job) shift; : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"; job "${1:?endpoint}" "${2:?input json}" ;;
  down) shift; : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"; down "${1:?endpoint}" "${2:-}"; echo "deleted ${1} ${2:-}" ;;
  plan)
    shift
    img="${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}"
    echo "# template (REST v1 POST /templates)"; template_payload "$img" fv-serve-plan
    echo "# queue endpoint (REST v1 POST /endpoints)"; endpoint_payload "<template id>" fv-serve-plan "${RUNPOD_VOLUME_DC:-<volume dc>}"
    echo "# load balancer (REST v2 POST /serverless)"
    RUNPOD_API_KEY="${RUNPOD_API_KEY:-}" lb_payload "$img" fv-serve-plan "${RUNPOD_VOLUME_DC:-<volume dc>}" ;;
  *) sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
