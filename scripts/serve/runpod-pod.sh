#!/usr/bin/env bash
# fv-serve on a Runpod pod (docs/serve/design.md §6.2 "Runpod pod"; WP-16).
#
#   runpod-pod.sh smoke [image]   create a pod running fv-serve, wait for /ping 200,
#                                 check /healthz, /fv/v1/capabilities, the info
#                                 envelope (NVENC encode) and one small job
#                                 (R2 media, D1 record), print timings, delete it
#   runpod-pod.sh up [image]      create; print "<pod id> <api key>" (the key is
#                                 per run: its SHA-256 goes in FV_API_KEYS)
#   runpod-pod.sh down <pod>      delete a pod
#   runpod-pod.sh plan [image]    print the create payload (no API call; secrets
#                                 appear only as {{ RUNPOD_SECRET_* }} references)
#
# The image must be digest-pinned (…@sha256:…); a tag (default
# ghcr.io/zaitrarrio/fastvideo-rs-serve:latest) is resolved to its digest first.
# fv-serve starts from the image ENTRYPOINT with `--config $FV_SERVE_CONFIG`.
#
# Money guards: the GPU's $/hr cap (RUNPOD_GPU_MAX_DPH, default 1.0), a
# wall-clock cap (FV_POD_CAP_S, default 1800 s) enforced by a detached
# backstop that deletes the pod even if this shell dies, a destroy-on-exit
# trap, a balance floor (FV_MIN_BALANCE, default 8 $), and a ledger
# (artifacts/runpod/serve/ledger.tsv).
#
# Env: RUNPOD_API_KEY; FV_SERVE_IMAGE; FV_SERVE_CONFIG (default
# /etc/fv/runpod-fake.toml: the fake engine until the CUDA backend lands;
# /etc/fv/runpod.toml for real models); RUNPOD_GPU_TYPES (comma list, cheapest
# first); RUNPOD_VOLUME_ID (optional weight volume at /workspace, read only:
# state goes to /fvstate on the container disk); RUNPOD_ALLOWED_CUDA (default
# 13.0; set it empty to drop the filter, which can hide stock); FV_SMOKE_MODEL (default fake-wan, the FastWan stand-in).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/lib.sh
source "$HERE/../gpu/lib.sh"

API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
IMAGE_DEFAULT="ghcr.io/zaitrarrio/fastvideo-rs-serve:latest"
CONFIG="${FV_SERVE_CONFIG:-/etc/fv/runpod-fake.toml}"
GPUS="${RUNPOD_GPU_TYPES:-NVIDIA RTX A4000,NVIDIA RTX A4500,NVIDIA RTX 4000 Ada Generation,NVIDIA RTX A5000,NVIDIA RTX 2000 Ada Generation,NVIDIA GeForce RTX 3090,NVIDIA L4}"
MAX_DPH="${RUNPOD_GPU_MAX_DPH:-1.0}"
CAP_S="${FV_POD_CAP_S:-1800}"
MIN_BALANCE="${FV_MIN_BALANCE:-8}"
CUDA="${RUNPOD_ALLOWED_CUDA-13.0}"
MODEL="${FV_SMOKE_MODEL:-fake-wan}"
LEDGER="${FV_SERVE_LEDGER:-$FV_ROOT/artifacts/runpod/serve/ledger.tsv}"
OUT_DIR="$FV_ROOT/artifacts/runpod/serve"

# The seven Cloudflare values plus the fal webhook key, as Runpod secret
# references (design §0 decision 7, §6.3). Never values.
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
  local method="$1" path="$2" body="${3:-}"
  curl -sS --fail-with-body -X "$method" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${body:+-d "$body"} "$API$path"
}
ledger() { mkdir -p "$(dirname "$LEDGER")"; printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }
now() { date +%s.%N; }

runpod_balance() {
  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' "$GQL" \
    -d '{"query":"{ myself { clientBalance } }"}' | jq -r '.data.myself.clientBalance // empty'
}

check_balance() {
  local b
  b="$(runpod_balance)"
  [[ -n "$b" ]] || die "could not read the Runpod balance"
  awk -v b="$b" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+0)}' \
    || die "Runpod balance \$$b is below the floor \$$MIN_BALANCE"
  log "Runpod balance \$$b (floor \$$MIN_BALANCE)"
}

# ghcr tag -> digest reference (public packages: anonymous pull token).
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

# $1 image, $2 gpu type, $3 name, $4 api-key hash -> PodCreateInput JSON
payload() {
  local image="$1" gpu="$2" name="$3" keyhash="$4" vol_json='{}'
  if [[ -n "${RUNPOD_VOLUME_ID:-}" ]]; then
    local dc
    dc="$(rest GET "/networkvolumes/$RUNPOD_VOLUME_ID" | jq -r '.dataCenterId')"
    vol_json="$(jq -n --arg v "$RUNPOD_VOLUME_ID" --arg dc "$dc" '{networkVolumeId: $v, volumeMountPath: "/workspace", dataCenterIds: [$dc]}')"
  fi
  jq -n --arg name "$name" --arg image "$image" --arg gpu "$gpu" --arg cfg "$CONFIG" --arg cuda "$CUDA" \
    --arg keys "$keyhash" --argjson secrets "$SECRET_ENV_JSON" --argjson vol "$vol_json" '{
      name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU",
      gpuTypeIds: [$gpu], gpuCount: 1, containerDiskInGb: 30, volumeInGb: 0,
      ports: ["8000/http", "70000/tcp"],
      dockerEntrypoint: ["/opt/fastvideo-rs/bin/fv-serve"],
      dockerStartCmd: ["--config", $cfg],
      env: ($secrets + {
        FV_SERVE_MODE: "http", FV_STATE_DIR: "/fvstate", FV_WEIGHTS: "/workspace/weights",
        FV_API_KEYS: $keys, FV_SERVE_FORWARD: "1", RUST_LOG: "info"
      })
    } + $vol + (if $cuda == "" then {} else {allowedCudaVersions: ($cuda | split(" "))} end)'
}

POD=""
cleanup() {
  local rc=$?
  if [[ -n "$POD" ]]; then
    log "destroy-on-exit: deleting pod $POD"
    if rest DELETE "/pods/$POD" >/dev/null 2>&1; then ledger "pod-deleted $POD"; else log "WARNING: delete of $POD failed; the ${CAP_S}s backstop will retry"; fi
    POD=""
  fi
  exit "$rc"
}

# Creates the pod on the first GPU type with capacity under the $/hr cap.
# Sets POD; prints nothing. $1 image, $2 api-key hash.
create() {
  local image="$1" keyhash="$2" gpu resp dph name
  name="${FV_POD_NAME_PREFIX:-fv-serve-smoke}-$(date -u +%m%d%H%M%S)"
  IFS=',' read -r -a types <<<"$GPUS"
  for gpu in "${types[@]}"; do
    if ! resp="$(rest POST /pods "$(payload "$image" "$gpu" "$name" "$keyhash")" 2>&1)"; then
      log "no pod on $gpu: $(head -c 200 <<<"$resp")"
      continue
    fi
    POD="$(jq -r '.id // empty' <<<"$resp")"
    [[ -n "$POD" ]] || { log "create returned no id: $(head -c 300 <<<"$resp")"; continue; }
    ledger "pod-created $POD $name gpu=$gpu image=$image"
    dph="$(jq -r '.costPerHr // 0' <<<"$resp")"
    if awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 > c+0)}'; then
      log "pod $POD on $gpu costs \$$dph/hr > cap \$$MAX_DPH: deleting"
      rest DELETE "/pods/$POD" >/dev/null || true
      ledger "pod-deleted $POD over-cap"
      POD=""
      continue
    fi
    # Wall-clock backstop, detached from this shell; the key stays in its env.
    # shellcheck disable=SC2016 # expanded by the child shell
    nohup env POD_ID="$POD" CAP="$CAP_S" API="$API" bash -c \
      'sleep "$CAP"; curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/pods/$POD_ID" >/dev/null' \
      >/dev/null 2>&1 &
    log "pod $POD: $gpu at \$$dph/hr (wall-clock cap ${CAP_S}s)"
    CREATED_GPU="$gpu"; CREATED_DPH="$dph"
    return 0
  done
  die "no GPU type in [$GPUS] could be created under \$$MAX_DPH/hr"
}

http() { curl -sS --max-time 60 "$@"; }

cmd_smoke() {
  require_tools curl jq openssl sha256sum
  : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
  check_balance
  local image key keyhash base t_create t_first="" t_ready="" code t0 info caps job id st result
  image="$(resolve_digest "${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}")"
  log "image $image"
  key="fvk-$(openssl rand -hex 16)"
  keyhash="$(printf '%s' "$key" | sha256sum | cut -d' ' -f1)"
  trap cleanup EXIT INT TERM
  t_create="$(now)"
  create "$image" "$keyhash"
  base="https://$POD-8000.proxy.runpod.net"
  # /ping: 502/404 while the container starts, 204 while loading, 200 ready.
  t0=$(date +%s)
  while :; do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$base/ping" || true)"
    [[ "$code" == 204 || "$code" == 200 ]] && [[ -z "$t_first" ]] && t_first="$(now)"
    [[ "$code" == 200 ]] && { t_ready="$(now)"; break; }
    (( $(date +%s) - t0 < ${FV_BOOT_WAIT_S:-1200} )) || die "pod $POD never answered /ping 200 (last $code)"
    sleep 5
  done
  log "ready: first /ping answer $(awk -v a="$t_first" -v b="$t_create" 'BEGIN{printf "%.1f", a-b}')s, ready $(awk -v a="$t_ready" -v b="$t_create" 'BEGIN{printf "%.1f", a-b}')s after create"
  http "$base/healthz" | jq -c . >&2
  caps="$(http -H "Authorization: Bearer $key" "$base/fv/v1/capabilities")"
  jq -e '.models | length > 0' <<<"$caps" >/dev/null || die "capabilities: $(head -c 300 <<<"$caps")"
  info="$(http --max-time 120 -H 'content-type: application/json' -d '{"kind":"info","nvenc":true}' "$base/fv/v1/forward")"
  jq -c '{gpu, ffmpeg_h264_nvenc, nvenc_encode_ok, nvidia_driver_capabilities, jobs_backend, artifacts_backend, webhook_key_configured, deploy: .deploy.platform, public_base_url: .deploy.public_base_url}' <<<"$info" >&2
  # One small job through the native API; the output goes to R2.
  t0="$(now)"
  job="$(http -H "Authorization: Bearer $key" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"a red fox trotting through fresh snow\",\"seed\":1}" "$base/fv/v1/jobs")"
  id="$(jq -r '.id // empty' <<<"$job")"
  [[ -n "$id" ]] || die "job submit: $(head -c 300 <<<"$job")"
  for _ in $(seq 1 120); do
    st="$(http -H "Authorization: Bearer $key" "$base/fv/v1/jobs/$id")"
    case "$(jq -r .status <<<"$st")" in succeeded | failed | cancelled) break ;; esac
    sleep 2
  done
  result="$(curl -sS -o /dev/null -w '%{http_code} %{redirect_url}' -H "Authorization: Bearer $key" "$base/fv/v1/jobs/$id/content")"
  local media_url="${result#* }" media_code="" media_bytes=""
  if [[ -n "$media_url" ]]; then
    read -r media_code media_bytes < <(curl -sS -o /dev/null -w '%{http_code} %{size_download}' --max-time 60 "$media_url") || true
  fi
  mkdir -p "$OUT_DIR"
  jq -n --arg pod "$POD" --arg gpu "$CREATED_GPU" --arg dph "$CREATED_DPH" --arg image "$image" \
    --arg tc "$t_create" --arg tf "$t_first" --arg tr "$t_ready" --arg tj "$t0" --arg te "$(now)" \
    --argjson st "$st" --argjson info "$info" --arg content "${result%% *}" --arg mcode "$media_code" --arg mbytes "$media_bytes" \
    --arg host "$(sed -E 's#^(https://[^/?]+).*#\1#' <<<"$media_url")" '{
      target: "runpod-pod", pod: $pod, gpu: $gpu, usd_per_hr: ($dph|tonumber), image: $image,
      create_to_first_ping_s: (($tf|tonumber) - ($tc|tonumber)),
      create_to_ready_s: (($tr|tonumber) - ($tc|tonumber)),
      job: {id: $st.id, model: $st.model, status: $st.status, wall_s: (($te|tonumber) - ($tj|tonumber)),
            content_status: ($content|tonumber), media_host: $host, media_status: $mcode, media_bytes: $mbytes},
      info: ($info | {gpu, ffmpeg_h264_nvenc, nvenc_encode_ok, nvenc_probe, nvidia_driver_capabilities, jobs_backend, artifacts_backend,
                      webhook_key_configured, ready_after_s, deploy})
    }' | tee "$OUT_DIR/pod-smoke-$(date -u +%m%d%H%M%S).json"
  [[ "$(jq -r .status <<<"$st")" == succeeded ]] || die "job $id ended $(jq -c '{status, error}' <<<"$st")"
}

cmd_up() {
  : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
  check_balance
  local image key
  image="$(resolve_digest "${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}")"
  key="fvk-$(openssl rand -hex 16)"
  create "$image" "$(printf '%s' "$key" | sha256sum | cut -d' ' -f1)"
  echo "$POD $key"
  POD=""
}

case "${1:-}" in
  smoke) shift; cmd_smoke "$@" ;;
  up) shift; cmd_up "$@" ;;
  down)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    rest DELETE "/pods/${2:?pod id}" >/dev/null && ledger "pod-deleted $2" && echo "deleted $2" ;;
  plan)
    shift
    IFS=',' read -r -a types <<<"$GPUS"
    payload "${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}" "${types[0]}" fv-serve-plan "<sha256 of the run key>" ;;
  *) sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
