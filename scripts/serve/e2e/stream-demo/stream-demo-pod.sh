#!/usr/bin/env bash
# A live-stream demo on one Runpod pod: fv-serve with the causal SF-Wan model
# publishes one continuous rollout over WHIP to a Cloudflare Stream live input;
# viewers play it over WHEP; prompts are switched while it runs
# (docs/serve/stream-outputs.md, docs/serve/console.md §4c).
#
#   stream-demo-pod.sh up        create the pod, wait until fv-serve is ready,
#                                print the demo URL, the console URL and the API key
#   stream-demo-pod.sh status    the pod and the live streams
#   stream-demo-pod.sh down      delete the pod (also deleted at FV_POD_CAP_S by
#                                a detached backstop, default 4 h)
#
# What runs on the pod (the one CUDA serve image; its fv-boot entrypoint is
# bypassed: this script fetches the weights and starts fv-serve itself):
# - the SF-Wan tree fetched from the Hub onto the container disk (27 GB,
#   about 2 min; nothing on a volume), TAEHV from its pinned commit;
# - fv-serve on :8000 (`configs/serve/runpod-sfwan.toml` + `[outputs.cloudflare]`
#   from FV_CF_ACCOUNT_ID / FV_CF_STREAM_API_TOKEN, API keys, the console on);
# - the demo page on :8001 (python3 http.server).
# Both ports go through Runpod's HTTPS proxy: https://<pod>-8000.proxy.runpod.net
# (API + /console/stream) and https://<pod>-8001.proxy.runpod.net/demo.html.
#
# GPU: H100 first (SF-Wan at about 2x realtime; no NVENC, so WHIP H.264 is the
# CPU encoder at 832x480), then RTX PRO 6000 (NVENC, about 1x realtime).
# Env: RUNPOD_API_KEY, CLOUDFLARE_API_KEY (Account > Stream > Edit) or
# FV_CF_STREAM_API_TOKEN, FV_CF_ACCOUNT_ID (default the repo's account),
# RUNPOD_GPU_TYPES, RUNPOD_GPU_MAX_DPH (4.0), FV_POD_CAP_S (14400),
# FV_MIN_BALANCE (8), FV_SERVE_IMAGE (digest ref; default: the :cuda tag resolved).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
# shellcheck source-path=SCRIPTDIR source=../../../gpu/runpod-price.sh
source "$ROOT/scripts/gpu/runpod-price.sh"
# shellcheck source-path=SCRIPTDIR source=../../../gpu/runpod-cuda.sh
source "$ROOT/scripts/gpu/runpod-cuda.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="https://api.runpod.io/graphql"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
GPUS="${RUNPOD_GPU_TYPES:-NVIDIA H100 80GB HBM3|NVIDIA RTX PRO 6000 Blackwell Server Edition|NVIDIA H100 NVL}"
MAX_DPH="${RUNPOD_GPU_MAX_DPH:-4.0}"
CAP_S="${FV_POD_CAP_S:-14400}"
MIN_BALANCE="${FV_MIN_BALANCE:-8}"
CF_ACCOUNT="${FV_CF_ACCOUNT_ID:-0cd06fd37ee4e07de370821cb3852a8a}"
CF_TOKEN="${FV_CF_STREAM_API_TOKEN:-${CLOUDFLARE_API_KEY:-}}"
STATE="${FV_DEMO_STATE:-$HOME/.config/fv/stream-demo.state}"
LEDGER="$ROOT/artifacts/runpod/serve/ledger.tsv"
mkdir -p "$(dirname "$STATE")" "$(dirname "$LEDGER")"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "$*"; exit 1; }
ledger() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }
rest() {
  curl -sS --fail-with-body -X "$1" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${3:+-d "$3"} "$API$2"
}
balance() {
  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' \
    -d '{"query":"{ myself { clientBalance } }"}' "$GQL" | jq -r '.data.myself.clientBalance // empty'
}
st() { [[ -r "$STATE" ]] && sed -n "s/^$1=//p" "$STATE" || true; }

resolve_digest() { # ref -> ghcr.io/repo@sha256:…
  local ref="$1" repo tag tok digest
  [[ "$ref" == *@sha256:* ]] && { echo "$ref"; return; }
  repo="${ref#ghcr.io/}"; tag="${repo##*:}"; repo="${repo%:*}"
  tok="$(curl -sS "https://ghcr.io/token?scope=repository:$repo:pull" | jq -r '.token // empty')"
  digest="$(curl -sS -I -H "Authorization: Bearer $tok" \
    -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json' \
    "https://ghcr.io/v2/$repo/manifests/$tag" | tr -d '\r' | awk -F': ' 'tolower($1)=="docker-content-digest"{print $2}')"
  [[ "$digest" == sha256:* ]] || die "could not resolve $ref"
  echo "ghcr.io/$repo@$digest"
}

# The pod's start command. Everything it needs beyond the image arrives in
# the env: the config, the TAE fetcher and the demo page, base64.
pod_cmd() {
  cat <<EOF
set -u
mkdir -p /fvscratch/demo /fvscratch/runs; echo "\$FV_DEMO_HTML_B64" | base64 -d >/fvscratch/demo/demo.html
( cd /fvscratch/demo && exec python3 -m http.server 8001 --bind 0.0.0.0 ) >/fvscratch/runs/demo-http.log 2>&1 &
exec >/fvscratch/runs/boot.log 2>&1
t0=\$(date +%s); say() { echo "[\$(( \$(date +%s) - t0 ))s] \$*"; }
say "weights"
W=/fvscratch/weights/sfwan21-1.3b; REPO=wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers; REV=4b44356635ae5e927ca552a220f768022be76004
mkdir -p "\$W"
curl -fsSL "https://huggingface.co/api/models/\$REPO/tree/\$REV?recursive=true" \
  | grep -oE '"path":"(model_index.json|scheduler/[^"]+|tokenizer/[^"]+|text_encoder/[^"]+|transformer/[^"]+|vae/[^"]+)"' \
  | sed 's/"path":"//;s/"\$//' >/fvscratch/runs/files.txt
while read -r f; do mkdir -p "\$W/\$(dirname "\$f")"; echo "\$f"; done </fvscratch/runs/files.txt \
  | xargs -P 8 -I{} curl -fsSL --retry 5 -o "\$W/{}" "https://huggingface.co/\$REPO/resolve/\$REV/{}"
echo "\$FV_TAE_B64" | base64 -d >/fetch_taehv.sh
FV_AUX_DIR=/nonexistent bash /fetch_taehv.sh /fvscratch/tae >/fvscratch/runs/tae.log 2>&1 || say "tae fetch failed"
echo "\$FV_DEMO_TOML_B64" | base64 -d >/fvscratch/serve.toml
say "fv-serve"
FV_WEIGHTS=/fvscratch/weights FASTVIDEO_TAE_DIR=/fvscratch/tae FV_BIND=0.0.0.0:8000 \
  FV_PUBLIC_BASE_URL="https://\${RUNPOD_POD_ID}-8000.proxy.runpod.net" \
  FV_STATE_DIR=/fvscratch/state FV_JOB_STORE=memory FV_ARTIFACTS=local FV_LOG_FORMAT=text \
  exec /opt/fastvideo-rs/bin/fv-serve --config /fvscratch/serve.toml
EOF
}

demo_toml() { # the SF-Wan preset; the Cloudflare output's account and token come from the env
  grep -v -E '^\s*(state_dir|pool)\s*=' "$ROOT/configs/serve/runpod-sfwan.toml"
}

payload() { # image gpu name keyhash
  jq -n --arg name "$3" --arg image "$1" --arg gpu "$2" --arg keys "$4" \
    --arg toml "$(demo_toml | base64 -w0)" --arg html "$(base64 -w0 <"$HERE/demo.html")" \
    --arg tae "$(base64 -w0 <"$ROOT/scripts/gpu/fetch_taehv.sh")" --arg cmd "$(pod_cmd)" \
    --arg cf_account "$CF_ACCOUNT" --arg cf_token "$CF_TOKEN" \
    --arg cuda "${RUNPOD_ALLOWED_CUDA-$(fv_runpod_cuda)}" '{
    name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU",
    gpuTypeIds: [$gpu], gpuCount: 1, containerDiskInGb: 80, volumeInGb: 0,
    ports: ["8000/http", "8001/http"],
    dockerEntrypoint: ["/bin/bash", "-c"], dockerStartCmd: [$cmd],
    env: {
      NVIDIA_DRIVER_CAPABILITIES: "compute,utility,video",
      FV_API_KEYS: $keys, FV_AUTH_MODE: "keys", FV_SERVE_MODE: "http", RUST_LOG: "info",
      FV_CF_ACCOUNT_ID: $cf_account, FV_CF_STREAM_API_TOKEN: $cf_token,
      FV_DEMO_TOML_B64: $toml, FV_DEMO_HTML_B64: $html, FV_TAE_B64: $tae
    }} + (if $cuda == "" then {} else {allowedCudaVersions: ($cuda | split(" "))} end)'
}

cmd_up() {
  [[ -n "$CF_TOKEN" ]] || die "no Cloudflare Stream token (FV_CF_STREAM_API_TOKEN or CLOUDFLARE_API_KEY)"
  [[ -z "$(st pod)" ]] || die "state $STATE already has pod $(st pod); run down first"
  local b image key keyhash name gpu resp pod dph
  b="$(balance)"; [[ -n "$b" ]] || die "no balance"
  awk -v b="$b" -v m="$MIN_BALANCE" -v d="$MAX_DPH" -v c="$CAP_S" 'BEGIN{exit !(b - d*c/3600 >= m)}' \
    || die "balance \$$b would drop below \$$MIN_BALANCE at the cap (${CAP_S}s at \$$MAX_DPH/h)"
  log "balance \$$b"
  image="$(resolve_digest "${FV_SERVE_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-serve:cuda}")"
  key="fvk-$(openssl rand -hex 16)"; keyhash="$(printf '%s' "$key" | sha256sum | cut -d' ' -f1)"
  name="fv-stream-demo-$(date -u +%m%d%H%M)"
  IFS='|' read -r -a types <<<"$GPUS"
  for gpu in "${types[@]}"; do
    fv_runpod_price_ok "$gpu" "$MAX_DPH" SECURE || continue
    if ! resp="$(rest POST /pods "$(payload "$image" "$gpu" "$name" "$keyhash")" 2>&1)"; then
      log "no pod on $gpu: $(head -c 200 <<<"$resp")"; continue
    fi
    pod="$(jq -r '.id // empty' <<<"$resp")"; dph="$(jq -r '.costPerHr // 0' <<<"$resp")"
    [[ -n "$pod" ]] || { log "no id: $(head -c 200 <<<"$resp")"; continue; }
    ledger "pod-created $pod $name gpu=$gpu dph=$dph image=$image demo"
    if awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 > c+0)}'; then
      rest DELETE "/pods/$pod" >/dev/null || true; ledger "pod-deleted $pod over-cap"; log "$gpu at \$$dph/hr > cap"; continue
    fi
    umask 077
    { echo "pod=$pod"; echo "key=$key"; echo "gpu=$gpu"; echo "dph=$dph"; echo "created=$(date +%s)"; } >"$STATE"
    # Detached backstop: the pod goes at the cap even if this shell dies.
    # shellcheck disable=SC2016
    setsid nohup env POD_ID="$pod" CAP="$CAP_S" API="$API" bash -c \
      'sleep "$CAP"; curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/pods/$POD_ID" >/dev/null' \
      >/dev/null 2>&1 < /dev/null &
    log "pod $pod ($name) on $gpu at \$$dph/hr; deleted after ${CAP_S}s"
    break
  done
  [[ -n "${pod:-}" ]] || die "no pod created"
  local base="https://$pod-8000.proxy.runpod.net" t0
  t0=$(date +%s)
  until curl -sS --max-time 20 -o /dev/null -w '%{http_code}' "$base/health" 2>/dev/null | grep -q '^200$'; do
    (( $(date +%s) - t0 < 1500 )) || { log "fv-serve not ready after 25 min: see $base and the boot log"; break; }
    sleep 20
  done
  local h; h="$(curl -sS --max-time 20 "$base/health" 2>/dev/null || true)"
  log "health: ${h:0:200}"
  printf '\nDemo page:   https://%s-8001.proxy.runpod.net/demo.html\nConsole:     %s/console/stream\nAPI:         %s\nAPI key:     %s\n\n' "$pod" "$base" "$base" "$key"
}

cmd_status() {
  local pod; pod="$(st pod)"; [[ -n "$pod" ]] || die "no pod in $STATE"
  rest GET "/pods/$pod" | jq -c '{id, desiredStatus, costPerHr, machine: .machine.gpuTypeId}'
  curl -sS --max-time 20 -H "Authorization: Bearer $(st key)" "https://$pod-8000.proxy.runpod.net/fv/v1/streams" | jq -c '.data[]? | {id, state, output: .output.state, playback: .playback.whep_url}' 2>/dev/null || true
}

cmd_down() {
  local pod; pod="$(st pod)"; [[ -n "$pod" ]] || die "no pod in $STATE"
  rest DELETE "/pods/$pod" >/dev/null && ledger "pod-deleted $pod demo" && log "deleted $pod"
  rm -f "$STATE"
}

case "${1:-}" in
  up) cmd_up ;;
  status) cmd_status ;;
  down) cmd_down ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
esac
