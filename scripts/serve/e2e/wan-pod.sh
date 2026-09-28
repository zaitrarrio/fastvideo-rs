#!/usr/bin/env bash
# GPU E2E for the Wan family on one Runpod pod (WP-18, design §7.6):
# FastWan batch through the public APIs, then SF-Wan live (native WHIP +
# WHEP viewer, Reactor causal) run by wan-pod-run.sh on the pod.
#
#   wan-pod.sh up [image]    create the pod (prints the pod id; the run key and
#                            state go to $FV_E2E_STATE, never to stdout)
#   wan-pod.sh batch         wait for phase 1, run wan_batch.py, signal phase 2
#   wan-pod.sh fetch         wait for DONE, download the results
#   wan-pod.sh down          delete the pod and verify it is gone
#
# Money guards: RUNPOD_GPU_MAX_DPH (default 3.6), a balance floor
# (FV_MIN_BALANCE, default 8 $), a detached backstop that deletes the pod
# after FV_POD_CAP_S (default 5400 s) even if this shell dies, and a ledger
# (artifacts/runpod/serve/ledger.tsv). The weight volume (RUNPOD_VOLUME_ID,
# default s2k01690bi, US-CA-2) is mounted at /workspace and only read.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="https://api.runpod.io/graphql"
STATE="${FV_E2E_STATE:-${TMPDIR:-/tmp}/fv-e2e-wan.state}"
GPUS="${RUNPOD_GPU_TYPES:-NVIDIA H100 80GB HBM3|NVIDIA H100 NVL|NVIDIA H100 PCIe}"
MAX_DPH="${RUNPOD_GPU_MAX_DPH:-3.6}"
CAP_S="${FV_POD_CAP_S:-5400}"
MIN_BALANCE="${FV_MIN_BALANCE:-8}"
VOLUME="${RUNPOD_VOLUME_ID:-s2k01690bi}"
OUT="${FV_E2E_OUT:-$ROOT/artifacts/serve/e2e/wan}"
LEDGER="$ROOT/artifacts/runpod/serve/ledger.tsv"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
rest() {
  curl -sS --fail-with-body -X "$1" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${3:+-d "$3"} "$API$2"
}
ledger() { mkdir -p "$(dirname "$LEDGER")"; printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }
st() { sed -n "s/^$1=//p" "$STATE" 2>/dev/null | tail -1; }
put() { echo "$1=$2" >>"$STATE"; }

balance() {
  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' "$GQL" \
    -d '{"query":"{ myself { clientBalance } }"}' | jq -r '.data.myself.clientBalance // empty'
}

resolve_digest() {
  local ref="$1" repo tag tok digest
  [[ "$ref" == *@sha256:* ]] && { echo "$ref"; return; }
  repo="${ref#ghcr.io/}"; tag="${repo##*:}"; repo="${repo%:*}"
  tok="$(curl -sS "https://ghcr.io/token?scope=repository:$repo:pull" | jq -r '.token // empty')"
  digest="$(curl -sS -I -H "Authorization: Bearer $tok" \
    -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json' \
    "https://ghcr.io/v2/$repo/manifests/$tag" | tr -d '\r' | awk -F': ' 'tolower($1)=="docker-content-digest"{print $2}')"
  [[ "$digest" == sha256:* ]] || die "could not resolve $ref"
  echo "ghcr.io/$repo@$digest"
}

payload() { # image gpu name keyhash
  local py run dc
  py="$(tar cz -C "$HERE" whep_viewer.py reactor_causal.py | base64 -w0)"
  run="$(base64 -w0 <"$HERE/wan-pod-run.sh")"
  dc="$(rest GET "/networkvolumes/$VOLUME" | jq -r '.dataCenterId')"
  jq -n --arg name "$3" --arg image "$1" --arg gpu "$2" --arg keys "$4" --arg py "$py" --arg run "$run" \
    --arg vol "$VOLUME" --arg dc "$dc" --arg tag "$(date -u +%m%d%H%M)" \
    --arg skip1 "${FV_E2E_SKIP_PHASE1:-0}" --arg live "${FV_E2E_LIVE_S:-300}" '{
    name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU",
    gpuTypeIds: [$gpu], gpuCount: 1, containerDiskInGb: 60, volumeInGb: 0,
    networkVolumeId: $vol, volumeMountPath: "/workspace", dataCenterIds: [$dc],
    allowedCudaVersions: ["13.0"],
    ports: ["8000/http", "8001/http", "70000/tcp"],
    dockerEntrypoint: ["/bin/bash", "-c"],
    dockerStartCmd: ["echo \"$FV_E2E_RUN_B64\" | base64 -d >/wan-pod-run.sh && exec bash /wan-pod-run.sh"],
    env: {
      FV_CF_ACCOUNT_ID: "{{ RUNPOD_SECRET_fv_cf_account_id }}",
      FV_CF_API_TOKEN: "{{ RUNPOD_SECRET_fv_cf_api_token }}",
      FV_D1_DATABASE_ID: "{{ RUNPOD_SECRET_fv_d1_database_id }}",
      FV_R2_BUCKET: "{{ RUNPOD_SECRET_fv_r2_bucket }}",
      FV_R2_ENDPOINT: "{{ RUNPOD_SECRET_fv_r2_endpoint }}",
      FV_R2_ACCESS_KEY_ID: "{{ RUNPOD_SECRET_fv_r2_access_key_id }}",
      FV_R2_SECRET_ACCESS_KEY: "{{ RUNPOD_SECRET_fv_r2_secret_access_key }}",
      FV_WEBHOOK_ED25519_KEY: "{{ RUNPOD_SECRET_fv_webhook_ed25519_key }}",
      FV_API_KEYS: $keys, FV_SERVE_MODE: "http", RUST_LOG: "info",
      FV_E2E_TAG: $tag, FV_E2E_PY_B64: $py, FV_E2E_RUN_B64: $run,
      FV_E2E_SKIP_PHASE1: $skip1, FV_E2E_LIVE_S: $live
    }}'
}

cmd_up() {
  : "${RUNPOD_API_KEY:?}"
  [[ -z "$(st pod)" ]] || die "state $STATE already has pod $(st pod); run down first"
  local b image key keyhash name gpu resp pod dph
  b="$(balance)"; [[ -n "$b" ]] || die "no balance"
  awk -v b="$b" -v m="$MIN_BALANCE" -v d="$MAX_DPH" -v c="$CAP_S" 'BEGIN{exit !(b - d*c/3600 >= m)}' \
    || die "balance \$$b would drop below \$$MIN_BALANCE at the cap"
  log "balance \$$b"
  image="$(resolve_digest "${1:-${FV_SERVE_IMAGE:?FV_SERVE_IMAGE}}")"
  key="fvk-$(openssl rand -hex 16)"; keyhash="$(printf '%s' "$key" | sha256sum | cut -d' ' -f1)"
  name="fv-e2e-d-$(date -u +%m%d%H%M%S)"
  IFS='|' read -r -a types <<<"$GPUS"
  local until=$(( $(date +%s) + ${FV_E2E_RETRY_S:-0} ))
  while :; do
  for gpu in "${types[@]}"; do
    if ! resp="$(rest POST /pods "$(payload "$image" "$gpu" "$name" "$keyhash")" 2>&1)"; then
      log "no pod on $gpu: $(head -c 200 <<<"$resp")"; continue
    fi
    pod="$(jq -r '.id // empty' <<<"$resp")"; dph="$(jq -r '.costPerHr // 0' <<<"$resp")"
    [[ -n "$pod" ]] || { log "no id: $(head -c 200 <<<"$resp")"; continue; }
    ledger "pod-created $pod $name gpu=$gpu image=$image"
    if awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 > c+0)}'; then
      rest DELETE "/pods/$pod" >/dev/null || true; ledger "pod-deleted $pod over-cap"; log "$gpu at \$$dph/hr > cap"; continue
    fi
    umask 077
    { echo "pod=$pod"; echo "key=$key"; echo "gpu=$gpu"; echo "dph=$dph"; echo "image=$image"; echo "created=$(date +%s)"; } >"$STATE"
    # shellcheck disable=SC2016
    setsid nohup env POD_ID="$pod" CAP="$CAP_S" API="$API" bash -c \
      'sleep "$CAP"; curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/pods/$POD_ID" >/dev/null' \
      >/dev/null 2>&1 < /dev/null &
    log "pod $pod ($name) on $gpu at \$$dph/hr; backstop ${CAP_S}s"
    echo "$pod"
    return 0
  done
  (( $(date +%s) < until )) || break
  log "no capacity; retrying in 60 s"
  sleep 60
  done
  die "no pod created"
}

proxy() { curl -sS --max-time 30 --fail "https://$(st pod)-$1.proxy.runpod.net/$2"; }

cmd_batch() {
  local pod key t0
  pod="$(st pod)"; key="$(st key)"; [[ -n "$pod" ]] || die "no pod"
  mkdir -p "$OUT/batch"
  t0=$(date +%s)
  until proxy 8001 e2e/phases.txt 2>/dev/null | grep -qE '^p1-(ready|failed)'; do
    (( $(date +%s) - t0 < 2400 )) || die "phase 1 never became ready"
    log "waiting for phase 1: $(proxy 8001 e2e/live.log 2>/dev/null | tail -1 || echo 'no files yet')"
    sleep 20
  done
  log "phase 1 ready ($(proxy 8001 e2e/phase1-ready-s.txt 2>/dev/null) s load)"
  "${FV_E2E_PY:-python3}" "$HERE/wan_batch.py" --base "https://$pod-8000.proxy.runpod.net" --key "$key" --out "$OUT/batch" || true
  curl -sS -o /dev/null -w 'phase-2 signal %{http_code}\n' -X DELETE -H "Authorization: Bearer $key" \
    "https://$pod-8000.proxy.runpod.net/fv/v1/streams/e2e-phase1-done" >&2
}

cmd_fetch() {
  local t0 f
  local LD="$OUT/${FV_E2E_LIVE_DIR:-live}"
  mkdir -p "$LD"
  t0=$(date +%s)
  until proxy 8001 e2e/phases.txt 2>/dev/null | grep -q '^done'; do
    (( $(date +%s) - t0 < ${FV_E2E_FETCH_WAIT_S:-3000} )) || { log "no DONE; fetching what exists"; break; }
    log "pod: $(proxy 8001 e2e/live.log 2>/dev/null | tail -1)"
    sleep 30
  done
  for f in live.log phases.txt box.txt setup.log wan.toml serve-wan.log phase1-ready-s.txt phase1-health.json phase1-info.json \
    phase1-metrics.txt phase1-gpu.csv mediamtx.log sfwan-sfwan.toml serve-sfwan.log phase2-ready-s.txt phase2-capabilities.json \
    phase2-gpu.csv live-post.json live-create.json live-streaming.json live-status-start.json live-status.jsonl live-switches.jsonl \
    live-status-end.json live-status-final.json live-clock.json mediamtx-paths.json viewer.json viewer.log viewer-frames.jsonl \
    rtsp.log rtsp-packets.csv rtsp-ffprobe.json sample-switch.mp4 reactor-session-before.json reactor.json reactor.log \
    reactor-session-after.json sfwan-sfwan-loopback.toml serve-sfwan-loopback.log phase2b-ready-s.txt reactor-loopback.json \
    reactor-loopback.log phase2-metrics.txt gpucheck.log gpucheck-files.txt DONE; do
    curl -sS --max-time 300 --fail "https://$(st pod)-8001.proxy.runpod.net/e2e/$f" -o "$LD/$f" 2>/dev/null || rm -f "$LD/$f"
  done
  for f in $(grep -oE '/fvscratch/gpucheck/[^ ]+\.json' "$LD/gpucheck-files.txt" 2>/dev/null | xargs -rn1 basename); do
    mkdir -p "$LD/gpucheck"
    curl -sS --max-time 60 --fail "https://$(st pod)-8001.proxy.runpod.net/e2e/gpucheck/$f" -o "$LD/gpucheck/$f" || true
  done
  log "results in $LD"
}

cmd_down() {
  local pod
  pod="$(st pod)"; [[ -n "$pod" ]] || die "no pod in $STATE"
  rest DELETE "/pods/$pod" >/dev/null || log "delete call failed"
  sleep 5
  if rest GET "/pods/$pod" >/dev/null 2>&1; then
    log "WARNING: pod $pod still listed"; rest GET "/pods/$pod" | jq -c '{id, desiredStatus}' >&2 || true; exit 1
  fi
  ledger "pod-deleted $pod"
  log "pod $pod deleted ($(( ($(date +%s) - $(st created)) ))s after create, \$$(st dph)/hr)"
  mv "$STATE" "$STATE.done.$(date +%s)"
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  batch) cmd_batch ;;
  fetch) cmd_fetch ;;
  down) cmd_down ;;
  *) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
