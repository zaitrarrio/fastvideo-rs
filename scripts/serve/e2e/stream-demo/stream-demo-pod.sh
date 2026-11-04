#!/usr/bin/env bash
# A live-stream demo on one Runpod pod: fv-serve with one of the serve presets
# (deploy/deploy.toml; any model `/fv/v1/streams` serves: the causal ones,
# SF-Wan and LongLive, as one continuous rollout, the clip ones, H3, LTX and
# Wan, as a clip queue) publishes a stream over WHIP to a Cloudflare Stream live
# input; viewers play it over WHEP; prompts are steered while it runs
# (docs/serve/stream-outputs.md, docs/serve/console.md §4c).
#
#   stream-demo-pod.sh up        create the pod, wait until fv-serve is ready,
#                                print the demo URL, the console URL and the API key
#   stream-demo-pod.sh plan      print the launch plan and the create payload
#                                (no pod; secrets only as {{ RUNPOD_SECRET_* }})
#   stream-demo-pod.sh status    the pod and the live streams
#   stream-demo-pod.sh down      delete the pod (also deleted at FV_POD_CAP_S by
#                                a detached backstop, default 4 h)
#
# The pod runs the one CUDA serve image through its own entrypoint, fv-boot
# (docs/serve/images.md, ADR-0014): the preset's worker config goes inline
# (FV_WORKER_TOML_B64) and the weights come from FV_DEMO_WEIGHTS:
#   volume  (default) the EU weights volume (scripts/gpu/volumes.sh: bz0ofkfx45
#           in EU-RO-1, H3 + LTX trees only) mounted read-only at /workspace;
#           the pod runs in that data centre on its GPU types. A preset whose
#           trees are not on it is refused with the remedy.
#   r2      the R2 weights store (docs/ops/r2-weights.md): the pod mints its
#           plan with r2-weights.py from the account's `fv_r2_*` Runpod secrets
#           and fetches the trees with r2-fetch.py onto its container disk
#           (sized from scripts/gpu/weights-r2.tsv), in any data centre.
#   hub     the Hugging Face Hub at the pinned revisions
#           (scripts/gpu/weights-revisions.tsv) through fv-boot's hub source
#           (fetch-hub-tree.py; the recorded hashes checked), any data centre.
# stream-demo-plan.py resolves the preset's trees against the source.
#
# Alongside fv-serve (:8000, API keys, the console on) the pod serves the demo
# page on :8001 (python3 http.server); both go through Runpod's HTTPS proxy:
# https://<pod>-8000.proxy.runpod.net (API + /console/stream) and
# https://<pod>-8001.proxy.runpod.net/demo.html.
#
# Env: RUNPOD_API_KEY; CLOUDFLARE_API_KEY (Account > Stream > Edit) or
# FV_CF_STREAM_API_TOKEN; FV_CF_ACCOUNT_ID (default the repo's account);
# FV_DEMO_PRESET (a deploy/deploy.toml preset id or a configs/serve file;
# default sfwan); FV_DEMO_WEIGHTS (volume | r2 | hub; default volume);
# FV_DEMO_SCRIPTS_REF (the git ref the pod fetches scripts/gpu from for r2 and
# hub; default HEAD, which must be pushed); RUNPOD_GPU_TYPES ('|' or ','
# separated; default: the volume's types with volume, else H100 / RTX PRO 6000
# / H100 NVL); RUNPOD_GPU_MAX_DPH (4.0); FV_POD_CAP_S (14400); FV_MIN_BALANCE
# (8); FV_DEMO_READY_S (2400); FV_SERVE_IMAGE (digest ref; default: :cuda).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
# shellcheck source-path=SCRIPTDIR source=../../../gpu/runpod-price.sh
source "$ROOT/scripts/gpu/runpod-price.sh"
# shellcheck source-path=SCRIPTDIR source=../../../gpu/runpod-cuda.sh
source "$ROOT/scripts/gpu/runpod-cuda.sh"
# shellcheck source-path=SCRIPTDIR source=../../../gpu/volumes.sh
source "$ROOT/scripts/gpu/volumes.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="https://api.runpod.io/graphql"
PRESET="${FV_DEMO_PRESET:-sfwan}"
WEIGHTS="${FV_DEMO_WEIGHTS:-volume}"
case "$WEIGHTS" in volume | r2 | hub) ;; *) echo "FV_DEMO_WEIGHTS=$WEIGHTS: volume | r2 | hub" >&2; exit 2 ;; esac
MAX_DPH="${RUNPOD_GPU_MAX_DPH:-4.0}"
CAP_S="${FV_POD_CAP_S:-14400}"
MIN_BALANCE="${FV_MIN_BALANCE:-8}"
READY_S="${FV_DEMO_READY_S:-2400}"
CF_ACCOUNT="${FV_CF_ACCOUNT_ID:-0cd06fd37ee4e07de370821cb3852a8a}"
CF_TOKEN="${FV_CF_STREAM_API_TOKEN:-${CLOUDFLARE_API_KEY:-}}"
STATE="${FV_DEMO_STATE:-$HOME/.config/fv/stream-demo.state}"
LEDGER="$ROOT/artifacts/runpod/serve/ledger.tsv"
SCRIPTS_REF="${FV_DEMO_SCRIPTS_REF:-$(git -C "$ROOT" rev-parse HEAD)}"
SCRIPTS_URL="https://raw.githubusercontent.com/zaitrarrio/fastvideo-rs/$SCRIPTS_REF/scripts/gpu"
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

# The launch plan: the preset's config and trees against the weights source
# (exit 2 with the reason when the source cannot serve it).
PLAN="$(python3 -I "$HERE/stream-demo-plan.py" --root "$ROOT" --preset "$PRESET" --source "$WEIGHTS")" || exit 2
plan() { jq -r "$@" <<<"$PLAN"; }
DISK_GB="$(plan .disk_gb)"
WEIGHTS_DIR="$(plan .weights)"
if [[ "$WEIGHTS" == volume ]]; then
  fv_check_volume "$(plan .volume.id)"
  GPUS="${RUNPOD_GPU_TYPES:-$(plan .volume.gpus)}"
else
  GPUS="${RUNPOD_GPU_TYPES:-NVIDIA H100 80GB HBM3|NVIDIA RTX PRO 6000 Blackwell Server Edition|NVIDIA H100 NVL}"
fi

# The pod's start command: the demo page on :8001, the weights where fv-boot
# does not fetch them itself (r2), then fv-boot as the image would run it.
pod_cmd() {
  cat <<'EOF'
set -u
mkdir -p /fvscratch/demo /fvscratch/runs; echo "$FV_DEMO_HTML_B64" | base64 -d >/fvscratch/demo/demo.html
( cd /fvscratch/demo && exec python3 -m http.server 8001 --bind 0.0.0.0 ) >/fvscratch/runs/demo-http.log 2>&1 &
export FV_PUBLIC_BASE_URL="https://${RUNPOD_POD_ID}-8000.proxy.runpod.net"
t0=$(date +%s); say() { echo "[stream-demo] [$(( $(date +%s) - t0 ))s] $*"; }
if [ -n "${FV_DEMO_R2_TREES:-}" ]; then
  # The R2 weights store without fv-control: the plan is minted here with the
  # account's R2 key (docs/ops/r2-weights.md §3; the weights bucket is always
  # fv-weights, whatever bucket the media secrets name), then fetched with the
  # repo's fetcher, resuming a partial tree across passes.
  say "weights: r2 ($FV_DEMO_R2_TREES) -> $FV_WEIGHTS"
  mkdir -p /fvscratch/r2 "$FV_WEIGHTS"
  for f in r2-weights.py r2-fetch.py weights-revisions.tsv weights-manifest.tsv weights-r2.tsv weights-sha256.tsv; do
    curl -fsSL --retry 5 -o "/fvscratch/r2/$f" "$FV_DEMO_SCRIPTS_URL/$f" || { say "fetch $f failed"; exit 1; }
  done
  # shellcheck disable=SC2086 # the tree list is words
  python3 -I /fvscratch/r2/r2-weights.py plan $FV_DEMO_R2_TREES --ttl 7200 --out /fvscratch/r2/plan.json >/fvscratch/runs/r2-plan.log 2>&1 \
    || { say "r2 plan failed: $(tail -n 3 /fvscratch/runs/r2-plan.log | tr '\n' ' ')"; exit 1; }
  ( while :; do { date -u +%T; df -h /fvscratch | tail -1; free -g | sed -n 2p; } >>/fvscratch/runs/sys.log; sleep 30; done ) &
  ok=0
  for pass in 1 2 3 4; do
    if python3 -I /fvscratch/r2/r2-fetch.py --root "$FV_WEIGHTS" --plan /fvscratch/r2/plan.json --status /fvscratch/runs/r2-status.json --timeout 30 >>/fvscratch/runs/r2-fetch.log 2>&1; then ok=1; break; fi
    say "r2 fetch pass $pass failed; resuming"
  done
  [ "$ok" = 1 ] || { say "r2 fetch failed: $(grep -a FAILED /fvscratch/runs/r2-fetch.log | tail -n 1 | cut -c1-240)"; exit 1; }
  say "weights fetched: $(du -sh "$FV_WEIGHTS" | cut -f1)"
fi
say "fv-boot (weights $FV_WEIGHTS_SOURCE at $FV_WEIGHTS)"
exec /opt/fastvideo-rs/bin/fv-boot
EOF
}

payload() { # image gpu name keyhash
  local src_env='{}' vol='{}' dcs='[]'
  case "$WEIGHTS" in
    volume)
      vol="$(plan '{networkVolumeId: .volume.id, volumeMountPath: .volume.mount}')"
      dcs="$(plan '[.volume.dc]')"
      src_env='{"FV_WEIGHTS_SOURCE": "volume"}' ;;
    r2)
      # The account's R2 key, resolved by Runpod inside the pod only; fv-boot
      # then starts on the fetched root with no source of its own.
      src_env="$(plan --arg url "$SCRIPTS_URL" '{
        FV_WEIGHTS_SOURCE: "none", FV_DEMO_R2_TREES: (.r2.trees | join(" ")), FV_DEMO_SCRIPTS_URL: $url,
        FV_WEIGHTS_R2_ACCESS_KEY_ID: "{{ RUNPOD_SECRET_fv_r2_access_key_id }}",
        FV_WEIGHTS_R2_SECRET_ACCESS_KEY: "{{ RUNPOD_SECRET_fv_r2_secret_access_key }}",
        FV_WEIGHTS_R2_ENDPOINT: "{{ RUNPOD_SECRET_fv_r2_endpoint }}", FV_WEIGHTS_R2_BUCKET: "fv-weights"}')" ;;
    hub)
      src_env="$(plan --arg url "$SCRIPTS_URL" '{
        FV_WEIGHTS_SOURCE: "hub", FV_WEIGHTS_TREES_B64: .hub.rows_b64, FV_HUB_SHA256_B64: .hub.sha_b64, FV_SCRIPTS_URL: $url}')" ;;
  esac
  jq -n --arg name "$3" --arg image "$1" --arg gpu "$2" --arg keys "$4" --argjson disk "$DISK_GB" \
    --arg toml "$(plan .config_b64)" --arg html "$(base64 -w0 <"$HERE/demo.html")" --arg cmd "$(pod_cmd)" \
    --arg weights "$WEIGHTS_DIR" --arg cf_account "$CF_ACCOUNT" --arg cf_token "$CF_TOKEN" \
    --argjson src "$src_env" --argjson vol "$vol" --argjson dcs "$dcs" \
    --arg cuda "${RUNPOD_ALLOWED_CUDA-$(fv_runpod_cuda)}" '{
    name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU",
    gpuTypeIds: [$gpu], gpuCount: 1, containerDiskInGb: $disk, volumeInGb: 0,
    ports: ["8000/http", "8001/http"],
    dockerEntrypoint: ["/bin/bash", "-c"], dockerStartCmd: [$cmd],
    env: ({
      NVIDIA_DRIVER_CAPABILITIES: "compute,utility,video",
      FV_WORKER_TOML_B64: $toml, FV_WEIGHTS: $weights, FV_STATE_DIR: "/fvstate",
      FV_API_KEYS: $keys, FV_AUTH_MODE: "keys", FV_SERVE_MODE: "http", FV_CONSOLE: "1",
      FV_JOB_STORE: "memory", FV_ARTIFACTS: "local", FV_LOG_FORMAT: "text", RUST_LOG: "info",
      FV_CF_ACCOUNT_ID: $cf_account, FV_CF_STREAM_API_TOKEN: $cf_token,
      FV_DEMO_HTML_B64: $html
    } + $src)}
    + $vol + (if $dcs == [] then {} else {dataCenterIds: $dcs} end)
    + (if $cuda == "" then {} else {allowedCudaVersions: ($cuda | split(" "))} end)'
}

describe() {
  log "preset $(plan .preset) ($(plan .config); models $(plan '.variants | join(", ")'))"
  case "$WEIGHTS" in
    volume) log "weights: the EU volume $(plan .volume.id) at $(plan .volume.mount) in $(plan .volume.dc); trees $(plan '.trees | join(", ")')" ;;
    r2) log "weights: R2 weights store, $(plan '.bytes / 1e9 | floor') GB onto a $DISK_GB GB container disk: $(plan '.r2.trees | join(", ")')" ;;
    hub) log "weights: the Hub at the pinned revisions, $(plan '.bytes / 1e9 | floor') GB onto a $DISK_GB GB container disk: $(plan '.hub.trees | join(", ")')"
      [[ "$(plan '.hub.skipped | length')" == 0 ]] || log "  no Hub source for $(plan '.hub.skipped | join(", ")') (pre-quantized trees): the engine quantizes at load" ;;
  esac
}

cmd_plan() {
  describe
  IFS='|' read -r -a types <<<"${GPUS//,/|}"
  payload "ghcr.io/zaitrarrio/fastvideo-rs-serve:cuda" "${types[0]}" "fv-stream-demo-plan" "<sha256 of the run key>" \
    | jq '.env.FV_DEMO_HTML_B64 = "<demo.html>" | .env.FV_WORKER_TOML_B64 = "<\(.env.FV_WORKER_TOML_B64 | @base64d | length) bytes of config>" | .dockerStartCmd = ["<\(.dockerStartCmd[0] | length) bytes>"]'
}

cmd_up() {
  : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
  [[ -n "$CF_TOKEN" ]] || die "no Cloudflare Stream token (FV_CF_STREAM_API_TOKEN or CLOUDFLARE_API_KEY)"
  [[ -z "$(st pod)" ]] || die "state $STATE already has pod $(st pod); run down first"
  describe
  if [[ "$WEIGHTS" != volume ]] && ! git -C "$ROOT" merge-base --is-ancestor "$SCRIPTS_REF" "$(git -C "$ROOT" rev-parse '@{upstream}' 2>/dev/null || echo "$SCRIPTS_REF")" 2>/dev/null; then
    log "note: $SCRIPTS_REF is not on the upstream branch; the pod fetches scripts/gpu from $SCRIPTS_URL (push first, or set FV_DEMO_SCRIPTS_REF)"
  fi
  local b image key keyhash name gpu resp pod dph
  b="$(balance)"; [[ -n "$b" ]] || die "no balance"
  awk -v b="$b" -v m="$MIN_BALANCE" -v d="$MAX_DPH" -v c="$CAP_S" 'BEGIN{exit !(b - d*c/3600 >= m)}' \
    || die "balance \$$b would drop below \$$MIN_BALANCE at the cap (${CAP_S}s at \$$MAX_DPH/h)"
  log "balance \$$b"
  image="$(resolve_digest "${FV_SERVE_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-serve:cuda}")"
  key="fvk-$(openssl rand -hex 16)"; keyhash="$(printf '%s' "$key" | sha256sum | cut -d' ' -f1)"
  name="fv-stream-demo-$(date -u +%m%d%H%M)"
  IFS='|' read -r -a types <<<"${GPUS//,/|}"
  for gpu in "${types[@]}"; do
    fv_runpod_price_ok "$gpu" "$MAX_DPH" SECURE || continue
    if ! resp="$(rest POST /pods "$(payload "$image" "$gpu" "$name" "$keyhash")" 2>&1)"; then
      log "no pod on $gpu: $(head -c 200 <<<"$resp")"; continue
    fi
    pod="$(jq -r '.id // empty' <<<"$resp")"; dph="$(jq -r '.costPerHr // 0' <<<"$resp")"
    [[ -n "$pod" ]] || { log "no id: $(head -c 200 <<<"$resp")"; continue; }
    ledger "pod-created $pod $name gpu=$gpu dph=$dph image=$image demo preset=$(plan .preset) weights=$WEIGHTS"
    if awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 > c+0)}'; then
      rest DELETE "/pods/$pod" >/dev/null || true; ledger "pod-deleted $pod over-cap"; log "$gpu at \$$dph/hr > cap"; continue
    fi
    umask 077
    { echo "pod=$pod"; echo "key=$key"; echo "gpu=$gpu"; echo "dph=$dph"; echo "preset=$(plan .preset)"; echo "weights=$WEIGHTS"; echo "created=$(date +%s)"; } >"$STATE"
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
    (( $(date +%s) - t0 < READY_S )) || { log "fv-serve not ready after ${READY_S}s: see $base and the pod's container log"; break; }
    sleep 20
  done
  local h; h="$(curl -sS --max-time 20 "$base/health" 2>/dev/null || true)"
  log "health: ${h:0:200}"
  printf '\nDemo page:   https://%s-8001.proxy.runpod.net/demo.html\nConsole:     %s/console/stream\nAPI:         %s\nAPI key:     %s\nModels:      %s\n\n' \
    "$pod" "$base" "$base" "$key" "$(plan '.variants | join(", ")')"
}

cmd_status() {
  : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
  local pod; pod="$(st pod)"; [[ -n "$pod" ]] || die "no pod in $STATE"
  rest GET "/pods/$pod" | jq -c '{id, desiredStatus, costPerHr, machine: .machine.gpuTypeId}'
  curl -sS --max-time 20 -H "Authorization: Bearer $(st key)" "https://$pod-8000.proxy.runpod.net/fv/v1/streams" | jq -c '.data[]? | {id, mode, state: .status.state, output: .status.output_state, playback: .playback.whep_url}' 2>/dev/null || true
}

cmd_down() {
  : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
  local pod; pod="$(st pod)"; [[ -n "$pod" ]] || die "no pod in $STATE"
  rest DELETE "/pods/$pod" >/dev/null && ledger "pod-deleted $pod demo" && log "deleted $pod"
  rm -f "$STATE"
}

case "${1:-}" in
  up) cmd_up ;;
  plan) cmd_plan ;;
  status) cmd_status ;;
  down) cmd_down ;;
  *) sed -n '2,45p' "$0"; exit 2 ;;
esac
