#!/usr/bin/env bash
# Runpod driver for the H3 / LTX / Hunyuan sweep.
# Weights first on a 1 TB network volume (cheap CPU pod + hf-fm). GPU pods
# only after every tree has a .complete marker. Hard cap: 3 GPU pods.
#
#   runpod.sh fetch     volume + CPU download + verify
#   runpod.sh verify    re-check shards on a live fetch pod or via SSH
#   runpod.sh gpu       three PRO 6000 96 GB pods (refuses if volume incomplete)
#   runpod.sh status    volume / pods / cost
#   runpod.sh reap      destroy fv-* GPU and fetch pods (keeps the volume)
set -euo pipefail
# shellcheck source=scripts/gpu/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

RUNPOD_API_BASE="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
RUNPOD_GQL="${RUNPOD_GQL:-https://api.runpod.io/graphql}"
RP_STATE="$FV_ROOT/artifacts/runpod"
RP_LIVE="$RP_STATE/live.log"
RP_SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
RP_IMAGE="${RUNPOD_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-runtime:latest}"
RP_FETCH_IMAGE="${RUNPOD_FETCH_IMAGE:-$RP_IMAGE}"
RP_GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA RTX PRO 6000 Blackwell Server Edition}"
RP_VOL_NAME="${RUNPOD_VOLUME_NAME:-fv-weights-h3-ltx-hy}"
RP_VOL_GB="${RUNPOD_VOLUME_GB:-1000}"
RP_MAX_GPU_PODS=3
RP_GPU_MAX_DPH="${RUNPOD_GPU_MAX_DPH:-2.50}"
MOUNT="/workspace"
WEIGHTS="$MOUNT/weights"
FETCH_NAME_PREFIX="fv-fetch"
GPU_NAME_PREFIX="fv-gpu"

mkdir -p "$RP_STATE"

rp_log() {
  local line
  line="$(printf '[%s] %s' "$(date -u +%H:%M:%S)" "$*")"
  printf '%s\n' "$line" | tee -a "$RP_LIVE" >&2
}

rp_load_key() {
  if [[ -z "${RUNPOD_API_KEY:-}" && -f "$HOME/.runpod/config.toml" ]]; then
    RUNPOD_API_KEY="$(python3 -c "import tomllib,pathlib; print(tomllib.loads((pathlib.Path.home()/'.runpod'/'config.toml').read_text())['apikey'])")"
    export RUNPOD_API_KEY
  fi
  [[ -n "${RUNPOD_API_KEY:-}" ]] || die "RUNPOD_API_KEY missing (env or ~/.runpod/config.toml)"
  if [[ -z "${HF_TOKEN:-}" && -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' <"${HF_HOME:-$HOME/.cache/huggingface}/token")"
    export HF_TOKEN
  fi
}

rp_rest() {
  local method="$1" path="$2" data="${3:-}"
  local resp_file http_code
  resp_file="$(mktemp)"
  local -a args=(-sS -w '%{http_code}' -o "$resp_file" -X "$method" "${RUNPOD_API_BASE}${path}"
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" -H "Content-Type: application/json")
  [[ -z "$data" ]] || args+=(-d "$data")
  http_code="$(curl "${args[@]}")"
  if [[ "$http_code" -lt 200 || "$http_code" -ge 300 ]]; then
    rp_log "FATAL: REST $method $path HTTP $http_code"
    python3 -c 'import sys; print(sys.stdin.read()[:2000])' <"$resp_file" | tee -a "$RP_LIVE" >&2
    rm -f "$resp_file"
    return 1
  fi
  cat "$resp_file"
  rm -f "$resp_file"
}

rp_gql() {
  local query="$1"
  curl -sS -H "Authorization: Bearer ${RUNPOD_API_KEY}" -H "Content-Type: application/json" \
    "$RUNPOD_GQL" --data-binary "$(jq -n --arg q "$query" '{query:$q}')"
}

rp_pubkey() {
  if [[ -f "${RP_SSH_KEY}.pub" ]]; then
    cat "${RP_SSH_KEY}.pub"
  else
    ssh-keygen -y -f "$RP_SSH_KEY"
  fi
}

rp_ssh() {
  local host="$1" port="$2"; shift 2
  ssh -i "$RP_SSH_KEY" -p "$port" -o IdentitiesOnly=yes "${FV_SSH_OPTS[@]}" "root@$host" "$@"
}

rp_rsync() {
  local host="$1" port="$2" src="$3" dst="$4"; shift 4
  rsync -az "$@" -e "ssh -i $RP_SSH_KEY -p $port -o IdentitiesOnly=yes ${FV_SSH_OPTS[*]}" "$src" "root@$host:$dst"
}

rp_pod_ssh_target() {
  local id="$1" raw host port
  raw="$(rp_rest GET "/pods/${id}" 2>/dev/null || true)"
  host="$(jq -r '.publicIp // empty' <<<"${raw:-}" 2>/dev/null || true)"
  port="$(jq -r '[.runtime.ports[]? | select(.privatePort==22)][0].publicPort // empty' <<<"${raw:-}" 2>/dev/null || true)"
  if [[ -z "$host" || -z "$port" || "$host" == "null" ]]; then
    raw="$(rp_gql 'query { myself { pods { id runtime { ports { ip isIpPublic privatePort publicPort type } } } } }')"
    host="$(jq -r --arg id "$id" '.data.myself.pods[] | select(.id==$id) | [.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true)][0].ip // empty' <<<"$raw")"
    port="$(jq -r --arg id "$id" '.data.myself.pods[] | select(.id==$id) | [.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true)][0].publicPort // empty' <<<"$raw")"
  fi
  [[ -n "$host" && -n "$port" && "$host" != "null" ]] || return 1
  printf '%s %s\n' "$host" "$port"
}

rp_wait_ssh() {
  local id="$1" timeout_s="${2:-600}" t0 host port status
  t0=$(date +%s)
  while :; do
    status="$(rp_rest GET "/pods/${id}" | jq -r '.desiredStatus // .lastStatusChange // "unknown"')"
    if read -r host port < <(rp_pod_ssh_target "$id" 2>/dev/null) && rp_ssh "$host" "$port" true 2>/dev/null; then
      rp_log "ssh ok root@$host:$port (pod $id)"
      printf '%s %s\n' "$host" "$port"
      return 0
    fi
    if (( $(date +%s) - t0 >= timeout_s )); then
      rp_log "FATAL: ssh to $id never came up (status=$status host=${host:-?} port=${port:-?})"
      return 1
    fi
    sleep 8
  done
}

rp_destroy_pod() {
  local id="$1"
  [[ -n "$id" ]] || return 0
  rp_log "destroying pod $id"
  rp_rest DELETE "/pods/${id}" >/dev/null || true
}

# --- volume ------------------------------------------------------------------

rp_pick_dc() {
  if [[ -n "${RUNPOD_DATACENTER:-}" ]]; then
    echo "$RUNPOD_DATACENTER"
    return 0
  fi
  local dc query resp stock
  # Prefer DCs that currently list PRO 6000 Server 96 GB and are in the REST enum.
  for dc in EUR-IS-1 EUR-IS-2 CA-MTL-3 EU-NL-1 EU-CZ-1 US-NE-1 US-MD-1 US-MO-2 US-NC-2; do
    query="query { gpuTypes(input: {id: \"${RP_GPU_TYPE}\"}) { lowestPrice(input: {gpuCount: 1, dataCenterId: \"${dc}\"}) { stockStatus uninterruptablePrice } } }"
    resp="$(rp_gql "$query")"
    stock="$(jq -r '.data.gpuTypes[0].lowestPrice.stockStatus // empty' <<<"$resp")"
    if [[ -n "$stock" && "$stock" != "null" ]]; then
      rp_log "dc $dc PRO 6000 stock=$stock \$$(jq -r '.data.gpuTypes[0].lowestPrice.uninterruptablePrice' <<<"$resp")/hr"
      echo "$dc"
      return 0
    fi
  done
  die "no datacenter currently lists $RP_GPU_TYPE"
}

rp_find_or_create_volume() {
  local vols existing id dc
  vols="$(rp_rest GET /networkvolumes)"
  existing="$(jq -r --arg n "$RP_VOL_NAME" --argjson gb "$RP_VOL_GB" \
    '.[] | select(.name==$n and .size>=$gb) | "\(.id)\t\(.dataCenterId)\t\(.size)"' <<<"$vols" | head -1)"
  if [[ -n "$existing" ]]; then
    id="${existing%%$'\t'*}"
    rp_log "reuse volume $existing"
    printf '%s\n' "$id"
    jq -n --arg id "$id" --arg raw "$existing" '{id:$id, reuse:true}' >"$RP_STATE/volume.json"
    return 0
  fi
  dc="$(rp_pick_dc)"
  rp_log "create volume name=$RP_VOL_NAME size=${RP_VOL_GB}GB dc=$dc"
  local payload resp
  payload="$(jq -n --arg name "$RP_VOL_NAME" --argjson size "$RP_VOL_GB" --arg dc "$dc" \
    '{name:$name, size:$size, dataCenterId:$dc}')"
  resp="$(rp_rest POST /networkvolumes "$payload")"
  id="$(jq -r '.id // empty' <<<"$resp")"
  [[ -n "$id" ]] || die "volume create returned no id: $resp"
  rp_log "volume id=$id name=$RP_VOL_NAME ${RP_VOL_GB}GB dc=$dc"
  printf '%s\n' "$resp" >"$RP_STATE/volume.json"
  printf '%s\n' "$id"
}

rp_volume_dc() {
  local id="$1"
  rp_rest GET "/networkvolumes/${id}" | jq -r '.dataCenterId'
}

rp_ensure_registry_auth() {
  if [[ -n "${RUNPOD_REGISTRY_AUTH_ID:-}" ]]; then
    echo "$RUNPOD_REGISTRY_AUTH_ID"
    return 0
  fi
  local token user payload resp id
  token="${GHCR_TOKEN:-}"
  if [[ -z "$token" ]] && command -v gh >/dev/null; then
    token="$(gh auth token 2>/dev/null || true)"
  fi
  [[ -n "$token" ]] || { echo ""; return 0; }
  user="${GHCR_USERNAME:-zaitrarrio}"
  payload="$(jq -n --arg name "fv-ghcr-$(date +%s)" --arg user "$user" --arg pass "$token" \
    '{name:$name, username:$user, password:$pass}')"
  resp="$(rp_rest POST /containerregistryauth "$payload" 2>/dev/null || true)"
  id="$(jq -r '.id // empty' <<<"$resp")"
  if [[ -n "$id" ]]; then
    rp_log "ghcr registry auth id=$id user=$user"
    echo "$id"
  else
    rp_log "ghcr registry auth skipped (image pull may fail if private)"
    echo ""
  fi
}

# --- weight trees ------------------------------------------------------------
# dest<TAB>hub<TAB>space-separated hf-fm globs (same families as validate.sh)

rp_weight_rows() {
  cat <<'EOF'
h3-8step	FastVideo/FastVideo-FastH3-8-Step-V2	tokenizer/* text_encoder/*.json text_encoder/model-0000[1-9]-of-00014.safetensors text_encoder/model-0001[01]-of-00014.safetensors text_encoder/model-0001[2-4]-of-00014.safetensors transformer/* audio_vae/* vae/*
h3-base	MiniMaxAI/MiniMax-H3	tokenizer/* text_encoder/*.json text_encoder/model-0000[1-9]-of-00014.safetensors text_encoder/model-0001[01]-of-00014.safetensors text_encoder/model-0001[2-4]-of-00014.safetensors transformer/* audio_vae/* vae/*
FastH3-4-step-Preview-v1-LoRA	FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA	dense-datafree/adapter_model.safetensors
FastH3-4-step-Preview-v1-VSA-DataFree	FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree	adapter_model.safetensors *.safetensors
upscaler	LBH-123-AI/Minimax_h3_latent_Upscaler	minimax_h3_latent_upscaler_3d_bf16.safetensors
h3-to-ltx	Efficient-Large-Model/H3-to-LTX-Latent-Adapter	config.json model.safetensors
ltx2	Lightricks/LTX-2	tokenizer/* text_encoder/model-* text_encoder/*.json vae/* audio_vae/* vocoder/* ltx-2-19b-distilled.safetensors
ltx25	Lightricks/LTX-2.5-Diffusers	tokenizer/* text_encoder/model-* text_encoder/*.json connectors/* transformer/* vae/* audio_vae/* vocoder/* latent_upsampler/* ltx-2.5-22b-distilled-lora-450-bf16.safetensors
ltx23	FastVideo/LTX-2.3-Distilled-Diffusers	tokenizer/* text_encoder/model-* text_encoder/*.json text_embedding_projection/* transformer/* vae/* audio_vae/* vocoder/* spatial_upscaler/* latent_upsampler/* ltx-2.3-22b-distilled-lora-384-1.1.safetensors ltx-2.3-22b-distilled-lora-384.safetensors
hy15-480-t2v	hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v	tokenizer/* tokenizer_2/* text_encoder/* text_encoder_2/* transformer/* vae/* scheduler/*
hy15-480-i2v	hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled	tokenizer/* tokenizer_2/* text_encoder/* text_encoder_2/* transformer/* vae/* scheduler/*
hy15-720-t2v	hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v	tokenizer/* tokenizer_2/* text_encoder/* text_encoder_2/* transformer/* vae/* scheduler/*
hy15-720-i2v	hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled	tokenizer/* tokenizer_2/* text_encoder/* text_encoder_2/* transformer/* vae/* scheduler/*
EOF
}

rp_volume_ready_file() { echo "$RP_STATE/volume-ready.json"; }

# --- fetch pod ---------------------------------------------------------------

rp_create_cpu_pod() {
  local vol="$1" dc="$2" auth="${3:-}"
  local name pubkey payload resp id
  name="${FETCH_NAME_PREFIX}-$(date -u +%Y%m%d%H%M%S)"
  pubkey="$(rp_pubkey)"
  payload="$(jq -n \
    --arg name "$name" \
    --arg image "$RP_FETCH_IMAGE" \
    --arg vol "$vol" \
    --arg dc "$dc" \
    --arg pubkey "$pubkey" \
    --arg hf "${HF_TOKEN:-}" \
    --arg auth "$auth" \
    '{
      name: $name,
      imageName: $image,
      cloudType: "SECURE",
      computeType: "CPU",
      cpuFlavorIds: ["cpu3c","cpu5c","cpu3g"],
      cpuFlavorPriority: "availability",
      vcpuCount: 8,
      containerDiskInGb: 20,
      volumeInGb: 0,
      networkVolumeId: $vol,
      volumeMountPath: "/workspace",
      ports: ["22/tcp"],
      dockerStartCmd: ["/bin/bash","-lc","mkdir -p /root/.ssh /run/sshd; printf '%s\\n' \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; exec sleep infinity"],
      env: {
        PUBLIC_KEY: $pubkey,
        HF_TOKEN: $hf,
        HUGGING_FACE_HUB_TOKEN: $hf,
        HF_HOME: "/workspace/hf",
        HF_HUB_ENABLE_HF_TRANSFER: "1"
      }
    } + (if $auth != "" then {containerRegistryAuthId:$auth} else {} end)')"
  rp_log "create CPU fetch pod name=$name image=$RP_FETCH_IMAGE dc=$dc vol=$vol"
  resp="$(rp_rest POST /pods "$payload")"
  id="$(jq -r '.id // empty' <<<"$resp")"
  [[ -n "$id" ]] || die "CPU pod create failed: $resp"
  local dph
  dph="$(jq -r '.costPerHr // 0' <<<"$resp")"
  rp_log "cpu pod $id  \$${dph}/hr  image=$RP_FETCH_IMAGE  dc=$dc"
  printf '%s\n' "$resp" >"$RP_STATE/fetch-pod.json"
  printf '%s\n' "$id"
}

rp_seed_token() {
  local host="$1" port="$2"
  [[ -n "${HF_TOKEN:-}" ]] || return 0
  rp_ssh "$host" "$port" "mkdir -p /root/.cache/huggingface $MOUNT/hf && chmod 700 /root/.cache/huggingface $MOUNT/hf"
  printf '%s\n' "$HF_TOKEN" | rp_ssh "$host" "$port" "cat > /root/.cache/huggingface/token && cp -f /root/.cache/huggingface/token $MOUNT/hf/token && chmod 600 /root/.cache/huggingface/token $MOUNT/hf/token"
  rp_log "seeded HF token on fetch pod"
}

rp_remote_fetch_all() {
  local host="$1" port="$2"
  # Upload current remote.sh so fetch/wait-weights match this tree.
  rp_ssh "$host" "$port" "mkdir -p /opt/fastvideo-rs/scripts/gpu $WEIGHTS $MOUNT/runs $MOUNT/gpucheck-out/logs"
  fv_rsync_to "$host" "$port" "$FV_ROOT/scripts/gpu/" "/opt/fastvideo-rs/scripts/gpu/" --exclude '.env*'
  rp_ssh "$host" "$port" "command -v hf-fm || command -v hf-fetch-model" >/dev/null \
    || die "hf-fm missing on fetch pod — image pull may have failed"
  rp_log "cpu fetch start (hf-fm) mount=$WEIGHTS"
  local dest repo globs dest_path
  while IFS=$'\t' read -r dest repo globs; do
    [[ -n "$dest" ]] || continue
    dest_path="$WEIGHTS/$dest"
    rp_log "fetch $repo → $dest_path  globs: $globs"
    # remote.sh fetch backgrounds hf-fm; start them all, then wait.
    rp_ssh "$host" "$port" "export FV_WORK=$MOUNT HF_HOME=$MOUNT/hf HF_TOKEN='${HF_TOKEN:-}' PATH=/usr/local/bin:\$PATH
      cd /opt/fastvideo-rs
      bash scripts/gpu/remote.sh fetch '$repo' '$dest_path' $globs"
  done < <(rp_weight_rows)

  local timeout_s=21600
  while IFS=$'\t' read -r dest repo globs; do
    [[ -n "$dest" ]] || continue
    dest_path="$WEIGHTS/$dest"
    rp_log "wait $dest ($repo)"
    local t0 secs size
    t0=$(date +%s)
    if ! rp_ssh "$host" "$port" "export FV_WORK=$MOUNT HF_HOME=$MOUNT/hf PATH=/usr/local/bin:\$PATH
      cd /opt/fastvideo-rs
      bash scripts/gpu/remote.sh wait-weights '$dest_path' $timeout_s"; then
      rp_log "FATAL: wait-weights failed for $dest"
      rp_ssh "$host" "$port" "tail -40 $MOUNT/gpucheck-out/logs/fetch-$dest.log" | tee -a "$RP_LIVE" >&2 || true
      return 1
    fi
    secs=$(( $(date +%s) - t0 ))
    size="$(rp_ssh "$host" "$port" "du -sh '$dest_path' | cut -f1")"
    rp_log "ok $dest  ${size}  ${secs}s  $repo"
  done < <(rp_weight_rows)
}

rp_verify_remote() {
  local host="$1" port="$2"
  rp_log "verify volume shards"
  # Slim runtime has no Python — bash existence + du only.
  rp_ssh "$host" "$port" 'bash -s' <<'EOS' | tee -a "$RP_LIVE" >&2
set -euo pipefail
root=/workspace/weights
fail=0
check() {
  local name="$1"; shift
  local d="$root/$name" miss="" p
  for p in "$@"; do
    if [[ ! -e "$d/$p" ]]; then
      miss="$miss $p"
    fi
  done
  local sz
  sz="$(du -sh "$d" 2>/dev/null | cut -f1 || echo 0)"
  if [[ -n "$miss" ]]; then
    echo "FAIL $name $sz missing:$miss"
    fail=1
  else
    echo "PASS $name $sz"
  fi
}
check h3-8step tokenizer text_encoder transformer audio_vae vae .complete
check h3-base tokenizer text_encoder transformer audio_vae vae .complete
check FastH3-4-step-Preview-v1-LoRA dense-datafree/adapter_model.safetensors .complete
check FastH3-4-step-Preview-v1-VSA-DataFree .complete
check upscaler minimax_h3_latent_upscaler_3d_bf16.safetensors .complete
check h3-to-ltx config.json model.safetensors .complete
check ltx2 tokenizer text_encoder vae audio_vae vocoder ltx-2-19b-distilled.safetensors .complete
check ltx25 tokenizer text_encoder connectors transformer vae audio_vae vocoder latent_upsampler .complete
check ltx23 tokenizer text_encoder transformer vae audio_vae vocoder .complete
if [[ ! -d $root/ltx23/spatial_upscaler && ! -d $root/ltx23/latent_upsampler ]]; then
  echo "FAIL ltx23 missing spatial_upscaler|latent_upsampler"
  fail=1
else
  echo "PASS ltx23-upscaler"
fi
check hy15-480-t2v tokenizer text_encoder transformer vae .complete
check hy15-480-i2v tokenizer text_encoder transformer vae .complete
check hy15-720-t2v tokenizer text_encoder transformer vae .complete
check hy15-720-i2v tokenizer text_encoder transformer vae .complete
echo "TOTAL $(du -sh $root | cut -f1)"
exit $fail
EOS
}

cmd_fetch() {
  require_tools curl jq ssh rsync python3
  rp_load_key
  [[ -f "$RP_SSH_KEY" ]] || die "ssh key $RP_SSH_KEY missing"
  rp_log "▶ fetch: CPU-only weight stage (no GPU pods)"
  local vol dc auth pod host port
  vol="$(rp_find_or_create_volume)"
  dc="$(rp_volume_dc "$vol")"
  rp_log "volume $vol dc=$dc ${RP_VOL_GB}GB (or existing size)"
  auth="$(rp_ensure_registry_auth)"
  pod="$(rp_create_cpu_pod "$vol" "$dc" "$auth")"
  echo "$pod" >"$RP_STATE/fetch-pod.id"
  # Wall cap: destroy the CPU box after 7h even if this shell dies.
  nohup bash -c "sleep 25200; curl -sS -X DELETE -H 'Authorization: Bearer ${RUNPOD_API_KEY}' '${RUNPOD_API_BASE}/pods/${pod}' >/dev/null" >/dev/null 2>&1 &
  read -r host port < <(rp_wait_ssh "$pod" 900)
  echo "$host $port" >"$RP_STATE/fetch-ssh"
  rp_seed_token "$host" "$port"
  rp_ssh "$host" "$port" "df -h $MOUNT; nvidia-smi >/dev/null 2>&1 && echo GPU_PRESENT && nvidia-smi -L || echo CPU_ONLY"
  if rp_ssh "$host" "$port" "command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1"; then
    rp_log "FATAL: fetch pod has a GPU — refusing to download on billed GPU. Destroying $pod"
    rp_destroy_pod "$pod"
    die "CPU fetch pod was a GPU"
  fi
  rp_remote_fetch_all "$host" "$port"
  rp_verify_remote "$host" "$port"
  jq -n --arg vol "$vol" --arg dc "$dc" --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{volume:$vol, datacenter:$dc, ready_at:$at, status:"ready"}' >"$(rp_volume_ready_file)"
  rp_log "volume ready  id=$vol  dc=$dc"
  rp_log "destroying CPU fetch pod $pod"
  rp_destroy_pod "$pod"
  rp_log "fetch PASS — GPU pods may start"
}

cmd_verify() {
  rp_load_key
  local pod host port
  pod="$(cat "$RP_STATE/fetch-pod.id" 2>/dev/null || true)"
  [[ -n "$pod" ]] || die "no fetch pod id; run fetch"
  read -r host port < <(rp_wait_ssh "$pod" 60)
  rp_verify_remote "$host" "$port"
}

cmd_status() {
  rp_load_key
  rp_log "status"
  rp_rest GET /networkvolumes | jq -r '.[] | "  vol \(.id)  \(.name)  \(.size)GB  dc=\(.dataCenterId)"' | tee -a "$RP_LIVE" >&2
  rp_rest GET /pods | jq -r '.[] | select((.name//"")|startswith("fv-")) | "  pod \(.id)  \(.name)  \(.desiredStatus)  \(.computeType)  $\(.costPerHr)/hr  \(.imageName)"' | tee -a "$RP_LIVE" >&2
  if [[ -f "$(rp_volume_ready_file)" ]]; then
    rp_log "volume-ready $(cat "$(rp_volume_ready_file)")"
  else
    rp_log "volume NOT ready (no $(rp_volume_ready_file))"
  fi
}

cmd_reap() {
  rp_load_key
  local ids
  ids="$(rp_rest GET /pods | jq -r '.[] | select((.name//"")|test("^fv-(fetch|gpu)")) | .id')"
  [[ -n "$ids" ]] || { rp_log "no fv-fetch/fv-gpu pods"; return 0; }
  local id
  for id in $ids; do rp_destroy_pod "$id"; done
}

rp_gpu_offer() {
  local dc="$1"
  local query resp stock price
  query="query { gpuTypes(input: {id: \"${RP_GPU_TYPE}\"}) { lowestPrice(input: {gpuCount: 1, dataCenterId: \"${dc}\"}) { stockStatus uninterruptablePrice } } }"
  resp="$(rp_gql "$query")"
  stock="$(jq -r '.data.gpuTypes[0].lowestPrice.stockStatus // empty' <<<"$resp")"
  price="$(jq -r '.data.gpuTypes[0].lowestPrice.uninterruptablePrice // empty' <<<"$resp")"
  rp_log "offer $RP_GPU_TYPE dc=$dc stock=${stock:-none} \$${price:-?}/hr"
  [[ -n "$stock" && "$stock" != "null" ]] || return 1
  if awk -v p="${price:-99}" -v cap="$RP_GPU_MAX_DPH" 'BEGIN{exit !(p+0 > cap+0)}'; then
    rp_log "FATAL: offer \$${price}/hr above ceiling \$${RP_GPU_MAX_DPH}"
    return 1
  fi
  printf '%s %s\n' "$stock" "$price"
}

rp_create_gpu_pod() {
  local name="$1" vol="$2" dc="$3" auth="${4:-}"
  local pubkey payload resp id dph
  pubkey="$(rp_pubkey)"
  payload="$(jq -n \
    --arg name "$name" \
    --arg image "$RP_IMAGE" \
    --arg vol "$vol" \
    --arg dc "$dc" \
    --arg gpu "$RP_GPU_TYPE" \
    --arg pubkey "$pubkey" \
    --arg auth "$auth" \
    '{
      name: $name,
      imageName: $image,
      cloudType: "SECURE",
      computeType: "GPU",
      gpuTypeIds: [$gpu],
      gpuCount: 1,
      dataCenterIds: [$dc],
      containerDiskInGb: 20,
      volumeInGb: 0,
      networkVolumeId: $vol,
      volumeMountPath: "/workspace",
      ports: ["22/tcp"],
      dockerStartCmd: ["/bin/bash","-lc","mkdir -p /root/.ssh /run/sshd; printf \"%s\\n\" \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; exec sleep infinity"],
      env: { PUBLIC_KEY: $pubkey }
    } + (if $auth != "" then {containerRegistryAuthId:$auth} else {} end)')"
  rp_log "create GPU pod name=$name image=$RP_IMAGE dc=$dc vol=$vol type=$RP_GPU_TYPE"
  resp="$(rp_rest POST /pods "$payload")"
  id="$(jq -r '.id // empty' <<<"$resp")"
  [[ -n "$id" ]] || die "GPU pod create failed: $resp"
  dph="$(jq -r '.costPerHr // 0' <<<"$resp")"
  rp_log "gpu pod $id  name=$name  \$${dph}/hr  image=$RP_IMAGE  dc=$dc"
  if awk -v p="$dph" -v cap="$RP_GPU_MAX_DPH" 'BEGIN{exit !(p+0 > cap+0)}'; then
    rp_log "FATAL: pod $id \$${dph}/hr above ceiling \$${RP_GPU_MAX_DPH} — destroying"
    rp_destroy_pod "$id"
    die "GPU price $dph > $RP_GPU_MAX_DPH"
  fi
  printf '%s\n' "$resp" >"$RP_STATE/${name}.json"
  printf '%s\n' "$id"
}

rp_start_family() {
  local family="$1" pod="$2" host="$3" port="$4" cap_s="$5"
  local remote="/opt/fastvideo-rs/scripts/gpu/runpod-matrix.sh"
  rp_ssh "$host" "$port" "mkdir -p /opt/fastvideo-rs/scripts/gpu /workspace/runs/$family /workspace/gpucheck-out/logs"
  rp_rsync "$host" "$port" "$FV_ROOT/scripts/gpu/runpod-matrix.sh" "$remote"
  rp_ssh "$host" "$port" "chmod +x $remote; echo CPU_CHECK; nvidia-smi -L; df -h /workspace | tail -1; ls /workspace/weights | head"
  rp_log "stage start $family pod=$pod cap=${cap_s}s"
  rp_ssh "$host" "$port" "nohup bash $remote $family > /workspace/runs/$family/driver.log 2>&1 & echo \$! > /workspace/runs/$family/pid"
  # Hard wall: destroy even if this driver dies.
  nohup bash -c "sleep ${cap_s}; curl -sS -X DELETE -H 'Authorization: Bearer ${RUNPOD_API_KEY}' '${RUNPOD_API_BASE}/pods/${pod}' >/dev/null" >/dev/null 2>&1 &
  printf '%s %s\n' "$host" "$port" >"$RP_STATE/gpu-${family}-ssh"
  printf '%s\n' "$pod" >"$RP_STATE/gpu-${family}.id"
  rp_log "stage ok $family started pid on $host:$port (cap ${cap_s}s)"
}

rp_restart_dest() {
  local host="$1" port="$2" dest="$3"
  local repo globs dest_path
  dest_path="$WEIGHTS/$dest"
  repo="$(rp_weight_rows | awk -F'\t' -v d="$dest" '$1==d{print $2; exit}')"
  globs="$(rp_weight_rows | awk -F'\t' -v d="$dest" '$1==d{print $3; exit}')"
  [[ -n "$repo" ]] || { rp_log "FATAL: unknown dest $dest"; return 1; }
  rp_log "restart fetch $dest ($repo) only"
  rp_ssh "$host" "$port" "rm -f $dest_path/.fetch.pid $dest_path/.complete"
  rp_ssh "$host" "$port" "export FV_WORK=$MOUNT HF_HOME=$MOUNT/hf HF_TOKEN='${HF_TOKEN:-}' PATH=/usr/local/bin:\$PATH
    cd /opt/fastvideo-rs
    bash scripts/gpu/remote.sh fetch '$repo' '$dest_path' $globs"
}

cmd_gpu() {
  require_tools curl jq ssh rsync python3
  rp_load_key
  [[ -f "$RP_SSH_KEY" ]] || die "ssh key $RP_SSH_KEY missing"
  local vol dc auth
  vol="$(cat "$RP_STATE/volume.id" 2>/dev/null || jq -r '.id // empty' "$RP_STATE/volume.json" 2>/dev/null || true)"
  [[ -n "$vol" ]] || die "no volume id — run fetch first"
  dc="$(rp_volume_dc "$vol")"
  if [[ ! -f "$(rp_volume_ready_file)" ]]; then
    local fpod fhost fport
    fpod="$(cat "$RP_STATE/fetch-pod.id" 2>/dev/null || true)"
    [[ -n "$fpod" ]] || die "volume not ready and no fetch pod"
    read -r fhost fport < <(rp_wait_ssh "$fpod" 120)
    rp_verify_remote "$fhost" "$fport" || die "volume shards incomplete"
    jq -n --arg vol "$vol" --arg dc "$dc" --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      '{volume:$vol, datacenter:$dc, ready_at:$at, status:"ready"}' >"$(rp_volume_ready_file)"
    rp_log "volume ready  id=$vol  dc=$dc"
  fi
  local live
  live="$(rp_rest GET /pods | jq -r --arg p "$GPU_NAME_PREFIX" '[.[] | select((.name//"")|startswith($p)) | .id] | length')"
  if (( live >= RP_MAX_GPU_PODS )); then
    die "already $live GPU pods with prefix $GPU_NAME_PREFIX (cap $RP_MAX_GPU_PODS)"
  fi
  rp_gpu_offer "$dc" >/dev/null || die "no $RP_GPU_TYPE stock in $dc (quota/stock)"
  auth="$(jq -r '.containerRegistryAuthId // empty' "$RP_STATE/fetch-pod.json" 2>/dev/null || true)"
  [[ -n "$auth" ]] || auth="$(rp_ensure_registry_auth)"
  rp_log "▶ gpu: three PRO 6000 pods on vol=$vol dc=$dc image=$RP_IMAGE auth=${auth:-none}"

  local h3_name ltx_name hy_name
  h3_name="${GPU_NAME_PREFIX}-h3-$(date -u +%Y%m%d%H%M%S)"
  ltx_name="${GPU_NAME_PREFIX}-ltx-$(date -u +%Y%m%d%H%M%S)"
  hy_name="${GPU_NAME_PREFIX}-hy-$(date -u +%Y%m%d%H%M%S)"

  local h3_id ltx_id hy_id
  h3_id="$(rp_create_gpu_pod "$h3_name" "$vol" "$dc" "$auth")"
  ltx_id="$(rp_create_gpu_pod "$ltx_name" "$vol" "$dc" "$auth")"
  hy_id="$(rp_create_gpu_pod "$hy_name" "$vol" "$dc" "$auth")"

  local h3_hp ltx_hp hy_hp
  read -r h3_hp < <(echo "$(rp_wait_ssh "$h3_id" 900)")
  read -r ltx_hp < <(echo "$(rp_wait_ssh "$ltx_id" 900)")
  read -r hy_hp < <(echo "$(rp_wait_ssh "$hy_id" 900)")

  local h3_host h3_port ltx_host ltx_port hy_host hy_port
  read -r h3_host h3_port <<<"$h3_hp"
  read -r ltx_host ltx_port <<<"$ltx_hp"
  read -r hy_host hy_port <<<"$hy_hp"

  rp_start_family h3 "$h3_id" "$h3_host" "$h3_port" 5400
  rp_start_family ltx "$ltx_id" "$ltx_host" "$ltx_port" 7200
  rp_start_family hunyuan "$hy_id" "$hy_host" "$hy_port" 5400

  rp_log "gpu jobs started  h3=$h3_id@$h3_host:$h3_port  ltx=$ltx_id@$ltx_host:$ltx_port  hy=$hy_id@$hy_host:$hy_port"
  rp_log "CPU fetch pod left up for ltx23 extra/logs — destroy after Hunyuan skip or extra fetch ends"
}

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

case "${1:-}" in
  fetch) shift; cmd_fetch "$@" ;;
  verify) cmd_verify ;;
  gpu) cmd_gpu ;;
  status) cmd_status ;;
  reap) cmd_reap ;;
  *) usage ;;
esac
