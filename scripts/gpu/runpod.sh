#!/usr/bin/env bash
# Runpod driver for the H3 / LTX / Hunyuan sweep.
# Weights first on a 1 TB network volume (cheap CPU pod + hf-fm). GPU pods
# only after every tree has a .complete marker. Hard cap: 3 GPU pods.
#
#   runpod.sh fetch     volume + CPU download + verify
#   runpod.sh verify    re-check shards on a live fetch pod or via SSH
#   runpod.sh manifest  print dests from weights-manifest.tsv
#   runpod.sh gpu       three PRO 6000 96 GB pods (refuses if volume incomplete)
#   runpod.sh smoke     cheap Blackwell nvrtc+kernels, then destroy
#   runpod.sh matrix    one PRO 6000; restart between H3/LTX/Hunyuan/Wan
#   runpod.sh rtx6000   one PRO 6000 on the CI image for HEAD: weight gate,
#                       then H3 / FastH3 / LTX-2.5 parity cells
#   runpod.sh b200      US volume + H3/FastH3/LTX fetch + 1× B200 warm E2E
#   runpod.sh offers    probe B200 / PRO 6000 stock (US DCs)
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
if [[ -z "${RUNPOD_IMAGE:-}" && -f "$RP_STATE/runtime-image.txt" ]]; then
  RUNPOD_IMAGE="$(tr -d '[:space:]' <"$RP_STATE/runtime-image.txt")"
fi
RP_IMAGE="${RUNPOD_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-runtime:latest}"
RP_FETCH_IMAGE="${RUNPOD_FETCH_IMAGE:-$RP_IMAGE}"
RP_GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA RTX PRO 6000 Blackwell Server Edition}"
RP_VOL_NAME="${RUNPOD_VOLUME_NAME:-fv-weights-h3-ltx-hy}"
RP_VOL_GB="${RUNPOD_VOLUME_GB:-1000}"
RP_MAX_GPU_PODS=3
RP_GPU_MAX_DPH="${RUNPOD_GPU_MAX_DPH:-20}"
RP_B200_TYPE="${RUNPOD_B200_TYPE:-NVIDIA B200}"
RP_B200_VOL_NAME="${RUNPOD_B200_VOLUME_NAME:-fv-weights-b200-us}"
RP_EUR_VOLUME_KEEP="jg48s6o1w0"
RP_B200_DESTS="${RUNPOD_FETCH_DESTS:-h3-8step h3-base FastH3-4-step-Preview-v1-LoRA upscaler h3-to-ltx ltx2 ltx25 ltx23}"
RP_US_DCS="${RUNPOD_US_DCS:-US-CA-2 US-CA-1 US-GA-1 US-GA-2 US-TX-3 US-IL-1 US-KS-2 US-WA-1 US-NC-1 US-OR-1 US-DE-1 US-NE-1 US-MD-1 US-MO-2 US-NC-2 US-TX-1 US-TX-4}"
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
  ssh -n -i "$RP_SSH_KEY" -p "$port" -o IdentitiesOnly=yes "${FV_SSH_OPTS[@]}" "root@$host" "$@"
}

rp_rsync() {
  local host="$1" port="$2" src="$3" dst="$4"; shift 4
  rsync -az "$@" -e "ssh -i $RP_SSH_KEY -p $port -o IdentitiesOnly=yes ${FV_SSH_OPTS[*]}" "$src" "root@$host:$dst"
}

rp_pod_ssh_target() {
  local id="$1" raw host port
  raw="$(rp_rest GET "/pods/${id}" 2>/dev/null || true)"
  host="$(jq -r '.publicIp // empty' <<<"${raw:-}" 2>/dev/null || true)"
  port="$(jq -r '
    (.portMappings["22"] // .portMappings."22" // empty) as $pm
    | if $pm != "" then $pm
      else [.runtime.ports[]? | select(.privatePort==22)][0].publicPort // empty
      end
  ' <<<"${raw:-}" 2>/dev/null || true)"
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
  if [[ "$id" == "$RP_EUR_VOLUME_KEEP" ]]; then
    die "refusing to reuse EUR volume $RP_EUR_VOLUME_KEEP"
  fi
  rp_log "volume id=$id name=$RP_VOL_NAME ${RP_VOL_GB}GB dc=$dc"
  printf '%s\n' "$resp" >"$RP_STATE/volume.json"
  printf '%s\n' "$id"
}

rp_probe_gpu_dc() {
  local gpu="$1" dc="$2"
  local query resp stock price
  query="query { gpuTypes(input: {id: \"${gpu}\"}) { id displayName lowestPrice(input: {gpuCount: 1, dataCenterId: \"${dc}\"}) { stockStatus uninterruptablePrice } } }"
  resp="$(rp_gql "$query")"
  stock="$(jq -r '.data.gpuTypes[0].lowestPrice.stockStatus // empty' <<<"$resp")"
  price="$(jq -r '.data.gpuTypes[0].lowestPrice.uninterruptablePrice // empty' <<<"$resp")"
  rp_log "offer $gpu dc=$dc stock=${stock:-none} \$${price:-?}/hr"
  printf '%s\t%s\t%s\t%s\n' "$gpu" "$dc" "${stock:-none}" "${price:-}"
}

rp_pick_us_b200_dc() {
  if [[ -n "${RUNPOD_DATACENTER:-}" ]]; then
    echo "$RUNPOD_DATACENTER"
    return 0
  fi
  local dc stock price best_dc="" best_price=""
  local line
  for dc in $RP_US_DCS; do
    line="$(rp_probe_gpu_dc "$RP_B200_TYPE" "$dc")"
    stock="$(cut -f3 <<<"$line")"
    price="$(cut -f4 <<<"$line")"
    if [[ -z "$stock" || "$stock" == "none" || "$stock" == "null" ]]; then
      continue
    fi
    if [[ -z "$price" || "$price" == "null" ]]; then
      continue
    fi
    if awk -v p="$price" -v cap="$RP_GPU_MAX_DPH" 'BEGIN{exit !(p+0 > cap+0)}'; then
      rp_log "skip $dc \$${price}/hr above ceiling \$${RP_GPU_MAX_DPH}"
      continue
    fi
    if [[ -z "$best_dc" ]] || awk -v p="$price" -v b="$best_price" 'BEGIN{exit !(p+0 < b+0)}'; then
      best_dc="$dc"
      best_price="$price"
    fi
  done
  [[ -n "$best_dc" ]] || return 1
  rp_log "pick B200 dc=$best_dc \$${best_price}/hr (cap \$${RP_GPU_MAX_DPH})"
  echo "$best_dc"
}

rp_create_us_volume() {
  local name="$1" dc="$2"
  local vols existing id
  vols="$(rp_rest GET /networkvolumes)"
  existing="$(jq -r --arg n "$name" --arg dc "$dc" --arg keep "$RP_EUR_VOLUME_KEEP" \
    '.[] | select(.name==$n and .dataCenterId==$dc and .id!=$keep) | "\(.id)\t\(.dataCenterId)\t\(.size)"' <<<"$vols" | head -1)"
  if [[ -n "$existing" ]]; then
    id="${existing%%$'\t'*}"
    [[ "$id" != "$RP_EUR_VOLUME_KEEP" ]] || die "refusing to reuse EUR volume $RP_EUR_VOLUME_KEEP"
    rp_log "reuse US volume $existing"
    printf '%s\n' "$id"
    jq -n --arg id "$id" --arg raw "$existing" '{id:$id, reuse:true, us:true}' >"$RP_STATE/volume-b200.json"
    return 0
  fi
  rp_log "create US volume name=$name size=${RP_VOL_GB}GB dc=$dc"
  local payload resp
  payload="$(jq -n --arg name "$name" --argjson size "$RP_VOL_GB" --arg dc "$dc" \
    '{name:$name, size:$size, dataCenterId:$dc}')"
  resp="$(rp_rest POST /networkvolumes "$payload")"
  id="$(jq -r '.id // empty' <<<"$resp")"
  [[ -n "$id" ]] || die "volume create returned no id: $resp"
  [[ "$id" != "$RP_EUR_VOLUME_KEEP" ]] || die "refusing to reuse EUR volume $RP_EUR_VOLUME_KEEP"
  rp_log "volume id=$id name=$name ${RP_VOL_GB}GB dc=$dc"
  printf '%s\n' "$resp" >"$RP_STATE/volume-b200.json"
  printf '%s\n' "$id"
}

rp_log_box_image() {
  local host="$1" port="$2" role="$3"
  local box
  box="$(rp_ssh "$host" "$port" "cat /opt/fastvideo-rs/target/release/fv-gpucheck.build-id 2>/dev/null || echo missing")"
  rp_log "$role pin=$RP_IMAGE box_build_id=$box"
  if [[ "$RP_IMAGE" == *":build-"* ]]; then
    local want="${RP_IMAGE##*:build-}"
    if [[ "$box" != "$want" ]]; then
      rp_log "FATAL: $role build-id $box != pinned $want ($RP_IMAGE)"
      return 1
    fi
  fi
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
  local f
  f="$(dirname "${BASH_SOURCE[0]}")/weights-manifest.tsv"
  [[ -f "$f" ]] || die "weights-manifest.tsv missing at $f"
  local rows
  rows="$(grep -vE '^[[:space:]]*(#|$)' "$f")"
  if [[ -n "${RUNPOD_FETCH_DESTS:-}" ]]; then
    echo "$rows" | awk -F'\t' -v allow="$RUNPOD_FETCH_DESTS" '
      BEGIN {
        n = split(allow, a, /[ ,]+/)
        for (i = 1; i <= n; i++) if (a[i] != "") ok[a[i]] = 1
      }
      $1 in ok
    '
  else
    printf '%s\n' "$rows"
  fi
}

cmd_manifest() {
  rp_weight_rows | awk -F'\t' '{print $1}'
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
      dockerStartCmd: ["/bin/bash","-lc","mkdir -p /root/.ssh /run/sshd; printf \"%s\\n\" \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; exec sleep infinity"],
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
  # rp_ssh uses ssh -n (stdin is /dev/null). Pipe the token via a temp file.
  local tmp
  tmp="$(mktemp)"
  printf '%s\n' "$HF_TOKEN" >"$tmp"
  chmod 600 "$tmp"
  rp_rsync "$host" "$port" "$tmp" "/root/.cache/huggingface/token"
  rm -f "$tmp"
  rp_ssh "$host" "$port" "cp -f /root/.cache/huggingface/token $MOUNT/hf/token && chmod 600 /root/.cache/huggingface/token $MOUNT/hf/token"
  rp_log "seeded HF token on fetch pod"
}

rp_remote_fetch_all() {
  local host="$1" port="$2"
  # Upload current remote.sh so fetch/wait-weights match this tree.
  rp_ssh "$host" "$port" "mkdir -p /opt/fastvideo-rs/scripts/gpu $WEIGHTS $MOUNT/runs $MOUNT/gpucheck-out/logs"
  rp_rsync "$host" "$port" "$FV_ROOT/scripts/gpu/" "/opt/fastvideo-rs/scripts/gpu/" --exclude '.env*'
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
  rp_remote_fetch_taeh3 "$host" "$port"
}

rp_remote_fetch_taeh3() {
  local host="$1" port="$2"
  local dest="$WEIGHTS/taeh3"
  rp_log "fetch taeh3 → $dest"
  rp_ssh "$host" "$port" "export FV_WORK=$MOUNT PATH=/usr/local/bin:\$PATH
    cd /opt/fastvideo-rs
    bash scripts/gpu/remote.sh fetch-taeh3 '$dest'"
  local t0 secs
  t0=$(date +%s)
  if ! rp_ssh "$host" "$port" "export FV_WORK=$MOUNT PATH=/usr/local/bin:\$PATH
    cd /opt/fastvideo-rs
    bash scripts/gpu/remote.sh wait-taeh3 '$dest' 600"; then
    rp_log "FATAL: wait-taeh3 failed"
    rp_ssh "$host" "$port" "tail -40 $MOUNT/gpucheck-out/logs/fetch-taeh3.log" | tee -a "$RP_LIVE" >&2 || true
    return 1
  fi
  secs=$(( $(date +%s) - t0 ))
  rp_log "ok taeh3  $(rp_ssh "$host" "$port" "du -sh '$dest' | cut -f1")  ${secs}s"
}

rp_verify_remote() {
  local host="$1" port="$2"
  rp_log "verify volume shards"
  # ssh -n: no stdin. Dest-level .complete from the committed manifest.
  local dests
  dests="$(rp_weight_rows | awk -F'\t' '{print $1}' | tr '\n' ' ')"
  rp_ssh "$host" "$port" "DESTS='$dests' bash -lc '
set -euo pipefail
root=/workspace/weights
missing=\"\"
for name in \$DESTS; do
  [[ -n \"\$name\" ]] || continue
  d=\"\$root/\$name\"
  sz=\"\$(du -sh \"\$d\" 2>/dev/null | cut -f1 || echo 0)\"
  if [[ ! -f \"\$d/.complete\" ]]; then
    echo \"MISSING \$name \$sz\"
    missing=\"\$missing \$name\"
  else
    echo \"PASS \$name \$sz\"
  fi
done
echo \"TOTAL \$(du -sh \$root 2>/dev/null | cut -f1 || echo 0)\"
if [[ -n \"\$missing\" ]]; then
  echo \"missing dests:\$missing\"
  exit 2
fi
'" | tee -a "$RP_LIVE" >&2
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

cmd_fetch_continue() {
  require_tools curl jq ssh rsync python3
  rp_load_key
  local pod host port
  pod="$(cat "$RP_STATE/fetch-pod.id" 2>/dev/null || true)"
  [[ -n "$pod" ]] || die "no fetch pod id; run fetch"
  if [[ -f "$RP_STATE/fetch-ssh" ]]; then
    read -r host port <"$RP_STATE/fetch-ssh"
  else
    read -r host port < <(rp_wait_ssh "$pod" 900)
    echo "$host $port" >"$RP_STATE/fetch-ssh"
  fi
  rp_log "▶ fetch-continue: pod $pod root@$host:$port"
  rp_seed_token "$host" "$port"
  rp_remote_fetch_all "$host" "$port"
  rp_verify_remote "$host" "$port"
  local vol dc
  vol="$(jq -r '.id // empty' "$RP_STATE/volume.json" 2>/dev/null || true)"
  [[ -n "$vol" ]] || vol="$(rp_find_or_create_volume)"
  dc="$(rp_volume_dc "$vol")"
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
  local name="$1" vol="$2" dc="$3" auth="${4:-}" gpu="${5:-$RP_GPU_TYPE}" pin_dc="${6:-0}"
  local pubkey payload resp id dph
  pubkey="$(rp_pubkey)"
  payload="$(jq -n \
    --arg name "$name" \
    --arg image "$RP_IMAGE" \
    --arg vol "$vol" \
    --arg dc "$dc" \
    --arg gpu "$gpu" \
    --arg pubkey "$pubkey" \
    --arg auth "$auth" \
    --argjson pin "$pin_dc" \
    '{
      name: $name,
      imageName: $image,
      cloudType: "SECURE",
      computeType: "GPU",
      gpuTypeIds: [$gpu],
      gpuCount: 1,
      containerDiskInGb: 20,
      volumeInGb: 0,
      networkVolumeId: $vol,
      volumeMountPath: "/workspace",
      ports: ["22/tcp"],
      dockerStartCmd: ["/bin/bash","-lc","mkdir -p /root/.ssh /run/sshd; printf \"%s\\n\" \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd; exec sleep infinity"],
      env: { PUBLIC_KEY: $pubkey }
    } + (if $pin == 1 then {dataCenterIds:[$dc]} else {} end)
      + (if $auth != "" then {containerRegistryAuthId:$auth} else {} end)')"
  rp_log "create GPU pod name=$name image=$RP_IMAGE dc=$dc pin_dc=$pin_dc vol=$vol type=$gpu"
  if ! resp="$(rp_rest POST /pods "$payload")"; then
    rp_log "GPU create failed for $gpu"
    return 1
  fi
  id="$(jq -r '.id // empty' <<<"$resp")"
  [[ -n "$id" ]] || { rp_log "GPU create returned no id for $gpu: $resp"; return 1; }
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
  h3_id="$(rp_create_gpu_pod "$h3_name" "$vol" "$dc" "$auth")" || die "h3 GPU create failed"
  ltx_id="$(rp_create_gpu_pod "$ltx_name" "$vol" "$dc" "$auth")" || die "ltx GPU create failed"
  hy_id="$(rp_create_gpu_pod "$hy_name" "$vol" "$dc" "$auth")" || die "hy GPU create failed"

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

cmd_smoke() {
  require_tools curl jq ssh rsync python3
  rp_load_key
  [[ -f "$RP_SSH_KEY" ]] || die "ssh key $RP_SSH_KEY missing"
  local vol dc auth name id host port gpu
  vol="$(jq -r '.id // empty' "$RP_STATE/volume.json" 2>/dev/null || cat "$RP_STATE/volume.id" 2>/dev/null || true)"
  [[ -n "$vol" ]] || die "no volume id — run fetch first"
  [[ -f "$(rp_volume_ready_file)" ]] || die "volume not verified"
  dc="$(rp_volume_dc "$vol")"
  auth="$(rp_ensure_registry_auth)"
  gpu="${RUNPOD_SMOKE_GPU:-}"
  name="fv-gpu-smoke-$(date -u +%Y%m%d%H%M%S)"
  rp_log "▶ smoke: vol=$vol image=$RP_IMAGE (no DC pin; volume colocates)"
  local g
  id=""
  if [[ -n "$gpu" ]]; then
    id="$(rp_create_gpu_pod "$name" "$vol" "$dc" "$auth" "$gpu" 0)" || true
  else
    for g in \
      "NVIDIA RTX PRO 4000 Blackwell" \
      "NVIDIA RTX PRO 4500 Blackwell" \
      "NVIDIA RTX PRO 4500 Blackwell Server Edition" \
      "NVIDIA GeForce RTX 5090" \
      "NVIDIA RTX PRO 6000 Blackwell Server Edition"; do
      rp_log "smoke try $g"
      if id="$(rp_create_gpu_pod "$name" "$vol" "$dc" "$auth" "$g" 0)"; then
        gpu="$g"
        break
      fi
    done
  fi
  [[ -n "$id" ]] || die "no cheap Blackwell stock for smoke"
  echo "$id" >"$RP_STATE/gpu-smoke.id"
  read -r host port < <(rp_wait_ssh "$id" 900)
  echo "$host $port" >"$RP_STATE/gpu-smoke-ssh"
  rp_log "smoke ssh root@$host:$port"
  rp_ssh "$host" "$port" "nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
  rp_ssh "$host" "$port" "mkdir -p /workspace/gpucheck-out /workspace/fv-libs"
  rp_rsync "$host" "$port" "$FV_ROOT/scripts/gpu/" "/opt/fastvideo-rs/scripts/gpu/" --exclude '.env*'
  local rc=0
  if ! rp_ssh "$host" "$port" "set -euo pipefail
    . /opt/fastvideo-rs/scripts/gpu/cuda-13.pins
    export PATH=/opt/fastvideo-rs/target/release:/usr/local/bin:/usr/local/cuda/bin:\$PATH
    export LD_LIBRARY_PATH=/workspace/fv-libs:/lib/x86_64-linux-gnu:/usr/local/cuda-13.0/lib64:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:\${LD_LIBRARY_PATH:-}
    bash /opt/fastvideo-rs/scripts/gpu/remote.sh env >/tmp/fv-env.json || true
    BIN=/opt/fastvideo-rs/target/release/fv-gpucheck
    echo \"▶ nvrtc\"
    timeout --signal=TERM --kill-after=15 180 \$BIN --out /workspace/gpucheck-out nvrtc
    echo \"▶ kernels\"
    timeout --signal=TERM --kill-after=15 300 \$BIN --mode fast --out /workspace/gpucheck-out kernels
  "; then
    rc=1
    rp_log "FATAL: smoke nvrtc/kernels failed"
    rp_ssh "$host" "$port" "tail -40 /workspace/gpucheck-out/logs/*.log 2>/dev/null || true" || true
  fi
  rp_log "destroying smoke pod $id"
  rp_destroy_pod "$id"
  [[ "$rc" -eq 0 ]] || die "smoke failed"
  rp_log "smoke PASS"
}

rp_recreate_matrix_pod() {
  local old="$1" vol dc auth name id host port
  vol="$(jq -r '.id // empty' "$RP_STATE/volume.json")"
  dc="$(rp_volume_dc "$vol")"
  auth="$(rp_ensure_registry_auth)"
  name="fv-gpu-matrix-$(date -u +%Y%m%d%H%M%S)"
  rp_log "recreate GPU between families (destroy $old)"
  rp_destroy_pod "$old" || true
  id="$(rp_create_gpu_pod "$name" "$vol" "$dc" "$auth" "$RP_GPU_TYPE" 0)" || die "recreate failed"
  echo "$id" >"$RP_STATE/gpu-matrix.id"
  nohup bash -c "sleep 10800; curl -sS -X DELETE -H 'Authorization: Bearer ${RUNPOD_API_KEY}' '${RUNPOD_API_BASE}/pods/${id}' >/dev/null" >/dev/null 2>&1 &
  read -r host port < <(rp_wait_ssh "$id" 900)
  printf '%s %s %s\n' "$id" "$host" "$port"
}

cmd_matrix() {
  require_tools curl jq ssh rsync python3
  rp_load_key
  [[ -f "$RP_SSH_KEY" ]] || die "ssh key $RP_SSH_KEY missing"
  local vol dc auth name id host port family
  vol="$(jq -r '.id // empty' "$RP_STATE/volume.json" 2>/dev/null || cat "$RP_STATE/volume.id" 2>/dev/null || true)"
  [[ -n "$vol" ]] || die "no volume id"
  [[ -f "$(rp_volume_ready_file)" ]] || die "volume not verified"
  dc="$(rp_volume_dc "$vol")"
  auth="$(rp_ensure_registry_auth)"
  name="fv-gpu-matrix-$(date -u +%Y%m%d%H%M%S)"
  rp_log "▶ matrix: one PRO 6000, restart between H3/LTX/Hunyuan/Wan  image=$RP_IMAGE"
  id="$(rp_create_gpu_pod "$name" "$vol" "$dc" "$auth" "$RP_GPU_TYPE" 0)" || die "matrix GPU create failed"
  echo "$id" >"$RP_STATE/gpu-matrix.id"
  nohup bash -c "sleep 10800; curl -sS -X DELETE -H 'Authorization: Bearer ${RUNPOD_API_KEY}' '${RUNPOD_API_BASE}/pods/${id}' >/dev/null" >/dev/null 2>&1 &
  read -r host port < <(rp_wait_ssh "$id" 900)
  echo "$host $port" >"$RP_STATE/gpu-matrix-ssh"
  mkdir -p "$RP_STATE/phase3-gate"
  local first=1
  for family in h3 ltx hunyuan wan; do
    if [[ "$first" -eq 0 ]]; then
      read -r id host port < <(rp_recreate_matrix_pod "$id")
      echo "$host $port" >"$RP_STATE/gpu-matrix-ssh"
    fi
    first=0
    rp_log "▶ family $family on $id root@$host:$port"
    rp_ssh "$host" "$port" "nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader; mkdir -p /workspace/runs/$family /workspace/gpucheck-out/logs"
    rp_rsync "$host" "$port" "$FV_ROOT/scripts/gpu/" "/opt/fastvideo-rs/scripts/gpu/" --exclude '.env*'
    if [[ "$family" == h3 ]]; then
      rp_ssh "$host" "$port" "python3 -c 'import json; print(json.load(open(\"/workspace/gpucheck-out/nvrtc.json\")).get(\"context\",{}).get(\"aot_sms\"))' 2>/dev/null || true" || true
    fi
    # 4 cells × 300s + load slack
    if ! rp_ssh "$host" "$port" "export FV_WORK=/workspace FV_GEN_TIMEOUT_S=300 PATH=/opt/fastvideo-rs/target/release:/usr/local/bin:/usr/local/cuda/bin:\$PATH
      bash /opt/fastvideo-rs/scripts/gpu/runpod-matrix.sh $family"; then
      rp_log "family $family returned non-zero (cells continue-on-fail inside the script)"
    fi
    rp_ssh "$host" "$port" "tail -30 /workspace/runs/$family/live.log" | tee -a "$RP_LIVE" >&2 || true
    mkdir -p "$RP_STATE/phase3-gate/runs/$family"
    rsync -az -e "ssh -i $RP_SSH_KEY -p $port -o IdentitiesOnly=yes ${FV_SSH_OPTS[*]}" \
      "root@$host:/workspace/runs/$family/" "$RP_STATE/phase3-gate/runs/$family/" || true
    rp_log "ok family $family logs → $RP_STATE/phase3-gate/runs/$family"
  done
  rp_log "destroying matrix pod $id"
  rp_destroy_pod "$id"
  rp_log "matrix PASS (see phase3-gate/runs)"
}

rp_rsync_and_build() {
  local host="$1" port="$2"
  local remote="/opt/fastvideo-rs"
  rp_log "rsync source → $host:$port:$remote"
  rp_ssh "$host" "$port" "mkdir -p $remote /workspace/src /workspace/cargo-target"
  (cd "$FV_ROOT" && git ls-files -z -- . ':!third_party' | rsync -az --files-from=- --from0 \
    -e "ssh -i $RP_SSH_KEY -p $port -o IdentitiesOnly=yes ${FV_SSH_OPTS[*]}" \
    . "root@$host:$remote/")
  if [[ -d "$FV_ROOT/third_party" ]]; then
    rsync -az --exclude '.git' -e "ssh -i $RP_SSH_KEY -p $port -o IdentitiesOnly=yes ${FV_SSH_OPTS[*]}" \
      "$FV_ROOT/third_party/" "root@$host:$remote/third_party/" || true
  fi
  rp_log "install rustc + nvcc if missing, then cargo build --release -p fastvideo-gpucheck"
  rp_ssh "$host" "$port" "set -euo pipefail
    . /opt/fastvideo-rs/scripts/gpu/cuda-13.pins
    export DEBIAN_FRONTEND=noninteractive
    if ! command -v rustc >/dev/null; then
      echo '▶ rustup'
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
    fi
    export PATH=\"\$HOME/.cargo/bin:/usr/local/cargo/bin:/usr/local/cuda-13.4/bin:/usr/local/cuda/bin:\$PATH\"
    if ! command -v nvcc >/dev/null; then
      echo '▶ apt cuda-nvcc + build-essential'
      apt-get update -qq
      apt-get install -y -qq --no-install-recommends build-essential pkg-config libssl-dev clang \
        \"\${CUDA_NVCC_PKG:-cuda-nvcc-13-4}\" \"\${CUDA_NVRTC_DEV_PKG:-cuda-nvrtc-dev-13-4}\" || true
    fi
    export NVCC=\"\$(command -v nvcc || echo /usr/local/cuda-13.4/bin/nvcc)\"
    export CUDARC_CUDA_VERSION=13000
    export CARGO_TARGET_DIR=/workspace/cargo-target
    export CARGO_PROFILE_RELEASE_LTO=off
    export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
    rustc --version
    nvcc --version | tail -1 || echo 'nvcc missing — build will fail if cubins are required'
    cd $remote
    echo '▶ cargo build --release -p fastvideo-gpucheck --features cuda'
    cargo build --release -p fastvideo-gpucheck --features cuda
    install -D /workspace/cargo-target/release/fv-gpucheck /opt/fastvideo-rs/target/release/fv-gpucheck
    echo remote-rsync-\$(date -u +%Y%m%dT%H%M%SZ) > /opt/fastvideo-rs/target/release/fv-gpucheck.build-id
    /opt/fastvideo-rs/target/release/fv-gpucheck --help >/dev/null
    echo BUILD_OK
  "
}

cmd_offers() {
  require_tools curl jq
  rp_load_key
  rp_log "▶ offers  cap=\$${RP_GPU_MAX_DPH}/hr"
  local dc
  for dc in $RP_US_DCS; do
    rp_probe_gpu_dc "$RP_B200_TYPE" "$dc" >/dev/null || true
  done
  local g
  for g in "NVIDIA B200 SXM" "NVIDIA B200 NVL"; do
    rp_probe_gpu_dc "$g" "US-CA-2" >/dev/null || true
  done
}

cmd_b200() {
  require_tools curl jq ssh rsync python3 git
  rp_load_key
  [[ -f "$RP_SSH_KEY" ]] || die "ssh key $RP_SSH_KEY missing"
  export RUNPOD_GPU_MAX_DPH="${RUNPOD_GPU_MAX_DPH:-$RP_GPU_MAX_DPH}"
  export RUNPOD_FETCH_DESTS="${RUNPOD_FETCH_DESTS:-$RP_B200_DESTS}"
  export RUNPOD_VOLUME_NAME="${RUNPOD_VOLUME_NAME:-$RP_B200_VOL_NAME}"
  RP_VOL_NAME="$RUNPOD_VOLUME_NAME"
  RP_GPU_TYPE="$RP_B200_TYPE"
  rp_log "▶ b200: vol_name=$RP_VOL_NAME dests=$RUNPOD_FETCH_DESTS cap=\$${RP_GPU_MAX_DPH} image=$RP_IMAGE"
  rp_log "keep EUR volume $RP_EUR_VOLUME_KEEP (never detach/delete)"

  local dc vol auth pod host port have_b200=0
  if dc="$(rp_pick_us_b200_dc)"; then
    have_b200=1
  else
    rp_log "no US B200 under \$${RP_GPU_MAX_DPH} — still creating US volume + fetch"
    cmd_offers
    dc="${RUNPOD_DATACENTER:-US-CA-2}"
  fi
  export RUNPOD_DATACENTER="$dc"
  vol="$(rp_create_us_volume "$RP_VOL_NAME" "$dc")"
  echo "$vol" >"$RP_STATE/volume-b200.id"
  rp_log "US volume $vol dc=$dc name=$RP_VOL_NAME"

  auth="$(rp_ensure_registry_auth)"
  pod="$(rp_create_cpu_pod "$vol" "$dc" "$auth")"
  echo "$pod" >"$RP_STATE/fetch-pod.id"
  nohup bash -c "sleep 25200; curl -sS -X DELETE -H 'Authorization: Bearer ${RUNPOD_API_KEY}' '${RUNPOD_API_BASE}/pods/${pod}' >/dev/null" >/dev/null 2>&1 &
  read -r host port < <(rp_wait_ssh "$pod" 900)
  echo "$host $port" >"$RP_STATE/fetch-ssh"
  rp_log_box_image "$host" "$port" "fetch-cpu" || die "fetch pod image is not the pinned runtime"
  rp_seed_token "$host" "$port"
  rp_ssh "$host" "$port" "df -h $MOUNT; echo CPU_ONLY"
  if rp_ssh "$host" "$port" "command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1"; then
    rp_log "FATAL: fetch pod has a GPU — destroying $pod"
    rp_destroy_pod "$pod"
    die "CPU fetch pod was a GPU"
  fi
  rp_remote_fetch_all "$host" "$port"
  rp_verify_remote "$host" "$port"
  if ! rp_ssh "$host" "$port" "test -f $WEIGHTS/taeh3/.complete -o -f $WEIGHTS/taeh3/taeh3.safetensors"; then
    rp_log "FATAL: taeh3 missing after fetch"
    return 1
  fi
  jq -n --arg vol "$vol" --arg dc "$dc" --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{volume:$vol, datacenter:$dc, ready_at:$at, status:"ready", us:true}' >"$RP_STATE/volume-b200-ready.json"
  rp_log "volume ready  id=$vol  dc=$dc"
  rp_log "destroying CPU fetch pod $pod"
  rp_destroy_pod "$pod"

  if [[ "$have_b200" -eq 0 ]]; then
    rp_log "stop: US volume $vol dc=$dc fetched; no B200 under \$${RP_GPU_MAX_DPH}"
    return 0
  fi

  local name id gpu_host gpu_port
  name="fv-gpu-b200-$(date -u +%Y%m%d%H%M%S)"
  if ! rp_gpu_offer "$dc" >/dev/null; then
    rp_log "stop: stock gone after fetch. US volume $vol dc=$dc kept."
    return 0
  fi
  id="$(rp_create_gpu_pod "$name" "$vol" "$dc" "$auth" "$RP_B200_TYPE" 1)" || die "B200 GPU create failed"
  echo "$id" >"$RP_STATE/gpu-b200.id"
  nohup bash -c "sleep 21600; curl -sS -X DELETE -H 'Authorization: Bearer ${RUNPOD_API_KEY}' '${RUNPOD_API_BASE}/pods/${id}' >/dev/null" >/dev/null 2>&1 &
  read -r gpu_host gpu_port < <(rp_wait_ssh "$id" 900)
  echo "$gpu_host $gpu_port" >"$RP_STATE/gpu-b200-ssh"
  rp_log "b200 ssh root@$gpu_host:$gpu_port pod=$id"
  rp_log_box_image "$gpu_host" "$gpu_port" "b200-gpu" || die "B200 pod image is not the pinned runtime"
  rp_ssh "$gpu_host" "$gpu_port" "nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
  rp_log "stage start b200-warm pod=$id"
  rp_ssh "$gpu_host" "$gpu_port" "mkdir -p /workspace/runs/b200 /workspace/gpucheck-out/logs"
  rp_rsync "$gpu_host" "$gpu_port" "$FV_ROOT/scripts/gpu/" "/opt/fastvideo-rs/scripts/gpu/" --exclude '.env*'
  if ! rp_ssh "$gpu_host" "$gpu_port" "export FV_WORK=/workspace FV_GEN_TIMEOUT_S=3600 PATH=/opt/fastvideo-rs/target/release:/usr/local/bin:/usr/local/cuda/bin:\$PATH
    bash /opt/fastvideo-rs/scripts/gpu/runpod-matrix.sh b200"; then
    rp_log "b200 matrix returned non-zero (cells continue-on-fail inside the script)"
  fi
  rp_ssh "$gpu_host" "$gpu_port" "tail -80 /workspace/runs/b200/live.log" | tee -a "$RP_LIVE" >&2 || true
  mkdir -p "$RP_STATE/b200/runs"
  rsync -az -e "ssh -i $RP_SSH_KEY -p $gpu_port -o IdentitiesOnly=yes ${FV_SSH_OPTS[*]}" \
    "root@$gpu_host:/workspace/runs/b200/" "$RP_STATE/b200/runs/" || true
  rp_log "destroying B200 pod $id"
  rp_destroy_pod "$id"
  rp_log "b200 PASS (see $RP_STATE/b200/runs) — volumes kept ($vol and $RP_EUR_VOLUME_KEEP)"
}

# RTX PRO 6000 parity baseline. The image is the CI build of the current
# commit (sha-<7>), never a locally built binary, so every result maps to a
# pushed main commit. Weight completeness is checked per cell on the box.
cmd_rtx6000() {
  require_tools curl jq ssh rsync git
  rp_load_key
  [[ -f "$RP_SSH_KEY" ]] || die "ssh key $RP_SSH_KEY missing"
  local sha img vol dc auth name id host port out
  sha="$(git -C "$FV_ROOT" rev-parse --short=7 HEAD)"
  git -C "$FV_ROOT" diff --quiet HEAD -- crates scripts || die "uncommitted changes: the CI image would not match"
  img="${RUNPOD_IMAGE_PIN:-ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-$sha}"
  RP_IMAGE="$img"
  vol="$(jq -r '.id // empty' "$RP_STATE/volume.json" 2>/dev/null || cat "$RP_STATE/volume.id" 2>/dev/null || true)"
  [[ -n "$vol" ]] || die "no volume id (run: runpod.sh fetch)"
  dc="$(rp_volume_dc "$vol")"
  auth="$(rp_ensure_registry_auth)"
  name="${GPU_NAME_PREFIX}-rtx6000-$(date -u +%Y%m%d%H%M%S)"
  rp_log "▶ rtx6000 parity: $RP_GPU_TYPE image=$img volume=$vol dc=$dc"
  id="$(rp_create_gpu_pod "$name" "$vol" "$dc" "$auth" "$RP_GPU_TYPE" 0)" || die "rtx6000 GPU create failed"
  echo "$id" >"$RP_STATE/gpu-rtx6000.id"
  nohup bash -c "sleep 14400; curl -sS -X DELETE -H 'Authorization: Bearer ${RUNPOD_API_KEY}' '${RUNPOD_API_BASE}/pods/${id}' >/dev/null" >/dev/null 2>&1 &
  read -r host port < <(rp_wait_ssh "$id" 900)
  rp_ssh "$host" "$port" "cat /opt/fastvideo-rs/target/release/fv-gpucheck.build-id; nvidia-smi -L"
  rp_ssh "$host" "$port" "FV_WEIGHTS=/workspace/weights bash /opt/fastvideo-rs/scripts/gpu/verify-weights.sh fasth3-8step fasth3-4step-vsa fasth3-4step-dense sol-h3 sol-h3-spark ltx25-two-stage" \
    || rp_log "WARN: some weight trees are incomplete; those cells will be skipped"
  rp_ssh "$host" "$port" "export FV_WORK=/workspace FV_GEN_TIMEOUT_S=${FV_GEN_TIMEOUT_S:-3600}
    bash /opt/fastvideo-rs/scripts/gpu/runpod-matrix.sh rtx6000" || rp_log "rtx6000 family returned non-zero"
  out="$RP_STATE/rtx6000/$sha"
  mkdir -p "$out"
  rsync -az --exclude 'frames/' -e "ssh -i $RP_SSH_KEY -p $port -o IdentitiesOnly=yes ${FV_SSH_OPTS[*]}" \
    "root@$host:/workspace/runs/rtx6000/" "$out/" || true
  rp_log "destroying rtx6000 pod $id"
  rp_destroy_pod "$id"
  rp_log "rtx6000 done → $out"
}

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

case "${1:-}" in
  fetch) shift; cmd_fetch "$@" ;;
  fetch-continue) cmd_fetch_continue ;;
  verify) cmd_verify ;;
  manifest) cmd_manifest ;;
  gpu) cmd_gpu ;;
  smoke) cmd_smoke ;;
  matrix) cmd_matrix ;;
  rtx6000) cmd_rtx6000 ;;
  b200) cmd_b200 ;;
  offers) cmd_offers ;;
  status) cmd_status ;;
  reap) cmd_reap ;;
  *) usage ;;
esac
