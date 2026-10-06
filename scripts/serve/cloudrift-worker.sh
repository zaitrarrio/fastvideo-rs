#!/usr/bin/env bash
# fv-serve on a CloudRift rental (docs/ops/cloudrift.md §6): a worker, or a
# standalone smoke.
#
#   cloudrift-worker.sh plan [image]          the rent payload, secrets masked (no API call;
#                                             FV_PLAN_ROLE=worker plans a worker,
#                                             FV_PLAN_SHOW_BOOT=1 adds the VM boot)
#   cloudrift-worker.sh up <family> [image]   a worker; prints "<id> <how it is reached>"
#   cloudrift-worker.sh smoke [image]         standalone fv-serve (fake engine) over HTTPS:
#                                             /health, /healthz, capabilities, one job,
#                                             timings, terminate
#   cloudrift-worker.sh down <id>             terminate
#
# GPUs: only RTX PRO 6000 and RTX 5090 (instance types rtxpro6000-*, rtx59-*;
# owner rule). Anything else is refused before a rent (cloudrift-lib.sh).
#
# Inbound (CLOUDRIFT_INBOUND):
#   none   (worker default) no published port at all. The worker dials OUT to
#          its family Durable Objects over WSS (edge_link.rs) and uploads its
#          output through the part URLs they mint (HTTPS); CloudRift's API is
#          HTTPS too. Needs FV_DISPATCH_DO_URL (https), FV_DISPATCH_FAMILIES
#          (default <family>) and the cluster's FV_INTERNAL_TOKEN_FILE.
#          FV_DISPATCH_SESSIONS is 0: a session needs a public endpoint.
#   https  (smoke default) HTTPS on the worker itself: Caddy on the VM with a
#          Let's Encrypt certificate for <dashed-ip>.sslip.io (or
#          CLOUDRIFT_TLS_HOSTNAME, our own name pointing at the VM), reverse
#          proxy to fv-serve on loopback; FV_PUBLIC_BASE_URL is that https URL
#          (session_ack endpoint, WHIP proxy). Never plain HTTP: if the address
#          cannot be found, no public URL is set.
#   tunnel-quick | tunnel-token | ssh   non-default fallbacks: a cloudflared quick
#          tunnel (URL read over SSH), a named tunnel (FV_CF_TUNNEL_TOKEN_FILE +
#          CLOUDRIFT_TUNNEL_HOSTNAME), or `ssh -L` (tests).
#
# Service (CLOUDRIFT_SERVICE):
#   vm     (default) a VM from CloudRift's NVIDIA Ubuntu recipe chosen per host
#          (nvidia_kernel_module_support ProprietaryOnly -> the proprietary-driver
#          recipe, else the newest open-driver one). Cloud-init installs Docker and
#          the NVIDIA container toolkit if missing, runs the serve image on the VM's
#          loopback (docker run --gpus all -p 127.0.0.1:8000:8000) and, for https,
#          Caddy. CLOUDRIFT_VOLUME mounts a weights volume at /workspace/weights.
#   docker the image as a CloudRift Docker rental with no published port: only for
#          CLOUDRIFT_INBOUND=none (nothing on the host to terminate TLS).
#
# Secrets: CloudRift has no secret store, so the env (run-key hash, internal
# token, FV_CLOUDRIFT_SECRETS_FILE values: FV_CF_*, FV_D1_DATABASE_ID, FV_R2_*,
# FV_WEBHOOK_ED25519_KEY, FV_URL_SIGNING_KEY) and any tunnel token are stored
# with the rental (base64 in its cloud-init or its Docker env). Use scoped,
# revocable tokens. plan masks them.
#
# Money guards: the balance floor (CLOUDRIFT_MIN_BALANCE, default 8 $), the $/hr cap
# (CLOUDRIFT_MAX_DPH, default 1.5) on the catalog price, a detached backstop
# (CLOUDRIFT_CAP_S: smoke 1800 s, up 3600 s), the fv-deadline tag fv-control enforces,
# terminate-on-exit (smoke), and the ledger artifacts/cloudrift/ledger.tsv.
#
# Env: CLOUDRIFT_API_KEY (or /root/.config/fv/cloudrift_api_key); FV_SERVE_IMAGE;
# FV_SERVE_CONFIG (default /etc/fv/runpod-fake.toml); CLOUDRIFT_GPUS (default
# "RTX PRO 6000,RTX 5090"); FV_DISPATCH_CAPACITY (2); FV_DISPATCH_SESSIONS;
# CLOUDRIFT_CADDY_IMAGE (caddy:2); CLOUDRIFT_SSH_KEY (only for the ssh and
# tunnel-quick fallbacks, or CLOUDRIFT_SSH_DEBUG=1); CLOUDRIFT_VM_USER (riftuser);
# FV_SMOKE_MODEL.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/cloudrift-lib.sh
source "$HERE/../gpu/cloudrift-lib.sh"

IMAGE_DEFAULT="ghcr.io/zaitrarrio/fastvideo-rs-serve:latest"
CONFIG="${FV_SERVE_CONFIG:-/etc/fv/runpod-fake.toml}"
GPUS="${CLOUDRIFT_GPUS:-$CR_ALLOWED_BRANDS}"
MAX_DPH="${CLOUDRIFT_MAX_DPH:-1.5}"
SERVICE="${CLOUDRIFT_SERVICE:-vm}"
MODEL="${FV_SMOKE_MODEL:-fake-wan}"
BOOT_WAIT_S="${CLOUDRIFT_BOOT_WAIT_S:-1200}"
SECRETS_FILE="${FV_CLOUDRIFT_SECRETS_FILE:-}"
TOKEN_FILE="${FV_INTERNAL_TOKEN_FILE:-}"
TUNNEL_TOKEN_FILE="${FV_CF_TUNNEL_TOKEN_FILE:-}"
TUNNEL_HOSTNAME="${CLOUDRIFT_TUNNEL_HOSTNAME:-}"
TLS_HOSTNAME="${CLOUDRIFT_TLS_HOSTNAME:-}"
CADDY_IMAGE="${CLOUDRIFT_CADDY_IMAGE:-caddy:2}"
VOLUME="${CLOUDRIFT_VOLUME:-}"
VM_USER="${CLOUDRIFT_VM_USER:-riftuser}"
SSH_KEY="${CLOUDRIFT_SSH_KEY:-$HOME/.ssh/id_ed25519_fv_cloudrift}"
SSH_BIN="${CLOUDRIFT_SSH_BIN:-ssh}"
OUT_DIR="${CLOUDRIFT_OUT_DIR:-$FV_ROOT/artifacts/cloudrift}/serve"
SECRET_NAMES='^(FV_CF_ACCOUNT_ID|FV_CF_API_TOKEN|FV_D1_DATABASE_ID|FV_R2_BUCKET|FV_R2_ENDPOINT|FV_R2_ACCESS_KEY_ID|FV_R2_SECRET_ACCESS_KEY|FV_WEBHOOK_ED25519_KEY|FV_URL_SIGNING_KEY)$'
FV_SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new
  -o "UserKnownHostsFile=${CLOUDRIFT_KNOWN_HOSTS:-$HOME/.ssh/known_hosts_fv_cloudrift}" -o IdentitiesOnly=yes)

# The VM boot (root, from cloud-init, in the background). Values come in as
# FVB_* lines prepended by vm_boot. Writes /var/lib/fv/{booted,public-url}
# (world-readable, no secrets); log /var/log/fv-boot.log.
# shellcheck disable=SC2016 # expanded on the VM
VM_BOOT_BODY='
set -eu
umask 077
mkdir -p /var/lib/fv/state
echo "$FVB_ENV_B64" | base64 -d >/var/lib/fv/env
step() { echo "[$(date -u +%T)] $*"; }
apt_get() { DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 "$@"; }
# The public IPv4, without depending on curl: bash /dev/tcp (plain HTTP to
# api.ipify.org) when curl is missing or fails.
public_ip() {
  local ip=""
  if command -v curl >/dev/null 2>&1; then ip=$(curl -fsS --max-time 10 https://api.ipify.org 2>/dev/null) || ip=""; fi
  if [ -z "$ip" ]; then
    ip=$(exec 3<>/dev/tcp/api.ipify.org/80 && printf "GET / HTTP/1.0\r\nHost: api.ipify.org\r\n\r\n" >&3 && timeout 10 cat <&3 | tail -n 1) || ip=""
  fi
  case "$ip" in *[!0-9.]* | "") echo "" ;; *) echo "$ip" ;; esac
}
if ! command -v docker >/dev/null 2>&1; then
  step "installing docker"; apt_get update -q && apt_get install -y -q docker.io
fi
if ! docker info 2>/dev/null | grep -qi nvidia; then
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    step "installing the NVIDIA container toolkit"
    command -v curl >/dev/null 2>&1 || apt_get install -y -q curl
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" \
      >/etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt_get update -q && apt_get install -y -q nvidia-container-toolkit
  fi
  nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
fi
(umask 022; nvidia-smi >/var/lib/fv/nvidia-smi.txt 2>&1 || true)
echo "FV_WORKER_ID=cr-$(hostname)" >>/var/lib/fv/env
url=""
if [ "$FVB_INBOUND" = https ]; then
  host="$FVB_TLS_HOSTNAME"
  if [ -z "$host" ]; then ip=$(public_ip); [ -n "$ip" ] && host="$(echo "$ip" | tr . -).sslip.io"; fi
  if [ -n "$host" ]; then
    step "HTTPS front: Caddy for $host"
    docker run -d --name fv-tls --restart unless-stopped --network host -v /var/lib/fv/caddy:/data "$FVB_CADDY_IMAGE" \
      caddy reverse-proxy --from "$host" --to 127.0.0.1:8000
    url="https://$host"
    echo "FV_PUBLIC_BASE_URL=$url" >>/var/lib/fv/env
  else
    step "WARNING: no public address found: no HTTPS front, no public URL (never plain HTTP)"
  fi
fi
vol=""
if [ -n "$FVB_VOLUME_PATH" ]; then vol="-v $FVB_VOLUME_PATH:/workspace/weights:ro -e FV_WEIGHTS=/workspace/weights"; fi
step "pulling $FVB_IMAGE"
docker pull -q "$FVB_IMAGE"
step "starting fv-serve"
# shellcheck disable=SC2086
docker run -d --name fv-serve --restart unless-stopped --gpus all -p 127.0.0.1:8000:8000 \
  --env-file /var/lib/fv/env -v /var/lib/fv/state:/fvstate $vol "$FVB_IMAGE" $FVB_ARGS
case "$FVB_INBOUND" in
  tunnel-quick | tunnel-token)
    if ! command -v cloudflared >/dev/null 2>&1; then
      step "installing cloudflared"
      curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
      dpkg -i /tmp/cloudflared.deb
    fi ;;
esac
case "$FVB_INBOUND" in
  tunnel-quick)
    nohup cloudflared tunnel --no-autoupdate --metrics 127.0.0.1:20241 --url http://127.0.0.1:8000 >/var/log/cloudflared.log 2>&1 &
    for _ in $(seq 120); do
      h=$(curl -fsS http://127.0.0.1:20241/quicktunnel 2>/dev/null | sed -n "s/.*\"hostname\":\"\([^\"]*\)\".*/\1/p")
      if [ -n "$h" ]; then url="https://$h"; break; fi
      sleep 2
    done ;;
  tunnel-token)
    TUNNEL_TOKEN="$(cat /var/lib/fv/tunnel-token)" nohup cloudflared tunnel --no-autoupdate run >/var/log/cloudflared.log 2>&1 &
    url="https://$FVB_TUNNEL_HOSTNAME" ;;
esac
(umask 022; echo "$url" >/var/lib/fv/public-url; date -u +%FT%TZ >/var/lib/fv/booted)
step "booted${url:+ ($url)}"'

# Secrets from the file: a JSON object (values never printed).
secrets_json() {
  local f="$1" line k v out='{}'
  [[ -n "$f" ]] || { echo '{}'; return; }
  [[ -r "$f" ]] || die "cannot read FV_CLOUDRIFT_SECRETS_FILE $f"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    k="${line%%=*}"; v="${line#*=}"
    [[ "$k" =~ $SECRET_NAMES ]] || die "secrets file: $k is not a secret this script passes"
    out="$(jq -c --arg k "$k" --arg v "$v" '. + {($k): $v}' <<<"$out")"
  done <"$f"
  echo "$out"
}

# inbound_for <role>: CLOUDRIFT_INBOUND, else none (worker) or https (smoke).
inbound_for() {
  local i="${CLOUDRIFT_INBOUND:-}"
  [[ -n "$i" ]] || { if [[ "$1" == worker ]]; then i=none; else i=https; fi; }
  case "$i" in none | https | tunnel-quick | tunnel-token | ssh) echo "$i" ;;
    *) die "CLOUDRIFT_INBOUND must be none, https, tunnel-quick, tunnel-token or ssh" ;; esac
}

# env_json <role> <keyhash|""> <family|""> <secrets json> <token|""> <inbound>
# The family Durable Object worker settings are the ones scripts/gcp/vm.sh and
# fv-control give a worker (docs/serve/dispatch-do-family.md).
env_json() {
  jq -nc --arg role "$1" --arg keys "$2" --arg fam "$3" --argjson sec "$4" --arg tok "$5" --arg inb "$6" \
    --arg cfg "$CONFIG" --arg dourl "${FV_DISPATCH_DO_URL:-}" --arg fams "${FV_DISPATCH_FAMILIES:-}" \
    --arg cap "${FV_DISPATCH_CAPACITY:-2}" --arg ses "${FV_DISPATCH_SESSIONS:-}" '
    {FV_SERVE_MODE: "http", FV_STATE_DIR: "/fvstate", FV_SERVE_CONFIG: $cfg, RUST_LOG: "info",
     NVIDIA_DRIVER_CAPABILITIES: "compute,utility,video"}
    + (if $role == "worker" then
        {FV_SERVE_ROLE: "worker", FV_INTERNAL_TOKEN: $tok, FV_JOBS_HEARTBEAT_S: "10",
         FV_DISPATCH_DO_URL: $dourl, FV_DISPATCH_FAMILIES: (if $fams == "" then $fam else $fams end),
         FV_DISPATCH_DIRECT_UPLOAD: "1", FV_DISPATCH_CAPACITY: $cap,
         FV_DISPATCH_SESSIONS: (if $ses != "" then $ses elif $inb == "none" then "0" else "1" end)}
       else {FV_API_KEYS: $keys} end)
    + $sec'
}

# vm_boot <image> <env json> <inbound> <tunnel token|""> -> the cloud-init command
# line: the boot script (FVB_* values + VM_BOOT_BODY) base64-encoded, decoded to a
# root-only file and started in the background.
vm_boot() {
  local image="$1" env="$2" inbound="$3" ttok="$4" envb64 script
  [[ "$CONFIG" != *[[:space:]]* ]] || die "FV_SERVE_CONFIG must not contain spaces"
  envb64="$(jq -r 'to_entries[] | "\(.key)=\(.value)"' <<<"$env" | base64 -w0)"
  script="$(printf 'FVB_IMAGE=%q\nFVB_ARGS=%q\nFVB_ENV_B64=%q\nFVB_INBOUND=%q\nFVB_TLS_HOSTNAME=%q\nFVB_CADDY_IMAGE=%q\nFVB_TUNNEL_HOSTNAME=%q\nFVB_VOLUME_PATH=%q\n' \
    "$image" "--config $CONFIG" "$envb64" "$inbound" "$TLS_HOSTNAME" "$CADDY_IMAGE" "$TUNNEL_HOSTNAME" "${VOLUME:+/workspace/weights}")"
  [[ -z "$ttok" ]] || script+=$'\n'"(umask 077; mkdir -p /var/lib/fv; printf '%s' $(printf '%q' "$ttok") >/var/lib/fv/tunnel-token)"
  script+="$VM_BOOT_BODY"
  printf 'umask 077; echo %s | base64 -d >/root/fv-boot.sh && nohup bash /root/fv-boot.sh >/var/log/fv-boot.log 2>&1 &' \
    "$(base64 -w0 <<<"$script")"
}

# payload <image> <variant> <dc> <env json> <kind> <deadline> <driver> <inbound> <tunnel token|""> <ssh pub|"">
payload() {
  local name tags
  name="fv-serve-$5-$(date -u +%m%d%H%M%S)"
  tags="$(cr_tags "serve-$5" "$6")"
  if [[ "$SERVICE" == vm ]]; then
    local img p
    if [[ "$2" == "<variant>" ]]; then img="<recipe image for driver $7>"; else img="$(cr_recipe_image "$7")" || return 1; fi
    p="$(cr_vm_payload "$2" "$3" "$name" "$img" "$(vm_boot "$1" "$4" "$8" "$9")" "${10}" "$tags")"
    if [[ -n "$VOLUME" ]]; then
      p="$(jq -c --arg v "$VOLUME" '.config.VirtualMachine.volumes = {Mounts: [{volume: {ByName: [$v]}, mount_path: "/workspace/weights"}]}' <<<"$p")"
    fi
    printf '%s\n' "$p"
  else
    # Docker: no published port (outbound only); the image ENTRYPOINT stays.
    cr_docker_payload "$2" "$3" "$name" "$1" "$(jq -nc --arg c "$CONFIG" '["--config", $c]')" "$4" '[]' "$tags"
  fi
}

validate() {
  local p="$1" it
  jq -e '.tags | index("fv-owner:fastvideo-rs")' <<<"$p" >/dev/null || die "the owner tag is missing"
  it="$(jq -r '.selector.ByInstanceTypeAndLocation.instance_type' <<<"$p")"
  [[ "$it" == "<variant>" ]] || cr_type_allowed "$it" || die "instance type $it is not allowed (rtxpro6000-*, rtx59-* only)"
  if jq -e '.config.VirtualMachine' <<<"$p" >/dev/null; then
    jq -e '.config.VirtualMachine.cloudinit_commands | length > 0' <<<"$p" >/dev/null || die "no cloud-init boot"
    bash -n <<<"$VM_BOOT_BODY" || die "the VM boot is not valid bash"
  else
    jq -e '(.config.Docker.ports // []) | length == 0' <<<"$p" >/dev/null || die "a Docker rental publishes no port (outbound only)"
    jq -e '.config.Docker.env | all(type == "array" and length == 2)' <<<"$p" >/dev/null || die "env must be [name, value] pairs"
  fi
}

# The payload as plan prints it: secret values masked; the VM boot replaced.
masked() {
  jq --arg re "$SECRET_NAMES" '
    if .config.Docker then
      .config.Docker.env |= map(if (.[0] | test($re)) or .[0] == "FV_INTERNAL_TOKEN" then [.[0], "<secret>"] else . end)
    else .config.VirtualMachine.cloudinit_commands = "<fv-boot: docker run on loopback + inbound front; env and tokens base64-embedded, not shown>" end' <<<"$1"
}

ssh_pub() {
  if [[ ! -f "$SSH_KEY" ]]; then
    require_tools ssh-keygen
    mkdir -p "$(dirname "$SSH_KEY")"
    ssh-keygen -q -t ed25519 -N '' -C fv-cloudrift -f "$SSH_KEY" >/dev/null
  fi
  if [[ -f "$SSH_KEY.pub" ]]; then cat "$SSH_KEY.pub"; else ssh-keygen -y -f "$SSH_KEY"; fi
}
vm_ssh() { # host cmd...
  local host="$1"; shift
  "$SSH_BIN" -n -i "$SSH_KEY" "${FV_SSH_OPTS[@]}" "$VM_USER@$host" "$@"
}
needs_ssh() { [[ "$1" == ssh || "$1" == tunnel-quick || "${CLOUDRIFT_SSH_DEBUG:-0}" == 1 ]]; }

ID="" SSH_TUNNEL_PID=""
cleanup() {
  local rc=$?
  [[ -z "$SSH_TUNNEL_PID" ]] || kill "$SSH_TUNNEL_PID" 2>/dev/null || true
  if [[ -n "$ID" ]]; then
    log "terminate-on-exit: $ID"
    cr_terminate "$ID" || log "WARNING: the backstop will retry $ID"
    ID=""
  fi
  exit "$rc"
}

# rent <image> <env json> <kind> <cap s> <inbound>: sets ID, VARIANT, DPH.
VARIANT="" DPH=""
rent() {
  local pick dc driver deadline p pub="" ttok=""
  case "$SERVICE" in
    vm) ;;
    docker) [[ "$5" == none ]] || die "a Docker rental has no host to terminate TLS on: CLOUDRIFT_SERVICE=docker only with CLOUDRIFT_INBOUND=none" ;;
    *) die "CLOUDRIFT_SERVICE must be vm or docker" ;;
  esac
  if [[ "$5" == tunnel-token ]]; then
    [[ -r "$TUNNEL_TOKEN_FILE" && -n "$TUNNEL_HOSTNAME" ]] \
      || die "tunnel-token: set FV_CF_TUNNEL_TOKEN_FILE (mode 600) and CLOUDRIFT_TUNNEL_HOSTNAME"
    ttok="$(tr -d '[:space:]' <"$TUNNEL_TOKEN_FILE")"
  fi
  cr_check_brands "$GPUS"
  cr_check_balance
  pick="$(cr_pick "$GPUS" "$MAX_DPH" 1 "$SERVICE")" || die "no free 1-GPU stock of [$GPUS] under \$$MAX_DPH/hr (allowed: $CR_ALLOWED_BRANDS)"
  read -r VARIANT DPH dc driver <<<"$pick"
  [[ "$dc" != - ]] || dc=""
  ! needs_ssh "$5" || [[ "$SERVICE" != vm ]] || pub="$(ssh_pub)"
  deadline=$(($(date +%s) + $4))
  p="$(payload "$1" "$VARIANT" "$dc" "$2" "$3" "$deadline" "$driver" "$5" "$ttok" "$pub")" || die "could not build the rent payload"
  validate "$p"
  ID="$(cr_rent "$p")" || die "rent failed ($VARIANT, $SERVICE)"
  cr_ledger "instance-rented $ID kind=serve-$3 service=$SERVICE variant=$VARIANT dph=$DPH image=$1 cap=$4s inbound=$5"
  cr_backstop "$ID" "$4"
  log "instance $ID: $VARIANT${dc:+ in $dc} at \$$DPH/hr, $SERVICE, inbound $5 (backstop $4 s)"
}

# base_url <host> <inbound> <t0> -> the URL the smoke talks to.
base_url() {
  local host="$1" inbound="$2" t0="$3" url lp
  case "$inbound" in
    https | tunnel-token)
      if [[ "$inbound" == tunnel-token ]]; then url="https://$TUNNEL_HOSTNAME"
      elif [[ -n "$TLS_HOSTNAME" ]]; then url="https://$TLS_HOSTNAME"
      else url="https://${host//./-}.sslip.io"; fi
      log "public URL $url"
      # Tests only: talk to the mock instead of the computed URL.
      echo "${CLOUDRIFT_TEST_BASE_URL:-$url}"
      return ;;
  esac
  until vm_ssh "$host" true 2>/dev/null; do
    (( $(date +%s) - t0 < BOOT_WAIT_S )) || die "no ssh to $VM_USER@$host after ${BOOT_WAIT_S}s"
    sleep "${CR_POLL_S:-10}"
  done
  until vm_ssh "$host" test -f /var/lib/fv/booted 2>/dev/null; do
    (( $(date +%s) - t0 < BOOT_WAIT_S )) || die "the VM boot did not finish after ${BOOT_WAIT_S}s"
    sleep "${CR_POLL_S:-10}"
  done
  if [[ "$inbound" == ssh ]]; then
    lp="${CLOUDRIFT_LOCAL_PORT:-18000}"
    "$SSH_BIN" -N -i "$SSH_KEY" "${FV_SSH_OPTS[@]}" -L "127.0.0.1:$lp:127.0.0.1:8000" "$VM_USER@$host" &
    SSH_TUNNEL_PID=$!
    echo "${CLOUDRIFT_TEST_BASE_URL:-http://127.0.0.1:$lp}"  # loopback through the SSH tunnel
    return
  fi
  url="$(vm_ssh "$host" cat /var/lib/fv/public-url 2>/dev/null | tr -d '[:space:]')"
  [[ "$url" == https://* ]] || die "no tunnel URL from the VM"
  log "public URL $url"
  echo "${CLOUDRIFT_TEST_BASE_URL:-$url}"
}

http() { curl -sS --max-time 60 "$@"; }

cmd_smoke() {
  require_tools curl jq openssl sha256sum
  cr_need_key
  local image key keyhash inst host base t0 t_active t_ready st job id health inbound
  image="${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}"
  [[ "$image" == *@sha256:* ]] || log "warning: the image is not digest-pinned"
  inbound="$(inbound_for standalone)"
  [[ "$inbound" != none ]] || die "smoke talks to fv-serve: it needs inbound (https by default)"
  key="fvk-$(openssl rand -hex 16)"
  keyhash="$(printf '%s' "$key" | sha256sum | cut -d' ' -f1)"
  trap cleanup EXIT INT TERM
  t0="$(date +%s)"
  rent "$image" "$(env_json standalone "$keyhash" "" '{}' "" "$inbound")" smoke "${CLOUDRIFT_CAP_S:-1800}" "$inbound"
  inst="$(cr_wait_active "$ID" "$BOOT_WAIT_S")" || die "instance $ID did not become Active"
  t_active="$(date +%s)"
  host="$(jq -r .host_address <<<"$inst")"
  base="$(base_url "$host" "$inbound" "$t0")"
  # TLS is verified (curl's default): the certificate is issued on the first boot.
  until curl -fsS --max-time 10 "$base/healthz" >/dev/null 2>&1; do
    (( $(date +%s) - t0 < BOOT_WAIT_S )) || die "$base/healthz never answered"
    sleep "${CR_POLL_S:-10}"
  done
  t_ready="$(date +%s)"
  log "Active after $((t_active - t0)) s, /healthz after $((t_ready - t0)) s ($base)"
  health="$(http "$base/health" || echo '{}')"
  jq -c '{state, build: .build.git_sha}' <<<"$health" >&2 || true
  http -H "Authorization: Bearer $key" "$base/fv/v1/capabilities" | jq -c '[.models[]?.caps.id // .models[]?.id]' >&2 || true
  job="$(http -H "Authorization: Bearer $key" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"a red fox trotting through fresh snow\",\"seed\":1}" "$base/fv/v1/jobs")"
  id="$(jq -r '.id // empty' <<<"$job")"
  st='{"status":"not submitted"}'
  if [[ -n "$id" ]]; then
    for _ in $(seq 1 60); do
      st="$(http -H "Authorization: Bearer $key" "$base/fv/v1/jobs/$id")"
      case "$(jq -r .status <<<"$st")" in succeeded | failed | cancelled) break ;; esac
      sleep 2
    done
  else
    log "job submit answered: $(head -c 300 <<<"$job")"
  fi
  mkdir -p "$OUT_DIR"
  jq -n --arg id "$ID" --arg v "$VARIANT" --argjson dph "$DPH" --arg image "$image" --arg base "$base" --arg svc "$SERVICE" --arg inb "$inbound" \
    --argjson ta "$((t_active - t0))" --argjson tr "$((t_ready - t0))" --argjson wall "$(($(date +%s) - t0))" --argjson st "$st" \
    --argjson health "$(jq -c '{state}' <<<"$health" 2>/dev/null || echo '{}')" \
    '{target: "cloudrift", service: $svc, inbound: $inb, instance: $id, variant: $v, usd_per_hr: $dph, image: $image, public_url: $base,
      rent_to_active_s: $ta, rent_to_healthz_s: $tr, active_to_healthz_s: ($tr - $ta), wall_s: $wall,
      est_cost_usd: ($dph * $wall / 3600), health: $health, job: ($st | {id, model, status})}' | tee "$OUT_DIR/smoke-$(date -u +%m%d%H%M%S).json"
  cr_terminate "$ID" && ID=""
  [[ "$(jq -r .status <<<"$st")" == succeeded ]] || die "job ended $(jq -c '{status, error}' <<<"$st")"
}

cmd_up() {
  local family="${1:?family (or pool) id}" image tok sec inst host inbound reach
  require_tools curl jq
  cr_need_key
  image="${2:-${FV_SERVE_IMAGE:-}}"
  [[ "$image" == *@sha256:* ]] || die "pin the image digest (FV_SERVE_IMAGE=ghcr.io/...@sha256:...)"
  [[ -r "$TOKEN_FILE" ]] || die "FV_INTERNAL_TOKEN_FILE: the cluster's internal token (a file, mode 600)"
  [[ -n "${FV_DISPATCH_DO_URL:-}" ]] || die "set FV_DISPATCH_DO_URL (the fv-edge Worker's https base URL): workers dial out to the family Durable Objects"
  [[ "$FV_DISPATCH_DO_URL" == https://* ]] || die "FV_DISPATCH_DO_URL must be https://"
  inbound="$(inbound_for worker)"
  tok="$(tr -d '[:space:]' <"$TOKEN_FILE")"
  sec="$(secrets_json "$SECRETS_FILE")"
  rent "$image" "$(env_json worker "" "$family" "$sec" "$tok" "$inbound")" "$family" "${CLOUDRIFT_CAP_S:-3600}" "$inbound"
  inst="$(cr_wait_active "$ID" "$BOOT_WAIT_S")" || { cr_terminate "$ID"; die "instance $ID did not become Active"; }
  host="$(jq -r .host_address <<<"$inst")"
  case "$inbound" in
    none) reach="outbound-only: dials $FV_DISPATCH_DO_URL (families ${FV_DISPATCH_FAMILIES:-$family}); no public port" ;;
    https) reach="$(base_url "$host" https 0) (sessions endpoint; dials $FV_DISPATCH_DO_URL)" ;;
    ssh) reach="ssh -L <port>:127.0.0.1:8000 $VM_USER@$host" ;;
    *) reach="$(base_url "$host" "$inbound" "$(date +%s)")" ;;
  esac
  echo "$ID $reach"
  log "worker $ID ($SERVICE, inbound $inbound): it shows up in the family Durable Object once booted"
}

case "${1:-}" in
  plan)
    shift
    secj="$(secrets_json "$SECRETS_FILE")"
    role="${FV_PLAN_ROLE:-standalone}"
    inb="$(inbound_for "$role")"
    p="$(payload "${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}" "<variant>" "" "$(env_json "$role" "<sha256 of the run key>" "${FV_PLAN_FAMILY:-h3-turbo}" "$secj" "<internal token>" "$inb")" plan "$(($(date +%s) + 1800))" "${FV_PLAN_DRIVER:-OpenAndProprietary}" "$inb" "$([[ "$inb" == tunnel-token ]] && echo '<tunnel token>')" "$(needs_ssh "$inb" && echo "ssh-ed25519 AAAA<public key> fv-cloudrift")")"
    validate "$p"
    echo "# POST $CR_API/api/v1/instances/rent  (version $CR_VERSION, service $SERVICE, inbound $inb)"
    masked "$p"
    if [[ "${FV_PLAN_SHOW_BOOT:-0}" == 1 ]] && jq -e '.config.VirtualMachine' <<<"$p" >/dev/null; then
      # The VM boot as the VM runs it, the env and the tunnel token masked.
      echo "# /root/fv-boot.sh"
      jq -r '.config.VirtualMachine.cloudinit_commands' <<<"$p" | sed -n 's/^umask 077; echo \([^ ]*\) | base64.*/\1/p' | base64 -d \
        | sed -e "s/^FVB_ENV_B64=.*/FVB_ENV_B64='<masked>'/" -e "s/^(umask 077; mkdir -p \/var\/lib\/fv; printf '%s' .*/: tunnel token written, masked/"
    fi
    log "plan: payload shape OK" ;;
  smoke) shift; cmd_smoke "$@" ;;
  up) shift; cmd_up "$@" ;;
  down) cr_need_key; cr_terminate "${2:?instance id}" && echo "terminated $2" ;;
  *) sed -n '2,62p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
