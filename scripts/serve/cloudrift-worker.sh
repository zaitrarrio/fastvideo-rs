#!/usr/bin/env bash
# fv-serve on a CloudRift rental (docs/ops/cloudrift.md §6): a standalone
# smoke, or a worker for a gateway's pod pool.
#
#   cloudrift-worker.sh plan [image]        print the rent payload; secrets masked (no API call)
#   cloudrift-worker.sh smoke [image]       standalone fv-serve (fake engine): rent, reach it through
#                                           the tunnel, /health + /healthz, capabilities, one job,
#                                           timings, terminate
#   cloudrift-worker.sh up <pool> [image]   a worker (FV_SERVE_ROLE=worker) for pool <pool>;
#                                           prints the id and the URL to give the gateway
#   cloudrift-worker.sh down <id>           terminate
#
# Two ways to run the container (CLOUDRIFT_SERVICE):
#   vm      A CloudRift VM from CloudRift's NVIDIA Ubuntu recipe, picked from the
#           host's nvidia_kernel_module_support (ProprietaryOnly hosts such as the
#           V100 nodes need the proprietary-driver recipe). Cloud-init installs
#           Docker and the NVIDIA container toolkit when the image lacks them and
#           starts the serve image with `docker run --gpus all`, published on the
#           VM's loopback only. The public front is a tunnel (CLOUDRIFT_TUNNEL).
#   docker  A CloudRift Docker rental (the image runs as the container). Plain
#           HTTP on the host's public port: smoke only, never a worker.
#   auto    (default) vm on ProprietaryOnly hosts, docker elsewhere.
# Live on 2026-10-06 every Docker rental on the V100 hosts (the only free stock)
# failed within seconds with "Internal provisioning error" (docs §8).
#
# Tunnel (CLOUDRIFT_TUNNEL; vm only; owner decision 2026-10-06: no plain-HTTP
# internal token over the internet):
#   quick   (smoke default) cloudflared quick tunnel: an https://*.trycloudflare.com
#           URL, read back over SSH (the VM writes it to /var/lib/fv/public-url).
#   token   (up default) a named Cloudflare tunnel run with FV_CF_TUNNEL_TOKEN_FILE
#           (its token, mode 600; set up in the Cloudflare dashboard with a public
#           hostname -> http://localhost:8000). CLOUDRIFT_TUNNEL_HOSTNAME is that
#           hostname; the script prints https://<hostname>.
#   ssh     nothing public: `ssh -L` to the VM's loopback port (tests).
#
# Reaching the gateway (pod pool kind, docs/serve/gateway.md §5.3): add the
# printed https URL to the gateway's FV_POOL_<POOL>_URLS (static pod pool).
#
# Secrets: CloudRift has no secret store, so the values of FV_CLOUDRIFT_SECRETS_FILE
# (KEY=VALUE lines: FV_CF_*, FV_D1_DATABASE_ID, FV_R2_*, FV_WEBHOOK_ED25519_KEY,
# FV_URL_SIGNING_KEY), FV_INTERNAL_TOKEN_FILE and the tunnel token go into the
# rental's cloud-init / env, which CloudRift stores with the rental. Use scoped,
# revocable tokens. They are never printed (plan masks them).
#
# Weights: CLOUDRIFT_VOLUME=<CloudRift volume name> mounts that volume at
# /workspace/weights in the VM and read-only into the container (FV_WEIGHTS).
# On 2026-10-06 no datacenter could create a volume (docs §3).
#
# Money guards: the balance floor (CLOUDRIFT_MIN_BALANCE, default 8 $), the $/hr cap
# (CLOUDRIFT_MAX_DPH, default 1.0) on the catalog price, a detached backstop
# (CLOUDRIFT_CAP_S: smoke 1800 s, up 3600 s), the fv-deadline tag fv-control enforces,
# terminate-on-exit (smoke), and the ledger artifacts/cloudrift/ledger.tsv.
#
# Env: CLOUDRIFT_API_KEY (or /root/.config/fv/cloudrift_api_key); FV_SERVE_IMAGE;
# FV_SERVE_CONFIG (default /etc/fv/runpod-fake.toml); CLOUDRIFT_GPUS (default
# "RTX PRO 6000,RTX 5090,RTX 4090,L40S"); CLOUDRIFT_SSH_KEY (default
# ~/.ssh/id_ed25519_fv_cloudrift, made if missing); CLOUDRIFT_VM_USER (riftuser);
# CLOUDRIFT_HOST_PORT (docker: 8000); FV_SMOKE_MODEL.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/cloudrift-lib.sh
source "$HERE/../gpu/cloudrift-lib.sh"

IMAGE_DEFAULT="ghcr.io/zaitrarrio/fastvideo-rs-serve:latest"
CONFIG="${FV_SERVE_CONFIG:-/etc/fv/runpod-fake.toml}"
GPUS="${CLOUDRIFT_GPUS:-RTX PRO 6000,RTX 5090,RTX 4090,L40S}"
MAX_DPH="${CLOUDRIFT_MAX_DPH:-1.0}"
SERVICE="${CLOUDRIFT_SERVICE:-auto}"
MODE="${CLOUDRIFT_CMD_MODE:-args}"
HOST_PORT="${CLOUDRIFT_HOST_PORT:-8000}"
MODEL="${FV_SMOKE_MODEL:-fake-wan}"
BOOT_WAIT_S="${CLOUDRIFT_BOOT_WAIT_S:-1200}"
SECRETS_FILE="${FV_CLOUDRIFT_SECRETS_FILE:-}"
TOKEN_FILE="${FV_INTERNAL_TOKEN_FILE:-}"
TUNNEL_TOKEN_FILE="${FV_CF_TUNNEL_TOKEN_FILE:-}"
TUNNEL_HOSTNAME="${CLOUDRIFT_TUNNEL_HOSTNAME:-}"
VOLUME="${CLOUDRIFT_VOLUME:-}"
VM_USER="${CLOUDRIFT_VM_USER:-riftuser}"
SSH_KEY="${CLOUDRIFT_SSH_KEY:-$HOME/.ssh/id_ed25519_fv_cloudrift}"
SSH_BIN="${CLOUDRIFT_SSH_BIN:-ssh}"
OUT_DIR="${CLOUDRIFT_OUT_DIR:-$FV_ROOT/artifacts/cloudrift}/serve"
SECRET_NAMES='^(FV_CF_ACCOUNT_ID|FV_CF_API_TOKEN|FV_D1_DATABASE_ID|FV_R2_BUCKET|FV_R2_ENDPOINT|FV_R2_ACCESS_KEY_ID|FV_R2_SECRET_ACCESS_KEY|FV_WEBHOOK_ED25519_KEY|FV_URL_SIGNING_KEY)$'
FV_SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new
  -o "UserKnownHostsFile=${CLOUDRIFT_KNOWN_HOSTS:-$HOME/.ssh/known_hosts_fv_cloudrift}" -o IdentitiesOnly=yes)

# Docker exec mode: the public URL first, then fv-serve through the image
# entrypoint. The per-variant serve images have no curl, so the lookup falls
# back to bash's /dev/tcp (plain HTTP to api.ipify.org).
# shellcheck disable=SC2016 # expanded inside the container
BOOT='set -u
ip=""
if command -v curl >/dev/null 2>&1; then
  for u in https://api.ipify.org https://ifconfig.me/ip; do
    ip=$(curl -fsS --max-time 10 "$u" 2>/dev/null) && [ -n "$ip" ] && break
  done
fi
if [ -z "$ip" ]; then
  ip=$(exec 3<>/dev/tcp/api.ipify.org/80 && printf "GET / HTTP/1.0\r\nHost: api.ipify.org\r\n\r\n" >&3 && timeout 10 cat <&3 | tail -n 1) || ip=""
fi
[ -n "$ip" ] && export FV_PUBLIC_BASE_URL="http://$ip:$FV_HOST_PORT"
mkdir -p /fvstate
entry=/opt/fastvideo-rs/bin/fv-entry; [ -x "$entry" ] || entry=/opt/fastvideo-rs/bin/fv-serve
exec "$entry" --config "$FV_SERVE_CONFIG"'

# The VM boot (runs as root from cloud-init, in the background). Values come in
# as FVB_* lines prepended by vm_boot. Writes /var/lib/fv/{booted,public-url}
# (world-readable, no secrets) and logs to /var/log/fv-boot.log.
# shellcheck disable=SC2016 # expanded on the VM
VM_BOOT_BODY='
set -eu
umask 077
mkdir -p /var/lib/fv/state
echo "$FVB_ENV_B64" | base64 -d >/var/lib/fv/env
step() { echo "[$(date -u +%T)] $*"; }
apt_get() { DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 "$@"; }
if ! command -v docker >/dev/null 2>&1; then
  step "installing docker"; apt_get update -q && apt_get install -y -q docker.io curl
fi
if ! docker info 2>/dev/null | grep -qi nvidia; then
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    step "installing the NVIDIA container toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" \
      >/etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt_get update -q && apt_get install -y -q nvidia-container-toolkit
  fi
  nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
fi
(umask 022; nvidia-smi >/var/lib/fv/nvidia-smi.txt 2>&1 || true)
vol=""
if [ -n "$FVB_VOLUME_PATH" ]; then vol="-v $FVB_VOLUME_PATH:/workspace/weights:ro -e FV_WEIGHTS=/workspace/weights"; fi
step "pulling $FVB_IMAGE"
docker pull -q "$FVB_IMAGE"
step "starting fv-serve"
# shellcheck disable=SC2086
docker run -d --name fv-serve --restart unless-stopped --gpus all -p 127.0.0.1:8000:8000 \
  --env-file /var/lib/fv/env -v /var/lib/fv/state:/fvstate $vol "$FVB_IMAGE" $FVB_ARGS
case "$FVB_TUNNEL" in
  quick | token)
    if ! command -v cloudflared >/dev/null 2>&1; then
      step "installing cloudflared"
      curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
      dpkg -i /tmp/cloudflared.deb
    fi ;;
esac
case "$FVB_TUNNEL" in
  quick)
    nohup cloudflared tunnel --no-autoupdate --metrics 127.0.0.1:20241 --url http://127.0.0.1:8000 >/var/log/cloudflared.log 2>&1 &
    for _ in $(seq 120); do
      h=$(curl -fsS http://127.0.0.1:20241/quicktunnel 2>/dev/null | sed -n "s/.*\"hostname\":\"\([^\"]*\)\".*/\1/p")
      if [ -n "$h" ]; then (umask 022; echo "https://$h" >/var/lib/fv/public-url); break; fi
      sleep 2
    done ;;
  token)
    TUNNEL_TOKEN="$(cat /var/lib/fv/tunnel-token)" nohup cloudflared tunnel --no-autoupdate run >/var/log/cloudflared.log 2>&1 &
    (umask 022; echo "https://$FVB_TUNNEL_HOSTNAME" >/var/lib/fv/public-url) ;;
esac
(umask 022; date -u +%FT%TZ >/var/lib/fv/booted)
step "booted"'

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

# env_json <role> <keyhash|""> <pool|""> <secrets json> <token|"">
env_json() {
  jq -nc --arg role "$1" --arg keys "$2" --arg pool "$3" --argjson sec "$4" --arg tok "$5" --arg cfg "$CONFIG" --arg hp "$HOST_PORT" '
    {FV_SERVE_MODE: "http", FV_STATE_DIR: "/fvstate", FV_SERVE_CONFIG: $cfg, FV_HOST_PORT: $hp, RUST_LOG: "info",
     NVIDIA_DRIVER_CAPABILITIES: "compute,utility,video"}
    + (if $role == "worker" then {FV_SERVE_ROLE: "worker", FV_GATEWAY_POOL: $pool, FV_INTERNAL_TOKEN: $tok, FV_JOBS_HEARTBEAT_S: "10"}
       else {FV_API_KEYS: $keys, FV_SERVE_FORWARD: "1"} end)
    + $sec'
}

command_json() {
  case "$MODE" in
    args) jq -nc --arg c "$CONFIG" '["--config", $c]' ;;
    exec) jq -nc --arg b "$BOOT" '["bash", "-c", $b]' ;;
    *) die "CLOUDRIFT_CMD_MODE must be args or exec" ;;
  esac
}

# The service for a host: CLOUDRIFT_SERVICE, or (auto) vm on ProprietaryOnly hosts.
service_for() {
  case "$SERVICE" in
    vm | docker) echo "$SERVICE" ;;
    auto) if [[ "$1" == ProprietaryOnly ]]; then echo vm; else echo docker; fi ;;
    *) die "CLOUDRIFT_SERVICE must be auto, vm or docker" ;;
  esac
}

# The tunnel of a VM: CLOUDRIFT_TUNNEL, else quick (smoke) or token (up).
tunnel_for() {
  local t="${CLOUDRIFT_TUNNEL:-}"
  [[ -n "$t" ]] || { if [[ "$1" == worker ]]; then t=token; else t=quick; fi; }
  case "$t" in quick | token | ssh) echo "$t" ;; *) die "CLOUDRIFT_TUNNEL must be quick, token or ssh" ;; esac
}

# vm_boot <image> <env json> <tunnel> <tunnel token|""> -> the cloud-init command
# line: the boot script (FVB_* values + VM_BOOT_BODY) base64-encoded, decoded to a
# root-only file and started in the background.
vm_boot() {
  local image="$1" env="$2" tunnel="$3" ttok="$4" envb64 script args
  envb64="$(jq -r 'to_entries[] | "\(.key)=\(.value)"' <<<"$env" | base64 -w0)"
  # In a VM the image entrypoint always stays (docker run <image> <args>).
  [[ "$CONFIG" != *[[:space:]]* ]] || die "FV_SERVE_CONFIG must not contain spaces"
  args="--config $CONFIG"
  script="$(printf 'FVB_IMAGE=%q\nFVB_ARGS=%q\nFVB_ENV_B64=%q\nFVB_TUNNEL=%q\nFVB_TUNNEL_HOSTNAME=%q\nFVB_VOLUME_PATH=%q\n' \
    "$image" "$args" "$envb64" "$tunnel" "$TUNNEL_HOSTNAME" "${VOLUME:+/workspace/weights}")"
  [[ -z "$ttok" ]] || script+=$'\n'"(umask 077; mkdir -p /var/lib/fv; printf '%s' $(printf '%q' "$ttok") >/var/lib/fv/tunnel-token)"
  script+="$VM_BOOT_BODY"
  printf 'umask 077; echo %s | base64 -d >/root/fv-boot.sh && nohup bash /root/fv-boot.sh >/var/log/fv-boot.log 2>&1 &' \
    "$(base64 -w0 <<<"$script")"
}

# payload <image> <variant> <dc> <env json> <kind> <deadline> <service> <driver> <tunnel> <tunnel token|""> <ssh pub|"">
payload() {
  local name tags
  name="fv-serve-$5-$(date -u +%m%d%H%M%S)"
  tags="$(cr_tags "serve-$5" "$6")"
  if [[ "$7" == vm ]]; then
    local img p
    if [[ "$2" == "<variant>" ]]; then img="<recipe image for driver $8>"; else img="$(cr_recipe_image "$8")" || return 1; fi
    p="$(cr_vm_payload "$2" "$3" "$name" "$img" "$(vm_boot "$1" "$4" "$9" "${10}")" "${11}" "$tags")"
    if [[ -n "$VOLUME" ]]; then
      p="$(jq -c --arg v "$VOLUME" '.config.VirtualMachine.volumes = {Mounts: [{volume: {ByName: [$v]}, mount_path: "/workspace/weights"}]}' <<<"$p")"
    fi
    printf '%s\n' "$p"
  else
    cr_docker_payload "$2" "$3" "$name" "$1" "$(command_json)" "$4" \
      "$(jq -nc --arg p "$HOST_PORT" '[($p + ":8000/tcp")]')" "$tags"
  fi
}

validate() {
  local p="$1"
  jq -e '.tags | index("fv-owner:fastvideo-rs")' <<<"$p" >/dev/null || die "the owner tag is missing"
  if jq -e '.config.VirtualMachine' <<<"$p" >/dev/null; then
    jq -e '.config.VirtualMachine.cloudinit_commands | length > 0' <<<"$p" >/dev/null || die "no cloud-init boot"
    bash -n <<<"$VM_BOOT_BODY" || die "the VM boot is not valid bash"
  else
    jq -e '.with_public_ip == true and (.config.Docker.ports | length) >= 1' <<<"$p" >/dev/null || die "no public port"
    jq -e '.config.Docker.env | all(type == "array" and length == 2)' <<<"$p" >/dev/null || die "env must be [name, value] pairs"
    [[ "$MODE" != exec ]] || bash -n <<<"$BOOT" || die "the boot command is not valid bash"
  fi
}

# The payload as plan prints it: secret values masked; the VM boot shown decoded
# with its env and tunnel token masked.
masked() {
  jq --arg re "$SECRET_NAMES" '
    if .config.Docker then
      .config.Docker.env |= map(if (.[0] | test($re)) or .[0] == "FV_INTERNAL_TOKEN" then [.[0], "<secret>"] else . end)
    else .config.VirtualMachine.cloudinit_commands = "<fv-boot: docker run + tunnel; env and tunnel token base64-embedded, not shown>" end' <<<"$1"
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

# rent <image> <env json> <kind> <cap s> <role> [tunnel token]: sets ID, VARIANT, DPH, SVC, TUNNEL.
VARIANT="" DPH="" SVC="" TUNNEL=""
rent() {
  local pick dc driver deadline p pub="" probe
  cr_check_balance
  # The service decides which catalog to read; auto reads the docker one first
  # and switches to vm for a ProprietaryOnly host.
  probe="$SERVICE"; [[ "$probe" != auto ]] || probe=docker
  pick="$(cr_pick "$GPUS" "$MAX_DPH" 1 "$probe")" \
    || { [[ "$SERVICE" == auto ]] && pick="$(cr_pick "$GPUS" "$MAX_DPH" 1 vm)"; } \
    || die "no free 1-GPU stock of [$GPUS] under \$$MAX_DPH/hr"
  read -r VARIANT DPH dc driver <<<"$pick"
  [[ "$dc" != - ]] || dc=""
  SVC="$(service_for "$driver")"
  if [[ "$SVC" == vm ]]; then
    TUNNEL="$(tunnel_for "$5")"
    pub="$(ssh_pub)"
    if [[ "$TUNNEL" == token ]]; then
      [[ -r "$TUNNEL_TOKEN_FILE" && -n "$TUNNEL_HOSTNAME" ]] \
        || die "tunnel token: set FV_CF_TUNNEL_TOKEN_FILE (mode 600) and CLOUDRIFT_TUNNEL_HOSTNAME"
    fi
  elif [[ "$5" == worker && "${CLOUDRIFT_ALLOW_PLAIN_HTTP:-0}" != 1 ]]; then
    die "a docker-mode worker would take the internal token over plain HTTP (owner decision: tunnel or SSH); use CLOUDRIFT_SERVICE=vm"
  fi
  deadline=$(($(date +%s) + $4))
  p="$(payload "$1" "$VARIANT" "$dc" "$2" "$3" "$deadline" "$SVC" "$driver" "$TUNNEL" \
    "$([[ "$TUNNEL" == token ]] && tr -d '[:space:]' <"$TUNNEL_TOKEN_FILE")" "$pub")" || die "could not build the rent payload"
  validate "$p"
  ID="$(cr_rent "$p")" || die "rent failed ($VARIANT, $SVC)"
  cr_ledger "instance-rented $ID kind=serve-$3 service=$SVC variant=$VARIANT dph=$DPH image=$1 cap=$4s${TUNNEL:+ tunnel=$TUNNEL}"
  cr_backstop "$ID" "$4"
  log "instance $ID: $VARIANT${dc:+ in $dc} at \$$DPH/hr, $SVC${TUNNEL:+ + $TUNNEL tunnel} (backstop $4 s)"
}

# vm_url <host> <t0> -> the base URL once the VM booted (quick/token: the tunnel
# URL the VM wrote; ssh: a local forward). Prints the boot log tail on timeout.
vm_url() {
  local host="$1" t0="$2" url lp
  until vm_ssh "$host" true 2>/dev/null; do
    (( $(date +%s) - t0 < BOOT_WAIT_S )) || die "no ssh to $VM_USER@$host after ${BOOT_WAIT_S}s"
    sleep "${CR_POLL_S:-10}"
  done
  log "ssh $VM_USER@$host after $(($(date +%s) - t0)) s"
  until vm_ssh "$host" test -f /var/lib/fv/booted 2>/dev/null; do
    if (( $(date +%s) - t0 >= BOOT_WAIT_S )); then
      vm_ssh "$host" sudo tail -n 40 /var/log/fv-boot.log >&2 || true
      die "the VM boot did not finish after ${BOOT_WAIT_S}s"
    fi
    sleep "${CR_POLL_S:-10}"
  done
  vm_ssh "$host" cat /var/lib/fv/nvidia-smi.txt 2>/dev/null | head -n 12 >&2 || true
  if [[ "$TUNNEL" == ssh ]]; then
    lp="${CLOUDRIFT_LOCAL_PORT:-18000}"
    "$SSH_BIN" -N -i "$SSH_KEY" "${FV_SSH_OPTS[@]}" -L "127.0.0.1:$lp:127.0.0.1:8000" "$VM_USER@$host" &
    SSH_TUNNEL_PID=$!
    echo "http://127.0.0.1:$lp"
    return
  fi
  url="$(vm_ssh "$host" cat /var/lib/fv/public-url 2>/dev/null | tr -d '[:space:]')"
  # (tests point the "tunnel" at the mock's plain-HTTP port)
  [[ "$url" == https://* || "${CLOUDRIFT_TEST_ALLOW_HTTP_URL:-0}" == 1 ]] || { vm_ssh "$host" sudo tail -n 40 /var/log/cloudflared.log >&2 || true; die "no tunnel URL from the VM"; }
  echo "$url"
}

http() { curl -sS --max-time 60 "$@"; }

cmd_smoke() {
  require_tools curl jq openssl sha256sum
  cr_need_key
  local image key keyhash inst host port base t0 t_active t_ready st job id health
  image="${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}"
  [[ "$image" == *@sha256:* ]] || log "warning: the image is not digest-pinned"
  key="fvk-$(openssl rand -hex 16)"
  keyhash="$(printf '%s' "$key" | sha256sum | cut -d' ' -f1)"
  trap cleanup EXIT INT TERM
  t0="$(date +%s)"
  rent "$image" "$(env_json standalone "$keyhash" "" '{}' "")" smoke "${CLOUDRIFT_CAP_S:-1800}" standalone
  inst="$(cr_wait_active "$ID" "$BOOT_WAIT_S")" || die "instance $ID did not become Active"
  t_active="$(date +%s)"
  host="$(jq -r .host_address <<<"$inst")"
  if [[ "$SVC" == vm ]]; then
    base="$(vm_url "$host" "$t0")"
  else
    log "warning: docker mode is plain HTTP (smoke only)"
    port="$(cr_host_port "$inst" 8000)"
    base="http://$host:$port"
  fi
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
  jq -n --arg id "$ID" --arg v "$VARIANT" --argjson dph "$DPH" --arg image "$image" --arg base "$base" --arg svc "$SVC" --arg tun "$TUNNEL" \
    --argjson ta "$((t_active - t0))" --argjson tr "$((t_ready - t0))" --argjson wall "$(($(date +%s) - t0))" --argjson st "$st" \
    --argjson health "$(jq -c '{state}' <<<"$health" 2>/dev/null || echo '{}')" \
    '{target: "cloudrift", service: $svc, tunnel: $tun, instance: $id, variant: $v, usd_per_hr: $dph, image: $image, public_url: $base,
      rent_to_active_s: $ta, rent_to_healthz_s: $tr, active_to_healthz_s: ($tr - $ta), wall_s: $wall,
      est_cost_usd: ($dph * $wall / 3600), health: $health, job: ($st | {id, model, status})}' | tee "$OUT_DIR/smoke-$(date -u +%m%d%H%M%S).json"
  cr_terminate "$ID" && ID=""
  [[ "$(jq -r .status <<<"$st")" == succeeded ]] || die "job ended $(jq -c '{status, error}' <<<"$st")"
}

cmd_up() {
  local pool="${1:?pool id}" image tok sec inst host port url
  require_tools curl jq
  cr_need_key
  image="${2:-${FV_SERVE_IMAGE:-}}"
  [[ "$image" == *@sha256:* ]] || die "pin the image digest (FV_SERVE_IMAGE=ghcr.io/...@sha256:...)"
  [[ -r "$TOKEN_FILE" ]] || die "FV_INTERNAL_TOKEN_FILE: the gateway's internal token (a file, mode 600)"
  tok="$(tr -d '[:space:]' <"$TOKEN_FILE")"
  sec="$(secrets_json "$SECRETS_FILE")"
  [[ "$MODE" != exec ]] || jq -e 'has("FV_D1_DATABASE_ID") and has("FV_CF_API_TOKEN")' <<<"$sec" >/dev/null \
    || die "exec mode registers through D1: the secrets file needs FV_CF_ACCOUNT_ID, FV_CF_API_TOKEN, FV_D1_DATABASE_ID"
  rent "$image" "$(env_json worker "" "$pool" "$sec" "$tok")" "$pool" "${CLOUDRIFT_CAP_S:-3600}" worker
  inst="$(cr_wait_active "$ID" "$BOOT_WAIT_S")" || { cr_terminate "$ID"; die "instance $ID did not become Active"; }
  host="$(jq -r .host_address <<<"$inst")"
  if [[ "$SVC" == vm ]]; then
    if [[ "$TUNNEL" == token ]]; then url="https://$TUNNEL_HOSTNAME"
    elif [[ "$TUNNEL" == quick ]]; then url="$(vm_url "$host" "$(date +%s)")"
    else url="ssh -L <port>:127.0.0.1:8000 $VM_USER@$host"; fi
  else
    port="$(cr_host_port "$inst" 8000)"
    url="http://$host:$port"
  fi
  echo "$ID $url"
  log "static pod pool: add $url to the gateway's FV_POOL_$(tr 'a-z-' 'A-Z_' <<<"$pool")_URLS ($SVC${TUNNEL:+, $TUNNEL tunnel})"
}

case "${1:-}" in
  plan)
    shift
    secj="$(secrets_json "$SECRETS_FILE")"
    role="${FV_PLAN_ROLE:-standalone}"
    svc="$SERVICE"; [[ "$svc" != auto ]] || svc=vm
    tun=""; [[ "$svc" != vm ]] || tun="$(tunnel_for "$role")"
    p="$(payload "${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}" "<variant>" "" "$(env_json "$role" "<sha256 of the run key>" "${FV_PLAN_POOL:-h3-turbo}" "$secj" "<internal token>")" plan "$(($(date +%s) + 1800))" "$svc" "${FV_PLAN_DRIVER:-ProprietaryOnly}" "$tun" "$([[ "$tun" == token ]] && echo '<tunnel token>')" "ssh-ed25519 AAAA<public key> fv-cloudrift")"
    validate "$p"
    echo "# POST $CR_API/api/v1/instances/rent  (version $CR_VERSION, service $svc${tun:+, tunnel $tun}, command mode $MODE)"
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
  *) sed -n '2,58p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
