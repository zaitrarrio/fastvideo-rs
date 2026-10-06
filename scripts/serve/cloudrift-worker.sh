#!/usr/bin/env bash
# fv-serve on a CloudRift Docker rental (docs/ops/cloudrift.md "Deployment"):
# a standalone smoke, or a worker for a gateway's pod pool.
#
#   cloudrift-worker.sh plan [image]        print the rent payload; secrets masked (no API call)
#   cloudrift-worker.sh smoke [image]       standalone fv-serve (fake engine): rent, wait for
#                                           /healthz on the public port, capabilities, one job,
#                                           timings, terminate
#   cloudrift-worker.sh up <pool> [image]   a worker (FV_SERVE_ROLE=worker) for pool <pool>;
#                                           prints the id and the URL to give the gateway
#   cloudrift-worker.sh down <id>           terminate
#
# Reaching the gateway (pod pool kind, docs/serve/gateway.md §5.3):
#   CLOUDRIFT_CMD_MODE=args (default)  `command` is fv-serve's arguments only (Docker
#       CMD semantics: the image ENTRYPOINT, fv-serve or fv-entry, stays). The worker
#       does not self-register; add the printed URL to the gateway's
#       FV_POOL_<POOL>_URLS (static pod pool, as runpod-cluster.sh does).
#   CLOUDRIFT_CMD_MODE=exec  `command` runs a shell (only if CloudRift's command replaces
#       the entrypoint; UNVERIFIED): the boot finds the public IP, sets
#       FV_PUBLIC_BASE_URL=http://<ip>:<port> and the worker registers itself in
#       gw_workers under FV_GATEWAY_POOL (needs the D1 secrets).
#
# Secrets: CloudRift has no secret store, so the values of FV_CLOUDRIFT_SECRETS_FILE
# (KEY=VALUE lines: FV_CF_*, FV_D1_DATABASE_ID, FV_R2_*, FV_WEBHOOK_ED25519_KEY,
# FV_URL_SIGNING_KEY) and of FV_INTERNAL_TOKEN_FILE go into the rental's env, which
# CloudRift stores with the rental. Use scoped, revocable tokens. They are never
# printed (plan masks them). Plain HTTP: the internal token crosses the internet in
# clear unless the gateway reaches the worker through a tunnel (docs).
#
# Money guards: the balance floor (CLOUDRIFT_MIN_BALANCE, default 8 $), the $/hr cap
# (CLOUDRIFT_MAX_DPH, default 1.0) on the catalog price, a detached backstop
# (CLOUDRIFT_CAP_S: smoke 1800 s, up 3600 s), the fv-deadline tag fv-control enforces,
# terminate-on-exit (smoke), and the ledger artifacts/cloudrift/ledger.tsv.
#
# Env: CLOUDRIFT_API_KEY (or /root/.config/fv/cloudrift_api_key); FV_SERVE_IMAGE;
# FV_SERVE_CONFIG (default /etc/fv/runpod-fake.toml); CLOUDRIFT_GPUS (default
# "RTX 4090,RTX 5090,RTX PRO 6000"); CLOUDRIFT_HOST_PORT (default 8000); FV_SMOKE_MODEL.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/cloudrift-lib.sh
source "$HERE/../gpu/cloudrift-lib.sh"

IMAGE_DEFAULT="ghcr.io/zaitrarrio/fastvideo-rs-serve:latest"
CONFIG="${FV_SERVE_CONFIG:-/etc/fv/runpod-fake.toml}"
GPUS="${CLOUDRIFT_GPUS:-RTX 4090,RTX 5090,RTX PRO 6000}"
MAX_DPH="${CLOUDRIFT_MAX_DPH:-1.0}"
MODE="${CLOUDRIFT_CMD_MODE:-args}"
HOST_PORT="${CLOUDRIFT_HOST_PORT:-8000}"
MODEL="${FV_SMOKE_MODEL:-fake-wan}"
BOOT_WAIT_S="${CLOUDRIFT_BOOT_WAIT_S:-1200}"
SECRETS_FILE="${FV_CLOUDRIFT_SECRETS_FILE:-}"
TOKEN_FILE="${FV_INTERNAL_TOKEN_FILE:-}"
OUT_DIR="${CLOUDRIFT_OUT_DIR:-$FV_ROOT/artifacts/cloudrift}/serve"
SECRET_NAMES='^(FV_CF_ACCOUNT_ID|FV_CF_API_TOKEN|FV_D1_DATABASE_ID|FV_R2_BUCKET|FV_R2_ENDPOINT|FV_R2_ACCESS_KEY_ID|FV_R2_SECRET_ACCESS_KEY|FV_WEBHOOK_ED25519_KEY|FV_URL_SIGNING_KEY)$'

# exec mode: the public URL first, then fv-serve through the image entrypoint.
# shellcheck disable=SC2016 # expanded inside the container
BOOT='set -u
ip=""
for u in https://api.ipify.org https://ifconfig.me/ip; do
  ip=$(curl -fsS --max-time 10 "$u" 2>/dev/null) && [ -n "$ip" ] && break
done
[ -n "$ip" ] && export FV_PUBLIC_BASE_URL="http://$ip:$FV_HOST_PORT"
mkdir -p /fvstate
entry=/opt/fastvideo-rs/bin/fv-entry; [ -x "$entry" ] || entry=/opt/fastvideo-rs/bin/fv-serve
exec "$entry" --config "$FV_SERVE_CONFIG"'

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

# payload <image> <variant> <dc> <env json> <kind> <deadline>
payload() {
  cr_docker_payload "$2" "$3" "fv-serve-$5-$(date -u +%m%d%H%M%S)" "$1" "$(command_json)" "$4" \
    "$(jq -nc --arg p "$HOST_PORT" '[($p + ":8000/tcp")]')" "$(cr_tags "serve-$5" "$6")"
}

validate() {
  local p="$1"
  jq -e '.with_public_ip == true and (.config.Docker.ports | length) >= 1' <<<"$p" >/dev/null || die "no public port"
  jq -e '.config.Docker.env | all(type == "array" and length == 2)' <<<"$p" >/dev/null || die "env must be [name, value] pairs"
  jq -e '.tags | index("fv-owner:fastvideo-rs")' <<<"$p" >/dev/null || die "the owner tag is missing"
  [[ "$MODE" != exec ]] || bash -n <<<"$BOOT" || die "the boot command is not valid bash"
}

# The payload as plan prints it: secret values masked.
masked() {
  jq --arg re "$SECRET_NAMES" '.config.Docker.env |= map(if (.[0] | test($re)) or .[0] == "FV_INTERNAL_TOKEN" then [.[0], "<secret>"] else . end)' <<<"$1"
}

ID=""
cleanup() {
  local rc=$?
  if [[ -n "$ID" ]]; then
    log "terminate-on-exit: $ID"
    cr_terminate "$ID" || log "WARNING: the backstop will retry $ID"
    ID=""
  fi
  exit "$rc"
}

# rent <image> <env json> <kind> <cap s>: sets ID, VARIANT, DPH.
VARIANT="" DPH=""
rent() {
  local pick dc deadline p
  cr_check_balance
  pick="$(cr_pick "$GPUS" "$MAX_DPH")" || die "no free 1-GPU stock of [$GPUS] under \$$MAX_DPH/hr"
  read -r VARIANT DPH dc <<<"$pick"
  deadline=$(($(date +%s) + $4))
  p="$(payload "$1" "$VARIANT" "${dc:-}" "$2" "$3" "$deadline")"
  validate "$p"
  ID="$(cr_rent "$p")" || die "rent failed ($VARIANT)"
  cr_ledger "instance-rented $ID kind=serve-$3 variant=$VARIANT dph=$DPH image=$1 cap=$4s"
  cr_backstop "$ID" "$4"
  log "instance $ID: $VARIANT at \$$DPH/hr (backstop $4 s)"
}

http() { curl -sS --max-time 60 "$@"; }

cmd_smoke() {
  require_tools curl jq openssl sha256sum
  cr_need_key
  local image key keyhash inst host port base t0 t_active t_ready st job id
  image="${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}"
  [[ "$image" == *@sha256:* ]] || log "warning: the image is not digest-pinned"
  key="fvk-$(openssl rand -hex 16)"
  keyhash="$(printf '%s' "$key" | sha256sum | cut -d' ' -f1)"
  trap cleanup EXIT INT TERM
  t0="$(date +%s)"
  rent "$image" "$(env_json standalone "$keyhash" "" '{}' "")" smoke "${CLOUDRIFT_CAP_S:-1800}"
  inst="$(cr_wait_active "$ID" "$BOOT_WAIT_S")" || die "instance $ID did not become Active"
  t_active="$(date +%s)"
  host="$(jq -r .host_address <<<"$inst")"
  port="$(cr_host_port "$inst" 8000)"
  base="http://$host:$port"
  until curl -fsS --max-time 10 "$base/healthz" >/dev/null 2>&1; do
    (( $(date +%s) - t0 < BOOT_WAIT_S )) || die "$base/healthz never answered"
    sleep "${CR_POLL_S:-10}"
  done
  t_ready="$(date +%s)"
  log "Active after $((t_active - t0)) s, /healthz after $((t_ready - t0)) s ($base)"
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
  fi
  mkdir -p "$OUT_DIR"
  jq -n --arg id "$ID" --arg v "$VARIANT" --argjson dph "$DPH" --arg image "$image" --arg base "$base" \
    --argjson ta "$((t_active - t0))" --argjson tr "$((t_ready - t0))" --argjson wall "$(($(date +%s) - t0))" --argjson st "$st" \
    '{target: "cloudrift", instance: $id, variant: $v, usd_per_hr: $dph, image: $image, public_url: $base,
      rent_to_active_s: $ta, rent_to_healthz_s: $tr, active_to_healthz_s: ($tr - $ta), wall_s: $wall,
      est_cost_usd: ($dph * $wall / 3600), job: ($st | {id, model, status})}' | tee "$OUT_DIR/smoke-$(date -u +%m%d%H%M%S).json"
  cr_terminate "$ID" && ID=""
  [[ "$(jq -r .status <<<"$st")" == succeeded ]] || die "job ended $(jq -c '{status, error}' <<<"$st")"
}

cmd_up() {
  local pool="${1:?pool id}" image tok sec inst host port
  require_tools curl jq
  cr_need_key
  image="${2:-${FV_SERVE_IMAGE:-}}"
  [[ "$image" == *@sha256:* ]] || die "pin the image digest (FV_SERVE_IMAGE=ghcr.io/...@sha256:...)"
  [[ -r "$TOKEN_FILE" ]] || die "FV_INTERNAL_TOKEN_FILE: the gateway's internal token (a file, mode 600)"
  tok="$(tr -d '[:space:]' <"$TOKEN_FILE")"
  sec="$(secrets_json "$SECRETS_FILE")"
  [[ "$MODE" != exec ]] || jq -e 'has("FV_D1_DATABASE_ID") and has("FV_CF_API_TOKEN")' <<<"$sec" >/dev/null \
    || die "exec mode registers through D1: the secrets file needs FV_CF_ACCOUNT_ID, FV_CF_API_TOKEN, FV_D1_DATABASE_ID"
  rent "$image" "$(env_json worker "" "$pool" "$sec" "$tok")" "$pool" "${CLOUDRIFT_CAP_S:-3600}"
  inst="$(cr_wait_active "$ID" "$BOOT_WAIT_S")" || { cr_terminate "$ID"; die "instance $ID did not become Active"; }
  host="$(jq -r .host_address <<<"$inst")"
  port="$(cr_host_port "$inst" 8000)"
  echo "$ID http://$host:$port"
  log "static pod pool: add http://$host:$port to the gateway's FV_POOL_$(tr 'a-z-' 'A-Z_' <<<"$pool")_URLS${MODE:+ (mode $MODE)}"
}

case "${1:-}" in
  plan)
    shift
    secj="$(secrets_json "$SECRETS_FILE")"
    p="$(payload "${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}" "<variant>" "" "$(env_json "${FV_PLAN_ROLE:-standalone}" "<sha256 of the run key>" "${FV_PLAN_POOL:-h3-turbo}" "$secj" "<internal token>")" plan "$(($(date +%s) + 1800))")"
    validate "$p"
    echo "# POST $CR_API/api/v1/instances/rent  (version $CR_VERSION, command mode $MODE)"
    masked "$p"
    log "plan: payload shape OK" ;;
  smoke) shift; cmd_smoke "$@" ;;
  up) shift; cmd_up "$@" ;;
  down) cr_need_key; cr_terminate "${2:?instance id}" && echo "terminated $2" ;;
  *) sed -n '2,37p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
