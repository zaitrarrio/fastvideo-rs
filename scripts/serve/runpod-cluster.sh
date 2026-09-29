#!/usr/bin/env bash
# A standing gateway cluster on Runpod for manual testing (docs/serve/gateway.md,
# docs/serve/e2e/cluster.md): one CPU pod running fv-serve as the gateway
# (configs/serve/gateway-pods.toml, `[engine] backend = "remote"`) in front of
# four GPU *pod* pools, one pod each:
#
#   h3-turbo  /etc/fv/runpod.toml         h3-max  /etc/fv/runpod-h3-max.toml
#   ltx       /etc/fv/runpod-ltx.toml     wan     /etc/fv/runpod-wan5b.toml
#
#   runpod-cluster.sh up [image]     create the gateway, then the four workers,
#                                    then give the gateway the worker URLs
#   runpod-cluster.sh wait           until every pool reports a ready worker
#   runpod-cluster.sh status         pods, deadline, pools as the gateway sees them
#   runpod-cluster.sh smoke          one small text-to-video per pool (API key
#                                    minted once with the admin token)
#   runpod-cluster.sh admin-token    print the gateway's admin token (fetched
#                                    sealed on first use, kept in the state file)
#   runpod-cluster.sh extend <min>   move the deadline (restarts the gateway pod
#                                    only: its watchdog holds the deadline)
#   runpod-cluster.sh down           delete every pod of the cluster, check they are gone
#
# Auth: FV_CLUSTER_AUTH (default keys) is the gateway's FV_AUTH_MODE: API
# keys are minted in /console/admin or POST /fv/v1/admin/keys. The admin
# routes (/fv/v1/admin/*, /fv/v1/gateway/pools, key minting) need the admin
# token whatever the auth mode. The gateway makes that token itself on its
# first start and keeps it in /fvstate/admin_token (mode 600, container
# disk: it survives a restart, e.g. `extend`, but not a re-creation); it is
# never in the pod's environment or its log. `up` creates an X25519 key pair
# ($STATE.admin-key.pem, mode 600) and gives the gateway the public half
# (FV_ADMIN_TOKEN_RECIPIENT); the gateway publishes the token sealed to it at
# /fv/v1/admin/token/sealed, and this script opens it with openssl and keeps
# a copy in the state file (`admin-token` prints it). Workers always need
# the internal token (every route but health, /metrics and signed /files).
#
# Backstops (deadline = create + FV_CLUSTER_CAP_S, default 6000 s):
# - pod side: the gateway pod runs a watchdog that deletes the four workers
#   and itself at the deadline, or as soon as the account balance drops
#   below FV_MIN_BALANCE (default 8.25 $). It holds the account API key
#   (FV_BACKSTOP_API_KEY in its env) for that.
# - detached here: one loop (setsid) that deletes every pod of the state
#   file at the deadline the state file holds (extend moves it).
# State: $FV_CLUSTER_STATE (default artifacts/runpod/serve/cluster.json, mode
# 600: it holds the internal token, the admin token once fetched and the
# smoke API key; nothing is printed but by `admin-token`).
# Ledger: artifacts/runpod/serve/ledger.tsv.
#
# Placement: FV_CLUSTER_REGIONS (default "eu us"): eu = volume jg48s6o1w0
# (EUR-IS-1, RTX PRO 6000 96 GB), us = s2k01690bi (US-CA-2, H100/H200). The
# gateway prefers the first region's DC. Guards: start needs a balance of at
# least FV_CLUSTER_MIN_START (default 20 $); a GPU over RUNPOD_GPU_MAX_DPH
# (default 3.6 $/hr) is refused.
# shellcheck disable=SC2016 # jq programs use $vars
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/lib.sh
source "$HERE/../gpu/lib.sh"

REST="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
IMAGE_DEFAULT="ghcr.io/zaitrarrio/fastvideo-rs-serve:latest"
STATE="${FV_CLUSTER_STATE:-$FV_ROOT/artifacts/runpod/serve/cluster.json}"
LEDGER="${FV_SERVE_LEDGER:-$FV_ROOT/artifacts/runpod/serve/ledger.tsv}"
OUT_DIR="$FV_ROOT/artifacts/serve/e2e/cluster"
CAP_S="${FV_CLUSTER_CAP_S:-6000}"
MIN_BALANCE="${FV_MIN_BALANCE:-8.25}"
MIN_START="${FV_CLUSTER_MIN_START:-20}"
MAX_DPH="${RUNPOD_GPU_MAX_DPH:-3.6}"
REGIONS="${FV_CLUSTER_REGIONS:-eu us}"
AUTH_MODE="${FV_CLUSTER_AUTH:-keys}"
ADMIN_KEY="$STATE.admin-key.pem"
CPU_FLAVORS="${FV_GATEWAY_CPU_FLAVORS:-cpu3c cpu5c cpu3g}"
POOLS=(h3-turbo h3-max ltx wan)

region_volume() { case $1 in eu) echo jg48s6o1w0 ;; us) echo s2k01690bi ;; esac; }
region_dc() { case $1 in eu) echo EUR-IS-1 ;; us) echo US-CA-2 ;; esac; }
region_gpus() {
  case $1 in
    eu) echo "NVIDIA RTX PRO 6000 Blackwell Server Edition" ;;
    us) echo "NVIDIA H100 80GB HBM3,NVIDIA H100 NVL,NVIDIA H200" ;;
  esac
}
pool_config() {
  case $1 in
    h3-turbo) echo /etc/fv/runpod.toml ;;
    h3-max) echo /etc/fv/runpod-h3-max.toml ;;
    ltx) echo /etc/fv/runpod-ltx.toml ;;
    wan) echo /etc/fv/runpod-wan5b.toml ;;
  esac
}
pool_env() { echo "FV_POOL_$(tr 'a-z-' 'A-Z_' <<<"$1")_URLS"; }

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

# The gateway pod's start command: the config from the env, the watchdog in
# the background, fv-serve in the foreground.
# shellcheck disable=SC2016 # runs on the pod
GATEWAY_BOOT='set -u
mkdir -p /fvstate
printf "%s" "$FV_GATEWAY_TOML_B64" | base64 -d > /fv-gateway.toml
export FV_PUBLIC_BASE_URL="https://${RUNPOD_POD_ID}-8000.proxy.runpod.net"
(
  api=https://rest.runpod.io/v1
  kill_all() {
    echo "[watchdog] $1: deleting ${FV_CLUSTER_PODS:-} and ${RUNPOD_POD_ID}" >&2
    for p in ${FV_CLUSTER_PODS:-} "$RUNPOD_POD_ID"; do
      for _ in 1 2 3; do
        curl -sS --max-time 30 -X DELETE -H "Authorization: Bearer $FV_BACKSTOP_API_KEY" "$api/pods/$p" >/dev/null && break
        sleep 5
      done
    done
  }
  n=0
  while :; do
    if [ "$(date +%s)" -ge "$FV_CLUSTER_DEADLINE" ]; then kill_all deadline; sleep 60; continue; fi
    if [ $((n % 2)) -eq 0 ]; then
      b=$(curl -sS --max-time 20 -H "Authorization: Bearer $FV_BACKSTOP_API_KEY" -H "content-type: application/json" \
        https://api.runpod.io/graphql -d "{\"query\":\"{ myself { clientBalance } }\"}" | sed -n "s/.*\"clientBalance\":\([0-9.]*\).*/\1/p")
      if [ -n "$b" ] && awk -v b="$b" -v m="$FV_MIN_BALANCE" "BEGIN{exit !(b+0 < m+0)}"; then kill_all "balance $b below $FV_MIN_BALANCE"; fi
    fi
    n=$((n + 1))
    sleep 30
  done
) &
exec /opt/fastvideo-rs/bin/fv-serve --config /fv-gateway.toml'

# A worker's start command: the image config with `[gateway] register = false`
# (its public base URL is the gateway's, so a registration would point at
# the gateway), worker id = pod id.
# shellcheck disable=SC2016 # runs on the pod
WORKER_BOOT='set -u
mkdir -p /fvstate
cp "$FV_WORKER_CONFIG" /fv-worker.toml
if grep -q "^\[gateway\]" /fv-worker.toml; then
  sed -i "/^\[gateway\]/a register = false" /fv-worker.toml
else
  printf "\n[gateway]\nregister = false\n" >> /fv-worker.toml
fi
export FV_WORKER_ID="${RUNPOD_POD_ID}"
exec /opt/fastvideo-rs/bin/fv-serve --config /fv-worker.toml'

rest() {
  curl -sS --fail-with-body --max-time 60 -X "$1" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${3:+-d "$3"} "$REST$2"
}
ledger() { mkdir -p "$(dirname "$LEDGER")"; printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }
balance() {
  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' "$GQL" \
    -d '{"query":"{ myself { clientBalance currentSpendPerHr } }"}' | jq -r '.data.myself | "\(.clientBalance) \(.currentSpendPerHr)"'
}
st() { jq -r "$1 // empty" "$STATE"; }
st_set() { local t; t="$(mktemp "$STATE.XXXX")"; jq "$@" "$STATE" >"$t" && chmod 600 "$t" && mv "$t" "$STATE"; }
utc() { date -u -d "@$1" +%FT%TZ; }

# ghcr tag -> digest reference (public package: anonymous pull token).
resolve_digest() {
  local ref="$1" repo tag tok digest
  if [[ "$ref" == *@sha256:* ]]; then echo "$ref"; return; fi
  repo="${ref#ghcr.io/}"; tag="${repo##*:}"; repo="${repo%:*}"
  tok="$(curl -sS "https://ghcr.io/token?scope=repository:$repo:pull" | jq -r '.token // empty')"
  digest="$(curl -sS -I -H "Authorization: Bearer $tok" \
    -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json' \
    "https://ghcr.io/v2/$repo/manifests/$tag" | tr -d '\r' | awk -F': ' 'tolower($1)=="docker-content-digest"{print $2}')"
  [[ "$digest" == sha256:* ]] || die "could not resolve $ref to a digest"
  echo "ghcr.io/$repo@$digest"
}

# The detached local backstop: deletes every pod of the state file once the
# state file's deadline passes (re-read each minute, so extend moves it).
spawn_local_backstop() {
  # shellcheck disable=SC2016 # expanded by the child shell
  setsid nohup env STATE="$STATE" API="$REST" LEDGER="$LEDGER" bash -c '
    while :; do
      [ -f "$STATE" ] || exit 0
      d=$(jq -r ".deadline // 0" "$STATE")
      [ "$(date +%s)" -ge "$d" ] && break
      sleep 60
    done
    for p in $(jq -r "[.gateway.pod] + [.workers[]?.pod] | .[] | select(. != null)" "$STATE"); do
      curl -sS --max-time 30 -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/pods/$p" >/dev/null \
        && printf "%s\tpod-deleted %s cluster-backstop\n" "$(date -u +%FT%TZ)" "$p" >>"$LEDGER"
    done' >/dev/null 2>&1 < /dev/null &
  st_set --arg pid "$!" '.local_backstop_pid = ($pid|tonumber)'
}

create_gateway() {
  local image="$1" dc="$2" flavor resp pod env payload dcs
  env="$(jq -n --argjson s "$SECRET_ENV_JSON" --arg tok "$(st .internal_token)" --arg sign "$(st .url_signing_key)" \
    --arg recipient "$(st .admin_recipient)" --arg auth "$AUTH_MODE" --arg dl "$(st .deadline)" --arg min "$MIN_BALANCE" \
    --arg key "$RUNPOD_API_KEY" --arg toml "$(base64 -w0 "$FV_ROOT/configs/serve/gateway-pods.toml")" '$s + {
      FV_SERVE_MODE: "http", FV_STATE_DIR: "/fvstate", FV_AUTH_MODE: $auth, FV_ADMIN_TOKEN_RECIPIENT: $recipient,
      FV_INTERNAL_TOKEN: $tok, FV_URL_SIGNING_KEY: $sign, FV_GATEWAY_TOML_B64: $toml,
      FV_CLUSTER_DEADLINE: $dl, FV_MIN_BALANCE: $min, FV_BACKSTOP_API_KEY: $key, RUST_LOG: "info"}')"
  for dcs in "[\"$dc\"]" "null"; do
    for flavor in $CPU_FLAVORS; do
      payload="$(jq -n --arg image "$image" --arg flavor "$flavor" --argjson dcs "$dcs" --arg boot "$GATEWAY_BOOT" \
        --argjson env "$env" --arg name "fv-cluster-gw-$(date -u +%m%d%H%M%S)" '{
          name: $name, imageName: $image, computeType: "CPU", cpuFlavorIds: [$flavor], vcpuCount: 2,
          containerDiskInGb: 20, ports: ["8000/http"],
          dockerEntrypoint: ["bash", "-c"], dockerStartCmd: [$boot], env: $env
        } + (if $dcs then {dataCenterIds: $dcs} else {} end)')"
      if resp="$(rest POST /pods "$payload" 2>&1)" && pod="$(jq -r '.id // empty' <<<"$resp")" && [[ -n "$pod" ]]; then
        st_set --arg pod "$pod" --arg f "$flavor" --arg dph "$(jq -r '.costPerHr // 0' <<<"$resp")" --arg t "$(date +%s)" \
          --arg dc "$(jq -r '.machine.dataCenterId // .dataCenterId // empty' <<<"$resp")" \
          '.gateway = {pod: $pod, cpu: $f, dph: ($dph|tonumber), created: ($t|tonumber), dc: $dc}'
        ledger "pod-created $pod cluster-gateway cpu=$flavor dph=$(st .gateway.dph) backstop=$(utc "$(st .deadline)")"
        log "gateway pod $pod ($flavor, \$$(st .gateway.dph)/hr)"
        return 0
      fi
      log "no gateway on $flavor in ${dcs}: $(jq -r '.error // .message // .' <<<"$resp" 2>/dev/null | head -c 160)"
    done
  done
  die "could not create the gateway pod"
}

# create_worker <pool> <image>: the first region with stock.
create_worker() {
  local pool="$1" image="$2" region vol dc gpu resp pod dph payload
  for region in $REGIONS; do
    vol="$(region_volume "$region")"; dc="$(region_dc "$region")"
    IFS=',' read -r -a gpus <<<"$(region_gpus "$region")"
    for gpu in "${gpus[@]}"; do
      payload="$(jq -n --arg image "$image" --arg gpu "$gpu" --arg vol "$vol" --arg dc "$dc" --arg boot "$WORKER_BOOT" \
        --argjson s "$SECRET_ENV_JSON" --arg tok "$(st .internal_token)" --arg sign "$(st .url_signing_key)" \
        --arg gw "$(st .gateway_url)" --arg cfg "$(pool_config "$pool")" --arg name "fv-cluster-$pool-$(date -u +%m%d%H%M%S)" '{
          name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU", gpuTypeIds: [$gpu], gpuCount: 1,
          containerDiskInGb: 40, volumeInGb: 0, networkVolumeId: $vol, volumeMountPath: "/workspace", dataCenterIds: [$dc],
          ports: ["8000/http", "70000/tcp"], dockerEntrypoint: ["bash", "-c"], dockerStartCmd: [$boot],
          env: ($s + {FV_SERVE_MODE: "http", FV_SERVE_ROLE: "worker", FV_INTERNAL_TOKEN: $tok, FV_URL_SIGNING_KEY: $sign,
            FV_PUBLIC_BASE_URL: $gw, FV_WORKER_CONFIG: $cfg, FV_STATE_DIR: "/fvstate", FV_WEIGHTS: "/workspace/weights",
            FV_JOBS_HEARTBEAT_S: "10", RUST_LOG: "info"})}')"
      if ! resp="$(rest POST /pods "$payload" 2>&1)"; then
        log "$pool: no $gpu in $dc: $(jq -r '.error // .message // .' <<<"$resp" 2>/dev/null | head -c 160)"
        continue
      fi
      pod="$(jq -r '.id // empty' <<<"$resp")"
      [[ -n "$pod" ]] || { log "$pool: create returned no id"; continue; }
      dph="$(jq -r '.costPerHr // 0' <<<"$resp")"
      ledger "pod-created $pod cluster-$pool gpu=$gpu dc=$dc dph=$dph backstop=$(utc "$(st .deadline)")"
      st_set --arg p "$pool" --arg pod "$pod" --arg gpu "$gpu" --arg dc "$dc" --arg dph "$dph" --arg t "$(date +%s)" \
        '.workers[$p] = {pod: $pod, gpu: $gpu, dc: $dc, dph: ($dph|tonumber), created: ($t|tonumber),
                          url: "https://\($pod)-8000.proxy.runpod.net"}'
      if awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 > c+0)}'; then
        log "$pool: $pod costs \$$dph/hr > \$$MAX_DPH: deleting"
        rest DELETE "/pods/$pod" >/dev/null && ledger "pod-deleted $pod over-cap"
        st_set --arg p "$pool" 'del(.workers[$p])'
        continue
      fi
      log "$pool: pod $pod on $gpu in $dc at \$$dph/hr"
      return 0
    done
  done
  log "$pool: no stock in [$REGIONS]"
  return 1
}

# Gives the gateway the worker URLs and the pod list (restarts its container).
patch_gateway() {
  local env
  # The whole env again (secret references, never values), plus the pools.
  env="$(jq -n --argjson s "$SECRET_ENV_JSON" --arg tok "$(st .internal_token)" --arg sign "$(st .url_signing_key)" \
    --arg recipient "$(st .admin_recipient)" --arg auth "$AUTH_MODE" --arg dl "$(st .deadline)" --arg min "$MIN_BALANCE" \
    --arg key "$RUNPOD_API_KEY" --arg toml "$(base64 -w0 "$FV_ROOT/configs/serve/gateway-pods.toml")" \
    --arg pods "$(jq -r '[.workers[]?.pod] | join(" ")' "$STATE")" --argjson urls "$(jq -c '.workers // {}' "$STATE")" '$s + {
      FV_SERVE_MODE: "http", FV_STATE_DIR: "/fvstate", FV_AUTH_MODE: $auth, FV_ADMIN_TOKEN_RECIPIENT: $recipient,
      FV_INTERNAL_TOKEN: $tok, FV_URL_SIGNING_KEY: $sign, FV_GATEWAY_TOML_B64: $toml,
      FV_CLUSTER_DEADLINE: $dl, FV_MIN_BALANCE: $min, FV_BACKSTOP_API_KEY: $key, FV_CLUSTER_PODS: $pods, RUST_LOG: "info"}
      + ($urls | to_entries | map({key: ("FV_POOL_" + (.key | ascii_upcase | gsub("-"; "_")) + "_URLS"), value: .value.url}) | from_entries)')"
  rest PATCH "/pods/$(st .gateway.pod)" "$(jq -nc --argjson e "$env" '{env: $e}')" >/dev/null
  ledger "pod-patched $(st .gateway.pod) cluster-gateway pools=$(jq -r '.workers | keys | join(",")' "$STATE") deadline=$(utc "$(st .deadline)")"
  log "gateway: worker URLs set ($(jq -r '.workers | keys | join(", ")' "$STATE")); container restarts"
}

cmd_up() {
  require_tools curl jq openssl base64 setsid
  : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
  [[ ! -s "$STATE" ]] || die "a cluster state exists ($STATE): run down first"
  local b spend image region dc
  read -r b spend <<<"$(balance)"
  awk -v b="$b" -v m="$MIN_START" 'BEGIN{exit !(b+0 >= m+0)}' || die "balance \$$b is below \$$MIN_START"
  log "balance \$$b, account spend \$$spend/hr"
  image="$(resolve_digest "${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}")"
  log "image $image"
  mkdir -p "$(dirname "$STATE")"
  # The key pair the gateway seals its admin token to (the private half
  # never leaves this machine).
  (umask 077; openssl genpkey -algorithm X25519 -out "$ADMIN_KEY" 2>/dev/null) || die "openssl cannot make an X25519 key"
  (umask 077; jq -n --arg image "$image" --arg tok "$(openssl rand -hex 32)" --arg sign "$(openssl rand -hex 32)" \
    --arg recipient "$(openssl pkey -in "$ADMIN_KEY" -pubout -outform DER | tail -c 32 | base64 -w0)" \
    --arg dl "$(( $(date +%s) + CAP_S ))" --arg auth "$AUTH_MODE" \
    '{image: $image, internal_token: $tok, url_signing_key: $sign, admin_recipient: $recipient, auth: $auth,
      deadline: ($dl|tonumber), workers: {}}' >"$STATE")
  log "deadline $(utc "$(st .deadline)") (${CAP_S}s)"
  region="${REGIONS%% *}"; dc="$(region_dc "$region")"
  create_gateway "$image" "$dc"
  st_set --arg u "https://$(st .gateway.pod)-8000.proxy.runpod.net" '.gateway_url = $u'
  spawn_local_backstop
  local pool failed=()
  for pool in "${POOLS[@]}"; do create_worker "$pool" "$image" || failed+=("$pool"); done
  patch_gateway
  jq -c '{gateway: .gateway, gateway_url, deadline_utc: (.deadline | todate), workers: (.workers | map_values({pod, gpu, dc, dph}))}' "$STATE"
  (( ${#failed[@]} == 0 )) || log "WARNING: no pod for: ${failed[*]}"
}

# Opens the sealed admin token (JSON on stdin) with $ADMIN_KEY; prints it.
# Scheme (crates/fastvideo-serve/src/admin_token.rs): s = X25519(key, epk);
# k = SHA-512("fv-admin-token-v1" | s | epk | our public key); the token is
# AES-256-CTR(k[0..32], iv), authenticated by HMAC-SHA256(k[32..64], iv | ct).
open_sealed() {
  local d rc=0
  d="$(mktemp -d)"
  # A subshell tested by || ignores set -e: every step checks its status.
  (
    hex() { od -An -tx1 | tr -d ' \n'; }
    cat >"$d/sealed.json" || exit 1
    [[ "$(jq -r .alg "$d/sealed.json")" == X25519-SHA512-AES256CTR-HMACSHA256 ]] || exit 1
    for f in epk iv ct tag; do jq -r ".$f" "$d/sealed.json" | base64 -d >"$d/$f" || exit 1; done
    # A raw X25519 public key as DER (SubjectPublicKeyInfo) for openssl.
    { printf '\x30\x2a\x30\x05\x06\x03\x2b\x65\x6e\x03\x21\x00'; cat "$d/epk"; } >"$d/epk.der" || exit 1
    openssl pkey -pubin -inform DER -in "$d/epk.der" -out "$d/epk.pem" 2>/dev/null || exit 1
    openssl pkeyutl -derive -inkey "$ADMIN_KEY" -peerkey "$d/epk.pem" -out "$d/shared" 2>/dev/null || exit 1
    openssl pkey -in "$ADMIN_KEY" -pubout -outform DER | tail -c 32 >"$d/rpk" || exit 1
    { printf 'fv-admin-token-v1'; cat "$d/shared" "$d/epk" "$d/rpk"; } | openssl dgst -sha512 -binary >"$d/k" || exit 1
    ek="$(head -c 32 "$d/k" | hex)"
    mk="$(tail -c 32 "$d/k" | hex)"
    tag="$(cat "$d/iv" "$d/ct" | openssl dgst -sha256 -mac HMAC -macopt "hexkey:$mk" -binary | hex)"
    [[ ${#ek} == 64 && ${#tag} == 64 && "$tag" == "$(hex <"$d/tag")" ]] || exit 1
    openssl enc -d -aes-256-ctr -K "$ek" -iv "$(hex <"$d/iv")" -in "$d/ct" || exit 1
  ) || rc=$?
  rm -rf "$d"
  return "$rc"
}

# The admin token: the state file's copy while the gateway accepts it, else
# fetched sealed from the gateway (a re-created gateway pod makes a new one).
admin_token() {
  local tok sealed code
  tok="$(st .admin_token)"
  if [[ -n "$tok" ]]; then
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 -H @<(printf 'Authorization: Bearer %s\n' "$tok") \
      "$(st .gateway_url)/fv/v1/gateway/pools" 2>/dev/null || true)"
    [[ "$code" == 401 ]] || { printf '%s' "$tok"; return 0; }
  fi
  [[ -s "$ADMIN_KEY" ]] || die "no admin key pair ($ADMIN_KEY): read /fvstate/admin_token on the gateway pod instead"
  sealed="$(curl -sS --fail --max-time 20 "$(st .gateway_url)/fv/v1/admin/token/sealed")" \
    || die "the gateway did not publish its sealed admin token (not up yet?)"
  tok="$(open_sealed <<<"$sealed")" || die "could not open the sealed admin token"
  [[ "$tok" == fvadm_* ]] || die "the sealed admin token does not look like one"
  st_set --arg t "$tok" '.admin_token = $t'
  printf '%s' "$tok"
}

# An admin-token call; the token goes in through a header file, never argv.
admin_call() {
  local method="$1" path="$2" tok
  shift 2
  tok="$(admin_token)"
  curl -sS --max-time 30 -X "$method" -H @<(printf 'Authorization: Bearer %s\n' "$tok") "$@" "$(st .gateway_url)$path"
}
admin_get() { admin_call GET "$1"; }

# The API key the smoke test uses (minted once, kept in the state file).
api_key() {
  local k
  k="$(st .smoke_api_key)"
  if [[ -z "$k" ]]; then
    k="$(admin_call POST /fv/v1/admin/keys -H 'content-type: application/json' -d '{"name":"cluster-smoke"}' | jq -r '.api_key // empty')"
    [[ -n "$k" ]] || die "minting an API key failed"
    st_set --arg k "$k" '.smoke_api_key = $k'
  fi
  printf '%s' "$k"
}

cmd_admin_token() {
  admin_token
  echo
}

cmd_status() {
  local p
  echo "deadline $(utc "$(st .deadline)") ($(( $(st .deadline) - $(date +%s) ))s left); gateway $(st .gateway_url)"
  for p in $(jq -r '[.gateway.pod] + [.workers[].pod] | .[]' "$STATE"); do
    rest GET "/pods/$p" 2>/dev/null | jq -r '[.id, .name, .desiredStatus, .costPerHr, (.machine.gpuDisplayName // .gpu.displayName // "cpu")] | @tsv' \
      || echo "$p gone"
  done
  admin_get /fv/v1/gateway/pools | jq -c '.pools[]? | {pool, available, queued, running, workers}' 2>/dev/null || true
}

cmd_wait() {
  local t0 view pools n
  t0=$(date +%s)
  while :; do
    # The admin view (worker details are not on the public routes).
    view="$( (admin_get /fv/v1/gateway/pools) 2>/dev/null || true)"
    pools="$(jq -c '[.state[]? | select(.available) | .id]' <<<"$view" 2>/dev/null || echo '[]')"
    n="$(jq '[.state[]? | select([.workers[]? | select(.ready)] | length > 0)] | length' <<<"$view" 2>/dev/null || echo 0)"
    log "available $pools; pools with a ready worker: $n/4 ($(( $(date +%s) - t0 ))s)"
    (( n >= 4 )) && return 0
    (( $(date +%s) - t0 < ${FV_WAIT_S:-1800} )) || die "not every pool became ready"
    sleep 20
  done
}

# One small text-to-video per pool through the native API (Bearer API key;
# none needed when the cluster runs FV_CLUSTER_AUTH=none).
cmd_smoke() {
  local gw pool model body id t0 st_json res='[]' size key=""
  gw="$(st .gateway_url)"
  mkdir -p "$OUT_DIR"
  [[ "$(st .auth)" == none ]] || key="$(api_key)"
  auth_hdr() { if [[ -n "$key" ]]; then printf 'Authorization: Bearer %s\n' "$key"; else printf 'X-No-Auth: 1\n'; fi; }
  for pool in "${POOLS[@]}"; do
    case $pool in
      h3-turbo) model=fasth3 ;; h3-max) model=sol-h3 ;; ltx) model=ltx25-distill-sol ;; wan) model=fastwan22-ti2v-5b ;;
    esac
    # The smallest short-edge tier the model advertises, 16:9, default length.
    size="$(curl -sS --max-time 20 -H @<(auth_hdr) "$gw/fv/v1/capabilities" | jq -c --arg m "$model" \
      '[.models[] | select(.caps.id == $m) | .caps.canvas.short_edges[]?] | min')"
    body="$(jq -nc --arg m "$model" --argjson se "$size" '{model: $m, prompt: "a red fox trotting through fresh snow, cinematic", seed: 1}
      + (if $se then {aspect_ratio: "16:9", short_edge: $se} else {} end)')"
    t0=$(date +%s.%N)
    id="$(curl -sS --max-time 60 -H @<(auth_hdr) -H 'content-type: application/json' -d "$body" "$gw/fv/v1/jobs" | jq -r '.id // empty')"
    [[ -n "$id" ]] || { log "$pool: submit refused"; res="$(jq -c --arg p "$pool" '. + [{pool: $p, status: "refused"}]' <<<"$res")"; continue; }
    while :; do
      st_json="$(curl -sS --max-time 30 -H @<(auth_hdr) "$gw/fv/v1/jobs/$id" || echo '{}')"
      case "$(jq -r '.status // ""' <<<"$st_json")" in succeeded | failed | cancelled) break ;; esac
      (( ${t0%.*} + 1200 > $(date +%s) )) || break
      sleep 1
    done
    res="$(jq -c --arg p "$pool" --arg m "$model" --argjson b "$body" --arg w "$(awk -v a="$(date +%s.%N)" -v b="$t0" 'BEGIN{printf "%.1f", a-b}')" \
      --argjson s "$st_json" '. + [{pool: $p, model: $m, request: ($b | del(.prompt)), wall_s: ($w|tonumber), status: $s.status,
        error: $s.error, worker: $s.worker, timings: ($s.timings // $s.metrics // null)}]' <<<"$res")"
    log "$pool: $(jq -c '.[-1] | {status, wall_s, error}' <<<"$res")"
  done
  jq . <<<"$res" | tee "$OUT_DIR/smoke-$(date -u +%m%d%H%M%S).json"
}

cmd_extend() {
  local min="${1:?minutes}" b spend dph new
  read -r b spend <<<"$(balance)"
  new=$(( $(st .deadline) + min * 60 ))
  dph="$(jq '[.gateway.dph] + [.workers[].dph] | add' "$STATE")"
  log "balance \$$b, account spend \$$spend/hr (cluster \$$dph/hr); new deadline $(utc "$new")"
  awk -v b="$b" -v s="$spend" -v h="$(( new - $(date +%s) ))" -v m="$MIN_BALANCE" 'BEGIN{exit !(b - s*h/3600 >= m)}' \
    || die "at \$$spend/hr the balance would fall below \$$MIN_BALANCE before $(utc "$new")"
  st_set --arg d "$new" '.deadline = ($d|tonumber)'
  patch_gateway
  ledger "cluster-extended deadline=$(utc "$new")"
}

cmd_down() {
  local p left=""
  for p in $(jq -r '[.workers[]?.pod] + [.gateway.pod] | .[] | select(. != null)' "$STATE"); do
    rest DELETE "/pods/$p" >/dev/null 2>&1 && ledger "pod-deleted $p cluster-down" && log "deleted $p"
  done
  sleep 5
  for p in $(jq -r '[.workers[]?.pod] + [.gateway.pod] | .[] | select(. != null)' "$STATE"); do
    rest GET "/pods/$p" >/dev/null 2>&1 && left+=" $p"
  done
  [[ -z "$left" ]] || die "still present:$left"
  local stamp
  stamp="$(date -u +%m%d%H%M%S)"
  mv "$STATE" "$STATE.down-$stamp"
  [[ ! -f "$ADMIN_KEY" ]] || mv "$ADMIN_KEY" "$ADMIN_KEY.down-$stamp"
  log "cluster deleted"
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  wait | status | smoke | down)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"; [[ -s "$STATE" ]] || die "no cluster state ($STATE)"; "cmd_$1" ;;
  admin-token) [[ -s "$STATE" ]] || die "no cluster state ($STATE)"; cmd_admin_token ;;
  extend) : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"; [[ -s "$STATE" ]] || die "no cluster state"; shift; cmd_extend "$@" ;;
  *) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
