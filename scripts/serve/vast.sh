#!/usr/bin/env bash
# fv-serve on a Vast instance (docs/serve/design.md §6.2 "Vast instance"; WP-16),
# over the REST form strobe proved (research-deploy §5.1): offers with
# direct_port_count >= 1, runtype ssh_direct, an onstart that launches
# fv-serve, and `env` as a JSON **object** ({"-p 8000:8000": "1", ...}).
#
#   vast.sh plan [image]    print the offer query and the create payload (no API
#                           call, no key needed) and validate their shape
#   vast.sh smoke [image]   rent the cheapest matching offer, wait for /healthz,
#                           check /fv/v1/capabilities, the info envelope and one
#                           small job, print timings, destroy (needs VAST_API_KEY)
#   vast.sh down <id>       destroy an instance
#
# Secrets (FV_CF_*, FV_R2_*, FV_WEBHOOK_ED25519_KEY) are Vast account env vars
# under the same upper-case names; Vast injects them, so they never appear in
# the payload or the onstart text. The payload carries ports, non-secret FV_*
# settings and the SHA-256 of a per-run API key.
#
# Money guards (scripts/gpu/lib.sh discipline): $/hr cap (VAST_MAX_DPH, default
# 0.60), wall-clock cap (FV_VAST_CAP_S, default 1800 s; a detached backstop
# destroys the instance), destroy-on-exit trap, ledger
# (artifacts/vast/serve/ledger.tsv). Without VAST_API_KEY only `plan` runs.
#
# Env: VAST_API_KEY, FV_SERVE_IMAGE (digest-pinned), FV_SERVE_CONFIG (default
# /etc/fv/runpod-fake.toml), VAST_DISK_GB (default 40), FV_SMOKE_MODEL.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/lib.sh
source "$HERE/../gpu/lib.sh"

VAST="${VAST_API_BASE:-https://console.vast.ai/api/v0}"
IMAGE_DEFAULT="ghcr.io/zaitrarrio/fastvideo-rs-serve:latest"
CONFIG="${FV_SERVE_CONFIG:-/etc/fv/runpod-fake.toml}"
MAX_DPH="${VAST_MAX_DPH:-0.60}"
CAP_S="${FV_VAST_CAP_S:-1800}"
DISK="${VAST_DISK_GB:-40}"
MODEL="${FV_SMOKE_MODEL:-fake-wan}"
LEDGER="${FV_SERVE_LEDGER:-$FV_ROOT/artifacts/vast/serve/ledger.tsv}"
MODEL_LOG=/var/log/fv-serve.log

ledger() { mkdir -p "$(dirname "$LEDGER")"; printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }
vast() {
  curl -sS --fail-with-body -X "$1" -H "Authorization: Bearer $VAST_API_KEY" \
    -H 'content-type: application/json' ${3:+-d "$3"} "$VAST$2"
}

# The offer filter: one GPU, CUDA 13 driver, a direct port (else :8000 is
# unreachable: proxied ports only forward ssh/jupyter), verified, cheap first.
offer_query() {
  jq -n --arg max "$MAX_DPH" --arg disk "$DISK" '{
    rentable: {eq: true}, verified: {eq: true}, num_gpus: {eq: 1},
    direct_port_count: {gte: 1}, cuda_max_good: {gte: 13.0},
    reliability2: {gte: 0.95}, disk_space: {gte: ($disk|tonumber)},
    dph_total: {lte: ($max|tonumber)}, type: "on-demand",
    order: [["dph_total", "asc"]], limit: 20
  }'
}

# ssh_direct: sshd is PID 1 and the image ENTRYPOINT never runs, so onstart
# starts fv-serve (same text as fastvideo_deploy::vast::onstart).
onstart() {
  cat <<EOF
#!/bin/bash
env | grep -E '^(FV_|VAST_|PUBLIC_IPADDR|CONTAINER_ID|RUST_LOG)' >> /etc/environment
mkdir -p /var/log /fvstate
nohup /opt/fastvideo-rs/bin/fv-serve --config $CONFIG >> $MODEL_LOG 2>&1 &
EOF
}

# The env object: published ports as keys, then plain settings. 70010/udp and
# 70000/tcp are symmetric requests (VAST_UDP_PORT_70010 / VAST_TCP_PORT_70000
# name the port fv-serve binds and advertises as its ICE candidates).
env_dict() {
  jq -n --arg keys "$1" '{
    "-p 8000:8000": "1", "-p 70010:70010/udp": "1", "-p 70000:70000": "1",
    FV_SERVE_MODE: "http", FV_STATE_DIR: "/fvstate", FV_API_KEYS: $keys,
    FV_SERVE_FORWARD: "1", RUST_LOG: "info"
  }'
}

create_payload() {
  local image="$1" keyhash="$2" label="$3"
  jq -n --arg image "$image" --arg label "$label" --arg disk "$DISK" --arg onstart "$(onstart)" \
    --argjson env "$(env_dict "$keyhash")" '{
    client_id: "me", image: $image, label: $label, disk: ($disk|tonumber),
    runtype: "ssh_direct", target_state: "running", cancel_unavail: true,
    onstart: $onstart, env: $env
  }'
}

# Shape checks the API enforces (or strobe observed it enforcing).
validate() {
  local p="$1"
  jq -e '.env | type == "object"' <<<"$p" >/dev/null || die "env must be a JSON object"
  jq -e '.env | to_entries | map(select(.key | test("^-p "))) | length >= 1' <<<"$p" >/dev/null || die "no published port"
  jq -e '.onstart | length < 4048' <<<"$p" >/dev/null || die "onstart over 4048 chars"
  jq -e '.runtype == "ssh_direct"' <<<"$p" >/dev/null || die "runtype must be ssh_direct"
  if jq -e '.env | keys[] | select(test("^FV_(CF|R2)_|WEBHOOK|SIGNING|WHIP_TOKEN|HF_TOKEN"))' <<<"$p" >/dev/null; then
    die "a secret name is in the env payload"
  fi
  [[ "$(jq -r .image <<<"$p")" == *@sha256:* ]] || log "warning: image is not digest-pinned"
}

ID=""
cleanup() {
  local rc=$?
  if [[ -n "$ID" ]]; then
    log "destroy-on-exit: instance $ID"
    if vast DELETE "/instances/$ID/" >/dev/null 2>&1 || { command -v vastai >/dev/null && vast_destroy "$ID"; }; then
      ledger "instance-destroyed $ID"
    fi
    ID=""
  fi
  exit "$rc"
}

cmd_plan() {
  local image="${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}" p
  echo "# POST $VAST/bundles/  (offer search)"
  offer_query
  p="$(create_payload "$image" "<sha256 of the run key>" "fvgpu-serve-plan")"
  validate "$p"
  echo "# PUT $VAST/asks/<offer id>/  (create)"
  echo "$p"
  log "plan: payload shape OK"
}

cmd_smoke() {
  require_tools curl jq openssl sha256sum
  [[ -n "${VAST_API_KEY:-}" ]] || die "VAST_API_KEY is not set: only 'vast.sh plan' runs without it (no machine is rented)"
  command -v vastai >/dev/null && vast_check_auth
  local image key keyhash offers offer dph resp inst ip port base t0 t_run="" t_ready="" st
  image="${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}"
  [[ "$image" == *@sha256:* ]] || die "pin the image digest (FV_SERVE_IMAGE=ghcr.io/...@sha256:...)"
  key="fvk-$(openssl rand -hex 16)"
  keyhash="$(printf '%s' "$key" | sha256sum | cut -d' ' -f1)"
  offers="$(vast POST /bundles/ "$(offer_query)")"
  offer="$(jq -r '.offers[0].id // empty' <<<"$offers")"
  dph="$(jq -r '.offers[0].dph_total // empty' <<<"$offers")"
  [[ -n "$offer" ]] || die "no offer under \$$MAX_DPH/hr with a direct port and CUDA >= 13"
  trap cleanup EXIT INT TERM
  t0="$(date +%s)"
  resp="$(vast PUT "/asks/$offer/" "$(create_payload "$image" "$keyhash" "fvgpu-serve-$(date -u +%m%d%H%M)")")"
  ID="$(jq -r '.new_contract // empty' <<<"$resp")"
  [[ -n "$ID" ]] || die "create failed: $(head -c 300 <<<"$resp")"
  ledger "instance-created $ID offer=$offer dph=$dph image=$image"
  # shellcheck disable=SC2016 # expanded by the child shell
  nohup env IID="$ID" CAP="$CAP_S" API="$VAST" bash -c \
    'sleep "$CAP"; curl -sS -X DELETE -H "Authorization: Bearer $VAST_API_KEY" "$API/instances/$IID/" >/dev/null' >/dev/null 2>&1 &
  log "instance $ID (offer $offer, \$$dph/hr, cap ${CAP_S}s)"
  while :; do
    inst="$(vast GET "/instances/$ID/" | jq -c '.instances // .')"
    st="$(jq -r '.actual_status // "?"' <<<"$inst")"
    case "$st" in exited | offline | unknown) die "instance $ID is $st (bad host)" ;; esac
    if [[ "$st" == running ]]; then
      [[ -n "$t_run" ]] || t_run="$(date +%s)"
      ip="$(jq -r '.public_ipaddr // empty' <<<"$inst")"
      port="$(jq -r '.ports["8000/tcp"][0].HostPort // empty' <<<"$inst")"
      if [[ -n "$ip" && -n "$port" ]]; then
        base="http://$ip:$port"
        if curl -fsS --max-time 10 "$base/healthz" >/dev/null 2>&1; then t_ready="$(date +%s)"; break; fi
      fi
    fi
    (( $(date +%s) - t0 < ${FV_BOOT_WAIT_S:-1200} )) || die "instance $ID never became healthy (status $st)"
    sleep 10
  done
  curl -fsS -H "Authorization: Bearer $key" "$base/fv/v1/capabilities" | jq -c '[.models[].caps.id]'
  curl -fsS -H 'content-type: application/json' -d '{"kind":"info","nvenc":true}' "$base/fv/v1/forward" \
    | jq -c '{gpu, nvenc_encode_ok, deploy: .deploy}'
  curl -fsS -H 'content-type: application/json' -d "{\"kind\":\"http\",\"path\":\"/fv/v1/jobs\",\"headers\":{\"authorization\":\"Bearer $key\"},\"body\":{\"model\":\"$MODEL\",\"prompt\":\"a fox\",\"seed\":1},\"wait\":true}" \
    "$base/fv/v1/forward" | jq -c '{status, job: .body.status}'
  log "running after $((t_run - t0)) s, healthy after $((t_ready - t0)) s"
}

case "${1:-}" in
  plan) shift; cmd_plan "$@" ;;
  smoke) shift; cmd_smoke "$@" ;;
  down)
    [[ -n "${VAST_API_KEY:-}" ]] || die "VAST_API_KEY is not set"
    vast DELETE "/instances/${2:?instance id}/" >/dev/null && ledger "instance-destroyed $2" && echo "destroyed $2" ;;
  *) sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
