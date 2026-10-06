#!/usr/bin/env bash
# CloudRift helpers shared by scripts/gpu/cloudrift.sh (GPU checks) and
# scripts/serve/cloudrift-worker.sh (fv-serve workers). Source, don't execute.
# docs/ops/cloudrift.md has the API notes and what is UNVERIFIED.
#
# API (https://api.cloudrift.ai/swagger-ui/, spec /api-docs/openapi.json,
# rift-server 0.62.1 on 2026-10-06): every call is a POST of
# {"version": "<date>", "data": {...}} to /api/v1/<path>; the answer is
# {"version", "data"}. Auth: the X-API-Key header. Money is in cents
# everywhere, observed live on 2026-10-06: catalog prices,
# resource_info.cost_per_hour and the account balance (account/info answered
# 2000 for a $20 top-up although the spec says "Balance in USD").
#
# The key comes from CLOUDRIFT_API_KEY or the file CLOUDRIFT_API_KEY_FILE
# (default /root/.config/fv/cloudrift_api_key). It only ever reaches curl as
# a header file descriptor (-H @<(...), as fv-control.sh does): never argv,
# never a log.
# shellcheck shell=bash disable=SC2034
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CR_API="${CLOUDRIFT_API_BASE:-https://api.cloudrift.ai}"
# v0.62.0 (2026-09-09): instances/rent accepts the v062 protocol only;
# 2026-09-08 is the date that maps to it (cloudrift-ai/emmy#809). list and
# terminate accept it too.
CR_VERSION="${CLOUDRIFT_API_VERSION:-2026-09-08}"
CR_KEY_FILE="${CLOUDRIFT_API_KEY_FILE:-/root/.config/fv/cloudrift_api_key}"
CR_MIN_BALANCE="${CLOUDRIFT_MIN_BALANCE:-8}"
CR_LEDGER="${CLOUDRIFT_LEDGER:-$FV_ROOT/artifacts/cloudrift/ledger.tsv}"
CR_OWNER_TAG="fv-owner:fastvideo-rs"

cr_ledger() { mkdir -p "$(dirname "$CR_LEDGER")"; printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$CR_LEDGER"; }

# Loads the key into CLOUDRIFT_API_KEY (exported for the detached backstop).
# Returns 1 when there is none; prints nothing about the key itself.
cr_load_key() {
  if [[ -z "${CLOUDRIFT_API_KEY:-}" && -r "$CR_KEY_FILE" ]]; then
    local perms
    perms="$(stat -c '%a' "$CR_KEY_FILE" 2>/dev/null || stat -f '%Lp' "$CR_KEY_FILE" 2>/dev/null || echo 600)"
    [[ "${perms: -2}" == "00" ]] || log "warning: $CR_KEY_FILE is readable by other users (mode $perms); chmod 600 it"
    CLOUDRIFT_API_KEY="$(tr -d '[:space:]' <"$CR_KEY_FILE")"
  fi
  [[ -n "${CLOUDRIFT_API_KEY:-}" ]] || return 1
  export CLOUDRIFT_API_KEY
}
cr_need_key() {
  cr_load_key || die "no CloudRift API key: set CLOUDRIFT_API_KEY or write it to $CR_KEY_FILE (mode 600)"
}

# Removes the key from text that goes to a log.
cr_scrub() {
  local s="$1"
  if [[ -n "${CLOUDRIFT_API_KEY:-}" ]]; then s="${s//"$CLOUDRIFT_API_KEY"/[redacted]}"; fi
  printf '%s' "$s"
}

# cr_post <path> <data json> [public] -> the answer's .data on stdout.
# A non-2xx answer logs the status and the first 300 characters and returns 1.
cr_post() {
  local path="$1" data="$2" public="${3:-}" out code body
  out="$(mktemp)"
  body="$(jq -nc --arg v "$CR_VERSION" --argjson d "$data" '{version: $v, data: $d}')"
  if [[ -n "$public" ]]; then
    code="$(curl -sS --max-time 60 -o "$out" -w '%{http_code}' -X POST -H 'content-type: application/json' \
      --data-binary @- "$CR_API/api/v1/$path" <<<"$body")" || code=000
  else
    [[ -n "${CLOUDRIFT_API_KEY:-}" ]] || { rm -f "$out"; log "cr_post $path: no API key loaded"; return 1; }
    code="$(curl -sS --max-time 60 -o "$out" -w '%{http_code}' -X POST -H 'content-type: application/json' \
      -H @<(printf 'X-API-Key: %s\n' "$CLOUDRIFT_API_KEY") \
      --data-binary @- "$CR_API/api/v1/$path" <<<"$body")" || code=000
  fi
  if [[ "$code" != 2?? ]]; then
    log "CloudRift $path: HTTP $code $(cr_scrub "$(head -c 300 "$out")")"
    rm -f "$out"
    return 1
  fi
  jq -c '.data // .' "$out"
  rm -f "$out"
}

# The balance in dollars. account/info answers cents (see the header).
cr_balance() { cr_post account/info '{}' | jq -r 'if .balance == null then empty else (.balance / 100) end'; }

# Refuses (exit 2) when the balance is unknown or below the floor.
cr_check_balance() {
  local b
  b="$(cr_balance)" || die "could not read the CloudRift balance"
  [[ -n "$b" ]] || die "the CloudRift balance answer had no balance"
  awk -v b="$b" -v m="$CR_MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+0)}' \
    || die "CloudRift balance \$$b is below the floor \$$CR_MIN_BALANCE: not renting"
  log "CloudRift balance \$$b (floor \$$CR_MIN_BALANCE)"
}

# The catalog (no key needed): one line per variant,
# "<variant>\t<brand>\t<gpus>\t<vram GB>\t<$/hr>\t<available nodes>\t<datacenters>\t<driver>".
# <driver> is the type's nvidia_kernel_module_support (e.g. ProprietaryOnly on
# the V100 hosts; "-" when absent): it decides the VM recipe (cr_recipe_image).
cr_catalog() {
  local svc="${1:-docker}" sel
  sel="$(jq -nc --arg s "$svc" '{selector: {ByServiceAndLocation: {services: [$s]}}}')"
  cr_post instance-types/list "$sel" public | jq -r '
    .instance_types[] | . as $t | .variants[]
    | [.name, ($t.brand_short // "?"), (.gpu_count // 0), ((.vram // 0) / 1073741824 | floor),
       ((.cost_per_hour * 100 | round) / 10000), (.available_nodes // 0),
       ([.available_nodes_per_dc // {} | to_entries[] | select(.value > 0) | .key] | join(",")),
       ($t.nvidia_kernel_module_support // "-")]
    | @tsv'
}

# cr_pick <comma list of brands, preferred first> <max $/hr> [gpu count] [service]
# -> "<variant> <$/hr> <datacenter> <driver>" of the first brand with free
# stock under the cap (cheapest variant of that brand; datacenter "-" when the
# listing names none). Returns 1 when none. service: docker (default) or vm.
cr_pick() {
  local brands="$1" cap="$2" gpus="${3:-1}" svc="${4:-docker}" cat b line
  local -a order
  cat="$(cr_catalog "$svc")" || return 1
  IFS=',' read -r -a order <<<"$brands"
  for b in "${order[@]}"; do
    line="$(awk -F'\t' -v b="$b" -v cap="$cap" -v g="$gpus" \
      '$2 == b && $3 == g && $6 > 0 && $5 + 0 <= cap + 0 { print $5 "\t" $1 "\t" $7 "\t" $8 }' <<<"$cat" \
      | sort -n | head -1)"
    if [[ -n "$line" ]]; then
      awk -F'\t' '{ split($3, dcs, ","); print $2, $1, (dcs[1] == "" ? "-" : dcs[1]), ($4 == "" ? "-" : $4) }' <<<"$line"
      return 0
    fi
  done
  return 1
}

# cr_instance <id> -> the instance JSON (connection and usage fields), empty when gone.
cr_instance() {
  cr_post instances/list "$(jq -nc --arg id "$1" '{selector: {ById: [$id]}, mask: {with_connection_info: true, with_usage_info: true}}')" \
    | jq -c '.instances[0] // empty'
}

# The host port CloudRift mapped to a container port: port_mappings is a
# list of [container port, host port] pairs (the order dstack's client reads;
# the CloudRift docs do not state it). Falls back to $2.
cr_host_port() {
  local inst="$1" want="$2"
  jq -r --argjson p "$want" '[.port_mappings[]? | select(.[0] == $p) | .[1]][0] // empty' <<<"$inst" | grep . || echo "$want"
}

# cr_terminate <id>: three tries; an instance that is already gone counts as done.
cr_terminate() {
  local id="$1" i st
  [[ -n "$id" ]] || return 0
  for i in 1 2 3; do
    if cr_post instances/terminate "$(jq -nc --arg id "$id" '{selector: {ById: [$id]}}')" >/dev/null; then
      cr_ledger "instance-terminated $id"
      return 0
    fi
    st="$(cr_instance "$id" 2>/dev/null | jq -r '.status // empty' || true)"
    if [[ -z "$st" || "$st" == Inactive ]]; then cr_ledger "instance-gone $id"; return 0; fi
    sleep $((i * 5))
  done
  log "WARNING: could not confirm termination of $id: check the CloudRift console"
  return 1
}

# Detached wall-clock backstop: terminates the instance after <cap s> even if
# this shell dies. The key reaches it through the environment only.
cr_backstop() {
  local id="$1" cap="$2" body
  body="$(jq -nc --arg v "$CR_VERSION" --arg id "$id" '{version: $v, data: {selector: {ById: [$id]}}}')"
  # shellcheck disable=SC2016 # expanded by the child shell
  nohup env CR_ID="$id" CR_CAP="$cap" CR_URL="$CR_API/api/v1/instances/terminate" CR_BODY="$body" bash -c '
    sleep "$CR_CAP"
    for _ in 1 2 3; do
      curl -sS --max-time 30 -X POST -H "content-type: application/json" \
        -H @<(printf "X-API-Key: %s\n" "$CLOUDRIFT_API_KEY") -d "$CR_BODY" "$CR_URL" >/dev/null && break
      sleep 10
    done' >/dev/null 2>&1 &
  disown 2>/dev/null || true
}

# Tags every rental carries: ours (CLAUDE.md: only touch what you created),
# its kind and the deadline fv-control enforces (collector) besides the
# local backstop.
cr_tags() {
  local kind="$1" deadline="$2"
  jq -nc --arg o "$CR_OWNER_TAG" --arg k "fv-kind:$kind" --arg d "fv-deadline:$deadline" '["fv", $o, $k, $d]'
}

# The rent payload of one Docker instance.
# cr_docker_payload <variant> <dc|""> <name> <image> <command json array> <env json object> <ports json array> <tags json>
cr_docker_payload() {
  jq -nc --arg it "$1" --arg dc "$2" --arg name "$3" --arg image "$4" --argjson cmd "$5" --argjson env "$6" \
    --argjson ports "$7" --argjson tags "$8" '{
      selector: {ByInstanceTypeAndLocation: ({instance_type: $it} + (if $dc == "" then {} else {datacenters: [$dc]} end))},
      with_public_ip: true,
      name: $name,
      tags: $tags,
      config: {Docker: {image: $image, command: $cmd, env: ($env | to_entries | map([.key, .value])), ports: $ports}}
    }'
}

# cr_recipe_image <driver> -> the VM image URL of CloudRift's NVIDIA Ubuntu
# recipe for a host (recipes/list). ProprietaryOnly hosts (Pascal/Volta, e.g.
# the V100 nodes) need the recipe tagged nvidia-driver-proprietary: the
# open-driver images do not boot there (the recipe's own description). Others
# get the newest Ubuntu tagged nvidia-driver without it. CLOUDRIFT_VM_IMAGE_URL
# overrides.
cr_recipe_image() {
  local driver="$1"
  if [[ -n "${CLOUDRIFT_VM_IMAGE_URL:-}" ]]; then printf '%s\n' "$CLOUDRIFT_VM_IMAGE_URL"; return 0; fi
  cr_post recipes/list '{}' | jq -r --arg d "$driver" '
    [.groups[]?.recipes[]? | select(.details.VirtualMachine.image_url != null)
     | select((.tags // []) | index("nvidia-driver"))
     | select(((.tags // []) | index("nvidia-driver-proprietary") != null) == ($d == "ProprietaryOnly"))
     | select(.name | test("Ubuntu"))
     | {u: .details.VirtualMachine.image_url, v: ((.name | capture("Ubuntu (?<v>[0-9]+\\.[0-9]+)").v // "0") | split(".") | map(tonumber)), n: .name}]
    | sort_by(.v) | last | .u // empty' | grep . || { log "no NVIDIA VM recipe for driver support $driver"; return 1; }
}

# The rent payload of one VM.
# cr_vm_payload <variant> <dc|""> <name> <image url> <cloud-init commands> <ssh public key|""> <tags json>
# No ports are requested: VMs expose every port on their address (CloudRift
# docs, "Port availability"), and our services bind to loopback behind a tunnel.
cr_vm_payload() {
  jq -nc --arg it "$1" --arg dc "$2" --arg name "$3" --arg img "$4" --arg ci "$5" --arg pk "$6" --argjson tags "$7" '{
      selector: {ByInstanceTypeAndLocation: ({instance_type: $it} + (if $dc == "" then {} else {datacenters: [$dc]} end))},
      with_public_ip: true,
      name: $name,
      tags: $tags,
      config: {VirtualMachine: ({image_url: $img, cloudinit_commands: $ci}
        + (if $pk == "" then {} else {ssh_key: {PublicKeys: [$pk]}} end))}
    }'
}

# cr_rent <payload> -> the instance id.
cr_rent() {
  local resp id
  resp="$(cr_post instances/rent "$1")" || return 1
  id="$(jq -r '.instance_ids[0] // empty' <<<"$resp")"
  [[ -n "$id" ]] || { log "rent answered without an instance id: $(head -c 300 <<<"$resp")"; return 1; }
  printf '%s\n' "$id"
}

# cr_wait_active <id> <timeout s> -> the instance JSON once Active with a host address.
cr_wait_active() {
  local id="$1" limit="$2" t0 inst st
  t0="$(date +%s)"
  while :; do
    inst="$(cr_instance "$id" || true)"
    st="$(jq -r '.status // "?"' <<<"${inst:-{\}}")"
    case "$st" in
      Failed) log "instance $id failed: $(jq -r '.failure.user_message // "no reason given"' <<<"$inst")"; return 1 ;;
      Inactive | Deactivating) log "instance $id is $st"; return 1 ;;
      Active)
        if [[ -n "$(jq -r '.host_address // empty' <<<"$inst")" ]]; then printf '%s\n' "$inst"; return 0; fi ;;
    esac
    (( $(date +%s) - t0 < limit )) || { log "instance $id not Active after ${limit}s (status $st)"; return 1; }
    sleep "${CR_POLL_S:-10}"
  done
}

# Mean GPU utilisation (percent) of an instance, empty when unknown.
cr_gpu_util() {
  cr_post instances/metrics "$(jq -nc --arg id "$1" '{selector: {ById: [$id]}}')" 2>/dev/null \
    | jq -r '[.metrics[0].gpus[]?.gpu_utilization_percent | select(. != null)] | if length == 0 then empty else (add / length) end'
}
