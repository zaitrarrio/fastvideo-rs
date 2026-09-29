#!/usr/bin/env bash
# Runpod GPU price check BEFORE a pod is created. Source, don't execute.
# shellcheck shell=bash
#
# Every script that creates a GPU pod calls `fv_runpod_price_ok <gpu type id>
# <cap $/hr> [SECURE|COMMUNITY] [datacenter]` before its `POST /pods` and
# skips the GPU type (or stops) when it fails. The old pattern (create, read
# `costPerHr` from the reply, delete when over the cap) created and deleted
# 18 H100 pods above a director-debug run's cap within seconds; each one was
# billed and briefly held a machine.
#
# The price is Runpod's GraphQL quote for the GPU type: `securePrice` (or
# `communityPrice`) and, with a datacenter, that datacenter's lowest
# uninterruptible price; the higher of the quotes counts. No quote (unknown
# type, API error) fails the check: a pod is never created at an unknown
# price. On success FV_QUOTED_DPH holds the quote. The API key goes in a
# header file, never on a command line.
#
# Env: RUNPOD_API_KEY, RUNPOD_GRAPHQL (default https://api.runpod.io/graphql).

# Prints the quoted $/hr for one GPU of type $1 ($2 cloud, $3 datacenter), or
# nothing.
fv_runpod_gpu_price() {
  local gpu="$1" cloud="${2:-SECURE}" dc="${3:-}" gql="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}" lowest="" secure=true q resp
  [[ "$cloud" == COMMUNITY ]] && secure=false
  if [[ -n "$dc" ]]; then
    lowest="lowestPrice(input: {gpuCount: 1, secureCloud: $secure, dataCenterId: $(jq -Rn --arg d "$dc" '$d')}) { uninterruptablePrice }"
  fi
  q="query { gpuTypes(input: {id: $(jq -Rn --arg g "$gpu" '$g')}) { id securePrice communityPrice $lowest } }"
  resp="$(curl -sS --max-time 30 -H @<(printf 'Authorization: Bearer %s\n' "${RUNPOD_API_KEY:-}") \
    -H 'content-type: application/json' -d "$(jq -nc --arg q "$q" '{query: $q}')" "$gql" 2>/dev/null)" || return 0
  jq -r --arg cloud "$cloud" '
    (.data.gpuTypes // [])[0] // empty
    | [(if $cloud == "COMMUNITY" then .communityPrice else .securePrice end), (.lowestPrice.uninterruptablePrice // null)]
    | map(select(type == "number" and . > 0))
    | if length == 0 then empty else max end' <<<"$resp" 2>/dev/null || true
}

# 0 when GPU type $1 is quoted at or under $2 $/hr ($3 cloud, $4 datacenter);
# else says why on stderr and returns 1. Call it before creating the pod.
fv_runpod_price_ok() {
  local gpu="$1" cap="$2" price
  FV_QUOTED_DPH=""
  price="$(fv_runpod_gpu_price "$gpu" "${3:-SECURE}" "${4:-}")"
  if [[ -z "$price" ]]; then
    printf '[%s] price check: no Runpod price quote for "%s"; not creating a pod at an unknown price\n' "$(date -u +%H:%M:%S)" "$gpu" >&2
    return 1
  fi
  if awk -v p="$price" -v c="$cap" 'BEGIN{exit !(p+0 > c+0)}'; then
    printf '[%s] price check: "%s" is quoted at $%s/hr, over the cap $%s/hr; not creating\n' "$(date -u +%H:%M:%S)" "$gpu" "$price" "$cap" >&2
    return 1
  fi
  FV_QUOTED_DPH="$price"
  export FV_QUOTED_DPH
  return 0
}
