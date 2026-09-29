# shellcheck shell=bash
# Cloudflare D1 over its HTTP API, for the release and deploy scripts
# (docs/serve/releases.md). Source after scripts/gpu/lib.sh (log, die).
#
#   fv_d1_available                 a token is configured
#   fv_d1_query <sql> [params-json] print the result rows as one JSON array
#   fv_d1_exec_file <file.sql>      run every statement of a file (schema)
#
# Credentials, first set wins (the same names fv-serve reads):
#   token     FV_CF_API_TOKEN, CLOUDFLARE_API_TOKEN, CLOUDFLARE_API_KEY
#   account   FV_CF_ACCOUNT_ID, else the token's first account
#   database  FV_D1_DATABASE_ID, else the database named $FV_D1_DATABASE_NAME
#             (default fv-jobs)
#   API base  FV_D1_API_BASE (default https://api.cloudflare.com/client/v4;
#             the tests point it at a mock)
# The token reaches curl through a header file descriptor, never argv, and
# is never printed.

FV_D1_API_BASE="${FV_D1_API_BASE:-https://api.cloudflare.com/client/v4}"

fv_d1_token() { printf '%s' "${FV_CF_API_TOKEN:-${CLOUDFLARE_API_TOKEN:-${CLOUDFLARE_API_KEY:-}}}"; }
fv_d1_available() { [[ -n "$(fv_d1_token)" ]]; }

# _fv_cf <method> <path> [body] -> the response body (the body goes on stdin).
_fv_cf() {
  local method="$1" path="$2" body="${3:-}"
  if [[ -n "$body" ]]; then
    curl -sS --max-time 60 -X "$method" -H @<(printf 'Authorization: Bearer %s\n' "$(fv_d1_token)") \
      -H 'content-type: application/json' --data-binary @- "$FV_D1_API_BASE$path" <<<"$body"
  else
    curl -sS --max-time 60 -X "$method" -H @<(printf 'Authorization: Bearer %s\n' "$(fv_d1_token)") "$FV_D1_API_BASE$path"
  fi
}

# Looks up (once) and exports FV_CF_ACCOUNT_ID and FV_D1_DATABASE_ID.
fv_d1_resolve() {
  fv_d1_available || { log "D1: no API token (FV_CF_API_TOKEN / CLOUDFLARE_API_TOKEN / CLOUDFLARE_API_KEY)"; return 1; }
  local r
  if [[ -z "${FV_CF_ACCOUNT_ID:-}" ]]; then
    r="$(_fv_cf GET /accounts)" || { log "D1: account lookup failed"; return 1; }
    FV_CF_ACCOUNT_ID="$(jq -r '.result[0].id // empty' <<<"$r" 2>/dev/null)"
    [[ -n "$FV_CF_ACCOUNT_ID" ]] || { log "D1: the token sees no account"; return 1; }
    export FV_CF_ACCOUNT_ID
  fi
  if [[ -z "${FV_D1_DATABASE_ID:-}" ]]; then
    local name="${FV_D1_DATABASE_NAME:-fv-jobs}"
    r="$(_fv_cf GET "/accounts/$FV_CF_ACCOUNT_ID/d1/database?name=$name")" || { log "D1: database lookup failed"; return 1; }
    FV_D1_DATABASE_ID="$(jq -r --arg n "$name" '[.result[]? | select(.name == $n)][0].uuid // empty' <<<"$r" 2>/dev/null)"
    [[ -n "$FV_D1_DATABASE_ID" ]] || { log "D1: no database named $name"; return 1; }
    export FV_D1_DATABASE_ID
  fi
}

# fv_d1_query <sql> [params: JSON array, `?` placeholders] -> rows (JSON array).
fv_d1_query() {
  local sql="$1" params="${2:-[]}" body resp
  fv_d1_resolve || return 1
  body="$(jq -nc --arg sql "$sql" --argjson p "$params" '{sql: $sql} + (if ($p | length) > 0 then {params: $p} else {} end)')" \
    || { log "D1: bad params $params"; return 1; }
  resp="$(_fv_cf POST "/accounts/$FV_CF_ACCOUNT_ID/d1/database/$FV_D1_DATABASE_ID/query" "$body")" \
    || { log "D1: request failed"; return 1; }
  if ! jq -e '.success == true' <<<"$resp" >/dev/null 2>&1; then
    log "D1: $(jq -c '.errors // .' <<<"$resp" 2>/dev/null | head -c 400)"
    return 1
  fi
  jq -c '[.result[]?.results[]?]' <<<"$resp"
}

# fv_d1_exec_file <file.sql>: comments dropped, statements sent together.
fv_d1_exec_file() {
  local sql
  sql="$(sed -e 's/--.*$//' "$1" | tr '\n' ' ' | tr -s ' ')"
  fv_d1_query "$sql" >/dev/null
}
