#!/usr/bin/env bash
# Google Cloud OAuth access token from a service-account key, with openssl
# only (no gcloud): a JWT signed RS256 with the key's private key, exchanged
# at oauth2.googleapis.com/token (the "JWT bearer" grant). docs/serve/deploy-gcp.md.
#
#   auth.sh check        mint a token and read the project (prints the project
#                        id, the service account e-mail and the token's
#                        lifetime; never the key or the token)
#   auth.sh self-test    offline: sign a JWT with a throwaway RSA key and verify
#                        the signature (no network, no credentials needed)
#
# Sourced (scripts/gcp/lib.sh does): gcp_token prints a token on stdout for
# $(...) capture; gcp_curl_auth adds it as a header without putting it on a
# command line. Tokens are cached (mode 600) in $FV_GCP_STATE/token until 5
# minutes before they expire.
#
# Env:
#   GCP_SA_KEY_JSON   the service-account key JSON (as downloaded), or its
#                     base64; or GCP_SA_KEY_FILE, a path to the key file
#   GCP_PROJECT       default: the key's project_id
#   FV_GCP_SCOPE      default https://www.googleapis.com/auth/cloud-platform
#   FV_GCP_STATE      token cache directory (default ~/.cache/fastvideo-rs/gcp)
# shellcheck shell=bash

FV_GCP_STATE="${FV_GCP_STATE:-${XDG_CACHE_HOME:-$HOME/.cache}/fastvideo-rs/gcp}"
FV_GCP_SCOPE="${FV_GCP_SCOPE:-https://www.googleapis.com/auth/cloud-platform}"

_gcp_log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

# The key JSON on stdout (never logged). Accepts raw JSON, base64 JSON or a file.
gcp_key_json() {
  local k="${GCP_SA_KEY_JSON:-}"
  if [[ -z "$k" && -n "${GCP_SA_KEY_FILE:-}" ]]; then
    [[ -r "$GCP_SA_KEY_FILE" ]] || { _gcp_log "GCP_SA_KEY_FILE is not readable"; return 1; }
    k="$(cat "$GCP_SA_KEY_FILE")"
  fi
  [[ -n "$k" ]] || { _gcp_log "GCP_SA_KEY_JSON (or GCP_SA_KEY_FILE) is not set"; return 1; }
  if [[ "${k#"${k%%[![:space:]]*}"}" != "{"* ]]; then
    k="$(printf '%s' "$k" | base64 -d 2>/dev/null)" || { _gcp_log "GCP_SA_KEY_JSON is neither JSON nor base64 JSON"; return 1; }
  fi
  jq -e '.type == "service_account" and (.private_key|type) == "string" and (.client_email|type) == "string"' \
    >/dev/null 2>&1 <<<"$k" || { _gcp_log "GCP_SA_KEY_JSON is not a service-account key (type/private_key/client_email)"; return 1; }
  printf '%s' "$k"
}

gcp_sa_email() { gcp_key_json | jq -r .client_email; }

# Project: GCP_PROJECT, else the key's project_id.
gcp_project() {
  if [[ -n "${GCP_PROJECT:-}" ]]; then echo "$GCP_PROJECT"; return; fi
  gcp_key_json | jq -r '.project_id // empty'
}

_b64url() { openssl base64 -A | tr '+/' '-_' | tr -d '='; }

# _gcp_jwt <key json> <scope> <now>: the signed assertion on stdout. The
# private key reaches openssl through a pipe fd, never a file or an argument.
_gcp_jwt() {
  local key="$1" scope="$2" now="$3" header claims input sig
  header="$(jq -cn --arg kid "$(jq -r '.private_key_id // ""' <<<"$key")" '{alg:"RS256",typ:"JWT"} + (if $kid == "" then {} else {kid:$kid} end)' | _b64url)"
  claims="$(jq -cn --arg iss "$(jq -r .client_email <<<"$key")" --arg scope "$scope" \
    --arg aud "$(jq -r '.token_uri // "https://oauth2.googleapis.com/token"' <<<"$key")" --argjson now "$now" \
    '{iss:$iss, scope:$scope, aud:$aud, iat:$now, exp:($now + 3600)}' | _b64url)"
  input="$header.$claims"
  sig="$(printf '%s' "$input" | openssl dgst -sha256 -sign /dev/fd/3 3< <(jq -r .private_key <<<"$key") | _b64url)" || return 1
  [[ -n "$sig" ]] || return 1
  printf '%s.%s' "$input" "$sig"
}

# gcp_token: a valid access token on stdout (cached). Callers capture it and
# hand it to curl through gcp_curl_auth; it is never echoed to the terminal.
gcp_token() {
  local cache="$FV_GCP_STATE/token" now exp tok key jwt resp
  now="$(date +%s)"
  if [[ -r "$cache" ]]; then
    exp="$(head -1 "$cache")"
    if [[ "$exp" =~ ^[0-9]+$ ]] && (( exp - 300 > now )); then
      sed -n 2p "$cache"; return 0
    fi
  fi
  key="$(gcp_key_json)" || return 1
  jwt="$(_gcp_jwt "$key" "$FV_GCP_SCOPE" "$now")" || { _gcp_log "JWT signing failed (bad private_key?)"; return 1; }
  resp="$(curl -sS --max-time 30 -X POST "$(jq -r '.token_uri // "https://oauth2.googleapis.com/token"' <<<"$key")" \
    -H 'content-type: application/x-www-form-urlencoded' \
    --data-urlencode 'grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer' \
    --data-urlencode "assertion@/dev/fd/3" 3< <(printf '%s' "$jwt"))" || { _gcp_log "token endpoint unreachable"; return 1; }
  tok="$(jq -r '.access_token // empty' <<<"$resp" 2>/dev/null)"
  if [[ -z "$tok" ]]; then
    _gcp_log "token exchange refused: $(jq -c '{error, error_description}' <<<"$resp" 2>/dev/null || head -c 200 <<<"$resp")"
    return 1
  fi
  exp=$(( now + $(jq -r '.expires_in // 3600' <<<"$resp") ))
  ( umask 077; mkdir -p "$FV_GCP_STATE"; printf '%s\n%s\n' "$exp" "$tok" >"$cache.tmp" && mv "$cache.tmp" "$cache" )
  printf '%s' "$tok"
}

# gcp_curl_auth <curl args...>: curl with the bearer token read from a
# config fd (not visible in the process list).
gcp_curl_auth() {
  local tok
  tok="$(gcp_token)" || return 1
  curl -K <(printf 'header = "Authorization: Bearer %s"\n' "$tok") "$@"
}

_gcp_self_test() {
  local d key now jwt h c s
  d="$(mktemp -d)"; trap 'rm -rf "$d"' RETURN
  openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$d/k.pem" 2>/dev/null
  openssl pkey -in "$d/k.pem" -pubout -out "$d/pub.pem" 2>/dev/null
  key="$(jq -n --rawfile pk "$d/k.pem" '{type:"service_account", project_id:"self-test", private_key_id:"abc123",
    private_key:$pk, client_email:"fv-e2e@self-test.iam.gserviceaccount.com", token_uri:"https://oauth2.googleapis.com/token"}')"
  now="$(date +%s)"
  jwt="$(_gcp_jwt "$key" "$FV_GCP_SCOPE" "$now")"
  IFS=. read -r h c s <<<"$jwt"
  _unb64() { local x="$1"; x="${x//-/+}"; x="${x//_//}"; while (( ${#x} % 4 )); do x+="="; done; printf '%s' "$x" | openssl base64 -d -A; }
  _unb64 "$h" | jq -e '.alg == "RS256" and .kid == "abc123"' >/dev/null || { echo "self-test: bad header"; return 1; }
  _unb64 "$c" | jq -e --argjson now "$now" '.iss == "fv-e2e@self-test.iam.gserviceaccount.com" and .aud == "https://oauth2.googleapis.com/token" and .exp == $now + 3600 and (.scope|test("cloud-platform"))' >/dev/null \
    || { echo "self-test: bad claims"; return 1; }
  _unb64 "$s" >"$d/sig"
  printf '%s.%s' "$h" "$c" | openssl dgst -sha256 -verify "$d/pub.pem" -signature "$d/sig" >/dev/null \
    || { echo "self-test: signature does not verify"; return 1; }
  # base64-wrapped key form parses too.
  GCP_SA_KEY_JSON="$(printf '%s' "$key" | base64 -w0)" GCP_SA_KEY_FILE="" gcp_key_json | jq -e '.project_id == "self-test"' >/dev/null \
    || { echo "self-test: base64 key form"; return 1; }
  echo "auth self-test ok: RS256 JWT signed and verified; header, claims and the base64 key form parse"
}

_gcp_check() {
  local project email resp exp
  local t
  for t in jq openssl curl; do command -v "$t" >/dev/null || { echo "need $t" >&2; return 2; }; done
  email="$(gcp_sa_email)" || return 1
  project="$(gcp_project)"
  [[ -n "$project" ]] || { _gcp_log "no GCP_PROJECT and no project_id in the key"; return 1; }
  gcp_token >/dev/null || return 1
  exp="$(head -1 "$FV_GCP_STATE/token")"
  resp="$(gcp_curl_auth -sS --max-time 30 "https://compute.googleapis.com/compute/v1/projects/$project?fields=name,defaultServiceAccount")" || return 1
  if jq -e '.name' >/dev/null 2>&1 <<<"$resp"; then
    echo "gcp auth ok: project=$project sa=$email token_valid_s=$(( exp - $(date +%s) )) compute_api=enabled"
  else
    echo "gcp auth ok (token minted) but Compute Engine refused: $(jq -c '.error | {code, message}' <<<"$resp" 2>/dev/null || head -c 300 <<<"$resp")" >&2
    return 1
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  case "${1:-}" in
    check) _gcp_check ;;
    self-test) _gcp_self_test ;;
    *) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
  esac
fi
