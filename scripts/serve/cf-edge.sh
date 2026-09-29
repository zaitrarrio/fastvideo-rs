#!/usr/bin/env bash
# fv-edge: the Cloudflare Worker + per-pool Durable Object dispatcher
# (crates/fastvideo-edge, docs/serve/gateway-cloudflare.md "How to run the
# parallel path"). STAGING ONLY: every Cloudflare resource is named
# `fv-edge-staging` and nothing else is touched.
#
#   cf-edge.sh build            wasm on the shared build pod (scripts/dev/build-pod.sh;
#                               FV_EDGE_BUILD=local: local cargo), then wasm-bindgen +
#                               bundle here -> artifacts/edge/build
#   cf-edge.sh dev [port]       wrangler dev: local workerd with local DO/D1 state
#                               (artifacts/edge/dev-state), default port 8787
#   cf-edge.sh deploy           D1 `fv-edge-staging` and R2 `fv-edge-staging-envelopes`
#                               (created once), secrets, wrangler deploy
#                               -> https://fv-edge-staging.<subdomain>.workers.dev
#   cf-edge.sh status [pool..]  the Worker's version and each pool's dispatcher status
#   cf-edge.sh down             delete the Worker (and its Durable Objects), the D1
#                               database and the R2 bucket; needs FV_EDGE_CONFIRM=fv-edge-staging
#   cf-edge.sh check-token      what the Cloudflare token can read (never printed)
#
# State (mode 700): ${FV_EDGE_STATE:-~/.config/fv-edge-staging}/
#   internal_token   the Worker's FV_INTERNAL_TOKEN (gateway + workers); made on first
#                    deploy unless FV_EDGE_INTERNAL_TOKEN_FILE names the gateway's
#   admin_token      FV_ADMIN_TOKEN (read-only status); FV_EDGE_ADMIN_TOKEN_FILE likewise
#   url, d1_id       what deploy created
# Cloudflare token: FV_CF_TOKEN_FILE (default /root/.config/fv/cf_api_token), read from
# the file into wrangler's environment and curl header files only; never printed.
# Env: FV_EDGE_POOL_LOCATIONS ('{"h3-turbo":"weur"}'), FV_EDGE_ACK_TIMEOUT_MS,
# FV_EDGE_RECONNECT_GRACE_MS, FV_EDGE_STALE_AFTER_MS, FV_EDGE_REDISPATCH_WAIT_MS,
# FV_EDGE_SPILL_BYTES (envelopes above it go to R2, default 1 MiB),
# FV_EDGE_BUILD_AGENT (build-pod agent name, default this worktree's), FV_EDGE_TOOLS.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

NAME="fv-edge-staging"
D1_NAME="fv-edge-staging"
R2_NAME="fv-edge-staging-envelopes"
COMPAT_DATE="2026-09-01"
WRANGLER_VERSION="4.143.0"
ESBUILD_VERSION="0.28.2"
# Must equal the wasm-bindgen crate version pinned in crates/fastvideo-edge/Cargo.toml.
WASM_BINDGEN_VERSION="0.2.129"
WASM_BINDGEN_SHA256="82d12bb940e2d4e72e0d5605387fc1b8ca179044e012b620f0ce4e7440e8320e"

STATE="${FV_EDGE_STATE:-${XDG_CONFIG_HOME:-$HOME/.config}/fv-edge-staging}"
TOOLS="${FV_EDGE_TOOLS:-${XDG_CACHE_HOME:-$HOME/.cache}/fv-edge-tools}"
OUT="$ROOT/artifacts/edge"
CF_TOKEN_FILE="${FV_CF_TOKEN_FILE:-/root/.config/fv/cf_api_token}"
API="https://api.cloudflare.com/client/v4"

umask 077
log() { printf '[%s] %s\n' "$(date -u +%T)" "$*" >&2; }
die() { log "error: $*"; exit 1; }

tools() {
  mkdir -p "$TOOLS"
  if [[ ! -x "$TOOLS/node_modules/.bin/wrangler" || "$("$TOOLS/node_modules/.bin/wrangler" --version 2>/dev/null | tail -1)" != "$WRANGLER_VERSION" ]]; then
    log "installing wrangler $WRANGLER_VERSION and esbuild $ESBUILD_VERSION into $TOOLS"
    (cd "$TOOLS" && { [[ -f package.json ]] || npm init -y >/dev/null; } && npm install --no-audit --no-fund "wrangler@$WRANGLER_VERSION" "esbuild@$ESBUILD_VERSION" >/dev/null)
  fi
  local wb="$TOOLS/wasm-bindgen-$WASM_BINDGEN_VERSION-x86_64-unknown-linux-musl/wasm-bindgen"
  if [[ ! -x "$wb" ]]; then
    log "downloading wasm-bindgen $WASM_BINDGEN_VERSION"
    local tgz="$TOOLS/wasm-bindgen-$WASM_BINDGEN_VERSION.tgz"
    curl -sSfL -o "$tgz" "https://github.com/wasm-bindgen/wasm-bindgen/releases/download/$WASM_BINDGEN_VERSION/wasm-bindgen-$WASM_BINDGEN_VERSION-x86_64-unknown-linux-musl.tar.gz"
    echo "$WASM_BINDGEN_SHA256  $tgz" | sha256sum -c --quiet || die "wasm-bindgen tarball checksum mismatch"
    tar -xzf "$tgz" -C "$TOOLS"
  fi
  WASM_BINDGEN="$wb"
  WRANGLER="$TOOLS/node_modules/.bin/wrangler"
}

cmd_build() {
  tools
  local rel="wasm32-unknown-unknown/release/fastvideo_edge.wasm"
  mkdir -p "$OUT"
  if [[ "${FV_EDGE_BUILD:-pod}" == "local" ]]; then
    local target="${CARGO_TARGET_DIR:-$OUT/target}"
    log "building crates/fastvideo-edge for wasm32 locally (target $target)"
    (cd "$ROOT" && CARGO_TARGET_DIR="$target" CARGO_PROFILE_RELEASE_PANIC=abort CARGO_PROFILE_RELEASE_LTO=off \
      cargo build -p fastvideo-edge --target wasm32-unknown-unknown --release)
    cp "$target/$rel" "$OUT/fastvideo_edge.wasm"
  else
    local agent="${FV_EDGE_BUILD_AGENT:-$(basename "$ROOT")}"
    log "building crates/fastvideo-edge for wasm32 on the build pod (agent $agent)"
    "$ROOT/scripts/dev/build-pod.sh" run "$agent" -- CARGO_PROFILE_RELEASE_PANIC=abort \
      cargo build -p fastvideo-edge --target wasm32-unknown-unknown --release
    "$ROOT/scripts/dev/build-pod.sh" fetch "$agent" "$rel" "$OUT/fastvideo_edge.wasm" >&2
  fi
  rm -rf "$OUT/build.tmp" && mkdir -p "$OUT/build.tmp"
  "$WASM_BINDGEN" "$OUT/fastvideo_edge.wasm" --out-dir "$OUT/build.tmp" --no-typescript --target module \
    --out-name index --experimental-reset-state-function --force-enable-abort-handler
  node "$ROOT/crates/fastvideo-edge/js/pack.mjs" "$OUT/build.tmp" "$TOOLS/node_modules/esbuild" >&2
  rm -rf "$OUT/build" && mv "$OUT/build.tmp" "$OUT/build"
  log "built $OUT/build ($(du -sh "$OUT/build" | cut -f1); wasm $(du -h "$OUT/build/index_bg.wasm" | cut -f1))"
}

need_build() { [[ -f "$OUT/build/index.js" && -f "$OUT/build/index_bg.wasm" ]] || die "no build: run $0 build first"; }

token_file() {
  local var="$1" file="$STATE/$2"
  mkdir -p "$STATE" && chmod 700 "$STATE"
  if [[ -n "${!var:-}" ]]; then
    [[ -s "${!var}" ]] || die "$var names an empty or missing file"
    printf '%s' "${!var}"
    return
  fi
  if [[ ! -s "$file" ]]; then
    head -c 32 /dev/urandom | base64 | tr -d '/+=\n' >"$file.tmp" && mv "$file.tmp" "$file"
    log "made a new $2 in $STATE (mode 600)"
  fi
  printf '%s' "$file"
}

# wrangler.toml for `deploy` (with the D1 id) or `dev`.
write_config() {
  local dest="$1" d1_id="$2" version="$3"
  local locations="${FV_EDGE_POOL_LOCATIONS:-{\}}"
  {
    printf 'name = "%s"\nmain = "build/index.js"\ncompatibility_date = "%s"\nworkers_dev = true\npreview_urls = false\n\n' "$NAME" "$COMPAT_DATE"
    printf '[[durable_objects.bindings]]\nname = "POOL_SCHEDULER"\nclass_name = "PoolScheduler"\n\n'
    printf '[[migrations]]\ntag = "v1"\nnew_sqlite_classes = ["PoolScheduler"]\n\n'
    printf '[[d1_databases]]\nbinding = "DB"\ndatabase_name = "%s"\ndatabase_id = "%s"\n\n' "$D1_NAME" "$d1_id"
    printf '[[r2_buckets]]\nbinding = "ENVELOPES"\nbucket_name = "%s"\n\n' "$R2_NAME"
    printf '[observability]\nenabled = true\n\n'
    printf '[vars]\nFV_EDGE_VERSION = "%s"\n' "$version"
    printf "POOL_LOCATIONS = '%s'\n" "$locations"
    [[ -n "${FV_EDGE_ACK_TIMEOUT_MS:-}" ]] && printf 'ACK_TIMEOUT_MS = "%s"\n' "$FV_EDGE_ACK_TIMEOUT_MS"
    [[ -n "${FV_EDGE_RECONNECT_GRACE_MS:-}" ]] && printf 'RECONNECT_GRACE_MS = "%s"\n' "$FV_EDGE_RECONNECT_GRACE_MS"
    [[ -n "${FV_EDGE_STALE_AFTER_MS:-}" ]] && printf 'STALE_AFTER_MS = "%s"\n' "$FV_EDGE_STALE_AFTER_MS"
    [[ -n "${FV_EDGE_REDISPATCH_WAIT_MS:-}" ]] && printf 'REDISPATCH_WAIT_MS = "%s"\n' "$FV_EDGE_REDISPATCH_WAIT_MS"
    [[ -n "${FV_EDGE_SPILL_BYTES:-}" ]] && printf 'SPILL_BYTES = "%s"\n' "$FV_EDGE_SPILL_BYTES"
    true
  } >"$dest"
}

version_string() {
  printf '%s-%s' "$(git -C "$ROOT" rev-parse --short=7 HEAD 2>/dev/null || echo unknown)" "$(date -u +%Y%m%dT%H%M%SZ)"
}

cmd_dev() {
  tools
  need_build
  local port="${1:-8787}" dir="$OUT/dev"
  mkdir -p "$dir" "$OUT/dev-state"
  rm -rf "$dir/build" && cp -r "$OUT/build" "$dir/build"
  write_config "$dir/wrangler.toml" "00000000-0000-0000-0000-000000000000" "dev-$(version_string)"
  local it at
  it="$(token_file FV_EDGE_INTERNAL_TOKEN_FILE internal_token)"
  at="$(token_file FV_EDGE_ADMIN_TOKEN_FILE admin_token)"
  printf 'FV_INTERNAL_TOKEN=%s\nFV_ADMIN_TOKEN=%s\n' "$(cat "$it")" "$(cat "$at")" >"$dir/.dev.vars"
  log "wrangler dev on 127.0.0.1:$port (state $OUT/dev-state; tokens from $STATE)"
  cd "$dir"
  exec env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy \
    "$WRANGLER" dev --port "$port" --ip 127.0.0.1 --persist-to "$OUT/dev-state" --show-interactive-dev-session=false
}

# Cloudflare credentials into the environment (wrangler) and a header file (curl).
cf_env() {
  [[ -s "$CF_TOKEN_FILE" ]] || die "no Cloudflare token at $CF_TOKEN_FILE"
  HDR="$(mktemp)"
  trap 'rm -f "$HDR"' EXIT
  printf 'Authorization: Bearer %s\n' "$(tr -d '\n' <"$CF_TOKEN_FILE")" >"$HDR"
  export CLOUDFLARE_API_TOKEN
  CLOUDFLARE_API_TOKEN="$(tr -d '\n' <"$CF_TOKEN_FILE")"
  if [[ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]]; then
    CLOUDFLARE_ACCOUNT_ID="${FV_CF_ACCOUNT_ID:-$(curl -sSf -H @"$HDR" "$API/accounts" | python3 -c 'import sys,json; r=json.load(sys.stdin)["result"]; print(r[0]["id"] if len(r)==1 else "")')}"
    [[ -n "$CLOUDFLARE_ACCOUNT_ID" ]] || die "several accounts: set FV_CF_ACCOUNT_ID"
  fi
  export CLOUDFLARE_ACCOUNT_ID
}

cf() { curl -sS -H @"$HDR" -H 'content-type: application/json' "$@"; }

subdomain() {
  cf "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/workers/subdomain" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["result"]["subdomain"] if d.get("success") else "")'
}

d1_id() {
  cf "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/d1/database?name=$D1_NAME" |
    python3 -c 'import sys,json; d=json.load(sys.stdin); print(next((x["uuid"] for x in d.get("result") or [] if x["name"]==sys.argv[1]), ""))' "$D1_NAME"
}

cmd_check_token() {
  cf_env
  local b="$API/accounts/$CLOUDFLARE_ACCOUNT_ID"
  for p in workers/scripts workers/subdomain workers/durable_objects/namespaces d1/database; do
    printf '%-40s ' "$p"
    cf "$b/$p" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("read ok" if d.get("success") else d.get("errors"))'
  done
  log "write access is only known when deploy runs (wrangler reports a missing permission by name)"
}

cmd_deploy() {
  tools
  need_build
  cf_env
  local id
  id="$(d1_id)"
  if [[ -z "$id" ]]; then
    log "creating D1 database $D1_NAME"
    local r
    r="$(cf -X POST "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/d1/database" -d "{\"name\":\"$D1_NAME\"}")"
    id="$(printf '%s' "$r" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["result"]["uuid"] if d.get("success") else "")')"
    [[ -n "$id" ]] || die "D1 create failed: $(printf '%s' "$r" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("errors"))') (the token needs Account > D1 > Edit)"
  fi
  mkdir -p "$STATE" && chmod 700 "$STATE"
  printf '%s\n' "$id" >"$STATE/d1_id"
  if ! cf "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/r2/buckets/$R2_NAME" | python3 -c 'import sys,json; sys.exit(0 if json.load(sys.stdin).get("success") else 1)'; then
    log "creating R2 bucket $R2_NAME"
    cf -X POST "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/r2/buckets" -d "{\"name\":\"$R2_NAME\"}" |
      python3 -c 'import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get("success") else "R2 bucket create failed: %s (the token needs Account > Workers R2 Storage > Edit)" % d.get("errors"))'
  fi
  local dir="$OUT/deploy" version
  version="$(version_string)"
  mkdir -p "$dir"
  rm -rf "$dir/build" && cp -r "$OUT/build" "$dir/build"
  write_config "$dir/wrangler.toml" "$id" "$version"
  local it at secrets
  it="$(token_file FV_EDGE_INTERNAL_TOKEN_FILE internal_token)"
  at="$(token_file FV_EDGE_ADMIN_TOKEN_FILE admin_token)"
  secrets="$STATE/secrets.json.tmp"
  python3 -c 'import json,sys; print(json.dumps({"FV_INTERNAL_TOKEN": open(sys.argv[1]).read().strip(), "FV_ADMIN_TOKEN": open(sys.argv[2]).read().strip()}))' "$it" "$at" >"$secrets"
  log "deploying $NAME (version $version)"
  local rc=0
  (cd "$dir" && "$WRANGLER" deploy --secrets-file "$secrets") || rc=$?
  rm -f "$secrets"
  if [[ $rc -ne 0 ]]; then
    die "wrangler deploy failed ($rc). If it names a permission, the token needs: Account > Workers Scripts > Edit (covers Durable Objects) and Account > D1 > Edit"
  fi
  local sub
  sub="$(subdomain)"
  [[ -n "$sub" ]] || die "the account has no workers.dev subdomain"
  printf 'https://%s.%s.workers.dev\n' "$NAME" "$sub" >"$STATE/url"
  log "deployed: $(cat "$STATE/url") (version $version)"
}

cmd_status() {
  local url
  url="$(cat "$STATE/url" 2>/dev/null)" || die "not deployed (no $STATE/url)"
  echo "url: $url"
  curl -sS --max-time 10 "$url/" && echo
  local at hdr
  at="$(token_file FV_EDGE_ADMIN_TOKEN_FILE admin_token)"
  hdr="$(mktemp)"
  printf 'Authorization: Bearer %s\n' "$(cat "$at")" >"$hdr"
  for p in "$@"; do
    echo "pool $p:"
    curl -sS --max-time 10 -H @"$hdr" "$url/pools/$p/status" | python3 -m json.tool || true
  done
  rm -f "$hdr"
}

cmd_down() {
  [[ "${FV_EDGE_CONFIRM:-}" == "$NAME" ]] || die "this deletes the Worker $NAME, its Durable Objects and the D1 database $D1_NAME: set FV_EDGE_CONFIRM=$NAME"
  tools
  cf_env
  log "deleting Worker $NAME (its Durable Object namespace goes with it)"
  "$WRANGLER" delete --name "$NAME" --force || log "wrangler delete failed (already gone?)"
  local id
  id="$(d1_id)"
  if [[ -n "$id" ]]; then
    log "deleting D1 database $D1_NAME"
    cf -X DELETE "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/d1/database/$id" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("deleted" if d.get("success") else d.get("errors"))'
  fi
  log "emptying and deleting R2 bucket $R2_NAME"
  local keys
  keys="$(cf "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/r2/buckets/$R2_NAME/objects?per_page=1000" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("\n".join(o["key"] for o in (d.get("result") or [])))' 2>/dev/null || true)"
  while IFS= read -r k; do
    [[ -n "$k" ]] && cf -X DELETE "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/r2/buckets/$R2_NAME/objects/$(python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$k")" >/dev/null
  done <<<"$keys"
  cf -X DELETE "$API/accounts/$CLOUDFLARE_ACCOUNT_ID/r2/buckets/$R2_NAME" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("deleted" if d.get("success") else d.get("errors"))'
  rm -f "$STATE/url" "$STATE/d1_id"
}

case "${1:-}" in
  build) shift; cmd_build "$@" ;;
  dev) shift; cmd_dev "$@" ;;
  deploy) shift; cmd_deploy "$@" ;;
  status) shift; cmd_status "$@" ;;
  down) shift; cmd_down "$@" ;;
  check-token) shift; cmd_check_token "$@" ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
esac
