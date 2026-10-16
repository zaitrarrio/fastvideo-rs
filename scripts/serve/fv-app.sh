#!/usr/bin/env bash
# fv-app (app/): the public consumer portal Worker.
#   fv-app.sh deploy staging   typecheck, unit tests, then wrangler deploy --env staging
# fv-app shares fv-control's D1; fv-control owns the schema (key_requests is
# control/migrations/0014_key_requests.sql), so deploy fv-control staging first
# on a fresh database (scripts/serve/fv-control.sh deploy staging).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
die() { echo "fv-app: $*" >&2; exit 1; }
cmd="${1:-}"; shift || true
case "$cmd" in
  deploy)
    [[ "${1:-}" == staging ]] || die "deploy staging (production is not set up: the top-level D1 id is a placeholder)"
    cd "$HERE/../../app"
    npm ci --silent
    npx tsc --noEmit
    npx vitest run test/unit
    if [[ -z "${CLOUDFLARE_API_TOKEN:-}" && -r /root/.config/fv/cf_api_token ]]; then CLOUDFLARE_API_TOKEN="$(cat /root/.config/fv/cf_api_token)"; export CLOUDFLARE_API_TOKEN; fi
    npx wrangler deploy --env staging ;;
  *) die "usage: fv-app.sh deploy staging" ;;
esac
