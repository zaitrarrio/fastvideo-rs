#!/usr/bin/env bash
# A small CLI over fv-control's JSON API (docs/control/README.md): for
# scripts and agents. The API token (fvc_…, minted in the dashboard under
# Settings, scope read or admin) goes to curl through a header file
# descriptor, never argv.
#
#   fv-control.sh status                      balance, burn, time to floor, clusters, alerts
#   fv-control.sh pods | alerts | costs [days]
#   fv-control.sh clusters | cluster <name>
#   fv-control.sh define <spec.json>          (or `define tiny-cpu <name>` / `define standard <name>`)
#   fv-control.sh price|start|stop <name>
#   fv-control.sh extend <name> <minutes>
#   fv-control.sh scale <name> <pool> <count>
#   fv-control.sh roll <name> <channel|sha|image> [--gateway]
#   fv-control.sh restart <name>               rolling restart of the pods whose env changed
#   fv-control.sh gateway-start|gateway-stop <name>
#   fv-control.sh wait <name>                  until the running operation ends; prints its log
#   fv-control.sh env <account|cluster <name>|pod <id>>            list
#   fv-control.sh env-set <account|cluster <name>|pod <id>> KEY VALUE [--secret]
#   fv-control.sh env-unset <account|cluster <name>|pod <id>> KEY
#   fv-control.sh effective-env <name>         per pod, masked; which pods need a restart
#   fv-control.sh logs <pod> [search] [level]
#   fv-control.sh import <cluster.json> [admin-key.pem] [name]
#   fv-control.sh promote <sha> [channel] [--dry-run] | rollback [channel] [--dry-run]
#   fv-control.sh api <METHOD> <path> [json]  anything else
#   fv-control.sh deploy staging               build, migrate and deploy the Worker (wrangler)
#   fv-control.sh edge-link staging            give the Worker the staging edge (EDGE_URL, EDGE_D1_DATABASE_ID,
#                                              EDGE_INTERNAL_TOKEN, EDGE_ADMIN_TOKEN as secrets) from
#                                              scripts/serve/cf-edge.sh's state (FV_EDGE_STATE); never printed
#
# Env: FV_CONTROL_URL (default https://fv-control-staging.maximalize.workers.dev),
# FV_CONTROL_TOKEN_FILE (default ~/.config/fv/fv-control-token, mode 600).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="${FV_CONTROL_URL:-https://fv-control-staging.maximalize.workers.dev}"
TOKEN_FILE="${FV_CONTROL_TOKEN_FILE:-$HOME/.config/fv/fv-control-token}"

die() { echo "fv-control: $*" >&2; exit 1; }
api() {
  local method="$1" path="$2" body="${3:-}"
  [[ -r "$TOKEN_FILE" ]] || die "no API token in $TOKEN_FILE (mint one under Settings)"
  curl -sS --fail-with-body -X "$method" -H @<(printf 'Authorization: Bearer %s\n' "$(tr -d '\n' <"$TOKEN_FILE")") \
    -H 'content-type: application/json' ${body:+-d "$body"} "$URL$path"
}
scope_path() { # account | cluster <name> | pod <id>  -> /api/env/…, shifts
  case "$1" in
    account) echo "/api/env/account" ;;
    cluster) echo "/api/env/cluster/$(api GET "/api/clusters/$2" | jq -r .cluster.id)" ;;
    pod) echo "/api/env/pod/$2" ;;
    *) die "scope: account | cluster <name> | pod <id>" ;;
  esac
}
cmd="${1:-}"; shift || true
case "$cmd" in
  status) api GET /api/overview | jq '{balance, burn_per_hr, hours_to_floor, cost_today_total, running_pods, idle_pods: [.idle_pods[] | {name, idle_min, cost_per_hr}], clusters, alerts: [.alerts[] | {severity, message}]}' ;;
  pods) api GET /api/pods | jq -r '.pods[] | [.pod_id, .name, .owner, .desired_status, .health // "-", (.cost_per_hr|tostring), ((.gpu_util // -1)|floor|tostring)] | @tsv' ;;
  alerts) api GET /api/alerts | jq -r '.alerts[] | [.severity, .kind, .message] | @tsv' ;;
  costs) api GET "/api/costs?days=${1:-7}" | jq '{by_day, by_owner, by_cluster}' ;;
  clusters) api GET /api/clusters | jq -r '.clusters[] | [.name, .status, (.deadline // 0 | . / 1000 | todate), (.op.kind // "-")] | @tsv' ;;
  cluster) api GET "/api/clusters/${1:?name}" | jq '{cluster: (.cluster | del(.spec)), pods: [.pods[] | {pod_id, role, pool, status, cost_per_hr}], op, drift: {channel: .drift.channel, head: .drift.head_sha, running: .drift.running_sha, drift: .drift.drift}}' ;;
  define)
    if [[ "${1:-}" == tiny-cpu || "${1:-}" == standard ]]; then api POST /api/clusters "$(jq -nc --arg t "$1" --arg n "${2:?name}" '{spec: {name: $n, template: $t}}')"
    else api POST /api/clusters "$(jq -c '{spec: .}' "${1:?spec.json}")"; fi | jq .cluster.id ;;
  price) api POST "/api/clusters/${1:?name}/price" '{}' | jq . ;;
  start | stop | restart) api POST "/api/clusters/${1:?name}/$cmd" '{}' | jq -c . ;;
  gateway-start | gateway-stop) api POST "/api/clusters/${1:?name}/gateway/${cmd#gateway-}" '{}' | jq -c . ;;
  extend) api POST "/api/clusters/${1:?name}/extend" "$(jq -nc --argjson m "${2:?minutes}" '{minutes: $m}')" | jq -c . ;;
  scale) api POST "/api/clusters/${1:?name}/scale" "$(jq -nc --arg p "${2:?pool}" --argjson n "${3:?count}" '{pool: $p, count: $n}')" | jq -c . ;;
  roll) api POST "/api/clusters/${1:?name}/roll" "$(jq -nc --arg t "${2:?target}" --argjson g "$([[ "${3:-}" == --gateway ]] && echo true || echo false)" '{target: $t, gateway: $g}')" | jq -c . ;;
  wait)
    while :; do
      op="$(api GET "/api/clusters/${1:?name}/ops" | jq -c '.operations[0]')"
      [[ "$(jq -r .status <<<"$op")" == running ]] || break
      sleep 10
    done
    jq -r '"\(.kind) \(.status) \(.error // "")", (.log[] | "  \(.at / 1000 | todate) \(.msg)")' <<<"$op" ;;
  env) api GET "$(scope_path "$@")" | jq -r '.vars[] | [.key, .value, (if .secret then "secret" else "" end)] | @tsv' ;;
  env-set)
    p="$(scope_path "$@")"; if [[ "$1" == account ]]; then shift; else shift 2; fi
    api PUT "$p/${1:?KEY}" "$(jq -nc --arg v "${2?VALUE}" --argjson s "$([[ "${3:-}" == --secret ]] && echo true || echo false)" '{value: $v, secret: $s}')" | jq -c . ;;
  env-unset)
    p="$(scope_path "$@")"; if [[ "$1" == account ]]; then shift; else shift 2; fi
    api DELETE "$p/${1:?KEY}" | jq -c . ;;
  effective-env) api GET "/api/clusters/$(api GET "/api/clusters/${1:?name}" | jq -r .cluster.id)/env" | jq '{needs_restart, pods: [.pods[] | {pod_id, role, pool, needs_restart, env: [.env[] | "\(.key)=\(.value) [\(.source)]"]}]}' ;;
  logs)
    q="pod=${1:?pod}&level=${3:-info}&limit=500"; [[ -n "${2:-}" ]] && q+="&q=$(jq -rn --arg s "$2" '$s|@uri')"
    api GET "/api/logs?$q" | jq -r '.lines[] | "\(.ts / 1000 | todate) \(.level | ascii_upcase) \(.target // ""): \(.msg) \(.fields // {} | tojson)"' ;;
  import)
    body="$(jq -nc --slurpfile s "${1:?cluster.json}" --arg pem "$([[ -n "${2:-}" ]] && cat "$2")" --arg n "${3:-}" '{state: $s[0]} + (if $pem != "" then {admin_key_pem: $pem} else {} end) + (if $n != "" then {name: $n} else {} end)')"
    api POST /api/clusters/import "$body" | jq '{id: .cluster.id, name: .cluster.name, note}' ;;
  promote) api POST /api/github/release "$(jq -nc --arg t "${1:?sha}" --arg c "${2:-stable}" --argjson d "$([[ " $* " == *" --dry-run "* ]] && echo true || echo false)" '{action: "promote", target: $t, channel: $c, dry_run: $d}')" | jq . ;;
  rollback) api POST /api/github/release "$(jq -nc --arg c "${1:-stable}" --argjson d "$([[ " $* " == *" --dry-run "* ]] && echo true || echo false)" '{action: "rollback", channel: (if $c == "--dry-run" then "stable" else $c end), dry_run: $d}')" | jq . ;;
  api) api "${1:?METHOD}" "${2:?path}" "${3:-}" ;;
  deploy)
    [[ "${1:-}" == staging ]] || die "deploy staging (production: add an [env.production] block first)"
    cd "$HERE/../../control"
    npm ci --silent
    npx tsc --noEmit
    npx vitest run
    if [[ -z "${CLOUDFLARE_API_TOKEN:-}" && -r /root/.config/fv/cf_api_token ]]; then CLOUDFLARE_API_TOKEN="$(cat /root/.config/fv/cf_api_token)"; export CLOUDFLARE_API_TOKEN; fi
    npx wrangler d1 migrations apply fv-control --remote --env staging
    npx wrangler deploy --env staging ;;
  edge-link)
    [[ "${1:-}" == staging ]] || die "edge-link staging"
    st="${FV_EDGE_STATE:-$HOME/.config/fv-edge-staging}"
    for f in url internal_token admin_token d1_id; do [[ -s "$st/$f" ]] || die "no $st/$f: run scripts/serve/cf-edge.sh deploy first"; done
    cd "$HERE/../../control"
    if [[ -z "${CLOUDFLARE_API_TOKEN:-}" && -r /root/.config/fv/cf_api_token ]]; then CLOUDFLARE_API_TOKEN="$(cat /root/.config/fv/cf_api_token)"; export CLOUDFLARE_API_TOKEN; fi
    (umask 077; tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
      python3 -c 'import json,sys; d=sys.argv[1]; r=lambda f: open(f"{d}/{f}").read().strip(); print(json.dumps({"EDGE_URL": r("url"), "EDGE_D1_DATABASE_ID": r("d1_id"), "EDGE_INTERNAL_TOKEN": r("internal_token"), "EDGE_ADMIN_TOKEN": r("admin_token")}))' "$st" >"$tmp"
      npx wrangler secret bulk "$tmp" --env staging >/dev/null)
    echo "fv-control staging: edge $(cat "$st/url")" ;;
  *) sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
