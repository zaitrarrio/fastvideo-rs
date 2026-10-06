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
#   fv-control.sh roll <name> <channel|sha|image>
#   fv-control.sh restart <name>               rolling restart of the pods whose env changed
#   fv-control.sh wait <name>                  until the running operation ends; prints its log
#   fv-control.sh env <account|cluster <name>|pod <id>>            list
#   fv-control.sh env-set <account|cluster <name>|pod <id>> KEY VALUE [--secret]
#   fv-control.sh env-unset <account|cluster <name>|pod <id>> KEY
#   fv-control.sh effective-env <name>         per pod, masked; which pods need a restart
#   fv-control.sh logs <pod> [search] [level]
#   fv-control.sh promote <sha> [channel] [--dry-run] | rollback [channel] [--dry-run]
#   fv-control.sh build-pod up [--region eu|us|<DC>] [--no-wait]
#                                              a build pod from fv-control (reuse / start / create; only fv-control
#                                              creates or deletes them), wait until ready, and write its id and token
#                                              to $FV_BUILD_STATE (~/.config/fv-build) for scripts/dev/build-pod.sh
#   fv-control.sh build-pod status | list      managed build pods: state, region, $/hr, spend, timers, runner
#   fv-control.sh build-pod stop [--force] [<id>]   (default: the one in $FV_BUILD_STATE; refused while busy)
#   fv-control.sh build-pod start <id> | delete [--force] <id> | token [<id>] | plan [region]
#   fv-control.sh build-pod policy [json]      show, or merge a JSON object into, the build_pods policy
#   fv-control.sh endpoint list [--all]       Runpod serverless endpoints fv-control made (docs/control/serverless.md)
#   fv-control.sh endpoint create <spec.json|-> | create <name> [variant]   (defaults: cpu = CPU fake engine)
#   fv-control.sh endpoint show <name|id> | update <name|id> <json> | scale <name|id> <max> [min]
#   fv-control.sh endpoint extend <name|id> <minutes> | delete <name|id> | logs <name|id> [worker]
#   fv-control.sh endpoint invoke <name|id> ['<input json>'] [--async]   queue: /runsync (default {"kind":"info"});
#                                              lb: invoke <name|id> '{"method":"GET","path":"/ping"}'
#   fv-control.sh api <METHOD> <path> [json]  anything else
#   fv-control.sh deploy staging               build, migrate and deploy the Worker (wrangler)
#   fv-control.sh edge-link staging            give the Worker the staging edge (EDGE_URL, EDGE_D1_DATABASE_ID,
#                                              EDGE_OUTPUTS_BUCKET, EDGE_INTERNAL_TOKEN, EDGE_ADMIN_TOKEN) from
#                                              scripts/serve/cf-edge.sh's state (FV_EDGE_STATE); never printed
#
# Env: FV_CONTROL_URL (default https://fv-control-staging.maximalize.workers.dev),
# FV_CONTROL_TOKEN_FILE (default ~/.config/fv/fv-control-token, mode 600; `build-pod up|token`
# need an admin token), FV_BUILD_STATE (default ~/.config/fv-build).
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
# Like api, without failing: sets API_CODE (HTTP status) and API_OUT (body); true for 2xx.
api_s() {
  local method="$1" path="$2" body="${3:-}" out
  [[ -r "$TOKEN_FILE" ]] || die "no API token in $TOKEN_FILE (mint one under Settings)"
  out="$(curl -sS -w '\n%{http_code}' -X "$method" -H @<(printf 'Authorization: Bearer %s\n' "$(tr -d '\n' <"$TOKEN_FILE")") \
    -H 'content-type: application/json' ${body:+-d "$body"} "$URL$path")" || die "fv-control unreachable ($URL)"
  API_CODE="${out##*$'\n'}"
  API_OUT="${out%$'\n'*}"
  [[ "$API_CODE" == 2* ]]
}
api_err() { echo "HTTP $API_CODE $(jq -r '.error // .' <<<"$API_OUT" 2>/dev/null || echo "$API_OUT")"; }
BP_STATE="${FV_BUILD_STATE:-${XDG_CONFIG_HOME:-$HOME/.config}/fv-build}"
bp_log() { echo "[build-pod] $*" >&2; }
# Writes the pod's id and token where build-pod.sh reads them (mode 600, never printed).
bp_save() { # <json with .pod and .token>
  local j="$1"
  (umask 077; mkdir -p "$BP_STATE"
    jq -r '.pod.pod_id' <<<"$j" >"$BP_STATE/pod.new" && mv -f "$BP_STATE/pod.new" "$BP_STATE/pod"
    jq -r '.pod.id' <<<"$j" >"$BP_STATE/pod-managed.new" && mv -f "$BP_STATE/pod-managed.new" "$BP_STATE/pod-managed"
    jq -j '.token' <<<"$j" >"$BP_STATE/token.new" && mv -f "$BP_STATE/token.new" "$BP_STATE/token"
    printf 'Authorization: Bearer %s\n' "$(cat "$BP_STATE/token")" >"$BP_STATE/auth-header.new" && mv -f "$BP_STATE/auth-header.new" "$BP_STATE/auth-header")
}
bp_id() { local id="${1:-}"; [[ -n "$id" ]] || id="$(cat "$BP_STATE/pod-managed" 2>/dev/null || true)"; [[ -n "$id" ]] || die "no build pod given and none in $BP_STATE/pod-managed (build-pod up first)"; echo "$id"; }
bp_line() { jq -r '.pods[]? // .pod | select(. != null) | "\(.id)  \(.name)  \(.pod_id // "-")  \(.state)\(if .phase then "/" + .phase else "" end)  \(.dc // "-") \(.flavor // "")-\(.vcpu // "")  $\(.cost_per_hr // 0)/hr  today $\(.spend_today * 100 | round / 100)  runner \(.runner.state // "none")\(if .outdated then "  OUTDATED" else "" end)\(if .health then "  up \(.health.uptime_s // 0 | . / 60 | floor) min, \(.health.jobs_active // 0) job(s), idle \(.health.idle_s // 0 | . / 60 | floor) min" else "" end)"'; }
cmd_build_pod() {
  local sub="${1:-status}"; shift || true
  local force=false region="" wait=1 out id
  case "$sub" in
    up)
      while (( $# )); do case "$1" in --region) region="${2:?--region eu|us|ca|ap|<DC>}"; shift 2 ;; --no-wait) wait=0; shift ;; *) die "build-pod up [--region R] [--no-wait]" ;; esac; done
      local tries=0
      until api_s POST /api/build-pods/up "$(jq -nc --arg r "$region" 'if $r == "" then {} else {region: $r} end')"; do
        if ! { [[ "$API_CODE" == 409 ]] && jq -e '.error | test("in progress")' <<<"$API_OUT" >/dev/null 2>&1 && (( tries++ < 30 )); }; then
          die "build-pod up: $(api_err)"
        fi
        sleep 10
      done
      out="$API_OUT"
      bp_save "$out"
      bp_log "$(jq -r '"\(.action) \(.pod.name) (\(.pod.pod_id)) in \(.pod.dc) at $\(.pod.cost_per_hr)/hr\(if (.replaced | length) > 0 then "; replaced stopped outdated " + (.replaced | join(" ")) else "" end)"' <<<"$out")"
      id="$(jq -r .pod.id <<<"$out")"
      if (( wait )); then
        local t0=$SECONDS phase="" last=""
        while :; do
          phase="$(api GET "/api/build-pods/$id" | jq -r '.pod.phase // .pod.state')"
          [[ "$phase" != "$last" ]] && { bp_log "pod: $phase"; last="$phase"; }
          [[ "$phase" == ready ]] && break
          [[ "$phase" == failed || "$phase" == stopped || "$phase" == deleted ]] && die "build pod $id is $phase"
          (( SECONDS - t0 < ${FV_BUILD_BOOT_WAIT_S:-1500} )) || die "build pod $id not ready after ${FV_BUILD_BOOT_WAIT_S:-1500}s (last: $phase)"
          sleep 10
        done
        bp_log "ready after $(( SECONDS - t0 ))s: $(jq -r .pod.url <<<"$out")"
      fi ;;
    status | list) api GET /api/build-pods | bp_line ;;
    stop | delete)
      [[ "${1:-}" == --force ]] && { force=true; shift; }
      id="$(bp_id "${1:-}")"
      if [[ "$sub" == stop ]]; then api_s POST "/api/build-pods/$id/stop" "{\"force\": $force}" || die "stop: $(api_err)"
      else api_s DELETE "/api/build-pods/$id$([[ $force == true ]] && echo '?force=1')" || die "delete: $(api_err)"; fi
      bp_line <<<"$API_OUT" ;;
    start) api_s POST "/api/build-pods/$(bp_id "${1:?id}")/start" '{}' || die "start: $(api_err)"; bp_line <<<"$API_OUT" ;;
    token) api_s GET "/api/build-pods/$(bp_id "${1:-}")/token" || die "token: $(api_err)"; bp_save "$API_OUT"; bp_log "token of $(jq -r .pod.name <<<"$API_OUT") written to $BP_STATE" ;;
    plan) api_s GET "/api/build-pods/plan${1:+?region=$1}" || die "plan: $(api_err)"; jq '{server, candidates: [.candidates[] | "\(.flavor)-\(.vcpu) \(.dc) \(.stock) $\(.price)/hr\(if .volume_id then " volume " + .volume_id else "" end)"]}' <<<"$API_OUT" ;;
    policy) if [[ -n "${1:-}" ]]; then api PUT /api/build-pods/policy "$(jq -c '{policy: .}' <<<"$1")" | jq .policy; else api GET /api/build-pods/policy | jq .policy; fi ;;
    *) die "build-pod up|status|list|stop|start|delete|token|plan|policy" ;;
  esac
}
ep_line() { jq -r '.endpoints[]? // .endpoint | select(. != null) | "\(.name)  \(.endpoint_id // "-")  \(.mode)  \(.status)  \(.spec.variant)/\(.spec.compute)  workers \(.workers // 0) (\(.spec.workers_min)..\(.spec.workers_max))  $\(.live_dph // 0)/hr  billed $\(.cost_usd * 1000 | round / 1000)\(if .deadline then "  \(.deadline_action) at " + (.deadline / 1000 | todate) else "" end)\(if .last_error then "  ERROR " + .last_error else "" end)"'; }
cmd_endpoint() {
  local sub="${1:-list}"; shift || true
  local id body
  case "$sub" in
    list) api GET "/api/serverless$([[ "${1:-}" == --all ]] && echo '?all=1')" | ep_line ;;
    create)
      if [[ -f "${1:-}" || "${1:-}" == - ]]; then body="$(jq -c '{spec: .}' "${1/#-//dev/stdin}")"
      else body="$(jq -nc --arg n "${1:?spec.json or a name}" --arg v "${2:-cpu}" '{spec: {name: $n, variant: $v}}')"; fi
      api_s POST /api/serverless "$body" || die "create: $(api_err)"; ep_line <<<"$API_OUT" ;;
    show) api GET "/api/serverless/${1:?name|id}" | jq '{endpoint: (.endpoint | del(.spec)), spec: .endpoint.spec, runpod: .runpod, health, stats, jobs: [.jobs[] | {id, route, status, cold, delay_ms, exec_ms, wall_ms}], costs}' ;;
    update) api_s PUT "/api/serverless/${1:?name|id}" "$(jq -c '{spec: .}' <<<"${2:?json (merged over the spec)}")" || die "update: $(api_err)"; ep_line <<<"$API_OUT" ;;
    scale) api_s POST "/api/serverless/${1:?name|id}/scale" "$(jq -nc --argjson max "${2:?workers_max}" --arg min "${3:-}" '{workers_max: $max} + (if $min == "" then {} else {workers_min: ($min|tonumber)} end)')" || die "scale: $(api_err)"; ep_line <<<"$API_OUT" ;;
    extend) api_s POST "/api/serverless/${1:?name|id}/extend" "$(jq -nc --argjson m "${2:?minutes}" '{minutes: $m}')" || die "extend: $(api_err)"; ep_line <<<"$API_OUT" ;;
    delete) api_s DELETE "/api/serverless/${1:?name|id}" || die "delete: $(api_err)"; ep_line <<<"$API_OUT"; [[ "$API_CODE" == 202 ]] && echo "(deleting: the cron retries every minute)" >&2 ;;
    invoke)
      id="${1:?name|id}"; shift
      local input='{"kind":"info"}' async=false out jid
      while (( $# )); do case "$1" in --async) async=true; shift ;; *) input="$1"; shift ;; esac; done
      # An lb endpoint takes {method, path, body}; a queue endpoint the job input.
      if jq -e 'has("path") and (has("kind") | not)' <<<"$input" >/dev/null 2>&1; then body="$input"
      else body="$(jq -nc --argjson i "$input" --argjson a "$async" '{input: $i, sync: ($a | not)}')"; fi
      api_s POST "/api/serverless/$id/invoke" "$body" || die "invoke: $(api_err)"
      out="$API_OUT"
      if [[ "$(jq -r '.done' <<<"$out")" == false ]]; then
        jid="$(jq -r .job <<<"$out")"
        echo "job $(jq -r .runpod_job <<<"$out") $(jq -r .status <<<"$out"): polling" >&2
        until [[ "$(jq -r '.job.finished_at // "null"' <<<"$out")" != null ]]; do sleep 3; out="$(api GET "/api/serverless/$id/jobs/$jid")"; done
      fi
      jq . <<<"$out" ;;
    logs) api GET "/api/serverless/${1:?name|id}/logs${2:+?worker=$2}" | jq -r '"workers: \(.workers | join(" ")); worker \(.worker // "-")", (.system[]? // empty), (.container[]? // empty), (.stored[]? | "\(.ts / 1000 | todate) \(.msg)")' ;;
    *) die "endpoint list|create|show|update|scale|extend|delete|invoke|logs" ;;
  esac
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
  extend) api POST "/api/clusters/${1:?name}/extend" "$(jq -nc --argjson m "${2:?minutes}" '{minutes: $m}')" | jq -c . ;;
  scale) api POST "/api/clusters/${1:?name}/scale" "$(jq -nc --arg p "${2:?pool}" --argjson n "${3:?count}" '{pool: $p, count: $n}')" | jq -c . ;;
  roll) api POST "/api/clusters/${1:?name}/roll" "$(jq -nc --arg t "${2:?target}" '{target: $t}')" | jq -c . ;;
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
  promote) api POST /api/github/release "$(jq -nc --arg t "${1:?sha}" --arg c "${2:-stable}" --argjson d "$([[ " $* " == *" --dry-run "* ]] && echo true || echo false)" '{action: "promote", target: $t, channel: $c, dry_run: $d}')" | jq . ;;
  rollback) api POST /api/github/release "$(jq -nc --arg c "${1:-stable}" --argjson d "$([[ " $* " == *" --dry-run "* ]] && echo true || echo false)" '{action: "rollback", channel: (if $c == "--dry-run" then "stable" else $c end), dry_run: $d}')" | jq . ;;
  build-pod) cmd_build_pod "$@" ;;
  endpoint) cmd_endpoint "$@" ;;
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
      python3 -c 'import json,sys; d=sys.argv[1]; r=lambda f: open(f"{d}/{f}").read().strip(); print(json.dumps({"EDGE_URL": r("url"), "EDGE_D1_DATABASE_ID": r("d1_id"), "EDGE_INTERNAL_TOKEN": r("internal_token"), "EDGE_ADMIN_TOKEN": r("admin_token"), "EDGE_OUTPUTS_BUCKET": sys.argv[2]}))' "$st" "${FV_EDGE_OUTPUTS_BUCKET:-fv-edge-staging-outputs}" >"$tmp"
      npx wrangler secret bulk "$tmp" --env staging >/dev/null)
    echo "fv-control staging: edge $(cat "$st/url")" ;;
  *) sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
