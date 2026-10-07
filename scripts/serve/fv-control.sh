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
#   fv-control.sh pod launch <name> [--preset P | --variant V (--config PATH | --config-toml FILE)]
#                     [--channel C | --sha S | --image REF] [--gpu TYPE]... [--cpu [FLAVOR]] [--region eu|EUR-IS-1]
#                     [--no-volume] [--env K=V]... [--secret-env K=V|K=@FILE]... [--deadline-min N] [--idle-stop-min N]
#                     [--max-dph X] [--no-start] [--wait] [--json FILE]
#                                              a standalone pod (docs/control/standalone-pods.md): one pod, not part of a
#                                              cluster, with the cluster pods' price check, image preflight, backstops,
#                                              cost ledger (owner pod:<name>) and logs from boot
#   fv-control.sh pod list | status <name> | start <name> | stop <name> | extend <name> <min> | delete <name> | wait <name>
#   fv-control.sh pod logs <name|pod id> [--source runpod|serve] [--follow]
#   fv-control.sh boot <pod id | cluster or standalone name>
#                                              each pod's boot timeline: create, machine, image pull, container, fv-serve,
#                                              volume, weights per component, warm-up, ready (and the phase it is in)
#   fv-control.sh endpoint list [--all]       Runpod serverless endpoints fv-control made (docs/control/serverless.md)
#   fv-control.sh endpoint create <spec.json|-> | create <name> [variant]   (defaults: cpu = CPU fake engine)
#   fv-control.sh endpoint show <name|id> | update <name|id> <json> | scale <name|id> <max> [min]
#   fv-control.sh endpoint extend <name|id> <minutes> | delete <name|id> | logs <name|id> [worker]
#   fv-control.sh endpoint invoke <name|id> ['<input json>'] [--async]   queue: /runsync (default {"kind":"info"});
#                                              lb: invoke <name|id> '{"method":"GET","path":"/ping"}'
#   fv-control.sh endpoint cancel <name|id> <job> [--fv-job ID [--fv-api native|openai_videos|fastwan|minimax_v2]] [--no-fv]
#                                              cancel a job (a Runpod job id, also one fv-control did not submit, or the
#                                              invoke number); a job a worker took also gets the fv-serve cancel (kind http)
#   fv-control.sh endpoint purge <name|id> [--yes]   drop every queued job (shows the count, asks for the name)
#   fv-control.sh jobs <cluster|pod name|pod id> [--status queued,running,…] [--pool P] [--limit N]
#                                              recent and running fv-serve jobs (the jobs D1): status, API, model, times
#   fv-control.sh job-cancel <job> [--cluster NAME]   cancel one fv-serve job by any of its ids (routed to its worker)
#   fv-control.sh jobs-cancel-queued <cluster|pod name> [--pool P] [--yes]   cancel every queued job
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
sp_line() { jq -r '.pods[]? // .pod | select(. != null) | "\(.name)  \(.status)  \(.pod.pod_id // "-")  \(.definition.variant) \(.definition.image | tostring)  \(.pod.gpu // .definition.gpu_types[0]? // .definition.compute) \(.pod.dc // .definition.dc)  $\(.pod.cost_per_hr // 0)/hr  today $\(.cost.today * 100 | round / 100) total $\(.cost.total * 100 | round / 100)  deadline \(if .deadline then (.deadline / 1000 | todate) else "-" end)\(if .pod.boot then "  boot: " + .pod.boot.phase else "" end)\(if .pool_status and .pool_status.status != "ready" and .pool_status.status != "starting" then "  " + .pool_status.status + ": " + (.pool_status.detail // "") else "" end)\(if .op then "  op " + .op.kind + "/" + .op.phase else "" end)"'; }
sp_wait() { # <name>: until no operation runs; prints the last operation's log
  local name="$1" op
  while :; do
    op="$(api GET "/api/standalone/$name" | jq -c '.pod.op')"
    [[ "$op" == null ]] && break
    sleep 10
  done
  api GET "/api/clusters/$name/ops" | jq -r '.operations[0] | "\(.kind) \(.status) \(.error // "")", (.log[] | "  \(.at / 1000 | todate) \(.msg)")'
}
cmd_pod() {
  local sub="${1:-list}"; shift || true
  local name body
  case "$sub" in
    launch)
      name="${1:?pod launch <name> [options]}"; shift
      local json="" wait=0 kv k v
      body="$(jq -nc --arg n "$name" '{name: $n, env: {}}')"
      while (( $# )); do
        case "$1" in
          --preset) body="$(jq -c --arg v "${2:?}" '.preset = $v' <<<"$body")"; shift 2 ;;
          --variant) body="$(jq -c --arg v "${2:?}" '.variant = $v' <<<"$body")"; shift 2 ;;
          --config) body="$(jq -c --arg v "${2:?}" '.config = $v' <<<"$body")"; shift 2 ;;
          --config-toml) body="$(jq -c --rawfile v "${2:?}" '.config_toml = $v' <<<"$body")"; shift 2 ;;
          --channel) body="$(jq -c --arg v "${2:?}" '.channel = $v' <<<"$body")"; shift 2 ;;
          --sha) body="$(jq -c --arg v "${2:?}" '.sha = $v' <<<"$body")"; shift 2 ;;
          --image) body="$(jq -c --arg v "${2:?}" '.image = $v' <<<"$body")"; shift 2 ;;
          --gpu) body="$(jq -c --arg v "${2:?}" '.gpu_types = ((.gpu_types // []) + [$v])' <<<"$body")"; shift 2 ;;
          --cpu) if [[ -n "${2:-}" && "${2:-}" != --* ]]; then body="$(jq -c --arg v "$2" '.compute = "CPU" | .cpu_flavors = [$v]' <<<"$body")"; shift 2; else body="$(jq -c '.compute = "CPU"' <<<"$body")"; shift; fi ;;
          --region) body="$(jq -c --arg v "${2:?}" '.region = $v' <<<"$body")"; shift 2 ;;
          --no-volume) body="$(jq -c '.volume = false' <<<"$body")"; shift ;;
          --env | --secret-env)
            kv="${2:?$1 K=V}"; k="${kv%%=*}"; v="${kv#*=}"
            [[ "$v" == @* ]] && v="$(cat "${v#@}")"
            body="$(jq -c --arg k "$k" --arg v "$v" --argjson s "$([[ "$1" == --secret-env ]] && echo true || echo false)" '.env[$k] = {value: $v, secret: $s}' <<<"$body")"; shift 2 ;;
          --deadline-min) body="$(jq -c --argjson v "${2:?}" '.deadline_min = $v' <<<"$body")"; shift 2 ;;
          --idle-stop-min) body="$(jq -c --argjson v "${2:?}" '.idle_stop_min = $v' <<<"$body")"; shift 2 ;;
          --max-dph) body="$(jq -c --argjson v "${2:?}" '.max_gpu_dph = $v' <<<"$body")"; shift 2 ;;
          --no-start) body="$(jq -c '.start = false' <<<"$body")"; shift ;;
          --wait) wait=1; shift ;;
          --json) json="${2:?}"; shift 2 ;;
          *) die "pod launch: unknown option $1" ;;
        esac
      done
      [[ -n "$json" ]] && body="$(jq -c --slurpfile f "$json" '. * $f[0]' <<<"$body")"
      api_s POST /api/standalone "$body" || die "pod launch: $(api_err)"
      sp_line <<<"$API_OUT"
      jq -r 'if .operation then "operation \(.operation) (fv-control.sh pod wait \(.pod.name))" else empty end' <<<"$API_OUT" >&2
      (( wait )) && sp_wait "$name" ;;
    list) api GET /api/standalone | sp_line ;;
    status) api GET "/api/standalone/${1:?name}" | jq '.pod' ;;
    start) api_s POST "/api/standalone/${1:?name}/start" '{}' || die "start: $(api_err)"; jq -c . <<<"$API_OUT" ;;
    stop) api_s POST "/api/standalone/${1:?name}/stop" '{}' || die "stop: $(api_err)"; jq -c . <<<"$API_OUT" ;;
    extend) api_s POST "/api/standalone/${1:?name}/extend" "$(jq -nc --argjson m "${2:?minutes}" '{minutes: $m}')" || die "extend: $(api_err)"; jq -c . <<<"$API_OUT" ;;
    delete) api_s DELETE "/api/standalone/${1:?name}" || die "delete: $(api_err)"; jq -c . <<<"$API_OUT" ;;
    wait) sp_wait "${1:?name}" ;;
    logs)
      local id="${1:?pod logs <name|pod id>}" src="" follow=0 after=0 out; shift
      while (( $# )); do case "$1" in --source) src="${2:?runpod|serve}"; shift 2 ;; --follow|-f) follow=1; shift ;; *) die "pod logs: unknown option $1" ;; esac; done
      # A standalone pod's name → its current (or last) pod id.
      if ! [[ "$id" =~ ^[a-z0-9]{14}$ ]]; then id="$(api GET "/api/standalone/$id" | jq -r '.pod.pod.pod_id // empty')"; [[ -n "$id" ]] || die "no pod (start it, or pass a pod id)"; fi
      while :; do
        out="$(api GET "/api/pods/$id/logs?limit=500${src:+&source=$src}$( (( after )) && echo "&after_id=$after")")"
        jq -r '.lines[] | "\(.ts / 1000 | todate) \(.level | ascii_upcase) \(.target // ""): \(.msg)"' <<<"$out"
        after="$(jq -r '.next_after_id' <<<"$out")"
        (( follow )) || break
        sleep 5
      done ;;
    *) die "pod launch|list|status|start|stop|extend|delete|wait|logs" ;;
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
    cancel)
      id="${1:?endpoint cancel <name|id> <job>}"; local job="${2:?endpoint cancel <name|id> <job>}"; shift 2
      body='{}'
      while (( $# )); do case "$1" in
        --fv-job) body="$(jq -c --arg v "${2:?--fv-job ID}" '. + {fv_job: $v}' <<<"$body")"; shift 2 ;;
        --fv-api) body="$(jq -c --arg v "${2:?--fv-api API}" '. + {fv_api: $v}' <<<"$body")"; shift 2 ;;
        --no-fv) body="$(jq -c '. + {stop_fv_job: false}' <<<"$body")"; shift ;;
        *) die "endpoint cancel: unknown option $1" ;;
      esac; done
      api_s POST "/api/serverless/$id/jobs/$job/cancel" "$body" || die "cancel: $(api_err)"
      jq -r '"\(.runpod_job): \(.before) -> \(.status)\(if .cancelled then " (cancelled)" else "" end); \(.note)",
        (if .fv_job then "fv-serve job \(.fv_job.api) \(.fv_job.id): \(if .fv_cancel.sent then "\(.fv_cancel.route) sent as queue job \(.fv_cancel.runpod_job) (fv-control #\(.fv_cancel.job)); " else "" end)\(.fv_cancel.reason)" else "fv-serve: \(.fv_cancel.reason)" end)' <<<"$API_OUT" ;;
    purge)
      id="${1:?endpoint purge <name|id>}"; local yes=0 q n name
      [[ "${2:-}" == --yes ]] && yes=1
      name="$(api GET "/api/serverless/$id" | jq -r .endpoint.name)"
      q="$(api GET "/api/serverless/$id/queue")"; n="$(jq -r .queued <<<"$q")"
      echo "$name: $n queued, $(jq -r .in_progress <<<"$q") running (a purge drops the queued ones only)" >&2
      if (( ! yes )); then
        [[ -t 0 ]] || die "purge: pass --yes (no terminal to confirm on)"
        read -r -p "type the endpoint name to purge its queue: " ans; [[ "$ans" == "$name" ]] || die "purge: not confirmed"
      fi
      api_s POST "/api/serverless/$id/purge" "$(jq -nc --arg c "$name" --argjson e "$n" '{confirm: $c, expected: $e}')" || die "purge: $(api_err)"
      jq -r '"removed \(.removed // "?") (Runpod: \(.status // "?")); queued \(.queued_before) -> \(.queued_after // "?"), running \(.in_progress // "?"); \(.note)"' <<<"$API_OUT" ;;
    *) die "endpoint list|create|show|update|scale|extend|delete|invoke|logs|cancel|purge" ;;
  esac
}
# A cluster or standalone pod name, or a pod id -> "<cluster id>[&pod=<pod id>]".
jobs_scope() {
  local cid=""
  if [[ "$1" =~ ^[a-z0-9]{14}$ ]]; then
    if api_s GET "/api/pods/$1"; then cid="$(jq -r '.controller.cluster_id // empty' <<<"$API_OUT")"; fi
    # Not collected yet: a live worker in some cluster's state.
    [[ -n "$cid" ]] || cid="$(api GET "/api/clusters?all=1" | jq -r --arg p "$1" 'first(.clusters[] | select([.state.workers[]?[]?.pod] | index($p)) | .id) // empty')"
  fi
  if [[ -n "$cid" ]]; then
    echo "$cid&pod=$1"
  else
    api_s GET "/api/clusters/$1" || die "no cluster, standalone pod or controller pod $1"
    jq -r .cluster.id <<<"$API_OUT"
  fi
}
job_line() { jq -r '.jobs[] | [.external_id, .api, .status + (if .cancel_requested and .status == "running" then " (cancel requested)" else "" end), .model, (.pool // "-"), (.worker // "edge queue"), (.created_at / 1000 | floor | todate), (if .started_at then (.started_at / 1000 | floor | todate) else "-" end)] | @tsv'; }
cmd_jobs() {
  local who="${1:?jobs <cluster|pod name|pod id> [--status …] [--pool P] [--limit N]}"; shift
  local qs="" sc cid out
  while (( $# )); do case "$1" in
    --status) qs+="&status=${2:?--status queued,running,…}"; shift 2 ;;
    --pool) qs+="&pool=${2:?--pool P}"; shift 2 ;;
    --limit) qs+="&limit=${2:?--limit N}"; shift 2 ;;
    *) die "jobs: unknown option $1" ;;
  esac; done
  sc="$(jobs_scope "$who")"; cid="${sc%%&*}"; [[ "$sc" == *"&"* ]] && qs+="&${sc#*&}"
  api_s GET "/api/clusters/$cid/jobs?${qs#&}" || die "jobs: $(api_err)"
  out="$API_OUT"
  jq -r '"\(.cluster.name) (\(.cluster.kind), \(.cluster.control_plane); jobs D1: \(.source)): " + ([.counts | to_entries[] | "\(.value) \(.key)"] | join(", "))' <<<"$out" >&2
  printf 'JOB\tAPI\tSTATUS\tMODEL\tPOOL\tWORKER\tSUBMITTED\tSTARTED\n'
  job_line <<<"$out"
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
  pod) cmd_pod "$@" ;;
  boot)
    t="${1:?boot <pod id | cluster or standalone name>}"
    if [[ "$t" =~ ^[a-z0-9]{14}$ ]]; then ids="$t"; else ids="$(api GET "/api/clusters/$t" | jq -r '.cluster.state.workers // {} | [.[][] | .pod] | join(" ")')"; fi
    [[ -n "$ids" ]] || die "no pods (a pod id works for deleted pods too)"
    for id in $ids; do
      api GET "/api/pods/$id/boot" | jq -r '"\(.pod_id)\(if .phase then "  (now: \(.phase.phase) for \(.phase.since_s) s)" else "" end)", (.timeline[] | select(.at != null) | "  \(.t_s | tostring | (" " * (7 - length)) + .) s  \(if .took_s != null then "+\(.took_s) s" else "" end | . + (" " * (10 - length)))  \(.phase)\(if .detail then "  (\(.detail))" else "" end)")'
    done ;;
  endpoint) cmd_endpoint "$@" ;;
  jobs) cmd_jobs "$@" ;;
  job-cancel)
    jid="${1:?job-cancel <job> [--cluster NAME]}"; shift
    body='{}'
    [[ "${1:-}" == --cluster ]] && body="$(jq -nc --arg c "${2:?--cluster NAME}" '{cluster: $c}')"
    api_s POST "/api/jobs/$jid/cancel" "$body" || die "job-cancel: $(api_err)"
    jq -r '"\(.external_id) (\(.api), \(.cluster)): \(.status)\(if .cancel_requested then ", cancel requested" else "" end) via \(.via // "-")\(if .pod then " on \(.pod)" else "" end); \(.note)"' <<<"$API_OUT" ;;
  jobs-cancel-queued)
    who="${1:?jobs-cancel-queued <cluster|pod name> [--pool P] [--yes]}"; shift
    pool="" yes=0
    while (( $# )); do case "$1" in --pool) pool="${2:?--pool P}"; shift 2 ;; --yes) yes=1; shift ;; *) die "jobs-cancel-queued: unknown option $1" ;; esac; done
    cid="$(jobs_scope "$who")"; cid="${cid%%&*}"
    api_s GET "/api/clusters/$cid/jobs?status=queued&limit=500${pool:+&pool=$pool}" || die "jobs: $(api_err)"
    n="$(jq '.jobs | length' <<<"$API_OUT")"
    echo "$who: $n queued job(s)${pool:+ in pool $pool}" >&2
    (( n )) || exit 0
    if (( ! yes )); then [[ -t 0 ]] || die "pass --yes (no terminal to confirm on)"; read -r -p "cancel all $n? [y/N] " ans; [[ "$ans" == y* ]] || die "not confirmed"; fi
    api_s POST "/api/clusters/$cid/jobs/cancel-queued" "$(jq -nc --arg p "$pool" '(if $p == "" then {} else {pool: $p} end)')" || die "cancel-queued: $(api_err)"
    jq -r '"cancelled \(.cancelled) of \(.queued)", (.failed[] | "  not cancelled: \(.job) (\(.api)): \(.note)")' <<<"$API_OUT" ;;
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
  *) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
