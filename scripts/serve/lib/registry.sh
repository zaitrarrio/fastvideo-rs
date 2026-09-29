# shellcheck shell=bash
# Release history and the deployment registry in D1 (docs/serve/releases.md;
# schema: deploy/d1/registry.sql). Source after scripts/gpu/lib.sh; sources
# d1.sh itself.
#
# Deployments (every deploy script, best effort: a D1 failure logs a warning
# and never fails the deploy):
#   fv_deploy_created <kind> <runpod id> [key=value …]   status creating
#   fv_deploy_ready   <kind> <runpod id> [key=value …]   status ready, ready_at
#   fv_deploy_deleted <kind> <runpod id> [status]        deleted_at (status deleted | gone)
#   fv_deploy_update  <kind> <runpod id> key=value …
#     keys: name pool variant image digest git_sha channel region dc gpu
#           cost_per_hr status meta (JSON); image=…@sha256:… fills digest,
#           and git_sha / channel from the releases table when not given.
#   kind: pod | endpoint | gateway
#
# Releases:
#   fv_release_current <channel>     the channel's current release row (JSON) or nothing
#   fv_release_rows [channel] [n]    newest first (JSON array)
#   fv_release_insert <json>         a row: {channel, git_sha, digests, action, notes, run_url,
#                                    templates_updated, source_release}; prints the id
#   fv_release_for_digest <digest>   {git_sha, key, channels:[…]} of the releases naming it
#
# FV_REGISTRY=0 turns the deployment writes off. FV_DEPLOYED_BY overrides
# the created_by / promoted_by word (default: ci:<workflow>#<run> in GitHub
# Actions, agent:<user> under Claude Code, else user:<user>@<host>).

# shellcheck source-path=SCRIPTDIR source=d1.sh
source "$(dirname "${BASH_SOURCE[0]}")/d1.sh"
FV_REGISTRY_SQL="${FV_REGISTRY_SQL:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/deploy/d1/registry.sql}"

fv_now_ms() { echo $(( $(date +%s) * 1000 )); }

fv_actor() {
  if [[ -n "${FV_DEPLOYED_BY:-}" ]]; then echo "$FV_DEPLOYED_BY"
  elif [[ "${GITHUB_ACTIONS:-}" == true ]]; then echo "ci:${GITHUB_WORKFLOW:-?}#${GITHUB_RUN_ID:-?}${GITHUB_ACTOR:+ by $GITHUB_ACTOR}"
  elif [[ -n "${CLAUDECODE:-}" ]]; then echo "agent:$(id -un 2>/dev/null || echo ?)"
  else echo "user:$(id -un 2>/dev/null || echo ?)@$(hostname -s 2>/dev/null || echo ?)"
  fi
}

# The schema, once per process tree.
fv_registry_schema() {
  [[ "${_FV_REGISTRY_SCHEMA_OK:-}" == 1 ]] && return 0
  fv_d1_exec_file "$FV_REGISTRY_SQL" || { log "D1: could not apply $FV_REGISTRY_SQL"; return 1; }
  export _FV_REGISTRY_SCHEMA_OK=1
}

_fv_registry_on() {
  [[ "${FV_REGISTRY:-1}" != 0 ]] || return 1
  if ! fv_d1_available; then
    [[ -n "${_FV_REGISTRY_WARNED:-}" ]] || log "registry: no D1 token; deployments are not recorded (docs/serve/releases.md)"
    export _FV_REGISTRY_WARNED=1
    return 1
  fi
  fv_registry_schema
}

# key=value args -> JSON object (values are strings; cost_per_hr a number,
# meta JSON).
_fv_kv_json() {
  local out='{}' kv k v
  for kv in "$@"; do
    k="${kv%%=*}"; v="${kv#*=}"
    [[ "$k" != "$kv" ]] || continue
    case "$k" in
      name | pool | variant | image | digest | git_sha | channel | region | dc | gpu | status) ;;
      cost_per_hr) [[ "$v" =~ ^[0-9.]+$ ]] || continue ;;
      meta) jq -e 'type == "object"' >/dev/null 2>&1 <<<"$v" || continue ;;
      *) log "registry: ignoring unknown field $k"; continue ;;
    esac
    [[ -n "$v" ]] || continue
    if [[ "$k" == cost_per_hr ]]; then out="$(jq -c --arg k "$k" --argjson v "$v" '. + {($k): $v}' <<<"$out")"
    else out="$(jq -c --arg k "$k" --arg v "$v" '. + {($k): $v}' <<<"$out")"
    fi
  done
  echo "$out"
}

# Fills digest / git_sha / channel / variant from the image and the releases.
_fv_enrich() {
  local f="$1" img digest hit
  img="$(jq -r '.image // empty' <<<"$f")"
  if [[ -z "$(jq -r '.digest // empty' <<<"$f")" && "$img" == *@sha256:* ]]; then
    f="$(jq -c --arg d "${img##*@}" '. + {digest: $d}' <<<"$f")"
  fi
  digest="$(jq -r '.digest // empty' <<<"$f")"
  if [[ -n "$digest" ]] && { [[ -z "$(jq -r '.git_sha // empty' <<<"$f")" ]] || [[ -z "$(jq -r '.variant // empty' <<<"$f")" ]]; }; then
    hit="$(fv_release_for_digest "$digest" 2>/dev/null || true)"
    if [[ -n "$hit" ]]; then
      f="$(jq -c --argjson h "$hit" '{git_sha: $h.git_sha, variant: $h.key} + (if ($h.channels | length) > 0 then {channel: $h.channels[0]} else {} end) + .' <<<"$f")"
    fi
  fi
  [[ -z "${FV_RELEASE_CHANNEL:-}" ]] || f="$(jq -c --arg c "$FV_RELEASE_CHANNEL" '{channel: $c} + .' <<<"$f")"
  echo "$f"
}

# _fv_deploy_upsert <kind> <id> <fields json> <status or ""> <extra set sql>
_fv_deploy_upsert() {
  local kind="$1" rid="$2" f="$3" now cols vals sets params
  now="$(fv_now_ms)"
  f="$(jq -c --arg id "$kind:$rid" --arg kind "$kind" --arg rid "$rid" --argjson now "$now" --arg by "$(fv_actor)/${FV_DEPLOY_SCRIPT:-$(basename "$0")}" \
    '{id: $id, kind: $kind, runpod_id: $rid, created_at: $now, updated_at: $now, created_by: $by, status: "creating"} + .' <<<"$f")"
  cols="$(jq -r 'keys_unsorted | join(", ")' <<<"$f")"
  vals="$(jq -r '[keys_unsorted[] | "?"] | join(", ")' <<<"$f")"
  # On conflict: every given column but the identity and creation ones.
  sets="$(jq -r '[keys_unsorted[] | select(IN("id", "kind", "runpod_id", "created_at", "created_by") | not) | "\(.) = excluded.\(.)"] | join(", ")' <<<"$f")"
  params="$(jq -c '[.[]]' <<<"$f")"
  fv_d1_query "INSERT INTO deployments ($cols) VALUES ($vals) ON CONFLICT (id) DO UPDATE SET $sets" "$params" >/dev/null
}

fv_deploy_created() {
  local kind="$1" rid="$2"; shift 2
  _fv_registry_on || return 0
  local f
  f="$(_fv_enrich "$(_fv_kv_json "$@")")"
  _fv_deploy_upsert "$kind" "$rid" "$f" || log "registry: could not record $kind $rid"
  return 0
}

fv_deploy_update() {
  local kind="$1" rid="$2"; shift 2
  _fv_registry_on || return 0
  local f sets params
  f="$(_fv_kv_json "$@")"
  f="$(jq -c --argjson now "$(fv_now_ms)" '. + {updated_at: $now}' <<<"$f")"
  sets="$(jq -r '[keys_unsorted[] | "\(.) = ?"] | join(", ")' <<<"$f")"
  params="$(jq -c --arg id "$kind:$rid" '[.[]] + [$id]' <<<"$f")"
  fv_d1_query "UPDATE deployments SET $sets WHERE id = ?" "$params" >/dev/null || log "registry: could not update $kind $rid"
  return 0
}

fv_deploy_ready() {
  local kind="$1" rid="$2"; shift 2
  _fv_registry_on || return 0
  fv_deploy_update "$kind" "$rid" status=ready "$@"
  fv_d1_query "UPDATE deployments SET ready_at = COALESCE(ready_at, ?) WHERE id = ?" "[$(fv_now_ms), \"$kind:$rid\"]" >/dev/null \
    || log "registry: could not mark $kind $rid ready"
  return 0
}

fv_deploy_deleted() {
  local kind="$1" rid="$2" status="${3:-deleted}"
  _fv_registry_on || return 0
  local now; now="$(fv_now_ms)"
  fv_d1_query "UPDATE deployments SET status = ?, deleted_at = COALESCE(deleted_at, ?), updated_at = ? WHERE id = ?" \
    "$(jq -nc --arg s "$status" --argjson n "$now" --arg id "$kind:$rid" '[$s, $n, $n, $id]')" >/dev/null \
    || log "registry: could not mark $kind $rid $status"
  return 0
}

# --- releases ------------------------------------------------------------------

fv_release_current() {
  fv_d1_query "SELECT * FROM releases WHERE channel = ? ORDER BY id DESC LIMIT 1" "$(jq -nc --arg c "$1" '[$c]')" | jq -c '.[0] // empty'
}

fv_release_rows() {
  local ch="${1:-}" n="${2:-20}"
  if [[ -n "$ch" ]]; then
    fv_d1_query "SELECT * FROM releases WHERE channel = ? ORDER BY id DESC LIMIT ?" "$(jq -nc --arg c "$ch" --argjson n "$n" '[$c, $n]')"
  else
    fv_d1_query "SELECT * FROM releases ORDER BY id DESC LIMIT ?" "[$n]"
  fi
}

# The current release of every channel (JSON array).
fv_release_heads() {
  fv_d1_query "SELECT r.* FROM releases r JOIN (SELECT channel, MAX(id) AS id FROM releases GROUP BY channel) h ON r.id = h.id ORDER BY r.channel"
}

fv_release_insert() {
  local row="$1" params
  params="$(jq -c --argjson now "$(fv_now_ms)" --arg by "$(fv_actor)" '[.channel, .git_sha, (.digests | tojson), .action, $now, (.promoted_by // $by),
    .notes, .run_url, (if .templates_updated then 1 else 0 end), .source_release]' <<<"$row")"
  fv_d1_query "INSERT INTO releases (channel, git_sha, digests, action, promoted_at, promoted_by, notes, run_url, templates_updated, source_release)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id" "$params" | jq -r '.[0].id'
}

# fv_release_for_digest <sha256:…> -> {git_sha, key, channels} or nothing.
fv_release_for_digest() {
  local d="$1" rows
  rows="$(fv_d1_query "SELECT id, channel, git_sha, digests FROM releases WHERE digests LIKE ? ORDER BY id DESC LIMIT 50" \
    "$(jq -nc --arg d "%$d%" '[$d]')")" || return 1
  local heads
  heads="$(fv_release_heads 2>/dev/null || echo '[]')"
  jq -c --arg d "$d" --argjson heads "$heads" '
    (.[0] // empty) as $r
    | ($r.digests | fromjson | to_entries | map(select(.value | endswith("@" + $d))) | .[0].key) as $key
    | {git_sha: $r.git_sha, key: $key,
       channels: [$heads[] | select(.digests | fromjson | to_entries | any(.value | endswith("@" + $d))) | .channel]}' <<<"$rows"
}
