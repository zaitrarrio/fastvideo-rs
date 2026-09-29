#!/usr/bin/env bash
# Release channels and deployments of fv-serve (docs/serve/releases.md).
#
#   release.sh list [--verify]              the current release of every channel, digests per image
#                                           (--verify: the channel tags in GHCR still match)
#   release.sh history [--channel C] [--limit N]
#   release.sh deployed [--probe] [--json]  live fv-serve pods / endpoints: version (git sha),
#                                           age, the channels naming their digest, drift vs the
#                                           channel they follow (--probe: ask each pod's /health)
#   release.sh promote <sha|digest|tag> [channel]
#                                           point :<channel> and :<variant>-<channel> at that
#                                           build (no rebuild), update the Runpod templates when
#                                           <channel> is the templates' channel, record it in D1
#   release.sh rollback [channel] [--to ID] re-promote the channel's previous release
#   release.sh redeploy <pool|all|gateway> [channel|sha]
#                                           rolling: a new worker on the target build, wait for
#                                           ready, drain the old one, delete it (the standing
#                                           cluster of runpod-cluster.sh)
#   release.sh reconcile [--dry-run] [--adopt] [--fix]
#                                           match the account's pods / endpoints / templates to
#                                           the deployments table: gone rows are marked deleted,
#                                           unknown and drifted resources are flagged (--adopt
#                                           records unknown ones, --fix records the live image)
#   release.sh resolve <sha|digest|tag>     the image set of one build (JSON; no change)
#   release.sh record-build <channel> <sha> <variants.tsv> [debug ref]
#                                           CI: a main build recorded as the `latest` release
#
# Channels: `stable` (what deploys and the Runpod templates default to) and
# `latest` (the newest green main build: CI moves it; promoting to it is for
# repairs). Any other lower-case word works as a channel.
#
# promote / rollback run where they are asked: in GitHub Actions (or with
# --local) they retag with `docker buildx imagetools create` (needs a GHCR
# login with packages:write) and call the Runpod and D1 APIs here; elsewhere
# they dispatch .github/workflows/release.yml through the GitHub API
# (GH_TOKEN / GITHUB_TOKEN with actions:write) unless --local. --dry-run
# prints the plan and changes nothing, anywhere.
# Flags: --dry-run --local --dispatch --notes TEXT --no-templates
#        --allow-partial (a build missing some variant images) --force
#
# Env: RUNPOD_API_KEY; D1 (scripts/serve/lib/d1.sh: FV_CF_API_TOKEN or
# CLOUDFLARE_API_TOKEN / CLOUDFLARE_API_KEY, FV_CF_ACCOUNT_ID, FV_D1_DATABASE_ID);
# FV_TEMPLATE_CHANNEL (default stable: the channel the Runpod templates
# follow); FV_SERVE_REPO; FV_GITHUB_REPO (default zaitrarrio/fastvideo-rs);
# FV_RELEASE_REF (the branch the workflow runs from, default main).
# Secrets are never printed: tokens reach curl through header descriptors,
# and only named fields of Runpod objects (never their env) are shown.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/lib.sh
source "$HERE/../gpu/lib.sh"
# shellcheck source-path=SCRIPTDIR source=variants.sh
source "$HERE/variants.sh"
# shellcheck source-path=SCRIPTDIR source=lib/ghcr.sh
source "$HERE/lib/ghcr.sh"
# shellcheck source-path=SCRIPTDIR source=lib/registry.sh
source "$HERE/lib/registry.sh"
export FV_DEPLOY_SCRIPT=release.sh

REST="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GH_API="${FV_GITHUB_API:-https://api.github.com}"
GH_REPO="${FV_GITHUB_REPO:-zaitrarrio/fastvideo-rs}"
TEMPLATE_CHANNEL="${FV_TEMPLATE_CHANNEL:-stable}"
STATE="${FV_CLUSTER_STATE:-$FV_ROOT/artifacts/runpod/serve/cluster.json}"

# ---- flags ------------------------------------------------------------------
DRY=0 MODE="" NOTES="" TEMPLATES=1 PARTIAL=0 FORCE=0 TO="" CHANNEL_OPT="" LIMIT=20 PROBE=0 JSON=0 VERIFY=0 ADOPT=0 FIX=0 TPL_UPDATED=0
ARGS=()
parse() {
  while (($#)); do
    case "$1" in
      --dry-run) DRY=1 ;;
      --local) MODE=local ;;
      --dispatch) MODE=dispatch ;;
      --notes) NOTES="${2:?--notes TEXT}"; shift ;;
      --no-templates) TEMPLATES=0 ;;
      --allow-partial) PARTIAL=1 ;;
      --force) FORCE=1 ;;
      --to) TO="${2:?--to ID}"; shift ;;
      --channel) CHANNEL_OPT="${2:?--channel C}"; shift ;;
      --limit) LIMIT="${2:?--limit N}"; shift ;;
      --probe) PROBE=1 ;;
      --json) JSON=1 ;;
      --verify) VERIFY=1 ;;
      --adopt) ADOPT=1 ;;
      --fix) FIX=1 ;;
      --templates-updated) TPL_UPDATED=1 ;;
      --) shift; ARGS+=("$@"); break ;;
      -*) die "unknown flag $1" ;;
      *) ARGS+=("$1") ;;
    esac
    shift
  done
  [[ "$LIMIT" =~ ^[0-9]+$ ]] || die "--limit takes a number"
  [[ -z "$TO" || "$TO" =~ ^[0-9]+$ ]] || die "--to takes a release id"
  if [[ -z "$MODE" ]]; then
    if [[ "${GITHUB_ACTIONS:-}" == true ]]; then MODE=local; else MODE=dispatch; fi
  fi
}

check_channel() {
  local c="$1" v
  [[ "$c" =~ ^[a-z][a-z0-9-]{1,30}$ ]] || die "channel '$c': lower-case letters, digits and '-'"
  [[ "$c" != sha-* && "$c" != buildcache* && "$c" != *-sha-* ]] || die "channel '$c' collides with the build tags"
  for v in $FV_VARIANTS; do [[ "$c" != "$v" ]] || die "channel '$c' is a variant name"; done
}

rp() {
  curl -sS --fail-with-body --max-time 60 -X "$1" -H @<(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY") \
    -H 'content-type: application/json' ${3:+--data-binary "$3"} "$REST$2"
}

short() { echo "${1:0:7}"; }
dshort() { local d="${1##*@}"; echo "${d:7:12}"; }
iso() { if [[ -z "${1:-}" || "$1" == null ]]; then echo "-"; else date -u -d "@$(( $1 / 1000 ))" +%FT%TZ; fi; }
age() {
  local s=$(( $(date +%s) - $1 ))
  if ((s < 3600)); then echo "$((s / 60))m"; elif ((s < 172800)); then echo "$((s / 3600))h"; else echo "$((s / 86400))d"; fi
}
need_d1() { fv_d1_available || die "D1 is the release history: set FV_CF_API_TOKEN (or CLOUDFLARE_API_TOKEN)"; fv_registry_schema || die "D1 schema"; }

# ---- resolving a build ------------------------------------------------------

# resolve <sha|digest|tag> -> {sha, short, digests}
resolve() {
  local in="$1" d="" labels rev s imgs k want got
  fv_ghcr_init || die "cannot read $FV_SERVE_REPO"
  if [[ "$in" =~ ^[0-9a-f]{7,40}$ ]]; then
    s="$(short "$in")"
  else
    if [[ "$in" == *sha256:* ]]; then d="sha256:${in##*sha256:}"
    else d="$(fv_ghcr_digest "${in##*:}")" || die "no tag ${in##*:} in $FV_SERVE_REPO"
    fi
    labels="$(fv_ghcr_labels "$d")" || die "cannot read the labels of $d"
    rev="$(jq -r '."org.opencontainers.image.revision" // empty' <<<"$labels")"
    [[ "$rev" =~ ^[0-9a-f]{7,40}$ ]] || die "$in: no git revision label"
    s="$(short "$rev")"
  fi
  imgs="$(fv_ghcr_set_for_sha "$s")"
  [[ "$imgs" != '{}' ]] || die "no image tagged sha-$s or <variant>-sha-$s in $FV_SERVE_REPO"
  # Every image of the set must carry the revision; the first full sha wins.
  rev=""
  for k in $(jq -r 'keys[]' <<<"$imgs"); do
    want="$(jq -r --arg k "$k" '.[$k]' <<<"$imgs")"
    got="$(fv_ghcr_labels "${want##*@}" | jq -r '."org.opencontainers.image.revision" // empty')" || got=""
    [[ -z "$got" || "$got" == unknown || "$got" == "$s"* ]] || die "$k ($want) is labelled $got, not $s"
    [[ -n "$rev" || ! "$got" =~ ^[0-9a-f]{40}$ ]] || rev="$got"
  done
  jq -nc --arg sha "${rev:-$s}" --arg s "$s" --argjson imgs "$imgs" '{sha: $sha, short: $s, digests: $imgs}'
}

missing_keys() { local k; for k in $FV_RELEASE_KEYS; do jq -e --arg k "$k" 'has($k)' >/dev/null <<<"$1" || echo "$k"; done; }

template_image() { [[ -n "${RUNPOD_API_KEY:-}" ]] && fv_template_get "$(fv_template_name "$1" "$2")" 2>/dev/null | jq -r '.imageName // empty' || true; }

# ---- promote / rollback -----------------------------------------------------

# apply <channel> <digests json> <action> <sha> [source release id]: the
# shared half of promote and rollback (local mode).
apply() {
  local ch="$1" digests="$2" action="$3" sha="$4" src="${5:-}" k ref tag v f cur tpl_ok=0 id
  log "$action ${sha:0:12} -> $ch"
  for k in $(jq -r 'keys[]' <<<"$digests"); do
    ref="$(jq -r --arg k "$k" '.[$k]' <<<"$digests")"
    for tag in $(fv_key_channel_tags "$k" "$ch"); do
      if ((DRY)); then echo "  retag  :$tag -> $ref"
      else fv_ghcr_retag "$ref" "$tag" || die "retag :$tag failed (nothing recorded; rerun to finish)"; log "  :$tag -> $(dshort "$ref")"
      fi
    done
  done
  if ((TEMPLATES)) && [[ "$ch" == "$TEMPLATE_CHANNEL" ]]; then
    for v in $FV_VARIANTS; do
      ref="$(jq -r --arg k "$v" '.[$k] // empty' <<<"$digests")"
      [[ -n "$ref" ]] || continue
      for f in $(fv_variant_flavours "$v"); do
        cur="$(template_image "$v" "$f")"
        if ((DRY)); then
          if [[ "$cur" == "$ref" ]]; then echo "  template $(fv_template_name "$v" "$f"): image already $(dshort "$ref"); env gains FV_RELEASE_CHANNEL=$ch"
          else echo "  template $(fv_template_name "$v" "$f"): ${cur:-?} -> $ref"
          fi
        fi
      done
      if ! ((DRY)); then
        : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing (the templates follow $TEMPLATE_CHANNEL; --no-templates skips them)}"
        FV_RELEASE_CHANNEL="$ch" bash "$HERE/runpod-templates.sh" sync "$v" "$ref" || die "template sync $v failed"
      fi
    done
    tpl_ok=1
  elif [[ "$ch" == "$TEMPLATE_CHANNEL" ]]; then
    log "  templates left as they are (--no-templates)"
  else
    log "  templates follow $TEMPLATE_CHANNEL, not $ch: unchanged"
  fi
  if ((DRY)); then echo "(dry run: nothing changed)"; return 0; fi
  if [[ "$action" == rollback ]]; then
    fv_d1_query "UPDATE releases SET rolled_back_at = ? WHERE channel = ? AND git_sha != ? AND rolled_back_at IS NULL AND id > ?" \
      "$(jq -nc --argjson n "$(fv_now_ms)" --arg c "$ch" --arg s "$sha" --argjson src "$src" '[$n, $c, $s, $src]')" >/dev/null \
      || log "WARNING: could not mark the rolled-back releases"
  fi
  id="$(fv_release_insert "$(jq -nc --arg c "$ch" --arg s "$sha" --argjson d "$digests" --arg a "$action" --arg n "$NOTES" \
    --arg run "${GITHUB_SERVER_URL:+$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/${GITHUB_RUN_ID:-}}" --argjson t "$tpl_ok" \
    --arg src "$src" '{channel: $c, git_sha: $s, digests: $d, action: $a, notes: (if $n == "" then null else $n end),
      run_url: (if $run == "" then null else $run end), templates_updated: ($t == 1),
      source_release: (if $src == "" then null else ($src | tonumber) end)}')")" || die "the images moved but the release was not recorded in D1"
  echo "release $id: $ch = $(short "$sha") ($action)"
  [[ -z "${GITHUB_STEP_SUMMARY:-}" ]] || {
    echo "### $action: \`$ch\` = \`$(short "$sha")\` (release $id)"; echo
    echo "| image | digest |"; echo "|---|---|"
    jq -r 'to_entries[] | "| \(.key) | `\(.value)` |"' <<<"$digests"
  } >>"$GITHUB_STEP_SUMMARY"
}

dispatch() {
  local action="$1" target="$2" ch="$3" tok body code
  tok="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
  [[ -n "$tok" ]] || die "dispatching the release workflow needs GH_TOKEN (actions:write); or run with --local"
  body="$(jq -nc --arg ref "${FV_RELEASE_REF:-main}" --arg a "$action" --arg t "$target" --arg c "$ch" --arg n "$NOTES" --arg to "$TO" \
    --arg tpl "$( ((TEMPLATES)) && echo true || echo false)" --arg p "$( ((PARTIAL)) && echo true || echo false)" \
    '{ref: $ref, inputs: {action: $a, target: $t, channel: $c, notes: $n, to: $to, templates: $tpl, allow_partial: $p}}')"
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 -X POST -H @<(printf 'Authorization: Bearer %s\n' "$tok") \
    -H 'Accept: application/vnd.github+json' --data-binary "$body" "$GH_API/repos/$GH_REPO/actions/workflows/release.yml/dispatches")"
  [[ "$code" == 204 ]] || die "workflow dispatch answered HTTP $code"
  echo "dispatched release.yml: $action ${target:-} -> $ch"
  echo "https://github.com/$GH_REPO/actions/workflows/release.yml"
}

cmd_promote() {
  local target="${ARGS[0]:?promote <sha|digest|tag> [channel]}" ch="${ARGS[1]:-${CHANNEL_OPT:-stable}}" rel miss cur
  check_channel "$ch"
  if [[ "$MODE" == dispatch && "$DRY" == 0 ]]; then dispatch promote "$target" "$ch"; return; fi
  ((DRY)) || need_d1
  rel="$(resolve "$target")"
  miss="$(missing_keys "$(jq -c .digests <<<"$rel")" | xargs)"
  if [[ -n "$miss" ]]; then
    ((PARTIAL)) || die "build $(jq -r .short <<<"$rel") has no image for: $miss (--allow-partial promotes the rest)"
    log "WARNING: no image for $miss; those keep their current :$ch"
  fi
  if fv_d1_available && cur="$(fv_release_current "$ch" 2>/dev/null)" && [[ -n "$cur" ]]; then
    log "current $ch: release $(jq -r .id <<<"$cur") = $(short "$(jq -r .git_sha <<<"$cur")") ($(jq -r .action <<<"$cur"), $(iso "$(jq -r .promoted_at <<<"$cur")"))"
    if [[ "$(jq -r .digests <<<"$cur" | jq -S -c .)" == "$(jq -S -c .digests <<<"$rel")" ]] && ((! FORCE)); then
      echo "$ch is already $(jq -r .short <<<"$rel") (--force re-applies)"; return 0
    fi
  fi
  apply "$ch" "$(jq -c .digests <<<"$rel")" promote "$(jq -r .sha <<<"$rel")"
}

cmd_rollback() {
  local ch="${ARGS[0]:-${CHANNEL_OPT:-stable}}" cur target k ref
  check_channel "$ch"
  if [[ "$MODE" == dispatch && "$DRY" == 0 ]]; then dispatch rollback "" "$ch"; return; fi
  need_d1
  cur="$(fv_release_current "$ch")"
  [[ -n "$cur" ]] || die "no release recorded for $ch"
  if [[ -n "$TO" ]]; then
    target="$(fv_d1_query "SELECT * FROM releases WHERE id = ? AND channel = ?" "$(jq -nc --argjson i "$TO" --arg c "$ch" '[$i, $c]')" | jq -c '.[0] // empty')"
    [[ -n "$target" ]] || die "no release $TO in $ch"
  else
    target="$(fv_d1_query "SELECT * FROM releases WHERE channel = ? AND id < ? AND git_sha != ? AND rolled_back_at IS NULL ORDER BY id DESC LIMIT 1" \
      "$(jq -c '[.channel, .id, .git_sha]' <<<"$cur")" | jq -c '.[0] // empty')"
    [[ -n "$target" ]] || die "$ch has no earlier release to roll back to (history: release.sh history --channel $ch)"
  fi
  log "rollback $ch: release $(jq -r .id <<<"$cur") ($(short "$(jq -r .git_sha <<<"$cur")")) -> release $(jq -r .id <<<"$target") ($(short "$(jq -r .git_sha <<<"$target")"))"
  # The recorded digests must still exist (GHCR retention).
  fv_ghcr_init || die "cannot read $FV_SERVE_REPO"
  for k in $(jq -r '.digests | fromjson | keys[]' <<<"$target"); do
    ref="$(jq -r --arg k "$k" '.digests | fromjson | .[$k]' <<<"$target")"
    fv_ghcr_digest "${ref##*@}" >/dev/null || die "$k: $ref is gone from the registry"
  done
  apply "$ch" "$(jq -c '.digests | fromjson' <<<"$target")" rollback "$(jq -r .git_sha <<<"$target")" "$(jq -r .id <<<"$target")"
}

# ---- views ------------------------------------------------------------------

cmd_list() {
  need_d1
  local heads k tag live mark row
  heads="$(fv_release_heads)"
  [[ "$heads" != '[]' ]] || { echo "no releases recorded yet (release.sh promote <sha> stable)"; return 0; }
  ((VERIFY)) && fv_ghcr_init
  while IFS= read -r row; do
    printf '%-8s release %-4s %s  %-9s %s  by %s%s\n' "$(jq -r .channel <<<"$row")" "$(jq -r .id <<<"$row")" \
      "$(short "$(jq -r .git_sha <<<"$row")")" "$(jq -r .action <<<"$row")" "$(iso "$(jq -r .promoted_at <<<"$row")")" \
      "$(jq -r .promoted_by <<<"$row")" "$(jq -r 'if .notes then "  (" + .notes + ")" else "" end' <<<"$row")"
    for k in $(jq -r '.digests | fromjson | keys[]' <<<"$row"); do
      mark=""
      if ((VERIFY)); then
        tag="$(fv_key_channel_tags "$k" "$(jq -r .channel <<<"$row")" | cut -d' ' -f1)"
        live="$(fv_ghcr_digest "$tag" 2>/dev/null || echo missing)"
        [[ "$live" == "$(jq -r --arg k "$k" '.digests | fromjson | .[$k] | split("@")[1]' <<<"$row")" ]] && mark="  :$tag ok" || mark="  :$tag DIFFERS ($live)"
      fi
      printf '    %-9s %s%s\n' "$k" "$(jq -r --arg k "$k" '.digests | fromjson | .[$k]' <<<"$row")" "$mark"
    done
  done < <(jq -c '.[]' <<<"$heads")
}

cmd_history() {
  need_d1
  local rows
  rows="$(fv_release_rows "$CHANNEL_OPT" "$LIMIT")"
  if ((JSON)); then jq . <<<"$rows"; return; fi
  printf '%-5s %-8s %-9s %-8s %-20s %-6s %s\n' ID CHANNEL ACTION SHA PROMOTED IMAGES BY
  jq -r '.[] | [.id, .channel, .action, .git_sha[0:7], .promoted_at, (.digests | fromjson | length), .promoted_by,
    (if .rolled_back_at then "rolled back" else "" end), (if .source_release then "of #\(.source_release)" else "" end), (.notes // "")] | @tsv' <<<"$rows" \
    | while IFS=$'\t' read -r id ch a s t n by rb src notes; do
        printf '%-5s %-8s %-9s %-8s %-20s %-6s %s %s %s %s\n' "$id" "$ch" "$a" "$s" "$(iso "$t")" "$n" "$by" "$src" "$rb" "$notes"
      done
}

# fetch_runpod: PODS, EPS, TPLS (raw REST v1 lists; only named fields of
# them are ever shown).
fetch_runpod() {
  [[ -n "${RUNPOD_API_KEY:-}" ]] || die "RUNPOD_API_KEY missing"
  PODS="$(rp GET /pods)" || die "Runpod: pods"
  EPS="$(rp GET /endpoints)" || die "Runpod: endpoints"
  TPLS="$(rp GET /templates)" || die "Runpod: templates"
}

# Live fv-serve resources (after fetch_runpod): [{kind, id, name, image,
# created (unix s), status, variant_hint, channel_hint, gpu, dc, dph}]. A
# pod counts when it runs an image of $FV_SERVE_REPO or is named fv-serve-*,
# fv-cluster-* or fv-gw-*; an endpoint when its template's image does or it
# is named fv-*.
live_resources() {
  jq -nc --arg repo "$FV_SERVE_REPO" --argjson pods "$PODS" --argjson eps "$EPS" --argjson tpls "$TPLS" '
    def ts: if . == null then null else (sub("\\.[0-9]+"; "") | sub(" \\+0000 UTC$"; "Z") | sub(" "; "T") | sub("Z?$"; "Z") | fromdateiso8601? // null) end;
    ($tpls | map({key: .id, value: {name, imageName}}) | from_entries) as $t
    | [ ($pods[] | select((.imageName // "" | startswith($repo)) or (.name // "" | test("^fv-(serve|cluster|gw)-")))
         | {kind: (if (.name // "" | test("-gw-|gateway")) then "gateway" else "pod" end), id, name, image: .imageName,
            created: (.createdAt | ts), status: .desiredStatus, variant_hint: (.env.FV_VARIANT // null),
            channel_hint: (.env.FV_RELEASE_CHANNEL // null), gpu: (.machine.gpuDisplayName // .gpu.displayName // null),
            dc: (.machine.dataCenterId // .dataCenterId // null), dph: .costPerHr}),
        ($eps[] | ($t[.templateId] // {}) as $tp
         | select(($tp.imageName // "" | startswith($repo)) or (.name // "" | test("^fv-")))
         | {kind: "endpoint", id, name, image: $tp.imageName, template: .templateId, template_name: $tp.name,
            created: (.createdAt | ts), status: "ENDPOINT", variant_hint: ($tp.name // "" | capture("^fv-serve-(?<v>.+)-(sls|pod)$").v // null),
            channel_hint: null, gpu: ((.gpuTypeIds // []) | join(",")), dc: ((.dataCenterIds // []) | join(",")), dph: null}) ]'
}

# pod_url <pod id> -> its HTTP base (FV_POD_URL_TEMPLATE, `{pod}` replaced).
pod_url() {
  local t="${FV_POD_URL_TEMPLATE:-}"
  [[ -n "$t" ]] || t='https://{pod}-8000.proxy.runpod.net'
  echo "${t//\{pod\}/$1}"
}

cmd_deployed() {
  local live rows heads out r d rel h id
  fetch_runpod
  live="$(live_resources)"
  rows='[]' heads='[]'
  if fv_d1_available && fv_registry_schema 2>/dev/null; then
    rows="$(fv_d1_query "SELECT * FROM deployments WHERE deleted_at IS NULL")" || rows='[]'
    heads="$(fv_release_heads)" || heads='[]'
  else
    log "no D1: showing Runpod only (no release names, no drift)"
  fi
  out='[]'
  while IFS= read -r r; do
    d="$(jq -r '.image // "" | split("@")[1] // ""' <<<"$r")"
    rel="$(jq -nc --arg d "$d" --argjson heads "$heads" --argjson rows "$rows" --argjson r "$r" --arg tc "$TEMPLATE_CHANNEL" '
      ($rows | map(select(.runpod_id == $r.id)) | .[0]) as $row
      | [$heads[] | . as $h | ($h.digests | fromjson) as $m
          | ($m | to_entries | map(select($d != "" and (.value | endswith("@" + $d)))) | .[0].key) as $k
          | {channel: $h.channel, sha: $h.git_sha, key: $k, map: $m}] as $hs
      | ([$hs[] | select(.key) | .key][0] // $row.variant // $r.variant_hint) as $key
      | ($row.channel // $r.channel_hint // (if $r.kind == "endpoint" and ($r.template_name // "" | startswith("fv-serve-")) then $tc else null end)
         // ([$hs[] | select(.key) | .channel][0]) // $tc) as $follows
      | ([$hs[] | select(.channel == $follows)][0]) as $fh
      | {sha: ($row.git_sha // ([$hs[] | select(.key) | .sha][0])), key: $key, channels: [$hs[] | select(.key) | .channel],
         follows: $follows, pool: ($row.pool // null), by: ($row.created_by // null), known: ($row != null),
         drift: (if $fh == null or $key == null or $d == "" then null
                 elif ($fh.map[$key] // "" | endswith("@" + $d)) then false
                 else ($fh.sha[0:7]) end)}')"
    out="$(jq -c --argjson r "$r" --argjson x "$rel" '. + [$r + $x]' <<<"$out")"
  done < <(jq -c '.[]' <<<"$live")
  if ((PROBE)); then
    while IFS= read -r id; do
      h="$(curl -sS --max-time 8 "$(pod_url "$id")/health" 2>/dev/null \
        | jq -c '{running: (.build.git_sha // null), version: (.version // null), state: (.state // .status // null)}' 2>/dev/null || true)"
      [[ -n "$h" ]] || h='{"running":null,"state":"unreachable"}'
      out="$(jq -c --arg id "$id" --argjson h "$h" 'map(if .id == $id then . + {probe: $h} else . end)' <<<"$out")"
    done < <(jq -r '.[] | select(.kind != "endpoint" and .status == "RUNNING") | .id' <<<"$out")
  fi
  if ((JSON)); then jq . <<<"$out"; return; fi
  [[ "$out" != '[]' ]] || { echo "no fv-serve pods or endpoints are live"; return 0; }
  printf '%-8s %-15s %-26s %-8s %-9s %-8s %-5s %-14s %s\n' KIND ID NAME STATUS IMAGE SHA AGE CHANNELS DRIFT
  jq -r '.[] | [.kind, .id, (.name // "-")[0:26], (.status // "-")[0:8], (.key // "?"), ((.sha // "?")[0:7]), (.created // 0),
      ((.channels | join(",")) | if . == "" then "-" else . end),
      (if .drift == null then "?" elif .drift == false then "ok (" + .follows + ")" else "DRIFT: " + .follows + " is " + .drift end),
      (if .known then "-" else "unregistered" end),
      (if .probe then "running=" + ((.probe.running // "?")[0:7]) + " " + (.probe.state // "") else "-" end)] | @tsv' <<<"$out" \
    | while IFS=$'\t' read -r k id n st key s c ch dr unk pr; do
        [[ "$unk" == - ]] && unk="" || unk=" $unk"
        [[ "$pr" == - ]] && pr="" || pr=" $pr"
        printf '%-8s %-15s %-26s %-8s %-9s %-8s %-5s %-14s %s%s%s\n' "$k" "$id" "$n" "$st" "$key" "$s" \
          "$(if [[ "$c" == 0 ]]; then echo "?"; else age "$c"; fi)" "$ch" "$dr" "$unk" "$pr"
      done
}

cmd_reconcile() {
  need_d1
  local live ids rows r id kind cur n_gone=0 n_unknown=0 n_drift=0 heads want v f
  fetch_runpod
  live="$(live_resources)"
  ids="$(jq -nc --argjson p "$PODS" --argjson e "$EPS" '[$p[].id, $e[].id]')"
  rows="$(fv_d1_query "SELECT * FROM deployments WHERE deleted_at IS NULL")"
  # Rows whose resource is gone from the account.
  while IFS= read -r r; do
    n_gone=$((n_gone + 1))
    echo "gone      $(jq -r '"\(.kind) \(.runpod_id) \(.name // "")  (created by \(.created_by))"' <<<"$r")"
    ((DRY)) || fv_deploy_deleted "$(jq -r .kind <<<"$r")" "$(jq -r .runpod_id <<<"$r")" gone
  done < <(jq -c --argjson ids "$ids" '.[] | select(.runpod_id as $i | $ids | index($i) | not)' <<<"$rows")
  # Live resources without a row, and rows whose image changed under them.
  while IFS= read -r r; do
    id="$(jq -r .id <<<"$r")"; kind="$(jq -r .kind <<<"$r")"
    cur="$(jq -c --arg i "$id" 'map(select(.runpod_id == $i)) | .[0] // empty' <<<"$rows")"
    if [[ -z "$cur" ]]; then
      n_unknown=$((n_unknown + 1))
      echo "unknown   $kind $id $(jq -r '"\(.name // "") \(.image // "")"' <<<"$r")"
      if ((ADOPT && ! DRY)); then
        FV_DEPLOYED_BY=reconcile fv_deploy_created "$kind" "$id" name="$(jq -r '.name // ""' <<<"$r")" image="$(jq -r '.image // ""' <<<"$r")" \
          gpu="$(jq -r '.gpu // ""' <<<"$r")" dc="$(jq -r '.dc // ""' <<<"$r")" \
          status="$(jq -r 'if .status == "RUNNING" or .status == "ENDPOINT" then "ready" else "creating" end' <<<"$r")" \
          meta="$(jq -c '{adopted: true} + (if .template then {template: .template} else {} end)' <<<"$r")"
      fi
    elif [[ -n "$(jq -r '.image // empty' <<<"$r")" && "$(jq -r '.image // ""' <<<"$cur")" != "$(jq -r .image <<<"$r")" ]]; then
      n_drift=$((n_drift + 1))
      echo "drifted   $kind $id recorded $(jq -r '.image // "-"' <<<"$cur") live $(jq -r .image <<<"$r")"
      if ((FIX && ! DRY)); then
        fv_deploy_update "$kind" "$id" image="$(jq -r .image <<<"$r")" digest="$(jq -r '.image | split("@")[1] // ""' <<<"$r")"
      fi
    fi
  done < <(jq -c '.[]' <<<"$live")
  # The shared templates vs the channel they follow.
  heads="$(fv_release_heads)"
  want="$(jq -c --arg c "$TEMPLATE_CHANNEL" '[.[] | select(.channel == $c)][0].digests // "{}" | fromjson' <<<"$heads")"
  for v in $FV_VARIANTS; do
    for f in $(fv_variant_flavours "$v"); do
      cur="$(jq -r --arg n "$(fv_template_name "$v" "$f")" '[.[] | select(.name == $n)][0].imageName // empty' <<<"$TPLS")"
      [[ -n "$cur" ]] || { echo "template  $(fv_template_name "$v" "$f") missing"; continue; }
      if [[ "$want" != '{}' && "$(jq -r --arg v "$v" '.[$v] // ""' <<<"$want")" != "$cur" ]]; then
        n_drift=$((n_drift + 1))
        echo "template  $(fv_template_name "$v" "$f") $(dshort "$cur") != $TEMPLATE_CHANNEL $(dshort "$(jq -r --arg v "$v" '.[$v] // "-"' <<<"$want")")"
      fi
    done
  done
  echo "reconcile: $n_gone gone$( ((DRY)) && echo " (not marked: dry run)"), $n_unknown unknown$( ((ADOPT && ! DRY)) && echo " (adopted)"), $n_drift drifted"
}

# ---- redeploy (rolling) -----------------------------------------------------

cmd_redeploy() {
  local sel="${ARGS[0]:?redeploy <pool|all|gateway> [channel|sha]}" target="${ARGS[1]:-stable}" rel digests pools p kind key img old specs=()
  [[ -s "$STATE" ]] || die "no cluster state ($STATE): redeploy rolls the standing cluster of runpod-cluster.sh"
  # A cluster started on per-variant images (`up sha-…` / `up <channel>`)
  # rolls onto the target's variant images, else onto its all-in-one image.
  kind="${FV_CLUSTER_IMAGE_KIND:-$(jq -r 'if (.images // {} | length) > 0 then "variant" else "debug" end' "$STATE")}"
  if [[ "$target" =~ ^[a-z][a-z0-9-]{1,30}$ ]] && fv_d1_available && rel="$(fv_release_current "$target" 2>/dev/null)" && [[ -n "$rel" ]]; then
    digests="$(jq -c '.digests | fromjson' <<<"$rel")"
    log "target: $target = release $(jq -r .id <<<"$rel") ($(short "$(jq -r .git_sha <<<"$rel")"))"
    export FV_RELEASE_CHANNEL="$target"
  else
    rel="$(resolve "$target")"
    digests="$(jq -c .digests <<<"$rel")"
    log "target: build $(jq -r .short <<<"$rel")"
  fi
  if [[ "$sel" == all ]]; then pools="$(jq -r '.workers | keys[]' "$STATE") gateway"; else pools="$sel"; fi
  for p in $pools; do
    if [[ "$p" == gateway ]]; then
      key=debug; [[ "$kind" == variant ]] && key=gateway
      old="$(jq -r '.gateway.image // .images.gateway // .image' "$STATE")"
    else
      jq -e --arg p "$p" '.workers | has($p)' "$STATE" >/dev/null || die "the cluster has no pool $p ($(jq -r '.workers | keys | join(", ")' "$STATE"))"
      key=debug
      if [[ "$kind" == variant ]]; then key="$p"; [[ "$p" == wan ]] && key=wan5b; fi
      old="$(jq -r --arg p "$p" '.workers[$p].image // .images[$p] // .image' "$STATE")"
    fi
    img="$(jq -r --arg k "$key" '.[$k] // empty' <<<"$digests")"
    [[ -n "$img" ]] || die "the target has no $key image"
    if [[ "$old" == "$img" ]] && ((! FORCE)); then log "$p: already $(dshort "$img")"; continue; fi
    echo "  $p: $(dshort "$old") -> $(dshort "$img")"
    specs+=("$p=$img")
  done
  ((${#specs[@]})) || { echo "nothing to redeploy"; return 0; }
  if ((DRY)); then echo "(dry run: nothing changed)"; return 0; fi
  bash "$HERE/runpod-cluster.sh" roll "${specs[@]}"
}

# ---- CI ---------------------------------------------------------------------

cmd_record_build() {
  local ch="${ARGS[0]:?record-build <channel> <sha> <variants.tsv> [debug ref]}" sha="${ARGS[1]:?sha}" tsv="${ARGS[2]:?variants.tsv}" dbg="${ARGS[3]:-}" digests id
  check_channel "$ch"
  need_d1
  digests="$(awk -F'\t' 'NF >= 2 && $2 ~ /@sha256:/ {print $1 "\t" $2}' "$tsv" | jq -R -s -c 'split("\n") | map(select(length > 0) | split("\t") | {(.[0]): .[1]}) | add // {}')"
  [[ -z "$dbg" ]] || digests="$(jq -c --arg d "$dbg" '. + {debug: $d}' <<<"$digests")"
  [[ "$digests" != '{}' ]] || die "no digests in $tsv"
  id="$(fv_release_insert "$(jq -nc --arg c "$ch" --arg s "$sha" --argjson d "$digests" --arg n "$NOTES" --argjson t "$TPL_UPDATED" \
    --arg run "${GITHUB_SERVER_URL:+$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/${GITHUB_RUN_ID:-}}" \
    '{channel: $c, git_sha: $s, digests: $d, action: "build", notes: (if $n == "" then null else $n end),
      run_url: (if $run == "" then null else $run end), templates_updated: ($t == 1)}')")"
  echo "release $id: $ch = $(short "$sha") (build, $(jq 'length' <<<"$digests") images)"
}

main() {
  local cmd="${1:-}"
  [[ -n "$cmd" ]] || { sed -n '2,48p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
  shift
  parse "$@"
  case "$cmd" in
    list) cmd_list ;;
    history) cmd_history ;;
    deployed) cmd_deployed ;;
    promote) cmd_promote ;;
    rollback) cmd_rollback ;;
    redeploy) cmd_redeploy ;;
    reconcile) cmd_reconcile ;;
    resolve) resolve "${ARGS[0]:?resolve <sha|digest|tag>}" | jq . ;;
    record-build) cmd_record_build ;;
    -h | --help | help) sed -n '2,48p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) die "unknown command $cmd (release.sh help)" ;;
  esac
}
main "$@"
