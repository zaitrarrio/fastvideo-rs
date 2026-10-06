#!/usr/bin/env bash
# Shared CPU build pods (docs/dev/build-pod.md), driven over the pod's HTTPS
# proxy (no SSH). Each agent gets its own worktree snapshot and
# CARGO_TARGET_DIR on the pod. fv-control creates, starts, stops and deletes
# the pods (docs/dev/build-pods-fv-control.md); this script never does: its
# lifecycle commands ask fv-control (scripts/serve/fv-control.sh build-pod).
#
#   build-pod.sh up                  ask fv-control for a pod (it reuses a running one,
#                                    starts a stopped one or creates one, in any region;
#                                    FV_BUILD_REGION=eu|us|<DC> to pick), wait until it
#                                    is ready; its id and token land in $FV_BUILD_STATE
#   build-pod.sh status              pod, $/hr, setup state, jobs, disk, the
#                                    self-stop timers (idle, time to idle / cap stop),
#                                    and per agent: idle hours, target/snapshot sizes,
#                                    when eviction takes them; recent evictions
#   build-pod.sh agents [--sizes]    agent dirs on the pod
#   build-pod.sh sync <agent>        mirror this worktree (tracked + untracked,
#                                    not ignored) to worktrees/<agent>/
#   build-pod.sh run <agent> [--no-sync] -- [K=V ...] <cmd...>
#                                    sync, then run an allowlisted command
#                                    (cargo check|build|test|clippy|fmt|doc|tree|
#                                    metadata, bash scripts/serve/check.sh,
#                                    bash scripts/gpu/lint.sh); streams the log
#                                    and exits with the command's status
#   build-pod.sh log <job>           re-attach to a job's log
#   build-pod.sh cancel <job>        cancel a job (e.g. after the client died)
#   build-pod.sh fetch <agent> <path> [dest]
#                                    copy target/<agent>/<path> back (gzip in
#                                    transit), e.g. release/fv-serve
#   build-pod.sh clean <agent> [target|worktree|all]   (default all)
#   build-pod.sh seed <agent> [--force]
#                                    sync, then build the deps seed of that worktree's
#                                    Cargo.lock now (jobs also start one after their
#                                    first success on a Cargo.lock without a seed)
#   build-pod.sh evict               run the pod's eviction pass now (it also runs
#                                    every minute: dirs unused > FV_BUILD_EVICT_HOURS,
#                                    default 6, then LRU target dirs while under
#                                    FV_BUILD_EVICT_FREE_GB, default 40, free)
#   build-pod.sh stop [--force]      ask fv-control to stop the pod (refused while jobs
#                                    or its GitHub runner are busy, unless --force)
#   build-pod.sh down [--force]      ask fv-control to delete the pod (same guard)
#   build-pod.sh release-artifacts <sha|ref> [--sets "a b"] [--force] [--no-upload] [--keep]
#                                    build that commit's release binaries on the
#                                    pod (scripts/dev/release-artifacts-pod.sh),
#                                    verify and upload them to R2 artifacts/<sha>/
#                                    for the image workflows (wakes the pod)
#   build-pod.sh plan [region]       fv-control's placement: candidates by stock and price
#
# <agent> is any [A-Za-z0-9._-] name; "." means this worktree's directory name.
#
# Money guards live in fv-control (its build_pods policy: balance floor + margin,
# $/hr cap, daily budget, max pods); each pod stops itself after its idle time
# (default 20 min without jobs) and its cap (default 8 h + 30 min grace), a curl
# watchdog in its start command repeats the cap, and fv-control's cron backstops
# both and removes the pod's GitHub runner.
#
# Env: FV_CONTROL_URL / FV_CONTROL_TOKEN_FILE (an admin token: up gets the pod
# token) for fv-control; FV_BUILD_STATE (default ~/.config/fv-build: pod id and
# token, mode 600, never printed); FV_BUILD_REGION for up; FV_BUILD_URL points
# the pod calls at a local test server. BASE_IMAGE_TAG below is the image pin
# fv-control reads from main for new pods.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/lib.sh
source "$HERE/../gpu/lib.sh"

STATE="${FV_BUILD_STATE:-${XDG_CONFIG_HOME:-$HOME/.config}/fv-build}"
# The base image (docker/build-base.Dockerfile): toolchains, CUDA, sccache, mold,
# ffmpeg, Node/Playwright/Chromium. The tag is a content hash of its inputs;
# `bash scripts/dev/build-base-tag.sh --pin` updates it, and the
# build-base-image workflow pushes the image and checks this pin. fv-control
# creates new pods with the pin on main (docs/dev/build-pods-fv-control.md).
BASE_IMAGE_TAG="bb-d872f7724765429b"
# shellcheck disable=SC2034  # fv-control reads the pin (control/src/buildpods.ts imagePin)
IMAGE="ghcr.io/zaitrarrio/fastvideo-rs-build-base:$BASE_IMAGE_TAG"
AUTH_FILE="$STATE/auth-header"
FVC="$HERE/../serve/fv-control.sh"
export FV_BUILD_STATE="$STATE"

umask 077
mkdir -p "$STATE"
chmod 700 "$STATE"

# FV_BUILD_URL points the service calls elsewhere (a local test server).
base_url() { echo "${FV_BUILD_URL:-https://$1-8000.proxy.runpod.net}"; }
pod_id() {
  local id
  [[ -n "${FV_BUILD_URL:-}" ]] && { echo local; return; }
  id="$(cat "$STATE/pod" 2>/dev/null || true)"
  [[ -n "$id" ]] || die "no build pod recorded in $STATE/pod; run: build-pod.sh up"
  echo "$id"
}
# Pod service call: $1 method, $2 path, rest = extra curl args. Token via header file.
svc() {
  local method="$1" path="$2"; shift 2
  [[ -s "$AUTH_FILE" ]] || die "no token in $STATE; run: build-pod.sh up (or fv-control.sh build-pod token <id>)"
  curl -sS --fail-with-body --max-time "${FV_BUILD_HTTP_TIMEOUT:-90}" -X "$method" -H @"$AUTH_FILE" \
    "$@" "$(base_url "$(pod_id)")$path"
}

# svc, dying with the pod's error (e.g. 507 "build pod disk full: ...") on failure.
svc_or_die() {
  local what="$1" out rc=0
  shift
  out="$(svc "$@" 2>&1)" || rc=$?
  (( rc == 0 )) && return 0
  # The pod's JSON error body and curl's own message, in either order.
  die "$what failed: $(grep -m1 '^{' <<<"$out" | jq -r '.error // empty' 2>/dev/null || true) [$(grep -v '^{' <<<"$out" | head -c 200)]"
}

agent_name() {
  local a="$1"
  [[ "$a" == . ]] && a="$(basename "$FV_ROOT")"
  [[ "$a" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || die "bad agent name '$a'"
  echo "$a"
}

server_sha() { sha256sum "$HERE/build-pod-server.py" | cut -c1-12; }

# fv-control hands out a ready pod (reuse / start / create) and its token; only
# it creates, replaces or deletes pods.
cmd_up() {
  [[ -n "${FV_BUILD_URL:-}" ]] && { log "FV_BUILD_URL is set: no pod to bring up"; return 0; }
  require_tools curl jq
  bash "$FVC" build-pod up ${FV_BUILD_REGION:+--region "$FV_BUILD_REGION"}
  local srv_pod
  srv_pod="$(svc GET /v1/status | jq -r '.server_sha // ""')"
  [[ "$srv_pod" == "$(server_sha)" ]] || log "note: the pod runs server $srv_pod (main's, from fv-control); this checkout has $(server_sha)"
}

cmd_status() {
  [[ -z "${FV_BUILD_URL:-}" ]] && { bash "$FVC" build-pod status || true; }
  local id st
  id="$(cat "$STATE/pod" 2>/dev/null || true)"
  [[ -n "${FV_BUILD_URL:-}" || -n "$id" ]] || { echo "no pod recorded in $STATE (build-pod.sh up)"; return 0; }
  echo "this worktree uses pod ${id:-local}"
  st="$(svc GET /v1/status 2>/dev/null)" || { echo "pod ${id:-local}: no answer (stopped?)"; return 0; }
  jq 'del(.agents)' <<<"$st"
  stop_timers <<<"$st"
  cache_summary <<<"$st"
  # A server older than eviction has no agent rows.
  if jq -e '.eviction' <<<"$st" >/dev/null; then
    jq -r '"agents on the container disk (\(.local_disk.free_gb // "?") GB free; evict after \(.eviction.idle_hours) h unused, LRU targets below \(.eviction.free_gb_floor) GB free; \(.eviction.protect // [] | join(" ")) held \(.eviction.hold_min // 0) min after use):",
      (["agent", "idle_h", "busy", "held", "target_gb", "worktree_gb", "evict_in_h"] | @tsv),
      (.agents // [] | .[] | [.agent, .idle_h, .busy, (.held // false),
         (if .target then (.target_gb // "?") else "-" end),
         (if .worktree then (.worktree_gb // "?") else "-" end), .evict_in_h] | @tsv)' <<<"$st" | tsv_table
  fi
}

# One line from /v1/status: idle time and when the pod stops itself. Servers
# before 2026-10-02 lack the *_in_s fields; derive them there.
# Aligned columns from TSV on stdin (`column` is not in every container).
tsv_table() {
  if command -v column >/dev/null; then column -t -s $'\t'; else tr '\t' ' '; fi
}

# Image, sccache hit rate, cache sizes and deps seeds, one line each.
cache_summary() {
  jq -r 'def gb: if . == null then "?" else "\(.) GB" end;
    "image: \(.image // "?")  (\(.rustc // "rustc ?"))",
    (if (.sccache | type) == "object" then
       "sccache: \(.sccache.hits) hits / \(.sccache.hits + .sccache.misses) cacheable compiles"
       + " (hit rate \(if .sccache.hit_rate == null then "-" else "\(.sccache.hit_rate * 100 | round) %" end)),"
       + " \(.sccache.not_cacheable // 0) not cacheable, \(.sccache.errors // 0) errors since boot;"
       + " cache \(.sccache.cache_size_gb | gb) of \(.sccache.max_cache_size_gb | gb)"
     else "sccache: NOT RUNNING (\(.sccache))" end),
    "caches: " + ([(.caches_gb // {}) | to_entries[] | select(.key != "t") | "\(.key) \(.value | gb)"] | join(", ")),
    "deps seeds (\(if .deps_seeds.enabled then "on" else "off" end), keep \(.deps_seeds.keep)): "
      + ([.deps_seeds.seeds[]? | "\(.key) \(.gb) GB (\(.unpacked_gb) GB unpacked, used \(.last_used_h_ago) h ago)"] | join("; "))
      + (if (.deps_seeds.building // []) | length > 0 then "; building: \(.deps_seeds.building | map(.argv[1]) | join(" "))" else "" end)' 2>/dev/null || true
}

stop_timers() {
  jq -r 'def mins: if . == null then "-" else "\((. / 60) | floor) min" end;
    (.jobs_active | length) as $n
    | (.idle_stop_in_s // (if $n > 0 then null else ([.idle_stop_s - .idle_s, 0] | max) end)) as $idle_in
    | (.max_stop_in_s // ([.max_s - .uptime_s, 0] | max)) as $max_in
    | "self-stop: up \(.uptime_s | mins), idle \(.idle_s | mins), \($n) job(s) active;"
      + " idle stop in \(if $idle_in == null then "- (jobs active)" else ($idle_in | mins) end),"
      + " cap stop in \($max_in | mins)"
      + (if (.self_stop.error // null) != null then "; LAST STOP ATTEMPT FAILED: \(.self_stop.error)"
         elif (.self_stop.ok // null) != null then "; stop accepted (\(.self_stop.ok))" else "" end)'
}

# Mirror tracked + untracked (not ignored) files of this worktree to the pod.
cmd_sync() {
  local agent tmp n_changed n_deleted t0
  agent="$(agent_name "${1:?agent}")"
  t0=$(date +%s)
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/fv-build-sync.XXXXXX")"
  # No RETURN trap: it is global, not function-local, and depending on the
  # bash version it fired again after this function (in cmd_run, or on the
  # script's last return) with the local $tmp gone, so set -u reported
  # "tmp: unbound variable" after successful runs. A global path plus an EXIT
  # trap covers die(); the normal path removes the directory itself.
  FV_SYNC_TMP="$tmp"
  trap 'rm -rf "${FV_SYNC_TMP:-}"; fv_rel_cleanup' EXIT
  svc GET "/v1/agents/$agent/manifest" >"$tmp/remote"
  {
    git -C "$FV_ROOT" ls-files -z --recurse-submodules
    git -C "$FV_ROOT" ls-files -z -o --exclude-standard
  } >"$tmp/files"
  python3 - "$FV_ROOT" "$tmp/files" "$tmp/remote" "$tmp/changed" "$tmp/deleted" <<'PY'
import os, sys
root, files, remote, changed, deleted = sys.argv[1:]
rem = {}
for line in open(remote, encoding="utf-8", errors="surrogateescape"):
    p, s, m = line.rstrip("\n").rsplit("\t", 2)
    rem[p] = (int(s), int(m))
local = {}
for p in open(files, "rb").read().decode("utf-8", "surrogateescape").split("\0"):
    if not p or p in local:
        continue
    try:
        st = os.lstat(os.path.join(root, p))
    except FileNotFoundError:
        continue
    if os.path.isdir(os.path.join(root, p)) and not os.path.islink(os.path.join(root, p)):
        continue
    local[p] = (st.st_size, int(st.st_mtime))
with open(changed, "wb") as f:
    f.write(b"".join(p.encode("utf-8", "surrogateescape") + b"\0" for p, v in sorted(local.items()) if rem.get(p) != v))
with open(deleted, "wb") as f:
    f.write(b"".join(p.encode("utf-8", "surrogateescape") + b"\0" for p in sorted(rem) if p not in local))
PY
  n_changed="$(tr -cd '\0' <"$tmp/changed" | wc -c)"
  n_deleted="$(tr -cd '\0' <"$tmp/deleted" | wc -c)"
  if (( n_changed > 0 )); then
    tar -C "$FV_ROOT" --null -T "$tmp/changed" --format=gnu -cf - | gzip -1 >"$tmp/upload.tgz"
    # A first full snapshot (~1400 files, 29 MB) takes about a minute to
    # extract onto the network volume; allow far more than the default 90 s.
    FV_BUILD_HTTP_TIMEOUT="${FV_BUILD_SYNC_TIMEOUT:-900}" svc_or_die "sync $agent" PUT "/v1/agents/$agent/files" \
      --data-binary @"$tmp/upload.tgz" -H 'content-type: application/gzip'
  fi
  if (( n_deleted > 0 )); then
    svc_or_die "sync $agent (deletions)" POST "/v1/agents/$agent/delete" --data-binary @"$tmp/deleted"
  fi
  rm -rf "$tmp"
  FV_SYNC_TMP=""
  log "sync $agent: $n_changed changed, $n_deleted deleted ($(( $(date +%s) - t0 ))s)"
}

follow() {
  local job="$1" off=0 resp fails=0 fin rc
  while :; do
    if ! resp="$(svc GET "/v1/jobs/$job?offset=$off&wait=20" 2>/dev/null)"; then
      fails=$((fails + 1))
      (( fails < 20 )) || die "lost the pod while following job $job (re-attach: build-pod.sh log $job)"
      sleep 5
      continue
    fi
    fails=0
    jq -j '.data' <<<"$resp"
    off="$(jq -r '.next' <<<"$resp")"
    fin="$(jq -r '.finished' <<<"$resp")"
    if [[ "$fin" == true ]]; then
      rc="$(jq -r '.exit // 1' <<<"$resp")"
      return "$rc"
    fi
  done
}

cmd_run() {
  local agent sync=1 env_json='{}' argv_json resp job
  agent="$(agent_name "${1:?agent}")"; shift
  [[ "${1:-}" == --no-sync ]] && { sync=0; shift; }
  [[ "${1:-}" == -- ]] && shift
  (( $# > 0 )) || die "usage: build-pod.sh run <agent> [--no-sync] -- [K=V ...] <cmd...>"
  while [[ "${1:-}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; do
    env_json="$(jq -c --arg k "${1%%=*}" --arg v "${1#*=}" '. + {($k): $v}' <<<"$env_json")"
    shift
  done
  argv_json="$(printf '%s\n' "$@" | jq -R . | jq -sc .)"
  (( sync )) && cmd_sync "$agent"
  resp="$(svc POST "/v1/agents/$agent/jobs" -H 'content-type: application/json' \
    -d "$(jq -nc --argjson a "$argv_json" --argjson e "$env_json" '{argv: $a, env: $e}')" 2>/dev/null)" \
    || die "rejected: $(jq -r '.error // .' <<<"$resp" 2>/dev/null || echo "$resp")"
  job="$(jq -r .id <<<"$resp")"
  log "job $job ($agent): $*"
  trap 'svc POST "/v1/jobs/'"$job"'/cancel" >/dev/null 2>&1; log "cancelled job '"$job"'"; exit 130' INT TERM
  local rc=0
  follow "$job" || rc=$?
  trap - INT TERM
  return "$rc"
}

cmd_fetch() {
  local agent path dest t0
  agent="$(agent_name "${1:?agent}")"; path="${2:?path under target/<agent>, e.g. release/fv-serve}"
  dest="${3:-$FV_ROOT/artifacts/build-pod/$agent/$path}"
  mkdir -p "$(dirname "$dest")"
  t0=$(date +%s)
  # Download first, so a 404 shows the pod's error instead of a gunzip one.
  if ! FV_BUILD_HTTP_TIMEOUT=900 svc GET "/v1/agents/$agent/artifact?path=$(jq -rn --arg p "$path" '$p|@uri')&gz=1" \
    -o "$dest.gz.part"; then
    local err
    err="$(jq -r '.error // empty' "$dest.gz.part" 2>/dev/null || true)"
    rm -f "$dest.gz.part"
    die "fetch $path failed${err:+: $err}"
  fi
  gunzip <"$dest.gz.part" >"$dest.part"
  rm -f "$dest.gz.part"
  mv -f "$dest.part" "$dest"
  chmod +x "$dest" 2>/dev/null || true
  log "fetched $path → $dest ($(du -h "$dest" | cut -f1), $(( $(date +%s) - t0 ))s)"
}

# Stop / delete go through fv-control (it removes the GitHub runner first and
# refuses while jobs or the runner are busy, unless --force).
cmd_stop() { bash "$FVC" build-pod stop "$@"; }
cmd_down() { bash "$FVC" build-pod delete "$@"; }

# ---- release artifacts (docs/dev/build-pod.md "Release artifacts") ----------
# Build one commit's release binaries on the pod (scripts/dev/release-artifacts-pod.sh),
# fetch the tarballs + manifest.json, check their sha256s and upload them to
# R2 under artifacts/<sha>/ (manifest.json last: its presence means complete).
REL_AGENT="${FV_RELEASE_AGENT:-fv-release}"
R2_ENV_FILE="${FV_R2_ARTIFACTS_ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/fv/r2-build-artifacts-rw.env}"
# The temporary worktree of a release build; also called from cmd_sync's EXIT trap.
FV_REL_WT=""
fv_rel_cleanup() {
  [[ -n "$FV_REL_WT" ]] || return 0
  git -C "$FV_ROOT" worktree remove --force "$FV_REL_WT" >/dev/null 2>&1 || rm -rf "$FV_REL_WT"
}
r2() { FV_R2_ARTIFACTS_ENV_FILE="$R2_ENV_FILE" python3 "$HERE/r2.py" "$@"; }

cmd_release_artifacts() {
  local rev="" sets="" force=0 upload=1 keep=0
  while (( $# )); do
    case "$1" in
      --sets) sets="${2:?--sets needs a list}"; shift 2 ;;
      --force) force=1; shift ;;
      --no-upload) upload=0; shift ;;
      --keep) keep=1; shift ;;
      -*) die "unknown flag $1" ;;
      *) [[ -z "$rev" ]] || die "one revision only"; rev="$1"; shift ;;
    esac
  done
  [[ -n "$rev" ]] || die "usage: build-pod.sh release-artifacts <sha|ref> [--sets \"a b\"] [--force] [--no-upload] [--keep]"
  require_tools git jq python3 sha256sum
  local sha
  if ! sha="$(git -C "$FV_ROOT" rev-parse -q --verify "$rev^{commit}")"; then
    git -C "$FV_ROOT" fetch -q origin || true
    sha="$(git -C "$FV_ROOT" rev-parse -q --verify "$rev^{commit}" || git -C "$FV_ROOT" rev-parse -q --verify "origin/$rev^{commit}")" \
      || die "unknown revision $rev"
  fi
  if (( upload )); then
    [[ -s "$R2_ENV_FILE" ]] || die "no R2 credentials in $R2_ENV_FILE (FV_R2_ARTIFACTS_*; docs/dev/build-pod.md \"Release artifacts\"); --no-upload builds without uploading"
    local rc=0
    r2 head "artifacts/$sha/manifest.json" || rc=$?
    case "$rc" in
      0) if (( !force )); then log "artifacts/$sha already in R2 (--force rebuilds)"; return 0; fi ;;
      1) ;;
      *) die "cannot read the R2 bucket with $R2_ENV_FILE (r2.py exit $rc)" ;;
    esac
  fi
  # One release build per container at a time: they share the pod's
  # $REL_AGENT snapshot and target dir.
  exec 9>"$STATE/release.lock"
  flock -w "${FV_RELEASE_LOCK_WAIT_S:-3600}" 9 || die "another release-artifacts run holds $STATE/release.lock"

  local wt="${TMPDIR:-/tmp}/fv-release-${sha:0:12}" out="${FV_RELEASE_OUT:-$FV_ROOT/artifacts/release/$sha}"
  local build_id build_time run_id t0=$SECONDS
  git -C "$FV_ROOT" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"
  git -C "$FV_ROOT" worktree add -q --detach "$wt" "$sha"
  FV_REL_WT="$wt"
  trap fv_rel_cleanup EXIT
  # The image workflows check out submodules too; only cutile-rs is built.
  git -C "$wt" submodule update -q --init third_party/cutile-rs
  build_id="$(bash "$wt/scripts/gpu/docker.sh" build-id)"
  build_time="$(cd "$wt" && TZ=UTC git log -1 --format=%cd --date=format-local:%Y-%m-%dT%H:%M:%SZ)"
  run_id="$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 3)"
  # The recipe comes from this checkout (it may postdate <sha>); its hash is
  # in the manifest (builder.recipe_sha256).
  install -m 755 "$HERE/release-artifacts-pod.sh" "$wt/scripts/dev/release-artifacts-pod.sh"
  log "release artifacts for $sha (build id $build_id, run $run_id)"

  cmd_up
  FV_ROOT="$wt" cmd_sync "$REL_AGENT"
  local env=(FV_REL_SHA="$sha" FV_GIT_SHA="$sha" FV_BUILD_TIME="$build_time" FV_BUILD_ID="$build_id" FV_REL_RUN_ID="$run_id")
  [[ -n "$sets" ]] && env+=(FV_REL_SETS="$sets")
  cmd_run "$REL_AGENT" --no-sync -- "${env[@]}" bash scripts/dev/release-artifacts-pod.sh \
    || die "pod build failed (log above; re-attach with build-pod.sh log <job>)"

  rm -rf "$out" && mkdir -p "$out"
  cmd_fetch "$REL_AGENT" "release-artifacts/$sha/manifest.json" "$out/manifest.json"
  chmod -x "$out/manifest.json"
  [[ "$(jq -r .sha "$out/manifest.json")" == "$sha" ]] || die "manifest is for another sha"
  local tb want
  while IFS=$'\t' read -r _ tb want; do
    cmd_fetch "$REL_AGENT" "release-artifacts/$sha/$tb" "$out/$tb"
    chmod -x "$out/$tb"
    [[ "$(sha256sum "$out/$tb" | cut -d' ' -f1)" == "$want" ]] || die "$tb: sha256 differs from the manifest"
  done < <(jq -r '.sets | to_entries[] | [.key, .value.tarball, .value.sha256] | @tsv' "$out/manifest.json")
  log "fetched and verified $(jq '.sets | length' "$out/manifest.json") sets ($(du -sh "$out" | cut -f1)) into $out"

  if (( upload )); then
    while IFS=$'\t' read -r _ tb; do
      r2 put "artifacts/$sha/$tb" "$out/$tb"
    done < <(jq -r '.sets | to_entries[] | [.key, .value.tarball] | @tsv' "$out/manifest.json")
    r2 put "artifacts/$sha/manifest.json" "$out/manifest.json"
    log "uploaded to R2: artifacts/$sha/ ($(( SECONDS - t0 ))s in all)"
    (( keep )) || rm -rf "$out"
  fi
}

case "${1:-}" in
  up) cmd_up ;;
  status) cmd_status ;;
  agents) shift; svc GET "/v1/agents$([[ "${1:-}" == --sizes ]] && echo '?sizes=1')" | jq . ;;
  sync) shift; cmd_sync "$@" ;;
  run) shift; cmd_run "$@" ;;
  log) follow "${2:?job id}" ;;
  cancel) svc POST "/v1/jobs/${2:?job id}/cancel" | jq -c '{id, agent, state}' ;;
  fetch) shift; cmd_fetch "$@" ;;
  clean) svc POST "/v1/agents/$(agent_name "${2:?agent}")/clean" -d "{\"what\":\"${3:-all}\"}" | jq -c . ;;
  seed) a="$(agent_name "${2:?agent}")"; cmd_sync "$a"
    svc POST "/v1/agents/$a/seed" -d "{\"force\":$([[ "${3:-}" == --force ]] && echo true || echo false)}" | jq -c . ;;
  evict) FV_BUILD_HTTP_TIMEOUT=900 svc POST /v1/evict | jq -c '.evicted[]' ;;
  stop) shift; cmd_stop "$@" ;;
  down) shift; cmd_down "$@" ;;
  release-artifacts) shift; cmd_release_artifacts "$@" ;;
  plan) bash "$FVC" build-pod plan "${2:-}" ;;
  volume-create) die "volumes are fv-control's (build_pods.volumes: datacenter -> volume id); create one in the Runpod console" ;;
  # The header comment (line 2 up to `set -euo pipefail`) is the usage.
  *) sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
