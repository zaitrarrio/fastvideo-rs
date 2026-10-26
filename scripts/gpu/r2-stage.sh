#!/usr/bin/env bash
# r2-stage.sh: the weights of a script-runner pod with R2 trees
# (runpod-http.sh run with FV_R2_TREES, docs/adr/0019-fv-lab.md §21.8): fv-lab's
# merged weight root of a `weights = "volume+r2"` tier (§16.3), made by the
# repository's own tools as fv-lab's runner does (§16.4), so runpod-matrix.sh,
# run with FV_WORK=<the root's parent>, reads every tree at the path fv-lab's
# arms do and its command lines equal theirs word for word.
#
#   r2-stage.sh <root> <state dir> <status dir> <cap s> <volume weights> "<link roots>" <dest@rev>...
#
#   <root>            the merged root on the container disk (/fvscratch/weights)
#   <state dir>       private (mode 700) and outside the served run directory:
#                     the plan of presigned URLs (bearer tokens) lives there
#                     while the fetch runs, and is removed whatever happened
#   <status dir>      where r2-fetch.py's status (r2-fetch.json) goes
#   <cap s>           the fetch's cap (the cell cap, as fv-lab's fetch_cap_s)
#   <volume weights>  the volume's weights directory (/workspace/weights): only read
#   <link roots>      the volume trees no other volume tree holds
#                     (fv-lab weights::TreeSources::link_roots), each linked at
#                     <root>/<tree>
#   <dest@rev>        the trees to fetch (weights-r2.tsv rows, checked by
#                     runpod-http.sh before the pod was made)
#
# The R2 key comes in FV_WEIGHTS_R2_ACCESS_KEY_ID / FV_WEIGHTS_R2_SECRET_ACCESS_KEY
# (the pod's Runpod secret references, filled by Runpod; runpod-http.sh's start
# command removed them from its own environment at once and hands them to this
# script alone). They are read into this shell's variables and unset at once,
# and only `r2-weights.py plan` gets them, in its environment (never an argv);
# r2-fetch.py reads the plan's presigned URLs. Nothing of the key is printed:
# the log says `r2_key=env|none|...` only. Never on the volume: the root, the
# state directory and every fetched tree's parent must resolve off
# /workspace and off <volume weights>; a link is never written through.
# python3: the runtime image has none (its CI job keeps it Python-free), so it
# is installed with apt under a 600 s cap, as fv-lab's runner does (§16.4).
#
# Exit 0: every tree is there and every link made; non-zero otherwise (the
# matrix's arms then record `weights incomplete`). scripts/gpu/tests/runpod-r2.test.sh
# runs it with a fake python.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The key first, before anything else runs.
# (A blank value is no key: the start command passes both names, empty when
# the pod had none.)
_id="${FV_WEIGHTS_R2_ACCESS_KEY_ID:-}" _sec="${FV_WEIGHTS_R2_SECRET_ACCESS_KEY:-}"
unset FV_WEIGHTS_R2_ACCESS_KEY_ID FV_WEIGHTS_R2_SECRET_ACCESS_KEY
_id="${_id//[[:space:]]/}" _sec="${_sec//[[:space:]]/}" _had=""
[[ -n "$_id$_sec" ]] && _had=1
if [[ -n "$_id" && -n "$_sec" && "$_id$_sec" != *'{{'* ]]; then
  key_src=env
elif [[ -n "$_had" && "$_id$_sec" == *'{{'* ]]; then
  key_src="none (env: unfilled secret reference)"; _id="" _sec=""
elif [[ -n "$_had" ]]; then
  key_src="none (env: incomplete)"; _id="" _sec=""
else
  key_src=none
fi

log() { printf '[%s] r2-stage: %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "FAILED: $*"; exit 1; }

(( $# >= 7 )) || die "usage: r2-stage.sh <root> <state dir> <status dir> <cap s> <volume weights> \"<link roots>\" <dest@rev>..."
ROOT="$1" STATE="$2" STATUS="$3" CAP="$4" VOL="$5" LINKS="$6"
shift 6
log "r2_key=$key_src"
[[ "$CAP" =~ ^[1-9][0-9]*$ ]] || die "cap '$CAP' is not a positive number of seconds"
NAME='[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*'
for t in $LINKS; do
  [[ "$t" =~ ^$NAME$ && "/$t/" != */./* && "/$t/" != */../* ]] || die "link root '$t' is not a tree name"
done
for spec in "$@"; do
  [[ "$spec" =~ ^($NAME)@[A-Za-z0-9._-]+$ && "/${spec%@*}/" != */./* && "/${spec%@*}/" != */../* ]] \
    || die "'$spec' is not <tree>@<revision>"
done

# Never on the volume: the root and the state directory resolve off /workspace
# and off the volume's weights directory, and neither is inside the other.
off_volume() { # <path>
  local r v
  r="$(readlink -m "$1")/" v="$(readlink -m "$VOL")/"
  [[ "$r" != /workspace/* && "$r" != "$v"* && "$v" != "$r"* ]]
}
off_volume "$ROOT" || die "the root $ROOT is on the volume ($VOL, /workspace)"
off_volume "$STATE" || die "the state directory $STATE is on the volume"
mkdir -p "$ROOT" "$STATE" && chmod 700 "$STATE" || die "cannot make $ROOT / $STATE"
PLAN="$STATE/plan.json"
rm -f "$PLAN"   # an interrupted run's
trap 'rm -f "$PLAN"' EXIT

# The volume's trees, linked as fv-lab links them (once; a path already there is left alone).
n=0
for t in $LINKS; do
  if [[ ! -e "$ROOT/$t" && ! -L "$ROOT/$t" ]]; then
    mkdir -p "$(dirname "$ROOT/$t")" && ln -s "$VOL/$t" "$ROOT/$t" || die "cannot link $ROOT/$t"
  fi
  n=$((n + 1))
done

# The trees to fetch: those not already complete at their revision.
todo=()
for spec in "$@"; do
  dest="${spec%@*}" rev="${spec##*@}"
  # Its parent must be the root's own directory, never a link into the volume.
  parent="$(readlink -m "$ROOT/$dest/..")"
  [[ "$parent/" == "$(readlink -m "$ROOT")/"* ]] && off_volume "$parent" \
    || die "$dest would land at $parent, outside $ROOT"
  if [[ -f "$ROOT/$dest/.complete" && "$(cat "$ROOT/$dest/.fv-revision" 2>/dev/null)" == "$rev" ]]; then
    log "$spec already in $ROOT"
  else
    todo+=("$spec")
  fi
done
if (( ${#todo[@]} )); then
  [[ "$key_src" == env ]] || die "no R2 key (r2_key=$key_src; the Runpod secrets fv_r2_access_key_id / fv_r2_secret_access_key)"
  PY="${FV_R2_PYTHON:-python3}"
  check='import concurrent.futures, hashlib, hmac, http.client, json, ssl, urllib.request, xml.etree.ElementTree'
  if ! "$PY" -c "$check" >/dev/null 2>&1; then
    log "no python3 in the image: apt-get install python3"
    OPTS="-o Acquire::http::Timeout=15 -o Acquire::Retries=3 -o DPkg::Lock::Timeout=120"
    DEBIAN_FRONTEND=noninteractive timeout --signal=TERM --kill-after=15 600 sh -c \
      "\"\$0\" $OPTS update -qq && \"\$0\" $OPTS install -y -qq --no-install-recommends python3" "${FV_R2_APT:-apt-get}" \
      || die "apt-get install python3 failed"
    "$PY" -c "$check" >/dev/null 2>&1 || die "python3 unavailable after apt-get"
  fi
  log "fetch ${todo[*]} into $ROOT (cap ${CAP}s)"
  FV_WEIGHTS_R2_ACCESS_KEY_ID="$_id" FV_WEIGHTS_R2_SECRET_ACCESS_KEY="$_sec" \
    "$PY" "$HERE/r2-weights.py" plan "${todo[@]}" --pin "$HERE/weights-r2.tsv" --ttl "$((CAP + 600))" --out "$PLAN" \
    || die "r2-weights.py plan failed"
  ( cd "$STATE" && env -u FV_R2_SKIP_TREES timeout --signal=TERM --kill-after=15 "$CAP" \
    "$PY" "$HERE/r2-fetch.py" --root "$ROOT" --plan "$PLAN" --skip-trees '' --status "$STATUS/r2-fetch.json" ) \
    || { rc=$?; rm -f "$PLAN"; die "r2-fetch.py exit $rc"; }
  rm -f "$PLAN"
fi
_id="" _sec=""
log "weights $ROOT: $n volume trees linked from $VOL; $# R2 trees (${#todo[@]} fetched): $*"
