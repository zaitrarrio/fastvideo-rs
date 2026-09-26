#!/usr/bin/env bash
# GPU oracle diff in one command (docs/oracle.md): the Python references and
# our Rust pipelines on the same GPU type, the reference's noise and text
# conditioning injected into ours, every step's latents and the first step's
# block outputs compared tensor by tensor.
#
#   scripts/gpu/oracle.sh [sha]
#
# 1. One upstream pod per reference stack (runpod-http.sh upstream, kept alive
#    with FV_KEEP_POD=1): the FastVideo image for the fasth3-* targets, the
#    sol-ltx25 image for the ltx25-* targets. Each runs
#    scripts/gpu/upstream/oracle.sh and serves oracle-<target>/oracle-dump.tar.
# 2. One runtime pod (runpod-http.sh run, FV_FAMILY=oracle) that downloads each
#    reference dump as soon as it is ready, runs our pipeline on it and
#    compares (runpod-matrix.sh `oracle`).
# 3. Every pod this script created is deleted; results land under
#    artifacts/runpod/{upstream,oracle}/<tag>/ (the dumps themselves are not
#    fetched; oracle-<target>-diff/gpucheck-out/*.json holds the numbers).
#
# Env: ORACLE_TARGETS (default: fasth3-8step fasth3-4step-vsa fasth3-8step-vsa0
# fasth3-4step-dense ltx25-512p-dense
# ltx25-512p), FASTVIDEO_DUMP_OPS (blocks whose inside is dumped, default
# 0,1,24,47), FV_ORACLE_F32 (1: the f32-activation control for every target,
# 0: none; default H3 only), plus runpod-http.sh's (RUNPOD_API_KEY, ...).
# Logs: $ORACLE_LOG_DIR (default /tmp/claude-0) oracle-*.log.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A run lasts hours and bash reads a script as it goes: run from a private
# copy so the repo copy can be edited meanwhile.
if [[ -z "${FV_ORACLE_COPY:-}" ]]; then
  copy="$(mktemp "${TMPDIR:-/tmp}/oracle.XXXXXX")"
  cp "${BASH_SOURCE[0]}" "$copy"
  FV_ORACLE_COPY=1 FV_ORACLE_HERE="$HERE" exec bash "$copy" "$@"
fi
HERE="${FV_ORACLE_HERE:-$HERE}"
ROOT="$(cd "$HERE/../.." && pwd)"
sha="${1:-$(git -C "$ROOT" rev-parse --short=7 origin/main)}"
targets="${ORACLE_TARGETS:-fasth3-8step fasth3-4step-vsa fasth3-8step-vsa0 fasth3-4step-dense ltx25-512p-dense ltx25-512p}"
ops="${FASTVIDEO_DUMP_OPS:-0,1,24,47}"
logs="${ORACLE_LOG_DIR:-/tmp/claude-0}"
work="$(mktemp -d "${TMPDIR:-/tmp}/fv-oracle.XXXXXX")"
mkdir -p "$logs"
log() { printf '[%s] oracle: %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$logs/oracle-driver.log" >&2; }

fv=() ltx=()
for t in $targets; do
  case "$t" in
    fasth3-*) fv+=("$t") ;;
    ltx25-*) ltx+=("$t") ;;
    *) log "unknown target $t"; exit 2 ;;
  esac
done

declare -A bg=() tg=()
pods=()
cleanup() {
  local p
  for p in "${pods[@]}"; do
    log "delete upstream pod $p"
    bash "$HERE/runpod-http.sh" down "$p" >/dev/null 2>&1 || log "WARN: delete of $p failed"
  done
}
trap cleanup EXIT

# start_up <image target> <oracle targets> <weight steps> <container disk GB> [env]: the
# upstream driver in the background; a pod create that fails outright (the API
# answers 500 now and then) is retried twice.
start_up() {
  local image="$1" list="$2" wsteps="$3" disk="$4" extra="${5:-}" csv
  csv="${list// /,}"
  tg[$image]="$list"
  log "upstream pod ($image): $wsteps oracle:$csv"
  (
    for attempt in 1 2 3; do
      FV_KEEP_POD=1 FV_POD_FILE="$work/$image.pod" UP_IMAGE_TARGET="$image" \
        UP_STEPS="info:box $wsteps oracle:$csv" UP_CELL_TIMEOUT_S="${UP_CELL_TIMEOUT_S:-7200}" \
        FV_EXTRA_ENV="FASTVIDEO_DUMP_OPS=$ops${extra:+ $extra}" FV_FETCH_SKIP='oracle-dump\.tar|/dump/|\.mp4$' FV_CONTAINER_DISK_GB="$disk" \
        bash "$HERE/runpod-http.sh" upstream "$sha" >>"$logs/oracle-up-$image.log" 2>&1 && exit 0
      [[ -s "$work/$image.pod" ]] && exit 1   # the pod came up; its run failed
      log "upstream $image: no pod (attempt $attempt)"
      sleep 60
    done
    exit 1
  ) &
  bg[$image]=$!
}

(( ${#fv[@]} )) && start_up fastvideo "${fv[*]}" "weights:fasth3-8step weights:h3-diffusers" 150
# RECON_ACCEPT_MISMATCH=1: the LTX-2.5 single-file packs are rebuilt from the
# Diffusers copy, and two of them (Gemma layers 12-47, the DiT's embedding
# connectors) carry the Diffusers numbers -- the ones our port loads -- in the
# original layout (upstream/pod.sh weights_ltx25); without it the rebuild stops.
(( ${#ltx[@]} )) && start_up sol-ltx25 "${ltx[*]}" "weights:ltx25" 250 RECON_ACCEPT_MISMATCH=1

# Wait for each upstream pod to be up (runpod-http writes "<id> <tag>"); a
# stack whose pod never comes up drops its targets.
urls="" targets=""
for image in "${!bg[@]}"; do
  f="$work/$image.pod"
  until [[ -s "$f" ]]; do
    if ! kill -0 "${bg[$image]}" 2>/dev/null; then
      log "upstream $image never came up (see $logs/oracle-up-$image.log); dropping ${tg[$image]}"
      continue 2
    fi
    sleep 20
  done
  read -r id tag <"$f"
  pods+=("$id")
  urls+="${urls:+ }https://$id-8000.proxy.runpod.net/upstream/$tag"
  targets+="${targets:+ }${tg[$image]}"
  log "upstream $image pod $id tag $tag"
done
[[ -n "$targets" ]] || { log "no upstream pod came up"; exit 1; }

log "runtime pod: targets $targets"
rc=0
FV_FAMILY=oracle FV_EXTRA_ENV="FV_ORACLE_URL='$urls' FV_ORACLE_TARGETS='$targets' FASTVIDEO_DUMP_OPS=$ops${FV_ORACLE_F32:+ FV_ORACLE_F32=$FV_ORACLE_F32}" \
  FV_GEN_TIMEOUT_S="${FV_GEN_TIMEOUT_S:-5400}" \
  bash "$HERE/runpod-http.sh" run "$sha" >"$logs/oracle-rust.log" 2>&1 || rc=$?
log "runtime pod finished rc=$rc"
for image in "${!bg[@]}"; do
  wait "${bg[$image]}" || log "upstream driver $image rc=$?"
done
log "results: artifacts/runpod/oracle and artifacts/runpod/upstream"
exit $rc
