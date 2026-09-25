#!/usr/bin/env bash
# SSH-free Runpod driver: one RTX PRO 6000 pod on the CI image for a pushed
# commit, driven entirely through the REST API and the pod's HTTPS proxy.
# For hosts (sandboxes, CI) that can reach https://*.runpod.io but not TCP 22.
#
#   runpod-http.sh run [sha]     create pod, run $FV_FAMILY (default rtx6000), pull
#                                summaries + logs, destroy the pod
#   runpod-http.sh weights       weight gate only (verify-weights.sh), no cells
#   runpod-http.sh kernels [sha] fv-gpucheck kernels (every kernel vs host math)
#   runpod-http.sh all [sha]     kernels, then the $FV_FAMILY cells, on one pod
#                                (pod creation retries for FV_CREATE_WAIT_S)
#   runpod-http.sh upstream [sha] upstream Python references (scripts/gpu/upstream/pod.sh):
#                                UP_IMAGE_TARGET picks the baked image
#                                ghcr.io/…/fastvideo-rs-upstream-<target>:sha-<sha>;
#                                UP_STEPS / UP_CELLS select work. All writes stay on
#                                the container disk (UP_LOCAL=1).
#   runpod-http.sh attach <pod> <tag>  re-attach to a running pod (wait, collect, delete)
#   runpod-http.sh status <pod>  print live.log from a running pod
#   runpod-http.sh down <pod>    destroy a pod
#
# The pod's start command runs everything; results are published read-only on
# port 8000 (python http.server over /workspace/runs) and fetched through
# https://<pod>-8000.proxy.runpod.net. The pod is always deleted at the end,
# and a wall-clock cap (FV_POD_CAP_S, default 4h) deletes it regardless.
#
# RUNPOD_VOLUME_NAME may name several weight volumes; the one whose DC has
# stock for the GPU is used.
# Env: FV_FAMILY (runpod-matrix.sh family, default rtx6000; rtx5090 is the
# sol-engine RTX 5090 suite), RUNPOD_GPU_TYPE (default RTX PRO 6000), RUNPOD_API_KEY, RUNPOD_VOLUME_ID (default: volume named
# fv-weights-h3-ltx-hy), FV_CELLS (subset of cells), FV_GEN_TIMEOUT_S
# (per-cell cap, default 3600), FV_EXTRA_ENV (space-separated K=V for cells).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Bash reads a script as it runs; a runs lasts hours, so run from a private
# copy and let the repo copy be edited meanwhile.
if [[ -z "${FV_HTTP_COPY:-}" ]]; then
  copy="$(mktemp "${TMPDIR:-/tmp}/runpod-http.XXXXXX")"
  cp "${BASH_SOURCE[0]}" "$copy"
  FV_HTTP_COPY=1 FV_HTTP_HERE="$HERE" exec bash "$copy" "$@"
fi
HERE="${FV_HTTP_HERE:-$HERE}"
ROOT="$(cd "$HERE/../.." && pwd)"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GPU="${RUNPOD_GPU_TYPE:-NVIDIA RTX PRO 6000 Blackwell Server Edition}"
VOL_NAME="${RUNPOD_VOLUME_NAME:-fv-weights-h3-ltx-hy fv-weights-b200-us}"
MAX_DPH="${RUNPOD_GPU_MAX_DPH:-5}"
CAP_S="${FV_POD_CAP_S:-14400}"
FAMILY="${FV_FAMILY:-rtx6000}"
[[ "${1:-}" == upstream ]] && FAMILY="${FV_FAMILY:-upstream}"
OUT_ROOT="$ROOT/artifacts/runpod/$FAMILY"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"

die() { echo "runpod-http: $*" >&2; exit 1; }
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
rest() {
  local method="$1" path="$2" body="${3:-}"
  curl -sS --fail-with-body -X "$method" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${body:+-d "$body"} "$API$path"
}
proxy() { curl -sS --max-time 30 --fail "https://$1-8000.proxy.runpod.net/$2"; }

# The weight volume to mount. RUNPOD_VOLUME_NAME may list several volumes
# (space-separated, e.g. "fv-weights-h3-ltx-hy fv-weights-b200-us"); the first
# whose datacenter currently reports stock for $GPU wins, else the first.
volume() {
  if [[ -n "${RUNPOD_VOLUME_ID:-}" ]]; then
    rest GET "/networkvolumes/$RUNPOD_VOLUME_ID" | jq -r '"\(.id) \(.dataCenterId)"'
    return
  fi
  local vols name line first="" dc stock
  vols="$(rest GET /networkvolumes)"
  for name in $VOL_NAME; do
    line="$(jq -r --arg n "$name" '.[] | select(.name==$n) | "\(.id) \(.dataCenterId)"' <<<"$vols" | head -1)"
    [[ -n "$line" ]] || continue
    first="${first:-$line}"
    dc="${line#* }"
    stock="$(curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' \
      https://api.runpod.io/graphql -d "{\"query\":\"{ dataCenters { id gpuAvailability { gpuTypeId stockStatus } } }\"}" \
      | jq -r --arg dc "$dc" --arg g "$GPU" '.data.dataCenters[] | select(.id==$dc) | .gpuAvailability[]? | select(.gpuTypeId==$g) | .stockStatus // empty' 2>/dev/null || true)"
    if [[ -n "$stock" ]]; then
      echo "$line"
      return
    fi
  done
  echo "$first"
}

# $1 = image, $2 = run tag, $3 = mode (run|weights)
start_cmd() {
  local tag="$2" mode="$3"
  if [[ "$mode" == upstream ]]; then
    upstream_start_cmd "$tag"
    return
  fi
  local cells="fasth3-8step fasth3-4step-vsa fasth3-4step-dense sol-h3 sol-h3-spark ltx25-two-stage"
  cat <<EOF
set -u
# All writes on the container disk: the weight volume is read-only for us
# (some hosts have silently dropped data writes to it).
SCRATCH=/fvscratch
OUT=\$SCRATCH/runs/$FAMILY/$tag
mkdir -p "\$OUT"
FV=/opt/fastvideo-rs/target/release/fv-gpucheck
if "\$FV" serve --help >/dev/null 2>&1; then
  "\$FV" serve --dir \$SCRATCH/runs --port 8000 >"\$OUT/http.log" 2>&1 &
else
  # Images before fv-gpucheck serve: Python's server (needs apt on the box).
  ( apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq python3-minimal >/dev/null 2>&1
    cd \$SCRATCH/runs && exec python3 -m http.server 8000 ) >"\$OUT/http.log" 2>&1 &
fi
{
  echo "image_build_id=\$(cat /opt/fastvideo-rs/target/release/fv-gpucheck.build-id 2>/dev/null)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} >"\$OUT/box.txt" 2>&1
( cd /workspace/weights && for d in *; do echo "== \$d"; find -L "\$d" -maxdepth 3 \( -name '*.safetensors' -o -name '*.json' \) -printf '%s %p\n' 2>/dev/null | head -60; done ) >"\$OUT/tree.txt" 2>&1
FV_WEIGHTS=/workspace/weights bash /opt/fastvideo-rs/scripts/gpu/verify-weights.sh $cells >"\$OUT/weights.log" 2>&1
echo "exit=\$?" >>"\$OUT/weights.log"
# Tiny-autoencoder weights (taeh3, taeltx2_3_wide) onto the container disk;
# the matrix's *-taeh3 / *-taehv cells read them from \$SCRATCH/tae.
if [ -f /opt/fastvideo-rs/scripts/gpu/fetch-tae.sh ]; then
  bash /opt/fastvideo-rs/scripts/gpu/fetch-tae.sh \$SCRATCH/tae >"\$OUT/tae.log" 2>&1
  echo "exit=\$?" >>"\$OUT/tae.log"
fi
if [ "$mode" = kernels ] || [ "$mode" = all ]; then
  cd "\$OUT" && /opt/fastvideo-rs/target/release/fv-gpucheck --keep-going --out "\$OUT/gpucheck" kernels >"\$OUT/kernels.out" 2>&1
  echo "exit=\$?" >>"\$OUT/kernels.out"
  cp "\$OUT"/gpucheck/*.json "\$OUT"/ 2>/dev/null
fi
if [ "$mode" = run ] || [ "$mode" = all ]; then
  env ${FV_EXTRA_ENV:-} FV_WORK=/workspace FV_SCRATCH=\$SCRATCH FV_RUN_TAG=$tag FV_CELLS="${FV_CELLS:-}" \
    FV_GEN_TIMEOUT_S=${FV_GEN_TIMEOUT_S:-3600} \
    bash /opt/fastvideo-rs/scripts/gpu/runpod-matrix.sh $FAMILY >"\$OUT/matrix.out" 2>&1
  echo "matrix_exit=\$?" >>"\$OUT/matrix.out"
fi
echo "done $tag" >"\$OUT/DONE"
exec sleep infinity
EOF
}

# Upstream mode: no fv-gpucheck in the image. Serve /workspace/runs with the
# image's Python, clone this repo at $UP_SHA, and hand over to pod.sh.
upstream_start_cmd() {
  local tag="$1" runs=/workspace/runs
  # UP_LOCAL=1: nothing is written to the volume (it is read for weights only);
  # venvs, derived weights and results live on the container disk.
  [[ "${UP_LOCAL:-1}" == 1 ]] && runs=/root/runs
  cat <<EOF
set -u
OUT=$runs/$FAMILY/$tag
mkdir -p "\$OUT"
if fv-gpucheck serve --help >/dev/null 2>&1; then
  ( while true; do fv-gpucheck serve --dir $runs --port 8000; sleep 2; done ) >/tmp/http.log 2>&1 &
else
  ( cd $runs && while true; do python3 -m http.server 8000; sleep 2; done ) >/tmp/http.log 2>&1 &
fi
# The runner scripts come from this repo at \$UP_SHA (so a script fix needs no
# image rebuild); baked images keep a copy in /opt/fvrs as the fallback.
RUNNER=/opt/fvrs
{
  command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }
  git init -q /opt/fvrs-live && git -C /opt/fvrs-live remote add origin https://github.com/zaitrarrio/fastvideo-rs.git \
    && git -C /opt/fvrs-live fetch -q --depth 1 origin $UP_SHA && git -C /opt/fvrs-live checkout -q FETCH_HEAD \
    && echo live
} >"\$OUT/clone.log" 2>&1 && RUNNER=/opt/fvrs-live
env ${FV_EXTRA_ENV:-} UP_STEPS="${UP_STEPS:-info:box}" UP_STEPS_BG="${UP_STEPS_BG:-}" UP_CELLS="${UP_CELLS:-}" \
  UP_CELL_TIMEOUT_S=${UP_CELL_TIMEOUT_S:-5400} UP_LOCAL=${UP_LOCAL:-1} \
  bash \$RUNNER/scripts/gpu/upstream/pod.sh "\$OUT" >"\$OUT/pod.out" 2>&1
echo "done $tag" >"\$OUT/DONE"
exec sleep infinity
EOF
}

create_pod() {
  local image="$1" tag="$2" mode="$3" vol dc payload resp id dph
  read -r vol dc < <(volume) || true
  [[ -n "${vol:-}" ]] || die "no network volume ($VOL_NAME)"
  payload="$(jq -n --arg name "fv-$FAMILY-$tag" --arg image "$image" --arg vol "$vol" \
    --arg dc "$dc" --arg gpu "$GPU" --arg disk "${FV_CONTAINER_DISK_GB:-120}" --arg cmd "$(start_cmd "$image" "$tag" "$mode")" '{
      name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU",
      gpuTypeIds: [$gpu], gpuCount: 1, containerDiskInGb: ($disk|tonumber), volumeInGb: 0,
      networkVolumeId: $vol, volumeMountPath: "/workspace", dataCenterIds: [$dc],
      ports: ["8000/http"], dockerStartCmd: ["/bin/bash", "-c", $cmd]
    }')"
  log "create pod gpu=\"$GPU\" image=$image volume=$vol dc=$dc"
  # Capacity in the volume's datacenter comes and goes; retry instead of failing.
  local t0 wait="${FV_CREATE_WAIT_S:-3600}"
  t0=$(date +%s)
  until resp="$(rest POST /pods "$payload" 2>&1)"; do
    if [[ "$resp" != *"no instances currently available"* ]] || (( $(date +%s) - t0 >= wait )); then
      die "pod create failed: $resp"
    fi
    log "no $GPU free in $dc; retrying in 60s"
    sleep 60
  done
  id="$(jq -r '.id // empty' <<<"$resp")"
  [[ -n "$id" ]] || die "pod create returned no id: $resp"
  dph="$(jq -r '.costPerHr // 0' <<<"$resp")"
  if awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 > c+0)}'; then
    rest DELETE "/pods/$id" >/dev/null || true
    die "pod $id at \$$dph/hr exceeds cap \$$MAX_DPH"
  fi
  log "pod $id  \$$dph/hr  (hard cap ${CAP_S}s)"
  # Wall-clock backstop independent of this shell's lifetime.
  nohup bash -c "sleep $CAP_S; curl -sS -X DELETE -H 'Authorization: Bearer $RUNPOD_API_KEY' '$API/pods/$id' >/dev/null" >/dev/null 2>&1 &
  echo "$id"
}

# Mirror a python http.server directory listing recursively; files larger than
# FV_FETCH_MAX_MB (default 400) are skipped.
fetch_tree() {
  local id="$1" rel="$2" dest="$3" e name
  mkdir -p "$dest"
  for e in $(proxy "$id" "$rel" 2>/dev/null | grep -oE 'href="[^"]+"' | sed 's/href="//;s/"$//'); do
    case "$e" in
      ../ | /* | \?*) continue ;;
      */) fetch_tree "$id" "$rel$e" "$dest/${e%/}" ;;
      *)
        name="$(printf '%b' "${e//%/\\x}")"
        curl -sS --max-time 900 --fail --max-filesize $(( ${FV_FETCH_MAX_MB:-400} << 20 )) \
          "https://$id-8000.proxy.runpod.net/$rel$e" -o "$dest/$name" 2>/dev/null || rm -f "$dest/$name" ;;
    esac
  done
}

fetch_results() {
  local id="$1" tag="$2" out="$OUT_ROOT/$tag" f cell
  mkdir -p "$out"
  if [[ "$FAMILY" == upstream ]]; then
    fetch_tree "$id" "$FAMILY/$tag/" "$out"
    log "results → $out"
    return
  fi
  for f in box.txt tree.txt weights.log matrix.out live.log kernels.out kernels.json; do
    proxy "$id" "$FAMILY/$tag/$f" >"$out/$f" 2>/dev/null || true
  done
  for cell in $(proxy "$id" "$FAMILY/$tag/" 2>/dev/null | grep -oE 'href="[^"/]+/"' | sed 's/href="//;s/\/"//'); do
    mkdir -p "$out/$cell"
    for f in summary.json stderr.log stdout.log; do
      proxy "$id" "$FAMILY/$tag/$cell/$f" >"$out/$cell/$f" 2>/dev/null || rm -f "$out/$cell/$f"
    done
    for f in $(proxy "$id" "$FAMILY/$tag/$cell/gpucheck-out/" 2>/dev/null | grep -oE 'href="[^"/]+\.json"' | sed 's/href="//;s/"//'); do
      mkdir -p "$out/$cell/gpucheck-out"
      proxy "$id" "$FAMILY/$tag/$cell/gpucheck-out/$f" >"$out/$cell/gpucheck-out/$f" 2>/dev/null || true
    done
    # Paired-clip reports (compare_cells writes them under compare/).
    for f in $(proxy "$id" "$FAMILY/$tag/$cell/" 2>/dev/null | grep -oE 'href="[^"/]*compare[^"/]*\.json"' | sed 's/href="//;s/"//'); do
      proxy "$id" "$FAMILY/$tag/$cell/$f" >"$out/$cell/$f" 2>/dev/null || rm -f "$out/$cell/$f"
    done
  done
  log "results → $out"
}

wait_done() {
  local id="$1" tag="$2" t0 last=""
  t0=$(date +%s)
  while :; do
    # The marker carries the tag: a proxy error page for a vanished pod must
    # never read as "finished".
    if [[ "$(proxy "$id" "$FAMILY/$tag/DONE" 2>/dev/null | head -1)" == "done $tag" ]]; then return 0; fi
    # Early upstream pods touched an empty marker; a served empty file is not an error page.
    if [[ "$FAMILY" == upstream ]] && curl -sS --max-time 30 --fail -o /dev/null -w '%{size_download}' \
      "https://$id-8000.proxy.runpod.net/$FAMILY/$tag/DONE" 2>/dev/null | grep -qx 0; then return 0; fi
    local status
    status="$(rest GET "/pods/$id" 2>/dev/null | jq -r '.desiredStatus // "GONE"' 2>/dev/null || echo UNKNOWN)"
    if [[ "$status" == "GONE" || "$status" == "EXITED" || "$status" == "TERMINATED" ]]; then
      log "pod $id is $status before finishing (deleted outside this driver?)"
      return 1
    fi
    local cur
    cur="$(proxy "$id" "$FAMILY/$tag/live.log" 2>/dev/null | tail -1 || true)"
    [[ -n "$cur" && "$cur" != "$last" ]] && { log "pod: $cur"; last="$cur"; }
    (( $(date +%s) - t0 < CAP_S )) || { log "cap reached"; return 1; }
    sleep 30
  done
}

# A pod that never gets a runtime (bad host, stalled image pull) is replaced.
wait_up() {
  local id="$1" t0
  t0=$(date +%s)
  while (( $(date +%s) - t0 < ${FV_BOOT_WAIT_S:-1200} )); do
    proxy "$id" "$FAMILY/" >/dev/null 2>&1 && return 0
    sleep 30
  done
  return 1
}

cmd_run() {
  local mode="$1" sha="${2:-}" image tag id rc=0 attempt
  sha="${sha:-$(git -C "$ROOT" rev-parse --short=7 origin/main)}"
  image="${RUNPOD_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-$sha}"
  if [[ "$mode" == upstream ]]; then
    # UP_IMAGE_TARGET=fastvideo|sol-h3|sol-h3-4step|sol-ltx25 boots the baked
    # image (docker/upstream.Dockerfile, built by CI); unset = plain PyTorch image
    # and pod-side installs.
    if [[ -n "${UP_IMAGE_TARGET:-}" ]]; then
      image="${RUNPOD_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-upstream-$UP_IMAGE_TARGET:${UP_IMAGE_TAG:-sha-$sha}}"
    else
      image="${RUNPOD_IMAGE:-runpod/pytorch:1.3.3-cu1300-torch2130-ubuntu2404}"
    fi
    UP_SHA="$(git -C "$ROOT" rev-parse "$sha")"
  fi
  for attempt in 1 2 3; do
    tag="$sha-$(date -u +%m%d%H%M)"
    id="$(create_pod "$image" "$tag" "$mode")"
    wait_up "$id" && break
    log "pod $id never came up; replacing (attempt $attempt)"
    rest DELETE "/pods/$id" >/dev/null || true
    id=""
  done
  [[ -n "$id" ]] || die "no pod came up after 3 attempts"
  echo "$id" >"${TMPDIR:-/tmp}/fv-$FAMILY.pod"
  wait_done "$id" "$tag" || rc=1
  fetch_results "$id" "$tag"
  if grep -q "matrix_exit=[1-9]" "$OUT_ROOT/$tag/matrix.out" 2>/dev/null; then
    log "matrix exited abnormally: $(grep matrix_exit "$OUT_ROOT/$tag/matrix.out")"
    rc=1
  fi
  log "destroy pod $id"
  rest DELETE "/pods/$id" >/dev/null || log "WARN: delete failed for $id"
  cat "$OUT_ROOT/$tag/weights.log" >&2 || true
  return $rc
}

case "${1:-}" in
  run) shift; cmd_run run "$@" ;;
  weights) shift; cmd_run weights "$@" ;;
  kernels) shift; cmd_run kernels "$@" ;;
  all) shift; cmd_run all "$@" ;;
  upstream) shift; cmd_run upstream "$@" ;;
  attach)
    # Re-attach to a running pod whose driver died: wait, collect, delete.
    id="${2:?pod}"; tag="${3:?tag (e.g. 33a2eac-09251907)}"; rc=0
    wait_done "$id" "$tag" || rc=1
    fetch_results "$id" "$tag"
    log "destroy pod $id"
    rest DELETE "/pods/$id" >/dev/null || log "WARN: delete failed for $id"
    exit $rc ;;
  status) proxy "${2:?pod}" "$FAMILY/" ;;
  down) rest DELETE "/pods/${2:?pod}" >/dev/null && echo "deleted ${2}" ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
