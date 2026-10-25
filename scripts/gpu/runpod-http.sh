#!/usr/bin/env bash
# SSH-free Runpod driver: one RTX PRO 6000 pod on the CI image for a pushed
# commit, driven entirely through the REST API and the pod's HTTPS proxy.
# For hosts (sandboxes, CI) that can reach https://*.runpod.io but not TCP 22.
#
#   runpod-http.sh run [sha]     create pod, run $FV_FAMILY (default rtx6000), pull
#                                summaries + logs, destroy the pod. A family retired
#                                to fv-lab (fv-lab-retired.sh: rtx6000, techniques,
#                                b6, rtx5090, fastvideo, h3arms, trace, ltxfps,
#                                ltxvae, precision; the restored modes precision-nvfp4,
#                                ltxvae-gen) runs `fv-lab pod up --yes --wait`
#                                instead (docs/adr/0019-fv-lab.md §10, §12); FV_DRY_RUN=1
#                                prints the route and the command, creates nothing
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
#   runpod-http.sh plan [sha] [tag]  print the pod body `run` would send (no pod, no
#                                price check): the payload golden of fv-lab pod up
#                                (scripts/gpu/tests/fv-lab-goldens.sh). The volume's
#                                datacentre comes from FV_PLAN_VOLUME="<id> <dc>" or,
#                                without it, the API lookup `run` makes. For a
#                                retired family: `fv-lab pod up` without --yes (its plan)
#
# The pod's start command runs everything; results are published read-only on
# port 8000 (python http.server over /workspace/runs) and fetched through
# https://<pod>-8000.proxy.runpod.net. The pod is always deleted at the end,
# and a wall-clock cap (FV_POD_CAP_S, default 4h) deletes it regardless: the
# driver's backstop, and the pod's own cap watchdog (start_cmd: FV_POD_CAP_S
# after the container starts, the pod deletes itself with the pod-scoped key,
# so a dead driver cannot leave it running); the pod also deletes itself after
# FV_IDLE_GPU_MIN (default 10) minutes at 0 % GPU once it has used the GPU
# (idle_watchdog). Both self-deletes are GraphQL podTerminate: Runpod refuses
# the pod-scoped key a REST DELETE (HTTP 403; ADR-0023 §4).
#
# RUNPOD_VOLUME_NAME may name several weight volumes; the one whose DC has
# stock for the GPU is used. RUNPOD_NO_VOLUME=1 mounts none and takes any
# datacenter (weight-free work: kernels, upstream bench:attn / bench:attn_dc);
# RUNPOD_CLOUD_TYPE=COMMUNITY allows community hosts; RUNPOD_ALLOWED_CUDA
# (a space list, e.g. "13.0") restricts hosts to exactly those CUDA labels. The
# default is scripts/gpu/runpod-cuda.sh fv_runpod_cuda: none today, because
# REST cannot name the 13.1 / 13.2 hosts ("13.0" alone hides every RTX PRO
# 6000); instead the pod checks its driver (>= FV_MIN_DRIVER, default 580) and
# stops early if older.
# Env: FV_FAMILY (runpod-matrix.sh family, default rtx6000; rtx5090 is the
# sol-engine RTX 5090 suite), RUNPOD_GPU_TYPE (default RTX PRO 6000), RUNPOD_API_KEY, RUNPOD_VOLUME_ID (default: volume named
# fv-weights-h3-ltx-500), FV_CELLS (subset of cells), FV_GEN_TIMEOUT_S
# (per-cell cap, default 3600), FV_PROMPTS / FV_LPIPS (forwarded to
# runpod-matrix.sh), FV_EXTRA_ENV (space-separated K=V for cells),
# FV_POD_FILES (space-separated local files, at most a few hundred KB in all,
# unpacked on the pod into /fvscratch/files/<basename>; run mode only),
# FV_AVOID_MACHINES (space-separated Runpod machine ids: a pod placed on one
# is deleted at once and created again),
# FV_POD_SCRIPT (the basename of one of FV_POD_FILES, run instead of the
# image's runpod-matrix.sh with the same environment, e.g. sol-bench-pod.sh;
# it skips the default weight gate).
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
# shellcheck source-path=SCRIPTDIR source=runpod-price.sh
source "$HERE/runpod-price.sh"
# shellcheck source-path=SCRIPTDIR source=volumes.sh
source "$HERE/volumes.sh"
# shellcheck source-path=SCRIPTDIR source=runpod-cuda.sh
source "$HERE/runpod-cuda.sh"
# shellcheck source-path=SCRIPTDIR source=fv-lab-retired.sh
source "$HERE/fv-lab-retired.sh"
ROOT="$(cd "$HERE/../.." && pwd)"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GPU="${RUNPOD_GPU_TYPE:-NVIDIA RTX PRO 6000 Blackwell Server Edition}"
# EU only since 2026-10 (scripts/gpu/volumes.sh); the US name joins the list
# again once FV_US_VOLUME_NAME is set there.
VOL_NAME="${RUNPOD_VOLUME_NAME:-$FV_EU_VOLUME_NAME${FV_US_VOLUME_NAME:+ $FV_US_VOLUME_NAME}}"
# shellcheck disable=SC2086 # a space-separated list of names
fv_check_volume $VOL_NAME "${RUNPOD_VOLUME_ID:-}"
MAX_DPH="${RUNPOD_GPU_MAX_DPH:-5}"
CAP_S="${FV_POD_CAP_S:-14400}"
FAMILY="${FV_FAMILY:-rtx6000}"
[[ "${1:-}" == upstream ]] && FAMILY="${FV_FAMILY:-upstream}"
OUT_ROOT="$ROOT/artifacts/runpod/$FAMILY"
# A dry run (FV_DRY_RUN=1: the route and the command, nothing created) needs no key.
[[ "${FV_DRY_RUN:-0}" == 1 ]] || : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"

die() { echo "runpod-http: $*" >&2; exit 1; }
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
rest() {
  local method="$1" path="$2" body="${3:-}"
  curl -sS --fail-with-body -X "$method" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${body:+-d "$body"} "$API$path"
}
proxy() { curl -sS --max-time 30 --fail "https://$1-8000.proxy.runpod.net/$2"; }

# The weight volume to mount. RUNPOD_VOLUME_NAME may list several volumes
# (space-separated, e.g. "fv-weights-h3-ltx-500 <a rebuilt US volume>"); the first
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

# The pod-side idle watchdog: the pod deletes itself (pod-scoped
# RUNPOD_API_KEY / RUNPOD_POD_ID, injected by Runpod) after FV_IDLE_GPU_MIN
# minutes (default 10; 0 disables) at 0 % GPU utilization, counted once the
# GPU has been busy; before that, after FV_IDLE_GPU_GRACE_MIN (default 30:
# setup, weight reads). Independent of this shell, like the wall-clock cap.
# With `net` (upstream pods), a minute in which the pod sent at least
# FV_IDLE_NET_BYTES (default 256 KiB) also counts as busy: a reference pod
# whose GPU work is done is still in use while its runtime pod downloads the
# dumps, and scripts/gpu/oracle.sh fetches its 1 MiB keepalive.bin every
# minute while the runtime pod is up. When the heartbeats stop (the runtime
# pod is gone, or the driver died) the pod goes after FV_IDLE_GPU_MIN.
idle_watchdog() {
  local idle="${FV_IDLE_GPU_MIN:-10}" grace="${FV_IDLE_GPU_GRACE_MIN:-30}" net="${1:-}"
  [[ "$idle" == 0 ]] && return 0
  cat <<WDEOF
( used=0; n=0; tx0=0
  fv_self_delete() {
    # GraphQL podTerminate: the pod-scoped key may not DELETE through REST (HTTP 403,
    # docs/adr/0023-lab-report-and-ci-bridge.md §4). Success: HTTP 200 and no "errors".
    # The key reaches curl on stdin (-H @-), never in its argv; the answer is logged, then removed.
    _fvsd_code=\$(printf 'Authorization: Bearer %s\\n' "\$RUNPOD_API_KEY" | curl -sS --max-time 30 -o "\$1.resp" \\
      -w '%{http_code}' -X POST https://api.runpod.io/graphql -H 'Content-Type: application/json' -H @- \\
      --data "{\\"query\\":\\"mutation { podTerminate(input: {podId: \\\\\\"\$RUNPOD_POD_ID\\\\\\"}) }\\"}" 2>>"\$1")
    { cat "\$1.resp"; echo " http=\$_fvsd_code"; } >>"\$1" 2>/dev/null
    _fvsd_ok=1
    [ "\$_fvsd_code" = 200 ] && ! grep -q '"errors"' "\$1.resp" 2>/dev/null && _fvsd_ok=0
    rm -f "\$1.resp"
    return "\$_fvsd_ok"
  }
  txb() { awk 'NR > 2 { sub(/^[ \t]+/, ""); split(\$0, f, /[: ]+/); if (f[1] != "lo") s += f[10] } END { printf "%.0f", s }' /proc/net/dev 2>/dev/null || echo 0; }
  [ "$net" = net ] && tx0=\$(txb)
  while sleep 60; do
    u=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1 | tr -d ' ')
    sent=0
    if [ "$net" = net ]; then tx1=\$(txb); sent=\$(( tx1 - tx0 )); tx0=\$tx1; fi
    if [ "\${u:-0}" -gt 0 ] 2>/dev/null || [ "\$sent" -ge ${FV_IDLE_NET_BYTES:-262144} ] 2>/dev/null; then used=1; n=0; else n=\$((n + 1)); fi
    lim=$grace; [ "\$used" = 1 ] && lim=$idle
    if [ "\$n" -ge "\$lim" ] && [ -n "\${RUNPOD_API_KEY:-}" ] && [ -n "\${RUNPOD_POD_ID:-}" ]; then
      echo "\$(date -u +%FT%TZ) idle \$n min at 0 % GPU: deleting pod \$RUNPOD_POD_ID" >>/tmp/idle-watchdog.log
      fv_self_delete /tmp/idle-watchdog.log
    fi
  done ) >/dev/null 2>&1 &
WDEOF
}

# $1 = image, $2 = run tag, $3 = mode (run|weights)
start_cmd() {
  local tag="$2" mode="$3"
  if [[ "$mode" == upstream ]]; then
    upstream_start_cmd "$tag"
    return
  fi
  local cells="fasth3-8step fasth3-4step-vsa fasth3-4step-dense sol-h3 sol-h3-spark ltx25-two-stage"
  idle_watchdog
  cat <<EOF
set -u
# The pod's own wall cap (FV_POD_CAP_S, docs/adr/0023-lab-report-and-ci-bridge.md §4):
# $CAP_S s after the container starts it deletes itself with the pod-scoped key
# Runpod injects (GraphQL podTerminate), retrying each minute, whatever the driver's host did.
if [ -n "\${RUNPOD_API_KEY:-}" ] && [ -n "\${RUNPOD_POD_ID:-}" ]; then
  ( fv_self_delete() {
      # GraphQL podTerminate: the pod-scoped key may not DELETE through REST (HTTP 403,
      # docs/adr/0023-lab-report-and-ci-bridge.md §4). Success: HTTP 200 and no "errors".
      # The key reaches curl on stdin (-H @-), never in its argv; the answer is logged, then removed.
      _fvsd_code=\$(printf 'Authorization: Bearer %s\\n' "\$RUNPOD_API_KEY" | curl -sS --max-time 30 -o "\$1.resp" \\
        -w '%{http_code}' -X POST https://api.runpod.io/graphql -H 'Content-Type: application/json' -H @- \\
        --data "{\\"query\\":\\"mutation { podTerminate(input: {podId: \\\\\\"\$RUNPOD_POD_ID\\\\\\"}) }\\"}" 2>>"\$1")
      { cat "\$1.resp"; echo " http=\$_fvsd_code"; } >>"\$1" 2>/dev/null
      _fvsd_ok=1
      [ "\$_fvsd_code" = 200 ] && ! grep -q '"errors"' "\$1.resp" 2>/dev/null && _fvsd_ok=0
      rm -f "\$1.resp"
      return "\$_fvsd_ok"
    }
    sleep $CAP_S
    while :; do
      echo "\$(date -u +%FT%TZ) wall cap $CAP_S s: deleting pod \$RUNPOD_POD_ID" >>/tmp/cap-watchdog.log
      fv_self_delete /tmp/cap-watchdog.log && break
      sleep 60
    done ) >/dev/null 2>&1 &
fi
# All writes on the container disk: the weight volume is read-only for us
# (some hosts have silently dropped data writes to it).
SCRATCH=/fvscratch
OUT=\$SCRATCH/runs/$FAMILY/$tag
mkdir -p "\$OUT"
# FV_POD_FILES: local files shipped in the pod's env (a base64 tar.gz),
# unpacked into \$SCRATCH/files (e.g. FV_HD_POST_URL=file:///fvscratch/files/hd-upscaler.sh).
if [ -n "\${FV_POD_FILES_TGZ:-}" ]; then
  mkdir -p \$SCRATCH/files && echo "\$FV_POD_FILES_TGZ" | base64 -d | tar -xz -C \$SCRATCH/files
fi
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
# The CUDA 13 image needs driver >= ${FV_MIN_DRIVER:-580}. Hosts are not
# filtered by allowedCudaVersions by default (that filter can hide stock), so
# stop here on an older driver instead of failing later.
drv=\$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
if [ -n "\$drv" ] && [ "\$drv" -lt ${FV_MIN_DRIVER:-580} ]; then
  echo "driver_too_old=\$drv" >>"\$OUT/box.txt"
  echo "done $tag" >"\$OUT/DONE"
  exec sleep infinity
fi
# kernels mode needs no weights (and may run without a volume).
if [ "$mode" != kernels ]; then
( cd /workspace/weights && for d in *; do echo "== \$d"; find -L "\$d" -maxdepth 3 \( -name '*.safetensors' -o -name '*.json' \) -printf '%s %p\n' 2>/dev/null | head -60; done ) >"\$OUT/tree.txt" 2>&1
# A pod script (FV_POD_SCRIPT) gates each of its cells on its own trees.
if [ -z "${FV_POD_SCRIPT:-}" ]; then
FV_WEIGHTS=/workspace/weights bash /opt/fastvideo-rs/scripts/gpu/verify-weights.sh $cells >"\$OUT/weights.log" 2>&1
echo "exit=\$?" >>"\$OUT/weights.log"
fi
# Tiny-autoencoder weights (taeh3, taeltx2_3_wide) onto the container disk;
# the matrix's *-taeh3 / *-taehv cells read them from \$SCRATCH/tae.
if [ "${FV_SKIP_TAE:-0}" != 1 ] && [ -f /opt/fastvideo-rs/scripts/gpu/fetch-tae.sh ]; then
  bash /opt/fastvideo-rs/scripts/gpu/fetch-tae.sh \$SCRATCH/tae >"\$OUT/tae.log" 2>&1
  echo "exit=\$?" >>"\$OUT/tae.log"
fi
fi
if [ "$mode" = kernels ] || [ "$mode" = all ]; then
  cd "\$OUT" && env ${FV_EXTRA_ENV:-} /opt/fastvideo-rs/target/release/fv-gpucheck --keep-going --out "\$OUT/gpucheck" kernels >"\$OUT/kernels.out" 2>&1
  echo "exit=\$?" >>"\$OUT/kernels.out"
  cp "\$OUT"/gpucheck/*.json "\$OUT"/ 2>/dev/null
fi
if [ "$mode" = run ] || [ "$mode" = all ]; then
  env ${FV_EXTRA_ENV:-} FV_WORK=/workspace FV_SCRATCH=\$SCRATCH FV_RUN_TAG=$tag FV_CELLS="${FV_CELLS:-}" \
    ${FV_PROMPTS:+FV_PROMPTS="$FV_PROMPTS"} ${FV_LPIPS:+FV_LPIPS=$FV_LPIPS} \
    FV_GEN_TIMEOUT_S=${FV_GEN_TIMEOUT_S:-3600} \
    bash ${FV_POD_SCRIPT:+/fvscratch/files/}${FV_POD_SCRIPT:-/opt/fastvideo-rs/scripts/gpu/runpod-matrix.sh} $FAMILY >"\$OUT/matrix.out" 2>&1
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
  idle_watchdog net
  cat <<EOF
set -u
OUT=$runs/$FAMILY/$tag
mkdir -p "\$OUT"
# What oracle.sh fetches as its heartbeat while the runtime pod is up.
head -c 1048576 /dev/urandom >$runs/keepalive.bin 2>/dev/null || true
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

# $1 = image, $2 = run tag, $3 = mode, $4 = volume id, $5 = its datacentre:
# the PodCreateInput `create_pod` sends (and `plan` prints).
pod_payload() {
  local image="$1" tag="$2" mode="$3" vol="$4" dc="$5" files="" f
  if [[ -n "${FV_POD_FILES:-}" ]]; then
    local ft
    ft="$(mktemp -d)"
    for f in $FV_POD_FILES; do cp "$f" "$ft/" || die "FV_POD_FILES: $f"; done
    files="$(tar -czf - -C "$ft" . | base64 -w0)"
    rm -rf "$ft"
  fi
  jq -n --arg files "$files" --arg name "${FV_POD_PREFIX:-fv}-$FAMILY-$tag" --arg image "$image" --arg vol "$vol" \
    --arg dc "$dc" --arg gpu "$GPU" --arg disk "${FV_CONTAINER_DISK_GB:-120}" --arg cmd "$(start_cmd "$image" "$tag" "$mode")" \
    --arg cloud "${RUNPOD_CLOUD_TYPE:-SECURE}" --arg cuda "${RUNPOD_ALLOWED_CUDA-$(fv_runpod_cuda)}" '{
      name: $name, imageName: $image, cloudType: $cloud, computeType: "GPU",
      gpuTypeIds: [$gpu], gpuCount: 1, containerDiskInGb: ($disk|tonumber), volumeInGb: 0,
      ports: ["8000/http"], dockerStartCmd: ["/bin/bash", "-c", $cmd]
    } + (if $cuda == "" then {} else {allowedCudaVersions: ($cuda | split(" "))} end)
      + (if $files == "" then {} else {env: {FV_POD_FILES_TGZ: $files}} end)
      + (if $vol == "" then {} else
      {networkVolumeId: $vol, volumeMountPath: "/workspace", dataCenterIds: [$dc]} end)'
}

create_pod() {
  local image="$1" tag="$2" mode="$3" vol="" dc="" payload resp id dph
  # RUNPOD_NO_VOLUME=1: no network volume, any datacenter with stock (for
  # work that needs no weights: `kernels`, upstream `bench:attn*`).
  # RUNPOD_CLOUD_TYPE=COMMUNITY takes community hosts (default SECURE).
  if [[ "${RUNPOD_NO_VOLUME:-0}" != 1 ]]; then
    read -r vol dc < <(volume) || true
    [[ -n "${vol:-}" ]] || die "no network volume ($VOL_NAME)"
  fi
  payload="$(pod_payload "$image" "$tag" "$mode" "$vol" "$dc")"
  # The price cap on Runpod's quote, BEFORE the create (scripts/gpu/runpod-price.sh).
  fv_runpod_price_ok "$GPU" "$MAX_DPH" "${RUNPOD_CLOUD_TYPE:-SECURE}" "$dc" || die "not creating a pod on \"$GPU\" over the \$$MAX_DPH/hr cap"
  log "create pod gpu=\"$GPU\" image=$image volume=${vol:-none} dc=${dc:-any} cloud=${RUNPOD_CLOUD_TYPE:-SECURE} quoted=\$$FV_QUOTED_DPH/hr"
  # Capacity in the volume's datacenter comes and goes; retry instead of failing.
  local t0 wait="${FV_CREATE_WAIT_S:-3600}"
  t0=$(date +%s)
  # FV_AVOID_MACHINES: Runpod machine ids a pod is deleted from on sight (a
  # host that never starts the image; 2026-10-06 s3p8exc9lcvi took five pods).
  while :; do
    until resp="$(rest POST /pods "$payload" 2>&1)"; do
      if [[ "$resp" != *"no instances currently available"* && "$resp" != *"does not have the resources"* ]] \
        || (( $(date +%s) - t0 >= wait )); then
        die "pod create failed: $resp"
      fi
      log "no $GPU free in $dc; retrying in 60s"
      sleep 60
    done
    id="$(jq -r '.id // empty' <<<"$resp")"
    [[ -n "$id" ]] || die "pod create returned no id: $resp"
    local machine
    machine="$(jq -r '.machineId // empty' <<<"$resp")"
    [[ -n "$machine" ]] || machine="$(rest GET "/pods/$id" 2>/dev/null | jq -r '.machineId // empty' 2>/dev/null || true)"
    if [[ -n "$machine" && " ${FV_AVOID_MACHINES:-} " == *" $machine "* ]]; then
      rest DELETE "/pods/$id" >/dev/null || true
      (( $(date +%s) - t0 < wait )) || die "only avoided hosts ($machine) offered within ${wait}s"
      log "pod $id landed on avoided host $machine; deleted, retrying in 45s"
      sleep 45
      continue
    fi
    break
  done
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
    esac
    # FV_FETCH_SKIP: an extended regex of relative paths not to fetch.
    if [[ -n "${FV_FETCH_SKIP:-}" ]] && grep -qE "$FV_FETCH_SKIP" <<<"$rel$e"; then continue; fi
    case "$e" in
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
  # FV_FETCH_TREE=1 mirrors the whole run directory (wavs, mp4s; FV_FETCH_SKIP
  # drops e.g. frame PNGs), as the upstream family always does.
  if [[ "$FAMILY" == upstream || "${FV_FETCH_TREE:-0}" == 1 ]]; then
    fetch_tree "$id" "$FAMILY/$tag/" "$out"
    log "results → $out"
    return
  fi
  for f in box.txt tree.txt weights.log matrix.out live.log kernels.out kernels.json lpips-fetch.log sysinfo.txt sampler.log; do
    proxy "$id" "$FAMILY/$tag/$f" >"$out/$f" 2>/dev/null || true
  done
  # The run directory's listing names the cells. One failed proxy read here
  # once lost every cell of a finished run (the pod is deleted right after),
  # so it is retried.
  local cells="" attempt
  for attempt in 1 2 3 4 5; do
    cells="$(proxy "$id" "$FAMILY/$tag/" 2>/dev/null | grep -oE 'href="[^"/]+/"' | sed 's/href="//;s/\/"//' || true)"
    [[ -n "$cells" ]] && break
    log "cell listing empty (attempt $attempt); retrying"
    sleep 10
  done
  for cell in $cells; do
    mkdir -p "$out/$cell"
    for f in summary.json stderr.log stdout.log; do
      proxy "$id" "$FAMILY/$tag/$cell/$f" >"$out/$cell/$f" 2>/dev/null || rm -f "$out/$cell/$f"
    done
    for f in $(proxy "$id" "$FAMILY/$tag/$cell/gpucheck-out/" 2>/dev/null | grep -oE 'href="[^"/]+\.json"' | sed 's/href="//;s/"//'); do
      mkdir -p "$out/$cell/gpucheck-out"
      proxy "$id" "$FAMILY/$tag/$cell/gpucheck-out/$f" >"$out/$cell/gpucheck-out/$f" 2>/dev/null || true
    done
    # benchmark.json, and the compare/ and gate/ reports (compare_cells,
    # gate_cells): every JSON directly in the cell directory.
    for f in $(proxy "$id" "$FAMILY/$tag/$cell/" 2>/dev/null | grep -oE 'href="[^"/]*\.json"' | sed 's/href="//;s/"//'); do
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

# The families retired to fv-lab (fv-lab-retired.sh, docs/adr/0019-fv-lab.md
# §10) have no runpod-matrix.sh branch: `run` and `plan` hand them to
# `fv-lab pod up` (its pod on fv-deploy's guards, its run, fetch and delete).
#
# The fv-lab binary: FV_LAB_BIN, else `fv-lab` on PATH, else ./fv-lab at the
# repository root (where docs/dev/fv-lab.md "Build" fetches it).
fv_lab_bin() {
  if [[ -n "${FV_LAB_BIN:-}" ]]; then
    [[ -x "$FV_LAB_BIN" ]] || { echo "runpod-http: FV_LAB_BIN=$FV_LAB_BIN is not an executable" >&2; return 1; }
    echo "$FV_LAB_BIN"
  elif command -v fv-lab >/dev/null 2>&1; then
    command -v fv-lab
  elif [[ -x "$ROOT/fv-lab" ]]; then
    echo "$ROOT/fv-lab"
  else
    return 1
  fi
}

# tier_value <tier> <key>: a top-level value of that [[tier]] in
# lab/tiers.toml (quotes and a trailing comment stripped), empty if absent.
tier_value() {
  awk -v t="$1" -v k="$2" '
    /^\[\[/ { in_t = 0; named = 0; next }
    /^name = "/ && !named { n = $0; gsub(/^name = "|"[[:space:]]*(#.*)?$/, "", n); in_t = (n == t); named = 1; next }
    in_t && $1 == k && $2 == "=" {
      v = $0; sub(/^[^=]*=[[:space:]]*/, "", v); sub(/[[:space:]]+#.*$/, "", v); gsub(/^"|"$/, "", v); print v; exit
    }' "$ROOT/lab/tiers.toml"
}

# What `run` reads from FV_EXTRA_ENV on the script path but no tier reads: the
# dropped modes' knobs (ADR-0019 §9.4, §10.4; the two modes restored as tiers,
# §12, are named by fv_lab_route; b4a's FV_B4A_CELLS since its branch went,
# §11.6, answered with FV_CELLS), the prompts fv-lab ignores (§7), and what
# the start command sets after FV_EXTRA_ENV anyway.
FV_LAB_NOT_READ="FV_PRECISION_ARM FV_NVFP4_WORKLOADS FV_VAE_GEN FV_VAE_TRACE FV_VAE_UNTRACED FV_VAE_WORKLOADS
  FV_LTXFPS FV_LTXFPS_FRAMES FV_TECH_CELLS FV_B6_CELLS FV_B4A_CELLS FV_PROMPTS FV_PROMPT FV_SEED
  FV_WORK FV_SCRATCH FV_RUN_TAG FV_CELLS FV_GEN_TIMEOUT_S"

# fv_lab_route run|plan <sha> [tag]: `fv-lab pod up` for $FAMILY with this
# driver's settings mapped across (FV_CELLS -> --cells, FV_BASELINE_SHA ->
# --baseline-sha, FV_EXTRA_ENV -> --extra-env, FV_LPIPS=1 -> --lpips,
# FV_POD_CAP_S below the tier's -> --wall-cap-s, the tag -> --tag); `run` adds
# --yes --wait, `plan` prints fv-lab's plan (no request). A setting fv-lab
# would not honour is refused, not dropped, unless fv-lab does the same thing
# anyway (the tier's value, fv-lab's fixed one): the run would silently differ
# from what was asked. FV_DRY_RUN=1 prints the command only.
fv_lab_route() {
  local what="$1" sha="$2" tag="${3:-}" lab v k e
  refuse() { die "$FAMILY runs through fv-lab (fv-lab-retired.sh), which $*; unset it (docs/dev/fv-lab.md)"; }
  # same <var> <what fv-lab does>: refused when set, non-empty and different.
  same() { [[ -z "${!1:-}" || "${!1}" == "$2" ]] || refuse "uses $1=$2, not ${!1}"; }
  # A restored mode's setting first (fv-lab-retired.sh FV_LAB_RESTORED,
  # ADR-0019 §12): the mode is a tier of its own, named, not routed (the run
  # would land in another run directory than FV_FAMILY's, under another pod
  # name). Checked before any other refusal, so the hint is the first answer.
  local r r_tier r_fam r_mode extra_in=()
  read -r -a extra_in <<<"${FV_EXTRA_ENV:-}"
  for e in ${extra_in[@]+"${extra_in[@]}"}; do
    for r in $FV_LAB_RESTORED; do
      IFS=: read -r r_tier r_fam r_mode <<<"$r"
      [[ "$e" == "$r_mode" ]] || continue
      [[ "$FAMILY" == "$r_tier" ]] \
        && die "FV_FAMILY=$r_tier is the $r_fam family's $r_mode mode already: drop $r_mode from FV_EXTRA_ENV (ADR-0019 §12)"
      die "$r_mode was a mode of runpod-matrix.sh $r_fam; it is the fv-lab tier $r_tier now: use FV_FAMILY=$r_tier (without $r_mode in FV_EXTRA_ENV; ADR-0019 §12, docs/dev/fv-lab.md)"
    done
  done
  local cap cell_s disk cloud gpu tae max_dph
  cap="$(tier_value "$FAMILY" wall_cap_s)"
  cell_s="$(tier_value "$FAMILY" cell_timeout_s)"
  disk="$(tier_value "$FAMILY" container_disk_gb)"
  cloud="$(tier_value "$FAMILY" cloud)"
  gpu="$(tier_value "$FAMILY" gpu)"
  tae="$(tier_value "$FAMILY" tae)"
  max_dph="$(tier_value "$FAMILY" max_usd_per_hr)"
  [[ -n "$cap" && -n "$cell_s" && -n "$disk" && -n "$cloud" && -n "$gpu" && -n "$tae" && -n "$max_dph" ]] \
    || die "$FAMILY: no complete [[tier]] in lab/tiers.toml"
  # No equivalent at all.
  for v in FV_POD_FILES FV_POD_SCRIPT FV_POD_FILE FV_PROMPTS FV_PROMPT FV_SEED FV_AVOID_MACHINES \
    FV_CREATE_WAIT_S FV_FETCH_SKIP FV_FETCH_MAX_MB RUNPOD_ALLOWED_CUDA; do
    [[ -z "${!v:-}" ]] || refuse "has no $v"
  done
  for v in FV_KEEP_POD FV_FETCH_TREE RUNPOD_NO_VOLUME; do
    [[ "${!v:-0}" != 1 ]] || refuse "has no $v=1"
  done
  # The tier's values (lab/tiers.toml).
  same FV_GEN_TIMEOUT_S "$cell_s"
  same FV_CONTAINER_DISK_GB "$disk"
  same RUNPOD_CLOUD_TYPE "$cloud"
  same RUNPOD_GPU_TYPE "$gpu"
  [[ "${FV_SKIP_TAE:-0}" == 1 && "$tae" == true ]] && refuse "fetches the TAE files for this tier (tae = true), FV_SKIP_TAE=1"
  # A higher price cap than the tier's is moot (fv-deploy holds the tier's);
  # a lower one would be exceeded.
  if [[ -n "${RUNPOD_GPU_MAX_DPH:-}" ]] && awk -v a="$RUNPOD_GPU_MAX_DPH" -v b="$max_dph" 'BEGIN { exit !(a + 0 < b + 0) }'; then
    refuse "caps the price at the tier's \$$max_dph/hr, above RUNPOD_GPU_MAX_DPH=$RUNPOD_GPU_MAX_DPH"
  fi
  # fv-lab's fixed values (StartOptions::for_tier, fv-deploy's readiness rule,
  # the EU volume, the image of the commit, its pod names).
  same FV_IDLE_GPU_MIN 10
  same FV_IDLE_GPU_GRACE_MIN 30
  same FV_IDLE_NET_BYTES 262144
  same FV_MIN_DRIVER 580
  same FV_BOOT_WAIT_S 600
  same RUNPOD_VOLUME_ID "$FV_EU_VOLUME_ID"
  same RUNPOD_VOLUME_NAME "$FV_EU_VOLUME_NAME"
  same FV_PLAN_VOLUME "$FV_EU_VOLUME_ID $FV_EU_DC"
  same RUNPOD_IMAGE "ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-$sha"
  same FV_POD_PREFIX fv-lab
  # (FV_EU_VOLUME_NAME needs no check: volumes.sh sets it on both paths.)
  local args=(pod up --sha "$sha" --family "$FAMILY" --root "$ROOT")
  [[ -n "$tag" ]] && args+=(--tag "$tag")
  # The wall cap: lowered with --wall-cap-s, never raised.
  if [[ -n "${FV_POD_CAP_S:-}" && "$FV_POD_CAP_S" != "$cap" ]]; then
    [[ "$FV_POD_CAP_S" =~ ^[1-9][0-9]*$ ]] && (( FV_POD_CAP_S < cap )) \
      || refuse "caps the pod at the tier's ${cap} s (lowered with FV_POD_CAP_S, never raised), not FV_POD_CAP_S=$FV_POD_CAP_S"
    args+=(--wall-cap-s "$FV_POD_CAP_S")
  fi
  # Space- or comma-separated lists, split without globbing.
  local cells=() extra=() cells_list
  cells_list="${FV_CELLS:-}"
  read -r -a cells <<<"${cells_list//,/ }"
  (( ${#cells[@]} )) && args+=(--cells "${cells[@]}")
  [[ -n "${FV_BASELINE_SHA:-}" ]] && args+=(--baseline-sha "$FV_BASELINE_SHA")
  read -r -a extra <<<"${FV_EXTRA_ENV:-}"
  # Every refused entry in one message (a retry must not reveal the next one).
  local keys=() why=()
  for e in ${extra[@]+"${extra[@]}"}; do
    k="${e%%=*}"
    case "$k" in
      FV_NVFP4_WORKLOADS|FV_VAE_WORKLOADS|FV_VAE_UNTRACED)
        keys+=("$k")
        why+=("$k: the restored tiers (precision-nvfp4, ltxvae-gen) have one cell per workload; narrow them by arm with FV_CELLS (ADR-0019 §12)") ;;
      FV_B4A_CELLS)
        keys+=("$k")
        why+=("$k: b4a selects by arm, as every tier: FV_CELLS=\"<cell>-base <cell>-pr\" (fv-lab pod up --cells; ADR-0019 §11.5, §11.6)") ;;
      *)
        if [[ " $(echo $FV_LAB_NOT_READ) " == *" $k "* ]]; then
          keys+=("$k")
          why+=("$k: dropped with the branch, or set by the start command (ADR-0019 §9.4, §10)")
        fi ;;
    esac
  done
  if (( ${#keys[@]} )); then
    refuse "reads no $(printf '%s, ' "${keys[@]}" | sed 's/, $//') from FV_EXTRA_ENV ($(printf '%s; ' "${why[@]}" | sed 's/; $//'))"
  fi
  (( ${#extra[@]} )) && args+=(--extra-env "${extra[@]}")
  [[ "${FV_LPIPS:-0}" == 1 ]] && args+=(--lpips)
  [[ "$what" == run ]] && args+=(--yes --wait)
  if [[ "${FV_DRY_RUN:-0}" == 1 ]]; then
    lab="$(fv_lab_bin)" || lab=fv-lab
    echo "runpod-http: dry run: $FAMILY is fv-lab only (fv-lab-retired.sh); nothing created. The command:" >&2
    printf '%q ' "$lab" "${args[@]}" | sed 's/ $//'
    echo
    return 0
  fi
  lab="$(fv_lab_bin)" || die "$FAMILY runs through fv-lab, and no fv-lab was found (FV_LAB_BIN, PATH, $ROOT/fv-lab). Build it on the build pod: bash scripts/dev/build-pod.sh run . -- cargo build --release --locked -p fastvideo-lab --features cli --bin fv-lab, then bash scripts/dev/build-pod.sh fetch . release/fv-lab ./fv-lab (docs/dev/fv-lab.md)"
  log "$FAMILY is fv-lab only (fv-lab-retired.sh): $(printf '%q ' "$lab" "${args[@]}")"
  exec "$lab" "${args[@]}"
}

cmd_run() {
  local mode="$1" sha="${2:-}" image tag id rc=0 attempt
  sha="${sha:-$(git -C "$ROOT" rev-parse --short=7 origin/main)}"
  # The `gold` tier was never a script family: its pod is fv-lab's only.
  if [[ "$FAMILY" == gold && ( "$mode" == run || "$mode" == all ) ]]; then
    die "gold is no runpod-matrix.sh family (docs/adr/0023-lab-report-and-ci-bridge.md): fv-lab pod up --sha $sha --family gold --yes --wait"
  fi
  if [[ "$mode" == run || "$mode" == all ]] && fv_lab_retired "$FAMILY"; then
    [[ "$mode" == run ]] || die "\`all\` runs runpod-matrix.sh $FAMILY after the kernels, and $FAMILY is fv-lab only (fv-lab-retired.sh): run \`kernels\`, then \`run\`"
    fv_lab_route run "$sha"
    return
  fi
  # FV_DRY_RUN=1: say what would run and create nothing (the commands
  # themselves: FV_DRY_RUN=1 bash scripts/gpu/runpod-matrix.sh <family>).
  if [[ "${FV_DRY_RUN:-0}" == 1 ]]; then
    echo "runpod-http: dry run: $mode $FAMILY at $sha on the script path, nothing created: a pod running runpod-matrix.sh $FAMILY${FV_CELLS:+ (FV_CELLS=$FV_CELLS)}; its body: runpod-http.sh plan $sha; its commands: FV_DRY_RUN=1 bash scripts/gpu/runpod-matrix.sh $FAMILY" >&2
    return 0
  fi
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
  # FV_POD_FILE: "<pod id> <run tag>" for a caller driving several pods (oracle.sh).
  [[ -n "${FV_POD_FILE:-}" ]] && echo "$id $tag" >"$FV_POD_FILE"
  wait_done "$id" "$tag" || rc=1
  fetch_results "$id" "$tag"
  if grep -q "driver_too_old" "$OUT_ROOT/$tag/box.txt" 2>/dev/null; then
    log "host driver too old ($(grep driver_too_old "$OUT_ROOT/$tag/box.txt")); rerun (RUNPOD_ALLOWED_CUDA=13.0 avoids old hosts but also hides the 13.1 / 13.2 ones)"
    rc=1
  fi
  if grep -q "matrix_exit=[1-9]" "$OUT_ROOT/$tag/matrix.out" 2>/dev/null; then
    log "matrix exited abnormally: $(grep matrix_exit "$OUT_ROOT/$tag/matrix.out")"
    rc=1
  fi
  # FV_KEEP_POD=1: the caller deletes the pod (it still serves results to
  # another pod); the FV_POD_CAP_S backstop deletes it regardless.
  if [[ "${FV_KEEP_POD:-0}" == 1 ]]; then
    log "keeping pod $id (FV_KEEP_POD=1)"
    return $rc
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
  plan)
    # The body `run` would send for <sha> and <tag>; nothing is created.
    sha="${2:-$(git -C "$ROOT" rev-parse --short=7 origin/main)}"
    if fv_lab_retired "$FAMILY"; then
      fv_lab_route plan "$sha" "${3:-}"
      exit 0
    fi
    tag="${3:-$sha-$(date -u +%m%d%H%M)}"
    vol="" dc=""
    if [[ "${RUNPOD_NO_VOLUME:-0}" != 1 ]]; then
      if [[ -n "${FV_PLAN_VOLUME:-}" ]]; then read -r vol dc <<<"$FV_PLAN_VOLUME"; else read -r vol dc < <(volume) || true; fi
      [[ -n "${vol:-}" ]] || die "no network volume ($VOL_NAME)"
    fi
    pod_payload "${RUNPOD_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-$sha}" "$tag" run "$vol" "$dc" ;;
  status) proxy "${2:?pod}" "$FAMILY/" ;;
  down) rest DELETE "/pods/${2:?pod}" >/dev/null && echo "deleted ${2}" ;;
  *) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2 ;;
esac
