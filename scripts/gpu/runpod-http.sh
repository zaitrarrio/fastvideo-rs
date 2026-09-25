#!/usr/bin/env bash
# SSH-free Runpod driver: one RTX PRO 6000 pod on the CI image for a pushed
# commit, driven entirely through the REST API and the pod's HTTPS proxy.
# For hosts (sandboxes, CI) that can reach https://*.runpod.io but not TCP 22.
#
#   runpod-http.sh run [sha]     create pod, run the rtx6000 family, pull
#                                summaries + logs, destroy the pod
#   runpod-http.sh weights       weight gate only (verify-weights.sh), no cells
#   runpod-http.sh status <pod>  print live.log from a running pod
#   runpod-http.sh down <pod>    destroy a pod
#
# The pod's start command runs everything; results are published read-only on
# port 8000 (python http.server over /workspace/runs) and fetched through
# https://<pod>-8000.proxy.runpod.net. The pod is always deleted at the end,
# and a wall-clock cap (FV_POD_CAP_S, default 4h) deletes it regardless.
#
# Env: RUNPOD_API_KEY, RUNPOD_VOLUME_ID (default: volume named
# fv-weights-h3-ltx-hy), FV_CELLS (subset of cells), FV_GEN_TIMEOUT_S
# (per-cell cap, default 3600), FV_EXTRA_ENV (space-separated K=V for cells).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GPU="${RUNPOD_GPU_TYPE:-NVIDIA RTX PRO 6000 Blackwell Server Edition}"
VOL_NAME="${RUNPOD_VOLUME_NAME:-fv-weights-h3-ltx-hy}"
MAX_DPH="${RUNPOD_GPU_MAX_DPH:-5}"
CAP_S="${FV_POD_CAP_S:-14400}"
OUT_ROOT="$ROOT/artifacts/runpod/rtx6000"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"

die() { echo "runpod-http: $*" >&2; exit 1; }
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
rest() {
  local method="$1" path="$2" body="${3:-}"
  curl -sS --fail-with-body -X "$method" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${body:+-d "$body"} "$API$path"
}
proxy() { curl -sS --max-time 30 --fail "https://$1-8000.proxy.runpod.net/$2"; }

volume() {
  if [[ -n "${RUNPOD_VOLUME_ID:-}" ]]; then
    rest GET "/networkvolumes/$RUNPOD_VOLUME_ID" | jq -r '"\(.id) \(.dataCenterId)"'
  else
    rest GET /networkvolumes | jq -r --arg n "$VOL_NAME" '.[] | select(.name==$n) | "\(.id) \(.dataCenterId)"' | head -1
  fi
}

# $1 = image, $2 = run tag, $3 = mode (run|weights)
start_cmd() {
  local tag="$2" mode="$3"
  local cells="fasth3-8step fasth3-4step-vsa fasth3-4step-dense sol-h3 sol-h3-spark ltx25-two-stage"
  cat <<EOF
set -u
OUT=/workspace/runs/rtx6000/$tag
mkdir -p "\$OUT"
( apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq python3-minimal >/dev/null 2>&1
  cd /workspace/runs && exec python3 -m http.server 8000 ) >"\$OUT/http.log" 2>&1 &
{
  echo "image_build_id=\$(cat /opt/fastvideo-rs/target/release/fv-gpucheck.build-id 2>/dev/null)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} >"\$OUT/box.txt" 2>&1
( cd /workspace/weights && for d in *; do echo "== \$d"; find -L "\$d" -maxdepth 3 \( -name '*.safetensors' -o -name '*.json' \) -printf '%s %p\n' 2>/dev/null | head -60; done ) >"\$OUT/tree.txt" 2>&1
FV_WEIGHTS=/workspace/weights bash /opt/fastvideo-rs/scripts/gpu/verify-weights.sh $cells >"\$OUT/weights.log" 2>&1
echo "exit=\$?" >>"\$OUT/weights.log"
if [ "$mode" = run ]; then
  env ${FV_EXTRA_ENV:-} FV_WORK=/workspace FV_RUN_TAG=$tag FV_CELLS="${FV_CELLS:-}" \
    FV_GEN_TIMEOUT_S=${FV_GEN_TIMEOUT_S:-3600} \
    bash /opt/fastvideo-rs/scripts/gpu/runpod-matrix.sh rtx6000 >"\$OUT/matrix.out" 2>&1
fi
touch "\$OUT/DONE"
exec sleep infinity
EOF
}

create_pod() {
  local image="$1" tag="$2" mode="$3" vol dc payload resp id dph
  read -r vol dc < <(volume) || true
  [[ -n "${vol:-}" ]] || die "no network volume ($VOL_NAME)"
  payload="$(jq -n --arg name "fv-rtx6000-$tag" --arg image "$image" --arg vol "$vol" \
    --arg dc "$dc" --arg gpu "$GPU" --arg cmd "$(start_cmd "$image" "$tag" "$mode")" '{
      name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU",
      gpuTypeIds: [$gpu], gpuCount: 1, containerDiskInGb: 40, volumeInGb: 0,
      networkVolumeId: $vol, volumeMountPath: "/workspace", dataCenterIds: [$dc],
      ports: ["8000/http"], dockerStartCmd: ["/bin/bash", "-c", $cmd]
    }')"
  log "create pod gpu=\"$GPU\" image=$image volume=$vol dc=$dc"
  resp="$(rest POST /pods "$payload")" || die "pod create failed: $resp"
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

fetch_results() {
  local id="$1" tag="$2" out="$OUT_ROOT/$tag" f cell
  mkdir -p "$out"
  for f in box.txt tree.txt weights.log matrix.out live.log; do
    proxy "$id" "rtx6000/$tag/$f" >"$out/$f" 2>/dev/null || true
  done
  for cell in $(proxy "$id" "rtx6000/$tag/" 2>/dev/null | grep -oE 'href="[^"/]+/"' | sed 's/href="//;s/\/"//'); do
    mkdir -p "$out/$cell"
    for f in summary.json stderr.log stdout.log; do
      proxy "$id" "rtx6000/$tag/$cell/$f" >"$out/$cell/$f" 2>/dev/null || rm -f "$out/$cell/$f"
    done
  done
  log "results → $out"
}

wait_done() {
  local id="$1" tag="$2" t0 last=""
  t0=$(date +%s)
  while :; do
    if proxy "$id" "rtx6000/$tag/DONE" >/dev/null 2>&1; then return 0; fi
    local cur
    cur="$(proxy "$id" "rtx6000/$tag/live.log" 2>/dev/null | tail -1 || true)"
    [[ -n "$cur" && "$cur" != "$last" ]] && { log "pod: $cur"; last="$cur"; }
    (( $(date +%s) - t0 < CAP_S )) || { log "cap reached"; return 1; }
    sleep 30
  done
}

cmd_run() {
  local mode="$1" sha="${2:-}" image tag id rc=0
  sha="${sha:-$(git -C "$ROOT" rev-parse --short=7 origin/main)}"
  image="${RUNPOD_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-$sha}"
  tag="$sha-$(date -u +%m%d%H%M)"
  id="$(create_pod "$image" "$tag" "$mode")"
  echo "$id" >"${TMPDIR:-/tmp}/fv-rtx6000.pod"
  wait_done "$id" "$tag" || rc=1
  fetch_results "$id" "$tag"
  log "destroy pod $id"
  rest DELETE "/pods/$id" >/dev/null || log "WARN: delete failed for $id"
  cat "$OUT_ROOT/$tag/weights.log" >&2 || true
  return $rc
}

case "${1:-}" in
  run) shift; cmd_run run "$@" ;;
  weights) shift; cmd_run weights "$@" ;;
  status) proxy "${2:?pod}" "rtx6000/" ;;
  down) rest DELETE "/pods/${2:?pod}" >/dev/null && echo "deleted ${2}" ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
