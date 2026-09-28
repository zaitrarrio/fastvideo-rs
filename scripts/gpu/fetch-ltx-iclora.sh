#!/usr/bin/env bash
# Fetch an LTX-2.5 IC-LoRA onto both weight volumes, add-only, with CPU pods
# driven through the Runpod REST API and the pods' HTTPS proxy (no SSH).
#
#   fetch-ltx-iclora.sh probe     per volume: drop this script's stale partial
#                                 folders, report Hub access (HTTP status per file)
#   fetch-ltx-iclora.sh           1. a CPU pod on the US volume (fv-weights-b200-us)
#                                    pulls the pinned Hub revision into
#                                    weights/<dest>.partial-*, checks SHA-256 against
#                                    the Hub's LFS oid, renames it to weights/<dest>
#                                    and serves it read-only;
#                                 2. a CPU pod on the EU volume (fv-weights-h3-ltx-hy)
#                                    pulls it from the US pod, checks SHA-256, renames;
#                                 3. both pods are deleted (and checked gone).
#
# Default: Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients @ 12040e4 into
# weights/ltx25-ic-lora-ingredients (1.31 GB; LTX-2.x Community License).
# An existing destination stops the run on that volume (add-only). The HF
# token comes from HF_TOKEN or the volume's /workspace/hf/token. FV_POD_CAP_S
# (default 3600) is a hard cap: a detached backstop deletes each pod.
# Pods are named ${FV_POD_PREFIX:-fv-ltxi}-fetch-*. FV_CPU_FLAVORS (default
# "cpu3c cpu5c cpu3g") and FV_CPU_VCPUS (default 4) pick the CPU pod shape
# when a datacenter has no stock for the default.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
CAP_S="${FV_POD_CAP_S:-3600}"
PREFIX="${FV_POD_PREFIX:-fv-ltxi}"
FLAVORS="${FV_CPU_FLAVORS:-cpu3c cpu5c cpu3g}"
VCPUS="${FV_CPU_VCPUS:-4}"
OUT="${FETCH_OUT:-$ROOT/artifacts/runpod/fetch-ltx-iclora}"
IC_REPO="${IC_REPO:-Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients}"
IC_REV="${IC_REV:-12040e4091ac2008d3906a594e31a7fb1ab9d546}"
IC_DEST="${IC_DEST:-/workspace/weights/ltx25-ic-lora-ingredients}"
default_files='{"ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors": ["ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95", 1308787472], "README.md": ["", 25643]}'
IC_FILES="${IC_FILES:-$default_files}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
mkdir -p "$OUT"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$OUT/driver.log" >&2; }
rest() {
  local method="$1" path="$2" body="${3:-}"
  curl -sS --fail-with-body -X "$method" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${body:+-d "$body"} "$API$path"
}
vol() { rest GET /networkvolumes | jq -r --arg n "$1" '.[] | select(.name==$n) | "\(.id) \(.dataCenterId)"' | head -1; }

py_b64="$(base64 -w0 "$HERE/fetch-ltx-iclora.py")"
pods=()
cleanup() {
  local p
  for p in "${pods[@]}"; do
    rest DELETE "/pods/$p" >/dev/null 2>&1 || true
    if rest GET "/pods/$p" >/dev/null 2>&1; then log "WARN: pod $p still listed"; else log "pod $p deleted"; fi
  done
}
trap cleanup EXIT

# start <role> <volume name> [src url] -> pod id
start() {
  local role="$1" name="$2" src="${3:-}" v id dc payload resp
  v="$(vol "$name")"
  [[ -n "$v" ]] || { log "no volume $name"; exit 1; }
  id="${v% *}"; dc="${v#* }"
  local run='mkdir -p /srv && (python -m http.server 8000 --directory /srv >/dev/null 2>&1 &) && echo "$FETCH_PY" | base64 -d > /opt/fetch.py && python /opt/fetch.py; sleep infinity'
  payload="$(jq -n --arg name "$PREFIX-fetch-$role-$(date -u +%m%d%H%M)" --arg vol "$id" --arg dc "$dc" \
    --arg run "$run" --arg py "$py_b64" --arg role "$role" --arg src "$src" \
    --argjson fl "$(jq -nc --arg f "$FLAVORS" '$f|split(" ")')" --argjson vc "$VCPUS" \
    --arg repo "$IC_REPO" --arg rev "$IC_REV" --arg dest "$IC_DEST" --arg files "$IC_FILES" '{
      name: $name, imageName: "python:3.12-slim", cloudType: "SECURE", computeType: "CPU",
      cpuFlavorIds: $fl, cpuFlavorPriority: "availability", vcpuCount: $vc,
      containerDiskInGb: 10, networkVolumeId: $vol, volumeMountPath: "/workspace",
      dataCenterIds: [$dc], ports: ["8000/http"],
      dockerStartCmd: ["/bin/bash","-lc",$run],
      env: {FETCH_PY: $py, ROLE: $role, SRC_URL: $src, IC_REPO: $repo, IC_REV: $rev, IC_DEST: $dest, IC_FILES: $files}
    }')"
  resp="$(rest POST /pods "$payload")"
  local pid
  pid="$(jq -r '.id // empty' <<<"$resp")"
  [[ -n "$pid" ]] || { log "pod create failed: $resp"; exit 1; }
  log "pod $pid ($role on $name, dc $dc) \$$(jq -r '.costPerHr' <<<"$resp")/hr"
  ( sleep "$CAP_S"; rest DELETE "/pods/$pid" >/dev/null 2>&1 || true ) >/dev/null 2>&1 &
  disown || true
  echo "$pid"
}

# wait_done <pod> -> prints the DONE json
wait_done() {
  local p="$1" t0 j
  t0=$(date +%s)
  while :; do
    if j="$(curl -sS --max-time 20 --fail "https://$p-8000.proxy.runpod.net/DONE" 2>/dev/null)"; then
      curl -sS --max-time 20 "https://$p-8000.proxy.runpod.net/log.txt" >"$OUT/log-$p.txt" 2>/dev/null || true
      echo "$j"
      return 0
    fi
    (( $(date +%s) - t0 < CAP_S )) || { log "pod $p: no DONE within ${CAP_S}s"; return 1; }
    # A pod Runpod exits at start (no CPU stock in the DC) never answers.
    if [[ "$(rest GET "/pods/$p" 2>/dev/null | jq -r '.desiredStatus // empty')" == EXITED ]]; then
      log "pod $p exited without DONE"; echo '{"ok": false, "error": "pod exited"}'; return 1
    fi
    sleep 20
  done
}

if [[ "${1:-}" == probe ]]; then
  # Remove this script's stale partial folders and report Hub access (no fetch).
  for name in fv-weights-b200-us fv-weights-h3-ltx-hy; do
    p="$(start probe "$name")"; pods+=("$p")
    log "$name: $(wait_done "$p")"
  done
  exit 0
fi
us="$(start hub fv-weights-b200-us)"; pods+=("$us")
jus="$(wait_done "$us")"; echo "$jus" >"$OUT/us.json"; log "US: $jus"
[[ "$(jq -r .ok <<<"$jus")" == true ]] || exit 1
eu="$(start copy fv-weights-h3-ltx-hy "https://$us-8000.proxy.runpod.net")"; pods+=("$eu")
jeu="$(wait_done "$eu")"; echo "$jeu" >"$OUT/eu.json"; log "EU: $jeu"
[[ "$(jq -r .ok <<<"$jeu")" == true ]] || exit 1
[[ "$(jq -c .sha256 <<<"$jus")" == "$(jq -c .sha256 <<<"$jeu")" ]] || { log "US/EU sha256 differ"; exit 1; }
log "ok: both volumes carry $IC_DEST ($(jq -c .bytes <<<"$jus"))"
