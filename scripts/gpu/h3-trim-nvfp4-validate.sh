#!/usr/bin/env bash
# FastH3 Trim NVFP4 GPU validation on ONE Runpod RTX PRO 6000 (docs/ports/h3-trim.md §5): the published NVFP4 DiT
# against the bf16 Trim and the served MXFP8 default (per-step latents, PSNR / LPIPS / gate) and the speed and peak
# memory of all three at 480p and 768p, by the pod script scripts/gpu/h3-trim-nvfp4-pod.sh.
#
#   h3-trim-nvfp4-validate.sh up IMAGE   create the pod (+ a detached wall-clock backstop that deletes it). IMAGE is
#                                        a runtime image built from this branch (fv-gpucheck + scripts/gpu), e.g.
#                                        ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-<commit>
#   h3-trim-nvfp4-validate.sh status <pod>    status.tsv, the log tail, DONE
#   h3-trim-nvfp4-validate.sh fetch <pod> DIR results.tgz (logs, reports, benchmark.json, parity clips) into DIR
#   h3-trim-nvfp4-validate.sh down <pod>      delete the pod and check it is gone
#
# Nothing is written to R2 or a volume: the R2 plan (presigned GETs from the read-only key, valid 3 h) reaches the pod
# gzip + base64 in its env (FV_PLAN_GZ_B64), and the scripts and binary are the image's. With stock in EU-RO-1 the
# pod mounts the EU weight volume (h3-base, read only); otherwise (or FV_NO_VOLUME=1) h3-base's tokenizer, audio VAE
# and FP8 text encoder come from R2 too.
#
# Env: RUNPOD_API_KEY; FV_POD_CAP_S (backstop, default 4200); FV_MIN_BALANCE (default 12: refuse unless the balance
# is 3 above it); FV_MAX_DPH (default 2.70); FV_SKIP / FV_CELL_RES (passed to the pod script).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=volumes.sh
source "$HERE/volumes.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
CAP_S="${FV_POD_CAP_S:-4200}"
MIN_BALANCE="${FV_MIN_BALANCE:-12}"
MAX_DPH="${FV_MAX_DPH:-2.70}"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
rest() {
  local method="$1" path="$2"
  if [[ "$method" == POST ]]; then
    curl -sS --fail-with-body -X POST -H @<(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY") -H 'content-type: application/json' --data-binary @- "$API$path"
  else
    curl -sS --fail-with-body -X "$method" -H @<(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY") "$API$path"
  fi
}
gql() { curl -sS -H @<(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY") -H 'content-type: application/json' "$GQL" --data-binary @-; }
balance() { gql <<<'{"query":"{ myself { clientBalance } }"}' | jq -r '.data.myself.clientBalance // empty'; }

cmd="${1:-}"
case "$cmd" in
  up)
    image="${2:?runtime image built from this branch}"
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    bal="$(balance)"
    awk -v b="$bal" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+3)}' || { log "balance \$$bal too close to the \$$MIN_BALANCE floor"; exit 1; }
    stock="$(gql <<<'{"query":"{ dataCenters { id gpuAvailability { gpuTypeId stockStatus } } }"}')"
    gpus=(); dcs=(); mode=volume
    for g in "NVIDIA RTX PRO 6000 Blackwell Server Edition" "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"; do
      p="$(jq -n --arg g "$g" '{query: "query($g: String) { gpuTypes(input: {id: $g}) { securePrice } }", variables: {g: $g}}' | gql | jq -r '.data.gpuTypes[0].securePrice // empty')"
      awk -v p="${p:-99}" -v m="$MAX_DPH" 'BEGIN{exit !(p+0 <= m+0)}' || { log "$g: \$${p:-?}/hr over \$$MAX_DPH"; continue; }
      gpus+=("$g")
    done
    gj="$(printf '%s\n' "${gpus[@]}" | jq -R . | jq -s .)"
    with_stock='.data.dataCenters[] | select([.gpuAvailability[]? | select((.gpuTypeId | IN($g[])) and (.stockStatus // "None") != "None")] | length > 0) | .id'
    mapfile -t dcs < <(jq -r --argjson g "$gj" "$with_stock" <<<"$stock")
    rest_dcs=" EU-RO-1 CA-MTL-1 CA-MTL-2 CA-MTL-3 EU-SE-1 EU-CZ-1 EU-NL-1 EUR-IS-1 EUR-IS-2 EUR-IS-3 US-IL-1 US-TX-1 US-TX-3 US-TX-4 US-KS-2 US-GA-2 US-WA-1 US-CA-2 US-NC-1 US-DE-1 OC-AU-1 AP-JP-1 "
    mapfile -t dcs < <(for d in "${dcs[@]}"; do [[ "$rest_dcs" == *" $d "* ]] && echo "$d"; done)
    if [[ "${FV_NO_VOLUME:-0}" != 1 ]] && printf '%s\n' "${dcs[@]}" | grep -qx "$FV_EU_DC"; then
      dcs=("$FV_EU_DC")
    else
      mode=r2
    fi
    (( ${#gpus[@]} && ${#dcs[@]} )) || { log "no RTX PRO 6000 in stock at or under \$$MAX_DPH/hr"; exit 1; }
    log "mode $mode: ${dcs[*]}"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    # The plan, from the read-only key (it cannot write): the Trim trees, and h3-base's parts in r2 mode.
    ro=/root/.config/fv
    FV_WEIGHTS_R2_ACCESS_KEY_ID="$(cat "$ro/r2_weights_ro_access_key_id")" FV_WEIGHTS_R2_SECRET_ACCESS_KEY="$(cat "$ro/r2_weights_ro_secret_access_key")" \
      python3 "$HERE/r2-weights.py" plan fasth3-trim-8step fasth3-trim-8step-nvfp4 h3-base h3-base/text_encoder_fp8 --ttl 10800 --out "$tmp/plan.json"
    python3 "$HERE/h3-trim-nvfp4-plan.py" "$tmp/plan.json" "$mode" "$tmp/plan.b64"
    # The runtime image has curl and coreutils but no Python: uv brings one (r2-fetch.py, the prompt files). The pod
    # deletes itself 20 min after DONE (Runpod's pod-scoped RUNPOD_API_KEY / RUNPOD_POD_ID); the backstop regardless.
    start='mkdir -p /opt/fv /srv && printf %s "$FV_PLAN_GZ_B64" | base64 -d | gunzip >/opt/fv/plan.json && chmod 600 /opt/fv/plan.json \
&& (curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uvbin UV_NO_MODIFY_PATH=1 sh) >>/srv/log.txt 2>&1 \
&& export PATH=/opt/uvbin:$PATH UV_PYTHON_INSTALL_DIR=/opt/uvpy && uv python install 3.12 >>/srv/log.txt 2>&1 && export PY="$(uv python find 3.12)" \
&& (cd /srv && "$PY" -m http.server 8000 >/dev/null 2>&1 &) && unset FV_PLAN_GZ_B64 && bash /opt/fastvideo-rs/scripts/gpu/h3-trim-nvfp4-pod.sh; [ -f /srv/DONE ] || echo 1 >/srv/DONE; \
sleep 1200; [ -n "${RUNPOD_POD_ID:-}" ] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"; sleep infinity'
    gpus_json="$(printf '%s\n' "${gpus[@]}" | jq -R . | jq -s .)"
    dcs_json="$(printf '%s\n' "${dcs[@]}" | jq -R . | jq -s .)"
    payload="$(jq -n --arg name "fv-h3trim-nvfp4-$(date -u +%m%d%H%M)" --arg image "$image" --arg vol "$FV_EU_VOLUME_ID" \
      --argjson dcs "$dcs_json" --arg mode "$mode" --arg start "$start" --argjson gpus "$gpus_json" \
      --rawfile plan "$tmp/plan.b64" --arg skip "${FV_SKIP:-}" --arg res "${FV_CELL_RES:-}" '{
        name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU", gpuCount: 1, gpuTypeIds: $gpus,
        gpuTypePriority: "custom", dataCenterIds: $dcs, ports: ["8000/http"], dockerStartCmd: ["/bin/bash", "-lc", $start],
        env: ({FV_PLAN_GZ_B64: $plan, FV_WEIGHTS_MODE: $mode}
              + (if $skip != "" then {FV_SKIP: $skip} else {} end) + (if $res != "" then {FV_CELL_RES: $res} else {} end))
      } + (if $mode == "volume" then {containerDiskInGb: 120, networkVolumeId: $vol, volumeMountPath: "/workspace"}
           else {containerDiskInGb: 150, allowedCudaVersions: ["13.0"]} end)')"
    resp="$(rest POST /pods <<<"$payload")" || { log "pod create failed: $(head -c 600 <<<"$resp")"; exit 1; }
    id="$(jq -r '.id // empty' <<<"$resp")"
    [[ -n "$id" ]] || { log "pod create failed: $(head -c 400 <<<"$resp")"; exit 1; }
    setsid nohup bash -c "sleep $CAP_S; curl -sS -X DELETE -H @<(printf 'Authorization: Bearer %s\\n' \"\$RUNPOD_API_KEY\") '$API/pods/$id' >/dev/null 2>&1" >/dev/null 2>&1 </dev/null &
    log "pod $id $(jq -r '.machine.gpuDisplayName // .gpuTypeId // "?"' <<<"$resp") \$$(jq -r '.costPerHr' <<<"$resp")/hr; balance \$$bal; backstop pid $! deletes it after ${CAP_S}s; https://$id-8000.proxy.runpod.net/"
    echo "$id"
    ;;
  status)
    id="${2:?pod id}"; u="https://$id-8000.proxy.runpod.net"
    curl -sS --max-time 20 --fail "$u/status.tsv" 2>/dev/null || true
    curl -sS --max-time 20 --fail "$u/log.txt" 2>/dev/null | tail -n "${FV_TAIL:-6}" || true
    printf 'DONE=%s\n' "$(curl -sS --max-time 20 --fail "$u/DONE" 2>/dev/null || true)"
    ;;
  fetch)
    id="${2:?pod id}"; out="${3:?dir}"; u="https://$id-8000.proxy.runpod.net"
    mkdir -p "$out"
    curl -sS --max-time 600 --fail "$u/results.tgz" -o "$out/results.tgz" && tar -xzf "$out/results.tgz" -C "$out" && rm -f "$out/results.tgz"
    log "fetched into $out"
    ;;
  down)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    id="${2:?pod id}"
    rest DELETE "/pods/$id" >/dev/null 2>&1 || true
    if rest GET "/pods/$id" >/dev/null 2>&1; then log "WARNING: pod $id still listed"; exit 1; fi
    log "pod $id deleted (GET /pods/$id: not found)"
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
