#!/usr/bin/env bash
# Family-pool GPU validation on ONE Runpod RTX PRO 6000 (docs/control/README.md §4 "Family templates", owner decision
# 2026-10-18): one fv-serve process with the shipped H3 family config (configs/serve/runpod-family-h3.toml) serves
# h3-turbo, then h3-max (swap), h3-max again, an NVFP4 release (swap across technique profiles) and h3-turbo again,
# with the timings of each swap; the pod script is scripts/gpu/family-pool-pod.sh.
#
#   family-pool-validate.sh up BIN          create the pod (+ a detached wall-clock backstop that deletes it); BIN is
#                                           the release fv-serve to run (--features cuda,http-client, built from this
#                                           tree on the build pod)
#   family-pool-validate.sh status <pod>    status.tsv, jobs.tsv, the log tail, DONE
#   family-pool-validate.sh fetch <pod> DIR the results (logs, job JSON, clips) into DIR
#   family-pool-validate.sh down <pod>      delete the pod, check it is gone, drop the staged bundle
#
# The pod: the h3-max serve image (FV_SERVE_IMAGE; it carries python3, curl and the CUDA libraries fv-serve links),
# RTX PRO 6000 Server or Workstation Edition, secure cloud, any data centre with stock; the trees come from the R2
# weights store onto the container disk. The R2 plan (presigned GETs, valid 3 h) and the bundle reach the pod through
# the staging bucket (a presigned URL, checked by SHA-256); no key goes to the pod.
#
# Env: RUNPOD_API_KEY; FV_POD_CAP_S (backstop, default 4200); FV_MIN_BALANCE (default 10: refuse unless the balance
# is 3 above it); FV_MAX_DPH (default 2.70); FV_PROMPT (passed to the pod script).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source-path=SCRIPTDIR source=runpod-cuda.sh
source "$HERE/runpod-cuda.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
CAP_S="${FV_POD_CAP_S:-4200}"
MIN_BALANCE="${FV_MIN_BALANCE:-10}"
MAX_DPH="${FV_MAX_DPH:-2.70}"
IMAGE="${FV_SERVE_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-serve:h3-max-latest}"
TREES=(h3-base FastH3-4-step-Preview-v1-LoRA fasth3-v2-nvfp4 auxiliary/tae)
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
    bin="${2:?release fv-serve}"
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    bal="$(balance)"
    awk -v b="$bal" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+3)}' || { log "balance \$$bal too close to the \$$MIN_BALANCE floor"; exit 1; }
    stock="$(gql <<<'{"query":"{ dataCenters { id gpuAvailability { gpuTypeId stockStatus } } }"}')"
    gpus=()
    for g in "NVIDIA RTX PRO 6000 Blackwell Server Edition" "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"; do
      p="$(jq -n --arg g "$g" '{query: "query($g: String) { gpuTypes(input: {id: $g}) { securePrice } }", variables: {g: $g}}' | gql | jq -r '.data.gpuTypes[0].securePrice // empty')"
      awk -v p="${p:-99}" -v m="$MAX_DPH" 'BEGIN{exit !(p+0 <= m+0)}' || { log "$g: \$${p:-?}/hr over \$$MAX_DPH"; continue; }
      gpus+=("$g")
    done
    gj="$(printf '%s\n' "${gpus[@]}" | jq -R . | jq -s .)"
    with_stock='.data.dataCenters[] | select([.gpuAvailability[]? | select((.gpuTypeId | IN($g[])) and (.stockStatus // "None") != "None")] | length > 0) | .id'
    mapfile -t dcs < <(jq -r --argjson g "$gj" "$with_stock" <<<"$stock")
    # The REST API's dataCenterIds enum is narrower than GraphQL's list.
    rest_dcs=" EU-RO-1 CA-MTL-1 CA-MTL-2 CA-MTL-3 EU-SE-1 EU-CZ-1 EU-NL-1 EUR-IS-1 EUR-IS-2 EUR-IS-3 US-IL-1 US-TX-1 US-TX-3 US-TX-4 US-KS-2 US-GA-2 US-WA-1 US-CA-2 US-NC-1 US-DE-1 OC-AU-1 AP-JP-1 "
    mapfile -t dcs < <(for d in "${dcs[@]}"; do [[ "$rest_dcs" == *" $d "* ]] && echo "$d"; done)
    (( ${#gpus[@]} && ${#dcs[@]} )) || { log "no RTX PRO 6000 in stock at or under \$$MAX_DPH/hr"; exit 1; }
    log "data centres: ${dcs[*]}"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    mkdir -p "$tmp/b"
    python3 "$HERE/r2-weights.py" plan "${TREES[@]}" --ttl 10800 --out "$tmp/b/plan.json"
    cp "$bin" "$tmp/b/fv-serve"
    chmod +x "$tmp/b/fv-serve"
    cp "$HERE/r2-fetch.py" "$HERE/family-pool-pod.sh" "$ROOT/configs/serve/runpod-family-h3.toml" "$tmp/b/"
    tar czf "$tmp/bundle.tgz" -C "$tmp/b" .
    sha="$(sha256sum "$tmp/bundle.tgz" | cut -d' ' -f1)"
    url="$(python3 "$HERE/r2-weights.py" --bucket fv-weights-staging put-temp "$tmp/bundle.tgz" --key "tmp/family-pool-bundle-$sha.tgz" --ttl 10800)"
    log "balance \$$bal; bundle ${sha:0:12} ($(du -h "$tmp/bundle.tgz" | cut -f1)) staged"
    # The pod deletes itself 20 min after DONE (Runpod injects a pod-scoped RUNPOD_API_KEY / RUNPOD_POD_ID); the local
    # backstop deletes it regardless.
    start='mkdir -p /opt/fv /srv && cd /opt/fv && curl -fsSL --retry 5 -o b.tgz "$FV_BUNDLE_URL" && echo "$FV_BUNDLE_SHA256  b.tgz" | sha256sum -c - >/srv/log.txt 2>&1 && tar xzf b.tgz \
&& (cd /srv && python3 -m http.server 8000 >/dev/null 2>&1 &) && bash /opt/fv/family-pool-pod.sh; [ -f /srv/DONE ] || echo 1 >/srv/DONE; \
sleep 1200; if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then until fv_self_delete /tmp/self-delete.log; do sleep 60; done; fi; sleep infinity'
    # fv_self_delete: GraphQL podTerminate with the pod-scoped key, retried each minute (ADR-0023 §4).
    start="$(<"$HERE/lib/self-delete.sh")"$'\n'"$start"
    export FVX_URL="$url"
    gpus_json="$(printf '%s\n' "${gpus[@]}" | jq -R . | jq -s .)"
    dcs_json="$(printf '%s\n' "${dcs[@]}" | jq -R . | jq -s .)"
    payload="$(jq -n --arg cuda "${RUNPOD_ALLOWED_CUDA-$(fv_runpod_cuda)}" --arg name "fv-familypool-$(date -u +%m%d%H%M)" --arg image "$IMAGE" --argjson dcs "$dcs_json" \
      --arg start "$start" --arg sha "$sha" --argjson gpus "$gpus_json" --arg prompt "${FV_PROMPT:-}" '{
        name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU", gpuCount: 1, gpuTypeIds: $gpus,
        gpuTypePriority: "custom", dataCenterIds: $dcs, ports: ["8000/http"],
        dockerEntrypoint: ["bash", "-c", $start], dockerStartCmd: [], containerDiskInGb: 260,
        env: ({FV_BUNDLE_URL: env.FVX_URL, FV_BUNDLE_SHA256: $sha} + (if $prompt != "" then {FV_PROMPT: $prompt} else {} end))
      } + (if $cuda == "" then {} else {allowedCudaVersions: ($cuda | split(" "))} end)')"
    unset FVX_URL
    resp="$(rest POST /pods <<<"$payload")" || { log "pod create failed: $(head -c 600 <<<"$resp")"; exit 1; }
    id="$(jq -r '.id // empty' <<<"$resp")"
    [[ -n "$id" ]] || { log "pod create failed: $(head -c 400 <<<"$resp")"; exit 1; }
    setsid nohup bash -c "sleep $CAP_S; curl -sS -X DELETE -H @<(printf 'Authorization: Bearer %s\\n' \"\$RUNPOD_API_KEY\") '$API/pods/$id' >/dev/null 2>&1" >/dev/null 2>&1 </dev/null &
    log "pod $id $(jq -r '.machine.gpuDisplayName // .gpuTypeId // "?"' <<<"$resp") \$$(jq -r '.costPerHr' <<<"$resp")/hr; backstop pid $! deletes it after ${CAP_S}s; https://$id-8000.proxy.runpod.net/"
    echo "$id"
    ;;
  status)
    id="${2:?pod id}"; u="https://$id-8000.proxy.runpod.net"
    curl -sS --max-time 20 --fail "$u/status.tsv" 2>/dev/null || true
    curl -sS --max-time 20 --fail "$u/jobs.tsv" 2>/dev/null || true
    curl -sS --max-time 20 --fail "$u/log.txt" 2>/dev/null | tail -n "${FV_TAIL:-6}" || true
    printf 'DONE=%s\n' "$(curl -sS --max-time 20 --fail "$u/DONE" 2>/dev/null || true)"
    ;;
  fetch)
    id="${2:?pod id}"; out="${3:?dir}"; u="https://$id-8000.proxy.runpod.net"
    mkdir -p "$out"
    for f in $(curl -sS --max-time 30 --fail "$u/" 2>/dev/null | grep -o 'href="[^"/]*"' | cut -d'"' -f2 | grep -E '\.(log|txt|json|tsv|csv|mp4)$'); do
      curl -sS --max-time 300 --fail "$u/$f" -o "$out/$f" 2>/dev/null || rm -f "$out/$f"
    done
    log "fetched into $out"
    ;;
  down)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    id="${2:?pod id}"
    sha="$(rest GET "/pods/$id" 2>/dev/null | jq -r '.env.FV_BUNDLE_SHA256 // ""' 2>/dev/null || true)"
    rest DELETE "/pods/$id" >/dev/null 2>&1 || true
    if rest GET "/pods/$id" >/dev/null 2>&1; then log "WARNING: pod $id still listed"; exit 1; fi
    log "pod $id deleted (GET /pods/$id: not found)"
    if [[ -n "${sha:-}" ]]; then
      python3 "$HERE/r2-weights.py" --bucket fv-weights-staging rm-temp --prefix "tmp/family-pool-bundle-$sha" || true
    else
      log "bundle sha256 not readable from pod $id: its staged bundle is left under tmp/family-pool-bundle-; remove it by its full key"
    fi
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
