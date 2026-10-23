#!/usr/bin/env bash
# FastH3 V2 / 4-step NVFP4 apples-to-apples on ONE Runpod RTX PRO 6000 (docs/ports/fasth3-nvfp4.md): upstream
# FastVideo's RTX PRO 6000 recipe and ours on the same pod, by the pod script scripts/gpu/experiments/fasth3-nvfp4-pod.sh.
#
#   fasth3-nvfp4-validate.sh up IMAGE BIN   create the pod (+ a detached wall-clock backstop that deletes it). IMAGE
#                                           is ghcr.io/zaitrarrio/fastvideo-rs-upstream-fastvideo-nvfp4:sha-<7>
#                                           (.github/workflows/upstream-images.yml, target fastvideo-nvfp4); BIN this
#                                           branch's release fv-gpucheck (build pod: build-pod.sh fetch)
#   fasth3-nvfp4-validate.sh status <pod>   status.tsv, the log tail, DONE
#   fasth3-nvfp4-validate.sh fetch <pod> DIR  results.tgz into DIR
#   fasth3-nvfp4-validate.sh down <pod>     delete the pod, check it is gone, drop the staged bundle by its key
#
# The bundle (BIN, scripts/gpu and the R2 plan: presigned GETs from the read-only key, 4 h; ~200 kB, too large for
# the pod's env) is staged in fv-weights-staging under tmp/fasth3-nvfp4-bundle-<sha256>.tgz (presigned GET, checked
# by SHA-256 on the pod) and removed by `down` (or a failed create) by that exact key. Weights come from R2 only (no
# volume: the EU volume holds H3 + LTX for serving, and these trees are R2-only), so any data centre with stock
# works; no CUDA-version filter (2026-10-14: the hosts with stock ran CUDA 13.2 drivers). FV_DISK_GB (default 420).
#
# Env: RUNPOD_API_KEY; FV_POD_CAP_S (backstop, default 6000); FV_MIN_BALANCE (default 12: refuse unless the balance
# is 3 above it); FV_MAX_DPH (default 2.70); FV_SKIP, FV_ONLY_V2 (passed to the pod script); FV_NO_DENSE=1 (no
# dense-control transformer); FV_DCS (space-separated data centres to allow; default: any with stock).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="$(cd "$HERE/.." && pwd)"   # scripts/gpu: the shared tooling these drivers use
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
CAP_S="${FV_POD_CAP_S:-6000}"
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
    image="${2:?upstream fastvideo-nvfp4 image}"
    bin="${3:?the release fv-gpucheck built from this branch}"
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
    allow=" ${FV_DCS:-EU-RO-1 CA-MTL-1 CA-MTL-2 CA-MTL-3 EU-SE-1 EU-CZ-1 EU-NL-1 EUR-IS-1 EUR-IS-2 EUR-IS-3 US-IL-1 US-TX-1 US-TX-3 US-TX-4 US-KS-2 US-GA-2 US-WA-1 US-CA-2 US-NC-1 US-DE-1 OC-AU-1 AP-JP-1} "
    mapfile -t dcs < <(for d in "${dcs[@]}"; do [[ "$allow" == *" $d "* ]] && echo "$d"; done)
    (( ${#gpus[@]} && ${#dcs[@]} )) || { log "no RTX PRO 6000 in stock at or under \$$MAX_DPH/hr"; exit 1; }
    log "data centres: ${dcs[*]}"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    ro=/root/.config/fv
    FV_WEIGHTS_R2_ACCESS_KEY_ID="$(cat "$ro/r2_weights_ro_access_key_id")" FV_WEIGHTS_R2_SECRET_ACCESS_KEY="$(cat "$ro/r2_weights_ro_secret_access_key")" \
      python3 "$GPU/r2-weights.py" plan fasth3-v2-nvfp4 fasth3-4step-vsa-nvfp4 h3-8step --ttl 14400 --out "$tmp/plan.json"
    flags=()
    [[ "${FV_NO_DENSE:-0}" == 1 ]] && flags+=(--no-dense)
    [[ "${FV_ONLY_V2:-0}" == 1 ]] && flags+=(--only-v2)
    python3 "$HERE/fasth3-nvfp4-plan.py" "$tmp/plan.json" "$tmp/plan.b64" "${flags[@]}"
    rm -f "$tmp/plan.json"
    # The bundle: this branch's fv-gpucheck and scripts/gpu (the image's scripts are main's).
    mkdir -p "$tmp/b"
    cp "$bin" "$tmp/b/fv-gpucheck" && chmod 755 "$tmp/b/fv-gpucheck"
    cp -r "$GPU" "$tmp/b/gpu"
    # The plan rides in the bundle (private bucket, SHA-256 checked, removed by `down`): at ~200 kB it is too large
    # for the pod's env (Runpod answered 500).
    base64 -d <"$tmp/plan.b64" | gunzip >"$tmp/b/plan.json" && chmod 600 "$tmp/b/plan.json"
    tar -czf "$tmp/bundle.tgz" -C "$tmp/b" fv-gpucheck gpu plan.json
    rm -rf "$tmp/b"
    sha="$(sha256sum "$tmp/bundle.tgz" | cut -d' ' -f1)"
    url="$(python3 "$GPU/r2-weights.py" --bucket fv-weights-staging put-temp "$tmp/bundle.tgz" --key "tmp/fasth3-nvfp4-bundle-$sha.tgz" --ttl "$CAP_S")"
    rm -f "$tmp/bundle.tgz"
    size="$(du -h "$bin" | cut -f1)"
    log "bundle ${sha:0:12} staged, binary $size"
    # The pod deletes itself 10 min after DONE (Runpod's pod-scoped RUNPOD_API_KEY / RUNPOD_POD_ID); the backstop
    # deletes it regardless.
    start='mkdir -p /opt/fv /srv && curl -fsSL --retry 5 -o /opt/fv/bundle.tgz "$FV_BUNDLE_URL" && echo "$FV_BUNDLE_SHA256  /opt/fv/bundle.tgz" | sha256sum -c - >>/srv/log.txt 2>&1 \
&& tar -xzf /opt/fv/bundle.tgz -C /opt/fv && rm -f /opt/fv/bundle.tgz && chmod 600 /opt/fv/plan.json \
&& (cd /srv && /opt/upstream/tools/bin/python -m http.server 8000 >/dev/null 2>&1 &) && bash /opt/fv/gpu/experiments/fasth3-nvfp4-pod.sh; [ -f /srv/DONE ] || echo 1 >/srv/DONE; \
sleep 600; [ -n "${RUNPOD_POD_ID:-}" ] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"; sleep infinity'
    gpus_json="$(printf '%s\n' "${gpus[@]}" | jq -R . | jq -s .)"
    dcs_json="$(printf '%s\n' "${dcs[@]}" | jq -R . | jq -s .)"
    export FVX_URL="$url"
    payload="$(jq -n --arg name "fv-fasth3-nvfp4-$(date -u +%m%d%H%M)" --arg image "$image" --argjson dcs "$dcs_json" \
      --arg start "$start" --argjson gpus "$gpus_json" --arg sha "$sha" \
      --arg skip "${FV_SKIP:-}" --arg only "${FV_ONLY_V2:-0}" --arg disk "${FV_DISK_GB:-420}" '{
        name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU", gpuCount: 1, gpuTypeIds: $gpus,
        gpuTypePriority: "custom", dataCenterIds: $dcs, dataCenterPriority: "availability", ports: ["8000/http"],
        dockerStartCmd: ["/bin/bash", "-lc", $start], containerDiskInGb: ($disk | tonumber),
        env: ({FV_BUNDLE_URL: env.FVX_URL, FV_BUNDLE_SHA256: $sha, FV_ONLY_V2: $only}
              + (if $skip != "" then {FV_SKIP: $skip} else {} end))
      }')"
    unset FVX_URL
    unstage() { python3 "$GPU/r2-weights.py" --bucket fv-weights-staging rm-temp --prefix "tmp/fasth3-nvfp4-bundle-$sha" >&2 || true; }
    resp="$(rest POST /pods <<<"$payload")" || { log "pod create failed: $(head -c 600 <<<"$resp")"; unstage; exit 1; }
    id="$(jq -r '.id // empty' <<<"$resp")"
    [[ -n "$id" ]] || { log "pod create failed: $(head -c 400 <<<"$resp")"; unstage; exit 1; }
    # A distinctive marker in the backstop's command line, so no generic pkill pattern matches it.
    setsid nohup bash -c ": fv-fasth3-nvfp4-backstop-$id; sleep $CAP_S; curl -sS -X DELETE -H @<(printf 'Authorization: Bearer %s\\n' \"\$RUNPOD_API_KEY\") '$API/pods/$id' >/dev/null 2>&1" >/dev/null 2>&1 </dev/null &
    log "pod $id $(jq -r '.machine.gpuDisplayName // .gpuTypeId // "?"' <<<"$resp") \$$(jq -r '.costPerHr' <<<"$resp")/hr in $(jq -r '.machine.dataCenterId // "?"' <<<"$resp"); balance \$$bal; backstop pid $! deletes it after ${CAP_S}s; https://$id-8000.proxy.runpod.net/"
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
    curl -sS --max-time 900 --fail "$u/results.tgz" -o "$out/results.tgz" && tar -xzf "$out/results.tgz" -C "$out" && rm -f "$out/results.tgz"
    log "fetched into $out"
    ;;
  down)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    id="${2:?pod id}"
    # This pod's own staged bundle, read from its env before it goes: other runs' bundles share the tmp/ prefix.
    sha="$(rest GET "/pods/$id" 2>/dev/null | jq -r '.env.FV_BUNDLE_SHA256 // empty' 2>/dev/null || true)"
    rest DELETE "/pods/$id" >/dev/null 2>&1 || true
    if rest GET "/pods/$id" >/dev/null 2>&1; then log "WARNING: pod $id still listed"; exit 1; fi
    log "pod $id deleted (GET /pods/$id: not found)"
    if [[ -n "$sha" ]]; then
      python3 "$GPU/r2-weights.py" --bucket fv-weights-staging rm-temp --prefix "tmp/fasth3-nvfp4-bundle-$sha" || true
    else
      log "bundle sha256 not readable from pod $id: remove tmp/fasth3-nvfp4-bundle-<sha>.tgz by its full key"
    fi
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
