#!/usr/bin/env bash
# MXFP8 decoder linears on the FastH3 Trim 26-block video VAE: ONE Runpod RTX PRO 6000 in EU-RO-1 with the EU weight
# volume (read only), running pod.sh. Adapted from scripts/gpu/h3-trim-validate.sh (bundle staging) and
# scripts/gpu/h3-trim-nvfp4-validate.sh (stock / price / balance checks).
#
#   launch.sh up IMAGE BIN    stage the bundle (BIN = this branch's fv-gpucheck, pod.sh, r2-fetch.py, the prompt
#                             set, the R2 plan of the Trim tree) and create the pod with a detached wall-clock backstop
#   launch.sh status POD      status.tsv, the log tail, DONE
#   launch.sh fetch POD DIR   results.tgz (no frames) into DIR
#   launch.sh down POD        delete the pod, check it is gone, drop the staged bundle
#
# Env: RUNPOD_API_KEY; FV_POD_CAP_S (backstop, default 2100 = $1.45 at $2.49/hr); FV_MIN_BALANCE (default 10);
# FV_MAX_DPH (default 2.70); FV_VERIFY=0 (passed to pod.sh: skip the =0 identity cells).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
G="$(cd "$HERE/../../../../scripts/gpu" && pwd)"
# shellcheck source=/dev/null
source "$G/volumes.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
CAP_S="${FV_POD_CAP_S:-2100}"
MIN_BALANCE="${FV_MIN_BALANCE:-10}"
MAX_DPH="${FV_MAX_DPH:-2.70}"
RO=/root/.config/fv
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
# Staging writes: the Cloudflare API token (r2-weights.py --cf-token-file default), as h3-trim-validate.sh.
rw() { env -u FV_WEIGHTS_R2_ACCESS_KEY_ID -u FV_WEIGHTS_R2_SECRET_ACCESS_KEY "$@"; }
ro() { FV_WEIGHTS_R2_ACCESS_KEY_ID="$(cat "$RO/r2_weights_ro_access_key_id")" FV_WEIGHTS_R2_SECRET_ACCESS_KEY="$(cat "$RO/r2_weights_ro_secret_access_key")" "$@"; }

case "${1:-}" in
  up)
    image="${2:?runtime image}" bin="${3:?fv-gpucheck built from this branch}"
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    bal="$(balance)"
    awk -v b="$bal" -v m="$MIN_BALANCE" -v c="$CAP_S" 'BEGIN{exit !(b - 2.70 * c / 3600 >= m)}' || { log "balance \$$bal: the backstop could take it under \$$MIN_BALANCE"; exit 1; }
    stock="$(gql <<<'{"query":"{ dataCenters { id gpuAvailability { gpuTypeId stockStatus } } }"}')"
    gpus=()
    for g in "NVIDIA RTX PRO 6000 Blackwell Server Edition" "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"; do
      p="$(jq -n --arg g "$g" '{query: "query($g: String) { gpuTypes(input: {id: $g}) { securePrice } }", variables: {g: $g}}' | gql | jq -r '.data.gpuTypes[0].securePrice // empty')"
      awk -v p="${p:-99}" -v m="$MAX_DPH" 'BEGIN{exit !(p+0 <= m+0)}' || { log "$g: \$${p:-?}/hr over \$$MAX_DPH"; continue; }
      in_eu="$(jq -r --arg g "$g" --arg dc "$FV_EU_DC" '[.data.dataCenters[] | select(.id == $dc) | .gpuAvailability[]? | select(.gpuTypeId == $g and (.stockStatus // "None") != "None")] | length' <<<"$stock")"
      [[ "$in_eu" -gt 0 ]] && gpus+=("$g")
    done
    (( ${#gpus[@]} )) || { log "no RTX PRO 6000 in stock in $FV_EU_DC at or under \$$MAX_DPH/hr"; exit 1; }
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    mkdir -p "$tmp/b"
    # The Trim tree without the ComfyUI INT8 VAE file (not read by the port).
    ro python3 "$G/r2-weights.py" plan fasth3-trim-8step --ttl 10800 --out "$tmp/plan-all.json"
    python3 - "$tmp/plan-all.json" "$tmp/b/plan.json" <<'PYEOF'
import json, os, sys
plan = json.load(open(sys.argv[1]))
for t in plan["trees"]:
    t["files"] = [f for f in t["files"] if "int8" not in f["path"]]
    t["total_bytes"] = sum(f["size"] for f in t["files"])
fd = os.open(sys.argv[2], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    json.dump(plan, f)
print(" ".join(f"{t['dest']}:{len(t['files'])}f/{t['total_bytes'] / 1e9:.1f}GB" for t in plan["trees"]), file=sys.stderr)
PYEOF
    cp "$bin" "$tmp/b/fv-gpucheck"
    cp "$HERE/pod.sh" "$G/r2-fetch.py" "$tmp/b/"
    cp "$HERE/../../h3-vae-decode-speed/driver/prompts-5x3.json" "$tmp/b/"
    tar czf "$tmp/bundle.tgz" -C "$tmp/b" .
    sha="$(sha256sum "$tmp/bundle.tgz" | cut -d' ' -f1)"
    url="$(rw python3 "$G/r2-weights.py" --bucket fv-weights-staging put-temp "$tmp/bundle.tgz" --key "tmp/h3trim-vaemx-bundle-$sha.tgz" --ttl 10800)"
    log "balance \$$bal; bundle ${sha:0:12} ($(du -h "$tmp/bundle.tgz" | cut -f1)) staged"
    # The pod deletes itself 5 min after DONE (Runpod's pod-scoped key); the local backstop regardless.
    start='mkdir -p /opt/fv /srv && cd /opt/fv && curl -fsSL --retry 5 -o b.tgz "$FV_BUNDLE_URL" && echo "$FV_BUNDLE_SHA256  b.tgz" | sha256sum -c - >/srv/log.txt 2>&1 && tar xzf b.tgz && chmod +x fv-gpucheck && unset FV_BUNDLE_URL \
&& (curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uvbin UV_NO_MODIFY_PATH=1 sh) >>/srv/log.txt 2>&1 \
&& export PATH=/opt/uvbin:$PATH UV_PYTHON_INSTALL_DIR=/opt/uvpy && uv python install 3.12 >>/srv/log.txt 2>&1 && export PY="$(uv python find 3.12)" \
&& (cd /srv && "$PY" -m http.server 8000 >/dev/null 2>&1 &) && bash /opt/fv/pod.sh; [ -f /srv/DONE ] || echo 1 >/srv/DONE; \
sleep 300; [ -n "${RUNPOD_POD_ID:-}" ] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"; sleep infinity'
    export FVX_URL="$url"
    gpus_json="$(printf '%s\n' "${gpus[@]}" | jq -R . | jq -s .)"
    payload="$(jq -n --arg name "fv-h3trim-vaemx-$(date -u +%m%d%H%M)" --arg image "$image" --arg vol "$FV_EU_VOLUME_ID" \
      --arg dc "$FV_EU_DC" --arg start "$start" --arg sha "$sha" --arg verify "${FV_VERIFY:-}" --argjson gpus "$gpus_json" '{
        name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU", gpuCount: 1, gpuTypeIds: $gpus,
        gpuTypePriority: "custom", dataCenterIds: [$dc], ports: ["8000/http"], dockerStartCmd: ["/bin/bash", "-lc", $start],
        env: ({FV_BUNDLE_URL: env.FVX_URL, FV_BUNDLE_SHA256: $sha} + (if $verify != "" then {FV_VERIFY: $verify} else {} end)),
        containerDiskInGb: 120, networkVolumeId: $vol, volumeMountPath: "/workspace"}')"
    unset FVX_URL
    resp="$(rest POST /pods <<<"$payload")" || { log "pod create failed: $(head -c 600 <<<"$resp")"; exit 1; }
    id="$(jq -r '.id // empty' <<<"$resp")"
    [[ -n "$id" ]] || { log "pod create failed: $(head -c 400 <<<"$resp")"; exit 1; }
    # A distinctive marker in the backstop's command line, so no generic pattern matches it.
    setsid nohup bash -c ": fv-h3trim-vaemx-backstop-$id; sleep $CAP_S; curl -sS -X DELETE -H @<(printf 'Authorization: Bearer %s\\n' \"\$RUNPOD_API_KEY\") '$API/pods/$id' >/dev/null 2>&1" >/dev/null 2>&1 </dev/null &
    log "pod $id $(jq -r '.machine.gpuDisplayName // .gpuTypeId // "?"' <<<"$resp") \$$(jq -r '.costPerHr' <<<"$resp")/hr; backstop pid $! (marker fv-h3trim-vaemx-backstop-$id) deletes it after ${CAP_S}s"
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
    curl -sS --max-time 600 --fail "$u/results.tgz" -o "$out/results.tgz"
    log "fetched $out/results.tgz"
    ;;
  down)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    id="${2:?pod id}"
    sha="$(rest GET "/pods/$id" 2>/dev/null | jq -r '.env.FV_BUNDLE_SHA256 // empty' 2>/dev/null || true)"
    rest DELETE "/pods/$id" >/dev/null 2>&1 || true
    if rest GET "/pods/$id" >/dev/null 2>&1; then log "WARNING: pod $id still listed"; exit 1; fi
    log "pod $id deleted (GET /pods/$id: not found)"
    if [[ -n "$sha" ]]; then
      rw python3 "$G/r2-weights.py" --bucket fv-weights-staging rm-temp --prefix "tmp/h3trim-vaemx-bundle-$sha" || true
    else
      log "bundle sha256 not readable from pod $id: remove tmp/h3trim-vaemx-bundle-* by hand"
    fi
    ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
