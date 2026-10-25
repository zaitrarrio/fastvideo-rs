#!/usr/bin/env bash
# H3 glue-fusion timing on ONE Runpod RTX PRO 6000 (docs/perf/h3-glue-fusion.md): two fv-gpucheck binaries, the
# branch point and the fusions, on the same pod, by the pod script scripts/gpu/experiments/h3-glue-pod.sh.
#
#   h3-glue-validate.sh up BASE_BIN NEW_BIN   create the pod (+ a detached wall-clock backstop that deletes it)
#   h3-glue-validate.sh status <pod>          status.tsv, the log tail, DONE
#   h3-glue-validate.sh fetch <pod> DIR       results.tgz into DIR
#   h3-glue-validate.sh down <pod>            delete the pod, check it is gone, drop the staged bundle by its key
#   h3-glue-validate.sh print-start           the pod's start command (what up sends)
#
# Where: EU-RO-1 when it has stock, mounting the EU weight volume (h3-base, the turbo LoRA and pre-quantized DiT) at
# /workspace; FastH3 V2 NVFP4's transformer/ (53.5 GB) comes from R2. FV_ALLOW_R2=1 also allows any other data centre
# with stock, h3-base (144 GB), the pre-quantized DiT and the LoRA then coming from R2 too. No CUDA-version filter
# (2026-10-14: hosts run CUDA 13.2 drivers). The R2 plan (presigned GETs) and the bundle reach the pod through the
# staging bucket (a presigned URL, checked by SHA-256); no key goes to the pod.
#
# Env: RUNPOD_API_KEY; FV_POD_CAP_S (backstop, default 3600); FV_MIN_BALANCE (default 10: refuse unless the balance
# is 3 above it); FV_MAX_DPH (default 2.70); FV_SKIP, FV_ONLY_NVFP4 (passed to the pod script); FV_RUNTIME_IMAGE.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="$(cd "$HERE/.." && pwd)"   # scripts/gpu: the shared tooling these drivers use
# shellcheck source-path=SCRIPTDIR/.. source=volumes.sh
source "$GPU/volumes.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
CAP_S="${FV_POD_CAP_S:-3600}"
MIN_BALANCE="${FV_MIN_BALANCE:-10}"
MAX_DPH="${FV_MAX_DPH:-2.70}"
IMAGE="${FV_RUNTIME_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-8f20822}"
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

# The pod's start command, as `up` sends it (`print-start` prints it): the
# self-delete library first (fv_self_delete, docs/adr/0023-lab-report-and-ci-bridge.md §4),
# then the run; 10 min after DONE the pod deletes itself, retried each minute.
start_cmd() {
  printf '%s\n' "$(<"$HERE/../lib/self-delete.sh")"
  printf '%s' 'mkdir -p /opt/fv /srv && cd /opt/fv && curl -fsSL --retry 5 -o b.tgz "$FV_BUNDLE_URL" && echo "$FV_BUNDLE_SHA256  b.tgz" | sha256sum -c - >/srv/log.txt 2>&1 && tar xzf b.tgz && rm -f b.tgz \
&& (curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uvbin UV_NO_MODIFY_PATH=1 sh) >>/srv/log.txt 2>&1 \
&& export PATH=/opt/uvbin:$PATH UV_PYTHON_INSTALL_DIR=/opt/uvpy && uv python install 3.12 >>/srv/log.txt 2>&1 && export PY="$(uv python find 3.12)" \
&& (cd /srv && "$PY" -m http.server 8000 >/dev/null 2>&1 &) && bash /opt/fv/h3-glue-pod.sh; [ -f /srv/DONE ] || echo 1 >/srv/DONE; \
sleep 600; if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then until fv_self_delete /tmp/self-delete.log; do sleep 60; done; fi; sleep infinity'
}
cmd="${1:-}"
case "$cmd" in
  print-start) start_cmd ;;
  up)
    base_bin="${2:?release fv-gpucheck of the branch point}"
    new_bin="${3:?release fv-gpucheck of this branch}"
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
    # The REST API's dataCenterIds enum is narrower than GraphQL's list (US-MO-2, US-NC-2, ... are refused, 400).
    rest_dcs=" EU-RO-1 CA-MTL-1 CA-MTL-2 CA-MTL-3 EU-SE-1 EU-CZ-1 EU-NL-1 EUR-IS-1 EUR-IS-2 EUR-IS-3 US-IL-1 US-TX-1 US-TX-3 US-TX-4 US-KS-2 US-GA-2 US-WA-1 US-CA-2 US-NC-1 US-DE-1 OC-AU-1 AP-JP-1 "
    mapfile -t dcs < <(for d in "${dcs[@]}"; do [[ "$rest_dcs" == *" $d "* ]] && echo "$d"; done)
    if printf '%s\n' "${dcs[@]}" | grep -qx "$FV_EU_DC"; then
      dcs=("$FV_EU_DC")
    elif [[ "${FV_ALLOW_R2:-0}" == 1 ]]; then
      mode=r2
    else
      log "no RTX PRO 6000 stock in $FV_EU_DC (FV_ALLOW_R2=1 fetches h3-base from R2 elsewhere)"; exit 1
    fi
    (( ${#gpus[@]} && ${#dcs[@]} )) || { log "no RTX PRO 6000 in stock at or under \$$MAX_DPH/hr"; exit 1; }
    log "mode $mode: ${dcs[*]}"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    mkdir -p "$tmp/b/base" "$tmp/b/new"
    trees=(fasth3-v2-nvfp4)
    if [[ "$mode" == r2 && "${FV_ONLY_NVFP4:-0}" == 1 ]]; then trees+=(h3-base)
    elif [[ "$mode" == r2 ]]; then trees+=(h3-base h3-base/transformer_prequant_4step-vsa_mxfp8 FastH3-4-step-Preview-v1-LoRA); fi
    ro=/root/.config/fv
    FV_WEIGHTS_R2_ACCESS_KEY_ID="$(cat "$ro/r2_weights_ro_access_key_id")" FV_WEIGHTS_R2_SECRET_ACCESS_KEY="$(cat "$ro/r2_weights_ro_secret_access_key")" \
      python3 "$GPU/r2-weights.py" plan "${trees[@]}" --ttl 10800 --out "$tmp/plan-all.json"
    python3 - "$tmp/plan-all.json" "$tmp/b" <<'PYEOF'
import fnmatch, json, os, sys
plan = json.load(open(sys.argv[1]))
keep = {"fasth3-v2-nvfp4": ("transformer/*",), "FastH3-4-step-Preview-v1-LoRA": ("vsa-datafree/*",)}
if os.environ.get("FV_ONLY_NVFP4") == "1":  # V2 reads h3-base's encoder, tokenizer and VAEs, not its DiT
    keep["h3-base"] = ("text_encoder/*", "tokenizer/*", "audio_vae/*", "vae/*", "*.json")
for t in plan["trees"]:
    if t["dest"] in keep:
        t["files"] = [f for f in t["files"] if any(fnmatch.fnmatch(f["path"], g) for g in keep[t["dest"]])]
        t["total_bytes"] = sum(f["size"] for f in t["files"])
for name, pick in (("plan-nvfp4.json", lambda d: d == "fasth3-v2-nvfp4"), ("plan-turbo.json", lambda d: d != "fasth3-v2-nvfp4")):
    part = dict(plan, trees=[t for t in plan["trees"] if pick(t["dest"])])
    fd = os.open(os.path.join(sys.argv[2], name), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(part, f)
print(" ".join(f"{t['dest']}:{len(t['files'])}f/{t['total_bytes'] / 1e9:.1f}GB" for t in plan["trees"]), file=sys.stderr)
PYEOF
    rm -f "$tmp/plan-all.json"
    cp "$base_bin" "$tmp/b/base/fv-gpucheck" && cp "$new_bin" "$tmp/b/new/fv-gpucheck"
    chmod 755 "$tmp/b/base/fv-gpucheck" "$tmp/b/new/fv-gpucheck"
    cp "$GPU/r2-fetch.py" "$HERE/h3-glue-pod.sh" "$GPU/prompts-hd5.json" "$tmp/b/"
    tar czf "$tmp/bundle.tgz" -C "$tmp/b" .
    rm -rf "$tmp/b"
    sha="$(sha256sum "$tmp/bundle.tgz" | cut -d' ' -f1)"
    url="$(python3 "$GPU/r2-weights.py" --bucket fv-weights-staging put-temp "$tmp/bundle.tgz" --key "tmp/h3-glue-bundle-$sha.tgz" --ttl 10800)"
    log "balance \$$bal; bundle ${sha:0:12} ($(du -h "$tmp/bundle.tgz" | cut -f1)) staged"
    rm -f "$tmp/bundle.tgz"
    # The runtime image has curl and sha256sum but no Python: uv brings one (r2-fetch.py, the prompt file). The pod
    # deletes itself 10 min after DONE (Runpod's pod-scoped RUNPOD_API_KEY / RUNPOD_POD_ID); the backstop regardless.
    start="$(start_cmd)"
    export FVX_URL="$url"
    gpus_json="$(printf '%s\n' "${gpus[@]}" | jq -R . | jq -s .)"
    dcs_json="$(printf '%s\n' "${dcs[@]}" | jq -R . | jq -s .)"
    payload="$(jq -n --arg name "fv-h3glue-$(date -u +%m%d%H%M)" --arg image "$IMAGE" --arg vol "$FV_EU_VOLUME_ID" --argjson dcs "$dcs_json" \
      --arg mode "$mode" --arg start "$start" --arg sha "$sha" --argjson gpus "$gpus_json" --arg skip "${FV_SKIP:-}" --arg only "${FV_ONLY_NVFP4:-0}" '{
        name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU", gpuCount: 1, gpuTypeIds: $gpus,
        gpuTypePriority: "custom", dataCenterIds: $dcs, ports: ["8000/http"], dockerStartCmd: ["/bin/bash", "-lc", $start],
        env: ({FV_BUNDLE_URL: env.FVX_URL, FV_BUNDLE_SHA256: $sha, FV_WEIGHTS_MODE: $mode}
              + (if $skip != "" then {FV_SKIP: $skip} else {} end) + {FV_ONLY_NVFP4: $only})
      } + (if $mode == "volume" then {containerDiskInGb: 150, networkVolumeId: $vol, volumeMountPath: "/workspace"}
           else {containerDiskInGb: 400} end)')"
    unset FVX_URL
    unstage() { python3 "$GPU/r2-weights.py" --bucket fv-weights-staging rm-temp --prefix "tmp/h3-glue-bundle-$sha" >&2 || true; }
    resp="$(rest POST /pods <<<"$payload")" || { log "pod create failed: $(head -c 600 <<<"$resp")"; unstage; exit 1; }
    id="$(jq -r '.id // empty' <<<"$resp")"
    [[ -n "$id" ]] || { log "pod create failed: $(head -c 400 <<<"$resp")"; unstage; exit 1; }
    # A distinctive marker in the backstop's command line, so no generic pkill pattern matches it.
    setsid nohup bash -c ": fv-h3glue-backstop-$id; sleep $CAP_S; curl -sS -X DELETE -H @<(printf 'Authorization: Bearer %s\\n' \"\$RUNPOD_API_KEY\") '$API/pods/$id' >/dev/null 2>&1" >/dev/null 2>&1 </dev/null &
    log "pod $id $(jq -r '.machine.gpuDisplayName // .gpuTypeId // "?"' <<<"$resp") \$$(jq -r '.costPerHr' <<<"$resp")/hr in $(jq -r '.machine.dataCenterId // "?"' <<<"$resp"); backstop pid $! deletes it after ${CAP_S}s; https://$id-8000.proxy.runpod.net/"
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
    own="$(rest GET "/pods/$id" 2>/dev/null | jq -r '.env.FV_BUNDLE_SHA256 // ""' 2>/dev/null || true)"
    rest DELETE "/pods/$id" >/dev/null 2>&1 || true
    if rest GET "/pods/$id" >/dev/null 2>&1; then log "WARNING: pod $id still listed"; exit 1; fi
    log "pod $id deleted (GET /pods/$id: not found)"
    if [[ -n "$own" ]]; then
      python3 "$GPU/r2-weights.py" --bucket fv-weights-staging rm-temp --prefix "tmp/h3-glue-bundle-$own" || true
    else
      log "bundle sha256 not readable from pod $id: its staged bundle is left under tmp/h3-glue-bundle-; remove it by its full key"
    fi
    ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
