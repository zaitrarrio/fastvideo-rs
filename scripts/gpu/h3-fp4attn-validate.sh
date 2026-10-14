#!/usr/bin/env bash
# H3 sparse FP4 VSA attention on ONE Runpod RTX PRO 6000 (docs/ports/fasth3-sparse-fp4-attn.md): the 768p trace of the
# NVFP4 path, the kernel's unit checks, parity and timings, by the pod script scripts/gpu/h3-fp4attn-pod.sh.
#
#   h3-fp4attn-validate.sh up IMAGE       create the pod (+ a detached wall-clock backstop that deletes it). IMAGE is a
#                                         fastvideo-rs runtime image (ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-<7>)
#                                         whose fv-gpucheck runs; the pod script travels in the env (FV_POD_SCRIPT_B64)
#   h3-fp4attn-validate.sh status <pod>   status.tsv, the log tail, DONE
#   h3-fp4attn-validate.sh fetch <pod> DIR  results.tgz into DIR
#   h3-fp4attn-validate.sh down <pod>     delete the pod and check it is gone (GET: 404)
#
# Nothing is written to R2 or a volume: the R2 plan (presigned GETs from the read-only key) reaches the pod gzip +
# base64 in its env. With stock in EU-RO-1 the pod mounts the EU weight volume read only (h3-base: text encoder,
# tokenizer, VAEs); otherwise (or FV_NO_VOLUME=1) those come from R2 too. No CUDA-version filter (2026-10-14: hosts
# run CUDA 13.2 drivers).
#
# Env: RUNPOD_API_KEY; FV_POD_CAP_S (backstop, default 3600); FV_MIN_BALANCE (default 10: refuse unless the balance
# is 3 above it); FV_MAX_DPH (default 2.70); FV_STEPS / FV_ARMS / FV_KERNELS_GATE (passed to the pod script);
# FV_WITH_4S=1 (also the 4-step tree).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=volumes.sh
source "$HERE/volumes.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
CAP_S="${FV_POD_CAP_S:-3600}"
MIN_BALANCE="${FV_MIN_BALANCE:-10}"
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
    image="${2:?fastvideo-rs runtime image}"
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
    allow=" ${FV_DCS:-EU-RO-1 CA-MTL-1 CA-MTL-2 CA-MTL-3 EU-SE-1 EU-CZ-1 EU-NL-1 EUR-IS-1 EUR-IS-2 EUR-IS-3 US-IL-1 US-TX-1 US-TX-3 US-TX-4 US-KS-2 US-GA-2 US-WA-1 US-CA-2 US-NC-1 US-DE-1 OC-AU-1 AP-JP-1} "
    mapfile -t dcs < <(for d in "${dcs[@]}"; do [[ "$allow" == *" $d "* ]] && echo "$d"; done)
    if [[ "${FV_NO_VOLUME:-0}" != 1 ]] && printf '%s\n' "${dcs[@]}" | grep -qx "$FV_EU_DC"; then
      dcs=("$FV_EU_DC")
    else
      mode=r2
    fi
    (( ${#gpus[@]} && ${#dcs[@]} )) || { log "no RTX PRO 6000 in stock at or under \$$MAX_DPH/hr"; exit 1; }
    log "mode $mode: ${dcs[*]}"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    trees=(fasth3-v2-nvfp4)
    [[ "${FV_WITH_4S:-0}" == 1 ]] && trees+=(fasth3-4step-vsa-nvfp4)
    [[ "$mode" == r2 ]] && trees+=(h3-base)
    ro=/root/.config/fv
    FV_WEIGHTS_R2_ACCESS_KEY_ID="$(cat "$ro/r2_weights_ro_access_key_id")" FV_WEIGHTS_R2_SECRET_ACCESS_KEY="$(cat "$ro/r2_weights_ro_secret_access_key")" \
      python3 "$HERE/r2-weights.py" plan "${trees[@]}" --ttl 10800 --out "$tmp/plan.json"
    # Only what the pod reads: the NVFP4 transformers (their NVFP4 vae/ is not read), and in r2 mode h3-base's
    # tokenizer, text encoder (streamed once per prompt, then cached) and VAEs, not its transformer.
    python3 - "$tmp/plan.json" "$tmp/plan.b64" <<'PYEOF'
import base64, fnmatch, gzip, json, os, sys
plan_path, out = sys.argv[1], sys.argv[2]
keep = {"fasth3-v2-nvfp4": ("transformer/*",), "fasth3-4step-vsa-nvfp4": ("transformer/*",),
        "h3-base": ("tokenizer/*", "text_encoder/*", "audio_vae/*", "vae/*", "model_index.json")}
plan = json.load(open(plan_path))
trees = []
for t in plan["trees"]:
    if t["dest"] not in keep:
        continue
    t["files"] = [f for f in t["files"] if any(fnmatch.fnmatch(f["path"], g) for g in keep[t["dest"]])]
    t["total_bytes"] = sum(f["size"] for f in t["files"])
    trees.append(t)
plan["trees"] = trees
b64 = base64.b64encode(gzip.compress(json.dumps(plan, separators=(",", ":")).encode(), 9))
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "wb") as f:
    f.write(b64)
if len(b64) > 4 * 60000:
    sys.exit(f"plan too large for the pod env: {len(b64)} bytes")
print(" ".join(f"{t['dest']}:{len(t['files'])}f/{t['total_bytes'] / 1e9:.1f}GB" for t in trees) + f"; env {len(b64)} bytes",
      file=sys.stderr)
PYEOF
    rm -f "$tmp/plan.json"
    gzip -9c "$HERE/h3-fp4attn-pod.sh" | base64 -w0 >"$tmp/script.b64"
    # The runtime image has curl and coreutils but no Python: uv brings one. The pod deletes itself 10 min after
    # DONE (Runpod's pod-scoped RUNPOD_API_KEY / RUNPOD_POD_ID); the backstop deletes it regardless.
    start='mkdir -p /opt/fv /srv && printf %s "${FV_PLAN_GZ_B64_0:-}" "${FV_PLAN_GZ_B64_1:-}" "${FV_PLAN_GZ_B64_2:-}" "${FV_PLAN_GZ_B64_3:-}" | base64 -d | gunzip >/opt/fv/plan.json && chmod 600 /opt/fv/plan.json \
&& printf %s "$FV_POD_SCRIPT_B64" | base64 -d | gunzip >/opt/fv/pod.sh \
&& (curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uvbin UV_NO_MODIFY_PATH=1 sh) >>/srv/log.txt 2>&1 \
&& export PATH=/opt/uvbin:$PATH UV_PYTHON_INSTALL_DIR=/opt/uvpy && uv python install 3.12 >>/srv/log.txt 2>&1 && export PY="$(uv python find 3.12)" \
&& (cd /srv && "$PY" -m http.server 8000 >/dev/null 2>&1 &) && unset FV_PLAN_GZ_B64_0 FV_PLAN_GZ_B64_1 FV_PLAN_GZ_B64_2 FV_PLAN_GZ_B64_3 FV_POD_SCRIPT_B64 && bash /opt/fv/pod.sh; [ -f /srv/DONE ] || echo 1 >/srv/DONE; \
sleep 600; [ -n "${RUNPOD_POD_ID:-}" ] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"; sleep infinity'
    gpus_json="$(printf '%s\n' "${gpus[@]}" | jq -R . | jq -s .)"
    dcs_json="$(printf '%s\n' "${dcs[@]}" | jq -R . | jq -s .)"
    payload="$(jq -n --arg name "fv-h3fp4attn-$(date -u +%m%d%H%M)" --arg image "$image" --arg vol "$FV_EU_VOLUME_ID" \
      --argjson dcs "$dcs_json" --arg mode "$mode" --arg start "$start" --argjson gpus "$gpus_json" \
      --rawfile plan "$tmp/plan.b64" --rawfile script "$tmp/script.b64" --arg steps "${FV_STEPS:-}" \
      --arg arms "${FV_ARMS:-}" --arg gate "${FV_KERNELS_GATE:-1}" '{
        name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU", gpuCount: 1, gpuTypeIds: $gpus,
        gpuTypePriority: "custom", dataCenterIds: $dcs, dataCenterPriority: "availability", ports: ["8000/http"],
        dockerStartCmd: ["/bin/bash", "-lc", $start],
        env: ({FV_WEIGHTS_MODE: $mode, FV_POD_SCRIPT_B64: $script, FV_KERNELS_GATE: $gate}
              + ([range(0; $plan | length; 60000) as $i | {key: "FV_PLAN_GZ_B64_\($i / 60000 | floor)", value: $plan[$i:$i + 60000]}] | from_entries)
              + (if $steps != "" then {FV_STEPS: $steps} else {} end) + (if $arms != "" then {FV_ARMS: $arms} else {} end))
      } + (if $mode == "volume" then {containerDiskInGb: 160, networkVolumeId: $vol, volumeMountPath: "/workspace"}
           else {containerDiskInGb: 260} end)')"
    resp="$(rest POST /pods <<<"$payload")" || { log "pod create failed: $(head -c 600 <<<"$resp")"; exit 1; }
    id="$(jq -r '.id // empty' <<<"$resp")"
    [[ -n "$id" ]] || { log "pod create failed: $(head -c 400 <<<"$resp")"; exit 1; }
    # A distinctive marker in the backstop's command line, so no generic pkill pattern matches it.
    setsid nohup bash -c ": fv-h3fp4attn-backstop-$id; sleep $CAP_S; curl -sS -X DELETE -H @<(printf 'Authorization: Bearer %s\\n' \"\$RUNPOD_API_KEY\") '$API/pods/$id' >/dev/null 2>&1" >/dev/null 2>&1 </dev/null &
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
    rest DELETE "/pods/$id" >/dev/null 2>&1 || true
    if rest GET "/pods/$id" >/dev/null 2>&1; then log "WARNING: pod $id still listed"; exit 1; fi
    log "pod $id deleted (GET /pods/$id: not found)"
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
