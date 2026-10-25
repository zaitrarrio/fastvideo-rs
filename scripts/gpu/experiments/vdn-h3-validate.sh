#!/usr/bin/env bash
# VDN-H3 GPU validation on ONE Runpod RTX PRO 6000 (docs/ports/vdn-h3.md §5): the VDN kernels against the host
# reference, parity against the upstream diffusers component (oracle dumps, per-step latents, decoded video + audio
# PSNR), and timing against FastH3 V2 and h3-turbo, on the pod script scripts/gpu/experiments/vdn-h3-pod.sh.
#
#   vdn-h3-validate.sh up BIN           create the pod (+ a detached wall-clock backstop that deletes it); BIN is the
#                                       release fv-gpucheck to run (built from this tree on the build pod)
#   vdn-h3-validate.sh status <pod>     status.tsv, the log tail, DONE
#   vdn-h3-validate.sh fetch <pod> DIR  the results (logs, JSON, comparisons, clips) into DIR
#   vdn-h3-validate.sh down <pod>       delete the pod, check it is gone, drop the staged bundle
#
# The pod: the runtime image (FV_RUNTIME_IMAGE), RTX PRO 6000 Server or Workstation Edition, secure cloud. With stock
# in EU-RO-1 it mounts the EU weight volume (h3-base, the turbo LoRA and its pre-quantized DiT) at /workspace and
# fetches the R2 trees vdn-h3-stage-b-2000 and h3-8step (DiT, VAEs) onto the container disk; otherwise (or with
# FV_NO_VOLUME=1) any data centre with stock, and h3-base comes from R2 too. The R2 plan (presigned GETs, valid 3 h)
# and the bundle reach the pod through the staging bucket (a presigned URL, checked by SHA-256); no key goes to the pod.
#
# Env: RUNPOD_API_KEY; FV_POD_CAP_S (backstop, default 5400); FV_MIN_BALANCE (default 12: refuse unless the balance
# is 3 above it); FV_MAX_DPH (default 2.70); FV_PROMPT / FV_SEED / FV_H / FV_W / FV_FRAMES / FV_STEPS / FV_SKIP /
# FV_UP_KERNELS / FV_UP_SOFTMAX (passed to the pod script).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="$(cd "$HERE/.." && pwd)"   # scripts/gpu: the shared tooling these drivers use
# shellcheck source-path=SCRIPTDIR/.. source=runpod-cuda.sh
source "$GPU/runpod-cuda.sh"
# shellcheck source-path=SCRIPTDIR/.. source=volumes.sh
source "$GPU/volumes.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
CAP_S="${FV_POD_CAP_S:-5400}"
MIN_BALANCE="${FV_MIN_BALANCE:-12}"
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

cmd="${1:-}"
case "$cmd" in
  up)
    bin="${2:?release fv-gpucheck}"
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    bal="$(balance)"
    awk -v b="$bal" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+3)}' || { log "balance \$$bal too close to the \$$MIN_BALANCE floor"; exit 1; }
    # Where: the EU volume's data centre when it has stock (h3-base read from the volume), else any data centre with
    # stock and the h3-base parts from R2 as well (FV_NO_VOLUME=1 forces that).
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
    if [[ "${FV_NO_VOLUME:-0}" != 1 ]] && printf '%s\n' "${dcs[@]}" | grep -qx "$FV_EU_DC"; then
      dcs=("$FV_EU_DC")
    elif [[ "${FV_REQUIRE_VOLUME:-0}" == 1 ]]; then
      # Elsewhere a host may first pull the runtime image for many minutes (2026-10-13: 12 min, no runtime yet).
      log "no stock in $FV_EU_DC (FV_REQUIRE_VOLUME=1)"; exit 1
    else
      mode=r2
    fi
    (( ${#gpus[@]} && ${#dcs[@]} )) || { log "no RTX PRO 6000 in stock at or under \$$MAX_DPH/hr"; exit 1; }
    log "mode $mode: ${dcs[*]}"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    mkdir -p "$tmp/b/upstream"
    # The R2 plan: the VDN tree whole and h3-8step's DiT and VAEs (FastH3 V2, timed); in r2 mode also h3-base
    # (transformer, VAEs, bf16 text encoder, tokenizer) and the turbo LoRA.
    trees=(vdn-h3-stage-b-2000 h3-8step)
    # FV_VDN_MODEL=8step: the DMD model (its turbo tree over stage B); FastH3 V2 is left out with FV_NO_V2=1.
    [[ "${FV_VDN_MODEL:-}" == 8step ]] && trees+=(vdn-h3-dmd-250-turbo)
    [[ "${FV_NO_V2:-0}" == 1 ]] && trees=("${trees[@]/h3-8step}") && mapfile -t trees < <(printf '%s\n' "${trees[@]}" | grep .)
    [[ "$mode" == r2 ]] && trees+=(h3-base FastH3-4-step-Preview-v1-LoRA)
    python3 "$GPU/r2-weights.py" plan "${trees[@]}" --ttl 10800 --out "$tmp/plan-all.json"
    python3 - "$tmp/plan-all.json" "$tmp/b/plan.json" <<'PYEOF'
import fnmatch, json, os, sys
plan = json.load(open(sys.argv[1]))
keep = {"h3-8step": ("transformer/*", "vae/*", "audio_vae/*", "*.json")}
for t in plan["trees"]:
    if t["dest"] in keep:
        t["files"] = [f for f in t["files"] if any(fnmatch.fnmatch(f["path"], g) for g in keep[t["dest"]])
                      and not f["path"].startswith(("text_encoder/", "tokenizer/"))]
        t["total_bytes"] = sum(f["size"] for f in t["files"])
# Two plans: what the VDN runs need first (plan.json), and FastH3 V2's tree, fetched while they run (plan-v2.json).
for name, pick in ((sys.argv[2], lambda d: d != "h3-8step"), (sys.argv[2].replace("plan.json", "plan-v2.json"), lambda d: d == "h3-8step")):
    part = dict(plan, trees=[t for t in plan["trees"] if pick(t["dest"])])
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(part, f)
print(" ".join(f"{t['dest']}:{len(t['files'])}f/{t['total_bytes'] / 1e9:.1f}GB" for t in plan["trees"]), file=sys.stderr)
PYEOF
    # BIN "-": the image's own fv-gpucheck (a CI runtime image of this branch, FV_RUNTIME_IMAGE=…:sha-<commit>).
    [[ "$bin" == - ]] || cp "$bin" "$tmp/b/fv-gpucheck"
    cp "$GPU/r2-fetch.py" "$HERE/vdn-h3-pod.sh" "$tmp/b/"
    cp "$GPU"/upstream/{oracle_dump.py,oracle_vdn.py,setup.sh,av_psnr.py} "$tmp/b/upstream/"
    tar czf "$tmp/bundle.tgz" -C "$tmp/b" .
    sha="$(sha256sum "$tmp/bundle.tgz" | cut -d' ' -f1)"
    url="$(python3 "$GPU/r2-weights.py" --bucket fv-weights-staging put-temp "$tmp/bundle.tgz" --key "tmp/vdn-h3-bundle-$sha.tgz" --ttl 10800)"
    log "balance \$$bal; bundle ${sha:0:12} ($(du -h "$tmp/bundle.tgz" | cut -f1)) staged"
    # The runtime image has curl and sha256sum but no Python: uv brings one. The pod deletes itself 20 min after
    # DONE (Runpod injects a pod-scoped RUNPOD_API_KEY / RUNPOD_POD_ID); the local backstop deletes it regardless.
    start='mkdir -p /opt/fv /srv && cd /opt/fv && curl -fsSL --retry 5 -o b.tgz "$FV_BUNDLE_URL" && echo "$FV_BUNDLE_SHA256  b.tgz" | sha256sum -c - >/srv/log.txt 2>&1 && tar xzf b.tgz && { [ ! -f fv-gpucheck ] || chmod +x fv-gpucheck; } \
&& (curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uvbin UV_NO_MODIFY_PATH=1 sh) >>/srv/log.txt 2>&1 \
&& export PATH=/opt/uvbin:$PATH UV_PYTHON_INSTALL_DIR=/opt/uvpy && uv python install 3.12 >>/srv/log.txt 2>&1 && export PY="$(uv python find 3.12)" \
&& (cd /srv && "$PY" -m http.server 8000 >/dev/null 2>&1 &) && bash /opt/fv/vdn-h3-pod.sh; [ -f /srv/DONE ] || echo 1 >/srv/DONE; \
sleep 1200; if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then until fv_self_delete /tmp/self-delete.log; do sleep 60; done; fi; sleep infinity'
    # fv_self_delete: GraphQL podTerminate with the pod-scoped key, retried each minute (ADR-0023 §4).
    start="$(<"$HERE/../lib/self-delete.sh")"$'\n'"$start"
    export FVX_URL="$url"
    gpus_json="$(printf '%s\n' "${gpus[@]}" | jq -R . | jq -s .)"
    dcs_json="$(printf '%s\n' "${dcs[@]}" | jq -R . | jq -s .)"
    payload="$(jq -n --arg cuda "${RUNPOD_ALLOWED_CUDA-$(fv_runpod_cuda)}" --arg name "fv-vdnh3-$(date -u +%m%d%H%M)" --arg image "$IMAGE" --arg vol "$FV_EU_VOLUME_ID" --argjson dcs "$dcs_json" --arg mode "$mode" \
      --arg start "$start" --arg sha "$sha" --argjson gpus "$gpus_json" \
      --arg prompt "${FV_PROMPT:-}" --arg seed "${FV_SEED:-}" --arg h "${FV_H:-}" --arg w "${FV_W:-}" --arg nf "${FV_FRAMES:-}" --arg skip "${FV_SKIP:-}" \
      --arg steps "${FV_STEPS:-}" --arg upk "${FV_UP_KERNELS:-}" --arg ups "${FV_UP_SOFTMAX:-}" \
      --arg model "${FV_VDN_MODEL:-}" --arg arms "${FV_INJ_ARMS:-}" '{
        name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU", gpuCount: 1, gpuTypeIds: $gpus,
        gpuTypePriority: "custom", dataCenterIds: $dcs, ports: ["8000/http"], dockerStartCmd: ["/bin/bash", "-lc", $start],
        env: ({FV_BUNDLE_URL: env.FVX_URL, FV_BUNDLE_SHA256: $sha, FV_WEIGHTS_MODE: $mode}
              + (if $prompt != "" then {FV_PROMPT: $prompt} else {} end) + (if $seed != "" then {FV_SEED: $seed} else {} end)
              + (if $h != "" then {FV_H: $h} else {} end) + (if $w != "" then {FV_W: $w} else {} end)
              + (if $nf != "" then {FV_FRAMES: $nf} else {} end) + (if $skip != "" then {FV_SKIP: $skip} else {} end)
              + (if $steps != "" then {FV_STEPS: $steps} else {} end)
              + (if $model != "" then {FV_VDN_MODEL: $model} else {} end) + (if $arms != "" then {FV_INJ_ARMS: $arms} else {} end)
              + (if $upk != "" then {FV_UP_KERNELS: $upk} else {} end) + (if $ups != "" then {FV_UP_SOFTMAX: $ups} else {} end))
      } + (if $mode == "volume" then {containerDiskInGb: 200, networkVolumeId: $vol, volumeMountPath: "/workspace"}
           else {containerDiskInGb: 400} + (if $cuda == "" then {} else {allowedCudaVersions: ($cuda | split(" "))} end) end)')"
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
    curl -sS --max-time 20 --fail "$u/log.txt" 2>/dev/null | tail -n "${FV_TAIL:-6}" || true
    printf 'DONE=%s\n' "$(curl -sS --max-time 20 --fail "$u/DONE" 2>/dev/null || true)"
    ;;
  fetch)
    id="${2:?pod id}"; out="${3:?dir}"; u="https://$id-8000.proxy.runpod.net"
    mkdir -p "$out"
    for f in log.txt status.tsv nvidia-smi.txt nvsmi.csv fetch-vdn.log fetch-v2.log setup.log upstream.log kernels.log \
             gc/kernels.json ours-text.log ours-inj.log ours-inj-served.log compare-dumps-inj.txt compare-dumps-inj-served.txt \
             av-psnr-inj.json av-psnr-inj-served.json up/result.json up/output.mp4 up/audio.wav \
             ours/seed-bf16/output.mp4 ours/inj/output.mp4 ours/inj-served/output.mp4 ours/inj/audio.wav ours/inj-served/audio.wav \
             sample-vdn-50.log ours/vdn50-480/output.mp4; do
      mkdir -p "$out/$(dirname "$f")"
      curl -sS --max-time 300 --fail "$u/$f" -o "$out/$f" 2>/dev/null || rm -f "$out/$f"
    done
    for n in vdn-480 vdn-768 turbo-480 turbo-768 v2-480 v2-768 vdn-768-long vdn-480-long; do
      curl -sS --max-time 120 --fail "$u/time-$n.log" -o "$out/time-$n.log" 2>/dev/null || true
      mkdir -p "$out/ours/$n"
      curl -sS --max-time 300 --fail "$u/ours/$n/output.mp4" -o "$out/ours/$n/output.mp4" 2>/dev/null || rm -f "$out/ours/$n/output.mp4"
    done
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
      python3 "$GPU/r2-weights.py" --bucket fv-weights-staging rm-temp --prefix "tmp/vdn-h3-bundle-$own" || true
    else
      log "bundle sha256 not readable from pod $id: its staged bundle is left under tmp/vdn-h3-bundle-; remove it by its full key"
    fi
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
