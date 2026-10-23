#!/usr/bin/env bash
# The H3 NVFP4 tier's three speed / quality options on ONE Runpod RTX PRO 6000 (docs/perf/h3-nvfp4-tier-options.md):
# TAEH3 decode (option 5), the published NVFP4 VAE decoder linears (option 6) and the Trim 8-step NVFP4 checkpoint
# (option 7), by the pod script scripts/gpu/experiments/h3-tier-options-pod.sh. One fetch from R2 serves every arm.
#
#   h3-tier-options-validate.sh up IMAGE BIN    create the pod (+ a detached wall-clock backstop that deletes it).
#                                               IMAGE: a fastvideo-rs runtime image (ghcr.io/zaitrarrio/
#                                               fastvideo-rs-runtime:sha-<7>, for its CUDA libraries); BIN: this
#                                               branch's release fv-gpucheck (build pod: build-pod.sh fetch)
#   h3-tier-options-validate.sh status <pod>    status.tsv, the log tail, DONE
#   h3-tier-options-validate.sh fetch <pod> DIR results.tgz into DIR (checked with gzip -t / tar -t first)
#   h3-tier-options-validate.sh down <pod>      delete the pod, check it is gone (GET: 404), drop the staged bundle
#
# The bundle (BIN, scripts/gpu and the R2 plan: presigned GETs from the read-only key) is staged in
# fv-weights-staging under tmp/h3-tier-options-bundle-<sha256>.tgz (presigned GET, SHA-256 checked on the pod) and
# removed by `down` (or a failed create) by that exact key. Nothing is written to R2's weights bucket or to a volume.
# With stock in EU-RO-1 the pod mounts the EU weight volume (h3-base: text encoder, tokenizer, VAEs) and only reads
# it; otherwise (or FV_NO_VOLUME=1) h3-base comes from R2 too.
#
# Env: RUNPOD_API_KEY; FV_POD_CAP_S (backstop, default 6000); FV_MIN_BALANCE (default 10: refuse unless the balance
# is 3 above it); FV_MAX_DPH (default 2.70); FV_STEPS / FV_RES / FV_ARMS (passed to the pod script); FV_DCS.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="$(cd "$HERE/.." && pwd)"   # scripts/gpu: the shared tooling these drivers use
# shellcheck source-path=SCRIPTDIR/.. source=volumes.sh
source "$GPU/volumes.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
CAP_S="${FV_POD_CAP_S:-6000}"
MIN_BALANCE="${FV_MIN_BALANCE:-10}"
MAX_DPH="${FV_MAX_DPH:-2.70}"
PREFIX=tmp/h3-tier-options-bundle-
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
    bin="${3:?the release fv-gpucheck built from this branch}"
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    bal="$(balance)"
    awk -v b="$bal" -v m="$MIN_BALANCE" 'BEGIN{exit !(b+0 >= m+3)}' || { log "balance \$$bal too close to the \$$MIN_BALANCE floor"; exit 1; }
    stock="$(gql <<<'{"query":"{ dataCenters { id gpuAvailability { gpuTypeId stockStatus } } }"}')"
    gpus=(); mode=volume
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
    trees=(fasth3-v2-nvfp4 fasth3-trim-8step fasth3-trim-8step-nvfp4 auxiliary/tae auxiliary/lpips)
    [[ "$mode" == r2 ]] && trees+=(h3-base)
    ro=/root/.config/fv
    FV_WEIGHTS_R2_ACCESS_KEY_ID="$(cat "$ro/r2_weights_ro_access_key_id")" FV_WEIGHTS_R2_SECRET_ACCESS_KEY="$(cat "$ro/r2_weights_ro_secret_access_key")" \
      python3 "$GPU/r2-weights.py" plan "${trees[@]}" --ttl 14400 --out "$tmp/plan.json"
    mkdir -p "$tmp/b"
    # Only what the pod reads: the V2 NVFP4 transformer + its NVFP4 vae/ shards; the Trim's transformer, its
    # 26-block VAE and configs (not the INT8 ComfyUI VAE); the Trim NVFP4 file; TAEH3; LPIPS; in r2 mode h3-base's
    # tokenizer, text encoder (streamed once per prompt, then cached) and VAEs.
    python3 - "$tmp/plan.json" "$tmp/b/plan.json" <<'PYEOF'
import fnmatch, json, os, sys
src, out = sys.argv[1], sys.argv[2]
keep = {
    "fasth3-v2-nvfp4": ("transformer/*", "vae/*"),
    "fasth3-trim-8step": ("transformer/*", "vae/config.json", "vae/diffusion_pytorch_model.safetensors", "*.json",
                          "scheduler/*", "audio_scheduler/*"),
    "fasth3-trim-8step-nvfp4": ("diffusion_models/*",),
    "auxiliary/tae": ("taeh3.safetensors",),
    "auxiliary/lpips": ("*",),
    "h3-base": ("tokenizer/*", "text_encoder/*", "audio_vae/*", "vae/*", "model_index.json"),
}
plan = json.load(open(src))
trees = []
for t in plan["trees"]:
    if t["dest"] not in keep:
        continue
    t["files"] = [f for f in t["files"] if any(fnmatch.fnmatch(f["path"], g) for g in keep[t["dest"]])]
    t["total_bytes"] = sum(f["size"] for f in t["files"])
    trees.append(t)
plan["trees"] = trees
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    json.dump(plan, f, separators=(",", ":"))
print(" ".join(f"{t['dest']}:{len(t['files'])}f/{t['total_bytes'] / 1e9:.1f}GB" for t in trees)
      + f"; {sum(t['total_bytes'] for t in trees) / 1e9:.1f} GB", file=sys.stderr)
PYEOF
    rm -f "$tmp/plan.json"
    cp "$bin" "$tmp/b/fv-gpucheck" && chmod 755 "$tmp/b/fv-gpucheck"
    cp -r "$GPU" "$tmp/b/gpu"
    tar -czf "$tmp/bundle.tgz" -C "$tmp/b" fv-gpucheck gpu plan.json
    rm -rf "$tmp/b"
    sha="$(sha256sum "$tmp/bundle.tgz" | cut -d' ' -f1)"
    url="$(python3 "$GPU/r2-weights.py" --bucket fv-weights-staging put-temp "$tmp/bundle.tgz" --key "$PREFIX$sha.tgz" --ttl "$CAP_S")"
    rm -f "$tmp/bundle.tgz"
    log "bundle ${sha:0:12} staged ($PREFIX$sha.tgz), binary $(du -h "$bin" | cut -f1)"
    # The runtime image has curl and coreutils but no Python: uv brings one. The pod deletes itself 30 min after
    # DONE (Runpod's pod-scoped RUNPOD_API_KEY / RUNPOD_POD_ID); the backstop deletes it regardless.
    start='mkdir -p /opt/fv /srv && curl -fsSL --retry 5 -o /opt/fv/bundle.tgz "$FV_BUNDLE_URL" && echo "$FV_BUNDLE_SHA256  /opt/fv/bundle.tgz" | sha256sum -c - >>/srv/log.txt 2>&1 \
&& tar -xzf /opt/fv/bundle.tgz -C /opt/fv && rm -f /opt/fv/bundle.tgz && chmod 600 /opt/fv/plan.json && unset FV_BUNDLE_URL \
&& (curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uvbin UV_NO_MODIFY_PATH=1 sh) >>/srv/log.txt 2>&1 \
&& export PATH=/opt/uvbin:$PATH UV_PYTHON_INSTALL_DIR=/opt/uvpy && uv python install 3.12 >>/srv/log.txt 2>&1 && export PY="$(uv python find 3.12)" \
&& (cd /srv && "$PY" -m http.server 8000 >/dev/null 2>&1 &) && bash /opt/fv/gpu/experiments/h3-tier-options-pod.sh; [ -f /srv/DONE ] || echo 1 >/srv/DONE; \
sleep 1800; [ -n "${RUNPOD_POD_ID:-}" ] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"; sleep infinity'
    gpus_json="$(printf '%s\n' "${gpus[@]}" | jq -R . | jq -s .)"
    dcs_json="$(printf '%s\n' "${dcs[@]}" | jq -R . | jq -s .)"
    export FVX_URL="$url"
    payload="$(jq -n --arg name "fv-h3opts-$(date -u +%m%d%H%M)" --arg image "$image" --arg vol "$FV_EU_VOLUME_ID" \
      --argjson dcs "$dcs_json" --arg mode "$mode" --arg start "$start" --argjson gpus "$gpus_json" --arg sha "$sha" \
      --arg steps "${FV_STEPS:-}" --arg res "${FV_RES:-}" --arg arms "${FV_ARMS:-}" '{
        name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU", gpuCount: 1, gpuTypeIds: $gpus,
        gpuTypePriority: "custom", dataCenterIds: $dcs, dataCenterPriority: "availability", ports: ["8000/http"],
        dockerStartCmd: ["/bin/bash", "-lc", $start],
        env: ({FV_BUNDLE_URL: env.FVX_URL, FV_BUNDLE_SHA256: $sha, FV_WEIGHTS_MODE: $mode}
              + (if $steps != "" then {FV_STEPS: $steps} else {} end) + (if $res != "" then {FV_RES: $res} else {} end)
              + (if $arms != "" then {FV_ARMS: $arms} else {} end))
      } + (if $mode == "volume" then {containerDiskInGb: 200, networkVolumeId: $vol, volumeMountPath: "/workspace"}
           else {containerDiskInGb: 280} end)')"
    unset FVX_URL
    unstage() { python3 "$GPU/r2-weights.py" --bucket fv-weights-staging rm-temp --prefix "$PREFIX$sha" >&2 || true; }
    resp="$(rest POST /pods <<<"$payload")" || { log "pod create failed: $(head -c 600 <<<"$resp")"; unstage; exit 1; }
    id="$(jq -r '.id // empty' <<<"$resp")"
    [[ -n "$id" ]] || { log "pod create failed: $(head -c 400 <<<"$resp")"; unstage; exit 1; }
    # A distinctive marker in the backstop's command line, so no generic pkill pattern matches it.
    setsid nohup bash -c ": fv-h3opts-backstop-$id; sleep $CAP_S; curl -sS -X DELETE -H @<(printf 'Authorization: Bearer %s\\n' \"\$RUNPOD_API_KEY\") '$API/pods/$id' >/dev/null 2>&1" >/dev/null 2>&1 </dev/null &
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
    curl -sS --max-time 900 --fail "$u/results.tgz" -o "$out/results.tgz" || { log "fetch failed: no results.tgz (keep the pod until its files are copied)"; exit 1; }
    if gzip -t "$out/results.tgz" && tar -tzf "$out/results.tgz" >/dev/null; then
      tar -xzf "$out/results.tgz" -C "$out" && rm -f "$out/results.tgz"
      log "fetched into $out ($(find "$out" -type f | wc -l) files)"
      curl -sS --max-time 300 --fail "$u/frames-mid.tgz" -o "$out/frames-mid.tgz" 2>/dev/null \
        && gzip -t "$out/frames-mid.tgz" && tar -xzf "$out/frames-mid.tgz" -C "$out" && rm -f "$out/frames-mid.tgz" || true
    else
      log "results.tgz is damaged: keep the pod and pack again"; exit 1
    fi
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
      python3 "$GPU/r2-weights.py" --bucket fv-weights-staging rm-temp --prefix "$PREFIX$sha" || true
    else
      log "bundle sha256 not readable from pod $id: remove $PREFIX<sha>.tgz by its full key"
    fi
    ;;
  *) sed -n '2,22p' "$0"; exit 2 ;;
esac
