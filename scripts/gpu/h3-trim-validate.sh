#!/usr/bin/env bash
# FastH3 Trim GPU validation on ONE Runpod RTX PRO 6000 (docs/ports/h3-trim.md §6): upstream FastVideo parity
# (oracle dumps, per-step latents, decoded video + audio PSNR) and the timing against FastH3 V2, on the pod script
# scripts/gpu/h3-trim-pod.sh.
#
#   h3-trim-validate.sh up BIN          create the pod (+ a detached wall-clock backstop that deletes it); BIN is the
#                                       release fv-gpucheck to run (built from this tree on the build pod)
#   h3-trim-validate.sh status <pod>    status.tsv, the log tail, DONE
#   h3-trim-validate.sh fetch <pod> DIR the results (logs, JSON, comparisons, clips) into DIR
#   h3-trim-validate.sh down <pod>      delete the pod, check it is gone, drop the staged bundle
#
# The pod: the runtime image (CUDA 13.4 libraries; FV_RUNTIME_IMAGE), RTX PRO 6000 Server or Workstation Edition,
# secure cloud. With stock in EU-RO-1 it mounts the EU weight volume (h3-base) at /workspace and fetches the R2
# trees fasth3-trim-8step and h3-8step (without its text encoder) onto the container disk; otherwise (or with
# FV_NO_VOLUME=1) any data centre with stock, and h3-base's text encoder, tokenizer and audio VAE come from R2 too. The R2 plan (presigned GETs, valid 3 h) and the
# bundle reach the pod through the staging bucket (a presigned URL, checked by SHA-256); no key goes to the pod.
#
# Env: FV_UP_DUMP / FV_UP_REF (a previous pod's dumps.tar and reference mp4: skip the upstream run);
# RUNPOD_API_KEY; FV_POD_CAP_S (backstop, default 4800); FV_MIN_BALANCE (default 12: refuse unless the balance
# is 3 above it); FV_MAX_DPH (default 2.70); FV_UP_REV (upstream FastVideo, default d5253fd, the first main with the
# Trim's adaln_rank); FV_PROMPT / FV_SEED / FV_H / FV_W / FV_FRAMES / FV_SKIP (passed to the pod script).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=volumes.sh
source "$HERE/volumes.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
CAP_S="${FV_POD_CAP_S:-4800}"
MIN_BALANCE="${FV_MIN_BALANCE:-12}"
MAX_DPH="${FV_MAX_DPH:-2.70}"
IMAGE="${FV_RUNTIME_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-8f20822}"
UP_REV="${FV_UP_REV:-d5253fd71055927f12aa6c4aa679b93a00238851}"
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
    else
      mode=r2
    fi
    (( ${#gpus[@]} && ${#dcs[@]} )) || { log "no RTX PRO 6000 in stock at or under \$$MAX_DPH/hr"; exit 1; }
    log "mode $mode: ${dcs[*]}"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    mkdir -p "$tmp/b/upstream"
    # The R2 plan: the Trim tree whole, h3-8step without its text encoder and tokenizer (h3-base's serve every
    # run); in r2 mode also h3-base's text encoder, tokenizer and audio VAE (not its 70 GB transformer).
    trees=(fasth3-trim-8step h3-8step)
    [[ "$mode" == r2 ]] && trees+=(h3-base)
    python3 "$HERE/r2-weights.py" plan "${trees[@]}" --ttl 10800 --out "$tmp/plan-all.json"
    python3 - "$tmp/plan-all.json" "$tmp/b/plan.json" <<'PYEOF'
import fnmatch, json, os, sys
plan = json.load(open(sys.argv[1]))
keep = {"h3-8step": ("transformer/*", "vae/*", "audio_vae/*", "*.json"),
        "h3-base": ("text_encoder/*", "tokenizer/*", "audio_vae/*", "model_index.json")}
for t in plan["trees"]:
    if t["dest"] in keep:
        t["files"] = [f for f in t["files"] if any(fnmatch.fnmatch(f["path"], g) for g in keep[t["dest"]])
                      and (t["dest"] == "h3-base" or not f["path"].startswith(("text_encoder/", "tokenizer/")))]
        t["total_bytes"] = sum(f["size"] for f in t["files"])
fd = os.open(sys.argv[2], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    json.dump(plan, f)
print(" ".join(f"{t['dest']}:{len(t['files'])}f/{t['total_bytes'] / 1e9:.1f}GB" for t in plan["trees"]), file=sys.stderr)
PYEOF
    cp "$bin" "$tmp/b/fv-gpucheck"
    cp "$HERE/r2-fetch.py" "$HERE/h3-trim-pod.sh" "$tmp/b/"
    cp -r "$HERE/upstream/oracle_site" "$HERE"/upstream/{oracle_dump.py,oracle_fastvideo.py,bench_fastvideo.py,setup.sh,av_psnr.py} "$tmp/b/upstream/"
    tar czf "$tmp/bundle.tgz" -C "$tmp/b" .
    sha="$(sha256sum "$tmp/bundle.tgz" | cut -d' ' -f1)"
    url="$(python3 "$HERE/r2-weights.py" --bucket fv-weights-staging put-temp "$tmp/bundle.tgz" --key "tmp/h3-trim-bundle-$sha.tgz" --ttl 10800)"
    log "balance \$$bal; bundle ${sha:0:12} ($(du -h "$tmp/bundle.tgz" | cut -f1)) staged"
    # The runtime image has curl and sha256sum but no Python: uv brings one. The pod deletes itself 20 min after
    # DONE (Runpod injects a pod-scoped RUNPOD_API_KEY / RUNPOD_POD_ID); the local backstop deletes it regardless.
    start='mkdir -p /opt/fv /srv && cd /opt/fv && curl -fsSL --retry 5 -o b.tgz "$FV_BUNDLE_URL" && echo "$FV_BUNDLE_SHA256  b.tgz" | sha256sum -c - >/srv/log.txt 2>&1 && tar xzf b.tgz && chmod +x fv-gpucheck \
&& (curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uvbin UV_NO_MODIFY_PATH=1 sh) >>/srv/log.txt 2>&1 \
&& export PATH=/opt/uvbin:$PATH UV_PYTHON_INSTALL_DIR=/opt/uvpy && uv python install 3.12 >>/srv/log.txt 2>&1 && export PY="$(uv python find 3.12)" \
&& (cd /srv && "$PY" -m http.server 8000 >/dev/null 2>&1 &) && bash /opt/fv/h3-trim-pod.sh; [ -f /srv/DONE ] || echo 1 >/srv/DONE; \
sleep 1200; [ -n "${RUNPOD_POD_ID:-}" ] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"; sleep infinity'
    # FV_UP_DUMP / FV_UP_REF: a previous pod's dumps.tar and up/ref.mp4 (fetched with `fetch`), staged so this pod
    # reuses the upstream run instead of installing and running FastVideo again.
    up_dump_url="" up_dump_sha="" up_ref_url=""
    if [[ -n "${FV_UP_DUMP:-}" ]]; then
      up_dump_sha="$(sha256sum "$FV_UP_DUMP" | cut -d' ' -f1)"
      up_dump_url="$(python3 "$HERE/r2-weights.py" --bucket fv-weights-staging put-temp "$FV_UP_DUMP" --key "tmp/h3-trim-bundle-updump-$up_dump_sha.tar" --ttl 10800)"
      up_ref_url="$(python3 "$HERE/r2-weights.py" --bucket fv-weights-staging put-temp "${FV_UP_REF:?FV_UP_REF with FV_UP_DUMP}" --key "tmp/h3-trim-bundle-upref-$up_dump_sha.mp4" --ttl 10800)"
      log "upstream dump ${up_dump_sha:0:12} staged: this pod reuses it"
    fi
    export FVX_URL="$url" FVX_UPD="$up_dump_url" FVX_UPR="$up_ref_url"
    gpus_json="$(printf '%s\n' "${gpus[@]}" | jq -R . | jq -s .)"
    dcs_json="$(printf '%s\n' "${dcs[@]}" | jq -R . | jq -s .)"
    payload="$(jq -n --arg name "fv-h3trim-$(date -u +%m%d%H%M)" --arg image "$IMAGE" --arg vol "$FV_EU_VOLUME_ID" --argjson dcs "$dcs_json" --arg mode "$mode" \
      --arg start "$start" --arg sha "$sha" --argjson gpus "$gpus_json" --arg rev "$UP_REV" \
      --arg prompt "${FV_PROMPT:-}" --arg seed "${FV_SEED:-}" --arg h "${FV_H:-}" --arg w "${FV_W:-}" --arg nf "${FV_FRAMES:-}" --arg skip "${FV_SKIP:-}" --arg upsha "$up_dump_sha" '{
        name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU", gpuCount: 1, gpuTypeIds: $gpus,
        gpuTypePriority: "custom", dataCenterIds: $dcs, ports: ["8000/http"], dockerStartCmd: ["/bin/bash", "-lc", $start],
        env: ({FV_BUNDLE_URL: env.FVX_URL, FV_BUNDLE_SHA256: $sha, FV_UP_REV: $rev, FV_WEIGHTS_MODE: $mode}
              + (if $prompt != "" then {FV_PROMPT: $prompt} else {} end) + (if $seed != "" then {FV_SEED: $seed} else {} end)
              + (if $h != "" then {FV_H: $h} else {} end) + (if $w != "" then {FV_W: $w} else {} end)
              + (if $nf != "" then {FV_FRAMES: $nf} else {} end) + (if $skip != "" then {FV_SKIP: $skip} else {} end)
              + (if $upsha != "" then {FV_UP_DUMP_URL: env.FVX_UPD, FV_UP_DUMP_SHA256: $upsha, FV_UP_REF_URL: env.FVX_UPR} else {} end))
      } + (if $mode == "volume" then {containerDiskInGb: 220, networkVolumeId: $vol, volumeMountPath: "/workspace"}
           else {containerDiskInGb: 300, allowedCudaVersions: ["13.0"]} end)')"
    unset FVX_URL FVX_UPD FVX_UPR
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
    for f in log.txt status.tsv nvidia-smi.txt nvsmi.csv fetch.log setup.log upstream.log fetch-status.json \
             ours-inj.log ours-inj-served.log compare-dumps-inj.txt compare-dumps-inj-served.txt \
             av-psnr-inj.json av-psnr-inj-served.json time-trim.log time-v2.log trim-vs-v2.json \
             up/result.json up/ref.mp4 ours/inj/output.mp4 ours/inj-served/output.mp4 ours/trim/output.mp4 \
             ours/v2/output.mp4 ours/inj/audio.wav ours/trim/audio.wav ours/v2/audio.wav; do
      mkdir -p "$out/$(dirname "$f")"
      curl -sS --max-time 300 --fail "$u/$f" -o "$out/$f" 2>/dev/null || rm -f "$out/$f"
    done
    log "fetched into $out"
    ;;
  down)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    id="${2:?pod id}"
    rest DELETE "/pods/$id" >/dev/null 2>&1 || true
    if rest GET "/pods/$id" >/dev/null 2>&1; then log "WARNING: pod $id still listed"; exit 1; fi
    log "pod $id deleted (GET /pods/$id: not found)"
    python3 "$HERE/r2-weights.py" --bucket fv-weights-staging rm-temp --prefix tmp/h3-trim-bundle- || true
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
