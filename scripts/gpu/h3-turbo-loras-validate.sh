#!/usr/bin/env bash
# Turbo LoRA GPU validation on ONE Runpod RTX PRO 6000 (docs/ports/h3-turbo-loras.md): parity against the
# MiniMax-H3-Turbo-Lora Space's own stack (merged weights, per-step latents, decoded video + audio PSNR) and the speed
# and quality of larry 6 / 4, lightx2v 4, h3-turbo and h3-max, on the pod script scripts/gpu/h3-turbo-loras-pod.sh.
#
#   h3-turbo-loras-validate.sh up BIN          create the pod (+ a detached wall-clock backstop that deletes it); BIN
#                                              is the release fv-gpucheck to run (built from this tree on the build pod)
#   h3-turbo-loras-validate.sh status <pod>    status.tsv, the log tail, DONE
#   h3-turbo-loras-validate.sh fetch <pod> DIR the results (logs, JSON, comparisons, small clips) into DIR
#   h3-turbo-loras-validate.sh down <pod>      delete the pod, check it is gone, drop the staged bundle
#
# The pod: the runtime image (FV_RUNTIME_IMAGE), RTX PRO 6000 Server or Workstation Edition, secure cloud, any data
# centre with stock; every tree comes from the R2 weights store onto the container disk (h3-base, the FastH3 LoRA for
# h3-turbo / h3-max, h3-turbo-loras, auxiliary/lpips). The R2 plan (presigned GETs, valid 3 h) and the bundle reach
# the pod through the staging bucket (a presigned URL, checked by SHA-256); no key goes to the pod.
#
# Env: RUNPOD_API_KEY; FV_POD_CAP_S (backstop, default 5400); FV_MIN_BALANCE (default 12: refuse unless the balance
# is 3 above it); FV_MAX_DPH (default 2.70); FV_SEED / FV_SKIP / FV_PARITY_PROMPT (passed to the pod script).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
    mkdir -p "$tmp/b/upstream"
    python3 "$HERE/r2-weights.py" plan h3-base FastH3-4-step-Preview-v1-LoRA h3-turbo-loras/larry h3-turbo-loras/lightx2v auxiliary/lpips \
      --ttl 10800 --out "$tmp/plan-all.json"
    # h3-base without what neither side reads (no Ref2VA, no processor).
    python3 - "$tmp/plan-all.json" "$tmp/b/plan.json" <<'PYEOF'
import fnmatch, json, os, sys
plan = json.load(open(sys.argv[1]))
keep = {"h3-base": ("transformer/*", "text_encoder/*", "tokenizer/*", "vae/*", "audio_vae/*", "model_index.json")}
for t in plan["trees"]:
    if t["dest"] in keep:
        t["files"] = [f for f in t["files"] if any(fnmatch.fnmatch(f["path"], g) for g in keep[t["dest"]])]
        t["total_bytes"] = sum(f["size"] for f in t["files"])
fd = os.open(sys.argv[2], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    json.dump(plan, f)
print(" ".join(f"{t['dest']}:{len(t['files'])}f/{t['total_bytes'] / 1e9:.1f}GB" for t in plan["trees"]), file=sys.stderr)
PYEOF
    cp "$bin" "$tmp/b/fv-gpucheck"
    cp "$HERE/r2-fetch.py" "$HERE/h3-turbo-loras-pod.sh" "$HERE/prompts-eval.json" "$tmp/b/"
    cp "$HERE"/upstream/{h3_turbo_lora_ref.py,merged_parity.py,av_psnr.py} "$tmp/b/upstream/"
    tar czf "$tmp/bundle.tgz" -C "$tmp/b" .
    sha="$(sha256sum "$tmp/bundle.tgz" | cut -d' ' -f1)"
    url="$(python3 "$HERE/r2-weights.py" --bucket fv-weights-staging put-temp "$tmp/bundle.tgz" --key "tmp/h3-turbo-loras-bundle-$sha.tgz" --ttl 10800)"
    log "balance \$$bal; bundle ${sha:0:12} ($(du -h "$tmp/bundle.tgz" | cut -f1)) staged"
    # The runtime image has curl and sha256sum but no Python: uv brings one. The pod deletes itself 20 min after
    # DONE (Runpod injects a pod-scoped RUNPOD_API_KEY / RUNPOD_POD_ID); the local backstop deletes it regardless.
    start='mkdir -p /opt/fv /srv && cd /opt/fv && curl -fsSL --retry 5 -o b.tgz "$FV_BUNDLE_URL" && echo "$FV_BUNDLE_SHA256  b.tgz" | sha256sum -c - >/srv/log.txt 2>&1 && tar xzf b.tgz && chmod +x fv-gpucheck \
&& (curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uvbin UV_NO_MODIFY_PATH=1 sh) >>/srv/log.txt 2>&1 \
&& export PATH=/opt/uvbin:$PATH UV_PYTHON_INSTALL_DIR=/opt/uvpy && uv python install 3.12 >>/srv/log.txt 2>&1 && export PY="$(uv python find 3.12)" \
&& (cd /srv && "$PY" -m http.server 8000 >/dev/null 2>&1 &) && bash /opt/fv/h3-turbo-loras-pod.sh; [ -f /srv/DONE ] || echo 1 >/srv/DONE; \
sleep 1200; [ -n "${RUNPOD_POD_ID:-}" ] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"; sleep infinity'
    export FVX_URL="$url"
    gpus_json="$(printf '%s\n' "${gpus[@]}" | jq -R . | jq -s .)"
    dcs_json="$(printf '%s\n' "${dcs[@]}" | jq -R . | jq -s .)"
    payload="$(jq -n --arg name "fv-h3turbolora-$(date -u +%m%d%H%M)" --arg image "$IMAGE" --argjson dcs "$dcs_json" \
      --arg start "$start" --arg sha "$sha" --argjson gpus "$gpus_json" \
      --arg seed "${FV_SEED:-}" --arg skip "${FV_SKIP:-}" --arg pp "${FV_PARITY_PROMPT:-}" '{
        name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU", gpuCount: 1, gpuTypeIds: $gpus,
        gpuTypePriority: "custom", dataCenterIds: $dcs, ports: ["8000/http"], dockerStartCmd: ["/bin/bash", "-lc", $start],
        containerDiskInGb: 260, allowedCudaVersions: ["13.0"],
        env: ({FV_BUNDLE_URL: env.FVX_URL, FV_BUNDLE_SHA256: $sha, FV_WEIGHTS_MODE: "r2"}
              + (if $seed != "" then {FV_SEED: $seed} else {} end) + (if $skip != "" then {FV_SKIP: $skip} else {} end)
              + (if $pp != "" then {FV_PARITY_PROMPT: $pp} else {} end))
      }')"
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
    files=(log.txt status.tsv nvidia-smi.txt nvsmi.csv fetch.log ref-setup.log ref.log ref/meta.json
           ours-pre.log ours-lightx.log ours-inj.log ours-inj-served.log merged-larry.txt merged-larry.json merged-lightx.txt
           merged-lightx.json compare-dumps-inj.txt compare-dumps-inj-served.txt av-psnr-inj.json av-psnr-inj-served.json
           av-psnr-inj.log av-psnr-inj-served.log ref/clip/output.mp4 ours/inj/output.mp4 ours/inj-served/output.mp4)
    for c in larry6 larry4 lightx4 turbo max base; do
      files+=("perf-$c.log" "perf/$c/benchmark.json" "perf/$c/report.json")
      for p in h3-demo-480p spark-mountain-lake-480p h3-demo-768p spark-mountain-lake-768p; do files+=("perf/$c/frames/$p/output.mp4"); done
    done
    for f in "${files[@]}"; do
      mkdir -p "$out/$(dirname "$f")"
      curl -sS --max-time 300 --fail "$u/$f" -o "$out/$f" 2>/dev/null || rm -f "$out/$f"
    done
    # compare-clips reports: the directory listing names them.
    mkdir -p "$out/compare"
    for f in $(curl -sS --max-time 30 --fail "$u/compare/" 2>/dev/null | grep -o 'href="[^"]*"' | cut -d'"' -f2 | grep -E '\.(out|json)$'); do
      curl -sS --max-time 60 --fail "$u/compare/$f" -o "$out/compare/$f" 2>/dev/null || true
    done
    log "fetched into $out"
    ;;
  down)
    : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
    id="${2:?pod id}"
    rest DELETE "/pods/$id" >/dev/null 2>&1 || true
    if rest GET "/pods/$id" >/dev/null 2>&1; then log "WARNING: pod $id still listed"; exit 1; fi
    log "pod $id deleted (GET /pods/$id: not found)"
    python3 "$HERE/r2-weights.py" --bucket fv-weights-staging rm-temp --prefix tmp/h3-turbo-loras-bundle- || true
    ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
