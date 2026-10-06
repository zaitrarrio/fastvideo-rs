#!/usr/bin/env bash
# WP-18 GPU end-to-end run on Google Cloud (docs/serve/design.md §7.6,
# docs/serve/deploy-gcp.md): one VM per family config, the API matrix
# against each, timings under artifacts/serve/gcp-e2e/<stamp>/, teardown.
#
#   e2e.sh [family...]     default: h3-turbo h3-max ltx-turbo wan-turbo
#
# Per family (VM from scripts/gcp/vm.sh, weights on the zone's Hyperdisk ML):
#   boot     create -> first /ping -> /ping 200 (the startup verifies the
#            weight cells first), serial progress lines
#   probe    /healthz, /fv/v1/capabilities, the forward `info` envelope
#            (GPU, NVENC probe, jobs/artifacts backends)
#   fal      scripts/serve/fal-queue-smoke.sh text-to-video and image-to-video
#   minimax  MiniMax V2 create -> query loop -> download content.url
#   ltx      LTX API v2 async text-to-video + v1 sync (ltx-turbo only)
#   openai   FastVideo /v1/videos create -> retrieve -> content
#   native   /fv/v1/jobs submit -> poll -> content
#   live     the fal director (tests/compat/suites/fal_director.mjs; h3-max,
#            whose app it drives) and Reactor (reactor_sdk_compat.py) pointed
#            at the VM; they need `tests/compat/run.sh --setup` and a client
#            with UDP (or ICE-TCP) reach to the VM; informational
#   encode   NVENC vs x264 on the fal T2V clip, run on the VM (G4 only):
#            wall time, fps, bytes, PSNR/SSIM
# Each check appends a JSON line to <family>/checks.jsonl; results.json per
# family and summary.json per run hold the timings. The serial log is saved
# with admin tokens masked.
#
# Money: FV_GCP_BUDGET_USD (default 40) caps the projected spend (each VM's
# $/hr x its wall-clock cap, plus the Hyperdisk ML hours) before anything is
# created, and the running estimate before each next VM. Every VM carries
# maxRunDuration + DELETE (FV_GCP_CAP_S, default 5400), and the EXIT trap
# deletes every VM and firewall rule this run created (and the Hyperdisk ML
# volume when this run created it).
#
# Env: as vm.sh; FV_GCP_E2E_PARALLEL (default 1 = one VM at a time),
# FV_GCP_E2E_DISK (auto: create the zone's Hyperdisk ML from the fv-weights
# image when missing and delete it at the end; keep: never delete; none),
# FV_GCP_E2E_LIVE (1), FV_GCP_ENCODE_BENCH (1 on G4), FV_GCP_E2E_OUT.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=vm.sh
source "$HERE/vm.sh"

FAMILIES=("$@")
[[ ${#FAMILIES[@]} -gt 0 ]] || FAMILIES=(h3-turbo h3-max ltx-turbo wan-turbo)
BUDGET="${FV_GCP_BUDGET_USD:-40}"
PARALLEL="${FV_GCP_E2E_PARALLEL:-1}"
DISK_MODE="${FV_GCP_E2E_DISK:-auto}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
OUT="${FV_GCP_E2E_OUT:-$FV_ROOT/artifacts/serve/gcp-e2e/$STAMP}"
[[ "$DRY" == 1 && -z "${FV_GCP_E2E_OUT:-}" ]] && OUT="$GCP_OUT/e2e-dry-$STAMP"
PROMPT="${FV_PROMPT:-A red fox trots through fresh snow at dawn, its breath visible in the cold air, cinematic}"
COMPAT_CACHE="${FV_COMPAT_DIR:-${CARGO_TARGET_DIR:-$FV_ROOT/target}/compat}"
mkdir -p "$OUT"
VMS_FILE="$OUT/vms.txt"
: >"$VMS_FILE"
CREATED_DISK=0

family_fal_app() {
  case "$1" in
    h3-turbo) echo minimax/h3-turbo ;; h3-max) echo minimax/h3-max ;;
    ltx-turbo) echo fastvideo/ltx25-distill-sol ;; wan-turbo) echo fastvideo/fastwan21-1.3b ;;
    fake) echo minimax/h3-turbo ;;
  esac
}
family_minimax_model() {
  case "$1" in
    h3-turbo) echo MiniMax-H3 ;; h3-max) echo MiniMax-H3-Max ;;
    ltx-turbo) echo ltx25-distill-sol ;; wan-turbo) echo fastwan21-1.3b ;; fake) echo MiniMax-H3 ;;
  esac
}
family_served() {
  case "$1" in
    h3-turbo) echo fasth3 ;; h3-max) echo sol-h3 ;; ltx-turbo) echo ltx25-distill-sol ;;
    wan-turbo) echo fastwan21-1.3b ;; fake) echo fake-h3-turbo ;;
  esac
}
family_reactor_mode() { case "$1" in wan-turbo) echo video ;; *) echo av ;; esac; }

# ------------------------------------------------------------------ budget
hdml_hr() { awk -v g="${FV_GCP_HDML_GB:-600}" -v t="${FV_GCP_HDML_MIBS:-1200}" -v a="$GCP_HDML_GIB_HR" -v b="$GCP_HDML_MIBS_HR" 'BEGIN{printf "%.4f", g*a + t*b}'; }
projected() {
  local total=0 fam m p dph cap_h waves
  cap_h="$(awk -v s="$CAP_S" 'BEGIN{printf "%.4f", s/3600}')"
  for fam in "${FAMILIES[@]}"; do
    m="$(family_machine "$fam")"; p="$(gcp_provisioning "$m")"; dph="$(gcp_price "$m" "$p")"
    [[ -n "$dph" ]] || die "no price for $m"
    total="$(awk -v t="$total" -v d="$dph" -v h="$cap_h" 'BEGIN{printf "%.2f", t + d*h}')"
  done
  waves=$(( (${#FAMILIES[@]} + PARALLEL - 1) / PARALLEL ))
  awk -v t="$total" -v d="$(hdml_hr)" -v w="$waves" -v h="$cap_h" 'BEGIN{printf "%.2f", t + d*w*h}'
}
spent_estimate() { # from the ledger lines this run wrote
  awk -F'\t' -v run="$STAMP" -v now="$(date +%s)" '
    $2 ~ "^vm-created" && index($2, "e2e=" run) {
      split($2, a, " "); name = a[2]
      for (i in a) if (a[i] ~ /^usd_per_hr=/) { sub("usd_per_hr=", "", a[i]); rate[name] = a[i] }
      cmd = "date -d " $1 " +%s"; cmd | getline t; close(cmd); start[name] = t
    }
    $2 ~ "^vm-deleted" { split($2, a, " "); cmd = "date -d " $1 " +%s"; cmd | getline t; close(cmd); stop[a[2]] = t }
    END { s = 0; for (n in rate) { e = (n in stop) ? stop[n] : now; s += rate[n] * (e - start[n]) / 3600 }; printf "%.2f", s }' "$GCP_LEDGER" 2>/dev/null || echo 0
}

# ------------------------------------------------------------------ cleanup
cleanup() {
  local rc=$? name
  trap - EXIT INT TERM
  while read -r name; do
    [[ -n "$name" ]] || continue
    [[ -f "$OUT/.down-$name" ]] && continue
    log "delete-on-exit: $name"
    vm_down "$name" || log "WARNING: delete of $name failed; its maxRunDuration (${CAP_S}s) deletes it"
  done <"$VMS_FILE"
  if [[ $CREATED_DISK == 1 && "$DISK_MODE" == auto ]]; then
    bash "$HERE/weights.sh" disk-down || log "WARNING: Hyperdisk ML $WEIGHTS_DISK not deleted: run weights.sh disk-down (it bills hourly)"
  fi
  exit "$rc"
}

# ------------------------------------------------------------------ checks
# check <file> <name> <ok 0|1> <json detail>
check() {
  jq -nc --arg c "$2" --argjson ok "$3" --argjson d "${4:-null}" '{check: $c, ok: ($ok == 1)} + ($d // {} | if type == "object" then . else {detail: .} end)' >>"$1"
  log "  $2: $([[ $3 == 1 ]] && echo ok || echo FAIL)"
}
now() { date +%s.%N; }
dt() { awk -v a="$1" -v b="$(now)" 'BEGIN{printf "%.1f", b-a}'; }
probe_mp4() {
  command -v ffprobe >/dev/null || { echo '""'; return; }
  ffprobe -v error -show_entries stream=codec_name,profile,width,height,nb_frames,r_frame_rate,sample_rate,channels -of json "$1" 2>/dev/null \
    | jq -c '[.streams[] | with_entries(select(.value != null))]' || echo '""'
}

# poll <url> <auth header> <jq status expr> <terminal regex> <timeout s>: last body on stdout
poll() {
  local url="$1" hdr="$2" expr="$3" term="$4" limit="$5" t0 body st
  t0="$(date +%s)"
  while :; do
    body="$(curl -sS --max-time 30 -H "$hdr" "$url" || true)"
    st="$(jq -r "$expr" <<<"$body" 2>/dev/null || true)"
    [[ "$st" =~ $term ]] && break
    (( $(date +%s) - t0 < limit )) || break
    sleep 3
  done
  printf '%s' "$body"
}

# download <url> <file> [auth header]: bytes, or 0
download() {
  local code
  code="$(curl -sS -L --max-time 600 -o "$2" -w '%{http_code}' ${3:+-H "$3"} "$1" || echo 000)"
  [[ "$code" == 200 && -s "$2" ]] && stat -c %s "$2" || echo 0
}

run_family_checks() {
  local fam="$1" name="$2" ip="$3" key="$4" dir="$5" base="http://$3:8000" C="$5/checks.jsonl" t0 r st body id url bytes
  local bearer="Authorization: Bearer $key"
  : >"$C"
  # probe
  r="$(curl -sS --max-time 30 "$base/healthz" || true)"; check "$C" healthz "$(jq -e . >/dev/null 2>&1 <<<"$r" && echo 1 || echo 0)" "$(jq -c '{healthz: .}' <<<"$r" 2>/dev/null || echo null)"
  r="$(curl -sS --max-time 30 -H "$bearer" "$base/fv/v1/capabilities" || true)"
  check "$C" capabilities "$(jq -e '.models | length > 0' >/dev/null 2>&1 <<<"$r" && echo 1 || echo 0)" "$(jq -c '{models: [.models[]? | (.id // .)]}' <<<"$r" 2>/dev/null || echo null)"
  r="$(curl -sS --max-time 120 -H "$bearer" -H 'content-type: application/json' -d '{"kind":"info","nvenc":true}' "$base/fv/v1/forward" || true)"
  check "$C" info "$(jq -e '.gpu' >/dev/null 2>&1 <<<"$r" && echo 1 || echo 0)" \
    "$(jq -c '{info: {gpu, ffmpeg_h264_nvenc, nvenc_encode_ok, nvidia_driver_capabilities, jobs_backend, artifacts_backend, webhook_key_configured, ready_after_s}}' <<<"$r" 2>/dev/null || echo null)"

  # fal queue: T2V + I2V
  local app sub line t2v_url=""
  app="$(family_fal_app "$fam")"
  for sub in text-to-video image-to-video; do
    t0="$(now)"
    line="$(FV_KEY="$key" FV_OUT="$dir/media" FV_POLL_S="${FV_POLL_S:-1500}" bash "$FV_ROOT/scripts/serve/fal-queue-smoke.sh" "$base" "$app" "$sub" 2>"$dir/fal-$sub.err" | tail -1 || true)"
    jq -e . >/dev/null 2>&1 <<<"$line" || line="$(jq -nc --arg e "$(tail -c 400 "$dir/fal-$sub.err")" '{error: $e}')"
    check "$C" "fal-$sub" "$(jq -e '.status == "COMPLETED" and .mp4_bytes > 0' >/dev/null <<<"$line" && echo 1 || echo 0)" \
      "$(jq -c --arg w "$(dt "$t0")" '{fal: ., wall_s: ($w|tonumber)}' <<<"$line")"
    [[ "$sub" == text-to-video ]] && t2v_url="$(jq -r '.file // empty' <<<"$line")"
  done

  # MiniMax V2
  t0="$(now)"
  r="$(curl -sS --max-time 60 -H "$bearer" -H 'content-type: application/json' \
    -d "$(jq -nc --arg m "$(family_minimax_model "$fam")" --arg p "$PROMPT" '{model: $m, content: [{type: "text", text: $p}], resolution: "768P", duration: 5, ratio: "16:9"}')" \
    "$base/v2/video_generation" || true)"
  id="$(jq -r '.task_id // empty' <<<"$r" 2>/dev/null)"
  if [[ -n "$id" ]]; then
    body="$(poll "$base/v2/query/video_generation/$id" "$bearer" '.task.status' '^(succeeded|failed)$' "${FV_POLL_S:-1500}")"
    url="$(jq -r '.task.content.url // empty' <<<"$body")"
    bytes=0; [[ -n "$url" ]] && bytes="$(download "$url" "$dir/media/minimax-$id.mp4")"
    check "$C" minimax "$([[ "$(jq -r .task.status <<<"$body")" == succeeded && $bytes -gt 0 ]] && echo 1 || echo 0)" \
      "$(jq -c --arg w "$(dt "$t0")" --argjson b "$bytes" --argjson p "$(probe_mp4 "$dir/media/minimax-$id.mp4")" '{wall_s: ($w|tonumber), status: .task.status, error: .task.error, usage: .task.usage, mp4_bytes: $b, ffprobe: $p}' <<<"$body")"
  else
    check "$C" minimax 0 "$(jq -nc --arg r "$(head -c 400 <<<"$r")" '{submit: $r}')"
  fi

  # LTX API (ltx-turbo only)
  if [[ "$fam" == ltx-turbo ]]; then
    t0="$(now)"
    r="$(curl -sS --max-time 60 -H "$bearer" -H 'content-type: application/json' \
      -d "$(jq -nc --arg p "$PROMPT" '{prompt: $p, model: "ltx-2-5-fast", duration: 6, resolution: "1920x1080"}')" "$base/v2/text-to-video" || true)"
    id="$(jq -r '.id // empty' <<<"$r" 2>/dev/null)"
    if [[ -n "$id" ]]; then
      body="$(poll "$base/v2/text-to-video/$id" "$bearer" '.status' '^(completed|failed)$' "${FV_POLL_S:-1500}")"
      url="$(jq -r '.result.video_url // empty' <<<"$body")"
      bytes=0; [[ -n "$url" ]] && bytes="$(download "$url" "$dir/media/ltx-v2-$id.mp4")"
      check "$C" ltx-v2 "$([[ "$(jq -r .status <<<"$body")" == completed && $bytes -gt 0 ]] && echo 1 || echo 0)" \
        "$(jq -c --arg w "$(dt "$t0")" --argjson b "$bytes" --argjson p "$(probe_mp4 "$dir/media/ltx-v2-$id.mp4")" '{wall_s: ($w|tonumber), status, error, mp4_bytes: $b, ffprobe: $p}' <<<"$body")"
    else
      check "$C" ltx-v2 0 "$(jq -nc --arg r "$(head -c 400 <<<"$r")" '{submit: $r}')"
    fi
    t0="$(now)"
    st="$(curl -sS --max-time 900 -o "$dir/media/ltx-v1.mp4" -w '%{http_code}' -H "$bearer" -H 'content-type: application/json' \
      -d "$(jq -nc --arg p "$PROMPT" '{prompt: $p, model: "ltx-2-5-fast", duration: 6, resolution: "1920x1080"}')" "$base/v1/text-to-video" || echo 000)"
    check "$C" ltx-v1-sync "$([[ "$st" == 200 ]] && echo 1 || echo 0)" \
      "$(jq -nc --arg w "$(dt "$t0")" --arg s "$st" --argjson p "$(probe_mp4 "$dir/media/ltx-v1.mp4")" '{wall_s: ($w|tonumber), http: $s, ffprobe: $p}')"
  fi

  # FastVideo /v1/videos (OpenAI shape)
  t0="$(now)"
  r="$(curl -sS --max-time 60 -H "$bearer" -H 'content-type: application/json' \
    -d "$(jq -nc --arg m "$fam" --arg p "$PROMPT" '{model: $m, prompt: $p}')" "$base/v1/videos" || true)"
  id="$(jq -r '.id // empty' <<<"$r" 2>/dev/null)"
  if [[ -n "$id" ]]; then
    body="$(poll "$base/v1/videos/$id" "$bearer" '.status' '^(completed|failed)$' "${FV_POLL_S:-1500}")"
    bytes="$(download "$base/v1/videos/$id/content" "$dir/media/openai-$id.mp4" "$bearer")"
    check "$C" openai-videos "$([[ "$(jq -r .status <<<"$body")" == completed && $bytes -gt 0 ]] && echo 1 || echo 0)" \
      "$(jq -c --arg w "$(dt "$t0")" --argjson b "$bytes" --argjson p "$(probe_mp4 "$dir/media/openai-$id.mp4")" '{wall_s: ($w|tonumber), status, model, size, seconds, error, mp4_bytes: $b, ffprobe: $p}' <<<"$body")"
  else
    check "$C" openai-videos 0 "$(jq -nc --arg r "$(head -c 400 <<<"$r")" '{submit: $r}')"
  fi

  # native /fv/v1/jobs
  t0="$(now)"
  r="$(curl -sS --max-time 60 -H "$bearer" -H 'content-type: application/json' \
    -d "$(jq -nc --arg m "$(family_served "$fam")" --arg p "$PROMPT" '{model: $m, prompt: $p, seed: 1}')" "$base/fv/v1/jobs" || true)"
  id="$(jq -r '.id // empty' <<<"$r" 2>/dev/null)"
  if [[ -n "$id" ]]; then
    body="$(poll "$base/fv/v1/jobs/$id" "$bearer" '.status' '^(succeeded|failed|cancelled)$' "${FV_POLL_S:-1500}")"
    bytes="$(download "$base/fv/v1/jobs/$id/content" "$dir/media/native-$id.mp4" "$bearer")"
    check "$C" native-jobs "$([[ "$(jq -r .status <<<"$body")" == succeeded && $bytes -gt 0 ]] && echo 1 || echo 0)" \
      "$(jq -c --arg w "$(dt "$t0")" --argjson b "$bytes" --argjson p "$(probe_mp4 "$dir/media/native-$id.mp4")" '{wall_s: ($w|tonumber), status, error, timings: (.timings // .metrics // null), mp4_bytes: $b, ffprobe: $p}' <<<"$body")"
  else
    check "$C" native-jobs 0 "$(jq -nc --arg r "$(head -c 400 <<<"$r")" '{submit: $r}')"
  fi

  # live: director + Reactor (informational)
  if [[ "${FV_GCP_E2E_LIVE:-1}" == 1 ]]; then
    if [[ "$fam" == h3-max && -d "$COMPAT_CACHE/node/node_modules" ]] && command -v node >/dev/null; then
      t0="$(now)"
      if timeout 900 node "$FV_ROOT/tests/compat/suites/fal_director.mjs" "$COMPAT_CACHE/node" "$base" "$key" >"$dir/director.log" 2>&1; then r=1; else r=0; fi
      check "$C" live-director "$r" "$(jq -nc --arg w "$(dt "$t0")" --arg l "$(tail -1 "$dir/director.log" | head -c 1500)" '{wall_s: ($w|tonumber), summary: $l, informational: true}')"
    elif [[ "$fam" == h3-max ]]; then
      check "$C" live-director 0 '{"skipped": "run tests/compat/run.sh --setup (node + playwright) first", "informational": true}'
    fi
    if [[ -x "$COMPAT_CACHE/venv/bin/python" ]]; then
      t0="$(now)"
      if timeout 600 "$COMPAT_CACHE/venv/bin/python" "$FV_ROOT/crates/fastvideo-reactor/tests/compat/reactor_sdk_compat.py" \
        --url "$base" --mode "$(family_reactor_mode "$fam")" >"$dir/reactor.log" 2>&1; then r=1; else r=0; fi
      check "$C" live-reactor "$r" "$(jq -nc --arg w "$(dt "$t0")" --arg l "$(tail -3 "$dir/reactor.log" | head -c 1500)" '{wall_s: ($w|tonumber), tail: $l, informational: true}')"
    else
      check "$C" live-reactor 0 '{"skipped": "run tests/compat/run.sh --setup (reactor_sdk venv) first", "informational": true}'
    fi
  fi

  # NVENC vs x264 on the T2V clip, on the VM (G4 has NVENC).
  if [[ "${FV_GCP_ENCODE_BENCH:-0}" == 1 ]]; then
    local src start=0 chunk bench=""
    src="$(jq -r 'select(.check == "fal-text-to-video") | .fal.video_host // empty' "$C" 2>/dev/null || true)"
    url=""
    [[ -n "$t2v_url" ]] && url="$(jq -r 'select(.check == "fal-text-to-video") | .fal.video_url // empty' "$C" 2>/dev/null || true)"
    # fal-queue-smoke prints only the host: re-read the response for the URL.
    [[ -z "$url" && -n "$src" ]] && url="$(curl -sS --max-time 30 -H "Authorization: Key $key" \
      "$base/$app/requests/$(jq -r 'select(.check == "fal-text-to-video") | .fal.id' "$C")" | jq -r '.video.url // empty' 2>/dev/null || true)"
    if [[ -n "$url" ]] && vm_set_metadata "$name" fv-encode-bench-src "$url"; then
      t0="$(date +%s)"
      while (( $(date +%s) - t0 < 900 )); do
        chunk="$(serial_read "$name" "$start" 2>/dev/null || true)"; start="$SERIAL_NEXT"
        bench="$(grep -m1 'FV-GCP ENCODE-BENCH [\[f]' <<<"$chunk" | sed 's/.*ENCODE-BENCH //' || true)"
        [[ -n "$bench" ]] && break
        sleep 10
      done
      if jq -e 'type == "array"' >/dev/null 2>&1 <<<"$bench"; then
        check "$C" encode-nvenc-vs-x264 1 "$(jq -c '{encode: .}' <<<"$bench")"
      else
        check "$C" encode-nvenc-vs-x264 0 "$(jq -nc --arg b "$bench" '{error: (if $b == "" then "no ENCODE-BENCH line in 900 s" else $b end)}')"
      fi
    else
      check "$C" encode-nvenc-vs-x264 0 '{"error": "no T2V clip URL"}'
    fi
  fi
}

# run_family <family>: VM up, checks, results, VM down. Runs in a subshell.
run_family() {
  local fam="$1" dir="$OUT/$1" t_create timings machine
  mkdir -p "$dir/media"
  machine="$(family_machine "$fam")"
  local bench=0; [[ "$machine" == g4-* ]] && bench="${FV_GCP_ENCODE_BENCH:-1}"
  t_create="$(now)"
  FV_GCP_RUN="$(date -u +%m%d%H%M%S)" FV_GCP_ENCODE_BENCH="$bench" vm_up "$fam" || { jq -nc --arg f "$fam" '{family: $f, error: "vm create failed"}' >"$dir/results.json"; return 1; }
  echo "$VM_NAME" >>"$VMS_FILE"
  # tag the ledger line with this run for the spend estimate
  gcp_ledger "vm-created $VM_NAME usd_per_hr=$VM_DPH e2e=$STAMP family=$fam"
  if [[ "$DRY" == 1 ]]; then
    log "dry run: would wait for $VM_NAME, then run probe fal minimax$([[ $fam == ltx-turbo ]] && echo ' ltx') openai native live$([[ $bench == 1 ]] && echo ' encode') against http://$VM_IP:8000"
    vm_down "$VM_NAME"; touch "$OUT/.down-$VM_NAME"
    jq -nc --arg f "$fam" --arg vm "$VM_NAME" --arg m "$VM_MACHINE" --arg p "$VM_PROV" --arg d "$VM_DPH" '{family: $f, vm: $vm, machine: $m, provisioning: $p, usd_per_hr: ($d|tonumber), dry_run: true}' >"$dir/results.json"
    return 0
  fi
  local rc=0
  if timings="$(vm_wait "$VM_NAME" "$VM_IP")"; then
    [[ "$(secrets_mode)" == metadata ]] && vm_scrub_secrets "$VM_NAME"
    FV_GCP_ENCODE_BENCH="$bench" run_family_checks "$fam" "$VM_NAME" "$VM_IP" "$VM_KEY" "$dir" || rc=1
  else
    timings='{"error": "not ready"}'; rc=1
  fi
  vm_logs "$VM_NAME" >"$dir/serial.log" 2>/dev/null || true
  vm_down "$VM_NAME" && touch "$OUT/.down-$VM_NAME"
  jq -s --arg f "$fam" --arg vm "$VM_NAME" --arg m "$VM_MACHINE" --arg p "$VM_PROV" --arg d "$VM_DPH" --arg z "$ZONE" \
    --arg img "$(grep -m1 'FV-GCP IMAGE' "$dir/serial.log" | sed 's/.*: //' || true)" --arg wall "$(dt "$t_create")" \
    --arg drv "$(grep -m1 'FV-GCP DRIVER' "$dir/serial.log" | sed 's/.*FV-GCP DRIVER //' || true)" --argjson t "$timings" '{
      family: $f, vm: $vm, machine: $m, provisioning: $p, zone: $z, usd_per_hr: ($d|tonumber), image: $img, driver_gpu: $drv,
      boot: $t, vm_wall_s: ($wall|tonumber), est_usd: (($d|tonumber) * ($wall|tonumber) / 3600),
      passed: [.[] | select(.ok) | .check], failed: [.[] | select(.ok|not) | .check], checks: .}' "$dir/checks.jsonl" >"$dir/results.json"
  jq -c '{family, machine, boot, passed, failed, est_usd}' "$dir/results.json" >&2
  return $rc
}

main() {
  require_tools curl jq openssl sha256sum awk
  gcp_require_project
  local fam proj
  for fam in "${FAMILIES[@]}"; do family_config "$fam" >/dev/null || die "unknown family $fam"; done
  proj="$(projected)"
  log "e2e $STAMP: ${FAMILIES[*]} in $ZONE, parallel $PARALLEL, cap ${CAP_S}s/VM; projected worst case \$$proj (budget \$$BUDGET) -> $OUT"
  awk -v p="$proj" -v b="$BUDGET" 'BEGIN{exit !(p+0 <= b+0)}' \
    || die "projected \$$proj > budget \$$BUDGET: lower FV_GCP_CAP_S, run fewer families, or raise FV_GCP_BUDGET_USD"
  trap cleanup EXIT INT TERM
  [[ "$DRY" == 1 ]] || vm_preflight
  if [[ "$DISK_MODE" != none && " ${FAMILIES[*]} " != " fake " ]]; then
    if [[ "$DRY" == 1 ]] || ! gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/disks/$WEIGHTS_DISK" >/dev/null 2>&1; then
      bash "$HERE/weights.sh" disk-up || die "no weight disk"
      CREATED_DISK=1
    fi
  fi
  local pids=() failed=0 running=0
  for fam in "${FAMILIES[@]}"; do
    local spent
    spent="$(spent_estimate)"
    if ! awk -v s="$spent" -v d="$(gcp_price "$(family_machine "$fam")" "$(gcp_provisioning "$(family_machine "$fam")")")" -v h="$(awk -v s="$CAP_S" 'BEGIN{print s/3600}')" -v b="$BUDGET" \
      'BEGIN{exit !(s + d*h <= b)}'; then
      log "budget: spent ~\$$spent; $fam could take the total over \$$BUDGET: skipped"; failed=1; continue
    fi
    ( run_family "$fam" ) &
    pids+=($!); running=$((running + 1))
    if (( running >= PARALLEL )); then wait -n || failed=1; running=$((running - 1)); fi
  done
  while (( running > 0 )); do wait -n || failed=1; running=$((running - 1)); done
  jq -s --arg s "$STAMP" --arg z "$ZONE" --arg spent "$(spent_estimate)" '{run: $s, zone: $z, est_spent_usd: ($spent|tonumber), families: .}' "$OUT"/*/results.json >"$OUT/summary.json" 2>/dev/null || true
  log "summary: $OUT/summary.json"
  jq -r '.families[] | "\(.family)\t\(.machine)\tready \(.boot.ready_s // "-")s\tpass \(.passed // [] | length)\tfail \(.failed // [] | join(","))"' "$OUT/summary.json" 2>/dev/null >&2 || true
  return $failed
}

main
