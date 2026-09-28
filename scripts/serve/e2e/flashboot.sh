#!/usr/bin/env bash
# FlashBoot on/off cold-start cycles on one queue endpoint (docs/serve/images.md
# §FlashBoot). One endpoint from runpod-endpoint.sh `up` (min 0 / max 1,
# idle FV_IDLE_TIMEOUT_S, wall-clock backstop), then per cycle:
#   let the worker idle out (the first cycle is the scale from 0), submit one
#   gen job (wait: true), then an info job on the same worker
#   (process_start_unix, ready_after_s).
# Idle-out: with FlashBoot on, /health keeps counting the stopped worker as
# idle/ready, so the wait is on the v2 worker list instead: the worker is
# down once its uptimeSeconds stops advancing (or it is gone), then
# FV_FB_IDLE_WAIT_S more. FV_FB_ENDPOINT="<ep> <template>" reuses an endpoint.
# FV_FB_ON_CYCLES cycles with FlashBoot on, then the endpoint is PATCHed to
# FlashBoot off and FV_FB_OFF_CYCLES more run as the same-day control.
# Output: artifacts/serve/e2e/flashboot/<name>.jsonl (one line per cycle).
#
# Env: RUNPOD_API_KEY; FV_SERVE_IMAGE (digest or ghcr tag); FV_SERVE_CONFIG
# (default /etc/fv/runpod-wan.toml); FV_FB_MODEL (default fastwan21-1.3b);
# FV_FB_IDLE_WAIT_S (extra wait after the worker went down, default 30);
# FV_FB_INFO_JSON (the info job, default {"kind":"info"}; add "nvenc":true for
# an NVENC probe);
# the runpod-endpoint.sh variables (RUNPOD_GPU_TYPES, RUNPOD_VOLUME_ID,
# FV_ENDPOINT_CAP_S, FV_MIN_BALANCE, FV_IDLE_TIMEOUT_S).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EP_SH="$HERE/../runpod-endpoint.sh"
# shellcheck source-path=SCRIPTDIR source=../../gpu/lib.sh
source "$HERE/../../gpu/lib.sh"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
export FV_SERVE_CONFIG="${FV_SERVE_CONFIG:-/etc/fv/runpod-wan.toml}"
export FV_ENDPOINT_PREFIX="${FV_ENDPOINT_PREFIX:-fv-flashboot}"
export FV_FLASHBOOT=1
MODEL="${FV_FB_MODEL:-fastwan21-1.3b}"
ON_CYCLES="${FV_FB_ON_CYCLES:-3}"
OFF_CYCLES="${FV_FB_OFF_CYCLES:-2}"
IDLE_WAIT="${FV_FB_IDLE_WAIT_S:-30}"
MIN_START="${FV_FB_MIN_BALANCE:-14}"
QUEUE="${RUNPOD_QUEUE_API:-https://api.runpod.ai/v2}"
REST="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
OUT="$FV_ROOT/artifacts/serve/e2e/flashboot"
NAME="${FV_FB_NAME:-wan-turbo-us-$(date -u +%m%d%H%M)}"
mkdir -p "$OUT"
JSONL="$OUT/$NAME.jsonl"

balance() {
  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' "$GQL" \
    -d '{"query":"{ myself { clientBalance } }"}' | jq -r '.data.myself.clientBalance'
}
guard() {
  local b; b="$(balance)"
  log "balance \$$b (stop under \$$MIN_START)"
  awk -v b="$b" -v m="$MIN_START" 'BEGIN{exit !(b+0 >= m+0)}' || die "balance \$$b under \$$MIN_START: stopping"
}
health() { curl -sS --max-time 30 -H "Authorization: Bearer $RUNPOD_API_KEY" "$QUEUE/$EP/health" || echo '{}'; }
# Sum of uptimeSeconds over the endpoint's workers (v2 list); frozen = down.
uptime_sum() {
  curl -sS --max-time 30 -H "Authorization: Bearer $RUNPOD_API_KEY" "https://api.runpod.io/v2/serverless/$EP/workers" \
    | jq -r '[.workers[]? | .uptimeSeconds // 0] | add // 0' 2>/dev/null || echo "?"
}

EP=""; TPL=""
cleanup() {
  local rc=$?
  [[ -n "$EP" ]] && bash "$EP_SH" down "$EP" "$TPL" || true
  exit "$rc"
}
trap cleanup EXIT INT TERM

guard
if [[ -n "${FV_FB_ENDPOINT:-}" ]]; then
  read -r EP TPL <<<"$FV_FB_ENDPOINT"
else
  read -r EP TPL < <(bash "$EP_SH" up "${FV_SERVE_IMAGE:-}" | tail -1)
fi
[[ -n "$EP" ]] || die "no endpoint"
log "endpoint $EP template $TPL flashboot=on"
echo "$EP $TPL" >"$OUT/$NAME.ids"
FIRST="${FV_FB_FIRST_CYCLE:-1}"

cycle() {
  local n="$1" fb="$2" h t0 gen info
  # Idle-out: wait until no worker of any state, then IDLE_WAIT more.
  if (( n > FIRST || FIRST > 1 )); then
    local t_idle prev="" cur same=0; t_idle="$(date +%s)"
    while :; do
      cur="$(uptime_sum)"
      if [[ "$cur" == "$prev" ]]; then same=$((same + 1)); else same=0; fi
      prev="$cur"
      (( same >= 4 )) && break   # unchanged for 4 samples (~60 s: the list refreshes slowly)
      (( $(date +%s) - t_idle < 900 )) || die "worker never idled out (uptime sum $cur)"
      sleep 15
    done
    log "cycle $n: worker down (uptime frozen at ${cur}s) after $(( $(date +%s) - t_idle ))s; waiting ${IDLE_WAIT}s"
    sleep "$IDLE_WAIT"
  fi
  guard
  h="$(health)"
  t0="$(date +%s.%N)"
  gen="$(bash "$EP_SH" job "$EP" "{\"kind\":\"http\",\"method\":\"POST\",\"path\":\"/fv/v1/jobs\",\"body\":{\"model\":\"$MODEL\",\"prompt\":\"a red fox trotting through fresh snow\",\"seed\":$n},\"wait\":true}" | tail -1)"
  info="$(bash "$EP_SH" job "$EP" "${FV_FB_INFO_JSON:-{\"kind\":\"info\"\}}" | tail -1)"
  jq -cn --argjson n "$n" --arg fb "$fb" --arg t0 "$t0" --argjson h "$h" --argjson gen "$gen" --argjson info "$info" '{
    cycle: $n, flashboot: $fb, submit_unix: ($t0|tonumber), health_before: $h,
    gen: ($gen | {status, delayTime, executionTime, workerId, client_wall_s, first_in_progress_s,
                  job_status: .output.body.status, bytes: .output.body.output.bytes, error}),
    info: ($info | {workerId, client_wall_s, gpu: .output.gpu, process_start_unix: .output.process_start_unix,
                    ready_after_s: .output.ready_after_s, uptime_s: .output.uptime_s,
                    nvenc_encode_ok: .output.nvenc_encode_ok, ffmpeg_h264_nvenc: .output.ffmpeg_h264_nvenc}),
    process_start_after_submit_s: (($info.output.process_start_unix // 0) - ($t0|tonumber)),
    ready_after_submit_s: (($info.output.process_start_unix // 0) + ($info.output.ready_after_s // 0) - ($t0|tonumber))
  }' | tee -a "$JSONL"
}

n=$((FIRST - 1))
for _ in $(seq "$FIRST" "$ON_CYCLES"); do n=$((n + 1)); cycle "$n" on; done
if (( OFF_CYCLES > 0 )); then
  curl -sS --fail-with-body -X PATCH -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' \
    -d '{"flashboot":false}' "$REST/endpoints/$EP" | jq -c '{id, flashboot}'
  for _ in $(seq 1 "$OFF_CYCLES"); do n=$((n + 1)); cycle "$n" off; done
fi
log "wrote $JSONL"
