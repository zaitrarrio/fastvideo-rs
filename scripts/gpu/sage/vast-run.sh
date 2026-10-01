#!/usr/bin/env bash
# Run the SageAttention batch (run.sh) on one rented Vast GPU without ssh:
# the scripts and our kernel sources travel as base64 chunks in the instance
# env, run.sh is the container command (runtype "args"), and its stdout comes
# back through the Vast logs API.
#
#   vast-run.sh pack                         build the payload, print its size
#   vast-run.sh up OFFER LABEL CAP_S [K=V..]  rent OFFER with the payload; a detached
#                                           backstop destroys it after CAP_S seconds
#   vast-run.sh status ID                   actual_status, $/hr, uptime
#   vast-run.sh logs ID [TAIL]              the container log (last TAIL lines)
#   vast-run.sh down ID                     destroy and confirm it is gone
#
# Needs VAST_API_KEY (never printed). Ledger: $SAGE_LEDGER.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
V="${VAST_API_BASE:-https://console.vast.ai/api/v0}"
IMAGE="${SAGE_IMAGE:-pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel}"
DISK="${SAGE_DISK_GB:-80}"
WORK="${SAGE_WORK:-${TMPDIR:-/tmp}/fv-sage-pay}"
LEDGER="${SAGE_LEDGER:-$WORK/ledger.tsv}"
CHUNK=8000

api() {
  curl -sS --fail-with-body -X "$1" -H "Authorization: Bearer $VAST_API_KEY" \
    -H 'content-type: application/json' ${3:+-d @"$3"} "$V$2"
}
ledger() { mkdir -p "$(dirname "$LEDGER")"; printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }

# kernels.cu is 280 KB; the harness needs only flash_mma_fwd2_d128 and its
# helpers (line ranges checked against the markers below).
fwd2_subset() {
  local k="$ROOT/crates/fastvideo-cudarc/src/wan/kernels.cu"
  sed -n 3356p "$k" | grep -q 'flash_mma_fwd2_body(' || { echo "kernels.cu moved: update fwd2_subset" >&2; exit 1; }
  sed -n 3688,3689p "$k" | grep -q 'flash_mma_fwd2_d128(' || { echo "kernels.cu moved: update fwd2_subset" >&2; exit 1; }
  echo "// Subset of crates/fastvideo-cudarc/src/wan/kernels.cu: flash_mma_fwd2_d128 only."
  sed -n 1,15p "$k"; sed -n 572,574p "$k"; sed -n 826p "$k"; sed -n 926,1030p "$k"
  sed -n 1194,1261p "$k"; echo '#endif'; sed -n 3333,3485p "$k"; sed -n 3670,3673p "$k"
  printf '#else\n#define FA2_ENTRY_BODY(D) __trap();\n#endif\n'; sed -n 3688,3694p "$k"; echo '}'
}

cmd_pack() {
  rm -rf "$WORK/pay" && mkdir -p "$WORK/pay"
  cp "$HERE"/{bench.py,capture.py,setup.sh,run.sh} "$WORK/pay/"
  [[ -d "$HERE/port" ]] && cp -r "$HERE/port" "$WORK/pay/"
  cp "$ROOT/crates/fastvideo-cudarc/src/wan/attn_fp8.cu" "$WORK/pay/"
  [[ -f "$ROOT/crates/fastvideo-cudarc/src/wan/attn_sage.cu" ]] && cp "$ROOT/crates/fastvideo-cudarc/src/wan/attn_sage.cu" "$WORK/pay/"
  fwd2_subset >"$WORK/pay/kernels.cu"
  tar -C "$WORK/pay" -czf "$WORK/pay.tgz" .
  base64 -w0 "$WORK/pay.tgz" >"$WORK/pay.b64"
  echo "payload $(wc -c <"$WORK/pay.tgz") B gz, $(wc -c <"$WORK/pay.b64") B b64, $(( ($(wc -c <"$WORK/pay.b64") + CHUNK - 1) / CHUNK )) chunks"
}

cmd_up() {
  local offer="$1" label="$2" cap="$3"; shift 3
  [[ "$label" == fv-sage-* ]] || { echo "label must start fv-sage-" >&2; exit 2; }
  cmd_pack >&2
  local n env script body resp id
  n=$(( ($(wc -c <"$WORK/pay.b64") + CHUNK - 1) / CHUNK ))
  env="$(jq -Rn --argjson c $CHUNK --arg n "$n" '
    (input) as $b | reduce range(0; ($n|tonumber)) as $i ({FVP_N: $n};
      . + {("FVP_\($i)"): $b[($i*$c):(($i+1)*$c)]})' <"$WORK/pay.b64")"
  for kv in "$@"; do env="$(jq --arg k "${kv%%=*}" --arg v "${kv#*=}" '. + {($k): $v}' <<<"$env")"; done
  # shellcheck disable=SC2016 # expanded in the container
  script='mkdir -p /root/w && cd /root/w && for i in $(seq 0 $((FVP_N-1))); do v=FVP_$i; printf %s "${!v}"; done | base64 -d | tar xz && echo "S payload ok" && timeout "${SAGE_TIMEOUT:-4800}" bash run.sh; echo "S run.sh exit $?"; sleep "${SAGE_LINGER:-900}"'
  body="$WORK/create.json"
  jq -n --arg image "$IMAGE" --arg label "$label" --argjson disk "$DISK" --arg s "$script" --argjson env "$env" '{
    client_id: "me", image: $image, label: $label, disk: $disk, runtype: "args",
    args: ["bash", "-c", $s], env: $env, target_state: "running", cancel_unavail: true }' >"$body"
  resp="$(api PUT "/asks/$offer/" "$body")" || { echo "create failed: $(head -c 400 <<<"$resp")" >&2; exit 1; }
  id="$(jq -r '.new_contract // empty' <<<"$resp")"
  [[ -n "$id" ]] || { echo "create failed: $(head -c 400 <<<"$resp")" >&2; exit 1; }
  ledger "created $id offer=$offer label=$label image=$IMAGE cap=${cap}s"
  # shellcheck disable=SC2016 # expanded by the child shell
  nohup env IID="$id" CAP="$cap" API="$V" LEDGER="$LEDGER" bash -c \
    'sleep "$CAP"; curl -sS -X DELETE -H "Authorization: Bearer $VAST_API_KEY" "$API/instances/$IID/" >/dev/null && printf "%s\tbackstop-destroyed %s\n" "$(date -u +%FT%TZ)" "$IID" >>"$LEDGER"' \
    >/dev/null 2>&1 &
  echo "$!" >"$WORK/backstop-$id.pid"
  echo "$id"
}

cmd_status() {
  api GET "/instances/$1/" | jq -c '(.instances // .) | {id, label, actual_status, cur_state, dph_total: (.dph_total*1000|round/1000), gpu_name, start_date, status_msg: (.status_msg // "" | .[0:160])}'
}

cmd_logs() {
  local id="$1" tail="${2:-400}" url i out
  url="$(curl -sS --fail-with-body -X PUT -H "Authorization: Bearer $VAST_API_KEY" -H 'content-type: application/json' \
    -d "{\"tail\": \"$tail\"}" "$V/instances/request_logs/$id/" | jq -r '.result_url // empty')"
  [[ -n "$url" ]] || { echo "no log url" >&2; return 1; }
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if out="$(curl -sS --fail "$url" 2>/dev/null)"; then printf '%s\n' "$out"; return 0; fi
    sleep 3
  done
  echo "log not ready" >&2; return 1
}

cmd_down() {
  local id="$1" st
  api DELETE "/instances/$id/" >/dev/null || true
  sleep 5
  st="$(api GET "/instances/$id/" 2>/dev/null | jq -r '(.instances // .) | .actual_status // "gone"' 2>/dev/null || echo gone)"
  ledger "destroy $id -> ${st:-gone}"
  [[ -f "$WORK/backstop-$id.pid" ]] && kill "$(cat "$WORK/backstop-$id.pid")" 2>/dev/null || true
  echo "instance $id: ${st:-gone}"
}

case "${1:-}" in
  pack) cmd_pack ;;
  up) shift; cmd_up "$@" ;;
  status) cmd_status "$2" ;;
  logs) cmd_logs "$2" "${3:-400}" ;;
  down) cmd_down "$2" ;;
  *) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
