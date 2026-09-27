#!/usr/bin/env bash
# One job through fv-serve's fal queue API (design §4.4) on a running server:
# submit, poll the status, fetch the response, download the MP4.
#
#   fal-queue-smoke.sh <base url> <app> <sub> [json body]
#
#   <app>   e.g. minimax/h3-turbo, minimax/h3-max
#   <sub>   text-to-video | image-to-video
#   body    default {"prompt": FV_PROMPT, "seed": 1}; image-to-video adds
#           image_url (FV_IMAGE_URL, default the repo's beach fixture)
#
# Env: FV_KEY (the API key; sent as `Authorization: Key <k>`), FV_PROMPT,
# FV_IMAGE_URL, FV_OUT (default artifacts/serve/fal-smoke), FV_POLL_S (900).
# Prints one JSON line: request id, status, wall seconds, fal timings, the
# video URL host, the MP4 bytes and its ffprobe summary when ffprobe exists.
set -euo pipefail
BASE="${1:?base url}"; APP="${2:?app}"; SUB="${3:?sub}"
: "${FV_KEY:?FV_KEY missing}"
PROMPT="${FV_PROMPT:-A red fox trots through fresh snow at dawn, its breath visible in the cold air, cinematic}"
IMG="${FV_IMAGE_URL:-https://raw.githubusercontent.com/zaitrarrio/fastvideo-rs/main/scripts/gpu/fixtures/ti2v-beach-832x480.jpg}"
OUT="${FV_OUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/artifacts/serve/fal-smoke}"
mkdir -p "$OUT"
if [[ -n "${4:-}" ]]; then
  body="$4"
elif [[ "$SUB" == image-to-video ]]; then
  body="$(jq -nc --arg p "$PROMPT" --arg i "$IMG" '{prompt: $p, image_url: $i, seed: 1}')"
else
  body="$(jq -nc --arg p "$PROMPT" '{prompt: $p, seed: 1}')"
fi
auth=(-H "Authorization: Key $FV_KEY")
t0=$(date +%s.%N)
sub="$(curl -sS --max-time 120 "${auth[@]}" -H 'content-type: application/json' -d "$body" "$BASE/$APP/$SUB")"
id="$(jq -r '.request_id // empty' <<<"$sub")"
[[ -n "$id" ]] || { echo "submit failed: $(head -c 400 <<<"$sub")" >&2; exit 1; }
st=""
deadline=$(( $(date +%s) + ${FV_POLL_S:-900} ))
while (( $(date +%s) < deadline )); do
  st="$(curl -sS --max-time 30 "${auth[@]}" "$BASE/$APP/requests/$id/status?logs=0" || true)"
  case "$(jq -r '.status // empty' <<<"$st" 2>/dev/null)" in COMPLETED) break ;; esac
  sleep 3
done
t1=$(date +%s.%N)
resp="$(curl -sS --max-time 60 "${auth[@]}" "$BASE/$APP/requests/$id")"
url="$(jq -r '.video.url // empty' <<<"$resp" 2>/dev/null || true)"
bytes=0 probe=""
file="$OUT/${APP//\//_}-$SUB-$id.mp4"
if [[ -n "$url" ]]; then
  curl -sS --max-time 300 -o "$file" "$url"
  bytes=$(stat -c %s "$file")
  if command -v ffprobe >/dev/null; then
    probe="$(ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames,r_frame_rate,sample_rate -of compact=p=0:nk=0 "$file" | tr '\n' ' ')"
  fi
fi
jq -nc --arg id "$id" --arg app "$APP" --arg sub "$SUB" --argjson st "${st:-null}" --arg wall "$(awk -v a="$t1" -v b="$t0" 'BEGIN{printf "%.1f", a-b}')" \
  --arg resp "$(head -c 2000 <<<"$resp")" --arg host "$(sed -E 's#^(https?://[^/?]+).*#\1#' <<<"$url")" \
  --arg bytes "$bytes" --arg probe "$probe" --arg file "$file" '
  {id: $id, app: $app, sub: $sub, status: ($st.status // null), wall_s: ($wall|tonumber),
   timings: (($resp | fromjson? // {}) | .timings // null), error: (($resp | fromjson? // {}) | .detail // null),
   video_host: $host, mp4_bytes: ($bytes|tonumber), ffprobe: $probe, file: $file}'
[[ "$bytes" -gt 0 ]]
