#!/bin/bash
# Causal director GPU smoke on the pod (docs/serve/director-causal.md):
# fv-serve (this branch) with LongLive-1.3B via [[models]] longlive = ...,
# HTTP and WebRTC on loopback only, driven by director_client on the pod.
#   bash dc.sh setup|serve|client|report
set -u
D=/e2e/dc; WK=/root/dc; W=/workspace/weights
mkdir -p $WK
log() { echo "$(date -u +%FT%TZ) | $*" | tee -a $WK/progress.log; }
case "${1:?stage}" in
setup)
  [ -s /e2e/serve.pid ] && kill -TERM "$(cat /e2e/serve.pid)" 2>/dev/null; sleep 2
  ls -la $W/sfwan21-1.3b $W/longlive-1.3b-safetensors $W/auxiliary/tae | head -40
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
  python3 - <<'PY' > $WK/prompts.txt
import json
for l in open('/workspace/weights/longlive-1.3b/prompts/interactive_example.jsonl'):
    l=l.strip()
    if not l: continue
    v=json.loads(l)
    ps = v.get('prompts') or v.get('prompt_list') or [v.get('prompt') or v.get('text')]
    for p in ps: print(p.replace('\n',' '))
    break
PY
  wc -l $WK/prompts.txt; cut -c1-120 $WK/prompts.txt
  cat > $WK/fv.toml <<TOML
[server]
bind = "127.0.0.1:8100"
state_dir = "/root/dc/state"
[auth]
mode = "none"
[engine]
backend = "cuda"
post_encoder = "auto"
[[models]]
id = "longlive-1.3b"
family = "wan"
recipe = "sfwan21-1.3b"
weights = "/workspace/weights/sfwan21-1.3b"
resident = true
longlive = "/workspace/weights/longlive-1.3b-safetensors"
[protocols]
openai_videos = false
fastwan = false
minimax = false
fal = true
fal_director = true
ltx = false
reactor = false
native = false
fal_apps = ["fastvideo/longlive"]
[director]
encoder = "auto"
[webrtc]
udp_port = 40110
tcp_port = 40100
public_ip = "127.0.0.1"
TOML
  sed -e 's/^id = "longlive-1.3b"/id = "ltx25-distill-sol"/' -e 's/^family = "wan"/family = "ltx2"/' -e 's/^recipe = "sfwan21-1.3b"/recipe = "ltx-turbo"/' \
      -e 's#^weights = .*#weights = "/workspace/weights/ltx25"#' -e '/^longlive = /d' -e 's#fastvideo/longlive#fastvideo/ltx-turbo#' $WK/fv.toml > $WK/fv-ltx.toml
  grep -E "^(id|family|recipe|weights|fal_apps)" $WK/fv-ltx.toml
  ls $W/ltx25 | head
  log "setup done"
  ;;
serve)
  [ -s $WK/serve.pid ] && kill -TERM "$(cat $WK/serve.pid)" 2>/dev/null && sleep 3
  chmod +x $D/fv-serve $D/director_client
  env -i PATH="$PATH" HOME=/root LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}" \
    NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}" RUST_LOG=info FV_JOB_STORE=memory \
    FV_WEIGHTS=$W FV_TAE_DIR=$W/auxiliary/tae FASTVIDEO_TAE_DIR=$W/auxiliary/tae FV_URL_SIGNING_KEY=dc-smoke \
    $D/fv-serve --config $WK/${CFG:-fv.toml} > $WK/serve-${CFG:-fv.toml}.log 2>&1 &
  echo $! > $WK/serve.pid
  t0=$(date +%s)
  for i in $(seq 1 ${READY_TRIES:-120}); do
    c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8100/ping || true)
    [ "$c" = 200 ] && { log "fv-serve ready in $(( $(date +%s) - t0 ))s"; exit 0; }
    kill -0 "$(cat $WK/serve.pid)" 2>/dev/null || { log "fv-serve exited"; tail -30 $WK/serve-${CFG:-fv.toml}.log; exit 1; }
    sleep 5
  done
  log "fv-serve not ready"; tail -30 $WK/serve-${CFG:-fv.toml}.log; exit 1
  ;;
client)
  R=$WK/${2:-run1}; mkdir -p $R
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -lms 1000 > $R/smi.csv 2>/dev/null &
  smi=$!
  timeout 500 $D/director_client --url http://127.0.0.1:8100 --app ${APP:-fastvideo/longlive} --seconds ${SECS:-60} \
    --switch-at "${SWITCH-15,30,45}" --prompts ${PROMPTS:-$WK/prompts.txt} --seed 7 ${RES:+--resolution $RES} --out $R > $R/client.out 2> $R/client.err
  rc=$?; kill $smi 2>/dev/null
  log "client $(basename $R) rc=$rc"
  tail -25 $R/client.err
  ;;
report)
  R=$WK/${2:-run1}
  grep -E "prompt switch|KV re-cache|deadline|director|ERROR|WARN" $WK/serve-${CFG:-fv.toml}.log | sed 's/\x1b\[[0-9;]*m//g' | tail -80 > $R/serve-grep.txt
  cat $R/serve-grep.txt | cut -c1-400
  ffprobe -v error -count_frames -show_entries stream=codec_name,width,height,nb_read_frames -of csv $R/video.h264 2>&1 | tail -2
  ffmpeg -v error -y -i $R/video.h264 -vf "fps=1,scale=208:120,tile=10x7" -frames:v 1 $R/sheet.jpg && log "sheet ok"
  ffmpeg -v error -y -i $R/video.h264 -vf "select='eq(n\,0)+eq(n\,232)+eq(n\,248)+eq(n\,264)+eq(n\,472)+eq(n\,488)+eq(n\,504)+eq(n\,712)+eq(n\,728)+eq(n\,744)',scale=416:240,tile=3x4" -vsync vfr -frames:v 1 $R/switches.jpg && log "switch sheet ok"
  cut -d, -f1 $R/smi.csv | sort -n | tail -1
  python3 -c "import json;s=json.load(open('$R/summary.json'));s.pop('session_info',None);print(json.dumps(s,indent=1)[:6000])"
  if [ -s $R/audio.opus ] && [ "$(stat -c %s $R/audio.opus)" -gt 2000 ]; then
    ffmpeg -v error -y -framerate ${FPS:-16} -i $R/video.h264 -i $R/audio.opus -c:v libx264 -crf 23 -preset veryfast -pix_fmt yuv420p -c:a aac -b:a 128k -shortest $R/session.mp4
  else
    ffmpeg -v error -y -framerate ${FPS:-16} -i $R/video.h264 -c:v libx264 -crf 23 -preset veryfast -pix_fmt yuv420p $R/session.mp4
  fi
  ls -la $R/session.mp4 && log "mp4 $(basename $R) $(stat -c %s $R/session.mp4) B"
  ;;
part)
  # part <file> <n>: base64 of the n-th 4 MiB piece
  dd if="$2" bs=4M skip="$3" count=1 status=none | base64 -w0
  ;;
esac
