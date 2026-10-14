#!/bin/bash
# GPU check 2026-10-13, on-pod driver (run through the e2e sidecar from /e2e/gc).
#   gc.sh prep            weights layout, R2 fetch (bg), mediamtx, nvidia-smi sampler
#   gc.sh ups             start the upsampler sidecar (fv-upsampler run) -> /e2e/gc/ups.env
#   gc.sh serve <cfg> [K=V ...]   (re)start fv-serve on /e2e/gc/<cfg>.toml, wait for /ping 200
#   gc.sh job <tag> <model> <w> <h> [frames]     native job, metrics -> out/jobs/<tag>.json
#   gc.sh director <name> <configure.json> <switch-chunks> <stop-after> [app]
#   gc.sh stream <kind>   SF-Wan native stream to MediaMTX (whip|rtmp|srt), 30 s, read back
#   gc.sh status          fetch progress, GPU, serve log tail
set -u
G=/e2e/gc
O=/e2e/out
W=/root/w
mkdir -p $O $O/jobs $O/director $O/streams
KEYH="Authorization: Bearer ${FV_KEY:-}"
log() { echo "[gc $(date -u +%H:%M:%S)] $*"; }

common() { # <state subdir>
  cat <<EOF
[server]
bind = "0.0.0.0:8000"
state_dir = "/fvstate/$1"
shutdown_grace_s = 10

[auth]
mode = "keys"

[artifacts]
backend = "local"
url_ttl_s = 86400

[jobs]
backend = "file"
progress_interval_ms = 1000
stale_after_s = 900

[engine]
backend = "cuda"
post_encoder = "auto"

[webrtc]
udp_port = 40010
tcp_port = 40000
public_ip = "127.0.0.1"

[outputs]
allow_private = true

[director]
allow_output = true

[limits]
queue_max = 16
body_max_mb = 64
EOF
}

upsampler_toml() {
  cat <<EOF

[upsampler]
base_url = "http://127.0.0.1:18080/v1"
model = "qwen3-vl-4b-instruct"
timeout_ms = 30000
hedge = false
vision = true
context_tokens = 28000

[upsampler.tokenizers]
h3 = "$W/h3-base/tokenizer/tokenizer.json"
EOF
}

write_configs() {
  { common trim; upsampler_toml; cat <<EOF

[[models]]
id = "fasth3-trim-8step"
family = "h3"
recipe = "h3-trim"
weights = "$W/fasth3-trim-8step"
resident = true
served_names = ["h3-trim", "fasth3-trim"]
warmup = true

[protocols]
openai_videos = false
fastwan = false
minimax = false
fal = true
fal_director = true
ltx = false
reactor = false
native = true
fal_apps = ["fastvideo/h3-trim"]
EOF
  } > $G/trim.toml
  { common turbo; upsampler_toml; cat <<EOF

[[models]]
id = "fasth3"
family = "h3"
recipe = "h3-turbo"
weights = "$W/h3-base"
resident = true
served_names = ["fasth3"]
continuity = "anchor-last-frame"
i2v_encoder = "auto"
warmup = true

[protocols]
openai_videos = false
fastwan = false
minimax = false
fal = true
fal_director = true
ltx = false
reactor = false
native = true
fal_apps = ["minimax/h3-turbo"]
EOF
  } > $G/turbo.toml
  { common sfwan; cat <<EOF

[[models]]
id = "sfwan21-1.3b"
family = "wan"
recipe = "sfwan21-1.3b"
weights = "$W/sfwan21-1.3b"
resident = true

[protocols]
openai_videos = false
fastwan = false
minimax = false
fal = false
fal_director = false
ltx = false
reactor = false
native = true

[streams]
causal_default_max_s = 60
causal_hard_max_s = 120
EOF
  } > $G/sfwan.toml
}

case "${1:-}" in
  prep)
    write_configs
    mkdir -p $W
    # The EU volume's trees, read only (never written): symlinks.
    for t in h3-base auxiliary; do [ -e /workspace/weights/$t ] && ln -sfn /workspace/weights/$t $W/$t; done
    ls -la $W /workspace/weights | head -40
    df -h / /root /workspace 2>/dev/null
    # nvidia-smi sampler (1 s): total used memory + per-process.
    nohup bash -c "nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits -l 1 > $O/smi.csv" >/dev/null 2>&1 &
    # R2 fetch: Qwen first (the sidecar), then Trim, then SF-Wan.
    nohup python3 /opt/fastvideo-rs/bin/r2-fetch.py --root $W --plan $G/plan.json --status $O/fetch-status.json --concurrency 32 \
      > $O/fetch.log 2>&1 &
    # MediaMTX v1.15.1 (the version the outputs_live test pins).
    (cd $G && curl -fsSL -o mtx.tgz https://github.com/bluenviron/mediamtx/releases/download/v1.15.1/mediamtx_v1.15.1_linux_amd64.tar.gz && tar xzf mtx.tgz mediamtx && rm mtx.tgz) && log "mediamtx $($G/mediamtx --version 2>&1 | head -1)"
    printf 'paths:\n  all_others:\n' > $G/mediamtx.yml
    openssl req -x509 -newkey rsa:2048 -nodes -days 2 -subj /CN=127.0.0.1 -keyout $G/server.key -out $G/server.crt >/dev/null 2>&1
    MTX_API=yes MTX_APIADDRESS=127.0.0.1:9997 MTX_RTSPADDRESS=127.0.0.1:8554 MTX_RTSPTRANSPORTS=tcp MTX_RTMPADDRESS=127.0.0.1:1935 \
    MTX_RTMPENCRYPTION=no MTX_HLS=no MTX_SRTADDRESS=127.0.0.1:8890 MTX_WEBRTCADDRESS=127.0.0.1:8889 MTX_WEBRTCLOCALUDPADDRESS=127.0.0.1:8189 \
    MTX_WEBRTCADDITIONALHOSTS=127.0.0.1 MTX_PATHDEFAULTS_SRTPUBLISHPASSPHRASE=fvgpucheck1013pass \
      nohup $G/mediamtx $G/mediamtx.yml > $O/mediamtx.log 2>&1 &
    sleep 2; curl -s http://127.0.0.1:9997/v3/paths/list | head -c 200; echo
    chmod +x $G/director_client
    log "prep done"
    ;;
  ups)
    t0=$(date +%s.%N)
    until [ -f $W/qwen3-vl-4b-instruct-gguf/.complete ]; do sleep 2; done
    log "qwen tree complete after $(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b-a}')s"
    FV_UPSAMPLER_ROOT=$W FV_UPSAMPLER_STATE=/fvstate FV_UPSAMPLER_LOG=$O/upsampler.log \
      /opt/fastvideo-rs/bin/fv-upsampler run > $G/ups.env 2>> $O/upsampler.log
    cat $G/ups.env; cat /fvstate/upsampler.steps 2>/dev/null; tail -3 $O/upsampler.log
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    ;;
  serve)
    cfg=$2; shift 2
    case $cfg in trim) tree=fasth3-trim-8step ;; sfwan) tree=sfwan21-1.3b ;; *) tree= ;; esac
    if [ -n "$tree" ]; then until [ -f $W/$tree/.complete ]; do sleep 2; done; fi
    # shellcheck disable=SC1090
    [ -s $G/ups.env ] && . $G/ups.env
    t0=$(date +%s)
    env FV_WEIGHTS=$W FV_ARTIFACTS_DIR=/fvstate/artifacts FV_STATE_DIR=/fvstate/$cfg "$@" /e2e/serve.sh $G/$cfg.toml
    for i in $(seq 1 360); do
      c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/ping)
      [ "$c" = 200 ] && break
      sleep 1
    done
    log "serve $cfg ($*): /ping $c after $(( $(date +%s) - t0 ))s"
    grep -E "upsampler|tokenizer|warm|ERROR|error|nvenc|encoder" /e2e/serve.log | tail -12
    ;;
  job)
    tag=$2 model=$3 w=$4 h=$5 frames=${6:-124}
    body=$(jq -nc --arg m "$model" --argjson w $w --argjson h $h --argjson f $frames '{model:$m, prompt:"A man in his thirties talking to the camera in a sunlit workshop, warm light, he smiles and says <d>Today we build something small and beautiful.</d> Sound: soft room tone, a distant radio.", seed:1024, width:$w, height:$h, num_frames:$f}')
    t0=$(date +%s.%N)
    r=$(curl -s -H "$KEYH" -H 'content-type: application/json' -d "$body" http://127.0.0.1:8000/fv/v1/jobs)
    id=$(jq -r '.id // empty' <<<"$r")
    [ -n "$id" ] || { log "$tag: submit failed: $(head -c 300 <<<"$r")"; exit 1; }
    while :; do
      j=$(curl -s -H "$KEYH" http://127.0.0.1:8000/fv/v1/jobs/$id)
      s=$(jq -r .status <<<"$j")
      case $s in succeeded|completed|failed|cancelled|canceled) break ;; esac
      sleep 0.5
    done
    wall=$(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.2f", b-a}')
    jq --arg tag "$tag" --argjson wall $wall '. + {gc_tag:$tag, gc_wall_s:$wall}' <<<"$j" > $O/jobs/$tag.json
    grep "h3 step" /e2e/serve.log | tail -8 | sed 's/.*h3 step/h3 step/' | tr '\n' ' ' > $O/jobs/$tag.steps
    log "$tag $s wall ${wall}s metrics $(jq -c '.metrics // .error' <<<"$j")"
    echo "steps: $(cat $O/jobs/$tag.steps)"
    ;;
  director)
    name=$2 conf=$3 sw=$4 stop=$5 app=${6:-fastvideo/h3-trim}
    d=$O/director/$name; rm -rf $d; mkdir -p $d
    # The opening prompt line carries the premise; the configure file the rest.
    timeout 1500 $G/director_client --url http://127.0.0.1:8000 --app "$app" --seconds 900 \
      --configure $G/$conf --prompts $G/prompts.txt --switch-at-chunks "$sw" --stop-after-chunks "$stop" --out $d > $d/client.log 2>&1
    rc=$?
    log "director $name rc=$rc"
    tail -25 $d/client.log
    if [ -s $d/video.h264 ]; then
      ffmpeg -v error -y -r 24 -i $d/video.h264 -i $d/audio.opus -map 0:v -map 1:a -c:v copy -c:a aac -shortest $d/session.mp4 \
        || ffmpeg -v error -y -r 24 -i $d/video.h264 -c:v copy $d/session.mp4
      for t in 1 6 11 16 21 26; do ffmpeg -v error -y -ss $t -i $d/session.mp4 -frames:v 1 -vf scale=480:-2 $d/frame-$t.jpg 2>/dev/null; done
      ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration -of json $d/session.mp4 > $d/probe.json
    fi
    grep -E "director|upsampl" /e2e/serve.log | tail -40 > $d/serve-director.log
    ls -la $d
    ;;
  stream)
    kind=$2
    case $kind in
      whip) out='{"kind":"whip","url":"http://127.0.0.1:8889/fvwhip/whip","target":"mediamtx"}'; path=fvwhip ;;
      rtmp) out='{"kind":"rtmp","url":"rtmp://127.0.0.1:1935/live","stream_key":"fvrtmp"}'; path=live/fvrtmp ;;
      srt) out='{"kind":"srt","url":"srt://127.0.0.1:8890","stream_id":"publish:fvsrt","passphrase":"fvgpucheck1013pass"}'; path=fvsrt ;;
    esac
    model=${STREAM_MODEL:-sfwan21-1.3b}
    body=$(jq -nc --arg m "$model" --argjson o "$out" '{model:$m, prompt:"a coastal road at golden hour, the camera glides along cliffs above a turquoise sea, gulls, warm light", max_seconds:34, output:$o}')
    t0=$(date +%s.%N)
    r=$(curl -s -H "$KEYH" -H 'content-type: application/json' -d "$body" http://127.0.0.1:8000/fv/v1/streams)
    id=$(jq -r .id <<<"$r")
    echo "$r" | jq -c '{id, status, output, playback}'
    # Wait for MediaMTX to have the path ready, then read back 30 s over RTSP.
    for i in $(seq 1 120); do
      rd=$(curl -s http://127.0.0.1:9997/v3/paths/get/$path | jq -r '.ready // false')
      [ "$rd" = true ] && break; sleep 0.5
    done
    ready=$(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.2f", b-a}')
    log "$kind: path $path ready=$rd after ${ready}s"
    curl -s http://127.0.0.1:9997/v3/paths/get/$path | jq -c '{ready, tracks, bytesReceived}' > $O/streams/$kind-mtx.json
    timeout 45 ffmpeg -v error -y -rtsp_transport tcp -i rtsp://127.0.0.1:8554/$path -t 30 -c copy $O/streams/$kind.mp4
    curl -s -H "$KEYH" http://127.0.0.1:8000/fv/v1/streams/$id > $O/streams/$kind-stream.json
    curl -s -X DELETE -H "$KEYH" http://127.0.0.1:8000/fv/v1/streams/$id > /dev/null
    ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=codec_name,profile,width,height,r_frame_rate,avg_frame_rate,nb_read_frames -of json $O/streams/$kind.mp4 > $O/streams/$kind-probe.json
    ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,channels -of csv=p=0 $O/streams/$kind.mp4
    ffmpeg -v error -y -ss 10 -i $O/streams/$kind.mp4 -frames:v 1 -vf scale=480:-2 $O/streams/$kind-frame10.jpg
    jq -c '.streams[0]' $O/streams/$kind-probe.json
    jq -c '{status, end_reason, output, encoder: (.status.encoder // .encoder), stats: (.status // .stats)}' $O/streams/$kind-stream.json | head -c 1500; echo
    ;;
  status)
    tail -2 $O/fetch.log; ls $W
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    tail -5 /e2e/serve.log
    ;;
  *) sed -n 2,10p "$0" ;;
esac
