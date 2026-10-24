#!/bin/bash
# H3 test round + Cloudflare stream outputs (docs/serve/e2e/h3-test-round-2026-10-14.md), on-pod driver run through
# the e2e sidecar (scripts/serve/e2e/pod.sh) from /e2e/hr (the bundle). Image: the full serve image (fv-serve and
# fv-gpucheck). Weights: the EU volume's trees linked read only under $W, the R2-only trees fetched beside them
# (plan.json, presigned). Nothing is written to /workspace.
#
#   h3-round.sh prep                      links, R2 fetch (background), nvidia-smi sampler
#   h3-round.sh t2v-all                   the T2V matrix with fv-gpucheck, one warm process per cell, then the
#                                         quality gate of every cell against the 49-forward base (background job:
#                                         $O/t2v-status.tsv, $O/t2v.log)
#   h3-round.sh t2v <cell> [frames]       one cell (cells: see cell()), prompts $O/prompts.json
#   h3-round.sh cmp <cell>                compare-clips <cell> vs base for each prompt
#   h3-round.sh serve <model> [K=V ...]   (re)start fv-serve on one H3 model, wait for /ping 200
#   h3-round.sh job <tag> <model> <task> <short edge>    one native job (t2v|i2v|fl2va|ref2v), metrics + mp4
#   h3-round.sh cond-all                  serve each image-capable model and run I2V / FL2VA (+ Ref2V) at 480 and
#                                         768 (background: $O/cond-status.tsv, $O/cond.log)
#   h3-round.sh cf                        Cloudflare: fv-serve on h3-turbo with the Cloudflare default, the five
#                                         stream kinds (cf_live_check), then the director x_fv_output copy
#   h3-round.sh status
set -u
G=/e2e/hr
O=/e2e/out
W=/root/w
L=/root/l
BIN=/opt/fastvideo-rs/target/release/fv-gpucheck
mkdir -p $O $L
KEYH="Authorization: Bearer ${FV_KEY:-}"
log() { echo "[hr $(date -u +%H:%M:%S)] $*"; }
I2V_PROMPT="Waves roll onto a rocky beach at sunset, sea foam swirling around the dark rocks, as the camera slowly pushes in"
REF_PROMPT="The camera glides slowly forward along the shoreline of the beach in Image 1, turquoise waves rolling in and breaking into white foam, bright sunny day, the sound of the surf and a light wind."

prompts() { # the standard pair (scripts/gpu/prompts-eval.json h3-demo, spark-mountain-lake) at 480p and 768p
  python3 - $G/prompts-eval.json $O/prompts.json $O/prompts-480.json <<'PYEOF'
import json, sys
e = {p["name"]: p for p in json.load(open(sys.argv[1]))["prompts"]}
rows = []
for size, (h, w) in (("480", (480, 832)), ("768", (768, 1344))):
    for k in ("h3-demo", "spark-mountain-lake"):
        rows.append({"name": f"{k}-{size}p", "prompt": e[k]["prompt"], "seed": e[k]["seed"], "height": h, "width": w})
json.dump({"prompts": rows}, open(sys.argv[2], "w"), indent=1)
json.dump({"prompts": [r for r in rows if r["height"] == 480]}, open(sys.argv[3], "w"), indent=1)
PYEOF
}

# cell -> "weights|recipe|profile|extra flags|env"
cell() {
  case $1 in
    base)      echo "h3-base|base||--dense|" ;;
    turbo)     echo "h3-base|4step-vsa|h3/fasth3_4step_vsa||" ;;
    max)       echo "h3-base|sol-h3|h3/sol_h3_4step_engine_ladder||" ;;
    draft)     echo "h3-base|4step-vsa|h3/fasth3_4step_vsa|--taeh3-weights $W/auxiliary/tae/taeh3.safetensors|" ;;
    v2)        echo "h3-8step|8step||--dense --adaln-cache $L/v2-adaln.cache|" ;;
    trim-mx)   echo "fasth3-trim-8step|trim-8step||--adaln-cache $L/trim-adaln.cache|" ;;
    trim-bf16) echo "fasth3-trim-8step|trim-8step||--adaln-cache $L/trim-adaln.cache|FASTVIDEO_H3_QUANT=off" ;;
    larry)     echo "h3-base|larry-turbo|h3/larry_turbo_dense||" ;;
    lightx)    echo "h3-base|lightx2v-turbo|h3/lightx2v_turbo_dense||" ;;
    *) return 1 ;;
  esac
}

# A tree fetched from R2 is usable once r2-fetch.py wrote its .complete; one linked from the volume (its top
# directory a symlink) at once.
wait_tree() { until [ -f "$W/$1/.complete" ] || [ -L "$W/${1%%/*}" ]; do sleep 5; done; }
wait_for() { # <cell or model>: every tree it reads
  wait_tree h3-base
  case $1 in
    turbo|draft) wait_tree FastH3-4-step-Preview-v1-LoRA; wait_tree h3-base/transformer_prequant_4step-vsa_mxfp8 ;;
    max) wait_tree FastH3-4-step-Preview-v1-LoRA; wait_tree h3-base/transformer_prequant_sol-h3_mxfp8 ;;
    larry) wait_tree h3-turbo-loras/larry ;;
    lightx) wait_tree h3-turbo-loras/lightx2v ;;
    v2) wait_tree h3-8step ;;
    trim*) wait_tree fasth3-trim-8step ;;
    ref2v*) wait_tree h3-ref2va ;;
  esac
  case $1 in draft) wait_tree auxiliary/tae ;; esac
}

t2v() { # <cell> [frames] [prompt file]
  local c=$1 nf=${2:-124} pf=${3:-$O/prompts.json} spec weights recipe prof extra envs tag
  spec="$(cell "$c")" || { log "unknown cell $c"; return 2; }
  IFS='|' read -r weights recipe prof extra envs <<<"$spec"
  [ "$c" = draft ] && pf=$O/prompts-480.json
  tag=$c; [ "$nf" != 124 ] && tag=$c-f$nf
  wait_for "$c"
  rm -rf $O/t2v/$tag; mkdir -p $O/t2v/$tag
  # shellcheck disable=SC2086
  env RUST_LOG=info FASTVIDEO_TAE_DIR=$W/auxiliary/tae $envs $BIN --mode fast ${prof:+--techniques $prof} \
    --out $O/t2v/$tag --tag "$tag" h3 gen --weights "$W/$weights" --text-weights "$W/h3-base" --text-encoder streamed \
    --text-cache $L/text-cache --prompt unused --prompts "$pf" --height 480 --width 832 --num-frames "$nf" \
    --h3-recipe "$recipe" --clip-dir $L/clips/$tag --warm $extra > $O/t2v/$tag/run.log 2>&1
  local rc=$?
  local d
  for d in "$L/clips/$tag"/*/; do
    [ -f "$d/output.mp4" ] && cp "$d/output.mp4" "$O/t2v/$tag/$(basename "$d").mp4"
  done
  return $rc
}

cmp() { # <cell>: against the base, every prompt both ran
  local c=$1 p d
  wait_tree auxiliary/lpips
  mkdir -p $O/cmp
  for d in "$L/clips/$c"/*/; do
    p=$(basename "$d")
    [ -d $L/clips/base/$p ] || continue
    $BIN --out $O/cmp --tag "base--$c-$p" compare-clips --baseline $L/clips/base/$p --candidate $L/clips/$c/$p \
      --lpips $W/auxiliary/lpips > $O/cmp/base--$c-$p.out 2>&1 || log "cmp $c $p rc=$?"
  done
}

step() { # <status file> <name> <cmd...>
  local f=$1 n=$2 t0=$SECONDS rc; shift 2
  log "begin $n"
  "$@"; rc=$?
  printf '%s\t%s\t%s\t%s\n' "$n" "$rc" "$((SECONDS - t0))" "$(date -u +%H:%M:%S)" >> "$f"
  log "end $n rc=$rc $((SECONDS - t0))s"
}

# ---- fv-serve ------------------------------------------------------------------------------------------------
model_toml() { # <model>: the [[models]] block, as the shipped configs (configs/serve/runpod-*.toml) but warmup off
  case $1 in
    turbo) cat <<EOF
[[models]]
variant = "h3-turbo"
id = "fasth3"
weights = "$W/h3-base"
resident = true
continuity = "anchor-last-frame"
i2v_encoder = "auto"
EOF
    ;;
    max) printf '[[models]]\nvariant = "h3-max"\nweights = "%s/h3-base"\nresident = true\ni2v_encoder = "auto"\n' $W ;;
    draft) printf '[[models]]\nvariant = "h3-draft"\nweights = "%s/h3-base"\nresident = true\ni2v_encoder = "auto"\n' $W ;;
    v2) printf '[[models]]\nvariant = "fasth3-8step-dense"\nweights = "%s/h3-8step"\nresident = true\ni2v_encoder = "auto"\n' $W ;;
    larry) printf '[[models]]\nid = "h3-larry-turbo"\nfamily = "h3"\nrecipe = "h3-larry-turbo"\nweights = "%s/h3-base"\nresident = true\nsteps = 6\nlora_strength = 1.0\ni2v_encoder = "auto"\n' $W ;;
    lightx) printf '[[models]]\nid = "h3-lightx2v-turbo"\nfamily = "h3"\nrecipe = "h3-lightx2v-turbo"\nweights = "%s/h3-base"\nresident = true\nsteps = 4\nlora_strength = 1.0\ni2v_encoder = "auto"\n' $W ;;
    ref2v-turbo) printf '[[models]]\nid = "h3-ref2v-turbo"\nfamily = "h3"\nrecipe = "h3-ref2v-turbo"\nweights = "%s/h3-base"\nresident = true\n' $W ;;
    ref2v-max) printf '[[models]]\nid = "h3-ref2v-max"\nfamily = "h3"\nrecipe = "h3-ref2v-max"\nweights = "%s/h3-base"\nresident = true\n' $W ;;
    *) return 1 ;;
  esac
}

write_config() { # <model> -> $G/<model>.toml
  {
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

[limits]
queue_max = 16
body_max_mb = 64

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
    model_toml "$1"
  } > $G/$1.toml
}

serve() { # <model> [K=V ...]
  local m=$1 c t0; shift
  write_config "$m" || { log "unknown model $m"; return 2; }
  wait_for "$m"
  [ -L "$W/h3-base" ] || wait_tree h3-base/text_encoder_fp8
  t0=$(date +%s.%N)
  env FV_WEIGHTS=$W FV_ARTIFACTS_DIR=/fvstate/artifacts FV_STATE_DIR=/fvstate/$m "$@" /e2e/serve.sh $G/$m.toml
  for _ in $(seq 1 900); do
    c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/ping)
    [ "$c" = 200 ] && break
    sleep 1
  done
  local s; s=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')
  echo "{\"model\":\"$m\",\"ready_s\":$s,\"ping\":\"$c\"}" > $O/serve-$m.json
  log "serve $m: /ping $c after ${s}s"
  grep -E "loaded|load |resident|encoder|nvenc|ERROR|error" /e2e/serve.log | tail -8
  [ "$c" = 200 ]
}

b64img() { printf 'data:image/jpeg;base64,%s' "$(base64 -w0 "$1")"; }

job() { # <tag> <model id> <task> <short edge>
  local tag=$1 model=$2 task=$3 se=$4 body t0 r id j s wall
  case $task in
    t2v) body=$(jq -nc --arg m "$model" --argjson se "$se" --arg p "$I2V_PROMPT" '{model:$m, prompt:$p, seed:7, aspect_ratio:"16:9", short_edge:$se, num_frames:124}') ;;
    i2v) body=$(jq -nc --arg m "$model" --argjson se "$se" --arg p "$I2V_PROMPT" --arg a "$(b64img $G/ti2v-beach-832x480.jpg)" \
           '{model:$m, prompt:$p, seed:7, short_edge:$se, num_frames:124, image_url:$a}') ;;
    fl2va) body=$(jq -nc --arg m "$model" --argjson se "$se" --arg p "$I2V_PROMPT" --arg a "$(b64img $G/ti2v-beach-832x480.jpg)" \
           --arg b "$(b64img $G/ti2v-beach-zoom-832x480.jpg)" '{model:$m, prompt:$p, seed:7, short_edge:$se, num_frames:124, image_url:$a, last_image_url:$b}') ;;
    ref2v) body=$(jq -nc --arg m "$model" --argjson se "$se" --arg p "$REF_PROMPT" --arg a "$(b64img $G/ti2v-beach-832x480.jpg)" \
           '{model:$m, prompt:$p, seed:7, short_edge:$se, num_frames:124, reference_urls:[$a]}') ;;
  esac
  mkdir -p $O/jobs
  t0=$(date +%s.%N)
  # Two data-URI images exceed the argument limit: the body goes through a file.
  printf '%s' "$body" > $L/body.json
  r=$(curl -s -H "$KEYH" -H 'content-type: application/json' --data-binary @$L/body.json http://127.0.0.1:8000/fv/v1/jobs)
  id=$(jq -r '.id // empty' <<<"$r")
  [ -n "$id" ] || { log "$tag: submit failed: $(head -c 300 <<<"$r")"; echo "{\"tag\":\"$tag\",\"error\":$(jq -c . <<<"$r" 2>/dev/null || echo null)}" > $O/jobs/$tag.json; return 1; }
  while :; do
    j=$(curl -s -H "$KEYH" http://127.0.0.1:8000/fv/v1/jobs/$id)
    s=$(jq -r .status <<<"$j")
    case $s in succeeded|completed|failed|cancelled|canceled) break ;; esac
    sleep 0.5
  done
  wall=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.2f", b-a}')
  jq --arg tag "$tag" --arg task "$task" --argjson wall $wall '. + {hr_tag:$tag, hr_task:$task, hr_wall_s:$wall}' <<<"$j" > $O/jobs/$tag.json
  # The output: the first artifact URL the job object names (local backend, served by fv-serve).
  local u; u=$(jq -r '[.. | strings | select(test("^https?://.*\\.mp4"))][0] // empty' <<<"$j")
  [ -n "$u" ] && curl -s -H "$KEYH" -o $O/jobs/$tag.mp4 "$(sed 's#^https\?://[^/]*#http://127.0.0.1:8000#' <<<"$u")"
  log "$tag $s wall ${wall}s metrics $(jq -c '.metrics // .error' <<<"$j" | head -c 600)"
  [ "$s" = succeeded ] || [ "$s" = completed ]
}

cond_model() { # <model> <served id> <tasks...>
  local m=$1 id=$2 t se; shift 2
  serve "$m" || return 1
  job "$m-warm" "$id" t2v 480 || true   # untimed: kernels, allocator, the conditioning path
  for t in "$@"; do
    for se in 480 768; do
      [ "$m" = draft ] && [ $se = 768 ] && continue
      job "$m-$t-$se" "$id" "$t" "$se" || true
    done
  done
}

case "${1:-}" in
  prep)
    mkdir -p $W/h3-turbo-loras
    for t in h3-base FastH3-4-step-Preview-v1-LoRA h3-ref2va auxiliary; do
      [ -e /workspace/weights/$t ] && ln -sfn /workspace/weights/$t $W/$t
    done
    prompts
    nohup bash -c "nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits -l 1 > $O/smi.csv" >/dev/null 2>&1 &
    fetcher=/opt/fastvideo-rs/bin/r2-fetch.py; [ -f $fetcher ] || fetcher=/opt/fastvideo-rs/scripts/gpu/r2-fetch.py
    if [ -s $G/plan.json ]; then
      nohup python3 $fetcher --root $W --plan $G/plan.json --status $O/fetch-status.json --concurrency 32 \
        > $O/fetch.log 2>&1 < /dev/null &
    fi
    chmod +x $G/cf_live_check $G/director_client 2>/dev/null
    ls -la $W /workspace/weights | head -30
    df -h / /workspace | cat
    log "prep done"
    ;;
  t2v) shift; t2v "$@" ;;
  cmp) shift; cmp "$@" ;;
  t2v-all)
    shift
    cells="${*:-base turbo max draft larry lightx trim-mx trim-bf16 v2}"
    (
      for c in $cells; do step $O/t2v-status.tsv "t2v-$c" t2v "$c"; done
      for c in $cells; do [ "$c" = base ] || step $O/t2v-status.tsv "cmp-$c" cmp "$c"; done
      echo "finished" >> $O/t2v-status.tsv
    ) > $O/t2v.log 2>&1 &
    log "t2v-all started: $cells"
    ;;
  serve) shift; serve "$@" ;;
  job) shift; job "$@" ;;
  cond-all)
    shift
    models="${*:-turbo max larry lightx v2 draft ref2v-turbo ref2v-max}"
    (
      for m in $models; do
        case $m in
          turbo) step $O/cond-status.tsv cond-turbo cond_model turbo fasth3 i2v fl2va ;;
          max) step $O/cond-status.tsv cond-max cond_model max sol-h3 i2v fl2va ;;
          larry) step $O/cond-status.tsv cond-larry cond_model larry h3-larry-turbo i2v fl2va ;;
          lightx) step $O/cond-status.tsv cond-lightx cond_model lightx h3-lightx2v-turbo i2v fl2va ;;
          v2) step $O/cond-status.tsv cond-v2 cond_model v2 fasth3-8step-dense i2v fl2va ;;
          draft) step $O/cond-status.tsv cond-draft cond_model draft fasth3-4step-vsa-480p-taeh3 i2v fl2va ;;
          ref2v-turbo) step $O/cond-status.tsv cond-ref2v-turbo cond_model ref2v-turbo h3-ref2v-turbo ref2v ;;
          ref2v-max)
            # 49 forwards: 480p only (768p took 681 s of denoise on 2026-10-04's run)
            serve ref2v-max && job ref2v-max-ref2v-480 h3-ref2v-max ref2v 480; printf 'cond-ref2v-max\t%s\n' $? >> $O/cond-status.tsv ;;
        esac
      done
      echo "finished" >> $O/cond-status.tsv
    ) > $O/cond.log 2>&1 &
    log "cond-all started: $models"
    ;;
  cf)
    # Cloudflare (owner approval 2026-10-14): the token is only in this pod's environment (Runpod secret
    # reference); nothing here prints it. Live inputs are named fv-test-*; only those this run creates are touched.
    shift
    [ -n "${FV_CF_STREAM_API_TOKEN:-}" ] && [ -n "${FV_CF_ACCOUNT_ID:-}" ] || { log "no Cloudflare env"; exit 1; }
    serve turbo FV_CF_STREAM_NAME_PREFIX=fv-test- || exit 1
    curl -s http://127.0.0.1:8000/fv/v1/outputs -H "$KEYH" | jq -c '{default, kinds, codecs}'
    mkdir -p $O/cf
    $G/cf_live_check list | tee $O/cf/list-before.txt
    $G/cf_live_check streams --model fasth3 --cases "${CF_CASES:-whip-h264,whip-vp8,whip-vp9,rtmps,srt}" \
      --seconds "${CF_SECONDS:-60}" --out $O/cf 2>&1 | tee $O/cf/streams.log | grep -E "RESULT|CREATED|LIST|created"
    ;;
  cf-hls-audio)
    # RTMPS / SRT again, with a side probe of Cloudflare's HLS audio: Cloudflare serves the audio as its own
    # rendition (#EXT-X-MEDIA TYPE=AUDIO), which the checker's video-variant read does not see. For each stream
    # the checker creates, ffprobe reads the whole master playlist (every stream) once it plays, and ffmpeg
    # measures the audio loudness over 10 s.
    mkdir -p $O/${CF_OUT:-cf2}
    $G/cf_live_check streams --model fasth3 --cases "${CF_CASES:-rtmps,srt}" --seconds "${CF_SECONDS:-60}" --out $O/${CF_OUT:-cf2} \
      > $O/${CF_OUT:-cf2}/streams.log 2>&1 &
    cpid=$!
    seen=""
    while kill -0 $cpid 2>/dev/null; do
      for id in $(grep -o 'stream fvstream_[0-9a-f]*' $O/${CF_OUT:-cf2}/streams.log | awk '{print $2}'); do
        case " $seen " in *" $id "*) continue ;; esac
        seen="$seen $id"
        (
          hls=$(curl -s -H "$KEYH" http://127.0.0.1:8000/fv/v1/streams/$id | jq -r '.playback.hls_url // empty')
          kind=$(curl -s -H "$KEYH" http://127.0.0.1:8000/fv/v1/streams/$id | jq -r '.output.kind')
          t0=$(date +%s)
          for _ in $(seq 1 60); do
            ffprobe -v error -show_entries stream=index,codec_type,codec_name,sample_rate,channels,width,height -of json "$hls" \
              > $O/${CF_OUT:-cf2}/$kind-master-probe.json 2>/dev/null && jq -e '[.streams[] | select(.codec_type == "audio")] | length > 0' \
              $O/${CF_OUT:-cf2}/$kind-master-probe.json >/dev/null && break
            sleep 3
          done
          echo "{\"kind\":\"$kind\",\"audio_listed_after_s\":$(( $(date +%s) - t0 ))}" > $O/${CF_OUT:-cf2}/$kind-audio-timing.json
          ffmpeg -v info -nostdin -t "${CF_AUDIO_SECS:-10}" -i "$hls" -map 0:a:0 -af volumedetect -f null - 2>&1 | grep -E "mean_volume|max_volume|Audio:" \
            > $O/${CF_OUT:-cf2}/$kind-audio-volume.txt
        ) &
      done
      sleep 2
    done
    wait
    grep -E "RESULT|CREATED|LIST" $O/${CF_OUT:-cf2}/streams.log | cut -c1-600
    for f in $O/${CF_OUT:-cf2}/*-master-probe.json; do echo "$f"; jq -c '[.streams[] | {codec_type, codec_name, width, height, sample_rate, channels}] | unique' "$f"; done
    cat $O/${CF_OUT:-cf2}/*-audio-volume.txt $O/${CF_OUT:-cf2}/*-audio-timing.json
    ;;
  cf-audio)
    # What audio Cloudflare delivers over WHEP, decoded (aiortc: Opus -> PCM, RMS per second), for a native WHIP
    # stream and for the director's x_fv_output copy. Needs fv-serve on turbo with the Cloudflare env.
    d=$O/cf-audio; mkdir -p $d
    [ -x /opt/aio/bin/python ] || { python3 -m venv /opt/aio && /opt/aio/bin/pip install -q aiortc==1.9.0 aiohttp numpy; } > $d/pip.log 2>&1
    cp $G/whep_audio.py /opt/whep_audio.py
    # 1. native stream, output cloudflare (WHIP H.264)
    body=$(jq -nc '{model:"fasth3", prompt:"A lighthouse keeper climbs the spiral stairs at dusk and lights the great lamp; waves crash below, gulls cry, wind howls against the glass.", output:{kind:"cloudflare"}, max_seconds:150}')
    r=$(curl -s -H "$KEYH" -H 'content-type: application/json' -d "$body" http://127.0.0.1:8000/fv/v1/streams)
    id=$(jq -r .id <<<"$r"); uid=$(jq -r '.playback.live_input_id // empty' <<<"$r"); url=$(jq -r '.playback.whep_url // empty' <<<"$r")
    log "native stream $id live input $uid"
    sleep 30
    /opt/aio/bin/python /opt/whep_audio.py --url "$url" --seconds 60 --out $d/native-whip-h264.json
    curl -s -H "$KEYH" http://127.0.0.1:8000/fv/v1/streams/$id | jq -c '.status | {output_state, encoder, video_codec, width, height, frames_sent, audio_packets_sent}'
    curl -s -X DELETE -H "$KEYH" http://127.0.0.1:8000/fv/v1/streams/$id > /dev/null
    sleep 4
    [ -n "$uid" ] && $G/cf_live_check cleanup --uids "$uid" | tee $d/native-cleanup.txt
    # 2. the director copy
    dd=$d/director; rm -rf $dd; mkdir -p $dd
    timeout 600 $G/director_client --url http://127.0.0.1:8000 --app minimax/h3-turbo --seconds 400 --configure $G/configure-cf.json \
      --prompts $G/prompts-cf.txt --switch-at-chunks 1 --stop-after-chunks "${CF_CHUNKS:-3}" --out $dd > $dd/client.log 2>&1 &
    cpid=$!
    url=""
    for _ in $(seq 1 300); do
      url=$(jq -r 'select(.msg.type == "output_ready") | .msg.playback.whep_url // empty' $dd/events.jsonl 2>/dev/null | head -1)
      [ -n "$url" ] && break
      sleep 1
    done
    uid=$(jq -r 'select(.msg.type == "output_ready") | .msg.playback.live_input_id // empty' $dd/events.jsonl | head -1)
    [ -n "$url" ] && /opt/aio/bin/python /opt/whep_audio.py --url "$url" --seconds 60 --out $d/director-copy.json
    wait $cpid
    ffmpeg -v info -nostdin -i $dd/audio.opus -af volumedetect -f null - 2>&1 | grep -E "mean_volume|max_volume" > $dd/main-peer-audio-volume.txt
    sleep 4
    [ -n "$uid" ] && $G/cf_live_check cleanup --uids "$uid" | tee $d/director-cleanup.txt
    ;;
  cf-audio2)
    # Native clip streams play each queued clip once (5 s) and then hold the last frame in silence, so a read that
    # starts after the clip hears nothing. Here each stream queues 8 clips (a new 5 s clip every ~26 s of build)
    # and the read starts at once: WHIP decoded over WHEP (whep_audio.py, retried until Cloudflare plays it), and
    # RTMPS / SRT over HLS (ffmpeg volumedetect on 60 s of the master playlist's audio rendition).
    d=$O/cf-audio2; mkdir -p $d; cp $G/whep_audio.py /opt/whep_audio.py
    [ -x /opt/aio/bin/python ] || { python3 -m venv /opt/aio && /opt/aio/bin/pip install -q aiortc==1.9.0 aiohttp numpy; } > $d/pip.log 2>&1
    for kind in cloudflare cloudflare-rtmps cloudflare-srt; do
      body=$(jq -nc --arg k "$kind" '{model:"fasth3", max_seconds:280, output:{kind:$k},
        prompt:"A lighthouse keeper climbs the spiral stairs at dusk and lights the great lamp; waves crash below, gulls cry, wind howls against the glass.",
        clips:[range(7) | {prompt:"Storm waves crash against the lighthouse rocks at night, thunder rolls, rain lashes the windows, the lamp beam sweeps the sea."}]}')
      t0=$(date +%s)
      r=$(curl -s -H "$KEYH" -H 'content-type: application/json' -d "$body" http://127.0.0.1:8000/fv/v1/streams)
      id=$(jq -r .id <<<"$r"); uid=$(jq -r '.playback.live_input_id // empty' <<<"$r")
      log "$kind: stream $id live input $uid"
      if [ "$kind" = cloudflare ]; then
        url=$(jq -r '.playback.whep_url // empty' <<<"$r")
        for _ in $(seq 1 40); do
          /opt/aio/bin/python /opt/whep_audio.py --url "$url" --seconds 100 --out $d/whip-h264.json > $d/whip-h264.out 2>&1 \
            && jq -e '.audio_frames > 0' $d/whip-h264.json >/dev/null && break
          sleep 2
        done
        tail -1 $d/whip-h264.out
      else
        hls=$(jq -r '.playback.hls_url // empty' <<<"$r")
        for _ in $(seq 1 80); do
          ffprobe -v error -show_entries stream=codec_type -of csv=p=0 "$hls" 2>/dev/null | grep -q audio && break
          sleep 3
        done
        echo "{\"kind\":\"$kind\",\"audio_listed_after_s\":$(( $(date +%s) - t0 ))}" > $d/$kind-timing.json
        ffmpeg -v info -nostdin -t 60 -i "$hls" -map 0:a:0 -af "volumedetect,astats=metadata=1:reset=47,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=$d/$kind-rms.txt" -f null - 2>&1 \
          | grep -E "Audio:|mean_volume|max_volume" > $d/$kind-volume.txt
        cat $d/$kind-timing.json $d/$kind-volume.txt
      fi
      curl -s -H "$KEYH" http://127.0.0.1:8000/fv/v1/streams/$id | jq -c '.status | {output_state, encoder, video_codec, width, height, frames_sent, audio_packets_sent}'
      curl -s -X DELETE -H "$KEYH" http://127.0.0.1:8000/fv/v1/streams/$id > /dev/null
      sleep 4
      [ -n "$uid" ] && $G/cf_live_check cleanup --uids "$uid" | tee $d/$kind-cleanup.txt
    done
    ;;
  cf-director)
    # The director's x_fv_output copy to Cloudflare (WHIP), read back over WHEP while the session runs.
    d=$O/cf/director; rm -rf $d; mkdir -p $d
    echo '{"x_fv_output": {"kind": "cloudflare"}, "seed": 1014}' > $G/configure-cf.json
    printf '%s\n%s\n' "A lighthouse keeper climbs the spiral stairs at dusk and lights the great lamp; waves crash below, gulls cry, wind howls against the glass." \
      "The lamp blazes to life and sweeps its beam across the stormy sea; thunder rolls, rain lashes the windows." > $G/prompts-cf.txt
    timeout 900 $G/director_client --url http://127.0.0.1:8000 --app minimax/h3-turbo --seconds 600 --configure $G/configure-cf.json \
      --prompts $G/prompts-cf.txt --switch-at-chunks 1 --stop-after-chunks "${CF_CHUNKS:-4}" --out $d > $d/client.log 2>&1 &
    cpid=$!
    url=""
    for _ in $(seq 1 600); do
      url=$(jq -r 'select(.msg.type == "output_ready") | .msg.playback.whep_url // empty' $d/events.jsonl 2>/dev/null | head -1)
      [ -n "$url" ] && break
      kill -0 $cpid 2>/dev/null || break
      sleep 1
    done
    jq -c 'select(.msg.type == "output_ready") | .t_ms as $t | .msg | {t_ms: $t, output, live_input: .playback.live_input_id, playback_keys: (.playback | keys)}' $d/events.jsonl | head -1 > $d/output-ready.json
    if [ -n "$url" ]; then
      $G/cf_live_check whep --url "$url" --codec h264 --seconds "${CF_WATCH:-40}" --out $d 2>&1 | grep RESULT
    else
      log "no output_ready"
    fi
    wait $cpid; log "director client rc=$?"
    tail -12 $d/client.log
    grep -E '"output_(ready|error|ended)"|configured' $d/events.jsonl | jq -c '.msg | del(.x_fv_output.fields?)' | head -8 > $d/output-events.jsonl
    cat $d/output-events.jsonl | head -c 1500; echo
    uid=$(jq -r '.live_input // empty' $d/output-ready.json)
    sleep 5
    [ -n "$uid" ] && $G/cf_live_check cleanup --uids "$uid" | tee $d/cleanup.txt
    grep -E "director|output" /e2e/serve.log | grep -v -i "token" | tail -30 > $d/serve-director.log
    ;;
  status)
    tail -3 $O/fetch.log 2>/dev/null; cat $O/t2v-status.tsv $O/cond-status.tsv 2>/dev/null | tail -20
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    df -h / | tail -1
    tail -3 /e2e/serve.log
    ;;
  *) sed -n 2,20p "$0" ;;
esac
