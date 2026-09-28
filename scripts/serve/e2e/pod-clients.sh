#!/usr/bin/env bash
# On-pod clients for the WP-18 E2E (run through the sidecar from /e2e, after
# `pod.sh bundle`): the streaming clients need a WebRTC path to fv-serve, and
# the pod's own loopback is the one path this harness has (the Runpod pod
# has no UDP, and the test driver's egress is an HTTPS proxy).
#
#   pod-clients.sh setup                 python venv + pinned clients, node, Chromium
#   pod-clients.sh director [res] [app]  tests/compat fal_director.mjs (headless Chromium)
#   pod-clients.sh reactor               reactor_sdk 1.6.0 clip mode (av)
#   pod-clients.sh console               tests/console/smoke.cjs against :8000
#   pod-clients.sh encode <mp4>          h264_nvenc vs libx264 on one clip
set -uo pipefail
E=/e2e
NODE_V=v22.12.0
export PATH="$E/node-dist/bin:$E/venv/bin:$PATH"
export PLAYWRIGHT_BROWSERS_PATH=$E/pw
BASE=http://127.0.0.1:8000

case "${1:-}" in
  setup)
    set -e
    t0=$(date +%s)
    # The pins need Python >= 3.11 (Ubuntu 22.04 has 3.10): uv's CPython.
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$E/uv sh >/dev/null
    rm -rf $E/venv; $E/uv/uv venv -q --python 3.11 $E/venv
    VIRTUAL_ENV=$E/venv $E/uv/uv pip install -q -r $E/tests/compat/requirements.txt
    echo "python clients: $(( $(date +%s) - t0 ))s"
    curl -fsSL "https://nodejs.org/dist/$NODE_V/node-$NODE_V-linux-x64.tar.xz" | tar xJ -C $E
    mv "$E/node-$NODE_V-linux-x64" $E/node-dist
    mkdir -p $E/node && cp $E/tests/compat/package.json $E/tests/compat/package-lock.json $E/node/
    (cd $E/node && PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm ci --no-audit --no-fund --loglevel=error)
    (cd $E/node && npx playwright install --with-deps chromium >/dev/null)
    echo "setup done: $(( $(date +%s) - t0 ))s"
    ;;
  director)
    # $2: resolution (480p | 768p), $3: app; the key comes from FV_KEY.
    FV_WMA_RESOLUTION="${2:-480p}" FV_WMA_APP="${3:-minimax/h3-turbo/director}" \
      timeout 900 node $E/tests/compat/suites/fal_director.mjs $E/node "$BASE" "$FV_KEY"
    ;;
  reactor)
    FV_REACTOR_CLIP_TIMEOUT_S="${FV_REACTOR_CLIP_TIMEOUT_S:-300}" \
      timeout 900 python $E/crates/fastvideo-reactor/tests/compat/reactor_sdk_compat.py --url "$BASE" --mode av
    ;;
  console)
    # $2 = public: the console at its public proxy origin (the upload URLs
    # fv-serve mints are on that origin, and /uploads has no CORS).
    origin=$BASE; [ "${2:-}" = public ] && origin="https://${RUNPOD_POD_ID}-8000.proxy.runpod.net"
    FV_CONSOLE_ORIGIN="$origin" FV_CONSOLE_TIMEOUT_MS="${FV_CONSOLE_TIMEOUT_MS:-420000}" NODE_PATH=$E/node/node_modules \
      timeout 1800 node $E/tests/console/smoke.cjs
    ;;
  encode)
    # Decode once to raw frames, then time each encoder on the same frames
    # (fv-serve's post encoders are ffmpeg h264_nvenc / libx264).
    src="$2"; raw=/tmp/enc.yuv
    case "$src" in http*) curl -fsS -o /tmp/enc-src.mp4 "$src"; src=/tmp/enc-src.mp4 ;; esac
    read -r W H R < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 "$src" | tr ',' ' ')
    ffmpeg -v error -y -i "$src" -f rawvideo -pix_fmt yuv420p $raw
    n=$(( $(stat -c %s $raw) / (W * H * 3 / 2) ))
    # The file_args of fastvideo-media FfmpegH264 (Nvenc / Libx264CpuTest) at cq/crf 19.
    for enc in "h264_nvenc -preset p5 -tune hq -profile:v high -rc vbr -cq 19 -b:v 0 -bf 0" "libx264 -preset veryfast -crf 19" "libx264 -preset medium -crf 19"; do
      /usr/bin/env bash -c "TIMEFORMAT='%R %U %S'; time ffmpeg -v error -y -f rawvideo -pix_fmt yuv420p -s ${W}x${H} -r $R -i $raw -c:v $enc /tmp/enc.mp4" 2> /tmp/enc.time
      read -r real user sys < <(tail -1 /tmp/enc.time)
      echo "{\"encoder\":\"$enc\",\"frames\":$n,\"size\":\"${W}x${H}\",\"wall_s\":$real,\"cpu_user_s\":$user,\"cpu_sys_s\":$sys,\"cpu_cores_avg\":$(awk -v u="$user" -v s="$sys" -v r="$real" 'BEGIN{printf "%.2f", (u+s)/r}'),\"bytes\":$(stat -c %s /tmp/enc.mp4)}"
    done
    echo "{\"nproc\":$(nproc)}"
    rm -f $raw /tmp/enc.mp4
    ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
