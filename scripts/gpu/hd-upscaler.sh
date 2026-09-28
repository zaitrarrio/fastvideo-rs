#!/usr/bin/env bash
# Upscaler benchmark for the hd matrix family (docs/serve/h3-1080p-and-upscaler.md).
# Runs ON the pod after the hd cells, before their PNG frames are deleted:
#   bash hd-upscaler.sh <RUNS>        (FV_BIN, FV_LPIPS_ARGS from runpod-matrix.sh)
# SeedVR2 (one-step diffusion VSR, Apache-2.0) through the numz standalone CLI
# on the turbo-768p clips: 768p -> 1088p short side (2 = 1.42x) and one clip
# at 1536p (2x). Python, torch and the weights live on the container disk
# (/root/hd-up); nothing is written to the network volume. Each run records
# wall time and peak GPU memory (nvidia-smi sampled every 0.5 s); the frames
# are resized to the native 1080p canvas and compared (fv-gpucheck
# compare-clips) with the Lanczos baseline of the same clip.
set -uo pipefail
RUNS="${1:?runs dir}"
BIN="${FV_BIN:-/opt/fastvideo-rs/target/release/fv-gpucheck}"
UP=/root/hd-up
SV_SHA="${FV_SEEDVR2_SHA:-4490bd1f482e026674543386bb2a4d176da245b9}"
OUT="$RUNS/upscaler"
mkdir -p "$UP" "$OUT"
log() { printf '[%s] [upscaler] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$OUT/upscaler.log" >>"$RUNS/live.log"; }

log "setup: uv + python 3.12 + torch cu128"
t0=$(date +%s)
{
  command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UP/bin" sh
  export PATH="$UP/bin:$PATH"
  uv venv -q -p 3.12 "$UP/venv"
  . "$UP/venv/bin/activate"
  uv pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
  git init -q "$UP/seedvr2" && git -C "$UP/seedvr2" remote add origin https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git
  git -C "$UP/seedvr2" fetch -q --depth 1 origin "$SV_SHA" && git -C "$UP/seedvr2" checkout -q FETCH_HEAD
  uv pip install -q -r "$UP/seedvr2/requirements.txt"
  python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
} >"$OUT/setup.log" 2>&1
rc=$?
log "setup exit=$rc $(( $(date +%s) - t0 ))s $(tail -1 "$OUT/setup.log")"
[[ $rc -eq 0 ]] || exit 1
export PATH="$UP/bin:$PATH"
. "$UP/venv/bin/activate"

# name model resolution clip [extra args...]
upscale() {
  local name="$1" model="$2" res="$3" clip="$4"
  shift 4
  local src="$RUNS/turbo-768p/frames/$clip" dir="$RUNS/$name" mem="$OUT/$name-$clip.mem"
  [[ -f "$src/frame-000.png" ]] || { log "skip $name/$clip: no frames"; return 0; }
  mkdir -p "$dir/raw/$clip" "$dir/frames/$clip"
  ffmpeg -nostdin -loglevel error -y -framerate 24 -start_number 0 -i "$src/frame-%03d.png" \
    -c:v libx264 -crf 8 -pix_fmt yuv444p "$UP/in-$clip.mp4"
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) >"$mem" 2>/dev/null &
  local smi=$! t1 t2 rc
  t1=$(date +%s.%N)
  python "$UP/seedvr2/inference_cli.py" "$UP/in-$clip.mp4" --output "$dir/raw/$clip" --output_format png \
    --model_dir "$UP/models" --dit_model "$model" --resolution "$res" "$@" \
    >"$OUT/$name-$clip.log" 2>&1
  rc=$?
  t2=$(date +%s.%N)
  kill "$smi" 2>/dev/null
  wait "$smi" 2>/dev/null
  local peak secs
  peak="$(sort -n "$mem" | tail -1)"
  secs="$(awk -v a="$t1" -v b="$t2" 'BEGIN{printf "%.1f", b-a}')"
  local n
  n="$(find "$dir/raw/$clip" -name '*.png' | wc -l)"
  local size
  size="$(find "$dir/raw/$clip" -name '*.png' | sort | head -1 | xargs -r python -c 'import sys; from PIL import Image; print("%dx%d" % Image.open(sys.argv[1]).size)' 2>/dev/null)"
  log "$name/$clip exit=$rc wall=${secs}s peak_gpu_mib=$peak frames=$n size=$size"
  printf '{"cell":"%s","clip":"%s","model":"%s","resolution":%s,"exit":%s,"wall_s":%s,"peak_gpu_mib":%s,"frames":%s,"size":"%s"}\n' \
    "$name" "$clip" "$model" "$res" "$rc" "$secs" "${peak:-null}" "$n" "$size" >>"$OUT/runs.jsonl"
  [[ $rc -eq 0 && $n -gt 0 ]] || return 0
  # To the native 1080p canvas, then against the Lanczos baseline of the same clip.
  find "$dir/raw/$clip" -name '*.png' | sort | awk '{printf "file '\''%s'\''\n", $0}' >"$UP/list-$name-$clip.txt"
  ffmpeg -nostdin -loglevel error -y -f concat -safe 0 -i "$UP/list-$name-$clip.txt" \
    -vf "scale=1920:1088:flags=lanczos" -start_number 0 "$dir/frames/$clip/frame-%03d.png"
  local base="$RUNS/turbo-768p-lanczos/frames/$clip"
  if [[ -f "$base/frame-000.png" ]]; then
    # shellcheck disable=SC2086
    "$BIN" --out "$RUNS/compare" --tag "turbo-768p-lanczos--$name-$clip" compare-clips \
      --baseline "$base" --candidate "$dir/frames/$clip" ${FV_LPIPS_ARGS:-} \
      >"$RUNS/compare/turbo-768p-lanczos--$name-$clip.stdout.log" 2>"$RUNS/compare/turbo-768p-lanczos--$name-$clip.stderr.log"
    log "compare $name/$clip exit=$?"
  fi
  rm -rf "$dir/raw/$clip"
}

mkdir -p "$RUNS/compare"
common=(--batch_size 33 --uniform_batch_size --temporal_overlap 3 --seed 42)
first=1
for clip in talking-head spark-mountain-lake ltx-frogyoga; do
  upscale seedvr2-3b-1088 seedvr2_ema_3b_fp16.safetensors 1088 "$clip" "${common[@]}"
  if [[ $first == 1 ]]; then
    # The first run downloads the weights and compiles nothing; run it again
    # for a warm number (models on disk, page cache hot).
    log "warm repeat"
    upscale seedvr2-3b-1088-warm seedvr2_ema_3b_fp16.safetensors 1088 "$clip" "${common[@]}"
    first=0
  fi
done
upscale seedvr2-3b-1536 seedvr2_ema_3b_fp16.safetensors 1536 spark-mountain-lake "${common[@]}" \
  --vae_encode_tiled --vae_decode_tiled
upscale seedvr2-7b-1088 seedvr2_ema_7b_fp16.safetensors 1088 spark-mountain-lake "${common[@]}"
log "done"
