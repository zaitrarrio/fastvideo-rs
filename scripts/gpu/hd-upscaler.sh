#!/usr/bin/env bash
# Upscaler benchmark for the hd matrix family (docs/serve/h3-1080p-and-upscaler.md).
# Runs ON the pod after the hd cells, before their PNG frames are pruned
# (FV_HD_POST_URL; runpod-http.sh FV_POD_FILES can ship it as a file:// URL):
#   bash hd-upscaler.sh <RUNS>        (FV_BIN, FV_LPIPS_ARGS from runpod-matrix.sh)
# It runs as the matrix cell `upscaler`: with FV_CELLS set, list it there
# (e.g. FV_CELLS="turbo-768p turbo-1080p upscaler").
# SeedVR2 3B fp16 (one-step diffusion VSR, Apache-2.0) through the numz
# standalone CLI (pinned, SDPA) on the turbo-768p clips: 768p -> 1088 short
# side (resized to the 1920x1088 native canvas) for every clip plus a warm
# repeat, and 768p -> 1440 short side. The weights are read from the volume
# (auxiliary/upscalers/seedvr2, verify-weights.sh upscalers) through symlinks
# in a container-disk model directory (the CLI writes a validation cache next
# to its models); Python, torch and all outputs live on the container disk.
# Each run records wall time, the CLI's processing time and peak GPU memory
# (nvidia-smi every 0.5 s). Then hd_upscaler_metrics.py (fetched next to this
# script) scores Lanczos, SeedVR2 and native 1080p (sharpness, detail above
# the 768p band, flow-warped temporal flicker, fidelity to the 768p input),
# and compare-clips runs SeedVR2 against the Lanczos baseline of each clip.
set -uo pipefail
RUNS="${1:?runs dir}"
BIN="${FV_BIN:-/opt/fastvideo-rs/target/release/fv-gpucheck}"
W="${FV_WEIGHTS:-${FV_WORK:-/workspace}/weights}"
UP=/root/hd-up
SV_SHA="${FV_SEEDVR2_SHA:-4490bd1f482e026674543386bb2a4d176da245b9}"
SV_W="$W/auxiliary/upscalers/seedvr2"
OUT="$RUNS/upscaler"
CLIPS="${FV_HD_UP_CLIPS:-talking-head spark-mountain-lake ltx-frogyoga}"
mkdir -p "$UP/models" "$OUT" "$RUNS/compare"
log() { printf '[%s] [upscaler] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$OUT/upscaler.log" >>"$RUNS/live.log"; }

# Weights: present with the pinned sizes (the CLI checks their SHA-256 itself).
for f in seedvr2_ema_3b_fp16.safetensors:6783018808 ema_vae_fp16.safetensors:501324814; do
  n="${f%%:*}"; sz="${f##*:}"
  got="$(stat -c %s "$SV_W/$n" 2>/dev/null || echo 0)"
  if [[ ! -f "$SV_W/.complete" || "$got" != "$sz" ]]; then
    log "weights missing or wrong size: $SV_W/$n ($got, want $sz); stop"
    exit 1
  fi
  ln -sf "$SV_W/$n" "$UP/models/$n"
done

log "setup: uv + python 3.12 + torch cu128 + SeedVR2 CLI @ ${SV_SHA:0:8}"
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
  uv pip install -q -r "$UP/seedvr2/requirements.txt" scipy
  python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
} >"$OUT/setup.log" 2>&1
rc=$?
log "setup exit=$rc $(( $(date +%s) - t0 ))s $(tail -1 "$OUT/setup.log")"
[[ $rc -eq 0 ]] || exit 1
export PATH="$UP/bin:$PATH"
. "$UP/venv/bin/activate"

# name resolution clip canvas(WxH|raw) [extra CLI args...]
upscale() {
  local name="$1" res="$2" clip="$3" canvas="$4"
  shift 4
  local src="$RUNS/turbo-768p/frames/$clip" dir="$RUNS/$name" mem="$OUT/$name-$clip.mem"
  [[ -f "$src/frame-000.png" ]] || { log "skip $name/$clip: no frames"; return 0; }
  mkdir -p "$dir/raw/$clip" "$dir/frames/$clip"
  # Near-lossless input (the CLI reads a video file).
  [[ -f "$UP/in-$clip.mp4" ]] || ffmpeg -nostdin -loglevel error -y -framerate 24 -start_number 0 -i "$src/frame-%03d.png" \
    -c:v libx264 -crf 8 -pix_fmt yuv444p "$UP/in-$clip.mp4"
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) >"$mem" 2>/dev/null &
  local smi=$! t1 t2 rc
  t1=$(date +%s.%N)
  python "$UP/seedvr2/inference_cli.py" "$UP/in-$clip.mp4" --output "$dir/raw/$clip" --output_format png \
    --model_dir "$UP/models" --dit_model seedvr2_ema_3b_fp16.safetensors --resolution "$res" "$@" \
    >"$OUT/$name-$clip.log" 2>&1
  rc=$?
  t2=$(date +%s.%N)
  kill "$smi" 2>/dev/null
  wait "$smi" 2>/dev/null
  local peak secs n size proc fps
  peak="$(sort -n "$mem" | tail -1)"
  secs="$(awk -v a="$t1" -v b="$t2" 'BEGIN{printf "%.1f", b-a}')"
  n="$(find "$dir/raw/$clip" -name '*.png' | wc -l)"
  size="$(find "$dir/raw/$clip" -name '*.png' | sort | head -1 | xargs -r python -c 'import sys; from PIL import Image; print("%dx%d" % Image.open(sys.argv[1]).size)' 2>/dev/null)"
  proc="$(grep -oE 'Processing time: [0-9.]+' "$OUT/$name-$clip.log" | tail -1 | grep -oE '[0-9.]+$')"
  fps="$(grep -oE 'Average FPS: [0-9.]+' "$OUT/$name-$clip.log" | tail -1 | grep -oE '[0-9.]+$')"
  log "$name/$clip exit=$rc wall=${secs}s processing=${proc:-?}s fps=${fps:-?} peak_gpu_mib=$peak frames=$n size=$size"
  printf '{"cell":"%s","clip":"%s","resolution":%s,"args":"%s","exit":%s,"wall_s":%s,"processing_s":%s,"avg_fps":%s,"peak_gpu_mib":%s,"frames":%s,"size":"%s"}\n' \
    "$name" "$clip" "$res" "$*" "$rc" "$secs" "${proc:-null}" "${fps:-null}" "${peak:-null}" "$n" "$size" >>"$OUT/runs.jsonl"
  [[ $rc -eq 0 && $n -gt 0 ]] || return 0
  # The CLI writes <raw>/<clip>/in-<clip>/<name>_NNNNNN.png: renumber them in
  # order (an ffmpeg concat list of PNGs kept only 4 frames), then resize.
  local seq="$UP/seq-$name-$clip" i=0 f
  rm -rf "$seq"; mkdir -p "$seq"
  while IFS= read -r f; do
    mv "$f" "$seq/$(printf 'frame-%03d.png' "$i")"; i=$((i + 1))
  done < <(find "$dir/raw/$clip" -name '*.png' | sort)
  if [[ "$canvas" == raw ]]; then
    mv "$seq"/frame-*.png "$dir/frames/$clip/"
  else
    ffmpeg -nostdin -loglevel error -y -start_number 0 -i "$seq/frame-%03d.png" \
      -vf "scale=${canvas/x/:}:flags=lanczos" -start_number 0 "$dir/frames/$clip/frame-%03d.png"
  fi
  log "$name/$clip: $(find "$dir/frames/$clip" -name 'frame-*.png' | wc -l) frames on the canvas"
  rm -rf "$dir/raw/$clip" "$seq"
  # An H.264 copy for viewing (fetched; the PNGs are pruned to keyframes).
  ffmpeg -nostdin -loglevel error -y -framerate 24 -start_number 0 -i "$dir/frames/$clip/frame-%03d.png" \
    -c:v libx264 -crf 16 -pix_fmt yuv420p "$dir/$clip.mp4"
}

common=(--batch_size 33 --uniform_batch_size --temporal_overlap 3 --seed 42)
first="${FV_HD_UP_WARM:-1}"
for clip in $CLIPS; do
  upscale seedvr2-3b-1088 1088 "$clip" 1920x1088 "${common[@]}"
  if [[ $first == 1 ]]; then
    # The first run hashes the weights (validation cache) and warms the page
    # cache; repeat it for a warm number.
    log "warm repeat"
    upscale seedvr2-3b-1088-warm 1088 "$clip" 1920x1088 "${common[@]}"
    first=0
  fi
done
for clip in ${FV_HD_UP_1440_CLIPS-$CLIPS}; do
  upscale seedvr2-3b-1440 1440 "$clip" raw "${common[@]}"
  if [[ ! -f "$RUNS/seedvr2-3b-1440/frames/$clip/frame-000.png" ]]; then
    log "1440 $clip failed untiled; retry with VAE tiling"
    upscale seedvr2-3b-1440 1440 "$clip" raw "${common[@]}" --vae_encode_tiled --vae_decode_tiled
  fi
done

# Lanczos baselines at the SeedVR2 1440 output size.
for clip in $CLIPS; do
  f="$RUNS/seedvr2-3b-1440/frames/$clip/frame-000.png"
  [[ -f "$f" ]] || continue
  wh="$(python -c 'import sys; from PIL import Image; print("%d:%d" % Image.open(sys.argv[1]).size)' "$f")"
  mkdir -p "$RUNS/turbo-768p-lanczos1440/frames/$clip"
  ffmpeg -nostdin -loglevel error -y -start_number 0 -i "$RUNS/turbo-768p/frames/$clip/frame-%03d.png" \
    -vf "scale=$wh:flags=lanczos" -start_number 0 "$RUNS/turbo-768p-lanczos1440/frames/$clip/frame-%03d.png"
done

# compare-clips: SeedVR2 against the Lanczos baseline of the same clip (same
# content, so LPIPS / PSNR are meaningful here).
for pair in turbo-768p-lanczos:seedvr2-3b-1088 turbo-768p-lanczos1440:seedvr2-3b-1440; do
  b="${pair%%:*}"; c="${pair##*:}"
  for clip in $CLIPS; do
    [[ -f "$RUNS/$b/frames/$clip/frame-000.png" && -f "$RUNS/$c/frames/$clip/frame-000.png" ]] || continue
    # shellcheck disable=SC2086
    "$BIN" --out "$RUNS/compare" --tag "$b--$c-$clip" compare-clips \
      --baseline "$RUNS/$b/frames/$clip" --candidate "$RUNS/$c/frames/$clip" ${FV_LPIPS_ARGS:-} \
      >"$RUNS/compare/$b--$c-$clip.stdout.log" 2>"$RUNS/compare/$b--$c-$clip.stderr.log"
    log "compare $b--$c/$clip exit=$?"
  done
done

# FlashVSR v1.1 (FV_HD_FLASHVSR=1): the official pipeline needs the
# Block-Sparse-Attention extension, built here for this GPU's arch against
# torch cu128 with a CUDA 12.8 nvcc (apt). Best effort under its own caps
# (FV_HD_FVSR_BUILD_S, FV_HD_FVSR_RUN_S); a failed build is a result.
FV_W="$W/auxiliary/upscalers/flashvsr-v1.1"
if [[ "${FV_HD_FLASHVSR:-0}" == 1 && -f "$FV_W/.complete" ]]; then
  FVSR_SHA="${FV_FLASHVSR_SHA:-cf910c61a60733e610e9c6e8b607f80c3a6c202b}"
  BSA_SHA="${FV_BSA_SHA:-49d6c39e4dc0303442cda3bb758b3925d4399c49}"
  arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)"
  log "flashvsr: setup (FlashVSR ${FVSR_SHA:0:8}, Block-Sparse-Attention ${BSA_SHA:0:8}, sm_$arch)"
  t0=$(date +%s)
  timeout "${FV_HD_FVSR_BUILD_S:-1500}" bash -c '
    set -euxo pipefail
    UP='"$UP"'; FVSR_SHA='"$FVSR_SHA"'; BSA_SHA='"$BSA_SHA"'; arch='"$arch"'
    export PATH="$UP/bin:$PATH"
    if [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
      pk="cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-cccl-12-8 cuda-libraries-dev-12-8"
      apt-get update -qq || true
      if ! DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $pk >/dev/null; then
        . /etc/os-release
        curl -fsSL -o /tmp/keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID/./}/x86_64/cuda-keyring_1.1-1_all.deb"
        dpkg -i /tmp/keyring.deb
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $pk >/dev/null
      fi
    fi
    /usr/local/cuda-12.8/bin/nvcc --version | tail -2
    uv venv -q -p 3.11 "$UP/fvenv"
    . "$UP/fvenv/bin/activate"
    uv pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    for r in FlashVSR:OpenImagingLab/FlashVSR:$FVSR_SHA Block-Sparse-Attention:mit-han-lab/Block-Sparse-Attention:$BSA_SHA; do
      IFS=: read -r d repo sha <<<"$r"
      rm -rf "$UP/$d"; git init -q "$UP/$d"; git -C "$UP/$d" remote add origin "https://github.com/$repo.git"
      git -C "$UP/$d" fetch -q --depth 1 origin "$sha"; git -C "$UP/$d" checkout -q FETCH_HEAD
    done
    grep -vE "^(torch|torchvision|torchaudio)==" "$UP/FlashVSR/requirements.txt" >"$UP/fvsr-req.txt"
    uv pip install -q -r "$UP/fvsr-req.txt" packaging ninja psutil wheel setuptools
    cd "$UP/Block-Sparse-Attention"
    CUDA_HOME=/usr/local/cuda-12.8 BLOCK_SPARSE_ATTN_CUDA_ARCHS="$arch" MAX_JOBS="${MAX_JOBS:-$(( $(nproc) / 2 ))}" \
      uv pip install --no-build-isolation -v . 2>&1 | tail -40
    # torch first: the extension links against libc10 from the torch wheel.
    cd / && python -c "import torch, block_sparse_attn; print(\"bsa ok\", torch.__version__)"
  ' >"$OUT/flashvsr-setup.log" 2>&1
  rc=$?
  log "flashvsr setup exit=$rc $(( $(date +%s) - t0 ))s $(tail -1 "$OUT/flashvsr-setup.log")"
  if [[ $rc -eq 0 ]] && curl -fsSL "${FV_HD_POST_URL%/*}/hd_flashvsr.py" -o "$UP/hd_flashvsr.py"; then
    mkdir -p "$UP/FlashVSR/examples/WanVSR/FlashVSR-v1.1"
    for f in "$FV_W"/*; do ln -sf "$f" "$UP/FlashVSR/examples/WanVSR/FlashVSR-v1.1/$(basename "$f")"; done
    mem="$OUT/flashvsr.mem"
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) >"$mem" 2>/dev/null &
    smi=$!
    t0=$(date +%s)
    # shellcheck disable=SC2086
    timeout "${FV_HD_FVSR_RUN_S:-900}" "$UP/fvenv/bin/python" "$UP/hd_flashvsr.py" "$UP/FlashVSR" "$RUNS" 1.5,2 $CLIPS \
      >"$OUT/flashvsr-runs.jsonl" 2>"$OUT/flashvsr-run.log"
    rc=$?
    kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
    log "flashvsr runs exit=$rc $(( $(date +%s) - t0 ))s peak_gpu_mib=$(sort -n "$mem" | tail -1)"
    # Lanczos baselines with the same scale and centre crop.
    for clip in $CLIPS; do
      for s in 1.5:2016:1152:1920 2:2688:1536:2688; do
        IFS=: read -r sc sw sh cw <<<"$s"
        [[ -f "$RUNS/flashvsr-v1.1-x$sc/frames/$clip/frame-000.png" ]] || continue
        mkdir -p "$RUNS/lanczos-x$sc/frames/$clip"
        ffmpeg -nostdin -loglevel error -y -start_number 0 -i "$RUNS/turbo-768p/frames/$clip/frame-%03d.png" \
          -vf "scale=$sw:$sh:flags=lanczos,crop=$cw:$sh" -start_number 0 "$RUNS/lanczos-x$sc/frames/$clip/frame-%03d.png"
        ffmpeg -nostdin -loglevel error -y -framerate 24 -start_number 0 -i "$RUNS/flashvsr-v1.1-x$sc/frames/$clip/frame-%03d.png" \
          -c:v libx264 -crf 16 -pix_fmt yuv420p "$RUNS/flashvsr-v1.1-x$sc/$clip.mp4"
      done
    done
  fi
fi

# No-reference metrics (all 124 frames; spectra on every 4th).
if [[ -n "${FV_HD_POST_URL:-}" ]] && curl -fsSL "${FV_HD_POST_URL%/*}/hd_upscaler_metrics.py" -o "$UP/metrics.py"; then
  t0=$(date +%s)
  python "$UP/metrics.py" "$RUNS" $CLIPS >"$OUT/metrics.json" 2>"$OUT/metrics.log"
  log "metrics exit=$? $(( $(date +%s) - t0 ))s"
else
  log "metrics script not found next to $FV_HD_POST_URL"
fi
log "done"
