#!/bin/bash
# Phase 3 (docs/perf/sage-attention.md) on one RTX PRO 6000 pod, EU volume
# mounted read-only at /workspace. Everything written goes under /root/p3.
#   bash phase3.sh setup|a|b|cd|e|lipsync|summary [cells...]
set -u
B=/e2e/p3/fv-gpucheck
W=/workspace/weights
P=/e2e/p3
WK=/root/p3
mkdir -p "$WK"
export RUST_LOG=info FASTVIDEO_CACHE=$WK/cache XDG_CACHE_HOME=$WK/cache
LOG=$WK/progress.log
log() { echo "$(date -u +%FT%TZ) | $*" | tee -a "$LOG"; }
LPIPS=(); [ -f $W/auxiliary/lpips/.complete ] && LPIPS=(--lpips $W/auxiliary/lpips)
POLICY=$P/gate-policy.toml

# run <cell dir> <env...> -- <args...>: one fv-gpucheck process, nvidia-smi peak.
run() {
  local J=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  mkdir -p "$J"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 >"$J/smi.csv" 2>/dev/null &
  local smi=$! t0; t0=$(date +%s)
  ( cd "$J" && env "${envs[@]}" timeout "${CELL_TIMEOUT:-2400}" "$B" "$@" ) >"$J/log.txt" 2>&1
  local rc=$?
  kill $smi 2>/dev/null
  echo "{\"rc\":$rc,\"wall_s\":$(( $(date +%s) - t0 )),\"peak_mib\":$(sort -n "$J/smi.csv" | tail -1)}" >"$J/run.json"
  log "cell $(basename "$J") rc=$rc wall=$(( $(date +%s) - t0 ))s peak=$(sort -n "$J/smi.csv" | tail -1)MiB"
  return $rc
}

# compare <base cell> <cand cell>: per-prompt compare-clips, then the gate.
compare() {
  local b=$1 c=$2 p tag cmp=()
  mkdir -p "$WK/compare" "$WK/gate"
  for p in "$b"/frames/*/; do
    p=$(basename "$p"); [[ $p == cold || $p == warmup ]] && continue
    [ -f "$c/frames/$p/frame-000.png" ] || continue
    tag="$(basename "$b")--$(basename "$c")-$p"
    "$B" --out "$WK/compare" --tag "$tag" compare-clips --baseline "$b/frames/$p" --candidate "$c/frames/$p" "${LPIPS[@]}" \
      >"$WK/compare/$tag.out" 2>&1
    [ -f "$WK/compare/compare-clips-$tag.json" ] && cmp+=(--compare "$WK/compare/compare-clips-$tag.json")
  done
  tag="$(basename "$b")--$(basename "$c")"
  if [ -f "$b/benchmark.json" ] && [ -f "$c/benchmark.json" ]; then
    "$B" --out "$WK/gate" --tag "$tag" gate --baseline "$b" --candidate "$c" --policy "$POLICY" --kind lossy "${cmp[@]}" \
      >"$WK/gate/$tag.out" 2>&1
    log "gate $tag: $(grep -o 'gate verdict: .*' "$WK/gate/$tag.out" | tail -1)"
  fi
}

case "${1:?stage}" in
setup)
  # Python venv for the lip-sync proxy; torch venv for the upscalers (background).
  { python3 -m venv $WK/lsv && $WK/lsv/bin/pip install -q numpy 'opencv-python-headless<5'; } >$WK/lsv.log 2>&1
  log "lipsync venv exit=$?"
  ;;
upsetup)
  # FlashVSR v1.1 official pipeline + Block-Sparse-Attention for this arch, and
  # Real-ESRGAN x2plus (67 MB checkpoint to the container disk). CPU only.
  UP=$WK/up; mkdir -p $UP
  arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
  t0=$(date +%s)
  timeout 2400 bash -c '
    set -euxo pipefail
    UP='"$UP"'; arch='"$arch"'
    command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UP/bin" sh
    export PATH="$UP/bin:$PATH"
    if [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
      pk="cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-cccl-12-8 cuda-libraries-dev-12-8"
      apt-get update -qq || true
      if ! DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $pk >/dev/null; then
        . /etc/os-release
        curl -fsSL -o /tmp/keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID/./}/x86_64/cuda-keyring_1.1-1_all.deb"
        dpkg -i /tmp/keyring.deb; apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends $pk >/dev/null
      fi
    fi
    uv venv -q -p 3.11 "$UP/fvenv"; . "$UP/fvenv/bin/activate"
    uv pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    for r in FlashVSR:OpenImagingLab/FlashVSR:cf910c61a60733e610e9c6e8b607f80c3a6c202b Block-Sparse-Attention:mit-han-lab/Block-Sparse-Attention:49d6c39e4dc0303442cda3bb758b3925d4399c49; do
      IFS=: read -r d repo sha <<<"$r"
      rm -rf "$UP/$d"; git init -q "$UP/$d"; git -C "$UP/$d" remote add origin "https://github.com/$repo.git"
      git -C "$UP/$d" fetch -q --depth 1 origin "$sha"; git -C "$UP/$d" checkout -q FETCH_HEAD
    done
    grep -vE "^(torch|torchvision|torchaudio)==" "$UP/FlashVSR/requirements.txt" >"$UP/fvsr-req.txt"
    uv pip install -q -r "$UP/fvsr-req.txt" packaging ninja psutil wheel setuptools pillow numpy
    cd "$UP/Block-Sparse-Attention"
    CUDA_HOME=/usr/local/cuda-12.8 BLOCK_SPARSE_ATTN_CUDA_ARCHS="$arch" MAX_JOBS=32 \
      uv pip install --no-build-isolation -v . 2>&1 | tail -20
    cd / && python -c "import torch, block_sparse_attn; print(\"bsa ok\", torch.__version__)"
    curl -fsSL -o "$UP/RealESRGAN_x2plus.pth" https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
    sha256sum "$UP/RealESRGAN_x2plus.pth"
  ' >$WK/upsetup.log 2>&1
  log "upscaler setup exit=$? $(( $(date +%s) - t0 ))s: $(tail -1 $WK/upsetup.log)"
  ;;
a)
  run $WK/a --  --out $WK/a/out --keep-going kernels --groups attn_sage,attn_fp8,attn3_bench
  grep -hE 'attn_sage|attn_fp8|attn3|FAIL|PASS' $WK/a/log.txt | cut -c1-300 | tail -60 >$WK/a/summary.txt
  ;;
b)
  # H3 five-prompt gate, 768p, SageAttention2 vs bf16. Each arm its own text
  # and AdaLN caches (symmetric arms: nothing is shared between processes).
  cells="${2:-max dense turbo}"
  for r in $cells; do
    case $r in
      max)   prof=sol_h3_4step_engine_ladder; rec=sol-h3 ;;
      dense) prof=sol_h3_4step; rec=sol-h3 ;;
      turbo) prof=fasth3_4step_vsa; rec=4step-vsa ;;
    esac
    for arm in bf16 sage; do
      c=$WK/b/h3$r-$arm; e=(); [ $arm = sage ] && e=(FASTVIDEO_ATTN_SAGE=2)
      run $c "${e[@]}" -- --mode fast --techniques h3/$prof h3 gen --weights $W/h3-base --h3-recipe $rec \
        --prompt "x" --prompts $P/prompts-eval.json --seconds 5 --warm --text-encoder resident-fp8 \
        --dit-offload resident --text-cache $c/text-cache --text-weights $W/h3-base \
        --adaln-cache $c/adaln.cache --clip-dir $c/frames
    done
    compare $WK/b/h3$r-bf16 $WK/b/h3$r-sage
  done
  ;;
cd)
  # LTX-2.5 two-stage (ltx-turbo: Sol stage 2) at the director canvases, bf16
  # vs Sage; then dense stage 2 at 1080p (the attention-bound control).
  cells="${2:-896x512 1280x768 1408x768 1920x1088 1920x1088d}"
  for g in $cells; do
    wh=${g%d}; extra=(); [ "$g" != "$wh" ] && extra=(--dense-stage2)
    for arm in bf16 sage; do
      c=$WK/cd/ltx$g-$arm; e=(); [ $arm = sage ] && e=(FASTVIDEO_ATTN_SAGE=2)
      run $c "${e[@]}" -- --mode fast --techniques ltx2/ltx25_distill_sol ltx2 gen --model-version 2.5 \
        --weights $W/ltx25 --dit $W/ltx25 --two-stage "${extra[@]}" --prompt x --prompts $P/prompts-ltx3.json \
        --width ${wh%x*} --height ${wh#*x} --num-frames 121 --frame-rate 24 --warm --text resident \
        --dit-offload resident --offload none --no-text-cache --clip $c/frames
    done
    compare $WK/cd/ltx$g-bf16 $WK/cd/ltx$g-sage
  done
  ;;
e)
  # 384p stage-1 only (single stage, 672x384) and the 768p two-stage it is the
  # first stage of (1344x768), bf16, same prompts and seeds.
  [ -f $WK/e/ltx384-s1/benchmark.json ] || run $WK/e/ltx384-s1 -- --mode fast ltx2 gen \
    --model-version 2.5 --weights $W/ltx25 --dit $W/ltx25 --prompt x --prompts $P/prompts-ltx3.json \
    --width 672 --height 384 --num-frames 121 --frame-rate 24 --warm --text resident --dit-offload resident \
    --offload none --no-text-cache --clip $WK/e/ltx384-s1/frames
  [ -f $WK/e/ltx768-2s/benchmark.json ] || run $WK/e/ltx768-2s -- --mode fast --techniques ltx2/ltx25_distill_sol ltx2 gen \
    --model-version 2.5 --weights $W/ltx25 --dit $W/ltx25 --two-stage --prompt x --prompts $P/prompts-ltx3.json \
    --width 1344 --height 768 --num-frames 121 --frame-rate 24 --warm --text resident --dit-offload resident \
    --offload none --no-text-cache --clip $WK/e/ltx768-2s/frames
  UP=$WK/up; clips="ltx-multishot ltx-newsbroadcast ltx-frogyoga"
  # Lanczos x2 (CPU, ffmpeg) for reference.
  for cl in $clips; do
    mkdir -p $WK/e/lanczos-x2/frames/$cl
    t0=$(date +%s.%N)
    ffmpeg -nostdin -loglevel error -y -start_number 0 -i $WK/e/ltx384-s1/frames/$cl/frame-%03d.png \
      -vf scale=1344:768:flags=lanczos -start_number 0 $WK/e/lanczos-x2/frames/$cl/frame-%03d.png
    log "lanczos-x2 $cl $(awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b-a}')s (incl. PNG io)"
  done
  if [ -f $UP/RealESRGAN_x2plus.pth ]; then
    srcs=(); for cl in $clips; do srcs+=($WK/e/ltx384-s1/frames/$cl); done
    $UP/fvenv/bin/python $P/esrgan_x2.py $UP/RealESRGAN_x2plus.pth $WK/e/esrgan-x2/frames "${srcs[@]}" \
      >$WK/e/esrgan.jsonl 2>$WK/e/esrgan.log
    log "esrgan exit=$? $(tail -1 $WK/e/esrgan.jsonl)"
  else
    log "esrgan skipped: no checkpoint (see upsetup.log)"
  fi
  if $UP/fvenv/bin/python -c "import torch, block_sparse_attn" 2>/dev/null; then
    # hd_flashvsr.py reads <RUNS>/turbo-768p/frames/<clip>; point that at the 384p frames.
    R=$WK/e/fvsr-runs; mkdir -p $R/turbo-768p/frames
    for cl in $clips; do ln -sfn $WK/e/ltx384-s1/frames/$cl $R/turbo-768p/frames/$cl; done
    mkdir -p $UP/FlashVSR/examples/WanVSR/FlashVSR-v1.1
    for f in $W/auxiliary/upscalers/flashvsr-v1.1/*; do ln -sf "$f" $UP/FlashVSR/examples/WanVSR/FlashVSR-v1.1/$(basename "$f"); done
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 >$WK/e/fvsr.smi 2>/dev/null &
    smi=$!
    timeout 1500 $UP/fvenv/bin/python $P/hd_flashvsr.py $UP/FlashVSR $R 2 $clips >$WK/e/flashvsr.jsonl 2>$WK/e/flashvsr.log
    rc=$?; kill $smi 2>/dev/null
    log "flashvsr exit=$rc peak=$(sort -n $WK/e/fvsr.smi | tail -1)MiB $(tail -1 $WK/e/flashvsr.jsonl)"
  else
    log "flashvsr skipped: block_sparse_attn import failed (see upsetup.log)"
  fi
  # Each upscale row against LTX's own upsampler + refine (1344x768; FlashVSR
  # output is centre-cropped to 1280x768, so the reference is cropped alike).
  mkdir -p $WK/compare
  for cl in $clips; do
    for row in lanczos-x2:$WK/e/lanczos-x2/frames/$cl esrgan-x2:$WK/e/esrgan-x2/frames/$cl fvsr-x2:$WK/e/fvsr-runs/flashvsr-v1.1-x2/frames/$cl; do
      n=${row%%:*}; d=${row#*:}; ref=$WK/e/ltx768-2s/frames/$cl
      [ -f $d/frame-000.png ] || continue
      if [ $n = fvsr-x2 ]; then
        ref=$WK/e/ref1280/$cl; mkdir -p $ref
        [ -f $ref/frame-000.png ] || ffmpeg -nostdin -loglevel error -y -start_number 0 -i $WK/e/ltx768-2s/frames/$cl/frame-%03d.png \
          -vf crop=1280:768 -start_number 0 $ref/frame-%03d.png
      fi
      "$B" --out $WK/compare --tag "ltx768-2s--$n-$cl" compare-clips --baseline $ref --candidate $d "${LPIPS[@]}" \
        >$WK/compare/ltx768-2s--$n-$cl.out 2>&1
      log "compare ltx768-2s--$n-$cl exit=$?"
    done
  done
  ;;
ctl)
  # Noise-floor control: bf16 again, with the dense kernel forced from cuDNN to
  # fwd2 (no composite plan exists, so auto falls back): an ulp-level change.
  r=${2:-dense}
  case $r in
    max)   prof=sol_h3_4step_engine_ladder; rec=sol-h3 ;;
    dense) prof=sol_h3_4step; rec=sol-h3 ;;
    turbo) prof=fasth3_4step_vsa; rec=4step-vsa ;;
  esac
  c=$WK/b/h3$r-ctl
  run $c FASTVIDEO_CUDNN_SDPA_GRAPH=composite -- --mode fast --techniques h3/$prof h3 gen --weights $W/h3-base --h3-recipe $rec \
    --prompt "x" --prompts $P/prompts-eval.json --seconds 5 --warm --text-encoder resident-fp8 \
    --dit-offload resident --text-cache $c/text-cache --text-weights $W/h3-base \
    --adaln-cache $c/adaln.cache --clip-dir $c/frames
  compare $WK/b/h3$r-bf16 $WK/b/h3$r-ctl
  ;;
lipsync)
  # Pooled audio-visual sync per arm over the speech prompts.
  for r in ${2:-max dense turbo}; do
    for arm in bf16 sage; do
      mp4s=""
      for pr in h3-demo ltx-newsbroadcast; do
        d=$WK/b/h3$r-$arm/frames/$pr; m=$(ls $d/*.mp4 2>/dev/null | head -1)
        if [ -z "$m" ] && [ -f $d/audio.wav ]; then
          m=$WK/b/h3$r-$arm/$pr.mp4
          ffmpeg -nostdin -loglevel error -y -framerate 24 -start_number 0 -i $d/frame-%03d.png -i $d/audio.wav \
            -c:v libx264 -crf 12 -pix_fmt yuv420p -c:a aac -b:a 192k -shortest $m
        fi
        [ -n "$m" ] && mp4s="$mp4s $m"
      done
      [ -n "$mp4s" ] || { log "lipsync h3$r-$arm: no mp4"; continue; }
      $WK/lsv/bin/python $P/lipsync_proxy.py --json $WK/b/h3$r-$arm/lipsync.json --pool h3$r-$arm $mp4s >$WK/b/h3$r-$arm/lipsync.out 2>&1
      log "lipsync h3$r-$arm exit=$?"
    done
  done
  ;;
esac
