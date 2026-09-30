#!/usr/bin/env bash
# On a Vast instance from pytorch/pytorch:*-devel (nvcc + torch): build upstream
# SageAttention2 and SageAttention3 for this GPU, our kernels as cubins, and
# fetch the FastWan 1.3B transformer (6 GB) for real Q/K/V. Logs in /root/w/logs.
# Usage: setup.sh [sage_commit]
set -uo pipefail
W=/root/w
mkdir -p "$W/logs"
cd "$W"
SAGE_REV="${1:-d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5}"
CC="$(python -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}.{b}")')"
SM="${CC/./}"
NP="$(nproc)"
echo "gpu cc=$CC nproc=$NP torch=$(python -c 'import torch;print(torch.__version__, torch.version.cuda)') nvcc=$(nvcc --version | tail -1)"
export TORCH_CUDA_ARCH_LIST="$CC" MAX_JOBS="$NP" EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 4"

pip install -q ninja packaging wheel setuptools huggingface_hub hf_transfer safetensors diffusers >"$W/logs/pip.log" 2>&1 || echo "pip failed (see logs/pip.log)"

# our kernels (same nvcc options build.rs uses)
ARCH="sm_$SM"; [[ "$SM" == 90 || "$SM" == 100 ]] && ARCH="sm_${SM}a"
( nvcc -cubin -arch "$ARCH" -O3 --fmad=true --prec-div=true --prec-sqrt=true --ftz=false -o kernels.cubin kernels.cu \
  && nvcc -cubin -arch "$ARCH" -O3 -std=c++17 --use_fast_math -o attn_fp8.cubin attn_fp8.cu \
  && echo ours-ok ) >"$W/logs/ours.log" 2>&1 &

git clone -q https://github.com/thu-ml/SageAttention.git sage && git -C sage checkout -q "$SAGE_REV"
( cd sage && pip install -v --no-build-isolation . && echo sage2-ok ) >"$W/logs/sage2.log" 2>&1 &
S2=$!
if [[ "$SM" == 120 || "$SM" == 121 || "$SM" == 100 ]]; then
  # CUTLASS pinned near the SageAttention3 release (v4.2.0); main as the fallback.
  ( cd sage/sageattention3_blackwell && export MAX_JOBS=$((NP / 2 > 1 ? NP / 2 : 1)) \
    && { { git clone -q --depth 1 --branch v4.2.0 https://github.com/NVIDIA/cutlass.git csrc/cutlass \
           && pip install -v --no-build-isolation . ; } \
         || { echo "== retry with cutlass main"; rm -rf csrc/cutlass build \
           && git clone -q --depth 1 https://github.com/NVIDIA/cutlass.git csrc/cutlass \
           && pip install -v --no-build-isolation . ; } ; } && echo sage3-ok ) >"$W/logs/sage3.log" 2>&1 &
  S3=$!
fi
( HF_HUB_ENABLE_HF_TRANSFER=1 python -c "
from huggingface_hub import snapshot_download
snapshot_download('FastVideo/FastWan2.1-T2V-1.3B-Diffusers', allow_patterns=['transformer/*'], local_dir='$W/fastwan')
" && echo fetch-ok ) >"$W/logs/fetch.log" 2>&1 &
wait
tail -n1 "$W"/logs/*.log
