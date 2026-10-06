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

pip install -q ninja packaging wheel setuptools huggingface_hub hf_transfer safetensors diffusers accelerate >"$W/logs/pip.log" 2>&1 || echo "pip failed (see logs/pip.log)"

# our kernels (same nvcc options build.rs uses)
ARCH="sm_$SM"; [[ "$SM" == 90 || "$SM" == 100 ]] && ARCH="sm_${SM}a"
( nvcc -cubin -arch "$ARCH" -O3 --fmad=true --prec-div=true --prec-sqrt=true --ftz=false -o kernels.cubin kernels.cu \
  && nvcc -cubin -arch "$ARCH" -O3 -std=c++17 --use_fast_math -o attn_fp8.cubin attn_fp8.cu \
  && { [[ ! -f attn_sage.cu ]] || nvcc -cubin -arch "$ARCH" -O3 -std=c++17 --use_fast_math -Xptxas=-v -o attn_sage.cubin attn_sage.cu; } \
  && echo ours-ok ) >"$W/logs/ours.log" 2>&1 &

# the image has no git: fetch source tarballs
fetch_tgz() {  # url dest
  python - "$1" "$2" <<'PY'
import io, sys, tarfile, urllib.request, os, shutil
url, dest = sys.argv[1], sys.argv[2]
try:
    data = urllib.request.urlopen(url, timeout=300).read()
except Exception as e:  # codeload -> github.com/archive (same tarball)
    alt = url.replace("https://codeload.github.com/", "https://github.com/").replace("/tar.gz/", "/archive/") + ".tar.gz"
    print("retry", alt, "after", e)
    data = urllib.request.urlopen(alt, timeout=300).read()
tmp = dest + ".tmp"; shutil.rmtree(tmp, ignore_errors=True)
tarfile.open(fileobj=io.BytesIO(data)).extractall(tmp)
(top,) = os.listdir(tmp); shutil.rmtree(dest, ignore_errors=True); os.rename(os.path.join(tmp, top), dest); os.rmdir(tmp)
print("fetched", url, len(data))
PY
}
if [[ "${SAGE_UPSTREAM:-1}" == 1 ]]; then
fetch_tgz "https://codeload.github.com/thu-ml/SageAttention/tar.gz/$SAGE_REV" sage
( cd sage && pip install -v --no-build-isolation . && echo sage2-ok ) >"$W/logs/sage2.log" 2>&1 &
S2=$!
if [[ "$SM" == 120 || "$SM" == 121 || "$SM" == 100 ]]; then
  # CUTLASS pinned near the SageAttention3 release (v4.2.0); main as the fallback.
  ( cd sage/sageattention3_blackwell && export MAX_JOBS=$((NP / 2 > 1 ? NP / 2 : 1)) \
    && { { fetch_tgz https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/v4.2.0 csrc/cutlass \
           && pip install -v --no-build-isolation . ; } \
         || { echo "== retry with cutlass main"; rm -rf csrc/cutlass build \
           && fetch_tgz https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/heads/main csrc/cutlass \
           && pip install -v --no-build-isolation . ; } ; } && echo sage3-ok ) >"$W/logs/sage3.log" 2>&1 &
  S3=$!
fi
fi
( HF_HUB_ENABLE_HF_TRANSFER=1 python -c "
from huggingface_hub import snapshot_download
snapshot_download('FastVideo/FastWan2.1-T2V-1.3B-Diffusers', allow_patterns=['transformer/*'], local_dir='$W/fastwan')
" && echo fetch-ok ) >"$W/logs/fetch.log" 2>&1 &
wait
tail -n1 "$W"/logs/*.log
