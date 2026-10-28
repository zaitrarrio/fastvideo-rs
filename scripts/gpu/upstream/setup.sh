#!/usr/bin/env bash
# Pod-side installers for the upstream (Python) references, into persistent
# venvs on the weight volume so the install cost is paid once:
#
#   $UP/fastvideo        FastVideo (hao-ai-lab/FastVideo @ FV_REV), torch 2.12 cu130
#   $UP/sol-h3-rtx5090   sol-engine MiniMax-H3 RTX5090 profile: pinned SGLang
#                        checkout + Sol-Attn (techniques/sparse_backends), torch 2.11 cu130
#   $UP/sol-h3-4step     sol-engine Sol-H3 package (models/minimax_h3/Sol-H3/requirements.txt)
#   $UP/sol-ltx25        sol-engine LTX-2.5 RTX5090: Lightricks/LTX-2 @ fd4ded7 (uv sync)
#                        + nvidia-cutlass-dsl + cuda-python + Sol-Attn
#
# Every installer is idempotent: a venv that already imports its stack and
# carries a matching .stamp is left alone. Sourced by pod.sh.
set -uo pipefail

UP="${UP:-/workspace/upstream}"
SRC="$UP/src"
# Revisions the references pin (see each reference's README / SOURCE_SNAPSHOT.json).
SOL_REPO="https://github.com/NVlabs/Sana.git"
SOL_REV="${SOL_REV:-6c2f582ba9681dfe352ae4ca23b6f0030cf120cf}"          # branch sol-engine
SGLANG_REV="${SGLANG_REV:-6fa3f9df11c8bdbc0e3b4ddc87a3d873343aca72}"    # RTX5090 SOURCE_SNAPSHOT
LTX2_REV="${LTX2_REV:-fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca}"        # ltx25/RTX5090 README
# PyAV of the sol-ltx25 stack, pinned exactly (not overridable): LTX-2 fd4ded7 ships no lockfile and
# ltx-pipelines asks for "av" unpinned, so `uv sync` took whatever PyAV was newest. PyAV 19 (FFmpeg 9) changed
# what upstream's decode_audio_from_file (media_io/decode.py:240-300) reads after its container.seek(0) on AAC
# with encoder priming: the first decoded frame is pts 1024, so the source audio starts 1024 samples (23 ms at
# 44.1 kHz) late, and FFmpeg 9 also drops the last frame's 32 padding samples. The retake fixture then decodes
# to [2, 221152] where PyAV 15.1-18.0 (and the reference of 2026-09-29, `v2v_audio_wave` n 444416) give
# [2, 222208]. 18.0.0 is the newest release measured to decode as before (ADR-0019 §20.10); check_ltx25_av
# holds the venv to it.
LTX25_AV_VERSION=18.0.0
# The source clip of the retake / extend references (oracle.sh LTX_EDIT_SOURCE_NAME): check_ltx25_av decodes
# it. Beside this file in the repo (pod.sh); the image build bind-mounts it at /opt/fixtures (upstream.Dockerfile).
LTX25_AUDIO_FIXTURE="${LTX25_AUDIO_FIXTURE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../fixtures/beach-push-768x512-24fps.mp4}"
FV_REPO="${FV_REPO:-https://github.com/hao-ai-lab/FastVideo.git}"
FV_REV="${FV_REV:-e90be598e56138af5c82590f0d72f6fa2dfce400}"            # FastVideo main, 2026-09-24
# FastH3 NVFP4 on sm_120 (docs/inference/fasth3_rtx_pro_6000.md upstream; docs/ports/fasth3-nvfp4.md here):
# FastVideo main with the ModelOpt NVFP4 conversion and the block-sparse FP4 attention, and the CUTLASS commit its
# Modal driver (scripts/benchmarks/minimax_h3_pro6000/app.py) builds fastvideo-kernel against.
FV_NVFP4_REV="${FV_NVFP4_REV:-d5253fd71055927f12aa6c4aa679b93a00238851}"   # FastVideo main, 2026-10-09
CUTLASS_REV="${CUTLASS_REV:-e67e63c331d6e4b729047c95cf6b92c8454cba89}"

export UV_PYTHON_INSTALL_DIR="$UP/python"   # interpreters on the volume: venvs survive the pod
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"   # container disk: keep small-file churn off the shared volume
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=300
export PATH="$UP/bin:$HOME/.local/bin:$PATH"

slog() { printf '[%s] [setup] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "${LIVE:-/dev/null}" >&2; }

ensure_base() {
  mkdir -p "$UP/bin" "$SRC"
  local need=()
  command -v git >/dev/null || need+=(git)
  command -v gcc >/dev/null || need+=(gcc g++)
  command -v cmake >/dev/null || need+=(cmake)
  command -v ninja >/dev/null || need+=(ninja-build)
  command -v ffmpeg >/dev/null || need+=(ffmpeg)
  command -v jq >/dev/null || need+=(jq)
  if (( ${#need[@]} )); then
    slog "apt install ${need[*]}"
    (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${need[@]}") >/dev/null 2>&1 \
      || slog "WARN apt install failed for ${need[*]}"
  fi
  if [[ ! -x "$UP/bin/uv" ]]; then
    slog "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UP/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1 \
      || { slog "uv install failed"; return 1; }
  fi
  uv --version >&2
}

# checkout <url> <rev> <dir>: shallow fetch of one commit.
checkout() {
  local url="$1" rev="$2" dir="$3"
  if [[ -d "$dir/.git" ]] && [[ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" == "$rev" ]]; then
    return 0
  fi
  slog "checkout $url @ $rev → $dir"
  rm -rf "$dir"
  git init -q "$dir" && git -C "$dir" remote add origin "$url" \
    && git -C "$dir" fetch -q --depth 1 origin "$rev" && git -C "$dir" checkout -q FETCH_HEAD
}

venv_new() {
  local v="$1" py="${2:-3.12}"
  [[ -x "$v/bin/python" ]] && "$v/bin/python" -c 'import sys' 2>/dev/null && return 0
  rm -rf "$v"
  uv venv -q --python "$py" --seed "$v"
}

# Import smoke test: full imports with a GPU; without one (image builds) Triton
# cannot pick a driver at import, so only resolve the modules.
smoke() {
  local py="$1"; shift
  if nvidia-smi -L >/dev/null 2>&1; then
    "$py" -c "import torch; print('torch', torch.__version__, torch.version.cuda); import $*" || return 1
  else
    "$py" -c "import importlib.util as u, torch; print('torch', torch.__version__, torch.version.cuda); m=[x for x in '$*'.replace(',', ' ').split() if u.find_spec(x) is None]; assert not m, m" || return 1
  fi
}

stamp_ok() { [[ -f "$1/.stamp" && "$(cat "$1/.stamp")" == "$2" ]]; }

sources() {
  checkout "$SOL_REPO" "$SOL_REV" "$SRC/sol-engine" || return 1
}

install_fastvideo() {
  local v="$UP/fastvideo" stamp="fastvideo:$FV_REV:v1"
  stamp_ok "$v" "$stamp" && { slog "fastvideo venv ok"; return 0; }
  checkout "$FV_REPO" "$FV_REV" "$SRC/FastVideo" || return 1
  venv_new "$v" 3.12 || return 1
  slog "installing FastVideo[fasth3] (torch cu130; PyPI fastvideo-kernel wheel, no sm_100a build)"
  # --no-sources: the fasth3 extra otherwise builds fastvideo-kernel from the
  # checkout for sm_100a; on sm_120 the PyPI wheel (Triton VSA) is the route.
  (cd "$SRC/FastVideo" && uv pip install -q --python "$v/bin/python" --torch-backend cu130 \
      --prerelease allow --no-sources -e ".[fasth3]") || return 1
  smoke "$v/bin/python" fastvideo || return 1
  echo "$stamp" >"$v/.stamp"
}

# FastVideo @ FV_NVFP4_REV as its RTX PRO 6000 FastH3 NVFP4 page runs it: the package with --no-sources (torch
# 2.12 cu130), then fastvideo-kernel built from the checkout for sm_120a with the SageAttention3 FP4 kernels
# (attn_qat_infer: dense and block-sparse FP4 attention) and without the Hopper TK kernels, against CUTLASS_REV.
# The build needs nvcc (CUDA 12.8+) but no GPU: the arch is explicit. FlashInfer (mm_fp4, nvfp4_quantize) JITs
# its kernels on first use, which also needs nvcc at run time.
install_fastvideo_nvfp4() {
  local v="$UP/fastvideo-nvfp4" stamp="fastvideo-nvfp4:$FV_NVFP4_REV:cutlass:$CUTLASS_REV:v1"
  local src="$SRC/FastVideo-nvfp4"
  stamp_ok "$v" "$stamp" && { slog "fastvideo-nvfp4 venv ok"; return 0; }
  checkout "$FV_REPO" "$FV_NVFP4_REV" "$src" || return 1
  venv_new "$v" 3.12 || return 1
  slog "installing FastVideo @ $FV_NVFP4_REV (torch cu130, --no-sources)"
  (cd "$src" && uv pip install -q --python "$v/bin/python" --torch-backend cu130 --prerelease allow --no-sources -e .) || return 1
  uv pip install -q --python "$v/bin/python" 'cmake==3.31.6' ninja 'scikit-build-core>=0.10' pybind11 || return 1
  if [[ ! -d "$SRC/cutlass/.git" ]] || [[ "$(git -C "$SRC/cutlass" rev-parse HEAD 2>/dev/null)" != "$CUTLASS_REV" ]]; then
    checkout https://github.com/NVIDIA/cutlass.git "$CUTLASS_REV" "$SRC/cutlass" || return 1
  fi
  rm -rf "$src/fastvideo-kernel/include/cutlass" && ln -s "$SRC/cutlass" "$src/fastvideo-kernel/include/cutlass"
  mkdir -p "$UP/wheels"
  rm -f "$UP"/wheels/fastvideo_kernel-*.whl
  slog "building fastvideo-kernel (attn_qat_infer, sm_120a)"
  (cd "$src/fastvideo-kernel" && env PATH="$v/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
      CC=gcc CXX=g++ CUDAHOSTCXX=g++ \
      CMAKE_ARGS="-DFASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER=ON -DFASTVIDEO_KERNEL_BUILD_TK=OFF -DGPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=120a" \
      "$v/bin/python" -m pip wheel . --no-build-isolation --no-deps -w "$UP/wheels") || return 1
  uv pip install -q --python "$v/bin/python" --no-deps --reinstall "$UP"/wheels/fastvideo_kernel-*.whl || return 1
  smoke "$v/bin/python" fastvideo, fastvideo_kernel, flashinfer || return 1
  echo "$stamp" >"$v/.stamp"
}

install_sol_h3_rtx5090() {
  local v="$UP/sol-h3-rtx5090" stamp="sglang:$SGLANG_REV:sol:$SOL_REV:v1"
  stamp_ok "$v" "$stamp" && { slog "sol-h3-rtx5090 venv ok"; return 0; }
  sources || return 1
  checkout https://github.com/sgl-project/sglang.git "$SGLANG_REV" "$SRC/sglang" || return 1
  venv_new "$v" 3.12 || return 1
  slog "installing SGLang[diffusion] @ $SGLANG_REV (torch 2.11 cu130)"
  (cd "$SRC/sglang/python" && SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION=0.5.99 uv pip install -q --python "$v/bin/python" \
      --torch-backend cu130 --prerelease allow -e ".[diffusion]") || return 1
  uv pip install -q --python "$v/bin/python" -e "$SRC/sol-engine/techniques/sparse_backends" || return 1
  # registration.py pins these source hashes; check the file without importing it.
  sha256sum "$SRC/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py" \
    | grep -q 5f87319969c446685ee93d422fc34a7c040defb238eff2274d664f2f8310e997 || { slog "sglang minimax_h3.py hash mismatch"; return 1; }
  smoke "$v/bin/python" sglang, sol_attn || return 1
  echo "$stamp" >"$v/.stamp"
}

install_sol_h3_4step() {
  local v="$UP/sol-h3-4step" stamp="solh3:$SOL_REV:v1"
  stamp_ok "$v" "$stamp" && { slog "sol-h3-4step venv ok"; return 0; }
  sources || return 1
  venv_new "$v" 3.12 || return 1
  slog "installing Sol-H3 requirements (torch 2.10 cu130)"
  uv pip install -q --python "$v/bin/python" --torch-backend cu130 \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 || return 1
  uv pip install -q --python "$v/bin/python" --torch-backend cu130 \
    -r "$SRC/sol-engine/models/minimax_h3/Sol-H3/requirements.txt" || return 1
  smoke "$v/bin/python" diffusers, transformers || return 1
  echo "$stamp" >"$v/.stamp"
}

# check_ltx25_av <python> [<fixture>]: the stack's PyAV is LTX25_AV_VERSION, and upstream's own
# decode_audio_from_file (the fd4ded7 file, as ltx25_edit.py calls it: start 0, max_duration 121 / 24) reads the
# retake fixture as [2, 222208] (44.1 kHz AAC, 218 packets of 1024 samples less the 1024 priming; the cap
# round(121 / 24 * 44100) = 222338 does not bite). decode.py is loaded without the package __init__s: they
# import the model blocks (Triton), which cannot import without a GPU, as at image build (smoke).
check_ltx25_av() {
  local py="$1" fx="${2:-$LTX25_AUDIO_FIXTURE}"
  [[ -f "$fx" ]] || { slog "sol-ltx25: no audio fixture at $fx for the PyAV check"; return 1; }
  fx="$(realpath -- "$fx")" || return 1
  (cd / && "$py" -I -B - "$LTX25_AV_VERSION" "$fx") <<'PY'
import importlib
import importlib.util
import os
import sys
import types

import av

want, fixture = sys.argv[1], sys.argv[2]
if av.__version__ != want:
    sys.exit(f"sol-ltx25: PyAV {av.__version__}, pinned {want} (setup.sh LTX25_AV_VERSION)")
root = importlib.util.find_spec("ltx_pipelines").submodule_search_locations[0]
for name, sub in (("ltx_pipelines", ""), ("ltx_pipelines.utils", "utils"),
                  ("ltx_pipelines.utils.media_io", os.path.join("utils", "media_io"))):
    pkg = types.ModuleType(name)
    pkg.__path__ = [os.path.join(root, sub)]
    sys.modules[name] = pkg
decode = importlib.import_module("ltx_pipelines.utils.media_io.decode")
audio = decode.decode_audio_from_file(fixture, "cpu", 0.0, 121 / 24)
shape = tuple(audio.waveform[0].shape) if audio is not None else None
if shape != (2, 222208):
    sys.exit(f"sol-ltx25: decode_audio_from_file({fixture}, 0, 121/24) gave {shape}, want (2, 222208) "
             f"(PyAV {av.__version__}, {decode.__file__})")
print(f"sol-ltx25: PyAV {av.__version__}, decode_audio_from_file {shape} at {audio.sampling_rate} Hz: ok")
PY
}

install_sol_ltx25() {
  local d="$UP/sol-ltx25" stamp="ltx2:$LTX2_REV:sol:$SOL_REV:av:$LTX25_AV_VERSION:v3"
  stamp_ok "$d" "$stamp" && { slog "sol-ltx25 venv ok"; return 0; }
  sources || return 1
  mkdir -p "$d"
  checkout https://github.com/Lightricks/LTX-2.git "$LTX2_REV" "$d/LTX-2" || return 1
  slog "uv sync LTX-2 @ $LTX2_REV"
  (cd "$d/LTX-2" && uv sync -q) || return 1
  local py="$d/LTX-2/.venv/bin/python"
  # The PyAV pin (LTX25_AV_VERSION's note): after the sync, which resolves "av" unpinned.
  uv pip install -q --python "$py" "av==$LTX25_AV_VERSION" || return 1
  # apache-tvm-ffi: the CuTe DSL from_dlpack path (Sol-Attn) imports tvm_ffi.
  uv pip install -q --python "$py" "nvidia-cutlass-dsl>=4.5" cuda-python apache-tvm-ffi || return 1
  uv pip install -q --python "$py" -e "$SRC/sol-engine/techniques/sparse_backends" || return 1
  smoke "$py" ltx_pipelines, sol_attn, tvm_ffi || return 1
  check_ltx25_av "$py" || return 1
  echo "$stamp" >"$d/.stamp"
}
