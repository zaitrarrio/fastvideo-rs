#!/usr/bin/env bash
# Fails (and names what is missing) unless this container has what a CI job
# needs. The jobs run in the build base image (docker/build-base.Dockerfile,
# pinned by scripts/dev/build-base-tag.sh) and install nothing at run time.
#
#   check-build-base.sh compat     ffmpeg with libvpx, python3 + venv, node, npm, openssl
#   check-build-base.sh browser    compat + Playwright's Chromium under PLAYWRIGHT_BROWSERS_PATH
#   check-build-base.sh cuda       nvcc, NVRTC, tileiras of CUDA 13.4
#   check-build-base.sh prebuilt   what scripts/ci/prebuilt.sh / tools-release.sh
#                                  need to fetch and verify a tools release
set -uo pipefail
missing=()
need() { command -v "$1" >/dev/null || missing+=("$1"); }
compat() {
  need ffmpeg; need python3; need node; need npm; need openssl
  # Captured first: `| grep -q` under pipefail fails when ffmpeg gets SIGPIPE.
  if command -v ffmpeg >/dev/null; then
    local enc
    enc="$(ffmpeg -hide_banner -encoders 2>/dev/null)"
    [[ "$enc" == *" libvpx "* ]] || missing+=("ffmpeg libvpx encoder")
  fi
  python3 -c 'import venv, ensurepip' 2>/dev/null || missing+=("python3 venv/ensurepip")
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null || missing+=("python3 >= 3.11 (tests/compat/requirements.txt)")
}
for p in "$@"; do
  case "$p" in
    compat) compat ;;
    browser)
      compat
      d="${PLAYWRIGHT_BROWSERS_PATH:-}"
      [[ -n "$d" ]] && compgen -G "$d/chromium*" >/dev/null || missing+=("Playwright Chromium (PLAYWRIGHT_BROWSERS_PATH=${d:-unset})") ;;
    cuda)
      c="${CUDA_HOME:-/usr/local/cuda-13.4}"
      for f in bin/nvcc bin/tileiras lib64/libnvrtc.so include/nvrtc.h; do [[ -e "$c/$f" ]] || missing+=("$c/$f"); done ;;
    prebuilt)
      for t in git curl jq sha256sum tar gzip find diff sort; do need "$t"; done ;;
    *) echo "check-build-base.sh: unknown profile $p" >&2; exit 2 ;;
  esac
done
if (( ${#missing[@]} )); then
  echo "::error::missing from the build base image ($(jq -r .rust /etc/fastvideo/build-base.json 2>/dev/null || echo 'no /etc/fastvideo/build-base.json')): ${missing[*]}. Jobs install nothing at run time: add it to docker/build-base.Dockerfile and re-pin (scripts/dev/build-base-tag.sh --pin)." >&2
  exit 1
fi
echo "build base image ok for: $*"
cat /etc/fastvideo/build-base.json 2>/dev/null || true
