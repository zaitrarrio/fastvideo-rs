#!/usr/bin/env bash
# Content tag of the build pod's base image (docker/build-base.Dockerfile,
# docs/dev/build-pod.md "Base image"): sha256 over the image's inputs, so a tag
# names exactly one set of inputs and an unchanged Dockerfile is never rebuilt.
#
#   build-base-tag.sh            print the tag (bb-<16 hex>)
#   build-base-tag.sh --image    print the full image reference
#   build-base-tag.sh --check    exit 1 unless build-pod.sh and every workflow
#                                that runs in the image (container: …build-base:<tag>)
#                                pin this tag
#   build-base-tag.sh --pin      rewrite those pins to this tag
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="ghcr.io/zaitrarrio/fastvideo-rs-build-base"
# Everything the Dockerfile COPYs, plus the Dockerfile itself.
INPUTS=(docker/build-base.Dockerfile scripts/gpu/cuda-13.pins rust-toolchain.toml)
POD_SH="$ROOT/scripts/dev/build-pod.sh"

tag() {
  local f
  for f in "${INPUTS[@]}"; do
    printf '%s\0' "$f"
    sha256sum <"$ROOT/$f" | cut -d' ' -f1
  done | sha256sum | cut -c1-16 | sed 's/^/bb-/'
}
pinned() { sed -n 's/^BASE_IMAGE_TAG="\(bb-[0-9a-f]*\)".*/\1/p' "$POD_SH"; }
# Workflow lines naming the image: "<file>:<line>:<tag>".
wf_pins() { grep -Hno "fastvideo-rs-build-base:bb-[0-9a-f]*" "$ROOT"/.github/workflows/*.yml 2>/dev/null | sed 's/fastvideo-rs-build-base://'; }

t="$(tag)"
case "${1:-}" in
  "") echo "$t" ;;
  --image) echo "$REPO:$t" ;;
  --check)
    p="$(pinned)"
    if [[ "$p" != "$t" ]]; then
      echo "build-pod.sh pins BASE_IMAGE_TAG=\"$p\" but the base image inputs hash to $t;" \
        "run: bash scripts/dev/build-base-tag.sh --pin" >&2
      exit 1
    fi
    bad="$(wf_pins | grep -v ":$t\$" || true)"
    if [[ -n "$bad" ]]; then
      echo "workflows pin another base image tag than $t (run: bash scripts/dev/build-base-tag.sh --pin):" >&2
      echo "$bad" >&2
      exit 1
    fi
    echo "pinned $t (build-pod.sh, $(wf_pins | wc -l) workflow line(s))" ;;
  --pin)
    sed -i "s/^BASE_IMAGE_TAG=\"bb-[0-9a-f]*\"/BASE_IMAGE_TAG=\"$t\"/" "$POD_SH"
    [[ "$(pinned)" == "$t" ]] || { echo "could not pin $t in $POD_SH" >&2; exit 1; }
    sed -i "s/fastvideo-rs-build-base:bb-[0-9a-f]*/fastvideo-rs-build-base:$t/g" "$ROOT"/.github/workflows/*.yml
    echo "pinned $t in scripts/dev/build-pod.sh and $(wf_pins | wc -l) workflow line(s)" ;;
  *) sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
