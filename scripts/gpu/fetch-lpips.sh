#!/usr/bin/env bash
# Download the LPIPS(net="alex") weights `fv-gpucheck compare-clips --lpips`
# and `fv-gpucheck lpips` read, into $1 (default $FV_SCRATCH/lpips, the pod's
# container disk). Idempotent: a file whose SHA-256 already matches is kept.
# The official .pth files are read as-is (fv-gpucheck parses torch's pickle
# formats), so nothing is converted:
#
#   alexnet-owt-7be5be79.pth  torchvision AlexNet IMAGENET1K_V1, the file
#       lpips.pretrained_networks.alexnet(pretrained=True) downloads
#       (244 MB; only the five conv layers are read)
#   lpips_v0.1_alex.pth       LPIPS v0.1 linear heads,
#       richzhang/PerceptualSimilarity lpips/weights/v0.1/alex.pth @ 082bb24
#       (the file the lpips 0.1.4 wheel ships)
#
#   fetch-lpips.sh [dest]
set -euo pipefail
dest="${1:-${FV_SCRATCH:-/fvscratch}/lpips}"
mkdir -p "$dest"

fetch() {
  local name="$1" url="$2" sha="$3" size="$4" out part got
  out="$dest/$name"
  if [[ -f "$out" ]] && [[ "$(sha256sum "$out" | awk '{print $1}')" == "$sha" ]]; then
    echo "lpips ok (cached): $out"
    return 0
  fi
  part="$out.part.$$"
  rm -f "$part"
  curl -fsSL --proto '=https' --retry 5 --retry-delay 3 --retry-all-errors -o "$part" "$url"
  got="$(sha256sum "$part" | awk '{print $1}')"
  if [[ "$got" != "$sha" ]]; then
    echo "lpips FAIL: $name sha256 $got, expected $sha" >&2
    rm -f "$part"
    return 1
  fi
  [[ "$(wc -c <"$part" | tr -d ' ')" == "$size" ]] || { echo "lpips FAIL: $name size" >&2; rm -f "$part"; return 1; }
  mv -f "$part" "$out"
  echo "lpips ok: $out ($size bytes)"
}

rc=0
fetch alexnet-owt-7be5be79.pth https://download.pytorch.org/models/alexnet-owt-7be5be79.pth \
  7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02 244408911 || rc=1
fetch lpips_v0.1_alex.pth \
  https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/082bb24f84c091ea94de2867d34c4544f68e0963/lpips/weights/v0.1/alex.pth \
  df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0 6009 || rc=1
exit $rc
