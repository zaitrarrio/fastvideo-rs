#!/usr/bin/env bash
# Check the Rust NCII guard port (fastvideo_models::ncii) against transformers' scores
# (crates/fastvideo-models/tests/fixtures/ncii_reference.json, written by ncii_reference.py).
# Fetches hfmlsoc/ncii-light-guard-v01 at the pinned revision from the Hub (SHA-256 checked against
# weights-sha256.tsv) into $FV_NCII_DIR (default ~/.cache/fv-ncii) unless it is there, then runs the
# ignored test. CPU only; meant for the build pod:
#   scripts/dev/build-pod.sh run <agent> -- bash scripts/gpu/ncii-check.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DIR="${FV_NCII_DIR:-$HOME/.cache/fv-ncii}"
REV=6a44e8b4b608078db164b0c946947922b2b98eab
mkdir -p "$DIR"
while IFS=$'\t' read -r dest path _bytes hash; do
  [[ "$dest" == auxiliary/ncii-light-guard ]] || continue
  want="${hash#sha256:}"
  f="$DIR/$path"
  if [[ ! -f "$f" ]] || [[ "$(sha256sum "$f" | cut -d' ' -f1)" != "$want" ]]; then
    curl -sSfL --retry 3 -o "$f.part" "https://huggingface.co/hfmlsoc/ncii-light-guard-v01/resolve/$REV/$path"
    got="$(sha256sum "$f.part" | cut -d' ' -f1)"
    [[ "$got" == "$want" ]] || { echo "sha256 mismatch for $path: $got" >&2; exit 1; }
    mv "$f.part" "$f"
  fi
done < "$HERE/weights-sha256.tsv"
cd "$ROOT"
FV_NCII_DIR="$DIR" cargo test --release -p fastvideo-models --lib ncii -- --ignored --nocapture
