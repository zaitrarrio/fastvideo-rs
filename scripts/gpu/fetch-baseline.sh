#!/usr/bin/env bash
# Fetch the fv-gpucheck binary of an earlier commit's runtime image
# (ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-<sha>) without docker: the OCI
# registry API, anonymous pull, and only the small layer that holds the
# binary. The `techniques` matrix family runs every cell with this binary
# and with the image's own, then compares the clips byte for byte.
#
#   fetch-baseline.sh <sha7> <dest dir>   -> <dest>/fv-gpucheck
#
# Needs curl, tar and gzip (all in the runtime image; no jq, no python).
set -euo pipefail
sha="${1:?usage: fetch-baseline.sh <sha7> <dest>}"
dest="${2:?usage: fetch-baseline.sh <sha7> <dest>}"
repo="${FV_BASELINE_REPO:-zaitrarrio/fastvideo-rs-runtime}"
path="opt/fastvideo-rs/target/release/fv-gpucheck"
mkdir -p "$dest"
if [[ -x "$dest/fv-gpucheck" ]]; then
  echo "baseline ok (cached): $dest/fv-gpucheck"
  exit 0
fi
tok="$(curl -sS "https://ghcr.io/token?scope=repository:$repo:pull" | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')"
[[ -n "$tok" ]] || { echo "fetch-baseline: no registry token" >&2; exit 1; }
reg() { curl -sS --fail -L -H "Authorization: Bearer $tok" "$@"; }
index="$(reg -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json' \
  "https://ghcr.io/v2/$repo/manifests/sha-$sha")"
# An index lists the amd64 image first (then its attestation); a plain
# manifest has "layers" already.
if grep -q '"layers"' <<<"$index"; then
  manifest="$index"
else
  digest="$(tr -d '\n ' <<<"$index" | grep -o '"digest":"sha256:[0-9a-f]*","size":[0-9]*,"platform":{"architecture":"amd64"' | head -1 | sed 's/"digest":"\([^"]*\)".*/\1/')"
  [[ -n "$digest" ]] || { echo "fetch-baseline: no amd64 manifest for sha-$sha" >&2; exit 1; }
  manifest="$(reg -H 'Accept: application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json' \
    "https://ghcr.io/v2/$repo/manifests/$digest")"
fi
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
# Layers under 64 MB, newest first: the binary is copied late in the Dockerfile.
for layer in $(tr -d '\n ' <<<"$manifest" | grep -o '"digest":"sha256:[0-9a-f]*","size":[0-9]*' | tac); do
  d="$(sed 's/"digest":"\([^"]*\)".*/\1/' <<<"$layer")"
  size="$(sed 's/.*"size":\([0-9]*\)/\1/' <<<"$layer")"
  (( size < 64 * 1024 * 1024 )) || continue
  reg "https://ghcr.io/v2/$repo/blobs/$d" -o "$tmp/layer"
  if tar -xzf "$tmp/layer" -C "$tmp" "$path" 2>/dev/null; then
    install -m 0755 "$tmp/$path" "$dest/fv-gpucheck"
    tar -xzf "$tmp/layer" -C "$tmp" "$path.build-id" 2>/dev/null \
      && cp "$tmp/$path.build-id" "$dest/fv-gpucheck.build-id"
    echo "baseline: sha-$sha layer $d -> $dest/fv-gpucheck ($(cat "$dest/fv-gpucheck.build-id" 2>/dev/null || echo no build-id))"
    exit 0
  fi
done
echo "fetch-baseline: no layer of sha-$sha holds $path" >&2
exit 1
