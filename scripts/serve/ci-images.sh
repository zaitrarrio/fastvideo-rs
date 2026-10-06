#!/usr/bin/env bash
# CI half of the per-variant serve images (docs/serve/images.md); run by
# .github/workflows/serve-image.yml after the legacy (debug) image.
#
#   ci-images.sh build <variant>…     build + push serve-<variant> (docker/gpucheck.Dockerfile);
#                                     appends "<variant> <repo>@<digest> <compressed MB>" to $FV_CI_OUT
#   ci-images.sh smoke <variant> <ref>   no-GPU checks of a pushed variant image
#   ci-images.sh size <ref>           compressed size (MB) and layer count of a pushed image
#   ci-images.sh layers <ref>         "<digest> <MB>" per layer (bottom first)
#   ci-images.sh summary              markdown table of $FV_CI_OUT (for $GITHUB_STEP_SUMMARY)
#
# Env: FV_IMAGE (repo, e.g. ghcr.io/zaitrarrio/fastvideo-rs-serve); FV_SHORT_SHA;
# FV_BUILD_ID; FV_SERVE_FEATURES (default cuda,http-client);
# FV_CPU_FEATURES (default http-client); FV_LATEST=1 also tags :<variant>
# and :<variant>-latest (the `latest` channel, docs/serve/releases.md);
# FV_BUILD_TIME (the commit time, for `fv-serve --version`);
# FV_COMPRESSION (gzip | zstd, default gzip); FV_CACHE_FROM (space list of
# registry cache refs); FV_CI_OUT (default artifacts/ci/variants.tsv);
# FV_BUILD_CONTEXTS (name=path lines from scripts/ci/prebuilt.sh: directories
# of build-pod binaries that replace the serve-build / cpu-build stages,
# so no cargo compile runs; empty: compile in the image); FV_TOOLS_TAG (the
# tools release they come from, a `dev.fastvideo.tools` label).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${FV_CI_OUT:-$ROOT/artifacts/ci/variants.tsv}"
COMPRESSION="${FV_COMPRESSION:-gzip}"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 2; }

# size <ref> -> "<MB> <layers>" (compressed, from the registry manifest)
size() {
  local raw d
  raw="$(docker buildx imagetools inspect --raw "$1")"
  if jq -e '.manifests' >/dev/null <<<"$raw"; then
    d="$(jq -r '[.manifests[] | select(.platform.architecture == "amd64")][0].digest' <<<"$raw")"
    raw="$(docker buildx imagetools inspect --raw "${1%@*}@$d")"
  fi
  jq -r '"\(([.layers[].size] | add) / 1e6 | . * 10 | floor / 10) \(.layers | length)"' <<<"$raw"
}

build() {
  local v tags cache_to meta digest ref force args=()
  : "${FV_IMAGE:?FV_IMAGE}" "${FV_SHORT_SHA:?FV_SHORT_SHA}"
  mkdir -p "$(dirname "$OUT")"
  for v in "$@"; do
    tags="$FV_IMAGE:$v-sha-$FV_SHORT_SHA"
    [[ "${FV_LATEST:-0}" == 1 ]] && tags+=",$FV_IMAGE:$v,$FV_IMAGE:$v-latest"
    [[ "$COMPRESSION" == gzip ]] || tags="${tags//:$v-sha-/:$v-$COMPRESSION-sha-}"
    # One registry cache per build graph: the CUDA variants share everything
    # up to the variant stage (h3-turbo exports it), cpu has its own.
    cache_to=()
    case "$v" in
      h3-turbo) cache_to=(--cache-to "type=registry,ref=$FV_IMAGE:buildcache-variants,mode=max,image-manifest=true,oci-mediatypes=true,ignore-error=true") ;;
      cpu) cache_to=(--cache-to "type=registry,ref=$FV_IMAGE:buildcache-cpu,mode=max,image-manifest=true,oci-mediatypes=true,ignore-error=true") ;;
    esac
    args=()
    for c in ${FV_CACHE_FROM:-} "$FV_IMAGE:buildcache-variants" "$FV_IMAGE:buildcache-cpu"; do
      args+=(--cache-from "type=registry,ref=$c")
    done
    while IFS= read -r c; do
      [[ -n "$c" ]] && args+=(--build-context "$c")
    done <<<"${FV_BUILD_CONTEXTS:-}"
    # gzip keeps the shared base layers' blobs (scripts/ci/base-images.sh) as
    # they are, so every image has the same digests; zstd re-encodes them all.
    force=false
    [[ "$COMPRESSION" == gzip ]] || force=true
    meta="$(mktemp)"
    log "build serve-$v -> $tags ($COMPRESSION)"
    docker buildx build "$ROOT" -f "$ROOT/docker/gpucheck.Dockerfile" --target "serve-$v" --platform linux/amd64 \
      --build-arg "BUILD_ID=${FV_BUILD_ID:-unknown}" \
      --build-arg "FV_GIT_SHA=${GITHUB_SHA:-unknown}" \
      --build-arg "FV_BUILD_TIME=${FV_BUILD_TIME:-}" \
      --build-arg "FV_SERVE_FEATURES=${FV_SERVE_FEATURES:-cuda,http-client}" \
      --build-arg "FV_CPU_FEATURES=${FV_CPU_FEATURES:-http-client}" \
      --label "org.opencontainers.image.revision=${GITHUB_SHA:-unknown}" \
      --label "dev.fastvideo.build-id=${FV_BUILD_ID:-unknown}" \
      --label "dev.fastvideo.variant=$v" \
      --label "dev.fastvideo.tools=${FV_TOOLS_TAG:-compiled}" \
      "${args[@]}" ${cache_to[@]+"${cache_to[@]}"} \
      --output "type=image,\"name=$tags\",push=true,compression=$COMPRESSION,force-compression=$force,oci-mediatypes=true" \
      --metadata-file "$meta"
    digest="$(jq -r '."containerimage.digest"' "$meta")"
    [[ "$digest" == sha256:* ]] || die "serve-$v: no digest in the build metadata"
    ref="$FV_IMAGE@$digest"
    printf '%s\t%s\t%s\n' "$v" "$ref" "$(size "$ref" | cut -d' ' -f1)" >>"$OUT"
    log "serve-$v: $ref ($(size "$ref") MB/layers)"
    rm -f "$meta"
  done
}

smoke() {
  local v="$1" img="$2" cid ok=0
  docker pull -q "$img" >/dev/null
  docker image inspect "$img" --format "size={{.Size}} entrypoint={{json .Config.Entrypoint}} env={{json .Config.Env}}"
  docker run --rm "$img" --version
  # Every dynamic library resolves; ffmpeg does what the code asks of it;
  # the baked config parses; CUDA variants have the CUDA libraries, the
  # cpu image none; none has the debug tooling.
  docker run --rm --entrypoint bash -e V="$v" "$img" -euo pipefail -c '
    bad() { echo "FAIL: $*"; exit 1; }
    for b in /opt/fastvideo-rs/bin/fv-serve /usr/local/bin/ffmpeg /usr/local/bin/ffprobe; do
      ldd "$b" | grep "not found" && bad "$b: unresolved libraries"
    done
    test "$FV_VARIANT" = "$V"
    test -f "$FV_CONFIG"
    cat /opt/fastvideo-rs/bin/fv-serve.features
    FV_ENGINE=fake /opt/fastvideo-rs/bin/fv-serve --config "$FV_CONFIG" --print-config >/dev/null
    FV_ENGINE=fake /opt/fastvideo-rs/bin/fv-serve --config /etc/fv/runpod-fake.toml --print-config >/dev/null
    ffmpeg -hide_banner -encoders | grep -E " (h264_nvenc|libx264|libvpx|aac) "
    ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc=s=64x48:r=24:d=0.2 -f lavfi -i sine=d=0.2 \
      -c:v libx264 -pix_fmt yuv420p -c:a aac -movflags +faststart -y /tmp/t.mp4
    ffprobe -v error -show_entries stream=codec_name -of csv=p=0 /tmp/t.mp4
    for x in sshd rsync fv-gpucheck hf-fm tileiras nvcc; do command -v "$x" && bad "$x is in the image"; done
    test -e /opt/fastvideo-rs/scripts && bad "scripts/gpu is in the image"
    test -e /opt/fastvideo-rs/oxide && bad "the oxide directory is in the image"
    ldconfig -p | grep -E "libcupti|libcudnn_adv|libcudnn_ext|libnvblas|libnvJitLink" && bad "CUPTI / cudnn_adv / cudnn_ext / nvblas / nvJitLink present"
    find / -xdev \( -name "*.a" -path "*cuda*" -o -name "cudnn*.h" -o -name "cublas*.h" \) 2>/dev/null | grep . && bad "static CUDA libraries or headers present"
    if [ "$V" = cpu ]; then
      grep -q cuda /opt/fastvideo-rs/bin/fv-serve.features && bad "cpu built with cuda"
      ldconfig -p | grep -E "libcudnn|libcublas|libnvrtc" && bad "CUDA libraries in the cpu image"
      test -e /usr/local/cuda-13.4 && bad "/usr/local/cuda-13.4 in the cpu image"
    else
      test "$NVIDIA_DRIVER_CAPABILITIES" = compute,utility,video
      for l in libnvrtc.so.13 libnvrtc.so libcublas.so.13 libcublasLt.so.13 libcublas.so libcudnn.so.9 libcudnn.so libcudart.so.13 libcudart.so \
               libcudnn_graph.so.9 libcudnn_ops.so.9 libcudnn_cnn.so.9 libcudnn_heuristic.so.9 \
               libcudnn_engines_precompiled.so.9 libcudnn_engines_runtime_compiled.so.9 \
               libcudnn_engines_tensor_ir.so.9; do
        ldconfig -p | grep -qE "^[[:space:]]+$l " || bad "missing $l"
      done
      ls /usr/local/cuda-13.4/lib64 | grep -q "libnvrtc-builtins" || bad "missing nvrtc-builtins"
    fi
    echo "smoke ok: $V"'
  # HTTP mode on the fake engine through the image entrypoint: /healthz 200.
  cid="$(docker run -d -p 18001:8000 -e FV_ENGINE=fake -e FV_AUTH_MODE=none "$img" --config /etc/fv/runpod-fake.toml)"
  for _ in $(seq 1 60); do
    if curl -fsS localhost:18001/healthz; then ok=1; break; fi
    sleep 1
  done
  echo
  docker logs "$cid" 2>&1 | tail -5
  docker rm -f "$cid" >/dev/null
  [[ "$ok" == 1 ]] || die "$v: /healthz never answered"
}

# layers <ref> -> "<digest> <MB>" per layer, bottom first (compressed)
layers() {
  local raw d
  raw="$(docker buildx imagetools inspect --raw "$1")"
  if jq -e '.manifests' >/dev/null <<<"$raw"; then
    d="$(jq -r '[.manifests[] | select(.platform.architecture == "amd64")][0].digest' <<<"$raw")"
    raw="$(docker buildx imagetools inspect --raw "${1%@*}@$d")"
  fi
  jq -r '.layers[] | "\(.digest) \(.size / 1e6 | . * 10 | floor / 10)"' <<<"$raw"
}

summary() {
  echo "| variant | image | compressed MB |"
  echo "|---|---|---:|"
  awk -F'\t' '{printf "| %s | `%s` | %s |\n", $1, $2, $3}' "$OUT"
}

case "${1:-}" in
  build) shift; build "$@" ;;
  smoke) shift; smoke "${1:?variant}" "${2:?image}" ;;
  size) shift; size "${1:?image}" ;;
  layers) shift; layers "${1:?image}" ;;
  summary) summary ;;
  *) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
