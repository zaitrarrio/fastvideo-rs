#!/usr/bin/env bash
# CI half of the serve images (docs/serve/images.md): `cuda`, the one CUDA image every preset runs, and `cpu`
# (the fake engine). Neither bakes a worker config: every launch sends one, and an image started without one
# refuses to boot (owner decision 2026-10-25, docs/adr/0011-deploy-section.md). Run by
# .github/workflows/serve-image.yml after the legacy (debug) image.
#
#   ci-images.sh build <image>…       build + push serve-<image> (docker/gpucheck.Dockerfile);
#                                     appends "<image> <repo>@<digest> <compressed MB>" to $FV_CI_OUT
#   ci-images.sh smoke <image> <ref>  no-GPU checks of a pushed serve image: the libraries, ffmpeg, the
#                                     refusal without a config, and the fake engine's /healthz through
#                                     FV_WORKER_CONFIG=/etc/fv/runpod-fake.toml (the one config it carries)
#   ci-images.sh size <ref>           compressed size (MB) and layer count of a pushed image
#   ci-images.sh layers <ref>         "<digest> <MB>" per layer (bottom first)
#   ci-images.sh summary              markdown table of $FV_CI_OUT (for $GITHUB_STEP_SUMMARY)
#
# Env: FV_IMAGE (repo, e.g. ghcr.io/zaitrarrio/fastvideo-rs-serve); FV_SHORT_SHA;
# FV_BUILD_ID; FV_SERVE_FEATURES (default cuda,http-client);
# FV_CPU_FEATURES (default http-client); FV_LATEST=1 also tags :<image>
# and :<image>-latest (the `latest` channel, docs/serve/releases.md);
# FV_BUILD_TIME (the commit time, for `fv-serve --version`);
# FV_COMPRESSION (gzip | zstd, default gzip); FV_CACHE_FROM (space list of
# registry cache refs); FV_CI_OUT (default artifacts/ci/images.tsv);
# FV_SMOKE_TIMEOUT_S (whole seconds, default 120: the bound on each `docker run` of the smoke, which
# then fails, naming the check, and removes the container; FV_SMOKE_PULL_TIMEOUT_S, default
# 600, bounds the pull); FV_BUILD_CONTEXTS (name=path lines from scripts/ci/prebuilt.sh: directories
# of build-pod binaries that replace the serve-build / cpu-build stages,
# so no cargo compile runs; empty: compile in the image); FV_TOOLS_TAG (the
# tools release they come from, a `dev.fastvideo.tools` label); FV_BOOT_SOURCE (prebuilt.sh's
# fv_boot: that release's tag, or compiled@<sha> when boot-build compiled this checkout's
# fv-boot; the `dev.fastvideo.fv-boot` label).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${FV_CI_OUT:-$ROOT/artifacts/ci/images.tsv}"
# The one config the serve images carry: CI's smoke check names it explicitly (FV_WORKER_CONFIG).
SMOKE_CONFIG=/etc/fv/runpod-fake.toml
COMPRESSION="${FV_COMPRESSION:-gzip}"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 2; }
SMOKE_TIMEOUT_S="${FV_SMOKE_TIMEOUT_S:-120}"

# bounded <what> <docker run args…>: `docker run --rm` under a deadline (FV_SMOKE_TIMEOUT_S),
# instead of waiting forever (serve-image run 286: an fv-boot that served instead of refusing).
# A run the deadline ends returns 124 after a FATAL line naming <what>. The container gets a
# name: `docker run` without a TTY forwards timeout's SIGTERM to it, so it usually exits and
# --rm removes it, but one that ignores the signal outlives the client and is removed here
# (`docker rm -f`, when `docker container inspect` still finds it). Otherwise it returns the
# container's exit status; its output passes through.
bounded() {
  local what="$1" name rc=0 t0=$SECONDS; shift
  name="fv-smoke-$$-$RANDOM"
  timeout -k 10 "$SMOKE_TIMEOUT_S" docker run --rm --name "$name" "$@" || rc=$?
  # 124 (timeout) or 137 (its KILL after -k) at the deadline; a container's own exit 124 / 137
  # (an OOM kill) before it is the container's status.
  if [[ "$rc" == 124 || "$rc" == 137 ]] && (( SECONDS - t0 >= SMOKE_TIMEOUT_S )); then
    local removed=""
    if timeout 60 docker container inspect "$name" >/dev/null 2>&1; then
      timeout 60 docker rm -f "$name" >/dev/null 2>&1 && removed="; container removed" \
        || removed="; could not remove container $name"
    fi
    log "FATAL: $what: still running after ${SMOKE_TIMEOUT_S}s (FV_SMOKE_TIMEOUT_S)$removed"
    return 124
  fi
  return "$rc"
}

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
    # One registry cache per build graph: cuda's, and cpu's own.
    cache_to=()
    case "$v" in
      cuda) cache_to=(--cache-to "type=registry,ref=$FV_IMAGE:buildcache-cuda,mode=max,image-manifest=true,oci-mediatypes=true,ignore-error=true") ;;
      cpu) cache_to=(--cache-to "type=registry,ref=$FV_IMAGE:buildcache-cpu,mode=max,image-manifest=true,oci-mediatypes=true,ignore-error=true") ;;
      *) die "unknown serve image $v (cuda, cpu; deploy/deploy.toml [[images]])" ;;
    esac
    args=()
    for c in ${FV_CACHE_FROM:-} "$FV_IMAGE:buildcache-cuda" "$FV_IMAGE:buildcache-cpu"; do
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
      --label "dev.fastvideo.fv-boot=${FV_BOOT_SOURCE:-compiled@${GITHUB_SHA:-unknown}}" \
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
  [[ "$SMOKE_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] \
    || die "FV_SMOKE_TIMEOUT_S=$SMOKE_TIMEOUT_S: whole seconds expected (e.g. 120), not a duration like 2m"
  timeout -k 10 "${FV_SMOKE_PULL_TIMEOUT_S:-600}" docker pull -q "$img" >/dev/null \
    || die "serve-$v: docker pull $img failed or took over ${FV_SMOKE_PULL_TIMEOUT_S:-600}s"
  docker image inspect "$img" --format "size={{.Size}} entrypoint={{json .Config.Entrypoint}} env={{json .Config.Env}}"
  # The entrypoint is fv-boot (docs/adr/0014-fv-boot.md).
  local ep; ep="$(docker image inspect "$img" --format '{{json .Config.Entrypoint}}')"
  [[ "$ep" == '["/opt/fastvideo-rs/bin/fv-boot"]' ]] || die "serve-$v: entrypoint $ep, expected [\"/opt/fastvideo-rs/bin/fv-boot\"]"
  bounded "serve-$v: --version" "$img" --version
  # Every dynamic library resolves; ffmpeg does what the code asks of it;
  # no config is baked (FV_CONFIG unset) and the one the image carries parses;
  # the CUDA image has the CUDA libraries, the cpu image none; none has the
  # debug tooling.
  SMOKE_TIMEOUT_S=$((SMOKE_TIMEOUT_S * 3)) bounded "serve-$v: image checks" --entrypoint bash -e V="$v" "$img" -euo pipefail -c '
    bad() { echo "FAIL: $*"; exit 1; }
    for b in /opt/fastvideo-rs/bin/fv-serve /opt/fastvideo-rs/bin/fv-boot /usr/local/bin/ffmpeg /usr/local/bin/ffprobe; do
      ldd "$b" | grep "not found" && bad "$b: unresolved libraries"
    done
    test "$FV_VARIANT" = "$V"
    test -z "${FV_CONFIG:-}" || bad "FV_CONFIG=$FV_CONFIG is baked (the images bake no config)"
    test "$(ls /etc/fv)" = runpod-fake.toml || bad "/etc/fv holds $(ls /etc/fv | tr "\n" " "), not only runpod-fake.toml"
    # the entrypoint is fv-boot (docs/adr/0014-fv-boot.md); fv-entry.sh is gone
    /opt/fastvideo-rs/bin/fv-boot version
    test -e /opt/fastvideo-rs/bin/fv-entry && bad "fv-entry is in the image"
    cat /opt/fastvideo-rs/bin/fv-serve.features
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
      # the prompt upsampler sidecar (docs/serve/upsampler-local.md): llama-server and its launcher, libraries resolved
      test -x /opt/fastvideo-rs/bin/llama-server || bad "no llama-server"
      test -x /opt/fastvideo-rs/bin/fv-upsampler || bad "no fv-upsampler"
      ldd /opt/fastvideo-rs/bin/llama-server | grep "not found" && bad "llama-server: unresolved libraries"
      cat /opt/fastvideo-rs/bin/llama-server.version
    fi
    echo "smoke ok: $V"'
  # Started without a config, the image refuses to boot: fv-boot exits non-zero and says why (no fv-serve default).
  # Bounded: an fv-boot that serves fv-serve's default instead never exits (serve-image run 286).
  local out rc=0
  out="$(bounded "serve-$v: the refusal without a config (an fv-boot that does not refuse serves fv-serve's default and never exits)" \
    -e FV_FAIL_SLEEP_S=0 "$img" 2>&1)" || rc=$?
  [[ "$rc" != 124 ]] || die "serve-$v: started without a config, it did not refuse within ${SMOKE_TIMEOUT_S}s (it serves instead: is the image's fv-boot older than ADR-0011?): $(tail -3 <<<"$out")"
  [[ "$rc" != 0 ]] || die "serve-$v: started without a config, it booted (exit 0)"
  grep -q "FAILED: no worker config" <<<"$out" || die "serve-$v: no refusal message without a config: $(tail -3 <<<"$out")"
  echo "refused without a config (exit $rc): $(grep -m1 'FAILED: no worker config' <<<"$out")"
  # HTTP mode on the fake engine through the image entrypoint, the config named explicitly
  # (FV_WORKER_CONFIG, the escape hatch for this check): /healthz 200.
  cid="$(timeout 120 docker run -d -p 18001:8000 -e FV_ENGINE=fake -e FV_AUTH_MODE=none -e "FV_WORKER_CONFIG=$SMOKE_CONFIG" "$img")"
  for _ in $(seq 1 60); do
    if curl -fsS -m 5 localhost:18001/healthz; then ok=1; break; fi
    sleep 1
  done
  echo
  timeout 30 docker logs "$cid" 2>&1 | tail -5
  timeout 60 docker rm -f "$cid" >/dev/null || true
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
  echo "| image | reference | compressed MB |"
  echo "|---|---|---:|"
  awk -F'\t' '{printf "| %s | `%s` | %s |\n", $1, $2, $3}' "$OUT"
}

case "${1:-}" in
  build) shift; build "$@" ;;
  smoke) shift; smoke "${1:?image id}" "${2:?image ref}" ;;
  size) shift; size "${1:?image}" ;;
  layers) shift; layers "${1:?image}" ;;
  summary) summary ;;
  *) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 2 ;;
esac
