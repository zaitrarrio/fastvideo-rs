#!/usr/bin/env bash
# The shared lower layers of every runtime image (docs/serve/images.md
# "Layers"): docker/gpucheck.Dockerfile stages `base-os` (Ubuntu + codec libs +
# ffmpeg) and `base-cuda` (+ the CUDA 13.4 runtime libraries), published once
# per content hash as
#   <repo>:base-os-<hash>  and  <repo>:base-cuda-<hash>
# (default repo ghcr.io/<owner>/fastvideo-rs-runtime, already public). Both
# image workflows (serve-image, gpucheck-runtime-image) pass them back as
# named build contexts that replace those stages, so fastvideo-rs-runtime,
# every fastvideo-rs-serve variant and the debug image sit on the same layer
# digests whichever workflow, runner or cache built them: a host that pulled
# one has the ~1 GB of CUDA libraries for all of them.
#
#   base-images.sh hash [stage] the content hash: the `ARG UBUNTU=` line, the
#                              Dockerfile between "# >>> shared base" and
#                              "# <<< shared base", scripts/gpu/cuda-13.pins
#                              and scripts/gpu/cuda-13-runtime.pins (stage
#                              llama-server: its own "# >>> llama-server"
#                              section instead of the shared base)
#   base-images.sh ensure [extra...]  resolve both tags (and the extra stages',
#                              e.g. llama-server); build + push the missing
#                              ones (base-cuda on top of the published base-os);
#                              append "contexts" (name=docker-image://repo@digest
#                              lines) and "hash" to $GITHUB_OUTPUT
#   base-images.sh contexts [extra...]  print the context lines of published
#                              bases (no build; status 1 if one is missing)
#
# Extra stages are published the same way, one tag per content hash:
#   llama-server   <repo>:llama-server-<hash>, a scratch image holding the CUDA
#                  llama-server binary (the prompt upsampler's sidecar,
#                  docs/serve/upsampler-local.md). It is a compile (nvcc for
#                  four CUDA archs), so CI runs it once per llama.cpp pin.
#                  serve-image passes it; gpucheck-runtime-image does not need it.
#
# Env: FV_BASE_REPO (default ghcr.io/${GITHUB_REPOSITORY_OWNER:-zaitrarrio}/
# fastvideo-rs-runtime); FV_BASE_DISABLE=1 (no contexts: the stages build
# inline, unshared); FV_BASE_REQUIRE=1 (fail instead of warning).
#
# Two workflows that both find a new hash missing at the same moment both
# build it, and the later push wins the tag: images of the earlier one keep
# their own (still valid) base layers until their next build. Base changes
# are rare (a CUDA pin, the ffmpeg build, the Ubuntu digest), so this costs
# at most one extra ~1 GB pull per host, once.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DOCKERFILE="$ROOT/docker/gpucheck.Dockerfile"
REPO="${FV_BASE_REPO:-ghcr.io/$(tr '[:upper:]' '[:lower:]' <<<"${GITHUB_REPOSITORY_OWNER:-zaitrarrio}")/fastvideo-rs-runtime}"
OUT="${GITHUB_OUTPUT:-/dev/null}"
STAGES=(base-os base-cuda)
log() { printf '[base %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

EXTRA_STAGES=(llama-server)

cmd_hash() {
  local section marker="shared base"
  [[ "${1:-}" == llama-server ]] && marker="llama-server"
  section="$(sed -n "/^# >>> $marker/,/^# <<< $marker/p" "$DOCKERFILE")"
  [[ -n "$section" ]] || { log "no $marker section in $DOCKERFILE"; exit 2; }
  { grep -m1 '^ARG UBUNTU=' "$DOCKERFILE"; printf '%s\n' "$section"; cat "$ROOT/scripts/gpu/cuda-13.pins" "$ROOT/scripts/gpu/cuda-13-runtime.pins"; } \
    | sha256sum | cut -c1-16
}

# digest <ref> -> the manifest digest, or nothing when the tag does not exist
digest() {
  docker buildx imagetools inspect "$1" --format '{{json .Manifest}}' 2>/dev/null | jq -r '.digest // empty' || true
}

build() {  # build <stage> <tag> [context lines…] -> digest
  local stage="$1" tag="$2" meta args=() c
  shift 2
  for c in "$@"; do args+=(--build-context "$c"); done
  meta="$(mktemp)"
  log "building $stage -> $tag"
  docker buildx build "$ROOT" -f "$DOCKERFILE" --target "$stage" --platform linux/amd64 \
    --provenance=false --sbom=false \
    --label "org.opencontainers.image.revision=${GITHUB_SHA:-unknown}" \
    --cache-from "type=registry,ref=$REPO:buildcache-base" \
    --cache-to "type=registry,ref=$REPO:buildcache-base,mode=max,image-manifest=true,oci-mediatypes=true,ignore-error=true" \
    ${args[@]+"${args[@]}"} \
    --output "type=image,name=$tag,push=true,compression=gzip,oci-mediatypes=true" \
    --metadata-file "$meta" >&2
  jq -r '."containerimage.digest"' "$meta"
  rm -f "$meta"
}

# resolve [build] [extra stages...] -> context lines on stdout
resolve() {
  local want_build="${1:-0}" h stage tag d lines=() stages=("${STAGES[@]}") x
  shift || true
  for x in "$@"; do
    [[ " ${EXTRA_STAGES[*]} " == *" $x "* ]] || { log "unknown extra stage $x (known: ${EXTRA_STAGES[*]})"; return 1; }
    stages+=("$x")
  done
  for stage in "${stages[@]}"; do
    case "$stage" in
      base-os | base-cuda) h="$(cmd_hash)" ;;
      *) h="$(cmd_hash "$stage")" ;;
    esac
    tag="$REPO:$stage-$h"
    d="$(digest "$tag")"
    if [[ -z "$d" ]]; then
      [[ "$want_build" == 1 ]] || { log "$tag is not published"; return 1; }
      # An extra stage builds on its own (no base contexts): its section starts FROM the pinned Ubuntu.
      case "$stage" in
        base-os | base-cuda) d="$(build "$stage" "$tag" ${lines[@]+"${lines[@]}"})" ;;
        *) d="$(build "$stage" "$tag")" ;;
      esac
      [[ "$d" == sha256:* ]] || { log "$stage: no digest from the build"; return 1; }
      log "published $tag@$d"
    else
      log "reusing $tag@$d"
    fi
    lines+=("$stage=docker-image://$REPO@$d")
  done
  printf '%s\n' "${lines[@]}"
}

cmd_ensure() {
  local ctx h
  h="$(cmd_hash)"
  if [[ "${FV_BASE_DISABLE:-0}" == 1 ]]; then
    log "FV_BASE_DISABLE=1: base stages build inline"
    printf 'hash=%s\ncontexts=\n' "$h" >>"$OUT"
    return 0
  fi
  if ! ctx="$(resolve 1 "$@")"; then
    if [[ "${FV_BASE_REQUIRE:-0}" == 1 ]]; then
      echo "::error title=Shared base images::could not publish or resolve $REPO:base-{os,cuda}-$h"
      exit 1
    fi
    echo "::warning title=Shared base images::could not publish or resolve $REPO:base-{os,cuda}-$h; this image builds its base inline and shares no CUDA layers with the others"
    printf 'hash=%s\ncontexts=\n' "$h" >>"$OUT"
    return 0
  fi
  echo "::notice title=Shared base images::$(tr '\n' ' ' <<<"$ctx")"
  {
    echo "hash=$h"
    echo "contexts<<FV_EOF"
    printf '%s\n' "$ctx"
    echo "FV_EOF"
  } >>"$OUT"
}

case "${1:-}" in
  hash) shift; cmd_hash "$@" ;;
  ensure) shift; cmd_ensure "$@" ;;
  contexts) shift; resolve 0 "$@" ;;
  *) sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
