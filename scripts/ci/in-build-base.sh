#!/usr/bin/env bash
# Run a command in the build-base image (docker/build-base.Dockerfile, the
# image the build pods run; tag from scripts/dev/build-base-tag.sh) on a
# GitHub-hosted runner, for tools-release.yml's fallback jobs
# (docs/dev/tools-releases.md "GitHub-hosted fallback").
#
#   in-build-base.sh <cmd...>
#
# The checkout is /src, the work dir ($FV_WORK, default /mnt/fv-work when
# /mnt is writable, else $RUNNER_TEMP/fv-work) is /work: CARGO_TARGET_DIR
# /work/target, release-cache /work/vol (FV_IMAGE_DIRS=1: the image's own, for
# a deps image). rustc goes through the image's sccache, stored in the GitHub
# Actions cache when the job exported ACTIONS_RESULTS_URL /
# ACTIONS_RUNTIME_TOKEN (passed by name, never on the command line). CARGO_PROFILE_* and FV_* from the
# job pass through. Files written in /work are handed back to the runner user.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
(( $# )) || { echo "usage: in-build-base.sh <cmd...>" >&2; exit 2; }
img="${FV_BUILD_BASE_IMAGE:-$(bash "$ROOT/scripts/dev/build-base-tag.sh" --image)}"
W="${FV_WORK:-}"
if [[ -z "$W" ]]; then
  if sudo -n mkdir -p /mnt/fv-work 2>/dev/null && sudo -n chown "$(id -u):$(id -g)" /mnt/fv-work; then W=/mnt/fv-work
  else W="${RUNNER_TEMP:-/tmp}/fv-work"
  fi
fi
mkdir -p "$W"
args=(--rm -v "$ROOT:/src" -v "$W:/work" -w /src
      -e CARGO_INCREMENTAL=0 -e CARGO_TERM_COLOR=never
      # The checkout belongs to the runner's uid, the container runs as root.
      -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e "GIT_CONFIG_VALUE_0=*"
      -e GITHUB_RUN_ID -e GITHUB_REPOSITORY -e GITHUB_WORKFLOW -e GITHUB_SHA -e GITHUB_OUTPUT=/work/.github-output)
# A deps image (docker/tools-deps.Dockerfile, FV_IMAGE_DIRS=1) brings its own
# /target (cooked dependencies) and /vol (oxide cache); else both in /work.
if [[ "${FV_IMAGE_DIRS:-0}" != 1 ]]; then
  args+=(-e CARGO_TARGET_DIR=/work/target -e FV_BUILD_VOLUME_DIR=/work/vol)
fi
while IFS='=' read -r k _; do
  if [[ "$k" =~ ^(CARGO_PROFILE_[A-Z0-9_]+|FV_[A-Z0-9_]+)$ && ! "$k" =~ ^FV_(WORK|IMAGE_DIRS|BUILD_BASE_IMAGE)$ ]]; then
    args+=(-e "$k")
  fi
done < <(env)
# rustc always goes through sccache (as in the deps images, so cargo sees the
# same wrapper); its store is the GitHub Actions cache when the job exported
# the credentials, else a local dir.
args+=(-e RUSTC_WRAPPER=/usr/local/bin/sccache)
if [[ -n "${ACTIONS_RESULTS_URL:-}" && -n "${ACTIONS_RUNTIME_TOKEN:-}" ]]; then
  args+=(-e SCCACHE_GHA_ENABLED=on -e ACTIONS_CACHE_SERVICE_V2=on
         -e ACTIONS_RESULTS_URL -e ACTIONS_RUNTIME_TOKEN -e "SCCACHE_GHA_VERSION=${FV_SCCACHE_GHA_VERSION:-fv-tools-1}")
else
  echo "::notice title=sccache::no Actions cache credentials in the job env: sccache uses a local dir only"
  args+=(-e SCCACHE_DIR=/work/sccache)
fi
: >"$W/.github-output"
rc=0
docker run "${args[@]}" "$img" bash -c '
  # git 2.34 (Ubuntu 22.04) honours safe.directory only in the global config.
  git config --global --add safe.directory "*"
  sccache --start-server >/dev/null 2>&1 || true
  rc=0; "$@" || rc=$?
  sccache --show-stats 2>/dev/null | sed -n "1,14p" || true
  exit "$rc"' _ "$@" || rc=$?
sudo -n chown -R "$(id -u):$(id -g)" "$W" 2>/dev/null || true
[[ -n "${GITHUB_OUTPUT:-}" ]] && cat "$W/.github-output" >>"$GITHUB_OUTPUT"
exit "$rc"
