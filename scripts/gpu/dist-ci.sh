#!/usr/bin/env bash
# dist-ci.sh: put the CI-built fv-gpucheck for the current sources into
# artifacts/gpucheck/dist, where validate.sh expects it.
#
# Compiling needs no GPU, so it should never rent one: the runtime-image
# workflow builds the release binary (nvcc cubins for every SM included) on a
# GitHub runner for every push and uploads it as `fv-gpucheck-<build id>`. The
# build id is a hash of the sources, so an artifact with that name *is* this
# tree's binary. build-remote.sh (a rented box) is the fallback for when CI is
# unavailable, not the default.
set -euo pipefail
# shellcheck source=scripts/gpu/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
DIST="$FV_ROOT/artifacts/gpucheck/dist"
command -v gh >/dev/null || die "gh (GitHub CLI) is required"

bid="$(bash "$FV_ROOT/scripts/gpu/docker.sh" build-id)"
if [[ "$(cat "$DIST/fv-gpucheck.build-id" 2>/dev/null)" == "$bid" ]]; then
  log "dist already at build $bid"
  exit 0
fi
name="fv-gpucheck-$bid"
run="$(gh api "repos/{owner}/{repo}/actions/artifacts?name=$name&per_page=1" \
  -q '.artifacts | map(select(.expired | not)) | .[0].workflow_run.id // empty' 2>/dev/null || true)"
if [[ -z "$run" ]]; then
  # The id covers untracked files under crates/ too, so a dirty tree has an id
  # CI has never seen. Say which of the usual reasons applies.
  if [[ -n "$(git -C "$FV_ROOT" status --porcelain -- crates Cargo.toml Cargo.lock rust-toolchain.toml docker/gpucheck.Dockerfile)" ]]; then
    die "no CI artifact $name: the tree has uncommitted source changes — commit and push, or run from a clean worktree of the pushed commit"
  fi
  if [[ -n "$(git -C "$FV_ROOT" log --oneline '@{u}..' 2>/dev/null)" ]]; then
    die "no CI artifact $name: HEAD is not pushed yet"
  fi
  die "no CI artifact $name yet: CI is probably still building (gh run list --workflow gpucheck-runtime-image.yml)"
fi
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
gh run download "$run" -n "$name" -D "$tmp" >/dev/null
[[ "$(cat "$tmp/fv-gpucheck.build-id")" == "$bid" ]] || die "artifact $name carries build id $(cat "$tmp/fv-gpucheck.build-id")"
mkdir -p "$DIST"
cp "$tmp/fv-gpucheck" "$tmp/fv-gpucheck.build-id" "$DIST/"
chmod +x "$DIST/fv-gpucheck"
log "dist: build $bid from CI run $run ($(du -h "$DIST/fv-gpucheck" | cut -f1))"
