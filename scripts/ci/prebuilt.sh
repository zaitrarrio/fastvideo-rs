#!/usr/bin/env bash
# GitHub half of the prebuilt release artifacts (docs/serve/images.md
# "Prebuilt binaries", docs/dev/build-pod.md "Release artifacts"): the build
# pod compiles a commit's binaries and uploads them to R2 under
# artifacts/<sha>/ (`build-pod.sh release-artifacts <sha>`); the workflows
# download them here and assemble the images without compiling.
#
#   prebuilt.sh fetch <sha> <set>...
#       Download artifacts/<sha>/manifest.json and the sets' tarballs, check
#       every tarball and every file in it against the manifest's sha256s,
#       extract to $FV_PREBUILT_DIR/<set>/. Writes to $GITHUB_OUTPUT:
#         ok=true|false   whether every requested set is usable
#         dir=<dir>
#         contexts=<name=path lines>   Docker named build contexts that replace
#                                      the compile stages (see CONTEXT below)
#       Missing secrets, a missing or partial upload, a build id or feature
#       mismatch, or a bad checksum all give ok=false with a loud ::warning::
#       (the workflow then compiles as before), unless FV_PREBUILT_REQUIRE=1,
#       which fails instead.
#   prebuilt.sh run-tests <dir>
#       Run the gpucheck-tests set (gpucheck-t0's unit tests, compiled on the
#       pod) from the checkout, each binary in its crate directory.
#
# Env: FV_R2_ARTIFACTS_ENDPOINT / _ACCESS_KEY_ID / _SECRET_ACCESS_KEY (a
# read-only key for the bucket; repository secrets), FV_R2_ARTIFACTS_BUCKET
# (default fv-build-artifacts); FV_PREBUILT_DIR (default
# $RUNNER_TEMP/prebuilt); FV_PREBUILT_BUILD_ID (expected
# scripts/gpu/docker.sh build-id); FV_PREBUILT_FEATURES ("set=features …",
# e.g. "serve-cuda=cuda,http-client"); FV_PREBUILT_WAIT_MIN (poll this long
# for the manifest to appear, default 0); FV_PREBUILT_DISABLE=1 (always
# compile); FV_PREBUILT_REQUIRE=1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
R2="$ROOT/scripts/dev/r2.py"
OUT="${GITHUB_OUTPUT:-/dev/null}"
log() { printf '[prebuilt %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

# set -> the Dockerfile stage its directory replaces (a named build context)
declare -A CONTEXT=(
  [oxide]=oxide [gpucheck]=binary [gpucheck-vast]=binary [hf-fm]=hf-fm
  [serve-cuda]=serve-build [serve-cpu]=cpu-build
)

fallback() {
  if [[ "${FV_PREBUILT_REQUIRE:-0}" == 1 ]]; then
    echo "::error title=Prebuilt artifacts::$1"
    exit 1
  fi
  echo "::warning title=Prebuilt artifacts missing: compiling on the runner::$1. This run compiles in the image as before (slow, uses runner minutes). Build them first: scripts/dev/build-pod.sh release-artifacts $SHA (docs/dev/build-pod.md)."
  echo "ok=false" >>"$OUT"
  echo "contexts=" >>"$OUT"
  exit 0
}

sha256() { sha256sum "$1" | cut -d' ' -f1; }

cmd_fetch() {
  SHA="${1:?sha}"; shift
  (( $# )) || { log "no sets"; exit 2; }
  [[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || fallback "not a full commit sha: $SHA"
  [[ "${FV_PREBUILT_DISABLE:-0}" == 1 ]] && fallback "FV_PREBUILT_DISABLE=1"
  [[ -n "${FV_R2_ARTIFACTS_ENDPOINT:-}" && -n "${FV_R2_ARTIFACTS_ACCESS_KEY_ID:-}" && -n "${FV_R2_ARTIFACTS_SECRET_ACCESS_KEY:-}" ]] \
    || fallback "no R2 read key (secrets FV_R2_ARTIFACTS_ENDPOINT, FV_R2_ARTIFACTS_ACCESS_KEY_ID, FV_R2_ARTIFACTS_SECRET_ACCESS_KEY)"
  local dir="${FV_PREBUILT_DIR:-${RUNNER_TEMP:-/tmp}/prebuilt}" m wait_s deadline
  rm -rf "$dir" && mkdir -p "$dir"
  m="$dir/manifest.json"
  wait_s=$(( ${FV_PREBUILT_WAIT_MIN:-0} * 60 )); deadline=$(( SECONDS + wait_s ))
  log "R2 artifacts/$SHA/manifest.json"
  until python3 "$R2" get "artifacts/$SHA/manifest.json" "$m"; do
    (( SECONDS < deadline )) || fallback "no artifacts/$SHA/manifest.json in R2"
    log "not there yet; waiting (up to ${FV_PREBUILT_WAIT_MIN} min)"
    sleep 30
  done
  [[ "$(jq -r .sha "$m")" == "$SHA" && "$(jq -r .schema "$m")" == 1 ]] || fallback "manifest is not for $SHA (schema 1)"
  if [[ -n "${FV_PREBUILT_BUILD_ID:-}" && "$(jq -r .build_id "$m")" != "$FV_PREBUILT_BUILD_ID" ]]; then
    fallback "build id $(jq -r .build_id "$m") in the manifest, $FV_PREBUILT_BUILD_ID expected"
  fi
  local kv s f
  for kv in ${FV_PREBUILT_FEATURES:-}; do
    s="${kv%%=*}"; f="${kv#*=}"
    if [[ " $* " == *" $s "* && "$(jq -r --arg s "$s" '.sets[$s].features // ""' "$m")" != "$f" ]]; then
      fallback "$s was built with features '$(jq -r --arg s "$s" '.sets[$s].features // ""' "$m")', '$f' requested"
    fi
  done
  log "built $(jq -r .created "$m") on the build pod: $(jq -r '.builder.rustc' "$m" | tr ';' ' ' | cut -c1-80)"
  local set tb want contexts=""
  for set in "$@"; do
    jq -e --arg s "$set" '.sets[$s]' "$m" >/dev/null || fallback "set $set is not in the manifest"
    tb="$(jq -r --arg s "$set" '.sets[$s].tarball' "$m")"
    want="$(jq -r --arg s "$set" '.sets[$s].sha256' "$m")"
    python3 "$R2" get "artifacts/$SHA/$tb" "$dir/$tb" || fallback "artifacts/$SHA/$tb missing"
    [[ "$(sha256 "$dir/$tb")" == "$want" ]] || fallback "$tb: sha256 differs from the manifest"
    mkdir -p "$dir/$set"
    tar -xzf "$dir/$tb" -C "$dir/$set"
    rm -f "$dir/$tb"
    # Every file, and nothing else, with the manifest's sha256.
    diff <(cd "$dir/$set" && find . -type f -printf '%P\n' | LC_ALL=C sort) \
         <(jq -r --arg s "$set" '.sets[$s].files | keys[]' "$m" | LC_ALL=C sort) >/dev/null \
      || fallback "$set: files differ from the manifest"
    while IFS=$'\t' read -r f want; do
      [[ "$(sha256 "$dir/$set/$f")" == "$want" ]] || fallback "$set/$f: sha256 differs from the manifest"
    done < <(jq -r --arg s "$set" '.sets[$s].files | to_entries[] | [.key, .value.sha256] | @tsv' "$m")
    log "$set: $(jq -r --arg s "$set" '.sets[$s].files | length' "$m") files verified ($(du -sh "$dir/$set" | cut -f1))"
    [[ -n "${CONTEXT[$set]:-}" ]] && contexts+="${CONTEXT[$set]}=$dir/$set"$'\n'
  done
  echo "::notice title=Prebuilt artifacts::using the build pod's binaries for $SHA ($*): no cargo compile on this runner"
  {
    echo "ok=true"
    echo "dir=$dir"
    echo "contexts<<FV_EOF"
    printf '%s' "$contexts"
    echo "FV_EOF"
  } >>"$OUT"
}

# Test binaries keep the pod's absolute source paths (file!(), CARGO_MANIFEST_DIR
# baked in); the pod's worktree path is linked to this checkout so they resolve.
cmd_run_tests() {
  local dir="${1:?dir}" src fail=0 n=0 bin pkg crate kind name
  src="$(cat "$dir/src-root")"; src="${src%/}"
  if [[ "$src" != "$ROOT" && ! -e "$src" ]]; then
    local sudo=""
    sudo -n true 2>/dev/null && sudo=sudo
    { $sudo mkdir -p "$(dirname "$src")" && $sudo ln -s "$ROOT" "$src"; } 2>/dev/null \
      || log "could not link $src to the checkout (tests that read sources may fail)"
  fi
  while IFS=$'\t' read -r bin pkg crate kind name; do
    n=$((n + 1))
    echo "::group::$pkg ($kind $name)"
    chmod +x "$dir/$bin"
    if ! (cd "$ROOT/$crate" && CARGO_MANIFEST_DIR="$ROOT/$crate" CARGO_PKG_NAME="$pkg" "$dir/$bin"); then
      fail=1; echo "::error title=unit tests::$pkg $kind $name failed"
    fi
    echo "::endgroup::"
  done <"$dir/tests.tsv"
  (( n > 0 )) || { echo "::error::no test binaries in $dir/tests.tsv"; exit 1; }
  return "$fail"
}

case "${1:-}" in
  fetch) shift; cmd_fetch "$@" ;;
  run-tests) shift; cmd_run_tests "$@" ;;
  *) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
