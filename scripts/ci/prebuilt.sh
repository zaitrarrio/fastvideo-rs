#!/usr/bin/env bash
# GitHub half of the prebuilt tools (docs/dev/tools-releases.md, docs/serve/
# images.md "Prebuilt binaries"): the build pod compiles and tests the tools
# and the coordinator publishes them as a SemVer GitHub release
# (`tools-v<X.Y.Z>`, scripts/ci/tools-release.sh publish); the workflows
# download a release here and assemble the images without compiling.
#
#   prebuilt.sh fetch <set>...
#       Pick a tools release for the checked-out commit (HEAD; on a pull
#       request the merge commit), download the sets, check every sha256
#       (scripts/ci/tools-release.sh fetch), extract to $FV_PREBUILT_DIR/<set>/.
#       Which release:
#         1. FV_TOOLS_VERSION (repository variable, a rollback pin; image
#            workflows only): that release, whatever its inputs;
#         2. the release built from exactly HEAD's tools inputs (input hash);
#         3. none matches (HEAD changed crates/, Cargo.lock, ... since the last
#            release): FV_PREBUILT_STALE=latest uses the highest SemVer release
#            anyway, with a warning, and the image is labelled with it (the
#            image workflows on main); FV_PREBUILT_STALE=compile (default; tests
#            and branch images, which must run their own code) falls back.
#       Writes to $GITHUB_OUTPUT:
#         ok=true|false     every requested set is usable (false: compile)
#         dir, tag, version, exact=true|false, build_id, source_commit
#         contexts=<name=path lines>   Docker named build contexts that replace
#                                      the compile stages (see CONTEXT below)
#       Fallback (ok=false, a loud ::warning::, the workflow compiles as
#       before) unless FV_PREBUILT_REQUIRE=1, which fails instead.
#   prebuilt.sh run-tests <dir>
#       Run the gpucheck-tests set (gpucheck-t0's unit tests, compiled on the
#       pod) from the checkout (or $FV_TESTS_ROOT), each in its crate directory.
#
# Env: GH_TOKEN / GITHUB_TOKEN (read access; anonymous works for a public
# repository); FV_PREBUILT_DIR (default $RUNNER_TEMP/prebuilt);
# FV_PREBUILT_FEATURES ("set=features …", e.g. "serve-cuda=cuda,http-client");
# FV_PREBUILT_STALE; FV_TOOLS_VERSION; FV_PREBUILT_DISABLE=1 (always compile);
# FV_PREBUILT_REQUIRE=1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TR="$HERE/tools-release.sh"
OUT="${GITHUB_OUTPUT:-/dev/null}"
log() { printf '[prebuilt %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

# set -> the Dockerfile stage its directory replaces (a named build context)
declare -A CONTEXT=(
  [oxide]=oxide [gpucheck]=binary [gpucheck-vast]=binary [hf-fm]=hf-fm
  [serve-cuda]=serve-build [serve-gateway]=gateway-build
)

fallback() {
  if [[ "${FV_PREBUILT_REQUIRE:-0}" == 1 ]]; then
    echo "::error title=Prebuilt tools::$1"
    exit 1
  fi
  echo "::warning title=Prebuilt tools not used: compiling on the runner::$1. This run compiles as before (slow, uses runner minutes). Tools releases: scripts/ci/tools-release.sh status / publish (docs/dev/tools-releases.md)."
  { echo "ok=false"; echo "exact=false"; echo "contexts="; } >>"$OUT"
  exit 0
}

cmd_fetch() {
  (( $# )) || { log "no sets"; exit 2; }
  [[ "${FV_PREBUILT_DISABLE:-0}" == 1 ]] && fallback "FV_PREBUILT_DISABLE=1"
  local dir="${FV_PREBUILT_DIR:-${RUNNER_TEMP:-/tmp}/prebuilt}" h rel tag exact m
  h="$(bash "$TR" input-hash HEAD)"
  local args=(--input-hash "$h")
  [[ -n "${FV_TOOLS_VERSION:-}" ]] && args=(--version "${FV_TOOLS_VERSION#v}" --input-hash "$h")
  rel="$(bash "$TR" resolve "${args[@]}")" || fallback "no usable tools release (${FV_TOOLS_VERSION:+pin $FV_TOOLS_VERSION, }input hash ${h:0:16})"
  tag="$(jq -r .tag <<<"$rel")"; exact="$(jq -r .exact <<<"$rel")"
  if [[ "$exact" != true ]]; then
    if [[ -n "${FV_TOOLS_VERSION:-}" ]]; then
      echo "::notice title=Tools pinned::FV_TOOLS_VERSION=$FV_TOOLS_VERSION: using $tag (built from $(jq -r '.source_commit[0:12]' <<<"$rel")), not this commit's tools sources"
    elif [[ "${FV_PREBUILT_STALE:-compile}" == latest ]]; then
      echo "::warning title=Tools release is older than this commit::this commit's tools inputs (input hash ${h:0:16}) have no release yet; the image uses the newest one, $tag (built from $(jq -r '.source_commit[0:12]' <<<"$rel")), and is labelled with it. Publish one: scripts/ci/tools-release.sh publish <sha>."
    else
      fallback "no tools release built from this commit's inputs (input hash ${h:0:16}; newest is $tag)"
    fi
  fi
  bash "$TR" fetch "$tag" "$dir" "$@" || fallback "$tag: download or sha256 check failed"
  m="$dir/manifest.json"
  local kv s f
  for kv in ${FV_PREBUILT_FEATURES:-}; do
    s="${kv%%=*}"; f="${kv#*=}"
    if [[ " $* " == *" $s "* && "$(jq -r --arg s "$s" '.sets[$s].features // ""' "$m")" != "$f" ]]; then
      fallback "$tag: $s was built with features '$(jq -r --arg s "$s" '.sets[$s].features // ""' "$m")', '$f' requested"
    fi
  done
  local set contexts=""
  for set in "$@"; do
    [[ -n "${CONTEXT[$set]:-}" ]] && contexts+="${CONTEXT[$set]}=$dir/$set"$'\n'
  done
  echo "::notice title=Prebuilt tools::$tag ($([[ $exact == true ]] && echo "built from this commit's tools inputs" || echo "older than this commit's tools inputs")) for $*: no cargo compile on this runner"
  {
    echo "ok=true"
    echo "dir=$dir"
    echo "tag=$tag"
    echo "version=$(jq -r .version "$m")"
    echo "exact=$exact"
    echo "build_id=$(jq -r .build_id "$m")"
    echo "source_commit=$(jq -r .source_commit "$m")"
    echo "contexts<<FV_EOF"
    printf '%s' "$contexts"
    echo "FV_EOF"
  } >>"$OUT"
}

# Test binaries keep the pod's absolute source paths (file!(), CARGO_MANIFEST_DIR
# baked in); the pod's worktree path is linked to this checkout so they resolve.
cmd_run_tests() {
  local dir="${1:?dir}" root="${FV_TESTS_ROOT:-$ROOT}" src fail=0 n=0 bin pkg crate kind name
  src="$(cat "$dir/src-root")"; src="${src%/}"
  if [[ "$src" != "$root" ]] && { [[ -L "$src" ]] || [[ ! -e "$src" ]]; }; then
    local sudo=""
    [[ "$(id -u)" != 0 ]] && sudo -n true 2>/dev/null && sudo=sudo
    { $sudo mkdir -p "$(dirname "$src")" && $sudo ln -sfn "$root" "$src"; } 2>/dev/null \
      || log "could not link $src to the checkout (tests that read sources may fail)"
  fi
  while IFS=$'\t' read -r bin pkg crate kind name; do
    n=$((n + 1))
    echo "::group::$pkg ($kind $name)"
    chmod +x "$dir/$bin"
    if ! (cd "$root/$crate" && CARGO_MANIFEST_DIR="$root/$crate" CARGO_PKG_NAME="$pkg" "$dir/$bin"); then
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
  *) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 2 ;;
esac
