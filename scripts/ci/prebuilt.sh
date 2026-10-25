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
#         With FV_PREBUILT_SELECT=newest (vast-pytorch-image: it always ships
#         the newest tools), 2 is skipped: the highest SemVer release, always.
#         3. none matches (HEAD changed crates/, Cargo.lock, ... since the last
#            release): FV_PREBUILT_STALE=latest uses the highest SemVer release
#            anyway, with a warning, and the image is labelled with it (the
#            image workflows on main); FV_PREBUILT_STALE=compile (default; tests
#            and branch images, which must run their own code) falls back.
#       Writes to $GITHUB_OUTPUT:
#         ok=true|false     every requested set is usable (false: compile)
#         dir, tag, version, exact=true|false, build_id, source_commit
#         contexts=<name=path lines>   Docker named build contexts that replace
#                                      the compile stages (see CONTEXT below;
#                                      boot-build when a serve set has out/fv-boot
#                                      and the release is exact (see contexts),
#                                      lab-build when the gpucheck set has out/fv-lab)
#       Fallback (ok=false, a loud ::warning::, the workflow compiles as
#       before) unless FV_PREBUILT_REQUIRE=1, which fails instead.
#   prebuilt.sh contexts [--stale] <dir> <set>...
#       The contexts lines fetch writes, for sets already in <dir> (tests);
#       --stale: the release is not built from HEAD's tools inputs, so its
#       fv-boot is not used (boot-build compiles this checkout's).
#   prebuilt.sh run-tests <dir>
#       Run the gpucheck-tests set (gpucheck-t0's unit tests, compiled on the
#       pod) from the checkout (or $FV_TESTS_ROOT), each in its crate directory.
#       FV_TESTS_SHARD=k/N runs only every N-th test (k = 1..N), so N jobs
#       share the run.
#
# Env: GH_TOKEN / GITHUB_TOKEN (read access; anonymous works for a public
# repository); FV_PREBUILT_DIR (default $RUNNER_TEMP/prebuilt);
# FV_PREBUILT_FEATURES ("set=features …", e.g. "serve-cuda=cuda,http-client");
# FV_PREBUILT_STALE; FV_PREBUILT_SELECT (exact | newest); FV_TOOLS_VERSION;
# FV_PREBUILT_DISABLE=1 (always compile); FV_PREBUILT_REQUIRE=1 (no fallback:
# fail the job).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TR="$HERE/tools-release.sh"
OUT="${GITHUB_OUTPUT:-/dev/null}"
log() { printf '[prebuilt %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

# set -> the Dockerfile stage its directory replaces (a named build context)
declare -A CONTEXT=(
  [oxide]=oxide [gpucheck]=binary [gpucheck-vast]=binary [hf-fm]=hf-fm
  [serve-cuda]=serve-build [serve-cpu]=cpu-build
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
  h="$(bash "$TR" input-hash HEAD)" || fallback "cannot hash this checkout's tools inputs (git)"
  local args=(--input-hash "$h")
  [[ "${FV_PREBUILT_SELECT:-exact}" == newest ]] && args+=(--newest)
  [[ -n "${FV_TOOLS_VERSION:-}" ]] && args=(--version "${FV_TOOLS_VERSION#v}" --input-hash "$h")
  rel="$(bash "$TR" resolve "${args[@]}")" \
    || fallback "no usable tools release in ${GITHUB_REPOSITORY:-the repository} (${FV_TOOLS_VERSION:+pin $FV_TOOLS_VERSION, }input hash ${h:0:16}): publish one with scripts/ci/tools-release.sh publish (docs/dev/tools-releases.md)"
  tag="$(jq -r .tag <<<"$rel")"; exact="$(jq -r .exact <<<"$rel")"
  if [[ "$exact" != true ]]; then
    if [[ "${FV_PREBUILT_SELECT:-exact}" == newest && -z "${FV_TOOLS_VERSION:-}" ]]; then
      echo "::notice title=Newest tools release::$tag (built from $(jq -r '.source_commit[0:12]' <<<"$rel"))"
    elif [[ -n "${FV_TOOLS_VERSION:-}" ]]; then
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
  local contexts stale_arg=()
  [[ "$exact" == true ]] || stale_arg=(--stale)
  contexts="$(cmd_contexts "${stale_arg[@]}" "$dir" "$@")"$'\n'
  local compiles="no cargo compile on this runner"
  if [[ "$contexts" != *"boot-build="* ]] && [[ " $* " == *" serve-cuda "* || " $* " == *" serve-cpu "* ]]; then
    if [[ "$exact" == true ]]; then
      echo "::notice title=fv-boot not in $tag::the image's boot-build stage compiles fv-boot (one small crate) until a tools release carries it"
    else
      echo "::notice title=fv-boot compiled from this commit::$tag is older than this commit's tools inputs: its fv-boot is not used (it enforces this commit's image contract, e.g. the refusal without a config), so the boot-build stage compiles this checkout's"
    fi
    compiles="only the boot-build stage compiles (fv-boot)"
  fi
  if [[ "$contexts" != *"lab-build="* ]] && [[ " $* " == *" gpucheck "* ]]; then
    echo "::notice title=fv-lab not in $tag::the image's lab-build stage compiles fv-lab (ADR-0019) until a tools release carries it"
    compiles="${compiles/no cargo compile on this runner/only the lab-build stage compiles (fv-lab)}"
    [[ "$compiles" == *fv-lab* ]] || compiles="$compiles; the lab-build stage compiles (fv-lab)"
  fi
  echo "::notice title=Prebuilt tools::$tag ($([[ $exact == true ]] && echo "built from this commit's tools inputs" || echo "older than this commit's tools inputs")) for $*: $compiles"
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

# The named build contexts of fetched sets in <dir> (one name=path line each). fv-boot
# (ADR-0014) rides in the serve sets of releases built after it and then replaces the
# boot-build stage too; a release from before fv-boot has none, and boot-build compiles it.
# --stale (a release not built from HEAD's tools inputs: main's FV_PREBUILT_STALE=latest,
# a FV_TOOLS_VERSION pin): its fv-boot is not used either. fv-boot is the image's
# entrypoint and enforces the image contract of the Dockerfile it ships in (since ADR-0011,
# the refusal of a boot without a config); an older one paired with a newer Dockerfile
# booted fv-serve's default config instead, and the serve-image smoke's refusal check
# waited on that server forever (serve-image run 286, tools-v0.1.50 in c3ceb96c's images).
# fv-lab (ADR-0019) rides in the gpucheck set the same way and replaces lab-build.
cmd_contexts() {
  local stale=0
  [[ "${1:-}" == --stale ]] && { stale=1; shift; }
  local dir="${1:?dir}" set boot="" lab=""; shift
  for set in "$@"; do
    [[ -n "${CONTEXT[$set]:-}" ]] && echo "${CONTEXT[$set]}=$dir/$set"
    if [[ "$stale" == 0 && -z "$boot" && ( "$set" == serve-cuda || "$set" == serve-cpu ) && -x "$dir/$set/out/fv-boot" ]]; then boot="$dir/$set"; fi
    if [[ "$set" == gpucheck && -x "$dir/$set/out/fv-lab" ]]; then lab="$dir/$set"; fi
  done
  [[ -z "$boot" ]] || echo "boot-build=$boot"
  [[ -z "$lab" ]] || echo "lab-build=$lab"
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
    local filt=()
    if [[ -n "${FV_TESTS_SHARD:-}" ]]; then
      # Shard k/N: every N-th test of this binary (by its --list order), by
      # exact name. A shard with no test of this binary skips it.
      local k="${FV_TESTS_SHARD%/*}" nn="${FV_TESTS_SHARD#*/}"
      mapfile -t filt < <(cd "$root/$crate" && "$dir/$bin" --list --format terse 2>/dev/null \
        | sed -n 's/: test$//p' | awk -v k="$k" -v n="$nn" '(NR - 1) % n == k - 1')
      if (( ${#filt[@]} == 0 )); then echo "(no tests of this binary in shard $FV_TESTS_SHARD)"; echo "::endgroup::"; continue; fi
      echo "shard $FV_TESTS_SHARD: ${#filt[@]} tests"
      filt=(--exact "${filt[@]}")
    fi
    if ! (cd "$root/$crate" && CARGO_MANIFEST_DIR="$root/$crate" CARGO_PKG_NAME="$pkg" "$dir/$bin" ${filt[@]+"${filt[@]}"}); then
      fail=1; echo "::error title=unit tests::$pkg $kind $name failed"
    fi
    echo "::endgroup::"
  done <"$dir/tests.tsv"
  (( n > 0 )) || { echo "::error::no test binaries in $dir/tests.tsv"; exit 1; }
  return "$fail"
}

case "${1:-}" in
  fetch) shift; cmd_fetch "$@" ;;
  contexts) shift; cmd_contexts "$@" ;;
  run-tests) shift; cmd_run_tests "$@" ;;
  *) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 2 ;;
esac
