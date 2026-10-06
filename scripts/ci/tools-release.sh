#!/usr/bin/env bash
# Tools releases on GitHub Releases (docs/dev/tools-releases.md): the build
# pod's prebuilt binaries (fv-serve cuda/cpu/fake, fv-gpucheck,
# gpucheck-vast, hf-fm, the gpucheck unit-test binaries, the oxide cubins),
# published as one SemVer release per tested set, `tools-v<MAJOR.MINOR.PATCH>`,
# and consumed by the image workflows instead of compiling.
#
#   tools-release.sh input-hash [rev]     content hash of the tools' inputs at rev
#   tools-release.sh version [rev]        the tools version at rev
#                                         ([workspace.package] version, Cargo.toml)
#   tools-release.sh status [rev]         version, input hash, latest release, and
#                                         whether rev needs a release / a bump
#   tools-release.sh list                 tools releases (version, tag, input hash,
#                                         source commit), highest version first
#   tools-release.sh resolve [--version V] [--input-hash H] [--exact|--newest]
#                                         pick a release; prints its JSON
#                                         (.exact: its input hash is H)
#   tools-release.sh fetch <tag> <dir> <set>...
#                                         download manifest.json + the sets, check
#                                         every sha256, extract to <dir>/<set>/
#   tools-release.sh bump <breaking|feature|fix>
#                                         raise the workspace version (Cargo.toml +
#                                         Cargo.lock) for a MINOR/MAJOR change
#   tools-release.sh notes <rev>          release notes since the previous release
#   tools-release.sh plan [rev] [--prerelease] [--force]
#                                         the version/tag a publish would use, or
#                                         skip (inputs already released); JSON
#   tools-release.sh build <rev> --local|--pod --version V --tag tools-vV [--out DIR]
#                                         compile + test gate + stage DIR/release
#   tools-release.sh pick-runner          where tools-release.yml builds: `pod` (an
#                                         online, idle self-hosted runner labelled
#                                         fv-build) or `github` (fallback)
#   tools-release.sh build-sets <rev> --sets "a b" --version V --out DIR [--test]
#   tools-release.sh assemble <rev> --version V --tag tools-vV --in DIR
#                                         the GitHub-hosted fallback: some sets per
#                                         parallel job, then join + -V + stage
#   tools-release.sh upload <DIR/release> [--dispatch]
#                                         draft -> assets -> publish -> verify ->
#                                         prune [-> dispatch the image workflows]
#   tools-release.sh publish <rev> [--prerelease] [--dispatch] [--dry-run|--no-upload]
#                                         plan + build --pod + upload, by hand from
#                                         the coordinator (the default publisher is
#                                         .github/workflows/tools-release.yml)
#   tools-release.sh prune [--keep N] [--dry-run]
#                                         delete tools releases beyond the newest
#                                         N stable ones (default 10) and stale
#                                         prereleases; never any other release
#
# Versioning (SemVer, pre-1.0 like Cargo): the version is the workspace
# version in Cargo.toml, which every binary reports (`fv-serve -V`,
# `fv-gpucheck -V`, /health). From 1.0: MAJOR = a breaking change to a tool's
# CLI/flags, config file format, wire/API protocol between components
# (fv-serve <-> edge/fv-control, dispatch protocol version) or the on-disk /
# weights layout the tools need; MINOR = a backwards-compatible feature (model,
# recipe, arm, flag, endpoint); PATCH = fixes and perf with no interface
# change. In 0.x (now): breaking -> MINOR, feature or fix -> PATCH.
# The release version needs no human step: the workspace version when it is
# above the highest release (a `bump` merged in the PR), else that release's
# PATCH + 1; the build embeds it (FV_RELEASE_VERSION -> `-V`, /health), so
# nothing is committed back. Inputs unchanged since a release: nothing to do.
#
# Env: FV_GITHUB_REPO (default $GITHUB_REPOSITORY or zaitrarrio/fastvideo-rs);
# auth: FV_GITHUB_TOKEN_FILE (publish/prune default ~/.config/fv/github_token,
# mode 600, read here only and sent only to api/uploads.github.com), else
# GH_TOKEN / GITHUB_TOKEN (CI), else anonymous (public repo, 60 requests/h).
# publish: FV_TOOLS_KEEP (10), FV_RELEASE_AGENT (fv-release) and
# FV_TOOLS_TEST_AGENT (<release agent>-test) on the build pod, FV_TOOLS_OUT
# (default artifacts/tools-release/<sha>), FV_TOOLS_DISPATCH (workflows run
# by --dispatch).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
REPO="${FV_GITHUB_REPO:-${GITHUB_REPOSITORY:-zaitrarrio/fastvideo-rs}}"
API="${GITHUB_API_URL:-https://api.github.com}"
UPLOADS="${FV_GITHUB_UPLOADS:-https://uploads.github.com}"
PREFIX="tools-v"
# The tools' inputs: every path whose content can change a binary of the set.
# The Dockerfiles are not here: with releases they only assemble images.
INPUTS=(crates profiles Cargo.toml Cargo.lock rust-toolchain.toml scripts/gpu/cuda-13.pins
        third_party/cutile-rs scripts/dev/release-artifacts-pod.sh docker/build-base.Dockerfile)
INPUT_SCHEMA="fv-tools-inputs/1"
SEMVER_RE='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?$'

log() { printf '[tools %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }

CLEANUP=()
# Last registered first (the token header file goes last).
cleanup() { local i; for (( i = ${#CLEANUP[@]} - 1; i >= 0; i-- )); do eval "${CLEANUP[$i]}" || true; done; }
trap cleanup EXIT

# ---- GitHub API -------------------------------------------------------------
# The token goes into a mode-600 header file (never on a command line, never
# printed), removed on exit. auth runs in the main shell (the dispatch below),
# never first inside a $(...) subshell, whose EXIT trap would not remove it.
HDR=""
auth() {
  [[ -n "$HDR" ]] && return 0
  local tok="" f="${FV_GITHUB_TOKEN_FILE:-}"
  # Writes: the job's token in a workflow, else the coordinator's token file.
  if [[ -z "$f" && "${FV_TOOLS_WRITE:-0}" == 1 && -z "${GH_TOKEN:-${GITHUB_TOKEN:-}}" ]]; then
    f="${XDG_CONFIG_HOME:-$HOME/.config}/fv/github_token"
  fi
  if [[ -n "$f" ]]; then
    [[ -r "$f" ]] || die "no token file $f (FV_GITHUB_TOKEN_FILE)"
    tok="$(tr -d ' \r\n' <"$f")"
  else
    tok="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
  fi
  HDR="$(umask 077 && mktemp)"
  CLEANUP+=("rm -f '$HDR'")
  printf 'X-GitHub-Api-Version: 2022-11-28\nUser-Agent: fv-tools-release\n' >"$HDR"
  [[ -n "$tok" ]] && printf 'Authorization: Bearer %s\n' "$tok" >>"$HDR"
  return 0
}
# api <METHOD> <path|url> [curl args...]: JSON in, JSON out; non-2xx fails
# with GitHub's message (not the request headers).
api() {
  auth
  local m="$1" u="$2"; shift 2
  [[ "$u" == http* ]] || u="$API$u"
  [[ -n "$HDR" ]] || die "api: auth not initialised"
  local out code rc=0
  out="$(mktemp)"
  code="$(curl -sS -X "$m" -H @"$HDR" -H 'Accept: application/vnd.github+json' -o "$out" -w '%{http_code}' "$@" "$u")" \
    || { log "$m ${u#"$API"}: curl failed"; rm -f "$out"; return 3; }
  if [[ "$code" != 2* ]]; then
    log "$m ${u#"$API"}: HTTP $code: $(jq -r '.message // empty' "$out" 2>/dev/null | head -c 300)"
    rc=3; [[ "$code" == 404 ]] && rc=4
  else
    cat "$out"
  fi
  rm -f "$out"
  return "$rc"
}
download_asset() { # <asset id> <dest>
  curl -sS --fail -L --retry 3 -H @"$HDR" -H 'Accept: application/octet-stream' \
    -o "$2" "$API/repos/$REPO/releases/assets/$1"
}

# ---- inputs and version -----------------------------------------------------
rev_sha() { git -c safe.directory="$ROOT" -C "$ROOT" rev-parse -q --verify "${1:-HEAD}^{commit}" || die "unknown revision ${1:-HEAD}"; }

# sha256 over the git tree entries (mode, blob/gitlink id, path) of INPUTS at
# a commit: a content hash that needs no checkout of that commit.
input_hash() {
  local sha; sha="$(rev_sha "${1:-HEAD}")"
  { echo "$INPUT_SCHEMA"; git -c safe.directory="$ROOT" -C "$ROOT" ls-tree -r --full-tree "$sha" -- "${INPUTS[@]}"; } | sha256sum | cut -d' ' -f1
}

tools_version() {
  local sha v; sha="$(rev_sha "${1:-HEAD}")"
  v="$(git -c safe.directory="$ROOT" -C "$ROOT" show "$sha:Cargo.toml" \
    | awk '/^\[/{s=($0=="[workspace.package]")} s && /^version *=/{gsub(/.*= *"|".*/,""); print; exit}')"
  [[ "$v" =~ $SEMVER_RE ]] || die "no SemVer [workspace.package] version in Cargo.toml at ${sha:0:12} ('$v')"
  echo "$v"
}

# ---- releases -----------------------------------------------------------------
# One JSON object per tools release (drafts included, flagged), sorted by
# SemVer, highest first. Machine-readable lines of the release body:
#   input-hash: <sha256>   source-commit: <sha>   manifest-sha256: <sha256>
RELEASES_CACHE=""
releases() { [[ -n "$RELEASES_CACHE" ]] || die "releases not loaded"; echo "$RELEASES_CACHE"; }
load_releases() {
  RELEASES_CACHE=""
  {
    local page=1 batch all='[]'
    # Tests: FV_TOOLS_RELEASES_FILE stands in for the API's release list.
    [[ -n "${FV_TOOLS_RELEASES_FILE:-}" ]] && { all="$(cat "$FV_TOOLS_RELEASES_FILE")"; page=99; }
    while (( page <= 20 )); do
      batch="$(api GET "/repos/$REPO/releases?per_page=100&page=$page")" || die "cannot list releases of $REPO"
      all="$(jq -c --argjson b "$batch" '. + $b' <<<"$all")"
      (( $(jq length <<<"$batch") == 100 )) || break
      page=$((page + 1))
    done
    RELEASES_CACHE="$(jq -c --arg p "$PREFIX" '
      def field($k): ((.body // "") | capture("(?m)^" + $k + ": *(?<v>[0-9a-f]+)") | .v) // "";
      def key: (.version | split("-")) as $s | ($s[0] | split(".") | map(tonumber)) + [(if ($s | length) > 1 then 0 else 1 end)];
      [ .[] | select(.tag_name | startswith($p))
        | {tag: .tag_name, version: (.tag_name | ltrimstr($p)), id, draft, prerelease,
           url: .html_url, created: .created_at,
           input_hash: field("input-hash"), source_commit: field("source-commit"),
           manifest_sha256: field("manifest-sha256"),
           assets: [.assets[] | {name, id, size, digest}]}
        | select(.version | test("^[0-9]+\\.[0-9]+\\.[0-9]+(-[0-9A-Za-z.-]+)?$")) ]
      | sort_by(key) | reverse' <<<"$all")"
  }
}

cmd_list() {
  releases | jq -r '.[] | [.version, .tag, (if .draft then "draft" elif .prerelease then "pre" else "stable" end),
    .input_hash[0:16], .source_commit[0:12], .created] | @tsv'
}

# resolve [--version V] [--input-hash H] [--exact]
#   --version V   that release (a rollback pin), whatever its inputs
#   --input-hash  prefer the highest release (prereleases too) built from
#                 exactly these inputs
#   --exact       fail (exit 1) unless one matches the input hash
#   --newest      the highest stable SemVer release even when an older one
#                 matches the input hash (.exact still reports the match)
#   otherwise     the highest stable SemVer release
cmd_resolve() {
  local want_v="" h="" exact=0 newest=0
  while (( $# )); do
    case "$1" in
      --version) want_v="${2#v}"; shift 2 ;;
      --input-hash) h="$2"; shift 2 ;;
      --exact) exact=1; shift ;;
      --newest) newest=1; shift ;;
      *) die "resolve: unknown argument $1" ;;
    esac
  done
  local rels r
  rels="$(releases | jq -c '[.[] | select(.draft | not)]')"
  if [[ -n "$want_v" ]]; then
    r="$(jq -c --arg v "$want_v" 'map(select(.version == $v)) | .[0] // empty' <<<"$rels")"
    [[ -n "$r" ]] || { log "no release $PREFIX$want_v"; return 1; }
  else
    [[ -n "$h" ]] && (( !newest )) && r="$(jq -c --arg h "$h" 'map(select(.input_hash == $h)) | .[0] // empty' <<<"$rels")"
    if [[ -z "${r:-}" ]]; then
      (( exact )) && { log "no tools release built from input hash ${h:0:16}"; return 1; }
      r="$(jq -c 'map(select(.prerelease | not)) | .[0] // empty' <<<"$rels")"
      [[ -n "$r" ]] || { log "no tools release in $REPO"; return 1; }
    fi
  fi
  jq -c --arg h "$h" '. + {exact: ($h != "" and .input_hash == $h)}' <<<"$r"
}

sha256() { sha256sum "$1" | cut -d' ' -f1; }

# fetch <tag> <dir> <set>...: manifest checked against the release body, each
# tarball against the manifest, each extracted file (and nothing else) too.
cmd_fetch() {
  local tag="${1:?tag}" dir="${2:?dir}"; shift 2
  (( $# )) || die "fetch: no sets"
  local rel m id want set tb f
  rel="$(releases | jq -c --arg t "$tag" '.[] | select(.tag == $t)')"
  [[ -n "$rel" ]] || die "no release $tag"
  rm -rf "$dir" && mkdir -p "$dir"
  m="$dir/manifest.json"
  id="$(jq -r '.assets[] | select(.name == "manifest.json") | .id' <<<"$rel")"
  [[ -n "$id" ]] || die "$tag has no manifest.json"
  download_asset "$id" "$m" || die "$tag: manifest.json download failed"
  want="$(jq -r .manifest_sha256 <<<"$rel")"
  [[ -n "$want" && "$(sha256 "$m")" == "$want" ]] || die "$tag: manifest.json sha256 differs from the release body"
  [[ "$(jq -r .tag "$m")" == "$tag" ]] || die "$tag: manifest is for $(jq -r .tag "$m")"
  for set in "$@"; do
    jq -e --arg s "$set" '.sets[$s]' "$m" >/dev/null || die "$tag: set $set is not in the manifest"
    tb="$(jq -r --arg s "$set" '.sets[$s].tarball' "$m")"
    want="$(jq -r --arg s "$set" '.sets[$s].sha256' "$m")"
    id="$(jq -r --arg n "$tb" '.assets[] | select(.name == $n) | .id' <<<"$rel")"
    [[ -n "$id" ]] || die "$tag: asset $tb missing"
    download_asset "$id" "$dir/$tb" || die "$tag: $tb download failed"
    [[ "$(sha256 "$dir/$tb")" == "$want" ]] || die "$tag: $tb sha256 differs from the manifest"
    mkdir -p "$dir/$set"
    tar -xzf "$dir/$tb" -C "$dir/$set"
    rm -f "$dir/$tb"
    diff <(cd "$dir/$set" && find . -type f -printf '%P\n' | LC_ALL=C sort) \
         <(jq -r --arg s "$set" '.sets[$s].files | keys[]' "$m" | LC_ALL=C sort) >/dev/null \
      || die "$tag: $set: files differ from the manifest"
    while IFS=$'\t' read -r f want; do
      [[ "$(sha256 "$dir/$set/$f")" == "$want" ]] || die "$tag: $set/$f sha256 differs from the manifest"
    done < <(jq -r --arg s "$set" '.sets[$s].files | to_entries[] | [.key, .value.sha256] | @tsv' "$m")
    log "$tag $set: $(jq -r --arg s "$set" '.sets[$s].files | length' "$m") files verified ($(du -sh "$dir/$set" | cut -f1))"
  done
}

cmd_status() {
  local sha h v r
  sha="$(rev_sha "${1:-HEAD}")"; h="$(input_hash "$sha")"; v="$(tools_version "$sha")"
  echo "commit      $sha"
  echo "version     $v"
  echo "input hash  $h"
  if r="$(cmd_resolve --input-hash "$h" 2>/dev/null)"; then
    echo "latest      $(jq -r '"\(.tag) (\(.source_commit[0:12]), input \(.input_hash[0:16]))"' <<<"$r")"
    if [[ "$(jq -r .exact <<<"$r")" == true ]]; then
      echo "state       released as $(jq -r .tag <<<"$r") (inputs unchanged)"
    else
      echo "state       inputs changed since $(jq -r .tag <<<"$r"): the next publish is $PREFIX$(next_version "$v" "$(releases | jq -r '[.[] | select((.draft or .prerelease) | not)] | .[0].version // empty')")"
    fi
  else
    echo "latest      (none)"
    echo "state       no tools release yet: ready to publish $PREFIX$v"
  fi
}

# version_gt A B: A > B in SemVer precedence (release > its prereleases).
version_gt() {
  [[ "$1" != "$2" ]] && [[ "$(jq -rn --arg a "$1" --arg b "$2" '
    def key: split("-") as $s | ($s[0] | split(".") | map(tonumber)) + [(if ($s | length) > 1 then 0 else 1 end)];
    [$a, $b] | sort_by(key) | .[1]')" == "$1" ]]
}

# ---- bump -------------------------------------------------------------------
cmd_bump() {
  local kind="${1:?bump breaking|feature|fix}" v maj min pat new
  v="$(awk '/^\[/{s=($0=="[workspace.package]")} s && /^version *=/{gsub(/.*= *"|".*/,""); print; exit}' "$ROOT/Cargo.toml")"
  [[ "$v" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || die "workspace version '$v' is not X.Y.Z"
  maj="${BASH_REMATCH[1]}"; min="${BASH_REMATCH[2]}"; pat="${BASH_REMATCH[3]}"
  case "$kind:$(( maj == 0 ))" in
    breaking:1) new="0.$((min + 1)).0" ;;
    feature:1|fix:1) new="0.$min.$((pat + 1))" ;;
    breaking:0) new="$((maj + 1)).0.0" ;;
    feature:0) new="$maj.$((min + 1)).0" ;;
    fix:0) new="$maj.$min.$((pat + 1))" ;;
    *) die "bump: breaking, feature or fix" ;;
  esac
  python3 - "$ROOT" "$v" "$new" <<'PY'
import pathlib, re, sys
root, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
toml = root / "Cargo.toml"
text = toml.read_text()
i = text.index("[workspace.package]")
j = text.index("\n[", i + 1) if "\n[" in text[i + 1:] else len(text)
sec = re.sub(r'(?m)^version = "[^"]*"', f'version = "{new}"', text[i:j], count=1)
toml.write_text(text[:i] + sec + text[j:])
# Cargo.lock: the workspace members that inherit the version (no `source`).
names = set()
for m in root.glob("crates/*/Cargo.toml"):
    t = m.read_text()
    if re.search(r"(?m)^version\.workspace *= *true", t):
        names.add(re.search(r'(?m)^name *= *"([^"]+)"', t).group(1))
lock = root / "Cargo.lock"
blocks = lock.read_text().split("\n[[package]]\n")
n = 0
for k, b in enumerate(blocks):
    nm = re.search(r'(?m)^name = "([^"]+)"', b)
    if nm and nm.group(1) in names and "\nsource = " not in b and f'\nversion = "{old}"' in b:
        blocks[k] = b.replace(f'\nversion = "{old}"', f'\nversion = "{new}"', 1)
        n += 1
lock.write_text("\n[[package]]\n".join(blocks))
print(f"Cargo.lock: {n} workspace packages {old} -> {new}", file=sys.stderr)
PY
  log "tools version $v -> $new ($kind); commit Cargo.toml and Cargo.lock"
}

# ---- notes ------------------------------------------------------------------
# Commits that touched the inputs since the previous stable release's source
# commit (or the last 50 such commits when there is none).
cmd_notes() {
  local sha prev range
  sha="$(rev_sha "${1:-HEAD}")"
  prev="$(releases | jq -r '[.[] | select((.draft or .prerelease) | not)] | .[0].source_commit // empty')"
  if [[ -n "$prev" ]] && ! git -c safe.directory="$ROOT" -C "$ROOT" cat-file -e "$prev^{commit}" 2>/dev/null; then
    git -c safe.directory="$ROOT" -C "$ROOT" fetch -q --deepen=1000 origin 2>/dev/null || true
  fi
  if [[ -n "$prev" ]] && git -c safe.directory="$ROOT" -C "$ROOT" cat-file -e "$prev^{commit}" 2>/dev/null; then
    range="$prev..$sha"
    echo "Changes to the tools' inputs since $(releases | jq -r '[.[] | select((.draft or .prerelease) | not)] | .[0].tag') (${prev:0:12}):"
  else
    range="$sha"
    echo "Recent changes to the tools' inputs:"
  fi
  echo
  git -c safe.directory="$ROOT" -C "$ROOT" log --no-merges --format='- %h %s' -n 50 "$range" -- "${INPUTS[@]}"
}

# ---- plan / build / upload / publish -----------------------------------------
# Two publishers share these stages (docs/dev/tools-releases.md):
#   .github/workflows/tools-release.yml (the default, on every push to main):
#     plan -> build --local on the build pod's self-hosted runner -> upload
#     from a GitHub-hosted job with the job's GITHUB_TOKEN;
#   the coordinator by hand: publish = plan -> build --pod -> upload.
SETS_ALL="oxide gpucheck gpucheck-vast hf-fm serve-cuda serve-cpu serve-fake gpucheck-tests"

# next_version <workspace version> <highest released version or "">: the
# workspace version when it is above the highest release (a manual bump in
# the PR), else that release's PATCH + 1 (0.x too). No commit back to main:
# the build embeds the version (FV_RELEASE_VERSION).
next_version() {
  local w="$1" l="$2"
  if [[ -z "$l" ]] || version_gt "$w" "$l"; then echo "$w"; return 0; fi
  [[ "$l" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || die "highest release $l is not X.Y.Z"
  echo "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.$(( BASH_REMATCH[3] + 1 ))"
}

# plan <rev> [--prerelease]: what a publish of rev would do, as JSON
# {sha, input_hash, workspace_version, latest, version, tag, prerelease, skip,
# skip_reason}; also key=value lines to $GITHUB_OUTPUT. skip: a release with
# these inputs exists (nothing to build).
PLAN=""
cmd_plan() {
  local rev="HEAD" pre=0 force=0
  while (( $# )); do
    case "$1" in
      --prerelease) pre=1; shift ;;
      --force) force=1; shift ;;   # a dry run: build even when released
      -*) die "plan: unknown flag $1" ;;
      *) rev="$1"; shift ;;
    esac
  done
  local sha h w latest v tag same skip=false why=""
  sha="$(rev_sha "$rev")"; h="$(input_hash "$sha")"; w="$(tools_version "$sha")"
  latest="$(releases | jq -r '[.[] | select((.draft or .prerelease) | not)] | .[0].version // empty')"
  v="$(next_version "$w" "$latest")"
  if (( pre )); then v="$v-pre.${sha:0:12}"; fi
  tag="$PREFIX$v"
  same="$(releases | jq -r --arg h "$h" '[.[] | select((.draft | not) and .input_hash == $h)] | .[0].tag // empty')"
  if [[ -n "$same" ]] && (( force )); then
    why="(forced) the inputs of $same (input hash ${h:0:16}); building anyway"
  elif [[ -n "$same" ]]; then
    skip=true; why="tools unchanged: ${sha:0:12} has the inputs of $same (input hash ${h:0:16})"
  elif (( force )) && releases | jq -e --arg t "$tag" 'any(.[]; .tag == $t and (.draft | not))' >/dev/null; then
    why="(forced) $tag exists"
  elif releases | jq -e --arg t "$tag" 'any(.[]; .tag == $t and (.draft | not))' >/dev/null; then
    die "tag $tag already exists but its inputs differ (input hash ${h:0:16}): another publish raced this one?"
  fi
  PLAN="$(jq -nc --arg sha "$sha" --arg h "$h" --arg w "$w" --arg l "$latest" --arg v "$v" --arg t "$tag" \
    --argjson pre "$([[ $pre == 1 ]] && echo true || echo false)" --argjson skip "$skip" --arg why "$why" \
    '{sha: $sha, input_hash: $h, workspace_version: $w, latest: $l, version: $v, tag: $t, prerelease: $pre,
      skip: $skip, skip_reason: $why}')"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    jq -r 'to_entries[] | "\(.key)=\(.value)"' <<<"$PLAN" >>"$GITHUB_OUTPUT"
  fi
  if [[ "$skip" == true ]]; then log "$why; nothing to publish"; else log "plan: $tag from ${sha:0:12} (input hash ${h:0:16}; workspace $w; highest release ${latest:-none})"; fi
}

# build <rev> --pod|--local --version V --tag T [--prerelease] [--out DIR]:
# compile every set, run the gate, and stage the release in DIR/release
# (tarballs, manifest.json, body.md). Nothing is uploaded.
#   --local  on this machine (the build pod's GitHub runner): this checkout
#            must be rev, clean; scripts/dev/release-artifacts-pod.sh runs here
#            with CARGO_TARGET_DIR (default $FV_BUILD_TARGET_BASE/gh-runner),
#            the gate's check.sh in <that>-test.
#   --pod    from the coordinator: build-pod.sh release-artifacts + its jobs.
cmd_build() {
  local rev="" mode="" v="" tag="" pre=0 out=""
  while (( $# )); do
    case "$1" in
      --pod) mode=pod; shift ;;
      --local) mode=local; shift ;;
      --version) v="$2"; shift 2 ;;
      --tag) tag="$2"; shift 2 ;;
      --prerelease) pre=1; shift ;;
      --out) out="$2"; shift 2 ;;
      -*) die "build: unknown flag $1" ;;
      *) rev="$1"; shift ;;
    esac
  done
  [[ -n "$rev" && -n "$mode" && "$v" =~ $SEMVER_RE && "$tag" == "$PREFIX$v" ]] \
    || die "usage: tools-release.sh build <rev> --pod|--local --version X.Y.Z --tag tools-vX.Y.Z [--prerelease] [--out DIR]"
  local sha h t0=$SECONDS
  sha="$(rev_sha "$rev")"; h="$(input_hash "$sha")"
  out="${out:-${FV_TOOLS_OUT:-$ROOT/artifacts/tools-release/$sha}}"
  rm -rf "$out" && mkdir -p "$out"
  if [[ "$mode" == pod ]]; then
    # The build runs this checkout's recipe; it must be the one the hash covers.
    cmp -s <(git -c safe.directory="$ROOT" -C "$ROOT" show "$sha:scripts/dev/release-artifacts-pod.sh") "$ROOT/scripts/dev/release-artifacts-pod.sh" \
      || die "scripts/dev/release-artifacts-pod.sh here differs from ${sha:0:12}'s (run from a checkout of that commit)"
    FV_RELEASE_VERSION="$v" FV_RELEASE_OUT="$out" bash "$ROOT/scripts/dev/build-pod.sh" release-artifacts "$sha" \
      || die "build failed on the build pod: nothing published"
  else
    # In a container the checkout belongs to another uid: let every git below
    # (docker.sh build-id, build scripts, submodules) read it. Command-line
    # scope (GIT_CONFIG_*), this build only.
    export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
    build_local "$sha" "$v" "$out" || die "build failed: nothing published"
  fi
  local m="$out/manifest.json" set
  [[ "$(jq -r .sha "$m")" == "$sha" ]] || die "build manifest is for $(jq -r .sha "$m")"
  for set in $SETS_ALL; do
    jq -e --arg s "$set" '.sets[$s]' "$m" >/dev/null || die "set $set missing from the build"
  done
  run_gate "$sha" "$v" "$out" "$mode" || die "tests failed: nothing published"
  stage_release "$sha" "$h" "$v" "$tag" "$pre" "$out"
  log "staged $tag in $out/release: built and tested in $(( SECONDS - t0 ))s"
}

build_local() {
  local sha="$1" v="$2" out="$3" sets="${4:-}" T shim
  [[ "$(git -c safe.directory="$ROOT" -C "$ROOT" rev-parse HEAD)" == "$sha" ]] || die "build --local: this checkout is not ${sha:0:12}"
  [[ -z "$(git -c safe.directory="$ROOT" -C "$ROOT" status --porcelain --untracked-files=no)" ]] || die "build --local: this checkout has local changes"
  git -c safe.directory="$ROOT" -C "$ROOT" submodule update -q --init third_party/cutile-rs
  T="${CARGO_TARGET_DIR:-${FV_BUILD_TARGET_BASE:?CARGO_TARGET_DIR or FV_BUILD_TARGET_BASE (the build pod runner sets it)}/gh-runner}"
  # scripts/gpu/docker.sh build-id wants shasum (perl), which the base image may lack.
  shim="$(mktemp -d)"; CLEANUP+=("rm -rf '$shim'")
  if ! command -v shasum >/dev/null; then
    printf '#!/bin/sh\n[ "$1" = -a ] && shift 2\nexec sha256sum "$@"\n' >"$shim/shasum"; chmod +x "$shim/shasum"
  fi
  local build_id build_time
  build_id="$(PATH="$shim:$PATH" bash "$ROOT/scripts/gpu/docker.sh" build-id)"
  build_time="$(cd "$ROOT" && TZ=UTC git log -1 --format=%cd --date=format-local:%Y-%m-%dT%H:%M:%SZ)"
  log "build --local ${sha:0:12} as $v (build id $build_id, CARGO_TARGET_DIR $T)"
  (cd "$ROOT" && CARGO_TARGET_DIR="$T" FV_REL_SHA="$sha" FV_GIT_SHA="$sha" FV_BUILD_TIME="$build_time" \
     FV_BUILD_ID="$build_id" FV_REL_RUN_ID="gh-${GITHUB_RUN_ID:-local}-$(date -u +%Y%m%dT%H%M%SZ)" FV_RELEASE_VERSION="$v" \
     FV_REL_SETS="$sets" bash scripts/dev/release-artifacts-pod.sh) || return 1
  cp "$T/release-artifacts/$sha/"*.tar.gz "$T/release-artifacts/$sha/manifest.json" "$out/"
  local tb want
  while IFS=$'\t' read -r tb want; do
    [[ "$(sha256 "$out/$tb")" == "$want" ]] || { log "$tb: sha256 differs from the manifest"; return 1; }
  done < <(jq -r '.sets[] | [.tarball, .sha256] | @tsv' "$out/manifest.json")
}

# build-sets <rev> --sets "a b" --version V --out DIR [--test]: some of the
# sets only (the GitHub-hosted fallback builds them in parallel jobs, one
# group per job; `assemble` joins them). --test also runs the shipped
# gpucheck/cudarc unit-test binaries (gate 2) when gpucheck-tests is built.
cmd_build_sets() {
  local rev="" sets="" v="" out="" test=0
  while (( $# )); do
    case "$1" in
      --sets) sets="$2"; shift 2 ;;
      --version) v="$2"; shift 2 ;;
      --out) out="$2"; shift 2 ;;
      --test) test=1; shift ;;
      -*) die "build-sets: unknown flag $1" ;;
      *) rev="$1"; shift ;;
    esac
  done
  [[ -n "$rev" && -n "$sets" && -n "$out" && "$v" =~ $SEMVER_RE ]] \
    || die "usage: tools-release.sh build-sets <rev> --sets \"a b\" --version X.Y.Z --out DIR [--test]"
  local sha set t0=$SECONDS; sha="$(rev_sha "$rev")"
  for set in $sets; do [[ " $SETS_ALL " == *" $set "* ]] || die "build-sets: unknown set $set"; done
  export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
  rm -rf "$out" && mkdir -p "$out"
  build_local "$sha" "$v" "$out" "$sets" || die "build of $sets failed"
  for set in $sets; do
    jq -e --arg s "$set" '.sets[$s]' "$out/manifest.json" >/dev/null || die "set $set missing from the build"
  done
  if (( test )) && [[ " $sets " == *" gpucheck-tests "* ]]; then
    local d="$out/gate"; rm -rf "$d" && mkdir -p "$d/gpucheck-tests"
    tar -xzf "$out/gpucheck-tests.tar.gz" -C "$d/gpucheck-tests"
    log "gate 2/3: the shipped gpucheck-tests binaries"
    FV_TESTS_ROOT="$ROOT" bash "$HERE/prebuilt.sh" run-tests "$d/gpucheck-tests" || die "unit tests failed"
    wc -l <"$d/gpucheck-tests/tests.tsv" >"$out/gate-unit-tests.count"
    rm -rf "$d"
  fi
  log "built $sets in $(( SECONDS - t0 ))s into $out"
}

# assemble <rev> --version V --tag T --in DIR [--prerelease] [--out DIR]: join
# the per-group builds (DIR/*/manifest.json + tarballs) into one release:
# every set exactly once, the same commit, every sha256; then gate 3 (`-V`)
# and stage DIR/release like `build`. Gates 1 (check.sh) and 2 (unit tests)
# ran in their own jobs, which the workflow requires first.
cmd_assemble() {
  local rev="" v="" tag="" in="" out="" pre=0
  while (( $# )); do
    case "$1" in
      --version) v="$2"; shift 2 ;;
      --tag) tag="$2"; shift 2 ;;
      --in) in="$2"; shift 2 ;;
      --out) out="$2"; shift 2 ;;
      --prerelease) pre=1; shift ;;
      -*) die "assemble: unknown flag $1" ;;
      *) rev="$1"; shift ;;
    esac
  done
  [[ -n "$rev" && -d "$in" && "$v" =~ $SEMVER_RE && "$tag" == "$PREFIX$v" ]] \
    || die "usage: tools-release.sh assemble <rev> --version X.Y.Z --tag tools-vX.Y.Z --in DIR [--out DIR]"
  local sha h ms; sha="$(rev_sha "$rev")"; h="$(input_hash "$sha")"
  out="${out:-$in/assembled}"; rm -rf "$out" && mkdir -p "$out"
  mapfile -t ms < <(find "$in" -mindepth 2 -maxdepth 2 -name manifest.json -not -path "$out/*" | LC_ALL=C sort)
  (( ${#ms[@]} )) || die "assemble: no */manifest.json under $in"
  jq -e -s --arg sha "$sha" 'all(.[]; .sha == $sha)' "${ms[@]}" >/dev/null || die "assemble: the builds are not all of ${sha:0:12}"
  # Sets must not repeat across groups; the rest comes from the first build,
  # with the per-set facts (oxide key, hf-fm version) from whichever has them.
  jq -e -s '[.[].sets | keys[]] | (length == (unique | length))' "${ms[@]}" >/dev/null || die "assemble: a set was built twice"
  jq -s '.[0] + {sets: (map(.sets) | add), oxide_key: ([.[].oxide_key | select(. != "" and . != null)][0] // ""),
         hf_fetch_model_version: ([.[].hf_fetch_model_version | select(. != "" and . != null)][0] // ""),
         build_seconds: ([.[].build_seconds] | max), groups: length}' "${ms[@]}" >"$out/manifest.json"
  local m set tb want d f
  for m in "${ms[@]}"; do d="$(dirname "$m")"; for f in "$d"/*.tar.gz; do cp "$f" "$out/"; done; done
  for set in $SETS_ALL; do
    jq -e --arg s "$set" '.sets[$s]' "$out/manifest.json" >/dev/null || die "assemble: set $set missing"
  done
  while IFS=$'\t' read -r tb want; do
    [[ "$(sha256 "$out/$tb")" == "$want" ]] || die "assemble: $tb sha256 differs from its manifest"
  done < <(jq -r '.sets[] | [.tarball, .sha256] | @tsv' "$out/manifest.json")
  # Gate 3: every shipped binary reports the version.
  d="$out/gate"; rm -rf "$d" && mkdir -p "$d"
  for set in gpucheck serve-cpu serve-fake; do mkdir -p "$d/$set" && tar -xzf "$out/$set.tar.gz" -C "$d/$set"; done
  local got bin
  for bin in "$d/serve-cpu/out/fv-serve" "$d/serve-fake/fv-serve" "$d/gpucheck/fv-gpucheck"; do
    got="$("$bin" -V 2>&1 | head -1)" || true
    [[ "$got" == *" $v" || "$got" == *" $v "* ]] || die "${bin#"$d/"} -V says '$got', expected $v"
    log "  ${bin#"$d/"}: $got"
  done
  rm -rf "$d"
  local nt; nt="$(cat "$in"/*/gate-unit-tests.count 2>/dev/null | head -1)"
  GATE_SUMMARY="scripts/serve/check.sh (lint and test stages) passed in parallel GitHub-hosted jobs in the build-base image; the ${nt:-?} shipped gpucheck/cudarc unit-test binaries passed; the binaries report $v; the builds' fv-gpucheck nvrtc gate (AOT + oxide cubins for sm_100/120) and glibc <= 2.35 check passed"
  stage_release "$sha" "$h" "$v" "$tag" "$pre" "$out"
  log "assembled $tag from ${#ms[@]} builds into $out/release"
}

stage_release() {
  local sha="$1" h="$2" v="$3" tag="$4" pre="$5" out="$6" m="$6/manifest.json" rm_json="$6/release/manifest.json" set tb
  mkdir -p "$out/release"
  jq --arg tag "$tag" --arg v "$v" --arg h "$h" --arg schema "$INPUT_SCHEMA" --arg repo "$REPO" \
     --argjson inputs "$(printf '%s\n' "${INPUTS[@]}" | jq -R . | jq -s .)" \
     --argjson pre "$([[ $pre == 1 ]] && echo true || echo false)" \
     --arg gate "$GATE_SUMMARY" --arg by "${GITHUB_WORKFLOW:+github:$GITHUB_REPOSITORY/actions/runs/${GITHUB_RUN_ID:-}}" --arg host "$(hostname)" \
     --arg at "$(date -u +%FT%TZ)" '
    {schema: 2, tag: $tag, version: $v, prerelease: $pre, input_hash: $h, input_schema: $schema, inputs: $inputs,
     source_commit: .sha, repo: $repo, build_id: .build_id, build_time: .build_time,
     tests: $gate, built_by: (if $by != "" then $by else $host end), staged_at: $at} + (del(.schema, .sha, .git_sha))' "$m" >"$rm_json"
  for set in $SETS_ALL; do
    tb="$(jq -r --arg s "$set" '.sets[$s].tarball' "$m")"
    cp "$out/$tb" "$out/release/$tb"
  done
  {
    echo "Tools $v: the build pod's prebuilt binaries, compiled and tested at ${sha}."
    echo
    cmd_notes "$sha"
    echo
    echo "Sets: $(jq -r '.sets | to_entries | map("\(.key) (\(.value.size / 1048576 | floor) MB)") | join(", ")' "$rm_json")"
    echo "Tests: $GATE_SUMMARY"
    echo "Verify: scripts/ci/tools-release.sh fetch $tag <dir> <set>... (docs/dev/tools-releases.md)"
    echo
    echo "input-hash: $h"
    echo "source-commit: $sha"
    echo "manifest-sha256: $(sha256 "$rm_json")"
  } >"$out/release/body.md"
}

# upload <stage dir> [--dispatch]: draft (no tag yet) -> assets -> publish ->
# download and verify every sha256 -> prune -> dispatch the image workflows.
# A failure before publishing deletes the draft. Needs contents:write (and
# actions:write for --dispatch): the workflow job's GITHUB_TOKEN, or the
# coordinator's token file.
cmd_upload() {
  local dir="" dispatch=0
  while (( $# )); do
    case "$1" in
      --dispatch) dispatch=1; shift ;;
      -*) die "upload: unknown flag $1" ;;
      *) dir="$1"; shift ;;
    esac
  done
  [[ -n "$dir" && -f "$dir/manifest.json" && -f "$dir/body.md" ]] || die "usage: tools-release.sh upload <stage dir (manifest.json, body.md, tarballs)> [--dispatch]"
  local m="$dir/manifest.json" tag v sha pre f t0=$SECONDS
  tag="$(jq -r .tag "$m")"; v="$(jq -r .version "$m")"; sha="$(jq -r .source_commit "$m")"; pre="$(jq -r '.prerelease // false' "$m")"
  [[ "$tag" == "$PREFIX$v" && "$v" =~ $SEMVER_RE ]] || die "upload: bad tag/version in $m"
  grep -q "^manifest-sha256: $(sha256 "$m")$" "$dir/body.md" || die "upload: body.md does not match manifest.json"
  # Every tarball the manifest names, with its sha256, before anything is created.
  local tb want
  while IFS=$'\t' read -r tb want; do
    [[ -f "$dir/$tb" && "$(sha256 "$dir/$tb")" == "$want" ]] || die "upload: $tb missing or its sha256 differs from the manifest"
  done < <(jq -r '.sets[] | [.tarball, .sha256] | @tsv' "$m")
  if releases | jq -e --arg t "$tag" 'any(.[]; .tag == $t and (.draft | not))' >/dev/null \
     || api GET "/repos/$REPO/git/ref/tags/$tag" >/dev/null 2>&1; then
    die "tag $tag already exists"
  fi
  local rel id
  rel="$(api POST "/repos/$REPO/releases" -H 'Content-Type: application/json' --data-binary @<(jq -n --arg t "$tag" --arg c "$sha" \
      --rawfile b "$dir/body.md" --argjson pre "$pre" \
      '{tag_name: $t, target_commitish: $c, name: $t, body: $b, draft: true, prerelease: $pre}'))" \
    || die "could not create the draft release (needs contents:write)"
  id="$(jq -r .id <<<"$rel")"
  CLEANUP+=("[[ -f '$dir/.published' ]] || { api DELETE /repos/$REPO/releases/$id >/dev/null && log 'deleted the draft $tag'; }")
  for f in "$dir"/*.tar.gz "$m"; do
    log "upload $(basename "$f") ($(du -h "$f" | cut -f1))"
    api POST "$UPLOADS/repos/$REPO/releases/$id/assets?name=$(basename "$f")" \
      -H "Content-Type: $([[ $f == *.json ]] && echo application/json || echo application/gzip)" \
      --data-binary @"$f" >/dev/null || die "upload of $(basename "$f") failed"
  done
  api PATCH "/repos/$REPO/releases/$id" -H 'Content-Type: application/json' --data-binary '{"draft": false, "make_latest": "false"}' >/dev/null \
    || die "could not publish the release"
  touch "$dir/.published"
  load_releases
  log "published $(releases | jq -r --arg t "$tag" '.[] | select(.tag == $t) | .url')"
  # What consumers will download, checked the way they check it.
  cmd_fetch "$tag" "$dir/verify" $SETS_ALL && rm -rf "$dir/verify"
  log "verified: every set downloads and matches its sha256 ($(( SECONDS - t0 ))s)"
  [[ "$pre" == true ]] || cmd_prune
  (( dispatch )) && dispatch_images "$v"
  return 0
}

# publish <rev> (the coordinator, by hand): plan -> build --pod -> upload.
cmd_publish() {
  local rev="" pre=0 dispatch=0 dry=0 upload=1
  while (( $# )); do
    case "$1" in
      --prerelease) pre=1; shift ;;
      --no-upload) upload=0; shift ;;
      --dispatch) dispatch=1; shift ;;
      --dry-run) dry=1; shift ;;
      -*) die "publish: unknown flag $1" ;;
      *) rev="$1"; shift ;;
    esac
  done
  [[ -n "$rev" ]] || die "usage: tools-release.sh publish <rev> [--prerelease] [--dispatch] [--dry-run|--no-upload]"
  git -c safe.directory="$ROOT" -C "$ROOT" fetch -q origin || true
  local sha; sha="$(rev_sha "$rev")"
  if (( !pre )); then
    git -c safe.directory="$ROOT" -C "$ROOT" merge-base --is-ancestor "$sha" origin/main \
      || die "${sha:0:12} is not on origin/main: stable releases come from main (--prerelease for a branch head)"
  fi
  local pflag=(); (( pre )) && pflag=(--prerelease)
  cmd_plan "$sha" "${pflag[@]}"
  [[ "$(jq -r .skip <<<"$PLAN")" == true ]] && return 0
  (( dry )) && { log "dry run: stop before building"; return 0; }
  local v tag out="${FV_TOOLS_OUT:-$ROOT/artifacts/tools-release/$sha}"
  v="$(jq -r .version <<<"$PLAN")"; tag="$(jq -r .tag <<<"$PLAN")"
  cmd_build "$sha" --pod --version "$v" --tag "$tag" "${pflag[@]}" --out "$out"
  (( upload )) || { log "--no-upload: $tag stays staged in $out/release"; return 0; }
  local dflag=(); (( dispatch )) && dflag=(--dispatch)
  cmd_upload "$out/release" "${dflag[@]}"
}

# The gate: the serve crates' CPU gate (scripts/serve/check.sh, in its own
# target dir, no debuginfo), the shipped gpucheck/cudarc unit-test binaries
# against the commit's sources, and every shipped binary reporting the version.
GATE_SUMMARY=""
run_gate() {
  local sha="$1" v="$2" out="$3" mode="$4" src d where
  if [[ "$mode" == pod ]]; then
    local agent="${FV_TOOLS_TEST_AGENT:-${FV_RELEASE_AGENT:-fv-release}-test}"
    src="${TMPDIR:-/tmp}/fv-tools-gate-${sha:0:12}"
    git -c safe.directory="$ROOT" -C "$ROOT" worktree remove --force "$src" >/dev/null 2>&1 || rm -rf "$src"
    git -c safe.directory="$ROOT" -C "$ROOT" worktree add -q --detach "$src" "$sha" || return 1
    CLEANUP+=("git -C '$ROOT' worktree remove --force '$src' >/dev/null 2>&1")
    log "gate 1/3: scripts/serve/check.sh on the build pod (agent $agent)"
    bash "$src/scripts/dev/build-pod.sh" run "$agent" -- CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 \
      bash scripts/serve/check.sh || { log "check.sh failed"; return 1; }
    where="on the build pod"
  else
    src="$ROOT"
    local tt="${FV_TOOLS_TEST_TARGET:-${CARGO_TARGET_DIR:-$FV_BUILD_TARGET_BASE/gh-runner}-test}"
    log "gate 1/3: scripts/serve/check.sh here (CARGO_TARGET_DIR $tt)"
    (cd "$ROOT" && env -u FV_RELEASE_VERSION CARGO_TARGET_DIR="$tt" CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 \
       bash scripts/serve/check.sh) || { log "check.sh failed"; return 1; }
    where="on the build pod's runner"
  fi

  log "gate 2/3: the shipped gpucheck-tests binaries"
  d="$out/gate"; rm -rf "$d" && mkdir -p "$d"
  local set; for set in gpucheck-tests gpucheck serve-cpu serve-fake; do
    mkdir -p "$d/$set" && tar -xzf "$out/$set.tar.gz" -C "$d/$set"
  done
  FV_TESTS_ROOT="$src" bash "$HERE/prebuilt.sh" run-tests "$d/gpucheck-tests" || { log "unit tests failed"; return 1; }

  log "gate 3/3: binaries report $v"
  local got bin note=""
  for bin in "$d/serve-cpu/out/fv-serve" "$d/serve-fake/fv-serve" "$d/gpucheck/fv-gpucheck"; do
    got="$("$bin" -V 2>&1 | head -1)" || true
    if [[ "$bin" == */fv-gpucheck && "$got" == *"unexpected argument"* ]]; then
      # fv-gpucheck learned -V with the tools releases; older commits lack it.
      note=" (fv-gpucheck at this commit predates -V)"; log "  gpucheck/fv-gpucheck: no -V at this commit"; continue
    fi
    [[ "$got" == *" $v" || "$got" == *" $v "* ]] || { log "${bin#"$d/"} -V says '$got', expected $v"; return 1; }
    log "  ${bin#"$d/"}: $got"
  done
  local ntests; ntests="$(wc -l <"$d/gpucheck-tests/tests.tsv")"
  rm -rf "$d"
  GATE_SUMMARY="scripts/serve/check.sh passed $where; the $ntests shipped gpucheck/cudarc unit-test binaries passed; the binaries report $v$note; the build's fv-gpucheck nvrtc gate (AOT + oxide cubins for sm_100/120) and glibc <= 2.35 check passed"
  return 0
}

# ---- runner choice --------------------------------------------------------------
# pick-runner: `pod` when at least one self-hosted runner with the label
# FV_RUNNER_LABEL (default fv-build) is online and idle, else `github` (the
# build then runs on a GitHub-hosted runner in the build-base image, same
# recipe and gate). Listing runners needs "Administration: read", which a
# workflow's GITHUB_TOKEN cannot have: the token comes from FV_RUNNER_READ_TOKEN
# (a repository secret, fine-grained PAT, Administration: read only); without
# it the choice is `github`. With FV_CONTROL_CI_TOKEN (an fv-control API token
# of scope ci, docs/dev/build-pods-fv-control.md §6) fv-control decides instead:
# `pod` for an idle runner; it wakes a build pod and answers `wait` while that
# comes up (polled for FV_CONTROL_WAIT_S, default 180 s); else `github`. The
# Administration-read PAT is then not needed. FV_BUILD_RUNNER=pod|github (a
# repository variable) forces it. Writes builder=..., reason=... to $GITHUB_OUTPUT; prints JSON.
cmd_pick_runner() {
  local mode="${FV_BUILD_RUNNER:-auto}" label="${FV_RUNNER_LABEL:-fv-build}" builder="" reason="" runners="" n
  case "$mode" in
    pod|github) builder="$mode"; reason="FV_BUILD_RUNNER=$mode" ;;
    auto|"")
      if [[ -n "${FV_CONTROL_CI_TOKEN:-}${FV_CONTROL_CI_FILE:-}" ]]; then
        local ans="" b="" h="" t0=$SECONDS n=0
        if [[ -z "${FV_CONTROL_CI_FILE:-}" ]]; then h="$(umask 077 && mktemp)"; printf 'Authorization: Bearer %s\n' "$FV_CONTROL_CI_TOKEN" >"$h"; fi
        while :; do
          n=$((n + 1))
          if [[ -n "${FV_CONTROL_CI_FILE:-}" ]]; then   # tests: line n is the n-th answer (the last repeats)
            ans="$(sed -n "${n}p" "$FV_CONTROL_CI_FILE")"; [[ -n "$ans" ]] || ans="$(tail -n1 "$FV_CONTROL_CI_FILE")"
          else
            ans="$(curl -sS --fail-with-body --max-time 120 -H @"$h" -H 'content-type: application/json' -X POST \
              "${FV_CONTROL_URL:-https://fv-control-staging.maximalize.workers.dev}/api/ci/build-runner" \
              -d "$(jq -nc --arg l "$label" --arg w "${GITHUB_WORKFLOW:-}" --arg r "${GITHUB_RUN_ID:-}" '{label: $l, workflow: $w, run_id: $r}')" 2>/dev/null)" || ans=""
          fi
          b="$(jq -r '.builder // empty' <<<"$ans" 2>/dev/null || true)"
          if [[ "$b" == wait ]] && (( SECONDS - t0 < ${FV_CONTROL_WAIT_S:-180} )); then
            log "fv-control: $(jq -r .reason <<<"$ans")"
            sleep "$(jq -r '.retry_after_s // 15' <<<"$ans")"
            continue
          fi
          break
        done
        [[ -n "$h" ]] && rm -f "$h"
        if [[ "$b" == pod ]]; then builder=pod; else builder=github; fi
        reason="fv-control: $(jq -r '.reason // empty' <<<"$ans" 2>/dev/null || true)"
        [[ "$b" == wait ]] && reason="fv-control: the woken pod's runner was not online within ${FV_CONTROL_WAIT_S:-180}s"
        [[ -n "$ans" ]] || reason="fv-control did not answer"
      elif [[ -n "${FV_TOOLS_RUNNERS_FILE:-}" ]]; then
        runners="$(cat "$FV_TOOLS_RUNNERS_FILE")"   # tests
      elif [[ -n "${FV_RUNNER_READ_TOKEN:-}" ]]; then
        local h; h="$(umask 077 && mktemp)"
        printf 'Authorization: Bearer %s\nAccept: application/vnd.github+json\nX-GitHub-Api-Version: 2022-11-28\n' "$FV_RUNNER_READ_TOKEN" >"$h"
        runners="$(curl -sS --fail-with-body -H @"$h" "$API/repos/$REPO/actions/runners?per_page=100" 2>/dev/null)" || runners=""
        rm -f "$h"
        [[ -n "$runners" ]] || reason="could not list the runners with FV_RUNNER_READ_TOKEN"
      else
        reason="no FV_RUNNER_READ_TOKEN secret: cannot see the build pods' runners"
      fi
      if [[ -n "$runners" ]]; then
        n="$(jq --arg l "$label" '[.runners[]? | select(.status == "online" and (.busy | not) and any(.labels[]?; .name == $l))] | length' <<<"$runners")"
        if (( n > 0 )); then builder=pod; reason="$n idle $label runner(s) online"
        else builder=github; reason="no idle $label runner online"
        fi
      elif [[ -z "$builder" ]]; then
        builder=github
      fi ;;
    *) die "FV_BUILD_RUNNER must be auto, pod or github (got $mode)" ;;
  esac
  [[ -n "${GITHUB_OUTPUT:-}" ]] && printf 'builder=%s\nreason=%s\n' "$builder" "$reason" >>"$GITHUB_OUTPUT"
  log "builder: $builder ($reason)"
  jq -nc --arg b "$builder" --arg r "$reason" '{builder: $b, reason: $r}'
}

# ---- prune ------------------------------------------------------------------
# Keep the newest N stable tools releases (and the pinned FV_TOOLS_VERSION:
# the repository variable, or FV_TOOLS_PIN when the token cannot read it),
# delete older ones and prereleases that are older than 14 days or below the
# highest stable version, with their tags. Only tags starting tools-v.
cmd_prune() {
  local keep="${FV_TOOLS_KEEP:-10}" dry=0
  while (( $# )); do
    case "$1" in
      --keep) keep="$2"; shift 2 ;;
      --dry-run) dry=1; shift ;;
      *) die "prune: unknown argument $1" ;;
    esac
  done
  local pin cutoff
  pin="${FV_TOOLS_PIN-$( [[ -n "${FV_TOOLS_RELEASES_FILE:-}" ]] || api GET "/repos/$REPO/actions/variables/FV_TOOLS_VERSION" 2>/dev/null | jq -r '.value // empty' || true)}"
  cutoff="$(date -u -d '14 days ago' +%FT%TZ)"
  local victims
  victims="$(releases | jq -r --argjson keep "$keep" --arg pin "${pin#v}" --arg cut "$cutoff" '
    ([.[] | select((.draft or .prerelease) | not)]) as $st
    | ($st[0].version // "") as $top
    | ($st[$keep:] | map(select(.version != $pin))) as $old
    | ([.[] | select(.prerelease and (.draft | not) and (.created < $cut or ((.version | split("-")[0]) as $b | $b == $top or ($st | any(.version == $b)))))]) as $pre
    | ($old + $pre)[] | [.id, .tag] | @tsv')"
  [[ -n "$victims" ]] || { log "prune: nothing to delete (keep $keep)"; return 0; }
  local id tag
  while IFS=$'\t' read -r id tag; do
    [[ "$tag" == "$PREFIX"* ]] || die "prune: refusing $tag"
    if (( dry )); then log "prune (dry run): $tag"; continue; fi
    api DELETE "/repos/$REPO/releases/$id" >/dev/null && log "prune: deleted release $tag"
    api DELETE "/repos/$REPO/git/refs/tags/$tag" >/dev/null 2>&1 && log "prune: deleted tag $tag" || true
  done <<<"$victims"
  load_releases
}

# Rebuild the images on main with the new release (needs actions:write). A
# release created with a workflow's GITHUB_TOKEN fires no `release` event for
# other workflows, so every publisher dispatches explicitly; vast-pytorch-image
# (always the newest release) gets the version as its tools_version input.
dispatch_images() {
  local v="$1" wf body
  for wf in ${FV_TOOLS_DISPATCH:-serve-image.yml gpucheck-runtime-image.yml vast-pytorch-image.yml}; do
    body='{"ref": "main"}'
    [[ "$wf" == vast-pytorch-image.yml ]] && body="$(jq -nc --arg v "$v" '{ref: "main", inputs: {tools_version: $v}}')"
    if api POST "/repos/$REPO/actions/workflows/$wf/dispatches" -H 'Content-Type: application/json' --data-binary "$body" >/dev/null; then
      log "dispatched $wf on main"
    else
      log "WARNING: could not dispatch $wf (token needs actions:write); run it by hand"
    fi
  done
}

case "${1:-}" in
  publish|upload|prune) export FV_TOOLS_WRITE=1 ;;
esac
case "${1:-}" in
  status|list|resolve|fetch|notes|plan|build|assemble|publish|upload|prune) auth; load_releases ;;
esac
case "${1:-}" in
  input-hash) shift; input_hash "$@" ;;
  version) shift; tools_version "$@" ;;
  status) shift; cmd_status "$@" ;;
  list) cmd_list ;;
  resolve) shift; cmd_resolve "$@" ;;
  fetch) shift; cmd_fetch "$@" ;;
  bump) shift; cmd_bump "$@" ;;
  notes) shift; cmd_notes "$@" ;;
  plan) shift; cmd_plan "$@"; echo "$PLAN" ;;
  pick-runner) cmd_pick_runner ;;
  build) shift; cmd_build "$@" ;;
  build-sets) shift; cmd_build_sets "$@" ;;
  assemble) shift; cmd_assemble "$@" ;;
  upload) shift; cmd_upload "$@" ;;
  publish) shift; cmd_publish "$@" ;;
  prune) shift; cmd_prune "$@" ;;
  *) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 2 ;;
esac
