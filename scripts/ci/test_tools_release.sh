#!/usr/bin/env bash
# Offline checks of scripts/ci/tools-release.sh's selection rules (no network:
# FV_TOOLS_RELEASES_FILE stands in for the release list). bash scripts/ci/test_tools_release.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T="$HERE/tools-release.sh"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
fail=0
check() { # <name> <expected> <actual>
  if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: expected '$2', got '$3'"; fail=1; fi
}
rel() { # tag id draft prerelease created input-hash
  jq -n --arg t "$1" --argjson id "$2" --argjson d "$3" --argjson p "$4" --arg c "$5" --arg h "$6" \
    '{tag_name: $t, id: $id, draft: $d, prerelease: $p, html_url: "u", created_at: $c,
      body: "notes\ninput-hash: \($h)\nsource-commit: c\($id)\nmanifest-sha256: m", assets: []}'
}
{
  rel tools-v0.1.10 10 false false 2026-10-01T00:00:00Z aaa10
  rel tools-v0.1.9 9 false false 2026-09-01T00:00:00Z aaa9
  rel tools-v0.1.11-pre.abc 12 false true 2099-01-01T00:00:00Z bbb
  rel tools-v0.1.10-pre.old 13 false true 2099-01-01T00:00:00Z ccc
  rel tools-v0.2.0 20 true false 2026-10-06T00:00:00Z ddd
  rel v1.0.0 1 false false 2026-01-01T00:00:00Z eee
  rel tools-v0.1.2 2 false false 2026-08-01T00:00:00Z aaa2
} | jq -s . >"$tmp/rels.json"
export FV_TOOLS_RELEASES_FILE="$tmp/rels.json" GH_TOKEN="" GITHUB_TOKEN=""

check "highest SemVer (0.1.10 > 0.1.9, drafts and other tags ignored)" tools-v0.1.10 "$(bash "$T" resolve | jq -r .tag)"
check "exact input hash" "tools-v0.1.9 true" "$(bash "$T" resolve --input-hash aaa9 | jq -r '"\(.tag) \(.exact)"')"
check "exact match may be a prerelease" tools-v0.1.11-pre.abc "$(bash "$T" resolve --input-hash bbb | jq -r .tag)"
check "no exact match: newest, not exact" "tools-v0.1.10 false" "$(bash "$T" resolve --input-hash zzz | jq -r '"\(.tag) \(.exact)"')"
check "--exact without a match fails" 1 "$(bash "$T" resolve --input-hash zzz --exact 2>/dev/null >/dev/null; echo $?)"
check "--newest ignores an older exact match" "tools-v0.1.10 false" "$(bash "$T" resolve --input-hash aaa9 --newest | jq -r '"\(.tag) \(.exact)"')"
check "--newest on the newest's inputs is exact" "tools-v0.1.10 true" "$(bash "$T" resolve --input-hash aaa10 --newest | jq -r '"\(.tag) \(.exact)"')"
echo '[]' >"$tmp/none.json"
check "vast mode (newest + require) without a release fails the job" 1 "$(
  FV_TOOLS_RELEASES_FILE="$tmp/none.json" FV_PREBUILT_SELECT=newest FV_PREBUILT_REQUIRE=1 GITHUB_OUTPUT="$tmp/out" \
    bash "$HERE/prebuilt.sh" fetch gpucheck-vast hf-fm >"$tmp/req.log" 2>&1; echo $?)"
grep -q "::error title=Prebuilt tools::no usable tools release" "$tmp/req.log" && echo "ok   (clear ::error)" || { echo "FAIL require message"; cat "$tmp/req.log"; fail=1; }
check "pin" tools-v0.1.2 "$(bash "$T" resolve --version v0.1.2 | jq -r .tag)"
check "list order" "0.2.0 0.1.11-pre.abc 0.1.10 0.1.10-pre.old 0.1.9 0.1.2" "$(bash "$T" list | cut -f1 | paste -sd' ')"
check "prune keeps N stable, drops released prereleases, never other tags" \
  "tools-v0.1.9 tools-v0.1.2 tools-v0.1.10-pre.old" \
  "$(FV_TOOLS_KEEP=1 FV_GITHUB_TOKEN_FILE=/dev/null bash "$T" prune --dry-run 2>&1 | sed -n 's/.*prune (dry run): //p' | paste -sd' ')"
# Version rule: the workspace version when above the highest release, else its PATCH + 1.
w="$(bash "$T" version HEAD)"
plan() { bash "$T" plan HEAD "$@" 2>/dev/null; }
check "version: highest release 0.1.10 >= workspace $w -> 0.1.11" "tools-v0.1.11 false" "$(plan | jq -r '"\(.tag) \(.skip)"')"
echo '[]' >"$tmp/empty.json"
check "version: no release yet -> the workspace version" "tools-v$w" "$(FV_TOOLS_RELEASES_FILE="$tmp/empty.json" plan | jq -r .tag)"
jq '[.[] | select(.tag_name == "v1.0.0")] + [{tag_name: "tools-v0.0.9", id: 3, draft: false, prerelease: false, html_url: "u",
     created_at: "2026-01-01T00:00:00Z", body: "input-hash: zzz", assets: []}]' "$tmp/rels.json" >"$tmp/low.json"
check "version: workspace above the highest release (a manual bump) -> it" "tools-v$w" "$(FV_TOOLS_RELEASES_FILE="$tmp/low.json" plan | jq -r .tag)"
check "version: drafts and prereleases do not count (0.2.0 draft, 0.1.11-pre)" "0.1.10" "$(plan | jq -r .latest)"
check "prerelease tag" "tools-v0.1.11-pre.$(git -C "$HERE/../.." rev-parse HEAD | cut -c1-12)" "$(plan --prerelease | jq -r .tag)"
check "plan writes GITHUB_OUTPUT" "version=0.1.11" "$(GITHUB_OUTPUT="$tmp/gho" plan >/dev/null; grep '^version=' "$tmp/gho")"
check "skip-if-unchanged: the inputs of a release" "true tools-v0.1.10" "$(
  h="$(bash "$T" input-hash HEAD)"
  jq --arg h "$h" '.[0].body |= sub("input-hash: aaa10"; "input-hash: \($h)")' "$tmp/rels.json" >"$tmp/same0.json"
  FV_TOOLS_RELEASES_FILE="$tmp/same0.json" plan | jq -r '"\(.skip) \(.skip_reason | capture("of (?<t>tools-v[^ ]+)").t)"')"
# upload refuses a stage whose tarball no longer matches its manifest (before any API write).
mkdir -p "$tmp/stage" && echo data >"$tmp/stage/oxide.tar.gz"
jq -n '{tag: "tools-v0.1.11", version: "0.1.11", source_commit: "abc", sets: {oxide: {tarball: "oxide.tar.gz", sha256: "0000"}}}' >"$tmp/stage/manifest.json"
printf 'notes\nmanifest-sha256: %s\n' "$(sha256sum "$tmp/stage/manifest.json" | cut -d' ' -f1)" >"$tmp/stage/body.md"
check "upload refuses a tampered tarball" 2 "$(FV_GITHUB_TOKEN_FILE=/dev/null bash "$T" upload "$tmp/stage" >"$tmp/up.log" 2>&1; echo $?)"
grep -q "oxide.tar.gz missing or its sha256 differs" "$tmp/up.log" && echo "ok   (said why)" || { echo "FAIL upload message"; cat "$tmp/up.log"; fail=1; }
# Builder choice: an online, idle fv-build runner -> pod; else GitHub-hosted.
runner() { # name status busy labels...
  local n="$1" st="$2" b="$3"; shift 3
  jq -n --arg n "$n" --arg s "$st" --argjson b "$b" --args '{name: $n, status: $s, busy: $b, labels: ($ARGS.positional | map({name: .}))}' "$@"
}
pick() { FV_TOOLS_RUNNERS_FILE="$1" bash "$T" pick-runner 2>/dev/null | jq -r .builder; }
{ runner p1 online false self-hosted fv-build; runner p2 online true self-hosted fv-build; } | jq -s '{runners: .}' >"$tmp/r-idle.json"
{ runner p2 online true self-hosted fv-build; runner p3 offline false self-hosted fv-build; runner x online false self-hosted other; } | jq -s '{runners: .}' >"$tmp/r-none.json"
check "runner: an idle online fv-build runner -> pod" pod "$(pick "$tmp/r-idle.json")"
check "runner: busy, offline or other labels only -> github" github "$(pick "$tmp/r-none.json")"
check "runner: no token to list runners -> github" github "$(env -u FV_TOOLS_RUNNERS_FILE FV_RUNNER_READ_TOKEN= bash "$T" pick-runner 2>/dev/null | jq -r .builder)"
check "runner: FV_BUILD_RUNNER=pod forces the pod" pod "$(FV_BUILD_RUNNER=pod pick "$tmp/r-none.json")"
check "runner: FV_BUILD_RUNNER=github forces GitHub" github "$(FV_BUILD_RUNNER=github pick "$tmp/r-idle.json")"
check "runner: writes builder to GITHUB_OUTPUT" "builder=pod" "$(GITHUB_OUTPUT="$tmp/gho2" pick "$tmp/r-idle.json" >/dev/null; grep '^builder=' "$tmp/gho2")"
# assemble: per-group builds (the hosted fallback) joined into one release.
mkgroup() { # dir sets... : a fake build of those sets for HEAD; binaries print the version
  local dir="$1" set; shift; mkdir -p "$dir"
  local sets='{}'
  for set in "$@"; do
    local st="$tmp/st-$set"; rm -rf "$st"; mkdir -p "$st"
    case "$set" in
      serve-cpu) mkdir -p "$st/out"; printf '#!/bin/sh\necho "fv-serve 0.1.11"\n' >"$st/out/fv-serve"; chmod +x "$st/out/fv-serve" ;;
      serve-fake) printf '#!/bin/sh\necho "fv-serve 0.1.11"\n' >"$st/fv-serve"; chmod +x "$st/fv-serve" ;;
      gpucheck) printf '#!/bin/sh\necho "fv-gpucheck 0.1.11"\n' >"$st/fv-gpucheck"; chmod +x "$st/fv-gpucheck" ;;
      *) echo "$set" >"$st/$set.txt" ;;
    esac
    tar -C "$st" -czf "$dir/$set.tar.gz" .
    sets="$(jq --arg s "$set" --arg h "$(sha256sum "$dir/$set.tar.gz" | cut -d' ' -f1)" '. + {($s): {tarball: "\($s).tar.gz", sha256: $h, size: 1}}' <<<"$sets")"
  done
  jq -n --arg sha "$(git -C "$HERE/../.." rev-parse HEAD)" --argjson sets "$sets" \
    '{schema: 1, sha: $sha, build_id: "b", build_time: "", oxide_key: "", hf_fetch_model_version: "", build_seconds: 1, sets: $sets}' >"$dir/manifest.json"
}
rm -rf "$tmp/in" && mkdir -p "$tmp/in"
mkgroup "$tmp/in/a" oxide serve-cuda serve-cpu
mkgroup "$tmp/in/b" serve-fake gpucheck gpucheck-tests
mkgroup "$tmp/in/c" gpucheck-vast hf-fm
echo 2 >"$tmp/in/b/gate-unit-tests.count"
check "assemble joins the groups into one staged release" "8 2" "$(
  bash "$T" assemble HEAD --version 0.1.11 --tag tools-v0.1.11 --in "$tmp/in" --out "$tmp/asm" >/dev/null 2>"$tmp/asm.log"
  jq -r '"\(.sets | length) \(.tests | capture("the (?<n>[0-9]+) shipped").n)"' "$tmp/asm/release/manifest.json" 2>/dev/null)"
grep -q "^manifest-sha256: $(sha256sum "$tmp/asm/release/manifest.json" 2>/dev/null | cut -d' ' -f1)$" "$tmp/asm/release/body.md" 2>/dev/null \
  && echo "ok   (body carries the manifest sha256)" || { echo "FAIL assemble body"; tail -5 "$tmp/asm.log"; fail=1; }
check "assemble refuses a wrong -V" 2 "$(bash "$T" assemble HEAD --version 0.1.12 --tag tools-v0.1.12 --in "$tmp/in" --out "$tmp/asm4" >/dev/null 2>"$tmp/asm4.log"; echo $?)"
grep -q "expected 0.1.12" "$tmp/asm4.log" && echo "ok   (because of -V)" || { echo "FAIL -V reason"; tail -3 "$tmp/asm4.log"; fail=1; }
mkgroup "$tmp/in/d" hf-fm
check "assemble refuses a set built twice" 2 "$(bash "$T" assemble HEAD --version 0.1.11 --tag tools-v0.1.11 --in "$tmp/in" --out "$tmp/asm2" >/dev/null 2>&1; echo $?)"
rm -rf "$tmp/in/d"; rm "$tmp/in/c/hf-fm.tar.gz"
check "assemble refuses a tarball that differs from its manifest" 2 "$(bash "$T" assemble HEAD --version 0.1.11 --tag tools-v0.1.11 --in "$tmp/in" --out "$tmp/asm3" >/dev/null 2>&1; echo $?)"
check "unchanged inputs: nothing to publish" 0 "$(
  h="$(bash "$T" input-hash HEAD)"
  jq --arg h "$h" '.[0].body |= sub("input-hash: aaa10"; "input-hash: \($h)")' "$tmp/rels.json" >"$tmp/same.json"
  FV_TOOLS_RELEASES_FILE="$tmp/same.json" FV_GITHUB_TOKEN_FILE=/dev/null bash "$T" publish HEAD --prerelease --dry-run 2>"$tmp/log" >/dev/null; echo $?)"
grep -q "nothing to publish" "$tmp/log" && echo "ok   (said nothing to publish)" || { echo "FAIL unchanged message"; fail=1; }
exit "$fail"
