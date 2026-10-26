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
printf '%s\n' '{"builder":"pod","reason":"1 idle fv-build runner(s) online"}' >"$tmp/fvc-pod"
printf '%s\n' '{"builder":"wait","reason":"created build pod x","retry_after_s":0}' '{"builder":"wait","reason":"x","retry_after_s":0}' '{"builder":"pod","reason":"1 idle"}' >"$tmp/fvc-wait"
printf '%s\n' '{"builder":"github","reason":"no idle runner and no pod: disabled"}' >"$tmp/fvc-gh"
printf '%s\n' '{"builder":"wait","reason":"x","retry_after_s":0}' >"$tmp/fvc-forever"
fvc() { FV_CONTROL_CI_FILE="$1" bash "$T" pick-runner 2>/dev/null | jq -r .builder; }
check "runner: fv-control says pod" pod "$(fvc "$tmp/fvc-pod")"
check "runner: fv-control wakes a pod, then pod" pod "$(fvc "$tmp/fvc-wait")"
check "runner: fv-control says github" github "$(fvc "$tmp/fvc-gh")"
check "runner: fv-control still waiting past FV_CONTROL_WAIT_S -> github" github "$(FV_CONTROL_WAIT_S=0 fvc "$tmp/fvc-forever")"
check "runner: writes builder to GITHUB_OUTPUT" "builder=pod" "$(GITHUB_OUTPUT="$tmp/gho2" pick "$tmp/r-idle.json" >/dev/null; grep '^builder=' "$tmp/gho2")"
# assemble: per-group builds (the hosted fallback) joined into one release.
mkgroup() { # dir sets... : a fake build of those sets for HEAD; binaries print the version
  local dir="$1" set; shift; mkdir -p "$dir"
  local sets='{}'
  for set in "$@"; do
    local st="$tmp/st-$set"; rm -rf "$st"; mkdir -p "$st"
    case "$set" in
      serve-cpu) mkdir -p "$st/out"; printf '#!/bin/sh\necho "fv-serve 0.1.11"\n' >"$st/out/fv-serve"; chmod +x "$st/out/fv-serve"
        # FAKE_BOOT_VERSION: the set also carries fv-boot (ADR-0014), whose `version` prints that.
        if [[ -n "${FAKE_BOOT_VERSION:-}" ]]; then printf '#!/bin/sh\n[ "$1" = version ] && echo "fv-boot %s"\n' "$FAKE_BOOT_VERSION" >"$st/out/fv-boot"; chmod +x "$st/out/fv-boot"; fi ;;
      serve-fake) printf '#!/bin/sh\necho "fv-serve 0.1.11"\n' >"$st/fv-serve"; chmod +x "$st/fv-serve" ;;
      gpucheck) printf '#!/bin/sh\necho "fv-gpucheck 0.1.11"\n' >"$st/fv-gpucheck"; chmod +x "$st/fv-gpucheck"
        # FAKE_LAB_VERSION: the set also carries fv-lab (ADR-0019), whose -V prints that.
        if [[ -n "${FAKE_LAB_VERSION:-}" ]]; then mkdir -p "$st/out"; printf '#!/bin/sh\necho "fv-lab %s"\n' "$FAKE_LAB_VERSION" >"$st/out/fv-lab"; chmod +x "$st/out/fv-lab"; fi ;;
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
# fv-boot in the serve set: its `version` must be the release's.
rm -rf "$tmp/inb" && cp -a "$tmp/in" "$tmp/inb" && FAKE_BOOT_VERSION=0.1.11 mkgroup "$tmp/inb/a" oxide serve-cuda serve-cpu
check "assemble checks fv-boot's version" 0 "$(bash "$T" assemble HEAD --version 0.1.11 --tag tools-v0.1.11 --in "$tmp/inb" --out "$tmp/asm5" >/dev/null 2>"$tmp/asm5.log"; echo $?)"
grep -q "serve-cpu/out/fv-boot: fv-boot 0.1.11" "$tmp/asm5.log" && echo "ok   (fv-boot 0.1.11 checked)" || { echo "FAIL fv-boot not checked"; tail -3 "$tmp/asm5.log"; fail=1; }
FAKE_BOOT_VERSION=0.1.1 mkgroup "$tmp/inb/a" oxide serve-cuda serve-cpu
check "assemble refuses an fv-boot with another version" 2 "$(bash "$T" assemble HEAD --version 0.1.11 --tag tools-v0.1.11 --in "$tmp/inb" --out "$tmp/asm6" >/dev/null 2>"$tmp/asm6.log"; echo $?)"
grep -q "fv-boot version says 'fv-boot 0.1.1', expected fv-boot 0.1.11" "$tmp/asm6.log" && echo "ok   (because of fv-boot version)" || { echo "FAIL fv-boot reason"; tail -3 "$tmp/asm6.log"; fail=1; }
# fv-lab in the gpucheck set: its -V must be the release's.
rm -rf "$tmp/inl" && cp -a "$tmp/in" "$tmp/inl" && FAKE_LAB_VERSION=0.1.11 mkgroup "$tmp/inl/b" serve-fake gpucheck gpucheck-tests
check "assemble checks fv-lab's version" 0 "$(bash "$T" assemble HEAD --version 0.1.11 --tag tools-v0.1.11 --in "$tmp/inl" --out "$tmp/asm7" >/dev/null 2>"$tmp/asm7.log"; echo $?)"
grep -q "gpucheck/out/fv-lab: fv-lab 0.1.11" "$tmp/asm7.log" && echo "ok   (fv-lab 0.1.11 checked)" || { echo "FAIL fv-lab not checked"; tail -3 "$tmp/asm7.log"; fail=1; }
FAKE_LAB_VERSION=0.1.1 mkgroup "$tmp/inl/b" serve-fake gpucheck gpucheck-tests
check "assemble refuses an fv-lab with another version" 2 "$(bash "$T" assemble HEAD --version 0.1.11 --tag tools-v0.1.11 --in "$tmp/inl" --out "$tmp/asm8" >/dev/null 2>"$tmp/asm8.log"; echo $?)"
grep -q "gpucheck/out/fv-lab -V says 'fv-lab 0.1.1', expected fv-lab 0.1.11" "$tmp/asm8.log" && echo "ok   (because of fv-lab version)" || { echo "FAIL fv-lab reason"; tail -3 "$tmp/asm8.log"; fail=1; }
mkgroup "$tmp/in/d" hf-fm
check "assemble refuses a set built twice" 2 "$(bash "$T" assemble HEAD --version 0.1.11 --tag tools-v0.1.11 --in "$tmp/in" --out "$tmp/asm2" >/dev/null 2>&1; echo $?)"
rm -rf "$tmp/in/d"; rm "$tmp/in/c/hf-fm.tar.gz"
check "assemble refuses a tarball that differs from its manifest" 2 "$(bash "$T" assemble HEAD --version 0.1.11 --tag tools-v0.1.11 --in "$tmp/in" --out "$tmp/asm3" >/dev/null 2>&1; echo $?)"
# run-tests shards: every test exactly once over the shards.
mkdir -p "$tmp/ut/bin"
cat >"$tmp/ut/bin/t1" <<'EOF'
#!/bin/sh
if [ "$1" = --list ]; then for t in a b c d e; do echo "$t: test"; done; echo "bench_x: bench"; exit 0; fi
[ "$1" = --exact ] && shift
for t in "$@"; do echo "ran $t" >>"$RAN"; done
EOF
chmod +x "$tmp/ut/bin/t1"
printf 'bin/t1\tpkg\t.\tlib\tt1\n' >"$tmp/ut/tests.tsv"; git -C "$HERE/../.." rev-parse --show-toplevel >"$tmp/ut/src-root"
: >"$tmp/ran"
for k in 1 2 3; do RAN="$tmp/ran" FV_TESTS_SHARD="$k/3" bash "$HERE/prebuilt.sh" run-tests "$tmp/ut" >/dev/null 2>&1; done
check "run-tests shards cover every test once" "a b c d e" "$(sed 's/ran //' "$tmp/ran" | sort | paste -sd' ')"
check "unchanged inputs: nothing to publish" 0 "$(
  h="$(bash "$T" input-hash HEAD)"
  jq --arg h "$h" '.[0].body |= sub("input-hash: aaa10"; "input-hash: \($h)")' "$tmp/rels.json" >"$tmp/same.json"
  FV_TOOLS_RELEASES_FILE="$tmp/same.json" FV_GITHUB_TOKEN_FILE=/dev/null bash "$T" publish HEAD --prerelease --dry-run 2>"$tmp/log" >/dev/null; echo $?)"
grep -q "nothing to publish" "$tmp/log" && echo "ok   (said nothing to publish)" || { echo "FAIL unchanged message"; fail=1; }
# prebuilt.sh contexts: fv-boot (ADR-0014) replaces the image's boot-build stage only when a serve set carries it.
P="$HERE/prebuilt.sh"
mkdir -p "$tmp/pre/serve-cuda/out" "$tmp/pre/serve-cpu/out" "$tmp/pre/oxide"
check "contexts: a release from before fv-boot (boot-build compiles)" "oxide=$tmp/pre/oxide serve-build=$tmp/pre/serve-cuda cpu-build=$tmp/pre/serve-cpu" \
  "$(bash "$P" contexts "$tmp/pre" oxide serve-cuda serve-cpu | paste -sd' ')"
printf '#!/bin/sh\necho fv-boot 0.1.11\n' >"$tmp/pre/serve-cpu/out/fv-boot"; chmod +x "$tmp/pre/serve-cpu/out/fv-boot"
check "contexts: a release with fv-boot replaces boot-build" "oxide=$tmp/pre/oxide serve-build=$tmp/pre/serve-cuda cpu-build=$tmp/pre/serve-cpu boot-build=$tmp/pre/serve-cpu" \
  "$(bash "$P" contexts "$tmp/pre" oxide serve-cuda serve-cpu | paste -sd' ')"
check "contexts: no serve set, no boot-build" "oxide=$tmp/pre/oxide" "$(bash "$P" contexts "$tmp/pre" oxide | paste -sd' ')"
# A release older than HEAD's tools inputs (main's FV_PREBUILT_STALE=latest, a pin): its fv-boot is not used, so
# the image's entrypoint is always this commit's (serve-image run 286: a pre-ADR-0011 fv-boot in the one image).
check "contexts --stale: an older release's fv-boot is not used (boot-build compiles)" "oxide=$tmp/pre/oxide serve-build=$tmp/pre/serve-cuda cpu-build=$tmp/pre/serve-cpu" \
  "$(bash "$P" contexts --stale "$tmp/pre" oxide serve-cuda serve-cpu | paste -sd' ')"
# fv-lab (ADR-0019) replaces the runtime image's lab-build stage only when the gpucheck set carries it.
mkdir -p "$tmp/pre/gpucheck/out"
check "contexts: a release from before fv-lab (lab-build compiles)" "oxide=$tmp/pre/oxide binary=$tmp/pre/gpucheck" \
  "$(bash "$P" contexts "$tmp/pre" oxide gpucheck | paste -sd' ')"
printf '#!/bin/sh\necho fv-lab 0.1.11\n' >"$tmp/pre/gpucheck/out/fv-lab"; chmod +x "$tmp/pre/gpucheck/out/fv-lab"
check "contexts: a release with fv-lab replaces lab-build" "oxide=$tmp/pre/oxide binary=$tmp/pre/gpucheck lab-build=$tmp/pre/gpucheck" \
  "$(bash "$P" contexts "$tmp/pre" oxide gpucheck | paste -sd' ')"
check "contexts: no gpucheck set, no lab-build" "oxide=$tmp/pre/oxide" "$(bash "$P" contexts "$tmp/pre" oxide | paste -sd' ')"
# A stale release's fv-lab is not used (ADR-0019 §20): the runtime image ships this commit's lab/, which an older
# fv-lab may refuse (deny_unknown_fields) in the smoke step's `fv-lab check`, after the sha-<7> tag is pushed.
check "contexts --stale: an older release's fv-lab is not used (lab-build compiles)" "oxide=$tmp/pre/oxide binary=$tmp/pre/gpucheck" \
  "$(bash "$P" contexts --stale "$tmp/pre" oxide gpucheck | paste -sd' ')"
# prebuilt.sh fetch end to end (offline: FV_TOOLS_RELEASES_FILE lists the release, FV_TOOLS_ASSETS_DIR serves its
# assets): which fv-boot the images get. Exact or pinned: the release's; stale (main's FV_PREBUILT_STALE=latest): this
# checkout's (no boot-build=), and the notice says why; a release without fv-boot says that instead.
fakerel() { # <dir> <tag> <input-hash> <with fv-boot: 1|0> -> writes <dir>/releases.json and <dir>/assets/
  local d="$1" tag="$2" ih="$3" boot="$4" set st files='{}' sets='{}' tb
  rm -rf "$d" && mkdir -p "$d/assets"
  local n=2 assets='[]'
  for set in serve-cuda serve-cpu; do
    st="$d/st-$set"; mkdir -p "$st/out"
    printf '#!/bin/sh\necho fv-serve\n' >"$st/out/fv-serve"; echo cuda >"$st/out/fv-serve.features"
    [[ "$boot" == 1 ]] && printf '#!/bin/sh\necho fv-boot\n' >"$st/out/fv-boot"
    chmod +x "$st/out/"*
    files="$(cd "$st" && for f in $(find . -type f -printf '%P\n' | LC_ALL=C sort); do jq -n --arg f "$f" --arg h "$(sha256sum "$f" | cut -d' ' -f1)" '{($f): {sha256: $h}}'; done | jq -s 'add')"
    tb="$set.tar.gz"; tar -C "$st" -czf "$d/assets/$n" .
    sets="$(jq --arg s "$set" --arg tb "$tb" --arg h "$(sha256sum "$d/assets/$n" | cut -d' ' -f1)" --argjson f "$files" \
      '. + {($s): {tarball: $tb, sha256: $h, files: $f}}' <<<"$sets")"
    assets="$(jq --arg n "$tb" --argjson id "$n" '. + [{name: $n, id: $id, size: 1, digest: ""}]' <<<"$assets")"
    n=$((n + 1))
  done
  jq -n --arg t "$tag" --argjson s "$sets" '{tag: $t, version: ($t | ltrimstr("tools-v")), build_id: "b", source_commit: "c", sets: $s}' >"$d/assets/1"
  jq -n --arg t "$tag" --arg ih "$ih" --arg ms "$(sha256sum "$d/assets/1" | cut -d' ' -f1)" --argjson a "$assets" \
    '[{tag_name: $t, id: 1, draft: false, prerelease: false, html_url: "u", created_at: "2026-10-01T00:00:00Z",
       body: "notes\ninput-hash: \($ih)\nsource-commit: c1\nmanifest-sha256: \($ms)",
       assets: ([{name: "manifest.json", id: 1, size: 1, digest: ""}] + $a)}]' >"$d/releases.json"
}
pfetch() { # <dir> [env…] -> the GITHUB_OUTPUT (contexts joined on one line) and the log in <dir>/fetch.log
  local d="$1"; shift
  : >"$d/gho"
  env FV_TOOLS_RELEASES_FILE="$d/releases.json" FV_TOOLS_ASSETS_DIR="$d/assets" FV_PREBUILT_DIR="$d/pre" GITHUB_OUTPUT="$d/gho" \
    FV_TOOLS_VERSION= FV_PREBUILT_SELECT= FV_PREBUILT_FEATURES= FV_PREBUILT_DISABLE= "$@" \
    bash "$P" fetch serve-cuda serve-cpu >"$d/fetch.log" 2>&1
  sed -n '/^contexts<<FV_EOF$/,/^FV_EOF$/{/FV_EOF/d; s/^/ctx:/; p}' "$d/gho" | paste -sd' '
  grep -E '^(ok|exact|fv_boot)=' "$d/gho" | paste -sd' '
}
head_sha="$(git -C "$HERE/../.." rev-parse HEAD)"
fr="$tmp/fr"
fakerel "$fr" tools-v0.1.20 abc 1
got="$(pfetch "$fr" FV_PREBUILT_STALE=latest | paste -sd' ')"
check "fetch, stale (FV_PREBUILT_STALE=latest): no boot-build=, fv-boot compiled from HEAD" \
  "ctx:serve-build=$fr/pre/serve-cuda ctx:cpu-build=$fr/pre/serve-cpu ok=true exact=false fv_boot=compiled@$head_sha" "$got"
grep -q "::notice title=fv-boot compiled from this commit::tools-v0.1.20 is older" "$fr/fetch.log" && echo "ok   (stale notice)" || { echo "FAIL stale notice"; cat "$fr/fetch.log"; fail=1; }
got="$(pfetch "$fr" FV_TOOLS_VERSION=0.1.20 | paste -sd' ')"
check "fetch, FV_TOOLS_VERSION pin (not exact): the pinned release's fv-boot stays with its fv-serve" \
  "ctx:serve-build=$fr/pre/serve-cuda ctx:cpu-build=$fr/pre/serve-cpu ctx:boot-build=$fr/pre/serve-cuda ok=true exact=false fv_boot=tools-v0.1.20" "$got"
fakerel "$fr" tools-v0.1.20 "$(bash "$T" input-hash HEAD)" 1
got="$(pfetch "$fr" | paste -sd' ')"
check "fetch, exact: the release's fv-boot" \
  "ctx:serve-build=$fr/pre/serve-cuda ctx:cpu-build=$fr/pre/serve-cpu ctx:boot-build=$fr/pre/serve-cuda ok=true exact=true fv_boot=tools-v0.1.20" "$got"
fakerel "$fr" tools-v0.1.20 abc 0
got="$(pfetch "$fr" FV_PREBUILT_STALE=latest | paste -sd' ')"
check "fetch, stale release without fv-boot: compiled" \
  "ctx:serve-build=$fr/pre/serve-cuda ctx:cpu-build=$fr/pre/serve-cpu ok=true exact=false fv_boot=compiled@$head_sha" "$got"
grep -q "::notice title=fv-boot not in tools-v0.1.20::" "$fr/fetch.log" && ! grep -q "fv-boot compiled from this commit" "$fr/fetch.log" \
  && echo "ok   (says the release has no fv-boot)" || { echo "FAIL no-fv-boot notice"; cat "$fr/fetch.log"; fail=1; }
exit "$fail"
