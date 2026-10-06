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
check "pin" tools-v0.1.2 "$(bash "$T" resolve --version v0.1.2 | jq -r .tag)"
check "list order" "0.2.0 0.1.11-pre.abc 0.1.10 0.1.10-pre.old 0.1.9 0.1.2" "$(bash "$T" list | cut -f1 | paste -sd' ')"
check "prune keeps N stable, drops released prereleases, never other tags" \
  "tools-v0.1.9 tools-v0.1.2 tools-v0.1.10-pre.old" \
  "$(FV_TOOLS_KEEP=1 FV_GITHUB_TOKEN_FILE=/dev/null bash "$T" prune --dry-run 2>&1 | sed -n 's/.*prune (dry run): //p' | paste -sd' ')"
check "unchanged inputs: nothing to publish" 0 "$(
  h="$(bash "$T" input-hash HEAD)"
  jq --arg h "$h" '.[0].body |= sub("input-hash: aaa10"; "input-hash: \($h)")' "$tmp/rels.json" >"$tmp/same.json"
  FV_TOOLS_RELEASES_FILE="$tmp/same.json" FV_GITHUB_TOKEN_FILE=/dev/null bash "$T" publish HEAD --prerelease --dry-run 2>"$tmp/log" >/dev/null; echo $?)"
grep -q "nothing to publish" "$tmp/log" && echo "ok   (said nothing to publish)" || { echo "FAIL unchanged message"; fail=1; }
exit "$fail"
