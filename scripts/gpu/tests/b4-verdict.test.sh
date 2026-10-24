#!/usr/bin/env bash
# The b4a / b4c verdicts (scripts/gpu/b4-verdict.sh) on stub run trees: a
# passing cell, and one failing case per rule (exit codes, the planned
# hit / miss, the H3 FP8 encoder, skipped / failed / missing compares,
# cache entries, temporary files, cells without a verdict).
#   bash scripts/gpu/tests/b4-verdict.test.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/gpu/b4-verdict.sh
. "$HERE/../b4-verdict.sh"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fails=0
n=0

# Pretty-printed like fv-gpucheck's reports.
summary() { # <cell> <exit>
  mkdir -p "$RUNS/$1"
  printf '{"cell":"%s","family":"b4c","exit":%s,"seconds":12,"ended":"x"}\n' "$1" "$2" >"$RUNS/$1/summary.json"
}
bench() { # <cell> <text_cache> <text_encoder>
  mkdir -p "$RUNS/$1"
  printf '{\n  "text_s": 1.5,\n  "text_cache": "%s",\n  "text_encoder": "%s",\n  "stage_seconds": {\n    "text": 1.5\n  }\n}\n' "$2" "$3" >"$RUNS/$1/benchmark.json"
}
compare() { # <base> <cand> ok|different|skipped
  mkdir -p "$RUNS/compare"
  local f="$RUNS/compare/compare-clips-$1--$2.json" ok=true
  if [[ "$3" == skipped ]]; then
    printf '{"stage":"compare-clips-%s--%s","status":"skipped","reason":"frames missing"}\n' "$1" "$2" >"$f"
    return
  fi
  [[ "$3" == ok ]] || ok=false
  cat >"$f" <<EOF
{
  "checks": [
    {
      "name": "frame_shape",
      "ok": true,
      "values": {},
      "limits": {}
    },
    {
      "name": "off_identity",
      "ok": $ok,
      "values": {
        "status": "x"
      },
      "limits": {
        "max_abs_diff_uint8": 0
      }
    },
    {
      "name": "off_identity_frame_count",
      "ok": true,
      "values": {},
      "limits": {}
    }
  ],
  "status": "$([[ $ok == true ]] && echo pass || echo fail)"
}
EOF
}
cache() { # <dir> <name> <content>
  mkdir -p "$B4C_CACHE/$1"
  printf '%s' "$3" >"$B4C_CACHE/$1/$2"
}

# A complete passing b4c cell.
good_cell() { # <cell> <miss encoder>
  local arm
  for arm in base pr; do
    summary "$1-$arm-miss" 0
    summary "$1-$arm-hit" 0
    bench "$1-$arm-miss" miss "$2"
    bench "$1-$arm-hit" hit cache
    cache "$1-$arm" "aa.h3text" "entry"
  done
  compare "$1-base-miss" "$1-pr-miss" ok
  compare "$1-base-hit" "$1-pr-hit" ok
}

fresh() {
  RUNS="$TMP/runs-$n"
  B4C_CACHE="$TMP/cache-$n"
  n=$((n + 1))
  mkdir -p "$RUNS" "$B4C_CACHE"
}

expect() { # <pass|fail> <why substring or -> <label> <command...>
  local want="$1" sub="$2" label="$3" out
  shift 3
  out="$("$@")"
  if [[ "$out" != *": $want"* || ( "$sub" != - && "$out" != *"$sub"* ) ]]; then
    printf 'FAIL %s: got %s\n' "$label" "$out" >&2
    fails=$((fails + 1))
  else
    printf 'ok   %s: %s\n' "$label" "$out"
  fi
}

c=b4c-h3

fresh; good_cell $c resident-fp8
expect pass - "all evidence present (H3, fp8)" b4c_verdict $c fp8
[[ -f "$RUNS/b4c-$c.verdict.json" ]] || { echo "FAIL no verdict file" >&2; fails=$((fails + 1)); }

fresh; good_cell $c streamed-fp8
expect pass - "streamed FP8 counts as fp8" b4c_verdict $c fp8

fresh; good_cell $c resident-fp8; summary $c-pr-hit 1
expect fail "$c-pr-hit:exit" "a pass that exited 1" b4c_verdict $c fp8

fresh; good_cell $c resident-fp8; summary $c-base-miss null
expect fail "$c-base-miss:exit" "a skipped pass (exit null)" b4c_verdict $c fp8

fresh; good_cell $c resident-fp8; rm "$RUNS/$c-pr-miss/summary.json"
expect fail "$c-pr-miss:exit" "a pass that never ran" b4c_verdict $c fp8

fresh; good_cell $c resident-fp8; bench $c-base-hit miss resident-fp8
expect fail "$c-base-hit:text_cache=miss" "a hit pass that missed" b4c_verdict $c fp8

fresh; good_cell $c resident-fp8; rm "$RUNS/$c-pr-hit/benchmark.json"
expect fail "$c-pr-hit:text_cache=none" "no benchmark.json" b4c_verdict $c fp8

fresh; good_cell $c streamed
expect fail "$c-base-miss:text_encoder=streamed" "bf16 streamed encoder (80 GB card)" b4c_verdict $c fp8

fresh; good_cell $c resident-fp8; bench $c-pr-hit hit resident-fp8
expect fail "$c-pr-hit:text_encoder=resident-fp8" "a hit that ran the encoder" b4c_verdict $c fp8

fresh; good_cell $c streamed
expect pass - "rule - ignores the encoder (LTX)" b4c_verdict $c -

fresh; good_cell $c resident-fp8; compare $c-base-miss $c-pr-miss skipped
expect fail "miss:not-identical" "a skipped compare" b4c_verdict $c fp8

fresh; good_cell $c resident-fp8; compare $c-base-hit $c-pr-hit different
expect fail "hit:not-identical" "a failed --off-identity" b4c_verdict $c fp8

fresh; good_cell $c resident-fp8; rm "$RUNS/compare/compare-clips-$c-base-hit--$c-pr-hit.json"
expect fail "hit:not-identical" "a missing compare report" b4c_verdict $c fp8

fresh; good_cell $c resident-fp8; cache $c-pr aa.h3text "other"
expect fail "cache-entries-differ" "different entry bytes" b4c_verdict $c fp8

fresh; good_cell $c resident-fp8; cache $c-pr bb.h3text "entry"
expect fail "cache-entries-differ" "an extra entry" b4c_verdict $c fp8

fresh; good_cell $c resident-fp8; rm -rf "$B4C_CACHE/$c-base" "$B4C_CACHE/$c-pr"
expect fail "no-cache-entries" "no cache written" b4c_verdict $c fp8

fresh; good_cell $c resident-fp8; cache $c-pr "k.1234.part" "half"; cache $c-base "k.1234.part" "half"
expect fail "pr:temporary-files" "a .part left" b4c_verdict $c fp8

fresh; good_cell $c resident-fp8; cache $c-base "aa.tmp77" "half"; cache $c-pr "aa.tmp77" "half"
expect fail "base:temporary-files" "a .tmp left" b4c_verdict $c fp8

# Families.
fresh; good_cell $c resident-fp8; good_cell b4c-ltx resident-fp8
b4c_verdict $c fp8 >/dev/null; b4c_verdict b4c-ltx - >/dev/null
expect pass - "family: both cells passed" b4_family_verdict b4c $c b4c-ltx
expect fail "b4c-typo:no-verdict" "family: a name that matches no cell" b4_family_verdict b4c $c b4c-typo
expect fail "no-cells" "family: no cells" b4_family_verdict b4c
printf '{"cell":"x","verdict":"fail","why":"y"}\n' >"$RUNS/b4c-b4c-ltx.verdict.json"
expect fail "b4c-ltx:fail" "family: a failed cell" b4_family_verdict b4c $c b4c-ltx

# b4a's pieces: exit and identity.
fresh; summary b4a-x-base 0; summary b4a-x-pr 0; compare b4a-x-base b4a-x-pr ok
if b4_exit_ok b4a-x-base && b4_identical b4a-x-base b4a-x-pr; then echo "ok   b4a pieces pass"; else echo "FAIL b4a pieces" >&2; fails=$((fails + 1)); fi
summary b4a-x-pr 10; compare b4a-x-base b4a-x-pr different
if b4_exit_ok b4a-x-pr || b4_identical b4a-x-base b4a-x-pr; then echo "FAIL b4a exit 10 / different" >&2; fails=$((fails + 1)); else echo "ok   b4a exit 10 and a failed compare are caught"; fi

if (( fails )); then
  echo "b4-verdict: $fails FAILED" >&2
  exit 1
fi
echo "b4-verdict: all ok"
