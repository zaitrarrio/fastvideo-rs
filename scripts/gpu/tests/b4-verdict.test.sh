#!/usr/bin/env bash
# The b4a / b4c / b4c2 verdicts (scripts/gpu/b4-verdict.sh) on stub run
# trees: a passing cell, and one failing case per rule (exit codes, the
# planned hit / miss, the H3 FP8 encoder, skipped / failed / missing
# compares, cache entries, temporary files, cells without a verdict).
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

# b4c2: a setting switched between a write and a read.
good_switch() { # <cell>
  summary "$1-base-write" 0; bench "$1-base-write" miss resident-fp8
  summary "$1-base-read" 0; bench "$1-base-read" hit cache
  summary "$1-pr-write" 0; bench "$1-pr-write" miss resident-fp8
  summary "$1-pr-read" 0; bench "$1-pr-read" miss resident-bf16
  summary "$1-pr-fresh" 0; bench "$1-pr-fresh" miss resident-bf16
  compare "$1-pr-fresh" "$1-pr-read" ok
  cache "$1-base" "aa.ltx2text" "fp8"
  cache "$1-pr" "bb.ltx2text" "fp8"
  cache "$1-pr" "cc.ltx2text" "bf16"
  cache "$1-pr-fresh" "cc.ltx2text" "bf16"
}
s2=b4c2-ltx-fp8

fresh; good_switch $s2
expect pass - "b4c2 switch: all evidence present" b4c2_switch_verdict $s2
[[ -f "$RUNS/b4c2-$s2.verdict.json" ]] || { echo "FAIL no b4c2 verdict file" >&2; fails=$((fails + 1)); }

fresh; good_switch $s2; cache $s2-pr-fresh "cc.ltx2text" "other"
expect fail "fresh-entry-not-stored-by-read" "b4c2 switch: the fresh encode stored other bytes" b4c2_switch_verdict $s2

fresh; good_switch $s2; rm "$B4C_CACHE/$s2-pr-fresh/cc.ltx2text"; cache $s2-pr-fresh "ee.ltx2text" "bf16"
expect fail "fresh-entry-not-stored-by-read" "b4c2 switch: the fresh encode stored under another name" b4c2_switch_verdict $s2

fresh; good_switch $s2; rm -rf "$B4C_CACHE/$s2-pr-fresh"
expect fail "pr-fresh:entries=0" "b4c2 switch: the fresh encode stored nothing" b4c2_switch_verdict $s2

fresh; good_switch $s2; summary $s2-pr-fresh 1
expect fail "$s2-pr-fresh:exit" "b4c2 switch: the fresh pass failed" b4c2_switch_verdict $s2

fresh; good_switch $s2; rm "$RUNS/$s2-base-read/summary.json"
expect fail "$s2-base-read:exit" "b4c2 switch: a pass that never ran" b4c2_switch_verdict $s2

fresh; good_switch $s2; bench $s2-pr-write hit cache
expect fail "$s2-pr-write:text_cache=hit" "b4c2 switch: the write hit (a stale cache)" b4c2_switch_verdict $s2

fresh; good_switch $s2; bench $s2-base-read miss resident-bf16
expect fail "$s2-base-read:text_cache=miss" "b4c2 switch: main did not reproduce the gap" b4c2_switch_verdict $s2

fresh; good_switch $s2; bench $s2-pr-read hit cache
expect fail "$s2-pr-read:text_cache=hit" "b4c2 switch: this build served the other setting" b4c2_switch_verdict $s2

fresh; good_switch $s2; bench $s2-pr-fresh hit cache
expect fail "$s2-pr-fresh:text_cache=hit" "b4c2 switch: the fresh pass hit (not a cold cache)" b4c2_switch_verdict $s2

fresh; good_switch $s2; compare $s2-pr-fresh $s2-pr-read different
expect fail "read-vs-fresh:not-identical" "b4c2 switch: the miss differs from a fresh encode" b4c2_switch_verdict $s2

fresh; good_switch $s2; compare $s2-pr-fresh $s2-pr-read skipped
expect fail "read-vs-fresh:not-identical" "b4c2 switch: a skipped compare" b4c2_switch_verdict $s2

fresh; good_switch $s2; rm "$B4C_CACHE/$s2-pr/cc.ltx2text"
expect fail "pr:entries=1" "b4c2 switch: this build stored one entry" b4c2_switch_verdict $s2

fresh; good_switch $s2; cache $s2-base "dd.ltx2text" "x"
expect fail "base:entries=2" "b4c2 switch: main stored two entries" b4c2_switch_verdict $s2

fresh; good_switch $s2; cache $s2-pr "k.1.part" "half"
expect fail "pr:temporary-files" "b4c2 switch: a .part left" b4c2_switch_verdict $s2

# b4c2: the no-change cell.
good_noop() { # <cell>
  local arm
  for arm in base pr; do
    summary "$1-$arm-miss" 0; bench "$1-$arm-miss" miss resident-fp8
    summary "$1-$arm-hit" 0; bench "$1-$arm-hit" hit cache
  done
  cache "$1-base" "aa.h3text" "entry"
  cache "$1-pr" "bb.h3text" "entry"
  compare "$1-base-miss" "$1-pr-miss" ok
  compare "$1-base-hit" "$1-pr-hit" ok
  compare "$1-pr-miss" "$1-pr-hit" ok
}
s3=b4c2-h3-noop

fresh; good_noop $s3
expect pass - "b4c2 noop: all evidence present" b4c2_noop_verdict $s3

fresh; good_noop $s3; summary $s3-pr-hit 1
expect fail "$s3-pr-hit:exit" "b4c2 noop: a pass that exited 1" b4c2_noop_verdict $s3

fresh; good_noop $s3; bench $s3-pr-hit miss resident-fp8
expect fail "$s3-pr-hit:text_cache=miss" "b4c2 noop: the hit missed" b4c2_noop_verdict $s3

fresh; good_noop $s3; compare $s3-base-miss $s3-pr-miss different
expect fail "miss:not-identical" "b4c2 noop: the miss moved an output bit" b4c2_noop_verdict $s3

fresh; good_noop $s3; compare $s3-pr-miss $s3-pr-hit different
expect fail "pr-hit-vs-miss:not-identical" "b4c2 noop: the hit differs from the miss" b4c2_noop_verdict $s3

fresh; good_noop $s3; rm "$RUNS/compare/compare-clips-$s3-base-hit--$s3-pr-hit.json"
expect fail "hit:not-identical" "b4c2 noop: a missing compare" b4c2_noop_verdict $s3

fresh; good_noop $s3; cache $s3-pr "bb.h3text" "other"
expect fail "entry-bytes-differ" "b4c2 noop: different conditioning bytes" b4c2_noop_verdict $s3

fresh; good_noop $s3; rm "$B4C_CACHE/$s3-pr/bb.h3text"; cache $s3-pr "aa.h3text" "entry"
expect fail "entry-name-unchanged" "b4c2 noop: the key did not move" b4c2_noop_verdict $s3

fresh; good_noop $s3; cache $s3-pr "cc.h3text" "entry"
expect fail "pr:entries=2" "b4c2 noop: an extra entry" b4c2_noop_verdict $s3

fresh; good_noop $s3; rm -rf "$B4C_CACHE/$s3-base" "$B4C_CACHE/$s3-pr"
expect fail "no-cache-entries" "b4c2 noop: no cache written" b4c2_noop_verdict $s3

fresh; good_noop $s3; cache $s3-base "aa.tmp9" "half"
expect fail "base:temporary-files" "b4c2 noop: a .tmp left" b4c2_noop_verdict $s3

fresh; good_noop $s3; good_switch $s2
b4c2_noop_verdict $s3 >/dev/null; b4c2_switch_verdict $s2 >/dev/null
expect pass - "b4c2 family: both cells passed" b4_family_verdict b4c2 $s3 $s2
bench $s2-pr-read hit cache; b4c2_switch_verdict $s2 >/dev/null
expect fail "$s2:fail" "b4c2 family: a failed cell" b4_family_verdict b4c2 $s3 $s2

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
