#!/usr/bin/env bash
# The b4a / b4b / b4c verdicts (scripts/gpu/b4-verdict.sh) on stub run
# trees: a passing cell, and one failing case per rule (exit codes, the
# planned hit / miss, the H3 FP8 encoder, skipped / failed / missing
# compares, cache entries, temporary files, the fused kernels of the traced
# step, cells without a verdict).
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

# b4b: the fused kernels of the CUPTI-traced step (stderr.log "/gpu_trace"
# line, as rt/gpu_trace.rs prints it: compact JSON, keys in order).
trace() { # <cell> <kernel:calls ...> (none: a trace without fused kernels)
  mkdir -p "$RUNS/$1"
  local cell="$1" k objs="" sep=""
  shift
  for k in gemm_bf16:200 vsa_mma_attn:50 "$@"; do
    objs+="$sep{\"calls\":${k#*:},\"category\":\"x\",\"name\":\"${k%%:*}\",\"share\":0.1,\"total_s\":0.01}"
    sep=","
  done
  {
    printf '[INFO] loading\n'
    printf '[INFO] h3/gpu_trace {"by_category":[{"calls":3,"category":"gemm"}],"pass":"h3","step":1,"top_kernels":[%s],"wall_s":1.0}\n' "$objs"
    printf '[INFO] done\n'
  } >"$RUNS/$cell/stderr.log"
}
good_b4b() { # <cell> <kernel:calls ...>
  local arm
  for arm in base pr; do
    summary "$1-$arm" 0
    trace "$1-$arm" "${@:2}"
  done
  compare "$1-base" "$1-pr" ok
}
on_kernels=(fvf_h3_gate_res_norm_mod:100 h3_norm_mod:2 fvf_h3_qkvg_heads:50 fvf_split_heads_rows:10)
off_kernels=(h3_norm_mod:2 h3_res_gate_norm_mod:100 fvf_split_heads_rows:10)
b=b4b-h3

fresh; good_b4b $b "${on_kernels[@]}"
expect pass - "b4b: fused kernels in both arms, identical" b4b_verdict $b on
[[ -f "$RUNS/b4b-$b.verdict.json" ]] || { echo "FAIL no b4b verdict file" >&2; fails=$((fails + 1)); }
if [[ "$(cat "$RUNS/$b-pr/fusion-evidence.log")" == $'fvf_h3_gate_res_norm_mod 100\nfvf_h3_qkvg_heads 50\nfvf_split_heads_rows 10\nh3_norm_mod 2' ]]; then
  echo "ok   b4b evidence: fused kernels only, name order, with calls"
else
  echo "FAIL b4b evidence: $(cat "$RUNS/$b-pr/fusion-evidence.log")" >&2
  fails=$((fails + 1))
fi

fresh; good_b4b $b "${off_kernels[@]}"
expect pass - "b4b off: no gated kernel, split rows allowed" b4b_verdict $b off

fresh; good_b4b $b "${off_kernels[@]}"
expect fail "base:no-fused-kernel" "b4b on: fusion did not run" b4b_verdict $b on

fresh; good_b4b $b "${on_kernels[@]}"
expect fail "pr:fused-kernel-ran" "b4b off: a gated kernel ran" b4b_verdict $b off

fresh; good_b4b $b "${on_kernels[@]}"; trace $b-pr fvf_h3_gate_res_norm_mod:99 h3_norm_mod:2 fvf_h3_qkvg_heads:50
expect fail "evidence-differs" "b4b: different call counts" b4b_verdict $b on

fresh; good_b4b $b "${on_kernels[@]}"; trace $b-base fvf_h3_gate_res_norm_mod:100 h3_norm_mod:2
expect fail "evidence-differs" "b4b: a fused kernel missing in one arm" b4b_verdict $b on

fresh; good_b4b $b "${on_kernels[@]}"; printf '[INFO] h3/gpu_trace {"pass":"h3","step":1,"error":"libcupti not found"}\n' >"$RUNS/$b-pr/stderr.log"
expect fail "pr:no-trace" "b4b: the trace failed (no kernel table)" b4b_verdict $b on

fresh; good_b4b $b "${on_kernels[@]}"; rm "$RUNS/$b-base/stderr.log"
expect fail "base:no-trace" "b4b: an arm that never ran" b4b_verdict $b off

fresh; good_b4b $b "${on_kernels[@]}"; summary $b-base 1
expect fail "base:exit" "b4b: an arm that exited 1" b4b_verdict $b on

fresh; good_b4b $b "${on_kernels[@]}"; compare $b-base $b-pr skipped
expect fail "not-identical" "b4b: a skipped compare" b4b_verdict $b on

fresh; good_b4b $b "${on_kernels[@]}"; compare $b-base $b-pr different
expect fail "not-identical" "b4b: a failed --off-identity" b4b_verdict $b on

fresh; good_b4b $b "${on_kernels[@]}"
expect fail "rule:maybe" "b4b: an unknown rule" b4b_verdict $b maybe

fresh; good_b4b $b "${on_kernels[@]}"; good_b4b b4b-ltx fvf_ltx_res_norm_mod:300 fvf_ltx_qk_norm_rope:200
b4b_verdict $b on >/dev/null; b4b_verdict b4b-ltx on >/dev/null
expect pass - "b4b family: both cells passed" b4_family_verdict b4b $b b4b-ltx
expect fail "b4b-ltx-off:no-verdict" "b4b family: a named cell that never ran" b4_family_verdict b4b $b b4b-ltx b4b-ltx-off

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
