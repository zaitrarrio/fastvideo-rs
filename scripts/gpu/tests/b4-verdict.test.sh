#!/usr/bin/env bash
# The b4b / b4c / b4c2 verdicts (scripts/gpu/b4-verdict.sh) on stub run
# trees: a passing cell, and one failing case per rule (exit codes, the
# planned hit / miss, the H3 FP8 encoder, skipped / failed / missing
# compares, cache entries, temporary files, the fused kernels of the traced
# step, cells without a verdict). b4a retired to fv-lab (ADR-0019 §11.6):
# its verdict is fv-lab's [verdict] step, held to the script's last b4a
# branch by crates/fastvideo-lab/tests/verdict.rs.
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

# b4b: the CUPTI-traced step (stderr.log "/gpu_trace" line, as
# rt/gpu_trace.rs prints it: compact JSON, keys in order).
# trace <cell> <top N or all> <dropped> <kernel:calls ...>: a step running
# two GEMM / attention kernels and the given ones; the table lists the N
# kernels with the most time (each listed kernel's time is its calls here).
trace() {
  mkdir -p "$RUNS/$1"
  local cell="$1" top="$2" dropped="$3" k objs="" sep="" total=0 n=0 rows
  shift 3
  rows="$(for k in gemm_bf16:200 vsa_mma_attn:150 "$@"; do printf '%s %s\n' "${k#*:}" "${k%%:*}"; done | sort -k1,1nr -k2,2)"
  while read -r calls name; do
    total=$((total + calls))
    n=$((n + 1))
    [[ "$top" != all ]] && (( n > top )) && continue
    objs+="$sep{\"calls\":$calls,\"category\":\"x\",\"name\":\"$name\",\"share\":0.1,\"total_s\":0.01}"
    sep=","
  done <<<"$rows"
  {
    printf '[INFO] loading\n'
    printf '[INFO] h3/gpu_trace {"by_category":[{"calls":3,"category":"gemm"}],"dropped_records":%s,"kernel_count":%s,"pass":"h3","step":1,"top_kernels":[%s],"wall_s":1.0}\n' "$dropped" "$total" "$objs"
    printf '[INFO] done\n'
  } >"$RUNS/$cell/stderr.log"
}
# good_b4b <cell> <base top> <kernel:calls ...>: both arms exit 0 with the
# same step, the base listing <base top> kernels, the PR all; identical.
good_b4b() {
  local cell="$1" top="$2"
  shift 2
  summary "$cell-base" 0
  summary "$cell-pr" 0
  trace "$cell-base" "$top" 0 "$@"
  trace "$cell-pr" all 0 "$@"
  compare "$cell-base" "$cell-pr" ok
}
# H3, 40 blocks: fused, block 0's norm and the last block's residual run
# alone, 39 deferred residuals fuse with the next norm.
h3_on=(h3_res_gate_norm_mod:40 h3_norm_mod:1 h3_gate_residual:1 fvf_h3_gate_res_norm_mod:39 fvf_h3_qkvg_heads:40 h3_swiglu_mx:40 fvf_split_heads_rows:10)
h3_off=(h3_res_gate_norm_mod:40 h3_norm_mod:40 h3_gate_residual:40 h3_qk_norm_rope:80 fvf_split_heads_rows:10)
on_named=(fvf_h3_gate_res_norm_mod fvf_h3_qkvg_heads)
off_named=(h3_norm_mod h3_gate_residual h3_qk_norm_rope)
b=b4b-h3

fresh; good_b4b $b 25 "${h3_on[@]}"
expect pass - "b4b on: named kernels, block structure, complete PR table" b4b_verdict $b on h3 "${on_named[@]}"
[[ -f "$RUNS/b4b-$b.verdict.json" ]] || { echo "FAIL no b4b verdict file" >&2; fails=$((fails + 1)); }
if [[ "$(head -2 "$RUNS/$b-pr/fusion-evidence.log")" == $'fvf_h3_gate_res_norm_mod 39\nfvf_h3_qkvg_heads 40' ]] &&
  ! grep -q gemm "$RUNS/$b-pr/fusion-evidence.log"; then
  echo "ok   b4b evidence: fused and bf16-chain kernels only, name order, with calls"
else
  echo "FAIL b4b evidence: $(cat "$RUNS/$b-pr/fusion-evidence.log")" >&2
  fails=$((fails + 1))
fi

fresh; good_b4b $b 3 "${h3_on[@]}"
expect pass - "b4b on: the base lists only its top 3 (the old limit)" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 "${h3_off[@]}"
expect pass - "b4b off: the unfused path with its call counts" b4b_verdict $b off h3 "${off_named[@]}"

fresh; good_b4b $b 25 "${h3_off[@]}"
expect fail "pr:missing:fvf_h3_gate_res_norm_mod" "b4b on: fusion did not run" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 h3_res_gate_norm_mod:40 h3_norm_mod:1 h3_gate_residual:1 fvf_h3_gate_res_norm_mod:39 h3_qk_norm_rope:80
expect fail "pr:missing:fvf_h3_qkvg_heads" "b4b on: one named kernel missing (glue)" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 "${h3_on[@]}"
expect fail "pr:fused-kernel-ran:fvf_h3_gate_res_norm_mod,fvf_h3_qkvg_heads" "b4b off: gated kernels ran" b4b_verdict $b off h3 "${off_named[@]}"

fresh; good_b4b $b 2 "${h3_off[@]}" fvf_ltx_gate_merge:1
expect fail "pr:fused-kernel-ran:fvf_ltx_gate_merge" "b4b off: a gated kernel below the base's cut" b4b_verdict $b off - "${off_named[@]}"

fresh; good_b4b $b 25 h3_res_gate_norm_mod:40 h3_norm_mod:40 h3_gate_residual:40
expect fail "pr:missing:h3_qk_norm_rope" "b4b off: an unfused-path kernel missing" b4b_verdict $b off h3 "${off_named[@]}"

fresh; good_b4b $b 25 h3_res_gate_norm_mod:40 h3_norm_mod:2 h3_gate_residual:1 fvf_h3_gate_res_norm_mod:39 fvf_h3_qkvg_heads:40
expect fail "relation:norm2+fused39!=blocks40" "b4b on: an extra norm breaks the block structure" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 h3_res_gate_norm_mod:40 h3_norm_mod:40 h3_gate_residual:39 h3_qk_norm_rope:80
expect fail "relation:norm40,residual39!=blocks40" "b4b off: a residual missing from the block structure" b4b_verdict $b off h3 "${off_named[@]}"

fresh; good_b4b $b 25 "${h3_on[@]}"; trace $b-pr all 0 h3_res_gate_norm_mod:40 h3_norm_mod:1 h3_gate_residual:1 fvf_h3_gate_res_norm_mod:38 fvf_h3_qkvg_heads:40 h3_swiglu_mx:40 fvf_split_heads_rows:10 h3_qk_norm_rope:1
expect fail "evidence-differs" "b4b: different call counts" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 "${h3_on[@]}"; trace $b-pr all 0 "${h3_on[@]}" mx_binary:4
expect fail "kernel-count-differs" "b4b: one more launch in the PR step" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 "${h3_on[@]}"; trace $b-pr 3 0 "${h3_on[@]}"
expect fail "pr:incomplete-trace" "b4b: the PR table cut (FASTVIDEO_GPU_TRACE_TOP not read)" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 "${h3_on[@]}"; trace $b-pr all 3 "${h3_on[@]}"
expect fail "pr:dropped-records=3" "b4b: CUPTI dropped records (PR)" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 "${h3_on[@]}"; trace $b-base 25 1 "${h3_on[@]}"
expect fail "base:dropped-records=1" "b4b: CUPTI dropped records (base)" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 "${h3_on[@]}"; sed -i 's/"dropped_records":0,//' "$RUNS/$b-pr/stderr.log"
expect fail "pr:dropped-records=none" "b4b: no dropped_records field" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 "${h3_on[@]}"; sed -i 's/"kernel_count":[0-9]*,//' "$RUNS/$b-pr/stderr.log"
expect fail "pr:no-kernel-count" "b4b: no kernel_count field" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 "${h3_on[@]}"; printf '[INFO] h3/gpu_trace {"pass":"h3","step":1,"error":"libcupti not found"}\n' >"$RUNS/$b-pr/stderr.log"
expect fail "pr:no-trace" "b4b: the trace failed (no kernel table)" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 "${h3_on[@]}"; rm "$RUNS/$b-base/stderr.log"
expect fail "base:no-trace" "b4b: an arm that never ran" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 "${h3_on[@]}"; summary $b-base 1
expect fail "base:exit" "b4b: an arm that exited 1" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 "${h3_on[@]}"; compare $b-base $b-pr skipped
expect fail "not-identical" "b4b: a skipped compare" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 "${h3_on[@]}"; compare $b-base $b-pr different
expect fail "not-identical" "b4b: a failed --off-identity" b4b_verdict $b on h3 "${on_named[@]}"

fresh; good_b4b $b 25 "${h3_on[@]}"
expect fail "rule:maybe" "b4b: an unknown rule" b4b_verdict $b maybe h3 "${on_named[@]}"

ltx_on=(fvf_ltx_res_norm_mod:300 fvf_ltx_qk_norm_rope:200 fvf_ltx_gate_merge:100 mx_binary:50)
fresh; good_b4b b4b-ltx 25 "${ltx_on[@]}"
expect pass - "b4b LTX on: all three fused kernels" b4b_verdict b4b-ltx on - fvf_ltx_res_norm_mod fvf_ltx_qk_norm_rope fvf_ltx_gate_merge
fresh; good_b4b b4b-ltx 25 fvf_ltx_qk_norm_rope:200 fvf_ltx_gate_merge:100 mx_binary:50
expect fail "pr:missing:fvf_ltx_res_norm_mod" "b4b LTX on: the q/k kernel alone is not enough" b4b_verdict b4b-ltx on - fvf_ltx_res_norm_mod fvf_ltx_qk_norm_rope fvf_ltx_gate_merge
fresh; good_b4b b4b-ltx 25 mx_rms_norm:400 mx_rope_half:200 mx_binary:300 fvf_split_heads_rows:200
expect pass - "b4b LTX off: the bf16 op chain ran" b4b_verdict b4b-ltx off - mx_rms_norm mx_rope_half

fresh; good_b4b $b 25 "${h3_on[@]}"; good_b4b b4b-ltx 25 "${ltx_on[@]}"
b4b_verdict $b on h3 "${on_named[@]}" >/dev/null; b4b_verdict b4b-ltx on - fvf_ltx_res_norm_mod >/dev/null
expect pass - "b4b family: both cells passed" b4_family_verdict b4b $b b4b-ltx
expect fail "b4b-ltx-off:no-verdict" "b4b family: a named cell that never ran" b4_family_verdict b4b $b b4b-ltx b4b-ltx-off

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

# The shared pieces on their own: exit and identity (every verdict above
# builds on them).
fresh; summary pair-x-base 0; summary pair-x-pr 0; compare pair-x-base pair-x-pr ok
if b4_exit_ok pair-x-base && b4_identical pair-x-base pair-x-pr; then echo "ok   exit and identity pieces pass"; else echo "FAIL exit and identity pieces" >&2; fails=$((fails + 1)); fi
summary pair-x-pr 10; compare pair-x-base pair-x-pr different
if b4_exit_ok pair-x-pr || b4_identical pair-x-base pair-x-pr; then echo "FAIL exit 10 / different" >&2; fails=$((fails + 1)); else echo "ok   exit 10 and a failed compare are caught"; fi

if (( fails )); then
  echo "b4-verdict: $fails FAILED" >&2
  exit 1
fi
echo "b4-verdict: all ok"
