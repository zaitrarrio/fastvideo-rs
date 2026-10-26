#!/usr/bin/env bash
# Verdicts of the plan B4 byte-identity families of runpod-matrix.sh (b4b,
# b4c, b4c2), sourced by it. b4a retired to fv-lab (docs/adr/0019-fv-lab.md
# §11.6): its verdict is the tier's typed [verdict] step, and the last script
# branch that used b4_exit_ok / b4_identical / b4_family_verdict for it is
# vendored, with this file as it was then, under
# crates/fastvideo-lab/fixtures/b4a-script/ (tests/verdict.rs). Pure functions over a run directory ($RUNS)
# and, for b4c / b4c2, the cache root ($B4C_CACHE), so scripts/gpu/tests/b4-verdict.test.sh
# holds them to stub trees. A verdict passes only on positive evidence: a
# missing file, a skipped compare or a run that did not exit 0 fails it.

# b4_flat <file>: the file on one line without whitespace (the reports are
# pretty-printed JSON; values here never contain spaces).
b4_flat() { { tr -d ' \t\n\r' <"$1"; } 2>/dev/null || true; }

# b4_field <file> <key>: the first string value of "key" in a JSON file.
b4_field() {
  b4_flat "$1" | grep -oE "\"$2\":\"[^\"]*\"" | head -1 | sed -E 's/^"[^"]*":"(.*)"$/\1/' || true
}

# b4_exit_ok <cell>: run_cell's summary.json for <cell> records exit 0 (not
# skipped, not failed, not absent).
b4_exit_ok() {
  [[ -f "$RUNS/$1/summary.json" ]] && [[ "$(b4_flat "$RUNS/$1/summary.json")" == *'"exit":0'[,}]* ]]
}

# b4_identical <base> <cand>: the compare-clips --off-identity report of the
# two cells exists, is not a skip, and both identity checks passed.
b4_identical() {
  local f="$RUNS/compare/compare-clips-$1--$2.json" flat
  [[ -f "$f" ]] || return 1
  flat="$(b4_flat "$f")"
  [[ "$flat" != *'"status":"skipped"'* ]] &&
    [[ "$flat" == *'"name":"off_identity","ok":true'* ]] &&
    [[ "$flat" == *'"name":"off_identity_frame_count","ok":true'* ]]
}

# b4c_entries <dir>: the cache directory's files, SHA-256 and name, in name
# order.
b4c_entries() {
  (cd "$1" 2>/dev/null && find . -maxdepth 1 -type f -printf '%P\n' | LC_ALL=C sort | while IFS= read -r f; do
    printf '%s %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$f"
  done)
}

# b4c_cache_clean <dir>: no temporary (*.tmp*) or partial (*.part) file left.
b4c_cache_clean() {
  [[ -z "$(find "$1" -maxdepth 1 \( -name '*.part' -o -name '*.tmp*' \) -print -quit 2>/dev/null)" ]]
}

# b4c_verdict <cell> <encoder rule>: $RUNS/b4c-<cell>.verdict.json from the
# four passes <cell>-{base,pr}-{miss,hit} and the caches
# $B4C_CACHE/<cell>-{base,pr}. Fails unless every pass exited 0 and its
# benchmark.json reports the planned "text_cache"; with rule "fp8", every
# miss pass's "text_encoder" contains fp8 and every hit pass's is "cache";
# base and pr are byte-identical (--off-identity) on both passes; both
# caches hold entries, the same names and bytes, and no temporary file.
# Prints the verdict line.
b4c_verdict() {
  local cell="$1" rule="$2" arm pass c state enc verdict=pass why=""
  for arm in base pr; do
    for pass in miss hit; do
      c="$cell-$arm-$pass"
      b4_exit_ok "$c" || why+=" $c:exit"
      state="$(b4_field "$RUNS/$c/benchmark.json" text_cache)"
      [[ "$state" == "$pass" ]] || why+=" $c:text_cache=${state:-none}"
      if [[ "$rule" == fp8 ]]; then
        enc="$(b4_field "$RUNS/$c/benchmark.json" text_encoder)"
        case "$pass" in
          miss) [[ "$enc" == *fp8* ]] || why+=" $c:text_encoder=${enc:-none}" ;;
          hit) [[ "$enc" == cache ]] || why+=" $c:text_encoder=${enc:-none}" ;;
        esac
      fi
    done
    b4c_entries "$B4C_CACHE/$cell-$arm" >"$RUNS/$cell.cache-$arm.txt"
    b4c_cache_clean "$B4C_CACHE/$cell-$arm" || why+=" $arm:temporary-files"
  done
  for pass in miss hit; do
    b4_identical "$cell-base-$pass" "$cell-pr-$pass" || why+=" $pass:not-identical"
  done
  if [[ ! -s "$RUNS/$cell.cache-base.txt" || ! -s "$RUNS/$cell.cache-pr.txt" ]]; then
    why+=" no-cache-entries"
  elif ! diff "$RUNS/$cell.cache-base.txt" "$RUNS/$cell.cache-pr.txt" >"$RUNS/$cell.cache.diff"; then
    why+=" cache-entries-differ"
  fi
  [[ -n "$why" ]] && verdict=fail
  printf '{"cell":"%s","verdict":"%s","why":"%s"}\n' "$cell" "$verdict" "${why# }" \
    >"$RUNS/b4c-$cell.verdict.json"
  printf 'b4c %s: %s%s\n' "$cell" "$verdict" "${why:+ (${why# })}"
}

# b4b_trace <cell>: the last "/gpu_trace " report of <cell>'s stderr.log (a
# CUPTI trace of one denoise step, rt/gpu_trace.rs) on one line; empty when
# none was logged.
b4b_trace() {
  grep -h '/gpu_trace ' "$RUNS/$1/stderr.log" 2>/dev/null | tail -1
}

# b4b_number <report> <key>: the integer value of "key" in the report.
b4b_number() {
  printf '%s' "$1" | grep -oE "\"$2\": *[0-9]+" | head -1 | grep -oE '[0-9]+$' || true
}

# b4b_kernels <report>: one "name calls" line per kernel of its
# "top_kernels" table, in name order.
b4b_kernels() {
  local obj name calls
  printf '%s' "$1" | grep -oE '"top_kernels":\[[^]]*\]' | head -1 | grep -oE '\{[^{}]*\}' |
    while IFS= read -r obj; do
      name="$(printf '%s' "$obj" | grep -oE '"name": *"[^"]*"' | sed -E 's/.*"([^"]*)"$/\1/')"
      calls="$(printf '%s' "$obj" | grep -oE '"calls": *[0-9]+' | grep -oE '[0-9]+$')"
      [[ -n "$name" ]] && printf '%s %s\n' "$name" "${calls:-0}"
    done | LC_ALL=C sort
}

# b4b_evidence <cell>: <cell>/fusion-evidence.log, the fused kernels of the
# traced step (fvf_* / h3_* / wan_*) and the bf16 activation kernels the
# unfused chains run (mx_*), "name calls" in name order; and
# <cell>/trace-meta.txt: "kernel_count dropped_records listed_calls" (the
# step's kernel launches, CUPTI's dropped records, the calls the table
# lists; "-" for a field the report lacks). Returns 1 when no report with a kernel table was logged (tracing
# off, libcupti missing, the run died first).
b4b_evidence() {
  local report all
  : >"$RUNS/$1/fusion-evidence.log" 2>/dev/null || return 1
  : >"$RUNS/$1/trace-meta.txt"
  report="$(b4b_trace "$1")"
  [[ "$report" == *'"top_kernels":['* ]] || return 1
  all="$(b4b_kernels "$report")"
  printf '%s\n' "$all" | grep -E '^(fvf_|h3_|wan_|mx_)' >"$RUNS/$1/fusion-evidence.log" || true
  local kc dr
  kc="$(b4b_number "$report" kernel_count)"
  dr="$(b4b_number "$report" dropped_records)"
  printf '%s %s %s\n' "${kc:--}" "${dr:--}" \
    "$(printf '%s\n' "$all" | awk '{ s += $2 } END { print s + 0 }')" >"$RUNS/$1/trace-meta.txt"
  return 0
}

# b4b_gated <evidence file>: the switch-gated fused kernels in it (every
# fvf_* kernel but fvf_split_heads_rows, which FASTVIDEO_SPLIT_ROWS drives).
b4b_gated() {
  grep -E '^fvf_' "$1" 2>/dev/null | grep -v '^fvf_split_heads_rows ' || true
}

# b4b_calls <evidence file> <kernel>: its calls (0 when not listed).
b4b_calls() {
  awk -v k="$2" '$1 == k { print $2; found = 1 } END { if (!found) print 0 }' "$1" 2>/dev/null
}

# b4b_relations <evidence file> <rule> <relation set>: why the call counts
# break the block structure, or nothing. "h3": one block runs one
# h3_res_gate_norm_mod; fused (on), its first norm is h3_norm_mod (block 0)
# or the previous block's deferred fvf_h3_gate_res_norm_mod, and its last
# residual is deferred into that kernel or (last block) h3_gate_residual;
# unfused (off), every block runs h3_norm_mod and h3_gate_residual.
b4b_relations() {
  local f="$1" rule="$2" rel="$3" nm rg gr fg
  [[ "$rel" == h3 ]] || return 0
  nm="$(b4b_calls "$f" h3_norm_mod)"
  rg="$(b4b_calls "$f" h3_res_gate_norm_mod)"
  gr="$(b4b_calls "$f" h3_gate_residual)"
  fg="$(b4b_calls "$f" fvf_h3_gate_res_norm_mod)"
  (( rg > 0 )) || { printf ' relation:no-blocks'; return 0; }
  case "$rule" in
    on)
      (( nm + fg == rg )) || printf ' relation:norm%s+fused%s!=blocks%s' "$nm" "$fg" "$rg"
      (( gr + fg == rg )) || printf ' relation:residual%s+fused%s!=blocks%s' "$gr" "$fg" "$rg"
      ;;
    off)
      (( nm == rg && gr == rg )) || printf ' relation:norm%s,residual%s!=blocks%s' "$nm" "$gr" "$rg"
      ;;
  esac
}

# b4b_verdict <cell> <rule on|off> <relations h3|-> <kernel...>:
# $RUNS/b4b-<cell>.verdict.json from the arms <cell>-base (the baseline
# binary, whose trace lists its 25 slowest kernels) and <cell>-pr (this
# build, FASTVIDEO_GPU_TRACE_TOP raised: every kernel of the step). Fails
# unless:
#   both arms exited 0 and --off-identity passed (not skipped);
#   both arms logged a trace with a kernel table, no dropped record, and
#     the same kernel_count (the step's launches);
#   the PR arm's table is complete (its calls sum to kernel_count);
#   every fused / bf16-chain kernel the base arm lists is in the PR arm
#     with the same calls;
#   rule on: every named kernel ran in the PR arm (and so, by the line
#     above, in the base arm wherever it lists it); rule off: no
#     switch-gated kernel ran in the PR arm and every named (unfused-path)
#     kernel did;
#   the PR arm's calls keep the block structure (b4b_relations).
# Prints the verdict line.
b4b_verdict() {
  local cell="$1" rule="$2" rel="$3" arm c k kc dr listed base_kc="" pr_kc="" verdict=pass why=""
  shift 3
  case "$rule" in on | off) ;; *) why+=" rule:$rule" ;; esac
  for arm in base pr; do
    c="$cell-$arm"
    b4_exit_ok "$c" || why+=" $arm:exit"
    if ! b4b_evidence "$c"; then
      why+=" $arm:no-trace"
      continue
    fi
    read -r kc dr listed <"$RUNS/$c/trace-meta.txt"
    [[ "$kc" != - ]] || why+=" $arm:no-kernel-count"
    [[ "$dr" == 0 ]] || why+=" $arm:dropped-records=$([[ "$dr" == - ]] && echo none || echo "$dr")"
    [[ "$arm" == base ]] && base_kc="$kc" || pr_kc="$kc"
    if [[ "$arm" == pr && "$listed" != "$kc" ]]; then
      why+=" pr:incomplete-trace($listed/$kc)"
    fi
  done
  b4_identical "$cell-base" "$cell-pr" || why+=" not-identical"
  local base_ev="$RUNS/$cell-base/fusion-evidence.log" pr_ev="$RUNS/$cell-pr/fusion-evidence.log"
  if [[ -n "$base_kc" && -n "$pr_kc" && "$base_kc" != - && "$base_kc" != "$pr_kc" ]]; then
    why+=" kernel-count-differs($base_kc/$pr_kc)"
  fi
  if [[ -s "$base_ev" && -f "$pr_ev" ]]; then
    LC_ALL=C comm -23 "$base_ev" "$pr_ev" >"$RUNS/$cell.fusion-evidence.diff"
    [[ -s "$RUNS/$cell.fusion-evidence.diff" ]] && why+=" evidence-differs"
  fi
  if [[ -f "$pr_ev" ]]; then
    case "$rule" in
      on)
        for k in "$@"; do
          (( $(b4b_calls "$pr_ev" "$k") > 0 )) || why+=" pr:missing:$k"
        done
        ;;
      off)
        [[ -z "$(b4b_gated "$pr_ev")" ]] || why+=" pr:fused-kernel-ran:$(b4b_gated "$pr_ev" | cut -d' ' -f1 | tr '\n' ',' | sed 's/,$//')"
        for k in "$@"; do
          (( $(b4b_calls "$pr_ev" "$k") > 0 )) || why+=" pr:missing:$k"
        done
        ;;
    esac
    why+="$(b4b_relations "$pr_ev" "$rule" "$rel")"
  fi
  [[ -n "$why" ]] && verdict=fail
  printf '{"cell":"%s","verdict":"%s","why":"%s"}\n' "$cell" "$verdict" "${why# }" \
    >"$RUNS/b4b-$cell.verdict.json"
  printf 'b4b %s: %s%s\n' "$cell" "$verdict" "${why:+ (${why# })}"
}

# b4c2_state_is <cell> <state...>: the pass's benchmark.json "text_cache"
# is one of the states; else appends " <cell>:text_cache=<got>" to $why.
b4c2_state_is() {
  local c="$1" state s
  shift
  state="$(b4_field "$RUNS/$c/benchmark.json" text_cache)"
  for s in "$@"; do
    [[ "$state" == "$s" ]] && return 0
  done
  why+=" $c:text_cache=${state:-none}"
}

# b4c2_count <dir>: how many files the cache directory holds.
b4c2_count() {
  find "$1" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' '
}

# b4c2_switch_verdict <cell>: $RUNS/b4c2-<cell>.verdict.json for a cell that
# writes the cache under one setting and reads it under another (ADR-0021
# "B4c2"): passes <cell>-{base,pr}-write (a miss that stores), then
# <cell>-{base,pr}-read over the same cache with the setting changed, and
# <cell>-pr-fresh (the read setting over its own new cache,
# $B4C_CACHE/<cell>-pr-fresh). Passes only when every pass exited 0; both
# writes missed; the baseline's read *hit* (the key gap is reproduced: main
# served the other setting's entry); this build's read missed and is
# byte-identical (--off-identity) to the fresh encode, which missed too; the
# baseline cache holds one entry, this build's two, the fresh cache one that
# is also in this build's (same name and bytes), and none a temporary file.
# Prints the verdict line.
b4c2_switch_verdict() {
  local cell="$1" c verdict=pass why=""
  for c in "$cell-base-write" "$cell-base-read" "$cell-pr-write" "$cell-pr-read" "$cell-pr-fresh"; do
    b4_exit_ok "$c" || why+=" $c:exit"
  done
  b4c2_state_is "$cell-base-write" miss
  b4c2_state_is "$cell-pr-write" miss
  b4c2_state_is "$cell-base-read" hit
  b4c2_state_is "$cell-pr-read" miss
  b4c2_state_is "$cell-pr-fresh" miss
  b4_identical "$cell-pr-fresh" "$cell-pr-read" || why+=" read-vs-fresh:not-identical"
  b4c_entries "$B4C_CACHE/$cell-base" >"$RUNS/$cell.cache-base.txt"
  b4c_entries "$B4C_CACHE/$cell-pr" >"$RUNS/$cell.cache-pr.txt"
  b4c_entries "$B4C_CACHE/$cell-pr-fresh" >"$RUNS/$cell.cache-pr-fresh.txt"
  [[ "$(b4c2_count "$B4C_CACHE/$cell-base")" == 1 ]] || why+=" base:entries=$(b4c2_count "$B4C_CACHE/$cell-base")"
  [[ "$(b4c2_count "$B4C_CACHE/$cell-pr")" == 2 ]] || why+=" pr:entries=$(b4c2_count "$B4C_CACHE/$cell-pr")"
  if [[ "$(b4c2_count "$B4C_CACHE/$cell-pr-fresh")" != 1 ]]; then
    why+=" pr-fresh:entries=$(b4c2_count "$B4C_CACHE/$cell-pr-fresh")"
  elif ! grep -qxF -f "$RUNS/$cell.cache-pr-fresh.txt" "$RUNS/$cell.cache-pr.txt"; then
    why+=" fresh-entry-not-stored-by-read"
  fi
  for c in base pr pr-fresh; do
    b4c_cache_clean "$B4C_CACHE/$cell-$c" || why+=" $c:temporary-files"
  done
  [[ -n "$why" ]] && verdict=fail
  printf '{"cell":"%s","verdict":"%s","why":"%s"}\n' "$cell" "$verdict" "${why# }" \
    >"$RUNS/b4c2-$cell.verdict.json"
  printf 'b4c2 %s: %s%s\n' "$cell" "$verdict" "${why:+ (${why# })}"
}

# b4c2_noop_verdict <cell>: $RUNS/b4c2-<cell>.verdict.json for the
# no-change cell: default settings, passes <cell>-{base,pr}-{miss,hit} over
# each arm's own new cache. Passes only when every pass exited 0 and reports
# the planned "text_cache"; base and pr are byte-identical (--off-identity)
# on the miss and on the hit; this build's hit is byte-identical to its miss;
# each cache holds exactly one entry, no temporary file, and the two entries
# have the same bytes under different names (the same conditioning under the
# B4c2 key). Prints the verdict line.
b4c2_noop_verdict() {
  local cell="$1" arm pass c verdict=pass why="" base_sum pr_sum base_name pr_name
  for arm in base pr; do
    for pass in miss hit; do
      c="$cell-$arm-$pass"
      b4_exit_ok "$c" || why+=" $c:exit"
      b4c2_state_is "$c" "$pass"
    done
    b4c_entries "$B4C_CACHE/$cell-$arm" >"$RUNS/$cell.cache-$arm.txt"
    [[ "$(b4c2_count "$B4C_CACHE/$cell-$arm")" == 1 ]] || why+=" $arm:entries=$(b4c2_count "$B4C_CACHE/$cell-$arm")"
    b4c_cache_clean "$B4C_CACHE/$cell-$arm" || why+=" $arm:temporary-files"
  done
  for pass in miss hit; do
    b4_identical "$cell-base-$pass" "$cell-pr-$pass" || why+=" $pass:not-identical"
  done
  b4_identical "$cell-pr-miss" "$cell-pr-hit" || why+=" pr-hit-vs-miss:not-identical"
  read -r base_sum base_name <"$RUNS/$cell.cache-base.txt" || true
  read -r pr_sum pr_name <"$RUNS/$cell.cache-pr.txt" || true
  if [[ -z "${base_sum:-}" || -z "${pr_sum:-}" ]]; then
    why+=" no-cache-entries"
  else
    [[ "$base_sum" == "$pr_sum" ]] || why+=" entry-bytes-differ"
    [[ "$base_name" != "$pr_name" ]] || why+=" entry-name-unchanged"
  fi
  [[ -n "$why" ]] && verdict=fail
  printf '{"cell":"%s","verdict":"%s","why":"%s"}\n' "$cell" "$verdict" "${why# }" \
    >"$RUNS/b4c2-$cell.verdict.json"
  printf 'b4c2 %s: %s%s\n' "$cell" "$verdict" "${why:+ (${why# })}"
}

# b4_family_verdict <family> <cell...>: $RUNS/<family>.verdict.json, a pass
# only when every named cell wrote a passing <family>-<cell>.verdict.json
# (a name that matches no cell, or a cell that never reached its verdict,
# fails). Prints the verdict line.
b4_family_verdict() {
  local fam="$1" cell f verdict=pass why=""
  shift
  (( $# > 0 )) || why=" no-cells"
  for cell in "$@"; do
    f="$RUNS/$fam-$cell.verdict.json"
    if [[ ! -f "$f" ]]; then
      why+=" $cell:no-verdict"
    elif [[ "$(b4_flat "$f")" != *'"verdict":"pass"'* ]]; then
      why+=" $cell:fail"
    fi
  done
  [[ -n "$why" ]] && verdict=fail
  printf '{"family":"%s","verdict":"%s","cells":"%s","why":"%s"}\n' "$fam" "$verdict" "$*" "${why# }" \
    >"$RUNS/$fam.verdict.json"
  printf '%s family: %s%s\n' "$fam" "$verdict" "${why:+ (${why# })}"
}
