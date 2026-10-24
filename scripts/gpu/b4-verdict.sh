#!/usr/bin/env bash
# Verdicts of the plan B4 byte-identity families of runpod-matrix.sh (b4a,
# b4c), sourced by it. Pure functions over a run directory ($RUNS) and, for
# b4c, the cache root ($B4C_CACHE), so scripts/gpu/tests/b4-verdict.test.sh
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
