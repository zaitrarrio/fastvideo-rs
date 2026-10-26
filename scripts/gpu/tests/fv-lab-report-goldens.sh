#!/usr/bin/env bash
# The goldens of `fv-lab report summary` and `report table` (docs/adr/0019-fv-lab.md
# §13): what the report scripts they replace print, run on committed run
# directories, frozen as fixtures. crates/fastvideo-lab/tests/report.rs holds
# fv-lab to them line for line.
#
#   fv-lab-report-goldens.sh             regenerate crates/fastvideo-lab/fixtures/report/
#   fv-lab-report-goldens.sh --check     fail when a script's output differs from its fixture
#   fv-lab-report-goldens.sh layout trim|fasth3 <dest>
#                                        an archived experiment's files in the layout its
#                                        summary reads (the Rust tests build theirs with it)
#
# Fixtures (crates/fastvideo-lab/fixtures/report/):
#   summary/<family>--<tag>.txt  summary-script.sh <family> artifacts/runpod/<family>/<tag>: the
#                                lines the retired families' runpod-matrix.sh branches
#                                copied into live.log (frozen from 5987e6a4); for the
#                                script-side runs in LIVE, every line is also checked to be
#                                in that run's own live.log, and (but for PARTIAL) the lines of
#                                live.log shaped as that family's summary lines are exactly the
#                                fixture's, in order
#   hd--<tag>.txt                scripts/gpu/hd_report.py artifacts/runpod/hd/<tag>, with numpy
#                                and Pillow stood in by empty modules (the committed runs hold no
#                                keyframes, so neither is called; fv-lab reads no PNG)
#   hd-synthetic.txt             hd_report.py on fixtures/report/hd-run (mp4s for `## streams`, a
#                                hidden directory, a failing probe) with fixtures/report/bin/ffprobe,
#                                a stand-in answering from the file name, first on PATH
#   h3-graphs.txt                experiments/h3-graphs-summary.py artifacts/perf/h3-denoise-graphs/run-2026-10-14
#   h3-tier-options.{md,json}    experiments/h3-tier-options-summary.py artifacts/perf/h3-nvfp4-tier-options --json
#   h3-trim-nvfp4.{md,json}      experiments/h3-trim-nvfp4-summary.py on `layout trim`
#   fasth3-nvfp4.{md,json}       experiments/fasth3-nvfp4-summary.py on `layout fasth3`
#   sol-bench--<set>.{md,results.json}
#                                sol_bench_table.py on a copy of artifacts/runpod/sol-bench/<set>/*/
#                                (it writes results.json beside them), and on
#                                fixtures/report/sol-bench-cells/{a,missing,b/} (`cells`: every
#                                PUBLISHED row, the LingBot / Cosmos3 paths, -s15, -rs3)
#   timings.md                   timings.sh on a copy of fixtures/report/timings-run
#
# Retired (ADR-0019 §13.5): sol_bench_table.py and the four experiments'
# *-summary.py had no caller but docs and are deleted; fv-lab is their only
# implementation. Their fixtures are frozen: this script runs each from
# $RETIRED_AT (the last commit with them) when the history holds it (a full
# clone), and otherwise keeps their fixtures as they are and says so (CI's
# shallow clone). hd_report.py (its keyframe sharpness is not ported) and
# timings.sh (validate.sh calls it) stay and always run.
#
# Python runs as `python3 -I` (isolated: no user site, no PYTHONPATH), stdlib
# only; every script under `env -i` with LC_ALL=C. No network, no GPU.
# timings.sh's fixture holds only numbers jq 1.6 and 1.7 print alike (1.7
# keeps an input number's spelling, 1.6 prints its shortest form).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="$ROOT/crates/fastvideo-lab/fixtures/report"
export LC_ALL=C

# layout <trim|fasth3> <dest>: an archived run's files where its summary reads them.
layout() {
  local what="$1" d="$2" a c ck arm
  mkdir -p "$d"
  case "$what" in
    trim)
      a="$ROOT/artifacts/ports/h3-trim/nvfp4"
      # runs/<cell>/benchmark.json from bench/<cell>.json; the gate reports;
      # no compare-clips reports were archived (an empty runs/compare).
      for c in "$a"/bench/*.json; do
        mkdir -p "$d/runs/$(basename "$c" .json)"
        cp "$c" "$d/runs/$(basename "$c" .json)/benchmark.json"
      done
      mkdir -p "$d/runs/gate" "$d/runs/compare"
      cp "$a"/gate/*.json "$d/runs/gate/"
      cp "$a"/compare-dumps-*.txt "$d/"
      ;;
    fasth3)
      a="$ROOT/artifacts/ports/fasth3-nvfp4/run-2026-10-14"
      for ck in v2 4s; do
        mkdir -p "$d/up-$ck" "$d/ours-$ck/compare"
        cp "$a/upstream/$ck-result.json" "$d/up-$ck/result.json"
        for arm in nvfp4 nvfp4ffn-mxfp8; do
          mkdir -p "$d/ours-$ck/$arm"
          cp "$a/ours/$ck-$arm-benchmark.json" "$d/ours-$ck/$arm/benchmark.json"
        done
        cp "$a/ours/$ck-compare-clips-vs-upstream.json" "$d/ours-$ck/compare/compare-clips-vs-upstream.json"
        cp "$a/compare-dumps-$ck.txt" "$a/av-psnr-$ck.json" "$d/"
      done
      ;;
    *) echo "layout: trim or fasth3" >&2; return 2 ;;
  esac
}

if [[ "${1:-}" == layout ]]; then
  layout "${2:?layout trim|fasth3 <dest>}" "${3:?layout <what> <dest>}"
  exit 0
fi
CHECK=0
[[ "${1:-}" == --check ]] && CHECK=1
command -v python3 >/dev/null || { echo "fv-lab-report-goldens: needs python3" >&2; exit 2; }
command -v jq >/dev/null || { echo "fv-lab-report-goldens: needs jq" >&2; exit 2; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
GEN="$TMP/report"
mkdir -p "$GEN/summary"
RETIRED_AT=f9e57ea2
FROZEN=0
# retired <repo path>: the retired script's text from $RETIRED_AT, into $TMP;
# prints its path, or fails when the history lacks it.
retired() {
  local p="$TMP/retired/$1"
  [[ -f "$p" ]] && { echo "$p"; return 0; }
  git -C "$ROOT" cat-file -e "$RETIRED_AT:$1" 2>/dev/null || return 1
  mkdir -p "$(dirname "$p")"
  git -C "$ROOT" show "$RETIRED_AT:$1" >"$p"
  echo "$p"
}
# keep <fixture>...: no script to run (shallow history): the frozen fixture stands.
keep() {
  local f
  for f in "$@"; do
    [[ -f "$OUT/$f" ]] || { echo "FAIL: frozen fixture $f is missing" >&2; fail=1; continue; }
    cp "$OUT/$f" "$GEN/$f"
    FROZEN=$((FROZEN + 1))
  done
}
clean() { env -i PATH="/usr/local/bin:/usr/bin:/bin" HOME="$TMP" LC_ALL=C "$@"; }
py() { clean python3 -I "$@"; }

# 1. The retired families' live.log summary lines.
SUMMARY_RUNS="
techniques/1de80f1-10240618 techniques/51898b8-10240107 techniques/90dd3f8-10240259 techniques/93696e5-10240625
techniques/a2ca568-10241022 techniques/c21d801-10241607 techniques/c21d801-1024155754-31bc
b6/a2ca568-10241022 b6/a2ca568-10241101 b6/a2ca568-10241146 b6/a2ca568-10241147 b6/a2ca568-10241148
b6/a2ca568-10241149 b6/c21d801-10241608 b6/c21d801-1024155958-468a
b4a/abccc38-10241423 b4a/abccc38-10241424 b4a/abccc38-10241425 b4a/e82b6b7-10252113 b4a/e82b6b7-1025211328-a2f3
h3arms/b0d0bb8-10251649 h3arms/b0d0bb8-1025164622-3ab2
trace/b0d0bb8-10251650 trace/b0d0bb8-1025164725-29e7
ltxfps/3df276d-09272000 ltxfps/b0d0bb8-10251654 ltxfps/b0d0bb8-1025165407-831d
ltxvae/5987e6a-10252157 ltxvae/b0d0bb8-10251657 ltxvae/b0d0bb8-1025165621-b3c7
ltxvae-gen/8b18d89-1025215640-5305"
# Script-side runs of the script version frozen here: their live.log holds every line.
# (51898b8-10240107, 90dd3f8-10240259 and a2ca568-10241101 ran earlier versions of
# the summaries; the fv-lab runs, tagged -<MMDDHHMMSS>-<hex>, print none.)
LIVE="
techniques/1de80f1-10240618 techniques/93696e5-10240625 techniques/a2ca568-10241022 techniques/c21d801-10241607
b6/a2ca568-10241146 b6/a2ca568-10241147 b6/a2ca568-10241148 b6/a2ca568-10241149 b6/c21d801-10241608
b4a/abccc38-10241423 b4a/abccc38-10241424 b4a/abccc38-10241425 b4a/e82b6b7-10252113
h3arms/b0d0bb8-10251649 trace/b0d0bb8-10251650 ltxfps/b0d0bb8-10251654
ltxvae/5987e6a-10252157 ltxvae/b0d0bb8-10251657"
# The lines of a family's summary, told apart in live.log by their shape (the
# reverse check: live.log holds no summary line the fixture lacks).
summary_re() {
  case "$1" in
    techniques) echo '^compare-clips-[^ ]*\.json: |^[a-z0-9.-]+-(base|env|prof) ("|$)' ;;
    b6) echo '^\[b6-[a-z0-9-]+-(base|pr)\] |^compare-clips-b6-[^ ]*\.json: ' ;;
    b4a) echo '^compare-clips-b4a-[^ ]*\.json: ' ;;
    h3arms) echo '^\[fasth3-8step-[0-9]+p-(sol|tea|soltea)\] teacache reused steps |^gate-[^ ]*\.json: ' ;;
    trace) echo '^\[[a-z0-9-]+\] .*/gpu_trace ' ;;
    ltxfps) echo '^\[(ltx25-1080p|fasth3-8step-sink)[^]]*\] ' ;;
    ltxvae | ltxvae-gen) echo '^\[(ltxvae|ltx25-[^]]*-gen)[^]]*\] ' ;;
  esac
}
# Committed with only live.log and compare/ (the arm directories were not kept):
# the forward check holds, the reverse one cannot.
PARTIAL="techniques/1de80f1-10240618 techniques/93696e5-10240625"
fail=0
for r in $SUMMARY_RUNS; do
  fam="${r%%/*}" tag="${r#*/}"
  o="$GEN/summary/$fam--$tag.txt"
  clean bash "$OUT/summary-script.sh" "$fam" "$ROOT/artifacts/runpod/$r" "$o"
  if [[ " $(echo $LIVE) " == *" $r "* ]]; then
    sed -E 's/^\[[0-9:]{8}\] \[[a-z0-9-]+\] //' "$ROOT/artifacts/runpod/$r/live.log" >"$TMP/live.txt"
    while IFS= read -r l; do
      grep -qxF -- "$l" "$TMP/live.txt" || { echo "FAIL: $r: not in its live.log: $l" >&2; fail=1; }
    done <"$o"
    if [[ " $PARTIAL " != *" $r "* ]] && ! diff <(grep -aE "$(summary_re "$fam")" "$TMP/live.txt") "$o" >"$TMP/rev.diff"; then
      echo "FAIL: $r: live.log's summary lines are not the fixture's:" >&2
      head -5 "$TMP/rev.diff" >&2
      fail=1
    fi
  fi
done

# The frozen lines are the script's (when the history holds 5987e6a4).
if git -C "$ROOT" cat-file -e 5987e6a4:scripts/gpu/runpod-matrix.sh 2>/dev/null; then
  git -C "$ROOT" show 5987e6a4:scripts/gpu/runpod-matrix.sh >"$TMP/main.sh"
  cp "$ROOT/crates/fastvideo-lab/fixtures/b4a-script/runpod-matrix.sh" "$TMP/b4a.sh"
  awk -v tmp="$TMP" '
    /^# --- verbatim / { src = $4; split($5, r, "-"); n = 0; inb = 1; next }
    /^# --- end$/ { inb = 0; f = tmp "/frozen-" (++k) ".txt"; printf "%s %d %d\n", src, r[1], r[2] > (tmp "/blocks"); next }
    inb { print > (tmp "/frozen-" (k + 1) ".txt") }
  ' "$OUT/summary-script.sh"
  k=0
  while read -r src a b; do
    k=$((k + 1))
    if ! diff <(sed -n "${a},${b}p" "$TMP/$src.sh") "$TMP/frozen-$k.txt" >/dev/null; then
      echo "FAIL: summary-script.sh block $k is not $src lines $a-$b" >&2
      fail=1
    fi
  done <"$TMP/blocks"
else
  echo "fv-lab-report-goldens: 5987e6a4 not in the history (shallow clone): the frozen lines are not compared with it" >&2
fi

# 2. hd_report.py, numpy and Pillow stood in by empty modules.
HD_STUBS='
import runpy, sys, types
sys.modules["numpy"] = types.ModuleType("numpy")
pil = types.ModuleType("PIL")
pil.Image = types.ModuleType("PIL.Image")
sys.modules["PIL"], sys.modules["PIL.Image"] = pil, pil.Image
sys.argv = [sys.argv[1], sys.argv[2]]
runpy.run_path(sys.argv[0], run_name="__main__")
'
for d in "$ROOT"/artifacts/runpod/hd/*/; do
  t="$(basename "$d")"
  py -c "$HD_STUBS" "$ROOT/scripts/gpu/hd_report.py" "${d%/}" >"$GEN/hd--$t.txt"
done
env -i PATH="$OUT/bin:/usr/local/bin:/usr/bin:/bin" HOME="$TMP" LC_ALL=C python3 -I -c "$HD_STUBS" \
  "$ROOT/scripts/gpu/hd_report.py" "$OUT/hd-run" >"$GEN/hd-synthetic.txt"

# 3. The experiment summaries (retired).
X=scripts/gpu/experiments
if s="$(retired "$X/h3-graphs-summary.py")"; then
  py "$s" "$ROOT/artifacts/perf/h3-denoise-graphs/run-2026-10-14" >"$GEN/h3-graphs.txt"
else keep h3-graphs.txt; fi
if s="$(retired "$X/h3-tier-options-summary.py")"; then
  py "$s" "$ROOT/artifacts/perf/h3-nvfp4-tier-options" --json "$GEN/h3-tier-options.json" >"$GEN/h3-tier-options.md"
else keep h3-tier-options.md h3-tier-options.json; fi
if s="$(retired "$X/h3-trim-nvfp4-summary.py")"; then
  layout trim "$TMP/trim"
  py "$s" "$TMP/trim" --json "$GEN/h3-trim-nvfp4.json" >"$GEN/h3-trim-nvfp4.md"
else keep h3-trim-nvfp4.md h3-trim-nvfp4.json; fi
if s="$(retired "$X/fasth3-nvfp4-summary.py")"; then
  layout fasth3 "$TMP/fasth3"
  py "$s" "$TMP/fasth3" --json "$GEN/fasth3-nvfp4.json" >"$GEN/fasth3-nvfp4.md"
else keep fasth3-nvfp4.md fasth3-nvfp4.json; fi

# 4. sol_bench_table.py (retired; it writes results.json beside the run directories: copies).
SETS="2026-10-06 2026-10-06-phaseB 2026-10-06-phaseB2"
if s="$(retired scripts/gpu/sol_bench_table.py)"; then
  for set in $SETS; do
    mkdir -p "$TMP/sb/$set"
    for d in "$ROOT/artifacts/runpod/sol-bench/$set"/*/; do cp -r "$d" "$TMP/sb/$set/"; done
    (cd "$TMP/sb/$set" && py "$s" */) >"$GEN/sol-bench--$set.md"
    cp "$TMP/sb/$set/results.json" "$GEN/sol-bench--$set.results.json"
  done
  cp -r "$OUT/sol-bench-cells" "$TMP/sb/cells"
  (cd "$TMP/sb/cells" && py "$s" a missing b/) >"$GEN/sol-bench--cells.md"
  cp "$TMP/sb/cells/results.json" "$GEN/sol-bench--cells.results.json"
else
  for set in $SETS cells; do keep "sol-bench--$set.md" "sol-bench--$set.results.json"; done
fi
(( FROZEN == 0 )) || echo "fv-lab-report-goldens: $RETIRED_AT not in the history (shallow clone): $FROZEN fixtures of retired scripts kept as frozen, not regenerated" >&2

# 5. timings.sh (it writes timings.md into the run directory: a copy).
cp -r "$OUT/timings-run" "$TMP/timings-run"
clean bash "$ROOT/scripts/gpu/timings.sh" "$TMP/timings-run" >/dev/null
cp "$TMP/timings-run/timings.md" "$GEN/timings.md"

# The generated set against the fixtures (the inputs fixtures/report/
# holds, summary-script.sh, sol-bench-cells/ and timings-run/, are not outputs).
outputs() { (cd "$1" && { find ./summary -maxdepth 1 -type f 2>/dev/null || true
  find . -maxdepth 1 -type f \( -name 'hd--*' -o -name 'hd-synthetic.txt' -o -name 'h3-*' -o -name 'fasth3-*' \
    -o -name 'sol-bench--*' -o -name 'timings.md' \); } | sort); }
if (( CHECK )); then
  if ! diff <(outputs "$GEN") <(outputs "$OUT") >"$TMP/list.diff"; then
    echo "FAIL: the fixture set differs from what the scripts write:" >&2
    cat "$TMP/list.diff" >&2
    fail=1
  fi
  while read -r f; do
    [[ -f "$OUT/$f" ]] || continue
    if ! cmp -s "$GEN/$f" "$OUT/$f"; then
      echo "FAIL: $f differs from the script's output:" >&2
      diff "$OUT/$f" "$GEN/$f" | head -20 >&2 || true
      fail=1
    fi
  done < <(outputs "$GEN")
  (( fail == 0 )) && echo "fv-lab-report-goldens: $(outputs "$GEN" | wc -l) fixtures are the scripts' output ($FROZEN of them frozen, not regenerated)"
  exit "$fail"
fi
(( fail == 0 )) || exit 1
rm -rf "$OUT/summary"
outputs "$OUT" | while read -r f; do rm -f "$OUT/$f"; done
cp -r "$GEN/." "$OUT/"
echo "fv-lab-report-goldens: wrote $(outputs "$OUT" | wc -l) fixtures under ${OUT#"$ROOT"/}"
