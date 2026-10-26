#!/usr/bin/env bash
# The GPU gate of D1a / D1b (docs/adr/0019-fv-lab.md §8): does an `fv-lab pod
# up` run reproduce a `runpod-http.sh run` of the same commit? Both run
# directories are what the fetches wrote under artifacts/runpod/<family>/<tag>/.
#
#   fv-lab-compare-runs.sh [--expect-verdicts N] [--cells DIR] <fv-lab run dir> <runpod-http run dir>
#
# Same when, for every arm directory of either run (the directories holding a
# summary.json):
#   - the arm exists in both;
#   - summary.json agrees on `exit` and `skipped` (seconds and dates differ);
#   - frames.sha256.json is byte-identical (both runners write it after an arm:
#     every frame's SHA-256; docs/perf/determinism.md promises the same bytes
#     for the same request and seed on the same GPU class), or absent in both;
#   - benchmark.json is present in both or in neither;
# and for every compare / gate report (compare/compare-clips-*.json,
# gate/gate-*.json, D1b's families) of either run:
#   - the report exists in both;
#   - a compare report's `status` and off-identity status
#     (context.off_identity.status: the byte-identity verdict of a
#     --off-identity compare) agree: both follow from the frames, which are
#     byte-identical on one GPU class. A gate's verdict also weighs the two
#     arms' timings, which differ from pod to pod, so a gate report is only
#     required in both runs;
# and for every file of the verdict step (D1d, ADR-0019 §11: the run's
# *.verdict.json and *.diff, each arm's *-evidence.log; `b4a`) of either run:
#   - the file exists in both and is byte-identical (no timings in them),
#     except a family verdict (<family>.verdict.json), whose `cells` and `why`
#     are compared as sets: fv-lab lists the cells in the tier's order, the
#     script in FV_B4A_CELLS's (ADR-0019 §11.5; a script run of b4a from
#     before its branch was deleted, §11.6).
# and for every file of the determinism families' hashes (D1d, ADR-0019 §15:
# the run's determinism.txt and hashes.txt, each arm's hashes.txt and
# hashes.json; `det-short`, `determinism`) of either run:
#   - the file exists in both and is byte-identical (pure functions of the
#     frames; an arm's digests.txt is not compared: it is read from stderr);
# and for every arm whose stderr.log holds `compare-dumps` lines (the
# `oracle` tier's tensor diffs against one upstream reference) in either run:
#   - those lines are the same in both (they follow from the dumps, which
#     follow from the frames' computation; timings are on other lines).
# and for every derived-clip report (ADR-0019 §20: `<dir>/metrics.txt`, the
# `oracle` tier's clip-metric targets, `oracle-<target>-frames/`) of either run:
#   - the report exists in both and is byte-identical; or, when only the
#     values differ (each line cut at its last `: ` is the same in both: the
#     same clips, measures and order), it is reported (`metrics <file>: the
#     same measurements, values differ`) and is no difference: the numbers
#     read the clips' mp4 encodes, which the frames records do not cover
#     (§20.6). Any other change (a line missing, another order) is one.
# (`fv-lab pod fetch` takes these; on the runpod-http.sh side they need
# FV_FETCH_TREE=1.) --expect-verdicts N fails unless at least N verdict files
# compared the same: a tier with verdict cells whose files never arrived
# would otherwise pass with none.
# Control arms (ADR-0019 §15.7): an arm a cell lists in `[repeats] controls`
# is reported, not judged, as fv-lab's verdict does (repeats.rs judges only
# `judged`): the timed-pick control (FASTVIDEO_SDPA_AUTO=timed) is not
# reproducible across processes (docs/perf/determinism.md): it changes the
# frames' bytes, not their number. Such an arm must still exist in both runs,
# agree on summary.json `exit` / `skipped` and benchmark.json's presence, and
# have a readable frames.sha256.json with the same frame count in both or in
# neither; but its frames' bytes, compare-dumps lines and per-arm verdict
# files (hashes.txt, hashes.json, *-evidence.log) are reported on a
# `control <arm>:` line and are neither differences nor verdict files (they
# count neither toward nor against --expect-verdicts). The run's
# determinism.txt and hashes.txt are compared without the control arms' lines
# (a line whose first field is `<arm>`, `<arm>:` or `<arm>/...`) and still
# count, unless nothing but control lines is left in both (then neither
# compared nor counted); an unreadable one is a difference. The controls
# come from the cells (both runs are of the checkout's commit): every name in
# a `[repeats] controls` of DIR/*.toml (--cells, default the checkout's
# lab/cells), less any name another cell declares as an `[[arm]]` it does not
# list as a control (that arm stays judged); a name outside fv-lab's arm-name
# grammar is a usage error (exit 2). A control arm does not count toward the
# arms that ran.
# Exit 0 when the runs agree and at least one arm ran (exit 0) with
# byte-identical frames in both; 1 otherwise (one line per difference). An
# unreadable summary, frames record or report is a difference; arms skipped or
# failed alike in both runs are counted, not evidence.
# Needs bash and jq.
set -euo pipefail
USAGE="usage: fv-lab-compare-runs.sh [--expect-verdicts N] [--cells DIR] <fv-lab run dir> <runpod-http run dir>"
EXPECT_VERDICTS=0
CELLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/lab/cells"
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --expect-verdicts) [[ "${2:-}" =~ ^[0-9]+$ ]] || { echo "$USAGE" >&2; exit 2; }; EXPECT_VERDICTS="$2" ;;
    --cells) [[ -n "${2:-}" ]] || { echo "$USAGE" >&2; exit 2; }; CELLS_DIR="$2" ;;
    *) echo "$USAGE" >&2; exit 2 ;;
  esac
  shift 2
done
A="${1:?$USAGE}"
B="${2:?$USAGE}"
[[ -d "$A" && -d "$B" ]] || { echo "fv-lab-compare-runs: not a directory: $A or $B" >&2; exit 2; }

# The control arms, from the cells' `[repeats] controls` (see above). A line
# reader for the cell files' shape (fv-lab's parser is the strict one): table
# headers, `name = "..."` in an `[[arm]]`, and a `controls = [...]` array of
# plain strings in `[repeats]`, on one line or several.
[[ -d "$CELLS_DIR" ]] || { echo "fv-lab-compare-runs: no cells directory $CELLS_DIR (--cells)" >&2; exit 2; }
# fv-lab's arm-name grammar (plan.rs is_name): [a-z0-9][a-z0-9._-]*, at most 64 bytes.
NAME_RE='^[a-z0-9][a-z0-9._-]*$'
name_ok() { [[ "$1" =~ $NAME_RE ]] && (( ${#1} <= 64 )); }
in_list() { local x="$1" y; shift; for y in "$@"; do [[ "$y" == "$x" ]] && return 0; done; return 1; }
LISTED=() JUDGED=()
for f in "$CELLS_DIR"/*.toml; do
  [[ -f "$f" ]] || { echo "fv-lab-compare-runs: no cell files in $CELLS_DIR" >&2; exit 2; }
  table="" acc="" in_ctl=0 ARMS_F=() CTL_F=()
  while IFS= read -r l || [[ -n "$l" ]]; do
    if (( in_ctl )); then
      acc+=" ${l%%#*}"
    elif [[ "$l" =~ ^[[:space:]]*\[\[?[A-Za-z0-9_.-]+\]\]?[[:space:]]*(#.*)?$ ]]; then
      table="${l%%#*}"; table="${table//[[:space:]]/}"
      continue
    elif [[ "$table" == '[[arm]]' && "$l" =~ ^[[:space:]]*name[[:space:]]*=[[:space:]]*\"([^\"]*)\" ]]; then
      n="${BASH_REMATCH[1]}"
      name_ok "$n" || { echo "fv-lab-compare-runs: $f: arm \`$n\` is not an arm name" >&2; exit 2; }
      ARMS_F+=("$n")
      continue
    elif [[ "$table" == '[repeats]' && "$l" =~ ^[[:space:]]*controls[[:space:]]*= ]]; then
      acc="${l#*=}"; acc="${acc%%#*}"; in_ctl=1
    else
      continue
    fi
    [[ "$acc" == *']'* ]] || continue
    in_ctl=0
    acc="${acc%%]*}"; acc="${acc#*\[}"
    while [[ "$acc" =~ ^[[:space:],]*\"([^\"]*)\"(.*)$ ]]; do
      n="${BASH_REMATCH[1]}"; acc="${BASH_REMATCH[2]}"
      name_ok "$n" || { echo "fv-lab-compare-runs: $f: control \`$n\` is not an arm name" >&2; exit 2; }
      CTL_F+=("$n")
    done
    [[ "$acc" =~ ^[[:space:],]*$ ]] || { echo "fv-lab-compare-runs: $f: unreadable [repeats] controls" >&2; exit 2; }
  done <"$f"
  (( in_ctl == 0 )) || { echo "fv-lab-compare-runs: $f: [repeats] controls has no closing ]" >&2; exit 2; }
  LISTED+=(${CTL_F[@]+"${CTL_F[@]}"})
  for a in ${ARMS_F[@]+"${ARMS_F[@]}"}; do
    in_list "$a" ${CTL_F[@]+"${CTL_F[@]}"} || JUDGED+=("$a")
  done
done
CONTROLS=()
for c in ${LISTED[@]+"${LISTED[@]}"}; do
  in_list "$c" ${JUDGED[@]+"${JUDGED[@]}"} ${CONTROLS[@]+"${CONTROLS[@]}"} || CONTROLS+=("$c")
done
is_control() { in_list "$1" ${CONTROLS[@]+"${CONTROLS[@]}"}; }
# A file without the lines of the control arms (determinism.txt, hashes.txt),
# then `.` (so trailing newlines survive $(...)); fails on an unreadable file.
without_controls() {
  local l f
  while IFS= read -r l || [[ -n "$l" ]]; do
    f="${l%% *}"; f="${f%:}"; f="${f%%/*}"
    is_control "$f" || printf '%s\n' "$l"
  done <"$1" && echo .
}

arms() { (cd "$1" && for s in */summary.json; do [[ -f "$s" ]] && dirname "$s"; done) | LC_ALL=C sort; }
diffs=0 same=0 ran=0 skipped=0 failed=0 ndumps=0 nctl=0
say() { echo "$*"; diffs=$((diffs + 1)); }
# {exit, skipped} of a summary.json, or `unreadable` (a difference even when
# both sides are unreadable: nothing was compared).
summary() { jq -ce '{exit, skipped}' "$1" 2>/dev/null || echo unreadable; }
while IFS= read -r arm; do
  if [[ ! -f "$A/$arm/summary.json" ]]; then say "DIFF $arm: only in the runpod-http run"; continue; fi
  if [[ ! -f "$B/$arm/summary.json" ]]; then say "DIFF $arm: only in the fv-lab run"; continue; fi
  ea="$(summary "$A/$arm/summary.json")" eb="$(summary "$B/$arm/summary.json")"
  if [[ "$ea" == unreadable || "$eb" == unreadable ]]; then say "DIFF $arm: summary.json unreadable ($ea / $eb)"; continue; fi
  [[ "$ea" == "$eb" ]] || { say "DIFF $arm: summary $ea (fv-lab) vs $eb (runpod-http)"; continue; }
  fa="$A/$arm/frames.sha256.json" fb="$B/$arm/frames.sha256.json"
  if is_control "$arm"; then
    # Reported, not judged: it exists in both and ran alike; its frames and
    # dump lines are only described.
    ba=0 bb=0
    [[ -f "$A/$arm/benchmark.json" ]] && ba=1
    [[ -f "$B/$arm/benchmark.json" ]] && bb=1
    [[ "$ba" == "$bb" ]] || { say "DIFF $arm: benchmark.json in one run only"; continue; }
    # Its frames' bytes may differ (the timed pick); their presence and
    # count may not.
    frames="no frames"
    if [[ -f "$fa" || -f "$fb" ]]; then
      if [[ ! -f "$fa" || ! -f "$fb" ]]; then say "DIFF $arm: frames.sha256.json in one run only"; continue; fi
      na="$(jq -er .count "$fa" 2>/dev/null || echo unreadable)" nb="$(jq -er .count "$fb" 2>/dev/null || echo unreadable)"
      if [[ "$na" == unreadable || "$nb" == unreadable ]]; then say "DIFF $arm: frames.sha256.json unreadable"; continue; fi
      [[ "$na" == "$nb" ]] || { say "DIFF $arm: a control with $na vs $nb frames"; continue; }
      if cmp -s "$fa" "$fb"; then frames="$na frames byte-identical"; else frames="frames differ ($na vs $nb frames)"; fi
    fi
    da="$(grep -a '^compare-dumps ' "$A/$arm/stderr.log" 2>/dev/null || true)"
    db="$(grep -a '^compare-dumps ' "$B/$arm/stderr.log" 2>/dev/null || true)"
    if [[ -n "$da$db" ]]; then
      if [[ "$da" == "$db" ]]; then frames+=", the same compare-dumps lines"; else frames+=", compare-dumps lines differ"; fi
    fi
    echo "control $arm: reported, not judged ($ea, $frames)"
    nctl=$((nctl + 1))
    continue
  fi
  if [[ -f "$fa" || -f "$fb" ]]; then
    if [[ ! -f "$fa" || ! -f "$fb" ]]; then say "DIFF $arm: frames.sha256.json in one run only"; continue; fi
    na="$(jq -er .count "$fa" 2>/dev/null || echo unreadable)" nb="$(jq -er .count "$fb" 2>/dev/null || echo unreadable)"
    if [[ "$na" == unreadable || "$nb" == unreadable ]]; then say "DIFF $arm: frames.sha256.json unreadable"; continue; fi
    cmp -s "$fa" "$fb" || { say "DIFF $arm: frames differ ($na vs $nb frames; diff the two frames.sha256.json)"; continue; }
  fi
  ba=0 bb=0
  [[ -f "$A/$arm/benchmark.json" ]] && ba=1
  [[ -f "$B/$arm/benchmark.json" ]] && bb=1
  [[ "$ba" == "$bb" ]] || { say "DIFF $arm: benchmark.json in one run only"; continue; }
  da="$(grep -a '^compare-dumps ' "$A/$arm/stderr.log" 2>/dev/null || true)"
  db="$(grep -a '^compare-dumps ' "$B/$arm/stderr.log" 2>/dev/null || true)"
  if [[ -n "$da$db" ]]; then
    [[ "$da" == "$db" ]] || { say "DIFF $arm: compare-dumps lines differ (diff the two stderr.log)"; continue; }
    ndumps=$((ndumps + 1))
  fi
  frames="no frames"
  [[ -f "$fa" ]] && frames="$na frames byte-identical"
  echo "same $arm: $ea, $frames"
  same=$((same + 1))
  # What the arm did in both runs: ran (exit 0, with byte-identical frames),
  # skipped, or failed alike (a crash in both is equal, not evidence).
  if [[ "$(jq -r '.skipped // empty' <<<"$ea")" != "" ]]; then
    skipped=$((skipped + 1))
  elif [[ "$(jq -r '.exit' <<<"$ea")" != 0 ]]; then
    failed=$((failed + 1))
  elif [[ -f "$fa" && "$na" -gt 0 ]]; then
    ran=$((ran + 1))
  fi
done < <({ arms "$A"; arms "$B"; } | LC_ALL=C sort -u)
reports() { (cd "$1" && for r in compare/compare-clips-*.json gate/gate-*.json; do [[ -f "$r" ]] && echo "$r"; done) | LC_ALL=C sort; }
verdict() {
  case "$1" in
    */gate/*) jq -e . "$1" >/dev/null 2>&1 && echo present || echo unreadable ;;
    *) jq -ce '{status: .status, off_identity: (.context.off_identity.status? // null)}' "$1" 2>/dev/null || echo unreadable ;;
  esac
}
nrep=0
while IFS= read -r r; do
  if [[ ! -f "$A/$r" ]]; then say "DIFF $r: only in the runpod-http run"; continue; fi
  if [[ ! -f "$B/$r" ]]; then say "DIFF $r: only in the fv-lab run"; continue; fi
  va="$(verdict "$A/$r")" vb="$(verdict "$B/$r")"
  if [[ "$va" == unreadable || "$vb" == unreadable ]]; then say "DIFF $r: unreadable ($va / $vb)"; continue; fi
  [[ "$va" == "$vb" ]] || { say "DIFF $r: $va (fv-lab) vs $vb (runpod-http)"; continue; }
  echo "same $r: $va"
  nrep=$((nrep + 1))
done < <({ reports "$A"; reports "$B"; } | LC_ALL=C sort -u)
vfiles() { (cd "$1" && for r in *.verdict.json *.diff */*-evidence.log determinism.txt hashes.txt */hashes.txt */hashes.json; do [[ -f "$r" ]] && echo "$r"; done) | LC_ALL=C sort; }
nver=0
while IFS= read -r r; do
  if [[ "$r" == */* ]] && is_control "${r%%/*}"; then
    # A control arm's own verdict file: reported, neither counted nor a difference.
    if [[ ! -f "$A/$r" ]]; then v="only in the runpod-http run"
    elif [[ ! -f "$B/$r" ]]; then v="only in the fv-lab run"
    elif cmp -s "$A/$r" "$B/$r"; then v="the same"
    else v="differs"
    fi
    echo "control ${r%%/*}: $r reported, not judged ($v)"
    continue
  fi
  if [[ ! -f "$A/$r" ]]; then say "DIFF $r: only in the runpod-http run"; continue; fi
  if [[ ! -f "$B/$r" ]]; then say "DIFF $r: only in the fv-lab run"; continue; fi
  if [[ -n "${CONTROLS[*]+x}" ]] && [[ "$r" == determinism.txt || "$r" == hashes.txt ]]; then
    # The run's hash report, compared without the control arms' lines.
    if [[ ! -r "$A/$r" || ! -r "$B/$r" ]] \
      || ! wa="$(without_controls "$A/$r" 2>/dev/null)" || ! wb="$(without_controls "$B/$r" 2>/dev/null)"; then
      say "DIFF $r: unreadable"; continue
    fi
    if [[ "$wa" == . && "$wb" == . ]]; then
      echo "$r: only control lines (not compared, not counted)"; continue
    fi
    [[ "$wa" == "$wb" ]] || { say "DIFF $r: differs without the control arms' lines (diff the two files)"; continue; }
    if cmp -s "$A/$r" "$B/$r"; then echo "same $r"; else echo "same $r (without the control arms' lines)"; fi
    nver=$((nver + 1))
    continue
  fi
  if ! cmp -s "$A/$r" "$B/$r"; then
    # A family verdict: the same cells and reasons in another order is the same.
    fam='select(has("family")) | .cells |= (split(" ") | sort | join(" ")) | .why |= (split(" ") | sort | join(" "))'
    na="$(jq -cS "$fam" "$A/$r" 2>/dev/null || true)" nb="$(jq -cS "$fam" "$B/$r" 2>/dev/null || true)"
    if [[ -z "$na" || "$na" != "$nb" ]]; then say "DIFF $r: differs (diff the two files)"; continue; fi
  fi
  echo "same $r"
  nver=$((nver + 1))
done < <({ vfiles "$A"; vfiles "$B"; } | LC_ALL=C sort -u)
# The derived-clip reports (ADR-0019 §20): metrics.txt, judged on its shape.
mfiles() { (cd "$1" && for r in */metrics.txt; do [[ -f "$r" ]] && echo "$r"; done) | LC_ALL=C sort; }
# Each line up to its last ": " (its words), `.` after (trailing lines kept).
mshape() { sed 's/\(.*\): .*/\1/' "$1" && echo .; }
nmet=0
while IFS= read -r r; do
  if [[ ! -f "$A/$r" ]]; then say "DIFF $r: only in the runpod-http run"; continue; fi
  if [[ ! -f "$B/$r" ]]; then say "DIFF $r: only in the fv-lab run"; continue; fi
  if cmp -s "$A/$r" "$B/$r"; then
    echo "same $r"
  else
    if ! ma="$(mshape "$A/$r" 2>/dev/null)" || ! mb="$(mshape "$B/$r" 2>/dev/null)" || [[ "$ma" != "$mb" ]]; then
      say "DIFF $r: other measurements (diff the two files)"; continue
    fi
    echo "metrics $r: the same measurements, values differ (reported, not judged: the clips' encodes)"
  fi
  nmet=$((nmet + 1))
done < <({ mfiles "$A"; mfiles "$B"; } | LC_ALL=C sort -u)
echo "$same arm(s) the same ($ran ran with byte-identical frames, $skipped skipped in both, $failed failed in both; $ndumps with the same compare-dumps lines), $nctl control arm(s) reported, $nrep report(s) the same, $nver verdict file(s) the same, $nmet metrics report(s), $diffs difference(s)"
if (( nver < EXPECT_VERDICTS )); then
  echo "expected at least $EXPECT_VERDICTS verdict file(s) the same, found $nver"
  exit 1
fi
if (( ran == 0 )); then
  echo "no arm ran with frames in both runs: nothing was compared"
  exit 1
fi
(( diffs == 0 ))
