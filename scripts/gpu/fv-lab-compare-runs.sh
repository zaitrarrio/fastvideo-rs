#!/usr/bin/env bash
# The GPU gate of D1a / D1b (docs/adr/0019-fv-lab.md §8): does an `fv-lab pod
# up` run reproduce a `runpod-http.sh run` of the same commit? Both run
# directories are what the fetches wrote under artifacts/runpod/<family>/<tag>/.
#
#   fv-lab-compare-runs.sh [--expect-verdicts N] <fv-lab run dir> <runpod-http run dir>
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
#     script in FV_B4A_CELLS's (ADR-0019 §11.5).
# (`fv-lab pod fetch` takes these; on the runpod-http.sh side they need
# FV_FETCH_TREE=1.) --expect-verdicts N fails unless at least N verdict files
# compared the same: a tier with verdict cells whose files never arrived
# would otherwise pass with none.
# Exit 0 when the runs agree and at least one arm ran (exit 0) with
# byte-identical frames in both; 1 otherwise (one line per difference). An
# unreadable summary, frames record or report is a difference; arms skipped or
# failed alike in both runs are counted, not evidence.
# Needs bash and jq.
set -euo pipefail
USAGE="usage: fv-lab-compare-runs.sh [--expect-verdicts N] <fv-lab run dir> <runpod-http run dir>"
EXPECT_VERDICTS=0
if [[ "${1:-}" == --expect-verdicts ]]; then
  [[ "${2:-}" =~ ^[0-9]+$ ]] || { echo "$USAGE" >&2; exit 2; }
  EXPECT_VERDICTS="$2"
  shift 2
fi
A="${1:?$USAGE}"
B="${2:?$USAGE}"
[[ -d "$A" && -d "$B" ]] || { echo "fv-lab-compare-runs: not a directory: $A or $B" >&2; exit 2; }

arms() { (cd "$1" && for s in */summary.json; do [[ -f "$s" ]] && dirname "$s"; done) | LC_ALL=C sort; }
diffs=0 same=0 ran=0 skipped=0 failed=0
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
vfiles() { (cd "$1" && for r in *.verdict.json *.diff */*-evidence.log; do [[ -f "$r" ]] && echo "$r"; done) | LC_ALL=C sort; }
nver=0
while IFS= read -r r; do
  if [[ ! -f "$A/$r" ]]; then say "DIFF $r: only in the runpod-http run"; continue; fi
  if [[ ! -f "$B/$r" ]]; then say "DIFF $r: only in the fv-lab run"; continue; fi
  if ! cmp -s "$A/$r" "$B/$r"; then
    # A family verdict: the same cells and reasons in another order is the same.
    fam='select(has("family")) | .cells |= (split(" ") | sort | join(" ")) | .why |= (split(" ") | sort | join(" "))'
    na="$(jq -cS "$fam" "$A/$r" 2>/dev/null || true)" nb="$(jq -cS "$fam" "$B/$r" 2>/dev/null || true)"
    if [[ -z "$na" || "$na" != "$nb" ]]; then say "DIFF $r: differs (diff the two files)"; continue; fi
  fi
  echo "same $r"
  nver=$((nver + 1))
done < <({ vfiles "$A"; vfiles "$B"; } | LC_ALL=C sort -u)
echo "$same arm(s) the same ($ran ran with byte-identical frames, $skipped skipped in both, $failed failed in both), $nrep report(s) the same, $nver verdict file(s) the same, $diffs difference(s)"
if (( nver < EXPECT_VERDICTS )); then
  echo "expected at least $EXPECT_VERDICTS verdict file(s) the same, found $nver"
  exit 1
fi
if (( ran == 0 )); then
  echo "no arm ran with frames in both runs: nothing was compared"
  exit 1
fi
(( diffs == 0 ))
