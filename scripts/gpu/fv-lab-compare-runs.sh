#!/usr/bin/env bash
# The D1a GPU gate (docs/adr/0019-fv-lab.md §8): does an `fv-lab pod up` run
# reproduce a `runpod-http.sh run` of the same commit? Both run directories are
# what the fetches wrote under artifacts/runpod/<family>/<tag>/.
#
#   fv-lab-compare-runs.sh <fv-lab run dir> <runpod-http run dir>
#
# Same when, for every arm directory of either run (the directories holding a
# summary.json):
#   - the arm exists in both;
#   - summary.json agrees on `exit` and `skipped` (seconds and dates differ);
#   - frames.sha256.json is byte-identical (both runners write it after an arm:
#     every frame's SHA-256; docs/perf/determinism.md promises the same bytes
#     for the same request and seed on the same GPU class), or absent in both;
#   - benchmark.json is present in both or in neither.
# Exit 0 when the runs agree, 1 (with one line per difference) otherwise.
# Needs bash and jq.
set -euo pipefail
A="${1:?usage: fv-lab-compare-runs.sh <fv-lab run dir> <runpod-http run dir>}"
B="${2:?usage: fv-lab-compare-runs.sh <fv-lab run dir> <runpod-http run dir>}"
[[ -d "$A" && -d "$B" ]] || { echo "fv-lab-compare-runs: not a directory: $A or $B" >&2; exit 2; }

arms() { (cd "$1" && for s in */summary.json; do [[ -f "$s" ]] && dirname "$s"; done) | LC_ALL=C sort; }
diffs=0 same=0
say() { echo "$*"; diffs=$((diffs + 1)); }
while IFS= read -r arm; do
  if [[ ! -f "$A/$arm/summary.json" ]]; then say "DIFF $arm: only in the runpod-http run"; continue; fi
  if [[ ! -f "$B/$arm/summary.json" ]]; then say "DIFF $arm: only in the fv-lab run"; continue; fi
  ea="$(jq -c '{exit, skipped}' "$A/$arm/summary.json")"
  eb="$(jq -c '{exit, skipped}' "$B/$arm/summary.json")"
  [[ "$ea" == "$eb" ]] || { say "DIFF $arm: summary $ea (fv-lab) vs $eb (runpod-http)"; continue; }
  fa="$A/$arm/frames.sha256.json" fb="$B/$arm/frames.sha256.json"
  if [[ -f "$fa" || -f "$fb" ]]; then
    if [[ ! -f "$fa" || ! -f "$fb" ]]; then say "DIFF $arm: frames.sha256.json in one run only"; continue; fi
    cmp -s "$fa" "$fb" || { say "DIFF $arm: frames differ ($(jq -r .count "$fa") vs $(jq -r .count "$fb") frames; diff the two frames.sha256.json)"; continue; }
  fi
  ba=0 bb=0
  [[ -f "$A/$arm/benchmark.json" ]] && ba=1
  [[ -f "$B/$arm/benchmark.json" ]] && bb=1
  [[ "$ba" == "$bb" ]] || { say "DIFF $arm: benchmark.json in one run only"; continue; }
  frames="no frames"
  [[ -f "$fa" ]] && frames="$(jq -r .count "$fa") frames byte-identical"
  echo "same $arm: $ea, $frames"
  same=$((same + 1))
done < <({ arms "$A"; arms "$B"; } | LC_ALL=C sort -u)
echo "$same arm(s) the same, $diffs difference(s)"
(( same > 0 && diffs == 0 ))
