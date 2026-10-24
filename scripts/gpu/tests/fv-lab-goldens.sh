#!/usr/bin/env bash
# The goldens of fv-lab (docs/adr/0019-fv-lab.md §5, §6): what today's scripts
# do for the rtx6000 family, from the scripts themselves, for fixed inputs.
# fv-lab must do the same (crates/fastvideo-lab/tests/{dry_run,payload_golden}.rs).
#
#   fv-lab-goldens.sh           regenerate crates/fastvideo-lab/fixtures/*
#   fv-lab-goldens.sh --check   fail when a script's output differs from the fixture
#
# Fixtures:
#   rtx6000-dry-run.txt        FV_DRY_RUN=1 runpod-matrix.sh rtx6000: every log line
#                              without its [HH:MM:SS] stamp, the volume at /workspace,
#                              the container disk at /fvscratch, run tag $TAG
#   rtx6000-gates.tsv          <arm> <verify-weights.sh cell> <its needs>: the
#                              gated_cell pairs of the family's `case` branch, and
#                              `verify-weights.sh --list` for each cell
#   runpod-http-rtx6000.json   runpod-http.sh plan $SHA $TAG with FV_POD_PREFIX=fv-lab:
#                              the PodCreateInput of the run (jq -S)
#   frames-sample.sha256.json  runpod-matrix.sh's frames_digest over frames-sample/frames
#                              (the per-arm byte-identity record, ADR-0019 §8)
#
# Every script runs under `env -i` (PATH, HOME, LC_ALL=C only): the caller's
# FV_* / RUNPOD_* settings cannot reach the fixtures.
#
# No network, no GPU, no weights: the dry run logs instead of running, and the
# plan takes the volume's datacentre from FV_PLAN_VOLUME. Needs bash and jq.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="$ROOT/crates/fastvideo-lab/fixtures"
CHECK=0
[[ "${1:-}" == --check ]] && CHECK=1
export LC_ALL=C

# The inputs (the Rust tests use the same).
FAMILY=rtx6000
SHA=1234567
TAG="$SHA-10241200"
VOLUME="bz0ofkfx45 EU-RO-1"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
[[ "$TMP" =~ ^[A-Za-z0-9/._-]+$ ]] || { echo "fv-lab-goldens: temp dir $TMP has shell-special characters" >&2; exit 2; }
mkdir -p "$TMP/work" "$TMP/scratch" "$TMP/new"

# A clean environment for every script run.
clean() { env -i PATH="$PATH" HOME="$HOME" LC_ALL=C "$@"; }

# 1. The dry run: the commands, in order.
clean FV_DRY_RUN=1 FV_WORK="$TMP/work" FV_SCRATCH="$TMP/scratch" FV_RUN_TAG="$TAG" \
  bash "$ROOT/scripts/gpu/runpod-matrix.sh" "$FAMILY" 2>/dev/null \
  | sed -E 's/^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\] //' \
  | grep -F "[$FAMILY] " \
  | sed -e "s#$TMP/scratch#/fvscratch#g" -e "s#$TMP/work#/workspace#g" >"$TMP/new/$FAMILY-dry-run.txt"

# 2. The weight gates: `gated_cell <arm> <cell>` of the family's branch.
awk -v f="  $FAMILY)" '$0 == f {on=1; next} on && /^    ;;/ {exit} on' "$ROOT/scripts/gpu/runpod-matrix.sh" \
  | grep -oE 'gated_cell [^ ]+ [^ ]+' >"$TMP/pairs.txt"
list="$(clean bash "$ROOT/scripts/gpu/verify-weights.sh" --list)"
while read -r _ arm cell; do
  needs="$(awk -v c="$cell" '$1 == c { $1 = ""; sub(/^ +/, ""); print; exit }' <<<"$list")"
  [[ -n "$needs" ]] || { echo "fv-lab-goldens: verify-weights.sh --list has no cell $cell" >&2; exit 2; }
  printf '%s\t%s\t%s\n' "$arm" "$cell" "$needs"
done <"$TMP/pairs.txt" >"$TMP/new/$FAMILY-gates.tsv"

# 3. The pod body.
clean RUNPOD_API_KEY=golden FV_PLAN_VOLUME="$VOLUME" FV_POD_PREFIX=fv-lab FV_FAMILY="$FAMILY" \
  bash "$ROOT/scripts/gpu/runpod-http.sh" plan "$SHA" "$TAG" | jq -S . >"$TMP/new/runpod-http-$FAMILY.json"

# 4. The frames digest: the script's own function, on a copy of the sample.
sed -n '/^frames_digest() {$/,/^}$/p' "$ROOT/scripts/gpu/runpod-matrix.sh" >"$TMP/digest.sh"
grep -q '^frames_digest() {$' "$TMP/digest.sh" || { echo "fv-lab-goldens: no frames_digest in runpod-matrix.sh" >&2; exit 2; }
cp -a "$OUT/frames-sample" "$TMP/sample"
clean bash -c '. "$1" && frames_digest "$2" sample' _ "$TMP/digest.sh" "$TMP/sample"
cp "$TMP/sample/frames.sha256.json" "$TMP/new/frames-sample.sha256.json"

rc=0
# 5. fv-lab-compare-runs.sh (the GPU gate's comparison) on two synthetic runs:
#    equal runs pass; a changed frame, exit or missing arm fails.
mk() { # dir arm exit frame-byte
  mkdir -p "$1/$2/frames"; printf '{"cell":"%s","exit":%s,"seconds":%s}\n' "$2" "$3" "$RANDOM" >"$1/$2/summary.json"
  printf '%s' "$4" >"$1/$2/frames/frame-000.png"; echo '{}' >"$1/$2/benchmark.json"
  (. "$TMP/digest.sh" && frames_digest "$1/$2" "$2")
}
CMP="$ROOT/scripts/gpu/fv-lab-compare-runs.sh"
mk "$TMP/ra" a 0 x; mk "$TMP/rb" a 0 x
mkdir -p "$TMP/ra/s" "$TMP/rb/s"
echo '{"cell":"s","exit":null,"skipped":"weights incomplete"}' | tee "$TMP/ra/s/summary.json" >"$TMP/rb/s/summary.json"
clean bash "$CMP" "$TMP/ra" "$TMP/rb" >/dev/null || { echo "fv-lab-goldens: compare-runs failed on equal runs" >&2; rc=1; }
cp -a "$TMP/rb" "$TMP/rc"; mk "$TMP/rc" a 0 y
cp -a "$TMP/rb" "$TMP/rd"; mk "$TMP/rd" a 3 x
cp -a "$TMP/rb" "$TMP/re"; rm -rf "$TMP/re/s"
for bad in rc rd re; do
  if clean bash "$CMP" "$TMP/ra" "$TMP/$bad" >/dev/null; then echo "fv-lab-goldens: compare-runs missed a difference ($bad)" >&2; rc=1; fi
done
mkdir -p "$OUT"
for f in "$FAMILY-dry-run.txt" "$FAMILY-gates.tsv" "runpod-http-$FAMILY.json" frames-sample.sha256.json; do
  [[ -s "$TMP/new/$f" ]] || { echo "fv-lab-goldens: $f came out empty" >&2; rc=1; continue; }
  if [[ "$CHECK" == 1 ]]; then
    if ! diff -u "$OUT/$f" "$TMP/new/$f"; then
      echo "fv-lab-goldens: $f differs from what the scripts do (rerun without --check, then make fv-lab match)" >&2
      rc=1
    fi
  else
    cp "$TMP/new/$f" "$OUT/$f"
    echo "wrote ${OUT#"$ROOT"/}/$f"
  fi
done
exit "$rc"
