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

# 1. The dry run: the commands, in order.
FV_DRY_RUN=1 FV_WORK="$TMP/work" FV_SCRATCH="$TMP/scratch" FV_RUN_TAG="$TAG" FV_CELLS="" FV_PROMPTS="" FV_LPIPS="" \
  FV_GPUCHECK="" FV_GEN_TIMEOUT_S="" bash "$ROOT/scripts/gpu/runpod-matrix.sh" "$FAMILY" 2>/dev/null \
  | sed -E 's/^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\] //' \
  | grep -F "[$FAMILY] " \
  | sed -e "s#$TMP/scratch#/fvscratch#g" -e "s#$TMP/work#/workspace#g" >"$TMP/new/$FAMILY-dry-run.txt"

# 2. The weight gates: `gated_cell <arm> <cell>` of the family's branch.
awk -v f="  $FAMILY)" '$0 == f {on=1; next} on && /^    ;;/ {exit} on' "$ROOT/scripts/gpu/runpod-matrix.sh" \
  | grep -oE 'gated_cell [^ ]+ [^ ]+' >"$TMP/pairs.txt"
list="$(bash "$ROOT/scripts/gpu/verify-weights.sh" --list)"
while read -r _ arm cell; do
  needs="$(awk -v c="$cell" '$1 == c { $1 = ""; sub(/^ +/, ""); print; exit }' <<<"$list")"
  [[ -n "$needs" ]] || { echo "fv-lab-goldens: verify-weights.sh --list has no cell $cell" >&2; exit 2; }
  printf '%s\t%s\t%s\n' "$arm" "$cell" "$needs"
done <"$TMP/pairs.txt" >"$TMP/new/$FAMILY-gates.tsv"

# 3. The pod body.
RUNPOD_API_KEY=golden FV_PLAN_VOLUME="$VOLUME" FV_POD_PREFIX=fv-lab FV_FAMILY="$FAMILY" \
  bash "$ROOT/scripts/gpu/runpod-http.sh" plan "$SHA" "$TAG" | jq -S . >"$TMP/new/runpod-http-$FAMILY.json"

rc=0
mkdir -p "$OUT"
for f in "$FAMILY-dry-run.txt" "$FAMILY-gates.tsv" "runpod-http-$FAMILY.json"; do
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
