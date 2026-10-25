#!/usr/bin/env bash
# The goldens of the restored script modes (docs/adr/0019-fv-lab.md §12): for
# each `<tier>:<family>:<K=V>` of FV_LAB_RESTORED (scripts/gpu/fv-lab-retired.sh),
# what runpod-matrix.sh and runpod-http.sh did in that mode at the last commit
# that had the family's branch, written as the tier's fixtures:
#
#   <tier>-dry-run.txt        <K=V> FV_DRY_RUN=1 runpod-matrix.sh <family>
#   <tier>-gates.tsv          its FV_DRY_RUN_GATES records with the cells' needs
#   runpod-http-<tier>.json   FV_FAMILY=<family> FV_EXTRA_ENV=<K=V> runpod-http.sh plan
#
# with the inputs and rewriting of fv-lab-goldens.sh (the same steps, the same
# env -i). The output is the script's, byte for byte: the family's name and the
# mode's setting are in it, and the Rust tests (dry_run.rs, payload_golden.rs)
# map them to the tier's name and no setting before comparing.
#
#   fv-lab-restored-goldens.sh           write them (they are frozen: run it only
#                                        to reproduce them, or after a deliberate change)
#   fv-lab-restored-goldens.sh --check   fail when a fixture is not what the old scripts write
#
# FV_RESTORED_FROM (default 5987e6a4, main before #185) is the commit read with
# `git archive`, so this needs the repository's history (not a shallow clone):
# a manual check, not CI's. No network, no GPU, no weights. Needs git, bash, jq.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="$ROOT/crates/fastvideo-lab/fixtures"
FROM="${FV_RESTORED_FROM:-5987e6a4}"
CHECK=0
[[ "${1:-}" == --check ]] && CHECK=1
export LC_ALL=C

# The inputs of fv-lab-goldens.sh (the Rust tests use the same).
SHA=1234567
TAG="$SHA-10241200"
BASELINE_SHA=7654321
VOLUME="bz0ofkfx45 EU-RO-1"

# shellcheck source=scripts/gpu/fv-lab-retired.sh
. "$ROOT/scripts/gpu/fv-lab-retired.sh"
[[ -n "${FV_LAB_RESTORED:-}" ]] || { echo "fv-lab-restored-goldens: FV_LAB_RESTORED is empty" >&2; exit 2; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
[[ "$TMP" =~ ^[A-Za-z0-9/._-]+$ ]] || { echo "fv-lab-restored-goldens: temp dir $TMP has shell-special characters" >&2; exit 2; }
OLD="$TMP/old"
mkdir -p "$OLD" "$TMP/new"
git -C "$ROOT" cat-file -e "$FROM^{commit}" 2>/dev/null \
  || { echo "fv-lab-restored-goldens: commit $FROM is not in this clone (fetch the history)" >&2; exit 2; }
git -C "$ROOT" archive "$FROM" scripts/gpu | tar -x -C "$OLD"

clean() { env -i PATH="$PATH" HOME="$HOME" LC_ALL=C "$@"; }
list="$(clean bash "$OLD/scripts/gpu/verify-weights.sh" --list)"

FIXTURES=()
for entry in $FV_LAB_RESTORED; do
  IFS=: read -r TIER FAMILY MODE <<<"$entry"
  [[ -n "$TIER" && -n "$FAMILY" && "$MODE" == *=* ]] \
    || { echo "fv-lab-restored-goldens: bad FV_LAB_RESTORED entry $entry" >&2; exit 2; }
  rm -rf "$TMP/work" "$TMP/scratch" "$TMP/pairs.txt"
  mkdir -p "$TMP/work/weights/auxiliary/tae" "$TMP/scratch"
  : >"$TMP/work/weights/auxiliary/tae/.complete"
  # 1. The dry run, in the mode.
  clean "$MODE" FV_DRY_RUN=1 FV_DRY_RUN_GATES="$TMP/pairs.txt" FV_WORK="$TMP/work" FV_SCRATCH="$TMP/scratch" \
    FV_RUN_TAG="$TAG" FV_BASELINE_SHA="$BASELINE_SHA" \
    bash "$OLD/scripts/gpu/runpod-matrix.sh" "$FAMILY" 2>/dev/null \
    | sed -E 's/^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\] //' \
    | grep -F "[$FAMILY] " \
    | sed -e "s#$TMP/scratch#/fvscratch#g" -e "s#$TMP/work#/workspace#g" -e "s#$OLD#/opt/fastvideo-rs#g" \
      >"$TMP/new/$TIER-dry-run.txt"
  # 2. The gates, in run order.
  [[ -s "$TMP/pairs.txt" ]] || { echo "fv-lab-restored-goldens: $entry gated no cell" >&2; exit 2; }
  while IFS=$'\t' read -r what a b c d; do
    case "$what" in
      arm)
        needs="$(awk -v x="$b" '$1 == x { $1 = ""; sub(/^ +/, ""); print; exit }' <<<"$list")"
        [[ -n "$needs" ]] || { echo "fv-lab-restored-goldens: verify-weights.sh --list has no cell $b" >&2; exit 2; }
        printf 'arm\t%s\t%s\t%s\t%s\n' "$a" "$b" "$c" "$needs" ;;
      gate) printf 'gate\t%s\t%s\t%s\t%s\n' "$a" "$b" "$c" "$d" ;;
      *) echo "fv-lab-restored-goldens: unknown FV_DRY_RUN_GATES line $what" >&2; exit 2 ;;
    esac
  done <"$TMP/pairs.txt" | sed -e "s#$TMP/scratch#/fvscratch#g" -e "s#$TMP/work#/workspace#g" >"$TMP/new/$TIER-gates.tsv"
  # 3. The pod body the mode's run created.
  clean RUNPOD_API_KEY=golden FV_PLAN_VOLUME="$VOLUME" FV_POD_PREFIX=fv-lab FV_FAMILY="$FAMILY" FV_EXTRA_ENV="$MODE" \
    bash "$OLD/scripts/gpu/runpod-http.sh" plan "$SHA" "$TAG" | jq -S . >"$TMP/new/runpod-http-$TIER.json"
  FIXTURES+=("$TIER-dry-run.txt" "$TIER-gates.tsv" "runpod-http-$TIER.json")
done

rc=0
for f in ${FIXTURES[@]+"${FIXTURES[@]}"}; do
  [[ -s "$TMP/new/$f" ]] || { echo "fv-lab-restored-goldens: $f came out empty" >&2; rc=1; continue; }
  if [[ "$CHECK" == 1 ]]; then
    if ! diff -u "$OUT/$f" "$TMP/new/$f"; then
      echo "fv-lab-restored-goldens: $f differs from what the scripts at $FROM write" >&2
      rc=1
    fi
  else
    cp "$TMP/new/$f" "$OUT/$f"
    echo "wrote ${OUT#"$ROOT"/}/$f (from $FROM)"
  fi
done
exit "$rc"
