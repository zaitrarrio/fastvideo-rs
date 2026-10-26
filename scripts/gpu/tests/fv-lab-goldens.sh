#!/usr/bin/env bash
# The goldens of fv-lab (docs/adr/0019-fv-lab.md §5, §6, D1b, §10): what the
# scripts do for every family lab/tiers.toml carries as a tier, from the
# scripts themselves, for fixed inputs. fv-lab must do the same
# (crates/fastvideo-lab/tests/{dry_run,payload_golden}.rs).
#
#   fv-lab-goldens.sh           regenerate crates/fastvideo-lab/fixtures/*
#   fv-lab-goldens.sh --check   fail when a script's output differs from the fixture
#
# Frozen since the GPU gate (ADR-0019 §10): a family of fv-lab-retired.sh
# passed its GPU gate and lost its runpod-matrix.sh branch, so the scripts can
# no longer produce its fixtures. Its three fixtures are fv-lab's own goldens
# from then on: never regenerated here, only checked to be present (and the
# family to be a tier with no branch left). A change to such a tier's commands
# or pod is deliberate: the fixture is edited in the same PR, and the Rust
# tests hold fv-lab to it.
# The restored modes (FV_LAB_RESTORED, ADR-0019 §12: precision-nvfp4,
# ltxvae-gen) are in that list from the start: their fixtures are the script's
# in that mode at 5987e6a4 (fv-lab-restored-goldens.sh reproduces them).
#
# Fixtures, per family <f> (every [[tier]] name of lab/tiers.toml but the
# frozen ones; for the `gold` tier, generated from lab/gold/ and no script
# family, only the pod body):
#   <f>-dry-run.txt            FV_DRY_RUN=1 runpod-matrix.sh <f>: every log line of the
#                              family without its [HH:MM:SS] stamp, the volume at
#                              /workspace (its auxiliary/tae complete), the container disk at /fvscratch, the
#                              repository at /opt/fastvideo-rs (the image's root), run
#                              tag $TAG, baseline commit $BASELINE_SHA
#   <f>-gates.tsv              in run order (FV_DRY_RUN_GATES): `arm <name> <verify-weights.sh
#                              cell> <TAE file or -> <the cell's needs>` per gated cell
#                              (`verify-weights.sh --list`), `gate <baseline> <candidate>
#                              <kind> <OFF arm or ->` per promotion gate
#   runpod-http-<f>.json       FV_FAMILY=<f> runpod-http.sh plan $SHA $TAG with
#                              FV_POD_PREFIX=fv-lab: the PodCreateInput of the run (jq -S)
# and once:
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
SHA=1234567
TAG="$SHA-10241200"
BASELINE_SHA=7654321
VOLUME="bz0ofkfx45 EU-RO-1"
# The families: the tiers of lab/tiers.toml, in order.
FAMILIES="$(awk '/^\[\[tier\]\]/ { t = 1; next } t && /^name = "/ { gsub(/^name = "|"$/, ""); print; t = 0 }' "$ROOT/lab/tiers.toml")"
[[ -n "$FAMILIES" ]] || { echo "fv-lab-goldens: no [[tier]] in lab/tiers.toml" >&2; exit 2; }
[[ "$ROOT" =~ ^[A-Za-z0-9/._-]+$ ]] || { echo "fv-lab-goldens: repository path $ROOT has shell-special characters" >&2; exit 2; }
# shellcheck source=scripts/gpu/fv-lab-retired.sh
. "$ROOT/scripts/gpu/fv-lab-retired.sh"
MATRIX="$ROOT/scripts/gpu/runpod-matrix.sh"
# A family's `case` branch in runpod-matrix.sh: a label of the top-level
# `case "$FAMILY" in` (two-space indent; `name)`, `name )`, `a|name)`, a
# trailing comment all count; nested cases' labels do not).
# grep reads all of its input (no -q): under pipefail an early exit would
# SIGPIPE sed and read as "no branch" whenever the branch is found.
has_branch() {
  sed -n '/^case "\$FAMILY" in$/,/^esac$/p' "$MATRIX" \
    | grep -E "^  ([A-Za-z0-9_-]+ *\| *)*$1( *\| *[A-Za-z0-9_-]+)* *\)" >/dev/null
}
for f in $FV_LAB_RETIRED; do
  grep -qx "$f" <<<"$FAMILIES" || { echo "fv-lab-goldens: retired family $f is no tier of lab/tiers.toml" >&2; exit 2; }
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
[[ "$TMP" =~ ^[A-Za-z0-9/._-]+$ ]] || { echo "fv-lab-goldens: temp dir $TMP has shell-special characters" >&2; exit 2; }
mkdir -p "$TMP/new"

# A clean environment for every script run.
clean() { env -i PATH="$PATH" HOME="$HOME" LC_ALL=C "$@"; }

list="$(clean bash "$ROOT/scripts/gpu/verify-weights.sh" --list)"
FIXTURES=()
FROZEN=()
for FAMILY in $FAMILIES; do
  # The `gold` tier (docs/adr/0023-lab-report-and-ci-bridge.md §3) is no
  # runpod-matrix.sh family: its cells come from lab/gold/ and fv-lab's own dry
  # run is its golden (fixtures/gold-dry-run.txt, tests/gold_tier.rs). Its pod
  # is runpod-http.sh's like every tier's, so only the pod body is written here.
  # Its container disk is its own (weights "volume+r2", ADR-0019 §14: the
  # largest arm's R2 trees on top of the volume tiers' 120 GB), passed as the
  # script's FV_CONTAINER_DISK_GB; its env (the R2 key as Runpod secret
  # references) is the one difference tests/payload_golden.rs allows.
  if [[ "$FAMILY" == gold ]]; then
    disk="$(awk '/^\[\[tier\]\]/ { t = 0 } /^name = "gold"$/ { t = 1 } t && /^container_disk_gb = / { print $3; exit }' "$ROOT/lab/tiers.toml")"
    [[ "$disk" =~ ^[0-9]+$ ]] || { echo "fv-lab-goldens: no container_disk_gb for the gold tier" >&2; exit 2; }
    clean RUNPOD_API_KEY=golden FV_PLAN_VOLUME="$VOLUME" FV_POD_PREFIX=fv-lab FV_FAMILY="$FAMILY" FV_CONTAINER_DISK_GB="$disk" \
      bash "$ROOT/scripts/gpu/runpod-http.sh" plan "$SHA" "$TAG" | jq -S . >"$TMP/new/runpod-http-$FAMILY.json"
    FIXTURES+=("runpod-http-$FAMILY.json")
    continue
  fi
  # Frozen since the GPU gate: present, and no branch left to regenerate them from.
  if fv_lab_retired "$FAMILY"; then
    if has_branch "$FAMILY"; then
      echo "fv-lab-goldens: $FAMILY is retired (fv-lab-retired.sh) but runpod-matrix.sh still has its branch" >&2
      exit 2
    fi
    FROZEN+=("$FAMILY-dry-run.txt" "$FAMILY-gates.tsv" "runpod-http-$FAMILY.json")
    continue
  fi
  has_branch "$FAMILY" || {
    echo "fv-lab-goldens: tier $FAMILY has no runpod-matrix.sh branch and is not in fv-lab-retired.sh (freeze its fixtures there, or restore the branch)" >&2
    exit 2
  }
  rm -rf "$TMP/work" "$TMP/scratch" "$TMP/pairs.txt"
  # The volume's TAE copy is complete, as on bz0ofkfx45 (auxiliary/tae): the
  # script's $TAE is then /workspace/weights/auxiliary/tae, not the container
  # disk, so a cell that hard-codes {scratch}/tae differs.
  mkdir -p "$TMP/work/weights/auxiliary/tae" "$TMP/scratch"
  : >"$TMP/work/weights/auxiliary/tae/.complete"
  # 1. The dry run: the commands, in order (the family's own log lines).
  clean FV_DRY_RUN=1 FV_DRY_RUN_GATES="$TMP/pairs.txt" FV_WORK="$TMP/work" FV_SCRATCH="$TMP/scratch" \
    FV_RUN_TAG="$TAG" FV_BASELINE_SHA="$BASELINE_SHA" \
    bash "$ROOT/scripts/gpu/runpod-matrix.sh" "$FAMILY" 2>/dev/null \
    | sed -E 's/^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\] //' \
    | grep -F "[$FAMILY] " \
    | sed -e "s#$TMP/scratch#/fvscratch#g" -e "s#$TMP/work#/workspace#g" -e "s#$ROOT#/opt/fastvideo-rs#g" \
      >"$TMP/new/$FAMILY-dry-run.txt"

  # 2. The gates, in run order: every `gated_cell` the dry run reached
  #    (arm, weight cell, TAE file or -, the cell's needs) and every
  #    `gate_cells` (gate, baseline, candidate, kind, OFF arm or -).
  [[ -s "$TMP/pairs.txt" ]] || { echo "fv-lab-goldens: $FAMILY gated no cell" >&2; exit 2; }
  while IFS=$'\t' read -r what a b c d; do
    case "$what" in
      arm)
        needs="$(awk -v x="$b" '$1 == x { $1 = ""; sub(/^ +/, ""); print; exit }' <<<"$list")"
        [[ -n "$needs" ]] || { echo "fv-lab-goldens: verify-weights.sh --list has no cell $b" >&2; exit 2; }
        printf 'arm\t%s\t%s\t%s\t%s\n' "$a" "$b" "$c" "$needs" ;;
      gate) printf 'gate\t%s\t%s\t%s\t%s\n' "$a" "$b" "$c" "$d" ;;
      *) echo "fv-lab-goldens: unknown FV_DRY_RUN_GATES line $what" >&2; exit 2 ;;
    esac
  done <"$TMP/pairs.txt" | sed -e "s#$TMP/scratch#/fvscratch#g" -e "s#$TMP/work#/workspace#g" >"$TMP/new/$FAMILY-gates.tsv"

  # 3. The pod body.
  clean RUNPOD_API_KEY=golden FV_PLAN_VOLUME="$VOLUME" FV_POD_PREFIX=fv-lab FV_FAMILY="$FAMILY" \
    bash "$ROOT/scripts/gpu/runpod-http.sh" plan "$SHA" "$TAG" | jq -S . >"$TMP/new/runpod-http-$FAMILY.json"
  FIXTURES+=("$FAMILY-dry-run.txt" "$FAMILY-gates.tsv" "runpod-http-$FAMILY.json")
done

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
# A compare and a gate report (D1b): equal compare verdicts pass, and so does
# another gate verdict (it weighs timings); another off-identity status or a
# missing report fails.
rep() { # dir file status off verdict
  mkdir -p "$(dirname "$1/$2")"
  printf '{"stage":"x","status":"%s","elapsed_s":%s,"context":{"off_identity":{"status":"%s"},"verdict":{"verdict":"%s"}}}\n' \
    "$3" "$RANDOM" "$4" "$5" >"$1/$2"
}
for r in ra rb; do
  rep "$TMP/$r" compare/compare-clips-a--s.json pass ok null
  rep "$TMP/$r" gate/gate-a--s.json pass ok pass
done
clean bash "$CMP" "$TMP/ra" "$TMP/rb" >/dev/null || { echo "fv-lab-goldens: compare-runs failed on equal runs" >&2; rc=1; }
cp -a "$TMP/rb" "$TMP/rc"; mk "$TMP/rc" a 0 y
cp -a "$TMP/rb" "$TMP/rd"; mk "$TMP/rd" a 3 x
cp -a "$TMP/rb" "$TMP/re"; rm -rf "$TMP/re/s"
cp -a "$TMP/rb" "$TMP/rf"; rep "$TMP/rf" compare/compare-clips-a--s.json fail fail null
cp -a "$TMP/rb" "$TMP/rg"; rep "$TMP/rg" gate/gate-a--s.json fail ok fail
clean bash "$CMP" "$TMP/ra" "$TMP/rg" >/dev/null || { echo "fv-lab-goldens: compare-runs failed on another gate verdict (timings)" >&2; rc=1; }
cp -a "$TMP/rb" "$TMP/rh"; rm "$TMP/rh/compare/compare-clips-a--s.json"
cp -a "$TMP/rb" "$TMP/ri"; rm "$TMP/ri/gate/gate-a--s.json"
# Nothing compared: both runs skipped every arm, both crashed every arm, or
# both summaries are unreadable. Equal, but no evidence: must fail.
for r in sa sb; do
  mkdir -p "$TMP/$r/s"; echo '{"cell":"s","exit":null,"skipped":"weights incomplete"}' >"$TMP/$r/s/summary.json"
done
for r in ca cb; do
  mkdir -p "$TMP/$r/a"; echo '{"cell":"a","exit":101,"seconds":3}' >"$TMP/$r/a/summary.json"
done
for r in ua ub; do
  mkdir -p "$TMP/$r/a"; echo 'not json' >"$TMP/$r/a/summary.json"
done
for pair in "sa sb" "ca cb" "ua ub"; do
  read -r x y <<<"$pair"
  if clean bash "$CMP" "$TMP/$x" "$TMP/$y" >/dev/null; then echo "fv-lab-goldens: compare-runs passed with nothing compared ($pair)" >&2; rc=1; fi
done
# Unreadable reports on both sides are a difference too.
cp -a "$TMP/rb" "$TMP/rj"; cp -a "$TMP/rb" "$TMP/rk"
echo 'not json' >"$TMP/rj/compare/compare-clips-a--s.json"; echo 'not json' >"$TMP/rk/compare/compare-clips-a--s.json"
if clean bash "$CMP" "$TMP/rj" "$TMP/rk" >/dev/null; then echo "fv-lab-goldens: compare-runs passed two unreadable reports" >&2; rc=1; fi
# The verdict step's files (D1d): equal in both pass; another verdict, an
# evidence file in one run only, or another evidence diff fails.
cp -a "$TMP/rb" "$TMP/va"; cp -a "$TMP/rb" "$TMP/vb"
for r in va vb; do
  echo '{"cell":"c","verdict":"pass","why":""}' >"$TMP/$r/fam-c.verdict.json"
  : >"$TMP/$r/c.evidence.diff"; echo 'h3 teacache step 1: REUSE' >"$TMP/$r/a/cache-evidence.log"
done
clean bash "$CMP" "$TMP/va" "$TMP/vb" >/dev/null || { echo "fv-lab-goldens: compare-runs failed on equal verdict files" >&2; rc=1; }
cp -a "$TMP/vb" "$TMP/vc"; echo '{"cell":"c","verdict":"fail","why":"pr:exit"}' >"$TMP/vc/fam-c.verdict.json"
cp -a "$TMP/vb" "$TMP/vd"; rm "$TMP/vd/a/cache-evidence.log"
cp -a "$TMP/vb" "$TMP/ve"; echo '1c1' >"$TMP/ve/c.evidence.diff"
for bad in vc vd ve; do
  if clean bash "$CMP" "$TMP/va" "$TMP/$bad" >/dev/null; then echo "fv-lab-goldens: compare-runs missed a verdict difference ($bad)" >&2; rc=1; fi
done
# --expect-verdicts: three files the same pass 3, not 4; runs without any fail 1.
clean bash "$CMP" --expect-verdicts 3 "$TMP/va" "$TMP/vb" >/dev/null || { echo "fv-lab-goldens: compare-runs --expect-verdicts 3 failed on three equal verdict files" >&2; rc=1; }
if clean bash "$CMP" --expect-verdicts 4 "$TMP/va" "$TMP/vb" >/dev/null; then echo "fv-lab-goldens: compare-runs --expect-verdicts 4 passed with three" >&2; rc=1; fi
if clean bash "$CMP" --expect-verdicts 1 "$TMP/ra" "$TMP/rb" >/dev/null; then echo "fv-lab-goldens: compare-runs --expect-verdicts 1 passed with no verdict file" >&2; rc=1; fi
# A family verdict lists its cells (and reasons) in the tier's order on fv-lab's
# side, FV_B4A_CELLS's on the script's: the same set is the same; another is not.
cp -a "$TMP/va" "$TMP/vf"; cp -a "$TMP/va" "$TMP/vg"; cp -a "$TMP/va" "$TMP/vh"
echo '{"family":"fam","verdict":"fail","cells":"c d","why":"c:fail d:fail"}' >"$TMP/vf/fam.verdict.json"
echo '{"family":"fam","verdict":"fail","cells":"d c","why":"d:fail c:fail"}' >"$TMP/vg/fam.verdict.json"
echo '{"family":"fam","verdict":"fail","cells":"d c","why":"d:fail"}' >"$TMP/vh/fam.verdict.json"
clean bash "$CMP" "$TMP/vf" "$TMP/vg" >/dev/null || { echo "fv-lab-goldens: compare-runs failed on a family verdict in another order" >&2; rc=1; }
if clean bash "$CMP" "$TMP/vf" "$TMP/vh" >/dev/null; then echo "fv-lab-goldens: compare-runs missed another family verdict" >&2; rc=1; fi
for bad in rc rd re rf rh ri; do
  if clean bash "$CMP" "$TMP/ra" "$TMP/$bad" >/dev/null; then echo "fv-lab-goldens: compare-runs missed a difference ($bad)" >&2; rc=1; fi
done
for f in ${FROZEN[@]+"${FROZEN[@]}"}; do
  [[ -s "$OUT/$f" ]] || { echo "fv-lab-goldens: frozen fixture $f is missing or empty (crates/fastvideo-lab/fixtures)" >&2; rc=1; continue; }
  [[ "$CHECK" == 1 ]] || echo "frozen ${OUT#"$ROOT"/}/$f (fv-lab-retired.sh: not regenerated)"
done
mkdir -p "$OUT"
for f in ${FIXTURES[@]+"${FIXTURES[@]}"} frames-sample.sha256.json; do
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
