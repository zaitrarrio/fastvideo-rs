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
# A sibling tier frozen with its family (FV_LAB_RETIRED_SIBLINGS, ADR-0019
# §22.19: hd5 with hd; §22.20: oracle-r2 with oracle) is in it too; so are the
# script-side GPU-check bodies of the volume+r2 families retired here
# (runpod-http-<f>-r2.json of serve-engine, sfstream and mmaudio, §22.19, and
# wan, with oracle's sfwan13 body, §22.20), which runpod-http.sh plan can no
# longer write (it routes those families to fv-lab).
# Since S8 (ADR-0019 §22.20) every runpod-matrix.sh family is frozen or
# retired with no tier, and the script is gone: nothing here runs it. Only
# the pod bodies of the tiers that never were script families (gold, the
# experiment tiers) and of the upstream pods are still written, by
# runpod-http.sh plan / plan-upstream, and the frames digest by the frozen
# copy of the script's frames_digest (fixtures/oracle-script/, §20.8 item 3).
#
# Fixtures, per family <f> (every [[tier]] name of lab/tiers.toml but the
# frozen ones, which since §22.20 leaves none; for the `gold` tier, generated
# from lab/gold/ and no script family, and the experiment tiers
# (FV_LAB_EXPERIMENTS), only the pod body), as they were written:
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
#   runpod-http-upstream-<s>.json  per stack <s> of lab/upstream.toml, runpod-http.sh
#                              plan-upstream with oracle.sh's settings for that stack
#                              (fv-lab oracle's upstream pods, ADR-0019 §19)
#   frames-sample.sha256.json  runpod-matrix.sh's frames_digest over frames-sample/frames
#                              (the per-arm byte-identity record, ADR-0019 §8), cut
#                              since §22.20 from the frozen copy
#                              crates/fastvideo-lab/fixtures/oracle-script/runpod-matrix.sh
#
# Every script runs under `env -i` (PATH, HOME, LC_ALL=C only): the caller's
# FV_* / RUNPOD_* settings cannot reach the fixtures. Until S8 (§22.20) one
# tier was the script under fixed settings (ADR-0019 §15): `oracle`, the
# family with FV_ORACLE_TARGETS naming its nine tensor-diff targets and the
# FV_GEN_TIMEOUT_S scripts/gpu/oracle.sh passed (5400), for all three
# fixtures, and its extra cells (ADR-0019 §20.3, the ten clip-metric targets)
# the family with FV_ORACLE_TARGETS naming those (oracle-extra-dry-run.txt,
# oracle-extra-gates.tsv); the sibling tier `oracle-r2` was the oracle family
# with FV_ORACLE_TARGETS=sfwan13. All frozen now. A `weights = "volume+r2"`
# tier's body (gold, the sol-bench tiers that fetch; oracle-r2, wan,
# serve-engine, sfstream and mmaudio, frozen) carries its container disk,
# idle limit and RAM floor.
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
# runpod-matrix.sh had no branch left after S8 and was deleted (ADR-0019
# §22.6, §22.20): it must stay gone, or its families' frozen fixtures would
# no longer be the only reference.
[[ ! -e "$MATRIX" ]] || { echo "fv-lab-goldens: $MATRIX is back; it was deleted with its last branch (ADR-0019 §22.20): every family is a frozen tier or retired with no tier" >&2; exit 2; }
# A family's `case` branch in runpod-matrix.sh: a label of the top-level
# `case "$FAMILY" in` (two-space indent; `name)`, `name )`, `a|name)`, a
# trailing comment all count; nested cases' labels do not). With the script
# gone no family has one; the checks below keep the lists' other rules.
has_branch() {
  [[ -f "$MATRIX" ]] || return 1
  # grep reads all of its input (no -q): under pipefail an early exit would
  # SIGPIPE sed and read as "no branch" whenever the branch is found.
  sed -n '/^case "\$FAMILY" in$/,/^esac$/p' "$MATRIX" \
    | grep -E "^  ([A-Za-z0-9_-]+ *\| *)*$1( *\| *[A-Za-z0-9_-]+)* *\)" >/dev/null
}
for f in $FV_LAB_RETIRED; do
  grep -qx "$f" <<<"$FAMILIES" || { echo "fv-lab-goldens: retired family $f is no tier of lab/tiers.toml" >&2; exit 2; }
done
for e in $FV_LAB_SIBLINGS; do
  t="${e%%:*}" f="${e#*:}"
  grep -qx "$t" <<<"$FAMILIES" || { echo "fv-lab-goldens: sibling tier $t is no tier of lab/tiers.toml" >&2; exit 2; }
  has_branch "$f" || { echo "fv-lab-goldens: sibling tier $t's family $f has no runpod-matrix.sh branch" >&2; exit 2; }
  ! has_branch "$t" || { echo "fv-lab-goldens: sibling tier $t has a runpod-matrix.sh branch of its own" >&2; exit 2; }
  ! fv_lab_retired "$t" && ! fv_lab_experiment "$t" || { echo "fv-lab-goldens: sibling tier $t is in another list of fv-lab-retired.sh" >&2; exit 2; }
done
# A sibling frozen with its family (FV_LAB_RETIRED_SIBLINGS, ADR-0019 §22.19):
# a tier, both it and its family retired to fv-lab, neither with a branch, no
# live sibling, and a <K=V> mode (the setting refusals answer with the tier).
for e in $FV_LAB_RETIRED_SIBLINGS; do
  IFS=: read -r t f m <<<"$e"
  [[ "$m" == ?*=?* ]] || { echo "fv-lab-goldens: retired sibling $e has no <tier>:<family>:<K=V> mode" >&2; exit 2; }
  grep -qx "$t" <<<"$FAMILIES" || { echo "fv-lab-goldens: retired sibling tier $t is no tier of lab/tiers.toml" >&2; exit 2; }
  fv_lab_retired "$t" && fv_lab_retired "$f" || { echo "fv-lab-goldens: retired sibling $t or its family $f is not in FV_LAB_RETIRED" >&2; exit 2; }
  ! has_branch "$t" && ! has_branch "$f" || { echo "fv-lab-goldens: retired sibling $t or its family $f has a runpod-matrix.sh branch" >&2; exit 2; }
  ! fv_lab_sibling "$t" >/dev/null || { echo "fv-lab-goldens: $t is both a live and a retired sibling (FV_LAB_SIBLINGS)" >&2; exit 2; }
done
for f in $FV_LAB_EXPERIMENTS; do
  grep -qx "$f" <<<"$FAMILIES" || { echo "fv-lab-goldens: experiment tier $f is no tier of lab/tiers.toml" >&2; exit 2; }
  fv_lab_retired "$f" && { echo "fv-lab-goldens: $f is both an experiment tier and retired (fv-lab-retired.sh)" >&2; exit 2; }
  ! has_branch "$f" || { echo "fv-lab-goldens: experiment tier $f has a runpod-matrix.sh branch" >&2; exit 2; }
done
# Retired with no tier (FV_SCRIPT_RETIRED, ADR-0019 §18): no fixtures, no tier,
# no branch, and in neither other list.
for f in $FV_SCRIPT_RETIRED; do
  ! grep -qx "$f" <<<"$FAMILIES" || { echo "fv-lab-goldens: $f is retired with no tier (FV_SCRIPT_RETIRED) but is a tier of lab/tiers.toml" >&2; exit 2; }
  ! fv_lab_retired "$f" && ! fv_lab_experiment "$f" || { echo "fv-lab-goldens: $f is in FV_SCRIPT_RETIRED and another list of fv-lab-retired.sh" >&2; exit 2; }
  ! has_branch "$f" || { echo "fv-lab-goldens: $f is retired with no tier (FV_SCRIPT_RETIRED) but runpod-matrix.sh still has its branch" >&2; exit 2; }
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
[[ "$TMP" =~ ^[A-Za-z0-9/._-]+$ ]] || { echo "fv-lab-goldens: temp dir $TMP has shell-special characters" >&2; exit 2; }
mkdir -p "$TMP/new"

# A clean environment for every script run.
clean() { env -i PATH="$PATH" HOME="$HOME" LC_ALL=C "$@"; }

FIXTURES=()
FROZEN=()
for FAMILY in $FAMILIES; do
  # The `gold` tier (docs/adr/0023-lab-report-and-ci-bridge.md §3) is no
  # runpod-matrix.sh family: its cells come from lab/gold/ and fv-lab's own dry
  # run is its golden (fixtures/gold-dry-run.txt, tests/gold_tier.rs). Its pod
  # is runpod-http.sh's like every tier's, so only the pod body is written here.
  # The tiers made from a port one-off (fv-lab-retired.sh FV_LAB_EXPERIMENTS,
  # ADR-0019 §14) are the same: no script family, their commands held to the
  # experiment's pod script by fv-lab-experiment-goldens.sh, their dry run
  # fv-lab's own (tests/experiment_tiers.rs).
  # The gold tier's container disk is its own (weights "volume+r2", ADR-0019
  # §16: the largest arm's R2 trees on top of the volume tiers' 120 GB), passed
  # as the script's FV_CONTAINER_DISK_GB, and so is its idle watchdog (an R2
  # fetch keeps the GPU at 0 %: weights::r2_idle_min, a 600 s apt-get plus one
  # fetch at the cell cap, in minutes, plus 15, at least 30; before and after
  # the GPU's first use), and so is its host-RAM floor (min_ram_gb, passed as
  # FV_MIN_RAM_GB); its env (the R2 key as Runpod secret references) is
  # the one difference tests/payload_golden.rs allows.
  # A volume+r2 tier's pod (gold and the sol-bench tiers that fetch; oracle-r2, wan, serve-engine, sfstream and mmaudio frozen): its container
  # disk, idle limit and host-RAM floor as the script's settings.
  tv() { awk -v k="$1" -v n="name = \"$FAMILY\"" '/^\[\[tier\]\]/ { t = 0 } $0 == n { t = 1 } t && $1 == k && $2 == "=" { gsub(/"/, "", $3); print $3; exit }' "$ROOT/lab/tiers.toml"; }
  r2_body=()
  if [[ "$(tv weights)" == volume+r2 ]]; then
    disk="$(tv container_disk_gb)"; cell="$(tv cell_timeout_s)"
    [[ "$disk" =~ ^[0-9]+$ && "$cell" =~ ^[0-9]+$ ]] || { echo "fv-lab-goldens: no container_disk_gb / cell_timeout_s for the $FAMILY tier" >&2; exit 2; }
    (( cell > 0 )) || cell=3600
    idle=$(( (600 + cell + 59) / 60 + 15 )); (( idle >= 30 )) || idle=30
    r2_body=(FV_CONTAINER_DISK_GB="$disk" FV_IDLE_GPU_MIN="$idle" FV_IDLE_GPU_GRACE_MIN="$idle")
    # Its host-RAM floor (min_ram_gb, ADR-0019 §16.10) is the script's
    # FV_MIN_RAM_GB: minRAMPerGPU in both bodies.
    ram="$(tv min_ram_gb)"
    if [[ -n "$ram" ]]; then
      [[ "$ram" =~ ^[1-9][0-9]*$ ]] || { echo "fv-lab-goldens: the $FAMILY tier's min_ram_gb '$ram' is not a positive whole number" >&2; exit 2; }
      r2_body+=(FV_MIN_RAM_GB="$ram")
    fi
  fi
  if [[ "$FAMILY" == gold ]] || fv_lab_experiment "$FAMILY"; then
    extra=(${r2_body[@]+"${r2_body[@]}"})
    # Its wall cap and cell cap as the script's FV_POD_CAP_S / FV_GEN_TIMEOUT_S
    # (runpod-http.sh's defaults, 14400 / 3600, for gold and h3-graphs; the
    # sol-bench tiers' 3900 / 3300, ADR-0019 §23), and a tier without the TAE
    # fetch (tae = false: sol-bench's rule) as the script's FV_SKIP_TAE=1.
    cap="$(tv wall_cap_s)"; gen="$(tv cell_timeout_s)"
    [[ "$cap" =~ ^[0-9]+$ && "$gen" =~ ^[0-9]+$ ]] || { echo "fv-lab-goldens: no wall_cap_s / cell_timeout_s for the $FAMILY tier" >&2; exit 2; }
    extra+=(FV_POD_CAP_S="$cap" FV_GEN_TIMEOUT_S="$gen")
    [[ "$(tv tae)" == false ]] && extra+=(FV_SKIP_TAE=1)
    clean RUNPOD_API_KEY=golden FV_PLAN_VOLUME="$VOLUME" FV_POD_PREFIX=fv-lab FV_FAMILY="$FAMILY" ${extra[@]+"${extra[@]}"} \
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
    # The oracle tier's extra cells (ADR-0019 §20.3): their dry run and gates
    # were the oracle family's with the ten clip-metric targets, frozen with it.
    [[ "$FAMILY" == oracle ]] && FROZEN+=("oracle-extra-dry-run.txt" "oracle-extra-gates.tsv")
    continue
  fi
  # Any other tier would need a runpod-matrix.sh branch to make its goldens
  # from, and the script is gone (ADR-0019 §22.20): a new tier is gold-like
  # (its own dry run, its pod body below) or an experiment tier.
  echo "fv-lab-goldens: tier $FAMILY is neither gold, an experiment tier nor retired (fv-lab-retired.sh); runpod-matrix.sh is gone (ADR-0019 §22.20), so no script makes its goldens" >&2
  exit 2
done

# 3a. The script runner's pod for the oracle-r2 GPU check (ADR-0019 §21.7,
#     §21.8): FV_FAMILY=oracle with FV_R2_TREES=sfwan21-1.3b (the R2 key as the
#     Runpod secret references, r2-stage.sh before the matrix, FV_WORK the
#     merged root's parent), the oracle-r2 tier's disk and cell cap. Written
#     here until oracle's branch went (§22.20): runpod-http.sh plan routes the
#     family to fv-lab now, so it is frozen as the script last wrote it
#     (tests/payload_golden.rs still holds fv-lab's references and idle limit
#     to it).
FROZEN+=("runpod-http-oracle-sfwan13-r2.json")
# The script sides of the serve-engine, sfstream and mmaudio GPU checks
# (ADR-0019 §22.12, §22.14, §22.16: FV_FAMILY=<f> with FV_R2_TREES and the
# tier's disk) were written here until their branches went (§22.19):
# runpod-http.sh plan routes those families to fv-lab now, so the bodies are
# frozen as the script last wrote them (tests/payload_golden.rs still holds
# fv-lab's R2 references, disk and idle limit to them).
FROZEN+=("runpod-http-serve-engine-r2.json" "runpod-http-sfstream-r2.json" "runpod-http-mmaudio-r2.json")
# The wan GPU check's (ADR-0019 §22, S6, §22.18) was the script side's
# FV_FAMILY=wan with FV_R2_TREES naming the two trees its checked arms read
# (fastwan21-1.3b: wan13-dmd, wan13-dmd-dense; fastwan22-ti2v-5b: cli-fw22,
# engine-fw22; kernels-wan reads none), staged at once before the matrix, so
# their 53.41 GB and the 120 GB: 174 GB (the tier's own body, above, has 210
# for fv-lab's drop-then-fetch plan of the whole family), the tier's host-RAM
# floor (FV_MIN_RAM_GB=256, so both pods are placed under the same floor) and
# its cell cap (runpod-http.sh's default FV_GEN_TIMEOUT_S, 3600).
# Written here until wan's branch went (§22.18 "Then", §22.20): frozen too.
FROZEN+=("runpod-http-wan-r2.json")

# 3b. The upstream pods of `fv-lab oracle` (ADR-0019 §19): runpod-http.sh
#     plan-upstream with the settings scripts/gpu/oracle.sh's start_up passed
#     (until it went with the script path, §22.20; fv-lab oracle starts these
#     bodies)
#     (image, disk, FV_EXTRA_ENV, UP_CELL_TIMEOUT_S), for one selection per
#     stack of lab/upstream.toml (UP_STEPS in §15.7's form; the clone commit
#     UP_SHA fixed), and for the fastvideo stack FV_MIN_RAM_GB=192, which
#     oracle.sh did not set but took from its caller's environment (§15.7's
#     command and docs/oracle.md "Upstream cells: host RAM" set it; fv-lab
#     takes it from the stack's min_ram_gb). The sol-ltx25 body carries
#     `env: {HF_TOKEN: "{{ RUNPOD_SECRET_fv_hf_token }}"}` (its weights:ltx25
#     step; fv-lab's `hub_token = true`), the fastvideo body no env.
#     tests/oracle_driver.rs holds fv-lab's upstream pod body to them (the cap
#     watchdog and fv-lab's `env {}` where the script sends none excepted).
UP_SHA_GOLDEN=1234567890abcdef1234567890abcdef12345678
upstream_golden() { # <stack> [K=V …]
  local stack="$1"
  shift
  clean RUNPOD_API_KEY=golden FV_PLAN_VOLUME="$VOLUME" FV_POD_PREFIX=fv-lab UP_SHA="$UP_SHA_GOLDEN" UP_IMAGE_TAG=latest \
    UP_CELL_TIMEOUT_S=7200 "$@" bash "$ROOT/scripts/gpu/runpod-http.sh" plan-upstream "$SHA" "$TAG" \
    | jq -S . >"$TMP/new/runpod-http-upstream-$stack.json"
  FIXTURES+=("runpod-http-upstream-$stack.json")
}
STACKS="$(awk '/^\[\[stack\]\]/ { t = 1; next } t && /^name = "/ { gsub(/^name = "|"$/, ""); print; t = 0 }' "$ROOT/lab/upstream.toml")"
[[ "$(echo $STACKS)" == "fastvideo sol-ltx25" ]] \
  || { echo "fv-lab-goldens: lab/upstream.toml's stacks are '$(echo $STACKS)': give each new stack its golden here" >&2; exit 2; }
upstream_golden fastvideo UP_IMAGE_TARGET=fastvideo FV_CONTAINER_DISK_GB=150 FV_MIN_RAM_GB=192 \
  "UP_STEPS=info:box weights:h3-diffusers oracle:fasth3-4step-vsa" "FV_EXTRA_ENV=FASTVIDEO_DUMP_OPS=0,1,24,47"
upstream_golden sol-ltx25 UP_IMAGE_TARGET=sol-ltx25 FV_CONTAINER_DISK_GB=250 \
  "UP_STEPS=info:box weights:ltx25 oracle:ltx25-512p,ltx25-512p-dense" \
  "FV_EXTRA_ENV=FASTVIDEO_DUMP_OPS=0,1,24,47 RECON_ACCEPT_MISMATCH=1"
# The fastvideo stack making sfwan13 (ADR-0019 §21): its r2:sfwan21-1.3b step
# (the R2 key as Runpod secret references in the body's env) and its
# target's 180 GB disk, as oracle.sh's start_up passed them for sfwan13.
upstream_golden fastvideo-sfwan13 UP_IMAGE_TARGET=fastvideo FV_CONTAINER_DISK_GB=180 FV_MIN_RAM_GB=192 \
  "UP_STEPS=info:box r2:sfwan21-1.3b oracle:sfwan13" "FV_EXTRA_ENV=FASTVIDEO_DUMP_OPS=0,1,24,47"
# The fastvideo stack making the wan tier's two references (ADR-0019 §22, S6):
# their r2:wan22-ti2v-5b and r2:fastwan22-ti2v-5b steps (the R2 key as Runpod
# secret references) and their targets' 210 GB disk, as fv-lab oracle selects
# both oracle cells of the wan tier (§22.18). scripts/gpu/oracle.sh never
# drove these two targets (its fastvideo targets were fasth3-*, sfwan13 and
# h3-ref2va-*): only fv-lab oracle starts this body.
upstream_golden fastvideo-wan22 UP_IMAGE_TARGET=fastvideo FV_CONTAINER_DISK_GB=210 FV_MIN_RAM_GB=192 \
  "UP_STEPS=info:box r2:wan22-ti2v-5b r2:fastwan22-ti2v-5b oracle:wan22-ti2v,fastwan22-ti2v" \
  "FV_EXTRA_ENV=FASTVIDEO_DUMP_OPS=0,1,24,47"

# 4. The frames digest: the script's own function, on a copy of the sample;
#    from the frozen copy of runpod-matrix.sh since the script went (ADR-0019
#    §22.20; fixtures/oracle-script/, its frame unchanged from 7e513f6a).
DIGEST_SRC="$OUT/oracle-script/runpod-matrix.sh"
sed -n '/^frames_digest() {$/,/^}$/p' "$DIGEST_SRC" >"$TMP/digest.sh"
grep -q '^frames_digest() {$' "$TMP/digest.sh" || { echo "fv-lab-goldens: no frames_digest in $DIGEST_SRC" >&2; exit 2; }
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
# The determinism families' hashes and the oracle's compare-dumps lines (D1d,
# ADR-0019 §15): equal in both pass; another hash or another dump line fails.
cp -a "$TMP/rb" "$TMP/da"; cp -a "$TMP/rb" "$TMP/db"
for r in da db; do
  echo 'a frames=1 frames_sha256=x audio_sha256=none' | tee "$TMP/$r/hashes.txt" >"$TMP/$r/determinism.txt"
  echo '{"cell":"a","frames":1,"frames_sha256":"x","audio_sha256":"none"}' >"$TMP/$r/a/hashes.json"
  printf 'compare-dumps latent rel_l2 1.0e-3\n[INFO] took %s s\n' "$RANDOM" >"$TMP/$r/a/stderr.log"
done
clean bash "$CMP" --expect-verdicts 3 "$TMP/da" "$TMP/db" >/dev/null || { echo "fv-lab-goldens: compare-runs failed on equal hashes and dump lines" >&2; rc=1; }
cp -a "$TMP/db" "$TMP/dc"; echo 'a IDENTICAL' >"$TMP/dc/determinism.txt"
cp -a "$TMP/db" "$TMP/dd"; echo 'compare-dumps latent rel_l2 2.0e-3' >"$TMP/dd/a/stderr.log"
cp -a "$TMP/db" "$TMP/de"; rm "$TMP/de/a/hashes.json"
for bad in dc dd de; do
  if clean bash "$CMP" "$TMP/da" "$TMP/$bad" >/dev/null; then echo "fv-lab-goldens: compare-runs missed a hash or dump difference ($bad)" >&2; rc=1; fi
done
# The derived-clip reports (ADR-0019 §20): metrics.txt equal passes; only
# measured values (ffmpeg's SSIM / PSNR form) differing passes and is
# reported; a `missing` line naming each run's own tag is the same; a line
# missing, another order, a probe line differing, an empty or failed value
# differing, or a report in one run only fails.
mrep() { # <run dir> <lines…>: oracle-t-frames/metrics.txt
  local d="$1"; shift
  mkdir -p "$d/oracle-t-frames"; printf '%s\n' "$@" >"$d/oracle-t-frames/metrics.txt"
}
MP='ours: stream|codec_type=video|codec_name=h264|width=768|height=512|r_frame_rate=24/1|nb_frames=121'
MS='ours vs reference clip ssim: SSIM Y:0.973852 (15.823547) U:0.983741 (17.887766) V:0.985163 (18.285787) All:0.977661 (16.509891)'
MQ='ours vs reference clip psnr: PSNR y:38.29 u:44.98 v:45.73 average:39.60 min:35.25 max:inf'
for r in ma mb mc md me mf mg mh mi mj mk ml; do cp -a "$TMP/rb" "$TMP/$r"; done
mrep "$TMP/ma" "$MP" "$MS" "$MQ"; mrep "$TMP/mb" "$MP" "$MS" "$MQ"
clean bash "$CMP" "$TMP/ma" "$TMP/mb" >/dev/null || { echo "fv-lab-goldens: compare-runs failed on equal metrics reports" >&2; rc=1; }
mrep "$TMP/mc" "$MP" "${MS/0.977661/0.977660}" "${MQ/average:39.60/average:39.61}"
out="$(clean bash "$CMP" "$TMP/ma" "$TMP/mc" || echo FAILED)"
{ grep -q '^metrics oracle-t-frames/metrics.txt: the same measurements, values differ' <<<"$out" && ! grep -q FAILED <<<"$out"; } \
  || { echo "fv-lab-goldens: compare-runs judged a metrics report whose values alone differ" >&2; rc=1; }
# A clip missing in both runs: its line names each run's own directory (the tags differ).
mrep "$TMP/mk" 'ours missing /fvscratch/runs/oracle/abc1234-1026120000-beef/oracle-t/frames/output.mp4'
mrep "$TMP/ml" 'ours missing /fvscratch/runs/oracle/abc1234-10261200/oracle-t/frames/output.mp4'
out="$(clean bash "$CMP" "$TMP/mk" "$TMP/ml" || echo FAILED)"
{ grep -qx 'same oracle-t-frames/metrics.txt' <<<"$out" && ! grep -q FAILED <<<"$out"; } \
  || { echo "fv-lab-goldens: compare-runs read a missing clip of two run tags as a difference" >&2; rc=1; }
mrep "$TMP/md" "$MS" "$MQ"                                         # a line missing
mrep "$TMP/me" "$MS" "$MP" "$MQ"                                   # another order
mrep "$TMP/mf" "${MP/nb_frames=121/nb_frames=120}" "$MS" "$MQ"      # a probe line differing
mrep "$TMP/mg" "$MP" 'ours vs reference clip ssim: ' "$MQ"          # an empty value
mrep "$TMP/mh" "$MP" 'ours vs reference clip ssim: Conversion failed!' "$MQ"   # a failed one
mrep "$TMP/mi" "$MP" "${MS/clip ssim/clip psnr}" "$MQ"              # other words
rm "$TMP/mj/oracle-t-frames" -rf                                   # in one run only
for bad in md me mf mg mh mi mj; do
  if clean bash "$CMP" "$TMP/ma" "$TMP/$bad" >/dev/null; then echo "fv-lab-goldens: compare-runs missed a metrics difference ($bad)" >&2; rc=1; fi
done
# The same empty or failed value in both runs is the same report.
cp -a "$TMP/mg" "$TMP/mg2"; cp -a "$TMP/mh" "$TMP/mh2"
clean bash "$CMP" "$TMP/mg" "$TMP/mg2" >/dev/null && clean bash "$CMP" "$TMP/mh" "$TMP/mh2" >/dev/null \
  || { echo "fv-lab-goldens: compare-runs failed on the same empty or failed value in both runs" >&2; rc=1; }
# Control arms (ADR-0019 §15.7): an arm a cell lists in `[repeats] controls`
# (the timed-pick control) is reported, not judged. Cells of their own
# (--cells): r1, r2 and tcx judged, tc a control (a multi-line array), tj a
# control of one cell but an arm another judges (so judged). Runs: equal ones
# pass; a control's frame bytes, hashes.json and lines of determinism.txt /
# hashes.txt (`tc/run1 ...`, `tc: no run1 hashes`) differing pass, with those
# two files still counted (6 = the two and the judged arms' hashes.json: r1,
# r2, tcx, tj). Fail: a judged arm's frames differing (kd), a control in one
# run only (ke), another control exit (kf), `tj` differing (kg), tcx's lines
# differing with equal frames (kt: `tc` is no prefix match), a control's
# frame count differing (kq).
mkdir -p "$TMP/kcells"
cat >"$TMP/kcells/k.toml" <<'EOF'
name = "k"
[[arm]]
name = "r1"
env = [
  "X=1",
]
[[arm]]
name = "r2"
[[arm]]
name = "tc"
[[arm]]
name = "tcx"
[[arm]]
name = "tj"
[repeats]
hash = "arm"
reference = "r1"
judged = ["r1", "r2", "tcx"]
controls = [
  "tc", # the timed pick
  "tj",
]
EOF
printf '%s\n' 'name = "j"' '[[arm]]' 'name = "tj"' >"$TMP/kcells/j.toml"
kmk() { # dir tc-byte tj-byte r2-byte [tcx's hash in the two reports]
  local a b
  for a in r1 tc tcx tj r2; do
    case "$a" in tc) b="$2" ;; tj) b="$3" ;; r2) b="$4" ;; *) b=x ;; esac
    mk "$1" "$a" 0 "$b"
    echo "{\"cell\":\"$a\",\"frames\":1,\"frames_sha256\":\"$b\"}" >"$1/$a/hashes.json"
    [[ "$a" == tcx ]] && b="${5:-x}"
    echo "$a frames=1 frames_sha256=$b audio_sha256=none" >>"$1/hashes.txt"
    echo "$a/run1 frames IDENTICAL $b" >>"$1/determinism.txt"
  done
}
kmk "$TMP/ka" x x x; kmk "$TMP/kb" x x x; kmk "$TMP/kc" y x x; kmk "$TMP/kd" x x y; kmk "$TMP/kg" x y x
kmk "$TMP/kt" x x x z
echo 'tc: no run1 hashes' >>"$TMP/kc/determinism.txt"
cp -a "$TMP/kb" "$TMP/ke"; rm -rf "$TMP/ke/tc"
cp -a "$TMP/kb" "$TMP/kf"; mk "$TMP/kf" tc 3 x
cp -a "$TMP/kb" "$TMP/kq"; printf x >"$TMP/kq/tc/frames/frame-001.png"; (. "$TMP/digest.sh" && frames_digest "$TMP/kq/tc" tc)
for good in kb kc; do
  clean bash "$CMP" --cells "$TMP/kcells" --expect-verdicts 6 "$TMP/ka" "$TMP/$good" >/dev/null || { echo "fv-lab-goldens: compare-runs failed on runs whose control arm alone differs ($good)" >&2; rc=1; }
done
out="$(clean bash "$CMP" --cells "$TMP/kcells" "$TMP/ka" "$TMP/kc" || true)"
grep -qx 'control tc: reported, not judged ({"exit":0,"skipped":null}, frames differ (1 vs 1 frames))' <<<"$out" \
  || { echo "fv-lab-goldens: compare-runs did not report the control arm's frames" >&2; rc=1; }
if clean bash "$CMP" --cells "$TMP/kcells" --expect-verdicts 7 "$TMP/ka" "$TMP/kc" >/dev/null; then echo "fv-lab-goldens: compare-runs counted a control's hashes.json as a verdict file" >&2; rc=1; fi
for bad in kd ke kf kg kt kq; do
  if clean bash "$CMP" --cells "$TMP/kcells" "$TMP/ka" "$TMP/$bad" >/dev/null; then echo "fv-lab-goldens: compare-runs missed a difference beside a control arm ($bad)" >&2; rc=1; fi
done
# A determinism.txt of control lines only, in both runs: neither the same nor
# counted (no verdict file at all, so --expect-verdicts 1 fails).
for r in ko kp; do mk "$TMP/$r" r1 0 x; done
mk "$TMP/ko" tc 0 x; mk "$TMP/kp" tc 0 y
echo 'tc/run1 frames IDENTICAL x' >"$TMP/ko/determinism.txt"; echo 'tc/run1 frames DIFFERENT y' >"$TMP/kp/determinism.txt"
out="$(clean bash "$CMP" --cells "$TMP/kcells" "$TMP/ko" "$TMP/kp" || echo FAILED)"
{ grep -qx 'determinism.txt: only control lines (not compared, not counted)' <<<"$out" && ! grep -q FAILED <<<"$out"; } \
  || { echo "fv-lab-goldens: compare-runs mishandled a determinism.txt of control lines only" >&2; rc=1; }
if clean bash "$CMP" --cells "$TMP/kcells" --expect-verdicts 1 "$TMP/ko" "$TMP/kp" >/dev/null; then echo "fv-lab-goldens: compare-runs counted a determinism.txt of control lines only" >&2; rc=1; fi
# An unreadable determinism.txt on both sides is a difference, as without
# controls (only where mode 000 stops a read: not as root).
cp -a "$TMP/ka" "$TMP/kua"; cp -a "$TMP/kb" "$TMP/kub"; chmod 000 "$TMP/kua/determinism.txt" "$TMP/kub/determinism.txt"
if [[ ! -r "$TMP/kua/determinism.txt" ]] && clean bash "$CMP" --cells "$TMP/kcells" "$TMP/kua" "$TMP/kub" >/dev/null 2>&1; then
  echo "fv-lab-goldens: compare-runs passed two unreadable determinism.txt" >&2; rc=1
fi
chmod 644 "$TMP/kua/determinism.txt" "$TMP/kub/determinism.txt"
# Without --cells the checkout's lab/cells decide: det-short-h3's h3-timed is a control.
for r in ha hb; do mk "$TMP/$r" h3-r1 0 x; done
mk "$TMP/ha" h3-timed 0 x; mk "$TMP/hb" h3-timed 0 y
clean bash "$CMP" "$TMP/ha" "$TMP/hb" >/dev/null || { echo "fv-lab-goldens: compare-runs judged h3-timed, a control of lab/cells/det-short-h3.toml" >&2; rc=1; }
# A controls array it cannot read is a usage error, not a pass.
mkdir -p "$TMP/kbad"; printf '%s\n' '[repeats]' 'controls = ["tc"' >"$TMP/kbad/k.toml"
clean bash "$CMP" --cells "$TMP/kbad" "$TMP/ka" "$TMP/kb" >/dev/null 2>&1 && st=0 || st=$?
[[ "$st" == 2 ]] || { echo "fv-lab-goldens: compare-runs exited $st (not 2) on an unclosed controls array" >&2; rc=1; }
# So is a name outside fv-lab's arm-name grammar.
mkdir -p "$TMP/kname"; printf '%s\n' '[repeats]' 'controls = ["a b"]' >"$TMP/kname/k.toml"
clean bash "$CMP" --cells "$TMP/kname" "$TMP/ka" "$TMP/kb" >/dev/null 2>&1 && st=0 || st=$?
[[ "$st" == 2 ]] || { echo "fv-lab-goldens: compare-runs exited $st (not 2) on a control named 'a b'" >&2; rc=1; }
for bad in rc rd re rf rh ri; do
  if clean bash "$CMP" "$TMP/ra" "$TMP/$bad" >/dev/null; then echo "fv-lab-goldens: compare-runs missed a difference ($bad)" >&2; rc=1; fi
done
# A file only in the runpod-http run (the second run) when the fv-lab run has
# no file of a lister's later glob (no gate report, no hashes.json, no
# metrics.txt): a difference too. The listers' loops once ended on that glob's
# failed test, which under `set -e` stopped before the second run's list, so
# such a file went unseen (ADR-0019 §22.18, found by its simulation, S6).
cp -a "$TMP/rb" "$TMP/xa"; rm -rf "$TMP/xa/gate"; cp -a "$TMP/xa" "$TMP/xb"
rm "$TMP/xa/compare/compare-clips-a--s.json"
cp -a "$TMP/xa" "$TMP/xc"; cp -a "$TMP/xa" "$TMP/xd"; cp -a "$TMP/xa" "$TMP/xe"
echo '{"cell":"c","verdict":"pass","why":""}' >"$TMP/xc/fam-c.verdict.json"
mkdir -p "$TMP/xd/oracle-t-frames"; echo 'ours: missing x' >"$TMP/xd/oracle-t-frames/metrics.txt"
mkdir -p "$TMP/xe/z"; echo '{"cell":"z","exit":0,"seconds":1}' >"$TMP/xe/z/summary.json"
for second in xb xc xd xe; do
  if clean bash "$CMP" "$TMP/xa" "$TMP/$second" >/dev/null; then
    echo "fv-lab-goldens: compare-runs missed a file only in the runpod-http run ($second)" >&2; rc=1
  fi
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
