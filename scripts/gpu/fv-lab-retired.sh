# shellcheck shell=bash
# Sourced, not run. The runpod-matrix.sh families retired to fv-lab
# (docs/adr/0019-fv-lab.md §8, §9.5, §10, §11.6 for b4a, the first with
# a verdict step, §15.10 for det-short and determinism, the [repeats]
# families, and §17.6 for fuse, attn3 and ltx, D1f's): each passed its GPU
# gate (one
# `fv-lab pod up --wait` run and one `runpod-http.sh run` of the same commit,
# `fv-lab-compare-runs.sh` exit 0), so its `case` branch is gone from
# runpod-matrix.sh and the tier lives only in lab/ (lab/tiers.toml,
# lab/cells/<tier>-*.toml).
#
# Read by:
#   runpod-http.sh             `run` / `plan` for one of these call fv-lab pod up
#                              (FV_SCRIPT_RETIRED below: refused)
#   runpod.sh                  `gpu` / `matrix`: fv-lab run for these, refused for
#                              FV_SCRIPT_RETIRED
#   tests/fv-lab-goldens.sh    their fixtures are frozen (no longer regenerated
#                              from the scripts, only checked to be present)
#   crates/fastvideo-lab/tests/dry_run.rs   the same list, the same checks
#
# A family joins this list in the PR that deletes its branch, never before.
# The restored modes below join it in the PR that restores them: their branch
# was already gone, so they start frozen.
FV_LAB_RETIRED="rtx6000 techniques b6 rtx5090 fastvideo h3arms trace ltxfps ltxvae precision precision-nvfp4 ltxvae-gen b4a det-short determinism fuse attn3 ltx"

# Script modes restored as tiers (ADR-0019 §12, owner decision 2026-10-25):
# `<tier>:<family>:<K=V>` is the mode `FV_EXTRA_ENV=<K=V> runpod-matrix.sh
# <family>` ran before #185 deleted the branch. The tier's fixtures are that
# mode's dry run, gates and `runpod-http.sh plan` body as the script wrote them
# at 5987e6a4 (scripts/gpu/tests/fv-lab-restored-goldens.sh); fv-lab's equal
# them but for the family's name (the tier's) and the mode's setting (none).
# runpod-http.sh refuses <K=V> in FV_EXTRA_ENV and names the tier.
FV_LAB_RESTORED="precision-nvfp4:precision:FV_PRECISION_ARM=ltx-nvfp4 ltxvae-gen:ltxvae:FV_VAE_GEN=1"

# Tiers made from a port one-off of scripts/gpu/experiments/ (ADR-0019 §14,
# D1e; the directory is gone since h3-graphs' GPU check, §14.4, and its pod
# script is read from the history): never runpod-matrix.sh families. Their
# commands are the experiment pod script's own, cut from it by
# tests/fv-lab-experiment-goldens.sh; their dry
# run is fv-lab's own golden (fixtures/<tier>-dry-run.txt) and their pod body
# runpod-http.sh plan's, like the `gold` tier's. runpod-http.sh refuses `run`
# and `all` for them and names the fv-lab command.
FV_LAB_EXPERIMENTS="h3-graphs"

# Sibling tiers of a live runpod-matrix.sh family (ADR-0019 §20.5, §21, §22):
# `<tier>:<family>` is a tier whose cells are targets of <family>'s branch
# that its own tier cannot run (another pod: `oracle-r2`, the oracle family's
# sfwan13 target, reads its tree from R2, `weights = "volume+r2"`; another
# prompt set: `hd5`, the hd family with FV_PROMPTS=scripts/gpu/prompts-hd5.json,
# the 1080P serve gate's, which fv-lab takes from a cell, §22 S2). Its
# fixtures are <family>'s dry run under the tier's settings
# (tests/fv-lab-goldens.sh fam_env) and runpod-http.sh plan's body for the
# tier; fv-lab's dry run equals the first with the tier's name for the
# family's and its merged weights root for the volume's
# (crates/fastvideo-lab/tests/dry_run.rs). No branch of its own:
# runpod-http.sh refuses `run` / `all` for it and names fv-lab's command; the
# script runs those targets as <family> (FV_FAMILY=<family>, with
# fv_lab_sibling_mode's settings). A sibling of a family that retires is
# frozen with it (FV_LAB_RETIRED, ADR-0019 §10).
FV_LAB_SIBLINGS="oracle-r2:oracle hd5:hd"

# The runpod-matrix.sh families retired with no tier (ADR-0019 §18, D1g,
# coordinator decision 2026-10-26): finished experiments and one-time
# measurements whose evidence is committed or recorded, families the EU volume
# cannot run, and ones another tier covers. Their `case` branches are deleted
# and nothing runs them: runpod-matrix.sh, runpod-http.sh run / all / plan and
# runpod.sh gpu / matrix refuse them (exit 2) with fv_script_retired_why's
# pointer, never route them to fv-lab (they are no tier of lab/tiers.toml).
# A family joins this list in the PR that deletes its branch.
FV_SCRIPT_RETIRED="h3 hunyuan sana-video b200 writer precision-debug identity b4b b4c b4c2 h3attn"

# fv_lab_retired <family>: true when the family is fv-lab only.
fv_lab_retired() {
  [[ " $FV_LAB_RETIRED " == *" ${1:-} "* ]]
}

# fv_lab_experiment <tier>: true for a tier made from a port one-off.
fv_lab_experiment() {
  [[ " $FV_LAB_EXPERIMENTS " == *" ${1:-} "* ]]
}

# fv_lab_sibling <tier>: the live family a sibling tier is cut from (empty
# and false for any other name).
fv_lab_sibling() {
  local e
  for e in $FV_LAB_SIBLINGS; do
    [[ "${e%%:*}" == "${1:-}" ]] && { echo "${e#*:}"; return 0; }
  done
  return 1
}

# fv_lab_sibling_mode <tier>: the settings under which the script runs a
# sibling tier's cells as its family (FV_FAMILY=<family> plus these), for the
# refusals; tests/fv-lab-goldens.sh makes the tier's dry-run golden under them.
fv_lab_sibling_mode() {
  case "${1:-}" in
    oracle-r2) echo "FV_ORACLE_TARGETS=sfwan13" ;;
    hd5) echo "FV_PROMPTS=/opt/fastvideo-rs/scripts/gpu/prompts-hd5.json" ;;
  esac
}

# fv_script_retired <family>: true when the family is retired with no tier.
fv_script_retired() {
  [[ " $FV_SCRIPT_RETIRED " == *" ${1:-} "* ]]
}

# fv_script_retired_why <family>: what replaces a family retired with no tier
# (ADR-0019 §18), for the refusals; empty for any other name.
fv_script_retired_why() {
  case "${1:-}" in
    h3) echo "the rtx6000 tier runs every arm but the refiner-less Spark fallback and the cold streamed-encoder runs, which nothing serves (ADR-0019 §17.4, §18): fv-lab pod up --family rtx6000, or FV_FAMILY=rtx6000 runpod-http.sh run; the h3 commands: git show b019ccb2:scripts/gpu/runpod-matrix.sh" ;;
    b4b | b4c | b4c2) echo "a finished experiment: its GPU gate passed when its change merged (b4b #179, b4c #177, b4c2 #178; artifacts/runpod/$1/); a later change to the fusions or the prompt cache gets its own byte-identity run (the b4a / techniques tiers' baseline-vs-this-build shape) and the CPU fixtures (rt/prompt_cache/equality.rs, rt/fuse/equality.rs) (ADR-0019 §17.3, §18)" ;;
    hunyuan) echo "nothing runs it: its tree is not on the EU volume and no variant or gold file serves it; its arm by hand: git show b019ccb2:scripts/gpu/runpod-matrix.sh (ADR-0019 §14.7, §18)" ;;
    sana-video) echo "nothing runs it: its tree is not on the EU volume and no variant or gold file serves it; its arms by hand: docs/ports/sana-video.md, or git show b019ccb2:scripts/gpu/runpod-matrix.sh (ADR-0019 §14.7, §18)" ;;
    b200) echo "the US volume it ran on is gone (CLAUDE.md, owner decision 2026-10-06) (ADR-0019 §14.7, §18)" ;;
    writer) echo "a one-time measurement of the frame writer (commit a8ecfec6); its commands: git show b019ccb2:scripts/gpu/runpod-matrix.sh (ADR-0019 §14.7, §18)" ;;
    precision-debug) echo "the precision tiers run the same precision variants, but not its per-step CUPTI trace (FASTVIDEO_GPU_TRACE_STEP) or its FASTVIDEO_DUMP_DIR + compare-dumps diffs; those by hand: git show b019ccb2:scripts/gpu/runpod-matrix.sh (ADR-0019 §14.7, §18)" ;;
    identity) echo "the det-short and determinism tiers cover it (ADR-0019 §14.7, §18)" ;;
    h3attn) echo "its arms are measured and recorded (docs/techniques.md \"H3 attention arms\"; the served variants' gold files); no attention arm of it is open (ADR-0019 §14.7, §18)" ;;
  esac
}
