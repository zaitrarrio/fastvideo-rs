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

# fv_script_retired <family>: true when the family is retired with no tier.
fv_script_retired() {
  [[ " $FV_SCRIPT_RETIRED " == *" ${1:-} "* ]]
}

# fv_script_retired_why <family>: what replaces a family retired with no tier
# (ADR-0019 §18), for the refusals; empty for any other name.
fv_script_retired_why() {
  case "${1:-}" in
    h3) echo "the rtx6000 tier runs its arms (ADR-0019 §17.4, §18): fv-lab pod up --family rtx6000, or FV_FAMILY=rtx6000 runpod-http.sh run" ;;
    b4b | b4c | b4c2) echo "a finished experiment: its GPU gate passed when its change merged (b4b #179, b4c #177, b4c2 #178; artifacts/runpod/$1/); a later change to the fusions or the prompt cache gets its own byte-identity run (the b4a / techniques tiers' baseline-vs-this-build shape) and the CPU fixtures (rt/prompt_cache/equality.rs, rt/fuse/equality.rs) (ADR-0019 §17.3, §18)" ;;
    hunyuan | sana-video) echo "its trees are not on the EU volume and nothing serves it; the gold files and served variants decide what runs (fv-lab pod up --family gold) (ADR-0019 §14.7, §18)" ;;
    b200) echo "the US volume it ran on is gone (CLAUDE.md, owner decision 2026-10-06) (ADR-0019 §14.7, §18)" ;;
    writer) echo "a one-time measurement of the frame writer (commit a8ecfec6); its commands: git show b019ccb2:scripts/gpu/runpod-matrix.sh (ADR-0019 §14.7, §18)" ;;
    precision-debug) echo "the precision and precision-nvfp4 tiers cover it (ADR-0019 §14.7, §18)" ;;
    identity) echo "the det-short and determinism tiers cover it (ADR-0019 §14.7, §18)" ;;
    h3attn) echo "its arms are measured and recorded (docs/techniques.md \"H3 attention arms\"; the served variants' gold files); no attention arm of it is open (ADR-0019 §14.7, §18)" ;;
  esac
}
