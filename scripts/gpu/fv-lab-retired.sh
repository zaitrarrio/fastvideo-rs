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

# fv_lab_retired <family>: true when the family is fv-lab only.
fv_lab_retired() {
  [[ " $FV_LAB_RETIRED " == *" ${1:-} "* ]]
}

# fv_lab_experiment <tier>: true for a tier made from a port one-off.
fv_lab_experiment() {
  [[ " $FV_LAB_EXPERIMENTS " == *" ${1:-} "* ]]
}
