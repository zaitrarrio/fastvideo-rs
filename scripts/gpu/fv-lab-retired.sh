# shellcheck shell=bash
# Sourced, not run. The runpod-matrix.sh families retired to fv-lab
# (docs/adr/0019-fv-lab.md §8, §9.5, §10): each passed its GPU gate (one
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
FV_LAB_RETIRED="rtx6000 techniques b6 rtx5090 fastvideo h3arms trace ltxfps ltxvae precision"

# fv_lab_retired <family>: true when the family is fv-lab only.
fv_lab_retired() {
  [[ " $FV_LAB_RETIRED " == *" ${1:-} "* ]]
}
