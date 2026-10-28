#!/usr/bin/env bash
# The fv-lab route of the retired families (docs/adr/0019-fv-lab.md §10),
# without a network, a key or a GPU:
#   - runpod-http.sh `run` / `plan` for a retired family: the settings mapped
#     across, the ones fv-lab would not honour refused (not dropped), the
#     dropped modes' knobs refused in FV_EXTRA_ENV (the two restored modes
#     naming their tier, ADR-0019 §12), lists split without globbing, and a
#     fake FV_LAB_BIN receiving exactly the command;
#   - `gold`, the experiment tiers (h3-graphs, ADR-0019 §14) and the sibling
#     tier (oracle-r2, §21) refused for run / all (their plan still the
#     script's body);
#   - hd (with its sibling hd5), serve-engine, sfstream and mmaudio, retired
#     to fv-lab after their GPU checks (§22.19, S7): routed, `all` refused,
#     fv-lab's plan, refused by runpod-matrix.sh; their knobs
#     (FV_ENGINE_CELLS, FV_SFSTREAM_RUNS, FV_SF_PROMPT, FV_SF_SWITCH,
#     FV_MMAUDIO_PROMPT) and the script side's FV_R2_TREES refused;
#   - wan (§22, S6) on the script path, its GPU check's FV_R2_TREES form
#     planned; its dry run with each gated arm's FASTVIDEO_TAE_DIR, no
#     --tae-dir, no TAE fetch, and sfwan's (its SF-Wan arms) the same;
#   - hd's dropped post-script hook (FV_HD_POST_URL, FV_HD_POST_TIMEOUT_S,
#     ADR-0019 §22, S2) refused by runpod-http.sh (script path and fv-lab
#     route);
#   - FV_POD_FILES / FV_POD_SCRIPT, gone with sol-bench-pod.sh (ADR-0019
#     §23.10), refused by runpod-http.sh on every path, naming the
#     sol-bench-<set> tiers;
#   - the families retired with no tier (fv-lab-retired.sh FV_SCRIPT_RETIRED,
#     ADR-0019 §18 and §22's S1, `headline` among them) refused by
#     runpod-http.sh run / all / plan (dry run or not, before the key) and by
#     runpod-matrix.sh, exit 2, naming what replaces each; never routed to
#     fv-lab;
#   - the usage texts print their whole header.
#
#   bash scripts/gpu/tests/fv-lab-route.test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HTTP="$ROOT/scripts/gpu/runpod-http.sh"
MATRIX="$ROOT/scripts/gpu/runpod-matrix.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
ok() { printf 'ok   %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1" >&2; fail=1; }
# A clean environment: the caller's FV_* / RUNPOD_* never reach a case.
clean() { env -i PATH="$PATH" HOME="$HOME" LC_ALL=C "$@"; }

# expect <name> <exit> <text in the output> -- <env K=V ...> <args ...>
expect() {
  local name="$1" want="$2" text="$3" out rc=0
  shift 4
  out="$(cd "$TMP" && clean "$@" 2>&1)" || rc=$?
  if [[ "$rc" == "$want" && "$out" == *"$text"* ]]; then ok "$name"; else bad "$name: exit $rc (want $want), output: $out"; fi
}
dry() { # dry <name> <exit> <text> <env...>: FV_DRY_RUN=1 runpod-http.sh run abc1234
  local name="$1" want="$2" text="$3"
  shift 3
  expect "$name" "$want" "$text" -- FV_DRY_RUN=1 "$@" bash "$HTTP" run abc1234
}

# Files a glob would match in the working directory of the cases.
touch "$TMP/aaa" "$TMP/bbb"

# 1. Mapped across, and what equals fv-lab's own values passes.
dry "retired run: the fv-lab command" 0 \
  "fv-lab pod up --sha abc1234 --family b6 --root $ROOT --cells b6-solh3-tree-base b6-solh3-tree-pr --baseline-sha e93e357 --extra-env FASTVIDEO_X=1 --lpips --yes --wait" \
  FV_FAMILY=b6 FV_CELLS="b6-solh3-tree-base,b6-solh3-tree-pr" FV_BASELINE_SHA=e93e357 FV_EXTRA_ENV=FASTVIDEO_X=1 FV_LPIPS=1
dry "the default family is rtx6000, fv-lab only" 0 "--family rtx6000 --root $ROOT --yes --wait"
dry "FV_POD_CAP_S below the tier's: --wall-cap-s" 0 "--wall-cap-s 1800 --yes --wait" FV_FAMILY=precision FV_POD_CAP_S=1800
dry "fv-lab's own values pass" 0 "--family trace --root $ROOT --yes --wait" FV_FAMILY=trace \
  FV_POD_CAP_S=14400 FV_GEN_TIMEOUT_S=3600 FV_CONTAINER_DISK_GB=120 RUNPOD_CLOUD_TYPE=SECURE \
  RUNPOD_GPU_TYPE="NVIDIA RTX PRO 6000 Blackwell Server Edition" RUNPOD_VOLUME_ID=bz0ofkfx45 \
  RUNPOD_VOLUME_NAME=fv-weights-h3-ltx-500 FV_IDLE_GPU_MIN=10 FV_IDLE_GPU_GRACE_MIN=30 \
  FV_IDLE_NET_BYTES=262144 FV_MIN_DRIVER=580 FV_BOOT_WAIT_S=600 FV_POD_PREFIX=fv-lab \
  RUNPOD_IMAGE=ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-abc1234 RUNPOD_GPU_MAX_DPH=5 FV_SKIP_TAE=0
dry "lists split without globbing" 0 "--cells sol-h3 \\* --extra-env FASTVIDEO_X=1 \\* --yes" \
  FV_FAMILY=rtx6000 FV_CELLS="sol-h3 *" FV_EXTRA_ENV="FASTVIDEO_X=1 *"

# 1. Refused: no equivalent, or not what fv-lab would do.
dry "the reviewer's case" 1 "which has no FV_AVOID_MACHINES" FV_FAMILY=precision FV_POD_CAP_S=1800 \
  RUNPOD_VOLUME_ID=other RUNPOD_CLOUD_TYPE=COMMUNITY FV_AVOID_MACHINES=m1 FV_SKIP_TAE=1
for kv in FV_POD_FILE=x FV_PROMPTS=5 FV_PROMPT=x FV_SEED=1 FV_AVOID_MACHINES=m1 \
  FV_CREATE_WAIT_S=60 RUNPOD_ALLOWED_CUDA=13.0 FV_MIN_RAM_GB=192; do
  dry "refused: $kv" 1 "which has no ${kv%%=*}" FV_FAMILY=trace "$kv"
done
# FV_POD_FILES / FV_POD_SCRIPT went with sol-bench-pod.sh (ADR-0019 §23.10):
# refused before anything else, on the fv-lab route, the script path and a plan.
for kv in FV_POD_FILES=x FV_POD_SCRIPT=sol-bench-pod.sh; do
  why="${kv%%=*} went with scripts/gpu/sol-bench-pod.sh, its only user (docs/adr/0019-fv-lab.md §23.10): the benchmark's sets are the fv-lab tiers sol-bench-<set>"
  dry "refused (gone): $kv, fv-lab route" 1 "$why" FV_FAMILY=trace "$kv"
  dry "refused (gone): $kv, script path" 1 "$why" FV_FAMILY=wan "$kv"
  expect "refused (gone): $kv, plan" 1 "$why" -- RUNPOD_API_KEY=t FV_PLAN_VOLUME="bz0ofkfx45 EU-RO-1" FV_FAMILY=wan \
    "$kv" bash "$HTTP" plan abc1234 abc1234-10261200
done
# A tier without a floor: FV_MIN_RAM_GB refused, naming the tier's min_ram_gb
# (ADR-0019 §16.10).
dry "refused: FV_MIN_RAM_GB names the tier's min_ram_gb" 1 "a pod's host-RAM floor is its tier's min_ram_gb in lab/tiers.toml" \
  FV_FAMILY=trace FV_MIN_RAM_GB=192
# FV_MIN_RAM_GB on the script path: minRAMPerGPU in the body; anything but a
# whole positive number of GB stops before jq or the API.
plan_env=(RUNPOD_API_KEY=t FV_PLAN_VOLUME="bz0ofkfx45 EU-RO-1" FV_FAMILY=oracle)
expect "plan: FV_MIN_RAM_GB=192 is minRAMPerGPU" 0 '"minRAMPerGPU": 192' -- \
  "${plan_env[@]}" FV_MIN_RAM_GB=192 bash "$HTTP" plan abc1234 abc1234-10261200
for v in 0 012 12.5 abc -1 "1 2"; do
  expect "refused: FV_MIN_RAM_GB='$v'" 1 "FV_MIN_RAM_GB is the pod's minRAMPerGPU in whole GB" -- \
    "${plan_env[@]}" FV_MIN_RAM_GB="$v" bash "$HTTP" plan abc1234 abc1234-10261200
done
# FV_FETCH_TREE=1 is fv-lab's --fetch-tree (ADR-0019 §13); its settings act only with it.
dry "FV_FETCH_TREE=1: --fetch-tree" 0 "--family trace --root $ROOT --fetch-tree --fetch-skip /frames/ --fetch-max-mb 10 --yes --wait" \
  FV_FAMILY=trace FV_FETCH_TREE=1 FV_FETCH_SKIP=/frames/ FV_FETCH_MAX_MB=10
expect "plan: FV_FETCH_TREE=1 fetches nothing" 0 "--family trace --root $ROOT --tag abc1234-1025120000-abcd" -- \
  FV_DRY_RUN=1 FV_FAMILY=trace FV_FETCH_TREE=1 bash "$HTTP" plan abc1234 abc1234-1025120000-abcd
for kv in FV_FETCH_SKIP=png FV_FETCH_MAX_MB=10; do
  dry "refused without the tree: $kv" 1 "reads ${kv%%=*} only with FV_FETCH_TREE=1" FV_FAMILY=trace "$kv"
done
for mb in 010 0x10 1e3 -1; do
  dry "refused: FV_FETCH_MAX_MB=$mb (not decimal)" 1 "as a decimal number of MiB" FV_FAMILY=trace FV_FETCH_TREE=1 FV_FETCH_MAX_MB="$mb"
done
for kv in FV_KEEP_POD=1 RUNPOD_NO_VOLUME=1; do
  dry "refused: $kv" 1 "which has no $kv" FV_FAMILY=trace "$kv"
done
for kv in FV_GEN_TIMEOUT_S=900 FV_CONTAINER_DISK_GB=200 RUNPOD_CLOUD_TYPE=COMMUNITY RUNPOD_GPU_TYPE=H100 \
  FV_IDLE_GPU_MIN=0 FV_IDLE_GPU_GRACE_MIN=5 FV_IDLE_NET_BYTES=1 FV_MIN_DRIVER=570 FV_BOOT_WAIT_S=1200 \
  RUNPOD_VOLUME_ID=other RUNPOD_VOLUME_NAME=other FV_PLAN_VOLUME="x y" RUNPOD_IMAGE=other FV_POD_PREFIX=fv; do
  dry "refused: $kv" 1 "not ${kv#*=}; unset it" FV_FAMILY=trace "$kv"
done
dry "refused: FV_SKIP_TAE=1 on a TAE tier" 1 "FV_SKIP_TAE=1" FV_FAMILY=rtx5090 FV_SKIP_TAE=1
dry "refused: a lower price cap" 1 "RUNPOD_GPU_MAX_DPH=2" FV_FAMILY=trace RUNPOD_GPU_MAX_DPH=2
dry "refused: FV_POD_CAP_S above the tier's" 1 "never raised" FV_FAMILY=trace FV_POD_CAP_S=20000
dry "refused: FV_POD_CAP_S not a number" 1 "never raised" FV_FAMILY=trace FV_POD_CAP_S=1h
# 2. The dropped modes' knobs (ADR-0019 §9.4) in FV_EXTRA_ENV.
for k in FV_PRECISION_ARM FV_NVFP4_WORKLOADS FV_VAE_TRACE FV_VAE_UNTRACED FV_VAE_WORKLOADS \
  FV_LTXFPS FV_LTXFPS_FRAMES FV_TECH_CELLS FV_B6_CELLS FV_B4A_CELLS FV_DET_REPEATS FV_DET_PROMPT FV_DET_T2V_PROMPT \
  FV_DET_I2V_PROMPT FV_ATTN3_ARMS FV_ATTN3_VSA FV_ENGINE_CELLS FV_SFSTREAM_RUNS FV_SF_PROMPT FV_SF_SWITCH \
  FV_MMAUDIO_PROMPT FV_PROMPTS FV_GEN_TIMEOUT_S FV_CELLS; do
  dry "refused in FV_EXTRA_ENV: $k" 1 "reads no $k from FV_EXTRA_ENV" FV_FAMILY=precision FV_EXTRA_ENV="FASTVIDEO_X=1 $k=1"
done
# The two modes restored as tiers (ADR-0019 §12): refused, naming the tier, on
# any retired family (not routed: the run directory and pod name are the tier's).
for fam in precision ltxvae rtx6000; do
  dry "restored mode named: FV_PRECISION_ARM=ltx-nvfp4 ($fam)" 1 \
    "it is the fv-lab tier precision-nvfp4 now: use FV_FAMILY=precision-nvfp4" \
    FV_FAMILY="$fam" FV_EXTRA_ENV="FASTVIDEO_X=1 FV_PRECISION_ARM=ltx-nvfp4"
  dry "restored mode named: FV_VAE_GEN=1 ($fam)" 1 "it is the fv-lab tier ltxvae-gen now: use FV_FAMILY=ltxvae-gen" \
    FV_FAMILY="$fam" FV_EXTRA_ENV="FV_VAE_GEN=1"
done
for k in FV_NVFP4_WORKLOADS FV_VAE_WORKLOADS FV_VAE_UNTRACED; do
  dry "workload knob: narrow by FV_CELLS ($k)" 1 "narrow them by arm with FV_CELLS" \
    FV_FAMILY=ltxvae-gen FV_EXTRA_ENV="$k=4k5s"
done
dry "FV_VAE_GEN=0 is no mode: the generic refusal" 1 "reads no FV_VAE_GEN from FV_EXTRA_ENV" \
  FV_FAMILY=ltxvae FV_EXTRA_ENV="FV_VAE_GEN=0"
# The mode's setting is answered first, before any other refusal of the same
# call (here a refused FV_VAE_TRACE before it, and FV_AVOID_MACHINES).
dry "restored mode named before other refusals" 1 "it is the fv-lab tier ltxvae-gen now: use FV_FAMILY=ltxvae-gen" \
  FV_FAMILY=ltxvae FV_EXTRA_ENV="FV_VAE_TRACE=0 FV_VAE_GEN=1" FV_AVOID_MACHINES=m1
# Several refused keys: all of them in one message.
dry "every refused key in one message" 1 "reads no FV_VAE_TRACE, FV_TECH_CELLS, FV_VAE_WORKLOADS from FV_EXTRA_ENV" \
  FV_FAMILY=ltxvae FV_EXTRA_ENV="FV_VAE_TRACE=0 FASTVIDEO_X=1 FV_TECH_CELLS=a FV_VAE_WORKLOADS=4k5s"
# A restored tier with its own mode's setting: drop it.
dry "restored tier with its own setting: precision-nvfp4" 1 \
  "FV_FAMILY=precision-nvfp4 is the precision family's FV_PRECISION_ARM=ltx-nvfp4 mode already: drop FV_PRECISION_ARM=ltx-nvfp4" \
  FV_FAMILY=precision-nvfp4 FV_EXTRA_ENV="FV_PRECISION_ARM=ltx-nvfp4"
dry "restored tier with its own setting: ltxvae-gen" 1 "FV_FAMILY=ltxvae-gen is the ltxvae family's FV_VAE_GEN=1 mode already" \
  FV_FAMILY=ltxvae-gen FV_EXTRA_ENV="FV_VAE_GEN=1"
# `plan` refuses the same way.
expect "plan: restored mode named" 1 "it is the fv-lab tier precision-nvfp4 now: use FV_FAMILY=precision-nvfp4" -- \
  FV_DRY_RUN=1 FV_FAMILY=precision FV_EXTRA_ENV="FV_PRECISION_ARM=ltx-nvfp4" bash "$HTTP" plan abc1234
expect "plan: restored tier is fv-lab's plan" 0 "--family precision-nvfp4 --root $ROOT --tag abc1234-1025120000-abcd" -- \
  FV_DRY_RUN=1 FV_FAMILY=precision-nvfp4 bash "$HTTP" plan abc1234 abc1234-1025120000-abcd
# runpod-matrix.sh names the restored tier: asked for as family + setting, or by name.
mx() { # mx <name> <text> <env...> <family>: the dry run's FATAL line (exit 2)
  local name="$1" text="$2" out rc=0
  shift 2
  mkdir -p "$TMP/mx/w/weights" "$TMP/mx/s"
  out="$(clean FV_DRY_RUN=1 FV_WORK="$TMP/mx/w" FV_SCRATCH="$TMP/mx/s" FV_RUN_TAG=t "${@:1:$#-1}" bash "$MATRIX" "${!#}" 2>&1)" || rc=$?
  if [[ "$rc" == 2 && "$out" == *"$text"* ]]; then ok "$name"; else bad "$name: exit $rc (want 2), output: $out"; fi
}
mx "matrix: precision + ltx-nvfp4 names precision-nvfp4" \
  "FATAL: FV_PRECISION_ARM=ltx-nvfp4 was a mode of precision; it is the fv-lab tier precision-nvfp4 now: fv-lab run precision-nvfp4" \
  FV_PRECISION_ARM=ltx-nvfp4 precision
mx "matrix: ltxvae + FV_VAE_GEN=1 names ltxvae-gen" "it is the fv-lab tier ltxvae-gen now: fv-lab run ltxvae-gen" FV_VAE_GEN=1 ltxvae
mx "matrix: precision-nvfp4 was never a script family" \
  "FATAL: precision-nvfp4 is an fv-lab tier (the precision family's FV_PRECISION_ARM=ltx-nvfp4 mode), never a runpod-matrix.sh family: fv-lab run precision-nvfp4" \
  precision-nvfp4
mx "matrix: ltxvae-gen was never a script family" "ltxvae-gen is an fv-lab tier (the ltxvae family's FV_VAE_GEN=1 mode)" ltxvae-gen
# hd's branch, which refused FV_HD_POST_* itself, is gone (§22.19): hd is
# refused before anything reads the hook.
mx "matrix: hd with FV_HD_POST_URL is refused as retired" "FATAL: hd is retired to fv-lab (fv-lab-retired.sh): fv-lab run hd " \
  FV_HD_POST_URL=file:///fvscratch/files/hd-upscaler.sh hd
mx "matrix: precision alone is the retired family" "FATAL: precision is retired to fv-lab (fv-lab-retired.sh): fv-lab run precision " \
  FV_PRECISION_ARM=all precision
# The restored tiers themselves are fv-lab only.
dry "restored tier: precision-nvfp4 through fv-lab" 0 \
  "fv-lab pod up --sha abc1234 --family precision-nvfp4 --root $ROOT --cells ltx25-4k5s-sol-bf16 ltx25-4k5s-sol-nvfp4 --lpips --yes --wait" \
  FV_FAMILY=precision-nvfp4 FV_CELLS="ltx25-4k5s-sol-bf16,ltx25-4k5s-sol-nvfp4" FV_LPIPS=1
dry "restored tier: ltxvae-gen through fv-lab" 0 "--family ltxvae-gen --root $ROOT --yes --wait" FV_FAMILY=ltxvae-gen
expect "restored tier: all refused" 1 "run \`kernels\`, then \`run\`" -- \
  RUNPOD_API_KEY=x FV_FAMILY=ltxvae-gen bash "$HTTP" all abc1234
expect "all: refused for a retired family" 1 "run \`kernels\`, then \`run\`" -- \
  RUNPOD_API_KEY=x FV_FAMILY=precision bash "$HTTP" all abc1234
expect "plan: fv-lab's plan, without --yes" 0 "--family trace --root $ROOT --tag abc1234-1025120000-abcd" -- \
  FV_DRY_RUN=1 FV_FAMILY=trace bash "$HTTP" plan abc1234 abc1234-1025120000-abcd

# 9. gold: never the script path for run / all; its plan stays the script's body.
for mode in run all; do
  expect "gold: $mode refused" 1 "fv-lab pod up --sha abc1234 --family gold" -- RUNPOD_API_KEY=x FV_FAMILY=gold bash "$HTTP" "$mode" abc1234
done
expect "gold: plan is runpod-http.sh's body" 0 '"name": "fv-lab-gold-abc1234-t"' -- RUNPOD_API_KEY=x \
  FV_PLAN_VOLUME="bz0ofkfx45 EU-RO-1" FV_POD_PREFIX=fv-lab FV_FAMILY=gold bash "$HTTP" plan abc1234 abc1234-t

# The tiers made from a port one-off (fv-lab-retired.sh FV_LAB_EXPERIMENTS,
# ADR-0019 §14): like gold, never the script path; their plan the script's body.
for mode in run all; do
  expect "h3-graphs: $mode refused" 1 "fv-lab pod up --sha abc1234 --family h3-graphs" -- RUNPOD_API_KEY=x FV_FAMILY=h3-graphs bash "$HTTP" "$mode" abc1234
done
expect "h3-graphs: plan is runpod-http.sh's body" 0 '"name": "fv-lab-h3-graphs-abc1234-t"' -- RUNPOD_API_KEY=x \
  FV_PLAN_VOLUME="bz0ofkfx45 EU-RO-1" FV_POD_PREFIX=fv-lab FV_FAMILY=h3-graphs bash "$HTTP" plan abc1234 abc1234-t

# A sibling tier (fv-lab-retired.sh FV_LAB_SIBLINGS, ADR-0019 §21: oracle-r2,
# the oracle family's sfwan13 on a volume+r2 pod; hd5 was one until hd's branch
# went, §22.19): like gold, never the script path (its targets run there as
# FV_FAMILY=oracle); its plan the script's body.
# FV_DRY_RUN=1: a refusal that broke would print the script path's dry run,
# never reach the API.
for mode in run all; do
  expect "oracle-r2: $mode refused" 1 "fv-lab oracle --family oracle-r2 --sha abc1234" -- FV_DRY_RUN=1 RUNPOD_API_KEY=x FV_FAMILY=oracle-r2 bash "$HTTP" "$mode" abc1234
  expect "oracle-r2: $mode names the script family" 1 "the script runs them as FV_FAMILY=oracle" -- FV_DRY_RUN=1 RUNPOD_API_KEY=x FV_FAMILY=oracle-r2 bash "$HTTP" "$mode" abc1234
done
expect "oracle-r2: plan is runpod-http.sh's body" 0 '"name": "fv-lab-oracle-r2-abc1234-t"' -- RUNPOD_API_KEY=x \
  FV_PLAN_VOLUME="bz0ofkfx45 EU-RO-1" FV_POD_PREFIX=fv-lab FV_FAMILY=oracle-r2 bash "$HTTP" plan abc1234 abc1234-t
# hd's post-script hook (FV_HD_POST_URL, its cap FV_HD_POST_TIMEOUT_S) was
# dropped with the family's derived step (ADR-0019 §22, S2): refused in
# FV_EXTRA_ENV on the script path (refuse_dropped: wan here, before any pod or
# key, since hd's own branch went, §22.19) and on the fv-lab route (hd's own
# included, run and plan).
for k in FV_HD_POST_URL FV_HD_POST_TIMEOUT_S; do
  for mode in run all plan; do
    expect "script path: $k refused ($mode)" 1 "$k: the hd family's post-script hook was dropped with its derived step" -- \
      FV_DRY_RUN=1 FV_FAMILY=wan FV_EXTRA_ENV="FASTVIDEO_X=1 $k=1" bash "$HTTP" "$mode" abc1234
  done
  for fam in precision hd; do
    dry "fv-lab route: $k refused ($fam)" 1 "reads no $k from FV_EXTRA_ENV ($k: the hd family's post-script hook was dropped" \
      FV_FAMILY="$fam" FV_EXTRA_ENV="$k=1"
  done
  expect "plan: $k refused (hd)" 1 "reads no $k from FV_EXTRA_ENV ($k: the hd family's post-script hook was dropped" -- \
    FV_DRY_RUN=1 FV_FAMILY=hd FV_EXTRA_ENV="$k=1" bash "$HTTP" plan abc1234
done
expect "wan: the script path without the hook is unchanged" 0 "a pod running runpod-matrix.sh wan" -- \
  FV_DRY_RUN=1 FV_FAMILY=wan FV_EXTRA_ENV="FASTVIDEO_X=1" bash "$HTTP" run abc1234

# hd (with its sibling hd5), serve-engine, sfstream and mmaudio (ADR-0019
# §22.19, S7): retired to fv-lab after their GPU checks (§22.10, §22.12,
# §22.14, §22.16). runpod-http.sh run routes each to fv-lab pod up (never the
# script path), `all` is refused, `plan` is fv-lab's plan; runpod-matrix.sh
# refuses each (exit 2) naming fv-lab's commands. The families are named here,
# not read from the list, so one dropped from FV_LAB_RETIRED fails.
# shellcheck source=scripts/gpu/fv-lab-retired.sh
. "$ROOT/scripts/gpu/fv-lab-retired.sh"
for fam in hd hd5 serve-engine sfstream mmaudio; do
  if fv_lab_retired "$fam" && ! fv_lab_sibling "$fam" >/dev/null && ! fv_script_retired "$fam"; then
    ok "$fam: in FV_LAB_RETIRED only"
  else
    bad "$fam: not retired to fv-lab (FV_LAB_RETIRED: $FV_LAB_RETIRED)"
  fi
  dry "$fam: run is fv-lab only" 0 "fv-lab pod up --sha abc1234 --family $fam --root $ROOT --yes --wait" FV_FAMILY="$fam"
  dry "$fam: never the script path" 0 "$fam is fv-lab only (fv-lab-retired.sh); nothing created" FV_FAMILY="$fam"
  expect "all: refused for $fam" 1 "run \`kernels\`, then \`run\`" -- RUNPOD_API_KEY=x FV_FAMILY="$fam" bash "$HTTP" all abc1234
  expect "plan: fv-lab's plan for $fam" 0 "--family $fam --root $ROOT --tag abc1234-1025120000-abcd" -- \
    FV_DRY_RUN=1 FV_FAMILY="$fam" bash "$HTTP" plan abc1234 abc1234-1025120000-abcd
  mx "matrix: $fam is retired" "FATAL: $fam is retired to fv-lab (fv-lab-retired.sh): fv-lab run $fam on the box, runpod-http.sh run or fv-lab pod up --family $fam from a host" "$fam"
done
# hd5 is frozen with hd (FV_LAB_RETIRED_SIBLINGS, with its mode): no longer a
# live sibling, so run reaches fv-lab rather than the sibling refusal.
hd5_mode=/opt/fastvideo-rs/scripts/gpu/prompts-hd5.json
[[ " $FV_LAB_RETIRED_SIBLINGS " == *" hd5:hd:FV_PROMPTS=$hd5_mode "* && " $FV_LAB_SIBLINGS " != *" hd5:"* ]] \
  && ok "hd5: a retired sibling of hd with its FV_PROMPTS mode, not a live one" || bad "hd5: FV_LAB_SIBLINGS=$FV_LAB_SIBLINGS FV_LAB_RETIRED_SIBLINGS=$FV_LAB_RETIRED_SIBLINGS"
dry "hd5: the 1080P gate by arm through fv-lab" 0 \
  "fv-lab pod up --sha abc1234 --family hd5 --root $ROOT --cells turbo-768p turbo-1080p max-768p max-1080p turbo-1080p-10s max-1080p-10s --yes --wait" \
  FV_FAMILY=hd5 FV_CELLS="turbo-768p,turbo-1080p,max-768p,max-1080p,turbo-1080p-10s,max-1080p-10s"
# The hd family's prompts-hd5.json mode is the hd5 tier: the script's form of
# the 1080P gate (docs/serve/h3-1080p-and-upscaler.md's run tables) is refused
# naming hd5, in every mode, from the host environment or FV_EXTRA_ENV, for any
# path to the file; refusing FV_PROMPTS alone would send it to hd's own
# prompt set under the same arm names. hd5 with its own setting: drop it.
# Another prompt set on hd: the generic refusal.
hd5_why="with FV_FAMILY=hd is the fv-lab tier hd5 now (hd's FV_PROMPTS mode, frozen with it: fv-lab-retired.sh FV_LAB_RETIRED_SIBLINGS, docs/adr/0019-fv-lab.md §22.19): use FV_FAMILY=hd5 without FV_PROMPTS (fv-lab pod up --family hd5); without FV_PROMPTS the run would be hd's own"
for v in "$hd5_mode" scripts/gpu/prompts-hd5.json; do
  for mode in run all plan; do
    expect "hd + FV_PROMPTS=$v names hd5 ($mode)" 1 "FV_PROMPTS=$v $hd5_why" -- \
      FV_DRY_RUN=1 RUNPOD_API_KEY=x FV_FAMILY=hd FV_PROMPTS="$v" FV_CELLS="turbo-768p turbo-1080p" bash "$HTTP" "$mode" abc1234
    expect "hd + FV_EXTRA_ENV FV_PROMPTS=$v names hd5 ($mode)" 1 "FV_PROMPTS=$v $hd5_why" -- \
      FV_DRY_RUN=1 RUNPOD_API_KEY=x FV_FAMILY=hd FV_EXTRA_ENV="FASTVIDEO_X=1 FV_PROMPTS=$v" FV_AVOID_MACHINES=m1 bash "$HTTP" "$mode" abc1234
  done
done
dry "hd5 with its own FV_PROMPTS: drop it" 1 "FV_FAMILY=hd5 is the hd family's FV_PROMPTS=$hd5_mode mode already: drop FV_PROMPTS" \
  FV_FAMILY=hd5 FV_PROMPTS="$hd5_mode"
dry "hd: another FV_PROMPTS refused, no hd5 in the answer" 1 "which has no FV_PROMPTS; unset it" \
  FV_FAMILY=hd FV_PROMPTS=/opt/fastvideo-rs/scripts/gpu/prompts-hd.json
dry "hd: FV_PROMPTS in FV_EXTRA_ENV refused (another prompt set)" 1 "reads no FV_PROMPTS from FV_EXTRA_ENV" \
  FV_FAMILY=hd FV_EXTRA_ENV=FV_PROMPTS=/opt/fastvideo-rs/scripts/gpu/prompts-hd.json
out="$(cd "$TMP" && clean FV_DRY_RUN=1 FV_FAMILY=hd FV_PROMPTS=/opt/fastvideo-rs/scripts/gpu/prompts-hd.json bash "$HTTP" run abc1234 2>&1 || true)"
[[ "$out" != *hd5* ]] && ok "hd: another prompt set's refusal does not name hd5" || bad "hd: $out"
dry "wan: prompts-hd5.json is not hd5's mode for another family" 0 "on the script path, nothing created" \
  FV_FAMILY=wan FV_PROMPTS="$hd5_mode"
for v in "$hd5_mode" scripts/gpu/prompts-hd5.json; do
  mx "matrix: hd + FV_PROMPTS=$v names hd5" \
    "FATAL: FV_PROMPTS=$v with hd is the fv-lab tier hd5 now (hd's FV_PROMPTS mode, frozen with it: fv-lab-retired.sh FV_LAB_RETIRED_SIBLINGS, ADR-0019 §22.19; without it the run would be hd's own): fv-lab run hd5 on the box, FV_FAMILY=hd5 runpod-http.sh run or fv-lab pod up --family hd5 from a host" \
    FV_PROMPTS="$v" hd
done
mx "matrix: hd + another FV_PROMPTS is the retired family" "FATAL: hd is retired to fv-lab (fv-lab-retired.sh): fv-lab run hd " \
  FV_PROMPTS=/opt/fastvideo-rs/scripts/gpu/prompts-hd.json hd
# The volume+r2 tiers' GPU-check form (FV_R2_TREES, §21.8) was the script
# side's; fv-lab fetches the tier's trees itself, so it is refused.
for kv in "serve-engine fastwan21-1.3b 150" "sfstream sfwan21-1.3b 150" "mmaudio fastwan21-1.3b 180"; do
  read -r fam tree disk <<<"$kv"
  dry "$fam: FV_R2_TREES refused (fv-lab fetches the tier's trees)" 1 "which has no FV_R2_TREES" \
    FV_FAMILY="$fam" FV_R2_TREES="$tree" FV_CONTAINER_DISK_GB="$disk"
  dry "$fam: the tier's disk passes" 0 "--family $fam --root $ROOT --yes --wait" FV_FAMILY="$fam" FV_CONTAINER_DISK_GB="$disk"
done
# Their knobs (§22.19, out of knobs::SCRIPT_KNOBS into MODE_KNOBS and
# FV_LAB_NOT_READ): refused in FV_EXTRA_ENV with the tier's answer, on any
# retired family, all in one message; serve-engine's group picker in the
# host environment too.
dry "serve-engine: FV_ENGINE_CELLS refused, FV_CELLS named" 1 \
  "reads no FV_ENGINE_CELLS from FV_EXTRA_ENV (FV_ENGINE_CELLS: serve-engine selects by arm, as every tier (FV_ENGINE_CELLS picked runpod-matrix.sh serve-engine's groups h3 / ltx / wan): FV_CELLS" \
  FV_FAMILY=serve-engine FV_EXTRA_ENV="FV_ENGINE_CELLS=h3"
dry "host env refused: FV_ENGINE_CELLS (serve-engine)" 1 "which has no FV_ENGINE_CELLS: it selects by arm, FV_CELLS" \
  FV_FAMILY=serve-engine FV_ENGINE_CELLS=h3
dry "sfstream: its run list and prompts refused in one message" 1 \
  "reads no FV_SFSTREAM_RUNS, FV_SF_PROMPT, FV_SF_SWITCH from FV_EXTRA_ENV (FV_SFSTREAM_RUNS: the sfstream tier's cell (lab/cells/sfstream-rollout.toml)" \
  FV_FAMILY=sfstream FV_EXTRA_ENV="FV_SFSTREAM_RUNS=--run FV_SF_PROMPT=x FASTVIDEO_X=1 FV_SF_SWITCH=y"
dry "mmaudio: FV_MMAUDIO_PROMPT refused, the cell named" 1 \
  "reads no FV_MMAUDIO_PROMPT from FV_EXTRA_ENV (FV_MMAUDIO_PROMPT: the mmaudio tier's cell (lab/cells/mmaudio-sidecar.toml)" \
  FV_FAMILY=mmaudio FV_EXTRA_ENV="FV_MMAUDIO_PROMPT=a"
dry "FV_MMAUDIO_PROMPT in another tier's FV_EXTRA_ENV: refused" 1 "reads no FV_MMAUDIO_PROMPT from FV_EXTRA_ENV" \
  FV_FAMILY=hd FV_EXTRA_ENV="FV_MMAUDIO_PROMPT=a"

# wan (ADR-0019 §22, S6): on the script path until S8, its GPU check's two trees from R2 (the
# script side stages both at once: 53.41 GB and the 120 GB); its dry run shows
# each gated arm's FASTVIDEO_TAE_DIR and fetches no TAE file; sfwan, which runs
# the branch's SF-Wan arms, inherits both.
expect "wan: run is the script path" 0 "a pod running runpod-matrix.sh wan" -- \
  FV_DRY_RUN=1 FV_FAMILY=wan bash "$HTTP" run abc1234
expect "wan: the GPU check's script side plans" 0 '"name": "fv-lab-wan-abc1234-t"' -- RUNPOD_API_KEY=x \
  FV_PLAN_VOLUME="bz0ofkfx45 EU-RO-1" FV_POD_PREFIX=fv-lab FV_FAMILY=wan "FV_R2_TREES=fastwan21-1.3b fastwan22-ti2v-5b" \
  FV_CONTAINER_DISK_GB=174 FV_MIN_RAM_GB=256 bash "$HTTP" plan abc1234 abc1234-t
expect "wan: FV_R2_TREES needs a disk for both trees" 2 "(53.41 GB) and 120 GB: FV_CONTAINER_DISK_GB=174 or more" -- \
  FV_DRY_RUN=1 FV_FAMILY=wan "FV_R2_TREES=fastwan21-1.3b fastwan22-ti2v-5b" FV_CONTAINER_DISK_GB=173 bash "$HTTP" run abc1234
for fam in wan sfwan; do
  rm -rf "$TMP/ww" "$TMP/ws"; mkdir -p "$TMP/ww/weights/auxiliary/tae" "$TMP/ws"
  : >"$TMP/ww/weights/auxiliary/tae/.complete"
  expect "$fam: each SF-Wan arm's own FASTVIDEO_TAE_DIR, no fetch in a dry run" 0 \
    "dry-run sfwan13-81f-flash: env FASTVIDEO_TAE_DIR=$TMP/ww/weights/auxiliary/tae FASTVIDEO_WAN_CAUSAL_FLASH=1 " -- \
    FV_DRY_RUN=1 FV_WORK="$TMP/ww" FV_SCRATCH="$TMP/ws" bash "$MATRIX" "$fam"
  if [[ -n "$(ls -A "$TMP/ww/weights/auxiliary/tae" | grep -v '^\.complete$')" || -e "$TMP/ws/runs/$fam/tae-fetch.log" ]]; then
    bad "$fam: a dry run fetched a TAE file"
  else
    ok "$fam: a dry run fetched nothing"
  fi
done
out="$(cd "$TMP" && clean FV_DRY_RUN=1 FV_WORK="$TMP/ww" FV_SCRATCH="$TMP/ws" bash "$MATRIX" wan 2>&1)" || true
if [[ "$out" != *"--tae-dir"* && "$out" == *"dry-run engine-fw22: env FASTVIDEO_TAE_DIR=$TMP/ww/weights/auxiliary/tae "* \
  && "$(grep -c ": env FASTVIDEO_TAE_DIR=$TMP/ww/weights/auxiliary/tae " <<<"$out")" == 45 \
  && "$out" == *"dry-run kernels-wan: "* && "$out" != *"dry-run kernels-wan: env "* ]]; then
  ok "wan: 45 gated arms with their own FASTVIDEO_TAE_DIR (kernels-wan none), engine-fw22 without --tae-dir"
else
  bad "wan: the dry run's TAE lines: $out"
fi

# b4a (ADR-0019 §11.6): fv-lab only, its GPU gate's cell selected by arm;
# the script side's FV_B4A_CELLS in FV_EXTRA_ENV is refused with that answer.
dry "b4a: fv-lab only, selected by arm" 0 \
  "fv-lab pod up --sha abc1234 --family b4a --root $ROOT --cells b4a-h3-4step-tea-base b4a-h3-4step-tea-pr --baseline-sha e93e357 --yes --wait" \
  FV_FAMILY=b4a FV_CELLS="b4a-h3-4step-tea-base b4a-h3-4step-tea-pr" FV_BASELINE_SHA=e93e357
dry "b4a: FV_BASELINE_SHA in FV_EXTRA_ENV passes to --extra-env" 0 \
  "--family b4a --root $ROOT --extra-env FV_BASELINE_SHA=e93e357 --yes --wait" \
  FV_FAMILY=b4a FV_EXTRA_ENV="FV_BASELINE_SHA=e93e357"
dry "b4a: FV_B4A_CELLS refused, FV_CELLS named" 1 \
  "reads no FV_B4A_CELLS from FV_EXTRA_ENV (FV_B4A_CELLS: b4a selects by arm, as every tier: FV_CELLS=\"<cell>-base <cell>-pr\"" \
  FV_FAMILY=b4a FV_EXTRA_ENV="FV_BASELINE_SHA=e93e357 FV_B4A_CELLS=b4a-h3-4step-tea"
dry "FV_B4A_CELLS in FV_EXTRA_ENV of another tier: no b4a in the answer" 1 \
  "(FV_B4A_CELLS: fv-lab selects by arm: FV_CELLS (fv-lab pod up --cells), not FV_B4A_CELLS" \
  FV_FAMILY=rtx6000 FV_EXTRA_ENV="FV_B4A_CELLS=b4a-h3-4step-tea"
# The pair selectors set in the host environment (not FV_EXTRA_ENV): refused,
# not ignored (fv-lab would run every cell of the tier).
for kv in "b4a FV_B4A_CELLS=b4a-h3-4step-tea" "techniques FV_TECH_CELLS=h3-480p-fullopt" "b6 FV_B6_CELLS=b6-solh3-tree"; do
  read -r fam setting <<<"$kv"
  dry "host env refused: ${setting%%=*} ($fam)" 1 "which has no ${setting%%=*}: it selects by arm, FV_CELLS" \
    FV_FAMILY="$fam" FV_BASELINE_SHA=e93e357 "$setting"
done
mx "matrix: b4a is retired" "FATAL: b4a is retired to fv-lab (fv-lab-retired.sh): fv-lab run b4a " \
  FV_BASELINE_SHA=e93e357 b4a

# det-short and determinism (ADR-0019 §15.10): fv-lab only, their GPU checks'
# groups selected by arm; the script side's FV_DET_CELLS is refused with that
# answer, in FV_EXTRA_ENV and in the host environment, and determinism's
# oracle prelude (FV_ORACLE_URL) is refused, not dropped.
dry "det-short: fv-lab only, selected by arm" 0 \
  "fv-lab pod up --sha abc1234 --family det-short --root $ROOT --cells h3-r1 h3-r2 h3-r3 h3-timed --fetch-tree --fetch-skip /frames/ --yes --wait" \
  FV_FAMILY=det-short FV_CELLS="h3-r1,h3-r2,h3-r3,h3-timed" FV_FETCH_TREE=1 FV_FETCH_SKIP=/frames/
dry "determinism: fv-lab only, selected by arm" 0 \
  "fv-lab pod up --sha abc1234 --family determinism --root $ROOT --cells h3-ab h3-c h3-timed --yes --wait" \
  FV_FAMILY=determinism FV_CELLS="h3-ab h3-c h3-timed"
dry "det-short: FV_DET_CELLS refused, FV_CELLS named" 1 \
  "reads no FV_DET_CELLS from FV_EXTRA_ENV (FV_DET_CELLS: det-short and determinism select by arm, as every tier: FV_CELLS" \
  FV_FAMILY=det-short FV_EXTRA_ENV="FV_DET_CELLS=h3"
dry "determinism: FV_DET_CELLS and FV_DET_REPEATS refused in one message" 1 \
  "reads no FV_DET_CELLS, FV_DET_REPEATS from FV_EXTRA_ENV" \
  FV_FAMILY=determinism FV_EXTRA_ENV="FV_DET_CELLS=h3-ab FV_DET_REPEATS=2"
dry "host env refused: FV_DET_CELLS (det-short)" 1 "which has no FV_DET_CELLS: it selects by arm, FV_CELLS" \
  FV_FAMILY=det-short FV_DET_CELLS=h3
dry "determinism: FV_ORACLE_URL refused" 1 \
  "(FV_ORACLE_URL: determinism no longer runs the oracle first; run the oracle on its own" \
  FV_FAMILY=determinism FV_EXTRA_ENV="FV_ORACLE_URL=https://x-8000.proxy.runpod.net/upstream/t"
dry "another tier: FV_ORACLE_URL passes to --extra-env" 0 \
  "--family det-short --root $ROOT --extra-env FV_ORACLE_URL=https://x-8000.proxy.runpod.net/upstream/t --yes --wait" \
  FV_FAMILY=det-short FV_EXTRA_ENV="FV_ORACLE_URL=https://x-8000.proxy.runpod.net/upstream/t"
dry "FV_EXTRA_ENV with a newline refused (later lines would be dropped)" 1 \
  "reads FV_EXTRA_ENV as one line of space-separated K=V entries, and it holds a newline" \
  FV_FAMILY=det-short FV_EXTRA_ENV="FASTVIDEO_X=1"$'\n'"FV_DET_CELLS=h3"
mx "matrix: det-short is retired" "FATAL: det-short is retired to fv-lab (fv-lab-retired.sh): fv-lab run det-short " det-short
mx "matrix: determinism is retired" "FATAL: determinism is retired to fv-lab (fv-lab-retired.sh): fv-lab run determinism " \
  determinism
# oracle keeps its branch (ADR-0019 §15.10): the script path, as before.
dry "oracle: still a script family" 0 "on the script path, nothing created" FV_FAMILY=oracle

# fuse, attn3 and ltx (ADR-0019 §17.6): fv-lab only, their GPU checks' arms
# selected by arm; attn3's arm pickers (FV_ATTN3_ARMS, FV_ATTN3_VSA) are
# refused with that answer, in FV_EXTRA_ENV and in the host environment.
dry "fuse: fv-lab only, selected by arm" 0 \
  "fv-lab pod up --sha abc1234 --family fuse --root $ROOT --wall-cap-s 3600 --cells kernels-fuse fasth3-4step-vsa-768p-off fasth3-4step-vsa-768p-on fasth3-8step-768p-off fasth3-8step-768p-on --yes --wait" \
  FV_FAMILY=fuse FV_POD_CAP_S=3600 \
  FV_CELLS="kernels-fuse fasth3-4step-vsa-768p-off fasth3-4step-vsa-768p-on fasth3-8step-768p-off fasth3-8step-768p-on"
dry "attn3: fv-lab only, selected by arm" 0 \
  "fv-lab pod up --sha abc1234 --family attn3 --root $ROOT --cells fasth3-4step-vsa-768p-b160 fasth3-4step-vsa-768p-b161 --yes --wait" \
  FV_FAMILY=attn3 FV_CELLS="fasth3-4step-vsa-768p-b160 fasth3-4step-vsa-768p-b161"
dry "ltx: fv-lab only, every arm" 0 "fv-lab pod up --sha abc1234 --family ltx --root $ROOT --yes --wait" FV_FAMILY=ltx
dry "attn3: FV_ATTN3_ARMS refused, FV_CELLS named" 1 \
  "reads no FV_ATTN3_ARMS from FV_EXTRA_ENV (FV_ATTN3_ARMS: attn3 runs the script's default arms (FV_ATTN3_ARMS \"v2 cudnn\", FV_ATTN3_VSA \"0 1\") and selects by arm, as every tier: FV_CELLS" \
  FV_FAMILY=attn3 FV_EXTRA_ENV="FV_ATTN3_ARMS=v2"
dry "attn3: FV_ATTN3_ARMS and FV_ATTN3_VSA refused in one message" 1 \
  "reads no FV_ATTN3_ARMS, FV_ATTN3_VSA from FV_EXTRA_ENV" \
  FV_FAMILY=attn3 FV_EXTRA_ENV="FV_ATTN3_ARMS=cudnn FASTVIDEO_X=1 FV_ATTN3_VSA=1"
dry "FV_ATTN3_VSA in another tier's FV_EXTRA_ENV: refused" 1 "reads no FV_ATTN3_VSA from FV_EXTRA_ENV" \
  FV_FAMILY=fuse FV_EXTRA_ENV="FV_ATTN3_VSA=0"
for kv in FV_ATTN3_ARMS=v2 FV_ATTN3_VSA=1; do
  dry "host env refused: ${kv%%=*} (attn3)" 1 "which has no ${kv%%=*}: it selects by arm, FV_CELLS" FV_FAMILY=attn3 "$kv"
done
for fam in fuse attn3 ltx; do
  mx "matrix: $fam is retired" "FATAL: $fam is retired to fv-lab (fv-lab-retired.sh): fv-lab run $fam " "$fam"
  expect "all: refused for $fam" 1 "run \`kernels\`, then \`run\`" -- RUNPOD_API_KEY=x FV_FAMILY="$fam" bash "$HTTP" all abc1234
  expect "plan: fv-lab's plan for $fam" 0 "--family $fam --root $ROOT --tag abc1234-1025120000-abcd" -- \
    FV_DRY_RUN=1 FV_FAMILY="$fam" bash "$HTTP" plan abc1234 abc1234-1025120000-abcd
done
# Retired with no tier (FV_SCRIPT_RETIRED): h3 (§17.4), the B4 families
# (§17.3 option (b)) and the §14.7 rows (ADR-0019 §18); then §22's S1 nine.
# No branch and no tier: runpod-http.sh run / all / plan refuse them (exit 2)
# before the key, with or without a dry run, naming what replaces each;
# runpod-matrix.sh refuses them too. Each has a pointer, none is a tier or in
# FV_LAB_RETIRED / FV_LAB_EXPERIMENTS. The loop runs over the families named
# here, not the list, so one dropped from the list fails its refusals too.
# shellcheck source=scripts/gpu/fv-lab-retired.sh
. "$ROOT/scripts/gpu/fv-lab-retired.sh"
retired_18="h3 hunyuan sana-video b200 writer precision-debug identity b4b b4c b4c2 h3attn"
retired_22="sfquality headline speechtest cold eval ltxoffload solbench sol-lingbot sol-cosmos3"
[[ "$FV_SCRIPT_RETIRED" == "$retired_18 $retired_22" ]] \
  && ok "FV_SCRIPT_RETIRED: the eleven families of ADR-0019 §18 and the nine of §22 (S1)" || bad "FV_SCRIPT_RETIRED: $FV_SCRIPT_RETIRED"
tiers="$(awk '/^\[\[tier\]\]/ { t = 1; next } t && /^name = "/ { gsub(/^name = "|"$/, ""); print; t = 0 }' "$ROOT/lab/tiers.toml")"
for fam in $retired_18 $retired_22; do
  why="$(fv_script_retired_why "$fam")"
  if [[ -n "$why" ]] && ! fv_lab_retired "$fam" && ! fv_lab_experiment "$fam" && ! grep -qx "$fam" <<<"$tiers"; then
    ok "$fam: a pointer, no tier, not routed to fv-lab"
  else
    bad "$fam: why '$why', retired to fv-lab: $(fv_lab_retired "$fam" && echo yes), a tier: $(grep -qx "$fam" <<<"$tiers" && echo yes)"
  fi
  msg="runpod-http: $fam is retired with no tier (fv-lab-retired.sh FV_SCRIPT_RETIRED, docs/adr/0019-fv-lab.md §18, §22): $why"
  dry "$fam: dry run refused, no tier" 2 "$msg" FV_FAMILY="$fam"
  expect "$fam: run refused without a key" 2 "$msg" -- FV_FAMILY="$fam" bash "$HTTP" run abc1234
  # No key: a regression past the refusal stops at the key check, never at the API.
  expect "$fam: all refused" 2 "$msg" -- FV_FAMILY="$fam" bash "$HTTP" all abc1234
  expect "$fam: plan refused" 2 "$msg" -- FV_DRY_RUN=1 FV_FAMILY="$fam" bash "$HTTP" plan abc1234 abc1234-1025120000-abcd
  mx "matrix: $fam is retired with no tier" "FATAL: $fam is retired with no tier (fv-lab-retired.sh FV_SCRIPT_RETIRED): $why" "$fam"
done

# Two pointers name a tier S7 retired from the script (ADR-0019 §22.19): the
# check they point at is the tier's, no script family's.
for kv in "sfquality sfstream" "speechtest mmaudio"; do
  read -r fam tier <<<"$kv"
  why="$(fv_script_retired_why "$fam")"
  [[ "$why" == *"he $tier tier (fv-lab run $tier) keeps"* && "$why" != *"$tier family"* ]] \
    && ok "$fam: its pointer names the $tier tier" || bad "$fam: $why"
done

# A script family's dry run creates nothing.
dry "script family: dry run" 0 "on the script path, nothing created" FV_FAMILY=wan

# A fake fv-lab: exactly the command, its exit code back.
cat >"$TMP/fake-fv-lab" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >>"$FAKE_LOG"
printf 'runs=%s cap=%s work=%s\n' "${FV_RUNS_DIR:-}" "${FV_GEN_TIMEOUT_S:-}" "${FV_WORK:-}" >>"$FAKE_LOG"
exit "${FAKE_EXIT:-0}"
EOF
chmod +x "$TMP/fake-fv-lab"
expect "a fake FV_LAB_BIN: its exit code back" 7 "fv-lab only" -- RUNPOD_API_KEY=x FV_LAB_BIN="$TMP/fake-fv-lab" \
  FAKE_LOG="$TMP/http.log" FAKE_EXIT=7 FV_FAMILY=techniques FV_CELLS="h3-480p-fullopt-base h3-480p-fullopt-env" \
  FV_BASELINE_SHA=e93e357 bash "$HTTP" run abc1234
if [[ "$(head -1 "$TMP/http.log")" == "pod up --sha abc1234 --family techniques --root $ROOT --cells h3-480p-fullopt-base h3-480p-fullopt-env --baseline-sha e93e357 --yes --wait" ]]; then
  ok "a fake FV_LAB_BIN: the exact arguments"
else
  bad "a fake FV_LAB_BIN got: $(head -1 "$TMP/http.log")"
fi
expect "no fv-lab: says how to build it" 1 "cargo build --release --locked -p fastvideo-lab --features cli" -- \
  RUNPOD_API_KEY=x FV_LAB_BIN="$TMP/none" FV_FAMILY=trace bash "$HTTP" run abc1234
# A family retired with no tier never reaches fv-lab (ADR-0019 §18).
expect "h3: refused, the fake FV_LAB_BIN not called" 2 "the rtx6000 tier runs every arm but" -- RUNPOD_API_KEY=x \
  FV_LAB_BIN="$TMP/fake-fv-lab" FAKE_LOG="$TMP/never.log" FV_FAMILY=h3 bash "$HTTP" run abc1234
[[ ! -e "$TMP/never.log" ]] && ok "h3: fv-lab not run" || bad "h3: fv-lab ran: $(cat "$TMP/never.log")"
# headline ran its cells through fv-lab (FV_LAB_BIN), even in a dry run; retired
# (ADR-0019 §22, S1), runpod-matrix.sh refuses it before fv-lab is called.
mx "matrix: headline refused, fv-lab not called" "FATAL: headline is retired with no tier (fv-lab-retired.sh FV_SCRIPT_RETIRED): its nine arms are cells of four tiers" \
  FV_LAB_BIN="$TMP/fake-fv-lab" FAKE_LOG="$TMP/never-headline.log" headline
[[ ! -e "$TMP/never-headline.log" ]] && ok "headline: fv-lab not run" || bad "headline: fv-lab ran: $(cat "$TMP/never-headline.log")"

# runpod-http.sh's private copy (TMPDIR/runpod-http.XXXXXX) is removed when the
# run ends: a refusal, a die, a dry run, the usage text, and the exec of fv-lab.
# With FV_HTTP_COPY set the script runs in place and removes nothing.
mkdir -p "$TMP/copies"
ok_copies=1
for args in "FV_FAMILY=h3 run abc1234" "FV_FAMILY=trace FV_AVOID_MACHINES=m1 FV_DRY_RUN=1 run abc1234" \
  "FV_FAMILY=trace FV_DRY_RUN=1 run abc1234" "usage" \
  "RUNPOD_API_KEY=x FV_LAB_BIN=$TMP/fake-fv-lab FAKE_LOG=$TMP/copies.log FV_FAMILY=trace run abc1234"; do
  read -r -a a <<<"$args"
  envs=() rest=()
  for w in "${a[@]}"; do if [[ "$w" == *=* ]]; then envs+=("$w"); elif [[ "$w" != usage ]]; then rest+=("$w"); fi; done
  (cd "$TMP" && clean TMPDIR="$TMP/copies" "${envs[@]}" bash "$HTTP" ${rest[@]+"${rest[@]}"} >/dev/null 2>&1) || true
  if [[ -n "$(ls -A "$TMP/copies")" ]]; then ok_copies=0; bad "private copy left after: $args ($(ls "$TMP/copies"))"; rm -f "$TMP/copies"/*; fi
done
[[ "$ok_copies" == 1 ]] && ok "runpod-http.sh: the private copy is removed after a refusal, a die, a dry run, the usage and fv-lab's exec"
cp "$HTTP" "$TMP/http-in-place.sh"
(cd "$TMP" && clean FV_HTTP_COPY=1 FV_FAMILY=h3 bash "$TMP/http-in-place.sh" run abc1234 >/dev/null 2>&1) || true
[[ -f "$TMP/http-in-place.sh" ]] && ok "runpod-http.sh with FV_HTTP_COPY set runs in place and removes nothing" \
  || bad "runpod-http.sh removed the script it ran in place"

# 6. The usage texts: the whole header, to its last entry.
expect "runpod-http.sh usage: whole header" 2 "either is refused." -- RUNPOD_API_KEY=x bash "$HTTP"
expect "runpod.sh usage: whole header" 2 "runpod.sh reap" -- bash "$ROOT/scripts/gpu/runpod.sh"

if (( fail )); then
  echo "fv-lab-route: FAILED" >&2
  exit 1
fi
echo "fv-lab-route: all ok"
