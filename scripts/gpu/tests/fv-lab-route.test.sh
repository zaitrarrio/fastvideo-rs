#!/usr/bin/env bash
# The fv-lab route of the retired families (docs/adr/0019-fv-lab.md §10),
# without a network, a key or a GPU:
#   - runpod-http.sh `run` / `plan` for a retired family: the settings mapped
#     across, the ones fv-lab would not honour refused (not dropped), the
#     dropped modes' knobs refused in FV_EXTRA_ENV (the two restored modes
#     naming their tier, ADR-0019 §12), lists split without globbing, and a
#     fake FV_LAB_BIN receiving exactly the command;
#   - `gold` refused for run / all (its plan still the script's body);
#   - runpod-matrix.sh `headline` on a fake box: its cells through a fake
#     fv-lab (FV_RUNS_DIR), a failing fv-lab logged, a missing one and
#     FV_PROMPTS / FV_PROMPT / FV_SEED refused (exit 2);
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
for kv in FV_POD_FILES=x FV_POD_SCRIPT=x FV_POD_FILE=x FV_PROMPTS=5 FV_PROMPT=x FV_SEED=1 FV_AVOID_MACHINES=m1 \
  FV_CREATE_WAIT_S=60 RUNPOD_ALLOWED_CUDA=13.0; do
  dry "refused: $kv" 1 "which has no ${kv%%=*}" FV_FAMILY=trace "$kv"
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
  FV_LTXFPS FV_LTXFPS_FRAMES FV_TECH_CELLS FV_B6_CELLS FV_B4A_CELLS FV_PROMPTS FV_GEN_TIMEOUT_S FV_CELLS; do
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

# A script family's dry run creates nothing.
dry "script family: dry run" 0 "on the script path, nothing created" FV_FAMILY=b4b

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

# 4. headline on a fake box (no GPU): fv-gpucheck, nvidia-smi and the CUDA
# library links faked, the cells through the fake fv-lab.
box() { # box <name>: a fresh fake box under $TMP/<name>
  local b="$TMP/$1"
  mkdir -p "$b/bin" "$b/work/weights" "$b/scratch/fv-libs"
  printf '#!/bin/sh\necho "GPU 0: fake"\n' >"$b/bin/nvidia-smi"
  printf '#!/bin/sh\nexit 0\n' >"$b/bin/fv-gpucheck"
  chmod +x "$b/bin/nvidia-smi" "$b/bin/fv-gpucheck"
  : >"$b/scratch/fv-libs/libcudnn.so"
  : >"$b/scratch/fv-libs/libcupti.so"
}
headline() { # headline <box> <env...>
  local b="$TMP/$1"
  shift
  clean PATH="$b/bin:$PATH" FV_WORK="$b/work" FV_SCRATCH="$b/scratch" FV_RUN_TAG=t \
    FV_GPUCHECK="$b/bin/fv-gpucheck" FAKE_LOG="$b/fv-lab.log" "$@" bash "$MATRIX" headline
}
box h1
rc=0
out="$(headline h1 FV_LAB_BIN="$TMP/fake-fv-lab" FAKE_EXIT=3 2>&1)" || rc=$?
runs="$TMP/h1/scratch/runs/headline/t"
if [[ "$rc" == 0 && "$(grep -c '^run ' "$TMP/h1/fv-lab.log")" == 4 ]] \
  && grep -qx "run fastvideo --cells fasth3-4step-vsa-480p fasth3-8step-480p fasth3-4step-vsa-768p fasth3-8step-768p" "$TMP/h1/fv-lab.log" \
  && grep -qx "runs=$runs cap=300 work=$TMP/h1/work" "$TMP/h1/fv-lab.log" \
  && [[ "$(grep -c 'FAIL fv-lab run [a-z0-9]* exit=3' <<<"$out")" == 4 ]] && grep -q 'matrix done' <<<"$out"; then
  ok "headline: four tiers through fv-lab into its run dir, a failing one logged"
else
  bad "headline with a failing fv-lab: exit $rc; $(cat "$TMP/h1/fv-lab.log" 2>/dev/null); $out"
fi
box h2
rc=0
out="$(headline h2 FV_LAB_BIN="$TMP/none" 2>&1)" || rc=$?
if [[ "$rc" == 2 && "$out" == *"FATAL: fv-lab missing at $TMP/none"* ]]; then ok "headline: no fv-lab is fatal (exit 2)"; else bad "headline without fv-lab: exit $rc; $out"; fi
for kv in FV_PROMPTS=5 FV_PROMPT=x FV_SEED=7; do
  box h3
  rc=0
  out="$(headline h3 FV_LAB_BIN="$TMP/fake-fv-lab" "$kv" 2>&1)" || rc=$?
  if [[ "$rc" == 2 && "$out" == *"FATAL: ${kv%%=*} is set"* && ! -s "$TMP/h3/fv-lab.log" ]]; then
    ok "headline: $kv refused (exit 2, nothing run)"
  else
    bad "headline with $kv: exit $rc; $out"
  fi
  rm -rf "$TMP/h3"
done

# 6. The usage texts: the whole header, to its last entry.
expect "runpod-http.sh usage: whole header" 2 "it skips the default weight gate)." -- RUNPOD_API_KEY=x bash "$HTTP"
expect "runpod.sh usage: whole header" 2 "runpod.sh reap" -- bash "$ROOT/scripts/gpu/runpod.sh"

if (( fail )); then
  echo "fv-lab-route: FAILED" >&2
  exit 1
fi
echo "fv-lab-route: all ok"
