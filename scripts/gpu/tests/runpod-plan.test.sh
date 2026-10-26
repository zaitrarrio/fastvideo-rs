#!/usr/bin/env bash
# runpod.sh `gpu` and `matrix` (the SSH drivers of the first volume fill)
# after the D1 retirements (docs/adr/0019-fv-lab.md §10, §17.6): what each
# pod runs per family, without a key, a network or a pod.
#   - `runpod.sh plan gpu|matrix`: a family retired to fv-lab (fv-lab-retired.sh,
#     here ltx) runs the image's `fv-lab run <tier>`, never runpod-matrix.sh,
#     which refuses it; the others (h3, hunyuan, wan) run runpod-matrix.sh;
#   - that command, run with a stand-in fv-lab: `run ltx`, FV_WORK=/workspace
#     (so /workspace/runs/ltx, driver.log and pid as before); without fv-lab in
#     the image it stops with exit 2 and names `fv-lab pod up`;
#   - a family nothing can run (an experiment tier, gold, an unknown name)
#     stops `gpu` / `matrix` before the key is read or a pod is created;
#   - no runpod-matrix.sh call of a retired family is left in runpod.sh.
#
#   bash scripts/gpu/tests/runpod-plan.test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RP="$ROOT/scripts/gpu/runpod.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
ok() { printf 'ok   %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1" >&2; fail=1; }
# A clean environment: no key, no ~/.runpod, nothing of the caller's.
clean() { env -i PATH="$PATH" HOME="$TMP/home" LC_ALL=C "$@"; }
mkdir -p "$TMP/home"
# shellcheck source=scripts/gpu/fv-lab-retired.sh
. "$ROOT/scripts/gpu/fv-lab-retired.sh"

LAB=/opt/fastvideo-rs/target/release/fv-lab
for mode in gpu matrix; do
  out="$(clean bash "$RP" plan "$mode")" || { bad "plan $mode: exit $?"; continue; }
  want="h3 ltx hunyuan"
  [[ "$mode" == matrix ]] && want="h3 ltx hunyuan wan"
  got="$(cut -f1 <<<"$out" | tr '\n' ' ' | sed 's/ $//')"
  [[ "$got" == "$want" ]] && ok "plan $mode: the families $want" || bad "plan $mode: families '$got', want '$want'"
  while IFS=$'\t' read -r fam cmd; do
    if fv_lab_retired "$fam"; then
      if [[ "$cmd" != *runpod-matrix.sh* && "$cmd" == *"\"\$lab\" run $fam'" && "$cmd" == *"lab=$LAB;"* ]]; then
        ok "plan $mode: $fam (retired) runs fv-lab run $fam, not runpod-matrix.sh"
      else
        bad "plan $mode: $fam (retired) runs: $cmd"
      fi
    elif [[ "$cmd" == "bash /opt/fastvideo-rs/scripts/gpu/runpod-matrix.sh $fam" ]]; then
      ok "plan $mode: $fam runs runpod-matrix.sh $fam"
    else
      bad "plan $mode: $fam runs: $cmd"
    fi
  done <<<"$out"
done
ltx="$(clean bash "$RP" plan gpu | awk -F'\t' '$1 == "ltx" { print $2 }')"
fv_lab_retired ltx && ok "ltx is in FV_LAB_RETIRED" || bad "ltx is not in FV_LAB_RETIRED"

# The ltx command as the pod's shell runs it (`nohup <cmd> > driver.log &`),
# with fv-lab's path pointed at a stand-in.
cat >"$TMP/fv-lab" <<'SH'
#!/bin/sh
printf 'args=%s work=%s\n' "$*" "${FV_WORK:-}" >"$FAKE_LOG"
SH
chmod +x "$TMP/fv-lab"
mkdir -p "$TMP/runs"
run_cmd() { # run_cmd <fv-lab path>: exit status of the pod's line, driver.log in $TMP/runs
  local c="${ltx//$LAB/$1}"
  clean FAKE_LOG="$TMP/fake.log" bash -c "nohup $c > $TMP/runs/driver.log 2>&1 & echo \$! > $TMP/runs/pid; wait \$(cat $TMP/runs/pid)"
}
rc=0; run_cmd "$TMP/fv-lab" || rc=$?
if [[ "$rc" == 0 && "$(cat "$TMP/fake.log" 2>/dev/null)" == "args=run ltx work=/workspace" && -s "$TMP/runs/pid" ]]; then
  ok "ltx on the pod: fv-lab run ltx with FV_WORK=/workspace, pid written"
else
  bad "ltx on the pod: exit $rc, fv-lab saw: $(cat "$TMP/fake.log" 2>/dev/null)"
fi
rc=0; run_cmd "$TMP/none" || rc=$?
if [[ "$rc" == 2 && "$(cat "$TMP/runs/driver.log")" == *"FATAL: no fv-lab in this image ($TMP/none): ltx is fv-lab only; fv-lab pod up --family ltx"* ]]; then
  ok "ltx on the pod without fv-lab: exit 2, fv-lab pod up named"
else
  bad "ltx on the pod without fv-lab: exit $rc, $(cat "$TMP/runs/driver.log")"
fi

# A family nothing can run stops gpu / matrix first: before the key (there is
# none here; its absence would be the next error) and before any pod.
for mode in gpu matrix; do
  for fam in gold h3-graphs nope; do
    v=RP_GPU_FAMILIES
    [[ "$mode" == matrix ]] && v=RP_MATRIX_FAMILIES
    rc=0
    out="$(clean "$v=h3 $fam" bash "$RP" "$mode" 2>&1)" || rc=$?
    if [[ "$rc" == 2 && "$out" == *"FATAL: $fam is neither a runpod-matrix.sh family nor a retired one"* \
      && "$out" != *RUNPOD_API_KEY* && "$out" != *"create GPU pod"* ]]; then
      ok "$mode: $fam refused before the key or a pod"
    else
      bad "$mode with $fam: exit $rc, $out"
    fi
  done
  # The default families pass the check: the next stop is a missing tool or key.
  rc=0
  out="$(clean bash "$RP" "$mode" 2>&1)" || rc=$?
  if [[ "$rc" != 0 && "$out" != *"is neither a runpod-matrix.sh family"* && "$out" != *"create GPU pod"* ]]; then
    ok "$mode: the default families pass the check (stopped later: ${out##*FATAL: })"
  else
    bad "$mode with the default families: exit $rc, $out"
  fi
done

# No runpod-matrix.sh call of a retired family is left in runpod.sh.
for fam in $FV_LAB_RETIRED; do
  if grep -nE "runpod-matrix\.sh +$fam([^A-Za-z0-9_-]|$)" "$RP" >/dev/null; then
    bad "runpod.sh still calls runpod-matrix.sh $fam: $(grep -nE "runpod-matrix\.sh +$fam([^A-Za-z0-9_-]|$)" "$RP")"
  fi
done
# rp_family_cmd's own printf is the one place a family's name follows it.
direct="$(grep -nE 'runpod-matrix\.sh \$family|bash \$remote' "$RP" | grep -vF "printf '%s' \"bash /opt/fastvideo-rs/scripts/gpu/runpod-matrix.sh \$family\"" || true)"
[[ -z "$direct" ]] && ok "runpod.sh starts every family through rp_family_cmd" \
  || bad "runpod.sh runs runpod-matrix.sh \$family without rp_family_cmd: $direct"

if (( fail )); then echo "runpod-plan: FAILED" >&2; exit 1; fi
echo "runpod-plan: all ok"
