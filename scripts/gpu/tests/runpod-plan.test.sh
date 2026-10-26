#!/usr/bin/env bash
# runpod.sh `gpu` and `matrix` (the SSH drivers of the first volume fill)
# after the D1 retirements (docs/adr/0019-fv-lab.md §10, §17.6, §18), without
# a key, a network or a pod. runpod.sh is sourced (FV_SOURCE_ONLY=1) and its
# Runpod, ssh and rsync calls are stubbed to record what they were given, so
# the commands checked are the ones cmd_gpu / cmd_matrix send to the pods:
#   - the lists are `gpu` (ltx) and `matrix` (ltx wan): h3 and hunyuan left
#     them when they retired with no tier (fv-lab-retired.sh FV_SCRIPT_RETIRED,
#     §18), and no family of that list (§18's and §22's) is in them;
#   - a family retired to fv-lab (fv-lab-retired.sh, here ltx) runs the
#     image's `fv-lab run <tier>` and never runpod-matrix.sh, which refuses it;
#     wan runs runpod-matrix.sh as before; `matrix` gives the retired family
#     the tier's per-arm cap (no FV_GEN_TIMEOUT_S=300);
#   - `gpu` creates one pod per family of its list and starts each;
#   - `plan gpu|matrix` prints those commands, and fails on a family nothing
#     can run;
#   - such a family (one retired with no tier, an experiment tier, gold, an
#     unknown name) stops `gpu` / `matrix` before the key is read or a pod is
#     created, and no recorded command starts runpod-matrix.sh for any retired
#     family; `runpod.sh b200` is refused (exit 2) without a key;
#   - the ltx command, as the pod's shell runs it, calls fv-lab as `run ltx`
#     with FV_WORK=/workspace; without fv-lab it exits 2 naming fv-lab pod up.
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
mkdir -p "$TMP/home"
# shellcheck source=scripts/gpu/fv-lab-retired.sh
. "$ROOT/scripts/gpu/fv-lab-retired.sh"
LAB=/opt/fastvideo-rs/target/release/fv-lab

# drive <name> <shell code>: source runpod.sh into a clean bash with every
# outside call stubbed, run the code; records under $TMP/<name>/:
#   ssh.log    "<host>\t<command>" per rp_ssh (newlines as spaces)
#   local.log  every local bash / rsync / nohup, and rp_load_key's call
#   pods.log   "create <name>" / "destroy <id>" / "backstop <id> <s>"
drive() {
  local name="$1" code="$2" d="$TMP/$1"
  mkdir -p "$d/state"
  : >"$d/ssh.log"; : >"$d/local.log"; : >"$d/pods.log"
  printf '{"id":"vol1"}\n' >"$d/state/volume.json"
  echo vol1 >"$d/state/volume.id"
  : >"$d/state/ready.json"; : >"$d/key"
  env -i PATH="$PATH" HOME="$TMP/home" LC_ALL=C D="$d" RP="$RP" CODE="$code" bash -c '
    set -euo pipefail
    FV_SOURCE_ONLY=1
    # shellcheck disable=SC1090
    . "$RP"
    RP_STATE="$D/state"; RP_LIVE="$D/live.log"; RP_SSH_KEY="$D/key"
    require_tools() { :; }
    rp_load_key() { echo "rp_load_key" >>"$D/local.log"; }
    rp_volume_dc() { echo EU-RO-1; }
    rp_volume_ready_file() { echo "$D/state/ready.json"; }
    rp_rest() { echo "[]"; }
    rp_gpu_offer() { echo "1 2.49"; }
    rp_ensure_registry_auth() { echo auth1; }
    rp_create_gpu_pod() { echo "create $1" >>"$D/pods.log"; echo "pod-$1"; }
    rp_wait_ssh() { echo "host-$1 22"; }
    rp_recreate_matrix_pod() { echo "$1 host-$1 22"; }
    rp_destroy_pod() { echo "destroy $1" >>"$D/pods.log"; }
    rp_backstop() { echo "backstop $*" >>"$D/pods.log"; }
    rp_ssh() { printf "%s\t%s\n" "$1" "${3//$'"'"'\n'"'"'/ }" >>"$D/ssh.log"; }
    rp_rsync() { :; }
    rsync() { echo "rsync $*" >>"$D/local.log"; }
    bash() { echo "bash $*" >>"$D/local.log"; }
    nohup() { echo "nohup $*" >>"$D/local.log"; }
    eval "$CODE"
  ' >"$d/out" 2>&1
}

# A recorded line that starts runpod-matrix.sh for a retired family (any
# quoting, braces or local / remote): none may exist.
matrix_call() { # matrix_call <file> <family>
  grep -E "runpod-matrix\.sh[\"'}]*[[:space:]]+[\"'{]*$2([\"'}[:space:]]|$)" "$1" || true
}

# 1. gpu: the real cmd_gpu.
rc=0; drive gpu cmd_gpu || rc=$?
d="$TMP/gpu"
if [[ "$rc" == 0 ]]; then ok "gpu: cmd_gpu runs to the end on the stubs"; else bad "gpu: exit $rc: $(cat "$d/out")"; fi
pods="$(grep -c '^create ' "$d/pods.log" || true)"
[[ "$pods" == 1 && "$(grep -c '^create fv-gpu-ltx-' "$d/pods.log")" == 1 ]] \
  && ok "gpu: one pod (ltx)" || bad "gpu: pods: $(cat "$d/pods.log")"
grep -q '^backstop pod-fv-gpu-ltx-[0-9]* 7200$' "$d/pods.log" && [[ "$(grep -c '^backstop ' "$d/pods.log")" == 1 ]] \
  && ok "gpu: ltx's backstop 7200 s as before" || bad "gpu: backstops: $(grep backstop "$d/pods.log")"
for fam in ltx; do
  line="$(grep -F "/workspace/runs/$fam/driver.log" "$d/ssh.log" | cut -f2- || true)"
  if fv_lab_retired "$fam"; then
    if [[ "$line" == "nohup bash -c 'lab=$LAB;"*"\"\$lab\" run $fam' > /workspace/runs/$fam/driver.log 2>&1 & echo \$! > /workspace/runs/$fam/pid" ]]; then
      ok "gpu: $fam (retired) starts fv-lab run $fam into runs/$fam/driver.log and pid"
    else
      bad "gpu: $fam starts: $line"
    fi
  elif [[ "$line" == "nohup bash /opt/fastvideo-rs/scripts/gpu/runpod-matrix.sh $fam > /workspace/runs/$fam/driver.log 2>&1 & echo \$! > /workspace/runs/$fam/pid" ]]; then
    ok "gpu: $fam starts runpod-matrix.sh $fam as before"
  else
    bad "gpu: $fam starts: $line"
  fi
done

# 2. matrix: the real cmd_matrix, its two families in turn.
rc=0; drive matrix cmd_matrix || rc=$?
d="$TMP/matrix"
if [[ "$rc" == 0 ]]; then ok "matrix: cmd_matrix runs to the end on the stubs"; else bad "matrix: exit $rc: $(cat "$d/out")"; fi
for fam in ltx wan; do
  line="$(grep -F 'export FV_WORK=/workspace' "$d/ssh.log" | cut -f2- \
    | grep -E "(runpod-matrix\.sh $fam|run $fam')\$" || true)"
  if fv_lab_retired "$fam"; then
    if [[ "$line" == *"\"\$lab\" run $fam'" && "$line" != *runpod-matrix.sh* && "$line" != *FV_GEN_TIMEOUT_S* ]]; then
      ok "matrix: $fam (retired) runs fv-lab run $fam, the tier's per-arm cap"
    else
      bad "matrix: $fam runs: $line"
    fi
  elif [[ "$line" == *"FV_GEN_TIMEOUT_S=300 "*"bash /opt/fastvideo-rs/scripts/gpu/runpod-matrix.sh $fam" ]]; then
    ok "matrix: $fam runs runpod-matrix.sh $fam with FV_GEN_TIMEOUT_S=300 as before"
  else
    bad "matrix: $fam runs: $line"
  fi
done

# 3. No recorded command, remote or local, starts runpod-matrix.sh for a
#    retired family (to fv-lab or with no tier), and neither list holds a
#    family retired with no tier.
for mode in gpu matrix; do
  hits=""
  for fam in $FV_LAB_RETIRED $FV_SCRIPT_RETIRED; do
    hits+="$(matrix_call "$TMP/$mode/ssh.log" "$fam")$(matrix_call "$TMP/$mode/local.log" "$fam")"
  done
  [[ -z "$hits" ]] && ok "$mode: no runpod-matrix.sh start of a retired family, over ssh or locally" \
    || bad "$mode: runpod-matrix.sh started for a retired family: $hits"
done

# 4. plan prints the same commands; a family nothing can run fails it.
for mode in gpu matrix; do
  rc=0
  out="$(env -i PATH="$PATH" HOME="$TMP/home" LC_ALL=C bash "$RP" plan "$mode")" || rc=$?
  want="ltx"; [[ "$mode" == matrix ]] && want="ltx wan"
  got="$(cut -f1 <<<"$out" | tr '\n' ' ' | sed 's/ $//')"
  ltxcmd="$(awk -F'\t' '$1 == "ltx" { print $2 }' <<<"$out")"
  if [[ "$rc" == 0 && "$got" == "$want" && "$ltxcmd" == *"\"\$lab\" run ltx'" && "$ltxcmd" != *runpod-matrix.sh* ]] \
    && { [[ "$mode" == gpu ]] || grep -qxF "wan	bash /opt/fastvideo-rs/scripts/gpu/runpod-matrix.sh wan" <<<"$out"; }; then
    ok "plan $mode: $want, ltx through fv-lab"
  else
    bad "plan $mode: exit $rc: $out"
  fi
done
rc=0; drive plan-bad 'RP_GPU_FAMILIES=(ltx nope); cmd_plan gpu' || rc=$?
if [[ "$rc" != 0 && "$(cat "$TMP/plan-bad/out")" == *"FATAL: nope is neither"* && "$(cat "$TMP/plan-bad/out")" != *$'nope\t'* ]]; then
  ok "plan: an unknown family fails it (exit $rc), nothing printed for it"
else
  bad "plan with an unknown family: exit $rc: $(cat "$TMP/plan-bad/out")"
fi

# 5. A family nothing can run stops gpu / matrix before the key and any pod,
#    and fails `plan gpu|matrix` with nothing printed for it.
#    A family retired with no tier (§18, and §22's S1 nine) says so and names
#    what replaces it. The retired families are named here, not read from the
#    list, so one dropped from FV_SCRIPT_RETIRED fails (it would be refused as
#    unknown, without its pointer).
no_tier="h3 hunyuan sana-video b200 writer precision-debug identity b4b b4c b4c2 h3attn"
no_tier+=" sfquality headline speechtest cold eval ltxoffload solbench sol-lingbot sol-cosmos3"
[[ "$FV_SCRIPT_RETIRED" == "$no_tier" ]] && ok "FV_SCRIPT_RETIRED: the families named here" \
  || bad "FV_SCRIPT_RETIRED: $FV_SCRIPT_RETIRED"
for mode in gpu matrix plan-gpu plan-matrix; do
  v=RP_GPU_FAMILIES; [[ "$mode" == *matrix ]] && v=RP_MATRIX_FAMILIES
  run="cmd_$mode"; [[ "$mode" == plan-* ]] && run="cmd_plan ${mode#plan-}"
  for fam in gold h3-graphs nope $no_tier; do
    rc=0; drive "$mode-$fam" "$v=(ltx $fam); $run" || rc=$?
    d="$TMP/$mode-$fam"
    want="FATAL: $fam is neither a runpod-matrix.sh family nor a retired one"
    [[ " $no_tier " == *" $fam "* ]] && want="FATAL: $fam is retired with no tier (fv-lab-retired.sh FV_SCRIPT_RETIRED, docs/adr/0019-fv-lab.md §18, §22): nothing runs it on the pod; $(fv_script_retired_why "$fam")"
    if [[ "$rc" == 2 && "$(cat "$d/out")" == *"$want"* && "$(cat "$d/out")" != *$'\t'* \
      && ! -s "$d/pods.log" && ! -s "$d/ssh.log" && ! -s "$d/local.log" ]]; then
      ok "$mode: $fam refused before the key or a pod"
    else
      bad "$mode with $fam: exit $rc, $(cat "$d/out"), pods: $(cat "$d/pods.log"), local: $(cat "$d/local.log")"
    fi
  done
done
rc=0
out="$(env -i PATH="$PATH" HOME="$TMP/home" LC_ALL=C bash "$RP" b200 2>&1)" || rc=$?
if [[ "$rc" == 2 && "$out" == *"FATAL: b200 is retired with no tier"*"the US volume it ran on is gone"* ]]; then
  ok "runpod.sh b200: refused (exit 2), no key read"
else
  bad "runpod.sh b200: exit $rc, $out"
fi

# 6. The ltx command as the pod's shell runs it (`nohup <cmd> > driver.log &`),
#    fv-lab's path pointed at a stand-in.
ltx="$(env -i PATH="$PATH" HOME="$TMP/home" bash "$RP" plan gpu | awk -F'\t' '$1 == "ltx" { print $2 }')"
cat >"$TMP/fv-lab" <<'SH'
#!/bin/sh
printf 'args=%s work=%s\n' "$*" "${FV_WORK:-}" >"$FAKE_LOG"
SH
chmod +x "$TMP/fv-lab"
mkdir -p "$TMP/runs"
run_cmd() { # run_cmd <fv-lab path>
  local c="${ltx//$LAB/$1}"
  env -i PATH="$PATH" FAKE_LOG="$TMP/fake.log" \
    bash -c "nohup $c > $TMP/runs/driver.log 2>&1 & echo \$! > $TMP/runs/pid; wait \$(cat $TMP/runs/pid)"
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

if (( fail )); then echo "runpod-plan: FAILED" >&2; exit 1; fi
echo "runpod-plan: all ok"
