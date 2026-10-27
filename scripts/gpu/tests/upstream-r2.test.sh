#!/usr/bin/env bash
# The upstream pod's R2 trees (docs/adr/0019-fv-lab.md §21: the `r2:<tree>`
# step of scripts/gpu/upstream/pod.sh, lab/upstream.toml's `r2`), without a
# pod, a key or a network:
#   - pod.sh's r2_key_init and weights_r2 (cut from the script, run in a clean
#     bash with a fake python standing for r2-weights.py / r2-fetch.py): the
#     key leaves the environment at once (no child sees it), reaches the plan
#     step alone and in its environment only (never an argv, an output or a
#     file), the plan pins weights-r2.tsv's revision and is removed whatever
#     the fetch did, the fetch writes into $UW (never /workspace) with
#     --skip-trees '' and a status file; no key, an unfilled reference, a
#     tree not in the table, a bad name, a tree already there;
#   - pod.sh dispatches r2:<tree>, writes r2_key=<source> to box.txt and reads
#     the key nowhere else;
#   - runpod-http.sh plan-upstream: the two Runpod secret references (never
#     values) for an r2:<tree> step whatever whitespace separates the steps,
#     none without one, beside the Hub token's;
#   - scripts/gpu/oracle.sh gives sfwan13 its r2 step and 180 GB, as
#     lab/upstream.toml's target;
#   - upstream/oracle.sh's oracle_wan22 / oracle_fastwan22 read their tree
#     from $UW when the r2 step completed it, else the volume's (ADR-0019 §22,
#     S6, as oracle_sfwan's §21.2), and lab/upstream.toml gives their targets
#     the r2 trees and the 210 GB disk.
#
#   bash scripts/gpu/tests/upstream-r2.test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
POD="$ROOT/scripts/gpu/upstream/pod.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
ok() { printf 'ok   %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1" >&2; fail=1; }

for fn in r2_key_init weights_r2; do
  sed -n "/^$fn() {\$/,/^}\$/p" "$POD" >>"$TMP/fn.sh"
  grep -q "^$fn() {\$" "$TMP/fn.sh" || { echo "upstream-r2: no $fn in pod.sh" >&2; exit 2; }
done
# r2_key_init is the script's first command: after `set -uo pipefail` only
# comments, its definition, then its call (a command substitution or a child
# before it would inherit the key).
first="$(awk '/^#/ || /^$/ { next } { print NR": "$0; if (++n == 2) exit }' "$POD")"
def_end="$(awk '/^r2_key_init\(\) \{$/ { f = 1 } f && /^}$/ { print NR; exit }' "$POD")"
call="$(grep -nx 'r2_key_init' "$POD" | cut -d: -f1)"
[[ "$(sed -n 1p <<<"$first")" == *": set -uo pipefail" && "$(sed -n 2p <<<"$first")" == *": r2_key_init() {" \
  && "$call" == "$((def_end + 1))" ]] && ok "pod.sh runs r2_key_init first, before any other command" \
  || bad "pod.sh: r2_key_init is not its first command ($first; def ends $def_end, call $call)"
# Cleared once the steps are done (the background steps awaited first).
grep -A1 -x '\[\[ -n "\$bg" \]\] && wait "\$bg"' "$POD" | grep -qx 'unset R2_KEY_ID R2_KEY_SECRET' \
  && ok "pod.sh clears the key after the steps" || bad "pod.sh does not clear the key after the steps"
grep -qE '^ +r2:\*\) weights_r2 "\$\{s#r2:\}" ;;$' "$POD" && ok "pod.sh dispatches r2:<tree> to weights_r2" \
  || bad "pod.sh dispatches r2:<tree>"
grep -qE '^ +echo "r2_key=\$R2_KEY_SRC"$' "$POD" && ok "info_box writes r2_key=\$R2_KEY_SRC" || bad "info_box writes r2_key="
# The key's values are read in r2_key_init and handed over in weights_r2 only.
uses="$(awk '/^r2_key_init\(\) \{$/,/^}$/ { next } /^weights_r2\(\) \{$/,/^}$/ { next } /^unset R2_KEY_ID R2_KEY_SECRET$/ { next } /FV_WEIGHTS_R2_|R2_KEY_ID|R2_KEY_SECRET/ && !/^ *#/' "$POD")"
[[ -z "$uses" ]] && ok "pod.sh reads the R2 key only in r2_key_init / weights_r2" || bad "pod.sh uses the key elsewhere: $uses"

# The plan's default directory (UP_R2_STATE unset): absolute, off the volume,
# outside both runs directories runpod-http.sh's upstream start command serves
# (/root/runs with UP_LOCAL=1, the default; /workspace/runs otherwise) and
# fv-lab's (src/upstream-start.sh.in).
def_state="$(sed -n 's/^  state="\${UP_R2_STATE:-\(.*\)}"$/\1/p' "$POD")"
served_runs="$(sed -n 's/^OUT=\(.*\)\/upstream\/@TAG@$/\1/p' "$ROOT/crates/fastvideo-lab/src/upstream-start.sh.in")"
[[ "$def_state" == /* && "$def_state" != *'$'* && "$served_runs" == /root/runs \
  && "$def_state/" != /root/runs/* && "$def_state/" != /workspace/* ]] \
  && grep -qF 'local tag="$1" runs=/workspace/runs' "$ROOT/scripts/gpu/runpod-http.sh" \
  && grep -qF '[[ "${UP_LOCAL:-1}" == 1 ]] && runs=/root/runs' "$ROOT/scripts/gpu/runpod-http.sh" \
  && ok "the plan's default directory $def_state: off the volume, outside the served runs" \
  || bad "the plan's default directory '$def_state' (served: '$served_runs')"

# A fake repository root for $HERE/../../..: the real weights-r2.tsv.
REPO="$TMP/repo"
mkdir -p "$REPO/scripts/gpu/upstream"
cp "$ROOT/scripts/gpu/weights-r2.tsv" "$REPO/scripts/gpu/"
REV="$(awk -F'\t' '$1 == "sfwan21-1.3b" { print $2; exit }' "$ROOT/scripts/gpu/weights-r2.tsv")"
[[ -n "$REV" ]] || { echo "upstream-r2: sfwan21-1.3b is not in weights-r2.tsv" >&2; exit 2; }
# The fake python ($UP/tools/bin/python): logs each call's argv and whether
# the key is in its env (never the values), writes the plan, and the fetch a
# complete tree (unless $TMP/fail-fetch exists).
UP="$TMP/up"
mkdir -p "$UP/tools/bin"
# The pod's layout: the start command serves $RUNS, pod.sh's $OUT is
# $RUNS/upstream/<tag>; the plan's marker stands for its presigned URLs.
RUNS="$TMP/runs" O="$TMP/runs/upstream/tag" MARK="X-Amz-Signature=FAKEsig0123456789"
cat >"$UP/tools/bin/python" <<EOF
#!/usr/bin/env bash
log="$TMP/calls.log"
printf 'argv:' >>"\$log"; printf ' %s' "\$@" >>"\$log"; echo >>"\$log"
echo "env: id=\${FV_WEIGHTS_R2_ACCESS_KEY_ID:+set} secret=\${FV_WEIGHTS_R2_SECRET_ACCESS_KEY:+set} skip=\${FV_R2_SKIP_TREES-unset}" >>"\$log"
env >"$TMP/env.\$(basename "\$1")"
case "\$(basename "\$1")" in
  r2-weights.py)
    out=""; prev=""; for a in "\$@"; do [[ "\$prev" == --out ]] && out="\$a"; prev="\$a"; done
    echo '{"url":"https://r2.example/x?$MARK"}' >"\$out"
    if [[ -e "$TMP/slow-plan" ]]; then sleep 30; fi ;;
  r2-fetch.py)
    [[ -e "$TMP/fail-fetch" ]] && { echo "FAILED sfwan21-1.3b: r2 503"; exit 1; }
    root=""; status=""; plan=""; prev=""
    for a in "\$@"; do [[ "\$prev" == --root ]] && root="\$a"; [[ "\$prev" == --status ]] && status="\$a"; [[ "\$prev" == --plan ]] && plan="\$a"; prev="\$a"; done
    cp "\$plan" "$TMP/plan-seen.json" 2>/dev/null
    mkdir -p "\$root/sfwan21-1.3b"; : >"\$root/sfwan21-1.3b/.complete"; echo "$REV" >"\$root/sfwan21-1.3b/.fv-revision"
    echo '{"state":"done"}' >"\$status" ;;
esac
EOF
chmod +x "$UP/tools/bin/python"

ID=AKIDtest0123 SECRET=secretvalue4567
# run <name> <uw> <tree> [K=V …]: r2_key_init then weights_r2 <tree> with the
# key env given; the output, the exit status, and what a child sees of the key.
run() {
  local name="$1" uw="$2" tree="$3"
  shift 3
  rm -rf "$RUNS" "$TMP/state" "$TMP/calls.log" "$TMP"/env.* "$TMP/plan-seen.json"
  mkdir -p "$O"
  set +e
  OUTPUT="$(env -i PATH="$PATH" HOME="$TMP" "$@" bash -c '
    . "$1"
    HERE="$2/scripts/gpu/upstream" UP="$3" UW="$4" OUT="$5" UP_R2_STATE="${UP_R2_STATE:-$6}" UP_CELL_TIMEOUT_S=60
    r2_key_init
    echo "src=$R2_KEY_SRC"
    bash -c "echo child: id=\${FV_WEIGHTS_R2_ACCESS_KEY_ID-unset} secret=\${FV_WEIGHTS_R2_SECRET_ACCESS_KEY-unset}"
    weights_r2 "$7"
  ' _ "$TMP/fn.sh" "$REPO" "$UP" "$uw" "$O" "$TMP/state" "$tree" 2>&1)"
  RC=$?
  set -e
  # The key's values and the plan's URLs: in no output, no argv, no file the
  # run left (the served runs directory, the calls, $UW, the state directory).
  local leak
  leak="$(grep -rlF -e "$ID" -e "$SECRET" -e "$MARK" "$RUNS" "$TMP/calls.log" "$uw" "$TMP/state" 2>/dev/null || true)"
  [[ "$OUTPUT" != *"$ID"* && "$OUTPUT" != *"$SECRET"* && "$OUTPUT" != *"$MARK"* && -z "$leak" ]] \
    && ok "$name: the key's values and the plan's URLs in no output, argv or file" || bad "$name: the key or the plan leaked ($leak)"
}
KEY=(FV_WEIGHTS_R2_ACCESS_KEY_ID="$ID" FV_WEIGHTS_R2_SECRET_ACCESS_KEY="$SECRET")

# 1. The key, a fetch.
run "fetch" "$TMP/uw" sfwan21-1.3b "${KEY[@]}"
[[ "$RC" == 0 ]] && ok "fetch: exit 0" || bad "fetch: exit $RC: $OUTPUT"
grep -qx 'src=env' <<<"$OUTPUT" && ok "fetch: r2_key=env" || bad "fetch: $OUTPUT"
grep -qx 'child: id=unset secret=unset' <<<"$OUTPUT" && ok "fetch: no child sees the key" || bad "fetch: a child sees the key: $OUTPUT"
grep -qx "argv: $REPO/scripts/gpu/upstream/../../../scripts/gpu/r2-weights.py plan sfwan21-1.3b@$REV --pin $REPO/scripts/gpu/upstream/../../../scripts/gpu/weights-r2.tsv --ttl 660 --out $TMP/state/plan-sfwan21-1.3b.json" "$TMP/calls.log" \
  && ok "fetch: the plan pins weights-r2.tsv's revision" || bad "fetch: plan argv: $(cat "$TMP/calls.log")"
[[ "$(sed -n 2p "$TMP/calls.log")" == "env: id=set secret=set skip=unset" ]] && ok "fetch: the plan step has the key" \
  || bad "fetch: plan env: $(sed -n 2p "$TMP/calls.log")"
grep -qx "argv: $REPO/scripts/gpu/upstream/../../../scripts/gpu/r2-fetch.py --root $TMP/uw --plan $TMP/state/plan-sfwan21-1.3b.json --skip-trees  --status $O/r2-sfwan21-1.3b.json" "$TMP/calls.log" \
  && ok "fetch: r2-fetch.py into \$UW, every tree, a status file" || bad "fetch: fetch argv: $(cat "$TMP/calls.log")"
[[ "$(sed -n 4p "$TMP/calls.log")" == "env: id= secret= skip=unset" ]] && ok "fetch: the fetch step has no key" \
  || bad "fetch: fetch env: $(sed -n 4p "$TMP/calls.log")"
! grep -q FV_WEIGHTS_R2_ "$TMP/env.r2-fetch.py" && ok "fetch: no key name in the fetch's environment" || bad "fetch: key in the fetch's env"
[[ -f "$TMP/plan-seen.json" && ! -e "$TMP/state/plan-sfwan21-1.3b.json" ]] && ok "fetch: the plan existed for the fetch, removed after" \
  || bad "fetch: plan lifecycle"
[[ "$(stat -c %a "$TMP/state")" == 700 ]] && ok "fetch: the plan's directory is private (700)" || bad "fetch: state mode $(stat -c %a "$TMP/state")"
[[ -f "$TMP/uw/sfwan21-1.3b/.complete" && -f "$O/r2-sfwan21-1.3b.json" ]] && ok "fetch: the tree in \$UW, the status in \$OUT" \
  || bad "fetch: no tree or status"
# 2. Already there at its revision: nothing called, no key needed.
run "present" "$TMP/uw" sfwan21-1.3b
[[ "$RC" == 0 && ! -s "$TMP/calls.log" ]] && grep -q "already at $TMP/uw/sfwan21-1.3b" <<<"$OUTPUT" \
  && ok "present: kept, nothing fetched" || bad "present: rc $RC, $OUTPUT"
# 3. A failed fetch: the step fails, the plan is still removed.
rm -rf "$TMP/uw"; : >"$TMP/fail-fetch"
run "failed fetch" "$TMP/uw" sfwan21-1.3b "${KEY[@]}"
[[ "$RC" != 0 && ! -e "$TMP/state/plan-sfwan21-1.3b.json" ]] && grep -q 'r2-fetch.py exit 1' <<<"$OUTPUT" \
  && ok "failed fetch: fails, the plan removed" || bad "failed fetch: rc $RC, $OUTPUT"
rm -f "$TMP/fail-fetch"
# 4. No usable key: nothing called, the step fails naming the secrets.
for c in "no env::none" \
  "unfilled references:FV_WEIGHTS_R2_ACCESS_KEY_ID={{ RUNPOD_SECRET_fv_r2_access_key_id }} FV_WEIGHTS_R2_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_fv_r2_secret_access_key }}:none (env: unfilled secret reference)" \
  "the id only:FV_WEIGHTS_R2_ACCESS_KEY_ID=$ID:none (env: incomplete)" \
  "blank:FV_WEIGHTS_R2_ACCESS_KEY_ID= FV_WEIGHTS_R2_SECRET_ACCESS_KEY= :none (env: incomplete)"; do
  IFS=: read -r name kv want <<<"$c"
  args=()
  if [[ "$kv" == *'{{'* ]]; then
    args=(FV_WEIGHTS_R2_ACCESS_KEY_ID='{{ RUNPOD_SECRET_fv_r2_access_key_id }}' FV_WEIGHTS_R2_SECRET_ACCESS_KEY='{{ RUNPOD_SECRET_fv_r2_secret_access_key }}')
  elif [[ -n "$kv" ]]; then
    read -r -a args <<<"$kv"
  fi
  run "$name" "$TMP/uw" sfwan21-1.3b ${args[@]+"${args[@]}"}
  [[ "$RC" != 0 && ! -s "$TMP/calls.log" ]] && grep -qx "src=$want" <<<"$OUTPUT" && grep -q 'no R2 key in the pod.s env' <<<"$OUTPUT" \
    && ok "$name: r2_key=$want, nothing fetched" || bad "$name: rc $RC, $OUTPUT"
done
# 5. Refused before anything: a tree the table lacks, a bad name, $UW on the volume.
run "unpublished" "$TMP/uw" no-such-tree "${KEY[@]}"
[[ "$RC" != 0 && ! -s "$TMP/calls.log" ]] && grep -q 'not in scripts/gpu/weights-r2.tsv' <<<"$OUTPUT" \
  && ok "unpublished: refused" || bad "unpublished: rc $RC, $OUTPUT"
for t in ../x /abs 'a b' a/../b; do
  run "name $t" "$TMP/uw" "$t" "${KEY[@]}"
  [[ "$RC" != 0 && ! -s "$TMP/calls.log" ]] && grep -q 'not a tree name' <<<"$OUTPUT" \
    && ok "name '$t': refused" || bad "name '$t': rc $RC, $OUTPUT"
done
run "on the volume" /workspace/upstream/weights sfwan21-1.3b "${KEY[@]}"
[[ "$RC" != 0 && ! -s "$TMP/calls.log" ]] && grep -q 'is on the volume' <<<"$OUTPUT" \
  && ok "on the volume: refused" || bad "on the volume: rc $RC, $OUTPUT"
# 6. The tree's parent resolves through a link inside $UW: onto the volume, or
#    outside $UW. Refused before anything.
rm -rf "$TMP/uw"; mkdir -p "$TMP/uw" "$TMP/elsewhere"
ln -s /workspace/weights/auxiliary "$TMP/uw/auxiliary"
run "parent on the volume" "$TMP/uw" auxiliary/ncii-light-guard "${KEY[@]}"
[[ "$RC" != 0 && ! -s "$TMP/calls.log" ]] && grep -q 'its parent resolves to /workspace/weights/auxiliary/' <<<"$OUTPUT" \
  && ok "parent through a link onto the volume: refused" || bad "parent on the volume: rc $RC, $OUTPUT"
rm "$TMP/uw/auxiliary"; ln -s "$TMP/elsewhere" "$TMP/uw/auxiliary"
run "parent outside \$UW" "$TMP/uw" auxiliary/ncii-light-guard "${KEY[@]}"
[[ "$RC" != 0 && ! -s "$TMP/calls.log" && -z "$(ls -A "$TMP/elsewhere")" ]] && grep -q 'outside' <<<"$OUTPUT" \
  && ok "parent through a link outside \$UW: refused, nothing written there" || bad "parent outside: rc $RC, $OUTPUT"
rm -rf "$TMP/uw"
# 7. The plan step under a time limit: a hung plan is killed, its file removed.
: >"$TMP/slow-plan"
t0=$SECONDS
run "hung plan" "$TMP/uw" sfwan21-1.3b "${KEY[@]}" UP_R2_PLAN_TIMEOUT_S=1
rm -f "$TMP/slow-plan"
[[ "$RC" != 0 && $((SECONDS - t0)) -lt 20 && -z "$(ls -A "$TMP/state" 2>/dev/null)" ]] && ! grep -q r2-fetch.py "$TMP/calls.log" \
  && grep -q 'r2-weights.py plan failed (exit 124)' <<<"$OUTPUT" \
  && ok "hung plan: killed by the timeout, the plan removed, nothing fetched" || bad "hung plan: rc $RC after $((SECONDS - t0)) s, $OUTPUT"
# 8. The state directory in the served runs directory or on the volume: refused.
for st in "$RUNS/r2" "$O/.r2" /workspace/r2-state; do
  run "state $st" "$TMP/uw" sfwan21-1.3b "${KEY[@]}" UP_R2_STATE="$st"
  [[ "$RC" != 0 && ! -s "$TMP/calls.log" && ! -e "$st" ]] && grep -q 'is in the served' <<<"$OUTPUT" \
    && ok "state $st: refused, nothing made" || bad "state $st: rc $RC, $OUTPUT"
done

# runpod-http.sh plan-upstream: the body's env.
plan() { # <K=V …>: the body's env as compact JSON
  env -i PATH="$PATH" HOME="$TMP" LC_ALL=C RUNPOD_API_KEY=test FV_PLAN_VOLUME="vol0 EU-RO-1" \
    UP_SHA=1234567890abcdef1234567890abcdef12345678 "$@" \
    bash "$ROOT/scripts/gpu/runpod-http.sh" plan-upstream 1234567 1234567-10261200 | jq -c '.env // {}'
}
R2='"FV_WEIGHTS_R2_ACCESS_KEY_ID":"{{ RUNPOD_SECRET_fv_r2_access_key_id }}","FV_WEIGHTS_R2_SECRET_ACCESS_KEY":"{{ RUNPOD_SECRET_fv_r2_secret_access_key }}"'
check_plan() { # <name> <want> <K=V …>
  local name="$1" want="$2" got
  shift 2
  got="$(plan "$@")"
  [[ "$got" == "$want" ]] && ok "plan-upstream $name: $got" || bad "plan-upstream $name: got $got, want $want"
}
check_plan "r2 step" "{$R2}" "UP_STEPS=info:box r2:sfwan21-1.3b oracle:sfwan13"
check_plan "r2 step after a tab" "{$R2}" "UP_STEPS=info:box"$'\t'"r2:sfwan21-1.3b oracle:sfwan13"
check_plan "r2 step in UP_STEPS_BG" "{$R2}" "UP_STEPS=info:box" "UP_STEPS_BG=r2:sfwan21-1.3b"
check_plan "no r2 step" "{}" "UP_STEPS=info:box weights:h3-diffusers oracle:fasth3-4step-vsa"
check_plan "a word holding r2:" "{}" "UP_STEPS=info:box xr2:y oracle:sfwan13"
check_plan "with the Hub token" "{\"HF_TOKEN\":\"{{ RUNPOD_SECRET_fv_hf_token }}\",$R2}" "UP_STEPS=info:box weights:ltx25 r2:x oracle:t"
# The start command: with an r2 step the key leaves its environment on its
# first line (before the watchdogs, the file server, the clone) and goes back
# to pod.sh alone; without one, no R2 line at all.
cmd() { # <K=V …>: the body's start command
  env -i PATH="$PATH" HOME="$TMP" LC_ALL=C RUNPOD_API_KEY=test FV_PLAN_VOLUME="vol0 EU-RO-1" \
    UP_SHA=1234567890abcdef1234567890abcdef12345678 "$@" \
    bash "$ROOT/scripts/gpu/runpod-http.sh" plan-upstream 1234567 1234567-10261200 | jq -r '.dockerStartCmd[2]'
}
CAPTURE='_fvr2_id="${FV_WEIGHTS_R2_ACCESS_KEY_ID:-}" _fvr2_secret="${FV_WEIGHTS_R2_SECRET_ACCESS_KEY:-}"; unset FV_WEIGHTS_R2_ACCESS_KEY_ID FV_WEIGHTS_R2_SECRET_ACCESS_KEY'
HANDOFF='FV_WEIGHTS_R2_ACCESS_KEY_ID="$_fvr2_id" FV_WEIGHTS_R2_SECRET_ACCESS_KEY="$_fvr2_secret" env '
C="$(cmd "UP_STEPS=info:box r2:sfwan21-1.3b oracle:sfwan13")"
[[ "$(head -1 <<<"$C")" == "$CAPTURE" ]] && ok "upstream start command: the key out of its environment on its first line" \
  || bad "upstream start command line 1: $(head -1 <<<"$C")"
[[ "$(grep -c '_fvr2_' <<<"$C")" == 2 ]] && grep -qF "${HANDOFF}" <<<"$C" \
  && [[ "$(grep -F "$HANDOFF" <<<"$C")" == *'bash $RUNNER/scripts/gpu/upstream/pod.sh "$OUT" >"$OUT/pod.out" 2>&1' ]] \
  && ok "upstream start command: the key back to pod.sh alone" || bad "upstream start command: the handoff: $(grep -n _fvr2_ <<<"$C")"
out="$(env -i PATH="$PATH" FV_WEIGHTS_R2_ACCESS_KEY_ID=AKIDtest0123 FV_WEIGHTS_R2_SECRET_ACCESS_KEY=secretvalue4567 \
  bash -c "$(head -1 <<<"$C"); env; echo \"held=\${#_fvr2_id}/\${#_fvr2_secret}\"")"
! grep -qE 'AKIDtest0123|secretvalue4567' <<<"$out" && grep -qx 'held=12/15' <<<"$out" \
  && ok "upstream start command: the shell holds the key, no child's environment does" || bad "upstream first line leaks: $out"
C="$(cmd "UP_STEPS=info:box weights:h3-diffusers oracle:fasth3-4step-vsa")"
! grep -qE '_fvr2_|FV_WEIGHTS_R2' <<<"$C" && ok "upstream start command without an r2 step: no R2 line" || bad "upstream: R2 lines without r2 step"

# scripts/gpu/oracle.sh: sfwan13's r2 step and disk, as lab/upstream.toml's target.
grep -qF 'fvw="${fvw:+$fvw }r2:sfwan21-1.3b"' "$ROOT/scripts/gpu/oracle.sh" && grep -qx '  fvdisk=180' "$ROOT/scripts/gpu/oracle.sh" \
  && ok "oracle.sh: sfwan13 fetches sfwan21-1.3b on a 180 GB disk" || bad "oracle.sh: sfwan13's r2 step / disk"
grep -qx '  rt_r2=sfwan21-1.3b' "$ROOT/scripts/gpu/oracle.sh" && grep -qx '  rt_disk="${FV_CONTAINER_DISK_GB:-150}"' "$ROOT/scripts/gpu/oracle.sh" \
  && ok "oracle.sh: its runtime pod reads sfwan13's tree from R2 (FV_R2_TREES, 150 GB)" || bad "oracle.sh: the runtime side's FV_R2_TREES"
grep -qF 'r2 = ["sfwan21-1.3b"]' "$ROOT/lab/upstream.toml" && grep -qx 'container_disk_gb = 180' "$ROOT/lab/upstream.toml" \
  && ok "lab/upstream.toml: the same tree and disk" || bad "lab/upstream.toml: sfwan13's r2 / disk"

# upstream/oracle.sh's Wan 2.2 TI2V-5B references (ADR-0019 §22, S6): the
# functions cut from the script, oracle_cell stubbed to print its argv.
ORA="$ROOT/scripts/gpu/upstream/oracle.sh"
: >"$TMP/ora.sh"
for fn in oracle_wan22 oracle_fastwan22; do
  sed -n "/^$fn() {\$/,/^}\$/p" "$ORA" >>"$TMP/ora.sh"
  grep -q "^$fn() {\$" "$TMP/ora.sh" || { echo "upstream-r2: no $fn in upstream/oracle.sh" >&2; exit 2; }
done
wan_model() { # <fn> <uw>: the --model its oracle_cell gets
  env -i PATH="$PATH" bash -c '. "$1"; oracle_cell() { local p=""; for a in "$@"; do [[ "$p" == --model ]] && echo "$a"; p="$a"; done; }
    W=/workspace/weights UW="$2" UP=/up HERE=/h; "$3"' _ "$TMP/ora.sh" "$2" "$1"
}
for c in "oracle_wan22 wan22-ti2v-5b" "oracle_fastwan22 fastwan22-ti2v-5b"; do
  read -r fn tree <<<"$c"
  rm -rf "$TMP/uww"; mkdir -p "$TMP/uww/$tree"
  [[ "$(wan_model "$fn" "$TMP/uww")" == "/workspace/weights/$tree" ]] \
    && ok "$fn: the volume's $tree without a complete \$UW copy" || bad "$fn: without \$UW: $(wan_model "$fn" "$TMP/uww")"
  : >"$TMP/uww/$tree/.complete"
  [[ "$(wan_model "$fn" "$TMP/uww")" == "$TMP/uww/$tree" ]] \
    && ok "$fn: \$UW/$tree once its r2 step completed it" || bad "$fn: with \$UW: $(wan_model "$fn" "$TMP/uww")"
done
for t in "wan22-ti2v wan22-ti2v-5b" "fastwan22-ti2v fastwan22-ti2v-5b"; do
  read -r target tree <<<"$t"
  blk="$(awk -v n="name = \"$target\"" '$0 == n { f = 1; next } f && /^\[/ { exit } f' "$ROOT/lab/upstream.toml")"
  grep -qx "r2 = \[\"$tree\"\]" <<<"$blk" && grep -qx 'container_disk_gb = 210' <<<"$blk" && grep -qx 'steps = \[\]' <<<"$blk" \
    && ok "lab/upstream.toml: $target fetches $tree on 210 GB" || bad "lab/upstream.toml: $target: $blk"
done

if (( fail )); then echo "upstream-r2: FAILED" >&2; exit 1; fi
echo "upstream-r2: all ok"
