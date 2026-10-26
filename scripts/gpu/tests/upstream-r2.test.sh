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
#     lab/upstream.toml's target.
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
grep -qx 'r2_key_init' "$POD" && ok "pod.sh runs r2_key_init at its start" || bad "pod.sh runs r2_key_init"
grep -qE '^ +r2:\*\) weights_r2 "\$\{s#r2:\}" ;;$' "$POD" && ok "pod.sh dispatches r2:<tree> to weights_r2" \
  || bad "pod.sh dispatches r2:<tree>"
grep -qE '^ +echo "r2_key=\$R2_KEY_SRC"$' "$POD" && ok "info_box writes r2_key=\$R2_KEY_SRC" || bad "info_box writes r2_key="
# The key's values are read in r2_key_init and handed over in weights_r2 only.
uses="$(awk '/^r2_key_init\(\) \{$/,/^}$/ { next } /^weights_r2\(\) \{$/,/^}$/ { next } /FV_WEIGHTS_R2_|R2_KEY_ID|R2_KEY_SECRET/ && !/^ *#/' "$POD")"
[[ -z "$uses" ]] && ok "pod.sh reads the R2 key only in r2_key_init / weights_r2" || bad "pod.sh uses the key elsewhere: $uses"

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
cat >"$UP/tools/bin/python" <<EOF
#!/usr/bin/env bash
log="$TMP/calls.log"
printf 'argv:' >>"\$log"; printf ' %s' "\$@" >>"\$log"; echo >>"\$log"
echo "env: id=\${FV_WEIGHTS_R2_ACCESS_KEY_ID:+set} secret=\${FV_WEIGHTS_R2_SECRET_ACCESS_KEY:+set} skip=\${FV_R2_SKIP_TREES-unset}" >>"\$log"
env >"$TMP/env.\$(basename "\$1")"
case "\$(basename "\$1")" in
  r2-weights.py)
    out=""; prev=""; for a in "\$@"; do [[ "\$prev" == --out ]] && out="\$a"; prev="\$a"; done
    echo '{"plan":"bearer-urls"}' >"\$out" ;;
  r2-fetch.py)
    [[ -e "$TMP/fail-fetch" ]] && { echo "FAILED sfwan21-1.3b: r2 503"; exit 1; }
    root=""; status=""; prev=""; for a in "\$@"; do [[ "\$prev" == --root ]] && root="\$a"; [[ "\$prev" == --status ]] && status="\$a"; prev="\$a"; done
    cp "$TMP/state/plan.json" "$TMP/plan-seen.json" 2>/dev/null
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
  rm -rf "$TMP/out" "$TMP/state" "$TMP/calls.log" "$TMP"/env.* "$TMP/plan-seen.json"
  mkdir -p "$TMP/out"
  set +e
  OUTPUT="$(env -i PATH="$PATH" HOME="$TMP" "$@" bash -c '
    . "$1"
    HERE="$2/scripts/gpu/upstream" UP="$3" UW="$4" OUT="$5" UP_R2_STATE="$6" UP_CELL_TIMEOUT_S=60
    r2_key_init
    echo "src=$R2_KEY_SRC"
    bash -c "echo child: id=\${FV_WEIGHTS_R2_ACCESS_KEY_ID-unset} secret=\${FV_WEIGHTS_R2_SECRET_ACCESS_KEY-unset}"
    weights_r2 "$7"
  ' _ "$TMP/fn.sh" "$REPO" "$UP" "$uw" "$TMP/out" "$TMP/state" "$tree" 2>&1)"
  RC=$?
  set -e
  # The key's values: in no output, no argv, no file the run left.
  local leak
  leak="$(grep -rlF -e "$ID" -e "$SECRET" "$TMP/out" "$TMP/calls.log" "$uw" "$TMP/state" 2>/dev/null || true)"
  [[ "$OUTPUT" != *"$ID"* && "$OUTPUT" != *"$SECRET"* && -z "$leak" ]] && ok "$name: the key's values in no output, argv or file" \
    || bad "$name: the key leaked ($leak)"
}
KEY=(FV_WEIGHTS_R2_ACCESS_KEY_ID="$ID" FV_WEIGHTS_R2_SECRET_ACCESS_KEY="$SECRET")

# 1. The key, a fetch.
run "fetch" "$TMP/uw" sfwan21-1.3b "${KEY[@]}"
[[ "$RC" == 0 ]] && ok "fetch: exit 0" || bad "fetch: exit $RC: $OUTPUT"
grep -qx 'src=env' <<<"$OUTPUT" && ok "fetch: r2_key=env" || bad "fetch: $OUTPUT"
grep -qx 'child: id=unset secret=unset' <<<"$OUTPUT" && ok "fetch: no child sees the key" || bad "fetch: a child sees the key: $OUTPUT"
grep -qx "argv: $REPO/scripts/gpu/upstream/../../../scripts/gpu/r2-weights.py plan sfwan21-1.3b@$REV --pin $REPO/scripts/gpu/upstream/../../../scripts/gpu/weights-r2.tsv --ttl 660 --out $TMP/state/plan.json" "$TMP/calls.log" \
  && ok "fetch: the plan pins weights-r2.tsv's revision" || bad "fetch: plan argv: $(cat "$TMP/calls.log")"
[[ "$(sed -n 2p "$TMP/calls.log")" == "env: id=set secret=set skip=unset" ]] && ok "fetch: the plan step has the key" \
  || bad "fetch: plan env: $(sed -n 2p "$TMP/calls.log")"
grep -qx "argv: $REPO/scripts/gpu/upstream/../../../scripts/gpu/r2-fetch.py --root $TMP/uw --plan $TMP/state/plan.json --skip-trees  --status $TMP/out/r2-sfwan21-1.3b.json" "$TMP/calls.log" \
  && ok "fetch: r2-fetch.py into \$UW, every tree, a status file" || bad "fetch: fetch argv: $(cat "$TMP/calls.log")"
[[ "$(sed -n 4p "$TMP/calls.log")" == "env: id= secret= skip=unset" ]] && ok "fetch: the fetch step has no key" \
  || bad "fetch: fetch env: $(sed -n 4p "$TMP/calls.log")"
! grep -q FV_WEIGHTS_R2_ "$TMP/env.r2-fetch.py" && ok "fetch: no key name in the fetch's environment" || bad "fetch: key in the fetch's env"
[[ -f "$TMP/plan-seen.json" && ! -e "$TMP/state/plan.json" ]] && ok "fetch: the plan existed for the fetch, removed after" \
  || bad "fetch: plan lifecycle"
[[ "$(stat -c %a "$TMP/state")" == 700 ]] && ok "fetch: the plan's directory is private (700)" || bad "fetch: state mode $(stat -c %a "$TMP/state")"
[[ -f "$TMP/uw/sfwan21-1.3b/.complete" && -f "$TMP/out/r2-sfwan21-1.3b.json" ]] && ok "fetch: the tree in \$UW, the status in \$OUT" \
  || bad "fetch: no tree or status"
# 2. Already there at its revision: nothing called, no key needed.
run "present" "$TMP/uw" sfwan21-1.3b
[[ "$RC" == 0 && ! -s "$TMP/calls.log" ]] && grep -q "already at $TMP/uw/sfwan21-1.3b" <<<"$OUTPUT" \
  && ok "present: kept, nothing fetched" || bad "present: rc $RC, $OUTPUT"
# 3. A failed fetch: the step fails, the plan is still removed.
rm -rf "$TMP/uw"; : >"$TMP/fail-fetch"
run "failed fetch" "$TMP/uw" sfwan21-1.3b "${KEY[@]}"
[[ "$RC" != 0 && ! -e "$TMP/state/plan.json" ]] && grep -q 'r2-fetch.py exit 1' <<<"$OUTPUT" \
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

# scripts/gpu/oracle.sh: sfwan13's r2 step and disk, as lab/upstream.toml's target.
grep -qF 'fvw="${fvw:+$fvw }r2:sfwan21-1.3b"' "$ROOT/scripts/gpu/oracle.sh" && grep -qx '  fvdisk=180' "$ROOT/scripts/gpu/oracle.sh" \
  && ok "oracle.sh: sfwan13 fetches sfwan21-1.3b on a 180 GB disk" || bad "oracle.sh: sfwan13's r2 step / disk"
grep -qF 'r2 = ["sfwan21-1.3b"]' "$ROOT/lab/upstream.toml" && grep -qx 'container_disk_gb = 180' "$ROOT/lab/upstream.toml" \
  && ok "lab/upstream.toml: the same tree and disk" || bad "lab/upstream.toml: sfwan13's r2 / disk"

if (( fail )); then echo "upstream-r2: FAILED" >&2; exit 1; fi
echo "upstream-r2: all ok"
