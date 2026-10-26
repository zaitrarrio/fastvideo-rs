#!/usr/bin/env bash
# runpod-http.sh run with FV_R2_TREES (docs/adr/0019-fv-lab.md §21.8: the
# script runner of the oracle-r2 GPU check reads its R2 trees as fv-lab's
# volume+r2 runner does), without a pod, a key or a network:
#   - the refusals, before the key and before anything is made: a bad name, a
#     tree weights-r2.tsv lacks, a tree on the EU volume, under one or holding
#     one, a tree twice, a container disk too small for the trees and 120 GB,
#     another mode than run / plan, RUNPOD_NO_VOLUME=1, a family fv-lab runs;
#   - unset is a no-op: the plan body byte-identical with FV_R2_TREES unset,
#     empty or blank, and none of the R2 lines in it;
#   - the body: the R2 key as the two Runpod secret references fv-lab's
#     volume+r2 pods carry (never a value), the disk asked for, fv-lab's idle
#     limit; its start command takes the key out of its own environment on
#     its first line (a child sees none) and gives it to r2-stage.sh alone;
#   - scripts/gpu/r2-stage.sh with a fake python and apt-get: the key in the plan
#     step's environment only (never an argv, an output or a file), the plan
#     pinned, in a 700 directory, removed whatever happened (a failed fetch
#     too); fv-lab's links into the volume made, never written through; the
#     root, the state directory or a tree landing in a link on the volume
#     refused; no key, an unfilled reference: nothing fetched; a complete tree
#     kept; python3 installed with apt when missing.
#
#   bash scripts/gpu/tests/runpod-r2.test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HTTP="$ROOT/scripts/gpu/runpod-http.sh"
STAGE="$ROOT/scripts/gpu/r2-stage.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
ok() { printf 'ok   %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1" >&2; fail=1; }

# http <K=V …> -- <args>: runpod-http.sh in a clean env; stdout to $TMP/out,
# stderr to $TMP/err, the status in RC.
http() {
  local kv=()
  while [[ "$1" != -- ]]; do kv+=("$1"); shift; done
  shift
  set +e
  env -i PATH="$PATH" HOME="$TMP" LC_ALL=C ${kv[@]+"${kv[@]}"} bash "$HTTP" "$@" >"$TMP/out" 2>"$TMP/err"
  RC=$?
  set -e
}
refused() { # <name> <want in stderr> <K=V …> -- <args>
  local name="$1" want="$2"
  shift 2
  http "$@"
  [[ "$RC" == 2 ]] && grep -qF -- "$want" "$TMP/err" && ok "refused: $name" \
    || bad "refused: $name: exit $RC, $(cat "$TMP/err")"
}
PLAN=(FV_PLAN_VOLUME="bz0ofkfx45 EU-RO-1" FV_POD_PREFIX=fv-lab FV_FAMILY=oracle FV_GEN_TIMEOUT_S=5400)

# 1. Refusals: no RUNPOD_API_KEY is set, so a refusal that came after the key
#    check would read "RUNPOD_API_KEY missing" instead.
refused "a bad name" "is not a tree name" FV_R2_TREES=../x FV_CONTAINER_DISK_GB=150 -- plan 1234567 t
refused "an absolute name" "is not a tree name" FV_R2_TREES=/abs FV_CONTAINER_DISK_GB=150 -- plan 1234567 t
refused "a tree the table lacks" "is not a dest of scripts/gpu/weights-r2.tsv" FV_R2_TREES=no-such-tree FV_CONTAINER_DISK_GB=150 -- plan 1234567 t
refused "a volume tree" "is on the EU volume (ltx25)" FV_R2_TREES=ltx25 FV_CONTAINER_DISK_GB=500 -- plan 1234567 t
refused "under a volume tree" "is on the EU volume (h3-base)" FV_R2_TREES=h3-base/text_encoder_fp8 FV_CONTAINER_DISK_GB=500 -- plan 1234567 t
refused "holding a volume tree" "holds the EU volume's tree auxiliary/tae" FV_R2_TREES=auxiliary FV_CONTAINER_DISK_GB=500 -- plan 1234567 t
refused "a tree twice" "sfwan21-1.3b twice" "FV_R2_TREES=sfwan21-1.3b sfwan21-1.3b" FV_CONTAINER_DISK_GB=500 -- plan 1234567 t
refused "a disk too small" "FV_CONTAINER_DISK_GB=149 or more" FV_R2_TREES=sfwan21-1.3b FV_CONTAINER_DISK_GB=148 -- plan 1234567 t
refused "the default disk" "FV_CONTAINER_DISK_GB=149 or more" FV_R2_TREES=sfwan21-1.3b -- plan 1234567 t
for mode in all kernels weights upstream plan-upstream; do
  refused "mode $mode" "for \`run\` and \`plan\` only" FV_R2_TREES=sfwan21-1.3b FV_CONTAINER_DISK_GB=150 -- "$mode" 1234567
done
refused "no volume" "not with RUNPOD_NO_VOLUME=1" FV_R2_TREES=sfwan21-1.3b FV_CONTAINER_DISK_GB=150 RUNPOD_NO_VOLUME=1 -- plan 1234567 t
# A family fv-lab runs takes its trees from its tier (weights = "volume+r2").
http FV_DRY_RUN=1 FV_FAMILY=rtx6000 FV_R2_TREES=sfwan21-1.3b FV_CONTAINER_DISK_GB=150 -- run 1234567
[[ "$RC" != 0 ]] && grep -q 'has no FV_R2_TREES' "$TMP/err" && ok "refused: a family fv-lab runs" \
  || bad "a family fv-lab runs: exit $RC, $(cat "$TMP/err")"

# 2. Unset is a no-op: unset, empty and blank give the same body, without R2 lines.
http "${PLAN[@]}" RUNPOD_API_KEY=x -- plan 1234567 1234567-10241200
cp "$TMP/out" "$TMP/unset.json"
for v in "" "  "; do
  http "${PLAN[@]}" RUNPOD_API_KEY=x FV_R2_TREES="$v" -- plan 1234567 1234567-10241200
  cmp -s "$TMP/out" "$TMP/unset.json" && ok "FV_R2_TREES='$v': the body unchanged" || bad "FV_R2_TREES='$v' changed the body"
done
FX="$ROOT/crates/fastvideo-lab/fixtures"
[[ "$(jq -S . "$TMP/unset.json")" == "$(cat "$FX/runpod-http-oracle.json")" ]] \
  && ok "unset: the body is the committed oracle golden, byte for byte" || bad "unset: the body differs from fixtures/runpod-http-oracle.json"
! grep -qE 'r2-stage|_fvr2|FV_WEIGHTS_R2|FV_WORK=\$SCRATCH' "$TMP/unset.json" && jq -e '.env == null' "$TMP/unset.json" >/dev/null \
  && ok "unset: no R2 line, no env, FV_WORK=/workspace" || bad "unset: R2 lines in the body"

# 3. The body with the tree.
http "${PLAN[@]}" RUNPOD_API_KEY=x FV_R2_TREES=sfwan21-1.3b FV_CONTAINER_DISK_GB=150 -- plan 1234567 1234567-10241200
B="$TMP/r2.json"; cp "$TMP/out" "$B"
[[ "$(jq -c .env "$B")" == '{"FV_WEIGHTS_R2_ACCESS_KEY_ID":"{{ RUNPOD_SECRET_fv_r2_access_key_id }}","FV_WEIGHTS_R2_SECRET_ACCESS_KEY":"{{ RUNPOD_SECRET_fv_r2_secret_access_key }}"}' ]] \
  && ok "body: the two Runpod secret references, nothing else" || bad "body env: $(jq -c .env "$B")"
[[ "$(jq .containerDiskInGb "$B")" == 150 ]] && ok "body: the disk asked for" || bad "body disk"
[[ "$(jq -S . "$B")" == "$(cat "$FX/runpod-http-oracle-sfwan13-r2.json")" ]] && ok "body: its golden" || bad "body differs from its golden"
CMD="$(jq -r '.dockerStartCmd[2]' "$B")"
[[ "$(head -1 <<<"$CMD")" == '_fvr2_id="${FV_WEIGHTS_R2_ACCESS_KEY_ID:-}" _fvr2_secret="${FV_WEIGHTS_R2_SECRET_ACCESS_KEY:-}"; unset FV_WEIGHTS_R2_ACCESS_KEY_ID FV_WEIGHTS_R2_SECRET_ACCESS_KEY' ]] \
  && ok "start command: the key out of its environment on its first line" || bad "start command line 1: $(head -1 <<<"$CMD")"
grep -qF 'lim=115; [ "$used" = 1 ] && lim=115' <<<"$CMD" && ok "start command: fv-lab's idle limit (r2_idle_min(5400) = 115)" \
  || bad "start command: idle limit"
grep -qF 'FV_WORK=$SCRATCH FV_SCRATCH=$SCRATCH' <<<"$CMD" && ok "start command: the matrix's \$W is the merged root" || bad "FV_WORK"
STAGE_LINE="$(grep -A1 'bash /opt/fastvideo-rs/scripts/gpu/r2-stage.sh' <<<"$CMD" | tr -d '\\\n')"
REV="$(awk -F'\t' '$1 == "sfwan21-1.3b" { print $2; exit }' "$ROOT/scripts/gpu/weights-r2.tsv")"
[[ "$STAGE_LINE" == 'FV_WEIGHTS_R2_ACCESS_KEY_ID="$_fvr2_id" FV_WEIGHTS_R2_SECRET_ACCESS_KEY="$_fvr2_secret" bash /opt/fastvideo-rs/scripts/gpu/r2-stage.sh '*" \$SCRATCH/weights \$SCRATCH/r2 \"\$OUT\" 5400 /workspace/weights \"FastH3-4-step-Preview-v1-LoRA auxiliary/lpips auxiliary/tae h3-base h3-ref2va h3-to-ltx ltx25 ltx25-dev ltx25-ic-lora-ingredients upscaler\" sfwan21-1.3b@$REV >\"\$OUT/r2-stage.log\" 2>&1" ]] \
  && ok "start command: r2-stage.sh with fv-lab's link roots and the pinned revision" || bad "stage line: $STAGE_LINE"
# The first line run with a key in the environment: a child sees none of it.
out="$(env -i PATH="$PATH" FV_WEIGHTS_R2_ACCESS_KEY_ID=AKIDtest0123 FV_WEIGHTS_R2_SECRET_ACCESS_KEY=secretvalue4567 \
  bash -c "$(head -1 <<<"$CMD"); env; echo \"held=\${#_fvr2_id}/\${#_fvr2_secret}\"")"
! grep -qE 'AKIDtest0123|secretvalue4567' <<<"$out" && grep -qx 'held=12/15' <<<"$out" \
  && ok "start command: the shell holds the key, no child's environment does" || bad "the first line leaks: $out"

# 4. r2-stage.sh, with a fake python (r2-weights.py / r2-fetch.py) and apt-get.
REPO="$TMP/repo"
mkdir -p "$REPO/scripts/gpu" "$TMP/bin"
cp "$STAGE" "$ROOT/scripts/gpu/weights-r2.tsv" "$REPO/scripts/gpu/"
cat >"$TMP/bin/python3" <<EOF
#!/usr/bin/env bash
if [[ "\$1" == -c ]]; then [[ -e "$TMP/have-python" ]]; exit; fi
log="$TMP/calls.log"
printf 'argv:' >>"\$log"; printf ' %s' "\$@" >>"\$log"; echo >>"\$log"
echo "env: id=\${FV_WEIGHTS_R2_ACCESS_KEY_ID:+set} secret=\${FV_WEIGHTS_R2_SECRET_ACCESS_KEY:+set} skip=\${FV_R2_SKIP_TREES-unset}" >>"\$log"
case "\$(basename "\$1")" in
  r2-weights.py)
    out=""; prev=""; for a in "\$@"; do [[ "\$prev" == --out ]] && out="\$a"; prev="\$a"; done
    echo '{"plan":"bearer-urls"}' >"\$out"
    if [[ -e "$TMP/fail-plan" ]]; then exit 1; fi ;;
  r2-fetch.py)
    root=""; status=""; plan=""; prev=""
    for a in "\$@"; do [[ "\$prev" == --root ]] && root="\$a"; [[ "\$prev" == --status ]] && status="\$a"; [[ "\$prev" == --plan ]] && plan="\$a"; prev="\$a"; done
    cp "\$plan" "$TMP/plan-seen.json" 2>/dev/null
    [[ -e "$TMP/fail-fetch" ]] && { echo "FAILED sfwan21-1.3b: r2 503"; exit 1; }
    mkdir -p "\$root/sfwan21-1.3b"; : >"\$root/sfwan21-1.3b/.complete"; echo "$REV" >"\$root/sfwan21-1.3b/.fv-revision"
    echo '{"state":"done"}' >"\$status" ;;
esac
EOF
cat >"$TMP/bin/apt-get" <<EOF
#!/usr/bin/env bash
echo "apt-get \$*" >>"$TMP/apt.log"
[[ "\$*" == *install* ]] && : >"$TMP/have-python"
exit 0
EOF
chmod +x "$TMP/bin/python3" "$TMP/bin/apt-get"
: >"$TMP/have-python"
VOL="$TMP/volume/weights"
mkdir -p "$VOL/h3-base" "$VOL/auxiliary/tae"
LINKS="auxiliary/tae h3-base"
ID=AKIDtest0123 SECRET=secretvalue4567
KEY=(FV_WEIGHTS_R2_ACCESS_KEY_ID="$ID" FV_WEIGHTS_R2_SECRET_ACCESS_KEY="$SECRET")
# stage <name> <root> <state> <K=V …> -- <specs…>
stage() {
  local name="$1" root="$2" state="$3" kv=()
  shift 3
  while [[ "$1" != -- ]]; do kv+=("$1"); shift; done
  shift
  rm -rf "$TMP/out" "$TMP/calls.log" "$TMP/plan-seen.json"
  mkdir -p "$TMP/out"
  set +e
  OUTPUT="$(env -i PATH="$TMP/bin:$PATH" HOME="$TMP" ${kv[@]+"${kv[@]}"} \
    bash "$REPO/scripts/gpu/r2-stage.sh" "$root" "$state" "$TMP/out" 60 "$VOL" "$LINKS" "$@" 2>&1)"
  RC=$?
  set -e
  local leak
  leak="$(grep -rlF -e "$ID" -e "$SECRET" "$TMP/out" "$TMP/calls.log" "$root" "$state" 2>/dev/null || true)"
  [[ "$OUTPUT" != *"$ID"* && "$OUTPUT" != *"$SECRET"* && -z "$leak" ]] && ok "$name: the key's values in no output, argv or file" \
    || bad "$name: the key leaked ($leak)"
}
R="$TMP/scratch/weights" S="$TMP/scratch/r2"
stage "fetch" "$R" "$S" "${KEY[@]}" -- "sfwan21-1.3b@$REV"
[[ "$RC" == 0 ]] && grep -q 'r2_key=env' <<<"$OUTPUT" && ok "fetch: exit 0, r2_key=env" || bad "fetch: exit $RC: $OUTPUT"
grep -qx "argv: $REPO/scripts/gpu/r2-weights.py plan sfwan21-1.3b@$REV --pin $REPO/scripts/gpu/weights-r2.tsv --ttl 660 --out $S/plan.json" "$TMP/calls.log" \
  && ok "fetch: the plan pins weights-r2.tsv's revision" || bad "fetch: plan argv: $(cat "$TMP/calls.log")"
[[ "$(sed -n 2p "$TMP/calls.log")" == "env: id=set secret=set skip=unset" ]] && ok "fetch: the plan step has the key" || bad "plan env"
grep -qx "argv: $REPO/scripts/gpu/r2-fetch.py --root $R --plan $S/plan.json --skip-trees  --status $TMP/out/r2-fetch.json" "$TMP/calls.log" \
  && ok "fetch: r2-fetch.py into the merged root, every tree, a status" || bad "fetch argv: $(cat "$TMP/calls.log")"
[[ "$(sed -n 4p "$TMP/calls.log")" == "env: id= secret= skip=unset" ]] && ok "fetch: the fetch step has no key" || bad "fetch env"
[[ -f "$TMP/plan-seen.json" && ! -e "$S/plan.json" ]] && ok "fetch: the plan there for the fetch, removed after" || bad "plan lifecycle"
[[ "$(stat -c %a "$S")" == 700 ]] && ok "fetch: the state directory is 700" || bad "state mode $(stat -c %a "$S")"
[[ "$(readlink "$R/h3-base")" == "$VOL/h3-base" && "$(readlink "$R/auxiliary/tae")" == "$VOL/auxiliary/tae" && -d "$R/auxiliary" && ! -L "$R/auxiliary" ]] \
  && ok "fetch: fv-lab's links (auxiliary/tae linked, auxiliary a directory)" || bad "links: $(ls -lR "$R" | head)"
[[ -f "$R/sfwan21-1.3b/.complete" && ! -L "$R/sfwan21-1.3b" ]] && ok "fetch: the tree a directory of the root" || bad "no tree"
stage "present" "$R" "$S" -- "sfwan21-1.3b@$REV"
[[ "$RC" == 0 && ! -s "$TMP/calls.log" ]] && grep -q 'already in' <<<"$OUTPUT" && ok "present: kept, nothing fetched, no key needed" \
  || bad "present: exit $RC, $OUTPUT"
rm -rf "$TMP/scratch"; : >"$TMP/fail-fetch"
stage "failed fetch" "$R" "$S" "${KEY[@]}" -- "sfwan21-1.3b@$REV"
[[ "$RC" != 0 && -f "$TMP/plan-seen.json" && ! -e "$S/plan.json" ]] && grep -q 'r2-fetch.py exit 1' <<<"$OUTPUT" \
  && ok "failed fetch: fails, the plan removed" || bad "failed fetch: exit $RC, $OUTPUT"
rm -f "$TMP/fail-fetch"; rm -rf "$TMP/scratch"
: >"$TMP/fail-plan"
stage "failed plan" "$R" "$S" "${KEY[@]}" -- "sfwan21-1.3b@$REV"
[[ "$RC" != 0 && ! -e "$S/plan.json" ]] && grep -q 'r2-weights.py plan failed' <<<"$OUTPUT" \
  && ok "failed plan: fails, a half-written plan removed" || bad "failed plan: exit $RC, $OUTPUT"
rm -f "$TMP/fail-plan"; rm -rf "$TMP/scratch"
for c in "no key::none" "blank:FV_WEIGHTS_R2_ACCESS_KEY_ID= FV_WEIGHTS_R2_SECRET_ACCESS_KEY=:none" "half:FV_WEIGHTS_R2_ACCESS_KEY_ID=$ID:none (env: incomplete)"; do
  IFS=: read -r name kv want <<<"$c"
  args=(); [[ -n "$kv" ]] && read -r -a args <<<"$kv"
  stage "$name" "$R" "$S" ${args[@]+"${args[@]}"} -- "sfwan21-1.3b@$REV"
  [[ "$RC" != 0 && ! -s "$TMP/calls.log" ]] && grep -q "r2_key=$want" <<<"$OUTPUT" && grep -q 'no R2 key' <<<"$OUTPUT" \
    && ok "$name: r2_key=$want, nothing fetched" || bad "$name: exit $RC, $OUTPUT"
done
stage "unfilled references" "$R" "$S" FV_WEIGHTS_R2_ACCESS_KEY_ID='{{ RUNPOD_SECRET_fv_r2_access_key_id }}' \
  FV_WEIGHTS_R2_SECRET_ACCESS_KEY='{{ RUNPOD_SECRET_fv_r2_secret_access_key }}' -- "sfwan21-1.3b@$REV"
[[ "$RC" != 0 && ! -s "$TMP/calls.log" ]] && grep -q 'r2_key=none (env: unfilled secret reference)' <<<"$OUTPUT" \
  && ok "unfilled references: nothing fetched" || bad "unfilled: exit $RC, $OUTPUT"
# python3 missing: apt-get installs it, then the fetch runs.
rm -f "$TMP/have-python" "$TMP/apt.log"; rm -rf "$TMP/scratch"
stage "no python3" "$R" "$S" "${KEY[@]}" -- "sfwan21-1.3b@$REV"
[[ "$RC" == 0 ]] && grep -q 'install -y -qq --no-install-recommends python3' "$TMP/apt.log" && ok "no python3: apt-get, then the fetch" \
  || bad "no python3: exit $RC, $OUTPUT $(cat "$TMP/apt.log" 2>/dev/null)"
# Never on the volume.
stage "root on /workspace" /workspace/x "$S" "${KEY[@]}" -- "sfwan21-1.3b@$REV"
[[ "$RC" != 0 && ! -s "$TMP/calls.log" ]] && grep -q 'is on the volume' <<<"$OUTPUT" && ok "root on /workspace: refused" || bad "root on /workspace: $OUTPUT"
stage "root in the volume's weights" "$VOL/merged" "$S" "${KEY[@]}" -- "sfwan21-1.3b@$REV"
[[ "$RC" != 0 && ! -e "$VOL/merged" ]] && ok "root in the volume's weights: refused, nothing made" || bad "root in volume: $OUTPUT"
stage "state in the volume" "$R" "$VOL/r2" "${KEY[@]}" -- "sfwan21-1.3b@$REV"
[[ "$RC" != 0 && ! -e "$VOL/r2" ]] && ok "state in the volume: refused" || bad "state in volume: $OUTPUT"
rm -rf "$TMP/scratch"
stage "a tree inside a link" "$R" "$S" "${KEY[@]}" -- "auxiliary/tae/x@r1"
[[ "$RC" != 0 && ! -s "$TMP/calls.log" ]] && grep -q 'outside' <<<"$OUTPUT" && [[ -z "$(ls -A "$VOL/auxiliary/tae")" ]] \
  && ok "a tree inside a link into the volume: refused, nothing written there" || bad "tree in link: $OUTPUT"
for spec in "../x@r1" "a b@r1" "sfwan21-1.3b" "x@r/1"; do
  stage "spec '$spec'" "$R" "$S" "${KEY[@]}" -- "$spec"
  [[ "$RC" != 0 && ! -s "$TMP/calls.log" ]] && ok "spec '$spec': refused" || bad "spec '$spec': $OUTPUT"
done

if (( fail )); then echo "runpod-r2: FAILED" >&2; exit 1; fi
echo "runpod-r2: all ok"
