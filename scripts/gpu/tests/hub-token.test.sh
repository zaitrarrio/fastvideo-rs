#!/usr/bin/env bash
# The upstream pod's Hugging Face token (docs/adr/0019-fv-lab.md §20.7,
# docs/ops/runpod-secrets.md), without a pod, a key or a network:
#   - scripts/gpu/upstream/pod.sh's hub_token_init (cut from the script, run
#     in a clean bash) on each input: no env, a blank env, an unfilled Runpod
#     secret reference, a real value (with whitespace), a token file, an
#     empty token file, and the reference with a file. HF_TOKEN and the
#     box.txt line (`hf_token=<source>`, as info_box writes it) are checked,
#     and the token's value never appears in the line or in any output;
#   - pod.sh prints HF_TOKEN nowhere (info_box writes its source only);
#   - runpod-http.sh plan-upstream: the reference (never a value) for a
#     weights:ltx25* step whatever whitespace separates the steps, none for
#     other steps, UP_HUB_TOKEN=1 / 0 forcing it, any other value refused.
#
#   bash scripts/gpu/tests/hub-token.test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
POD="$ROOT/scripts/gpu/upstream/pod.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
ok() { printf 'ok   %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1" >&2; fail=1; }

sed -n '/^hub_token_init() {$/,/^}$/p' "$POD" >"$TMP/fn.sh"
grep -q '^hub_token_init() {$' "$TMP/fn.sh" || { echo "hub-token: no hub_token_init in pod.sh" >&2; exit 2; }
grep -qx 'hub_token_init /workspace/hf/token' "$POD" || bad "pod.sh calls hub_token_init /workspace/hf/token"
# The line info_box writes into box.txt.
BOX_LINE="$(grep -E '^ +echo "hf_token=\$HF_TOKEN_SRC"$' "$POD" | sed 's/^ *//')"
[[ "$BOX_LINE" == 'echo "hf_token=$HF_TOKEN_SRC"' ]] && ok "info_box writes hf_token=\$HF_TOKEN_SRC" \
  || bad "info_box writes hf_token=\$HF_TOKEN_SRC"
# Every read of HF_TOKEN's value in pod.sh is inside hub_token_init.
uses="$(grep -nE '\$\{?HF_TOKEN([^_A-Za-z]|$)' "$POD" | grep -vF -e 'v="${HF_TOKEN:-}"' -e '"${HF_TOKEN+x}"' || true)"
[[ -z "$uses" ]] && ok "pod.sh reads HF_TOKEN's value only in hub_token_init" || bad "pod.sh uses HF_TOKEN elsewhere: $uses"

FILE="$TMP/hf/token"
mkdir -p "$TMP/hf"
# case <name> <env: unset | the value> <file: none | its content> <want box line> <want HF_TOKEN or ->
case_() {
  local name="$1" env="$2" file="$3" want_box="$4" want_tok="$5" out tok secret
  rm -f "$FILE"
  [[ "$file" == none ]] || printf '%s' "$file" >"$FILE"
  local args=(env -i PATH="$PATH" HOME="$TMP")
  [[ "$env" == unset ]] || args+=(HF_TOKEN="$env")
  # The function's own output, then the box line, then HF_TOKEN as a child sees it (to a file).
  out="$("${args[@]}" bash -c '. "$1"; hub_token_init "$2"; echo "hf_token=$HF_TOKEN_SRC"; bash -c '"'"'printf %s "${HF_TOKEN-<unset>}"'"'"' >"$3"' \
    _ "$TMP/fn.sh" "$FILE" "$TMP/tok" 2>&1)"
  tok="$(cat "$TMP/tok")"
  [[ "$out" == "hf_token=$want_box" ]] && ok "$name: box.txt hf_token=$want_box" || bad "$name: output '$out', want 'hf_token=$want_box'"
  [[ "$want_tok" == - ]] && want_tok="<unset>"
  [[ "$tok" == "$want_tok" ]] && ok "$name: HF_TOKEN ${want_tok/#[a-z]*/(the value)}" || bad "$name: HF_TOKEN not what was wanted"
  # No token value in what was printed (the box line included).
  for secret in envtok456 filetok123; do
    [[ "$out" != *"$secret"* ]] || bad "$name: the token's value is in the output"
  done
}
REF='{{ RUNPOD_SECRET_fv_hf_token }}'
case_ "no env, no file"          unset            none                    "none"                                      -
case_ "blank env"                $' \t '          none                    "none (env: blank)"                         -
case_ "unfilled reference"       "$REF"           none                    "none (env: unfilled secret reference)"     -
case_ "a value, trimmed"         $' envtok456 \n' none                    "env"                                       envtok456
case_ "a value beats the file"   envtok456        $'filetok123\n'         "env"                                       envtok456
case_ "the file"                 unset            $'filetok123\n'         "$FILE"                                     filetok123
case_ "an empty file"            unset            $' \n'                  "none (empty $FILE)"                        -
case_ "reference, then the file" "$REF"           filetok123              "$FILE (env: unfilled secret reference)"    filetok123
case_ "blank, empty file"        ""               ""                      "none (empty $FILE) (env: blank)"           -

# runpod-http.sh plan-upstream: the pod body's env.
plan() { # <K=V …>: the body's env as compact JSON, or "refused"
  env -i PATH="$PATH" HOME="$TMP" LC_ALL=C RUNPOD_API_KEY=test FV_PLAN_VOLUME="vol0 EU-RO-1" \
    UP_SHA=1234567890abcdef1234567890abcdef12345678 "$@" \
    bash "$ROOT/scripts/gpu/runpod-http.sh" plan-upstream 1234567 1234567-10261200 2>"$TMP/err" \
    | jq -c '.env // {}' 2>/dev/null || echo refused
}
WANT='{"HF_TOKEN":"{{ RUNPOD_SECRET_fv_hf_token }}"}'
check_plan() { # <name> <want> <K=V …>
  local name="$1" want="$2" got
  shift 2
  got="$(plan "$@")"
  [[ "$got" == "$want" ]] && ok "plan-upstream $name: $got" || bad "plan-upstream $name: got $got, want $want"
}
check_plan "weights:ltx25" "$WANT" "UP_STEPS=info:box weights:ltx25 oracle:ltx25-512p"
check_plan "weights:ltx25-dev after a tab" "$WANT" $'UP_STEPS=info:box\tweights:ltx25-dev oracle:ltx25-a2v-guided'
check_plan "weights:ltx25 after a newline" "$WANT" $'UP_STEPS=info:box\nweights:ltx25'
check_plan "weights:ltx25 in UP_STEPS_BG" "$WANT" "UP_STEPS=info:box" "UP_STEPS_BG=weights:ltx25"
check_plan "the H3 steps" '{}' "UP_STEPS=info:box weights:h3-diffusers oracle:fasth3-4step-vsa"
check_plan "UP_HUB_TOKEN=1" "$WANT" "UP_STEPS=info:box" UP_HUB_TOKEN=1
check_plan "UP_HUB_TOKEN=0" '{}' "UP_STEPS=info:box weights:ltx25" UP_HUB_TOKEN=0
check_plan "UP_HUB_TOKEN=yes" refused "UP_STEPS=info:box weights:ltx25" UP_HUB_TOKEN=yes
grep -q "UP_HUB_TOKEN is 1" "$TMP/err" && ok "UP_HUB_TOKEN=yes: refused with its reason" || bad "UP_HUB_TOKEN=yes: no reason"

exit "$fail"
