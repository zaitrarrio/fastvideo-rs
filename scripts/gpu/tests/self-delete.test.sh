#!/usr/bin/env bash
# The in-pod self-delete (docs/adr/0023-lab-report-and-ci-bridge.md §4): inside a
# Runpod pod, the pod-scoped RUNPOD_API_KEY may not DELETE /v1/pods/$RUNPOD_POD_ID
# through REST (HTTP 403); GraphQL podTerminate with the same key works. Every
# in-pod self-delete calls fv_self_delete, defined once in
# scripts/gpu/lib/self-delete.sh. This check fails when:
#
#   1. REST: (a) any tracked file deletes the pod's own id (RUNPOD_POD_ID or its
#      alias FV_POD_ID) through REST, except with fv-control's account key
#      (FV_BACKSTOP_API_KEY) and the build pod's start command (REST, then GraphQL);
#      (b) any file outside LOCAL below (the files that run on the caller's machine
#      with the account key, or are tests) DELETEs a Runpod pod through REST at
#      all, whatever the variable: curl, `rest DELETE`, `.delete(` (Python
#      statements are joined across lines);
#   2. a copy of fv_self_delete differs from the library, or a call comes before
#      the definition (inline in a pod script, pod-start.sh.in, the
#      runpod-http-*.json fixtures that pin runpod-http.sh's start command);
#   3. a launcher (a script that calls fv_self_delete without defining it) does
#      not render, through the same function its `up` sends, a command with
#      exactly one copy defined before its call; that command's self-delete,
#      run with a stand-in curl and sleep, makes exactly one GraphQL call on
#      HTTP 200 and retries on a 403 (renderers: `<x>-validate.sh print-start`,
#      `ltx-pod.sh entry`);
#   4. the function itself misbehaves against a stand-in curl: success only on
#      HTTP 200 without "errors"; the endpoint, headers and mutation; the key on
#      curl's stdin, never in its argv or the log; the answer file removed.
#
#   bash scripts/gpu/tests/self-delete.test.sh
#
# No network. Needs bash, awk, jq, git, timeout.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LIB=scripts/gpu/lib/self-delete.sh
ME=scripts/gpu/tests/self-delete.test.sh
cd "$ROOT" || exit 2
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
bad() { printf 'self-delete: %s\n' "$*" >&2; fail=1; }

# Files that may DELETE a Runpod pod through REST (rule 1b): they run on the
# caller's machine (or fv-control, or the autoscaler) with the account key, are
# tests, or run on a pod with a key that is not pod-scoped / a GraphQL fallback.
LOCAL=(
  # in-pod, fv-control's account key FV_BACKSTOP_API_KEY (rule 1a still checks the line)
  control/src/cluster/payloads.ts
  crates/fastvideo-deploy/src/boot/watchdog.rs
  # in-pod build pod: REST stop, REST delete, then GraphQL podTerminate
  control/src/buildpod-start.ts
  scripts/dev/build-pod-server.py
  # fv-control, the autoscaler (account key)
  control/src/runpod.ts
  crates/fastvideo-autoscale/src/provider/runpod.rs
  # tests
  crates/fastvideo-deploy/src/provider/tests.rs
  crates/fastvideo-deploy/tests/fv_deploy_bin.rs
  scripts/dev/test_build_pod_server.py
  # local launchers: create, delete and back-stop their pods from the caller's machine
  scripts/gpu/family-pool-validate.sh
  scripts/gpu/experiments/fasth3-nvfp4-validate.sh
  scripts/gpu/experiments/h3-fp4attn-validate.sh
  scripts/gpu/experiments/h3-glue-validate.sh
  scripts/gpu/experiments/h3-graphs-validate.sh
  scripts/gpu/experiments/h3-tier-options-validate.sh
  scripts/gpu/experiments/h3-tier-ship-validate.sh
  scripts/gpu/experiments/h3-trim-nvfp4-validate.sh
  scripts/gpu/experiments/h3-trim-validate.sh
  scripts/gpu/experiments/h3-turbo-loras-validate.sh
  scripts/gpu/experiments/vdn-h3-validate.sh
  scripts/gpu/fetch-h3-ref2va.sh
  scripts/gpu/fetch-hub-tree.sh
  scripts/gpu/fetch-ltx-iclora.sh
  scripts/gpu/fetch-mmaudio.sh
  scripts/gpu/r2-copy-volume.sh
  scripts/gpu/r2-fetch-bench.sh
  scripts/gpu/r2-fill-volume.sh
  scripts/gpu/r2-publish-hub.sh
  scripts/gpu/runpod-http.sh
  scripts/gpu/runpod.sh
  scripts/serve/e2e/ltx-pod.sh
  scripts/serve/e2e/pod.sh
  scripts/serve/e2e/segments-pod.sh
  scripts/serve/e2e/wan-pod.sh
  scripts/serve/runpod-pod.sh
  scripts/serve/runpod-sfwan-whip.sh
)
is_local() { local x; for x in "${LOCAL[@]}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }

# Every fv_self_delete block in stdin, dedented to the column of its
# `fv_self_delete() {`, each followed by a line "@@END@@".
blocks() {
  awk '
    !inb && (c = index($0, "fv_self_delete() {")) > 0 {
      inb = 1; col = c - 1; pad = sprintf("%" col "s", ""); print "fv_self_delete() {"; next
    }
    inb {
      if ($0 == pad "}") { print "}"; print "@@END@@"; inb = 0; next }
      if (substr($0, 1, col) != pad && $0 != "") { print "@@BADINDENT@@ " $0 }
      print substr($0, col + 1)
    }
    END { if (inb) print "@@UNCLOSED@@" }'
}
blocks <"$LIB" >"$TMP/canon"
canon_n="$(grep -c '^@@END@@$' "$TMP/canon")"
[[ "$canon_n" == 1 ]] || { bad "$LIB: expected one fv_self_delete definition, found $canon_n"; exit 1; }
sed '/^@@END@@$/d' "$TMP/canon" >"$TMP/canon.fn"

# Compare every block of $2 (a file of text) with the library; $1 names it.
# The number of blocks lands in NCOPIES.
check_copies() {
  local what="$1" text="$2" n=0 f
  NCOPIES=0
  blocks <"$text" >"$TMP/b"
  if grep -q '^@@\(BADINDENT\|UNCLOSED\)@@' "$TMP/b"; then bad "$what: a malformed fv_self_delete block"; return; fi
  awk -v d="$TMP/blk." '/^@@END@@$/ { k++; next } { print > (d k) }' "$TMP/b"
  for f in "$TMP"/blk.*; do
    [[ -e "$f" ]] || continue
    n=$((n + 1))
    if ! diff -u "$TMP/canon.fn" "$f" >"$TMP/diff"; then
      bad "$what: copy $n of fv_self_delete differs from $LIB"; cat "$TMP/diff" >&2
    fi
    rm -f "$f"
  done
  NCOPIES=$n
}

# The definition comes before the first call (a call: the name as a command
# word on a line that is not a comment). Prints the line numbers "def call".
first_def_call() {
  awk '
    !d && index($0, "fv_self_delete() {") { d = NR }
    !c && $0 !~ /^[[:space:]]*#/ && !index($0, "fv_self_delete() {") \
      && match($0, /(^|[^_[:alnum:]])fv_self_delete([[:space:]]|;|$)/) { c = NR }
    END { print d + 0, c + 0 }' "$1"
}
check_order() { # $1 what, $2 text
  local d c
  read -r d c < <(first_def_call "$2")
  if (( c > 0 && (d == 0 || d > c) )); then
    bad "$1: fv_self_delete is called (line $c) before it is defined (line $d)"
    return 1
  fi
  return 0
}

# Statements of a file (backslash continuations joined; in Python, lines joined
# while a bracket is open), "<line>: <text>" each.
statements() {
  case "$1" in
    *.py) awk '{ buf = (buf == "" ? NR ": " : buf " ") $0
                 t = $0; gsub(/#.*/, "", t); o = gsub(/[([{]/, "", t); x = gsub(/[])}]/, "", t)
                 depth += o - x; if (depth <= 0) { print buf; buf = ""; depth = 0 } }
               END { if (buf != "") print buf }' "$1" ;;
    *) awk '{ if (sub(/\\$/, "")) { if (buf == "") start = NR; buf = buf $0; next }
              print (buf == "" ? NR : start) ": " buf $0; buf = "" }' "$1" ;;
  esac
}
# Rule 1 on one file: $1 the file, $2 the path it is judged as.
rest_rule() {
  local f="$1" as="$2" hit
  statements "$f" >"$TMP/stmts"
  # (a) the pod's own id, anywhere.
  while IFS= read -r hit; do
    [[ "$as" == control/src/buildpod-start.ts ]] && continue
    [[ "$hit" == *FV_BACKSTOP_API_KEY* ]] && continue
    bad "$as:${hit%%: *}: REST DELETE of the pod's own id (the pod-scoped key gets HTTP 403; use fv_self_delete from $LIB)"
  done < <(grep -E 'DELETE|\.delete\(' "$TMP/stmts" | grep -E 'RUNPOD_POD_ID|FV_POD_ID' | grep -E 'rest\.runpod\.io|/pods/')
  # (b) any REST pod delete in a file that is not known to run locally.
  is_local "$as" && return 0
  while IFS= read -r hit; do
    bad "$as:${hit%%: *}: REST DELETE of a pod in a file that is not in LOCAL (a pod-side script? use fv_self_delete; a local launcher? add it to LOCAL in $ME)"
  done < <(grep -E '(-X DELETE|"DELETE"|'"'"'DELETE'"'"'|rest DELETE|rp_rest DELETE|\.delete\()' "$TMP/stmts" \
             | grep -E 'rest\.runpod\.io[^ ]*/pods|/pods/' | grep -vE '^[0-9]+: *(#|//)')
}

# The files to scan: tracked, minus the evidence under artifacts/, the docs and this check.
mapfile -t FILES < <(git ls-files -- scripts control/src deploy docker crates .github \
  | grep -vE '^(artifacts/|docs/)' | grep -vxF "$ME" | grep -vE '/target/')
for x in "${LOCAL[@]}"; do [[ -f "$x" ]] || bad "LOCAL names $x, which does not exist"; done

# 1. No REST self-delete; the rule's own cases first.
mkdir -p "$TMP/cases"
printf '%s\n' 'curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \' \
  '  "https://rest.runpod.io/v1/pods/$FV_POD_ID"' >"$TMP/cases/alias.sh"
printf '%s\n' 'import requests' 'requests.delete(' '    f"https://rest.runpod.io/v1/pods/{pod}",' \
  '    headers={"Authorization": f"Bearer {key}"},' ')' >"$TMP/cases/multi.py"
printf '%s\n' 'curl -sS -X DELETE -H "Authorization: Bearer $K" "$API/pods/$ME_ID"' >"$TMP/cases/onpod.sh"
printf '%s\n' 'curl -sS -X DELETE -H "Authorization: Bearer $K" "$API/pods/$POD_ID"' >"$TMP/cases/local.sh"
for c in alias.sh multi.py onpod.sh; do
  saved=$fail; fail=0
  rest_rule "$TMP/cases/$c" "case/$c" 2>/dev/null
  (( fail )) || { saved=1; echo "self-delete: rule 1 misses its own case $c" >&2; }
  fail=$saved
done
LOCAL+=(case/local.sh)
saved=$fail; fail=0; rest_rule "$TMP/cases/local.sh" case/local.sh 2>/dev/null
(( fail )) && { saved=1; echo "self-delete: rule 1 flags a local backstop (case local.sh)" >&2; }
fail=$saved
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || continue
  grep -qE 'DELETE|\.delete\(' "$f" || continue
  rest_rule "$f" "$f"
done

# 2. Every copy is the library's text, defined before its first call; 3. every
#    launcher renders a command with it.
copies=0
LAUNCHERS=()
for f in "${FILES[@]}"; do
  [[ -f "$f" && "$f" != "$LIB" ]] || continue
  grep -q 'fv_self_delete' "$f" || continue
  case "$f" in
    # runpod-http.sh writes the function through an unquoted heredoc (escaped
    # text): its rendered start commands, the runpod-http-*.json fixtures that
    # fv-lab-goldens.sh --check pins to the script, are checked instead.
    scripts/gpu/runpod-http.sh) continue ;;
    crates/fastvideo-lab/fixtures/runpod-http-*.json)
      k=0
      while IFS= read -r -d '' s; do
        k=$((k + 1)); printf '%s' "$s" >"$TMP/text"
        check_copies "$f (string $k)" "$TMP/text"
        copies=$((copies + NCOPIES))
        check_order "$f (string $k)" "$TMP/text"
      done < <(jq -j '.. | strings | select(contains("fv_self_delete")) | ., "\u0000"' "$f")
      continue ;;
    *.json) continue ;;
    *.rs) continue ;; # Rust tests name the function (pod.rs pins pod-start.sh.in's calls)
  esac
  if grep -q 'fv_self_delete() {' "$f"; then
    check_copies "$f" "$f"
    copies=$((copies + NCOPIES))
    check_order "$f" "$f"
  elif grep -qE '^[^#]*fv_self_delete [/$"]' "$f"; then
    LAUNCHERS+=("$f")
  fi
done
(( copies >= 37 )) || bad "only $copies inline copies of fv_self_delete found (expected the pod scripts, pod-start.sh.in and the fixtures)"

# Stand-ins: curl (records each call, answers $FAKE_CODE / $FAKE_BODY), and for
# the rendered commands a shell `sleep` that ends the run once $MAXCALLS calls
# were made (or at `sleep infinity`) and an nvidia-smi at 0 %.
mkdir -p "$TMP/bin"
cat >"$TMP/bin/curl" <<'EOF'
#!/usr/bin/env bash
# Records its arguments (one per line) and stdin (with -H @-); appends a line to
# $FAKE_CALLS; writes $FAKE_BODY to -o; prints $FAKE_CODE (-w).
printf '%s\n' "$@" >"$FAKE_ARGS"
out=
stdin=0
for a in "$@"; do [[ "$a" == @- ]] && stdin=1; done
(( stdin )) && cat >"$FAKE_STDIN" || : >"$FAKE_STDIN"
while (( $# )); do case "$1" in -o) out="$2"; shift ;; esac; shift; done
echo call >>"${FAKE_CALLS:-/dev/null}"
[[ -n "$out" && -n "${FAKE_BODY+x}" ]] && printf '%s' "$FAKE_BODY" >"$out"
printf '%s' "$FAKE_CODE"
exit "${FAKE_RC:-0}"
EOF
chmod +x "$TMP/bin/curl"
PREAMBLE='sleep() {
  case "$1" in infinity) exit 0 ;; esac
  [ "$(wc -l <"$FAKE_CALLS")" -ge "$MAXCALLS" ] && exit 0
  _st_n=$((${_st_n:-0} + 1)); [ "$_st_n" -le 50 ] || exit 0
  return 0
}
nvidia-smi() { echo 0; }
'

# Run the self-delete of a rendered command ($2) with the stand-ins: the text
# up to the end of the definition, then the line of the first call (cut where it
# goes to the background); /tmp/ is redirected. $1 what, $3 HTTP code, $4 the
# calls to stop at; the number of calls made lands in RUNCALLS.
run_rendered() {
  local what="$1" text="$2" code="$3" max="$4" d c dend rd="$TMP/run"
  rm -rf "$rd"; mkdir -p "$rd"; : >"$rd/calls"
  read -r d c < <(first_def_call "$text")
  dend="$(awk -v d="$d" 'NR > d && /^[[:space:]]*}$/ { print NR; exit }' "$text")"
  { printf '%s' "$PREAMBLE"; sed -n "1,${dend}p" "$text"; sed -n "${c}p" "$text" | sed 's/ ) >\/dev\/null 2>&1 & .*$/ )/'; } \
    | sed "s#/tmp/#$rd/#g" >"$rd/cmd.sh"
  env -i PATH="$TMP/bin:$PATH" FAKE_ARGS="$rd/args" FAKE_STDIN="$rd/stdin" FAKE_CALLS="$rd/calls" \
    FAKE_CODE="$code" FAKE_BODY='{"data":{"podTerminate":null}}' MAXCALLS="$max" \
    RUNPOD_API_KEY=rpa_SECRET_POD_KEY RUNPOD_POD_ID=pod42 \
    timeout 20 bash "$rd/cmd.sh" >/dev/null 2>"$rd/err"
  local rc=$?
  (( rc == 0 )) || bad "$what: the self-delete run exited $rc ($(head -c 200 "$rd/err"))"
  if grep -rq rpa_SECRET_POD_KEY "$rd/args" "$rd"/*.log 2>/dev/null; then bad "$what: the key reached argv or a log"; fi
  RUNCALLS="$(wc -l <"$rd/calls" | tr -d ' ')"
}

for f in "${LAUNCHERS[@]+"${LAUNCHERS[@]}"}"; do
  case "$f" in
    *-validate.sh)
      grep -qxF '    start="$(start_cmd)"' "$f" || bad "$f: up does not send start_cmd's output"
      grep -q '^  print-start) start_cmd ;;$' "$f" || bad "$f: no print-start"
      render=(bash "$f" print-start) ;;
    scripts/serve/e2e/ltx-pod.sh)
      grep -qF -- '--arg cmd "$(entry_cmd)"' "$f" || bad "$f: the payload does not send entry_cmd's output"
      render=(env FV_IDLE_GPU_MIN=1 FV_IDLE_GPU_GRACE_MIN=1 RUNPOD_API_KEY=self-delete-test bash "$f" entry) ;;
    *) bad "$f: calls fv_self_delete without defining it, and this check has no renderer for it"; continue ;;
  esac
  if ! env -i PATH="$PATH" HOME="$HOME" "${render[@]}" >"$TMP/r" 2>"$TMP/r.err"; then
    bad "$f: ${render[*]:1} failed: $(head -3 "$TMP/r.err")"; continue
  fi
  bash -n "$TMP/r" || { bad "$f: the rendered command is not valid bash"; continue; }
  check_copies "$f (rendered)" "$TMP/r"
  [[ "$NCOPIES" == 1 ]] || { bad "$f (rendered): expected one fv_self_delete, found $NCOPIES"; continue; }
  check_order "$f (rendered)" "$TMP/r" || continue
  run_rendered "$f" "$TMP/r" 200 1; n=$RUNCALLS
  [[ "$n" == 1 ]] || bad "$f: the rendered self-delete made $n GraphQL calls on HTTP 200, want 1"
  run_rendered "$f" "$TMP/r" 403 3; n=$RUNCALLS
  [[ "$n" == 3 ]] || bad "$f: the rendered self-delete made $n GraphQL calls on HTTP 403 (retries), want 3"
done
(( ${#LAUNCHERS[@]} >= 12 )) || bad "only ${#LAUNCHERS[@]} launchers found (expected the eleven *-validate.sh and ltx-pod.sh)"

# 4. The function against the stand-in curl.
# $1 code, $2 body (unset when "-"), $3 curl's exit; the function's status.
run_fn() {
  rm -f "$TMP/log" "$TMP/log.resp" "$TMP/args" "$TMP/stdin"
  local -a body=()
  [[ "$2" != - ]] && body=(FAKE_BODY="$2")
  env -i PATH="$TMP/bin:$PATH" FAKE_ARGS="$TMP/args" FAKE_STDIN="$TMP/stdin" FAKE_CODE="$1" FAKE_RC="$3" ${body[@]+"${body[@]}"} \
    RUNPOD_API_KEY=rpa_SECRET_POD_KEY RUNPOD_POD_ID=pod42 \
    bash -c '. "$1" && fv_self_delete "$2"' _ "$ROOT/$LIB" "$TMP/log"
}
expect() { # $1 want (0|1), $2 label, rest: run_fn args
  local want="$1" label="$2" got
  shift 2
  run_fn "$@"; got=$?
  (( (got == 0) == (want == 0) )) || bad "fv_self_delete: $label: status $got, want $( ((want == 0)) && echo success || echo failure)"
  if grep -q rpa_SECRET_POD_KEY "$TMP/log" "$TMP/args" 2>/dev/null; then bad "fv_self_delete: $label: the key reached argv or the log"; fi
  [[ ! -e "$TMP/log.resp" ]] || bad "fv_self_delete: $label: the answer file was left behind"
}
expect 0 "200 podTerminate:null" 200 '{"data":{"podTerminate":null}}' 0
# The request: POST, the GraphQL endpoint, JSON, the key on stdin, the mutation.
mapfile -t a <"$TMP/args"
want_args=(-sS --max-time 30 -o "$TMP/log.resp" -w '%{http_code}' -X POST https://api.runpod.io/graphql
  -H 'Content-Type: application/json' -H @-
  --data '{"query":"mutation { podTerminate(input: {podId: \"pod42\"}) }"}')
[[ "${a[*]}" == "${want_args[*]}" ]] || bad "fv_self_delete: curl got: ${a[*]}"
[[ "$(cat "$TMP/stdin")" == "Authorization: Bearer rpa_SECRET_POD_KEY" ]] || bad "fv_self_delete: curl's stdin is not the Authorization header"
grep -q '{"data":{"podTerminate":null}} http=200' "$TMP/log" || bad "fv_self_delete: the answer and status are not in the log"
expect 1 "200 with errors" 200 '{"errors":[{"message":"not authorized"}],"data":null}' 0
expect 1 "403" 403 'Forbidden' 0
expect 1 "500" 500 '' 0
expect 1 "transport error" 000 - 7
expect 1 "201" 201 '{"data":{"podTerminate":null}}' 0

if (( fail )); then
  echo "self-delete check FAILED" >&2
  exit 1
fi
printf 'self-delete check ok (%s inline copies and %s rendered launcher commands match %s, each defined before its call; no REST self-delete)\n' \
  "$copies" "${#LAUNCHERS[@]}" "$LIB"
