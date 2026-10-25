#!/usr/bin/env bash
# The in-pod self-delete (docs/adr/0023-lab-report-and-ci-bridge.md §4): inside a
# Runpod pod, the pod-scoped RUNPOD_API_KEY may not DELETE /v1/pods/$RUNPOD_POD_ID
# through REST (HTTP 403); GraphQL podTerminate with the same key works. Every
# in-pod self-delete calls fv_self_delete, defined once in
# scripts/gpu/lib/self-delete.sh. This check fails when:
#
#   1. any script, start command or fixture still deletes $RUNPOD_POD_ID through
#      REST (a DELETE of rest.runpod.io/.../pods/ or $API/pods/ naming
#      RUNPOD_POD_ID), except with fv-control's account key (FV_BACKSTOP_API_KEY,
#      control/src/cluster/payloads.ts) and the build pod's start command
#      (control/src/buildpod-start.ts: REST, then GraphQL podTerminate);
#   2. a copy of fv_self_delete (inline in a pod script, pod-start.sh.in, the
#      runpod-http-*.json fixtures that pin runpod-http.sh's start command, or
#      ltx-pod.sh's rendered entry command) differs from the library's text;
#   3. a script calls fv_self_delete without defining it and without embedding
#      the library (or names a library path that does not exist);
#   4. the function itself misbehaves against a stand-in curl: success only on
#      HTTP 200 without "errors", the GraphQL endpoint and mutation, the key
#      never written to the log.
#
#   bash scripts/gpu/tests/self-delete.test.sh
#
# No network. Needs bash, awk, jq, git.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LIB=scripts/gpu/lib/self-delete.sh
ME=scripts/gpu/tests/self-delete.test.sh
cd "$ROOT" || exit 2
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
bad() { printf 'self-delete: %s\n' "$*" >&2; fail=1; }

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
  local what="$1" text="$2" n=0
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

# The files to scan: tracked, minus the evidence under artifacts/, the docs and this check.
mapfile -t FILES < <(git ls-files -- scripts control/src deploy docker crates .github \
  | grep -vE '^(artifacts/|docs/)' | grep -vxF "$ME" | grep -vE '/target/')

# 1. No REST self-delete of the pod's own id.
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || continue
  grep -q 'RUNPOD_POD_ID' "$f" || continue
  grep -q 'DELETE' "$f" || continue
  # Join backslash-continued lines, so a command split over lines is one.
  awk '{ if (sub(/\\$/, "")) { buf = buf $0; next } print NR ": " buf $0; buf = "" }' "$f" \
    | grep 'DELETE' | grep 'RUNPOD_POD_ID' | grep -E 'rest\.runpod\.io|/pods/' >"$TMP/hits" || continue
  while IFS= read -r hit; do
    case "$f" in
      control/src/buildpod-start.ts) continue ;; # REST stop, REST delete, then GraphQL podTerminate
    esac
    [[ "$hit" == *FV_BACKSTOP_API_KEY* ]] && continue # fv-control's account key, not the pod's
    bad "$f:${hit%%: *}: REST DELETE of RUNPOD_POD_ID (the pod-scoped key gets HTTP 403; use fv_self_delete from $LIB)"
  done <"$TMP/hits"
done

# 2. Every copy is the library's text; 3. every caller has the function.
copies=0
for f in "${FILES[@]}"; do
  [[ -f "$f" && "$f" != "$LIB" ]] || continue
  grep -q 'fv_self_delete' "$f" || continue
  case "$f" in
    # runpod-http.sh writes the function through an unquoted heredoc (escaped
    # text): its rendered start commands, the runpod-http-*.json fixtures that
    # fv-lab-goldens.sh --check pins to the script, are compared instead.
    scripts/gpu/runpod-http.sh) continue ;;
    crates/fastvideo-lab/fixtures/runpod-http-*.json)
      jq -r '.. | strings | select(contains("fv_self_delete() {"))' "$f" >"$TMP/text" ;;
    *.json) continue ;;
    *.rs) continue ;; # Rust tests name the function (pod.rs pins pod-start.sh.in's calls)
    *) cp "$f" "$TMP/text" ;;
  esac
  if grep -q 'fv_self_delete() {' "$TMP/text"; then
    check_copies "$f" "$TMP/text"
    copies=$((copies + NCOPIES))
  elif grep -qE '^[^#]*fv_self_delete [/$"]' "$f"; then
    # A launcher that embeds the library into the command it sends.
    mapfile -t refs < <(grep -oE '\$\(<"\$(HERE|ROOT)/[^"]*lib/self-delete\.sh"\)' "$f")
    if (( ${#refs[@]} == 0 )); then
      bad "$f: calls fv_self_delete but neither defines it nor embeds $LIB"
    fi
    for r in "${refs[@]+"${refs[@]}"}"; do
      rel="${r#\$(<\"}"; rel="${rel%\")}"
      case "$rel" in
        '$HERE/'*) p="$(dirname "$f")/${rel#\$HERE/}" ;;
        '$ROOT/'*) p="${rel#\$ROOT/}" ;;
      esac
      [[ "$(cd "$(dirname "$p")" 2>/dev/null && pwd)/$(basename "$p")" == "$ROOT/$LIB" ]] \
        || bad "$f: embeds $rel, which is not $LIB"
    done
  fi
done
(( copies >= 20 )) || bad "only $copies inline copies of fv_self_delete found (expected the pod scripts, pod-start.sh.in and the fixtures)"

# ltx-pod.sh sends a rendered command: render it and compare.
if ! env -i PATH="$PATH" HOME="$HOME" RUNPOD_API_KEY=self-delete-test bash scripts/serve/e2e/ltx-pod.sh entry >"$TMP/ltx" 2>"$TMP/ltx.err"; then
  bad "ltx-pod.sh entry failed: $(head -3 "$TMP/ltx.err")"
else
  bash -n "$TMP/ltx" || bad "ltx-pod.sh entry: not valid bash"
  check_copies "ltx-pod.sh entry" "$TMP/ltx"
  [[ "$NCOPIES" == 1 ]] || bad "ltx-pod.sh entry: expected one fv_self_delete, found $NCOPIES"
  grep -q 'fv_self_delete /tmp/idle-watchdog.log' "$TMP/ltx" || bad "ltx-pod.sh entry: the idle watchdog does not call fv_self_delete"
fi

# 4. The function against a stand-in curl.
mkdir -p "$TMP/bin"
cat >"$TMP/bin/curl" <<'EOF'
#!/usr/bin/env bash
# Records its arguments (one per line), writes $FAKE_BODY to -o, prints $FAKE_CODE (-w).
printf '%s\n' "$@" >"$FAKE_ARGS"
out=
while (( $# )); do case "$1" in -o) out="$2"; shift ;; esac; shift; done
[[ -n "$out" && -n "${FAKE_BODY+x}" ]] && printf '%s' "$FAKE_BODY" >"$out"
printf '%s' "$FAKE_CODE"
exit "${FAKE_RC:-0}"
EOF
chmod +x "$TMP/bin/curl"
# $1 code, $2 body (unset when "-"), $3 curl's exit; the function's status.
run_fn() {
  rm -f "$TMP/log" "$TMP/log.resp" "$TMP/args"
  local -a body=()
  [[ "$2" != - ]] && body=(FAKE_BODY="$2")
  env -i PATH="$TMP/bin:$PATH" FAKE_ARGS="$TMP/args" FAKE_CODE="$1" FAKE_RC="$3" ${body[@]+"${body[@]}"} \
    RUNPOD_API_KEY=rpa_SECRET_POD_KEY RUNPOD_POD_ID=pod42 \
    bash -c '. "$1" && fv_self_delete "$2"' _ "$ROOT/$LIB" "$TMP/log"
}
expect() { # $1 want (0|1), $2 label, rest: run_fn args
  local want="$1" label="$2" got
  shift 2
  run_fn "$@"; got=$?
  (( (got == 0) == (want == 0) )) || bad "fv_self_delete: $label: status $got, want $( ((want == 0)) && echo success || echo failure)"
  if grep -q rpa_SECRET_POD_KEY "$TMP/log" "$TMP/log.resp" 2>/dev/null; then bad "fv_self_delete: $label: the key reached the log"; fi
}
expect 0 "200 podTerminate:null" 200 '{"data":{"podTerminate":null}}' 0
# The request: POST, the GraphQL endpoint, JSON, the Bearer key, the mutation.
mapfile -t a <"$TMP/args"
want_args=(-sS --max-time 30 -o "$TMP/log.resp" -w '%{http_code}' -X POST https://api.runpod.io/graphql
  -H 'Content-Type: application/json' -H 'Authorization: Bearer rpa_SECRET_POD_KEY'
  --data '{"query":"mutation { podTerminate(input: {podId: \"pod42\"}) }"}')
[[ "${a[*]}" == "${want_args[*]}" ]] || bad "fv_self_delete: curl got: ${a[*]}"
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
printf 'self-delete check ok (%s inline copies + ltx-pod.sh entry match %s; no REST self-delete with the pod key)\n' "$copies" "$LIB"
