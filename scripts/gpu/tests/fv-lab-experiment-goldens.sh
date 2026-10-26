#!/usr/bin/env bash
# The goldens of the experiment tiers (docs/adr/0019-fv-lab.md §14, D1e): what a
# port one-off's own pod script would run, cut from the script and run with its
# GPU binary stood in, so fv-lab's cells can be held to it word for word.
#
#   fv-lab-experiment-goldens.sh           regenerate the fixtures below
#   fv-lab-experiment-goldens.sh --check   fail when the script's output differs
#
# h3-graphs (scripts/gpu/experiments/h3-graphs-pod.sh; docs/perf/h3-denoise-graphs.md):
#   crates/fastvideo-lab/fixtures/experiments/h3-graphs-script.txt
#       every fv-gpucheck command line the pod script's `trace` and `cells`
#       steps start, in its order, for every checkpoint it knows (v2, 4s,
#       turbo) and every arm (sync, lazy, graphs): `<dir> <argv…>`, one line
#       per process, the arguments separated by one space (none holds a
#       space; checked). <dir> is the process's directory under /srv. The
#       pod's paths are the script's own: /srv (results), /fvw (its scratch
#       disk), /workspace/weights (the EU volume), the image's fv-gpucheck.
#   scripts/gpu/prompts-h3-graphs-768.json, scripts/gpu/prompts-h3-graphs-trace-768p.json
#       the prompt sets the script's `prep` step writes as /fvw/prompts-768.json
#       and /fvw/prompts-trace-768p.json (its own Python, on
#       scripts/gpu/prompts-hd5.json), byte for byte: the h3-graphs cells read
#       these files.
# crates/fastvideo-lab/tests/experiment_tiers.rs maps the h3-graphs tier's dry
# run onto the script's turbo lines (the mapping is ADR-0019 §14.3's table)
# and finds them equal.
#
# How: the script's `COMMON`, `ARMS`, `CKS`, `arm_env`, `ck_args`, `gen`, the
# prep Python and its `trace` / `cells` blocks are cut out of it (sed, between
# their first and last lines) and run under `env -i` with `step` running its
# command, `pack` / `log` doing nothing, `want` true for trace and cells only
# (no hash step: no frame exists), the h3-turbo trees present, and `env` on
# PATH a stand-in that prints its arguments instead of running them. So the
# command lines are the script's, not a copy of them.
#
# After the pod script is deleted (once its tier's GPU check has passed), it
# is read from the parent of the commit that deleted it when the history holds
# that commit; otherwise (CI's shallow clone) the fixtures are kept as they are
# and the script says so. No network, no GPU; needs bash, awk (gawk, mawk or
# busybox: plain-string matches only) and python3.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="$ROOT/crates/fastvideo-lab/fixtures/experiments"
CHECK=0
[[ "${1:-}" == --check ]] && CHECK=1
export LC_ALL=C

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0

# source_of <path>: the file, or its last version from the history, or nothing.
source_of() {
  local rel="$1" del
  if [[ -f "$ROOT/$rel" ]]; then
    echo "$ROOT/$rel"
    return 0
  fi
  del="$(git -C "$ROOT" log -1 --no-renames --diff-filter=D --format=%H HEAD -- "$rel" 2>/dev/null)" || return 1
  [[ -n "$del" ]] && git -C "$ROOT" cat-file -e "$del^:$rel" 2>/dev/null || return 1
  mkdir -p "$TMP/history/$(dirname "$rel")"
  git -C "$ROOT" show "$del^:$rel" >"$TMP/history/$rel"
  echo "$TMP/history/$rel"
}

# cut_block <file> <start> <how> [<end>]: the lines from the first line that
# starts with <start> through the line that ends the block, inclusive: <how>
# `line` (the start line alone), `exact` (the next line equal to <end>) or
# `suffix` (the next line ending in <end>). Plain strings, no regular
# expressions, handed to awk through the environment (`-v` would read
# backslash escapes), so gawk, mawk and busybox awk cut the same lines. Empty
# is an error (the script changed shape).
cut_block() {
  local out
  out="$(CUT_START="$2" CUT_HOW="$3" CUT_END="${4:-}" awk '
    BEGIN { a = ENVIRON["CUT_START"]; how = ENVIRON["CUT_HOW"]; b = ENVIRON["CUT_END"] }
    go == 0 && substr($0, 1, length(a)) == a {
      go = 1; print
      if (how == "line" || (how == "suffix" && length($0) >= length(b) && substr($0, length($0) - length(b) + 1) == b)) exit
      next
    }
    go == 1 {
      print
      if (how == "exact" && $0 == b) exit
      if (how == "suffix" && length($0) >= length(b) && substr($0, length($0) - length(b) + 1) == b) exit
    }' "$1")"
  [[ -n "$out" ]] || { echo "fv-lab-experiment-goldens: no block from [$2] ($3 [${4:-}]) in $1" >&2; return 1; }
  printf '%s\n' "$out"
}

POD=scripts/gpu/experiments/h3-graphs-pod.sh
if ! src="$(source_of "$POD")"; then
  for f in "$OUT/h3-graphs-script.txt" "$ROOT/scripts/gpu/prompts-h3-graphs-768.json" \
    "$ROOT/scripts/gpu/prompts-h3-graphs-trace-768p.json"; do
    [[ -s "$f" ]] || { echo "fv-lab-experiment-goldens: frozen fixture ${f#"$ROOT"/} is missing" >&2; fail=1; }
  done
  echo "fv-lab-experiment-goldens: $POD is neither in the tree nor in the history (shallow clone): its fixtures are kept as frozen, not regenerated" >&2
  exit "$fail"
fi

# 1. The prep step's prompt sets: its own Python, on its own input.
mkdir -p "$TMP/fvw" "$TMP/srv" "$TMP/stub" "$TMP/new"
# The heredoc's body: the lines between its opening line (matched as a plain
# string, through the environment as in cut_block) and `PYEOF`.
PREP_OPEN="  \"\$PY\" - \"\$G/prompts-hd5.json\" \"\$L\" <<'PYEOF'" \
  awk 'BEGIN { o = ENVIRON["PREP_OPEN"] } go == 0 && $0 == o { go = 1; next } go && $0 == "PYEOF" { exit } go { print }' \
  "$src" >"$TMP/prep.py"
[[ -s "$TMP/prep.py" ]] || { echo "fv-lab-experiment-goldens: no prep Python in $POD" >&2; exit 2; }
env -i PATH="/usr/local/bin:/usr/bin:/bin" HOME="$TMP" LC_ALL=C \
  python3 -I "$TMP/prep.py" "$ROOT/scripts/gpu/prompts-hd5.json" "$TMP/fvw"
cp "$TMP/fvw/prompts-768.json" "$TMP/new/prompts-h3-graphs-768.json"
cp "$TMP/fvw/prompts-trace-768p.json" "$TMP/new/prompts-h3-graphs-trace-768p.json"

# 2. The trace and cells steps' commands.
{
  echo 'set -u'
  cut_block "$src" 'ARMS="' line
  cut_block "$src" 'CKS="' line
  cut_block "$src" 'COMMON=(' suffix ')'
  cut_block "$src" 'arm_env() {' exact '}'
  cut_block "$src" 'ck_args() {' exact '}'
  cut_block "$src" 'gen() {' exact '}'
  cut_block "$src" 'if want trace; then' exact 'fi'
  cut_block "$src" 'if want cells; then' exact 'fi'
} >"$TMP/steps.sh"
# The stand-in for env(1): the process's directory and its arguments, which
# must hold no whitespace (the fixture is space-separated).
cat >"$TMP/stub/env" <<'EOF'
#!/bin/bash
for a in "$@"; do
  [[ -n "$a" && "$a" != *[[:space:]]* ]] || { echo "argument with whitespace or empty: [$a]" >&2; exit 3; }
done
printf '%s %s\n' "$PWD" "env $*" >>"$FV_STUB_LOG"
EOF
chmod +x "$TMP/stub/env"
env -i PATH="$TMP/stub:/usr/local/bin:/usr/bin:/bin" HOME="$TMP" LC_ALL=C FV_STUB_LOG="$TMP/calls.txt" \
  SRV="$TMP/srv" L="$TMP/fvw" W=/workspace/weights BIN=/opt/fastvideo-rs/target/release/fv-gpucheck \
  TURBO_OK=1 bash -c '
    step() { shift; "$@"; }
    pack() { :; }
    log() { :; }
    want() { [[ "$1" == trace || "$1" == cells ]]; }
    . "$1"' _ "$TMP/steps.sh"
grep -q . "$TMP/calls.txt" || { echo "fv-lab-experiment-goldens: the cut steps started no process" >&2; exit 2; }
# Every gen call failed in the stand-in otherwise (exit 3 lands in stderr.log).
if grep -rl 'argument with whitespace' "$TMP/srv" >/dev/null 2>&1; then
  echo "fv-lab-experiment-goldens: an argument holds whitespace: $(grep -rh 'argument with whitespace' "$TMP/srv" | head -1)" >&2
  exit 2
fi
# Plain-string replacement (the temp dir may hold `.`, which sed reads as any character).
python3 -I -c '
import sys
tmp, src, dst = sys.argv[1:]
text = open(src).read().replace(tmp + "/srv", "/srv").replace(tmp + "/fvw", "/fvw")
open(dst, "w").write(text)' "$TMP" "$TMP/calls.txt" "$TMP/new/h3-graphs-script.txt"

# 3. Write or compare.
mkdir -p "$OUT"
for f in h3-graphs-script.txt prompts-h3-graphs-768.json prompts-h3-graphs-trace-768p.json; do
  case "$f" in
    prompts-*) dest="$ROOT/scripts/gpu/$f" ;;
    *) dest="$OUT/$f" ;;
  esac
  if [[ "$CHECK" == 1 ]]; then
    if ! diff -u "$dest" "$TMP/new/$f"; then
      echo "fv-lab-experiment-goldens: ${dest#"$ROOT"/} differs from what $POD does (rerun without --check, then make the cells match)" >&2
      fail=1
    fi
  else
    cp "$TMP/new/$f" "$dest"
    echo "wrote ${dest#"$ROOT"/}"
  fi
done
exit "$fail"
