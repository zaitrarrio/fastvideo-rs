#!/usr/bin/env bash
# The goldens of the experiment tiers (docs/adr/0019-fv-lab.md §14, D1e): what a
# port one-off's own pod script would run, cut from the script and run with its
# GPU binary stood in, so fv-lab's cells can be held to it word for word.
#
#   fv-lab-experiment-goldens.sh           regenerate the fixtures below
#   fv-lab-experiment-goldens.sh --check   fail when the script's output differs
#
# h3-graphs (scripts/gpu/experiments/h3-graphs-pod.sh, deleted after the tier's GPU
# check, ADR-0019 §14.4, so read from the history; docs/perf/h3-denoise-graphs.md):
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
# sol-bench (scripts/gpu/sol-bench-pod.sh, deleted after the tiers' GPU check,
# ADR-0019 §23.7 and §23.10, so read from the history with its two Cosmos3
# inputs; docs/perf/sol-bench.md; ADR-0019 §23):
#   crates/fastvideo-lab/fixtures/experiments/sol-bench-script.txt
#       every command each of the script's sets (FV_SOL_SET, every label of its
#       `case "$SET"`, in the script's order) hands its `cell`, in order: a
#       line `== <set> <cell> <verify-weights.sh cell> <estimate s>`, then the
#       command (what the script runs under `timeout`), one argument per line
#       (none is empty or holds a newline; checked). The script's paths are
#       its own for FAMILY=sol-bench-<set> (the tier's name, as phase A ran
#       it) and run tag 1234567-10241200: $RUNS
#       /fvscratch/runs/sol-bench-<set>/1234567-10241200, $SCRATCH /fvscratch,
#       $W /workspace/weights (no FV_R2_TREES), the image's fv-gpucheck and
#       scripts/gpu/sol.
# crates/fastvideo-lab/tests/experiment_tiers.rs finds each sol-bench-<set>
# tier's arms equal to its set's commands, word for word; the one mapping is
# $W to the tier's weight root (/fvscratch/weights for a volume+r2 tier: the
# script's run under runpod-http.sh FV_R2_TREES, ADR-0019 §21.8).
#
# How: the script's `COMMON`, `ARMS`, `CKS`, `arm_env`, `ck_args`, `gen`, the
# prep Python and its `trace` / `cells` blocks are cut out of it (sed, between
# their first and last lines) and run under `env -i` with `step` running its
# command, `pack` / `log` doing nothing, `want` true for trace and cells only
# (no hash step: no frame exists), the h3-turbo trees present, and `env` on
# PATH a stand-in that prints its arguments instead of running them. So the
# command lines are the script's, not a copy of them.
#
# sol-bench's cell definitions (from `# ---- cell definitions` up to the
# set's `log "set …"` line), its PROMPT and SEED defaults and its `case "$SET"`
# block are cut the same way and run once per set under `env -i`, with `cell`
# a stand-in that writes its arguments instead of gating, timing and running
# them (the script's own `cell` adds only the weight gate, the budget and the
# timeout, which the runner replaces, ADR-0019 §23). SOL_IN points at
# scripts/gpu/sol so its `$(cat …)` prompts read the committed files.
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

# deleted_in <path>: the commit that deleted the file (empty while it exists).
deleted_in() {
  git -C "$ROOT" log -1 --no-renames --diff-filter=D --format=%H HEAD -- "$1" 2>/dev/null
}

# source_of <path>: the file, or its last version from the history, or nothing.
source_of() {
  local rel="$1" del
  if [[ -f "$ROOT/$rel" ]]; then
    echo "$ROOT/$rel"
    return 0
  fi
  del="$(deleted_in "$rel")" || return 1
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

# frozen <script> <fixture…>: the script is in neither the tree nor the
# history (a shallow clone): its fixtures must exist, and are not regenerated.
frozen() {
  local pod="$1" f
  shift
  for f in "$@"; do
    [[ -s "$f" ]] || { echo "fv-lab-experiment-goldens: frozen fixture ${f#"$ROOT"/} is missing" >&2; fail=1; }
  done
  echo "fv-lab-experiment-goldens: $pod is neither in the tree nor in the history (shallow clone): its fixtures are kept as frozen, not regenerated" >&2
}
mkdir -p "$TMP/new"
NEW=()   # the fixtures regenerated here, compared or written in step 3

POD=scripts/gpu/experiments/h3-graphs-pod.sh
if ! src="$(source_of "$POD")"; then
  frozen "$POD" "$OUT/h3-graphs-script.txt" "$ROOT/scripts/gpu/prompts-h3-graphs-768.json" \
    "$ROOT/scripts/gpu/prompts-h3-graphs-trace-768p.json"
else

# 1. The prep step's prompt sets: its own Python, on its own input. Once the
#    pod script is read from the history, its input is too (from the same
#    commit, the deleting commit's parent), so a later edit of
#    prompts-hd5.json cannot pair with the old script; CI's shallow clone keeps
#    the fixtures instead, so both agree.
mkdir -p "$TMP/fvw" "$TMP/srv" "$TMP/stub"
HD5="$ROOT/scripts/gpu/prompts-hd5.json"
if [[ ! -f "$ROOT/$POD" ]]; then
  del="$(deleted_in "$POD")"
  HD5="$TMP/history/scripts/gpu/prompts-hd5.json"
  mkdir -p "$(dirname "$HD5")"
  git -C "$ROOT" show "$del^:scripts/gpu/prompts-hd5.json" >"$HD5" \
    || { echo "fv-lab-experiment-goldens: no scripts/gpu/prompts-hd5.json at $del^" >&2; exit 2; }
fi
# The heredoc's body: the lines between its opening line (matched as a plain
# string, through the environment as in cut_block) and `PYEOF`.
PREP_OPEN="  \"\$PY\" - \"\$G/prompts-hd5.json\" \"\$L\" <<'PYEOF'" \
  awk 'BEGIN { o = ENVIRON["PREP_OPEN"] } go == 0 && $0 == o { go = 1; next } go && $0 == "PYEOF" { exit } go { print }' \
  "$src" >"$TMP/prep.py"
[[ -s "$TMP/prep.py" ]] || { echo "fv-lab-experiment-goldens: no prep Python in $POD" >&2; exit 2; }
env -i PATH="/usr/local/bin:/usr/bin:/bin" HOME="$TMP" LC_ALL=C \
  python3 -I "$TMP/prep.py" "$HD5" "$TMP/fvw"
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
NEW+=(h3-graphs-script.txt prompts-h3-graphs-768.json prompts-h3-graphs-trace-768p.json)
fi

# 2b. sol-bench: every set's commands (ADR-0019 §23).
SOL=scripts/gpu/sol-bench-pod.sh
SOL_INPUTS=(scripts/gpu/sol/cosmos3-default.txt scripts/gpu/sol/cosmos3-negative.txt)
if ! sol_src="$(source_of "$SOL")"; then
  frozen "$SOL" "$OUT/sol-bench-script.txt"
else
  # Its inputs from the same commit as the script (the deleting commit's
  # parent once it is deleted), as h3-graphs' prompts-hd5.json.
  SOLDIR="$TMP/sol-in"
  mkdir -p "$SOLDIR"
  for f in "${SOL_INPUTS[@]}"; do
    if [[ -f "$ROOT/$SOL" ]]; then
      cp "$ROOT/$f" "$SOLDIR/"
    else
      git -C "$ROOT" show "$(deleted_in "$SOL")^:$f" >"$SOLDIR/$(basename "$f")" \
        || { echo "fv-lab-experiment-goldens: no $f beside $SOL in the history" >&2; exit 2; }
    fi
  done
  # sol_cut <script> <inputs dir> <work dir> <dest>: every set's commands into
  # <dest> (the format above); on anything it cannot vouch for, a message and 1:
  # (c) the paths the stand-in gives the cut must be the script's own lines
  #     (WORK, W, SCRATCH, RUNS, BIN: each assigned once, as below, and nowhere
  #     else), and the script may export nothing but PATH and LD_LIBRARY_PATH (an
  #     exported FASTVIDEO_* setting would reach every cell outside its command);
  # (b) every arm of its `case "$SET"` is a set the parser accepts (`  <label>)`,
  #     [a-z0-9][a-z0-9_-]*) or the `*)` refusal: the `;;` count must agree;
  # (a) every command a set runs goes through `cell`: $BIN is a stand-in that
  #     records a direct call, an unknown command is recorded by bash's
  #     command_not_found_handle, and the cut runs under errexit; any record fails.
  # The self-test below runs it on mutated copies, each of which must fail.
  sol_cut() {
    local src="$1" soldir="$2" work="$3" dest="$4" set arms
    local -a sets=()
    rm -rf "$work" && mkdir -p "$work/stub" || return 1
    python3 -I - "$src" <<'PY' || return 1
import re, sys
lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
want = {
    "WORK": 'WORK="${FV_WORK:-/workspace}"',
    "W": 'W="$WORK/weights"',
    "SCRATCH": 'SCRATCH="${FV_SCRATCH:-/fvscratch}"',
    "RUNS": 'RUNS="$SCRATCH/runs/$FAMILY/${FV_RUN_TAG:?FV_RUN_TAG}"',
    "BIN": "BIN=/opt/fastvideo-rs/target/release/fv-gpucheck",
}
bad = []
for name, line in want.items():
    hits = [l for l in lines if re.match(r"\s*((local|declare|export|readonly)(\s+-\w+)*\s+)?" + name + "=", l)]
    if hits != [line]:
        bad.append(f"{name} assigned as {hits!r}, want exactly [{line!r}]")
for l in lines:
    m = re.match(r"\s*(export|declare\s+-\w*x\w*)\s+([A-Za-z_][A-Za-z0-9_]*)", l)
    if m and m.group(2) not in ("PATH", "LD_LIBRARY_PATH"):
        bad.append(f"exports {m.group(2)} ({l.strip()})")
if bad:
    sys.exit("fv-lab-experiment-goldens: sol-bench-pod.sh: " + "; ".join(bad))
PY
    {
      echo 'set -eu'
      cut_block "$src" 'PROMPT="${FV_PROMPT:-' line || return 1
      cut_block "$src" 'SEED="${FV_SEED:-' line || return 1
      # The definitions: from their banner up to the set's first log line.
      SOL_START='# ---- cell definitions' SOL_END='log "set $SET budget' awk '
        BEGIN { a = ENVIRON["SOL_START"]; b = ENVIRON["SOL_END"] }
        go == 0 && substr($0, 1, length(a)) == a { go = 1 }
        go == 1 && substr($0, 1, length(b)) == b { exit }
        go == 1 { print }' "$src"
      cut_block "$src" 'case "$SET" in' exact 'esac' || return 1
    } >"$work/sol.sh" || return 1
    # SOL_IN to the inputs (its `$(cat …)` prompts), named back below.
    python3 -I -c '
import sys
path, soldir = sys.argv[1:]
text = open(path, encoding="utf-8").read()
line = "SOL_IN=/opt/fastvideo-rs/scripts/gpu/sol\n"
if text.count(line) != 1:
    sys.exit("fv-lab-experiment-goldens: SOL_IN=/opt/fastvideo-rs/scripts/gpu/sol is not one line of the cut")
open(path, "w", encoding="utf-8").write(text.replace(line, "SOL_IN=" + soldir + "\n"))' "$work/sol.sh" "$soldir" || return 1
    grep -q '^cosmos3() {' "$work/sol.sh" && grep -q '^esac$' "$work/sol.sh" \
      || { echo "fv-lab-experiment-goldens: the sol-bench cut lacks its definitions or its case block" >&2; return 1; }
    # (b) The sets, in the script's order, and every arm accounted for.
    cut_block "$src" 'case "$SET" in' exact 'esac' >"$work/case.sh" || return 1
    mapfile -t sets < <(awk '/^  [a-z0-9][a-z0-9_-]*\)/ { sub(/^  /, ""); sub(/\).*/, ""); print }' "$work/case.sh")
    arms="$(sed -e 's/#.*//' -e 's/[[:space:]]*$//' "$work/case.sh" | grep -c ';;$' || true)"
    grep -q '^  \*) ' "$work/case.sh" \
      || { echo "fv-lab-experiment-goldens: sol-bench-pod.sh's case has no \`*)\` refusal" >&2; return 1; }
    (( ${#sets[@]} > 0 && ${#sets[@]} + 1 == arms )) \
      || { echo "fv-lab-experiment-goldens: sol-bench-pod.sh's case has $arms arms but ${#sets[@]} set label(s) parse (${sets[*]}): a label the parser does not accept" >&2; return 1; }
    # (a) $BIN called outside `cell`: recorded, and fails.
    printf '#!/bin/bash\nprintf "%%s\\n" "BIN $*" >>"%s/unrecorded.txt"\nexit 3\n' "$work" >"$work/stub/fv-gpucheck"
    chmod +x "$work/stub/fv-gpucheck"
    : >"$work/calls.txt"
    for set in "${sets[@]}"; do
      env -i PATH="/usr/local/bin:/usr/bin:/bin" HOME="$work" LC_ALL=C SET="$set" NL=$'\n' WORKDIR="$work" \
        RUNS="/fvscratch/runs/sol-bench-$set/1234567-10241200" SCRATCH=/fvscratch W=/workspace/weights \
        BIN="$work/stub/fv-gpucheck" bash -c '
          command_not_found_handle() { printf "%s\n" "CMD $*" >>"$WORKDIR/unrecorded.txt"; return 127; }
          log() { :; }
          cell() {
            local name="$1" wcell="$2" est="$3" a
            shift 3
            printf "== %s %s %s %s\n" "$SET" "$name" "$wcell" "$est"
            for a in "$@"; do
              [[ -n "$a" && "$a" != *"$NL"* ]] || { echo "an empty or multi-line argument in $name" >&2; exit 3; }
              printf "%s\n" "$a"
            done
          }
          . "$1"' _ "$work/sol.sh" >>"$work/calls.txt" 2>"$work/stderr-$set.txt" \
        || { echo "fv-lab-experiment-goldens: sol-bench set $set failed in the cut: $(cat "$work/unrecorded.txt" 2>/dev/null | head -3) $(head -3 "$work/stderr-$set.txt")" >&2; return 1; }
    done
    if [[ -s "$work/unrecorded.txt" ]]; then
      echo "fv-lab-experiment-goldens: sol-bench-pod.sh runs a command outside \`cell\`: $(head -3 "$work/unrecorded.txt")" >&2
      return 1
    fi
    grep -q '^== ' "$work/calls.txt" || { echo "fv-lab-experiment-goldens: the sol-bench sets started no cell" >&2; return 1; }
    python3 -I -c '
import sys
soldir, stub, src, dst = sys.argv[1:]
text = open(src, encoding="utf-8").read()
text = text.replace(soldir, "/opt/fastvideo-rs/scripts/gpu/sol").replace(stub, "/opt/fastvideo-rs/target/release/fv-gpucheck")
open(dst, "w", encoding="utf-8").write(text)' "$soldir" "$work/stub/fv-gpucheck" "$work/calls.txt" "$dest" || return 1
  }
  sol_cut "$sol_src" "$SOLDIR" "$TMP/sol-cut" "$TMP/new/sol-bench-script.txt" || exit 2
  NEW+=(sol-bench-script.txt)

  # sol_cut's own holes, each closed: a mutated copy of the script must fail
  # (ADR-0019 §23.5; the review of #235): (a) a command run outside `cell`,
  # through $BIN or as an unknown command; (b) a set label the parser does not
  # accept; (c) $W assigned otherwise, and an exported FASTVIDEO_* setting.
  sol_mutant() { # <name> <old> <new>: the mutant must fail
    local name="$1"
    SOL_OLD="$2" SOL_NEW="$3" python3 -I -c '
import os, sys
src, dst = sys.argv[1:]
text = open(src, encoding="utf-8").read()
old, new = os.environ["SOL_OLD"], os.environ["SOL_NEW"]
if text.count(old) < 1:
    sys.exit("fv-lab-experiment-goldens: mutant pattern not in sol-bench-pod.sh: " + old)
open(dst, "w", encoding="utf-8").write(text.replace(old, new, 1))' "$sol_src" "$TMP/mutant.sh" || exit 2
    if sol_cut "$TMP/mutant.sh" "$SOLDIR" "$TMP/mutant-cut" "$TMP/mutant-out.txt" 2>/dev/null; then
      echo "fv-lab-experiment-goldens: sol_cut accepted the mutant \`$name\`" >&2
      exit 2
    fi
  }
  sol_mutant direct-bin '  a1) # ' '  a1) "$BIN" --mode fast kernels --groups pisa
    # '
  sol_mutant direct-bin-guarded '  a1) # ' '  a1) "$BIN" --mode fast kernels --groups pisa || true
    # '
  sol_mutant direct-command '  a1) # ' '  a1) fv-gpucheck-direct --mode fast kernels
    # '
  sol_mutant label '  g2) # ' '  g2|h2) # '
  sol_mutant w-path 'W="$WORK/weights"' 'W="$WORK/weights2"'
  sol_mutant export 'BIN=/opt/fastvideo-rs/target/release/fv-gpucheck' 'BIN=/opt/fastvideo-rs/target/release/fv-gpucheck
export FASTVIDEO_WAN_QUANT=off'

  # The script's own `cell` under FV_CELLS (the GPU check runs one cell on the
  # script side, ADR-0019 §23.7): an unselected cell is logged and leaves no
  # directory, as fv-lab's --cells; a selected one runs; none set runs all.
  # Its weight gate, GPU wait and clock are stood in.
  cut_block "$sol_src" 'cell() {' exact '}' >"$TMP/cell.sh"
  printf 'exit 0\n' >"$TMP/verify-ok.sh"
  cells_check() { # <FV_CELLS or -> <want dirs…>
    local sel="$1" d got
    local -a fvc=()
    shift
    [[ "$sel" == - ]] || fvc=("FV_CELLS=$sel")
    rm -rf "$TMP/cells-runs" && mkdir -p "$TMP/cells-runs"
    env -i PATH="/usr/local/bin:/usr/bin:/bin" HOME="$TMP" LC_ALL=C RUNS="$TMP/cells-runs" BUDGET_S=100000 \
      VERIFY="$TMP/verify-ok.sh" W=/nonexistent ${fvc[@]+"${fvc[@]}"} bash -c '
        log() { printf "%s\n" "$*" >>"$RUNS/live.log"; }
        uptime_s() { echo 0; }
        gpu_clean() { return 0; }
        . "$1"
        cell one h3-base 1 true
        cell two h3-base 1 true
        cell three h3-base 1 true' _ "$TMP/cell.sh" >/dev/null 2>&1 \
      || { echo "fv-lab-experiment-goldens: sol-bench-pod.sh cell failed under FV_CELLS=$sel" >&2; exit 2; }
    got="$(cd "$TMP/cells-runs" && for d in */summary.json; do [[ -f "$d" ]] && dirname "$d"; done | LC_ALL=C sort | tr '\n' ' ')"
    [[ "$got" == "$* " ]] || { echo "fv-lab-experiment-goldens: sol-bench-pod.sh cell under FV_CELLS=$sel ran [$got], want [$* ]" >&2; exit 2; }
    for d in one two three; do
      [[ " $* " == *" $d "* ]] && continue
      [[ ! -e "$TMP/cells-runs/$d" ]] || { echo "fv-lab-experiment-goldens: unselected cell $d left a directory" >&2; exit 2; }
      grep -qxF "skip $d (not in FV_CELLS)" "$TMP/cells-runs/live.log" \
        || { echo "fv-lab-experiment-goldens: unselected cell $d not logged as skipped" >&2; exit 2; }
    done
  }
  cells_check - one three two
  cells_check two two
  cells_check "one,three" one three
  cells_check "three one" one three
fi

# 3. Write or compare.
mkdir -p "$OUT"
for f in ${NEW[@]+"${NEW[@]}"}; do
  case "$f" in
    prompts-*) dest="$ROOT/scripts/gpu/$f" ;;
    *) dest="$OUT/$f" ;;
  esac
  if [[ "$CHECK" == 1 ]]; then
    if ! diff -u "$dest" "$TMP/new/$f"; then
      echo "fv-lab-experiment-goldens: ${dest#"$ROOT"/} differs from what its pod script does (rerun without --check, then make the cells match)" >&2
      fail=1
    fi
  else
    cp "$TMP/new/$f" "$dest"
    echo "wrote ${dest#"$ROOT"/}"
  fi
done
exit "$fail"
