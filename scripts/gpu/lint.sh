#!/usr/bin/env bash
# Shell checks for the harness itself, for the traps that only bite on the
# machine that runs it. macOS ships bash 3.2, where `"${arr[@]}"` on an EMPTY
# array under `set -u` is an "unbound variable" error (bash 4.4+ made it fine),
# so a script that works on the rented Linux box can still die on the laptop.
#
# Scope: the scripts that run locally. Anything inside a remote heredoc executes
# on Ubuntu's bash 5 and is exempt.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_SCRIPTS=(scripts/gpu/validate.sh scripts/gpu/lib.sh scripts/gpu/docker.sh scripts/gpu/timings.sh scripts/gpu/lint.sh)
fail=0

note() { printf '%s\n' "$*" >&2; }

cd "$ROOT"

# 1. Parse every shell script, including the ones that only run remotely.
while IFS= read -r f; do
  bash -n "$f" || { note "PARSE $f"; fail=1; }
done < <(find scripts -name '*.sh' | sort)

# 2. Empty-array expansion, unguarded. An array assigned `()` anywhere in a
#    locally-run script must be expanded as ${a[@]+"${a[@]}"}.
for f in "${LOCAL_SCRIPTS[@]}"; do
  [[ -f "$f" ]] || continue
  while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    # Expansions of this array that are not the guarded form.
    while IFS= read -r hit; do
      [[ -n "$hit" ]] || continue
      note "EMPTY-ARRAY $f:$hit"
      note "  '$name' can be empty; bash 3.2 + set -u needs \${$name[@]+\"\${$name[@]}\"}"
      fail=1
    done < <(grep -n "\"\${$name\[@\]}\"" "$f" | grep -v "\${$name\[@\]+" || true)
  done < <(grep -hoE '^[[:space:]]*(local[[:space:]]+)?[a-zA-Z_][a-zA-Z0-9_]*=\(\)[[:space:]]*$' "$f" \
             | sed -E 's/^[[:space:]]*(local[[:space:]]+)?([a-zA-Z_][a-zA-Z0-9_]*)=\(\)[[:space:]]*$/\2/' | sort -u)
done

if (( fail )); then
  note "shell lint FAILED"
  exit 1
fi
printf 'shell lint ok (%s scripts parsed)\n' "$(find scripts -name '*.sh' | wc -l | tr -d ' ')"
