#!/usr/bin/env bash
# Drift check for the upstream API specifications vendored under
# docs/serve/specs/ (plan E1). Re-fetches every entry of the SOURCES.toml
# files that has a URL, normalises it the same way as the stored copy and
# compares; exits 1 listing every drifted, unreachable or newly available
# spec. `--update` rewrites the stored files and SOURCES.toml.
#
#   bash scripts/serve/specs-drift.sh            # check (CI: .github/workflows/specs-drift.yml)
#   bash scripts/serve/specs-drift.sh --update   # accept upstream changes
#   bash scripts/serve/specs-drift.sh --only fal # one upstream directory
#
# bash + curl + python3 (standard library). Downloads are untrusted data:
# they are parsed under `python3 -I`, never executed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 -I "$ROOT/scripts/serve/lib/specs-drift.py" --specs-dir "$ROOT/docs/serve/specs" "$@"
