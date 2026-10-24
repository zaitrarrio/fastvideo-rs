#!/usr/bin/env bash
# Test coverage of the fv-serve adapter crates (plan E4): `cargo test --locked`
# of the adapters and the protocol / serve-kit / engine-service crates they
# sit on, instrumented by cargo-llvm-cov, with an lcov file, an HTML report
# and a line / region / function summary. CPU only, no CUDA, no GPU.
#
#   bash scripts/serve/coverage.sh          # llvm-cov: lcov.info, html/, summary.{json,md}
#                                           # under $FV_COV_OUT (default target/llvm-cov)
#   bash scripts/serve/coverage.sh --tests  # the same `cargo test --locked` set with no
#                                           # instrumentation (what the build pod can run
#                                           # before its image has cargo-llvm-cov)
#
# The crate set is CRATES below: every adapter (openai-videos, minimax, ltxapi,
# fal, reactor), the serve crate that mounts them, and the CPU crates they
# build on. Media and WebRTC tests that need ffmpeg skip (pass with a note)
# where it is missing; the build base image has it.
#
# Needs: `cargo llvm-cov` (https://github.com/taiki-e/cargo-llvm-cov) and the
# toolchain's `llvm-tools-preview` component. The adapter-tests workflow
# installs both at run time; the build base image does not have them yet.
# A GITHUB_STEP_SUMMARY, when set, gets the markdown summary appended.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE=cov
for a in "$@"; do
  case "$a" in
    --tests) MODE=tests ;;
    *) echo "coverage.sh: unknown argument $a (--tests)" >&2; exit 2 ;;
  esac
done

CRATES=(
  fastvideo-protocol
  fastvideo-trace
  fastvideo-engine-service
  fastvideo-serve-kit
  fastvideo-upsampler
  fastvideo-openai-videos
  fastvideo-minimax
  fastvideo-ltxapi
  fastvideo-fal
  fastvideo-reactor
  fastvideo-serve
)
PKGS=()
for c in "${CRATES[@]}"; do PKGS+=(-p "$c"); done

run() { printf '\n==> %s\n' "$*" >&2; "$@"; }

if [[ "$MODE" == tests ]]; then
  run cargo test --locked "${PKGS[@]}"
  printf '\nadapter tests: OK (%d crates, no instrumentation)\n' "${#CRATES[@]}" >&2
  exit 0
fi

cargo llvm-cov --version >/dev/null 2>&1 || {
  echo "coverage.sh: cargo-llvm-cov is not installed (cargo install cargo-llvm-cov --locked," \
       "or taiki-e/install-action in CI) and the toolchain needs: rustup component add llvm-tools-preview" >&2
  exit 1
}

OUT="${FV_COV_OUT:-target/llvm-cov}"
mkdir -p "$OUT"

# One instrumented `cargo test` of the set, then every report from its
# profiles: no second compile, no second run.
run cargo llvm-cov clean --workspace
run cargo llvm-cov --no-report --locked "${PKGS[@]}"
run cargo llvm-cov report --lcov --output-path "$OUT/lcov.info"
run cargo llvm-cov report --html --output-dir "$OUT/html"
run cargo llvm-cov report --json --summary-only --output-path "$OUT/summary.json"
run cargo llvm-cov report --summary-only

# The totals as a markdown table (summary.md and, in CI, the job summary).
if command -v python3 >/dev/null; then
  python3 -I - "$OUT/summary.json" "${CRATES[@]}" >"$OUT/summary.md" <<'PY'
import json, sys
path, crates = sys.argv[1], sys.argv[2:]
with open(path) as f:
    totals = json.load(f)["data"][0]["totals"]
rows = []
for key, label in (("lines", "Lines"), ("regions", "Regions"), ("functions", "Functions")):
    t = totals.get(key)
    if t:
        rows.append(f"| {label} | {t['covered']} / {t['count']} | {t['percent']:.2f}% |")
print("### Adapter test coverage (cargo-llvm-cov)")
print()
print("| | covered / total | % |")
print("|---|---|---|")
print("\n".join(rows))
print()
print(f"Crates under test ({len(crates)}): " + ", ".join(f"`{c}`" for c in crates) + ".")
print("Reports: `lcov.info` and `html/` in the `adapter-coverage` artifact.")
PY
  cat "$OUT/summary.md"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    cat "$OUT/summary.md" >>"$GITHUB_STEP_SUMMARY"
  fi
fi

printf '\nadapter coverage: OK (%s)\n' "$OUT" >&2
