#!/usr/bin/env bash
# Offline check of scripts/gcp (no credentials, no API calls, no spend): the
# linter, the JWT self-test, and every command in FV_GCP_DRY_RUN=1 mode
# with fake secret values that must never appear in the output.
#
#   selftest.sh [out dir]   (default: a temp dir; payloads kept there)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$(mktemp -d "${TMPDIR:-/tmp}/fv-gcp-selftest.XXXXXX")}"
mkdir -p "$OUT"
fail=0
step() { printf '== %s\n' "$*"; }

step shellcheck
if command -v shellcheck >/dev/null; then
  shellcheck -x "$HERE"/*.sh || fail=1
else
  echo "shellcheck not installed: skipped"
fi

step "auth self-test"
bash "$HERE/auth.sh" self-test || fail=1

# Fake secrets: none may leak into any dry-run output.
SENTINEL="fvsentinel$(date +%s)"
export FV_GCP_DRY_RUN=1 GCP_PROJECT=fv-selftest GCP_ZONE=us-central1-b \
  FV_GCP_LEDGER="$OUT/ledger.tsv" FV_GCP_OUT="$OUT/artifacts" FV_GCP_SOURCE_CIDR="198.51.100.7/32" \
  FV_CF_ACCOUNT_ID="${SENTINEL}a" FV_CF_API_TOKEN="${SENTINEL}b" FV_R2_SECRET_ACCESS_KEY="${SENTINEL}c" \
  FV_WEBHOOK_ED25519_KEY="${SENTINEL}d" HF_TOKEN="${SENTINEL}e" \
  FV_SERVE_IMAGE="ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:0000000000000000000000000000000000000000000000000000000000000000"
unset GCP_SA_KEY_JSON GCP_SA_KEY_FILE

run() { # name, command...
  local name="$1"; shift
  if "$@" >"$OUT/$name.out" 2>&1; then echo "ok   $name"; else echo "FAIL $name (rc=$?): $(tail -3 "$OUT/$name.out" | tr '\n' ' ')"; fail=1; fi
}
for fam in h3-turbo h3-max ltx-turbo wan-turbo fake; do run "vm-plan-$fam" bash "$HERE/vm.sh" plan "$fam"; done
run vm-plan-a3-spot env FV_GCP_MACHINE=a3-highgpu-1g FV_GCP_MAX_DPH=8 bash "$HERE/vm.sh" plan h3-max
run vm-plan-g2-wan env FV_GCP_MACHINE_WAN_TURBO=g2-standard-16 bash "$HERE/vm.sh" plan wan-turbo
run vm-plan-secret-manager env FV_GCP_SECRETS=secret-manager bash "$HERE/vm.sh" plan h3-turbo
run vm-down bash "$HERE/vm.sh" down fv-h3-turbo-0101000000
run vm-list bash "$HERE/vm.sh" list
run vm-logs bash "$HERE/vm.sh" ssh-free-logs fv-h3-turbo-0101000000
run vm-secrets-push bash "$HERE/vm.sh" secrets-push
run vm-preflight bash "$HERE/vm.sh" preflight
for c in plan bucket populate quantize image disk-up disk-down work-down; do run "weights-$c" bash "$HERE/weights.sh" "$c"; done
run e2e bash "$HERE/e2e.sh"

step "price cap refuses an expensive machine"
if FV_GCP_MACHINE=a3-highgpu-1g bash "$HERE/vm.sh" plan h3-max >"$OUT/cap.out" 2>&1; then
  echo "FAIL: a3-highgpu-1g passed the default \$6/hr cap"; fail=1
else
  if grep -q 'cap' "$OUT/cap.out"; then echo "ok   cap refusal"; else echo "FAIL cap message"; fail=1; fi
fi

step "no secret value in any output"
if grep -rl "$SENTINEL" "$OUT" >/dev/null 2>&1; then
  echo "FAIL: a secret value leaked into: $(grep -rl "$SENTINEL" "$OUT" | tr '\n' ' ')"; fail=1
else
  echo "ok   no leak ($(grep -rh '^DRY-RUN ' "$OUT"/*.out | wc -l) REST calls printed, outputs in $OUT)"
fi
step "every mutating call targets a labelled or fv- resource"
grep -rh '^DRY-RUN \(POST\|DELETE\)' "$OUT"/*.out | sed -E 's/[0-9]{10}/<run>/g; s/[0-9]{8}-[0-9]{4}/<date>/g' | sort | uniq -c
exit $fail
