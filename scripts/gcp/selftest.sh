#!/usr/bin/env bash
# Offline check of scripts/gcp (no credentials, no API calls, no spend): the
# linter, the JWT self-test, every command in FV_GCP_DRY_RUN=1 mode with fake
# secret values that must never appear in the output, and the mocked-API
# test of vm.sh create/delete/reap (tests/vm.test.sh).
#
#   selftest.sh [out dir]   (default: a temp dir; payloads kept there)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$(mktemp -d "${TMPDIR:-/tmp}/fv-gcp-selftest.XXXXXX")}"
mkdir -p "$OUT"
fail=0
step() { printf '== %s\n' "$*"; }

step "bash -n + shellcheck"
for f in "$HERE"/*.sh "$HERE"/tests/*.sh; do bash -n "$f" || { echo "PARSE $f"; fail=1; }; done
if command -v shellcheck >/dev/null; then
  shellcheck -x "$HERE"/*.sh "$HERE"/tests/*.sh || fail=1
else
  echo "shellcheck not installed: skipped"
fi

step "auth self-test"
bash "$HERE/auth.sh" self-test || fail=1

step "mocked API: vm.sh create / delete / reap"
bash "$HERE/tests/vm.test.sh" >"$OUT/vm-test.out" 2>&1 || { fail=1; grep -E '^FAIL' -A6 "$OUT/vm-test.out" | head -40; }
tail -1 "$OUT/vm-test.out"

# Fake secrets: none may leak into any dry-run output.
SENTINEL="fvsentinel$(date +%s)"
export FV_GCP_DRY_RUN=1 GCP_PROJECT=fv-selftest GCP_ZONE=europe-west4-b FV_GCP_STATE="$OUT/state" \
  FV_GCP_LEDGER="$OUT/ledger.tsv" FV_GCP_OUT="$OUT/artifacts" FV_GCP_SOURCE_CIDR="198.51.100.7/32" \
  FV_GCP_KEY_DEFAULT="$OUT/no-key" \
  FV_CF_ACCOUNT_ID="${SENTINEL}a" FV_CF_API_TOKEN="${SENTINEL}b" FV_R2_SECRET_ACCESS_KEY="${SENTINEL}c" \
  FV_WEBHOOK_ED25519_KEY="${SENTINEL}d" HF_TOKEN="${SENTINEL}e" \
  FV_SERVE_IMAGE="ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:0000000000000000000000000000000000000000000000000000000000000000"
unset GCP_SA_KEY_JSON GCP_SA_KEY_FILE FV_GCP_ROLE FV_DISPATCH_DO_URL FV_INTERNAL_TOKEN FV_ADMIN_TOKEN

run() { # name, command...
  local name="$1"; shift
  if "$@" >"$OUT/$name.out" 2>&1; then echo "ok   $name"; else echo "FAIL $name (rc=$?): $(tail -3 "$OUT/$name.out" | tr '\n' ' ')"; fail=1; fi
}
while read -r fam; do run "vm-plan-$fam" bash "$HERE/vm.sh" plan "$fam"; done < <(awk -F'\t' '!/^#/ && NF > 1 {print $1}' "$HERE/families.tsv")
run vm-plan-a3-spot env FV_GCP_MACHINE=a3-highgpu-1g FV_GCP_MAX_DPH=8 bash "$HERE/vm.sh" plan h3-max
run vm-plan-g2-wan env FV_GCP_MACHINE_WAN=g2-standard-16 bash "$HERE/vm.sh" plan wan
run vm-plan-secret-manager env FV_GCP_SECRETS=secret-manager bash "$HERE/vm.sh" plan h3-turbo
WORKER_ENV=(FV_GCP_ROLE=worker FV_DISPATCH_DO_URL=https://fv-edge.example.workers.dev FV_INTERNAL_TOKEN="${SENTINEL}f")
run vm-plan-worker-h3 env "${WORKER_ENV[@]}" bash "$HERE/vm.sh" plan h3-turbo
run vm-plan-worker-direct env "${WORKER_ENV[@]}" FV_GCP_DIRECT=1 FV_ADMIN_TOKEN="fvadm_${SENTINEL}g" FV_D1_DATABASE_ID=d1 bash "$HERE/vm.sh" plan ltx
run vm-down bash "$HERE/vm.sh" down fv-h3-turbo-0101000000
run vm-reap bash "$HERE/vm.sh" reap
run vm-list bash "$HERE/vm.sh" list
run vm-logs bash "$HERE/vm.sh" ssh-free-logs fv-h3-turbo-0101000000
run vm-secrets-push bash "$HERE/vm.sh" secrets-push
run vm-preflight bash "$HERE/vm.sh" preflight
for c in plan bucket quantize image disk-up disk-down work-down; do run "weights-$c" bash "$HERE/weights.sh" "$c"; done
run weights-populate env FV_GCP_WEIGHTS_APPROVED=1 bash "$HERE/weights.sh" populate
run weights-populate-extra env FV_GCP_WEIGHTS_APPROVED=1 FV_GCP_WEIGHTS_EXTRA="mmaudio-44k-v2 fastwan22-ti2v-5b" bash "$HERE/weights.sh" populate
run e2e bash "$HERE/e2e.sh"

step "guards"
if FV_GCP_MACHINE=a3-highgpu-1g bash "$HERE/vm.sh" plan h3-max >"$OUT/cap.out" 2>&1; then
  echo "FAIL: a3-highgpu-1g passed the default \$6/hr cap"; fail=1
elif grep -q 'cap' "$OUT/cap.out"; then echo "ok   cap refusal"; else echo "FAIL cap message"; fail=1; fi
if bash "$HERE/weights.sh" populate >"$OUT/approve.out" 2>&1; then
  echo "FAIL: populate ran without FV_GCP_WEIGHTS_APPROVED"; fail=1
elif grep -q "owner's approval" "$OUT/approve.out"; then echo "ok   populate needs the owner's approval"; else echo "FAIL approval message"; fail=1; fi
if grep -q 'FV_DISPATCH_FAMILIES\\":\\"h3\\"' "$OUT/vm-plan-worker-h3.out"; then
  echo "ok   worker plan joins the h3 family Durable Object"; else echo "FAIL worker plan"; fail=1; fi

step "no secret value in any output"
if grep -rl "$SENTINEL" "$OUT" --exclude=vm-test.out >/dev/null 2>&1; then
  echo "FAIL: a secret value leaked into: $(grep -rl "$SENTINEL" "$OUT" | tr '\n' ' ')"; fail=1
else
  echo "ok   no leak ($(grep -rh '^DRY-RUN ' "$OUT"/*.out | wc -l) REST calls printed, outputs in $OUT)"
fi
step "every mutating call targets a labelled or fv- resource"
grep -rh '^DRY-RUN \(POST\|DELETE\)' "$OUT"/*.out | sed -E 's/[0-9]{10}/<run>/g; s/[0-9]{8}-[0-9]{4}/<date>/g' | sort | uniq -c
exit $fail
