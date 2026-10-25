#!/usr/bin/env bash
# The serve images bake no worker config (one image, docs/adr/0011-deploy-section.md): the pod scripts that used to
# fall back to an image's baked FV_CONFIG send one instead. No network, no pod.
#
#   bash scripts/serve/tests/no-baked-config.test.sh      # needs bash, curl, python3
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
for t in curl python3; do command -v "$t" >/dev/null || { echo "no-baked-config.test: skipped (no $t)"; exit 0; }; done
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
PASS=0 FAIL=0
ok() { PASS=$((PASS + 1)); printf 'ok   %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL %s\n' "$1"; [[ -z "${2:-}" ]] || printf '%s\n' "$2" | sed 's/^/     /' | head -20; }

# A stand-in fv-boot: records its arguments, refuses (exits 1), so the smoke stops at once.
cat >"$T/fv-boot" <<'SH'
#!/usr/bin/env bash
[ "${1:-}" = version ] && { echo stand-in; exit 0; }
printf '%s\n' "$@" >"$FAKE_ARGS"
exit 1
SH
chmod +x "$T/fv-boot"
smoke() { env -u FV_CONFIG -u FV_WORKER_CONFIG -u FV_SMOKE_CONFIG "$@" FAKE_ARGS="$T/args" FV_BOOT_BIN="$T/fv-boot" FV_SMOKE_OUT="$T/out" FV_SMOKE_WAIT_S=3 \
  timeout 60 bash "$ROOT/scripts/gpu/fv-boot-smoke-pod.sh" >/dev/null 2>&1; tr '\n' ' ' <"$T/args"; }

got="$(smoke)"
[[ "$got" == "--config /etc/fv/runpod-fake.toml " ]] && ok "fv-boot-smoke-pod.sh: no config given, the one the CUDA image carries" || bad "fv-boot-smoke-pod.sh: default config" "$got"
got="$(smoke FV_CONFIG=/etc/fv/runpod.toml)"
[[ "$got" == "--config /etc/fv/runpod-fake.toml " ]] && ok "fv-boot-smoke-pod.sh: an old image's FV_CONFIG is not read" || bad "fv-boot-smoke-pod.sh: FV_CONFIG ignored" "$got"
got="$(smoke FV_WORKER_CONFIG=/x/w.toml)"
[[ "$got" == "--config /x/w.toml " ]] && ok "fv-boot-smoke-pod.sh: FV_WORKER_CONFIG" || bad "fv-boot-smoke-pod.sh: FV_WORKER_CONFIG" "$got"
got="$(smoke FV_WORKER_CONFIG=/x/w.toml FV_SMOKE_CONFIG=/x/s.toml)"
[[ "$got" == "--config /x/s.toml " ]] && ok "fv-boot-smoke-pod.sh: FV_SMOKE_CONFIG wins" || bad "fv-boot-smoke-pod.sh: FV_SMOKE_CONFIG" "$got"

# segments-pod.sh: a volume or R2 run must name its config (refused before any API call); the SF-Wan path ships
# configs/serve/runpod-sfwan.toml, which must exist.
out="$(env -u FV_SEG_CONFIG RUNPOD_API_KEY=not-used RUNPOD_API_BASE=http://127.0.0.1:9 RUNPOD_IMAGE=x FV_SEG_VOLUME=v timeout 20 bash "$ROOT/scripts/serve/e2e/segments-pod.sh" 2>&1)"; rc=$?
[[ "$rc" == 2 && "$out" == *"set FV_SEG_CONFIG"* ]] && ok "segments-pod.sh: a volume run without FV_SEG_CONFIG is refused" || bad "segments-pod.sh: volume without config" "rc=$rc $out"
out="$(env -u FV_SEG_CONFIG RUNPOD_API_KEY=not-used RUNPOD_API_BASE=http://127.0.0.1:9 RUNPOD_IMAGE=x FV_SEG_R2_TREES=t timeout 20 bash "$ROOT/scripts/serve/e2e/segments-pod.sh" 2>&1)"; rc=$?
[[ "$rc" == 2 && "$out" == *"set FV_SEG_CONFIG"* ]] && ok "segments-pod.sh: an R2 run without FV_SEG_CONFIG is refused" || bad "segments-pod.sh: R2 without config" "rc=$rc $out"
if grep -q 'SEG_CONFIG="$ROOT/configs/serve/runpod-sfwan.toml"' "$ROOT/scripts/serve/e2e/segments-pod.sh" && [[ -f "$ROOT/configs/serve/runpod-sfwan.toml" ]]; then
  ok "segments-pod.sh: the SF-Wan path ships configs/serve/runpod-sfwan.toml"
else bad "segments-pod.sh: the SF-Wan default config"; fi

printf '\nno-baked-config.test: %d passed, %d failed\n' "$PASS" "$FAIL"
[[ "$FAIL" == 0 ]]
