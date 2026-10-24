#!/usr/bin/env bash
# The payload goldens of fv-deploy (docs/adr/0018-fv-deploy.md §4): what
# scripts/serve/runpod-pod.sh and scripts/serve/runpod-endpoint.sh send today,
# from their own `plan` commands, for fixed inputs. The Rust provider code
# must build the same bodies (crates/fastvideo-deploy/tests/payload_goldens.rs).
#
#   fv-deploy-goldens.sh           regenerate crates/fastvideo-deploy/fixtures/deploy/*.json
#   fv-deploy-goldens.sh --check   fail when a script's payload differs from the fixture
#
# No network: a fake curl on PATH answers the one lookup the pod plan makes
# (GET /networkvolumes/<id> -> its data centre) and refuses anything else.
# Needs bash and jq.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="$ROOT/crates/fastvideo-deploy/fixtures/deploy"
CHECK=0
[[ "${1:-}" == --check ]] && CHECK=1

# The inputs (tests/payload_goldens.rs uses the same).
IMAGE="ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:1111111111111111111111111111111111111111111111111111111111111111"
GPU="NVIDIA RTX PRO 6000 Blackwell Server Edition"
CONFIG="/etc/fv/runpod.toml"
VOLUME="bz0ofkfx45"
DC="EU-RO-1"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin"
cat >"$TMP/bin/curl" <<EOF
#!/usr/bin/env bash
for a in "\$@"; do
  case "\$a" in */networkvolumes/$VOLUME) echo '{"id":"$VOLUME","dataCenterId":"$DC"}'; exit 0 ;; esac
done
echo "fake curl: no network in the goldens (\$*)" >&2
exit 7
EOF
chmod +x "$TMP/bin/curl"

# runpod-pod.sh plan: one PodCreateInput.
PATH="$TMP/bin:$PATH" RUNPOD_API_KEY=golden RUNPOD_VOLUME_ID="$VOLUME" RUNPOD_GPU_TYPES="$GPU" \
  FV_SERVE_CONFIG="$CONFIG" RUNPOD_ALLOWED_CUDA="" FV_RELEASE_CHANNEL="" \
  bash "$ROOT/scripts/serve/runpod-pod.sh" plan "$IMAGE" | jq -S . >"$TMP/runpod-pod.json"

# runpod-endpoint.sh plan: the template, the queue endpoint (and the load
# balancer, not covered: Runpod LB mode is an open owner decision).
PATH="$TMP/bin:$PATH" RUNPOD_VOLUME_ID="$VOLUME" RUNPOD_VOLUME_DC="$DC" RUNPOD_GPU_TYPES="$GPU" \
  FV_SERVE_CONFIG="$CONFIG" RUNPOD_ALLOWED_CUDA="" FV_RELEASE_CHANNEL="" FV_FLASHBOOT=0 FV_IDLE_TIMEOUT_S=5 \
  bash "$ROOT/scripts/serve/runpod-endpoint.sh" plan "$IMAGE" 2>/dev/null >"$TMP/endpoint-plan.txt"
awk '/^# template/{f="t"; next} /^# queue endpoint/{f="e"; next} /^# load balancer/{f=""; next}
     f=="t"{print > "'"$TMP"'/t.json"} f=="e"{print > "'"$TMP"'/e.json"}' "$TMP/endpoint-plan.txt"
jq -S . "$TMP/t.json" >"$TMP/runpod-template.json"
jq -S . "$TMP/e.json" >"$TMP/runpod-endpoint.json"

rc=0
mkdir -p "$OUT"
for f in runpod-pod.json runpod-template.json runpod-endpoint.json; do
  if [[ "$CHECK" == 1 ]]; then
    if ! diff -u "$OUT/$f" "$TMP/$f"; then
      echo "fv-deploy-goldens: $f differs from what the script sends (rerun without --check and fix the Rust payload)" >&2
      rc=1
    fi
  else
    cp "$TMP/$f" "$OUT/$f"
    echo "wrote ${OUT#"$ROOT"/}/$f"
  fi
done
exit "$rc"
