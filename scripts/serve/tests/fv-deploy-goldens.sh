#!/usr/bin/env bash
# The payload goldens of fv-deploy (docs/adr/0018-fv-deploy.md §4, §8): what
# scripts/serve/runpod-pod.sh, scripts/serve/runpod-endpoint.sh,
# scripts/serve/vast.sh and scripts/gcp/vm.sh send today, from their own
# `plan` commands (vm.sh: its `firewall_payload` / `instance_payload`, the
# functions `vm.sh plan` and `up` call), for fixed inputs. The Rust provider
# code must build the same bodies
# (crates/fastvideo-deploy/tests/payload_goldens.rs).
#
#   fv-deploy-goldens.sh           regenerate crates/fastvideo-deploy/fixtures/deploy/*.json
#   fv-deploy-goldens.sh --check   fail when a script's payload differs from the fixture
#
# No network: a fake curl on PATH answers the one lookup the pod plan makes
# (GET /networkvolumes/<id> -> its data centre) and refuses anything else;
# the GCP functions run in vm.sh's dry-run mode with a fixed project. In the
# GCP instance, the metadata items that carry a repository file
# (startup-script, fv-config, the verify scripts and tables) are written as
# "<file>" and fv-env as the object it encodes, so the fixture does not
# change with those files; the Rust test checks those items against the
# files themselves.
#
# C3d (ADR-0018 §10): `runpod-templates.sh plan` gives the serve templates'
# create bodies (serve-templates.json); fv-deploy-release-goldens.sh runs
# release.sh promote / rollback / reconcile, runpod-templates.sh sync and the
# deployments writes of registry.sh against scripts/serve/tests/mock_api.py
# and records what they send (release.json). Needs bash and jq; the release
# goldens also python3 and curl (no network: the mock is local).
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

# A step's stderr goes to a file and is printed when the step fails (CI then
# names the cause); a step that succeeds stays quiet.
step() {
  local name="$1"
  shift
  if ! "$@" 2>"$TMP/$name.err"; then
    echo "fv-deploy-goldens: $name failed:" >&2
    cat "$TMP/$name.err" >&2
    exit 1
  fi
}
# Sorted keys, numbers in their canonical form: jq 1.7 keeps a literal's
# spelling (13.0, 0.60), jq 1.6 (ubuntu-22.04) prints 13, 0.6; `. + 0`
# prints the same under both.
canon() { jq -S '(.. | numbers) |= . + 0' "$@"; }
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
  step runpod-pod bash "$ROOT/scripts/serve/runpod-pod.sh" plan "$IMAGE" >"$TMP/pod-plan.json"
step runpod-pod-json canon "$TMP/pod-plan.json" >"$TMP/runpod-pod.json"

# runpod-endpoint.sh plan: the template, the queue endpoint (and the load
# balancer, not covered: Runpod LB mode is an open owner decision).
PATH="$TMP/bin:$PATH" RUNPOD_VOLUME_ID="$VOLUME" RUNPOD_VOLUME_DC="$DC" RUNPOD_GPU_TYPES="$GPU" \
  FV_SERVE_CONFIG="$CONFIG" RUNPOD_ALLOWED_CUDA="" FV_RELEASE_CHANNEL="" FV_FLASHBOOT=0 FV_IDLE_TIMEOUT_S=5 \
  step runpod-endpoint bash "$ROOT/scripts/serve/runpod-endpoint.sh" plan "$IMAGE" >"$TMP/endpoint-plan.txt"
awk '/^# template/{f="t"; next} /^# queue endpoint/{f="e"; next} /^# load balancer/{f=""; next}
     f=="t"{print > "'"$TMP"'/t.json"} f=="e"{print > "'"$TMP"'/e.json"}' "$TMP/endpoint-plan.txt"
step runpod-template-json canon "$TMP/t.json" >"$TMP/runpod-template.json"
step runpod-endpoint-json canon "$TMP/e.json" >"$TMP/runpod-endpoint.json"

# vast.sh plan: the offer search and the create body (VAST_MAX_DPH 0.60,
# VAST_DISK_GB 40, FV_SERVE_CONFIG /etc/fv/runpod-fake.toml: the defaults).
step vast env -u VAST_API_KEY -u VAST_MAX_DPH -u VAST_DISK_GB -u FV_SERVE_CONFIG \
  bash "$ROOT/scripts/serve/vast.sh" plan "$IMAGE" >"$TMP/vast-plan.txt"
awk '/^# POST /{f="o"; next} /^# PUT /{f="c"; next}
     f=="o"{print > "'"$TMP"'/o.json"} f=="c"{print > "'"$TMP"'/c.json"}' "$TMP/vast-plan.txt"
step vast-offers-json canon "$TMP/o.json" >"$TMP/vast-offers.json"
step vast-create-json canon "$TMP/c.json" >"$TMP/vast-create.json"

# vm.sh: the firewall rule and the instance of `vm.sh up h3-turbo` (serve
# role, no secrets set, cap 1800 s, a fixed run stamp and deadline).
GCP_VM="fv-h3-turbo-1025120000"
step gcp env -i PATH="$PATH" HOME="$TMP/home" FV_GCP_DRY_RUN=1 GCP_PROJECT=golden-project GCP_ZONE=europe-west4-b \
  FV_GCP_CAP_S=1800 FV_GCP_SOURCE_CIDR=198.51.100.7/32 FV_GCP_KEY_DEFAULT=/nonexistent \
  bash -c 'source "$1/scripts/gcp/vm.sh" && VM_ROLE=serve &&
    instance_payload "$2" h3-turbo g4-standard-48 "$3" "<sha256 of the run key>" 1025120000 1790000000 >"$4/i.json" &&
    firewall_payload "$2" 1025120000 "$(source_cidrs)" >"$4/f.json"' _ "$ROOT" "$GCP_VM" "$IMAGE" "$TMP"
step gcp-instance-items jq '.metadata.items |= map(
    if (.key | IN("startup-script", "fv-config", "fv-script-verify-weights", "fv-script-verify-safetensors", "fv-manifest", "fv-sha256"))
    then .value = "<file>"
    elif .key == "fv-env" then .value |= fromjson
    else . end)' "$TMP/i.json" >"$TMP/i2.json"
step gcp-instance-json canon "$TMP/i2.json" >"$TMP/gcp-instance.json"
step gcp-firewall-json canon "$TMP/f.json" >"$TMP/gcp-firewall.json"

# runpod-templates.sh plan: the create bodies of fv-serve-cuda-sls, fv-serve-cuda-pod (the CUDA image) and
# fv-serve-cpu-pod (the CPU image, a second digest), on the stable channel.
IMAGE_CPU="ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:2222222222222222222222222222222222222222222222222222222222222222"
FV_RELEASE_CHANNEL=stable step serve-templates-cuda bash "$ROOT/scripts/serve/runpod-templates.sh" plan cuda "$IMAGE" >"$TMP/tc.json"
FV_RELEASE_CHANNEL=stable step serve-templates-cpu bash "$ROOT/scripts/serve/runpod-templates.sh" plan cpu "$IMAGE_CPU" >>"$TMP/tc.json"
step serve-templates-json bash -c 'jq -s . "$1" | jq -S "(.. | numbers) |= . + 0"' _ "$TMP/tc.json" >"$TMP/serve-templates.json"

# release.sh / runpod-templates.sh sync / registry.sh against the mock.
step release bash "$HERE/fv-deploy-release-goldens.sh" "$TMP/release.json"

rc=0
mkdir -p "$OUT"
for f in runpod-pod.json runpod-template.json runpod-endpoint.json vast-offers.json vast-create.json \
  gcp-instance.json gcp-firewall.json serve-templates.json release.json; do
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
