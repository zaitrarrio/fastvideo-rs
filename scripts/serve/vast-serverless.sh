#!/usr/bin/env bash
# fv-serve on Vast serverless (docs/serve/design.md §6.2 "Vast serverless"; WP-16):
# endpoint -> workergroup -> template whose onstart starts fv-serve (with the
# forwarder route) and the Vast PyWorker running deploy/vast/worker.py.
#
#   vast-serverless.sh plan [image]   print the template onstart, the CLI calls and
#                                     validate them (no API call, no key needed)
#   vast-serverless.sh up [image]     create template + endpoint + workergroup
#                                     (needs VAST_API_KEY and the vastai CLI)
#   vast-serverless.sh down <endpoint name>
#
# The PyWorker is the supported Vast path (research-deploy §3.4 option a):
# start_server.sh at a pinned PYWORKER_REF with the vastai SDK at a pinned
# SDK_VERSION. Our worker.py is baked into the `serve` image at
# /opt/fastvideo-rs/deploy/vast/worker.py; onstart turns it into a one-file
# local git repo (start_server.sh clones PYWORKER_REPO and runs
# `python -m worker`). Batch only; streams go to Vast instances (vast.sh).
#
# Secrets are Vast account env vars (FV_CF_*, FV_R2_*, FV_WEBHOOK_ED25519_KEY),
# never in the template. Money: max_workers 1, cold_workers 0, and every
# created id is appended to the ledger; `down` removes them.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=../gpu/lib.sh
source "$HERE/../gpu/lib.sh"

IMAGE_DEFAULT="ghcr.io/zaitrarrio/fastvideo-rs-serve:latest"
CONFIG="${FV_SERVE_CONFIG:-/etc/fv/runpod-fake.toml}"
# Pinned PyWorker bootstrap and SDK (research-deploy: pw@60cfeca, vastai 1.8.2).
PYWORKER_PIN="${PYWORKER_PIN:-60cfeca889f979fd73cf9e00adcd7b0ebc016fdc}"
SDK_VERSION="${VAST_SDK_VERSION:-1.8.2}"
ENDPOINT="${VAST_ENDPOINT_NAME:-fv-serve}"
GPU_RAM="${VAST_GPU_RAM_GB:-16}"
MAX_DPH="${VAST_MAX_DPH:-0.60}"
LEDGER="${FV_SERVE_LEDGER:-$FV_ROOT/artifacts/vast/serve/ledger.tsv}"
MODEL_LOG=/var/log/fv-serve.log

ledger() { mkdir -p "$(dirname "$LEDGER")"; printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }

onstart() {
  cat <<EOF
#!/bin/bash
set -u
env | grep -E '^(FV_|VAST_|PUBLIC_IPADDR|CONTAINER_ID|RUST_LOG)' >> /etc/environment
mkdir -p /var/log /fvstate
FV_SERVE_FORWARD=1 nohup /opt/fastvideo-rs/bin/fv-serve --config $CONFIG >> $MODEL_LOG 2>&1 &
command -v git >/dev/null && command -v python3 >/dev/null || { apt-get update -qq && apt-get install -y -qq git python3 python3-venv >/dev/null; }
R=/opt/fv-pyworker; rm -rf \$R; mkdir -p \$R
cp /opt/fastvideo-rs/deploy/vast/worker.py \$R/worker.py
git -C \$R init -q && git -C \$R add worker.py && git -C \$R -c user.name=fv -c user.email=fv@localhost commit -qm worker
export PYWORKER_REPO=\$R PYWORKER_REF=\$(git -C \$R rev-parse HEAD) SDK_VERSION=$SDK_VERSION MODEL_LOG=$MODEL_LOG
curl -fsSL https://raw.githubusercontent.com/vast-ai/pyworker/$PYWORKER_PIN/start_server.sh -o /opt/start_server.sh
bash /opt/start_server.sh >> /var/log/pyworker.log 2>&1 &
EOF
}

# Template env: the docker-flag form the CLI takes (WORKER_PORT is the
# PyWorker's TLS port; 8000 stays inside the box).
template_env() {
  printf '%s' "-p 3000:3000 -e FV_SERVE_MODE=http -e FV_STATE_DIR=/fvstate -e FV_AUTH_MODE=trust-gateway -e WORKER_PORT=3000 -e RUST_LOG=info"
}

search_params() {
  printf 'num_gpus=1 cuda_max_good>=13.0 direct_port_count>=1 verified=true rentable=true dph_total<=%s reliability2>=0.95' "$MAX_DPH"
}

validate() {
  local s
  s="$(onstart)"
  (( ${#s} < 4048 )) || die "onstart is ${#s} chars (Vast caps it at 4048)"
  grep -q "vast-ai/pyworker/$PYWORKER_PIN/" <<<"$s" || die "PyWorker ref is not pinned"
  grep -qE 'FV_(CF|R2)_|WEBHOOK_ED25519' <<<"$s$(template_env)" && die "a secret is in the template"
  [[ -f "$FV_ROOT/deploy/vast/worker.py" ]] || die "deploy/vast/worker.py is missing"
  if command -v python3 >/dev/null; then
    python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$FV_ROOT/deploy/vast/worker.py" || die "worker.py does not parse"
  fi
  bash -n <<<"$s" || die "onstart is not valid bash"
}

cmd_plan() {
  local image="${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}"
  validate
  echo "# template onstart"
  onstart
  echo "# CLI"
  # shellcheck disable=SC2016 # printed literally
  printf 'vastai create template --name %q --image %q --env %q --onstart-cmd "$(onstart)" --disk_space 40 --ssh --direct\n' \
    "$ENDPOINT-tpl" "$image" "$(template_env)"
  printf 'vastai create endpoint --endpoint_name %q --max_workers 1 --cold_workers 0 --inactivity_timeout 60\n' "$ENDPOINT"
  printf 'vastai create workergroup --endpoint_name %q --template_hash <hash> --gpu_ram %s --search_params %q\n' \
    "$ENDPOINT" "$GPU_RAM" "$(search_params)"
  log "plan: onstart/template OK (PyWorker $PYWORKER_PIN, SDK $SDK_VERSION)"
}

cmd_up() {
  [[ -n "${VAST_API_KEY:-}" ]] || die "VAST_API_KEY is not set: only 'vast-serverless.sh plan' runs without it"
  require_tools vastai jq
  vast_check_auth
  validate
  local image="${1:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}" out hash
  [[ "$image" == *@sha256:* ]] || die "pin the image digest"
  out="$(vastai create template --name "$ENDPOINT-tpl" --image "$image" --env "$(template_env)" \
    --onstart-cmd "$(onstart)" --disk_space 40 --ssh --direct --raw)"
  hash="$(jq -r '.template.hash_id // .hash_id // empty' <<<"$out")"
  [[ -n "$hash" ]] || die "template create: $(head -c 300 <<<"$out")"
  ledger "vast-template-created $hash $image"
  vastai create endpoint --endpoint_name "$ENDPOINT" --max_workers 1 --cold_workers 0 --inactivity_timeout 60 --raw >/dev/null
  ledger "vast-endpoint-created $ENDPOINT"
  vastai create workergroup --endpoint_name "$ENDPOINT" --template_hash "$hash" --gpu_ram "$GPU_RAM" \
    --search_params "$(search_params)" --raw >/dev/null
  ledger "vast-workergroup-created $ENDPOINT template=$hash"
  log "endpoint $ENDPOINT up (template $hash); remove with: $0 down $ENDPOINT"
}

cmd_down() {
  [[ -n "${VAST_API_KEY:-}" ]] || die "VAST_API_KEY is not set"
  require_tools vastai jq
  local name="${1:?endpoint name}" id
  id="$(vastai show endpoints --raw | jq -r --arg n "$name" '.[] | select(.endpoint_name == $n) | .id' | head -1)"
  [[ -n "$id" ]] || die "no endpoint named $name"
  vastai delete endpoint "$id" && ledger "vast-endpoint-deleted $name ($id)"
}

case "${1:-}" in
  plan) shift; cmd_plan "$@" ;;
  up) shift; cmd_up "$@" ;;
  down) shift; cmd_down "$@" ;;
  *) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
