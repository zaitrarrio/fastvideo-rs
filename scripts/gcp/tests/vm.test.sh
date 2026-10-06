#!/usr/bin/env bash
# Tests for scripts/gcp/vm.sh (create / delete / reap) and auth.sh against
# scripts/gcp/tests/gcp_mock.py, a fake OAuth token endpoint plus Compute
# Engine API. The service-account key is a throwaway RSA key made here whose
# token_uri points at the mock. No network, no credentials, nothing rented.
#
#   bash scripts/gcp/tests/vm.test.sh     # needs bash, curl, jq, openssl, python3
# shellcheck disable=SC2016 # $vars inside bash -c checks expand when they run
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GCP="$(cd "$HERE/.." && pwd)"
for t in curl jq openssl python3 sha256sum; do command -v "$t" >/dev/null || { echo "vm.test: skipped (no $t)"; exit 0; }; done

T="$(mktemp -d)"
cleanup() { [[ -n "${MOCK_PID:-}" ]] && kill "$MOCK_PID" 2>/dev/null; wait 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
python3 -I "$HERE/gcp_mock.py" "$T/port" &
MOCK_PID=$!
for _ in $(seq 50); do [[ -s "$T/port" ]] && break; sleep 0.1; done
M="http://127.0.0.1:$(cat "$T/port")"
TOKEN="ya29.mock-access-token-0123456789"

# A throwaway service-account key for the mock's token endpoint.
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$T/k.pem" 2>/dev/null
( umask 077; jq -n --rawfile pk "$T/k.pem" --arg tu "$M/token" '{type: "service_account", project_id: "fv-mock",
  private_key_id: "k1", private_key: $pk, client_email: "fv-test@fv-mock.iam.gserviceaccount.com", token_uri: $tu}' >"$T/key.json" )
PK_LINE="$(sed -n 2p "$T/k.pem")" # a line of the private key: must never be printed

unset GCP_SA_KEY_JSON GCP_PROJECT FV_GCP_DRY_RUN FV_DISPATCH_DO_URL FV_INTERNAL_TOKEN FV_ADMIN_TOKEN FV_GCP_SECRETS_FILE
for n in FV_CF_ACCOUNT_ID FV_CF_API_TOKEN FV_D1_DATABASE_ID FV_R2_BUCKET FV_R2_ENDPOINT FV_R2_ACCESS_KEY_ID \
  FV_R2_SECRET_ACCESS_KEY FV_WEBHOOK_ED25519_KEY FV_URL_SIGNING_KEY FV_GCP_ROLE FV_GCP_TLS FV_GCP_MACHINE; do unset "$n"; done
export FV_GCP_COMPUTE_API="$M/compute/v1" FV_GCP_STATE="$T/state" FV_GCP_LEDGER="$T/ledger.tsv" FV_GCP_OUT="$T/out" \
  FV_GCP_KEY_DEFAULT="$T/no-default-key" GCP_ZONE=europe-west4-b FV_GCP_SOURCE_CIDR="198.51.100.7/32"
IMG="ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:$(printf 'a%.0s' $(seq 64))"
export FV_SERVE_IMAGE="$IMG"

PASS=0 FAIL=0
ok() { PASS=$((PASS + 1)); printf 'ok   %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL %s\n' "$1"; [[ -z "${2:-}" ]] || printf '%s\n' "$2" | sed 's/^/     /' | head -30; }
check() { local name="$1"; shift; if "$@"; then ok "$name"; else bad "$name" "${OUT:-}"; fi; }
state() { curl -sS "$M/__state"; }
# snap [name]: the mock state into $T/<name>.json; prints the path (checks read
# files: the state outgrows one argument).
snap() { local f="$T/${1:-s}.json"; curl -sS "$M/__state" >"$f"; echo "$f"; }
# instance <state file> <name>: that instance into its own file; prints the path.
instance() { local f="$T/i-$2.json"; jq --arg n "$2" '.instances[$n]' "$1" >"$f"; echo "$f"; }
seed() { curl -sS -X POST -d "$1" "$M/__seed" >/dev/null; }
inserts() { state | jq '.inserts | length'; }
ALL_OUT=""
run() { OUT="$("$@" 2>&1)"; RC=$?; ALL_OUT+="$OUT"$'\n'; }
VM="$GCP/vm.sh"
meta() { jq -r --arg k "$2" '.metadata.items[] | select(.key == $k) | .value' <<<"$1"; }

# ---- no key: nothing is called
run bash "$VM" up h3-turbo
check "up without a key: refused, nothing created" bash -c '[[ $0 != 0 && $1 == *"GCP_SA_KEY_JSON"* && $2 == 0 ]]' "$RC" "$OUT" "$(inserts)"

export GCP_SA_KEY_FILE="$T/key.json"
seed '{"disks": {"fv-weights-hdml-europe-west4-b": {"name": "fv-weights-hdml-europe-west4-b", "_zone": "europe-west4-b", "labels": {"fv-owner": "fastvideo-rs"}, "users": []}}}'

# ---- auth against the mock token endpoint
run bash "$GCP/auth.sh" check
check "auth check: token minted from the key, project read" bash -c '[[ $0 == 0 && $1 == *"project=fv-mock"* ]]' "$RC" "$OUT"

# ---- guards before any create
run env FV_GCP_MACHINE=a3-highgpu-1g bash "$VM" up h3-max "$IMG"
check "price cap: a3-highgpu-1g over \$6/hr is refused before any call" bash -c '[[ $0 != 0 && $1 == *"cap"* && $2 == 0 ]]' "$RC" "$OUT" "$(inserts)"
run bash "$VM" up no-such-family "$IMG"
check "unknown family refused" bash -c '[[ $0 != 0 && $1 == *"unknown family"* && $2 == 0 ]]' "$RC" "$OUT" "$(inserts)"

# ---- standalone create
export FV_CF_API_TOKEN="cf-sentinel-secret-1" FV_R2_SECRET_ACCESS_KEY="r2-sentinel-secret-2"
run env FV_GCP_RUN=0101000001 bash "$VM" up h3-turbo "$IMG"
UP_OUT="$OUT"
S="$(snap)"
I="$(instance "$S" fv-h3-turbo-0101000001)"
check "up: prints <vm> <ip> <key>" bash -c '[[ $0 == 0 && $(tail -1 <<<"$1") =~ ^fv-h3-turbo-0101000001\ 127\.0\.0\.1\ fvk-[0-9a-f]{32}$ ]]' "$RC" "$UP_OUT"
check "up: labels owner/kind/run/family/role/deadline" bash -c 'jq -e ".labels | .[\"fv-owner\"] == \"fastvideo-rs\" and .[\"fv-family\"] == \"h3-turbo\" and .[\"fv-role\"] == \"serve\" and .[\"fv-run\"] == \"0101000001\" and (.[\"fv-deadline\"] | tonumber) > now" "$0" >/dev/null' "$I"
check "up: maxRunDuration 5400 with DELETE, no restart, TERMINATE on maintenance" bash -c 'jq -e ".scheduling | .maxRunDuration.seconds == \"5400\" and .instanceTerminationAction == \"DELETE\" and .automaticRestart == false and .onHostMaintenance == \"TERMINATE\"" "$0" >/dev/null' "$I"
check "up: G4 boots from hyperdisk-balanced; the weight disk is READ_ONLY" bash -c 'jq -e "(.machineType | endswith(\"g4-standard-48\")) and (.disks[0].initializeParams.diskType | endswith(\"hyperdisk-balanced\")) and .disks[1].mode == \"READ_ONLY\" and (.disks[1].source | endswith(\"fv-weights-hdml-europe-west4-b\"))" "$0" >/dev/null' "$I"
check "up: the VM runs as the fv-vm service account, not the deploy key's" bash -c 'jq -e ".serviceAccounts[0].email == \"fv-vm@fv-mock.iam.gserviceaccount.com\"" "$0" >/dev/null' "$I"
check "up: FV_API_KEYS is the key's SHA-256" bash -c 'k=$(tail -1 <<<"$1" | cut -d" " -f3); h=$(printf %s "$k" | sha256sum | cut -d" " -f1); jq -e --arg h "$h" ".metadata.items[] | select(.key == \"fv-env\") | .value | fromjson | .FV_API_KEYS == \$h and (has(\"FV_SERVE_ROLE\") | not)" "$0" >/dev/null' "$I" "$UP_OUT"
check "up: secrets go as fv-secret-* metadata, verify scripts shipped" bash -c 'jq -e "[.metadata.items[].key] | (index(\"fv-secret-FV_CF_API_TOKEN\") != null) and (index(\"fv-script-verify-weights\") != null) and (index(\"fv-sha256\") != null) and (index(\"fv-config\") != null)" "$0" >/dev/null' "$I"
check "up: the config is gcp-h3-turbo.toml" bash -c '[[ "$(jq -r ".metadata.items[] | select(.key == \"fv-config\") | .value" "$0")" == "$(cat "$1")" ]]' "$I" "$GCP/../../configs/serve/gcp-h3-turbo.toml"
check "up: firewall from the source CIDR, tcp 8000+40000, udp 40010, tagged to the VM" bash -c 'jq -e ".firewalls[\"fv-serve-fv-h3-turbo-0101000001\"] | .sourceRanges == [\"198.51.100.7/32\"] and .allowed[0].ports == [\"8000\",\"40000\"] and .allowed[1].ports == [\"40010\"] and .targetTags == [\"fv-h3-turbo-0101000001\"] and (.description | contains(\"fv-owner=fastvideo-rs\"))" "$0" >/dev/null' "$S"

# ---- worker: joins the family Durable Object
run bash "$VM" worker h3-turbo "$IMG"
check "worker without FV_DISPATCH_DO_URL: refused" bash -c '[[ $0 != 0 && $1 == *FV_DISPATCH_DO_URL* ]]' "$RC" "$OUT"
export FV_DISPATCH_DO_URL="https://fv-edge.example.workers.dev"
run bash "$VM" worker h3-turbo "$IMG"
check "worker without FV_INTERNAL_TOKEN: refused" bash -c '[[ $0 != 0 && $1 == *FV_INTERNAL_TOKEN* ]]' "$RC" "$OUT"
( umask 077; printf 'FV_INTERNAL_TOKEN=internal-sentinel-3\nFV_ADMIN_TOKEN=fvadm_sentinel4\nFV_CF_ACCOUNT_ID=acct\nFV_D1_DATABASE_ID=d1\n' >"$T/secrets.env" )
export FV_GCP_SECRETS_FILE="$T/secrets.env"
run bash "$VM" worker ltx "ghcr.io/x/y:latest"
check "worker: refuses an unpinned image" bash -c '[[ $0 != 0 && $1 == *"pin the image digest"* ]]' "$RC" "$OUT"
n0="$(inserts)"
run env -u FV_GCP_SOURCE_CIDR FV_GCP_RUN=0101000002 bash "$VM" worker ltx "$IMG"
S="$(snap)"
W="$(instance "$S" fv-w-ltx-0101000002)"
check "worker: created; prints <vm> <ip> https://<ip>.sslip.io" bash -c '[[ $0 == 0 && $(tail -1 <<<"$1") == "fv-w-ltx-0101000002 127.0.0.1 https://127-0-0-1.sslip.io" ]]' "$RC" "$OUT"
check "worker env: role worker, family ltx, DO URL, direct uploads, capacity, digest" bash -c 'jq -e ".metadata.items[] | select(.key == \"fv-env\") | .value | fromjson | .FV_SERVE_ROLE == \"worker\" and .FV_DISPATCH_FAMILIES == \"ltx\" and .FV_DISPATCH_DO_URL == \"https://fv-edge.example.workers.dev\" and .FV_DISPATCH_DIRECT_UPLOAD == \"1\" and .FV_DISPATCH_CAPACITY == \"2\" and (.FV_IMAGE_DIGEST | startswith(\"sha256:\")) and (has(\"FV_API_KEYS\") | not) and (has(\"FV_WORKER_DIRECT\") | not)" "$0" >/dev/null' "$W"
check "worker: internal token only as a secret item, never in fv-env" bash -c 'jq -e "([.metadata.items[] | select(.key == \"fv-secret-FV_INTERNAL_TOKEN\")] | length == 1) and ([.metadata.items[] | select(.key == \"fv-env\") | .value | contains(\"internal-sentinel\")] | any | not)" "$0" >/dev/null' "$W"
check "worker: cap 14400 s, idle stop, TLS sslip, role label" bash -c 'jq -e "(.scheduling.maxRunDuration.seconds == \"14400\") and .labels[\"fv-role\"] == \"worker\" and ([.metadata.items[] | select(.key == \"fv-tls\") | .value] == [\"sslip\"]) and ([.metadata.items[] | select(.key == \"fv-idle-s\") | .value] == [\"1800\"])" "$0" >/dev/null' "$W"
check "worker firewall: 80/443/40000 + udp from anywhere, not 8000" bash -c 'jq -e ".firewalls[\"fv-serve-fv-w-ltx-0101000002\"] | .sourceRanges == [\"0.0.0.0/0\"] and .allowed[0].ports == [\"80\",\"443\",\"40000\"]" "$0" >/dev/null' "$S"
run env FV_GCP_DIRECT=1 FV_GCP_RUN=0101000003 bash "$VM" worker wan "$IMG"
check "direct worker: needs the D1 key store (FV_CF_API_TOKEN set above, so created)" bash -c '[[ $0 == 0 ]]' "$RC" "$OUT"
check "direct worker env: FV_WORKER_DIRECT, keys auth, D1 key store, family wan" bash -c 'jq -e ".instances[\"fv-w-wan-0101000003\"].metadata.items[] | select(.key == \"fv-env\") | .value | fromjson | .FV_WORKER_DIRECT == \"1\" and .FV_AUTH_MODE == \"keys\" and .FV_KEY_STORE == \"d1\" and .FV_DISPATCH_FAMILIES == \"wan\"" "$0" >/dev/null' "$(snap)"
run env FV_GCP_DIRECT=1 FV_GCP_SECRETS_FILE= FV_INTERNAL_TOKEN=x bash "$VM" worker wan "$IMG"
check "direct worker without FV_ADMIN_TOKEN: refused" bash -c '[[ $0 != 0 && $1 == *FV_ADMIN_TOKEN* ]]' "$RC" "$OUT"
run env FV_GCP_SECRETS=secret-manager FV_GCP_RUN=0101000004 bash "$VM" worker h3-max "$IMG"
check "secret-manager mode: names listed, no values in metadata" bash -c 'jq -e ".instances[\"fv-w-h3-max-0101000004\"].metadata | ([.items[] | select(.key | startswith(\"fv-secret-FV_\"))] | length == 0) and ([.items[] | select(.key == \"fv-secret-names\") | .value | contains(\"FV_INTERNAL_TOKEN\")] | any)" "$0" >/dev/null' "$(snap)"
printf 'AWS_SECRET=x\n' >"$T/bad.env"
run env FV_GCP_SECRETS_FILE="$T/bad.env" bash "$VM" worker h3-max "$IMG"
check "secrets file: an unknown name is refused" bash -c '[[ $0 != 0 && $1 == *"not a secret this script ships"* ]]' "$RC" "$OUT"
unset FV_GCP_SECRETS_FILE FV_DISPATCH_DO_URL
check "four VMs created in total" bash -c '[[ $(( $1 - $0 )) == 3 ]]' "$n0" "$(inserts)"

# ---- down: ours only
run bash "$VM" down fv-h3-turbo-0101000001
S="$(snap)"
check "down: deletes the VM and its firewall rule" bash -c 'jq -e "(.instances | has(\"fv-h3-turbo-0101000001\") | not) and (.firewalls | has(\"fv-serve-fv-h3-turbo-0101000001\") | not)" "$0" >/dev/null' "$S"
seed '{"instances": {"someone-else": {"name": "someone-else", "_zone": "europe-west4-b", "status": "RUNNING", "labels": {"team": "other", "fv-deadline": "1"}, "metadata": {}}}}'
run bash "$VM" down someone-else
check "down: refuses an instance without fv-owner" bash -c '[[ $0 != 0 && $1 == *"refusing"* ]] && jq -e ".instances | has(\"someone-else\")" "$2" >/dev/null' "$RC" "$OUT" "$(snap)"

# ---- reap: past deadline or stopped, ours only
past=$(( $(date +%s) - 60 )) future=$(( $(date +%s) + 3600 ))
seed "$(jq -nc --arg p "$past" --arg f "$future" '{instances: {
  "fv-old": {name: "fv-old", _zone: "europe-west1-c", status: "RUNNING", labels: {"fv-owner": "fastvideo-rs", "fv-deadline": $p}, metadata: {}},
  "fv-stopped": {name: "fv-stopped", _zone: "europe-west4-b", status: "TERMINATED", labels: {"fv-owner": "fastvideo-rs", "fv-deadline": $f}, metadata: {}},
  "fv-live": {name: "fv-live", _zone: "europe-west4-b", status: "RUNNING", labels: {"fv-owner": "fastvideo-rs", "fv-deadline": $f}, metadata: {}}},
  firewalls: {"fv-serve-fv-gone": {name: "fv-serve-fv-gone", description: "fv-owner=fastvideo-rs fv-run=x"},
              "fv-serve-other": {name: "fv-serve-other", description: "someone else"}}}')"
run bash "$VM" reap --dry-run
check "reap --dry-run: lists, deletes nothing" bash -c '[[ $0 == 0 && $1 == *"would delete fv-old"* && $1 == *"would delete fv-stopped"* ]] && jq -e ".instances | has(\"fv-old\") and has(\"fv-stopped\")" "$2" >/dev/null' "$RC" "$OUT" "$(snap)"
run bash "$VM" reap
S="$(snap)"
check "reap: deletes ours past fv-deadline (any zone) and stopped" bash -c 'jq -e "(.instances | has(\"fv-old\") or has(\"fv-stopped\")) | not" "$0" >/dev/null' "$S"
check "reap: keeps ours before the deadline and the foreign VM" bash -c 'jq -e ".instances | has(\"fv-live\") and has(\"someone-else\") and has(\"fv-w-ltx-0101000002\")" "$0" >/dev/null' "$S"
check "reap: removes our orphan firewall rules, keeps foreign and live ones" bash -c 'jq -e ".firewalls | (has(\"fv-serve-fv-gone\") | not) and has(\"fv-serve-other\") and has(\"fv-serve-fv-w-ltx-0101000002\")" "$0" >/dev/null' "$S"

# ---- secrets never leak
check "no token, key or secret value in any output" bash -c 'for s in "$1" "$2" cf-sentinel-secret-1 r2-sentinel-secret-2 internal-sentinel-3 fvadm_sentinel4; do [[ $0 != *"$s"* ]] || { echo "leak: ${s:0:6}"; exit 1; }; done' "$ALL_OUT" "$TOKEN" "$PK_LINE"
check "no token or secret in the ledger" bash -c '! grep -qE "ya29|sentinel" "$0"' "$FV_GCP_LEDGER"
check "ledger records creates, deletes and reaps" bash -c 'grep -q vm-created "$0" && grep -q vm-deleted "$0" && grep -q reaped "$0"' "$FV_GCP_LEDGER"
check "every Compute call carried the bearer token" bash -c 'jq -e "[.requests[] | select(.path | startswith(\"/compute/\")) | .auth] | all and length > 10" "$0" >/dev/null' "$(snap)"
check "token cache is mode 600" bash -c 'f=$(ls "$0"/token-* | head -1); [[ $(stat -c %a "$f") == 600 ]]' "$FV_GCP_STATE"

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
[[ "$FAIL" -eq 0 ]]
