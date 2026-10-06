#!/usr/bin/env bash
# Tests for scripts/gpu/cloudrift.sh and scripts/serve/cloudrift-worker.sh
# against scripts/gpu/tests/cloudrift_mock.py (a fake CloudRift API that also
# answers for the rented fv-serve container). ssh and rsync are stubs. No
# network, no key, nothing rented.
#
#   bash scripts/gpu/tests/cloudrift.test.sh     # needs bash, curl, jq, python3
# shellcheck disable=SC2016 # $vars inside the stubs expand when they run
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="$(cd "$HERE/.." && pwd)"
SERVE="$(cd "$GPU/../serve" && pwd)"
for t in curl jq python3; do command -v "$t" >/dev/null || { echo "cloudrift.test: skipped (no $t)"; exit 0; }; done

T="$(mktemp -d)"
cleanup() { [[ -n "${MOCK_PID:-}" ]] && kill "$MOCK_PID" 2>/dev/null; wait 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
python3 -I "$HERE/cloudrift_mock.py" "$T/port" &
MOCK_PID=$!
for _ in $(seq 50); do [[ -s "$T/port" ]] && break; sleep 0.1; done
M="http://127.0.0.1:$(cat "$T/port")"
KEY=test-cloudrift-key-0123456789

unset CLOUDRIFT_API_KEY
export FV_ENV_FILE=/dev/null CLOUDRIFT_API_BASE="$M" CLOUDRIFT_API_KEY_FILE="$T/no-key"
export CLOUDRIFT_LEDGER="$T/ledger.tsv" CLOUDRIFT_SSH_KEY="$T/id_test" CR_POLL_S=0.2 CLOUDRIFT_CAP_S=600
export CLOUDRIFT_SSH_BIN="$T/bin/ssh" CLOUDRIFT_RSYNC_BIN="$T/bin/rsync" CLOUDRIFT_OUT_DIR="$T/out"

# A stand-in key pair (the stubs never read it).
printf 'not a real key\n' >"$T/id_test"; printf 'ssh-ed25519 AAAATEST fv-cloudrift\n' >"$T/id_test.pub"

# Stubs: ssh answers "the check is done" unless STUB_NOT_DONE is set; rsync
# writes what the container would have written.
mkdir -p "$T/bin"
cat >"$T/bin/ssh" <<'EOF'
#!/usr/bin/env bash
cmd="${*: -1}"
case "$cmd" in
  true) exit 0 ;;
  *"test -f"*) [[ -z "${STUB_NOT_DONE:-}" ]] ;;
  *stat*) echo 100 ;;
  *) exit 0 ;;
esac
EOF
cat >"$T/bin/rsync" <<'EOF'
#!/usr/bin/env bash
dst="${*: -1}"; mkdir -p "$dst"; echo 0 >"$dst/DONE"; echo "== fv-gpucheck nvrtc" >"$dst/run.log"
EOF
chmod +x "$T/bin/ssh" "$T/bin/rsync"

PASS=0 FAIL=0
ok() { PASS=$((PASS + 1)); printf 'ok   %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL %s\n' "$1"; [[ -z "${2:-}" ]] || printf '%s\n' "$2" | sed 's/^/     /' | head -30; }
check() { local name="$1"; shift; if "$@"; then ok "$name"; else bad "$name" "${OUT:-}"; fi; }
state() { curl -sS "$M/__state"; }
seed() { curl -sS -X POST -d "$1" "$M/__seed" >/dev/null; }
rents() { state | jq '.rents | length'; }
ALL_OUT=""
run() { OUT="$("$@" 2>&1)"; RC=$?; ALL_OUT+="$OUT"$'\n'; }

# ---- no key: plan and catalog work, nothing else does
run bash "$GPU/cloudrift.sh" plan
check "gpucheck plan: no key, valid payload" bash -c '[[ $0 == 0 ]] && jq -e ".config.Docker.command[0] == \"bash\"" >/dev/null <<<"$(sed -n "/^{/,/^}/p" <<<"$1")"' "$RC" "$OUT"
run bash "$SERVE/cloudrift-worker.sh" plan
check "worker plan: args mode, entrypoint kept" bash -c '[[ $0 == 0 ]] && jq -e ".config.Docker.command == [\"--config\", \"/etc/fv/runpod-fake.toml\"]" >/dev/null <<<"$(sed -n "/^{/,/^}/p" <<<"$1")"' "$RC" "$OUT"
printf 'FV_CF_API_TOKEN=cf-secret-value-1\nFV_D1_DATABASE_ID=d1-id\nFV_CF_ACCOUNT_ID=acct\n' >"$T/secrets.env"
run env FV_CLOUDRIFT_SECRETS_FILE="$T/secrets.env" FV_PLAN_ROLE=worker CLOUDRIFT_CMD_MODE=exec bash "$SERVE/cloudrift-worker.sh" plan
check "worker plan (exec, worker): secrets masked, pool set" bash -c '[[ $0 == 0 && $1 != *cf-secret-value-1* && $1 == *FV_GATEWAY_POOL* && $1 == *"<secret>"* && $1 == *api.ipify.org* ]]' "$RC" "$OUT"
printf 'AWS_SECRET=x\n' >"$T/bad.env"
run env FV_CLOUDRIFT_SECRETS_FILE="$T/bad.env" bash "$SERVE/cloudrift-worker.sh" plan
check "worker plan: an unknown name in the secrets file is refused" bash -c '[[ $0 != 0 && $1 == *"not a secret this script passes"* ]]' "$RC" "$OUT"
run bash "$GPU/cloudrift.sh" catalog
check "catalog is public (no key) and prices are dollars" bash -c '[[ $0 == 0 && $1 == *"rtx49-t.1"*"0.39"* ]]' "$RC" "$OUT"
run bash "$GPU/cloudrift.sh" balance
check "balance without a key: refused" bash -c '[[ $0 != 0 && $1 == *"no CloudRift API key"* ]]' "$RC" "$OUT"

# ---- the key from the file (mode 600)
( umask 077; printf '%s\n' "$KEY" >"$T/key" )
export CLOUDRIFT_API_KEY_FILE="$T/key"
run bash "$GPU/cloudrift.sh" balance
check "balance with the key file" bash -c '[[ $0 == 0 && $1 == "50"* ]]' "$RC" "$OUT"

# ---- guards before any rent
seed '{"balance": 5}'
run bash "$GPU/cloudrift.sh" smoke
check "balance floor: no rent below \$8" bash -c '[[ $0 != 0 && $1 == *"below the floor"* && $2 == 0 ]]' "$RC" "$OUT" "$(rents)"
seed '{"balance": 50}'
run env CLOUDRIFT_MAX_DPH=0.2 bash "$GPU/cloudrift.sh" smoke
check "price cap: no rent over the cap" bash -c '[[ $0 != 0 && $1 == *"no free 1-GPU stock"* && $2 == 0 ]]' "$RC" "$OUT" "$(rents)"

# ---- gpucheck smoke: rent, ssh, results, terminate
run bash "$GPU/cloudrift.sh" smoke
S="$(state)"
check "smoke: PASS" bash -c '[[ $0 == 0 && $1 == *PASS* ]]' "$RC" "$OUT"
check "smoke: rented the preferred brand with stock, v062 payload" bash -c 'jq -e ".rents[-1] | .selector.ByInstanceTypeAndLocation == {instance_type: \"rtxpro6000-t.1\", datacenters: [\"us-t-1\"]} and (.tags | index(\"fv-owner:fastvideo-rs\")) and (.config.Docker.ports == [\"2222:22/tcp\"])" >/dev/null <<<"$0"' "$S"
check "smoke: the instance was terminated" bash -c 'jq -e "(.terminates | length) == 1 and ([.instances[] | select(.status != \"Inactive\")] | length) == 0" >/dev/null <<<"$0"' "$S"
check "smoke: every keyed call carried X-API-Key, none a bearer" bash -c 'jq -e "[.requests[] | select(.path != \"/api/v1/instance-types/list\" and (.path | startswith(\"/api/\"))) | .key_ok] | all and length > 3" >/dev/null <<<"$0" && jq -e "[.requests[].auth_header] | any | not" >/dev/null <<<"$0"' "$S"
check "smoke: results and summary fetched" bash -c 'f=$(ls "$0"/*/summary.json 2>/dev/null | head -1); [[ -n $f ]] && jq -e ".check_rc == \"0\" and .usd_per_hr == 1.3936" "$f" >/dev/null' "$T/out"

# ---- idle guard
seed '{"gpu_util": 0}'
run env STUB_NOT_DONE=1 CLOUDRIFT_IDLE_MIN=0 bash "$GPU/cloudrift.sh" smoke
check "idle guard: an idle check is terminated and fails" bash -c '[[ $0 != 0 && $1 == *"idle guard"* ]] && jq -e "([.instances[] | select(.status != \"Inactive\")] | length) == 0" >/dev/null <<<"$2"' "$RC" "$OUT" "$(state)"
seed '{"gpu_util": 40}'

# ---- launch: the detached backstop terminates on its own
run env CLOUDRIFT_CAP_S=1 bash "$GPU/cloudrift.sh" launch
LID="$(awk 'END{print $1}' <<<"$OUT")"
sleep 3
check "launch: the wall-clock backstop terminated $LID" bash -c 'jq -e --arg id "$0" ".instances[\$id].status == \"Inactive\"" >/dev/null <<<"$1"' "$LID" "$(state)"
check "launch: the deadline tag is set" bash -c 'jq -e --arg id "$0" ".instances[\$id].tags | map(select(startswith(\"fv-deadline:\"))) | length == 1" >/dev/null <<<"$1"' "$LID" "$(state)"

# ---- worker smoke: the public port, a job, terminate
run bash "$SERVE/cloudrift-worker.sh" smoke
S="$(state)"
check "worker smoke: /healthz on the public port, job succeeded" bash -c '[[ $0 == 0 ]] && jq -e ".job.status == \"succeeded\" and (.public_url | startswith(\"http://127.0.0.1:\"))" >/dev/null <<<"$(sed -n "/^{/,/^}/p" <<<"$1")"' "$RC" "$OUT"
check "worker smoke: terminated; FV_API_KEYS is a hash" bash -c 'jq -e "([.instances[] | select(.status != \"Inactive\")] | length) == 0 and (.rents[-1].config.Docker.env | map(select(.[0] == \"FV_API_KEYS\"))[0][1] | test(\"^[0-9a-f]{64}$\"))" >/dev/null <<<"$0"' "$S"

# ---- worker up: digest pin, internal token, URL for the gateway
run bash "$SERVE/cloudrift-worker.sh" up h3-turbo ghcr.io/x/y:latest
check "worker up: refuses an unpinned image" bash -c '[[ $0 != 0 && $1 == *"pin the image digest"* ]]' "$RC" "$OUT"
( umask 077; printf 'internal-token-abc\n' >"$T/tok" )
run env FV_INTERNAL_TOKEN_FILE="$T/tok" FV_CLOUDRIFT_SECRETS_FILE="$T/secrets.env" bash "$SERVE/cloudrift-worker.sh" up h3-turbo "ghcr.io/x/y@sha256:$(printf 'a%.0s' $(seq 64))"
S="$(state)"
check "worker up: prints the id and the URL for FV_POOL_H3_TURBO_URLS" bash -c '[[ $0 == 0 && $1 == *"FV_POOL_H3_TURBO_URLS"* && $1 == *"http://127.0.0.1:"* && $1 != *internal-token-abc* && $1 != *cf-secret-value-1* ]]' "$RC" "$OUT"
check "worker up: role worker, pool, token in env" bash -c 'jq -e ".rents[-1].config.Docker.env | (map(select(.[0] == \"FV_SERVE_ROLE\"))[0][1] == \"worker\") and (map(select(.[0] == \"FV_GATEWAY_POOL\"))[0][1] == \"h3-turbo\") and (map(select(.[0] == \"FV_INTERNAL_TOKEN\"))[0][1] == \"internal-token-abc\")" >/dev/null <<<"$0"' "$S"

# ---- status and reap touch only our rentals
seed "$(jq -nc --argjson s "$(state)" '{instances: ($s.instances + {"foreign": {id: "foreign", status: "Active", tags: ["someone-else"], host_address: "127.0.0.1", port_mappings: []}})}')"
run bash "$GPU/cloudrift.sh" reap
S="$(state)"
check "reap: terminates ours, never the foreign rental" bash -c 'jq -e ".instances.foreign.status == \"Active\" and ([.instances[] | select(.id != \"foreign\" and .status != \"Inactive\")] | length) == 0" >/dev/null <<<"$0"' "$S"

# ---- the key never leaks
check "the key is in no output and no ledger line" bash -c '[[ $0 != *"$1"* ]] && ! grep -q "$1" "$2"' "$ALL_OUT" "$KEY" "$CLOUDRIFT_LEDGER"
check "ledger records rents and terminations" bash -c 'grep -q instance-rented "$0" && grep -q instance-terminated "$0"' "$CLOUDRIFT_LEDGER"

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
[[ "$FAIL" -eq 0 ]]
