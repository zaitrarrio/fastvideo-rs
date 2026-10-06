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
export CLOUDRIFT_KNOWN_HOSTS="$T/known_hosts" STUB_PUBLIC_URL="https://stub.trycloudflare.test" CLOUDRIFT_BOOT_WAIT_S=20

# A stand-in key pair (the stubs never read it).
printf 'not a real key\n' >"$T/id_test"; printf 'ssh-ed25519 AAAATEST fv-cloudrift\n' >"$T/id_test.pub"

# Stubs: ssh answers "the check is done" unless STUB_NOT_DONE is set; rsync
# writes what the container would have written.
mkdir -p "$T/bin"
cat >"$T/bin/ssh" <<'EOF'
#!/usr/bin/env bash
# The worker's VM over SSH (the ssh / tunnel fallbacks): /var/lib/fv/public-url
# answers STUB_PUBLIC_URL.
cmd="$*"
case "$cmd" in
  *" true") exit 0 ;;
  *"-L "*) exit 0 ;;
  *public-url*) echo "${STUB_PUBLIC_URL:-}" ;;
  *nvidia-smi.txt*) echo "NVIDIA-SMI 580.95 (stub)" ;;
  *"/var/lib/fv/booted"*) exit 0 ;;
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
check "gpucheck plan: no key, valid payload" bash -c '[[ $0 == 0 ]] && jq -e ".config.Docker.command[0] == \"bash\"" >/dev/null <<<"$(sed -n "/^{\$/,/^}\$/p" <<<"$1")"' "$RC" "$OUT"
printf 'FV_CF_API_TOKEN=cf-secret-value-1\nFV_D1_DATABASE_ID=d1-id\nFV_CF_ACCOUNT_ID=acct\n' >"$T/secrets.env"
run env FV_CLOUDRIFT_SECRETS_FILE="$T/secrets.env" FV_PLAN_ROLE=worker FV_DISPATCH_DO_URL=https://edge.example.test FV_PLAN_SHOW_BOOT=1 bash "$SERVE/cloudrift-worker.sh" plan
check "worker plan (default): VM, outbound only (no Caddy, no tunnel, sessions 0), loopback publish, nothing secret shown" bash -c '[[ $0 == 0 && $1 == *VirtualMachine* && $1 == *"inbound none"* && $1 == *"FVB_INBOUND=none"* && $1 == *"-p 127.0.0.1:8000:8000"* && $1 != *cf-secret-value-1* && $1 != *ssh_key* ]]' "$RC" "$OUT"
check "worker plan: the decoded boot is valid bash, curl-free IP lookup" bash -c 'b="$(sed -n "/^# \/root\/fv-boot.sh/,\$p" <<<"$0" | sed 1d)"; [[ -n $b && $b == */dev/tcp/api.ipify.org/80* ]] && bash -n <<<"$b"' "$OUT"
run env FV_PLAN_ROLE=worker CLOUDRIFT_SERVICE=docker FV_DISPATCH_DO_URL=https://edge.example.test bash "$SERVE/cloudrift-worker.sh" plan
check "worker plan (docker): no published port, entrypoint kept, family DO settings" bash -c '[[ $0 == 0 ]] && jq -e ".config.Docker | (.ports == []) and (.command == [\"--config\", \"/etc/fv/runpod-fake.toml\"]) and (.env | map(select(.[0] == \"FV_DISPATCH_SESSIONS\"))[0][1] == \"0\") and (.env | map(select(.[0] == \"FV_INTERNAL_TOKEN\"))[0][1] == \"<secret>\")" >/dev/null <<<"$(sed -n "/^{\$/,/^}\$/p" <<<"$1")"' "$RC" "$OUT"
run env FV_PLAN_SHOW_BOOT=1 bash "$SERVE/cloudrift-worker.sh" plan
check "smoke plan (default): HTTPS on the VM (Caddy, sslip.io), never plain HTTP" bash -c '[[ $0 == 0 && $1 == *"FVB_INBOUND=https"* && $1 == *"caddy reverse-proxy"* && $1 == *sslip.io* && $1 == *"FV_PUBLIC_BASE_URL=\$url"* && $1 != *"FV_PUBLIC_BASE_URL=http:"* ]]' "$RC" "$OUT"
printf 'AWS_SECRET=x\n' >"$T/bad.env"
run env FV_CLOUDRIFT_SECRETS_FILE="$T/bad.env" bash "$SERVE/cloudrift-worker.sh" plan
check "worker plan: an unknown name in the secrets file is refused" bash -c '[[ $0 != 0 && $1 == *"not a secret this script passes"* ]]' "$RC" "$OUT"
run bash "$GPU/cloudrift.sh" catalog
check "catalog is public (no key), dollars, allowed types only (no RTX 4090, no V100)" bash -c '[[ $0 == 0 && $1 == *"rtxpro6000-t.1"*"1.3936"* && $1 == *"rtx59-t.1"* && $1 != *rtx49-t* && $1 != *v100-t* ]]' "$RC" "$OUT"
run bash "$GPU/cloudrift.sh" balance
check "balance without a key: refused" bash -c '[[ $0 != 0 && $1 == *"no CloudRift API key"* ]]' "$RC" "$OUT"

# ---- the key from the file (mode 600)
( umask 077; printf '%s\n' "$KEY" >"$T/key" )
export CLOUDRIFT_API_KEY_FILE="$T/key"
run bash "$GPU/cloudrift.sh" balance
check "balance with the key file: cents read as dollars (5000 -> 50)" bash -c '[[ $0 == 0 && $1 == "50 USD" ]]' "$RC" "$OUT"

# ---- guards before any rent
seed '{"balance": 500}'
run bash "$GPU/cloudrift.sh" smoke
check "balance floor: no rent below \$8 (500 cents = \$5)" bash -c '[[ $0 != 0 && $1 == *"below the floor"* && $2 == 0 ]]' "$RC" "$OUT" "$(rents)"
seed '{"balance": 5000}'
run env CLOUDRIFT_MAX_DPH=0.2 bash "$GPU/cloudrift.sh" smoke
check "price cap: no rent over the cap" bash -c '[[ $0 != 0 && $1 == *"no free 1-GPU stock"* && $2 == 0 ]]' "$RC" "$OUT" "$(rents)"
for g in "V100 SXM2" "RTX 4090"; do
  run env CLOUDRIFT_GPUS="$g" bash "$SERVE/cloudrift-worker.sh" smoke
  check "allow-list: $g refused (in stock, cheap) before any rent" bash -c '[[ $0 != 0 && $1 == *"is not allowed: only RTX PRO 6000,RTX 5090"* && $2 == 0 ]]' "$RC" "$OUT" "$(rents)"
  run env CLOUDRIFT_GPUS="$g" bash "$GPU/cloudrift.sh" smoke
  check "allow-list (gpucheck): $g refused before any rent" bash -c '[[ $0 != 0 && $1 == *"is not allowed"* && $2 == 0 ]]' "$RC" "$OUT" "$(rents)"
done
check "allow-list: cr_rent never sends a refused type" bash -c 'source "$0/cloudrift-lib.sh"; cr_load_key; ! cr_rent "{\"selector\":{\"ByInstanceTypeAndLocation\":{\"instance_type\":\"v100-t.1\"}}}" 2>/dev/null' "$GPU"
check "allow-list: nothing was rented by the refusals" bash -c '[[ $(jq ".rents | length" <<<"$0") == 0 ]]' "$(state)"

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

# ---- worker smoke (default): VM, HTTPS on the VM, a job, terminate
run env CLOUDRIFT_TEST_BASE_URL="$M" bash "$SERVE/cloudrift-worker.sh" smoke
S="$(state)"
check "smoke: HTTPS URL on sslip.io for the VM's address, /health, job succeeded" bash -c '[[ $0 == 0 && $1 == *"public URL https://127-0-0-1.sslip.io"* ]] && jq -e ".service == \"vm\" and .inbound == \"https\" and .job.status == \"succeeded\" and .health.state == \"AVAILABLE\"" >/dev/null <<<"$(sed -n "/^{\$/,/^}\$/p" <<<"$1")"' "$RC" "$OUT"
check "smoke: open-driver recipe on an open host, no ssh key, terminated" bash -c 'jq -e ".rents[-1] | (.selector.ByInstanceTypeAndLocation.instance_type == \"rtxpro6000-t.1\") and (.config.VirtualMachine.image_url == \"https://img.test/u24-open.img\") and (.config.VirtualMachine.ssh_key == null) and (.config.Docker == null)" >/dev/null <<<"$0" && jq -e "([.instances[] | select(.status != \"Inactive\")] | length) == 0" >/dev/null <<<"$0"' "$S"
check "smoke: the run key reaches the VM only as its hash" bash -c 'env=$(jq -r ".rents[-1].config.VirtualMachine.cloudinit_commands" <<<"$0" | sed -n "s/^umask 077; echo \([^ ]*\) | base64.*/\1/p" | base64 -d | sed -n "s/^FVB_ENV_B64=//p" | base64 -d); grep -Eq "^FV_API_KEYS=[0-9a-f]{64}$" <<<"$env"' "$S"
run env CLOUDRIFT_TEST_BASE_URL="$M" CLOUDRIFT_TLS_HOSTNAME=w1.example.test bash "$SERVE/cloudrift-worker.sh" smoke
check "smoke: our own TLS hostname when configured" bash -c '[[ $0 == 0 && $1 == *"public URL https://w1.example.test"* ]]' "$RC" "$OUT"
check "recipe per host: proprietary for ProprietaryOnly, newest open otherwise" bash -c 'source "$0/cloudrift-lib.sh"; cr_load_key; [[ "$(cr_recipe_image ProprietaryOnly)" == https://img.test/u24-proprietary.img && "$(cr_recipe_image OpenAndProprietary)" == https://img.test/u24-open.img ]]' "$GPU"
run env CLOUDRIFT_INBOUND=ssh CLOUDRIFT_TEST_BASE_URL="$M" CLOUDRIFT_GPUS="RTX 5090" bash "$SERVE/cloudrift-worker.sh" smoke
check "smoke (ssh fallback): RTX 5090, ssh key sent only for this mode" bash -c '[[ $0 == 0 ]] && jq -e ".rents[-1] | (.selector.ByInstanceTypeAndLocation.instance_type == \"rtx59-t.1\") and (.config.VirtualMachine.ssh_key.PublicKeys | length == 1)" >/dev/null <<<"$2"' "$RC" "$OUT" "$(state)"
run env CLOUDRIFT_SERVICE=docker bash "$SERVE/cloudrift-worker.sh" smoke
check "smoke (docker): refused, no host for TLS (never plain HTTP)" bash -c '[[ $0 != 0 && $1 == *"no host to terminate TLS on"* ]]' "$RC" "$OUT"

# ---- Docker rentals failing on the platform (as seen live)
seed '{"docker_fails": true}'
( umask 077; printf 'internal-token-abc\n' >"$T/tok"; printf 'cf-tunnel-token-xyz\n' >"$T/ttok" )
PIN="ghcr.io/x/y@sha256:$(printf 'a%.0s' $(seq 64))"
run env FV_INTERNAL_TOKEN_FILE="$T/tok" FV_DISPATCH_DO_URL=https://edge.example.test CLOUDRIFT_SERVICE=docker bash "$SERVE/cloudrift-worker.sh" up wan "$PIN"
check "docker worker: a platform failure is reported and the rental dismissed" bash -c '[[ $0 != 0 && $1 == *"Internal provisioning error"* ]] && jq -e "([.instances[] | select(.status != \"Inactive\")] | length) == 0" >/dev/null <<<"$2"' "$RC" "$OUT" "$(state)"
seed '{"docker_fails": false}'

# ---- worker up: digest pin, family DO URL, outbound only by default
run bash "$SERVE/cloudrift-worker.sh" up wan ghcr.io/x/y:latest
check "worker up: refuses an unpinned image" bash -c '[[ $0 != 0 && $1 == *"pin the image digest"* ]]' "$RC" "$OUT"
run env FV_INTERNAL_TOKEN_FILE="$T/tok" bash "$SERVE/cloudrift-worker.sh" up wan "$PIN"
check "worker up: needs FV_DISPATCH_DO_URL" bash -c '[[ $0 != 0 && $1 == *FV_DISPATCH_DO_URL* ]]' "$RC" "$OUT"
run env FV_INTERNAL_TOKEN_FILE="$T/tok" FV_DISPATCH_DO_URL=http://edge.example.test bash "$SERVE/cloudrift-worker.sh" up wan "$PIN"
check "worker up: an http:// DO URL is refused" bash -c '[[ $0 != 0 && $1 == *"must be https://"* ]]' "$RC" "$OUT"
run env FV_INTERNAL_TOKEN_FILE="$T/tok" FV_CLOUDRIFT_SECRETS_FILE="$T/secrets.env" FV_DISPATCH_DO_URL=https://edge.example.test bash "$SERVE/cloudrift-worker.sh" up wan "$PIN"
S="$(state)"
check "worker up (default): outbound only, no secret printed" bash -c '[[ $0 == 0 && $1 == *"outbound-only"* && $1 == *"https://edge.example.test"* && $1 != *internal-token-abc* && $1 != *cf-secret-value-1* ]]' "$RC" "$OUT"
check "worker up (default): family DO env, sessions 0, no public URL, no Caddy" bash -c 'boot=$(jq -r ".rents[-1].config.VirtualMachine.cloudinit_commands" <<<"$0" | sed -n "s/^umask 077; echo \([^ ]*\) | base64.*/\1/p" | base64 -d); env=$(sed -n "s/^FVB_ENV_B64=//p" <<<"$boot" | base64 -d); grep -qx FV_SERVE_ROLE=worker <<<"$env" && grep -qx FV_DISPATCH_DO_URL=https://edge.example.test <<<"$env" && grep -qx FV_DISPATCH_FAMILIES=wan <<<"$env" && grep -qx FV_DISPATCH_DIRECT_UPLOAD=1 <<<"$env" && grep -qx FV_DISPATCH_SESSIONS=0 <<<"$env" && grep -qx FV_INTERNAL_TOKEN=internal-token-abc <<<"$env" && ! grep -q FV_PUBLIC_BASE_URL <<<"$env" && grep -qx FVB_INBOUND=none <<<"$boot"' "$S"
run env FV_INTERNAL_TOKEN_FILE="$T/tok" FV_DISPATCH_DO_URL=https://edge.example.test CLOUDRIFT_INBOUND=https bash "$SERVE/cloudrift-worker.sh" up wan "$PIN"
check "worker up (https): sessions on, the sslip.io HTTPS endpoint printed" bash -c '[[ $0 == 0 && $1 == *"https://127-0-0-1.sslip.io"* ]] && boot=$(jq -r ".rents[-1].config.VirtualMachine.cloudinit_commands" <<<"$2" | sed -n "s/^umask 077; echo \([^ ]*\) | base64.*/\1/p" | base64 -d) && grep -qx FVB_INBOUND=https <<<"$boot" && sed -n "s/^FVB_ENV_B64=//p" <<<"$boot" | base64 -d | grep -qx FV_DISPATCH_SESSIONS=1' "$RC" "$OUT" "$(state)"
run env FV_INTERNAL_TOKEN_FILE="$T/tok" FV_DISPATCH_DO_URL=https://edge.example.test CLOUDRIFT_INBOUND=tunnel-token bash "$SERVE/cloudrift-worker.sh" up wan "$PIN"
check "worker up (tunnel-token fallback): needs the token file and hostname" bash -c '[[ $0 != 0 && $1 == *FV_CF_TUNNEL_TOKEN_FILE* ]]' "$RC" "$OUT"
run env FV_INTERNAL_TOKEN_FILE="$T/tok" FV_DISPATCH_DO_URL=https://edge.example.test CLOUDRIFT_INBOUND=tunnel-token FV_CF_TUNNEL_TOKEN_FILE="$T/ttok" CLOUDRIFT_TUNNEL_HOSTNAME=w2.example.test bash "$SERVE/cloudrift-worker.sh" up wan "$PIN"
check "worker up (tunnel-token fallback): token written apart from the env, not printed" bash -c '[[ $0 == 0 && $1 == *"https://w2.example.test"* && $1 != *cf-tunnel-token-xyz* ]] && boot=$(jq -r ".rents[-1].config.VirtualMachine.cloudinit_commands" <<<"$2" | sed -n "s/^umask 077; echo \([^ ]*\) | base64.*/\1/p" | base64 -d) && grep -q "cf-tunnel-token-xyz.*tunnel-token" <<<"$boot" && ! sed -n "s/^FVB_ENV_B64=//p" <<<"$boot" | base64 -d | grep -q cf-tunnel-token-xyz' "$RC" "$OUT" "$(state)"

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
