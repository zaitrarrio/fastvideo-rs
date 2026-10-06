#!/usr/bin/env bash
# CloudRift driver for GPU checks and benchmarks (docs/ops/cloudrift.md): the
# fv-gpucheck runtime image in a CloudRift Docker rental, sshd inside it for
# the results, then a guaranteed termination. Mirrors runpod.sh smoke.
#
#   cloudrift.sh plan [image]       print the rent payload (no API call, no key)
#   cloudrift.sh catalog            GPU types, $/hr and free stock (public, no key)
#   cloudrift.sh balance            the account balance
#   cloudrift.sh smoke [image]      rent the cheapest matching GPU, run nvrtc +
#                                   fast kernels, fetch results, terminate
#   cloudrift.sh run [image] -- <fv-gpucheck args>
#                                   the same with one custom step (a benchmark)
#   cloudrift.sh launch [image]     rent only (backstop armed); prints the id
#   cloudrift.sh wait <id>          until the check wrote DONE (idle guard on)
#   cloudrift.sh fetch <id>         rsync the results to artifacts/cloudrift/<id>/
#   cloudrift.sh down <id>          terminate (and confirm)
#   cloudrift.sh status             our rentals (tag fv-owner:fastvideo-rs)
#   cloudrift.sh reap               terminate every rental of ours
#
# Money guards: the balance floor (CLOUDRIFT_MIN_BALANCE, default 8 $; no
# rental below it), a $/hr cap (CLOUDRIFT_MAX_DPH, default 1.5) checked on
# the catalog price before the rent, a detached wall-clock backstop
# (CLOUDRIFT_CAP_S, default 1800 s) that terminates the instance even if this
# shell dies, an fv-deadline:<unix> tag fv-control's collector also enforces,
# an idle guard (GPU under CLOUDRIFT_IDLE_GPU_PCT % with no new output for
# CLOUDRIFT_IDLE_MIN minutes: terminate), the container's own exit after
# FV_IDLE_S, a terminate-on-exit trap, and a ledger
# (artifacts/cloudrift/ledger.tsv).
#
# Env: CLOUDRIFT_API_KEY (or /root/.config/fv/cloudrift_api_key);
# CLOUDRIFT_GPUS (brands, preferred first; default "RTX PRO 6000,RTX 5090,RTX 4090");
# CLOUDRIFT_IMAGE (default the runtime image's :latest; pin a digest);
# CLOUDRIFT_SSH_KEY (default ~/.ssh/id_ed25519_fv_cloudrift, made if missing);
# CLOUDRIFT_API_BASE (tests: the mock in scripts/gpu/tests/).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=cloudrift-lib.sh
source "$HERE/cloudrift-lib.sh"

IMAGE_DEFAULT="${CLOUDRIFT_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-runtime:latest}"
GPUS="${CLOUDRIFT_GPUS:-RTX PRO 6000,RTX 5090,RTX 4090}"
MAX_DPH="${CLOUDRIFT_MAX_DPH:-1.5}"
CAP_S="${CLOUDRIFT_CAP_S:-1800}"
BOOT_WAIT_S="${CLOUDRIFT_BOOT_WAIT_S:-900}"
IDLE_PCT="${CLOUDRIFT_IDLE_GPU_PCT:-5}"
IDLE_MIN="${CLOUDRIFT_IDLE_MIN:-15}"
STEP_TIMEOUT_S="${CLOUDRIFT_STEP_TIMEOUT_S:-600}"
CONTAINER_IDLE_S="${FV_IDLE_S:-1200}"
SSH_KEY="${CLOUDRIFT_SSH_KEY:-$HOME/.ssh/id_ed25519_fv_cloudrift}"
SSH_BIN="${CLOUDRIFT_SSH_BIN:-ssh}"
RSYNC_BIN="${CLOUDRIFT_RSYNC_BIN:-rsync}"
OUT_ROOT="${CLOUDRIFT_OUT_DIR:-$FV_ROOT/artifacts/cloudrift}"
REMOTE_OUT=/workspace/gpucheck-out
SSH_PORT_REQ="${CLOUDRIFT_SSH_HOST_PORT:-2222}"

# The container's command. The runtime image has no ENTRYPOINT, so this runs
# as `bash -c` whether CloudRift's `command` replaces the entrypoint or only
# the CMD (docs/ops/cloudrift.md, UNVERIFIED item 1). FV_STEPS_B64: one
# fv-gpucheck argument line per step.
# shellcheck disable=SC2016 # expanded inside the container
BOOT='set -u
out=/workspace/gpucheck-out
mkdir -p "$out" /root/.ssh /run/sshd
printf "%s\n" "$FV_SSH_PUBKEY" > /root/.ssh/authorized_keys
chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
/usr/sbin/sshd -p 22 -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password || echo "sshd failed" >> "$out/boot.log"
date -u +%FT%TZ > "$out/started"
export PATH=/opt/fastvideo-rs/target/release:/usr/local/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.4/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
{ nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; } > "$out/nvidia-smi.txt" 2>&1
cat /opt/fastvideo-rs/target/release/fv-gpucheck.build-id > "$out/build-id" 2>/dev/null || true
printf "%s" "$FV_STEPS_B64" | base64 -d > /tmp/fv-steps
rc=0
while IFS= read -r step; do
  [ -n "$step" ] || continue
  echo "== fv-gpucheck $step" >> "$out/run.log"
  # shellcheck disable=SC2086 # a step is an argument line
  timeout --signal=TERM --kill-after=15 "$FV_STEP_TIMEOUT_S" fv-gpucheck --out "$out" $step >> "$out/run.log" 2>&1 || { rc=$?; break; }
done < /tmp/fv-steps
echo "$rc" > "$out/DONE"
sleep "$FV_IDLE_S"'

ssh_pub() {
  if [[ ! -f "$SSH_KEY" ]]; then
    require_tools ssh-keygen
    mkdir -p "$(dirname "$SSH_KEY")"
    ssh-keygen -q -t ed25519 -N '' -C fv-cloudrift -f "$SSH_KEY" >/dev/null
  fi
  if [[ -f "$SSH_KEY.pub" ]]; then cat "$SSH_KEY.pub"; else ssh-keygen -y -f "$SSH_KEY"; fi
}

# payload <image> <variant> <dc> <steps (newline-separated)> <pubkey> <deadline>
payload() {
  local env
  env="$(jq -nc --arg k "$5" --arg s "$(printf '%s' "$4" | base64 | tr -d '\n')" --arg t "$STEP_TIMEOUT_S" --arg i "$CONTAINER_IDLE_S" \
    '{FV_SSH_PUBKEY: $k, FV_STEPS_B64: $s, FV_STEP_TIMEOUT_S: $t, FV_IDLE_S: $i, NVIDIA_DRIVER_CAPABILITIES: "compute,utility"}')"
  cr_docker_payload "$2" "$3" "fv-gpucheck-$(date -u +%m%d%H%M%S)" "$1" "$(jq -nc --arg b "$BOOT" '["bash", "-c", $b]')" \
    "$env" "$(jq -nc --arg p "$SSH_PORT_REQ" '[($p + ":22/tcp")]')" "$(cr_tags gpucheck "$6")"
}

# Shape checks (the spec's RentInstanceRequest; v062).
validate() {
  local p="$1"
  jq -e '.selector.ByInstanceTypeAndLocation.instance_type | type == "string" and length > 0' <<<"$p" >/dev/null || die "no instance type"
  jq -e '.with_public_ip == true' <<<"$p" >/dev/null || die "with_public_ip must be true (the ssh port)"
  jq -e '.config.Docker.env | all(type == "array" and length == 2)' <<<"$p" >/dev/null || die "env must be [name, value] pairs"
  jq -e '.config.Docker.ports | all(test("^[0-9]+:[0-9]+/(tcp|udp|sctp)$"))' <<<"$p" >/dev/null || die "ports must be <host>:<container>/<proto>"
  jq -e '.tags | index("fv-owner:fastvideo-rs")' <<<"$p" >/dev/null || die "the owner tag is missing"
  if jq -e '.config.Docker.env[] | select(.[0] | test("KEY$|TOKEN|SECRET|PASSWORD")) | select(.[0] != "FV_SSH_PUBKEY")' <<<"$p" >/dev/null; then
    die "a secret-looking name is in the env"
  fi
  bash -n <<<"$BOOT" || die "the boot command is not valid bash"
  [[ "$(jq -r .config.Docker.image <<<"$p")" == *@sha256:* ]] || log "warning: the image is not digest-pinned"
}

ID=""
cleanup() {
  local rc=$?
  if [[ -n "$ID" ]]; then
    log "terminate-on-exit: $ID"
    cr_terminate "$ID" || log "WARNING: the ${CAP_S}s backstop will retry $ID"
    ID=""
  fi
  exit "$rc"
}

ssh_to() { # host port cmd...
  local host="$1" port="$2"; shift 2
  "$SSH_BIN" -n -i "$SSH_KEY" -p "$port" -o IdentitiesOnly=yes "${FV_SSH_OPTS[@]}" "root@$host" "$@"
}

# launch <image> <steps>: sets ID, VARIANT and DPH (no subshell: the exit
# trap must see ID).
VARIANT="" DPH=""
launch() {
  local image="$1" steps="$2" pick dc deadline pub p
  cr_check_balance
  pick="$(cr_pick "$GPUS" "$MAX_DPH")" || die "no free 1-GPU stock of [$GPUS] under \$$MAX_DPH/hr (cloudrift.sh catalog)"
  read -r VARIANT DPH dc <<<"$pick"
  deadline=$(($(date +%s) + CAP_S))
  pub="$(ssh_pub)"
  p="$(payload "$image" "$VARIANT" "${dc:-}" "$steps" "$pub" "$deadline")"
  validate "$p"
  ID="$(cr_rent "$p")" || die "rent failed ($VARIANT${dc:+ in $dc})"
  cr_ledger "instance-rented $ID variant=$VARIANT dph=$DPH dc=${dc:-any} image=$image cap=${CAP_S}s"
  cr_backstop "$ID" "$CAP_S"
  log "instance $ID: $VARIANT${dc:+ in $dc} at \$$DPH/hr (backstop ${CAP_S}s, deadline $deadline)"
}

# target <id> -> "host port" once Active and sshd answers
target() {
  local id="$1" inst host port t0
  inst="$(cr_wait_active "$id" "$BOOT_WAIT_S")" || die "instance $id did not become Active"
  host="$(jq -r '.host_address' <<<"$inst")"
  port="$(cr_host_port "$inst" 22)"
  [[ "$port" != 22 ]] || port="$SSH_PORT_REQ"
  t0="$(date +%s)"
  until ssh_to "$host" "$port" true 2>/dev/null; do
    (( $(date +%s) - t0 < BOOT_WAIT_S )) || die "no ssh on $host:$port after ${BOOT_WAIT_S}s (image pull?)"
    sleep "${CR_POLL_S:-10}"
  done
  printf '%s %s\n' "$host" "$port"
}

# wait_done <id> <host> <port>: the idle guard runs while waiting.
wait_done() {
  local id="$1" host="$2" port="$3" size last_size=-1 idle_since="" util t0 now
  t0="$(date +%s)"
  while :; do
    if ssh_to "$host" "$port" "test -f $REMOTE_OUT/DONE" 2>/dev/null; then return 0; fi
    now="$(date +%s)"
    size="$(ssh_to "$host" "$port" "stat -c %s $REMOTE_OUT/run.log 2>/dev/null || echo 0" 2>/dev/null || echo 0)"
    util="$(cr_gpu_util "$id" || true)"
    if [[ "$size" == "$last_size" ]] && [[ -z "$util" ]] || { [[ "$size" == "$last_size" ]] && awk -v u="$util" -v m="$IDLE_PCT" 'BEGIN{exit !(u+0 < m+0)}'; }; then
      idle_since="${idle_since:-$now}"
      if (( now - idle_since >= IDLE_MIN * 60 )); then
        log "idle guard: no new output and GPU ${util:-?}% for ${IDLE_MIN} min: terminating $id"
        return 3
      fi
    else
      idle_since=""
    fi
    last_size="$size"
    (( now - t0 < CAP_S )) || { log "the check is still running at the ${CAP_S}s cap"; return 4; }
    sleep "${CR_POLL_S:-15}"
  done
}

fetch() {
  local id="$1" host="$2" port="$3" dst="$OUT_ROOT/$1"
  mkdir -p "$dst"
  "$RSYNC_BIN" -az -e "$SSH_BIN -i $SSH_KEY -p $port -o IdentitiesOnly=yes ${FV_SSH_OPTS[*]}" "root@$host:$REMOTE_OUT/" "$dst/"
  log "results in ${dst#"$FV_ROOT"/}"
}

run_check() {
  local image="$1" steps="$2" host port hp rc=0 t0 t_ssh
  require_tools curl jq base64
  cr_need_key
  t0="$(date +%s)"
  trap cleanup EXIT INT TERM
  launch "$image" "$steps"
  hp="$(target "$ID")" || exit 2
  read -r host port <<<"$hp"
  t_ssh="$(date +%s)"
  log "ssh root@$host:$port $((t_ssh - t0))s after the rent"
  wait_done "$ID" "$host" "$port" || rc=$?
  fetch "$ID" "$host" "$port" || log "WARNING: fetch failed"
  local done_rc
  done_rc="$(cat "$OUT_ROOT/$ID/DONE" 2>/dev/null || echo "?")"
  mkdir -p "$OUT_ROOT/$ID"
  jq -n --arg id "$ID" --arg v "$VARIANT" --argjson dph "$DPH" --arg image "$image" --argjson boot "$((t_ssh - t0))" \
    --argjson wall "$(($(date +%s) - t0))" --arg rc "$done_rc" '{instance: $id, variant: $v, usd_per_hr: $dph,
      image: $image, rent_to_ssh_s: $boot, wall_s: $wall, check_rc: $rc, est_cost_usd: ($dph * $wall / 3600)}' \
    | tee "$OUT_ROOT/$ID/summary.json"
  cr_terminate "$ID" && ID=""
  [[ "$rc" -eq 0 && "$done_rc" == 0 ]] || die "check failed (wait rc $rc, check rc $done_rc)"
  log "PASS"
}

SMOKE_STEPS=$'nvrtc\n--mode fast kernels'

case "${1:-}" in
  plan)
    shift
    p="$(payload "${1:-$IMAGE_DEFAULT}" "<variant>" "" "$SMOKE_STEPS" "ssh-ed25519 AAAA<public key> fv-cloudrift" "$(($(date +%s) + CAP_S))")"
    validate "$p"
    echo "# POST $CR_API/api/v1/instances/rent  (version $CR_VERSION)"
    jq . <<<"$p"
    log "plan: payload shape OK" ;;
  catalog)
    printf 'variant\tgpu\tgpus\tvram_gb\tusd_hr\tfree_nodes\tdatacenters\n'
    cr_catalog docker | awk -F'\t' '$3 > 0' ;;
  balance) cr_need_key; echo "$(cr_balance) USD" ;;
  smoke) shift; run_check "${1:-$IMAGE_DEFAULT}" "$SMOKE_STEPS" ;;
  run)
    shift
    image="$IMAGE_DEFAULT"
    if [[ "${1:-}" != "--" && -n "${1:-}" ]]; then image="$1"; shift; fi
    [[ "${1:-}" == "--" ]] || die "usage: cloudrift.sh run [image] -- <fv-gpucheck args>"
    shift
    (($#)) || die "no fv-gpucheck arguments"
    run_check "$image" "$*" ;;
  launch)
    shift; cr_need_key; require_tools curl jq base64
    launch "${1:-$IMAGE_DEFAULT}" "$SMOKE_STEPS"; echo "$ID $VARIANT $DPH"; ID="" ;;
  wait)
    cr_need_key; id="${2:?instance id}"
    hp="$(target "$id")" || exit 2
    read -r host port <<<"$hp"
    wait_done "$id" "$host" "$port" ;;
  fetch)
    cr_need_key; id="${2:?instance id}"
    hp="$(target "$id")" || exit 2
    read -r host port <<<"$hp"
    fetch "$id" "$host" "$port" ;;
  down) cr_need_key; cr_terminate "${2:?instance id}" && echo "terminated $2" ;;
  status)
    cr_need_key
    cr_post instances/list "$(jq -nc --arg o "$CR_OWNER_TAG" '{selector: {ByTags: {all: [$o]}}, mask: {with_connection_info: true, with_usage_info: true}}')" \
      | jq -r '.instances[] | [.id, .status, (.instance_name // "-"), (.resource_info.instance_type // "-"), (.resource_info.cost_per_hour // 0), (.host_address // "-"), ((.tags // []) | join(","))] | @tsv' ;;
  reap)
    cr_need_key
    for id in $(cr_post instances/list "$(jq -nc --arg o "$CR_OWNER_TAG" '{selector: {ByTags: {all: [$o]}}}')" \
      | jq -r '.instances[] | select(.status == "Active" or .status == "Initializing" or .status == "Failed") | .id'); do
      cr_terminate "$id" && echo "terminated $id"
    done ;;
  *) sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
