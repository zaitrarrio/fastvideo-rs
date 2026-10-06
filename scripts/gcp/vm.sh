#!/usr/bin/env bash
# fv-serve on a Google Cloud GPU VM through the Compute Engine REST API (no
# gcloud). docs/serve/deploy-gcp.md; the GCP counterpart of
# scripts/serve/runpod-pod.sh and scripts/serve/cloudrift-worker.sh.
#
#   vm.sh preflight              token, project, Compute API, GPU quotas in
#                                $GCP_REGION, the boot image, the machine types
#                                in $GCP_ZONE, the weight disk
#   vm.sh up <family> [image]    a standalone VM for one family (families.tsv:
#                                h3-turbo h3-max ltx wan fake); prints
#                                "<vm> <ip> <api key>" (the key is per run: only
#                                its SHA-256 goes to the VM)
#   vm.sh worker <family> <image@sha256:...>
#                                a worker that joins the family Durable Object
#                                queue (docs/serve/dispatch-do-family.md), as a
#                                Runpod or CloudRift worker does: needs
#                                FV_DISPATCH_DO_URL and FV_INTERNAL_TOKEN;
#                                FV_GCP_DIRECT=1 adds the gateway-less settings
#                                (FV_WORKER_DIRECT, FV_ADMIN_TOKEN, D1 keys);
#                                prints "<vm> <ip> <public base url>"
#   vm.sh wait <vm> <ip> [url]   wait for /ping 200 on http://<ip>:8000, or on the
#                                base URL `worker` printed; fail early on a FAILED line
#   vm.sh down <vm>              delete the VM and its firewall rule (ours only)
#   vm.sh reap [--dry-run]       delete our VMs past their fv-deadline label or
#                                stopped (TERMINATED), then our orphan firewall
#                                rules; never touches anything without
#                                fv-owner=fastvideo-rs
#   vm.sh list                   our VMs, disks, images and firewall rules
#   vm.sh ssh-free-logs <vm> [--follow] [--banner]
#                                the serial console (startup progress + the
#                                fv-serve log); admin tokens are masked unless
#                                --banner is given
#   vm.sh plan <family> [image]  print the REST payloads (same as FV_GCP_DRY_RUN=1 up;
#                                FV_GCP_ROLE=worker plans a worker)
#   vm.sh secrets-push           copy the FV_* secrets from this environment to
#                                Secret Manager (ids = lower-case names)
#   vm.sh gc                     delete our firewall rules whose VM is gone
#
# Money guards (the Runpod scripts' set, cloud-side where possible):
#   - $/hr cap FV_GCP_MAX_DPH (default 6.0) against the price table in lib.sh;
#   - wall clock FV_GCP_CAP_S (default 5400; worker 14400):
#     scheduling.maxRunDuration with instanceTerminationAction=DELETE, so
#     Compute Engine deletes the VM even if this shell, this container or its
#     host dies; the same deadline is the fv-deadline label `reap` reads;
#   - idle stop FV_GCP_IDLE_S (default 1800; 0 = off): the VM deletes itself
#     after that long at 0 % GPU (startup.sh; it powers off if the delete is
#     refused, and `reap` deletes stopped VMs);
#   - onHostMaintenance=TERMINATE, automaticRestart=false (a GPU VM never
#     comes back on its own);
#   - labels fv-owner=fastvideo-rs, fv-run, fv-kind, fv-family, fv-role,
#     fv-deadline on every VM; down/reap/gc refuse resources without
#     fv-owner=fastvideo-rs;
#   - a ledger (artifacts/gcp/ledger.tsv); e2e.sh adds a delete-on-exit trap.
#
# Env: GCP_SA_KEY_JSON / GCP_SA_KEY_FILE / /root/.config/fv/gcp_sa_key.json,
# GCP_PROJECT, GCP_REGION (europe-west4), GCP_ZONE (europe-west4-b);
# FV_SERVE_IMAGE (ghcr.io/zaitrarrio/fastvideo-rs-serve:latest, pinned to its
# digest; `worker` refuses a tag); FV_GCP_MACHINE or FV_GCP_MACHINE_<FAMILY>
# (default per family, families.tsv); FV_GCP_SPOT=1 (Spot; a3-highgpu-1g is
# always Spot); FV_GCP_SOURCE_CIDR (who may reach the VM; default this
# machine's public IPs as /32 for `up`, 0.0.0.0/0 for `worker`, whose clients
# are the public); FV_GCP_TLS (none | sslip: Caddy on 443 with a Let's Encrypt
# certificate for <ip>.sslip.io; default none for `up`, sslip for `worker`);
# FV_GCP_SECRETS (metadata | secret-manager | none; default metadata when any
# secret is set, else none); FV_GCP_SECRETS_FILE (KEY=VALUE lines of the
# secret names below, read instead of the environment); FV_GCP_WEIGHTS_DISK
# (default fv-weights-hdml-$GCP_ZONE, read-only); FV_GCP_SA_EMAIL (the VMs'
# service account; default fv-vm@$GCP_PROJECT.iam.gserviceaccount.com); FV_GCP_ENCODE_BENCH=1 (VM waits
# for a clip to compare NVENC with x264); FV_GCP_DRY_RUN=1.
# Worker: FV_DISPATCH_DO_URL, FV_DISPATCH_FAMILIES (default the family's,
# families.tsv), FV_DISPATCH_CAPACITY (2), FV_DISPATCH_SESSIONS (1),
# FV_GCP_DIRECT=1 with FV_ADMIN_TOKEN, FV_GCP_AUTH_MODE (keys).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$HERE/lib.sh"

IMAGE_DEFAULT="ghcr.io/zaitrarrio/fastvideo-rs-serve:latest"
MAX_DPH="${FV_GCP_MAX_DPH:-6.0}"
WEIGHTS_DISK="${FV_GCP_WEIGHTS_DISK:-fv-weights-hdml-$ZONE}"
UDP_PORTS="${FV_GCP_UDP_PORTS:-40010}"
VM_ROLE="${FV_GCP_ROLE:-serve}"
# Secrets: never printed, never in a payload on stdout (plan redacts them),
# shipped to the VM through Secret Manager or instance metadata.
SECRET_NAMES=(FV_CF_ACCOUNT_ID FV_CF_API_TOKEN FV_D1_DATABASE_ID FV_R2_BUCKET FV_R2_ENDPOINT
  FV_R2_ACCESS_KEY_ID FV_R2_SECRET_ACCESS_KEY FV_WEBHOOK_ED25519_KEY FV_URL_SIGNING_KEY
  FV_INTERNAL_TOKEN FV_ADMIN_TOKEN)

cap_s() { echo "${FV_GCP_CAP_S:-$([[ "$VM_ROLE" == worker ]] && echo 14400 || echo 5400)}"; }
idle_s() { echo "${FV_GCP_IDLE_S:-1800}"; }
tls_mode() { echo "${FV_GCP_TLS:-$([[ "$VM_ROLE" == worker ]] && echo sslip || echo none)}"; }

family_config() { gcp_family "$1" config; }
family_cells() { gcp_family "$1" verify_cells; }
# FV_GCP_MACHINE_<FAMILY> > FV_GCP_MACHINE > families.tsv. G4 (RTX PRO 6000
# 96 GB, NVENC) by default: one H3 DiT is ~41 GB and LTX-2.5 is 22B. wan
# also fits an L4 (FV_GCP_MACHINE_WAN=g2-standard-16); fake needs no GPU
# memory (g2-standard-8).
family_machine() {
  local var="FV_GCP_MACHINE_${1//-/_}"; var="${var^^}"
  if [[ -n "${!var:-}" ]]; then echo "${!var}"; return; fi
  if [[ -n "${FV_GCP_MACHINE:-}" ]]; then echo "$FV_GCP_MACHINE"; return; fi
  gcp_family "$1" machine
}

# FV_GCP_SECRETS_FILE: KEY=VALUE lines (only SECRET_NAMES and HF_TOKEN),
# exported into this process; values never echoed.
load_secrets_file() {
  local f="${FV_GCP_SECRETS_FILE:-}" line k v ok n
  [[ -n "$f" ]] || return 0
  [[ -r "$f" ]] || die "cannot read FV_GCP_SECRETS_FILE"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    k="${line%%=*}"; v="${line#*=}"; ok=0
    for n in "${SECRET_NAMES[@]}" HF_TOKEN; do [[ "$k" == "$n" ]] && ok=1; done
    (( ok )) || die "FV_GCP_SECRETS_FILE: $k is not a secret this script ships"
    export "$k=$v"
  done <"$f"
}

secrets_mode() {
  if [[ -n "${FV_GCP_SECRETS:-}" ]]; then echo "$FV_GCP_SECRETS"; return; fi
  local n
  for n in "${SECRET_NAMES[@]}"; do [[ -n "${!n:-}" ]] && { echo metadata; return; }; done
  echo none
}

# The VMs' own service account: a separate, low-privilege one (Secret Manager
# accessor on the fv secrets, delete on fv-* instances: docs/serve/deploy-gcp.md
# "Auth"), never the deploy key's account, whose token any process on the VM
# could read from the metadata server.
vm_sa_email() { echo "${FV_GCP_SA_EMAIL:-fv-vm@$PROJECT.iam.gserviceaccount.com}"; }

# The family Durable Object worker settings (docs/serve/dispatch-do-family.md
# §12): the same variables fv-control's workerSystemEnv and
# scripts/serve/edge-gpu-test.sh give a Runpod worker. Checked before spending.
worker_check() {
  local family="$1"
  [[ "$family" != fake ]] || [[ -n "${FV_DISPATCH_FAMILIES:-}" ]] || die "worker fake: set FV_DISPATCH_FAMILIES (the fake config has no family)"
  [[ -n "${FV_DISPATCH_DO_URL:-}" ]] || die "worker: set FV_DISPATCH_DO_URL (the fv-edge Worker's https base URL)"
  [[ "$FV_DISPATCH_DO_URL" == https://* || "$DRY" == 1 || "${FV_GCP_ALLOW_HTTP_DO:-0}" == 1 ]] || die "worker: FV_DISPATCH_DO_URL must be https://"
  [[ -n "${FV_INTERNAL_TOKEN:-}" ]] || die "worker: set FV_INTERNAL_TOKEN (the cluster's internal token; env or FV_GCP_SECRETS_FILE)"
  if [[ "${FV_GCP_DIRECT:-0}" == 1 ]]; then
    [[ -n "${FV_ADMIN_TOKEN:-}" ]] || die "worker FV_GCP_DIRECT=1: set FV_ADMIN_TOKEN (every worker of the cluster shares it)"
    [[ -n "${FV_CF_ACCOUNT_ID:-}" && -n "${FV_CF_API_TOKEN:-}" && -n "${FV_D1_DATABASE_ID:-}" ]] \
      || die "worker FV_GCP_DIRECT=1: the minted API keys live in D1 (FV_CF_ACCOUNT_ID, FV_CF_API_TOKEN, FV_D1_DATABASE_ID)"
  fi
}

# env_json <family> <keyhash>: the container's non-secret environment.
env_json() {
  local family="$1" keyhash="$2" fams
  fams="${FV_DISPATCH_FAMILIES:-$(gcp_family "$family" dispatch_family)}"
  jq -nc --arg role "$VM_ROLE" --arg keys "$keyhash" --arg udp "$UDP_PORTS" --arg fams "$fams" \
    --arg dourl "${FV_DISPATCH_DO_URL:-}" --arg cap "${FV_DISPATCH_CAPACITY:-2}" --arg ses "${FV_DISPATCH_SESSIONS:-1}" \
    --arg direct "${FV_GCP_DIRECT:-0}" --arg auth "${FV_GCP_AUTH_MODE:-keys}" --arg img "${IMAGE_REF:-}" '
    {FV_SERVE_MODE: "http", FV_WEIGHTS: "/workspace/weights", FV_STATE_DIR: "/fvstate",
     RUST_LOG: "info", FV_GCP_UDP_PORTS: $udp, FV_IMAGE_REF: $img}
    + (if $img | test("@sha256:") then {FV_IMAGE_DIGEST: ($img | split("@")[1])} else {} end)
    + (if $role == "worker" then
        {FV_SERVE_ROLE: "worker", FV_DISPATCH_DO_URL: $dourl, FV_DISPATCH_FAMILIES: $fams,
         FV_DISPATCH_DIRECT_UPLOAD: "1", FV_DISPATCH_CAPACITY: $cap, FV_DISPATCH_SESSIONS: $ses,
         FV_JOBS_HEARTBEAT_S: "10"}
        + (if $direct == "1" then {FV_WORKER_DIRECT: "1", FV_AUTH_MODE: $auth, FV_KEY_STORE: "d1"} else {} end)
       else {FV_API_KEYS: $keys, FV_SERVE_FORWARD: "1"} end)'
}

# metadata_items <family> <image> <keyhash> -> JSON array of {key,value}
metadata_items() {
  local family="$1" image="$2" keyhash="$3" cfg mode names=() n
  cfg="$(family_config "$family")"
  mode="$(secrets_mode)"
  for n in "${SECRET_NAMES[@]}"; do
    if [[ "$mode" == secret-manager || -n "${!n:-}" ]]; then names+=("$n"); fi
  done
  {
    jq -nc --rawfile s "$HERE/startup.sh" '{key: "startup-script", value: $s}'
    jq -nc --rawfile s "$FV_ROOT/configs/serve/$cfg" '{key: "fv-config", value: $s}'
    jq -nc --arg v serve '{key: "fv-role", value: $v}'
    jq -nc --arg v "$image" '{key: "fv-image", value: $v}'
    jq -nc --arg v "$(IMAGE_REF="$image" env_json "$family" "$keyhash")" '{key: "fv-env", value: $v}'
    jq -nc --arg v "$(family_cells "$family")" '{key: "fv-verify-cells", value: $v}'
    jq -nc --arg v "$mode" '{key: "fv-secrets-mode", value: $v}'
    jq -nc --arg v "${names[*]:-}" '{key: "fv-secret-names", value: $v}'
    jq -nc --arg v "${FV_GCP_ENCODE_BENCH:-0}" '{key: "fv-encode-bench", value: $v}'
    jq -nc --arg v "$(idle_s)" '{key: "fv-idle-s", value: $v}'
    jq -nc --arg v "$(tls_mode)" '{key: "fv-tls", value: $v}'
    jq -nc --arg v "${FV_GCP_CADDY_IMAGE:-caddy:2}" '{key: "fv-caddy-image", value: $v}'
    # verify-weights.sh runs on the host: the per-variant serve images carry no scripts.
    jq -nc --rawfile s "$FV_ROOT/scripts/gpu/verify-weights.sh" '{key: "fv-script-verify-weights", value: $s}'
    jq -nc --rawfile s "$FV_ROOT/scripts/gpu/verify-safetensors.sh" '{key: "fv-script-verify-safetensors", value: $s}'
    jq -nc --rawfile s "$FV_ROOT/scripts/gpu/weights-manifest.tsv" '{key: "fv-manifest", value: $s}'
    jq -nc --rawfile s "$FV_ROOT/scripts/gpu/weights-sha256.tsv" '{key: "fv-sha256", value: $s}'
    if [[ "$mode" == metadata ]]; then
      for n in ${names[@]+"${names[@]}"}; do jq -nc --arg k "fv-secret-$n" --arg v "${!n}" '{key: $k, value: $v}'; done
    fi
  } | jq -sc .
}

# instance_payload <name> <family> <machine> <image> <keyhash> <run> <deadline>
instance_payload() {
  local name="$1" family="$2" machine="$3" image="$4" keyhash="$5" run="$6" deadline="$7" prov boot disks
  prov="$(gcp_provisioning "$machine")"
  boot="$(gcp_boot_disk_type "$machine")"
  disks="$(jq -nc --arg zone "$ZONE" --arg img "$GCP_IMAGE" --arg bt "$boot" --argjson l "$(gcp_labels serve "$run")" '[{
      boot: true, autoDelete: true, deviceName: "boot",
      initializeParams: {sourceImage: $img, diskSizeGb: "100", diskType: ("zones/" + $zone + "/diskTypes/" + $bt), labels: $l}
    }]')"
  if [[ "$family" != fake ]]; then
    disks="$(jq -c --arg src "projects/$PROJECT/zones/$ZONE/disks/$WEIGHTS_DISK" \
      '. + [{source: $src, mode: "READ_ONLY", autoDelete: false, deviceName: "fv-weights"}]' <<<"$disks")"
  fi
  jq -n --arg name "$name" --arg zone "$ZONE" --arg mt "$machine" --arg prov "$prov" --argjson cap "$(cap_s)" \
    --arg net "projects/$PROJECT/global/networks/$NETWORK" --arg sa "$(vm_sa_email)" \
    --argjson labels "$(gcp_labels "$VM_ROLE" "$run" "$(jq -nc --arg f "$family" --arg r "$VM_ROLE" --arg d "$deadline" '{"fv-family": $f, "fv-role": $r, "fv-deadline": $d}')")" \
    --argjson disks "$disks" --argjson items "$(metadata_items "$family" "$image" "$keyhash")" '{
      name: $name,
      description: "fastvideo-rs fv-serve (fv-owner=fastvideo-rs)",
      machineType: ("zones/" + $zone + "/machineTypes/" + $mt),
      labels: $labels,
      tags: {items: ["fv-serve", $name]},
      scheduling: ({
        onHostMaintenance: "TERMINATE", automaticRestart: false,
        provisioningModel: $prov, instanceTerminationAction: "DELETE",
        maxRunDuration: {seconds: ($cap|tostring)}
      }),
      disks: $disks,
      networkInterfaces: [{
        network: $net, nicType: "GVNIC", stackType: "IPV4_ONLY",
        accessConfigs: [{type: "ONE_TO_ONE_NAT", name: "External NAT", networkTier: "PREMIUM"}]
      }],
      serviceAccounts: [{email: $sa, scopes: ["https://www.googleapis.com/auth/cloud-platform"]}],
      shieldedInstanceConfig: {enableSecureBoot: false, enableVtpm: true, enableIntegrityMonitoring: true},
      metadata: {items: $items}
    }'
}

# TCP ports the firewall rule opens: 8000 (fv-serve) unless TLS fronts it;
# 80 + 443 for Caddy (80: the ACME HTTP-01 challenge); 40000 ICE-TCP.
tcp_ports() {
  if [[ "$(tls_mode)" == sslip ]]; then jq -nc '["80", "443", "40000"]'; else jq -nc '["8000", "40000"]'; fi
}

firewall_payload() {
  local name="$1" run="$2" cidrs="$3"
  jq -n --arg name "fv-serve-$name" --arg tag "$name" --arg run "$run" --argjson src "$cidrs" \
    --arg net "projects/$PROJECT/global/networks/$NETWORK" --arg udp "$UDP_PORTS" --argjson tcp "$(tcp_ports)" '{
      name: $name, network: $net, direction: "INGRESS", priority: 1000,
      description: ("fv-owner=fastvideo-rs fv-run=" + $run + " fv-serve TCP " + ($tcp | join(",")) + ", ICE-UDP " + $udp),
      targetTags: [$tag], sourceRanges: $src,
      allowed: [{IPProtocol: "tcp", ports: $tcp}, {IPProtocol: "udp", ports: [$udp]}]
    }'
}

source_cidrs() {
  if [[ -z "${FV_GCP_SOURCE_CIDR:-}" && "$VM_ROLE" == worker ]]; then echo '["0.0.0.0/0"]'; return; fi
  gcp_source_cidrs
}

# shellcheck disable=SC2034 # read by e2e.sh, which sources this file
VM_NAME="" VM_IP="" VM_KEY="" VM_DPH="" VM_MACHINE="" VM_PROV="" VM_URL=""

# vm_up <family> [image]: sets VM_NAME/VM_IP/VM_KEY/VM_DPH/VM_URL (for e2e.sh,
# which sources this file); prints nothing on stdout.
vm_up() {
  local family="$1" image machine prov dph run cidrs key="" keyhash="" resp deadline
  family_config "$family" >/dev/null || die "unknown family $family ($(gcp_families | tr '\n' ' '))"
  require_tools curl jq openssl sha256sum
  load_secrets_file
  [[ "$VM_ROLE" == serve || "$VM_ROLE" == worker ]] || die "FV_GCP_ROLE must be serve or worker"
  [[ "$VM_ROLE" != worker ]] || worker_check "$family"
  case "$(tls_mode)" in none | sslip) ;; *) die "FV_GCP_TLS must be none or sslip" ;; esac
  gcp_require_project
  machine="$(family_machine "$family")"
  prov="$(gcp_provisioning "$machine")"
  dph="$(gcp_price "$machine" "$prov")"
  [[ -n "$dph" ]] || die "no price for $machine ($prov) in the lib.sh table: add it before spending"
  awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 <= c+0)}' || die "$machine $prov is \$$dph/hr > cap \$$MAX_DPH (FV_GCP_MAX_DPH)"
  image="${2:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}"
  if [[ "$VM_ROLE" == worker && "$image" != *@sha256:* && "$DRY" != 1 ]]; then
    die "worker: pin the image digest (ghcr.io/...@sha256:...), as fv-control does"
  fi
  image="$(gcp_resolve_digest "$image")"
  run="${FV_GCP_RUN:-$(date -u +%m%d%H%M%S)}"
  deadline=$(( $(date +%s) + $(cap_s) ))
  VM_NAME="fv-${family//./-}-$run"
  [[ "$VM_ROLE" == worker ]] && VM_NAME="fv-w-${family//./-}-$run"
  if [[ "$VM_ROLE" == serve ]]; then
    key="fvk-$(openssl rand -hex 16)"
    keyhash="$(printf '%s' "$key" | sha256sum | cut -d' ' -f1)"
  fi
  cidrs="$(source_cidrs)"
  [[ "$(jq length <<<"$cidrs")" -gt 0 ]] || die "no source CIDR (set FV_GCP_SOURCE_CIDR)"
  log "vm $VM_NAME ($VM_ROLE): $machine $prov \$$dph/hr in $ZONE, cap $(cap_s)s (self-delete), idle $(idle_s)s, tls $(tls_mode), image $image, from $(jq -r 'join(",")' <<<"$cidrs")"
  if [[ "$family" != fake && "$DRY" != 1 ]]; then
    resp="$(gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/disks/$WEIGHTS_DISK")" \
      || die "weight disk $WEIGHTS_DISK not found in $ZONE: scripts/gcp/weights.sh disk-up (or set FV_GCP_WEIGHTS_DISK)"
  fi
  resp="$(gce POST "$COMPUTE/projects/$PROJECT/global/firewalls" "$(firewall_payload "$VM_NAME" "$run" "$cidrs")")" \
    || die "firewall create: $(jq -c '.error.message // .' <<<"$resp" 2>/dev/null | head -c 400)"
  gce_wait "$resp" || die "firewall create failed"
  gcp_ledger "firewall-created fv-serve-$VM_NAME"
  if ! resp="$(gce POST "$COMPUTE/projects/$PROJECT/zones/$ZONE/instances" "$(instance_payload "$VM_NAME" "$family" "$machine" "$image" "$keyhash" "$run" "$deadline")")" \
    || ! gce_wait "$resp"; then
    log "instance create failed: $(jq -c '.error.message // .error // empty' <<<"$resp" 2>/dev/null | head -c 600)"
    firewall_delete "$VM_NAME" || true
    return 1
  fi
  gcp_ledger "vm-created $VM_NAME role=$VM_ROLE machine=$machine prov=$prov usd_per_hr=$dph cap_s=$(cap_s) deadline=$deadline family=$family image=$image"
  resp="$(gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/instances/$VM_NAME")"
  VM_IP="$(jq -r '.networkInterfaces[0].accessConfigs[0].natIP // empty' <<<"$resp")"
  if [[ "$(tls_mode)" == sslip ]]; then VM_URL="https://${VM_IP//./-}.sslip.io"; else VM_URL="http://$VM_IP:8000"; fi
  # shellcheck disable=SC2034 # read by e2e.sh
  VM_KEY="$key" VM_DPH="$dph" VM_MACHINE="$machine" VM_PROV="$prov"
  log "vm $VM_NAME is up at $VM_IP ($VM_URL; serial: vm.sh ssh-free-logs $VM_NAME)"
}

firewall_delete() {
  local name="$1" fw resp
  fw="fv-serve-$name"
  resp="$(gce GET "$COMPUTE/projects/$PROJECT/global/firewalls/$fw")" || return 0
  gcp_is_ours "$resp" || { log "firewall $fw is not ours: left alone"; return 1; }
  resp="$(gce DELETE "$COMPUTE/projects/$PROJECT/global/firewalls/$fw")" && gce_wait "$resp" && gcp_ledger "firewall-deleted $fw"
}

# vm_delete <name> [zone]: delete one instance after checking it is ours.
vm_delete() {
  local name="$1" zone="${2:-$ZONE}" resp
  if resp="$(gce GET "$COMPUTE/projects/$PROJECT/zones/$zone/instances/$name")"; then
    gcp_is_ours "$resp" || die "instance $name has no fv-owner=$FV_OWNER_LABEL label: refusing to delete"
    resp="$(gce DELETE "$COMPUTE/projects/$PROJECT/zones/$zone/instances/$name")" && gce_wait "$resp" \
      && gcp_ledger "vm-deleted $name" && log "deleted $name"
  else
    log "instance $name not found (already deleted by its maxRunDuration?)"
  fi
}

vm_down() {
  gcp_require_project
  vm_delete "$1"
  firewall_delete "$1" || true
}

# vm_reap [--dry-run]: the label-based reaper. Deletes our VMs (any zone) past
# fv-deadline, or stopped (TERMINATED / STOPPED: an idle stop that could not
# delete itself), then our firewall rules whose VM is gone. Run it from cron,
# fv-control or by hand; it is idempotent.
vm_reap() {
  local dry=0 now list name zone status deadline n=0
  [[ "${1:-}" == --dry-run ]] && dry=1
  gcp_require_project
  now="$(date +%s)"
  list="$(gce GET "$COMPUTE/projects/$PROJECT/aggregated/instances?filter=labels.fv-owner%3D$FV_OWNER_LABEL")" || die "list failed: $(head -c 300 <<<"$list")"
  while IFS=$'\t' read -r name zone status deadline; do
    [[ -n "$name" ]] || continue
    local why=""
    case "$status" in TERMINATED | STOPPED | SUSPENDED) why="status $status" ;; esac
    if [[ -z "$why" && "$deadline" =~ ^[0-9]+$ ]] && (( deadline < now )); then why="past fv-deadline $(date -u -d "@$deadline" +%FT%TZ)"; fi
    if [[ -z "$why" && -z "$deadline" ]]; then
      log "reap: $name has no fv-deadline label (maxRunDuration still applies): kept"; continue
    fi
    [[ -n "$why" ]] || continue
    if (( dry )); then log "reap (dry run): would delete $name in $zone ($why)"; continue; fi
    log "reap: $name in $zone ($why)"
    vm_delete "$name" "$zone" && n=$((n + 1))
    gcp_ledger "reaped $name ($why)"
  done < <(jq -r --arg o "$FV_OWNER_LABEL" '.items // {} | to_entries[] | .value.instances[]?
      | select((.labels["fv-owner"] // "") == $o)
      | [.name, (.zone | split("/") | last), .status, (.labels["fv-deadline"] // "")] | @tsv' <<<"$list")
  (( dry )) || vm_gc
  log "reap: $n deleted"
}

# serial_read <vm> <start>: prints contents; sets SERIAL_NEXT.
SERIAL_NEXT=0
serial_read() {
  local resp
  resp="$(gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/instances/$1/serialPort?port=1&start=$2")" || return 1
  SERIAL_NEXT="$(jq -r '.next // 0' <<<"$resp")"
  jq -r '.contents // ""' <<<"$resp"
}

mask_tokens() { sed -E 's/fvadm_[A-Za-z0-9_-]+/fvadm_<masked: vm.sh ssh-free-logs --banner>/g'; }

vm_logs() {
  local name="$1" follow=0 banner=0 start=0 a
  shift
  for a in "$@"; do case "$a" in --follow) follow=1 ;; --banner) banner=1 ;; esac; done
  gcp_require_project
  while :; do
    if [[ $banner == 1 ]]; then
      serial_read "$name" "$start" | grep -A4 'admin token (generated' || true
    else
      serial_read "$name" "$start" | mask_tokens
    fi
    start="$SERIAL_NEXT"
    [[ $follow == 1 && "$DRY" != 1 ]] || break
    sleep 5
  done
}

# vm_wait <vm> <ip> [base url]: /ping 200 or die; prints timings JSON on stdout.
vm_wait() {
  local name="$1" url="${3:-http://$2:8000}" t0 first="" code start=0 chunk
  t0="$(date +%s)"
  [[ "$DRY" == 1 ]] && { jq -nc '{first_ping_s: 0, ready_s: 0, dry_run: true}'; return 0; }
  while :; do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$url/ping" || true)"
    [[ -z "$first" && ( "$code" == 204 || "$code" == 200 ) ]] && first="$(date +%s)"
    [[ "$code" == 200 ]] && break
    chunk="$(serial_read "$name" "$start" 2>/dev/null || true)"; start="$SERIAL_NEXT"
    grep 'FV-GCP ' <<<"$chunk" | grep -v 'FV-GCP verify:' | mask_tokens | sed 's/^/  serial: /' >&2 || true
    if grep -q 'FV-GCP FAILED' <<<"$chunk"; then die "$name: $(grep -m1 'FV-GCP FAILED' <<<"$chunk")"; fi
    (( $(date +%s) - t0 < ${FV_BOOT_WAIT_S:-2700} )) || die "$name never answered /ping 200 (last $code)"
    sleep "${FV_GCP_POLL_S:-10}"
  done
  jq -nc --argjson f "$(( ${first:-$(date +%s)} - t0 ))" --argjson r "$(( $(date +%s) - t0 ))" '{first_ping_s: $f, ready_s: $r}'
}

# Removes fv-secret-* metadata once the container has its env (metadata mode).
vm_scrub_secrets() {
  local name="$1" inst body resp
  inst="$(gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/instances/$name")" || return 0
  jq -e '[.metadata.items[]? | select(.key|startswith("fv-secret-FV_"))] | length > 0' >/dev/null <<<"$inst" || return 0
  body="$(jq -c '{fingerprint: .metadata.fingerprint, items: [.metadata.items[] | select(.key|startswith("fv-secret-FV_")|not)]}' <<<"$inst")"
  resp="$(gce POST "$COMPUTE/projects/$PROJECT/zones/$ZONE/instances/$name/setMetadata" "$body")" && gce_wait "$resp" \
    && log "secrets removed from $name metadata"
}

# vm_set_metadata <vm> <key> <value>
vm_set_metadata() {
  local name="$1" k="$2" v="$3" inst body resp
  inst="$(gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/instances/$name")" || return 1
  body="$(jq -c --arg k "$k" --arg v "$v" '{fingerprint: .metadata.fingerprint,
    items: ([.metadata.items[]? | select(.key != $k)] + [{key: $k, value: $v}])}' <<<"$inst")"
  resp="$(gce POST "$COMPUTE/projects/$PROJECT/zones/$ZONE/instances/$name/setMetadata" "$body")" && gce_wait "$resp"
}

vm_list() {
  gcp_require_project
  local f="labels.fv-owner%3D$FV_OWNER_LABEL"
  echo "# instances (name zone machine status kind role deadline created)"
  gce GET "$COMPUTE/projects/$PROJECT/aggregated/instances?filter=$f" \
    | jq -r '.items // {} | to_entries[] | .value.instances[]? | [.name, (.zone|split("/")|last), (.machineType|split("/")|last), .status, (.labels["fv-kind"] // ""), (.labels["fv-role"] // ""), (.labels["fv-deadline"] // ""), .creationTimestamp] | @tsv'
  echo "# disks"
  gce GET "$COMPUTE/projects/$PROJECT/aggregated/disks?filter=$f" \
    | jq -r '.items // {} | to_entries[] | .value.disks[]? | [.name, (.zone|split("/")|last), (.type|split("/")|last), .sizeGb, (.accessMode // ""), ((.users // [])|length|tostring) + " users"] | @tsv'
  echo "# images"
  gce GET "$COMPUTE/projects/$PROJECT/global/images?filter=$f" | jq -r '.items[]? | [.name, .diskSizeGb, .status, .creationTimestamp] | @tsv'
  echo "# firewall rules"
  gce GET "$COMPUTE/projects/$PROJECT/global/firewalls" \
    | jq -r --arg o "fv-owner=$FV_OWNER_LABEL" '.items[]? | select((.description // "") | contains($o)) | [.name, (.sourceRanges|join(","))] | @tsv'
}

# vm_gc: our firewall rules (fv-serve-<vm>) whose VM no longer exists in any zone.
vm_gc() {
  gcp_require_project
  local fw vm resp live
  live="$(gce GET "$COMPUTE/projects/$PROJECT/aggregated/instances?filter=labels.fv-owner%3D$FV_OWNER_LABEL" \
    | jq -r '.items // {} | to_entries[] | .value.instances[]? | .name')" || return 1
  while read -r fw; do
    [[ -n "$fw" ]] || continue
    vm="${fw#fv-serve-}"
    grep -qxF "$vm" <<<"$live" && continue
    resp="$(gce DELETE "$COMPUTE/projects/$PROJECT/global/firewalls/$fw")" && gce_wait "$resp" && gcp_ledger "firewall-deleted $fw (gc)" && log "gc: deleted $fw"
  done < <(gce GET "$COMPUTE/projects/$PROJECT/global/firewalls" \
    | jq -r --arg o "fv-owner=$FV_OWNER_LABEL" '.items[]? | select((.description // "") | contains($o)) | select(.name|startswith("fv-serve-")) | .name')
}

vm_preflight() {
  require_tools curl jq openssl sha256sum
  gcp_require_project
  local ok=1 resp m q
  if [[ "$DRY" != 1 ]]; then
    bash "$HERE/auth.sh" check || die "auth failed"
  fi
  if resp="$(gce GET "$COMPUTE/$GCP_IMAGE")"; then
    log "boot image: $(jq -r '.name // "?"' <<<"$resp") ($GCP_IMAGE)"
  else
    log "boot image $GCP_IMAGE not readable: $(jq -c '.error.message // .' <<<"$resp" 2>/dev/null | head -c 200)"; ok=0
  fi
  for m in g4-standard-48 a3-highgpu-1g g2-standard-8 g2-standard-16 c3-standard-8; do
    if gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/machineTypes/$m" >/dev/null 2>&1; then log "machine type $m: offered in $ZONE"; else log "machine type $m: NOT offered in $ZONE"; fi
  done
  resp="$(gce GET "$COMPUTE/projects/$PROJECT/regions/$REGION")" || { log "region $REGION: $(head -c 200 <<<"$resp")"; ok=0; }
  q="$(jq -r '.quotas[]? | select(.metric|test("GPU|HDML|HYPERDISK|SSD_TOTAL|CPUS$|C3_CPUS|N2_CPUS|IN_USE_ADDRESSES")) | "  \(.metric)\t\(.usage)/\(.limit)"' <<<"$resp")"
  log "quotas in $REGION (usage/limit):"; printf '%s\n' "$q" >&2
  # Metric names: NVIDIA_L4_GPUS-style or GPU_FAMILY:NVIDIA_RTX_PRO_6000-style (both in the docs).
  grep -qE '(RTX_PRO_6000|NVIDIA_L4|H100)[A-Z_]*[[:space:]]+[0-9.]+/([1-9])' <<<"$q" \
    || log "WARNING: no GPU quota > 0 in $REGION yet: request it (docs/serve/deploy-gcp.md, Quotas)"
  resp="$(gce GET "$COMPUTE/projects/$PROJECT")" || true
  q="$(jq -r '.quotas[]? | select(.metric|test("GPUS_ALL_REGIONS")) | "\(.usage)/\(.limit)"' <<<"$resp" 2>/dev/null || true)"
  log "GPUS_ALL_REGIONS: ${q:-unknown}"
  if gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/disks/$WEIGHTS_DISK" >/dev/null 2>&1; then
    log "weight disk $WEIGHTS_DISK: present"
  else
    log "weight disk $WEIGHTS_DISK: missing (scripts/gcp/weights.sh plan)"
  fi
  (( ok )) || die "preflight found problems"
  log "preflight done (project $PROJECT, zone $ZONE)"
}

secrets_push() {
  gcp_require_project
  load_secrets_file
  local n id resp data
  for n in "${SECRET_NAMES[@]}" HF_TOKEN; do
    [[ -n "${!n:-}" ]] || { log "skip $n (unset)"; continue; }
    id="${n,,}"
    resp="$(gce POST "$SECRETS_API/projects/$PROJECT/secrets?secretId=$id" \
      "$(jq -nc --argjson l "$(gcp_labels secret)" '{replication: {automatic: {}}, labels: $l}')")" \
      || jq -e '.error.status == "ALREADY_EXISTS"' >/dev/null 2>&1 <<<"$resp" || { log "secret $id: $(jq -c '.error.message // .' <<<"$resp" 2>/dev/null | head -c 200)"; continue; }
    data="$(printf '%s' "${!n}" | base64 -w0)"
    if gce POST "$SECRETS_API/projects/$PROJECT/secrets/$id:addVersion" "$(jq -nc --arg d "$data" '{payload: {data: $d}}')" >/dev/null; then
      log "secret $id: new version"; gcp_ledger "secret-version $id"
    else
      log "secret $id: addVersion failed"
    fi
  done
}

main() {
  case "${1:-}" in
    preflight) vm_preflight ;;
    up)
      shift
      VM_ROLE=serve
      vm_up "$@"
      echo "$VM_NAME $VM_IP $VM_KEY" ;;
    worker)
      shift
      VM_ROLE=worker
      vm_up "$@"
      echo "$VM_NAME $VM_IP $VM_URL" ;;
    wait) vm_wait "${2:?vm}" "${3:?ip}" "${4:-}" ;;
    down) vm_down "${2:?vm name}" ;;
    reap) shift; vm_reap "$@" ;;
    list) vm_list ;;
    ssh-free-logs | logs) shift; vm_logs "${1:?vm name}" "${@:2}" ;;
    plan)
      shift
      DRY=1
      vm_up "$@" ;;
    secrets-push) secrets_push ;;
    gc) vm_gc ;;
    *) sed -n '2,72p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
