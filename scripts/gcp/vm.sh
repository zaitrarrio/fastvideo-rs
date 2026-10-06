#!/usr/bin/env bash
# fv-serve on a Google Cloud GPU VM through the Compute Engine REST API (no
# gcloud). docs/serve/deploy-gcp.md; the GCP counterpart of
# scripts/serve/runpod-pod.sh.
#
#   vm.sh preflight              token, project, Compute API, GPU quotas in
#                                $GCP_REGION, the boot image, the machine types
#                                in $GCP_ZONE, the weight disk
#   vm.sh up <family> [image]    create a VM for one family (h3-turbo, h3-max,
#                                ltx-turbo, wan-turbo, fake); print
#                                "<vm> <ip> <api key>" (the key is per run: only
#                                its SHA-256 goes to the VM)
#   vm.sh wait <vm> <ip>         wait for /ping 200; fail early on a FAILED line
#   vm.sh down <vm>              delete the VM and its firewall rule (ours only)
#   vm.sh list                   our VMs, disks, images and firewall rules
#   vm.sh ssh-free-logs <vm> [--follow] [--banner]
#                                the serial console (startup progress + the
#                                fv-serve log); admin tokens are masked unless
#                                --banner is given
#   vm.sh plan <family> [image]  print the REST payloads (same as FV_GCP_DRY_RUN=1 up)
#   vm.sh secrets-push           copy the FV_* secrets from this environment to
#                                Secret Manager (ids = lower-case names)
#   vm.sh gc                     delete our firewall rules whose VM is gone
#
# Money guards (the Runpod scripts' set, cloud-side where possible):
#   - $/hr cap FV_GCP_MAX_DPH (default 6.0) against the price table in lib.sh;
#   - wall clock FV_GCP_CAP_S (default 5400): scheduling.maxRunDuration with
#     instanceTerminationAction=DELETE, so Compute Engine deletes the VM even
#     if this shell, this container or its host dies;
#   - onHostMaintenance=TERMINATE, automaticRestart=false (a GPU VM never
#     comes back on its own);
#   - labels fv-owner=fastvideo-rs, fv-run, fv-kind, fv-family on everything;
#     down/gc refuse resources without fv-owner=fastvideo-rs;
#   - a ledger (artifacts/gcp/ledger.tsv); e2e.sh adds a delete-on-exit trap.
#
# Env: GCP_SA_KEY_JSON, GCP_PROJECT, GCP_REGION (us-central1), GCP_ZONE
# (us-central1-b); FV_SERVE_IMAGE (ghcr.io/zaitrarrio/fastvideo-rs-serve:latest,
# pinned to its digest); FV_GCP_MACHINE (default per family, below);
# FV_GCP_SPOT=1 (Spot; a3-highgpu-1g is always Spot); FV_GCP_SOURCE_CIDR
# (who may reach tcp:8000,40000 and udp:40010; default this machine's public
# IPs as /32); FV_GCP_SECRETS (metadata | secret-manager | none; default
# metadata when any FV_CF_* / FV_R2_* value is set, else none);
# FV_GCP_WEIGHTS_DISK (default fv-weights-hdml-$GCP_ZONE, read-only);
# FV_GCP_SA_EMAIL (VM service account; default the key's client_email);
# FV_GCP_ENCODE_BENCH=1 (VM waits for a clip to compare NVENC with x264);
# FV_GCP_DRY_RUN=1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=lib.sh
source "$HERE/lib.sh"

IMAGE_DEFAULT="ghcr.io/zaitrarrio/fastvideo-rs-serve:latest"
MAX_DPH="${FV_GCP_MAX_DPH:-6.0}"
CAP_S="${FV_GCP_CAP_S:-5400}"
WEIGHTS_DISK="${FV_GCP_WEIGHTS_DISK:-fv-weights-hdml-$ZONE}"
UDP_PORTS="${FV_GCP_UDP_PORTS:-40010}"
SECRET_NAMES=(FV_CF_ACCOUNT_ID FV_CF_API_TOKEN FV_D1_DATABASE_ID FV_R2_BUCKET FV_R2_ENDPOINT
  FV_R2_ACCESS_KEY_ID FV_R2_SECRET_ACCESS_KEY FV_WEBHOOK_ED25519_KEY FV_URL_SIGNING_KEY)

# family -> config, verify-weights cells, default machine type.
family_config() {
  case "$1" in
    h3-turbo) echo gcp-h3-turbo.toml ;;
    h3-max) echo gcp-h3-max.toml ;;
    ltx-turbo) echo gcp-ltx-turbo.toml ;;
    wan-turbo) echo gcp-wan-turbo.toml ;;
    fake) echo runpod-fake.toml ;;
    *) return 1 ;;
  esac
}
family_cells() {
  case "$1" in
    h3-turbo) echo "fasth3-4step-vsa aux" ;;
    h3-max) echo "sol-h3 aux" ;;
    ltx-turbo) echo "ltx25-two-stage aux" ;;
    wan-turbo) echo "fastwan21-1.3b aux" ;;
    fake) echo "" ;;
  esac
}
# G4 (RTX PRO 6000 96 GB, NVENC) for everything by default: one H3 DiT is
# ~41 GB and LTX-2.5 is 22B, and the WP-11 numbers are from the same card.
# wan-turbo also fits an L4 (FV_GCP_MACHINE_WAN_TURBO=g2-standard-16); fake
# needs no GPU memory (g2-standard-8).
family_machine() {
  local var="FV_GCP_MACHINE_${1//-/_}"; var="${var^^}"
  if [[ -n "${!var:-}" ]]; then echo "${!var}"; return; fi
  if [[ -n "${FV_GCP_MACHINE:-}" ]]; then echo "$FV_GCP_MACHINE"; return; fi
  case "$1" in
    fake) echo g2-standard-8 ;;
    *) echo g4-standard-48 ;;
  esac
}

secrets_mode() {
  if [[ -n "${FV_GCP_SECRETS:-}" ]]; then echo "$FV_GCP_SECRETS"; return; fi
  local n
  for n in "${SECRET_NAMES[@]}"; do [[ -n "${!n:-}" ]] && { echo metadata; return; }; done
  echo none
}

vm_sa_email() {
  if [[ -n "${FV_GCP_SA_EMAIL:-}" ]]; then echo "$FV_GCP_SA_EMAIL"; return; fi
  if [[ "$DRY" == 1 ]]; then (gcp_sa_email 2>/dev/null) || echo "fv-e2e@$PROJECT.iam.gserviceaccount.com"; return; fi
  gcp_sa_email
}

# metadata_items <family> <image> <keyhash> -> JSON array of {key,value}
metadata_items() {
  local family="$1" image="$2" keyhash="$3" cfg mode names=() n env_json
  cfg="$(family_config "$family")"
  mode="$(secrets_mode)"
  for n in "${SECRET_NAMES[@]}"; do
    if [[ "$mode" == secret-manager || -n "${!n:-}" ]]; then names+=("$n"); fi
  done
  env_json="$(jq -nc --arg keys "$keyhash" --arg udp "$UDP_PORTS" '{
    FV_SERVE_MODE: "http", FV_WEIGHTS: "/workspace/weights", FV_STATE_DIR: "/fvstate",
    FV_API_KEYS: $keys, FV_SERVE_FORWARD: "1", RUST_LOG: "info", FV_GCP_UDP_PORTS: $udp
  }')"
  {
    jq -nc --rawfile s "$HERE/startup.sh" '{key: "startup-script", value: $s}'
    jq -nc --rawfile s "$FV_ROOT/configs/serve/$cfg" '{key: "fv-config", value: $s}'
    jq -nc --arg v serve '{key: "fv-role", value: $v}'
    jq -nc --arg v "$image" '{key: "fv-image", value: $v}'
    jq -nc --arg v "$env_json" '{key: "fv-env", value: $v}'
    jq -nc --arg v "$(family_cells "$family")" '{key: "fv-verify-cells", value: $v}'
    jq -nc --arg v "$mode" '{key: "fv-secrets-mode", value: $v}'
    jq -nc --arg v "${names[*]:-}" '{key: "fv-secret-names", value: $v}'
    jq -nc --arg v "${FV_GCP_ENCODE_BENCH:-0}" '{key: "fv-encode-bench", value: $v}'
    if [[ "$mode" == metadata ]]; then
      for n in "${names[@]}"; do jq -nc --arg k "fv-secret-$n" --arg v "${!n}" '{key: $k, value: $v}'; done
    fi
  } | jq -sc .
}

# instance_payload <name> <family> <machine> <image> <keyhash> <run>
instance_payload() {
  local name="$1" family="$2" machine="$3" image="$4" keyhash="$5" run="$6" prov boot disks
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
  jq -n --arg name "$name" --arg zone "$ZONE" --arg mt "$machine" --arg prov "$prov" --argjson cap "$CAP_S" \
    --arg net "projects/$PROJECT/global/networks/$NETWORK" --arg sa "$(vm_sa_email)" \
    --argjson labels "$(gcp_labels serve "$run" "$(jq -nc --arg f "$family" '{"fv-family": $f}')")" \
    --argjson disks "$disks" --argjson items "$(metadata_items "$family" "$image" "$keyhash")" '{
      name: $name,
      description: "fastvideo-rs fv-serve e2e (fv-owner=fastvideo-rs)",
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

firewall_payload() {
  local name="$1" run="$2" cidrs="$3"
  jq -n --arg name "fv-serve-$name" --arg tag "$name" --arg run "$run" --argjson src "$cidrs" \
    --arg net "projects/$PROJECT/global/networks/$NETWORK" --arg udp "$UDP_PORTS" '{
      name: $name, network: $net, direction: "INGRESS", priority: 1000,
      description: ("fv-owner=fastvideo-rs fv-run=" + $run + " fv-serve HTTP 8000, ICE-TCP 40000, ICE-UDP " + $udp),
      targetTags: [$tag], sourceRanges: $src,
      allowed: [{IPProtocol: "tcp", ports: ["8000", "40000"]}, {IPProtocol: "udp", ports: [$udp]}]
    }'
}

# shellcheck disable=SC2034 # read by e2e.sh, which sources this file
VM_NAME="" VM_IP="" VM_KEY="" VM_DPH="" VM_MACHINE="" VM_PROV=""

# vm_up <family> [image]: sets VM_NAME/VM_IP/VM_KEY/VM_DPH (for e2e.sh, which
# sources this file); prints nothing on stdout.
vm_up() {
  local family="$1" image machine prov dph run cidrs key keyhash resp
  family_config "$family" >/dev/null || die "unknown family $family (h3-turbo h3-max ltx-turbo wan-turbo fake)"
  require_tools curl jq openssl sha256sum
  gcp_require_project
  machine="$(family_machine "$family")"
  prov="$(gcp_provisioning "$machine")"
  dph="$(gcp_price "$machine" "$prov")"
  [[ -n "$dph" ]] || die "no price for $machine ($prov) in the lib.sh table: add it before spending"
  awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 <= c+0)}' || die "$machine $prov is \$$dph/hr > cap \$$MAX_DPH (FV_GCP_MAX_DPH)"
  image="$(gcp_resolve_digest "${2:-${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}}")"
  run="${FV_GCP_RUN:-$(date -u +%m%d%H%M%S)}"
  VM_NAME="fv-${family//./-}-$run"
  key="fvk-$(openssl rand -hex 16)"
  keyhash="$(printf '%s' "$key" | sha256sum | cut -d' ' -f1)"
  cidrs="$(gcp_source_cidrs)"
  [[ "$(jq length <<<"$cidrs")" -gt 0 ]] || die "no source CIDR (set FV_GCP_SOURCE_CIDR)"
  log "vm $VM_NAME: $machine $prov \$$dph/hr in $ZONE, cap ${CAP_S}s (self-delete), image $image, from $(jq -r 'join(",")' <<<"$cidrs")"
  if [[ "$family" != fake && "$DRY" != 1 ]]; then
    resp="$(gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/disks/$WEIGHTS_DISK")" \
      || die "weight disk $WEIGHTS_DISK not found in $ZONE: run scripts/gcp/weights.sh disk-up (or set FV_GCP_WEIGHTS_DISK)"
  fi
  resp="$(gce POST "$COMPUTE/projects/$PROJECT/global/firewalls" "$(firewall_payload "$VM_NAME" "$run" "$cidrs")")" \
    || die "firewall create: $(jq -c '.error.message // .' <<<"$resp" 2>/dev/null | head -c 400)"
  gce_wait "$resp" || die "firewall create failed"
  gcp_ledger "firewall-created fv-serve-$VM_NAME"
  if ! resp="$(gce POST "$COMPUTE/projects/$PROJECT/zones/$ZONE/instances" "$(instance_payload "$VM_NAME" "$family" "$machine" "$image" "$keyhash" "$run")")" \
    || ! gce_wait "$resp"; then
    log "instance create failed: $(jq -c '.error.message // .error // empty' <<<"$resp" 2>/dev/null | head -c 600)"
    firewall_delete "$VM_NAME" || true
    return 1
  fi
  gcp_ledger "vm-created $VM_NAME machine=$machine prov=$prov usd_per_hr=$dph cap_s=$CAP_S family=$family image=$image"
  resp="$(gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/instances/$VM_NAME")"
  VM_IP="$(jq -r '.networkInterfaces[0].accessConfigs[0].natIP // empty' <<<"$resp")"
  # shellcheck disable=SC2034 # read by e2e.sh
  VM_KEY="$key" VM_DPH="$dph" VM_MACHINE="$machine" VM_PROV="$prov"
  log "vm $VM_NAME is up at $VM_IP (serial: vm.sh ssh-free-logs $VM_NAME)"
}

firewall_delete() {
  local name="$1" fw resp
  fw="fv-serve-$name"
  resp="$(gce GET "$COMPUTE/projects/$PROJECT/global/firewalls/$fw")" || return 0
  gcp_is_ours "$resp" || { log "firewall $fw is not ours: left alone"; return 1; }
  resp="$(gce DELETE "$COMPUTE/projects/$PROJECT/global/firewalls/$fw")" && gce_wait "$resp" && gcp_ledger "firewall-deleted $fw"
}

vm_down() {
  local name="$1" resp
  gcp_require_project
  if resp="$(gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/instances/$name")"; then
    gcp_is_ours "$resp" || die "instance $name has no fv-owner=$FV_OWNER_LABEL label: refusing to delete"
    resp="$(gce DELETE "$COMPUTE/projects/$PROJECT/zones/$ZONE/instances/$name")" && gce_wait "$resp" \
      && gcp_ledger "vm-deleted $name" && log "deleted $name"
  else
    log "instance $name not found (already deleted by its maxRunDuration?)"
  fi
  firewall_delete "$name" || true
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

# vm_wait <vm> <ip>: /ping 200 or die; prints timings JSON on stdout.
vm_wait() {
  local name="$1" ip="$2" t0 first="" code start=0 chunk
  t0="$(date +%s)"
  [[ "$DRY" == 1 ]] && { jq -nc '{first_ping_s: 0, ready_s: 0, dry_run: true}'; return 0; }
  while :; do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "http://$ip:8000/ping" || true)"
    [[ -z "$first" && ( "$code" == 204 || "$code" == 200 ) ]] && first="$(date +%s)"
    [[ "$code" == 200 ]] && break
    chunk="$(serial_read "$name" "$start" 2>/dev/null || true)"; start="$SERIAL_NEXT"
    grep 'FV-GCP ' <<<"$chunk" | grep -v 'FV-GCP verify:' | mask_tokens | sed 's/^/  serial: /' >&2 || true
    if grep -q 'FV-GCP FAILED' <<<"$chunk"; then die "$name: $(grep -m1 'FV-GCP FAILED' <<<"$chunk")"; fi
    (( $(date +%s) - t0 < ${FV_BOOT_WAIT_S:-2700} )) || die "$name never answered /ping 200 (last $code)"
    sleep 10
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
  echo "# instances"
  gce GET "$COMPUTE/projects/$PROJECT/aggregated/instances?filter=$f" \
    | jq -r '.items // {} | to_entries[] | .value.instances[]? | [.name, (.zone|split("/")|last), (.machineType|split("/")|last), .status, (.labels["fv-kind"] // ""), .creationTimestamp] | @tsv'
  echo "# disks"
  gce GET "$COMPUTE/projects/$PROJECT/aggregated/disks?filter=$f" \
    | jq -r '.items // {} | to_entries[] | .value.disks[]? | [.name, (.zone|split("/")|last), (.type|split("/")|last), .sizeGb, (.accessMode // ""), ((.users // [])|length|tostring) + " users"] | @tsv'
  echo "# images"
  gce GET "$COMPUTE/projects/$PROJECT/global/images?filter=$f" | jq -r '.items[]? | [.name, .diskSizeGb, .status, .creationTimestamp] | @tsv'
  echo "# firewall rules"
  gce GET "$COMPUTE/projects/$PROJECT/global/firewalls" \
    | jq -r --arg o "fv-owner=$FV_OWNER_LABEL" '.items[]? | select((.description // "") | contains($o)) | [.name, (.sourceRanges|join(","))] | @tsv'
}

vm_gc() {
  gcp_require_project
  local fw vm resp
  while read -r fw; do
    [[ -n "$fw" ]] || continue
    vm="${fw#fv-serve-}"
    if ! gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/instances/$vm" >/dev/null 2>&1; then
      resp="$(gce DELETE "$COMPUTE/projects/$PROJECT/global/firewalls/$fw")" && gce_wait "$resp" && gcp_ledger "firewall-deleted $fw (gc)" && log "gc: deleted $fw"
    fi
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
  grep -qE '(RTX_PRO_6000|NVIDIA_L4|H100)[A-Z_]*GPUS[[:space:]]+[0-9.]+/([1-9])' <<<"$q" \
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
      vm_up "$@"
      echo "$VM_NAME $VM_IP $VM_KEY" ;;
    wait) vm_wait "${2:?vm}" "${3:?ip}" ;;
    down) vm_down "${2:?vm name}" ;;
    list) vm_list ;;
    ssh-free-logs | logs) shift; vm_logs "${1:?vm name}" "${@:2}" ;;
    plan)
      shift
      DRY=1
      vm_up "$@" ;;
    secrets-push) secrets_push ;;
    gc) vm_gc ;;
    *) sed -n '2,46p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
