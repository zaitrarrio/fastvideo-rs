#!/usr/bin/env bash
# Model weights on Google Cloud for the fv-serve GPU VMs (docs/serve/deploy-gcp.md,
# "Weights"). A cache of manifest trees: the EU Runpod volume stays the record
# of truth (CLAUDE.md); every tree comes from the Hugging Face Hub at the
# revision in scripts/gpu/weights-revisions.tsv (and the pinned auxiliary/
# URLs), and is checked with verify-weights.sh before it is used.
#
#   weights.sh plan          the trees, their sizes and the monthly/hourly cost
#   weights.sh bucket        create the GCS bucket (regional, uniform access,
#                            public access prevented): the durable copy
#   weights.sh populate      a CPU VM (c3-standard-8) writes the trees onto a
#                            new Hyperdisk Balanced work disk from the Hub at
#                            the pinned revisions, the auxiliary/ files (and
#                            MMAudio when asked), runs verify-weights.sh and
#                            rsyncs to the bucket; the VM is deleted, the disk
#                            kept. A large download: refused unless
#                            FV_GCP_WEIGHTS_APPROVED=1 (the owner's approval)
#   weights.sh quantize      a G4 VM attaches the work disk read-write and writes
#                            h3-base/ and ltx25/ text_encoder_fp8 with
#                            `fv-gpucheck quantize-text-encoder`, then
#                            verify-weights.sh text-fp8; the VM is deleted
#                            (the alternative, a byte copy from the EU volume,
#                            is in the doc)
#   weights.sh image         a disk image (family fv-weights) from the work disk
#   weights.sh disk-up       a Hyperdisk ML volume from the newest fv-weights
#                            image in $GCP_ZONE, READ_ONLY_MANY (vm.sh attaches
#                            it read-only to every serve VM in that zone)
#   weights.sh disk-down     delete that Hyperdisk ML volume (ours, unattached)
#   weights.sh work-down     delete the work disk (after `image`)
#
# Keep between campaigns: the image (and the bucket). Create the Hyperdisk ML
# volume for a test campaign and delete it after: it bills provisioned space
# AND provisioned throughput by the hour.
#
# Env: as vm.sh, plus HF_TOKEN (LTX-2.5 is gated: accept its terms on the Hub
# with that account first), FV_GCP_WEIGHTS_FAMILIES (default "h3-turbo h3-max
# ltx wan": their weight_trees in families.tsv), FV_GCP_WEIGHTS_EXTRA (more
# dests, e.g. "mmaudio-44k-v2 fastwan22-ti2v-5b upscaler"), FV_GCP_BUCKET
# (default <project>-fv-weights), FV_GCP_WORK_DISK (fv-weights-work),
# FV_GCP_WORK_GB (450), FV_GCP_HDML_GB (450), FV_GCP_HDML_MIBS (1200,
# provisioned MiB/s), FV_GCP_POPULATE_MACHINE (c3-standard-8),
# FV_GCP_QUANT_MACHINE (g4-standard-48), FV_GCP_WEIGHTS_GCS=0 (skip the
# bucket copy), FV_GCP_WEIGHTS_APPROVED=1 (populate only).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=vm.sh
source "$HERE/vm.sh"

MANIFEST="$FV_ROOT/scripts/gpu/weights-manifest.tsv"
REVISIONS="$FV_ROOT/scripts/gpu/weights-revisions.tsv"
WORK_DISK="${FV_GCP_WORK_DISK:-fv-weights-work}"
WORK_GB="${FV_GCP_WORK_GB:-450}"
HDML_GB="${FV_GCP_HDML_GB:-450}"
HDML_MIBS="${FV_GCP_HDML_MIBS:-1200}"
HDML_DISK="${FV_GCP_WEIGHTS_DISK:-fv-weights-hdml-$ZONE}"
BUCKET="${FV_GCP_BUCKET:-${PROJECT:-project}-fv-weights}"
POP_MACHINE="${FV_GCP_POPULATE_MACHINE:-c3-standard-8}"
QUANT_MACHINE="${FV_GCP_QUANT_MACHINE:-g4-standard-48}"
WEIGHT_FAMILIES="${FV_GCP_WEIGHTS_FAMILIES:-h3-turbo h3-max ltx wan}"

# Tree sizes in bytes (docs/ops/runpod-volumes.md §2, du -sb on the volumes;
# GB = 1e9). A dest missing here plans as 0 and is flagged.
SIZES="h3-base	144030000000
FastH3-4-step-Preview-v1-LoRA	6820000000
ltx25	125120000000
fastwan21-1.3b	29212131136
fastwan22-ti2v-5b	24201770562
sfwan21-1.3b	28928823445
wan21-t2v-14b	80406933703
wan22-ti2v-5b	34201427557
upscaler	690000000
h3-to-ltx	390000000
mmaudio-44k-v2	21460000000"
# Not Hub rows: the derived FP8 text encoders (quantize) and auxiliary/ (pinned URLs).
OTHER="text_encoder_fp8 (h3-base + ltx25)	fv-gpucheck quantize-text-encoder on a G4	38874575461
auxiliary/	pinned URLs + SHA-256 (weights-manifest.tsv)	372000000"
# The verify-weights.sh cell of a dest (the family cells come from families.tsv).
dest_cell() {
  case "$1" in
    fastwan22-ti2v-5b | sfwan21-1.3b | wan21-t2v-14b | wan22-ti2v-5b | mmaudio-44k-v2) echo "$1" ;;
    upscaler) echo sol-h3-spark ;;
    *) echo "" ;;
  esac
}

# The dests to fetch: the families' weight_trees plus FV_GCP_WEIGHTS_EXTRA (unique, in order).
selected_dests() {
  local f
  {
    for f in $WEIGHT_FAMILIES; do gcp_family "$f" weight_trees || die "unknown family $f"; done | tr ' ' '\n'
    tr ' ' '\n' <<<"${FV_GCP_WEIGHTS_EXTRA:-}"
  } | awk 'NF && !seen[$0]++'
}

# dest<TAB>revision<TAB>bytes for the selected Hub trees (mmaudio is not one
# Hub tree: fetch-mmaudio.py handles it).
selected_trees() {
  local dest rev bytes
  while read -r dest; do
    [[ "$dest" == mmaudio-44k-v2 ]] && continue
    rev="$(awk -F'\t' -v d="$dest" '$1 == d {print $3; exit}' "$REVISIONS")"
    [[ "$rev" =~ ^[0-9a-f]{40}$ ]] || die "no pinned revision for $dest in weights-revisions.tsv"
    bytes="$(awk -F'\t' -v d="$dest" '$1 == d {print $2; exit}' <<<"$SIZES")"
    printf '%s\t%s\t%s\n' "$dest" "$rev" "${bytes:-0}"
  done < <(selected_dests)
}
want_mmaudio() { selected_dests | grep -qx mmaudio-44k-v2; }

# The verify cells the populated disk must pass.
selected_cells() {
  local f d
  {
    for f in $WEIGHT_FAMILIES; do gcp_family "$f" verify_cells; done | tr ' ' '\n'
    while read -r d; do dest_cell "$d"; done < <(selected_dests)
  } | awk 'NF && !seen[$0]++' | tr '\n' ' ' | sed 's/ $//'
}

# fv-trees rows: dest<TAB>repo<TAB>revision<TAB>globs (from the manifest).
trees_metadata() {
  local dest rev bytes row
  while IFS=$'\t' read -r dest rev bytes; do
    row="$(awk -F'\t' -v d="$dest" '$1 == d {print $2 "\t" $3; exit}' "$MANIFEST")"
    [[ -n "$row" ]] || die "no manifest row for $dest"
    printf '%s\t%s\t%s\t%s\n' "$dest" "${row%%$'\t'*}" "$rev" "${row#*$'\t'}"
  done < <(selected_trees)
}

cmd_plan() {
  local total=0 dest rev bytes what
  printf '%-36s %-44s %10s\n' "tree" "source @ revision" "GB"
  while IFS=$'\t' read -r dest rev bytes; do
    printf '%-36s %-44s %10.2f\n' "$dest" "$(awk -F'\t' -v d="$dest" '$1==d{print $2; exit}' "$MANIFEST")@${rev:0:7}" "$(awk -v b="$bytes" 'BEGIN{print b/1e9}')"
    total=$(( total + bytes ))
  done < <(selected_trees)
  if want_mmaudio; then
    bytes="$(awk -F'\t' '$1 == "mmaudio-44k-v2" {print $2}' <<<"$SIZES")"
    printf '%-36s %-44s %10.2f\n' mmaudio-44k-v2 "fetch-mmaudio.py (3 repos + conversion)" "$(awk -v b="$bytes" 'BEGIN{print b/1e9}')"
    total=$(( total + bytes ))
  fi
  while IFS=$'\t' read -r dest what bytes; do
    printf '%-36s %-44s %10.2f\n' "$dest" "${what:0:44}" "$(awk -v b="$bytes" 'BEGIN{print b/1e9}')"
    total=$(( total + bytes ))
  done <<<"$OTHER"
  local gib hdml_hr
  gib="$(awk -v b="$total" 'BEGIN{printf "%.1f", b/1073741824}')"
  printf '%-36s %-44s %10.2f  (%s GiB; families: %s)\n' "total" "" "$(awk -v b="$total" 'BEGIN{print b/1e9}')" "$gib" "$WEIGHT_FAMILIES"
  awk -v g="$gib" -v w="$WORK_GB" -v h="$HDML_GB" 'BEGIN{exit !(g + 0 < w * 0.95 && g + 0 < h * 0.95)}' \
    || log "WARNING: $gib GiB does not fit FV_GCP_WORK_GB=$WORK_GB / FV_GCP_HDML_GB=$HDML_GB with 5% headroom"
  hdml_hr="$(awk -v g="$HDML_GB" -v t="$HDML_MIBS" -v a="$GCP_HDML_GIB_HR" -v b="$GCP_HDML_MIBS_HR" 'BEGIN{printf "%.3f", g*a + t*b}')"
  cat <<EOF

Cost (us-central1 list prices, docs/serve/deploy-gcp.md for sources):
  GCS Standard, regional       $gib GiB x \$$(awk -v p="$GCP_GCS_GIB_HR" 'BEGIN{printf "%.3f", p*730}')/GiB-month = \$$(awk -v g="$gib" -v p="$GCP_GCS_GIB_HR" 'BEGIN{printf "%.2f", g*p*730}')/month
  disk image (family fv-weights) <= $WORK_GB GiB x \$$(awk -v p="$GCP_IMAGE_GIB_HR" 'BEGIN{printf "%.3f", p*730}')/GiB-month = <= \$$(awk -v g="$WORK_GB" -v p="$GCP_IMAGE_GIB_HR" 'BEGIN{printf "%.2f", g*p*730}')/month
  Hyperdisk ML $HDML_GB GiB + $HDML_MIBS MiB/s   \$$hdml_hr/hour (\$$(awk -v h="$hdml_hr" 'BEGIN{printf "%.0f", h*730}')/month if left up;
                               the default throughput MAX(24*GiB, 400) = $(( HDML_GB * 24 )) MiB/s would be \$$(awk -v g="$HDML_GB" -v a="$GCP_HDML_GIB_HR" -v b="$GCP_HDML_MIBS_HR" 'BEGIN{printf "%.0f", (g*a + g*24*b)*730}')/month: always set it)
  work disk (Hyperdisk Balanced) $WORK_GB GiB x \$$(awk -v p="$GCP_HDB_GIB_HR" 'BEGIN{printf "%.3f", p*730}')/GiB-month while it exists (\$$(awk -v g="$WORK_GB" -v p="$GCP_HDB_GIB_HR" 'BEGIN{printf "%.3f", g*p}')/hour)
  populate VM $POP_MACHINE     \$$(gcp_price "$POP_MACHINE" STANDARD)/hour, ~1-2 h (Hub download ~$gib GiB; MMAudio conversion when selected)
  quantize VM $QUANT_MACHINE   \$$(gcp_price "$QUANT_MACHINE" STANDARD)/hour, ~0.5-1 h
  Hub -> GCP ingress and GCS -> VM in-region reads: free; Internet egress \$0.12/GiB (only MP4s pulled to the runner)

Steps: bucket -> populate -> quantize -> image -> work-down; per campaign: disk-up -> e2e.sh -> disk-down
populate downloads ~$gib GiB: it needs the owner's approval (FV_GCP_WEIGHTS_APPROVED=1).
EOF
}

cmd_bucket() {
  gcp_require_project
  local resp
  resp="$(gce POST "$STORAGE/b?project=$PROJECT" "$(jq -nc --arg n "$BUCKET" --arg loc "$REGION" --argjson l "$(gcp_labels weights)" '{
    name: $n, location: $loc, storageClass: "STANDARD", labels: $l,
    iamConfiguration: {uniformBucketLevelAccess: {enabled: true}, publicAccessPrevention: "enforced"}}')")" \
    || { jq -e '.error.code == 409' >/dev/null 2>&1 <<<"$resp" && { log "bucket $BUCKET exists"; return 0; }; die "bucket: $(jq -c '.error.message // .' <<<"$resp" | head -c 300)"; }
  gcp_ledger "bucket-created $BUCKET"
  log "bucket gs://$BUCKET in $REGION"
}

# ensure_work_disk: the read-write Hyperdisk Balanced work disk in $ZONE.
ensure_work_disk() {
  local resp
  if resp="$(gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/disks/$WORK_DISK")" && [[ "$DRY" != 1 ]]; then
    gcp_is_ours "$resp" || die "disk $WORK_DISK exists and is not ours"
    log "work disk $WORK_DISK exists ($(jq -r .sizeGb <<<"$resp") GB)"; return 0
  fi
  resp="$(gce POST "$COMPUTE/projects/$PROJECT/zones/$ZONE/disks" "$(jq -nc --arg n "$WORK_DISK" --arg z "$ZONE" --arg gb "$WORK_GB" \
    --argjson l "$(gcp_labels weights-work)" '{name: $n, sizeGb: $gb, type: ("zones/" + $z + "/diskTypes/hyperdisk-balanced"),
      provisionedIops: "6000", provisionedThroughput: "600", labels: $l}')")" || die "work disk: $(jq -c '.error.message // .' <<<"$resp" | head -c 300)"
  gce_wait "$resp" || die "work disk create failed"
  gcp_ledger "disk-created $WORK_DISK hyperdisk-balanced ${WORK_GB}GB"
}

# helper_vm <role> <machine> <cap_s> <extra metadata items json>: create a VM
# with the work disk attached read-write; wait for "<ROLE> DONE rc=N" on the
# serial console; delete the VM (trap). Returns rc.
# The verify scripts and tables startup.sh installs on every role.
script_items() {
  jq -nc --rawfile vw "$FV_ROOT/scripts/gpu/verify-weights.sh" --rawfile vs "$FV_ROOT/scripts/gpu/verify-safetensors.sh" \
    --rawfile mf "$MANIFEST" --rawfile sh "$FV_ROOT/scripts/gpu/weights-sha256.tsv" '[
      {key: "fv-script-verify-weights", value: $vw}, {key: "fv-script-verify-safetensors", value: $vs},
      {key: "fv-manifest", value: $mf}, {key: "fv-sha256", value: $sh}]'
}

HELPER=""
helper_cleanup() {
  local rc=$?
  if [[ -n "$HELPER" ]]; then vm_down "$HELPER" || true; HELPER=""; fi
  exit "$rc"
}
helper_vm() {
  local role="$1" machine="$2" cap="$3" extra="$4" prov dph run payload resp start=0 chunk done_line boot
  prov="$(gcp_provisioning "$machine")"
  dph="$(gcp_price "$machine" "$prov")"
  [[ -n "$dph" ]] || die "no price for $machine"
  awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 <= c+0)}' || die "$machine is \$$dph/hr > cap \$$MAX_DPH"
  run="$(date -u +%m%d%H%M%S)"
  HELPER="fv-$role-$run"
  boot="$(gcp_boot_disk_type "$machine")"
  payload="$(jq -n --arg name "$HELPER" --arg zone "$ZONE" --arg mt "$machine" --arg prov "$prov" --argjson cap "$cap" \
    --arg img "$GCP_IMAGE" --arg bt "$boot" --arg work "projects/$PROJECT/zones/$ZONE/disks/$WORK_DISK" \
    --arg net "projects/$PROJECT/global/networks/$NETWORK" --arg sa "$(vm_sa_email)" \
    --argjson labels "$(gcp_labels "$role" "$run" "$(jq -nc --arg d "$(( $(date +%s) + cap ))" '{"fv-deadline": $d}')")" \
    --rawfile startup "$HERE/startup.sh" --arg role "$role" --argjson extra "$extra" --argjson scripts "$(script_items)" '{
      name: $name, description: "fastvideo-rs weights (fv-owner=fastvideo-rs)",
      machineType: ("zones/" + $zone + "/machineTypes/" + $mt), labels: $labels,
      scheduling: {onHostMaintenance: "TERMINATE", automaticRestart: false, provisioningModel: $prov,
        instanceTerminationAction: "DELETE", maxRunDuration: {seconds: ($cap|tostring)}},
      disks: [
        {boot: true, autoDelete: true, deviceName: "boot",
         initializeParams: {sourceImage: $img, diskSizeGb: "100", diskType: ("zones/" + $zone + "/diskTypes/" + $bt), labels: $labels}},
        {source: $work, mode: "READ_WRITE", autoDelete: false, deviceName: "fv-work"}],
      networkInterfaces: [{network: $net, nicType: "GVNIC", accessConfigs: [{type: "ONE_TO_ONE_NAT", name: "External NAT"}]}],
      serviceAccounts: [{email: $sa, scopes: ["https://www.googleapis.com/auth/cloud-platform"]}],
      metadata: {items: ([{key: "startup-script", value: $startup}, {key: "fv-role", value: $role}] + $scripts + $extra)}
    }')"
  trap helper_cleanup EXIT INT TERM
  log "$role VM $HELPER: $machine $prov \$$dph/hr, cap ${cap}s (self-delete)"
  resp="$(gce POST "$COMPUTE/projects/$PROJECT/zones/$ZONE/instances" "$payload")" || die "create $HELPER: $(jq -c '.error.message // .' <<<"$resp" | head -c 400)"
  gce_wait "$resp" || die "create $HELPER failed"
  gcp_ledger "vm-created $HELPER machine=$machine prov=$prov usd_per_hr=$dph cap_s=$cap role=$role"
  mkdir -p "$GCP_OUT/weights"
  local logf="$GCP_OUT/weights/$HELPER.serial.log"
  [[ "$DRY" == 1 ]] && { log "dry run: would wait for '${role^^} DONE' on the serial console"; return 0; }
  local t0; t0="$(date +%s)"
  while :; do
    chunk="$(serial_read "$HELPER" "$start" 2>/dev/null || true)"; start="$SERIAL_NEXT"
    printf '%s' "$chunk" | mask_tokens >>"$logf"
    grep 'FV-GCP ' <<<"$chunk" | grep -v 'FV-GCP verify:' | sed 's/^/  serial: /' >&2 || true
    done_line="$(grep -m1 -E "FV-GCP (${role^^} DONE|FAILED)" <<<"$chunk" || true)"
    [[ -n "$done_line" ]] && break
    (( $(date +%s) - t0 < cap )) || { log "$HELPER hit its cap"; return 1; }
    sleep 20
  done
  log "$HELPER: $done_line (serial log: $logf)"
  vm_down "$HELPER"; HELPER=""; trap - EXIT INT TERM
  [[ "$done_line" == *"rc=0"* ]]
}

cmd_populate() {
  gcp_require_project
  require_tools curl jq openssl
  ensure_work_disk
  local items hf=""
  [[ "${FV_GCP_WEIGHTS_APPROVED:-0}" == 1 ]] \
    || die "populate downloads every selected tree (weights.sh plan for the size): CLAUDE.md asks the owner's approval first; then FV_GCP_WEIGHTS_APPROVED=1"
  [[ -n "${HF_TOKEN:-}" ]] || log "WARNING: HF_TOKEN unset: the gated LTX-2.5 tree will fail"
  [[ -n "${HF_TOKEN:-}" ]] && hf="$HF_TOKEN"
  items="$(jq -nc --arg trees "$(trees_metadata)" \
    --rawfile mm "$FV_ROOT/scripts/gpu/fetch-mmaudio.py" --arg cells "$(selected_cells)" --arg hf "$hf" \
    --arg mmaudio "$(want_mmaudio && echo 1 || echo 0)" \
    --arg bucket "$([[ "${FV_GCP_WEIGHTS_GCS:-1}" == 1 ]] && echo "$BUCKET")" '[
      {key: "fv-trees", value: $trees}, {key: "fv-mmaudio-py", value: $mm}, {key: "fv-mmaudio", value: $mmaudio},
      {key: "fv-verify-cells", value: $cells},
      {key: "fv-gcs-bucket", value: $bucket}] + (if $hf == "" then [] else [{key: "fv-secret-HF_TOKEN", value: $hf}] end)')"
  helper_vm populate "$POP_MACHINE" "${FV_GCP_POPULATE_CAP_S:-14400}" "$items"
}

cmd_quantize() {
  gcp_require_project
  local image
  image="$(gcp_resolve_digest "${FV_SERVE_IMAGE:-$IMAGE_DEFAULT}")"
  helper_vm quantize "$QUANT_MACHINE" "${FV_GCP_QUANT_CAP_S:-5400}" "$(jq -nc --arg i "$image" '[{key: "fv-image", value: $i}]')"
}

cmd_image() {
  gcp_require_project
  local name resp
  name="fv-weights-$(date -u +%Y%m%d-%H%M)"
  resp="$(gce POST "$COMPUTE/projects/$PROJECT/global/images" "$(jq -nc --arg n "$name" --arg d "projects/$PROJECT/zones/$ZONE/disks/$WORK_DISK" \
    --arg r "$REGION" --argjson l "$(gcp_labels weights)" '{name: $n, family: "fv-weights", sourceDisk: $d, storageLocations: [$r], labels: $l,
      description: "fastvideo-rs weight tree (weights/<dest>), ext4, see docs/serve/deploy-gcp.md"}')")" \
    || die "image: $(jq -c '.error.message // .' <<<"$resp" | head -c 300)"
  gce_wait "$resp" || die "image create failed (is the work disk still attached?)"
  gcp_ledger "image-created $name from $WORK_DISK"
  log "image $name (family fv-weights)"
}

cmd_disk_up() {
  gcp_require_project
  local resp img
  if [[ "$DRY" != 1 ]] && resp="$(gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/disks/$HDML_DISK")"; then
    log "$HDML_DISK exists ($(jq -r '.accessMode // "?"' <<<"$resp"), $(jq -r .sizeGb <<<"$resp") GB)"; return 0
  fi
  img="${FV_GCP_WEIGHTS_IMAGE:-projects/$PROJECT/global/images/family/fv-weights}"
  resp="$(gce POST "$COMPUTE/projects/$PROJECT/zones/$ZONE/disks" "$(jq -nc --arg n "$HDML_DISK" --arg z "$ZONE" --arg img "$img" \
    --arg gb "$HDML_GB" --arg t "$HDML_MIBS" --argjson l "$(gcp_labels weights-hdml)" '{
      name: $n, type: ("zones/" + $z + "/diskTypes/hyperdisk-ml"), sourceImage: $img, sizeGb: $gb,
      provisionedThroughput: $t, accessMode: "READ_ONLY_MANY", labels: $l}')")" \
    || die "Hyperdisk ML: $(jq -c '.error.message // .' <<<"$resp" | head -c 300)"
  gce_wait "$resp" || die "Hyperdisk ML create failed"
  gcp_ledger "disk-created $HDML_DISK hyperdisk-ml ${HDML_GB}GB ${HDML_MIBS}MiBps"
  log "$HDML_DISK ready (READ_ONLY_MANY, ${HDML_MIBS} MiB/s): \$$(awk -v g="$HDML_GB" -v t="$HDML_MIBS" -v a="$GCP_HDML_GIB_HR" -v b="$GCP_HDML_MIBS_HR" 'BEGIN{printf "%.3f", g*a + t*b}')/hour until disk-down"
}

# delete_disk <name>: ours and unattached only.
delete_disk() {
  local name="$1" resp
  resp="$(gce GET "$COMPUTE/projects/$PROJECT/zones/$ZONE/disks/$name")" || { log "$name not found"; return 0; }
  gcp_is_ours "$resp" || die "disk $name is not ours: refusing"
  [[ "$(jq '(.users // []) | length' <<<"$resp")" == 0 ]] || die "disk $name is attached to $(jq -r '.users | map(split("/")|last) | join(",")' <<<"$resp")"
  resp="$(gce DELETE "$COMPUTE/projects/$PROJECT/zones/$ZONE/disks/$name")" && gce_wait "$resp" && gcp_ledger "disk-deleted $name" && log "deleted $name"
}

case "${1:-}" in
  plan) cmd_plan ;;
  bucket) cmd_bucket ;;
  populate) cmd_populate ;;
  quantize) cmd_quantize ;;
  image) cmd_image ;;
  disk-up) cmd_disk_up ;;
  disk-down) gcp_require_project; delete_disk "$HDML_DISK" ;;
  work-down) gcp_require_project; delete_disk "$WORK_DISK" ;;
  *) sed -n '2,43p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
