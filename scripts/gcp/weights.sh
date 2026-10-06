#!/usr/bin/env bash
# Model weights on Google Cloud for the fv-serve GPU VMs (docs/serve/deploy-gcp.md).
# Everything comes from the Hugging Face Hub (and the pinned auxiliary/ URLs)
# directly into GCP; nothing is copied from the Runpod volumes.
#
#   weights.sh plan          the trees, their sizes and the monthly/hourly cost
#   weights.sh bucket        create the GCS bucket (regional, uniform access,
#                            public access prevented): the durable copy
#   weights.sh populate      a CPU VM (c3-standard-8) writes every tree onto a
#                            new Hyperdisk Balanced work disk from the Hub at
#                            the pinned revisions, the auxiliary/ files and
#                            MMAudio, runs verify-weights.sh and rsyncs to the
#                            bucket; the VM is deleted, the disk kept
#   weights.sh quantize      a G4 VM attaches the work disk read-write and writes
#                            h3-base/ and ltx25/ text_encoder_fp8 with
#                            `fv-gpucheck quantize-text-encoder`, then
#                            verify-weights.sh text-fp8; the VM is deleted
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
# with that account first), FV_GCP_BUCKET (default <project>-fv-weights),
# FV_GCP_WORK_DISK (fv-weights-work), FV_GCP_WORK_GB (600),
# FV_GCP_HDML_GB (600), FV_GCP_HDML_MIBS (1200, provisioned MiB/s),
# FV_GCP_POPULATE_MACHINE (c3-standard-8), FV_GCP_QUANT_MACHINE
# (g4-standard-48), FV_GCP_WEIGHTS_EXTRA ("upscaler h3-to-ltx" to add the
# Sol-H3 spark trees), FV_GCP_WEIGHTS_GCS=0 (skip the bucket copy).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=vm.sh
source "$HERE/vm.sh"

MANIFEST="$FV_ROOT/scripts/gpu/weights-manifest.tsv"
WORK_DISK="${FV_GCP_WORK_DISK:-fv-weights-work}"
WORK_GB="${FV_GCP_WORK_GB:-600}"
HDML_GB="${FV_GCP_HDML_GB:-600}"
HDML_MIBS="${FV_GCP_HDML_MIBS:-1200}"
HDML_DISK="${FV_GCP_WEIGHTS_DISK:-fv-weights-hdml-$ZONE}"
BUCKET="${FV_GCP_BUCKET:-${PROJECT:-project}-fv-weights}"
POP_MACHINE="${FV_GCP_POPULATE_MACHINE:-c3-standard-8}"
QUANT_MACHINE="${FV_GCP_QUANT_MACHINE:-g4-standard-48}"

# The trees the GCP families need (h3-base, FastH3 = h3-base + the Preview
# LoRA, Sol-H3 = h3-base + the dense LoRA, LTX-2.5, FastWan 1.3B, SF-Wan,
# Wan 14B, TI2V-5B), each at a pinned Hub revision. The four Wan revisions
# are the ones on the Runpod volumes (docs/gaps/2026-09-27-volume-sync.md);
# the others are the Hub `main` of 2026-09-28, whose last change predates
# that sync. dest<TAB>revision<TAB>bytes (volume-sync du -sb, GB = 1e9)
TREES="h3-base	42ed227ee7df40d41602854ae760620d6eb651fe	144030000000
FastH3-4-step-Preview-v1-LoRA	f509e629374cac104e7f62daecce6d1488a3041d	6820000000
ltx25	426936f8b22dc28e4def61e515478b0b7e4a53cc	125120000000
fastwan21-1.3b	25e7ed7f41fd8ce2fdd108688c65e8caf0ce3aef	29212131136
sfwan21-1.3b	4b44356635ae5e927ca552a220f768022be76004	28928823445
wan21-t2v-14b	38ec498cb3208fb688890f8cc7e94ede2cbd7f68	80406933703
wan22-ti2v-5b	b8fff7315c768468a5333511427288870b2e9635	34201427557
upscaler	3f941d5d182014dd5c0a5e16330420ee2d4aa0c6	690000000
h3-to-ltx	1792c42689a0f22de880eaf57a187c6a373a636d	390000000"
# Not Hub rows: MMAudio (fetch-mmaudio.py, 3 repos + conversion), the
# derived FP8 text encoders (quantize) and auxiliary/ (pinned URLs).
OTHER="mmaudio-44k-v2	fetch-mmaudio.py (hkchengrex/MMAudio@eb13a1a + bigvgan + DFN5B CLIP)	21460000000
text_encoder_fp8 (h3-base + ltx25)	fv-gpucheck quantize-text-encoder on a G4	38874575461
auxiliary/	pinned URLs + SHA-256 (weights-manifest.tsv)	372000000"
CELLS="h3-base fasth3-4step-vsa fasth3-4step-dense sol-h3 ltx25-two-stage fastwan21-1.3b sfwan21-1.3b wan21-t2v-14b wan22-ti2v-5b mmaudio-44k-v2 aux"

selected_trees() {
  local extra=" ${FV_GCP_WEIGHTS_EXTRA:-} " dest rev bytes
  while IFS=$'\t' read -r dest rev bytes; do
    case "$dest" in
      upscaler | h3-to-ltx) [[ "$extra" == *" $dest "* ]] || continue ;;
    esac
    printf '%s\t%s\t%s\n' "$dest" "$rev" "$bytes"
  done <<<"$TREES"
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
  while IFS=$'\t' read -r dest what bytes; do
    printf '%-36s %-44s %10.2f\n' "$dest" "${what:0:44}" "$(awk -v b="$bytes" 'BEGIN{print b/1e9}')"
    total=$(( total + bytes ))
  done <<<"$OTHER"
  local gib hdml_hr
  gib="$(awk -v b="$total" 'BEGIN{printf "%.1f", b/1073741824}')"
  printf '%-36s %-44s %10.2f  (%s GiB; the manifest union is 933.5 GB)\n' "total" "" "$(awk -v b="$total" 'BEGIN{print b/1e9}')" "$gib"
  hdml_hr="$(awk -v g="$HDML_GB" -v t="$HDML_MIBS" -v a="$GCP_HDML_GIB_HR" -v b="$GCP_HDML_MIBS_HR" 'BEGIN{printf "%.3f", g*a + t*b}')"
  cat <<EOF

Cost (us-central1 list prices, docs/serve/deploy-gcp.md for sources):
  GCS Standard, regional       $gib GiB x \$$(awk -v p="$GCP_GCS_GIB_HR" 'BEGIN{printf "%.3f", p*730}')/GiB-month = \$$(awk -v g="$gib" -v p="$GCP_GCS_GIB_HR" 'BEGIN{printf "%.2f", g*p*730}')/month
  disk image (family fv-weights) <= $WORK_GB GiB x \$$(awk -v p="$GCP_IMAGE_GIB_HR" 'BEGIN{printf "%.3f", p*730}')/GiB-month = <= \$$(awk -v g="$WORK_GB" -v p="$GCP_IMAGE_GIB_HR" 'BEGIN{printf "%.2f", g*p*730}')/month
  Hyperdisk ML $HDML_GB GiB + $HDML_MIBS MiB/s   \$$hdml_hr/hour (\$$(awk -v h="$hdml_hr" 'BEGIN{printf "%.0f", h*730}')/month if left up;
                               the default throughput MAX(24*GiB, 400) = $(( HDML_GB * 24 )) MiB/s would be \$$(awk -v g="$HDML_GB" -v a="$GCP_HDML_GIB_HR" -v b="$GCP_HDML_MIBS_HR" 'BEGIN{printf "%.0f", (g*a + g*24*b)*730}')/month: always set it)
  work disk (Hyperdisk Balanced) $WORK_GB GiB x \$$(awk -v p="$GCP_HDB_GIB_HR" 'BEGIN{printf "%.3f", p*730}')/GiB-month while it exists (\$$(awk -v g="$WORK_GB" -v p="$GCP_HDB_GIB_HR" 'BEGIN{printf "%.3f", g*p}')/hour)
  populate VM $POP_MACHINE     \$$(gcp_price "$POP_MACHINE" STANDARD)/hour, ~1-2 h (Hub download ~$gib GiB + MMAudio conversion)
  quantize VM $QUANT_MACHINE   \$$(gcp_price "$QUANT_MACHINE" STANDARD)/hour, ~0.5-1 h
  Hub -> GCP ingress and GCS -> VM in-region reads: free; Internet egress \$0.12/GiB (only MP4s pulled to the runner)

Steps: bucket -> populate -> quantize -> image -> work-down; per campaign: disk-up -> e2e.sh -> disk-down
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
    --argjson labels "$(gcp_labels "$role" "$run")" --rawfile startup "$HERE/startup.sh" --arg role "$role" --argjson extra "$extra" '{
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
      metadata: {items: ([{key: "startup-script", value: $startup}, {key: "fv-role", value: $role}] + $extra)}
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
  [[ -n "${HF_TOKEN:-}" ]] || log "WARNING: HF_TOKEN unset: the gated LTX-2.5 tree will fail"
  [[ -n "${HF_TOKEN:-}" ]] && hf="$HF_TOKEN"
  items="$(jq -nc --arg trees "$(trees_metadata)" --rawfile vw "$FV_ROOT/scripts/gpu/verify-weights.sh" \
    --rawfile vs "$FV_ROOT/scripts/gpu/verify-safetensors.sh" --rawfile mf "$MANIFEST" \
    --rawfile mm "$FV_ROOT/scripts/gpu/fetch-mmaudio.py" --arg cells "$CELLS" --arg hf "$hf" \
    --arg bucket "$([[ "${FV_GCP_WEIGHTS_GCS:-1}" == 1 ]] && echo "$BUCKET")" '[
      {key: "fv-trees", value: $trees}, {key: "fv-script-verify-weights", value: $vw},
      {key: "fv-script-verify-safetensors", value: $vs}, {key: "fv-manifest", value: $mf},
      {key: "fv-mmaudio-py", value: $mm}, {key: "fv-verify-cells", value: $cells},
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
  *) sed -n '2,37p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
