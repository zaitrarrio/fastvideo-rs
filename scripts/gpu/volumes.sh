# shellcheck shell=bash disable=SC2034 # the variables are read by the scripts that source this
# The Runpod weights network volumes (CLAUDE.md, docs/ops/runpod-volumes.md).
# Sourced by the Runpod scripts; defines variables and functions only.
#
# EU only (owner decision, 2026-10-06): Runpod deleted the US volume
# (s2k01690bi, fv-weights-b200-us, US-CA-2) on about 2026-10-05 while the
# account balance was negative. Every default is EU, and asking for the US
# volume or region fails with FV_US_GONE instead of mounting a missing volume.
#
# To bring US back after a rebuild (docs/ops/runpod-volumes.md §5,
# scripts/gpu/rebuild-volume.sh us): set FV_US_VOLUME_ID (and
# FV_US_VOLUME_NAME) below to the new volume. That one change re-enables the
# `us` region everywhere that sources this file. (fv-control has the same
# switch: US_VOLUME_ID in control/src/cluster/regions.ts.)

# EU (owner decision 2026-10-13, "eu volume - reduce to 500 gb plan"): fv-weights-h3-ltx-500, 500 GB in
# EU-RO-1, H3 + LTX trees only (FV_EU_VOLUME_TREES); everything else comes from R2 (weights_source r2).
# It replaced jg48s6o1w0 (fv-weights-h3-ltx-hy, 2000 GB, EUR-IS-1; Runpod no longer creates volumes there),
# which the coordinator deletes once a pod has read the new volume (docs/ops/runpod-volumes.md §0).
FV_EU_VOLUME_ID=bz0ofkfx45
FV_EU_VOLUME_NAME=fv-weights-h3-ltx-500
FV_EU_DC=EU-RO-1
FV_EU_GPUS="NVIDIA RTX PRO 6000 Blackwell Server Edition"
# The trees on the EU volume (462.45 GB in weights-r2.tsv), filled from R2 by r2-fill-volume.sh on 2026-10-13.
# control/src/volumes.ts EU_TREES holds the same list (a unit test compares them).
FV_EU_VOLUME_TREES="h3-base h3-base/text_encoder_fp8 h3-base/transformer_prequant_4step-vsa_mxfp8 \
h3-base/transformer_prequant_sol-h3_mxfp8 FastH3-4-step-Preview-v1-LoRA h3-ref2va ltx25 ltx25-dev \
ltx25-ic-lora-ingredients upscaler h3-to-ltx auxiliary/tae auxiliary/lpips"

FV_US_VOLUME_ID=""   # empty = US unavailable (was s2k01690bi, deleted)
FV_US_VOLUME_NAME="" # empty = US unavailable (was fv-weights-b200-us, deleted)
FV_US_DC=US-CA-2
FV_US_GPUS="NVIDIA H100 80GB HBM3,NVIDIA H100 NVL,NVIDIA H200"

# The deleted US volume: refused by id always, by name while US is unset.
FV_US_DELETED_VOLUME_ID=s2k01690bi
FV_US_DELETED_VOLUME_NAME=fv-weights-b200-us
FV_US_GONE="US weights volume deleted 2026-10; EU only, see docs/ops/runpod-volumes.md"

fv_us_gone() { printf '[%s] FATAL: %s\n' "$(date -u +%H:%M:%S)" "$FV_US_GONE" >&2; exit 2; }

# fv_us_available: true when a US volume is configured.
fv_us_available() { [[ -n "$FV_US_VOLUME_ID" ]]; }

# fv_check_volume <id or name>...: exit 2 with FV_US_GONE when any argument
# names the deleted US volume (or the US volume while it is unset).
fv_check_volume() {
  local v
  for v in "$@"; do
    [[ "$v" == "$FV_US_DELETED_VOLUME_ID" ]] && fv_us_gone
    if ! fv_us_available && [[ "$v" == "$FV_US_DELETED_VOLUME_NAME" ]]; then fv_us_gone; fi
  done
  return 0
}

# fv_check_regions <region>...: exit 2 on us while it is unavailable, or on an
# unknown region. Accepts words or one space-separated string.
fv_check_regions() {
  local r
  # shellcheck disable=SC2068 # split a "eu us" string into words on purpose
  for r in $@; do
    case "$r" in
      eu) ;;
      us) fv_us_available || fv_us_gone ;;
      *) printf '[%s] FATAL: unknown region %s (eu%s)\n' "$(date -u +%H:%M:%S)" "$r" "$(fv_us_available && echo ', us')" >&2; exit 2 ;;
    esac
  done
}

# fv_region_volume / fv_region_dc / fv_region_gpus <eu|us>: the region's
# volume id, data centre and GPU list (comma separated). us while it is
# unavailable exits 2 with FV_US_GONE.
fv_region_volume() { fv_check_regions "$1"; case $1 in eu) echo "$FV_EU_VOLUME_ID" ;; us) echo "$FV_US_VOLUME_ID" ;; esac; }
fv_region_dc() { fv_check_regions "$1"; case $1 in eu) echo "$FV_EU_DC" ;; us) echo "$FV_US_DC" ;; esac; }
fv_region_gpus() { fv_check_regions "$1"; case $1 in eu) echo "$FV_EU_GPUS" ;; us) echo "$FV_US_GPUS" ;; esac; }
