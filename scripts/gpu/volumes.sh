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

FV_EU_VOLUME_ID=jg48s6o1w0
FV_EU_VOLUME_NAME=fv-weights-h3-ltx-hy
FV_EU_DC=EUR-IS-1
FV_EU_GPUS="NVIDIA RTX PRO 6000 Blackwell Server Edition"

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
