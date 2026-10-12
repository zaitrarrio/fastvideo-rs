#!/usr/bin/env bash
# Runs ON the copy pod (r2-copy-volume.sh): copies the EU weight volume, mounted READ-ONLY by convention at
# /workspace, into the R2 weights store (docs/ops/r2-weights.md §6). Nothing here writes to /workspace: logs,
# status and Python's caches live on the container disk (/srv, /opt/fv). Idempotent: a rerun (a new pod after a
# crash) skips every published tree and every uploaded file (r2-weights.py is add-only and resumes).
#
# Env: FV_WEIGHTS_R2_ACCESS_KEY_ID / _SECRET_ACCESS_KEY (read-write, the fv-weights bucket only), FV_R2_RO_ID /
# FV_R2_RO_SECRET (fv-control's read-only pair: the deep verify runs with it), FV_WEIGHTS_R2_ENDPOINT,
# FV_COPY_REV (the label of volume-only trees, eu-<YYYYMMDD>), FV_COPY_JOBS (trees at once, default 3),
# FV_COPY_FILES / FV_COPY_WINDOW (files per tree / parts per file in flight), FV_COPY_ONLY (space-separated
# dests: only these), FV_COPY_SKIP_VERIFY=1.
#
# /srv (served on :8000 through the Runpod proxy): log.txt, status.tsv (one line per step: what, rc, seconds),
# inventory.txt, wan-listing.txt, pub-<dest>.log, compare.txt, verify.txt, record.tsv, DONE (0 = all ok).
set -u
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
V="${FV_COPY_VOLUME:-/workspace}"
W="$V/weights"
SRV="${FV_COPY_SRV:-/srv}"

P="python3 ${FV_COPY_TOOLS:-/opt/fv}/r2-weights.py"
REV="${FV_COPY_REV:?FV_COPY_REV}"
JOBS="${FV_COPY_JOBS:-3}"
cd "$SRV" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >>$SRV/log.txt; }
step() { # name, command...: runs it, appends "name rc seconds" to status.tsv
  local name="$1" t0=$SECONDS rc
  shift
  "$@"
  rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS - t0))" >>$SRV/status.tsv
  return "$rc"
}
log "start: rev $REV, $JOBS trees at once"
# Gated Hub repos (Lightricks) list their real LFS SHA-256 only to an authenticated caller: use the project's Hub
# token kept on the volume (the fetchers read the same file), read only, never printed.
if [ -z "${HF_TOKEN:-}" ] && [ -r "$V/hf/token" ]; then HF_TOKEN="$(cat "$V/hf/token")"; export HF_TOKEN; log "Hub token: from the volume's hf/token"; fi

# ---- inventory (read only)
{
  echo "## ls -la $V"; ls -la "$V"
  echo "## ls -la $W"; ls -la "$W"
  echo "## df"; df -h "$V" "$SRV"
} >$SRV/inventory.txt 2>&1
RUNS=""; WAN=""
for d in "$V/runs" "$W/runs"; do [ -d "$d" ] && RUNS="$d" && break; done
for d in "$W/wan" "$V/wan"; do [ -d "$d" ] && WAN="$d" && break; done
if [ -n "$WAN" ]; then
  {
    echo "## $WAN (bytes path)"
    find "$WAN" -type f -printf '%s\t%P\n' | sort -k2
    for f in $(find "$WAN" -maxdepth 3 -type f \( -name 'config.json' -o -iname 'README*' -o -name 'model_index.json' \) | head -10); do
      echo "## $f"; head -c 4000 "$f"; echo
    done
  } >$SRV/wan-listing.txt 2>&1
fi
log "inventory done (runs: ${RUNS:-none}, wan: ${WAN:-none})"

# ---- the trees
HUB="$($P status 2>>$SRV/log.txt | awk '$2 == "hub" {print $1}')"
VOL="h3-base/text_encoder_fp8 ltx25/text_encoder_fp8 h3-base/transformer_prequant_4step-vsa_mxfp8 h3-base/transformer_prequant_sol-h3_mxfp8 longlive-1.3b-safetensors mmaudio-44k-v2 h3-ref2va auxiliary/tae auxiliary/lpips"
if [ -n "${FV_COPY_ONLY:-}" ]; then
  HUB="$(for d in $HUB; do case " $FV_COPY_ONLY " in *" $d "*) echo "$d" ;; esac; done)"
  VOL="$(for d in $VOL; do case " $FV_COPY_ONLY " in *" $d "*) echo "$d" ;; esac; done)"
fi
[ -n "$HUB$VOL" ] || { log "no trees"; echo 1 >$SRV/DONE; exit 1; }
log "hub trees: $(echo $HUB | wc -w); volume-only: $(echo $VOL | wc -w)"

pub() { # dest [--rev X]
  local d="$1" l
  shift
  l="$SRV/pub-$(printf %s "$d" | tr / _).log"
  step "publish $d" $P publish-dir "$d" "$W/$d" "$@" --files "${FV_COPY_FILES:-6}" --window "${FV_COPY_WINDOW:-4}" --note "EU volume jg48s6o1w0" >>"$l" 2>&1 \
    && log "published $d" || log "FAILED $d (tail: $(tail -1 "$l" | cut -c1-200))"
}
export -f pub step log
export P W REV SRV
# Largest first keeps the parallel jobs busy to the end.
order() { for d in "$@"; do printf '%s\t%s\n' "$(du -sbL "$W/$d" 2>/dev/null | cut -f1)" "$d"; done | sort -rn | cut -f2; }
# shellcheck disable=SC2086
order $HUB | xargs -P "$JOBS" -I{} bash -c 'pub "$1"' _ {}
# shellcheck disable=SC2086
order $VOL | xargs -P "$JOBS" -I{} bash -c 'pub "$1" --rev "$REV"' _ {}

# ---- the archive (no manifest, no presets): runs/ and wan/
[ -n "$RUNS" ] && step "archive runs" $P archive runs "$RUNS" --files 48 --window 2 >>$SRV/archive-runs.log 2>&1
[ -n "$WAN" ] && step "archive wan" $P archive wan "$WAN" --files 16 --window 4 >>$SRV/archive-wan.log 2>&1
log "archive done"

# ---- the volume against R2: volume-only trees by SHA-256 (re-read), Hub trees by size (their bytes were hashed
#      against the Hub's LFS SHA-256 on the way up)
if [ "${FV_COPY_SKIP_COMPARE:-0}" != 1 ]; then
for d in $VOL; do step "compare $d" $P compare-dir "$d" "$W/$d" --workers 32 >>$SRV/compare.txt 2>&1; done
for d in $HUB; do step "compare-size $d" $P compare-dir "$d" "$W/$d" --sizes-only >>$SRV/compare.txt 2>&1; done
log "compare done"
fi

# ---- deep verify with fv-control's READ-ONLY key (it must read every object), recorded
if [ "${FV_COPY_SKIP_VERIFY:-0}" != 1 ]; then
  # shellcheck disable=SC2086
  FV_WEIGHTS_R2_VERIFY_ACCESS_KEY_ID="$FV_R2_RO_ID" FV_WEIGHTS_R2_VERIFY_SECRET_ACCESS_KEY="$FV_R2_RO_SECRET" \
    FV_VERIFY_WINDOW=24 FV_R2_TIMEOUT_S=60 step "verify-deep" $P verify --deep --record --skip-recorded --workers 8 $HUB $VOL >>$SRV/verify.txt 2>&1
  log "verify done"
fi
$P record >$SRV/record.tsv 2>>$SRV/log.txt
$P status >$SRV/status-final.txt 2>&1
bad="$(awk -F'\t' '$2 != 0' $SRV/status.tsv | wc -l)"
log "end: $bad step(s) failed"
echo $(( bad > 0 ? 1 : 0 )) >$SRV/DONE
