#!/usr/bin/env bash
# Runs ON the fill pod (r2-fill-volume.sh): fills a weights network volume, mounted at /workspace, from the R2
# weights store, or (FV_FILL_MODE=compare) only compares a volume with the R2 manifests and writes nothing to it.
#
#   fill:    r2-weights.py plan (the read-only key presigns the GETs) -> r2-fetch.py --add-only --full-hash into
#            /workspace/weights: each tree is staged under .r2-partial/<dest>@<rev>.<pod>, every 64 MiB part and
#            every whole file checked against the manifest's SHA-256, then renamed into place with .complete,
#            .fv-revision, .fv-source, .fv-manifest.json and sha256.txt (.fv-sha256.txt where the tree ships its own).
#            Add-only: a tree already present at its revision is kept; one present at another revision stops the run.
#            Then compare-dir reads every tree back from the volume (SHA-256 of every part).
#   compare: compare-dir only (e.g. the old volume against the same manifests). Nothing is written to /workspace.
#
# Env: FV_R2_RO_ID / FV_R2_RO_SECRET (the read-only key pair, fv-weights only), FV_WEIGHTS_R2_ENDPOINT,
# FV_FILL_TREES (space-separated dests), FV_FILL_MODE (fill | compare), FV_POD_ID.
# /srv (served on :8000 through the Runpod proxy): log.txt, status.tsv (step, rc, seconds), fetch.log,
# fetch-status.json, compare.txt, inventory.txt, du.tsv, DONE (0 = every step ok).
set -u
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
V="${FV_FILL_VOLUME:-/workspace}"
W="$V/weights"
SRV="${FV_FILL_SRV:-/srv}"
T="${FV_FILL_TOOLS:-/opt/fv}"
MODE="${FV_FILL_MODE:-fill}"
TREES="${FV_FILL_TREES:?FV_FILL_TREES}"
FV_WEIGHTS_R2_ACCESS_KEY_ID="${FV_R2_RO_ID:?FV_R2_RO_ID}"
FV_WEIGHTS_R2_SECRET_ACCESS_KEY="${FV_R2_RO_SECRET:?FV_R2_RO_SECRET}"
export FV_WEIGHTS_R2_ACCESS_KEY_ID FV_WEIGHTS_R2_SECRET_ACCESS_KEY
P=(python3 "$T/r2-weights.py")
cd "$SRV" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >>"$SRV/log.txt"; }
step() { # name, command...: runs it, appends "name rc seconds" to status.tsv
  local name="$1" t0=$SECONDS rc
  shift
  "$@"
  rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS - t0))" >>"$SRV/status.tsv"
  return "$rc"
}
log "start: mode $MODE, trees: $TREES"
{ echo "## ls -la $V"; ls -la "$V"; echo "## ls -la $W"; ls -la "$W"; } >"$SRV/inventory.txt" 2>&1

case "$MODE" in
  fill)
    mkdir -p "$W"
    # shellcheck disable=SC2086 # the tree list splits on purpose
    step plan "${P[@]}" plan $TREES --ttl 21600 --out /root/plan.json >>"$SRV/log.txt" 2>&1 || { echo 1 >"$SRV/DONE"; exit 1; }
    step fetch python3 "$T/r2-fetch.py" --root "$W" --plan /root/plan.json --add-only --pod "fill-${FV_POD_ID:-$$}" \
      --full-hash --status "$SRV/fetch-status.json" >>"$SRV/fetch.log" 2>&1 || log "fetch failed (fetch.log)"
    ;;
  compare) ;;
  *) log "unknown FV_FILL_MODE $MODE"; echo 1 >"$SRV/DONE"; exit 1 ;;
esac

for d in $TREES; do
  step "compare $d" "${P[@]}" compare-dir "$d" "$W/$d" --workers 32 >>"$SRV/compare.txt" 2>&1
done
if [ "$MODE" = fill ]; then
  for d in $TREES; do
    printf '%s\t%s\t%s\t%s\n' "$d" "$(du -sb "$W/$d" 2>/dev/null | cut -f1)" \
      "$(cat "$W/$d/.fv-revision" 2>/dev/null)" "$([ -f "$W/$d/.complete" ] && echo complete || echo MISSING)" >>"$SRV/du.tsv"
  done
  { echo "## du -sb $W"; du -sb "$W"; echo "## ls -la $W $W/.r2-partial"; ls -la "$W" "$W/.r2-partial"; } >>"$SRV/inventory.txt" 2>&1
fi
bad="$(awk -F'\t' '$2 != 0' "$SRV/status.tsv" | wc -l)"
log "end: $bad step(s) failed"
echo $((bad > 0 ? 1 : 0)) >"$SRV/DONE"
