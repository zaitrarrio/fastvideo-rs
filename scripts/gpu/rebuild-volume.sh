#!/usr/bin/env bash
# Rebuild (or top up) one weight network volume from the Hub, add-only.
# Runs ON a CPU pod that mounts the volume at /workspace
# (docs/ops/runpod-volumes.md "Rebuild a weight volume").
#
#   rebuild-volume.sh <us|eu> --dry-run   print the plan (works anywhere: no volume,
#                                         no GPU, no network; shows each tree's state
#                                         when $FV_WEIGHTS is readable)
#   rebuild-volume.sh <us|eu>             run it on the pod
#   options: --only <dest>[,<dest>...]    restrict to these trees (plus nothing else)
#            --deep                       also run the sha:<dest> cells (re-reads
#                                         h3-ref2va, fastwan22, IC-LoRA, MMAudio .pth)
#
# For every tree, in order (small first, then by family):
#   - missing        -> fetch into <parent>/.<name>.partial-<stamp>, verify every file
#                       against the Hub (LFS SHA-256 / git blob SHA-1) or the pinned
#                       hash, write .complete, rename into place;
#   - present + .complete + its verify-weights.sh cell ok -> skip;
#   - present but no .complete, or its cell fails -> left alone, reported, exit 1
#                       (add-only: nothing on the volume is modified or deleted; a
#                       human decides).
# Only this script's own temp folders (".<name>.partial-*") are ever removed.
# The derived FP8 text encoders need a GPU (or a copy from the other volume):
# they are printed as a MANUAL step and checked by the text-fp8 cell.
#
# Sources: weights-manifest.tsv (dest, repo, globs), weights-revisions.tsv
# (pinned revisions), weights-sha256.tsv (recorded hashes), fetch-hub-tree.py,
# fetch-h3-ref2va.py, fetch-mmaudio.py, verify-weights.sh.
#
# Env: FV_WEIGHTS (default /workspace/weights); HF_TOKEN (gated repos; else
# /workspace/hf/token is used by the fetchers); FV_REBUILD_LOG (default
# /srv/rebuild: per-tree logs and sha256.txt, serve it over the pod's HTTP
# proxy); FV_REBUILD_EXPECT_DIR (the other volume's per-tree sha256.txt lists,
# named <dest with / as ->.sha256.txt: every file must match them too);
# FV_REBUILD_PIP=0 skips installing huggingface_hub / torch-cpu.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/weights-manifest.tsv"
REVS="$HERE/weights-revisions.tsv"
W="${FV_WEIGHTS:-/workspace/weights}"
LOGDIR="${FV_REBUILD_LOG:-/srv/rebuild}"

usage() { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
side="${1:-}"; shift || true
case "$side" in
  us) VOL_ID=s2k01690bi; VOL_NAME=fv-weights-b200-us; VOL_DC=US-CA-2 ;;
  eu) VOL_ID=jg48s6o1w0; VOL_NAME=fv-weights-h3-ltx-hy; VOL_DC=EUR-IS-1 ;;
  *) usage ;;
esac
DRY=0; DEEP=0; ONLY=""
while (( $# )); do
  case "$1" in
    --dry-run) DRY=1 ;;
    --deep) DEEP=1 ;;
    --only) ONLY="${2:?--only needs a list}"; shift ;;
    *) usage ;;
  esac
  shift
done

# The plan: kind<TAB>dest<TAB>verify cells run to decide "already done"
# ("-" = .complete plus a length check of every safetensors in the tree)
# <TAB>approximate bytes (docs/ops/runpod-volumes.md; GB-rounded where only a
# survey figure exists).
PLAN="aux	auxiliary	aux	372975478
hub	upscaler	-	690592992
hub	h3-to-ltx	-	390000000
hub	FastH3-4-step-Preview-v1-LoRA	fasth3-4step-vsa fasth3-4step-dense	6820000000
hub	ltx25-ic-lora-ingredients	ltx25-ic-lora-ingredients	1308813115
hub	h3-base	h3-base	144030000000
hub	h3-8step	fasth3-8step	147850000000
ref2va	h3-ref2va	h3-ref2va-turbo	69059483520
hub	ltx25	ltx25-two-stage	125120000000
hub	ltx25-dev	ltx25-dev ltx25-a2v-guided	37976670004
hub	ltx2	ltx2	94740000000
hub	ltx23	ltx23	71559124637
hub	fastwan21-1.3b	fastwan21-1.3b	29212131136
hub	wan22-ti2v-5b	wan22-ti2v-5b	34201427557
hub	fastwan22-ti2v-5b	fastwan22-ti2v-5b	24201770562
hub	wan21-t2v-14b	wan21-t2v-14b	80406933703
hub	sfwan21-1.3b	sfwan21-1.3b	28928823445
hub	hy15-480-t2v	hy15-480-t2v	53384330435
hub	hy15-480-i2v	hy15-480-i2v	33780496799
hub	hy15-720-t2v	hy15-720-t2v	53384330435
hub	hy15-720-i2v	hy15-720-i2v	53384305661
mmaudio	mmaudio-44k-v2	mmaudio-44k-v2	21460000000
hub	longlive-1.3b	sha:longlive-1.3b	8476402298
hub	longlive2-5b	sha:longlive2-5b	9999858697
hub	longlive2-5b-nvfp4-s4	sha:longlive2-5b-nvfp4-s4	2945864769
hub	longlive2-5b-nvfp4-s2	sha:longlive2-5b-nvfp4-s2	2945864769
hub	longlive-plug/minimax-h3-few-step	sha:longlive-plug/minimax-h3-few-step	2768406790
hub	longlive-plug/minimax-h3-cfg	sha:longlive-plug/minimax-h3-cfg	2767312418
hub	longlive-plug/wan21-t2v-14b-few-step	sha:longlive-plug/wan21-t2v-14b-few-step	1226929535
hub	longlive-plug/wan21-t2v-14b-cfg	sha:longlive-plug/wan21-t2v-14b-cfg	2453790804
hub	longlive-plug/wan22-ti2v-5b-few-step	sha:longlive-plug/wan22-ti2v-5b-few-step	1289840234
hub	longlive-plug/wan22-ti2v-5b-cfg	sha:longlive-plug/wan22-ti2v-5b-cfg	644966189
hub	auxiliary/upscalers/seedvr2	-	7284343622
hub	auxiliary/upscalers/flashvsr-v1.1	upscalers	6948393656
fp8	h3-base/text_encoder_fp8	text-fp8	25950727517
fp8	ltx25/text_encoder_fp8	text-fp8	12923849944"
# Composite cells, run once every tree is in place.
FINAL_CELLS="aux fasth3-8step h3-base fasth3-4step-vsa fasth3-4step-dense sol-h3 sol-h3-spark h3-ref2va h3-ref2va-turbo
ltx25-two-stage ltx25-dev ltx25-a2v-guided ltx25-ic-lora-ingredients ltx25-ref2v ltx2 ltx23 fastwan21-1.3b wan22-ti2v-5b
fastwan22-ti2v-5b wan21-t2v-14b sfwan21-1.3b hy15-480-t2v hy15-480-i2v hy15-720-t2v hy15-720-i2v mmaudio-44k-v2 upscalers text-fp8
longlive2-5b longlive2-5b-nvfp4 longlive-plug"
# longlive-1.3b-safetensors is derived on a CPU pod (scripts/gpu/convert-longlive.py, on wip/longlive until merged;
# docs/ops/runpod-volumes.md §3); check it with verify-weights.sh longlive-1.3b afterwards.
SHA_CELLS="sha:ltx25-ic-lora-ingredients sha:fastwan22-ti2v-5b sha:mmaudio-44k-v2 sha:h3-ref2va"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
selected() { [[ -z "$ONLY" ]] || [[ ",$ONLY," == *",$1,"* ]]; }
manifest_row() { awk -F'\t' -v d="$1" '$1==d {print; exit}' "$MANIFEST"; }
rev_of() { awk -F'\t' -v d="$1" -v r="$2" '$1==d && $2==r {print $3 "\t" $4; exit}' "$REVS"; }
gb() { awk -v b="$1" 'BEGIN{printf "%.2f", b/1e9}'; }
slug() { local s="$1"; echo "${s//\//-}"; }

# State of a tree: missing | present-no-marker | present
tree_state() {
  local p="$W/$1"
  if [[ -e "$p" ]]; then
    if [[ -f "$p/.complete" || "$1" == auxiliary || "$1" == */text_encoder_fp8 ]]; then echo present; else echo present-no-marker; fi
  else
    echo missing
  fi
}

# verify <dest> <cells>: the tree is done when its cells pass (or, with no
# cell, .complete exists and every safetensors in it is its full length).
verify_tree() {
  local dest="$1" cells="$2"
  if [[ "$cells" == - ]]; then
    [[ -f "$W/$dest/.complete" ]] || return 1
    if find -L "$W/$dest" -name '*.safetensors' -print -quit | grep -q .; then
      FV_WEIGHTS="$W" bash "$HERE/verify-safetensors.sh" --dir "$W/$dest" >/dev/null
    fi
    return 0
  fi
  # shellcheck disable=SC2086  # cells is a word list
  FV_WEIGHTS="$W" bash "$HERE/verify-weights.sh" $cells
}

describe() {
  local kind="$1" dest="$2" row repo rev basis
  case "$kind" in
    hub)
      row="$(manifest_row "$dest")"; repo="$(cut -f2 <<<"$row")"
      IFS=$'\t' read -r rev basis <<<"$(rev_of "$dest" "$repo")"
      printf '%s@%s (%s)' "$repo" "${rev:0:12}" "${basis:-NO REVISION}" ;;
    ref2va) printf 'MiniMaxAI/MiniMax-H3@42ed227 + lightx2v/Minimax-h3-Turbo@3ec17a3 (fetch-h3-ref2va.py pins)' ;;
    mmaudio) printf 'hkchengrex/MMAudio + nvidia/bigvgan_v2 + apple/DFN5B-CLIP (main; .pth md5-pinned; converted)' ;;
    aux) printf 'weights-manifest.tsv auxiliary/ url rows (pinned URL + SHA-256)' ;;
    fp8) printf 'DERIVED: fv-gpucheck quantize-text-encoder (GPU) or copy from the other volume' ;;
  esac
}

command_for() {
  local kind="$1" dest="$2" row repo globs rev
  case "$kind" in
    hub)
      row="$(manifest_row "$dest")"; repo="$(cut -f2 <<<"$row")"; globs="$(cut -f3 <<<"$row")"
      rev="$(rev_of "$dest" "$repo" | cut -f1)"
      printf 'FETCH_REPO=%q FETCH_REVISION=%q FETCH_DEST=%q FETCH_GLOBS=%q FETCH_WEIGHTS=%q FETCH_SRV=%q python3 %q' \
        "$repo" "$rev" "$dest" "$globs" "$W" "$LOGDIR/$(slug "$dest")" "$HERE/fetch-hub-tree.py" ;;
    ref2va) printf 'FETCH_WEIGHTS=%q python3 %q   # log in /srv' "$W" "$HERE/fetch-h3-ref2va.py" ;;
    mmaudio) printf 'MMAUDIO_ROOT=%q FETCH_SRV=%q python3 %q && mv <partial> %q' \
      "$W/.mmaudio-44k-v2.partial-<stamp>" "$LOGDIR/mmaudio-44k-v2" "$HERE/fetch-mmaudio.py" "$W/mmaudio-44k-v2" ;;
    aux) printf 'curl <pinned url> -> auxiliary/<dir>/.<file>.partial-<stamp>; size + SHA-256; mv' ;;
    fp8)
      local root="${dest%/text_encoder_fp8}" fam=h3
      [[ "$root" == ltx25 ]] && fam=ltx2-gemma4
      printf 'GPU pod: fv-gpucheck --out /tmp/q quantize-text-encoder --family %s --root %q --tree %q && mv that %q' \
        "$fam" "$W/$root" "$W/$root/.text_encoder_fp8.partial-<stamp>" "$W/$dest" ;;
  esac
}

# --- fetchers (run mode) -------------------------------------------------------
drop_own_partials() {  # only this script's / the fetchers' own temp folders
  local parent="$1" name="$2" old
  for old in "$parent"/."$name".partial-*; do
    [[ -e "$old" ]] || continue
    log "removing own unfinished temp folder $old"
    rm -rf -- "$old"
  done
}

fetch_hub() {
  local dest="$1" row repo globs rev srv exp=""
  row="$(manifest_row "$dest")"; repo="$(cut -f2 <<<"$row")"; globs="$(cut -f3 <<<"$row")"
  rev="$(rev_of "$dest" "$repo" | cut -f1)"
  [[ "$rev" =~ ^[0-9a-f]{40}$ ]] || { log "no pinned revision for $dest in weights-revisions.tsv"; return 1; }
  srv="$LOGDIR/$(slug "$dest")"; mkdir -p "$srv"; rm -f "$srv/DONE"
  if [[ -n "${FV_REBUILD_EXPECT_DIR:-}" && -f "$FV_REBUILD_EXPECT_DIR/$(slug "$dest").sha256.txt" ]]; then
    exp="$(base64 -w0 "$FV_REBUILD_EXPECT_DIR/$(slug "$dest").sha256.txt")"
  fi
  FETCH_REPO="$repo" FETCH_REVISION="$rev" FETCH_DEST="$dest" FETCH_GLOBS="$globs" \
    FETCH_WEIGHTS="$W" FETCH_SRV="$srv" EXPECT_SHA256="$exp" python3 "$HERE/fetch-hub-tree.py" || true
  [[ "$(cat "$srv/DONE" 2>/dev/null)" == 0 ]]
}

fetch_ref2va() {
  local rc=0
  mkdir -p /srv "$LOGDIR/h3-ref2va"
  FETCH_WEIGHTS="$W" python3 "$HERE/fetch-h3-ref2va.py" || rc=$?
  cp -f /srv/log.txt /srv/sha256.txt "$LOGDIR/h3-ref2va/" 2>/dev/null || true
  return $rc
}

fetch_mmaudio() {
  local tmp srv="$LOGDIR/mmaudio-44k-v2"
  drop_own_partials "$W" mmaudio-44k-v2
  tmp="$W/.mmaudio-44k-v2.partial-$(date -u +%Y%m%d%H%M%S)"
  mkdir -p "$srv"; rm -f "$srv/DONE"
  MMAUDIO_ROOT="$tmp" FETCH_SRV="$srv" HF_HUB_ENABLE_HF_TRANSFER=0 python3 "$HERE/fetch-mmaudio.py" || true
  [[ "$(tr -d '[:space:]' <"$srv/DONE" 2>/dev/null)" == 0 && -f "$tmp/.complete" ]] || { log "mmaudio fetch failed; $tmp left in place"; return 1; }
  sync
  [[ ! -e "$W/mmaudio-44k-v2" ]] || { log "$W/mmaudio-44k-v2 appeared meanwhile; $tmp left in place"; return 1; }
  mv -T -- "$tmp" "$W/mmaudio-44k-v2"
}

fetch_aux() {
  local rel url meta sha size dir name part got rc=0
  while IFS=$'\t' read -r rel url meta; do
    url="${url#url:}"; sha="${meta#sha256:}"; sha="${sha%% *}"; size="${meta##*size:}"
    dir="$W/$(dirname "$rel")"; name="$(basename "$rel")"
    if [[ -e "$W/$rel" ]]; then
      if [[ "$(sha256sum "$W/$rel" | awk '{print $1}')" == "$sha" ]]; then log "aux ok (present): $rel"
      else log "aux $rel present with another SHA-256: left alone"; rc=1; fi
      continue
    fi
    mkdir -p "$dir"
    part="$dir/.$name.partial-$(date -u +%Y%m%d%H%M%S)"
    curl -fsSL --proto '=https' --retry 5 --retry-delay 3 --retry-all-errors -o "$part" "$url" || { log "aux download failed: $url"; rm -f -- "$part"; rc=1; continue; }
    got="$(sha256sum "$part" | awk '{print $1}')"
    if [[ "$got" != "$sha" || "$(wc -c <"$part" | tr -d ' ')" != "$size" ]]; then
      log "aux $rel: sha256 $got / size mismatch; temp file removed"; rm -f -- "$part"; rc=1; continue
    fi
    sync
    mv -n -- "$part" "$W/$rel"
    log "aux added: $rel ($size bytes)"
  done < <(grep -E $'^auxiliary/[^\t]+\turl:' "$MANIFEST")
  for dir in "$W"/auxiliary/tae "$W"/auxiliary/lpips; do
    [[ -d "$dir" && ! -e "$dir/.complete" ]] && date -u +%FT%TZ >"$dir/.complete"
  done
  return $rc
}

# --- plan / run ------------------------------------------------------------------
volume_readable=0; [[ -d "$W" ]] && volume_readable=1
echo "Rebuild plan for $side: volume $VOL_NAME ($VOL_ID, $VOL_DC), weights root $W$([[ $volume_readable == 1 ]] || echo ' (not mounted here)')"
(( DRY )) && echo "Mode: dry run (nothing is fetched or written)" || echo "Mode: RUN (add-only)"
echo

need=0; total=0; n=0
declare -a TODO=()
while IFS=$'\t' read -r kind dest cells bytes; do
  selected "$dest" || continue
  n=$((n + 1)); total=$((total + bytes))
  state="unknown (volume not mounted)"
  if (( volume_readable )); then state="$(tree_state "$dest")"; fi
  printf '%2d. %-34s %8s GB  %s\n' "$n" "$dest" "$(gb "$bytes")" "$(describe "$kind" "$dest")"
  printf '    state: %s; done when: %s\n' "$state" "$( [[ "$cells" == - ]] && echo '.complete + safetensors lengths' || echo "verify-weights.sh $cells")"
  if [[ "$state" == missing || "$state" == unknown* || "$kind" == aux ]]; then
    need=$((need + bytes))
    printf '    fetch: %s\n' "$(command_for "$kind" "$dest")"
  fi
  TODO+=("$kind	$dest	$cells	$state")
done <<<"$PLAN"
echo
echo "Trees: $n, about $(gb "$total") GB in all; to add now: about $(gb "$need") GB (derived FP8 trees included)."
echo "Hub download at the measured 150-250 MB/s per pod: about $(awk -v b="$need" 'BEGIN{printf "%.0f-%.0f", b/250e6/60, b/150e6/60}') min, plus hashing."
echo "Final check (without --only): verify-weights.sh $(tr '\n' ' ' <<<"$FINAL_CELLS")$( ((DEEP)) && echo "$SHA_CELLS")"
if [[ "$side" == eu && -z "${FV_REBUILD_EXPECT_DIR:-}" ]]; then
  echo "Note: set FV_REBUILD_EXPECT_DIR to the US run's per-tree sha256.txt lists to require US = EU file by file."
fi
if (( DRY )); then exit 0; fi

# Run mode.
[[ -d "$W" ]] || { echo "$W does not exist: mount the volume at /workspace first" >&2; exit 1; }
[[ -w "$W" ]] || { echo "$W is not writable" >&2; exit 1; }
free="$(df -PB1 "$W" | awk 'NR==2 {print $4}')"
if (( free < need + need / 20 )); then
  echo "not enough space: $(gb "$free") GB free, about $(gb "$need") GB to fetch (+5%)" >&2; exit 1
fi
mkdir -p "$LOGDIR"
if [[ "${FV_REBUILD_PIP:-1}" != 0 ]]; then
  python3 -c 'import huggingface_hub' 2>/dev/null || pip install -q --no-cache-dir "huggingface_hub>=0.34" hf_xet >>"$LOGDIR/pip.log" 2>&1
fi

declare -a MANUAL=() FAILED=()
for item in ${TODO[@]+"${TODO[@]}"}; do
  IFS=$'\t' read -r kind dest cells state <<<"$item"
  if [[ "$kind" == aux ]]; then  # per file, idempotent: present files are hash-checked, never replaced
    if fetch_aux && verify_tree "$dest" "$cells"; then log "ok: $dest"; else FAILED+=("$dest"); fi
    continue
  fi
  if [[ "$kind" == fp8 ]]; then
    if [[ "$state" == present ]] && verify_tree "$dest" "$cells" >/dev/null 2>&1; then log "ok (verified): $dest"
    else MANUAL+=("$dest: $(command_for fp8 "$dest")"); fi
    continue
  fi
  if [[ "$state" == present || "$state" == present-no-marker ]]; then
    if [[ "$state" == present ]] && verify_tree "$dest" "$cells"; then log "ok (verified, skipped): $dest"
    else log "PRESENT BUT NOT VERIFIED: $dest ($state); left alone (add-only)"; FAILED+=("$dest"); fi
    continue
  fi
  log "fetch $dest"
  ok=0
  case "$kind" in
    hub) fetch_hub "$dest" && ok=1 ;;
    ref2va) fetch_ref2va && ok=1 ;;
    mmaudio)
      if [[ "${FV_REBUILD_PIP:-1}" != 0 ]] && ! python3 -c 'import torch, safetensors, numpy' 2>/dev/null; then
        pip install -q --no-cache-dir safetensors numpy >>"$LOGDIR/pip.log" 2>&1
        pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu >>"$LOGDIR/pip.log" 2>&1
      fi
      fetch_mmaudio && ok=1 ;;
    aux) fetch_aux && ok=1 ;;
  esac
  if (( ok )) && verify_tree "$dest" "$cells"; then log "added and verified: $dest"
  else log "FAILED: $dest (see $LOGDIR)"; FAILED+=("$dest"); fi
done

log "final check"
final_rc=0
if [[ -n "$ONLY" ]]; then  # --only: just the selected trees' own cells
  sel=""; for item in ${TODO[@]+"${TODO[@]}"}; do c="$(cut -f3 <<<"$item")"; [[ "$c" == - ]] || sel+=" $c"; done
  read -r -a cells_all <<<"$(tr ' ' '\n' <<<"$sel" | awk 'NF && !seen[$0]++' | tr '\n' ' ')"
else
  read -r -a cells_all <<<"$(tr '\n' ' ' <<<"$FINAL_CELLS")"
fi
if (( ${#cells_all[@]} == 0 )); then log "no cells to check"; cells_all=(--list); fi
if (( DEEP )); then read -r -a extra <<<"$SHA_CELLS"; cells_all+=("${extra[@]}"); fi
FV_WEIGHTS="$W" bash "$HERE/verify-weights.sh" "${cells_all[@]}" 2>&1 | tee "$LOGDIR/verify.txt" || final_rc=1
if (( ${#MANUAL[@]} )); then
  echo "MANUAL (derived trees, not built on a CPU pod):"; printf '  %s\n' "${MANUAL[@]}"
fi
if (( ${#FAILED[@]} )); then
  echo "FAILED / left alone: ${FAILED[*]}" >&2; exit 1
fi
exit $final_rc
