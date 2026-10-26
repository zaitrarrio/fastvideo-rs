#!/usr/bin/env bash
# The summary lines the retired families' runpod-matrix.sh branches copied
# into live.log after their arms (docs/adr/0019-fv-lab.md §9.4, §13): their
# lines, frozen, in a frame that runs them on a fetched run directory. This is
# the reference `fv-lab report summary --kind tier` is held to
# (scripts/gpu/tests/fv-lab-report-goldens.sh writes the fixtures from it;
# crates/fastvideo-lab/tests/report.rs compares).
#
#   summary-script.sh <family> <run dir> <out file>
#
# Every line between `# --- verbatim <source> <first>-<last>` and `# --- end`
# is that source's, unchanged: `main` is `git show
# 5987e6a4:scripts/gpu/runpod-matrix.sh | sed -n '<first>,<last>p'` (the last
# commit with the branches), `b4a` is
# crates/fastvideo-lab/fixtures/b4a-script/runpod-matrix.sh (the frozen copy
# of b4a's branch, ADR-0019 §11.6). `fv-lab-report-goldens.sh --check`
# compares them whenever the history holds 5987e6a4. The rest is the frame:
# `log` without the [HH:MM:SS] [family] stamp the script's log() adds, and the
# loops that stood around the per-arm lines (b6_pair over $b6_cells and
# base / pr; h3arms over 768p / 480p and sol / tea / soltea).
set -euo pipefail
FAMILY="$1"
RUNS="$2"
LOG="$3"
: >"$LOG"
log() { printf '%s\n' "$*" >>"$LOG"; }
exec >/dev/null
case "$FAMILY" in
  techniques)
# --- verbatim main 2024-2030
    for f in "$RUNS"/compare/compare-clips-*.json; do
      [[ -f "$f" ]] && log "$(basename "$f"): $(grep -oE '"(status|off_identity|max_abs_diff_uint8)": *[^,}]*' "$f" | tr '\n' ' ')"
    done
    for c in "$RUNS"/*-base "$RUNS"/*-env "$RUNS"/*-prof; do
      [[ -d "$c" ]] || continue
      log "$(basename "$c") $(grep -ho '"denoise_s":[0-9.]*\|"total_s":[0-9.]*' "$c/stderr.log" 2>/dev/null | tail -2 | tr '\n' ' ')"
    done
# --- end
    ;;
  b6)
# --- verbatim main 2056-2056
    b6_cells="${FV_B6_CELLS:-b6-solh3-tree b6-solh3-merge b6-4stepvsa-merge b6-h3-streamed b6-h3-ref2va b6-ltx25-ref2v b6-ltx25-a2v-guided b6-ltx25-offload-cpu}"
# --- end
    for cell in $b6_cells; do
      for arm in base pr; do
# --- verbatim main 2073-2074
        grep -h 'h3 dit\|h3 text encoder: auto\|h3 dit offload\|h3 i2v encoder\|h3 sol-h3\|h3 fasth3-preview\|h3 lora merge\|ltx2 dit offload\|ltx2 offload\|ltx2 ic-lora\|ltx2 lora\|load/io' \
          "$RUNS/$cell-$arm/stderr.log" 2>/dev/null | sed "s/^/[$cell-$arm] /" | tee -a "$LOG" >/dev/null || true
# --- end
      done
    done
# --- verbatim main 2119-2121
    for f in "$RUNS"/compare/compare-clips-b6-*.json; do
      [[ -f "$f" ]] && log "$(basename "$f"): $(grep -oE '"(status|off_identity|max_abs_diff_uint8)": *[^,}]*' "$f" | tr '\n' ' ')"
    done
# --- end
    ;;
  b4a)
# --- verbatim b4a 569-571
    for f in "$RUNS"/compare/compare-clips-b4a-*.json; do
      [[ -f "$f" ]] && log "$(basename "$f"): $(grep -oE '"(status|off_identity|max_abs_diff_uint8)": *[^,}]*' "$f" | tr '\n' ' ')"
    done
# --- end
    ;;
  h3arms)
    for res in 768p 480p; do
      base="fasth3-8step-$res"
      for arm in sol tea soltea; do
# --- verbatim main 2690-2691
        grep -h "h3 teacache step" "$RUNS/$base-$arm/stderr.log" 2>/dev/null | grep -c REUSE \
          | sed "s/^/[$base-$arm] teacache reused steps (all passes): /" | tee -a "$LOG" || true
# --- end
      done
    done
# --- verbatim main 2725-2727
    for f in "$RUNS"/gate/gate-*.json; do
      [[ -f "$f" ]] && log "$(basename "$f"): $(grep -o '"verdict": "[a-z]*"' "$f" | head -1)"
    done
# --- end
    ;;
  trace)
# --- verbatim main 1761-1763
    for cell in fasth3-4step-vsa-768p fasth3-8step-768p fasth3-4step-dense-768p ltx25-4k5s-sol ltx25-1080p20s-sol; do
      grep -h '/gpu_trace ' "$RUNS/$cell/stderr.log" 2>/dev/null | tail -1 | cut -c1-400 | sed "s/^/[$cell] /" | tee -a "$LOG" || true
    done
# --- end
    ;;
  ltxfps)
# --- verbatim main 2845-2848
    for cell in $(ls "$RUNS" 2>/dev/null | grep -E '^(ltx25-1080p|fasth3-8step-sink)'); do
      grep -h 'PASS\|FAIL' "$RUNS/$cell/stderr.log" 2>/dev/null | grep -E 'sink|mp4_frames|no_wav|gen\.frames' \
        | cut -c1-300 | sed "s/^/[$cell] /" | tee -a "$LOG" || true
    done
# --- end
    ;;
  ltxvae | ltxvae-gen)
# --- verbatim main 2901-2903
    for cell in $(ls "$RUNS" 2>/dev/null | grep -E '^(ltxvae|ltx25-.*-gen)'); do
      grep -h 'vae-bench\|vae_check\|video decode' "$RUNS/$cell/stderr.log" 2>/dev/null | tail -3 | cut -c1-400 | sed "s/^/[$cell] /" | tee -a "$LOG" || true
    done
# --- end
    ;;
  *) ;;
esac
exit 0
