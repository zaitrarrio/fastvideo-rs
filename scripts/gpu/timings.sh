#!/usr/bin/env bash
# timings.sh <run-dir>: generation time, accuracy and mp4 for every test in a run.
# Reads <run-dir>/remote/*.json (fv-gpucheck stage reports) and writes
# <run-dir>/timings.md (also printed). Called by validate.sh after each run.
set -euo pipefail

run_dir="${1:?usage: timings.sh <run-dir>}"
reports="$run_dir/remote"
out="$run_dir/timings.md"
command -v jq >/dev/null || { echo "timings.sh needs jq" >&2; exit 2; }
compgen -G "$reports/*.json" >/dev/null || { echo "no stage reports in $reports" >&2; exit 1; }

{
  echo "# Test timings — $(basename "$run_dir")"
  echo
  if [[ -f "$run_dir/summary.json" ]]; then
    jq -r '"Run **\(.status)** in \(.wall_minutes) min, ~$\(.est_cost_usd) at $\(.dph)/hr (instance \(.instance))."' "$run_dir/summary.json"
    echo
  fi
  if compgen -G "$reports/env.json" >/dev/null; then
    jq -r '"GPU: \(.gpu) (\(.vram_mib) MiB, sm \(.compute_cap), driver CUDA \(.driver_cuda)), \(.cores) cores, \(.ram_gb) GB RAM."' "$reports/env.json"
    echo
  fi

  echo "## Stages"
  echo
  echo "| stage | status | wall s |"
  echo "|---|---|---:|"
  for f in "$reports"/*.json; do
    jq -r 'select(.stage != null) | "| \(.stage) | \(.status) | \(.elapsed_s * 10 | round / 10) |"' "$f" 2>/dev/null || true
  done

  echo
  echo "## Generated outputs"
  echo
  echo "Time is end-to-end generation on the stage's device (sampler videos: denoise + VAE decode). GPU rows compare against the CPU-path reference."
  echo
  echo "| stage | output | seconds | rel_l2 | PSNR dB | mp4 |"
  echo "|---|---|---:|---:|---:|---|"
  for f in "$reports"/*.json; do
    jq -r '
      select(.stage != null and (.stage | test("^(model|parity)-"))) as $r
      | ($r.checks | map(select(.name | startswith("video/"))) | map({key: (.name | ltrimstr("video/")), value: .values}) | from_entries) as $videos
      | $r.checks[]
      | select(.values.seconds != null)
      | (.name | sub("/finite$"; "")) as $name
      | ($videos[$name] // {}) as $v
      | "| \($r.stage) | \($name) | \(.values.seconds * 1000 | round / 1000) | \(if .values.rel_l2 == null then "—" else (.values.rel_l2 | tostring) end) | \(if .values.psnr_db == null then "—" else (.values.psnr_db | tostring | .[0:6]) end) | \(if $v.mp4 then ($v.mp4 | split("/gpucheck-out/") | last) else (if $v.error then "encode failed" else "—" end) end) |"
    ' "$f" 2>/dev/null || true
  done

  if compgen -G "$reports/clip-*.json" >/dev/null; then
    echo
    echo "## Clips"
    echo
    echo "| clip | frames | resolution | steps | denoise s | per step s | VAE s | total s | s per video-second | peak MiB | mp4 |"
    echo "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"
    for f in "$reports"/clip-*.json; do
      jq -r '
        .context as $c
        | "| \(.stage) | \($c.spec.frames // "—") | \($c.spec.width // "?")x\($c.spec.height // "?") | \($c.spec.steps // "—") | \($c.timings.denoise_s // "—" | tostring | .[0:7]) | \($c.timings.per_step_s // "—" | tostring | .[0:7]) | \($c.timings.vae_decode_s // "—" | tostring | .[0:7]) | \($c.timings.total_s // "—" | tostring | .[0:7]) | \($c.timings.seconds_per_video_second // "—" | tostring | .[0:6]) | \($c.timings.peak_mib.denoise // "—") | \(if $c.artifacts.mp4 then ($c.artifacts.mp4 | split("/gpucheck-out/") | last) else "—" end) |"
      ' "$f" 2>/dev/null || true
    done
  fi

  if compgen -G "$reports/probe-*.json" >/dev/null; then
    echo
    echo "## Projections"
    echo
    for f in "$reports"/probe-*.json; do
      jq -r '"- \(.stage): \(.status); projected total \(.context.projection.total_s // "?" | tostring | .[0:7]) s (DiT forward \(.context.projection.dit_forward_s // "?" | tostring | .[0:7]) s at \(.context.projection.target_tokens // "?") tokens), peak \(.context.projection.peak_mib // "?" | tostring | .[0:7]) MiB"' "$f" 2>/dev/null || true
    done
  fi
} >"$out"

cat "$out"
