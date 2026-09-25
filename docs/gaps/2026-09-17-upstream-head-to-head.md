# Upstream head-to-head on one box

Date: 2026-09-17 (measured), closed 2026-09-18
Compares: this tree vs hao-ai-lab/FastVideo running the same weights and clip on the **same rented GPU**.
Decision-log entries: `FVID-2026-09-17-upstream-head-to-head`, `FVID-2026-09-17-exact-parity-24gb-oom`, `FVID-2026-09-18-vsa-port`, `FVID-2026-09-18-vae-decode`.

## Method

A `compare` tier (T4) in `validate.sh` runs our clip stages, then installs
upstream FastVideo in its own uv venv on the same instance and times the same
8 s clip. Identical hardware by construction rather than by matching model
names. Model load is timed separately and excluded; one warm-up, then median
of 2. The tier skips T1/T2 (it benchmarks, it does not re-validate).

Workload: RTX 3090 Ti, 448x832, 129 frames, 3 DMD steps, timesteps
1000/757/522, FastWan2.1-T2V-1.3B weights. Upstream torch 2.12.0+cu126,
fastvideo 0.2.1. Run `20260917T220533Z-compare`.

## Result

| RTX 3090 Ti | Ours (dense) | Upstream (VSA) |
|---|---:|---:|
| Generation | ~108.4 s | **55.3 s** |
| - denoise | 82.7 s | - |
| - VAE decode | 21.9 s | - |
| - write | 3.8 s | - |
| Model load (excluded) | 12 s | 49.9 s |
| First run (warm-up) | 82.7 s | 95.9 s |

**Upstream 1.96x faster.** The gap was the optimization we had deliberately
not ported: Video Sparse Attention cuts the quadratic attention term, and our
48k-token forward was dominated by dense attention (the flash experiment had
shown the same thing from the other side).

Two details for reading it honestly:

- Upstream's first generation took 95.9 s against 55.3 s steady-state because
  Triton compiles on first use. A single-shot benchmark would have called
  upstream slower than us.
- The like-for-like dense number does not exist on upstream's side.
  `TORCH_SDPA` cannot load this checkpoint (`Parameter
  blocks.0.to_gate_compress.bias not found`) because FastWan ships VSA gate
  weights their dense model class does not define. Upstream on this model
  *is* the VSA configuration.

Nine attempts were needed to get the number; the failures were ours (bad-host
recording on mid-run ssh death, disk precheck crediting fetched data,
empty-array expansion under bash 3.2, exact-mode OOM on 24 GB cards). The
exact-parity OOM was recorded and routed around, not fixed.

## How the gap was closed (2026-09-18)

VSA was ported by reading upstream's implementation: `(4,4,4)` tiles of 64
slots, coarse stage attends over tile means, top-k picks the tiles the fine
stage attends at full resolution, `out = coarse * to_gate_compress + sparse`.
The fine stage gathers each query tile's selected K/V into a dense bf16 buffer
and runs batched cuBLAS GEMMs rather than a fused kernel. The coarse stage is
pinned to f32 because tile selection is discrete and bf16 rounding flipped a
near-tie.

| RTX 3090 Ti, 8 s clip | Dense | VSA |
|---|---:|---:|
| Denoise | 82.7 s | **45.6 s** (1.81x) |
| Per step | 27.6 s | 15.1 s |

Scaling matched the algorithm: 2x slower at 1,456 tokens (fixed overhead),
parity at 4,368, 16% faster at 13,104, 1.81x at 48,048.

After VSA, generation was ~71 s (45.6 denoise + 21.9 VAE + 3.8 write) against
upstream's 55.3 s. VAE decode became the largest remaining single cost, which
led to the SiLU-into-RMSNorm fold (-10.8%) and chunked decode (-8.6% at
chunk=2; chunk=4 OOMs on 24 GB) the same day, and later to TAEHV.

## Open at the time

- VSA output is a different sample with visibly more saturated colour
  (`clipped_fraction` 0.0218 vs 0.008 dense); never verified against
  upstream's own VSA frames.
- A fused block-sparse kernel would remove the gather. (Measured the same
  day, `FVID-2026-09-18-fused-block-sparse-rejected`: the scalar fused kernel
  was 5.5-8.6x slower than the gather; the gather path stayed, and the fine
  stage moved to `mma.sync` tensor cores on 2026-09-19.)
- Exact-mode parity OOM on plain 24 GB cards; suspects are the mempool release
  threshold (`u64::MAX`) and cuDNN conv3d workspace.
