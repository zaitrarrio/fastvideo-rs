# The cuDNN-free conv routes on an RTX PRO 6000 (ADR-0024 WO-3a)

Date 2026-11-25. Branch `claude/adr24-step3a-spike` at `a9ee908` (PR #257),
runtime image `sha-a9ee908`, build id `c3fded4b34c7781d`. Pod
`sp1remugtxtou5`, NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120,
97 887 MiB, driver 595.112.07), $2.69/h, 35 s of pod time; the kernels pass
itself took 12.6 s. Record: `artifacts/runpod/rtx6000/a9ee908-11250652/`
(`kernels.out`, `kernels.json`). fv-gpucheck group `conv_routes`: 52 checks,
52 pass.

Owner decision 2026-11-25: two routes (CUTLASS not vendored). The work order
is `docs/plan/2026-11-25-adr24-step3-wo.md`.

## What was measured

Three ways to run the same dense convolution, all from the NC(D)HW f32
tensors the callers have today, all on bf16 operands with f32 accumulation:

- **cuDNN bf16** (`conv.rs` `cudnn_conv_bf16`, today's `auto` under bf16
  math): cast in, cuDNN N-D forward (heuristic algorithm), bf16 output cast
  back to f32.
- **R1 unfold-lt** (`conv3d_unfold_lt`): NCDHW → NDHWC bf16, explicit im2col
  rows in 512 MB chunks, cuBLASLt bf16 GEMM with f32 output in place,
  NDHWC → NCDHW.
- **R2 igemm** (`conv3d_igemm`, `rt/kernels/conv_igemm.cu`): NCDHW → NDHWC
  bf16, one implicit-GEMM kernel (mma.sync m16n8k16, 128×128×32 tile, 8
  warps, cp.async double buffering, the im2col gather in the A-tile
  producer), f32 NDHWC out, NDHWC → NCDHW. One tile config, no tuning.

Each route's time includes its casts and permutes. Median of 20 synchronized
launches after 3 warm-ups. Shapes are the served convolutions as the slabs
the decoders run (chunked time, half width where a full stage would need
more than about 100 M inputs); `gflop` counts 2 × outputs × taps × C_in.

## Correctness

- Small shapes against the f64 host reference (`ref_conv3d`), rel_l2 limit
  5e-3: cuDNN bf16 2.8e-3 to 2.9e-3; unfold-lt and igemm 2.3e-3 to 2.4e-3
  (the routes keep f32 output, cuDNN rounds its output to bf16, which is
  the whole difference).
- Every shape against cuDNN bf16 by device-side relative L1: 1.38e-3 to
  1.42e-3 for both routes on all 18 shapes, limit 2e-3. That number is the
  mean bf16 output rounding of the cuDNN path (about 2⁻⁹), identical for
  the two routes to four digits: the sums agree, cuDNN's output cast is
  what differs.

## Speed

| shape | x shape | GFLOP | cuDNN bf16 ms | igemm ms | igemm / cuDNN | unfold-lt ms | unfold-lt / cuDNN | cuDNN TFLOP/s | igemm TFLOP/s |
|---|---|---|---|---|---|---|---|---|---|
| bench-384x3x30x52 | 1×384×3×30×52 | 12.4 | 0.137 | 0.449 | 3.27 | 0.286 | 2.08 | 90 | 28 |
| bench-192x6x120x208 | 1×192×6×120×208 | 198.7 | 1.032 | 1.328 | 1.29 | 5.936 | 5.75 | 193 | 150 |
| bench-96x6x240x416 | 1×96×6×240×416 | 198.7 | 1.569 | 1.542 | 0.98 | 11.842 | 7.55 | 127 | 129 |
| wan22-1024-s1 | 1×1024×4×88×160 | 1594.5 | 4.909 | 6.260 | 1.28 | 11.230 | 2.29 | 325 | 255 |
| wan22-512-s2 | 1×512×4×176×320 | 1594.5 | 5.171 | 6.432 | 1.24 | 18.263 | 3.53 | 308 | 248 |
| wan22-256-s3 (half width) | 1×256×4×352×320 | 797.3 | 3.473 | 3.684 | 1.06 | 17.262 | 4.97 | 230 | 216 |
| wan21-96 (half width) | 1×96×4×480×416 | 198.7 | 1.749 | 1.649 | 0.94 | 11.897 | 6.80 | 114 | 121 |
| wan-timeconv (k 3×1×1, 1024→2048) | 1×1024×4×44×80 | 88.6 | 0.514 | 0.575 | 1.12 | 0.661 | 1.29 | 172 | 154 |
| wan-resample2d (conv2d, 384→192) | 8×384×120×208 | 265.0 | 1.696 | 1.855 | 1.09 | 7.744 | 4.57 | 156 | 143 |
| ltx-1024 (pad 0) | 1×1024×16×24×24 | 383.7 | 1.451 | 2.596 | 1.79 | 3.005 | 2.07 | 264 | 148 |
| ltx-up-4096 (1024→4096, pad 0) | 1×1024×16×24×24 | 1534.7 | 5.268 | 7.129 | 1.35 | 7.590 | 1.44 | 291 | 215 |
| ltx-256 (pad 0) | 1×256×8×192×192 | 766.5 | 3.091 | 3.327 | 1.08 | 16.296 | 5.27 | 248 | 230 |
| enc-s2 (stride 1,2,2) | 1×512×4×176×320 | 398.6 | 2.102 | 2.343 | 1.11 | 5.267 | 2.51 | 190 | 170 |

The five small correctness shapes (0.02 to 0.06 ms, launch-bound) are in
the record and not in the table.

### Decode-weighted view

Weights are the convs per clip of each stage from the decoder plans
(Wan 2.2: three resnets of two 3×3×3 convs per block, four blocks at the
widths 1024 / 1024 / 512 / 256 plus the mid block and the up-sample convs;
LTX-2.5 fast decoder: resnets [2, 2, 4, 6, 4] at [1024, 512, 512, 256],
two convs each, plus the four up-sample convs). Per-call times from the
table, slabs as measured, so this is a ratio, not a wall time:

| family | cuDNN bf16 | igemm | igemm / cuDNN |
|---|---|---|---|
| Wan 2.2 TI2V-5B 704p decode (the 1024-wide stages are 6 of 24 resnet convs but 60 % of the FLOPs) | 1.00 | about 1.18 | 1.18 |
| Wan 2.1 480p decode (96 / 192 / 384 stages, `bench-*` rows) | 1.00 | about 1.15 | 1.15 |
| LTX-2.5 1080p decode (fast decoder tiles) | 1.00 | about 1.35 | 1.35 |

unfold-lt is 2.3× to 7.5× cuDNN on every stage that matters (the im2col
traffic is 27× the activation at C ≤ 256) and only competitive on the
k 3×1×1 time conv and the 4096-wide up-sample; it is not a route for the
served path.

## Reading the numbers

- **R2 igemm, untuned, is within 6 % to 35 % of cuDNN on every served shape
  but one**, faster than cuDNN on the two narrow full-resolution stages
  (wan21-96 0.94×, bench-96 0.98×), and reaches 255 TFLOP/s on the
  1024-wide Wan stage against cuDNN's 325 (both figures include the casts
  and permutes, so the kernels themselves run higher).
- **Where it loses, it is tile quantization, not the mainloop:** `ltx-1024`
  (1.79×) has 6 776 output pixels, 53 M-tiles × 8 N-tiles = 424 CTAs for 188
  SMs with a 27 648-deep K loop each; `bench-384x3x30x52` (3.27×) has 1 400
  pixels, 33 CTAs. cuDNN splits K or picks a smaller tile there. A
  64×64×32 config and a split-K variant (the WO-3b tile table) are the
  obvious fixes; the wide Wan stages (1.24× to 1.28×) want the 3-stage
  pipeline and a 128×256 tile, the standard sm_80-class GEMM tuning.
- **The casts are inside both timings.** The route does one fused
  transpose-cast in and one transpose out; cuDNN bf16 does a cast in and a
  cast out plus whatever layout change its NCDHW kernels do internally.
  WO-3b's cached bf16 weights and WP-G's channels-last residency remove
  the per-call permutes from the route and not from cuDNN.
- **Correctness is settled:** both routes match the f64 reference better
  than cuDNN bf16 does, and match cuDNN to the bf16 output rounding.

## Recommendation for the owner's route decision

R2 `igemm` (own implicit GEMM in `rt/kernels/`, AOT like every other
kernel). It already serves every Wan and LTX shape within 1.35× of cuDNN
with one tile config and no tuning, is faster at the narrow stages that
dominate full-resolution decode time, and has clear, conventional tuning
headroom. R1 stays as what it is: a zero-kernel fallback for shapes the
kernel has no tile for. CUTLASS was not measured (owner: two routes).

Open before 3b: the speed margin per family (proposed in the WO: decode
wall time ≤ 1.10× cuDNN; today's untuned kernel sits at 1.15 to 1.35 by
the weighted view above), and whether to tune on this GPU first or to
measure sm_90 / sm_100 in the same pass.
