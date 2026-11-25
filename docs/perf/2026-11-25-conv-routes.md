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

## WO-3b increment 1: the tuning runs (2026-11-25, same GPU class, PR #258)

Four more `conv_routes` runs on RTX PRO 6000 pods (each about a minute of
pod time; records beside the spike's under `artifacts/runpod/rtx6000/`).
Ratios are igemm / cuDNN bf16 on the same pod; absolute times move with
the pod (the e78a557 pod ran everything, cuDNN included, 10-40 % slower).

| run | change | wan22-1024 | wan22-512 | wan21-96 | ltx-1024 | ltx-up-4096 | bench-384 (tiny M) |
|---|---|---|---|---|---|---|---|
| a9ee908 (spike) | 128×128, 2 stages, 2 CTAs/SM | 1.28 | 1.24 | 0.94 | 1.79 | 1.35 | 3.27 |
| 9ff8be5 | 3 stages at **1 CTA/SM**, 64 tile + split-K | 2.07 | 1.86 | 1.22 | 2.15 | 2.03 | 1.42 |
| 42aaad2 | 3 stages at 2 CTAs/SM (four variants) | 1.33 | 1.28 | 0.92 | 1.72 | 1.42 | 1.31 |
| 98ab3c6 | + 128×256, 128×64 tiles, four-wave split-K | 1.34 | 1.22 | 0.97 | 1.39 | 1.35 | 1.05 |
| e78a557 | + loader decodes k once per k-tile, paired B ldmatrix | **1.21** | **1.13** | **0.93** | **1.22** | **1.25** | **1.00** |

What the runs settled:

- **Occupancy decides, not tile size.** The 3-stage pipeline at one CTA per
  SM (`__launch_bounds__(256, 1)`, 48 KB smem) lost 60-80 % against the
  spike; the same kernel at two CTAs per SM is the best variant on every
  shape. The 128×256 tile (one CTA per SM, 128 accumulators), the 128×64
  tile and the 64×64 tile all lost to 128×128 on every shape, including
  the narrow-N ones they were meant for. The 64×64 warp tile (4 warps,
  two CTAs per SM) equals the 64×32 one within noise.
- **Split-K fixes the small-M shapes.** Deterministic (one f32 plane per
  split, summed in order by the transpose-back kernel): bench-384 went
  3.27× → 1.00×, ltx-1024 1.79× → 1.22× (four splits), wan-timeconv and
  enc-s2 by two splits. Rule: enough splits for four waves of two CTAs per
  SM, each split at least eight k-tiles, at most 16.
- **The loader's integer divisions were about 10 % of the mainloop.** The
  A-tile producer decoded (tap, channel, kd, kh, kw) per 16-byte chunk;
  decoding the tile's first k once per k-tile and advancing within the
  tap took every shape down 5-12 points. B fragments now load two n-tiles
  per ldmatrix.x4 (as the attention kernels load K).
- **Correctness held at every step:** every run passed every check (the
  routes match the f64 reference at rel_l2 2.3e-3 and cuDNN bf16 at the
  bf16 output rounding), including the split-K planes.

Decode-weighted at e78a557: about 1.12× cuDNN on the Wan decoders, about
1.2× on LTX-2.5. Of the remaining gap on the wide stages, the NCDHW →
NDHWC transpose in, the transpose out and the per-call weight repack are
about 7 % of the route's time and sit outside the kernel: WO-3b increment 2
caches the bf16 weights per layer and gives `vae_fast.rs` (already
channels-last) the NDHWC-native entry, which skips the transposes
entirely. In-kernel, the epilogue's scattered f32 stores and a BK = 64 /
dynamic-smem variant are the next candidates once the real decode gates
say where the wall time goes.

## WO-3b increment 2: the first decode gate (convgate tier, 2026-11-25)

`fv-lab pod up --family convgate` at `6dea13e` (PR #258) on an RTX PRO 6000:
the LTX-2.5 fast decoder at 1080p 20 s (481 frames, synthetic latents, the
real tiles and temporal chunks), cuDNN bf16 NHWC convs (today's path)
against the implicit GEMM through its NDHWC entry (`FASTVIDEO_CONV3D=igemm`,
no transposes, bf16 from the epilogue). Record:
`artifacts/runpod/convgate/6dea13e-1125094821-944c/`.

| | cuDNN bf16 (NHWC plans) | igemm |
|---|---|---|
| decode wall time (481 frames) | 13.17 s | 18.74 s (**1.42×**) |
| peak device memory | 8 834 MiB | 7 042 MiB (no cuDNN workspace) |
| first tile vs the streaming decoder (`FASTVIDEO_LTX_VAE_CHECK`), rel_l2 | 7.2e-3 | 6.8e-3 |
| frames, igemm vs cuDNN: PSNR mean / min | | 55.8 dB / 54.9 dB |
| LPIPS mean / max (44 sampled pairs) | | 1.4e-4 / 1.9e-4 |
| sharpness ratio, temporal jitter ratio | | 1.0000, 1.0000 |
| max abs pixel difference (uint8) | | 6 |

Quality is settled: the two decoders differ by bf16 summation order only
(55.8 dB, LPIPS 1e-4, no sharpness or jitter change), and the igemm arm
sits closer to the f32 streaming decoder than cuDNN does.

Speed is not: 1.42× on the real decode against the 1.2× the micro-benchmark
suggested. The difference is the baseline, not the kernel: the fast
decoder's cuDNN runs NHWC-native plans (no layout change inside cuDNN),
while `conv_routes` compared against `cudnn_conv_bf16` on NCDHW tensors,
which pays a layout change inside cuDNN. Against cuDNN's best kernels the
implicit GEMM is 1.4× behind on LTX's shapes. The owner's margin (proposed
1.10×) is not met; what remains is mainloop work: BK = 64 with dynamic
shared memory (half the barriers per k), the classic 256×128 CTA tile with
64×64 warp tiles, and the epilogue's scattered stores.

## WO-3b option 1: the capped tuning rounds (2026-11-25, PR #258)

After the convgate gate the owner chose to keep tuning the mainloop, capped
at three more rounds, and to decide on the margin after them. Each round is
one `conv_routes` run on an RTX PRO 6000 pod (about a minute of pod time,
900 s wall cap; records under `artifacts/runpod/rtx6000/`). Ratios are
igemm / cuDNN bf16 (NCDHW plans) on the same pod.

### Round 1 (195a944): where the time goes

The kernel loads the whole k-tile's fragments before any mma (two-deep
fragment ring) and gains two timing-only probes, launched on the same grid
as the real kernel and excluded from the correctness checks:

- `probe_dense`: the A loader reads a dense (non-gathered) tile, so the
  gather's address arithmetic and its scattered 16-byte reads are gone,
  the pipeline, barriers and mma unchanged.
- `probe_noload`: no global loads at all (the smem ring is used as is),
  so only the pipeline overhead and the mma remain: the kernel's ceiling.

| shape | igemm | probe_dense | probe_noload | cuDNN ms |
|---|---|---|---|---|
| wan22-1024-s1 | 1.24 | 1.14 | **0.90** | 4.99 |
| wan22-512-s2 | 1.15 | 1.03 | 0.83 | 5.41 |
| wan21-96 | 0.93 | 0.84 | 0.72 | 1.73 |
| ltx-1024 | 1.33 | 1.21 | 0.99 | 1.45 |
| ltx-up-4096 | 1.28 | 1.20 | 0.98 | 5.44 |
| ltx-256 | 1.07 | 0.99 | 0.82 | 3.07 |
| bench-192x6x120x208 | 1.23 | 1.08 | 0.87 | 1.02 |

What the probes say: the gather costs 8-12 %, every other global load a
further 10-20 %, and the no-load kernel, which does nothing but the smem
pipeline and the mma, is at 0.9× cuDNN on the wide Wan shape and 1.0× on
the LTX shapes: about 350 TFLOP/s against the GPU's ~1 PFLOP/s dense bf16
peak. So the limiter is not the memory system but the per-k-tile
overhead (one `__syncthreads` and a full fragment reload every 32 k, at
four warps' worth of mma between them). Round 2 therefore attacks that
overhead directly: a BK = 64 variant (half the barriers per FLOP, 64 KB
dynamic shared memory) and a 256×128 tile with 16 warps (twice the mma per
barrier at the same warps per SM, 72 KB), measured side by side with the
current kernel.

### Round 2 (14300de): BK = 64 and the 256×128 tile

Two variants beside the current kernel (`w32`), record
`artifacts/runpod/rtx6000/14300de-11251423/`, every check passing:

- `bk64`: 128×128×64, two stages, 64 KB dynamic smem. ptxas gives it 184
  registers, so it runs one CTA per SM.
- `m256`: 256×128×32, 16 warps (512 threads), three stages, 72 KB: one
  CTA per SM, the same 16 warps per SM as two base CTAs, twice the mma
  per barrier.

| shape | w32 | bk64 | m256 | probe_dense | probe_noload |
|---|---|---|---|---|---|
| wan22-1024-s1 | **1.34** | 1.63 | 1.64 | 1.15 | 0.92 |
| wan22-512-s2 | **1.23** | 1.47 | 1.51 | 1.01 | 0.84 |
| wan21-96 | **0.92** | 1.11 | 1.09 | 0.84 | 0.73 |
| ltx-1024 | **1.34** | 1.64 | 1.72 | 1.25 | 0.99 |
| ltx-up-4096 | **1.28** | 1.60 | 1.60 | 1.21 | 0.98 |
| ltx-256 | **1.07** | 1.31 | 1.36 | 0.97 | 0.81 |
| bench-192x6x120x208 | **1.24** | 1.49 | 1.55 | 1.09 | 0.88 |

Both lose by 20-30 % on every shape. The barrier count per FLOP is not
the limiter: `m256` has the same warps per SM as the base kernel and half
the barriers, and is the slower of the two. What both share is one CTA
per SM: a `__syncthreads` then stalls all 16 warps of the SM's work at
once, where two independent CTAs cover each other's barrier and pipeline
waits. The round confirms round 1's finding from the other side:
occupancy (two CTAs per SM) is worth more than anything inside the
k-tile.

### Round 3 (bb7ff1f, the last): deeper rings in dynamic smem

The probes' arithmetic for the wide shapes: no loads 0.92, dense loads
1.15, the gather 1.34. The 23 points between no loads and dense loads are
the cost of plain, coalesced cp.async traffic that a ring only two stages
deep (three stages, one being computed) does not hide at 32 k per stage.
Round 3 deepened the ring in dynamic smem: `s4` (four stages, 64 KB per
CTA) and `s6` (six stages, 96 KB), both at two CTAs per SM by
`__launch_bounds__` and 113-114 registers, and `bk64` again held to two
CTAs per SM by registers (126, no spills). Record
`artifacts/runpod/rtx6000/bb7ff1f-11251452/`, every check passing.

| shape | w32 | bk64 | s4 | s6 | probe_dense | probe_noload |
|---|---|---|---|---|---|---|
| wan22-1024-s1 | **1.31** | 1.79 | 1.99 | 1.96 | 1.17 | 0.95 |
| wan22-512-s2 | **1.23** | 1.62 | 1.80 | 1.77 | 1.07 | 0.88 |
| wan21-96 | **0.89** | 1.13 | 1.18 | 1.17 | 0.81 | 0.71 |
| ltx-1024 | **1.28** | 1.70 | 1.87 | 1.85 | 1.18 | 0.98 |
| ltx-up-4096 | **1.34** | 1.77 | 1.96 | 1.94 | 1.23 | 1.03 |
| ltx-256 | **1.12** | 1.49 | 1.65 | 1.63 | 1.02 | 0.86 |
| bench-192x6x120x208 | **1.18** | 1.52 | 1.73 | 1.72 | 1.07 | 0.83 |

All three dynamic-smem variants lose 35-50 %, the deeper rings most, and
their numbers are the one-CTA-per-SM numbers of round 2 again. The
reason is the GPU, not the launch bounds: compute capability 12.0 (the
RTX PRO 6000 Blackwell, GB202) has 128 KB of shared memory per SM and
99 KB per block, not the 228 KB of the data-centre parts. Two CTAs of
64 KB (plus the 1 KB the driver reserves per CTA) do not fit, so every
variant above 63 KB runs one CTA per SM whatever its registers say; the
base kernel's 48 KB static ring is already the deepest ring that keeps
two CTAs resident. On this GPU the design space the three rounds
explored is closed: two CTAs per SM is required (rounds 1-3), two CTAs
per SM caps the ring at three stages of 32 k (round 3), and within that
ring the loads cost 35-40 points over the no-load ceiling (round 1).

### Outcome of the three rounds

| | wan22-1024 | wan22-512 | wan21-96 | ltx-1024 | ltx-up-4096 | ltx-256 |
|---|---|---|---|---|---|---|
| before the rounds (e78a557) | 1.21 | 1.13 | 0.93 | 1.22 | 1.25 | 1.07 |
| after (w32 at bb7ff1f) | 1.31 | 1.23 | 0.89 | 1.28 | 1.34 | 1.12 |

The three rounds found no faster kernel: the base kernel of e78a557
(which round 1's fragment ring did not change within the run-to-run
spread of about 10 % between pods) remains the best, at 1.2-1.3× cuDNN's
NCDHW plans on the micro-benchmark and 1.42× cuDNN's NHWC plans on the
real LTX-2.5 decode. What the rounds bought is the explanation: on
sm_120 the kernel's no-load ceiling is cuDNN's speed, cuDNN hides its
loads almost entirely inside the same 128 KB smem budget, and this
kernel does not. Closing that needs a different mainloop, not a deeper
one: a warp-specialised producer (one warp issuing cp.async or TMA-style
bulk copies while the other warps only compute, as CUTLASS's sm80
"multistage" does with its software pipelining of the fragment loads
across the k-tile boundary), the im2col gather precomputed as offsets
rather than decoded per chunk, or the epilogue's scattered f32 stores
replaced by a staged smem transpose. Each is a WO of its own, outside
the three rounds the owner capped.

## WO-3b-ii: the warp-specialised mainloop (2026-11-25, PR #258)

Owner decision FVID-2026-11-25-conv-igemm-ws-wo: one more work order
(`docs/plan/2026-11-25-adr24-step3-wo.md` "WORK ORDER 3b-ii") before the
margin decision. The kernel `conv_igemm_bf16_ws`: 128×128×32, four
consumer warps (2×2 of 64×64) that never issue a global load, one
producer warp that owns the im2col gather and the weight rows, named
barriers per stage, the 48 KB three-stage ring, two CTAs per SM. ptxas:
168 registers (five warps a CTA, ten an SM, three on one of the four
sub-partitions: 65536 / 3 / 32), 50-120 bytes of spills on every SM
(each branch alone fits without; the union does not).

### Run 1 (a9ae65d): correct, half the base kernel's speed

Record `artifacts/runpod/rtx6000/a9ae65d-11251818/`, every check passing
(the ws kernel matches the f64 reference and cuDNN bf16 exactly as the
base kernel does).

| shape | w32 | ws | ws_probe_noepi | probe_dense | probe_noload |
|---|---|---|---|---|---|
| wan22-1024-s1 | **1.24** | 2.68 | 2.56 | 1.11 | 0.90 |
| wan22-512-s2 | **1.22** | 2.40 | 2.39 | 1.06 | 0.87 |
| wan21-96 | **0.89** | 1.50 | 1.47 | 0.82 | 0.71 |
| ltx-1024 | **1.26** | 2.41 | 2.42 | 1.17 | 0.97 |
| ltx-up-4096 | **1.34** | 2.61 | 2.61 | 1.24 | 1.03 |
| ltx-256 | **1.13** | 2.21 | 2.19 | 1.02 | 0.85 |

The epilogue is not the cost (the no-store probe matches the kernel).
The producer signalled stage kt after issuing kt + 1 and waiting for all
but its newest group: one compute period of latency hiding, where the
base kernel's `wait_group(STAGES − 2)` at the top of each iteration gives
two. Commit 5111c99 signals after issuing kt + 2 (`wait_group 2`, the
tail flushed in two steps) and adds `ws2`, two producer warps (192
threads, 8 + 8 chunks per lane, the same 168-register budget), to
separate issue rate from latency in run 2.

### Run 2 (5111c99): the producer side is the limit

Record `artifacts/runpod/rtx6000/5111c99-11251921/`, every check passing
(the owner waived the balance floor for this run).

| shape | w32 | ws (run 1, lag 1) | ws (lag 2) | ws2 (two producers, lag 2) | probe_dense | probe_noload |
|---|---|---|---|---|---|---|
| wan22-1024-s1 | **1.31** | 2.68 | 3.55 | 1.89 | 1.17 | 0.95 |
| wan22-512-s2 | **1.21** | 2.40 | 3.06 | 1.69 | 1.06 | 0.87 |
| wan21-96 | **0.89** | 1.50 | 1.85 | 1.15 | 0.82 | 0.71 |
| ltx-1024 | **1.32** | 2.41 | 3.15 | 1.82 | 1.23 | 1.01 |
| ltx-up-4096 | **1.34** | 2.61 | 3.30 | 1.86 | 1.24 | 1.02 |
| ltx-256 | **1.12** | 2.21 | 2.79 | 1.57 | 1.02 | 0.85 |

Two producer warps run the kernel in half the time of one (ws2 vs ws,
same signalling), and signalling later (more latency hiding) made the
one-producer kernel slower, not faster. So the warp-specialised kernel
is not latency-bound; it is bound by how fast a warp can issue the
gathered `cp.async` traffic: a producer warp's 32 copies per k-tile (16 B
per lane, eight 64-byte row segments per instruction) serialise in that
warp's memory pipeline, and the time per k-tile scales with the number
of warps that share the issuing. The base kernel already spreads the
same copies over its eight warps; that is why it is the fastest
arrangement found.

Why there is no room to add producers: with 64×64 consumer tiles (128
accumulators) a CTA can hold two producer warps at two CTAs per SM
(twelve warps per SM, 168 registers); a third puts four warps on a
sub-partition (128 registers) and spills the accumulators; eight 64×32
consumers (the base kernel's warps) plus P producers must fit twenty
warps per SM at about 100 registers, so P ≤ 2 again. One CTA per SM
would allow more producers but halves the computing warps per SM, which
every one-CTA variant of the tuning rounds already showed losing.

### Outcome: the kill criterion applies

The WO's stop rule (above 1.15× on the carrying shapes once correct at
two CTAs per SM): ws2 is at 1.7-1.9×, the base kernel at 1.2-1.3×, and
the measurements say the remaining cost on this GPU is the memory
pipeline's handling of 64-byte row segments (half a 128-byte line per
`cp.async` row at BK = 32), which only a BK = 64 tile (128-byte
segments) would change; at two CTAs per SM that tile does not fit the
128 KB of shared memory of compute capability 12.0 (round 2 and 3). No
`convgate` run. The implicit GEMM stays at about 1.2-1.3× cuDNN's NCDHW
plans on the micro-benchmark and 1.42× cuDNN's NHWC plans on the LTX-2.5
decode, correct and quality-gated; the margin decision returns to the
owner with the three options of 2026-11-25 (accept the slowdown on the
static image only; park step 3; a datacentre-only kernel as WP-G).

One in-budget idea the rounds did not try, for the record: a
128×64×64 two-stage tile (A 16 KB + B 8 KB per stage, 48 KB) keeps two
CTAs per SM with 128-byte row segments at the cost of half the N reuse
and a two-stage ring (the spike's two-stage kernel was within 5 % of the
three-stage one).
