# Determinism

Promise: **the same request with the same seed gives the same bytes**, in one
process and in any other process on the same GPU type, driver and image. The
frames, the audio and every latent in between. Decision
`FVID-2026-09-29-determinism` in `decision-log.md`.

It did not hold at 1080p. LTX 1080p output differed between pod boots (33.6 to
34.3 dB PSNR, whatever the RoPE path) and even between two I2V jobs of one boot
(35.6 dB), while 720p was bit-identical (datacenter-profile.md "WP-F results").

## Sources found and what fixed them

| Source | Where | Effect | Fix |
|---|---|---|---|
| **Timed dense-attention pick** | `wan/attn.rs` `auto_cudnn_or_v2` (sm_12x, bf16 out) | The first call of each `(bh, sq, sk, d)` in a process timed cuDNN's SDPA against `flash_mma_fwd2` and kept the faster. The two round 1-2 bf16 ulp apart, and at shapes where they tie (every text or audio cross-attention: 0.63 vs 0.63 ms) the winner changed from boot to boot. Logged picks of the same shape went both ways on one day (Wan 720p `sk=512`: cuDNN in four runs, fwd2 in two) | A fixed rule per (arch, shape class): `wan/sdpa_rule.rs`. The timing lives on as an offline tool that prints the table |
| **Conditioning cache miss vs hit** | `ltx2/pipeline.rs` text encode | A fresh encode handed the DiT the connectors' output as it came (bf16 storage with bf16 activations); a cache hit handed it host f32. The first job of a new prompt and the next job of the same prompt could differ (the two I2V jobs of a boot: a miss and a hit) | One representation in every case: f32 values uploaded as an f32 device tensor (`canonical_contexts`) |
| **H3 `auto` text encoder after its release** | `h3/pipeline.rs`, `h3/text.rs` | `auto` starts resident at weight-only FP8 when the card is empty, and releases the encoder before a denoise that needs the memory (1080p). Later prompts then streamed at bf16, a different function: a prompt's conditioning depended on what the process had run before, and the cache key did not tell the two apart | Without a resident encoder, stream at the numbers the resolved choice stands for (`StreamedFp8Encoder`, the resident FP8 numbers bit for bit); FP8 cache entries are keyed apart |
| **Float atomics in reductions** | `abs_diff_sum` (TeaCache distance), `ltx_abs_diff_sums` (FBCache distance), `attn_fp8_colsum_*` (FP8 attention's K smoothing) | Per-block partials added with `atomicAdd` in whatever order blocks finished. The cache distances decide skips at a threshold, so a last-bit change can flip a step | Per-block partials, then a second pass in block order (on the host for the two distances, `attn_fp8_colsum_reduce` on the device) |
| **Float atomics in SageAttention2's K smoothing** | `attn_sage_colsum_bf16` (`wan/attn_sage.cu`; on by default for `ltx-pro` on sm_120) | The per-head column sums of K (subtracted before the INT8 quantization) were added with `atomicAdd` per block, so a last-bit change in the mean could move an INT8 code and the output | Per-block partials, then `attn_sage_colsum_reduce` adds them in block order, as for FP8 attention. `attn_sage_vmax` keeps `atomicMax` on the float bits, which is order-independent |
| **cuDNN nondeterministic engines** | `wan/cudnn_sdpa.rs` plan build; `wan/conv.rs` transposed conv | The heuristic may rank an engine that declares `CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC` first; backward-data `ALGO_0` (transposed conv: the audio VAE and vocoder upsamplers) accumulates with atomics | SDPA configs with that note are skipped (heuristic order otherwise kept); `ALGO_0` becomes `ALGO_1` |

Checked and found deterministic, no change:

- **cuDNN convolution forward** (VAE conv2d / conv3d, `vae_fast.rs`): the
  algorithm comes from `cudnnGetConvolutionForwardAlgorithm_v7` (a static
  heuristic, the same answer for the same descriptors, GPU and cuDNN), and
  every forward algorithm is deterministic. The per-shape timed backends
  (`FASTVIDEO_CONV3D_TIMED=1`, `FASTVIDEO_LTX_VAE_CONV_ALGO=tune`) were
  already opt-in: the "LTX conv3d fix". The 1080p runs below re-verify it.
- **cuBLAS / cuBLASLt**: one handle on one stream, a fixed workspace; the
  Lt heuristics are asked with a fixed workspace size. cuBLAS guarantees the
  same bits for the same call on the same architecture and SM count.
- **Split-K and KV splits**: Sol's KV splits (`sol_splits_for`) depend only
  on the grid and the SM count, and are combined by a kernel in a fixed order.
  VSA's top-k and all block reductions use fixed trees.
- **Scatter**: `atomicMax` on the FP8 amax and `atomicOr` on the VSA tile
  bitmaps are order-independent; no other atomics exist in our kernels.
- **Host threads**: rayon only splits outputs (`par_chunks_mut`), never a
  float reduction; the RoPE LUT build is sequential; the relay threads move
  bytes.
- **Anything seeded from time**: only a temporary file name
  (`i2v_encode.rs`). Noise comes from `StdRng::seed_from_u64(seed)`.
- **Free-memory decisions**: text residency (LTX two-stage always streams),
  DiT offload, decode placement: they move where weights live, not the
  arithmetic, except the H3 `auto` encoder above. What remains: `auto`'s
  FP8-or-bf16 choice is made once at load from the free memory then (an
  empty card: FP8); `--text-encoder resident-fp8|streamed` pins it.

## The dense-attention rule

`wan/sdpa_rule.rs` `RULES`, first match wins, else `flash_mma_fwd2`:

| Arch | Head dim | Queries | Keys | Kernel |
|---|---|---|---|---|
| sm_12x | 128 | >= 24 576 | 24 576 to 128 000 | cuDNN unified SDPA (heuristic mode A, first deterministic config) |
| any other | | | | fwd2 (sm_90 / sm_100 take the datacenter kernel before this rule) |

Measured (RTX PRO 6000, cuDNN 9.26, median of three synchronized calls):
cuDNN is 3-10 % faster on self-attention from 27k to 124k tokens, fwd2 1 %
faster at 23.6k (24 heads) and 1.2 % at 130k (LTX 4K 5 s), and the two tie on
every cross-attention (keys <= a few thousand). The rule's cost against the
faster kernel: see "Results" below (end to end against the timed pick;
the per-shape `sdpa_rule` kernels group is the offline tool).

### Where SageAttention2 sits

The dense path in `wan/nn.rs` decides in this order, and every step is a
pure function of the recipe, the device and the shape (no timing):

1. **Sage** (`wan/attn_sage.rs`) when it is enabled for this request and
   both the query and key lengths are at least `FASTVIDEO_ATTN_SAGE_MIN_SEQ`
   (6 144). Enabled means: `FASTVIDEO_ATTN_SAGE=2` (any sm_89+), or unset
   and the recipe opts in on sm_120 (today `ltx25-distill-dense`, the
   `ltx-pro` two-stage route, on by default), never with
   `FASTVIDEO_ATTN_SAGE=0`. Its kernel has one variant and fixed reductions.
2. **The kernel seam** (`[kernels] dense_attention`, else
   `FASTVIDEO_FLASH_KERNEL`) when it names a kernel.
3. **`auto`**: Dc on sm_90 / sm_100; on sm_12x with bf16 out, the rule
   table above; else fwd2 when its grid fills the GPU (a function of the SM
   count), V1 otherwise.

So under `ltx-pro` on sm_120, the long self-attention runs Sage and the
cross-attention (text / audio keys, below 6 144) falls to the rule. The
rule does not change Sage's default.

Regenerating it (offline, e.g. for a new GPU or cuDNN):

```
FV_KERNEL_GROUPS=sdpa_rule fv-gpucheck --mode fast kernels
# or on a pod: FV_EXTRA_ENV="FV_KERNEL_GROUPS=sdpa_rule" scripts/gpu/runpod-http.sh kernels <sha>
```

Per shape it reports both times, the faster kernel, the rule's pick and the
rule's loss; `sdpa_rule_table` lists the shapes where the rule loses more than
`FV_SDPA_RULE_TOL` (2 %). Move the boundaries in `RULES` accordingly; the
rule stays a pure function of the shape.

## Overrides (all kept)

| Setting | Effect |
|---|---|
| `FASTVIDEO_FLASH_KERNEL=v1\|v2\|v3\|v3s\|cudnn\|dc` | fixes the dense kernel outright |
| `FASTVIDEO_ATTN_SAGE=0\|2`, `FASTVIDEO_ATTN_SAGE_MIN_SEQ` | Sage off everywhere / on everywhere (sm_89+); its sequence threshold (default 6 144) |
| `FASTVIDEO_SDPA_AUTO=timed` | the old per-process timing on sm_12x (not reproducible across processes) |
| `FASTVIDEO_CONV3D=cudnn\|unfold\|cudnn-bf16`, `FASTVIDEO_CONV3D_TIMED=1` | 3-D conv backend; per-shape timing (not reproducible) |
| `FASTVIDEO_LTX_VAE_CONV_ALGO=tune` | LTX VAE conv algorithms by timing (not reproducible) |
| `FASTVIDEO_CUDNN_SDPA_NONDETERMINISTIC=1` | allows cuDNN SDPA engines that declare themselves nondeterministic |
| `FASTVIDEO_CONV_BWD_DATA_NONDETERMINISTIC=1` | keeps cuDNN's `ALGO_0` for transposed convs |
| `FASTVIDEO_CUDNN_SDPA_CFG`, `FASTVIDEO_CUDNN_SDPA_GRAPH` | cuDNN SDPA engine config / graph (unchanged) |

## Tests

- Host: `crates/fastvideo-cudarc/tests/determinism.rs`. The seeded noise
  (LTX token-major draws and the stage-2 renoise, H3 rows), the RoPE LUTs at
  the 1080p stage-1 / stage-2 grids and with a keyframe block, the distilled
  sigmas, the SDPA rule over a shape sweep and the ordered reduction pass are
  digested twice in one process and once in a child process (the test binary
  re-run): all three digests equal. A conditioning-cache round trip is bit
  exact. `sdpa_rule` unit tests pin the measured shapes; `h3::text` checks
  the streamed-FP8 encoder against the resident FP8 one bit for bit.
- GPU, kernels: `fv-gpucheck kernels --groups determinism` runs the dense
  SDPA under `auto` (a cuDNN shape and a fwd2 shape), both cache distances,
  a transposed conv and FP8 attention three times each; every repeat must
  equal the first bit for bit.
- GPU, end to end: `runpod-matrix.sh determinism` (below).

## Results

### 2026-10-06: RTX PRO 6000, short configs, three fresh processes each

One RTX PRO 6000 Blackwell Server Edition pod (`8opjl8w358z3tn`, EUR-IS-1,
driver 595.91.07, $2.09/hr) on the EU weight volume `jg48s6o1w0` (read
only), image `fastvideo-rs-runtime:sha-6ec6c64` (this branch: the rule,
the ordered reductions, Sage's ordered column sums). Family
`runpod-matrix.sh det-short`: each arm three times, every run a new process
with its own caches (no text, AdaLN or conditioning cache shared between
runs), plus one `FASTVIDEO_SDPA_AUTO=timed` process per bf16 arm (the old
pick) as the speed baseline. Hash = sha256 over every frame PNG in order;
audio = sha256 of `audio.wav`. Logs and hashes:
`artifacts/runpod/det-short/6ec6c64-10061138/`.

| arm | config | r1 / r2 / r3 | timed control |
|---|---|---|---|
| H3 | FastH3 4-step dense, 768x1344, 5 s (124 frames), seed 7, resident FP8 text | **identical** | identical (made the rule's picks this time) |
| Wan 5B | FastWan2.2 TI2V-5B, 3 steps, 704x1280x121, seed 7, full VAE | **identical** | **different** (cuDNN on the 512-key cross-attention: 0.61 vs 0.63 ms) |
| LTX dense | LTX-2.5 distilled two-stage, dense stage 2, bf16 attention (`FASTVIDEO_ATTN_SAGE=0`), 1088x1920x97, seed 7 | **identical** (frames and audio) | **different** (cuDNN on the 1 024-key cross-attention: 1.41 vs 1.42 ms) |
| LTX + Sage | the same with SageAttention2 (`FASTVIDEO_ATTN_SAGE=2`, what `ltx-pro` serves on sm_120) | **identical** (frames and audio) | (not run) |

Reference hashes, **RTX PRO 6000 (sm_120) only**; another GPU type, driver
or cuDNN gives other bytes (cuBLAS and cuDNN pick per architecture and SM
count):

| arm | frames sha256 | audio sha256 |
|---|---|---|
| H3 | `94a46573fba4fe21f2ab165970cea598f74839bcc772c757fa0e978c0e04fef1` | `cd01c4e4e74e236ccf93ad4daf07031eb1ff510bfcd80b4685556f482e3789ab` |
| Wan 5B | `1c95fa6fd705130d33cccf2dd779c59e761045f0c4f71b3091b1323e48c0aefa` | (none) |
| LTX dense | `8a9e1a65941eac5c52d28ba0114aa3e2091d5ff24d56141c9f161c3afd191b2e` | `ee31308f8fd3164d7cb9a8340118edd51fd5bb0c7a43c3c9eb175a12076c6a90` |
| LTX + Sage | `9eb5028c255ec9386fa25169d994e99b4fa1f9cbdcf91e7cce28175b4ea8afac` | `eb8b2b2327b73377abf4d3e8a842f2760424859b57e5e6ab50972941395f2215` |

The H200 (sm_90) LTX reference stays `ad76ebfc…` for the serverless fox
request (docs/gaps/2026-09-27-cold-start.md, the conv3d fix); it is a
different GPU, request and route and was not re-run here. This branch also
changes the LTX conditioning to one f32 form for a cache miss and a hit, so
an H200 re-run may move that hash once; the next H200 session should record
the new value.

**Picks the rule made** (logged once per shape):

| arm | shape (bh x sq x sk, d) | rule | timed control measured |
|---|---|---|---|
| H3 | 56 x 37 751 x 37 751, 128 | cuDNN | cuDNN 102.96 / fwd2 109.52 ms |
| H3 (audio) | 224 x 1 797 x 1 797, 64 | fwd2 | cuDNN 0.65 / fwd2 0.64 ms |
| Wan 5B | 24 x 27 280 x 27 280, 128 | cuDNN | cuDNN 23.66 / fwd2 24.96 ms |
| Wan 5B | 24 x 27 280 x 512, 128 | fwd2 | cuDNN 0.61 / fwd2 0.63 ms -> cuDNN |
| LTX stage 2 | 32 x 26 520 x 26 520, 128 | cuDNN | cuDNN 28.35 / fwd2 30.70 ms |
| LTX | 32 x 26 520 x 1 024, 128 | fwd2 | cuDNN 1.41 / fwd2 1.42 ms -> cuDNN |
| LTX (audio) | 32 x 26 520 x 101, 64 | fwd2 | cuDNN 0.25 / fwd2 0.20 ms |

LTX stage 1 (6 630 tokens) does not fill fwd2's grid and runs V1, a fixed
pick. Under Sage, both stages' video self-attention (6 630 and 26 520
tokens) runs `attn_sage`; only the cross-attention reaches the rule.

**Speed cost of the fixed rule** against the timed pick (denoise, s):

| arm | rule r1 / r2 / r3 | timed | steps after the first, rule vs timed |
|---|---|---|---|
| H3 | 31.71 / 31.74 / 31.73 | 31.96 | 7.86-7.91 vs 7.89-7.90 per step |
| Wan 5B | 4.72 / 4.72 / 4.73 | 4.96 | 1.465-1.473 vs 1.467-1.473 per step |
| LTX dense | 19.67 / 19.39 / 19.94 | 20.08 | |
| LTX + Sage | 17.88 / 17.74 / 17.89 | | |

None measurable: where the timed pick differs from the rule it is a 0.01-0.02
ms tie, and the timed process pays its first-call timing (0.2-0.25 s at
step 1). Sage is 1.10x faster on LTX denoise here, as in sage-attention.md §9.

**LTX conv3d fix, re-verified.** No LTX run logged a `conv3d x=…` timing
line (the timed backend stays behind `FASTVIDEO_CONV3D_TIMED=1`), and the
latent upsampler's convolutions (`ltx2 upsample`, the first 128 -> 1024 conv
of the old H200 tie included) ran the fixed bf16 cuDNN pick: the three
processes' frames and audio are identical bit for bit.

**Spend:** pod 8opjl8w358z3tn ran 11:38-12:16 UTC (about 38 min, ≈ $1.32),
deleted by the driver (GET 404); build pod (shared) about 10 min of jobs
(≈ $0.16). Balance $34.54 -> $31.95 over the session, other agents' pods
included.
