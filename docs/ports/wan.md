# Wan 2.1 / 2.2 (FastWan, SF-Wan)

Code: `crates/fastvideo-cudarc/src/wan/`. Reference: FastVideo
`fastvideo/models/wan/transformer.py` (DiT), `layers/layernorm.py`.

## Serving path: harness, TAEHV, caches, sparse routes

### Harness

`fv-gpucheck wan gen` runs a prompt set through one resident `WanPipeline`
(UMT5, DiT and decoder on the device) and writes `benchmark.json` beside the
clip directory, as the H3 and LTX-2 cells do. `--warm` runs one untimed
generation first. The timed span (`total_s`) is text + denoise + decode
through a finished mp4. PNG frames for `compare-clips` are written after the
mp4 and are not in it. `frames_sha256` (SHA-256 over the PNG frames) is the
byte-identity check. `peak_memory_mb` is device memory in use, weights
included. `WanPipeline::generate_to` is `generate()` with timings: the decode
streams each chunk through the H3/LTX frame drain into `wan/writer.rs`'s
`VideoWriter`.

Matrix: `runpod-matrix.sh wan`. FastWan 1.3B cells run on
`fv-weights-h3-ltx-hy`; 14B, TI2V-5B and SF-Wan cells run on
`fv-weights-b200-us`. Upstream: `scripts/gpu/upstream/bench_fastwan.py`
(FastVideo `basic_dmd.py` recipe: VSA 0.8, text encoder on the GPU, Triton
VSA on sm_120; one excluded warm-up, then the median of 3 per prompt and the
median over prompts), run by the `pod.sh` cells `fv-fastwan13-dmd`,
`fv-wan21-14b`, `fv-wan22-5b` and `fv-sfwan13`.

### Defaults and switches

| What | Default | Opt-out / switch |
|---|---|---|
| Decoder for distilled requests (DMD, rCM, causal DMD) | TAEHV (`taew2_1`) when found (`FASTVIDEO_TAEHV_WEIGHTS`, `FASTVIDEO_TAE_DIR`, `<weights>/taehv`, `<weights>/../taehv`, `~/.cache/fastvideo/taehv`; fetch with `scripts/gpu/fetch_taehv.sh`) | `FASTVIDEO_WAN_VAE=full` (`wan gen --full-vae`); `taehv` forces it |
| Full Wan VAE latent frames per pass | 2 | `FASTVIDEO_VAE_CHUNK=1` |
| Text K/V, text embedding, time modulation | computed once (per denoise, per timestep vector) | `FASTVIDEO_WAN_COND_CACHE=0` |
| UMT5 prompt disk cache | `wan gen`: `<clip dir>/../text-cache`; CLI: `~/.cache/fastvideo/wan-text` | `--no-text-cache`, `FASTVIDEO_WAN_TEXT_CACHE=off` |
| Negative prompt | encoded only when a sampler reads it (not for DMD, rCM or guidance 1) | |
| Sol-Attn | off | `FASTVIDEO_WAN_SOL_ATTN=1` (14B: 10 dense transformer calls and layer 0; 1.3B: layer 0 only) |
| SF-Wan sampling | block by block through the KV cache (FastVideo's causal DMD), flash attention | `FASTVIDEO_WAN_CAUSAL_AR=0` (whole clip, masked), `FASTVIDEO_WAN_CAUSAL_FLASH=0` (composed reference) |
| TeaCache / Sol caches | off | `FASTVIDEO_TEACACHE=1`, `FASTVIDEO_WAN_SOL_CACHE=teacache\|easycache\|taylorseer` |

Other fixes in the same pass:

- A model id containing `vsa` no longer sets `FASTVIDEO_SDPA=sparse`. That
  setting sent cross-attention to the host-only `attn::block_sparse_sdpa`.
  `FASTVIDEO_VSA=1` alone enables the device VSA kernels, in self-attention
  only.
- The TeaCache, Sol TeaCache, EasyCache and A14B decision metrics reduce on
  the device (`ops::abs_diff_sums_device`); only two scalars come back.
- The Sol-Attn dense-step guard counts transformer calls (cond, then uncond),
  as the reference's per-forward step clock does. It no longer counts denoise
  steps, which gave 14B twice the intended dense steps.
- The Morton3D reorder is a device row gather over an index buffer uploaded
  once per grid. Before, each call gathered q/k/v/out through `host_cow`.
- `tokenizer.json` (16 MB) is parsed and hashed once per process. Before, it
  cost ~0.45 s on every request.

### FastWan 1.3B: before and after

DMD 3 steps (1000/757/522), 480x832, 81 frames, VSA 0.8, one RTX PRO 6000
(sm_120). Each cell is warm, medians over the 5 prompts of `prompts-eval.json`.
Times are in seconds. LPIPS(alex) is the median over prompts of each clip's
mean against the reference cell's clip.

| Cell (run) | Text | Denoise | Decode | Total | Peak MiB | vs | LPIPS | PSNR dB |
|---|---|---|---|---|---|---|---|---|
| **Baseline**: pre-change code, Wan VAE chunk 1 (`425784e-09270342` `wan13-dmd`) | 1.28 | 2.49 | 3.16 | **6.95** | 23 908 | | | |
| Dense attention, pre-change (`wan13-dmd-dense`) | 1.25 | 3.29 | 3.16 | 7.71 | 23 876 | baseline | 0.539 | 12.3 |
| After this pass, f32 activations (`35607a6-09270404` `wan13-dmd`: TAEHV, caches) | 0.50 | 2.50 | 0.34 | **3.35** | 23 044 | full VAE | 0.033 | 32.8 |
| same, full VAE chunk 1 (`wan13-dmd-fullvae-chunk1`) | 0.45 | 2.50 | 3.14 | 6.11 | 24 004 | | byte-identical to the baseline's frames | |
| same, full VAE chunk 2 (`wan13-dmd-fullvae`) | 0.52 | 2.50 | 2.97 | 6.05 | 29 540 | chunk 1 | 0.000 | 61.4 |
| same, caches off (`wan13-dmd-nocache`) | 0.49 | 2.50 | 0.34 | 3.33 | 23 428 | cached | byte-identical (5/5 sha) | |
| **Final** with the bf16 DiT kernels merged (`d18eae2-09270447` `wan13-dmd`) | 0.04 | 1.96 | 0.31 | **2.32** | 23 044 | | | |
| final, caches off (`wan13-dmd-nocache`) | 0.06 | 1.96 | 0.31 | 2.34 | 23 044 | final | byte-identical (5/5 sha) | |
| final, full VAE (`wan13-dmd-fullvae`) | 0.04 | 1.96 | 2.92 | 4.93 | 29 700 | final | TAEHV vs VAE 0.032 | 32.8 |
| final + Sol-Attn 1.3B (`wan13-dmd-sol`, lossy) | 0.04 | 1.80 | 0.31 | 2.17 | 23 140 | final | 0.476 | 13.1 |
| final + TeaCache 1.3B (`wan13-dmd-teacache`, lossy) | 0.04 | 1.96 | 0.31 | 2.32 | 23 428 | final | 0 (never reuses in 3 steps) | |
| final, dense attention (`wan13-dmd-dense`) | 0.04 | 2.64 | 0.30 | 2.99 | 23 108 | final | 0.547 | 12.1 |
| **Upstream FastVideo** (`e90be598`, `a535bf3-09270324` `fv-fastwan13-dmd`) | 0.09 | 3.48 | 3.43 (+0.08 post, +0.40 mp4) | **7.63** | 30 246 (torch) / 34 388 (smi) | | | |

What each change bought (same process state, same run unless noted):

- **TAEHV by default**: the decode drops from 2.97–3.14 s to 0.31–0.34 s. The
  frames stay close to the full VAE: LPIPS 0.03, PSNR 32.8 dB. They are
  softer, with sharpness ratio 0.73 on `h3-demo` and 1.02 median. The gate
  fails that one prompt's sharpness, and its performance check uses
  `denoise_s`, which TAEHV does not change. The full VAE stays one switch away.
- **Full-VAE chunk 2**: decode 3.14 → 2.97 s (−5%). Peak goes up 5.5 GiB. The
  frames are not byte-identical to chunk 1 (PSNR 61 dB, LPIPS 0.000).
- **Streaming writer**: the mp4 is finished inside the decode (the upstream
  column adds 0.4 s of VideoSave after its decode). It lands in the harness
  commit, so the baseline already has it.
- **Text**: skipping the unused negative prompt took text from 1.28 to
  0.50 s. Parsing `tokenizer.json` once took it to 0.04 s. A disk-cache hit
  costs 0.009 s, against 0.02–0.26 s to encode (`h3-demo`, the only prompt
  seen twice per cell, is the hit).
- **Invariant caches** (text K/V, text and time embeddings): byte-identical,
  and no measurable denoise change at this shape (1.96 vs 1.96 s). The
  cross-attention K/V over 512 text tokens is small next to 32 760-token
  self-attention.
- **Sol-Attn on 1.3B** (lossy, opt-in): denoise −8%. LPIPS 0.48 against the
  VSA baseline is about the same distance as dense vs VSA (0.55): a different
  sparse approximation of a VSA-trained student gives a different sample.
  Not a default.
- **TeaCache (1.3B poly, threshold 0.08)**: a 3-step DMD schedule gives it no
  step to skip (it always computes the first step, and the accumulator
  crosses the threshold). The output is byte-identical to the baseline.

Against upstream FastVideo on the same card, the final is 2.32 s vs 7.63 s
(3.3x). Denoise is 1.96 vs 3.48 s, decode + mp4 is 0.31 vs 3.9 s (their
full VAE, then VideoSave), and text is 0.04 vs 0.09 s. Upstream's torch peak
is 30.2 GiB, ours 22.5 GiB in use.

### Wan2.1 T2V-14B on H100 (2026-09-27)

480x832, 81 frames, UniPC 50 steps, CFG 5, flow shift 3.0 (FastVideo
`WanT2V480PConfig`), the matrix prompt, seed 1024. One **NVIDIA H100 80GB
HBM3** in US-CA-2 (`fv-weights-b200-us`). The RTX PRO 6000 had no stock
there from 11:33 to 11:55 UTC. Image `sha-d18eae2`, run
`wan/d18eae2-09271156`. The 50-step cells run one cold generation each (no
`--warm`), with the load excluded. Upstream: `fv-wan21-14b`, run
`upstream/36ce5a2-09271156`, on the same GPU type. It used image
`fastvideo-rs-upstream-fastvideo:sha-7a0f247` (FastVideo `e90be598`),
`FLASH_ATTN` requested, and fell back to Torch SDPA because the image has no
`flash_attn` wheel. It ran one excluded warm-up, then the median of 3.
LPIPS(alex), PSNR and sharpness are against `wan14`.

| Cell | Text s | Denoise s | Decode s | Total s | Peak MiB | Steps computed | LPIPS mean / max | PSNR dB | Denoise speedup | Gate (lossy) |
|---|---|---|---|---|---|---|---|---|---|---|
| `wan14` (baseline) | 0.64 | 432.27 | 2.52 | **435.45** | 57 246 | 50/50 | — | — | 1.00x | — |
| `wan14-easycache` (0.036, retain 7) | 0.62 | 185.17 | 2.63 | 188.43 | 56 478 | 21/50 | 0.037 / 0.050 | 26.1 | 2.33x | fail: warm; sharpness 0.949 (min 0.95) |
| `wan14-teacache` (Sol TeaCache 0.12) | 0.63 | 69.90 | 2.52 | 73.08 | 57 982 | 8/50 | **0.561 / 0.571** | 15.1 | 6.18x | fail: warm; sharpness 0.81; jitter 1.58; LPIPS |
| `wan14-sol` (Sol-Attn tau 1.0) | 0.65 | 292.05 | 2.52 | 295.24 | 57 758 | 50/50 | 0.164 / 0.179 | 20.0 | 1.48x | fail: warm only (sharpness 0.978) |
| `wan14-fullstack` (EasyCache + Sol-Attn) | 0.65 | 133.96 | 2.57 | **137.20** | 56 702 | 21/50 | 0.150 / 0.172 | 20.6 | 3.23x | fail: warm; sharpness 0.920 |
| `wan14-4step` | 0.67 | 34.61 | 2.56 | 37.85 | 56 254 | 4/4 | | | | |
| `wan14-4step-nocache` (`WAN_COND_CACHE=0`, no text cache) | 0.59 | 34.61 | 2.51 | 37.73 | 56 254 | 4/4 | 0.0001 vs `wan14-4step` | 63.5 | | **off-identity fails**: frames differ |
| **Upstream FastVideo** `fv-wan21-14b` | 0.06 | 499.79 | 2.32 (+0.06 post, +0.29 save) | **502.63** | 76 227 (torch) | 50/50 | | | | |

What the run shows:

- **Baseline vs upstream**: 435.5 s vs 502.6 s, 1.15x. Denoise is 8.65
  s/step vs 10.0 s/step. Peak memory is 55.9 GiB vs 74.4 GiB (torch).
  Ours is a single cold generation and upstream is a warm median, so the
  comparison does not favour ours. The fullstack arm is 3.66x upstream's
  total.
- **The gate fails every lossy arm on `performance/warm`**. The 50-step
  cells are cold by design (see the matrix comment), and
  `gate-policy.toml` requires a warm-up for promotion. Apart from that
  check, Sol-Attn passes (LPIPS 0.16, sharpness 0.98). EasyCache misses
  sharpness by 0.001 (0.949), and fullstack misses it at 0.92.
- **Sol TeaCache (threshold 0.12) is far too aggressive on 14B**. It
  computes 8 of 50 steps, and LPIPS is 0.56. Do not use it at this
  threshold.
- **The exact-cache identity does not hold on 14B**. `wan14-4step` and
  `wan14-4step-nocache` differ (`frames_sha256` `5211920c…` vs
  `2d36a122…`, PSNR 63.5 dB, LPIPS 0.0001). The same pair was
  byte-identical on 1.3B. It is not yet known whether this comes from the
  invariant caches under CFG batching or from run-to-run nondeterminism on
  sm_90. A repeat of `wan14-4step` would tell them apart. Reported, not
  fixed (model code).
- Load (not in the totals): ours about 150–190 s per cell from the network
  volume (UMT5 f32 and the fp32 DiT converted to bf16). Upstream 147.6 s.
- 720p (FastVideo's registry default for this checkpoint:
  `WanT2V720PConfig`, 720x1280, shift 5.0) was not run. It costs about
  4x per generation.

The TI2V-5B cells (`wan5b`, `wan5b-easycache`)
and the matching upstream cell `fv-wan22-5b`
have not been measured (SF-Wan is measured below, on H100). Their weights are only on `fv-weights-b200-us`
(US-CA-2), and that datacenter had no RTX PRO 6000 in stock from 04:00 to
07:50 UTC on 2026-09-27. Pod creation retried the whole time, and no pod ever
started. To run them: `RUNPOD_VOLUME_NAME=fv-weights-b200-us FV_FAMILY=wan
FV_CELLS="wan5b wan5b-easycache" runpod-http.sh run <sha>`, then `runpod-http.sh upstream`
with `UP_CELLS`. Before trusting the 5B cells, expect a load failure or wrong
output: `WanPipeline` sizes latents at /8 and builds the Wan 2.1 VAE with
`z_dim` 48. Wan 2.2's VAE downsamples 16x and has a different decoder.

## DiT kernels (bf16 activations, fusion, FP8, block-causal attention)

### bf16 activations (default)

`WanPipeline::load_with` makes bf16 activations the process default, as the
H3 and LTX-2 pipelines do. FastVideo runs the Wan DiT in bf16
(`dit_precision`), UMT5 and the VAE in fp32 (`text_encoder_precisions`,
`vae_precision`): `Umt5Encoder::forward` and `AutoencoderKlWan`'s encode /
decode run inside `with_bf16_act(false)`, and the residual stream becomes
bf16 after the patch embedding. `FASTVIDEO_BF16_ACT=0` restores f32
activations everywhere (the numerics before this change).

The block keeps FastVideo's rounding points (`wan::fuse`):

| Step | Reference | Here |
|---|---|---|
| norm1 + AdaLN | `bf16(LN(h.float()) * (1 + scale) + shift)` | `ln_adaln_e` (one rounding) |
| self-attn residual + norm2 | `ScaleResidualLayerNormScaleShift`, f32 compute: hidden `h + a * gate` in f32; norm2 (affine) reads the **unrounded** sum; both cast to bf16 | `fuse::self_residual_norm` |
| cross residual + FFN norm | `h + a` in bf16; `FP32LayerNorm` rounds to bf16, then `* (1 + scale) + shift` in f32, rounded | `fuse::cross_residual_norm_mod` |
| FFN residual | `ScaleResidual`: `h + ff * gate` (bf16 x f32 → f32), rounded once | `fuse::gate_residual` |
| q/k norm (across heads) | `RMSNorm.forward_native`: `bf16(x * rsqrt(mean(x²) + eps))`, then `* w` in bf16; RoPE in f32, rounded once | `fuse::qk_norm_rope` (`wan_qk_norm_rope16`) |
| bias + GELU | `addmm` with the bias in the epilogue, then GELU-tanh in bf16 | unchanged: cuBLASLt bias epilogue + bf16 `mx_unary` (both were already bf16 in / out) |

Every step reads bf16 operands directly (no widening copies); the previous
f32-activation path is untouched.

### Residual + norm fusion (`FASTVIDEO_WAN_FUSE`, default on)

The two residual + norm steps run as one kernel each (`wan_res_ln`). `=0`
runs the same math as two kernels (`wan_res_gate` stores what `wan_ln`
reads: the f32 sum for the self-attention residual, the bf16 hidden for the
cross one). All three kernels spell every operation with `_rn` intrinsics, so
no FMA contraction separates them: fused and unfused clips are byte-identical
(`frames_sha256` equal, compare-clips off-identity pass on all 5 prompts).

### FP8 (`FASTVIDEO_WAN_QUANT=off|w8a8|mxfp8`)

`WanQuantPlan` quantizes every block's attention (self and cross) and FFN
linears — FastVideo `fp8_config._FP8_SUFFIXES` for Wan: `to_q/k/v`,
`to_out`, `ffn.fc_in/fc_out`. W8A8 keeps one tensor scale per original
linear (fused QKV / KV are sections); MXFP8 takes each stack whole. The I2V
image K/V, the VSA gate, embedders and the head stay bf16. Unset: MXFP8 on
sm_100-class GPUs (block-scaled FP8 tensor cores, the H3 default there),
off elsewhere — on RTX PRO 6000 neither recipe won speed and quality (below).
`FASTVIDEO_FP8` (W8A8 on every linear) still applies and takes precedence
per linear.

### Datacenter dense attention (`attn_dc.cu`, default on B200 and H100 / H200)

Dense SDPA at head dim 128 on a 10.0 (B200) or 9.0 (H100 / H200) device runs
`attn_dc.cu`; every other GPU keeps the mma.sync kernels (sm_120 unchanged).

- `fa_dc100_fwd_d128` (sm_100a): tcgen05 MMA into tensor memory, TMA loads,
  warp-specialised in the FlashAttention-4 / CUTLASS sm100 FMHA order (256
  queries per CTA as two 128-row tiles; one MMA thread; TMEM S0 | S1 | O0 |
  O1; P written over S and read by the P.V MMA from TMEM; one thread per
  query row for the softmax, which also rescales O in TMEM when a row's max
  grows).
- `fa_dc90_fwd_d128` (sm_90a): wgmma + TMA, FlashAttention-3 structure (a
  producer warpgroup, two consumer warpgroups of 64 rows; QK of tile j and
  PV of tile j-1 in flight while the softmax of tile j runs).

Same per-row arithmetic as `flash_mma_fwd2` except the max advances per 128
keys, so both agree with V2 to bf16-P rounding (rel L2 1e-4 to 8e-4), not
bit for bit. `fv-gpucheck kernels` (group `attn_bench`, Rust path, bf16 out):

| GPU | shape | V2 TFLOPS | dc TFLOPS | cuDNN SDPA (torch) |
|---|---|---|---|---|
| B200 | H3 768p | 369 | 1173 | 1490 |
| B200 | LTX 1080p 20 s | 372 | 1148 | 1401 |
| B200 | LTX 4K 5 s | 373 | 1152 | 1405 |
| H100 SXM | H3 768p | 303 | 616 | 598 |
| H100 SXM | LTX 1080p 20 s | 326 | 565 | 587 |
| H100 SXM | LTX 4K 5 s | 321 | 570 | 632 |

Escape hatch everywhere: `FASTVIDEO_FLASH_KERNEL=v2` (or `[kernels]
dense_attention = "nvcc:v2"`). build.rs builds the arch-specific cubins
(sm_90a, sm_100a) next to the per-SM kernels.cu cubins.
`scripts/gpu/upstream/attn_dc_bench.cu` (pod step `bench:attn_dc`, dev loop
`dev:attn_dc`) is the standalone parity / timing harness; `fv-gpucheck
kernels --groups attn_dc` the Rust parity group. Sol, VSA and block-causal
still run mma.sync on these GPUs.

### Block-causal flash attention (SF-Wan, `FASTVIDEO_WAN_CAUSAL_FLASH`, default on)

SF-Wan's self-attention takes one of two routes (see "SF-Wan: FastVideo's
causal inference" below). The default, block by block through the KV cache,
is unmasked attention of a block's queries against the cache window: the
dense flash kernel with `Sq != Sk` (`nn::sdpa_kv_window`). The whole-clip
route (`FASTVIDEO_WAN_CAUSAL_AR=0`) uses the masked kernel
`flash_mma_fwd2_causal_d{64,128}`. It takes the mask as parameters
(`BlockCausal`: tokens per block, window in blocks counting the query's
own, sink blocks). It walks only the sink tiles and the causal band of key
tiles that a 128-query CTA can see. Tiles that straddle the boundary are
masked per score with `-inf`, which the online softmax treats as an absent
key. The arithmetic is the dense V2 kernel's, and that kernel's code is
unchanged. `FASTVIDEO_WAN_CAUSAL_FLASH=0` sends both routes to
`sdpa_composed` (f32 scores and softmax; the whole-clip route materializes
the `[S, S]` mask), the reference path. So do CPU runs,
sequence-parallel shards and unsupported head dims.

The whole-clip mask is FastVideo's `_prepare_blockwise_causal_attn_mask`
(`causal_temporal_mask`). Blocks are `num_frames_per_block` (3) latent
frames, and a query sees keys before the end of its block. A window of
`local_attn_size` frames counts back from that end, token-granular. The
query always sees itself, and there is no sink. The kernel takes windows of
whole blocks, and any other window is refused
(`attn::block_causal_for`). FastVideo's inference never uses this mask.

### Measurements: FastWan 1.3B, 480x832, 81 frames, RTX PRO 6000 (sm_120)

DMD 3 steps, VSA (the checkpoint's gates), warm, medians over the 5 prompts
of `prompts-eval.json`, one pod (image `sha-f6e8c66`, run
`wan/f6e8c66-09270353`). Baseline `wan13-f32act` = `FASTVIDEO_BF16_ACT=0`,
the pre-change numerics. LPIPS is the mean over prompts of each clip's mean
LPIPS against the baseline clip (per prompt in brackets). Gate: `fv-gpucheck
gate`, `scripts/gpu/gate-policy.toml`.

| Arm | Denoise s | Decode s | Total s | Peak MiB | LPIPS vs f32act | Gate vs f32act |
|---|---|---|---|---|---|---|
| f32act (pre-change) | 2.498 | 3.107 | 6.90 | 24 036 | — | — |
| bf16act (`WAN_FUSE=0`) | 2.008 | 3.110 | 6.37 | 23 908 | 0.107 (0.021 / 0.184 / 0.113 / 0.121 / 0.096) | pass, 1.24x |
| **fuse (default)** | **1.961** | 3.105 | 6.12 | 24 036 | 0.107 (identical to bf16act) | pass, 1.27x |
| mxfp8 | 2.025 | 3.110 | 6.41 | 23 428 | 0.215 | pass, 1.23x |
| w8a8 | 2.190 | 3.110 | 6.29 | 23 332 | 0.207 | **fail** (sharpness 0.94 on spark-mountain-lake) |

Against their own parent: fuse vs bf16act is byte-identical (exact gate:
off-identity pass; performance 1.024x, under the 1.03 experimental bar — the
fusion saves ~45 ms of 2 s here; the kernel-level win is below). mxfp8 vs
fuse: 0.968x, LPIPS 0.202 → fail (performance). w8a8 vs fuse: 0.896x,
LPIPS 0.215 → fail (performance and quality). Text-encoder time varies
0.98–1.28 s between cells (UMT5 is f32 in all arms).

What changed the default: bf16 activations and the fusion (both pass, and
bf16 is how FastVideo runs the DiT). FP8 stays off on sm_120: at 32 760
tokens x 1536 the 1.3B's GEMMs are too small for FP8 to beat bf16 once the
activation quantize pass is paid, and both recipes move LPIPS twice as far
as bf16 does. MXFP8 remains the sm_100 default by analogy with H3 (not
measured on B200 in this change).

Kernel timings (`fv-gpucheck kernels --groups wan_fusion`, 32 760 x 1536):
self-attention residual + norm2 0.369 ms fused vs 0.679 ms unfused; cross
residual + FFN norm 0.363 vs 0.572 ms; the f32-activation chain for the
same two steps 1.42 ms.

### Block-causal attention checks (`fv-gpucheck kernels --groups wan_causal_attn`)

RTX PRO 6000, image `sha-f6e8c66` (run `wankernels/f6e8c66-09270420`,
exact GEMM math). Synthetic bf16 Q/K/V (std 1); limit 2e-2 max abs error.

Before blocks: the window then counted frames back from the query's frame,
and the kernel now counts blocks, the query's own included.

| Case (b, h, S, d; frame tokens, window, sink) | vs f64 SDPA | vs `sdpa_composed` |
|---|---|---|
| 1, 2, 300, 128; 60, 0, 0 | 1.27e-3 | 1.27e-3 |
| 2, 3, 257, 128; 37, 2, 0 | 1.56e-3 | 1.56e-3 |
| 1, 2, 511, 128; 50, 3, 1 | 1.69e-3 | 1.69e-3 |
| 1, 4, 190, 64; 19, 0, 2 | 1.46e-3 | 1.46e-3 |
| 1, 1, 129, 128; 1, 0, 0 (per-token causal) | 2.02e-3 | 2.02e-3 |
| 1, 2, 640, 128; 128, 1, 0 | 9.1e-4 | 9.1e-4 |
| 1, 2, 96, 64; 200, 0, 0 (one frame) | 9.6e-4 | 9.6e-4 |

All pass (the composed path itself is within 1e-6 of f64 in exact math; the
kernel's error is its bf16 P). With one frame spanning the sequence the
causal kernel equals the dense V2 kernel bit for bit (d = 64 and 128). The
`*_routed_bf16` checks in that run failed by construction: in an exact-math
context `nn::sdpa_block_causal` takes the composed fallback, not the
kernel; the check now runs only where the fused kernels are the default.

Timing, 12 heads x d 128, 1560 tokens per frame (SF-Wan 1.3B 480x832):

| Frames (tokens) | Causal flash | Dense flash (V2) | `sdpa_composed` (f32 math) |
|---|---|---|---|
| 7 (10 920) | 1.57 ms | 2.32 ms | 40.0 ms |
| 21 (32 760) | 10.91 ms | 17.98 ms | does not fit (two 51 GB f32 `[12, S, S]` buffers) |

The TI2V-5B / T2V-14B arms are in `runpod-matrix.sh wan` but were not run:
see "Not measured" below. SF-Wan is measured in the next section.


### Not measured

- Wan2.2 TI2V-5B and
  Wan2.1 T2V-14B (bf16act / mxfp8): the cells are in the `wan` family
  (`wan5b-{f32act,bf16act,mxfp8}`, `wan14b-{f32act,bf16act,mxfp8}`; image
  `sha-d18eae2` carries them). Their weights are on `fv-weights-b200-us`
  (US-CA-2) only, and that datacenter had no RTX PRO 6000 for the whole
  hour the driver retried (2026-09-27 04:53–05:53 UTC). To run:
  `RUNPOD_VOLUME_NAME=fv-weights-b200-us FV_FAMILY=wan FV_PROMPTS=5
  FV_LPIPS=1 FV_CELLS="wan5b-bf16act ..."
  scripts/gpu/runpod-http.sh run <sha>`.
- SF-Wan on RTX PRO 6000: none was in stock in US-CA-2, so the numbers
  below are from H100. To rerun: `RUNPOD_VOLUME_NAME=fv-weights-b200-us
  ORACLE_TARGETS=sfwan13 FV_ORACLE_FAMILY=sfwan UP_AFTER=cells:fv-sfwan13
  UP_IMAGE_TAG=latest FASTVIDEO_DUMP_OPS=0,1,15,29 FV_PROMPTS=5 FV_LPIPS=1
  scripts/gpu/oracle.sh <sha>`. This runs the oracle and the upstream bench
  on one pod, and the oracle and SF-Wan cells on the other.
- MXFP8 on sm_100 (B200): default by analogy with H3, not measured here.
- The fused norm kernels do not yet write MXFP8 activations directly (H3's
  `NormOut::Mx`); with FP8 off on sm_120 they have no consumer there.

## SF-Wan: FastVideo's causal inference (parity)

Reference: FastVideo e90be59, `models/wan/causal_transformer.py`
(`CausalWanSelfAttention`, `_forward_inference`) and
`pipelines/basic/wan/stages/causal_denoising.py`
(`CausalDMDDenosingStage`, run by `WanCausalDMDPipeline`). Checkpoint:
`wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers`. Its `transformer/config.json` sets
none of the causal fields, so `WanVideoArchConfig` defaults apply. Its
scheduler is `SelfForcingFlowMatchScheduler` (shift 5, 1000 steps,
`extra_one_step`).

| | FastVideo | This port before | Now |
|---|---|---|---|
| Generation | Block by block. 3 latent frames per block (`num_frames_per_block`), 7 blocks for 81 frames. | The whole clip (21 latent frames) denoised at once. | As FastVideo (`wan/pipeline.rs causal_dmd_denoise`). |
| Self-attention in inference | A block's queries attend, unmasked, to the KV cache window `[max(0, end - max_attention), end)` up to the end of the block, their own block included (both ways within the block). | A per-latent-frame causal mask over the whole clip: a frame saw itself and earlier frames, not the rest of its block. | As FastVideo (`wan/causal.rs`, `forward_kv`). |
| Window / sink | `local_attn_size` -1 for this checkpoint: a 21-frame cache (`sliding_window_num_frames`), and more frames is an error. `sink_size` 0. With a window, a full cache drops its oldest tokens after the sink frames (`num_evicted_tokens`). | `local_attn_size` 21 frames (no effect up to 21 frames), sink in the mask. | `local_attn_size` -1, sink 0. Rolling eviction ported and unit-tested. |
| KV cache | Per layer: roped K and V in bf16. The next step of the same block overwrites the block's slots. After the last step, a context pass at `t = context_noise` (0) on the clean latents rewrites them. | None. The NVFP4-only host cache (`ar_cache.rs`) went frame by frame. | As FastVideo. |
| RoPE | Absolute, table from `start_frame` (the block's first frame). The `relativistic` policy is opt-in and unused. | Whole-clip table. | Absolute from `start_frame`. `relativistic` not ported. |
| Timesteps | `[1000, 750, 500, 250]` warped through the scheduler table (`timesteps[1000 - t]`): 1000, 937.5, 833.3, 625, fractional into the DiT. sigma(t) = t/1000. | 1000/750/500/250 unwarped, sigmas from FastWan's shift-8 table: 1.0, 0.96, 0.889, 0.727. | As FastVideo (`schedulers::SelfForcingSchedule`). |
| Per step | x0 = x - sigma * v (f64, cast to bf16). Re-noise to the next sigma with a fresh `[B, 3, C, H, W]` bf16 draw from the request generator (3 draws per block). The last step keeps x0. | The same update on the whole clip, one draw per step. | As FastVideo, including the bf16 rounding points. |
| Whole-clip mask (training forward) | 3-frame blocks, token window from the block end, self always visible, no sink. | Per frame. | As FastVideo (kernel for windows of whole blocks). |

`FASTVIDEO_WAN_CAUSAL_AR=0` keeps the whole-clip sampler. With
`FASTVIDEO_WAN_CAUSAL_FPB=1` added, it is exactly the path before this change.
Unit tests (`wan::causal`) cover the cache pointer arithmetic (append,
overwrite, eviction with a sink). They also check that running blocks
through the cache at one timestep gives the whole clip under the blockwise
mask (tiny DiT, CPU, max abs error <= 1e-4).

Not covered: SF-Wan 2.2 A14B / I2V (`sf_wan_2_2_*` presets). They still map
to the non-causal MoE configs, and FastVideo runs them with a 1-frame first
block and two caches.

### Oracle (`scripts/gpu/oracle.sh`, target `sfwan13`)

This run was on H100 80GB HBM3, because US-CA-2 had no RTX PRO 6000. Both
sides ran on the same GPU type. Runtime image `sha-fadb6f1` (run
`sfwan/fadb6f1-09271211`), upstream `fastvideo-rs-upstream-fastvideo:latest`
(FastVideo e90be59, attention backend TORCH_SDPA; the image has no
flash-attn). Settings: the matrix prompt, seed 1024, 480x832x81.
`oracle_dump.py` hooks the stage and the causal DiT. Our run injects the
initial latents (`sf_latents_in`), the text states (`text_hidden`) and all
21 re-noise draws (`sf_noise_*`), and decodes with the full Wan VAE.

rel-L2 of ours against FastVideo:

| | Before (whole clip, per-frame mask) | Now (flash) | Now (composed) | Floor: ours bf16 vs ours f32 |
|---|---|---|---|---|
| Block 0, step 1, DiT block 0 / 15 / 29 output | — | 7.2e-3 / 2.5e-2 / 1.7e-2 | 7.3e-3 / — / 1.7e-2 | 4.9e-3 / 4.3e-2 / 2.0e-2 |
| Block 0, step 1 prediction (frames 0-2, t = 1000) | **0.403** | 1.9e-2 | 1.9e-2 | 2.3e-2 |
| Block 0 final latents | — | 8.4e-2 | | 9.7e-2 |
| Block 1, step 1: layer 0 / 29 K window (6 frames) | — | 1.3e-2 / 2.7e-2 | | |
| Block 1, step 1 prediction (first read of the cache) | — | 8.2e-2 | 6.8e-2 | 9.6e-2 |
| Block 3 / 6 final latents | — | 0.140 / 0.234 | — / 0.259 | 0.166 / 0.310 |
| Final latents (21 frames) | **0.971** | 0.165 | 0.198 | 0.216 |
| Frames vs FastVideo's mp4: PSNR / LPIPS mean | **11.9 dB / 0.650** | 28.5 dB / 0.071 | | |

The frames-vs-mp4 comparison has its own codec floor: our PNGs against our
own mp4 score 39.0 dB and LPIPS 0.023.

Divergence before the fix starts at the first forward. At t = 1000, with
identical noise and text, frames 0-2 differ by 0.40, because the per-frame
mask hides frames 1-2 from frame 0 and FastVideo's block does not. It then
compounds through the different timesteps (750 vs 937.5), sigmas and
sampling order to 0.97 on the final latents, a different video (LPIPS 0.65).

After the fix, the first forward matches block by block. DiT block outputs
grow smoothly from 7e-3 to 3e-2, with no jump. The first read of the cache
(block 1) is at 8e-2, because the context carries block 0's accumulated
difference. Each K window has FastVideo's length: 74 strided rows for block
0 and 147 for block 1, which is 3 and 6 frames. Every number is at or under
our own bf16-vs-f32 floor. The residual grows block over block as the
chaotic amplification of bf16 rounding, as in the H3 oracle. Layer 15's
attention output sits at 0.26 in both our floor and our comparison with
FastVideo: a large-activation layer, not a divergence. Flash vs composed
(ours against ours) is 1.7e-2 at the first prediction and 0.21 at the end,
the same floor.

### SF-Wan 1.3B timings (H100 80GB HBM3, warm, medians over the 5 prompts)

`runpod-matrix.sh wan` cells (also run by the `sfwan` family), 4 steps,
480x832x81, bf16. Upstream: `bench_fastwan.py` (`pod.sh fv-sfwan13`: one
excluded warm-up, then the median of 3 per prompt, median over the same 5
prompts). FastVideo defaults: full VAE (bf16 decode), then VideoSave.

| Cell | Total s | Denoise s | Decode (+ mp4) s | Peak |
|---|---|---|---|---|
| `sfwan13-81f-flash` (default: KV cache, flash, TAEHV) | **4.41** | 3.83 | 0.46 | 22.6 GiB |
| `sfwan13-81f-fullvae` (full Wan VAE) | 6.12 | 3.83 | 2.20 | 31.2 GiB |
| `sfwan13-81f-composed` (`FASTVIDEO_WAN_CAUSAL_FLASH=0`) | 21.0 | 19.85 | 1.08 | 42.2 GiB |
| `sfwan13-81f-wholeclip` (`FASTVIDEO_WAN_CAUSAL_AR=0`, not FastVideo's algorithm) | 2.92 | 2.52 | 0.30 | 21.0 GiB |
| FastVideo `fv-sfwan13` | 7.62 | 4.73 | 2.32 + 0.34 save | 29.5 GiB (torch) |

Against FastVideo, our full-VAE cell (same decoder) is 1.25x faster end to
end, and denoise is 1.24x (3.83 vs 4.73 s). The default with TAEHV is 1.73x.
Text is 0.07 vs 0.09 s.

Flash vs composed, over the 5 prompts: LPIPS 0.040 / 0.053 / 0.469 / 0.206
/ 0.245 (h3-demo, frogyoga, multishot, newsbroadcast, spark-mountain-lake),
PSNR 18-32 dB. The gate passes (lossy). This is the rounding floor above:
the two routes round the attention differently, and the autoregressive
rollout amplifies that across 7 blocks. Flash is 5.2x faster on denoise.
The whole-clip route is faster, but it is a different sampler (LPIPS
0.48-0.72 against the flash cell), kept only as a diagnostic.

TAEHV against the full VAE on the same latents: LPIPS 0.016-0.103, PSNR
28-33 dB.

Kernel checks on this run (`kernels-wan`, H100) all pass: the whole-clip
causal kernel against f64 and the composed path (max abs <= 2.0e-3), and
unmasked causal equal to dense V2 (0 ulps). In this run the
`kv_window_flash_vs_sdpa_composed` check and the `kv_window_timing_3x21f`
note went through the routed entry point. In the exact-math `kernels`
context that entry point takes cuBLAS, not the mma kernel (0.0 difference,
42.9 vs 47.5 ms). Both now call `attn::device_mma_sdpa` directly; not
re-run. The flash route's speed is measured end to end above. Whole-clip
timing (12 heads, 1560 tokens per frame, 3-frame blocks): causal kernel
12.6 ms vs dense flash 20.4 ms at 21 frames, and 1.97 vs 2.38 ms (composed
42.2 ms) at 7 frames.
