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

### Block-causal flash attention (SF-Wan, `FASTVIDEO_WAN_CAUSAL_FLASH`, default on)

A causal Wan forward without the AR cache used to build the `[S, S]`
additive mask on the host and run `sdpa_composed` (f32 `[H, S, S]` scores
and softmax). `flash_mma_fwd2_causal_d{64,128}` takes the mask as
parameters (`BlockCausal`: tokens per frame, window, sink — the predicate of
`causal_temporal_mask`), walks only the sink tiles and the causal band of
key tiles a 128-query CTA can see, and masks straddling tiles per score
(`-inf`, which the online softmax treats as an absent key). Same arithmetic
as the dense V2 kernel, whose code is unchanged. `=0`, CPU runs,
sequence-parallel shards and unsupported head dims materialize the mask and
take the composed path.

Parity gap (not changed here): the Rust mask is per latent frame. FastVideo's
causal Wan (`causal_transformer.py _prepare_blockwise_causal_attn_mask`)
groups `num_frames_per_block = 3` frames that see each other, with a
token-granular window from the block end. The kernel supports blocks
(`frame_tokens = 3 * frame_seqlen`); the token-granular window does not map
to the frame window when `local_attn_size % 3 != 0`.

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

End-to-end SF-Wan cells (`sfwan13-33f-*`, `sfwan13-81f-flash`) and the
TI2V-5B / T2V-14B arms are in `runpod-matrix.sh wan` but were not run: see
"Not measured" below.

### Not measured

- SF-Wan 1.3B end to end (flash vs composed LPIPS), Wan2.2 TI2V-5B and
  Wan2.1 T2V-14B (bf16act / mxfp8): the cells exist in the `wan` family
  (commits after `f6e8c66`), but no CI image was built for them — pushing
  the branch that carries them was refused by the session's permission
  system. Their weights are on `fv-weights-b200-us` only
  (`RUNPOD_VOLUME_NAME=fv-weights-b200-us`).
- MXFP8 on sm_100 (B200): default by analogy with H3, not measured here.
- The fused norm kernels do not yet write MXFP8 activations directly (H3's
  `NormOut::Mx`); with FP8 off on sm_120 they have no consumer there.
