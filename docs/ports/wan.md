# Wan 2.1 / 2.2 (FastWan, SF-Wan)

Code: `crates/fastvideo-cudarc/src/wan/`. Reference: FastVideo
`fastvideo/models/wan/transformer.py` (DiT), `layers/layernorm.py`.

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
  Wan2.1 T2V-14B (bf16act / mxfp8): the cells are in the `wan` family
  (`sfwan13-33f-composed`, `sfwan13-33f-flash`, `sfwan13-81f-flash`,
  `wan5b-{f32act,bf16act,mxfp8}`, `wan14b-{f32act,bf16act,mxfp8}`; image
  `sha-d18eae2` carries them). Their weights are on `fv-weights-b200-us`
  (US-CA-2) only, and that datacenter had no RTX PRO 6000 for the whole
  hour the driver retried (2026-09-27 04:53–05:53 UTC). To run:
  `RUNPOD_VOLUME_NAME=fv-weights-b200-us FV_FAMILY=wan FV_PROMPTS=5
  FV_LPIPS=1 FV_CELLS="sfwan13-33f-composed sfwan13-33f-flash ..."
  scripts/gpu/runpod-http.sh run <sha>`.
- MXFP8 on sm_100 (B200): default by analogy with H3, not measured here.
- The fused norm kernels do not yet write MXFP8 activations directly (H3's
  `NormOut::Mx`); with FP8 off on sm_120 they have no consumer there.
