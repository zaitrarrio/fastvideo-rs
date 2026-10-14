# VDN-H3 (Video DeltaNet on MiniMax-H3)

Target: [`OpenVDN/vdn-minimax-h3`](https://huggingface.co/OpenVDN/vdn-minimax-h3) at
`2ce18deb1d1c6b115d6bb3732f3750fdbac95173` (lastModified 2026-10-05), code
[`OpenVDN/vdn-minimax-h3`](https://github.com/OpenVDN/vdn-minimax-h3) at `e262cb5` (2026-10-04), paper arXiv 2609.20744,
blog https://openvdn.github.io/. Owner request 2026-10-13: "let's implement https://huggingface.co/OpenVDN/vdn-minimax-h3".
Download approval (2026-10-13): `stage-b-step-2000/` plus the small root files only.

Status: see §0. Ported and validated (stage 1): the stage-B model, served as `vdn-h3`; parity §5, timing §6, open items §7.

## 0. Status and the one open decision

**The 8-step model is not the approved tree.** The repository holds two VDN checkpoints over the same `h3-base/`:

| directory | model card name | what it is | files beyond the other |
|---|---|---|---|
| `stage-b-step-2000/` | **VDN-H3-50-step** | the hybrid model after stage B (linear branch + rank-64 LoRA on the attention), sampled with 50 NFE | |
| `stage-dmd-step-250/` | **VDN-H3-8-step** | the same branch and the same default LoRA (byte-identical files: LFS `dec6981c…`, `58558fef…`), **plus** `adapters/turbo/` (0.85 GB, DMD2 without GAN, 250 steps, teacher = stage B) | `adapters/turbo/adapter_model.safetensors` (851,452,696 B, LFS `24fc93c8…`) and three small JSON files |

`modular_model_index.json` points the diffusers pipeline at `stage-dmd-step-250/diffusers`; every 8-step number on the model
card (SGLang 6.9 s on 8xB200, `8nfe_*.yaml`) is that directory. The brief's "distilled to 8 steps" therefore needs the turbo
adapter, which the approval does not cover. This port loads any list of VDN adapters (the turbo adapter's mixed ranks and its
FFN / AdaLN / `norm_out` targets included, §2.6), so the 8-step model is one add-only publish away
(`stage-dmd-step-250/adapters/turbo/*` + its `adapter_spec.json`, `metadata.json`, `model_spec.json`, `diffusers/config.json`;
the branch and default adapter would be reused from `vdn-h3-stage-b-2000`). That download (0.85 GB) needs the owner's yes.

## 1. The reference, read in full

Sources (all at the revisions above): the remote code `stage-*/diffusers/modeling_vdn_h3.py` (`M:`, 3,602 lines, one
generated file concatenating the code repo's `src/models/*`; identical blob `6cf82b7e` in both stages), the code repo's
`src/inference/render.py` (`R:`), `configs/inference/{8,50}nfe*.yaml`, `src/config/inference.py`, `diffusers_patches/`, the
checkpoint JSON files (`model_spec.json`, `adapter_spec.json`, `linear_branch/config.json`, `metadata.json`), the safetensors
headers, the blog, and SGLang `main` @ `684fea8` (2026-10-13) `runtime/layers/attention/backends/hybrid_window_attn_h3.py`,
`kernels/jit/csrc/diffusion/vdn_delta_factors.cuh`, `configs/sample/minimax_h3_vdn.py`.

### 1.1 Which blocks are hybrid

All 50 DiT blocks (`M:apply_hybrid_attention_transform`): each block's `attn` becomes a `HybridAttention` that wraps the original
module (`attn.orig.*`, the base weights) and adds the branch. The two token-refiner blocks are not converted (they attend over text
only). Nothing else in the DiT changes: AdaLN, FFN, RoPE, packing, time embedding and output heads are MiniMax-H3's.

Per block (`M:HybridAttention._hybrid_forward`), with `x` the block's normed, modulated attention input `[S, 5376]`:

```
q_raw, k_raw, v = orig.to_q(x), orig.to_k(x), orig.to_v(x)          # [S, 56, 128], LoRA merged into the weights
q, k = rope(rms_norm_q(q_raw)), rope(rms_norm_k(k_raw))             # exactly H3's processor
o_sm = window_softmax(q, k, v)                                       # §1.3
out  = orig.to_out(  (o_sm * softmax_gate(x)).flatten()  )           # softmax_gate: sigmoid(Linear(5376 -> 56, bias)), per (row, head)
out[video] += to_out_linear( linear_branch(x[video], q_raw, k_raw, v on video rows; x, k_raw, v on text rows) )
```

`to_out_linear` is `Linear(7168 -> 5376, bias=False)`. The softmax gate is applied to **every** row (text and audio too). When
the window covers every frame (`F <= 10` latent frames under chunk 5 / radius 1, below H3's 4 s minimum) the softmax is plain
dense attention and the branch is skipped (it would double count); the gate still applies.

### 1.2 The linear branch (`M:BidirectionalLinearBranch`, transform v2: `delta_rule vdn_solve`, `bridge alpha`, `a_fp32`, `enable_text_state`, `short_conv {k, v}`, `linear_head_dim 128`, `anchor_frames both`)

It reuses the softmax branch's **raw** projections (pre-QK-norm, pre-RoPE; NoPE) and adds no Q/K/V projection of its own.
`anchor_frames = both` makes frames 0 and F-1 exact softmax in both directions, so the branch **drops them from its input** and
runs on the inner frames `1..F-2` (F' = F-2); their readout rows are exactly zero. Window bounds rebase by one.

Per inner frame `f`, head `h` (`S` tokens per frame, `d = 128`):

1. **Features** (`M:prepare_linear_features`). `q = l2norm(silu(q_raw))`. `k = l2norm(silu(tconv(sconv(k_raw))))`,
   `v = silu(tconv(sconv(v)))`. `sconv`: depthwise 5x5 conv per frame over the `(h, w)` token grid, zero padding, no bias
   (`short_conv.{k,v}_sp.weight [7168, 1, 5, 5]`), bf16 (cuDNN). `tconv`: depthwise 5-tap conv over the inner frames, zero padding
   (the dropped anchors are not read), no bias (`{k,v}_tm.weight [7168, 1, 5]`), accumulated in fp32 and rounded once in the
   inference kernel. `l2norm` is per head over `d`: `x / max(||x||, 1e-6)`. bf16 outputs.
2. **beta** `= sigmoid(beta_proj(x))`, `beta_proj: Linear(5376 -> 56, no bias)`, per token and head.
3. **Frame statistics** (`M:_frame_statistics`): `A[f,h] = sum_s beta k k^T` in **fp32** (TF32 GEMM on the inference path, from
   fp32 copies of bf16 `k` and fp32 `beta*k`), symmetrized `A = (A + A^T) / 2`; `B[f,h] = sum_s (v * beta)_bf16 k^T`, a bf16 GEMM
   whose output is rounded to bf16, then widened to fp32. `B` is `[d_v, d_k]`.
4. **Decay** (`M:FrameKDAAlpha`, KDA's gate, an fp32 island): `m_f = mean_s x[f, s]` (fp32 mean of the bf16 input),
   `alpha[f,h,:] = exp(-exp(A_log[h]) * softplus(up(down(m_f)) + dt_bias))`, `down: 5376 -> 128`, `up: 128 -> 7168` (no biases),
   weights widened to fp32. Per frame, head and key channel.
5. **vdn_solve** (`M:VdnDelta`): `T_f = Diag(alpha_f) (I + A_f)^{-1}` (row scaling), `J_f = B_f (I + A_f)^{-1}`, fp32
   (Cholesky + one triangular solve + `L^-T L^-1`). The recurrence is `S_f = S_{f-1} T_f + J_f`, i.e. the frame-wise regularized
   delta rule `S_f = (S_{f-1} Diag(alpha_f) + B_f)(I + A_f)^{-1}` (blog "Video Delta Attention").
6. **Text state** (`M:_text_chunk_state`): the prompt rows (`text_indices` only: not audio, not keyframe rows) as one chunk, no
   conv: `k_t = l2norm(silu(k_raw))`, `v_t = silu(v)`, `beta_t = sigmoid(beta_proj(x_text))`, `A_t, B_t` as in 3,
   `S_text = B_t (I + A_t)^{-1}`; **both** scans start from `0.5 * S_text`.
7. **Scans** (`M:_run_scans`): forward `P[f] = P[f-1] T_f + J_f` (`P[-1] = 0.5 S_text`), reverse `Q[f] = Q[f+1] T_f + J_f`
   (`Q[F'] = 0.5 S_text`), fp32, `[F', 56, 128, 128]` each.
8. **Gather** (`M:gather_linear_state`, `bridge = alpha`): inner frame `t` with rebased window `[lo, hi]`:
   `before = P[lo-1]` (or `0.5 S_text` if `lo-1 < 0`), `after = Q[hi+1]` (or `0.5 S_text` if `hi+1 >= F'`), each decayed into `t`
   column-wise (over `d_k`) by `prod alpha_u`, `u` in `[max(lo,0), t]` for `before` and `[t, min(hi, F'-1)]` for `after` (log-cumsum
   of `max(alpha, 1e-12)`); `state_t = before' + after'`, rounded to bf16. The window frames themselves are not injected (softmax
   covers them), so nothing is counted twice.
9. **Readout**: `r[s] = state_t q[s]` (bf16 GEMM, `[d_v]` per head), `RMSNorm(d=128, eps=1e-6, weight [128] shared by heads)`
   (fp32 mean square, then bf16 products), times `output_gate(x) = sigmoid(up(down(x)))` (`down: 5376 -> 128` no bias,
   `up: 128 -> 7168` with bias, per token, head and channel), flattened to 7168 for `to_out_linear`.

The branch tensors (`linear_branch/model.safetensors`, 800 tensors, all bf16, 4.28 GB): per block `alpha.{A_log [56], dt_bias [7168],
down.weight [128, 5376], up.weight [7168, 128]}`, `beta_proj.weight [56, 5376]`, `norm.weight [128]`,
`output_gate.{down.weight [128, 5376], up.weight [7168, 128], up.bias [7168]}`, `short_conv.{k,v}_{sp,tm}.weight`,
`softmax_gate.up.{weight [56, 5376], bias [56]}`, `to_out_linear.weight [5376, 7168]`. The loader casts every fp32 hybrid parameter
to bf16 after loading (`M:from_pretrained`), so `A_log` / `dt_bias` are bf16 values used in fp32 arithmetic.

### 1.3 The window softmax (`softmax_attention: chunk 5, radius 1`, `anchor_frames both`)

Exact softmax over a mask on the packed sequence `[text | keyframe rows | audio | video]` (`R:generate_latents`):

- **video query, video key**: kept iff the key frame is in the query frame's window or either frame is an anchor. The window is
  **chunk-aligned**: frame `t` is in chunk `c = t // 5` and sees whole chunks `c-1..c+1`, frames `[(c-1)*5, (c+2)*5 - 1]`
  clamped (`M:window_bounds`), i.e. 10-15 frames. Anchors: every video query sees frames 0 and F-1 (columns), and the queries of
  frames 0 and F-1 see every row (rows).
- **any pair with a non-video row** (text, audio, FL2VA keyframe rows): kept, both directions ("global" rows).
- Bidirectional, not causal; no padding rows. Scale `1/sqrt(128)`.

Upstream runs it as FlexAttention with a BlockMask (training, sm90) or as a union of dense calls ("decomposed", the inference
default on CUDA and SGLang's only path): the global + anchor-row queries against all keys, then per chunk one varlen call against
gathered `[globals | window frames | anchor frames]` K/V. Both are the same function up to bf16 reduction order. At 768 x 1344
the 14.4 s clip (102 latent frames) keeps about 18% of the video-video pairs (blog: "18% softmax / 82% linear"); anchors add
3.57% density.

### 1.4 Audio, keyframes, text

Audio rows are global softmax rows only: they attend densely, everything attends to them, and they never enter the linear
branch (the text state reads `text_indices`, not audio). FL2VA / I2VA / L2VA keyframe rows (`MiniMaxH3PrepareLayoutStep`
`keyframe_anchors`, between text and audio) sit outside the video span, so they are global too; their timesteps are pinned at
0.999 as in H3 (`R:generate_latents`). The same checkpoint serves T2VA, I2VA, L2VA and FL2VA (model card, 2026-09-08); SGLang
serves T2VA and FL2VA and refuses Ref2VA ("not trained"). The text encoder is H3's Qwen3-VL-32B `hidden_states[50]`, unchanged.

### 1.5 LoRA targets and the merge (`M:merge_lora_state`)

`W += B @ A` in fp32, added in the weight's dtype, **scale 1.0** (the remote loader never applies alpha/rank; every VDN pair has
alpha = rank, so it is 1 anyway). Keys are peft names with an adapter infix (`.lora_A.default.weight`, `.lora_A.turbo.weight`)
on the hybrid module tree (`transformer_blocks.N.attn.orig.to_q`); `.attn.orig.` resolves to the base `.attn.`.

- `default` (both stages; 416 tensors, rank 64, alpha 64): `attn.orig.to_{q,k,v,out.0}` of all 50 blocks and
  `token_refiner.refiner_blocks.{0,1}.attn.to_{q,k,v,out.0}`.
- `turbo` (8-step only; 726 tensors; `family larryvrh_v4_step600_ema`): rank 64 on the same attention targets plus
  `ff.net.0.proj`, `ff.net.2` (blocks and refiner); rank 16 / alpha 16 on `transformer_blocks.N.adaln_proj.linear` and
  `norm_out.linear`. Applied after `default`.

The merge happens before any quantization. The branch itself has no LoRA.

### 1.6 Schedule and guidance

`R:generate_latents`: `MiniMaxH3Scheduler(shift=12)` for video and `shift=3` for audio, `set_timesteps(num_steps + 1)`, i.e.
the uniform grid `linspace(1, 0, N+1)` shifted per modality (`sigma' = s*sigma / (1 + (s-1)*sigma)`), Euler steps,
`t = 1 - sigma` fed to the DiT. **No CFG**: one conditional forward per step for both models (the 50-step stage-B model
included; nothing in `render.py` or the configs builds a negative branch). 8-step = 9 grid points; 50-step = 51 points.
`metadata.json` of the DMD stage: `turbo_num_steps 8`, `video_shift 12.0`, `audio_shift 3.0`, `objective dmd2_no_gan`.
This is our `base-<N>step` contract (`H3InferenceContract::base(N+1)`: uniform grid, shifts 12 / 3, guidance 1, dense).

The noise: video `randn([1, 24, F, H/16, W/16])` then audio `randn([2 * Na, 32])` from one generator (keyframe noise first);
the oracle injects its own.

### 1.7 Resolution and frames

Upstream renders 768 x 1344 (`LATENT_H, LATENT_W = 48, 84`, a `24 x 42` token grid, 1008 tokens per latent frame) and 345 frames
by default (`align_num_frames(345, 17, 5)` = 345, 14.4 s at 24 fps, 102 latent frames, 100 of them in the linear branch). The
diffusers pipeline accepts H3's canvas and 5-15 s range. Our H3 geometry (768 / 480 short edge, 4-15 s) applies unchanged.

### 1.8 fp8

Opt-in, never the default (`spec.fp8 = false`; `Fp8Config.enabled = false`, "fp8 changes the sample"): torchao / `Fp8Linear`
e4m3 on every Linear at least 4096 wide on both sides (the QKV projections, `to_out`, both FFN GEMMs and `to_out_linear`),
after the LoRA merge; activation scales per row on sm90 and per tensor on sm100 (a dispatch fact of torch 2.13), weight scales
per output channel / per tensor likewise, fast accumulation, activation amax floored at `448e-12` (rows of exact zeros reach
`to_out_linear`: the anchor frames). `skip_end_blocks` keeps the first and last N blocks bf16 (4 by default, 0 in the
`8nfe_tuned_fp8` recipe). The gates, `beta_proj`, the alpha MLP and AdaLN stay bf16 / fp32 (narrow). Our MXFP8 path
(sm_120) is a different fp8 recipe (32-element block scales) and is evaluated in §5.

### 1.9 Inference-kernel numerics

`inference_kernels = true` (the default in both `config.json` files) swaps eager ops for compiled fusions that round once instead
of per op: QK-norm + RoPE, the block's norm/modulate and gated residual, SwiGLU, the branch's tconv+SiLU+L2 (Triton), the
RMSNorm + gate epilogue, the gather, and TF32 for the `A` GEMM. Upstream calls these "not bitwise, ~1 bf16 ulp, in the accurate
direction". The port follows the inference path (one rounding per fused stage, fp32 accumulation).

## 2. The port

### 2.1 Where it lives

| piece | file |
|---|---|
| checkpoint reading, transform config (`chunk 5`, `radius 1`, `linear_head_dim 128`, strict JSON), window bounds, adapter key rerooting, packed-row layout, gather plan, recipe names (`vdn-h3` = 50, `vdn-h3-<N>step`), and the **f32 / f64 host reference** of the whole branch and of the window softmax | `crates/fastvideo-models/src/h3/vdn.rs` |
| schedule: VDN recipes map to `H3InferenceContract::base(N + 1)` (uniform grid, shifts 12 / 3, guidance 1) | `crates/fastvideo-models/src/h3/config.rs` |
| the window softmax kernel `flash_mma_fwd2_window_d128` (our FA2 MMA kernel with a chunk-aligned tile walk) | `crates/fastvideo-cudarc/src/wan/kernels.cu`, `wan/attn.rs` (`VdnWindow`), `wan/nn.rs` (`sdpa_vdn_window`) |
| the linear-branch kernels | `crates/fastvideo-cudarc/src/h3/vdn.cu` (AOT table `AOT_H3_VDN` in `build.rs`, IEEE flags) |
| branch loader, plan, launchers, GPU self-check | `crates/fastvideo-cudarc/src/h3/vdn.rs` |
| the hybrid block (`AttnMode::Vdn`, `Attention::forward_vdn`, `H3Transformer::attach_vdn`) | `crates/fastvideo-cudarc/src/h3/transformer.rs` |
| LoRA merge at load (`H3LoraFuse::open_vdn`) | `crates/fastvideo-cudarc/src/h3/lora.rs` |
| pipeline option `vdn` / `FASTVIDEO_H3_VDN`, per-request `VdnPlan` | `crates/fastvideo-cudarc/src/h3/pipeline.rs` |
| serve catalog `vdn-h3` (`H3Recipe.vdn_weights`) | `crates/fastvideo-engine-service/src/cuda/caps.rs` |
| oracle, pod runner, validation driver | `scripts/gpu/upstream/oracle_vdn.py`, `scripts/gpu/vdn-h3-pod.sh`, `scripts/gpu/vdn-h3-validate.sh` |

### 2.2 The block

`Attention::forward_vdn` replaces the dense attention of all 50 DiT blocks (the text refiner is untouched):

1. the fused QKV + gate GEMM as in H3; one more stacked GEMM (`aux`) gives `beta_proj` (H columns), the output-gate down
   projection and the softmax gate (H) in one pass over the normalized hidden state;
2. QK-norm + RoPE on q / k for the softmax only; the branch reads the **raw** q / k / v (§1.2);
3. the window softmax (§2.3); the per-(row, head) softmax gate `sigmoid(sg)` multiplies its output; `to_out` as usual;
4. the branch (§2.4) writes only the inner video rows (frames 1..F-2); `to_out_linear` and a row-add into the attention output.

The anchor frames, text, audio and keyframe rows see the dense softmax (they are global rows and columns); a clip whose
latent frames all fall inside every row's window (`VdnTransform::full_cover`) degenerates to dense attention with no branch, which is
bitwise our dense kernel (test `a_covered_clip_with_an_open_gate_is_dense_attention`, GPU check `full_cover_vs_dense`).

### 2.3 The window softmax

`flash_mma_fwd2_window_d128` is `flash_mma_fwd2` (bf16 MMA, fp32 softmax) with a key-tile walk over at most three merged
ranges per query tile (the window of its frames, the anchor frames, the global rows); tiles wholly inside the mask run
unmasked, and only boundary tiles test `vdn_win_lo / hi / allows` per score. Query tiles of global rows walk every key.
Cost scales with `window / total` frames: at 5 s (32 latent frames) it is 0.73x of dense at 480p and 0.61x at 768p.

### 2.4 The linear branch kernels (stage 1: fp32 CUDA cores)

| kernel | does |
|---|---|
| `vdn_act_rows` | q = l2(silu(q)) per head (128) |
| `vdn_sconv5`, `vdn_tconv5_act` | 5 x 5 spatial then 5-tap temporal depthwise conv on k and v; then silu (+ l2 for k) |
| `vdn_frame_stats` | per frame and head A = sum beta k k^T (fp32, symmetrized), B = sum bf16(v beta) k^T (rounded to bf16) |
| `vdn_frame_mean`, `vdn_alpha_down`, `vdn_alpha_up` | KDA decay alpha = exp(-exp(A_log) softplus(up(down(mean)) + dt_bias)) |
| `vdn_solve` | Gauss-Jordan (I + A)^-1 in 66 KB of shared memory, T = Diag(alpha) (I + A)^-1, J = B (I + A)^-1; text state 0.5 B_t (I + A_t)^-1 |
| `vdn_scan` | forward and reverse scans over frames (one CTA per head, 32-row block and direction) |
| `vdn_logprefix`, `vdn_gather` | the alpha bridge (log-prefix in double) and the bidirectional gather |
| `vdn_readout` | r = state q (bf16 state), RMSNorm (eps 1e-6) x sigmoid(og_up(og_down x)) |
| `vdn_head_gate`, `vdn_add_rows`, `vdn_copy_cols` | the softmax gate and the row bookkeeping |

Every kernel matches the f32 / f64 host reference in `fastvideo-models::h3::vdn` (§4). Tensor-core versions of
`frame_stats`, `scan` and `readout` are the next stage; at 5 s the branch costs about as much as the window saves (§6).

### 2.5 Frames and rows

`VdnLayout::from_packed` reads our packed H3 row order (text, video frames, audio, keyframes) and gives the inner video
rows; `VdnPlan` (per request) holds the window bounds, `branch_active` and the gather plan, uploaded once. Clips of any
H3 length work: the 14.4 s / 345-frame clip (102 latent frames, 100 in the branch, 100,800 inner tokens at 768p) is
covered (the spatial conv puts tokens on grid x for that reason).

### 2.6 LoRA merge at load

`H3LoraFuse::open_vdn(base, adapters)` reads peft keys (`…attn.orig.to_q.lora_A.default.weight`), reroots `.attn.orig.`
to our `.attn.`, concatenates ranks across adapters per target (the turbo adapter's mixed rank 16 / 64 and its FFN, AdaLN
and `norm_out` targets included) and merges with multiplier 1.0 into the bf16 base weights at load, before any
quantization. Stage B: rank 64 on attention q / k / v / out.0 of the 50 blocks and the refiner, 208 pairs.
The manifest tests pin every adapter key to a published h3-base parameter (208 pairs for the default adapter, 363 for
turbo, from the safetensors headers).

### 2.7 fp8

The served default on sm_120 keeps our MXFP8 linears (blocks 2-46, as FastH3 V2); the branch's narrow projections stay
bf16 and the attention itself is bf16 (`AttnMode::Vdn` never takes the MX attention path). `FASTVIDEO_H3_QUANT=off`
gives bf16 linears like the upstream default. Parity of both is in §5.

## 3. Weights and licence

R2 tree `vdn-h3-stage-b-2000` (docs/ops/r2-weights.md §13): `stage-b-step-2000/linear_branch/` (4,279,428,112 B,
`sha256:dec6981c…`) and `stage-b-step-2000/adapters/default/adapter_model.safetensors` (334,026,912 B, `sha256:58558fef…`)
plus configs, `LICENSE`, `README.md`, `modular_model_index.json`; 11 files, 4,613,679,322 B. Recorded in
`scripts/gpu/weights-{manifest,revisions,sha256,r2}.tsv` and `verify-weights.sh` (cell `vdn-h3` = h3-base + the branch
and adapter). The DiT, VAEs, text encoder and tokenizer are our `h3-base` (the repository's `h3-base/` is MiniMax-H3
@ `42ed227`, 28 of 29 files byte-identical, README aside).

Licence: the weights carry the **MiniMax H3 Community License** (the base model's), whose territory clause excludes the
EU, UK, Korea and USA; the owner reportedly holds an H3 licence, which this inherits. The code repository is Apache-2.0
with a NOTICE; nothing of its code is vendored here (the port is a reimplementation; the oracle imports the remote module
on the pod).

## 4. Tests

- `fastvideo-models` h3 suite: 130 passed, 11 of them VDN (transform parsing and strictness, window bounds and full cover,
  adapter key rerooting, the packed layout and gather plan, the host branch against a direct dense reference, recipe
  names and contracts), plus manifest tests `vdn_branch_asks_for_exactly_the_published_branch`,
  `vdn_adapters_target_published_base_parameters`, `vdn_configs_parse`.
- `fastvideo-cudarc` h3 / `wan::attn` / `wan::quant`: 114 passed, including the CPU tests
  `a_covered_clip_with_an_open_gate_is_dense_attention`, `the_branch_writes_only_the_inner_video_rows`,
  `the_window_softmax_is_the_reference_mask`.
- `fastvideo-engine-service --features cuda` `cuda::` 20 passed (`vdn_catalog_entry` new); without cuda 65 passed;
  `fastvideo-serve` e2e `shipped_configs_parse` passed (the new config).
- control: `tsc`, `tsc -p ui`, vitest 445 / 445, integration 36 / 36 steps.
- GPU (`fv-gpucheck kernels`, group `vdn`, RTX PRO 6000), each kernel against the host reference, rel-L2:

| check | rel-L2 | check | rel-L2 |
|---|---:|---|---:|
| window attention vs f64 | 2.05e-3 | full cover vs dense kernel | 0 (bitwise) |
| q features | 1.58e-3 | spatial conv | 1.66e-3 |
| temporal conv | 1.67e-3 | stats A / B | 8.7e-8 / 1.3e-6 |
| KDA alpha | 7e-8 | solve T / J | 3.0e-7 / 4.5e-7 |
| text state | 4.6e-7 | scans | 2.9e-7 |
| gather | 1.66e-3 | readout | 2.35e-3 |
| branch end to end | 4.23e-3 | | |

(1e-3-level entries are bf16 outputs against an f64 reference.)

## 5. Parity against the upstream reference (RTX PRO 6000, 2026-10-14)

Reference: the remote `modeling_vdn_h3.py` of `stage-b-step-2000` through diffusers `3a2f35d` + the code repo's patch,
torch 2.13.0 cu129, `scripts/gpu/upstream/oracle_vdn.py --kernels eager --softmax ref` (the eager path, exact masked
softmax), the `render.py` sampler replicated (uniform grid, shifts 12 / 3, no CFG, Euler). Geometry 832 x 480, 124 frames
(32 latent frames, 30 in the branch), seed 1024, **8 steps on the stage-B weights** (the approved tree; the same per-step
work as the 8-step model, a cheaper comparison than 50). Upstream ran on our text encoder's hidden states; ours then ran on
the reference's noise and text (`FASTVIDEO_INJECT_DIR`), so every remaining difference is the denoiser's. `fv-gpucheck
compare-dumps`, rel-L2 of ours against the reference (`artifacts/ports/vdn-h3/validate/compare-dumps-inj{,-served}.txt`):

| tensor | ours, bf16 linears (`FASTVIDEO_H3_QUANT=off`) | ours, served default (MXFP8 blocks 2-46) |
|---|---:|---:|
| sigmas, timesteps, injected noise | 0 (bitwise) | 0 |
| text hidden states (our streamed bf16 encoder vs the injected) | 3.8e-4 | |
| step-1 block 0 / 1 / 10 / 20 / 30 / 41 / 49 | 1.4e-2 / 6.4e-3 / 5.6e-3 / 4.8e-3 / 1.9e-2 / 7.1e-2 / 3.4e-2 | 1.4e-2 / 6.4e-3 / 8.2e-3 / 1.6e-2 / 5.8e-2 / 0.15 / 6.1e-2 |
| **step-1 velocity, video / audio** | **1.9e-2 / 1.1e-2** | 3.2e-2 / 3.7e-2 |
| video latents after steps 1 / 2 / 4 / 8 | 4.2e-4 / 1.1e-3 / 5.7e-3 / 0.13 | 7.3e-4 / 1.9e-3 / 1.5e-2 / 0.34 |
| audio latents after steps 1 / 8 | 5.5e-4 / 0.11 | 2.0e-3 / 0.24 |

Decoded (`av_psnr.py`; `artifacts/ports/vdn-h3/validate/av-psnr-inj{,-served}.json`):

| pair | video PSNR | min frame | audio PSNR |
|---|---:|---:|---:|
| upstream vs ours, bf16, injected | **29.0 dB** | 26.0 dB | **32.7 dB** |
| upstream vs ours, MXFP8 (served), injected | 20.0 dB | 18.5 dB | 22.9 dB |

Reading: the hybrid attention, branch and merged LoRA track the reference far more closely than our other H3 ports did
(FastH3 Trim bf16: 16.6 / 15.9 dB, step-8 latents 0.47): the window softmax is exact (no top-k selection) and the
branch is deterministic, so nothing amplifies a rounding difference into a different sample. Block 0's 1.4e-2 (then
6e-3 from block 1) is most likely the rounding of the merged LoRA weights (both sides merge `W + BA` into bf16, in a different
operation order) on the block whose inputs are largest; it does not grow. The clips show the same person,
framing and room (`artifacts/ports/vdn-h3/samples/frame62-upstream-ours-ours768.jpg`: upstream, ours bf16, ours 768p).

**MXFP8 stays the sm_120 default**, as for every other H3 model: it is 1.20x faster per step (2.81 vs 3.37 s at 480p)
and the first steps stay at the bf16 level (step-1 latents 7.3e-4); the 9 dB lower clip PSNR is the later-step
divergence of a sample that is still the same scene. Upstream's own fp8 (opt-in) "changes the sample" too.
`FASTVIDEO_H3_QUANT=off` gives the bf16 reference behaviour.

## 6. Timing (RTX PRO 6000 Blackwell 96 GB, same pod, served defaults, `--warm`)

One process per model and geometry; a cold generation, then the measured one (text from the cache). Ours: MXFP8 linears,
FP8 text encoder resident (streamed for the long clips), DiT resident. VDN runs use the stage-B weights on the 8-step
grid: per step identical to the 8-step model (its turbo LoRA is merged at load, no runtime cost). Logs:
`artifacts/ports/vdn-h3/validate/time-*.log`.

| warm | VDN-H3, 8 steps | FastH3 V2 8-step, dense | h3-turbo (FastH3 4-step VSA) |
|---|---:|---:|---:|
| 832 x 480, 124 f: denoise (s / step) | 22.50 s (2.81) | **17.04 s (2.13)** | **7.03 s (1.76)** |
| 832 x 480: end to end | 25.60 s | 20.15 s | 10.13 s |
| 1344 x 768, 124 f: denoise (s / step) | **60.73 s (7.59)** | 67.19 s (8.40) | 19.18 s (4.79) |
| 1344 x 768: end to end | 67.64 s | 74.11 s | 26.12 s |
| peak memory (768p) | 61.5 GiB | 61.9 GiB | 67.9 GiB |

Long clip, 345 frames (14.4 s, 102 latent frames, 100 in the branch), VDN 8 steps, encoder streamed: **768p fits**,
23.3 s / step, denoise 186.3 s, decode 18.2 s, end to end 205.4 s, peak 47.7 GiB; 480p 8.0 s / step, 72.2 s end to end,
peak 39.3 GiB (`samples/ours-768-345f-8step.mp4`, `frames-50step-long768.jpg`). Dense attention at that length is not
measured here (quadratic: 3.2x the tokens of the 5 s clip). The 50-step stage-B model at 480p: 140.8 s denoise
(2.84 s / step; `samples/ours-480-50step.mp4`). Upstream's eager reference on the same pod: 4.22 s / step at 480p
(its fused-kernel path was not timed).

Reading: at 5 s the hybrid is a 1.11x win over dense at 768p and a 1.32x loss at 480p, where the stage-1 branch kernels
(fp32 CUDA cores, about 22 ms per block at 480p, 35 ms at 768p) cost more than the window saves. Its case is long clips,
where the softmax cost stays linear in frames. h3-turbo (4 steps, VSA) remains 3x faster at 5 s. Next: tensor-core
`frame_stats` / `scan` / `readout` and fusing the feature convs (§7).

Load: 316-331 s for the hybrid (h3-base read, 4.3 GB branch, LoRA merge and MXFP8 quantization), vs 205-209 s for
h3-turbo on the same pod; not optimized.

## 7. Cost, cleanup and what is not done

- Pods: publish `xfrhraloihumz6` (CPU, $0.24/hr, ~3 min) and validation `lldpc5jbqska0t` (RTX PRO 6000, $2.49/hr,
  00:38-01:57 UTC, about $3.3); both deleted and verified gone (`GET /pods/<id>`: not found); the validation bundle in
  `fv-weights-staging/tmp/` removed. Builds and CPU tests on the shared build pod.
- **Not done:** the 8-step distilled model (needs the 0.85 GB turbo adapter, §0; the loader and the manifest test for it
  are in place); image-to-video and first/last-frame (keyframe rows are handled as global rows in the masks, but no
  serve task is exposed); tensor-core branch kernels; upstream's fused-kernel / flex-softmax path as a second oracle
  (the eager path was used); an H100 run (sm_90: the window kernel is our FA2 MMA kernel, the branch kernels are
  generic CUDA, built but not run there); a served end-to-end request through fv-serve on a GPU (the catalog entry,
  config and preset are tested on CPU only); dense timing at 345 frames.
