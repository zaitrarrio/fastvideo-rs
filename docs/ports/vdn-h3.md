# VDN-H3 (Video DeltaNet on MiniMax-H3)

Target: [`OpenVDN/vdn-minimax-h3`](https://huggingface.co/OpenVDN/vdn-minimax-h3) at
`2ce18deb1d1c6b115d6bb3732f3750fdbac95173` (lastModified 2026-10-05), code
[`OpenVDN/vdn-minimax-h3`](https://github.com/OpenVDN/vdn-minimax-h3) at `e262cb5` (2026-10-04), paper arXiv 2609.20744,
blog https://openvdn.github.io/. Owner request 2026-10-13: "let's implement https://huggingface.co/OpenVDN/vdn-minimax-h3".
Download approval (2026-10-13): `stage-b-step-2000/` plus the small root files only.

Status: see §0.

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
