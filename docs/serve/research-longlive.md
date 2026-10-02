# LongLive 1.0 / 2.0 / Plug: method notes, weights, and a port plan

Date: 2026-10-01 (Plug results §12: 2026-10-02). Status: research (Phase A). LongLive-1.3B inference is
implemented on branch `wip/longlive`. It is opt-in and has no GPU run yet. No
weights are downloaded (§8 says why). Nothing in the default serve path
changes.

## 0. Sources

| Key | Source (pinned) |
|---|---|
| `LL/…` | `github.com/NVlabs/LongLive` @ `fb16a879f46e604df4a0ca48ce2bf45adfa90352` (2026-09-30). `LL1/` = `LongLive1.0/`, `LL2/` = `LongLive2.0/`, `LLP/` = `LongLive-Plug/` |
| HF cards | Hub API (`/api/models/<repo>?blobs=true`) and `raw/main/README.md`, read 2026-10-01, at the revisions in §4 |
| papers | arXiv 2509.22622 (LongLive, 2025-09-26), 2605.18739 (LongLive-2.0, 2026-05-18), 2609.38154 (LongLive-Plug, 2026-09-29): abstracts via the arXiv API |
| docs sites | `nvlabs.github.io/LongLive/docs/`, `…/LongLive2/docs/`, `…/LongLive-Plug/` (text read 2026-10-01) |
| ours | `crates/…`, `docs/…` in this repo |

**INFERRED** marks my own reading where no source states it. **UNVERIFIED**
marks a claim in the task brief that I could not confirm at the sources.
Nothing was measured on a GPU for this document.

---

## 1. LongLive 1.0 (LongLive-1.3B)

### 1.1 What it is

Wan2.1-T2V-1.3B, made causal and few-step the Self-Forcing way, then tuned
for long interactive rollouts. Training has two stages (`LL1/configs/`):

1. **`longlive_train_init.yaml`.** Self-Forcing DMD from `ode_init.pt` (real
   score from Wan2.1-T2V-14B, fake score from 1.3B), on 21 latent frames.
   It already uses the short window and the sink (`local_attn_size: 12`,
   `sink_size: 3`). 700 iterations. The result is `longlive_init.pt`, which
   is the Hub's `longlive_base.pt`.
2. **`longlive_train_long.yaml`, streaming long tuning.** LoRA rank 256 /
   alpha 256 (`apply_to_critic: true`), `distribution_loss: dmd_switch`,
   `streaming_training: true`, chunks of 21 latent frames up to 240,
   prompt switches at random block boundaries (`switch_choices` 21 … 201),
   `global_sink: false`. 3000 iterations. The result is the Hub's `lora.pt`.

So **the LoRA is not an optional variant.** Base + LoRA *is* LongLive. The
base alone is the short-clip init (§7.1). The abstract puts the fine-tune at
32 GPU-days. The README says 32 H100 GPU-days.

### 1.2 Inference contract (`LL1/configs/longlive_interactive_inference.yaml`)

| Setting | Value | Notes |
|---|---|---|
| `denoising_step_list` | `[1000, 750, 500, 250]` | `warp_denoising_step: true`: mapped onto `FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True).timesteps` (`LL1/pipeline/causal_inference.py`). This is SF-Wan's `SelfForcingSchedule` exactly. |
| `model_kwargs.timestep_shift` | 5.0 | (the wrapper's default is 8.0; the config overrides it) |
| `num_frame_per_block` | 3 latent frames | 12 pixel frames at 16 fps (9 for block 0) |
| `local_attn_size` | 12 latent frames | KV cache size = 12 · 1560 tokens; **the sink counts inside it** |
| `sink_size` | 3 latent frames | the first block |
| `global_sink` | `true` | the sink survives a re-cache (§1.4) |
| `context_noise` | 0 | the clean pass after each block runs at t = 0 |
| canvas | 480 × 832 (latent 60 × 104, 1560 tokens per frame) | |
| `num_output_frames` | 240 latent (≈ 60 s) interactive; 120 single prompt; 1050 infinity | |
| `switch_frame_indices` | `40, 80, 120, 160, 200` (latent frames) | six prompts, one per 10 s |
| adapter | PEFT LoRA r = 256, alpha = 256, bf16, on every `nn.Linear` of every `CausalWanAttentionBlock` | `LL1/utils/lora_utils.py` |

### 1.3 Frame sink + short window (`LL1/wan/modules/causal_model.py`)

- **Cache.** One rolling KV buffer per layer, `local_attn_size · 1560`
  tokens. A new block that does not fit evicts the oldest tokens *after* the
  sink: `[sink, sink + evicted)` is dropped and the rest moves down.
- **What a query block reads.** The sink (`k[:, :sink_tokens]`) plus the newest
  `max_attention_size − sink_tokens` tokens up to the block's end.
  `max_attention_size = local_attn_size · 1560`. That is **3 sink + 9 local
  frames = 12**, the block's own 3 frames included, with no mask inside the
  window (`attention(roped_query, k_cat, v_cat)`). Before the cache fills,
  the query reads everything so far.
- **Sink protection.** A write that *recomputes* cached frames
  (`is_recompute = current_end <= global_end and current_start > 0`) starts
  at `max(local_start, sink_tokens)`. The sink slots are not overwritten,
  unless `sink_recache_after_switch` (the `global_sink: false` re-cache).
  Ordinary blocks, and every forward of block 0, write in full.

This is exactly our rolling `KvSpec` with `local_attn_frames = 12`,
`sink_frames = 3`. Our `max_attention = capacity = 12 frames` reads
`[local_end − 12f, local_end)`, which is the sink plus the newest 9 once the
cache is full (`crates/fastvideo-cudarc/src/wan/causal.rs`). Only the
recompute guard was missing (§7.1).

### 1.4 KV re-cache at a prompt switch (`LL1/pipeline/interactive_causal_inference.py::_recache_after_switch`)

It runs once per switch, before the first block of the new segment:

1. With `global_sink: false`, zero the KV tensors. The pointers are kept,
   because the commented-out lines leave `global_end_index` / `local_end_index`
   alone.
2. Reset the cross-attention cache, so the next forwards encode the *new*
   prompt.
3. If `current_start_frame == 0`, return.
4. `n = min(local_attn_size, current_start_frame)` and
   `start = current_start_frame − n`. Take `output[:, start:current]`, the
   clean latents already generated.
5. Run one generator forward of those `n` frames at `timestep = context_noise`
   (0), with the new prompt's conditioning, at `current_start = start · 1560`,
   through the same KV cache. With RoPE at their own frames this is a
   *recompute*: `current_end == global_end`, so the pointers do not move and
   the rolling branch is not taken. The keys and values overwrite slots
   `[0, n)`. Under the global sink, slots `[0, sink)` are kept (the guard)
   **unless `start == 0`**. Then `current_start > 0` fails and the sink is
   re-written too: a switch within the first 12 frames re-encodes the sink
   under the new prompt. Inside this forward the 12 queries attend, unmasked,
   to the old sink plus the 9 newly written frames. The first 3 recomputed
   frames' K/V fall on the sink slots and are dropped.
6. Reset the cross-attention cache again.

Cost: one forward of 12 frames (18 720 tokens) per switch, about 4 block
forwards' worth of work (**INFERRED**). The paper's point is that this gives
"smooth, adherent switches". Keeping the cache only conditions the next
blocks on the new prompt, which adheres slowly. A full reset loses
continuity.

### 1.5 RoPE: absolute, and the "infinity" relative mode

- **Released interactive config.** `causal_rope_apply(start_frame =
  current_start // frame_seqlen)`. Keys are cached *roped at their absolute
  frame*. The sink stays at positions 0–2 while the queries move on. Wan's
  temporal RoPE table has 1024 positions (`rope_params(1024, …)`), which
  explains the stated limit: "up to 240-second videos" (960 latent frames at
  16 fps).
- **`longlive_inference_infinity.yaml`** (`use_infinite_attention: true`,
  `LL1/wan/modules/causal_model_infinity.py`, "adapted from Infinity-RoPE",
  README news 2026.01.11). The cache holds **un-roped** keys. Each forward
  ropes the cached window at its *slot* indices `0 … num_cache_frames − 1`:
  the sink at 0–2, then the rolled frames. The queries are roped at their own
  slot indices, the tail of the window in steady state (9–11). Positions stay
  in `[0, 12)` for ever.
- **Our equivalents.** The absolute mode is our `KvRope::Absolute`. The
  infinity mode is FastVideo's `relativistic` policy, our
  `KvRope::Relativistic`: un-roped cache, window roped from 0, queries at the
  tail. `KvRope::RebasedSink` gives the same query-key offsets at the
  absolute policy's cost: it re-ropes only the sink, to sit just before the
  rolled part. A test checks this:
  `causal::tests::rebased_sink_matches_relativistic`. During a re-cache the
  rebased sink's target (`sink_target(current)`) equals the re-cache's first
  frame, so the offsets match there too. Verdict: **reuse, do not
  re-implement.** LongLive mode defaults to `Absolute`, which is faithful to
  the released interactive config. `relative_rope` picks `RebasedSink`, or
  `Relativistic` on request.

### 1.6 Weights format (Hub `Efficient-Large-Model/LongLive-1.3B`)

- **`models/longlive_base.pt`** (5 676 334 208 B, torch pickle): a dict with
  `generator` (and possibly `generator_ema`; `use_ema: false` reads
  `generator`). The state dict is `WanDiffusionWrapper`'s, so keys are
  `model.<original Wan name>`: `patch_embedding`,
  `text_embedding.{0,2}`, `time_embedding.{0,2}`, `time_projection.1`,
  `blocks.N.{self_attn,cross_attn}.{q,k,v,o,norm_q,norm_k}`,
  `blocks.N.norm3`, `blocks.N.ffn.{0,2}`, `blocks.N.modulation`,
  `head.head`, `head.modulation`. The size fits 1.42 B parameters in f32
  (**INFERRED**: the dtype is not in any source I read; the converter logs
  it).
- **`models/lora.pt`** (2 800 056 690 B): `{"generator_lora": …}`, PEFT
  `get_peft_model_state_dict` keys
  `base_model.model.blocks.N.<module>.lora_{A,B}.weight`. That is 10 linears
  per block (q, k, v, o of both attentions, `ffn.0`, `ffn.2`) times 30 blocks
  = 600 tensors. The size is about twice a single rank-256 f32 adapter, so
  the file probably also holds the critic's LoRA (`apply_to_critic`).
  **INFERRED**.
- Loading (`LL1/interactive_inference.py`): `load_state_dict(generator)`,
  then `peft.get_peft_model` (r 256, alpha 256), then
  `set_peft_model_state_dict(generator_lora)`, then `.to(bfloat16)`. PEFT
  computes `base(x) + B(A(x))` unmerged. We merge `W + (α/r)·B@A` once in
  f32 and round to bf16. That is numerically close but not bit-identical
  (**INFERRED**: standard practice).

### 1.7 Claims

| Claim | Source |
|---|---|
| 20.7 FPS on one H100 (sustained, 832×480) | abstract, README, HF card |
| "24.8 FPS with FP8" (README, HF card) **vs** "INT8-quantized inference with only marginal quality loss" (arXiv abstract, docs site) | the sources disagree on FP8/INT8 |
| up to 240 s on one H100 (absolute RoPE); unbounded with the infinity config | README |
| VBench 84.87 (repo model table) | `LL/README.md` |
| ≥ 40 GB GPU memory tested (A100, H100) | HF card |

---

## 2. LongLive 2.0 (LongLive-2.0-5B, NVFP4 S4 / S2)

From `LL2/README.md`, `LL2/configs/inference.yaml`, `LL2/configs/nvfp4/`,
`LL2/pipeline/causal_diffusion_inference.py`, the docs site, the HF cards and
the abstract.

| Item | Value |
|---|---|
| Base | Wan2.2-TI2V-5B (48-channel latent, Wan2.2 VAE) |
| Training | AR teacher forcing with **Balanced SP** (clean-history / noisy-target chunk pairs per rank), directly from the diffusion model: no ODE init. Then DMD into a **LoRA** (4 → 2 steps). NVFP4 or BF16 throughout. T2V and I2V. |
| Chunk | `num_frame_per_block: 8` latent frames |
| Window | `local_attn_size: 32` latent frames |
| Sink | `inference.sink_size: 8`. `multi_shot_sink: true` anchors the first 8 frames for good (`global_sink_size`) |
| Multi-shot | At a **scene cut** (a prompt change that `_is_scene_cut` detects; `scene_cut_prefix`), the chunk just generated is **pinned** as a shot sink after its clean pass (`_pin_current_chunk`). At each **shot boundary** the model's `rope_temporal_offset` becomes `shot_index · multi_shot_rope_offset` (8): a phase jump between shots. |
| Prompts | **Per chunk** (`conditional_dict_list[chunk_index]`). The cross-attention cache is re-initialised every chunk. A prompt change does **not** recompute history (unlike 1.0's re-cache). Optional `shot_clean_recache` zeroes the KV after `current_start` before the clean pass. |
| Steps | `sampling_steps: 4`. S2: 2. `guidance_scale: 1.0` |
| Canvas | `image_or_video_shape: [1, 128, 48, 44, 80]`: latent 44 × 80 → **704 × 1280** (the brief's "1280×720" is the nominal 720p class; the config is 704). 24 fps output (`save_video(…, fps=24)`) |
| LoRA | rank 128, alpha 128. **The Hub BF16 checkpoint `model_bf16.pt` is already merged** (docs quick start: `merged_checkpoint_path`). The card's "base + LoRA" text predates it |
| NVFP4 | W4A4 generator (`model_quant`, scale rule `mse`) via **FourOverSix** (`model_4o6.pt`: materialised `quantized_weight_*` buffers) or **TransformerEngine** (`model_te.pt`). NVFP4 KV cache (`kv_quant`, `mse`, a CUDA dequant extension). `CUDA_ARCHS=100` (B200/GB200/GB300); 120 for "RTX 50/60". The 2026.05.25 news: fused Triton RoPE/adaLN, in-place quantised KV updates, faster FP4 KV dequant, pinned VAE transfers: +18.6 % |
| Decode | `streaming_vae` (chunk-wise cached decode), `async_vae` (a second CUDA stream, overlapping the next chunk's diffusion), or `vae_device` (a VAE thread on another GPU) |
| Other | TorchAO FP8 W8A8 PTQ (SM ≥ 8.9; 300 core linears FP8, six small projections bf16), SP inference for non-Blackwell, I2V (`independent_first_frame`) |
| Throughput (repo table) | BF16 24.8 FPS, NVFP4-4step 29.7, NVFP4-2step **45.7**. VBench 85.06 / 84.51 / 83.14. The table names no GPU. "On GB200" is in the brief and the project page, **UNVERIFIED** in the sources I read; the abstract: "for inference on Blackwell GPUs … 1.84× in inference" |
| Licence | NVIDIA Open Model License (§4) |

---

## 3. LongLive-Plug (once-for-all distillation LoRAs)

From `LLP/README.md`, the six HF cards and configs, and the abstract.

- **Idea.** Distil a *capability* once per backbone family as a LoRA:
  single-pass CFG, few-step sampling, or long-context error correction for AR
  rollouts. Plug it into downstream models of the same family without
  retraining, even when they add conditioning branches or output channels.
  The paper reports 54 downstream models and 8 task categories.
- **Merge** (`LLP/scripts/merge_lora.py`, README):
  `W = W_downstream + 1.0 · ΔW_few_step + 0.5 · ΔW_cfg`, with
  `ΔW = (alpha / rank) · B @ A`. Released adapters have alpha = rank.
  Inference runs 4 steps with the sampler's native CFG at **1.0**
  (conditional pass only). The script takes native-Wan-named safetensors and
  strips `base_model.model.`.
- **MiniMax-H3.** The cards say: "use this adapter and the [other] LoRA
  **separately**. Combined use … is not recommended at present." The H3
  few-step adapter is a 4-step student, run by its pinned four-call runner
  (`run_inference.py`: `minimax_h3.infer_student_4step`, base revision
  pinned in `provenance.json`). "MiniMax-H3 uses its published four-forward
  configuration" (project page). The H3 CFG adapter: teacher CFG 3.0,
  250 steps, r 128 / α 128, audio:video loss 2:1, f32. It ships a
  bit-exact SGLang fused copy too. **H3 training code: "will be released
  later."**
- **Long-context LoRA: not released** (no Hub repo in the collection; only
  the paper and page describe it).
- **Wan adapters.**

  | Adapter | Rank / α | Format | Inference contract |
  |---|---|---|---|
  | Wan2.1-14B few-step | (lightx2v export) | `generator_lora_lightx2v.safetensors` | 4 steps `[1000, 750, 500, 250]`, shift 5, 832×480×81, `enable_cfg: false` |
  | Wan2.1-14B CFG | 128 / 128 | PEFT `adapter_model.safetensors` (+ an identical-size `generator_lora.pt`) | CFG-only: native 50 steps, guidance 1.0, shift 5 |
  | Wan2.2-5B few-step | 128 / 128 | PEFT (600 tensors) | 4 steps, guidance 1.0, shift 5 |
  | Wan2.2-5B CFG | **64 / 64** | PEFT | CFG-only: 50 steps, guidance 1.0 |

- **Training budgets** (README): CFG-only 250 iterations, Wan2.1 few-step 600,
  Wan2.2 few-step 1000, on 16 GPUs for the 5B recipe.

---

## 4. Files, sizes, revisions, licences

Sizes are bytes from the Hub API (`siblings[].size`); SHA-256 values are the
LFS `sha256`. Only the files a port needs are listed. "Skip" rows are left out
on purpose.

### 4.1 Download set (35 518 190 070 bytes ≈ 35.5 GB per volume)

| Dest (proposed) | Repo @ revision | File | Bytes | LFS SHA-256 |
|---|---|---|---:|---|
| `longlive-1.3b` | `Efficient-Large-Model/LongLive-1.3B` @ `cda9138d93872ef109d9a74c270184eb3a5acc6e` | `models/longlive_base.pt` | 5 676 334 208 | `10a2aa8fcf89c77d9033f4c117405412a690e289625766619d293f0c5a208ee7` |
| | | `models/lora.pt` | 2 800 056 690 | `c4e43b87d62d4b0614b496773639f1ab170a7ee486dc23407901e9d3a5ebc07a` |
| | | `README.md`, `prompts/interactive_example.jsonl` | 7 971 + 3 429 | (git) |
| `longlive2-5b` | `Efficient-Large-Model/LongLive-2.0-5B` @ `8521079b863720a57c1a8d9b19c8d9e6ccb04c0f` | `model_bf16.pt` (merged) | 9 999 853 030 | `ec9063a44ea3c91e8ff55edcdd58dba3f1bcf6ac9091249629cb57fcebe35fd8` |
| `longlive2-5b-nvfp4-s4` | `…/LongLive-2.0-5B-NVFP4-S4` @ `427ffb75f0a3ae82714cd4b3aaa7463839da38a7` | `model_4o6.pt` | 2 945 857 184 | `b9a2b67bc1a00390d5888cbdd7f3af27c4b5c1d3281c03a895adef97669d3716` |
| `longlive2-5b-nvfp4-s2` | `…/LongLive-2.0-5B-NVFP4-S2` @ `9ab6c9c70db97e1994d87bca3b8c4ed31e5b62e2` | `model_4o6.pt` | 2 945 857 184 | `e767e58414fb8d961ea0705f710b3272723a7389dbe6dc50ee1b0dbff68b0469` |
| `longlive-plug/minimax-h3-few-step` | `…/LongLive-Plug-MiniMax-H3-few-step` @ `b3686f432d3f20d6d6462a91d4c09ba7f189c0a2` | `generator_lora.pt` (+ ~0.75 MB of code, configs, LICENSE) | 2 767 464 857 | `465bd51317544c4d0d9bfb84776927fac599c5e9be055989e9588de4172f3184` |
| `longlive-plug/minimax-h3-cfg` | `…/LongLive-Plug-MiniMax-H3-cfg` @ `de1f4e8c7310917a7bf0417ff041abb8bb7b0927` | `adapter_model.safetensors` (+ configs, LICENSE, NOTICE) | 2 767 268 872 | `04593217a4e96ccad38bf3c510b28b0b7c8207084b0638f8d7c19dd2ee478432` |
| `longlive-plug/wan21-t2v-14b-few-step` | `…/LongLive-Plug-Wan2.1-T2V-14B-few-step` @ `f125af0533b94e2cee0163a7a42bb736d59a1d37` | `generator_lora_lightx2v.safetensors` | 1 226 922 392 | `743fc9a44e118a09f932ae5a0420c88b46bb6afdf605230f115e93eb710884ba` |
| `longlive-plug/wan21-t2v-14b-cfg` | `…/LongLive-Plug-Wan2.1-T2V-14B-cfg` @ `32b8aa3d1db3d178a9c82f3731cc917a34fb7693` | `adapter_model.safetensors` | 2 453 769 728 | `1a311f6030a74e739d9705347079a131bd5a5579bd693c993026d63e36d4bea0` |
| `longlive-plug/wan22-ti2v-5b-few-step` | `…/LongLive-Plug-Wan2.2-TI2V-5B-few-step` @ `38a6ec4e8f5b644749d3072bfe1ac383d00951c0` | `adapter_model.safetensors` | 1 289 824 400 | `69807b3bdfc4d9d153a5660dcd70812f9c973eb8bc52861fea19a5e563341316` |
| `longlive-plug/wan22-ti2v-5b-cfg` | `…/LongLive-Plug-Wan2.2-TI2V-5B-cfg` @ `fa1f9280778aa27c8038fe06e29fb5f2ed6be451` | `adapter_model.safetensors` | 644 949 288 | `c16212b483e3fd3d3a8a91b9732342777addc23d08ef0be055ff442c650944d4` |

The Hub repo id in the 1.0 README's download command,
`Efficient-Large-Model/LongLive`, redirects (HTTP 307) to `…/LongLive-1.3B`.
The 2.0 cards' download commands name temporary `Perflow-Shuai/…` repos; use
the `Efficient-Large-Model` ones. **The 5B-cfg repo changed today
(2026-10-01 15:11 UTC)**: pin the revision above.

Derived (written on the volume, not downloaded):
`longlive-1.3b-safetensors/` = `scripts/gpu/convert-longlive.py` output,
`longlive_base.safetensors` + `lora.safetensors`, same keys, about
8.48 GB. Per volume that is **≈ 44 GB** with the download.

### 4.2 Left out (would need their own approval)

| Repo | File(s) | Bytes | Why |
|---|---|---:|---|
| NVFP4-S4 / S2 | `model_te.pt` each | 2 × 9 999 838 648 | TransformerEngine path (bf16-sized weights, quantised at load). Our NVFP4 work would read the FourOverSix codes or quantise the BF16 model itself |
| H3-cfg | `sglang/adapter_model.safetensors` | 2 910 369 576 | the same LoRA, fused layout for SGLang (bit-exact) |
| Wan2.1-14B few-step | `training_checkpoint/` (model + 64 optimizer shards) | ≈ 14.8 GB | training state |
| Wan 5B / 14B CFG, 5B few-step | `generator_lora.pt` | same as the safetensors | LongLive-format duplicate of the PEFT file |
| LongLive-1.3B | `assets/`, `prompts/vidprom_filtered_extended*.txt` | 13 MB + 277 MB | images; training prompt lists |

### 4.3 Licences

| Tree | Licence as stated | Commercial use | Notes |
|---|---|---|---|
| LongLive-1.3B weights | HF card front matter `license: cc-by-nc-sa-4.0`. Card body: "LongLive-1.3B model weight is under CC-BY-NC 4.0 license." | **Treat as non-commercial** | See below |
| LongLive code (all three dirs) | Apache-2.0 (`LL/LICENSE`, `LL1/LICENSE`, the `LL2` / `LLP` READMEs, docs site "License: Apache 2.0") | yes | `LL1/wan/modules/causal_model*.py` still carry `SPDX-License-Identifier: CC-BY-NC-SA-4.0` (adopted from Self-Forcing). Our port reimplements; it copies no code |
| LongLive-2.0-5B, NVFP4-S4, -S2 | `license: other`, `nvidia-open-model-license` ("Use of this model is governed by the NVIDIA Open Model License Agreement") | yes, under NVIDIA OML terms (attribution; guardrail and other clauses apply) | The cards also say "This trial service is governed by the NVIDIA API Trial Terms of Service", which looks like boilerplate for a hosted trial and not for a download. **Owner should read before serving** |
| Plug MiniMax-H3 few-step, CFG | `license: other`, `minimax-h3-community-license-agreement`. `LICENSE` is **byte-identical** (git blob `c389b458…`) to `MiniMaxAI/MiniMax-H3`'s | as H3 | H3's territory clause (docs/ports/h3-ref2v.md §1.1): **not EU, UK, KR, US** without a MiniMax licence; "MiniMax H3" display; outputs not to train other models; the USD 20 M revenue threshold |
| Plug Wan2.1-14B / Wan2.2-5B (4 repos) | `license: apache-2.0` | yes | base models Apache-2.0 |

**The LongLive-1.3B contradiction, as far as the sources go.** The card was
last edited **2025-09-29** (Hub commit history). The code repo's README
announced on **2025-11-01**: "The license has been changed from CC-BY-NC-SA
4.0 to **Apache 2.0**". The repo's own `LICENSE` and the docs site say
Apache-2.0. Before the change, the README's only licence line was the
weights line ("LongLive-1.3B model weight is under CC-BY-NC 4.0"). So the
announcement *probably* meant to cover the weights (**INFERRED**). But the
licence attached where the weights are distributed, the HF card, still says
CC-BY-NC-SA-4.0. The card body says CC-BY-NC 4.0 (without SA). The code
files derived from Self-Forcing remain CC-BY-NC-SA. Our own
`docs/serve/research-avatar-v2v.md` already lists LongLive-1.3B as
non-commercial. **Recommendation:** treat the weights as non-commercial
(research and evaluation only) until NVIDIA updates the HF card or confirms
in writing. That is an owner/legal decision; the port does not depend on it.

---

## 5. Mapping to our stack

| LongLive need | What we have | Gap |
|---|---|---|
| Causal Wan 1.3B, 3-frame blocks, 4 DMD steps, shift 5, clean pass at t=0 | SF-Wan (`wan::stream::CausalRollout`, `WanTransformer3D::forward_kv`, `SelfForcingSchedule`) | none |
| Rolling KV 12 frames with 3-frame sink | `KvSpec::rolling(…, 12, 3, …)` | none (SF-Wan defaults to 21 / 15) |
| Recompute guard for the sink | — | **done** (`CausalKvCache::set_sink_guard`) |
| KV re-cache at a switch | periodic strobe-style `Recache` (clears and restarts at 0: a different thing) | **done** (`PromptSwitch::Recache`) |
| Absolute RoPE / infinity RoPE | `KvRope::Absolute` / `Relativistic` / `RebasedSink` | none: reused |
| `.pt` checkpoints | safetensors-only loaders | **done**: converter script plus Rust rename and merge (`wan::longlive`) |
| LoRA r=256 merge | H3 / LTX LoRA merges (`crates/fastvideo-models/src/h3/lora.rs`, `ltx2/lora.rs`) | **done** for Wan names (host f32 merge) |
| CUDA graphs per cache state | `wan::graph`, static KV cache | the re-cache runs eagerly on the graph stream; pointers unchanged, so block graphs replay |
| Prompt switching on our surfaces | Reactor causal `set_prompt` → `CausalControl` → `CausalDriver::block` → `CausalRollout::set_prompt` | wired: the recipe opts in. The director serves causal models too since `wip/director-causal` (docs/serve/director-causal.md): its `prompt` updates reach the same `set_prompt` at the next block boundary |
| TAEHV per block | `taew2_1` | none |
| **2.0**: Wan2.2-TI2V-5B DiT, VAE, `taew2_2` | the 5B port (`vae22.rs`, `wan_2_2_ti2v_5b`), FastWan2.2 | causal 5B preset (8-frame chunks, window 32, sink 8); `forward_kv` is architecture-generic (**INFERRED**: not run on 5B) |
| 2.0 multi-shot sink + RoPE shot offset | single sink | a pinned-segment cache (global sink + shot sinks + rolling), a per-shot temporal RoPE offset |
| 2.0 NVFP4 W4A4 | `nvfp4.rs` (FourOverSix/LongLive MSE rule, dequant-beforehand), `nvfp4_gemm.rs` / `nvfp4_linear.rs` (LTX FFN on FP4 tensor cores, sm_120) | Wan linears on the FP4 GEMM; read `model_4o6.pt` codes or quantise BF16; activation quantisation |
| 2.0 NVFP4 KV cache | bf16 KV | FP4 KV store + dequant (or FP4-aware attention) kernel |
| 2.0 async decode | TAEHV decode inline per block | decode on a second stream (or the pacer thread), overlapping the next chunk |
| **Plug** Wan LoRA merge 1.0 + 0.5 | Wan 5B, Wan2.1 14B ports; a merge helper (this branch, f32 host) | **done** (§12): PEFT and lightx2v key maps, multi-adapter merge, UniPC-4 (5B) and LightX2V step-distill Euler (14B) |
| Plug H3 LoRAs | H3 port, FastH3 4-step, an H3 LoRA loader | **done** (§12): `.pt` reader, separate use, the four-forward grid with the fresh-noise step; H3 licence still applies |

---

## 6. Phased plan and effort

Estimates are agent-days of implementation plus GPU hours on one RTX PRO
6000 ($2.09/hr). All are **INFERRED**.

**(A) LongLive-1.3B on the SF-Wan engine.** Code is done on `wip/longlive`
(§7).
1. Weights: fetch per §4.1 to both volumes, run `convert-longlive.py` on
   each, and record manifest, revisions and hashes (rows ready in §8). About
   1 hour, under $1 of CPU pods.
2. GPU check (§9): about 1 hour, about $2–4.
3. Upstream parity (optional): run `LL1/interactive_inference.py` on the
   same pod against our rollout. Compare the first block exactly (same noise
   injected) and the re-cache visually. About 2 hours of setup plus 0.5 hours
   GPU, about $1.
4. Director: a causal director engine, so its `prompt` updates drive
   `set_prompt` on an SF-Wan/LongLive session. That needs a causal chunk
   loop in `director/engine.rs`, caps that accept `StreamCaps::Causal`, and
   `max_session_seconds` against design §5.2's causal limits. About
   2–3 agent-days.
5. Serving: register a catalog model (`longlive-1.3b`, tier none, window 12,
   sink 3) **only after the licence decision** (§4.3).

Total remaining: about 1 day plus 3 for the director, about $5 of GPU.

**(B) LongLive-2.0-5B causal Wan2.2.**
1. BF16. Causal TI2V-5B preset; `.pt` → safetensors for `model_bf16.pt`;
   Wan2.2 key renames (the same original-Wan names); 8-frame chunks, window
   32, sink 8; per-chunk prompts (the text cache already re-keys per encoder
   tensor); a multi-shot sink (pinned segments in `CausalKvCache`) and the
   per-shot RoPE offset; 704×1280 at 24 fps with TAEHV `taew2_2` per chunk.
   4–6 agent-days, about 3 GPU hours (about $6). Expected speed on a PRO 6000
   is below 24 fps (**INFERRED**). The 5B has about 4× the 1.3B's parameters
   per token. A 704×1280 chunk is 8 · 44 · 80 / 4 = 7040 tokens, against
   4680 for a 1.3B block, and attends to up to 32 frames (28 160 keys)
   against 12. It is a B200/H100-class model for real time.
2. Async decode: TAEHV on a second stream, overlapped with the next chunk.
   About 1 day.
3. NVFP4 W4A4 on Wan linears (our FP4 GEMM, sm_120 and sm_100), loading
   `model_4o6.pt` codes, the MSE scale rule. Then the NVFP4 KV cache with a
   dequant kernel. 5–8 agent-days, about 4 GPU hours. B200 checks at $6.79/hr
   are needed for the 45.7 FPS class claim.

Total: about 2–3 weeks, about $30–50 of GPU.

**(C) Plug LoRAs.**
- Wan2.2-5B: merge few-step 1.0 + CFG 0.5 into `wan22-ti2v-5b`, 4 steps,
  shift 5, CFG 1. Compare with FastWan2.2 (3-step DMD) on our eval prompts.
  1–2 days, 1 GPU hour.
- Wan2.1-14B: the same plus the lightx2v key map. 1–2 days, 1–2 GPU hours.
- H3: each LoRA alone (few-step 4-call; CFG-only 1 forward per step); compare
  with FastH3 4-step. 2–3 days, 2–3 GPU hours on the 96 GB card. **Licence
  territory issue as for all H3.**
- Downstream transfers (SCOPE, Matrix-Game 3.0 …): out of scope until a
  downstream model is in our catalog.

**(D) Causal / long H3 and LTX by the LongLive-2.0 recipe: training, owner
decision.** No training code is released for H3 ("will be released later").
2.0's recipe is: AR teacher-forcing tuning with Balanced SP, then DMD LoRA,
optionally NVFP4.
- Scale: LongLive 1.0's long tuning took 32 H100 GPU-days for 1.3B at 480p.
  The H3 DiT is 70.1 GB bf16 (≈ 35 B parameters, 26 GB of it AdaLN,
  docs/ports/h3.md). LTX-2.5 is 22 B. Per-token compute is about 15–25× the
  1.3B. 768p tokens are about 2–2.5× 480p per frame. Joint audio adds a few
  per cent.
- Estimate: AR conversion + long tuning + DMD ≈ **1 500–5 000 H100
  GPU-days** per model, i.e. **$125 k–420 k** at $3.49/hr (Runpod H100),
  before failed runs. It needs 64–256 GPUs with SP/FSDP, a long multi-shot
  captioned video set (the 2.0 abstract credits "a high-quality …
  dataset"), and about 2–3 engineer-months. **INFERRED**, order of
  magnitude only.
- Cheaper alternatives:
  - Plug's long-context LoRA, if NVIDIA releases one for H3. Inference
    only; it would follow the (C) path.
  - LTX's retake/extend for long video (bidirectional chunks), already in
    our repo's direction.
  - Running H3/LTX with a sink+window *without* training: they are
    bidirectional models, so they would need AR tuning anyway. Not viable.

---

## 7. Phase A: what is implemented (`wip/longlive`)

### 7.1 Code

- **`crates/fastvideo-cudarc/src/wan/longlive.rs`** (new).
  - `LongLiveConfig::{interactive, infinity}`: the YAML values above.
  - `rollout()` maps the config to `RolloutConfig`: steps, shift, window 12,
    sink 3, RoPE, `PromptSwitch::Recache { global_sink }`, no periodic
    re-cache.
  - `recache_plan()` / `recached_slots()` are `_recache_after_switch`'s
    arithmetic, including the `start == 0` sink rewrite.
  - `diffusers_key()` holds the original-Wan → Diffusers renames (the
    `convert_wan_to_diffusers.py` table: `norm3` → `norm2`, `ffn.0` →
    `ffn.net.0.proj`, `modulation` → `scale_shift_table` …), stripping the
    `model.` / FSDP / PEFT wrappers.
  - `lora_key()` and `merge_lora()` compute `W += α/r · B@A`, row-parallel,
    f32.
  - `build_transformer_tensors()` / `load_transformer_map()` rename, merge,
    and keep 2-D+ weights in bf16 (biases and modulation tables in their
    dtype), producing a `WeightMap`.
- **`causal.rs`.** `CausalKvCache::set_sink_guard`: LongLive's `is_recompute`
  rule. While it is on, a write that re-runs cached frames
  (`current_start > 0`, `current_end ≤ global_end`) skips the sink slots.
  This applies to both the allocating and the static (CUDA-graph) cache.
- **`stream.rs`.** `PromptSwitch::Recache { global_sink }` and a history of
  clean block latents covering the window. `set_prompt` marks a switch, and
  the next block first runs the re-cache: one `forward_kv` of the last
  `min(12, frames)` latents at t = 0 under the new prompt, guard on. Several
  switches before a block re-cache once, with the latest prompt (LongLive
  only switches at block boundaries). `BlockTimings::recache_s` and
  `switch_recaches()` report it. `global_sink: false` with `RebasedSink` is
  refused at open: that combination would need the un-roped sink copy
  retaken at the re-cache's position.
- **`pipeline.rs`.** `WanPipeline::load_with_dit(root, preset, parts, map)`:
  the transformer from a map, the rest from the SF-Wan tree.
- **engine-service.**
  - `SfWanRecipe.longlive: Option<LongLiveRecipe>` (serde default `None`,
    skipped when `None`, so the catalog and the fal schemas are unchanged).
    A deployment opts in with `weights`, `lora`, `recache`, `global_sink` and
    `relative_rope`, and loads through `load_pipeline_with_dit`.
  - The standalone `CausalCudaBackend` honours `FV_LONGLIVE_WEIGHTS`
    (+ `FV_LONGLIVE_RECACHE=0`, `FV_LONGLIVE_INFINITY=1`).
  - Reactor `set_prompt` reaches the re-cache unchanged.
- **gpucheck `wan stream`.**
  - `--longlive DIR` / `--longlive-no-lora`.
  - `--switch-prompts FILE` takes LongLive's `interactive_example.jsonl` (the
    first prompt starts the run) or plain lines; `--switch-prompts-line N`
    picks the line.
  - Run keys: `longlive=1` (window 12, sink 3, absolute RoPE, re-cache),
    `switch=recache|recache_sink`, and several `switch_at=15/30/45`.
  - Reports `switch_recaches` and a per-block `recache_s`.
- **`scripts/gpu/convert-longlive.py`.** `.pt` → same-key safetensors with a
  tensor-by-tensor round trip, add-only (temp name, rename). A local tool: it
  creates no pods.

### 7.2 Host tests (build pod, `cargo test`, all pass)

- `fastvideo-cudarc` `wan::{causal, longlive, stream, weights}`: **28
  passed.** New ones:
  - `sink_guard_keeps_the_sink_through_a_recache` checks LongLive's geometry
    (window 12, sink 3) after 15 frames, for allocating and static caches:
    sink slots kept, slots 3–11 = re-cached frames 6–14, pointers unchanged,
    and the next block evicts and reads sink + 9–14 + block.
  - `early_recache_rewrites_the_sink`: a switch at frame 6 rewrites
    everything; the guard is inert on new blocks.
  - `recache_through_the_dit` runs the tiny causal DiT through a switch with
    re-cache, for Absolute, RebasedSink and Relativistic. The allocating and
    static caches are bit-identical, the guarded sink keys are unchanged in
    every layer, and the output differs from keeping the cache.
  - `recache_plans_follow_longlive`, `configs_match_the_yaml`,
    `original_wan_keys_map_to_diffusers`, `lora_keys_parse` and
    `merge_is_w_plus_scaled_b_at_a`.
  - `transformer_tensors_rename_and_merge` covers merge, bf16 policy, and the
    refusal of stray or half LoRA pairs.
  - `renamed_map_loads_the_tiny_dit`: a generator written under original
    names, renamed, loads into `WanTransformer3D` and gives bit-identical
    outputs to the Diffusers-named weights.
- `fastvideo-gpucheck` (`--features cuda`): `longlive_run_spec` and
  `switch_prompts_from_jsonl_or_lines` pass.
- `fastvideo-engine-service` lib tests: 61 pass with `--features cuda` and
  58 without. The new `sfwan_longlive_is_opt_in` passes on its own (run after
  the full suite).
- `fastvideo-fal --test family_schemas`: 4 pass, so the schemas are
  unchanged.
- CUDA type-check of `fastvideo-cudarc`, `fastvideo-gpucheck` and
  `fastvideo-engine-service` (`--features …/cuda --all-targets`): clean. No
  new warnings.

Not covered without a GPU and weights:
- the real checkpoint's keys (the converter writes `keys-*.txt`; the loader
  errors on any unknown key);
- the f32-vs-bf16 dtype of `longlive_base.pt`;
- re-cache latency;
- picture quality.

---

## 8. Downloads: done (2026-10-02)

The owner approved the download ("longlive download approved"). All ten
Hub trees and the converted 1.3B safetensors are on **both** weight volumes,
add-only (temp name, SHA-256 check, rename):

- `scripts/gpu/fetch-hub-tree.sh <dest> <revision>` per tree, one CPU pod
  at a time per volume. **EU first** (US-CA-2 had no CPU stock at the
  first attempt), then US with the EU `sha256.txt` as the expected list.
  Every file matched the Hub (LFS SHA-256 or git blob) at the §4.1
  revisions. The 11 large files match the §4.1 SHA-256s. US = EU file by
  file. Logs: `artifacts/runpod/fetch-longlive*-fv-weights-*/`.
- Bytes per volume: 35 519 236 503 for the Hub trees (§4.1's 35 518 190 070
  plus small files). `longlive-plug/minimax-h3-few-step` is the whole repo
  (`*`): `generator_lora.pt` plus 54 small files (code snapshot, recipe,
  configs, LICENSE) of 941 933 B. No SGLang copy and no duplicate `.pt`
  LoRA was fetched. In `minimax-h3-cfg`, `*.json` also matched the small
  `sglang/adapter_config.json`, but not the 2.9 GB SGLang safetensors.
- `longlive-1.3b-safetensors`: `convert-longlive.py` on a CPU pod per
  volume. `longlive_base` is **f32**, 825 tensors (5 676 075 416 B), and
  `lora` is f32, 600 tensors (1 399 924 800 B; the `generator_lora` sub-dict
  only). The round trip is `torch.equal`. The two volumes' base files
  differ only in the header's metadata order
  (docs/ops/runpod-volumes.md §3). That adds 7 076 000 216 B per volume,
  so about **42.6 GB per volume** in all.
- Records: `weights-manifest.tsv` (licences in the comment block),
  `weights-revisions.tsv`, `weights-sha256.tsv` (106 rows),
  `verify-weights.sh` cells `longlive-1.3b`, `longlive2-5b`,
  `longlive2-5b-nvfp4` and `longlive-plug` (plus `sha:<dest>`), the
  `rebuild-volume.sh` plan, and `docs/ops/runpod-volumes.md` rows with the
  licence column. LongLive-1.3B is recorded as **non-commercial** (§4.3).

---

## 9. GPU check plan (weights on EU since 2026-10-02)

One RTX PRO 6000 (96 GB, sm_120) in **EUR-IS-1** on the EU volume
`jg48s6o1w0`, $2.09/hr. It needs `sfwan21-1.3b`, `auxiliary/tae/taew2_1`,
`longlive-1.3b(-safetensors)`, and `fv-gpucheck` built on the build pod
(`cargo build --release -p fastvideo-gpucheck --features cuda`, about 6.5
min, then `fetch`). Pod rules apply: our own pod only, a wall-clock backstop
of 90 min, delete and verify 404, balance floor $8.

Runs (`fv-gpucheck wan stream --weights $W/sfwan21-1.3b --height 480 --width
832 --fps 16`; LongLive runs add `--longlive $W/longlive-1.3b-safetensors
--switch-prompts $W/longlive-1.3b/prompts/interactive_example.jsonl`):

| # | Run | What it answers |
|---|---|---|
| 1 | `--run sf,seconds=60` (SF-Wan default: window 21, sink 15, rebased) | today's baseline on this card (15.2 fps expected, docs/perf/raw-inference.md) |
| 2 | `--run sf12,seconds=60,window=12,sink=3,rope=abs` | the same geometry without LongLive weights (speed effect of the shorter window) |
| 3 | `--run ll,longlive=1,seconds=60` | LongLive fps at 480p (sink + window), TTFF, memory, drift windows |
| 4 | `--run llsw,longlive=1,seconds=60,switch_at=15/30/45` | **60 s interactive, 3 switches, re-cache on**: `switch_recaches = 3`, `recache_s` per switch, the block p90 hit |
| 5 | `--run llkeep,longlive=1,switch=keep,seconds=60,switch_at=15/30/45` | the same, re-cache off (ablation) |
| 6 | `--run llinf,longlive=1,rope=rebased,seconds=60,switch_at=15/30/45` | infinity (relative) RoPE |
| 7 | `--run llrel,longlive=1,rope=rel,seconds=20` | `RebasedSink` ≡ `Relativistic` on real weights (latent hash / drift) |
| 8 | `--run lleager,longlive=1,graphs=0,seconds=20,switch_at=10` | graph vs eager through a re-cache (latent hashes per window) |
| 9 | optional: `--run ll240,longlive=1,seconds=240,sheet=1` | the 240 s claim with absolute RoPE (stays inside the 1024 table) |

Metrics we have:
- `wan stream`'s report: steady fps (wall and engine), block p50/p90, TTFF,
  memory over time, `recache_s` per block, per-window luma, contrast,
  motion, top-band banding (pixels and latents), fresh-decode MAD (decoder
  state drift), and contact sheets per window.
- At the switch points, compare the sheets of runs 4 and 5 by eye
  (adherence: is the new prompt visible within one or two blocks;
  continuity: no cut).
- For a VBench-style proxy, we have no VBench or CLIP scorer in the repo.
  Temporal flicker (`warp_err`, `lum_flicker`) from
  `scripts/gpu/hd_upscaler_metrics.py` applies to a recorded run: one 60 s
  Reactor causal session with `FV_LONGLIVE_WEIGHTS` (the
  `docs/serve/e2e/wan.md` recording flow), about 10 extra minutes.
- A CLIP text–frame score per segment would need a CLIP checkpoint
  (≈ 0.6 GB, a new download, so ask first).

Expectations (**INFERRED**): at 480p a block attends 12 instead of 21 frames
of keys, about 30 % fewer attention FLOPs. That suggests roughly 18–20 fps
on the PRO 6000 against SF-Wan's 15.2, which would be real time at 16 fps.
A re-cache should cost about 0.3–0.5 s once per switch (one block's delay).

Cost:
- Pod boot and image pull about 10 min. Load (UMT5 + DiT merge) about 2–3
  min per invocation; group the runs into 2–3 invocations.
- Runs 1–8: 7 × 60 s + 3 × 20 s of video at 15–20 fps, about 10 min of
  generation; with load and reports about 30–40 min.
- Optional 240 s run plus the Reactor recording: +15 min.
- **About 1 hour, ≈ $2.10, budget cap $4** (90-min backstop). Plus the build
  pod's share, about $0.20.

---

## 10. Open items

- ~~Read the real `keys-*.txt` and the dtype~~: done (§11; f32, keys load
  with nothing skipped).
- LongLive-1.3B weights licence: owner/legal (§4.3).
- NVIDIA OML: whether the "API Trial Terms" sentence on the 2.0 cards
  matters for self-hosting (owner).
- H3 Plug LoRAs: the H3 territory clause blocks serving from US/EU without a
  MiniMax licence (as for H3 itself).
- ~~The director has no causal mode (§5, §6 A4)~~: done, docs/serve/director-causal.md.

---

## 11. GPU check results (2026-10-02, RTX PRO 6000)

**Setup.** One RTX PRO 6000 Blackwell Server Edition (pod `uhkrvq8551gvsp`,
EUR-IS-1, driver 595.91.07, $2.09/hr, 29 min), with the EU volume
`jg48s6o1w0` at `/workspace` (nothing written to it). `fv-gpucheck` was
built on the build pod from `wip/longlive` and run on the runtime image
`fastvideo-rs-runtime:sha-4eeb801`
(`@sha256:dccdf956…`, the Sage Phase 3 image), driven through
`scripts/serve/e2e/pod.sh` (sidecar exec). Flags: `--mode fast`, 832×480,
16 fps, `FASTVIDEO_TAE_DIR=$W/auxiliary/tae` (TAEHV decode every block),
CUDA graphs on unless noted. Prompts: LongLive's
`prompts/interactive_example.jsonl` line 0. The first prompt runs the
whole rollout, and the switches at 15 / 30 / 45 s take prompts 1–3.
Driver, reports, sheets: `artifacts/perf/longlive-gpucheck/`.

**Loading the real checkpoint.** `longlive_base.safetensors` (825 f32
tensors) is renamed to Diffusers names, and the 300 LoRA modules
(r = α = 256) are merged at scale 1, with no unknown or skipped keys (32.7 s).
The whole pipeline loads in 82 s (UMT5 + DiT + merge). The §10 dtype
question is answered: the base is f32, merged and rounded to bf16.

### 11.1 Throughput (steady state, wall clock; block = 3 latent frames = 12 frames)

| Run | Model | Window / sink / RoPE | 10 s fps | 60 s fps | Block p50 / p90 | KV | Device mem |
|---|---|---|---:|---:|---|---:|---:|
| `sf10` / `sf60` | SF-Wan (today's default) | 21 / 15 / rebased | 14.87 | 14.80 (engine 15.09) | 0.795 / 0.795 s | 9871 MiB | 33.2 GB |
| `sf12_10` / `sf12_60` | SF-Wan | 12 / 3 / abs | 20.86 | 20.38 (20.95) | 0.573 / 0.573 s | 3290 MiB | 33.2 GB |
| `ll10` / `ll60` | **LongLive-1.3B** (base + LoRA merged) | 12 / 3 / abs | **20.96** | **20.40** (20.97) | 0.572 / 0.573 s | 3290 MiB | 26.4 GB |
| `ll240` | LongLive-1.3B | 12 / 3 / abs | | 240 s: **20.34** (20.94) | 0.573 / 0.573 s, max 0.606 | 3290 MiB | 27.5 GB, flat |
| `llrel20` | LongLive-1.3B | 12 / 3 / relativistic | 20 s: 18.03 | | 0.651 / 0.652 s | | |
| `lleager` | LongLive-1.3B, `graphs=0` | 12 / 3 / abs | 20 s: 17.55 | | 0.645 / 0.648 s | | |

LongLive runs at **+38 % fps over SF-Wan's default** (20.4 against 14.8 fps), and
above real time at 16 fps. The gain is the shorter window, not the
weights: SF-Wan with the same 12 / 3 geometry runs at the same speed. TTFF is
0.42 s warm. §9 estimated 18–20 fps; it came in slightly above that.

### 11.2 Interactive 60 s, switches at 15 / 30 / 45 s

| | `llsw` (re-cache ON) | `llkeep` (re-cache OFF) |
|---|---|---|
| fps (60 s) | 19.68 (engine 20.37) | 20.34 (20.93) |
| Re-cache per switch | **0.392 / 0.391 / 0.390 s** (one t = 0 forward over the 12-frame window) | — |
| Switch block | 0.965 s (p50 0.573 s): one block late by 0.39 s; p90 unchanged (0.574 s) | 0.57 s |
| Prompt encode | 0.020–0.022 s (UMT5 resident) | same |
| `warp_err` / `warp_err_hf` (whole 60 s) | 1.986 / 1.391 | 1.971 / 1.374 |
| `lum_flicker` (whole 60 s) | 0.649 | 0.544 |
| ±1 s around 15 / 30 / 45 s: `warp_err` (max) | 1.16 (2.12) / 2.37 (4.07) / 2.38 (3.43) | 1.24 (2.46) / 2.32 (3.71) / 2.18 (3.26) |
| ±1 s: largest frame-mean luma jump | 0.60 / 3.14 / 4.56 | 0.87 / 2.51 / 1.32 |
| 5 s segments, `warp_err` range | 1.06–3.03 | 1.06–3.01 |

Metrics come from `scripts/gpu/hd_upscaler_metrics.py`'s definitions
(Farneback flow, forward–backward consistent pixels) on every frame
(`wan stream … dump=1`, `driver/drift.py`). For scale: a hard cut gives
`warp_err` in the tens. The switch windows sit inside the range of the
normal 5 s segments.

**Continuity and adherence** (`sheets/switch-llsw.jpg`,
`sheets/switch-llkeep.jpg`: frames −1 s, −0.25 s, +0.25 s, +0.75 s,
+1.5 s and +2.5 s around each switch). Neither mode cuts. Player, table,
lighting and camera stay continuous through all three switches in both. With
re-cache the new prompt takes over sooner. At 30 s ("a patron claps", the
player settles) the arms come down and the clapping hands appear within
about 1.5 s. Without re-cache the previous prompt's arms-out pose carries
on past +2.5 s. That is LongLive's claim: the re-cache trades a 0.39 s
one-off for faster adherence without a cut. The larger luma jump at 45 s
with re-cache is the scene changing faster (a brighter wide shot), not
flicker; `lum_flicker` stays in the normal range afterwards.

### 11.3 RoPE, CUDA graphs, long run

- **Graph vs eager through a re-cache** (`llgraph` / `lleager`, 20 s, switch at
  10 s): the latent hashes of both windows are **bit-identical**
  (`40f52022…`, `0b8766e0…`). The re-cache is captured correctly: the
  switch block replays 0.391 s graph against 0.401 s eager. Eager is 11 %
  slower overall (17.55 fps).
- **Determinism:** `ll240`'s first six window hashes equal `ll60`'s.
  `llkeep` (frame dump on) equals `llkeep2` (dump off) in every window.
- **Absolute vs rebased (infinity) vs relativistic** (20 s, no switch): three
  different latent streams (`40f52022…` / `27092645…` / `b0aa674f…`), with
  similar picture statistics (luma 56.8–57.6, sharpness 14.4–14.6). So
  `RebasedSink` ≡ `Relativistic` holds on the host tests' tiny DiT
  but not bit-for-bit on real weights in bf16. Relativistic costs 12 %
  (more kernels per block). `llinf` (rebased, 3 switches) re-caches
  in 0.389–0.390 s, 19.77 fps. It stays coherent but reframes more (a wider
  shot from about 40 s, `sheets/llinf.jpg`), and absolute held the framing.
- **240 s, absolute RoPE, one prompt** (`ll240`, inside the 1024-frame RoPE
  table): 20.34 fps flat. Memory is flat at 27 478 MiB, luma 56.9–61.9,
  sharpness 14.4–16.1, and the fresh-state decode MAD 0.020–0.023 in every
  window. The 24-window sheet (`sheets/ll240.jpg`) shows **no top-band
  artefact and no banding** over 4 minutes. That is the failure SF-Wan's
  one-block sink showed by 30–45 s (`docs/serve/e2e/wan.md`, R12). The
  scene is fairly static (single prompt), so this is a stability result,
  not a motion-quality one.
- **One memory step.** `llkeep` stepped +864 MiB once (block 55 of 81) and stayed
  flat, which tripped the stage's 256 MiB growth check. The identical
  `llkeep2` (same latents) stayed flat at 27 478 MiB, as did `ll240`. So it is
  a one-off allocation in the process, not growth per block. Noted, not
  chased.

### 11.4 Verdict

**Pass.** It runs correctly on the real checkpoint (clean load, every
run's frame count). Throughput is ≥ SF-Wan (+38 %, real time at 480p), and
the re-cache works (0.39 s per switch, graph = eager bit for bit, no
cut). `wip/longlive` merges to main **default-off / opt-in**:
`SfWanRecipe.longlive` stays `None`, the catalog and fal schemas are
unchanged, and `FV_LONGLIVE_WEIGHTS` is unset by default. The weights
licence stays an owner decision (§4.3: treat as non-commercial).

**Cost of the check.** GPU pod 29 min ≈ $1.01 (about 8 min lost to two
setup mistakes: no `FASTVIDEO_TAE_DIR`, then gpucheck's default exact mode
at 1.37 fps). Build pod ≈ 45 min shared. Downloads, conversion and
verification on CPU pods ≈ $0.25.

---

## 12. LongLive-Plug: opt-in recipes and GPU check (2026-10-02)

### 12.1 What each recipe runs, and the source for it

Read at the §0 pins: the Plug README, the six cards with their
`adapter_config.json` / `inference_config*` / `provenance.json` /
`training_config*`, `LongLive-Plug/{inference.py, model/base.py,
scripts/merge_lora.py}`, the H3 few-step card's `run_inference.py` and
`source_snapshot/minimax_h3/{infer_student_4step.py,
fresh_noise_scheduler_4step.py, cfg_guidance.py}`, and LightX2V's
`WanStepDistillScheduler` (`ModelTC/LightX2V@8a97c75`).

| recipe | adapters (merge weight; r / α) | sampler | source |
|---|---|---|---|
| `h3-plug-4step` | H3 few-step `generator_lora.pt` (1.0; 128 / 128, 312 modules: 50 blocks + 2 refiner blocks, attention and FFN) | `MiniMaxH3Scheduler.set_timesteps(5)`: 4 forwards, video shift 12, audio shift 3. Each step predicts `x0 = x + (1 - t) v`, then re-noises with **fresh** noise to the next point, `t' x0 + (1 - t') n`, video before audio. The last step returns `x0`. No CFG and no negative prompt | `infer_student_4step.py` (`num_inference_steps=NUM_SIGMA_ENDPOINTS=5`), `fresh_noise_scheduler_4step.py`, `provenance.json` (`nfe 4`, `cfg false`, 768 × 1344 × 124) |
| `h3-plug-cfg` | H3 CFG `adapter_model.safetensors` (1.0; 128 / 128, 312 modules) | the base grid: 50 points, 49 positive forwards, shifts 12 / 3, Euler | `training_config.json` (50 native grid points, `student_conditioning: positive_only`, teacher CFG 3.0); `evaluation_summary.json` (trained scale 1.0, median implied teacher CFG **1.24** at that scale) |
| `wan5b-plug-4step` | 5B few-step (1.0; 128 / 128) + 5B CFG (0.5; **64 / 64**), 300 modules each | `FlowUniPCMultistepScheduler`, 4 steps, shift 5, guidance 1.0 | README ("4-step, CFG-free"), `inference.py` / `model/base.py` (UniPC rollout), card `inference_config.yaml` |
| `wan14b-plug-4step` | 14B few-step `generator_lora_lightx2v.safetensors` (1.0; bf16, bare original-Wan names, metadata r = α = 128) + 14B CFG (0.5; 128 / 128), 400 modules each | LightX2V step-distill: `[1000, 750, 500, 250]` mapped onto the shift-5 table, i.e. timesteps 1000 / 937.5 / 833.3 / 625, deterministic Euler `x' = x + (σ' − σ) v`, no CFG | card `inference_config.json` (`denoising_step_list`, `sample_shift 5`, `enable_cfg false`) |

Two facts changed the plan from the brief:

* **The released MiniMax-H3 base is already guidance-distilled.** The
  official modular pipeline exposes no `guidance_scale` or
  `negative_prompt`; the student runner refuses them; `dmd_core.py` fails
  closed on any unconditional call; `cfg_guidance.py` says "the released
  H3 base is already CFG-distilled". The CFG LoRA distils an *extra*
  guidance of 3.0 against a text-only reference condition, and the release
  does not publish that condition. So "base H3 with CFG" has no published
  form to compare against. Our port does not run one, and the "time
  saved" from the CFG LoRA is zero against the base as it ships (both use
  49 positive forwards). Against a hypothetical extra-CFG-3 base it would
  be 49 of 98 forwards. `h3-plug-cfg` is therefore compared with the base
  at the same 49 forwards.
* **The H3 four-forward grid is the one Sol-H3 already uses**
  (`H3InferenceContract::sol_h3`). Only the transition differs. Our
  scheduler had only the deterministic Euler step. `h3-plug-4step` adds
  the student's fresh-noise step (`H3InferenceContract::fresh_noise`). The
  noise comes from our own seeded generator, so the sample for a given
  seed differs from upstream's torch draws, as for every other H3 recipe.

Licences: the H3 adapters follow the MiniMax H3 Community Licence
(territory clause, §4.3). The four Wan adapters are Apache-2.0.

### 12.2 Code (`wip/plug-lora`)

* `fastvideo_loader::pth`: a torch `.pt` reader without Python. The LPIPS
  port's pickle interpreter moved here and was generalized: nested dicts
  (`student_lora` beside a config dict), `_rebuild_from_type_v2` (the H3
  file wraps every tensor in it), bf16 / f16 / f64 storages, and more
  opcodes. `lpips.rs` re-exports it.
* `fastvideo_models::plug`: the recipe catalog; LoRA key parsing (PEFT
  `base_model.model.`, `.default`, lightx2v bare names, `diffusion_model.`);
  `AdapterConfig` (alpha / r from `adapter_config.json`, else from
  metadata, else α = r; rank / alpha patterns, DoRA and rsLoRA are
  refused); `plan_adapter`, which counts matched, skipped and unknown keys
  per adapter; `delta_scale`; `merge_pair`.
* H3: the contracts `h3-plug-4step` and `h3-plug-cfg`;
  `H3JointSchedule.fresh_noise`; `pipeline::fresh_noise_step`;
  `H3LoraFuse::open_plug`, which reads the `.pt` or safetensors adapter,
  logs the plan, refuses any skip, and then reuses the FastH3 fuse path
  (host or device merge, refiner included).
* Wan: `wan::plug::load_merged_transformer`. It loads the Diffusers
  transformer, renames each adapter module with
  `longlive::diffusers_key`, adds every adapter's delta in f32, and stores
  the result in the base dtype. `WanPipeline::set_step_distill` adds the
  LightX2V Euler sampler.
* `fv-gpucheck wan gen --plug <recipe> [--plug-root]`; `h3 gen
  --h3-recipe h3-plug-4step | h3-plug-cfg`.

Host tests (build pod, all pass): loader 25, models 457, cudarc 513.
The new tests cover key mapping for every released prefix, the
matched / skipped / unknown counts (a missing base block, a module family
without a Diffusers name, a shape mismatch, an unpaired half, a non-LoRA
key, a declared-rank mismatch), the two-adapter merge math against a
hand-computed W + 1.0·(α/r)BA + 0.5·(α/r)BA (bf16 and f32 bases,
untouched tensors byte-identical), adapter_config / metadata alpha, the
`.pt` reader (root dict, nested dict beside a config, the H3 release's
`_rebuild_from_type_v2` layout, half floats), the H3 contracts, the
fresh-noise step, and the LightX2V sigmas (1 / 0.9375 / 0.8333 / 0.625,
ending on `x0`). The real `generator_lora.pt` pickle (its first 148 KB,
fetched by HTTP range) parses: 624 tensors under `student_lora`, 312 of
them `lora_A`.

### 12.3 GPU check

**Setup.** Two RTX PRO 6000 Blackwell Server Edition pods in EUR-IS-1, run
in parallel ($2.09/hr each, driver 595.91.07, 1.5 TB RAM): `614nezj1d5lnee`
(H3 768p 4-step, Wan 5B) and `ofxzv506u7rq2j` (Wan 14B, H3 49-forward).
Both mounted the EU volume `jg48s6o1w0` and wrote nothing to it. Each had
a 5400 s backstop and a 20 min idle guard. `fv-gpucheck` was built on the
build pod from `4ebe48d` and run on the runtime image `sha-4eeb801`
(`@sha256:dccdf956…`) with `--mode fast`, through
`scripts/serve/e2e/pod.sh` (sidecar exec). Driver, summaries, per-clip
statistics and sheets: `artifacts/perf/plug-gpucheck/`. Timings are
medians over the prompts in one warm process (one untimed generation
first) unless noted. "denoise" is the sampler alone; "total" adds text
(cached), decode and write. Quality uses `compare-clips` with LPIPS(alex)
and the policy's sharpness / temporal-jitter ratios (candidate over
baseline). `clipstats` adds reference-free numbers: mean luma Laplacian
variance ("sharpness") and mean |frame diff| ("motion"). There is no
vision judge, so the visual notes come from the contact sheets.

Prompts: H3 used three of the five gate prompts (`h3-demo` seed 0,
`ltx-frogyoga` and `spark-mountain-lake` seed 42). Wan used `beach_dog`
and `city_rain` (seed 1024) and `spark-mountain-lake` (seed 42). The
49- and 50-step references ran on two of these prompts, and the 768p H3
base on one, without the warm pass.

**Loads: every adapter matched completely.**

| recipe | adapter | modules matched / skipped / unknown keys | merge |
|---|---|---|---|
| `h3-plug-4step` | `generator_lora.pt` (f32, r = α = 128) | **312 / 0 / 0** | FastH3 fuse path (no extra time visible in the 136–146 s load) |
| `h3-plug-cfg` | `adapter_model.safetensors` (f32, 128 / 128) | **312 / 0 / 0** | as above |
| `wan5b-plug-4step` | few-step (128 / 128) · CFG (64 / 64) at 0.5 | **300 / 0 / 0** each; 300 base weights changed | 86.6 s host |
| `wan14b-plug-4step` | lightx2v few-step (bf16, 128 / 128) · CFG (128 / 128) at 0.5 | **400 / 0 / 0** each; 400 base weights changed | 175.1 s host |

**H3, 768 × 1344, 124 frames (5 s), 3 prompts.** Our runs carry no
technique profile (no MXFP8), so these times are not the Phase 3 table's.

| arm | forwards | denoise s | total s | load s | peak GiB |
|---|---|---|---|---|---|
| **`h3-plug-4step`** (dense) | 4 | **32.65** | **39.7** | 146 | 57.7 |
| same, bf16 control (dense kernel cuDNN → fwd2, `FASTVIDEO_CUDNN_SDPA_GRAPH=composite`) | 4 | 33.41 | 40.4 | 136 | 57.7 |
| FastH3 4-step VSA (`4step-vsa`, the h3-turbo recipe) | 4 | 19.46 | 26.5 | 134 | 67.3 |
| base H3 (`base`, dense; `spark-mountain-lake` only, no warm pass) | 49 | 401.63 | 409.0 | 113 | 57.7 |

| pair (baseline → candidate) | LPIPS | PSNR dB | sharpness ratio | jitter ratio |
|---|---|---|---|---|
| bf16 control: plug-4step → plug-4step (kernel swap only) | 0.34–0.42 | 14.6–19.9 | 1.01–1.08 | 0.97–1.14 |
| FastH3 4-step VSA → plug-4step | 0.61–0.73 | 9.1–14.3 | 0.79–1.36 | 0.44–0.81 |
| base (49 forwards) → plug-4step (`spark-mountain-lake`) | 0.537 | 15.4 | **1.04** | **0.99** |
| base (49 forwards) → FastH3 4-step VSA (`spark-mountain-lake`) | 0.597 | 13.1 | 1.23 | 1.89 |

Reference-free (median over prompts): plug-4step has sharpness 129 and
motion 6.5; FastH3 4-step VSA has 218 and 13.8. The H3 noise floor is as
high as Phase 3 found. A kernel swap alone moves a plug-4step clip to
LPIPS 0.34–0.42 and moves sharpness up to 8 %, and the fresh-noise
sampler, like FastH3's, ends on equally valid but different samples.
The sheet (`sheets/h3-768p-f062.jpg`) shows clean frames in every arm.
On `ltx-frogyoga`, plug-4step keeps the prompt's frogs doing yoga, while
FastH3 VSA draws a woman doing yoga with frogs around her. Plug-4step is
softer and calmer than FastH3 VSA on two of the three prompts. On the
prompt with a 768p base reference, plug-4step stays inside the policy's
sharpness (0.95–1.08) and jitter (0.85–1.20) ranges against the base. FastH3
VSA does not: it is 23 % sharper and moves 1.9× as much as the base
(`sheets/h3-768p-base-f062.jpg`). That is one prompt, at the H3 noise floor
above. Plug-4step's pod-2 rerun of that prompt reproduced its pod-1 clip
statistics exactly, so the arms are deterministic across processes. It costs
the same as Sol-H3 4-step dense (the same 4 dense forwards plus a merged
LoRA) and **1.68× FastH3 VSA's denoise**, because its attention is dense.

**Wan 2.2 TI2V-5B, 121 frames at 24 fps (5 s), full VAE decode in every arm, 3 prompts.**

| arm | 720p (1280 × 704) denoise / total s | 480p (832 × 480) denoise / total s | load s (720p) |
|---|---|---|---|
| **`wan5b-plug-4step`** (UniPC 4, guidance 1) | **5.97 / 19.7** | **2.08 / 7.7** | 142 (merge 87) |
| same, bf16 control | 6.10 / 19.8 | — | 150 |
| wan-turbo (FastWan2.2-5B, DMD 3 steps) | 4.47 / 18.2 | 1.56 / 7.2 | 91 |
| wan-max (UniPC 50, CFG 5, 100 forwards) | 150.24 / 164.0 | 49.05 / 54.7 | 137 |

| pair | LPIPS | PSNR dB | sharpness ratio | jitter ratio |
|---|---|---|---|---|
| bf16 control: plug → plug (720p) | 0.023–0.062 | 25.2–31.9 | 0.997–1.000 | 1.00 |
| wan-max → plug, 720p | 0.67–0.80 | 6.8–10.5 | 0.45–0.70 | 0.63–6.15 |
| wan-max → wan-turbo, 720p | 0.70–0.76 | 7.8–11.1 | 0.46–1.08 | 0.72–1.98 |
| wan-max → plug, 480p | 0.68–0.71 | 7.2–11.7 | 0.69–0.75 | 0.36–5.76 |
| wan-max → wan-turbo, 480p | 0.63–0.77 | 6.9–13.0 | 0.55–0.89 | 0.57–1.85 |

Reference-free medians at 720p (sharpness / motion): plug 331 / 13.9,
turbo 623 / 5.9, max 1183 / 4.8. At 480p: plug 424 / 9.2, turbo 332 / 6.0,
max 1150 / 10.4. Unlike H3, Wan 5B is not chaotic: the bf16 control stays
at LPIPS ≤ 0.06. So the 0.7 LPIPS between the distilled arms and wan-max
measures genuinely different samples, as expected from different samplers.
Both fast arms (plug and turbo) "fail" the policy's 0.95 sharpness floor
against wan-max on most prompts. The 50-step CFG-5 baseline has far more
high-frequency contrast. Neither fast arm is sharper than the other
across prompts: plug/turbo sharpness is 0.53, 1.01 and 1.27 at 720p and
1.28, 0.66 and 1.54 at 480p. The plug clips move more (the eagle and the
camera on `spark-mountain-lake`), which is where the 6× jitter ratio
against an almost static wan-max clip comes from. In the sheets
(`sheets/w5-720p-f060.jpg`, `w5-480p-f060.jpg`) every arm is coherent and
on-prompt.

**Wan 2.1 T2V-14B, 832 × 480, 81 frames at 16 fps (5 s), full VAE.**

| arm | denoise s | total s | load s | peak GiB |
|---|---|---|---|---|
| **`wan14b-plug-4step`** (step-distill Euler, 4 forwards; 3 prompts) | **19.00** (4.75 s / step) | **22.0** | 313 (merge 175) | 50.9 |
| base 14B (UniPC 50, CFG 5, 100 forwards; 2 prompts, no warm pass) | 477.24 (9.55 s / step) | 481.3 | 170 | 51.6 |

base → plug: LPIPS 0.69–0.70, PSNR 8.6–8.9 dB, sharpness ratio 1.07 and
1.64, jitter 0.80 and 1.38. That is **25.1× less denoise and 21.9× less
end to end**, and the plug clips are sharper than the base's on both
prompts. The base arm also serves as a smoke test of our 14B port on
real weights. It loads and samples coherent clips; no parity reference was
run.

**H3, 49 forwards, 480 × 832, 124 frames, 2 prompts (`h3-demo`, `spark-mountain-lake`), no warm pass.**

| arm | denoise s | total s | load s |
|---|---|---|---|
| base H3 (`base`, dense) | 98.87 | 102.5 | 116 |
| **`h3-plug-cfg`** | **98.50** | **101.8** | 126 |
| base, bf16 control (cuDNN → fwd2) | 100.59 | 103.9 | 119 |

| pair | LPIPS | PSNR dB | sharpness ratio | jitter ratio |
|---|---|---|---|---|
| bf16 control: base → base | 0.26–0.31 | 17.8–18.3 | 1.00–1.01 | 0.95–1.03 |
| base → `h3-plug-cfg` | 0.34–0.42 | 14.3–17.6 | 1.00–1.12 | 0.97–1.03 |

The CFG LoRA moves the clip a little beyond the bf16 floor. Frames stay
clean (`sheets/h3-480p-49fwd-f062.jpg`); `spark-mountain-lake` gets 12 %
sharper. It saves no time against our base, which already runs one
positive pass (above).

### 12.4 Verdicts and catalog proposal

| recipe | modules | speed | quality | verdict |
|---|---|---|---|---|
| `h3-plug-4step` | 312 / 0 skipped | 32.7 s denoise at 768p: 12.2× faster than the 49-forward base (401.6 s), 1.68× slower than FastH3 4-step VSA (19.5 s), the same as Sol-H3 4-step dense | inside the sharpness / jitter ranges against the base on the one prompt with a base reference, where FastH3 VSA is not; follows `ltx-frogyoga` where FastH3 VSA does not; otherwise within H3's chaos floor | **works; keep opt-in.** Candidate for **h3-max** (today Sol-H3 4-step, the same cost), not for h3-turbo |
| `h3-plug-cfg` | 312 / 0 skipped | 98.5 s at 480p, the same 49 forwards as base | small move past the bf16 floor (LPIPS 0.34–0.42 against 0.26–0.31) | **works; no serving value.** The base is already guidance-distilled, so there is nothing to save. Keep opt-in for use with downstream H3 variants |
| `wan5b-plug-4step` | 300 + 300 / 0 skipped | 720p: 6.0 s denoise, 19.7 s total, 25× / 8.3× faster than wan-max; 33 % more denoise than FastWan turbo (4 against 3 steps, total +8 %) | coherent, on-prompt; softer than wan-max and more motion; mixed against turbo | **works; keep opt-in.** It is not better than wan-turbo. Its serving argument is that it needs no second checkpoint: the base weights plus about 1.9 GB of adapters give the fast tier (an unmerged, switchable LoRA would let one resident 5B serve wan-max and a fast tier) |
| `wan14b-plug-4step` | 400 + 400 / 0 skipped | 19.0 s denoise, 22.0 s total at 480p: 25× / 22× faster than base 14B (477 s) | coherent; sharper than base on both prompts | **works; keep opt-in.** We have no 14B tier; the **proposal is a `wan14b-turbo` tier** (or a 14B quality option of wan-turbo at 480p) on this recipe, which brings 14B to the cost of our 5B turbo at 720p |

Catalog proposals (for the owner to decide; nothing was added):
**`wan14b-plug-4step` as a new 14B fast tier**, and **`h3-plug-4step` as an
h3-max alternative to Sol-H3 4-step** (the same cost, closer to the base on the
one base-referenced prompt, but one prompt is not enough). A five-prompt
768p base reference would settle it, at about 7 min per prompt. The H3
adapters carry the MiniMax H3 licence's territory clause (§4.3).

### 12.5 Spend

GPU: `614nezj1d5lnee` 3747 s and `ofxzv506u7rq2j` 3585 s at $2.09/hr, ≈ **$4.26**.
Both were deleted (GET 404) after their outputs were pulled. Build pod: a
share of the shared `fv-build` pod (≈ 40 min of jobs at $0.96/hr). No
weights were written; the adapters were already on both volumes (§8).
Balance $64.08 before the first pod, $55.06 after the second (other
agents ran in parallel).


### 12.6 Five-prompt check against base H3: rule, fixed before any run

Owner decision on 12.4: run the five-prompt 768p check against base H3 and
decide whether `h3-plug-4step` should replace Sol-H3 as `h3-max`.

**Workload.** RTX PRO 6000, 768 × 1344, 124 frames (5 s), the five gate
prompts with the calibrated gate's seeds
(`artifacts/perf/sage-calibrated/driver/prompts-5x3.json`: each prompt at its
own seed, 0 for h3-demo and 42 for the rest, plus +1000 and +2000).

**Arms.** One `fv-gpucheck h3 gen` process per arm, all with
`FASTVIDEO_ATTN_SAGE=0`, `FASTVIDEO_FLASH_KERNEL=cudnn` (picks pinned, so
every arm is deterministic), `--text-encoder resident-fp8 --dit-offload
resident` and their own text cache:

| arm | recipe | forwards | clips |
|---|---|---|---|
| `base` (the reference) | `--h3-recipe base --dense` | 49 | 5: each prompt at its own seed (no warm pass) |
| `plug` | `--h3-recipe h3-plug-4step --dense` | 4 | 15 |
| `max` (`h3-max` as served) | `--techniques h3/sol_h3_4step_engine_ladder --h3-recipe sol-h3` | 4 | 15 |
| `turbo` (`h3-turbo` as served) | `--techniques h3/fasth3_4step_vsa --h3-recipe 4step-vsa` | 4 | 15 |
| `plugfw2` (control) | as `plug`, with `FASTVIDEO_FLASH_KERNEL=v2 FASTVIDEO_CUDNN_SDPA_GRAPH=composite` | 4 | 15 |

The control and a second `plug` run happen on another pod (pod-hour cap);
the two `plug` runs must be byte-identical (frame sha256), so the control
pair `plug`/`plugfw2` measures the one kernel switch.

**Pairs.** `compare-clips` (LPIPS alex, PSNR, sharpness and temporal-jitter
ratios, candidate over base) of every 4-step clip against the base clip of
its prompt. Base exists at the prompt's own seed only, so the +1000 / +2000
clips are compared with that same base clip. That is fair to every arm: at
4 forwards no sampler reproduces the base sample even at the same seed
(12.3: LPIPS 0.54 for plug at the same seed), so all three pairs per prompt
measure distance to the base's look, not sample identity. The same-seed pair
is reported on its own as well. Control: `plug` vs `plugfw2` per clip (15).

**Per prompt and arm:** the median over the 3 seeds of LPIPS, PSNR,
sharpness ratio, jitter ratio and |ln ratio| of the last two.

**Rule.**

1. *Closer:* `plug` is closer to base than `max` on a prompt when its median
   LPIPS is lower. Required on **at least 4 of 5** prompts.
2. *No outlier beyond the control:* on every prompt, `plug`'s median
   |ln sharpness ratio| and median |ln jitter ratio| against base stay within
   max(1.5 × Cmax, Cmax + d), where Cmax is the largest |ln ratio| over the 15
   control clips and d = 0.02 (sharpness), 0.03 (jitter), as in
   docs/perf/sage-attention.md section 8.2.

**Verdict:** `h3-plug-4step` is the better `h3-max` candidate when 1 and 2
both hold. `max` and `turbo` are judged by rule 2 too and reported (not part
of the verdict). Not in the verdict but in the proposal: denoise and total
time per arm (median over its clips), and prompt adherence from frame
sheets (for example, frogs on ltx-frogyoga). Analysis:
`artifacts/perf/plug-five/driver/plugcmp.py`. Defaults are not changed; the
owner decides.

### 12.7 Five-prompt check against base H3: results

Two RTX PRO 6000 Server Edition pods, one after the other (EUR-IS-1,
$2.09/hr, driver 595.91.07, EU volume `jg48s6o1w0`, nothing written to it),
image `ghcr.io/zaitrarrio/fastvideo-rs-serve:sha-ee9de1a` (main HEAD, pinned
by digest). Pod `p63ptu6lo1wbgb` ran the control pair after the LTX gate
(docs/perf/sage-attention.md section 9); pod `ujao4w8j4b4e8e` ran base, plug,
max and turbo and the 45 base pairs. Rows, compare JSONs, frame hashes, the
analysis output and the sheet: `artifacts/perf/plug-five/`.

**Determinism.** No arm logged an `sdpa auto` timing or an `attn_sage` line.
The two `plug` runs, on different pods, are byte-identical on all 16 clips
(warm-up included; `plug/frames-pod{1,2}.sha`), so the control pair measures
the one kernel switch.

**Control (`plug` vs `plugfw2`, 15 clips):** LPIPS 0.32-0.51, PSNR
13.5-21.4 dB, Cmax |ln sharpness| 0.101 and |ln jitter| 0.168, so the rule 2
bound is sharpness 0.859-1.164 and jitter 0.777-1.287.

Per prompt, median over 3 seeds against the base clip (LPIPS at the same
seed in brackets):

| prompt | arm | LPIPS | PSNR dB | sharpness ratio | jitter ratio | rule 2 |
|---|---|---|---|---|---|---|
| h3-demo | **plug** | **0.663** (0.612) | **10.52** | 1.216 | 1.326 | out (sharpness, jitter) |
| | max | 0.722 (0.676) | 9.90 | 1.477 | 2.529 | out |
| | turbo | 0.693 (0.693) | 9.72 | 0.894 | 2.140 | out (jitter) |
| ltx-multishot | **plug** | **0.719** (0.680) | **11.86** | 0.989 | 1.711 | out (jitter) |
| | max | 0.735 (0.678) | 11.30 | 1.483 | 2.514 | out |
| | turbo | 0.722 (0.722) | 9.91 | 1.499 | 2.381 | out |
| ltx-newsbroadcast | **plug** | **0.693** (0.624) | **11.01** | 1.026 | 1.188 | **in** |
| | max | 0.775 (0.761) | 9.40 | 1.914 | 2.089 | out |
| | turbo | 0.747 (0.752) | 9.44 | 1.477 | 2.130 | out |
| ltx-frogyoga | **plug** | **0.695** (0.660) | 10.15 | 1.032 | 1.325 | out (jitter) |
| | max | 0.710 (0.693) | **10.28** | 1.411 | 2.584 | out |
| | turbo | 0.729 (0.729) | 9.14 | 1.296 | 3.621 | out |
| spark-mountain-lake | **plug** | **0.570** (0.555) | **15.18** | 1.082 | 1.386 | out (jitter) |
| | max | 0.666 (0.607) | 13.13 | 1.539 | 2.301 | out |
| | turbo | 0.651 (0.579) | 12.79 | 1.632 | 2.092 | out |

| arm | forwards | denoise s (median) | total s (median) | load s | peak GiB (nvidia-smi) |
|---|---|---|---|---|---|
| base | 49 | 409.6 | 417.2 | 147 | 57.8 |
| **plug** (`h3-plug-4step`) | 4 | **33.27** | **40.59** | 148 | 58.3 |
| max (`h3-max`, Sol-H3 engine ladder) | 4 | 21.90 | 29.04 | 149 | 58.8 |
| turbo (`h3-turbo`, FastH3 VSA) | 4 | 19.89 | 27.32 | 164 | 68.4 |
| plugfw2 (control, other pod) | 4 | 34.24 | 41.42 | | 58.2 |

* **Rule 1: pass, 5 of 5.** plug has the lowest median LPIPS against base on
  every prompt, and per clip it is closer than max on 13 of 15 (PSNR higher on
  13 of 15) and closer than turbo on 12 of 15.
* **Rule 2: fail.** plug's median sharpness and jitter are inside the control
  band only on ltx-newsbroadcast. Its jitter ratio against base is 1.19-1.71
  (sharpness 0.99-1.22): plug moves more than the 49-forward base. max and
  turbo break the band on every prompt and by far more: sharpness 1.41-1.91
  (max) and jitter 2.09-2.58 (max), 2.09-3.62 (turbo). Per clip, plug's
  |ln sharpness| is smaller than max's on 15 of 15 and its |ln jitter| on 14
  of 15.
* **Look (sheet `h3-768p-f062-base-plug-max-turbo.jpg`, columns base, plug,
  max, turbo; rows the five prompts at their own seed, t = 2.6 s).** plug keeps
  base's tone and contrast; max and turbo are visibly more saturated and
  contrasty (the 1.4-1.9 sharpness ratios). On ltx-newsbroadcast base and plug
  are both mid-pan across the field at that moment (the prompt's "camera pans
  right"), max and turbo still hold the reporter close-up. On ltx-frogyoga plug
  is the only arm with a **frog** instructor ("the senior frog instructor sits
  cross-legged at the center"); base draws an old man among frogs, max and
  turbo an elderly woman among frogs. The other three prompts are on-prompt
  in all four arms.
* **Cost.** plug is 33.3 s denoise / 40.6 s total against h3-max's 21.9 /
  29.0 s: **+11.4 s (+52 %) denoise, +40 % per clip**, because it runs all 200
  attention calls dense and without MXFP8 (h3-max routes 144 of 200 through
  Sol and uses the MXFP8 linears). It is 12.3x faster than base.

**Verdict (rule as written): FAIL.** Rule 1 passes 5/5, rule 2 fails on 4 of
5 prompts (plug moves more than base, beyond the rounding-level band).

**Proposal (owner decides; no default changed).** The rule-2 bound is
rounding noise, and none of the three 4-step arms gets inside it against a
49-forward base; plug is the closest of the three on every prompt and on
almost every clip, by LPIPS, PSNR, sharpness and jitter, and it is the
only arm that follows the frog-yoga prompt. So the choice is cost, not
quality: (a) keep Sol-H3 (engine ladder) as `h3-max` and offer
`h3-plug-4step` as an explicit "closest to base" recipe; or (b) make
`h3-plug-4step` `h3-max` and accept +11 s per 5 s clip; or (c) first run plug
with the h3-max techniques (MXFP8 linears, the Sol engine ladder on forwards
1-3), which would bring it to about h3-max's cost, and re-check it against
this base set (the 5 base clips' statistics are kept, but their frames
are not; a rerun of base costs ~37 min, ~$1.3). I would recommend (c), then
(b) if it holds. A looser rule-2 bound (for example h3-max's own spread) is
not proposed after the fact.

**Spend.** Pod 1 4 344 s (LTX gate, serve check, control pair) and pod 2
5 346 s at $2.09/hr: ≈ $2.52 + $3.10 = **≈ $5.62** for both sections. Both
pods were deleted after their outputs were pulled (GET 404). Each had a 5 400 s
DELETE backstop, the on-pod 20 min idle guard and a local balance watchdog
(delete below $12). Balance $46.61 before the first pod, $26.79 after the
second (other agents ran in parallel).
