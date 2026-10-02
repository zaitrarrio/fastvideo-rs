# LongLive 1.0 / 2.0 / Plug: method notes, weights, and a port plan

Date: 2026-10-01. Status: research (Phase A). LongLive-1.3B inference is
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
| Prompt switching on our surfaces | Reactor causal `set_prompt` → `CausalControl` → `CausalDriver::block` → `CausalRollout::set_prompt` | wired: the recipe opts in. **The director refuses causal models** (`director/service.rs::caps_for` needs `StreamCaps::Clip`), so director `prompt` updates cannot reach it without a causal director engine (Phase A follow-up, §6) |
| TAEHV per block | `taew2_1` | none |
| **2.0**: Wan2.2-TI2V-5B DiT, VAE, `taew2_2` | the 5B port (`vae22.rs`, `wan_2_2_ti2v_5b`), FastWan2.2 | causal 5B preset (8-frame chunks, window 32, sink 8); `forward_kv` is architecture-generic (**INFERRED**: not run on 5B) |
| 2.0 multi-shot sink + RoPE shot offset | single sink | a pinned-segment cache (global sink + shot sinks + rolling), a per-shot temporal RoPE offset |
| 2.0 NVFP4 W4A4 | `nvfp4.rs` (FourOverSix/LongLive MSE rule, dequant-beforehand), `nvfp4_gemm.rs` / `nvfp4_linear.rs` (LTX FFN on FP4 tensor cores, sm_120) | Wan linears on the FP4 GEMM; read `model_4o6.pt` codes or quantise BF16; activation quantisation |
| 2.0 NVFP4 KV cache | bf16 KV | FP4 KV store + dequant (or FP4-aware attention) kernel |
| 2.0 async decode | TAEHV decode inline per block | decode on a second stream (or the pacer thread), overlapping the next chunk |
| **Plug** Wan LoRA merge 1.0 + 0.5 | Wan 5B, Wan2.1 14B ports; a merge helper (this branch, f32 host) | key maps for PEFT names (done) and lightx2v names (todo), multi-LoRA weights, 4-step schedules for non-causal Wan |
| Plug H3 LoRAs | H3 port, FastH3 4-step, an H3 LoRA loader | separate use only; the four-forward runner contract; H3 licence |

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
- The director has no causal mode (§5, §6 A4).

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
