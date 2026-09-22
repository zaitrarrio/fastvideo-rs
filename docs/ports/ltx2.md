# LTX-2 (19B, distilled) on the cudarc backend — port specification

Target: text → synchronized video + audio with `Lightricks/LTX-2`'s **distilled**
checkpoint on one RTX PRO 6000 (96 GB, sm_120), every stage judged against
diffusers by `scripts/gpu/ltx2_oracle.py`. Config structs and the sigma schedule
live in `crates/fastvideo-models/src/ltx2/{config,schedule}.rs`.

For **LTX-2.5** (Gemma 4, gated DiT, conv VAE, BWE vocoder, ancestral stage-1)
see [ltx25.md](ltx25.md).

Source abbreviations (line numbers are for the copies read on 2026-09-19;
diffusers `main`, transformers `v4.57.3`, Lightricks/LTX-2 `main` unless a commit
is given):

| tag | file |
|---|---|
| `T` | diffusers `models/transformers/transformer_ltx2.py` |
| `P` | diffusers `pipelines/ltx2/pipeline_ltx2.py` |
| `C` | diffusers `pipelines/ltx2/connectors.py` |
| `V` | diffusers `models/autoencoders/autoencoder_kl_ltx2.py` |
| `A` | diffusers `models/autoencoders/autoencoder_kl_ltx2_audio.py` |
| `VO` | diffusers `pipelines/ltx2/vocoder.py` |
| `S` | diffusers `schedulers/scheduling_flow_match_euler_discrete.py` |
| `U` | diffusers `pipelines/ltx2/utils.py` |
| `G` | transformers `models/gemma3/modeling_gemma3.py` |
| `FV` | FastVideo `fastvideo/models/dits/ltx2.py` |
| `LT:` | `github.com/Lightricks/LTX-2/packages/…` |

The diffusers files are shared with LTX-2.3 and 2.5, so they carry many branches
the 2.0 checkpoint never takes. **Everything gated on these config keys is off
for LTX-2.0** (absent from `transformer/config.json`, so class defaults apply,
`T:1152-1186`): `gated_attn`, `cross_attn_mod`, `audio_gated_attn`,
`audio_cross_attn_mod` (all false → 6-row AdaLN tables, no `prompt_adaln`, no
`to_gate_logits`), `perturbed_attn` false, `use_prompt_embeddings` true (the
caption projections live in the DiT), `use_keyframes_abs_pos_embedding` false.
In the connectors: `per_modality_projections` false, `*_gated_attn` false. The
vocoder is the plain `LTX2Vocoder` (no BWE, no Snake). There is no
`duration_head`, no `prompt_enhancer`, no diffusion decoder in this repo.

---

## a. Inference contract

### Which weights are "distilled"

`Lightricks/LTX-2` ships two layouts:

* a diffusers layout (`transformer/`, `connectors/`, `vae/`, `audio_vae/`,
  `vocoder/`, `text_encoder/`, `tokenizer/`, `scheduler/`, `latent_upsampler/`).
  **`transformer/` and `connectors/` are the dev model.** Evidence: the model
  card runs them with 40 steps and CFG 4 and applies the distilled LoRA for
  stage 2; the diffusers docs take the distilled checkpoint from a different
  repo, `rootonchair/LTX-2-19b-distilled`; and the LFS hashes differ —
  transformer shard 1 `c4cebec5…` (dev) vs `8ffa46d6…` (distilled), connectors
  `c7c0ad36…` vs `60e44935…`. `vae/`, `audio_vae/`, `vocoder/`,
  `latent_upsampler/`, `tokenizer/` and all 11 `text_encoder/model-*` shards are
  byte-identical between the two repos.
* single files in the original key naming: `ltx-2-19b-dev.safetensors`,
  `ltx-2-19b-distilled.safetensors` (43.29 GB each: DiT + connectors + video VAE
  + audio VAE + vocoder, everything except Gemma), `-fp8`, `-fp4`, and
  `ltx-2-19b-distilled-lora-384.safetensors` (7.67 GB, rank 384).

The distilled DiT is a **separate full set of weights**, not "dev + LoRA" at
load time. The LoRA exists so the *dev* model can run the distilled stage 2 of
the two-stage recipe; applying it (`W + B·A`, strength 1.0) to the dev weights is
the third way to get a distilled model. Note the LoRA also touches
`text_embedding_projection.aggregate_embed` — the connectors' `text_proj_in` —
which is why the distilled checkpoint has its own connectors. **Using the dev
`connectors/` with the distilled transformer is wrong.**

Recommendation (§g): load `ltx-2-19b-distilled.safetensors` directly with a key
rename table; use `rootonchair/LTX-2-19b-distilled` for the oracle.

### How the distilled model is run

* 8 model evaluations on a fixed sigma list, `U:27` = `LT:ltx-pipelines/…/utils/constants.py:17`:
  `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]` then 0.
* **No guidance of any kind**: CFG = 1 (model card: "8 steps, CFG=1"); the
  Lightricks `DistilledPipeline` wraps the model in `SimpleDenoiser`
  (`LT:ltx-pipelines/…/distilled.py:266`) — one forward per step, no negative
  prompt, no STG, no modality guidance, no rescale. **diffusers caveat:** the
  `LTX2Pipeline.__call__` defaults in current `main` are LTX-2.5's
  (`guidance_scale=3, stg_scale=1, modality_scale=3, guidance_rescale=0.7,
  audio_guidance_scale=7`, `P:941-949`). Passing only `guidance_scale=1.0` as the
  docs example does leaves STG and modality guidance on (3 forwards/step). A
  faithful distilled run passes `guidance_scale=1, audio_guidance_scale=1,
  stg_scale=0, audio_stg_scale=0, modality_scale=1, audio_modality_scale=1,
  guidance_rescale=0, audio_guidance_rescale=0` (`P:899-908`). The oracle drives
  the transformer directly, so it is not exposed to this.
* Scheduler: `FlowMatchEulerDiscreteScheduler` with `use_dynamic_shifting=false`,
  `shift_terminal=null`, `shift=1` so the list passes through unchanged (§f). The
  `scheduler/` in `Lightricks/LTX-2` is the **dev** config and would shift it.
* Deterministic Euler. (Lightricks switches stage 1 to an ancestral sampler only
  for `model_version ≥ 2.5`, `LT:…/distilled.py:62,76-84`.)
* Dtype: bf16 everywhere in both references (`distilled.py:109`); latents and the
  Euler update are float32 in diffusers (`P:1282-1293`, `P:1422-1423`, `S:517`).

### Two stages, and what we need first

The production recipe (`LT:…/distilled.py:249-317`, diffusers docs "Distilled
checkpoint generation"):

1. **Stage 1**: 8 steps at *half* the target resolution (default 768×512).
2. `LTX2LatentUpsamplerModel` (`latent_upsampler/`, 0.996 GB): ×2 spatial on the
   **de-normalised** video latent (Conv3d + GroupNorm(32) + SiLU res-blocks,
   Conv2d→PixelShuffle(2)). Audio latents pass through. A ×2 temporal upscaler
   also exists as a single file; no default pipeline uses it.
3. **Stage 2**: re-noise both latents to `σ = 0.909375`
   (`x ← σ·ε + (1-σ)·x`, `P:716-722`) and run 3 steps
   `[0.909375, 0.725, 0.421875]` (`U:36`) at full resolution (1536×1024 →
   24 576 video tokens), same distilled weights, no guidance.

**First target: stage 1 only** — 8 steps at 768×512×121, decode, mux. It is a
complete, officially supported T2AV result (the diffusers single-stage call with
`output_type="np"`), it exercises every component, and it needs neither
GroupNorm nor the 24 576-token attention. Stage 2 is a follow-up: one new model
(the upsampler) and no new DiT code.

### Shapes for the default request

Defaults (`P:932-937`, `LT:…/constants.py:41-46`): 768×512 (W×H), 121 frames,
24 fps, seed 10 (Lightricks). Constraints (model card): H, W divisible by 32
(64 for two-stage), `num_frames = 8k + 1`.

| quantity | formula | default |
|---|---|---|
| latent frames | `(F-1)//8 + 1` (`P:1261`) | 16 |
| latent H, W | `H//32`, `W//32` | 16, 24 |
| video latent | `[1, 128, 16, 16, 24]` | |
| **video tokens** | `F·H·W` (patch 1×1×1) | **6 144** (stage 2: 16·32·48 = **24 576**) |
| audio latents/s | `16000 / 160 / 4` (`P:1296-1298`) | 25 |
| **audio tokens** | `round(F/fps · 25)`, half-to-even (`P:1299`) | **126** |
| audio latent | `[1, 8, 126, 16]` → packed `[1, 126, 128]` | |
| text tokens | fixed 1024, both streams | 1 024 |
| mel out | `[1, 2, 4L-3, 64]` | `[1, 2, 501, 64]` |
| waveform | `240 · (4L-3)` samples, **24 kHz** stereo | 120 240 (5.01 s) |
| video out | `[1, 3, 8(F_lat-1)+1, 512, 768]` | 121 frames (5.04 s) |

The audio VAE's mel domain is 16 kHz / hop 160 (10 ms per mel frame); the
vocoder emits 24 kHz (`vocoder/config.json: output_sampling_rate`), 240 samples
per mel frame, still 10 ms. Mux at 24 000 Hz. The audio is ~30 ms shorter than
the video; diffusers muxes as is.

Packing (`P:648-668`, `P:724-743`): video `[B,C,F,H,W] → [B, F·H·W, C]`, token
order frame-major, then row, then column; audio `[B,8,L,16] → [B, L, 128]` with
feature index `channel·16 + mel_bin`.

Latent normalisation: the DiT works in normalised space. Video:
`z = ẑ·std/scaling_factor + mean` per channel (128 values, `scaling_factor=1`)
before the VAE (`P:694-701`, `P:1638`). Audio: `z = ẑ·std + mean` applied to the
**packed** `[B,L,128]` tensor — the statistics are per (channel, mel-bin) pair —
and only then unpacked (`P:1607-1610`, `A:745-748`).

Other constants: `causal_offset = 1` and `vae_scale_factors = [8,32,32]` enter
only through RoPE positions (§e); `timestep_scale_multiplier = 1000` means the
model input is `1000·σ` — diffusers passes `scheduler.timesteps` which already is
that (`T:1406-1408`), FastVideo/Lightricks pass σ and multiply inside
(`FV:1006`). `pos_embed_max_pos = 20` is the RoPE time base in **seconds**, which
is also the pipeline's `max_seconds` (`P:936`): ≤ 20 s clips.

Seeds: diffusers draws the video latent then the audio latent from one
generator, float32 (`P:804`, `P:844`). CUDA RNG streams are not reproducible
from Rust, so our contract is CPU-seeded float32 noise, and the oracle saves the
tensors it used.

---

## b. Text path

### Tokenisation

`GemmaTokenizerFast` (`tokenizer/tokenizer.json`, 33 MB; vocab 262 208).
`P:327-341`: `prompt.strip()`, **no chat template, no system prompt**,
`add_special_tokens=True` (prepends `<bos>` = 2, no `<eos>`), truncate and pad to
`max_length = 1024` on the **left** with `<pad>` = 0. Lightricks does the same
with `TOKENIZER_MAX_LENGTH = 1024` (`LT:ltx-core/…/gemma/gemma_assets.py:162`,
`tokenizer.py:32-57`). Result: `ids[1024]`, `mask[1024]` with the `n` real tokens
last.

### Gemma-3-12B, text only

`Gemma3ForConditionalGeneration` called with `input_ids`, `attention_mask`,
`output_hidden_states=True` (`P:347-349`). No image tokens, so the SigLIP vision
tower and `multi_modal_projector` (1.69 GB of the checkpoint) are never touched:
**skip every key not under `language_model.`**; `lm_head` is tied and absent.

`text_config` (fetched): hidden 3840, 48 layers, **16 query heads, 8 KV heads,
head_dim 256** (so q is 4096 wide, k/v 2048 — wider than the residual stream),
MLP 15 360, `query_pre_attn_scalar = 256` → softmax scale `256^-0.5 = 1/16`,
`rms_norm_eps = 1e-6`, no attention bias, no logit soft-capping, vocab 262 208.

Per layer (`G:360-400`):

```
h = x + post_attention_layernorm(attn(input_layernorm(x)))
y = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
mlp(u) = down_proj(gelu_tanh(gate_proj(u)) * up_proj(u))          # G:121-123
```

* **RMSNorm is zero-centred**: `x̂ = x·rsqrt(mean(x²) + eps)·(1 + w)`, computed in
  float32 then cast back (`G:132-140`). Load `w + 1` and the existing RMSNorm
  applies. Four per layer plus the final `norm`.
* **Attention** (`G:268-340`): q/k/v projections → reshape to heads →
  **per-head** `q_norm`/`k_norm` (the same zero-centred RMSNorm over
  `head_dim = 256`, weight `[256]` shared by all heads) → rotate_half RoPE →
  GQA (each KV head serves 2 query heads, `repeat_interleave` order) → causal
  SDPA with scale 1/16 → `o_proj` (4096 → 3840).
* **Layer types**: `layer_types[i]` is `full_attention` when `(i+1) % 6 == 0`
  (layers 5, 11, …, 47), else `sliding_attention` with window 1024.
  The sliding overlay is `kv > q - 1024` (`masking_utils.py:87-88`), which for a
  1024-token sequence admits every causal pair — **at our length the sliding
  layers are plain causal attention**; only the RoPE differs.
* **RoPE** (`G:146-178`, `G:478-480`, `G:559-560`): standard
  `inv_freq_j = θ^(-2j/256)`, tables `cos/sin(cat(f, f))` of width 256,
  `q' = q·cos + rotate_half(q)·sin` with `rotate_half(x) = cat(-x[128:], x[:128])`.
  Sliding layers: θ = 10 000, unscaled. Global layers: θ = 1 000 000 with
  `rope_scaling {linear, factor 8}` → `inv_freq / 8` (positions ÷ 8).
* **Positions**: the pipeline passes no `position_ids`, so transformers uses
  `arange(1024)` over the *padded* sequence (`G:521-530`); real tokens sit at
  positions `1024-n … 1023`. RoPE is relative, so running only the `n` real
  tokens at `0 … n-1` is mathematically identical, but bf16 rounding differs —
  use the offset positions to match the oracle.
* **Mask**: causal ∧ key-is-real. Pad *queries* produce garbage that is
  discarded downstream (next section), so the port can simply drop the pad rows:
  run `S = n` tokens, causal, no padding mask.
* **Embedding**: `embed_tokens[id] · sqrt(3840)`, but the scale is cast to the
  weight dtype first (`G:107`): in bf16 the multiplier is exactly **62.0**, not
  61.9677. Lightricks does the same (`encoder_configurator.py:394-395`).
* **Output**: `hidden_states` is 49 tensors — the scaled embeddings, the raw
  outputs of layers 0…46, and the output of layer 47 **after the final norm**
  (`G:563-591`). The pipeline stacks them on a new last axis and flattens:
  `[B, 1024, 3840, 49] → [B, 1024, 188 160]`, feature index
  `channel·49 + state` (`P:350-352`).

The shards are **float32 on disk** (48.75 GB total, 47.06 GB language model);
both references cast to bf16 at load.

### Cross-check of `fastvideo_cudarc::llm` (`DecoderConfig::gemma3_12b_text()`)

Checked field by field against `text_encoder/config.json → text_config` and the
headers of `text_encoder/model-0000{1..11}-of-00011.safetensors`:

| `DecoderConfig` | preset | checkpoint | |
|---|---|---|---|
| `hidden` | 3840 | `hidden_size 3840`; `embed_tokens.weight [262208, 3840]` | ✓ |
| layers | 48 | `num_hidden_layers 48`; `layers.0 … layers.47` | ✓ |
| `heads` / `kv_heads` / `head_dim` | 16 / 8 / 256 | `q_proj [4096,3840]`, `k_proj`,`v_proj [2048,3840]`, `o_proj [3840,4096]`, `q_norm`,`k_norm [256]` | ✓ |
| `intermediate` | 15360 | `gate_proj`,`up_proj [15360,3840]`, `down_proj [3840,15360]` | ✓ |
| `rms_eps`, `norm_offset` | 1e-6, 1.0 | `rms_norm_eps 1e-6`; `(1 + w)` `G:139` | ✓ |
| `sandwich_norms` | true | `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`, all `[3840]` | ✓ |
| `qk_norm` | per head, before RoPE | `G:312-316` | ✓ |
| `act` | `GeluTanh` | `hidden_activation gelu_pytorch_tanh`; `down(act(gate)·up)` | ✓ |
| `attn_scale` | `256^-0.5` | `query_pre_attn_scalar 256` | ✓ |
| global layers | `(i+1) % 6 == 0`, θ 1e6, factor 8, no window | `layer_types`: `full_attention` at 5, 11, …, 47; `rope_theta 1e6`, `rope_scaling {linear, 8}` | ✓ |
| local layers | θ 1e4, factor 1, window 1024, `q - k < 1024` | `rope_local_base_freq 1e4`, `sliding_window 1024`, overlay `kv > q - 1024` | ✓ |
| biases | none | no `.bias` key anywhere in the language model | ✓ |
| `layer_prefix` / `embed_key` / `final_norm_key` | `language_model.model.layers` / `…embed_tokens.weight` / `…norm.weight` | exactly these in the 11-shard set | ✓ |
| tap numbering | tap `k` = state before layer `k`; tap 48 post-norm | `G:563-591` | ✓ |
| **`embed_scale`** | `sqrt(3840)` = 61.967735 | the bf16 reference multiplies by **62.0** (`G:107` casts the scale to the weight dtype) | **✗ for bf16 parity** |
| on-disk dtype | loader narrows F32 → bf16 | every Gemma tensor is **F32** | ✓ (must narrow, 48.7 GB) |

Two things to change or know:

1. **`embed_scale` should be 62.0** to reproduce the product (and the oracle,
   which runs bf16). 61.9677 is a uniform 5.2e-4 scale error on tap 0. That is
   below the bf16 rounding floor of any single tap (≈ 2e-3), so the llm gate
   will not catch it — but it is a *bias*, not noise, and the residual stream is
   not scale-invariant (the sandwich post-norms rescale each branch before the
   add, so the embedding's weight relative to every branch shifts by that
   factor). Make it a config choice: 62.0 for bf16 parity, `sqrt(3840)` against
   a float32 reference.
2. **Positions.** `llm::hidden_states`' doc comment says a left-padded prompt
   "starts its real tokens at 0". transformers does not: with `position_ids`
   omitted it uses `arange(S)` over the padded sequence, so the real tokens sit at
   `S-n … S-1`. `ltx2_oracle.py --llm-out` writes `positions = arange(1024)`,
   which is what the reference used. (RoPE being relative, either choice is the
   same function; only bf16 rounding differs.)

**Which shard set.** `text_encoder/` holds two: `model-0000{1..11}-of-00011` +
`model.safetensors.index.json` (48.75 GB, keys `language_model.model.*`,
`vision_tower.*`, `multi_modal_projector.*`) — **load this one** — and a stale
`diffusion_pytorch_model-0000{1..12}-of-00012` (51.6 GB, the same Gemma under
`base_text_encoder.language_model.model.*` plus an old copy of the connectors).
`LazyStore::open` maps *every* `.safetensors` under a directory; the prefixes do
not collide so it would not error, but do not download the second set
(`hf download Lightricks/LTX-2 --include "text_encoder/model-*" "text_encoder/*.json"`).

**Which hidden states feed the connectors.** All of them: `P:350-352` does
`torch.stack(text_encoder_outputs.hidden_states, dim=-1)` over the whole
49-tuple and flattens to `[B, 1024, 188160]`; `C:425-427` unflattens to
`[B, 1024, 3840, 49]`; `C:450-461` normalises each of the 49 states separately
and one `Linear(188160, 3840)` mixes them. There is no layer selection —
`text_proj_in_factor = 49` is `num_hidden_layers + 1`. So the llm oracle file
holds taps 0…48 (753 MB as float32).

### Connectors (`LTX2TextConnectors`, `C:335-478`)

1. **Per-state masked normalisation** (`C:13-77`, = `LT:…/feature_extractor.py:12-45`).
   For each of the 49 states `l` independently, over the `n` real tokens and all
   3840 channels: `mean_l = Σx / (n·3840 + 1e-6)`, `min_l`, `max_l`;
   `x̂ = 8 · (x - mean_l) / (max_l - min_l + 1e-6)`; pad rows set to 0. The
   reference does this in **bf16** (the dtype of `prompt_embeds`) — see §j.
2. **`text_proj_in`**: `Linear(188 160 → 3840, bias=False)`, one projection shared
   by both modalities (`C:372`, `C:459-461`).
3. Two independent **1-D connector transformers** (`video_connector`,
   `audio_connector`; identical shape, separate weights), `C:218-332`, each
   30 heads × 128 = **3840 wide**, 2 layers:
   * **Registers** (`C:289-318`): a learned `[128, 3840]` table tiled 8× to
     `[1024, 3840]`. The `n` real tokens are moved to the **front** in order
     (positions `0…n-1`) and every position `p ≥ n` is filled with
     `registers[p mod 128]`. The attention mask is then replaced by zeros: **full
     unmasked attention over all 1024 positions**.
   * **RoPE**: 1-D "split" RoPE (§e math) with `dim = 3840`, 1920 frequencies, no
     padding, 64 per head; position `p` has fraction `p / 4096`
     (`connector_rope_base_seq_len`), `C:111-171`.
   * Block (`C:201-215`): `x += attn1(rms(x))`; `x += ff(rms(x))`. `rms` is
     `RMSNorm(eps=1e-6)` **without** weight. `attn1` is `LTX2Attention` (§e):
     q/k/v/out Linears 3840² with bias, RMSNorm **with** weight over the full
     3840-wide q and k, RoPE on q and k, SDPA scale `128^-0.5`. `ff` is
     `Linear(3840,15360)` → tanh-GELU → `Linear(15360,3840)`.
   * Final weightless `RMSNorm` (`C:274`, `C:330`).
4. Returns `video [B,1024,3840]`, `audio [B,1024,3840]`, and a mask that is **all
   ones** (`C:472-478`). Consequently **no attention in the DiT is masked**.

The projection from 3840 to the streams' widths — what `cross_attention_dim 4096`
and `audio_cross_attention_dim 2048` refer to — is inside the DiT:
`caption_projection` / `audio_caption_projection` = `Linear(3840, D)` →
tanh-GELU → `Linear(D, D)` with D = 4096 / 2048 (`T:1207-1210`, `T:1594-1600`).
It is timestep-independent: compute once per prompt, not per step.

**Reference divergence.** Lightricks `main` rewrote the register fill to keep
real tokens where they are (left-padded → at the *end*) and put registers in the
pad slots (`embeddings_connector.py:139-152`). The release-day code (commit
`9ce438b353`, 2026-01-05, same file `:131-157`) front-aligns exactly as diffusers
does, and diffusers says so in a comment (`C:305-308`). The two are not
equivalent (different register indices next to the text, different absolute
positions). The 2.0 weights were released with the front-aligned code; **follow
diffusers**. Flagged in §k.

---

## c. Video VAE decoder (`AutoencoderKLLTX2Video`)

Decoder only, 1.057 GB bf16, 58 tensors: every learned layer is a 3×3×3 conv.
No attention, no learned norm, no shortcut convs (`norm3`/`conv_shortcut` exist
in code, `V:170-175`, but every resnet here has `in == out`).

**What the pipeline passes** (`P:1621-1643`): `timestep_conditioning` is false
→ `temb = None`, no noise is mixed into the latents, `decode_timestep` and
`decode_noise_scale` are ignored; `decoder_inject_noise` is all false → the
`per_channel_scale*` noise path is absent. `causal=None` → `decoder_causal =
false` (`V:979`). Output is nominally `[-1, 1]`; post-process
`clamp(x/2 + 0.5, 0, 1)`.

Primitives:

* **Conv** `LTX2VideoCausalConv3d` (`V:63-111`): kernel 3×3×3, stride 1.
  Time: **non-causal replicate** — one copy of the first frame in front and one
  of the last frame behind (`V:104-106`). Space: `padding=(0,1,1)` with
  `padding_mode="reflect"` (`decoder_spatial_padding_mode`): reflect without
  repeating the edge (`x[-1] = x[1]`, `x[N] = x[N-2]`). Bias present.
* **Norm** `PerChannelRMSNorm` (`V:30-60`): `x / sqrt(mean_c(x²) + 1e-8)` over
  the channel axis at every (t, h, w); no weight. (`resnet_norm_eps = 1e-6` only
  feeds the unused `norm3`.) This is the existing `rms_norm_channels` kernel with
  γ = 1, eps 1e-8, and its fused SiLU.
* **Resnet** (`V:187-237`): `x + conv2(silu(norm(conv1(silu(norm(x))))))`.
* **Upsampler** `LTX2VideoUpsampler3d` (`V:288-336`), stride (2,2,2),
  `upscale_factor = 2`, residual on:
  `y = conv(x)` to `4·C_in` channels; depth-to-space
  `out[c, 2f+i, 2h+j, 2w+k] = y[((c·2+i)·2+j)·2+k, f, h, w]` → `C_in/2` channels;
  **drop the first frame** (`V:331-332`). Residual: the same depth-to-space on
  `x` itself (→ `C_in/8` channels), channel-tiled ×4 (`repeat` on the channel
  axis: output channel `c` reads residual channel `c mod C_in/8`), first frame
  dropped, added.
* **Unpatchify** (`V:1016-1021`), patch 4: note the axis pairing differs from
  the upsampler — `out[c, f, 4h+b, 4w+a] = y[c·16 + a·4 + b, f, h, w]`.

Layer by layer for the default latent `[1,128,16,16,24]` (f32 activation sizes):

| stage | op | output | f32 |
|---|---|---|---|
| `conv_in` | conv 128→1024 | `[1024,16,16,24]` | 24 MB |
| `mid_block` | 5 resnets @1024 | same | |
| `up_blocks.0.upsamplers.0` | conv 1024→4096, d2s, drop 1 | `[512,31,32,48]` | conv out 96 MB |
| `up_blocks.0.resnets` | 5 resnets @512 | same | |
| `up_blocks.1.upsamplers.0` | conv 512→2048, d2s, drop 1 | `[256,61,64,96]` | conv out 372 MB |
| `up_blocks.1.resnets` | 5 resnets @256 | same | |
| `up_blocks.2.upsamplers.0` | conv 256→1024, d2s, drop 1 | `[128,121,128,192]` | conv out 1.43 GB |
| `up_blocks.2.resnets` | 5 resnets @128 | same | 1.42 GB each |
| `norm_out`, SiLU, `conv_out` | conv 128→48 | `[48,121,128,192]` | 545 MB |
| unpatchify | | `[3,121,512,768]` | 545 MB |

Frames: each stage maps `F → 2F - 1`, so `F → 8(F-1)+1` (16 → 121). In an
up-block the upsampler runs **before** the resnets (`V:652-684`).

Chunking (`V:1176-1190`, `V:1405-1533`): tiling is opt-in. Spatial tiles of
512 px with stride 448 (latent 16/14) and linear blends; temporal tiles of 16
frames stride 8 (latent 2/1). Because the decoder is non-causal with replicate
padding and frame drops, tiled output is **not** equal to the untiled output —
tiling is an approximation, and parity must be measured untiled. At 768×512×121
the untiled peak is ~5 GB of f32 activations plus the conv workspace, so the
first target does not tile. At 1536×1024 (×4) temporal chunking is required
(§h).

**What the port does instead (exact).** Input-side chunking cannot be exact at
our lengths either: 41 convs deep, the temporal receptive field is ≈ 22 latent
frames, more than a 16-frame clip has. But apart from the convs' temporal kernel
every op in the decoder is per-frame, and a radius-1 temporal conv needs only a
two-frame memory. `ltx2::vae` therefore streams each *convolution*: it keeps the
last two input frames, emits output `t` as soon as input `t+1` exists, and
replicate-pads only at the true ends of the clip; skip paths (resnet identity,
the upsampler's tiled residual) are FIFO-buffered until the delayed main path
catches up, and the upsampler drops the first frame of the *stream*, not of a
chunk. This is the same sum of the same products as the one-shot decode — the
host test decodes with chunk sizes 1, 2, 3, 5 and compares to the whole-clip
result. The low-resolution stages run whole; the last up-block and the head run
in chunks of `FASTVIDEO_LTX2_VAE_CHUNK` frames (default 8 → 16 output frames)
and hand finished frames to the sink. On the device a different chunk shape can
make cuDNN pick a different algorithm, so GPU results agree to rounding, not to
the bit (`ltx2 vae` measures it).

---

## d. Audio VAE decoder + vocoder

### `AutoencoderKLLTX2Audio` decoder (`A:469-665`), 63.8 MB

A 2-D conv net over `[B, C, time, mel]`; `causality_axis = "height"` makes the
**time axis causal**. `norm_type = "pixel"`: `x / sqrt(mean_c(x²) + 1e-6)`, no
weight (`A:82-95`). No attention (`attn_resolutions` null,
`mid_block_add_attention` false). No timestep input (`temb_ch = 0`).

* **Causal conv** (`A:31-79`): 3×3 conv, stride 1, zero padding `F.pad(x,
  (1, 1, 2, 0))` — mel axis 1/1, time axis 2 before and 0 after. 1×1 convs pad
  nothing.
* **Resnet** (`A:203-219`): `x' + conv2(silu(norm(conv1(silu(norm(x))))))`,
  `x' = nin_shortcut(x)` (1×1 conv) when channels change.
* **Upsample** (`A:269-284`): nearest ×2 on **both** axes, causal conv, then drop
  the first time row: `T → 2T - 1`, `M → 2M`.

| stage | op | output for `[8, L, 16]` |
|---|---|---|
| `conv_in` | 8→512 | `[512, L, 16]` |
| `mid.block_1`, `mid.block_2` | resnets @512 | |
| `up.2` | 3 resnets @512, upsample | `[512, 2L-1, 32]` |
| `up.1` | resnets 512→256 (nin), 256, 256, upsample | `[256, 4L-3, 64]` |
| `up.0` | resnets 256→128 (nin), 128, 128 | `[128, 4L-3, 64]` |
| `norm_out`, SiLU, `conv_out` | 128→2 | `[2, 4L-3, 64]` |

`up` is indexed by level and executed from 2 down to 0 (`A:588`, `A:628`). The
output already has the target length `max(4L-3, 1)` and 64 bins, so the
crop/pad at `A:641-659` is a no-op. The output is a stereo log-mel spectrogram
(16 kHz, hop 160), no activation.

### `LTX2Vocoder` (`VO:279-418`), 111 MB — HiFi-GAN generator

Input mel `[B, 2, T, 64]` → `transpose(2,3)` → `flatten(1,2)` → `[B, 128, T]`,
channel index `stereo·64 + mel_bin` (`VO:391-394`).

```
x = conv_in(x)                                   # Conv1d 128→1024, k=7, pad 3
for i in 0..5:
    x = leaky_relu(x, 0.1)
    x = upsamplers[i](x)                         # ConvTranspose1d
    x = mean(resnets[3i+j](x) for j in 0..3)     # three parallel ResBlocks
x = leaky_relu(x, 0.01)                          # bare nn.LeakyReLU(): default slope, VO:367-369
x = tanh(conv_out(x))                            # Conv1d 32→2, k=7, pad 3
```

| i | ConvTranspose1d (in→out, k, stride, pad) | length |
|---|---|---|
| 0 | 1024→512, 16, 6, 5 | 6T |
| 1 | 512→256, 15, 5, 5 | 30T |
| 2 | 256→128, 8, 2, 3 | 60T |
| 3 | 128→64, 4, 2, 1 | 120T |
| 4 | 64→32, 4, 2, 1 | **240T** |

`pad = (k - stride)//2`, no output padding, so `L_out = (L-1)·s - 2p + k = L·s`
exactly. `ResBlock` (`VO:214-276`), kernels `{3, 7, 11}` per upsample stage, each
with dilations `(1, 3, 5)`:

```
for d in (1, 3, 5):
    x = x + conv2_d(leaky_relu(conv1_d(leaky_relu(x, 0.1)), 0.1))
```

`conv1_d`: `Conv1d(C, C, k, dilation=d, padding="same")` (= `d·(k-1)/2` each
side); `conv2_d`: same with dilation 1. Weights are plain (weight-norm already
folded). The config has no `act_fn` key → `"leaky_relu"`, `antialias` false:
**no Snake, no anti-aliased activations** in LTX-2.0. (A typo at `VO:263` assigns
the second activation to the wrong variable; the effect is that `acts2` reuses
the last `acts1` LeakyReLU(0.1) — same function, no behavioural difference.)

Output `[B, 2, 240·T]` in `[-1, 1]` at **24 kHz**. `LTX2VocoderWithBWE`
(`VO:479-597`, mel-STFT + second vocoder + resampler to 48 kHz) is LTX-2.3's; the
2.0 repo's `model_index.json` names `LTX2Vocoder`.

### `ltx2_diffusion_decoder.py`

`LTX2VideoDiffusionDecoderModel` — "the LTX-2 diffusion video decoder,
introduced in LTX-2.5": a pixel-space diffusion model with neighbourhood
attention and SwiGLU that replaces the *video* VAE decoder, driven by
`LTX2VideoDiffusionDecodePipeline` with its own scheduler. It is not in
`Lightricks/LTX-2`, `LTX2Pipeline` never calls it (`P:1643` uses `vae.decode`),
and **LTX-2.0 T2AV does not use it**. Out of scope.

---

## e. The DiT (`LTX2VideoTransformer3DModel`)

18.88 B parameters: video stream 12.89 B, audio stream 3.22 B, audio↔video
cross-attention 2.42 B, globals 0.34 B. 48 identical blocks, each holding *both*
streams (772 MB bf16 per block).

### Inputs and embedders (`T:1493-1600`)

* `proj_in: Linear(128, 4096)`, `audio_proj_in: Linear(128, 2048)`.
* Text: `caption_projection` / `audio_caption_projection` (§b).
* Four-plus-two timestep MLPs, all `LTX2AdaLayerNormSingle(D, k)` (`T:104-142`):

  ```
  s   = sinusoid_256(t)            # [cos(t·f_i) | sin(t·f_i)], f_i = 10000^(-i/128), i<128
  e   = linear_2(silu(linear_1(s)))            # 256 → D → D      ("embedded_timestep")
  mod = linear(silu(e))                        # D → k·D
  ```

  (`Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)` — this is
  exactly the backend's existing `sinusoidal_timesteps`.)

  | module | D | k | input | used for |
  |---|---|---|---|---|
  | `time_embed` | 4096 | 6 | `t` | video self-attn + FFN AdaLN; `e` for the video head |
  | `audio_time_embed` | 2048 | 6 | `t` | audio ditto |
  | `av_cross_attn_video_scale_shift` | 4096 | 4 | `t` | video side of a↔v scale/shift |
  | `av_cross_attn_audio_scale_shift` | 2048 | 4 | `t` | audio side |
  | `av_cross_attn_video_a2v_gate` | 4096 | 1 | `t · (1000/1000)` | a2v output gate |
  | `av_cross_attn_audio_v2a_gate` | 2048 | 1 | `t · (1000/1000)` | v2a output gate |

  `t = 1000·σ`, one scalar per batch element for T2AV (`P:1396`), so every
  modulation is a per-step constant vector broadcast over tokens.
  `cross_attn_timestep_scale_multiplier / timestep_scale_multiplier = 1`
  (`T:1523-1525`). `use_cross_timestep` (pipeline default True) swaps which
  modality's σ feeds the cross-attention modulation; in T2AV both σ are equal,
  so it changes nothing (`T:1560`, `T:1576`).

### Block (`T:597-811`), in order

Notation: `rms(x) = x·rsqrt(mean(x²) + 1e-6)` over the last axis, **no weight**
(`norm_elementwise_affine = false`). `tab + mod` means
`scale_shift_table[None,None] + mod.reshape(B, 1, rows, D)`, unbound along rows
(`T:586-595`). Tables are F32 on disk, everything else bf16.

1. **Self-attention**, per stream. Rows of `scale_shift_table [6, D]` in order:
   `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`.
   ```
   h  = rms(x)·(1 + scale_msa) + shift_msa
   x += attn1(h, rope = self-attn table) · gate_msa
   ```
   Video: 32 heads × 128, `video_rotary_emb`. Audio: 32 × 64, `audio_rotary_emb`.
2. **Text cross-attention**, per stream — no modulation, **no gate**, no RoPE, no
   mask: `x += attn2(rms(x), kv = projected text [B,1024,D])`. Video `attn2` is
   4096-wide (k/v `Linear(4096, 4096)`), audio `audio_attn2` 2048-wide.
3. **Audio↔video cross-attention**, both directions every block, both computed
   from the **same pre-update** normalised states (`T:738-739`):
   ```
   nv = rms(x_v);  na = rms(x_a)
   # video table [5,4096] + av_cross_attn_video_scale_shift(t):  rows 0..3 =
   #   (a2v_scale, a2v_shift, v2a_scale, v2a_shift)   — scale FIRST here (T:749)
   # row 4 + av_cross_attn_video_a2v_gate(t) = a2v_gate.  Audio table likewise,
   #   row 4 + av_cross_attn_audio_v2a_gate(t) = v2a_gate.
   a2v:  x_v += a2v_gate · audio_to_video_attn(q = nv·(1+v.a2v_scale)+v.a2v_shift,
                                               kv = na·(1+a.a2v_scale)+a.a2v_shift,
                                               q_rope = cross_video, k_rope = cross_audio)
   v2a:  x_a += v2a_gate · video_to_audio_attn(q = na·(1+a.v2a_scale)+a.v2a_shift,
                                               kv = nv·(1+v.v2a_scale)+v.v2a_shift,
                                               q_rope = cross_audio, k_rope = cross_video)
   ```
   Both use the **audio** head layout, 32 heads × 64 = 2048 (`T:529-557`):
   a2v `to_q: 4096→2048`, `to_k/to_v: 2048→2048`, `to_out: 2048→4096`;
   v2a `to_q: 2048→2048`, `to_k/to_v: 4096→2048`, `to_out: 2048→2048`. No mask.
   `nv` for v2a is the state *before* the a2v update of the same block.
4. **FFN**, per stream:
   `x += ff(rms(x)·(1 + scale_mlp) + shift_mlp) · gate_mlp`,
   `ff = Linear(D, 4D) → gelu_tanh → Linear(4D, D)` (`activation_fn:
   "gelu-approximate"`; 16 384 / 8 192 inner).

### Attention (`LTX2Attention`, `T:161-228`, `T:330-409`)

```
q = to_q(x);  k = to_k(ctx);  v = to_v(ctx)                 # all with bias
q = norm_q(q); k = norm_k(k)     # RMSNorm over the FULL inner dim (heads·head_dim),
                                 # eps 1e-6, WITH weight [inner]
q = rope(q, q_table); k = rope(k, k_table or q_table)       # skipped for text cross-attn
out = to_out(sdpa(split_heads(q,k,v), scale = head_dim^-0.5))
```

`qk_norm = "rms_norm_across_heads"` means the normalisation statistic is taken
across **all heads jointly** (one mean over 4096 or 2048 values per token), with
a per-channel weight. This is what the backend's fused QK-norm already does (it
is the Wan convention); it is *not* a per-head norm. The a↔v attentions normalise
over 2048.

### RoPE — "split", fractional positions, float64 frequency grid

`LTX2AudioVideoRotaryPosEmbed` (`T:814-1078`, identical to
`LT:ltx-core/…/transformer/rope.py:87-184`).

**Positions** are physical coordinates of each token's *extent*, and the RoPE
uses the midpoint (`use_middle_indices_grid`, `T:1013-1016`):

* Video token at latent `(f, h, w)` (`T:906-941`):
  * time: `[start, end) = clamp(8·[f, f+1) + 1 - 8, min 0)` pixel frames
    (`causal_offset = 1`: latent 0 covers frame `[0,1)`, latent 1 `[1,9)`, …),
    divided by fps → seconds. Midpoints at 24 fps: `0.5/24, 5/24, 13/24, …`.
  * height: `[32h, 32h + 32)` px, midpoint `32h + 16`; width likewise.
* Audio token `i` (`T:972-995`): `start = max(4i + 1 - 4, 0)·160/16000`,
  `end = max(4(i+1) + 1 - 4, 0)·160/16000` seconds. Midpoints: `0.005`,
  `0.03`, `0.07`, … (then +0.04 each).

**Fractions** (`T:1019-1024`): `frac = midpoint / max_pos` with `max_pos =
(20, 2048, 2048)` for video (seconds, px, px) and `(20,)` for audio.

**Frequencies** (`T:1028-1039`): for a table of width `dim` over `A` axes,
`n = dim // (2A)` and

```
freq_k = θ^(k/(n-1)) · π/2,   k = 0..n-1,  θ = 10000    # linspace in float64, then cast to f32
angle[token, k, axis] = freq_k · (2·frac_axis - 1)      # f32
flat = angle laid out frequency-major:  [k=0: t,h,w | k=1: t,h,w | …]   # transpose(-1,-2).flatten
```

`flat` has `n·A` entries; it is **left-padded** with identity slots (cos 1,
sin 0) to `dim/2`, then **reshaped to `[heads, dim/2/heads]`** (`T:1054-1076`):
head `j` owns flat slots `[j·r, (j+1)·r)`. **Different heads rotate with
different frequencies** — the table is `[B, H, S, r]`, not `[S, r]`.

| table | dim | axes | n | pad | heads × r |
|---|---|---|---|---|---|
| video self (`rope`) | 4096 | 3 | 682 | 2 | 32 × 64 |
| audio self (`audio_rope`) | 2048 | 1 | 1024 | 0 | 32 × 32 |
| a↔v video side (`cross_attn_rope`) | 2048 | 1 (time only) | 1024 | 0 | 32 × 32 |
| a↔v audio side (`cross_attn_audio_rope`) | 2048 | 1 | 1024 | 0 | 32 × 32 |
| connectors | 3840 | 1 | 1920 | 0 | 30 × 64 |

The a↔v tables use only the **time** coordinate of each stream
(`video_coords[:, 0:1]`, `T:1508-1511`), both normalised by
`max(20, 20) = 20` s, so a video token and an audio token at the same instant
get the same rotation: the cross-attention is aligned in wall-clock time.

**Application** (`T:46-84`), per head, `x = [x₁ | x₂]` halves of width `r`:

```
y₁ = x₁·cos - x₂·sin
y₂ = x₂·cos + x₁·sin            # rotate_half with a half-width table, in float32
```

applied to the *flat* `[B, S, inner]` q/k after the QK-norm and before the head
split. With 2 identity slots in front, video head 0's first two pairs are not
rotated.

**Mapping onto the backend's `rope_half`.** `CudaTensor::rope_half` rotates
channels `[0, R)` of `[B, H, S, D]` with `[S, R]` tables, pairing channel `j` with
`j + R/2`: `y_j = x_j·cos[p,j] + (j < R/2 ? -x_{j+R/2} : x_{j-R/2})·sin[p,j]`.
LTX-2's "split" is **the same rotate_half convention with `R = D`** (the whole
head, pairs `(j, j + D/2)`, *not* interleaved pairs, and *no* per-axis channel
blocks as in Wan). Exactly two things differ:

1. the reference table is half-width, one value per pair — for `rope_half`
   duplicate it: `cos_full[…, j] = cos_full[…, j + D/2] = cos_ltx[…, j]`;
2. the table has a **head axis**: `[H, S, D/2]`. Index map, with `r = D/2`,
   `A` axes, `n = inner // (2A)` frequencies per axis and `P = inner/2 - n·A`
   identity slots: head `h`, pair `j` reads flat slot `m = h·r + j`; if `m < P`
   the pair is not rotated (cos 1, sin 0); otherwise `m' = m - P`, frequency
   index `k = m' // A`, axis `a = m' % A`, and
   `angle = θ^(k/(n-1)) · π/2 · (2·frac_a(token) - 1)`.
   In the flat `[B, S, inner]` layout (head `h` = channels `[h·D, (h+1)·D)`), the
   rotated pairs are channels `(h·D + j, h·D + j + r)`.

Because the kernel indexes its table by `(i / D) % S`, the head axis can be
folded into the sequence axis with **no new kernel**: reshape q from
`[B, H, S, D]` to `[B, 1, H·S, D]`, call `rope_half` with `[H·S, D]` tables laid
out head-major (`row = h·S + s`), reshape back. Batch-major memory order makes
this exact for any `B` as long as the tables are the same for every batch
element (they are: one prompt geometry per run). a↔v attention simply calls it
twice with different tables — q with its own stream's time table
(`[32·S_q, 64]`), k with the other stream's (`[32·S_kv, 64]`). A `[H, S, R]`
table argument would be the cleaner long-term signature; it is not required.
The existing fused `qk_norm_rope_bhsd` must be called with `rope = None` (its
RoPE is Wan's interleaved one) and followed by `rope_half`.

### Output heads (`T:1676-1689`)

```
(shift, scale) = scale_shift_table[2, D] + e[:, :, None]      # row 0 = shift, row 1 = scale; e = embedded_timestep
out = proj_out( layer_norm(x, eps 1e-6, no affine) · (1 + scale) + shift )      # 4096→128 / 2048→128
```

Note `LayerNorm` here, `RMSNorm` everywhere else. The output is the **velocity**
`v = ε - x₀` for each stream, `[B,S,128]` and `[B,L,128]`. Unpatchify is the
inverse of the packing in §a.

### Sequence lengths

Video 6 144 (stage 2: 24 576) × 4096; audio 126 × 2048; text 1 024 per stream.
Per block: video self 6144², audio self 126², text cross 6144×1024 and 126×1024,
a2v 6144×126, v2a 126×6144.

### FastVideo

`FV` is a port of the Lightricks `ltx-core` model (same original key names, σ in
and ×1000 inside, `FV:998-1010`). For 2.0 it computes the same function as
diffusers: identical block order, `vx_norm3`/`ax_norm3` taken once before both
a↔v directions (`FV:2141-2142`), identical RoPE (`FV:792-935`), a↔v gate factor
`av_ca_timestep_scale_multiplier / timestep_scale_multiplier` (`FV:1182`). No
behavioural difference was found that the weights would care about.

---

## f. Sampling loop

`scheduler_config.json` (dev): `num_train_timesteps 1000, shift 1.0,
use_dynamic_shifting true, base_shift 0.95, max_shift 2.05, base_image_seq_len
1024, max_image_seq_len 4096, shift_terminal 0.1, time_shift_type exponential,
stochastic_sampling false`, all `use_*_sigmas` false, `invert_sigmas` false.
Distilled (`rootonchair` repo): same with `use_dynamic_shifting false`,
`shift_terminal null`.

`set_timesteps(sigmas=…)` (`S:343-380`): cast to f32 → (dynamic shift | static
`shift·σ/(1+(shift-1)σ)`, identity at shift 1) → (terminal stretch if set) →
`timesteps = σ·1000`, append σ = 0.

Distilled loop (both streams, same σ, `P:1389-1580` with every guidance off):

```
x_v, x_a ~ N(0, I)  float32, packed                         # σ₀ = 1: pure noise
for i in 0..8:
    t = float32(σ_i) · 1000
    v_v, v_a = DiT(x_v, x_a, text_v, text_a, t)             # one joint forward
    x_v += (σ_{i+1} - σ_i) · v_v;   x_a += (σ_{i+1} - σ_i) · v_a      # float32
```

diffusers converts `v → x₀ = x - σv → v' = (x - x₀)/σ` even with no guidance
(`P:1466-1467`, `P:1573-1574`); that is the identity up to one float32 rounding
and the port may skip it. Steps: `dt = -0.00625` ×4, then `-0.065625,
-0.184375, -0.303125, -0.421875` (sum −1).

For reference, the **dev** schedule (`P:1334-1360`, = `LT:…/schedulers.py:21-57`):
`linspace(1, 1/N, N)` → `σ' = e^μ/(e^μ + 1/σ - 1)` with
`μ = 0.95 + (S_video - 1024)·1.1/3072` (**not clamped**: μ = 2.783 at 6 144
tokens, 9.383 at 24 576) → stretch `σ'' = 1 - (1-σ')·0.9/(1-σ'_last)` so the last
value is 0.1. Dev guidance: CFG 4 (card) / 3 (constants) with STG on block 29
and modality guidance 3, 40 steps — 3–4 forwards per step, not a first target.
`schedule.rs` implements both and tests them against hand-computed values.

---

## g. Weights

On-disk dtypes are from the safetensors headers (HTTP range reads; no shard
was downloaded).

### Totals

| component | file(s) | tensors | bytes | dtype |
|---|---|---|---|---|
| DiT | `transformer/*-0000{1..8}-of-00008` | 3 510 | 37 758 861 824 (35.17 GiB) | BF16; 13 MB of F32 AdaLN tables |
| connectors | `connectors/diffusion_pytorch_model.safetensors` | 59 | 2 862 950 400 | BF16 |
| video VAE | `vae/…` | 184 | 2 444 959 714 — **decoder 1 056 761 952**, encoder 1 388 197 762 | BF16 |
| audio VAE | `audio_vae/…` | 102 | 106 496 804 — decoder 63 830 788 | BF16 |
| vocoder | `vocoder/…` (own folder, own file) | 194 | 111 185 604 | BF16 |
| Gemma-3-12B | `text_encoder/model-0000{1..11}-of-00011` | 1 065 | 48 749 300 160 — **language model 47 064 136 704** (embeddings 4 027 514 880), vision 1 685 163 456 | **F32** |
| latent upsampler | `latent_upsampler/…` | 73 | 995 735 858 | BF16 |
| single file | `ltx-2-19b-distilled.safetensors` | 4 052 | 43 285 058 186 | BF16 + F32 tables |
| single file fp8 | `ltx-2-19b-distilled-fp8.safetensors` | 6 404 | 27 078 716 346 | F8_E4M3 16.21 GB, BF16 10.86 GB |
| distilled LoRA | `ltx-2-19b-distilled-lora-384.safetensors` | 2 742 | 7 674 558 424 | BF16, rank 384 |

`text_encoder/diffusion_pytorch_model-*-of-00012` is a **stale duplicate**
(Gemma under `base_text_encoder.*` plus an old copy of the connectors,
51.6 GB). Ignore it; load `model-*-of-00011` via `model.safetensors.index.json`.

### DiT (diffusers names; `D` = 4096 video / 2048 audio)

| key family | shape | dtype |
|---|---|---|
| `proj_in.{weight,bias}` / `audio_proj_in.*` | `[4096,128]` / `[2048,128]` | BF16 |
| `proj_out.*` / `audio_proj_out.*` | `[128,4096]` / `[128,2048]` | BF16 |
| `caption_projection.linear_1.*`, `.linear_2.*` | `[4096,3840]`, `[4096,4096]` | BF16 |
| `audio_caption_projection.linear_1.*`, `.linear_2.*` | `[2048,3840]`, `[2048,2048]` | BF16 |
| `{time_embed, av_cross_attn_video_scale_shift, av_cross_attn_video_a2v_gate}.emb.timestep_embedder.linear_{1,2}.*` | `[4096,256]`, `[4096,4096]` | BF16 |
| `…linear.*` for those three | `[24576,4096]`, `[16384,4096]`, `[4096,4096]` | BF16 |
| `{audio_time_embed, av_cross_attn_audio_scale_shift, av_cross_attn_audio_v2a_gate}.emb.…linear_{1,2}.*` | `[2048,256]`, `[2048,2048]` | BF16 |
| `…linear.*` for those three | `[12288,2048]`, `[8192,2048]`, `[2048,2048]` | BF16 |
| `scale_shift_table` / `audio_scale_shift_table` | `[2,4096]` / `[2,2048]` | **F32** |
| `transformer_blocks.N.{attn1,attn2}.{to_q,to_k,to_v,to_out.0}.*` | `[4096,4096]` + `[4096]` | BF16 |
| `transformer_blocks.N.{attn1,attn2}.{norm_q,norm_k}.weight` | `[4096]` | BF16 |
| `transformer_blocks.N.{audio_attn1,audio_attn2}.{to_q,to_k,to_v,to_out.0}.*`, `norm_{q,k}.weight` | `[2048,2048]`, `[2048]` | BF16 |
| `transformer_blocks.N.audio_to_video_attn.{to_q | to_k,to_v | to_out.0}.weight` | `[2048,4096]` \| `[2048,2048]` \| `[4096,2048]` | BF16 |
| `transformer_blocks.N.video_to_audio_attn.{to_q | to_k,to_v | to_out.0}.weight` | `[2048,2048]` \| `[2048,4096]` \| `[2048,2048]` | BF16 |
| `…{audio_to_video_attn,video_to_audio_attn}.norm_{q,k}.weight` | `[2048]` | BF16 |
| `transformer_blocks.N.ff.net.0.proj.*`, `ff.net.2.*` | `[16384,4096]`, `[4096,16384]` | BF16 |
| `transformer_blocks.N.audio_ff.net.0.proj.*`, `audio_ff.net.2.*` | `[8192,2048]`, `[2048,8192]` | BF16 |
| `transformer_blocks.N.scale_shift_table` / `audio_scale_shift_table` | `[6,4096]` / `[6,2048]` | **F32** |
| `transformer_blocks.N.video_a2v_cross_attn_scale_shift_table` / `audio_a2v_…` | `[5,4096]` / `[5,2048]` | **F32** |

All Linears have biases. The block norms and the output LayerNorms have no
parameters. (diffusers casts the F32 tables to bf16 at load; keeping them F32 is
harmless and is what our f32 device tensors want anyway.)

### Connectors

`text_proj_in.weight [3840,188160]` (1.445 GB, no bias);
`{video,audio}_connector.learnable_registers [128,3840]`;
`{video,audio}_connector.transformer_blocks.{0,1}.attn1.{to_q,to_k,to_v,to_out.0}.{weight [3840,3840], bias}`,
`.attn1.norm_{q,k}.weight [3840]`, `.ff.net.0.proj.* [15360,3840]`,
`.ff.net.2.* [3840,15360]`. All BF16.

### Video VAE decoder

`decoder.conv_in.conv.* [1024,128,3,3,3]`;
`decoder.mid_block.resnets.{0..4}.conv{1,2}.conv.* [1024,1024,3,3,3]`;
`decoder.up_blocks.{0,1,2}.upsamplers.0.conv.conv.weight`
`[4096,1024,3,3,3]`, `[2048,512,3,3,3]`, `[1024,256,3,3,3]`;
`decoder.up_blocks.{0,1,2}.resnets.{0..4}.conv{1,2}.conv.*` at 512, 256, 128;
`decoder.conv_out.conv.* [48,128,3,3,3]`; `latents_mean [128]`,
`latents_std [128]`. All BF16, all with bias.

### Audio VAE decoder and vocoder

`decoder.conv_in.conv.* [512,8,3,3]`; `decoder.mid.block_{1,2}.conv{1,2}.conv.*
[512,512,3,3]`; `decoder.up.2.block.{0,1,2}.conv{1,2}.conv.*` @512,
`decoder.up.2.upsample.conv.conv.* [512,512,3,3]`;
`decoder.up.1.block.0.{conv1.conv [256,512,3,3], nin_shortcut.conv [256,512,1,1]}`,
rest @256, `decoder.up.1.upsample.conv.conv.* [256,256,3,3]`;
`decoder.up.0.block.0.{conv1.conv [128,256,3,3], nin_shortcut.conv [128,256,1,1]}`,
rest @128; `decoder.conv_out.conv.* [2,128,3,3]`; `latents_mean [128]`,
`latents_std [128]`.

Vocoder: `conv_in.* [1024,128,7]`; `upsamplers.{0..4}.weight`
`[1024,512,16]`, `[512,256,15]`, `[256,128,8]`, `[128,64,4]`, `[64,32,4]`
(ConvTranspose1d layout `[in, out, k]`); `resnets.{0..14}.convs{1,2}.{0,1,2}.*
[C,C,k]` with `C = 512,256,128,64,32` for resnets `3i…3i+2` and `k = 3,7,11`;
`conv_out.* [2,32,7]`.

### Gemma (keys under `language_model.model.`)

`embed_tokens.weight [262208,3840]`; per layer `self_attn.q_proj [4096,3840]`,
`k_proj`/`v_proj [2048,3840]`, `o_proj [3840,4096]`, `q_norm`/`k_norm [256]`,
`mlp.gate_proj`/`up_proj [15360,3840]`, `down_proj [3840,15360]`,
`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`,
`post_feedforward_layernorm [3840]`; `norm.weight [3840]`. No biases. All F32.
224.1 M parameters per layer (896 MB F32, 448 MB bf16).

### Single-file naming → diffusers naming

From diffusers `scripts/convert_ltx2_to_diffusers.py:34-52, 60-86, 117-146`. The
conversion is a pure rename (plus dropping unused statistics) — no tensor is
reshaped, fused or transposed.

| single file | diffusers |
|---|---|
| `model.diffusion_model.` | (DiT root) |
| `patchify_proj` / `audio_patchify_proj` | `proj_in` / `audio_proj_in` |
| `adaln_single` / `audio_adaln_single` | `time_embed` / `audio_time_embed` |
| `av_ca_video_scale_shift_adaln_single` | `av_cross_attn_video_scale_shift` |
| `av_ca_audio_scale_shift_adaln_single` | `av_cross_attn_audio_scale_shift` |
| `av_ca_a2v_gate_adaln_single` / `av_ca_v2a_gate_adaln_single` | `av_cross_attn_video_a2v_gate` / `av_cross_attn_audio_v2a_gate` |
| `…scale_shift_table_a2v_ca_video` / `_audio` | `video_a2v_cross_attn_scale_shift_table` / `audio_a2v_…` |
| `q_norm` / `k_norm` | `norm_q` / `norm_k` |
| `model.diffusion_model.{video,audio}_embeddings_connector.transformer_1d_blocks` | `connectors`: `{video,audio}_connector.transformer_blocks` |
| `text_embedding_projection.aggregate_embed` | `connectors`: `text_proj_in` |
| `vae.decoder.up_blocks.{0 \| 1,3,5 \| 2,4,6}` | `mid_block` \| `up_blocks.{0,1,2}.upsamplers.0` \| `up_blocks.{0,1,2}` |
| `vae.…res_blocks` | `resnets` |
| `vae.per_channel_statistics.{mean-of-means, std-of-means}` | `latents_mean`, `latents_std` (other statistics dropped) |
| `audio_vae.per_channel_statistics.{mean-of-means, std-of-means}` | `latents_mean`, `latents_std` |
| `vocoder.{conv_pre, ups, resblocks, conv_post}` | `{conv_in, upsamplers, resnets, conv_out}` |

The single file's `__metadata__.config` JSON carries the original config
(`positional_embedding_max_pos [20,2048,2048]`, `use_middle_indices_grid true`,
`rope_type split`, `frequencies_precision float64`,
`connector_positional_embedding_max_pos [4096]`, `causal_temporal_positioning
true`, …) and agrees with the diffusers config.

### What to load

1. **DiT + connectors + VAEs + vocoder: `ltx-2-19b-distilled.safetensors`**, the
   official Lightricks artefact, through the rename table above in the streaming
   loader (one 603 KB header, tensors read by offset; never the whole 43 GB in
   host RAM). Skip `vae.encoder.*`, `audio_vae.encoder.*`.
2. **Gemma: `Lightricks/LTX-2/text_encoder/model-*-of-00011`**, keys under
   `language_model.` only, **cast F32 → bf16 while streaming**.
3. Tokenizer: `tokenizer/tokenizer.json`.
4. Oracle: `rootonchair/LTX-2-19b-distilled` (community diffusers conversion of
   the same single file; shares every non-DiT, non-connector blob with
   `Lightricks/LTX-2`). On the box, hash a handful of renamed tensors against it
   once — that closes the "is the community conversion faithful" question.

---

## h. Memory plan

### 96 GB (RTX PRO 6000) — everything resident

| item | bytes |
|---|---|
| DiT bf16 (F32 tables) | 37.76 GB |
| connectors bf16 | 2.86 GB |
| video VAE decoder bf16 | 1.06 GB |
| audio VAE decoder + vocoder | 0.18 GB |
| Gemma LM bf16 (if kept resident for a warm pipeline) | 23.53 GB (21.5 GB without the embedding table — gather rows on the host) |
| **weights total** | **65.4 GB** warm / 41.9 GB with Gemma streamed and freed |

Activations at the default 6 144 video tokens, f32 device tensors, B = 1:

| tensor | size |
|---|---|
| video hidden `[6144,4096]` | 96 MB (audio `[126,2048]`: 1 MB) |
| FFN inner `[6144,16384]` | 384 MB |
| q, k, v | 3 × 96 MB |
| video RoPE cos + sin `[32,6144,64]` | 2 × 48 MB (cross tables 2 × 24 MB) |
| projected text `[1024,4096]` + `[1024,2048]` | 25 MB, once per prompt |
| self-attn scores, dense | 144 MB per head, **4.5 GB for 32 heads** → query-chunk, or flash (d = 128 qualifies: `d % 32 == 0`, `d ≤ 128`) |
| text cross-attn scores `[32,6144,1024]` | 750 MB dense |
| a2v / v2a scores `[32,6144,126]` | 95 MB each |

Peak well under 8 GB with chunked dense SDPA, under 2 GB with flash. At stage-2
resolution (24 576 tokens): hidden 384 MB, FFN inner 1.5 GB, dense self-attn
scores 2.3 GB **per head** (72 GB for all) — flash or aggressive chunking is
mandatory; text cross 3 GB dense.

**Gemma, streamed layer by layer**: 448 MB bf16 per layer uploaded, run, freed;
host reads F32 (896 MB/layer) and narrows. Working set: hidden `[n,3840]`
(≤ 15 MB), MLP inner `[n,15360]` (≤ 60 MB), causal scores `[16,n,n]` ≤ 64 MB at
n = 1024 — **the materialised `[B,H,S,S]` mask is affordable here**, it is not a
blocker for Gemma. The 49-state stack is `n × 3840 × 49 × 4` B (735 MB at
n = 1024, typically 50–150 MB); keep it on the host, where the masked
normalisation (§b) is also cheapest to do. `text_proj_in` then takes a
`[n, 188160]` input (≤ 735 MB f32) through one bf16 GEMM.

**VAE**: §c table — ~5 GB f32 at 768×512×121, untiled. At 1536×1024 the last
stage is 5.7 GB per tensor (≥ 25 GB live) — decode in temporal chunks of ≤ 4
latent frames with overlap, accepting the blend approximation, or free the DiT
first.

### 32 GB (RTX 5090)

bf16 DiT (37.8 GB) does not fit. Options: (1) the fp8 single file; (2) stream
DiT blocks from pinned host memory (772 MB per block, 48 per step, 8 steps).

`ltx-2-19b-distilled-fp8.safetensors`: **per-tensor** scaling, **F8_E4M3FN**
weights, scalar F32 `weight_scale` and a scalar F32 static `input_scale` per
Linear (`__metadata__._quantization_metadata`: `format_version 1.0`, every entry
`{"format": "float8_e4m3fn"}`). Only the 28 attention/FFN Linears of **blocks
1–42** are quantised (1 176 layers); blocks 0 and 43–47, every bias, every
norm weight, the AdaLN MLPs, embedders, connectors, VAEs and vocoder stay BF16.
Semantics (`LT:ltx-core/…/quantization/fp8_scaled_mm.py:49-66, 79-98, 173-186`):
`W = fp8.float() · weight_scale`; activations
`x_q = clamp(x / input_scale, ±448).to(fp8)`, then `_scaled_mm(x_q, W_q,
scale_a = input_scale, scale_b = weight_scale)`, bias added in bf16. DiT
footprint: 20.86 GB blocks + 0.69 GB globals ≈ **21.6 GB**, leaving ~8 GB for
activations at 6 144 tokens with Gemma streamed — feasible for stage 1 only.
`-fp4` is NVFP4 (block-scaled), not considered.

---

## i. Ops checklist

Status against the working tree on `feat/vast-ui` (2026-09-19), where the lead's
foundation has landed: lazy shard loader (`WeightMap::open` / `open_files`, F32/F16
narrowed to bf16 on upload, FP8 readable as raw bytes), `CudaTensor::{rope_half,
repeat_kv, leaky_relu, snake_beta, gelu_erf, conv1d, conv_transpose1d, pad
(zeros | reflect | replicate), group_norm}`, WAV + AAC mux, and the streaming
decoder `fastvideo_cudarc::llm`. **have** = usable as is; **compose** = no kernel
needed, built from existing ops; **new** = still to write.

Loader
- **have** lazy reads by offset, one tensor of host RAM, F32 → bf16 narrowing
  (Gemma), `open_files` for the single 43 GB checkpoint.
- **done** key-rename view for the single-file naming (§g table):
  `ltx2::keys::Keys` renames whole dot-separated segments of a diffusers name,
  so graph code asks for diffusers names under either layout. It covers the DiT
  and the connectors; the VAEs, vocoder and Gemma load from the diffusers
  folders (byte-identical in both repos), so the `up_blocks` index remap of the
  single file's VAE is not needed. Nothing has to be filtered: the lazy store
  only reads the keys a loader asks for. The root of
  `text_embedding_projection.aggregate_embed` is probed (bare, then under
  `model.diffusion_model.`).
- **new (5090 only)** consume F8_E4M3 weight + scalar `weight_scale` /
  `input_scale` with a *static* input scale; `fp8.rs` currently quantises
  dynamically.

Gemma-3 — **have**, via `llm::hidden_states` (§b cross-check)
- GQA 16/8 with `repeat_kv`, head_dim 256 (dense SDPA; flash is `d ≤ 128`),
  rotate_half RoPE with explicit positions and per-layer θ/factor, per-head
  `(1+w)` QK-norm, sandwich norms, GeGLU tanh, causal + key-padding mask
  (`[1,1,S,S]` f32 ≤ 4 MB per distinct window at S = 1024 — fine), lazy embedding
  row gather, streaming one layer at a time.
- **change** `embed_scale` 61.9677 → **62.0** for bf16 parity; feed the oracle's
  `positions`.
- Not needed: erf-GELU, SwiGLU, logit soft-capping, KV cache, vision tower.
  The sliding window never bites at S ≤ 1024.

Connectors
- **compose (host)** masked mean / min / max over `(tokens, channels)` for each of
  the 49 states, affine — once per prompt, on the host copy of the state stack
  (f64 accumulation). No device reduction kernel required. The port never
  materialises the pad rows: Gemma runs the `n` real tokens at positions
  `1024-n … 1023`, the packed input is `[n, 188160]`, and because
  `text_proj_in` has no bias the reference's zeroed pad rows project to zero and
  are then overwritten by registers anyway.
- **compose** front-gather of real tokens + register fill (`cat` / `narrow`).
- **have** bf16 Linear for `text_proj_in` (188160 → 3840, input ≤ 735 MB f32).
- Blocks reuse the DiT attention below (30 heads × 128, 1-D table).

DiT
- **compose** "split" RoPE through `rope_half` with the head axis folded into
  the sequence axis (§e, "Mapping onto the backend's `rope_half`"). Optional
  nicety: a `[H, S, R]` table signature.
- **new (host, pure Rust)** the fractional-position table builder: f64
  `θ^(k/(n-1))·π/2` grid cast to f32, extent midpoints in seconds / pixels,
  frequency-major flattening, left identity pad, per-head chunking, cos
  duplicated across halves; four tables per run + one for the connectors.
- **have** QK-norm across `heads·head_dim` with weight — the existing fused
  kernel with `rope = None`; widths 4096, 2048, 3840.
- **compose / new (perf)** weightless RMSNorm + AdaLN
  `rms(x)·(1+scale)+shift`: since every modulation is a per-step constant
  vector at batch 1, the port passes `1 + scale` as the RMSNorm kernel's
  *weight* and follows with one broadcast add (two launches, not three); gated
  residuals are `residual_gate_add_e`, the output heads `ln_adaln_e`; the fused `ln_adaln_e` is LayerNorm-based, so an RMS
  variant is the one DiT kernel worth adding for speed (4 sites × 2 streams × 48
  blocks per step). Modulation vectors are per-step constants
  (`table + mod`, computed once per block per step).
- **have** gated residual (broadcast gate), plain residual, tanh-GELU FFN,
  affine-free LayerNorm (heads), `[cos | sin]` 256-wide timestep sinusoid
  (`sinusoidal_timesteps` matches `flip_sin_to_cos=True, shift 0`).
- **verify** SDPA with `S_q ≠ S_kv` in both directions (6144 × 126 and 126 × 6144,
  d = 64) on the flash and chunked-dense paths; d = 128 and d = 64 both satisfy
  the flash constraint.
- Not needed: **any masked SDPA** (every DiT and connector mask is all ones),
  per-token timesteps (I2V only), STG / perturbed attention, gated attention,
  prompt AdaLN, CFG batching.

Video VAE
- **have** `pad(Reflect)` on H, W (1 px) and `pad(Replicate)` on T (1 frame each
  side), then `conv3d` with pad 0; conv bias.
- **have** channel RMS norm (+ fused SiLU): `rms_norm_channels` with γ = 1 and
  **eps 1e-8**.
- **compose** 3-D depth-to-space (2,2,2) and the final 4×4 unpatchify via
  `reshape` + `permute` with the index maps in §c (note the swapped H/W pairing
  in the unpatchify); channel tiling ×4 via `cat`; frame drop via `narrow`.
  The device gather handles rank ≤ 6, and the reference's one-shot reshape is
  rank 8, so depth-to-space is two moves (space at rank 6 with `c` and `i`
  merged, then time at rank 4) and the unpatchify one rank-6 move with the
  batch axis dropped — batch 1 only.

Audio VAE + vocoder
- **have** asymmetric zero padding by per-axis `pad` (time 2/0, mel 1/1) then
  `conv2d` pad 0; nearest ×2 (`upsample_nearest2d`); `narrow`; channel RMS norm
  with **eps 1e-6**.
- **have** `conv1d` with dilation (k ∈ {3, 7, 11}, d ∈ {1, 3, 5}, "same" =
  `d(k-1)/2`), `conv_transpose1d` for `(k, s, p)` ∈ {(16,6,5), (15,5,5), (8,2,3),
  (4,2,1)}, no output padding — **check the weight layout is `[in, out, k]`**.
- **have** `leaky_relu` (slopes 0.1 and 0.01), tanh; mean of three = `lincomb`.
- Not needed for 2.0: Snake / SnakeBeta, anti-aliased activations, STFT.
  GroupNorm(32) is needed only by the stage-2 latent upsampler.

Output
- **have** WAV + AAC mux — must be driven at **24 000 Hz**, 2 channels (the
  vocoder's rate, not the VAE's 16 kHz).

Net: after the foundation, LTX-2 stage 1 needs **no mandatory new CUDA kernel** —
one host-side RoPE table builder, one loader rename view, and graph code. The
RMS-AdaLN fusion and a head-aware RoPE signature are performance/ergonomics
items.

---

## j. Oracle plan

`scripts/gpu/ltx2_oracle.py` follows `upstream_oracle.py`: float32 safetensors,
CPU-seeded noise saved alongside the outputs, a meta JSON with versions,
shapes, timings and per-tensor mean/std/absmax. One stage per model, each freed
before the next. Every comparison feeds **our** stage the **oracle's** input so
errors are attributed, and an `e2e` variant chains our own outputs.

| stage | tensors | compared how | proposed limit (rel. RMSE unless noted) |
|---|---|---|---|
| tokenizer | `text.input_ids`, `text.attention_mask` | exact | 0 mismatches |
| Gemma (`fv-gpucheck llm --family gemma3-12b`) | `--llm-out` file: `input_ids [S]`, `positions [S]` (= `arange(1024)`, what transformers used), `attend [S]`, `hidden_0 … hidden_48 [1,S,3840]`, all f32; reference is **bf16 on GPU** (recorded in the meta JSON) | rel-L2 per tap over attended rows; taps 1 and 6 (first global layer) localise an early divergence | tap 0 ≤ 3e-3 — the reference stores `bf16(w)·62` *in bf16* (7 mantissa bits → ≈ 2e-3 rel-L2 of pure rounding), so tap 0 measures the floor, not us; taps 1, 6 ≤ 5e-3; late taps ≤ 2e-2. A float32 rerun (`--text-dtype float32`, 49 GB, fits) tightens all of these to ≤ 1e-4 and is the run that can certify the arithmetic — with `embed_scale = sqrt(3840)` in that case |
| Gemma, main file | `text.hidden_states [n,3840,49]` (real tokens only) | input to the connector stage | f32 exact mode vs bf16 oracle: ≤ 2e-2 on late states, ≤ 1e-3 on state 0; bf16-vs-bf16: ≤ 5e-3 |
| connectors | `conn.proj`, `conn.video`, `conn.audio` (pipeline bf16) and `conn.*_f32` | ours(f32) vs `_f32` | ≤ 1e-4; and report ours vs bf16 — the gap between the two oracle variants is the noise floor |
| RoPE | `dit.rope.{video,audio,cross_video,cross_audio}.{cos,sin}`, `dit.*_coords` | abs max | ≤ 1e-6 (pure host math; anything larger is a layout bug) |
| DiT forward | `dit.video_in`, `dit.audio_in`, `dit.timestep` → `dit.block{00,24,47}.{video,audio}`, `dit.{video,audio}_out`, on oracle connector outputs | per tap | block 0 ≤ 2e-3, final ≤ 2e-2 vs a bf16 oracle; ≤ 1e-4 if the oracle is rerun with `--dit-dtype float32` (fits in 96 GB) |
| video VAE | `vae.latent [1,128,3,8,12]` → `vae.video [1,3,17,256,384]`, f32 oracle | abs max / RMSE | ≤ 1e-3 abs on `[-1,1]`, ≤ 1e-4 RMSE |
| audio VAE | `audio.latent [1,26,128]` → `audio.mel [1,2,101,64]` | RMSE | ≤ 1e-4 |
| vocoder | `audio.mel` → `audio.wave [1,2,24240]` | RMSE, plus SNR | ≤ 1e-4, SNR ≥ 60 dB |
| sampling (`--sample`) | `sample.{video,audio}_noise`, `sample.step{0..7}.*`, 3 decoded frames, `sample.mel`, `sample.wave` | per step | drift grows with steps: step 0 ≤ 5e-3, final latent ≤ 5e-2; decoded frames PSNR ≥ 35 dB vs oracle |

Notes:

* bf16 references set a noise floor of roughly 1e-3…1e-2 through 48 layers, so a
  bf16 oracle cannot certify better than that. Where the card allows (connectors,
  DiT at 76 GB, VAEs) run the oracle in **float32** for the tight limits and
  bf16 for "matches the product".
* The per-state normalisation runs in bf16 in both references, including a sum
  over up to 3.9 M values. `conn.video` vs `conn.video_f32` measures what that
  costs; if it is large, the port should mirror the bf16 arithmetic rather than
  be "more correct".
* The DiT stage uses `σ = 0.725` (a schedule entry, mid-trajectory) on pure
  seeded noise — not a realistic latent, but a deterministic, full-rank input.
  The `--sample` trajectory covers realistic ones.
* The sampling stage builds its scheduler explicitly (distilled settings) so a
  dev `scheduler/` folder cannot silently shift the sigmas.

---

## k. Open questions

1. **Register placement** (§b): diffusers and release-day Lightricks front-align
   the text; Lightricks `main` leaves it in place. We follow diffusers. If
   prompt adherence looks off, this is the first thing to A/B.
2. **bf16 normalisation** in the connectors (§j) — how torch's bf16 `sum`
   accumulates decides whether f32 is closer or further from the product. Measured
   by the oracle, not assumed.
3. **`rootonchair/LTX-2-19b-distilled`** is a community conversion. It shares
   blobs with the official repo for everything but the DiT and connectors; those
   two should be hashed tensor-by-tensor against
   `ltx-2-19b-distilled.safetensors` once on the box.
4. **Guidance defaults** in diffusers `main` are LTX-2.5's; any pipeline-level
   (rather than module-level) oracle must switch all of them off explicitly (§a).
5. **Audio length**: 126 latents → 5.01 s of audio for 5.04 s of video. Both
   references mux as is; we do the same.
6. Gemma positions: the oracle uses `arange(1024)` over the padded sequence. The
   port should use the same offsets for parity and may drop the pad rows.

---

## l. Implementation map (stage 1)

| piece | where | gpucheck stage |
|---|---|---|
| rotary tables (host, f32 in the reference's op order; golden test against a numpy transliteration of `T:906-1076` / `C:111-171`) | `fastvideo-models/src/ltx2/rope.rs` | `ltx2 dit --rope-only` |
| key view, `LTX2Attention` + FFN + folded-head `rope_half` | `fastvideo-cudarc/src/ltx2/{keys,attention}.rs` | — |
| tokenise / left-pad, 49-state stack, normalisation, connectors | `ltx2/text.rs` | `ltx2 text` |
| audio VAE decoder, vocoder (weight-norm folded at load if present) | `ltx2/{audio_vae,vocoder}.rs` | `ltx2 audio` |
| video VAE decoder, exact conv streaming | `ltx2/vae.rs` | `ltx2 vae` |
| DiT | `ltx2/transformer.rs` | `ltx2 dit` |
| noise, 8-step Euler, decode + mux | `ltx2/pipeline.rs` | `ltx2 loop`, `ltx2 gen` |

`--mode exact` (float32 GEMMs, TF32 off) is the mode for `text`, `audio` and
`vae`, whose oracles are float32. `dit`, `loop` and `gen` need `--mode fast`:
under `exact` every Linear would hold float32 weights, 76 GB for the DiT.

The oracle script tolerates both library generations (`dtype=` /
`torch_dtype=`, hidden-state access through the wrapper or the language model,
a hard check that the tokenizer really left-padded, and a sigma schedule
computed directly with the diffusers scheduler only asked to agree).

Not reproduced, on purpose: the pipeline's `v → x₀ → v` float32 round trip with
guidance off (§f), and diffusers' blended VAE tiling (§c).

### Checked without a GPU

* **Key manifests** (`ltx2/manifests/*.json`, `ltx2/manifest_tests.rs`): the
  safetensors headers of every published file (HTTP range reads; repo, revision
  and file recorded in each), and one test per loader that runs the real loader
  at the production config against a recording weight generator: every
  requested `(key, shape)` must exist, and no key of the component may go
  unrequested. The DiT loads blocks 0 and 47 and substitutes the index for the
  rest; both layouts go through `Keys`, and a separate test shows the rename
  view is a bijection onto the single file. This settled §k.1's neighbour: the
  single file keeps `text_embedding_projection.aggregate_embed` at its bare root.
* **diffusers at toy sizes** (`scripts/gpu/ltx2_tiny_reference.py`,
  `ltx2/fixtures/`, `ltx2/reference_tests.rs`): diffusers' own classes with tiny
  configs and seeded weights, float32 on the CPU; the production loaders and
  graphs reproduce the DiT (every block, all 48 sub-layer taps, both heads), the
  connectors, the video VAE (whole and streamed), the audio VAE and the vocoder
  to ≤ 5e-5.
* **Scalar division.** torch on CUDA divides a tensor by a Python scalar as a
  multiply by the float32 reciprocal; on the CPU it divides. In the rotary
  coordinates (`/ fps`, `/ max_pos`, `· hop / rate`) that is one ulp, and at the
  top of the frequency grid one float32 ulp of angle is 2⁻¹⁰ rad — the exact
  residue (2⁻¹⁰ video time, 2⁻⁹ audio) the first hardware run measured.
  `ScalarDivision::Reciprocal` is the production setting; the CPU fixtures use
  `Exact`.

### Reading a bf16 reference

The reference keeps the residual stream in bf16 through 192 residual adds; the
port keeps it in float32. Against a bf16 dump alone the reference's rounding is
indistinguishable from a port error, so `ltx2_oracle.py --dit-dtype both` also
runs the same module widened to float32 (`dit32.*`, `sample32.*`; TF32 off, math
SDPA) and records the bf16-vs-float32 distance per tap. `ltx2 dit` / `ltx2 loop`
then gate *ours vs float32* at `--floor-factor` (1.25) times that floor. Blocks
0 and 47 and both heads are tapped sub-layer by sub-layer
(`blockNN.{video,audio}.{attn1,attn2,av,ff}_{in,out,after}`,
`head.*.{norm,modulated}`; every 16th video token).

### The text path's cost

Measured: 15.7 s per new prompt, all of it moving weights — 47 GB of float32
read, narrowed on the host, 23.5 GB uploaded — for 31 MB of conditioning. Three
independent remedies, cheapest first:

* **Conditioning cache** (`ltx2/text_cache.rs`, `ltx2 gen --text-cache <dir>`,
  on by default under `~/.cache/fastvideo/ltx2-text`). The two connector outputs
  `[1, 1024, 3840]` — before the DiT's caption projections, so an entry does not
  depend on the DiT — keyed by sha256 over the stripped prompt, the tokenizer
  file, the padded length and an identity of the Gemma and connector *files*
  (name, size, first and last MiB: no store is opened on a hit). Entries carry
  their token ids and a hash of themselves; anything short, corrupt or made from
  other ids is a miss.
* **Slim checkpoint** (`ltx2/slim.rs`, `fv-gpucheck ltx2 slim-text --weights
  <root> --slim <out>`, CPU only). Language model only, projections narrowed
  once with the loader's own `half::bf16::from_f32` (bit-identical device
  weights), norms and — by default — the embedding left float32, tensors in load
  order, ~5 GB shards: 47.06 GB → 25.55 GB (23.53 GB with `--embed bf16`). Use
  with `ltx2 gen --text-weights <out>`. Not for `--mode exact` parity work.
* **Resident text** (`TextResidency`, `--text auto|resident|streamed`,
  `FASTVIDEO_LTX2_TEXT`). Gemma (21.5 GB of bf16 projections) and the connectors
  (2.9 GB) stay on the device beside the 38 GB DiT; decided on the first prompt
  that actually has to be encoded, from the device's free memory (needs the
  resident bytes + 8 GB). A process that only ever hits the cache never loads
  Gemma.

`Ltx2Pipeline` holds the DiT and decoders across generations; `ltx2 gen --warm`
runs one untimed generation first (cache bypassed) and reports the second.

## I2V encode (this tree)

| layer | status |
|---|---|
| First-frame RGB → latent (`ltx2/i2v_encode.rs`) | landed |
| Diffusers `encoder.*` partial path (`ltx2/vae_encoder.rs`) | landed when keys present |
| Full `AutoencoderKLLTX2Video` encoder ResNet/downsample parity | external (Hub weights) |
| Spatial stub fallback (no encoder keys) | landed |
| Per-token timesteps / STG | deferred |

`--image` prefers the Diffusers encoder path when `vae/` has `encoder.*` keys;
otherwise the spatial stub conditions latent frame 0.
