# FastH3 Trim 8-step

Target: [`FastVideo/FastVideo-FastH3-Trim-8-Step`](https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step)
at `9edcb6511d5ff8dd1579e8eec8afee77e098b8d8` (the official release only; no community finetunes or mirrors),
served beside FastH3 as `fasth3-trim-8step` (alias `h3-trim`). Owner request and download approval: 2026-10-13.

It is FastH3 V2 (docs/ports/h3.md) with 8 of the 50 transformer blocks removed, each block's AdaLN timestep
projection replaced by a shared rank-16 basis, retrained with 8-step DMD2 (checkpoint 300), and a lighter video
VAE. The model card: faster and smaller than FastH3 V2, with some quality cost.

Status (2026-10-13): **ported and served (bf16 / the release's float16 AdaLN), validated on one RTX PRO 6000**
(§6). NVFP4: published to R2, not loaded yet (§5).

Sources read (FastVideo `main` @ `d5253fd`, 2026-10-09, the first `main` with `adaln_rank`):
`fastvideo/models/dits/minimax_h3.py` (`F/dit`), `fastvideo/configs/models/dits/minimax_h3.py` (`F/dcfg`),
`fastvideo/pipelines/basic/minimax_h3/minimax_h3_pipeline.py` (`F/pipe`), `fastvideo/configs/pipelines/minimax_h3.py`
(`F/pcfg`), `fastvideo/configs/models/encoders/minimax_h3_qwen3_vl.py` (`F/qwencfg`),
`scripts/checkpoint_conversion/convert_minimax_h3_adaln_rank.py`, the Hub files (`HF/*`) and safetensors headers.

## 1. What differs from FastH3 V2

| | FastH3 V2 (`h3-8step`) | FastH3 Trim (`fasth3-trim-8step`) |
|---|---|---|
| DiT blocks | 50 | 42: source blocks `0-5, 8, 10-12, 14, 17-21, 24-49` (dropped 6, 7, 9, 13, 15, 16, 22, 23) |
| AdaLN per block | `Linear(2688, 96768)` on `silu(temb)`, bf16 (520 MB a block) | `Linear(16, 96768)` on `r`, float16 (6.2 MB a block) |
| shared AdaLN basis | none | `adaln_basis: Linear(2688, 16, bias=False)`: `r = adaln_basis(fp16(silu(temb)))` |
| `norm_out.linear` | `Linear(2688, 10752)` on `silu(temb)` | `Linear(16, 10752)` on `r` (no second silu) |
| transformer on disk | 70.10 GB bf16, 14 shards | 37.52 GB, 8 shards (bf16; the AdaLN factors stored F32, loaded as float16) |
| video VAE decoder | 36 ViT blocks, F32 (10.4 GB with encoder) | **26** ViT blocks, F16 (4.23 GB with encoder), the same latent space (`latents_mean` / `latents_std` equal) |
| audio VAE, tokenizer, processor | | byte-identical to H3's (§4) |
| schedule (`fastvideo_inference.json`) | rungs 999..125, shifts 10 / 3, VSA 0.8, tile 64 | **the same** (the file differs only in `checkpoint_step`, `training_parent_run_id`) |
| text encoder | Qwen3-VL-32B bf16, `hidden_states[50]` | the release ships it NVFP4 and truncated to 50 layers; we read h3-base's bf16 / FP8 (§3) |

Everything else (dims 5376 / 56 x 128 / ffn 14336, in 24, audio 32, text 5120, 2 refiner layers, RoPE, packing,
VSA-H3, the gate `to_gate_compress`, the scheduler, the 768p / 480p canvas and 5-15 s) is FastH3 V2's.

### 1.1 `kept_block_indices` is metadata

`transformer/config.json` carries `num_layers: 42`, `source_num_layers: 50` and `kept_block_indices`. Upstream reads
none of the last two: `MiniMaxH3ArchConfig` (`F/dcfg`) has no such fields, `kept_block_indices` appears nowhere in
FastVideo's model, pipeline, VSA or attention code, and the stored blocks are renumbered `transformer_blocks.0..41`
(the safetensors index). Nothing in a block depends on its index: RoPE is a function of the packed positions, VSA-H3
uses one sparsity and tile for every layer (no dense-layer overrides in the recipe), and the AdaLN table is per
block by storage order. So the port loads blocks `0..num_layers` in order and keeps the source indices only to
check the config (`H3TransformerConfig::with_pruning`: 42 strictly increasing indices below 50) and log them.

The index-dependent features of this repo are not used by the Trim recipe: the Sol-Attn policies (blocks 0-1
dense, 2-49 sparse) are Sol-H3 recipes, and a VSA schedule's `dense_layers` (techniques) would name the Trim's
own 0..41. Neither is set by `trim-8step`.

### 1.2 The factorized AdaLN (`F/dit` `_get_parameter_dtype`, `forward`, `MiniMaxH3AdaLayerNormModulation`)

```
temb = time_embedder(time_proj(t))                               # float32 (fp32 island), as V2
r    = adaln_basis(fp16(silu(temb)))                             # [rows, 16], float16
y_b  = adaln_proj_b.linear(r)                                    # [rows, 96768], float16, NO silu (apply_silu = False)
shift_msa .. gate_mlp = y_b.to(bf16).view(-1, 6 * 5376).chunk(6) # the block casts to hidden_states.dtype
out  = norm_out.linear(r)                                        # [rows, 10752], float16, no silu
```

`_get_parameter_dtype` pins `adaln_basis`, every `.adaln_proj.` and `norm_out.linear.` to float16 ("BF16 is ~1.7x
worse there"); the checkpoint stores them in F32 and the load casts. As in V2 the whole table is a constant of the
checkpoint's ladder, so `AdaLnTable` precomputes it at load (`crates/fastvideo-cudarc/src/h3/transformer.rs`
`FactorizedAdaLn`): weights, inputs and every output rounded to float16 as `F.linear` in half precision does (f64
sums, one rounding per output), on the host (about 1.1 G multiply-adds for 42 blocks, ~1 s). The block path's
existing bf16 rounding of each table value is then exactly the reference's `.to(hidden_states.dtype)`. The memoized
table file (`--adaln-cache`) is keyed by the rank too.

One deviation, measured in §6: `norm_out`'s `1 + scale` is rounded to bf16 under bf16 activations (V2's rule, where
the reference's linear is bf16); the Trim reference keeps it in float16 and multiplies in float32.

### 1.3 The lightweight VAE

`vae/config.json` is H3's `AutoencoderKLMiniMaxH3` with `decoder_num_layers: 26`; every key, shape and the latent
statistics are H3's (checked against `crates/fastvideo-cudarc/src/h3/manifests/vae.json`), stored F16. FastVideo
loads it with `vae_precision = fp32` (`F/pcfg`). The port reads `vae/config.json`
(`h3::pipeline::video_vae_config`: `decoder_num_layers` from the file, every other field checked against H3's) and
loads the F16 weights through the same decoder. The release's `vae/minimax_h3_video_vae_int8_convrot.safetensors`
(Kijai's INT8, ComfyUI layout: fused `to_qkv`, `w1`/`w2`, per-row scales) is published but not read.

## 2. Code

| Where | What |
|---|---|
| `crates/fastvideo-models/src/h3/config.rs` | `H3TransformerConfig::{adaln_rank, source_num_layers, kept_block_indices}`, `fasth3_trim_8step()`, `with_pruning()` (checks), `adaln_in_dim()`; `H3VideoVaeConfig::fasth3_trim_8step()` (26 blocks); recipe names `trim-8step` / `fasth3-trim-8step` / `fasth3-trim` = the V2 contract; `FASTH3_TRIM_KEPT_BLOCKS` |
| `crates/fastvideo-cudarc/src/h3/transformer.rs` | `FactorizedAdaLn` (§1.2) in `AdaLnTable::precompute_with`; the AdaLN cache keyed by the rank |
| `crates/fastvideo-cudarc/src/h3/pipeline.rs` | `transformer_config()` / `video_vae_config()` read `transformer/config.json` (or `transformer_ref/`) and `vae/config.json` (strict: any other dim than H3's is an error); `audio_vae_dir()`: a snapshot without `audio_vae/` decodes with the one beside the text encoder (`text_root`, h3-base) |
| `crates/fastvideo-engine-service/src/cuda/caps.rs` | catalog `fasth3-trim-8step` (served `h3-trim`; `recipe = "trim-8step"` / `"h3-trim"`), weights `fasth3-trim-8step` + `text_weights` h3-base, VSA (not dense), T2V only, untiered |
| `configs/serve/runpod-h3-trim.toml` | the worker config (openai_videos, minimax, fal, native) |
| `control/` | preset `fasth3-trim` (weights `fasth3-trim-8step` + `h3-base`, prequant `h3-base/text_encoder_fp8`), `RECIPE_TREES["h3-trim"]`, catalog / serverless recipe tables, Brev tree size |
| `scripts/gpu/` | `weights-manifest.tsv` / `weights-revisions.tsv` / `weights-sha256.tsv` / `weights-r2.tsv` rows, `verify-weights.sh` cells `fasth3-trim-8step` and `fasth3-trim-8step-nvfp4`, `r2-publish-hub.sh` (Hub -> R2 from a CPU pod), `h3-trim-validate.sh` + `h3-trim-pod.sh` (§6), `upstream/av_psnr.py` |

Tests: `fastvideo-models` `h3::config` (the Trim config is H3's with 42 blocks and rank 16; the pruning fields are
checked; the Trim VAE is H3's with 26 blocks; the recipe is the V2 contract), `fastvideo-cudarc` `h3::transformer`
(the factorized table equals `adaln_proj(adaln_basis(fp16(silu(temb))))` in float16 for every block, step and
modality, and `norm_out`'s; float16 rounding; the cache), `fastvideo-engine-service` `cuda::caps::trim_catalog_entry`,
`control` preset / recipe-tree / catalog tests.

## 3. Text encoder

The release's `text_encoder/` is an NVFP4 Qwen3-VL (`quantization_config.quant_method: nvfp4`, `producer:
convert_minimax_h3_text_encoder_nvfp4.py`, `kept_language_layers: 50`) with `num_hidden_layers_override: 50`. H3
reads `hidden_states[50]` (`F/qwencfg`: `output_hidden_state_index = 50`, `num_hidden_layers_override = 50`), the
residual stream after decoder layer 49: exactly the layers our encoder runs (`H3TextEncoderConfig::layers_to_run`,
docs/ports/h3.md §b). So the Trim's encoder computes the same function as h3-base's up to its NVFP4 quantization,
and the port uses h3-base's (bf16 streamed, or the resident FP8 tree `h3-base/text_encoder_fp8`); it was not
downloaded. The same tokenizer (byte-identical) feeds it.

## 4. Weights (R2 store, bucket `fv-weights`)

Published 2026-10-13 with `scripts/gpu/r2-publish-hub.sh` (one CPU pod streaming Hub -> R2, every file checked
against the Hub's LFS SHA-256 or git blob SHA-1 and `weights-sha256.tsv` before its upload completed), then
`verify --deep --record` (every byte re-read: part and file SHA-256 = manifest).

| Tree (`weights/<dest>/<rev>/`) | Source | Files | Bytes | Manifest SHA-256 |
|---|---|---|---|---|
| `fasth3-trim-8step` @ `9edcb65…` | `FastVideo/FastVideo-FastH3-Trim-8-Step`: `transformer/*` (8 shards), `vae/*` (F16 decoder + INT8 convrot), `scheduler/`, `audio_scheduler/`, `fastvideo_inference.json`, `metadata.json`, `model_index.json`, `README.md` | 19 | 43,875,… (43.88 GB) | MANIFEST_TRIM |
| `fasth3-trim-8step-nvfp4` @ `f66c13d…` | `FastVideo/FastVideo-FastH3-Trim-Comfy`: `diffusion_models/fastvideo_fasth3_trim_8step_nvfp4.safetensors` (11.89 GB, LFS `748b2bcc…`), `README.md` | 2 | 11.89 GB | MANIFEST_NVFP4 |

**Reused, not duplicated** (SHA-256 / git blob compared with the Hub listings of h3-base `42ed227` and FastH3 V2
`3da2ddf`):

| Component | Trim | h3-base / V2 | Verdict |
|---|---|---|---|
| `audio_vae/diffusion_pytorch_model.safetensors` | LFS `52c59e67ba8de547…` (605,429,340 B) | same | identical: read from `h3-base/audio_vae` |
| `audio_vae/config.json` | blob `15df6fe0…` | same | identical |
| `tokenizer/*`, `processor/*` | blobs `c6cc1014…`, `20024bfe…`, `204d76f7…`, `4783fe10…`, … | same | identical: `h3-base/tokenizer` |
| `scheduler/`, `audio_scheduler/` configs | blobs `9caf98ef…`, `9403a37b…` | = V2's (shift 10 / 3); h3-base's are 12 / 3 | kept in the Trim tree (tiny) |
| `vae/` | F16, 26 decoder blocks, LFS `73fc5a61…` | F32, 36 blocks, 3 shards | **different**: published |
| `transformer/` | 42 blocks, rank 16 | 50 blocks | different: published |
| `text_encoder/` | NVFP4 | bf16 | not downloaded (owner's scope); h3-base's is used |

`processor/` (11 MB, needed by FastVideo's pipeline but not by ours) is in neither tree; the validation pod fetches
it from the Hub at the pinned revision. The `fasth3-trim` preset fetches `fasth3-trim-8step` + `h3-base` (+ the FP8
encoder): h3-base's 70 GB transformer comes along unused because R2 trees are fetched whole (a slim text-encoder
tree would save it; not done). Not on the 500 GB EU volume (owner decision 2026-10-13: H3 + LTX trees only, 462 GB
used), so a Runpod pool of the preset defaults to `weights_source r2`.

Records: `scripts/gpu/weights-manifest.tsv`, `weights-revisions.tsv`, `weights-sha256.tsv` (the 11 LFS hashes),
`weights-r2.tsv`, `verify-weights.sh fasth3-trim-8step | fasth3-trim-8step-nvfp4 | sha:fasth3-trim-8step`.

## 5. NVFP4 (`fastvideo_fasth3_trim_8step_nvfp4.safetensors`)

Published to R2 (`fasth3-trim-8step-nvfp4`), **not loaded**. The file is a ComfyUI single file, not the diffusers
layout: its own key names (`blocks.N.attn.qkv_proj`, `mlp.fc1/fc2`, `final_layer.*`, `video_patch_proj`,
`condition_proj`, a precomputed `adaln_t_table [1025, 16]`), fused QKV, and comfy-kitchen NVFP4 linears
(`weight` U8 `[out, in/2]`, `weight_scale` F8_E4M3 `[out, in/16]`, a scalar `weight_scale_2`, `comfy_quant`
metadata) for attention, FFN and the VSA gate. The repo's NVFP4 path (`wan/nvfp4_linear.rs`, cuBLASLt `VEC16_UE4M3`,
TE `static_6`) quantizes bf16 weights at load into cuBLASLt's operand layout; reading this file needs (a) a key map
and QKV split, (b) the comfy scale layout mapped onto cuBLASLt's 128x4 swizzle (or re-swizzled) and `weight_scale_2`
onto the operand's global scale, (c) the nibble order checked, and (d) a GPU parity run against the bf16 Trim.
None of that fit this change's GPU budget. The cheaper route on sm_120 today is the bf16 checkpoint with the
existing load-time quantization (`FASTVIDEO_NVFP4`, the LTX draft's path), which is untested on H3 Trim.

## 6. GPU validation (RTX PRO 6000, 2026-10-13)

RESULTS_PLACEHOLDER

## 7. What is untested

UNTESTED_PLACEHOLDER
