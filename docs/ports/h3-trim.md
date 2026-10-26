# FastH3 Trim 8-step

Target: [`FastVideo/FastVideo-FastH3-Trim-8-Step`](https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step)
at `9edcb6511d5ff8dd1579e8eec8afee77e098b8d8` (the official release only; no community finetunes or mirrors),
served beside FastH3 as `fasth3-trim-8step` (alias `h3-trim`). Owner request and download approval: 2026-10-13.

It is FastH3 V2 (docs/ports/h3.md) with 8 of the 50 transformer blocks removed, each block's AdaLN timestep
projection replaced by a shared rank-16 basis, retrained with 8-step DMD2 (checkpoint 300), and a lighter video
VAE. The model card: faster and smaller than FastH3 V2, with some quality cost.

Status (2026-10-13): **ported and in the serve catalog; validated on an RTX PRO 6000 against upstream FastVideo**
(AdaLN table to 3e-5, step-1 latents 3.8e-4, the same scene and motion; §6.1) and **1.21x faster denoise, 1.23x end to
end than FastH3 V2** at 480p (§6.2). NVFP4 (2026-10-14): loaded as published and served opt-in as `fasth3-trim-nvfp4`; 1.18x faster
denoise than the MXFP8 default, 9.3 GiB of DiT, but it fails the quality gate (§5).

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
On sm_100+ its block linears run as MXFP8 by default (`FASTVIDEO_H3_VAE_MXFP8`, since 2026-10-14; `=0` for bf16):
−22 % video VAE time at 480p and 768p, +1.7 GiB resident, quality gate passed against the bf16 decode (LPIPS 0.001,
PSNR 50 dB; docs/perf/h3-vae-decode-speed.md §6).

## 2. Code

| Where | What |
|---|---|
| `crates/fastvideo-models/src/h3/config.rs` | `H3TransformerConfig::{adaln_rank, source_num_layers, kept_block_indices}`, `fasth3_trim_8step()`, `with_pruning()` (checks), `adaln_in_dim()`; `H3VideoVaeConfig::fasth3_trim_8step()` (26 blocks); recipe names `trim-8step` / `fasth3-trim-8step` / `fasth3-trim` = the V2 contract; `FASTH3_TRIM_KEPT_BLOCKS` |
| `crates/fastvideo-cudarc/src/h3/transformer.rs` | `FactorizedAdaLn` (§1.2) in `AdaLnTable::precompute_with`; the AdaLN cache keyed by the rank |
| `crates/fastvideo-cudarc/src/h3/pipeline.rs` | `transformer_config()` / `video_vae_config()` read `transformer/config.json` (or `transformer_ref/`) and `vae/config.json` (strict: any other dim than H3's is an error); `audio_vae_dir()`: a snapshot without `audio_vae/` decodes with the one beside the text encoder (`text_root`, h3-base); `vae_files()`: only the `diffusion_pytorch_model*` shards of `vae/` (the Trim's also holds the ComfyUI INT8 file) |
| `crates/fastvideo-cudarc/src/wan/quant.rs` | MXFP8 keeps the last three blocks bf16 on a shorter stack too (`mxfp8_last_block`: 46 of 50, 38 of the Trim's 42 = source block 46) |
| `crates/fastvideo-cudarc/src/h3/manifests/trim_{transformer,vae}.json`, `fixtures/*_config.json` | the Trim's safetensors key manifests (`scripts/gpu/h3_manifests.py --prefix trim_`) and the Hub's config files, for the tests |
| `crates/fastvideo-engine-service/src/cuda/caps.rs` | catalog `fasth3-trim-8step` (served `h3-trim`; `recipe = "trim-8step"` / `"h3-trim"`), weights `fasth3-trim-8step` + `text_weights` h3-base, VSA (not dense), T2V only, untiered |
| `configs/serve/runpod-h3-trim.toml` | the worker config (openai_videos, minimax, fal, native) |
| `control/` | preset `fasth3-trim` (weights `fasth3-trim-8step` + `h3-base`, prequant `h3-base/text_encoder_fp8`), `RECIPE_TREES["h3-trim"]`, catalog / serverless recipe tables, Brev tree size |
| `scripts/gpu/` | `weights-manifest.tsv` / `weights-revisions.tsv` / `weights-sha256.tsv` / `weights-r2.tsv` rows, `verify-weights.sh` cells `fasth3-trim-8step` and `fasth3-trim-8step-nvfp4`, `r2-publish-hub.sh` (Hub -> R2 from a CPU pod), `experiments/h3-trim-validate.sh` + `experiments/h3-trim-pod.sh` (§6), `upstream/av_psnr.py` |

Tests (§6.4 has the counts): `fastvideo-cudarc` `h3::manifest_tests` (every Trim DiT and VAE key requested with its
published shape, none left over, the dtypes), `h3::pipeline` (the Hub's own config files parse into the port's
configs, foreign ones are refused, the VAE file filter, the audio-VAE fallback), `wan::quant` (the Trim's MXFP8
blocks); `fastvideo-models` `h3::config` (the Trim config is H3's with 42 blocks and rank 16; the pruning fields are
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
| `fasth3-trim-8step` @ `9edcb65…` | `FastVideo/FastVideo-FastH3-Trim-8-Step`: `transformer/*` (8 shards), `vae/*` (F16 decoder + INT8 convrot), `scheduler/`, `audio_scheduler/`, `fastvideo_inference.json`, `metadata.json`, `model_index.json`, `README.md` | 19 | 43,878,908,891 | `2ccd02ce…f4f19339` |
| `fasth3-trim-8step-nvfp4` @ `f66c13d…` | `FastVideo/FastVideo-FastH3-Trim-Comfy`: `diffusion_models/fastvideo_fasth3_trim_8step_nvfp4.safetensors` (11.89 GB, LFS `748b2bcc…`), `README.md` | 2 | 11,894,625,413 | `3e536acd…fa5a44fe` |

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

Status (2026-10-14): **loaded as published, served opt-in as `fasth3-trim-nvfp4` (served `h3-trim-nvfp4`, recipe
`trim-8step-nvfp4`), validated on one RTX PRO 6000: 1.18-1.19x faster denoise than the served MXFP8 Trim and 1.49-1.55x
faster than bf16, 9.3 GiB of resident DiT instead of 20.3 / 33.2 GiB; it FAILS the quality gate** (§5.4). The default
Trim (`fasth3-trim-8step`, preset `fasth3-trim`) is unchanged: bf16 checkpoint, MXFP8 linears.

### 5.1 The file

A ComfyUI single file (`FastVideo/FastVideo-FastH3-Trim-Comfy` @ `f66c13d`, R2 tree `fasth3-trim-8step-nvfp4`), not
the diffusers layout. Each block's five big linears are comfy-kitchen NVFP4 tensors (`comfy_quant` =
`{"format": "nvfp4"}`); everything else is bf16 / F32 under ComfyUI names (`video_patch_proj`, `condition_proj`,
`token_refiner.*`, `final_layer.*`, a precomputed `adaln_t_table [1025, 16]`, the per-block `adaln_proj` F32).

| key (`blocks.{i}.`) | dtype, shape | ours (`transformer_blocks.{i}.`) |
|---|---|---|
| `attn.qkv_proj.weight` | U8 `[21504, 2688]` | `attn.to_q` / `to_k` / `to_v`, in that order |
| `attn.to_gate_compress.weight` | U8 `[7168, 2688]` | `attn.to_gate_compress` (own global scale) |
| `attn.out_proj.weight` | U8 `[5376, 3584]` | `attn.to_out.0` |
| `mlp.fc1.weight` | U8 `[28672, 2688]` | `ff.net.0.proj`, **halves swapped** (file rows 14336.. are ours 0..) |
| `mlp.fc2.weight` | U8 `[5376, 7168]` | `ff.net.2` |
| `<l>.weight_scale` | F8_E4M3 `[out, in/16]` | per-16 block scales |
| `<l>.weight_scale_2` | F32 scalar | the tensor's global decode factor (about `amax / (6 · 448)`) |
| `norm1`, `norm2`, `attn.q_norm`, `attn.k_norm` | BF16 | `norm1`, `norm2`, `attn.norm_q`, `attn.norm_k` (bit-identical to the bf16 tree) |

Layout, measured on the Hub bytes against the bf16 Trim (block 0 `to_out.0` whole; 128-row tiles of blocks 0, 20 and
41 for the other four, numpy over HTTP ranges of the Hub files, 2026-10-14): `w = e2m1(code) · e4m3(scale) · weight_scale_2` with
**the even element in the high nibble** and the scales **already in the 128x4 tiled (`to_blocked`) layout** cuBLASLt
reads gives rel-L2 0.093-0.101 against the bf16 weight, the same as our own TE `static_6` quantization of it (0.095).
Low-nibble-first or a row-major scale reading gives 0.72-1.45; `fc1` read in the diffusers order gives 1.2-1.7 and
0.094 with its halves swapped; `qkv_proj` rows are q, k, v (0.093-0.094 in each third).

### 5.2 Code

| Where | What |
|---|---|
| `crates/fastvideo-cudarc/src/h3/nvfp4.rs` | the file: `resolve_file` / `sibling_tree`, `PublishedLinear::read` (dtype, shape and `comfy_quant` checks, nibble swap to low-first), `swap_row_halves` (fc1), `Nvfp4Dit::open` (every block's keys and shapes; the block norms bit-equal to the bf16 tree's, so the two trees are the same model), `device_linears` (a block's `qkvg` = `[qkv; gate]` stack, `to_out`, `ff_in`, `ff_out`); `require_tensor_cores` |
| `crates/fastvideo-cudarc/src/wan/nvfp4_linear.rs` | `Nvfp4Operand::from_host_layout` (codes and scales as published, the global factor through the operand's amax: `amax = weight_scale_2 · 6 · 448`, which `nvfp4_alpha` turns back into `weight_scale_2`); `gemm_into` (a GEMM into columns of a wider output, `ld` in the plan key); `Nvfp4Stack` (one activation quantization, one GEMM per part, each with its own global scale) |
| `crates/fastvideo-cudarc/src/wan/nn.rs` | `Linear::from_nvfp4`: W4A4 forward (activation quantized per call with the TE `static_6` rule, a dynamic per-tensor scale as comfy-kitchen without an `input_scale`), the quantized-linear epilogue; never quantized again, snapshotted or LoRA-merged |
| `crates/fastvideo-cudarc/src/h3/transformer.rs` | `H3Transformer::load_with_nvfp4` / `Block::load_nvfp4`: `QuantMode::Nvfp4` (bf16 activations; `FASTVIDEO_H3_QUANT` not read for the blocks), resident only, no adapter |
| `crates/fastvideo-cudarc/src/h3/pipeline.rs` | `H3PipelineOptions::nvfp4_dit` / `nvfp4_dit_file`: an NVFP4 recipe reads `../fasth3-trim-8step-nvfp4` beside the snapshot; the check against the bf16 tree; the snapshot's block linears are not prefetched or read; no pre-quantized DiT tree; a streamed `dit_offload` is refused |
| `crates/fastvideo-models/src/h3/config.rs` | `is_trim_nvfp4_recipe` (`trim-8step-nvfp4`, `fasth3-trim-nvfp4`, `h3-trim-nvfp4`, `fasth3-trim-8step-nvfp4`) = the V2 contract |
| `crates/fastvideo-engine-service/src/cuda/caps.rs`, `configs/serve/runpod-h3-trim-nvfp4.toml` | catalog `fasth3-trim-nvfp4` (untiered, T2V only, served `h3-trim-nvfp4`); the worker config |
| `control/` | preset `fasth3-trim-nvfp4` (weights `fasth3-trim-8step`, `fasth3-trim-8step-nvfp4`, `h3-base`; prequant `h3-base/text_encoder_fp8`; R2 by default), `RECIPE_TREES["h3-trim-nvfp4"]`, catalog / serverless tables, Brev tree size, `scripts/serve/variants.sh` |
| `fv-gpucheck` | `h3 gen --nvfp4-dit`; `kernels --groups nvfp4_linear` check `nvfp4_published_stack_matches_host` |
| `scripts/gpu/experiments/h3-trim-nvfp4-{validate,pod}.sh`, `h3-trim-nvfp4-plan.py`, `h3-trim-nvfp4-summary.py` (since D1c `fv-lab report summary --kind h3-trim-nvfp4`) | §5.4 |

**Hardware.** The GEMM is cuBLASLt `VEC16_UE4M3`, which exists on sm_100 / sm_120 only. On any other GPU (or a CPU
build) the load fails with "the published NVFP4 DiT runs only on NVFP4 tensor cores ... No bf16 fallback"; the
published weight has no bf16 form to fall back to, and the catalog entry is opt-in.

**Not read from the file**: the patch projections, text refiner, AdaLN (`adaln_proj`, `adaln_t_table`), output head:
they come from the bf16 Trim tree, which the load needs anyway (`transformer/config.json`, the schedule, the VAE). So a
pool fetches both trees (44 + 12 GB) plus h3-base. The AdaLN table is computed as for the bf16 Trim (§1.2).

### 5.3 Tests

`cargo test -p fastvideo-cudarc --lib -- h3::nvfp4` (+ `--features cuda`): the blocked-scale offset against a tile
reshape and cuBLASLt's `swizzle_scales_host`; a file written as the release writes it (our TE quantization,
high-nibble-first, blocked scales, `weight_scale_2`) dequantizes to the quantized weight, and its swapped halves are
the other half's rows exactly; other formats / dtypes / shapes refused; the Trim's five linear shapes; recipe names.
`fastvideo-models h3::config` (the NVFP4 names are the V2 contract), `fastvideo-engine-service`
`cuda::caps::trim_nvfp4_catalog_entry` (66 passed), `fastvideo-serve --test e2e shipped_configs_parse`. On the GPU:
`kernels --groups nvfp4_linear` `nvfp4_published_stack_matches_host` (two stacked parts with global scales 100x apart,
m = 300) PASS, rel-L2 1.66e-3 against exact host math (limit 4e-3). `control`: `gen-configs`, `tsc` (both projects),
`vitest` 466 passed, integration 38 steps passed.

### 5.4 GPU validation (RTX PRO 6000 Blackwell, EU-RO-1, 2026-10-14)

`scripts/gpu/experiments/h3-trim-nvfp4-validate.sh up ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-b407063` (this branch's runtime
image; the R2 plan from the read-only key passed in the pod's env, nothing staged in R2), EU volume for h3-base (read
only), the two Trim trees from R2 (54 GB in 51 s). The bf16 text encoder streamed once per prompt, then from the
conditioning cache, so every arm runs on bit-identical text (`text_hidden` / `text_refined` rel-L2 0). Arms: **bf16**
(`FASTVIDEO_H3_QUANT=off`), **MXFP8** (the served default: blocks 2-38), **NVFP4** (`--h3-recipe trim-8step-nvfp4`:
all 42 blocks). Results: `artifacts/ports/h3-trim/nvfp4/` (`summary.md` from `scripts/gpu/experiments/h3-trim-nvfp4-summary.py`, now `fv-lab report summary --kind h3-trim-nvfp4 DIR --json OUT`).

There is no upstream NVFP4 reference to compare with: FastVideo does not load this ComfyUI file (its pipeline reads the
diffusers bf16 checkpoint; the file targets ComfyUI's comfy-kitchen kernels), and running ComfyUI was out of scope. So
parity is against our bf16 Trim, which §6.1 tied to upstream FastVideo (step-1 latents 3.8e-4).

**Parity** (832 x 480, 124 frames, seed 1024, the matrix prompt; rel-L2 against our bf16 Trim):

| tensor | MXFP8 (served) | NVFP4 |
|---|---:|---:|
| step-1 block 0 / 20 / 41 output | 0 / 4.7e-2 / 8.3e-2 | 2.6e-2 / 0.15 / 0.20 |
| step-1 velocity | 5.9e-2 | 0.18 |
| **video latents after step 1** / 2 / 4 / 8 | **1.1e-3** / 4.8e-3 / 2.7e-2 / 0.48 | **3.4e-3** / 1.1e-2 / 4.6e-2 / 0.70 |
| audio latents after step 1 / 8 | 2.1e-3 / 0.42 | 6.8e-3 / 0.67 |
| decoded clip PSNR (pooled) / LPIPS (alex, all frames) | 18.2 dB / 0.263 | **13.1 dB / 0.443** |

NVFP4 starts about 3x further from bf16 than MXFP8 does at every step (W4A4 in all 42 blocks, against W8A8 with
E8M0 32-blocks in 37 of them), and the few-step DMD sampler amplifies it to a different sample: the same scene,
person and framing, different detail and motion (`frame-768p-*-bf16-mxfp8-nvfp4.jpg`, `clips/parity-480p-*.mp4`).

**Speed and memory** (5 s = 124 frames; one process per cell, `--warm`, medians over three prompts of
`prompts-hd5.json`: talking-head, spark-mountain-lake, ltx-newsbroadcast; peak = nvidia-smi device memory of the
measured runs, the conditioning cached; DiT = the ledger's resident `dit_nonlinear`):

| canvas | arm | denoise s | s / forward | end to end s | peak GiB | resident DiT GiB |
|---|---|---:|---:|---:|---:|---:|
| 832 x 480 | bf16 | 16.86 | 2.11 | 19.2 ¹ | 43.9 | 33.2 |
| 832 x 480 | MXFP8 (served) | 13.01 | 1.63 | 15.38 | 31.2 | 20.3 |
| 832 x 480 | **NVFP4** | **10.90** | **1.36** | **13.26** | **20.2** | **9.3** |
| 1344 x 768 | bf16 | 48.31 | 6.04 | 53.41 | 51.2 | 33.2 |
| 1344 x 768 | MXFP8 (served) | 38.15 | 4.77 | 43.26 | 38.5 | 20.3 |
| 1344 x 768 | **NVFP4** | **32.42** | **4.05** | **37.54** | **27.4** | **9.3** |

¹ The talking-head prompt (its text cached); the cell's median total includes streaming the encoder for the two
prompts it saw first.

NVFP4 against MXFP8: denoise 1.19x (480p) / 1.18x (768p), end to end 1.16x / 1.15x, -11 GiB peak; against bf16:
1.55x / 1.49x denoise, -24 GiB. The VAE decode is unchanged (2.06 / 4.76 s). The activation is quantized by its own
two kernels before every NVFP4 GEMM (an amax pass, then codes + scales), not fused into the norm / SwiGLU producers
as MXFP8's is, so the 4x smaller weights give less than the GEMM ratio.

**Quality gate** (`fv-gpucheck gate`, `scripts/gpu/gate-policy.toml`, kind `lossy`, `compare-clips` with LPIPS per
prompt; the policy's hard checks are frame count, sharpness ratio 0.95-1.08, temporal-jitter ratio 0.85-1.20, LPIPS
available; PSNR / LPIPS limits are telemetry):

| baseline -> candidate | verdict | denoise speed-up | failed hard checks |
|---|---|---:|---|
| bf16 -> MXFP8, 480p | pass | 1.30 | - |
| bf16 -> MXFP8, 768p | pass | 1.27 | - |
| **MXFP8 -> NVFP4, 480p** | **FAIL** | 1.19 | sharpness 1.34 (lake), jitter 1.23 (news) |
| **MXFP8 -> NVFP4, 768p** | **FAIL** | 1.18 | sharpness 0.91 (news) and 1.24 (lake), jitter 2.08 (news) and 1.24 (talking head) |
| bf16 -> NVFP4, 480p | FAIL | 1.55 | sharpness 1.43 (lake) |
| bf16 -> NVFP4, 768p | FAIL | 1.49 | sharpness 0.90 / 1.21, jitter 2.08 / 1.29 |

Per prompt, NVFP4 against MXFP8: PSNR 10.5-15.3 dB, LPIPS 0.44-0.70; MXFP8 against bf16 on the same prompts: PSNR
12.4-19.0 dB, LPIPS 0.27-0.56. **Verdict: NVFP4 fails the promotion gate at both resolutions** (performance passes;
quantitative quality fails on sharpness and temporal jitter in two or three of three prompts). The clips are coherent
and on the prompt, but they move further from the bf16 / MXFP8 samples than MXFP8 moves from bf16, with visibly more
motion blur on the newsreader (768p jitter 2.08x). It stays opt-in; the bf16 / MXFP8 Trim stays the default.

Cost: two pods (the first stopped at its first step: the resident FP8 text encoder checks its bf16 source shards, which
an R2-mode pod did not have; deleted after 5 min), the second 28 min; about $1.38 of GPU time at $2.49/hr. Both deleted
and checked gone (`GET /pods/<id>`: not found).

## 6. GPU validation (RTX PRO 6000, 2026-10-13)

Scripts: `scripts/gpu/experiments/h3-trim-validate.sh` (launcher: stock / price check, the R2 plan, the staged bundle, a
wall-clock backstop) and `scripts/gpu/experiments/h3-trim-pod.sh` (on the pod). Image `fastvideo-rs-runtime:sha-8f20822` with this
branch's `fv-gpucheck` (built on the build pod); RTX PRO 6000 Blackwell Workstation Edition (driver 580.178.04, CUDA
13.0), EU-RO-1, the EU volume for h3-base, the Trim and V2 trees from R2 (fetched at 0.5-1 GB/s).

Request: the matrix prompt ("A man in his thirties talking to the camera…", with `<d>` dialogue), seed 1024,
**832 x 480, 124 frames (5.2 s)**, the trained recipe: 8 DMD rungs, shifts 10 / 3, VSA-H3 0.8 at tile 64.

### 6.1 Parity against upstream FastVideo

Reference: FastVideo `main` @ `d5253fd` (the first with `adaln_rank`), `examples/inference/basic/basic_fasth3_8step.py`
through `scripts/gpu/upstream/oracle_fastvideo.py`, strict eager profile, Triton VSA, no compile, on a Trim snapshot
overlay (the Trim `transformer/`, `vae/`, schedulers and `fastvideo_inference.json`; h3-base's bf16 `text_encoder/`,
tokenizer and audio VAE; `processor/` from the Hub). It logged "FastH3 checkpoint schedule: 8 transformer forwards,
DMD rungs=[999, …, 125], video/audio shifts=10.0/3.0" and selected `VIDEO_SPARSE_ATTN_H3`. The oracle hooks
(`oracle_dump.py`) dumped noise, text, per-step latents and velocities, every step-1 block output (every 64th row)
and the inside of blocks 0, 1, 20 and 41. Ours ran with the reference's noise and text injected
(`FASTVIDEO_INJECT_DIR`), so every remaining difference is the denoiser's. `fv-gpucheck compare-dumps`, rel-L2 of
ours against the reference (`artifacts/ports/h3-trim/compare-dumps-{bf16,mxfp8}.txt`):

| tensor | ours, bf16 linears (`FASTVIDEO_H3_QUANT=off`, as the reference) | ours, served default (MXFP8 blocks 2-38) | FastH3 V2, 768p (docs/oracle.md, for scale) |
|---|---:|---:|---:|
| sigmas, timesteps, injected noise | 0 (bitwise) | 0 | 0 |
| text hidden states (our encoder, before injection) | 1.29e-2 (bf16 streamed) | 1.34e-2 (FP8 resident) | 1.29e-2 |
| refined text | 8.1e-3 | 8.1e-3 | 8.2e-3 |
| packed DiT input | 3.2e-3 | 3.2e-3 | 3.2e-3 |
| **AdaLN rows, blocks 0 / 1 / 20 / 41** (the factorized table) | **3.4e-5 / 2.7e-5 / 3.6e-5 / 3.9e-5** | same | 3e-5 to 1e-4 |
| step-1 block 0 / 1 / 10 / 20 / 30 / 41 | 1.4e-3 / 3.2e-3 / 4.3e-3 / 1.6e-2 / 9.2e-2 / 4.3e-2 | 1.4e-3 / 3.2e-3 / 6.9e-3 / 4.7e-2 / 1.8e-1 / 7.5e-2 | blocks 0 / 12 / 24 / 49: 4.8e-3 / 5.3e-3 / 9.0e-3 / 0.25 |
| video latents after steps 1 / 2 / 3 / 4 | 3.8e-4 / 3.0e-3 / 1.1e-2 / 2.4e-2 | 8.8e-4 / 5.2e-3 / 1.7e-2 / 3.6e-2 | 3.4e-3 / 1.0e-2 / 2.0e-2 / 3.6e-2 |
| video latents after steps 5 / 6 / 7 / 8 | 4.8e-2 / 9.8e-2 / 0.23 / 0.47 | 6.9e-2 / 0.14 / 0.30 / 0.61 | 6.4e-2 / 0.12 / 0.27 / 0.53 |
| audio latents after steps 1 / 2 / 3 / 8 | 6.4e-4 / 1.6e-3 / 7.3e-3 / 0.77 | 2.5e-3 / 4.2e-3 / 1.5e-2 / 0.82 | |

Reading: the factorized AdaLN table matches the reference to 3e-5 (float16 values rounded to bf16 on both sides; V2's
full-rank table was 3e-5 to 1e-4), the inputs, schedule and first step are at bf16-rounding level, and inside
blocks 0, 1, 20 and 41 no op leaves its input's error level. From there the profile is FastH3 V2's: the per-block
difference grows through the middle of the stack (as V2's does from block ~25), and the few-step DMD sampler with
VSA's top-k tile selection amplifies it step by step until the two finish on different, equally valid samples
(docs/oracle.md "Why the ladder is gated on its first three steps"). The Trim tracks the reference more closely
than V2 did at steps 1-4 (fewer blocks, 480p).

Decoded clips (`scripts/gpu/upstream/av_psnr.py`; video: every frame as rgb24, PSNR pooled over the clip; audio:
stereo at 32 kHz against peak 1.0, best lag 0):

| pair | video PSNR | min frame | audio PSNR |
|---|---:|---:|---:|
| upstream vs ours, bf16, injected | 16.6 dB | 15.6 dB | 15.9 dB |
| upstream vs ours, MXFP8 (served), injected | 14.3 dB | 13.6 dB | 15.5 dB |
| ours bf16 vs ours MXFP8, same inputs | 13.8 dB | | 19.0 dB |

The numbers are those of two diverged few-step samples, not of a broken decoder: the upstream and our injected clips
show the same scene, person, framing, pose and lighting (middle frame side by side:
`artifacts/ports/h3-trim/frame62-upstream-ours-trim-v2.jpg`), with different fine detail. The reference's video and
audio went through H.264 / AAC; ours are compared from the same codecs (our mp4) except the audio, read from our WAV.
For scale, the FastH3 4-step VSA oracle gave 19.2-19.7 dB at 768p (docs/oracle.md), on a sampler with half the
steps over which to diverge.

### 6.2 Timing against FastH3 V2 (same pod, same request)

Ours as served: the default MXFP8 linears on sm_120 (blocks 2-46 of V2, 2-38 of the Trim: the last three stay bf16),
the FP8 text encoder resident (`auto`), VSA-H3 0.8, the official VAEs. One process per model, `--warm`: a cold
generation, then the measured one (text from the conditioning cache in both). Logs:
`artifacts/ports/h3-trim/timing-{trim,v2}.jsonl`.

| 832 x 480, 124 frames, warm | FastH3 Trim | FastH3 V2 | Trim speed-up |
|---|---:|---:|---:|
| denoise (8 forwards) | **20.22 s** (2.48-2.54 s/forward) | 24.40 s (3.00-3.07 s/forward) | **1.21x** |
| video decode (VAE + mp4) | 3.13 s (VAE 3.01 s) | 4.34 s (VAE 4.22 s) | 1.39x |
| audio decode | 0.07 s | 0.07 s | |
| end to end (text cached) | **23.84 s** | 29.22 s | **1.23x** |
| peak device memory (`peak_mib`, FP8 encoder resident) | 55.3 GiB (56 639 MiB) | 60.2 GiB (61 631 MiB) | -4.9 GiB |
| resident DiT (ledger `dit_nonlinear`) | 20.30 GiB | | |
| load (encoder from the volume, DiT and VAE from local disk) | 37.6 s | 13.4 s (encoder in the page cache) | |

42 of 50 blocks predict 1.19x on the DiT; 1.21x is measured. The 26-block VAE decodes in 71% of the 36-block one's
time. Upstream FastVideo on the same request (its first and only request, strict eager, Triton kernels compiled on
first use): denoise 32.1 s, end to end 44.2 s, peak 36.1 GB (`artifacts/ports/h3-trim/upstream-result.json`); not a
like-for-like speed comparison.

Aside, from the parity pod (another RTX PRO 6000 Workstation): the Trim with **bf16** linears ran its forwards 2-8 at
2.14-2.22 s (the first, 10.0 s, includes kernel compilation), faster than the MXFP8 default's 2.48-2.54 s at this
480p size; its VAE took 2.04 s. So at 480p on sm_120 the MXFP8 recipe costs time rather than saving it, for the Trim
and probably for V2 as well; this was not measured on one pod for both models and is left open (§7).

Samples (H.264 + AAC, `artifacts/ports/h3-trim/`): `upstream-fastvideo-trim-480p.mp4` (the reference),
`ours-trim-injected-{bf16,mxfp8}-480p.mp4` (ours on its noise and text), `ours-trim-480p.mp4` and
`ours-fasth3-v2-480p.mp4` (ours, seed 1024, the timed runs). Trim against V2 on the same seed: 10.7 dB video / 13.9 dB
audio (different models, so different samples; both coherent and on the prompt).

### 6.3 Cost and clean-up

Seven GPU pods, all deleted and checked gone (`GET /pods/<id>`: not found), about **$2.9** of GPU time:

| pod | what happened | minutes |
|---|---|---:|
| 1 (EU-RO-1) | upstream parity reference (dump kept); our runs failed on a too-strict VAE config check (fixed: float tolerance) | 12.4 |
| 2 (R2 mode, an IE host) | no container runtime after 12 min (image pull); deleted | 12.0 |
| 3 (EU-RO-1) | the host's GPU was already 95.8 GiB in use (no process of ours); deleted. The pod script now stops on that | 15.0 |
| 4 (EU-RO-1) | the Trim `vae/` also holds the ComfyUI INT8 file (duplicate keys; fixed: `vae_files`) | 3.8 |
| 5 (EU-RO-1) | bf16 parity run; then `av_psnr.py`'s ffmpeg blocked on stdin (fixed: `-nostdin`); deleted | 17.3 |
| 6 (EU-RO-1) | the busy host again, stopped at its first check | 0.8 |
| 7 (EU-RO-1) | MXFP8 parity run and both timings | 6.7 |

Plus the R2 publish CPU pod (12 min at $0.48/hr) and the shared build pod. Staged bundles in `fv-weights-staging/tmp/`
were removed by each `down`.

### 6.4 Tests run (2026-10-13, build pod and this checkout)

| Suite | Result |
|---|---|
| `cargo test -p fastvideo-models -p fastvideo-cudarc --lib -- h3:: wan::quant` | 120 + 97 passed (13 of them new for the Trim) |
| `cargo test -p fastvideo-engine-service --lib` | 64 passed (`cuda::caps::trim_catalog_entry` new) |
| `cargo test -p fastvideo-serve --test e2e shipped_configs_parse` | passed (`runpod-h3-trim.toml` resolves against the catalog) |
| `cargo check -p fastvideo-serve --features cuda,http-client`; `cargo build --release -p fastvideo-gpucheck --features cuda` | clean |
| `control`: `npx tsc --noEmit`, `npx tsc -p ui`, `npx vitest run`, `node test/integration/run.mjs` | clean, clean, 435 passed (0 failed), 36 integration steps passed |
| `python3 -m unittest discover -s scripts/gpu/tests -p 'test_r2*.py'` | 41 passed |

## 7. What is untested

- **NVFP4** (§5): not compared with an upstream NVFP4 run (none loads this file but ComfyUI); measured at 480p
  and 768p, 5 s, three prompts; the served pool (`fasth3-trim-nvfp4`) was not booted on a GPU worker; the activation
  quantization is not fused into the producers (the speed headroom).
- **Image conditioning**: FL2VA / I2V / Ref2VA on the Trim DiT run through the same code but were not tried; the
  catalog offers text-to-video only.
- **768p and 10-15 s**: validated at 832 x 480 x 124 only; the canvas and duration caps are V2's.
- **The served path end to end**: the `fasth3-trim` pool / fv-serve config, its warm-up and the R2 boot (h3-base +
  the Trim tree) were exercised by unit / integration tests only, not on a GPU worker.
- **MXFP8 versus bf16 speed**: the Trim's bf16 forwards were faster than MXFP8 at 480p on another pod (§6.2); a
  same-pod measurement for both models and resolutions would decide the default.
- `norm_out`'s `1 + scale` is bf16-rounded where the reference keeps float16 (§1.2); its effect was not isolated.
- Quality beyond these clips (the model card says Trim trades some quality for speed): no prompt set or LPIPS gate.
- 1080p, TAEH3, TeaCache, Sol-Attn and LoRA paths on the Trim.
- The Trim's own VAE INT8 convrot file, and the release's NVFP4 text encoder.
