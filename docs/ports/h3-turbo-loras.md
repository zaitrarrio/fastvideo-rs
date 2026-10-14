# MiniMax-H3 Turbo LoRAs (larry, lightx2v) and the NCII prompt guard

Owner request 2026-10-13: implement what the
[`MiniMaxAI/MiniMax-H3-Turbo-Lora`](https://huggingface.co/spaces/MiniMaxAI/MiniMax-H3-Turbo-Lora) Space runs
(revision `9a76d679`), plus its optional NCII prompt guard. The Space runs **unquantized bf16 MiniMax-H3** with a
4-step "Turbo" LoRA folded into the weights at load (`h3_lora.py`), 6 sampling steps by default, dense attention
(cuDNN), diffusers' modular MiniMax-H3 pipeline from huggingface/diffusers#14371 @ `665f5782`, and a separate Space
for the Qwen3-VL conditioner.

What this port runs: base MiniMax-H3 (`h3-base`) with the same fold (float32 delta, one bf16 rounding), **then**
MXFP8 linears as every other served H3 recipe (merge first, quantize second), dense attention, our VAEs. Two serve
models, `h3-larry-turbo` and `h3-lightx2v-turbo`, and an optional admission check, `[moderation] ncii`.

GPU validation: §6 (parity), §7 (speed and quality). Results are filled in from the validation pod's reports
(`artifacts/ports/h3-turbo-loras/`).

## 1. The two LoRAs

| | larry | lightx2v |
|---|---|---|
| Repo @ rev | `larryvrh/MiniMax-H3-Turbo-Lora` @ `43a74557` | `lightx2v/Minimax-h3-Turbo` @ `3ec17a32` |
| Licence | Apache-2.0 (base: `Comfy-Org/MiniMax-H3`) | Apache-2.0 (base: `MiniMaxAI/MiniMax-H3`) |
| File (ours) | `minimax_h3_turbo_4step_ema_ckpt850.safetensors` (0.78 GB), the Space's default | `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` (1.38 GB), the newest 4-step 768p file |
| Layout | ComfyUI (reference) module tree: `blocks.N.attn.qkv_proj`, `blocks.N.mlp.fc1/fc2`, `blocks.N.attn.out_proj`, `blocks.N.adaln_proj.linear`, `token_refiner.blocks.N.*`, `final_layer.adaln_proj.linear` | PEFT on the diffusers tree: `transformer_blocks.N.attn.to_q.lora_A.default.weight` and friends (no AdaLN) |
| Rank, scale | 64 for the linears, 16 for the AdaLN; `alpha == rank` (scale 1, `W += B @ A`) | 128, alpha 8 (metadata `alpha: 8` on v1.x): scale 0.0625 |
| Pairs | 259 modules → 363 diffusers targets (QKV split in three) | 312 |
| Trained for | 4-8 steps (card: 4 is the minimum, 6-8 "noticeably better", past 8 over-sharp), shifts 12 / 3, strength 1.0 ("the sharpness dial": >1 against smear, <1 against grain) | FL2VA / T2VA, 4 NFE; v1.x 768p trained at video shift **6** / audio 3, v0.1 at 12 / 3 |

**Which lightx2v file.** ModelTC/Minimax-H3-Turbo "Model specs" lists FL2VA 4-step v0.1 (544p, shifts 12 / 3), 8-step
v1.0, 4-step v1.0 768p and 8-step v1.0 768p (shifts 6 / 3); the Hub repo adds v1.1 and v1.2 768p. The Space folds
**v0.1** (its `H3_LIGHTX_FILE` default). The owner's approval covered about 2.2 GB, so the R2 tree holds larry ckpt850
and lightx2v **v1.2 768p** (2.16 GB); `lightx2v-turbo-v0.1` (shifts 12 / 3) is a recipe that finds the v0.1 file
beside `h3-base` or through the adapter override, and publishing it (1.38 GB) needs the owner. All T2V/FL2V files sit
at the revision the repo already pins for the Ref2VA LoRAs (`3ec17a32`, the repo's head on 2026-10-14), so no newer
revision is needed; the Ref2VA tree (`h3-ref2va`) is unchanged.

**larry v4.** larry's card now recommends `minimax_h3_turbo_v4_step600_ema.safetensors` (better static and
small-motion shots; smear at 4 steps with fast motion). The Space and this port default to ckpt850; v4-600 is the
same layout and loads through the same path (`--adapter <file>` in `fv-gpucheck h3 gen`), not published.

## 2. Key conversion (larry)

A port of the Space's `_larry_targets` (`crates/fastvideo-models/src/h3/turbo_lora.rs` `comfy_targets`,
`map_rows`), applied to `lora_B` only (rows of `B @ A` are rows of `B`), each target keeping its `lora_A`:

| ComfyUI module | Diffusers target(s) | Rows of `lora_B` |
|---|---|---|
| `blocks.N.attn.qkv_proj` | `transformer_blocks.N.attn.to_q`, `to_k`, `to_v` | thirds `[0, 7168)`, `[7168, 14336)`, `[14336, 21504)` (q, k, v; the in-memory reference layout, after its load-time de-interleave) |
| `blocks.N.mlp.fc1` | `transformer_blocks.N.ff.net.0.proj` | `[gate; value]` → `[value; gate]` (diffusers' SwiGLU reads `[value; gate]`) |
| `blocks.N.mlp.fc2` | `transformer_blocks.N.ff.net.2` | all |
| `blocks.N.attn.out_proj` | `transformer_blocks.N.attn.to_out.0` | all |
| `blocks.N.adaln_proj.linear` | `transformer_blocks.N.adaln_proj.linear` | all (identical row layout) |
| `token_refiner.blocks.N.*` | `token_refiner.refiner_blocks.N.*` (same rules) | |
| `final_layer.adaln_proj.linear` | `norm_out.linear` | all |

The same transforms as diffusers' `scripts/convert_minimax_h3_to_diffusers.py` (`split_fused_qkv`, the fc1 half
swap) and FastVideo's `convert_minimax_h3_comfy_lora.py`, which documents the swap as "no error, no shape mismatch,
just a quietly wrong model" when missed. Every target must exist in the base checkpoint with the shape the row map
gives; anything else (an `alpha` tensor, a half pair, an unknown module) is refused at load.

Unit tests (synthetic tensors that mirror the Space's arithmetic): `turbo_lora::tests::names_follow_the_space`,
`row_maps_reproduce_the_converted_delta` (each target's `B' @ A` equals the conversion of the fused `B @ A`),
`layout_detection_and_pairs`; `fastvideo-cudarc` `h3::lora::tests::comfy_fold_matches_the_space` (the whole fuse:
`strength * B @ A` of the fused module, converted, against what the fuse writes for each target, ranks 2 and 1 in one
file).

## 3. The fold and quantization

`H3LoraFuse::open_turbo` (`crates/fastvideo-cudarc/src/h3/lora.rs`): larry through `from_comfy` (format `Comfy`,
`alpha = rank` per module, so the multiplier is the strength for the rank-64 linears and the rank-16 AdaLN alike),
lightx2v through the PEFT planner with alpha 8 (checked against the file's `alpha` metadata when present).
`load_linear` merges each touched weight on the device (`lora_fuse_bf16`: the base bf16 weight, `+= m * B @ A` in
float32, one bf16 rounding; the host path is the same arithmetic) and returns a bf16 linear; the block loader then
quantizes it to MXFP8 (sm_100+; W8A8 below). The AdaLN table is precomputed from the merged `adaln_proj.linear` /
`norm_out.linear` (a fused adapter disables the AdaLN cache). No pre-quantized DiT tree exists for these recipes
(`dit_tree_identity` returns none), so each load quantizes; a tree like h3-turbo's would cut the load.

The Space folds into bf16 and runs bf16 (and AoTI); this port's served form adds MXFP8, the parity run (§6) uses bf16
linears (`FASTVIDEO_H3_QUANT=off`) against the Space and the served MXFP8 against the same reference.

## 4. Recipes and serving

| Recipe | Adapter | Forwards | Grid | Shifts (video / audio) | Attention | Guidance |
|---|---|---|---|---|---|---|
| `larry-turbo` (`-<N>step`, N 3-8) | larry ckpt850 | 6 (default) | `linspace(1, 0, N + 1)` shifted | 12 / 3 | dense | 1 (no CFG) |
| `lightx2v-turbo` (`-<N>step`) | lightx2v v1.2 768p | 4 | same | 6 / 3 | dense | 1 |
| `lightx2v-turbo-v0.1` (`-<N>step`) | lightx2v v0.1 (not in R2) | 4 | same | 12 / 3 | dense | 1 |

**Steps count transformer forwards**, as everywhere in this repo. The Space's `steps` is diffusers'
`num_inference_steps`, the number of grid points (`MiniMaxH3Scheduler.set_timesteps`: "the schedule holds
`num_inference_steps` sigmas and drives `num_inference_steps - 1` model evaluations"), so the Space's default **6 is 5
forwards** (`larry-turbo-5step`) and its "4" is 3 forwards (`larry-turbo-3step`, below the card's minimum). larry's
own `generate.py` (`timesteps(n)`, n + 1 points) and lightx2v's "Note on shift" (`q_i = (N - i) / N`) count forwards:
their "4 steps" is our 4. The serve models default to the cards' numbers (larry 6, lightx2v 4).

| Serve model | Catalog | Profile | Tasks | Config |
|---|---|---|---|---|
| `h3-larry-turbo` | untiered (`h3-larry-turbo-ckpt850-dense`) | `h3/larry_turbo_dense` (dense attention, MXFP8, bf16 activations) | T2V, I2V (first frame), Keyframes (first + last frame), with audio; 768p (default) and 480p, 5-15 s | `configs/serve/runpod-h3-larry-turbo.toml` |
| `h3-lightx2v-turbo` | untiered (`h3-lightx2v-turbo-v1.2-768p-dense`) | `h3/lightx2v_turbo_dense` | same | `configs/serve/runpod-h3-lightx2v-turbo.toml` |

`[[models]]` settings: `steps = 3..8` (forwards; recipe `larry-turbo-<N>step`) and `lora_strength = 0..4`
(default 1.0; `FV_H3_TURBO_LORA_STRENGTH` for `fv-gpucheck`). Steps are a load-time setting (the AdaLN table is
precomputed per ladder), not a per-request knob, so a 4-step and a 6-step larry are two models (two pools). The tier
aliases are unchanged: `h3-turbo` stays FastH3 4-step VSA, `h3-max` Sol-H3. Each model needs its own 96 GB GPU (one
H3 DiT per card).

fv-control: presets `h3-larry-turbo`, `h3-lightx2v-turbo` (variant `h3-max`, config inline), recipe trees
`h3-base` + `h3-turbo-loras/<larry|lightx2v>` (+ `h3-base/text_encoder_fp8` with r2). The adapters are in R2 only,
so these pools fetch from R2 by default (the 500 GB EU volume holds H3 + LTX trees only).

## 5. Weights (R2)

| Tree | Rev | Files | Bytes | Manifest SHA-256 |
|---|---|---:|---:|---|
| `h3-turbo-loras/larry` | `43a74557ac3f6539db8e0f2a959d03feb7a81480` | 2 | 779,857,264 | `901f8cf5…f16ef0e4e0` |
| `h3-turbo-loras/lightx2v` | `3ec17a324ced54151364f24f8b5fb6bf7e26414f` | 2 | 1,383,679,204 | `ffdc68e0…2b02025d919c3c251de1974` |
| `auxiliary/ncii-light-guard` | `6a44e8b4b608078db164b0c946947922b2b98eab` | 5 | 569,621,401 | `9118b8cd…a9a36d9494594992f42a049b` |

File SHA-256 (Hub LFS): larry ckpt850 `5a6eeba1…3528b4ea`, lightx2v v1.2 `c3d4a2cf…cc6814b689`, guard
`model.safetensors` `d5d7ca0d…9c215c49b0`, `tokenizer.json` `ceeb59cb…3649fbeb0`. Recorded in
`scripts/gpu/weights-{manifest,revisions,sha256,r2}.tsv` and the `verify-weights.sh` cells `h3-turbo-loras` and
`ncii-guard`; docs/ops/r2-weights.md §13. The base transformer, text encoder and VAEs are `h3-base`'s.

## 6. Parity against the Space

TBD (filled in from the validation pod).

## 7. Speed and quality

TBD (filled in from the validation pod).

## 8. The NCII prompt guard

**What it is.** [`hfmlsoc/ncii-light-guard-v01`](https://huggingface.co/hfmlsoc/ncii-light-guard-v01), MIT, a rank-8
LoRA fine-tune (merged) of `microsoft/harrier-oss-v1-270m` (MIT): `Gemma3TextForSequenceClassification`, 18 layers,
hidden 640, 4 query heads over 1 KV head of 256, 270M parameters, 0.54 GB bf16 (0.57 GB with its tokenizer). **Text
only**: it scores whether an *edit prompt* risks non-consensual intimate imagery ("nudifying or adding a clear sexual
dimension to the subject's direct situation"); it never sees the image. Labels `safe` / `ncii`. Its card's limits:
English-centric, gender-dependent discrepancies, a heuristic and not a guarantee, weaker on tag soup and adversarial
phrasing, in-distribution F1 0.92-0.94.

**How the Space uses it** (`ncii_guard.py`, `app.py check_prompt`): `transformers.pipeline("text-classification",
device="cpu")` in a separate process, **only on requests that carry a keyframe** ("the edit-on-a-real-photo case"),
before the conditioner and the GPU booking; `label == "ncii"` refuses with "This prompt was flagged by a content
filter and wasn't run." and logs `prompt refused (ncii <score>): <prompt>`.

**This port** (`crates/fastvideo-models/src/ncii.rs`): a plain-Rust CPU implementation (Gemma3 RMSNorm `x (1 + w)`,
q/k norms, RoPE theta 1e6, causal attention scale 1/16, GELU-tanh gated MLP, the score head at the last token, the
embedding scale rounded to bf16 as the bf16 checkpoint applies it) with the repo's own `tokenizer.json` through the
`tokenizers` crate (BOS + prompt + EOS; truncation off, as the pipeline's `truncation=True` truncates at an unbounded
`model_max_length`; at most 2048 tokens). Checked against `transformers` 5.8 / torch 2.11 CPU on 8 prompts
(`scripts/gpu/ncii_reference.py` → `crates/fastvideo-models/tests/fixtures/ncii_reference.json`): token ids equal on
all 8 (8 to 434 tokens), labels equal, largest `|p_ncii - transformers|` **0.0075** (ours f32, the reference bf16).
CPU cost on the build pod (32 vCPU): load 1.4 s, 0.08-0.11 s per short prompt, 1.3 s for 434 tokens; about 0.75 GB
of RAM (linears f32, embedding table bf16). It runs on the blocking pool, never on the GPU.

**Admission check** (default **off**):

```toml
[moderation]
ncii = "keyframe"          # off (default) | keyframe | all   (FV_MODERATION_NCII)
# ncii_dir = "/workspace/weights/auxiliary/ncii-light-guard"   (FV_MODERATION_NCII_DIR; default $FV_WEIGHTS/...)
# ncii_threshold = 0.5     (FV_MODERATION_NCII_THRESHOLD: refuse when p(ncii) >= it; 0.5 = the pipeline's top label)
# ncii_concurrency = 2     (prompts classified at once)
```

`keyframe` checks requests with visual input (keyframes / first or last frame, image or video references, a source
video to edit), the Space's rule; `all` checks every prompt. A configured guard that cannot load stops fv-serve (fail
closed), and a prompt the guard cannot score is refused as an internal error. The refusal is
`prompt refused (ncii 0.87): this prompt was flagged by a content filter and was not run`, as
`ErrorKind::ContentFiltered`, which every API renders as its own content-policy error:

| API | Refusal |
|---|---|
| fal (queue, sync) | 422 `{"detail":[{..., "type": "content_policy_violation"}]}` |
| MiniMax | 422 `unprocessable_entity_error`, `base_resp.status_code` 1026 |
| LTX API | 422 `content_filtered_error` |
| OpenAI videos / FastVideo | 400 (`content_filtered`) |
| native `/fv/v1` | 422 `content_filtered` |
| fal director (WMA) | `configure`: session error `content_policy` (ends the session); `prompt` / `script`: `prompt_rejected{reason: "content_policy"}`. A session opened on an image counts its later prompts as visual input |
| Reactor | the session stops with RT's `moderation` notice (`{action: "terminate"}`); the command is not run |

The check runs in `ServeCtx::moderate_request` after admission and before the upsampler, ingestion and the engine, so
a refused prompt costs no GPU time and never becomes a job. **Logging:** fv-serve logs no prompt text anywhere, and
the guard keeps it that way (unlike the Space, which prints the refused prompt): a refusal logs the score, token
count, character count and the first 12 hex of the prompt's SHA-256; `fv_moderation_total{check="ncii",result}`
counts outcomes. The weights are the R2 tree `auxiliary/ncii-light-guard` (not part of any preset's trees; a pool
that turns the guard on must fetch it).

## 9. What is untested

TBD.
