# MiniMax-H3 Turbo LoRAs (larry, lightx2v) and the NCII prompt guard

Owner request 2026-10-13: implement what the
[`MiniMaxAI/MiniMax-H3-Turbo-Lora`](https://huggingface.co/spaces/MiniMaxAI/MiniMax-H3-Turbo-Lora) Space runs
(revision `9a76d679`), plus its optional NCII prompt guard. The Space runs **unquantized bf16 MiniMax-H3** with a
4-step "Turbo" LoRA folded into the weights at load (`h3_lora.py`), 6 sampling steps by default, dense attention
(cuDNN), diffusers' modular MiniMax-H3 pipeline from huggingface/diffusers#14371 @ `665f5782`, and a separate Space
for the Qwen3-VL conditioner.

What this port runs: base MiniMax-H3 (`h3-base`) with the same fold (float32 delta, one bf16 rounding), **then**
MXFP8 linears as every other served H3 recipe (merge first, quantize second), dense attention, our VAEs. Three serve
models, `h3-larry-turbo`, `h3-lightx2v-turbo` (v1.2 768p) and `h3-lightx2v-turbo-v01` (v0.1, the Space's lightx2v
file), and an optional admission check, `[moderation] ncii`.

GPU validation: §6 (parity), §7 (speed and quality); reports and samples in `artifacts/ports/h3-turbo-loras/` (run 1,
2026-10-14) and `artifacts/ports/h3-turbo-loras/run2/` (run 2, the same day: lightx2v v0.1 parity, 124-frame speed and
quality).

## 1. The two LoRAs

| | larry | lightx2v |
|---|---|---|
| Repo @ rev | `larryvrh/MiniMax-H3-Turbo-Lora` @ `43a74557` | `lightx2v/Minimax-h3-Turbo` @ `3ec17a32` |
| Licence | Apache-2.0 (base: `Comfy-Org/MiniMax-H3`) | Apache-2.0 (base: `MiniMaxAI/MiniMax-H3`) |
| File (ours) | `minimax_h3_turbo_4step_ema_ckpt850.safetensors` (0.78 GB), the Space's default | `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` (1.38 GB), the newest 4-step 768p file (default); `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` (1.38 GB), the Space's file |
| Layout | ComfyUI (reference) module tree: `blocks.N.attn.qkv_proj`, `blocks.N.mlp.fc1/fc2`, `blocks.N.attn.out_proj`, `blocks.N.adaln_proj.linear`, `token_refiner.blocks.N.*`, `final_layer.adaln_proj.linear` | PEFT on the diffusers tree: `transformer_blocks.N.attn.to_q.lora_A.default.weight` and friends (no AdaLN) |
| Rank, scale | 64 for the linears, 16 for the AdaLN; `alpha == rank` (scale 1, `W += B @ A`) | 128, alpha 8 (metadata `alpha: 8` on v1.x): scale 0.0625 |
| Pairs | 259 modules → 363 diffusers targets (QKV split in three) | 312 |
| Trained for | 4-8 steps (card: 4 is the minimum, 6-8 "noticeably better", past 8 over-sharp), shifts 12 / 3, strength 1.0 ("the sharpness dial": >1 against smear, <1 against grain) | FL2VA / T2VA, 4 NFE; v1.x 768p trained at video shift **6** / audio 3, v0.1 at 12 / 3 |

**Which lightx2v file.** ModelTC/Minimax-H3-Turbo "Model specs" lists FL2VA 4-step v0.1 (544p, shifts 12 / 3), 8-step
v1.0, 4-step v1.0 768p and 8-step v1.0 768p (shifts 6 / 3); the Hub repo adds v1.1 and v1.2 768p. The Space folds
**v0.1** (its `H3_LIGHTX_FILE` default). The first approval covered about 2.2 GB, so the first R2 trees hold larry
ckpt850 and lightx2v **v1.2 768p** (2.16 GB); the owner approved v0.1 (1.38 GB) on 2026-10-14 and it is its own tree,
`h3-turbo-loras/lightx2v-v0.1`, fetched by the recipe `lightx2v-turbo-v0.1` (shifts 12 / 3) and the serve model
`h3-lightx2v-turbo-v01`. v1.2 768p stays the `h3-lightx2v-turbo` default (owner's call; §7 compares the two). All
T2V/FL2V files sit
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
| `lightx2v-turbo-v0.1` (`-<N>step`) | lightx2v v0.1 (R2 `h3-turbo-loras/lightx2v-v0.1`) | 4 | same | 12 / 3 | dense | 1 |

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
| `h3-lightx2v-turbo-v01` (alias `MiniMax-H3-Turbo-Lightx2v-v0.1`) | untiered (recipe `lightx2v-turbo-v0.1`) | the recipe's (dense) | same | `configs/serve/runpod-h3-lightx2v-turbo-v01.toml` |

`[[models]]` settings: `steps = 3..8` (forwards; recipe `larry-turbo-<N>step`) and `lora_strength = 0..4`
(default 1.0; `FV_H3_TURBO_LORA_STRENGTH` for `fv-gpucheck`). Steps are a load-time setting (the AdaLN table is
precomputed per ladder), not a per-request knob, so a 4-step and a 6-step larry are two models (two pools). A Turbo
model's profile serves every step count of its LoRA (`ProcessPlan::for_models` accepts `larry-turbo-4step` under the
`larry-turbo` profile, and `lightx2v-turbo-v0.1` under lightx2v's: the fold and the shifts come from the recipe). The tier
aliases are unchanged: `h3-turbo` stays FastH3 4-step VSA, `h3-max` Sol-H3. Each model needs its own 96 GB GPU (one
H3 DiT per card).

fv-control: presets `h3-larry-turbo`, `h3-lightx2v-turbo`, `h3-lightx2v-turbo-v01` (variant `h3-max`, config
inline), recipe trees `h3-base` + `h3-turbo-loras/<larry|lightx2v|lightx2v-v0.1>` (+ `h3-base/text_encoder_fp8` with
r2). The adapters are in R2 only,
so these pools fetch from R2 by default (the 500 GB EU volume holds H3 + LTX trees only).

## 5. Weights (R2)

| Tree | Rev | Files | Bytes | Manifest SHA-256 |
|---|---|---:|---:|---|
| `h3-turbo-loras/larry` | `43a74557ac3f6539db8e0f2a959d03feb7a81480` | 2 | 779,857,264 | `901f8cf5…f16ef0e4e0` |
| `h3-turbo-loras/lightx2v` | `3ec17a324ced54151364f24f8b5fb6bf7e26414f` | 2 | 1,383,679,204 | `ffdc68e0…2b02025d919c3c251de1974` |
| `h3-turbo-loras/lightx2v-v0.1` (2026-10-14) | `3ec17a324ced54151364f24f8b5fb6bf7e26414f` | 2 | 1,383,679,284 | `e51cf57f…dc9fee332a61c396` |
| `auxiliary/ncii-light-guard` | `6a44e8b4b608078db164b0c946947922b2b98eab` | 5 | 569,621,401 | `9118b8cd…a9a36d9494594992f42a049b` |

File SHA-256 (Hub LFS): larry ckpt850 `5a6eeba1…3528b4ea`, lightx2v v1.2 `c3d4a2cf…cc6814b689`, lightx2v v0.1
`5ff4a12c…a15e395b29` (1,383,677,888 bytes; published add-only like the others: manifest written last, then `verify --deep
--record`), guard
`model.safetensors` `d5d7ca0d…9c215c49b0`, `tokenizer.json` `ceeb59cb…3649fbeb0`. Recorded in
`scripts/gpu/weights-{manifest,revisions,sha256,r2}.tsv` and the `verify-weights.sh` cells `h3-turbo-loras` and
`ncii-guard`; docs/ops/r2-weights.md §13. The base transformer, text encoder and VAEs are `h3-base`'s.

## 6. Parity against the Space

One RTX PRO 6000 Blackwell Server Edition (Runpod pod `dj7z8zfge2ldud`, 2026-10-14, 37 min, $1.52), fv-gpucheck of
this branch, every tree from R2 (`scripts/gpu/h3-turbo-loras-validate.sh` / `-pod.sh`). The reference is the Space's
own stack: diffusers #14371 @ `665f5782` (installed as 0.40.0.dev0), transformers 5.8.0, torch 2.11.0+cu130, its
`MiniMaxH3GeneratorBlocks` and `h3_lora.py` verbatim (Space revision `9a76d679`), bf16 transformer, float32 VAEs,
cuDNN attention (`scripts/gpu/upstream/h3_turbo_lora_ref.py`). Request: T2V, 832x480, 124 frames, seed 1024, the
Space's default `steps = 6` (5 forwards: our `larry-turbo-5step`), strength 1.0. Text: our Qwen3-VL layer-50 rows fed
to both (the Space's conditioner is another Space). Noise: the reference's draws (its generator) injected into ours
(`FASTVIDEO_INJECT_DIR`). Reports: `artifacts/ports/h3-turbo-loras/parity/`.

**Merged weights** (`merged_parity.py`; ours dumped with `FV_H3_DUMP_MERGED`, the reference after
`apply_lora`). Both sides are bf16, so "differ" is the share of elements one or more bf16 ulps apart; the delta rel-L2
compares `merged - base` (the LoRA's own contribution, about 1e-4 to 3e-3 of the base's norm, so a 1-ulp rounding
difference of the sum is a visible share of it).

| larry, target type (worst of the sampled linears) | merged rel-L2 | delta rel-L2 | elements differing | max ulps |
|---|---|---|---|---|
| attn.to_out.0 (out_proj), blocks 0 / 49 | 4.0e-6 | 1.1e-2 | 0.001% | 2 |
| ff.net.0.proj (fc1, SwiGLU swap), blocks 0 / 49 | 7.7e-6 | 1.1e-2 | 0.001% | 1 |
| ff.net.2 (fc2), blocks 0 / 49 | 1.4e-5 | 1.2e-2 | 0.002% | 1 |
| block adaln_proj.linear, block 25 | 6.8e-7 | 7.8e-3 | 0.000% | 1 |
| norm_out.linear (final AdaLN) | 2.3e-6 | 7.6e-3 | 0.000% | 1 |
| refiner attn.to_out.0 / ff.net.0.proj | 2.0e-6 | 1.1e-2 | 0.001% | 1 |

The fold is the Space's to bf16 rounding: one element in 50 000 to 100 000 lands one ulp away (float32 summation order
of `B @ A`, CPU vs our device kernel). **Not dumped:** `to_q/to_k/to_v`, whose merge happens on the fused QKV(G) load
path rather than in `load_linear`; they are covered by the step-1 latents below (a wrong third or a wrong order would
not give 8e-4). lightx2v in run 1: the reference's lightx weights came from the Space's `set_active("lightx")`
*after* the larry fold (unfold larry, fold lightx: two more bf16 roundings), so 0.04-0.5% of elements differed, by up
to 27 000 ulps (values crossing zero), merged rel-L2 <= 8.4e-5: the Space's switch residue. Run 2 folds lightx alone
(below).

**Denoise and decode** (`compare-dumps`, rel-L2 per step; `av_psnr.py` on the decoded clips):

| | bf16 linears (`FASTVIDEO_H3_QUANT=off`) | served (MXFP8) |
|---|---|---|
| starting noise, text | 0 (injected) | 0 |
| sigmas (video, audio) | 0 | 0 |
| video velocity, step 1 | 1.67e-2 (cos 0.99990) | 4.31e-2 (cos 0.99917) |
| **video latents after step 1** | **8.26e-4** | **2.13e-3** |
| audio latents after step 1 | 1.04e-3 | 3.88e-3 |
| video latents after step 5 (final) | 2.43e-1 (cos 0.971) | 2.64e-1 (cos 0.965) |
| audio latents after step 5 | 9.99e-2 (cos 0.995) | n/a |
| decoded video PSNR (124 frames) | **22.7 dB** (frame min 20.1) | 22.5 dB |
| decoded audio PSNR (32 kHz stereo) | **46.2 dB** (lag 0) | 36.0 dB |

The first step matches to the bf16 noise floor (the velocity's 1.7e-2 is bf16 accumulation over 50 blocks with a
different attention kernel). The later steps grow quickly: from step 2 on the video velocity differs by 0.18-0.21
(cos 0.98), i.e. this 5-forward distilled trajectory amplifies small differences far more than the 49-forward base
(docs/oracle.md); the decoded clip is visibly the same scene (PSNR 22.7 dB; the reference decodes its video VAE in
float16 autocast, ours in its own precision, part of the gap). Reference wall: 29.4 s for denoise + decode at 480p,
peak 77.3 GiB allocated (bf16 transformer + float32 VAEs resident); the Space's fold took 113 s on the CPU.

Samples: `artifacts/ports/h3-turbo-loras/samples/parity-space-ref-larry5-480p.mp4` (the Space's stack) and
`parity-ours-larry5-480p-bf16.mp4` (ours, its noise and our text).

### 6.1 lightx2v v0.1 against the Space (run 2)

One RTX PRO 6000 Blackwell (Workstation Edition, 2 MiB used at boot; Runpod pod `cj9lreo21zatea`, 2026-10-14, 28 min),
the same reference stack with `h3_turbo_lora_ref.py --active lightx`: `H3_LORA=off`, `H3_LORA_DEFAULT=lightx`, so the
Space folds **only** v0.1 (the file its `H3_LIGHTX_FILE` names, scale alpha 8 / rank 128) into the bf16 base, as a
Space that starts on lightx does. Same request as §6 (T2V 832x480, 124 frames, seed 1024, `steps = 6` = 5 forwards,
shifts 12 / 3: our `lightx2v-turbo-v0.1-5step`), its noise injected, our text. Reports:
`artifacts/ports/h3-turbo-loras/run2/parity/`.

| lightx2v v0.1, target type (worst sampled) | merged rel-L2 | delta rel-L2 | elements differing | max ulps |
|---|---|---|---|---|
| attn.to_out.0, blocks 0 / 49 | 4.4e-6 | 1.4e-2 | 0.001% | 1 |
| ff.net.0.proj, blocks 0 / 49 | 6.6e-6 | 1.2e-2 | 0.001% | 2 |
| ff.net.2, blocks 0 / 49 | 6.0e-6 | 1.4e-2 | 0.001% | 1 |
| refiner attn.to_out.0 / ff.net.0.proj | 1.3e-6 | 1.4e-2 | 0.000% | 8 |

| | bf16 linears | served (MXFP8) |
|---|---|---|
| starting noise, text, sigmas | 0 | 0 |
| video velocity, step 1 | 2.19e-2 (cos 0.99978) | 5.73e-2 (cos 0.99840) |
| **video latents after step 1** | **9.16e-4** | **2.39e-3** |
| audio latents after step 1 | 1.28e-3 | 3.05e-3 |
| video latents after step 5 (final) | 2.41e-1 (cos 0.971) | 3.38e-1 (cos 0.943) |
| audio latents after step 5 | 3.88e-1 (cos 0.930) | 4.75e-1 (cos 0.881) |
| decoded video PSNR (124 frames) | **22.4 dB** (frame min 20.1) | 18.9 dB (frame min 14.3) |
| decoded audio PSNR | **43.9 dB** (lag 0) | 43.4 dB |

The lightx2v fold matches the Space's as closely as larry's (one element in 100 000 one ulp away), and the denoise
matches at step 1 to the same floor as larry (9.2e-4 against 8.3e-4), then drifts the same way over the distilled
trajectory; the decoded clip is the Space's scene at the same PSNR as larry's (22.4 against 22.7 dB). The served
MXFP8 form drifts further for lightx2v than for larry (18.9 against 22.5 dB video). Reference wall 77 s for the run
(load, fold, denoise, decode). Samples: `run2/samples/parity-space-ref-lightx-v01-5fwd-480p.mp4`,
`parity-ours-lightx-v01-5fwd-480p-bf16.mp4`.

## 7. Speed and quality

Run 2 (pod `cj9lreo21zatea`, RTX PRO 6000 Blackwell Workstation Edition), our recipes as served (their profiles:
MXFP8 linears, bf16 activations; text encoder streamed with a shared conditioning cache), each one warm process over
two prompts of `scripts/gpu/prompts-eval.json` (`h3-demo` seed 0, `spark-mountain-lake` seed 42) at 480p (832x480)
and 768p (1344x768), **124 frames (5 s at 24 fps)** for every clip, the base included. Numbers:
`artifacts/ports/h3-turbo-loras/run2/perf/*-benchmark.json`; medians per prompt.

| Recipe (forwards) | 480p denoise / total (s) | 768p denoise / total (s) | peak (GiB) 480p / 768p | load (s) |
|---|---|---|---|---|
| `h3-larry-turbo` (6) | 12.5-13.0 / 16.2-18.1 | 52.5-53.3 / 59.5-60.4 | 29.8 / 33.8 | 11.1 |
| `larry-turbo-4step` (4) | 8.4-8.8 / 11.6-11.9 | 35.1-35.6 / 42.1-42.7 | 29.8 / 33.7 | 12.0 |
| `h3-lightx2v-turbo` (v1.2, 4) | 8.4-8.8 / 11.6-12.0 | 35.0-35.6 / 42.1-42.6 | 30.0 / 33.9 | 13.5 |
| `h3-lightx2v-turbo-v01` (v0.1, 4) | 8.4-8.8 / 11.6-11.9 | 35.0-35.6 / 42.0-42.6 | 29.9 / 33.9 | 11.4 |
| `h3-turbo` (FastH3 4-step, VSA 0.9) | 7.6-8.1 / 10.8-11.3 | 20.7-21.5 / 27.8-28.6 | 35.7 / 42.9 | 15.8 |
| `h3-max` (Sol-H3 4-step, Sol tau ladder) | 6.8-7.2 / 10.0-10.4 | 23.3-23.9 / 30.3-31.0 | 30.0 / 34.3 | 12.0 |
| base, 49 forwards, dense | 103.9-108.3 / 107.1-111.6 | - | 29.2 | 10.0 |

Peak is the process's device memory per prompt. Denoise per forward: 2.15 s at 480p for every dense recipe (the base
and the LoRAs alike), 8.8 s at 768p for the LoRAs against 5.3 s (h3-turbo's VSA) and 5.9 s (h3-max's Sol route). The
two lightx2v files cost the same (same rank, same fold). At 480p a 4-step LoRA clip takes 12 s end to end against the
base's 109 s (9x); at 768p it costs 1.7x h3-turbo's denoise and 6 forwards 2.5x. The LoRA merge is 3.4 s on the
device (`load/io ... lora_s`); load is short because the trees were in the page cache. Decode 3.0 s (480p) / 6.8 s
(768p). Run 1 (pod `dj7z8zfge2ldud`, Server Edition, 107-frame clips by mistake) is kept in `perf/`.

**Quality** (`fv-gpucheck compare-clips`, LPIPS(alex) on the GPU, the repo's gate metrics, docs/techniques.md; all
pairs 124 / 124 frames). Each cell is h3-demo / spark-mountain-lake.

Against the **49-forward base at 480p** (the gate's baseline):

| Candidate | LPIPS | PSNR (dB) | sharpness ratio | jitter ratio |
|---|---|---|---|---|
| larry 6 | 0.520 / 0.513 | 12.5 / 15.1 | 1.21 / 0.88 | 0.78 / 0.78 |
| larry 4 | 0.555 / 0.536 | 12.6 / 14.8 | 1.15 / 0.93 | 0.67 / 0.80 |
| lightx2v v1.2, 4 | 0.558 / 0.510 | 12.5 / 15.3 | 1.13 / 0.81 | 1.31 / 0.74 |
| lightx2v v0.1, 4 | 0.531 / 0.619 | 12.8 / 13.0 | 0.97 / 0.91 | 1.36 / 1.05 |
| h3-turbo | 0.693 / 0.673 | 10.8 / 11.3 | 0.99 / 1.29 | 1.59 / 1.22 |
| h3-max | 0.619 / 0.732 | 11.8 / 11.0 | 1.23 / 1.47 | 2.24 / 2.22 |

Against **h3-max at 768p**:

| Candidate | LPIPS | PSNR (dB) | sharpness ratio | jitter ratio |
|---|---|---|---|---|
| larry 6 | 0.675 / 0.581 | 10.1 / 15.9 | 0.69 / 0.86 | 0.39 / 0.55 |
| larry 4 | 0.675 / 0.591 | 10.0 / 15.9 | 0.71 / 0.78 | 0.43 / 0.52 |
| lightx2v v1.2, 4 | 0.648 / 0.617 | 10.1 / 15.1 | 0.77 / 0.62 | 0.67 / 0.38 |
| lightx2v v0.1, 4 | 0.620 / 0.598 | 10.8 / 15.1 | 0.71 / 0.71 | 0.79 / 0.50 |
| h3-turbo | 0.703 / 0.595 | 9.4 / 15.6 | 0.65 / 0.94 | 1.07 / 0.86 |

Within one LoRA: larry 4 vs larry 6 (baseline) LPIPS 0.453 (h3-demo 768p) / 0.373 (spark 480p), sharpness 1.03 /
1.06; lightx2v v0.1 vs v1.2 (baseline) LPIPS 0.519 (h3-demo 768p) / 0.551 (spark 480p), sharpness 0.94 / 1.11,
jitter 1.33 / 1.42.

Reading: every candidate is a different sample of the same model and seed, not the base's trajectory (LPIPS >= 0.5
for all), so the gate's pass/fail rules (docs/techniques.md) do not decide anything here; the comparison is relative.
**All four LoRA recipes stay closer to the 49-forward base than the repo's own 4-step tiers** (LPIPS 0.51-0.56, one
0.62, against 0.62-0.73 for h3-turbo and h3-max), with the base's motion level (jitter 0.7-1.4 of the base's; h3-max
2.2x, h3-turbo 1.2-1.6x). Against h3-max at 768p the LoRA clips are softer (sharpness 0.62-0.86) and calmer (jitter
0.4-0.8). larry 4 is close to larry 6 (its card: 4 the minimum, 6-8 better). **v0.1 vs v1.2:** no consistent winner.
v0.1 is closer to the base on h3-demo (0.531 vs 0.558) and further on spark-mountain-lake (0.619 vs 0.510), closer to
h3-max on both 768p prompts (0.620 / 0.598 vs 0.648 / 0.617), with more motion (jitter 1.33-1.42 of v1.2's); v0.1 was
trained at 544p, v1.2 at 768p. These numbers give no reason to change the `h3-lightx2v-turbo` default (v1.2);
`h3-lightx2v-turbo-v01` serves the Space's file for anyone who wants the Space's output (§6.1: it matches it). Samples
(<= 10 MB): `run2/samples/lightx4-spark-mountain-lake-768p.mp4` (v1.2), `run2/samples/lightxv01-spark-mountain-lake-768p.mp4`
(v0.1); run 1's in `samples/`.

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

- larry at 7-8 forwards, strengths other than 1.0, larry v4-600, lightx2v at other step counts, and 1080p. Quality
  is measured on two prompts with one seed each.
- FL2VA (first / last frame) with either LoRA on the GPU: only T2V ran. The keyframe path is the base pipeline's,
  unchanged by the adapter.
- `to_q/k/v` merged weights (fused load path, not dumped; covered by the step-1 latents of both parity runs).
- The served path end to end on fv-serve (the catalog, configs and control presets are unit- and integration-tested
  on CPU; no fv-serve process served these models on a GPU), and the fv-control presets on a real pool.
- The NCII guard inside a running fv-serve on a pod, the director and Reactor refusals end to end (unit-tested at the
  seam), and the guard's accuracy beyond its card's numbers. Its weights are not part of any preset's trees.
- Cost: run 1 $2.26 (pod 1 $1.52, pod 2 $0.74 without results); run 2 $1.23 (pod `cj9lreo21zatea`, 28.4 min at
  $2.59/h). Every pod deleted and checked gone.
