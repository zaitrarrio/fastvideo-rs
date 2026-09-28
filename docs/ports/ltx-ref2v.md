# LTX reference-to-video: research and plan

Question (serve E-ltx-ref2v): what does LTX offer for reference-driven video,
which weights does it need, and what can we serve faithfully on our LTX-2.5
distilled base? Read 2026-09-28: Lightricks/LTX-2 `fd4ded7`
(`packages/ltx-pipelines`, `packages/ltx-core`), the Hub model list of the
`Lightricks` org, docs/serve/research-ltx-api.md and the fal catalogue
(docs/serve/fal-parity.md §2).

## 1. What exists

| Mechanism | Where | What it does | Base |
|---|---|---|---|
| Image conditioning (I2V, keyframes) | `DistilledPipeline(images=…)`, `combined_image_conditionings` | pins pixel frames to images (frame 0 replaces latent frame 0; any other frame is appended tokens at that time) | any 2.x; **served (E5/E9)**, docs/ports/ltx25.md "Image conditioning" |
| **IC-LoRA "Ingredients"** | `ICLoraPipeline` (`ic_lora.py`) + `Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients` | a *reference sheet* image (character / prop / location panels) is VAE-encoded and appended as clean reference tokens (`VideoConditionByReferenceLatent`); an IC-LoRA trained on sheet→video pairs makes the video use the sheet's subjects. The prompt names the sheet ("Reference sheet: … Generated video: …") | **2.5 build exists** (also 2.3) |
| IC-LoRA control (union / canny / depth / pose / motion-track) | `ICLoraPipeline` + `LTX-2.3-22b-IC-LoRA-Union-Control`, `…-Motion-Track-Control`, `LTX-2-19b-IC-LoRA-*-Control` | a *control video* (pose, depth, edges) drives motion/structure | 2.3 / 2.0 only; no 2.5 build listed |
| IC-LoRA effects (clean-plate, colorization, day-to-night, deblur, decompression, water-simulation, pixel upscaler; 2.3 also HDR, relight, in/outpainting, Dub-It) | same pipeline | video-to-video effects | mixed; not reference generation |
| Retake / extend / A2V | `retake.py`, `keyframe_interpolation.py`, `a2vid_two_stage.py` | edit or continue a given video, audio-driven video | not reference generation; LTX API answers these 403 today |

The hosted LTX API (docs/serve/research-ltx-api.md) has **no** reference
endpoint: its surfaces are text-to-video, image-to-video (first/last frame),
audio-to-video, retake, extend and the HDR/reframe video-to-video. fal's
reference-like LTX apps are `fal-ai/ltx-2.3-quality/ingredient` (the
Ingredients IC-LoRA: `image_url` = the sheet, `ingredient_strength` and
`reference_strength` 0-2, default 1536x896 = a 768x448 first stage + 2x
refine) and `fal-ai/ltx-2.3-22b/reference-video-to-video` (control IC-LoRAs,
2.3 only).

**The most faithful open reference mode for our 2.5 base is the Ingredients
IC-LoRA.** Identity/"subject" references in the MiniMax H3 sense (several
free-form reference images, videos, audio) have no LTX equivalent; the sheet
is the LTX way to pass several subjects at once (one image with panels).

## 2. Weights

| Repo | Revision | File | Size | SHA-256 (Hub LFS) | License |
|---|---|---|---|---|---|
| `Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients` | `12040e4091ac2008d3906a594e31a7fb1ab9d546` | `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors` | 1 308 787 472 B (1.31 GB) | `ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95` | `ltx-2.x-community-license` (free below USD 10M annual revenue; a commercial licence above), **gated: auto-approval** |
| same | same | `README.md` | 25 643 B | — | |

Nothing else is needed: the base DiT, VAE and text encoder are the 2.5
weights already on both volumes (`weights/ltx25`). Total 1.31 GB, far below
the 100 GB stop line. Destination (add-only, both volumes):
`weights/ltx25-ic-lora-ingredients/`. Script: `scripts/gpu/fetch-ltx-iclora.sh`
(US pod pulls the pinned revision and checks the LFS SHA-256, renames the
temp folder into place, serves it; the EU pod copies from it, checks, renames;
`probe` reports Hub access).

**Status: on both volumes** (2026-09-28). The first fetch got HTTP 403 (the
volumes' HF token had not accepted the gate); once it had, the tree landed on
the US volume `s2k01690bi` and then the EU volume `jg48s6o1w0`, SHA-256
identical (`ff873a5b…8715e95`), recorded in `scripts/gpu/weights-manifest.tsv`
and the `ltx25-ic-lora-ingredients` cell of `scripts/gpu/verify-weights.sh`
(plus the composite `ltx25-ref2v` cell: the two-stage base and the LoRA).

## 3. How the reference mode works (upstream)

`ICLoraPipeline.__call__` (`ic_lora.py`), two stages:

1. Stage 1 at half size, **with the IC-LoRA fused** (`DiffusionStage` built
   with `loras=[ingredients]`), deterministic Euler over `DISTILLED_SIGMAS`
   (this pipeline does not switch to the ancestral sampler), conditionings =
   the image conditionings (if any) **plus** one
   `VideoConditionByReferenceLatent` per reference:
   * the reference (image or video, `decode_video_by_frame` + `video_preprocess`,
     no CRF) is resized/cropped to `(H/2)/s × (W/2)/s` with `s` =
     `reference_downscale_factor` from the LoRA's safetensors metadata, and
     VAE-encoded;
   * its tokens are appended with denoise mask `1 − strength`, clean latent =
     the encoded reference, noisy placeholder zeros;
   * their RoPE positions are the reference grid's own pixel coordinates
     (causal fix on), time `/ fps` (× `reference_temporal_scale_factor`),
     spatial `× s`, so the sheet overlays the target frame;
   * `conditioning_attention_strength < 1` or a mask adds a self-attention
     bias (`ConditioningItemAttentionStrengthWrapper`); at 1.0 there is none.
2. Upsample ×2, stage 2 at full size **without the LoRA** and without the
   reference, only the image conditionings (same as `DistilledPipeline`).

Everything but the LoRA fuse is already in our engine after E5/E9: appended
clean tokens with their own positions and mask, per-token timesteps, the
mask-aware samplers, `clear_conditioning`, and stage-2 image conditioning.
What remains:

| Piece | Work |
|---|---|
| IC-LoRA fuse at stage 1, unfused at stage 2 | `ltx2/lora.rs` already fuses a Comfy-keyed LoRA into a resident DiT and switches strength (`set_lora_strength`, the 2.3 dev path). Point it at the Ingredients file for stage 1 (strength 1 × `ingredient_strength`) and 0 for stage 2; read `reference_downscale_factor` from the metadata |
| Reference tokens | a `StageConditioning` segment kind "reference": positions from the reference grid (causal fix), `× s` spatially; mask `1 − reference_strength` |
| Stage-1 sampler | Euler (not ancestral) for this mode, as upstream |
| Serve | `Task::Ref2V` for LTX-2.5 when the LoRA is on the volume (`RefLimits`: 1 image); fal `ingredient` fields `image_url`, `ingredient_strength`, `reference_strength`; native `reference_urls` |
| Parity | oracle target running `ic_lora.py --video-conditioning <sheet> 1.0 --lora <ingredients> 1.0` against ours, as ltx25-i2v |

Estimated at one engine session plus one GPU oracle run, once the weights are
on the volumes.

## 4. Implementation (2026-09-28)

The model card adds two facts the pipeline code does not show: the LoRA is
**rank 128 on every block's `attn1` / `attn2` q/k/v/out and feed-forward**
(video stream only), and its reference input is **a static video**: the sheet
looped to the output's length and frame rate (at least 121 frames), so the
reference is a full clip of latent frames, not one frame. Trained bucket:
768x448, 121 frames, 24 fps (the stage-1 size of a 1536x896 output),
`reference_downscale_factor` 1.

| Piece | Where | What |
|---|---|---|
| IC-LoRA attach | `ltx2::pipeline::load_transformer_ic`, `ltx2::lora::{install, ic_lora_info}` | `PipelineOptions::ic_lora` (or `FASTVIDEO_LTX2_IC_LORA`) loads the DiT with the LoRA's factors attached at strength 0: every touched linear keeps its unfused base `W0` beside the live weight (480 linears, ~26 GB on the 22B DiT, plus the f32 factors ~2.6 GB). `reference_downscale_factor` / `reference_temporal_scale_factor` come from the safetensors `__metadata__` (default 1; a temporal factor other than 1 is refused) |
| Stage-1 fuse, stage-2 unfuse | `Ltx2Pipeline::set_ic_strength` | `W = W0 + s·B·A` rebuilt from the kept base (f32 accumulate, one bf16 rounding; upstream rounds `(B·s)@A` to bf16 then adds in bf16) before stage 1, and `W = W0` again before stage 2 (exact). Every request sets it, so a plain request on the same pipeline runs the base weights |
| Reference tokens | `i2v_encode::ReferenceTokens`, `StageConditioning::with_reference` | the sheet decoded (EXIF orientation applied), `resize_and_center_crop` to the stage-1 size over the downscale factor, no CRF, VAE-encoded once: a static clip of identical frames encodes to one latent frame repeated (every causal conv and space-to-depth sees identical frames), so the `(num_frames−1)/8+1` latent frames are that frame repeated. Appended after the image conditionings, clean latent = the reference, noisy placeholder zeros, denoise mask `1 − strength` (per-token timestep 0 at strength 1) |
| Positions | `fastvideo_models::ltx2::rope::ReferenceBlock`, `Ropes::with_conditioning` | the reference grid's own pixel extents with the causal fix, seconds at the target fps, spatial × downscale; at downscale 1 and the full clip they equal the target grid's positions, so each sheet token overlays the target token it came from |
| Sampling | `generate_hooked` | stage 1: `DiffusionStage`'s default Euler over the 8 distilled sigmas (not the ancestral sampler `DistilledPipeline` uses on 2.5), bf16 state, noise drawn over grid + reference rows (the reference rows then lerp back to clean); `clear_conditioning` drops the reference; upsample; stage 2 dense, no LoRA, no reference, image conditionings only |
| CLI | `fv-gpucheck ltx2 gen --two-stage --dense-stage2 --reference SHEET --ic-lora FILE [--reference-strength S] [--ic-lora-strength S]` | |
| Serve | `ltx25-ref2v` (catalog), `configs/serve/runpod-ltx-ref2v.toml` | the Ref2V companion of `ltx-pro` (tier Max, `route_task`): `Task::Ref2V` only, `RefLimits::ltx_ingredients()` (one image), knobs `seed` + `reference_strength`/`reference_lora_strength`, canvas default 896 short edge, frames 9 to 241. Native `/fv/v1/jobs`: `reference_urls`, `reference_strength` (0 to 1), `reference_lora_strength` (0 to 2). fal: `fal-ai/ltx-2.3-quality/ingredient` (`image_url`, `ingredient_strength`, `reference_strength`, `num_frames`, `frames_per_second`, `generate_audio`, 1536x896) |

Known differences from upstream, by design:

* The sheet is read as an image and looped in memory; upstream reads a
  static *video file*. With a lossless clip (the oracle's) the pixels are
  identical; with a lossy H.264 clip upstream sees codec noise we do not add.
* The fused weight is rounded once (f32 accumulate) instead of twice (bf16
  product, bf16 add): sub-ulp per element.
* Memory: the kept base costs ~26 GB of device memory for the pipeline's
  lifetime (fits an 80 GB H100 or a 96 GB RTX PRO 6000 at 1536x896). A host
  copy of the base (restore by H2D per request) would drop that at a few
  seconds per request; not done.

## 5. Results

* **GPU oracle** (docs/oracle.md, "LTX-2.5 reference-to-video"; H100, US-CA-2,
  2026-09-28): identical token layout (5376 grid + 5376 reference rows at
  stage 1, all 480 LoRA pairs attached), stage-1 block 0 at 2.1e-3 and the
  stage-1 latents on the T2V profile (steps 1-4 ≤ 1.8e-3, final 0.31 vs 0.34
  for T2V), a bump to ~0.1 in video blocks 25-33 at step 1 that recovers by
  block 47, encoder 1.7e-2, pixels exact, stage 2 and decode as T2V (clip
  SSIM 0.982 / PSNR 38.9 dB against the reference's with the stage-2 entry
  injected). Pass.
* **Serve**: CPU-tested (check.sh); the GPU E2E is pending (docs/serve/e2e/ltx.md).
* **Spend**: about $1.2 of H100 (upstream pod 13 min, runtime pod 8 min).

## 6. Options before the weights landed (historical)

1. Serve reference mode as **first-frame I2V from the sheet**: wrong
   (the sheet would become frame 0), so we do not.
2. Refuse: LTX `Task::Ref2V` stays a 400 (`task` unsupported for the model);
   this is what the server does today.
3. Unblock the weights (accept the gate), then implement §3 (recommended).
