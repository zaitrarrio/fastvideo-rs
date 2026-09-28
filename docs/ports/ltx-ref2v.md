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

**Status: blocked on Hub access.** The first fetch (2026-09-28, CPU pod on
the US volume) got **HTTP 403** from the Hub for the LoRA: the HF token on the
volumes (`/workspace/hf/token`) belongs to an account that has not accepted
this repo's gate. The gate is auto-approved, but it must be accepted once on
https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients by the
token's account (a human click; we do not do that on anyone's behalf). After
that, `bash scripts/gpu/fetch-ltx-iclora.sh` does both volumes in a few
minutes (about $0.02 of CPU pods). Nothing was written to either volume
except this script's own temp folder, which its `probe` mode removes.

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

## 4. Options until then

1. Serve reference mode as **first-frame I2V from the sheet**: wrong
   (the sheet would become frame 0), so we do not.
2. Refuse: LTX `Task::Ref2V` stays a 400 (`task` unsupported for the model);
   this is what the server does today.
3. Unblock the weights (accept the gate), then implement §3 (recommended).
