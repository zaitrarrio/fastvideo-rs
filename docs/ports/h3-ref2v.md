# H3 reference-to-video (Ref2VA): upstream, checkpoints, port

Package E11 (docs/serve/design.md §8). Research date 2026-09-28. Sources:
the Hub API and files of the repositories named below at the pinned
revisions, FastVideo `e90be598` (`/home/user/hao-ai-lab/fastvideo`),
sol-engine `6c2f582b` (NVlabs/Sana branch `sol-engine`), and this repo's
fal / MiniMax research (docs/serve/research-fal.md §5.2,
docs/serve/research-minimax-fastvideo.md).

## 1. Answer first

**An open Ref2VA checkpoint exists.** It is part of the main MiniMax release,
not a separate model:

| What | Repo @ revision | Files | Bytes | License |
|---|---|---|---:|---|
| Ref2VA DiT (diffusers layout) | `MiniMaxAI/MiniMax-H3` @ `42ed227ee7df40d41602854ae760620d6eb651fe` | `transformer_ref/` (14 shards + index + config) | 66 280 569 250 | MiniMax H3 Community License |
| Same weights, original layout | same | `Ref2VA/transformer/` (13 shards) | 66.28 GB | same (not fetched: duplicate) |
| Qwen3-VL processor configs | same | `processor/` | 11.5 MB | same |
| Ref2VA 4-step turbo LoRA | `lightx2v/Minimax-h3-Turbo` @ `3ec17a324ced54151364f24f8b5fb6bf7e26414f` | `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` | 1 383 677 768 | Apache-2.0 card; `base_model: MiniMaxAI/MiniMax-H3` (derivative of the H3 weights) |
| Ref2VA 8-step turbo LoRA, 768p | same | `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` | 1 383 677 808 | same |

Everything else Ref2VA needs (Qwen3-VL-32B text/vision encoder, video VAE
*with its encoder*, audio VAE, tokenizer) is the same as T2V/I2V and is
already on both volumes in `weights/h3-base`.

What is **not** open: **H3-Context-IR** (the hosted prompt/context rewriter,
README "it is not included in this open-source release", only an API) and
**H3-Regenerate-2K** (hosted). The earlier finding stands: "H3-Context-IR" is
not a checkpoint. The MiniMax README's model table lists exactly three
open DiTs: H3-Base T2VA, H3-Base FL2VA (`transformer/`) and **H3-Base Ref2VA
(`transformer_ref/`)**. The lightx2v org also publishes
`MiniMax-H3-Prompt-Rewriter-LoRA{,-8B,-Omni}` (prompt rewriters, not needed
for Ref2VA itself; unevaluated here).

### 1.1 License: read before serving

`LICENSE` (MiniMax H3 Community License Agreement) §I.3/I.5: the "Applicable
Territory" is **worldwide excluding the European Union, the United Kingdom,
the Republic of Korea and the United States**. §V.4 and Exhibit A §1: no use,
reproduction, modification, distribution or display of the H3 works *or
their outputs* outside the Applicable Territory; organisations there are
invited to apply for a separate license (`docs/QA-about-License.md`,
platform.minimax.io/h3-license). Other terms: separate authorisation above
US$20M yearly revenue (§IV.1); "MiniMax H3" must be displayed prominently in
a commercial UI (§IV.2); outputs may not be used to improve other models
(§V.3); a provider of a hosted service must run and review abuse safeguards
and keep a reporting mechanism (§V.5).

This applies equally to the `h3-base` and `h3-8step` trees already on both
volumes: the US volume (`s2k01690bi`, US-CA-2) is inside an excluded
territory; the EU volume (`jg48s6o1w0`, EUR-IS-1) is in Iceland (EEA, not
EU). The Ref2VA download was made because the owner approved it; **serving
H3 (any task) from US or EU infrastructure, or to users there, needs the
owner's legal decision or a MiniMax license.** Not a technical blocker, so
it is recorded here and in the report rather than acted on.

## 2. Upstream implementations

| Upstream | Entry | Weights | Steps / guidance | Attention |
|---|---|---|---|---|
| MiniMax / diffusers | `Ref2VA/model_index.json`, `scripts/readme/reproducible-768p-ref2va-request.sh` (SGLang `--model-variant ref2va`) | `transformer_ref` | 50-point grid (49 forwards), shift 12 / 3, guidance 1.0 | dense |
| FastVideo `e90be598` | `MiniMaxH3Ref2VAModularPipeline` (`pipelines/basic/minimax_h3/minimax_h3_pipeline.py:454`, `_extra_config_module_map = {"transformer": "transformer_ref"}`), example `examples/inference/basic/basic_minimax_h3_ref2va.py`, preset `minimax_h3_ref2va` | `transformer_ref` | `--steps 50`, `guidance_scale=1.0`, 768×1344×124 default | dense (FLASH_ATTN) |
| sol-engine Sol-H3 `6c2f582b` | `models/minimax_h3/Sol-H3/infer.py --task ref2va`, `download_checkpoints.py --task ref2va` | `transformer_ref` + lightx2v `ref2v_turbo_4step_v0.1` fused | 4 forwards, uniform 5-point grid, shift 12 / 3 | `sol_bsa` (default) or `dense`; the T2V Sol policy is refused for Ref2VA (`infer.py:91`) |
| LightX2V | github.com/ModelTC/Minimax-H3-Turbo (not read here) | `transformer_ref` + `ref2v_turbo_8step_v1.0_768p` | 8 (schedule not verified) | — |

The FastVideo pipeline is the parity reference (same method as the other H3
parity work, docs/oracle.md). The Sol-H3 route is the reference for the
turbo tier.

## 3. Conditioning mechanism

In-context condition tokens in the one packed joint sequence, plus the
references shown to the multimodal text encoder. No subject embedding, no
adapter, no cross-attention branch: the Ref2VA DiT has the base config
(`transformer_ref/config.json` is byte-identical in content to
`transformer/config.json`: 50 layers, hidden 5376, 56 heads) with its own
weights (**no shard is shared** with `transformer/`; LFS SHA-256s all differ).

1. **Text + vision prompt** (`stages/minimax_h3_conditioning.py:59-90,175-181`).
   Qwen3-VL-32B sees the prompt with each reference inlined in order:
   `<Picture k>: ` + `<|vision_start|>` + image pad tokens + `<|vision_end|>`,
   `<Video k>: ` + video pad tokens (sampled at 2 fps, temporal patch 2),
   `<Audio k>: ` (text label only). Layer-50 hidden states become the text
   rows; vision tokens are tagged as vision in the row tags.
2. **Reference latents** (`stages/minimax_h3_latent_preparation.py:226-277`).
   Images: video-VAE encoder at 2048 short edge (official) or the target
   canvas (`match`, Sol-H3's fast default), one latent frame each. Videos:
   resampled to 24 fps, 768 short edge, trimmed to the `17n+5` VAE chunk grid
   and encoded. Audio (and a video's soundtrack): audio-VAE encoder, stereo as
   two channel streams. Visual condition rows get noise augmentation
   `scale_noise(clean, 0.999, noise)` on the clean-at-1 timestep convention
   (`packing.py:38`), i.e. ~0.1 % noise, drawn from the request generator
   *before* the target noise.
3. **Packing** (`packing.py:347-`): `[text | ordered references | target audio
   | target video]`. RoPE time: text 0..T-1, then each image one time step,
   each audio/video its own span, then the target. Condition audio rows are
   interleaved with condition video rows, which breaks VSA's
   contiguous-prefix tiles, so the Rust port runs such requests dense
   (`pipeline.rs` "interleaved condition audio").
4. Denoising updates only the target rows; condition rows stay fixed.

Prompting: the model was trained on Context-IR output (sections
`subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music`, labels
`<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>`; guide
`docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`). Plain prompts work but the
README calls Context-IR "critical to the quality of the final output". fal's
schema tells users to write "Image 1 / Video 1 / Audio 1" (research-fal §5.2);
the model's own label is `<Picture N>`.

## 4. Input limits

Same on MiniMax README, FastVideo (`reference.py:26-33`), fal (research-fal
§5.2) and our `RefLimits::h3()`:

| | limit |
|---|---|
| images | ≤ 9 |
| videos | ≤ 3, each 2–15 s, total ≤ 15 s |
| audio | ≤ 3, each 2–15 s, total ≤ 15 s |
| total files | ≤ 12 |
| at least one | visual (image or video) reference: audio alone is refused (FastVideo, Sol-H3 `infer.py:73`) |
| mixing | no first/last frame keyframes with references (FL2VA and Ref2VA are separate DiTs) |
| output | same grid as T2V: 24 fps, `17n+5` frames 5–15 s (MiniMax 4 s = 107 frames), 768 short edge (16:9 = 1344×768) |

## 5. Shared DiT or separate?

**Separate, full DiT: 66.28 GB BF16** (not the ~41 GB guessed in the brief;
the same size as `transformer/`). It cannot be expressed as a LoRA on the
base without a lossy low-rank fit nobody publishes. The turbo LoRAs sit *on
top of* `transformer_ref`.

Residency options for serving (weights only; add ~25 GB of activations at
5 s 768p, `denoise_reserve_bytes`):

| Card | Base T2V/I2V DiT + Ref2VA DiT + FP8 text encoder (25.95 GB) + VAEs (~11 GB) | Plan |
|---|---|---|
| B200 180 GB | BF16: 66.3 + 66.3 + 26 + 11 = ~170 GB: no room for activations. MXFP8 DiTs (default on sm_100+): ~35 + 35 + 26 + 11 = ~107 GB | co-resident with MXFP8 |
| H200 141 GB | W8A8 below sm_100 (c054278): ~107 GB + ~25 GB activations | co-resident, tight; verify |
| H100 80 GB, RTX PRO 6000 96 GB | one BF16 DiT + encoder already fills the card | separate process / pool (a Ref2VA-only worker), or swap mode (`resident = false`, reload ~66 GB from the volume per switch) |

## 6. Download (done)

Tree `weights/h3-ref2va/` (plain files): `transformer_ref/*`, `processor/*`,
`model_index.json`, `LICENSE`, `README.md` at `42ed227`, and
`Minimax-h3-Turbo/{README.md, the two ref2v turbo LoRAs}` at `3ec17a3`:
29 files, **69 059 483 520 bytes per volume** (138.1 GB for both, under the
150 GB cap). `scripts/gpu/fetch-h3-ref2va.sh <volume id>` (CPU pod, add-only:
refuses if the tree exists; writes `.h3-ref2va.partial-<stamp>`, checks every
LFS file's SHA-256 and every file's size against the Hub, fsyncs and re-reads
every file, then renames). Results in §8.

## 7. Port plan (what this package builds)

- `crates/fastvideo-cudarc/src/h3/`: Ref2VA already existed (packing,
  reference encode, Qwen-VL multimodal text, `transformer_ref` load, Sol-H3
  turbo adapter) but had never run on a GPU. Added: `ref_root` (the
  `transformer_ref/` + adapter tree lives outside `h3-base`), the `base` recipe
  (the upstream 50-point grid without an adapter), oracle injection of the
  `[condition | target]` starting rows.
- Recipes / tiers: **Max** = `base` (49 forwards, dense), the upstream
  default. **Turbo** = `sol-h3-ref2va` (4 forwards, lightx2v 4-step v0.1,
  dense), the Sol-H3 route. The 8-step v1.0 LoRA is downloaded but not wired
  (its schedule is not verified against LightX2V).
- Engine: `Task::Ref2V` on Ref2VA models only (a `transformer_ref` pipeline
  refuses non-reference requests); requests for another task on the tier
  alias go to the base model and Ref2V requests to its Ref2VA companion.
- Serve: fal `reference-to-video` (`reference_image_urls` ≤ 9,
  `reference_video_urls` ≤ 3, `reference_audio_urls` ≤ 3, `aspect_ratio`),
  MiniMax `reference_image` / `reference_video` / `reference_audio` content
  roles (V2) and `subject_reference` (V1 shape), console reference uploads.

## 8. Results

### 8.1 Download

| Volume | Pod (cpu3c, $0.24/hr) | Wall | Result |
|---|---|---:|---|
| US `s2k01690bi` | `davmpuszksq3y8` | 442 s | 29 files, 69 059 483 520 bytes; every LFS SHA-256 = Hub; re-read after fsync; renamed from `.h3-ref2va.partial-*` |
| EU `jg48s6o1w0` | `652qtvg5a6hqfy` | 296 s | same 29 SHA-256 / sizes as US |

Both pods deleted (verified). The `h3-base` snapshot on both volumes is
`42ed227`, the same revision. Hashes: `artifacts/runpod/fetch-h3-ref2va/*/sha256.txt`.

### 8.2 Parity against FastVideo (RTX PRO 6000, EUR-IS-1)

Target `h3-ref2va-4step`: FastVideo `MiniMaxH3Ref2VAModularPipeline`
(`e90be598`, strict eager, FLASH_ATTN, `--steps 5`) against ours
`h3 gen --h3-recipe base-4step --dense --ref beach.jpg --ref-root $W/h3-ref2va`,
768x1344x124, one 832x480 image reference, seed 1024, bf16 (`FASTVIDEO_H3_QUANT=off`),
the reference's starting noise injected.

**Run 1** (upstream `2d9dd45-09282004`, ours `9b2c963-09282013`): the
reference's packed rows are 44 400 = 7 104 condition + 37 296 target, as our
layout expects; sigmas and timesteps are identical. The text was not: our
multimodal prompt had **440 tokens against 7 154**. FastVideo shows Qwen-VL
the *prepared* reference image (PIL Lanczos to the 2048-short-edge canvas,
3552x2048: 7 104 vision tokens); we showed it the 832x480 file (390). Our
VAE path also resized with nearest neighbour. With different conditioning the
velocities agree only loosely (video `vel_step01` cosine 0.937, rel-L2 0.357;
latents after step 1 rel-L2 2.4e-2, after step 4 0.378), and the block dumps
are not comparable (different sequence lengths). Fixed in `64a3ff0`: one
Lanczos-prepared image feeds both the VAE and Qwen-VL.

Timings, 4 forwards (not a benchmark; one request each, cold):

| | load | text | denoise | video decode | peak |
|---|---:|---:|---:|---:|---:|
| FastVideo (TE and VAE offloaded) | 339.9 s | 12.9 s (conditioning) | 81.8 s | 9.7 s | |
| ours (streamed TE) | 91.9 s | 86.1 s (streamed Qwen-VL) | 86.1 s (first step 48.2 s, then 12.6 s/step) | 6.5 s | 50.2 GiB |

(Run 1 had 440 text tokens instead of 7 154, so its denoise is not like for like.)
