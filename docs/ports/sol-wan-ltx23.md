# sol-engine cells: Wan2.1-T2V-1.3B, Wan2.2-T2V-A14B, LTX-2.3 HQ

Status 2026-10-06. Everything here is checked on the CPU against the
published checkpoints' tensor headers and configs. **No GPU frame has been
produced yet for any of the three**: the weights are not on the EU volume
(see [Weights to fetch](#weights-to-fetch-owner-approval)) and no pod was
started. The benchmark arms are written and ready
(`runpod-matrix.sh solbench`, [below](#benchmark-arms)).

Sources:

- sol-engine: NVlabs/Sana branch `sol-engine` @ `670482d`. The profiles are
  `models/{wan21_t2v_1_3b,wan22_t2v_a14b,ltx23}.toml`, the arms
  `config/<model>/*.toml`, and the golden runs `evals/_golden/wan14b_*_1g`.
- Hub headers: two HTTP range requests per safetensors file, fetched
  2026-10-06. They are stripped to `key → [dtype, shape]` in
  `crates/fastvideo-cudarc/src/{wan,ltx2}/manifests/`. Repo and revision are
  recorded inside each file.

| Model | Loader vs Hub header | Sampler vs sol-engine | Memory on one RTX PRO 6000 | Arms |
|---|---|---|---|---|
| Wan2.1-T2V-1.3B base | exact (825 keys) | UniPC 50, shift 3, CFG 6. Diffusers sigmas opt-in | 1.4 B params, trivial | `wan13-sol-*` |
| Wan2.2-T2V-A14B | exact (1095 keys × 2 experts) | UniPC 40, shift 12, CFG 4 / 3, boundary 0.875 → 26 / 14 steps (= golden 52 / 28 forwards) | expert swap (new) or MXFP8; B200 both resident | `a14b-sol-*` |
| LTX-2.3 HQ | exact after 4 fixes (5947-key dev pack, FastVideo split tree, LoRA, upscaler) | 15 res2s + 3-sigma, CFG 3, LoRA 0.25 / 0.5 | dev DiT resident, LoRA base in pinned host | `ltx23-hq-*` |

## Wan2.1-T2V-1.3B (base)

The preset is `wan_t2v_1_3b` and its checkpoint `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
@ `0fad780`.

- **Config.** `WanVideoArchConfig::from_preset("wan_t2v_1_3b")` equals the Hub
  `transformer/config.json`: 12 heads × 128, FFN 8960, 30 layers, 16 → 16
  channels, patch (1, 2, 2), text 4096, freq 256, eps 1e-6, rope 1024,
  `rms_norm_across_heads`, no image projection. The test is
  `wan::manifest_tests::wan21_t2v_1_3b_base_loads_from_the_published_transformer`.
- **Keys.** The DiT loader asks for exactly the 825 published keys, with
  their shapes. There is no VSA gate on the base tree.
- **The rest of the tree.** `text_encoder/` (UMT5-XXL, f32, 22.7 GB) and
  `vae/` have the same Hub LFS SHA-256 as `wan21-t2v-14b`'s, which is already
  on the volume.
- **sol-engine profile.** 832x480, 81 frames at 16 fps, 50 steps,
  `flow_shift` 3.0, CFG 6.0, seed 1024. The runner is Diffusers' `WanPipeline`
  with `UniPCMultistepScheduler.from_config(flow_shift=3)`. Our registry
  preset keeps FastVideo's defaults (CFG 3), so the arm passes the sol-engine
  values on the command line.
- **Sigmas.** Diffusers shifts the flow sigmas once. FastVideo's
  `FlowUniPCMultistepScheduler(shift=s)` also shifts its training sigmas, so
  the inference sigmas are shifted twice (σ₁ 0.99313 vs 0.99291 at 50 / 3).
  `FASTVIDEO_WAN_UNIPC_SIGMAS=diffusers` selects Diffusers' list
  (`FlowUniPCMultistepScheduler::new_single_shift`), which is checked to
  1e-6 against numpy float64. FastVideo's list stays the default.

```bash
FASTVIDEO_WAN_UNIPC_SIGMAS=diffusers FASTVIDEO_WAN_QUANT=off FASTVIDEO_WAN_VAE=full \
fv-gpucheck --mode fast wan gen --weights $W/wan21-t2v-1.3b --preset wan_t2v_1_3b \
  --unipc --steps 50 --guidance 6.0 --flow-shift 3.0 --fps 16 \
  --height 480 --width 832 --num-frames 81 --seed 1024 \
  --negative "<Wan Chinese negative prompt>" \
  --prompts scripts/gpu/prompts-sol-wan-t2v5.json --warm --clip-dir out/wan13
```

The optimized arm adds `FASTVIDEO_WAN_SOL_ATTN=fullstack
FASTVIDEO_WAN_SOL_CACHE=easycache FASTVIDEO_WAN_EASYCACHE_PROFILE=fullstack`.
That reproduces `config/wan21_t2v_1_3b/wan21_fullstack_sol.toml`:

- EasyCache 0.036.
- Sol-Attn at tau 1.0 with `diag` thresholds.
- The first 10 forwards dense, layer 0 dense, global Morton3D.

`fullstack` is new: on the 1.3B it selects the 14B guards. Plain `1` keeps the
DMD students' route (no dense forwards). sol-engine's `regional_compile`
(torch.compile) has no counterpart; our kernels are native.

**Published:** none (the profile has no measured run), so the cells measure
our base vs our optimized arm.

## Wan2.2-T2V-A14B

The preset is `wan_2_2_t2v_a14b` and its checkpoint
`Wan-AI/Wan2.2-T2V-A14B-Diffusers` @ `5be7df9`.

- **Keys.** `transformer/` (high noise) and `transformer_2/` (low noise)
  carry the same 1095 keys, dtypes and shapes. Both are float32: 57.15 GB
  each, 14 288 491 584 parameters (= the index's `total_size` / 4). Both
  experts load through the same loader, which asks for exactly those keys
  (`wan::manifest_tests::wan22_a14b_experts_load_from_the_published_transformers`).
  The config matches the Hub's (the 14B geometry).
- **Text encoder and VAE.** The A14B ships UMT5 in bf16 (11.36 GB, 3 shards),
  with the same 242 keys as Wan2.1-14B's f32 copy. The VAE is byte-identical to
  Wan2.1's.
- **Boundary.** `model_index.json` gives `boundary_ratio` 0.875. Diffusers
  uses `transformer` while `t >= 875`. At 40 steps and shift 12, 26 steps
  run on the high-noise expert and 14 on the low-noise one, under both sigma
  lists. With CFG that is 52 / 28 forwards: sol-engine's golden optimized run
  records exactly that (`evals/_golden/wan14b_opt_1g/benchmark.json`
  `pisa_step_tracking`). The test is
  `wan::moe::tests::a14b_boundary_split_matches_the_sol_engine_golden_run`.
- **Sampler.** UniPC, 40 steps, shift 12, CFG 4.0 for the high-noise expert
  and 3.0 for the low-noise one (`wan gen --guidance 4.0 --guidance-2 3.0`,
  new). 1280x720, 81 frames at 16 fps, seed 1024, and Wan's Chinese negative
  prompt.

**Published (1x GB200, 5-prompt median, hot):**

| Arm | Total | Denoise |
|---|---|---|
| Base | 449.67 s | 434.93 s |
| Optimized (`singlegpu_opt.toml`) | 207.01 s (2.17x) | 192.79 s |

The optimized arm is kernels + EasyCache 0.30 (start 5, tail 3, max reuse 1)
+ PISA density 0.10 (dense layers 0-3 and 40-43, dense steps 0-3 and 37-39).

Ours is `FASTVIDEO_WAN_SOL_CACHE=easycache` (the A14B controller: block 0
fresh, blocks 1-39 residual) with `FASTVIDEO_WAN_PISA=1` (the same dense
sets).

### Fitting one 96 GB card: the expert swap

As bf16 each expert is 26.6 GiB. The f32 UMT5 takes 21.1 GiB. Two experts,
UMT5, 720p activations and the VAE decode do not fit 94.97 GiB with a margin.
`FASTVIDEO_WAN_MOE` (in `fastvideo_models::wan::moe`) decides:

- **`swap`.** Both experts load parked: each block's linears are copied into
  one pinned host buffer as the block loads, so loading holds one block on the
  device at a time.
  - At the start of a generation, the high-noise expert's first forward
    queues the copies of all 40 blocks on a second stream. Each block waits
    only for its own copy, so the upload overlaps the compute. A whole-model
    ring (`BlockWeights::with_whole_ring`) then keeps the expert resident.
  - At the first step below the boundary, `pick_expert` drops the high
    expert's device copy (`WanTransformer3D::park`). The low expert comes in
    the same way.
  - The next generation swaps back.
  - Cost: two H2D copies of 26.2 GiB per generation (about 1 s each at PCIe 5
    rates, mostly hidden behind block compute), which is under 1 % of a
    20-minute 720p generation. Each copy's throughput and hidden fraction is
    logged per generation (`wan dit offload …`).
  - Host memory: both experts' block weights pinned (52.4 GiB) plus the page
    cache of the f32 shards while loading. **Use a pod with at least 96 GB of
    RAM.**
- **`both`.** Both experts resident. This is the default (`auto`) when the
  free memory after UMT5 covers 2 × 26.6 GiB + 24 GiB of headroom
  (`FASTVIDEO_WAN_MOE_HEADROOM_GIB`): B200 and H200, not the PRO 6000.
- **FP8 alternative.** `FASTVIDEO_WAN_MOE=both FASTVIDEO_WAN_QUANT=mxfp8` puts
  the reference MXFP8 recipe on every block linear (about 13.7 GiB per
  expert), so both fit resident on the PRO 6000. It is lossy: on sm_120 the
  Wan FP8 arms were slower than bf16 and doubled LPIPS (`docs/scope.md`).
  Cell `a14b-sol-mxfp8`.

The swap reuses the layerwise offload machinery that H3 and LTX-2 already
use (`wan/offload.rs`). Wan's DiT blocks now live in `BlockWeights` too, so
`FASTVIDEO_DIT_OFFLOAD=streamed` streams any Wan DiT layer by layer. Parked,
streamed and resident blocks give bit-identical outputs, before and after a
park (`wan::transformer::tests::parked_and_streamed_blocks_match_resident`,
`wan::offload::tests::whole_ring_is_bit_identical_and_survives_release`).

## LTX-2.3

The checkpoint trees:

- `ltx23` on the volume is `FastVideo/LTX-2.3-Distilled-Diffusers` @ `22b09fb`.
- The HQ cell needs `Lightricks/LTX-2.3` @ `3c6a4e6`:
  `ltx-2.3-22b-dev.safetensors`, `ltx-2.3-22b-distilled-lora-384-1.1.safetensors`
  and `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`.

How the trees relate:

- **Pack key sets.** The dev pack, the distilled pack, distilled-1.1 and the
  FastVideo tree's `download/ltx-2.3-22b-distilled.safetensors` share one key
  set: 5947 keys with the same dtypes and shapes.
- **The FastVideo folders are that file split:**
  - `transformer/` is `model.diffusion_model.*` without the connectors (4186
    keys).
  - `vae/` (170), `audio_vae/` (102) and `vocoder/` (1227) are `vae.*`,
    `audio_vae.*` and `vocoder.*`.
  - `text_embedding_projection/` (262) is the connectors with
    `video_embeddings_connector` renamed `embeddings_connector`, plus the two
    aggregate projections.
- **LoRA and upscaler.** The FastVideo tree ships the older LoRA 384 (not
  1.1) and an x2 upscaler file that is not `x2-1.1` (different LFS hashes,
  same keys).

### Loader fixes (each found by a header test, `ltx2::manifest_tests::ltx23`)

1. **Video and audio VAE.** The 2.3 folders keep the original (ltx-core)
   names:
   - a flat decoder `up_blocks.{0..8}`, with `res_blocks` and upsampler `conv`
     entries in turn;
   - the encoder `down_blocks.{0..8}`;
   - `per_channel_statistics.{mean,std}-of-means`.

   The loaders ask for diffusers names (`decoder.mid_block.resnets.0.conv1…`).
   That is the error that ended the H200 and B200 suites. A `WeightMap` alias
   view (`ltx2::keys::vae_view`, the rename table `ltx_core_vae_key`) now
   answers the diffusers names. The encoder's down-block probe stops before
   the mid entry.
2. **Vocoder.** It keeps HiFi-GAN names under `vocoder.` and `bwe_generator.`:
   `conv_pre` / `conv_post`, `ups`, `resblocks`, `act_post`. Fixed by
   `keys::vocoder_view`.
3. **Connectors.**
   - FastVideo's folder spells the video connector `embeddings_connector`
     (`keys::connectors_view`).
   - The single file keeps the per-modality text projections at
     `text_embedding_projection.{video,audio}_aggregate_embed`, outside the
     DiT root. `Keys::key` now spells them so.
4. **DiT.** The 2.3 DiT has a **biased video FFN** (`ff.net.0.proj.bias`,
   `ff.net.2.bias`); 2.5's has none. `Ltx2TransformerConfig::ltx2_23_22b()`
   inherited `ff_bias = false`, which would have dropped 96 bias tensors
   without an error.

The keys and shapes of the DiT under both names, the upscaler, and every
LoRA pair are clean. There are 1660 pairs: rank 384, and rank 32 on the
`to_gate_logits` heads. Each pair targets a dev-pack weight in both layouts
with `B·A` of the weight's shape.

### HQ cell (sol-engine `models/ltx23.toml [official_config]`)

`fv-gpucheck ltx2 gen --model-version 2.3 --hq` (new) runs:

- the dev bundle (`ltx2_23_22b()`: dynamic-shift schedule);
- the distilled LoRA found beside `--dit`, fused at 0.25 for stage 1 and 0.5
  for stage 2, with the unfused base kept in pinned host memory;
- 1920x1088, 241 frames at 24 fps;
- a 15-step res2s stage 1 (29 calls) at CFG 3 and audio CFG 7, then the
  3-sigma stage 2 (0.909375, 0.725, 0.421875);
- seed 42, with the profile's prompt and negative prompt
  (`fastvideo_models::ltx2::hq`).

`FASTVIDEO_LTX2_UPSAMPLER` (new) points at the x2-1.1 upscaler file.

```bash
FASTVIDEO_LTX2_UPSAMPLER=$W/ltx23-dev/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
fv-gpucheck --mode fast ltx2 gen --model-version 2.3 --hq --weights $W/ltx23 \
  --dit $W/ltx23-dev/ltx-2.3-22b-dev.safetensors --prompt "<ltx23 prompt>" --seed 42 \
  --text streamed --dit-offload resident --warm --dense-stage2 --clip out/ltx23-hq
```

The optimized arm swaps `--dense-stage2` for `--pisa-stage2` and adds
`FASTVIDEO_LTX2_STAGE1_CACHE=1 FASTVIDEO_LTX2_MIDPOINT_PRUNE=1
FASTVIDEO_NVFP4=1`. That is `config/ltx23/fullopt.toml` minus its KWL Triton
fusions, which have no counterpart here:

- stage-1 SCSP over res2s calls 16-28;
- stage-2 PISA at sparsity 0.9, block 64, layers 0-1 dense;
- the NVFP4 video FFN;
- the stage-2 feature-norm prune at 0.5 on steps 1-2.

**Published:** 2.40x fullopt / baseline on 1x GB200. There is no absolute
baseline (`[baseline] measured = false`).

**Known drift** (`docs/gaps/2026-09-25-sol-engine-code-level.md`):

- Our stage 1 is the ODE res2s (29 calls). sol-engine's runtime is the SDE
  variant.
- Our stage 2 is 3-forward Euler; theirs is res2s.

Both arms share the drift, so the ratio compares like with like.

## Benchmark arms

`runpod-matrix.sh solbench` (default `FV_CELLS` in bold):

| Cell | What | Reproduces |
|---|---|---|
| **`wan13-sol-base`** | 1.3B, 5 prompts after a warm generation | `config/wan21_t2v_1_3b/baseline.toml` |
| `wan13-sol-easycache` | + EasyCache 0.036 | `cache_only.toml` |
| `wan13-sol-attn` | + Sol-Attn `fullstack` | `wan21_sol_only.toml` |
| **`wan13-sol-fullstack`** | both | `wan21_fullstack_sol.toml` |
| **`a14b-sol-base`** | A14B, `FASTVIDEO_WAN_MOE=auto` (swap on PRO 6000), prompt p0 | `config/wan22_t2v_a14b/baseline.toml` (449.67 s GB200) |
| **`a14b-sol-fullopt`** | + A14B EasyCache + PISA | `singlegpu_opt.toml` (207.01 s GB200) |
| `a14b-sol-mxfp8` | both experts resident at MXFP8 | (fit alternative, lossy) |
| **`ltx23-hq-base`** | HQ, dense stage 2, warm | `config/ltx23/baseline.toml` |
| **`ltx23-hq-fullopt`** | SCSP + PISA + NVFP4 FFN + prune | `config/ltx23/fullopt.toml` (2.40x) |

All Wan cells run with:

- Diffusers sigmas (`FASTVIDEO_WAN_UNIPC_SIGMAS=diffusers`);
- bf16 GEMMs (`FASTVIDEO_WAN_QUANT=off`; on sm_100 it would default to
  mxfp8);
- the full Wan VAE.

`compare_cells` and `gate_cells` compare each arm with its base.

```bash
# RTX PRO 6000 (EUR-IS-1, the EU weight volume), every default cell:
FV_FAMILY=solbench FV_POD_CAP_S=10800 bash scripts/gpu/runpod-http.sh run <sha>
# B200 (both A14B experts resident; needs B200 stock in the volume's DC):
RUNPOD_GPU_TYPE="NVIDIA B200" FV_FAMILY=solbench \
  FV_CELLS="a14b-sol-base a14b-sol-fullopt" FV_EXTRA_ENV="FV_SOL_A14B_MOE=both" \
  FV_POD_CAP_S=5400 bash scripts/gpu/runpod-http.sh run <sha>
```

Per-cell wall caps:

| Variable | Default |
|---|---|
| `FV_SOL_W13_CAP_S` | 1800 |
| `FV_SOL_A14B_CAP_S` | 5400 |
| `FV_SOL_LTX_CAP_S` | 2400 |

`FV_SOL_PROMPTS=5` makes the A14B cells run sol-engine's five prompts after a
warm generation, as the published median does. That is about 6x the A14B
time.

### Estimated GPU time (not measured)

The estimates are scaled from measured cells:

- Wan2.1-14B 480p base on PRO 6000: denoise 477 s for 50 steps (9.5 s/step);
- FastWan 1.3B: denoise 0.4 s per forward;
- the LTX-2.5 B200 / PRO 6000 suites.

| Cells | RTX PRO 6000 ($2.09/h) | B200 ($6.79/h) |
|---|---|---|
| `wan13-sol-base` + `-fullstack` (6 generations each, ~80 s / ~40 s) | ~20 min | — |
| `wan13-sol-easycache` + `-attn` (opt-in) | ~14 min | — |
| `a14b-sol-base` (load ~5 min; ~32 s/step → ~22 min denoise) | ~28 min | ~12 min |
| `a14b-sol-fullopt` (~1.9x on the denoise) | ~18 min | ~7 min |
| `a14b-sol-mxfp8` (opt-in) | ~20 min | — |
| `ltx23-hq-base` + `-fullopt` (load ~4 min, warm + 1 each) | ~16 min | — |
| Pod start, image pull, weight gate | ~10 min | ~10 min |
| **Default cells** | **~90 min ≈ $3.2** | A14B pair **~30 min ≈ $3.4** |
| With every opt-in cell | ~125 min ≈ $4.4 | |

### Weights to fetch (owner approval)

None of these are on any volume. The rows are in `weights-manifest.tsv` and
`weights-revisions.tsv` **commented out**. `verify-weights.sh` has the cells
`wan21-t2v-1.3b`, `wan22-t2v-a14b` and `ltx23-hq`, so the matrix skips cleanly
until the trees exist.

| Tree (dest) | Hub repo @ revision | Files | Size |
|---|---|---|---|
| `wan21-t2v-1.3b` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` @ `0fad780a534b6463e45facd96134c9f345acfa5b` | `model_index.json scheduler/* tokenizer/* text_encoder/* transformer/* vae/*` | 28.94 GB. Alternatively 6.2 GB: transformer 5.68 + vae 0.51 + configs, with `text_encoder/` taken from `wan21-t2v-14b/` (same LFS SHA-256 per shard) |
| `wan22-t2v-a14b` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` @ `5be7df9619b54f4e2667b2755bc6a756675b5cd7` | `model_index.json scheduler/* tokenizer/* text_encoder/* transformer/* transformer_2/* vae/*` | 126.20 GB |
| `ltx23-dev` | `Lightricks/LTX-2.3` @ `3c6a4e66e5d0a684231950b9c74dd4ded7b6fadc` | `ltx-2.3-22b-dev.safetensors ltx-2.3-22b-distilled-lora-384-1.1.safetensors ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 54.75 GB |

The total is **209.9 GB** (187.2 GB with the shared 1.3B text encoder), of
about 557 GB free on the EU volume. Licences:

- Wan: Apache-2.0.
- LTX-2.3: LTX-2 Community License, not gated.
