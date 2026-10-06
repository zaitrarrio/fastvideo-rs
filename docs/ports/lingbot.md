# LingBot-Video — port specification

FastVideo family `lingbot`: Robbyant LingBot-Video, Dense 1.3B and **MoE
30B-A3B** text-to-video, with the **1080p refiner** stage, Qwen3-VL text,
Wan 2.1 VAE and FlowUniPC sampling; plus the sol-engine optimized arm.

Sources (read 2026-10-06):

- Hub `robbyant/lingbot-video-moe-30b-a3b` @ `f2e538f64afe00cc4ae674db2aeb52e2945edfd5`
  (Apache-2.0): `transformer/config.json`, `refiner/config.json` (identical
  but the diffusers version), the 977-key shard index (same for both DiTs) and
  the safetensors headers of shards 1 and 13 — vendored in
  `crates/fastvideo-models/src/lingbot/hub/`.
- NVlabs/Sana `sol-engine`: `models/lingbot_video.toml` (official config,
  published numbers), `models/lingbot_video/{baseline,optimized}/lingbot_src`
  (the vendored upstream LingBot code: `transformer_lingbot_video.py`,
  `pipeline_lingbot_video.py`, `runner.py`, `utils.py`,
  `scheduling_flow_unipc.py`), `config/lingbot_video/*` (the arms).
- transformers v5.8.1 (`utils/output_capturing.py`): `hidden_states[-1]` is
  tied to `last_hidden_state`.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `lingbot_dense_1_3b` | `robbyant/lingbot-video-dense-1.3b` | Dense T2V |
| `lingbot_moe_30b` | `robbyant/lingbot-video-moe-30b-a3b` | MoE 30B-A3B + `refiner/` |

## DiT (`LingBotVideoTransformer3DModel`)

| piece | Dense 1.3B | MoE 30B-A3B (Hub) |
|---|---|---|
| hidden / heads / head_dim | 2048 / 16 / 128 | 2048 / 16 / 128 |
| depth | 24 | **48** |
| FFN | SwiGLU 6144 | **128 experts × SwiGLU 768, top-8**, + **1 shared** SwiGLU 768 |
| router | — | sigmoid scores; selection on `score + e_score_correction_bias`; **group-limited**: 4 groups of 32, each scored by the sum of its two best, top-2 groups eligible; gate = bias-free scores, L1-normalized, **× 2.5**, rounded to bf16 |
| rope | theta 256, axes (32, 48, 48), complex pairs | same, `axes_lens` (4096, 512, 512) |
| params | 1.3B | 30.08B total, 2.9B active per token |

The previous preset (hidden 4096, depth 40, expert width 512, no shared
expert, no group routing, scale 1) and the previous block (separate self /
cross attention, `attn1` / `attn2`, no AdaLN) did not match the checkpoint;
both are replaced.

Block (single-stream over the joint `[video; text]` sequence):

```
mod = time_modulation(silu(t_emb)) + scale_shift_table     # 6 × D, f32
x  += tanh(g1) · RMSNorm_post_attn(attn(RMSNorm1(x)·(1+s1) + b1))
x  += tanh(g2) · RMSNorm_post_ffn(ffn(RMSNorm2(x)·(1+s2) + b2))
attn: to_q/k/v (no bias) → per-head RMS norm_q/k → 3D RoPE → full SDPA → to_out (bias)
out:  LayerNorm(no affine)·(1 + scale) + shift (norm_out_modulation) → proj_out on video tokens
```

Positions: video `(t, y, x)` at `(text_len + 1 + t, y, x)`, text token `i` at
`(i + 1, 0, 0)`. Text embedder: RMSNorm(2560) → Linear-SiLU-Linear. Timestep:
`Timesteps(256, flip_sin_to_cos)` → `TimestepEmbedding` (f32).

## Text / VAE / schedule

| piece | notes |
|---|---|
| Qwen3-VL-4B text (`text_encoder/`) | `PROMPT_TEMPLATE.format(prompt)`, processor tokenizer (`processor/tokenizer.json`), **final-normed** last hidden state (tap 36; the old tap 35 was the pre-norm residual), first `crop_start` tokens dropped (140 for the published processor, recomputed from the tokenizer) |
| negative prompt | base: `DEFAULT_NEGATIVE_PROMPT` (the JSON one); refiner: zeros (`null_cond_clone_zero`) |
| Wan 2.1 VAE | decode; the refiner also **encodes** — `wan::vae21_encode` is the chunked, causal-cached encoder (frame 0, then 4 per pass, `ZeroPad2d(0,1,0,1)` stride-2 downsample) a 1088×1920×121 clip needs |
| base schedule | FlowUniPC (`bh2`, order 2, predict-x0), table `shift=1` (σ 0.999 → 0), `set_timesteps(40, shift=3)` |
| refiner schedule | `compute_refiner_sigmas(steps=8, shift=3, t_thresh=0.85, tail=2)`: the shifted grid kept ≤ 0.85 with 0.85 inserted first, plus 2 linear tail sigmas → **8 steps** |
| transformer timestep | `int64(σ·1000)/1000` rounded to bf16, × 1000 |

## Refiner stage

`runner.py`: base frames → mp4 → read back → `compute_training_frame_budget`
(121 frames at 24 fps: all kept) → bicubic (`align_corners=False`) to
1920×1088, clamp → Wan VAE encode → `(z − mean)/std` → `(1 − 0.85)·x_up +
0.85·noise` → refiner DiT (same architecture, `refiner/` weights) with CFG 3
on the tail → decode.

**Differences from the reference** (intended, recorded):

- the base frames are quantized to u8 in memory instead of an H.264 round trip;
- the VAE encode uses the posterior mean, the reference samples
  (`latent_dist.sample`; the Wan posterior std is tiny);
- noise comes from our seeded generator, not torch's (no bit-identical noise).

## Sol-engine optimized arm (`FASTVIDEO_LINGBOT_SOL` / `--arm`)

Published winner `config/lingbot_video/*cudnn_pisa_easycache_refiner*`
(2.60× = cuDNN attention 1.79× · refiner PISA 1.12× · EasyCache 1.30×):

| technique | upstream | here |
|---|---|---|
| kernel | cuDNN flash attention instead of FA2 | our fastest dense attention in **both** arms (no separate arm) |
| EasyCache | reuse the last CFG output pair while `mean|x−ref|/mean|ref| < thr` (ref = latent at the last computed step), within `[head, n−tail)`, ≤ `max_reuse` in a row; base 0.08 / 4 / 2 / 2, refiner 0.25 / 2 / 1 / 2 | `lingbot::sol::EasyCache`, same rule and numbers |
| PISA | refiner only, density 0.10, block 64, layers 0–3 dense, refiner steps 0, 1 and the last dense | `crate::pisa_attn` (sparsity 0.9) through `AttnRoute`, `PisaPolicy::published()` |
| topology | CP4 Ulysses + FSDP + batched CFG on 4× GB200 | not reproduced (one GPU) |

`fullopt` = EasyCache + PISA; `easycache`, `pisa` select one.

## Memory and residency (one GPU)

| GPU | plan |
|---|---|
| B200 (192 GB) | `--residency both`: base + refiner DiTs resident (120.5 GB bf16) + Qwen3-VL (8.9 GB) + VAE; refiner activations at 253k tokens ≈ 20–30 GB (MoE dispatch chunked, `FASTVIDEO_LINGBOT_MOE_CHUNK`) |
| RTX PRO 6000 (96 GB) | `--residency swap`: text encoder → dropped; base DiT (60.3 GB) → dropped; refiner DiT (60.3 GB). bf16 throughout, no FP8 or expert offload needed; the swap (≈60 GB read per stage, from the page cache after the first request) is timed separately and excluded from `request_s` |

Expert offload is not needed at bf16 on either card. If a 96 GB run ever
needs both DiTs resident, the next step is weight-only FP8 experts
(`Linear::load_fp8_rows`) — not implemented here.

## Comparison framing

The published numbers are **4× GB200** (CP4 + FSDP + batched CFG), timed from
both models resident to the refined mp4 exported (load excluded, 3-prompt
median, not warm). This port runs **one GPU**, so the comparison is:

- **like for like on work**: same canvases, steps, guidance, refiner tail,
  prompts (`scripts/gpu/sol/lingbot-t2v-val3.txt`), seed 42;
- **reported as**: our single-GPU `request_s` median (text encode + both
  stages + exports, loads excluded) next to the published 375.53 s / 144.36 s,
  **and** normalized to GPU-seconds (ours × 1 vs theirs × 4 =
  1502 / 577 GPU-s) — the per-GPU comparison is the meaningful one;
- the **arm ratio** (baseline / fullopt on the same box) against their 2.60×,
  noting that 1.79× of theirs is the FA2 → cuDNN kernel swap, which our
  baseline already includes (expect our ratio ≈ 1.45× = 1.12 · 1.30 if the
  techniques transfer).

## Tests (CPU)

- `fastvideo-models::lingbot::config` — preset = Hub `config.json`; every
  expected key/shape = the 977-key Hub index and the two shard headers;
  parameter counts (30.08B / 2.9B active).
- `routing` — group-limited routing, bias asymmetry, bf16 gate rounding,
  stable dispatch; the device kernel `moe_group_topk` mirrors it
  (`fv-gpucheck sol lingbot-router` checks it on a GPU).
- `rope`, `schedule` (base σ, refiner σ against `compute_refiner_sigmas`,
  bf16 timestep), `refiner` (frame budget, bicubic, start latent), `sol`
  (arm parsing, EasyCache rule, PISA guards).
- `fastvideo-cudarc::lingbot::transformer::golden_tiny_moe_matches_numpy_reference`
  — the Rust DiT on random tiny-MoE weights vs `scripts/ref/lingbot_reference.py`,
  a NumPy transcription of the upstream forward: max |Δ| < 2e-4. Plus
  MoE dispatch = per-token reference, chunking invariance, PISA at density 1
  = dense, a two-stage sampling smoke test with CFG / EasyCache / PISA.
- `wan::vae21_encode` — layout vs Diffusers' down blocks, chunked shapes.

## Weights

`weights-manifest.tsv` row `lingbot-video-moe-30b-a3b` (proposed, not on a
volume): `transformer/` 60.27 GB, `refiner/` 60.27 GB, `text_encoder/`
8.89 GB, `vae/` 0.51 GB, `processor/` + `scheduler/` ≈ 0.01 GB = **129.95 GB**.
Fetch (owner approval first): `scripts/gpu/fetch-hub-tree.sh
lingbot-video-moe-30b-a3b f2e538f64afe00cc4ae674db2aeb52e2945edfd5`; verify
cell `lingbot-moe`.

## GPU benchmark arm (not run)

`FV_FAMILY=sol-lingbot scripts/gpu/runpod-http.sh run <sha>` (runpod-matrix
family `sol-lingbot`): `fv-gpucheck sol lingbot-router`, then
`sol lingbot-gen --arm baseline` and `--arm fullopt` over the three prompts.
Knobs: `FV_LINGBOT_RESIDENCY` (`swap` default, `both` on B200),
`FV_LINGBOT_PROMPTS` (3), `FV_LINGBOT_ARMS`, `FV_SOL_TIMEOUT_S`. B200:
`RUNPOD_GPU_TYPE="NVIDIA B200" RUNPOD_GPU_MAX_DPH=8`. Estimates in
docs/perf/sol-lingbot-cosmos3-plan.md.

## Port status

| layer | status |
|---|---|
| MoE preset = Hub config, key map = Hub index | landed, CPU-tested |
| DiT (dense / MoE, group routing, shared expert) | landed, golden-tested on CPU |
| Qwen3-VL text (tap, tokenizer path, crop) | landed (needs a GPU run) |
| base FlowUniPC + CFG | landed |
| refiner (upsample, chunked Wan 2.1 encode, tail schedule) | landed |
| EasyCache / PISA arm | landed |
| device router kernel / MoE combine kernel | landed, type-checked; GPU check pending |
| GPU run | **not run** (no weights on the volume) |
