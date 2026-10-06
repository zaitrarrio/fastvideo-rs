# LingBot-Video MoE and Cosmos3-Super vs sol-engine — GPU plan (not run)

Status 2026-10-06: both ports are code-complete and CPU-tested
(docs/ports/lingbot.md, docs/ports/cosmos3.md); **nothing has run on a GPU**
and **no weights are on a volume**. This page is the download list, the
benchmark arms and the cost estimate the owner approves before anything is
spent.

## 1. Downloads (owner approval needed; EU volume `jg48s6o1w0`, ~557 GB free)

| dest (`weights-manifest.tsv`) | Hub repo @ revision | licence | size | contents |
|---|---|---|---:|---|
| `lingbot-video-moe-30b-a3b` | `robbyant/lingbot-video-moe-30b-a3b` @ `f2e538f64afe00cc4ae674db2aeb52e2945edfd5` | Apache-2.0 | 129.95 GB | `transformer/` 60.27, `refiner/` 60.27, `text_encoder/` 8.89, `vae/` 0.51, `processor/`, `scheduler/`, `model_index.json` |
| `cosmos3-super` | `nvidia/Cosmos3-Super` @ `f543c56225b2e04d0ad141e29655be3a45d9c455` | OpenMDW 1.1 (not gated) | 129.47 GB | `transformer/` 128.04 (27 shards), `vae/` 1.41, `text_tokenizer/`, `scheduler/`, `model_index.json` |
| **total** | | | **259.4 GB** | leaves ≈ 298 GB free |

Not needed for T2V: Cosmos3 `vision_encoder/` (1.19 GB), `sound_tokenizer/`
(1.99 GB), `assets/`. Neither VAE is reusable from the volume (LingBot's is
the Wan 2.1 VAE but stored as its own file; Cosmos3's bf16 Wan 2.2 VAE differs
byte-wise from the TI2V-5B f32 file).

Fetch, one tree at a time (CPU pod, add-only, LFS SHA-256 checked, backstop):

```bash
scripts/gpu/fetch-hub-tree.sh lingbot-video-moe-30b-a3b f2e538f64afe00cc4ae674db2aeb52e2945edfd5
scripts/gpu/fetch-hub-tree.sh cosmos3-super f543c56225b2e04d0ad141e29655be3a45d9c455
```

Then record the revisions in `weights-revisions.tsv`, the hashes in
`weights-sha256.tsv`, and move the manifest rows out of the "PROPOSED" block.

## 2. Arms

Both run as runpod-matrix families through `runpod-http.sh` on an image built
from this branch (the published image does not have `fv-gpucheck sol`).

| family / cell | what | published reference |
|---|---|---|
| `sol-lingbot` / `lingbot-router` | device group-limited router vs host on 65k random rows (no weights) | — |
| `sol-lingbot` / `lingbot-baseline` | base 832×480×121, 40 steps + refiner 1920×1088, 8 steps, CFG 3, 3 prompts, dense attention | 375.53 s (4× GB200, FA2) |
| `sol-lingbot` / `lingbot-fullopt` | + EasyCache (base + refiner) + refiner PISA 0.10 | 144.36 s (2.60×, incl. FA2→cuDNN 1.79×) |
| `sol-cosmos3` / `cosmos3-baseline` | 1280×720×189, 35 steps, CFG 6, warmup + 1 timed | 130.41 s (4× GB200) |
| `sol-cosmos3` / `cosmos3-teacache` | + TeaCache 1.15 / 10 / 3 | — |
| `sol-cosmos3` / `cosmos3-teacache-fp8` | + `FASTVIDEO_FP8=1` (W8A8, all steps; theirs: NVFP4 middle steps) | 2.26× |

Commands (B200; RTX PRO 6000 drops the GPU type and the residency):

```bash
FV_FAMILY=sol-lingbot RUNPOD_GPU_TYPE="NVIDIA B200" RUNPOD_GPU_MAX_DPH=8 FV_POD_CAP_S=7200 \
  FV_EXTRA_ENV="FV_LINGBOT_RESIDENCY=both" scripts/gpu/runpod-http.sh run <sha>
FV_FAMILY=sol-cosmos3 RUNPOD_GPU_TYPE="NVIDIA B200" RUNPOD_GPU_MAX_DPH=8 FV_POD_CAP_S=5400 \
  FV_EXTRA_ENV="FASTVIDEO_COSMOS3_UND=resident" scripts/gpu/runpod-http.sh run <sha>
# RTX PRO 6000: FV_FAMILY=sol-lingbot FV_POD_CAP_S=14400 scripts/gpu/runpod-http.sh run <sha>
```

Cheaper first pass: `FV_EXTRA_ENV="FV_LINGBOT_PROMPTS=1"` (one prompt per
arm) and `FV_EXTRA_ENV="FV_COSMOS3_ARMS=baseline"`.

## 3. Estimates (FLOP-based, not measured)

Work per request, from the configs: LingBot base ≈ 1.2 PFLOP per DiT forward
(48 384 tokens; attention 0.95, MoE + projections 0.28) × 80 forwards ≈ 98
PFLOP; refiner ≈ 26.8 PFLOP per forward (253 680 tokens; attention 95 %) × 16
forwards (8 steps × CFG) ≈ 430 PFLOP. Cosmos3 ≈ 6.9 PFLOP per gen-tower
forward (44 160 tokens; linears 2.8, attention 4.1) × 70 ≈ 480 PFLOP.
Assumed sustained throughput of our kernels: **B200 ≈ 1.0 PFLOP/s, RTX PRO
6000 ≈ 0.3 PFLOP/s** bf16 (to be replaced by measurements). Overheads: pod
start and image pull ~10 min, weight load ~5 min per 60 GB from the volume.

| run | B200 GPU-min | B200 $ ($6.79/h) | PRO 6000 GPU-min | PRO 6000 $ ($2.09/h) |
|---|---:|---:|---:|---:|
| LingBot baseline, 1 prompt (≈ 9.5 / 31 min) + fullopt, 1 prompt (≈ 4.5 / 15 min) + overhead | ≈ 35 | ≈ 4.0 | ≈ 70 | ≈ 2.4 |
| LingBot, 3 prompts per arm | ≈ 60 | ≈ 6.8 | ≈ 175 | ≈ 6.1 |
| Cosmos3 baseline only (warmup + 1) | ≈ 25 | ≈ 2.8 | ≈ 50 | ≈ 1.7 |
| Cosmos3, 3 arms | ≈ 40 | ≈ 4.5 | ≈ 95 | ≈ 3.3 |

Expected single-GPU results if the estimates hold: LingBot baseline ≈ 570 s
(B200) / ≈ 1850 s (PRO 6000) per prompt vs 375.53 s on 4× GB200 (1502
GPU-s); Cosmos3 baseline ≈ 510 s (B200) / ≈ 1700 s (PRO 6000) vs 130.41 s on
4× GB200 (522 GPU-s).

Mind the $8 balance floor (CLAUDE.md): the 3-prompt LingBot run on a B200
alone is close to it; run the 1-prompt passes first and only one pod at a
time. B200 stock in EUR-IS-1 (where the volume is) is unconfirmed; with no
B200 there, the PRO 6000 numbers apply.

## 4. Risks to check on the first GPU run

- `lingbot-router`: fast-math sigmoid can flip a near-tie; the check allows
  0.1 % of rows. Larger: run with `FASTVIDEO_LINGBOT_ROUTER=host`.
- LingBot refiner peak memory at 253k tokens (attention via our dense kernel;
  MoE dispatch chunk 32 768 tokens): watch `peak_mib`; lower
  `FASTVIDEO_LINGBOT_MOE_CHUNK` if needed.
- Wan 2.1 VAE encode at 1088×1920 (new chunked encoder): first real use.
- Cosmos3: the text tower streaming time on a 96 GB card (31 GB read per
  request) and the Qwen2 tokenizer's handling of the chat special tokens.
- Cosmos3 `flow_shift=10` is inert under the Hub scheduler config in
  Diffusers; if SGLang applies it, the step grids differ (the published
  timings would not change, the frames would).
