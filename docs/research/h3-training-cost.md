# Producing our own H3 variant: time, effort and cost

Owner question (2026-10-14): look at FastVideo's (and Sol-engine's) training code to understand the time, effort and
cost of producing our own MiniMax H3 variant: pruned, step-distilled or fine-tuned. This is desk research only:
nothing was trained, built or run on a GPU, and no weights were downloaded (model cards, configs and source files
only).

**Short answer.** The cheapest thing with a real chance of being useful is a **data-free DMD2 LoRA** (option a). It
fits on one 8×H100 80GB pod. A 300-step pilot costs about **$250** (about 60 H100-hours). A full run costs about
**$1.6k-2.4k** (about 470-610 H100-hours, 2.5-3 days of wall-clock time). The engineering is **8-12 person-days**,
because working H3 LoRA-DMD trainers already exist (LongLive-Plug's source snapshot, VDN's Stage-DMD). Two other
options are FastVideo-scale jobs: full-weight step distillation (b) and pruning plus healing (c). Each needs 32 GPUs
across nodes, about **750-1,000 B200-hours a run** (roughly **$4-8k a run, $10-25k with retries**) and 20-40
person-days, and the H3 DMD model class they need is not public. Fine-tuning on our own data (d) is cheap to compute
but expensive in data, and it gives a 50-step model, so it needs (a) as well. NVFP4 QAT (e) has the weakest case:
FastVideo reports that it did not beat PTQ on H3, and there is no H3 recipe. **Recommendation (§5):** first a
$50-capped throughput probe, then a $300-capped LoRA-DMD pilot. Only go further if the pilot beats the turbo LoRAs we
already serve on our gate, or if we need a variant that nobody publishes, such as a turbo LoRA for I2V/FL2VA tuned to
our shifts, or one for the Trim.

Pinned sources (all claims below cite these):

| key | source | revision |
|---|---|---|
| `FV/` | github.com/hao-ai-lab/FastVideo (Apache-2.0) | `d5253fd71055927f12aa6c4aa679b93a00238851` (2026-10-09) |
| `VDN/` | github.com/OpenVDN/vdn-minimax-h3 (code Apache-2.0) | `e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b` (2026-10-04) |
| `SANA/` | github.com/NVlabs/Sana, branch `sol-engine` | `62da34b14a53d7289e82fa56692f34e96e089bee` (2026-10-11) |
| `HF:base` | MiniMaxAI/MiniMax-H3 | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| `HF:v2` | FastVideo/FastVideo-FastH3-8-Step-V2 | `3da2ddfe1954d9cda4c05b643dc0f26007a655c5` |
| `HF:lora` | FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA | `f509e629374cac104e7f62daecce6d1488a3041d` |
| `HF:trim` | FastVideo/FastVideo-FastH3-Trim-8-Step | `9edcb6511d5ff8dd1579e8eec8afee77e098b8d8` |
| `HF:trimc` | FastVideo/FastVideo-FastH3-Trim-Comfy | `f66c13dc17391dd4da99242a407918b92bf7d5fc` |
| `HF:v2fp4` | FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4 | `0aa5247ba4a5d3f9f4fdb99536d8cf55bdc5e2f0` |
| `HF:vsafp4` | FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4 | `d7adcabeb19a81bb79cf9c4a04f5e366f1f2e90a` |
| `HF:llfew` | Efficient-Large-Model/LongLive-Plug-MiniMax-H3-few-step | `b3686f432d3f20d6d6462a91d4c09ba7f189c0a2` |
| `HF:llcfg` | Efficient-Large-Model/LongLive-Plug-MiniMax-H3-cfg | `de1f4e8c7310917a7bf0417ff041abb8bb7b0927` |
| `HF:vdn` | OpenVDN/vdn-minimax-h3 | `2ce18deb1d1c6b115d6bb3732f3750fdbac95173` |
| `HF:larry` | larryvrh/MiniMax-H3-Turbo-Lora | `43a74557ac3f6539db8e0f2a959d03feb7a81480` |
| `HF:lx2v` | lightx2v/Minimax-h3-Turbo | `3ec17a324ced54151364f24f8b5fb6bf7e26414f` |
| blogs | haoailab.com/blogs/fasth3-preview/, /blogs/fasth3-rtx/; openvdn.github.io; arXiv 2609.20744 (VDN) | read 2026-10-14 |

The Hub revisions are the ones in `control/src/weights-sources.ts` and `scripts/gpu/weights-manifest.tsv`.

---

## 1. Inventory: what training code exists

### 1.1 The base model we would train from

- **Size.** The H3-Omni-Transformer is a "33B-parameter dense, single-stream Transformer, with approximately 13B
  parameters residing in AdaLN-related branches", and the full weights are released "to support further development,
  including fine-tuning" (`HF:base` README.md:142). On disk it is 70.10 GB of bf16, 26.02 GB of which is AdaLN
  (docs/ports/h3.md, correction 7). Each block has width 5376, attention inner width 7168 (56 × 128), a gated FFN of
  14336, and there are 50 blocks plus 2 token-refiner blocks (docs/ports/h3-trim.md §1). That gives about 385 M
  matmul parameters a block and **19.3 B in the 50 blocks**. The AdaLN (13 B) runs once per timestep, not once per
  token, so it costs almost no compute but costs full memory when trained.
- **Other components** (all frozen in every option here): the Qwen3-VL-32B text encoder, read at `hidden_states[50]`
  (`HF:base` README.md:118), the video VAE (ViT decoder, 36 blocks) and the audio VAE.
- **Licence** (`HF:base` LICENSE). The *MiniMax H3 Community License* explicitly allows Model Derivatives
  ("create derivative works (including Model Derivatives), and modify", §II, line 22). Distillation and training on
  H3's synthetic outputs are themselves Model Derivatives (§I.11, line 16). The conditions:
  - Modified files must carry a notice of modification (§III.2).
  - A `NOTICE` file must ship with any distribution other than a hosted service (§III.4).
  - A commercial product must show "Powered by MiniMax H3" and "MiniMax H3" in its UI (§III.3, §IV.2).
  - Revenue over US$20 M a year needs written authorization (§IV.1).
  - H3 outputs must not be used to improve any non-H3 model (§V.3). Synthetic teacher rollouts may therefore train
    only H3 derivatives, which is all that is proposed here.
  - The licence does not apply in the EU, UK, Republic of Korea or USA (§I.5). The owner states that he holds a
    MiniMax licence that covers those territories (docs/ops/open-items.md, 2026-10-11), but the document is not in
    the repo. **Before we train, confirm that it covers creating and serving our own derivatives, not only serving
    published ones.**

### 1.2 Code by source

| source | what it trains for H3 | licence | runs H3 out of the box? |
|---|---|---|---|
| **FastVideo** `FV/fastvideo/train/` (the new modular trainer) | **SFT only.** `MiniMaxH3Model` (full) and `MiniMaxH3LoraModel` (`FV/fastvideo/train/models/minimax_h3/minimax_h3.py:50, :429`); flow-matching `FineTuneMethod` | Apache-2.0 | **Yes for SFT**, with restrictions: SDPA dense attention only (`:81`, `:355`), `train_batch_size=1` (`:87`), no CFG dropout (`:91`), and `predict_noise` refuses an unconditional branch (`:353`), so DMD2's `cfg_uncond` cannot be used with it. Validated only as a 400-step single-sample overfit on **64 H200 (8 nodes, sp 8)** at 1344×768×124 (`FV/examples/train/configs/overfit_minimax_h3_t2va{,_lora}.yaml`, `FV/examples/train/launch_minimax_h3_t2va_crush_smol_validation.sh:43,67`). Ref2VA SFT exists too (`minimax_h3_ref2va.py`) |
| FastVideo DMD2 / sparse distillation | generic `DMD2Method` (`FV/fastvideo/train/methods/distribution_matching/dmd2.py`); public configs for **Wan** and Kandinsky only (`FV/examples/train/configs/distribution_matching/{wan,kandinsky5}/`) | Apache-2.0 | **No.** The H3 DMD runs that produced FastH3 used `fastvideo.train.models.minimax_h3.MiniMaxH3DMDModel` together with method options `rollout_carry`, `audio_anchor`, `modality_dmd_safeguards`, `score_timestep_shift` and `fake_score_loss_space` (`HF:trim` metadata.json, `HF:vsafp4` checkpoint_metadata.json). **None of these is in public `main`** (grep of `FV/` at `d5253fd`: no matches). The FastH3 4-step run records `fastvideo_commit 48a047c…` (`HF:vsafp4` provenance.json). It could not be checked here (GitHub's API was not reachable from this session); docs/ports/h3.md found V2's recorded commit not publicly reachable |
| FastVideo pruning ("Trim") | `FV/scripts/checkpoint_conversion/convert_minimax_h3_adaln_rank.py` (training-free SVD of the AdaLN to rank r; its docstring says it removes "~39% of the model's parameters") | Apache-2.0 | The AdaLN step is yes. **Block selection and the healing DMD are not published.** The method is described only in the blog (§1.3) |
| FastVideo LoRA extraction | `FV/scripts/lora_extraction/extract_lora.py` (+ README) | Apache-2.0 | **Yes.** It is how the FastH3 "LoRA" repo was made: a rank-64 randomized SVD of the full FastH3 fine-tune against the base, with exact deltas for the input/output projections and `.set_weight` for the VSA gates (`FV/scripts/lora_extraction/README.md`, "For large transformers"). So **FastH3's adapters were extracted from full fine-tunes, not trained as LoRAs** |
| FastVideo QAT | `ATTN_QAT_TRAIN` plus NVFP4 linears with STE (`FV/examples/train/configs/fine_tuning/ltx2/nvfp4_qat_t2v.yaml`, `ltx2_3/`, `kandinsky5/`; `FV/docs/training/attn_qat.md`) | Apache-2.0 | **No H3 config.** The plugin would need the QAT backend wired in, and the H3 plugin accepts only SDPA |
| **LongLive-Plug** `HF:llfew` `source_snapshot/minimax_h3/*.py`, `training_recipe/*.py` | **A complete H3 LoRA-DMD trainer**: `train_dmd_longrun_{4,8}step.py`, `dmd_core.py`, `train_utils.py` (FSDP FULL_SHARD, CPU offload of the frozen base, saved-tensor CPU offload), plus a CFG-distillation trainer (configs only, in `HF:llcfg`) | `LICENSE-CODE` Apache-2.0 (weights: H3 Community) | **Close.** It targets the diffusers H3 at base revision `fa9c8ab…`, torch 2.11 + CUDA 13.0, diffusers 0.40.0.dev0 and peft 0.20 (`environment/observed_versions.json`). It is a snapshot from an internal tree whose paths point at a Lustre filesystem, so expect a day or two of environment repair |
| **OpenVDN** `VDN/src/training/train_stage_dmd.py`, `dmd.py`, `fsdp_stage.py` | H3 LoRA-DMD (DMD2, no GAN), one FSDP2 model carrying three roles (generator, real score, fake), LoRA rank 128; the real score is `dense` (the base) or `vdn` (`VDN/src/config/stage_dmd.py:3-13, :40`) | Apache-2.0 (weights: H3 Community) | **Close for LoRA-DMD.** The trainer requires a VDN Stage-B seed (`train_stage_dmd.py:109-114`), so a dense-only run needs that check and the linear branch removed. It also has a ready latent dataset format (`VDN/README.md` "Data Preprocess") |
| lightx2v `ModelTC/Minimax-H3-Turbo` | none published (inference only; the README gives no method, compute or data) | Apache-2.0 | n/a |
| larryvrh turbo LoRA | none published (`HF:larry` README: checkpoints at "step" 150/200/500/600/850, an EMA, "training continues") | Apache-2.0 (weights) | n/a |
| **Sol-engine** `SANA/` | `models/streamrender_h3/render/vendor/dev/yanzuolu/projects/minimax_h3*/meta_models/*_streaming_{sft,distill}.py` and `streaming_sft.md` describe a streaming (windowed, causal) SFT/distillation of H3. They plug into `dev.yanzuolu.engines.diffusion_finetuning.DiffusionFinetuning` (`streaming_sft.md:10-12`), **which is not in the tree** (no `engines/` among the 193 vendored files). The rest of the H3 code is inference: `models/minimax_h3/{Sol-H3,HyperFlow,GB200,H100,...}`, LoRA *fusion* only | the branch has no root LICENSE file (Sana `main` is Apache-2.0; the per-model `licenses/` hold upstream notices) | **No.** These are inference-side fragments of a training system that was not released. Useful only as a design reference for streaming SFT |

### 1.3 What is documented about how our served checkpoints were made

| checkpoint we serve | method | GPUs and settings | steps | data | compute documented |
|---|---|---|---|---|---|
| FastH3 4-step Preview VSA/Data-Free (and its LoRA, its NVFP4) | full-weight DMD2, student VSA 0.9 / tile 64, teacher and critic dense FLASH_ATTN, `cfg_uncond.text: zero`, `real_score_guidance_scale 1.0`, generator every 5 critic steps, lr 2e-6, fp32 master (`HF:vsafp4` checkpoint_metadata.json) | **32 GPUs, sp 4, HSDP shard 32, grad-accum 8** (so a global batch of 8 × 8 = 64 samples), torch.compile | released at **1300** of `max_train_steps 4000` | **text-only** (prompt embeddings) of five preprocessed sets: `nuva_50k_720_mixed_len`, `video_5s_768p`, `nuva_10k_720_mixed_len`, `nuva_10k_mixed_res_len`, `fastgen_vidprom_150k`, native-shape bucketing, base 768×1344×124 | blog: "**1k+ B200 training hours**" (the GPU type is named only there; no count). The LoRA is a rank-64 SVD extraction (§1.2) |
| FastH3 8-Step V2 | "data-free DMD2 and VSA-H3 at 80% sparsity", step 1300 (`HF:v2` README) | not on the card; the blog thanks NVIDIA FastGen for "the score clock, modality shifts, and backward simulation" | 1300 | data-free | not documented |
| FastH3 Trim 8-step (and its NVFP4) | 8 least-important blocks removed (6, 7, 9, 13, 15, 16, 22, 23), AdaLN rank-16 SVD, then 8-step DMD2 "with the FastH3 V2 objective" (blog). Config: student `initial-student` (the pruned net), teacher and critic = base H3, VSA 0.8, `audio_anchor.weight 0.1` (`HF:trim` metadata.json) | **32 GPUs, sp 4, HSDP 32, grad-accum 8** (run name `vsa80-32gpu-anchor-v2`), fp32 master, lr 2e-6 | released at **300** of `max_train_steps 1500` | the same five sets as T2AV latents (`preprocessed_data_type t2va`, so real/synthetic video latents, not text only) | not documented. Block ranking was 600 forward measurements (4 example types × 3 noise levels, each block skipped once) (blog) |
| FastH3 NVFP4 (V2, Trim, 4-step) | **post-training quantization**: ModelOpt NVFP4 W4A4 on the MLPs (plus the VAE decoder for V2), static activation scales calibrated on "1,000 prompts with complete eight-forward denoising trajectories" (`HF:v2fp4` README). For the Trim: "Quantization-aware distillation (QAD) did not beat post-training quantization in our side-by-side comparisons" (fasth3-rtx blog) | calibration only | 0 | 1,000 prompts | not documented (calibration is inference-only) |
| LongLive-Plug H3 few-step LoRA | LoRA-DMD (rank 128 / α 128, 312 modules: attn q/k/v/out + FFN of the 50 blocks and 2 refiner blocks; no AdaLN), 4 NFE, `predict_x0_fresh_renoise`, teacher = adapter-free base, no CFG; an extra audio objective regresses to **50-step frozen-base Euler endpoints** (`HF:llfew` training_config.json: `audio_objective.*`) | **16 GPUs (4 nodes × 4), FSDP FULL_SHARD, frozen base CPU-offloaded**, per-rank batch 1, global batch 16, 768×1344×124 (`distributed_contract.*`). The GPU is H100-class: "The released 33B checkpoint runs close to the H100 memory ceiling" (`source_snapshot/minimax_h3/train_utils.py:845`) | **1000** outer steps = **1000 critic + 200 student updates** (`provenance.json`) of a planned 1500 | **prompts only**: 24,000 conditions drawn from `prompts/vidprom_filtered_extended.txt` (248,221 rows, from `Efficient-Large-Model/LongLive-1.3B`) mixed 50/50 with 256 authored audio prompts over 32 scenes | not documented (no wall-clock or GPU type in any file). Self-assessment: `near_Base50_audio_quality_proven: false`, and the card says not to combine it with the CFG LoRA |
| LongLive-Plug H3 CFG LoRA | CFG-only distillation, offline teacher-trajectory regression toward `v_uncond + 3(v_cond - v_uncond)`, rank 128 (`HF:llcfg` training_config.json) | world 16, global batch 16 | **250** | 64 full guided teacher trajectories × 49 evaluations = 4,032 records | not documented |
| VDN-H3 (hybrid attention, 50 and 8 step) | stages A1 (200 steps), A2 (500), B (LoRA + linear branch, 2000), then DMD (8-step `turbo` LoRA, **initialized from larry's v4 step-600 EMA LoRA**, fake score rank 128, 3 fake updates a step) (`VDN/README.md` Training Recipe; `VDN/configs/training/stage_dmd_vdn.yaml`; `HF:vdn` stage-dmd-step-250/metadata.json) | DMD: "**4 nodes x 8 GPUs**" (`VDN/scripts/training/stage_dmd_vdn.sh`), FSDP2 shard 8, activation offload; the paper says only "H200 and B200 clusters" | DMD **250** | **10,015 clips** at 1344×768, 345 frames (14.4 s) (arXiv 2609.20744); DMD reads only the prompts | not documented |
| larry / lightx2v turbo LoRAs | not documented (lightx2v: trained at 544p / 768p, shifts 12/3 and 6/3, 4 and 8 NFE) | not documented | larry: up to step 850 (v1), 600 (v4) | not documented | not documented |

**What this tells us.** All three published H3 step distillations are DMD2 against the frozen base, run data-free
(prompt embeddings only) at 768×1344×124. The full-weight ones (FastH3) used 32 GPUs with a global batch of 64 for
300-1300 steps. The LoRA ones (LongLive, VDN) used 16-32 GPUs with a global batch of 16-32 for 250-1000 steps.
**The only compute figure published anywhere is FastVideo's "1k+ B200 training hours" for the FastH3 Preview family.**

---

## 2. Per-option estimates

### 2.0 The cost model (estimates; the arithmetic is shown so it can be re-run)

**One H3 forward at 768p, 5 s (`F`).** The video latent is 37 × 48 × 84 with a 2 × 2 spatial patch, which gives
37 × 24 × 42 = 37,296 tokens. Audio adds 124/24 s × 40 Hz ≈ 207 tokens and the text about 256, so N ≈ 37,759 rows.

- Linear: 2 × 19.27e9 × 37,759 = 1.46e15 FLOPs.
- Attention: 4 × N² × 7168 × 50 = 2.04e15 FLOPs.
- **F ≈ 3.5e15 FLOPs**, 58% of it attention. 480p is about 4× cheaper, and 10-15 s clips are 3-8× dearer.

As a sanity check, our measured dense MXFP8 forward on an RTX PRO 6000 is 8.1-8.2 s at 768p
(docs/serve/e2e/h3-test-round-2026-10-14.md, base rows), which implies 0.43 PFLOP/s.

**Effective training throughput (estimate).** H100 SXM: 0.35 PFLOP/s, about 35% of 989 dense bf16, which gives
**10 s/F**. B200: 0.75 PFLOP/s, about 33% of about 2.25, which gives **4.7 s/F**. H200 is taken equal to H100 in
compute.

**Calibration against the one documented number.** FastH3 4-step was 1300 steps × 64 samples × about 7 F (below)
= 582k F, which is about 2.0e21 FLOPs. At 4.7 s/F that is **about 760 B200-hours**. The blog says "1k+ B200 training
hours" for the family, which also includes the dense, synthetic-1300 and synthetic-1900 runs. So the model is about
right, or optimistic by up to about 2×. **The first thing the pilot (§5) buys is the real seconds per step, which
replaces this assumption.**

**Forwards per training sample (estimate).** A forward plus backward with full activation checkpointing is about 4 F.

| step type | per sample |
|---|---|
| DMD2 critic step | student no-grad rollout to a random stage (about 1.5 F at 4 steps, about 3.5 F at 8) + critic fwd/bwd 4 F |
| DMD2 generator step, every 5th | rollout + student fwd/bwd 4 F + teacher 1 F (2 F with a CFG teacher) + critic 1 F |
| **DMD2 outer step, average** | **4-step: 1.5 + 4 + (1.5 + 6)/5 ≈ 7 F; 8-step: 3.5 + 4 + (3.5 + 6)/5 ≈ 9.4 F** |
| SFT step | 4 F |

LoRA does not reduce this much: the backward still carries activation gradients through every block. LoRA saves
memory, not FLOPs.

**Memory (arithmetic).** Full-weight training with an fp32 master (as FastVideo runs it: `dit_precision: fp32`)
holds about 16 B/param: 4 B param, 4 B grad and 8 B AdamW. For 33 B params that is **about 530 GB per trainable
role**. DMD trains two roles (student and critic), plus a frozen bf16 teacher (66 GB), so about **1.12 TB before
activations**.

- **8×H100 80GB (640 GB): does not fit.**
- **8×B200 (about 1.44 TB): fits on paper**, with about 40 GB a GPU left for activations, which is too tight to plan
  on. Hence FastVideo's 32 GPUs.

LoRA-DMD with a frozen bf16 base FSDP-sharded across 8 ranks needs about 8 GB a GPU of base weights. Two rank-128
adapters are about 1.4 B params, so about 22 GB in fp32 with Adam, which is 3 GB a GPU. One 37.8k-row sample's
checkpointed block inputs are about 50 × 37.8k × 5376 × 2 B ≈ 20 GB. **That fits on 80 GB**, and LongLive trained on
H100-class GPUs with CPU offload (`train_utils.py:845`).

**Prices** (on-demand, read 2026-10-14):

| provider | H100 SXM | H200 | B200 | RTX PRO 6000 | source |
|---|---:|---:|---:|---:|---|
| Runpod pods (main pricing table) | $3.99 | $5.29 | $7.99 | $2.49 | runpod.io/pricing. The same page's structured data lists Secure Cloud at H100 $3.49, H200 $4.59, B200 $6.79. Our own H100 pods were billed $3.99 (docs/ops/r2-bench-2026-10-12.md) and RTX PRO 6000 $2.49 (h3-test-round) |
| Runpod Instant Clusters (multi-node, up to 64 GPUs) | contact sales | $4.31 | contact sales | | runpod.io/pricing |
| GMI Cloud | from $2.60 | from $3.20 | from $5.00 ("Limited Availability") | | gmicloud.ai/pricing ("from" prices; not labelled on-demand or reserved) |
| Brev | not priced | | | | Brev has no public price list. `brev search` gives live per-type prices (docs/serve/deploy-gmi-brev.md §3), and it was not run for this research |

The dollar figures below use **Runpod's $3.99 (H100) / $7.99 (B200)** and GMI's "from" prices. A **+30%
overhead** for loading, validation renders, checkpoints and idle time is added where noted. Runpod caps a single pod
at 8 GPUs, so 16-32-GPU jobs need Instant Clusters or GMI.

### 2.1 (a) LoRA turbo / step-distillation adapter (DMD2, data-free)

| | |
|---|---|
| GPUs | **one 8×H100 80GB pod** (or 8×H200 / 8×B200). Data parallel 8, one 768p sample a rank, FSDP-sharded frozen base, saved-tensor CPU offload (LongLive's layout at half its world size). No B200 needed. 480p pilots fit easily |
| Pilot | 300 steps × 8 samples × 7 F = 16,800 F × 10 s = **47 H100-h** (+30% ≈ 60) → **about 7.5 h wall-clock**, **about $240** at Runpod (about $160 at GMI) |
| Full run | 1,500 steps × 16 samples (grad-accum 2) × 7 F = 168,000 F → **467 H100-h** (+30% ≈ 610) → **2.5-3.2 days** on 8 GPUs → **$1.9k-2.4k** at Runpod / **$1.2k-1.6k** at GMI. On 8×B200: about 220-285 B200-h → about 1.2-1.5 days → $1.8k-2.3k at Runpod |
| Warm start | VDN reached its released turbo in **250 steps** by starting from larry's LoRA. Starting from a LoRA we already serve (larry, Apache-2.0; lightx2v, Apache-2.0; FastH3's extracted LoRA, H3 licence) and running 250 × 16 × 7 F = 28k F → **about 80-100 H100-h, about $320-400** |
| 8-step instead of 4 | × 1.34 (9.4 F per sample) |
| Data | **No videos.** Text embeddings for 10k-50k prompts. LongLive used 24k from a 248k-row VidProM-derived list. **VidProM's prompts are licensed CC BY-NC 4.0 (WenhaoWang/VidProM dataset card), so for commercial use write or license our own prompts.** Pre-encoding with Qwen3-VL-32B costs about 1 s a prompt on an H100, so 25k prompts ≈ 7 GPU-h |
| Engineering | **8-12 person-days.** 2-3 to get LongLive's `train_dmd_longrun_{4,8}step.py` (or VDN's Stage-DMD with `real_score: dense` and the VDN-seed checks removed) running on our base revision. 1-2 to adapt it from 16 to 8 ranks and write the prompt cache. 1 for a throughput probe. 2 to run and watch the pilot. 2-3 for evaluation and serving (§4) |
| Main risks | (1) **Quality ceiling**: four LoRA turbos already exist, and LongLive's own self-assessment is guarded (`near_Base50_audio_quality_proven: false`). A short run may not beat larry or lightx2v. (2) **Audio** is the weak modality in every published distillation (larry's card; LongLive added a 50-NFE audio endpoint loss, which costs another 50 F a student update and would roughly double the run). (3) DMD instability (mode collapse, over-saturation); a run needs checkpoints every 50-100 steps and a person watching it. (4) Host RAM: CPU offload needs about 1 TB on the pod (check the pod type) |

### 2.2 (b) Full step distillation of the DiT (DMD2, FastH3-style)

| | |
|---|---|
| GPUs | **32 GPUs across nodes** (FastH3's documented layout: sp 4, HSDP shard 32, grad-accum 8). **It does not fit on 8×H100** (about 1.12 TB of states against 640 GB, §2.0). 8×B200 might hold the states but not 768p activations at sp 1. Plan 32×H200/H100 or 16-32×B200 |
| GPU-hours | 4-step: 1300 × 64 × 7 F = 582k F → **760 B200-h** (documented: "1k+ B200 h" for the family). 8-step: × 1.34 → **about 1,020 B200-h**. On H200: 582k F × 10 s = 1,620 H200-h |
| Wall-clock | about 24-32 h on 32 B200; about 51 h on 32 H200 |
| $ a run | 4-step: **$6.1k** (B200 at $7.99) / **$3.8k** (GMI $5.00) / **$7.0k** (Runpod H200 cluster at $4.31). 8-step: about $8.2k / $5.1k. Budget for 2-3 runs (FastVideo released checkpoint 1300 of a planned 4000 and checkpoint 300 of 1500, so steps are chosen by validation) → **$10-25k** |
| Data | Data-free: text embeddings only (FastH3 used five preprocessed sets, text only for the 4-step). Optionally synthetic base-H3 rollouts: a 50-step dense 768p rollout is 49 F + decode ≈ 8 min on an H100, so 1k clips ≈ 140 H100-h ≈ $550 |
| Engineering | **25-40 person-days.** Write the missing `MiniMaxH3DMDModel`: the uncond branch, VSA-H3 in training (the public plugin refuses it), the audio safeguards and the continuous score clock. FastVideo notes the 4-step "predates later continuous score-clock and FastGen-alignment corrections" (`HF:vsafp4` provenance.json). Then multi-node launch, checkpoint/resume, a 70 GB checkpoint export |
| Main risks | The recipe details are unpublished. The cost is high per lesson learned. We would be re-creating what FastVideo already ships for free (V2 passes our gate at 768p; §4). Multi-node capacity and cost control on Runpod Instant Clusters is untested here |

### 2.3 (c) Pruning plus healing distillation (Trim-style)

| | |
|---|---|
| Steps | (1) Rank the blocks by skipping each one and measuring the change in the video and audio predictions (12 conditions × 50 blocks + 12 baselines ≈ 612 F ≈ **1.7 H100-h, about $7**). (2) Rank-16 AdaLN (`convert_minimax_h3_adaln_rank.py`, CPU/GPU minutes). (3) Heal with DMD2: the pruned student against the full teacher and critic |
| GPUs | as (b): a full-weight student (28 B) plus a critic plus a teacher (Trim used 32 GPUs, sp 4) |
| GPU-hours | Released checkpoint 300: 300 × 64 × 9.4 F × about 0.93 (student 42/50 blocks; teacher and critic full) ≈ 168k F → **about 220 B200-h** (about 7 h on 32 B200) → **$1.75k** at Runpod rates / **$1.1k** at GMI. If the run went to its configured 1500 steps, it cost 5× that. Budget 2-3 block sets / runs → **$4-9k** |
| Data | as FastH3 Trim: T2AV latents (real or synthetic video) or text-only |
| Engineering | **20-35 person-days** standalone, or **8-12 on top of (b)'s infrastructure**. Serving is nearly free: our loader already reads `num_layers`, `kept_block_indices` and `adaln_rank` (`crates/fastvideo-cudarc/src/h3/pipeline.rs:2812-2861`; docs/ports/h3-trim.md §1.1) |
| A cheaper, unproven variant | heal with a LoRA on the pruned net, on 8×H100 (as a). No published evidence that LoRA capacity is enough to recover 8 removed blocks |
| Main risks | FastVideo already publishes a 42-block Trim. Ours would matter only for a deeper prune, or a prune of a different base. FastVideo's earlier activation-based prune had weak motion and detail (blog). The Trim fails our gate against the base, as every few-step recipe does (§4) |

### 2.4 (d) Fine-tuning on our own data

| | |
|---|---|
| GPUs | **LoRA SFT: 8×H100** (public FastVideo `MiniMaxH3LoraModel`, rank 32 on the same 312 modules, or VDN's latent format). Full SFT: 530 GB of states for one role → 8×B200 or 16-32 GPUs |
| GPU-hours | LoRA pilot: 500 steps × 8 × 4 F = 16k F → **44 H100-h, about $180**. Full LoRA SFT: 2,000 × 8 × 4 F = 64k F → **178 H100-h** (about 22 h) → **$710** at Runpod / **$460** at GMI |
| Data | **2k-10k licensed clips with audio** (H3 is a joint audio-video model, and its loss includes the audio latent; silent clips would teach silence). Captions must be in H3's prompt style, including `<d>` dialogue markup. VDN used 10,015 clips. Preprocessing (VAE encode + Qwen3-VL text + VLM captioning) ≈ 10 s a clip ≈ 28 H100-h for 10k clips. **Getting the data and its rights is the real cost** |
| Engineering | **10-20 person-days**, most of it the data pipeline. The H3 SFT plugin is public but has only been validated as a single-sample overfit |
| Main risks | The result is a **50-step CFG model**. To serve it fast we must either stack a turbo LoRA on top (adapters trained on the unmodified base may not compose; LongLive warns against combining its two LoRAs) or distill it (option a), so the true cost is (d) + (a). Forgetting, audio regression, and the cost of data rights |

### 2.5 (e) Quantization-aware tuning for NVFP4

| | |
|---|---|
| GPUs | Blackwell for real NVFP4 GEMMs in the forward pass (FastVideo's LTX QAT uses flashinfer CUTLASS with an STE backward). Full-weight QAT of the student only (about 530 GB of states) → **8×B200**. A LoRA-QAT (QLoRA-style) could run on fewer, but is unproven for H3 |
| GPU-hours | QAT by self-distillation (SFT toward the bf16 model's own samples): 1,000 steps × 32 × 4 F × 1.2 (quantization overhead) ≈ 154k F → **about 200 B200-h** (about 25 h on 8 B200) → **$1.6k** at Runpod / **$1.0k** at GMI. QAD (DMD with a quantized student, Kandinsky-style) costs about 2× that |
| Data | 1k-10k prompts and their bf16 teacher outputs (data-free possible) |
| Engineering | **10-15 person-days** (wire `ATTN_QAT_TRAIN` / the NVFP4 linears into the H3 plugin, which today accepts only SDPA), **plus 3-5 days to try better PTQ first** |
| Main risks | **FastVideo tested QAD on the Trim and found it did not beat PTQ.** Our gate fails the published NVFP4 PTQ checkpoints (Trim: sharpness and jitter, docs/ports/h3-trim.md §5.4; V2 NVFP4's own card reports PSNR 11.9 dB and LPIPS 0.52 against its bf16 source). Cheaper levers come first: keep sensitive layers in MXFP8, calibrate per timestep, use an SVDQuant-style low-rank residual. NVFP4 only pays off on Blackwell, and MXFP8 (already served) passes our gate |

---

## 3. Evidence quality

| number | status | where from |
|---|---|---|
| H3 is 33 B params, about 13 B of it AdaLN; bf16 checkpoint 70.10 GB | **documented** | `HF:base` README:142; docs/ports/h3.md |
| FastH3 4-step: 32 GPUs, sp 4, accum 8, step 1300, VSA 0.9, data-free, lr 2e-6, fp32 | **documented** (config) | `HF:vsafp4` checkpoint_metadata.json, provenance.json |
| FastH3 family "1k+ B200 training hours" | **documented** (blog; no breakdown) | haoailab.com/blogs/fasth3-preview |
| FastH3 Trim: blocks removed, rank 16, 32 GPUs, step 300 of 1500, t2va data | **documented** (config + blog) | `HF:trim` metadata.json, README; fasth3-rtx blog |
| FastH3 NVFP4: PTQ, 1,000 calibration prompts; "QAD did not beat PTQ" | **documented** | `HF:v2fp4` README; fasth3-rtx blog |
| LongLive LoRA: 16 GPUs (4 × 4), global batch 16, 1000 steps / 200 student updates, rank 128, H100-class | **documented** (config; the GPU is inferred from a code comment) | `HF:llfew` training_config.json, provenance.json, train_utils.py:845 |
| VDN DMD: 4 × 8 GPUs, 250 steps, warm start from larry v4-600, 10,015 clips | **documented** | `VDN/scripts/training/stage_dmd_vdn.sh`, `configs/training/stage_dmd_vdn.yaml`; arXiv 2609.20744 |
| Wall-clock, GPU type or $ for LongLive, VDN, larry, lightx2v | **not documented anywhere** | |
| F = 3.5e15 FLOPs per 768p forward | **estimate** (arithmetic §2.0, from documented dims) | |
| H100 10 s/F, B200 4.7 s/F in training | **estimate** (MFU assumption), cross-checked once against "1k+ B200 h" (within 1-2×) | |
| 7 F / 9.4 F per DMD sample, 4 F per SFT sample | **estimate** (DMD2 algorithm structure in `FV/.../dmd2.py`: generator every 5 critic steps) | |
| Every GPU-hour, wall-clock and $ figure in §2 | **estimate** = steps × samples × F-count × s/F (shown inline) × price | |
| Person-days | **estimate** (judgement, scaled from the size of the code to port: LongLive trainer about 0.3 MB of Python, VDN Stage-DMD about 2.9k lines) | |
| Memory fit (8×H100 yes for LoRA-DMD, no for full-weight DMD) | **estimate** (byte arithmetic §2.0) plus LongLive's documented H100-class fit | |

The single biggest uncertainty is **s/F in training on our hardware**. A one-hour probe (§5) replaces it with a
measurement and rescales every row of §2 at once.

---

## 4. How we would evaluate a result, and what serving would need

### 4.1 Quality gate

Our gate is `fv-gpucheck gate` with `scripts/gpu/gate-policy.toml`, kind `lossy`. Its hard checks are frame count,
a sharpness ratio of 0.95-1.08, a temporal-jitter ratio of 0.85-1.20, and LPIPS available. PSNR and LPIPS are recorded
as telemetry. The inputs come from `fv-gpucheck compare-clips` over the 5 prompts of `scripts/gpu/prompts-eval.json`
(`sol-engine-eval-5`) × 3 seeds (docs/techniques.md:565).

It is a **regression gate against a baseline sample, not a quality metric.** On 2026-10-14, every few-step recipe we
serve failed it against the 49-forward base (larry, lightx2v, FastH3 4-step, Sol-H3, the Trim). Only FastH3 V2 at
768p passed (docs/serve/e2e/h3-test-round-2026-10-14.md, Summary and §A). A new checkpoint therefore needs:

1. **The gate against the nearest served peer**, not the base: a new turbo LoRA against `h3-larry-turbo` and
   `h3-lightx2v-turbo` at the same NFE; a new prune against `fasth3-trim-8step`; an SFT model against `h3-base`; a
   QAT model against the same network's MXFP8 build.
   - Pass: within the ratio windows, LPIPS no worse than the peer's own spread.
   - Against the base: sharpness and jitter as telemetry, with the goal of moving toward 1.0. FastH3 V2's 768p row
     (0.97-1.06 / 1.00-1.14) is the bar to beat.
2. **Blind A/B** of the 5 × 3 grid against the peer, judged by the owner or a vision judge. The gate's
   `[visual_artifact]` judge is `deferred` today.
3. **Audio**, the known weak spot. The gate has no audio check, so add log-mel cosine against the base (the V2 NVFP4
   card's metric) and an ASR exact-match on the dialogue prompts (as LongLive did, 23/29).
4. **Training-time validation**: a fixed held-out prompt set (FastVideo used `heldout60.json`), rendered every 100
   steps at 480p to choose the checkpoint. Both FastH3 releases were chosen mid-run (1300 of 4000, 300 of 1500).

### 4.2 Serving changes

| option | loader | presets / recipes | weights |
|---|---|---|---|
| (a) LoRA | **none if the adapter is PEFT** on the diffusers module names (`transformer_blocks.N.attn.to_q.lora_A...`). Our H3 LoRA fold already reads larry (ComfyUI keys) and lightx2v (PEFT) (docs/ports/h3-turbo-loras.md §2-3, `crates/fastvideo-cudarc/src/h3/lora.rs`). An adapter that also touches AdaLN or the VSA gate (like FastVideo's extracted ones) needs the `.diff` / `.set_weight` handling the FastH3 LoRA path already has | a technique profile (shifts, NFE ladder, strength) and `configs/serve/runpod-h3-<name>.toml`, the same as `h3-larry-turbo` | about 1.4-2.8 GB (rank 64-128, bf16/fp32) as a new R2 tree. Add-only, sha256-verified, recorded in `scripts/gpu/weights-manifest.tsv` and `verify-weights.sh` (CLAUDE.md) |
| (b) full DiT | none if exported in diffusers layout (FastVideo's export is). A full checkpoint loads like `h3-8step` | a new preset like `h3-8step` (schedule from `fastvideo_inference.json`) | 70 GB tree + MXFP8 prequant (docs/gaps/2026-10-07-fast-boot.md) |
| (c) pruned | none for a Trim-format checkpoint (`num_layers`, `kept_block_indices`, `adaln_rank` are read) | as `fasth3-trim-8step` | about 37.5 GB |
| (d) SFT | as (a) or (b) | plus a turbo pairing | as (a) or (b) |
| (e) NVFP4 | ModelOpt layout is loaded (docs/ports/fasth3-nvfp4.md) | opt-in preset | about 11-20 GB |

Our own trees must carry the H3 licence's modification notice and `NOTICE` file (§1.1) if they are ever distributed
beyond our hosted service.

---

## 5. Recommendation

- **Not worth it now: (b) full step distillation, (c) pruning and (e) QAT.** Each is a $4-25k, multi-node,
  multi-week job that re-creates or slightly varies what FastVideo already publishes for free, and two of them use
  methods FastVideo itself found marginal (QAD did not beat PTQ; the Trim's quality cost). Revisit (c) only if we
  need a model below the Trim's size; revisit (e) only after better PTQ has been tried.
- **(d) fine-tuning** makes sense only with a concrete product need (a style, a domain, a face or voice policy) and
  licensed audio-video data. Without data it is not actionable, and it still needs (a) to be served fast.
- **Worth a pilot: (a) a data-free DMD2 LoRA**, on one 8×H100 pod, from LongLive's or VDN's trainer. It is the only
  option that fits on hardware we can rent as a single pod, needs no video data, and drops into our serving path with
  only a recipe and a config. The case for owning one over larry/lightx2v/FastH3: a turbo trained on our shifts and
  ladder, covering T2V **and** I2V/FL2VA (FastH3 is T2V only), with audio weighted the way we want, and a base for a
  Trim-turbo or Ref2VA turbo nobody publishes.

**The smallest useful pilot, cost-capped, in two stages:**

1. **Throughput probe. Cap $50, about 1 h on 8×H100.** Bring up the LongLive 4-step trainer, or VDN Stage-DMD with
   `real_score: dense`, at 768p and at 480p. Run 10 outer steps. Record s/step, peak memory a GPU and host RAM.
   - This replaces the s/F assumption that every number in §2 rests on.
   - **Stop** if it does not fit 8×80 GB at 480p.
2. **Pilot. Cap $300, about 60 H100-h.** 300 outer steps, global batch 8, 4-step, 480p, warm-started from larry's
   ckpt850 (the VDN recipe, which shortened their run to 250 steps). Use 10k self-written or licence-clean prompts.
   Checkpoints every 50 steps.
   - Evaluate per §4.1 against `h3-larry-turbo` at the same NFE.
   - **Go on** to a full 768p run (about $2k, §2.1) only if the pilot is at least on par on the gate and A/B, and
     better on audio or I2V.
   - **Stop** if it is worse after 300 steps.

Under the GPU-pod rules in CLAUDE.md, both stages need: the owner's approval of the spend; a Runpod balance that
stays above $8 after the cap; a wall-clock backstop on the pod; and outputs written to R2 as new add-only trees. No
large new download is needed: h3-base is already in R2.
