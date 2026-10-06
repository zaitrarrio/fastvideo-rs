# Runpod network volumes: manifest and rebuild

Date: 2026-09-30; updated 2026-10-06 (§0: the US volume is gone, EU only). This page lists what is on each of our Runpod network
volumes and how to rebuild them from scratch. Figures come from the repo
(fetch logs, `docs/gaps/2026-09-27-volume-sync.md` and the port docs), from a
read-only Runpod API listing of the volumes, and from read-only Hub API reads
that expand short revisions. Nothing here was measured on the volumes today
(the balance was negative, so no pods ran). Anything not backed by a
recorded source is marked **UNVERIFIED** or **UNKNOWN**.

## 0. Status 2026-10-06: EU only

- **The US weights volume is gone.** Runpod deleted `s2k01690bi`
  (`fv-weights-b200-us`, 2000 GB, US-CA-2) on about **2026-10-05**, while the
  account balance was negative. Nothing in the repo deleted it. Its data is
  lost; the US column in §2 records what it held.
- **The EU volume is intact.** `jg48s6o1w0` (`fv-weights-h3-ltx-hy`,
  EUR-IS-1) was verified on 2026-10-06: every manifest tree is present and
  passes its `verify-weights.sh` cell, and every recorded hash
  (`weights-sha256.tsv`, the `sha:<dest>` cells, `aux`, `upscalers`,
  `text-fp8`) checks out. About **1.45 TB** of the 2000 GB is used. One
  unlisted tree was found: `weights/wan/`, **12.46 GB**, not in
  `weights-manifest.tsv` and read by no cell. Leave it in place (add-only);
  nothing needs it.
- **Owner decision (2026-10-06): EU only for now.** US is not rebuilt. New
  weights go on EU alone (CLAUDE.md); the "both volumes" rule is suspended
  until the owner rebuilds US. Every script default, `configs/serve/autoscale.toml`
  and fv-control use EU; asking for the US volume or the `us` region fails
  with "US weights volume deleted 2026-10; EU only, see
  docs/ops/runpod-volumes.md".
- **Sol-engine benchmark trees added 2026-10-06** (owner-approved, EU only):
  `sana-video-2b-480p`, `wan21-t2v-1.3b`, `ltx23-dev`, `wan22-t2v-a14b`,
  `lingbot-video-moe-30b-a3b`, `cosmos3-super` (483.30 GB, §2). Each was
  fetched add-only by `fetch-hub-tree.sh` (temp folder, every file checked
  against the Hub at the pinned revision, then renamed), and its cell
  (`sana-video-2b-480p wan21-t2v-1.3b ltx23-hq wan22-t2v-a14b lingbot-moe
  cosmos3-super`) passed on a fresh pod. `du -sb /workspace` was then
  1938.92 GB: **about 61 GB free**. New large trees need space freed or the
  volume grown first (owner). `statvfs` on the mount reports the whole
  Runpod cluster (hundreds of PB); `fetch-hub-tree.sh` now passes the
  volume size so the fetch checks size minus `du` against a 50 GB floor.
- **Rebuilding US later** (from EU or the Hub): §5, `scripts/gpu/rebuild-volume.sh us`,
  and the switch-back list in §5.0.

Data files beside this page:

| File | What it holds |
|---|---|
| `scripts/gpu/weights-manifest.tsv` | dest, Hub repo, globs (plus `auxiliary/` URL rows with SHA-256 and size) |
| `scripts/gpu/weights-revisions.tsv` | the pinned Hub revision of every tree, and how it is known |
| `scripts/gpu/weights-sha256.tsv` | per-file hashes recorded at fetch time (`verify-weights.sh sha:<dest>`) |
| `scripts/gpu/verify-weights.sh` | completeness and hash gates (cells) |
| `scripts/gpu/rebuild-volume.sh` | one entry point: plan (`--dry-run`) or run the fetch and verify sequence on a pod |

## 1. The volumes

| Id | Name | Size | DC | Role |
|---|---|---:|---|---|
| ~~`s2k01690bi`~~ | ~~`fv-weights-b200-us`~~ | 2000 GB | US-CA-2 | **deleted by Runpod ~2026-10-05** (negative balance). Was the US weights volume (GPU pods and serverless in US-CA-2). Not rebuilt (§0) |
| `jg48s6o1w0` | `fv-weights-h3-ltx-hy` | 2000 GB | EUR-IS-1 | **the** weights volume (EU only since 2026-10-06; RTX PRO 6000 pods in EUR-IS-1). Verified 2026-10-06, ~1.45 TB used |
| `pxy4hlsnwq` | `fv-build` | 200 GB | EU-RO-1 | build caches for the shared build pod (toolchains, sccache, crates). No weights. Section 8 |

Other volumes on the account are **not ours and not covered here**. Nothing
in this repo refers to their ids or names. Four of them were also gone by
2026-10-06 (deleted with the US weights volume during the negative balance,
as far as the listing shows):

| Id | Name | Size | DC | Status 2026-10-06 |
|---|---|---:|---|---|
| `1nh52zvqku` | strobe | 200 GB | US-KS-2 | present |
| `gbfb1w87lc` | strobe-weights | 50 GB | US-KS-2 | **gone** |
| `zqe9uhus9s` | realvideo-models | 300 GB | US-NE-1 | present |
| `4odffuh7in` | fierce_aquamarine_platypus | 50 GB | US-CA-2 | **gone** |
| `weovb2qs46` | fierce_aquamarine_platypus | 50 GB | EUR-IS-4 | present |
| `nevbj2zhv8` | systematic_olive_wren | 100 GB | US-NE-1 | present |
| `whygavxxyi` | elderly_silver_bobolink | 100 GB | EU-RO-1 | present |
| `q3sihbv963` | eastern_ivory_mink | 10 GB | US-CA-2 | **gone** |
| `tvlsbglwur` | formal_aqua_ape | 50 GB | US-TX-3 | **gone** |

### Rules (CLAUDE.md)

- New weights go on the **EU** weights volume `jg48s6o1w0`, never only on a
  pod's container disk. **EU only since 2026-10-06** (owner decision, §0):
  the earlier rule, both volumes eventually in sync, is suspended until the
  owner rebuilds US. Do not create, mount or copy to a US volume without the
  owner.
- **Add-only.** Write under a temporary name (`<parent>/.<name>.partial-<stamp>`),
  verify (SHA-256), then rename. Never modify or delete existing volume data,
  and never delete a volume. A fetcher may remove only its *own* unfinished
  temp folders.
- Record every new tree in `weights-manifest.tsv` (plus its revision in
  `weights-revisions.tsv`) and give it a `verify-weights.sh` cell.
- Large new downloads need the owner's approval. A full rebuild is about
  1.17 TB per volume.
- Pods: only touch pods you created. Give each one a wall-clock backstop and
  delete it when done. Stop before the Runpod balance would drop below $8.

### Costs

Network volume storage is $0.07/GB-month for the first 1 TB and $0.05 beyond
(docs.runpod.io/storage/network-volumes, quoted in
`docs/gaps/2026-09-27-volume-sync.md`; not re-checked today):

| Volume | Size | $/month |
|---|---:|---:|
| ~~`fv-weights-b200-us`~~ | deleted ~2026-10-05 | **$0** (was $120) |
| `fv-weights-h3-ltx-hy` | 2000 GB | 70 + 50 = **$120** |
| `fv-build` | 200 GB | **$14** |

A volume can grow but cannot shrink. Billing is on the provisioned size, not
on use.

CPU pods for fetching (`python:3.12-slim`, secure cloud): cpu3c 8 vCPU
$0.24/hr, 2 vCPU $0.06/hr (volume-sync doc, 2026-09-27). Hub downloads into
Runpod are not billed as egress. The copy between regions goes through the
Runpod HTTPS proxy, with no charge recorded in the repo (**UNVERIFIED**).

## 2. Contents: per-tree manifest

Paths are under `/workspace/weights/` on the weight volume. The **US
column is history**: what `s2k01690bi` held before Runpod deleted it
(~2026-10-05). The EU column was re-verified on 2026-10-06 (§0). Bytes are
exact where a fetch log or `du -sb` recorded them. "~" marks the
2026-09-27 survey figure (10^9 bytes, 2 decimals). Revisions are the
40-hex SHAs in `weights-revisions.tsv`, shortened here to 7 characters.
The basis column says how each one is known:

- **rec**: recorded in the repo.
- **short**: a short SHA was recorded, and the Hub API expanded it on 2026-09-30.
- **inf**: nothing was recorded. The tree was fetched unpinned (hf-fm, `main`)
  no earlier than 2026-09-22, and the repo's `main` has not changed since before
  then, so it is inferred. Confirm it against a surviving volume's
  `models--*/snapshots/<rev>`.

The "sha256 source" column says where file hashes can be checked.

| Volume path | Source (repo @ rev, basis) | Files / globs | Bytes | sha256 source | US | EU | Needed by (cells / models) | Licence |
|---|---|---|---:|---|:-:|:-:|---|---|
| `h3-8step` | FastVideo/FastVideo-FastH3-8-Step-V2 @ `3da2ddf` inf | manifest row (tokenizer, text_encoder 14 shards, transformer, vae, audio_vae) | ~147.85 GB | Hub LFS (not recorded in repo) | yes | yes | `fasth3-8step` (FastH3 8-step) | MiniMax H3 community |
| `h3-base` | MiniMaxAI/MiniMax-H3 @ `42ed227` rec (docs/ports/h3-ref2v.md: snapshot on both) | manifest row | ~144.03 GB | Hub LFS (not recorded) | yes | yes | `h3-base`, `sol-h3`, `sol-h3-spark`, FastH3 4-step (base + LoRA), `h3-ref2va` | MiniMax H3 Community Licence (gated: no; read §1.1 of docs/ports/h3-ref2v.md before serving) |
| `h3-base/text_encoder_fp8` | **derived** (§3) | `manifest.json`, `model.safetensors` | 2 965 + 25 950 724 552 | `verify-weights.sh text-fp8` | yes | yes (copied US→EU) | optional, faster H3 text-encoder load (E13) | as h3-base |
| `FastH3-4-step-Preview-v1-LoRA` | FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA @ `f509e62` inf | `dense-datafree/`, `vsa-datafree/adapter_model.safetensors` | ~6.82 GB | Hub LFS (not recorded) | yes | yes | `fasth3-4step-vsa`, `fasth3-4step-dense`, `sol-h3*` | minimax-h3-community |
| `upscaler` | LBH-123-AI/Minimax_h3_latent_Upscaler @ `3f941d5` rec | `minimax_h3_latent_upscaler_3d_conv_v1/…_bf16.safetensors` | 690 592 992 | `verify-weights.sh sol-h3-spark` (`4f57821f…46a5e6`) | yes (HF cache) | yes (plain file, plus the older top-level copy of the same bytes) | `sol-h3-spark` (Spark 1080p latent upscale) | apache-2.0 |
| `h3-to-ltx` | Efficient-Large-Model/H3-to-LTX-Latent-Adapter @ `1792c42` inf | `config.json`, `model.safetensors` | ~0.39 GB | not recorded | yes | yes | `sol-h3-spark` | none stated on the Hub card (**check before serving**) |
| `h3-ref2va` | MiniMaxAI/MiniMax-H3 @ `42ed227` rec + lightx2v/Minimax-h3-Turbo @ `3ec17a3` rec | `transformer_ref/*`, `processor/*`, `model_index.json`, `LICENSE`, `README.md`, `Minimax-h3-Turbo/…ref2v_turbo_{4step_v0.1,8step_v1.0_768p}_bf16.safetensors` | 69 059 483 520 | `weights-sha256.tsv` (29 files; also `sha256.txt` in the tree) → `sha:h3-ref2va` | yes | yes | `h3-ref2va`, `h3-ref2va-turbo` (H3 reference-to-video) | H3 community; turbo LoRA apache-2.0 card (derivative of H3) |
| `ltx25` | Lightricks/LTX-2.5-Diffusers @ `426936f` inf | manifest row (incl. `ltx-2.5-22b-distilled-lora-450-bf16.safetensors`) | ~125.12 GB | not recorded | yes | yes | `ltx25-two-stage`, `sol-h3-spark`, `ltx25-a2v-guided`, `ltx25-ref2v` | LTX-2 community licence (gated: auto) |
| `ltx25/text_encoder_fp8` | **derived** (§3) | `manifest.json`, `model.safetensors` | 1 408 + 12 923 848 536 | `verify-weights.sh text-fp8` | yes | yes (copied US→EU) | optional LTX-2.5 Gemma FP8 load | as ltx25 |
| `ltx25-dev` | Lightricks/LTX-2.5-Diffusers @ `426936f` rec | `transformer_full/*` (4 shards + index) + `model_index.json` | 37 976 670 004 | Hub LFS, checked at fetch, not recorded | yes | yes | `ltx25-dev`, `ltx25-a2v-guided` (guided A2V, ltx-pro) | LTX-2 community (gated: auto) |
| `ltx25-ic-lora-ingredients` | Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients @ `12040e4` rec | `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors`, `README.md` | 1 308 787 472 + 25 643 | `weights-sha256.tsv` (`ff873a5b…8715e95`) → `sha:ltx25-ic-lora-ingredients` | yes | yes | `ltx25-ic-lora-ingredients`, `ltx25-ref2v` | ltx-2.x community (gated: auto) |
| `ltx2` | Lightricks/LTX-2 @ `dfcc210` inf | manifest row (tokenizer, text_encoder `model-*`, vae, audio_vae, vocoder, `ltx-2-19b-distilled.safetensors`) | US ~94.74 GB; EU ~146.35 GB | not recorded | yes | yes (+51.61 GB extra, §4) | `ltx2` (new cell); ltx2 matrix cells (`--dit ltx2/ltx-2-19b-distilled.safetensors`) | LTX-2 community |
| `ltx23` | FastVideo/LTX-2.3-Distilled-Diffusers @ `22b09fb` short | manifest row (incl. `text_encoder/gemma/*`, `text_embedding_projection/*`) | US 71 559 124 637; EU 71 559 124 839 | Hub LFS (Gemma 5/5, projection 1/1 checked in the sync, not listed) | yes | yes | `ltx23` | derived from LTX-2.3 (LTX-2 community); no licence field on the card |
| `hy15-480-t2v` | hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v @ `286be7c` short | manifest row | 53 384 330 435 | Hub LFS 14/14 checked in the sync, not listed | yes | yes | `hy15-480-t2v` | Tencent Hunyuan community (territory limits) |
| `hy15-480-i2v` | …-480p_i2v_step_distilled @ `854c04a` short | manifest row | 33 780 496 799 | as above (8/8) | yes | yes | `hy15-480-i2v` | as above |
| `hy15-720-t2v` | …-720p_t2v @ `f4dbc4a` short | manifest row | 53 384 330 435 | as above (14/14) | yes | yes | `hy15-720-t2v` | as above |
| `hy15-720-i2v` | …-720p_i2v_distilled @ `a1d10cf` short | manifest row | 53 384 305 661 | as above (10/10) | yes | yes | `hy15-720-i2v` | as above |
| `fastwan21-1.3b` | FastVideo/FastWan2.1-T2V-1.3B-Diffusers @ `25e7ed7` short | manifest row | 29 212 131 136 | Hub LFS 9/9 (sync), not listed | yes | yes | `fastwan21-1.3b` (wan-turbo) | apache-2.0 |
| `wan22-ti2v-5b` | Wan-AI/Wan2.2-TI2V-5B-Diffusers @ `b8fff73` short | manifest row | 34 201 427 557 | Hub LFS 11/11 (sync), not listed | yes | yes | `wan22-ti2v-5b` | apache-2.0 |
| `fastwan22-ti2v-5b` | FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers @ `3e18704` rec (fetch log) | manifest row | 24 201 770 562 | `weights-sha256.tsv` (15 files, US = EU) → `sha:fastwan22-ti2v-5b` | yes | yes | `fastwan22-ti2v-5b` | apache-2.0 |
| `wan21-t2v-14b` | Wan-AI/Wan2.1-T2V-14B-Diffusers @ `38ec498` short | manifest row | 80 406 933 703 | Hub LFS 20/20 (sync), not listed | yes | yes | `wan21-t2v-14b` | apache-2.0 |
| `sfwan21-1.3b` | wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers @ `4b44356` short | manifest row | 28 928 823 445 | Hub LFS 9/9 (sync), not listed | yes | yes | `sfwan21-1.3b` | apache-2.0 |
| `longlive-1.3b` | Efficient-Large-Model/LongLive-1.3B @ `cda9138` rec | `README.md`, `models/longlive_base.pt`, `models/lora.pt`, `prompts/interactive_example.jsonl` | 8 476 402 298 | `weights-sha256.tsv` (4 files, US = EU) → `sha:longlive-1.3b` | yes | yes | `longlive-1.3b` (LongLive on the SF-Wan causal engine, `wip/longlive`) | **NON-COMMERCIAL: treat as research/evaluation only** (HF card `cc-by-nc-sa-4.0`; card body CC-BY-NC 4.0; the code repo moved to Apache-2.0 on 2025-11-01 but the card was not updated; docs/serve/research-longlive.md §4.3) |
| `longlive-1.3b-safetensors` | **derived** (§3): `convert-longlive.py` from `longlive-1.3b` | `longlive_base.safetensors` (825 f32 tensors), `lora.safetensors` (600 f32, PEFT rank 256), `keys-*.txt`, `sha256.txt` | 5 676 075 416 + 1 399 924 800 | `weights-sha256.tsv` → `sha:longlive-1.3b-safetensors` (base: per-volume hashes, metadata order only; §3) | yes | yes | `longlive-1.3b` (`wan stream --longlive`, `FV_LONGLIVE_WEIGHTS`) | as `longlive-1.3b` (non-commercial) |
| `longlive2-5b` | Efficient-Large-Model/LongLive-2.0-5B @ `8521079` rec | `README.md`, `model_bf16.pt` (LoRA merged) | 9 999 858 697 | `weights-sha256.tsv` → `sha:longlive2-5b` | yes | yes | `longlive2-5b` (not ported yet; needs `wan22-ti2v-5b`) | NVIDIA Open Model License (read the card's "API Trial Terms" line before serving) |
| `longlive2-5b-nvfp4-s4`, `longlive2-5b-nvfp4-s2` | …/LongLive-2.0-5B-NVFP4-S4 @ `427ffb7`, -S2 @ `9ab6c9c` rec | `README.md`, `model_4o6.pt` (FourOverSix NVFP4) each | 2 945 864 769 each | `weights-sha256.tsv` → `sha:longlive2-5b-nvfp4-s{4,2}` | yes | yes | `longlive2-5b-nvfp4` (not ported; Blackwell NVFP4) | NVIDIA Open Model License |
| `longlive-plug/minimax-h3-few-step` | …/LongLive-Plug-MiniMax-H3-few-step @ `b3686f4` rec | the whole repo (`*`): `generator_lora.pt` 2 767 464 857 + 54 small files (code snapshot, recipe, configs, LICENSE) 941 933 | 2 768 406 790 | `weights-sha256.tsv` (55 files) → `sha:longlive-plug/minimax-h3-few-step` | yes | yes | `longlive-plug` (H3 few-step LoRA; not ported) | MiniMax H3 Community (territory clause as H3: not EU/UK/KR/US without a MiniMax licence) |
| `longlive-plug/minimax-h3-cfg` | …/LongLive-Plug-MiniMax-H3-cfg @ `de1f4e8` rec | `adapter_model.safetensors` + 10 small files (`*.json` also matched `sglang/adapter_config.json`; the 2.9 GB `sglang/adapter_model.safetensors` is **not** fetched) | 2 767 312 418 | `weights-sha256.tsv` (11 files) → `sha:longlive-plug/minimax-h3-cfg` | yes | yes | `longlive-plug` | MiniMax H3 Community (as above) |
| `longlive-plug/wan21-t2v-14b-few-step`, `…/wan21-t2v-14b-cfg` | …/LongLive-Plug-Wan2.1-T2V-14B-few-step @ `f125af0`, -cfg @ `32b8aa3` rec | `generator_lora_lightx2v.safetensors` + 4 small; `adapter_model.safetensors` + 7 small | 1 226 929 535; 2 453 790 804 | `weights-sha256.tsv` → `sha:longlive-plug/wan21-t2v-14b-{few-step,cfg}` | yes | yes | `longlive-plug` (+ `wan21-t2v-14b`) | apache-2.0 |
| `longlive-plug/wan22-ti2v-5b-few-step`, `…/wan22-ti2v-5b-cfg` | …/LongLive-Plug-Wan2.2-TI2V-5B-few-step @ `38a6ec4`, -cfg @ `fa1f928` rec | `adapter_model.safetensors` + 6 / + 7 small | 1 289 840 234; 644 966 189 | `weights-sha256.tsv` → `sha:longlive-plug/wan22-ti2v-5b-{few-step,cfg}` | yes | yes | `longlive-plug` (+ `wan22-ti2v-5b`) | apache-2.0 |
| `mmaudio-44k-v2` | hkchengrex/MMAudio @ `eb13a1a`, nvidia/bigvgan_v2_44khz_128band_512x @ `95a9d1d`, apple/DFN5B-CLIP-ViT-H-14-384 @ **UNKNOWN** (all fetched from `main`; §3) | upstream `.pth` tree + converted `safetensors/` + `MANIFEST.json` | ~21.46 GB | md5 of the 3 `.pth` (`weights-sha256.tsv` → `sha:mmaudio-44k-v2`); converted files not hashed | yes | yes | `mmaudio-44k-v2` (Wan audio sidecar, V2A/T2A) | **MMAudio cc-by-nc-4.0 (non-commercial)**; BigVGAN MIT; DFN5B not read (Hub API 307) |
| `auxiliary/tae`, `auxiliary/lpips` | pinned URLs (madebyollin/taehv @ `e589fdd` / `32ac014`, download.pytorch.org, richzhang/PerceptualSimilarity @ `082bb24`) | 6 files | 372 975 478 | `weights-manifest.tsv` rows → `aux` | yes | yes | TAE decoders (H3, LTX, Wan), LPIPS (eval) | taehv MIT; torchvision BSD; LPIPS BSD-2 (from the upstream repos; **UNVERIFIED** here) |
| `auxiliary/upscalers/seedvr2` | numz/SeedVR2_comfyUI @ `09ced71` rec | `seedvr2_ema_3b_fp16.safetensors`, `ema_vae_fp16.safetensors` | 7 284 343 622 | `verify-weights.sh upscalers` | yes | yes | upscaler benchmark (docs/serve/h3-1080p-and-upscaler.md) | apache-2.0 |
| `auxiliary/upscalers/flashvsr-v1.1` | JunhaoZhuang/FlashVSR-v1.1 @ `27561b1` rec | 5 files + `model_index.json` | 6 948 393 656 | `verify-weights.sh upscalers` | yes | yes | upscaler benchmark | apache-2.0 |
| `sana-video-2b-480p` | Efficient-Large-Model/SANA-Video_2B_480p_diffusers @ `db5f398` rec | manifest row (transformer 2 shards, Gemma-2-2B text_encoder, tokenizer, Wan VAE, scheduler, `model_index.json`; the EU copy also holds `LICENSE`, 11 358 B, outside the globs) | 14 002 542 321 | `weights-sha256.tsv` (17 files) → `sha:sana-video-2b-480p` | — | yes (2026-10-06) | SANA-Video 2B (sol-engine benchmark; port on `wip/sol-sana-hunyuan`) | apache-2.0 |
| `wan21-t2v-1.3b` | Wan-AI/Wan2.1-T2V-1.3B-Diffusers @ `0fad780` rec | manifest row (whole Diffusers tree) | 28 928 887 859 | `weights-sha256.tsv` (19 files) → `sha:wan21-t2v-1.3b` | — | yes (2026-10-06) | `wan21-t2v-1.3b` (`wan_t2v_1_3b` preset). UMT5 / VAE are the same LFS objects as `wan21-t2v-14b`'s (kept as a copy: the loader reads `<root>/text_encoder`) | apache-2.0 |
| `ltx23-dev` | Lightricks/LTX-2.3 @ `3c6a4e6` rec | `ltx-2.3-22b-dev.safetensors`, `ltx-2.3-22b-distilled-lora-384-1.1.safetensors`, `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (the EU copy also holds `LICENSE`, 21 399 B, outside the globs; the LoRA and upscaler were added into the tree with `FETCH_ADD_INTO=1`) | 54 750 595 790 | `weights-sha256.tsv` (3 files) → `sha:ltx23-dev` | — | yes (2026-10-06) | `ltx23-hq` (LTX-2.3 HQ two-stage: dev DiT + distilled LoRA 384 v1.1 + x2 v1.1 upscaler here; Gemma, VAEs and vocoder from `ltx23`) | LTX-2 community |
| `wan22-t2v-a14b` | Wan-AI/Wan2.2-T2V-A14B-Diffusers @ `5be7df9` rec | manifest row (`transformer` + `transformer_2`, 12 shards each, UMT5 bf16, VAE) | 126 199 274 206 | `weights-sha256.tsv` (41 files) → `sha:wan22-t2v-a14b` | — | yes (2026-10-06) | `wan22-t2v-a14b` (two-expert MoE; UMT5 = `wan22-ti2v-5b`'s, VAE = Wan2.1's, kept as copies) | apache-2.0 |
| `lingbot-video-moe-30b-a3b` | robbyant/lingbot-video-moe-30b-a3b @ `f2e538f` rec | manifest row (`transformer`, `refiner`, Qwen3-VL `text_encoder`, `processor`, Wan 2.1 VAE, scheduler) + `model_index.json` | 129 952 345 105 | `weights-sha256.tsv` (59 files) → `sha:lingbot-video-moe-30b-a3b` | — | yes (2026-10-06) | `lingbot-moe` (`lingbot_moe_30b`, base + 1080p refiner; docs/ports/lingbot.md) | apache-2.0 |
| `cosmos3-super` | nvidia/Cosmos3-Super @ `f543c56` rec (not gated) | manifest row (`transformer` 27 shards, Wan 2.2 `vae`, Qwen2 `text_tokenizer`, scheduler) + `model_index.json`; `sound_tokenizer`, `vision_encoder` and assets are not fetched (not needed for T2V) | 129 465 061 778 | `weights-sha256.tsv` (40 files) → `sha:cosmos3-super` | — | yes (2026-10-06) | `cosmos3-super` (Cosmos3-Super 64B T2V; docs/ports/cosmos3.md) | OpenMDW 1.1 |
| `taeh3` (legacy) | copy of `auxiliary/tae/taeh3.safetensors` | `taeh3.safetensors`, `.complete` | 22 709 752 | `4fd022bf…` (= aux row) | yes | yes | nothing required; `auxiliary/tae` is read first | as aux |

**Totals** (weights only, from the table; "~" rows at survey precision):

| | US `s2k01690bi` | EU `jg48s6o1w0` |
|---|---:|---:|
| Trees in the table | ~1208.0 GB (LongLive +42.60 on 2026-10-02) | ~1743.6 GB (ltx2 +51.61, second upscaler copy +0.69; LongLive +42.60; sol-engine trees +483.30 on 2026-10-06) |
| EU-only unlisted weight tree (§4) | — | ~77.97 GB |
| Non-weight data (`upstream/`, `runs/`, `.cache/`) | ~1.78 GB (`runs/`) | ~104.1 GB |
| **Approx. used** | **~1210 GB** of 2000 | **1938.92 GB** of 2000 (`du -sb /workspace`, 2026-10-06 after the sol-engine trees; **61.08 GB free**) |
| What `rebuild-volume.sh` writes on an empty volume | 1200.9 GB (+35.52 LongLive Hub trees; the 7.08 GB converted tree is a manual step, §3) | 1200.9 GB |

These are sums of recorded figures, not a fresh `du`. The US column is
history (the volume was deleted ~2026-10-05). On 2026-10-06 EU reported
about 1.45 TB used, including the unlisted `weights/wan/` (12.46 GB, §4). The last survey
(2026-09-27) predates `h3-ref2va`, `ltx25-dev`, the IC-LoRA, `fastwan22`,
the upscalers and the FP8 copy to EU. The US volume reported 957.1 GB used
after the sync, before those roughly 205 GB of additions.

## 3. Derived trees (not a plain download)

### Pre-quantized FP8 text encoders (E13)

`h3-base/text_encoder_fp8` and `ltx25/text_encoder_fp8` were produced on
US on 2026-09-27 by `fv-gpucheck quantize-text-encoder`
(`crates/fastvideo-gpucheck/src/coldstart.rs`). It ran through the serverless
worker's `quantize` job (`scripts/gpu/serverless-worker.sh`) on a 1x H200
queue endpoint in US-CA-2 with the US volume at `/runpod-volume`
(docs/gaps/2026-09-27-cold-start.md "E13 trees"). EU received byte copies
(docs/gaps/2026-09-27-volume-sync.md).

**Preferred rebuild: copy from the surviving volume**, as the 2026-09-27
copy did. A CPU pod on the source volume serves the two folders read-only
over HTTP with Range support, under a random path. A CPU pod on the target
volume first checks that neither destination exists. It then pulls into
`<root>/.text_encoder_fp8.partial-<stamp>`, fsyncs, and re-reads every file
to compare its SHA-256 with the source's. It checks the file list and sizes
against `manifest.json`, and only then renames. That took 416 s for 38.9 GB
(~93 MB/s), about $0.11 of pods. Afterwards run
`FV_VERIFY_FP8_SHA=1 verify-weights.sh text-fp8`.

**If EU is lost too (US already is), regenerate it** (needs a GPU and a CUDA build of
`fv-gpucheck`, e.g. the runtime image `ghcr.io/zaitrarrio/fastvideo-rs-runtime`).
The run used an H200. The minimum GPU memory was not measured
(**UNVERIFIED**). Write to a temp name, then rename:

```bash
W=/workspace/weights; S=$(date -u +%Y%m%d%H%M%S)
fv-gpucheck --out /tmp/q-h3 quantize-text-encoder --family h3 \
  --root $W/h3-base --tree $W/h3-base/.text_encoder_fp8.partial-$S \
  && mv -T $W/h3-base/.text_encoder_fp8.partial-$S $W/h3-base/text_encoder_fp8
fv-gpucheck --out /tmp/q-ltx quantize-text-encoder --family ltx2-gemma4 \
  --root $W/ltx25 --tree $W/ltx25/.text_encoder_fp8.partial-$S \
  && mv -T $W/ltx25/.text_encoder_fp8.partial-$S $W/ltx25/text_encoder_fp8
FV_VERIFY_FP8_SHA=1 bash scripts/gpu/verify-weights.sh text-fp8
```

The tool hashes the source shards into `manifest.json`, then loads the tree
back and requires the codes-and-scales digest to equal the load-time one:
h3 `548475f2…`, ltx25 `d9a78567…`. The write took 763 s and 323 s, at
~35-40 MB/s. The quantization is deterministic, but `manifest.json` has a
`created_by` field. A regenerated tree may therefore differ from the SHA-256
recorded in `verify-weights.sh` (`e5504561…`, `bb5a3049…`), and so may the
model file if the writer changed (**UNVERIFIED**). If it differs, record the
new hashes in `FP8_TREES` rather than overwriting anything. Through the
serverless worker, the same run is the job `{"kind":"quantize","family":"h3","root":"h3-base"}`.

### MMAudio: converted safetensors

`scripts/gpu/fetch-mmaudio.py` downloads three `.pth` checkpoints from
`hkchengrex/MMAudio` and checks their md5 (pinned from MMAudio's
`download_utils.py`). It also downloads BigVGAN v2 (`bigvgan_generator.pt`) and
DFN5B CLIP (`open_clip_pytorch_model.bin`). It then converts each checkpoint to
`safetensors/<name>.safetensors`, with the same keys and dtype and a round-trip
equality check, and writes `MANIFEST.json` and `.complete`. The converted
files are what the Rust port reads. Their hashes were not recorded.

The script takes `main` of all three repos. `apple/DFN5B-CLIP-ViT-H-14-384`
now answers the Hub API with a 307 (moved). `hf_hub_download` follows
redirects, but pin a revision the next time the tree is fetched. On its own,
`fetch-mmaudio.py` writes into the final folder. `rebuild-volume.sh` runs it
into `.mmaudio-44k-v2.partial-<stamp>` and renames afterwards.

### Fetched subsets and merged layouts

- `ltx25-dev` holds only `transformer_full/` of the same repo and revision as
  the `ltx25` bundle (`426936f`). The rest of the guided A2V bundle comes from
  `ltx25`.
- `h3-ref2va` merges two repos into one tree (`fetch-h3-ref2va.py` pins both
  revisions and keeps `sha256.txt` and `SOURCE.json` in the tree).
- The `h3-base` and `h3-8step` text encoders take 14 shard globs, not
  `text_encoder/*`.
- `ltx23` needs `text_encoder/gemma/*`. The old glob `text_encoder/model-*`
  missed it until 2026-09-27. The manifest's `latent_upsampler/*` and
  `ltx-2.3-22b-distilled-lora-384*.safetensors` globs match nothing at
  `22b09fb`.

### LongLive-1.3B safetensors (`longlive-1.3b-safetensors`)

Written on 2026-10-02 on each volume by `scripts/gpu/convert-longlive.py`
(`wip/longlive`) from that volume's `longlive-1.3b` tree, on a CPU pod
(`python:3.12-slim`, 8 vCPU / 32 GB `cpu3g`, CPU torch + safetensors, the
volume at `/workspace`, a 1 h backstop; the §5.2 pattern):

```bash
python convert-longlive.py /workspace/weights/longlive-1.3b /workspace/weights/longlive-1.3b-safetensors
```

It writes `.longlive-1.3b-safetensors.partial-<stamp>`, reads every tensor
back (`torch.equal`), then renames. Output: `longlive_base.safetensors` (the
`generator` state dict: 825 tensors, **f32**) and `lora.safetensors`
(`generator_lora`: 600 tensors, f32, rank 256), with `keys-*.txt` and
`sha256.txt`. Logs: `artifacts/runpod/convert-longlive-1.3b-fv-weights-*/`.

The two volumes' `longlive_base.safetensors` have **different file
SHA-256s** (US `113656f2…`, EU `ff64e39c…`). The only difference is the
order of the two `__metadata__` entries in the JSON header (safetensors
serializes them from a hash map). The tensor table (sha256 of the sorted
header without metadata: `ab499911…`) and the tensor data (sha256 of the
bytes after the header: `ad413c12…`) are identical on both, checked on
both volumes the same day. `weights-sha256.tsv` lists both file hashes
(`a|b`). `lora.safetensors` happens to match (`7ec78e63…`). The converter
now writes a single metadata entry, so a rebuild gives one hash per run
(a new value; record it).

### Layout note

The older trees (`h3-*`, `ltx2`, `ltx25`, `ltx23`, the hy15 set and
`fastwan21` on EU) are in the HF-cache layout: `models--*/blobs` +
`snapshots/<rev>` + top-level links. The newer trees and every tree
`rebuild-volume.sh` writes use `local_dir` (plain files). Readers see the
same paths either way. On an old tree, the revision it holds is the
`snapshots/<rev>` directory name.

## 4. Differences between the volumes, and known gaps

**On one volume only / different there:**

| Where | What | Size | Status |
|---|---|---:|---|
| EU only | `FastH3-4-step-Preview-v1-VSA-DataFree` | ~77.97 GB | unlisted. Its transformer is incomplete despite `.complete` (only `.chunked.part` blobs). The rest duplicates h3 blobs. Not needed. Do not rebuild |
| EU only | `ltx2/…/text_encoder/diffusion_pytorch_model-000{01..12}-of-00012` | ~51.61 GB | a second Gemma layout outside the manifest glob. The loader reads `model-*`. Not needed |
| EU only | `upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors` (older top-level path) | 690 592 992 | same bytes as the pinned file (`4f57821f…`). Harmless |
| EU only | `upstream/`, `runs/`, `.cache/` | ~104.1 GB | not weights. Nothing reads them with `UP_LOCAL=1` (the default) |
| EU only | `wan/` (found 2026-10-06) | 12.46 GB | unlisted: not in `weights-manifest.tsv`, read by no cell. Not needed. Leave it (add-only); do not rebuild |
| US only (gone) | `runs/` | ~1.78 GB | not weights; lost with the volume |
| layout | the FastH3 LoRA `vsa-datafree` adapter and the upscaler are plain files on EU, HF blobs on US | 0 | same bytes |

Every manifest tree was on both volumes (sync of 2026-09-27, and each later
addition landed on both) until the US volume was deleted (~2026-10-05).
On 2026-10-06 every manifest tree and every recorded hash checked out on EU.

**Known gaps:**

- **ltx23 Gemma**: listed missing on both volumes in the 2026-09-27 survey.
  **Resolved the same day.** With the owner's approval, `text_encoder/gemma/*`
  (24.37 GB) was added to both, and `verify-weights.sh ltx23` is ok on both
  (commit `1a47986`).
- **EU Spark latent upscaler** (docs/serve/h3-1080p-and-upscaler.md:275
  says it is missing on EU): **resolved 2026-09-29.** The pinned
  `minimax_h3_latent_upscaler_3d_conv_v1/…` file was added on EU, and
  `sol-h3-spark` is ok on both (commit `582cd3f`). The line in that doc
  describes the gate run just before the fix.
- **Revisions not recorded at fetch time**: `h3-8step`, `FastH3-4-step-Preview-v1-LoRA`,
  `h3-to-ltx`, `ltx2`, `ltx25`. They are inferred from the Hub (see the basis
  column). Confirm them with one read-only listing of
  `*/models--*/snapshots/` on either volume.
- **DFN5B CLIP revision**: UNKNOWN (fetched from `main`, and the repo moved).
- **File hashes not recorded in the repo** for most large trees. They were
  checked against the Hub's LFS SHA-256 at fetch, and on a rebuild
  `fetch-hub-tree.py` checks them again at the pinned revision. Only these
  have repo-recorded per-file hashes: `h3-ref2va`, `fastwan22-ti2v-5b`, the
  IC-LoRA, the MMAudio `.pth` (md5), `auxiliary/`, the upscalers, the Spark
  upscaler file and the FP8 trees. To close the gap, run a read-only
  inventory pod per volume (the survey method in the sync doc: `find` plus
  `sha256sum`, served over the proxy, container-disk only). Then commit the
  lists to `weights-sha256.tsv`.
- **`verify-weights.sh` cannot see stale `.complete` markers on unlisted
  trees**, such as the EU VSA-DataFree tree.

## 5. Rebuild a weight volume

### 5.0 Rebuilding US (deferred: owner decision 2026-10-06)

US is not being rebuilt now. When the owner decides to:

1. Get the owner's approval for the downloads (about 1.2 TB) and check the
   balance (§7).
2. Create a new volume (§5.1). It gets a new id; `s2k01690bi` never comes
   back.
3. Fill it from EU (copy, as the FP8 trees were in 2026-09; §5.4 in
   reverse) or from the Hub with `rebuild-volume.sh us` (§5.2-5.3), with
   EU's per-tree `sha256.txt` lists as `FV_REBUILD_EXPECT_DIR` so every file
   must equal EU. The FP8 trees are copied from EU (§3).
4. Verify (§5.6), then switch US back on, one id each:
   - `scripts/gpu/volumes.sh`: `FV_US_VOLUME_ID` and `FV_US_VOLUME_NAME`
     (every Runpod script, `runpod-cluster.sh` regions, the fetchers);
   - `control/src/cluster/regions.ts`: `US_VOLUME_ID` (fv-control's `us`
     region), then add `us` back to the cluster specs that should use it;
   - `configs/serve/autoscale.toml` and the `PodConfig` default in
     `crates/fastvideo-autoscale/src/config.rs`: a US-CA-2 placement;
   - CLAUDE.md and §1 of this page: restore the both-volumes rule.

Needs: the owner's approval (about 1.17 TB of downloads per volume), a Runpod
balance well above $8, `RUNPOD_API_KEY`, and an `HF_TOKEN` whose account has
accepted the gates of the auto-gated repos (Lightricks LTX-2.5,
LTX-2.5-IC-LoRA). The fetchers also read `/workspace/hf/token` if the volume
has one. A fresh volume does not.

### 5.1 Create the volume (only if it is gone)

```bash
# Owner approval first. US (deleted ~2026-10-05; rebuild deferred, §5.0): fv-weights-b200-us,
# US-CA-2. EU: fv-weights-h3-ltx-hy, EUR-IS-1.
curl -sS -X POST https://rest.runpod.io/v1/networkvolumes \
  -H @<(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY") -H 'content-type: application/json' \
  -d '{"name":"fv-weights-b200-us","size":2000,"dataCenterId":"US-CA-2"}' | jq '{id,name,size,dataCenterId}'
```

Size: 1500 GB would hold the 1165 GB rebuild set with room for about
another model family ($95/month instead of $120). 2000 GB matches today. A
new volume gets a **new id**. The scripts take both regions' ids and names
from `scripts/gpu/volumes.sh`, and fv-control from
`control/src/cluster/regions.ts` (§5.0 has the full list). Check for any
other hard-coded id (`grep -rn 's2k01690bi\|jg48s6o1w0' scripts docs configs control crates`).

### 5.2 Start a cheap CPU pod on it, with a backstop

The pattern is the same as `fetch-hub-tree.sh`: `python:3.12-slim`,
secure-cloud CPU, `cpu3c`/`cpu5c`/`cpu3g`, 8 vCPU (about $0.24/hr), 30 GB
container disk (the MMAudio step installs CPU torch), the volume at
`/workspace`, and port 8000 served over the proxy for the logs. Ship the
scripts in the environment:

```bash
cd scripts/gpu
TGZ=$(tar czf - rebuild-volume.sh verify-weights.sh verify-safetensors.sh weights-manifest.tsv \
      weights-revisions.tsv weights-sha256.tsv fetch-hub-tree.py fetch-h3-ref2va.py fetch-mmaudio.py | base64 -w0)  # ~31 KB
START='mkdir -p /srv/rebuild /opt/fv && cd /opt/fv && echo "$FV_TGZ" | base64 -d | tar xzf - \
  && (python -m http.server 8000 --directory /srv >/dev/null 2>&1 &) \
  && apt-get update -qq && apt-get install -y -qq curl >/dev/null \
  && bash /opt/fv/rebuild-volume.sh "$FV_SIDE" >/srv/rebuild/run.log 2>&1; echo $? >/srv/rebuild/EXIT; sleep infinity'
jq -n --arg vol "$VOL_ID" --arg dc "$DC" --arg tgz "$TGZ" --arg start "$START" --arg side us --arg hf "$HF_TOKEN" '{
  name: "fv-rebuild-\($side)", imageName: "python:3.12-slim", cloudType: "SECURE", computeType: "CPU",
  cpuFlavorIds: ["cpu3c","cpu5c","cpu3g"], cpuFlavorPriority: "availability", vcpuCount: 8,
  containerDiskInGb: 30, networkVolumeId: $vol, volumeMountPath: "/workspace", dataCenterIds: [$dc],
  ports: ["8000/http"], dockerStartCmd: ["/bin/bash","-lc",$start],
  env: {FV_TGZ: $tgz, FV_SIDE: $side, HF_TOKEN: $hf, HF_HUB_DISABLE_PROGRESS_BARS: "1"}}' > payload.json
# POST /pods with payload.json. Then arm a detached backstop that DELETEs the pod after
# about 4 h, as fetch-hub-tree.sh does (setsid nohup … sleep; curl -X DELETE …/pods/$id).
```

The 31 KB environment variable has not been tried on Runpod
(**UNVERIFIED**; the existing fetchers ship about 10 KB this way). If it is
refused, ship the scripts in two variables, or clone the repo on the pod.
Watch `https://<pod>-8000.proxy.runpod.net/rebuild/run.log`. Delete the pod
when `EXIT` appears, and check that it is gone.

### 5.3 What `rebuild-volume.sh` does

Run `scripts/gpu/rebuild-volume.sh us --dry-run` anywhere to print the
plan. It works without the volume or a GPU. On the pod it runs, in this
order:

1. `auxiliary/` (TAE, LPIPS): each pinned URL is downloaded into a temp file,
   checked for size and SHA-256, then renamed.
2. The small trees: `upscaler`, `h3-to-ltx`, the FastH3 LoRA, the IC-LoRA.
3. `h3-base`, `h3-8step`, `h3-ref2va`, `ltx25`, `ltx25-dev`, `ltx2`,
   `ltx23`, the Wan set, the hy15 set, `mmaudio-44k-v2`, the upscalers.
4. The FP8 trees are printed as MANUAL (§3). A CPU pod cannot build them.
5. It runs `verify-weights.sh` over every cell. `--deep` adds the `sha:<dest>`
   cells.

Each Hub tree goes through `fetch-hub-tree.py`. It pins the revision from
`weights-revisions.tsv` and downloads into `.<dest>.partial-<stamp>`. Every
file is checked against the Hub (LFS SHA-256, or the git blob SHA-1 for small
files). If `FV_REBUILD_EXPECT_DIR` holds the other volume's lists, each file
must also match them. Only then are `.complete` and `sha256.txt` written and
the folder renamed.

The run is idempotent. A tree that is present, has `.complete` and passes
its cell is skipped. A tree that is present but unverified is left alone and
reported, and the run exits 1. `--only a,b` restricts the run to the named
trees.

### 5.4 Copy to the other volume

The 2026-09-27 method (docs/gaps/2026-09-27-volume-sync.md) re-downloads from
the Hub in the other volume's own data centre, at the same revisions. This
is the simplest option and avoids transfers between regions. Keep the first
volume's per-tree `sha256.txt` from `/srv/rebuild/<dest>/`, and give them to
the second run as `FV_REBUILD_EXPECT_DIR` (one `<dest>.sha256.txt` per tree),
so that every file must be equal on both volumes. Ship the lists in the
tarball.

The derived FP8 trees are copied between the volumes over HTTP (§3). A
failed copy is retried, not skipped.

### 5.5 Time, bytes, cost

| Item | Figure | Source |
|---|---|---|
| Bytes per volume | ~1165 GB (1127 GB Hub + 38.9 GB FP8) | §2 |
| Hub download rate | 150-260 MB/s per 8 vCPU pod (h3-ref2va 69 GB in 442 s US / 296 s EU; fastwan22 24.2 GB in 94 s) | fetch logs |
| Download time | about 1.3-2.1 h, plus hashing (every file is read again). Estimate **3-4 h per volume** | estimate |
| FP8 copy between regions | ~93 MB/s, 7 min for 38.9 GB | sync doc |
| FP8 regeneration (GPU) | 763 s + 323 s of writing, plus load, on an H200 | cold-start doc |
| CPU pod cost | about $1 per volume (4 h × $0.24/hr) | estimate |
| Storage | $120/month per 2000 GB volume | §1 |

Keep `HF_XET_HIGH_PERFORMANCE` off. It got 8 vCPU pods OOM-killed during the
sync. `fetch-hub-tree.py` uses 4 workers.

### 5.6 Verify

```bash
FV_WEIGHTS=/workspace/weights bash scripts/gpu/verify-weights.sh \
  aux fasth3-8step h3-base fasth3-4step-vsa fasth3-4step-dense sol-h3 sol-h3-spark \
  h3-ref2va h3-ref2va-turbo ltx25-two-stage ltx25-dev ltx25-a2v-guided ltx25-ic-lora-ingredients \
  ltx25-ref2v ltx2 ltx23 fastwan21-1.3b wan22-ti2v-5b fastwan22-ti2v-5b wan21-t2v-14b \
  sfwan21-1.3b hy15-480-t2v hy15-480-i2v hy15-720-t2v hy15-720-i2v mmaudio-44k-v2 upscalers text-fp8
# recorded per-file hashes (reads ~95 GB)
bash scripts/gpu/verify-weights.sh sha:h3-ref2va sha:fastwan22-ti2v-5b sha:ltx25-ic-lora-ingredients sha:mmaudio-44k-v2
FV_VERIFY_FP8_SHA=1 bash scripts/gpu/verify-weights.sh text-fp8   # ~40 GB read
```

`verify-weights.sh --list` prints every cell and what it needs.

## 6. Adding a new tree (the normal path, not a rebuild)

`scripts/gpu/fetch-hub-tree.sh <dest> <revision>` (from this container: it
creates the CPU pod, sets the backstop and deletes the pod afterwards) runs
against the EU volume (`fv-weights-h3-ltx-hy`, the default). EU only since
2026-10-06: there is no second volume to copy to. (While US existed, it ran
against US first, then against EU with the US `sha256.txt` as the third
argument; a rebuilt US would get the EU lists the same way.) Add the manifest
row, the revision row in `weights-revisions.tsv`, the per-file hashes in
`weights-sha256.tsv` (from `artifacts/runpod/fetch-<dest>-<volume>/sha256.txt`),
a `verify-weights.sh` cell, a line in `rebuild-volume.sh`'s `PLAN`, and a row
in §2 above. `FETCH_ADD_INTO=1 fetch-hub-tree.sh <dest> <rev>` adds a
row's missing files to an existing tree the same add-only way (temp folder,
Hub check, then each file renamed in; used for `ltx23-dev` on 2026-10-06).
On a 2 vCPU / 4 GB CPU pod use `FETCH_WORKERS=1` (four parallel 5 GB shards
were OOM-killed there). Do not edit `fetch-hub-tree.sh` while a fetch runs:
bash reads the script as it goes.

## 7. Money and safety checklist for a rebuild

- The balance is at least $8 and covers the pods (each fetcher checks
  `FV_MIN_BALANCE`).
- The owner has approved the download.
- Every pod has a detached DELETE backstop, and you checked each one is gone
  (`GET /pods/<id>` → 404).
- Nothing ran `rm` on the volume except a fetcher's own
  `.<name>.partial-*` folders.
- `weights-manifest.tsv`, `weights-revisions.tsv` and this page are updated
  for anything new.

## 8. Rebuilding fv-build (`pxy4hlsnwq`, 200 GB, EU-RO-1)

`fv-build` holds only caches (docs/dev/build-pod.md): `rustup/` (the stable
toolchain with rustfmt and clippy, per `rust-toolchain.toml`),
`cargo/registry/cache` and `cargo/git/db`, `sccache/` (40 GB cap),
`cuda-13.4/` (the NVIDIA redist tarballs, sha256-checked), `node-v22.23.3/`,
`playwright-1.56.1/`, `pw-browsers/`, `jobs/`, `logs/` and the pod-side
`ledger.tsv`. Agents' worktree snapshots and `CARGO_TARGET_DIR`s live on the
pod's **container disk** (`/root/fvb/{worktrees,target}/<agent>/`), not on
the volume. They are gone whenever the pod stops, whatever happens to the
volume. `FV_BUILD_TARGETS=volume` would put them back on the volume. CLAUDE.md
still describes the older on-volume layout.

Losing it loses no data. What is lost: the downloaded toolchains, the crate
cache, the sccache (so the first builds are cold), the job logs and the
pod-side ledger (the local copy is in `~/.config/fv-build/ledger.tsv`).

Rebuild:

```bash
scripts/dev/build-pod.sh volume-create   # POST /networkvolumes {name: fv-build, size: 200, dataCenterId: EU-RO-1}
scripts/dev/build-pod.sh up              # first boot installs everything onto the volume
```

Cold timings (2026-09-28, docs/dev/build-pod.md): about 2 min from first boot
to `ready` (the CUDA redist takes about 90 s), then 1-2 min for the
Node/Chromium extras in the background. A cold `cargo check` of the serve
crates took 3.5 min on 16 vCPU with a warm sccache, and 18 min with a cold
registry and cold sccache when the targets were on the volume. Expect
**about 25-30 min** until an agent's first check is back to normal speed
(**estimate**). Release `fv-serve` / `fv-gpucheck` builds take 6.5 min each
once sccache is warm. Cost: $14/month for the volume. The pod costs
$0.96/hr (cpu3c 32 vCPU) and stops itself after 20 idle minutes.
