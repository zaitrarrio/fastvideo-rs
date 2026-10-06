# Research: text/image-to-audio+video on an 8 GB GPU

Status: research note, 2026-09-30. Docs only: no GPU pods, no weight
downloads, no code changes.

The owner asked: which text-to-audio+video (T2AV) and image-to-audio+video
(I2AV) models, among the ones we have explored or others on Hugging Face,
could run in **8 GB of GPU memory** if we use every trick available?

Conventions (the same as [research-avatar-v2v.md](research-avatar-v2v.md)):

- Every external fact has a link. **INFERRED** marks a conclusion that no
  source states directly. **UNVERIFIED** marks a claim we did not measure or
  could not check (vendor or community VRAM claims, spec-sheet figures).
- "Quality" only repeats what a model's authors (or a competitor's paper)
  claim. No clip was generated for this note.
- Repo sizes and file lists come from
  `https://huggingface.co/api/models/<repo>?blobs=true`, fetched 2026-09-30.
  Parameter counts and dtypes come from the **safetensors headers** (a range
  read of the first bytes of each file; no weights were downloaded). Where
  only `.pt`/`.pth`/`.bin`/`.ckpt` exists, the size is given and the dtype
  is marked INFERRED.
- Our measurements are cited by repo path: [raw-inference.md](../perf/raw-inference.md)
  (RTX PRO 6000, 2026-09-30), [ltx25.md](../ports/ltx25.md) (LTX-2.5 offload
  and NVFP4 cells), [mmaudio.md](../ports/mmaudio.md), [wan.md](../ports/wan.md).
  **Nothing here was measured on an 8 GB card.**
- GB = 10^9 bytes (file sizes); GiB = 2^30 bytes (device memory).

---

## 0. Answer first

**Short answer.** Nothing that makes speech runs *fully resident* in 8 GB
at 16-bit. With 4-bit weights, three joint audio+video models with speech fit
resident: **NAVA 6.3B**, **DreamX-Creator 7B** and **UniAVGen 7B** (Ovi 11.7B
misses by about 0.4 GiB). All three are 50-step models, so on an 8 GB card
they take minutes per clip. The larger distilled models (**LTX-2.5 22B**,
**MagiHuman 15B**) run in 8 GB only with **block streaming from host RAM**.
Because they need few steps, block streaming is the faster route today, not
the slower one. For video without speech, our **Wan 1.3B / 5B** ports plus a
post-hoc audio model (MMAudio) fit resident once the text encoder is no longer
kept on the device.

We already have one measurement that matters here: LTX-2.5 22B in our
`--offload cpu` mode peaked at **5.72 GiB allocated / 6.31 GiB reserved** at
512p ([ltx25.md](../ports/ltx25.md), "Offload placement", 512p `cpu` row).
That run was on a 96 GB card, but the pool peak already fits an 8 GB budget.

### Top 3 with speech (8 GB card)

| # | Model | 8 GB verdict | Why | Porting work |
|---|---|---|---|---|
| 1 | **LTX-2.5 22B distilled** (T2AV, I2AV, A2V; Lightricks) | **Fits only with block streaming.** Measured pool peak 6.31 GiB reserved at 512p (bf16 blocks streamed) | Already ported and served. Distilled 8+3 steps with CFG = 1, so there are only 11 DiT forwards to stream. Joint speech, sound effects and music | No new model. Needs an **8 GB profile**: quantized host copies for the stream ring (FP8 on Ada, NVFP4 on Blackwell), so host RAM drops from ~41 to ~21 / ~12 GiB and PCIe traffic halves or quarters; a quantized Gemma 4 run once and evicted (or on the CPU) with the prompt cache; a cap on the cuDNN workspace; TAE decode (`taeltx2_3_wide`); partial residency. Ampere needs int4 weight-only kernels (we have none) |
| 2 | **NAVA 6.3B** (Baidu ERNIE; T2AV, I2AV, reference-voice timbre) | **Fits resident** at 4 bits (≈5.7 GiB at 720p). With the shipped FP8 file, 480p needs about 0.5 GiB streamed | Best numbers in its own comparison table against Ovi 1.1, MOVA, MagiHuman and LTX 2.3 (authors' claims). Apache-2.0. **Built on Wan2.2-5B dimensions** (3072 wide, FFN 14336, 30 layers). Uses UMT5, the Wan2.2 VAE and the **LTX-2.3 audio VAE + vocoder**, all of which we already have | New family: MMDiT with 10 double-stream + 20 single-stream blocks, 1D audio RoPE, 3-way CFG, UniPC, and a ReDimNet speaker encoder (optional). Load the FP8 file, add NVFP4. **Slow: 50 steps × 3-4 CFG passes** (minutes per clip) until someone distills it |
| 3 | **daVinci-MagiHuman 15B, distill 256p** (T2AV, TI2AV, audio-conditioned) | **Fits only with block streaming.** About 5 GiB of 4-bit or FP8 weights stay resident and the rest is streamed | 8 steps without CFG. At 256p there are only ~3k tokens, so compute is small and transfer dominates. Its authors claim an 80% win rate over Ovi 1.1, 61% over LTX 2.3, and a lower WER. FastVideo has a bit-exact reference port | New family (MMDiT, 40 layers), **T5Gemma 9B encoder** (quantize and evict, or run on the CPU), SAO VAE (we have a `stable_audio` port), Wan2.2 VAE (we have it). 256p only: the 540p/1080p SR stages are another 15B each |

Runners-up with speech:

- **DreamX-Creator 7B** (Apache, I2AV only; Wan-5B video tower plus a
  Wan-1.3B-shaped audio tower; fits resident at 4 bits; 50 steps).
- **UniAVGen 7B** (Apache, Wan-5B based; fits resident at 4 bits).
- **Ovi 1.1 / Hallo-Live 11.7B**. Hallo-Live is the few-step streaming
  distillation of Ovi. Both need partial streaming at 4 bits, and their audio
  VAE is MMAudio's, whose checkpoints are **CC-BY-NC**.
- **Talker-T2AV** (~1.1B AR + LIA-X renderer; Apache). Talking heads only;
  fits trivially.
- **Any small TTS + SoulX-FlashHead-1.3B or LiveTalk-1.3B.** Not joint, but
  fits resident and is the only near-real-time path on 8 GB.

### Top 3 without speech

| # | Model | 8 GB verdict | Porting work |
|---|---|---|---|
| 1 | **FastWan / SF-Wan 1.3B + MMAudio** (post-hoc soundtrack) | **Fits resident**, run in sequence (~3 GiB video stage, ≤ 6 GiB audio stage) | None new. An 8 GB profile: UMT5 encoded then evicted (or quantized, or on the CPU) with the prompt cache, TAEHV, FP8 on Ada (MXFP8 is Blackwell-only), Wan DiT evicted before the MMAudio stage. **MMAudio weights are CC-BY-NC**; for commercial use, port an Apache V2A model (ThinkSound) |
| 2 | **Wan2.2 TI2V-5B turbo + MMAudio** (T2V/I2V, 3 steps) | **Fits resident** with FP8 or 4-bit weights (≈6.1–6.7 GiB with FP8 at 480p–704p) | The same 8 GB profile. Ampere needs int4/int8 weight-only kernels or streaming, because bf16 is 9.3 GiB |
| 3 | **JavisDiT++ 2.1B** (native joint sounding video, Wan2.1-1.3B based; MIT) | **Fits resident** even at bf16 (≈5.5 GiB at 480p) | New family (Wan 1.3B blocks + MS-MoE + TA-RoPE) and an AudioLDM2 VAE + vocoder. **That component is CC-BY-NC-SA**. 50-step class, so minutes per clip |

Every speech model above also generates ambience, effects and music. If
quality matters more than speed, NAVA or LTX-2.5 without dialogue is a
stronger "no speech" option than #3.

### What every pick needs (one shared "8 GB profile")

1. **Evict the text encoder** after encoding, and serve repeat prompts from
   the prompt disk cache. The encoder can instead run quantized once (UMT5 Q8
   6.0 GB, T5Gemma Q6 11.9 GB, Gemma 4 Q4 8.4 GB) or on the CPU. Our
   measured Wan peaks (21.4–26.7 GiB) are mostly the resident UMT5 and
   workspace.
2. **Quantized weights**: NVFP4 on Blackwell (we have it for LTX's video FFN
   only), FP8 W8A8 on Ada and Blackwell (we have it), and **int4 weight-only
   on Ampere (we have none; `QuantMode::parse("int4")` is rejected in
   `crates/fastvideo-cudarc/src/wan/quant.rs`)**.
3. **Streaming with partial residency.** Keep as many blocks resident as
   fit, and stream the rest through the existing ring
   (`crates/fastvideo-cudarc/src/wan/offload.rs`, LTX `--offload cpu`). Keep
   the host copies quantized, too.
4. **TAE decode** (taew2_1, taew2_2, taeltx2_3_wide exist) or tiled VAE, and
   a **cap on the cuDNN workspace** plus pool trims between phases.
5. **FFN row chunking** (exists for LTX) and flash attention (our generic
   kernels are built for sm_75–120 by default; FP8 attention needs sm_89+,
   `crates/fastvideo-cudarc/build.rs`).
6. An **8 GiB memory-cap test cell** on our usual pods, plus one timing run
   on a real 8 GB consumer card (availability UNVERIFIED).

---

## 1. The 8 GB budget, the trick stack, and what each GPU generation can use

### 1.1 Budget

An 8 GB GeForce has 8 GiB. The desktop and driver take about 0.3–1 GiB when
the card drives a display (WanGP notes that 1–5 GB can be freed by turning
off browser GPU use:
https://github.com/deepbeepmeep/Wan2GP README). The CUDA context and the
cuBLAS/cuDNN workspaces take about 0.6 GiB. **INFERRED budget:**

- **7.6 GiB** for the process;
- **7.0 GiB** for weights, activations and decode.

A "fits resident" verdict below means weights + activations + 0.6 GiB of
overhead ≤ 7.6 GiB.

### 1.2 The trick stack and hardware support

| Trick | RTX 5060 / 5060 Ti 8 GB (Blackwell, sm_120) | RTX 4060 / 4060 Ti 8 GB (Ada, sm_89) | RTX 3060 Ti / 3070 8 GB (Ampere, sm_86) | Ours today |
|---|---|---|---|---|
| **4-bit weights and activations (NVFP4)**, 0.5625 B/param | yes | **no** | **no** | LTX video FFN on NVFP4 tensor cores; elsewhere a dequant-beforehand path ([techniques.md](../techniques.md), `nvfp4`) |
| **FP8 W8A8 (E4M3)**, 1 B/param | yes | yes | **no** | `FASTVIDEO_FP8=1` W8A8 on LTX-2 and Wan; MXFP8 (Blackwell) on H3 and Wan |
| **int4/int8 weight-only (GGUF-style, W4A16)**, ~0.56 / 1.06 B/param | yes | yes | yes (the only 4-bit option) | **none** |
| Text encoder run once then evicted, or on the CPU | all | all | all | Wan: prompt cache, UMT5 resident by default; LTX: Gemma dropped after encode, or streamed per layer (`TextResidency::Streamed`) |
| Sequential stage loading (text → DiT → decode → audio) | all | all | all | LTX `--offload cpu` does this; Wan keeps everything resident |
| TAE / tiled VAE decode | all | all | all | taew2_1, taew2_2, taeltx2_3_wide, taeh3 |
| Flash attention | own `flash_mma_fwd2` + cuDNN measured on sm_120 | generic kernels built (sm_89); FP8 attention sm_89+ | generic kernels built (sm_86) | consumer SMs were never benchmarked |
| **Block streaming** from pinned host RAM | PCIe 5.0 x8 | PCIe 4.0 x8 | PCIe 4.0 x16 | 2-slot ring with lookahead; byte-identical to resident ([ltx25.md](../ports/ltx25.md)) |

The PCIe link widths are spec-sheet figures (**UNVERIFIED**, not fetched
here). Effective host-to-device bandwidth is roughly 12 GB/s for PCIe 4.0 x8
and roughly 25 GB/s for PCIe 5.0 x8 or 4.0 x16 (**INFERRED**). For
reference, our RTX PRO 6000 run measured 57 GB/s.

**Speed scaling (INFERRED).** An 8 GB card has 24–46 SMs. The RTX PRO 6000
has 188. So compute-bound time is taken as **6–8×** our RTX PRO 6000
numbers, and **15–25×** H100 numbers. Speed classes:

- **RT**: real-time streaming.
- **fast**: under 1 min per 5 s clip.
- **medium**: 1–5 min.
- **slow**: over 5 min.

---

## 2. Per-model facts

"Params" and dtype come from the safetensors headers unless marked.
Components are listed as they are needed at inference.

### 2.1 Joint audio+video generators

| Model | License | Modes | Params (DiT) | Components (bytes on the Hub, dtype) | Native res / length | Vendor / community VRAM claim |
|---|---|---|---|---|---|---|
| **LTX-2.5 22B distilled** (in repo) | LTX-2 community licence (free below USD 10M revenue; [ltx-ref2v.md](../ports/ltx-ref2v.md) §2) | T2AV, I2AV, A2V, retake, extend; speech yes; clip | ~22B (LTX-2 19B DiT is 37.76 GB bf16, [ltx2.md](../ports/ltx2.md)) | Community NVFP4 transformer 18.72 GB, of which 4.8B params stay BF16 (header: U8 8.1B, F8 1.0B, BF16 4.8B; [BennyDaBall/…-nvfp4-comfy-v2](https://huggingface.co/BennyDaBall/LTX-2.5-22b-distilled-nvfp4-comfy-v2)). GGUF Q3_K_S 12.65 / Q4_K_M 15.69 / Q8_0 23.61 GB ([Abiray/LTX-2.5-Distilled-GGUF](https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF)). Gemma 4 12B text encoder: bf16 26.3 GB, Q4_K_M 8.42 GB ([elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF](https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF)), NVFP4 10.6 GB ([Deadshot699/…](https://huggingface.co/Deadshot699/ltx-2.5-gemma4-12b-comfy-nvfp4)). Video VAE 1.47, audio VAE 0.365, upsampler 0.996 GB | 1280×768×121 (our turbo), 768×512 validation; 24 fps | **Measured by us:** 512p `cpu` offload, pool 5.72 / 6.31 GiB, smi 7.8 GiB, 15.3 s e2e on RTX PRO 6000 ([ltx25.md](../ports/ltx25.md)). Resident 720p turbo: 55.9–66 GiB ([raw-inference.md](../perf/raw-inference.md)). Card: Q3_K "extreme low-VRAM" (UNVERIFIED) |
| **LTX-2.3 22B** (weights `ltx23` on volumes) | same | same | ~22B | GGUF Q2_K 7.94 / Q4_K_M 14.19 / Q8_0 22.76 GB, connectors 2.31 GB ([unsloth/LTX-2.3-GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF)). Official FP8 29.5 GB ([Lightricks/LTX-2.3-fp8](https://huggingface.co/Lightricks/LTX-2.3-fp8)), official NVFP4 (dev) 21.7 GB ([Lightricks/LTX-2.3-nvfp4](https://huggingface.co/Lightricks/LTX-2.3-nvfp4)) | as 2.5 | none on the cards read |
| **NAVA** ([baidu/NAVA](https://huggingface.co/baidu/NAVA), arXiv [2605.30073](https://arxiv.org/abs/2605.30073)) | **Apache-2.0**. Ships the LTX-2.3 audio VAE under the **LTX-2 community licence** (`params/LTX2/LICENSE`) | T2AV, **I2AV** (`image_path`), multi-speaker speech with **reference-voice timbre** (≤ 2 WAVs); clip | **6.297B** (`NAVA.safetensors` 25.19 GB **F32**). `NAVA_fp8.safetensors` 6.94 GB: F8_E4M3 5.66B + BF16 0.64B | UMT5-xxl 11.36 GB bf16. Wan2.2 VAE 2.82 GB (`.pth`). LTX-2.3 audio VAE + vocoder 0.365 GB. ReDimNet speaker encoder ~50 MB (card) | **1280×704** (960×960 also), "37 frames @ 24 fps ≈ 6 s" (card; INFERRED 37 latent frames), audio 25 tokens/s ≤ 10 s; 50 UniPC steps; 3-way CFG | "720p in ~1 min via 8-GPU Ulysses SP" (card; UNVERIFIED) |
| **daVinci-MagiHuman 15B** ([GAIR/daVinci-MagiHuman](https://huggingface.co/GAIR/daVinci-MagiHuman); [research-avatar-v2v.md](research-avatar-v2v.md) §2.1) | **Apache-2.0**. T5Gemma is under the Gemma terms (manual gate); the SAO VAE is under the Stability community licence | T2AV, TI2AV, audio-conditioned (undocumented); speech (7 languages); clip | **15.3B** (distill-256p transformer 61.20 GB **F32**; FastVideo bf16 conversion 30.64 GB) | T5Gemma 9B-9B 40.71 GB (Q6_K GGUF 11.94 GB, [realrebelai/…](https://huggingface.co/realrebelai/DaVinci_MagiHuman_fp8_merges); int8 10.16 GB, [DeepBeepMeep/MagiHuman](https://huggingface.co/DeepBeepMeep/MagiHuman)). SAO file 4.85 GB (the VAE is a small part). Wan2.2 VAE 2.82 GB. Turbo VAE 0.44 GB. FP8 distill 15.32 GB ([SanDiegoDude/daVinci-MagiHuman-FP8](https://huggingface.co/SanDiegoDude/daVinci-MagiHuman-FP8)); int8 15.33 GB (DeepBeepMeep) | 448×256 base, 25 fps, 4–5 s; SR to 540p/1080p (another 15B each) | 5 s 256p in 2.0 s on an H100 (README). Consumer cards "need CPU offload" (RTX 5090 example). Both UNVERIFIED |
| **Ovi 1.0 / 1.1** ([chetwinlow1/Ovi](https://huggingface.co/chetwinlow1/Ovi), arXiv [2510.01284](https://arxiv.org/abs/2510.01284)) | **Apache-2.0**. The audio VAE and vocoder are MMAudio's `ext_weights/v1-16.pth` (0.687 GB) and `best_netG.pt` (0.449 GB) ([download_weights.py](https://github.com/character-ai/Ovi/blob/main/download_weights.py)). MMAudio checkpoints are **CC-BY-NC 4.0** ([MMAudio README](https://github.com/hkchengrex/MMAudio#license)) | T2AV, I2AV; speech (`<S>…<E>`); clip | **11.661B** BF16 (video 5.566B + audio 6.094B), 23.32 GB per variant (720×720_5s, 960×960_5s, 960×960_10s). FP8 11.66 GB ([rkfg/Ovi-fp8_quantized](https://huggingface.co/rkfg/Ovi-fp8_quantized)) | UMT5 11.36 GB. Wan2.2 VAE 2.82 GB. MMAudio 16k VAE + vocoder 1.14 GB | 720×720 5 s (1.0); 960×960 5/10 s at 24 fps (1.1); 50 UniPC steps, video CFG 4 + audio CFG 3 + SLG | "Minimum 32 GB; 24 GB with fp8 + cpu_offload"; 83–140 s for 720×720×121 at 50 steps (card; GPU unnamed; UNVERIFIED) |
| **Hallo-Live** ([fudan-generative-ai/Hallo-Live](https://huggingface.co/fudan-generative-ai/Hallo-Live), arXiv [2604.23632](https://arxiv.org/abs/2604.23632)) | MIT (Ovi base: Apache; MMAudio VAE: CC-BY-NC) | T2AV avatar, **streaming**, causal with KV cache, DMD few-step; speech | 11.66B (INFERRED: `hallolive_dit.pt` is 23.32 GB, the same as Ovi bf16) | as Ovi | Ovi resolutions (INFERRED) | 20.38 FPS, 0.94 s latency on 2× H200 (authors). Training needs 8× H200 |
| **DreamX-Creator 1.0** ([GD-ML/DreamX-Creator](https://huggingface.co/GD-ML/DreamX-Creator), arXiv [2608.31106](https://arxiv.org/abs/2608.31106)) | **Apache-2.0** ([GitHub LICENSE](https://github.com/AMAP-ML/DreamX-Creator)) | **I2AV only** (first frame + text); speech (Verse-Bench talking case) and effects; clip | **7.06B** F32: video 5.0B (Wan2.2-5B), audio 1.419B (the same count as the Wan2.1-1.3B DiT), cross-modal 0.639B; 28.23 GB | UMT5 11.36 GB. Wan2.2 VAE 2.82 GB. DAC audio VAE 0.372B bf16 (0.743 GB). Optional 2K refiner (SR-DiT 5B, 10.0 GB) | 880 spatial tokens (≈1280×704), 24 fps, 5 s, 50 steps, multimodal CFG | Text-encoder and VAE CPU offload flags (README); no figure. Q8 GGUF 12.83 GB for a CPU-only Rust runtime ([EvoAwaken-Workshop/DreamX-Creator-gguf](https://huggingface.co/EvoAwaken-Workshop/DreamX-Creator-gguf)) |
| **UniAVGen** ([MCG-NJU/UniAVGen](https://huggingface.co/MCG-NJU/UniAVGen), arXiv [2511.03334](https://arxiv.org/abs/2511.03334)) | **Apache-2.0** | Joint AV generation and continuation, V2A dubbing, A2V (abstract); I2AV (pipeline tag, INFERRED); speech (lip-sync, timbre) | **7.06B** BF16 (Wan-5B blocks 4.91B + audio blocks 1.39B + a2v 0.43B + v2a 0.21B), 14.12 GB | UMT5 11.36 GB. Wan2.2 VAE 2.82 GB. `code2wav_bigvgan_model` 0.46 GB | not read (UNVERIFIED) | none found |
| **UniVerse-1** ([dorni/UniVerse-1-Base](https://huggingface.co/dorni/UniVerse-1-Base), arXiv [2509.06155](https://arxiv.org/abs/2509.06155)) | **Apache-2.0** (depends on Wan2.1-1.3B and ACE-Step 3.5B, both Apache) | I2AV ("from a reference image and a text prompt"); speech, instruments, ambience | **7.05B F32** (28.20 GB): Wan2.1-1.3B stitched with ACE-Step-v1-3.5B | UMT5 (Wan). ACE-Step text/lyric encoders, music DCAE and vocoder ([ACE-Step/ACE-Step-v1-3.5B](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B), 8.28 GB repo) | Wan 1.3B class (480p, INFERRED) | none found |
| **JavisDiT++ (v1.0)** ([JavisVerse/JavisDiT-v1.0-jav](https://huggingface.co/JavisVerse/JavisDiT-v1.0-jav), arXiv [2602.19163](https://arxiv.org/abs/2602.19163)) | MIT (weights). Uses the **AudioLDM2** VAE and vocoder, which are **CC-BY-NC-SA 4.0** ([cvssp/audioldm2](https://huggingface.co/cvssp/audioldm2)) | T2AV "sounding video"; speech not claimed | **2.1B** (GitHub table); 4.76 GB `.bin` + 0.26 GB LoRA (dtype INFERRED bf16) | UMT5 and Wan2.1 VAE (from Wan2.1-T2V-1.3B). AudioLDM2 VAE + vocoder | 240p–480p, 2–5 s, 16 fps ([GitHub](https://github.com/JavisVerse/JavisDiT)) | none found |
| **Talker-T2AV** ([HKUSTAudio/Talker-T2AV](https://huggingface.co/HKUSTAudio/Talker-T2AV), arXiv [2604.23586](https://arxiv.org/abs/2604.23586)) | **Apache-2.0**; LIA-X Apache-2.0 ([YaohuiW/LIA-X](https://huggingface.co/YaohuiW/LIA-X)) | **Talking head**: text + reference voice + reference face → speech + face video; also V2A and A2V | **1.068B** (Qwen3-0.6B backbone 0.75B BF16 + two diffusion heads and encoders 0.32B F32), 2.77 GB | WhisperX-VAE 6.91 GB `.ckpt`. LIA-X motion renderer 3.64 GB `.pt`. WavLM-Large speaker encoder (fetched separately) | 25 Hz motion/audio latents; portrait size set by LIA-X (INFERRED 512²) | none found |
| **MOVA-360p / 720p** ([OpenMOSS-Team/MOVA-360p](https://huggingface.co/OpenMOSS-Team/MOVA-360p), arXiv [2602.08794](https://arxiv.org/abs/2602.08794)) | Apache-2.0 | IT2VA (T2VA listed); lip-synced speech | **32B MoE, 18B active** (card). Two 14.3B video experts (28.58 GB each, bf16), audio DiT 1.419B, bridge 2.66B | UMT5 11.36 GB. Video VAE 0.25 GB. Audio VAE 0.74 GB | 360p / 720p | none read |
| **DreamID-Omni** ([XuGuo699/DreamID-Omni](https://huggingface.co/XuGuo699/DreamID-Omni), arXiv [2602.12160](https://arxiv.org/abs/2602.12160)) | licence field empty on the Hub; README says "academic research … only"; Ovi + MMAudio underneath | reference-to-AV (identity + voice); speech | **11.661B F32** (46.64 GB): the Ovi architecture | as Ovi | as Ovi | none |
| **MiniMax H3** (in repo) | MiniMax licence, **excluded in the US, EU, UK and KR** ([research-avatar-v2v.md](research-avatar-v2v.md) §6) | T2AV, I2AV, ref2v | ~35B; DiT 70.10 GB bf16; text encoder layers needed 46.86 GiB ([h3.md](../ports/h3.md)) | community "pruned" Q5_0 GGUF 13.92 GB, Qwen3-VL-32B NVFP4 15.69 GB ([Akalabeth12/…](https://huggingface.co/Akalabeth12/TextImageReference-to-Video-Audio)) | 832×480–768p | resident 52–60 GiB measured ([raw-inference.md](../perf/raw-inference.md)) |

### 2.2 Speech avatars driven by audio (need a TTS in front)

These models are not joint generators. Speech comes from a separate TTS or
audio LM, and they animate a portrait from it.
[research-avatar-v2v.md](research-avatar-v2v.md) §2.2–2.3 has the full list.
The small ones are listed here.

| Model | License | Params (header) | Other components | Claims |
|---|---|---|---|---|
| **SoulX-FlashHead-1.3B** ([Soul-AILab/SoulX-FlashHead-1_3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B)) | Apache-2.0 | Lite 1.527B F32 (6.11 GB); Pro 1.508B F32 (6.03 GB) | Lite: LTX-Video VAE 1.68 GB. Pro: Wan2.1 VAE 0.51 GB. wav2vec2-base 0.38 GB | Lite 96 FPS on one RTX 4090; Pro 10.8 FPS on a 4090 (authors); streaming; 512² demos |
| **LiveTalk-1.3B** ([GAIR/LiveTalk-1.3B-V0.1](https://huggingface.co/GAIR/LiveTalk-1.3B-V0.1)) | Apache-2.0 | 1.421B F32 (5.69 GB) | UMT5, Wan2.1 VAE, wav2vec2 | 24.82 FPS, 0.33 s first frame; "≥ 24 GB GPU", "~20 GB" for the offline script ([GitHub](https://github.com/GAIR-NLP/LiveTalk)) |
| OmniAvatar-1.3B, EchoMimicV3 (1.3B), StableAvatar | Apache / Apache / MIT | LoRA 0.36 GB + Wan 1.3B; 3.4–3.7 GB; 1.3B | wav2vec2 | [research-avatar-v2v.md](research-avatar-v2v.md) §2.2 |

### 2.3 Video-only small models and audio models to pair with them

| Model | License | Size (Hub) | Notes |
|---|---|---|---|
| **Wan2.1-T2V-1.3B** (FastWan / SF-Wan in repo) | Apache-2.0 | DiT 5.68 GB F32 (1.42B). GGUF Q4_K_M 0.98 / Q8_0 1.54 GB ([samuelchristlie/…](https://huggingface.co/samuelchristlie/Wan2.1-T2V-1.3B-GGUF)) | Vendor: "requires only 8.19 GB VRAM" ([Wan-AI card](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B), UNVERIFIED). **Ours measured: 21.5 GiB** FastWan max, 26.1–26.7 GiB SF-Wan, with UMT5 resident |
| **Wan2.2-TI2V-5B** (in repo) | Apache-2.0 | DiT 20.0 GB F32 (5.0B). GGUF Q4_K_M 3.43 / Q8_0 5.40 GB ([QuantStack/…](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)). VAE 2.82 GB | Vendor: "at least 24 GB (4090)" with offload + `--t5_cpu` ([Wan-AI card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), UNVERIFIED). **Ours measured: 21.4 GiB** (max arm) |
| UMT5-xxl encoder (Wan, Ovi, NAVA, DreamX, UniAVGen, LiveTalk) | Apache-2.0 | 11.36 GB bf16. GGUF Q8_0 6.04 / Q4_K_M 3.66 GB ([city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf)) | runs once per prompt |
| **MMAudio** (in repo: large-44k-v2) | code MIT; **checkpoints CC-BY-NC 4.0** | small_16k / small_44k 0.157B, medium 0.62B, large_v2 1.03B (f32 files 0.63 / 0.63 / 2.49 / 4.12 GB). CLIP DFN5B ViT-H 1.97 GB fp16 ([Kijai/MMAudio_safetensors](https://huggingface.co/Kijai/MMAudio_safetensors)). Synchformer 0.475 GB fp16 | Vendor: "around 6 GB" at 16-bit (README). **Ours measured upstream: 5.8 GiB allocated** (large, bf16). **No intelligible speech** ([mmaudio.md](../ports/mmaudio.md)) |
| ThinkSound ([FunAudioLLM/ThinkSound](https://huggingface.co/FunAudioLLM/ThinkSound)) | **Apache-2.0** | full 21.06 GB, light 5.73 GB `.ckpt`, VAE 2.52 GB, Synchformer 0.95 GB | Apache V2A alternative to MMAudio; optional MLLM chain-of-thought step (not needed for plain V2A, INFERRED) |
| HunyuanVideo-Foley ([tencent/HunyuanVideo-Foley](https://huggingface.co/tencent/HunyuanVideo-Foley)) | Tencent Hunyuan community licence (same family excludes EU/UK/KR, [research-avatar-v2v.md](research-avatar-v2v.md) §6) | 10.30 GB; XL 5.85 GB; VAE 1.49 GB | XL release adds "offload inference … significantly reducing VRAM" (card) |
| Stable Audio Open Small ([stabilityai/stable-audio-open-small](https://huggingface.co/stabilityai/stable-audio-open-small)) | Stability AI community licence | 1.68 GB | text-to-audio only (no video conditioning); our `stable_audio` port covers it ([stable-audio.md](../ports/stable-audio.md)) |
| LTX-2.3 Foley V2A LoRA ([Lightricks/LTX-2.3-22b-LoRA-Foley-V2A](https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Foley-V2A)) | LTX-2 community (gated) | 0.227 GB on the 22B base | only useful if LTX is already loaded |

---

## 3. Fit arithmetic

### 3.1 Model

The weights model:

- `W = P × (0.97 × b + 0.03 × 2)` bytes.
- `b` is 2 (bf16), 1 (FP8) or 0.5625 (NVFP4 or Q4_K: 4 bits plus a scale
  per 16).
- 3% of parameters (norms, embeddings, modulation, heads) stay bf16
  (INFERRED).

The activation model, for the one block in flight (**INFERRED**):

- Per block: `S × d × 12 B`. That is an f32 residual plus q, k, v and the
  attention output in bf16.
- Plus an FFN row chunk: `4096 × F × 4 B`.
- Plus about 0.1 GiB of tables and latents.
- Flash attention adds only O(S) workspace.
- CFG passes run one after another, so they do not multiply activations.
- A dual-tower model adds its second tower's `S_a × d_a × 12 B`.

Tokens:

- **Wan2.2 VAE** (16×16×4, patch 2): 32×32 px per token.
  - 832×480 = 390 tokens/frame.
  - 1280×704 = 880 tokens/frame.
  - 121 frames = 31 latent frames.
- **Wan2.1 VAE** (8×8×4, patch 2): 16×16 px per token. 832×480×81 = 32,760
  tokens.

Overhead: 0.6 GiB. Budget: 7.6 GiB per process (§1.1).

### 3.2 Resident fit (weights + activations + 0.6 GiB overhead, GiB)

| Model, workload | S (video) | bf16 | FP8 | 4-bit | Verdict |
|---|---:|---:|---:|---:|---|
| Wan2.1 1.3B, 832×480×81 | 32,760 | 2.64 + 0.80 + 0.6 = **4.0** | **2.8** | **2.2** | fits at every precision |
| FlashHead / LiveTalk 1.3B, 512² block (+ KV cache ~2 GiB, INFERRED) | ~6k | 3.7 (+2) | 2.4 (+2) | 1.8 (+2) | fits |
| JavisDiT++ 2.1B, 832×480×81 | 32,760 | **5.5** | 3.6 | 2.7 | fits at every precision |
| Wan2.2 5B, 832×480×121 | 12,090 | 10.7 | **6.1** | 4.2 | FP8 / 4-bit |
| Wan2.2 5B, 1280×704×121 | 27,280 | 11.2 | **6.7** | 4.7 | FP8 / 4-bit |
| **NAVA 6.3B**, 832×480, 37 latent frames | 14,430 | 13.2 | 6.46 (the shipped FP8 file) + 0.9 + 0.6 = **8.0** | 3.6 + 0.9 + 0.6 = **5.1** | **4-bit resident**; FP8 needs ~0.5 GiB streamed |
| NAVA 6.3B, 1280×704, 37 latent frames | 32,560 | 13.7 | 8.4 | **5.7** | 4-bit resident |
| DreamX-Creator 7.06B, 480p 5 s | 12,090 | 14.6 | 8.3 | **5.5** | 4-bit resident |
| DreamX-Creator 7.06B, native 880 tokens, 5 s | 27,280 | 15.2 | 8.8 | **6.0** | 4-bit resident |
| UniAVGen 7.06B, 480p 5 s | 12,090 | 14.5 | 8.2 | **5.4** | 4-bit resident |
| UniVerse-1 7.05B, 832×480×81 | 32,760 | 14.7 | 8.4 | **5.6** | 4-bit resident |
| Ovi 11.66B, 480×480×121 | 6,975 | 23.1 | 12.6 | **8.0** | **misses by ~0.4 GiB**: stream 2–3 block pairs |
| Ovi 11.66B, 704×704×121 | 15,004 | 23.4 | 12.8 | 8.2 | partial streaming |
| MagiHuman 15.3B, 448×256×101 (d ≈ 5120, INFERRED) | ~3,000 | 29.7 | 15.9 | 9.8 | **streaming only** |
| LTX-2.5 22B, 768×512×121 | 6,144 | ~44 | ~23 | ~13 | **streaming only** (measured, §3.3) |
| MOVA 18B active (32B total) | — | ~37 active | ~19 | ~11 | streaming only; host RAM holds both experts |
| H3 ~35B + 47 GiB encoder | — | 65+ | 33+ | 19+ | streaming only; impractical; licence-excluded |

Worked example, NAVA at 4 bits and 480p:

- **Weights:** `6.3e9 × (0.97 × 0.5625 + 0.06) = 3.81e9 B = 3.55 GiB`.
- **Activations:** `14,430 × 3072 × 12 = 0.50 GiB`, plus an FFN chunk of
  `4096 × 14336 × 4 = 0.22 GiB`, plus 0.1 GiB of tables and latents. Total
  0.8 GiB, with about 150 audio tokens on top (negligible).
- **Total:** 3.55 + 0.8 + 0.6 = **5.0 GiB**. That leaves about 2.5 GiB for
  the taew2_2 decode, or for a 720p run.

Notes on items outside the DiT:

- **Streaming models and the KV cache.** LiveTalk, FlashHead and Hallo-Live
  carry a KV cache: `2 × tokens_in_window × d × 2 B × layers`. For Wan 1.3B
  at 512² with a 12-latent-frame window that is about 2.1 GiB (INFERRED).
  For Hallo-Live (two 3072-wide towers, 30 layers each, window about 4k
  tokens) it is about 1.5 GiB. That pushes Hallo-Live firmly into
  streaming.
- **Decode.** TAE decoders need well under 1 GiB (INFERRED). Full VAEs have
  to be tiled or chunked, and the DiT is evicted first. The Wan2.2 VAE's
  largest conv at 704×1280 needs 4.6 GB per 2 latents ([wan.md](../ports/wan.md)),
  so it needs 1-latent chunks and spatial tiles on 8 GB.

### 3.3 Streaming ("slow category", but the fast route for distilled models)

Transfer time per clip:

`t_xfer = N_forwards × (W_q − W_resident) / BW_pcie`

Our ring prefetches block `i+1` while block `i` computes
([ltx25.md](../ports/ltx25.md)), so e2e ≈ max(compute, transfer) plus the
first block.

Assumptions: `BW_pcie` is 11.2 GiB/s on PCIe 4.0 x8 and 23 GiB/s on
PCIe 5.0 x8 (INFERRED). `W_resident` is ~3–5 GiB of blocks kept resident
(partial residency).

| Model, workload | Forwards | Streamed per forward | Total streamed | t_xfer 4.0 x8 / 5.0 x8 | Compute, INFERRED from our or vendor timings | Host RAM (pinned) |
|---|---:|---:|---:|---|---|---|
| **LTX-2.5**, 512p, 8 + 3 steps, CFG 1 | 11 | bf16 34.6 GiB (measured: "8 forwards × 48 blocks = 277 GiB") / FP8 ~17 / NVFP4 ~10 | 380 / 190 / 107 GiB | 34 / 17 / **9.5 s**; 17 / 8 / **4.7 s** | resident 512p e2e 7.17 s on RTX PRO 6000 ([ltx25.md](../ports/ltx25.md)) × 6–8 = **45–60 s** | bf16 ~41 GiB (too much for a 32 GB host), FP8 ~21 GiB, NVFP4 ~12 GiB, plus a Gemma copy |
| **MagiHuman** distill 256p, 8 steps, no CFG | 8 | FP8 ~9.3 GiB / 4-bit ~3.6 GiB | 74 / 29 GiB | 6.6 / **2.6 s** | 1.6 s base on an H100 (README) × 15–25 = **25–40 s** | FP8 ~14 GiB, 4-bit ~8 GiB, plus T5Gemma |
| **Ovi** 4-bit, 480×480, 50 steps × 2–3 passes | 100–150 | ~0.5–1 GiB | 50–150 GiB | 5–13 s | 83–118 s at 720×720 (card) × 15–25, times ~0.5 for 480² = **10–25 min** | ~7 GiB |
| **MOVA** 4-bit, active expert | ~100 (50 steps × CFG, INFERRED) | ~6 GiB | ~600 GiB | ~55 s | tens of minutes (INFERRED) | ~18 GiB (both experts) |

**Measured anchor:** LTX-2.5 `cpu` offload at 512p peaked at **5.72 /
6.31 GiB** (allocated / reserved) and **7.8 GiB smi**, e2e 15.3 s on a
57 GB/s link, frames byte-identical to resident ([ltx25.md](../ports/ltx25.md)).
The 7.8 GiB smi includes the CUDA context and a cuDNN conv workspace cache
that "grows to the largest conv and is never shrunk" (the same doc). On a
real 8 GiB card that cache has to be capped. The doc already names
releasing it as "the next cut if a smaller card needs one". The ring slots
are 2 × ~0.79 GB of bf16 blocks; FP8 slots save ~0.7 GiB.

### 3.4 Text encoders (run once, then evict)

| Encoder | bf16 | 8 GB options |
|---|---:|---|
| UMT5-xxl (Wan, Ovi, NAVA, DreamX, UniAVGen, JavisDiT++, LiveTalk) | 11.36 GB | Q8_0 6.04 GB or Q4_K_M 3.66 GB fits alone, then evict. Or per-layer streaming. Or the CPU (a one-off per prompt; cache hits are free) |
| Gemma 4 12B (LTX-2.5) | 26.3 GB | Q4_K_M 8.42 GB does **not** fit alone, so per-layer streaming (exists: `TextResidency::Streamed`) or the CPU. An uncached streamed encode costs 43–47 s on the pod ([ltx25.md](../ports/ltx25.md)). On LTX-2 a new prompt reads 47 GB of f32 Gemma ([ltx2.md](../ports/ltx2.md)). A quantized host copy is needed |
| T5Gemma 9B encoder (MagiHuman) | ~18 GB needed (FastVideo notes) | Q6_K file 11.94 GB, int8 10.16 GB. Per-layer streaming or the CPU |
| Qwen3-0.6B (Talker-T2AV) | 1.5 GB | resident |

Quantizing the text encoder changes the conditioning. That is a quality
risk and was not measured.

---

## 4. Verdicts, reuse, speed and quality evidence

| Model | License | Modes | Params | 8 GB verdict | Key trick | Reuse (our ports) | Speed on 8 GB (INFERRED) | Quality evidence (claims) |
|---|---|---|---|---|---|---|---|---|
| **LTX-2.5 22B** | LTX-2 community | T2AV, I2AV, A2V, speech | ~22B | **streaming** (measured 6.31 GiB pool at 512p) | FP8/NVFP4 host copies + streaming + Gemma quantized/streamed + TAE | **full** (served) | **medium** (~1 min at 512p) | NAVA table: LTX 2.3 WER 0.106, Sync-C 7.25. MagiHuman claims a 60.9% win over LTX 2.3. NVFP4 FFN vs bf16, measured by us: LPIPS 0.19–0.25, PSNR 18–19 dB (the bf16 "chaos floor" policy, [ltx25.md](../ports/ltx25.md)) |
| **NAVA** | Apache (+ LTX audio VAE community licence) | T2AV, I2AV, timbre | 6.3B | **resident at 4-bit**; FP8 + ~0.5 GiB streamed | NVFP4 (Blackwell) / FP8 + partial streaming (Ada) / int4 W4A16 (Ampere) | Wan 5B dims, UMT5, Wan2.2 VAE + taew2_2, **LTX audio VAE + vocoder** | **slow** (50 steps × 3-way CFG: ~8–11 min at 480p, ~20–30 min at 720p) | Best Sync-C 7.79, WER 0.099, video quality 0.659 against Ovi 1.1, MOVA, MagiHuman and LTX 2.3 (its own table). Seed-TTS WER 5.81 |
| **MagiHuman 15B** distill 256p | Apache (+ Gemma, SAO terms) | T2AV, TI2AV, A2V | 15.3B | **streaming** | 4-bit/FP8 host copies + partial residency; T5Gemma on the CPU | Wan2.2 VAE, SAO VAE (`stable_audio`) | **fast–medium** (~30–45 s + encode) | 80% win vs Ovi 1.1, 60.9% vs LTX 2.3, WER 14.6% (own README). In NAVA's table it has the highest WER (0.151) |
| DreamX-Creator | Apache | I2AV, speech | 7.06B | **resident at 4-bit** | NVFP4 / int4 | Wan 5B + Wan-1.3B-shaped audio tower, UMT5, Wan2.2 VAE | **slow** (50 steps + multimodal CFG) | "competitive with state-of-the-art open-source" (abstract) |
| UniAVGen | Apache | joint, continuation, V2A, A2V | 7.06B | **resident at 4-bit** | NVFP4 / int4 | Wan 5B blocks, UMT5, Wan2.2 VAE | slow (steps UNVERIFIED) | better sync, timbre and emotion with 1.3M vs 30.1M training samples (abstract) |
| UniVerse-1 | Apache | I2AV, speech, music | 7.05B | **resident at 4-bit** | NVFP4 / int4 | Wan 1.3B blocks; ACE-Step new | slow (UNVERIFIED) | "strong alignment for speech" (abstract) |
| Ovi 1.0 / 1.1 | Apache + **NC audio VAE** | T2AV, I2AV, speech | 11.66B | **partial streaming** at 4-bit (misses by ~0.4 GiB) | 4-bit + stream 2–3 block pairs | **both towers are Wan-5B-shaped**, UMT5, Wan2.2 VAE; MMAudio 16k VAE code close to our 44k port | **slow** (10–25 min at 480²) | Omni-LiveAvatar's table: UTMOS 3.10, SyncNet 6.88. NAVA's table: WER 0.102 |
| Hallo-Live | MIT + **NC audio VAE** | T2AV streaming avatar | 11.66B | **streaming** (weights + KV) | as Ovi | as Ovi + SF-Wan-style causal rollout | **medium** (few-step; about 1 fps INFERRED) | comparable to Ovi on VideoAlign and Sync-C at 16× throughput (abstract). Omni-LiveAvatar reports its SyncNet at 4.50 |
| Talker-T2AV | Apache | talking head (T2AV, V2A, A2V) | 1.07B + renderer | **resident** | none needed | none (new: LLM + LIA-X) | **fast** (INFERRED) | beats dual-branch baselines on lip-sync, video and audio (abstract) |
| TTS + SoulX-FlashHead / LiveTalk | Apache | image + audio → avatar | 1.4–1.5B | **resident** | bf16/FP8, UMT5 evicted | **W1.3 causal rollout** ([research-avatar-v2v.md](research-avatar-v2v.md) §2.3) | **RT for FlashHead Lite** (96 FPS on a 4090, about ¼ of that on 30–36 SMs: ~24 FPS, INFERRED); LiveTalk ~5 FPS | HDTF/VFHQ SOTA (FlashHead); matches full-step baselines at 20× less cost (LiveTalk) |
| FastWan / SF-Wan 1.3B + MMAudio | Apache + **NC** (MMAudio) | T2V (+ soundtrack), no speech | 1.42B + 1.03B | **resident**, in sequence | UMT5 evicted, TAEHV | **full** | **fast** (~30 s: 2.26 s + ~2 s on RTX PRO 6000 × 6–8); SF-Wan streaming ~2 FPS at 832×480 | MMAudio: ambience only; speech unintelligible ([mmaudio.md](../ports/mmaudio.md)) |
| Wan2.2 TI2V-5B turbo + MMAudio | Apache + NC | T2V, I2V (+ soundtrack) | 5.0B + 1.03B | **resident** at FP8/4-bit | FP8 (Ada/Blackwell), int4 (Ampere) | **full** | **fast** (~25 s: 1.49 s + 2 s × 6–8) | — |
| JavisDiT++ | MIT + **NC-SA** (AudioLDM2) | T2AV sounding video | 2.1B | **resident** (even bf16) | UMT5 evicted | Wan 1.3B blocks, UMT5, Wan2.1 VAE | medium–slow (steps UNVERIFIED) | "significantly outperforming prior approaches" (abstract) |
| MOVA | Apache | IT2VA, speech | 18B active / 32B | streaming only | 4-bit, both experts in host RAM | Wan-A14B-shaped experts (not in the loader) | slow | Elo/win-rate claims on its card (not read) |
| H3 | excluded licence | T2AV, I2AV | ~35B | streaming only; **impractical** | — | full, but licence-excluded | very slow | — |

---

## 5. Checked and set aside

| Candidate | Why it is not in the tables |
|---|---|
| **AV-DiT** (arXiv [2406.07686](https://arxiv.org/abs/2406.07686)) | No official weights found on the Hub. The only hits are unofficial experiment repos (`Fishy1234/DiT_XL2_aist_*`, 0 downloads) |
| **SyncFlow** (arXiv [2412.15220](https://arxiv.org/abs/2412.15220)) | No weights found on the Hub |
| **TAVDiff** | The arXiv paper with that name ([2504.14267](https://arxiv.org/abs/2504.14267)) is video **saliency prediction**, not a generator |
| **Seeing and Hearing** (arXiv [2402.17723](https://arxiv.org/abs/2402.17723)) | An optimization-time ImageBind aligner over older video and audio models; no separate weights |
| **MMDisCo** ([AkioHayakawa/MMDisCo](https://huggingface.co/AkioHayakawa/MMDisCo), MIT) | 51–156 MB guidance modules over AudioLDM + AnimateDiff or Auffusion + VideoCrafter2 (2024 bases). Fits 8 GB trivially, but that is 2024-era quality and the base models carry their own licences |
| **JavisDiT v0.1** (OpenSora-based, 7.45 GB) | Superseded by JavisDiT++ |
| **Omni-LiveAvatar** (LTX-2 distilled streaming, arXiv [2608.13602](https://arxiv.org/abs/2608.13602)) | "Code and checkpoints are not released yet" ([GitHub](https://github.com/Aoko955/Omni-LiveAvatar)). **Watch it**: it is a real-time LTX-2 student and would reuse our LTX port |
| Ripple (2607.26818), MM-Sonate (2601.01568), StreamChar (2605.25659), Encore (2609.04249), 3MDiT (2511.21780), CineDance (2606.09639) | No weights found on the Hub (searched 2026-09-30) |
| TaoMate ([TaoLiveAIGC/TaoMate](https://huggingface.co/TaoLiveAIGC/TaoMate)) | H3-based, so licence-excluded |
| `luyu1021/turbo_t2av` | Its card is a copy of the MOVA card, the core is 65.3 GB, and what "turbo" means is unclear (UNVERIFIED) |
| Wan2GP (`DeepBeepMeep/*`) | This is a runner rather than a model. It runs Ovi, MagiHuman and LTX-2.x on small cards with MMGP offload and int8/GGUF/NVFP4. Its own docs say "The old fixed '6 GB / 12 GB / 20 GB' model tiers are … not reliable" ([docs/MODELS.md](https://github.com/deepbeepmeep/Wan2GP/blob/main/docs/MODELS.md)). So there is no per-model 8 GB claim to cite (UNVERIFIED) |

---

## 6. Porting work per pick

The shared 8 GB profile (§0) comes first. It benefits every pick and the
models we already serve.

**LTX-2.5 on 8 GB (speech #1).** No new model is needed. Work:

1. Store the streamed blocks in FP8 (the W8A8 path exists) or NVFP4, not
   bf16. On Blackwell, NVFP4 has to cover every linear, not just the video
   FFN.
2. Keep a quantized Gemma 4 host copy, stream it per layer, and add a CPU
   option. Keep the prompt disk cache on.
3. Release the cuDNN workspace cache in the phase trims, and cap cuBLAS
   workspaces.
4. Make `taeltx2_3_wide` the default decode for the profile.
5. Add partial residency to the ring: K blocks resident, the rest streamed.
6. For Ampere, add int4/int8 weight-only GEMMs. These do not exist yet.

Test: an 8 GiB memory-cap cell in `runpod-matrix.sh ltxoffload`, then one
consumer card. Host RAM: ≥ 32 GB with FP8 blocks (INFERRED).

**NAVA (speech #2).** New family `nava`:

- Double-stream blocks with separate QKV/FFN over a joint attention of
  `[video; audio]`, then single-stream blocks, all at Wan-5B dims.
- 3D RoPE for video and 1D for audio; text cross-attention.
- UniPC, 50 steps; 3-way CFG (`video_align_guidance_scale`,
  `audio_align_guidance_scale`).
- ReDimNet speaker embedding (optional).
- LTX-2.3 audio VAE + vocoder: check that it is the same module as our
  ltx2 audio decode (the file name `ltx-2.3-22b-dev_audio_vae.safetensors`
  matches unsloth's LTX-2.3 split, INFERRED).
- UMT5 and Wan2.2 VAE (+ taew2_2) from the Wan port.
- A FP8 loader for `NAVA_fp8.safetensors`, then NVFP4.

Weights: 25.2 GB F32 plus 6.9 GB FP8. Per CLAUDE.md a large new download
needs owner approval, and the weights go on the EU volume (EU only since
2026-10-06, docs/ops/runpod-volumes.md). Its prompts are
trained on Chinese dense captions; the card recommends a Qwen3-4B rewriter
for short or English prompts. It is slow until distilled.

**MagiHuman distill 256p (speech #3).** New family:

- Use FastVideo's bit-exact port as the oracle
  ([research-avatar-v2v.md](research-avatar-v2v.md) §2.1).
- T5Gemma encoder: new, quantized and streamed, or on the CPU.
- SAO VAE via the `stable_audio` port; Wan2.2 VAE or Turbo VAE for video.
- Streaming over 40 layers.

On 8 GB the SR stages are out of scope.

**Without speech.**

1. **FastWan/SF-Wan + MMAudio:** the 8 GB profile only. Swap MMAudio for
   ThinkSound if a commercial licence matters.
2. **Wan 5B turbo + MMAudio:** the same. On Ampere it needs int4 kernels or
   streaming.
3. **JavisDiT++:** new family on Wan 1.3B blocks (MS-MoE, TA-RoPE) plus the
   AudioLDM2 VAE + vocoder (NC-SA).

---

## 7. Open questions

- None of the fit numbers has been run on an 8 GB card. The activation
  model (§3.1) is an estimate. The LTX streaming peak is the only measured
  8 GB-class number, and it came from a 96 GB card.
- 4-bit quality for NAVA, Ovi, DreamX and MagiHuman is unmeasured. Our only
  4-bit evidence is LTX's NVFP4 video FFN (LPIPS 0.19–0.25 vs bf16).
- Consumer-card speed: the 6–8× (vs RTX PRO 6000) and 15–25× (vs H100)
  factors are SM-count estimates. Our kernels were never benchmarked on
  sm_86 or sm_89.
- NAVA: "37 frames" is read as latent frames (≈145 video frames at 24 fps
  is about 6 s, which matches the card's "≈ 6 s"). The single-GPU speed is
  not stated.
- Licences to read before any product use: the LTX-2 community licence as it
  applies to NAVA's bundled audio VAE; the MMAudio CC-BY-NC terms as they
  apply to Ovi's audio VAE and vocoder; AudioLDM2's CC-BY-NC-SA terms for
  JavisDiT++; DreamID-Omni's missing licence field.
- Host RAM: streaming picks assume 32 GB of system RAM (FP8 blocks + a
  quantized encoder). A 16 GB host would need NVFP4 blocks and CPU-side text
  encoding from disk.

---

## Sources

Hub metadata and headers (all fetched 2026-09-30 via
`https://huggingface.co/api/models/<repo>?blobs=true` and safetensors
header range reads):

- [baidu/NAVA](https://huggingface.co/baidu/NAVA)
- [chetwinlow1/Ovi](https://huggingface.co/chetwinlow1/Ovi)
- [rkfg/Ovi-fp8_quantized](https://huggingface.co/rkfg/Ovi-fp8_quantized)
- [fudan-generative-ai/Hallo-Live](https://huggingface.co/fudan-generative-ai/Hallo-Live)
- [GAIR/daVinci-MagiHuman](https://huggingface.co/GAIR/daVinci-MagiHuman)
- [SII-GAIR/daVinci-MagiHuman-Distill-256p](https://huggingface.co/SII-GAIR/daVinci-MagiHuman-Distill-256p)
- [SanDiegoDude/daVinci-MagiHuman-FP8](https://huggingface.co/SanDiegoDude/daVinci-MagiHuman-FP8)
- [DeepBeepMeep/MagiHuman](https://huggingface.co/DeepBeepMeep/MagiHuman)
- [realrebelai/DaVinci_MagiHuman_fp8_merges](https://huggingface.co/realrebelai/DaVinci_MagiHuman_fp8_merges)
- [GD-ML/DreamX-Creator](https://huggingface.co/GD-ML/DreamX-Creator)
- [EvoAwaken-Workshop/DreamX-Creator-gguf](https://huggingface.co/EvoAwaken-Workshop/DreamX-Creator-gguf)
- [MCG-NJU/UniAVGen](https://huggingface.co/MCG-NJU/UniAVGen)
- [dorni/UniVerse-1-Base](https://huggingface.co/dorni/UniVerse-1-Base)
- [JavisVerse/JavisDiT-v1.0-jav](https://huggingface.co/JavisVerse/JavisDiT-v1.0-jav)
- [HKUSTAudio/Talker-T2AV](https://huggingface.co/HKUSTAudio/Talker-T2AV)
- [YaohuiW/LIA-X](https://huggingface.co/YaohuiW/LIA-X)
- [OpenMOSS-Team/MOVA-360p](https://huggingface.co/OpenMOSS-Team/MOVA-360p)
- [luyu1021/turbo_t2av](https://huggingface.co/luyu1021/turbo_t2av)
- [XuGuo699/DreamID-Omni](https://huggingface.co/XuGuo699/DreamID-Omni)
- [AkioHayakawa/MMDisCo](https://huggingface.co/AkioHayakawa/MMDisCo)
- [Soul-AILab/SoulX-FlashHead-1_3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B)
- [GAIR/LiveTalk-1.3B-V0.1](https://huggingface.co/GAIR/LiveTalk-1.3B-V0.1)
- [unsloth/LTX-2.3-GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF)
- [Abiray/LTX-2.5-Distilled-GGUF](https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF)
- [BennyDaBall/LTX-2.5-22b-distilled-nvfp4-comfy-v2](https://huggingface.co/BennyDaBall/LTX-2.5-22b-distilled-nvfp4-comfy-v2)
- [Lightricks/LTX-2.3-nvfp4](https://huggingface.co/Lightricks/LTX-2.3-nvfp4)
- [Lightricks/LTX-2.3-fp8](https://huggingface.co/Lightricks/LTX-2.3-fp8)
- [elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF](https://huggingface.co/elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF)
- [Deadshot699/ltx-2.5-gemma4-12b-comfy-nvfp4](https://huggingface.co/Deadshot699/ltx-2.5-gemma4-12b-comfy-nvfp4)
- [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
- [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
- [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)
- [samuelchristlie/Wan2.1-T2V-1.3B-GGUF](https://huggingface.co/samuelchristlie/Wan2.1-T2V-1.3B-GGUF)
- [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf)
- [hkchengrex/MMAudio](https://huggingface.co/hkchengrex/MMAudio)
- [Kijai/MMAudio_safetensors](https://huggingface.co/Kijai/MMAudio_safetensors)
- [FunAudioLLM/ThinkSound](https://huggingface.co/FunAudioLLM/ThinkSound)
- [tencent/HunyuanVideo-Foley](https://huggingface.co/tencent/HunyuanVideo-Foley)
- [stabilityai/stable-audio-open-small](https://huggingface.co/stabilityai/stable-audio-open-small)
- [Lightricks/LTX-2.3-22b-LoRA-Foley-V2A](https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Foley-V2A)
- [cvssp/audioldm2](https://huggingface.co/cvssp/audioldm2)
- [ACE-Step/ACE-Step-v1-3.5B](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)
- [google/t5gemma-9b-9b-ul2](https://huggingface.co/google/t5gemma-9b-9b-ul2)

READMEs and code:

- Ovi: [README](https://github.com/character-ai/Ovi) (memory table, fp8
  notes) and [download_weights.py](https://github.com/character-ai/Ovi/blob/main/download_weights.py)
- [MMAudio README](https://github.com/hkchengrex/MMAudio) (licence, "around
  6 GB")
- [Hallo-Live](https://github.com/fudan-generative-vision/Hallo-Live)
- [LiveTalk](https://github.com/GAIR-NLP/LiveTalk) (≥ 24 GB, ~20 GB)
- [DreamX-Creator](https://github.com/AMAP-ML/DreamX-Creator) and its
  `audio_video_generation/README.md` (880 tokens, 50 steps, offload flags)
- [JavisDiT](https://github.com/JavisVerse/JavisDiT) (2.1B, 240p–480p,
  AudioLDM2)
- [Talker-T2AV](https://github.com/zhenye234/Talker-T2AV)
- [UniVerse-1 code](https://github.com/Dorniwang/UniVerse-1-code)
- [Omni-LiveAvatar](https://github.com/Aoko955/Omni-LiveAvatar)
- [Wan2GP README and docs](https://github.com/deepbeepmeep/Wan2GP)

Papers (abstracts read via the arXiv API):

- [2605.30073](https://arxiv.org/abs/2605.30073) NAVA
- [2510.01284](https://arxiv.org/abs/2510.01284) Ovi
- [2604.23632](https://arxiv.org/abs/2604.23632) Hallo-Live
- [2608.31106](https://arxiv.org/abs/2608.31106) DreamX-Creator
- [2511.03334](https://arxiv.org/abs/2511.03334) UniAVGen
- [2509.06155](https://arxiv.org/abs/2509.06155) UniVerse-1
- [2602.19163](https://arxiv.org/abs/2602.19163) JavisDiT++
- [2503.23377](https://arxiv.org/abs/2503.23377) JavisDiT
- [2604.23586](https://arxiv.org/abs/2604.23586) Talker-T2AV
- [2602.08794](https://arxiv.org/abs/2602.08794) MOVA
- [2602.12160](https://arxiv.org/abs/2602.12160) DreamID-Omni
- [2608.13602](https://arxiv.org/abs/2608.13602) Omni-LiveAvatar
- [2512.23576](https://arxiv.org/abs/2512.23576) LiveTalk
- [2602.07449](https://arxiv.org/abs/2602.07449) SoulX-FlashHead
- [2406.07686](https://arxiv.org/abs/2406.07686) AV-DiT
- [2412.15220](https://arxiv.org/abs/2412.15220) SyncFlow
- [2402.17723](https://arxiv.org/abs/2402.17723) Seeing and Hearing
- [2405.17842](https://arxiv.org/abs/2405.17842) MMDisCo
- [2504.14267](https://arxiv.org/abs/2504.14267) TAVDiff (saliency)
- [2607.26818](https://arxiv.org/abs/2607.26818) Ripple
- [2601.01568](https://arxiv.org/abs/2601.01568) MM-Sonate
- [2605.25659](https://arxiv.org/abs/2605.25659) StreamChar
- [2609.04249](https://arxiv.org/abs/2609.04249) Encore
- [2511.21780](https://arxiv.org/abs/2511.21780) 3MDiT
- [2606.09639](https://arxiv.org/abs/2606.09639) CineDance

In repo:

- [raw-inference.md](../perf/raw-inference.md)
- [ltx25.md](../ports/ltx25.md) (offload placement, NVFP4)
- [ltx2.md](../ports/ltx2.md)
- [h3.md](../ports/h3.md)
- [wan.md](../ports/wan.md)
- [mmaudio.md](../ports/mmaudio.md)
- [stable-audio.md](../ports/stable-audio.md)
- [research-avatar-v2v.md](research-avatar-v2v.md)
- [techniques.md](../techniques.md)
- `crates/fastvideo-cudarc/src/wan/{offload,quant}.rs`
- `crates/fastvideo-cudarc/build.rs`
