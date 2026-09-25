# Decoder gap

Date: 2026-09-24
Compares: official video VAEs vs tiny decoders (TAEHV / TAEH3) vs published end-to-end numbers.
Sources: decision-log Sep 19 H3/LTX port; TAEH3 A/B 2026-09-19; TAEHV A/B 2026-09-18; 5090 streamed TAEHV 2026-09-19; LTX-2.5 two-stage 2026-09-21; DiffVAE 2026-09-22 (`FVID-2026-09-22-ltx25-diffvae`).

Official video VAEs are the leftover wall time after DiT. Published FastH3 /
FastWan-QAD E2E numbers already assume a tiny or highly optimised decode.
Audio VAEs are not the gap (~0.4 s).

## Headline

| | |
|---|---|
| H3 official VAE (124 f, 1344x768) | 23-29 s |
| TAEH3, same clip | 0.98 s |
| Published FastH3 full E2E on B200 | 16.2 s |
| LTX DiffVAE (host NATTEN) | 884 s |

**Official H3 decode is larger than the blog E2E.** FastH3 Preview v1 is
16.2 s warm wall on 1x B200 for the whole clip. Our official H3 video VAE
alone was 22.9 s (PRO 6000) to 28.97 s (Max-Q) for the same 5 s 1344x768
canvas. We could not hit the published number on the official decoder, on any
card we had used, unless decode moved to TAEH3 (0.98 s) or an equivalent.

## Official VAE vs tiny decoder (same latents)

| Decoder | Video decode (s) |
|---|---:|
| H3 official (PRO 6000) | 22.9 |
| H3 official (Max-Q) | 28.97 |
| H3 TAEH3 (Max-Q) | 0.98 |
| Wan official (3090, 8 s clip) | 16.18 |
| Wan TAEHV (3090) | 1.66 |
| Wan TAEHV (5090, streamed) | 0.64 |
| LTX-2 conv (PRO 6000) | 1.59 |
| LTX-2.5 conv two-stage | 3.9 |
| LTX DiffVAE, host NATTEN | 884 |

DiffVAE is ~230x the conv path.

## Per decoder

| Decoder | What we run | Published / reference | Gap |
|---|---|---|---|
| H3 official video VAE | 22.9 s PRO 6000 / 28.97 s Max-Q. 19% of a warm 8-step clip. Gate never reached decode (300 s wall). | Inside FastH3 16.2 s E2E on B200. No separate decode figure. | Time: official decode > entire published clip. Close it with TAEH3 (~30x, 0.98 s), not more cuDNN on the ViT. |
| H3 TAEH3 | Port exists, opt-in. 0.98 s vs 28.97 s. Denoise unchanged (~125 s). | Same family as Wan TAEHV; FastH3 blog does not name TAEH3. | Implementation gap is closed. Product gap: matrix / gate still used the official VAE. |
| H3 / LTX audio VAE + vocoder | Audio decode 0.39-0.43 s. BWE MelSTFT still on host. Vocoder 40.9 dB SNR vs 50 dB gate. | No published audio-decode latency. | Not the E2E bottleneck. Quality gate miss is the vocoder, not the VAE. |
| Wan official VAE | 16.18 s on 3090 (129 f). Gate used this path: 1.12 s for 17 f (scales to ~8-9 s at 129 f). | QAD 3.4 s E2E on 4090 assumes TAEHV, not this decoder. Pre-TAEHV H100: 3.9 s of a 23.7 s clip. | Using the official VAE on the gate made decode ~ denoise (1.12 vs 1.20 s) on a 1 s clip. |
| Wan TAEHV | 1.66 s vs 16.18 s (9.7x), 34.16 dB vs official, -44% VRAM. Our port 0.44 s vs PyTorch TAEHV 1.33 s (3x) on a 3060. 5090 streamed: 0.50-0.64 s. | QAD's 3.4 s stack includes TAEHV. | Decoder time is not why we trail QAD. Remaining QAD gap is FP8 attention + compile, not VAE. |
| LTX conv VAE | 2.0: 1.59 s. 2.5 two-stage: ~3.9 s. Exact chunk vs diffusers blend (91 dB PSNR). 2.5 OOM if ~38 GiB DiT stays resident. | Lightricks publishes no decode time. | Speed is fine vs our DiT. Reliability gap: drop-DiT before decode (landed, not proven this gate). |
| LTX DiffVAE | 884 s decode. Neighborhood attention still on host. Drops DiT (70 GiB peak). | Official 2.5 pack's peer-quality path. | The large decoder hole. Need device NATTEN, not more conv tuning. |

## Quality vs official

- Wan TAEHV 34.2 dB vs official VAE.
- LTX conv 91 dB (exact chunk vs diffusers blend).
- Vocoder 40.9 dB vs the 50 dB gate.

TAEHV sits closer to the official Wan VAE than our fast path sits to exact
(27.8 dB). H3 audio/video official decoders matched float32 oracles
(1.3e-5 / 1.5e-6). Hunyuan VAE was still unvalidated on GPU.

## What would close E2E

- H3: default TAEH3 on the gate (saves ~22-28 s).
- Wan: TAEHV (already default on `validate.sh gen`, not on the Phase 3 matrix clip).
- LTX-2.5: prove conv decode after drop-DiT; do not ship DiffVAE until NATTEN is on device.
- None of that replaces a B200 gen if the target is the 16.2 s blog number.

## What happened after

The H200 / B200 warm suites (2026-09-24 / 25) ran with TAEH3: decode 2.76 s
inside a 26.5 s B200 generate. DiffVAE and device NATTEN remain open.
