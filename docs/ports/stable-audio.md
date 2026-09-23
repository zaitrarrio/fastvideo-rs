# Stable Audio Open — port specification

FastVideo family `stable_audio`: `StableAudioDiTModel` text-to-audio.
Hubs `FastVideo/stable-audio-open-1.0-Diffusers` and
`FastVideo/stable-audio-open-small-Diffusers`.

Upstream registers these under a generic T2V workload option; this port exposes
**T2A** (wav out).

Sources (read 2026-09-22): Diffusers `StableAudioPipeline`, FastVideo support
matrix / stable_audio stages.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `stable_audio_open_1_0` | `FastVideo/stable-audio-open-1.0-Diffusers` | full |
| `stable_audio_open_small` | `FastVideo/stable-audio-open-small-Diffusers` | small |

Sample rate **44100**, stereo, latent hop **2048**, latent length **1024**
(≈ 47 s audio domain / `sample_size=2097152`).

---

## DiT (`StableAudioDiTModel`) — Open 1.0 defaults

| piece | value |
|---|---|
| `in/out_channels` | **64** / **64** |
| layers / heads / head_dim | **24** / **24** / **64** |
| KV heads | **12** |
| `cross_attention_dim` | **768** |
| `global_states_input_dim` | **1536** |
| `sample_size` (latent) | **1024** |

Small preset: fewer layers (scaffold uses **12**).

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::stable_audio` | landed |
| Registry + generate (wav scaffold) | landed |
| cudarc DiT (tiny zeros + load hook) | landed |
| Audio VAE / T5 encode | T5-11B or CLIP text when `text_encoder/` present; audio VAE still stub decode |
