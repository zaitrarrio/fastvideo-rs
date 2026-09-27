# MMAudio

FastVideo family `mmaudio`: MMAudio large-44k-v2 (hkchengrex/MMAudio, commit
`974010a`, the one strobe pins), video-to-audio and text-to-audio at 44.1 kHz.
It also runs as an opt-in soundtrack stage after the video-only Wan family.

## Pipeline

`eval_utils.generate` from upstream, on the device:

| Piece | Weights (`safetensors/`) | Notes |
|---|---|---|
| CLIP DFN5B ViT-H/14-384 | `clip_dfn5b_h14_384` | text (77 tokens, open_clip BPE) and 8 fps visual tokens at 384x384 |
| Synchformer (visual half) | `synchformer` | 25 fps, 224x224, clips of 16 at stride 8, 8 tokens per clip, 768-d |
| DiT | `mmaudio_large_44k_v2` | 7 joint + 14 fused blocks, hidden 896, 14 heads of 64, latent 40 |
| VAE decoder | `vae_44k` | latent to 128-band mel |
| BigVGAN v2 | `bigvgan_v2_44k` | mel to waveform, 44.1 kHz |

The sampler is Euler flow matching, 25 steps, CFG 4.5, negative text `""`.
Upstream runs the network as `net.to(bfloat16)`, and the port rounds to bf16
wherever torch does. That covers the noise, every intermediate latent, the
timestep, each guided flow, and the buffers `.to()` rounds (`t_embed.freqs`,
latent mean/std, empty features, `sync_pos_emb`). The step itself is a few
thousand values and runs on the host.

Duration is the clip's `frames / fps`. That matches strobe's sidecar, which
passes `clip_s`. The clip is read back from the mp4 with ffmpeg, as upstream
reads it with PyAV, so both sides condition on the encoded frames.

## Weights

`scripts/gpu/fetch-mmaudio.sh` (CPU pod, REST API only) runs
`scripts/gpu/fetch-mmaudio.py` onto a weight volume under
`weights/mmaudio-44k-v2/`:

- The upstream layout: `weights/`, `ext_weights/`,
  `bigvgan_v2_44khz_128band_512x/` and `DFN5B-CLIP-ViT-H-14-384/`. The md5s
  are the ones pinned in `download_utils.py`.
- `safetensors/`: one file per checkpoint, with the same keys and dtype
  (f32), checked by a round-trip equality check.
- `MANIFEST.json` and a `.complete` marker, written last. The total is
  21.46 GB.

The fetch is add-only. It stops if the tree already has files, and
`FETCH_ADD_ONLY=0` re-fetches in place. `RUNPOD_VOLUME_NAME` selects the
volume. Both `fv-weights-b200-us` (US-CA-2) and `fv-weights-h3-ltx-hy`
(EUR-IS-1) hold the tree. `verify-weights.sh mmaudio-44k-v2` is the gate.

## Running it

- `fv-gpucheck mmaudio v2a --weights <root> --video clip.mp4 --prompt ...
  [--runs N]` times N generations on a clip. The first is a warm-up when
  N > 1. It writes `mmaudio.wav`, `mmaudio.f32` and a muxed mp4.
- Wan sidecar: `--audio mmaudio` (CLI and `fv-gpucheck wan gen`), or
  `FASTVIDEO_WAN_AUDIO=mmaudio`. After the video it muxes an AAC soundtrack
  into the mp4. `benchmark.json` gains the video / audio / end-to-end split
  and the audio real-time factor.

| Env | Effect |
|---|---|
| `FASTVIDEO_MMAUDIO_WEIGHTS`, `MMAUDIO_MODEL_PATH` | weight root (default `$FV_WEIGHTS/mmaudio-44k-v2`, then `/workspace/weights/mmaudio-44k-v2`) |
| `FASTVIDEO_WAN_AUDIO_PROMPT` | audio text condition (default: the video prompt) |
| `FASTVIDEO_WAN_AUDIO_STEPS` / `_CFG` / `_SEED` | 25 / 4.5 / the video seed |

## Oracle

`scripts/gpu/mmaudio_oracle.py` runs upstream MMAudio with strobe's sidecar
settings. It does timed runs plus a dump in the `docs/oracle.md` format
(`mm_*`), with the sampler written out step by step. It also dumps an f32
VAE decode and vocode of the same latent from freshly loaded f32 weights
(`mm_mel_f32`, `mm_wave_f32`). On our side, `FASTVIDEO_DUMP_DIR` dumps the
same names. `FASTVIDEO_INJECT_DIR` reads the reference noise (`mm_x0`), and
`FASTVIDEO_MMAUDIO_INJECT` selects what else is injected:

- `pixels`: the reference's preprocessed frames.
- `features`: the reference's encoder outputs.
- `x1`: the reference's final latent.
- `mel`: the reference's mel.

`scripts/gpu/mmaudio_wave_compare.py` compares two waveforms. It reports
sample rel-L2 and cosine, and phase-tolerant spectral measures: log-mel L1
in dB, spectral convergence, log-mel correlation and mel band-energy cosine.

`runpod-matrix.sh mmaudio` runs the whole thing on one GPU pod:

- the FastWan 1.3B and SF-Wan 1.3B cells with the sidecar;
- upstream (timed) on the FastWan clip;
- our port (timed) on the same clip;
- the three inject arms with `compare-dumps`;
- the waveform comparisons.

`mmaudio_tiny_reference.py` writes the host fixtures for
`reference_tests.rs`.

## Results

These come from `runpod-matrix.sh mmaudio` at `778c65e`, on an RTX PRO 6000
Blackwell with the EU volume, on 2026-09-27. The clip is FastWan 1.3B,
480x832, 81 frames at 16 fps. Upstream truncates the audio to 5.00 s,
because Synchformer needs 125 frames at 25 fps, and the port matches that.
Results are in `artifacts/runpod/mmaudio/778c65e-09271737/`.

### Timing (warm, 5.0 s of audio)

| | generate | peak |
|---|---|---|
| Upstream (PyTorch bf16, `load_video` excluded) | 0.94 s | 5.8 GiB allocated |
| Ours (`mmaudio v2a`, fast mode) | 1.91 s (RTF 0.38) | |

Ours broken down: preprocessing 0.07 s, CLIP visual 0.60 s, Synchformer
0.17 s, text 0.10 s, DiT 0.84 s (25 steps with CFG), VAE plus BigVGAN 0.08 s.
The gap to upstream is mostly the CLIP visual pass and the DiT. Closing it
is open work.

As a Wan sidecar, audio adds about 2.1 s after the video:

| Cell | Video | Audio stage | End to end | Peak |
|---|---|---|---|---|
| FastWan 1.3B DMD | 2.29 s | 2.08 s | 4.37 s (0.86x real time) | 29.6 GiB |
| SF-Wan 1.3B | 4.45 s | 2.03 s | 6.48 s (1.28x real time) | 30.8 GiB |

### Parity against upstream

All three arms share the reference noise (`mm_x0`), and rel-L2 is measured
against upstream bf16:

| Tensor | pixels injected | features injected | end to end |
|---|---|---|---|
| CLIP / Synchformer pixels | 0.47% / 0.06% | same | same |
| CLIP visual / sync / text features | 1.3% / 1.2% / 1.3% | (injected) | 2.2% / 1.2% / 1.3% |
| DiT blocks, step 0 | 0.4 to 1.4% | 0.4 to 1.2% | 0.4 to 1.5% |
| Flow, step 0 (cond / uncond) | 0.91% / 0.84% | 0.82% / 0.84% | 0.91% / 0.84% |
| Final latent `x1` (cosine) | 8.3% (0.9966) | 4.6% (0.9990) | 8.6% (0.9963) |

The per-step drift is bf16-sized, and it compounds over 25 Euler steps.

Waveforms (`mmaudio_wave_compare.py`):

| Pair | log-mel L1 | log-mel corr | band cos | sample rel-L2 |
|---|---|---|---|---|
| Upstream bf16 vs upstream f32 decode, same latent (the noise floor) | 2.25 dB | 0.959 | 1.000 | 0.113 |
| Our VAE + BigVGAN on the reference latent vs upstream f32 | 0.02 dB | 1.000 | 1.000 | 0.0009 |
| Ours end to end vs upstream f32 decode | 1.18 dB | 0.995 | 0.978 | 0.557 |
| Ours end to end vs upstream bf16 | 2.86 dB | 0.952 | 0.977 | 0.561 |
| Ours with our own noise vs upstream with torch noise | 15.2 dB | 0.581 | 0.981 | 1.77 |

In the f32 path, the decoder and vocoder match upstream exactly. End to end,
ours is spectrally closer to upstream than upstream's own bf16 decode is to
its f32 decode. Phase drifts, so sample-domain numbers are not a parity
measure. The RMS is about 10% lower (0.184 vs 0.204), which is within the
latent drift. Our sampler draws noise from its own RNG, so with the same seed
it makes a different (but similar-sounding) track to torch unless `mm_x0` is
injected.
