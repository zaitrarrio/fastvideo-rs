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
volume (default `fv-weights-h3-ltx-hy`, EUR-IS-1, which holds the tree).
`fv-weights-b200-us` (US-CA-2) held it too until Runpod deleted that volume
(2026-10); the script refuses it now (EU only). `verify-weights.sh mmaudio-44k-v2` is the gate.

## Running it

- `fv-gpucheck mmaudio v2a --weights <root> --video clip.mp4 --prompt ...
  [--runs N]` times N generations on a clip. The first is a warm-up when
  N > 1. It writes `mmaudio.wav`, `mmaudio.f32` and a muxed mp4.
- `fv-gpucheck mmaudio t2a --weights <root> --prompt ... [--seeds 1,2,3]
  [--duration 8]` is text-to-audio with no video. One loaded pipeline
  writes `seed-<seed>.wav` per seed.
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

## Speech

The question is whether MMAudio can say a given line. `runpod-matrix.sh
speechtest` measures it. It generates every clip on one pod and transcribes
each one with Whisper large-v3 (`speech_transcribe.py`: English forced,
temperature 0). WER is computed against the intended line, lowercased, with
punctuation stripped and numbers spelled out. `no_speech_prob` is Whisper's
value for the first segment, and 1.0 when it finds no segment at all.
"p(en)" is the unforced language-ID probability of English. Seeds are 1000,
1001 and 1002 throughout. The run is at `fc6053d` on an RTX PRO 6000
(driver 595.91), on 2026-09-27. The wavs, `whisper/speech.json` and
`transcripts.tsv` are in `artifacts/runpod/speechtest/fc6053d-09271842/`.

Cases:

- **T2A** runs 8 s from a text prompt only, in three styles. "fox" is
  `A man says clearly: "The quick brown fox jumps over the lazy dog."`.
  "station" is `A woman announces: "Welcome to the station, the next train
  leaves at nine."`. "fox, tagged" is strobe's `<S>..<E>` form with an
  `Audio: male speech, clear voice, quiet room` line. The upstream README has
  no speech prompt style. Its "Known limitations" section says instead that
  "the model sometimes generates unintelligible human speech-like sounds".
- **V2A** runs 5 s on one FastWan 1.3B clip of a man speaking to camera
  (seed 1024). It runs once without a text prompt and once with the "fox"
  line as the text prompt.
- **H3 baseline** is FastH3 4-step VSA at 480p, 5 s, with native joint audio.
  The same lines are written in H3 dialogue markup,
  `... He says clearly: <d>[English] The quick brown fox jumps over the lazy dog.</d>`.

| Case | Seed | Whisper transcript | WER | no_speech_prob | p(en) |
|---|---|---|---|---|---|
| T2A fox | 1000 | (none) | 1.00 | 1.00 | 0.13 |
| T2A fox | 1001 | (none) | 1.00 | 1.00 | 0.17 |
| T2A fox | 1002 | (none) | 1.00 | 1.00 | 0.24 |
| T2A station | 1000 | "And still, for the naive or young, this lift has 10-inch doors with a deep-fitting smile. 12 companions or a family tree." | 2.30 | 0.39 | 0.19 |
| T2A station | 1001 | "The" | 0.90 | 0.57 | 0.32 |
| T2A station | 1002 | "Thank you for watching. Please subscribe to our channel. Thank you." | 1.10 | 0.54 | 0.14 |
| T2A fox, tagged | 1000 | "Lizzie Risset and Little Miss Lask." | 1.00 | 0.40 | 0.68 |
| T2A fox, tagged | 1001 | "I don't like when I have to use my career as a sub-doub. And pointing down in the live, uh... What's up, pal? ..." | 3.11 | 0.22 | 0.84 |
| T2A fox, tagged | 1002 | "Great. Hey, Matt. This is called true. Yay." | 1.00 | 0.21 | 0.80 |
| V2A, no text | 1000 | "I'm breaking free g on the same this other effort that says we have gotten so cage at them some" | 2.11 | 0.33 | 0.11 |
| V2A, no text | 1001 | "I'm William Bedeus, getting into Sydney PSD news. We're in Dark Isle League, Tariff Adventure." | 1.67 | 0.25 | 0.98 |
| V2A, no text | 1002 | "I will be in prison on Sunday because I need to be in that condition." | 1.67 | 0.36 | 0.38 |
| V2A + fox line | 1000 | "I'm playing for a G on the side because they said they have it. ..." | 2.89 | 0.26 | 0.12 |
| V2A + fox line | 1001 | "I'm William Verdeers, getting in a sitting there. This day is where doctors will leave time for a time." | 2.11 | 0.27 | 0.91 |
| V2A + fox line | 1002 | "online and previous thank you so much beyond sky you just beyond that can you get yourself now" | 2.00 | 0.35 | 0.65 |
| H3 fox | 1000 | "The quick brown fox jumps over the lazy dog." | 0.00 | 0.03 | 1.00 |
| H3 fox | 1001 | "The quick brown fox jumps over the lazy dog." | 0.00 | 0.02 | 0.99 |
| H3 fox | 1002 | "The quick brown fox jumps over the lazy dog." | 0.00 | 0.07 | 0.98 |
| H3 station | 1000 | "Welcome to the station. The next train leaves at nine." | 0.00 | 0.05 | 0.98 |
| H3 station | 1001 | "Welcome to the station. The next train leaves at nine." | 0.00 | 0.06 | 0.96 |
| H3 station | 1002 | "Welcome to the station. The next train leaves at nine." | 0.00 | 0.25 | 0.93 |

Mean WER is 1.00 for T2A fox, 1.43 for T2A station, 1.70 for T2A fox
tagged, 1.81 for V2A without text, 2.33 for V2A with the line, and 0.00
(6/6 exact) for H3.

What the numbers show:

- **MMAudio never produced the requested words**: 0 of 15 clips, with no
  intended word recovered beyond chance. With the plain quoted line in T2A,
  "fox" gets no segment at all (no_speech_prob 1.0). "station" gets noise
  that Whisper decodes into its known hallucinations ("Thank you for
  watching..."). The clips are not silent (RMS 0.03 to 0.15).
- **Babble looks like speech but carries no words.** The "Audio: male
  speech" tag and the talking-face video both give voice-like audio.
  no_speech_prob drops to 0.2 to 0.4, p(en) reaches 0.8 to 0.98 on some
  seeds, and Whisper confidently transcribes invented sentences. This is the
  upstream README's "unintelligible human speech-like sounds".
- **The text line does not steer the words.** V2A with the fox line does no
  better than V2A without it (WER 2.33 vs 1.81).
- **H3's joint audio says the line verbatim** on all 6 clips, with
  no_speech_prob of 0.02 to 0.25. That also confirms the Whisper setup reads
  real speech correctly.

The limit is in the model, not the port. MMAudio is trained on
sound-effect captions (VGGSound and similar), with no text-to-phoneme path.
Our port matches upstream spectrally (see Parity), so upstream would behave
the same way. For dialogue, use a joint audio model (H3, LTX-2) or a TTS
stage. The MMAudio soundtrack is for ambience and effects only.
