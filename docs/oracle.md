# GPU oracle diff

Numerical parity of our Rust pipelines against the Python references, on the
same GPU type (RTX PRO 6000, sm_120), tensor by tensor.

## What it compares

Torch's Philox RNG is not reproduced, so the reference's random draws are
*injected* into our run; after that, every remaining difference is the
denoiser's own.

| side | switch | what |
|---|---|---|
| Python | `FV_ORACLE_DUMP_DIR` + `PYTHONPATH=scripts/gpu/upstream/oracle_site` | `oracle_dump.py` hooks the reference (every process, via `sitecustomize`) and writes our dump format |
| Rust | `FASTVIDEO_INJECT_DIR=<python dump>` | reads the reference's noise (and text conditioning) back (`wan/inject.rs`) |
| Rust | `FASTVIDEO_DUMP_DIR=<dir>` | our dump (`wan/dump.rs`) |
| both | `FASTVIDEO_DUMP_OPS=0,1,24,47` | blocks whose inside is dumped at the first step (H3: `adaln`, `attn_in`, `attn_out`, `resid_msa`, `ffn_in`, `ffn_out`) |
| host | `fv-gpucheck compare-dumps --baseline <python> --candidate <ours>` | per-tensor rel-L2 / cosine / max-abs |

Format: `<name>.f32` (raw little-endian f32; bf16 widened exactly) and
`<name>.shape`. Block outputs keep every 64th row (LTX audio blocks: every row).

Names:

* H3 (FastVideo FastH3 8-step V2, FastH3 4-step VSA LoRA): `text_hidden`
  (ours before injection: the text-encoder parity), `text_refined`,
  `{video,audio}_{sigmas,timesteps}`, `{video,audio}_step00_in` (injected),
  `step00_packed_in`, `rope_{cos,sin}`, `step00_block_<i>`, `step00_b<i>_<op>`,
  `{video,audio}_vel_stepNN`, `{video,audio}_stepNN`.
* LTX-2.5 distilled two-stage (sol-engine `gpu_infer.py` on `ltx_pipelines`):
  `text_{video,audio}_ctx` (ours before injection), the text stages
  `text_input_ids`, `text_attention_mask`, `text_hidden_<k>`,
  `text_{video,audio}_feats`, `text_{video,audio}_ctx_real` (the first prompt
  encoded; a text-cache hit writes none), `noise_seed<seed>_<k>`
  (every `torch.randn` on a seeded generator, replayed by our `NoiseStream`),
  per stage `s1_`/`s2_`: `sigmas`, `{video,audio}_step00_in`,
  `step00_{video,audio}_in`, `step00_{video,audio}_block_<i>`,
  `{video,audio}_{vel,x0}_stepNN`, `{video,audio}_stepNN`; `s2_upsampled` and
  `s2_entry_{video,audio}` (our stage-2 entry before the reference's is
  injected; `FASTVIDEO_INJECT_STAGE2=0` keeps ours).
* H3 Ref2VA (FastVideo `MiniMaxH3Ref2VAModularPipeline` on `transformer_ref`,
  target `h3-ref2va-<N>step`; ours `h3 gen --h3-recipe base-<N>step --ref <image>
  --ref-root $W/h3-ref2va`): the H3 names above. `video_step00_in` is the
  reference's `[condition | target]` rows; only the target part is injected
  (`FASTVIDEO_INJECT_COND=1` also takes the condition rows), so
  `step00_packed_in` compares our reference-image encode. Results:
  docs/ports/h3-ref2v.md §8.
* SF-Wan 1.3B (FastVideo `WanCausalDMDPipeline`, target `sfwan13`):
  `sf_latents_in`, `text_hidden`, `sf_noise_<k>` (injected), per causal block
  `c` and step `i` `sf_c<c>_s<i>_{flow,x0}`, `sf_c<c>_out`, `sf_latents_out`,
  and for block 0 step 0, block 1 step 0 and block 0's context pass
  (`sf_c0_ctx_`) the block outputs `..._block_<l>` plus, for
  `FASTVIDEO_DUMP_OPS` layers, the K window `..._b<l>_kwin` and the attention
  output `..._b<l>_attn_x`. Results: docs/ports/wan.md "SF-Wan".

The FastVideo side runs its strict eager route (`--profile strict
--no-inference-torch-compile`): hooks inside fullgraph-compiled blocks would
break the graph, and the `all` profile's fusions are FastVideo's own
report-only reorderings.

## Running it

```
scripts/gpu/oracle.sh [sha]            # ORACLE_TARGETS="fasth3-8step fasth3-4step-vsa ltx25-512p-dense ltx25-512p"
```

(`scripts/gpu/oracle.sh` runs from a private copy, so the repo copy can be
edited during a run.) It starts one upstream pod per reference image (`runpod-http.sh upstream`, pod
step `oracle:<targets>`, kept alive to serve its dumps), and one runtime pod
(`FV_FAMILY=oracle`) that downloads each `oracle-<target>/oracle-dump.tar` as
soon as it is ready, runs ours injected, and compares. Targets:
`fasth3-8step`, `fasth3-4step-vsa`, the controls `fasth3-8step-vsa0` (VSA
sparsity 0 on both sides: every tile, gated compression kept, no top-k
selection; `FASTVIDEO_VSA_SPARSITY` on ours) and `fasth3-4step-dense` (the
dense-datafree LoRA, FLASH_ATTN, no VSA anywhere), and `ltx25-512p[-dense]`,
`ltx25-4k[-dense]`. For H3 targets the runtime pod also runs ours with f32
activations (`FV_ORACLE_F32`, default on for H3) and reports
`oracle-<t>-bf16-vs-f32`: how far bf16 rounding alone moves our pipeline,
the floor the Rust-vs-Python profile is judged against.
Results: `artifacts/runpod/oracle/<tag>/oracle-<target>-diff/` (the stderr
table and `gpucheck-out/*.json`); the dumps themselves stay on the pods.

Pass: step-1 per-block rel-L2 at bf16-rounding level (~2e-3) growing smoothly
with depth, no block where it jumps. A jump is bisected with the block's op
dumps (`FASTVIDEO_DUMP_OPS=<i>`): the first op whose rel-L2 leaves the bf16
floor is the divergent one.

## Results: H3 (2026-09-26, RTX PRO 6000, runtime f49db62 / 5312e4b / b1226e9)

Reference: FastVideo e90be59, strict eager route, 768x1344x124, the matrix
prompt, seed 1024. rel-L2 of ours against the reference.

Inputs: noise injected exactly (0), video/audio sigmas and timesteps equal
(0). Our Qwen hidden states vs FastVideo's: 1.29e-2 (then the reference's are
injected). Refined text: 8.2e-3. Packed DiT input: 3.2e-3. RoPE tables:
1.0e-3 / 1.4e-3 (the reference rounds cos/sin to bf16; ours are f32). AdaLN
rows (dense run, after the dump fix): 3e-5 to 1e-4.

Step-1 block outputs:

| block | 8-step VSA 0.8 | 4-step VSA 0.9 | 4-step dense |
|---|---|---|---|
| 0 | 4.8e-3 | 8.2e-4 | 4.1e-3 |
| 1 | 5.9e-3 | 3.9e-3 | 3.8e-3 |
| 12 | 5.3e-3 | 5.9e-3 | 4.4e-3 |
| 17 | 4.2e-3 | 7.1e-3 | 6.1e-3 |
| 18 | 5.8e-3 | 1.35e-2 | 6.5e-3 |
| 24 | 9.0e-3 | 1.53e-2 | 8.3e-3 |
| 30 | 3.9e-2 | 4.7e-2 | 3.1e-2 |
| 36 | 9.0e-2 | 1.23e-1 | 8.9e-2 |
| 42 | 1.73e-1 | 2.24e-1 | 1.52e-1 |
| 48 | 3.28e-1 | 4.54e-1 | 1.43e-1 |
| 49 | 2.54e-1 | 3.56e-1 | 9.3e-2 |

Inside blocks 0, 1, 24, 47 (attention in/out, residual, FFN in/out) no op
leaves its input's error level: each propagates what it is given.

Video latents after each step:

| step | 8-step VSA | 4-step VSA | 4-step dense |
|---|---|---|---|
| 1 | 3.4e-3 | 7.2e-3 | 2.2e-3 |
| 2 | 1.04e-2 | 2.49e-2 | 2.33e-2 |
| 3 | 1.99e-2 | 8.46e-2 | 8.13e-2 |
| 4 | 3.55e-2 | 4.35e-1 | 4.35e-1 |
| 5 | 6.37e-2 | | |
| 6 | 1.22e-1 | | |
| 7 | 2.67e-1 | | |
| 8 | 5.26e-1 | | |

First reading: from about block 25 the difference grows ~15% per block to
0.15 (dense) to 0.45 (VSA), and the final latents differ by 0.4-0.5. The
dense control grows the same way, so VSA's tile selection is not the cause,
and no single op diverges.

### The controls (runtime b1226e9, upstream 78729d6)

The same injected inputs, three runs of the 8-step checkpoint: ours bf16
(the default), ours with f32 activations (`FASTVIDEO_BF16_ACT=0`), and the
reference (bf16 throughout). Pairwise rel-L2, VSA 0.8:

| | ours-bf16 vs ref | ours-f32 vs ref | **ours-bf16 vs ours-f32** |
|---|---|---|---|
| block 0 | 4.8e-3 | 5.2e-3 | 1.0e-3 |
| block 12 | 5.3e-3 | 5.1e-3 | 5.0e-3 |
| block 24 | 9.0e-3 | 1.75e-2 | 1.30e-2 |
| block 30 | 3.9e-2 | 4.1e-2 | 2.5e-2 |
| block 36 | 9.0e-2 | 9.3e-2 | 8.7e-2 |
| block 42 | 1.73e-1 | 1.70e-1 | 1.58e-1 |
| block 48 | 3.28e-1 | 3.02e-1 | 2.68e-1 |
| velocity, step 1 | 1.73e-1 | 1.58e-1 | 1.34e-1 |
| latents, step 4 | 3.5e-2 | 4.0e-2 | 3.4e-2 |
| latents, step 8 | 5.26e-1 | 5.78e-1 | 4.83e-1 |

At VSA sparsity 0 on both sides (every tile, gated compression kept):

| | ours-bf16 vs ref | ours-f32 vs ref | ours-bf16 vs ours-f32 |
|---|---|---|---|
| block 24 | 8.6e-3 | 1.39e-2 | 1.22e-2 |
| block 36 | 7.1e-2 | 7.2e-2 | 6.2e-2 |
| block 48 | 2.80e-1 | 2.68e-1 | 2.46e-1 |
| latents, step 8 | 3.74e-1 | 3.45e-1 | 3.60e-1 |

**Root cause: bf16 rounding amplified by depth, not a systematic
difference.** Our own pipeline, run in bf16 and in f32 from bit-identical
inputs, parts by as much as ours parts from the reference, block for block
(block 48: 0.27 vs 0.33; final latents 0.48 vs 0.53), and ours in f32 is no
closer to the reference than ours in bf16. The three runs are roughly
equidistant: three rounding paths of one chaotic function, with no
systematic offset for a bisect to find. Sparsity 0 lowers all three pairs
alike (final latents 0.35-0.37): part of the VSA level is top-k tile flips,
which rounding noise triggers, and it is shared. Late H3 blocks amplify any
perturbation ~15% per block (residual outliers reach 4e4 by block 34, and
every block renormalizes them), so the ~2e-3 target cannot be met by any
two implementations of this model that round differently. No fix on our side.

Verdict: **H3 matches the reference to the model's own bf16/f32 noise floor**
(the strict criterion of "~2e-3 flat" does not hold even for our pipeline
against itself).

## LTX-2.5 distilled two-stage, 512p (sol-engine gpu_infer.py, bf16 pipeline)

Reference: Lightricks/LTX-2 fd4ded7 driven by sol-engine's RTX5090
`gpu_infer.py`, 768x512x121, packs rebuilt from the Diffusers copy
(`RECON_ACCEPT_MISMATCH=1`: the two mismatched packs keep the Diffusers
bytes our port loads). Injected: all 18 seeded `torch.randn` draws (initial
noise, 14 ancestral draws, stage-2 renoise), both text contexts, and the
stage-2 entry state.

| | dense stage 2 | Sol stage 2 |
|---|---|---|
| s1 block 0 / 12 / 24 (video) | 3.3e-3 / 4.2e-3 / 5.4e-3 | same run |
| s1 block 36 / 42 / 44 / 47 | 1.7e-2 / 4.5e-2 / 8.7e-2 / 2.3e-2 | same |
| s1 audio block 0 / 24 / 47 | 3.6e-3 / 6.1e-3 / 3.4e-2 | same |
| s1 velocity step 1 | 4.1e-2 | same |
| s1 latents steps 1-8 | 9e-4, 1.4e-3, 1.8e-3, 2.2e-3, 7.1e-3, 7.5e-2, 0.25, 0.34 | same |
| s2 block 0 / 24 / 36 / 47 (video) | 2.9e-3 / 5.0e-3 / 1.7e-2 / 1.5e-2 | 2.9e-3 / 8.7e-3 / 2.9e-2 / 2.3e-2 |
| s2 latents steps 1-3 | 1.3e-2, 4.8e-2, 8.0e-2 | 1.8e-2, 7.4e-2, 0.125 |

Stage 1 is identical in both arms (the arms differ only in stage 2). The
stage-1 "jump" at step 6 follows the schedule, not the model: steps 1-4 move
sigma by 0.006 each (1.0 to 0.975), so the state barely changes whatever the
velocity, and step 6 (0.909 to 0.725) is the first large step. The
per-forward difference is the step-1 velocity, 4e-2, against 1.3-1.7e-1 for
H3. Block profiles grow smoothly with no jump. Sol stage 2 runs upstream
(141 Sol kernel calls; `tvm_ffi` is present in the sol-ltx25 image) and adds
the expected top-k-selection spread on top of dense.

The f32 noise floor (dense stage 2, `FV_ORACLE_F32=1`):

| | ours-bf16 vs ref | ours-f32 vs ref | **ours-bf16 vs ours-f32** |
|---|---|---|---|
| s1 block 0 / 12 / 24 | 3.3e-3 / 4.2e-3 / 5.4e-3 | 4.0e-3 / 7.3e-3 / 1.1e-2 | 4.4e-3 / 7.1e-3 / 1.0e-2 |
| s1 block 36 / 44 / 47 | 1.7e-2 / 8.7e-2 / 2.3e-2 | 4.5e-2 / 1.22e-1 / 3.4e-2 | 4.2e-2 / 1.23e-1 / 3.2e-2 |
| s1 velocity step 1 | 4.1e-2 | 5.2e-2 | 4.7e-2 |
| s1 latents step 5 / 6 / 8 | 7.1e-3 / 7.5e-2 / 0.34 | 7.6e-3 / 8.5e-2 / 0.37 | 8.8e-3 / 0.10 / 0.44 |
| s2 block 24 / 36 / 47 | 5.0e-3 / 1.7e-2 / 1.5e-2 | 1.2e-2 / 3.5e-2 / 2.5e-2 | 1.2e-2 / 3.4e-2 / 2.4e-2 |
| s2 latents step 1 / 3 | 1.3e-2 / 8.0e-2 | 1.8e-2 / 0.106 | 1.9e-2 / 0.117 |

Verdict: **LTX-2.5 passes.** Our bf16 run is closer to the reference than to
our own f32 run at every block and step (typically by 2x in the blocks):
it reproduces the reference's bf16 rounding points, and what is left is
below the model's bf16/f32 noise floor. No fix needed.

Found, outside the denoiser: **our text contexts differed from the
reference's by 0.34 (video) and 0.27 (audio) rel-L2**, with the same weights
on both sides (the rebuilt packs keep the Diffusers bytes). That run injected
them away. The text path is fixed now (next section).

### LTX-2.5 text path (runtime 6925e27, upstream 78729d6)

Cause: our Gemma-4-12B was Gemma 3's layer with Gemma 4's shapes. What the
reference runs is transformers' `Gemma4UnifiedText*`
(`models/gemma4_unified/modeling_gemma4_unified.py`, called from
`ltx_core/text_encoders/gemma/encoders/base_encoder.py:59-71`), and it differs
from Gemma 3 in six ways. The reference's own buffers confirm each one
(`oracle_meta.json` `gemma`):

| | Gemma 3 (what we ran) | Gemma 4 (reference) |
|---|---|---|
| RMSNorm | `x·(1+w)` | `x·w` (layer-0 norm weights average 19 / 0.86 / 7.8 / 1.7) |
| softmax scale | 256^-0.5 | 1 (`self.scaling = 1.0`) |
| V | raw | RMS-normed per head, no weight (`v_norm`) |
| layer output | as is | `*= layer_scalar` (0.60 / 0.05 alternating) |
| V projection | own `v_proj` | `k_proj` on full-attention layers only; sliding layers keep `v_proj` |
| full-attention RoPE | θ=1e6, positions ÷ 8, rotating the first 128 channels | `proportional`: no factor, a 512-wide table with 64 of 256 angles non-zero, so `rotate_half` pairs `(k, k+256)` |

Also: a `<bos>` is prepended when the tokenizer's post-processor adds none
(`tokenizer.py:44-46`). Our ids were already identical here.
The rest was already right: left padding to 1024, positions `0..1023` across
the padding, all 49 states, per-token RMS, the `sqrt(out/3840)` rescale, the
biased aggregate embeds (`feature_extractor.py:111-192`), the right-pad sort,
the registers, and the 8-layer gated connectors (`embeddings_processor.py`,
`embeddings_connector.py`).

After the fix, rel-L2 of ours against the reference, by stage (the
`text_*` dumps):

| stage | rel-L2 |
|---|---|
| input ids, attention mask | 0 |
| hidden 0 (embeddings) / 1 / 2 | 1.7e-3 / 5.5e-3 / 5.9e-3 |
| hidden 5 / 6 (first full layer) / 7 | 8.9e-3 / 1.0e-2 / 1.1e-2 |
| hidden 12 / 24 / 36 / 47 / 48 | 1.2e-2 / 1.1e-2 / 9.4e-3 / 4.0e-3 / 3.5e-3 |
| aggregate embeds (video / audio) | 1.5e-2 / 1.0e-2 |
| **contexts (video / audio)** | **5.9e-3 / 6.0e-3** (were 0.34 / 0.27) |

End to end, with the noise injected and our own contexts
(`FV_ORACLE_OWN_TEXT=1`, `FASTVIDEO_INJECT_TEXT=0`), dense stage 2:

| | text injected | our text | ours: own text vs injected |
|---|---|---|---|
| s1 velocity step 1 | 4.1e-2 | 5.0e-2 | 4.5e-2 |
| s1 latents step 1 / 8 | 8.9e-4 / 0.34 | 1.0e-3 / 0.36 | 9.4e-4 / 0.32 |
| s2 latents step 3 | 8.0e-2 | 7.5e-2 | 7.6e-2 |

The own-text arm lands where the text-injected one does, within the
pipeline's bf16/f32 floor (s1 step 8: 0.44; s2 step 3: 0.117). The
`a_gemma4_model_matches_the_transformers_formulas` host test (llm.rs) pins the
layer. Reverting any one of proportional RoPE, `v_norm` or `layer_scalar` on
its own fails it.

Not compared: 4K (not run, to save budget; the 512p profiles show no
resolution-specific hazard), and our upsampler in isolation (`s2_upsampled`,
0.35, inherits stage 1's 0.34).

## LTX-2.5 image conditioning (I2V and keyframes, serve E5 / E9)

Reference: the same sol-engine / Lightricks/LTX-2 `fd4ded7` driver with
`--image PATH FRAME_IDX STRENGTH` (docs/ports/ltx25.md "Image
conditioning"), 768x512x121, dense stage 2, the matrix prompt, seed 1024.
`ltx25-i2v`: the TI2V beach fixture at frame 0. `ltx25-kf`: the same at
frame 0 plus its 1.35x zoom at pixel frame 120 (an appended keyframe block).
Runs: runtime 82ce54a / 0ead025, upstream `sol-ltx25:latest` with the scripts
at the same shas, RTX PRO 6000 (EUR-IS-1), 2026-09-28. Injected as for T2V
(noise, text), plus the conditioning (three arms, below). rel-L2 of ours
against the reference.

**Denoiser under conditioning** (the reference's conditioning latents
injected; this isolates the per-token timesteps, the appended keyframe tokens
and their RoPE, and the masked samplers):

| | ltx25-i2v | ltx25-kf | T2V 512p dense (above) |
|---|---|---|---|
| s1 block 0 / 24 / 47 (step 1) | 5.1e-3 / 7.9e-3 / 2.5e-2 | 3.4e-3 / 6.9e-3 / 1.0e-2 | 3.3e-3 / 5.4e-3 / 2.3e-2 |
| s1 latents steps 1-4 | 1.0e-3 … 2.3e-3 | 6.5e-4 … 1.3e-3 | 9e-4 … 2.2e-3 |
| s1 latents steps 5 / 6 / 7 / 8 | 7.8e-3 / 3.5e-2 / 0.10 / 0.16 | 2.4e-3 / 5.7e-3 / 1.8e-2 / 3.0e-2 | 7.1e-3 / 7.5e-2 / 0.25 / 0.34 |
| s2 block 0 / 24 / 47 (step 1) | 2.3e-3 / 6.5e-3 / 8.9e-3 | 1.8e-3 / 7.0e-3 / 6.8e-3 | 2.9e-3 / 5.0e-3 / 1.5e-2 |
| s2 latents steps 1 / 2 / 3 | 6.8e-3 / 2.0e-2 / 3.6e-2 | 3.6e-3 / 7.9e-3 / 1.3e-2 | 1.3e-2 / 4.8e-2 / 8.0e-2 |
| stage-1 / stage-2 entry state | 0 / 0 | 0 / 0 (1632 = 1536 grid + 96 appended rows) | |

Both are at or below the T2V profile at every step (the pinned frames anchor
the trajectory), with no block where the error jumps. The keyframe run
matches the reference's appended-token layout exactly (row counts and the
entry states are identical).

**Encoder and preprocessing:**

| | s1 (384x256) | s2 (768x512) |
|---|---|---|
| preprocessed pixels (CRF re-encode + resize) | 1.2e-2 | 1.2e-2 |
| conditioning latent, reference pixels injected (`-ownenc`: our encoder alone, f32 vs the reference's bf16) | 1.5e-2 | 1.7e-2 |
| conditioning latent, all ours (`-ownimg`) | 8.2e-2 | 0.14 |

The encoder is at the bf16 floor. The rest of the end-to-end latent gap is
the H.264 re-encode: our `ffmpeg` CLI libx264 and PyAV's libx264 / swscale
give pixels 1.2e-2 apart (max 0.06, about 7/255), and the one-frame CRF
round trip is not bit-reproducible across builds. End to end (`-ownimg`)
the final latents are 5.5e-2 (i2v) and 5.1e-2 (kf) from the reference.

**Frame fidelity** (the pinned output frames against the conditioning
images, cover + center crop to 768x512, ffmpeg SSIM / PSNR):

| | ours (reference latents) | ours end to end | reference |
|---|---|---|---|
| i2v frame 0 | 0.9666 / 39.46 dB | 0.9668 / 39.59 dB | 0.9661 / 39.25 dB |
| kf frame 0 | 0.9658 / 39.38 dB | 0.9659 / 39.49 dB | 0.9655 / 39.16 dB |
| kf frame 120 (last) | 0.9691 / 40.14 dB | 0.9689 / 40.13 dB | 0.9685 / 39.91 dB |

Verdict: **E5 and E9 pass.** The conditioned denoiser matches the reference
to its bf16 floor, the encoder too, and the pinned frames are as faithful to
the images as the reference's. Timings (ours, RTX PRO 6000, 512p, warm
weights): image preprocessing + encode for both stages 1.7-2.3 s,
denoise 10.8-12.5 s.

## LTX-2.5 reference-to-video (Ingredients IC-LoRA)

Reference: Lightricks/LTX-2 `fd4ded7` `python -m ltx_pipelines.ic_lora`
(`ICLoraPipeline`) with `--lora ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors
1.0 --video-conditioning reference.mov 1.0 --offload cpu`, 1536x896x121 at
24 fps (stage 1 at the LoRA's 768x448 bucket), seed 1024, the reference-sheet
prompt of `scripts/gpu/upstream/oracle.sh` (`LTX_REF_PROMPT`). The reference
clip is `scripts/gpu/fixtures/ltx-ref-sheet-768x448.png` looped into 121
lossless PNG frames (QuickTime), so upstream decodes the sheet's exact pixels.
Target `ltx25-ref2v`; ours: `fv-gpucheck ltx2 gen --two-stage --dense-stage2
--reference <sheet> --ic-lora <file>`. Run 2026-09-28, runtime and scripts at
`b38a408`, upstream `sol-ltx25:latest`, both on H100 80GB HBM3 in US-CA-2 on the
US volume `s2k01690bi`. Injected as for T2V (noise, text); the main arm also
injects the reference latent, `-ownimg` preprocesses and encodes the sheet
on our own. Stage 2 starts from the reference's entry state
(`FASTVIDEO_INJECT_STAGE2`, as every LTX target), so the stage-2 rows and the
clip metrics measure stage 2 and the decode alone.

Layout: identical. The reference latent is `[1, 128, 16, 14, 24]` upstream
(5376 tokens, downscale 1 and temporal scale 1 from the LoRA metadata), the
stage-1 sequence 10752 = 5376 grid + 5376 reference rows, the noise draws
`[1, 10752, 128]` then audio then the stage-2 renoise; ours attached all 480
LoRA pairs (every block's attn1 / attn2 q/k/v/out and FF).

| | main (reference latent injected) | -ownimg (all ours) |
|---|---|---|
| reference pixels (sheet, 768x448) | 1.6e-6 | 1.6e-6 |
| reference latent (our f32 encoder vs their bf16) | 1.7e-2 (ours, dumped) | 1.7e-2 |
| s1 step-0 input | 0 | 1.3e-2 (the reference rows) |
| s1 block 0 / 12 / 24 / 36 / 47 (step 1) | 2.1e-3 / 5.5e-3 / 1.7e-2 / 4.1e-2 / 3.5e-2 | |
| s1 latents steps 1-4 | 7.4e-4 … 1.8e-3 | 1.3e-2 (the reference rows' offset) |
| s1 latents steps 5 / 6 / 7 / 8 | 9.8e-3 / 5.9e-2 / 0.17 / 0.31 | 1.7e-2 / 5.7e-2 / 0.16 / 0.30 |
| s2 block 0 / 24 / 47 (step 1) | 2.4e-3 / 6.2e-3 / 1.6e-2 | same |
| s2 latents steps 1 / 2 / 3 | 1.2e-2 / 5.6e-2 / 0.10 | same |
| decoded clip vs the reference's (121 frames) | SSIM 0.982, PSNR 38.9 dB (min 32.2) | same |

Reading: the conditioned stage 1 with the LoRA fused starts at the bf16
floor (block 0 2.1e-3, the T2V 512p dense row above has 3.3e-3) and its
latents follow the T2V profile (T2V 512p dense: 9e-4 … 2.2e-3, then 7.1e-3 /
7.5e-2 / 0.25 / 0.34): the 8-step distilled stage 1 amplifies bf16 noise in
its last three steps, here as there. One difference from T2V: at step 1 the
video blocks 25-33 rise to 6e-2 … 0.12 (max-abs outliers up to ~830 in the
strided rows) and fall back to 3.5e-2 by block 47, while stage 2 (same
weights without the LoRA and without the reference) stays smooth. The
likeliest cause is the large-magnitude activations of the clean (timestep 0)
reference rows under the fused weights, where our single-rounding fuse and
upstream's double bf16 rounding differ by an ulp; it was not isolated
further (the GPU budget of this run). The encoder is at the bf16 floor, as
for I2V (1.5e-2 there), and preprocessing is exact. Stage 2 and the decode
match as for T2V. Verdict: **pass** at the bf16 floor, with the stage-1
mid-block bump noted.

Timings (ours, H100, warm steps): stage 1 about 1.0 s per step at 10752
tokens (the reference doubles the sequence), stage 2 1.8-2.1 s per step at
21504, upsample 2.8 s, decode 1.6 s; the IC-LoRA fuse 0.20 s and unfuse
0.10 s. Device memory 64 GiB live (the DiT with the kept base of the 480
LoRA linears is 61.8 GiB), peak 70 GiB. Upstream (`--offload cpu`) ran the
cell in 116 s, peak 9.3 GiB. Spend: upstream pod 13 min, runtime pod 8 min.

## Wan 2.2 TI2V-5B modules (Diffusers)

`scripts/gpu/upstream/oracle_wan22.py` (upstream step `oracle:wan22-ti2v`,
47 s on H100) dumps Diffusers' `AutoencoderKLWan` encode and decode of a
9-frame 704x1280 clip (fp32) and one bf16 `WanTransformer3DModel` forward
per timestep layout (`t2v_`: one timestep; `i2v_`: frame 0 at 0), with the
patch embedding, time projection and every block output. `fv-gpucheck wan
oracle` (matrix cells `wan5b-oracle`, `wan5b-oracle-exact`) injects the same
inputs into ours and `compare-dumps` diffs them. H100, `wan/cfce899-09271307`:
VAE decode PSNR 65.7 dB (rel-L2 1.8e-3; exact f32 mode 1.4e-4), encode
rel-L2 1.8e-3, DiT output rel-L2 2.7e-2 (t2v) / 1.5e-2 (i2v per-frame
timesteps), from 3e-3 at the patch embedding and block 0 with no jump.
Details in [ports/wan.md](ports/wan.md).
