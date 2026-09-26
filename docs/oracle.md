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
  `text_{video,audio}_ctx` (ours before injection), `noise_seed<seed>_<k>`
  (every `torch.randn` on a seeded generator, replayed by our `NoiseStream`),
  per stage `s1_`/`s2_`: `sigmas`, `{video,audio}_step00_in`,
  `step00_{video,audio}_in`, `step00_{video,audio}_block_<i>`,
  `{video,audio}_{vel,x0}_stepNN`, `{video,audio}_stepNN`; `s2_upsampled` and
  `s2_entry_{video,audio}` (our stage-2 entry before the reference's is
  injected; `FASTVIDEO_INJECT_STAGE2=0` keeps ours).

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

Found, outside the denoiser: **our text contexts differ from the
reference's by 0.34 (video) and 0.27 (audio) rel-L2**, with the same weights
on both sides (the rebuilt packs keep the Diffusers bytes). Every one of the
1024 rows is a normalized, distinct row on both sides (the registers fill
the padding), so it is not padding layout. It is injected away here and needs its own bisect of
the Gemma feature extraction and connectors (not done).

Not compared: 4K (not run, to save budget; the 512p profiles show no
resolution-specific hazard), and our upsampler in isolation (`s2_upsampled`,
0.35, inherits stage 1's 0.34).
