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

## Results (2026-09-26, RTX PRO 6000, runtime f49db62 / 5312e4b)

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

Verdict: **not passed.** Blocks 0 to ~24 sit at 4-8e-3, but from about
block 25 the difference grows ~15% per block to 0.15 (dense) to 0.45 (VSA),
and the final latents differ by 0.4-0.5 rel-L2. It is not VSA's tile
selection (the dense control grows the same way; the VSA recipes add a step
at block 18 and roughly double the late-block level), and no single op
diverges. The open question is whether this is bf16 rounding amplified by
depth (the reference itself runs everything in bf16, including RoPE tables
and `1 + scale`) or a systematic difference; the f32-activation control and
the sparsity-0 control were queued when the Runpod account balance ran out,
and are the next run.

Not compared yet: LTX-2.5 (the reference's single-file packs need
`RECON_ACCEPT_MISMATCH=1`, now set by the driver; the rerun was cut by the
balance), and 4K.
