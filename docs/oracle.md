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

starts one upstream pod per reference image (`runpod-http.sh upstream`, pod
step `oracle:<targets>`, kept alive to serve its dumps), and one runtime pod
(`FV_FAMILY=oracle`) that downloads each `oracle-<target>/oracle-dump.tar` as
soon as it is ready, runs ours injected, and compares. Targets:
`fasth3-8step`, `fasth3-4step-vsa`, `ltx25-512p[-dense]`, `ltx25-4k[-dense]`.
Results: `artifacts/runpod/oracle/<tag>/oracle-<target>-diff/` (the stderr
table and `gpucheck-out/*.json`); the dumps themselves stay on the pods.

Pass: step-1 per-block rel-L2 at bf16-rounding level (~2e-3) growing smoothly
with depth, no block where it jumps. A jump is bisected with the block's op
dumps (`FASTVIDEO_DUMP_OPS=<i>`): the first op whose rel-L2 leaves the bf16
floor is the divergent one.

## Results

See the section below, filled from the latest run.
