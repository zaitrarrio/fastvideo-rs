# fastvideo-rs

Rust inference port of [FastVideo](https://github.com/hao-ai-lab/FastVideo) for the
Wan / FastWan family. **Primary generate path: cudarc CUDA** (lean
`--features cuda-cudarc`). Candle remains a frozen behavioral oracle for ports;
Luminal is frozen (no new Wan features).

Real inference runs on Runpod NVIDIA GPUs (RTX PRO 6000 by default, the EU
weights volume). Mac/CI stay on CPU (cudarc without `--features cuda` /
`cuda-cudarc` errors if CUDA is requested).

## Status

| Piece | State |
| --- | --- |
| Wan/FastWan HF id registry | done |
| UniPC (Wan T2V) + FastWan DMD `[1000, 757, 522]` | done |
| **cudarc** Wan generate (Diffusers load) | **primary** |
| MoE `transformer_2` + dual CFG | cudarc done |
| I2V CLIP + VAE encode + 36-ch pack | cudarc done |
| Causal Self-Forcing mask | cudarc done |
| MP4 mux | optional (`--save-mp4` or `FASTVIDEO_SAVE_MP4=1`) |
| TOML generate overlay | `--config file.toml` (height/width/frames/steps/…) |
| TeaCache / chunked SDPA / resident weights | TeaCache Wan2.1 poly (`FASTVIDEO_TEACACHE=1`); residency always on with a CUDA device (`FASTVIDEO_RESIDENT=0` is refused), BF16 default-on (`FASTVIDEO_BF16=0` escape); dense SDPA (+ SP via `--num-gpus`); Hopper defaults: TF32 (`FASTVIDEO_TF32=0` off), device UniPC (`FASTVIDEO_DEVICE_SCHED=0` off), SDPA chunk (`FASTVIDEO_SDPA_CHUNK`); logging via `FASTVIDEO_LOG` (`0`/`info`/`debug`); GPU-path auditing: any op without a device path hard-fails with a live device (no silent CPU fallback); `FASTVIDEO_DEVICE_STATS=1` prints a non-fatal per-op device-vs-host dispatch summary after `generate()` |
| Fun Control / Lucy edit | supported via `--control` / `--image` (latent inject or I2V pack) |
| Fun InP | supported (1.3B arch + `--image` I2V pack path) |
| Candle / Luminal | **frozen** |
| Sequence parallel | `--num-gpus N` (query-seq shard, real per-rank devices via `FASTVIDEO_SP_WORLD`, host-mediated all-gather) |
| VSA | `FASTVIDEO_VSA=1` → in-tree block-sparse SDPA (hard-fail without flag) |
| Dense SDPA | one entry (`rt::sdpa::sdpa`), kernel chosen by `rt::sdpa_rule` (fused tensor-core / cuDNN / attn_dc); `FASTVIDEO_SDPA=cublas` / `flash` overrides (ADR-0015) |
| GPU tests + benches | Runpod (`task rtx6000`, `scripts/gpu/runpod-http.sh`) |

## Tasks

Everything below is wrapped in a [Taskfile](https://taskfile.dev) — `task` alone lists them.

```bash
task check      # shell lint, rustfmt --check, clippy -D warnings, tests, the NVRTC gate, cuda type-check — no GPU, no cost
task rtx6000    # one RTX PRO 6000 on Runpod: weight gate, then H3 / FastH3 / LTX-2.5 parity cells (~$2/hr)
task clip       # the older Vast ladder (scripts/gpu/validate.sh): T1 → T3 on one box
task gen PROMPT="a dog running on a beach"
```

Tasks that rent hardware say so in their description, with what a run costs.

## CLI (CPU / CI)

Mac/CI stay on the zero-weight graph. **Do not** load 1.3B on CPU.

```bash
cargo test --workspace
cargo run -p fastvideo-cli -- generate \
  --model FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
  --tiny --output /tmp/fastvideo-tiny
```

Default `--backend` is `cudarc`.

## GPU runs (Runpod)

GPU work runs on Runpod pods that mount the EU weights volume
(`bz0ofkfx45`, `fv-weights-h3-ltx-500`, EU-RO-1; the rest of the trees come
from the R2 bucket `fv-weights`, `docs/ops/r2-weights.md`). Nothing compiles on
a GPU box: CI builds the runtime image for every push, and
`scripts/gpu/runpod-http.sh` drives one RTX PRO 6000 on that image over the
REST API and the pod's HTTPS proxy (no SSH), with a wall-clock backstop and
the pod deleted at the end.

```bash
task rtx6000                              # weight gate, then H3 / FastH3 / LTX-2.5 parity cells
scripts/gpu/runpod-http.sh run [sha]      # the same by hand: create, run $FV_FAMILY, pull, destroy
scripts/gpu/runpod-http.sh kernels [sha]  # every kernel vs host math, no weights
scripts/gpu/runpod-http.sh upstream [sha] # the upstream Python references (scripts/gpu/upstream/)
```

`scripts/gpu/README.md` has the family matrix (now the fv-lab tiers of `lab/`; `runpod-matrix.sh` was deleted, ADR-0019 §22.20), the
evaluation layer and the older Vast ladder (`validate.sh`, `task clip`); the
port experiment drivers that were under `scripts/gpu/experiments/` are deleted
or fv-lab tiers (`h3-graphs`, `docs/adr/0019-fv-lab.md` §14). Serving
(fv-serve, images, Runpod serverless and pod deployments, GCP) is documented
under `docs/serve/`, starting at `docs/serve/design.md` and
`docs/serve/images.md`.

## Building

Rust compiles and tests on the shared Runpod **build pod** (a CPU pod on the
`fv-build` network volume), driven by `scripts/dev/build-pod.sh`: see
`docs/dev/build-pod.md`. Mac/CI build the CPU paths with plain `cargo`; the
CUDA feature type-checks without a GPU (`task build:cuda`), and the kernels
compile for every SM in Docker (`task nvrtc`).

## Layout

```
crates/
  fastvideo-ops        shared numeric helpers (FP8 E4M3 conversions)
  fastvideo-core       registry, SamplingParam, VideoGenerator
  fastvideo-models     schedulers, packing, architecture configs
  fastvideo-loader     Diffusers safetensors (mmap / lazy)
  fastvideo-cudarc     **primary** Wan / LTX / H3 generate (cuBLAS / NVRTC / cuDNN)
  fastvideo-cli        `fastvideo` binary
  fastvideo-gpucheck   GPU parity / stage checks
scripts/
  gpu/                 Runpod drivers (runpod-http.sh, runpod.sh; the tiers run through fv-lab),
                       weights tooling (r2-*.py, verify-weights.sh, the TSVs),
                       the Vast ladder (validate.sh) and experiments/
  dev/build-pod.sh     the shared CPU build pod (docs/dev/build-pod.md)
  serve/               fv-serve images, variants, releases and E2E clients
```

## License

Apache-2.0. Derived from FastVideo (Apache-2.0); see `NOTICE`.
