# cudarc GPU validation ladder (`fv-gpucheck` + `scripts/gpu/validate.sh`)

Validates the cudarc Wan backend on rented GPUs, cheapest proof first. Each
tier runs only after every cheaper tier has passed on the same box, and a
failed stage stops the run. The goal is to find each bug on the cheapest
hardware that can show it.

Nothing is compared against another framework. The references are:
- **Plain-Rust math** for every kernel (softmax, norms, GEMM, attention, conv).
- **cudarc's own CPU path**: the same binary, run on the box with
  `--device cpu --mode exact`, dumps reference outputs from seeded weights and
  inputs. The GPU run recomputes them and compares.
- **GPU self-consistency**: batched rows must equal single-row forwards,
  forwards must be reproducible, and fast-path clips must stay close to
  exact-path clips.

## The ladder

| Tier | GPU | Proves | Typical wall time | Est. cost* |
|---|---|---|---|---|
| preflight `local` | none (Docker) | unit tests, NVRTC compile of every kernel (needs only `libnvrtc`), and the release binary the rented box runs, all built in Docker | 10–15 min cold, ~2 min warm | free |
| `run mathprobe` | any (pick with `FV_OFFER_QUERY_EXTRA=gpu_name=…`) | which cuBLAS math (FP32 / TF32 / bf16 compute / bf16 buffers) this GPU honors, with the image's cuBLAS and a newer one | ~5–7 min | ~$0.01–0.02 |
| **T1** `run kernels` | cheapest Ampere+ ≥8 GB | CUDA context; every NVRTC kernel, cuBLAS GEMMs and bf16 linears, dense/flash attention, cuDNN conv2d/conv3d and temporal unfold vs plain-Rust math; random-weight UMT5/DiT/VAE/UniPC/DMD GPU vs CPU path (exact and fast) | 10–15 min | $0.02–0.04 |
| `run compare` | 24GB+, 180GB disk | our clip stages **and** upstream FastVideo on the same box: install torch+fastvideo in a venv, time the same 8s clip per attention backend | ~50-70 min | ~$0.20-0.30 |
| `run compare-flux2` | 24GB+, 180GB disk | Flux2 T2I (default Klein 4B) rust `fastvideo bench` **and** upstream FastVideo `--workload t2i` on the same box and prompt | ~30–90 min (weight fetch) | ~$0.15–0.40 |
| **T2** `run parity` | ≥16 GB | T1, plus real 1.3B DiT forward, VAE decode and 2-step UniPC, GPU vs CPU path | 45–75 min (CPU reference is slow) | $0.10–0.30 |
| **T3** `run clip` | ≥24 GB, ≥80 GB RAM | T2, plus real prompt embeddings from cudarc UMT5-XXL on the GPU; a probe that projects 8s-clip time and VRAM before committing; two 8s clips (129 frames, 448×832, FastWan DMD) with per-step NaN and time guards and video quality gates; exact vs fast drift on a 2s clip | 90–150 min | $0.30–0.90 |

Compare-tier knobs: `FV_UPSTREAM_BACKENDS` (default `TORCH_SDPA VIDEO_SPARSE_ATTN`),
`FV_UPSTREAM_RUNS`, `FV_VSA_SPARSITY`, `FV_TORCH_BACKEND` (default `cu126`). Upstream
VSA ships tuned C++ kernels only for H100 (sm_90a); elsewhere it falls back to Triton.

The harness runs on macOS bash 3.2, where `"${arr[@]}"` on an empty array under
`set -u` is an error; expand possibly-empty arrays as `${arr[@]+"${arr[@]}"}`.
`scripts/gpu/lint.sh` checks this (and parses every script); `validate.sh local`
runs it.

`FASTVIDEO_CONV3D` picks the 3-D conv backend: `auto` (default, times each
candidate once per shape and keeps the winner), `cudnn`, `unfold` or
`cudnn-bf16`. bf16 only competes in `auto` under fast mode, since it changes
the numerics.

`FASTVIDEO_VAE_CHUNK` decodes several latent frames per pass instead of one.
`2` is ~9% faster on an 8s clip and produces identical frames; `4` runs out of
memory on a 24GB card, so the default stays `1`.

`FV_STAGE_ENV` passes environment to the stage process itself, for A/B runs.
For a same-GPU A/B, rent once with `--keep` and rerun against `--instance <id>`:
`FV_STAGE_ENV="FASTVIDEO_ATTN_PROBS_BF16=0" scripts/gpu/validate.sh run clip --instance 12345`.
`FV_STAGE_ENV="FASTVIDEO_SDPA=flash" scripts/gpu/validate.sh run clip`. Each report
records every `FASTVIDEO_*` it ran under, so the setting is visible in the artifacts.
Pair it with `FV_OFFER_QUERY_EXTRA=gpu_name=RTX_A5000` to compare against an earlier run on the same GPU.

\*Estimates from typical Vast on-demand prices. Check live prices with
`validate.sh offers <tier>`. `run` refuses offers above the tier's $/hr
ceiling and prints the worst-case cost (price × wall cap) before renting.

## Quick start

```bash
cp .env.example .env && chmod 600 .env   # then set VAST_API_KEY in .env
scripts/gpu/validate.sh local            # free preflight (run also does this)
scripts/gpu/validate.sh offers kernels   # see prices; rents nothing
scripts/gpu/validate.sh run kernels      # T1
scripts/gpu/validate.sh run clip         # T1 → T2 → T3 on one box
scripts/gpu/validate.sh offers compare-flux2
scripts/gpu/validate.sh run compare-flux2  # Flux2 rust vs upstream FastVideo
```

Flux2 compare knobs: `FV_FLUX2_REPO` (default `black-forest-labs/FLUX.2-klein-4B`),
`FV_FLUX2_STEPS` (4), `FV_FLUX2_GUIDANCE` (1.0), `FV_FLUX2_HEIGHT` / `FV_FLUX2_WIDTH`
(1024), `FV_FLUX2_PROMPT`. Rust `fastvideo bench` now matches upstream's
warm-median protocol: `--warmup` (default 1) + `--runs` (default 2). Override
with `FV_FLUX2_WARMUP` / `FV_FLUX2_RUNS`, or set `FV_UPSTREAM_RUNS` to move
both sides together. Compare rust `median_ms` to upstream `median_seconds`.
`profile.json` is the last timed rust generate.

SDPA A/B (default stays `dense`, same as tag `flux2-rope-p0-pre-fused-sdpa`):

```bash
FASTVIDEO_SDPA=dense FV_UPSTREAM_RUNS=2 scripts/gpu/validate.sh run compare-flux2
FASTVIDEO_SDPA=fused FV_UPSTREAM_RUNS=2 scripts/gpu/validate.sh run compare-flux2
```

Do **not** set `FASTVIDEO_SDPA=flash` (12–35× slower than dense on Wan).
`fused` / `mem_eff` is the query-tiled online-softmax path. See
[docs/flux2-generate-gap.md](../../docs/flux2-generate-gap.md).

Rust generate uses real Qwen3/Mistral3 text + the
full 2D VAE when those shards load; `FASTVIDEO_FLUX2_DUMMY_TEXT=1` and
`FASTVIDEO_FLUX2_TEXT_LEN` are A/B overrides. `docker.sh dist` builds
`fv-gpucheck` and the `fastvideo` CLI; the CLI is what `remote.sh
flux2-rust-bench` times. Details:
[docs/flux2-port.md](../../docs/flux2-port.md). No secrets in the repo — use
`VAST_API_KEY` from `.env`.

Options for `run`: `--offer ID`, `--instance ID` (reuse a running box; it is
never destroyed), `--keep`, `--skip-local`, `--clip-budget-min N`.
Settings come from `.env` (gitignored; `FV_ENV_FILE` points elsewhere), and shell variables take precedence: `VAST_API_KEY` (required; overrides the `vastai` CLI's saved key), `MAX_DPH`, `MAX_MINUTES`, `VAST_IMAGE`, `VAST_SSH_KEY`. The key is exported only to local `vastai` calls. `.env` is excluded from the repo sync, so it never reaches the rented box.

## Docker

Everything builds in Docker (`docker/gpucheck.Dockerfile`, driven by
`scripts/gpu/docker.sh`), never on the host:

```bash
scripts/gpu/docker.sh test            # unit tests
scripts/gpu/docker.sh nvrtc           # compile every kernel for sm 7.5–9.0 (no GPU)
scripts/gpu/docker.sh dist            # release binary → artifacts/gpucheck/dist/
scripts/gpu/docker.sh refs [--parity] # CPU-path references → artifacts/gpucheck/refs/<ref key>/
scripts/gpu/docker.sh image           # CUDA runtime image with the binary
scripts/gpu/docker.sh gpu <stage...>  # GPU stages on a local NVIDIA GPU
```

- **Builder image:** Ubuntu 22.04, the same glibc as the Vast image, with Rust and NVRTC 12.4. It needs no CUDA toolkit, because cudarc loads CUDA libraries at run time. The cargo registry and target dir live in named volumes, so rebuilds are incremental. The repo is mounted with `.env` masked.
- **Build id:** a hash of `crates/`, `Cargo.toml`, `Cargo.lock` and `rust-toolchain.toml`. `run` refuses a stale binary. The rented box only receives `scripts/` and the binary, so it compiles nothing.
- **References:** CPU-path references are cached in `artifacts/gpucheck/refs/<ref key>/`. The ref key hashes only the sources that shape CPU outputs: host math, model graphs, schedulers, weight loading, seeded test inputs and the dump format. GPU-only files (`kernels.rs`, `device.rs`, `conv.rs`, `stats.rs`, `log.rs`) are excluded, so kernel work keeps references valid.
  - `run` saves every reference its box dumps, and uploads cached ones to later boxes, skipping `model-cpu-ref` / `parity-cpu-ref`. `timings.md` marks those stages "cached (earlier run)".
  - The binary refuses a reference whose key differs from the run's.
  - `docker.sh refs [--parity]` fills the same cache locally. `--parity` needs the local HF snapshot and ≥28 GB of Docker VM memory.
- **cuBLAS:** bootstrap requires cuBLAS ≥ 12.9.1 (`FV_CUBLAS_MIN`) and installs `nvidia-cublas-cu12==12.9.1.4` beside older images. cuBLAS 12.4 predates Blackwell: on an RTX 5060 Ti it runs generic FP32 kernels, ignores TF32/bf16 compute and is 2.6× slower even in FP32 (`run mathprobe`, 2026-09-17).
- **Fast mode:** DiT/UMT5 linears keep bf16 weights on the device and multiply bf16 buffers (≈4× FP32 on Ampere and Blackwell); attention and other GEMMs use `CUBLAS_COMPUTE_32F_FAST_16BF` on F32 buffers; cuDNN convs allow TF32.
- **Local GPU runs:** only on Linux with an NVIDIA GPU and `nvidia-container-toolkit`. Docker Desktop on macOS has no GPU passthrough, and Macs have no NVIDIA GPUs, so on a Mac the GPU stages run only on rented boxes. There `docker.sh gpu` stops with that explanation.

## Fail-fast design

**Before renting:**
- vast auth works.
- The preflight passes.
- The offer is under the $/hr ceiling.
- Your balance covers the worst case.

**On the box, in order:**
1. `env`: driver CUDA ≥ 12.4, CUDA libraries present, disk free.
2. `bootstrap`: the uploaded binary must run on the box (glibc/arch check); installs ffmpeg and the HF downloader.
3. Background weight downloads (only the components a tier loads).
4. `nvrtc` → `device` → `kernels` → `model`. None of these need weights.
5. `parity` (real weights).
6. `embed`.
7. `probe`: times DiT forwards at 1, 3 and 9 latent frames, fits `a·S + b·S²`, measures VRAM, and projects the full clip. It exits with code 3 before the long run if the projection exceeds the budget or VRAM headroom.
8. `clip`: after every step, checks latents for NaN/Inf and re-projects total time, aborting as soon as either fails. It then applies quality gates: flat or saturated frames, frozen or incoherent motion, brightness flashes.

**Within a stage:** `kernels` runs with `--keep-going`, so one cheap rental
lists every broken kernel; the stage still fails at the end. Every other stage
stops at its first failed check.

**Teardown:**
- An exit trap pulls artifacts, then destroys the instance.
- A detached watchdog destroys at the wall cap, even if the local shell dies.
- `validate.sh reap` destroys every `fvgpu-*` instance.

**Remote stages survive dropped SSH connections.** They run detached and are
polled. A stage that dies without an exit code (for example, killed by OOM)
fails on the next poll.

## Modes and limits

- `--mode exact` sets `FASTVIDEO_BF16=0`, `FASTVIDEO_TF32=0` and
  `FASTVIDEO_DEVICE_SCHED=0`. The GPU must match the CPU path to round-off.
- `--mode fast` uses production defaults. Its looser limits bound the cost of
  BF16, TF32 and the order-1 device sampler.
- Limits live in `crates/fastvideo-gpucheck/src/mode.rs`. Reports record every
  measured value. Calibrate from real runs, and never loosen a limit without
  explaining the gap.

## Artifacts

`artifacts/gpucheck/runs/<stamp>-<tier>/` contains:
- `summary.json`: pass/fail, $/hr, wall minutes, estimated cost, and each stage's exit code and seconds.
- `stages.jsonl`
- `remote/<stage>.json`: every check with its values and limits, plus env flags, device and timings.
- `remote/logs/`
- `remote/refs/`: CPU-path reference outputs.
- `remote/embeds/`
- `remote/probe-8s.json`: the fitted projection.
- `remote/clips/<name>/`: `frames/` (with `frames/output.mp4`), `contact_sheet.png`, `latents.safetensors`.
  PNG frames stay on the instance; the mp4, contact sheet and latents are pulled after every stage.
- `artifacts/clips/<run>/<name>.mp4` (+ `<name>.png` contact sheet): every generated clip, copied as soon as its stage is pulled.
