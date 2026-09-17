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
| preflight `local` | none (Docker) | unit tests, NVRTC compile of all 25 kernels (needs only `libnvrtc`), and the release binary the rented box runs, all built in Docker | 10–15 min cold, ~2 min warm | free |
| **T1** `run kernels` | cheapest Ampere+ ≥8 GB | CUDA context; every NVRTC kernel, cuBLAS F32/TF32/BF16 GEMM, flash/dense attention, conv3d, cuDNN conv2d vs plain-Rust math; random-weight UMT5/DiT/VAE/UniPC/DMD GPU vs CPU path (exact and fast) | 15–25 min | $0.02–0.06 |
| **T2** `run parity` | ≥16 GB | T1, plus real 1.3B DiT forward, VAE decode and 2-step UniPC, GPU vs CPU path | 45–75 min (CPU reference is slow) | $0.10–0.30 |
| **T3** `run clip` | ≥24 GB, ≥80 GB RAM | T2, plus real prompt embeddings from cudarc UMT5-XXL on the GPU; a probe that projects 8s-clip time and VRAM before committing; two 8s clips (129 frames, 448×832, FastWan DMD) with per-step NaN and time guards and video quality gates; exact vs fast drift on a 2s clip | 90–150 min | $0.30–0.90 |

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
```

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
scripts/gpu/docker.sh refs [--parity] # CPU-path references → artifacts/gpucheck/refs/
scripts/gpu/docker.sh image           # CUDA runtime image with the binary
scripts/gpu/docker.sh gpu <stage...>  # GPU stages on a local NVIDIA GPU
```

- **Builder image:** Ubuntu 22.04, the same glibc as the Vast image, with Rust and NVRTC 12.4. It needs no CUDA toolkit, because cudarc loads CUDA libraries at run time. The cargo registry and target dir live in named volumes, so rebuilds are incremental. The repo is mounted with `.env` masked.
- **Build id:** a hash of `crates/`, `Cargo.toml`, `Cargo.lock` and `rust-toolchain.toml`. `run` refuses a stale binary. The rented box only receives `scripts/` and the binary, so it compiles nothing.
- **References:** `refs` dumps CPU-path references with the same binary. When their build id matches, `run` uploads them and skips the box's own CPU-reference stages. `--parity` needs the local HF snapshot and ≥28 GB of Docker VM memory, and saves the slowest idle-GPU stage of T2.
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
