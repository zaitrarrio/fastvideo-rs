# fastvideo-rs

Rust inference port of [FastVideo](https://github.com/hao-ai-lab/FastVideo) for the
Wan / FastWan family. **Primary generate path: cudarc CUDA** (lean
`--features cuda-cudarc`). Candle remains a frozen behavioral oracle for ports;
Luminal is frozen (no new Wan features).

Real inference targets Vast.ai NVIDIA GPUs. Mac/CI stay on CPU (cudarc without
`--features cuda` / `cuda-cudarc` errors if CUDA is requested).

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
| TeaCache / chunked SDPA / resident weights | TeaCache Wan2.1 poly (`FASTVIDEO_TEACACHE=1`); residency + BF16 default-on CUDA (`FASTVIDEO_RESIDENT=0` / `FASTVIDEO_BF16=0` escape); dense SDPA (+ SP via `--num-gpus`); Hopper defaults: TF32 (`FASTVIDEO_TF32=0` off), device UniPC (`FASTVIDEO_DEVICE_SCHED=0` off), SDPA chunk (`FASTVIDEO_SDPA_CHUNK`); logging via `FASTVIDEO_LOG` (`0`/`info`/`debug`); GPU-path auditing: `FASTVIDEO_STRICT_DEVICE=1` hard-fails a hot op (layer_norm/modulate/gate_mul/attention) that silently falls back to host compute with a live device instead of quietly running slower; `FASTVIDEO_DEVICE_STATS=1` prints a non-fatal per-op device-vs-host dispatch summary after `generate()` |
| Fun Control / Lucy edit | supported via `--control` / `--image` (latent inject or I2V pack) |
| Fun InP | supported (1.3B arch + `--image` I2V pack path) |
| Candle / Luminal | **frozen** |
| Sequence parallel | `--num-gpus N` (query-seq shard, real per-rank devices via `FASTVIDEO_SP_WORLD`, host-mediated all-gather) |
| VSA | `FASTVIDEO_VSA=1` → in-tree block-sparse SDPA (hard-fail without flag) |
| Flash-style SDPA | default (`FASTVIDEO_SDPA=flash`); `dense` / `sparse` overrides |
| GPU tests + benches | Vast (`scripts/vast-gpu-bench.sh`) |

## Tasks

Everything below is wrapped in a [Taskfile](https://taskfile.dev) — `task` alone lists them.

```bash
task check      # lint, tests, the NVRTC gate, cuda type-check — no GPU, no cost
task up         # build the dist binary, then open the control panel
task clip       # rent a GPU and run the full validation tier (~$0.12)
task gen PROMPT="a dog running on a beach"
task instances  # what is currently billing
task reap       # destroy every fvgpu-* instance
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

## GPU tests and benches (Vast)

```bash
./scripts/vast-sync.sh
# on the instance, once: bash scripts/vast-setup-cuda.sh
./scripts/vast-gpu-bench.sh          # cuda-cudarc tests + 1.3B smoke
./scripts/vast-gpu-bench.sh full     # plus 480p / 8-step
```

Default bench backend is `cudarc` (`FASTVIDEO_BENCH_BACKENDS=cudarc`).

Weights live on the instance (`scripts/vast-pull-weights.sh`): 1.3B T2V (fits 24GB bf16) plus I2V `image_encoder/` (CLIP ViT-H). Full I2V 14B / A14B DiTs need 48GB+ VRAM (`PULL_I2V_FULL=1` / `PULL_A14B=1`).

## GPU on Vast

Bring-up host: running RTX 4090 instance. Default cudarc path uses **device-resident F32** with optional **BF16 DiT GEMM** (`FASTVIDEO_BF16=0` to disable). Compare vs Candle on the same 256² / 9f / 2-step smoke via `scripts/vast-gpu-bench.sh`.

```bash
./scripts/vast-sync.sh
ssh -i ~/.ssh/id_strobe_vast -p 41695 root@<vast-host> 'bash /workspace/fastvideo-rs/scripts/vast-setup-cuda.sh'
./scripts/vast-generate.sh tiny
# then, after weights are on disk:
./scripts/vast-generate.sh 1.3b
```

Cargo registry and `target/` stay on the instance (`CARGO_HOME=/workspace/.cargo`; rsync skips `target`). Rebuilds after the first compile are incremental.

On the instance:

```bash
cargo run -p fastvideo-cli --release --features cuda-cudarc -- generate \
  --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --device cuda \
  --frames 9 --steps 2 --height 256 --width 256 \
  --output /workspace/fastvideo-out \
  --prompt "A curious raccoon in a field of sunflowers."
```

`--weights` is optional when the Diffusers snapshot is already in the HF cache.
Layout: `transformer/`, `vae/`, `text_encoder/`, `tokenizer/tokenizer.json`, and
`transformer_2/` for Wan 2.2 MoE. Bring-up checkpoint:
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.

## Local Linux CUDA build (Docker)

This Mac cannot compile CUDA natively (no Linux `nvcc`). Docker Desktop can, without a GPU:

```bash
./scripts/docker-build-cuda.sh
# → ./target-linux/release/fastvideo  (x86_64 Linux, cuda-cudarc)
```

Crates cache in the Docker volume `fastvideo-rs-cargo-registry`. Copy the binary onto Vast; you still need CUDA 12.4 runtime libs there (`nvrtc`, `cublas`, `curand`). You cannot *run* the CUDA binary in Docker on this Mac (no NVIDIA device).

## Layout

```
crates/
  fastvideo-ops        TensorBackend trait + host CPU reference
  fastvideo-core       registry, SamplingParam, VideoGenerator
  fastvideo-models     schedulers, packing, architecture configs
  fastvideo-loader     Diffusers safetensors (mmap / lazy)
  fastvideo-cudarc     **primary** Wan / LTX / H3 generate (cuBLAS / NVRTC / cuDNN)
  fastvideo-cli        `fastvideo` binary
  fastvideo-gpucheck   GPU parity / stage checks
scripts/
  vast-sync.sh         rsync onto the Vast box
  vast-setup-cuda.sh   rustup + CUDA 12.4 nvcc
  vast-pull-weights.sh 1.3B T2V + I2V CLIP on the instance
  vast-gpu-bench.sh    CUDA tests + generate/CLIP benches
  vast-generate.sh     CUDA tiny or 1.3B generate (cudarc)
```

## License

Apache-2.0. Derived from FastVideo (Apache-2.0); see `NOTICE`.
