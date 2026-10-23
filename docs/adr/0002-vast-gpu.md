# ADR-0002: Real GPU inference on Vast CUDA hardware

- Status: Accepted (amended 2026-09-10; cudarc-primary). Superseded in
  practice 2026-10-23: GPU work runs on Runpod (`task rtx6000`,
  `scripts/gpu/runpod-http.sh`) and compiles on the shared build pod
  (docs/dev/build-pod.md); the Vast-era helper scripts are gone.
- Date: 2026-09-09
- Decision-id: FVID-2026-09-09-vast-gpu

## Context

Phase 1 proved the Wan graph on Candle CPU with `--tiny` zero weights. That is
not a product bar: 1.3B UMT5+DiT+VAE will not run in useful time on a laptop,
and this Mac cannot compile CUDA kernels.

The account already has Vast.ai GPU capacity. A running RTX 4090
(`loom-bench-rtx-4090`) is the bring-up host.

## Decision

- Target hardware for real inference is **Vast.ai NVIDIA CUDA** (24GB+ VRAM).
- **Primary GPU path is cudarc** via `--features cuda-cudarc` (lean; skips
  Candle CUDA kernels and Burn CubeCL). Full `--features cuda` remains for
  frozen Candle/Burn benches only.
- Mac/CI stay on CPU (`cargo test` without cuda features).
- First GPU gate is tiny generate on CUDA; 1.3B Diffusers weights follow on the
  same box.

## Consequences

- Default rentals use the **slim** GHCR image (CUDA runtime libs + `fv-gpucheck`,
  no PyTorch). Nothing compiles on a GPU box: CI builds the binary and the
  shared build pod (docs/dev/build-pod.md) is the fallback.
- Tiers that need Python+torch (oracle, compare, taehv, H3/LTX-2 reference
  stages) rent images **FROM `vastai/pytorch`** (`docker/vast-pytorch.Dockerfile`),
  selected automatically by `validate.sh` (`VAST_IMAGE_FLAVOR=auto`).
- The GPU drivers (`scripts/gpu/runpod-http.sh`, `scripts/gpu/validate.sh`)
  and the default bench backends target cudarc.
- Weights live on the instance disk, not in this git repo.

Verified (historical): tiny Candle CUDA generate (`--dtype f32`) wrote PNG
frames on RTX 4090 instance `50416610`. Current scripts use cudarc.
