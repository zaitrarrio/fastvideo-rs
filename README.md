# fastvideo-rs

Rust inference port of [FastVideo](https://github.com/hao-ai-lab/FastVideo) for the
Wan / FastWan family. **Real inference targets Vast.ai NVIDIA GPUs** via Candle
CUDA. Burn and Luminal backends remain stubs. Mac/CI stay on CPU.

## Status

| Piece | State |
| --- | --- |
| Wan/FastWan HF id registry | done |
| UniPC (Wan T2V) + FastWan DMD `[1000, 757, 522]` | done |
| Candle Wan T2V: UMT5 → DiT → VAE PNG | 1.3B Diffusers (auto HF cache) |
| I2V 36-ch pack + VAE encode + CLIP ViT-H | `--image`; CLIP from `image_encoder/` |
| Wan 2.2 MoE `transformer_2` | route by `boundary_ratio` |
| GPU tests + benches | Vast RTX 4090 (`scripts/vast-gpu-bench.sh`) |
| Burn Flex / Luminal graphs | stubs (PNG via Candle if `--weights`) |

## CLI (CPU / CI)

Mac/CI stay on the zero-weight graph. **Do not** load 1.3B on CPU.

```bash
cargo test --workspace
cargo run -p fastvideo-cli -- generate \
  --model FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
  --tiny --output /tmp/fastvideo-tiny
```

## GPU tests and benches (Vast)

```bash
./scripts/vast-sync.sh
# on the instance, once: bash scripts/vast-setup-cuda.sh
./scripts/vast-gpu-bench.sh          # CUDA tests + 1.3B smoke + CLIP encode
./scripts/vast-gpu-bench.sh full     # plus 480p / 8-step
```

Weights live on the instance (`scripts/vast-pull-weights.sh`): 1.3B T2V (fits 24GB bf16) plus I2V `image_encoder/` (CLIP ViT-H). Full I2V 14B / A14B DiTs need 48GB+ VRAM (`PULL_I2V_FULL=1` / `PULL_A14B=1`).

## GPU on Vast

Bring-up host: running RTX 4090 instance. Default dtype on CUDA is **BF16**.

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
cargo run -p fastvideo-cli --release --features cuda -- generate \
  --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --device cuda \
  --dtype bf16 \
  --frames 9 --steps 2 --height 256 --width 256 \
  --output /workspace/fastvideo-out \
  --prompt "A curious raccoon in a field of sunflowers."
```

`--weights` is optional when the Diffusers snapshot is already in the HF cache.
Layout: `transformer/`, `vae/`, `text_encoder/`, `tokenizer/tokenizer.json`, and
`transformer_2/` for Wan 2.2 MoE. Bring-up checkpoint:
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.

## Local Linux CUDA build (Docker)

This Mac cannot compile Candle `--features cuda` natively (no Linux `nvcc`). Docker Desktop can, without a GPU:

```bash
./scripts/docker-build-cuda.sh
# → ./target-linux/release/fastvideo  (x86_64 Linux)
```

Crates cache in the Docker volume `fastvideo-rs-cargo-registry`. Copy the binary onto Vast; you still need CUDA 12.4 runtime libs there (`nvrtc`, `cublas`, `curand`). You cannot *run* the CUDA binary in Docker on this Mac (no NVIDIA device).

## Layout

```
crates/
  fastvideo-ops        TensorBackend trait + host CPU reference
  fastvideo-core       registry, SamplingParam, VideoGenerator
  fastvideo-models     Wan DiT / VAE / UMT5 + schedulers
  fastvideo-loader     Diffusers safetensors load
  fastvideo-candle     Candle backend
  fastvideo-burn       Burn backend stub
  fastvideo-luminal    Luminal backend stub
  fastvideo-cli        `fastvideo` binary
scripts/
  vast-sync.sh         rsync onto the Vast box
  vast-setup-cuda.sh   rustup + CUDA 12.4 nvcc
  vast-pull-weights.sh 1.3B T2V + I2V CLIP on the instance
  vast-gpu-bench.sh    CUDA tests + generate/CLIP benches
  vast-generate.sh     CUDA tiny or 1.3B generate
```

## License

Apache-2.0. Derived from FastVideo (Apache-2.0); see `NOTICE`.
