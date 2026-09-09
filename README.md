# fastvideo-rs

Rust inference port of [FastVideo](https://github.com/hao-ai-lab/FastVideo) for the
Wan / FastWan family. **Real inference targets Vast.ai NVIDIA GPUs** via Candle
CUDA. Burn and Luminal backends remain stubs. Mac/CI stay on CPU.

## Status

| Piece | State |
| --- | --- |
| Wan/FastWan HF id registry | done |
| UniPC sigma table + Euler step | done |
| FastWan DMD timesteps `[1000, 757, 522]` | done |
| Candle CPU UMT5 + DiT + VAE decode | done (`--tiny`) |
| Candle CUDA (`--features cuda --device cuda`) | Vast RTX 4090 |
| Diffusers safetensors + tokenizer.json | local dir |
| Burn Flex / Luminal graphs | stubs |

## CLI (CPU / CI)

```bash
cargo test --workspace
cargo run -p fastvideo-cli -- generate \
  --model FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
  --tiny --output /tmp/fastvideo-tiny
```

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
  --weights /workspace/weights/Wan2.1-T2V-1.3B-Diffusers \
  --output /workspace/fastvideo-out \
  --prompt "A curious raccoon in a field of sunflowers."
```

`--weights` is a Diffusers layout: `transformer/`, `vae/`, `text_encoder/`, and
`tokenizer/tokenizer.json`. Bring-up checkpoint:
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
  vast-generate.sh     CUDA tiny or 1.3B generate
```

## License

Apache-2.0. Derived from FastVideo (Apache-2.0); see `NOTICE`.
