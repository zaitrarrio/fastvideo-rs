# fastvideo-rs

Rust inference port of [FastVideo](https://github.com/hao-ai-lab/FastVideo) for the
Wan / FastWan family. Models currently run on **Candle** (CPU). Burn and Luminal
backends remain stubs.

This is Phase 1: UMT5, WanTransformer3D, Wan-VAE decode, a sampling loop, and PNG
frame write. Full 1.3B quality still needs a local Diffusers checkpoint; CI uses
`--tiny` zero weights so tests never download UMT5-XXL.

## Status

| Piece | State |
| --- | --- |
| Wan/FastWan HF id registry | done |
| UniPC sigma table + Euler step | done |
| FastWan DMD timesteps `[1000, 757, 522]` | done |
| Candle CPU UMT5 + DiT + VAE decode | done (tiny + 1.3B graph) |
| Diffusers safetensors loader | done (local dir) |
| Burn Flex / Luminal graphs | stubs |

## CLI

```bash
cargo run -p fastvideo-cli -- list-models
cargo run -p fastvideo-cli -- schedule --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers

# Zero-weight smoke test (no Hub download)
cargo run -p fastvideo-cli -- generate \
  --model FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
  --backend candle \
  --tiny \
  --output /tmp/fastvideo-tiny \
  --prompt "A curious raccoon in a field of sunflowers."

# Real Wan 2.1 1.3B Diffusers weights (local checkout)
cargo run -p fastvideo-cli -- generate \
  --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --backend candle \
  --weights /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --output outputs \
  --prompt "A curious raccoon in a field of sunflowers."
```

`--weights` must be a Diffusers layout with `transformer/`, `vae/`, and
`text_encoder/` safetensors. Tokenizer-backed prompts are not wired yet; the
graph still runs with dummy token ids. Bring-up checkpoint:
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.

## Layout

```
crates/
  fastvideo-ops        TensorBackend trait + host CPU reference
  fastvideo-core       registry, SamplingParam, VideoGenerator
  fastvideo-models     Wan DiT / VAE / UMT5 + schedulers
  fastvideo-loader     Diffusers safetensors mmap
  fastvideo-candle     Candle backend
  fastvideo-burn       Burn backend stub
  fastvideo-luminal    Luminal backend stub
  fastvideo-cli        `fastvideo` binary
```

## License

Apache-2.0. Derived from FastVideo (Apache-2.0); see `NOTICE`.
