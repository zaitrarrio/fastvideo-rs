# fastvideo-rs

Rust inference port of [FastVideo](https://github.com/hao-ai-lab/FastVideo) for the
Wan / FastWan family. Models are written once against a `TensorBackend` trait and
run on **Burn**, **Candle**, or **Luminal**.

This is Phase 0: workspace, registry, sampling presets, and flow-match / DMD
schedulers. DiT, VAE, and UMT5 forwards are next (Phase 1).

## Status

| Piece | State |
| --- | --- |
| Wan/FastWan HF id registry | done |
| UniPC sigma table + Euler step | done |
| FastWan DMD timesteps `[1000, 757, 522]` | done |
| Candle CPU alloc / add / matmul | done |
| Burn Flex / Luminal graphs | stubs |
| WanTransformer3D + Wan-VAE + UMT5 | not yet |

## CLI

```bash
cargo run -p fastvideo-cli -- list-models
cargo run -p fastvideo-cli -- schedule --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers
cargo run -p fastvideo-cli -- generate \
  --model FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
  --backend candle \
  --prompt "A curious raccoon in a field of sunflowers."
```

`generate` resolves the checkpoint and sampling preset, then exits with
`WanTransformer3D is not implemented yet` until Phase 1.

## Layout

```
crates/
  fastvideo-ops        TensorBackend trait + host CPU reference
  fastvideo-core       registry, SamplingParam, VideoGenerator
  fastvideo-models     Wan configs + schedulers
  fastvideo-loader     Diffusers name mapping (download in Phase 1)
  fastvideo-candle     Candle backend
  fastvideo-burn       Burn backend stub
  fastvideo-luminal    Luminal backend stub
  fastvideo-cli        `fastvideo` binary
```

## License

Apache-2.0. Derived from FastVideo (Apache-2.0); see `NOTICE`.
