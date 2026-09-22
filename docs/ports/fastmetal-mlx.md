# FastMetal / FastH3 MLX — port notes

Apple Silicon native runtime for FastVideo FastMetal-QAD and FastH3 Preview.
Lives in crate `fastvideo-mlx` (not `fastvideo-cudarc`).

---

## Requirements

| item | value |
|---|---|
| OS | **macOS 14+** |
| CPU | **Apple Silicon** (`aarch64-apple-darwin`) — Intel/Rosetta is not supported |
| Memory | **16 GB+** unified for 1.3B/5B; **36 GB+** for 14B / FastH3 Preview |
| Runtime | FastVideo MLX (INT8 DiT + TAEHV); optional `--features mlx` in this crate |

This machine may be x86_64 or Rosetta: `cargo test -p fastvideo-mlx` still
compiles the scaffold and asserts platform gating.

---

## Metal / mlx-rs gate

[`mlx-rs`](https://crates.io/crates/mlx-rs) **0.25+** exists and exposes Metal
(`features = ["metal"]`), but it only builds on Apple Silicon. This tree:

- Ships host stubs (`MlxArrayStub`, `MetalGate`) that compile everywhere.
- Does **not** declare `mlx-rs` in workspace `Cargo.toml` (keeps x86_64/Linux CI green).
- Documents the target-specific dep to add on an aarch64 Mac when wiring graphs:

```toml
[target.'cfg(all(target_os = "macos", target_arch = "aarch64"))'.dependencies]
mlx-rs = { version = "0.25", optional = true, default-features = false, features = ["metal"] }
```

`MetalGate::metal_ready()` stays false until that dep is linked.

---

## Hub ids

| preset | Hub id | notes |
|---|---|---|
| `fastmetal_qad_1_3b` | `FastVideo/FastMetal-1.3B-QAD` | 480×832, 81f, 3-step DMD |
| `fastmetal_qad_5b` | `FastVideo/FastMetal-5B-QAD` | 480p/720p |
| `fastmetal_qad_14b` | `FastVideo/FastMetal-14B-QAD` | 36 GB+ |
| `fasth3_mlx_preview` | `FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2` | local DiT conversion |

CUDA FastWan-QAD remains the NVIDIA path; do not load MLX packs into cudarc.

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-mlx` crate (configs + scaffold generate) | landed |
| Platform / Metal gate + host `MlxArrayStub` + weights layout check | landed |
| mlx-rs / Metal DiT + TAEHV | external (Apple Silicon + target-specific mlx-rs dep) |
| Registry Hub ids for CLI list | deferred (CUDA registry stays NVIDIA; MLX is separate entry) |
