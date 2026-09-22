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
| Platform gate + tiny tests | landed |
| mlx-rs / Metal DiT + TAEHV | external blocker (Apple Silicon + MLX bindings) |
| Registry Hub ids for CLI list | deferred (CUDA registry stays NVIDIA; MLX is separate entry) |
