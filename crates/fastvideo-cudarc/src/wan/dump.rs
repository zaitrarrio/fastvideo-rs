//! `FASTVIDEO_DUMP_DIR=<dir>`: write intermediate tensors of a denoise to
//! `<dir>/<name>.f32` (raw little-endian f32) with `<dir>/<name>.shape`
//! (the dims, space-separated), for comparing two runs of the same seed
//! offline (`fv-gpucheck compare-dumps`).
//!
//! It answers "where do two precision paths part": the per-step latents and
//! velocities of both runs, and at the first step each block's output (every
//! [`BLOCK_ROW_STRIDE`]-th row: the block outputs are too wide to keep whole).
//! Off, every hook is one cached env read. A dump downloads (and therefore
//! synchronizes); it is a debugging switch, never a timed-run switch.

use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

use super::tensor::{CudaTensor, Result, TensorError};

/// Block outputs keep every this-many-th row.
pub const BLOCK_ROW_STRIDE: usize = 64;

/// The dump directory, when dumping is on.
pub fn dir() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        std::env::var_os("FASTVIDEO_DUMP_DIR")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    })
    .as_ref()
}

pub fn enabled() -> bool {
    dir().is_some()
}

fn err(e: impl std::fmt::Display) -> TensorError {
    TensorError::Message(format!("FASTVIDEO_DUMP_DIR: {e}"))
}

fn write(name: &str, shape: &[usize], data: &[f32]) -> Result<()> {
    let Some(dir) = dir() else { return Ok(()) };
    std::fs::create_dir_all(dir).map_err(err)?;
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut f = std::fs::File::create(dir.join(format!("{name}.f32"))).map_err(err)?;
    f.write_all(&bytes).map_err(err)?;
    let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
    std::fs::write(dir.join(format!("{name}.shape")), dims.join(" ")).map_err(err)
}

/// The whole tensor (f32 on the host; a bf16 tensor is widened exactly).
pub fn tensor(name: &str, t: &CudaTensor) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    write(name, &t.shape, &t.host_cow()?)
}

/// Every `stride`-th row of `t` viewed as `[rows, last_dim]`.
pub fn rows_strided(name: &str, t: &CudaTensor, stride: usize) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    let width = *t.shape.last().unwrap_or(&1);
    let host = t.host_cow()?;
    let rows = host.len() / width.max(1);
    let mut out = Vec::with_capacity(rows.div_ceil(stride.max(1)) * width);
    for r in (0..rows).step_by(stride.max(1)) {
        out.extend_from_slice(&host[r * width..(r + 1) * width]);
    }
    let kept = out.len() / width.max(1);
    write(name, &[kept, width], &out)
}
