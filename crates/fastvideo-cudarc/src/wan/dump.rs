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
//!
//! `FASTVIDEO_DUMP_OPS=0,25,49` (default `0`) also dumps, at the first step,
//! the inside of those blocks (`step00_b<i>_<op>`: the modulated norm feeding
//! attention, the attention output, the modulated norm feeding the FFN, the
//! FFN output, and the six gathered AdaLN rows), for bisecting a diverging
//! block to the op. `scripts/gpu/upstream/oracle_dump.py` writes the same
//! names from the Python references, and [`super::inject`] reads their inputs
//! back into our run.

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
    write_raw(&dir.join(name), shape, data)
}

/// `<prefix>.f32` (raw little-endian f32) and `<prefix>.shape` (the dims,
/// space-separated): the dump format, at any path.
pub fn write_raw(prefix: &std::path::Path, shape: &[usize], data: &[f32]) -> Result<()> {
    if let Some(parent) = prefix.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(err)?;
    }
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let with = |ext: &str| {
        let mut p = prefix.as_os_str().to_owned();
        p.push(ext);
        PathBuf::from(p)
    };
    let mut f = std::fs::File::create(with(".f32")).map_err(err)?;
    f.write_all(&bytes).map_err(err)?;
    let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
    std::fs::write(with(".shape"), dims.join(" ")).map_err(err)
}

/// Read back what [`write_raw`] wrote: `(shape, data)`.
pub fn read_raw(prefix: &std::path::Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let with = |ext: &str| {
        let mut p = prefix.as_os_str().to_owned();
        p.push(ext);
        PathBuf::from(p)
    };
    let shape: Vec<usize> = std::fs::read_to_string(with(".shape"))
        .map_err(err)?
        .split_whitespace()
        .map(|d| d.parse::<usize>().map_err(err))
        .collect::<Result<_>>()?;
    let bytes = std::fs::read(with(".f32")).map_err(err)?;
    if bytes.len() != 4 * shape.iter().product::<usize>() {
        return Err(err(format!(
            "{}: {} bytes for shape {shape:?}",
            prefix.display(),
            bytes.len()
        )));
    }
    let data = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok((shape, data))
}

/// Host values under `name` (e.g. a schedule's sigmas).
pub fn host(name: &str, shape: &[usize], data: &[f32]) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    write(name, shape, data)
}

/// Every `stride`-th row of host rows `data` (`width` wide).
pub fn host_rows_strided(name: &str, data: &[f32], width: usize, stride: usize) -> Result<()> {
    if !enabled() {
        return Ok(());
    }
    let out = strided(data, width, stride);
    write(name, &[out.len() / width.max(1), width], &out)
}

fn strided(data: &[f32], width: usize, stride: usize) -> Vec<f32> {
    let (width, stride) = (width.max(1), stride.max(1));
    let rows = data.len() / width;
    let mut out = Vec::with_capacity(rows.div_ceil(stride) * width);
    for r in (0..rows).step_by(stride) {
        out.extend_from_slice(&data[r * width..(r + 1) * width]);
    }
    out
}

/// Blocks whose inside is dumped at the first step (`FASTVIDEO_DUMP_OPS`,
/// comma-separated indices; default block 0; `none` for none).
pub fn op_blocks() -> &'static [usize] {
    static OPS: OnceLock<Vec<usize>> = OnceLock::new();
    OPS.get_or_init(|| {
        parse_op_blocks(&std::env::var("FASTVIDEO_DUMP_OPS").unwrap_or_else(|_| "0".into()))
    })
}

fn parse_op_blocks(v: &str) -> Vec<usize> {
    v.split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect()
}

thread_local! {
    static OP_BLOCK: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Mark the block whose ops [`op`] dumps (`None` stops). The caller decides
/// when (first step, a block listed in [`op_blocks`]).
pub fn set_op_block(block: Option<usize>) {
    OP_BLOCK.with(|c| c.set(block));
}

/// The block [`op`] currently dumps, if any.
pub fn op_block() -> Option<usize> {
    if !enabled() {
        return None;
    }
    OP_BLOCK.with(std::cell::Cell::get)
}

/// `step00_b<block>_<name>`, rows strided like the block outputs, while a
/// block is marked by [`set_op_block`].
pub fn op(name: &str, t: &CudaTensor) -> Result<()> {
    match op_block() {
        Some(b) => rows_strided(&format!("step00_b{b}_{name}"), t, BLOCK_ROW_STRIDE),
        None => Ok(()),
    }
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
    let out = strided(&t.host_cow()?, width, stride);
    write(name, &[out.len() / width.max(1), width], &out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_keeps_every_nth_row() {
        let data: Vec<f32> = (0..10).map(|v| v as f32).collect();
        // 5 rows of 2; stride 2 keeps rows 0, 2, 4.
        assert_eq!(strided(&data, 2, 2), vec![0.0, 1.0, 4.0, 5.0, 8.0, 9.0]);
        assert_eq!(strided(&data, 2, 64), vec![0.0, 1.0]);
    }

    #[test]
    fn op_blocks_parse() {
        assert_eq!(parse_op_blocks("0, 25,49"), vec![0, 25, 49]);
        assert!(parse_op_blocks("none").is_empty());
    }
}
