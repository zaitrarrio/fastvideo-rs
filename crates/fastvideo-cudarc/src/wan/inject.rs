//! `FASTVIDEO_INJECT_DIR=<dir>`: read a reference run's inputs back from a
//! dump directory (the [`super::dump`] format: `<name>.f32`, raw little-endian
//! f32, plus `<name>.shape`), so a Rust denoise starts from exactly the noise
//! (and, optionally, the text conditioning) a Python reference drew. Torch's
//! Philox generator is not reproduced; with its draws injected, whatever still
//! differs between the two runs is the denoiser's own.
//!
//! Which names a pipeline reads is its own business (H3: `video_step00_in`,
//! `audio_step00_in`, `text_hidden`; LTX-2: see its pipeline). An absent name
//! is not injected (and logged); a present one whose size disagrees is an
//! error. `FASTVIDEO_INJECT_TEXT=0` keeps our own text conditioning even when
//! the directory holds the reference's.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::tensor::{Result, TensorError};

pub fn dir() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        std::env::var_os("FASTVIDEO_INJECT_DIR")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    })
    .as_ref()
}

pub fn enabled() -> bool {
    dir().is_some()
}

/// Whether text conditioning is injected too (`FASTVIDEO_INJECT_TEXT`, default on).
pub fn text_enabled() -> bool {
    enabled() && std::env::var("FASTVIDEO_INJECT_TEXT").map_or(true, |v| v != "0")
}

fn err(e: impl std::fmt::Display) -> TensorError {
    TensorError::Message(format!("FASTVIDEO_INJECT_DIR: {e}"))
}

/// `(shape, values)` of `dir/name.{f32,shape}`; `None` when the file is absent.
pub fn read(dir: &Path, name: &str) -> Result<Option<(Vec<usize>, Vec<f32>)>> {
    let data = dir.join(format!("{name}.f32"));
    if !data.is_file() {
        return Ok(None);
    }
    let shape_text = std::fs::read_to_string(dir.join(format!("{name}.shape"))).map_err(err)?;
    let shape: Vec<usize> = shape_text
        .split_whitespace()
        .map(|s| s.parse::<usize>().map_err(err))
        .collect::<Result<_>>()?;
    let bytes = std::fs::read(&data).map_err(err)?;
    let numel: usize = shape.iter().product();
    if bytes.len() != numel * 4 {
        return Err(err(format!(
            "{}: {} bytes for shape {shape:?}",
            data.display(),
            bytes.len()
        )));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(Some((shape, values)))
}

/// [`read`] from `FASTVIDEO_INJECT_DIR`, checked to hold `numel` values (the
/// reference's shape may carry a batch dim ours does not, so only the count is
/// compared). `None` when injection is off or the reference has no `name`.
pub fn load_numel(name: &str, numel: usize) -> Result<Option<Vec<f32>>> {
    let Some(dir) = dir() else { return Ok(None) };
    match read(dir, name)? {
        None => {
            crate::wan::log::info(format_args!(
                "inject: {} has no {name}; not injected",
                dir.display()
            ));
            Ok(None)
        }
        Some((shape, v)) if v.len() == numel => {
            crate::wan::log::info(format_args!("inject: {name} {shape:?} from the reference"));
            Ok(Some(v))
        }
        Some((shape, v)) => Err(err(format!(
            "{name}: reference shape {shape:?} ({} values), ours needs {numel}",
            v.len()
        ))),
    }
}

/// [`read`] from `FASTVIDEO_INJECT_DIR` with the shape the reference wrote.
pub fn load(name: &str) -> Result<Option<(Vec<usize>, Vec<f32>)>> {
    let Some(dir) = dir() else { return Ok(None) };
    let got = read(dir, name)?;
    match &got {
        Some((shape, _)) => {
            crate::wan::log::info(format_args!("inject: {name} {shape:?} from the reference"))
        }
        None => crate::wan::log::info(format_args!(
            "inject: {} has no {name}; not injected",
            dir.display()
        )),
    }
    Ok(got)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_dump_format() {
        let dir = std::env::temp_dir().join(format!("fv-inject-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let v = [1.5f32, -2.0, 3.25];
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(dir.join("t.f32"), &bytes).unwrap();
        std::fs::write(dir.join("t.shape"), "1 3").unwrap();
        let (shape, got) = read(&dir, "t").unwrap().unwrap();
        assert_eq!(shape, vec![1, 3]);
        assert_eq!(got, v);
        assert!(read(&dir, "absent").unwrap().is_none());
        std::fs::write(dir.join("t.shape"), "2 3").unwrap();
        assert!(read(&dir, "t").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
