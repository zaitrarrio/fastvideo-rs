//! Encoder identities computed from weight *files*, without opening them as
//! checkpoints (ADR-0021 "Key"). Two schemes, kept as they were:
//!
//! * [`files_head_tail`] (LTX-2, pre-B4c `ltx2/text_cache.rs:61-113`): every
//!   `.safetensors` file's name, size, first and last mebibyte, each a
//!   length-prefixed field; a missing or unreadable file is an error.
//! * [`files_head`] (Wan, pre-B4c `wan/pipeline.rs:730-758`): every
//!   `.safetensors` file's name, size and the bytes of one `read` of up to
//!   256 KiB, hashed *without* length prefixes; unreadable files and
//!   directories contribute what could be read and never fail.
//!
//! H3's identity is built from tensors instead (the original shards and the
//! slim re-pack share entries) and lives with its tap list in
//! `h3::text_cache::encoder_identity`.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::rt::tensor::{Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// A length-prefixed field, so `("ab", "c")` and `("a", "bc")` hash differently.
fn field(h: &mut Sha256, data: &[u8]) {
    h.update((data.len() as u64).to_le_bytes());
    h.update(data);
}

/// How much of each end of a weight file goes into [`files_head_tail`].
pub const PROBE: u64 = 1 << 20;

/// Identity of a checkpoint — a file, or every `.safetensors` directly under a
/// directory (optionally only those whose name starts with `prefix`) — from
/// file names, sizes and the first and last mebibyte of each. The head is the
/// safetensors header (every key, dtype, shape and offset), the tail is real
/// tensor bytes, so a fine-tune with the same layout still keys differently.
/// No timestamps: a re-download of the same files keeps its cache.
pub fn files_head_tail(path: &Path, prefix: Option<&str>) -> Result<[u8; 32]> {
    let io = |e: std::io::Error, p: &Path| msg(format!("{}: {e}", p.display()));
    let mut files: Vec<PathBuf> = if path.is_file() {
        vec![path.to_path_buf()]
    } else {
        std::fs::read_dir(path)
            .map_err(|e| io(e, path))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .filter(|p| {
                prefix.is_none_or(|pre| {
                    p.file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with(pre))
                })
            })
            .collect()
    };
    if files.is_empty() {
        return Err(msg(format!(
            "{}: no safetensors files to identify",
            path.display()
        )));
    }
    files.sort();
    let mut h = Sha256::new();
    for file in &files {
        let mut f = std::fs::File::open(file).map_err(|e| io(e, file))?;
        let size = f.metadata().map_err(|e| io(e, file))?.len();
        field(
            &mut h,
            file.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
                .as_bytes(),
        );
        field(&mut h, &size.to_le_bytes());
        let mut buf = vec![0u8; PROBE.min(size) as usize];
        f.read_exact(&mut buf).map_err(|e| io(e, file))?;
        field(&mut h, &buf);
        if size > PROBE {
            let tail = PROBE.min(size - PROBE);
            f.seek(SeekFrom::End(-(tail as i64)))
                .map_err(|e| io(e, file))?;
            buf.truncate(tail as usize);
            f.read_exact(&mut buf).map_err(|e| io(e, file))?;
            field(&mut h, &buf);
        }
    }
    Ok(h.finalize().into())
}

/// How many leading bytes of each file [`files_head`] reads (in one `read`).
pub const HEAD: usize = 256 << 10;

/// Identity of the `.safetensors` files directly under `dir` (none for
/// `None`): file names, sizes and each file's first [`HEAD`] bytes — the
/// safetensors header and the start of the data. Infallible on purpose, as
/// before: what cannot be listed, sized or read contributes nothing (a size
/// of 0, no bytes), so the key degrades instead of failing a generation.
pub fn files_head(dir: Option<&Path>) -> [u8; 32] {
    let mut h = Sha256::new();
    if let Some(dir) = dir {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .map(|rd| rd.flatten().map(|e| e.path()).collect())
            .unwrap_or_default();
        files.retain(|p| p.extension().is_some_and(|e| e == "safetensors"));
        files.sort();
        for f in files {
            let name = f.file_name().map(|n| n.to_string_lossy().into_owned());
            h.update(name.unwrap_or_default().as_bytes());
            h.update(
                std::fs::metadata(&f)
                    .map(|m| m.len())
                    .unwrap_or(0)
                    .to_le_bytes(),
            );
            if let Ok(mut file) = std::fs::File::open(&f) {
                let mut buf = vec![0u8; HEAD];
                let n = file.read(&mut buf).unwrap_or(0);
                h.update(&buf[..n]);
            }
        }
    }
    h.finalize().into()
}
