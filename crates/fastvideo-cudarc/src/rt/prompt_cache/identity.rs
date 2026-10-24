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
    let files = head_tail_files(path, prefix)?;
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

/// The files [`files_head_tail`] identifies, sorted: `path` itself, or every
/// `.safetensors` directly under it whose name starts with `prefix`.
fn head_tail_files(path: &Path, prefix: Option<&str>) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = if path.is_file() {
        vec![path.to_path_buf()]
    } else {
        std::fs::read_dir(path)
            .map_err(|e| msg(format!("{}: {e}", path.display())))?
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
    Ok(files)
}

/// What the memo of [`files_head_tail_cached`] is keyed by: the path, the
/// prefix and every identified file's path, size and modification time.
type Stamp = (
    PathBuf,
    Option<String>,
    Vec<(PathBuf, u64, Option<std::time::SystemTime>)>,
);

/// [`files_head_tail`], computed once per process for the same files: the
/// heads and tails are read again only when a file's size or modification
/// time changed, or the set of files did (a weight swap under the same
/// path). Each call lists and stats the files, which is cheap; reading
/// 2 MiB per file on every prompt is not. The digest is the same as
/// [`files_head_tail`]'s.
pub fn files_head_tail_cached(path: &Path, prefix: Option<&str>) -> Result<[u8; 32]> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock, PoisonError};
    static MEMO: OnceLock<Mutex<HashMap<Stamp, [u8; 32]>>> = OnceLock::new();
    let files = head_tail_files(path, prefix)?;
    let mut stat = Vec::with_capacity(files.len());
    for file in files {
        let meta = std::fs::metadata(&file).map_err(|e| msg(format!("{}: {e}", file.display())))?;
        stat.push((file, meta.len(), meta.modified().ok()));
    }
    let stamp: Stamp = (path.to_path_buf(), prefix.map(str::to_owned), stat);
    let memo = MEMO.get_or_init(Default::default);
    let known = memo
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&stamp)
        .copied();
    if let Some(digest) = known {
        return Ok(digest);
    }
    let digest = files_head_tail(path, prefix)?;
    memo.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(stamp, digest);
    Ok(digest)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cached_identity_is_the_identity_and_follows_a_changed_file() {
        let dir = std::env::temp_dir().join(format!("fv-identity-memo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("a.safetensors");
        std::fs::write(&file, vec![1u8; 3 * PROBE as usize]).unwrap();
        std::fs::write(dir.join("b.safetensors"), b"bb").unwrap();
        let first = files_head_tail_cached(&dir, None).unwrap();
        assert_eq!(first, files_head_tail(&dir, None).unwrap());
        assert_eq!(first, files_head_tail_cached(&dir, None).unwrap());
        // One file of the pair alone, by prefix and by path.
        assert_eq!(
            files_head_tail_cached(&dir, Some("a")).unwrap(),
            files_head_tail(&file, None).unwrap()
        );
        assert_eq!(
            files_head_tail_cached(&file, None).unwrap(),
            files_head_tail(&file, None).unwrap()
        );
        // A changed size is read again, whatever the clock's resolution.
        std::fs::write(&file, vec![2u8; 3 * PROBE as usize + 1]).unwrap();
        let second = files_head_tail_cached(&dir, None).unwrap();
        assert_ne!(second, first);
        assert_eq!(second, files_head_tail(&dir, None).unwrap());
        // A new file is a new set.
        std::fs::write(dir.join("c.safetensors"), b"c").unwrap();
        assert_eq!(
            files_head_tail_cached(&dir, None).unwrap(),
            files_head_tail(&dir, None).unwrap()
        );
        assert!(files_head_tail_cached(&dir.join("missing"), None).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
