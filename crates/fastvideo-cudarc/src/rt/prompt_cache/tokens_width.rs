//! The `[tokens, width]` entry of [`super::Layout::TokensWidth`] in the
//! [`super::H3`] format, with the directory-and-key calls H3 and Wan make
//! (pre-B4c `h3/text_cache.rs:160-278`, which Wan called directly).

use std::path::{Path, PathBuf};

use super::{
    hex, Cached, Check, Entry, EntryRef, Header, Outcome, PromptCache, PromptCacheError, TensorRef,
};
use crate::rt::tensor::{Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// One cached conditioning: the token ids it was computed from and `[1, S, width]`.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedText {
    pub ids: Vec<u32>,
    pub width: usize,
    pub data: Vec<f32>,
}

impl Cached for CachedText {
    fn from_entry(entry: Entry) -> Option<Self> {
        let Header::Width(width) = entry.header else {
            return None;
        };
        let [tensor] = <[_; 1]>::try_from(entry.tensors).ok()?;
        Some(Self {
            ids: entry.ids,
            width,
            data: tensor.data,
        })
    }

    fn with_tensors<R>(
        &self,
        f: impl FnOnce(&[TensorRef<'_>]) -> R,
    ) -> std::result::Result<R, PromptCacheError> {
        let shape = [self.ids.len(), self.width];
        Ok(f(&[TensorRef {
            shape: &shape,
            data: &self.data,
        }]))
    }
}

fn cache(dir: &Path) -> PromptCache {
    PromptCache::new(dir, &super::H3)
}

/// `<dir>/<hex key>.h3text`.
pub fn entry_path(dir: &Path, key: &[u8; 32]) -> PathBuf {
    cache(dir).path(&hex(key))
}

/// The entry for `key`, if one exists, is intact, and was computed from
/// exactly `ids`. Anything else is a miss.
pub fn load(dir: &Path, key: &[u8; 32], ids: &[u32], width: usize) -> Option<CachedText> {
    let check = Check {
        ids,
        header: Header::Width(width),
    };
    cache(dir)
        .load(&hex(key), &check)
        .and_then(CachedText::from_entry)
}

/// Write atomically (temp file + rename), so a run killed mid-write leaves no
/// half entry under the final name. Failing to cache is never fatal.
pub fn store(
    dir: &Path,
    key: &[u8; 32],
    entry: &CachedText,
) -> std::result::Result<PathBuf, PromptCacheError> {
    let cache = cache(dir);
    entry
        .with_tensors(|tensors| {
            cache.store(
                &hex(key),
                &EntryRef {
                    ids: &entry.ids,
                    header: Header::Width(entry.width),
                    tensors,
                },
            )
        })
        .and_then(|r| r)
}

/// `load`, else `compute` and `store`. Returns the entry and whether it was a
/// hit. A `compute` that returns other than `ids.len() * width` values is an
/// error (nothing is stored).
pub fn get_or_compute(
    dir: &Path,
    key: &[u8; 32],
    ids: &[u32],
    width: usize,
    compute: impl FnOnce() -> Result<Vec<f32>>,
) -> Result<(CachedText, bool)> {
    let check = Check {
        ids,
        header: Header::Width(width),
    };
    let (entry, outcome) = cache(dir).get_or_compute(&hex(key), &check, || {
        let data = compute()?;
        if data.len() != ids.len() * width {
            return Err(msg(format!(
                "text encoder produced {} values for {} tokens of width {width}",
                data.len(),
                ids.len()
            )));
        }
        Ok(CachedText {
            ids: ids.to_vec(),
            width,
            data,
        })
    })?;
    Ok((entry, outcome == Outcome::Hit))
}
