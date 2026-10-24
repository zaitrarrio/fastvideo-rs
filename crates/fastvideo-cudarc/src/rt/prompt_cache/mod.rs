//! The disk prompt cache: one implementation for every family's text
//! conditioning (plan B4c, `docs/adr/0021-prompt-cache.md`).
//!
//! Encoding a prompt is the most expensive thing a repeated prompt pays for
//! and the cheapest to skip: H3 streams 50 GB of encoder weights for a 700 KB
//! `[1, S, 5120]` result, LTX-2 47 GB of Gemma for 31 MB of connector
//! contexts, Wan a UMT5 forward for `[1, S, 4096]`. The result is a pure
//! function of the prompt, the tokenizer, a length and the encoder weights,
//! so it is stored under a key made of exactly those ([`key`]).
//!
//! Three copies of this existed before B4c (H3 `h3/text_cache.rs`, LTX
//! `ltx2/text_cache.rs`, Wan's key in `wan/pipeline.rs` over H3's store).
//! They agree on the protocol and differ in details that are on disk today;
//! every difference is a named parameter here, not a normalisation:
//!
//! * the key: [`KeyRecipe`] (domain string, prompt trimming) and its
//!   arguments (what `length` and the identity fields are);
//! * the entry file: [`Format`] — magic, extension, [`Layout`] of the header
//!   and tensors, [`TempName`], [`Durability`], [`TempCleanup`],
//!   [`Finite`] check on load, [`ErrorText`] and the [`StoreLog`] line.
//!
//! What every family shares: an entry is `magic | header | ids | tensors |
//! sha256(all of that)`, named `<hex key>.<extension>` in the cache
//! directory, written to a temporary name and renamed (a reader never sees
//! half an entry under the final name), and read back only when the seal,
//! the magic, the layout, the token ids and the header all match. Anything
//! else — truncated, bit-flipped, foreign, from another format version, made
//! from other token ids — is a **miss**, never an error and never a misread:
//! a cache can only make a run faster, not make it fail or lie. A cache that
//! cannot be written is logged and costs the next run time, not this run its
//! result.

pub mod identity;
pub mod key;
pub mod tokens_width;

#[cfg(test)]
mod equality;
#[cfg(test)]
pub(crate) mod legacy;

use std::io::Write;
use std::path::{Path, PathBuf};

pub use key::{hex, sha256, Fields, KeyRecipe, PromptNorm, H3_KEY, LTX2_KEY, WAN_KEY};

/// H3's entry magic, also the first field of its key ([`H3_KEY`]). Bump when
/// the H3 file layout or key derivation changes.
pub(crate) const H3_MAGIC: [u8; 8] = *b"H3TEXT01";

/// Bytes of the trailing seal: SHA-256 of everything before it.
const SEAL: usize = 32;

/// Whether the cache answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// No cache directory was given, or this call bypassed it.
    Off,
    Hit,
    Miss,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Hit => "hit",
            Self::Miss => "miss",
        }
    }
}

/// Why an entry was not written. Never fatal: [`PromptCache::get_or_compute`]
/// logs it ([`StoreLog`]) and returns the computed value.
#[derive(Debug, thiserror::Error)]
pub enum PromptCacheError {
    /// [`Layout::TokensWidth`]: no tokens, or not `tokens · width` values.
    #[error("{values} values for {tokens} tokens of width {width}")]
    Shape {
        values: usize,
        tokens: usize,
        width: usize,
    },
    /// An entry that does not fit the format's [`Layout`] (wrong header kind,
    /// tensor count, or a shape that does not match its data).
    #[error("{format} entry: {detail}")]
    Layout {
        format: &'static str,
        detail: String,
    },
    /// [`ErrorText::EntryPath`]: the path that failed (the directory or the
    /// entry), then the error.
    #[error("{}: {source}", path.display())]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// [`ErrorText::CacheDir`]: `text cache <dir>: <error>` for any step.
    #[error("text cache {}: {source}", dir.display())]
    CacheDir {
        dir: PathBuf,
        source: std::io::Error,
    },
    /// A device tensor could not be read back to the host (its message).
    #[error("{0}")]
    Tensor(String),
}

/// The header word after the token count, and what a hit is checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Header {
    /// [`Layout::TokensWidth`]: the hidden width (H3 5120, Wan 4096).
    Width(usize),
    /// [`Layout::LenRealShaped`]: how many trailing ids are real (LTX).
    Real(usize),
}

/// How header and tensors are laid out between the magic and the seal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// `tokens u64 | width u64 | ids u32[tokens] | data f32[tokens·width]`:
    /// one `[tokens, width]` tensor whose shape *is* the header (H3, Wan).
    /// `tokens` and `width` must be non-zero.
    TokensWidth,
    /// `len u32 | real u32 | ids u32[len] | tensors × (rank u32 |
    /// shape u64[rank] | data f32[∏ shape])` (LTX: video then audio).
    /// `len`, `real` and `rank` are written with `as u32`, as before.
    LenRealShaped { tensors: usize, max_rank: usize },
}

/// The temporary name an entry is written under before the rename.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TempName {
    /// The entry path with its extension replaced by `tmp<pid>`
    /// (`<key>.tmp<pid>`; H3, Wan).
    ReplaceExtension,
    /// `<key>.<pid>.part` beside it (LTX).
    PidPart,
}

/// How the temporary file is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    /// `fs::write` of the whole entry, no fsync (H3, Wan).
    Write,
    /// Create, write the body, write the seal, `sync_all` (LTX).
    SyncAll,
}

/// What happens to the temporary file when writing or renaming fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TempCleanup {
    /// Removed (errors ignored) (H3, Wan).
    Remove,
    /// Left in place; it never matches an entry name, so it is never read
    /// (LTX, as before).
    Leave,
}

/// Whether a hit may carry non-finite values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finite {
    /// Any NaN or infinity makes the entry a miss (H3, Wan).
    Reject,
    /// Values are returned as stored (LTX).
    Accept,
}

/// How an I/O failure while storing is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorText {
    /// [`PromptCacheError::Io`] naming the directory (creating it) or the
    /// entry path (writing, renaming) (H3, Wan).
    EntryPath,
    /// [`PromptCacheError::CacheDir`] naming the cache directory (LTX).
    CacheDir,
}

/// The log line of an entry that could not be stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreLog {
    /// `<prefix>: not written (<error>)` (H3 and Wan: `h3 text cache`, the
    /// text Wan printed through H3's store).
    NotWritten(&'static str),
    /// `<prefix>: not stored: <error>` (LTX: `ltx2 text cache`).
    NotStored(&'static str),
}

impl StoreLog {
    pub fn line(self, error: &PromptCacheError) -> String {
        match self {
            Self::NotWritten(prefix) => format!("{prefix}: not written ({error})"),
            Self::NotStored(prefix) => format!("{prefix}: not stored: {error}"),
        }
    }
}

/// One family's entry file: every way the families' files differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    /// For messages.
    pub name: &'static str,
    pub magic: [u8; 8],
    pub extension: &'static str,
    pub layout: Layout,
    pub temp: TempName,
    pub durability: Durability,
    pub cleanup: TempCleanup,
    pub finite: Finite,
    pub errors: ErrorText,
    pub store_log: StoreLog,
}

/// H3 conditioning, `<key>.h3text` (pre-B4c `h3/text_cache.rs:35-278`).
/// Wan's UMT5 conditioning uses it too, as it did through H3's store.
pub const H3: Format = Format {
    name: "h3text",
    magic: H3_MAGIC,
    extension: "h3text",
    layout: Layout::TokensWidth,
    temp: TempName::ReplaceExtension,
    durability: Durability::Write,
    cleanup: TempCleanup::Remove,
    finite: Finite::Reject,
    errors: ErrorText::EntryPath,
    store_log: StoreLog::NotWritten("h3 text cache"),
};

/// LTX-2 connector contexts, `<key>.ltx2text` (pre-B4c
/// `ltx2/text_cache.rs:36, 152-255`).
pub const LTX2: Format = Format {
    name: "ltx2text",
    magic: *b"FVLTX2T1",
    extension: "ltx2text",
    layout: Layout::LenRealShaped {
        tensors: 2,
        max_rank: 8,
    },
    temp: TempName::PidPart,
    durability: Durability::SyncAll,
    cleanup: TempCleanup::Leave,
    finite: Finite::Accept,
    errors: ErrorText::CacheDir,
    store_log: StoreLog::NotStored("ltx2 text cache"),
};

/// A host tensor read back from an entry.
#[derive(Debug, Clone, PartialEq)]
pub struct HostTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// A tensor to store. For [`Layout::TokensWidth`] the shape is not written
/// (the header is); only `data` counts.
#[derive(Debug, Clone, Copy)]
pub struct TensorRef<'a> {
    pub shape: &'a [usize],
    pub data: &'a [f32],
}

/// One entry as read back.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub ids: Vec<u32>,
    pub header: Header,
    pub tensors: Vec<HostTensor>,
}

/// One entry to write.
#[derive(Debug, Clone, Copy)]
pub struct EntryRef<'a> {
    pub ids: &'a [u32],
    pub header: Header,
    pub tensors: &'a [TensorRef<'a>],
}

/// What a hit must match: the token ids the entry was computed from and the
/// header word. A miss computes from exactly these and stores them.
#[derive(Debug, Clone, Copy)]
pub struct Check<'a> {
    pub ids: &'a [u32],
    pub header: Header,
}

/// A family's cached value: how it is rebuilt from a hit and how its
/// tensors are stored after a miss.
pub trait Cached: Sized {
    /// The value of a hit; `None` turns the hit into a miss.
    fn from_entry(entry: Entry) -> Option<Self>;
    /// Calls `f` with the tensors to store (read back to the host if need be).
    fn with_tensors<R>(&self, f: impl FnOnce(&[TensorRef<'_>]) -> R)
        -> Result<R, PromptCacheError>;
}

fn u32_le(s: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(s.try_into().ok()?))
}

fn u64_le(s: &[u8]) -> Option<u64> {
    Some(u64::from_le_bytes(s.try_into().ok()?))
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

impl Format {
    fn layout_error(&self, detail: impl Into<String>) -> PromptCacheError {
        PromptCacheError::Layout {
            format: self.name,
            detail: detail.into(),
        }
    }

    /// The whole file: `magic | header | ids | tensors | seal`.
    pub fn encode(&self, entry: &EntryRef<'_>) -> Result<Vec<u8>, PromptCacheError> {
        let mut bytes = Vec::new();
        match (self.layout, entry.header) {
            (Layout::TokensWidth, Header::Width(width)) => {
                let [tensor] = entry.tensors else {
                    return Err(self.layout_error(format!(
                        "{} tensors, the layout holds one",
                        entry.tensors.len()
                    )));
                };
                let tokens = entry.ids.len();
                if tokens == 0 || tokens.checked_mul(width) != Some(tensor.data.len()) {
                    return Err(PromptCacheError::Shape {
                        values: tensor.data.len(),
                        tokens,
                        width,
                    });
                }
                bytes.reserve(24 + tokens * 4 + tensor.data.len() * 4 + SEAL);
                bytes.extend_from_slice(&self.magic);
                bytes.extend_from_slice(&(tokens as u64).to_le_bytes());
                bytes.extend_from_slice(&(width as u64).to_le_bytes());
                bytes.extend(entry.ids.iter().flat_map(|v| v.to_le_bytes()));
                bytes.extend(tensor.data.iter().flat_map(|v| v.to_le_bytes()));
            }
            (Layout::LenRealShaped { tensors, .. }, Header::Real(real)) => {
                if entry.tensors.len() != tensors {
                    return Err(self.layout_error(format!(
                        "{} tensors, the layout holds {tensors}",
                        entry.tensors.len()
                    )));
                }
                for t in entry.tensors {
                    let n = t.shape.iter().try_fold(1usize, |a, d| a.checked_mul(*d));
                    if n != Some(t.data.len()) {
                        return Err(self.layout_error(format!(
                            "shape {:?} for {} values",
                            t.shape,
                            t.data.len()
                        )));
                    }
                }
                bytes.extend_from_slice(&self.magic);
                bytes.extend_from_slice(&(entry.ids.len() as u32).to_le_bytes());
                bytes.extend_from_slice(&(real as u32).to_le_bytes());
                for id in entry.ids {
                    bytes.extend_from_slice(&id.to_le_bytes());
                }
                for t in entry.tensors {
                    bytes.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
                    for d in t.shape {
                        bytes.extend_from_slice(&(*d as u64).to_le_bytes());
                    }
                    for v in t.data {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            (layout, header) => {
                return Err(self.layout_error(format!("{header:?} header for {layout:?}")));
            }
        }
        let seal = sha256(&bytes);
        bytes.extend_from_slice(&seal);
        Ok(bytes)
    }

    /// The entry in `bytes`, if the seal, the magic and the layout all hold.
    /// Never panics, never allocates more than the file's own size.
    pub fn decode(&self, bytes: &[u8]) -> Option<Entry> {
        let (body, seal) = bytes.split_at_checked(bytes.len().checked_sub(SEAL)?)?;
        if body.get(..8)? != self.magic || sha256(body) != seal {
            return None;
        }
        match self.layout {
            Layout::TokensWidth => {
                if body.len() < 24 {
                    return None;
                }
                let word = |at: usize| usize::try_from(u64_le(body.get(at..at + 8)?)?).ok();
                let (tokens, width) = (word(8)?, word(16)?);
                let ids_end = 24usize.checked_add(tokens.checked_mul(4)?)?;
                if tokens == 0
                    || width == 0
                    || body.len()
                        != ids_end.checked_add(tokens.checked_mul(width)?.checked_mul(4)?)?
                {
                    return None;
                }
                Some(Entry {
                    ids: u32s(&body[24..ids_end]),
                    header: Header::Width(width),
                    tensors: vec![HostTensor {
                        shape: vec![tokens, width],
                        data: f32s(&body[ids_end..]),
                    }],
                })
            }
            Layout::LenRealShaped { tensors, max_rank } => {
                let mut at = 8usize;
                let mut take = |n: usize| -> Option<&[u8]> {
                    let s = body.get(at..at.checked_add(n)?)?;
                    at += n;
                    Some(s)
                };
                let len = u32_le(take(4)?)? as usize;
                let real = u32_le(take(4)?)? as usize;
                let ids = u32s(take(len.checked_mul(4)?)?);
                let mut out = Vec::with_capacity(tensors);
                for _ in 0..tensors {
                    let rank = u32_le(take(4)?)? as usize;
                    if rank > max_rank {
                        return None;
                    }
                    let shape: Vec<usize> = take(rank * 8)?
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|c| u64::from_le_bytes(*c) as usize)
                        .collect();
                    let n = shape.iter().try_fold(1usize, |a, d| a.checked_mul(*d))?;
                    let data = f32s(take(n.checked_mul(4)?)?);
                    out.push(HostTensor { shape, data });
                }
                (at == body.len()).then_some(Entry {
                    ids,
                    header: Header::Real(real),
                    tensors: out,
                })
            }
        }
    }
}

/// One cache directory read and written in one [`Format`].
#[derive(Debug, Clone)]
pub struct PromptCache {
    dir: PathBuf,
    format: &'static Format,
}

impl PromptCache {
    pub fn new(dir: impl Into<PathBuf>, format: &'static Format) -> Self {
        Self {
            dir: dir.into(),
            format,
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn format(&self) -> &'static Format {
        self.format
    }

    /// `<dir>/<name>.<extension>`; `name` is the key in hex.
    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{name}.{}", self.format.extension))
    }

    /// The entry under `name`, if it exists, is intact, and was computed from
    /// exactly `check`. Anything else is a miss.
    pub fn load(&self, name: &str, check: &Check<'_>) -> Option<Entry> {
        let entry = self.format.decode(&std::fs::read(self.path(name)).ok()?)?;
        let finite = match self.format.finite {
            Finite::Reject => entry
                .tensors
                .iter()
                .all(|t| t.data.iter().all(|v| v.is_finite())),
            Finite::Accept => true,
        };
        (entry.ids == check.ids && entry.header == check.header && finite).then_some(entry)
    }

    /// Write atomically: a temporary name, then a rename, so a reader never
    /// sees half an entry under the final name. Returns the entry's path.
    pub fn store(&self, name: &str, entry: &EntryRef<'_>) -> Result<PathBuf, PromptCacheError> {
        self.store_as(name, entry, std::process::id())
    }

    /// [`Self::store`] as writer `writer` (the process id; tests stand in
    /// other processes with other ids).
    fn store_as(
        &self,
        name: &str,
        entry: &EntryRef<'_>,
        writer: u32,
    ) -> Result<PathBuf, PromptCacheError> {
        let bytes = self.format.encode(entry)?;
        let path = self.path(name);
        let err = |at: &Path, source: std::io::Error| match self.format.errors {
            ErrorText::EntryPath => PromptCacheError::Io {
                path: at.to_path_buf(),
                source,
            },
            ErrorText::CacheDir => PromptCacheError::CacheDir {
                dir: self.dir.clone(),
                source,
            },
        };
        std::fs::create_dir_all(&self.dir).map_err(|e| err(&self.dir, e))?;
        let tmp = match self.format.temp {
            TempName::ReplaceExtension => path.with_extension(format!("tmp{writer}")),
            TempName::PidPart => self.dir.join(format!("{name}.{writer}.part")),
        };
        let written = match self.format.durability {
            Durability::Write => std::fs::write(&tmp, &bytes),
            Durability::SyncAll => {
                let (body, seal) = bytes.split_at(bytes.len() - SEAL);
                std::fs::File::create(&tmp).and_then(|mut f| {
                    f.write_all(body)
                        .and_then(|()| f.write_all(seal))
                        .and_then(|()| f.sync_all())
                })
            }
        };
        written
            .and_then(|()| std::fs::rename(&tmp, &path))
            .map_err(|e| {
                if self.format.cleanup == TempCleanup::Remove {
                    let _ = std::fs::remove_file(&tmp);
                }
                err(&path, e)
            })?;
        Ok(path)
    }

    /// Load, else `compute` and store. A hit never runs `compute`; a store
    /// that fails is logged ([`Format::store_log`]) and the computed value
    /// returned. `compute`'s error is the call's error.
    pub fn get_or_compute<T: Cached, E>(
        &self,
        name: &str,
        check: &Check<'_>,
        compute: impl FnOnce() -> Result<T, E>,
    ) -> Result<(T, Outcome), E> {
        if let Some(hit) = self.load(name, check).and_then(T::from_entry) {
            return Ok((hit, Outcome::Hit));
        }
        let fresh = compute()?;
        let stored = fresh
            .with_tensors(|tensors| {
                self.store(
                    name,
                    &EntryRef {
                        ids: check.ids,
                        header: check.header,
                        tensors,
                    },
                )
            })
            .and_then(|r| r);
        if let Err(e) = stored {
            crate::rt::log::info(format_args!("{}", self.format.store_log.line(&e)));
        }
        Ok((fresh, Outcome::Miss))
    }
}
