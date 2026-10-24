//! The three disk prompt caches as they were before B4c (main `e18ec010`),
//! frozen as the reference the equality fixtures (`equality.rs`) hold the
//! one implementation to. Test-only; not code to maintain (ADR-0021).
//!
//! Verbatim, except:
//! - imports and paths are made absolute (`crate::...`); `tap_keys` (not
//!   changed by B4c) is the live one;
//! - H3's `#[cfg(test)]` tests and LTX's `default_dir` (moved unchanged) are
//!   left out;
//! - Wan's `text_cache_key` was a method of `WanPipeline`: it is a free
//!   function of what it read from `self` and `cfg` (`text_dir`,
//!   `tokenizer_path`, `text_len`), with the per-pipeline `OnceLock` memo of
//!   the identity dropped (it memoised a pure function of the files); the
//!   process-wide tokenizer memo is kept, so fixtures use one tokenizer path
//!   per content. Its store was H3's `get_or_compute` (`encode_one`).
//! - LTX's protocol `cached_or` was in `ltx2/pipeline.rs:2664-2684` (private
//!   there, `pub` here);
//! - formatted by rustfmt as nested modules.
#![allow(dead_code, clippy::all)]

/// `crates/fastvideo-cudarc/src/h3/text_cache.rs:23-278` (minus `tap_keys`).
pub mod h3 {
    use std::path::{Path, PathBuf};

    use fastvideo_loader::LazyStore;
    use sha2::{Digest, Sha256};

    use crate::h3::text_cache::tap_keys;
    use crate::llm::DecoderConfig;
    use crate::rt::tensor::{Result, TensorError};

    fn msg(s: impl Into<String>) -> TensorError {
        TensorError::Message(s.into())
    }

    /// Bump when the file layout or the key derivation changes.
    const MAGIC: &[u8; 8] = b"H3TEXT01";

    /// SHA-256 with length-prefixed fields, so `("ab", "c")` and `("a", "bc")`
    /// hash differently. `sha2` rather than std's hasher: these digests are file
    /// names on disk and must not change with the Rust version.
    struct Fields(Sha256);

    impl Fields {
        fn new() -> Self {
            Self(Sha256::new())
        }

        fn field(&mut self, data: &[u8]) {
            self.0.update((data.len() as u64).to_le_bytes());
            self.0.update(data);
        }

        fn finish(self) -> [u8; 32] {
            self.0.finalize().into()
        }
    }

    pub fn sha256(data: &[u8]) -> [u8; 32] {
        Sha256::digest(data).into()
    }

    pub fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// How many leading bytes of a tensor go into the identity.
    const SAMPLE_BYTES: usize = 4096;

    /// A fingerprint of the encoder weights tap `tap` depends on: every such
    /// tensor's name, dtype and shape, plus the first bytes of the embedding and of
    /// the first and last layer's input norm. Reads a few kilobytes; independent of
    /// how the tensors are spread over files.
    pub fn encoder_identity(
        store: &LazyStore,
        cfg: &DecoderConfig,
        tap: usize,
    ) -> Result<[u8; 32]> {
        let mut h = Fields::new();
        for key in tap_keys(cfg, tap) {
            let view = store.view(&key).map_err(|e| msg(e.to_string()))?;
            h.field(key.as_bytes());
            h.field(view.dtype.as_str().as_bytes());
            h.field(
                &view
                    .shape
                    .iter()
                    .flat_map(|d| (*d as u64).to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
        }
        let last = tap.saturating_sub(1);
        for key in [
            cfg.embed_key.clone(),
            format!("{}.0.input_layernorm.weight", cfg.layer_prefix),
            format!("{}.{last}.input_layernorm.weight", cfg.layer_prefix),
        ] {
            let view = store.view(&key).map_err(|e| msg(e.to_string()))?;
            h.field(&view.bytes[..SAMPLE_BYTES.min(view.bytes.len())]);
        }
        Ok(h.finish())
    }

    /// The cache key. Every input is length-prefixed; the magic versions the scheme.
    pub fn cache_key(
        prompt: &str,
        tokenizer_sha256: &[u8; 32],
        tap: usize,
        encoder_identity: &[u8; 32],
    ) -> [u8; 32] {
        let mut h = Fields::new();
        h.field(MAGIC);
        h.field(prompt.as_bytes());
        h.field(tokenizer_sha256);
        h.field(&(tap as u64).to_le_bytes());
        h.field(encoder_identity);
        h.finish()
    }

    // ---------------------------------------------------------------------------
    // Entries
    // ---------------------------------------------------------------------------

    /// One cached conditioning: the token ids it was computed from and `[1, S, width]`.
    #[derive(Debug, Clone, PartialEq)]
    pub struct CachedText {
        pub ids: Vec<u32>,
        pub width: usize,
        pub data: Vec<f32>,
    }

    pub fn entry_path(dir: &Path, key: &[u8; 32]) -> PathBuf {
        dir.join(format!("{}.h3text", hex(key)))
    }

    /// `magic | tokens u64 | width u64 | ids u32[] | data f32[] | sha256 of all that`.
    fn serialize(entry: &CachedText) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(24 + entry.ids.len() * 4 + entry.data.len() * 4 + 32);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(entry.ids.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(entry.width as u64).to_le_bytes());
        bytes.extend(entry.ids.iter().flat_map(|v| v.to_le_bytes()));
        bytes.extend(entry.data.iter().flat_map(|v| v.to_le_bytes()));
        let digest = sha256(&bytes);
        bytes.extend_from_slice(&digest);
        bytes
    }

    fn deserialize(bytes: &[u8]) -> Option<CachedText> {
        let (body, digest) = bytes.split_at_checked(bytes.len().checked_sub(32)?)?;
        if body.len() < 24 || &body[..8] != MAGIC || sha256(body) != digest {
            return None;
        }
        let word =
            |at: usize| usize::try_from(u64::from_le_bytes(body[at..at + 8].try_into().ok()?)).ok();
        let (tokens, width) = (word(8)?, word(16)?);
        let ids_end = 24usize.checked_add(tokens.checked_mul(4)?)?;
        if tokens == 0
            || width == 0
            || body.len() != ids_end.checked_add(tokens.checked_mul(width)?.checked_mul(4)?)?
        {
            return None;
        }
        let ids = body[24..ids_end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let data = body[ids_end..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(CachedText { ids, width, data })
    }

    /// The entry for `key`, if one exists, is intact, and was computed from
    /// exactly `ids`. Anything else is a miss.
    pub fn load(dir: &Path, key: &[u8; 32], ids: &[u32], width: usize) -> Option<CachedText> {
        let entry = deserialize(&std::fs::read(entry_path(dir, key)).ok()?)?;
        (entry.ids == ids && entry.width == width && entry.data.iter().all(|v| v.is_finite()))
            .then_some(entry)
    }

    /// Write atomically (temp file + rename), so a run killed mid-write leaves no
    /// half entry under the final name. Returns the error as text for a log line:
    /// failing to cache is never fatal.
    pub fn store(
        dir: &Path,
        key: &[u8; 32],
        entry: &CachedText,
    ) -> std::result::Result<PathBuf, String> {
        if entry.ids.is_empty() || entry.data.len() != entry.ids.len() * entry.width {
            return Err(format!(
                "{} values for {} tokens of width {}",
                entry.data.len(),
                entry.ids.len(),
                entry.width
            ));
        }
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = entry_path(dir, key);
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, serialize(entry))
            .and_then(|()| std::fs::rename(&tmp, &path))
            .map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                format!("{}: {e}", path.display())
            })?;
        Ok(path)
    }

    /// `load`, else `compute` and `store`. Returns the entry and whether it was a hit.
    pub fn get_or_compute(
        dir: &Path,
        key: &[u8; 32],
        ids: &[u32],
        width: usize,
        compute: impl FnOnce() -> Result<Vec<f32>>,
    ) -> Result<(CachedText, bool)> {
        if let Some(hit) = load(dir, key, ids, width) {
            return Ok((hit, true));
        }
        let entry = CachedText {
            ids: ids.to_vec(),
            width,
            data: compute()?,
        };
        if entry.data.len() != ids.len() * width {
            return Err(msg(format!(
                "text encoder produced {} values for {} tokens of width {width}",
                entry.data.len(),
                ids.len()
            )));
        }
        if let Err(e) = store(dir, key, &entry) {
            crate::rt::log::info(format_args!("h3 text cache: not written ({e})"));
        }
        Ok((entry, false))
    }
}

/// `crates/fastvideo-cudarc/src/ltx2/text_cache.rs:26-134, 152-255` and
/// `cached_or` from `ltx2/pipeline.rs:2664-2684`.
pub mod ltx2 {
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};

    use crate::rt::tensor::{CudaTensor, Result};

    use sha2::{Digest, Sha256};

    use crate::ltx2::msg;
    use crate::ltx2::text::PaddedPrompt;

    const MAGIC: &[u8; 8] = b"FVLTX2T1";

    fn digest(data: &[u8]) -> [u8; 32] {
        Sha256::digest(data).into()
    }

    /// A length-prefixed field, so `("ab", "c")` and `("a", "bc")` hash differently.
    fn field(h: &mut Sha256, data: &[u8]) {
        h.update((data.len() as u64).to_le_bytes());
        h.update(data);
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// How much of each end of a weight file goes into its identity.
    const PROBE: u64 = 1 << 20;

    /// The two contexts of one prompt, as the connectors emit them.
    pub struct CachedContexts {
        pub video: CudaTensor,
        pub audio: CudaTensor,
    }

    /// Identity of a checkpoint — a file, or every `.safetensors` directly under a
    /// directory (optionally only those whose name starts with `prefix`) — from
    /// file names, sizes and the first and last mebibyte of each.
    pub fn weights_identity(path: &Path, prefix: Option<&str>) -> Result<[u8; 32]> {
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

    /// Everything the conditioning depends on, hashed.
    pub fn cache_key(
        prompt: &str,
        tokenizer_json: &[u8],
        max_len: usize,
        text_encoder_kind: &[u8],
        gemma: &[u8; 32],
        connectors: &[u8; 32],
    ) -> String {
        let mut h = Sha256::new();
        field(&mut h, b"fastvideo ltx2 text conditioning v1");
        // The pipeline strips the prompt before tokenising; so does the key.
        field(&mut h, prompt.trim().as_bytes());
        field(&mut h, &digest(tokenizer_json));
        field(&mut h, &(max_len as u64).to_le_bytes());
        field(&mut h, text_encoder_kind);
        field(&mut h, gemma);
        field(&mut h, connectors);
        hex(&h.finalize())
    }

    pub struct TextCache {
        dir: PathBuf,
    }

    impl TextCache {
        pub fn new(dir: impl Into<PathBuf>) -> Self {
            Self { dir: dir.into() }
        }

        pub fn path(&self, key: &str) -> PathBuf {
            self.dir.join(format!("{key}.ltx2text"))
        }

        /// Write atomically: a reader never sees half an entry under the final name.
        pub fn store(
            &self,
            key: &str,
            prompt: &PaddedPrompt,
            video: &CudaTensor,
            audio: &CudaTensor,
        ) -> Result<()> {
            let io = |e: std::io::Error| msg(format!("text cache {}: {e}", self.dir.display()));
            let mut body = Vec::new();
            body.extend_from_slice(MAGIC);
            body.extend_from_slice(&(prompt.ids.len() as u32).to_le_bytes());
            body.extend_from_slice(&(prompt.real as u32).to_le_bytes());
            for id in &prompt.ids {
                body.extend_from_slice(&id.to_le_bytes());
            }
            for t in [video, audio] {
                body.extend_from_slice(&(t.rank() as u32).to_le_bytes());
                for d in &t.shape {
                    body.extend_from_slice(&(*d as u64).to_le_bytes());
                }
                for v in t.host_cow()?.iter() {
                    body.extend_from_slice(&v.to_le_bytes());
                }
            }
            let seal = digest(&body);
            std::fs::create_dir_all(&self.dir).map_err(io)?;
            let part = self.dir.join(format!("{key}.{}.part", std::process::id()));
            let mut f = std::fs::File::create(&part).map_err(io)?;
            f.write_all(&body)
                .and_then(|()| f.write_all(&seal))
                .and_then(|()| f.sync_all())
                .map_err(io)?;
            drop(f);
            std::fs::rename(&part, self.path(key)).map_err(io)
        }

        /// The entry under `key`, if it is whole and was made from exactly these
        /// token ids. Anything else is a miss.
        pub fn load(&self, key: &str, prompt: &PaddedPrompt) -> Option<CachedContexts> {
            let bytes = std::fs::read(self.path(key)).ok()?;
            let (body, seal) = bytes.split_at_checked(bytes.len().checked_sub(32)?)?;
            if digest(body) != *seal {
                return None;
            }
            let mut at = 0usize;
            let mut take = |n: usize| -> Option<&[u8]> {
                let s = body.get(at..at.checked_add(n)?)?;
                at += n;
                Some(s)
            };
            let u32_at = |s: &[u8]| u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
            if take(8)? != MAGIC {
                return None;
            }
            let (len, real) = (u32_at(take(4)?) as usize, u32_at(take(4)?) as usize);
            let ids: Vec<u32> = take(len.checked_mul(4)?)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|s| u32::from_le_bytes(*s))
                .collect();
            if ids != prompt.ids || real != prompt.real {
                return None;
            }
            let mut tensor = || -> Option<CudaTensor> {
                let rank = u32_at(take(4)?) as usize;
                if rank > 8 {
                    return None;
                }
                let shape: Vec<usize> = take(rank * 8)?
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| {
                        u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                            as usize
                    })
                    .collect();
                let n = shape.iter().try_fold(1usize, |a, d| a.checked_mul(*d))?;
                let data: Vec<f32> = take(n.checked_mul(4)?)?
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                CudaTensor::from_vec(data, shape).ok()
            };
            let (video, audio) = (tensor()?, tensor()?);
            (at == body.len()).then_some(CachedContexts { video, audio })
        }
    }

    /// Whether the conditioning cache answered (`ltx2/pipeline.rs:2173-2190`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CacheOutcome {
        Off,
        Hit,
        Miss,
    }

    /// The cache protocol, apart from what it caches: look up, else compute and
    /// store. A cache that cannot be written is reported by the log, not by failing
    /// a generation whose conditioning is already in hand.
    pub fn cached_or(
        cache: Option<&TextCache>,
        key: Option<&str>,
        padded: &PaddedPrompt,
        compute: impl FnOnce() -> Result<CachedContexts>,
    ) -> Result<(CachedContexts, CacheOutcome)> {
        let (Some(cache), Some(key)) = (cache, key) else {
            return Ok((compute()?, CacheOutcome::Off));
        };
        if let Some(hit) = cache.load(key, padded) {
            return Ok((hit, CacheOutcome::Hit));
        }
        let fresh = compute()?;
        if let Err(e) = cache.store(key, padded, &fresh.video, &fresh.audio) {
            crate::rt::log::info(format_args!("ltx2 text cache: not stored: {e}"));
        }
        Ok((fresh, CacheOutcome::Miss))
    }
}

/// `crates/fastvideo-cudarc/src/wan/pipeline.rs:728-799` (`text_cache_key`)
/// as a free function, and `encode_one`'s store (`:700-726`, H3's
/// `get_or_compute` at width 4096).
pub mod wan {
    use std::path::Path;

    use crate::wan::pipeline::{PipelineError, Result};

    /// The disk-cache key of `prompt`: the text, `tokenizer.json`, the padded
    /// length and an identity of the UMT5 weights (file names, sizes and each
    /// file's first 256 KiB: the safetensors header and the start of the data).
    pub fn text_cache_key(
        text_dir: Option<&Path>,
        tokenizer_path: Option<String>,
        text_len: usize,
        prompt: &str,
    ) -> Result<[u8; 32]> {
        use sha2::{Digest, Sha256};
        let identity: [u8; 32] = {
            let mut h = Sha256::new();
            if let Some(dir) = &text_dir {
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
                        use std::io::Read as _;
                        let mut buf = vec![0u8; 256 << 10];
                        let n = file.read(&mut buf).unwrap_or(0);
                        h.update(&buf[..n]);
                    }
                }
            }
            h.finalize().into()
        };
        // tokenizer.json is ~16 MB: hash it once per path per process.
        static TOKENIZER_SHA: std::sync::OnceLock<
            std::sync::Mutex<std::collections::HashMap<String, [u8; 32]>>,
        > = std::sync::OnceLock::new();
        let path = tokenizer_path.clone().unwrap_or_default();
        let known = TOKENIZER_SHA
            .get_or_init(Default::default)
            .lock()
            .expect("tokenizer sha")
            .get(&path)
            .copied();
        let tokenizer_sha = match known {
            Some(s) => s,
            None => {
                let bytes = if path.is_empty() {
                    Vec::new()
                } else {
                    std::fs::read(&path)
                        .map_err(|e| PipelineError::Message(format!("tokenizer: {e}")))?
                };
                let s = super::h3::sha256(&bytes);
                TOKENIZER_SHA
                    .get_or_init(Default::default)
                    .lock()
                    .expect("tokenizer sha")
                    .insert(path, s);
                s
            }
        };
        let mut h = Sha256::new();
        for field in [
            b"WANTEXT1".as_slice(),
            prompt.as_bytes(),
            tokenizer_sha.as_slice(),
            (text_len as u64).to_le_bytes().as_slice(),
            identity.as_slice(),
        ] {
            h.update((field.len() as u64).to_le_bytes());
            h.update(field);
        }
        Ok(h.finalize().into())
    }
}
