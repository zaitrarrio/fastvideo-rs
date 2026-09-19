//! A cache of finished text conditioning, because the text encoder is the most
//! expensive thing a repeated prompt pays for and the cheapest to skip.
//!
//! Encoding a prompt reads 47 GB of Gemma shards and uploads 23.5 GB to produce
//! 31 MB: the two `[1, 1024, 3840]` connector outputs. Those depend on nothing
//! but the prompt, the tokenizer, the padded length and the Gemma + connector
//! weights — not on the seed, the resolution, the DiT or the step count — so
//! they are stored under a key made of exactly those things. What is cached is
//! the *connector* output, before the DiT's caption projections: that keeps the
//! entry independent of the DiT checkpoint, and the projections are two small
//! GEMMs.
//!
//! The key must be computable **without opening the weights**, or a hit would
//! still pay for mapping 12 files and parsing their headers. A checkpoint is
//! therefore identified by its files: name, size, and a hash of the first and
//! last mebibyte of each — the head is the safetensors header (every key, dtype,
//! shape and offset), the tail is real tensor bytes, so a fine-tune with the
//! same layout still keys differently. No timestamps: a re-download of the same
//! files keeps its cache.
//!
//! An entry carries the padded token ids it was made from and ends in a hash of
//! its own bytes. A hit re-tokenises the prompt and compares; a short, corrupt
//! or foreign file is a miss, never an error — the cache can only make a run
//! faster, not make it fail.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::wan::tensor::{CudaTensor, Result};

use sha2::{Digest, Sha256};

use super::msg;
use super::text::PaddedPrompt;

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
            .filter(|p| prefix.is_none_or(|pre| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(pre))))
            .collect()
    };
    if files.is_empty() {
        return Err(msg(format!("{}: no safetensors files to identify", path.display())));
    }
    files.sort();
    let mut h = Sha256::new();
    for file in &files {
        let mut f = std::fs::File::open(file).map_err(|e| io(e, file))?;
        let size = f.metadata().map_err(|e| io(e, file))?.len();
        field(&mut h, file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default().as_bytes());
        field(&mut h, &size.to_le_bytes());
        let mut buf = vec![0u8; PROBE.min(size) as usize];
        f.read_exact(&mut buf).map_err(|e| io(e, file))?;
        field(&mut h, &buf);
        if size > PROBE {
            let tail = PROBE.min(size - PROBE);
            f.seek(SeekFrom::End(-(tail as i64))).map_err(|e| io(e, file))?;
            buf.truncate(tail as usize);
            f.read_exact(&mut buf).map_err(|e| io(e, file))?;
            field(&mut h, &buf);
        }
    }
    Ok(h.finalize().into())
}

/// Everything the conditioning depends on, hashed.
pub fn cache_key(prompt: &str, tokenizer_json: &[u8], max_len: usize, gemma: &[u8; 32], connectors: &[u8; 32]) -> String {
    let mut h = Sha256::new();
    field(&mut h, b"fastvideo ltx2 text conditioning v1");
    // The pipeline strips the prompt before tokenising; so does the key.
    field(&mut h, prompt.trim().as_bytes());
    field(&mut h, &digest(tokenizer_json));
    field(&mut h, &(max_len as u64).to_le_bytes());
    field(&mut h, gemma);
    field(&mut h, connectors);
    hex(&h.finalize())
}

/// `$FASTVIDEO_CACHE`, else `$XDG_CACHE_HOME/fastvideo`, else `~/.cache/fastvideo`,
/// under `ltx2-text`. `None` when the environment names no home at all.
pub fn default_dir() -> Option<PathBuf> {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let root = var("FASTVIDEO_CACHE").or_else(|| var("XDG_CACHE_HOME").map(|p| p.join("fastvideo"))).or_else(|| var("HOME").map(|p| p.join(".cache").join("fastvideo")))?;
    Some(root.join("ltx2-text"))
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
    pub fn store(&self, key: &str, prompt: &PaddedPrompt, video: &CudaTensor, audio: &CudaTensor) -> Result<()> {
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
        f.write_all(&body).and_then(|()| f.write_all(&seal)).and_then(|()| f.sync_all()).map_err(io)?;
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
        let ids: Vec<u32> = take(len.checked_mul(4)?)?.chunks_exact(4).map(u32_at).collect();
        if ids != prompt.ids || real != prompt.real {
            return None;
        }
        let mut tensor = || -> Option<CudaTensor> {
            let rank = u32_at(take(4)?) as usize;
            if rank > 8 {
                return None;
            }
            let shape: Vec<usize> = take(rank * 8)?.chunks_exact(8).map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as usize).collect();
            let n = shape.iter().try_fold(1usize, |a, d| a.checked_mul(*d))?;
            let data: Vec<f32> = take(n.checked_mul(4)?)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            CudaTensor::from_vec(data, shape).ok()
        };
        let (video, audio) = (tensor()?, tensor()?);
        (at == body.len()).then_some(CachedContexts { video, audio })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fv-ltx2-textcache-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn contexts() -> (PaddedPrompt, CudaTensor, CudaTensor) {
        let prompt = PaddedPrompt::from_ids(&[2, 17, 99], 8).unwrap();
        let video = CudaTensor::from_vec((0..24).map(|i| i as f32 * 0.5 - 3.0).collect(), vec![1, 8, 3]).unwrap();
        let audio = CudaTensor::from_vec((0..16).map(|i| (i as f32).sin()).collect(), vec![1, 8, 2]).unwrap();
        (prompt, video, audio)
    }

    #[test]
    fn the_key_is_stable_and_moves_with_every_input() {
        let (g, c) = ([1u8; 32], [2u8; 32]);
        let key = cache_key("a red fox", b"tokenizer", 1024, &g, &c);
        assert_eq!(key.len(), 64);
        // Pinned: a changed key format silently orphans every cache on disk.
        assert_eq!(key, "2f4bbf8d7094ee3ca1f6e7144c94ec227c84b98444bfaeace4d64076982c1445");
        assert_eq!(key, cache_key("  a red fox\n", b"tokenizer", 1024, &g, &c), "the prompt is stripped, as the pipeline strips it");
        for other in [
            cache_key("a red fox.", b"tokenizer", 1024, &g, &c),
            cache_key("a red fox", b"tokenizer2", 1024, &g, &c),
            cache_key("a red fox", b"tokenizer", 512, &g, &c),
            cache_key("a red fox", b"tokenizer", 1024, &c, &c),
            cache_key("a red fox", b"tokenizer", 1024, &g, &g),
        ] {
            assert_ne!(key, other);
        }
    }

    #[test]
    fn a_stored_entry_comes_back_bit_for_bit() {
        let cache = TextCache::new(scratch("hit"));
        let (prompt, video, audio) = contexts();
        assert!(cache.load("k", &prompt).is_none(), "an empty cache misses");
        cache.store("k", &prompt, &video, &audio).unwrap();
        let hit = cache.load("k", &prompt).expect("hit");
        assert_eq!(hit.video.shape, vec![1, 8, 3]);
        assert_eq!(&*hit.video.host_cow().unwrap(), &*video.host_cow().unwrap());
        assert_eq!(&*hit.audio.host_cow().unwrap(), &*audio.host_cow().unwrap());
        assert!(cache.load("another-key", &prompt).is_none());
        // No .part files are left behind.
        let names: Vec<_> = std::fs::read_dir(cache.path("k").parent().unwrap()).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(names.len(), 1, "{names:?}");
    }

    #[test]
    fn an_entry_made_from_other_token_ids_is_a_miss() {
        let cache = TextCache::new(scratch("ids"));
        let (prompt, video, audio) = contexts();
        cache.store("k", &prompt, &video, &audio).unwrap();
        let other = PaddedPrompt::from_ids(&[2, 17, 100], 8).unwrap();
        assert!(cache.load("k", &other).is_none());
        let longer = PaddedPrompt::from_ids(&[2, 17, 99], 16).unwrap();
        assert!(cache.load("k", &longer).is_none());
    }

    #[test]
    fn a_truncated_or_corrupted_entry_is_a_miss_not_an_error() {
        let cache = TextCache::new(scratch("trunc"));
        let (prompt, video, audio) = contexts();
        cache.store("k", &prompt, &video, &audio).unwrap();
        let whole = std::fs::read(cache.path("k")).unwrap();
        for cut in [0, 7, 40, whole.len() / 2, whole.len() - 1] {
            std::fs::write(cache.path("k"), &whole[..cut]).unwrap();
            assert!(cache.load("k", &prompt).is_none(), "{cut} of {} bytes", whole.len());
        }
        let mut flipped = whole.clone();
        flipped[whole.len() / 2] ^= 1;
        std::fs::write(cache.path("k"), &flipped).unwrap();
        assert!(cache.load("k", &prompt).is_none(), "one flipped bit");
        std::fs::write(cache.path("k"), &whole).unwrap();
        assert!(cache.load("k", &prompt).is_some(), "the intact entry still hits");
    }

    #[test]
    fn a_checkpoint_is_identified_by_its_files_without_being_parsed() {
        let dir = scratch("ident");
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), vec![7u8; 3000]).unwrap();
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), vec![9u8; 100]).unwrap();
        std::fs::write(dir.join("diffusion_pytorch_model-00001-of-00012.safetensors"), vec![1u8; 50]).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        let id = weights_identity(&dir, Some("model-")).unwrap();
        assert_eq!(id, weights_identity(&dir, Some("model-")).unwrap());
        // The stale duplicate set is outside the prefix; touching it changes nothing.
        std::fs::write(dir.join("diffusion_pytorch_model-00001-of-00012.safetensors"), vec![2u8; 60]).unwrap();
        assert_eq!(id, weights_identity(&dir, Some("model-")).unwrap());
        assert_ne!(id, weights_identity(&dir, None).unwrap());
        // Same size, different bytes: a fine-tune with the same layout.
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), vec![8u8; 100]).unwrap();
        assert_ne!(id, weights_identity(&dir, Some("model-")).unwrap());
        assert!(weights_identity(&dir, Some("nothing-")).is_err());
        // A single file identifies itself.
        assert!(weights_identity(&dir.join("model-00001-of-00002.safetensors"), None).is_ok());
    }
}
