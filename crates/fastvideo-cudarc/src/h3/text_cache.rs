//! A disk cache for H3 prompt conditioning.
//!
//! Encoding a new prompt costs ~10 s, and none of it is compute: a 35-token
//! forward is milliseconds, the rest is 50 GB of bf16 weights streaming through
//! host-to-device copies. The result is `[1, S, 5120]` float32 — 700 KB for that
//! prompt — and a pure function of four things, which are exactly what the key
//! hashes:
//!
//! * the prompt bytes;
//! * `tokenizer.json` (a different vocabulary is a different function);
//! * the tap index;
//! * an identity of the encoder weights that is cheap to compute: for every
//!   tensor the tap reads, its name, dtype and shape, plus a few kilobytes of
//!   actual weight bytes so a fine-tune with identical headers is still a
//!   different encoder. It is built from tensors, not files, so the original
//!   14-shard layout and the slim re-pack ([`super::slim`]) share entries.
//!
//! A cache must never be the reason a run fails or lies. A file that is
//! truncated, bit-flipped, from another format version or that disagrees with
//! the token ids of the prompt is a **miss**, not an error; a cache that cannot
//! be written costs the next run time, not this run its result.

use std::path::{Path, PathBuf};

use fastvideo_loader::LazyStore;
use sha2::{Digest, Sha256};

use crate::llm::DecoderConfig;
use crate::wan::tensor::{Result, TensorError};

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

// ---------------------------------------------------------------------------
// Key
// ---------------------------------------------------------------------------

/// Tensor names tap `tap` reads, in load order: the embedding, then each
/// layer's tensors in the order [`crate::llm`] asks for them. Also the content
/// list of the slim checkpoint.
pub fn tap_keys(cfg: &DecoderConfig, tap: usize) -> Vec<String> {
    let mut keys = vec![cfg.embed_key.clone()];
    for layer in 0..tap {
        let p = format!("{}.{layer}", cfg.layer_prefix);
        let mut names = Vec::new();
        if cfg.sandwich_norms {
            names.extend(["post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"]);
        } else {
            names.push("post_attention_layernorm.weight");
        }
        names.extend(["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight"]);
        names.extend(["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]);
        if cfg.qk_norm {
            names.extend(["self_attn.q_norm.weight", "self_attn.k_norm.weight"]);
        }
        names.push("input_layernorm.weight");
        keys.extend(names.into_iter().map(|n| format!("{p}.{n}")));
    }
    keys
}

/// How many leading bytes of a tensor go into the identity.
const SAMPLE_BYTES: usize = 4096;

/// A fingerprint of the encoder weights tap `tap` depends on: every such
/// tensor's name, dtype and shape, plus the first bytes of the embedding and of
/// the first and last layer's input norm. Reads a few kilobytes; independent of
/// how the tensors are spread over files.
pub fn encoder_identity(store: &LazyStore, cfg: &DecoderConfig, tap: usize) -> Result<[u8; 32]> {
    let mut h = Fields::new();
    for key in tap_keys(cfg, tap) {
        let view = store.view(&key).map_err(|e| msg(e.to_string()))?;
        h.field(key.as_bytes());
        h.field(view.dtype.as_str().as_bytes());
        h.field(&view.shape.iter().flat_map(|d| (*d as u64).to_le_bytes()).collect::<Vec<u8>>());
    }
    let last = tap.saturating_sub(1);
    for key in [cfg.embed_key.clone(), format!("{}.0.input_layernorm.weight", cfg.layer_prefix), format!("{}.{last}.input_layernorm.weight", cfg.layer_prefix)] {
        let view = store.view(&key).map_err(|e| msg(e.to_string()))?;
        h.field(&view.bytes[..SAMPLE_BYTES.min(view.bytes.len())]);
    }
    Ok(h.finish())
}

/// The cache key. Every input is length-prefixed; the magic versions the scheme.
pub fn cache_key(prompt: &str, tokenizer_sha256: &[u8; 32], tap: usize, encoder_identity: &[u8; 32]) -> [u8; 32] {
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
    let word = |at: usize| usize::try_from(u64::from_le_bytes(body[at..at + 8].try_into().ok()?)).ok();
    let (tokens, width) = (word(8)?, word(16)?);
    let ids_end = 24usize.checked_add(tokens.checked_mul(4)?)?;
    if tokens == 0 || width == 0 || body.len() != ids_end.checked_add(tokens.checked_mul(width)?.checked_mul(4)?)? {
        return None;
    }
    let ids = body[24..ids_end].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let data = body[ids_end..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Some(CachedText { ids, width, data })
}

/// The entry for `key`, if one exists, is intact, and was computed from
/// exactly `ids`. Anything else is a miss.
pub fn load(dir: &Path, key: &[u8; 32], ids: &[u32], width: usize) -> Option<CachedText> {
    let entry = deserialize(&std::fs::read(entry_path(dir, key)).ok()?)?;
    (entry.ids == ids && entry.width == width && entry.data.iter().all(|v| v.is_finite())).then_some(entry)
}

/// Write atomically (temp file + rename), so a run killed mid-write leaves no
/// half entry under the final name. Returns the error as text for a log line:
/// failing to cache is never fatal.
pub fn store(dir: &Path, key: &[u8; 32], entry: &CachedText) -> std::result::Result<PathBuf, String> {
    if entry.ids.is_empty() || entry.data.len() != entry.ids.len() * entry.width {
        return Err(format!("{} values for {} tokens of width {}", entry.data.len(), entry.ids.len(), entry.width));
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = entry_path(dir, key);
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, serialize(entry)).and_then(|()| std::fs::rename(&tmp, &path)).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", path.display())
    })?;
    Ok(path)
}

/// `load`, else `compute` and `store`. Returns the entry and whether it was a hit.
pub fn get_or_compute(dir: &Path, key: &[u8; 32], ids: &[u32], width: usize, compute: impl FnOnce() -> Result<Vec<f32>>) -> Result<(CachedText, bool)> {
    if let Some(hit) = load(dir, key, ids, width) {
        return Ok((hit, true));
    }
    let entry = CachedText { ids: ids.to_vec(), width, data: compute()? };
    if entry.data.len() != ids.len() * width {
        return Err(msg(format!("text encoder produced {} values for {} tokens of width {width}", entry.data.len(), ids.len())));
    }
    if let Err(e) = store(dir, key, &entry) {
        crate::wan::log::info(format_args!("h3 text cache: not written ({e})"));
    }
    Ok((entry, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fv-h3-text-cache-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn the_key_is_stable_and_every_input_moves_it() {
        let (tok, enc) = ([1u8; 32], [2u8; 32]);
        let key = cache_key("a dog <d>hi</d>", &tok, 50, &enc);
        // Pinned: changing the derivation silently would orphan every cache on disk.
        assert_eq!(hex(&key), hex(&cache_key("a dog <d>hi</d>", &tok, 50, &enc)));
        assert_eq!(hex(&key), "86ae3a4c85ef1725102bce88852a0d444061550701b8b40f29a6e3cca11652b5", "update this constant together with MAGIC");
        for other in [cache_key("a dog <d>hi</d> ", &tok, 50, &enc), cache_key("a dog <d>hi</d>", &[3; 32], 50, &enc), cache_key("a dog <d>hi</d>", &tok, 49, &enc), cache_key("a dog <d>hi</d>", &tok, 50, &[4; 32])] {
            assert_ne!(other, key);
        }
    }

    #[test]
    fn a_second_request_is_a_hit_and_never_recomputes() {
        let dir = temp_dir("hit");
        let key = cache_key("p", &[0; 32], 50, &[0; 32]);
        let ids = [7u32, 8, 9];
        let mut calls = 0;
        let mut run = |ids: &[u32]| {
            get_or_compute(&dir, &key, ids, 2, || {
                calls += 1;
                Ok(vec![0.5, -1.0, 2.0, 3.5, f32::MIN_POSITIVE, 1e30])
            })
            .unwrap()
        };
        let (first, hit) = run(&ids);
        assert!(!hit);
        let (second, hit) = run(&ids);
        assert!(hit);
        assert_eq!(first, second, "bit-exact through the file");
        // Same key but other token ids (a tokenizer that changed under the same hash cannot
        // happen; a colliding or stale file can): a miss that recomputes and replaces.
        let (_, hit) = run(&[7, 8, 10]);
        assert!(!hit);
        assert_eq!(calls, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_damaged_entry_is_a_miss_not_an_error() {
        let dir = temp_dir("damage");
        let key = [9u8; 32];
        let entry = CachedText { ids: vec![1, 2], width: 3, data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
        let path = store(&dir, &key, &entry).unwrap();
        assert_eq!(load(&dir, &key, &[1, 2], 3), Some(entry.clone()));
        let good = std::fs::read(&path).unwrap();

        for cut in [good.len() - 1, good.len() - 32, 30, 8, 0] {
            std::fs::write(&path, &good[..cut]).unwrap();
            assert_eq!(load(&dir, &key, &[1, 2], 3), None, "truncated to {cut} bytes");
        }
        let mut flipped = good.clone();
        flipped[30] ^= 0x10;
        std::fs::write(&path, &flipped).unwrap();
        assert_eq!(load(&dir, &key, &[1, 2], 3), None, "one flipped bit");
        let mut grown = good.clone();
        grown.extend_from_slice(&[0; 4]);
        std::fs::write(&path, &grown).unwrap();
        assert_eq!(load(&dir, &key, &[1, 2], 3), None, "trailing bytes");
        // A header that claims an absurd size must not allocate or overflow.
        let mut huge = good.clone();
        huge[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, &huge).unwrap();
        assert_eq!(load(&dir, &key, &[1, 2], 3), None);

        // ...and the next request simply recomputes over it.
        let (again, hit) = get_or_compute(&dir, &key, &[1, 2], 3, || Ok(entry.data.clone())).unwrap();
        assert!(!hit);
        assert_eq!(again, entry);
        assert_eq!(load(&dir, &key, &[1, 2], 3), Some(entry));
        assert_eq!(load(&dir, &[8u8; 32], &[1, 2], 3), None, "an absent file is a miss");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tap_keys_list_the_embedding_and_whole_layers_in_order() {
        let cfg = DecoderConfig::qwen3_vl_32b_text();
        let keys = tap_keys(&cfg, 50);
        assert_eq!(keys.len(), 1 + 50 * 11);
        assert_eq!(keys[0], "model.language_model.embed_tokens.weight");
        assert!(keys[1].starts_with("model.language_model.layers.0."));
        assert!(keys.last().unwrap().starts_with("model.language_model.layers.49."));
        assert!(keys.iter().all(|k| !k.contains("layers.50.") && !k.ends_with("language_model.norm.weight")));
    }
}
