//! H3's use of the disk prompt cache ([`crate::rt::prompt_cache`], ADR-0021):
//! its key inputs and its entry type. The cache itself — key derivation,
//! file format, atomic write, hit and miss — is the shared one, with the
//! [`prompt_cache::H3`] format and the [`prompt_cache::H3_KEY`] recipe.
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

use fastvideo_loader::LazyStore;

use crate::llm::DecoderConfig;
use crate::rt::prompt_cache::{self, Fields};
use crate::rt::tensor::{Result, TensorError};

pub use crate::rt::prompt_cache::{hex, sha256};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
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
            names.extend([
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
            ]);
        } else {
            names.push("post_attention_layernorm.weight");
        }
        names.extend([
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ]);
        names.extend([
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]);
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

/// The identity of the same encoder run with weight-only FP8 rows: a
/// different function of the same weights (`h3/text.rs`).
pub fn fp8_rows_identity(identity: &[u8; 32]) -> [u8; 32] {
    sha256(&[identity.as_slice(), b"/fp8-rows".as_slice()].concat())
}

/// The identity of the recovered-8B encoder: its kind and tap. Its weights
/// are not a `LazyStore` under `text_encoder/`, so the kind string stands in.
pub fn recovered_identity(tap: usize) -> [u8; 32] {
    sha256(format!("recovered-8b-tap{tap}").as_bytes())
}

/// The cache key ([`prompt_cache::H3_KEY`]): the format magic, the prompt,
/// the tokenizer digest, the tap and the encoder identity, each a
/// length-prefixed field.
pub fn cache_key(
    prompt: &str,
    tokenizer_sha256: &[u8; 32],
    tap: usize,
    encoder_identity: &[u8; 32],
) -> [u8; 32] {
    prompt_cache::H3_KEY.key(
        prompt,
        tokenizer_sha256,
        tap as u64,
        &[encoder_identity.as_slice()],
    )
}

// ---------------------------------------------------------------------------
// Entries: the shared `[tokens, width]` entry (H3 and Wan)
// ---------------------------------------------------------------------------

pub use crate::rt::prompt_cache::tokens_width::{
    entry_path, get_or_compute, load, store, CachedText,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("fv-h3-text-cache-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn the_key_is_stable_and_every_input_moves_it() {
        let (tok, enc) = ([1u8; 32], [2u8; 32]);
        let key = cache_key("a dog <d>hi</d>", &tok, 50, &enc);
        // Pinned: changing the derivation silently would orphan every cache on disk.
        assert_eq!(
            hex(&key),
            hex(&cache_key("a dog <d>hi</d>", &tok, 50, &enc))
        );
        assert_eq!(
            hex(&key),
            "86ae3a4c85ef1725102bce88852a0d444061550701b8b40f29a6e3cca11652b5",
            "update this constant together with MAGIC"
        );
        for other in [
            cache_key("a dog <d>hi</d> ", &tok, 50, &enc),
            cache_key("a dog <d>hi</d>", &[3; 32], 50, &enc),
            cache_key("a dog <d>hi</d>", &tok, 49, &enc),
            cache_key("a dog <d>hi</d>", &tok, 50, &[4; 32]),
        ] {
            assert_ne!(other, key);
        }
    }

    #[test]
    fn cache_key_ignores_recipe_and_follows_encoder_identity() {
        let tok = [9u8; 32];
        let enc_a = [1u8; 32];
        let enc_b = [2u8; 32];
        let k_sol = cache_key("same prompt", &tok, 50, &enc_a);
        let k_spark = cache_key("same prompt", &tok, 50, &enc_a);
        assert_eq!(
            k_sol, k_spark,
            "recipe is not part of the cache key; only encoder identity is"
        );
        assert_ne!(cache_key("same prompt", &tok, 50, &enc_b), k_sol);
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
        let entry = CachedText {
            ids: vec![1, 2],
            width: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let path = store(&dir, &key, &entry).unwrap();
        assert_eq!(load(&dir, &key, &[1, 2], 3), Some(entry.clone()));
        let good = std::fs::read(&path).unwrap();

        for cut in [good.len() - 1, good.len() - 32, 30, 8, 0] {
            std::fs::write(&path, &good[..cut]).unwrap();
            assert_eq!(
                load(&dir, &key, &[1, 2], 3),
                None,
                "truncated to {cut} bytes"
            );
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
        let (again, hit) =
            get_or_compute(&dir, &key, &[1, 2], 3, || Ok(entry.data.clone())).unwrap();
        assert!(!hit);
        assert_eq!(again, entry);
        assert_eq!(load(&dir, &key, &[1, 2], 3), Some(entry));
        assert_eq!(
            load(&dir, &[8u8; 32], &[1, 2], 3),
            None,
            "an absent file is a miss"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tap_keys_list_the_embedding_and_whole_layers_in_order() {
        let cfg = DecoderConfig::qwen3_vl_32b_text();
        let keys = tap_keys(&cfg, 50);
        assert_eq!(keys.len(), 1 + 50 * 11);
        assert_eq!(keys[0], "model.language_model.embed_tokens.weight");
        assert!(keys[1].starts_with("model.language_model.layers.0."));
        assert!(keys
            .last()
            .unwrap()
            .starts_with("model.language_model.layers.49."));
        assert!(keys
            .iter()
            .all(|k| !k.contains("layers.50.") && !k.ends_with("language_model.norm.weight")));
    }
}
