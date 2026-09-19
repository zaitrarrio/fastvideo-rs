//! A slim re-pack of the H3 text encoder: exactly what tap 50 reads, in the
//! order it is read.
//!
//! The published `text_encoder/` is a whole Qwen3-VL-32B: 66.7 GB in 14 shards
//! with a vision tower, an `lm_head`, a final norm and 14 decoder layers H3
//! never runs, filled in *lexicographic* key order (`layers.10` before
//! `layers.2`), so streaming layers 0..49 hops between shards. The slim copy
//! holds `embed_tokens` and layers 0..=49 only — 50.3 GB, bf16 bytes verbatim —
//! with tensors laid out in the loader's own request order and cut into shards
//! whose names sort in that order, so a cold encode is one sequential read.
//!
//! [`super::text`] needs no switch for it: tensors are resolved by name, so
//! `<slim root>/text_encoder` + `<slim root>/tokenizer` is just another root.
//! The content list is [`super::text_cache::tap_keys`], which a test holds
//! equal to what [`crate::llm`] actually asks for, in order.

use std::path::{Path, PathBuf};

use fastvideo_loader::{LazyStore, SafetensorsWriter, TensorSpec};
use fastvideo_models::h3::config::H3TextEncoderConfig;

use super::text_cache::tap_keys;
use crate::llm::DecoderConfig;
use crate::wan::tensor::{Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Default shard size. Shards break only between layers.
pub const SLIM_SHARD_BYTES: u64 = 5 << 30;

#[derive(Debug, Clone)]
pub struct SlimReport {
    pub files: Vec<PathBuf>,
    pub tensors: usize,
    /// Tensor bytes written (file sizes are these plus headers).
    pub bytes: u64,
    /// What the source holds and the slim copy leaves out.
    pub skipped_tensors: usize,
    pub skipped_bytes: u64,
}

/// Write `out_dir/model-NNNNN-of-MMMMM.safetensors` holding the tensors tap
/// `tap` of `cfg` reads from `store`, in load order, bytes as stored.
pub fn write_slim(store: &LazyStore, cfg: &DecoderConfig, tap: usize, out_dir: &Path, shard_bytes: u64) -> Result<SlimReport> {
    if tap == 0 || tap >= cfg.num_layers() {
        return Err(msg(format!("slim: tap {tap} of a {}-layer decoder (the last tap is normed and needs the whole model)", cfg.num_layers())));
    }
    let keys = tap_keys(cfg, tap);
    let per_layer = (keys.len() - 1) / tap;
    // Plan first: a safetensors header needs every offset before any data.
    let mut shards: Vec<Vec<TensorSpec>> = vec![Vec::new()];
    let (mut in_shard, mut total) = (0u64, 0u64);
    for (i, key) in keys.iter().enumerate() {
        let view = store.view(key).map_err(|e| msg(e.to_string()))?;
        let at_layer_start = i >= 1 && (i - 1) % per_layer == 0;
        if at_layer_start && in_shard >= shard_bytes.max(1) {
            shards.push(Vec::new());
            in_shard = 0;
        }
        in_shard += view.bytes.len() as u64;
        total += view.bytes.len() as u64;
        shards.last_mut().expect("one shard").push(TensorSpec::new(key.clone(), view.dtype.clone(), view.shape.to_vec()));
    }

    std::fs::create_dir_all(out_dir).map_err(|e| msg(format!("{}: {e}", out_dir.display())))?;
    let tap_text = tap.to_string();
    let mut files = Vec::with_capacity(shards.len());
    for (n, specs) in shards.iter().enumerate() {
        let path = out_dir.join(format!("model-{:05}-of-{:05}.safetensors", n + 1, shards.len()));
        let metadata = [("format", "pt"), ("slim_for", "MiniMax-H3 text conditioning"), ("hidden_states_tap", tap_text.as_str()), ("order", "load order of fastvideo_cudarc::llm")];
        let mut writer = SafetensorsWriter::create(&path, specs, &metadata).map_err(|e| msg(e.to_string()))?;
        for spec in specs {
            let view = store.view(&spec.name).map_err(|e| msg(e.to_string()))?;
            writer.write(&spec.name, view.bytes).map_err(|e| msg(e.to_string()))?;
        }
        writer.finish().map_err(|e| msg(e.to_string()))?;
        crate::wan::log::info(format_args!("slim text encoder: shard {}/{} written", n + 1, shards.len()));
        files.push(path);
    }
    let all_bytes = store.bytes_with_prefix("") as u64;
    Ok(SlimReport { files, tensors: keys.len(), bytes: total, skipped_tensors: store.keys().count() - keys.len(), skipped_bytes: all_bytes - total })
}

/// `fv-gpucheck h3 slim-text`: `<root>/text_encoder` to `<out>/text_encoder`
/// (slim) and `<root>/tokenizer` to `<out>/tokenizer` (copied), so `<out>` is
/// a root [`super::text`] accepts as is. CPU only.
pub fn write_slim_text_root(root: &Path, out: &Path, shard_bytes: u64) -> Result<SlimReport> {
    let store = LazyStore::open(&root.join("text_encoder")).map_err(|e| msg(e.to_string()))?;
    let tap = H3TextEncoderConfig::fasth3_8step().output_hidden_state_index;
    let report = write_slim(&store, &DecoderConfig::qwen3_vl_32b_text(), tap, &out.join("text_encoder"), shard_bytes)?;
    let (from, to) = (root.join("tokenizer"), out.join("tokenizer"));
    std::fs::create_dir_all(&to).map_err(|e| msg(format!("{}: {e}", to.display())))?;
    for entry in std::fs::read_dir(&from).map_err(|e| msg(format!("{}: {e}", from.display())))? {
        let path = entry.map_err(|e| msg(e.to_string()))?.path();
        if let (true, Some(name)) = (path.is_file(), path.file_name()) {
            std::fs::copy(&path, to.join(name)).map_err(|e| msg(format!("{}: {e}", path.display())))?;
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h3::text::{encode_ids, HiddenStateEncoder, StreamedEncoder};
    use crate::h3::text_cache::{cache_key, encoder_identity, get_or_compute};
    use crate::llm::{Act, LayerAttn};
    use crate::wan::weights::WeightMap;
    use fastvideo_loader::LazyDType;
    use std::sync::{Arc, Mutex};

    fn tiny() -> DecoderConfig {
        DecoderConfig {
            vocab: 6,
            hidden: 8,
            heads: 4,
            kv_heads: 2,
            head_dim: 4,
            intermediate: 12,
            rms_eps: 1e-6,
            norm_offset: 0.0,
            act: Act::Silu,
            qk_norm: true,
            sandwich_norms: false,
            embed_scale: 1.0,
            attn_scale: 0.5,
            layers: vec![LayerAttn { rope_theta: 5_000_000.0, rope_factor: 1.0, window: None }; 4],
            layer_prefix: "model.language_model.layers".into(),
            embed_key: "model.language_model.embed_tokens.weight".into(),
            final_norm_key: "model.language_model.norm.weight".into(),
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fv-h3-slim-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A "published" toy checkpoint: all 4 layers, the final norm, an lm_head and
    /// a vision tensor, spread over two shards in lexicographic key order.
    fn publish(dir: &Path, cfg: &DecoderConfig) {
        let mut all: Vec<(String, Vec<usize>)> = Vec::new();
        let (h, d) = (cfg.hidden, cfg.head_dim);
        all.push((cfg.embed_key.clone(), vec![6, h]));
        all.push((cfg.final_norm_key.clone(), vec![h]));
        all.push(("lm_head.weight".into(), vec![6, h]));
        all.push(("model.visual.patch_embed.weight".into(), vec![3, 5]));
        for l in 0..cfg.num_layers() {
            let p = format!("{}.{l}", cfg.layer_prefix);
            for (n, shape) in [
                ("input_layernorm.weight", vec![h]),
                ("post_attention_layernorm.weight", vec![h]),
                ("self_attn.q_proj.weight", vec![cfg.heads * d, h]),
                ("self_attn.k_proj.weight", vec![cfg.kv_heads * d, h]),
                ("self_attn.v_proj.weight", vec![cfg.kv_heads * d, h]),
                ("self_attn.o_proj.weight", vec![h, cfg.heads * d]),
                ("self_attn.q_norm.weight", vec![d]),
                ("self_attn.k_norm.weight", vec![d]),
                ("mlp.gate_proj.weight", vec![cfg.intermediate, h]),
                ("mlp.up_proj.weight", vec![cfg.intermediate, h]),
                ("mlp.down_proj.weight", vec![h, cfg.intermediate]),
            ] {
                all.push((format!("{p}.{n}"), shape));
            }
        }
        all.sort();
        let half = all.len() / 2;
        for (n, part) in [&all[..half], &all[half..]].into_iter().enumerate() {
            let specs: Vec<TensorSpec> = part.iter().map(|(k, s)| TensorSpec::new(k.clone(), LazyDType::BF16, s.clone())).collect();
            let mut w = SafetensorsWriter::create(&dir.join(format!("model-{:05}-of-00002.safetensors", n + 1)), &specs, &[]).unwrap();
            for (key, shape) in part {
                let seed = key.bytes().fold(7u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
                let bytes: Vec<u8> = (0..shape.iter().product::<usize>())
                    .flat_map(|i| {
                        let u = (seed.wrapping_add(i as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / (1u32 << 24) as f32;
                        let v = if key.contains("norm") { 0.5 + u } else { u - 0.5 };
                        half::bf16::from_f32(v).to_bits().to_le_bytes()
                    })
                    .collect();
                w.write(key, &bytes).unwrap();
            }
            w.finish().unwrap();
        }
    }

    /// The content list is not a transcription to be trusted: it is held equal,
    /// name for name and in order, to what the decoder really asks a map for.
    #[test]
    fn the_slim_set_is_exactly_what_the_loader_requests_in_order() {
        for (cfg, tap) in [(tiny(), 2usize), (DecoderConfig::gemma3_12b_text(), 1)] {
            let asked = Arc::new(Mutex::new(Vec::<String>::new()));
            let sink = asked.clone();
            let map = WeightMap::generated(move |key, shape| {
                sink.lock().unwrap().push(key.to_string());
                vec![0.0; shape.iter().product()]
            });
            let mut small = cfg.clone();
            if small.hidden > 64 {
                // Gemma's real widths would allocate gigabytes; the key order does not depend on them.
                (small.hidden, small.heads, small.kv_heads, small.head_dim, small.intermediate) = (8, 4, 2, 4, 12);
            }
            crate::llm::hidden_states(&map, &small, &[1, 2], &[0, 1], &[true, true], &[tap]).unwrap();
            assert_eq!(*asked.lock().unwrap(), tap_keys(&small, tap), "sandwich_norms = {}", cfg.sandwich_norms);
        }
    }

    #[test]
    fn production_slim_is_embed_plus_50_layers_and_50_3_gb() {
        let doc: serde_json::Value = serde_json::from_str(include_str!("manifests/text_encoder.json")).unwrap();
        let tensors = doc["tensors"].as_object().unwrap();
        let keys = tap_keys(&DecoderConfig::qwen3_vl_32b_text(), 50);
        let bytes: u64 = keys
            .iter()
            .map(|k| {
                let t = tensors.get(k).unwrap_or_else(|| panic!("{k} is not in the published checkpoint"));
                assert_eq!(t[0], "BF16");
                2 * t[1].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).product::<u64>()
            })
            .sum();
        assert_eq!((keys.len(), bytes), (551, 50_315_658_240), "46.86 GiB of the published 66.7 GB");
    }

    #[test]
    fn slim_holds_the_same_bytes_encodes_identically_and_shares_the_cache() {
        let (cfg, tap) = (tiny(), 2usize);
        let root = temp_dir("roundtrip");
        let (published, slim) = (root.join("published"), root.join("slim"));
        std::fs::create_dir_all(&published).unwrap();
        publish(&published, &cfg);
        let source = LazyStore::open(&published).unwrap();
        // A tiny shard budget forces a break at every layer boundary.
        let report = write_slim(&source, &cfg, tap, &slim, 1).unwrap();
        assert_eq!((report.files.len(), report.tensors), (3, 1 + 2 * 11), "embed | layer 0 | layer 1");
        assert_eq!(report.skipped_tensors, 3 + 2 * 11, "norm, lm_head, vision, layers 2 and 3");
        assert!(report.files.windows(2).all(|w| w[0] < w[1]), "shard names sort in load order");

        let packed = LazyStore::open(&slim).unwrap();
        let mut keys: Vec<&str> = packed.keys().collect();
        keys.sort_unstable();
        let mut want = tap_keys(&cfg, tap);
        want.sort_unstable();
        assert_eq!(keys, want);
        for key in &want {
            let (a, b) = (source.view(key).unwrap(), packed.view(key).unwrap());
            assert!(a.bytes == b.bytes && a.shape == b.shape && a.dtype == b.dtype, "{key}");
        }
        assert_eq!(report.bytes, want.iter().map(|k| packed.view(k).unwrap().bytes.len() as u64).sum::<u64>());
        assert_eq!(encoder_identity(&source, &cfg, tap).unwrap(), encoder_identity(&packed, &cfg, tap).unwrap(), "one encoder, two layouts");
        assert!(write_slim(&source, &cfg, 4, &root.join("bad"), 1).is_err(), "the last tap needs the final norm");

        // Same hidden state from either layout...
        let ids = [1u32, 4, 2];
        let (full_map, slim_map) = (WeightMap::open(&published).unwrap(), WeightMap::open(&slim).unwrap());
        let a = encode_ids(&full_map, &cfg, &ids, tap).unwrap().host_cow().unwrap().into_owned();
        let b = encode_ids(&slim_map, &cfg, &ids, tap).unwrap().host_cow().unwrap().into_owned();
        assert_eq!(a, b);
        // ...and an entry cached from one is a hit from the other, without a forward.
        let cache = root.join("cache");
        let key = |store: &LazyStore| cache_key("p", &[5; 32], tap, &encoder_identity(store, &cfg, tap).unwrap());
        let encoder = StreamedEncoder { map: &full_map, cfg: &cfg };
        let (_, hit) = get_or_compute(&cache, &key(&source), &ids, cfg.hidden, || Ok(encoder.hidden_state(&ids, tap)?.host_cow()?.into_owned())).unwrap();
        assert!(!hit);
        let (entry, hit) = get_or_compute(&cache, &key(&packed), &ids, cfg.hidden, || panic!("a hit must not run the encoder")).unwrap();
        assert!(hit);
        assert_eq!(entry.data, a);
        let _ = std::fs::remove_dir_all(&root);
    }
}
