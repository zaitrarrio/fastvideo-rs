//! A slim text-encoder checkpoint: the language model only, its Linears already
//! bfloat16, laid out in the order the streaming decoder reads it.
//!
//! `Lightricks/LTX-2` ships Gemma-3-12B as 48.7 GB of float32 — including a
//! vision tower the text-only path never touches and, in the same folder, a
//! stale 52 GB duplicate. Every prompt therefore reads 47 GB, narrows it to
//! bf16 on the host and uploads 23.5 GB. Narrowing once, offline, removes the
//! conversion and halves the read.
//!
//! **What is narrowed and what is not.** The loaders do not treat every tensor
//! alike, and a slim checkpoint must load to the *same device bits*:
//!
//! * 2-D projection weights go to the device as bf16 through
//!   `half::bf16::from_f32` (`WeightMap::lazy_bf16`). They are stored as that
//!   exact conversion, so the device weight is bit-identical.
//! * norm weights (1-D) are used as float32 on the device. Stored float32,
//!   untouched — they are 0.004% of the bytes.
//! * the embedding table is gathered row by row and used as float32. It is kept
//!   float32 by default so conditioning is bit-identical to the original
//!   checkpoint's; `EmbedDtype::Bf16` saves another 2 GB and makes the rows
//!   what the bf16 *reference* uses, at the price of not being bit-identical to
//!   our own float32-row path.
//!
//! The keys are unchanged, so every loader works on either layout. (A
//! `--mode exact` run on a slim checkpoint sees bf16-rounded projection weights
//! widened back to float32: use the original for float32 parity work.)
//!
//! Tensors are written in load order — embedding, layer 0 … n-1 in the order
//! [`crate::llm`] asks for them, final norm — and sharded, with names that sort
//! in that order, so a streamed forward is one sequential read.

use std::path::{Path, PathBuf};

use fastvideo_loader::{LazyDType, LazyStore, SafetensorsWriter, TensorSpec};
use rayon::prelude::*;

use crate::llm::DecoderConfig;
use crate::wan::tensor::Result;

use super::msg;

/// How the embedding table is stored (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedDtype {
    F32,
    Bf16,
}

#[derive(Debug, Clone)]
pub struct SlimOptions {
    pub embed: EmbedDtype,
    /// Soft cap per output file; a tensor is never split.
    pub shard_bytes: u64,
}

impl Default for SlimOptions {
    fn default() -> Self {
        Self {
            embed: EmbedDtype::F32,
            shard_bytes: 5 << 30,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SlimReport {
    pub files: Vec<(PathBuf, u64)>,
    pub tensors: usize,
    /// Bytes of the source tensors that were read.
    pub bytes_in: u64,
    /// Tensor bytes written (headers excluded).
    pub bytes_out: u64,
    pub narrowed: usize,
}

/// Every key of the language model in the order the streaming decoder requests
/// it. Mirrors `llm::Layer::load`; a test holds the two together.
pub fn load_order(cfg: &DecoderConfig) -> Vec<String> {
    let mut keys = vec![cfg.embed_key.clone()];
    for i in 0..cfg.num_layers() {
        let p = format!("{}.{i}", cfg.layer_prefix);
        let norms: &[&str] = if cfg.sandwich_norms {
            &[
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ]
        } else {
            &["post_attention_layernorm"]
        };
        keys.extend(norms.iter().map(|n| format!("{p}.{n}.weight")));
        for lin in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            if cfg.layer_k_eq_v(i) && lin == "self_attn.v_proj" {
                continue;
            }
            keys.push(format!("{p}.{lin}.weight"));
        }
        if cfg.qk_norm {
            keys.extend(
                ["self_attn.q_norm", "self_attn.k_norm"]
                    .iter()
                    .map(|n| format!("{p}.{n}.weight")),
            );
        }
        keys.push(format!("{p}.input_layernorm.weight"));
        if cfg.layer_scalar {
            keys.push(format!("{p}.{}", crate::llm::LAYER_SCALAR));
        }
    }
    keys.push(cfg.final_norm_key.clone());
    keys
}

/// What one tensor becomes: `(dtype on disk, narrowed?)`.
fn target(
    store: &LazyStore,
    key: &str,
    cfg: &DecoderConfig,
    options: &SlimOptions,
) -> Result<(LazyDType, Vec<usize>)> {
    let view = store.view(key).map_err(|e| msg(format!("slim: {e}")))?;
    let shape = view.shape.to_vec();
    let float = matches!(
        view.dtype,
        LazyDType::F32 | LazyDType::F16 | LazyDType::BF16
    );
    if !float {
        return Err(msg(format!(
            "slim: `{key}` is {:?}; only float checkpoints can be narrowed",
            view.dtype
        )));
    }
    let narrow = if key == cfg.embed_key {
        options.embed == EmbedDtype::Bf16
    } else {
        shape.len() == 2
    };
    // Never widen: a tensor already narrower than float32 is copied as it is.
    let dtype = if narrow && *view.dtype == LazyDType::F32 {
        LazyDType::BF16
    } else {
        view.dtype.clone()
    };
    Ok((dtype, shape))
}

/// The bytes of `key` as they go into the slim file.
fn payload(store: &LazyStore, key: &str, dtype: &LazyDType) -> Result<Vec<u8>> {
    let view = store.view(key).map_err(|e| msg(format!("slim: {e}")))?;
    if view.dtype == dtype {
        return Ok(view.bytes.to_vec());
    }
    // The one conversion there is: float32 → bfloat16, by the function the
    // loader itself uses when it narrows at load time.
    Ok(view
        .bytes
        .par_chunks_exact(4)
        .flat_map_iter(|c| {
            half::bf16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .to_bits()
                .to_le_bytes()
        })
        .collect())
}

/// Write the slim language model of the checkpoint in `source` (a directory of
/// shards, or one file) into `out_dir`. One tensor is in memory at a time.
pub fn write_slim_decoder(
    source: &Path,
    out_dir: &Path,
    cfg: &DecoderConfig,
    options: &SlimOptions,
) -> Result<SlimReport> {
    let store = if source.is_file() {
        LazyStore::open_files(&[source.to_path_buf()])
    } else {
        LazyStore::open(source)
    }
    .map_err(|e| msg(format!("slim: {}: {e}", source.display())))?;
    let keys = load_order(cfg);
    let missing: Vec<&String> = keys.iter().filter(|k| !store.contains(k)).take(5).collect();
    if !missing.is_empty() {
        return Err(msg(format!(
            "slim: {} lacks language-model keys, e.g. {missing:?}",
            source.display()
        )));
    }
    let plan: Vec<(String, LazyDType, Vec<usize>, u64)> = keys
        .iter()
        .map(|k| {
            let (dtype, shape) = target(&store, k, cfg, options)?;
            let bytes = shape.iter().product::<usize>() as u64 * dtype.size().unwrap_or(0) as u64;
            Ok((k.clone(), dtype, shape, bytes))
        })
        .collect::<Result<_>>()?;

    // Greedy shards in load order.
    let mut shards: Vec<Vec<usize>> = vec![Vec::new()];
    let mut filled = 0u64;
    for (i, (_, _, _, bytes)) in plan.iter().enumerate() {
        if filled > 0 && filled + bytes > options.shard_bytes.max(1) {
            shards.push(Vec::new());
            filled = 0;
        }
        shards
            .last_mut()
            .ok_or_else(|| msg("slim: no shard"))?
            .push(i);
        filled += bytes;
    }

    std::fs::create_dir_all(out_dir)
        .map_err(|e| msg(format!("slim: {}: {e}", out_dir.display())))?;
    let mut report = SlimReport {
        files: Vec::new(),
        tensors: plan.len(),
        bytes_in: 0,
        bytes_out: 0,
        narrowed: 0,
    };
    let total = shards.len();
    let source_text = source.display().to_string();
    for (n, shard) in shards.iter().enumerate() {
        // Zero-padded, so lexicographic order is load order.
        let path = out_dir.join(format!("model-slim-{:05}-of-{total:05}.safetensors", n + 1));
        let specs: Vec<TensorSpec> = shard
            .iter()
            .map(|&i| TensorSpec::new(plan[i].0.clone(), plan[i].1.clone(), plan[i].2.clone()))
            .collect();
        let metadata = [
            ("source", source_text.as_str()),
            ("content", "language model only; 2-D float32 weights narrowed with half::bf16::from_f32; tensors in load order"),
            ("embedding", if options.embed == EmbedDtype::F32 { "float32 (bit-identical rows)" } else { "bfloat16" }),
        ];
        let mut writer = SafetensorsWriter::create(&path, &specs, &metadata)
            .map_err(|e| msg(format!("slim: {e}")))?;
        for &i in shard {
            let (key, dtype, _, bytes) = &plan[i];
            let source_view = store.view(key).map_err(|e| msg(format!("slim: {e}")))?;
            report.bytes_in += source_view.bytes.len() as u64;
            report.narrowed += usize::from(source_view.dtype != dtype);
            let data = payload(&store, key, dtype)?;
            if data.len() as u64 != *bytes {
                return Err(msg(format!(
                    "slim: `{key}` produced {} bytes, planned {bytes}",
                    data.len()
                )));
            }
            writer
                .write(key, &data)
                .map_err(|e| msg(format!("slim: {e}")))?;
            report.bytes_out += *bytes;
        }
        writer.finish().map_err(|e| msg(format!("slim: {e}")))?;
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        crate::wan::log::info(format_args!(
            "slim: wrote {} ({:.2} GiB)",
            path.display(),
            size as f64 / f64::from(1u32 << 30)
        ));
        report.files.push((path, size));
    }
    Ok(report)
}

/// Copy every regular file of `from` into `to` (the tokenizer folder).
pub fn copy_dir_files(from: &Path, to: &Path) -> Result<usize> {
    let io = |e: std::io::Error, p: &Path| msg(format!("slim: {}: {e}", p.display()));
    std::fs::create_dir_all(to).map_err(|e| io(e, to))?;
    let mut copied = 0;
    for entry in std::fs::read_dir(from).map_err(|e| io(e, from))? {
        let path = entry.map_err(|e| io(e, from))?.path();
        if path.is_file() {
            let name = path.file_name().ok_or_else(|| msg("slim: unnamed file"))?;
            std::fs::copy(&path, to.join(name)).map_err(|e| io(e, &path))?;
            copied += 1;
        }
    }
    Ok(copied)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::llm::{self, Act, LayerAttn};
    use crate::wan::weights::WeightMap;

    fn tiny(sandwich: bool) -> DecoderConfig {
        DecoderConfig {
            vocab: 5,
            hidden: 8,
            heads: 4,
            kv_heads: 2,
            head_dim: 4,
            intermediate: 12,
            rms_eps: 1e-6,
            norm_offset: if sandwich { 1.0 } else { 0.0 },
            act: if sandwich { Act::GeluTanh } else { Act::Silu },
            qk_norm: true,
            sandwich_norms: sandwich,
            embed_scale: 1.0,
            attn_scale: 0.5,
            layers: vec![LayerAttn::global(10_000.0, 1.0); 2],
            layer_prefix: "language_model.model.layers".into(),
            embed_key: "language_model.model.embed_tokens.weight".into(),
            final_norm_key: "language_model.model.norm.weight".into(),
            attention_k_eq_v: false,
            v_norm: false,
            layer_scalar: false,
        }
    }

    /// What the decoder asks a map for, in order, with shapes.
    fn requested(cfg: &DecoderConfig, vocab: usize) -> Vec<(String, Vec<usize>)> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let map = WeightMap::generated(move |key, shape| {
            sink.lock().unwrap().push((key.to_string(), shape.to_vec()));
            vec![0.0; shape.iter().product()]
        });
        let last = vocab as u32 - 1;
        llm::hidden_states(&map, cfg, &[last], &[0], &[true], &[cfg.num_layers()]).unwrap();
        let out = seen.lock().unwrap().clone();
        out
    }

    #[test]
    fn load_order_is_the_order_the_decoder_asks_in() {
        for sandwich in [false, true] {
            let cfg = tiny(sandwich);
            let asked: Vec<String> = requested(&cfg, 5).into_iter().map(|(k, _)| k).collect();
            assert_eq!(load_order(&cfg), asked, "sandwich={sandwich}");
        }
    }

    /// A float32 checkpoint with the language model, a vision tower and a stale
    /// duplicate, on disk.
    fn source_checkpoint(
        dir: &Path,
        cfg: &DecoderConfig,
        vocab: usize,
    ) -> BTreeMap<String, Vec<f32>> {
        std::fs::create_dir_all(dir).unwrap();
        let mut tensors: Vec<(String, Vec<usize>)> = requested(cfg, vocab);
        tensors.push(("vision_tower.patch.weight".into(), vec![3, 3]));
        tensors.push((
            "base_text_encoder.language_model.model.norm.weight".into(),
            vec![8],
        ));
        tensors.sort();
        let values: BTreeMap<String, Vec<f32>> = tensors
            .iter()
            .map(|(k, shape)| {
                let seed = k
                    .bytes()
                    .fold(3u32, |a, b| a.wrapping_mul(33).wrapping_add(u32::from(b)));
                let n: usize = shape.iter().product();
                // Values with more mantissa than bf16 holds, so narrowing is visible.
                (
                    k.clone(),
                    (0..n)
                        .map(|i| {
                            ((seed.wrapping_add(i as u32).wrapping_mul(2_654_435_761) >> 7) as f32
                                / 1e6)
                                .sin()
                                * 1.234_567
                        })
                        .collect(),
                )
            })
            .collect();
        let specs: Vec<TensorSpec> = tensors
            .iter()
            .map(|(k, s)| TensorSpec::new(k.clone(), LazyDType::F32, s.clone()))
            .collect();
        let mut w =
            SafetensorsWriter::create(&dir.join("model-00001-of-00001.safetensors"), &specs, &[])
                .unwrap();
        for (k, _) in &tensors {
            let bytes: Vec<u8> = values[k].iter().flat_map(|v| v.to_le_bytes()).collect();
            w.write(k, &bytes).unwrap();
        }
        w.finish().unwrap();
        values
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fv-ltx2-slim-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn the_slim_checkpoint_is_exactly_the_language_model_narrowed_by_the_loaders_own_rounding() {
        let cfg = tiny(true);
        let (src, out) = (scratch("src"), scratch("out"));
        let values = source_checkpoint(&src, &cfg, 5);
        // Tiny shards, to exercise the split: each holds a few tensors.
        let report = write_slim_decoder(
            &src,
            &out,
            &cfg,
            &SlimOptions {
                embed: EmbedDtype::F32,
                shard_bytes: 600,
            },
        )
        .unwrap();
        assert!(report.files.len() > 2, "{} shards", report.files.len());
        assert!(report.bytes_out < report.bytes_in);

        let slim = LazyStore::open(&out).unwrap();
        let asked = requested(&cfg, 5);
        // The key set is exactly what the decoder requests — no vision tower, no duplicate.
        let mut have: Vec<&str> = slim.keys().collect();
        have.sort_unstable();
        let mut want: Vec<&str> = asked.iter().map(|(k, _)| k.as_str()).collect();
        want.sort_unstable();
        assert_eq!(have, want);

        for (key, shape) in &asked {
            let view = slim.view(key).unwrap();
            assert_eq!(view.shape, &shape[..], "{key}");
            let linear = shape.len() == 2 && *key != cfg.embed_key;
            if linear {
                assert_eq!(*view.dtype, LazyDType::BF16, "{key}");
                let want: Vec<u8> = values[key]
                    .iter()
                    .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
                    .collect();
                assert_eq!(
                    view.bytes,
                    &want[..],
                    "{key}: the device would not get the same bf16 bits"
                );
            } else {
                assert_eq!(*view.dtype, LazyDType::F32, "{key}");
                let want: Vec<u8> = values[key].iter().flat_map(|v| v.to_le_bytes()).collect();
                assert_eq!(
                    view.bytes,
                    &want[..],
                    "{key}: float32 tensors are copied untouched"
                );
            }
        }

        // Shards sort in load order, and inside a shard tensors sit in load order.
        let order = load_order(&cfg);
        let mut position = 0usize;
        for (path, _) in &report.files {
            let shard = LazyStore::open_files(std::slice::from_ref(path)).unwrap();
            let mut keys: Vec<(usize, &str)> = shard
                .keys()
                .map(|k| (order.iter().position(|o| o == k).unwrap(), k))
                .collect();
            keys.sort_unstable();
            for (at, _) in keys {
                assert_eq!(at, position, "{}", path.display());
                position += 1;
            }
        }
        assert_eq!(position, order.len());
        let mut names: Vec<_> = report.files.iter().map(|(p, _)| p.clone()).collect();
        names.sort();
        assert_eq!(
            names,
            report
                .files
                .iter()
                .map(|(p, _)| p.clone())
                .collect::<Vec<_>>()
        );
    }

    /// Either layout, the same model: on the CPU path both maps widen to float32,
    /// so the slim one computes with bf16-rounded projections — close, and equal
    /// to the original once *its* projections are rounded the same way.
    #[test]
    fn the_decoder_loads_from_the_slim_layout() {
        let cfg = tiny(true);
        let (src, out) = (scratch("src2"), scratch("out2"));
        source_checkpoint(&src, &cfg, 5);
        write_slim_decoder(&src, &out, &cfg, &SlimOptions::default()).unwrap();
        let run = |dir: &Path| {
            let map = WeightMap::open(dir).unwrap();
            llm::hidden_states(&map, &cfg, &[1, 4, 2], &[0, 1, 2], &[true; 3], &[2]).unwrap()[0]
                .host_cow()
                .unwrap()
                .into_owned()
        };
        let (a, b) = (run(&src), run(&out));
        assert_eq!(a.len(), b.len());
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(worst < 0.05, "slim and original disagree by {worst}");
        assert!(worst > 0.0, "the test values must exercise the narrowing");
    }

    #[test]
    fn a_bf16_embedding_is_an_option_and_a_second_run_never_widens() {
        let cfg = tiny(false);
        let (src, out, again) = (scratch("src3"), scratch("out3"), scratch("out3b"));
        source_checkpoint(&src, &cfg, 5);
        let first = write_slim_decoder(
            &src,
            &out,
            &cfg,
            &SlimOptions {
                embed: EmbedDtype::Bf16,
                ..SlimOptions::default()
            },
        )
        .unwrap();
        let slim = LazyStore::open(&out).unwrap();
        assert_eq!(*slim.view(&cfg.embed_key).unwrap().dtype, LazyDType::BF16);
        // Slimming a slim checkpoint copies it: nothing left to narrow, nothing widened.
        let second = write_slim_decoder(&out, &again, &cfg, &SlimOptions::default()).unwrap();
        assert_eq!((second.narrowed, second.bytes_out), (0, first.bytes_out));
        // A checkpoint without the language model is refused by name.
        let err = write_slim_decoder(
            &src,
            &scratch("out3c"),
            &DecoderConfig {
                layer_prefix: "nowhere.layers".into(),
                ..cfg
            },
            &SlimOptions::default(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("lacks language-model keys"),
            "{err}"
        );
    }

    /// The production byte count, from the published headers alone.
    #[test]
    fn the_published_gemma_slims_from_47_gb_to_25_gb_of_tensors() {
        let doc: serde_json::Value =
            serde_json::from_str(include_str!("manifests/gemma.json")).unwrap();
        let tensors = doc["tensors"].as_object().unwrap();
        let cfg = DecoderConfig::gemma3_12b_text();
        let (mut before, mut f32_embed, mut bf16_embed) = (0u64, 0u64, 0u64);
        for key in load_order(&cfg) {
            let entry = tensors
                .get(&key)
                .unwrap_or_else(|| panic!("{key} is not in the published checkpoint"));
            assert_eq!(entry[0], "F32");
            let shape: Vec<u64> = entry[1]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap())
                .collect();
            let n: u64 = shape.iter().product();
            before += n * 4;
            let linear = shape.len() == 2 && key != cfg.embed_key;
            f32_embed += n * if linear { 2 } else { 4 };
            bf16_embed += n * if shape.len() == 2 { 2 } else { 4 };
        }
        assert_eq!(
            load_order(&cfg).len(),
            tensors.len(),
            "every published language-model key is written"
        );
        assert_eq!(before, 47_064_136_704);
        assert_eq!(f32_embed, 25_547_357_184); // 23.79 GiB
        assert_eq!(bf16_embed, 23_533_599_744); // 21.92 GiB
    }
}
