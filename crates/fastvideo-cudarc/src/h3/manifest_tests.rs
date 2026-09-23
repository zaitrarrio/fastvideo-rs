//! Every H3 loader against the published checkpoint's key manifests — no GPU,
//! no weights.
//!
//! A wrong key or shape is the one failure that can always be found without
//! hardware. `manifests/*.json` are the safetensors *headers* of the real
//! shards (`scripts/gpu/h3_manifests.py`: two HTTP range requests per shard,
//! stripped to `key -> [dtype, shape]`, repo and revision recorded inside).
//! Each test runs the real loader at the **production** config against a
//! recording [`WeightMap::generated`] closure, which sees every `(key, expected
//! shape)` the loader asks for. Then:
//!
//! * every requested key must exist in the manifest with exactly that shape;
//! * where the loader is meant to consume a whole key family, no manifest key
//!   of that family may go unrequested — and where it is meant to *skip* one
//!   (the 26 GB of AdaLN projections as resident weights, the encoders, the
//!   vision tower), that is asserted too.
//!
//! The generator returns zeros and the heavy tests share one lock so their
//! peaks do not stack. The DiT is 35B parameters, so the stack is loaded with a
//! one-block config, block 49 is loaded for real on its own, and the other 48
//! are checked by substituting the block index, which is all the loader's
//! per-block code does with it. The same substitution covers the 36 VAE blocks
//! and the 50 text-encoder layers.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use fastvideo_models::h3::config::{H3AudioVaeConfig, H3TransformerConfig, H3VideoVaeConfig};
use fastvideo_models::h3::schedule::{H3JointSchedule, H3Schedule};

use crate::llm::{self, DecoderConfig};
use crate::wan::weights::WeightMap;

use super::audio_vae::H3AudioDecoder;
use super::transformer::{AdaLnTable, Block, H3TextRefiner, H3Transformer};
use super::vae::H3VideoDecoder;

type Requests = BTreeMap<String, Vec<usize>>;

/// `key -> shape` of one published component.
fn manifest(json: &str) -> Requests {
    let doc: serde_json::Value = serde_json::from_str(json).expect("manifest is JSON");
    for field in ["repo", "revision", "files"] {
        assert!(doc.get(field).is_some(), "manifest records its {field}");
    }
    doc["tensors"]
        .as_object()
        .expect("manifest has tensors")
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v[1].as_array()
                    .expect("shape")
                    .iter()
                    .map(|d| d.as_u64().expect("dim") as usize)
                    .collect(),
            )
        })
        .collect()
}

fn dtypes(json: &str) -> BTreeMap<String, String> {
    let doc: serde_json::Value = serde_json::from_str(json).expect("manifest is JSON");
    doc["tensors"]
        .as_object()
        .expect("tensors")
        .iter()
        .map(|(k, v)| (k.clone(), v[0].as_str().expect("dtype").to_string()))
        .collect()
}

/// A map that invents zeros for any key and remembers what was asked.
fn recording() -> (WeightMap, Arc<Mutex<Requests>>) {
    let seen = Arc::new(Mutex::new(Requests::new()));
    let sink = seen.clone();
    let map = WeightMap::generated(move |key, shape| {
        sink.lock()
            .expect("recorder lock")
            .insert(key.to_string(), shape.to_vec());
        vec![0.0; shape.iter().product()]
    });
    (map, seen)
}

/// One lock for the tests that allocate gigabytes, so their peaks do not add up.
fn heavy() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Requests for `<stem>.<from>.…` repeated for every index in `all`.
fn for_every_index(
    requested: &Requests,
    stem: &str,
    from: usize,
    all: std::ops::Range<usize>,
) -> Requests {
    let needle = format!("{stem}.{from}.");
    let mut out = Requests::new();
    for (key, shape) in requested {
        match key.strip_prefix(&needle) {
            Some(rest) => out.extend(
                all.clone()
                    .map(|i| (format!("{stem}.{i}.{rest}"), shape.clone())),
            ),
            None => {
                out.insert(key.clone(), shape.clone());
            }
        }
    }
    out
}

fn mismatches(requested: &Requests, published: &Requests) -> Vec<String> {
    requested
        .iter()
        .filter_map(|(key, want)| match published.get(key) {
            None => Some(format!(
                "loader asks for `{key}` {want:?}: not in the checkpoint"
            )),
            Some(have) if have != want => Some(format!(
                "`{key}`: loader expects {want:?}, checkpoint has {have:?}"
            )),
            Some(_) => None,
        })
        .collect()
}

fn unrequested(
    requested: &Requests,
    published: &Requests,
    owned: impl Fn(&str) -> bool,
) -> Vec<String> {
    published
        .keys()
        .filter(|k| owned(k) && !requested.contains_key(*k))
        .map(|k| format!("checkpoint key `{k}` is never loaded"))
        .collect()
}

fn assert_clean(what: &str, problems: Vec<String>) {
    assert!(
        problems.is_empty(),
        "{what}: {} problem(s)\n  {}",
        problems.len(),
        problems.join("\n  ")
    );
}

#[test]
fn audio_decoder_asks_for_exactly_the_published_decoder() {
    let published = manifest(include_str!("manifests/audio_vae.json"));
    let (map, seen) = recording();
    H3AudioDecoder::load(H3AudioVaeConfig::fasth3_8step(), &map).expect("load");
    let seen = seen.lock().expect("lock").clone();
    let mut problems = mismatches(&seen, &published);
    // One of the 254 identical filter buffers is read (the `h3 audio-vae` stage
    // checks they are the same bytes); everything else of the decoder is consumed.
    problems.extend(unrequested(&seen, &published, |k| {
        (k.starts_with("decoder.") || k.starts_with("dec_in_proj.")) && !k.ends_with("filter")
    }));
    assert_clean("audio_vae", problems);
    assert_eq!(
        published
            .keys()
            .filter(|k| k.ends_with("filter") && k.starts_with("decoder."))
            .count(),
        254
    );
    assert!(
        seen.keys()
            .all(|k| !k.starts_with("encoder.") && !k.starts_with("pre_block.")),
        "T2AV never reads the audio encoder"
    );
    assert!(
        dtypes(include_str!("manifests/audio_vae.json"))
            .values()
            .all(|d| d == "F32"),
        "the audio VAE must stay float32"
    );
}

#[test]
fn video_decoder_asks_for_exactly_the_published_decoder() {
    let _guard = heavy();
    let published = manifest(include_str!("manifests/vae.json"));
    let (map, seen) = recording();
    let mut cfg = H3VideoVaeConfig::fasth3_8step();
    let layers = cfg.decoder_num_layers;
    cfg.decoder_num_layers = 1;
    H3VideoDecoder::load(cfg, &map).expect("load");
    let seen = for_every_index(
        &seen.lock().expect("lock"),
        "decoder.transformer_blocks",
        0,
        0..layers,
    );
    let mut problems = mismatches(&seen, &published);
    problems.extend(unrequested(&seen, &published, |k| {
        k.starts_with("decoder.") || k.starts_with("post_quant_conv.")
    }));
    assert_clean("vae", problems);
    assert!(
        seen.keys()
            .all(|k| !k.starts_with("encoder.") && !k.starts_with("quant_conv.")),
        "T2AV never reads the video encoder"
    );
    assert!(
        !published
            .keys()
            .any(|k| k.contains("norm_q") || k.contains("norm_k")),
        "the decoder's QK norm has no parameters"
    );
}

#[test]
fn dit_loaders_cover_the_published_transformer_and_nothing_else() {
    let _guard = heavy();
    let published = manifest(include_str!("manifests/transformer.json"));
    let cfg = H3TransformerConfig::fasth3_8step();
    let layers = cfg.num_layers;
    // A one-rung ladder keeps the host-side AdaLN GEMM small; the keys do not depend on it.
    let rung = |shift| H3Schedule::from_dmd_rungs(&[500], shift).expect("schedule");
    let schedule = H3JointSchedule {
        video: rung(10.0),
        audio: rung(3.0),
    };

    // Resident stack with the VSA gates, block 0 only; then block 49 for real.
    let (map, seen) = recording();
    let mut one = cfg.clone();
    one.num_layers = 1;
    H3Transformer::load(one, &map, &schedule, true).expect("load");
    let mut requested = for_every_index(
        &seen.lock().expect("lock"),
        "transformer_blocks",
        0,
        0..layers,
    );
    let (map, seen) = recording();
    Block::load(
        &map,
        &format!("transformer_blocks.{}", layers - 1),
        &cfg,
        true,
        &mut None,
    )
    .expect("last block");
    requested.extend(seen.lock().expect("lock").clone());
    // The text refiner, loaded on its own and dropped after one use.
    let (map, seen) = recording();
    H3TextRefiner::load(&cfg, &map).expect("refiner");
    requested.extend(seen.lock().expect("lock").clone());

    let mut problems = mismatches(&requested, &published);
    problems.extend(unrequested(&requested, &published, |_| true));
    assert_clean("transformer", problems);
    assert_eq!(published.len(), 688);
    assert!(dtypes(include_str!("manifests/transformer.json"))
        .values()
        .all(|d| d == "BF16"));

    // Dense mode must not pay for the gates, and the resident loaders must not
    // touch the AdaLN projections: only the table precompute streams them.
    let (map, seen) = recording();
    Block::load(&map, "transformer_blocks.7", &cfg, false, &mut None).expect("dense block");
    let dense = seen.lock().expect("lock").clone();
    assert!(
        dense
            .keys()
            .all(|k| !k.contains("to_gate_compress") && !k.contains("adaln_proj")),
        "{:?}",
        dense.keys().collect::<Vec<_>>()
    );
    let (map, seen) = recording();
    let mut one = cfg.clone();
    one.num_layers = 1;
    AdaLnTable::precompute(&one, &map, &schedule).expect("table");
    let table: Vec<String> = seen.lock().expect("lock").keys().cloned().collect();
    assert_eq!(
        table,
        [
            "norm_out.linear.bias",
            "norm_out.linear.weight",
            "time_embedder.linear_1.bias",
            "time_embedder.linear_1.weight",
            "time_embedder.linear_2.bias",
            "time_embedder.linear_2.weight",
            "transformer_blocks.0.adaln_proj.linear.bias",
            "transformer_blocks.0.adaln_proj.linear.weight",
        ]
    );
}

#[test]
fn text_encoder_reads_only_layers_present_in_shards_1_to_11() {
    let _guard = heavy();
    let published = manifest(include_str!("manifests/text_encoder.json"));
    let mut cfg = DecoderConfig::qwen3_vl_32b_text();
    let taps = 50usize;
    cfg.layers.truncate(2); // tap 1 of a 2-layer stack: layer 0 runs, no final norm
    let (map, seen) = recording();
    // The last vocabulary row, so the generated table has the published height.
    let last_id = (published[&cfg.embed_key][0] - 1) as u32;
    llm::hidden_states(&map, &cfg, &[last_id], &[0], &[true], &[1]).expect("forward");
    let seen = for_every_index(&seen.lock().expect("lock"), &cfg.layer_prefix, 0, 0..taps);
    // Every key tap 50 needs lives in shards 1..11: shards 12..14 need not be on disk.
    assert_clean("text_encoder", mismatches(&seen, &published));
    assert!(
        seen.keys()
            .all(|k| !k.ends_with("language_model.norm.weight")),
        "tap 50 is un-normed"
    );
    let problems = unrequested(&seen, &published, |k| {
        k == cfg.embed_key
            || k.strip_prefix("model.language_model.layers.")
                .and_then(|r| r.split('.').next())
                .and_then(|n| n.parse::<usize>().ok())
                .is_some_and(|n| n < taps)
    });
    assert_clean("text_encoder layers 0..49", problems);
    assert!(dtypes(include_str!("manifests/text_encoder.json"))
        .values()
        .all(|d| d == "BF16"));
}
