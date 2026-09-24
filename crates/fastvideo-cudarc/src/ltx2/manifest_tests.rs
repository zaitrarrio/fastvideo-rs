//! Every loader against the published checkpoints' key manifests — no GPU, no
//! weights.
//!
//! A key or shape mismatch is the one failure that can always be found without
//! hardware, and the first LTX-2 stage to reach a GPU died on exactly that. So:
//! `manifests/*.json` are the safetensors *headers* of the real files (fetched
//! with HTTP range requests, stripped to `key → [dtype, shape]`, repo and
//! revision recorded inside), and each test runs the real loader at the
//! **production** config against a recording [`WeightMap::generated`] closure,
//! which sees every `(key, expected shape)` the loader asks for. Then:
//!
//! * every requested key must exist in the manifest with exactly that shape;
//! * where the loader is meant to consume a whole component, no manifest key
//!   of that component may go unrequested.
//!
//! The generator returns zeros (the allocator hands out untouched pages), and
//! the heavy tests take one lock so their peaks do not stack. The DiT would be
//! 19B parameters, so blocks 0 and 47 are loaded for real and the other 46 are
//! checked by substituting the block index — the loader's per-block code does
//! not depend on it. Both layouts go through [`Keys`], so the single-file
//! rename table is itself what is tested.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use fastvideo_models::ltx2::config::ltx2_19b_distilled;

use crate::llm::{self, DecoderConfig};
use crate::wan::weights::WeightMap;

use super::audio_vae::AudioDecoder;
use super::audio_vae::AudioEncoder;
use super::keys::{Keys, Layout};
use super::text::TextConnectors;
use super::transformer::Ltx2Transformer;
use super::vae::VideoDecoder;
use super::vocoder::Vocoder;

type Requests = BTreeMap<String, Vec<usize>>;

/// `key → shape` of one published component.
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
            let shape = v[1]
                .as_array()
                .expect("shape")
                .iter()
                .map(|d| d.as_u64().expect("dim") as usize)
                .collect();
            (k.clone(), shape)
        })
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

/// Requested keys that are absent or have another shape, as readable lines.
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

/// Published keys accepted by `owned` that the loader never asked for.
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
fn audio_vae_loader_asks_for_exactly_the_published_decoder() {
    let published = manifest(include_str!("manifests/audio_vae.json"));
    let (map, seen) = recording();
    AudioDecoder::load(&map, &ltx2_19b_distilled().audio_vae).expect("load");
    let seen = seen.lock().expect("lock").clone();
    let mut problems = mismatches(&seen, &published);
    problems.extend(unrequested(&seen, &published, |k| {
        k.starts_with("decoder.") || k.starts_with("latents_")
    }));
    assert_clean("audio_vae", problems);
}

#[test]
fn audio_encoder_loader_asks_for_exactly_the_published_encoder() {
    let published = manifest(include_str!("manifests/audio_vae.json"));
    let (map, seen) = recording();
    AudioEncoder::load(&map, &ltx2_19b_distilled().audio_vae).expect("load");
    let seen = seen.lock().expect("lock").clone();
    let mut problems = mismatches(&seen, &published);
    problems.extend(unrequested(&seen, &published, |k| {
        k.starts_with("encoder.") || k.starts_with("latents_")
    }));
    assert_clean("audio_vae encoder", problems);
}

#[test]
fn vocoder_loader_asks_for_exactly_the_published_file() {
    let published = manifest(include_str!("manifests/vocoder.json"));
    let (map, seen) = recording();
    Vocoder::load(&map, &ltx2_19b_distilled().vocoder).expect("load");
    let seen = seen.lock().expect("lock").clone();
    let mut problems = mismatches(&seen, &published);
    problems.extend(unrequested(&seen, &published, |_| true));
    assert_clean("vocoder", problems);
}

#[test]
fn video_vae_loader_asks_for_exactly_the_published_decoder() {
    let _guard = heavy();
    let published = manifest(include_str!("manifests/vae.json"));
    let (map, seen) = recording();
    VideoDecoder::load(&map, &ltx2_19b_distilled().vae).expect("load");
    let seen = seen.lock().expect("lock").clone();
    let mut problems = mismatches(&seen, &published);
    problems.extend(unrequested(&seen, &published, |k| {
        k.starts_with("decoder.") || k == "latents_mean" || k == "latents_std"
    }));
    assert_clean("vae", problems);
}

/// The connectors under one layout: the diffusers folder, or the single file
/// through the rename view. In the single file they share a root with the DiT,
/// so "owned" is the two connector subtrees plus the text projection.
fn connectors_against(layout: Layout, published: &Requests) -> Vec<String> {
    let (map, seen) = recording();
    // A generated map holds no tensors, so the single-file probe for the text
    // projection cannot succeed; resolve it against the manifest instead and
    // load the rest through the real loader.
    let keys = Keys::connectors(layout);
    let cfg = ltx2_19b_distilled().connectors;
    let mut problems = Vec::new();
    match layout {
        Layout::Diffusers => {
            TextConnectors::load(&map, &keys, &cfg).expect("load");
        }
        Layout::LtxCore => unreachable!("2.0 connector manifest is not the 2.3 folder layout"),
        Layout::SingleFile => {
            let candidates = Keys::text_proj_in_candidates();
            let found: Vec<_> = candidates
                .iter()
                .filter(|c| published.contains_key(&format!("{c}.weight")))
                .collect();
            if found.len() != 1 {
                problems.push(format!("text projection: {found:?} of the candidates {candidates:?} exist in the single file"));
            }
            TextConnectors::load_with_projection(
                &map,
                &keys,
                &cfg,
                found.first().map_or("text_proj_in", |s| s.as_str()),
            )
            .expect("load");
        }
    }
    let seen = seen.lock().expect("lock").clone();
    problems.extend(mismatches(&seen, published));
    problems.extend(unrequested(&seen, published, |k| match layout {
        Layout::Diffusers | Layout::LtxCore => true,
        Layout::SingleFile => {
            k.contains("_embeddings_connector.") || k.starts_with("text_embedding_projection.")
        }
    }));
    problems
}

#[test]
fn connector_loader_matches_the_diffusers_folder() {
    let _guard = heavy();
    assert_clean(
        "connectors (diffusers)",
        connectors_against(
            Layout::Diffusers,
            &manifest(include_str!("manifests/connectors.json")),
        ),
    );
}

#[test]
fn connector_loader_matches_the_single_file_through_the_rename_view() {
    let _guard = heavy();
    assert_clean(
        "connectors (single file)",
        connectors_against(
            Layout::SingleFile,
            &manifest(include_str!("manifests/single_file.json")),
        ),
    );
}

/// Globals plus blocks 0 and 47 loaded for real; every other block's keys are
/// block 0's with the index substituted.
fn transformer_against(layout: Layout, published: &Requests) -> Vec<String> {
    let cfg = ltx2_19b_distilled().transformer;
    let keys = Keys::transformer(layout);
    let (map, seen) = recording();
    let last = cfg.num_layers - 1;
    drop(Ltx2Transformer::load_blocks(&map, &keys, &cfg, &[0, last]).expect("load"));
    let seen = seen.lock().expect("lock").clone();

    let block = |i: usize| keys.key(&format!("transformer_blocks.{i}."));
    let of_block = |i: usize| -> Requests {
        seen.iter()
            .filter(|(k, _)| k.starts_with(&block(i)))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    };
    let (first, final_block) = (of_block(0), of_block(last));
    let mut problems = Vec::new();
    // The loader must treat every block alike for substitution to be sound.
    let renumbered: Requests = first
        .iter()
        .map(|(k, v)| (k.replacen(&block(0), &block(last), 1), v.clone()))
        .collect();
    if renumbered != final_block {
        problems.push(format!(
            "block 0 and block {last} request different keys or shapes"
        ));
    }
    let mut all = seen.clone();
    for i in 1..last {
        all.extend(
            first
                .iter()
                .map(|(k, v)| (k.replacen(&block(0), &block(i), 1), v.clone())),
        );
    }
    problems.extend(mismatches(&all, published));
    problems.extend(unrequested(&all, published, |k| match layout {
        Layout::Diffusers | Layout::LtxCore => true,
        // Everything under the DiT root that is not a connector.
        Layout::SingleFile => {
            k.starts_with("model.diffusion_model.") && !k.contains("_embeddings_connector.")
        }
    }));
    problems
}

#[test]
fn transformer_loader_matches_the_diffusers_shards() {
    let _guard = heavy();
    assert_clean(
        "transformer (diffusers)",
        transformer_against(
            Layout::Diffusers,
            &manifest(include_str!("manifests/transformer.json")),
        ),
    );
}

#[test]
fn transformer_loader_matches_the_single_file_through_the_rename_view() {
    let _guard = heavy();
    assert_clean(
        "transformer (single file)",
        transformer_against(
            Layout::SingleFile,
            &manifest(include_str!("manifests/single_file.json")),
        ),
    );
}

/// The two layouts are the same tensors under two names: the rename view must
/// be a bijection between the diffusers DiT + connectors and the single file's
/// DiT root + text projection, shapes included.
#[test]
fn the_rename_view_maps_the_diffusers_layout_onto_the_single_file() {
    let single = manifest(include_str!("manifests/single_file.json"));
    let mut renamed = Requests::new();
    for (k, v) in manifest(include_str!("manifests/transformer.json")) {
        renamed.insert(Keys::transformer(Layout::SingleFile).key(&k), v);
    }
    for (k, v) in manifest(include_str!("manifests/connectors.json")) {
        let name = match k.strip_prefix("text_proj_in.") {
            Some(rest) => format!("text_embedding_projection.aggregate_embed.{rest}"),
            None => Keys::connectors(Layout::SingleFile).key(&k),
        };
        renamed.insert(name, v);
    }
    let mut problems = mismatches(&renamed, &single);
    problems.extend(unrequested(&renamed, &single, |k| {
        k.starts_with("model.diffusion_model.") || k.starts_with("text_embedding_projection.")
    }));
    assert_clean("rename view", problems);
}

/// Gemma through the streaming decoder: one token through layer 0 records a
/// whole layer's keys and the embedding; the other 47 layers follow by
/// substitution and the final norm by name. (With a generated map the decoder
/// sizes the embedding table from the ids, so only its width is comparable.)
#[test]
fn gemma_loader_asks_for_the_published_language_model_keys() {
    let _guard = heavy();
    let published = manifest(include_str!("manifests/gemma.json"));
    let cfg = DecoderConfig::gemma3_12b_text();
    let (map, seen) = recording();
    llm::hidden_states(&map, &cfg, &[0], &[0], &[true], &[1]).expect("one token through layer 0");
    let mut seen = seen.lock().expect("lock").clone();

    let mut problems = Vec::new();
    let embed = seen
        .remove(&cfg.embed_key)
        .expect("the embedding was requested");
    match published.get(&cfg.embed_key) {
        Some(have) if have.len() == 2 && have[1] == embed[1] => {}
        other => problems.push(format!(
            "embedding `{}`: loader expects [vocab, {}], checkpoint has {other:?}",
            cfg.embed_key, embed[1]
        )),
    }
    let layer = |i: usize| format!("{}.{i}.", cfg.layer_prefix);
    let first: Requests = seen
        .iter()
        .filter(|(k, _)| k.starts_with(&layer(0)))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    assert_eq!(
        first.len(),
        seen.len(),
        "tap 1 reads layer 0 and nothing else: {:?}",
        seen.keys().collect::<Vec<_>>()
    );
    let mut all = Requests::new();
    for i in 0..cfg.num_layers() {
        all.extend(
            first
                .iter()
                .map(|(k, v)| (k.replacen(&layer(0), &layer(i), 1), v.clone())),
        );
    }
    all.insert(cfg.final_norm_key.clone(), vec![cfg.hidden]);
    problems.extend(mismatches(&all, &published));
    all.insert(cfg.embed_key.clone(), Vec::new());
    problems.extend(unrequested(&all, &published, |k| {
        k.starts_with("language_model.")
    }));
    assert_clean("gemma", problems);
}

/// The manifests themselves: the sizes docs/ports/ltx2.md quotes.
#[test]
fn manifests_hold_the_published_tensor_counts() {
    let count = |json: &str| manifest(json).len();
    assert_eq!(count(include_str!("manifests/audio_vae.json")), 102);
    assert_eq!(count(include_str!("manifests/vocoder.json")), 194);
    assert_eq!(count(include_str!("manifests/vae.json")), 184);
    assert_eq!(count(include_str!("manifests/connectors.json")), 59);
    assert_eq!(count(include_str!("manifests/transformer.json")), 3510);
    assert_eq!(count(include_str!("manifests/single_file.json")), 4052);
    let families: BTreeSet<String> = manifest(include_str!("manifests/single_file.json"))
        .keys()
        .map(|k| k.split('.').next().unwrap_or("").to_string())
        .collect();
    assert_eq!(
        families.into_iter().collect::<Vec<_>>(),
        [
            "audio_vae",
            "model",
            "text_embedding_projection",
            "vae",
            "vocoder"
        ]
    );
}
