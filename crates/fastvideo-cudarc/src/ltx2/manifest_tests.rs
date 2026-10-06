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

/// LTX-2.3 (22B): every loader against `Lightricks/LTX-2.3`'s dev pack
/// (`ltx-2.3-22b-dev.safetensors`; the distilled packs have the same keys and
/// shapes) and against FastVideo's split of it, the `ltx23` tree on the
/// volume (`FastVideo/LTX-2.3-Distilled-Diffusers`). The split is derived from
/// the single file by the rules the manifest's `note` records, so both
/// layouts are judged against one set of real header names. The first two
/// GPU attempts on 2.3 died on keys (`prompt_adaln_single`, then
/// `decoder.mid_block.resnets.0.conv1.conv.weight`).
mod ltx23 {
    use super::*;

    use fastvideo_models::ltx2::config::{ltx2_23_22b, ltx2_23_22b_distilled, Ltx2Config};

    use crate::ltx2::keys::{connector_folder_alias, ltx_core_vae_key};
    use crate::ltx2::latent_upsampler::LatentUpsampler;

    const DIT_ROOT: &str = "model.diffusion_model.";

    fn single() -> Requests {
        manifest(include_str!("manifests/ltx23_single_file.json"))
    }

    fn cfg() -> Ltx2Config {
        ltx2_23_22b()
    }

    fn is_connector(k: &str) -> bool {
        k.contains("_embeddings_connector.")
    }

    /// `prefix.*` of the single file with the prefix stripped.
    fn strip(published: &Requests, prefix: &str) -> Requests {
        published
            .iter()
            .filter_map(|(k, v)| k.strip_prefix(prefix).map(|r| (r.to_string(), v.clone())))
            .collect()
    }

    /// FastVideo `transformer/`: the DiT root without the connectors.
    fn split_transformer(s: &Requests) -> Requests {
        strip(s, DIT_ROOT)
            .into_iter()
            .filter(|(k, _)| !is_connector(k))
            .collect()
    }

    /// FastVideo `text_embedding_projection/`: both connectors (the video one
    /// as plain `embeddings_connector`) and the two aggregate projections.
    fn split_connectors(s: &Requests) -> Requests {
        let mut out = Requests::new();
        for (k, v) in strip(s, DIT_ROOT) {
            if let Some(rest) = k.strip_prefix("video_embeddings_connector.") {
                out.insert(format!("embeddings_connector.{rest}"), v);
            } else if is_connector(&k) {
                out.insert(k, v);
            }
        }
        out.extend(strip(s, "text_embedding_projection."));
        out
    }

    #[test]
    fn the_split_tree_has_the_published_component_sizes() {
        // Tensor counts of the FastVideo folders' own headers (fetched
        // 2026-10-06 from FastVideo/LTX-2.3-Distilled-Diffusers@22b09fb).
        let s = single();
        assert_eq!(s.len(), 5947);
        assert_eq!(split_transformer(&s).len(), 4186);
        assert_eq!(split_connectors(&s).len(), 262);
        assert_eq!(strip(&s, "vae.").len(), 170);
        assert_eq!(strip(&s, "audio_vae.").len(), 102);
        assert_eq!(strip(&s, "vocoder.").len(), 1227);
        assert_eq!(
            manifest(include_str!("manifests/ltx23_spatial_upscaler_x2.json")).len(),
            72
        );
    }

    /// Globals plus the first and last block loaded for real, every other
    /// block by index substitution (as [`transformer_against`]).
    fn dit_against(
        layout: Layout,
        published: &Requests,
        owned: impl Fn(&str) -> bool,
    ) -> Vec<String> {
        let cfg = cfg().transformer;
        let keys = Keys::transformer(layout);
        let (map, seen) = recording();
        let last = cfg.num_layers - 1;
        drop(Ltx2Transformer::load_blocks(&map, &keys, &cfg, &[0, last]).expect("load"));
        let seen = seen.lock().expect("lock").clone();
        let block = |i: usize| keys.key(&format!("transformer_blocks.{i}."));
        let first: Requests = seen
            .iter()
            .filter(|(k, _)| k.starts_with(&block(0)))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let mut all = seen.clone();
        for i in 1..last {
            all.extend(
                first
                    .iter()
                    .map(|(k, v)| (k.replacen(&block(0), &block(i), 1), v.clone())),
            );
        }
        let mut problems = mismatches(&all, published);
        problems.extend(unrequested(&all, published, owned));
        problems
    }

    #[test]
    fn dev_dit_loads_from_the_single_file() {
        let _guard = heavy();
        let s = single();
        assert_clean(
            "2.3 dit (single file)",
            dit_against(Layout::SingleFile, &s, |k| {
                k.starts_with(DIT_ROOT) && !is_connector(k)
            }),
        );
    }

    #[test]
    fn distilled_dit_loads_from_the_split_transformer_folder() {
        let _guard = heavy();
        assert_clean(
            "2.3 dit (FastVideo transformer/)",
            dit_against(Layout::LtxCore, &split_transformer(&single()), |_| true),
        );
    }

    fn connectors_23(layout: Layout, published: &Requests) -> Vec<String> {
        let (map, seen) = recording();
        let cfg = cfg().connectors;
        let keys = Keys::connectors(layout);
        TextConnectors::load_with_projection(&map, &keys, &cfg, "unused_shared_projection")
            .expect("load");
        let seen = seen.lock().expect("lock").clone();
        // The FastVideo folder names the video connector `embeddings_connector`;
        // the loader asks for the ltx-core name and the folder's alias answers.
        let on_disk: Requests = seen
            .into_iter()
            .map(|(k, v)| match layout {
                Layout::LtxCore => (connector_folder_alias(&k).unwrap_or(k), v),
                _ => (k, v),
            })
            .collect();
        let mut problems = mismatches(&on_disk, published);
        problems.extend(unrequested(&on_disk, published, |k| match layout {
            Layout::SingleFile => is_connector(k) || k.starts_with("text_embedding_projection."),
            _ => true,
        }));
        problems
    }

    #[test]
    fn connectors_load_from_the_single_file() {
        let _guard = heavy();
        assert_clean(
            "2.3 connectors (single file)",
            connectors_23(Layout::SingleFile, &single()),
        );
    }

    /// What the HQ path runs (`load_connectors` with `--dit` the dev single
    /// file): `TextConnectors::load` itself, not the probe-free entry. Phase B
    /// died here on a probe for LTX-2.0's shared `aggregate_embed`.
    #[test]
    fn hq_connectors_load_from_the_dev_single_file() {
        let _guard = heavy();
        let (map, seen) = recording();
        TextConnectors::load(
            &map,
            &Keys::connectors(Layout::SingleFile),
            &cfg().connectors,
        )
        .expect("2.3 connectors from the single file");
        let seen = seen.lock().expect("lock").clone();
        for k in [
            "text_embedding_projection.video_aggregate_embed.weight",
            "text_embedding_projection.audio_aggregate_embed.weight",
        ] {
            assert!(seen.contains_key(k), "{k} not requested");
        }
        assert_clean("2.3 HQ connectors", mismatches(&seen, &single()));
    }

    #[test]
    fn connectors_load_from_the_split_text_embedding_projection() {
        let _guard = heavy();
        let s = single();
        let folder = split_connectors(&s);
        // The probe the pipeline runs on the real folder.
        assert!(folder.contains_key("video_aggregate_embed.weight"));
        assert_clean(
            "2.3 connectors (FastVideo folder)",
            connectors_23(Layout::LtxCore, &folder),
        );
    }

    /// The VAE asks for diffusers names; the ltx-core view of the folder
    /// answers them under the original ones.
    fn on_disk_vae(seen: Requests) -> Requests {
        seen.into_iter()
            .map(|(k, v)| (ltx_core_vae_key(&k, 8).unwrap_or(k), v))
            .collect()
    }

    #[test]
    fn video_decoder_loads_from_the_ltx_core_vae() {
        let _guard = heavy();
        let published = strip(&single(), "vae.");
        let (map, seen) = recording();
        VideoDecoder::load(&map, &cfg().vae).expect("load");
        let seen = on_disk_vae(seen.lock().expect("lock").clone());
        let mut problems = mismatches(&seen, &published);
        problems.extend(unrequested(&seen, &published, |k| {
            k.starts_with("decoder.") || k.starts_with("per_channel_statistics.")
        }));
        assert_clean("2.3 vae decoder", problems);
    }

    #[test]
    fn ltx_core_vae_names_cover_the_encoder_too() {
        // The image-conditioning encoder walks diffusers names by probing; every
        // ltx-core encoder key must be the image of one diffusers name.
        let published = strip(&single(), "vae.");
        let mut images = BTreeSet::new();
        for i in 0..4 {
            for r in 0..8 {
                for leaf in [
                    "conv1.conv.weight",
                    "conv1.conv.bias",
                    "conv2.conv.weight",
                    "conv2.conv.bias",
                ] {
                    images.insert(format!("encoder.down_blocks.{i}.resnets.{r}.{leaf}"));
                    images.insert(format!("encoder.mid_block.resnets.{r}.{leaf}"));
                }
            }
            for leaf in ["conv.conv.weight", "conv.conv.bias"] {
                images.insert(format!("encoder.down_blocks.{i}.downsamplers.0.{leaf}"));
            }
        }
        let mapped: BTreeSet<String> = images
            .iter()
            .filter_map(|k| ltx_core_vae_key(k, 8))
            .collect();
        let missing: Vec<_> = published
            .keys()
            .filter(|k| k.starts_with("encoder.down_blocks.") && !mapped.contains(*k))
            .collect();
        assert!(missing.is_empty(), "unmapped encoder keys: {missing:?}");
    }

    #[test]
    fn audio_vae_loads_from_the_ltx_core_folder() {
        let published = strip(&single(), "audio_vae.");
        let (map, seen) = recording();
        AudioDecoder::load(&map, &cfg().audio_vae).expect("load");
        AudioEncoder::load(&map, &cfg().audio_vae).expect("load");
        let seen = on_disk_vae(seen.lock().expect("lock").clone());
        let mut problems = mismatches(&seen, &published);
        problems.extend(unrequested(&seen, &published, |_| true));
        assert_clean("2.3 audio vae", problems);
    }

    /// The vocoder probes for optional tensors (biases, Snake, BWE), which a
    /// generated map would answer "yes" to; so this one holds zeros under
    /// the folder's real names and shapes, seen through the pipeline's
    /// HiFi-GAN view, and records anything it still has to invent.
    #[test]
    fn bwe_vocoder_loads_from_the_folder() {
        let published = strip(&single(), "vocoder.");
        let invented = Arc::new(Mutex::new(Requests::new()));
        let sink = invented.clone();
        let map = WeightMap::from_f32_tensors(
            published
                .iter()
                .map(|(k, shape)| (k.clone(), shape.clone(), vec![0.0; shape.iter().product()])),
        )
        .with_generator(move |key, shape| {
            sink.lock()
                .expect("lock")
                .insert(key.to_string(), shape.to_vec());
            vec![0.0; shape.iter().product()]
        });
        let map = crate::ltx2::keys::vocoder_view(map);
        Vocoder::load(&map, &cfg().vocoder).expect("load");
        let invented = invented.lock().expect("lock").clone();
        assert_clean(
            "2.3 vocoder",
            invented
                .iter()
                .map(|(k, s)| format!("loader asks for `{k}` {s:?}: not in the folder"))
                .collect(),
        );
    }

    #[test]
    fn spatial_upscaler_loads_from_the_x2_file() {
        let published = manifest(include_str!("manifests/ltx23_spatial_upscaler_x2.json"));
        let (map, seen) = recording();
        let ucfg = cfg().latent_upsampler.expect("2.3 has an upsampler");
        LatentUpsampler::load(&map, &ucfg).expect("load");
        let seen = seen.lock().expect("lock").clone();
        let mut problems = mismatches(&seen, &published);
        problems.extend(unrequested(&seen, &published, |_| true));
        assert_clean("2.3 spatial upscaler", problems);
    }

    /// The distilled LoRA (rank 384) the HQ recipe fuses at 0.25 / 0.5: every
    /// pair targets a DiT weight the dev pack has, in the shapes `B·A` needs,
    /// under the single file's names and under the split folder's.
    #[test]
    fn distilled_lora_targets_exist_in_both_layouts() {
        use fastvideo_models::ltx2::lora::{weight_key_aliases, weight_key_for_lora_a};
        let lora = manifest(include_str!("manifests/ltx23_distilled_lora_384.json"));
        let s = single();
        let split = split_transformer(&s);
        let mut pairs = 0;
        let mut problems = Vec::new();
        for (key, a) in &lora {
            let Some(stem) = weight_key_for_lora_a(key) else {
                continue;
            };
            pairs += 1;
            let b = lora.get(&format!("{stem}.lora_B.weight"));
            let aliases = weight_key_aliases(stem);
            let base_single = aliases.iter().find_map(|k| s.get(k));
            let base_split = aliases.iter().find_map(|k| split.get(k));
            match (b, base_single, base_split) {
                (Some(b), Some(w), Some(w2)) => {
                    let (rank, out_in) = (a[0], (w[0], w[1]));
                    // Rank 384, except the 32-wide gate logits (rank 32).
                    if w != w2 || rank > 384 || b[1] != rank || (b[0], a[1]) != out_in {
                        problems.push(format!("{stem}: A {a:?} B {b:?} base {w:?}"));
                    }
                }
                other => problems.push(format!("{stem}: B / single / split = {other:?}")),
            }
        }
        assert_eq!(pairs, 1660);
        assert_clean("2.3 distilled lora", problems);
    }

    #[test]
    fn distilled_and_dev_bundles_share_the_architecture() {
        let (dev, distilled) = (ltx2_23_22b(), ltx2_23_22b_distilled());
        assert_eq!(dev.transformer, distilled.transformer);
        assert_eq!(dev.vae, distilled.vae);
        assert_eq!(dev.connectors, distilled.connectors);
        assert!(dev.scheduler.use_dynamic_shifting);
        assert!(!distilled.scheduler.use_dynamic_shifting);
    }
}
