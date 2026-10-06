//! The Wan DiT loader against the published checkpoints' tensor headers — no
//! GPU, no weights (the method of `ltx2/manifest_tests.rs`).
//!
//! `manifests/*.json` are the safetensors headers of
//! `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` and `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
//! (`transformer/`; the A14B's `transformer_2/` is the same set), fetched
//! with HTTP range requests and stripped to `key → [dtype, shape]`, plus each
//! repo's `transformer/config.json`. The loader runs at the production
//! config with one block (a recording generator answers every key with
//! zeros); the other blocks are block 0 with the index substituted, which is
//! sound because the per-block loader does not look at the index.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use fastvideo_models::wan::WanVideoArchConfig;

use super::transformer::WanTransformer3D;
use super::weights::WeightMap;

type Requests = BTreeMap<String, Vec<usize>>;

struct Manifest {
    tensors: Requests,
    config: serde_json::Value,
}

fn manifest(json: &str) -> Manifest {
    let doc: serde_json::Value = serde_json::from_str(json).expect("manifest is JSON");
    for field in ["repo", "revision", "files", "config"] {
        assert!(doc.get(field).is_some(), "manifest records its {field}");
    }
    let tensors = doc["tensors"]
        .as_object()
        .expect("tensors")
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
        .collect();
    Manifest {
        tensors,
        config: doc["config"].clone(),
    }
}

/// One lock: the 14B block and globals are ~2.5 GB of f32 zeros.
fn heavy() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The preset's architecture is the Hub `config.json`'s.
fn config_matches(cfg: &WanVideoArchConfig, hub: &serde_json::Value) {
    let u = |k: &str| hub[k].as_u64().unwrap_or_else(|| panic!("config.{k}")) as usize;
    assert_eq!(cfg.num_attention_heads, u("num_attention_heads"));
    assert_eq!(cfg.attention_head_dim, u("attention_head_dim"));
    assert_eq!(cfg.ffn_dim, u("ffn_dim"));
    assert_eq!(cfg.num_layers, u("num_layers"));
    assert_eq!(cfg.in_channels, u("in_channels"));
    assert_eq!(cfg.out_channels, u("out_channels"));
    assert_eq!(cfg.text_dim, u("text_dim"));
    assert_eq!(cfg.freq_dim, u("freq_dim"));
    assert_eq!(cfg.rope_max_seq_len, u("rope_max_seq_len"));
    let patch: Vec<usize> = hub["patch_size"]
        .as_array()
        .expect("patch_size")
        .iter()
        .map(|v| v.as_u64().expect("patch") as usize)
        .collect();
    assert_eq!(cfg.patch_size.to_vec(), patch);
    assert!((f64::from(cfg.eps) - hub["eps"].as_f64().expect("eps")).abs() < 1e-12);
    assert_eq!(hub["qk_norm"], "rms_norm_across_heads");
    assert_eq!(hub["cross_attn_norm"], true);
    assert!(hub["image_dim"].is_null() && hub["added_kv_proj_dim"].is_null());
    assert!(cfg.image_dim.is_none() && cfg.added_kv_proj_dim.is_none());
}

/// Every key the loader asks for, all blocks, against the manifest.
fn problems(preset: &str, published: &Requests) -> Vec<String> {
    let full = WanVideoArchConfig::from_preset(preset);
    let one = WanVideoArchConfig {
        num_layers: 1,
        ..full.clone()
    };
    let seen = Arc::new(Mutex::new(Requests::new()));
    let sink = seen.clone();
    let map = WeightMap::generated(move |key, shape| {
        sink.lock()
            .expect("recorder")
            .insert(key.to_string(), shape.to_vec());
        vec![0.0; shape.iter().product()]
    });
    drop(WanTransformer3D::load(one, &map).expect("load"));
    let seen = seen.lock().expect("recorder").clone();
    let mut all = Requests::new();
    for (k, v) in &seen {
        // The VSA gate is probed with `contains`, which a generated map
        // always answers yes; the base checkpoints have none.
        if k.contains("to_gate_compress") {
            continue;
        }
        match k.strip_prefix("blocks.0.") {
            Some(rest) => {
                for i in 0..full.num_layers {
                    all.insert(format!("blocks.{i}.{rest}"), v.clone());
                }
            }
            None => {
                all.insert(k.clone(), v.clone());
            }
        }
    }
    let mut out: Vec<String> = all
        .iter()
        .filter_map(|(k, want)| match published.get(k) {
            None => Some(format!(
                "loader asks for `{k}` {want:?}: not in the checkpoint"
            )),
            Some(have) if have != want => Some(format!(
                "`{k}`: loader expects {want:?}, checkpoint has {have:?}"
            )),
            Some(_) => None,
        })
        .collect();
    out.extend(
        published
            .keys()
            .filter(|k| !all.contains_key(*k))
            .map(|k| format!("checkpoint key `{k}` is never loaded")),
    );
    out
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
fn wan21_t2v_1_3b_base_loads_from_the_published_transformer() {
    let _guard = heavy();
    let m = manifest(include_str!("manifests/wan21_t2v_1_3b_transformer.json"));
    assert_eq!(m.tensors.len(), 825);
    config_matches(&WanVideoArchConfig::from_preset("wan_t2v_1_3b"), &m.config);
    assert_clean("wan 1.3B", problems("wan_t2v_1_3b", &m.tensors));
}

#[test]
fn wan22_a14b_experts_load_from_the_published_transformers() {
    let _guard = heavy();
    let m = manifest(include_str!("manifests/wan22_t2v_a14b_transformer.json"));
    assert_eq!(m.tensors.len(), 1095);
    let cfg = WanVideoArchConfig::from_preset("wan_2_2_t2v_a14b");
    config_matches(&cfg, &m.config);
    // model_index.json `boundary_ratio`.
    assert_eq!(cfg.boundary_ratio, Some(0.875));
    // The low-noise expert is the same architecture under the same names
    // (manifest note), so one check covers `transformer_2/` too.
    assert_clean("wan 2.2 A14B", problems("wan_2_2_t2v_a14b", &m.tensors));
    // Parameter count the swap plan sizes from = the checkpoint's.
    let params: u64 = m
        .tensors
        .values()
        .map(|s| s.iter().product::<usize>() as u64)
        .sum();
    assert_eq!(
        fastvideo_models::wan::moe::expert_bf16_bytes(&cfg),
        2 * params
    );
}
