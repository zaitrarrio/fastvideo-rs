//! A decoder-only text encoder against transformers, on the oracle's own
//! tokens.
//!
//! MiniMax-H3 conditions on hidden states of Qwen3-VL-32B and LTX-2 on
//! Gemma-3-12B; both run through `fastvideo_cudarc::llm`. The reference file
//! carries the token ids it was produced from, so this stage judges the model
//! arithmetic alone — a tokenizer or chat-template difference cannot hide in
//! it, and cannot be blamed for it either.
//!
//! Oracle file (all float32, the only dtype `st` reads):
//!
//! | key          | shape            | meaning                                   |
//! |--------------|------------------|-------------------------------------------|
//! | `input_ids`  | `[S]`            | token ids (exact as f32: vocab < 2^24)     |
//! | `positions`  | `[S]`            | rotary positions                           |
//! | `attend`     | `[S]`            | 1 where the position may be a key, else 0  |
//! | `hidden_<k>` | `[1, S, hidden]` | `output_hidden_states[k]`, one per tap     |
//!
//! Rows where `attend` is 0 are padding: what a model computes there is
//! unspecified, so they are left out of the comparison on both sides.

use std::path::Path;

use fastvideo_cudarc::llm::{self, DecoderConfig};
use fastvideo_cudarc::wan::weights::WeightMap;
use serde_json::json;

use crate::metrics::diff;
use crate::model::measure;
use crate::report::{Report, StageResult};
use crate::st;

fn family(name: &str) -> anyhow::Result<DecoderConfig> {
    match name {
        "qwen3-vl-32b" => Ok(DecoderConfig::qwen3_vl_32b_text()),
        "gemma3-12b" => Ok(DecoderConfig::gemma3_12b_text()),
        other => anyhow::bail!("unknown llm family '{other}' (qwen3-vl-32b|gemma3-12b)"),
    }
}

/// A `[S]` float tensor of small non-negative integers, as integers.
fn ints(t: &st::F32Tensor, name: &str) -> anyhow::Result<Vec<u32>> {
    t.data
        .iter()
        .map(|&v| {
            if v < 0.0 || v.fract() != 0.0 || v > 16_777_216.0 {
                anyhow::bail!("{name}: {v} is not an id");
            }
            Ok(v as u32)
        })
        .collect()
}

pub fn run(
    report: &mut Report,
    weights: &Path,
    family_name: &str,
    layer_prefix: Option<&str>,
    oracle: &Path,
    device: &str,
    max_rel: f64,
) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    let mut cfg = family(family_name)?;
    let mut orc = st::load(oracle)?;
    let ids = ints(&st::take(&mut orc, "input_ids", oracle)?, "input_ids")?;
    let positions = ints(&st::take(&mut orc, "positions", oracle)?, "positions")?;
    let attend: Vec<bool> = st::take(&mut orc, "attend", oracle)?.data.iter().map(|&v| v != 0.0).collect();
    let mut taps: Vec<(usize, st::F32Tensor)> = orc
        .into_iter()
        .filter_map(|(k, t)| k.strip_prefix("hidden_").and_then(|n| n.parse().ok()).map(|n| (n, t)))
        .collect();
    taps.sort_by_key(|(k, _)| *k);
    if taps.is_empty() {
        return Err(anyhow::anyhow!("{}: no hidden_<k> tensors", oracle.display()).into());
    }

    let map = WeightMap::open(weights)?;
    // Checkpoints disagree on where the language model sits (`model.language_model`,
    // `language_model.model`, bare `model`); take the caller's word, else probe.
    let probe = |p: &str| map.has_tensor(&format!("{p}.0.input_layernorm.weight"));
    let prefix = match layer_prefix {
        Some(p) => p.to_string(),
        None => [cfg.layer_prefix.as_str(), "model.language_model.layers", "language_model.model.layers", "model.layers"]
            .into_iter()
            .find(|p| probe(p))
            .ok_or_else(|| anyhow::anyhow!("no decoder layers found under {}", weights.display()))?
            .to_string(),
    };
    let root = prefix.strip_suffix(".layers").unwrap_or(&prefix).to_string();
    cfg.layer_prefix = prefix;
    cfg.embed_key = format!("{root}.embed_tokens.weight");
    cfg.final_norm_key = format!("{root}.norm.weight");
    let store = map.lazy().expect("opened lazily");
    report.set(
        "model",
        json!({
            "family": family_name,
            "layer_prefix": cfg.layer_prefix,
            "layers": cfg.num_layers(),
            "checkpoint_gib": store.bytes_with_prefix("") as f64 / f64::from(1u32 << 30),
            "tokens": ids.len(),
            "attended": attend.iter().filter(|a| **a).count(),
            "taps": taps.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        }),
    );

    let want: Vec<usize> = taps.iter().map(|(k, _)| *k).collect();
    let (ours, seconds) = measure(report, "forward", || {
        let out = llm::hidden_states(&map, &cfg, &ids, &positions, &attend, &want)?;
        out.iter().map(|t| Ok(t.host_cow()?.into_owned())).collect::<anyhow::Result<Vec<_>>>()
    })?;
    report.note("forward", json!({"seconds": seconds, "layers_run": want.iter().max()}));

    let h = cfg.hidden;
    let keep = |v: &[f32]| -> Vec<f32> {
        v.chunks_exact(h).zip(&attend).filter(|(_, a)| **a).flat_map(|(row, _)| row.iter().copied()).collect()
    };
    // Every tap's metric lands before the first gate can stop the stage: the
    // layer at which the error starts to grow is the diagnosis.
    let diffs: Vec<_> = taps
        .iter()
        .zip(&ours)
        .map(|((k, t), got)| {
            if t.data.len() != got.len() {
                anyhow::bail!("hidden_{k}: oracle has {} values, we produced {}", t.data.len(), got.len());
            }
            Ok((*k, diff(&keep(got), &keep(&t.data))))
        })
        .collect::<anyhow::Result<_>>()?;
    report.set("metrics", diffs.iter().map(|(k, d)| (format!("hidden_{k}"), d.to_json())).collect::<serde_json::Map<_, _>>());
    for (k, d) in &diffs {
        report.check(&format!("hidden_{k}"), d.within(max_rel), d.to_json(), json!({"rel_l2": max_rel}))?;
    }
    Ok(())
}
