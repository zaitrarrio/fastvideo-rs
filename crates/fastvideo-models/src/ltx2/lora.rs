//! Distilled LoRA fusion from `ltx_core/loader/fuse_loras.py` on
//! NVlabs/Sana `sol-engine` (`models/ltx25/GB200/ltx_src`).
//!
//! Each `{stem}.lora_A.weight` / `{stem}.lora_B.weight` pair updates
//! `{stem}.weight`. `A` is `[rank, in]`, `B` is `[out, rank]`. The published
//! product is `(B * strength) @ A`, added onto the base weight. This loader
//! accumulates that product in f32, which is the dtype the DiT loader
//! materializes. LTX-2.3 uses one file at 0.25 on stage 1 and 0.5 on stage 2
//! (`models/ltx23.toml`, `run_ltx23_common.sh`). LTX-2.5:
//! * the distilled two-stage (`ltx_pipelines/distilled.py`, RTX5090
//!   `run_ltx25_gpu.sh` passes no `--lora`) fuses nothing on either stage;
//! * the dev two-stage (`ti2vid_two_stages.py:147-151`) fuses
//!   `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` on stage 2 only, at
//!   `DEFAULT_LORA_STRENGTH = 1.0` (`utils/args.py:166`,
//!   `ti2vid_two_stages_mgpu.py:77`);
//! * the refiners (`ltx2.5-refiner/GB200/refiner_head_cp.py:336-358`,
//!   `Sol-H3-Spark/runtime/stage2_ops/models.py:76`) fuse it at 0.8 onto the
//!   dev transformer — [`REFINER_STRENGTH`].

use super::config::Ltx2ModelVersion;
use super::pisa::{STAGE1_LORA_STRENGTH, STAGE2_LORA_STRENGTH};
use super::sol::LORA_STRENGTH;

/// Distilled LoRA strength of the stage-2 refiners (GB200 refiner, Spark).
pub const REFINER_STRENGTH: f32 = LORA_STRENGTH as f32;

/// Stage-2 distilled LoRA strength of the LTX-2.5 dev two-stage
/// (`DEFAULT_LORA_STRENGTH`, `ltx_pipelines/utils/args.py:166`).
pub const LTX25_DEV_STAGE2_STRENGTH: f32 = 1.0;

/// `ltx-2.3-22b-distilled-lora-384-1.1.safetensors`, then the unsuffixed name.
pub const LTX23_LORA_NAMES: &[&str] = &[
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
    "ltx-2.3-22b-distilled-lora-384.safetensors",
];

/// Stage-2 refiner adapter. Stage 1 of a 2.5 two-stage run stays unfused.
pub const LTX25_LORA_NAMES: &[&str] = &["ltx-2.5-22b-distilled-lora-450-bf16.safetensors"];

/// The LTX-2.5 IC-LoRA "Ingredients" (`Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients`
/// @ `12040e4`): a reference sheet in context. `ICLoraPipeline`
/// (`ltx_pipelines/ic_lora.py`) fuses it into stage 1 only (`DiffusionStage`
/// built with `loras=[…]`); stage 2 runs the plain distilled DiT. Rank 128 on
/// every block's `attn1` / `attn2` q/k/v/out and feed-forward (model card).
pub const LTX25_INGREDIENTS_FILE: &str = "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors";

/// Where the weight volumes keep it (`weights/<dir>/<file>`).
pub const LTX25_INGREDIENTS_DIR: &str = "ltx25-ic-lora-ingredients";

/// `read_lora_reference_downscale_factor` / `…_temporal_scale_factor`
/// (`ltx_pipelines/iclora_utils.py`): an integer in the safetensors
/// `__metadata__`, 1 when absent. An unparsable value is 1 too, as the
/// reference's `except Exception` fallback.
pub fn reference_scale(metadata: &std::collections::HashMap<String, String>, key: &str) -> usize {
    metadata
        .get(key)
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(1)
}

/// `(stage1, stage2)` strengths of a *dev* two-stage generate. `None` for 2.0,
/// which has no distilled LoRA in the sol-engine profiles. Distilled
/// checkpoints never fuse (the adapter is baked in), and the refiners use
/// [`REFINER_STRENGTH`] instead.
pub fn stage_strengths(version: Ltx2ModelVersion) -> Option<(f32, f32)> {
    match version {
        Ltx2ModelVersion::V23 => Some((STAGE1_LORA_STRENGTH as f32, STAGE2_LORA_STRENGTH as f32)),
        Ltx2ModelVersion::V25 => Some((0.0, LTX25_DEV_STAGE2_STRENGTH)),
        Ltx2ModelVersion::V20 => None,
    }
}

pub fn file_names(version: Ltx2ModelVersion) -> &'static [&'static str] {
    match version {
        Ltx2ModelVersion::V23 => LTX23_LORA_NAMES,
        Ltx2ModelVersion::V25 => LTX25_LORA_NAMES,
        Ltx2ModelVersion::V20 => &[],
    }
}

/// `{stem}.lora_A.weight` → the stem whose `.weight` is fused.
pub fn weight_key_for_lora_a(key: &str) -> Option<&str> {
    key.strip_suffix(".lora_A.weight")
}

/// On-disk weight key, plus the `diffusion_model.` / `model.diffusion_model.`
/// spellings the single-file and Comfy exports use for the same module, plus
/// the diffusers name of a DiT module the original (`ltx-core`) naming calls
/// otherwise (`adaln_single` → `time_embed`, `patchify_proj` → `proj_in`, the
/// `av_ca_*_adaln_single` embedders, the prompt AdaLNs): the distilled LoRA
/// speaks the original names, `LTX-2.5-Diffusers/transformer_full` the
/// diffusers ones.
pub fn weight_key_aliases(stem: &str) -> Vec<String> {
    let mut out = vec![format!("{stem}.weight")];
    let rest = stem
        .strip_prefix("model.diffusion_model.")
        .or_else(|| stem.strip_prefix("diffusion_model."));
    if let Some(rest) = rest {
        push_unique(&mut out, format!("{rest}.weight"));
        push_unique(&mut out, format!("diffusion_model.{rest}.weight"));
        push_unique(&mut out, format!("model.diffusion_model.{rest}.weight"));
    } else {
        push_unique(&mut out, format!("diffusion_model.{stem}.weight"));
        push_unique(&mut out, format!("model.diffusion_model.{stem}.weight"));
    }
    let bare = rest.unwrap_or(stem);
    let diffusers = diffusers_transformer_name(bare);
    if diffusers != bare {
        push_unique(&mut out, format!("{diffusers}.weight"));
    }
    out
}

/// The diffusers spelling of an original (`ltx-core`) DiT module path,
/// segment by segment (diffusers' `convert_ltx2_to_diffusers.py`, the
/// inverse of `ltx2/keys.rs`'s transformer renames).
pub fn diffusers_transformer_name(original: &str) -> String {
    original
        .split('.')
        .map(|seg| match seg {
            "patchify_proj" => "proj_in",
            "audio_patchify_proj" => "audio_proj_in",
            "adaln_single" => "time_embed",
            "audio_adaln_single" => "audio_time_embed",
            "prompt_adaln_single" => "prompt_adaln",
            "audio_prompt_adaln_single" => "audio_prompt_adaln",
            "av_ca_video_scale_shift_adaln_single" => "av_cross_attn_video_scale_shift",
            "av_ca_audio_scale_shift_adaln_single" => "av_cross_attn_audio_scale_shift",
            "av_ca_a2v_gate_adaln_single" => "av_cross_attn_video_a2v_gate",
            "av_ca_v2a_gate_adaln_single" => "av_cross_attn_audio_v2a_gate",
            other => other,
        })
        .collect::<Vec<_>>()
        .join(".")
}

fn push_unique(out: &mut Vec<String>, key: String) {
    if !out.iter().any(|have| have == &key) {
        out.push(key);
    }
}

/// `weight += (B * strength) @ A` with `A: [rank, inn]`, `B: [out, rank]`.
pub fn fuse_into(
    weight: &mut [f32],
    out: usize,
    inn: usize,
    b: &[f32],
    a: &[f32],
    strength: f32,
) -> Result<(), String> {
    if !strength.is_finite() {
        return Err("lora strength must be finite".into());
    }
    if strength == 0.0 {
        return Ok(());
    }
    if out == 0 || inn == 0 || weight.len() != out * inn {
        return Err(format!("lora weight len {} != {out}x{inn}", weight.len()));
    }
    if b.is_empty() || b.len() % out != 0 {
        return Err(format!(
            "lora B len {} is not a multiple of out {out}",
            b.len()
        ));
    }
    let rank = b.len() / out;
    if a.len() != rank * inn {
        return Err(format!("lora A len {} != rank {rank} * in {inn}", a.len()));
    }
    for o in 0..out {
        let row = &mut weight[o * inn..(o + 1) * inn];
        for r in 0..rank {
            let scale = strength * b[o * rank + r];
            if scale == 0.0 {
                continue;
            }
            let a_row = &a[r * inn..(r + 1) * inn];
            for (dst, &av) in row.iter_mut().zip(a_row) {
                *dst += scale * av;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_matches_b_times_strength_matmul_a() {
        // out=2, inn=2, rank=1. B = [2, 3], A = [1, 0], strength 0.5.
        // delta row0 = 1 * [1, 0], row1 = 1.5 * [1, 0].
        let mut weight = vec![10.0, 20.0, 30.0, 40.0];
        fuse_into(&mut weight, 2, 2, &[2.0, 3.0], &[1.0, 0.0], 0.5).unwrap();
        assert_eq!(weight, vec![11.0, 20.0, 31.5, 40.0]);
    }

    #[test]
    fn zero_strength_leaves_the_weight() {
        let mut weight = vec![1.0, 2.0];
        fuse_into(&mut weight, 1, 2, &[4.0], &[1.0, 1.0], 0.0).unwrap();
        assert_eq!(weight, vec![1.0, 2.0]);
    }

    #[test]
    fn lora_a_key_names_the_weight() {
        let stem = "model.diffusion_model.transformer_blocks.0.attn1.to_q";
        let key = format!("{stem}.lora_A.weight");
        assert_eq!(weight_key_for_lora_a(&key), Some(stem));
        assert!(weight_key_aliases(stem).contains(&format!("{stem}.weight")));
        assert!(weight_key_aliases(stem)
            .contains(&"transformer_blocks.0.attn1.to_q.weight".to_string()));
    }

    #[test]
    fn original_names_alias_the_diffusers_modules() {
        let a = weight_key_aliases("diffusion_model.adaln_single.linear");
        assert!(a.contains(&"time_embed.linear.weight".to_owned()), "{a:?}");
        assert!(a.contains(&"adaln_single.linear.weight".to_owned()));
        let a = weight_key_aliases("diffusion_model.av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_1");
        assert!(a.contains(&"av_cross_attn_video_a2v_gate.emb.timestep_embedder.linear_1.weight".to_owned()));
        assert!(weight_key_aliases("diffusion_model.patchify_proj").contains(&"proj_in.weight".to_owned()));
        assert!(weight_key_aliases("diffusion_model.audio_prompt_adaln_single.linear")
            .contains(&"audio_prompt_adaln.linear.weight".to_owned()));
        // Block linears are spelled alike in both namings.
        let a = weight_key_aliases("diffusion_model.transformer_blocks.3.audio_ff.net.0.proj");
        assert_eq!(a.len(), 3, "{a:?}");
        assert_eq!(diffusers_transformer_name("proj_out"), "proj_out");
    }

    #[test]
    fn reference_scales_default_to_one() {
        let mut m = std::collections::HashMap::new();
        assert_eq!(reference_scale(&m, "reference_downscale_factor"), 1);
        m.insert("reference_downscale_factor".to_string(), "2".to_string());
        m.insert("reference_temporal_scale_factor".to_string(), "x".to_string());
        assert_eq!(reference_scale(&m, "reference_downscale_factor"), 2);
        assert_eq!(reference_scale(&m, "reference_temporal_scale_factor"), 1);
    }

    #[test]
    fn strengths_match_the_profiles() {
        assert_eq!(stage_strengths(Ltx2ModelVersion::V23), Some((0.25, 0.5)));
        assert_eq!(stage_strengths(Ltx2ModelVersion::V25), Some((0.0, 1.0)));
        assert_eq!(REFINER_STRENGTH, 0.8);
        assert_eq!(stage_strengths(Ltx2ModelVersion::V20), None);
        assert_eq!(file_names(Ltx2ModelVersion::V23)[0], LTX23_LORA_NAMES[0]);
    }
}
