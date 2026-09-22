//! Documented Diffusers safetensors key prefixes for Phase 4–5 DiT / VAE packs.
//!
//! `load()` probes these probes so a standard Hub/Diffusers layout either
//! wires tensors into the scaffold graph or fails with a clear missing-key
//! message (never silent zeros when a weight map was opened).

use crate::wan::weights::WeightMap;

/// SD3 / SD3.5 `transformer/` (see Diffusers `SD3Transformer2DModel`).
pub mod sd35 {
    pub const PROBES: &[&str] = &[
        "pos_embed.proj.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "context_embedder.weight",
        "transformer_blocks.0.norm1.linear.weight",
        "norm_out.linear.weight",
        "proj_out.weight",
    ];
}

/// FLUX.1 `transformer/` (`FluxTransformer2DModel`).
pub mod flux {
    pub const PROBES: &[&str] = &[
        "x_embedder.weight",
        "context_embedder.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "transformer_blocks.0.norm1.linear.weight",
        "single_transformer_blocks.0.norm.linear.weight",
        "proj_out.weight",
    ];
}

/// FLUX.2 `transformer/` (Klein / Dev; same embedder names as FLUX.1 family).
pub mod flux2 {
    pub const PROBES: &[&str] = &[
        "x_embedder.weight",
        "context_embedder.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "transformer_blocks.0.norm1.linear.weight",
        "proj_out.weight",
    ];
}

/// Z-Image `transformer/` (`ZImageTransformer2DModel`).
pub mod zimage {
    pub const PROBES: &[&str] = &[
        "all_x_embedder.weight",
        "x_embedder.weight",
        "cap_embedder.0.weight",
        "cap_embedder.weight",
        "final_layer.linear.weight",
        "proj_out.weight",
        "noise_refiner.0.adaLN_modulation.1.weight",
        "layers.0.adaLN_modulation.1.weight",
    ];
}

/// GLM-Image `transformer/`.
pub mod glm_image {
    pub const PROBES: &[&str] = &[
        "x_embedder.weight",
        "pos_embed.proj.weight",
        "context_embedder.weight",
        "txt_in.weight",
        "transformer_blocks.0.norm1.weight",
        "proj_out.weight",
        "final_layer.linear.weight",
    ];
}

/// GLM-Image AR tower under `vision_language_encoder/`.
pub mod glm_image_ar {
    pub const PROBES: &[&str] = &[
        "model.language_model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "language_model.layers.0.self_attn.q_proj.weight",
    ];
}

/// GLM-Image ByT5 glyph encoder under `text_encoder/`.
pub mod glm_image_byt5 {
    pub const PROBES: &[&str] = &[
        "encoder.block.0.layer.0.SelfAttention.q.weight",
        "encoder.embed_tokens.weight",
        "shared.weight",
        "encoder.final_layer_norm.weight",
    ];
}

/// MMAudio Synchformer visual under `image_encoder/`.
pub mod mmaudio_synchformer {
    pub const PROBES: &[&str] = &[
        "vfeat_extractor.patch_embed_3d.proj.weight",
        "vfeat_extractor.blocks.0.attn.qkv.weight",
        "vfeat_extractor.spatial_attn_agg.cls_token",
        "vfeat_extractor.norm.weight",
        "patch_embed_3d.proj.weight",
        "blocks.0.attn.qkv.weight",
        "spatial_attn_agg.cls_token",
    ];
}

/// Stable Audio DiT `transformer/`.
pub mod stable_audio {
    pub const PROBES: &[&str] = &[
        "preprocessor_mlp.0.weight",
        "timestep_proj.weight",
        "global_proj.weight",
        "transformer_blocks.0.ff.net.0.proj.weight",
        "proj_out.weight",
    ];
}

/// Diffusers `AutoencoderKL` `vae/` (2D).
pub mod autoencoder_kl {
    pub const PROBES: &[&str] = &[
        "encoder.conv_in.weight",
        "encoder.mid_block.resnets.0.conv1.weight",
        "decoder.conv_in.weight",
        "decoder.mid_block.resnets.0.conv1.weight",
        "decoder.conv_out.weight",
        "quant_conv.weight",
        "post_quant_conv.weight",
    ];
}

/// LTX-2 video VAE encoder half (`vae/`).
pub mod ltx2_vae_encoder {
    pub const PROBES: &[&str] = &[
        "encoder.conv_in.conv.weight",
        "encoder.down_blocks.0.resnets.0.conv1.conv.weight",
        "encoder.mid_block.resnets.0.conv1.conv.weight",
        "encoder.conv_out.conv.weight",
        "encoder.conv_out.weight",
        "latents_mean",
        "latents_std",
    ];
}

/// World-control injectors (GameCraft / DreamX / Matrix-Game / LingBot-World / HY-World).
pub mod world {
    pub const CAMERA_NET: &[&str] = &[
        "camera_net.conv_in.weight",
        "camera_net.blocks.0.weight",
        "CameraNet.conv_in.weight",
    ];
    pub const PROPE: &[&str] = &[
        "control_adapter.proj.weight",
        "prope.proj.weight",
        "cam_proj.weight",
    ];
    pub const ACTION: &[&str] = &[
        "action_blocks.0.attn.to_q.weight",
        "action_embedder.weight",
        "action_in.weight",
    ];
    pub const CAM_INJECTOR: &[&str] = &[
        "cam_injector.proj.weight",
        "camera_injector.weight",
        "plucker_proj.weight",
    ];
    pub const SIGLIP: &[&str] = &[
        "image_encoder.vision_model.embeddings.patch_embedding.weight",
        "siglip.embeddings.patch_embedding.weight",
        "vision_model.embeddings.patch_embedding.weight",
    ];
}

/// First present probe key, or `None`.
pub fn first_present(map: &WeightMap, probes: &[&str]) -> Option<String> {
    probes.iter().find(|k| map.contains(k)).map(|s| (*s).to_string())
}

/// Require at least one probe key; return the hit or a clear error listing probes.
pub fn require_any(map: &WeightMap, family: &str, probes: &[&str]) -> Result<String, String> {
    if let Some(k) = first_present(map, probes) {
        return Ok(k);
    }
    Err(format!(
        "{family} load: no Diffusers keys matched probes {:?}; open a standard Hub pack under transformer/ or vae/",
        probes.iter().take(4).collect::<Vec<_>>()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_lists_nonempty() {
        assert!(!sd35::PROBES.is_empty());
        assert!(!flux::PROBES.is_empty());
        assert!(!zimage::PROBES.is_empty());
        assert!(!autoencoder_kl::PROBES.is_empty());
        assert!(!world::CAMERA_NET.is_empty());
        assert!(!glm_image_ar::PROBES.is_empty());
        assert!(!mmaudio_synchformer::PROBES.is_empty());
        assert!(ltx2_vae_encoder::PROBES.iter().any(|k| k.contains("down_blocks")));
    }
}
