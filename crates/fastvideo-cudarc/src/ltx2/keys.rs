//! One set of loaders, two checkpoint namings.
//!
//! `Lightricks/LTX-2` publishes the distilled DiT and its connectors only
//! inside `ltx-2-19b-distilled.safetensors`, in the original `ltx-core` key
//! names; the diffusers folders in the same repo are the *dev* model. diffusers'
//! converter turns one naming into the other by renaming alone — no tensor is
//! reshaped, fused or transposed — so the graph code asks for diffusers names
//! (the ones docs/ports/ltx2.md §g documents) and this view spells them the way
//! the file on disk does. Nothing is copied or rewritten.

use crate::wan::weights::WeightMap;

use super::msg;
use crate::wan::tensor::Result;

/// How the checkpoint on disk names its tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// A diffusers component folder (`transformer/`, `connectors/`).
    Diffusers,
    /// `ltx-2-19b-*.safetensors`: everything under `model.diffusion_model.`.
    SingleFile,
}

/// Which diffusers component a name belongs to; the single file keeps the DiT
/// and the connectors under one root, told apart only by their leading segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Component {
    Transformer,
    Connectors,
}

#[derive(Debug, Clone, Copy)]
pub struct Keys {
    layout: Layout,
    component: Component,
}

const SINGLE_FILE_ROOT: &str = "model.diffusion_model";

impl Keys {
    pub fn transformer(layout: Layout) -> Self {
        Self { layout, component: Component::Transformer }
    }

    pub fn connectors(layout: Layout) -> Self {
        Self { layout, component: Component::Connectors }
    }

    /// Guess the layout from what the map holds. A generated map (tests) has no
    /// tensors and reads as diffusers.
    pub fn detect(map: &WeightMap) -> Layout {
        if map.has_tensor(&format!("{SINGLE_FILE_ROOT}.patchify_proj.weight")) {
            Layout::SingleFile
        } else {
            Layout::Diffusers
        }
    }

    /// The on-disk spelling of a diffusers name. Works on whole keys
    /// (`….to_q.weight`) and on module prefixes (`….to_q`) alike, because every
    /// rename replaces whole dot-separated segments.
    pub fn key(&self, diffusers: &str) -> String {
        if self.layout == Layout::Diffusers {
            return diffusers.to_string();
        }
        let renamed: Vec<&str> = diffusers
            .split('.')
            .map(|seg| match (self.component, seg) {
                (_, "norm_q") => "q_norm",
                (_, "norm_k") => "k_norm",
                (Component::Transformer, "proj_in") => "patchify_proj",
                (Component::Transformer, "audio_proj_in") => "audio_patchify_proj",
                (Component::Transformer, "time_embed") => "adaln_single",
                (Component::Transformer, "audio_time_embed") => "audio_adaln_single",
                (Component::Transformer, "av_cross_attn_video_scale_shift") => "av_ca_video_scale_shift_adaln_single",
                (Component::Transformer, "av_cross_attn_audio_scale_shift") => "av_ca_audio_scale_shift_adaln_single",
                (Component::Transformer, "av_cross_attn_video_a2v_gate") => "av_ca_a2v_gate_adaln_single",
                (Component::Transformer, "av_cross_attn_audio_v2a_gate") => "av_ca_v2a_gate_adaln_single",
                (Component::Transformer, "video_a2v_cross_attn_scale_shift_table") => "scale_shift_table_a2v_ca_video",
                (Component::Transformer, "audio_a2v_cross_attn_scale_shift_table") => "scale_shift_table_a2v_ca_audio",
                (Component::Connectors, "video_connector") => "video_embeddings_connector",
                (Component::Connectors, "audio_connector") => "audio_embeddings_connector",
                (Component::Connectors, "transformer_blocks") => "transformer_1d_blocks",
                (_, other) => other,
            })
            .collect();
        format!("{SINGLE_FILE_ROOT}.{}", renamed.join("."))
    }

    /// Module prefix of the connectors' `text_proj_in` (188160 → 3840). In the
    /// single file it is `text_embedding_projection.aggregate_embed`, and the
    /// releases disagree on whether that sits under the DiT root, so probe.
    pub fn text_proj_in(&self, map: &WeightMap) -> Result<String> {
        if self.layout == Layout::Diffusers {
            return Ok("text_proj_in".to_string());
        }
        let candidates = [
            "text_embedding_projection.aggregate_embed".to_string(),
            format!("{SINGLE_FILE_ROOT}.text_embedding_projection.aggregate_embed"),
        ];
        candidates
            .iter()
            .find(|c| map.has_tensor(&format!("{c}.weight")))
            .cloned()
            .ok_or_else(|| msg(format!("no text projection in the checkpoint; looked for {candidates:?} (+ `.weight`)")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffusers_names_pass_through() {
        let k = Keys::transformer(Layout::Diffusers);
        assert_eq!(k.key("transformer_blocks.3.attn1.norm_q.weight"), "transformer_blocks.3.attn1.norm_q.weight");
    }

    /// The table in docs/ports/ltx2.md §g, read right to left.
    #[test]
    fn single_file_names_follow_the_converter_table() {
        let k = Keys::transformer(Layout::SingleFile);
        let cases = [
            ("proj_in", "patchify_proj"),
            ("audio_proj_in.bias", "audio_patchify_proj.bias"),
            ("proj_out.weight", "proj_out.weight"),
            ("time_embed.emb.timestep_embedder.linear_1", "adaln_single.emb.timestep_embedder.linear_1"),
            ("audio_time_embed.linear", "audio_adaln_single.linear"),
            ("av_cross_attn_video_scale_shift.linear", "av_ca_video_scale_shift_adaln_single.linear"),
            ("av_cross_attn_audio_scale_shift.linear", "av_ca_audio_scale_shift_adaln_single.linear"),
            ("av_cross_attn_video_a2v_gate.linear", "av_ca_a2v_gate_adaln_single.linear"),
            ("av_cross_attn_audio_v2a_gate.linear", "av_ca_v2a_gate_adaln_single.linear"),
            ("transformer_blocks.7.video_a2v_cross_attn_scale_shift_table", "transformer_blocks.7.scale_shift_table_a2v_ca_video"),
            ("transformer_blocks.7.audio_a2v_cross_attn_scale_shift_table", "transformer_blocks.7.scale_shift_table_a2v_ca_audio"),
            ("transformer_blocks.7.audio_scale_shift_table", "transformer_blocks.7.audio_scale_shift_table"),
            ("transformer_blocks.7.audio_to_video_attn.norm_k.weight", "transformer_blocks.7.audio_to_video_attn.k_norm.weight"),
            ("transformer_blocks.7.ff.net.0.proj", "transformer_blocks.7.ff.net.0.proj"),
            ("caption_projection.linear_1", "caption_projection.linear_1"),
        ];
        for (diffusers, original) in cases {
            assert_eq!(k.key(diffusers), format!("model.diffusion_model.{original}"), "{diffusers}");
        }
    }

    #[test]
    fn connector_blocks_are_renamed_but_dit_blocks_are_not() {
        let c = Keys::connectors(Layout::SingleFile);
        assert_eq!(
            c.key("video_connector.transformer_blocks.1.attn1.norm_q.weight"),
            "model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.1.attn1.q_norm.weight"
        );
        assert_eq!(c.key("audio_connector.learnable_registers"), "model.diffusion_model.audio_embeddings_connector.learnable_registers");
        let t = Keys::transformer(Layout::SingleFile);
        assert!(t.key("transformer_blocks.0.attn1.to_q").contains(".transformer_blocks.0."));
    }

    #[test]
    fn a_generated_map_reads_as_diffusers() {
        let map = WeightMap::generated(|_, shape| vec![0.0; shape.iter().product()]);
        assert_eq!(Keys::detect(&map), Layout::Diffusers);
        assert_eq!(Keys::connectors(Layout::Diffusers).text_proj_in(&map).unwrap(), "text_proj_in");
        assert!(Keys::connectors(Layout::SingleFile).text_proj_in(&map).is_err());
    }
}
