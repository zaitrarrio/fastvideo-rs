//! SANA-Video prompt encoding: Gemma-2-2B (`Gemma2Model`, last hidden state
//! after the final norm) through [`crate::llm`], then the pipeline's
//! `select_index` cut with the padding dropped
//! ([`fastvideo_models::sana_video::text::PromptLayout`]).

use fastvideo_models::sana_video::text::PromptLayout;
use fastvideo_models::sana_video::Gemma2TextConfig;

use crate::llm::{self, Act, DecoderConfig, LayerAttn};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// transformers' `Gemma2Model` as a [`DecoderConfig`]: sandwich norms with
/// `1 + w` weights, GeGLU (tanh), `sqrt(hidden)` embedding scale,
/// `query_pre_attn_scalar ** -0.5` softmax scale, logit soft-capping, and
/// sliding windows on the layers `layer_types` marks.
pub fn gemma2_decoder(cfg: &Gemma2TextConfig) -> DecoderConfig {
    let layers = cfg
        .sliding_layers
        .iter()
        .map(|&sliding| {
            if sliding {
                LayerAttn::sliding(cfg.rope_theta, cfg.sliding_window)
            } else {
                LayerAttn::global(cfg.rope_theta, 1.0)
            }
        })
        .collect();
    DecoderConfig {
        vocab: cfg.vocab_size,
        hidden: cfg.hidden_size,
        heads: cfg.num_attention_heads,
        kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        intermediate: cfg.intermediate_size,
        rms_eps: cfg.rms_norm_eps,
        norm_offset: 1.0,
        act: Act::GeluTanh,
        qk_norm: false,
        sandwich_norms: true,
        // transformers casts the normalizer to the hidden dtype; sqrt(2304)
        // is exactly 48, so bf16 and f32 agree for the 2B model.
        embed_scale: (cfg.hidden_size as f32).sqrt(),
        attn_scale: (cfg.query_pre_attn_scalar as f32).powf(-0.5),
        layers,
        layer_prefix: "layers".into(),
        embed_key: "embed_tokens.weight".into(),
        final_norm_key: "norm.weight".into(),
        attention_k_eq_v: false,
        v_norm: false,
        layer_scalar: false,
        attn_softcap: cfg.attn_logit_softcapping,
    }
}

/// Encode one prompt layout: `[1, kept, hidden]` (the DiT's caption tokens).
pub fn encode_layout(
    map: &WeightMap,
    cfg: &DecoderConfig,
    layout: &PromptLayout,
) -> Result<CudaTensor> {
    if layout.ids.is_empty() || layout.kept.is_empty() {
        return Err(msg("sana text: empty prompt"));
    }
    let n = layout.ids.len();
    let positions: Vec<u32> = (0..n as u32).collect();
    let attend = vec![true; n];
    let hidden = llm::hidden_states(
        map,
        cfg,
        &layout.ids,
        &positions,
        &attend,
        &[cfg.num_layers()],
    )?
    .pop()
    .ok_or_else(|| msg("sana text: no hidden state"))?;
    let h = cfg.hidden;
    let rows = hidden.reshape(vec![n, h])?;
    let kept = rows.embedding_rows(&layout.kept)?;
    kept.reshape(vec![1, layout.kept.len(), h])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma2_decoder_settings() {
        let d = gemma2_decoder(&Gemma2TextConfig::gemma2_2b());
        assert_eq!(d.num_layers(), 26);
        assert_eq!(d.layers[0].window, Some(4096));
        assert_eq!(d.layers[1].window, None);
        assert_eq!(d.embed_scale, 48.0);
        assert!((d.attn_scale - 1.0 / 16.0).abs() < 1e-9);
        assert_eq!(d.attn_softcap, Some(50.0));
        assert_eq!(d.norm_offset, 1.0);
    }

    /// The streamed encoder reads exactly the Gemma-2 keys of the Hub tree.
    #[test]
    fn tiny_gemma2_encodes_and_reads_the_key_spec() {
        use std::sync::{Arc, Mutex};
        let cfg = Gemma2TextConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 24,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            query_pre_attn_scalar: 8.0,
            attn_logit_softcapping: Some(50.0),
            sliding_window: 4096,
            sliding_layers: vec![true, false],
        };
        let seen = Arc::new(Mutex::new(Vec::new()));
        let rec = seen.clone();
        let map = WeightMap::generated(move |key, shape| {
            rec.lock().unwrap().push((key.to_string(), shape.to_vec()));
            let n: usize = shape.iter().product();
            (0..n).map(|i| ((i * 7 + key.len()) % 13) as f32 * 0.01 - 0.06).collect()
        });
        let layout = PromptLayout::new(vec![2, 5, 9, 11, 3], 7, 4);
        let out = encode_layout(&map, &gemma2_decoder(&cfg), &layout).unwrap();
        assert_eq!(layout.kept, vec![0, 4]);
        assert_eq!(out.shape, vec![1, 2, 16]);
        assert!(out.host_cow().unwrap().iter().all(|v| v.is_finite()));
        let got: std::collections::BTreeMap<String, Vec<usize>> =
            seen.lock().unwrap().iter().cloned().collect();
        let want = fastvideo_models::sana_video::keys::gemma2_keys(&cfg);
        assert_eq!(got, want);
    }
}
