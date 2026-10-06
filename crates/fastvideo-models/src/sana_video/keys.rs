//! Every tensor the SANA-Video loaders read, with its expected shape.
//! Checked against the Hub header of
//! `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` @ db5f398b
//! (`fixtures/hub_*_keys.tsv`).

use crate::key_fixture::KeySpec;

use super::config::{Gemma2TextConfig, SanaVideoTransformerConfig};

/// `transformer/` keys (`SanaVideoTransformer3DModel`).
pub fn transformer_keys(cfg: &SanaVideoTransformerConfig) -> KeySpec {
    let d = cfg.inner_dim();
    let hid = cfg.ff_hidden();
    let [pt, ph, pw] = cfg.patch_size;
    let mut k = KeySpec::new();
    let mut put = |name: String, shape: Vec<usize>| {
        k.insert(name, shape);
    };
    put("patch_embedding.weight".into(), vec![d, cfg.in_channels, pt, ph, pw]);
    put("patch_embedding.bias".into(), vec![d]);
    put("time_embed.emb.timestep_embedder.linear_1.weight".into(), vec![d, 256]);
    put("time_embed.emb.timestep_embedder.linear_1.bias".into(), vec![d]);
    put("time_embed.emb.timestep_embedder.linear_2.weight".into(), vec![d, d]);
    put("time_embed.emb.timestep_embedder.linear_2.bias".into(), vec![d]);
    put("time_embed.linear.weight".into(), vec![6 * d, d]);
    put("time_embed.linear.bias".into(), vec![6 * d]);
    put("caption_projection.linear_1.weight".into(), vec![d, cfg.caption_channels]);
    put("caption_projection.linear_1.bias".into(), vec![d]);
    put("caption_projection.linear_2.weight".into(), vec![d, d]);
    put("caption_projection.linear_2.bias".into(), vec![d]);
    put("caption_norm.weight".into(), vec![d]);
    put("scale_shift_table".into(), vec![2, d]);
    put("proj_out.weight".into(), vec![cfg.patch_volume() * cfg.out_channels, d]);
    put("proj_out.bias".into(), vec![cfg.patch_volume() * cfg.out_channels]);
    for i in 0..cfg.num_layers {
        let p = |n: &str| format!("transformer_blocks.{i}.{n}");
        put(p("scale_shift_table"), vec![6, d]);
        for proj in ["to_q", "to_k", "to_v"] {
            put(p(&format!("attn1.{proj}.weight")), vec![d, d]);
            if cfg.attention_bias {
                put(p(&format!("attn1.{proj}.bias")), vec![d]);
            }
            put(p(&format!("attn2.{proj}.weight")), vec![d, d]);
            put(p(&format!("attn2.{proj}.bias")), vec![d]);
        }
        for a in ["attn1", "attn2"] {
            put(p(&format!("{a}.norm_q.weight")), vec![d]);
            put(p(&format!("{a}.norm_k.weight")), vec![d]);
            put(p(&format!("{a}.to_out.0.weight")), vec![d, d]);
            put(p(&format!("{a}.to_out.0.bias")), vec![d]);
        }
        put(p("ff.conv_inverted.weight"), vec![2 * hid, d, 1, 1]);
        put(p("ff.conv_inverted.bias"), vec![2 * hid]);
        put(p("ff.conv_depth.weight"), vec![2 * hid, 1, 3, 3]);
        put(p("ff.conv_depth.bias"), vec![2 * hid]);
        put(p("ff.conv_point.weight"), vec![d, hid, 1, 1]);
        put(p("ff.conv_temp.weight"), vec![d, d, 3, 1]);
    }
    k
}

/// `text_encoder/` keys (`Gemma2Model`, no `model.` prefix).
pub fn gemma2_keys(cfg: &Gemma2TextConfig) -> KeySpec {
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    let mut k = KeySpec::new();
    k.insert("embed_tokens.weight".into(), vec![cfg.vocab_size, h]);
    k.insert("norm.weight".into(), vec![h]);
    for i in 0..cfg.num_hidden_layers {
        let p = |n: &str| format!("layers.{i}.{n}");
        k.insert(p("self_attn.q_proj.weight"), vec![q, h]);
        k.insert(p("self_attn.k_proj.weight"), vec![kv, h]);
        k.insert(p("self_attn.v_proj.weight"), vec![kv, h]);
        k.insert(p("self_attn.o_proj.weight"), vec![h, q]);
        k.insert(p("mlp.gate_proj.weight"), vec![cfg.intermediate_size, h]);
        k.insert(p("mlp.up_proj.weight"), vec![cfg.intermediate_size, h]);
        k.insert(p("mlp.down_proj.weight"), vec![h, cfg.intermediate_size]);
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            k.insert(p(&format!("{n}.weight")), vec![h]);
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key_fixture::{diff_against, parse_fixture};

    #[test]
    fn transformer_keys_match_the_hub_checkpoint() {
        let header = parse_fixture(include_str!("fixtures/hub_transformer_keys.tsv")).unwrap();
        let spec = transformer_keys(&SanaVideoTransformerConfig::sana_video_2b_480p());
        let problems = diff_against(&spec, &header, |_| true);
        assert!(problems.is_empty(), "{problems:#?}");
        assert_eq!(spec.len(), 496);
    }

    #[test]
    fn gemma2_keys_match_the_hub_checkpoint() {
        let header = parse_fixture(include_str!("fixtures/hub_text_encoder_keys.tsv")).unwrap();
        let spec = gemma2_keys(&Gemma2TextConfig::gemma2_2b());
        let problems = diff_against(&spec, &header, |_| true);
        assert!(problems.is_empty(), "{problems:#?}");
    }
}
