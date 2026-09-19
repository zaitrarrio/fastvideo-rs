//! H3 text conditioning: the prompt through Qwen3-VL-32B's language model, read
//! at HF `hidden_states[50]`.
//!
//! Three things are easy to get wrong and all still produce a plausible
//! `[1, S, 5120]` tensor (docs/ports/h3.md, section b):
//!
//! * the prompt is tokenized verbatim — no chat template, no BOS — but with the
//!   seven `<d>`-style markers transformers appends to the vocabulary
//!   ([`fastvideo_models::h3::tokenizer`]);
//! * index 50 of `output_hidden_states` is the residual stream after decoder
//!   layer **49**, and it is *not* normed — [`crate::llm::hidden_states`] uses
//!   the same numbering, so this is tap 50 and layers 50..63 are never read;
//! * attention is causal even though the result is used as an encoder output.

use std::path::Path;

use fastvideo_models::h3::config::H3TextEncoderConfig;
use fastvideo_models::h3::tokenizer::H3Tokenizer;

use crate::llm::{self, DecoderConfig};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Token ids of `prompt` under the checkpoint at `root` (`root/tokenizer/tokenizer.json`).
pub fn tokenize(root: &Path, prompt: &str) -> Result<Vec<u32>> {
    let path = root.join("tokenizer").join("tokenizer.json");
    let tokenizer = H3Tokenizer::from_file(&path).map_err(msg)?;
    if !tokenizer.added_ids_match_reference() {
        return Err(msg(format!(
            "{}: the H3 marker tokens landed on {:?}, not 151669..=151675; this is not the checkpoint's tokenizer",
            path.display(),
            tokenizer.added_special_token_ids()
        )));
    }
    tokenizer.encode(prompt).map_err(msg)
}

/// `hidden_states[tap]` for already-tokenized text: `[1, S, hidden]`, un-normed
/// for any tap short of the last layer. One unpadded sequence, so positions
/// are `0..S` and every token may be attended.
pub fn encode_ids(map: &WeightMap, cfg: &DecoderConfig, ids: &[u32], tap: usize) -> Result<CudaTensor> {
    if tap >= cfg.num_layers() {
        // The last tap is the one entry HF norms; H3 was trained on an
        // un-normed mid-stack state and a normed one is a different input.
        return Err(msg(format!("h3 text: tap {tap} of a {}-layer decoder would be post-norm", cfg.num_layers())));
    }
    let positions: Vec<u32> = (0..ids.len() as u32).collect();
    let attend = vec![true; ids.len()];
    let mut taps = llm::hidden_states(map, cfg, ids, &positions, &attend, &[tap])?;
    taps.pop().ok_or_else(|| msg("h3 text: the decoder returned no hidden state"))
}

/// The conditioning the DiT consumes for `prompt`: token ids and `[1, S, 5120]`.
/// The ~50 GB of decoder weights stream through one layer at a time and are
/// gone when this returns.
pub fn encode_prompt(root: &Path, prompt: &str) -> Result<(Vec<u32>, CudaTensor)> {
    let ids = tokenize(root, prompt)?;
    let map = WeightMap::open(&root.join("text_encoder"))?;
    // The reference encoder runs in bf16; follow its constant casts.
    let cfg = DecoderConfig::qwen3_vl_32b_text().for_bf16_reference();
    let tap = H3TextEncoderConfig::fasth3_8step().output_hidden_state_index;
    let hidden = encode_ids(&map, &cfg, &ids, tap)?;
    Ok((ids, hidden))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Act, LayerAttn};

    fn tiny() -> DecoderConfig {
        DecoderConfig {
            hidden: 8,
            heads: 4,
            kv_heads: 2,
            head_dim: 4,
            intermediate: 12,
            rms_eps: 1e-6,
            norm_offset: 0.0,
            act: Act::Silu,
            qk_norm: true,
            sandwich_norms: false,
            embed_scale: 1.0,
            attn_scale: 0.5,
            layers: vec![LayerAttn { rope_theta: 5_000_000.0, rope_factor: 1.0, window: None }; 3],
            layer_prefix: "m.layers".into(),
            embed_key: "m.embed.weight".into(),
            final_norm_key: "m.norm.weight".into(),
        }
    }

    fn weights() -> WeightMap {
        WeightMap::generated(|key, shape| {
            let seed = key.bytes().fold(3u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
            (0..shape.iter().product::<usize>())
                .map(|i| {
                    let v = (seed.wrapping_add(i as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / (1u32 << 24) as f32;
                    if key.contains("norm") { 0.5 + v } else { v - 0.5 }
                })
                .collect()
        })
    }

    #[test]
    fn the_tap_is_the_stream_after_that_many_layers_and_is_not_normed() {
        let (cfg, ids) = (tiny(), [1u32, 3, 0, 2]);
        let got = encode_ids(&weights(), &cfg, &ids, 2).unwrap();
        assert_eq!(got.shape, vec![1, 4, 8]);
        // A two-layer model's last tap is the same stream, but normed: the
        // un-normed tap must differ from it by exactly that norm.
        let mut two = cfg.clone();
        two.layers.truncate(2);
        let pos: Vec<u32> = (0..4).collect();
        let normed = llm::hidden_states(&weights(), &two, &ids, &pos, &[true; 4], &[2]).unwrap().remove(0);
        let w = crate::wan::weights::cuda_tensor_shaped(&weights(), "m.norm.weight", &[8]).unwrap();
        let want = got.rms_norm(&w, cfg.rms_eps).unwrap();
        let (a, b) = (want.host_cow().unwrap(), normed.host_cow().unwrap());
        assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-6));
        assert!(got.host_cow().unwrap().iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-3));
    }

    #[test]
    fn a_tap_that_would_be_normed_is_refused() {
        let err = encode_ids(&weights(), &tiny(), &[1, 2], 3).unwrap_err();
        assert!(err.to_string().contains("post-norm"), "{err}");
    }
}
