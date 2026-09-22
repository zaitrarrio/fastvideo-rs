//! Cosmos T5 encode helper (tokenizer + `text_encoder/`).

use std::path::Path;

use fastvideo_models::cosmos::{tokenize_t5, T5Config};

use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::WeightMap;

use super::t5::T5Encoder;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Encode prompt to `[1, S, d_model]`, zeroing pad positions (Diffusers Cosmos).
pub fn encode_prompt(root: &Path, prompt: &str, text_dim: usize) -> Result<CudaTensor> {
    let cfg = T5Config::t5_11b();
    if cfg.d_model != text_dim {
        return Err(msg(format!(
            "cosmos T5 d_model {} vs DiT text_embed_dim {text_dim}",
            cfg.d_model
        )));
    }
    let (ids, mask) = tokenize_t5(root, prompt, cfg.max_sequence_length).map_err(msg)?;
    let map = WeightMap::open(&root.join("text_encoder")).map_err(|e| msg(e.to_string()))?;
    let enc = T5Encoder::load(cfg.clone(), &map)?;
    let mut embeds = enc.forward(&ids, 1, ids.len(), Some(&mask))?;
    // Zero pad rows like Diffusers `_get_t5_prompt_embeds`.
    let mut host = embeds.host_cow()?.to_vec();
    let d = cfg.d_model;
    for (i, &keep) in mask.iter().enumerate() {
        if !keep {
            for c in 0..d {
                host[i * d + c] = 0.0;
            }
        }
    }
    embeds = CudaTensor::from_vec(host, embeds.shape.clone())?;
    Ok(embeds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_fallback_shape_doc() {
        let z = CudaTensor::zeros(&[1, 16, 1024]);
        assert_eq!(z.shape, vec![1, 16, 1024]);
    }
}
