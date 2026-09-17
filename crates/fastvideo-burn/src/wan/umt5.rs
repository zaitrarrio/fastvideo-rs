//! Tiny UMT5 encoder matching Diffusers / Candle key layout (zeros init).

use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::module::embedding;

use fastvideo_models::Umt5Config;

use super::nn::{gelu_tanh, Linear, B, Device};
use super::weights::{self, WeightMap};
use crate::error::Result;

fn t5_layer_norm(xs: Tensor<B, 3>, weight: &Tensor<B, 1>, eps: f32) -> Tensor<B, 3> {
    let mean_sq = xs.clone().powf_scalar(2.0).mean_dim(2);
    let y = xs / (mean_sq.add_scalar(eps)).sqrt();
    let w = weight.clone().reshape([1, 1, weight.dims()[0]]);
    y * w
}

fn relative_position_bucket(
    seq_len: usize,
    num_buckets: usize,
    max_distance: usize,
    device: &Device,
) -> Tensor<B, 2, Int> {
    let mut buckets = vec![0i32; seq_len * seq_len];
    let num_buckets = num_buckets as i32;
    let max_exact = num_buckets / 4;
    let max_distance = max_distance as i32;
    for i in 0..seq_len as i32 {
        for j in 0..seq_len as i32 {
            let mut relative = j - i;
            let mut bucket = 0i32;
            let n_buckets = num_buckets / 2;
            if relative < 0 {
                bucket += n_buckets;
                relative = -relative;
            }
            let is_small = relative < max_exact;
            let relative_log = ((relative as f64 / max_exact as f64).ln()
                / (max_distance as f64 / max_exact as f64).ln()
                * (n_buckets - max_exact) as f64)
                .floor() as i32
                + max_exact;
            let relative_bucket = if is_small {
                relative
            } else {
                relative_log.min(n_buckets - 1)
            };
            buckets[(i as usize) * seq_len + j as usize] = bucket + relative_bucket;
        }
    }
    Tensor::<B, 1, Int>::from_ints(buckets.as_slice(), device).reshape([seq_len, seq_len])
}

#[derive(Debug, Clone)]
struct DenseGated {
    wi_0: Linear,
    wi_1: Linear,
    wo: Linear,
}

impl DenseGated {
    fn zeros(cfg: &Umt5Config, device: &Device) -> Self {
        Self {
            wi_0: Linear::zeros(cfg.d_model, cfg.d_ff, device),
            wi_1: Linear::zeros(cfg.d_model, cfg.d_ff, device),
            wo: Linear::zeros(cfg.d_ff, cfg.d_model, device),
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &Umt5Config, device: &Device) -> Result<Self> {
        Ok(Self {
            wi_0: Linear::load(
                map,
                &weights::join_key(prefix, "wi_0"),
                cfg.d_model,
                cfg.d_ff,
                false,
                device,
            )?,
            wi_1: Linear::load(
                map,
                &weights::join_key(prefix, "wi_1"),
                cfg.d_model,
                cfg.d_ff,
                false,
                device,
            )?,
            wo: Linear::load(
                map,
                &weights::join_key(prefix, "wo"),
                cfg.d_ff,
                cfg.d_model,
                false,
                device,
            )?,
        })
    }

    fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let gelu = gelu_tanh(self.wi_0.forward_3(xs.clone()));
        let linear = self.wi_1.forward_3(xs);
        self.wo.forward_3(gelu * linear)
    }
}

#[derive(Debug, Clone)]
struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    relative_bias: Tensor<B, 2>,
    n_heads: usize,
    d_kv: usize,
}

impl SelfAttention {
    fn zeros(cfg: &Umt5Config, device: &Device) -> Self {
        let inner = cfg.num_heads * cfg.d_kv;
        Self {
            q: Linear::zeros_no_bias(cfg.d_model, inner, device),
            k: Linear::zeros_no_bias(cfg.d_model, inner, device),
            v: Linear::zeros_no_bias(cfg.d_model, inner, device),
            o: Linear::zeros_no_bias(inner, cfg.d_model, device),
            relative_bias: Tensor::zeros([cfg.relative_attention_num_buckets, cfg.num_heads], device),
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &Umt5Config, device: &Device) -> Result<Self> {
        let inner = cfg.num_heads * cfg.d_kv;
        Ok(Self {
            q: Linear::load(map, &weights::join_key(prefix, "q"), cfg.d_model, inner, false, device)?,
            k: Linear::load(map, &weights::join_key(prefix, "k"), cfg.d_model, inner, false, device)?,
            v: Linear::load(map, &weights::join_key(prefix, "v"), cfg.d_model, inner, false, device)?,
            o: Linear::load(map, &weights::join_key(prefix, "o"), inner, cfg.d_model, false, device)?,
            relative_bias: weights::tensor2_shaped(
                map,
                &weights::join_key(prefix, "relative_attention_bias.weight"),
                [cfg.relative_attention_num_buckets, cfg.num_heads],
                device,
            )?,
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
        })
    }

    fn forward(
        &self,
        xs: Tensor<B, 3>,
        mask: Option<Tensor<B, 4>>,
        buckets: &Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [b, s, _] = xs.dims();
        let q = self
            .q
            .forward_3(xs.clone())
            .reshape([b, s, self.n_heads, self.d_kv])
            .swap_dims(1, 2);
        let k = self
            .k
            .forward_3(xs.clone())
            .reshape([b, s, self.n_heads, self.d_kv])
            .swap_dims(1, 2);
        let v = self
            .v
            .forward_3(xs)
            .reshape([b, s, self.n_heads, self.d_kv])
            .swap_dims(1, 2);
        let mut scores = q.matmul(k.swap_dims(2, 3));
        let flat = buckets.clone().reshape([s * s]);
        let bias = self
            .relative_bias
            .clone()
            .select(0, flat)
            .reshape([s, s, self.n_heads])
            .permute([2, 0, 1])
            .unsqueeze::<4>();
        scores = scores + bias;
        if let Some(m) = mask {
            let neg = Tensor::<B, 4>::full(scores.dims(), f32::NEG_INFINITY, &scores.device());
            let m = m.expand(scores.dims());
            let false_mask = m.lower_equal_elem(0.5);
            scores = scores.mask_where(false_mask, neg);
        }
        let attn = softmax(scores, 3);
        let ctx = attn
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([b, s, self.n_heads * self.d_kv]);
        self.o.forward_3(ctx)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    attn: SelfAttention,
    ff: DenseGated,
    ln1: Tensor<B, 1>,
    ln2: Tensor<B, 1>,
    eps: f32,
}

impl EncoderLayer {
    fn zeros(cfg: &Umt5Config, device: &Device) -> Self {
        Self {
            attn: SelfAttention::zeros(cfg, device),
            ln1: Tensor::zeros([cfg.d_model], device),
            ff: DenseGated::zeros(cfg, device),
            ln2: Tensor::zeros([cfg.d_model], device),
            eps: cfg.eps as f32,
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &Umt5Config, device: &Device) -> Result<Self> {
        Ok(Self {
            attn: SelfAttention::load(
                map,
                &weights::join_key(prefix, "layer.0.SelfAttention"),
                cfg,
                device,
            )?,
            ln1: weights::tensor1_shaped(
                map,
                &weights::join_key(prefix, "layer.0.layer_norm.weight"),
                cfg.d_model,
                device,
            )?,
            ff: DenseGated::load(
                map,
                &weights::join_key(prefix, "layer.1.DenseReluDense"),
                cfg,
                device,
            )?,
            ln2: weights::tensor1_shaped(
                map,
                &weights::join_key(prefix, "layer.1.layer_norm.weight"),
                cfg.d_model,
                device,
            )?,
            eps: cfg.eps as f32,
        })
    }

    fn forward(
        &self,
        xs: Tensor<B, 3>,
        mask: Option<Tensor<B, 4>>,
        buckets: &Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let normed = t5_layer_norm(xs.clone(), &self.ln1, self.eps);
        let xs = xs + self.attn.forward(normed, mask, buckets);
        let normed = t5_layer_norm(xs.clone(), &self.ln2, self.eps);
        xs + self.ff.forward(normed)
    }
}

#[derive(Debug, Clone)]
pub struct Umt5Encoder {
    pub cfg: Umt5Config,
    embed: Tensor<B, 2>,
    layers: Vec<EncoderLayer>,
    final_ln: Tensor<B, 1>,
    device: Device,
}

impl Umt5Encoder {
    pub fn zeros(cfg: Umt5Config, device: &Device) -> Self {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            layers.push(EncoderLayer::zeros(&cfg, device));
        }
        Self {
            embed: Tensor::zeros([cfg.vocab_size, cfg.d_model], device),
            final_ln: Tensor::zeros([cfg.d_model], device),
            layers,
            device: device.clone(),
            cfg,
        }
    }

    pub fn load(cfg: Umt5Config, map: &WeightMap, device: &Device) -> Result<Self> {
        Self::from_map(cfg, map, device)
    }

    pub fn from_map(cfg: Umt5Config, map: &WeightMap, device: &Device) -> Result<Self> {
        let embed_key = if map.contains("encoder.embed_tokens.weight") {
            "encoder.embed_tokens.weight"
        } else {
            "shared.weight"
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(EncoderLayer::load(
                map,
                &format!("encoder.block.{i}"),
                &cfg,
                device,
            )?);
        }
        Ok(Self {
            embed: weights::tensor2_shaped(
                map,
                embed_key,
                [cfg.vocab_size, cfg.d_model],
                device,
            )?,
            final_ln: weights::tensor1_shaped(
                map,
                "encoder.final_layer_norm.weight",
                cfg.d_model,
                device,
            )?,
            layers,
            device: device.clone(),
            cfg,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let [_b, s] = input_ids.dims();
        let mut hidden = embedding(self.embed.clone(), input_ids);
        let buckets = relative_position_bucket(
            s,
            self.cfg.relative_attention_num_buckets,
            self.cfg.relative_attention_max_distance,
            &self.device,
        );
        for layer in &self.layers {
            hidden = layer.forward(hidden, attention_mask.clone(), &buckets);
        }
        t5_layer_norm(hidden, &self.final_ln, self.cfg.eps as f32)
    }
}

/// Trim true token length then right-pad to `text_len`.
pub fn pad_prompt_embeds(
    embeds: Tensor<B, 3>,
    seq_lens: &[usize],
    text_len: usize,
) -> Result<Tensor<B, 3>> {
    let [b, _s, d] = embeds.dims();
    let device = embeds.device();
    let mut rows = Vec::with_capacity(b);
    for (i, &len) in seq_lens.iter().enumerate() {
        let len = len.min(text_len);
        let row = embeds.clone().narrow(0, i, 1).narrow(1, 0, len);
        // Burn CUDA `cat` panics on a zero-length axis (FastDivmod divisor=0).
        let padded = if len >= text_len {
            row
        } else {
            let pad = Tensor::<B, 3>::zeros([1, text_len - len, d], &device);
            Tensor::cat(vec![row, pad], 1)
        };
        rows.push(padded);
    }
    if rows.len() == 1 {
        return Ok(rows.pop().unwrap());
    }
    Ok(Tensor::cat(rows, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_umt5_forward_shape() {
        let device = Default::default();
        let cfg = Umt5Config::tiny();
        let enc = Umt5Encoder::zeros(cfg, &device);
        let ids = Tensor::<B, 2, Int>::zeros([1, 6], &device);
        let out = enc.forward(ids, None);
        assert_eq!(out.dims(), [1, 6, 16]);
        let padded = pad_prompt_embeds(out, &[4], 8).unwrap();
        assert_eq!(padded.dims(), [1, 8, 16]);
    }
}
