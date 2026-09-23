//! Diffusers `T5EncoderModel` on host CudaTensor (classic Relu + shared relative bias).

use fastvideo_models::cosmos::T5Config;

use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{self, WeightMap};

fn t5_layer_norm(xs: &CudaTensor, weight: &CudaTensor, eps: f32) -> Result<CudaTensor> {
    xs.rms_norm(weight, eps)
}

fn relative_position_bucket(seq_len: usize, num_buckets: usize, max_distance: usize) -> Vec<usize> {
    let mut buckets = vec![0usize; seq_len * seq_len];
    let num_buckets = num_buckets as i64;
    let max_exact = num_buckets / 4;
    let max_distance = max_distance as i64;
    for i in 0..seq_len as i64 {
        for j in 0..seq_len as i64 {
            let mut relative = j - i;
            let mut bucket = 0i64;
            let n_buckets = num_buckets / 2;
            if relative > 0 {
                bucket += n_buckets;
            }
            relative = relative.abs();
            let is_small = relative < max_exact;
            let relative_log = ((relative as f64 / max_exact as f64).ln()
                / (max_distance as f64 / max_exact as f64).ln()
                * (n_buckets - max_exact) as f64)
                .floor() as i64
                + max_exact;
            let relative_bucket = if is_small {
                relative
            } else {
                relative_log.min(n_buckets - 1)
            };
            buckets[(i as usize) * seq_len + j as usize] = (bucket + relative_bucket) as usize;
        }
    }
    buckets
}

#[derive(Debug, Clone)]
enum DenseFf {
    Relu {
        wi: Linear,
        wo: Linear,
    },
    Gated {
        wi_0: Linear,
        wi_1: Linear,
        wo: Linear,
    },
}

impl DenseFf {
    fn zeros(cfg: &T5Config) -> Self {
        if cfg.is_gated {
            Self::Gated {
                wi_0: Linear::zeros(cfg.d_model, cfg.d_ff, false),
                wi_1: Linear::zeros(cfg.d_model, cfg.d_ff, false),
                wo: Linear::zeros(cfg.d_ff, cfg.d_model, false),
            }
        } else {
            Self::Relu {
                wi: Linear::zeros(cfg.d_model, cfg.d_ff, false),
                wo: Linear::zeros(cfg.d_ff, cfg.d_model, false),
            }
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &T5Config) -> Result<Self> {
        if cfg.is_gated {
            Ok(Self::Gated {
                wi_0: Linear::load(
                    map,
                    &weights::join_key(prefix, "wi_0"),
                    cfg.d_model,
                    cfg.d_ff,
                    false,
                )?,
                wi_1: Linear::load(
                    map,
                    &weights::join_key(prefix, "wi_1"),
                    cfg.d_model,
                    cfg.d_ff,
                    false,
                )?,
                wo: Linear::load(
                    map,
                    &weights::join_key(prefix, "wo"),
                    cfg.d_ff,
                    cfg.d_model,
                    false,
                )?,
            })
        } else {
            Ok(Self::Relu {
                wi: Linear::load(
                    map,
                    &weights::join_key(prefix, "wi"),
                    cfg.d_model,
                    cfg.d_ff,
                    false,
                )?,
                wo: Linear::load(
                    map,
                    &weights::join_key(prefix, "wo"),
                    cfg.d_ff,
                    cfg.d_model,
                    false,
                )?,
            })
        }
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        match self {
            Self::Relu { wi, wo } => {
                let h = wi.forward(xs)?.clamp(0.0, f32::MAX);
                wo.forward(&h)
            }
            Self::Gated { wi_0, wi_1, wo } => {
                let gelu = nn::gelu_tanh(&wi_0.forward(xs)?);
                let linear = wi_1.forward(xs)?;
                wo.forward(&gelu.mul(&linear)?)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    /// Only layer 0 owns the table; later layers reuse it.
    relative_bias: Option<CudaTensor>,
    n_heads: usize,
    d_kv: usize,
}

impl SelfAttention {
    fn zeros(cfg: &T5Config, has_bias: bool) -> Self {
        let inner = cfg.num_heads * cfg.d_kv;
        Self {
            q: Linear::zeros(cfg.d_model, inner, false),
            k: Linear::zeros(cfg.d_model, inner, false),
            v: Linear::zeros(cfg.d_model, inner, false),
            o: Linear::zeros(inner, cfg.d_model, false),
            relative_bias: if has_bias {
                Some(CudaTensor::zeros(&[
                    cfg.relative_attention_num_buckets,
                    cfg.num_heads,
                ]))
            } else {
                None
            },
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &T5Config, has_bias: bool) -> Result<Self> {
        let inner = cfg.num_heads * cfg.d_kv;
        Ok(Self {
            q: Linear::load(
                map,
                &weights::join_key(prefix, "q"),
                cfg.d_model,
                inner,
                false,
            )?,
            k: Linear::load(
                map,
                &weights::join_key(prefix, "k"),
                cfg.d_model,
                inner,
                false,
            )?,
            v: Linear::load(
                map,
                &weights::join_key(prefix, "v"),
                cfg.d_model,
                inner,
                false,
            )?,
            o: Linear::load(
                map,
                &weights::join_key(prefix, "o"),
                inner,
                cfg.d_model,
                false,
            )?,
            relative_bias: if has_bias {
                Some(weights::cuda_tensor_shaped(
                    map,
                    &weights::join_key(prefix, "relative_attention_bias.weight"),
                    &[cfg.relative_attention_num_buckets, cfg.num_heads],
                )?)
            } else {
                None
            },
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
        })
    }

    fn forward(
        &self,
        xs: &CudaTensor,
        buckets: &[usize],
        shared_bias: &CudaTensor,
        key_mask: Option<&[bool]>,
    ) -> Result<CudaTensor> {
        let (b, s, _) = (xs.shape[0], xs.shape[1], xs.shape[2]);
        let q = self
            .q
            .forward(xs)?
            .reshape(vec![b, s, self.n_heads, self.d_kv])?
            .transpose(1, 2)?;
        let k = self
            .k
            .forward(xs)?
            .reshape(vec![b, s, self.n_heads, self.d_kv])?
            .transpose(1, 2)?;
        let v = self
            .v
            .forward(xs)?
            .reshape(vec![b, s, self.n_heads, self.d_kv])?
            .transpose(1, 2)?;
        let k_t = k.transpose(2, 3)?;
        let mut scores = q.matmul(&k_t)?;
        let bias_table = self.relative_bias.as_ref().unwrap_or(shared_bias);
        let mut bias = vec![0.0f32; self.n_heads * s * s];
        for (pos, &bucket) in buckets.iter().enumerate() {
            let row = bucket.min(bias_table.shape[0] - 1);
            for h in 0..self.n_heads {
                bias[h * s * s + pos] = bias_table.data[row * self.n_heads + h];
            }
        }
        let bias = CudaTensor::from_vec(bias, vec![1, self.n_heads, s, s])?;
        scores = scores.add(&bias)?;
        if let Some(mask) = key_mask {
            if mask.len() != s {
                return Err(TensorError::Message(format!(
                    "t5 key_mask len {} vs seq {s}",
                    mask.len()
                )));
            }
            let mut host = scores.host_cow()?.to_vec();
            for bi in 0..b {
                for h in 0..self.n_heads {
                    for qi in 0..s {
                        for kj in 0..s {
                            if !mask[kj] {
                                let idx = (((bi * self.n_heads + h) * s + qi) * s) + kj;
                                host[idx] = -1.0e4;
                            }
                        }
                    }
                }
            }
            scores = CudaTensor::from_vec(host, scores.shape.clone())?;
        }
        let attn = scores.softmax(-1)?;
        let ctx =
            attn.matmul(&v)?
                .transpose(1, 2)?
                .reshape(vec![b, s, self.n_heads * self.d_kv])?;
        self.o.forward(&ctx)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    attn: SelfAttention,
    ff: DenseFf,
    ln1: CudaTensor,
    ln2: CudaTensor,
    eps: f32,
}

impl EncoderLayer {
    fn zeros(cfg: &T5Config, has_bias: bool) -> Self {
        Self {
            attn: SelfAttention::zeros(cfg, has_bias),
            ff: DenseFf::zeros(cfg),
            ln1: CudaTensor::ones(&[cfg.d_model]),
            ln2: CudaTensor::ones(&[cfg.d_model]),
            eps: cfg.eps as f32,
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &T5Config, has_bias: bool) -> Result<Self> {
        Ok(Self {
            attn: SelfAttention::load(
                map,
                &weights::join_key(prefix, "layer.0.SelfAttention"),
                cfg,
                has_bias,
            )?,
            ln1: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "layer.0.layer_norm.weight"),
                &[cfg.d_model],
            )?,
            ff: DenseFf::load(
                map,
                &weights::join_key(prefix, "layer.1.DenseReluDense"),
                cfg,
            )?,
            ln2: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "layer.1.layer_norm.weight"),
                &[cfg.d_model],
            )?,
            eps: cfg.eps as f32,
        })
    }

    fn forward(
        &self,
        xs: &CudaTensor,
        buckets: &[usize],
        shared_bias: &CudaTensor,
        key_mask: Option<&[bool]>,
    ) -> Result<CudaTensor> {
        let normed = t5_layer_norm(xs, &self.ln1, self.eps)?;
        let xs = xs.add(&self.attn.forward(&normed, buckets, shared_bias, key_mask)?)?;
        let normed = t5_layer_norm(&xs, &self.ln2, self.eps)?;
        xs.add(&self.ff.forward(&normed)?)
    }
}

#[derive(Debug, Clone)]
pub struct T5Encoder {
    pub cfg: T5Config,
    embed: CudaTensor,
    layers: Vec<EncoderLayer>,
    final_ln: CudaTensor,
}

impl T5Encoder {
    pub fn zeros(cfg: T5Config) -> Self {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(EncoderLayer::zeros(&cfg, i == 0));
        }
        Self {
            embed: CudaTensor::zeros(&[cfg.vocab_size, cfg.d_model]),
            final_ln: CudaTensor::ones(&[cfg.d_model]),
            layers,
            cfg,
        }
    }

    pub fn load(cfg: T5Config, map: &WeightMap) -> Result<Self> {
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
                i == 0,
            )?);
        }
        Ok(Self {
            embed: weights::cuda_tensor_shaped(map, embed_key, &[cfg.vocab_size, cfg.d_model])?,
            final_ln: weights::cuda_tensor_shaped(
                map,
                "encoder.final_layer_norm.weight",
                &[cfg.d_model],
            )?,
            layers,
            cfg,
        })
    }

    /// `input_ids` length `batch * seq`; `key_mask` length `seq` (true = attend).
    pub fn forward(
        &self,
        input_ids: &[u32],
        batch: usize,
        seq: usize,
        key_mask: Option<&[bool]>,
    ) -> Result<CudaTensor> {
        if input_ids.len() != batch * seq {
            return Err(TensorError::Message("t5 input_ids length mismatch".into()));
        }
        let indices: Vec<usize> = input_ids.iter().map(|&i| i as usize).collect();
        let mut hidden = self.embed.embedding_rows(&indices)?;
        hidden = hidden.reshape(vec![batch, seq, self.cfg.d_model])?;
        let buckets = relative_position_bucket(
            seq,
            self.cfg.relative_attention_num_buckets,
            self.cfg.relative_attention_max_distance,
        );
        let shared_bias = self.layers[0]
            .attn
            .relative_bias
            .as_ref()
            .ok_or_else(|| TensorError::Message("t5 missing layer-0 relative bias".into()))?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &buckets, shared_bias, key_mask)?;
        }
        t5_layer_norm(&hidden, &self.final_ln, self.cfg.eps as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_forward_shapes() {
        let cfg = T5Config::tiny();
        let enc = T5Encoder::zeros(cfg.clone());
        let ids = vec![1u32, 2, 3, 0];
        let mask = vec![true, true, true, false];
        let out = enc.forward(&ids, 1, 4, Some(&mask)).unwrap();
        assert_eq!(out.shape, vec![1, 4, cfg.d_model]);
    }
}
