//! UMT5 encoder on host CudaTensor (tiny + zeros bring-up).

use fastvideo_models::wan::Umt5Config;

use super::nn::{self, Linear};
use super::tensor::{CudaTensor, Result, TensorError};
use super::weights::{self, WeightMap};

fn t5_layer_norm(xs: &CudaTensor, weight: &CudaTensor, eps: f32) -> Result<CudaTensor> {
    // RMS over last dim (T5 style — no mean subtract).
    xs.rms_norm(weight, eps)
}

fn relative_position_bucket(
    seq_len: usize,
    num_buckets: usize,
    max_distance: usize,
) -> Vec<usize> {
    let mut buckets = vec![0usize; seq_len * seq_len];
    let num_buckets = num_buckets as i64;
    let max_exact = num_buckets / 4;
    let max_distance = max_distance as i64;
    for i in 0..seq_len as i64 {
        for j in 0..seq_len as i64 {
            let mut relative = j - i;
            let mut bucket = 0i64;
            let n_buckets = num_buckets / 2;
            // The upper half of the table is for keys *after* the query:
            // HF adds the offset on `relative_position > 0`. Inverting this
            // swaps the two halves and the encoder reads word order backwards.
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
struct DenseGated {
    wi_0: Linear,
    wi_1: Linear,
    wo: Linear,
}

impl DenseGated {
    fn zeros(cfg: &Umt5Config) -> Self {
        Self {
            wi_0: Linear::zeros(cfg.d_model, cfg.d_ff, true),
            wi_1: Linear::zeros(cfg.d_model, cfg.d_ff, true),
            wo: Linear::zeros(cfg.d_ff, cfg.d_model, true),
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &Umt5Config) -> Result<Self> {
        Ok(Self {
            wi_0: Linear::load(map, &weights::join_key(prefix, "wi_0"), cfg.d_model, cfg.d_ff, false)?,
            wi_1: Linear::load(map, &weights::join_key(prefix, "wi_1"), cfg.d_model, cfg.d_ff, false)?,
            wo: Linear::load(map, &weights::join_key(prefix, "wo"), cfg.d_ff, cfg.d_model, false)?,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let gelu = nn::gelu_tanh(&self.wi_0.forward(xs)?);
        let linear = self.wi_1.forward(xs)?;
        self.wo.forward(&gelu.mul(&linear)?)
    }
}

#[derive(Debug, Clone)]
struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    relative_bias: CudaTensor, // [buckets, heads]
    n_heads: usize,
    d_kv: usize,
}

impl SelfAttention {
    fn zeros(cfg: &Umt5Config) -> Self {
        let inner = cfg.num_heads * cfg.d_kv;
        Self {
            q: Linear::zeros(cfg.d_model, inner, false),
            k: Linear::zeros(cfg.d_model, inner, false),
            v: Linear::zeros(cfg.d_model, inner, false),
            o: Linear::zeros(inner, cfg.d_model, false),
            relative_bias: CudaTensor::zeros(&[cfg.relative_attention_num_buckets, cfg.num_heads]),
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &Umt5Config) -> Result<Self> {
        let inner = cfg.num_heads * cfg.d_kv;
        Ok(Self {
            q: Linear::load(map, &weights::join_key(prefix, "q"), cfg.d_model, inner, false)?,
            k: Linear::load(map, &weights::join_key(prefix, "k"), cfg.d_model, inner, false)?,
            v: Linear::load(map, &weights::join_key(prefix, "v"), cfg.d_model, inner, false)?,
            o: Linear::load(map, &weights::join_key(prefix, "o"), inner, cfg.d_model, false)?,
            relative_bias: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "relative_attention_bias.weight"),
                &[cfg.relative_attention_num_buckets, cfg.num_heads],
            )?,
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
        })
    }

    fn forward(&self, xs: &CudaTensor, buckets: &[usize]) -> Result<CudaTensor> {
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
        // scores [B,H,S,S]
        let k_t = k.transpose(2, 3)?;
        let mut scores = q.matmul(&k_t)?;
        // bias [H,S,S] from embedding lookup
        let mut bias = vec![0.0f32; self.n_heads * s * s];
        for (pos, &bucket) in buckets.iter().enumerate() {
            let row = bucket.min(self.relative_bias.shape[0] - 1);
            for h in 0..self.n_heads {
                bias[h * s * s + pos] = self.relative_bias.data[row * self.n_heads + h];
            }
        }
        let bias = CudaTensor::from_vec(bias, vec![1, self.n_heads, s, s])?;
        scores = scores.add(&bias)?;
        let attn = scores.softmax(-1)?;
        let ctx = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape(vec![b, s, self.n_heads * self.d_kv])?;
        self.o.forward(&ctx)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    attn: SelfAttention,
    ff: DenseGated,
    ln1: CudaTensor,
    ln2: CudaTensor,
    eps: f32,
}

impl EncoderLayer {
    fn zeros(cfg: &Umt5Config) -> Self {
        Self {
            attn: SelfAttention::zeros(cfg),
            ff: DenseGated::zeros(cfg),
            ln1: CudaTensor::ones(&[cfg.d_model]),
            ln2: CudaTensor::ones(&[cfg.d_model]),
            eps: cfg.eps as f32,
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &Umt5Config) -> Result<Self> {
        Ok(Self {
            attn: SelfAttention::load(
                map,
                &weights::join_key(prefix, "layer.0.SelfAttention"),
                cfg,
            )?,
            ln1: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "layer.0.layer_norm.weight"),
                &[cfg.d_model],
            )?,
            ff: DenseGated::load(map, &weights::join_key(prefix, "layer.1.DenseReluDense"), cfg)?,
            ln2: weights::cuda_tensor_shaped(
                map,
                &weights::join_key(prefix, "layer.1.layer_norm.weight"),
                &[cfg.d_model],
            )?,
            eps: cfg.eps as f32,
        })
    }

    fn forward(&self, xs: &CudaTensor, buckets: &[usize]) -> Result<CudaTensor> {
        let normed = t5_layer_norm(xs, &self.ln1, self.eps)?;
        let xs = xs.add(&self.attn.forward(&normed, buckets)?)?;
        let normed = t5_layer_norm(&xs, &self.ln2, self.eps)?;
        xs.add(&self.ff.forward(&normed)?)
    }
}

#[derive(Debug, Clone)]
pub struct Umt5Encoder {
    pub cfg: Umt5Config,
    embed: CudaTensor, // [vocab, d_model]
    layers: Vec<EncoderLayer>,
    final_ln: CudaTensor,
}

impl Umt5Encoder {
    pub fn zeros(cfg: Umt5Config) -> Self {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            layers.push(EncoderLayer::zeros(&cfg));
        }
        Self {
            embed: CudaTensor::zeros(&[cfg.vocab_size, cfg.d_model]),
            final_ln: CudaTensor::ones(&[cfg.d_model]),
            layers,
            cfg,
        }
    }

    pub fn load(cfg: Umt5Config, map: &WeightMap) -> Result<Self> {
        Self::from_map(cfg, map)
    }

    pub fn from_map(cfg: Umt5Config, map: &WeightMap) -> Result<Self> {
        let embed_key = if map.contains("encoder.embed_tokens.weight") {
            "encoder.embed_tokens.weight"
        } else {
            "shared.weight"
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(EncoderLayer::load(map, &format!("encoder.block.{i}"), &cfg)?);
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

    pub fn forward(&self, input_ids: &[u32], batch: usize, seq: usize) -> Result<CudaTensor> {
        if input_ids.len() != batch * seq {
            return Err(TensorError::Message("input_ids length mismatch".into()));
        }
        let indices: Vec<usize> = input_ids.iter().map(|&i| i as usize).collect();
        // The [vocab, d_model] table stays on the host (see `embedding_rows`).
        let mut hidden = self.embed.embedding_rows(&indices)?;
        hidden = hidden.reshape(vec![batch, seq, self.cfg.d_model])?;
        let buckets = relative_position_bucket(
            seq,
            self.cfg.relative_attention_num_buckets,
            self.cfg.relative_attention_max_distance,
        );
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &buckets)?;
        }
        t5_layer_norm(&hidden, &self.final_ln, self.cfg.eps as f32)
    }
}

pub fn pad_prompt_embeds(
    embeds: &CudaTensor,
    seq_lens: &[usize],
    text_len: usize,
) -> Result<CudaTensor> {
    let (b, _s, d) = (embeds.shape[0], embeds.shape[1], embeds.shape[2]);
    let mut rows = Vec::with_capacity(b);
    for (i, &len) in seq_lens.iter().enumerate() {
        let len = len.min(text_len);
        let row = embeds.narrow(0, i, 1)?.narrow(1, 0, len)?;
        let pad = CudaTensor::zeros(&[1, text_len - len, d]);
        rows.push(CudaTensor::cat(&[&row, &pad], 1)?);
    }
    let refs: Vec<&CudaTensor> = rows.iter().collect();
    CudaTensor::cat(&refs, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Buckets for a 12-token encoder, from HF's `UMT5Attention._relative_position_bucket`
    /// (`relative_buckets += (relative_position > 0) * num_buckets; abs(...)`).
    ///
    /// This is the one property the whole encoder's word order rests on: UMT5
    /// has no absolute or rotary positions, so a mirrored table is the only
    /// thing standing between "a dog running on a beach" and "a dog".
    #[test]
    fn relative_buckets_match_the_hf_reference() {
        let s = 12;
        let got = relative_position_bucket(s, 32, 128);
        // Query 0: every key is at or after it, so all land in the upper half.
        let row0: Vec<usize> = (0..s).map(|j| got[j]).collect();
        assert_eq!(row0, vec![0, 17, 18, 19, 20, 21, 22, 23, 24, 24, 24, 24]);
        // Query 5: keys before it stay low, keys after it take the +16 offset.
        let row5: Vec<usize> = (0..s).map(|j| got[5 * s + j]).collect();
        assert_eq!(row5, vec![5, 4, 3, 2, 1, 0, 17, 18, 19, 20, 21, 22]);
        // Query 11: every key is at or before it, so nothing takes the offset.
        let row11: Vec<usize> = (0..s).map(|j| got[11 * s + j]).collect();
        assert_eq!(row11, vec![8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
        // Only the diagonal is symmetric; swapping the halves would keep just it.
        for i in 0..s {
            assert_eq!(got[i * s + i], 0, "self-attention distance is bucket 0");
        }
    }
}
