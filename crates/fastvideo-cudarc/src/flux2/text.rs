//! Qwen3 (Klein) and Mistral3 (dev) decoder-only encoders on `CudaTensor`.
//!
//! Mirrors `fastvideo_models::flux2::text`: RMSNorm + GQA + SwiGLU + NeoX RoPE,
//! QK-Norm only for Qwen3, then stack hidden layers `(9,18,27)` / `(10,20,30)`.

use fastvideo_models::flux2::{
    stack_layers_host, Flux2TextKind, Qwen3Config, FLUX2_SYSTEM_MESSAGE,
};

use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{self, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

pub fn flux2_dummy_text() -> bool {
    matches!(
        std::env::var("FASTVIDEO_FLUX2_DUMMY_TEXT").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

pub fn flux2_text_len(default: usize) -> usize {
    std::env::var("FASTVIDEO_FLUX2_TEXT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
        .max(1)
}

pub fn format_flux2_prompt(kind: Flux2TextKind, prompt: &str) -> String {
    match kind {
        Flux2TextKind::Qwen3 => {
            format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        }
        Flux2TextKind::Mistral3 => {
            let cleaned = prompt.replace("[IMG]", "");
            format!("[SYSTEM_PROMPT]{FLUX2_SYSTEM_MESSAGE}[/SYSTEM_PROMPT][INST]{cleaned}[/INST]")
        }
    }
}

pub fn pad_token_ids(ids: &[u32], text_len: usize, pad_id: u32) -> (Vec<u32>, usize) {
    let valid = ids.len().min(text_len).max(1);
    let mut out = vec![pad_id; text_len];
    let copy = ids.len().min(text_len);
    if copy > 0 {
        out[..copy].copy_from_slice(&ids[..copy]);
    }
    (out, valid)
}

fn detect_text_prefix(map: &WeightMap, vocab: usize, hidden: usize) -> Result<String> {
    const PREFIXES: &[&str] = &[
        "",
        "model",
        "language_model.model",
        "model.language_model.model",
        "language_model",
        "model.language_model",
    ];
    let mut last = None;
    for prefix in PREFIXES {
        let key = if prefix.is_empty() {
            "embed_tokens.weight".to_string()
        } else {
            format!("{prefix}.embed_tokens.weight")
        };
        match weights::cuda_tensor_shaped(map, &key, &[vocab, hidden]) {
            Ok(_) => return Ok((*prefix).to_string()),
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| msg("no embed_tokens.weight under known Flux2 text prefixes")))
}

fn key(prefix: &str, rest: &str) -> String {
    if prefix.is_empty() {
        rest.to_string()
    } else {
        format!("{prefix}.{rest}")
    }
}

struct RmsNorm {
    weight: CudaTensor,
    eps: f32,
}

impl RmsNorm {
    fn zeros(dim: usize, eps: f32) -> Self {
        Self {
            weight: CudaTensor::from_vec(vec![1.0; dim], vec![dim]).expect("rms"),
            eps,
        }
    }

    fn load(map: &WeightMap, prefix: &str, dim: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: weights::cuda_tensor_shaped(map, &weights::join_key(prefix, "weight"), &[dim])?,
            eps,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        nn::rms_norm(xs, &self.weight, self.eps)
    }
}

fn apply_head_rms(norm: &RmsNorm, xs: &CudaTensor) -> Result<CudaTensor> {
    let last = *xs.shape.last().ok_or_else(|| msg("head rms on scalar"))?;
    let flat = xs.reshape(vec![xs.numel() / last, last])?;
    let y = norm.forward(&flat)?;
    y.reshape(xs.shape.clone())
}

fn repeat_kv(xs: &CudaTensor, reps: usize) -> Result<CudaTensor> {
    if reps == 1 {
        return Ok(xs.clone());
    }
    let h = xs.shape[2];
    let mut heads = Vec::with_capacity(h * reps);
    for hi in 0..h {
        let slice = xs.narrow(2, hi, 1)?;
        for _ in 0..reps {
            heads.push(slice.clone());
        }
    }
    let refs: Vec<&CudaTensor> = heads.iter().collect();
    CudaTensor::cat(&refs, 2)
}

fn apply_rope_neox(xs: &CudaTensor, theta: f32) -> Result<CudaTensor> {
    let (b, s, h, d) = (xs.shape[0], xs.shape[1], xs.shape[2], xs.shape[3]);
    let half = d / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(i as f32 / half as f32))
        .collect();
    let mut cos = vec![0.0f32; b * s * h * half];
    let mut sin = vec![0.0f32; b * s * h * half];
    for p in 0..s {
        for i in 0..half {
            let ang = (p as f32) * inv[i];
            let (c, sn) = (ang.cos(), ang.sin());
            for bi in 0..b {
                for hi in 0..h {
                    let dst = ((bi * s + p) * h + hi) * half + i;
                    cos[dst] = c;
                    sin[dst] = sn;
                }
            }
        }
    }
    let cos = CudaTensor::from_vec(cos, vec![b, s, h, half])?;
    let sin = CudaTensor::from_vec(sin, vec![b, s, h, half])?;
    let x1 = xs.narrow(3, 0, half)?;
    let x2 = xs.narrow(3, half, half)?;
    let y1 = x1.mul(&cos)?.sub(&x2.mul(&sin)?)?;
    let y2 = x2.mul(&cos)?.add(&x1.mul(&sin)?)?;
    CudaTensor::cat(&[&y1, &y2], 3)
}

fn text_sdpa(q: &CudaTensor, k: &CudaTensor, v: &CudaTensor, mask: &CudaTensor) -> Result<CudaTensor> {
    let d = q.shape[3] as f32;
    let scale = 1.0 / d.sqrt();
    let mut scores = q.matmul(&k.transpose(2, 3)?)?.try_mul_scalar(scale)?;
    scores = scores.add(mask)?;
    scores.softmax(-1)?.matmul(v)
}

fn causal_pad_mask(batch: usize, heads: usize, seq: usize, valid_len: Option<usize>) -> Result<CudaTensor> {
    let valid = valid_len.unwrap_or(seq).min(seq);
    let mut data = vec![0.0f32; batch * heads * seq * seq];
    for q in 0..seq {
        for k in 0..seq {
            let v = if k > q || k >= valid { -1e9 } else { 0.0 };
            for bi in 0..batch {
                for h in 0..heads {
                    data[((bi * heads + h) * seq + q) * seq + k] = v;
                }
            }
        }
    }
    CudaTensor::from_vec(data, vec![batch, heads, seq, seq])
}

struct DecoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
}

impl DecoderAttention {
    fn zeros(cfg: &Qwen3Config) -> Self {
        let q = cfg.num_attention_heads * cfg.head_dim;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        Self {
            q_proj: Linear::zeros(cfg.hidden_size, q, false),
            k_proj: Linear::zeros(cfg.hidden_size, kv, false),
            v_proj: Linear::zeros(cfg.hidden_size, kv, false),
            o_proj: Linear::zeros(q, cfg.hidden_size, false),
            q_norm: cfg.qk_norm.then(|| RmsNorm::zeros(cfg.head_dim, cfg.rms_norm_eps as f32)),
            k_norm: cfg.qk_norm.then(|| RmsNorm::zeros(cfg.head_dim, cfg.rms_norm_eps as f32)),
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &Qwen3Config) -> Result<Self> {
        let q = cfg.num_attention_heads * cfg.head_dim;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let attn = weights::join_key(prefix, "self_attn");
        Ok(Self {
            q_proj: Linear::load(map, &weights::join_key(&attn, "q_proj"), cfg.hidden_size, q, false)?,
            k_proj: Linear::load(map, &weights::join_key(&attn, "k_proj"), cfg.hidden_size, kv, false)?,
            v_proj: Linear::load(map, &weights::join_key(&attn, "v_proj"), cfg.hidden_size, kv, false)?,
            o_proj: Linear::load(map, &weights::join_key(&attn, "o_proj"), q, cfg.hidden_size, false)?,
            q_norm: if cfg.qk_norm {
                Some(RmsNorm::load(
                    map,
                    &weights::join_key(&attn, "q_norm"),
                    cfg.head_dim,
                    cfg.rms_norm_eps as f32,
                )?)
            } else {
                None
            },
            k_norm: if cfg.qk_norm {
                Some(RmsNorm::load(
                    map,
                    &weights::join_key(&attn, "k_norm"),
                    cfg.head_dim,
                    cfg.rms_norm_eps as f32,
                )?)
            } else {
                None
            },
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    fn forward(&self, xs: &CudaTensor, valid_len: Option<usize>) -> Result<CudaTensor> {
        let (b, s, _) = (xs.shape[0], xs.shape[1], xs.shape[2]);
        let q = self
            .q_proj
            .forward(xs)?
            .reshape(vec![b, s, self.heads, self.head_dim])?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape(vec![b, s, self.kv_heads, self.head_dim])?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape(vec![b, s, self.kv_heads, self.head_dim])?;
        let q = match &self.q_norm {
            Some(n) => apply_head_rms(n, &q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(n) => apply_head_rms(n, &k)?,
            None => k,
        };
        let q = apply_rope_neox(&q, self.rope_theta)?;
        let k = apply_rope_neox(&k, self.rope_theta)?;
        let k = repeat_kv(&k, self.heads / self.kv_heads.max(1))?;
        let v = repeat_kv(&v, self.heads / self.kv_heads.max(1))?;
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let mask = causal_pad_mask(b, self.heads, s, valid_len)?;
        // Local SDPA — do not use Wan sequence-parallel `scaled_dot_product_attention`.
        let attn = text_sdpa(&q, &k, &v, &mask)?;
        let attn = attn
            .transpose(1, 2)?
            .reshape(vec![b, s, self.heads * self.head_dim])?;
        self.o_proj.forward(&attn)
    }
}

struct DecoderMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl DecoderMlp {
    fn zeros(cfg: &Qwen3Config) -> Self {
        Self {
            gate: Linear::zeros(cfg.hidden_size, cfg.intermediate_size, false),
            up: Linear::zeros(cfg.hidden_size, cfg.intermediate_size, false),
            down: Linear::zeros(cfg.intermediate_size, cfg.hidden_size, false),
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &Qwen3Config) -> Result<Self> {
        let mlp = weights::join_key(prefix, "mlp");
        Ok(Self {
            gate: Linear::load(
                map,
                &weights::join_key(&mlp, "gate_proj"),
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
            )?,
            up: Linear::load(
                map,
                &weights::join_key(&mlp, "up_proj"),
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
            )?,
            down: Linear::load(
                map,
                &weights::join_key(&mlp, "down_proj"),
                cfg.intermediate_size,
                cfg.hidden_size,
                false,
            )?,
        })
    }

    fn forward(&self, xs: &CudaTensor) -> Result<CudaTensor> {
        let gated = self.gate.forward(xs)?.silu();
        self.down.forward(&gated.mul(&self.up.forward(xs)?)?)
    }
}

struct DecoderLayer {
    input_norm: RmsNorm,
    attn: DecoderAttention,
    post_norm: RmsNorm,
    mlp: DecoderMlp,
}

impl DecoderLayer {
    fn zeros(cfg: &Qwen3Config) -> Self {
        Self {
            input_norm: RmsNorm::zeros(cfg.hidden_size, cfg.rms_norm_eps as f32),
            attn: DecoderAttention::zeros(cfg),
            post_norm: RmsNorm::zeros(cfg.hidden_size, cfg.rms_norm_eps as f32),
            mlp: DecoderMlp::zeros(cfg),
        }
    }

    fn load(map: &WeightMap, prefix: &str, cfg: &Qwen3Config) -> Result<Self> {
        Ok(Self {
            input_norm: RmsNorm::load(
                map,
                &weights::join_key(prefix, "input_layernorm"),
                cfg.hidden_size,
                cfg.rms_norm_eps as f32,
            )?,
            attn: DecoderAttention::load(map, prefix, cfg)?,
            post_norm: RmsNorm::load(
                map,
                &weights::join_key(prefix, "post_attention_layernorm"),
                cfg.hidden_size,
                cfg.rms_norm_eps as f32,
            )?,
            mlp: DecoderMlp::load(map, prefix, cfg)?,
        })
    }

    fn forward(&self, xs: &CudaTensor, valid_len: Option<usize>) -> Result<CudaTensor> {
        let h = xs.add(&self.attn.forward(&self.input_norm.forward(xs)?, valid_len)?)?;
        h.add(&self.mlp.forward(&self.post_norm.forward(&h)?)?)
    }
}

pub struct Qwen3Encoder {
    embed: CudaTensor,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    pub cfg: Qwen3Config,
}

pub type Mistral3Encoder = Qwen3Encoder;

impl Qwen3Encoder {
    pub fn zeros(cfg: Qwen3Config) -> Self {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::zeros(&cfg));
        }
        Self {
            embed: CudaTensor::zeros(&[cfg.vocab_size, cfg.hidden_size]),
            layers,
            norm: RmsNorm::zeros(cfg.hidden_size, cfg.rms_norm_eps as f32),
            cfg,
        }
    }

    pub fn load(cfg: Qwen3Config, map: &WeightMap) -> Result<Self> {
        let prefix = detect_text_prefix(map, cfg.vocab_size, cfg.hidden_size)?;
        let embed = weights::cuda_tensor_shaped(map, &key(&prefix, "embed_tokens.weight"), &[
            cfg.vocab_size,
            cfg.hidden_size,
        ])?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(map, &key(&prefix, &format!("layers.{i}")), &cfg)?);
        }
        Ok(Self {
            embed,
            layers,
            norm: RmsNorm::load(map, &key(&prefix, "norm"), cfg.hidden_size, cfg.rms_norm_eps as f32)?,
            cfg,
        })
    }

    pub fn forward_hidden(
        &self,
        input_ids: &[u32],
        batch: usize,
        seq: usize,
        valid_len: Option<usize>,
    ) -> Result<(CudaTensor, Vec<CudaTensor>)> {
        if input_ids.len() != batch * seq {
            return Err(msg("input_ids length mismatch"));
        }
        let indices: Vec<usize> = input_ids
            .iter()
            .map(|&i| (i as usize).min(self.cfg.vocab_size.saturating_sub(1)))
            .collect();
        let mut hidden = self.embed.embedding_rows(&indices)?.reshape(vec![batch, seq, self.cfg.hidden_size])?;
        let mut all = vec![hidden.clone()];
        for layer in &self.layers {
            hidden = layer.forward(&hidden, valid_len)?;
            all.push(hidden.clone());
        }
        hidden = self.norm.forward(&hidden)?;
        *all.last_mut().unwrap() = hidden.clone();
        Ok((hidden, all))
    }
}

pub struct Flux2TextEncoder {
    pub kind: Flux2TextKind,
    lm: Option<Qwen3Encoder>,
    dummy_dim: usize,
}

impl Flux2TextEncoder {
    pub fn dummy(kind: Flux2TextKind, dim: usize) -> Self {
        Self {
            kind,
            lm: None,
            dummy_dim: dim,
        }
    }

    pub fn qwen3(enc: Qwen3Encoder) -> Self {
        Self {
            kind: Flux2TextKind::Qwen3,
            lm: Some(enc),
            dummy_dim: 0,
        }
    }

    pub fn mistral3(enc: Mistral3Encoder) -> Self {
        Self {
            kind: Flux2TextKind::Mistral3,
            lm: Some(enc),
            dummy_dim: 0,
        }
    }

    pub fn lm_config(&self) -> Option<&Qwen3Config> {
        self.lm.as_ref().map(|e| &e.cfg)
    }

    pub fn encode_ids(&self, ids: &[u32], valid_len: Option<usize>) -> Result<CudaTensor> {
        if let Some(enc) = &self.lm {
            let seq = ids.len();
            let (_last, all) = enc.forward_hidden(ids, 1, seq, valid_len)?;
            return stack_selected(&all, self.kind.out_layers());
        }
        let seq = ids.len().max(1);
        let data: Vec<f32> = (0..seq * self.dummy_dim)
            .map(|i| ((ids[i % ids.len()] as f32) * 0.01 + (i as f32) * 0.001) % 1.0)
            .collect();
        CudaTensor::from_vec(data, vec![1, seq, self.dummy_dim])
    }
}

fn stack_selected(all: &[CudaTensor], layers: &[usize]) -> Result<CudaTensor> {
    let mut owned = Vec::new();
    let mut refs: Vec<Vec<f32>> = Vec::new();
    for &idx in layers {
        let t = all
            .get(idx.min(all.len().saturating_sub(1)))
            .ok_or_else(|| msg(format!("missing hidden layer {idx}")))?;
        refs.push(t.host_cow()?.to_vec());
    }
    let (seq, hidden) = {
        let t = &all[0];
        (t.shape[1], t.shape[2])
    };
    for row in &refs {
        if row.len() != seq * hidden {
            return Err(msg("stacked hidden shape mismatch"));
        }
        owned.push(row.as_slice());
    }
    let stacked = stack_layers_host(&owned, seq, hidden).map_err(msg)?;
    CudaTensor::from_vec(stacked, vec![1, seq, layers.len() * hidden])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_qwen3_encodes() {
        let enc = Qwen3Encoder::zeros(Qwen3Config::tiny());
        let ids = vec![1u32, 2, 3, 4];
        let (last, all) = enc.forward_hidden(&ids, 1, 4, None).unwrap();
        assert_eq!(last.shape, vec![1, 4, 16]);
        assert_eq!(all.len(), 3);
        let text = Flux2TextEncoder::qwen3(enc);
        let stacked = text.encode_ids(&ids, None).unwrap();
        assert_eq!(stacked.shape, vec![1, 4, 48]);
    }

    #[test]
    fn tiny_mistral3_encodes() {
        let enc = Qwen3Encoder::zeros(Qwen3Config::mistral3_tiny());
        let text = Flux2TextEncoder::mistral3(enc);
        let stacked = text.encode_ids(&[1, 2, 3, 4], Some(3)).unwrap();
        assert_eq!(stacked.shape, vec![1, 4, 48]);
        assert_eq!(text.kind, Flux2TextKind::Mistral3);
    }

}
