//! NCII prompt guard: [`hfmlsoc/ncii-light-guard-v01`](https://huggingface.co/hfmlsoc/ncii-light-guard-v01)
//! (MIT; a rank-8 LoRA fine-tune of `microsoft/harrier-oss-v1-270m`, MIT),
//! a 270M Gemma3 text classifier that scores whether an image-edit prompt
//! risks creating non-consensual intimate imagery. It is what the
//! `MiniMaxAI/MiniMax-H3-Turbo-Lora` Space runs (`ncii_guard.py`) on every
//! request that carries a keyframe. Text only: it never sees the image.
//!
//! This is a plain-Rust CPU port of `Gemma3TextForSequenceClassification`
//! (`transformers` 5.x) with the repo's own `tokenizer.json` (BOS + prompt +
//! EOS; the pipeline's `truncation=True` truncates at `model_max_length`,
//! which this repo leaves unbounded, so the file's 256-token truncation is
//! off here too; [`NCII_MAX_TOKENS`] bounds the CPU work):
//! scaled embeddings (the scale rounded to bf16, as the bf16 checkpoint
//! applies it), 18 pre/post-normed layers (Gemma RMSNorm `x * (1 + w)`,
//! q/k norms, RoPE theta 1e6 on the 256-wide heads, one KV head for four
//! query heads, causal attention with scale `1/16`, GELU-tanh gated MLP), the
//! final norm and the `score` head read at the last token. Weights are bf16
//! on disk; the linears are held in f32 (about 0.4 GB), the embedding table
//! stays bf16 (0.34 GB). docs/ports/h3-turbo-loras.md "NCII guard".

use std::collections::HashMap;
use std::path::Path;

use rayon::prelude::*;

/// Files the guard reads from its directory.
pub const NCII_FILES: [&str; 3] = ["config.json", "model.safetensors", "tokenizer.json"];
/// The Hub repo and pinned revision (`weights-revisions.tsv`).
pub const NCII_REPO: &str = "hfmlsoc/ncii-light-guard-v01";
pub const NCII_REVISION: &str = "6a44e8b4b608078db164b0c946947922b2b98eab";
/// The tree under the weight root.
pub const NCII_TREE: &str = "auxiliary/ncii-light-guard";
/// Tokens classified at most (the first ones; BOS included). The pipeline
/// itself has no bound; H3 prompts stay far below it (1024 Qwen tokens).
pub const NCII_MAX_TOKENS: usize = 2048;

/// The architecture numbers the port supports (checked against `config.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct NciiConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub query_pre_attn_scalar: f64,
    pub pad_token_id: u32,
    /// `id2label`, index = class.
    pub labels: Vec<String>,
}

impl NciiConfig {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("ncii config.json: {e}"))?;
        let u = |k: &str| -> Result<usize, String> {
            v.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|x| x as usize)
                .ok_or_else(|| format!("ncii config.json: missing {k}"))
        };
        if v.get("model_type").and_then(|x| x.as_str()) != Some("gemma3_text") {
            return Err("ncii config.json: model_type is not gemma3_text".into());
        }
        if v.get("use_bidirectional_attention")
            .and_then(|x| x.as_bool())
            == Some(true)
        {
            return Err("ncii config.json: bidirectional attention is not supported".into());
        }
        for k in ["attn_logit_softcapping", "final_logit_softcapping"] {
            if v.get(k).is_some_and(|x| !x.is_null()) {
                return Err(format!("ncii config.json: {k} is not supported"));
            }
        }
        if v.get("hidden_activation").and_then(|x| x.as_str()) != Some("gelu_pytorch_tanh") {
            return Err("ncii config.json: hidden_activation is not gelu_pytorch_tanh".into());
        }
        let layers = u("num_hidden_layers")?;
        if let Some(types) = v.get("layer_types").and_then(|x| x.as_array()) {
            if types.len() != layers || types.iter().any(|t| t.as_str() != Some("full_attention")) {
                return Err("ncii config.json: only full-attention layers are supported".into());
            }
        }
        let rope_theta = v
            .pointer("/rope_parameters/full_attention/rope_theta")
            .or_else(|| v.get("rope_theta"))
            .and_then(serde_json::Value::as_f64)
            .ok_or("ncii config.json: missing rope_theta")?;
        if v.pointer("/rope_parameters/full_attention/rope_type")
            .and_then(|x| x.as_str())
            .is_some_and(|t| t != "default")
        {
            return Err("ncii config.json: only the default RoPE is supported".into());
        }
        let id2label = v
            .get("id2label")
            .and_then(|x| x.as_object())
            .ok_or("ncii config.json: missing id2label")?;
        let mut labels = vec![String::new(); id2label.len()];
        for (k, l) in id2label {
            let i: usize = k
                .parse()
                .map_err(|_| format!("ncii config.json: id2label key {k}"))?;
            *labels
                .get_mut(i)
                .ok_or("ncii config.json: id2label is not dense")? =
                l.as_str().unwrap_or_default().to_owned();
        }
        Ok(Self {
            hidden: u("hidden_size")?,
            layers,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            intermediate: u("intermediate_size")?,
            vocab: u("vocab_size")?,
            eps: v
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-6) as f32,
            rope_theta,
            query_pre_attn_scalar: v
                .get("query_pre_attn_scalar")
                .and_then(|x| x.as_f64())
                .unwrap_or(u("head_dim")? as f64),
            pad_token_id: u("pad_token_id").unwrap_or(0) as u32,
            labels,
        })
    }
}

/// The classifier's answer for one prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct NciiVerdict {
    /// The top label (`safe` or `ncii`), as `transformers`' pipeline returns it.
    pub label: String,
    /// Softmax probability of the top label (the pipeline's `score`).
    pub score: f32,
    /// Softmax probability of `ncii`.
    pub ncii: f32,
    /// Tokens classified (BOS and EOS included, after truncation).
    pub tokens: usize,
}

impl NciiVerdict {
    pub fn is_ncii(&self) -> bool {
        self.label == "ncii"
    }
}

struct Layer {
    input_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    pre_ffn_norm: Vec<f32>,
    post_ffn_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

/// The loaded guard. `classify` is `&self` and thread-safe.
pub struct NciiGuard {
    cfg: NciiConfig,
    tokenizer: tokenizers::Tokenizer,
    /// bf16 bits, `[vocab, hidden]`.
    embed: Vec<u16>,
    embed_scale: f32,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    score: Vec<f32>,
    ncii_index: usize,
    /// `[head_dim / 2]` inverse frequencies.
    inv_freq: Vec<f64>,
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// Round to bf16 (nearest even), as `torch.tensor(x).to(bfloat16)`.
pub fn round_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// A safetensors file: name -> (dtype, shape, bytes range).
struct SafeTensors {
    data: Vec<u8>,
    start: usize,
    tensors: HashMap<String, (String, Vec<usize>, usize, usize)>,
}

impl SafeTensors {
    fn read(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if data.len() < 8 {
            return Err(format!("{}: not a safetensors file", path.display()));
        }
        let n = u64::from_le_bytes(data[..8].try_into().expect("8 bytes")) as usize;
        let header = data
            .get(8..8 + n)
            .ok_or_else(|| format!("{}: truncated header", path.display()))?;
        let header: serde_json::Value = serde_json::from_slice(header)
            .map_err(|e| format!("{}: header: {e}", path.display()))?;
        let mut tensors = HashMap::new();
        for (name, t) in header
            .as_object()
            .ok_or("safetensors header is not an object")?
        {
            if name == "__metadata__" {
                continue;
            }
            let dtype = t["dtype"].as_str().unwrap_or_default().to_owned();
            let shape = t["shape"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64())
                        .map(|x| x as usize)
                        .collect()
                })
                .unwrap_or_default();
            let off = t["data_offsets"].as_array().ok_or("missing data_offsets")?;
            let (a, b) = (
                off.first().and_then(|x| x.as_u64()).unwrap_or(0) as usize,
                off.get(1).and_then(|x| x.as_u64()).unwrap_or(0) as usize,
            );
            tensors.insert(name.clone(), (dtype, shape, a, b));
        }
        let start = 8 + n;
        Ok(Self {
            data,
            start,
            tensors,
        })
    }

    fn raw(&self, name: &str, shape: &[usize]) -> Result<(&str, &[u8]), String> {
        let (dtype, s, a, b) = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("ncii weights: missing {name}"))?;
        if s != shape {
            return Err(format!(
                "ncii weights: {name} has shape {s:?}, expected {shape:?}"
            ));
        }
        let bytes = self
            .data
            .get(self.start + a..self.start + b)
            .ok_or_else(|| format!("ncii weights: {name} is out of the file"))?;
        Ok((dtype.as_str(), bytes))
    }

    fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let (dtype, bytes) = self.raw(name, shape)?;
        let n: usize = shape.iter().product();
        let out: Vec<f32> = match dtype {
            "BF16" => bytes
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            "F32" => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            other => {
                return Err(format!(
                    "ncii weights: {name} is {other}; BF16 or F32 expected"
                ))
            }
        };
        if out.len() != n {
            return Err(format!(
                "ncii weights: {name} holds {} values, expected {n}",
                out.len()
            ));
        }
        Ok(out)
    }

    fn bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>, String> {
        let (dtype, bytes) = self.raw(name, shape)?;
        match dtype {
            "BF16" => Ok(bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()),
            "F32" => Ok(bytes
                .chunks_exact(4)
                .map(|c| {
                    (round_bf16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_bits() >> 16)
                        as u16
                })
                .collect()),
            other => Err(format!(
                "ncii weights: {name} is {other}; BF16 or F32 expected"
            )),
        }
    }
}

/// `y[n, out] = x[n, inn] @ w[out, inn]^T`.
fn linear(x: &[f32], n: usize, inn: usize, w: &[f32], out: usize) -> Vec<f32> {
    let mut y = vec![0f32; n * out];
    y.par_chunks_mut(out).enumerate().for_each(|(t, row)| {
        let xr = &x[t * inn..(t + 1) * inn];
        for (o, dst) in row.iter_mut().enumerate() {
            let wr = &w[o * inn..(o + 1) * inn];
            *dst = xr.iter().zip(wr).map(|(a, b)| a * b).sum();
        }
    });
    y
}

/// Gemma RMSNorm over rows of `dim`: `x * rsqrt(mean(x^2) + eps) * (1 + w)`.
fn rms_norm(x: &mut [f32], dim: usize, w: &[f32], eps: f32) {
    for row in x.chunks_exact_mut(dim) {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let r = 1.0 / (ms + eps).sqrt();
        for (v, g) in row.iter_mut().zip(w) {
            *v = *v * r * (1.0 + g);
        }
    }
}

fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2 / pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

impl NciiGuard {
    /// Load `config.json`, `model.safetensors` and `tokenizer.json` from `dir`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        Self::load_with(dir, "model.safetensors")
    }

    /// [`Self::load`] with another name for the weights file.
    pub fn load_with(dir: &Path, weights: &str) -> Result<Self, String> {
        let cfg = NciiConfig::from_json(
            &std::fs::read_to_string(dir.join("config.json"))
                .map_err(|e| format!("{}: {e}", dir.join("config.json").display()))?,
        )?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| format!("{}: {e}", dir.join("tokenizer.json").display()))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| format!("ncii tokenizer: {e}"))?;
        let st = SafeTensors::read(&dir.join(weights))?;
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        let embed = st.bf16("model.embed_tokens.weight", &[cfg.vocab, h])?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("model.layers.{i}");
            let f = |n: &str, s: &[usize]| st.f32(&format!("{p}.{n}"), s);
            layers.push(Layer {
                input_norm: f("input_layernorm.weight", &[h])?,
                post_attn_norm: f("post_attention_layernorm.weight", &[h])?,
                pre_ffn_norm: f("pre_feedforward_layernorm.weight", &[h])?,
                post_ffn_norm: f("post_feedforward_layernorm.weight", &[h])?,
                q: f("self_attn.q_proj.weight", &[cfg.heads * hd, h])?,
                k: f("self_attn.k_proj.weight", &[cfg.kv_heads * hd, h])?,
                v: f("self_attn.v_proj.weight", &[cfg.kv_heads * hd, h])?,
                o: f("self_attn.o_proj.weight", &[h, cfg.heads * hd])?,
                q_norm: f("self_attn.q_norm.weight", &[hd])?,
                k_norm: f("self_attn.k_norm.weight", &[hd])?,
                gate: f("mlp.gate_proj.weight", &[cfg.intermediate, h])?,
                up: f("mlp.up_proj.weight", &[cfg.intermediate, h])?,
                down: f("mlp.down_proj.weight", &[h, cfg.intermediate])?,
            });
        }
        let norm = st.f32("model.norm.weight", &[h])?;
        let score = st.f32("score.weight", &[cfg.labels.len(), h])?;
        let ncii_index = cfg
            .labels
            .iter()
            .position(|l| l == "ncii")
            .ok_or("ncii config.json: no `ncii` label")?;
        let inv_freq = (0..hd / 2)
            .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f64 / hd as f64))
            .collect();
        Ok(Self {
            // `Gemma3TextScaledWordEmbedding`: sqrt(hidden) cast to the weight dtype (bf16).
            embed_scale: round_bf16((h as f32).sqrt()),
            cfg,
            tokenizer,
            embed,
            layers,
            norm,
            score,
            ncii_index,
            inv_freq,
        })
    }

    pub fn config(&self) -> &NciiConfig {
        &self.cfg
    }

    /// Token ids as the pipeline sees them (BOS, prompt, EOS; truncated).
    pub fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, String> {
        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| format!("ncii tokenizer: {e}"))?;
        let ids = enc.get_ids();
        Ok(ids[..ids.len().min(NCII_MAX_TOKENS)].to_vec())
    }

    /// The `score` logits at the last non-pad token.
    pub fn logits(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let (h, hd, nh, nkv) = (c.hidden, c.head_dim, c.heads, c.kv_heads);
        let n = ids.len();
        if n == 0 {
            return Err("ncii: empty input".into());
        }
        let mut x = vec![0f32; n * h];
        for (t, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= c.vocab {
                return Err(format!("ncii: token {id} out of the vocabulary"));
            }
            for (d, e) in x[t * h..(t + 1) * h]
                .iter_mut()
                .zip(&self.embed[id * h..(id + 1) * h])
            {
                *d = bf16_to_f32(*e) * self.embed_scale;
            }
        }
        // RoPE tables: `cat(freqs, freqs)` with rotate_half.
        let half = hd / 2;
        let mut cos = vec![0f32; n * half];
        let mut sin = vec![0f32; n * half];
        for t in 0..n {
            for i in 0..half {
                let a = t as f64 * self.inv_freq[i];
                cos[t * half + i] = a.cos() as f32;
                sin[t * half + i] = a.sin() as f32;
            }
        }
        let rope = |v: &mut [f32], t: usize| {
            for i in 0..half {
                let (a, b) = (v[i], v[i + half]);
                let (co, si) = (cos[t * half + i], sin[t * half + i]);
                v[i] = a * co - b * si;
                v[i + half] = b * co + a * si;
            }
        };
        let scale = (1.0 / c.query_pre_attn_scalar.sqrt()) as f32;
        let group = nh / nkv;
        for layer in &self.layers {
            // Attention block.
            let mut hn = x.clone();
            rms_norm(&mut hn, h, &layer.input_norm, c.eps);
            let mut q = linear(&hn, n, h, &layer.q, nh * hd);
            let mut k = linear(&hn, n, h, &layer.k, nkv * hd);
            let v = linear(&hn, n, h, &layer.v, nkv * hd);
            rms_norm(&mut q, hd, &layer.q_norm, c.eps);
            rms_norm(&mut k, hd, &layer.k_norm, c.eps);
            for t in 0..n {
                for head in 0..nh {
                    rope(&mut q[(t * nh + head) * hd..][..hd], t);
                }
                for head in 0..nkv {
                    rope(&mut k[(t * nkv + head) * hd..][..hd], t);
                }
            }
            let mut attn = vec![0f32; n * nh * hd];
            attn.par_chunks_mut(nh * hd)
                .enumerate()
                .for_each(|(t, out)| {
                    let mut p = vec![0f32; t + 1];
                    for head in 0..nh {
                        let kvh = head / group;
                        let qv = &q[(t * nh + head) * hd..][..hd];
                        let mut max = f32::NEG_INFINITY;
                        for (s, ps) in p.iter_mut().enumerate() {
                            let kv = &k[(s * nkv + kvh) * hd..][..hd];
                            *ps = qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale;
                            max = max.max(*ps);
                        }
                        let mut sum = 0f32;
                        for ps in p.iter_mut() {
                            *ps = (*ps - max).exp();
                            sum += *ps;
                        }
                        let o = &mut out[head * hd..(head + 1) * hd];
                        for (s, ps) in p.iter().enumerate() {
                            let w = ps / sum;
                            for (d, vv) in o.iter_mut().zip(&v[(s * nkv + kvh) * hd..][..hd]) {
                                *d += w * vv;
                            }
                        }
                    }
                });
            let mut a = linear(&attn, n, nh * hd, &layer.o, h);
            rms_norm(&mut a, h, &layer.post_attn_norm, c.eps);
            for (xv, av) in x.iter_mut().zip(&a) {
                *xv += av;
            }
            // MLP block.
            let mut hn = x.clone();
            rms_norm(&mut hn, h, &layer.pre_ffn_norm, c.eps);
            let g = linear(&hn, n, h, &layer.gate, c.intermediate);
            let u = linear(&hn, n, h, &layer.up, c.intermediate);
            let act: Vec<f32> = g.iter().zip(&u).map(|(g, u)| gelu_tanh(*g) * u).collect();
            let mut m = linear(&act, n, c.intermediate, &layer.down, h);
            rms_norm(&mut m, h, &layer.post_ffn_norm, c.eps);
            for (xv, mv) in x.iter_mut().zip(&m) {
                *xv += mv;
            }
        }
        // `Gemma3ForSequenceClassification`: the last token that is not padding.
        let last = ids
            .iter()
            .rposition(|&id| id != c.pad_token_id)
            .unwrap_or(n - 1);
        let mut row = x[last * h..(last + 1) * h].to_vec();
        rms_norm(&mut row, h, &self.norm, c.eps);
        Ok(linear(&row, 1, h, &self.score, c.labels.len()))
    }

    /// Classify one prompt (the Space's `classifier(prompt, truncation=True)[0]`).
    pub fn classify(&self, prompt: &str) -> Result<NciiVerdict, String> {
        let ids = self.tokenize(prompt)?;
        let logits = self.logits(&ids)?;
        let probs = softmax(&logits);
        let top = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(NciiVerdict {
            label: self.cfg.labels[top].clone(),
            score: probs[top],
            ncii: probs[self.ncii_index],
            tokens: ids.len(),
        })
    }
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{"model_type":"gemma3_text","hidden_size":8,"num_hidden_layers":1,
        "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"intermediate_size":6,
        "vocab_size":16,"rms_norm_eps":1e-6,"query_pre_attn_scalar":4,"pad_token_id":0,
        "hidden_activation":"gelu_pytorch_tanh","use_bidirectional_attention":false,
        "attn_logit_softcapping":null,"final_logit_softcapping":null,
        "layer_types":["full_attention"],
        "rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"}},
        "id2label":{"0":"safe","1":"ncii"}}"#;

    #[test]
    fn config_parses_and_refuses_unsupported() {
        let c = NciiConfig::from_json(CONFIG).unwrap();
        assert_eq!((c.hidden, c.heads, c.kv_heads, c.head_dim), (8, 2, 1, 4));
        assert_eq!(c.labels, ["safe", "ncii"]);
        assert_eq!(c.rope_theta, 1e6);
        for (from, to) in [
            ("\"gemma3_text\"", "\"llama\""),
            (
                "\"use_bidirectional_attention\":false",
                "\"use_bidirectional_attention\":true",
            ),
            (
                "\"layer_types\":[\"full_attention\"]",
                "\"layer_types\":[\"sliding_attention\"]",
            ),
            (
                "\"final_logit_softcapping\":null",
                "\"final_logit_softcapping\":30.0",
            ),
        ] {
            assert!(
                NciiConfig::from_json(&CONFIG.replace(from, to)).is_err(),
                "{to}"
            );
        }
    }

    #[test]
    fn bf16_rounding_and_embed_scale() {
        // sqrt(640) = 25.298 -> bf16 25.25, what the bf16 checkpoint multiplies by.
        assert_eq!(round_bf16(640f32.sqrt()), 25.25);
        assert_eq!(round_bf16(1.0), 1.0);
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
    }

    #[test]
    fn norms_and_activation() {
        let mut x = vec![3.0f32, 4.0];
        rms_norm(&mut x, 2, &[0.0, 1.0], 0.0);
        // rms = sqrt(12.5); weights apply as (1 + w).
        let r = 12.5f32.sqrt();
        assert!((x[0] - 3.0 / r).abs() < 1e-6 && (x[1] - 8.0 / r).abs() < 1e-6);
        assert!((gelu_tanh(1.0) - 0.841_192).abs() < 1e-5);
        assert_eq!(gelu_tanh(0.0), 0.0);
        let y = linear(&[1.0, 2.0], 1, 2, &[1.0, 0.0, 0.5, 0.5], 2);
        assert_eq!(y, vec![1.0, 1.5]);
        let p = softmax(&[0.0, 0.0]);
        assert_eq!(p, vec![0.5, 0.5]);
    }

    /// The real guard against `transformers`' scores, when the weights are
    /// present: `FV_NCII_DIR=<dir> cargo test -p fastvideo-models ncii -- --ignored`.
    /// `ncii_reference.json` (beside this file's tests, generated by
    /// scripts/gpu/ncii_reference.py) lists prompts with the pipeline's label
    /// and `ncii` probability.
    #[test]
    #[ignore]
    fn matches_transformers_when_weights_are_present() {
        // FV_NCII_DIR, else an untracked copy at tests/fixtures/ncii-guard
        // (how the build pod, which takes no environment, gets it).
        let local =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ncii-guard");
        // (`*.safetensors` is gitignored, and the pod sync skips ignored files:
        // the local copy is named `model.st`.)
        let (dir, weights) = match std::env::var_os("FV_NCII_DIR") {
            Some(d) => (std::path::PathBuf::from(d), "model.safetensors"),
            None if local.join("model.st").is_file() => (local, "model.st"),
            None => {
                eprintln!(
                    "ncii: no weights (FV_NCII_DIR or tests/fixtures/ncii-guard/model.st); skipped"
                );
                return;
            }
        };
        let t = std::time::Instant::now();
        let guard = NciiGuard::load_with(&dir, weights).unwrap();
        eprintln!("ncii: loaded in {:.2} s", t.elapsed().as_secs_f64());
        let reference: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/ncii_reference.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let mut worst = 0f32;
        for case in reference["cases"].as_array().unwrap() {
            let prompt = case["prompt"].as_str().unwrap();
            let want_ids: Vec<u32> = case["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(guard.tokenize(prompt).unwrap(), want_ids, "{prompt}");
            let t = std::time::Instant::now();
            let got = guard.classify(prompt).unwrap();
            eprintln!(
                "ncii: {:5} p={:.4} (transformers {:.4}) {} tokens {:.3} s",
                got.label,
                got.ncii,
                case["ncii"].as_f64().unwrap(),
                got.tokens,
                t.elapsed().as_secs_f64()
            );
            let want = case["ncii"].as_f64().unwrap() as f32;
            worst = worst.max((got.ncii - want).abs());
            assert_eq!(
                got.label,
                case["label"].as_str().unwrap(),
                "{prompt}: {got:?} vs {want}"
            );
        }
        eprintln!("ncii: max |p_ncii - transformers| = {worst}");
        assert!(worst < 0.02, "{worst}");
    }
}
