//! Flux2 text helpers: chat wrap, tokenize, and Qwen3 / Mistral3 configs.
//!
//! Encoders live in `fastvideo-cudarc`. This module is host-only (no Candle).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flux2TextKind {
    /// FLUX.2-dev: Mistral3, layers (10, 20, 30).
    Mistral3,
    /// Klein: Qwen3, layers (9, 18, 27).
    Qwen3,
}

impl Flux2TextKind {
    pub fn from_preset(preset: &str) -> Self {
        if preset.contains("klein") {
            Self::Qwen3
        } else {
            Self::Mistral3
        }
    }

    pub fn out_layers(self) -> &'static [usize] {
        match self {
            Self::Mistral3 => &[10, 20, 30],
            Self::Qwen3 => &[9, 18, 27],
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mistral3 => "mistral3",
            Self::Qwen3 => "qwen3",
        }
    }
}

pub const FLUX2_SYSTEM_MESSAGE: &str =
    "You are an AI that reasons about image descriptions. You give structured \
     responses focusing on object relationships, object\nattribution and actions \
     without speculation.";

/// Qwen3 chat string matching BFL `apply_chat_template(..., enable_thinking=False)`.
pub fn format_qwen3_chat(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

/// Mistral3 / Pixtral-style instruct wrap used by Diffusers Flux2 `format_input`.
pub fn format_mistral3_chat(prompt: &str) -> String {
    let cleaned = prompt.replace("[IMG]", "");
    format!("[SYSTEM_PROMPT]{FLUX2_SYSTEM_MESSAGE}[/SYSTEM_PROMPT][INST]{cleaned}[/INST]")
}

pub fn format_flux2_prompt(kind: Flux2TextKind, prompt: &str) -> String {
    match kind {
        Flux2TextKind::Qwen3 => format_qwen3_chat(prompt),
        Flux2TextKind::Mistral3 => format_mistral3_chat(prompt),
    }
}

/// `FASTVIDEO_FLUX2_DUMMY_TEXT=1` keeps the prompt-hash stand-in (A/B vs real text).
pub fn flux2_dummy_text() -> bool {
    matches!(
        std::env::var("FASTVIDEO_FLUX2_DUMMY_TEXT").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// Override stacked-encoder sequence length (`FASTVIDEO_FLUX2_TEXT_LEN`).
pub fn flux2_text_len(default: usize) -> usize {
    std::env::var("FASTVIDEO_FLUX2_TEXT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
        .max(1)
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

/// Tokenize an already formatted Flux2 chat string. Specials live in the wrap.
pub fn tokenize_flux2(path: &str, formatted: &str, max_len: usize) -> Result<(Vec<u32>, usize), String> {
    let tokenizer = tokenizers::Tokenizer::from_file(path)
        .map_err(|e| format!("tokenizer load failed: {e}"))?;
    let encoding = tokenizer
        .encode(formatted, false)
        .map_err(|e| format!("tokenize failed: {e}"))?;
    let mut ids = encoding.get_ids().to_vec();
    if ids.is_empty() {
        ids.push(0);
    }
    if ids.len() > max_len {
        ids.truncate(max_len);
    }
    let len = ids.len();
    Ok((ids, len))
}

/// Decoder-only LM config shared by Qwen3 (Klein) and Mistral3 (dev).
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub text_len: usize,
    pub pad_token_id: u32,
    pub qk_norm: bool,
}

pub type Mistral3Config = Qwen3Config;

impl Qwen3Config {
    pub fn klein_4b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 2560,
            intermediate_size: 9728,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 40960,
            text_len: 512,
            pad_token_id: 151643,
            qk_norm: true,
        }
    }

    pub fn klein_9b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 12288,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            ..Self::klein_4b()
        }
    }

    pub fn from_preset(preset: &str) -> Self {
        if preset.contains("9b") {
            Self::klein_9b()
        } else if preset.contains("klein") {
            Self::klein_4b()
        } else {
            Self::mistral3_24b()
        }
    }

    pub fn mistral3_24b() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 5120,
            intermediate_size: 32768,
            num_hidden_layers: 40,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 1_000_000_000.0,
            max_position_embeddings: 131072,
            text_len: 512,
            pad_token_id: 0,
            qk_norm: false,
        }
    }

    pub fn tiny() -> Self {
        Self {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            text_len: 8,
            pad_token_id: 0,
            qk_norm: true,
        }
    }

    pub fn mistral3_tiny() -> Self {
        Self {
            qk_norm: false,
            pad_token_id: 0,
            ..Self::tiny()
        }
    }

    pub fn from_hf_json(kind: Flux2TextKind, raw: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(raw).map_err(|e| format!("text_encoder/config.json: {e}"))?;
        let text = v
            .get("text_config")
            .filter(|c| c.is_object())
            .cloned()
            .unwrap_or(v);
        let mut cfg = match kind {
            Flux2TextKind::Qwen3 => Self::klein_4b(),
            Flux2TextKind::Mistral3 => Self::mistral3_24b(),
        };
        if kind == Flux2TextKind::Qwen3 {
            if let Some(h) = text.get("hidden_size").and_then(|x| x.as_u64()) {
                if h == 4096 {
                    cfg = Self::klein_9b();
                }
            }
        }
        if let Some(n) = text.get("vocab_size").and_then(|x| x.as_u64()) {
            cfg.vocab_size = n as usize;
        }
        if let Some(n) = text.get("hidden_size").and_then(|x| x.as_u64()) {
            cfg.hidden_size = n as usize;
        }
        if let Some(n) = text.get("intermediate_size").and_then(|x| x.as_u64()) {
            cfg.intermediate_size = n as usize;
        }
        if let Some(n) = text.get("num_hidden_layers").and_then(|x| x.as_u64()) {
            cfg.num_hidden_layers = n as usize;
        }
        if let Some(n) = text.get("num_attention_heads").and_then(|x| x.as_u64()) {
            cfg.num_attention_heads = n as usize;
        }
        if let Some(n) = text.get("num_key_value_heads").and_then(|x| x.as_u64()) {
            cfg.num_key_value_heads = n as usize;
        }
        if let Some(n) = text.get("head_dim").and_then(|x| x.as_u64()) {
            cfg.head_dim = n as usize;
        }
        if let Some(n) = text.get("rms_norm_eps").and_then(|x| x.as_f64()) {
            cfg.rms_norm_eps = n;
        }
        if let Some(n) = text.get("rope_theta").and_then(|x| x.as_f64()) {
            cfg.rope_theta = n as f32;
        }
        if let Some(n) = text.get("max_position_embeddings").and_then(|x| x.as_u64()) {
            cfg.max_position_embeddings = n as usize;
        }
        if let Some(n) = text.get("pad_token_id").and_then(|x| x.as_u64()) {
            cfg.pad_token_id = n as u32;
        }
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_ids_keeps_valid_prefix() {
        let (out, valid) = pad_token_ids(&[1, 2, 3], 5, 0);
        assert_eq!(out, vec![1, 2, 3, 0, 0]);
        assert_eq!(valid, 3);
    }

    #[test]
    fn klein_uses_qwen3() {
        assert_eq!(Flux2TextKind::from_preset("flux2_klein_4b"), Flux2TextKind::Qwen3);
        assert_eq!(Flux2TextKind::from_preset("flux2_dev"), Flux2TextKind::Mistral3);
        assert_eq!(Qwen3Config::klein_4b().hidden_size, 2560);
        assert_eq!(Qwen3Config::mistral3_24b().hidden_size, 5120);
    }
}
