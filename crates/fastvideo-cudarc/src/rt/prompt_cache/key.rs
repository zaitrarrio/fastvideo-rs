//! The cache key: one derivation for every family (ADR-0021 "Key").
//!
//! Each pre-B4c cache hashed the same shape of input, a SHA-256 over
//! length-prefixed fields in the order
//!
//! ```text
//! domain | prompt | sha256(tokenizer.json) | length as u64 LE | encoder identity fields...
//! ```
//!
//! and differed only in the domain string, in whether the prompt is trimmed
//! first, and in what `length` and the identity fields are. Those differences
//! are the [`KeyRecipe`] (domain, [`PromptNorm`]) and the arguments of
//! [`KeyRecipe::key`]; the three recipes are [`H3_KEY`], [`WAN_KEY`] and
//! [`LTX2_KEY`]. B4c2 gave each a new domain and more identity fields (the
//! weights read, the numerics, LTX's mode); the derivation itself is
//! unchanged, and the pre-B4c2 recipes stay as `*_KEY_V1` for the tests.

use sha2::{Digest, Sha256};

/// SHA-256 with length-prefixed fields, so `("ab", "c")` and `("a", "bc")`
/// hash differently: each field is its length as a little-endian `u64`, then
/// its bytes. `sha2` rather than std's hasher: these digests are file names on
/// disk and must not change with the Rust version.
pub struct Fields(Sha256);

impl Fields {
    pub fn new() -> Self {
        Self(Sha256::new())
    }

    pub fn field(&mut self, data: &[u8]) {
        self.0.update((data.len() as u64).to_le_bytes());
        self.0.update(data);
    }

    pub fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

impl Default for Fields {
    fn default() -> Self {
        Self::new()
    }
}

pub fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

/// Lower-case hex, two digits per byte: the entry's file stem.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// How the prompt enters the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptNorm {
    /// The prompt's bytes as given (H3, Wan: their tokenizers see it as is).
    Verbatim,
    /// `str::trim` first (LTX: the pipeline strips the prompt before
    /// tokenising, `ltx2/text.rs` `PaddedPrompt::tokenize`).
    Trim,
}

impl PromptNorm {
    pub fn apply(self, prompt: &str) -> &str {
        match self {
            Self::Verbatim => prompt,
            Self::Trim => prompt.trim(),
        }
    }
}

/// One family's key derivation: its domain string and prompt normalisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyRecipe {
    /// The first field; it versions the scheme and keeps families apart.
    pub domain: &'static [u8],
    pub prompt: PromptNorm,
}

/// H3 before B4c2 (`h3/text_cache.rs` `cache_key` pre-B4c): the domain is
/// the entry format's magic, `H3TEXT01`; `length` is the tap (hidden-state
/// index, 50); the identity is one field, the tensor fingerprint of
/// `h3::text_cache::encoder_identity` (FP8 and recovered-8B variants derived
/// from it, `h3/text.rs`). Superseded by [`H3_KEY`]; kept so the tests pin
/// what entries on disk were written under.
pub const H3_KEY_V1: KeyRecipe = KeyRecipe {
    domain: &super::H3_MAGIC,
    prompt: PromptNorm::Verbatim,
};

/// H3 since B4c2 (ADR-0021 "B4c2"): the domain is new, so no entry written
/// under [`H3_KEY_V1`] is ever looked up; `length` is the tap; the identity
/// fields are the encoder identity (recovered 8B: a fingerprint of its three
/// weight files), the weights read
/// ([`super::numerics::WeightsRead`]), the FP8 tree's identity (empty unless
/// the tree is read) and the [`super::numerics::Numerics`].
pub const H3_KEY: KeyRecipe = KeyRecipe {
    domain: b"fastvideo h3 text conditioning v2",
    prompt: PromptNorm::Verbatim,
};

/// Wan before B4c2 (`wan/pipeline.rs` `text_cache_key` pre-B4c): `WANTEXT1`;
/// `length` is the DiT's `text_len` (512); the identity is one field,
/// [`super::identity::files_head`] of `text_encoder/`. Superseded by
/// [`WAN_KEY`].
pub const WAN_KEY_V1: KeyRecipe = KeyRecipe {
    domain: b"WANTEXT1",
    prompt: PromptNorm::Verbatim,
};

/// Wan since B4c2: `WANTEXT2`; the identity fields are
/// [`super::identity::files_head`] of `text_encoder/` and the
/// [`super::numerics::Numerics`] (gap 3: the GEMM math and the card).
pub const WAN_KEY: KeyRecipe = KeyRecipe {
    domain: b"WANTEXT2",
    prompt: PromptNorm::Verbatim,
};

/// LTX-2 before B4c2 (`ltx2/text_cache.rs` `cache_key` pre-B4c): the domain
/// names the scheme; the prompt is trimmed; `length` is
/// `max_sequence_length` (1024); the identity is three fields: the
/// text-encoder kind (encoder and feature normalisation,
/// `TextEncoder::text_encoder_kind`), then
/// [`super::identity::files_head_tail`] of Gemma and of the connectors.
/// Superseded by [`LTX2_KEY`].
pub const LTX2_KEY_V1: KeyRecipe = KeyRecipe {
    domain: b"fastvideo ltx2 text conditioning v1",
    prompt: PromptNorm::Trim,
};

/// LTX-2 since B4c2: v2 of the scheme; the identity fields are the three of
/// [`LTX2_KEY_V1`], then the weights Gemma's linears were read from
/// ([`super::numerics::WeightsRead`]: bf16, FP8 rows quantised at load, or
/// the FP8 tree), the FP8 tree's identity (empty unless read), the mode
/// (`resident` / `streamed`) and the [`super::numerics::Numerics`].
pub const LTX2_KEY: KeyRecipe = KeyRecipe {
    domain: b"fastvideo ltx2 text conditioning v2",
    prompt: PromptNorm::Trim,
};

impl KeyRecipe {
    /// `domain | prompt | tokenizer_sha256 | length | encoder[0] | encoder[1] ...`,
    /// every one a length-prefixed field.
    pub fn key(
        &self,
        prompt: &str,
        tokenizer_sha256: &[u8; 32],
        length: u64,
        encoder: &[&[u8]],
    ) -> [u8; 32] {
        let mut h = Fields::new();
        h.field(self.domain);
        h.field(self.prompt.apply(prompt).as_bytes());
        h.field(tokenizer_sha256);
        h.field(&length.to_le_bytes());
        for part in encoder {
            h.field(part);
        }
        h.finish()
    }
}
