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

/// Something that turns token ids into `hidden_states[tap]`. Two kinds exist:
/// the streaming encoder below, which holds nothing between prompts and pays
/// ~10 s of weight transfer per prompt, and a resident one ([`crate::llm`]'s
/// resident mode, FP8 to fit beside the DiT), which pays once. The pipeline
/// only ever sees this trait, so which one runs is a request-level choice.
pub trait HiddenStateEncoder {
    fn hidden_state(&self, ids: &[u32], tap: usize) -> Result<CudaTensor>;
    /// For reports: `"streamed"`, `"resident-bf16"`, `"resident-fp8"`.
    fn kind(&self) -> &'static str;
    /// Device bytes this encoder keeps between prompts.
    fn resident_bytes(&self) -> u64 {
        0
    }
}

/// One layer on the device at a time, nothing kept: the ~50 GB of decoder
/// weights stream through and are gone when a call returns.
pub struct StreamedEncoder<'a> {
    pub map: &'a WeightMap,
    pub cfg: &'a DecoderConfig,
}

impl HiddenStateEncoder for StreamedEncoder<'_> {
    fn hidden_state(&self, ids: &[u32], tap: usize) -> Result<CudaTensor> {
        encode_ids(self.map, self.cfg, ids, tap)
    }

    fn kind(&self) -> &'static str {
        "streamed"
    }
}

/// [`crate::llm::ResidentDecoder`]: layers 0..=49 stay on the device, a new
/// prompt costs a forward instead of 50 GB of transfers. At the checkpoint's
/// bf16 that is 50 GB, which does not fit beside the 41 GB DiT on a 96 GB card;
/// with weight-only FP8 rows (E4M3 codes + a scale per output row, dequantized
/// to bf16 per GEMM, activations untouched) it is 24.4 GB and does.
impl HiddenStateEncoder for crate::llm::ResidentDecoder {
    fn hidden_state(&self, ids: &[u32], tap: usize) -> Result<CudaTensor> {
        let positions: Vec<u32> = (0..ids.len() as u32).collect();
        let mut taps = self.hidden_states(ids, &positions, &vec![true; ids.len()], &[tap])?;
        taps.pop().ok_or_else(|| msg("h3 text: the resident decoder returned no hidden state"))
    }

    fn kind(&self) -> &'static str {
        match self.precision() {
            crate::llm::WeightPrecision::Native => "resident-bf16",
            crate::llm::WeightPrecision::Fp8Rows => "resident-fp8",
        }
    }

    fn resident_bytes(&self) -> u64 {
        self.device_bytes()
    }
}

/// Load the resident encoder for `root` (`root/text_encoder`, either layout):
/// exactly the layers tap 50 needs, no final norm. The precision is
/// per-instance; the process-wide `FASTVIDEO_FP8` flag is not involved (it
/// would quantize the DiT as well).
pub fn load_resident_encoder(root: &Path, precision: crate::llm::WeightPrecision) -> Result<crate::llm::ResidentDecoder> {
    let map = WeightMap::open(&root.join("text_encoder"))?;
    let cfg = DecoderConfig::qwen3_vl_32b_text().for_bf16_reference();
    crate::llm::ResidentDecoder::load_with(&map, &cfg, H3TextEncoderConfig::fasth3_8step().output_hidden_state_index, precision)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStatus {
    /// No cache directory was given.
    Disabled,
    Hit,
    Miss,
}

impl CacheStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            CacheStatus::Disabled => "disabled",
            CacheStatus::Hit => "hit",
            CacheStatus::Miss => "miss",
        }
    }
}

pub struct TextConditioning {
    pub ids: Vec<u32>,
    /// `[1, S, 5120]`, un-normed `hidden_states[50]`.
    pub hidden: CudaTensor,
    pub cache: CacheStatus,
    /// Which encoder ran; `"cache"` on a hit, when none did.
    pub encoder: &'static str,
}

/// The conditioning the DiT consumes for `prompt`.
///
/// `root` holds `tokenizer/tokenizer.json` and `text_encoder/*.safetensors`:
/// either the published snapshot or the slim re-pack of [`super::slim`] — the
/// loader resolves tensors by name, so the layouts are interchangeable (and
/// share cache entries, whose key is built from tensors, not files).
///
/// With `cache_dir`, a prompt seen before costs a file read: the shards are
/// mapped for their headers (the key needs the encoder's identity) but no
/// weight is streamed and no forward runs. `resident` replaces the streaming
/// encoder on a miss.
pub fn encode_prompt_with(root: &Path, prompt: &str, cache_dir: Option<&Path>, resident: Option<&dyn HiddenStateEncoder>) -> Result<TextConditioning> {
    use super::text_cache as cache;
    let ids = tokenize(root, prompt)?;
    let map = WeightMap::open(&root.join("text_encoder"))?;
    // The reference encoder runs in bf16; follow its constant casts.
    let cfg = DecoderConfig::qwen3_vl_32b_text().for_bf16_reference();
    let tap = H3TextEncoderConfig::fasth3_8step().output_hidden_state_index;
    let streamed = StreamedEncoder { map: &map, cfg: &cfg };
    let encoder: &dyn HiddenStateEncoder = resident.unwrap_or(&streamed);
    let Some(dir) = cache_dir else {
        let hidden = encoder.hidden_state(&ids, tap)?;
        return Ok(TextConditioning { ids, hidden, cache: CacheStatus::Disabled, encoder: encoder.kind() });
    };

    let tokenizer_path = root.join("tokenizer").join("tokenizer.json");
    let tokenizer_bytes = std::fs::read(&tokenizer_path).map_err(|e| msg(format!("{}: {e}", tokenizer_path.display())))?;
    let store = map.lazy().ok_or_else(|| msg("h3 text: the encoder checkpoint was not opened lazily"))?;
    let key = cache::cache_key(prompt, &cache::sha256(&tokenizer_bytes), tap, &cache::encoder_identity(store, &cfg, tap)?);
    let (entry, hit) = cache::get_or_compute(dir, &key, &ids, cfg.hidden, || Ok(encoder.hidden_state(&ids, tap)?.host_cow()?.into_owned()))?;
    let hidden = CudaTensor::from_vec(entry.data, vec![1, ids.len(), cfg.hidden])?.to_device()?;
    Ok(TextConditioning {
        ids,
        hidden,
        cache: if hit { CacheStatus::Hit } else { CacheStatus::Miss },
        encoder: if hit { "cache" } else { encoder.kind() },
    })
}

/// [`encode_prompt_with`] without a cache or a resident encoder.
pub fn encode_prompt(root: &Path, prompt: &str) -> Result<(Vec<u32>, CudaTensor)> {
    let text = encode_prompt_with(root, prompt, None, None)?;
    Ok((text.ids, text.hidden))
}

/// SearchingMan recovered-8B path: tokenize with the DiT snapshot's H3
/// tokenizer, encode with a resident [`super::recovered_8b::Recovered8bEncoder`]
/// (tap 24 + adapter → `[1, S, 5120]`). Cache entries are keyed as width 5120.
pub fn encode_prompt_recovered(
    tokenizer_root: &Path,
    _encoder_root: &Path,
    prompt: &str,
    cache_dir: Option<&Path>,
    resident: Option<&dyn HiddenStateEncoder>,
) -> Result<TextConditioning> {
    use super::recovered_8b::RECOVERED_8B_TAP;
    use super::text_cache as cache;
    let encoder = resident.ok_or_else(|| msg("recovered-8b: resident encoder required"))?;
    let ids = tokenize(tokenizer_root, prompt)?;
    let tap = RECOVERED_8B_TAP;
    let out_hidden = 5120usize;
    let Some(dir) = cache_dir else {
        let hidden = encoder.hidden_state(&ids, tap)?;
        return Ok(TextConditioning {
            ids,
            hidden,
            cache: CacheStatus::Disabled,
            encoder: encoder.kind(),
        });
    };
    let tokenizer_path = tokenizer_root.join("tokenizer").join("tokenizer.json");
    let tokenizer_bytes =
        std::fs::read(&tokenizer_path).map_err(|e| msg(format!("{}: {e}", tokenizer_path.display())))?;
    // Identity is the encoder kind + tap; recovered weights are not a LazyStore
    // under text_encoder/, so we fingerprint the kind string instead.
    let identity = cache::sha256(format!("recovered-8b-tap{tap}").as_bytes());
    let key = cache::cache_key(prompt, &cache::sha256(&tokenizer_bytes), tap, &identity);
    let (entry, hit) = cache::get_or_compute(dir, &key, &ids, out_hidden, || {
        Ok(encoder.hidden_state(&ids, tap)?.host_cow()?.into_owned())
    })?;
    let hidden = CudaTensor::from_vec(entry.data, vec![1, ids.len(), out_hidden])?.to_device()?;
    Ok(TextConditioning {
        ids,
        hidden,
        cache: if hit { CacheStatus::Hit } else { CacheStatus::Miss },
        encoder: if hit { "cache" } else { encoder.kind() },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Act, LayerAttn};

    fn tiny() -> DecoderConfig {
        DecoderConfig {
            vocab: 16,
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
    fn the_resident_encoder_is_the_streamed_one_with_the_weights_left_in_place() {
        let (cfg, ids, map) = (tiny(), [1u32, 3, 0, 2], weights());
        let streamed = StreamedEncoder { map: &map, cfg: &cfg }.hidden_state(&ids, 2).unwrap();
        let resident = crate::llm::ResidentDecoder::load(&map, &cfg, 2).unwrap();
        let again = HiddenStateEncoder::hidden_state(&resident, &ids, 2).unwrap();
        assert_eq!(&*streamed.host_cow().unwrap(), &*again.host_cow().unwrap());
        assert_eq!((StreamedEncoder { map: &map, cfg: &cfg }.kind(), resident.kind()), ("streamed", "resident-bf16"));
        // Weight-only FP8 is a different, nearby function: close, not equal, and it says what it is.
        let fp8 = crate::llm::ResidentDecoder::load_with(&map, &cfg, 2, crate::llm::WeightPrecision::Fp8Rows).unwrap();
        let quantized = HiddenStateEncoder::hidden_state(&fp8, &ids, 2).unwrap();
        let (a, b) = (streamed.host_cow().unwrap(), quantized.host_cow().unwrap());
        let (err, norm) = a.iter().zip(b.iter()).fold((0f64, 0f64), |(e, n), (x, y)| (e + f64::from(x - y).powi(2), n + f64::from(*x).powi(2)));
        let rel = (err / norm).sqrt();
        assert!(rel > 0.0 && rel < 0.1, "fp8 rows vs native: rel {rel}");
        assert_eq!(fp8.kind(), "resident-fp8");
    }

    #[test]
    fn a_tap_that_would_be_normed_is_refused() {
        let err = encode_ids(&weights(), &tiny(), &[1, 2], 3).unwrap_err();
        assert!(err.to_string().contains("post-norm"), "{err}");
    }
}
