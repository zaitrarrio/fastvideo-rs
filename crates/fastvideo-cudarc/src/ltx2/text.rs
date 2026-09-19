//! The text path: prompt → Gemma-3-12B hidden states → the two connector
//! transformers → one context per stream.
//!
//! LTX-2 does not condition on a text encoder's output. It stacks **all 49**
//! hidden states of a decoder-only LLM (the scaled embeddings, 47 raw layer
//! outputs and the normed last one), normalises each state separately, mixes
//! the 188 160-wide stack down to 3840 with one bias-free Linear, and runs the
//! result through two small transformers — one for the video stream, one for
//! the audio stream — whose padding slots are filled with learned *register*
//! tokens. After that there is no padding left, which is why nothing
//! downstream is ever masked.
//!
//! What is host-side here, and why that is not a CPU fallback: the per-state
//! statistics (a masked mean, min and max over `tokens × 3840` values, 49
//! times) run once per prompt on the host copy of the stack. The backend has
//! no device reduction, the input has to cross the bus anyway to be packed
//! into `[tokens, 188160]`, and the references do this step in bf16 — a
//! 3.9M-term bf16 sum — so there is no "device arithmetic" worth reproducing;
//! f64 accumulation is the definition. Everything from `text_proj_in` onward
//! is on the device.
//!
//! Only the real tokens are ever computed. The pipeline left-pads to 1024 and
//! runs all 1024 rows through Gemma; the pad rows come out as attention-masked
//! garbage that the normalisation zeroes and the registers overwrite. Running
//! the `n` real tokens at rotary positions `1024-n … 1023` is the same
//! function of the same numbers. See docs/ports/ltx2.md §b.

use std::path::Path;

use fastvideo_models::ltx2::config::Ltx2ConnectorsConfig;
use fastvideo_models::ltx2::rope::connector_fractions;
use fastvideo_models::ltx2::SplitRope;

use crate::llm::{self, DecoderConfig};
use crate::wan::nn::Linear;
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

use super::attention::{Attention, AttentionDims, DeviceRope, FeedForward};
use super::keys::Keys;
use super::{msg, ones};

/// A prompt as the pipeline presents it to Gemma: left-padded to a fixed length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaddedPrompt {
    /// `max_len` ids; the first `max_len - real` are `<pad>`.
    pub ids: Vec<u32>,
    /// How many trailing ids are real (`<bos>` included).
    pub real: usize,
}

/// Gemma's `<pad>`.
pub const PAD_ID: u32 = 0;

impl PaddedPrompt {
    /// Left-pad already-tokenised ids (special tokens included) to `max_len`,
    /// keeping the first `max_len` when there are more — the tokenizer's
    /// truncation side is the right even though its padding side is the left.
    pub fn from_ids(ids: &[u32], max_len: usize) -> Result<Self> {
        if ids.is_empty() || max_len == 0 {
            return Err(msg(format!("prompt: {} ids into {max_len} slots", ids.len())));
        }
        let real = ids.len().min(max_len);
        let mut padded = vec![PAD_ID; max_len - real];
        padded.extend_from_slice(&ids[..real]);
        Ok(Self { ids: padded, real })
    }

    /// `pipeline_ltx2.py:327-341`: the stripped prompt, no chat template, no
    /// system prompt, `<bos>` prepended by the tokenizer's own post-processor.
    pub fn tokenize(tokenizer_json: &Path, prompt: &str, max_len: usize) -> Result<Self> {
        let path = tokenizer_json.to_str().ok_or_else(|| msg(format!("tokenizer path {} is not UTF-8", tokenizer_json.display())))?;
        let (ids, _) = fastvideo_models::tokenize_prompt(path, prompt.trim(), max_len)
            .map_err(|e| msg(format!("tokenize with {}: {e}", tokenizer_json.display())))?;
        Self::from_ids(&ids, max_len)
    }

    pub fn max_len(&self) -> usize {
        self.ids.len()
    }

    pub fn real_ids(&self) -> &[u32] {
        &self.ids[self.ids.len() - self.real..]
    }

    /// Rotary positions of the real tokens: transformers numbers a left-padded
    /// sequence `0..S` across the padding, so they are the last `real` integers.
    pub fn real_positions(&self) -> Vec<u32> {
        ((self.ids.len() - self.real) as u32..self.ids.len() as u32).collect()
    }

    /// 1 for real tokens, 0 for padding — the tokenizer's `attention_mask`.
    pub fn attention_mask(&self) -> Vec<u32> {
        (0..self.ids.len()).map(|i| u32::from(i >= self.ids.len() - self.real)).collect()
    }
}

/// The hidden states of the real tokens, one `[tokens · hidden]` vector per
/// state, on the host.
#[derive(Debug, Clone)]
pub struct HiddenStack {
    pub tokens: usize,
    pub hidden: usize,
    pub states: Vec<Vec<f32>>,
}

impl HiddenStack {
    /// From the oracle's / pipeline's `[tokens, hidden, states]` layout.
    pub fn from_interleaved(data: &[f32], tokens: usize, hidden: usize, states: usize) -> Result<Self> {
        if data.len() != tokens * hidden * states || data.is_empty() {
            return Err(msg(format!("hidden stack: {} values for [{tokens}, {hidden}, {states}]", data.len())));
        }
        let states = (0..states).map(|l| data.iter().skip(l).step_by(states).copied().collect()).collect();
        Ok(Self { tokens, hidden, states })
    }

    /// Run Gemma over the real tokens and keep every hidden state. `cfg` decides
    /// the embedding scale (`for_bf16_reference` to match the product).
    pub fn encode(map: &WeightMap, cfg: &DecoderConfig, prompt: &PaddedPrompt) -> Result<Self> {
        let taps: Vec<usize> = (0..=cfg.num_layers()).collect();
        let out = llm::hidden_states(map, cfg, prompt.real_ids(), &prompt.real_positions(), &vec![true; prompt.real], &taps)?;
        let states = out.iter().map(|t| Ok(t.host_cow()?.into_owned())).collect::<Result<Vec<_>>>()?;
        Ok(Self { tokens: prompt.real, hidden: cfg.hidden, states })
    }

    /// `per_layer_masked_mean_norm` over the real tokens, packed the way
    /// `text_proj_in` wants it: `[tokens, hidden · states]`, feature index
    /// `channel · states + state`.
    ///
    /// Each state on its own: `8 · (x - mean) / (max - min + eps)` with the mean,
    /// min and max taken over every token and channel of that state. Pad rows
    /// are zero in the reference and `text_proj_in` has no bias, so leaving them
    /// out changes nothing.
    pub fn normalized(&self, scale: f64, eps: f64) -> Vec<f32> {
        let (n, width) = (self.states.len(), self.hidden * self.states.len());
        let mut out = vec![0f32; self.tokens * width];
        for (l, state) in self.states.iter().enumerate() {
            let sum: f64 = state.iter().map(|&v| f64::from(v)).sum();
            let mean = sum / (state.len() as f64 + eps);
            let (lo, hi) = state.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            let k = scale / (f64::from(hi) - f64::from(lo) + eps);
            for (i, &v) in state.iter().enumerate() {
                let (t, c) = (i / self.hidden, i % self.hidden);
                out[t * width + c * n + l] = ((f64::from(v) - mean) * k) as f32;
            }
        }
        out
    }
}

/// One connector's hyper-parameters, out of the flat `connectors/config.json`.
#[derive(Debug, Clone, Copy)]
struct ConnectorShape {
    heads: usize,
    head_dim: usize,
    layers: usize,
    registers: usize,
}

struct Block1d {
    attn: Attention,
    ff: FeedForward,
}

/// `LTX2ConnectorTransformer1d`: registers, a 1-D split rotary, pre-norm blocks
/// with weightless RMSNorms, a weightless RMSNorm out.
pub struct Connector {
    registers: CudaTensor,
    blocks: Vec<Block1d>,
    ones: CudaTensor,
    heads: usize,
    dim: usize,
    eps: f32,
}

impl Connector {
    fn load(map: &WeightMap, keys: &Keys, name: &str, shape: ConnectorShape, eps: f32) -> Result<Self> {
        let ConnectorShape { heads, head_dim, layers, registers } = shape;
        let dim = heads * head_dim;
        let dims = AttentionDims { query_dim: dim, context_dim: dim, heads, head_dim };
        let blocks = (0..layers)
            .map(|i| {
                let p = format!("{name}.transformer_blocks.{i}");
                Ok(Block1d {
                    attn: Attention::load(map, keys, &format!("{p}.attn1"), dims, eps)?,
                    ff: FeedForward::load(map, keys, &format!("{p}.ff"), dim, dim * 4)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut table = cuda_tensor_shaped(map, &keys.key(&format!("{name}.learnable_registers")), &[registers, dim])?;
        table.pin_device()?;
        Ok(Self { registers: table, blocks, ones: ones(dim)?, heads, dim, eps })
    }

    /// `proj`: `[tokens, dim]`, the real tokens only, in order. Returns
    /// `[1, total, dim]`: the real tokens at the front, every later position
    /// `p` filled with `registers[p mod R]`, then attended without a mask.
    fn forward(&self, proj: &CudaTensor, rope: &DeviceRope, total: usize) -> Result<CudaTensor> {
        let [tokens, dim] = proj.shape[..] else {
            return Err(msg(format!("connector expects [tokens, {}], got {:?}", self.dim, proj.shape)));
        };
        let count = self.registers.shape[0];
        if dim != self.dim || tokens == 0 || tokens > total || count == 0 || !total.is_multiple_of(count) {
            return Err(msg(format!("connector: {tokens} tokens of width {dim} into {total} slots with {count} registers")));
        }
        let mut x = if tokens == total {
            proj.clone()
        } else {
            let fill: Vec<usize> = (tokens..total).map(|p| p % count).collect();
            CudaTensor::cat(&[proj, &self.registers.index_select_rows(&fill)?], 0)?
        }
        .reshape(vec![1, total, dim])?;
        for block in &self.blocks {
            let h = x.rms_norm(&self.ones, self.eps)?;
            x = x.add(&block.attn.forward(&h, None, Some(rope), None)?)?;
            let h = x.rms_norm(&self.ones, self.eps)?;
            x = x.add(&block.ff.forward(&h)?)?;
        }
        x.rms_norm(&self.ones, self.eps)
    }
}

/// What the DiT receives: one `[1, 1024, 3840]` context per stream, no mask.
pub struct TextContexts {
    /// `text_proj_in` output for the real tokens, `[tokens, 3840]` — kept
    /// because it is the first thing worth diffing against the reference.
    pub proj: CudaTensor,
    pub video: CudaTensor,
    pub audio: CudaTensor,
}

/// `LTX2TextConnectors` for LTX-2.0: one shared projection, two connectors.
pub struct TextConnectors {
    text_proj_in: Linear,
    video: Connector,
    audio: Connector,
    cfg: Ltx2ConnectorsConfig,
}

impl TextConnectors {
    pub fn load(map: &WeightMap, keys: &Keys, cfg: &Ltx2ConnectorsConfig) -> Result<Self> {
        if cfg.per_modality_projections || cfg.proj_bias {
            return Err(msg("connectors: per-modality projections / projection bias are LTX-2.3, not supported"));
        }
        if cfg.video_connector_num_attention_heads * cfg.video_connector_attention_head_dim != cfg.caption_channels
            || cfg.audio_connector_num_attention_heads * cfg.audio_connector_attention_head_dim != cfg.caption_channels
        {
            return Err(msg("connectors: LTX-2.0 connectors are as wide as the caption channels"));
        }
        let eps = cfg.norm_eps as f32;
        Ok(Self {
            text_proj_in: Linear::load(map, &keys.text_proj_in(map)?, cfg.text_proj_in_features(), cfg.caption_channels, false)?,
            video: Connector::load(
                map,
                keys,
                "video_connector",
                ConnectorShape {
                    heads: cfg.video_connector_num_attention_heads,
                    head_dim: cfg.video_connector_attention_head_dim,
                    layers: cfg.video_connector_num_layers,
                    registers: cfg.video_connector_num_learnable_registers,
                },
                eps,
            )?,
            audio: Connector::load(
                map,
                keys,
                "audio_connector",
                ConnectorShape {
                    heads: cfg.audio_connector_num_attention_heads,
                    head_dim: cfg.audio_connector_attention_head_dim,
                    layers: cfg.audio_connector_num_layers,
                    registers: cfg.audio_connector_num_learnable_registers,
                },
                eps,
            )?,
            cfg: cfg.clone(),
        })
    }

    /// `total` is the padded prompt length (1024): the register fill and the
    /// rotary table both depend on it, not on how many tokens are real.
    pub fn forward(&self, stack: &HiddenStack, total: usize) -> Result<TextContexts> {
        if stack.states.len() != self.cfg.text_proj_in_factor || stack.hidden != self.cfg.caption_channels {
            return Err(msg(format!(
                "connectors want {} states of width {}, got {} of width {}",
                self.cfg.text_proj_in_factor,
                self.cfg.caption_channels,
                stack.states.len(),
                stack.hidden
            )));
        }
        let packed = stack.normalized(self.cfg.norm_scale_factor, self.cfg.norm_eps);
        let packed = CudaTensor::from_vec(packed, vec![stack.tokens, self.cfg.text_proj_in_features()])?;
        let proj = self.text_proj_in.forward(&packed)?;
        drop(packed);
        // Both connectors have the same geometry, hence one table.
        let table = SplitRope::from_fractions(
            &connector_fractions(total, self.cfg.connector_rope_base_seq_len),
            1,
            self.video.dim,
            self.video.heads,
            self.cfg.rope_theta,
        );
        let rope = DeviceRope::upload(&table)?;
        let video = self.video.forward(&proj, &rope, total)?;
        let audio = if (self.audio.heads, self.audio.dim) == (self.video.heads, self.video.dim) {
            self.audio.forward(&proj, &rope, total)?
        } else {
            let table = SplitRope::from_fractions(
                &connector_fractions(total, self.cfg.connector_rope_base_seq_len),
                1,
                self.audio.dim,
                self.audio.heads,
                self.cfg.rope_theta,
            );
            self.audio.forward(&proj, &DeviceRope::upload(&table)?, total)?
        };
        Ok(TextContexts { proj, video, audio })
    }
}

#[cfg(test)]
mod tests {
    use super::super::attention::tests::{assert_close, attention_reference, get, linear, rms, rows, weights};
    use super::super::keys::Layout;
    use super::*;

    #[test]
    fn prompts_are_left_padded_and_positions_count_across_the_padding() {
        let p = PaddedPrompt::from_ids(&[2, 7, 9], 8).unwrap();
        assert_eq!(p.ids, vec![0, 0, 0, 0, 0, 2, 7, 9]);
        assert_eq!(p.real_ids(), &[2, 7, 9]);
        assert_eq!(p.real_positions(), vec![5, 6, 7]);
        assert_eq!(p.attention_mask(), vec![0, 0, 0, 0, 0, 1, 1, 1]);
        // Too long: the head survives (right truncation), nothing is padded.
        let long = PaddedPrompt::from_ids(&[2, 3, 4, 5, 6], 4).unwrap();
        assert_eq!((long.ids.clone(), long.real), (vec![2, 3, 4, 5], 4));
        assert_eq!(long.real_positions(), vec![0, 1, 2, 3]);
        assert!(PaddedPrompt::from_ids(&[], 4).is_err());
    }

    #[test]
    fn each_state_is_normalised_by_its_own_mean_and_range() {
        // 2 tokens × 3 channels × 2 states, interleaved [t, c, l].
        let data: Vec<f32> = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, -60.0];
        let stack = HiddenStack::from_interleaved(&data, 2, 3, 2).unwrap();
        assert_eq!(stack.states[0], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(stack.states[1], vec![10.0, 20.0, 30.0, 40.0, 50.0, -60.0]);
        let got = stack.normalized(8.0, 1e-6);
        assert_eq!(got.len(), 12);
        for (l, (mean, range)) in [(3.5f32, 5.0f32), (15.0, 110.0)].into_iter().enumerate() {
            for t in 0..2 {
                for c in 0..3 {
                    let want = 8.0 * (data[(t * 3 + c) * 2 + l] - mean) / range;
                    let at = t * 6 + c * 2 + l;
                    assert!((got[at] - want).abs() < 1e-5, "state {l} token {t} ch {c}: {} vs {want}", got[at]);
                }
            }
        }
    }

    fn tiny() -> Ltx2ConnectorsConfig {
        Ltx2ConnectorsConfig {
            caption_channels: 8,
            text_proj_in_factor: 3,
            video_connector_num_attention_heads: 2,
            video_connector_attention_head_dim: 4,
            video_connector_num_layers: 2,
            video_connector_num_learnable_registers: 4,
            audio_connector_num_attention_heads: 2,
            audio_connector_attention_head_dim: 4,
            audio_connector_num_layers: 1,
            audio_connector_num_learnable_registers: 4,
            connector_rope_base_seq_len: 16,
            ..Ltx2ConnectorsConfig::ltx2_19b()
        }
    }

    /// The whole connector path against loops: normalise, project, front-align
    /// with registers indexed by absolute position, two pre-norm blocks with
    /// the 1-D split rotary, weightless norm out.
    #[test]
    fn connectors_match_a_loop_reference() {
        let cfg = tiny();
        let map = weights();
        let model = TextConnectors::load(&map, &Keys::connectors(Layout::Diffusers), &cfg).unwrap();
        let (tokens, total, dim) = (3usize, 8usize, 8usize);
        let data: Vec<f32> = (0..tokens * dim * 3).map(|i| (i as f32 * 0.31).sin() * 2.0 + 0.2).collect();
        let stack = HiddenStack::from_interleaved(&data, tokens, dim, 3).unwrap();
        let got = model.forward(&stack, total).unwrap();
        assert_eq!(got.video.shape, vec![1, total, dim]);
        assert_eq!(got.audio.shape, vec![1, total, dim]);

        let packed = stack.normalized(8.0, 1e-6);
        let w = get(&map, "text_proj_in.weight", &[dim, dim * 3]);
        let proj: Vec<Vec<f32>> = packed.chunks_exact(dim * 3).map(|r| linear(r, &w, &vec![0.0; dim])).collect();
        assert_close(&rows(&got.proj, dim), &proj, 1e-5, "text_proj_in");

        let table = SplitRope::from_fractions(&connector_fractions(total, 16), 1, dim, 2, 10_000.0);
        let dims = AttentionDims { query_dim: dim, context_dim: dim, heads: 2, head_dim: 4 };
        for (name, layers, ours) in [("video_connector", 2, &got.video), ("audio_connector", 1, &got.audio)] {
            let reg = get(&map, &format!("{name}.learnable_registers"), &[4, dim]);
            let mut x: Vec<Vec<f32>> = (0..total).map(|p| if p < tokens { proj[p].clone() } else { reg[(p % 4) * dim..(p % 4 + 1) * dim].to_vec() }).collect();
            for i in 0..layers {
                let p = format!("{name}.transformer_blocks.{i}");
                let h: Vec<Vec<f32>> = x.iter().map(|v| rms(v, None, 1e-6)).collect();
                let a = attention_reference(&map, &format!("{p}.attn1"), dims, &h, &h, Some(&table), None);
                x.iter_mut().zip(&a).for_each(|(v, a)| v.iter_mut().zip(a).for_each(|(v, a)| *v += a));
                let (w0, b0) = (get(&map, &format!("{p}.ff.net.0.proj.weight"), &[32, dim]), get(&map, &format!("{p}.ff.net.0.proj.bias"), &[32]));
                let (w2, b2) = (get(&map, &format!("{p}.ff.net.2.weight"), &[dim, 32]), get(&map, &format!("{p}.ff.net.2.bias"), &[dim]));
                let gelu = |v: f32| 0.5 * v * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044_715 * v * v * v)).tanh());
                for v in &mut x {
                    let up: Vec<f32> = linear(&rms(v, None, 1e-6), &w0, &b0).into_iter().map(gelu).collect();
                    v.iter_mut().zip(linear(&up, &w2, &b2)).for_each(|(v, f)| *v += f);
                }
            }
            let want: Vec<Vec<f32>> = x.iter().map(|v| rms(v, None, 1e-6)).collect();
            assert_close(&rows(ours, dim), &want, 5e-5, name);
        }
    }

    #[test]
    fn a_full_length_prompt_uses_no_registers() {
        let cfg = tiny();
        let model = TextConnectors::load(&weights(), &Keys::connectors(Layout::Diffusers), &cfg).unwrap();
        let data: Vec<f32> = (0..4 * 8 * 3).map(|i| (i as f32 * 0.17).cos()).collect();
        let stack = HiddenStack::from_interleaved(&data, 4, 8, 3).unwrap();
        assert_eq!(model.forward(&stack, 4).unwrap().video.shape, vec![1, 4, 8]);
        // 6 slots cannot be tiled by 4 registers.
        assert!(model.forward(&stack, 6).is_err());
        // A stack of the wrong depth is refused before any arithmetic.
        let shallow = HiddenStack::from_interleaved(&data[..4 * 8 * 2], 4, 8, 2).unwrap();
        assert!(model.forward(&shallow, 4).is_err());
    }
}
