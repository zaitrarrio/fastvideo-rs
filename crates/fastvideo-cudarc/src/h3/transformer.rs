//! The MiniMax-H3 DiT (`MiniMaxH3Transformer3DModel`): one stack of 50 blocks
//! over one packed `[text | audio | video]` sequence, full self-attention.
//!
//! Three decisions shape this file (docs/ports/h3.md, sections e and h):
//!
//! **The AdaLN projections are never resident.** Each block owns a
//! `Linear(2688, 96768)` — 26 GB across the stack, more than a third of the
//! checkpoint — whose input is a function of the timestep alone. FastH3 runs a
//! fixed 8-rung ladder, so the whole modulation table is a constant of the
//! checkpoint: at load each block's projection is streamed to the device once,
//! evaluated for the 8 x 2 timesteps, reduced to the three `(timestep,
//! modality)` rows a T2AV forward reads, and dropped. What stays is
//! `[8, 50, 3, 6, 5376]` float32 (155 MB) on the host. `time_embedder` and
//! `norm_out.linear` go the same way.
//!
//! **Modality is a row range.** With no keyframe rows the packed layout is
//! three contiguous segments, so "index the AdaLN table by row" is three
//! `narrow`s, a broadcast multiply-add each, and a `cat` — no gather kernels.
//!
//! **Activation lifetime is managed by hand.** Q/K/V are 7168 wide (wider than
//! the 5376 residual) and the FFN is 28672; at 109k rows nothing of that size
//! may outlive its use. The QKVG GEMM is one fused projection (`[q; k; v]` or
//! `[q; k; v; gate]`), split into BHSD immediately, and dropped; the FFN runs
//! in row chunks.
//!
//! Attention here is dense, which is what the diffusers oracle judges. The
//! trained recipe (VSA-H3 with the `to_gate_compress` branch) plugs in through
//! [`AttnMode`].

use fastvideo_models::h3::config::{H3TransformerConfig, MODALITY_NUM, TAG_AUDIO, TAG_TEXT, TAG_VIDEO};
use fastvideo_models::h3::packing::{H3PackedLayout, RowRange};
use fastvideo_models::h3::schedule::H3JointSchedule;

use crate::wan::nn::{scaled_dot_product_attention, Linear};
use crate::wan::stats::phase;
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

fn pinned(data: Vec<f32>, shape: Vec<usize>) -> Result<CudaTensor> {
    let mut t = CudaTensor::from_vec(data, shape)?;
    t.pin_device()?;
    Ok(t)
}

fn pinned_weight(map: &WeightMap, key: &str, shape: &[usize]) -> Result<CudaTensor> {
    let mut t = cuda_tensor_shaped(map, key, shape)?;
    t.pin_device()?;
    Ok(t)
}

/// Rows per FFN pass when the full `[S, 2*ffn]` f32 buffer would exceed
/// [`FFN_WHOLE_BYTES`]. `[8192, 28672]` is just under 1 GiB.
const FFN_ROW_CHUNK: usize = 8192;
/// 5s H3 is 4.0 GiB for that buffer (M=37966); keep one GEMM. 109k-row
/// layouts still chunk.
const FFN_WHOLE_BYTES: u64 = 6 << 30;

fn ffn_row_chunk(rows: usize, ffn_dim: usize) -> usize {
    let whole = rows as u64 * (2 * ffn_dim as u64) * 4;
    if whole <= FFN_WHOLE_BYTES {
        rows
    } else {
        FFN_ROW_CHUNK
    }
}

/// Parameter order inside one modality's AdaLN slice (`y.chunk(6, dim=-1)`).
const SHIFT_MSA: usize = 0;
const SCALE_MSA: usize = 1;
const GATE_MSA: usize = 2;
const SHIFT_MLP: usize = 3;
const SCALE_MLP: usize = 4;
const GATE_MLP: usize = 5;
const ADALN_PARAMS: usize = 6;
/// "H3ADALN1": bump the digit when the table layout changes.
const CACHE_MAGIC: u64 = u64::from_le_bytes(*b"H3ADALN1");

/// Called with a name and a tensor at the points the oracle hooks
/// (`block_<i>`: that block's `[1, S, hidden]` output).
pub type Observer<'a> = &'a mut dyn FnMut(&str, &CudaTensor) -> Result<()>;

/// How the blocks attend. The refiner is always dense.
#[derive(Clone, Copy)]
pub enum AttnMode<'a> {
    /// Full softmax attention without the compression-gate branch: the
    /// function diffusers implements, and the oracle's reference.
    Dense,
    /// VSA-H3 with `to_gate_compress`: the function the checkpoint was trained as.
    Vsa(&'a super::vsa::H3Vsa),
}

// ---------------------------------------------------------------------------
// Time embedding and the precomputed AdaLN table
// ---------------------------------------------------------------------------

/// `time_embedder(time_proj(t))` for each `t`, on the host in float32:
/// sinusoid with cos first and an unscaled `t in [0, 1]`, then
/// `linear_2(silu(linear_1(.)))`. 16 vectors per checkpoint, so exactness
/// matters more than speed; sums accumulate in float64.
pub fn time_embeddings(cfg: &H3TransformerConfig, map: &WeightMap, timesteps: &[f32]) -> Result<Vec<Vec<f32>>> {
    let (freq, hidden, out) = (cfg.freq_dim, cfg.time_embed_hidden_dim, cfg.time_embed_dim);
    let host = |key: &str, shape: &[usize]| -> Result<Vec<f32>> { Ok(cuda_tensor_shaped(map, key, shape)?.host_cow()?.into_owned()) };
    let (w1, b1) = (host("time_embedder.linear_1.weight", &[hidden, freq])?, host("time_embedder.linear_1.bias", &[hidden])?);
    let (w2, b2) = (host("time_embedder.linear_2.weight", &[out, hidden])?, host("time_embedder.linear_2.bias", &[out])?);
    let half = freq / 2;
    let linear = |x: &[f32], w: &[f32], b: &[f32]| -> Vec<f32> {
        use rayon::prelude::*;
        b.par_iter()
            .enumerate()
            .map(|(r, &bias)| (f64::from(bias) + x.iter().zip(&w[r * x.len()..(r + 1) * x.len()]).map(|(a, c)| f64::from(*a) * f64::from(*c)).sum::<f64>()) as f32)
            .collect()
    };
    Ok(timesteps
        .iter()
        .map(|&t| {
            let mut emb = vec![0f32; freq];
            for k in 0..half {
                let f = (-(10000f32.ln()) * k as f32 / half as f32).exp();
                emb[k] = (t * f).cos();
                emb[half + k] = (t * f).sin();
            }
            let h: Vec<f32> = linear(&emb, &w1, &b1).into_iter().map(|v| v / (1.0 + (-v).exp())).collect();
            linear(&h, &w2, &b2)
        })
        .collect())
}

/// Every AdaLN modulation a T2AV run of one ladder reads, on the host.
///
/// Slots are `[video, text, audio]` (the modality tags). Video and text rows
/// take the video timestep, audio rows the audio one. Scales are stored as
/// `1 + scale`, the form the block multiplies by.
pub struct AdaLnTable {
    steps: usize,
    blocks: usize,
    hidden: usize,
    /// `[steps, blocks, 3, 6, hidden]`.
    block_mods: Vec<f32>,
    /// `[steps, 2 (video, audio timestep), 2 (shift, 1 + scale), hidden]`.
    out_mods: Vec<f32>,
    /// `[steps, 2, time_embed_dim]`: `temb` for (video, audio), kept for the oracle.
    pub temb: Vec<Vec<f32>>,
}

impl AdaLnTable {
    /// Stream each projection through the device once. Device peak is one
    /// `adaln_proj` (520 MB as bf16) plus a `[2 * steps, 96768]` result.
    pub fn precompute(cfg: &H3TransformerConfig, map: &WeightMap, schedule: &H3JointSchedule) -> Result<Self> {
        let steps = schedule.num_steps();
        let (hidden, te) = (cfg.hidden_size, cfg.time_embed_dim);
        // Row 2i is the video timestep of step i, row 2i + 1 the audio one.
        let timesteps: Vec<f32> = (0..steps).flat_map(|i| [schedule.video.timesteps[i], schedule.audio.timesteps[i]]).collect();
        let temb = time_embeddings(cfg, map, &timesteps)?;
        // silu in float32, THEN the cast the projection applies.
        let silu: Vec<f32> = temb.iter().flatten().map(|&v| v / (1.0 + (-v).exp())).collect();
        let s = pinned(silu, vec![2 * steps, te])?;

        let slice = ADALN_PARAMS * hidden;
        let mut block_mods = vec![0f32; steps * cfg.num_layers * MODALITY_NUM * slice];
        for b in 0..cfg.num_layers {
            let proj = Linear::load(map, &format!("transformer_blocks.{b}.adaln_proj.linear"), te, cfg.adaln_out_dim(), true)?;
            let y = proj.forward(&s)?;
            let y = y.host_cow()?;
            drop(proj);
            let width = cfg.adaln_out_dim();
            for i in 0..steps {
                for tag in [TAG_VIDEO, TAG_TEXT, TAG_AUDIO] {
                    let m = usize::from(tag);
                    // Within one timestep's output, modality m owns [m * 6H, (m + 1) * 6H).
                    let row = if tag == TAG_AUDIO { 2 * i + 1 } else { 2 * i };
                    let src = &y[row * width + m * slice..row * width + (m + 1) * slice];
                    let dst = &mut block_mods[((i * cfg.num_layers + b) * MODALITY_NUM + m) * slice..][..slice];
                    dst.copy_from_slice(src);
                    for p in [SCALE_MSA, SCALE_MLP] {
                        dst[p * hidden..(p + 1) * hidden].iter_mut().for_each(|v| *v += 1.0);
                    }
                }
            }
            crate::wan::log::info(format_args!("h3 adaln table: block {}/{}", b + 1, cfg.num_layers));
        }

        let proj = Linear::load(map, "norm_out.linear", te, 2 * hidden, true)?;
        let y = proj.forward(&s)?;
        let y = y.host_cow()?;
        // `shift, scale = chunk(2)`: shift first.
        let mut out_mods = y.to_vec();
        for row in out_mods.chunks_exact_mut(2 * hidden) {
            row[hidden..].iter_mut().for_each(|v| *v += 1.0);
        }
        Ok(Self { steps, blocks: cfg.num_layers, hidden, block_mods, out_mods, temb })
    }

    /// [`Self::precompute`], memoized in `cache`. Building the table reads
    /// 26 GB of projections that are needed for nothing else, so a warm start
    /// skips more than a third of the checkpoint. The file is keyed by the
    /// ladder's timesteps and a fingerprint of the checkpoint (block 0's AdaLN
    /// bias bytes); anything that does not match is rebuilt, never trusted.
    pub fn load_or_precompute(cfg: &H3TransformerConfig, map: &WeightMap, schedule: &H3JointSchedule, cache: Option<&std::path::Path>) -> Result<Self> {
        let (Some(path), Some(lazy)) = (cache, map.lazy()) else {
            return Self::precompute(cfg, map, schedule);
        };
        let fingerprint = {
            let view = lazy.view("transformer_blocks.0.adaln_proj.linear.bias").map_err(|e| msg(e.to_string()))?;
            view.bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3))
        };
        let steps = schedule.num_steps();
        let mut header: Vec<u64> = vec![CACHE_MAGIC, fingerprint, steps as u64, cfg.num_layers as u64, cfg.hidden_size as u64, cfg.time_embed_dim as u64];
        header.extend((0..steps).flat_map(|i| [schedule.video.timesteps[i], schedule.audio.timesteps[i]]).map(|t| u64::from(t.to_bits())));
        let header: Vec<u8> = header.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sizes = [steps * cfg.num_layers * MODALITY_NUM * ADALN_PARAMS * cfg.hidden_size, 2 * steps * 2 * cfg.hidden_size, 2 * steps * cfg.time_embed_dim];

        if let Ok(bytes) = std::fs::read(path) {
            let want = header.len() + 4 * sizes.iter().sum::<usize>();
            if bytes.len() == want && bytes[..header.len()] == header[..] {
                let mut values = bytes[header.len()..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                let block_mods: Vec<f32> = values.by_ref().take(sizes[0]).collect();
                let out_mods: Vec<f32> = values.by_ref().take(sizes[1]).collect();
                let flat: Vec<f32> = values.collect();
                let temb = flat.chunks_exact(cfg.time_embed_dim).map(<[f32]>::to_vec).collect();
                crate::wan::log::info(format_args!("h3 adaln table: read from {}", path.display()));
                return Ok(Self { steps, blocks: cfg.num_layers, hidden: cfg.hidden_size, block_mods, out_mods, temb });
            }
            crate::wan::log::info(format_args!("h3 adaln table: {} is for another checkpoint or ladder; rebuilding", path.display()));
        }
        let table = Self::precompute(cfg, map, schedule)?;
        let mut bytes = header;
        bytes.extend(table.block_mods.iter().chain(&table.out_mods).chain(table.temb.iter().flatten()).flat_map(|v| v.to_le_bytes()));
        // A cache that cannot be written costs the next start some time, not this run its result.
        let written = path.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|()| std::fs::write(path, &bytes));
        if let Err(e) = written {
            crate::wan::log::info(format_args!("h3 adaln table: could not write {}: {e}", path.display()));
        }
        Ok(table)
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    /// `[6, hidden]` for `(step, block, tag)`: shift_msa, 1 + scale_msa,
    /// gate_msa, shift_mlp, 1 + scale_mlp, gate_mlp.
    pub fn block_slot(&self, step: usize, block: usize, tag: u8) -> &[f32] {
        let slice = ADALN_PARAMS * self.hidden;
        &self.block_mods[((step * self.blocks + block) * MODALITY_NUM + usize::from(tag)) * slice..][..slice]
    }

    /// `(shift, 1 + scale)` of `norm_out` for the video (`audio = false`) or
    /// audio timestep of `step`. Per timestep, not per modality.
    pub fn out_slot(&self, step: usize, audio: bool) -> (&[f32], &[f32]) {
        let row = &self.out_mods[(2 * step + usize::from(audio)) * 2 * self.hidden..][..2 * self.hidden];
        row.split_at(self.hidden)
    }
}

/// One block's modulation on the device: a `[1, 6, hidden]` table per segment.
struct BlockMods {
    /// In packed order: text, audio, video.
    segments: [(RowRange, CudaTensor); 3],
}

impl BlockMods {
    fn upload(table: &AdaLnTable, step: usize, block: usize, layout: &H3PackedLayout) -> Result<Self> {
        let up = |tag: u8| CudaTensor::from_vec(table.block_slot(step, block, tag).to_vec(), vec![1, ADALN_PARAMS, table.hidden])?.to_device();
        Ok(Self { segments: [(layout.text, up(TAG_TEXT)?), (layout.audio, up(TAG_AUDIO)?), (layout.video, up(TAG_VIDEO)?)] })
    }

    /// `n * (1 + scale[a]) + shift[a]` with `a` the row's modality.
    fn modulate(&self, n: &CudaTensor, scale: usize, shift: usize) -> Result<CudaTensor> {
        let parts = self
            .segments
            .iter()
            .filter(|(range, _)| range.len > 0)
            .map(|(range, e)| n.narrow(1, range.start, range.len)?.mul(&e.narrow(1, scale, 1)?)?.add(&e.narrow(1, shift, 1)?))
            .collect::<Result<Vec<_>>>()?;
        CudaTensor::cat(&parts.iter().collect::<Vec<_>>(), 1)
    }

    /// `x + gate[a] * update`.
    fn gated_add(&self, x: &CudaTensor, update: &CudaTensor, gate: usize) -> Result<CudaTensor> {
        let parts = self
            .segments
            .iter()
            .filter(|(range, _)| range.len > 0)
            .map(|(range, e)| x.narrow(1, range.start, range.len)?.residual_gate_add_e(&update.narrow(1, range.start, range.len)?, e, gate))
            .collect::<Result<Vec<_>>>()?;
        CudaTensor::cat(&parts.iter().collect::<Vec<_>>(), 1)
    }
}

// ---------------------------------------------------------------------------
// Attention, FFN, blocks
// ---------------------------------------------------------------------------

pub(crate) struct Attention {
    /// `[q; k; v]` or `[q; k; v; gate]` stacked on the output dim. One GEMM
    /// so the 520 MB activation is read once; per-head RMSNorm+RoPE still
    /// run on the Q/K slices (H3 norms over `head_dim`, not Wan's full dim).
    qkvg: Linear,
    to_out: Linear,
    /// VSA's compression-branch gate lives in `qkvg`'s last `inner` columns.
    has_gate: bool,
    /// `[head_dim]`, shared by every head.
    norm_q: CudaTensor,
    norm_k: CudaTensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
}

impl Attention {
    fn load(map: &WeightMap, prefix: &str, cfg: &H3TransformerConfig, gate: bool) -> Result<Self> {
        let (hidden, inner, d) = (cfg.hidden_size, cfg.inner_dim(), cfg.attention_head_dim);
        let mut names = vec![format!("{prefix}.to_q"), format!("{prefix}.to_k"), format!("{prefix}.to_v")];
        if gate {
            names.push(format!("{prefix}.to_gate_compress"));
        }
        let keys: Vec<&str> = names.iter().map(String::as_str).collect();
        Ok(Self {
            qkvg: Linear::load_fused(map, &keys, hidden, inner, false)?,
            to_out: Linear::load(map, &format!("{prefix}.to_out.0"), inner, hidden, false)?,
            has_gate: gate,
            norm_q: pinned_weight(map, &format!("{prefix}.norm_q.weight"), &[d])?,
            norm_k: pinned_weight(map, &format!("{prefix}.norm_k.weight"), &[d])?,
            heads: cfg.num_attention_heads,
            head_dim: d,
            eps: cfg.qk_norm_eps as f32,
        })
    }

    /// `n`: `[1, S, hidden]`. `rope`: `[S, R]` cos/sin, or `None` (refiner).
    fn forward(&self, n: &CudaTensor, rope: Option<(&CudaTensor, &CudaTensor)>, mode: AttnMode<'_>) -> Result<CudaTensor> {
        let packed = phase("h3_attn_qkvg", || self.qkvg.forward(n))?;
        let inner = self.heads * self.head_dim;
        // Per-head RMSNorm over D (one [D] weight for all heads), then RoPE.
        // Wan's fused `qk_norm_rope_bhsd` norms over heads*d — wrong here.
        let q = phase("h3_attn_q", || {
            let t = packed.split_heads_bhsd(0, self.heads, self.head_dim)?.rms_norm(&self.norm_q, self.eps)?;
            match rope {
                Some((cos, sin)) => t.rope_half(cos, sin),
                None => Ok(t),
            }
        })?;
        let k = phase("h3_attn_k", || {
            let t = packed.split_heads_bhsd(inner, self.heads, self.head_dim)?.rms_norm(&self.norm_k, self.eps)?;
            match rope {
                Some((cos, sin)) => t.rope_half(cos, sin),
                None => Ok(t),
            }
        })?;
        let v = phase("h3_attn_v", || packed.split_heads_bhsd(2 * inner, self.heads, self.head_dim))?;
        let out = match mode {
            AttnMode::Dense => scaled_dot_product_attention(&q, &k, &v, None)?,
            AttnMode::Vsa(vsa) => {
                // Gate is a plain projection of the same input: not normed, not rotated.
                let gate = if self.has_gate {
                    Some(phase("h3_attn_gate", || packed.split_heads_bhsd(3 * inner, self.heads, self.head_dim))?)
                } else {
                    None
                };
                vsa.attend(q, k, v, gate)?
            }
        };
        drop(packed);
        phase("h3_attn_out", || self.to_out.forward(&out.merge_heads()?))
    }
}

/// Bias-free SwiGLU with the **value half first**: `ff_out(v * silu(g))` for
/// `(v, g) = chunk(ff_in(x), 2)`. Row-chunked: the `[S, 2 * ffn]` intermediate
/// is never whole.
pub(crate) struct FeedForward {
    ff_in: Linear,
    ff_out: Linear,
    ffn_dim: usize,
}

impl FeedForward {
    fn load(map: &WeightMap, prefix: &str, cfg: &H3TransformerConfig) -> Result<Self> {
        Ok(Self {
            ff_in: Linear::load(map, &format!("{prefix}.net.0.proj"), cfg.hidden_size, 2 * cfg.ffn_dim, false)?,
            ff_out: Linear::load(map, &format!("{prefix}.net.2"), cfg.ffn_dim, cfg.hidden_size, false)?,
            ffn_dim: cfg.ffn_dim,
        })
    }

    fn forward(&self, n: &CudaTensor) -> Result<CudaTensor> {
        let rows = n.shape[1];
        let chunk = ffn_row_chunk(rows, self.ffn_dim);
        let mut parts = Vec::with_capacity(rows.div_ceil(chunk).max(1));
        let mut start = 0;
        while start < rows {
            let len = chunk.min(rows - start);
            let h = phase("h3_ffn_in", || self.ff_in.forward(&n.narrow(1, start, len)?))?;
            let act = phase("h3_ffn_act", || {
                let value = h.narrow(2, 0, self.ffn_dim)?;
                let gate = h.narrow(2, self.ffn_dim, self.ffn_dim)?.silu();
                value.mul(&gate)
            })?;
            drop(h);
            parts.push(phase("h3_ffn_out", || self.ff_out.forward(&act))?);
            start += len;
        }
        if parts.len() == 1 {
            return Ok(parts.pop().unwrap());
        }
        CudaTensor::cat(&parts.iter().collect::<Vec<_>>(), 1)
    }
}

pub(crate) struct Block {
    norm1: CudaTensor,
    norm2: CudaTensor,
    attn: Attention,
    ff: FeedForward,
}

impl Block {
    pub(crate) fn load(map: &WeightMap, prefix: &str, cfg: &H3TransformerConfig, gate: bool) -> Result<Self> {
        Ok(Self {
            norm1: pinned_weight(map, &format!("{prefix}.norm1.weight"), &[cfg.hidden_size])?,
            norm2: pinned_weight(map, &format!("{prefix}.norm2.weight"), &[cfg.hidden_size])?,
            attn: Attention::load(map, &format!("{prefix}.attn"), cfg, gate)?,
            ff: FeedForward::load(map, &format!("{prefix}.ff"), cfg)?,
        })
    }
}

/// `context_embedder` + the two timestep-free refiner blocks + `final_norm`.
/// Runs once per prompt; drop it afterwards (1.6 GB).
pub struct H3TextRefiner {
    context_embedder: Linear,
    blocks: Vec<Block>,
    final_norm: CudaTensor,
    eps: f32,
    final_eps: f32,
}

impl H3TextRefiner {
    pub fn load(cfg: &H3TransformerConfig, map: &WeightMap) -> Result<Self> {
        Ok(Self {
            context_embedder: Linear::load(map, "context_embedder", cfg.text_dim, cfg.hidden_size, true)?,
            blocks: (0..cfg.num_refiner_layers)
                .map(|i| Block::load(map, &format!("token_refiner.refiner_blocks.{i}"), cfg, false))
                .collect::<Result<Vec<_>>>()?,
            final_norm: pinned_weight(map, "token_refiner.final_norm.weight", &[cfg.hidden_size])?,
            eps: cfg.norm_eps as f32,
            final_eps: cfg.final_norm_eps as f32,
        })
    }

    /// `[1, N, text_dim]` hidden states to `[1, N, hidden]`: plain pre-norm
    /// blocks, bidirectional attention, no RoPE, no AdaLN.
    pub fn forward(&self, text: &CudaTensor) -> Result<CudaTensor> {
        let mut e = self.context_embedder.forward(text)?;
        for block in &self.blocks {
            let a = block.attn.forward(&e.rms_norm(&block.norm1, self.eps)?, None, AttnMode::Dense)?;
            e = e.add(&a)?;
            let f = block.ff.forward(&e.rms_norm(&block.norm2, self.eps)?)?;
            e = e.add(&f)?;
        }
        e.rms_norm(&self.final_norm, self.final_eps)
    }
}

/// The packed layout with its rotary tables on the device: built once per
/// request, shared by every block of every step.
pub struct DeviceLayout {
    pub layout: H3PackedLayout,
    cos: CudaTensor,
    sin: CudaTensor,
}

impl DeviceLayout {
    pub fn new(cfg: &H3TransformerConfig, layout: H3PackedLayout) -> Result<Self> {
        let (cos, sin) = layout.rope_tables(&cfg.rope_inv_freq());
        let shape = vec![layout.sequence_length(), cfg.rotary_dim()];
        Ok(Self { cos: pinned(cos, shape.clone())?, sin: pinned(sin, shape)?, layout })
    }
}

pub struct H3Transformer {
    cfg: H3TransformerConfig,
    proj_in: Linear,
    audio_proj_in: Linear,
    blocks: Vec<Block>,
    norm_out: CudaTensor,
    proj_out: Linear,
    audio_proj_out: Linear,
    table: AdaLnTable,
    has_gate: bool,
}

impl H3Transformer {
    /// Loads the resident part of the stack (41 GiB as bf16 with the VSA gates,
    /// 37 GiB without) and precomputes the AdaLN table for `schedule`.
    /// `with_gate` loads `to_gate_compress`, which only [`AttnMode::Vsa`] reads.
    pub fn load(cfg: H3TransformerConfig, map: &WeightMap, schedule: &H3JointSchedule, with_gate: bool) -> Result<Self> {
        Self::load_cached(cfg, map, schedule, with_gate, None)
    }

    /// [`Self::load`] with the AdaLN table memoized at `adaln_cache`
    /// (see [`AdaLnTable::load_or_precompute`]).
    pub fn load_cached(cfg: H3TransformerConfig, map: &WeightMap, schedule: &H3JointSchedule, with_gate: bool, adaln_cache: Option<&std::path::Path>) -> Result<Self> {
        if cfg.rotary_dim() > cfg.attention_head_dim || cfg.freq_dim % 2 != 0 {
            return Err(msg(format!("h3 dit: {} rotary channels of a {}-wide head", cfg.rotary_dim(), cfg.attention_head_dim)));
        }
        let table = AdaLnTable::load_or_precompute(&cfg, map, schedule, adaln_cache)?;
        let blocks = (0..cfg.num_layers)
            .map(|i| {
                let block = Block::load(map, &format!("transformer_blocks.{i}"), &cfg, with_gate);
                crate::wan::log::info(format_args!("h3 dit: block {}/{} resident", i + 1, cfg.num_layers));
                block
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            proj_in: Linear::load(map, "proj_in", cfg.video_patch_dim(), cfg.hidden_size, true)?,
            audio_proj_in: Linear::load(map, "audio_proj_in", cfg.audio_in_channels, cfg.hidden_size, true)?,
            blocks,
            norm_out: pinned_weight(map, "norm_out.norm.weight", &[cfg.hidden_size])?,
            proj_out: Linear::load(map, "proj_out", cfg.hidden_size, cfg.video_patch_dim(), true)?,
            audio_proj_out: Linear::load(map, "audio_proj_out", cfg.hidden_size, cfg.audio_in_channels, true)?,
            table,
            has_gate: with_gate,
            cfg,
        })
    }

    pub fn config(&self) -> &H3TransformerConfig {
        &self.cfg
    }

    pub fn adaln_table(&self) -> &AdaLnTable {
        &self.table
    }

    pub fn has_gate(&self) -> bool {
        self.has_gate
    }

    /// One forward at ladder step `step`. `video_rows`: `[Nv, 96]` patchified
    /// latents; `audio_rows`: `[2 Na, 32]`; `text`: the **refined** prompt
    /// `[1, N, hidden]`. Returns the data-ward velocities, same shapes as the
    /// two row inputs.
    pub fn forward(
        &self,
        step: usize,
        video_rows: &CudaTensor,
        audio_rows: &CudaTensor,
        text: &CudaTensor,
        layout: &DeviceLayout,
        mode: AttnMode<'_>,
        mut observer: Option<Observer<'_>>,
    ) -> Result<(CudaTensor, CudaTensor)> {
        let cfg = &self.cfg;
        let l = &layout.layout;
        let hidden = cfg.hidden_size;
        if step >= self.table.steps() {
            return Err(msg(format!("h3 dit: step {step} of a {}-step AdaLN table", self.table.steps())));
        }
        if video_rows.shape != [l.video.len, cfg.video_patch_dim()] || audio_rows.shape != [l.audio.len, cfg.audio_in_channels] || text.shape != [1, l.text.len, hidden] {
            return Err(msg(format!(
                "h3 dit: rows video {:?} audio {:?} text {:?} for a layout of {} + {} + {}",
                video_rows.shape, audio_rows.shape, text.shape, l.text.len, l.audio.len, l.video.len
            )));
        }
        if matches!(mode, AttnMode::Vsa(_)) && !self.has_gate {
            return Err(msg("h3 dit: VSA needs to_gate_compress; load the transformer with the gate"));
        }
        let seq = l.sequence_length();
        let x = {
            let video = self.proj_in.forward(video_rows)?;
            let audio = self.audio_proj_in.forward(audio_rows)?;
            CudaTensor::cat(&[&text.reshape(vec![l.text.len, hidden])?, &audio, &video], 0)?.reshape_owned(vec![1, seq, hidden])?
        };
        let rope = Some((&layout.cos, &layout.sin));
        let eps = cfg.norm_eps as f32;

        let mut x = x;
        for (index, block) in self.blocks.iter().enumerate() {
            let mods = BlockMods::upload(&self.table, step, index, l)?;
            let n = phase("h3_1_norm_msa", || mods.modulate(&x.rms_norm(&block.norm1, eps)?, SCALE_MSA, SHIFT_MSA))?;
            let a = phase("h3_2_attn", || block.attn.forward(&n, rope, mode))?;
            x = phase("h3_3_residual_msa", || mods.gated_add(&x, &a, GATE_MSA))?;
            drop(a);
            let n = phase("h3_4_norm_ffn", || mods.modulate(&x.rms_norm(&block.norm2, eps)?, SCALE_MLP, SHIFT_MLP))?;
            let f = phase("h3_5_ffn", || block.ff.forward(&n))?;
            x = phase("h3_6_residual_ffn", || mods.gated_add(&x, &f, GATE_MLP))?;
            drop(f);
            if let Some(observe) = observer.as_mut() {
                observe(&format!("block_{index}"), &x)?;
            }
            crate::wan::log::info(format_args!("h3 dit step {step}: block {}/{}", index + 1, self.blocks.len()));
        }

        // Both heads are defined on every row; only each modality's own rows are read.
        let head = |range: RowRange, audio: bool, proj: &Linear| -> Result<CudaTensor> {
            let (shift, scale) = self.table.out_slot(step, audio);
            let scale = CudaTensor::from_vec(scale.to_vec(), vec![hidden])?;
            let shift = CudaTensor::from_vec(shift.to_vec(), vec![hidden])?;
            let n = x.narrow(1, range.start, range.len)?.rms_norm(&self.norm_out, cfg.final_norm_eps as f32)?;
            proj.forward(&n.mul(&scale)?.add(&shift)?)?.reshape(vec![range.len, proj.out_dim()])
        };
        Ok((head(l.video, false, &self.proj_out)?, head(l.audio, true, &self.audio_proj_out)?))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use fastvideo_models::h3::schedule::H3RowTimesteps;

    pub(crate) fn tiny_cfg() -> H3TransformerConfig {
        H3TransformerConfig {
            num_attention_heads: 3,
            attention_head_dim: 8,
            hidden_size: 12,
            num_layers: 2,
            num_refiner_layers: 1,
            ffn_dim: 10,
            in_channels: 2,
            audio_in_channels: 3,
            patch_size: [1, 2, 2],
            text_dim: 7,
            freq_dim: 8,
            time_embed_hidden_dim: 9,
            time_embed_dim: 5,
            rope_freq_dim: 1,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }

    #[test]
    fn five_second_h3_ffn_is_one_gemm() {
        assert_eq!(ffn_row_chunk(37_966, 14_336), 37_966);
        assert_eq!(ffn_row_chunk(109_000, 14_336), FFN_ROW_CHUNK);
    }

    pub(crate) fn weights() -> WeightMap {
        WeightMap::generated(|key, shape| {
            let seed = key.bytes().fold(13u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
            let n: usize = shape.iter().product();
            (0..n)
                .map(|i| {
                    let u = (seed.wrapping_add(i as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / (1u32 << 24) as f32;
                    if key.contains("norm") { 0.5 + u } else { (u - 0.5) * 0.8 }
                })
                .collect()
        })
    }

    fn get(map: &WeightMap, key: &str, shape: &[usize]) -> Vec<f32> {
        cuda_tensor_shaped(map, key, shape).unwrap().host_cow().unwrap().into_owned()
    }

    fn lin(map: &WeightMap, prefix: &str, x: &[f32], o: usize, bias: bool) -> Vec<f32> {
        let i = x.len();
        let w = get(map, &format!("{prefix}.weight"), &[o, i]);
        let b = if bias { get(map, &format!("{prefix}.bias"), &[o]) } else { vec![0.0; o] };
        (0..o).map(|r| b[r] + (0..i).map(|c| x[c] * w[r * i + c]).sum::<f32>()).collect()
    }

    fn rms(v: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let ms = v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32;
        v.iter().zip(w).map(|(a, g)| a / (ms + eps).sqrt() * g).collect()
    }

    fn silu(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }

    /// One attention + FFN sublayer pair over rows, dense, with optional
    /// per-row `(shift, scale, gate)` modulation; plain loops.
    #[allow(clippy::too_many_arguments)]
    fn ref_block(
        cfg: &H3TransformerConfig,
        map: &WeightMap,
        prefix: &str,
        x: &[Vec<f32>],
        angles: Option<&[[f32; 3]]>,
        mods: Option<&[Vec<f32>]>,
    ) -> Vec<Vec<f32>> {
        let (h, heads, d, eps) = (cfg.hidden_size, cfg.num_attention_heads, cfg.attention_head_dim, 1e-5f32);
        let s = x.len();
        let (n1, n2) = (get(map, &format!("{prefix}.norm1.weight"), &[h]), get(map, &format!("{prefix}.norm2.weight"), &[h]));
        let (nq, nk) = (get(map, &format!("{prefix}.attn.norm_q.weight"), &[d]), get(map, &format!("{prefix}.attn.norm_k.weight"), &[d]));
        // mods[row] = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] x hidden, RAW scales.
        let modulate = |v: Vec<f32>, row: usize, shift: usize, scale: usize| -> Vec<f32> {
            match mods {
                Some(m) => (0..h).map(|c| v[c] * (1.0 + m[row][scale * h + c]) + m[row][shift * h + c]).collect(),
                None => v,
            }
        };
        let gate = |row: usize, p: usize, c: usize| mods.map_or(1.0, |m| m[row][p * h + c]);
        let rope = |v: &[f32], row: usize| -> Vec<f32> {
            let Some(a) = angles else { return v.to_vec() };
            let mut out = v.to_vec();
            for axis in 0..3 {
                let (c, sn) = (a[row][axis].cos(), a[row][axis].sin());
                out[axis] = v[axis] * c - v[axis + 3] * sn;
                out[axis + 3] = v[axis + 3] * c + v[axis] * sn;
            }
            out
        };
        let normed: Vec<Vec<f32>> = x.iter().enumerate().map(|(r, v)| modulate(rms(v, &n1, eps), r, 0, 1)).collect();
        let proj = |name: &str, norm: Option<&[f32]>| -> Vec<Vec<Vec<f32>>> {
            normed
                .iter()
                .enumerate()
                .map(|(r, v)| {
                    let full = lin(map, &format!("{prefix}.attn.{name}"), v, heads * d, false);
                    (0..heads)
                        .map(|hd| {
                            let t = &full[hd * d..(hd + 1) * d];
                            norm.map_or(t.to_vec(), |w| rope(&rms(t, w, eps), r))
                        })
                        .collect()
                })
                .collect()
        };
        let (q, k, v) = (proj("to_q", Some(&nq)), proj("to_k", Some(&nk)), proj("to_v", None));
        (0..s)
            .map(|i| {
                let mut attn = vec![0f32; heads * d];
                for hd in 0..heads {
                    let scores: Vec<f32> = (0..s).map(|j| q[i][hd].iter().zip(&k[j][hd]).map(|(a, b)| a * b).sum::<f32>() / (d as f32).sqrt()).collect();
                    let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                    let z: f32 = scores.iter().map(|sc| (sc - mx).exp()).sum();
                    for (j, sc) in scores.iter().enumerate() {
                        for c in 0..d {
                            attn[hd * d + c] += (sc - mx).exp() / z * v[j][hd][c];
                        }
                    }
                }
                let o = lin(map, &format!("{prefix}.attn.to_out.0"), &attn, h, false);
                let after: Vec<f32> = (0..h).map(|c| x[i][c] + gate(i, 2, c) * o[c]).collect();
                let m = modulate(rms(&after, &n2, eps), i, 3, 4);
                let f = lin(map, &format!("{prefix}.ff.net.0.proj"), &m, 2 * cfg.ffn_dim, false);
                let act: Vec<f32> = (0..cfg.ffn_dim).map(|c| f[c] * silu(f[cfg.ffn_dim + c])).collect();
                let o = lin(map, &format!("{prefix}.ff.net.2"), &act, h, false);
                (0..h).map(|c| after[c] + gate(i, 5, c) * o[c]).collect()
            })
            .collect()
    }

    fn ref_temb(cfg: &H3TransformerConfig, map: &WeightMap, t: f32) -> Vec<f32> {
        let half = cfg.freq_dim / 2;
        let mut emb = vec![0f32; cfg.freq_dim];
        for k in 0..half {
            let f = (-(10000f32.ln()) * k as f32 / half as f32).exp();
            emb[k] = (t * f).cos();
            emb[half + k] = (t * f).sin();
        }
        let h: Vec<f32> = lin(map, "time_embedder.linear_1", &emb, cfg.time_embed_hidden_dim, true).into_iter().map(silu).collect();
        lin(map, "time_embedder.linear_2", &h, cfg.time_embed_dim, true)
    }

    fn seeded(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * k + 0.3).sin()).collect()
    }

    #[test]
    fn the_refiner_is_a_plain_pre_norm_stack_without_rope() {
        let (cfg, map) = (tiny_cfg(), weights());
        let text = seeded(4 * cfg.text_dim, 0.7);
        let got = H3TextRefiner::load(&cfg, &map).unwrap().forward(&CudaTensor::from_vec(text.clone(), vec![1, 4, cfg.text_dim]).unwrap()).unwrap();
        assert_eq!(got.shape, vec![1, 4, cfg.hidden_size]);
        let x: Vec<Vec<f32>> = text.chunks(cfg.text_dim).map(|r| lin(&map, "context_embedder", r, cfg.hidden_size, true)).collect();
        let x = ref_block(&cfg, &map, "token_refiner.refiner_blocks.0", &x, None, None);
        let fin = get(&map, "token_refiner.final_norm.weight", &[cfg.hidden_size]);
        let got = got.host_cow().unwrap();
        for (r, row) in x.iter().enumerate() {
            for (c, w) in rms(row, &fin, 1e-5).iter().enumerate() {
                assert!((got[r * cfg.hidden_size + c] - w).abs() < 2e-5, "row {r} ch {c}");
            }
        }
    }

    /// The whole forward against loops that evaluate every `adaln_proj` from
    /// its raw weights per row (`timestep_index * 3 + tag`), so the precomputed
    /// table's slicing is judged by code that does not share it.
    #[test]
    fn a_forward_matches_a_per_row_loop_reference() {
        let (cfg, map) = (tiny_cfg(), weights());
        let schedule = H3JointSchedule::fasth3_8step();
        let model = H3Transformer::load(cfg.clone(), &map, &schedule, false).unwrap();
        let layout = H3PackedLayout::new(2, (2, 2, 4), 1, cfg.patch_size).unwrap();
        let (nv, na, nt) = (layout.video.len, layout.audio.len, layout.text.len);
        assert_eq!((nv, na, nt), (4, 2, 2));
        let video = seeded(nv * cfg.video_patch_dim(), 0.41);
        let audio = seeded(na * cfg.audio_in_channels, 0.23);
        let text = seeded(nt * cfg.hidden_size, 0.57);
        let step = 3;
        let dl = DeviceLayout::new(&cfg, layout.clone()).unwrap();
        let mut seen = Vec::new();
        let (gv, ga) = model
            .forward(
                step,
                &CudaTensor::from_vec(video.clone(), vec![nv, cfg.video_patch_dim()]).unwrap(),
                &CudaTensor::from_vec(audio.clone(), vec![na, cfg.audio_in_channels]).unwrap(),
                &CudaTensor::from_vec(text.clone(), vec![1, nt, cfg.hidden_size]).unwrap(),
                &dl,
                AttnMode::Dense,
                Some(&mut |name, t| {
                    seen.push((name.to_string(), t.shape.clone()));
                    Ok(())
                }),
            )
            .unwrap();
        assert_eq!(seen, vec![("block_0".to_string(), vec![1, 8, 12]), ("block_1".to_string(), vec![1, 8, 12])]);

        // --- reference ---
        let h = cfg.hidden_size;
        let ts = schedule.row_timesteps(step).unwrap();
        let ts_index = layout.timestep_indices(&ts);
        let temb: Vec<Vec<f32>> = ts.timesteps.iter().map(|&t| ref_temb(&cfg, &map, t)).collect();
        let mut x: Vec<Vec<f32>> = text.chunks(h).map(<[f32]>::to_vec).collect();
        x.extend(audio.chunks(cfg.audio_in_channels).map(|r| lin(&map, "audio_proj_in", r, h, true)));
        x.extend(video.chunks(cfg.video_patch_dim()).map(|r| lin(&map, "proj_in", r, h, true)));
        let angles: Vec<[f32; 3]> = layout.position_ids.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
        for b in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{b}");
            // y.view(n_t * 3, 6H): row = timestep_index * 3 + tag.
            let tables: Vec<Vec<f32>> = temb
                .iter()
                .map(|e| lin(&map, &format!("{p}.adaln_proj.linear"), &e.iter().map(|&v| silu(v)).collect::<Vec<_>>(), 18 * h, true))
                .collect();
            let mods: Vec<Vec<f32>> = (0..x.len())
                .map(|r| {
                    let a = ts_index[r] * 3 + usize::from(layout.token_tags[r]);
                    tables[a / 3][(a % 3) * 6 * h..(a % 3 + 1) * 6 * h].to_vec()
                })
                .collect();
            x = ref_block(&cfg, &map, &p, &x, Some(&angles), Some(&mods));
        }
        let nw = get(&map, "norm_out.norm.weight", &[h]);
        let out_rows = |range: RowRange, proj: &str, width: usize| -> Vec<f32> {
            (range.start..range.end())
                .flat_map(|r| {
                    let ss = lin(&map, "norm_out.linear", &temb[ts_index[r]].iter().map(|&v| silu(v)).collect::<Vec<_>>(), 2 * h, true);
                    let n = rms(&x[r], &nw, 1e-5);
                    let y: Vec<f32> = (0..h).map(|c| n[c] * (1.0 + ss[h + c]) + ss[c]).collect();
                    lin(&map, proj, &y, width, true)
                })
                .collect()
        };
        let (wv, wa) = (out_rows(layout.video, "proj_out", cfg.video_patch_dim()), out_rows(layout.audio, "audio_proj_out", cfg.audio_in_channels));
        assert_eq!((gv.shape.clone(), ga.shape.clone()), (vec![nv, cfg.video_patch_dim()], vec![na, cfg.audio_in_channels]));
        for (name, got, want) in [("video", gv.host_cow().unwrap(), wv), ("audio", ga.host_cow().unwrap(), wa)] {
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            assert!(scale > 1e-3, "{name}: a zero reference proves nothing");
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() < 1e-4 * scale.max(1.0), "{name}[{i}]: {g} vs {w}");
            }
        }
    }

    /// The base model's zero-initialised gate makes the compression branch an
    /// exact no-op, so VSA with every tile kept must reproduce the dense
    /// forward through the whole stack; with a live gate it must not.
    #[test]
    fn vsa_with_every_tile_and_a_zero_gate_is_the_dense_forward() {
        use crate::h3::vsa::{H3Vsa, H3VsaConfig};
        let cfg = tiny_cfg();
        let schedule = H3JointSchedule::fasth3_8step();
        let zero_gate = || {
            let base = weights();
            WeightMap::generated(move |key, shape| {
                if key.contains("to_gate_compress") {
                    return vec![0.0; shape.iter().product()];
                }
                cuda_tensor_shaped(&base, key, shape).unwrap().host_cow().unwrap().into_owned()
            })
        };
        let layout = H3PackedLayout::new(3, (2, 4, 4), 2, cfg.patch_size).unwrap();
        let (nv, na, nt) = (layout.video.len, layout.audio.len, layout.text.len);
        let video = CudaTensor::from_vec(seeded(nv * cfg.video_patch_dim(), 0.41), vec![nv, cfg.video_patch_dim()]).unwrap();
        let audio = CudaTensor::from_vec(seeded(na * cfg.audio_in_channels, 0.23), vec![na, cfg.audio_in_channels]).unwrap();
        let text = CudaTensor::from_vec(seeded(nt * cfg.hidden_size, 0.57), vec![1, nt, cfg.hidden_size]).unwrap();
        let vsa = H3Vsa::new(&layout, cfg.num_attention_heads, cfg.attention_head_dim, H3VsaConfig { sparsity: 0.0, group: 1 }).unwrap();
        let dl = DeviceLayout::new(&cfg, layout).unwrap();
        let run = |map: &WeightMap, mode: AttnMode<'_>| -> Vec<f32> {
            let model = H3Transformer::load(cfg.clone(), map, &schedule, true).unwrap();
            let (v, a) = model.forward(0, &video, &audio, &text, &dl, mode, None).unwrap();
            v.host_cow().unwrap().iter().chain(a.host_cow().unwrap().iter()).copied().collect()
        };
        let dense = run(&zero_gate(), AttnMode::Dense);
        let sparse = run(&zero_gate(), AttnMode::Vsa(&vsa));
        assert!(dense.iter().zip(&sparse).all(|(a, b)| (a - b).abs() < 1e-5), "zero gate: VSA at sparsity 0 is dense");
        let gated = run(&weights(), AttnMode::Vsa(&vsa));
        let dense_live = run(&weights(), AttnMode::Dense);
        assert!(gated.iter().zip(&dense_live).any(|(a, b)| (a - b).abs() > 1e-4), "a trained gate changes the output");
        // Without the gate weights loaded, VSA is refused rather than silently run gateless.
        let no_gate = H3Transformer::load(cfg.clone(), &weights(), &schedule, false).unwrap();
        assert!(no_gate.forward(0, &video, &audio, &text, &dl, AttnMode::Vsa(&vsa), None).is_err());
    }

    #[test]
    fn the_table_holds_the_three_rows_a_t2av_forward_reads() {
        let (cfg, map) = (tiny_cfg(), weights());
        let schedule = H3JointSchedule::fasth3_8step();
        let table = AdaLnTable::precompute(&cfg, &map, &schedule).unwrap();
        let (h, step, block) = (cfg.hidden_size, 5, 1);
        let ts = H3RowTimesteps::new(schedule.video.timesteps[step], schedule.audio.timesteps[step]);
        assert_eq!(ts.adaln_rows(), [0, 1, 5]);
        for (tag, t) in [(TAG_VIDEO, schedule.video.timesteps[step]), (TAG_TEXT, schedule.video.timesteps[step]), (TAG_AUDIO, schedule.audio.timesteps[step])] {
            let s: Vec<f32> = ref_temb(&cfg, &map, t).into_iter().map(silu).collect();
            let y = lin(&map, &format!("transformer_blocks.{block}.adaln_proj.linear"), &s, 18 * h, true);
            let want = &y[usize::from(tag) * 6 * h..(usize::from(tag) + 1) * 6 * h];
            let got = table.block_slot(step, block, tag);
            for p in 0..6 {
                let plus = if p == SCALE_MSA || p == SCALE_MLP { 1.0 } else { 0.0 };
                for c in 0..h {
                    assert!((got[p * h + c] - (want[p * h + c] + plus)).abs() < 1e-5, "tag {tag} param {p} ch {c}");
                }
            }
        }
        let want = ref_temb(&cfg, &map, schedule.audio.timesteps[step]);
        assert!(table.temb[2 * step + 1].iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-6));
    }
}
