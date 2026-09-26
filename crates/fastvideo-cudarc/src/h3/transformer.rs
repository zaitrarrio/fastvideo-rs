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
//! Attention defaults to dense, which is what the diffusers oracle judges. The
//! trained recipe (VSA-H3 with the `to_gate_compress` branch) and the opt-in
//! Sol-Attn policies (engine / RTX / Spark, [`fastvideo_models::h3::sol`])
//! plug in through [`AttnMode`].

use std::sync::{Arc, Mutex};

use fastvideo_models::h3::config::{
    H3TransformerConfig, MODALITY_NUM, TAG_AUDIO, TAG_TEXT, TAG_VIDEO,
};
use fastvideo_models::h3::packing::{H3PackedLayout, RowRange, KEYFRAME_NOISE_AUG};
use fastvideo_models::h3::schedule::H3JointSchedule;
use fastvideo_models::h3::sol::H3TeaCache;

use super::fused16::{self, AdaRows, NormOut};
use crate::wan::nn::{scaled_dot_product_attention, Linear};
use crate::wan::offload::{BlockWeights, OffloadBlock, Residency};
use crate::wan::quant::{H3QuantPlan, QuantKind, QuantMode, Section};
use crate::wan::tensor::bf16_activations;
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

fn pinned_weight(
    map: &WeightMap,
    key: &str,
    shape: &[usize],
    lora: &mut Option<super::lora::H3LoraFuse>,
) -> Result<CudaTensor> {
    let Some(fuse) = lora.as_mut() else {
        let mut t = cuda_tensor_shaped(map, key, shape)?;
        t.pin_device()?;
        return Ok(t);
    };
    let (got, mut data) = map.get_f32(key)?;
    if got.as_slice() != shape {
        return Err(msg(format!(
            "key {key}: shape {got:?} != expected {shape:?}"
        )));
    }
    fuse.fuse(key, &mut data, shape)?;
    pinned(data, shape.to_vec())
}

/// `Linear::load`, or the same matrix after a Sol-H3 adapter update.
fn load_linear(
    map: &WeightMap,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
    has_bias: bool,
    lora: &mut Option<super::lora::H3LoraFuse>,
) -> Result<Linear> {
    let Some(fuse) = lora.as_mut() else {
        return Linear::load(map, prefix, in_dim, out_dim, has_bias);
    };
    let wkey = format!("{prefix}.weight");
    let (shape, mut weight) = map.get_f32(&wkey)?;
    if shape != [out_dim, in_dim] {
        return Err(msg(format!(
            "key {wkey}: shape {shape:?} != expected {:?}",
            [out_dim, in_dim]
        )));
    }
    fuse.fuse(&wkey, &mut weight, &shape)?;
    let bias = if has_bias {
        let bkey = format!("{prefix}.bias");
        let (bs, mut bias) = map.get_f32(&bkey)?;
        if bs != [out_dim] {
            return Err(msg(format!("key {bkey}: shape {bs:?} != [{out_dim}]")));
        }
        fuse.fuse(&bkey, &mut bias, &bs)?;
        Some(CudaTensor::from_vec(bias, vec![out_dim])?)
    } else {
        None
    };
    Linear::from_tensors(CudaTensor::from_vec(weight, vec![out_dim, in_dim])?, bias)
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
/// "H3ADALN3": v3 stores KEYFRAME_NOISE_AUG AdaLN for all modality tags.
const CACHE_MAGIC: u64 = u64::from_le_bytes(*b"H3ADALN3");

/// Called with a name and a tensor at the points the oracle hooks
/// (`block_<i>`: that block's `[1, S, hidden]` output).
pub type Observer<'a> = &'a mut dyn FnMut(&str, &CudaTensor) -> Result<()>;

/// TeaCache `sum|current - previous| / sum|previous|` (`teacache.py`
/// `_relative_l1`). On the device only the two sums come back.
fn tea_relative_l1(current: &CudaTensor, previous: &CudaTensor) -> Result<f64> {
    if current.shape != previous.shape {
        return Err(msg(format!(
            "h3 teacache probe shape changed: {:?} != {:?}",
            current.shape, previous.shape
        )));
    }
    #[cfg(feature = "cuda")]
    if let (Some(a), Some(b)) = (current.dev()?, previous.dev()?) {
        let (diff, prev) = crate::wan::ops::abs_diff_sums_device(&a, &b)?;
        return Ok(fastvideo_models::h3::sol::relative_l1_from_sums(diff, prev));
    }
    crate::wan::stats::host_fallback(
        "h3_teacache_relative_l1",
        format_args!("{:?}", current.shape),
    )?;
    Ok(fastvideo_models::h3::sol::relative_l1(
        &current.host_cow()?,
        &previous.host_cow()?,
    ))
}

/// Row indices for a sequence gather, uploaded to the device on first use.
pub struct RowGather {
    host: Vec<usize>,
    #[cfg(feature = "cuda")]
    dev: std::sync::OnceLock<cudarc::driver::CudaSlice<u32>>,
}

impl RowGather {
    pub fn new(indices: Vec<usize>) -> Self {
        Self {
            host: indices,
            #[cfg(feature = "cuda")]
            dev: std::sync::OnceLock::new(),
        }
    }

    pub fn indices(&self) -> &[usize] {
        &self.host
    }

    /// `out[:, r] = t[:, idx[r]]` for `t: [1, S, D]`: one gather kernel over a
    /// resident index buffer; the host gather runs only off-device.
    pub fn apply(&self, t: &CudaTensor) -> Result<CudaTensor> {
        if t.rank() != 3 || t.shape[0] != 1 || t.shape[1] != self.host.len() {
            return Err(msg(format!(
                "h3 sol gather: expected [1, {}, D], got {:?}",
                self.host.len(),
                t.shape
            )));
        }
        let (s, d) = (t.shape[1], t.shape[2]);
        #[cfg(feature = "cuda")]
        if let Some(table) = t.dev()? {
            let idx = match self.dev.get() {
                Some(idx) => idx,
                None => {
                    let host: Vec<u32> = self.host.iter().map(|&i| i as u32).collect();
                    let up = crate::wan::ops::upload_row_indices(&host)?;
                    let _ = self.dev.set(up);
                    self.dev.get().expect("h3 sol gather indices")
                }
            };
            let out = crate::wan::ops::index_select_rows_idx_device(&table, d, idx)?;
            return CudaTensor::from_device_slice(out, vec![1, s, d]);
        }
        t.reshape(vec![s, d])?
            .index_select_rows(&self.host)?
            .reshape_owned(vec![1, s, d])
    }
}

/// Spark `[visual | text+audio]` permutation: the input-row gather, its
/// inverse on the attention output, and the RoPE tables re-ordered once.
pub struct H3SolPermutation {
    pub forward: RowGather,
    pub inverse: RowGather,
    pub cos: CudaTensor,
    pub sin: CudaTensor,
}

/// H3 Sol-Attn policy for one request: the per-layer route plus ONE
/// contiguous sink range in the coordinates of the Q/K/V the kernel sees
/// (packed `[text | cond | audio | video]` order, or the permuted order when
/// [`Self::permutation`] is set). The sink rows are also the dense query rows.
pub struct H3SolPolicy {
    pub kind: fastvideo_models::h3::sol::H3SolAttnPolicy,
    pub sink: Option<(usize, usize)>,
    pub permutation: Option<H3SolPermutation>,
}

impl H3SolPolicy {
    pub fn from_layout(
        kind: fastvideo_models::h3::sol::H3SolAttnPolicy,
        layout: &DeviceLayout,
    ) -> Result<Self> {
        let spec = fastvideo_models::h3::sol::sink_spec(kind, &layout.layout).map_err(msg)?;
        let permutation = match spec.plan {
            None => None,
            Some(plan) => {
                let forward = RowGather::new(plan.permutation);
                let inverse = RowGather::new(plan.inverse);
                let s = layout.layout.sequence_length();
                let gather_table = |t: &CudaTensor| -> Result<CudaTensor> {
                    let r = t.numel() / s.max(1);
                    forward
                        .apply(&t.reshape(vec![1, s, r])?)?
                        .reshape_owned(t.shape.clone())
                };
                let cos = gather_table(&layout.cos)?;
                let sin = gather_table(&layout.sin)?;
                Some(H3SolPermutation {
                    forward,
                    inverse,
                    cos,
                    sin,
                })
            }
        };
        Ok(Self {
            kind,
            sink: spec.sink,
            permutation,
        })
    }

    pub fn route(
        &self,
        step: usize,
        layer: usize,
    ) -> std::result::Result<fastvideo_models::h3::sol::H3SolRoute, String> {
        fastvideo_models::h3::sol::policy_route(self.kind, step, layer)
    }

    /// One Sol layer on BHSD q/k/v already in kernel order: Sol-Attn with the
    /// exact KV sink, then the sink's query rows replaced by dense attention.
    fn attend(
        &self,
        q: &CudaTensor,
        k: &CudaTensor,
        v: &CudaTensor,
        tau: f64,
    ) -> Result<CudaTensor> {
        let (start, len) = match self.sink {
            Some((start, len)) => (Some(start), len),
            None => (None, 0),
        };
        let out = crate::sol_attn::sol_attn(q, k, v, tau, None, start, len)?;
        match self.sink {
            Some(range) => crate::sol_attn::splice_dense_ranges(&out, q, k, v, &[range], None),
            None => Ok(out),
        }
    }
}

/// How the blocks attend. The refiner is always dense.
#[derive(Clone, Copy)]
pub enum AttnMode<'a> {
    /// Full softmax attention without the compression-gate branch: the
    /// function diffusers implements, and the oracle's reference.
    Dense,
    /// VSA-H3 with `to_gate_compress`: the function the checkpoint was trained as.
    Vsa(&'a super::vsa::H3Vsa),
    /// Engine / RTX / Spark Sol-Attn policy. The block loop resolves the per-layer tau.
    Sol(&'a H3SolPolicy),
    /// One Sol-Attn layer at this tau.
    SolLayer { tau: f64, policy: &'a H3SolPolicy },
}

/// `benchmark.json` attention route of one block (see [`crate::wan::evalstats`]).
fn record_attn(step: usize, layer: usize, mode: &AttnMode<'_>) {
    use crate::wan::evalstats::{attn, AttnKind};
    let kind = match mode {
        AttnMode::Dense | AttnMode::Sol(_) => AttnKind::Dense,
        AttnMode::Vsa(_) => AttnKind::Vsa,
        AttnMode::SolLayer { tau, .. } => AttnKind::Sol { tau: *tau },
    };
    attn(Some(step), layer, kind);
}

// ---------------------------------------------------------------------------
// Time embedding and the precomputed AdaLN table
// ---------------------------------------------------------------------------

/// `time_embedder(time_proj(t))` for each `t`, on the host in float32:
/// sinusoid with cos first and an unscaled `t in [0, 1]`, then
/// `linear_2(silu(linear_1(.)))`. 16 vectors per checkpoint, so exactness
/// matters more than speed; sums accumulate in float64.
pub fn time_embeddings(
    cfg: &H3TransformerConfig,
    map: &WeightMap,
    timesteps: &[f32],
    lora: &mut Option<super::lora::H3LoraFuse>,
) -> Result<Vec<Vec<f32>>> {
    let (freq, hidden, out) = (cfg.freq_dim, cfg.time_embed_hidden_dim, cfg.time_embed_dim);
    let mut host = |key: &str, shape: &[usize]| -> Result<Vec<f32>> {
        let mut data = cuda_tensor_shaped(map, key, shape)?
            .host_cow()?
            .into_owned();
        if let Some(fuse) = lora.as_mut() {
            fuse.fuse(key, &mut data, shape)?;
        }
        Ok(data)
    };
    let (w1, b1) = (
        host("time_embedder.linear_1.weight", &[hidden, freq])?,
        host("time_embedder.linear_1.bias", &[hidden])?,
    );
    let (w2, b2) = (
        host("time_embedder.linear_2.weight", &[out, hidden])?,
        host("time_embedder.linear_2.bias", &[out])?,
    );
    let half = freq / 2;
    let linear = |x: &[f32], w: &[f32], b: &[f32]| -> Vec<f32> {
        use rayon::prelude::*;
        b.par_iter()
            .enumerate()
            .map(|(r, &bias)| {
                (f64::from(bias)
                    + x.iter()
                        .zip(&w[r * x.len()..(r + 1) * x.len()])
                        .map(|(a, c)| f64::from(*a) * f64::from(*c))
                        .sum::<f64>()) as f32
            })
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
            let h: Vec<f32> = linear(&emb, &w1, &b1)
                .into_iter()
                .map(|v| v / (1.0 + (-v).exp()))
                .collect();
            linear(&h, &w2, &b2)
        })
        .collect())
}

/// Every AdaLN modulation a T2AV/FL2VA run of one ladder reads, on the host.
///
/// Slots are `[video, text, audio]` (the modality tags). Video and text rows
/// take the video timestep, audio rows the audio one. Keyframe condition rows
/// (FL2VA) take [`KEYFRAME_NOISE_AUG`] with the video tag — a constant of the
/// ladder, stored once per block. Scales are stored as `1 + scale`.
pub struct AdaLnTable {
    steps: usize,
    blocks: usize,
    hidden: usize,
    /// `[steps, blocks, 3, 6, hidden]`.
    block_mods: Vec<f32>,
    /// `[steps, 2 (video, audio timestep), 2 (shift, 1 + scale), hidden]`.
    out_mods: Vec<f32>,
    /// `[blocks, 3, 6, hidden]`: AdaLN at [`KEYFRAME_NOISE_AUG`] for every modality tag.
    keyframe_mods: Vec<f32>,
    /// `[steps, 2, time_embed_dim]`: `temb` for (video, audio), kept for the oracle.
    pub temb: Vec<Vec<f32>>,
}

impl AdaLnTable {
    /// Stream each projection through the device once. Device peak is one
    /// `adaln_proj` (520 MB as bf16) plus a `[2 * steps, 96768]` result.
    pub fn precompute(
        cfg: &H3TransformerConfig,
        map: &WeightMap,
        schedule: &H3JointSchedule,
    ) -> Result<Self> {
        Self::precompute_with(cfg, map, schedule, &mut None)
    }

    pub fn precompute_with(
        cfg: &H3TransformerConfig,
        map: &WeightMap,
        schedule: &H3JointSchedule,
        lora: &mut Option<super::lora::H3LoraFuse>,
    ) -> Result<Self> {
        let steps = schedule.num_steps();
        let (hidden, te) = (cfg.hidden_size, cfg.time_embed_dim);
        // Row 2i is the video timestep of step i, row 2i + 1 the audio one;
        // final row is KEYFRAME_NOISE_AUG for FL2VA condition AdaLN.
        let mut timesteps: Vec<f32> = (0..steps)
            .flat_map(|i| [schedule.video.timesteps[i], schedule.audio.timesteps[i]])
            .collect();
        timesteps.push(KEYFRAME_NOISE_AUG);
        let temb_all = time_embeddings(cfg, map, &timesteps, lora)?;
        let temb: Vec<Vec<f32>> = temb_all[..2 * steps].to_vec();
        // silu in float32, THEN the cast the projection applies.
        let silu: Vec<f32> = temb_all
            .iter()
            .flatten()
            .map(|&v| v / (1.0 + (-v).exp()))
            .collect();
        let s = pinned(silu, vec![2 * steps + 1, te])?;

        let slice = ADALN_PARAMS * hidden;
        let mut block_mods = vec![0f32; steps * cfg.num_layers * MODALITY_NUM * slice];
        let mut keyframe_mods = vec![0f32; cfg.num_layers * MODALITY_NUM * slice];
        for b in 0..cfg.num_layers {
            let proj = load_linear(
                map,
                &format!("transformer_blocks.{b}.adaln_proj.linear"),
                te,
                cfg.adaln_out_dim(),
                true,
                lora,
            )?;
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
                    let dst = &mut block_mods
                        [((i * cfg.num_layers + b) * MODALITY_NUM + m) * slice..][..slice];
                    dst.copy_from_slice(src);
                    for p in [SCALE_MSA, SCALE_MLP] {
                        dst[p * hidden..(p + 1) * hidden]
                            .iter_mut()
                            .for_each(|v| *v += 1.0);
                    }
                }
            }
            // KEYFRAME_NOISE_AUG row → all modality tags for Ref2VA/FL2VA condition rows.
            let kf_row = 2 * steps;
            for tag in [TAG_VIDEO, TAG_TEXT, TAG_AUDIO] {
                let m = usize::from(tag);
                let src = &y[kf_row * width + m * slice..kf_row * width + (m + 1) * slice];
                let dst = &mut keyframe_mods[(b * MODALITY_NUM + m) * slice..][..slice];
                dst.copy_from_slice(src);
                for p in [SCALE_MSA, SCALE_MLP] {
                    dst[p * hidden..(p + 1) * hidden]
                        .iter_mut()
                        .for_each(|v| *v += 1.0);
                }
            }
            crate::wan::log::info(format_args!(
                "h3 adaln table: block {}/{}",
                b + 1,
                cfg.num_layers
            ));
        }

        // Out mods only need the ladder timesteps (video/audio heads), not keyframe.
        let s_ladder = s.narrow(0, 0, 2 * steps)?;
        let proj = load_linear(map, "norm_out.linear", te, 2 * hidden, true, lora)?;
        let y = proj.forward(&s_ladder)?;
        let y = y.host_cow()?;
        // `shift, scale = chunk(2)`: shift first.
        let mut out_mods = y.to_vec();
        for row in out_mods.chunks_exact_mut(2 * hidden) {
            row[hidden..].iter_mut().for_each(|v| *v += 1.0);
        }
        Ok(Self {
            steps,
            blocks: cfg.num_layers,
            hidden,
            block_mods,
            out_mods,
            keyframe_mods,
            temb,
        })
    }

    /// [`Self::precompute`], memoized in `cache`. Building the table reads
    /// 26 GB of projections that are needed for nothing else, so a warm start
    /// skips more than a third of the checkpoint. The file is keyed by the
    /// ladder's timesteps and a fingerprint of the checkpoint (block 0's AdaLN
    /// bias bytes); anything that does not match is rebuilt, never trusted.
    pub fn load_or_precompute(
        cfg: &H3TransformerConfig,
        map: &WeightMap,
        schedule: &H3JointSchedule,
        cache: Option<&std::path::Path>,
    ) -> Result<Self> {
        Self::load_or_precompute_with(cfg, map, schedule, cache, &mut None)
    }

    pub fn load_or_precompute_with(
        cfg: &H3TransformerConfig,
        map: &WeightMap,
        schedule: &H3JointSchedule,
        cache: Option<&std::path::Path>,
        lora: &mut Option<super::lora::H3LoraFuse>,
    ) -> Result<Self> {
        // A fused adapter changes the projections the cache was built from.
        if lora.is_some() {
            return Self::precompute_with(cfg, map, schedule, lora);
        }
        let (Some(path), Some(lazy)) = (cache, map.lazy()) else {
            return Self::precompute(cfg, map, schedule);
        };
        let fingerprint = {
            let view = lazy
                .view("transformer_blocks.0.adaln_proj.linear.bias")
                .map_err(|e| msg(e.to_string()))?;
            view.bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
            })
        };
        let steps = schedule.num_steps();
        let mut header: Vec<u64> = vec![
            CACHE_MAGIC,
            fingerprint,
            steps as u64,
            cfg.num_layers as u64,
            cfg.hidden_size as u64,
            cfg.time_embed_dim as u64,
        ];
        header.extend(
            (0..steps)
                .flat_map(|i| [schedule.video.timesteps[i], schedule.audio.timesteps[i]])
                .map(|t| u64::from(t.to_bits())),
        );
        header.push(u64::from(KEYFRAME_NOISE_AUG.to_bits()));
        let header: Vec<u8> = header.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sizes = [
            steps * cfg.num_layers * MODALITY_NUM * ADALN_PARAMS * cfg.hidden_size,
            2 * steps * 2 * cfg.hidden_size,
            cfg.num_layers * MODALITY_NUM * ADALN_PARAMS * cfg.hidden_size,
            2 * steps * cfg.time_embed_dim,
        ];

        if let Ok(bytes) = std::fs::read(path) {
            let want = header.len() + 4 * sizes.iter().sum::<usize>();
            if bytes.len() == want && bytes[..header.len()] == header[..] {
                let mut values = bytes[header.len()..]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                let block_mods: Vec<f32> = values.by_ref().take(sizes[0]).collect();
                let out_mods: Vec<f32> = values.by_ref().take(sizes[1]).collect();
                let keyframe_mods: Vec<f32> = values.by_ref().take(sizes[2]).collect();
                let flat: Vec<f32> = values.collect();
                let temb = flat
                    .chunks_exact(cfg.time_embed_dim)
                    .map(<[f32]>::to_vec)
                    .collect();
                crate::wan::log::info(format_args!("h3 adaln table: read from {}", path.display()));
                return Ok(Self {
                    steps,
                    blocks: cfg.num_layers,
                    hidden: cfg.hidden_size,
                    block_mods,
                    out_mods,
                    keyframe_mods,
                    temb,
                });
            }
            crate::wan::log::info(format_args!(
                "h3 adaln table: {} is for another checkpoint or ladder; rebuilding",
                path.display()
            ));
        }
        let table = Self::precompute_with(cfg, map, schedule, lora)?;
        let mut bytes = header;
        bytes.extend(
            table
                .block_mods
                .iter()
                .chain(&table.out_mods)
                .chain(&table.keyframe_mods)
                .chain(table.temb.iter().flatten())
                .flat_map(|v| v.to_le_bytes()),
        );
        // A cache that cannot be written costs the next start some time, not this run its result.
        let written = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(path, &bytes));
        if let Err(e) = written {
            crate::wan::log::info(format_args!(
                "h3 adaln table: could not write {}: {e}",
                path.display()
            ));
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
        &self.block_mods[((step * self.blocks + block) * MODALITY_NUM + usize::from(tag)) * slice..]
            [..slice]
    }

    /// AdaLN at [`KEYFRAME_NOISE_AUG`] for `(block, tag)`.
    pub fn keyframe_slot(&self, block: usize, tag: u8) -> &[f32] {
        let slice = ADALN_PARAMS * self.hidden;
        &self.keyframe_mods[(block * MODALITY_NUM + usize::from(tag)) * slice..][..slice]
    }

    /// `(shift, 1 + scale)` of `norm_out` for the video (`audio = false`) or
    /// audio timestep of `step`. Per timestep, not per modality.
    pub fn out_slot(&self, step: usize, audio: bool) -> (&[f32], &[f32]) {
        let row =
            &self.out_mods[(2 * step + usize::from(audio)) * 2 * self.hidden..][..2 * self.hidden];
        row.split_at(self.hidden)
    }
}

/// One block's modulation on the device: a `[1, 6, hidden]` table per contiguous
/// same-tag run (T2AV/FL2VA and interleaved Ref2VA).
struct BlockMods {
    segments: Vec<(RowRange, CudaTensor)>,
}

impl BlockMods {
    fn upload(
        table: &AdaLnTable,
        step: usize,
        block: usize,
        layout: &H3PackedLayout,
    ) -> Result<Self> {
        // This step's ladder rows + the keyframe table once per step; later
        // blocks only `narrow`.
        let (ladder, keyframe) = adaln_device_tables(table, step)?;
        let ladder = ladder.reshape(vec![
            table.blocks * MODALITY_NUM,
            ADALN_PARAMS,
            table.hidden,
        ])?;
        let keyframe = keyframe.reshape(vec![
            table.blocks * MODALITY_NUM,
            ADALN_PARAMS,
            table.hidden,
        ])?;
        let mut segments = Vec::new();
        for (range, tag) in tag_runs(layout) {
            if range.len == 0 {
                continue;
            }
            let e = if row_uses_ladder(layout, range) {
                let row = block * MODALITY_NUM + usize::from(tag);
                ladder.narrow(0, row, 1)?
            } else {
                let row = block * MODALITY_NUM + usize::from(tag);
                keyframe.narrow(0, row, 1)?
            };
            segments.push((range, e));
        }
        Ok(Self { segments })
    }

    /// `n * (1 + scale[a]) + shift[a]` with `a` the row's modality.
    fn modulate(&self, n: &CudaTensor, scale: usize, shift: usize) -> Result<CudaTensor> {
        let parts = self
            .segments
            .iter()
            .filter(|(range, _)| range.len > 0)
            .map(|(range, e)| {
                n.narrow(1, range.start, range.len)?
                    .mul(&e.narrow(1, scale, 1)?)?
                    .add(&e.narrow(1, shift, 1)?)
            })
            .collect::<Result<Vec<_>>>()?;
        CudaTensor::cat(&parts.iter().collect::<Vec<_>>(), 1)
    }

    /// `x + gate[a] * update`.
    fn gated_add(&self, x: &CudaTensor, update: &CudaTensor, gate: usize) -> Result<CudaTensor> {
        let parts = self
            .segments
            .iter()
            .filter(|(range, _)| range.len > 0)
            .map(|(range, e)| {
                x.narrow(1, range.start, range.len)?.residual_gate_add_e(
                    &update.narrow(1, range.start, range.len)?,
                    e,
                    gate,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        CudaTensor::cat(&parts.iter().collect::<Vec<_>>(), 1)
    }
}

/// `(table, step)` of the uploaded rows, then the step's ladder rows and the
/// keyframe table.
type AdaLnDeviceRows = Option<((usize, usize), CudaTensor, CudaTensor)>;

thread_local! {
    static ADALN_DEVICE: std::cell::RefCell<AdaLnDeviceRows> =
        const { std::cell::RefCell::new(None) };
}

/// Upload step `step`'s `[blocks, 3, 6, hidden]` ladder rows + the keyframe
/// table once per step; every block of the step reuses them via `narrow`.
/// Only one step lives on the device (18 MB at 50 blocks), not the whole
/// `[steps, ...]` ladder (0.9 GiB for a 50-forward recipe), and
/// [`release_adaln_device_tables`] frees it after the denoise.
fn adaln_device_tables(table: &AdaLnTable, step: usize) -> Result<(CudaTensor, CudaTensor)> {
    let key = (table.block_mods.as_ptr() as usize, step);
    ADALN_DEVICE.with(|c| {
        if let Some((k, ladder, kf)) = c.borrow().as_ref() {
            if *k == key {
                return Ok((ladder.clone(), kf.clone()));
            }
        }
        let per_step = table.blocks * MODALITY_NUM * ADALN_PARAMS * table.hidden;
        let rows = table
            .block_mods
            .get(step * per_step..(step + 1) * per_step)
            .ok_or_else(|| msg(format!("h3 adaln: step {step} of {}", table.steps)))?;
        // Drop the previous step's rows before the next upload.
        *c.borrow_mut() = None;
        let ladder = CudaTensor::from_vec(
            rows.to_vec(),
            vec![table.blocks, MODALITY_NUM, ADALN_PARAMS, table.hidden],
        )?
        .to_device()?;
        let kf = CudaTensor::from_vec(
            table.keyframe_mods.clone(),
            vec![table.blocks, MODALITY_NUM, ADALN_PARAMS, table.hidden],
        )?
        .to_device()?;
        crate::wan::ledger::set(
            crate::wan::ledger::ADALN_TABLE,
            (ladder.numel() + kf.numel()) as u64 * 4,
        );
        *c.borrow_mut() = Some((key, ladder.clone(), kf.clone()));
        Ok((ladder, kf))
    })
}

/// Free the AdaLN rows [`adaln_device_tables`] keeps on this thread.
pub fn release_adaln_device_tables() {
    ADALN_DEVICE.with(|c| *c.borrow_mut() = None);
    ADALN_ROWS16.with(|c| *c.borrow_mut() = None);
    crate::wan::ledger::clear(crate::wan::ledger::ADALN_TABLE);
}

thread_local! {
    /// `(table, step)` and that step's `[2 * blocks * 3, 6, hidden]` f32 rows
    /// for the fused bf16 kernels.
    static ADALN_ROWS16: std::cell::RefCell<Option<((usize, usize), CudaTensor)>> =
        const { std::cell::RefCell::new(None) };
}

/// The bf16-activation value of one table entry: the projection output rounds
/// to bf16 (its linear is bf16); `1 + scale` is then formed in f32, as the
/// fused kernels do. Idempotent, so an f32-built cache converts too.
fn bf16_table_value(v: f32, is_scale: bool) -> f32 {
    if is_scale {
        crate::wan::quant::bf16_round(v - 1.0) + 1.0
    } else {
        crate::wan::quant::bf16_round(v)
    }
}

/// Step `step`'s ladder rows then the keyframe table, as the fused kernels
/// read them, uploaded once per step.
fn adaln_rows16(table: &AdaLnTable, step: usize, layout: &DeviceLayout) -> Result<AdaRows> {
    let key = (table.block_mods.as_ptr() as usize, step);
    let tab = ADALN_ROWS16.with(|c| -> Result<CudaTensor> {
        if let Some((k, t)) = c.borrow().as_ref() {
            if *k == key {
                return Ok(t.clone());
            }
        }
        let per_step = table.blocks * MODALITY_NUM * ADALN_PARAMS * table.hidden;
        let rows = table
            .block_mods
            .get(step * per_step..(step + 1) * per_step)
            .ok_or_else(|| msg(format!("h3 adaln: step {step} of {}", table.steps)))?;
        *c.borrow_mut() = None;
        let h = table.hidden;
        let data: Vec<f32> = rows
            .iter()
            .chain(table.keyframe_mods.iter())
            .enumerate()
            .map(|(i, &v)| {
                let p = (i / h) % ADALN_PARAMS;
                bf16_table_value(v, p == SCALE_MSA || p == SCALE_MLP)
            })
            .collect();
        let t = CudaTensor::from_vec(data, vec![2 * table.blocks * MODALITY_NUM, ADALN_PARAMS, h])?
            .to_device()?;
        *c.borrow_mut() = Some((key, t.clone()));
        Ok(t)
    })?;
    Ok(AdaRows {
        tab,
        hidden: table.hidden,
        idx: layout.adaln_idx.clone(),
        #[cfg(feature = "cuda")]
        idx_dev: layout.adaln_idx_dev.clone(),
    })
}

/// The reference FP8 recipes are bf16-activation recipes: with
/// `FASTVIDEO_H3_QUANT` set, H3 runs bf16 activations even without
/// `FASTVIDEO_BF16_ACT`.
fn with_quant_act<R>(quant: QuantMode, f: impl FnOnce() -> R) -> R {
    if quant != QuantMode::Off && !bf16_activations() {
        crate::wan::tensor::with_bf16_act(true, f)
    } else {
        f()
    }
}

/// Which AdaLN timestep a run reads (`build_row_timesteps`, mirrored by
/// [`H3PackedLayout::timestep_indices`]): every row of the text span, including
/// the Qwen vision-pad rows that carry the video tag in FL2VA/I2V prompts, and
/// the target audio/video take this step's ladder timestep. Only the condition
/// rows between text and target audio (FL2VA keyframes, Ref2VA references)
/// read the KEYFRAME_NOISE_AUG table.
fn row_uses_ladder(layout: &H3PackedLayout, range: RowRange) -> bool {
    range.end() <= layout.text.end() || range.start >= layout.audio.start
}

/// Contiguous same-tag runs over the packed sequence, also split where the
/// text span ends and where target audio starts so each run reads one
/// AdaLN timestep.
fn tag_runs(layout: &H3PackedLayout) -> Vec<(RowRange, u8)> {
    let tags = &layout.token_tags;
    let breaks = [layout.text.end(), layout.audio.start];
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < tags.len() {
        let tag = tags[i];
        let mut j = i + 1;
        while j < tags.len() && tags[j] == tag && !breaks.contains(&j) {
            j += 1;
        }
        out.push((
            RowRange {
                start: i,
                len: j - i,
            },
            tag,
        ));
        i = j;
    }
    out
}

// ---------------------------------------------------------------------------
// Attention, FFN, blocks
// ---------------------------------------------------------------------------

#[derive(Clone)]
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
    fn load(
        map: &WeightMap,
        prefix: &str,
        cfg: &H3TransformerConfig,
        gate: bool,
        lora: &mut Option<super::lora::H3LoraFuse>,
    ) -> Result<Self> {
        let (hidden, inner, d) = (cfg.hidden_size, cfg.inner_dim(), cfg.attention_head_dim);
        let mut names = vec![
            format!("{prefix}.to_q"),
            format!("{prefix}.to_k"),
            format!("{prefix}.to_v"),
        ];
        if gate {
            names.push(format!("{prefix}.to_gate_compress"));
        }
        if lora.is_some() {
            let mut stacked = Vec::with_capacity(names.len() * inner * hidden);
            for name in &names {
                let key = format!("{name}.weight");
                let (shape, mut row) = match map.get_f32(&key) {
                    Ok(v) => v,
                    Err(e) => match lora.as_ref().and_then(|f| f.replacement(&key)) {
                        Some(v) => v,
                        None => return Err(e),
                    },
                };
                if shape != [inner, hidden] {
                    return Err(msg(format!(
                        "key {key}: shape {shape:?} != [{inner}, {hidden}]"
                    )));
                }
                lora.as_mut().unwrap().fuse(&key, &mut row, &shape)?;
                stacked.extend(row);
            }
            let qkvg = Linear::from_tensors(
                CudaTensor::from_vec(stacked, vec![names.len() * inner, hidden])?,
                None,
            )?;
            let to_out = load_linear(
                map,
                &format!("{prefix}.to_out.0"),
                inner,
                hidden,
                false,
                lora,
            )?;
            return Ok(Self {
                qkvg,
                to_out,
                has_gate: gate,
                norm_q: pinned_weight(map, &format!("{prefix}.norm_q.weight"), &[d], lora)?,
                norm_k: pinned_weight(map, &format!("{prefix}.norm_k.weight"), &[d], lora)?,
                heads: cfg.num_attention_heads,
                head_dim: d,
                eps: cfg.qk_norm_eps as f32,
            });
        }
        let keys: Vec<&str> = names.iter().map(String::as_str).collect();
        let (qkvg, to_out) = if let Some(bits) = crate::wan::affine::bits_from_env() {
            (
                Linear::load_fused_affine(map, &keys, hidden, inner, false, bits)?,
                Linear::load_affine(
                    map,
                    &format!("{prefix}.to_out.0"),
                    inner,
                    hidden,
                    false,
                    bits,
                )?,
            )
        } else {
            (
                Linear::load_fused(map, &keys, hidden, inner, false)?,
                Linear::load(map, &format!("{prefix}.to_out.0"), inner, hidden, false)?,
            )
        };
        Ok(Self {
            qkvg,
            to_out,
            has_gate: gate,
            norm_q: pinned_weight(map, &format!("{prefix}.norm_q.weight"), &[d], lora)?,
            norm_k: pinned_weight(map, &format!("{prefix}.norm_k.weight"), &[d], lora)?,
            heads: cfg.num_attention_heads,
            head_dim: d,
            eps: cfg.qk_norm_eps as f32,
        })
    }

    /// `n`: `[1, S, hidden]`. `rope`: `[S, R]` cos/sin, or `None` (refiner).
    fn forward(
        &self,
        n: &CudaTensor,
        rope: Option<(&CudaTensor, &CudaTensor)>,
        mode: AttnMode<'_>,
    ) -> Result<CudaTensor> {
        // Spark Sol layers attend in `[visual | text+audio]` order. Every op
        // before attention is row-wise, so gather the input rows (and the RoPE
        // rows) rather than Q/K/V; the inverse gather follows `to_out`.
        let permutation = match mode {
            AttnMode::SolLayer { policy, .. } => policy.permutation.as_ref(),
            _ => None,
        };
        let permuted;
        let (n, rope) = match permutation {
            Some(p) => {
                permuted = phase("h3_attn_sol_gather", || p.forward.apply(n))?;
                (&permuted, rope.map(|_| (&p.cos, &p.sin)))
            }
            None => (n, rope),
        };
        let packed = phase("h3_attn_qkvg", || self.qkvg.forward(n))?;
        self.attend_packed(packed, rope, mode, permutation)
    }

    /// Whether the fused norm+modulate may hand this attention an MXFP8
    /// activation: an all-MXFP8 QKV (no bf16 gate rows) and no row gather.
    fn accepts_mx(&self, mode: AttnMode<'_>) -> bool {
        let gathered = matches!(mode, AttnMode::SolLayer { policy, .. } if policy.permutation.is_some());
        self.qkvg.quant_kind() == Some(QuantKind::Mxfp8) && !self.has_gate && !gathered
    }

    /// [`Self::forward`] on a fused producer's output.
    fn forward_in(
        &self,
        n: NormOut,
        rows: usize,
        rope: Option<(&CudaTensor, &CudaTensor)>,
        mode: AttnMode<'_>,
    ) -> Result<CudaTensor> {
        let _ = rows;
        match n {
            NormOut::T(t) => self.forward(&t, rope, mode),
            #[cfg(feature = "cuda")]
            NormOut::Mx(act) => {
                let width = self.qkvg.out_dim();
                let packed = phase("h3_attn_qkvg", || {
                    self.qkvg.forward_mx(&act, vec![1, rows, width])
                })?;
                drop(act);
                self.attend_packed(packed, rope, mode, None)
            }
        }
    }

    fn attend_packed(
        &self,
        packed: CudaTensor,
        rope: Option<(&CudaTensor, &CudaTensor)>,
        mode: AttnMode<'_>,
        permutation: Option<&H3SolPermutation>,
    ) -> Result<CudaTensor> {
        let inner = self.heads * self.head_dim;
        // Per-head RMSNorm over D (one [D] weight for all heads), then RoPE.
        // Wan's fused `qk_norm_rope_bhsd` norms over heads*d — wrong here.
        // bf16 activations: Sol-H3's fused qk-norm + partial RoPE (f32
        // normalizer and tables, one rounding) straight from the projection.
        let fused = bf16_activations();
        let q = phase("h3_attn_q", || {
            if fused {
                return fused16::qk_norm_rope(
                    &packed, &self.norm_q, rope, self.heads, self.head_dim, 0, self.eps,
                );
            }
            let t = packed
                .split_heads_bhsd(0, self.heads, self.head_dim)?
                .rms_norm(&self.norm_q, self.eps)?;
            match rope {
                Some((cos, sin)) => t.rope_half(cos, sin),
                None => Ok(t),
            }
        })?;
        let k = phase("h3_attn_k", || {
            if fused {
                return fused16::qk_norm_rope(
                    &packed, &self.norm_k, rope, self.heads, self.head_dim, inner, self.eps,
                );
            }
            let t = packed
                .split_heads_bhsd(inner, self.heads, self.head_dim)?
                .rms_norm(&self.norm_k, self.eps)?;
            match rope {
                Some((cos, sin)) => t.rope_half(cos, sin),
                None => Ok(t),
            }
        })?;
        let v = phase("h3_attn_v", || {
            packed.split_heads_bhsd(2 * inner, self.heads, self.head_dim)
        })?;
        let (k, v) = crate::wan::nvfp4::maybe_kv(k, v)?;
        let out = match mode {
            AttnMode::Dense => scaled_dot_product_attention(&q, &k, &v, None)?,
            AttnMode::Vsa(vsa) => {
                // Gate is a plain projection of the same input: not normed, not rotated.
                let gate = if self.has_gate {
                    Some(phase("h3_attn_gate", || {
                        packed.split_heads_bhsd(3 * inner, self.heads, self.head_dim)
                    })?)
                } else {
                    None
                };
                vsa.attend(q, k, v, gate)?
            }
            AttnMode::Sol(_) => {
                return Err(msg(
                    "h3 attn: Sol policy must be resolved to a per-layer tau before Attention::forward",
                ));
            }
            AttnMode::SolLayer { tau, policy } => policy.attend(&q, &k, &v, tau)?,
        };
        drop(packed);
        let out = phase("h3_attn_out", || self.to_out.forward(&out.merge_heads()?))?;
        match permutation {
            Some(p) => phase("h3_attn_sol_gather", || p.inverse.apply(&out)),
            None => Ok(out),
        }
    }
}

/// Bias-free SwiGLU with the **value half first**: `ff_out(v * silu(g))` for
/// `(v, g) = chunk(ff_in(x), 2)`. The act is one `swiglu_value_first` pass.
/// Row-chunked only when `[S, 2 * ffn]` f32 would exceed [`FFN_WHOLE_BYTES`].
#[derive(Clone)]
pub(crate) struct FeedForward {
    ff_in: Linear,
    ff_out: Linear,
    ffn_dim: usize,
}

/// `FASTVIDEO_H3_FFN_FP8` (a per-tensor E4M3 FFN matching no reference,
/// 17-20 dB) is retired: refuse it rather than silently run bf16.
fn refuse_retired_ffn_fp8() -> Result<()> {
    if crate::wan::envflag::bool_flag("FASTVIDEO_H3_FFN_FP8", false) {
        return Err(msg(
            "FASTVIDEO_H3_FFN_FP8 is retired; use FASTVIDEO_H3_QUANT=w8a8|mxfp8 (the reference recipes)",
        ));
    }
    Ok(())
}

impl FeedForward {
    fn load(
        map: &WeightMap,
        prefix: &str,
        cfg: &H3TransformerConfig,
        lora: &mut Option<super::lora::H3LoraFuse>,
    ) -> Result<Self> {
        if lora.is_some() {
            return Ok(Self {
                ff_in: load_linear(
                    map,
                    &format!("{prefix}.net.0.proj"),
                    cfg.hidden_size,
                    2 * cfg.ffn_dim,
                    false,
                    lora,
                )?,
                ff_out: load_linear(
                    map,
                    &format!("{prefix}.net.2"),
                    cfg.ffn_dim,
                    cfg.hidden_size,
                    false,
                    lora,
                )?,
                ffn_dim: cfg.ffn_dim,
            });
        }
        let (ff_in, ff_out) = if let Some(bits) = crate::wan::affine::bits_from_env() {
            (
                Linear::load_affine(
                    map,
                    &format!("{prefix}.net.0.proj"),
                    cfg.hidden_size,
                    2 * cfg.ffn_dim,
                    false,
                    bits,
                )?,
                Linear::load_affine(
                    map,
                    &format!("{prefix}.net.2"),
                    cfg.ffn_dim,
                    cfg.hidden_size,
                    false,
                    bits,
                )?,
            )
        } else {
            refuse_retired_ffn_fp8()?;
            (
                Linear::load(
                    map,
                    &format!("{prefix}.net.0.proj"),
                    cfg.hidden_size,
                    2 * cfg.ffn_dim,
                    false,
                )?,
                Linear::load(
                    map,
                    &format!("{prefix}.net.2"),
                    cfg.ffn_dim,
                    cfg.hidden_size,
                    false,
                )?,
            )
        };
        Ok(Self {
            ff_in,
            ff_out,
            ffn_dim: cfg.ffn_dim,
        })
    }

    /// Whether the fused residual+norm+modulate may hand this FFN an MXFP8
    /// activation (MXFP8 up projection, one row chunk).
    fn accepts_mx(&self, rows: usize) -> bool {
        self.ff_in.quant_kind() == Some(QuantKind::Mxfp8) && ffn_row_chunk(rows, self.ffn_dim) >= rows
    }

    /// [`Self::forward`] on a fused producer's output; with an MXFP8 down
    /// projection the SwiGLU writes its MXFP8 input directly (`fused_swiglu_mxfp8`).
    fn forward_in(&self, n: NormOut, rows: usize) -> Result<CudaTensor> {
        let hidden = self.ff_out.out_dim();
        let t = match n {
            NormOut::T(t) => t,
            #[cfg(feature = "cuda")]
            NormOut::Mx(act) => {
                let h = phase("h3_ffn_in", || {
                    self.ff_in.forward_mx(&act, vec![1, rows, 2 * self.ffn_dim])
                })?;
                drop(act);
                crate::wan::evalstats::ffn(1);
                return self.down(h, rows, hidden);
            }
        };
        if !bf16_activations() {
            return self.forward(&t);
        }
        let chunk = ffn_row_chunk(rows, self.ffn_dim);
        let mut parts = Vec::with_capacity(rows.div_ceil(chunk).max(1));
        let mut start = 0;
        while start < rows {
            let len = chunk.min(rows - start);
            let h = phase("h3_ffn_in", || self.ff_in.forward(&t.narrow(1, start, len)?))?;
            parts.push(self.down(h, len, hidden)?);
            start += len;
        }
        crate::wan::evalstats::ffn(parts.len());
        if parts.len() == 1 {
            return Ok(parts.pop().unwrap());
        }
        CudaTensor::cat(&parts.iter().collect::<Vec<_>>(), 1)
    }

    /// SwiGLU then the down projection for one chunk of `rows`.
    fn down(&self, h: CudaTensor, rows: usize, hidden: usize) -> Result<CudaTensor> {
        #[cfg(feature = "cuda")]
        if self.ff_out.quant_kind() == Some(QuantKind::Mxfp8) {
            if let Some(act) = phase("h3_ffn_act", || fused16::swiglu_mx(&h))? {
                drop(h);
                return phase("h3_ffn_out", || self.ff_out.forward_mx(&act, vec![1, rows, hidden]));
            }
        }
        let _ = (rows, hidden);
        let act = phase("h3_ffn_act", || h.swiglu_value_first())?;
        drop(h);
        phase("h3_ffn_out", || self.ff_out.forward(&act))
    }

    fn forward(&self, n: &CudaTensor) -> Result<CudaTensor> {
        let rows = n.shape[1];
        let chunk = ffn_row_chunk(rows, self.ffn_dim);
        let mut parts = Vec::with_capacity(rows.div_ceil(chunk).max(1));
        let mut start = 0;
        while start < rows {
            let len = chunk.min(rows - start);
            let h = phase("h3_ffn_in", || {
                self.ff_in.forward(&n.narrow(1, start, len)?)
            })?;
            let act = phase("h3_ffn_act", || h.swiglu_value_first())?;
            drop(h);
            parts.push(phase("h3_ffn_out", || self.ff_out.forward(&act))?);
            start += len;
        }
        crate::wan::evalstats::ffn(parts.len());
        if parts.len() == 1 {
            return Ok(parts.pop().unwrap());
        }
        CudaTensor::cat(&parts.iter().collect::<Vec<_>>(), 1)
    }
}

#[derive(Clone)]
pub(crate) struct Block {
    norm1: CudaTensor,
    norm2: CudaTensor,
    attn: Attention,
    ff: FeedForward,
}

impl Block {
    pub(crate) fn load(
        map: &WeightMap,
        prefix: &str,
        cfg: &H3TransformerConfig,
        gate: bool,
        lora: &mut Option<super::lora::H3LoraFuse>,
    ) -> Result<Self> {
        Ok(Self {
            norm1: pinned_weight(
                map,
                &format!("{prefix}.norm1.weight"),
                &[cfg.hidden_size],
                lora,
            )?,
            norm2: pinned_weight(
                map,
                &format!("{prefix}.norm2.weight"),
                &[cfg.hidden_size],
                lora,
            )?,
            attn: Attention::load(map, &format!("{prefix}.attn"), cfg, gate, lora)?,
            ff: FeedForward::load(map, &format!("{prefix}.ff"), cfg, lora)?,
        })
    }
}

impl Block {
    /// [`Self::load`], then the reference FP8 recipe on the attention and FFN
    /// linears (after any bf16 LoRA merge, as the reference quantizes).
    pub(crate) fn load_quant(
        map: &WeightMap,
        prefix: &str,
        cfg: &H3TransformerConfig,
        gate: bool,
        lora: &mut Option<super::lora::H3LoraFuse>,
        quant: Option<QuantKind>,
    ) -> Result<Self> {
        let mut block = Self::load(map, prefix, cfg, gate, lora)?;
        if let Some(kind) = quant {
            block.quantize(kind, cfg)?;
        }
        Ok(block)
    }

    /// W8A8 keeps one tensor scale per original linear, so the fused QKV is
    /// three sections; MXFP8 scales per 32 values and quantizes it whole
    /// (`to_qkv`). The VSA gate rows (`to_gate_compress`) are in neither
    /// reference table and stay bf16.
    fn quantize(&mut self, kind: QuantKind, cfg: &H3TransformerConfig) -> Result<()> {
        let inner = cfg.inner_dim();
        let q = |rows| Section {
            rows,
            quantized: true,
        };
        let mut qkv = match kind {
            QuantKind::W8A8 => vec![q(inner), q(inner), q(inner)],
            QuantKind::Mxfp8 => vec![q(3 * inner)],
        };
        if self.attn.has_gate {
            qkv.push(Section {
                rows: inner,
                quantized: false,
            });
        }
        self.attn.qkvg.quantize(kind, qkv)?;
        self.attn.to_out.quantize(kind, vec![q(cfg.hidden_size)])?;
        self.ff.ff_in.quantize(kind, vec![q(2 * cfg.ffn_dim)])?;
        self.ff.ff_out.quantize(kind, vec![q(cfg.hidden_size)])
    }

    /// The block under bf16 activations with Sol-H3's fused elementwise chain
    /// ([`fused16`]); MXFP8 consumers receive their activation pre-quantized.
    #[allow(clippy::too_many_arguments)]
    fn forward_fused(
        &self,
        x: &CudaTensor,
        rows_tab: &AdaRows,
        base: usize,
        rope: Option<(&CudaTensor, &CudaTensor)>,
        mode: AttnMode<'_>,
        eps: f32,
        slots: [usize; 6],
    ) -> Result<CudaTensor> {
        let [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] = slots;
        let rows = x.shape[1];
        let n = phase("h3_1_norm_msa", || {
            fused16::norm_mod(
                x,
                &self.norm1,
                rows_tab,
                base,
                scale_msa,
                shift_msa,
                eps,
                self.attn.accepts_mx(mode),
            )
        })?;
        let a = phase("h3_2_attn", || self.attn.forward_in(n, rows, rope, mode))?;
        let (x, n) = phase("h3_3_residual_msa", || {
            fused16::res_gate_norm_mod(
                x,
                &a,
                &self.norm2,
                rows_tab,
                base,
                (gate_msa, scale_mlp, shift_mlp),
                eps,
                self.ff.accepts_mx(rows),
            )
        })?;
        drop(a);
        let f = phase("h3_5_ffn", || self.ff.forward_in(n, rows))?;
        phase("h3_6_residual_ffn", || {
            fused16::gate_residual(&x, &f, rows_tab, base, gate_mlp)
        })
    }
}

impl OffloadBlock for Block {
    fn for_each_linear_mut(&mut self, f: &mut dyn FnMut(&mut Linear) -> Result<()>) -> Result<()> {
        f(&mut self.attn.qkvg)?;
        f(&mut self.attn.to_out)?;
        f(&mut self.ff.ff_in)?;
        f(&mut self.ff.ff_out)
    }
}

/// `context_embedder` + the two timestep-free refiner blocks + `final_norm`.
/// Runs once per prompt; drop it afterwards (1.6 GB).
pub struct H3TextRefiner {
    context_embedder: Linear,
    blocks: BlockWeights<Block>,
    final_norm: CudaTensor,
    eps: f32,
    final_eps: f32,
    quant: QuantMode,
}

impl H3TextRefiner {
    pub fn load(cfg: &H3TransformerConfig, map: &WeightMap) -> Result<Self> {
        Self::load_with(cfg, map, &mut None)
    }

    pub fn load_with(
        cfg: &H3TransformerConfig,
        map: &WeightMap,
        lora: &mut Option<super::lora::H3LoraFuse>,
    ) -> Result<Self> {
        Self::load_with_residency(cfg, map, lora, Residency::Resident)
    }

    /// [`Self::load_with`] with the refiner blocks resident or streamed.
    pub fn load_with_residency(
        cfg: &H3TransformerConfig,
        map: &WeightMap,
        lora: &mut Option<super::lora::H3LoraFuse>,
        residency: Residency,
    ) -> Result<Self> {
        let mut blocks = BlockWeights::new("h3 refiner", residency)
            .with_ledger(crate::wan::ledger::REFINER_RING);
        let quant = QuantMode::from_env().map_err(msg)?;
        let plan = H3QuantPlan::new(quant, cfg.num_layers, cfg.num_refiner_layers);
        for i in 0..cfg.num_refiner_layers {
            blocks.push(Block::load_quant(
                map,
                &format!("token_refiner.refiner_blocks.{i}"),
                cfg,
                false,
                lora,
                plan.refiner(),
            )?)?;
        }
        Ok(Self {
            context_embedder: load_linear(
                map,
                "context_embedder",
                cfg.text_dim,
                cfg.hidden_size,
                true,
                lora,
            )?,
            blocks,
            final_norm: pinned_weight(
                map,
                "token_refiner.final_norm.weight",
                &[cfg.hidden_size],
                lora,
            )?,
            eps: cfg.norm_eps as f32,
            final_eps: cfg.final_norm_eps as f32,
            quant,
        })
    }

    /// `[1, N, text_dim]` hidden states to `[1, N, hidden]`: plain pre-norm
    /// blocks, bidirectional attention, no RoPE, no AdaLN.
    pub fn forward(&self, text: &CudaTensor) -> Result<CudaTensor> {
        with_quant_act(self.quant, || self.forward_inner(text))
    }

    fn forward_inner(&self, text: &CudaTensor) -> Result<CudaTensor> {
        let mut e = self.context_embedder.forward(text)?;
        for index in 0..self.blocks.len() {
            e = self.blocks.with(index, |block| {
                let a = block.attn.forward(
                    &e.rms_norm(&block.norm1, self.eps)?,
                    None,
                    AttnMode::Dense,
                )?;
                let e = e.add(&a)?;
                let f = block.ff.forward(&e.rms_norm(&block.norm2, self.eps)?)?;
                e.add(&f)
            })?;
        }
        self.blocks.report("text refine");
        // Once per prompt: a streamed refiner's slots are not kept for the DiT.
        self.blocks.release_device();
        e.rms_norm(&self.final_norm, self.final_eps)
    }
}

/// The packed layout with its rotary tables on the device: built once per
/// request, shared by every block of every step.
pub struct DeviceLayout {
    pub layout: H3PackedLayout,
    cos: CudaTensor,
    sin: CudaTensor,
    /// Per row: which AdaLN table row the fused kernels read
    /// (`src * blocks * 3 + modality`, `src` 0 = ladder, 1 = keyframe).
    adaln_idx: Arc<Vec<u32>>,
    #[cfg(feature = "cuda")]
    adaln_idx_dev: Option<Arc<cudarc::driver::CudaSlice<u32>>>,
}

impl DeviceLayout {
    pub fn new(cfg: &H3TransformerConfig, layout: H3PackedLayout) -> Result<Self> {
        let (cos, sin) = layout.rope_tables(&cfg.rope_inv_freq());
        let shape = vec![layout.sequence_length(), cfg.rotary_dim()];
        let mut idx = vec![0u32; layout.sequence_length()];
        for (range, tag) in tag_runs(&layout) {
            let src = usize::from(!row_uses_ladder(&layout, range));
            let v = (src * cfg.num_layers * MODALITY_NUM + usize::from(tag)) as u32;
            idx[range.start..range.end()].fill(v);
        }
        #[cfg(feature = "cuda")]
        let adaln_idx_dev = if crate::wan::stats::device_expected() {
            Some(Arc::new(crate::wan::ops::upload_row_indices(&idx)?))
        } else {
            None
        };
        Ok(Self {
            cos: pinned(cos, shape.clone())?,
            sin: pinned(sin, shape)?,
            layout,
            adaln_idx: Arc::new(idx),
            #[cfg(feature = "cuda")]
            adaln_idx_dev,
        })
    }
}

pub struct H3Transformer {
    cfg: H3TransformerConfig,
    proj_in: Linear,
    audio_proj_in: Linear,
    blocks: BlockWeights<Block>,
    norm_out: CudaTensor,
    proj_out: Linear,
    audio_proj_out: Linear,
    table: AdaLnTable,
    /// `norm_out` modulation `[steps * 2 * 2, hidden]` resident on the device:
    /// row `(2 * step + audio) * 2` is the shift, the next row `1 + scale`.
    out_mods: CudaTensor,
    has_gate: bool,
    /// Empty unless `FASTVIDEO_H3_SOL_CACHE=teacache`.
    sol_tea: Arc<Mutex<Option<H3TeaRuntime>>>,
    /// `FASTVIDEO_H3_QUANT`: the reference FP8 recipe on the block linears.
    quant: QuantMode,
}

struct H3TeaRuntime {
    state: H3TeaCache,
    signal: Option<CudaTensor>,
    residual: Option<CudaTensor>,
    pending: Option<CudaTensor>,
}

impl H3TeaRuntime {
    /// Book the buffers kept between forwards in the device ledger.
    fn book(&self) {
        let bytes = [&self.signal, &self.residual, &self.pending]
            .iter()
            .filter_map(|t| t.as_ref())
            .map(CudaTensor::stored_bytes)
            .sum();
        crate::wan::ledger::set(crate::wan::ledger::STEP_CACHE, bytes);
    }
}

impl H3Transformer {
    /// Loads the resident part of the stack (41 GiB as bf16 with the VSA gates,
    /// 37 GiB without) and precomputes the AdaLN table for `schedule`.
    /// `with_gate` loads `to_gate_compress`, which only [`AttnMode::Vsa`] reads.
    pub fn load(
        cfg: H3TransformerConfig,
        map: &WeightMap,
        schedule: &H3JointSchedule,
        with_gate: bool,
    ) -> Result<Self> {
        Self::load_cached(cfg, map, schedule, with_gate, None)
    }

    /// [`Self::load`] with the AdaLN table memoized at `adaln_cache`
    /// (see [`AdaLnTable::load_or_precompute`]).
    pub fn load_cached(
        cfg: H3TransformerConfig,
        map: &WeightMap,
        schedule: &H3JointSchedule,
        with_gate: bool,
        adaln_cache: Option<&std::path::Path>,
    ) -> Result<Self> {
        Self::load_with(cfg, map, schedule, with_gate, adaln_cache, &mut None)
    }

    pub fn load_with(
        cfg: H3TransformerConfig,
        map: &WeightMap,
        schedule: &H3JointSchedule,
        with_gate: bool,
        adaln_cache: Option<&std::path::Path>,
        lora: &mut Option<super::lora::H3LoraFuse>,
    ) -> Result<Self> {
        Self::load_with_residency(
            cfg,
            map,
            schedule,
            with_gate,
            adaln_cache,
            lora,
            Residency::Resident,
        )
    }

    /// [`Self::load_with`] with the 50 blocks resident or streamed
    /// ([`crate::wan::offload`]): streamed, each block leaves the device as
    /// soon as it is loaded and a forward holds `lookahead + 1` of them.
    pub fn load_with_residency(
        cfg: H3TransformerConfig,
        map: &WeightMap,
        schedule: &H3JointSchedule,
        with_gate: bool,
        adaln_cache: Option<&std::path::Path>,
        lora: &mut Option<super::lora::H3LoraFuse>,
        residency: Residency,
    ) -> Result<Self> {
        if cfg.rotary_dim() > cfg.attention_head_dim || cfg.freq_dim % 2 != 0 {
            return Err(msg(format!(
                "h3 dit: {} rotary channels of a {}-wide head",
                cfg.rotary_dim(),
                cfg.attention_head_dim
            )));
        }
        let quant = QuantMode::from_env().map_err(msg)?;
        // The AdaLN projections are evaluated in bf16 under the recipes too.
        let table = with_quant_act(quant, || {
            AdaLnTable::load_or_precompute_with(&cfg, map, schedule, adaln_cache, lora)
        })?;
        let plan = H3QuantPlan::new(quant, cfg.num_layers, cfg.num_refiner_layers);
        if quant != QuantMode::Off {
            crate::wan::log::info(format_args!(
                "h3 quant: {} ({} reference linears incl. refiner; {}), bf16 activations",
                quant.as_str(),
                plan.reference_linear_count(),
                match quant {
                    QuantMode::W8A8 => "FastVideo tensorwise W8A8, all blocks + refiner",
                    _ => "Sol-H3 MXFP8, blocks 2..=46",
                }
            ));
        }
        let mut blocks = BlockWeights::new("h3 dit", residency);
        for i in 0..cfg.num_layers {
            blocks.push(Block::load_quant(
                map,
                &format!("transformer_blocks.{i}"),
                &cfg,
                with_gate,
                lora,
                plan.dit_block(i),
            )?)?;
            crate::wan::log::info(format_args!(
                "h3 dit: block {}/{} {}",
                i + 1,
                cfg.num_layers,
                residency.as_str()
            ));
        }
        crate::wan::log::info(format_args!("{}", blocks.describe()));
        // `_keep_in_fp32_modules`: the patch projections stay f32 under bf16
        // activations (the reference casts into and out of them).
        let island = |mut l: Linear| {
            l.set_f32_island();
            l
        };
        Ok(Self {
            proj_in: island(load_linear(
                map,
                "proj_in",
                cfg.video_patch_dim(),
                cfg.hidden_size,
                true,
                lora,
            )?),
            audio_proj_in: island(load_linear(
                map,
                "audio_proj_in",
                cfg.audio_in_channels,
                cfg.hidden_size,
                true,
                lora,
            )?),
            blocks,
            norm_out: pinned_weight(map, "norm_out.norm.weight", &[cfg.hidden_size], lora)?,
            proj_out: island(load_linear(
                map,
                "proj_out",
                cfg.hidden_size,
                cfg.video_patch_dim(),
                true,
                lora,
            )?),
            audio_proj_out: island(load_linear(
                map,
                "audio_proj_out",
                cfg.hidden_size,
                cfg.audio_in_channels,
                true,
                lora,
            )?),
            out_mods: pinned(
                table.out_mods.clone(),
                vec![table.out_mods.len() / cfg.hidden_size, cfg.hidden_size],
            )?,
            table,
            has_gate: with_gate,
            sol_tea: Default::default(),
            quant,
            cfg,
        })
    }

    /// Install RTX TeaCache. The default path leaves this empty.
    pub fn enable_sol_teacache(&self, num_forwards: usize) -> Result<()> {
        let state = H3TeaCache::official(num_forwards).map_err(msg)?;
        crate::wan::log::info(format_args!(
            "{}",
            fastvideo_models::h3::sol::TEACACHE_APPLIED
        ));
        *self.sol_tea.lock().expect("h3 sol tea") = Some(H3TeaRuntime {
            state,
            signal: None,
            residual: None,
            pending: None,
        });
        Ok(())
    }

    /// End of a denoise: free the TeaCache signal / residual / pending rows
    /// (three `[S, hidden]` float32 buffers, 2.2 GiB at 768p 5 s) and restart
    /// its state, so nothing of this request rides into the decode or the
    /// next request. Also frees this step's AdaLN rows.
    pub fn end_denoise(&self) {
        if let Some(rt) = self.sol_tea.lock().expect("h3 sol tea").as_mut() {
            rt.signal = None;
            rt.residual = None;
            rt.pending = None;
            rt.state.reset();
        }
        crate::wan::ledger::clear(crate::wan::ledger::STEP_CACHE);
        release_adaln_device_tables();
    }

    pub fn sol_teacache_enabled(&self) -> bool {
        self.sol_tea.lock().expect("h3 sol tea").is_some()
    }

    /// `Some(true)` reuses the block residual. `Some(false)` runs the blocks.
    /// `None` leaves the forward dense.
    fn begin_sol_tea(
        &self,
        step: usize,
        hidden: &CudaTensor,
        layout: &H3PackedLayout,
        eps: f32,
    ) -> Result<Option<bool>> {
        let mut slot = self.sol_tea.lock().expect("h3 sol tea");
        let Some(runtime) = slot.as_mut() else {
            return Ok(None);
        };
        let mods = BlockMods::upload(&self.table, step, 0, layout)?;
        let probe = mods.modulate(
            &hidden.rms_norm(&self.blocks.skeleton(0).norm1, eps)?,
            SCALE_MSA,
            SHIFT_MSA,
        )?;
        let rel = if runtime.state.needs_signal(step) {
            let previous = runtime.signal.as_ref().expect("h3 teacache signal");
            tea_relative_l1(&probe, previous)?
        } else {
            0.0
        };
        let decision = runtime.state.decide(step, rel);
        crate::wan::evalstats::teacache_decision(
            step,
            decision.compute,
            decision.reason,
            decision.relative_l1,
            decision.indicator,
            decision.accumulator,
        );
        runtime.signal = Some(probe);
        if decision.compute {
            runtime.pending = Some(hidden.clone());
            runtime.book();
            Ok(Some(false))
        } else {
            crate::wan::log::debug(format_args!(
                "h3 sol teacache reuse step {step} reason {}",
                decision.reason
            ));
            Ok(Some(true))
        }
    }

    fn add_sol_tea_residual(&self, hidden: CudaTensor) -> Result<CudaTensor> {
        let residual = {
            let slot = self.sol_tea.lock().expect("h3 sol tea");
            let runtime = slot.as_ref().expect("h3 sol tea");
            runtime.residual.clone().expect("h3 sol tea residual")
        };
        hidden.add(&residual)
    }

    fn finish_sol_tea(&self, after: &CudaTensor) -> Result<()> {
        let mut slot = self.sol_tea.lock().expect("h3 sol tea");
        let runtime = slot.as_mut().expect("h3 sol tea");
        let before = runtime.pending.take().expect("h3 sol tea pending");
        runtime.state.note_computed();
        runtime.residual = Some(after.sub(&before)?);
        runtime.book();
        Ok(())
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

    /// Where the blocks live for this run.
    pub fn residency(&self) -> Residency {
        self.blocks.residency()
    }

    /// Log (and reset) the block streaming counters: H2D throughput and how
    /// much of each copy the previous block's compute hid.
    pub fn report_offload(&self, what: &str) -> crate::wan::offload::OffloadStats {
        self.blocks.report(what)
    }

    /// Streamed: free the device slots until the next forward (before a
    /// decode that needs the memory). Resident: nothing.
    pub fn release_offload_device(&self) {
        self.blocks.release_device();
    }

    /// One forward at ladder step `step`. `video_rows` / `audio_rows` are the
    /// **target** modality rows. `cond_rows` / `cond_audio_rows` are fixed
    /// Ref2VA/FL2VA condition rows (video-tagged and audio-tagged).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        step: usize,
        video_rows: &CudaTensor,
        audio_rows: &CudaTensor,
        text: &CudaTensor,
        layout: &DeviceLayout,
        mode: AttnMode<'_>,
        observer: Option<Observer<'_>>,
        cond_rows: Option<&CudaTensor>,
        cond_audio_rows: Option<&CudaTensor>,
    ) -> Result<(CudaTensor, CudaTensor)> {
        with_quant_act(self.quant, || {
            self.forward_inner(
                step,
                video_rows,
                audio_rows,
                text,
                layout,
                mode,
                observer,
                cond_rows,
                cond_audio_rows,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        step: usize,
        video_rows: &CudaTensor,
        audio_rows: &CudaTensor,
        text: &CudaTensor,
        layout: &DeviceLayout,
        mode: AttnMode<'_>,
        mut observer: Option<Observer<'_>>,
        cond_rows: Option<&CudaTensor>,
        cond_audio_rows: Option<&CudaTensor>,
    ) -> Result<(CudaTensor, CudaTensor)> {
        let cfg = &self.cfg;
        let l = &layout.layout;
        let hidden = cfg.hidden_size;
        if step >= self.table.steps() {
            return Err(msg(format!(
                "h3 dit: step {step} of a {}-step AdaLN table",
                self.table.steps()
            )));
        }
        if video_rows.shape != [l.video.len, cfg.video_patch_dim()]
            || audio_rows.shape != [l.audio.len, cfg.audio_in_channels]
            || text.shape != [1, l.text.len, hidden]
        {
            return Err(msg(format!(
                "h3 dit: rows video {:?} audio {:?} text {:?} for a layout of {} + {} + {}",
                video_rows.shape,
                audio_rows.shape,
                text.shape,
                l.text.len,
                l.audio.len,
                l.video.len
            )));
        }
        let want_v = l.num_condition_video_rows;
        let want_a = l.num_condition_audio_rows;
        match (want_v, cond_rows) {
            (0, None) => {}
            (n, Some(c)) if c.shape == [n, cfg.video_patch_dim()] => {}
            (n, got) => {
                return Err(msg(format!(
                    "h3 dit: need {n} cond video rows, got {:?}",
                    got.map(|t| t.shape.clone())
                )));
            }
        }
        match (want_a, cond_audio_rows) {
            (0, None) => {}
            (n, Some(c)) if c.shape == [n, cfg.audio_in_channels] => {}
            (n, got) => {
                return Err(msg(format!(
                    "h3 dit: need {n} cond audio rows, got {:?}",
                    got.map(|t| t.shape.clone())
                )));
            }
        }
        if matches!(mode, AttnMode::Vsa(_)) && !self.has_gate {
            return Err(msg(
                "h3 dit: VSA needs to_gate_compress; load the transformer with the gate",
            ));
        }
        if matches!(mode, AttnMode::Vsa(_)) && want_a > 0 {
            return Err(msg("h3 dit: interleaved Ref2VA condition audio needs dense attention (VSA prefix assumes contiguous layout)"));
        }
        let seq = l.sequence_length();
        let x = {
            let video = self.proj_in.forward(video_rows)?;
            let audio = self.audio_proj_in.forward(audio_rows)?;
            let text_rows = text.reshape(vec![l.text.len, hidden])?;
            if want_v == 0 && want_a == 0 {
                CudaTensor::cat(&[&text_rows, &audio, &video], 0)?
                    .reshape_owned(vec![1, seq, hidden])?
            } else if want_a == 0 {
                // Contiguous condition video (FL2VA / image-only Ref2VA).
                let cond = self.proj_in.forward(cond_rows.unwrap())?;
                CudaTensor::cat(&[&text_rows, &cond, &audio, &video], 0)?
                    .reshape_owned(vec![1, seq, hidden])?
            } else {
                // Interleaved Ref2VA: project streams then cat by ref_segments.
                let cv = self.proj_in.forward(cond_rows.unwrap())?;
                let ca = self.audio_proj_in.forward(cond_audio_rows.unwrap())?;
                let mut parts: Vec<CudaTensor> = Vec::with_capacity(2 + l.ref_segments.len() + 2);
                parts.push(text_rows);
                let (mut vi, mut ai) = (0usize, 0usize);
                for seg in &l.ref_segments {
                    match *seg {
                        fastvideo_models::h3::reference::RefSegment::Video { rows } => {
                            parts.push(cv.narrow(0, vi, rows)?);
                            vi += rows;
                        }
                        fastvideo_models::h3::reference::RefSegment::Audio { rows } => {
                            parts.push(ca.narrow(0, ai, rows)?);
                            ai += rows;
                        }
                    }
                }
                parts.push(audio);
                parts.push(video);
                let refs: Vec<&CudaTensor> = parts.iter().collect();
                CudaTensor::cat(&refs, 0)?.reshape_owned(vec![1, seq, hidden])?
            }
        };
        let rope = Some((&layout.cos, &layout.sin));
        let eps = cfg.norm_eps as f32;

        // bf16 activations: the residual stream is bf16 and every block runs
        // Sol-H3's fused elementwise chain over one step-wide AdaLN table.
        let fused = bf16_activations();
        let mut x = if fused { x.quantize_bf16()? } else { x };
        let ada = if fused {
            Some(adaln_rows16(&self.table, step, layout)?)
        } else {
            None
        };
        let tea = self.begin_sol_tea(step, &x, l, eps)?;
        if tea == Some(true) {
            x = self.add_sol_tea_residual(x)?;
        } else {
            for index in 0..self.blocks.len() {
                if let Some(ada) = &ada {
                    let layer_mode = match mode {
                        AttnMode::Sol(policy) => match policy.route(step, index).map_err(msg)? {
                            fastvideo_models::h3::sol::H3SolRoute::Dense => AttnMode::Dense,
                            fastvideo_models::h3::sol::H3SolRoute::Sol { tau } => {
                                AttnMode::SolLayer { tau, policy }
                            }
                        },
                        other => other,
                    };
                    record_attn(step, index, &layer_mode);
                    x = self.blocks.with(index, |block| {
                        block.forward_fused(
                            &x,
                            ada,
                            index * MODALITY_NUM,
                            rope,
                            layer_mode,
                            eps,
                            [SHIFT_MSA, SCALE_MSA, GATE_MSA, SHIFT_MLP, SCALE_MLP, GATE_MLP],
                        )
                    })?;
                    if let Some(observe) = observer.as_mut() {
                        observe(&format!("block_{index}"), &x)?;
                    }
                    crate::wan::log::info(format_args!(
                        "h3 dit step {step}: block {}/{}",
                        index + 1,
                        self.blocks.len()
                    ));
                    continue;
                }
                let mods = BlockMods::upload(&self.table, step, index, l)?;
                let layer_mode = match mode {
                    AttnMode::Sol(policy) => match policy.route(step, index).map_err(msg)? {
                        fastvideo_models::h3::sol::H3SolRoute::Dense => AttnMode::Dense,
                        fastvideo_models::h3::sol::H3SolRoute::Sol { tau } => {
                            AttnMode::SolLayer { tau, policy }
                        }
                    },
                    other => other,
                };
                record_attn(step, index, &layer_mode);
                x = self.blocks.with(index, |block| {
                    let n = phase("h3_1_norm_msa", || {
                        mods.modulate(&x.rms_norm(&block.norm1, eps)?, SCALE_MSA, SHIFT_MSA)
                    })?;
                    let a = phase("h3_2_attn", || block.attn.forward(&n, rope, layer_mode))?;
                    drop(n);
                    let x = phase("h3_3_residual_msa", || mods.gated_add(&x, &a, GATE_MSA))?;
                    drop(a);
                    let n = phase("h3_4_norm_ffn", || {
                        mods.modulate(&x.rms_norm(&block.norm2, eps)?, SCALE_MLP, SHIFT_MLP)
                    })?;
                    let f = phase("h3_5_ffn", || block.ff.forward(&n))?;
                    drop(n);
                    phase("h3_6_residual_ffn", || mods.gated_add(&x, &f, GATE_MLP))
                })?;
                if let Some(observe) = observer.as_mut() {
                    observe(&format!("block_{index}"), &x)?;
                }
                crate::wan::log::info(format_args!(
                    "h3 dit step {step}: block {}/{}",
                    index + 1,
                    self.blocks.len()
                ));
            }
            if tea == Some(false) {
                self.finish_sol_tea(&x)?;
            }
        }

        // Both heads are defined on every row; only each modality's own rows are read.
        let head = |range: RowRange, audio: bool, proj: &Linear| -> Result<CudaTensor> {
            let row = (2 * step + usize::from(audio)) * 2;
            let shift = self.out_mods.narrow(0, row, 1)?.reshape(vec![hidden])?;
            let scale = self.out_mods.narrow(0, row + 1, 1)?.reshape(vec![hidden])?;
            let n = x
                .narrow(1, range.start, range.len)?
                .rms_norm(&self.norm_out, cfg.final_norm_eps as f32)?;
            // bf16: `norm(x) * (1 + scale) + shift` in bf16 (the reference's
            // norm_out linear is bf16, so `1 + scale` rounds), then the f32
            // `proj_out` island.
            let (scale, shift) = if bf16_activations() {
                (scale.quantize_bf16()?, shift.quantize_bf16()?)
            } else {
                (scale, shift)
            };
            proj.forward(&n.mul(&scale)?.add(&shift)?)?
                .reshape(vec![range.len, proj.out_dim()])
        };
        Ok((
            head(l.video, false, &self.proj_out)?,
            head(l.audio, true, &self.audio_proj_out)?,
        ))
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

    #[test]
    fn the_retired_ffn_fp8_flag_is_refused_not_ignored() {
        assert!(refuse_retired_ffn_fp8().is_ok(), "unset by default");
    }

    #[test]
    fn swiglu_value_first_matches_narrow_silu_mul() {
        let (rows, half) = (5usize, 7usize);
        let x: Vec<f32> = (0..rows * 2 * half)
            .map(|i| (i as f32 * 0.17 - 1.3).sin())
            .collect();
        let t = CudaTensor::from_vec(x.clone(), vec![1, rows, 2 * half]).unwrap();
        let got = t.swiglu_value_first().unwrap();
        assert_eq!(got.shape, vec![1, rows, half]);
        let want: Vec<f32> = (0..rows * half)
            .map(|i| {
                let r = i / half;
                let c = i % half;
                x[r * 2 * half + c] * silu(x[r * 2 * half + half + c])
            })
            .collect();
        let g = got.host_cow().unwrap();
        for (i, (a, b)) in g.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-6, "i={i} {a} vs {b}");
        }
        let odd = CudaTensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        assert!(odd.swiglu_value_first().is_err());
    }

    pub(crate) fn weights() -> WeightMap {
        WeightMap::generated(|key, shape| {
            let seed = key
                .bytes()
                .fold(13u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
            let n: usize = shape.iter().product();
            (0..n)
                .map(|i| {
                    let u = (seed.wrapping_add(i as u32).wrapping_mul(2_654_435_761) >> 8) as f32
                        / (1u32 << 24) as f32;
                    if key.contains("norm") {
                        0.5 + u
                    } else {
                        (u - 0.5) * 0.8
                    }
                })
                .collect()
        })
    }

    fn get(map: &WeightMap, key: &str, shape: &[usize]) -> Vec<f32> {
        cuda_tensor_shaped(map, key, shape)
            .unwrap()
            .host_cow()
            .unwrap()
            .into_owned()
    }

    fn lin(map: &WeightMap, prefix: &str, x: &[f32], o: usize, bias: bool) -> Vec<f32> {
        let i = x.len();
        let w = get(map, &format!("{prefix}.weight"), &[o, i]);
        let b = if bias {
            get(map, &format!("{prefix}.bias"), &[o])
        } else {
            vec![0.0; o]
        };
        (0..o)
            .map(|r| b[r] + (0..i).map(|c| x[c] * w[r * i + c]).sum::<f32>())
            .collect()
    }

    fn rms(v: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let ms = v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32;
        v.iter()
            .zip(w)
            .map(|(a, g)| a / (ms + eps).sqrt() * g)
            .collect()
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
        let (h, heads, d, eps) = (
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.attention_head_dim,
            1e-5f32,
        );
        let s = x.len();
        let (n1, n2) = (
            get(map, &format!("{prefix}.norm1.weight"), &[h]),
            get(map, &format!("{prefix}.norm2.weight"), &[h]),
        );
        let (nq, nk) = (
            get(map, &format!("{prefix}.attn.norm_q.weight"), &[d]),
            get(map, &format!("{prefix}.attn.norm_k.weight"), &[d]),
        );
        // mods[row] = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] x hidden, RAW scales.
        let modulate = |v: Vec<f32>, row: usize, shift: usize, scale: usize| -> Vec<f32> {
            match mods {
                Some(m) => (0..h)
                    .map(|c| v[c] * (1.0 + m[row][scale * h + c]) + m[row][shift * h + c])
                    .collect(),
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
        let normed: Vec<Vec<f32>> = x
            .iter()
            .enumerate()
            .map(|(r, v)| modulate(rms(v, &n1, eps), r, 0, 1))
            .collect();
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
        let (q, k, v) = (
            proj("to_q", Some(&nq)),
            proj("to_k", Some(&nk)),
            proj("to_v", None),
        );
        (0..s)
            .map(|i| {
                let mut attn = vec![0f32; heads * d];
                for hd in 0..heads {
                    let scores: Vec<f32> = (0..s)
                        .map(|j| {
                            q[i][hd]
                                .iter()
                                .zip(&k[j][hd])
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                / (d as f32).sqrt()
                        })
                        .collect();
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
                let f = lin(
                    map,
                    &format!("{prefix}.ff.net.0.proj"),
                    &m,
                    2 * cfg.ffn_dim,
                    false,
                );
                let act: Vec<f32> = (0..cfg.ffn_dim)
                    .map(|c| f[c] * silu(f[cfg.ffn_dim + c]))
                    .collect();
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
        let h: Vec<f32> = lin(
            map,
            "time_embedder.linear_1",
            &emb,
            cfg.time_embed_hidden_dim,
            true,
        )
        .into_iter()
        .map(silu)
        .collect();
        lin(map, "time_embedder.linear_2", &h, cfg.time_embed_dim, true)
    }

    fn seeded(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * k + 0.3).sin()).collect()
    }

    #[test]
    fn the_refiner_is_a_plain_pre_norm_stack_without_rope() {
        let (cfg, map) = (tiny_cfg(), weights());
        let text = seeded(4 * cfg.text_dim, 0.7);
        let got = H3TextRefiner::load(&cfg, &map)
            .unwrap()
            .forward(&CudaTensor::from_vec(text.clone(), vec![1, 4, cfg.text_dim]).unwrap())
            .unwrap();
        assert_eq!(got.shape, vec![1, 4, cfg.hidden_size]);
        let x: Vec<Vec<f32>> = text
            .chunks(cfg.text_dim)
            .map(|r| lin(&map, "context_embedder", r, cfg.hidden_size, true))
            .collect();
        let x = ref_block(&cfg, &map, "token_refiner.refiner_blocks.0", &x, None, None);
        let fin = get(&map, "token_refiner.final_norm.weight", &[cfg.hidden_size]);
        let got = got.host_cow().unwrap();
        for (r, row) in x.iter().enumerate() {
            for (c, w) in rms(row, &fin, 1e-5).iter().enumerate() {
                assert!(
                    (got[r * cfg.hidden_size + c] - w).abs() < 2e-5,
                    "row {r} ch {c}"
                );
            }
        }
    }

    /// Streamed blocks (weights off the device between forwards, installed per
    /// block) run the same kernels on the same weights: the outputs are equal
    /// bit for bit, over two steps (a full wrap of the ring) and the refiner.
    #[test]
    fn streamed_blocks_match_resident_bit_for_bit() {
        let (cfg, map) = (tiny_cfg(), weights());
        let schedule = H3JointSchedule::fasth3_8step();
        let load = |residency| {
            H3Transformer::load_with_residency(
                cfg.clone(),
                &map,
                &schedule,
                false,
                None,
                &mut None,
                residency,
            )
            .unwrap()
        };
        let (resident, streamed) = (load(Residency::Resident), load(Residency::Streamed));
        assert_eq!(streamed.residency(), Residency::Streamed);
        let layout = H3PackedLayout::new(2, (2, 2, 4), 1, cfg.patch_size).unwrap();
        let (nv, na, nt) = (layout.video.len, layout.audio.len, layout.text.len);
        let dl = DeviceLayout::new(&cfg, layout).unwrap();
        let video = CudaTensor::from_vec(
            seeded(nv * cfg.video_patch_dim(), 0.41),
            vec![nv, cfg.video_patch_dim()],
        )
        .unwrap();
        let audio = CudaTensor::from_vec(
            seeded(na * cfg.audio_in_channels, 0.23),
            vec![na, cfg.audio_in_channels],
        )
        .unwrap();
        let text = CudaTensor::from_vec(
            seeded(nt * cfg.hidden_size, 0.57),
            vec![1, nt, cfg.hidden_size],
        )
        .unwrap();
        let bits = |t: &CudaTensor| -> Vec<u32> {
            t.host_cow().unwrap().iter().map(|v| v.to_bits()).collect()
        };
        for step in [0, 3] {
            let run = |m: &H3Transformer| {
                let mut blocks = Vec::new();
                let (v, a) = m
                    .forward(
                        step,
                        &video,
                        &audio,
                        &text,
                        &dl,
                        AttnMode::Dense,
                        Some(&mut |_, t| {
                            blocks.push(bits(t));
                            Ok(())
                        }),
                        None,
                        None,
                    )
                    .unwrap();
                (bits(&v), bits(&a), blocks)
            };
            let (want, got) = (run(&resident), run(&streamed));
            assert_eq!(want.2.len(), cfg.num_layers);
            assert_eq!(want, got, "step {step}");
        }
        let prompt =
            CudaTensor::from_vec(seeded(3 * cfg.text_dim, 0.7), vec![1, 3, cfg.text_dim]).unwrap();
        let refine = |r| {
            let refiner = H3TextRefiner::load_with_residency(&cfg, &map, &mut None, r).unwrap();
            bits(&refiner.forward(&prompt).unwrap())
        };
        assert_eq!(refine(Residency::Resident), refine(Residency::Streamed));
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
                None,
                None,
            )
            .unwrap();
        assert_eq!(
            seen,
            vec![
                ("block_0".to_string(), vec![1, 8, 12]),
                ("block_1".to_string(), vec![1, 8, 12])
            ]
        );

        // --- reference ---
        let h = cfg.hidden_size;
        let ts = schedule.row_timesteps(step).unwrap();
        let ts_index = layout.timestep_indices(&ts);
        let temb: Vec<Vec<f32>> = ts
            .timesteps
            .iter()
            .map(|&t| ref_temb(&cfg, &map, t))
            .collect();
        let mut x: Vec<Vec<f32>> = text.chunks(h).map(<[f32]>::to_vec).collect();
        x.extend(
            audio
                .chunks(cfg.audio_in_channels)
                .map(|r| lin(&map, "audio_proj_in", r, h, true)),
        );
        x.extend(
            video
                .chunks(cfg.video_patch_dim())
                .map(|r| lin(&map, "proj_in", r, h, true)),
        );
        let angles: Vec<[f32; 3]> = layout
            .position_ids
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        for b in 0..cfg.num_layers {
            let p = format!("transformer_blocks.{b}");
            // y.view(n_t * 3, 6H): row = timestep_index * 3 + tag.
            let tables: Vec<Vec<f32>> = temb
                .iter()
                .map(|e| {
                    lin(
                        &map,
                        &format!("{p}.adaln_proj.linear"),
                        &e.iter().map(|&v| silu(v)).collect::<Vec<_>>(),
                        18 * h,
                        true,
                    )
                })
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
                    let ss = lin(
                        &map,
                        "norm_out.linear",
                        &temb[ts_index[r]]
                            .iter()
                            .map(|&v| silu(v))
                            .collect::<Vec<_>>(),
                        2 * h,
                        true,
                    );
                    let n = rms(&x[r], &nw, 1e-5);
                    let y: Vec<f32> = (0..h).map(|c| n[c] * (1.0 + ss[h + c]) + ss[c]).collect();
                    lin(&map, proj, &y, width, true)
                })
                .collect()
        };
        let (wv, wa) = (
            out_rows(layout.video, "proj_out", cfg.video_patch_dim()),
            out_rows(layout.audio, "audio_proj_out", cfg.audio_in_channels),
        );
        assert_eq!(
            (gv.shape.clone(), ga.shape.clone()),
            (
                vec![nv, cfg.video_patch_dim()],
                vec![na, cfg.audio_in_channels]
            )
        );
        for (name, got, want) in [
            ("video", gv.host_cow().unwrap(), wv),
            ("audio", ga.host_cow().unwrap(), wa),
        ] {
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            assert!(scale > 1e-3, "{name}: a zero reference proves nothing");
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() < 1e-4 * scale.max(1.0),
                    "{name}[{i}]: {g} vs {w}"
                );
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
                cuda_tensor_shaped(&base, key, shape)
                    .unwrap()
                    .host_cow()
                    .unwrap()
                    .into_owned()
            })
        };
        let layout = H3PackedLayout::new(3, (2, 4, 4), 2, cfg.patch_size).unwrap();
        let (nv, na, nt) = (layout.video.len, layout.audio.len, layout.text.len);
        let video = CudaTensor::from_vec(
            seeded(nv * cfg.video_patch_dim(), 0.41),
            vec![nv, cfg.video_patch_dim()],
        )
        .unwrap();
        let audio = CudaTensor::from_vec(
            seeded(na * cfg.audio_in_channels, 0.23),
            vec![na, cfg.audio_in_channels],
        )
        .unwrap();
        let text = CudaTensor::from_vec(
            seeded(nt * cfg.hidden_size, 0.57),
            vec![1, nt, cfg.hidden_size],
        )
        .unwrap();
        let vsa = H3Vsa::new(
            &layout,
            cfg.num_attention_heads,
            cfg.attention_head_dim,
            H3VsaConfig {
                sparsity: 0.0,
                group: 1,
                tile_size: 64,
            },
        )
        .unwrap();
        let dl = DeviceLayout::new(&cfg, layout).unwrap();
        let run = |map: &WeightMap, mode: AttnMode<'_>| -> Vec<f32> {
            let model = H3Transformer::load(cfg.clone(), map, &schedule, true).unwrap();
            let (v, a) = model
                .forward(0, &video, &audio, &text, &dl, mode, None, None, None)
                .unwrap();
            v.host_cow()
                .unwrap()
                .iter()
                .chain(a.host_cow().unwrap().iter())
                .copied()
                .collect()
        };
        let dense = run(&zero_gate(), AttnMode::Dense);
        let sparse = run(&zero_gate(), AttnMode::Vsa(&vsa));
        assert!(
            // Up to upstream's bf16 combine rounding inside VSA.
            dense
                .iter()
                .zip(&sparse)
                .all(|(a, b)| (a - b).abs() < 5e-3 * (1.0 + a.abs())),
            "zero gate: VSA at sparsity 0 is dense"
        );
        let gated = run(&weights(), AttnMode::Vsa(&vsa));
        let dense_live = run(&weights(), AttnMode::Dense);
        assert!(
            gated
                .iter()
                .zip(&dense_live)
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "a trained gate changes the output"
        );
        // Without the gate weights loaded, VSA is refused rather than silently run gateless.
        let no_gate = H3Transformer::load(cfg.clone(), &weights(), &schedule, false).unwrap();
        assert!(no_gate
            .forward(
                0,
                &video,
                &audio,
                &text,
                &dl,
                AttnMode::Vsa(&vsa),
                None,
                None,
                None
            )
            .is_err());
    }

    #[test]
    fn the_table_holds_the_three_rows_a_t2av_forward_reads() {
        let (cfg, map) = (tiny_cfg(), weights());
        let schedule = H3JointSchedule::fasth3_8step();
        let table = AdaLnTable::precompute(&cfg, &map, &schedule).unwrap();
        let (h, step, block) = (cfg.hidden_size, 5, 1);
        let ts = H3RowTimesteps::new(
            schedule.video.timesteps[step],
            schedule.audio.timesteps[step],
        );
        assert_eq!(ts.adaln_rows(), [0, 1, 5]);
        for (tag, t) in [
            (TAG_VIDEO, schedule.video.timesteps[step]),
            (TAG_TEXT, schedule.video.timesteps[step]),
            (TAG_AUDIO, schedule.audio.timesteps[step]),
        ] {
            let s: Vec<f32> = ref_temb(&cfg, &map, t).into_iter().map(silu).collect();
            let y = lin(
                &map,
                &format!("transformer_blocks.{block}.adaln_proj.linear"),
                &s,
                18 * h,
                true,
            );
            let want = &y[usize::from(tag) * 6 * h..(usize::from(tag) + 1) * 6 * h];
            let got = table.block_slot(step, block, tag);
            for p in 0..6 {
                let plus = if p == SCALE_MSA || p == SCALE_MLP {
                    1.0
                } else {
                    0.0
                };
                for c in 0..h {
                    assert!(
                        (got[p * h + c] - (want[p * h + c] + plus)).abs() < 1e-5,
                        "tag {tag} param {p} ch {c}"
                    );
                }
            }
        }
        let want = ref_temb(&cfg, &map, schedule.audio.timesteps[step]);
        assert!(table.temb[2 * step + 1]
            .iter()
            .zip(&want)
            .all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn block_mods_narrows_the_uploaded_table() {
        let (cfg, map) = (tiny_cfg(), weights());
        let schedule = H3JointSchedule::fasth3_8step();
        let table = AdaLnTable::precompute(&cfg, &map, &schedule).unwrap();
        let layout = H3PackedLayout::new(2, (1, 2, 2), 1, cfg.patch_size).unwrap();
        let step = 3usize;
        let mods = BlockMods::upload(&table, step, 1, &layout).unwrap();
        assert!(!mods.segments.is_empty());
        for (range, e) in &mods.segments {
            let tag = layout.token_tags[range.start];
            let want = table.block_slot(step, 1, tag);
            let got = e.host_cow().unwrap();
            assert_eq!(e.shape, vec![1, ADALN_PARAMS, table.hidden]);
            assert_eq!(got.as_ref(), want);
        }
        let again = BlockMods::upload(&table, step, 0, &layout).unwrap();
        assert_eq!(again.segments.len(), mods.segments.len());
    }

    #[test]
    fn sol_layers_with_every_block_exact_are_dense_in_every_sink_order() {
        use fastvideo_models::h3::sol::H3SolAttnPolicy;
        let (cfg, map) = (tiny_cfg(), weights());
        let schedule = H3JointSchedule::fasth3_8step();
        let model = H3Transformer::load(cfg.clone(), &map, &schedule, false).unwrap();
        let layout = H3PackedLayout::new(2, (2, 2, 4), 1, cfg.patch_size).unwrap();
        let s = layout.sequence_length();
        let dl = DeviceLayout::new(&cfg, layout).unwrap();
        let n = CudaTensor::from_vec(
            seeded(s * cfg.hidden_size, 0.37),
            vec![1, s, cfg.hidden_size],
        )
        .unwrap();
        let attn = &model.blocks.skeleton(0).attn;
        let rope = Some((&dl.cos, &dl.sin));
        let dense = attn.forward(&n, rope, AttnMode::Dense).unwrap();
        let dense = dense.host_cow().unwrap().into_owned();
        for kind in [
            H3SolAttnPolicy::Engine,
            H3SolAttnPolicy::Rtx,
            H3SolAttnPolicy::Spark,
        ] {
            let policy = H3SolPolicy::from_layout(kind, &dl).unwrap();
            assert_eq!(policy.permutation.is_some(), kind == H3SolAttnPolicy::Spark);
            // tau -1000 routes every block exactly (the upstream correctness
            // gate), so any sink placement and permutation must give dense.
            let got = attn
                .forward(
                    &n,
                    rope,
                    AttnMode::SolLayer {
                        tau: -1000.0,
                        policy: &policy,
                    },
                )
                .unwrap();
            let got = got.host_cow().unwrap();
            let err = got
                .iter()
                .zip(&dense)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(err < 1e-4, "{kind:?}: max err {err}");
        }
    }

    #[test]
    fn spark_policy_gathers_rope_rows_and_round_trips() {
        use fastvideo_models::h3::sol::H3SolAttnPolicy;
        let cfg = tiny_cfg();
        let layout = H3PackedLayout::new(2, (2, 2, 4), 1, cfg.patch_size).unwrap();
        let s = layout.sequence_length();
        let dl = DeviceLayout::new(&cfg, layout).unwrap();
        let policy = H3SolPolicy::from_layout(H3SolAttnPolicy::Spark, &dl).unwrap();
        let p = policy.permutation.as_ref().unwrap();
        // [video 4 | text 2 + audio 2] with the sink as the 4-row suffix.
        assert_eq!(policy.sink, Some((4, 4)));
        assert_eq!(p.forward.indices(), &[4, 5, 6, 7, 0, 1, 2, 3]);
        let r = dl.cos.shape[1];
        let cos = dl.cos.host_cow().unwrap();
        let got = p.cos.host_cow().unwrap();
        for (dst, &src) in p.forward.indices().iter().enumerate() {
            assert_eq!(&got[dst * r..(dst + 1) * r], &cos[src * r..(src + 1) * r]);
        }
        let x = CudaTensor::from_vec(seeded(s * 3, 0.2), vec![1, s, 3]).unwrap();
        let back = p.inverse.apply(&p.forward.apply(&x).unwrap()).unwrap();
        assert_eq!(back.host_cow().unwrap(), x.host_cow().unwrap());
        // Engine and RTX sinks stay in packed order.
        let engine = H3SolPolicy::from_layout(H3SolAttnPolicy::Engine, &dl).unwrap();
        assert_eq!(engine.sink, Some((0, dl.layout.video.start)));
        let rtx = H3SolPolicy::from_layout(H3SolAttnPolicy::Rtx, &dl).unwrap();
        assert_eq!(rtx.sink, Some((0, dl.layout.text.len)));
    }

    #[test]
    fn i2v_vision_pads_read_the_ladder_and_keyframes_the_noise_aug_table() {
        // FL2VA text: "<Picture 1>: " + <vision_start> pads <vision_end> +
        // prompt. The vision rows carry the video tag but sit in the text span,
        // so `timestep_indices` gives them the video timestep, not the
        // condition one. Only the keyframe latent rows read KEYFRAME_NOISE_AUG.
        let (cfg, map) = (tiny_cfg(), weights());
        let schedule = H3JointSchedule::fasth3_8step();
        let table = AdaLnTable::precompute(&cfg, &map, &schedule).unwrap();
        let mut layout = H3PackedLayout::with_keyframes(
            6,
            (2, 2, 2),
            1,
            cfg.patch_size,
            &[fastvideo_models::h3::packing::KeyframeAnchor::First],
        )
        .unwrap();
        let text_tags = [
            TAG_TEXT, TAG_VIDEO, TAG_VIDEO, TAG_VIDEO, TAG_TEXT, TAG_VIDEO,
        ];
        layout.set_text_token_tags(&text_tags).unwrap();
        assert!(layout.cond.len > 0);
        // The trailing text-span video row must not merge with the cond run.
        assert_eq!(layout.token_tags[layout.cond.start], TAG_VIDEO);
        let ts = schedule.row_timesteps(2).unwrap();
        let indices = layout.timestep_indices(&ts);
        let (step, block) = (2usize, 0usize);
        let mods = BlockMods::upload(&table, step, block, &layout).unwrap();
        let mut covered = 0usize;
        for (range, e) in &mods.segments {
            let tag = layout.token_tags[range.start];
            let in_cond = range.start >= layout.cond.start && range.end() <= layout.cond.end();
            for row in range.start..range.end() {
                assert_eq!(layout.token_tags[row], tag);
                let want_index = if in_cond {
                    ts.condition_index
                } else if row >= layout.audio.start && row < layout.audio.end() {
                    ts.audio_index
                } else {
                    ts.video_index
                };
                assert_eq!(indices[row], want_index, "row {row}");
            }
            let want = if in_cond {
                table.keyframe_slot(block, tag)
            } else {
                assert!(row_uses_ladder(&layout, *range));
                table.block_slot(step, block, tag)
            };
            assert_eq!(e.host_cow().unwrap().as_ref(), want, "run {range:?}");
            covered += range.len;
        }
        assert_eq!(covered, layout.sequence_length());
        let pads = RowRange { start: 1, len: 3 };
        assert!(row_uses_ladder(&layout, pads));
        assert!(!row_uses_ladder(&layout, layout.cond));
    }

    fn quant_cfg() -> H3TransformerConfig {
        let mut c = tiny_cfg();
        c.num_attention_heads = 2;
        c.attention_head_dim = 32;
        c.hidden_size = 64;
        c.ffn_dim = 32;
        c
    }

    /// One `[1, 6, hidden]` AdaLN row (SCALE slots hold `1 + scale`).
    fn synthetic_rows(h: usize, rows: usize) -> AdaRows {
        let tab: Vec<f32> = (0..6 * h)
            .map(|i| {
                let p = i / h;
                let u = crate::wan::quant::bf16_round((i as f32 * 0.37).sin() * 0.1);
                if p == SCALE_MSA || p == SCALE_MLP {
                    1.0 + u
                } else {
                    u
                }
            })
            .collect();
        AdaRows {
            tab: CudaTensor::from_vec(tab, vec![1, 6, h]).unwrap(),
            hidden: h,
            idx: Arc::new(vec![0; rows]),
            #[cfg(feature = "cuda")]
            idx_dev: None,
        }
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum();
        let na: f64 = a.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let nb: f64 = b.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        dot / (na.sqrt() * nb.sqrt())
    }

    #[test]
    fn the_fused_bf16_forward_tracks_the_f32_forward() {
        let (cfg, map) = (tiny_cfg(), weights());
        let schedule = H3JointSchedule::fasth3_8step();
        let layout = H3PackedLayout::new(2, (2, 2, 4), 1, cfg.patch_size).unwrap();
        let (nv, na, nt) = (layout.video.len, layout.audio.len, layout.text.len);
        let dl = DeviceLayout::new(&cfg, layout).unwrap();
        let video = CudaTensor::from_vec(seeded(nv * cfg.video_patch_dim(), 0.41), vec![nv, cfg.video_patch_dim()]).unwrap();
        let audio = CudaTensor::from_vec(seeded(na * cfg.audio_in_channels, 0.23), vec![na, cfg.audio_in_channels]).unwrap();
        let text = CudaTensor::from_vec(seeded(nt * cfg.hidden_size, 0.57), vec![1, nt, cfg.hidden_size]).unwrap();
        let run = |on: bool| {
            crate::wan::tensor::with_bf16_act(on, || {
                let model = H3Transformer::load(cfg.clone(), &map, &schedule, false).unwrap();
                let (v, a) = model
                    .forward(3, &video, &audio, &text, &dl, AttnMode::Dense, None, None, None)
                    .unwrap();
                let out: Vec<f32> = v.host_cow().unwrap().iter().chain(a.host_cow().unwrap().iter()).copied().collect();
                // The proj_out island keeps the head f32.
                (out, v.is_bf16())
            })
        };
        let (f32_out, _) = run(false);
        let (bf_out, head_bf16) = run(true);
        assert!(!head_bf16, "proj_out is an f32 island (_keep_in_fp32_modules)");
        let cos = cosine(&bf_out, &f32_out);
        assert!(cos > 0.999, "bf16 forward cosine {cos}");
    }

    #[test]
    fn the_reference_fp8_recipes_track_the_bf16_block() {
        let (cfg, map) = (quant_cfg(), weights());
        let (rows, h) = (16usize, cfg.hidden_size);
        let x = CudaTensor::from_vec(seeded(rows * h, 0.31), vec![1, rows, h])
            .unwrap()
            .quantize_bf16()
            .unwrap();
        let ada = synthetic_rows(h, rows);
        let run = |quant: Option<QuantKind>| {
            crate::wan::tensor::with_bf16_act(true, || {
                let b = Block::load_quant(&map, "transformer_blocks.0", &cfg, false, &mut None, quant)
                    .unwrap();
                if let Some(kind) = quant {
                    assert_eq!(b.attn.qkvg.quant_kind(), Some(kind));
                    assert_eq!(b.ff.ff_out.quant_kind(), Some(kind));
                }
                let y = b
                    .forward_fused(
                        &x,
                        &ada,
                        0,
                        None,
                        AttnMode::Dense,
                        1e-5,
                        [SHIFT_MSA, SCALE_MSA, GATE_MSA, SHIFT_MLP, SCALE_MLP, GATE_MLP],
                    )
                    .unwrap();
                assert!(y.is_bf16());
                y.host_cow().unwrap().into_owned()
            })
        };
        let base = run(None);
        for kind in [QuantKind::W8A8, QuantKind::Mxfp8] {
            let q = run(Some(kind));
            let cos = cosine(&q, &base);
            assert!(cos > 0.995, "{kind:?} block cosine {cos}");
            assert!(q != base, "{kind:?} must actually quantize");
        }
    }

    #[test]
    fn a_vsa_gate_stays_bf16_under_the_recipes() {
        let (cfg, map) = (quant_cfg(), weights());
        let b = Block::load_quant(&map, "transformer_blocks.0", &cfg, true, &mut None, Some(QuantKind::W8A8))
            .unwrap();
        assert_eq!(b.attn.qkvg.quant_kind(), Some(QuantKind::W8A8));
        assert!(!b.attn.accepts_mx(AttnMode::Dense), "W8A8 quantizes its input itself");
        let b = Block::load_quant(&map, "transformer_blocks.0", &cfg, true, &mut None, Some(QuantKind::Mxfp8))
            .unwrap();
        assert!(!b.attn.accepts_mx(AttnMode::Dense), "a bf16 gate section needs the bf16 input");
        let b = Block::load_quant(&map, "transformer_blocks.0", &cfg, false, &mut None, Some(QuantKind::Mxfp8))
            .unwrap();
        assert!(b.attn.accepts_mx(AttnMode::Dense));
    }

    #[test]
    fn bf16_table_values_round_the_projection_and_keep_one_plus_scale_in_f32() {
        let v = 1.0 + 0.123_456_7f32;
        let t = bf16_table_value(v, true);
        assert_eq!(t, crate::wan::quant::bf16_round(v - 1.0) + 1.0);
        assert_eq!(bf16_table_value(t, true), t, "idempotent");
        assert_eq!(bf16_table_value(0.3, false), crate::wan::quant::bf16_round(0.3));
    }
}
