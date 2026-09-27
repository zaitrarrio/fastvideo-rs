//! Wan 2.2 VAE (TI2V-5B): Diffusers `AutoencoderKLWan` with `is_residual`,
//! `patch_size = 2`, `z_dim = 48`, encoder base 160, decoder base 256.
//!
//! What differs from the Wan 2.1 graph in [`super::vae`] (whose causal conv,
//! residual block, attention, resample and feat cache it reuses):
//!
//! * `patchify` / `unpatchify` (2×2 space-to-channel) around the network:
//!   the encoder reads 12 channels, the decoder writes 12 — 16× spatial in all.
//! * Encoder down blocks are `WanResidualDownBlock`s: the resnets and the
//!   downsampler, plus an `AvgDown3D` shortcut of the block input
//!   (space/time-to-channel, then the mean over channel groups).
//! * Decoder up blocks are `WanResidualUpBlock`s: `num_res_blocks + 1`
//!   resnets and an upsampler whose conv keeps the width, plus a `DupUp3D`
//!   shortcut (channel repeat, then channel-to-space/time), which on the first
//!   chunk drops its `factor_t - 1` leading frames exactly as the upsampler's
//!   first pass skips temporal doubling.
//! * The encoder's downsamplers pad right/bottom only (`ZeroPad2d((0,1,0,1))`)
//!   before a stride-2 3×3 conv, and a `downsample3d` runs its stride-2 time
//!   conv *after* the spatial conv, on the previous chunk's last frame
//!   followed by this chunk (the first chunk only primes the cache).
//!
//! Encode runs Diffusers' chunking (frame 0 alone, then 4 frames per pass,
//! with the feat cache); decode reuses [`super::vae`]'s streaming loop.

use fastvideo_models::wan::WanVaeConfig;

use super::tensor::{CudaTensor, Result, TensorError};
use super::vae::{
    conv_cached, gamma, last_frames, pinned, rms_silu_video, AttentionBlock, CacheSlot,
    CausalConv3d, FeatCache, Resample, ResampleMode, ResidualBlock,
};
use super::weights::{self, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

fn dims5(x: &CudaTensor, what: &str) -> Result<[usize; 5]> {
    match x.shape[..] {
        [1, c, t, h, w] => Ok([1, c, t, h, w]),
        _ => Err(msg(format!(
            "{what}: expected [1, C, T, H, W], got {:?}",
            x.shape
        ))),
    }
}

/// Diffusers `patchify`: `[1, C, F, H, W]` → `[1, C·p², F, H/p, W/p]`, channel
/// `(c·p + pw)·p + ph` (the width offset before the height offset).
pub fn patchify(x: &CudaTensor, p: usize) -> Result<CudaTensor> {
    if p <= 1 {
        return Ok(x.clone());
    }
    let [_, c, f, h, w] = dims5(x, "patchify")?;
    if h % p != 0 || w % p != 0 {
        return Err(msg(format!("patchify: {h}x{w} not divisible by {p}")));
    }
    let (hp, wp) = (h / p, w / p);
    // [C, F, H', ph, W', pw] → [C, pw, ph, F, H', W']
    x.reshape(vec![c, f, hp, p, wp, p])?
        .permute(&[0, 5, 3, 1, 2, 4])?
        .reshape(vec![1, c * p * p, f, hp, wp])
}

/// Diffusers `unpatchify`, the inverse of [`patchify`].
pub fn unpatchify(x: &CudaTensor, p: usize) -> Result<CudaTensor> {
    if p <= 1 {
        return Ok(x.clone());
    }
    let [_, cp, f, h, w] = dims5(x, "unpatchify")?;
    if cp % (p * p) != 0 {
        return Err(msg(format!("unpatchify: {cp} channels, patch {p}")));
    }
    let c = cp / (p * p);
    // [C, pw, ph, F, H, W] → [C, F, H, ph, W, pw]
    x.reshape(vec![c, p, p, f, h, w])?
        .permute(&[0, 3, 4, 2, 5, 1])?
        .reshape(vec![1, c, f, h * p, w * p])
}

/// Diffusers `AvgDown3D(in_c, out_c, factor_t, factor_s)`: zero frames in
/// front up to a multiple of `ft`, fold `ft × fs × fs` into channels
/// (`((c·ft + t)·fs + h)·fs + w`), then average groups of consecutive
/// channels down to `out_c`.
pub fn avg_down(x: &CudaTensor, out_c: usize, ft: usize, fs: usize) -> Result<CudaTensor> {
    let [_, c, t, h, w] = dims5(x, "avg_down")?;
    let pad_t = (ft - t % ft) % ft;
    let x = x.pad_zeros(2, pad_t, 0)?;
    let t = t + pad_t;
    let f = ft * fs * fs;
    if (c * f) % out_c != 0 || h % fs != 0 || w % fs != 0 {
        return Err(msg(format!(
            "avg_down: {c}x{f} channels to {out_c}, {h}x{w} by {fs}"
        )));
    }
    let g = c * f / out_c;
    if f == 1 && g == 1 {
        return Ok(x);
    }
    if f % g != 0 {
        return Err(msg(format!("avg_down: group {g} does not divide {f}")));
    }
    let (tp, hp, wp) = (t / ft, h / fs, w / fs);
    // [C·T', ft, H', fs, W', fs] → [C·T', ft, fs, fs, H', W'] = [C, T', F, S]
    let y = x
        .reshape(vec![c * tp, ft, hp, fs, wp, fs])?
        .permute(&[0, 1, 3, 5, 2, 4])?;
    // [C, T', F/G, G, S] → [G, C, F/G, T', S] = [G, O·T'·S]
    let n = out_c * tp * hp * wp;
    let y = y
        .reshape(vec![c, tp, f / g, g, hp * wp])?
        .permute(&[3, 0, 2, 1, 4])?
        .reshape(vec![g, n])?;
    let out = if g == 1 {
        y
    } else {
        let parts: Vec<CudaTensor> = (0..g).map(|i| y.narrow(0, i, 1)).collect::<Result<_>>()?;
        let scale = 1.0 / g as f32;
        let terms: Vec<(f32, &CudaTensor)> = parts.iter().map(|p| (scale, p)).collect();
        CudaTensor::lincomb(&terms)?
    };
    out.reshape(vec![1, out_c, tp, hp, wp])
}

/// Diffusers `DupUp3D(in_c, out_c, factor_t, factor_s)`: repeat each channel
/// `out_c·ft·fs²/in_c` times, unfold `ft × fs × fs` out of the channels into
/// time and space; on the first chunk drop the leading `ft - 1` frames.
pub fn dup_up(
    x: &CudaTensor,
    out_c: usize,
    ft: usize,
    fs: usize,
    first_chunk: bool,
) -> Result<CudaTensor> {
    let [_, c, t, h, w] = dims5(x, "dup_up")?;
    let f = ft * fs * fs;
    if (out_c * f) % c != 0 {
        return Err(msg(format!("dup_up: {c} channels to {out_c}x{f}")));
    }
    let r = out_c * f / c;
    let n = t * h * w;
    let x1 = x.reshape(vec![c, 1, n])?;
    let rep = if r > 1 {
        CudaTensor::cat(&vec![&x1; r], 1)?
    } else {
        x1
    };
    // [O, F, T, S] → [O, T, S, F] = [O·T, H, W, ft, fs, fs]
    //   → [O·T, ft, H, fs, W, fs] = [O, T·ft, H·fs, W·fs]
    let y = rep
        .reshape(vec![out_c, f, t, h * w])?
        .permute(&[0, 2, 3, 1])?
        .reshape(vec![out_c * t, h, w, ft, fs, fs])?
        .permute(&[0, 3, 1, 4, 2, 5])?
        .reshape(vec![1, out_c, t * ft, h * fs, w * fs])?;
    if first_chunk && ft > 1 {
        y.narrow(2, ft - 1, t * ft - (ft - 1))
    } else {
        Ok(y)
    }
}

/// Encoder `WanResample` in `downsample2d` / `downsample3d` mode.
#[derive(Debug, Clone)]
pub(super) struct Downsample {
    spatial_w: CudaTensor,
    spatial_b: CudaTensor,
    time_conv: Option<CausalConv3d>,
}

impl Downsample {
    pub(super) fn zeros(dim: usize, temporal: bool) -> Self {
        Self {
            spatial_w: pinned(CudaTensor::zeros(&[dim, dim, 3, 3])),
            spatial_b: pinned(CudaTensor::zeros(&[dim])),
            time_conv: temporal
                .then(|| CausalConv3d::zeros(dim, dim, [3, 1, 1], [2, 1, 1], [0, 0, 0])),
        }
    }

    pub(super) fn load(map: &WeightMap, prefix: &str, dim: usize, temporal: bool) -> Result<Self> {
        let key = |n: &str| weights::join_key(prefix, n);
        Ok(Self {
            spatial_w: pinned(weights::cuda_tensor_shaped(
                map,
                &key("resample.1.weight"),
                &[dim, dim, 3, 3],
            )?),
            spatial_b: pinned(weights::cuda_tensor_shaped(
                map,
                &key("resample.1.bias"),
                &[dim],
            )?),
            time_conv: if temporal {
                Some(CausalConv3d::load(
                    map,
                    &key("time_conv"),
                    dim,
                    dim,
                    [3, 1, 1],
                    [2, 1, 1],
                    [0, 0, 0],
                )?)
            } else {
                None
            },
        })
    }

    /// `ZeroPad2d((0, 1, 0, 1))` + stride-2 conv per frame; then, for
    /// `downsample3d` with a cache, the stride-2 time conv over the previous
    /// pass's last frame and this pass (the first pass stores its frames and
    /// skips the time conv, Diffusers' `feat_cache[idx] is None` branch).
    pub(super) fn forward(
        &self,
        xs: &CudaTensor,
        cache: Option<&mut FeatCache>,
    ) -> Result<CudaTensor> {
        let [b, c, t, h, w] = dims5(xs, "downsample")?;
        let x2 = xs
            .permute(&[0, 2, 1, 3, 4])?
            .reshape(vec![b * t, c, h, w])?
            .pad_zeros(2, 0, 1)?
            .pad_zeros(3, 0, 1)?;
        let y = x2.conv2d(&self.spatial_w, Some(&self.spatial_b), 0, 2)?;
        let (oc, hh, ww) = (y.shape[1], y.shape[2], y.shape[3]);
        let y = y
            .reshape(vec![b, t, oc, hh, ww])?
            .permute(&[0, 2, 1, 3, 4])?;
        let Some(tc) = &self.time_conv else {
            return Ok(y);
        };
        let Some(cache) = cache else {
            // Uncached (a whole clip in one pass): Diffusers' encoder always
            // runs with the cache, so this is only the shape-level fallback.
            return tc.forward(&y);
        };
        let i = cache.reserve();
        match cache.slots[i].clone() {
            CacheSlot::Tensor(prev) => {
                let last = prev.narrow(2, prev.dim(2)? - 1, 1)?;
                let keep = last_frames(&y, 1)?;
                let out = tc.forward(&CudaTensor::cat(&[&last, &y], 2)?)?;
                cache.slots[i] = CacheSlot::Tensor(keep);
                Ok(out)
            }
            _ => {
                cache.slots[i] = CacheSlot::Tensor(y.clone());
                Ok(y)
            }
        }
    }
}

/// `WanResidualDownBlock`.
#[derive(Debug, Clone)]
struct DownBlock {
    resnets: Vec<ResidualBlock>,
    down: Option<Downsample>,
    out_c: usize,
    ft: usize,
    fs: usize,
}

impl DownBlock {
    fn forward(&self, xs: &CudaTensor, cache: &mut FeatCache) -> Result<CudaTensor> {
        let mut x = xs.clone();
        for r in &self.resnets {
            x = r.forward(&x, Some(cache))?;
        }
        if let Some(d) = &self.down {
            x = d.forward(&x, Some(cache))?;
        }
        x.add(&avg_down(xs, self.out_c, self.ft, self.fs)?)
    }
}

/// Layout of one residual down / up block, shared by `zeros` and `load`.
struct BlockPlan {
    in_c: usize,
    out_c: usize,
    temporal: bool,
    resample: bool,
}

fn encoder_plan(cfg: &WanVaeConfig) -> Vec<BlockPlan> {
    let dims: Vec<usize> = std::iter::once(1)
        .chain(cfg.dim_mult.iter().copied())
        .map(|u| cfg.base_dim * u)
        .collect();
    // Encoder `temperal_downsample` is the decoder's flag list reversed.
    let down: Vec<bool> = cfg.temporal_upsample.iter().rev().copied().collect();
    let last = cfg.dim_mult.len() - 1;
    (0..cfg.dim_mult.len())
        .map(|i| BlockPlan {
            in_c: dims[i],
            out_c: dims[i + 1],
            temporal: i != last && down.get(i).copied().unwrap_or(false),
            resample: i != last,
        })
        .collect()
}

fn decoder_plan(cfg: &WanVaeConfig) -> Vec<BlockPlan> {
    let top = *cfg.dim_mult.last().expect("dim_mult");
    let dims: Vec<usize> = std::iter::once(top)
        .chain(cfg.dim_mult.iter().rev().copied())
        .map(|u| cfg.decoder_base_dim * u)
        .collect();
    let last = cfg.dim_mult.len() - 1;
    (0..cfg.dim_mult.len())
        .map(|i| BlockPlan {
            in_c: dims[i],
            out_c: dims[i + 1],
            temporal: i != last && cfg.temporal_upsample.get(i).copied().unwrap_or(false),
            resample: i != last,
        })
        .collect()
}

/// `WanEncoder3d` with `is_residual = true`, plus `quant_conv`.
#[derive(Debug, Clone)]
pub(super) struct Encoder22 {
    patch: usize,
    z_dim: usize,
    conv_in: CausalConv3d,
    blocks: Vec<DownBlock>,
    mid_res0: ResidualBlock,
    mid_attn: AttentionBlock,
    mid_res1: ResidualBlock,
    norm_out: CudaTensor,
    conv_out: CausalConv3d,
    quant: CausalConv3d,
}

impl Encoder22 {
    pub(super) fn zeros(cfg: &WanVaeConfig) -> Self {
        let plan = encoder_plan(cfg);
        let top = plan.last().map(|p| p.out_c).unwrap_or(cfg.base_dim);
        let blocks = plan
            .iter()
            .map(|p| {
                let mut cur = p.in_c;
                let resnets = (0..cfg.num_res_blocks)
                    .map(|_| {
                        let r = ResidualBlock::zeros(cur, p.out_c);
                        cur = p.out_c;
                        r
                    })
                    .collect();
                DownBlock {
                    resnets,
                    down: p.resample.then(|| Downsample::zeros(p.out_c, p.temporal)),
                    out_c: p.out_c,
                    ft: if p.temporal { 2 } else { 1 },
                    fs: if p.resample { 2 } else { 1 },
                }
            })
            .collect();
        let z2 = cfg.z_dim * 2;
        Self {
            patch: cfg.patch_size.max(1),
            z_dim: cfg.z_dim,
            conv_in: CausalConv3d::zeros(
                cfg.io_channels(),
                cfg.base_dim,
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
            ),
            blocks,
            mid_res0: ResidualBlock::zeros(top, top),
            mid_attn: AttentionBlock::zeros(top),
            mid_res1: ResidualBlock::zeros(top, top),
            norm_out: pinned(CudaTensor::ones(&[top])),
            conv_out: CausalConv3d::zeros(top, z2, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            quant: CausalConv3d::zeros(z2, z2, [1, 1, 1], [1, 1, 1], [0, 0, 0]),
        }
    }

    pub(super) fn load(map: &WeightMap, cfg: &WanVaeConfig) -> Result<Self> {
        let plan = encoder_plan(cfg);
        let top = plan.last().map(|p| p.out_c).unwrap_or(cfg.base_dim);
        let mut blocks = Vec::with_capacity(plan.len());
        for (i, p) in plan.iter().enumerate() {
            let pre = format!("encoder.down_blocks.{i}");
            let mut cur = p.in_c;
            let mut resnets = Vec::new();
            for j in 0..cfg.num_res_blocks {
                resnets.push(ResidualBlock::load(
                    map,
                    &format!("{pre}.resnets.{j}"),
                    cur,
                    p.out_c,
                )?);
                cur = p.out_c;
            }
            let down = if p.resample {
                Some(Downsample::load(
                    map,
                    &format!("{pre}.downsampler"),
                    p.out_c,
                    p.temporal,
                )?)
            } else {
                None
            };
            blocks.push(DownBlock {
                resnets,
                down,
                out_c: p.out_c,
                ft: if p.temporal { 2 } else { 1 },
                fs: if p.resample { 2 } else { 1 },
            });
        }
        let z2 = cfg.z_dim * 2;
        let mid = "encoder.mid_block";
        Ok(Self {
            patch: cfg.patch_size.max(1),
            z_dim: cfg.z_dim,
            conv_in: CausalConv3d::load(
                map,
                "encoder.conv_in",
                cfg.io_channels(),
                cfg.base_dim,
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
            )?,
            blocks,
            mid_res0: ResidualBlock::load(map, &format!("{mid}.resnets.0"), top, top)?,
            mid_attn: AttentionBlock::load(map, &format!("{mid}.attentions.0"), top)?,
            mid_res1: ResidualBlock::load(map, &format!("{mid}.resnets.1"), top, top)?,
            norm_out: gamma(map, "encoder.norm_out.gamma", &[top, 1, 1, 1])?,
            conv_out: CausalConv3d::load(
                map,
                "encoder.conv_out",
                top,
                z2,
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
            )?,
            quant: CausalConv3d::load(map, "quant_conv", z2, z2, [1, 1, 1], [1, 1, 1], [0, 0, 0])?,
        })
    }

    fn forward_chunk(&self, xs: &CudaTensor, cache: &mut FeatCache) -> Result<CudaTensor> {
        let mut x = conv_cached(&self.conv_in, xs, Some(cache))?;
        for b in &self.blocks {
            x = b.forward(&x, cache)?;
        }
        x = self.mid_res0.forward(&x, Some(cache))?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward(&x, Some(cache))?;
        x = rms_silu_video(&x, &self.norm_out)?;
        conv_cached(&self.conv_out, &x, Some(cache))
    }

    /// `[1, 3, F, H, W]` in `[-1, 1]` → the posterior mean `[1, z, 1 + (F-1)/4,
    /// H/16, W/16]` (Diffusers `_encode` + `latent_dist.mode()`), f32.
    pub(super) fn encode(&self, video: &CudaTensor) -> Result<CudaTensor> {
        let x = patchify(video, self.patch)?;
        let t = x.dim(2)?;
        if t == 0 {
            return Err(msg("encode: no frames"));
        }
        let iters = 1 + (t - 1) / 4;
        let mut cache = FeatCache::new();
        let mut outs = Vec::with_capacity(iters);
        for i in 0..iters {
            cache.begin_pass();
            let chunk = if i == 0 {
                x.narrow(2, 0, 1)?
            } else {
                x.narrow(2, 1 + 4 * (i - 1), 4)?
            };
            outs.push(self.forward_chunk(&chunk, &mut cache)?);
        }
        let refs: Vec<&CudaTensor> = outs.iter().collect();
        let enc = self.quant.forward(&CudaTensor::cat(&refs, 2)?)?;
        enc.narrow(1, 0, self.z_dim)
    }
}

/// `WanResidualUpBlock`.
#[derive(Debug, Clone)]
struct UpBlock {
    resnets: Vec<ResidualBlock>,
    up: Option<Resample>,
    /// `DupUp3D(in, out, ft, 2)` when the block upsamples.
    shortcut: Option<(usize, usize)>,
}

impl UpBlock {
    fn forward(
        &self,
        xs: &CudaTensor,
        mut cache: Option<&mut FeatCache>,
        first_chunk: bool,
    ) -> Result<CudaTensor> {
        let mut x = xs.clone();
        for r in &self.resnets {
            x = r.forward(&x, cache.as_deref_mut())?;
        }
        if let Some(up) = &self.up {
            x = up.forward(&x, cache.as_deref_mut())?;
        }
        match self.shortcut {
            Some((out_c, ft)) => x.add(&dup_up(xs, out_c, ft, 2, first_chunk)?),
            None => Ok(x),
        }
    }
}

/// `WanDecoder3d` with `is_residual = true` (12 output channels: the
/// caller unpatchifies and clamps).
#[derive(Debug, Clone)]
pub(super) struct Decoder22 {
    conv_in: CausalConv3d,
    mid_res0: ResidualBlock,
    mid_attn: AttentionBlock,
    mid_res1: ResidualBlock,
    blocks: Vec<UpBlock>,
    norm_out: CudaTensor,
    conv_out: CausalConv3d,
}

impl Decoder22 {
    pub(super) fn zeros(cfg: &WanVaeConfig) -> Self {
        let plan = decoder_plan(cfg);
        let top = plan[0].in_c;
        let out_dim = plan.last().map(|p| p.out_c).unwrap_or(top);
        let blocks = plan
            .iter()
            .map(|p| {
                let mut cur = p.in_c;
                let resnets = (0..=cfg.num_res_blocks)
                    .map(|_| {
                        let r = ResidualBlock::zeros(cur, p.out_c);
                        cur = p.out_c;
                        r
                    })
                    .collect();
                let mode = if p.temporal {
                    ResampleMode::Upsample3d
                } else {
                    ResampleMode::Upsample2d
                };
                UpBlock {
                    resnets,
                    up: p.resample.then(|| Resample::zeros(p.out_c, p.out_c, mode)),
                    shortcut: p
                        .resample
                        .then_some((p.out_c, if p.temporal { 2 } else { 1 })),
                }
            })
            .collect();
        Self {
            conv_in: CausalConv3d::zeros(cfg.z_dim, top, [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            mid_res0: ResidualBlock::zeros(top, top),
            mid_attn: AttentionBlock::zeros(top),
            mid_res1: ResidualBlock::zeros(top, top),
            blocks,
            norm_out: pinned(CudaTensor::ones(&[out_dim])),
            conv_out: CausalConv3d::zeros(
                out_dim,
                cfg.io_channels(),
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
            ),
        }
    }

    pub(super) fn load(map: &WeightMap, cfg: &WanVaeConfig) -> Result<Self> {
        let plan = decoder_plan(cfg);
        let top = plan[0].in_c;
        let out_dim = plan.last().map(|p| p.out_c).unwrap_or(top);
        let mut blocks = Vec::with_capacity(plan.len());
        for (i, p) in plan.iter().enumerate() {
            let pre = format!("decoder.up_blocks.{i}");
            let mut cur = p.in_c;
            let mut resnets = Vec::new();
            for j in 0..=cfg.num_res_blocks {
                resnets.push(ResidualBlock::load(
                    map,
                    &format!("{pre}.resnets.{j}"),
                    cur,
                    p.out_c,
                )?);
                cur = p.out_c;
            }
            let mode = if p.temporal {
                ResampleMode::Upsample3d
            } else {
                ResampleMode::Upsample2d
            };
            let up = if p.resample {
                Some(Resample::load(
                    map,
                    &format!("{pre}.upsampler"),
                    p.out_c,
                    p.out_c,
                    mode,
                )?)
            } else {
                None
            };
            blocks.push(UpBlock {
                resnets,
                up,
                shortcut: p
                    .resample
                    .then_some((p.out_c, if p.temporal { 2 } else { 1 })),
            });
        }
        let mid = "decoder.mid_block";
        Ok(Self {
            conv_in: CausalConv3d::load(
                map,
                "decoder.conv_in",
                cfg.z_dim,
                top,
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
            )?,
            mid_res0: ResidualBlock::load(map, &format!("{mid}.resnets.0"), top, top)?,
            mid_attn: AttentionBlock::load(map, &format!("{mid}.attentions.0"), top)?,
            mid_res1: ResidualBlock::load(map, &format!("{mid}.resnets.1"), top, top)?,
            blocks,
            norm_out: gamma(map, "decoder.norm_out.gamma", &[out_dim, 1, 1, 1])?,
            conv_out: CausalConv3d::load(
                map,
                "decoder.conv_out",
                out_dim,
                cfg.io_channels(),
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
            )?,
        })
    }

    /// One decode pass over latent frames `zs` (post `post_quant_conv`):
    /// `[1, 12, frames, H·8, W·8]`, before `unpatchify` and the clamp.
    pub(super) fn forward(
        &self,
        zs: &CudaTensor,
        mut cache: Option<&mut FeatCache>,
        first_chunk: bool,
    ) -> Result<CudaTensor> {
        let mut x = conv_cached(&self.conv_in, zs, cache.as_deref_mut())?;
        x = self.mid_res0.forward(&x, cache.as_deref_mut())?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward(&x, cache.as_deref_mut())?;
        for b in &self.blocks {
            x = b.forward(&x, cache.as_deref_mut(), first_chunk)?;
        }
        x = rms_silu_video(&x, &self.norm_out)?;
        conv_cached(&self.conv_out, &x, cache.as_deref_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(shape: &[usize]) -> CudaTensor {
        let n: usize = shape.iter().product();
        CudaTensor::from_vec((0..n).map(|i| i as f32).collect(), shape.to_vec()).unwrap()
    }

    #[test]
    fn patchify_matches_diffusers_layout() {
        // [1, 1, 1, 2, 4], p 2: channel (c·2 + pw)·2 + ph.
        let x = ramp(&[1, 1, 1, 2, 4]);
        let y = patchify(&x, 2).unwrap();
        assert_eq!(y.shape, vec![1, 4, 1, 1, 2]);
        let v = y.host_cow().unwrap().into_owned();
        // x[h, w] = 4h + w. Channel k = 2·pw + ph, position j: x[ph, 2j + pw].
        let want: Vec<f32> = (0..4)
            .flat_map(|k| {
                let (pw, ph) = (k / 2, k % 2);
                (0..2).map(move |j| (4 * ph + 2 * j + pw) as f32)
            })
            .collect();
        assert_eq!(v, want);
        let back = unpatchify(&y, 2).unwrap();
        assert_eq!(back.shape, x.shape);
        assert_eq!(
            back.host_cow().unwrap().into_owned(),
            x.host_cow().unwrap().into_owned()
        );
    }

    #[test]
    fn avg_down_means_folded_groups() {
        // 2 channels, 2 frames, 2x2 → ft 2, fs 2: F = 8, 16 folded channels,
        // out 4 → groups of 4 consecutive folded channels.
        let x = ramp(&[1, 2, 2, 2, 2]);
        let y = avg_down(&x, 4, 2, 2).unwrap();
        assert_eq!(y.shape, vec![1, 4, 1, 1, 1]);
        let xv = x.host_cow().unwrap().into_owned();
        let at = |c: usize, t: usize, h: usize, w: usize| xv[((c * 2 + t) * 2 + h) * 2 + w];
        let mut folded = Vec::new();
        for c in 0..2 {
            for t in 0..2 {
                for h in 0..2 {
                    for w in 0..2 {
                        folded.push(at(c, t, h, w));
                    }
                }
            }
        }
        let want: Vec<f32> = folded
            .chunks(4)
            .map(|g| g.iter().sum::<f32>() / 4.0)
            .collect();
        let got = y.host_cow().unwrap().into_owned();
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5, "{got:?} vs {want:?}");
        }
        // One frame with ft 2: a zero frame goes in front.
        let one = ramp(&[1, 2, 1, 2, 2]);
        let y1 = avg_down(&one, 4, 2, 2).unwrap();
        assert_eq!(y1.shape, vec![1, 4, 1, 1, 1]);
        let v1 = y1.host_cow().unwrap().into_owned();
        // Groups 0 and 2 are the zero frame of channels 0 and 1.
        assert_eq!(v1[0], 0.0);
        assert_eq!(v1[2], 0.0);
        assert!((v1[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn dup_up_unfolds_repeated_channels() {
        // 2 channels → out 1 with ft 2, fs 2: r = 4, F = 8.
        let x = ramp(&[1, 2, 1, 1, 1]);
        let y = dup_up(&x, 1, 2, 2, false).unwrap();
        assert_eq!(y.shape, vec![1, 1, 2, 2, 2]);
        // Folded channel f = (a·2 + p)·2 + q came from input channel f / 4:
        // frame a = 0 is channel 0, frame 1 is channel 1.
        let v = y.host_cow().unwrap().into_owned();
        assert_eq!(v, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let first = dup_up(&x, 1, 2, 2, true).unwrap();
        assert_eq!(first.shape, vec![1, 1, 1, 2, 2]);
        assert_eq!(first.host_cow().unwrap().into_owned(), vec![1.0; 4]);
    }

    /// The Wan 2.2 geometry with narrow widths: 16× spatial, `4T - 3`
    /// frames, 12-channel patchified I/O, whatever the decode chunking.
    #[test]
    fn wan22_decode_and_encode_geometry() {
        let cfg = WanVaeConfig {
            base_dim: 4,
            decoder_base_dim: 8,
            z_dim: 6,
            latents_mean: vec![0.0; 6],
            latents_std: vec![1.0; 6],
            load_encoder: true,
            ..WanVaeConfig::wan_2_2()
        };
        let vae = super::super::vae::AutoencoderKlWan::zeros(cfg);
        let z = CudaTensor::zeros(&[1, 6, 3, 2, 3]);
        for chunk in ["1", "2"] {
            std::env::set_var("FASTVIDEO_VAE_CHUNK", chunk);
            let out = vae.decode(&z).unwrap();
            assert_eq!(out.shape, vec![1, 3, 9, 32, 48], "chunk {chunk}");
        }
        std::env::remove_var("FASTVIDEO_VAE_CHUNK");
        let video = CudaTensor::zeros(&[1, 3, 9, 32, 48]);
        let lat = vae.encode_video(&video).unwrap();
        assert_eq!(lat.shape, vec![1, 6, 3, 2, 3]);
        let image = CudaTensor::zeros(&[1, 3, 1, 32, 48]);
        assert_eq!(vae.encode_video(&image).unwrap().shape, vec![1, 6, 1, 2, 3]);
    }
}
