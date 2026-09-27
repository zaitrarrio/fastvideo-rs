//! The 44k latent VAE, decoder half (`ext/autoencoder/vae.py`, `Decoder1D`):
//! `[B, 40, N]` latents to a normalized 128-band log-mel `[B, 128, 2N]`, then
//! `unnormalize` with `DATA_MEAN_128D` / `DATA_STD_128D`.
//!
//! Magnitude-preserving layers (EDM2): every `MPConv1D` weight is
//! force-normalized once at load exactly as `remove_weight_norm` does it
//! (`w / (1e-4 + ||w_o|| / sqrt(in k)) / sqrt(in k)` per output row, in f32);
//! `mp_silu(x) = silu(x) / 0.596`; `mp_sum(a, b, 0.3) = (0.7 a + 0.3 b) /
//! sqrt(0.58)`. The pixel norm (`normalize(x, dim=1)`, `x / (1e-4 + ||x|| /
//! sqrt(C))`) runs as an RMS norm over channels without the 1e-4 (a relative
//! 1e-4 / rms difference; the activations are unit-scale by construction).
//! Runs in f32 (upstream runs this module in bf16).

use fastvideo_models::mmaudio::{MmAudioVaeConfig, DATA_MEAN_128D, DATA_STD_128D};

use super::layers::{host_values, msg, pinned, Conv1d};
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::WeightMap;

const MP_SILU: f32 = 1.0 / 0.596;

/// `MPConv1D` with its weight normalized (and `gain` folded in).
fn mp_conv(map: &WeightMap, key: &str, cin: usize, cout: usize, k: usize, gain: f32) -> Result<Conv1d> {
    let w = host_values(map, &format!("{key}.weight"), &[cout, cin, k])?;
    let fan = cin * k;
    let alpha = (1.0 / fan as f64).sqrt();
    let mut out = Vec::with_capacity(w.len());
    for row in w.chunks_exact(fan) {
        let norm = row.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt() as f32;
        let denom = 1e-4f32 + norm * alpha as f32;
        let scale = 1.0 / (fan as f32).sqrt();
        out.extend(row.iter().map(|&x| (x / denom) * scale * gain));
    }
    Conv1d::from_host(out, None, cout, cin, k)
}

fn mp_silu(x: &CudaTensor) -> Result<CudaTensor> {
    x.silu().try_mul_scalar(MP_SILU)
}

fn mp_sum(a: &CudaTensor, b: &CudaTensor, t: f32) -> Result<CudaTensor> {
    let s = 1.0 / ((1.0 - t) * (1.0 - t) + t * t).sqrt();
    CudaTensor::lincomb(&[((1.0 - t) * s, a), (t * s, b)])
}

/// `normalize(x, dim=channel)` of `[B, C, L]` (see the module note).
fn pixel_norm(x: &CudaTensor, ones: &CudaTensor) -> Result<CudaTensor> {
    x.transpose(1, 2)?.rms_norm(ones, 0.0)?.transpose(1, 2)
}

struct ResBlock {
    conv1: Conv1d,
    conv2: Conv1d,
    nin: Option<Conv1d>,
    ones: CudaTensor,
}

impl ResBlock {
    fn load(map: &WeightMap, p: &str, cin: usize, cout: usize) -> Result<Self> {
        Ok(Self {
            conv1: mp_conv(map, &format!("{p}.conv1"), cin, cout, 3, 1.0)?,
            conv2: mp_conv(map, &format!("{p}.conv2"), cout, cout, 3, 1.0)?,
            nin: if cin != cout {
                Some(mp_conv(map, &format!("{p}.nin_shortcut"), cin, cout, 1, 1.0)?)
            } else {
                None
            },
            ones: pinned(vec![1.0; cin], vec![cin])?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let x = pixel_norm(x, &self.ones)?;
        let h = self.conv1.forward(&mp_silu(&x)?)?;
        let h = self.conv2.forward(&mp_silu(&h)?)?;
        let x = match &self.nin {
            Some(n) => n.forward(&x)?,
            None => x,
        };
        mp_sum(&x, &h, 0.3)
    }
}

/// `AttnBlock1D` (one head over all channels).
struct AttnBlock {
    qkv: Conv1d,
    proj: Conv1d,
    ones: CudaTensor,
    c: usize,
}

impl AttnBlock {
    fn load(map: &WeightMap, p: &str, c: usize) -> Result<Self> {
        Ok(Self {
            qkv: mp_conv(map, &format!("{p}.qkv"), c, 3 * c, 1, 1.0)?,
            proj: mp_conv(map, &format!("{p}.proj_out"), c, c, 1, 1.0)?,
            ones: pinned(vec![1.0; c], vec![c])?,
            c,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let [b, _, l] = x.shape[..] else {
            return Err(msg(format!("vae attn {:?}", x.shape)));
        };
        let c = self.c;
        // qkv rows are (c j): reshape(B, 1, C, 3, L); normalize over C.
        let y = self.qkv.forward(x)?.reshape(vec![b, c, 3, l])?;
        let y = y.permute(&[0, 2, 3, 1])?; // [B, 3, L, C]
        let y = y.rms_norm(&self.ones, 0.0)?;
        let part = |j: usize| -> Result<CudaTensor> {
            y.narrow(1, j, 1) // [B, 1, L, C]
        };
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);
        let h = crate::wan::nn::scaled_dot_product_attention(&q, &k, &v, None)?; // [B, 1, L, C]
        let h = h.reshape(vec![b, l, c])?.transpose(1, 2)?;
        mp_sum(x, &self.proj.forward(&h)?, 0.3)
    }
}

pub struct MmAudioVaeDecoder {
    cfg: MmAudioVaeConfig,
    conv_in: Conv1d,
    mid1: ResBlock,
    mid_attn: AttnBlock,
    mid2: ResBlock,
    /// `up[level]`: blocks, optional upsample conv.
    up: Vec<(Vec<ResBlock>, Option<Conv1d>)>,
    conv_out: Conv1d,
    mean: CudaTensor,
    std: CudaTensor,
}

impl MmAudioVaeDecoder {
    /// Reads `decoder.*` of the VAE checkpoint.
    pub fn load(cfg: MmAudioVaeConfig, map: &WeightMap) -> Result<Self> {
        let levels = cfg.ch_mult.len();
        let dim = cfg.hidden_dim;
        let top = dim * cfg.ch_mult[levels - 1];
        let conv_in = mp_conv(map, "decoder.conv_in", cfg.embed_dim, top, 3, 1.0)?;
        let mid1 = ResBlock::load(map, "decoder.mid.block_1", top, top)?;
        let mid_attn = AttnBlock::load(map, "decoder.mid.attn_1", top)?;
        let mid2 = ResBlock::load(map, "decoder.mid.block_2", top, top)?;
        let mut up: Vec<(Vec<ResBlock>, Option<Conv1d>)> = (0..levels).map(|_| (Vec::new(), None)).collect();
        let mut block_in = top;
        for level in (0..levels).rev() {
            let block_out = dim * cfg.ch_mult[level];
            let mut blocks = Vec::new();
            for i in 0..=cfg.num_res_blocks {
                blocks.push(ResBlock::load(
                    map,
                    &format!("decoder.up.{level}.block.{i}"),
                    block_in,
                    block_out,
                )?);
                block_in = block_out;
            }
            let upsample = if cfg.up_levels.contains(&level) {
                Some(mp_conv(map, &format!("decoder.up.{level}.upsample.conv"), block_in, block_in, 3, 1.0)?)
            } else {
                None
            };
            up[level] = (blocks, upsample);
        }
        let gain = host_values(map, "decoder.learnable_gain", &[])?
            .first()
            .copied()
            .unwrap_or(0.0)
            + 1.0;
        let conv_out = mp_conv(map, "decoder.conv_out", block_in, cfg.data_dim, 3, gain)?;
        let (mean, std) = if cfg.data_dim == 128 {
            (DATA_MEAN_128D.to_vec(), DATA_STD_128D.to_vec())
        } else {
            (vec![0.0; cfg.data_dim], vec![1.0; cfg.data_dim])
        };
        Ok(Self {
            mean: pinned(mean, vec![1, cfg.data_dim, 1])?,
            std: pinned(std, vec![1, cfg.data_dim, 1])?,
            conv_in,
            mid1,
            mid_attn,
            mid2,
            up,
            conv_out,
            cfg,
        })
    }

    /// `[B, 40, N]` (unnormalized DiT latents, channels first) to the mel
    /// `[B, 128, 2N]`.
    pub fn decode(&self, z: &CudaTensor) -> Result<CudaTensor> {
        let clip = self.cfg.clip_act;
        let h = self.conv_in.forward(z)?;
        let h = self.mid1.forward(&h)?;
        let h = self.mid_attn.forward(&h)?;
        let mut h = self.mid2.forward(&h)?.clamp(-clip, clip);
        for level in (0..self.up.len()).rev() {
            let (blocks, upsample) = &self.up[level];
            for b in blocks {
                h = b.forward(&h)?.clamp(-clip, clip);
            }
            if let Some(conv) = upsample {
                // nearest-exact x2 = each sample twice.
                let [bsz, c, l] = h.shape[..] else {
                    return Err(msg("vae upsample"));
                };
                let rows = h.reshape(vec![bsz * c * l, 1])?;
                let rep: Vec<usize> = (0..bsz * c * l).flat_map(|i| [i, i]).collect();
                let up = rows.index_select_rows(&rep)?.reshape(vec![bsz, c, 2 * l])?;
                h = conv.forward(&up)?;
            }
        }
        let h = self.conv_out.forward(&mp_silu(&h)?)?;
        h.mul(&self.std)?.add(&self.mean)
    }
}
