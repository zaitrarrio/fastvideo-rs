//! BigVGAN v2 generator (`ext/bigvgan_v2/bigvgan.py`), the 44k MMAudio
//! vocoder `nvidia/bigvgan_v2_44khz_128band_512x`: a 128-band log-mel
//! `[B, 128, T]` to a waveform `[B, 512 T]` in `[-1, 1]`.
//!
//! Same building blocks as the H3 audio decoder (`h3::audio_vae`): weight
//! norm folded at load (`weight_g`/`weight_v`, or torch's
//! `parametrizations.weight.original0/1`), alias-free SnakeBeta (Kaiser-sinc
//! x2 up, activation, x2 down, 12 taps, replicate padding), AMPBlock1 with
//! dilations (1, 3, 5), the parallel kernels (3, 7, 11) averaged, no final
//! tanh (a clamp) and no final bias. Runs in f32.

use fastvideo_models::mmaudio::BigVganConfig;

use super::layers::{host_values, msg, pinned};
use crate::wan::ops::PadMode;
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::WeightMap;

const FILTER_TAPS: usize = 12;
const RATIO: usize = 2;
const UP_PAD: usize = FILTER_TAPS / RATIO - 1; // 5
const UP_CROP_LEFT: usize = UP_PAD * RATIO + (FILTER_TAPS - RATIO) / 2; // 15
const UP_CROP_RIGHT: usize = UP_PAD * RATIO + (FILTER_TAPS - RATIO + 1) / 2; // 15
const DOWN_PAD_LEFT: usize = FILTER_TAPS / 2 - 1; // 5
const DOWN_PAD_RIGHT: usize = FILTER_TAPS / 2; // 6

/// `kaiser_sinc_filter1d(cutoff=0.25, half_width=0.3, kernel_size=12)` — the
/// buffer every `Activation1d` holds (used when the checkpoint has none).
pub fn kaiser_sinc_12() -> Vec<f32> {
    let (cutoff, half_width, k) = (0.25f64, 0.3f64, FILTER_TAPS);
    let half = k / 2;
    let delta_f = 4.0 * half_width;
    let a = 2.285 * (half as f64 - 1.0) * std::f64::consts::PI * delta_f + 7.95;
    let beta = if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    };
    // torch.kaiser_window(periodic=False): I0(beta sqrt(1 - (2n/(N-1) - 1)^2)) / I0(beta)
    let i0 = |x: f64| -> f64 {
        let mut sum = 1.0;
        let mut term = 1.0;
        for m in 1..50 {
            term *= (x / 2.0) * (x / 2.0) / (m as f64 * m as f64);
            sum += term;
        }
        sum
    };
    let taps: Vec<f64> = (0..k)
        .map(|n| {
            let r = 2.0 * n as f64 / (k as f64 - 1.0) - 1.0;
            let w = i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0(beta);
            let t = n as f64 - half as f64 + 0.5;
            let x = 2.0 * cutoff * t;
            let sinc = if x == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };
            2.0 * cutoff * w * sinc
        })
        .collect();
    let s: f64 = taps.iter().sum();
    taps.iter().map(|v| (v / s) as f32).collect()
}

/// `g * v / ||v||` per dim-0 slice.
fn fold(g: &[f32], v: &[f32], rows: usize) -> Result<Vec<f32>> {
    if g.len() != rows || v.len() % rows != 0 {
        return Err(msg("weight norm shapes"));
    }
    let width = v.len() / rows;
    let mut out = Vec::with_capacity(v.len());
    for (row, &gain) in v.chunks_exact(width).zip(g) {
        let norm = row.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
        let s = (f64::from(gain) / norm) as f32;
        out.extend(row.iter().map(|&x| x * s));
    }
    Ok(out)
}

struct Conv {
    weight: CudaTensor,
    bias: Option<CudaTensor>,
}

impl Conv {
    fn load(map: &WeightMap, prefix: &str, shape: [usize; 3], bias: Option<usize>) -> Result<Self> {
        let (gk, vk) = if map.contains(&format!("{prefix}.weight_g")) {
            (format!("{prefix}.weight_g"), format!("{prefix}.weight_v"))
        } else if map.contains(&format!("{prefix}.parametrizations.weight.original0")) {
            (
                format!("{prefix}.parametrizations.weight.original0"),
                format!("{prefix}.parametrizations.weight.original1"),
            )
        } else {
            (String::new(), String::new())
        };
        let w = if gk.is_empty() {
            host_values(map, &format!("{prefix}.weight"), &shape)?
        } else {
            let g = host_values(map, &gk, &[shape[0], 1, 1])?;
            let v = host_values(map, &vk, &shape)?;
            fold(&g, &v, shape[0])?
        };
        Ok(Self {
            weight: pinned(w, shape.to_vec())?,
            bias: match bias {
                Some(n) => Some(pinned(host_values(map, &format!("{prefix}.bias"), &[n])?, vec![n])?),
                None => None,
            },
        })
    }
}

fn depthwise(taps: &[f32], c: usize) -> Result<CudaTensor> {
    pinned((0..c).flat_map(|_| taps.iter().copied()).collect(), vec![c, 1, FILTER_TAPS])
}

struct AliasFreeSnake {
    alpha: CudaTensor,
    inv_beta: CudaTensor,
    up: CudaTensor,
    down: CudaTensor,
}

impl AliasFreeSnake {
    fn load(map: &WeightMap, prefix: &str, c: usize, taps: &[f32]) -> Result<Self> {
        let alpha = host_values(map, &format!("{prefix}.act.alpha"), &[c])?;
        let beta = host_values(map, &format!("{prefix}.act.beta"), &[c])?;
        let read = |k: &str| -> Result<Vec<f32>> {
            if map.contains(k) {
                host_values(map, k, &[1, 1, FILTER_TAPS])
            } else {
                Ok(taps.to_vec())
            }
        };
        let up = read(&format!("{prefix}.upsample.filter"))?;
        let down = read(&format!("{prefix}.downsample.lowpass.filter"))?;
        Ok(Self {
            alpha: pinned(alpha.iter().map(|a| a.exp()).collect(), vec![c])?,
            inv_beta: pinned(beta.iter().map(|b| 1.0 / (b.exp() + 1e-9)).collect(), vec![c])?,
            up: depthwise(&up, c)?,
            down: depthwise(&down, c)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let (c, len) = (x.shape[1], x.shape[2]);
        let up = x
            .pad(2, UP_PAD, UP_PAD, PadMode::Replicate)?
            .conv_transpose1d(&self.up, None, 0, RATIO, 1, c, 0)?
            .try_mul_scalar(RATIO as f32)?;
        let up = up.narrow(2, UP_CROP_LEFT, up.shape[2] - UP_CROP_LEFT - UP_CROP_RIGHT)?;
        if up.shape[2] != RATIO * len {
            return Err(msg(format!("alias-free upsample: {} from {len}", up.shape[2])));
        }
        up.snake_beta(&self.alpha, &self.inv_beta)?
            .pad(2, DOWN_PAD_LEFT, DOWN_PAD_RIGHT, PadMode::Replicate)?
            .conv1d(&self.down, None, 0, RATIO, 1, c)
    }
}

struct AmpBlock {
    stages: Vec<(AliasFreeSnake, Conv, usize, AliasFreeSnake, Conv)>,
    kernel: usize,
}

impl AmpBlock {
    fn load(map: &WeightMap, p: &str, c: usize, k: usize, dil: &[usize], taps: &[f32]) -> Result<Self> {
        let stages = dil
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                Ok((
                    AliasFreeSnake::load(map, &format!("{p}.activations.{}", 2 * i), c, taps)?,
                    Conv::load(map, &format!("{p}.convs1.{i}"), [c, c, k], Some(c))?,
                    d,
                    AliasFreeSnake::load(map, &format!("{p}.activations.{}", 2 * i + 1), c, taps)?,
                    Conv::load(map, &format!("{p}.convs2.{i}"), [c, c, k], Some(c))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { stages, kernel: k })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let mut x = x.clone();
        for (a1, c1, d, a2, c2) in &self.stages {
            let pad1 = (self.kernel * d - d) / 2;
            let r = a1.forward(&x)?.conv1d(&c1.weight, c1.bias.as_ref(), pad1, 1, *d, 1)?;
            let r = a2
                .forward(&r)?
                .conv1d(&c2.weight, c2.bias.as_ref(), (self.kernel - 1) / 2, 1, 1, 1)?;
            x = x.add(&r)?;
        }
        Ok(x)
    }
}

pub struct BigVgan {
    cfg: BigVganConfig,
    conv_pre: Conv,
    ups: Vec<Conv>,
    resblocks: Vec<AmpBlock>,
    act_post: AliasFreeSnake,
    conv_post: Conv,
}

impl BigVgan {
    pub fn load(cfg: BigVganConfig, map: &WeightMap) -> Result<Self> {
        let taps = kaiser_sinc_12();
        let ch0 = cfg.upsample_initial_channel;
        let conv_pre = Conv::load(map, "conv_pre", [ch0, cfg.num_mels, 7], Some(ch0))?;
        let mut ups = Vec::new();
        let mut resblocks = Vec::new();
        let nk = cfg.resblock_kernel_sizes.len();
        for (i, (&u, &k)) in cfg.upsample_rates.iter().zip(&cfg.upsample_kernel_sizes).enumerate() {
            let (cin, cout) = (ch0 >> i, ch0 >> (i + 1));
            ups.push(Conv::load(map, &format!("ups.{i}.0"), [cin, cout, k], Some(cout))?);
            let _ = u;
            for (j, (&rk, dil)) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(&cfg.resblock_dilation_sizes)
                .enumerate()
            {
                resblocks.push(AmpBlock::load(
                    map,
                    &format!("resblocks.{}", i * nk + j),
                    cout,
                    rk,
                    dil,
                    &taps,
                )?);
            }
        }
        let last = ch0 >> cfg.upsample_rates.len();
        let bias = cfg.use_bias_at_final.then_some(1);
        Ok(Self {
            act_post: AliasFreeSnake::load(map, "activation_post", last, &taps)?,
            conv_post: Conv::load(map, "conv_post", [1, last, 7], bias)?,
            conv_pre,
            ups,
            resblocks,
            cfg,
        })
    }

    /// `[B, num_mels, T]` to `[B, hop T]`.
    pub fn forward(&self, mel: &CudaTensor) -> Result<CudaTensor> {
        let [b, _, t] = mel.shape[..] else {
            return Err(msg(format!("vocoder input {:?}", mel.shape)));
        };
        let mut x = mel.conv1d(&self.conv_pre.weight, self.conv_pre.bias.as_ref(), 3, 1, 1, 1)?;
        let nk = self.cfg.resblock_kernel_sizes.len();
        for (i, up) in self.ups.iter().enumerate() {
            let (u, k) = (self.cfg.upsample_rates[i], self.cfg.upsample_kernel_sizes[i]);
            x = x.conv_transpose1d(&up.weight, up.bias.as_ref(), (k - u) / 2, u, 1, 1, 0)?;
            let mut sum: Option<CudaTensor> = None;
            for blk in &self.resblocks[i * nk..(i + 1) * nk] {
                let y = blk.forward(&x)?;
                sum = Some(match sum {
                    Some(s) => s.add(&y)?,
                    None => y,
                });
            }
            x = sum.ok_or_else(|| msg("no resblocks"))?.try_mul_scalar(1.0 / nk as f32)?;
        }
        let x = self.act_post.forward(&x)?;
        let x = x.conv1d(&self.conv_post.weight, self.conv_post.bias.as_ref(), 3, 1, 1, 1)?;
        let x = if self.cfg.use_tanh_at_final {
            return Err(msg("tanh final not ported"));
        } else {
            x.clamp(-1.0, 1.0)
        };
        let n = t * self.cfg.hop();
        if x.shape != [b, 1, n] {
            return Err(msg(format!("vocoder produced {:?}, want [{b}, 1, {n}]", x.shape)));
        }
        x.reshape(vec![b, n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kaiser_filter_is_symmetric_and_normalized() {
        let f = kaiser_sinc_12();
        let s: f32 = f.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        for i in 0..6 {
            assert!((f[i] - f[11 - i]).abs() < 1e-7);
        }
    }
}
