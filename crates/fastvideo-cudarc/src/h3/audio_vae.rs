//! MiniMax-H3 audio VAE, decoder half: 40 Hz latents to a 32 kHz waveform
//! (`AutoencoderKLMiniMaxH3Audio.decode`, a BigVGAN-v2 stack).
//!
//! The model is mono. Stereo is two latent streams pushed through the same
//! weights as a batch of two, with no op that mixes them, so everything here
//! is `[B, C, L]` and `B = 2` is the caller's business.
//!
//! What makes this decoder unlike the convolutional VAEs already in the tree:
//!
//! * **Weight norm on disk.** Every conv but `dec_in_proj` is stored as
//!   `weight_g`, `weight_v`; the effective weight is `g * v / ||v||` with the
//!   norm over every dim but 0. For a transposed conv dim 0 is the *input*
//!   channel. Folded once on the host at load — there is no reason to carry
//!   the parametrization to the device.
//! * **Alias-free activations.** Each SnakeBeta runs at twice the sample rate:
//!   a depthwise transposed Kaiser-sinc upsample (replicate pad 5/5, gain 2,
//!   crop 15/15), the activation, a depthwise strided low-pass (replicate pad
//!   5/6). The 12-tap filter is a buffer of the checkpoint; all 128 copies are
//!   equal, so one is loaded and expanded per channel count.
//! * **Log-space SnakeBeta.** `x + sin^2(exp(alpha) x) / (exp(beta) + 1e-9)`;
//!   the exponentials are taken at load.
//! * **float32 only.** The reference pins this model to float32 because bf16
//!   costs ~20 dB; there is no `Linear` here, so nothing drops to bf16.
//!
//! See docs/ports/h3.md, section d.

use fastvideo_models::h3::config::H3AudioVaeConfig;

use crate::wan::ops::PadMode;
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// Taps of the anti-aliasing filter, and the geometry that follows from a
/// 12-tap kernel at ratio 2 (`MiniMaxH3AudioUpSample1d`, `...LowPassFilter1d`).
const FILTER_TAPS: usize = 12;
const RATIO: usize = 2;
const UP_PAD: usize = FILTER_TAPS / RATIO - 1; // 5
const UP_CROP_LEFT: usize = UP_PAD * RATIO + (FILTER_TAPS - RATIO) / 2; // 15
const UP_CROP_RIGHT: usize = UP_PAD * RATIO + (FILTER_TAPS - RATIO + 1) / 2; // 15
const DOWN_PAD_LEFT: usize = FILTER_TAPS / 2 - 1; // 5
const DOWN_PAD_RIGHT: usize = FILTER_TAPS / 2; // 6

fn host_values(map: &WeightMap, key: &str, shape: &[usize]) -> Result<Vec<f32>> {
    Ok(cuda_tensor_shaped(map, key, shape)?.host_cow()?.into_owned())
}

fn pinned(data: Vec<f32>, shape: Vec<usize>) -> Result<CudaTensor> {
    let mut t = CudaTensor::from_vec(data, shape)?;
    t.pin_device()?;
    Ok(t)
}

/// `g * v / ||v||`, the norm over each dim-0 slice of `v` (`torch._weight_norm`
/// with `dim = 0`). `rows` is `v.shape[0]`: output channels of a `Conv1d`,
/// *input* channels of a `ConvTranspose1d`.
pub fn fold_weight_norm(g: &[f32], v: &[f32], rows: usize) -> Result<Vec<f32>> {
    if rows == 0 || g.len() != rows || v.len() % rows != 0 {
        return Err(msg(format!("weight norm: {} gains for {rows} rows of {} values", g.len(), v.len())));
    }
    let width = v.len() / rows;
    let mut out = Vec::with_capacity(v.len());
    for (row, &gain) in v.chunks_exact(width).zip(g) {
        let norm = row.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
        let scale = (f64::from(gain) / norm) as f32;
        out.extend(row.iter().map(|&x| x * scale));
    }
    Ok(out)
}

/// A weight-normed convolution, folded. `shape` is the weight's `[d0, d1, K]`.
struct Conv {
    weight: CudaTensor,
    bias: Option<CudaTensor>,
}

impl Conv {
    fn load_weight_normed(map: &WeightMap, prefix: &str, shape: [usize; 3], bias_len: Option<usize>) -> Result<Self> {
        let g = host_values(map, &format!("{prefix}.weight_g"), &[shape[0], 1, 1])?;
        let v = host_values(map, &format!("{prefix}.weight_v"), &shape)?;
        let weight = pinned(fold_weight_norm(&g, &v, shape[0])?, shape.to_vec())?;
        let bias = match bias_len {
            Some(n) => Some(pinned(host_values(map, &format!("{prefix}.bias"), &[n])?, vec![n])?),
            None => None,
        };
        Ok(Self { weight, bias })
    }
}

/// The shared 12-tap low-pass, expanded to `[C, 1, 12]` for a depthwise conv.
fn depthwise_filter(taps: &[f32], channels: usize) -> Result<CudaTensor> {
    let data: Vec<f32> = (0..channels).flat_map(|_| taps.iter().copied()).collect();
    pinned(data, vec![channels, 1, FILTER_TAPS])
}

/// `Activation1d(SnakeBeta)`: upsample 2x, activate, downsample 2x.
struct AliasFreeSnake {
    /// `exp(alpha)`, `[C]`.
    alpha: CudaTensor,
    /// `1 / (exp(beta) + 1e-9)`, `[C]`.
    inv_beta: CudaTensor,
    filter: CudaTensor,
}

impl AliasFreeSnake {
    fn load(map: &WeightMap, prefix: &str, channels: usize, taps: &[f32]) -> Result<Self> {
        let alpha = host_values(map, &format!("{prefix}.act.alpha"), &[channels])?;
        let beta = host_values(map, &format!("{prefix}.act.beta"), &[channels])?;
        Ok(Self {
            alpha: pinned(alpha.iter().map(|a| a.exp()).collect(), vec![channels])?,
            inv_beta: pinned(beta.iter().map(|b| 1.0 / (b.exp() + 1e-9f32)).collect(), vec![channels])?,
            filter: depthwise_filter(taps, channels)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let (channels, len) = (x.shape[1], x.shape[2]);
        // Up: length L -> (L + 10 - 1) * 2 + 12 = 2L + 30 -> crop to 2L.
        let up = x
            .pad(2, UP_PAD, UP_PAD, PadMode::Replicate)?
            .conv_transpose1d(&self.filter, None, 0, RATIO, 1, channels, 0)?
            .try_mul_scalar(RATIO as f32)?;
        let up = up.narrow(2, UP_CROP_LEFT, up.shape[2] - UP_CROP_LEFT - UP_CROP_RIGHT)?;
        if up.shape[2] != RATIO * len {
            return Err(msg(format!("alias-free upsample produced {} samples from {len}", up.shape[2])));
        }
        let act = up.snake_beta(&self.alpha, &self.inv_beta)?;
        // Down: 2L + 11 -> (2L + 11 - 12) / 2 + 1 = L.
        act.pad(2, DOWN_PAD_LEFT, DOWN_PAD_RIGHT, PadMode::Replicate)?.conv1d(&self.filter, None, 0, RATIO, 1, channels)
    }
}

/// BigVGAN `AMPBlock1`: per dilation, `x += conv2(act2(conv1(act1(x))))`.
struct AmpBlock {
    /// `(act1, conv1, dilation, act2, conv2)`.
    stages: Vec<(AliasFreeSnake, Conv, usize, AliasFreeSnake, Conv)>,
    kernel: usize,
}

impl AmpBlock {
    fn load(map: &WeightMap, prefix: &str, channels: usize, kernel: usize, dilations: &[usize], taps: &[f32]) -> Result<Self> {
        let shape = [channels, channels, kernel];
        let stages = dilations
            .iter()
            .enumerate()
            .map(|(d, &dilation)| {
                // `activations[0::2]` feed `convs1`, `activations[1::2]` feed `convs2`.
                Ok((
                    AliasFreeSnake::load(map, &format!("{prefix}.activations.{}", 2 * d), channels, taps)?,
                    Conv::load_weight_normed(map, &format!("{prefix}.convs1.{d}"), shape, Some(channels))?,
                    dilation,
                    AliasFreeSnake::load(map, &format!("{prefix}.activations.{}", 2 * d + 1), channels, taps)?,
                    Conv::load_weight_normed(map, &format!("{prefix}.convs2.{d}"), shape, Some(channels))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { stages, kernel })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let mut x = x.clone();
        for (act1, conv1, dilation, act2, conv2) in &self.stages {
            let pad1 = (self.kernel * dilation - dilation) / 2;
            let r = act1.forward(&x)?.conv1d(&conv1.weight, conv1.bias.as_ref(), pad1, 1, *dilation, 1)?;
            let r = act2.forward(&r)?.conv1d(&conv2.weight, conv2.bias.as_ref(), (self.kernel - 1) / 2, 1, 1, 1)?;
            x = x.add(&r)?;
        }
        Ok(x)
    }
}

pub struct H3AudioDecoder {
    cfg: H3AudioVaeConfig,
    /// `latents_std`, `latents_mean` as `[1, C, 1]`.
    std: CudaTensor,
    mean: CudaTensor,
    dec_in_proj: Conv,
    conv_pre: Conv,
    ups: Vec<Conv>,
    /// `resblocks[3 i + j]`: stage `i`, kernel `j`.
    resblocks: Vec<AmpBlock>,
    activation_post: AliasFreeSnake,
    conv_post: Conv,
    /// `[1]` tensor holding the number of parallel AMP blocks (the `/ 3`).
    num_kernels: CudaTensor,
}

impl H3AudioDecoder {
    /// Reads `dec_in_proj.*` and `decoder.*` only; the encoder half of the
    /// checkpoint is never touched.
    pub fn load(cfg: H3AudioVaeConfig, map: &WeightMap) -> Result<Self> {
        let stages = cfg.decoder_rates.len();
        if cfg.decoder_dim >> stages == 0 {
            return Err(msg(format!("audio decoder: {} channels cannot halve {stages} times", cfg.decoder_dim)));
        }
        // All 128 filter buffers hold the same Kaiser-sinc kernel; read one.
        let taps = host_values(map, "decoder.activation_post.upsample.filter", &[1, 1, FILTER_TAPS])?;
        let (lc, ld) = (cfg.latent_channels, cfg.latent_dim);
        let dec_in_proj = Conv {
            weight: pinned(host_values(map, "dec_in_proj.weight", &[ld, lc, 1])?, vec![ld, lc, 1])?,
            bias: Some(pinned(host_values(map, "dec_in_proj.bias", &[ld])?, vec![ld])?),
        };
        let conv_pre = Conv::load_weight_normed(map, "decoder.conv_pre", [cfg.decoder_dim, ld, 7], Some(cfg.decoder_dim))?;
        let mut ups = Vec::with_capacity(stages);
        let mut resblocks = Vec::with_capacity(stages * cfg.resblock_kernel_sizes.len());
        for i in 0..stages {
            let (cin, cout) = cfg.upsampler_channels(i);
            // ConvTranspose1d weight is [in, out, K]: weight norm runs over dim 0 = in.
            ups.push(Conv::load_weight_normed(map, &format!("decoder.ups.{i}.0"), [cin, cout, cfg.decoder_kernel_sizes[i]], Some(cout))?);
            for (j, (&kernel, dilations)) in cfg.resblock_kernel_sizes.iter().zip(&cfg.resblock_dilation_sizes).enumerate() {
                let index = i * cfg.resblock_kernel_sizes.len() + j;
                resblocks.push(AmpBlock::load(map, &format!("decoder.resblocks.{index}"), cout, kernel, dilations, &taps)?);
            }
        }
        let last = cfg.decoder_dim >> stages;
        let to_col = |v: &[f64]| pinned(v.iter().map(|&x| x as f32).collect(), vec![1, v.len(), 1]);
        Ok(Self {
            std: to_col(&cfg.latents_std[..lc.min(cfg.latents_std.len())])?,
            mean: to_col(&cfg.latents_mean[..lc.min(cfg.latents_mean.len())])?,
            dec_in_proj,
            conv_pre,
            ups,
            resblocks,
            activation_post: AliasFreeSnake::load(map, "decoder.activation_post", last, &taps)?,
            conv_post: Conv::load_weight_normed(map, "decoder.conv_post", [1, last, 7], None)?,
            num_kernels: pinned(vec![cfg.resblock_kernel_sizes.len() as f32], vec![1])?,
            cfg,
        })
    }

    pub fn config(&self) -> &H3AudioVaeConfig {
        &self.cfg
    }

    /// DiT-space latents `[B, 32, L]` to a waveform `[B, 800 L]` in `[-1, 1]`.
    /// Applies `latents_std` / `latents_mean` itself.
    pub fn decode(&self, latents: &CudaTensor) -> Result<CudaTensor> {
        let [batch, channels, len] = latents.shape[..] else {
            return Err(msg(format!("audio decode expects [B, C, L], got {:?}", latents.shape)));
        };
        if channels != self.cfg.latent_channels || len == 0 {
            return Err(msg(format!("audio decode: latents {:?} for {} latent channels", latents.shape, self.cfg.latent_channels)));
        }
        let z = latents.mul(&self.std)?.add(&self.mean)?;
        let x = z.conv1d(&self.dec_in_proj.weight, self.dec_in_proj.bias.as_ref(), 0, 1, 1, 1)?;
        let mut x = x.conv1d(&self.conv_pre.weight, self.conv_pre.bias.as_ref(), 3, 1, 1, 1)?;
        let kernels = self.cfg.resblock_kernel_sizes.len();
        for (i, up) in self.ups.iter().enumerate() {
            x = x.conv_transpose1d(&up.weight, up.bias.as_ref(), self.cfg.upsampler_padding(i), self.cfg.decoder_rates[i], 1, 1, 0)?;
            // The parallel AMP blocks all read the same upsampled signal.
            let mut sum: Option<CudaTensor> = None;
            for block in &self.resblocks[i * kernels..(i + 1) * kernels] {
                let y = block.forward(&x)?;
                sum = Some(match sum {
                    Some(s) => s.add(&y)?,
                    None => y,
                });
            }
            x = sum.ok_or_else(|| msg("audio decoder without resblocks"))?.div(&self.num_kernels)?;
        }
        let x = self.activation_post.forward(&x)?;
        let x = x.conv1d(&self.conv_post.weight, None, 3, 1, 1, 1)?.clamp(-1.0, 1.0);
        let samples = len * self.cfg.hop_length();
        if x.shape != [batch, 1, samples] {
            return Err(msg(format!("audio decode produced {:?}, expected [{batch}, 1, {samples}]", x.shape)));
        }
        x.reshape(vec![batch, samples])
    }

    /// The DiT's audio rows `[channels * Na, 32]` (channel-major: all of the
    /// left channel's latents, then the right's) to a waveform `[channels, 800 Na]`.
    pub fn decode_rows(&self, rows: &CudaTensor, audio_channels: usize) -> Result<CudaTensor> {
        let [n, width] = rows.shape[..] else {
            return Err(msg(format!("audio rows must be [rows, {}], got {:?}", self.cfg.latent_channels, rows.shape)));
        };
        if audio_channels == 0 || n % audio_channels != 0 || width != self.cfg.latent_channels {
            return Err(msg(format!("audio rows {:?} over {audio_channels} channels", rows.shape)));
        }
        let latents = rows.reshape(vec![audio_channels, n / audio_channels, width])?.permute(&[0, 2, 1])?;
        self.decode(&latents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> H3AudioVaeConfig {
        let mut cfg = H3AudioVaeConfig::fasth3_8step();
        cfg.latent_dim = 6;
        cfg.latent_channels = 4;
        cfg.decoder_dim = 128; // 64, 32, 16, 8, 4, 2, 1
        cfg.decoder_rates = [5, 2, 2, 2, 2, 2, 2];
        cfg.decoder_kernel_sizes = [9, 4, 4, 4, 4, 4, 4];
        cfg
    }

    fn weights() -> WeightMap {
        WeightMap::generated(|key, shape| {
            let seed = key.bytes().fold(11u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
            let n: usize = shape.iter().product();
            if key.ends_with("filter") {
                // A plausible normalized low-pass; the same for every key, as in the checkpoint.
                let raw: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 - 5.5).abs().recip()).collect();
                let sum: f32 = raw.iter().sum();
                return raw.iter().map(|v| v / sum).collect();
            }
            let fan = (n / shape[0].max(1)).max(1) as f32;
            (0..n)
                .map(|i| {
                    let u = (seed.wrapping_add(i as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / (1u32 << 24) as f32;
                    if key.ends_with("weight_g") { 0.5 + u } else { (u - 0.5) * 2.0 / fan.sqrt() }
                })
                .collect()
        })
    }

    // ---- an independent reference, plain loops over [C][L] signals ----------

    type Signal = Vec<Vec<f32>>;

    fn get(map: &WeightMap, key: &str, shape: &[usize]) -> Vec<f32> {
        cuda_tensor_shaped(map, key, shape).unwrap().host_cow().unwrap().into_owned()
    }

    /// Effective weight of a weight-normed conv, `[d0][d1][k]` flattened.
    fn wn(map: &WeightMap, prefix: &str, shape: [usize; 3]) -> Vec<f32> {
        let g = get(map, &format!("{prefix}.weight_g"), &[shape[0], 1, 1]);
        let v = get(map, &format!("{prefix}.weight_v"), &shape);
        let width = shape[1] * shape[2];
        let mut w = vec![0f32; v.len()];
        for r in 0..shape[0] {
            let norm = v[r * width..(r + 1) * width].iter().map(|x| x * x).sum::<f32>().sqrt();
            for i in 0..width {
                w[r * width + i] = g[r] * v[r * width + i] / norm;
            }
        }
        w
    }

    fn ref_conv(x: &Signal, w: &[f32], bias: Option<&[f32]>, out: usize, k: usize, pad: usize, dilation: usize) -> Signal {
        let (cin, len) = (x.len(), x[0].len());
        let lo = len + 2 * pad - dilation * (k - 1);
        (0..out)
            .map(|o| {
                (0..lo)
                    .map(|t| {
                        let mut acc = bias.map_or(0.0, |b| b[o]);
                        for c in 0..cin {
                            for j in 0..k {
                                let p = t + j * dilation;
                                if p >= pad && p - pad < len {
                                    acc += x[c][p - pad] * w[(o * cin + c) * k + j];
                                }
                            }
                        }
                        acc
                    })
                    .collect()
            })
            .collect()
    }

    /// ConvTranspose1d, weight `[in][out][k]`: scatter each input sample.
    fn ref_conv_transpose(x: &Signal, w: &[f32], bias: &[f32], out: usize, k: usize, stride: usize, pad: usize) -> Signal {
        let (cin, len) = (x.len(), x[0].len());
        let full = (len - 1) * stride + k;
        let mut y = vec![vec![0f32; full]; out];
        for c in 0..cin {
            for t in 0..len {
                for o in 0..out {
                    for j in 0..k {
                        y[o][t * stride + j] += x[c][t] * w[(c * out + o) * k + j];
                    }
                }
            }
        }
        y.iter().enumerate().map(|(o, row)| row[pad..full - pad].iter().map(|v| v + bias[o]).collect()).collect()
    }

    fn replicate(row: &[f32], left: usize, right: usize) -> Vec<f32> {
        let mut out = vec![row[0]; left];
        out.extend_from_slice(row);
        out.extend(std::iter::repeat_n(row[row.len() - 1], right));
        out
    }

    fn ref_alias_free(map: &WeightMap, prefix: &str, x: &Signal, taps: &[f32]) -> Signal {
        let c = x.len();
        let alpha = get(map, &format!("{prefix}.act.alpha"), &[c]);
        let beta = get(map, &format!("{prefix}.act.beta"), &[c]);
        x.iter()
            .enumerate()
            .map(|(ch, row)| {
                let padded = replicate(row, 5, 5);
                let mut up = vec![0f32; (padded.len() - 1) * 2 + 12];
                for (t, v) in padded.iter().enumerate() {
                    for (j, f) in taps.iter().enumerate() {
                        up[2 * t + j] += 2.0 * v * f;
                    }
                }
                let up = &up[15..up.len() - 15];
                let act: Vec<f32> = up.iter().map(|&v| v + (alpha[ch].exp() * v).sin().powi(2) / (beta[ch].exp() + 1e-9)).collect();
                let padded = replicate(&act, 5, 6);
                (0..row.len()).map(|t| (0..12).map(|j| padded[2 * t + j] * taps[j]).sum()).collect()
            })
            .collect()
    }

    fn reference(cfg: &H3AudioVaeConfig, map: &WeightMap, latent: &Signal) -> Vec<f32> {
        let taps = get(map, "decoder.activation_post.upsample.filter", &[1, 1, 12]);
        let (lc, ld) = (cfg.latent_channels, cfg.latent_dim);
        let z: Signal = latent
            .iter()
            .enumerate()
            .map(|(c, row)| row.iter().map(|v| v * cfg.latents_std[c] as f32 + cfg.latents_mean[c] as f32).collect())
            .collect();
        let x = ref_conv(&z, &get(map, "dec_in_proj.weight", &[ld, lc, 1]), Some(&get(map, "dec_in_proj.bias", &[ld])), ld, 1, 0, 1);
        let d = cfg.decoder_dim;
        let mut x = ref_conv(&x, &wn(map, "decoder.conv_pre", [d, ld, 7]), Some(&get(map, "decoder.conv_pre.bias", &[d])), d, 7, 3, 1);
        for i in 0..7 {
            let (cin, cout) = cfg.upsampler_channels(i);
            let (k, r) = (cfg.decoder_kernel_sizes[i], cfg.decoder_rates[i]);
            let p = format!("decoder.ups.{i}.0");
            x = ref_conv_transpose(&x, &wn(map, &p, [cin, cout, k]), &get(map, &format!("{p}.bias"), &[cout]), cout, k, r, (k - r) / 2);
            let mut sum = vec![vec![0f32; x[0].len()]; cout];
            for (j, &kernel) in cfg.resblock_kernel_sizes.iter().enumerate() {
                let p = format!("decoder.resblocks.{}", 3 * i + j);
                let mut h = x.clone();
                for (di, &dil) in cfg.resblock_dilation_sizes[j].iter().enumerate() {
                    let a = ref_alias_free(map, &format!("{p}.activations.{}", 2 * di), &h, &taps);
                    let c1 = format!("{p}.convs1.{di}");
                    let r1 = ref_conv(&a, &wn(map, &c1, [cout, cout, kernel]), Some(&get(map, &format!("{c1}.bias"), &[cout])), cout, kernel, dil * (kernel - 1) / 2, dil);
                    let a = ref_alias_free(map, &format!("{p}.activations.{}", 2 * di + 1), &r1, &taps);
                    let c2 = format!("{p}.convs2.{di}");
                    let r2 = ref_conv(&a, &wn(map, &c2, [cout, cout, kernel]), Some(&get(map, &format!("{c2}.bias"), &[cout])), cout, kernel, (kernel - 1) / 2, 1);
                    for (hr, rr) in h.iter_mut().zip(&r2) {
                        hr.iter_mut().zip(rr).for_each(|(a, b)| *a += b);
                    }
                }
                for (sr, hr) in sum.iter_mut().zip(&h) {
                    sr.iter_mut().zip(hr).for_each(|(a, b)| *a += b);
                }
            }
            x = sum.into_iter().map(|row| row.into_iter().map(|v| v / 3.0).collect()).collect();
        }
        let x = ref_alias_free(map, "decoder.activation_post", &x, &taps);
        let last = x.len();
        let y = ref_conv(&x, &wn(map, "decoder.conv_post", [1, last, 7]), None, 1, 7, 3, 1);
        y[0].iter().map(|v| v.clamp(-1.0, 1.0)).collect()
    }

    #[test]
    fn weight_norm_folds_over_every_dim_but_the_first() {
        // Two rows of a [2, 1, 2] weight: norms 5 and 13.
        let w = fold_weight_norm(&[10.0, 26.0], &[3.0, 4.0, 5.0, 12.0], 2).unwrap();
        assert_eq!(w, vec![6.0, 8.0, 10.0, 24.0]);
        assert!(fold_weight_norm(&[1.0], &[1.0, 2.0, 3.0], 2).is_err());
    }

    #[test]
    fn decode_matches_a_loop_reference_per_channel() {
        let (cfg, map) = (tiny_cfg(), weights());
        let dec = H3AudioDecoder::load(cfg.clone(), &map).unwrap();
        let (b, c, l) = (2usize, cfg.latent_channels, 3usize);
        let lat: Vec<f32> = (0..b * c * l).map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.07).collect();
        let got = dec.decode(&CudaTensor::from_vec(lat.clone(), vec![b, c, l]).unwrap()).unwrap();
        assert_eq!(got.shape, vec![b, l * cfg.hop_length()]);
        assert_eq!(cfg.hop_length(), 320);
        let got = got.host_cow().unwrap();
        let mut energy = 0.0f32;
        for bi in 0..b {
            let signal: Signal = (0..c).map(|ch| lat[(bi * c + ch) * l..(bi * c + ch + 1) * l].to_vec()).collect();
            let want = reference(&cfg, &map, &signal);
            assert_eq!(want.len(), l * 320);
            for (t, (g, w)) in got[bi * l * 320..(bi + 1) * l * 320].iter().zip(&want).enumerate() {
                assert!((g - w).abs() < 2e-5, "channel {bi} sample {t}: {g} vs {w}");
                energy += w * w;
            }
        }
        assert!(energy > 1e-6, "a silent reference proves nothing");
    }

    #[test]
    fn rows_are_channel_major_so_stereo_is_a_batch_of_two() {
        let (cfg, map) = (tiny_cfg(), weights());
        let dec = H3AudioDecoder::load(cfg.clone(), &map).unwrap();
        let (na, c) = (2usize, cfg.latent_channels);
        let rows: Vec<f32> = (0..2 * na * c).map(|i| (i as f32 * 0.31).sin()).collect();
        let stereo = dec.decode_rows(&CudaTensor::from_vec(rows.clone(), vec![2 * na, c]).unwrap(), 2).unwrap();
        // The right channel alone: rows Na..2Na, as [1, C, Na].
        let right: Vec<f32> = (0..c).flat_map(|ch| (0..na).map(move |a| (na + a, ch))).map(|(r, ch)| rows[r * c + ch]).collect();
        let mono = dec.decode(&CudaTensor::from_vec(right, vec![1, c, na]).unwrap()).unwrap();
        let (s, m) = (stereo.host_cow().unwrap(), mono.host_cow().unwrap());
        assert_eq!(&s[m.len()..], &m[..]);
    }
}
