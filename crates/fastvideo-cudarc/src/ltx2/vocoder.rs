//! `LTX2Vocoder` / `LTX2VocoderWithBWE`: HiFi-GAN from stereo log-mel to waveform.
//!
//! The mel arrives as `[B, 2, T, 64]` and is read as 128 channels over `T`
//! frames (channel index `stereo · 64 + mel_bin`). Upsample stages telescope
//! the time axis; every stage satisfies `kernel - 2·pad = stride`, so lengths
//! are exact multiples. After each upsample, three residual blocks (kernels
//! 3 / 7 / 11) run **in parallel on the same input** and are averaged.
//!
//! LTX-2.0: LeakyReLU path, 24 kHz out, no BWE.
//! LTX-2.5: SnakeBeta + antialias main stack @ 16 kHz, then MelSTFT →
//! `bwe_generator` residual + Hann-sinc ×3 skip → 48 kHz.
//!
//! Weight norm: the published diffusers checkpoint stores plain weights. An
//! original HiFi-GAN checkpoint stores `weight_g` / `weight_v` (or torch's
//! `parametrizations.weight.original{0,1}`); those are folded to
//! `g · v / ‖v‖` once at load, so the graph never sees them.
//! See docs/ports/ltx2.md §d.

use fastvideo_models::ltx2::config::{Ltx2BweConfig, Ltx2VocoderConfig};

use crate::wan::ops::PadMode;
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

use super::{msg, tanh};

/// Alias-free SnakeBeta geometry — same 12-tap / ratio-2 layout as H3 BigVGAN
/// (`MiniMaxH3AudioUpSample1d` / `LowPassFilter1d`) and LTX-2.5 `Activation1d`.
const FILTER_TAPS: usize = 12;
const AA_RATIO: usize = 2;
const UP_PAD: usize = FILTER_TAPS / AA_RATIO - 1; // 5
const UP_CROP_LEFT: usize = UP_PAD * AA_RATIO + (FILTER_TAPS - AA_RATIO) / 2; // 15
const UP_CROP_RIGHT: usize = UP_PAD * AA_RATIO + (FILTER_TAPS - AA_RATIO + 1) / 2; // 15
const DOWN_PAD_LEFT: usize = FILTER_TAPS / 2 - 1; // 5
const DOWN_PAD_RIGHT: usize = FILTER_TAPS / 2; // 6

/// Hann-sinc `UpSample1d(ratio=3, window_type="hann")` constants from
/// diffusers `vocoder.py` (rolloff 0.99, lowpass_filter_width 6).
const HANN_ROLLOFF: f32 = 0.99;
const HANN_LOWPASS_WIDTH: f32 = 6.0;

/// Nearest-neighbor upsample along the last axis of `[B, C, L]` — fallback when
/// BWE checkpoint keys are absent.
fn upsample_time_nearest(x: &CudaTensor, ratio: usize) -> Result<CudaTensor> {
    let [b, c, l] = match x.shape[..] {
        [b, c, l] => [b, c, l],
        _ => return Err(msg(format!("vocoder rate upsample expects [B, C, L], got {:?}", x.shape))),
    };
    if ratio == 0 {
        return Err(msg("vocoder rate upsample ratio must be ≥ 1"));
    }
    if ratio == 1 {
        return Ok(x.clone());
    }
    let src = x.host_cow()?;
    let mut out = Vec::with_capacity(b * c * l * ratio);
    for v in src.iter() {
        for _ in 0..ratio {
            out.push(*v);
        }
    }
    CudaTensor::from_vec(out, vec![b, c, l * ratio])
}

fn pinned(data: Vec<f32>, shape: Vec<usize>) -> Result<CudaTensor> {
    let mut t = CudaTensor::from_vec(data, shape)?;
    t.pin_device()?;
    Ok(t)
}

fn host_values(map: &WeightMap, key: &str, shape: &[usize]) -> Result<Vec<f32>> {
    Ok(cuda_tensor_shaped(map, key, shape)?.host_cow()?.into_owned())
}

fn depthwise_filter(taps: &[f32], channels: usize) -> Result<CudaTensor> {
    let data: Vec<f32> = (0..channels).flat_map(|_| taps.iter().copied()).collect();
    pinned(data, vec![channels, 1, taps.len()])
}

/// `sinc(x) = sin(πx)/(πx)` (PyTorch / numpy).
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        (std::f32::consts::PI * x).sin() / (std::f32::consts::PI * x)
    }
}

/// Hann-windowed sinc kernel for `UpSample1d(..., window_type="hann")`.
fn hann_sinc_kernel(ratio: usize) -> (Vec<f32>, usize, usize, usize, usize) {
    let width = (HANN_LOWPASS_WIDTH / HANN_ROLLOFF).ceil() as usize;
    let kernel_size = 2 * width * ratio + 1;
    let pad = width;
    let pad_left = 2 * width * ratio;
    let pad_right = kernel_size - ratio;
    let mut taps = Vec::with_capacity(kernel_size);
    for i in 0..kernel_size {
        let time = (i as f32 / ratio as f32 - width as f32) * HANN_ROLLOFF;
        let clamped = time.clamp(-HANN_LOWPASS_WIDTH, HANN_LOWPASS_WIDTH);
        let window = (clamped * std::f32::consts::PI / HANN_LOWPASS_WIDTH / 2.0).cos().powi(2);
        taps.push(sinc(time) * window * HANN_ROLLOFF / ratio as f32);
    }
    (taps, kernel_size, pad, pad_left, pad_right)
}

/// Alias-free SnakeBeta: upsample ×2 → SnakeBeta → downsample ×2.
struct AliasFreeSnake {
    alpha: CudaTensor,
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
        let up = x
            .pad(2, UP_PAD, UP_PAD, PadMode::Replicate)?
            .conv_transpose1d(&self.filter, None, 0, AA_RATIO, 1, channels, 0)?
            .try_mul_scalar(AA_RATIO as f32)?;
        let up = up.narrow(2, UP_CROP_LEFT, up.shape[2] - UP_CROP_LEFT - UP_CROP_RIGHT)?;
        if up.shape[2] != AA_RATIO * len {
            return Err(msg(format!("vocoder alias-free upsample produced {} samples from {len}", up.shape[2])));
        }
        let act = up.snake_beta(&self.alpha, &self.inv_beta)?;
        act.pad(2, DOWN_PAD_LEFT, DOWN_PAD_RIGHT, PadMode::Replicate)?.conv1d(&self.filter, None, 0, AA_RATIO, 1, channels)
    }
}

/// `w = g · v / ‖v‖`, the norm taken over everything but axis 0 (torch's
/// `weight_norm(dim=0)`). `v` is row-major with `rows` leading entries.
fn fold_weight_norm(g: &[f32], v: &[f32], rows: usize) -> Vec<f32> {
    let inner = v.len() / rows.max(1);
    v.chunks_exact(inner.max(1))
        .zip(g)
        .flat_map(|(row, g)| {
            let norm = row.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt();
            let k = (f64::from(*g) / norm) as f32;
            row.iter().map(move |a| a * k)
        })
        .collect()
}

/// A conv weight of `shape`, folding weight norm when the checkpoint has it.
fn conv_weight(map: &WeightMap, prefix: &str, shape: &[usize]) -> Result<CudaTensor> {
    let plain = format!("{prefix}.weight");
    let pairs = [
        (format!("{prefix}.weight_g"), format!("{prefix}.weight_v")),
        (format!("{prefix}.parametrizations.weight.original0"), format!("{prefix}.parametrizations.weight.original1")),
    ];
    let folded = if map.has_tensor(&plain) { None } else { pairs.iter().find(|(g, v)| map.has_tensor(g) && map.has_tensor(v)) };
    let mut w = match folded {
        None => cuda_tensor_shaped(map, &plain, shape)?,
        Some((g, v)) => {
            let v = cuda_tensor_shaped(map, v, shape)?;
            let (g_shape, g) = map.get_f32(g)?;
            if g.len() != shape[0] {
                return Err(msg(format!("{prefix}: weight-norm gain {g_shape:?} for a weight of {shape:?}")));
            }
            CudaTensor::from_vec(fold_weight_norm(&g, &v.host_cow()?, shape[0]), shape.to_vec())?
        }
    };
    w.pin_device()?;
    Ok(w)
}

struct Conv1d {
    weight: CudaTensor,
    bias: CudaTensor,
    padding: usize,
    dilation: usize,
}

impl Conv1d {
    /// `padding = "same"` for an odd kernel: `dilation · (k - 1) / 2` each side.
    /// Bias is optional (`final_bias=false` on LTX-2.5 `conv_out`).
    fn load(map: &WeightMap, prefix: &str, cin: usize, cout: usize, kernel: usize, dilation: usize) -> Result<Self> {
        if kernel.is_multiple_of(2) {
            return Err(msg(format!("{prefix}: \"same\" padding needs an odd kernel, got {kernel}")));
        }
        let bias_key = format!("{prefix}.bias");
        let has_weight = map.has_tensor(&format!("{prefix}.weight"))
            || map.has_tensor(&format!("{prefix}.weight_g"))
            || map.has_tensor(&format!("{prefix}.parametrizations.weight.original0"));
        // Omit → zeros when the checkpoint has weights but no bias (`final_bias=false`).
        // Synthetic `WeightMap::generated` invents every key so host refs stay in sync.
        let mut bias = if map.has_tensor(&bias_key) || (!has_weight && map.contains(&bias_key)) {
            cuda_tensor_shaped(map, &bias_key, &[cout])?
        } else {
            CudaTensor::zeros(&[cout])
        };
        bias.pin_device()?;
        Ok(Self { weight: conv_weight(map, prefix, &[cout, cin, kernel])?, bias, padding: dilation * (kernel - 1) / 2, dilation })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        x.conv1d(&self.weight, Some(&self.bias), self.padding, 1, self.dilation, 1)
    }
}

struct Upsampler {
    /// `ConvTranspose1d` layout: `[in, out, k]`.
    weight: CudaTensor,
    bias: CudaTensor,
    stride: usize,
    padding: usize,
}

/// One HiFi-GAN residual block: for each dilation, a dilated conv then a plain
/// one, LeakyReLU before each, added back. LTX-2.0 only.
struct ResBlock {
    convs: Vec<(Conv1d, Conv1d)>,
}

impl ResBlock {
    fn forward(&self, x: &CudaTensor, slope: f32) -> Result<CudaTensor> {
        let mut x = x.clone();
        for (dilated, plain) in &self.convs {
            let h = dilated.forward(&x.leaky_relu(slope))?;
            x = x.add(&plain.forward(&h.leaky_relu(slope))?)?;
        }
        Ok(x)
    }
}

/// BigVGAN `AMPBlock1`: per dilation `x += conv2(act2(conv1(act1(x))))` with
/// alias-free SnakeBeta. LTX-2.5 `vocoder.resnets.*`.
struct AmpBlock {
    stages: Vec<(AliasFreeSnake, Conv1d, AliasFreeSnake, Conv1d)>,
}

impl AmpBlock {
    fn load(map: &WeightMap, prefix: &str, channels: usize, kernel: usize, dilations: &[usize], taps: &[f32]) -> Result<Self> {
        let stages = dilations
            .iter()
            .enumerate()
            .map(|(d, &dilation)| {
                Ok((
                    AliasFreeSnake::load(map, &format!("{prefix}.acts1.{d}"), channels, taps)?,
                    Conv1d::load(map, &format!("{prefix}.convs1.{d}"), channels, channels, kernel, dilation)?,
                    AliasFreeSnake::load(map, &format!("{prefix}.acts2.{d}"), channels, taps)?,
                    Conv1d::load(map, &format!("{prefix}.convs2.{d}"), channels, channels, kernel, 1)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { stages })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let mut x = x.clone();
        for (act1, conv1, act2, conv2) in &self.stages {
            let r = act1.forward(&x)?.conv1d(&conv1.weight, Some(&conv1.bias), conv1.padding, 1, conv1.dilation, 1)?;
            let r = act2.forward(&r)?.conv1d(&conv2.weight, Some(&conv2.bias), conv2.padding, 1, conv2.dilation, 1)?;
            x = x.add(&r)?;
        }
        Ok(x)
    }
}

enum StageResnets {
    Leaky(Vec<ResBlock>),
    Amp(Vec<AmpBlock>),
}

/// Geometry shared by the main HiFi-GAN and `bwe_generator`.
struct StackGeom {
    in_channels: usize,
    hidden_channels: usize,
    out_channels: usize,
    upsample_kernel_sizes: Vec<usize>,
    upsample_factors: Vec<usize>,
    resnet_kernel_sizes: [usize; 3],
    resnet_dilations: [[usize; 3]; 3],
    leaky_relu_negative_slope: f64,
    final_leaky_relu_negative_slope: f64,
    final_tanh: bool,
}

impl StackGeom {
    fn from_main(cfg: &Ltx2VocoderConfig) -> Self {
        Self {
            in_channels: cfg.in_channels,
            hidden_channels: cfg.hidden_channels,
            out_channels: cfg.out_channels,
            upsample_kernel_sizes: cfg.upsample_kernel_sizes.clone(),
            upsample_factors: cfg.upsample_factors.clone(),
            resnet_kernel_sizes: cfg.resnet_kernel_sizes,
            resnet_dilations: cfg.resnet_dilations,
            leaky_relu_negative_slope: cfg.leaky_relu_negative_slope,
            final_leaky_relu_negative_slope: cfg.final_leaky_relu_negative_slope,
            final_tanh: cfg.final_tanh,
        }
    }

    fn from_bwe(cfg: &Ltx2VocoderConfig, bwe: &Ltx2BweConfig) -> Self {
        Self {
            in_channels: bwe.in_channels,
            hidden_channels: bwe.hidden_channels,
            out_channels: bwe.out_channels,
            upsample_kernel_sizes: bwe.upsample_kernel_sizes.clone(),
            upsample_factors: bwe.upsample_factors.clone(),
            resnet_kernel_sizes: cfg.resnet_kernel_sizes,
            resnet_dilations: cfg.resnet_dilations,
            leaky_relu_negative_slope: cfg.leaky_relu_negative_slope,
            final_leaky_relu_negative_slope: cfg.final_leaky_relu_negative_slope,
            final_tanh: false,
        }
    }

    fn upsample_padding(&self, stage: usize) -> usize {
        (self.upsample_kernel_sizes[stage] - self.upsample_factors[stage]) / 2
    }

    fn stage_channels(&self, stage: usize) -> usize {
        self.hidden_channels >> (stage + 1)
    }
}

struct HiFiGan {
    geom: StackGeom,
    conv_in: Conv1d,
    stages: Vec<(Upsampler, StageResnets)>,
    act_out: Option<AliasFreeSnake>,
    conv_out: Conv1d,
}

impl HiFiGan {
    /// `force_snake`: invent SnakeBeta when the map is a generator (host tests for
    /// LTX-2.5). Real checkpoints are detected via `act_out.act.alpha` keys.
    fn load(map: &WeightMap, root: &str, geom: StackGeom, force_snake: bool) -> Result<Self> {
        let snake = map.has_tensor(&format!("{root}act_out.act.alpha"))
            || map.has_tensor(&format!("{root}resnets.0.acts1.0.act.alpha"))
            || (force_snake && map.contains(&format!("{root}act_out.act.alpha")));
        let taps = if snake {
            // Prefer checkpoint taps; for WeightMap::generated invent zeros →
            // fall back to a unit impulse so alias-free geometry still runs.
            let key = format!("{root}act_out.upsample.filter");
            let mut t = host_values(map, &key, &[1, 1, FILTER_TAPS])?;
            if t.iter().all(|v| *v == 0.0) {
                t[FILTER_TAPS / 2] = 1.0;
            }
            Some(t)
        } else {
            None
        };
        let per_stage = geom.resnet_kernel_sizes.len();
        let mut stages = Vec::with_capacity(geom.upsample_factors.len());
        let mut cin = geom.hidden_channels;
        for (i, (&stride, &kernel)) in geom.upsample_factors.iter().zip(&geom.upsample_kernel_sizes).enumerate() {
            let cout = geom.stage_channels(i);
            if kernel < stride || cout == 0 || cout * 2 != cin {
                return Err(msg(format!("{root}stage {i}: kernel {kernel}, stride {stride}, {cin} -> {cout} channels")));
            }
            let prefix = format!("{root}upsamplers.{i}");
            let bias_key = format!("{prefix}.bias");
            let has_weight = map.has_tensor(&format!("{prefix}.weight"))
                || map.has_tensor(&format!("{prefix}.weight_g"))
                || map.has_tensor(&format!("{prefix}.parametrizations.weight.original0"));
            let mut bias = if map.has_tensor(&bias_key) || (!has_weight && map.contains(&bias_key)) {
                cuda_tensor_shaped(map, &bias_key, &[cout])?
            } else {
                CudaTensor::zeros(&[cout])
            };
            bias.pin_device()?;
            let up = Upsampler {
                weight: conv_weight(map, &prefix, &[cin, cout, kernel])?,
                bias,
                stride,
                padding: geom.upsample_padding(i),
            };
            let blocks = if let Some(taps) = taps.as_deref() {
                let amps = geom
                    .resnet_kernel_sizes
                    .iter()
                    .zip(&geom.resnet_dilations)
                    .enumerate()
                    .map(|(j, (&k, dilations))| AmpBlock::load(map, &format!("{root}resnets.{}", i * per_stage + j), cout, k, dilations, taps))
                    .collect::<Result<Vec<_>>>()?;
                StageResnets::Amp(amps)
            } else {
                let leaky = geom
                    .resnet_kernel_sizes
                    .iter()
                    .zip(&geom.resnet_dilations)
                    .enumerate()
                    .map(|(j, (&k, dilations))| {
                        let p = format!("{root}resnets.{}", i * per_stage + j);
                        let convs = dilations
                            .iter()
                            .enumerate()
                            .map(|(n, &d)| {
                                Ok((
                                    Conv1d::load(map, &format!("{p}.convs1.{n}"), cout, cout, k, d)?,
                                    Conv1d::load(map, &format!("{p}.convs2.{n}"), cout, cout, k, 1)?,
                                ))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Ok(ResBlock { convs })
                    })
                    .collect::<Result<Vec<_>>>()?;
                StageResnets::Leaky(leaky)
            };
            stages.push((up, blocks));
            cin = cout;
        }
        let act_out = taps.as_deref().map(|t| AliasFreeSnake::load(map, &format!("{root}act_out"), cin, t)).transpose()?;
        Ok(Self {
            conv_in: Conv1d::load(map, &format!("{root}conv_in"), geom.in_channels, geom.hidden_channels, 7, 1)?,
            stages,
            act_out,
            conv_out: Conv1d::load(map, &format!("{root}conv_out"), cin, geom.out_channels, 7, 1)?,
            geom,
        })
    }

    /// Mel `[B, C, T, M]` → waveform `[B, out, L]`.
    fn forward(&self, mel: &CudaTensor) -> Result<CudaTensor> {
        let [b, c, t, m] = mel.shape[..] else {
            return Err(msg(format!("HiFi-GAN expects mel [B, C, T, M], got {:?}", mel.shape)));
        };
        if c * m != self.geom.in_channels || t == 0 {
            return Err(msg(format!("HiFi-GAN expects {} mel channels, got {c} x {m} in {:?}", self.geom.in_channels, mel.shape)));
        }
        let slope = self.geom.leaky_relu_negative_slope as f32;
        let snake = self.act_out.is_some();
        let mut x = self.conv_in.forward(&mel.permute(&[0, 1, 3, 2])?.reshape(vec![b, c * m, t])?)?;
        for (up, blocks) in &self.stages {
            x = if snake {
                x.conv_transpose1d(&up.weight, Some(&up.bias), up.padding, up.stride, 1, 1, 0)?
            } else {
                x.leaky_relu(slope).conv_transpose1d(&up.weight, Some(&up.bias), up.padding, up.stride, 1, 1, 0)?
            };
            let outs = match blocks {
                StageResnets::Leaky(rs) => rs.iter().map(|r| r.forward(&x, slope)).collect::<Result<Vec<_>>>()?,
                StageResnets::Amp(amps) => amps.iter().map(|r| r.forward(&x)).collect::<Result<Vec<_>>>()?,
            };
            let share = 1.0 / outs.len() as f32;
            x = CudaTensor::lincomb(&outs.iter().map(|o| (share, o)).collect::<Vec<_>>())?;
        }
        let x = if let Some(act) = &self.act_out {
            self.conv_out.forward(&act.forward(&x)?)?
        } else {
            self.conv_out.forward(&x.leaky_relu(self.geom.final_leaky_relu_negative_slope as f32))?
        };
        if self.geom.final_tanh {
            tanh(&x)
        } else {
            Ok(x)
        }
    }
}

/// Causal MelSTFT: checkpoint `forward_basis` + `mel_basis` (bf16 in the pack).
struct MelStft {
    forward_basis: CudaTensor, // [2·n_freqs, 1, filter_length]
    mel_basis: CudaTensor,     // [num_mel, n_freqs]
    hop_length: usize,
    window_length: usize,
    n_freqs: usize,
    num_mel: usize,
}

impl MelStft {
    fn load(map: &WeightMap, bwe: &Ltx2BweConfig) -> Result<Self> {
        let n_freqs = bwe.n_freqs();
        Ok(Self {
            forward_basis: {
                let mut t = cuda_tensor_shaped(map, "mel_stft.stft_fn.forward_basis", &[n_freqs * 2, 1, bwe.filter_length])?;
                t.pin_device()?;
                t
            },
            mel_basis: {
                let mut t = cuda_tensor_shaped(map, "mel_stft.mel_basis", &[bwe.num_mel_channels, n_freqs])?;
                t.pin_device()?;
                t
            },
            hop_length: bwe.hop_length,
            window_length: bwe.window_length,
            n_freqs,
            num_mel: bwe.num_mel_channels,
        })
    }

    /// Waveform `[N, L]` → log-mel `[N, num_mel, frames]`.
    fn forward_flat(&self, wave: &CudaTensor) -> Result<CudaTensor> {
        let [n, l] = match wave.shape[..] {
            [n, l] => [n, l],
            _ => return Err(msg(format!("MelSTFT expects [N, L], got {:?}", wave.shape))),
        };
        let left = self.window_length.saturating_sub(self.hop_length);
        let x = wave.reshape(vec![n, 1, l])?.pad(2, left, 0, PadMode::Zeros)?;
        let spec = x.conv1d(&self.forward_basis, None, 0, self.hop_length, 1, 1)?;
        // Magnitude + mel projection on host: `matmul` refuses the unbatched
        // `[M, F] @ [N, F, T]` broadcast on CUDA builds (no device path).
        let frames = spec.shape[2];
        let data = spec.host_cow()?;
        let mel_w = self.mel_basis.host_cow()?;
        let mut log_mel = vec![0f32; n * self.num_mel * frames];
        for bi in 0..n {
            for m in 0..self.num_mel {
                for t in 0..frames {
                    let mut acc = 0f32;
                    for f in 0..self.n_freqs {
                        let re = data[((bi * 2 * self.n_freqs + f) * frames) + t];
                        let im = data[((bi * 2 * self.n_freqs + self.n_freqs + f) * frames) + t];
                        let mag = (re * re + im * im).sqrt();
                        acc += mel_w[m * self.n_freqs + f] * mag;
                    }
                    log_mel[(bi * self.num_mel + m) * frames + t] = acc.max(1e-5).ln();
                }
            }
        }
        CudaTensor::from_vec(log_mel, vec![n, self.num_mel, frames])
    }
}

/// Hann-sinc skip resampler: 16 kHz → 48 kHz.
struct HannResampler {
    taps: Vec<f32>,
    ratio: usize,
    pad: usize,
    pad_left: usize,
    pad_right: usize,
}

impl HannResampler {
    fn new(ratio: usize) -> Self {
        let (taps, _k, pad, pad_left, pad_right) = hann_sinc_kernel(ratio);
        Self { taps, ratio, pad, pad_left, pad_right }
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let [b, c, l] = match x.shape[..] {
            [b, c, l] => [b, c, l],
            _ => return Err(msg(format!("Hann resampler expects [B, C, L], got {:?}", x.shape))),
        };
        let filter = depthwise_filter(&self.taps, c)?;
        let up = x
            .pad(2, self.pad, self.pad, PadMode::Replicate)?
            .conv_transpose1d(&filter, None, 0, self.ratio, 1, c, 0)?
            .try_mul_scalar(self.ratio as f32)?;
        let crop = up.shape[2].saturating_sub(self.pad_left + self.pad_right);
        if crop == 0 {
            return Err(msg(format!("Hann resampler crop empty from len {} (in {l})", up.shape[2])));
        }
        let out = up.narrow(2, self.pad_left, crop)?;
        if out.shape != [b, c, l * self.ratio] && out.shape[2] < l * self.ratio {
            // Length can be slightly longer than ratio·L; caller crops.
        }
        let _ = (b, c);
        Ok(out)
    }
}

struct BwePath {
    mel_stft: MelStft,
    generator: HiFiGan,
    resampler: HannResampler,
    hop_length: usize,
    rate: usize,
}

impl BwePath {
    fn load(map: &WeightMap, cfg: &Ltx2VocoderConfig, bwe: &Ltx2BweConfig) -> Result<Self> {
        Ok(Self {
            mel_stft: MelStft::load(map, bwe)?,
            generator: HiFiGan::load(map, "bwe_generator.", StackGeom::from_bwe(cfg, bwe), true)?,
            resampler: HannResampler::new(cfg.rate_upsample()),
            hop_length: bwe.hop_length,
            rate: cfg.rate_upsample(),
        })
    }

    /// 16 kHz waveform `[B, C, L]` → 48 kHz `[B, C, rate·L]` in `[-1, 1]`.
    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let [b, c, num_samples] = match x.shape[..] {
            [b, c, n] => [b, c, n],
            _ => return Err(msg(format!("BWE expects [B, C, L], got {:?}", x.shape))),
        };
        let rem = num_samples % self.hop_length;
        let x_pad = if rem == 0 {
            x.clone()
        } else {
            x.pad(2, 0, self.hop_length - rem, PadMode::Zeros)?
        };
        let flat = x_pad.reshape(vec![b * c, x_pad.shape[2]])?;
        let mel = self.mel_stft.forward_flat(&flat)?; // [B·C, M, frames]
        let frames = mel.shape[2];
        let mel = mel.reshape(vec![b, c, self.mel_stft.num_mel, frames])?.permute(&[0, 1, 3, 2])?;
        let residual = self.generator.forward(&mel)?;
        let skip = self.resampler.forward(x)?;
        // Align lengths then residual-add + clamp.
        let out_len = num_samples * self.rate;
        let r_len = residual.shape[2].min(skip.shape[2]).min(out_len);
        let residual = residual.narrow(2, 0, r_len)?;
        let skip = skip.narrow(2, 0, r_len)?;
        let mut wave = residual.add(&skip)?.clamp(-1.0, 1.0);
        if r_len < out_len {
            wave = wave.pad(2, 0, out_len - r_len, PadMode::Zeros)?;
        } else if wave.shape[2] > out_len {
            wave = wave.narrow(2, 0, out_len)?;
        }
        Ok(wave)
    }
}

fn should_load_bwe(map: &WeightMap, cfg: &Ltx2VocoderConfig) -> bool {
    if !cfg.with_bwe || cfg.bwe.is_none() {
        return false;
    }
    if map.has_tensor("mel_stft.mel_basis")
        || map.has_tensor("bwe_generator.conv_in.weight")
        || map.has_tensor("bwe_generator.conv_in.weight_g")
    {
        return true;
    }
    // WeightMap::generated: invent BWE so host tests exercise the path.
    map.contains("mel_stft.mel_basis") && !map.has_tensor("conv_in.weight") && !map.has_tensor("vocoder.conv_in.weight")
}

pub struct Vocoder {
    cfg: Ltx2VocoderConfig,
    main: HiFiGan,
    bwe: Option<BwePath>,
}

impl Vocoder {
    /// `map` is the diffusers `vocoder/` folder.
    pub fn load(map: &WeightMap, cfg: &Ltx2VocoderConfig) -> Result<Self> {
        // LTX-2.5 nests the HiFi-GAN under `vocoder.*` and adds `bwe_generator.*`.
        let root = if map.has_tensor("vocoder.conv_in.weight")
            || map.has_tensor("vocoder.conv_in.weight_g")
            || map.has_tensor("vocoder.parametrizations.weight.original0")
            || (cfg.with_bwe && map.contains("vocoder.conv_in.weight") && !map.has_tensor("conv_in.weight"))
        {
            "vocoder."
        } else {
            ""
        };
        let snake = map.has_tensor(&format!("{root}act_out.act.alpha"))
            || map.has_tensor(&format!("{root}resnets.0.acts1.0.act.alpha"))
            || (cfg.with_bwe && map.contains(&format!("{root}act_out.act.alpha")));
        let load_bwe = should_load_bwe(map, cfg);
        if cfg.with_bwe && snake && load_bwe {
            eprintln!("ltx2 vocoder: SnakeBeta main (`{root}`) + BWE (Hann-sinc ×{} + MelSTFT + bwe_generator)", cfg.rate_upsample());
        } else if cfg.with_bwe && snake {
            eprintln!(
                "ltx2 vocoder: SnakeBeta main (`{root}`); BWE keys absent — nearest ×{} lifts 16→48 kHz",
                cfg.rate_upsample()
            );
        } else if cfg.with_bwe {
            eprintln!(
                "ltx2 vocoder: with_bwe=true but no SnakeBeta keys under `{root}` — \
                 using LeakyReLU path (synthetic / incomplete checkpoint)"
            );
        }
        let main = HiFiGan::load(map, root, StackGeom::from_main(cfg), cfg.with_bwe)?;
        let bwe = if load_bwe {
            let bwe_cfg = cfg.bwe.as_ref().ok_or_else(|| msg("with_bwe but missing Ltx2BweConfig"))?;
            Some(BwePath::load(map, cfg, bwe_cfg)?)
        } else {
            None
        };
        Ok(Self { cfg: cfg.clone(), main, bwe })
    }

    /// Mel `[B, 2, T, 64]` → waveform `[B, 2, factor·T]` in `[-1, 1]` at
    /// `output_sampling_rate`.
    pub fn forward(&self, mel: &CudaTensor) -> Result<CudaTensor> {
        let x = self.main.forward(mel)?;
        let ratio = self.cfg.rate_upsample();
        if ratio <= 1 {
            return Ok(x);
        }
        if let Some(bwe) = &self.bwe {
            return bwe.forward(&x);
        }
        upsample_time_nearest(&x, ratio)
    }

    pub fn sample_rate(&self) -> usize {
        self.cfg.output_sampling_rate
    }
}

#[cfg(test)]
mod tests {
    use super::super::attention::tests::{get, weights};
    use super::*;

    /// `[c, l]` signal for the loop reference.
    #[derive(Clone)]
    struct Sig {
        c: usize,
        l: usize,
        v: Vec<f32>,
    }

    fn conv1d(x: &Sig, w: &[f32], b: &[f32], k: usize, d: usize) -> Sig {
        let pad = (d * (k - 1) / 2) as isize;
        let mut v = vec![0f32; b.len() * x.l];
        for (o, bias) in b.iter().enumerate() {
            for t in 0..x.l {
                let mut acc = *bias;
                for i in 0..x.c {
                    for j in 0..k {
                        let s = t as isize + (j * d) as isize - pad;
                        if s >= 0 && (s as usize) < x.l {
                            acc += w[(o * x.c + i) * k + j] * x.v[i * x.l + s as usize];
                        }
                    }
                }
                v[o * x.l + t] = acc;
            }
        }
        Sig { c: b.len(), l: x.l, v }
    }

    /// Scatter form of `ConvTranspose1d`, weight `[in, out, k]`.
    fn conv_transpose1d(x: &Sig, w: &[f32], b: &[f32], k: usize, stride: usize, pad: usize) -> Sig {
        let (cout, lo) = (b.len(), (x.l - 1) * stride + k - 2 * pad);
        let mut v: Vec<f32> = b.iter().flat_map(|b| std::iter::repeat_n(*b, lo)).collect();
        for i in 0..x.c {
            for t in 0..x.l {
                for o in 0..cout {
                    for j in 0..k {
                        let at = (t * stride + j) as isize - pad as isize;
                        if at >= 0 && (at as usize) < lo {
                            v[o * lo + at as usize] += x.v[i * x.l + t] * w[(i * cout + o) * k + j];
                        }
                    }
                }
            }
        }
        Sig { c: cout, l: lo, v }
    }

    fn leaky(x: &Sig, slope: f32) -> Sig {
        Sig { v: x.v.iter().map(|&a| if a >= 0.0 { a } else { a * slope }).collect(), ..x.clone() }
    }

    fn tiny() -> Ltx2VocoderConfig {
        Ltx2VocoderConfig {
            in_channels: 6,
            hidden_channels: 8,
            upsample_kernel_sizes: vec![7, 4, 4, 4, 4],
            upsample_factors: vec![3, 2, 2, 2, 2],
            ..Ltx2VocoderConfig::ltx2_19b()
        }
    }

    #[test]
    fn vocoder_matches_a_loop_reference() {
        // The five-stage shape is fixed by the config arrays; halving 8 channels
        // five times does not work, so check the geometry error first…
        assert!(Vocoder::load(&weights(), &tiny()).is_err(), "8 → 4 → 2 → 1 → 0 channels must be refused");
        // …and run the reference on a config that does: 64 hidden channels.
        let cfg = Ltx2VocoderConfig { hidden_channels: 64, ..tiny() };
        let map = weights();
        let voc = Vocoder::load(&map, &cfg).unwrap();
        let (t, m) = (3usize, 3usize);
        let mel: Vec<f32> = (0..2 * t * m).map(|i| (i as f32 * 0.61).sin()).collect();
        let got = voc.forward(&CudaTensor::from_vec(mel.clone(), vec![1, 2, t, m]).unwrap()).unwrap();
        assert_eq!(got.shape, vec![1, 2, t * 3 * 2 * 2 * 2 * 2]);

        // Channel = stereo · M + bin, time last.
        let mut x = Sig { c: 2 * m, l: t, v: vec![0.0; 2 * m * t] };
        for s in 0..2 {
            for ti in 0..t {
                for bin in 0..m {
                    x.v[(s * m + bin) * t + ti] = mel[(s * t + ti) * m + bin];
                }
            }
        }
        let load = |p: &str, o: usize, i: usize, k: usize| (get(&map, &format!("{p}.weight"), &[o, i, k]), get(&map, &format!("{p}.bias"), &[o]));
        let (w, b) = load("conv_in", 64, 6, 7);
        let mut x = conv1d(&x, &w, &b, 7, 1);
        let mut cin = 64;
        for (i, (&s, &k)) in cfg.upsample_factors.iter().zip(&cfg.upsample_kernel_sizes).enumerate() {
            let cout = cin / 2;
            let (w, b) = (get(&map, &format!("upsamplers.{i}.weight"), &[cin, cout, k]), get(&map, &format!("upsamplers.{i}.bias"), &[cout]));
            x = conv_transpose1d(&leaky(&x, 0.1), &w, &b, k, s, (k - s) / 2);
            let mut mean = vec![0f32; x.v.len()];
            for j in 0..3 {
                let p = format!("resnets.{}", i * 3 + j);
                let ks = cfg.resnet_kernel_sizes[j];
                let mut y = x.clone();
                for n in 0..3 {
                    let d = cfg.resnet_dilations[j][n];
                    let (w1, b1) = load(&format!("{p}.convs1.{n}"), cout, cout, ks);
                    let (w2, b2) = load(&format!("{p}.convs2.{n}"), cout, cout, ks);
                    let h = conv1d(&leaky(&y, 0.1), &w1, &b1, ks, d);
                    let h = conv1d(&leaky(&h, 0.1), &w2, &b2, ks, 1);
                    for (a, b) in y.v.iter_mut().zip(&h.v) {
                        *a += *b;
                    }
                }
                for (a, b) in mean.iter_mut().zip(&y.v) {
                    *a += *b / 3.0;
                }
            }
            x.v = mean;
            cin = cout;
        }
        let (w, b) = load("conv_out", 2, cin, 7);
        let want: Vec<f32> = conv1d(&leaky(&x, 0.01), &w, &b, 7, 1).v.iter().map(|v| v.tanh()).collect();
        let got = got.host_cow().unwrap();
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 2e-4, "sample {i}: {a} vs {b}");
        }
        // The output activation is the soft one: slope 0.1 there would differ.
        assert!(want.iter().all(|v| v.abs() <= 1.0));
    }

    #[test]
    fn production_lengths_are_exactly_240_samples_per_mel_frame() {
        let cfg = Ltx2VocoderConfig::ltx2_19b();
        assert_eq!(cfg.waveform_samples(501), 120_240);
        for i in 0..5 {
            assert_eq!(cfg.upsample_kernel_sizes[i] - 2 * cfg.upsample_padding(i), cfg.upsample_factors[i], "stage {i}");
        }
    }

    #[test]
    fn weight_norm_folds_to_gain_times_direction() {
        // Two rows: ‖(3, 4)‖ = 5, ‖(0, 2)‖ = 2.
        let w = fold_weight_norm(&[10.0, -1.0], &[3.0, 4.0, 0.0, 2.0], 2);
        assert_eq!(w, vec![6.0, 8.0, 0.0, -1.0]);
    }

    #[test]
    fn hann_sinc_kernel_sums_near_one_and_has_expected_geometry() {
        let (taps, k, pad, pad_left, pad_right) = hann_sinc_kernel(3);
        assert_eq!((k, pad, pad_left, pad_right), (43, 7, 42, 40));
        let sum: f32 = taps.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum={sum}");
    }

    #[test]
    fn ltx25_bwe_path_loads_and_forwards() {
        let cfg = Ltx2VocoderConfig::ltx2_5_22b_bwe();
        let voc = Vocoder::load(&weights(), &cfg).expect("load BWE vocoder");
        assert!(voc.bwe.is_some(), "generated map should invent BWE weights");
        let mel = CudaTensor::zeros(&[1, 2, 2, 64]);
        let wave = voc.forward(&mel).expect("BWE forward");
        assert_eq!(wave.shape[0], 1);
        assert_eq!(wave.shape[1], 2);
        assert_eq!(wave.shape[2], cfg.waveform_samples(2));
        assert!(wave.host_cow().unwrap().iter().all(|v| v.is_finite() && *v >= -1.0 && *v <= 1.0));
    }

    #[test]
    fn a_mel_with_the_wrong_channel_count_is_refused() {
        let voc = Vocoder::load(&weights(), &Ltx2VocoderConfig { hidden_channels: 64, ..tiny() }).unwrap();
        assert!(voc.forward(&CudaTensor::zeros(&[1, 2, 4, 5])).is_err());
    }
}
