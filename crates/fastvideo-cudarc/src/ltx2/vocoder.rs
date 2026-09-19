//! `LTX2Vocoder`: a HiFi-GAN generator from the stereo log-mel to a 24 kHz
//! stereo waveform.
//!
//! The mel arrives as `[B, 2, T, 64]` and is read as 128 channels over `T`
//! frames (channel index `stereo · 64 + mel_bin`). Five transposed convolutions
//! upsample ×6·5·2·2·2 = ×240; every stage satisfies `kernel - 2·pad = stride`,
//! so the lengths are exact multiples with no output padding. After each one,
//! three residual blocks with kernels 3, 7 and 11 run **in parallel on the same
//! input** and are averaged — not chained.
//!
//! Two constants are not what the config suggests: the activation before
//! `conv_out` is a bare `nn.LeakyReLU()` — slope **0.01**, not the 0.1 used
//! everywhere else — and LTX-2.0 has neither Snake activations nor the
//! anti-aliased resampling around them (that is LTX-2.3's vocoder).
//!
//! Weight norm: the published diffusers checkpoint stores plain weights. An
//! original HiFi-GAN checkpoint stores `weight_g` / `weight_v` (or torch's
//! `parametrizations.weight.original{0,1}`); those are folded to
//! `g · v / ‖v‖` once at load, so the graph never sees them.
//! See docs/ports/ltx2.md §d.

use fastvideo_models::ltx2::config::Ltx2VocoderConfig;

use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

use super::{msg, tanh};

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
    fn load(map: &WeightMap, prefix: &str, cin: usize, cout: usize, kernel: usize, dilation: usize) -> Result<Self> {
        if kernel.is_multiple_of(2) {
            return Err(msg(format!("{prefix}: \"same\" padding needs an odd kernel, got {kernel}")));
        }
        let mut bias = cuda_tensor_shaped(map, &format!("{prefix}.bias"), &[cout])?;
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
/// one, LeakyReLU before each, added back.
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

pub struct Vocoder {
    cfg: Ltx2VocoderConfig,
    conv_in: Conv1d,
    /// Per upsample stage: the transposed conv and its parallel residual blocks.
    stages: Vec<(Upsampler, Vec<ResBlock>)>,
    conv_out: Conv1d,
}

impl Vocoder {
    /// `map` is the diffusers `vocoder/` folder.
    pub fn load(map: &WeightMap, cfg: &Ltx2VocoderConfig) -> Result<Self> {
        let per_stage = cfg.resnet_kernel_sizes.len();
        let mut stages = Vec::with_capacity(cfg.upsample_factors.len());
        let mut cin = cfg.hidden_channels;
        for (i, (&stride, &kernel)) in cfg.upsample_factors.iter().zip(&cfg.upsample_kernel_sizes).enumerate() {
            let cout = cfg.stage_channels(i);
            if kernel < stride || cout == 0 || cout * 2 != cin {
                return Err(msg(format!("vocoder stage {i}: kernel {kernel}, stride {stride}, {cin} -> {cout} channels")));
            }
            let prefix = format!("upsamplers.{i}");
            let mut bias = cuda_tensor_shaped(map, &format!("{prefix}.bias"), &[cout])?;
            bias.pin_device()?;
            let up = Upsampler { weight: conv_weight(map, &prefix, &[cin, cout, kernel])?, bias, stride, padding: cfg.upsample_padding(i) };
            let blocks = cfg
                .resnet_kernel_sizes
                .iter()
                .zip(&cfg.resnet_dilations)
                .enumerate()
                .map(|(j, (&k, dilations))| {
                    let p = format!("resnets.{}", i * per_stage + j);
                    let convs = dilations
                        .iter()
                        .enumerate()
                        .map(|(n, &d)| Ok((Conv1d::load(map, &format!("{p}.convs1.{n}"), cout, cout, k, d)?, Conv1d::load(map, &format!("{p}.convs2.{n}"), cout, cout, k, 1)?)))
                        .collect::<Result<Vec<_>>>()?;
                    Ok(ResBlock { convs })
                })
                .collect::<Result<Vec<_>>>()?;
            stages.push((up, blocks));
            cin = cout;
        }
        Ok(Self {
            conv_in: Conv1d::load(map, "conv_in", cfg.in_channels, cfg.hidden_channels, 7, 1)?,
            stages,
            conv_out: Conv1d::load(map, "conv_out", cin, cfg.out_channels, 7, 1)?,
            cfg: cfg.clone(),
        })
    }

    /// Mel `[B, 2, T, 64]` → waveform `[B, 2, 240·T]` in `[-1, 1]` at
    /// `output_sampling_rate`.
    pub fn forward(&self, mel: &CudaTensor) -> Result<CudaTensor> {
        let [b, c, t, m] = mel.shape[..] else {
            return Err(msg(format!("vocoder expects mel [B, C, T, M], got {:?}", mel.shape)));
        };
        if c * m != self.cfg.in_channels || t == 0 {
            return Err(msg(format!("vocoder expects {} mel channels, got {c} x {m} in {:?}", self.cfg.in_channels, mel.shape)));
        }
        let slope = self.cfg.leaky_relu_negative_slope as f32;
        // [B, C, T, M] → [B, C, M, T] → [B, C·M, T].
        let mut x = self.conv_in.forward(&mel.permute(&[0, 1, 3, 2])?.reshape(vec![b, c * m, t])?)?;
        for (up, blocks) in &self.stages {
            x = x.leaky_relu(slope).conv_transpose1d(&up.weight, Some(&up.bias), up.padding, up.stride, 1, 1, 0)?;
            let outs = blocks.iter().map(|r| r.forward(&x, slope)).collect::<Result<Vec<_>>>()?;
            let share = 1.0 / outs.len() as f32;
            x = CudaTensor::lincomb(&outs.iter().map(|o| (share, o)).collect::<Vec<_>>())?;
        }
        let x = self.conv_out.forward(&x.leaky_relu(self.cfg.final_leaky_relu_negative_slope as f32))?;
        if self.cfg.final_tanh {
            tanh(&x)
        } else {
            Ok(x)
        }
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
            upsample_kernel_sizes: [7, 4, 4, 4, 4],
            upsample_factors: [3, 2, 2, 2, 2],
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
            for (j, &rk) in [3usize, 7, 11].iter().enumerate() {
                let mut r = x.clone();
                for (n, d) in [1usize, 3, 5].into_iter().enumerate() {
                    let p = format!("resnets.{}", i * 3 + j);
                    let (w1, b1) = load(&format!("{p}.convs1.{n}"), cout, cout, rk);
                    let (w2, b2) = load(&format!("{p}.convs2.{n}"), cout, cout, rk);
                    let h = conv1d(&leaky(&conv1d(&leaky(&r, 0.1), &w1, &b1, rk, d), 0.1), &w2, &b2, rk, 1);
                    r.v.iter_mut().zip(&h.v).for_each(|(a, b)| *a += b);
                }
                mean.iter_mut().zip(&r.v).for_each(|(a, b)| *a += b / 3.0);
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
    fn a_mel_with_the_wrong_channel_count_is_refused() {
        let voc = Vocoder::load(&weights(), &Ltx2VocoderConfig { hidden_channels: 64, ..tiny() }).unwrap();
        assert!(voc.forward(&CudaTensor::zeros(&[1, 2, 4, 5])).is_err());
    }
}
