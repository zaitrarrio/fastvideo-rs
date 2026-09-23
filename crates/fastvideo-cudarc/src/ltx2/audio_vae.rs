//! `AutoencoderKLLTX2Audio`: the decoder turns audio latents into a stereo
//! log-mel spectrogram (16 kHz, hop 160, 64 bins) for the vocoder, and the
//! encoder is `encode_audio` (resample, slaney log-mel, causal `AudioEncoder`).
//!
//! A small 2-D conv net over `[B, C, time, mel]` in which only the **time**
//! axis is causal (`causality_axis = "height"`): every 3×3 conv pads time with
//! two zero rows in front and none behind, and mel with one bin on each side.
//! Causality is also why each ×2 upsample drops its first time row — nearest
//! doubling makes `2T` rows of which the first is a duplicate the causal conv
//! cannot tell apart — so `T → 2T - 1` per stage and `L → 4L - 3` overall.
//!
//! Norms are `PixelNorm`: an RMS over the channel axis at each (time, mel)
//! cell with no learned weight and eps 1e-6, which is the backend's channel
//! RMS kernel with γ = 1 (and its fused SiLU, since SiLU always follows).
//!
//! The latent statistics are per *packed feature* — one mean/std for each
//! (channel, mel-bin) pair — so de-normalisation happens on the DiT's
//! `[B, L, 128]` layout, before unpacking. See docs/ports/ltx2.md §d.

use fastvideo_models::ltx2::config::Ltx2AudioVaeConfig;

use crate::wan::ops::PadMode;
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

use super::{msg, ones};

/// `LTX2AudioCausalConv2d` with a square kernel, stride 1.
struct CausalConv2d {
    weight: CudaTensor,
    bias: CudaTensor,
    kernel: usize,
}

impl CausalConv2d {
    fn load(map: &WeightMap, prefix: &str, cin: usize, cout: usize, kernel: usize) -> Result<Self> {
        let mut weight = cuda_tensor_shaped(
            map,
            &format!("{prefix}.conv.weight"),
            &[cout, cin, kernel, kernel],
        )?;
        let mut bias = cuda_tensor_shaped(map, &format!("{prefix}.conv.bias"), &[cout])?;
        weight.pin_device()?;
        bias.pin_device()?;
        Ok(Self {
            weight,
            bias,
            kernel,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let reach = self.kernel - 1;
        // F.pad(x, (w//2, w - w//2, h, 0)): mel symmetric, time in front only.
        let x = x
            .pad(3, reach / 2, reach - reach / 2, PadMode::Zeros)?
            .pad(2, reach, 0, PadMode::Zeros)?;
        x.conv2d(&self.weight, Some(&self.bias), 0, 1)
    }
}

struct Resnet {
    conv1: CausalConv2d,
    conv2: CausalConv2d,
    /// 1×1 projection of the skip path when the channel count changes.
    shortcut: Option<CausalConv2d>,
    ones_in: CudaTensor,
    ones_out: CudaTensor,
}

impl Resnet {
    fn load(map: &WeightMap, prefix: &str, cin: usize, cout: usize) -> Result<Self> {
        Ok(Self {
            conv1: CausalConv2d::load(map, &format!("{prefix}.conv1"), cin, cout, 3)?,
            conv2: CausalConv2d::load(map, &format!("{prefix}.conv2"), cout, cout, 3)?,
            shortcut: if cin == cout {
                None
            } else {
                Some(CausalConv2d::load(
                    map,
                    &format!("{prefix}.nin_shortcut"),
                    cin,
                    cout,
                    1,
                )?)
            },
            ones_in: ones(cin)?,
            ones_out: ones(cout)?,
        })
    }

    fn forward(&self, x: &CudaTensor, eps: f32) -> Result<CudaTensor> {
        let h = self
            .conv1
            .forward(&x.rms_norm_channels_act(&self.ones_in, eps, true)?)?;
        let h = self
            .conv2
            .forward(&h.rms_norm_channels_act(&self.ones_out, eps, true)?)?;
        match &self.shortcut {
            Some(s) => s.forward(x)?.add(&h),
            None => x.add(&h),
        }
    }
}

struct Level {
    blocks: Vec<Resnet>,
    upsample: Option<CausalConv2d>,
}

pub struct AudioDecoder {
    cfg: Ltx2AudioVaeConfig,
    /// `[token_channels]` each: per (channel, mel-bin) statistics.
    latents_mean: CudaTensor,
    latents_std: CudaTensor,
    conv_in: CausalConv2d,
    mid: [Resnet; 2],
    /// In execution order: the deepest level (`up.{n-1}`) first.
    levels: Vec<Level>,
    ones_out: CudaTensor,
    conv_out: CausalConv2d,
}

impl AudioDecoder {
    /// `map` is the diffusers `audio_vae/` folder.
    pub fn load(map: &WeightMap, cfg: &Ltx2AudioVaeConfig) -> Result<Self> {
        if !cfg.causal_time_axis || cfg.mid_block_add_attention {
            return Err(msg(
                "audio vae: only the time-causal, attention-free decoder of LTX-2.0 is supported",
            ));
        }
        let levels_n = cfg.ch_mult.len();
        let top = cfg.base_channels * cfg.ch_mult[levels_n - 1];
        let mut levels = Vec::with_capacity(levels_n);
        let mut cin = top;
        for level in (0..levels_n).rev() {
            let cout = cfg.base_channels * cfg.ch_mult[level];
            let blocks = (0..=cfg.num_res_blocks)
                .map(|i| {
                    Resnet::load(
                        map,
                        &format!("decoder.up.{level}.block.{i}"),
                        if i == 0 { cin } else { cout },
                        cout,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let upsample = if level == 0 {
                None
            } else {
                Some(CausalConv2d::load(
                    map,
                    &format!("decoder.up.{level}.upsample.conv"),
                    cout,
                    cout,
                    3,
                )?)
            };
            levels.push(Level { blocks, upsample });
            cin = cout;
        }
        let features = cfg.token_channels();
        let stat = |key: &str| -> Result<CudaTensor> {
            let mut t = cuda_tensor_shaped(map, key, &[features])?;
            t.pin_device()?;
            Ok(t)
        };
        Ok(Self {
            latents_mean: stat("latents_mean")?,
            latents_std: stat("latents_std")?,
            conv_in: CausalConv2d::load(map, "decoder.conv_in", cfg.latent_channels, top, 3)?,
            mid: [
                Resnet::load(map, "decoder.mid.block_1", top, top)?,
                Resnet::load(map, "decoder.mid.block_2", top, top)?,
            ],
            levels,
            ones_out: ones(cin)?,
            conv_out: CausalConv2d::load(map, "decoder.conv_out", cin, cfg.output_channels, 3)?,
            cfg: cfg.clone(),
        })
    }

    /// The DiT's packed, normalised audio latent `[1, L, 128]` → mel
    /// `[1, 2, 4L - 3, 64]`: de-normalise, unpack, decode.
    pub fn decode_packed(&self, packed: &CudaTensor) -> Result<CudaTensor> {
        let (c, m) = (self.cfg.latent_channels, self.cfg.latent_mel_bins());
        let [b, l, f] = packed.shape[..] else {
            return Err(msg(format!(
                "audio latent must be packed [B, L, {}], got {:?}",
                c * m,
                packed.shape
            )));
        };
        if f != c * m || l == 0 {
            return Err(msg(format!(
                "audio latent {:?}: expected {} features per frame",
                packed.shape,
                c * m
            )));
        }
        let z = packed.mul(&self.latents_std)?.add(&self.latents_mean)?;
        // Feature index is channel · bins + bin: [B, L, C, M] → [B, C, L, M].
        self.decode(&z.reshape(vec![b, l, c, m])?.permute(&[0, 2, 1, 3])?)
    }

    /// De-normalised latent `[B, 8, L, 16]` → mel `[B, 2, 4L - 3, 64]`.
    pub fn decode(&self, z: &CudaTensor) -> Result<CudaTensor> {
        let [_, c, frames, _] = z.shape[..] else {
            return Err(msg(format!(
                "audio vae expects [B, C, L, M], got {:?}",
                z.shape
            )));
        };
        if c != self.cfg.latent_channels || frames == 0 {
            return Err(msg(format!(
                "audio vae expects {} latent channels, got {:?}",
                self.cfg.latent_channels, z.shape
            )));
        }
        let eps = self.cfg.pixel_norm_eps as f32;
        let mut x = self.conv_in.forward(z)?;
        for block in &self.mid {
            x = block.forward(&x, eps)?;
        }
        for level in &self.levels {
            for block in &level.blocks {
                x = block.forward(&x, eps)?;
            }
            if let Some(conv) = &level.upsample {
                let (t, m) = (x.shape[2], x.shape[3]);
                let y = conv.forward(&x.upsample_nearest2d(2 * t, 2 * m)?)?;
                x = y.narrow(2, 1, 2 * t - 1)?;
            }
        }
        let x = self
            .conv_out
            .forward(&x.rms_norm_channels_act(&self.ones_out, eps, true)?)?;
        // Crop, then zero-pad, to the nominal `[4L - 3, mel_bins]`. With the
        // published three levels the decoder already lands there exactly.
        let (want_t, want_m) = (self.cfg.mel_frames(frames), self.cfg.mel_bins);
        let (t, m) = (x.shape[2].min(want_t), x.shape[3].min(want_m));
        x.narrow(2, 0, t)?
            .narrow(3, 0, m)?
            .pad(2, 0, want_t - t, PadMode::Zeros)?
            .pad(3, 0, want_m - m, PadMode::Zeros)
    }
}

/// `Downsample` on the time axis (`CausalityAxis::HEIGHT`): pad
/// `(left, right, top, bottom) = (0, 1, 2, 0)`, then a 3×3 stride-2 conv.
struct Downsample {
    weight: CudaTensor,
    bias: CudaTensor,
}

impl Downsample {
    fn load(map: &WeightMap, prefix: &str, channels: usize) -> Result<Self> {
        let mut weight = cuda_tensor_shaped(
            map,
            &format!("{prefix}.conv.weight"),
            &[channels, channels, 3, 3],
        )?;
        let mut bias = cuda_tensor_shaped(map, &format!("{prefix}.conv.bias"), &[channels])?;
        weight.pin_device()?;
        bias.pin_device()?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let x = x
            .pad(3, 0, 1, PadMode::Zeros)?
            .pad(2, 2, 0, PadMode::Zeros)?;
        x.conv2d(&self.weight, Some(&self.bias), 0, 2)
    }
}

struct EncLevel {
    blocks: Vec<Resnet>,
    downsample: Option<Downsample>,
}

/// `ltx_core.model.audio_vae.AudioEncoder`. `double_z` keeps the mean half,
/// then patchify → per-feature normalize → unpatchify.
pub struct AudioEncoder {
    cfg: Ltx2AudioVaeConfig,
    latents_mean: CudaTensor,
    latents_std: CudaTensor,
    conv_in: CausalConv2d,
    levels: Vec<EncLevel>,
    mid: [Resnet; 2],
    ones_out: CudaTensor,
    conv_out: CausalConv2d,
}

impl AudioEncoder {
    /// `map` is the diffusers `audio_vae/` folder, or a single file whose keys
    /// use that layout (or an `audio_vae.` prefix).
    pub fn load(map: &WeightMap, cfg: &Ltx2AudioVaeConfig) -> Result<Self> {
        if !cfg.causal_time_axis || cfg.mid_block_add_attention || !cfg.double_z {
            return Err(msg(
                "audio vae: only the time-causal, attention-free, double_z encoder of LTX-2 is supported",
            ));
        }
        let root = encoder_prefix(map);
        let levels_n = cfg.ch_mult.len();
        let mut levels = Vec::with_capacity(levels_n);
        let mut block_in = cfg.base_channels;
        for level in 0..levels_n {
            let cin = cfg.base_channels * in_ch_mult(cfg, level);
            let cout = cfg.base_channels * cfg.ch_mult[level];
            let mut blocks = Vec::with_capacity(cfg.num_res_blocks);
            let mut ch = cin;
            for i in 0..cfg.num_res_blocks {
                blocks.push(Resnet::load(
                    map,
                    &format!("{root}encoder.down.{level}.block.{i}"),
                    ch,
                    cout,
                )?);
                ch = cout;
            }
            let downsample = if level + 1 == levels_n {
                None
            } else {
                Some(Downsample::load(
                    map,
                    &format!("{root}encoder.down.{level}.downsample"),
                    cout,
                )?)
            };
            levels.push(EncLevel { blocks, downsample });
            block_in = cout;
        }
        let z_out = cfg.latent_channels * 2;
        let features = cfg.token_channels();
        let stat = |key: &str| -> Result<CudaTensor> {
            let mut t = cuda_tensor_shaped(map, &stat_key(map, key), &[features])?;
            t.pin_device()?;
            Ok(t)
        };
        Ok(Self {
            latents_mean: stat("latents_mean")?,
            latents_std: stat("latents_std")?,
            conv_in: CausalConv2d::load(
                map,
                &format!("{root}encoder.conv_in"),
                cfg.in_channels,
                cfg.base_channels,
                3,
            )?,
            levels,
            mid: [
                Resnet::load(
                    map,
                    &format!("{root}encoder.mid.block_1"),
                    block_in,
                    block_in,
                )?,
                Resnet::load(
                    map,
                    &format!("{root}encoder.mid.block_2"),
                    block_in,
                    block_in,
                )?,
            ],
            ones_out: ones(block_in)?,
            conv_out: CausalConv2d::load(
                map,
                &format!("{root}encoder.conv_out"),
                block_in,
                z_out,
                3,
            )?,
            cfg: cfg.clone(),
        })
    }

    /// Log-mel `[B, C, time, mel]` → normalized latent `[B, latent_channels, L, mel/4]`.
    pub fn encode_spectrogram(&self, spectrogram: &CudaTensor) -> Result<CudaTensor> {
        let [_, c, _, _] = spectrogram.shape[..] else {
            return Err(msg(format!(
                "audio encode expects [B, C, T, M], got {:?}",
                spectrogram.shape
            )));
        };
        if c != self.cfg.in_channels {
            return Err(msg(format!(
                "audio encode expects {} spectrogram channels, got {:?}",
                self.cfg.in_channels, spectrogram.shape
            )));
        }
        let eps = self.cfg.pixel_norm_eps as f32;
        let mut h = self.conv_in.forward(spectrogram)?;
        for level in &self.levels {
            for block in &level.blocks {
                h = block.forward(&h, eps)?;
            }
            if let Some(down) = &level.downsample {
                h = down.forward(&h)?;
            }
        }
        for block in &self.mid {
            h = block.forward(&h, eps)?;
        }
        let h = self
            .conv_out
            .forward(&h.rms_norm_channels_act(&self.ones_out, eps, true)?)?;
        self.normalize_latents(&h)
    }

    /// Planar channel-major waveform → the normalized latent `encode_audio` returns.
    pub fn encode_waveform(
        &self,
        planar: &[f32],
        channels: usize,
        sample_rate: u32,
    ) -> Result<CudaTensor> {
        if channels == 0 || channels != self.cfg.in_channels {
            return Err(msg(format!(
                "audio encode expects {} channels, got {channels}",
                self.cfg.in_channels
            )));
        }
        let mel = log_mel(planar, channels, sample_rate, &self.cfg)?;
        self.encode_spectrogram(&mel)
    }

    /// `torch.chunk(h, 2, dim=1)[0]`, then `"b c t f -> b t (c f)"`, normalize, unpatchify.
    fn normalize_latents(&self, h: &CudaTensor) -> Result<CudaTensor> {
        let [b, c, t, m] = h.shape[..] else {
            return Err(msg(format!(
                "audio encoder output must be [B, C, T, M], got {:?}",
                h.shape
            )));
        };
        let keep = self.cfg.latent_channels;
        if c != keep * 2 || m != self.cfg.latent_mel_bins() || t == 0 {
            return Err(msg(format!(
                "audio encoder output {:?}: expected [B, {}, T, {}]",
                h.shape,
                keep * 2,
                self.cfg.latent_mel_bins()
            )));
        }
        let means = h.narrow(1, 0, keep)?;
        let packed = means
            .permute(&[0, 2, 1, 3])?
            .reshape(vec![b, t, keep * m])?;
        let normed = packed.sub(&self.latents_mean)?.div(&self.latents_std)?;
        normed.reshape(vec![b, t, keep, m])?.permute(&[0, 2, 1, 3])
    }
}

/// `encode_source_audio`: copy the time prefix into `frames`, zero the tail.
pub fn conform_audio_time(z: &CudaTensor, frames: usize) -> Result<CudaTensor> {
    let [b, c, t, m] = z.shape[..] else {
        return Err(msg(format!(
            "audio latent must be [B, C, T, M], got {:?}",
            z.shape
        )));
    };
    if frames == 0 {
        return Err(msg("audio conform: no target frames"));
    }
    if t == frames {
        return Ok(z.clone());
    }
    let src = z.host_cow()?;
    let n = t.min(frames);
    let mut out = vec![0f32; b * c * frames * m];
    for bc in 0..b * c {
        for ti in 0..n {
            let from = (bc * t + ti) * m;
            let to = (bc * frames + ti) * m;
            out[to..to + m].copy_from_slice(&src[from..from + m]);
        }
    }
    CudaTensor::from_vec(out, vec![b, c, frames, m])
}

/// Packed DiT audio `[B, T, C·M]` from a normalized `[B, C, T, M]` latent.
pub fn pack_audio_latent(z: &CudaTensor) -> Result<CudaTensor> {
    let [b, c, t, m] = z.shape[..] else {
        return Err(msg(format!(
            "pack_audio expects [B, C, T, M], got {:?}",
            z.shape
        )));
    };
    z.permute(&[0, 2, 1, 3])?.reshape(vec![b, t, c * m])
}

fn in_ch_mult(cfg: &Ltx2AudioVaeConfig, level: usize) -> usize {
    if level == 0 {
        1
    } else {
        cfg.ch_mult[level - 1]
    }
}

fn encoder_prefix(map: &WeightMap) -> &'static str {
    if map.has_tensor("audio_vae.encoder.conv_in.conv.weight")
        && !map.has_tensor("encoder.conv_in.conv.weight")
    {
        "audio_vae."
    } else {
        ""
    }
}

fn stat_key(map: &WeightMap, plain: &str) -> String {
    if map.has_tensor(plain) {
        return plain.to_string();
    }
    let nested = format!("audio_vae.{plain}");
    if map.has_tensor(&nested) {
        return nested;
    }
    let legacy = match plain {
        "latents_mean" => "mean-of-means",
        "latents_std" => "std-of-means",
        _ => "",
    };
    if !legacy.is_empty() && map.has_tensor(legacy) {
        return legacy.to_string();
    }
    plain.to_string()
}

/// `torchaudio.functional.resample`, `sinc_interp_hann`, width 6, rolloff 0.99.
fn sinc_resample(samples: &[f32], orig_freq: u32, new_freq: u32) -> Result<Vec<f32>> {
    if samples.is_empty() || orig_freq == 0 || new_freq == 0 {
        return Err(msg("audio resample: empty waveform or zero rate"));
    }
    if orig_freq == new_freq {
        return Ok(samples.to_vec());
    }
    let gcd = gcd_u32(orig_freq, new_freq);
    let orig_r = orig_freq / gcd;
    let new_r = new_freq / gcd;
    let lowpass = 6.0f32;
    let base = (orig_r.min(new_r) as f32) * 0.99;
    let width = (lowpass * orig_r as f32 / base).ceil() as usize;
    let k = 2 * width + orig_r as usize;
    let mut kernel = vec![0f32; new_r as usize * k];
    for j in 0..new_r as usize {
        let t0 = -(j as f32) / new_r as f32;
        for i in 0..k {
            let idx = (i as isize - width as isize) as f32 / orig_r as f32;
            let mut t = (t0 + idx) * base;
            t = t.clamp(-lowpass, lowpass);
            let window = (t * std::f32::consts::PI / lowpass / 2.0).cos().powi(2);
            t *= std::f32::consts::PI;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernel[j * k + i] = sinc * window * (base / orig_r as f32);
        }
    }
    let left = width;
    let right = width + orig_r as usize;
    let padded_len = samples.len() + left + right;
    if padded_len < k {
        return Err(msg("audio resample: waveform shorter than the kernel"));
    }
    let mut padded = vec![0f32; padded_len];
    padded[left..left + samples.len()].copy_from_slice(samples);
    let stride = orig_r as usize;
    let conv_len = (padded_len - k) / stride + 1;
    let mut out = vec![0f32; conv_len * new_r as usize];
    for n in 0..conv_len {
        let start = n * stride;
        for j in 0..new_r as usize {
            let mut acc = 0f32;
            for i in 0..k {
                acc += padded[start + i] * kernel[j * k + i];
            }
            out[n * new_r as usize + j] = acc;
        }
    }
    let target = (u64::from(new_r) * samples.len() as u64).div_ceil(u64::from(orig_r)) as usize;
    out.truncate(target);
    Ok(out)
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Reflect about the edge without repeating it. `[-1]` of `[a,b,c,d]` is `b`.
fn reflect_at(samples: &[f32], index: isize) -> f32 {
    let n = samples.len() as isize;
    if n <= 1 {
        return samples.first().copied().unwrap_or(0.0);
    }
    let period = 2 * (n - 1);
    let mut i = index.rem_euclid(period);
    if i >= n {
        i = period - i;
    }
    samples[i as usize]
}

fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos())
        .collect()
}

/// Unnormalized radix-2 DFT, the `normalized=False` torch.stft sign.
fn fft_radix2(input: &[f32]) -> Vec<(f32, f32)> {
    let n = input.len();
    let mut re = input.to_vec();
    let mut im = vec![0f32; n];
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -std::f32::consts::TAU / len as f32;
        let (wlen_re, wlen_im) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut w_re, mut w_im) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let (u_re, u_im) = (re[i + k], im[i + k]);
                let (v_re, v_im) = (
                    re[i + k + len / 2] * w_re - im[i + k + len / 2] * w_im,
                    re[i + k + len / 2] * w_im + im[i + k + len / 2] * w_re,
                );
                re[i + k] = u_re + v_re;
                im[i + k] = u_im + v_im;
                re[i + k + len / 2] = u_re - v_re;
                im[i + k + len / 2] = u_im - v_im;
                let next_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_re;
            }
            i += len;
        }
        len <<= 1;
    }
    re.into_iter().zip(im).collect()
}

fn hz_to_mel_slaney(freq: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f32.ln() / 27.0;
    if freq >= min_log_hz {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    } else {
        freq / f_sp
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f32.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_sp * mel
    }
}

fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    (0..n)
        .map(|i| start + (end - start) * i as f32 / (n - 1) as f32)
        .collect()
}

/// Slaney mel filterbank, shape `[n_freqs, n_mels]`, column-major filters.
fn mel_fbanks(n_freqs: usize, n_mels: usize, sample_rate: u32, f_min: f32, f_max: f32) -> Vec<f32> {
    let all_freqs = linspace(0.0, (sample_rate / 2) as f32, n_freqs);
    let m_min = hz_to_mel_slaney(f_min);
    let m_max = hz_to_mel_slaney(f_max);
    let m_pts = linspace(m_min, m_max, n_mels + 2);
    let f_pts: Vec<f32> = m_pts.iter().copied().map(mel_to_hz_slaney).collect();
    let mut fb = vec![0f32; n_freqs * n_mels];
    for fi in 0..n_freqs {
        for mi in 0..n_mels {
            let down = (all_freqs[fi] - f_pts[mi]) / (f_pts[mi + 1] - f_pts[mi]);
            let up = (f_pts[mi + 2] - all_freqs[fi]) / (f_pts[mi + 2] - f_pts[mi + 1]);
            fb[fi * n_mels + mi] = down.min(up).max(0.0);
        }
    }
    for mi in 0..n_mels {
        let enorm = 2.0 / (f_pts[mi + 2] - f_pts[mi]);
        for fi in 0..n_freqs {
            fb[fi * n_mels + mi] *= enorm;
        }
    }
    fb
}

const N_FFT: usize = 1024;
const MEL_FLOOR: f32 = 1e-5;

/// `AudioProcessor.waveform_to_mel`: resample, magnitude STFT, slaney log-mel,
/// `[1, C, time, n_mels]`.
fn log_mel(
    planar: &[f32],
    channels: usize,
    sample_rate: u32,
    cfg: &Ltx2AudioVaeConfig,
) -> Result<CudaTensor> {
    if channels == 0 || planar.len() % channels != 0 {
        return Err(msg(format!(
            "audio mel: {} samples is not {} channels",
            planar.len(),
            channels
        )));
    }
    let n = planar.len() / channels;
    let target = u32::try_from(cfg.sample_rate).map_err(|_| msg("audio mel: sample rate"))?;
    let hop = cfg.mel_hop_length;
    let n_mels = cfg.mel_bins;
    if hop == 0 || n_mels == 0 || !N_FFT.is_power_of_two() {
        return Err(msg("audio mel: bad frontend config"));
    }
    let mut resampled = Vec::with_capacity(channels);
    for c in 0..channels {
        let start = c * n;
        resampled.push(sinc_resample(
            &planar[start..start + n],
            sample_rate,
            target,
        )?);
    }
    let frames_n = resampled[0].len();
    let n_frames = 1 + frames_n / hop;
    let n_freqs = N_FFT / 2 + 1;
    let window = hann_periodic(N_FFT);
    let fb = mel_fbanks(n_freqs, n_mels, target, 0.0, target as f32 / 2.0);
    let mut mel = vec![0f32; channels * n_frames * n_mels];
    for c in 0..channels {
        let wave = &resampled[c];
        for frame in 0..n_frames {
            let mut windowed = vec![0f32; N_FFT];
            let start = frame * hop;
            for i in 0..N_FFT {
                let index = start as isize + i as isize - (N_FFT / 2) as isize;
                windowed[i] = reflect_at(wave, index) * window[i];
            }
            let spec = fft_radix2(&windowed);
            for mi in 0..n_mels {
                let mut acc = 0f32;
                for fi in 0..n_freqs {
                    let mag = spec[fi].0.hypot(spec[fi].1);
                    acc += mag * fb[fi * n_mels + mi];
                }
                mel[(c * n_frames + frame) * n_mels + mi] = acc.max(MEL_FLOOR).ln();
            }
        }
    }
    let mut out = CudaTensor::from_vec(mel, vec![1, channels, n_frames, n_mels])?;
    out.pin_device()?;
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::attention::tests::{get, weights};
    use super::*;

    /// `[c, h, w]` feature map for the loop reference.
    #[derive(Clone)]
    pub(crate) struct Map3 {
        pub c: usize,
        pub h: usize,
        pub w: usize,
        pub v: Vec<f32>,
    }

    impl Map3 {
        fn at(&self, c: usize, y: isize, x: isize) -> f32 {
            if y < 0 || x < 0 || y as usize >= self.h || x as usize >= self.w {
                0.0
            } else {
                self.v[(c * self.h + y as usize) * self.w + x as usize]
            }
        }
    }

    /// k×k conv with zero padding `top` rows above, `side` columns each side,
    /// nothing below: the time-causal layout.
    fn causal_conv(x: &Map3, w: &[f32], b: &[f32], k: usize) -> Map3 {
        let (top, side) = ((k - 1) as isize, ((k - 1) / 2) as isize);
        let cout = b.len();
        let mut v = vec![0f32; cout * x.h * x.w];
        for o in 0..cout {
            for y in 0..x.h {
                for xx in 0..x.w {
                    let mut acc = b[o];
                    for i in 0..x.c {
                        for ky in 0..k {
                            for kx in 0..k {
                                acc += w[((o * x.c + i) * k + ky) * k + kx]
                                    * x.at(
                                        i,
                                        y as isize + ky as isize - top,
                                        xx as isize + kx as isize - side,
                                    );
                            }
                        }
                    }
                    v[(o * x.h + y) * x.w + xx] = acc;
                }
            }
        }
        Map3 {
            c: cout,
            h: x.h,
            w: x.w,
            v,
        }
    }

    fn pixel_norm_silu(x: &Map3, eps: f32) -> Map3 {
        let mut out = x.clone();
        for p in 0..x.h * x.w {
            let ms = (0..x.c)
                .map(|c| x.v[c * x.h * x.w + p].powi(2))
                .sum::<f32>()
                / x.c as f32;
            for c in 0..x.c {
                let n = x.v[c * x.h * x.w + p] / (ms + eps).sqrt();
                out.v[c * x.h * x.w + p] = n / (1.0 + (-n).exp());
            }
        }
        out
    }

    fn conv(map: &WeightMap, prefix: &str, x: &Map3, cout: usize, k: usize) -> Map3 {
        causal_conv(
            x,
            &get(map, &format!("{prefix}.conv.weight"), &[cout, x.c, k, k]),
            &get(map, &format!("{prefix}.conv.bias"), &[cout]),
            k,
        )
    }

    fn resnet(map: &WeightMap, prefix: &str, x: &Map3, cout: usize) -> Map3 {
        let h = conv(
            map,
            &format!("{prefix}.conv1"),
            &pixel_norm_silu(x, 1e-6),
            cout,
            3,
        );
        let h = conv(
            map,
            &format!("{prefix}.conv2"),
            &pixel_norm_silu(&h, 1e-6),
            cout,
            3,
        );
        let skip = if x.c == cout {
            x.clone()
        } else {
            conv(map, &format!("{prefix}.nin_shortcut"), x, cout, 1)
        };
        Map3 {
            v: skip.v.iter().zip(&h.v).map(|(a, b)| a + b).collect(),
            ..h
        }
    }

    fn tiny() -> Ltx2AudioVaeConfig {
        Ltx2AudioVaeConfig {
            base_channels: 2,
            ch_mult: [1, 2, 4],
            num_res_blocks: 1,
            latent_channels: 2,
            mel_bins: 8,
            ..Ltx2AudioVaeConfig::ltx2_19b()
        }
    }

    #[test]
    fn decoder_matches_a_loop_reference_and_grows_time_causally() {
        let cfg = tiny();
        let map = weights();
        let dec = AudioDecoder::load(&map, &cfg).unwrap();
        let (l, m) = (3usize, cfg.latent_mel_bins());
        assert_eq!(m, 2);
        let packed: Vec<f32> = (0..l * 2 * m).map(|i| (i as f32 * 0.9).sin()).collect();
        let got = dec
            .decode_packed(&CudaTensor::from_vec(packed.clone(), vec![1, l, 2 * m]).unwrap())
            .unwrap();
        assert_eq!(got.shape, vec![1, 2, 4 * l - 3, 8]);

        // De-normalise per packed feature, then unpack feature = channel·M + bin.
        let (mean, std) = (
            get(&map, "latents_mean", &[2 * m]),
            get(&map, "latents_std", &[2 * m]),
        );
        let mut z = Map3 {
            c: 2,
            h: l,
            w: m,
            v: vec![0.0; 2 * l * m],
        };
        for t in 0..l {
            for f in 0..2 * m {
                z.v[((f / m) * l + t) * m + f % m] = packed[t * 2 * m + f] * std[f] + mean[f];
            }
        }
        let mut x = conv(&map, "decoder.conv_in", &z, 8, 3);
        x = resnet(&map, "decoder.mid.block_1", &x, 8);
        x = resnet(&map, "decoder.mid.block_2", &x, 8);
        for (level, cout) in [(2usize, 8usize), (1, 4), (0, 2)] {
            for i in 0..2 {
                x = resnet(&map, &format!("decoder.up.{level}.block.{i}"), &x, cout);
            }
            if level != 0 {
                let mut up = Map3 {
                    c: x.c,
                    h: 2 * x.h,
                    w: 2 * x.w,
                    v: vec![0.0; x.c * 4 * x.h * x.w],
                };
                for c in 0..x.c {
                    for y in 0..up.h {
                        for xx in 0..up.w {
                            up.v[(c * up.h + y) * up.w + xx] =
                                x.at(c, (y / 2) as isize, (xx / 2) as isize);
                        }
                    }
                }
                let y = conv(
                    &map,
                    &format!("decoder.up.{level}.upsample.conv"),
                    &up,
                    cout,
                    3,
                );
                // Drop the first time row.
                let mut v = Vec::with_capacity(y.c * (y.h - 1) * y.w);
                for c in 0..y.c {
                    v.extend_from_slice(&y.v[(c * y.h + 1) * y.w..(c + 1) * y.h * y.w]);
                }
                x = Map3 {
                    c: y.c,
                    h: y.h - 1,
                    w: y.w,
                    v,
                };
            }
        }
        let want = conv(&map, "decoder.conv_out", &pixel_norm_silu(&x, 1e-6), 2, 3);
        assert_eq!((want.h, want.w), (4 * l - 3, 8));
        let got = got.host_cow().unwrap();
        for (i, (a, b)) in got.iter().zip(&want.v).enumerate() {
            assert!(
                (a - b).abs() < 1e-4 * (1.0 + b.abs()),
                "mel[{i}]: {a} vs {b}"
            );
        }
    }

    /// Time-causal: a later latent frame cannot change earlier mel frames. With
    /// `L → 4L - 3`, latent frame `j` first touches mel frame `4j - 3`.
    #[test]
    fn later_latent_frames_do_not_change_earlier_mel_frames() {
        let cfg = tiny();
        let dec = AudioDecoder::load(&weights(), &cfg).unwrap();
        let base: Vec<f32> = (0..4 * 4).map(|i| (i as f32 * 0.37).cos()).collect();
        let mut changed = base.clone();
        changed[3 * 4..].iter_mut().for_each(|v| *v += 1.0);
        let run = |v: &[f32]| {
            dec.decode_packed(&CudaTensor::from_vec(v.to_vec(), vec![1, 4, 4]).unwrap())
                .unwrap()
                .host_cow()
                .unwrap()
                .into_owned()
        };
        let (a, b) = (run(&base), run(&changed));
        let (frames, bins) = (13usize, 8usize);
        for ch in 0..2 {
            for t in 0..frames {
                let same = (0..bins).all(|m| {
                    (a[(ch * frames + t) * bins + m] - b[(ch * frames + t) * bins + m]).abs() < 1e-6
                });
                assert_eq!(
                    same,
                    t < 9,
                    "mel frame {t} (latent frame 3 starts at mel frame 9)"
                );
            }
        }
    }

    #[test]
    fn a_latent_with_the_wrong_feature_width_is_refused() {
        let dec = AudioDecoder::load(&weights(), &tiny()).unwrap();
        assert!(dec.decode_packed(&CudaTensor::zeros(&[1, 3, 5])).is_err());
        assert!(dec.decode(&CudaTensor::zeros(&[1, 3, 3, 2])).is_err());
    }

    #[test]
    fn resample_halves_a_constant_and_keeps_its_level() {
        let n = 3200usize;
        let wave = vec![1.0f32; n];
        let out = super::sinc_resample(&wave, 32_000, 16_000).unwrap();
        assert_eq!(out.len(), n.div_ceil(2));
        let mid = out[out.len() / 2];
        assert!((mid - 1.0).abs() < 1e-3, "mid sample {mid}");
    }

    #[test]
    fn silence_is_the_mel_floor() {
        let cfg = Ltx2AudioVaeConfig::ltx2_19b();
        let mel = super::log_mel(&vec![0.0f32; 1600], 1, 16_000, &cfg).unwrap();
        assert_eq!(mel.shape[1], 1);
        assert_eq!(mel.shape[3], cfg.mel_bins);
        assert_eq!(mel.shape[2], 1 + 1600 / cfg.mel_hop_length);
        let floor = (1e-5f32).ln();
        for v in mel.host_cow().unwrap().iter() {
            assert!((v - floor).abs() < 1e-5, "{v} vs {floor}");
        }
    }

    #[test]
    fn an_impulse_fft_is_flat() {
        let mut x = vec![0f32; 1024];
        x[0] = 1.0;
        let spec = super::fft_radix2(&x);
        for (re, im) in spec {
            assert!((re - 1.0).abs() < 1e-4 && im.abs() < 1e-4, "{re} {im}");
        }
    }

    #[test]
    fn encoder_two_causal_downs_match_the_latent_grid() {
        let cfg = tiny();
        let enc = AudioEncoder::load(&weights(), &cfg).unwrap();
        let (t, m) = (17usize, cfg.mel_bins);
        let mel = CudaTensor::zeros(&[1, cfg.in_channels, t, m]);
        let z = enc.encode_spectrogram(&mel).unwrap();
        assert_eq!(
            z.shape,
            vec![1, cfg.latent_channels, 5, cfg.latent_mel_bins()]
        );
        let longer = {
            let mut v = z.host_cow().unwrap().into_owned();
            v[0] = 3.5;
            CudaTensor::from_vec(v, z.shape.clone()).unwrap()
        };
        let fit = super::conform_audio_time(&longer, 8).unwrap();
        assert_eq!(
            fit.shape,
            vec![1, cfg.latent_channels, 8, cfg.latent_mel_bins()]
        );
        let host = fit.host_cow().unwrap();
        assert!((host[0] - 3.5).abs() < 1e-6);
        let tail = cfg.latent_mel_bins();
        assert!(host[5 * tail..6 * tail].iter().all(|v| *v == 0.0));
    }
}
