//! `AutoencoderKLLTX2Audio`, decoder half: audio latents → a stereo log-mel
//! spectrogram (16 kHz, hop 160, 64 bins) for the vocoder.
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
}
