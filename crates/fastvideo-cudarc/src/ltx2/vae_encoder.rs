//! LTX-2 video VAE **encoder**, for image conditioning (I2V, keyframes).
//!
//! `ltx_core.model.video_vae.VideoEncoder` (Lightricks/LTX-2 `fd4ded7`,
//! `video_vae.py:148-330`), loaded from the Diffusers key layout
//! (`AutoencoderKLLTX2Video`, `encoder.*`):
//!
//! * patchify 4×4 into 48 channels, channel = `(c·4 + pw)·4 + ph`
//!   (`ops.py:patchify`, `"b c (f p) (h q) (w r) -> b (c p r q) f h w"`);
//! * `conv_in`, then per block `resnets.*` (`res_x`: PixelNorm → SiLU →
//!   conv, twice, plus the input) and `downsamplers.0`
//!   (`SpaceToDepthDownsample`, `sampling.py:12-60`: the first frame
//!   duplicated for a temporal stride, a causal conv then space-to-depth,
//!   plus the space-to-depth of the input averaged over channel groups);
//! * `mid_block.resnets.*`, PixelNorm → SiLU → `conv_out` (`latent + 1`
//!   channels; the first `latent` are the means, the last the shared
//!   log-variance), then the per-channel statistics normalize the means.
//!
//! Every conv is causal in time (the first frame repeated `k − 1` times in
//! front) with the encoder's spatial padding (`zeros` unless the config
//! says otherwise). The strides of each downsampler follow from its weight
//! (`out = in·2 → conv out = out / prod(stride)`): 4 is `(1, 2, 2)`, 2 is
//! `(2, 1, 1)`, 8 is `(2, 2, 2)`.
//!
//! The reference runs the encoder in bf16; this runs it in f32 over the bf16
//! weights, so its latents differ from the reference's by bf16 rounding
//! (the oracle dumps both, `s{n}_cond{i}_latent`).

use std::path::Path;

use crate::wan::ops::PadMode;
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

fn pinned(mut t: CudaTensor) -> Result<CudaTensor> {
    t.pin_device()?;
    Ok(t)
}

fn ones(c: usize) -> Result<CudaTensor> {
    pinned(CudaTensor::ones(&[c]))
}

/// `CausalConv3d` (`convolution.py:270-317`): 3×3×3, stride 1, the first
/// frame repeated twice in front, spatial pad 1.
struct CausalConv {
    weight: CudaTensor,
    bias: CudaTensor,
    reflect: bool,
}

impl CausalConv {
    fn load(map: &WeightMap, prefix: &str, cin: usize, cout: usize, reflect: bool) -> Result<Self> {
        Ok(Self {
            weight: pinned(cuda_tensor_shaped(
                map,
                &format!("{prefix}.conv.weight"),
                &[cout, cin, 3, 3, 3],
            )?)?,
            bias: pinned(cuda_tensor_shaped(map, &format!("{prefix}.conv.bias"), &[cout])?)?,
            reflect,
        })
    }

    fn out_channels(&self) -> usize {
        self.weight.shape[0]
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let first = x.narrow(2, 0, 1)?;
        let x = CudaTensor::cat(&[&first, &first, x], 2)?;
        if self.reflect {
            let x = x
                .pad(3, 1, 1, PadMode::Reflect)?
                .pad(4, 1, 1, PadMode::Reflect)?;
            Ok(x.conv3d(&self.weight, Some(&self.bias), [0, 0, 0], [1, 1, 1])?)
        } else {
            Ok(x.conv3d(&self.weight, Some(&self.bias), [0, 1, 1], [1, 1, 1])?)
        }
    }
}

/// `ResnetBlock3D` with `in == out` (`resnet.py`): the encoder's only kind.
struct Resnet {
    conv1: CausalConv,
    conv2: CausalConv,
    ones: CudaTensor,
}

impl Resnet {
    fn load(map: &WeightMap, prefix: &str, ch: usize, reflect: bool) -> Result<Self> {
        Ok(Self {
            conv1: CausalConv::load(map, &format!("{prefix}.conv1"), ch, ch, reflect)?,
            conv2: CausalConv::load(map, &format!("{prefix}.conv2"), ch, ch, reflect)?,
            ones: ones(ch)?,
        })
    }

    fn forward(&self, x: CudaTensor, eps: f32) -> Result<CudaTensor> {
        let h = x.rms_norm_channels_act(&self.ones, eps, true)?;
        let h = self.conv1.forward(&h)?;
        let h = h.rms_norm_channels_act(&self.ones, eps, true)?;
        let h = self.conv2.forward(&h)?;
        Ok(x.add(&h)?)
    }
}

/// `SpaceToDepthDownsample` (`sampling.py:12-60`).
struct Downsampler {
    conv: CausalConv,
    stride: [usize; 3],
    out_channels: usize,
}

impl Downsampler {
    fn forward(&self, x: CudaTensor) -> Result<CudaTensor> {
        let x = if self.stride[0] == 2 {
            let first = x.narrow(2, 0, 1)?;
            CudaTensor::cat(&[&first, &x], 2)?
        } else {
            x
        };
        let skip = space_to_depth(&x, self.stride)?;
        let group = skip.shape[1] / self.out_channels;
        let skip = group_mean(&skip, self.out_channels, group)?;
        let y = self.conv.forward(&x)?;
        drop(x);
        let y = space_to_depth(&y, self.stride)?;
        Ok(y.add(&skip)?)
    }
}

/// `"b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w"` on the device, in two
/// moves of rank ≤ 6 (time, then space), which give the same channel order.
fn space_to_depth(x: &CudaTensor, stride: [usize; 3]) -> Result<CudaTensor> {
    let [b, c, f, h, w] = x.shape[..] else {
        return Err(msg(format!("ltx2 encoder s2d: shape {:?}", x.shape)));
    };
    let [st, sh, sw] = stride;
    if b != 1 || f % st != 0 || h % sh != 0 || w % sw != 0 {
        return Err(msg(format!(
            "ltx2 encoder s2d: {:?} not divisible by {stride:?}",
            x.shape
        )));
    }
    let mut y = x.clone();
    let (mut c, mut f) = (c, f);
    if st > 1 {
        y = y
            .reshape(vec![c, f / st, st, h * w])?
            .permute(&[0, 2, 1, 3])?
            .reshape(vec![1, c * st, f / st, h, w])?;
        c *= st;
        f /= st;
    }
    if sh > 1 || sw > 1 {
        y = y
            .reshape(vec![c, f, h / sh, sh, w / sw, sw])?
            .permute(&[0, 3, 5, 1, 2, 4])?
            .reshape(vec![1, c * sh * sw, f, h / sh, w / sw])?;
    }
    Ok(y)
}

/// `rearrange(x, "b (c g) d h w -> b c g d h w").mean(dim=2)`.
fn group_mean(x: &CudaTensor, out: usize, group: usize) -> Result<CudaTensor> {
    if group == 1 {
        return Ok(x.clone());
    }
    let rest: usize = x.shape[2..].iter().product();
    let v = x.reshape(vec![out, group, rest])?;
    let parts = (0..group)
        .map(|k| v.narrow(1, k, 1))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let terms: Vec<(f32, &CudaTensor)> = parts.iter().map(|p| (1.0 / group as f32, p)).collect();
    let mut shape = x.shape.clone();
    shape[1] = out;
    Ok(CudaTensor::lincomb(&terms)?.reshape(shape)?)
}

struct DownBlock {
    resnets: Vec<Resnet>,
    downsampler: Option<Downsampler>,
}

/// The LTX-2 video encoder, resident until dropped.
pub struct VideoEncoder {
    conv_in: CausalConv,
    blocks: Vec<DownBlock>,
    mid: Vec<Resnet>,
    conv_out: CausalConv,
    ones_out: CudaTensor,
    mean: Vec<f32>,
    std: Vec<f32>,
    patch: usize,
    eps: f32,
    latent_channels: usize,
}

fn count(map: &WeightMap, key: impl Fn(usize) -> String) -> usize {
    (0..64).take_while(|&i| map.contains(&key(i))).count()
}

fn padding_is_reflect(vae_dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(vae_dir.join("config.json")) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    v.get("encoder_spatial_padding_mode")
        .or_else(|| v.get("spatial_padding_mode"))
        .and_then(|m| m.as_str())
        == Some("reflect")
}

impl VideoEncoder {
    /// Whether `vae_dir` carries the encoder.
    pub fn present(map: &WeightMap) -> bool {
        map.contains("encoder.conv_in.conv.weight")
            && map.contains("encoder.conv_out.conv.weight")
            && map.contains("latents_mean")
    }

    /// Load `encoder.*` and the latent statistics from a Diffusers `vae/`.
    pub fn load(vae_dir: &Path, patch: usize, eps: f64) -> Result<Self> {
        let map = WeightMap::open(vae_dir).map_err(|e| msg(e.to_string()))?;
        if !Self::present(&map) {
            return Err(msg(format!(
                "ltx2 image conditioning: {} has no video encoder (encoder.* keys)",
                vae_dir.display()
            )));
        }
        let reflect = padding_is_reflect(vae_dir);
        let shape = |k: &str| {
            map.shape(k)
                .ok_or_else(|| msg(format!("ltx2 encoder: missing {k}")))
        };
        let cin = shape("encoder.conv_in.conv.weight")?;
        let conv_in = CausalConv::load(&map, "encoder.conv_in", cin[1], cin[0], reflect)?;
        if cin[1] != 3 * patch * patch {
            return Err(msg(format!(
                "ltx2 encoder: conv_in takes {} channels, patch {patch} gives {}",
                cin[1],
                3 * patch * patch
            )));
        }
        let mut ch = conv_in.out_channels();
        let n_blocks = count(&map, |i| format!("encoder.down_blocks.{i}.resnets.0.conv1.conv.weight"));
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let n_res = count(&map, |r| {
                format!("encoder.down_blocks.{i}.resnets.{r}.conv1.conv.weight")
            });
            let resnets = (0..n_res)
                .map(|r| Resnet::load(&map, &format!("encoder.down_blocks.{i}.resnets.{r}"), ch, reflect))
                .collect::<Result<Vec<_>>>()?;
            let dkey = format!("encoder.down_blocks.{i}.downsamplers.0.conv.conv.weight");
            let downsampler = match map.shape(&dkey) {
                None => None,
                Some(ds) => {
                    // The next width: the next block's resnets, else the mid block's.
                    let next = if i + 1 < n_blocks {
                        format!("encoder.down_blocks.{}.resnets.0.conv1.conv.weight", i + 1)
                    } else {
                        "encoder.mid_block.resnets.0.conv1.conv.weight".to_string()
                    };
                    let out = shape(&next)?[0];
                    let prod = out / ds[0];
                    let stride = match prod {
                        2 => [2, 1, 1],
                        4 => [1, 2, 2],
                        8 => [2, 2, 2],
                        _ => {
                            return Err(msg(format!(
                                "ltx2 encoder: block {i} downsampler {ds:?} → {out} channels"
                            )))
                        }
                    };
                    let conv = CausalConv::load(
                        &map,
                        &format!("encoder.down_blocks.{i}.downsamplers.0.conv"),
                        ch,
                        ds[0],
                        reflect,
                    )?;
                    ch = out;
                    Some(Downsampler {
                        conv,
                        stride,
                        out_channels: out,
                    })
                }
            };
            blocks.push(DownBlock {
                resnets,
                downsampler,
            });
        }
        let n_mid = count(&map, |r| format!("encoder.mid_block.resnets.{r}.conv1.conv.weight"));
        let mid = (0..n_mid)
            .map(|r| Resnet::load(&map, &format!("encoder.mid_block.resnets.{r}"), ch, reflect))
            .collect::<Result<Vec<_>>>()?;
        let co = shape("encoder.conv_out.conv.weight")?;
        let conv_out = CausalConv::load(&map, "encoder.conv_out", ch, co[0], reflect)?;
        let latent_channels = co[0] - 1;
        let stat = |k: &str| -> Result<Vec<f32>> {
            Ok(cuda_tensor_shaped(&map, k, &[latent_channels])?
                .host_cow()?
                .into_owned())
        };
        Ok(Self {
            conv_in,
            blocks,
            mid,
            conv_out,
            ones_out: ones(ch)?,
            mean: stat("latents_mean")?,
            std: stat("latents_std")?,
            patch,
            eps: eps as f32,
            latent_channels,
        })
    }

    pub fn latent_channels(&self) -> usize {
        self.latent_channels
    }

    /// One frame, `pixels` `[3, H, W]` row-major in `[-1, 1]` → the
    /// normalized latent `[1, C, 1, H/32, W/32]`.
    pub fn encode_image(&self, pixels: &[f32], height: usize, width: usize) -> Result<CudaTensor> {
        self.encode_video(pixels, 1, height, width)
    }

    /// A clip, `pixels` `[3, F, H, W]` row-major in `[-1, 1]` (`F = 8k + 1`)
    /// → the normalized latent `[1, C, (F − 1)/8 + 1, H/32, W/32]` on the
    /// host (`VideoEncoder.forward`: causal in time, the first frame alone
    /// in the first latent frame).
    pub fn encode_video(&self, pixels: &[f32], frames: usize, height: usize, width: usize) -> Result<CudaTensor> {
        let p = self.patch;
        if frames == 0
            || pixels.len() != 3 * frames * height * width
            || height % p != 0
            || width % p != 0
        {
            return Err(msg(format!(
                "ltx2 encoder: {} pixels for 3x{frames}x{height}x{width} (patch {p})",
                pixels.len()
            )));
        }
        let (ph, pw) = (height / p, width / p);
        let cin = 3 * p * p;
        let plane = height * width;
        let mut patched = vec![0f32; cin * frames * ph * pw];
        for c in 0..3 {
            for t in 0..frames {
                let src = &pixels[(c * frames + t) * plane..(c * frames + t + 1) * plane];
                for y in 0..height {
                    for x in 0..width {
                        // channel = (c·p + r)·p + q, r the column and q the row in the patch.
                        let k = (c * p + x % p) * p + y % p;
                        patched[((k * frames + t) * ph + y / p) * pw + x / p] = src[y * width + x];
                    }
                }
            }
        }
        let mut x = CudaTensor::from_vec(patched, vec![1, cin, frames, ph, pw])?.to_device()?;
        x = self.conv_in.forward(&x)?;
        for block in &self.blocks {
            for r in &block.resnets {
                x = r.forward(x, self.eps)?;
            }
            if let Some(d) = &block.downsampler {
                x = d.forward(x)?;
            }
        }
        for r in &self.mid {
            x = r.forward(x, self.eps)?;
        }
        let x = x.rms_norm_channels_act(&self.ones_out, self.eps, true)?;
        let x = self.conv_out.forward(&x)?;
        let c = self.latent_channels;
        let means = x.narrow(1, 0, c)?;
        let shape = means.shape.clone();
        let n: usize = shape[2..].iter().product();
        let host = means.host_cow()?;
        let mut out = vec![0f32; c * n];
        for ch in 0..c {
            let (m, s) = (self.mean[ch], self.std[ch]);
            for i in 0..n {
                out[ch * n + i] = (host[ch * n + i] - m) / s;
            }
        }
        Ok(CudaTensor::from_vec(out, shape)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_to_depth_matches_the_einops_order() {
        // [1, 2, 2, 2, 2] with value = index; stride (2, 2, 2).
        let v: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x = CudaTensor::from_vec(v, vec![1, 2, 2, 2, 2]).unwrap();
        let y = space_to_depth(&x, [2, 2, 2]).unwrap();
        assert_eq!(y.shape, vec![1, 16, 1, 1, 1]);
        let y = y.host_cow().unwrap().into_owned();
        // out channel ((c·2 + pt)·2 + ph)·2 + pw reads x[c, pt, ph, pw].
        for c in 0..2 {
            for pt in 0..2 {
                for ph in 0..2 {
                    for pw in 0..2 {
                        let o = ((c * 2 + pt) * 2 + ph) * 2 + pw;
                        let i = ((c * 2 + pt) * 2 + ph) * 2 + pw;
                        assert_eq!(y[o], i as f32);
                    }
                }
            }
        }
    }

    #[test]
    fn space_to_depth_spatial_only() {
        // [1, 1, 1, 2, 4]: out channel ph·2 + pw at (h', w').
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let x = CudaTensor::from_vec(v, vec![1, 1, 1, 2, 4]).unwrap();
        let y = space_to_depth(&x, [1, 2, 2]).unwrap();
        assert_eq!(y.shape, vec![1, 4, 1, 1, 2]);
        let y = y.host_cow().unwrap().into_owned();
        // x[h, w] = h·4 + w; out[(ph·2+pw), 0, w'] = x[ph, w'·2 + pw].
        assert_eq!(y, vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
    }

    #[test]
    fn group_mean_averages_consecutive_channels() {
        let x = CudaTensor::from_vec(vec![1.0, 3.0, 5.0, 7.0], vec![1, 4, 1, 1, 1]).unwrap();
        let y = group_mean(&x, 2, 2).unwrap();
        assert_eq!(y.shape, vec![1, 2, 1, 1, 1]);
        assert_eq!(y.host_cow().unwrap().into_owned(), vec![2.0, 6.0]);
    }
}
