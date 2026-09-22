//! LTX-2 video VAE **encoder** (`AutoencoderKLLTX2Video` / Diffusers `encoder.*`).
//!
//! Full causal ResNet + [`LTXVideoDownsampler3d`]-style space-to-depth stack when
//! `encoder.down_blocks.*` keys are present. Falls back to patchify + `conv_in`
//! (+ channel fold) when only the stem is available. I2V uses this for real
//! encode whenever `vae/` is present with encoder probes.

use std::path::Path;

use fastvideo_models::ltx2::config::Ltx2VideoVaeConfig;
use image::RgbImage;

use crate::hub_keys::{self, ltx2_vae_encoder as ekeys};
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

/// Causal 3×3×3 (or 1×1×1) conv: time pad on the left only, spatial reflect.
struct CausalConv {
    weight: CudaTensor,
    bias: CudaTensor,
    stride: [usize; 3],
    spatial_pad: usize,
    temporal_pad: usize,
}

impl CausalConv {
    fn load(
        map: &WeightMap,
        prefix: &str,
        cin: usize,
        cout: usize,
        kernel: [usize; 3],
        stride: [usize; 3],
    ) -> Result<Self> {
        // Diffusers LTX nests `CausalConv3d` as `{prefix}.conv.weight`.
        let w_key = format!("{prefix}.conv.weight");
        let b_key = format!("{prefix}.conv.bias");
        let (w_key, b_key) = if map.contains(&w_key) {
            (w_key, b_key)
        } else {
            (format!("{prefix}.weight"), format!("{prefix}.bias"))
        };
        let spatial_pad = kernel[1] / 2;
        let temporal_pad = kernel[0].saturating_sub(1);
        Ok(Self {
            weight: pinned(cuda_tensor_shaped(
                map,
                &w_key,
                &[cout, cin, kernel[0], kernel[1], kernel[2]],
            )?)?,
            bias: pinned(cuda_tensor_shaped(map, &b_key, &[cout])?)?,
            stride,
            spatial_pad,
            temporal_pad,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let mut x = x.clone();
        if self.spatial_pad > 0 {
            let p = self.spatial_pad;
            x = x.pad(3, p, p, PadMode::Reflect)?.pad(4, p, p, PadMode::Reflect)?;
        }
        if self.temporal_pad > 0 {
            // Causal: replicate first frame (Diffusers LTX) — use zeros pad for
            // stills; replicate is closer for video. Prefer replicate via cat.
            let first = x.narrow(2, 0, 1)?;
            let mut parts = Vec::with_capacity(self.temporal_pad + 1);
            for _ in 0..self.temporal_pad {
                parts.push(first.clone());
            }
            parts.push(x);
            let refs: Vec<&CudaTensor> = parts.iter().collect();
            x = CudaTensor::cat(&refs, 2)?;
        }
        x.conv3d(&self.weight, Some(&self.bias), [0, 0, 0], self.stride)
            .map_err(Into::into)
    }
}

struct Resnet {
    conv1: CausalConv,
    conv2: CausalConv,
    ones: CudaTensor,
    shortcut: Option<CausalConv>,
}

impl Resnet {
    fn load(map: &WeightMap, prefix: &str, cin: usize, cout: usize) -> Result<Self> {
        let shortcut = if cin != cout {
            Some(CausalConv::load(
                map,
                &format!("{prefix}.conv_shortcut"),
                cin,
                cout,
                [1, 1, 1],
                [1, 1, 1],
            )?)
        } else {
            None
        };
        Ok(Self {
            conv1: CausalConv::load(map, &format!("{prefix}.conv1"), cin, cout, [3, 3, 3], [1, 1, 1])?,
            conv2: CausalConv::load(map, &format!("{prefix}.conv2"), cout, cout, [3, 3, 3], [1, 1, 1])?,
            ones: {
                let mut t = CudaTensor::ones(&[cout]);
                t.pin_device()?;
                t
            },
            shortcut,
        })
    }

    fn forward(&self, x: &CudaTensor, eps: f32) -> Result<CudaTensor> {
        let ones_in = if x.shape[1] == self.ones.shape[0] {
            self.ones.clone()
        } else {
            let mut t = CudaTensor::ones(&[x.shape[1]]);
            t.pin_device()?;
            t
        };
        let h = x.rms_norm_channels_act(&ones_in, eps, true)?;
        let h = self.conv1.forward(&h)?;
        let h = h.rms_norm_channels_act(&self.ones, eps, true)?;
        let h = self.conv2.forward(&h)?;
        match &self.shortcut {
            Some(sc) => Ok(sc.forward(x)?.add(&h)?),
            None => Ok(x.add(&h)?),
        }
    }
}

/// Diffusers `LTXVideoDownsampler3d`: stride-1 causal conv then space-to-depth.
struct Downsampler3d {
    conv: CausalConv,
    stride: [usize; 3],
    out_channels: usize,
}

impl Downsampler3d {
    fn try_load(map: &WeightMap, prefix: &str, in_ch: usize, out_ch: usize) -> Result<Option<Self>> {
        // Nested: `{prefix}.conv.conv.weight` (Downsampler wraps CausalConv).
        let nested = format!("{prefix}.conv.conv.weight");
        let flat = format!("{prefix}.conv.weight");
        let (st, sh, sw) = infer_stride(map, &nested, &flat, in_ch, out_ch)?;
        let prod = st * sh * sw;
        if prod == 0 {
            return Ok(None);
        }
        let conv_out = out_ch / prod;
        if conv_out == 0 || out_ch % prod != 0 {
            return Err(msg(format!(
                "ltx2 encoder downsampler: out_ch {out_ch} not divisible by stride ({st},{sh},{sw})"
            )));
        }
        let conv_prefix = if map.contains(&nested) {
            format!("{prefix}.conv")
        } else if map.contains(&flat) {
            prefix.to_string()
        } else {
            return Ok(None);
        };
        Ok(Some(Self {
            conv: CausalConv::load(map, &conv_prefix, in_ch, conv_out, [3, 3, 3], [1, 1, 1])?,
            stride: [st, sh, sw],
            out_channels: out_ch,
        }))
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let [st, sh, sw] = self.stride;
        // Diffusers prepends `stride_t - 1` frames for temporal alignment.
        let x = if st > 1 {
            let first = x.narrow(2, 0, 1)?;
            let mut parts = Vec::with_capacity(st);
            for _ in 0..(st - 1) {
                parts.push(first.clone());
            }
            parts.push(x.clone());
            let refs: Vec<&CudaTensor> = parts.iter().collect();
            CudaTensor::cat(&refs, 2)?
        } else {
            x.clone()
        };
        let y = self.conv.forward(&x)?;
        space_to_depth(&y, [st, sh, sw], self.out_channels)
    }
}

fn infer_stride(
    map: &WeightMap,
    nested: &str,
    flat: &str,
    in_ch: usize,
    out_ch: usize,
) -> Result<(usize, usize, usize)> {
    if map.contains(nested) {
        for (st, sh, sw) in [(2usize, 2, 2), (1, 2, 2), (2, 1, 1), (2, 2, 1), (1, 1, 1)] {
            let prod = st * sh * sw;
            if prod == 0 || out_ch % prod != 0 {
                continue;
            }
            let cout = out_ch / prod;
            if cuda_tensor_shaped(map, nested, &[cout, in_ch, 3, 3, 3]).is_ok() {
                return Ok((st, sh, sw));
            }
        }
        return Err(msg(format!(
            "ltx2 encoder: cannot infer stride for {nested} (in={in_ch} out={out_ch})"
        )));
    } else if map.contains(flat) {
        return Ok((2, 2, 2));
    }
    Ok((0, 0, 0))
}

/// Pixel-unshuffle / space-to-depth: spatial shrink × stride, channels × product.
fn space_to_depth(y: &CudaTensor, stride: [usize; 3], out_channels: usize) -> Result<CudaTensor> {
    let [st, sh, sw] = stride;
    let prod = st * sh * sw;
    let [b, c, f, h, w] = match y.shape[..] {
        [b, c, f, h, w] => [b, c, f, h, w],
        _ => return Err(msg(format!("space_to_depth rank {:?}", y.shape))),
    };
    if b != 1 || f % st != 0 || h % sh != 0 || w % sw != 0 {
        return Err(msg(format!(
            "space_to_depth: shape {:?} not divisible by stride {stride:?}",
            y.shape
        )));
    }
    let (nf, nh, nw) = (f / st, h / sh, w / sw);
    // Host rearrange matching Diffusers permute after unflatten.
    let host = y.host_cow()?;
    let mut out = vec![0f32; out_channels * nf * nh * nw];
    let oc_expected = c * prod;
    if oc_expected != out_channels {
        return Err(msg(format!(
            "space_to_depth: c*{prod}={oc_expected} vs out_channels {out_channels}"
        )));
    }
    for oc in 0..out_channels {
        let (ic, rem) = (oc / prod, oc % prod);
        let ft = rem / (sh * sw);
        let rem = rem % (sh * sw);
        let fy = rem / sw;
        let fx = rem % sw;
        for t in 0..nf {
            for yy in 0..nh {
                for xx in 0..nw {
                    let src_t = t * st + ft;
                    let src_y = yy * sh + fy;
                    let src_x = xx * sw + fx;
                    let src = ((ic * f + src_t) * h + src_y) * w + src_x;
                    let dst = ((oc * nf + t) * nh + yy) * nw + xx;
                    out[dst] = host[src];
                }
            }
        }
    }
    CudaTensor::from_vec(out, vec![1, out_channels, nf, nh, nw]).map_err(Into::into)
}

struct DownBlock {
    resnets: Vec<Resnet>,
    downsampler: Option<Downsampler3d>,
}

impl DownBlock {
    fn load(
        map: &WeightMap,
        idx: usize,
        in_ch: usize,
        out_ch: usize,
        n_res: usize,
        has_scale: bool,
    ) -> Result<Self> {
        let prefix = format!("encoder.down_blocks.{idx}");
        let mut resnets = Vec::with_capacity(n_res);
        for i in 0..n_res {
            // Resnets keep `in_ch` until after downsample (Diffusers LTXVideoDownBlock3D).
            resnets.push(Resnet::load(
                map,
                &format!("{prefix}.resnets.{i}"),
                in_ch,
                in_ch,
            )?);
        }
        let downsampler = if has_scale {
            Downsampler3d::try_load(map, &format!("{prefix}.downsamplers.0"), in_ch, out_ch)?
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsampler,
        })
    }

    fn forward(&self, mut x: CudaTensor, eps: f32) -> Result<CudaTensor> {
        for r in &self.resnets {
            x = r.forward(&x, eps)?;
        }
        if let Some(ds) = &self.downsampler {
            x = ds.forward(&x)?;
        }
        Ok(x)
    }
}

/// Full Diffusers encoder when down_blocks present; else stem-only.
pub struct VideoEncoder {
    pub cfg: Ltx2VideoVaeConfig,
    pub loaded_key: String,
    pub full_stack: bool,
    conv_in: Option<CausalConv>,
    down_blocks: Vec<DownBlock>,
    mid_resnets: Vec<Resnet>,
    conv_out: Option<CausalConv>,
    /// Stem-only fallback weights (host).
    conv_in_w: Option<Vec<f32>>,
    conv_in_shape: Option<[usize; 5]>,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    #[allow(dead_code)]
    mid_ch: usize,
}

impl VideoEncoder {
    pub fn try_load(map: &WeightMap, cfg: &Ltx2VideoVaeConfig) -> Result<Option<Self>> {
        let Some(hit) = hub_keys::first_present(map, ekeys::PROBES) else {
            return Ok(None);
        };
        let z = cfg.latent_channels;
        let mean = if map.contains("latents_mean") {
            cuda_tensor_shaped(map, "latents_mean", &[z])?
                .host_cow()?
                .to_vec()
        } else {
            vec![0f32; z]
        };
        let std = if map.contains("latents_std") {
            cuda_tensor_shaped(map, "latents_std", &[z])?
                .host_cow()?
                .to_vec()
        } else {
            vec![1f32; z]
        };

        let full = map.contains("encoder.down_blocks.0.resnets.0.conv1.conv.weight")
            || map.contains("encoder.down_blocks.0.resnets.0.conv1.weight");

        if full {
            return Self::load_full(map, cfg, hit, mean, std).map(Some);
        }

        // Stem-only: patchify + conv_in center tap.
        let cin = 3 * cfg.patch_size * cfg.patch_size;
        let cout = cfg.block_out_channels.first().copied().unwrap_or(128);
        let mut conv_in_w = None;
        let mut conv_in_shape = None;
        for key in ["encoder.conv_in.conv.weight", "encoder.conv_in.weight"] {
            if map.contains(key) {
                if let Ok(t) = cuda_tensor_shaped(map, key, &[cout, cin, 3, 3, 3]) {
                    conv_in_w = Some(t.host_cow()?.to_vec());
                    conv_in_shape = Some([cout, cin, 3, 3, 3]);
                    break;
                }
            }
        }
        Ok(Some(Self {
            cfg: cfg.clone(),
            loaded_key: hit,
            full_stack: false,
            conv_in: None,
            down_blocks: Vec::new(),
            mid_resnets: Vec::new(),
            conv_out: None,
            conv_in_w,
            conv_in_shape,
            latents_mean: mean,
            latents_std: std,
            mid_ch: cout,
        }))
    }

    fn load_full(
        map: &WeightMap,
        cfg: &Ltx2VideoVaeConfig,
        hit: String,
        mean: Vec<f32>,
        std: Vec<f32>,
    ) -> Result<Self> {
        // Channel ladder from weights / config.
        let mut widths = cfg.block_out_channels.clone();
        if widths.is_empty() {
            widths = vec![128, 256, 512, 1024];
        }
        // Infer first width from conv_in.
        let cin_patch = 3 * cfg.patch_size * cfg.patch_size;
        let first = widths[0];
        let conv_in = CausalConv::load(
            map,
            "encoder.conv_in",
            cin_patch,
            first,
            [3, 3, 3],
            [1, 1, 1],
        )
        .or_else(|_| {
            // Manifest / LTX-2 sometimes starts at 128 even when config lists 256.
            CausalConv::load(map, "encoder.conv_in", cin_patch, 128, [3, 3, 3], [1, 1, 1])
        })?;
        let mut ch = conv_in.weight.shape[0];
        let n_down = (0..8)
            .take_while(|i| {
                map.contains(&format!(
                    "encoder.down_blocks.{i}.resnets.0.conv1.conv.weight"
                )) || map.contains(&format!(
                    "encoder.down_blocks.{i}.resnets.0.conv1.weight"
                ))
            })
            .count();
        let mut down_blocks = Vec::with_capacity(n_down);
        for i in 0..n_down {
            let n_res = (0..16)
                .take_while(|r| {
                    map.contains(&format!(
                        "encoder.down_blocks.{i}.resnets.{r}.conv1.conv.weight"
                    )) || map.contains(&format!(
                        "encoder.down_blocks.{i}.resnets.{r}.conv1.weight"
                    ))
                })
                .count()
                .max(1);
            let has_ds = map.contains(&format!(
                "encoder.down_blocks.{i}.downsamplers.0.conv.conv.weight"
            )) || map.contains(&format!(
                "encoder.down_blocks.{i}.downsamplers.0.conv.weight"
            ));
            // Next channel: peek downsampler out or next block's resnet.
            let next_ch = if i + 1 < n_down {
                guess_resnet_channels(map, i + 1).unwrap_or(ch * 2)
            } else if has_ds {
                guess_down_out(map, i, ch).unwrap_or(ch * 2)
            } else {
                ch
            };
            let block = DownBlock::load(map, i, ch, next_ch, n_res, has_ds)?;
            if has_ds {
                ch = next_ch;
            }
            down_blocks.push(block);
        }
        let mut mid_resnets = Vec::new();
        let mid_n = (0..8)
            .take_while(|r| {
                map.contains(&format!("encoder.mid_block.resnets.{r}.conv1.conv.weight"))
                    || map.contains(&format!("encoder.mid_block.resnets.{r}.conv1.weight"))
            })
            .count();
        for r in 0..mid_n {
            mid_resnets.push(Resnet::load(
                map,
                &format!("encoder.mid_block.resnets.{r}"),
                ch,
                ch,
            )?);
        }
        let z = cfg.latent_channels;
        let conv_out = CausalConv::load(
            map,
            "encoder.conv_out",
            ch,
            z + 1, // Diffusers emits mean + logvar (+ extra)
            [3, 3, 3],
            [1, 1, 1],
        )
        .ok();
        Ok(Self {
            cfg: cfg.clone(),
            loaded_key: hit,
            full_stack: true,
            conv_in: Some(conv_in),
            down_blocks,
            mid_resnets,
            conv_out,
            conv_in_w: None,
            conv_in_shape: None,
            latents_mean: mean,
            latents_std: std,
            mid_ch: ch,
        })
    }

    pub fn load(map: &WeightMap, cfg: &Ltx2VideoVaeConfig) -> Result<Self> {
        Self::try_load(map, cfg)?.ok_or_else(|| {
            msg(hub_keys::require_any(map, "ltx2_vae_encoder", ekeys::PROBES).unwrap_err())
        })
    }

    pub fn encode_first_frame(
        &self,
        path: &Path,
        pixel_height: usize,
        pixel_width: usize,
    ) -> Result<CudaTensor> {
        let img = image::open(path)
            .map_err(|e| msg(format!("ltx2 encode open {}: {e}", path.display())))?
            .into_rgb8();
        let img = image::imageops::resize(
            &img,
            pixel_width as u32,
            pixel_height as u32,
            image::imageops::FilterType::Lanczos3,
        );
        self.encode_rgb8(&img, pixel_height, pixel_width)
    }

    pub fn encode_rgb8(
        &self,
        img: &RgbImage,
        pixel_height: usize,
        pixel_width: usize,
    ) -> Result<CudaTensor> {
        if self.full_stack {
            return self.encode_rgb8_full(img, pixel_height, pixel_width);
        }
        self.encode_rgb8_stem(img, pixel_height, pixel_width)
    }

    fn encode_rgb8_full(
        &self,
        img: &RgbImage,
        pixel_height: usize,
        pixel_width: usize,
    ) -> Result<CudaTensor> {
        let p = self.cfg.patch_size.max(1);
        let ph = pixel_height / p;
        let pw = pixel_width / p;
        if ph == 0 || pw == 0 {
            return Err(msg(format!(
                "ltx2 encode: {pixel_height}x{pixel_width} too small for patch {p}"
            )));
        }
        let cin = 3 * p * p;
        // Patchify Diffusers order: flatten (c, pt, p_w, p_h) — spatial p×p.
        let mut patched = vec![0f32; cin * ph * pw];
        for y in 0..ph {
            for x in 0..pw {
                for dy in 0..p {
                    for dx in 0..p {
                        let py = (y * p + dy).min(pixel_height - 1);
                        let px = (x * p + dx).min(pixel_width - 1);
                        let pix = img.get_pixel(px as u32, py as u32);
                        for ch in 0..3 {
                            let v = f32::from(pix[ch]) / 127.5 - 1.0;
                            // Match encoder forward permute: (c, p_w, p_h) with
                            // more-significant patch index on width (Diffusers).
                            let cidx = (ch * p + dx) * p + dy;
                            patched[(cidx * ph + y) * pw + x] = v;
                        }
                    }
                }
            }
        }
        let mut x = CudaTensor::from_vec(patched, vec![1, cin, 1, ph, pw])?;
        let eps = self.cfg.pixel_norm_eps as f32;
        let conv_in = self
            .conv_in
            .as_ref()
            .ok_or_else(|| msg("ltx2 encoder: full stack missing conv_in"))?;
        x = conv_in.forward(&x)?;
        for block in &self.down_blocks {
            x = block.forward(x, eps)?;
        }
        for r in &self.mid_resnets {
            x = r.forward(&x, eps)?;
        }
        let ones = {
            let mut t = CudaTensor::ones(&[x.shape[1]]);
            t.pin_device()?;
            t
        };
        x = x.rms_norm_channels_act(&ones, eps, true)?;
        if let Some(co) = &self.conv_out {
            x = co.forward(&x)?;
            // Take first `latent_channels` (drop logvar / extra).
            let z = self.cfg.latent_channels;
            if x.shape[1] > z {
                x = x.narrow(1, 0, z)?;
            }
        }
        // Normalize to DiT space.
        let [_, c, f, h, w] = match x.shape[..] {
            [1, c, f, h, w] => [1, c, f, h, w],
            _ => return Err(msg(format!("ltx2 encode out {:?}", x.shape))),
        };
        let host = x.host_cow()?;
        let mut lat = vec![0f32; c * f * h * w];
        for ch in 0..c {
            let mean = self.latents_mean.get(ch).copied().unwrap_or(0.0);
            let std = self.latents_std.get(ch).copied().unwrap_or(1.0).max(1e-6);
            for i in 0..(f * h * w) {
                let v = host[ch * f * h * w + i];
                lat[ch * f * h * w + i] =
                    (v - mean) / std * self.cfg.scaling_factor as f32;
            }
        }
        CudaTensor::from_vec(lat, vec![1, c, f, h, w]).map_err(Into::into)
    }

    fn encode_rgb8_stem(
        &self,
        img: &RgbImage,
        pixel_height: usize,
        pixel_width: usize,
    ) -> Result<CudaTensor> {
        let spat = self.cfg.spatial_compression_ratio.max(1);
        let lh = pixel_height / spat;
        let lw = pixel_width / spat;
        if lh == 0 || lw == 0 {
            return Err(msg(format!(
                "ltx2 encode: {pixel_height}x{pixel_width} too small"
            )));
        }
        let z = self.cfg.latent_channels;
        let p = self.cfg.patch_size.max(1);
        let ph = (pixel_height / p).max(1);
        let pw = (pixel_width / p).max(1);
        let cin = 3 * p * p;
        let mut patched = vec![0f32; cin * ph * pw];
        for y in 0..ph {
            for x in 0..pw {
                for dy in 0..p {
                    for dx in 0..p {
                        let py = (y * p + dy).min(pixel_height - 1);
                        let px = (x * p + dx).min(pixel_width - 1);
                        let pix = img.get_pixel(px as u32, py as u32);
                        for ch in 0..3 {
                            let v = f32::from(pix[ch]) / 127.5 - 1.0;
                            let cidx = (ch * p + dy) * p + dx;
                            patched[(cidx * ph + y) * pw + x] = v;
                        }
                    }
                }
            }
        }
        let mut feat = patched;
        let mut feat_c = cin;
        let fh = ph;
        let fw = pw;
        if let (Some(w), Some(shape)) = (&self.conv_in_w, self.conv_in_shape) {
            let [cout, cin_w, _, _, _] = shape;
            if cin_w == cin {
                let mut out = vec![0f32; cout * ph * pw];
                for oc in 0..cout {
                    for y in 0..ph {
                        for x in 0..pw {
                            let mut acc = 0f32;
                            for ic in 0..cin {
                                let widx = (((oc * cin + ic) * 3 + 1) * 3 + 1) * 3 + 1;
                                acc += w[widx] * feat[(ic * ph + y) * pw + x];
                            }
                            out[(oc * ph + y) * pw + x] = acc;
                        }
                    }
                }
                feat = out;
                feat_c = cout;
            }
        }
        let mut lat = vec![0f32; z * lh * lw];
        for y in 0..lh {
            for x in 0..lw {
                let y0 = y * fh / lh;
                let x0 = x * fw / lw;
                for ch in 0..z {
                    let src_c = ch % feat_c;
                    let v = feat[(src_c * fh + y0.min(fh - 1)) * fw + x0.min(fw - 1)];
                    let mean = self.latents_mean.get(ch).copied().unwrap_or(0.0);
                    let std = self.latents_std.get(ch).copied().unwrap_or(1.0).max(1e-6);
                    lat[(ch * lh + y) * lw + x] =
                        (v - mean) / std * self.cfg.scaling_factor as f32;
                }
            }
        }
        CudaTensor::from_vec(lat, vec![1, z, 1, lh, lw]).map_err(Into::into)
    }
}

fn guess_resnet_channels(map: &WeightMap, block: usize) -> Option<usize> {
    let k = format!("encoder.down_blocks.{block}.resnets.0.conv1.conv.weight");
    for c in [128usize, 256, 512, 1024, 2048, 64] {
        if cuda_tensor_shaped(map, &k, &[c, c, 3, 3, 3]).is_ok() {
            return Some(c);
        }
    }
    None
}

fn guess_down_out(map: &WeightMap, block: usize, in_ch: usize) -> Option<usize> {
    let k = format!("encoder.down_blocks.{block}.downsamplers.0.conv.conv.weight");
    for (st, sh, sw) in [(2usize, 2, 2), (1, 2, 2), (2, 1, 1)] {
        let prod = st * sh * sw;
        for out in [128usize, 256, 512, 1024, 2048] {
            if out % prod != 0 {
                continue;
            }
            let cout = out / prod;
            if cuda_tensor_shaped(map, &k, &[cout, in_ch, 3, 3, 3]).is_ok() {
                return Some(out);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn probes_documented() {
        assert!(ekeys::PROBES.iter().any(|k| k.contains("encoder")));
        assert!(ekeys::PROBES.iter().any(|k| k.contains("down_blocks")));
    }

    #[test]
    fn encode_rgb_without_conv_weights() {
        let cfg = Ltx2VideoVaeConfig::ltx2_19b();
        let enc = VideoEncoder {
            cfg: cfg.clone(),
            loaded_key: "encoder.conv_in.conv.weight".into(),
            full_stack: false,
            conv_in: None,
            down_blocks: Vec::new(),
            mid_resnets: Vec::new(),
            conv_out: None,
            conv_in_w: None,
            conv_in_shape: None,
            latents_mean: vec![0f32; cfg.latent_channels],
            latents_std: vec![1f32; cfg.latent_channels],
            mid_ch: 128,
        };
        let mut img = RgbImage::new(64, 64);
        for p in img.pixels_mut() {
            *p = Rgb([10, 20, 30]);
        }
        let lat = enc.encode_rgb8(&img, 64, 64).unwrap();
        assert_eq!(lat.shape, vec![1, cfg.latent_channels, 1, 2, 2]);
    }
}
