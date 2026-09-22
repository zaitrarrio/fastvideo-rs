//! MiniMax-H3 video VAE **encoder** half (`AutoencoderKLMiniMaxH3.encode`).
//!
//! Runs entirely on the cudarc device graph: reflect-padded causal Conv3d,
//! frame-isolated GroupNorm, ResNet down blocks, then `quant_conv`. A single
//! keyframe is encoded as `[1, 3, 1, H, W]` (no temporal chunk pad) and yields
//! one latent frame — matching the conditioning MiniMax-H3 was trained with.
//!
//! See diffusers `autoencoder_kl_minimax_h3.py` and docs/ports/h3.md §c.

use fastvideo_models::h3::config::{H3VideoVaeConfig, H3_PIXEL_MEAN, H3_PIXEL_STD};
use fastvideo_models::h3::packing::KEYFRAME_NOISE_AUG;

use crate::wan::ops::PadMode;
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

fn pinned(mut t: CudaTensor) -> Result<CudaTensor> {
    t.pin_device()?;
    Ok(t)
}

/// Reflect in H/W, causal zeros in T, then unpadded conv3d.
struct CausalConv3d {
    weight: CudaTensor,
    bias: CudaTensor,
    stride: [usize; 3],
    spatial_pad: usize,
    temporal_pad: usize,
}

impl CausalConv3d {
    fn load(
        map: &WeightMap,
        prefix: &str,
        cin: usize,
        cout: usize,
        kernel: [usize; 3],
        stride: [usize; 3],
        spatial_pad: usize,
        temporal_pad: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: pinned(cuda_tensor_shaped(
                map,
                &format!("{prefix}.weight"),
                &[cout, cin, kernel[0], kernel[1], kernel[2]],
            )?)?,
            bias: pinned(cuda_tensor_shaped(map, &format!("{prefix}.bias"), &[cout])?)?,
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
            x = x.pad(2, self.temporal_pad, 0, PadMode::Zeros)?;
        }
        x.conv3d(&self.weight, Some(&self.bias), [0, 0, 0], self.stride)
    }
}

/// GroupNorm with the temporal axis folded into batch so stats never mix frames.
struct FrameGroupNorm {
    weight: CudaTensor,
    bias: CudaTensor,
    groups: usize,
    eps: f32,
}

impl FrameGroupNorm {
    fn load(map: &WeightMap, prefix: &str, channels: usize, groups: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: pinned(cuda_tensor_shaped(map, &format!("{prefix}.weight"), &[channels])?)?,
            bias: pinned(cuda_tensor_shaped(map, &format!("{prefix}.bias"), &[channels])?)?,
            groups,
            eps: eps as f32,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        // [N, C, T, H, W] → [N*T, C, 1, H, W]
        let (n, c, t, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]);
        let flat = x
            .permute(&[0, 2, 1, 3, 4])?
            .reshape(vec![n * t, c, 1, h, w])?;
        let y = flat.group_norm(self.groups, &self.weight, &self.bias, self.eps, false)?;
        y.reshape(vec![n, t, c, h, w])?.permute(&[0, 2, 1, 3, 4])
    }
}

struct ResnetBlock {
    norm1: FrameGroupNorm,
    conv1: CausalConv3d,
    norm2: FrameGroupNorm,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResnetBlock {
    fn load(
        map: &WeightMap,
        prefix: &str,
        cin: usize,
        cout: usize,
        groups: usize,
        eps: f64,
    ) -> Result<Self> {
        let shortcut = if cin != cout {
            Some(CausalConv3d::load(
                map,
                &format!("{prefix}.conv_shortcut"),
                cin,
                cout,
                [1, 1, 1],
                [1, 1, 1],
                0,
                0,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: FrameGroupNorm::load(map, &format!("{prefix}.norm1"), cin, groups, eps)?,
            conv1: CausalConv3d::load(
                map,
                &format!("{prefix}.conv1"),
                cin,
                cout,
                [3, 3, 3],
                [1, 1, 1],
                1,
                2,
            )?,
            norm2: FrameGroupNorm::load(map, &format!("{prefix}.norm2"), cout, groups, eps)?,
            conv2: CausalConv3d::load(
                map,
                &format!("{prefix}.conv2"),
                cout,
                cout,
                [3, 3, 3],
                [1, 1, 1],
                1,
                2,
            )?,
            shortcut,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let residual = match &self.shortcut {
            Some(sc) => sc.forward(x)?,
            None => x.clone(),
        };
        let h = self.norm1.forward(x)?.silu();
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward(&h)?.silu();
        let h = self.conv2.forward(&h)?;
        residual.add(&h)
    }
}

struct Downsample {
    conv: CausalConv3d,
    spatial_stride: usize,
}

impl Downsample {
    fn load(
        map: &WeightMap,
        prefix: &str,
        channels: usize,
        temporal_stride: usize,
        spatial_stride: usize,
    ) -> Result<Self> {
        Ok(Self {
            conv: CausalConv3d::load(
                map,
                &format!("{prefix}.conv"),
                channels,
                channels,
                [3, 3, 3],
                [temporal_stride, spatial_stride, spatial_stride],
                0,
                2,
            )?,
            spatial_stride,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let x = if self.spatial_stride == 2 {
            // Asymmetric bottom/right reflect pad of 1 → ceil(size/2).
            x.pad(3, 0, 1, PadMode::Reflect)?.pad(4, 0, 1, PadMode::Reflect)?
        } else {
            x.clone()
        };
        self.conv.forward(&x)
    }
}

struct DownBlock {
    resnets: Vec<ResnetBlock>,
    downsamplers: Option<Vec<Downsample>>,
}

impl DownBlock {
    fn load(
        map: &WeightMap,
        prefix: &str,
        cin: usize,
        cout: usize,
        num_layers: usize,
        temporal_ds: usize,
        spatial_ds: usize,
        groups: usize,
        eps: f64,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let in_c = if i == 0 { cin } else { cout };
            resnets.push(ResnetBlock::load(
                map,
                &format!("{prefix}.resnets.{i}"),
                in_c,
                cout,
                groups,
                eps,
            )?);
        }
        let downsamplers = if temporal_ds * spatial_ds > 1 {
            Some(vec![Downsample::load(
                map,
                &format!("{prefix}.downsamplers.0"),
                cout,
                temporal_ds,
                spatial_ds,
            )?])
        } else {
            None
        };
        Ok(Self { resnets, downsamplers })
    }

    fn forward(&self, mut x: CudaTensor) -> Result<CudaTensor> {
        for r in &self.resnets {
            x = r.forward(&x)?;
        }
        if let Some(ds) = &self.downsamplers {
            for d in ds {
                x = d.forward(&x)?;
            }
        }
        Ok(x)
    }
}

struct Encoder3d {
    conv_in: CausalConv3d,
    down_blocks: Vec<DownBlock>,
    norm_out: FrameGroupNorm,
    conv_out: CausalConv3d,
}

impl Encoder3d {
    fn load(cfg: &H3VideoVaeConfig, map: &WeightMap) -> Result<Self> {
        let groups = cfg.norm_num_groups;
        let eps = cfg.norm_eps;
        let blocks = &cfg.block_out_channels;
        let mut down_blocks = Vec::with_capacity(blocks.len());
        for (i, &out_c) in blocks.iter().enumerate() {
            let cin = if i == 0 { blocks[0] } else { blocks[i - 1] };
            down_blocks.push(DownBlock::load(
                map,
                &format!("encoder.down_blocks.{i}"),
                cin,
                out_c,
                cfg.layers_per_block,
                cfg.temporal_downsample_factors[i],
                cfg.spatial_downsample_factors[i],
                groups,
                eps,
            )?);
        }
        let last = *blocks.last().unwrap();
        Ok(Self {
            conv_in: CausalConv3d::load(
                map,
                "encoder.conv_in",
                cfg.in_channels,
                blocks[0],
                [3, 3, 3],
                [1, 1, 1],
                1,
                2,
            )?,
            down_blocks,
            norm_out: FrameGroupNorm::load(map, "encoder.norm_out", last, groups, eps)?,
            conv_out: CausalConv3d::load(
                map,
                "encoder.conv_out",
                last,
                2 * cfg.latent_channels,
                [3, 3, 3],
                [1, 1, 1],
                1,
                2,
            )?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let mut h = self.conv_in.forward(x)?;
        for b in &self.down_blocks {
            h = b.forward(h)?;
        }
        let h = self.norm_out.forward(&h)?.silu();
        self.conv_out.forward(&h)
    }
}

/// Full encode path: encoder → quant_conv → mean of the diagonal Gaussian.
pub struct H3VideoEncoder {
    cfg: H3VideoVaeConfig,
    encoder: Encoder3d,
    quant_conv: CausalConv3d,
    mean: CudaTensor,
    std: CudaTensor,
}

impl H3VideoEncoder {
    pub fn load(cfg: H3VideoVaeConfig, map: &WeightMap) -> Result<Self> {
        let lc = cfg.latent_channels;
        let mean = pinned(CudaTensor::from_vec(
            cfg.latents_mean.iter().take(lc).map(|&x| x as f32).collect(),
            vec![1, lc, 1, 1, 1],
        )?)?;
        let std = pinned(CudaTensor::from_vec(
            cfg.latents_std.iter().take(lc).map(|&x| x as f32).collect(),
            vec![1, lc, 1, 1, 1],
        )?)?;
        Ok(Self {
            encoder: Encoder3d::load(&cfg, map)?,
            quant_conv: CausalConv3d::load(
                map,
                "quant_conv",
                2 * lc,
                2 * lc,
                [1, 1, 1],
                [1, 1, 1],
                0,
                0,
            )?,
            mean,
            std,
            cfg,
        })
    }

    pub fn config(&self) -> &H3VideoVaeConfig {
        &self.cfg
    }

    /// Encode RGB video `[1, 3, T, H, W]` already in ImageNet-normalized space.
    /// Returns **normalized** latents `[1, C, t, h, w]` — the DiT / decode
    /// input convention (`(mean - latents_mean) / latents_std`), matching
    /// FastVideo `normalize_latents(posterior.mode())`.
    ///
    /// `T = 1` skips temporal chunking (FL2VA / image refs). `T > 1` pads to a
    /// multiple of [`H3VideoVaeConfig::clip_length`], encodes each clip, then
    /// drops [`H3VideoVaeConfig::token_drop`] trailing latents — matching
    /// diffusers `_encode`.
    pub fn encode(&self, x: &CudaTensor) -> Result<CudaTensor> {
        if x.rank() != 5 || x.shape[1] != 3 {
            return Err(msg(format!("h3 encode expects [1,3,T,H,W], got {:?}", x.shape)));
        }
        let moments = self.encode_temporal(x)?;
        // DiagonalGaussian: first half = mean (mode).
        let mean = moments.narrow(1, 0, self.cfg.latent_channels)?;
        mean.sub(&self.mean)?.div(&self.std)
    }

    fn encode_temporal(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let clip = self.cfg.clip_length;
        let num_frames = x.shape[2];
        if num_frames == 1 {
            return self.encode_clip(x);
        }
        let mut x = x.clone();
        if num_frames % clip != 0 {
            let pad = (clip - num_frames % clip) % clip;
            let last = x.narrow(2, num_frames - 1, 1)?;
            let mut parts = Vec::with_capacity(1 + pad);
            parts.push(x);
            for _ in 0..pad {
                parts.push(last.clone());
            }
            let refs: Vec<&CudaTensor> = parts.iter().collect();
            x = CudaTensor::cat(&refs, 2)?;
        }
        let n_clips = x.shape[2] / clip;
        let mut clips = Vec::with_capacity(n_clips);
        for i in 0..n_clips {
            clips.push(self.encode_clip(&x.narrow(2, i * clip, clip)?)?);
        }
        let refs: Vec<&CudaTensor> = clips.iter().collect();
        let mut moments = CudaTensor::cat(&refs, 2)?;
        if self.cfg.token_drop > 0 {
            let keep = moments.shape[2].saturating_sub(self.cfg.token_drop);
            if keep == 0 {
                return Err(msg("h3 encode: token_drop removed every latent frame"));
            }
            moments = moments.narrow(2, 0, keep)?;
        }
        Ok(moments)
    }

    fn encode_clip(&self, x: &CudaTensor) -> Result<CudaTensor> {
        // Spatial tiling for large canvases (same geometry as the decoder).
        let (h, w) = (x.shape[3], x.shape[4]);
        let tile = self.cfg.tile_sample_min_size;
        let overlap = self.cfg.tile_sample_min_overlap;
        if !self.needs_tiling(h, w) {
            let hiddens = self.encoder.forward(x)?;
            return self.quant_conv.forward(&hiddens);
        }
        let (y_idx, y_len, y_ov) = split_tiles(h, tile, overlap, self.cfg.spatial_compression_ratio());
        let (x_idx, x_len, x_ov) = split_tiles(w, tile, overlap, self.cfg.spatial_compression_ratio());
        let ratio = self.cfg.spatial_compression_ratio();
        let mut rows: Vec<Vec<CudaTensor>> = Vec::with_capacity(y_idx.len());
        for (&yi, &ylen) in y_idx.iter().zip(&y_len) {
            let mut row = Vec::with_capacity(x_idx.len());
            for (&xj, &xlen) in x_idx.iter().zip(&x_len) {
                let tile = x.narrow(3, yi, ylen)?.narrow(4, xj, xlen)?;
                let hiddens = self.encoder.forward(&tile)?;
                row.push(self.quant_conv.forward(&hiddens)?);
            }
            rows.push(row);
        }
        let y_ov: Vec<usize> = y_ov.iter().map(|o| o / ratio).collect();
        let x_ov: Vec<usize> = x_ov.iter().map(|o| o / ratio).collect();
        stitch_tiles(rows, &y_ov, &x_ov)
    }

    fn needs_tiling(&self, h: usize, w: usize) -> bool {
        let t = self.cfg.tile_sample_min_size;
        h > t || w > t
    }

    /// Load a PNG/JPEG, canvas-fit to `(height, width)`, ImageNet-normalize,
    /// encode to one latent frame, optionally `scale_noise` for FL2VA.
    pub fn encode_keyframe_file(
        &self,
        path: &std::path::Path,
        height: usize,
        width: usize,
        stretch: bool,
        noise_aug: bool,
        seed: u64,
    ) -> Result<CudaTensor> {
        let rgb = load_rgb_imagenet(path, width, height, stretch)?;
        let x = CudaTensor::from_vec(rgb, vec![1, 3, 1, height, width])?.to_device()?;
        let mut z = self.encode(&x)?;
        if noise_aug {
            z = scale_noise_latent(&z, KEYFRAME_NOISE_AUG, seed)?;
        }
        Ok(z)
    }

    /// Encode a channel-major RGB video already resized to `(height, width)`.
    /// `frames` is `T * 3 * H * W` in `[0,1]` (or call with ImageNet-normalized
    /// values via [`Self::encode`] directly). Values here are ImageNet-normalized
    /// the same way as keyframes.
    pub fn encode_rgb_frames(
        &self,
        frames_u8: &[u8],
        num_frames: usize,
        height: usize,
        width: usize,
        noise_aug: bool,
        seed: u64,
    ) -> Result<CudaTensor> {
        if frames_u8.len() != num_frames * height * width * 3 || num_frames == 0 {
            return Err(msg(format!(
                "h3 encode frames: {} bytes for {num_frames}x{height}x{width}",
                frames_u8.len()
            )));
        }
        let mut rgb = vec![0f32; num_frames * 3 * height * width];
        // u8 HWC frames → ImageNet NCHW (channel-major over T).
        for t in 0..num_frames {
            for y in 0..height {
                for x in 0..width {
                    let src = ((t * height + y) * width + x) * 3;
                    for c in 0..3 {
                        let v01 = frames_u8[src + c] as f32 / 255.0;
                        let mean = H3_PIXEL_MEAN[c] as f32;
                        let std = H3_PIXEL_STD[c] as f32;
                        // layout: [C, T, H, W]
                        let dst = ((c * num_frames + t) * height + y) * width + x;
                        rgb[dst] = (v01 - mean) / std;
                    }
                }
            }
        }
        let x = CudaTensor::from_vec(rgb, vec![1, 3, num_frames, height, width])?.to_device()?;
        let mut z = self.encode(&x)?;
        if noise_aug {
            z = scale_noise_latent(&z, KEYFRAME_NOISE_AUG, seed)?;
        }
        Ok(z)
    }
}

/// `x_t = t * x0 + (1 - t) * noise` on device.
fn scale_noise_latent(clean: &CudaTensor, t: f32, seed: u64) -> Result<CudaTensor> {
    use rand::{Rng, SeedableRng};
    use rand_distr::StandardNormal;
    let n = clean.shape.iter().product::<usize>();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let noise: Vec<f32> = (0..n).map(|_| rng.sample::<f32, _>(StandardNormal)).collect();
    let noise = CudaTensor::from_vec(noise, clean.shape.clone())?.to_device()?;
    let a = clean.mul_scalar(t);
    let b = noise.mul_scalar(1.0 - t);
    a.add(&b)
}

fn load_rgb_imagenet(path: &std::path::Path, width: usize, height: usize, stretch: bool) -> Result<Vec<f32>> {
    let img = image::open(path)
        .map_err(|e| msg(format!("open {}: {e}", path.display())))?
        .into_rgb8();
    let (sw, sh) = (img.width() as usize, img.height() as usize);
    let raw: Vec<u8> = img.into_raw();
    let fitted = fastvideo_models::h3::packing::prepare_keyframe_rgb(&raw, sw, sh, width, height, stretch)
        .map_err(msg)?;
    // prepare_keyframe_rgb is in [-1,1]; convert to [0,1] then ImageNet.
    let mut out = vec![0f32; fitted.len()];
    for c in 0..3 {
        let mean = H3_PIXEL_MEAN[c] as f32;
        let std = H3_PIXEL_STD[c] as f32;
        for i in 0..height * width {
            let v01 = (fitted[c * height * width + i] + 1.0) * 0.5;
            out[c * height * width + i] = (v01 - mean) / std;
        }
    }
    Ok(out)
}

fn split_tiles(length: usize, tile_size: usize, min_overlap: usize, align: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    if tile_size >= length {
        return (vec![0], vec![length], vec![]);
    }
    let mut num_tiles = (length + tile_size - 1) / tile_size;
    while tile_size * num_tiles < min_overlap * (num_tiles - 1) + length {
        num_tiles += 1;
    }
    let mut overlaps = vec![min_overlap; num_tiles.saturating_sub(1)];
    let mut remaining = tile_size * num_tiles - overlaps.iter().sum::<usize>() - length;
    let mut i = 0;
    while remaining >= align && !overlaps.is_empty() {
        let idx = i % overlaps.len();
        overlaps[idx] += align;
        remaining -= align;
        i += 1;
    }
    let mut starts = Vec::with_capacity(num_tiles);
    let mut lens = Vec::with_capacity(num_tiles);
    let mut pos = 0usize;
    for t in 0..num_tiles {
        starts.push(pos);
        lens.push(tile_size.min(length.saturating_sub(pos).max(1)));
        if t + 1 < num_tiles {
            pos += tile_size - overlaps[t];
        }
    }
    (starts, lens, overlaps)
}

fn stitch_tiles(mut rows: Vec<Vec<CudaTensor>>, y_ov: &[usize], x_ov: &[usize]) -> Result<CudaTensor> {
    // Linear blend on overlaps (same as decoder `_stitch_tiles`).
    for i in 0..rows.len() {
        for j in 0..rows[i].len() {
            if i > 0 {
                let ov = y_ov[i - 1];
                rows[i][j] = blend(&rows[i - 1][j], &rows[i][j], ov, 3)?;
            }
            if j > 0 {
                let ov = x_ov[j - 1];
                rows[i][j] = blend(&rows[i][j - 1], &rows[i][j], ov, 4)?;
            }
        }
    }
    let mut row_cats = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let mut parts = Vec::with_capacity(row.len());
        for (j, tile) in row.iter().enumerate() {
            let mut t = tile.clone();
            if i + 1 < rows.len() {
                let ov = y_ov[i];
                let keep = t.shape[3].saturating_sub(ov);
                t = t.narrow(3, 0, keep)?;
            }
            if j + 1 < row.len() {
                let ov = x_ov[j];
                let keep = t.shape[4].saturating_sub(ov);
                t = t.narrow(4, 0, keep)?;
            }
            parts.push(t);
        }
        let refs: Vec<&CudaTensor> = parts.iter().collect();
        row_cats.push(CudaTensor::cat(&refs, 4)?);
    }
    let refs: Vec<&CudaTensor> = row_cats.iter().collect();
    CudaTensor::cat(&refs, 3)
}

fn blend(a: &CudaTensor, b: &CudaTensor, overlap: usize, dim: usize) -> Result<CudaTensor> {
    if overlap == 0 {
        return Ok(b.clone());
    }
    let a_tail = a.narrow(dim, a.shape[dim] - overlap, overlap)?;
    let b_head = b.narrow(dim, 0, overlap)?;
    let mut blended = Vec::with_capacity(overlap);
    for i in 0..overlap {
        let w = (i as f32 + 1.0) / (overlap as f32 + 1.0);
        let ai = a_tail.narrow(dim, i, 1)?;
        let bi = b_head.narrow(dim, i, 1)?;
        blended.push(ai.mul_scalar(1.0 - w).add(&bi.mul_scalar(w))?);
    }
    let refs: Vec<&CudaTensor> = blended.iter().collect();
    let mid = CudaTensor::cat(&refs, dim)?;
    let rest = b.narrow(dim, overlap, b.shape[dim] - overlap)?;
    CudaTensor::cat(&[&mid, &rest], dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_tiles_covers_length() {
        let (starts, lens, ov) = split_tiles(768, 256, 64, 16);
        assert!(starts.len() >= 3);
        assert_eq!(starts.len(), lens.len());
        assert_eq!(ov.len(), starts.len() - 1);
        let end = starts.last().unwrap() + lens.last().unwrap();
        assert!(end >= 768);
    }
}
