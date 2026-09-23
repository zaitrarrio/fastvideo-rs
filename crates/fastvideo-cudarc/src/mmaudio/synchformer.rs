//! Synchformer visual encoder for MMAudio V2A sync features.
//!
//! Upstream: hkchengrex/MMAudio `ext/synchformer` (MotionFormer visual half) and
//! v-iashin/Synchformer. Frames @ 25 fps → overlapping clips of 16 with stride 8
//! → 8 tokens/clip → ~24 fps sync features of dim **768**.
//!
//! Weight keys live under `image_encoder/` (Diffusers) or a remapped
//! `synchformer_state_dict` (`vfeat_extractor.*`).

use std::path::Path;

use image::RgbImage;

use crate::hub_keys;
use crate::wan::nn::{self, Linear};
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::tensor::CudaTensor;
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// Synchformer / MotionFormer sync feature width.
pub const SYNC_DIM: usize = 768;
/// Frames per Synchformer clip.
pub const SEGMENT_SIZE: usize = 16;
/// Clip hop (frames).
pub const STEP_SIZE: usize = 8;
/// Spatial resolution expected by the visual encoder.
pub const FRAME_SIZE: usize = 224;
/// Temporal tokens emitted per clip after space-time factorize (`temp_attn_agg=Identity`).
pub const TOKENS_PER_CLIP: usize = 8;

/// Diffusers / native key probes for the visual feature extractor.
pub mod probes {
    pub const PROBES: &[&str] = &[
        "vfeat_extractor.patch_embed_3d.proj.weight",
        "vfeat_extractor.blocks.0.attn.qkv.weight",
        "vfeat_extractor.spatial_attn_agg.cls_token",
        "vfeat_extractor.norm.weight",
        "patch_embed_3d.proj.weight",
        "blocks.0.attn.qkv.weight",
        "spatial_attn_agg.cls_token",
    ];
}

fn prefix_for(map: &WeightMap) -> Option<&'static str> {
    if map.contains("vfeat_extractor.patch_embed_3d.proj.weight")
        || map.contains("vfeat_extractor.blocks.0.attn.qkv.weight")
        || map.contains("vfeat_extractor.spatial_attn_agg.cls_token")
    {
        Some("vfeat_extractor")
    } else if map.contains("patch_embed_3d.proj.weight")
        || map.contains("blocks.0.attn.qkv.weight")
        || map.contains("spatial_attn_agg.cls_token")
    {
        Some("")
    } else {
        None
    }
}

fn key(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

struct VitBlock {
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
    norm1_w: CudaTensor,
    norm1_b: CudaTensor,
    norm2_w: CudaTensor,
    norm2_b: CudaTensor,
    heads: usize,
    head_dim: usize,
}

impl VitBlock {
    fn try_load(map: &WeightMap, prefix: &str, dim: usize) -> Result<Option<Self>> {
        let qkv_k = format!("{prefix}.attn.qkv.weight");
        if !map.contains(&qkv_k) {
            return Ok(None);
        }
        let heads = 12;
        let head_dim = dim / heads;
        Ok(Some(Self {
            qkv: Linear::load(map, &format!("{prefix}.attn.qkv"), dim, 3 * dim, true)?,
            proj: Linear::load(map, &format!("{prefix}.attn.proj"), dim, dim, true)?,
            fc1: Linear::load(map, &format!("{prefix}.mlp.fc1"), dim, dim * 4, true)?,
            fc2: Linear::load(map, &format!("{prefix}.mlp.fc2"), dim * 4, dim, true)?,
            norm1_w: cuda_tensor_shaped(map, &format!("{prefix}.norm1.weight"), &[dim])?,
            norm1_b: cuda_tensor_shaped(map, &format!("{prefix}.norm1.bias"), &[dim])?,
            norm2_w: cuda_tensor_shaped(map, &format!("{prefix}.norm2.weight"), &[dim])?,
            norm2_b: cuda_tensor_shaped(map, &format!("{prefix}.norm2.bias"), &[dim])?,
            heads,
            head_dim,
        }))
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        // x: [B, N, D]
        let nrm = x.layer_norm(1e-6, Some(&self.norm1_w), Some(&self.norm1_b))?;
        let qkv = self.qkv.forward(&nrm)?;
        let [b, n, _] = match qkv.shape[..] {
            [b, n, c] if c == 3 * self.heads * self.head_dim => [b, n, c],
            _ => return Err(msg(format!("synchformer qkv shape {:?}", qkv.shape))),
        };
        let hd = self.heads * self.head_dim;
        let q = qkv
            .narrow(2, 0, hd)?
            .reshape(vec![b, n, self.heads, self.head_dim])?
            .permute(&[0, 2, 1, 3])?;
        let k = qkv
            .narrow(2, hd, hd)?
            .reshape(vec![b, n, self.heads, self.head_dim])?
            .permute(&[0, 2, 1, 3])?;
        let v = qkv
            .narrow(2, 2 * hd, hd)?
            .reshape(vec![b, n, self.heads, self.head_dim])?
            .permute(&[0, 2, 1, 3])?;
        let attn = nn::scaled_dot_product_attention(&q, &k, &v, None)?;
        let attn = attn.permute(&[0, 2, 1, 3])?.reshape(vec![b, n, hd])?;
        let x = x.add(&self.proj.forward(&attn)?)?;
        let nrm = x.layer_norm(1e-6, Some(&self.norm2_w), Some(&self.norm2_b))?;
        let h = nn::gelu(&self.fc1.forward(&nrm)?);
        Ok(x.add(&self.fc2.forward(&h)?)?)
    }
}

/// Spatial aggregator: CLS + TransformerEncoderLayer-style attention over `h*w`.
struct SpatialAgg {
    cls: CudaTensor,
    in_proj: Linear,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
    norm1_w: CudaTensor,
    norm1_b: CudaTensor,
    norm2_w: CudaTensor,
    norm2_b: CudaTensor,
    heads: usize,
    dim: usize,
}

fn load_mha_in_proj(map: &WeightMap, prefix: &str, dim: usize) -> Result<Linear> {
    // nn.MultiheadAttention stores fused QKV as `in_proj_weight` / `in_proj_bias`.
    let w_key = format!("{prefix}.self_attn.in_proj_weight");
    let b_key = format!("{prefix}.self_attn.in_proj_bias");
    if map.contains(&w_key) {
        let w = cuda_tensor_shaped(map, &w_key, &[3 * dim, dim])?;
        let b = if map.contains(&b_key) {
            Some(cuda_tensor_shaped(map, &b_key, &[3 * dim])?)
        } else {
            None
        };
        return Ok(Linear::from_tensors(w, b)?);
    }
    Ok(Linear::load(
        map,
        &format!("{prefix}.self_attn.in_proj"),
        dim,
        3 * dim,
        true,
    )?)
}

impl SpatialAgg {
    fn try_load(map: &WeightMap, prefix: &str, dim: usize) -> Result<Option<Self>> {
        let cls_k = format!("{prefix}.cls_token");
        let in_k = format!("{prefix}.self_attn.in_proj_weight");
        let in_alt = format!("{prefix}.self_attn.in_proj.weight");
        if !map.contains(&cls_k) && !map.contains(&in_k) && !map.contains(&in_alt) {
            return Ok(None);
        }
        let cls = if map.contains(&cls_k) {
            let t = cuda_tensor_shaped(map, &cls_k, &[1, 1, dim])
                .or_else(|_| cuda_tensor_shaped(map, &cls_k, &[1, dim]))?;
            if t.shape.len() == 2 {
                t.reshape(vec![1, 1, dim])?
            } else {
                t
            }
        } else {
            CudaTensor::zeros(&[1, 1, dim])
        };
        if !map.contains(&in_k) && !map.contains(&in_alt) {
            return Err(msg(format!(
                "synchformer: {prefix} has cls_token but missing self_attn.in_proj_weight"
            )));
        }
        Ok(Some(Self {
            cls,
            in_proj: load_mha_in_proj(map, prefix, dim)?,
            out_proj: Linear::load(map, &format!("{prefix}.self_attn.out_proj"), dim, dim, true)?,
            linear1: Linear::load(map, &format!("{prefix}.linear1"), dim, dim * 4, true)?,
            linear2: Linear::load(map, &format!("{prefix}.linear2"), dim * 4, dim, true)?,
            norm1_w: cuda_tensor_shaped(map, &format!("{prefix}.norm1.weight"), &[dim])?,
            norm1_b: cuda_tensor_shaped(map, &format!("{prefix}.norm1.bias"), &[dim])?,
            norm2_w: cuda_tensor_shaped(map, &format!("{prefix}.norm2.weight"), &[dim])?,
            norm2_b: cuda_tensor_shaped(map, &format!("{prefix}.norm2.bias"), &[dim])?,
            heads: 12,
            dim,
        }))
    }

    /// `x`: `[BT, HW, D]` → `[BT, D]` (CLS).
    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let [bt, hw, d] = match x.shape[..] {
            [bt, hw, d] if d == self.dim => [bt, hw, d],
            _ => {
                return Err(msg(format!(
                    "spatial_agg want [BT,HW,{}], got {:?}",
                    self.dim, x.shape
                )))
            }
        };
        let cls_row = self.cls.reshape(vec![1, d])?.host_cow()?.to_vec();
        let mut cls_host = Vec::with_capacity(bt * d);
        for _ in 0..bt {
            cls_host.extend_from_slice(&cls_row[..d]);
        }
        let cls = CudaTensor::from_vec(cls_host, vec![bt, 1, d])?;
        let x = CudaTensor::cat(&[&cls, x], 1)?; // [BT, 1+HW, D]
        let n = hw + 1;
        let nrm = x.layer_norm(1e-6, Some(&self.norm1_w), Some(&self.norm1_b))?;
        let qkv = self.in_proj.forward(&nrm)?;
        let hd = self.heads * (self.dim / self.heads);
        let head_dim = self.dim / self.heads;
        let q = qkv
            .narrow(2, 0, hd)?
            .reshape(vec![bt, n, self.heads, head_dim])?
            .permute(&[0, 2, 1, 3])?;
        let k = qkv
            .narrow(2, hd, hd)?
            .reshape(vec![bt, n, self.heads, head_dim])?
            .permute(&[0, 2, 1, 3])?;
        let v = qkv
            .narrow(2, 2 * hd, hd)?
            .reshape(vec![bt, n, self.heads, head_dim])?
            .permute(&[0, 2, 1, 3])?;
        let attn = nn::scaled_dot_product_attention(&q, &k, &v, None)?;
        let attn = attn.permute(&[0, 2, 1, 3])?.reshape(vec![bt, n, hd])?;
        let x = x.add(&self.out_proj.forward(&attn)?)?;
        let nrm = x.layer_norm(1e-6, Some(&self.norm2_w), Some(&self.norm2_b))?;
        let h = nn::gelu(&self.linear1.forward(&nrm)?);
        let x = x.add(&self.linear2.forward(&h)?)?;
        Ok(x.narrow(1, 0, 1)?.reshape(vec![bt, d])?)
    }
}

/// Loaded Synchformer visual path (patch embed + optional ViT blocks + spatial agg).
pub struct SynchformerVisual {
    pub embed_dim: usize,
    pub loaded_key: String,
    /// `[D, 3, Tt, Ph, Pw]` host weights when present.
    patch_w: Option<Vec<f32>>,
    patch_shape: Option<[usize; 5]>,
    patch_bias: Option<Vec<f32>>,
    blocks: Vec<VitBlock>,
    norm_w: Option<CudaTensor>,
    norm_b: Option<CudaTensor>,
    spatial: Option<SpatialAgg>,
    patch: usize,
    tubelet: usize,
}

impl SynchformerVisual {
    pub fn try_load(map: &WeightMap) -> Result<Option<Self>> {
        let Some(pfx) = prefix_for(map) else {
            return Ok(None);
        };
        let hit = hub_keys::first_present(map, probes::PROBES)
            .unwrap_or_else(|| key(pfx, "patch_embed_3d.proj.weight"));
        let dim = SYNC_DIM;
        let patch = 16usize;
        let tubelet = 2usize;
        let mut patch_w = None;
        let mut patch_shape = None;
        let mut patch_bias = None;
        let pw_key = key(pfx, "patch_embed_3d.proj.weight");
        if map.contains(&pw_key) {
            // Common MotionFormer 3D patch: [D, 3, Tt, P, P]
            if let Ok(t) = cuda_tensor_shaped(map, &pw_key, &[dim, 3, tubelet, patch, patch]) {
                patch_w = Some(t.host_cow()?.to_vec());
                patch_shape = Some([dim, 3, tubelet, patch, patch]);
            } else if let Ok(t) = cuda_tensor_shaped(map, &pw_key, &[dim, 3, 1, patch, patch]) {
                patch_w = Some(t.host_cow()?.to_vec());
                patch_shape = Some([dim, 3, 1, patch, patch]);
            }
            let pb = key(pfx, "patch_embed_3d.proj.bias");
            if map.contains(&pb) {
                patch_bias = Some(cuda_tensor_shaped(map, &pb, &[dim])?.host_cow()?.to_vec());
            }
        }
        let mut blocks = Vec::new();
        for i in 0..24 {
            let bp = key(pfx, &format!("blocks.{i}"));
            match VitBlock::try_load(map, &bp, dim)? {
                Some(b) => blocks.push(b),
                None => break,
            }
        }
        let norm_w = {
            let k = key(pfx, "norm.weight");
            if map.contains(&k) {
                Some(cuda_tensor_shaped(map, &k, &[dim])?)
            } else {
                None
            }
        };
        let norm_b = {
            let k = key(pfx, "norm.bias");
            if map.contains(&k) {
                Some(cuda_tensor_shaped(map, &k, &[dim])?)
            } else {
                None
            }
        };
        let spatial = SpatialAgg::try_load(map, &key(pfx, "spatial_attn_agg"), dim)?;
        if patch_w.is_none() && blocks.is_empty() && spatial.is_none() {
            return Err(msg(format!(
                "synchformer: matched prefix `{pfx}` but could not load patch_embed / blocks / spatial_attn_agg"
            )));
        }
        Ok(Some(Self {
            embed_dim: dim,
            loaded_key: hit,
            patch_w,
            patch_shape,
            patch_bias,
            blocks,
            norm_w,
            norm_b,
            spatial,
            patch,
            tubelet: patch_shape.map(|s| s[2]).unwrap_or(tubelet),
        }))
    }

    pub fn load(map: &WeightMap) -> Result<Self> {
        Self::try_load(map)?.ok_or_else(|| {
            msg(hub_keys::require_any(map, "synchformer", probes::PROBES).unwrap_err())
        })
    }

    /// Expected sync sequence length for `T` frames (~24 fps for T≈25·sec).
    pub fn sync_len(num_frames: usize) -> usize {
        if num_frames < SEGMENT_SIZE {
            return 0;
        }
        let segments = (num_frames - SEGMENT_SIZE) / STEP_SIZE + 1;
        segments * TOKENS_PER_CLIP
    }

    /// Encode RGB frames `[T][3*H*W]` already at [`FRAME_SIZE`] → `[1, L, 768]`.
    pub fn encode_frame_chw(&self, frames: &[Vec<f32>]) -> Result<CudaTensor> {
        let t = frames.len();
        if t < SEGMENT_SIZE {
            return Err(msg(format!(
                "synchformer: need ≥{SEGMENT_SIZE} frames @ {FRAME_SIZE}², got {t}"
            )));
        }
        let segments = (t - SEGMENT_SIZE) / STEP_SIZE + 1;
        let mut all_tokens: Vec<f32> = Vec::new();
        for s in 0..segments {
            let start = s * STEP_SIZE;
            let clip: Vec<&[f32]> = (0..SEGMENT_SIZE)
                .map(|i| frames[start + i].as_slice())
                .collect();
            let tokens = self.encode_one_clip(&clip)?; // [8, D]
            all_tokens.extend_from_slice(&tokens);
        }
        let l = segments * TOKENS_PER_CLIP;
        CudaTensor::from_vec(all_tokens, vec![1, l, self.embed_dim]).map_err(Into::into)
    }

    fn encode_one_clip(&self, frames: &[&[f32]]) -> Result<Vec<f32>> {
        let h = FRAME_SIZE;
        let w = FRAME_SIZE;
        let p = self.patch;
        let tt = self.tubelet.max(1);
        let gh = h / p;
        let gw = w / p;
        let gt = SEGMENT_SIZE / tt;
        let dim = self.embed_dim;
        // Patchify → tokens [gt*gh*gw, D]
        let n = gt * gh * gw;
        let mut tokens = vec![0f32; n * dim];
        if let (Some(pw), Some(shape)) = (&self.patch_w, self.patch_shape) {
            let [dout, cin, kt, ph, pw_] = shape;
            let _ = (cin, ph, pw_);
            for ti in 0..gt {
                for yi in 0..gh {
                    for xi in 0..gw {
                        let tok = ((ti * gh) + yi) * gw + xi;
                        for od in 0..dout {
                            let mut acc = self.patch_bias.as_ref().map(|b| b[od]).unwrap_or(0.0);
                            for ct in 0..kt {
                                let fr = frames[ti * tt + ct];
                                for c in 0..3 {
                                    for dy in 0..p {
                                        for dx in 0..p {
                                            let py = yi * p + dy;
                                            let px = xi * p + dx;
                                            let pix = fr[(c * h + py) * w + px];
                                            let widx = ((((od * 3 + c) * kt + ct) * p + dy) * p
                                                + dx)
                                                as usize;
                                            acc += pw[widx] * pix;
                                        }
                                    }
                                }
                            }
                            tokens[tok * dim + od] = acc;
                        }
                    }
                }
            }
        } else {
            // No patch weights: mean-pool RGB patches into first 3 dims (still a real graph path
            // only when blocks/spatial exist — try_load requires at least one piece).
            for ti in 0..gt {
                for yi in 0..gh {
                    for xi in 0..gw {
                        let tok = ((ti * gh) + yi) * gw + xi;
                        let mut acc = [0f32; 3];
                        let mut n_pix = 0usize;
                        for ct in 0..tt {
                            let fr = frames[ti * tt + ct];
                            for dy in 0..p {
                                for dx in 0..p {
                                    let py = yi * p + dy;
                                    let px = xi * p + dx;
                                    for c in 0..3 {
                                        acc[c] += fr[(c * h + py) * w + px];
                                    }
                                    n_pix += 1;
                                }
                            }
                        }
                        for c in 0..3 {
                            tokens[tok * dim + c] = acc[c] / n_pix as f32;
                        }
                    }
                }
            }
        }

        let mut x = CudaTensor::from_vec(tokens, vec![1, n, dim])?;
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        if let (Some(nw), Some(nb)) = (&self.norm_w, &self.norm_b) {
            x = x.layer_norm(1e-6, Some(nw), Some(nb))?;
        } else if let Some(nw) = &self.norm_w {
            x = x.rms_norm(nw, 1e-6)?;
        }

        // Factorize: [1, gt*gh*gw, D] → per-time spatial tokens → aggregate → [gt, D]
        let mut out = vec![0f32; gt * dim];
        if let Some(agg) = &self.spatial {
            for ti in 0..gt {
                let start = ti * gh * gw;
                let slice = x.narrow(1, start, gh * gw)?; // [1, HW, D]
                let pooled = agg.forward(&slice.reshape(vec![1, gh * gw, dim])?)?;
                let host = pooled.host_cow()?;
                out[ti * dim..(ti + 1) * dim].copy_from_slice(&host[..dim]);
            }
        } else {
            // Mean-pool spatial dims per temporal tubelet.
            let host = x.host_cow()?;
            for ti in 0..gt {
                for d in 0..dim {
                    let mut acc = 0f32;
                    for s in 0..(gh * gw) {
                        acc += host[((ti * gh * gw) + s) * dim + d];
                    }
                    out[ti * dim + d] = acc / (gh * gw) as f32;
                }
            }
        }
        // MotionFormer with 16 frames / tubelet 2 → gt=8 tokens (Identity time agg).
        Ok(out)
    }
}

/// Load RGB frames from a directory of images or a single image (repeated).
/// Values are CLIP/Synchformer-ish `[0,1]` CHW at [`FRAME_SIZE`].
pub fn load_sync_frames(path: &Path, min_frames: usize) -> Result<Vec<Vec<f32>>> {
    let mut imgs: Vec<RgbImage> = Vec::new();
    if path.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(path)
            .map_err(|e| msg(e.to_string()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                matches!(
                    p.extension().and_then(|s| s.to_str()),
                    Some("png" | "jpg" | "jpeg" | "webp" | "bmp")
                )
            })
            .collect();
        entries.sort();
        for p in entries {
            let img = image::open(&p)
                .map_err(|e| msg(format!("synchformer open {}: {e}", p.display())))?
                .into_rgb8();
            imgs.push(img);
        }
    } else if path.is_file() {
        let img = image::open(path)
            .map_err(|e| msg(format!("synchformer open {}: {e}", path.display())))?
            .into_rgb8();
        imgs.push(img);
    } else {
        return Err(msg(format!(
            "synchformer: video path {} is not a file or frame directory",
            path.display()
        )));
    }
    if imgs.is_empty() {
        return Err(msg(format!(
            "synchformer: no frames under {}",
            path.display()
        )));
    }
    let n0 = imgs.len();
    while imgs.len() < min_frames {
        imgs.push(imgs[imgs.len() % n0].clone());
    }
    let mut out = Vec::with_capacity(imgs.len());
    for img in imgs {
        let img = image::imageops::resize(
            &img,
            FRAME_SIZE as u32,
            FRAME_SIZE as u32,
            image::imageops::FilterType::Lanczos3,
        );
        let mut chw = vec![0f32; 3 * FRAME_SIZE * FRAME_SIZE];
        // ImageNet/CLIP-ish normalize used by Synchformer inputs in MMAudio.
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        for y in 0..FRAME_SIZE {
            for x in 0..FRAME_SIZE {
                let p = img.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let v = f32::from(p[c]) / 255.0;
                    chw[(c * FRAME_SIZE + y) * FRAME_SIZE + x] = (v - mean[c]) / std[c];
                }
            }
        }
        out.push(chw);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_len_formula() {
        // 25 fps × 8 s = 200 frames → L = 8 * ((200-16)/8 + 1) = 8*24 = 192 → 24 fps
        assert_eq!(SynchformerVisual::sync_len(200), 192);
        assert_eq!(SynchformerVisual::sync_len(16), 8);
        assert_eq!(SynchformerVisual::sync_len(15), 0);
    }

    #[test]
    fn probes_nonempty() {
        assert!(probes::PROBES.iter().any(|k| k.contains("vfeat_extractor")));
    }
}
