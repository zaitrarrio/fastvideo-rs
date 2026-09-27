//! open_clip `ViT-H-14-378-quickgelu` (weights `apple/DFN5B-CLIP-ViT-H-14-384`)
//! as MMAudio's `FeaturesUtils` uses it:
//!
//! * `encode_video_with_clip`: frames `[T, 3, 384, 384]` (the checkpoint's
//!   native size is 378; the stride-14 patch conv drops the last 6 rows and
//!   columns, 27 x 27 patches either way) to `F.normalize(encode_image(x))`,
//!   `[T, 1024]`;
//! * `encode_text` patched by `patch_clip`: every position's `ln_final`
//!   output, L2-normalized per token, `[B, 77, 1024]` (no projection).
//!
//! Residual attention blocks with `nn.MultiheadAttention` (`in_proj_weight`
//! rows `[q | k | v]`), QuickGELU MLPs, LayerNorm eps 1e-5. Keys are
//! open_clip's (`visual.*`, `transformer.*`, `token_embedding`, …).

use super::layers::{gather_rows, host_values, linear, msg, pinned, quick_gelu, weight};
use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::WeightMap;

#[derive(Debug, Clone)]
pub struct ClipTowerConfig {
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
}

struct ResBlock {
    ln1: (CudaTensor, CudaTensor),
    in_proj: Linear,
    out_proj: Linear,
    ln2: (CudaTensor, CudaTensor),
    c_fc: Linear,
    c_proj: Linear,
    heads: usize,
}

fn ln(map: &WeightMap, p: &str, d: usize) -> Result<(CudaTensor, CudaTensor)> {
    Ok((
        weight(map, &format!("{p}.weight"), &[d])?,
        weight(map, &format!("{p}.bias"), &[d])?,
    ))
}

impl ResBlock {
    fn load(map: &WeightMap, p: &str, d: usize, heads: usize) -> Result<Self> {
        let w = host_values(map, &format!("{p}.attn.in_proj_weight"), &[3 * d, d])?;
        let b = host_values(map, &format!("{p}.attn.in_proj_bias"), &[3 * d])?;
        Ok(Self {
            ln1: ln(map, &format!("{p}.ln_1"), d)?,
            in_proj: super::layers::linear_from(w, Some(b), 3 * d, d)?,
            out_proj: linear(map, &format!("{p}.attn.out_proj"), d, d, true)?,
            ln2: ln(map, &format!("{p}.ln_2"), d)?,
            c_fc: linear(map, &format!("{p}.mlp.c_fc"), d, 4 * d, true)?,
            c_proj: linear(map, &format!("{p}.mlp.c_proj"), 4 * d, d, true)?,
            heads,
        })
    }

    fn forward(&self, x: &CudaTensor, mask: Option<&CudaTensor>) -> Result<CudaTensor> {
        let [b, n, d] = x.shape[..] else {
            return Err(msg(format!("clip block {:?}", x.shape)));
        };
        let h = self.heads;
        let hd = d / h;
        let xn = x.layer_norm(1e-5, Some(&self.ln1.0), Some(&self.ln1.1))?;
        let qkv = self.in_proj.forward(&xn)?;
        let part = |i: usize| -> Result<CudaTensor> {
            qkv.narrow(2, i * d, d)?
                .reshape(vec![b, n, h, hd])?
                .permute(&[0, 2, 1, 3])
        };
        let o = nn::scaled_dot_product_attention_masked(&part(0)?, &part(1)?, &part(2)?, None, mask)?;
        let o = o.permute(&[0, 2, 1, 3])?.reshape(vec![b, n, d])?;
        let x = x.add(&self.out_proj.forward(&o)?)?;
        let xn = x.layer_norm(1e-5, Some(&self.ln2.0), Some(&self.ln2.1))?;
        let m = self.c_proj.forward(&quick_gelu(&self.c_fc.forward(&xn)?)?)?;
        x.add(&m)
    }
}

fn blocks(map: &WeightMap, prefix: &str, cfg: &ClipTowerConfig) -> Result<Vec<ResBlock>> {
    (0..cfg.layers)
        .map(|i| ResBlock::load(map, &format!("{prefix}.resblocks.{i}"), cfg.width, cfg.heads))
        .collect()
}

pub struct ClipVisual {
    pub cfg: ClipTowerConfig,
    patch: usize,
    conv1: CudaTensor,
    class_embedding: Vec<f32>,
    pos: CudaTensor,
    ln_pre: (CudaTensor, CudaTensor),
    blocks: Vec<ResBlock>,
    ln_post: (CudaTensor, CudaTensor),
    /// `proj^T` as a linear: `[1024, 1280]`.
    proj: Linear,
}

impl ClipVisual {
    pub fn vit_h14() -> ClipTowerConfig {
        ClipTowerConfig { width: 1280, layers: 32, heads: 16 }
    }

    pub fn load(cfg: ClipTowerConfig, map: &WeightMap, patch: usize, grid: usize, embed: usize) -> Result<Self> {
        let d = cfg.width;
        let proj = host_values(map, "visual.proj", &[d, embed])?;
        // x @ proj == Linear with weight proj^T.
        let mut wt = vec![0.0f32; d * embed];
        for i in 0..d {
            for j in 0..embed {
                wt[j * d + i] = proj[i * embed + j];
            }
        }
        Ok(Self {
            patch,
            conv1: weight(map, "visual.conv1.weight", &[d, 3, patch, patch])?,
            class_embedding: host_values(map, "visual.class_embedding", &[d])?,
            pos: weight(map, "visual.positional_embedding", &[grid * grid + 1, d])?,
            ln_pre: ln(map, "visual.ln_pre", d)?,
            blocks: blocks(map, "visual.transformer", &cfg)?,
            ln_post: ln(map, "visual.ln_post", d)?,
            proj: super::layers::linear_from(wt, None, embed, d)?,
            cfg,
        })
    }

    /// `[T, 3, S, S]` CLIP-normalized pixels to unit-norm `[T, embed]`.
    pub fn encode(&self, pixels: &CudaTensor) -> Result<CudaTensor> {
        let t = pixels.shape[0];
        let d = self.cfg.width;
        let p = self.patch;
        let patches = nn::conv2d(pixels, &self.conv1, 0, p)?; // [T, D, g, g]
        let patches = patches.flatten_from(2)?.transpose(1, 2)?; // [T, g*g, D]
        let cls = CudaTensor::from_vec(self.class_embedding.repeat(t), vec![t, 1, d])?;
        let x = CudaTensor::cat(&[&cls, &patches], 1)?.add(&self.pos)?;
        let mut x = x.layer_norm(1e-5, Some(&self.ln_pre.0), Some(&self.ln_pre.1))?;
        for b in &self.blocks {
            x = b.forward(&x, None)?;
        }
        let n = x.shape[1];
        let cls_rows: Vec<usize> = (0..t).map(|i| i * n).collect();
        let pooled = gather_rows(&x, &cls_rows)?; // [T, D]
        let pooled = pooled.layer_norm(1e-5, Some(&self.ln_post.0), Some(&self.ln_post.1))?;
        l2_normalize(&self.proj.forward(&pooled)?)
    }
}

/// `F.normalize(x, dim=-1)` (eps 1e-12 is below any norm we see): an RMS
/// norm with weight `1 / sqrt(D)`.
pub fn l2_normalize(x: &CudaTensor) -> Result<CudaTensor> {
    let d = *x.shape.last().ok_or_else(|| msg("normalize scalar"))?;
    let w = CudaTensor::from_vec(vec![1.0 / (d as f32).sqrt(); d], vec![d])?;
    x.rms_norm(&w, 0.0)
}

pub struct ClipText {
    pub cfg: ClipTowerConfig,
    /// `[V, D]`, kept on the host (a lookup of 77 rows per prompt).
    token_embedding: CudaTensor,
    pos: CudaTensor,
    blocks: Vec<ResBlock>,
    ln_final: (CudaTensor, CudaTensor),
    context: usize,
}

impl ClipText {
    pub fn h14_text() -> ClipTowerConfig {
        ClipTowerConfig { width: 1024, layers: 24, heads: 16 }
    }

    pub fn load(cfg: ClipTowerConfig, map: &WeightMap, vocab: usize, context: usize) -> Result<Self> {
        let d = cfg.width;
        Ok(Self {
            token_embedding: crate::wan::weights::cuda_tensor_shaped(map, "token_embedding.weight", &[vocab, d])?,
            pos: weight(map, "positional_embedding", &[context, d])?,
            blocks: blocks(map, "transformer", &cfg)?,
            ln_final: ln(map, "ln_final", d)?,
            context,
            cfg,
        })
    }

    /// Token ids (77) to `[1, 77, D]` per-token unit-norm features.
    pub fn encode(&self, ids: &[u32]) -> Result<CudaTensor> {
        let n = self.context;
        if ids.len() != n {
            return Err(msg(format!("clip text wants {n} ids, got {}", ids.len())));
        }
        let d = self.cfg.width;
        let idx: Vec<usize> = ids.iter().map(|&i| i as usize).collect();
        let x = self.token_embedding.embedding_rows(&idx)?.reshape(vec![1, n, d])?;
        let mut x = x.add(&self.pos)?;
        let mut mask = vec![0.0f32; n * n];
        for i in 0..n {
            for j in i + 1..n {
                mask[i * n + j] = f32::NEG_INFINITY;
            }
        }
        let mask = pinned(mask, vec![1, 1, n, n])?;
        for b in &self.blocks {
            x = b.forward(&x, Some(&mask))?;
        }
        let x = x.layer_norm(1e-5, Some(&self.ln_final.0), Some(&self.ln_final.1))?;
        l2_normalize(&x)
    }
}
