//! Qwen3-VL vision tower for MiniMax-H3 multimodal text
//! (`MiniMaxH3Qwen3VLVisionModel`).
//!
//! Patch-embed → interpolated absolute PE → ViT blocks (windowed by image) →
//! spatial merger (+ DeepStack mergers at layers 8/16/24). Language-model
//! injection (scatter into embeds + DeepStack residual) lives in the text
//! encoder path.

use fastvideo_models::h3::config::H3VisionConfig;
use fastvideo_models::h3::vision_preprocess::{PreparedVisionImage, VisionGrid};

use crate::wan::nn::{scaled_dot_product_attention, Linear};
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

fn pinned(data: Vec<f32>, shape: Vec<usize>) -> Result<CudaTensor> {
    let mut t = CudaTensor::from_vec(data, shape)?;
    t.pin_device()?;
    Ok(t)
}

fn host_values(map: &WeightMap, key: &str, shape: &[usize]) -> Result<Vec<f32>> {
    Ok(cuda_tensor_shaped(map, key, shape)?
        .host_cow()?
        .into_owned())
}

fn rotate_half_host(x: &[f32], dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0f32; x.len()];
    for (row, chunk) in x.chunks_exact(dim).enumerate() {
        for i in 0..half {
            out[row * dim + i] = -chunk[i + half];
            out[row * dim + i + half] = chunk[i];
        }
    }
    out
}

/// Merger: LayerNorm → Linear → GELU → Linear, with optional post-shuffle norm.
struct PatchMerger {
    norm_w: CudaTensor,
    norm_b: CudaTensor,
    fc1: Linear,
    fc2: Linear,
    use_postshuffle_norm: bool,
    hidden: usize,
}

impl PatchMerger {
    fn load(
        map: &WeightMap,
        prefix: &str,
        cfg: &H3VisionConfig,
        use_postshuffle_norm: bool,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
        let norm_size = if use_postshuffle_norm {
            hidden
        } else {
            cfg.hidden_size
        };
        Ok(Self {
            norm_w: pinned(
                host_values(map, &format!("{prefix}.norm.weight"), &[norm_size])?,
                vec![norm_size],
            )?,
            norm_b: pinned(
                host_values(map, &format!("{prefix}.norm.bias"), &[norm_size])?,
                vec![norm_size],
            )?,
            fc1: Linear::load(map, &format!("{prefix}.linear_fc1"), hidden, hidden, true)?,
            fc2: Linear::load(
                map,
                &format!("{prefix}.linear_fc2"),
                hidden,
                cfg.out_hidden_size,
                true,
            )?,
            use_postshuffle_norm,
            hidden,
        })
    }

    fn forward(&self, x: &CudaTensor, merge: usize) -> Result<CudaTensor> {
        // x: [tokens, C] before spatial merge grouping.
        let [n, c] = x.shape[..] else {
            return Err(msg(format!("merger expects [N,C], got {:?}", x.shape)));
        };
        let area = merge * merge;
        if n % area != 0 {
            return Err(msg(format!(
                "merger: {n} tokens not divisible by merge²={area}"
            )));
        }
        let groups = n / area;
        let x = if self.use_postshuffle_norm {
            x.reshape(vec![groups, self.hidden])?.layer_norm(
                1e-6,
                Some(&self.norm_w),
                Some(&self.norm_b),
            )?
        } else {
            let nrm = x.layer_norm(1e-6, Some(&self.norm_w), Some(&self.norm_b))?;
            nrm.reshape(vec![groups, self.hidden])?
        };
        let _ = c;
        let h = self.fc1.forward(&x)?.gelu_erf();
        self.fc2.forward(&h)
    }
}

struct VisionBlock {
    norm1_w: CudaTensor,
    norm1_b: CudaTensor,
    norm2_w: CudaTensor,
    norm2_b: CudaTensor,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionBlock {
    fn load(map: &WeightMap, prefix: &str, cfg: &H3VisionConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            norm1_w: pinned(
                host_values(map, &format!("{prefix}.norm1.weight"), &[h])?,
                vec![h],
            )?,
            norm1_b: pinned(
                host_values(map, &format!("{prefix}.norm1.bias"), &[h])?,
                vec![h],
            )?,
            norm2_w: pinned(
                host_values(map, &format!("{prefix}.norm2.weight"), &[h])?,
                vec![h],
            )?,
            norm2_b: pinned(
                host_values(map, &format!("{prefix}.norm2.bias"), &[h])?,
                vec![h],
            )?,
            qkv: Linear::load(map, &format!("{prefix}.attn.qkv"), h, 3 * h, true)?,
            proj: Linear::load(map, &format!("{prefix}.attn.proj"), h, h, true)?,
            fc1: Linear::load(
                map,
                &format!("{prefix}.mlp.linear_fc1"),
                h,
                cfg.intermediate_size,
                true,
            )?,
            fc2: Linear::load(
                map,
                &format!("{prefix}.mlp.linear_fc2"),
                cfg.intermediate_size,
                h,
                true,
            )?,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(
        &self,
        x: &CudaTensor,
        seq_lens: &[usize],
        cos: &CudaTensor,
        sin: &CudaTensor,
    ) -> Result<CudaTensor> {
        let n1 = x.layer_norm(1e-6, Some(&self.norm1_w), Some(&self.norm1_b))?;
        let a = self.attn(&n1, seq_lens, cos, sin)?;
        let x = x.add(&a)?;
        let n2 = x.layer_norm(1e-6, Some(&self.norm2_w), Some(&self.norm2_b))?;
        let m = self.fc2.forward(&self.fc1.forward(&n2)?.gelu_tanh())?;
        x.add(&m)
    }

    fn attn(
        &self,
        x: &CudaTensor,
        seq_lens: &[usize],
        cos: &CudaTensor,
        sin: &CudaTensor,
    ) -> Result<CudaTensor> {
        let [n, _] = x.shape[..] else {
            return Err(msg(format!("vision attn expects [N,C], got {:?}", x.shape)));
        };
        let qkv = self.qkv.forward(x)?; // [N, 3H]
        let hd = self.head_dim;
        let nh = self.num_heads;
        // Host RoPE + per-image SDPA: vision sequences are short.
        let qkv_h = qkv.host_cow()?.into_owned();
        let cos_h = cos.host_cow()?.into_owned();
        let sin_h = sin.host_cow()?.into_owned();
        let mut q = vec![0f32; n * nh * hd];
        let mut k = vec![0f32; n * nh * hd];
        let mut v = vec![0f32; n * nh * hd];
        for t in 0..n {
            for h in 0..nh {
                for d in 0..hd {
                    q[t * nh * hd + h * hd + d] = qkv_h[t * 3 * nh * hd + h * hd + d];
                    k[t * nh * hd + h * hd + d] = qkv_h[t * 3 * nh * hd + nh * hd + h * hd + d];
                    v[t * nh * hd + h * hd + d] = qkv_h[t * 3 * nh * hd + 2 * nh * hd + h * hd + d];
                }
            }
        }
        // Apply rotate-half RoPE with shared cos/sin [N, hd].
        let apply = |tensor: &mut [f32]| {
            let rotated = rotate_half_host(tensor, hd);
            for t in 0..n {
                for h in 0..nh {
                    for d in 0..hd {
                        let i = t * nh * hd + h * hd + d;
                        let c = cos_h[t * hd + d];
                        let s = sin_h[t * hd + d];
                        tensor[i] = tensor[i] * c + rotated[i] * s;
                    }
                }
            }
        };
        apply(&mut q);
        apply(&mut k);

        let mut out = vec![0f32; n * nh * hd];
        let mut offset = 0usize;
        let scale = 1.0 / (hd as f32).sqrt();
        for &slen in seq_lens {
            let q_t = CudaTensor::from_vec(
                q[offset * nh * hd..(offset + slen) * nh * hd].to_vec(),
                vec![1, nh, slen, hd],
            )?;
            let k_t = CudaTensor::from_vec(
                k[offset * nh * hd..(offset + slen) * nh * hd].to_vec(),
                vec![1, nh, slen, hd],
            )?;
            let v_t = CudaTensor::from_vec(
                v[offset * nh * hd..(offset + slen) * nh * hd].to_vec(),
                vec![1, nh, slen, hd],
            )?;
            let a = scaled_dot_product_attention(&q_t, &k_t, &v_t, Some(scale))?;
            let ah = a.host_cow()?.into_owned();
            // [1, H, S, D] → [S, H, D]
            for t in 0..slen {
                for h in 0..nh {
                    for d in 0..hd {
                        out[(offset + t) * nh * hd + h * hd + d] = ah[h * slen * hd + t * hd + d];
                    }
                }
            }
            offset += slen;
        }
        let flat = CudaTensor::from_vec(out, vec![n, nh * hd])?.to_device()?;
        self.proj.forward(&flat)
    }
}

/// Full vision tower. Checkpoint keys under `model.visual.*`.
pub struct H3VisionTower {
    cfg: H3VisionConfig,
    patch_weight: CudaTensor,
    patch_bias: CudaTensor,
    pos_embed: CudaTensor,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    deepstack: Vec<PatchMerger>,
    deepstack_indexes: [usize; 3],
    inv_freq: Vec<f32>,
}

impl H3VisionTower {
    pub fn load(cfg: H3VisionConfig, map: &WeightMap) -> Result<Self> {
        let prefix = "model.visual";
        let (cin, hs, pt, tt) = (
            cfg.in_channels,
            cfg.hidden_size,
            cfg.patch_size,
            cfg.temporal_patch_size,
        );
        // Conv3d weight [out, in, Tt, P, P].
        let patch_weight = pinned(
            host_values(
                map,
                &format!("{prefix}.patch_embed.proj.weight"),
                &[hs, cin, tt, pt, pt],
            )?,
            vec![hs, cin, tt, pt, pt],
        )?;
        let patch_bias = pinned(
            host_values(map, &format!("{prefix}.patch_embed.proj.bias"), &[hs])?,
            vec![hs],
        )?;
        let pos_embed = pinned(
            host_values(
                map,
                &format!("{prefix}.pos_embed.weight"),
                &[cfg.num_position_embeddings, hs],
            )?,
            vec![cfg.num_position_embeddings, hs],
        )?;
        let blocks = (0..cfg.depth)
            .map(|i| VisionBlock::load(map, &format!("{prefix}.blocks.{i}"), &cfg))
            .collect::<Result<Vec<_>>>()?;
        let merger = PatchMerger::load(map, &format!("{prefix}.merger"), &cfg, false)?;
        let deepstack = cfg
            .deepstack_visual_indexes
            .iter()
            .enumerate()
            .map(|(i, _)| {
                PatchMerger::load(
                    map,
                    &format!("{prefix}.deepstack_merger_list.{i}"),
                    &cfg,
                    true,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let half = cfg.head_dim() / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / cfg.head_dim() as f32))
            .collect();
        Ok(Self {
            deepstack_indexes: cfg.deepstack_visual_indexes,
            cfg,
            patch_weight,
            patch_bias,
            pos_embed,
            blocks,
            merger,
            deepstack,
            inv_freq,
        })
    }

    pub fn config(&self) -> &H3VisionConfig {
        &self.cfg
    }

    /// Encode prepared patches → `(merged [tokens, 5120], deepstack×3)`.
    pub fn forward(&self, prepared: &PreparedVisionImage) -> Result<(CudaTensor, Vec<CudaTensor>)> {
        self.forward_grids(
            &prepared.pixels,
            prepared.patch_dim,
            &[prepared.grid.clone()],
        )
    }

    pub fn forward_grids(
        &self,
        pixels: &[f32],
        patch_dim: usize,
        grids: &[VisionGrid],
    ) -> Result<(CudaTensor, Vec<CudaTensor>)> {
        let total: usize = grids.iter().map(|g| g.num_patches()).sum();
        if pixels.len() != total * patch_dim {
            return Err(msg(format!(
                "vision: {} values for {total} patches of dim {patch_dim}",
                pixels.len()
            )));
        }
        let (cin, tt, pt, hs) = (
            self.cfg.in_channels,
            self.cfg.temporal_patch_size,
            self.cfg.patch_size,
            self.cfg.hidden_size,
        );
        // [N, C, T, P, P]
        let x = CudaTensor::from_vec(pixels.to_vec(), vec![total, cin, tt, pt, pt])?.to_device()?;
        let mut h = x
            .conv3d(
                &self.patch_weight,
                Some(&self.patch_bias),
                [0, 0, 0],
                [tt, pt, pt],
            )?
            .reshape(vec![total, hs])?;
        h = h.add(&self.interpolate_pos(grids)?)?;

        let (cos, sin) = self.rope_tables(grids)?;
        let seq_lens: Vec<usize> = grids
            .iter()
            .flat_map(|g| std::iter::repeat_n(g.height * g.width, g.temporal))
            .collect();

        let mut deepstack = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, &seq_lens, &cos, &sin)?;
            if let Some(di) = self.deepstack_indexes.iter().position(|&idx| idx == i) {
                deepstack.push(self.deepstack[di].forward(&h, self.cfg.spatial_merge_size)?);
            }
        }
        let merged = self.merger.forward(&h, self.cfg.spatial_merge_size)?;
        Ok((merged, deepstack))
    }

    fn interpolate_pos(&self, grids: &[VisionGrid]) -> Result<CudaTensor> {
        // Bilinear interpolate the square PE grid into each image's H×W in merge order.
        let side = self.cfg.num_grid_per_side();
        let hs = self.cfg.hidden_size;
        let merge = self.cfg.spatial_merge_size;
        let pe = self.pos_embed.host_cow()?.into_owned();
        let mut out = Vec::new();
        for g in grids {
            let (gh, gw) = (g.height, g.width);
            let (mh, mw) = (gh / merge, gw / merge);
            for _t in 0..g.temporal {
                for bh in 0..mh {
                    for bw in 0..mw {
                        for ih in 0..merge {
                            for iw in 0..merge {
                                let r = bh * merge + ih;
                                let c = bw * merge + iw;
                                let y = if gh == 1 {
                                    0.0
                                } else {
                                    r as f64 * (side - 1) as f64 / (gh - 1) as f64
                                };
                                let x = if gw == 1 {
                                    0.0
                                } else {
                                    c as f64 * (side - 1) as f64 / (gw - 1) as f64
                                };
                                let y0 = y.floor() as usize;
                                let x0 = x.floor() as usize;
                                let y1 = (y0 + 1).min(side - 1);
                                let x1 = (x0 + 1).min(side - 1);
                                let fy = (y - y0 as f64) as f32;
                                let fx = (x - x0 as f64) as f32;
                                for d in 0..hs {
                                    let v00 = pe[(y0 * side + x0) * hs + d];
                                    let v01 = pe[(y0 * side + x1) * hs + d];
                                    let v10 = pe[(y1 * side + x0) * hs + d];
                                    let v11 = pe[(y1 * side + x1) * hs + d];
                                    let top = v00 * (1.0 - fx) + v01 * fx;
                                    let bot = v10 * (1.0 - fx) + v11 * fx;
                                    out.push(top * (1.0 - fy) + bot * fy);
                                }
                            }
                        }
                    }
                }
            }
        }
        CudaTensor::from_vec(out, vec![grids.iter().map(|g| g.num_patches()).sum(), hs])?
            .to_device()
    }

    fn rope_tables(&self, grids: &[VisionGrid]) -> Result<(CudaTensor, CudaTensor)> {
        let hd = self.cfg.head_dim();
        let half = hd / 2;
        let merge = self.cfg.spatial_merge_size;
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for g in grids {
            let (gh, gw) = (g.height, g.width);
            let (mh, mw) = (gh / merge, gw / merge);
            for _t in 0..g.temporal {
                for bh in 0..mh {
                    for bw in 0..mw {
                        for ih in 0..merge {
                            for iw in 0..merge {
                                let r = (bh * merge + ih) as f64;
                                let c = (bw * merge + iw) as f64;
                                let mut freqs = vec![0f32; half];
                                for k in 0..half / 2 {
                                    freqs[k] = (r * self.inv_freq[k] as f64) as f32;
                                }
                                for k in half / 2..half {
                                    freqs[k] = (c * self.inv_freq[k] as f64) as f32;
                                }
                                // cat(freqs, freqs) → [hd]
                                for &f in freqs.iter().chain(freqs.iter()) {
                                    cos.push(f.cos());
                                    sin.push(f.sin());
                                }
                            }
                        }
                    }
                }
            }
        }
        let n = grids.iter().map(|g| g.num_patches()).sum::<usize>();
        Ok((
            CudaTensor::from_vec(cos, vec![n, hd])?.to_device()?,
            CudaTensor::from_vec(sin, vec![n, hd])?.to_device()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_models::h3::vision_preprocess::prepare_vision_image;

    fn tiny_cfg() -> H3VisionConfig {
        let mut c = H3VisionConfig::fasth3_8step();
        c.depth = 3;
        c.hidden_size = 32;
        c.intermediate_size = 64;
        c.num_heads = 4;
        c.patch_size = 4;
        c.spatial_merge_size = 2;
        c.temporal_patch_size = 2;
        c.out_hidden_size = 16;
        c.num_position_embeddings = 64; // 8×8
        c.deepstack_visual_indexes = [0, 1, 2];
        c.min_pixels = 16 * 16;
        c.max_pixels = 64 * 64;
        c
    }

    fn weights(_cfg: &H3VisionConfig) -> WeightMap {
        WeightMap::generated(|key, shape| {
            let seed = key
                .bytes()
                .fold(3u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
            let n: usize = shape.iter().product();
            (0..n)
                .map(|i| {
                    let u = (seed.wrapping_add(i as u32).wrapping_mul(2_654_435_761) >> 8) as f32
                        / (1u32 << 24) as f32;
                    if key.contains("norm") {
                        0.5 + u
                    } else {
                        (u - 0.5) * 0.02
                    }
                })
                .collect()
        })
    }

    #[test]
    fn tiny_tower_runs() {
        let cfg = tiny_cfg();
        let map = weights(&cfg);
        // Remap keys: generated map needs exact shapes — load will request them.
        // Build a map that answers any key with the requested shape.
        let tower = H3VisionTower::load(cfg.clone(), &map).unwrap();
        let rgb = vec![90u8; 32 * 32 * 3];
        let prep = prepare_vision_image(&rgb, 32, 32, &cfg).unwrap();
        let (merged, deep) = tower.forward(&prep).unwrap();
        let tokens = prep.grid.num_tokens(cfg.spatial_merge_size).unwrap();
        assert_eq!(merged.shape, vec![tokens, cfg.out_hidden_size]);
        assert_eq!(deep.len(), 3);
    }
}
