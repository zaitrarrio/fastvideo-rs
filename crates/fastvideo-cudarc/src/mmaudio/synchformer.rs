//! Synchformer's visual half (`ext/synchformer`: `MotionFormer`, divided
//! space-time ViT-B/16 over 16-frame segments, `factorize_space_time`,
//! spatial `TransformerEncoderLayer` aggregation, identity time aggregation)
//! as `FeaturesUtils.encode_video_with_sync` runs it: 25 fps frames
//! `[T, 3, 224, 224]` in `[-1, 1]`, segments of 16 with stride 8, 8 tokens per
//! segment, `[1, 8 S, 768]` out.
//!
//! Per segment: `patch_embed_3d` (a `(2, 16, 16)` conv at its own stride:
//! 8 x 14 x 14 tokens), a CLS token, spatial position embeddings tiled over
//! time plus `temp_embed`, then 12 `DividedSpaceTimeBlock`s:
//! `x += timeattn(norm3(x))` (each spatial position attends over its 8
//! frames plus the CLS key/value; the CLS query attends to every token),
//! `x += attn(norm1(x))` (each frame over its 196 patches plus CLS), `x +=
//! mlp(norm2(x))`. Then `norm` on the patch tokens and, per frame, a
//! pre-norm encoder layer whose CLS output is that frame's feature.
//!
//! Segments start at even frames, so segment `s`'s temporal patches are
//! patches `4 s .. 4 s + 8` of one strided conv over the whole clip: the
//! patch embedding runs once, not once per overlapping segment.

use super::layers::{gather_rows, host_values, linear, msg, weight};
use crate::wan::nn::{self, Linear};
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::WeightMap;

pub const SYNC_DIM: usize = 768;
pub const SEGMENT_SIZE: usize = 16;
pub const STEP_SIZE: usize = 8;
pub const FRAME_SIZE: usize = 224;
const HEADS: usize = 12;
const T_TOK: usize = 8; // temporal patches per segment
const GRID: usize = 14;
const HW: usize = GRID * GRID; // 196
const TOK: usize = 1 + T_TOK * HW; // 1569

/// Number of segments for `t` frames (`(t - 16) // 8 + 1`).
pub fn num_segments(t: usize) -> usize {
    if t < SEGMENT_SIZE {
        0
    } else {
        (t - SEGMENT_SIZE) / STEP_SIZE + 1
    }
}

fn ln(map: &WeightMap, p: &str) -> Result<(CudaTensor, CudaTensor)> {
    Ok((
        weight(map, &format!("{p}.weight"), &[SYNC_DIM])?,
        weight(map, &format!("{p}.bias"), &[SYNC_DIM])?,
    ))
}

fn norm(x: &CudaTensor, w: &(CudaTensor, CudaTensor)) -> Result<CudaTensor> {
    x.layer_norm(1e-6, Some(&w.0), Some(&w.1))
}

/// `[G, S, 3D]` rows to heads `[G, H, S, 64]` of part `i` (q/k/v).
fn heads(qkv: &CudaTensor, i: usize, from: usize) -> Result<CudaTensor> {
    let [g, s, _] = qkv.shape[..] else {
        return Err(msg(format!("synchformer qkv {:?}", qkv.shape)));
    };
    let d = SYNC_DIM;
    qkv.narrow(2, i * d, d)?
        .narrow(1, from, s - from)?
        .reshape(vec![g, s - from, HEADS, d / HEADS])?
        .permute(&[0, 2, 1, 3])
}

fn merge(o: &CudaTensor) -> Result<CudaTensor> {
    let [g, h, s, hd] = o.shape[..] else {
        return Err(msg("synchformer attn out"));
    };
    o.permute(&[0, 2, 1, 3])?.reshape(vec![g, s, h * hd])
}

/// Row index plans of the divided attention for `bs` segments.
struct Plan {
    /// Per (b, n): CLS then frames 0..8 at patch n: `[bs * 196 * 9]`.
    time: Vec<usize>,
    /// Back from (b, n, f) to (b, f, n): `[bs * 1568]`.
    time_back: Vec<usize>,
    /// Per (b, f): CLS then patches 0..196: `[bs * 8 * 197]`.
    space: Vec<usize>,
    /// The CLS rows: `[bs]`.
    cls: Vec<usize>,
    /// The patch rows in order: `[bs * 1568]`.
    patches: Vec<usize>,
}

impl Plan {
    fn new(bs: usize) -> Self {
        let mut p = Plan {
            time: Vec::with_capacity(bs * HW * (T_TOK + 1)),
            time_back: Vec::with_capacity(bs * T_TOK * HW),
            space: Vec::with_capacity(bs * T_TOK * (HW + 1)),
            cls: (0..bs).map(|b| b * TOK).collect(),
            patches: Vec::with_capacity(bs * T_TOK * HW),
        };
        for b in 0..bs {
            let base = b * TOK;
            for n in 0..HW {
                p.time.push(base);
                for f in 0..T_TOK {
                    p.time.push(base + 1 + f * HW + n);
                }
            }
            for f in 0..T_TOK {
                for n in 0..HW {
                    // row of (b, n, f) in the time-grouped output
                    p.time_back.push((b * HW + n) * T_TOK + f);
                    p.patches.push(base + 1 + f * HW + n);
                }
            }
            for f in 0..T_TOK {
                p.space.push(base);
                for n in 0..HW {
                    p.space.push(base + 1 + f * HW + n);
                }
            }
        }
        p
    }
}

struct DividedAttn {
    qkv: Linear,
    proj: Linear,
}

impl DividedAttn {
    fn load(map: &WeightMap, p: &str) -> Result<Self> {
        Ok(Self {
            qkv: linear(map, &format!("{p}.qkv"), SYNC_DIM, 3 * SYNC_DIM, true)?,
            proj: linear(map, &format!("{p}.proj"), SYNC_DIM, SYNC_DIM, true)?,
        })
    }

    /// `x`: `[bs, 1569, D]` (already normed). `time`: attend over frames.
    fn forward(&self, x: &CudaTensor, plan: &Plan, time: bool) -> Result<CudaTensor> {
        let bs = x.shape[0];
        let d = SYNC_DIM;
        let qkv = self.qkv.forward(x)?; // [bs, 1569, 3D]
        // CLS query against every token of its segment.
        let q_cls = gather_rows(&qkv, &plan.cls)?.reshape(vec![bs, 1, 3 * d])?;
        let cls_out = nn::scaled_dot_product_attention(
            &heads(&q_cls, 0, 0)?,
            &heads(&qkv, 1, 0)?,
            &heads(&qkv, 2, 0)?,
            None,
        )?;
        let cls_out = merge(&cls_out)?; // [bs, 1, D]
        let (idx, groups, len) = if time {
            (&plan.time, bs * HW, T_TOK + 1)
        } else {
            (&plan.space, bs * T_TOK, HW + 1)
        };
        let g = gather_rows(&qkv, idx)?.reshape(vec![groups, len, 3 * d])?;
        let o = nn::scaled_dot_product_attention(&heads(&g, 0, 1)?, &heads(&g, 1, 0)?, &heads(&g, 2, 0)?, None)?;
        let o = merge(&o)?; // [groups, len - 1, D]
        let o = if time {
            gather_rows(&o, &plan.time_back)?
        } else {
            o
        }
        .reshape(vec![bs, T_TOK * HW, d])?;
        let out = CudaTensor::cat(&[&cls_out, &o], 1)?;
        self.proj.forward(&out)
    }
}

struct Block {
    norm1: (CudaTensor, CudaTensor),
    norm2: (CudaTensor, CudaTensor),
    norm3: (CudaTensor, CudaTensor),
    attn: DividedAttn,
    timeattn: DividedAttn,
    fc1: Linear,
    fc2: Linear,
}

impl Block {
    fn load(map: &WeightMap, p: &str) -> Result<Self> {
        Ok(Self {
            norm1: ln(map, &format!("{p}.norm1"))?,
            norm2: ln(map, &format!("{p}.norm2"))?,
            norm3: ln(map, &format!("{p}.norm3"))?,
            attn: DividedAttn::load(map, &format!("{p}.attn"))?,
            timeattn: DividedAttn::load(map, &format!("{p}.timeattn"))?,
            fc1: linear(map, &format!("{p}.mlp.fc1"), SYNC_DIM, 4 * SYNC_DIM, true)?,
            fc2: linear(map, &format!("{p}.mlp.fc2"), 4 * SYNC_DIM, SYNC_DIM, true)?,
        })
    }

    fn forward(&self, x: &CudaTensor, plan: &Plan) -> Result<CudaTensor> {
        let x = x.add(&self.timeattn.forward(&norm(x, &self.norm3)?, plan, true)?)?;
        let x = x.add(&self.attn.forward(&norm(&x, &self.norm1)?, plan, false)?)?;
        let h = self.fc1.forward(&norm(&x, &self.norm2)?)?.gelu_erf();
        x.add(&self.fc2.forward(&h)?)
    }
}

/// `SpatialTransformerEncoderLayer` (pre-norm `nn.TransformerEncoderLayer`
/// with a prepended CLS; only the CLS row is kept, so only it is computed).
struct SpatialAgg {
    cls: Vec<f32>,
    in_proj: Linear,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
    norm1: (CudaTensor, CudaTensor),
    norm2: (CudaTensor, CudaTensor),
}

impl SpatialAgg {
    fn load(map: &WeightMap, p: &str) -> Result<Self> {
        let w = host_values(map, &format!("{p}.self_attn.in_proj_weight"), &[3 * SYNC_DIM, SYNC_DIM])?;
        let b = host_values(map, &format!("{p}.self_attn.in_proj_bias"), &[3 * SYNC_DIM])?;
        Ok(Self {
            cls: host_values(map, &format!("{p}.cls_token"), &[1, 1, SYNC_DIM])?,
            in_proj: super::layers::linear_from(w, Some(b), 3 * SYNC_DIM, SYNC_DIM)?,
            out_proj: linear(map, &format!("{p}.self_attn.out_proj"), SYNC_DIM, SYNC_DIM, true)?,
            linear1: linear(map, &format!("{p}.linear1"), SYNC_DIM, 4 * SYNC_DIM, true)?,
            linear2: linear(map, &format!("{p}.linear2"), 4 * SYNC_DIM, SYNC_DIM, true)?,
            norm1: ln(map, &format!("{p}.norm1"))?,
            norm2: ln(map, &format!("{p}.norm2"))?,
        })
    }

    /// `[G, 196, D]` (one frame of one segment per row group) to `[G, D]`.
    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let g = x.shape[0];
        let d = SYNC_DIM;
        let cls = CudaTensor::from_vec(self.cls.repeat(g), vec![g, 1, d])?;
        let x = CudaTensor::cat(&[&cls, x], 1)?; // [G, 197, D]
        let qkv = self.in_proj.forward(&norm(&x, &self.norm1)?)?;
        let q = heads(&qkv.narrow(1, 0, 1)?, 0, 0)?;
        let o = nn::scaled_dot_product_attention(&q, &heads(&qkv, 1, 0)?, &heads(&qkv, 2, 0)?, None)?;
        let c = cls.add(&self.out_proj.forward(&merge(&o)?)?)?; // [G, 1, D]
        let h = self.linear1.forward(&norm(&c, &self.norm2)?)?.gelu_erf();
        c.add(&self.linear2.forward(&h)?)?.reshape(vec![g, d])
    }
}

pub struct Synchformer {
    patch_w: CudaTensor,
    patch_b: CudaTensor,
    cls_token: Vec<f32>,
    /// `pos_embed[0]` (CLS) and the tiled spatial + temporal table `[1568, D]`.
    cls_pos: Vec<f32>,
    patch_pos: CudaTensor,
    blocks: Vec<Block>,
    norm: (CudaTensor, CudaTensor),
    agg: SpatialAgg,
}

impl Synchformer {
    /// Keys under `vfeat_extractor.` of `synchformer_state_dict.pth`.
    pub fn load(map: &WeightMap) -> Result<Self> {
        let p = "vfeat_extractor";
        let d = SYNC_DIM;
        let pos = host_values(map, &format!("{p}.pos_embed"), &[1, HW + 1, d])?;
        let temp = host_values(map, &format!("{p}.temp_embed"), &[1, T_TOK, d])?;
        let mut table = Vec::with_capacity(T_TOK * HW * d);
        for f in 0..T_TOK {
            for n in 0..HW {
                for c in 0..d {
                    table.push(pos[(1 + n) * d + c] + temp[f * d + c]);
                }
            }
        }
        Ok(Self {
            patch_w: weight(map, &format!("{p}.patch_embed_3d.proj.weight"), &[d, 3, 2, 16, 16])?,
            patch_b: weight(map, &format!("{p}.patch_embed_3d.proj.bias"), &[d])?,
            cls_token: host_values(map, &format!("{p}.cls_token"), &[1, 1, d])?,
            cls_pos: pos[..d].to_vec(),
            patch_pos: super::layers::pinned(table, vec![T_TOK * HW, d])?,
            blocks: (0..12)
                .map(|i| Block::load(map, &format!("{p}.blocks.{i}")))
                .collect::<Result<Vec<_>>>()?,
            norm: ln(map, &format!("{p}.norm"))?,
            agg: SpatialAgg::load(map, &format!("{p}.spatial_attn_agg"))?,
        })
    }

    /// `[T, 3, 224, 224]` sync frames to `[1, 8 S, 768]`.
    pub fn encode(&self, frames: &CudaTensor) -> Result<CudaTensor> {
        let t = frames.shape[0];
        let segs = num_segments(t);
        if segs == 0 {
            return Err(msg(format!("synchformer needs >= 16 frames, got {t}")));
        }
        let d = SYNC_DIM;
        // One strided patch conv over the clip: [1, D, T/2, 14, 14].
        let x = frames.permute(&[1, 0, 2, 3])?.unsqueeze(0)?; // [1, 3, T, H, W]
        let p = x.conv3d(&self.patch_w, Some(&self.patch_b), [0, 0, 0], [2, 16, 16])?;
        let tp = p.shape[2];
        let p = p.reshape(vec![d, tp * HW])?.transpose(0, 1)?; // [tp*196, D]
        // Segment s: temporal patches 4s .. 4s+8.
        let idx: Vec<usize> = (0..segs)
            .flat_map(|s| (0..T_TOK * HW).map(move |r| 4 * s * HW + r))
            .collect();
        let tokens = p.index_select_rows(&idx)?.reshape(vec![segs, T_TOK * HW, d])?;
        let tokens = tokens.add(&self.patch_pos)?;
        let cls: Vec<f32> = self.cls_token.iter().zip(&self.cls_pos).map(|(a, b)| a + b).collect();
        let cls = CudaTensor::from_vec(cls.repeat(segs), vec![segs, 1, d])?;
        let mut x = CudaTensor::cat(&[&cls, &tokens], 1)?; // [S, 1569, D]
        let plan = Plan::new(segs);
        for b in &self.blocks {
            x = b.forward(&x, &plan)?;
        }
        let pt = gather_rows(&x, &plan.patches)?.reshape(vec![segs * T_TOK, HW, d])?;
        let pt = norm(&pt, &self.norm)?;
        self.agg.forward(&pt)?.reshape(vec![1, segs * T_TOK, d])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments() {
        assert_eq!(num_segments(125), 14);
        assert_eq!(num_segments(200), 24);
        assert_eq!(num_segments(15), 0);
    }

    #[test]
    fn plan_shapes() {
        let p = Plan::new(2);
        assert_eq!(p.time.len(), 2 * 196 * 9);
        assert_eq!(p.space.len(), 2 * 8 * 197);
        assert_eq!(p.time_back.len(), 2 * 1568);
        // (b=0, f=1, n=0) sits at time-grouped row (n=0)*8 + 1.
        assert_eq!(p.time_back[196], 1);
        assert_eq!(p.patches[0], 1);
    }
}
