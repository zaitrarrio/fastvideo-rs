//! VSA-H3: the block-sparse attention FastH3 was distilled with (FastVideo
//! `attention/backends/video_sparse_attn_h3.py`), over the packed
//! `[text | audio | video]` sequence.
//!
//! It is Wan's VSA ([`crate::wan::vsa`]) with three differences, all of which
//! follow from the sequence not being a pure video grid:
//!
//! 1. **Tiles.** Text and audio rows come first and are cut into 64-row tiles
//!    that never mix the two ("segment-pure"); the video token grid is then cut
//!    into `(4, 4, 4)` cubes as usual. Partial tiles fill their first slots.
//! 2. **Selection ("exempt" mode).** Only video-to-video attention is
//!    sparsified. Per head, a query tile keeps the top `k_vid = ceil(0.2 *
//!    video_tiles)` *video* key tiles, and every text / audio key tile is
//!    always kept: they do not compete for the budget. Text and audio *query*
//!    tiles are dense: they see every tile.
//! 3. **The compression branch is trained.** `out = sparse + softmax(pooled
//!    scores over ALL tiles) @ pooled_v * to_gate_compress(x)`, on every row,
//!    prefix rows included. The base model zero-initialises that gate; this
//!    checkpoint ships 50 nonzero ones.
//!
//! On the device this reuses Wan's kernels unchanged. The forced prefix
//! columns are expressed by adding a huge bias to the prefix columns of a copy
//! of the coarse scores before the ordinary top-`(P + k_vid)`: every biased
//! column beats every video column, so the result is exactly "all prefix tiles
//! plus the top `k_vid` video tiles". The dense prefix query rows (a few
//! hundred) are recomputed with ordinary SDPA against all keys and spliced in.
//!
//! Top-k is discontinuous, so VSA output cannot be compared with a reference
//! at tight tolerance; it is validated structurally: sparsity 0 without the
//! gate equals dense attention, and the device path equals
//! [`attention_host`], the plain-loop statement of the rule above.
//! See docs/ports/h3.md, sections e and j.

use fastvideo_models::h3::config::H3InferenceContract;
use fastvideo_models::h3::packing::H3PackedLayout;

use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::vsa::{TilePlan, TILE_ELEMS};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H3VsaConfig {
    /// Fraction of *video* key tiles dropped per query tile.
    pub sparsity: f64,
    /// Query tiles per pass of the gather fine stage (memory bound; unused by
    /// the tensor-core kernel).
    pub group: usize,
}

impl H3VsaConfig {
    pub fn fasth3_8step() -> Self {
        Self { sparsity: H3InferenceContract::fasth3_8step().vsa_sparsity, group: crate::wan::envflag::usize_flag("FASTVIDEO_VSA_GROUP", 8).max(1) }
    }
}

/// Tiles over the packed sequence: `prefix_tiles` segment-pure 1-D tiles, then
/// the video cubes. Tile `i` owns slots `[64 i, 64 i + block_sizes[i])`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H3TilePlan {
    pub seq: usize,
    /// Text + audio rows; they are the first rows of the packed sequence.
    pub prefix_rows: usize,
    pub prefix_tiles: usize,
    pub video_tiles: usize,
    /// Packed row held by each slot, `-1` for padding.
    pub slot_src: Vec<i32>,
    pub block_sizes: Vec<u32>,
    /// Tile of each prefix row.
    pub prefix_row_tile: Vec<u32>,
}

impl H3TilePlan {
    pub fn new(layout: &H3PackedLayout) -> Result<Self> {
        if layout.text.start != 0 || layout.audio.start != layout.text.end() || layout.video.start != layout.audio.end() {
            return Err(msg("vsa-h3: the packed layout must be [text | audio | video]"));
        }
        let mut slot_src: Vec<i32> = Vec::new();
        let mut block_sizes: Vec<u32> = Vec::new();
        let mut prefix_row_tile = Vec::with_capacity(layout.video.start);
        // Zero-length segments are dropped; a segment's last tile may be short.
        for segment in [layout.text, layout.audio] {
            let mut row = segment.start;
            while row < segment.end() {
                let size = TILE_ELEMS.min(segment.end() - row);
                let tile = block_sizes.len() as u32;
                slot_src.extend((row..row + size).map(|r| r as i32));
                slot_src.extend(std::iter::repeat_n(-1, TILE_ELEMS - size));
                prefix_row_tile.extend(std::iter::repeat_n(tile, size));
                block_sizes.push(size as u32);
                row += size;
            }
        }
        let prefix_tiles = block_sizes.len();
        let video = TilePlan::new(layout.token_grid)?;
        if video.seq != layout.video.len {
            return Err(msg(format!("vsa-h3: grid {:?} holds {} tokens, the layout has {} video rows", layout.token_grid, video.seq, layout.video.len)));
        }
        let offset = layout.video.start as i32;
        slot_src.extend(video.slot_src.iter().map(|&s| if s < 0 { -1 } else { s + offset }));
        block_sizes.extend_from_slice(&video.block_sizes);
        Ok(Self {
            seq: layout.sequence_length(),
            prefix_rows: layout.video.start,
            prefix_tiles,
            video_tiles: video.num_tiles(),
            slot_src,
            block_sizes,
            prefix_row_tile,
        })
    }

    pub fn num_tiles(&self) -> usize {
        self.block_sizes.len()
    }

    /// `clamp(ceil((1 - sparsity) * video_tiles), 1, video_tiles)`.
    pub fn k_vid(&self, sparsity: f64) -> usize {
        crate::wan::vsa::topk_for(sparsity, self.video_tiles)
    }
}

/// The block mask of one head, "exempt" mode: `mask[i * n + j]` says whether
/// query tile `i` attends key tile `j`. `scores` is that head's `[n, n]`
/// pooled score matrix. Ties go to the lower tile index, as the device top-k.
pub fn block_mask(scores: &[f32], prefix_tiles: usize, k_vid: usize) -> Vec<bool> {
    let n = (scores.len() as f64).sqrt() as usize;
    let n_video = n - prefix_tiles;
    if k_vid >= n_video {
        return vec![true; n * n];
    }
    let mut mask = vec![false; n * n];
    for i in 0..n {
        let row = &scores[i * n..(i + 1) * n];
        let mut order: Vec<usize> = (prefix_tiles..n).collect();
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        for &j in &order[..k_vid] {
            mask[i * n + j] = true; // top-k over VIDEO columns only
        }
        for j in 0..prefix_tiles {
            mask[i * n + j] = true; // every query sees every text / audio tile
        }
        if i < prefix_tiles {
            mask[i * n..(i + 1) * n].fill(true); // text / audio queries are dense
        }
    }
    mask
}

/// VSA-H3 in plain loops: `[heads, seq, dim]` row-major inputs in packed
/// order. The statement of the algorithm the device path is judged against.
pub fn attention_host(q: &[f32], k: &[f32], v: &[f32], gate: Option<&[f32]>, plan: &H3TilePlan, k_vid: usize, heads: usize, dim: usize) -> Result<Vec<f32>> {
    let (seq, n) = (plan.seq, plan.num_tiles());
    let want = heads * seq * dim;
    if [q, k, v].iter().any(|b| b.len() != want) || gate.is_some_and(|g| g.len() != want) {
        return Err(msg(format!("vsa-h3: q/k/v/gate must each hold {want} values")));
    }
    let scale = 1.0 / (dim as f64).sqrt();
    let tile_rows = |t: usize| plan.slot_src[t * TILE_ELEMS..t * TILE_ELEMS + plan.block_sizes[t] as usize].iter().map(|&s| s as usize);
    // Heads are independent; rows within a head are what the loops spell out.
    use rayon::prelude::*;
    let mut out = vec![0f32; want];
    out.par_chunks_mut(seq * dim).enumerate().for_each(|(h, out)| {
        let base = h * seq * dim;
        // Mean over the VALID rows of each tile.
        let pool = |x: &[f32]| -> Vec<f64> {
            let mut p = vec![0f64; n * dim];
            for t in 0..n {
                for r in tile_rows(t) {
                    for d in 0..dim {
                        p[t * dim + d] += f64::from(x[base + r * dim + d]);
                    }
                }
                p[t * dim..(t + 1) * dim].iter_mut().for_each(|s| *s /= f64::from(plan.block_sizes[t]));
            }
            p
        };
        let (qbar, kbar, vbar) = (pool(q), pool(k), pool(v));
        let scores: Vec<f32> = (0..n * n)
            .map(|ij| ((0..dim).map(|d| qbar[(ij / n) * dim + d] * kbar[(ij % n) * dim + d]).sum::<f64>() * scale) as f32)
            .collect();
        let mask = block_mask(&scores, plan.prefix_tiles, k_vid);
        for i in 0..n {
            // Compression branch: softmax over ALL tiles, no mask.
            let mx = scores[i * n..(i + 1) * n].iter().cloned().fold(f32::MIN, f32::max);
            let weights: Vec<f64> = scores[i * n..(i + 1) * n].iter().map(|s| f64::from(s - mx).exp()).collect();
            let z: f64 = weights.iter().sum();
            let compressed: Vec<f64> = (0..dim).map(|d| (0..n).map(|j| weights[j] / z * vbar[j * dim + d]).sum()).collect();

            let keys: Vec<usize> = (0..n).filter(|&j| mask[i * n + j]).flat_map(tile_rows).collect();
            for r in tile_rows(i) {
                let logits: Vec<f64> = keys
                    .iter()
                    .map(|&kr| (0..dim).map(|d| f64::from(q[base + r * dim + d]) * f64::from(k[base + kr * dim + d])).sum::<f64>() * scale)
                    .collect();
                let mx = logits.iter().cloned().fold(f64::MIN, f64::max);
                let z: f64 = logits.iter().map(|l| (l - mx).exp()).sum();
                for d in 0..dim {
                    let sparse: f64 = keys.iter().zip(&logits).map(|(&kr, l)| (l - mx).exp() / z * f64::from(v[base + kr * dim + d])).sum();
                    let branch = gate.map_or(0.0, |g| compressed[d] * f64::from(g[base + r * dim + d]));
                    out[r * dim + d] = (sparse + branch) as f32;
                }
            }
        }
    });
    Ok(out)
}

/// VSA-H3 for one request: built once, shared by every block of every step.
pub struct H3Vsa {
    plan: H3TilePlan,
    k_vid: usize,
    heads: usize,
    head_dim: usize,
    #[cfg(feature = "cuda")]
    group: usize,
    #[cfg(feature = "cuda")]
    device: Option<DevicePlan>,
}

#[cfg(feature = "cuda")]
struct DevicePlan {
    plan: crate::wan::ops::VsaPlanDev,
    /// `[n]`: huge on the prefix columns, zero on the video ones.
    prefix_bias: CudaTensor,
    /// For each head and prefix row, the row of the `[heads * n, dim]`
    /// compression result that belongs to it.
    prefix_branch_rows: Vec<usize>,
}

/// Larger than any pooled score, small enough that `score + BIAS` is finite.
#[cfg(feature = "cuda")]
const PREFIX_BIAS: f32 = 1.0e30;

impl H3Vsa {
    pub fn new(layout: &H3PackedLayout, heads: usize, head_dim: usize, cfg: H3VsaConfig) -> Result<Self> {
        if !(0.0..=1.0).contains(&cfg.sparsity) || heads == 0 || head_dim == 0 {
            return Err(msg(format!("vsa-h3: sparsity {} with {heads} heads of {head_dim}", cfg.sparsity)));
        }
        let plan = H3TilePlan::new(layout)?;
        let k_vid = plan.k_vid(cfg.sparsity);
        crate::wan::log::info(format_args!(
            "vsa-h3: {} prefix + {} video tiles, k_vid {k_vid}, {} of {} rows dense",
            plan.prefix_tiles, plan.video_tiles, plan.prefix_rows, plan.seq
        ));
        #[cfg(feature = "cuda")]
        let device = if crate::wan::stats::device_expected() {
            let n = plan.num_tiles();
            let mut bias = CudaTensor::from_vec((0..n).map(|j| if j < plan.prefix_tiles { PREFIX_BIAS } else { 0.0 }).collect(), vec![n])?;
            bias.pin_device()?;
            let prefix_branch_rows = (0..heads).flat_map(|h| plan.prefix_row_tile.iter().map(move |&t| h * n + t as usize)).collect();
            Some(DevicePlan { plan: crate::wan::ops::vsa_plan_upload(&plan.slot_src, &plan.block_sizes, TILE_ELEMS)?, prefix_bias: bias, prefix_branch_rows })
        } else {
            None
        };
        Ok(Self {
            plan,
            k_vid,
            heads,
            head_dim,
            #[cfg(feature = "cuda")]
            group: cfg.group,
            #[cfg(feature = "cuda")]
            device,
        })
    }

    pub fn plan(&self) -> &H3TilePlan {
        &self.plan
    }

    pub fn k_vid(&self) -> usize {
        self.k_vid
    }

    /// `q`, `k`, `v` (post-norm, post-RoPE) and `gate` (`to_gate_compress`,
    /// neither normed nor rotated): `[1, H, S, D]` in packed order. Without a
    /// gate the compression branch is absent. Takes ownership so the inputs
    /// die as soon as their tiled copies exist.
    pub fn attend(&self, q: CudaTensor, k: CudaTensor, v: CudaTensor, gate: Option<CudaTensor>) -> Result<CudaTensor> {
        let shape = vec![1, self.heads, self.plan.seq, self.head_dim];
        if q.shape != shape || k.shape != shape || v.shape != shape || gate.as_ref().is_some_and(|g| g.shape != shape) {
            return Err(msg(format!("vsa-h3: q {:?} k {:?} v {:?} for a plan of {shape:?}", q.shape, k.shape, v.shape)));
        }
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.device {
            return self.attend_device(device, q, k, v, gate);
        }
        // CPU runs and tests only; with a device live this is an error, not a fallback.
        crate::wan::stats::host_fallback("vsa_h3", format_args!("{shape:?}"))?;
        let g = match &gate {
            Some(g) => Some(g.host_cow()?),
            None => None,
        };
        let out = attention_host(&q.host_cow()?, &k.host_cow()?, &v.host_cow()?, g.as_deref(), &self.plan, self.k_vid, self.heads, self.head_dim)?;
        CudaTensor::from_vec(out, shape)
    }

    #[cfg(feature = "cuda")]
    fn attend_device(&self, dp: &DevicePlan, q: CudaTensor, k: CudaTensor, v: CudaTensor, gate: Option<CudaTensor>) -> Result<CudaTensor> {
        use crate::wan::stats::phase;
        use crate::wan::{device, ops};
        let (bh, seq, dim, n) = (self.heads, self.plan.seq, self.head_dim, self.plan.num_tiles());
        let scale = 1.0 / (dim as f32).sqrt();
        let err = |e: device::DeviceError| msg(e.to_string());
        let on_device = |t: &CudaTensor| -> Result<CudaTensor> {
            let mut t = t.clone();
            t.ensure_device()?;
            Ok(t)
        };
        let (q, k, v) = (on_device(&q)?, on_device(&k)?, on_device(&v)?);
        let gate = gate.as_ref().map(on_device).transpose()?;
        fn slice(t: &CudaTensor) -> Result<&cudarc::driver::CudaSlice<f32>> {
            t.device_slice().ok_or_else(|| msg("vsa-h3: tensor has no device buffer"))
        }
        let (qd, kd, vd) = (slice(&q)?, slice(&k)?, slice(&v)?);

        // 1. Pooled scores in pinned float32: they decide the selection.
        let (scores, coarse) = phase("vsa_h3_1_coarse", || {
            let qc = ops::vsa_tile_mean_device(qd, &dp.plan, bh, seq, dim)?;
            let kc = ops::vsa_tile_mean_device(kd, &dp.plan, bh, seq, dim)?;
            let mut scores = ops::alloc(bh * n * n)?;
            device::matmul_linear_wt_strided_batched_f32(&qc, &kc, &mut scores, bh, n, dim, n, scale).map_err(err)?;
            // 2. Compression branch: softmax over all tiles, unmasked.
            let coarse = match &gate {
                Some(_) => {
                    let vc = ops::vsa_tile_mean_device(vd, &dp.plan, bh, seq, dim)?;
                    let probs = ops::softmax_last_device(&scores, n)?;
                    let mut coarse = ops::alloc(bh * n * dim)?;
                    device::matmul_2d_strided_batched_f32(&probs, &vc, &mut coarse, bh, n, n, dim).map_err(err)?;
                    coarse
                }
                // `vsa_combine` adds `coarse` ungated when there is no gate; H3
                // without a gate has no branch at all.
                None => ops::fill_device(bh * n * dim, 0.0)?,
            };
            Ok::<_, TensorError>((scores, coarse))
        })?;

        // 3. Selection: all prefix tiles + top-k_vid video tiles, as one top-k
        //    over scores whose prefix columns were lifted above everything.
        let topk = if self.k_vid >= self.plan.video_tiles { n } else { self.plan.prefix_tiles + self.k_vid };
        let selected = phase("vsa_h3_2_select", || {
            let biased = CudaTensor::from_device_slice(scores, vec![bh * n, n])?.add(&dp.prefix_bias)?;
            ops::vsa_topk_device(slice(&biased)?, bh * n, n, topk)
        })?;

        // 4. Fine stage over every query tile with that uniform list.
        let mut out = ops::fill_device(bh * seq * dim, 0.0)?;
        let gate_slice = gate.as_ref().map(slice).transpose()?;
        let sm = device::global_device().map_or(0, |d| d.sm_major);
        let forced_gather = crate::wan::envflag::string_flag("FASTVIDEO_VSA_KERNEL", "auto") == "gather";
        if sm >= 8 && dim == 128 && !forced_gather {
            let sparse = phase("vsa_h3_3_mma", || ops::vsa_mma_attn_device(qd, kd, vd, &selected, &dp.plan, bh, seq, dim, topk, scale))?;
            ops::vsa_combine_device(&sparse, &coarse, gate_slice, &dp.plan, &mut out, bh, n, 0, seq, dim)?;
        } else {
            let (rows, len) = (TILE_ELEMS, topk * TILE_ELEMS);
            let mut q_base = 0usize;
            while q_base < n {
                let g = self.group.min(n - q_base);
                let (kg, vg) = ops::vsa_gather_kv_device(kd, vd, &selected, &dp.plan, bh, g, seq, dim, topk, q_base)?;
                let qt = ops::vsa_gather_q_device(qd, &dp.plan, bh, g, q_base, seq, dim)?;
                let mut s = ops::alloc(bh * g * rows * len)?;
                device::matmul_linear_wt_strided_batched_bf16(&qt, &kg, &mut s, bh * g, rows, dim, len, scale).map_err(err)?;
                ops::vsa_mask_pad_device(&mut s, &selected, &dp.plan, bh, g, rows, topk, q_base)?;
                let p = ops::softmax_last_bf16_device(&s, len)?;
                drop(s);
                let mut sparse = ops::alloc(bh * g * rows * dim)?;
                device::matmul_2d_strided_batched_bf16(&p, &vg, &mut sparse, bh * g, rows, len, dim).map_err(err)?;
                ops::vsa_combine_device(&sparse, &coarse, gate_slice, &dp.plan, &mut out, bh, g, q_base, seq, dim)?;
                q_base += g;
            }
        }
        let out = CudaTensor::from_device_slice(out, vec![1, bh, seq, dim])?;
        let prefix = self.plan.prefix_rows;
        if topk == n || prefix == 0 {
            return Ok(out); // every query already saw every tile
        }

        // 5. Text / audio query rows are dense: recompute those few rows
        //    against all keys and add their compression term.
        phase("vsa_h3_4_prefix_dense", || {
            let mut dense = crate::wan::nn::scaled_dot_product_attention(&q.narrow(2, 0, prefix)?, &k, &v, Some(scale))?;
            if let Some(g) = &gate {
                let branch = CudaTensor::from_device_slice(coarse, vec![bh * n, dim])?.index_select_rows(&dp.prefix_branch_rows)?;
                dense = dense.add(&branch.reshape(vec![1, bh, prefix, dim])?.mul(&g.narrow(2, 0, prefix)?)?)?;
            }
            CudaTensor::cat(&[&dense, &out.narrow(2, prefix, seq - prefix)?], 2)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wan::nn::scaled_dot_product_attention;

    fn seeded(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * k).sin() * 1.7 + (i as f32 * 0.013).cos()) * 0.6).collect()
    }

    /// 70 text rows, 2 x 33 audio rows, a 5 x 4 x 6 video grid.
    fn layout() -> H3PackedLayout {
        H3PackedLayout::new(70, (5, 8, 12), 33, [1, 2, 2]).unwrap()
    }

    #[test]
    fn prefix_tiles_are_segment_pure_and_video_tiles_are_cubes() {
        let l = layout();
        let p = H3TilePlan::new(&l).unwrap();
        // text 70 -> 64 + 6; audio 66 -> 64 + 2; video 5x4x6 -> 2 x 1 x 2 cubes.
        assert_eq!((p.prefix_tiles, p.video_tiles, p.prefix_rows), (4, 4, 136));
        assert_eq!(&p.block_sizes[..4], &[64, 6, 64, 2]);
        assert_eq!(&p.block_sizes[4..], &[4 * 4 * 4, 4 * 4 * 2, 4 * 4, 4 * 2]);
        assert_eq!(p.slot_src[64 + 5], 69, "last text row");
        assert_eq!(p.slot_src[64 + 6], -1, "a text tile is never topped up with audio");
        assert_eq!(p.slot_src[2 * 64], 70, "audio starts its own tile");
        assert_eq!(p.slot_src[4 * 64], 136, "first video token");
        assert_eq!(p.slot_src[4 * 64 + 1], 137);
        assert_eq!((p.prefix_row_tile[63], p.prefix_row_tile[64], p.prefix_row_tile[70], p.prefix_row_tile[135]), (0, 1, 2, 3));
        // Every packed row is in exactly one slot.
        let mut seen = vec![0u8; p.seq];
        p.slot_src.iter().filter(|&&s| s >= 0).for_each(|&s| seen[s as usize] += 1);
        assert!(seen.iter().all(|&c| c == 1));
    }

    #[test]
    fn the_spec_tile_counts_hold_for_5_and_15_seconds() {
        use fastvideo_models::h3::config::H3Geometry;
        for (seconds, prefix, video, k_vid) in [(5, 11, 660, 132), (15, 23, 1782, 357)] {
            let l = H3PackedLayout::from_geometry(&H3Geometry::default_16x9(seconds).unwrap(), 256).unwrap();
            let p = H3TilePlan::new(&l).unwrap();
            assert_eq!((p.prefix_tiles, p.video_tiles, p.k_vid(0.8)), (prefix, video, k_vid), "{seconds} s");
        }
    }

    #[test]
    fn the_mask_exempts_prefix_columns_and_keeps_prefix_queries_dense() {
        // 2 prefix + 4 video tiles, k_vid = 1. Scores favour column 0 everywhere
        // and, among video columns, column 3 for even rows and 5 for odd.
        let n = 6;
        let scores: Vec<f32> = (0..n * n)
            .map(|ij| {
                let (i, j) = (ij / n, ij % n);
                if j == 0 { 9.0 } else if j == 3 + 2 * (i % 2) { 5.0 } else { j as f32 * 0.1 }
            })
            .collect();
        let mask = block_mask(&scores, 2, 1);
        let row = |i: usize| -> Vec<usize> { (0..n).filter(|&j| mask[i * n + j]).collect() };
        assert_eq!(row(0), vec![0, 1, 2, 3, 4, 5], "a prefix query tile sees everything");
        assert_eq!(row(1), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(row(2), vec![0, 1, 3], "prefix columns forced; one video column by score");
        assert_eq!(row(3), vec![0, 1, 5]);
        // A prefix column with a poor score still does not cost a video slot.
        assert!(mask[4 * n + 1] && scores[4 * n + 1] < scores[4 * n + 4]);
        assert!(block_mask(&scores, 2, 4).iter().all(|&m| m), "k_vid = all video tiles is dense");
    }

    #[test]
    fn sparsity_zero_without_the_gate_is_dense_attention() {
        let l = layout();
        let (heads, dim, seq) = (2usize, 8usize, l.sequence_length());
        let shape = vec![1, heads, seq, dim];
        let t = |k: f32| CudaTensor::from_vec(seeded(heads * seq * dim, k), shape.clone()).unwrap();
        let (q, k, v) = (t(0.37), t(0.71), t(0.19));
        let dense = scaled_dot_product_attention(&q, &k, &v, None).unwrap();
        let vsa = H3Vsa::new(&l, heads, dim, H3VsaConfig { sparsity: 0.0, group: 2 }).unwrap();
        let got = vsa.attend(q, k, v, None).unwrap();
        for (i, (g, w)) in got.host_cow().unwrap().iter().zip(dense.host_cow().unwrap().iter()).enumerate() {
            assert!((g - w).abs() < 2e-5, "value {i}: {g} vs {w}");
        }
    }

    #[test]
    fn with_the_gate_at_sparsity_zero_the_difference_is_the_compression_term() {
        let l = layout();
        let (heads, dim, seq) = (1usize, 4usize, l.sequence_length());
        let shape = vec![1, heads, seq, dim];
        let t = |k: f32| CudaTensor::from_vec(seeded(heads * seq * dim, k), shape.clone()).unwrap();
        let (q, k, v, gate) = (t(0.37), t(0.71), t(0.19), t(0.53));
        let vsa = H3Vsa::new(&l, heads, dim, H3VsaConfig { sparsity: 0.0, group: 2 }).unwrap();
        let plan = vsa.plan().clone();
        let dense = scaled_dot_product_attention(&q, &k, &v, None).unwrap();
        let (qh, kh, vh, gh) = (q.host_cow().unwrap().into_owned(), k.host_cow().unwrap().into_owned(), v.host_cow().unwrap().into_owned(), gate.host_cow().unwrap().into_owned());
        let got = vsa.attend(q, k, v, Some(gate)).unwrap();
        let (got, dense) = (got.host_cow().unwrap(), dense.host_cow().unwrap());

        // Compression term of tile i, from first principles.
        let n = plan.num_tiles();
        let rows = |t: usize| plan.slot_src[t * 64..t * 64 + plan.block_sizes[t] as usize].iter().map(|&s| s as usize).collect::<Vec<_>>();
        let pool = |x: &[f32], t: usize| -> Vec<f32> { (0..dim).map(|d| rows(t).iter().map(|&r| x[r * dim + d]).sum::<f32>() / plan.block_sizes[t] as f32).collect() };
        let mut differs = false;
        for i in 0..n {
            let qb = pool(&qh, i);
            let logits: Vec<f32> = (0..n).map(|j| qb.iter().zip(pool(&kh, j)).map(|(a, b)| a * b).sum::<f32>() / (dim as f32).sqrt()).collect();
            let z: f32 = logits.iter().map(|s| s.exp()).sum();
            for &r in &rows(i) {
                for d in 0..dim {
                    let term: f32 = (0..n).map(|j| logits[j].exp() / z * pool(&vh, j)[d]).sum::<f32>() * gh[r * dim + d];
                    let delta = got[r * dim + d] - dense[r * dim + d];
                    assert!((delta - term).abs() < 5e-5, "tile {i} row {r} ch {d}: {delta} vs {term}");
                    differs |= term.abs() > 1e-3;
                }
            }
        }
        assert!(differs, "a zero compression term proves nothing");
    }

    #[test]
    fn sparse_video_rows_attend_only_their_selected_tiles() {
        let l = layout();
        let (heads, dim, seq) = (2usize, 4usize, l.sequence_length());
        let (q, k, v) = (seeded(heads * seq * dim, 0.37), seeded(heads * seq * dim, 0.71), seeded(heads * seq * dim, 0.19));
        let plan = H3TilePlan::new(&l).unwrap();
        let k_vid = plan.k_vid(0.5);
        assert_eq!(k_vid, 2);
        let sparse = attention_host(&q, &k, &v, None, &plan, k_vid, heads, dim).unwrap();
        let dense = attention_host(&q, &k, &v, None, &plan, plan.video_tiles, heads, dim).unwrap();
        let prefix = plan.prefix_rows * dim;
        for h in 0..heads {
            let base = h * seq * dim;
            assert!(sparse[base..base + prefix].iter().zip(&dense[base..base + prefix]).all(|(a, b)| (a - b).abs() < 1e-6), "prefix rows stay dense");
            assert!(sparse[base + prefix..base + seq * dim].iter().zip(&dense[base + prefix..base + seq * dim]).any(|(a, b)| (a - b).abs() > 1e-4), "video rows are sparsified");
        }
        // Changing a video key in a tile nobody selected must change nothing
        // for video queries... but every prefix query still sees it.
        let shape = vec![1, heads, seq, dim];
        let vsa = H3Vsa::new(&l, heads, dim, H3VsaConfig { sparsity: 0.5, group: 2 }).unwrap();
        let t = |x: &[f32]| CudaTensor::from_vec(x.to_vec(), shape.clone()).unwrap();
        let via_struct = vsa.attend(t(&q), t(&k), t(&v), None).unwrap();
        assert_eq!(&*via_struct.host_cow().unwrap(), &sparse[..], "H3Vsa::attend is the host reference on CPU runs");
    }
}
