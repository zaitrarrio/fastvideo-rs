//! Video Sparse Attention (upstream FastVideo's VSA), ported.
//!
//! Dense attention over an 8s clip is 48k tokens of quadratic work. VSA cuts
//! the quadratic term by scoring whole spatiotemporal tiles first and running
//! fine-grained attention only inside the tiles that score highest:
//!
//! 1. **Tile** the `(T, H, W)` latent grid into `(4, 4, 4)` cubes, each padded
//!    to 64 slots. Tokens keep the cube's row-major order; padding sits at the
//!    end of each tile.
//! 2. **Coarse**: mean-pool Q/K/V per tile and run dense attention over the
//!    tiles (819 of them for an 8s clip, so this is cheap). Each tile's result
//!    broadcasts back to its tokens.
//! 3. **Sparse**: take the top-k tiles per query tile from those same coarse
//!    scores and attend at full resolution only to those.
//! 4. **Combine**: `out = coarse * gate + sparse`, where `gate` is the
//!    checkpoint's `to_gate_compress` projection of the same normed hidden
//!    states Q/K/V come from (no RoPE, no norm).
//!
//! With `sparsity = 0.8` a query tile sees 164 of 819 tiles, so the fine stage
//! does ~4.6x less score work than dense.
//!
//! This module holds the geometry and the host reference. The host path is a
//! test oracle and a CPU-run fallback, not something a GPU run may reach:
//! [`super::stats::host_fallback`] still guards the device entry points.

use super::tensor::{Result, TensorError};

/// Upstream's `VSA_TILE_SIZE`: 64 tokens per tile.
pub const TILE: (usize, usize, usize) = (4, 4, 4);
/// Slots per tile, padding included.
pub const TILE_ELEMS: usize = TILE.0 * TILE.1 * TILE.2;

/// Tiles to keep for a sparsity level, as upstream's `compute_topk`.
pub fn topk_for(sparsity: f64, num_tiles: usize) -> usize {
    let keep = ((1.0 - sparsity) * num_tiles as f64).ceil() as usize;
    keep.clamp(1, num_tiles.max(1))
}

/// Tiling geometry for one latent grid: which token lands in which padded slot,
/// and how many real tokens each tile holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePlan {
    pub grid: (usize, usize, usize),
    pub tiles: (usize, usize, usize),
    /// Real tokens, `T*H*W`.
    pub seq: usize,
    /// Slots, `num_tiles * TILE_ELEMS`.
    pub padded: usize,
    /// Source token for each padded slot, `-1` where the slot is padding.
    pub slot_src: Vec<i32>,
    /// Padded slot holding each token: the inverse of the non-pad `slot_src`.
    pub token_slot: Vec<u32>,
    /// Real tokens in each tile.
    pub block_sizes: Vec<u32>,
}

impl TilePlan {
    pub fn num_tiles(&self) -> usize {
        self.block_sizes.len()
    }

    /// Build the plan for a `(T, H, W)` grid of patch tokens.
    pub fn new(grid: (usize, usize, usize)) -> Result<Self> {
        let (t, h, w) = grid;
        if t == 0 || h == 0 || w == 0 {
            return Err(TensorError::Message(format!("vsa: empty grid {grid:?}")));
        }
        let (ts, hs, ws) = TILE;
        let tiles = (t.div_ceil(ts), h.div_ceil(hs), w.div_ceil(ws));
        let num_tiles = tiles.0 * tiles.1 * tiles.2;
        let seq = t * h * w;
        let padded = num_tiles * TILE_ELEMS;

        let mut slot_src = vec![-1i32; padded];
        let mut token_slot = vec![0u32; seq];
        let mut block_sizes = vec![0u32; num_tiles];
        // Tiles in (t, h, w) order; tokens inside a tile keep the sub-cube's
        // row-major order, so a partial tile fills its first slots only.
        let mut tile = 0usize;
        for tt in 0..tiles.0 {
            for hh in 0..tiles.1 {
                for ww in 0..tiles.2 {
                    let mut slot = tile * TILE_ELEMS;
                    for z in tt * ts..((tt + 1) * ts).min(t) {
                        for y in hh * hs..((hh + 1) * hs).min(h) {
                            for x in ww * ws..((ww + 1) * ws).min(w) {
                                let token = (z * h + y) * w + x;
                                slot_src[slot] = token as i32;
                                token_slot[token] = slot as u32;
                                slot += 1;
                            }
                        }
                    }
                    block_sizes[tile] = (slot - tile * TILE_ELEMS) as u32;
                    tile += 1;
                }
            }
        }
        Ok(Self { grid, tiles, seq, padded, slot_src, token_slot, block_sizes })
    }
}

/// Host reference for one VSA attention call, `[b, heads, seq, dim]` inputs in
/// row-major order. `gate` is `to_gate_compress`, already split into heads.
///
/// The oracle for the device kernels; also the shape the kernels implement.
#[allow(clippy::too_many_arguments)]
pub fn vsa_attention_host(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: Option<&[f32]>,
    plan: &TilePlan,
    topk: usize,
    b: usize,
    heads: usize,
    dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    let (seq, nb) = (plan.seq, plan.num_tiles());
    let want = b * heads * seq * dim;
    for (name, buf) in [("q", q), ("k", k), ("v", v)] {
        if buf.len() != want {
            return Err(TensorError::Message(format!("vsa: {name} has {} elements, want {want}", buf.len())));
        }
    }
    if gate.is_some_and(|g| g.len() != want) {
        return Err(TensorError::Message("vsa: gate shape must match q".into()));
    }
    let topk = topk.clamp(1, nb);
    let mut out = vec![0.0f32; want];

    for bh in 0..b * heads {
        let base = bh * seq * dim;
        fn row(buf: &[f32], base: usize, token: usize, dim: usize) -> &[f32] {
            let s = base + token * dim;
            &buf[s..s + dim]
        }

        // 1. Tile means. Padding contributes nothing and the divisor is the
        //    tile's real token count, so a partial tile is not diluted.
        let mut qc = vec![0.0f32; nb * dim];
        let mut kc = vec![0.0f32; nb * dim];
        let mut vc = vec![0.0f32; nb * dim];
        for tile in 0..nb {
            let n = plan.block_sizes[tile] as f32;
            if n == 0.0 {
                continue;
            }
            for slot in tile * TILE_ELEMS..tile * TILE_ELEMS + plan.block_sizes[tile] as usize {
                let token = plan.slot_src[slot] as usize;
                let (qr, kr, vr) = (row(q, base, token, dim), row(k, base, token, dim), row(v, base, token, dim));
                for d in 0..dim {
                    qc[tile * dim + d] += qr[d];
                    kc[tile * dim + d] += kr[d];
                    vc[tile * dim + d] += vr[d];
                }
            }
            for d in 0..dim {
                qc[tile * dim + d] /= n;
                kc[tile * dim + d] /= n;
                vc[tile * dim + d] /= n;
            }
        }

        // 2. Coarse attention over tiles, in f64 to keep the oracle exact.
        let mut coarse = vec![0.0f32; nb * dim];
        let mut scores = vec![0.0f32; nb * nb];
        for qt in 0..nb {
            let s = &mut scores[qt * nb..(qt + 1) * nb];
            for (kt, slot) in s.iter_mut().enumerate() {
                let dot: f64 =
                    (0..dim).map(|d| f64::from(qc[qt * dim + d]) * f64::from(kc[kt * dim + d])).sum();
                *slot = (dot * f64::from(scale)) as f32;
            }
            let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f64> = s.iter().map(|x| f64::from(x - m).exp()).collect();
            let z: f64 = exps.iter().sum();
            for (kt, e) in exps.iter().enumerate() {
                let wgt = e / z;
                for d in 0..dim {
                    coarse[qt * dim + d] += (wgt * f64::from(vc[kt * dim + d])) as f32;
                }
            }
        }

        // 3. Top-k tiles per query tile, ties broken by the lower index so the
        //    selection is deterministic.
        let mut selected = vec![0u32; nb * topk];
        let mut order: Vec<u32> = (0..nb as u32).collect();
        for qt in 0..nb {
            let s = &scores[qt * nb..(qt + 1) * nb];
            order.sort_by(|&a, &b| {
                s[b as usize].total_cmp(&s[a as usize]).then(a.cmp(&b))
            });
            selected[qt * topk..(qt + 1) * topk].copy_from_slice(&order[..topk]);
            order.sort_unstable();
        }

        // 4. Fine attention inside the selected tiles, then combine.
        for qt in 0..nb {
            let picks = &selected[qt * topk..(qt + 1) * topk];
            for qslot in qt * TILE_ELEMS..qt * TILE_ELEMS + plan.block_sizes[qt] as usize {
                let qtok = plan.slot_src[qslot] as usize;
                let qr = row(q, base, qtok, dim);
                let mut m = f32::NEG_INFINITY;
                let mut acc = vec![0.0f64; dim];
                let mut z = 0.0f64;
                for &kt in picks {
                    let kt = kt as usize;
                    for kslot in kt * TILE_ELEMS..kt * TILE_ELEMS + plan.block_sizes[kt] as usize {
                        let ktok = plan.slot_src[kslot] as usize;
                        let dot: f64 =
                            (0..dim).map(|d| f64::from(qr[d]) * f64::from(k[base + ktok * dim + d])).sum();
                        let s = (dot * f64::from(scale)) as f32;
                        // Online softmax: rescale the running sums when a new
                        // maximum arrives, so long tile lists stay stable.
                        if s > m {
                            let shrink = if m.is_finite() { f64::from(m - s).exp() } else { 0.0 };
                            z *= shrink;
                            for a in acc.iter_mut() {
                                *a *= shrink;
                            }
                            m = s;
                        }
                        let e = f64::from(s - m).exp();
                        z += e;
                        for d in 0..dim {
                            acc[d] += e * f64::from(v[base + ktok * dim + d]);
                        }
                    }
                }
                let dst = base + qtok * dim;
                for d in 0..dim {
                    let sparse = if z > 0.0 { (acc[d] / z) as f32 } else { 0.0 };
                    let c = coarse[qt * dim + d] * gate.map_or(1.0, |g| g[dst + d]);
                    out[dst + d] = c + sparse;
                }
            }
        }
    }
    Ok(out)
}

/// Which fine-stage implementation runs.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FineKernel {
    /// Gather selected K/V into dense buffers per query tile, then cuBLAS.
    /// Materialises every tile ~topk times; measured at ~21% of the DiT.
    Gather,
    /// The scalar-f32 fused kernel: streams K/V, but no tensor cores. Kept
    /// as the measured negative result it is (8x slower than Gather).
    FusedScalar,
    /// Streams K/V through `mma.sync` bf16 tensor cores with an online
    /// softmax, as the reference kernels do. Needs sm80+ and dim 128.
    Mma,
}

/// `FASTVIDEO_VSA_KERNEL=auto|gather|fused|mma`. `FASTVIDEO_VSA_FUSED=1` is
/// the old spelling of `fused`.
///
/// `auto` picks by hardware: the tensor-core kernel wherever `mma.sync` bf16
/// exists (sm80+) at Wan's geometry, the gather path otherwise. It became the
/// default in the commit that recorded its verification — kernels tier on an
/// RTX 5090, rel_l2 0.0029/0.0031 against the host reference, 2.98x / 2.85x /
/// 5.42x faster than the gather path at 1,456 / 4,368 / 13,104 tokens. Its
/// first run had been selected by default *before* verification, on a build
/// where a missing arch mapping compiled the body out; a default may only
/// point at a verified kernel, which is why the flip is its own commit.
#[cfg(feature = "cuda")]
fn fine_kernel(dim: usize, tile_elems: usize) -> FineKernel {
    use super::envflag::{bool_flag, string_flag};
    let pick = string_flag("FASTVIDEO_VSA_KERNEL", "auto");
    let sm = super::device::global_device().map(|d| d.sm_major).unwrap_or(0);
    let mma_ok = sm >= 8 && dim == 128 && tile_elems == TILE_ELEMS;
    let chosen = match pick.as_str() {
        "gather" => FineKernel::Gather,
        "fused" => FineKernel::FusedScalar,
        "mma" => FineKernel::Mma,
        _ if bool_flag("FASTVIDEO_VSA_FUSED", false) => FineKernel::FusedScalar,
        _ if mma_ok => FineKernel::Mma,
        _ => FineKernel::Gather,
    };
    // An explicit ask for a kernel the hardware cannot run is an error at the
    // call site, not a silent substitution; `auto` is the only fallback.
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    super::log::info_once(&LOGGED, format_args!("vsa fine kernel: {chosen:?} (sm{sm}, dim {dim}, {pick})"));
    chosen
}

/// Everything a VSA layer needs for one latent grid: the uploaded tiling, how
/// many tiles each query tile attends to, and the query-tile group size that
/// bounds the gathered buffer. Built once per grid and shared by every layer
/// and denoising step.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct VsaCtx {
    pub plan: super::ops::VsaPlanDev,
    pub topk: usize,
    pub seq: usize,
    pub group: usize,
}

#[cfg(feature = "cuda")]
impl VsaCtx {
    /// Build from a grid. `sparsity` follows upstream's `VSA_sparsity`.
    pub fn new(grid: (usize, usize, usize), sparsity: f64, group: usize) -> Result<Self> {
        let plan = TilePlan::new(grid)?;
        let topk = topk_for(sparsity, plan.num_tiles());
        let dev = super::ops::vsa_plan_upload(&plan.slot_src, &plan.block_sizes, TILE_ELEMS)?;
        Ok(Self { plan: dev, topk, seq: plan.seq, group: group.max(1) })
    }
}

/// VSA on device, `[b, heads, seq, dim]` in and out.
///
/// The fine stage gathers each query tile's selected K/V into a dense buffer
/// and runs a batched GEMM, rather than a fused attention kernel: it reuses the
/// bf16 tensor-core path, and the gather still moves far less than dense
/// attention's score matrix. Query tiles are processed in groups so the
/// gathered buffer stays bounded.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn vsa_attention_device(
    q: &cudarc::driver::CudaSlice<f32>,
    k: &cudarc::driver::CudaSlice<f32>,
    v: &cudarc::driver::CudaSlice<f32>,
    gate: Option<&cudarc::driver::CudaSlice<f32>>,
    plan: &super::ops::VsaPlanDev,
    topk: usize,
    bh: usize,
    seq: usize,
    dim: usize,
    scale: f32,
    group: usize,
) -> Result<cudarc::driver::CudaSlice<f32>> {
    use super::{device, ops};
    let nb = plan.num_tiles;
    let topk = topk.clamp(1, nb);
    let err = |e: device::DeviceError| TensorError::Message(e.to_string());

    use super::stats::phase;
    // 1. Tile means, then coarse attention over tiles.
    let (qc, kc, vc) = phase("vsa_1_tile_mean", || {
        Ok::<_, TensorError>((
            ops::vsa_tile_mean_device(q, plan, bh, seq, dim)?,
            ops::vsa_tile_mean_device(k, plan, bh, seq, dim)?,
            ops::vsa_tile_mean_device(v, plan, bh, seq, dim)?,
        ))
    })?;
    let (scores, coarse) = phase("vsa_2_coarse", || {
        let mut scores = ops::alloc(bh * nb * nb)?;
        device::matmul_linear_wt_strided_batched_f32(&qc, &kc, &mut scores, bh, nb, dim, nb, scale).map_err(err)?;
        let probs = ops::softmax_last_device(&scores, nb)?;
        let mut coarse = ops::alloc(bh * nb * dim)?;
        device::matmul_2d_strided_batched_f32(&probs, &vc, &mut coarse, bh, nb, nb, dim).map_err(err)?;
        Ok::<_, TensorError>((scores, coarse))
    })?;

    // 2. Tiles to attend to, from the same coarse scores (pre-softmax order is
    //    the same, but top-k on the raw scores matches upstream).
    let selected = phase("vsa_3_topk", || ops::vsa_topk_device(&scores, bh * nb, nb, topk))?;
    drop(scores);

    // 3. Fine stage. The fused kernel streams K/V from the tiled layout; the
    //    gather path materialises them and lets cuBLAS use tensor cores. Which
    //    is faster is hardware-dependent, so it is a flag, not a decision.
    let mut out = ops::fill_device(bh * seq * dim, 0.0)?;
    match fine_kernel(dim, plan.tile_elems) {
        FineKernel::Mma => {
            let sparse = phase("vsa_4_mma", || {
                ops::vsa_mma_attn_device(q, k, v, &selected, plan, bh, seq, dim, topk, scale)
            })?;
            phase("vsa_8_combine", || {
                ops::vsa_combine_device(&sparse, &coarse, gate, plan, &mut out, bh, nb, 0, seq, dim)
            })?;
            return Ok(out);
        }
        FineKernel::FusedScalar => {
            let sparse = ops::vsa_fused_attn_device(q, k, v, &selected, plan, bh, seq, dim, topk, scale)?;
            ops::vsa_combine_device(&sparse, &coarse, gate, plan, &mut out, bh, nb, 0, seq, dim)?;
            return Ok(out);
        }
        FineKernel::Gather => {}
    }
    let len = topk * plan.tile_elems;
    let group = group.clamp(1, nb);
    let mut q_base = 0usize;
    while q_base < nb {
        let g = group.min(nb - q_base);
        // The gather is what the rejected fused kernel existed to remove; it is
        // timed separately so its real share is a number rather than a guess.
        let (kg, vg) = phase("vsa_4_gather_kv", || {
            ops::vsa_gather_kv_device(k, v, &selected, plan, bh, g, seq, dim, topk, q_base)
        })?;
        // Queries for these tiles, in padded slot order.
        let qt = phase("vsa_5_gather_q", || ops::vsa_gather_q_device(q, plan, bh, g, q_base, seq, dim))?;
        let rows = plan.tile_elems;
        let p = phase("vsa_6_fine_qk_softmax", || {
            let mut s = ops::alloc(bh * g * rows * len)?;
            device::matmul_linear_wt_strided_batched_bf16(&qt, &kg, &mut s, bh * g, rows, dim, len, scale)
                .map_err(err)?;
            ops::vsa_mask_pad_device(&mut s, &selected, plan, bh, g, rows, topk, q_base)?;
            ops::softmax_last_bf16_device(&s, len)
        })?;
        let sparse = phase("vsa_7_fine_pv", || {
            let mut sparse = ops::alloc(bh * g * rows * dim)?;
            device::matmul_2d_strided_batched_bf16(&p, &vg, &mut sparse, bh * g, rows, len, dim).map_err(err)?;
            Ok::<_, TensorError>(sparse)
        })?;
        phase("vsa_8_combine", || {
            ops::vsa_combine_device(&sparse, &coarse, gate, plan, &mut out, bh, g, q_base, seq, dim)
        })?;
        q_base += g;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_of(grid: (usize, usize, usize)) -> TilePlan {
        TilePlan::new(grid).unwrap()
    }

    #[test]
    fn tiling_is_a_permutation_with_padding() {
        for grid in [(4, 4, 4), (33, 28, 52), (3, 5, 7), (1, 1, 1)] {
            let p = plan_of(grid);
            assert_eq!(p.seq, grid.0 * grid.1 * grid.2);
            assert_eq!(p.padded, p.num_tiles() * TILE_ELEMS);
            // Every token appears exactly once, and slot_src/token_slot agree.
            let mut seen = vec![false; p.seq];
            for (slot, &src) in p.slot_src.iter().enumerate() {
                if src < 0 {
                    continue;
                }
                let token = src as usize;
                assert!(!seen[token], "token {token} twice");
                seen[token] = true;
                assert_eq!(p.token_slot[token] as usize, slot);
            }
            assert!(seen.iter().all(|&s| s), "grid {grid:?} lost a token");
            assert_eq!(p.block_sizes.iter().map(|&n| n as usize).sum::<usize>(), p.seq);
            assert!(p.block_sizes.iter().all(|&n| n as usize <= TILE_ELEMS));
        }
    }

    #[test]
    fn full_tiles_have_no_padding() {
        let p = plan_of((8, 8, 8));
        assert!(p.slot_src.iter().all(|&s| s >= 0));
        assert!(p.block_sizes.iter().all(|&n| n as usize == TILE_ELEMS));
    }

    #[test]
    fn partial_tile_keeps_subcube_order() {
        // 1x1x5 -> two tiles along w: 4 tokens then 1, the rest padding.
        let p = plan_of((1, 1, 5));
        assert_eq!(p.num_tiles(), 2);
        assert_eq!(p.block_sizes, vec![4, 1]);
        assert_eq!(&p.slot_src[..4], &[0, 1, 2, 3]);
        assert!(p.slot_src[4..TILE_ELEMS].iter().all(|&s| s == -1));
        assert_eq!(p.slot_src[TILE_ELEMS], 4);
    }

    #[test]
    fn topk_matches_upstream_rule() {
        assert_eq!(topk_for(0.8, 819), 164);
        assert_eq!(topk_for(0.0, 819), 819);
        assert_eq!(topk_for(1.0, 819), 1); // never below one tile
        assert_eq!(topk_for(0.9, 10), 1);
    }

    /// The load-bearing invariant: with every tile selected and the gate zero,
    /// VSA's fine stage is exactly dense attention.
    #[test]
    fn all_tiles_selected_and_zero_gate_is_dense_attention() {
        let (b, heads, dim) = (1, 2, 8);
        let grid = (2, 3, 5);
        let plan = plan_of(grid);
        let seq = plan.seq;
        let n = b * heads * seq * dim;
        let mk = |seed: u64| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                })
                .collect()
        };
        let (q, k, v) = (mk(1), mk(2), mk(3));
        let gate = vec![0.0f32; n];
        let scale = 1.0 / (dim as f32).sqrt();

        let got = vsa_attention_host(&q, &k, &v, Some(&gate), &plan, plan.num_tiles(), b, heads, dim, scale)
            .unwrap();

        // Plain dense attention over the same tokens.
        let mut want = vec![0.0f32; n];
        for bh in 0..b * heads {
            let base = bh * seq * dim;
            for i in 0..seq {
                let s: Vec<f32> = (0..seq)
                    .map(|j| {
                        (0..dim).map(|d| q[base + i * dim + d] * k[base + j * dim + d]).sum::<f32>() * scale
                    })
                    .collect();
                let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f64> = s.iter().map(|x| f64::from(x - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for d in 0..dim {
                    let acc: f64 = (0..seq).map(|j| e[j] * f64::from(v[base + j * dim + d])).sum();
                    want[base + i * dim + d] = (acc / z) as f32;
                }
            }
        }
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 2e-5, "element {i}: vsa {g} vs dense {w}");
        }
    }

    /// With a unit gate and every tile selected, the coarse branch adds exactly
    /// the tile-mean attention on top.
    #[test]
    fn gate_adds_the_coarse_branch() {
        let (b, heads, dim) = (1, 1, 4);
        let plan = plan_of((1, 4, 4));
        let n = b * heads * plan.seq * dim;
        let q: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
        let k: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.2 - 0.4).collect();
        let v: Vec<f32> = (0..n).map(|i| (i % 3) as f32 * 0.3 - 0.3).collect();
        let scale = 1.0 / (dim as f32).sqrt();
        let nb = plan.num_tiles();

        let zero = vsa_attention_host(&q, &k, &v, Some(&vec![0.0; n]), &plan, nb, b, heads, dim, scale).unwrap();
        let one = vsa_attention_host(&q, &k, &v, Some(&vec![1.0; n]), &plan, nb, b, heads, dim, scale).unwrap();
        let none = vsa_attention_host(&q, &k, &v, None, &plan, nb, b, heads, dim, scale).unwrap();
        // gate=None means a gate of one, and `one - zero` is exactly the
        // coarse term: individual components can vanish, so compare in bulk.
        for i in 0..n {
            assert!((one[i] - none[i]).abs() < 1e-6, "gate=1 must equal gate=None at {i}");
        }
        let spread = one.iter().zip(&zero).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(spread > 1e-4, "gating the coarse branch changed nothing (max delta {spread})");
    }

    #[test]
    fn fewer_tiles_changes_the_result_but_stays_finite() {
        let (b, heads, dim) = (1, 1, 8);
        let plan = plan_of((4, 8, 8));
        let n = b * heads * plan.seq * dim;
        let mut s = 99u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let (q, k, v): (Vec<f32>, Vec<f32>, Vec<f32>) =
            ((0..n).map(|_| rnd()).collect(), (0..n).map(|_| rnd()).collect(), (0..n).map(|_| rnd()).collect());
        let scale = 1.0 / (dim as f32).sqrt();
        let nb = plan.num_tiles();
        let sparse = vsa_attention_host(&q, &k, &v, None, &plan, topk_for(0.8, nb), b, heads, dim, scale).unwrap();
        assert!(sparse.iter().all(|x| x.is_finite()));
        let dense = vsa_attention_host(&q, &k, &v, None, &plan, nb, b, heads, dim, scale).unwrap();
        assert!(dense.iter().all(|x| x.is_finite()));
        assert!(sparse.iter().zip(&dense).any(|(a, b)| (a - b).abs() > 1e-6), "sparsity changed nothing");
    }
}
