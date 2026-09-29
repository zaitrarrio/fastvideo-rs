//! `vsa_dc`: the datacenter VSA kernels of attn_dc.cu (WP-D) on 9.0 / 10.0.
//!
//! Parity, per workload (H3 480p / 768p / 1080p token grids, tile (4, 4, 4),
//! d = 128) and sparsity, on spatially smooth q/k (neighbouring query tiles
//! select overlapping key tiles, as real video attention does) and on white
//! noise (the worst case for the sm_100 union of two query tiles):
//!
//! - the fine output (`attn_dc::vsa_fine`, sparse layout) against the
//!   mma.sync kernel it replaces (`vsa_mma_attn_tma2`) at the same selection:
//!   rel_l2 <= 1e-3 (plan §5.2, "as the dense row");
//! - both against an f64 block-masked dense reference on sampled query
//!   tiles: <= 3.5e-3, and the dc kernel no worse than tma2 + 5%;
//! - a partial query range with an odd base (H3 skips its prefix tiles);
//! - the fused prep (`attn_dc::vsa_prep`) bit for bit against
//!   `vsa_tile_qkv` + `vsa_tile_mean` (f32 round16, f32 plain, bf16), so the
//!   f32 scores and the selection are unchanged;
//! - the fused H3 combine (`VsaEpilogue::CombineRound16`) bit for bit
//!   against `Sparse` + `vsa_combine(round16)` / `vsa_combine_g16`.
//!
//! Timings (`vsa_dc_bench_*`, 56 heads as FastH3): prep, fine and combine,
//! old kernels against new, and the mean union size per query-tile pair.

use super::*;
use fastvideo_cudarc::wan::attn_dc::{self, VsaEpilogue, VsaGate, VsaPrepIn};
use half::bf16;

const D: usize = 128;
const TILE: usize = 64;

/// (name, token grid (t, h, w)) of FastH3's video latents.
const GRIDS: &[(&str, (usize, usize, usize))] = &[
    ("h3_480p", (37, 15, 26)),
    ("h3_768p", (37, 24, 42)),
    ("h3_1080p", (37, 34, 60)),
];

fn median3(f: &mut dyn FnMut() -> anyhow::Result<()>) -> anyhow::Result<f64> {
    f()?;
    device::synchronize()?;
    let mut t = Vec::new();
    for _ in 0..3 {
        let s = std::time::Instant::now();
        f()?;
        device::synchronize()?;
        t.push(s.elapsed().as_secs_f64());
    }
    t.sort_by(|a, b| a.total_cmp(b));
    Ok(t[1])
}

fn mismatched_bits(a: &[f32], b: &[f32]) -> usize {
    if a.len() != b.len() {
        return a.len().max(b.len());
    }
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

/// `heads` heads of `[seq, 128]`, raster (t, h, w) tokens. `smooth`: a sum
/// of low-frequency plane waves per channel plus 30% noise (neighbouring
/// tiles look alike); else white noise.
fn field(c: &mut Ctx<'_>, grid: (usize, usize, usize), heads: usize, smooth: bool, amp: f32) -> Vec<f32> {
    let (gt, gh, gw) = grid;
    let seq = gt * gh * gw;
    let mut out = Vec::with_capacity(heads * seq * D);
    for _ in 0..heads {
        let noise = c.rand(seq * D, if smooth { 0.3 * amp } else { amp });
        if !smooth {
            out.extend_from_slice(&noise);
            continue;
        }
        let p = c.rand(4 * D, 1.0);
        let freq = |x: f32| 0.05 + 0.25 * x.abs().min(2.0);
        for tok in 0..seq {
            let (ti, rem) = (tok / (gh * gw), tok % (gh * gw));
            let (hi, wi) = ((rem / gw) as f32, (rem % gw) as f32);
            let ti = ti as f32;
            for d in 0..D {
                let ph = freq(p[4 * d]) * ti + freq(p[4 * d + 1]) * hi + freq(p[4 * d + 2]) * wi + 3.0 * p[4 * d + 3];
                out.push(amp * ph.sin() + noise[tok * D + d]);
            }
        }
    }
    out
}

fn bf16_rows(s: &CudaSlice<bf16>, off: usize, n: usize) -> anyhow::Result<Vec<f32>> {
    let v = dev()?.stream.memcpy_dtov(&s.slice(off..off + n))?;
    Ok(v.iter().map(|x| x.to_f32()).collect())
}

/// f64 block-masked attention of query tile `qt` of one head: tiled bf16
/// q/k/v of that head as f32, `sel` its selected key tiles.
fn ref_tile(q: &[f32], k: &[f32], v: &[f32], qt: usize, sel: &[u32], sizes: &[u32], scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; TILE * D];
    for r in 0..TILE {
        let qr = &q[(qt * TILE + r) * D..(qt * TILE + r + 1) * D];
        let mut scores = Vec::new();
        let mut rows = Vec::new();
        for &kt in sel {
            for j in 0..sizes[kt as usize] as usize {
                let row = kt as usize * TILE + j;
                let kr = &k[row * D..(row + 1) * D];
                let s: f64 = qr.iter().zip(kr).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
                scores.push(s * f64::from(scale));
                rows.push(row);
            }
        }
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let z: f64 = w.iter().sum();
        for d in 0..D {
            let acc: f64 = w.iter().zip(&rows).map(|(wi, &row)| wi * f64::from(v[row * D + d])).sum();
            out[r * D + d] = (acc / z) as f32;
        }
    }
    out
}

/// Mean |sel(2p) ∪ sel(2p + 1)| / topk over the pairs the sm_100 kernel
/// forms in `[q_base, nb)` (1.0 = identical selections, 2.0 = disjoint).
fn union_ratio(sel: &[u32], bh: usize, nb: usize, topk: usize, q_base: usize) -> f64 {
    let (mut sum, mut pairs) = (0.0, 0usize);
    for h in 0..bh {
        let mut t = q_base;
        while t + 1 < nb {
            let a = &sel[(h * nb + t) * topk..(h * nb + t + 1) * topk];
            let b = &sel[(h * nb + t + 1) * topk..(h * nb + t + 2) * topk];
            let mut u: Vec<u32> = a.iter().chain(b).copied().collect();
            u.sort_unstable();
            u.dedup();
            sum += u.len() as f64 / topk as f64;
            pairs += 1;
            t += 2;
        }
    }
    if pairs == 0 {
        1.0
    } else {
        sum / pairs as f64
    }
}

struct Case {
    plan: vsa::TilePlan,
    dev_plan: ops::VsaPlanDev,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    bh: usize,
}

fn case(c: &mut Ctx<'_>, grid: (usize, usize, usize), bh: usize, smooth: bool, repeat: bool) -> anyhow::Result<Case> {
    let plan = vsa::TilePlan::new(grid)?;
    let dev_plan = ops::vsa_plan_upload(&plan.slot_src, &plan.block_sizes, vsa::TILE_ELEMS)?;
    let heads = if repeat { 1 } else { bh };
    let mut mk = |amp: f32| -> anyhow::Result<CudaSlice<f32>> {
        let one = field(c, grid, heads, smooth, amp);
        if repeat && bh > 1 {
            let mut all = Vec::with_capacity(one.len() * bh);
            for _ in 0..bh {
                all.extend_from_slice(&one);
            }
            up(&all)
        } else {
            up(&one)
        }
    };
    let (q, k, v) = (mk(1.0)?, mk(1.0)?, mk(1.0)?);
    Ok(Case { plan, dev_plan, q, k, v, bh })
}

/// Pooled f32 scores (H3's round16 tile means) and the top-k selection.
fn select(cs: &Case, topk: usize) -> anyhow::Result<CudaSlice<u32>> {
    let (bh, seq, nb) = (cs.bh, cs.plan.seq, cs.plan.num_tiles());
    let qc = ops::vsa_tile_mean_round_device(&cs.q, &cs.dev_plan, bh, seq, D, true)?;
    let kc = ops::vsa_tile_mean_round_device(&cs.k, &cs.dev_plan, bh, seq, D, true)?;
    let mut scores = ops::alloc(bh * nb * nb)?;
    let scale = 1.0 / (D as f32).sqrt();
    device::matmul_linear_wt_strided_batched_f32(&qc, &kc, &mut scores, bh, nb, D, nb, scale)?;
    Ok(ops::vsa_topk_device(&scores, bh * nb, nb, topk)?)
}

fn tiled(cs: &Case) -> anyhow::Result<(CudaSlice<bf16>, CudaSlice<bf16>, CudaSlice<bf16>)> {
    let (bh, seq) = (cs.bh, cs.plan.seq);
    Ok((
        ops::vsa_tile_qkv_device(&cs.q, &cs.dev_plan, bh, seq, D)?,
        ops::vsa_tile_qkv_device(&cs.k, &cs.dev_plan, bh, seq, D)?,
        ops::vsa_tile_qkv_device(&cs.v, &cs.dev_plan, bh, seq, D)?,
    ))
}

/// The mma.sync ring kernel on the same tiles (the incumbent).
#[allow(clippy::too_many_arguments)]
fn fine_tma2(
    t: &(CudaSlice<bf16>, CudaSlice<bf16>, CudaSlice<bf16>),
    sel: &CudaSlice<u32>,
    cs: &Case,
    topk: usize,
    q_base: usize,
    q_tiles: usize,
) -> anyhow::Result<CudaSlice<f32>> {
    std::env::set_var("FASTVIDEO_VSA_KERNEL", "tma2");
    let r = ops::vsa_mma_attn_tiled_device(
        &t.0, &t.1, &t.2, sel, &cs.dev_plan, cs.bh, D, topk, 1.0 / (D as f32).sqrt(), q_base, q_tiles,
    );
    std::env::remove_var("FASTVIDEO_VSA_KERNEL");
    Ok(r?)
}

fn fine_dc(
    t: &(CudaSlice<bf16>, CudaSlice<bf16>, CudaSlice<bf16>),
    sel: &CudaSlice<u32>,
    cs: &Case,
    topk: usize,
    q_base: usize,
    q_tiles: usize,
) -> anyhow::Result<CudaSlice<f32>> {
    let nb = cs.plan.num_tiles();
    let mut out = ops::fill_device(cs.bh * nb * TILE * D, 0.0)?;
    let ran = attn_dc::vsa_fine(
        &t.0,
        &t.1,
        &t.2,
        sel,
        &cs.dev_plan,
        cs.bh,
        topk,
        1.0 / (D as f32).sqrt(),
        q_base,
        q_tiles,
        VsaEpilogue::Sparse(&mut out),
    )?;
    anyhow::ensure!(ran, "attn_dc::vsa_fine declined the shape");
    Ok(out)
}

pub(super) fn run(c: &mut Ctx<'_>) -> StageResult<()> {
    let dev = dev()?;
    let Some(k) = attn_dc::vsa() else {
        c.report.note(
            "vsa_dc_skipped",
            json!({"sm": dev.sm_major * 10 + dev.sm_minor, "needs": "sm 9.0 or 10.0 with the attn_dc module"}),
        );
        return Ok(());
    };
    c.report.note("vsa_dc_kernel", json!({"sm": k.sm, "origin": k.origin}));
    let scale = 1.0 / (D as f32).sqrt();
    let only = std::env::var("FV_VSA_WORKLOADS").unwrap_or_default();
    let bench = std::env::var("FV_VSA_DC_BENCH").map_or(true, |v| v != "0");

    // ---- parity ----
    for &(name, grid) in GRIDS {
        if !only.is_empty() && !only.split(',').any(|s| s.trim() == name) {
            continue;
        }
        for smooth in [true, false] {
            let data = if smooth { "smooth" } else { "noise" };
            let bh = if name == "h3_1080p" { 2 } else { 3 };
            let cs = case(c, grid, bh, smooth, false)?;
            let (seq, nb) = (cs.plan.seq, cs.plan.num_tiles());
            let t = tiled(&cs)?;
            let sizes = cs.plan.block_sizes.clone();
            // Sparsity 0.5 (long lists) on the smallest grid only: the f64
            // reference is the harness's slowest part.
            let sps: &[f64] = if name == "h3_480p" { &[0.5, 0.8, 0.9] } else { &[0.8, 0.9] };
            for &sp in sps {
                let topk = vsa::topk_for(sp, nb);
                let sel = select(&cs, topk)?;
                let sel_h = dev.stream.memcpy_dtov(&sel)?;
                for q_base in [0usize, 5] {
                    let q_tiles = nb - q_base;
                    let tag = format!("{name}_{data}_s{sp}_k{topk}_q{q_base}");
                    let old = down(&fine_tma2(&t, &sel, &cs, topk, q_base, q_tiles)?)?;
                    let new = down(&fine_dc(&t, &sel, &cs, topk, q_base, q_tiles)?)?;
                    let lo = q_base * TILE * D;
                    let rows = |x: &[f32]| -> Vec<f32> {
                        (0..bh).flat_map(|h| x[h * nb * TILE * D + lo..(h + 1) * nb * TILE * D].to_vec()).collect()
                    };
                    let dv = diff(&rows(&new), &rows(&old));
                    // Sampled query tiles against the f64 block-masked reference.
                    let (mut got_dc, mut got_old, mut want) = (Vec::new(), Vec::new(), Vec::new());
                    for h in 0..bh.min(2) {
                        let hoff = h * nb * TILE * D;
                        let (qh, kh, vh) = (
                            bf16_rows(&t.0, hoff, nb * TILE * D)?,
                            bf16_rows(&t.1, hoff, nb * TILE * D)?,
                            bf16_rows(&t.2, hoff, nb * TILE * D)?,
                        );
                        for s in 0..4 {
                            let qt = q_base + (s * 7919 + h * 131) % q_tiles;
                            let sl = &sel_h[(h * nb + qt) * topk..(h * nb + qt + 1) * topk];
                            want.extend(ref_tile(&qh, &kh, &vh, qt, sl, &sizes, scale));
                            let o = hoff + qt * TILE * D;
                            got_dc.extend_from_slice(&new[o..o + TILE * D]);
                            got_old.extend_from_slice(&old[o..o + TILE * D]);
                        }
                    }
                    let (dr, or) = (diff(&got_dc, &want), diff(&got_old, &want));
                    c.report.check(
                        format!("vsa_dc_{tag}_vs_tma2"),
                        dv.within(1e-3),
                        dv.to_json(),
                        json!({"rel_l2": 1e-3}),
                    )?;
                    c.report.check(
                        format!("vsa_dc_{tag}_vs_f64_masked"),
                        dr.within(3.5e-3) && dr.rel_l2 <= 1.05 * or.rel_l2.max(1e-4),
                        json!({"dc": dr.to_json(), "tma2_rel_l2": or.rel_l2}),
                        json!({"rel_l2": 3.5e-3, "or_no_worse_than_tma2": 1.05}),
                    )?;
                    if q_base == 0 {
                        c.report.note(
                            format!("vsa_dc_{tag}_union"),
                            json!({"union_over_topk": union_ratio(&sel_h, bh, nb, topk, 0), "topk": topk, "tiles": nb}),
                        );
                    }
                }
                // Fused H3 combine vs Sparse + vsa_combine(round16) / _g16.
                if sp == 0.8 {
                    let coarse = up(&c.rand(bh * nb * D, 1.0))?;
                    let gate = up(&c.rand(bh * seq * D, 1.0))?;
                    let gate16 = CudaTensor::from_device_slice(gate.clone(), vec![1, bh, seq, D])?.quantize_bf16()?;
                    let gate16 = gate16
                        .device_slice_bf16()
                        .ok_or_else(|| anyhow::anyhow!("bf16 gate"))?
                        .clone();
                    let sparse = fine_dc(&t, &sel, &cs, topk, 0, nb)?;
                    let mut want32 = ops::fill_device(bh * seq * D, 0.0)?;
                    ops::vsa_combine_round_device(&sparse, &coarse, Some(&gate), &cs.dev_plan, &mut want32, bh, nb, 0, seq, D, true)?;
                    let mut want16 = ops::fill_device(bh * seq * D, 0.0)?;
                    ops::vsa_combine_gate16_device(&sparse, &coarse, Some(&gate16), &cs.dev_plan, &mut want16, bh, nb, 0, seq, D)?;
                    let mut want0 = ops::fill_device(bh * seq * D, 0.0)?;
                    ops::vsa_combine_round_device(&sparse, &coarse, None, &cs.dev_plan, &mut want0, bh, nb, 0, seq, D, true)?;
                    let mut bad = Vec::new();
                    for (label, gate, want) in [
                        ("f32_gate", VsaGate::F32(&gate), &want32),
                        ("bf16_gate", VsaGate::Bf16(&gate16), &want16),
                        ("no_gate", VsaGate::None, &want0),
                    ] {
                        let mut got = ops::fill_device(bh * seq * D, 0.0)?;
                        let ran = attn_dc::vsa_fine(
                            &t.0, &t.1, &t.2, &sel, &cs.dev_plan, bh, topk, scale, 0, nb,
                            VsaEpilogue::CombineRound16 { out: &mut got, coarse: &coarse, gate, seq },
                        )?;
                        if !ran {
                            return Err(anyhow::anyhow!("vsa_fine declined").into());
                        }
                        bad.push((label, mismatched_bits(&down(&got)?, &down(want)?)));
                    }
                    c.report.check(
                        format!("vsa_dc_{name}_{data}_fused_combine_bitexact"),
                        bad.iter().all(|(_, n)| *n == 0),
                        json!(bad.iter().map(|(l, n)| (l.to_string(), *n)).collect::<std::collections::BTreeMap<_, _>>()),
                        json!({"mismatched": 0}),
                    )?;
                }
            }
            // Fused prep vs vsa_tile_qkv + vsa_tile_mean, bit for bit.
            if smooth {
                let mut bad = std::collections::BTreeMap::new();
                for round16 in [true, false] {
                    let p = attn_dc::vsa_prep(
                        VsaPrepIn::F32 { q: &cs.q, k: &cs.k, v: &cs.v, round16 },
                        &cs.dev_plan,
                        bh,
                        seq,
                        true,
                    )?
                    .ok_or_else(|| anyhow::anyhow!("vsa_prep declined"))?;
                    let mut n = 0;
                    for (x, xt, xc) in [(&cs.q, &p.qt, &p.qc), (&cs.k, &p.kt, &p.kc), (&cs.v, &p.vt, p.vc.as_ref().unwrap())] {
                        let wt = ops::vsa_tile_qkv_device(x, &cs.dev_plan, bh, seq, D)?;
                        let wc = ops::vsa_tile_mean_round_device(x, &cs.dev_plan, bh, seq, D, round16)?;
                        let (a, b) = (dev.stream.memcpy_dtov(xt)?, dev.stream.memcpy_dtov(&wt)?);
                        n += a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                        n += mismatched_bits(&down(xc)?, &down(&wc)?);
                    }
                    bad.insert(format!("f32_round16_{round16}"), n);
                }
                let to16 = |x: &CudaSlice<f32>| -> anyhow::Result<CudaSlice<bf16>> {
                    let t = CudaTensor::from_device_slice(x.clone(), vec![1, bh, seq, D])?.quantize_bf16()?;
                    Ok(t.device_slice_bf16().ok_or_else(|| anyhow::anyhow!("bf16"))?.clone())
                };
                let (q16, k16, v16) = (to16(&cs.q)?, to16(&cs.k)?, to16(&cs.v)?);
                let p = attn_dc::vsa_prep(VsaPrepIn::Bf16 { q: &q16, k: &k16, v: &v16 }, &cs.dev_plan, bh, seq, false)?
                    .ok_or_else(|| anyhow::anyhow!("vsa_prep declined"))?;
                let mut n = 0;
                for (x, xt, xc) in [(&q16, &p.qt, Some(&p.qc)), (&k16, &p.kt, Some(&p.kc)), (&v16, &p.vt, None)] {
                    let wt = ops::vsa_tile_qkv_bf16_device(x, &cs.dev_plan, bh, seq, D)?;
                    let (a, b) = (dev.stream.memcpy_dtov(xt)?, dev.stream.memcpy_dtov(&wt)?);
                    n += a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                    if let Some(xc) = xc {
                        let wc = ops::vsa_tile_mean_bf16_device(x, &cs.dev_plan, bh, seq, D)?;
                        n += mismatched_bits(&down(xc)?, &down(&wc)?);
                    }
                }
                bad.insert("bf16".into(), n);
                c.report.check(
                    format!("vsa_dc_{name}_prep_bitexact"),
                    bad.values().all(|n| *n == 0),
                    json!(bad),
                    json!({"mismatched": 0}),
                )?;
            }
        }
    }

    // ---- timings at FastH3's 56 heads ----
    if !bench {
        return Ok(());
    }
    for &(name, grid) in GRIDS {
        if !only.is_empty() && !only.split(',').any(|s| s.trim() == name) {
            continue;
        }
        let bh = 56;
        let cs = case(c, grid, bh, true, true)?;
        let (seq, nb) = (cs.plan.seq, cs.plan.num_tiles());
        let mut row = serde_json::Map::new();
        row.insert("tiles".into(), json!(nb));
        row.insert("tokens".into(), json!(seq));
        let old_prep = median3(&mut || {
            for x in [&cs.q, &cs.k, &cs.v] {
                ops::vsa_tile_mean_round_device(x, &cs.dev_plan, bh, seq, D, true)?;
                ops::vsa_tile_qkv_device(x, &cs.dev_plan, bh, seq, D)?;
            }
            Ok(())
        })?;
        let new_prep = median3(&mut || {
            attn_dc::vsa_prep(
                VsaPrepIn::F32 { q: &cs.q, k: &cs.k, v: &cs.v, round16: true },
                &cs.dev_plan,
                bh,
                seq,
                true,
            )?;
            Ok(())
        })?;
        row.insert("prep_old_ms".into(), json!(old_prep * 1e3));
        row.insert("prep_fused_ms".into(), json!(new_prep * 1e3));
        let t = tiled(&cs)?;
        let coarse = ops::fill_device(bh * nb * D, 0.01)?;
        let gate = ops::fill_device(bh * seq * D, 0.5)?;
        for sp in [0.8, 0.9] {
            let topk = vsa::topk_for(sp, nb);
            let sel = select(&cs, topk)?;
            let union = union_ratio(&dev.stream.memcpy_dtov(&sel)?, 1, nb, topk, 0);
            let tma2 = median3(&mut || {
                fine_tma2(&t, &sel, &cs, topk, 0, nb)?;
                Ok(())
            })?;
            let dc = median3(&mut || {
                fine_dc(&t, &sel, &cs, topk, 0, nb)?;
                Ok(())
            })?;
            let sparse = fine_tma2(&t, &sel, &cs, topk, 0, nb)?;
            let mut out = ops::fill_device(bh * seq * D, 0.0)?;
            let combine = median3(&mut || {
                ops::vsa_combine_round_device(&sparse, &coarse, Some(&gate), &cs.dev_plan, &mut out, bh, nb, 0, seq, D, true)?;
                Ok(())
            })?;
            let fused = median3(&mut || {
                attn_dc::vsa_fine(
                    &t.0, &t.1, &t.2, &sel, &cs.dev_plan, bh, topk, scale, 0, nb,
                    VsaEpilogue::CombineRound16 { out: &mut out, coarse: &coarse, gate: VsaGate::F32(&gate), seq },
                )?;
                Ok(())
            })?;
            let flops = 4.0 * (bh * nb * TILE) as f64 * (topk * TILE) as f64 * D as f64;
            row.insert(
                format!("sparsity{sp}"),
                json!({
                    "topk": topk,
                    "union_over_topk": union,
                    "fine_tma2_ms": tma2 * 1e3,
                    "fine_dc_ms": dc * 1e3,
                    "fine_speedup": tma2 / dc,
                    "tma2_tflops": flops / tma2 / 1e12,
                    "dc_tflops": flops / dc / 1e12,
                    "combine_ms": combine * 1e3,
                    "dc_fused_combine_ms": fused * 1e3,
                    "old_prep_fine_combine_ms": (old_prep + tma2 + combine) * 1e3,
                    "new_prep_fine_combine_ms": (new_prep + fused) * 1e3,
                    "stage_speedup": (old_prep + tma2 + combine) / (new_prep + fused),
                }),
            );
        }
        c.report.note(format!("vsa_dc_bench_{name}"), serde_json::Value::Object(row));
    }
    Ok(())
}
