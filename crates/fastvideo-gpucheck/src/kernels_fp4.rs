//! `h3_fp4_attn`: the sparse FP4 VSA attention kernel
//! ([`fastvideo_cudarc::wan::attn_fp4`], docs/ports/fasth3-sparse-fp4-attn.md)
//! on synthetic data, sm_120 only.
//!
//! 1. **mma layout**: one block-scaled FP4 `mma.sync` on random fragments
//!    against the host product under the documented fragment and scale
//!    layouts (all-unit scales first, so a data-layout and a scale-layout
//!    error read apart);
//! 2. **quantizer**: q / k / v (f32 and bf16 inputs) bit for bit against
//!    the host twin of upstream's per-16 rule (k rows permuted);
//! 3. **fine stage**, VSA-H3 tile lists at sparsity 0.8, rel-L2 of
//!    - the kernel against a host emulation of the same FP4 algorithm (same
//!      codes, online softmax in the kernel's tile order): the layout check,
//!      tight;
//!    - the kernel against token-masked attention in f64 on the bf16 inputs:
//!      the FP4 noise floor, gated at upstream's 0.19 (single-level P is
//!      reported beside the default two-level);
//!    - a full block list (every tile) against dense f64 attention, same bound;
//!    - our bf16 VSA kernel against the same references, for scale;
//! 4. **timing** at the 480p and 768p 5 s plans (56 heads): the bf16 fine
//!    stage (tile gather + tensor-core kernel) against FP4 (quantize + kernel).

use cudarc::driver::CudaSlice;
use fastvideo_cudarc::h3::vsa::H3TilePlan;
use fastvideo_cudarc::wan::attn_fp4::{self as fp4, Fp4In};
use fastvideo_cudarc::wan::{device, ops};
use fastvideo_models::h3::config::H3Geometry;
use fastvideo_models::h3::packing::H3PackedLayout;
use rayon::prelude::*;
use serde_json::json;

use crate::report::{Report, StageResult};

/// Upstream's FP4 noise floor: SageAttention3 sparse FP4 against
/// token-masked fp32 (attn_qat_infer tests, docs/ports/fasth3-nvfp4.md).
const NOISE_FLOOR: f64 = 0.19;
/// Kernel vs the host emulation of the same FP4 algorithm: only exp2
/// approximation, f32 accumulation order and the rare E2M1 / E4M3 rounding
/// flip they cause separate the two.
const EMULATION_LIMIT: f64 = 2e-2;

fn dev() -> anyhow::Result<std::sync::Arc<device::DeviceContext>> {
    device::global_device().ok_or_else(|| anyhow::anyhow!("no live CUDA device"))
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unif(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.unif().max(1e-300), self.unif());
        ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
    }
}

fn bf16r(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += (f64::from(*x) - f64::from(*y)).powi(2);
        den += f64::from(*y).powi(2);
    }
    (num / den.max(1e-300)).sqrt()
}

pub fn run(report: &mut Report, seed: u64) -> StageResult<()> {
    if !fp4::supported() {
        report.note("h3_fp4_attn_skipped", json!({"reason": "needs sm_120"}));
        return Ok(());
    }
    mma_layout(report, seed)?;
    let (layout, heads) = (
        H3PackedLayout::new(300, (8, 32, 64), 32, [1, 2, 2]).map_err(anyhow::Error::msg)?,
        2,
    );
    fine_stage(report, seed, &layout, heads)?;
    timing(report, seed)?;
    Ok(())
}

// ---- 1. one mma ---------------------------------------------------------------------------------------------------

/// The documented fragment layouts (attn_fp4.cu `f4_mma`), host side:
/// `a[l][r]` nibble `i` is A[row][k] with row = g + 8 (r & 1), k = 8t + i + 32 (r >> 1);
/// `b[l][r]` nibble `i` is B[k][n] with n = g, k = 8t + i + 32 r; lane l's sfa
/// is row g + 8 (l & 1); sfb column g; d[l] = D[g][2t, 2t+1], D[g+8][2t, 2t+1].
fn mma_host(a: &[u32], b: &[u32], sfa: &[u32], sfb: &[u32]) -> Vec<f32> {
    let mut am = vec![0f64; 16 * 64];
    let mut bm = vec![0f64; 64 * 8];
    let mut sa = vec![0f64; 16 * 4];
    let mut sb = vec![0f64; 8 * 4];
    for l in 0..32 {
        let (g, t) = (l >> 2, l & 3);
        for r in 0..4 {
            for i in 0..8 {
                let row = g + 8 * (r & 1);
                let k = 8 * t + i + 32 * (r >> 1);
                am[row * 64 + k] =
                    f64::from(fp4::e2m1_value(((a[4 * l + r] >> (4 * i)) & 15) as u8));
            }
        }
        for r in 0..2 {
            for i in 0..8 {
                bm[(8 * t + i + 32 * r) * 8 + g] =
                    f64::from(fp4::e2m1_value(((b[2 * l + r] >> (4 * i)) & 15) as u8));
            }
        }
        if t < 2 {
            for j in 0..4 {
                sa[(g + 8 * (l & 1)) * 4 + j] =
                    f64::from(fp4::e4m3_value((sfa[l] >> (8 * j)) as u8));
            }
        }
        if t == 0 {
            for j in 0..4 {
                sb[g * 4 + j] = f64::from(fp4::e4m3_value((sfb[l] >> (8 * j)) as u8));
            }
        }
    }
    let mut d = vec![0f32; 128];
    for l in 0..32 {
        let (g, t) = (l >> 2, l & 3);
        for (e, (row, col)) in [
            (g, 2 * t),
            (g, 2 * t + 1),
            (g + 8, 2 * t),
            (g + 8, 2 * t + 1),
        ]
        .into_iter()
        .enumerate()
        {
            let mut acc = 0f64;
            for k in 0..64 {
                acc += am[row * 64 + k]
                    * sa[row * 4 + k / 16]
                    * bm[k * 8 + col]
                    * sb[col * 4 + k / 16];
            }
            d[4 * l + e] = acc as f32;
        }
    }
    d
}

fn mma_layout(report: &mut Report, seed: u64) -> StageResult<()> {
    let mut rng = Rng(seed ^ 0xf4f4);
    let word =
        |rng: &mut Rng| (0..8).fold(0u32, |w, i| w | (((rng.next() % 16) as u32) << (4 * i)));
    let a: Vec<u32> = (0..128).map(|_| word(&mut rng)).collect();
    let b: Vec<u32> = (0..64).map(|_| word(&mut rng)).collect();
    // Scales: byte j of lane l's sfa is row g + 8 (l & 1), group j; lanes l and
    // l ^ 2 carry the same row, so either may be the one read.
    let ones = vec![0x3838_3838u32; 32];
    let row_scale = |row: usize, j: usize| -> u32 { 0x30 + ((row * 5 + j * 3) % 16) as u32 };
    let col_scale = |col: usize, j: usize| -> u32 { 0x34 + ((col * 7 + j * 5) % 12) as u32 };
    let sfa: Vec<u32> = (0..32)
        .map(|l| {
            (0..4).fold(0u32, |w, j| {
                w | (row_scale((l >> 2) + 8 * (l & 1), j) << (8 * j))
            })
        })
        .collect();
    let sfb: Vec<u32> = (0..32)
        .map(|l| (0..4).fold(0u32, |w, j| w | (col_scale(l >> 2, j) << (8 * j))))
        .collect();
    let mut results = serde_json::Map::new();
    let mut ok = true;
    for (name, sa, sb) in [("unit_scales", &ones, &ones), ("block_scales", &sfa, &sfb)] {
        let got = fp4::mma_probe(&a, &b, sa, sb)?;
        let want = mma_host(&a, &b, sa, sb);
        let max_abs = got
            .iter()
            .zip(&want)
            .fold(0f64, |m, (x, y)| m.max(f64::from((x - y).abs())));
        let wrong: Vec<usize> = (0..128)
            .filter(|&i| (got[i] - want[i]).abs() > 1e-3 * want[i].abs().max(1.0))
            .collect();
        // Which outputs are wrong, by (row, col), for a layout diagnosis.
        let cells: Vec<String> = wrong
            .iter()
            .take(12)
            .map(|&i| {
                let (l, e) = (i / 4, i % 4);
                format!(
                    "D[{}][{}]={} want {}",
                    (l >> 2) + 8 * (e >> 1),
                    2 * (l & 3) + (e & 1),
                    got[i],
                    want[i]
                )
            })
            .collect();
        ok &= wrong.is_empty();
        results.insert(
            name.into(),
            json!({"max_abs": max_abs, "wrong": wrong.len(), "first": cells}),
        );
    }
    report.check(
        "mma_mxf4nvf4_layout",
        ok,
        serde_json::Value::Object(results),
        json!({"wrong": 0}),
    )
}

// ---- host twins -----------------------------------------------------------------------------------------------------

/// Host quantization of one tile-slot row of 128 values: (codes [64], scales [8]).
fn quant_row(x: &[f32]) -> ([u8; 64], [u8; 8]) {
    let (mut codes, mut sf) = ([0u8; 64], [0u8; 8]);
    for g in 0..8 {
        let (s, c) = fp4::quant16(&x[16 * g..16 * g + 16]);
        sf[g] = s;
        for i in 0..8 {
            codes[8 * g + i] = c[2 * i] | (c[2 * i + 1] << 4);
        }
    }
    (codes, sf)
}

/// Dequantized value of nibble `i` in a code row.
fn nib(codes: &[u8], i: usize) -> f32 {
    fp4::e2m1_value((codes[i / 2] >> (4 * (i % 2))) & 15)
}

/// A problem: q / k / v `[bh, seq, 128]` (bf16-representable f32), the plan
/// and per-(head, query tile) key-tile lists.
struct Problem {
    plan: H3TilePlan,
    bh: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

/// q / k shaped like post-RMSNorm, post-RoPE activations: unit-RMS rows, a
/// smooth function of the token's position shared by q and k (so nearby
/// tokens attend each other and the pooled top-k means something), and four
/// outlier channels; v unit normal.
fn problem(layout: &H3PackedLayout, bh: usize, seed: u64) -> anyhow::Result<Problem> {
    let plan = H3TilePlan::new(layout).map_err(|e| anyhow::anyhow!("{e}"))?;
    let seq = plan.seq;
    let mut rng = Rng(seed);
    let (gt, gh, gw) = layout.token_grid;
    let prefix = plan.prefix_rows;
    let pos = |r: usize| -> [f32; 3] {
        if r < prefix {
            [r as f32 / prefix as f32, 0.5, 0.5]
        } else {
            let i = r - prefix;
            [
                (i / (gh * gw)) as f32 / gt as f32,
                ((i / gw) % gh) as f32 / gh as f32,
                (i % gw) as f32 / gw as f32,
            ]
        }
    };
    let freqs: Vec<[f32; 3]> = (0..128)
        .map(|d| {
            [
                (d % 7) as f32 * 2.1 + 0.5,
                (d % 5) as f32 * 3.3 + 0.7,
                (d % 3) as f32 * 4.7 + 0.3,
            ]
        })
        .collect();
    let mut gen = |shared: f32| -> Vec<f32> {
        let mut x = vec![0f32; bh * seq * 128];
        for h in 0..bh {
            for r in 0..seq {
                let p = pos(r);
                for d in 0..128 {
                    let f = freqs[d];
                    let smooth =
                        (f[0] * p[0] * 6.0 + f[1] * p[1] * 6.0 + f[2] * p[2] * 6.0 + h as f32)
                            .sin();
                    let outlier = if d % 37 == 5 { 4.0 } else { 1.0 };
                    x[(h * seq + r) * 128 + d] =
                        bf16r((0.6 * rng.normal() + shared * smooth) * outlier);
                }
            }
        }
        x
    };
    let q = gen(1.6);
    let k = gen(1.6);
    let v = gen(0.0);
    Ok(Problem { plan, bh, q, k, v })
}

/// Per (head, query tile) key-tile lists: VSA-H3's rule from pooled scores
/// (every prefix tile, then the top `k_vid` video tiles), or every tile.
fn select(p: &Problem, k_vid: Option<usize>) -> (Vec<u32>, usize) {
    let (n, plan) = (p.plan.num_tiles(), &p.plan);
    let Some(k_vid) = k_vid else {
        let lists = (0..p.bh * n).flat_map(|_| 0..n as u32).collect();
        return (lists, n);
    };
    let topk = plan.prefix_tiles + k_vid;
    let pool = |x: &[f32], h: usize, t: usize| -> Vec<f64> {
        let rows = &plan.slot_src[t * 64..t * 64 + plan.block_sizes[t] as usize];
        let mut m = vec![0f64; 128];
        for &r in rows {
            for d in 0..128 {
                m[d] += f64::from(x[(h * plan.seq + r as usize) * 128 + d]);
            }
        }
        m.iter_mut().for_each(|v| *v /= rows.len() as f64);
        m
    };
    let mut lists = vec![0u32; p.bh * n * topk];
    for h in 0..p.bh {
        let qm: Vec<Vec<f64>> = (0..n).map(|t| pool(&p.q, h, t)).collect();
        let km: Vec<Vec<f64>> = (0..n).map(|t| pool(&p.k, h, t)).collect();
        for i in 0..n {
            let mut vids: Vec<(f64, usize)> = (plan.prefix_tiles..n)
                .map(|j| (qm[i].iter().zip(&km[j]).map(|(a, b)| a * b).sum::<f64>(), j))
                .collect();
            vids.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let row = &mut lists[(h * n + i) * topk..(h * n + i + 1) * topk];
            for (s, j) in row
                .iter_mut()
                .zip((0..plan.prefix_tiles).chain(vids.iter().take(k_vid).map(|v| v.1)))
            {
                *s = j as u32;
            }
        }
    }
    (lists, topk)
}

/// Token-masked attention in f64 for query tiles `[q_base, n)`: `[bh, padded,
/// 128]` in tile-slot order (zero elsewhere), as the kernel writes it.
fn reference(p: &Problem, lists: &[u32], topk: usize, q_base: usize) -> Vec<f32> {
    let (n, plan, seq) = (p.plan.num_tiles(), &p.plan, p.plan.seq);
    let scale = 1.0 / (128f64).sqrt();
    let mut out = vec![0f32; p.bh * n * 64 * 128];
    out.par_chunks_mut(64 * 128)
        .enumerate()
        .for_each(|(ht, o)| {
            let (h, i) = (ht / n, ht % n);
            if i < q_base {
                return;
            }
            let keys: Vec<usize> = lists[(h * n + i) * topk..(h * n + i + 1) * topk]
                .iter()
                .flat_map(|&j| {
                    plan.slot_src
                        [j as usize * 64..j as usize * 64 + plan.block_sizes[j as usize] as usize]
                        .iter()
                })
                .map(|&r| r as usize)
                .collect();
            for slot in 0..plan.block_sizes[i] as usize {
                let r = plan.slot_src[i * 64 + slot] as usize;
                let qr = &p.q[(h * seq + r) * 128..(h * seq + r + 1) * 128];
                let s: Vec<f64> = keys
                    .iter()
                    .map(|&kr| {
                        let kk = &p.k[(h * seq + kr) * 128..(h * seq + kr + 1) * 128];
                        qr.iter()
                            .zip(kk)
                            .map(|(a, b)| f64::from(*a) * f64::from(*b))
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let mx = s.iter().cloned().fold(f64::MIN, f64::max);
                let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                let z: f64 = w.iter().sum();
                for d in 0..128 {
                    let acc: f64 = keys
                        .iter()
                        .zip(&w)
                        .map(|(&kr, wi)| wi * f64::from(p.v[(h * seq + kr) * 128 + d]))
                        .sum();
                    o[slot * 128 + d] = (acc / z) as f32;
                }
            }
        });
    out
}

/// The kernel's algorithm on the host: the same codes, the online softmax in
/// list order with P quantized per 16 keys of each tile against the running
/// max (two-level unless `single`), f64 arithmetic.
fn emulate(p: &Problem, lists: &[u32], topk: usize, q_base: usize, single: bool) -> Vec<f32> {
    let (n, plan, seq) = (p.plan.num_tiles(), &p.plan, p.plan.seq);
    let sl2 = 1.0 / (128f64).sqrt() * std::f64::consts::LOG2_E;
    let pmul = if single { 1.0 } else { 2688.0 };
    let row_of = |x: &[f32], h: usize, slot: usize| -> Vec<f32> {
        let s = plan.slot_src[slot];
        if s < 0 {
            vec![0.0; 128]
        } else {
            x[(h * seq + s as usize) * 128..(h * seq + s as usize + 1) * 128].to_vec()
        }
    };
    let deq = |x: &[f32]| -> Vec<f64> {
        let (c, sf) = quant_row(x);
        (0..128)
            .map(|d| f64::from(nib(&c, d)) * f64::from(fp4::e4m3_value(sf[d / 16])))
            .collect()
    };
    let mut out = vec![0f32; p.bh * n * 64 * 128];
    out.par_chunks_mut(64 * 128)
        .enumerate()
        .for_each(|(ht, o)| {
            let (h, i) = (ht / n, ht % n);
            if i < q_base {
                return;
            }
            // Dequantized K rows and V (per head dim, 16 slots per scale) of the listed tiles.
            let list = &lists[(h * n + i) * topk..(h * n + i + 1) * topk];
            let tiles: Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>)> = list
                .iter()
                .map(|&j| {
                    let j = j as usize;
                    let kd: Vec<Vec<f64>> =
                        (0..64).map(|s| deq(&row_of(&p.k, h, j * 64 + s))).collect();
                    let raw: Vec<Vec<f32>> = (0..64).map(|s| row_of(&p.v, h, j * 64 + s)).collect();
                    let mut vd = vec![vec![0f64; 128]; 64];
                    for d in 0..128 {
                        for g in 0..4 {
                            let col: Vec<f32> = (0..16).map(|s| raw[16 * g + s][d]).collect();
                            let (sb, c) = fp4::quant16(&col);
                            for s in 0..16 {
                                vd[16 * g + s][d] = f64::from(fp4::e2m1_value(c[s]))
                                    * f64::from(fp4::e4m3_value(sb));
                            }
                        }
                    }
                    (kd, vd)
                })
                .collect();
            for slot in 0..64 {
                let qd = deq(&row_of(&p.q, h, i * 64 + slot));
                let (mut m, mut l) = (f64::NEG_INFINITY, 0f64);
                let mut acc = vec![0f64; 128];
                for (t, &j) in list.iter().enumerate() {
                    let len = plan.block_sizes[j as usize] as usize;
                    let (kd, vd) = &tiles[t];
                    let s: Vec<f64> = (0..64)
                        .map(|c| {
                            if c < len {
                                qd.iter().zip(&kd[c]).map(|(a, b)| a * b).sum()
                            } else {
                                f64::NEG_INFINITY
                            }
                        })
                        .collect();
                    let mn = s.iter().cloned().fold(m, f64::max);
                    let ms = if mn == f64::NEG_INFINITY {
                        0.0
                    } else {
                        mn * sl2
                    };
                    let alpha = if m == f64::NEG_INFINITY {
                        0.0
                    } else {
                        (m * sl2 - ms).exp2()
                    };
                    m = mn;
                    let pt: Vec<f64> = s
                        .iter()
                        .map(|x| {
                            if *x == f64::NEG_INFINITY {
                                0.0
                            } else {
                                (x * sl2 - ms).exp2()
                            }
                        })
                        .collect();
                    l = l * alpha + pt.iter().sum::<f64>();
                    acc.iter_mut().for_each(|a| *a *= alpha);
                    for g in 0..4 {
                        let gm = pt[16 * g..16 * g + 16].iter().cloned().fold(0f64, f64::max);
                        let sb = fp4::e4m3_encode((gm * pmul / 6.0) as f32);
                        let sf = f64::from(fp4::e4m3_value(sb));
                        let inv = if sf == 0.0 { 0.0 } else { pmul / sf };
                        for c in 16 * g..16 * g + 16 {
                            let pq =
                                f64::from(fp4::e2m1_value(fp4::e2m1_encode((pt[c] * inv) as f32)))
                                    * sf;
                            if pq != 0.0 {
                                acc.iter_mut().zip(&vd[c]).for_each(|(a, v)| *a += pq * v);
                            }
                        }
                    }
                }
                for d in 0..128 {
                    o[slot * 128 + d] = if l > 0.0 {
                        (acc[d] / (l * pmul)) as f32
                    } else {
                        0.0
                    };
                }
            }
        });
    out
}

/// Rows of live query slots in tiles `[q_base, n)`.
fn live(p: &Problem, x: &[f32], q_base: usize) -> Vec<f32> {
    let n = p.plan.num_tiles();
    let mut v = Vec::new();
    for h in 0..p.bh {
        for i in q_base..n {
            for s in 0..p.plan.block_sizes[i] as usize {
                let o = ((h * n + i) * 64 + s) * 128;
                v.extend_from_slice(&x[o..o + 128]);
            }
        }
    }
    v
}

// ---- 2. / 3. ---------------------------------------------------------------------------------------------------------

fn fine_stage(
    report: &mut Report,
    seed: u64,
    layout: &H3PackedLayout,
    heads: usize,
) -> StageResult<()> {
    let d = dev()?;
    let p = problem(layout, heads, seed)?;
    let (n, seq, bh) = (p.plan.num_tiles(), p.plan.seq, p.bh);
    let plan = ops::vsa_plan_upload(&p.plan.slot_src, &p.plan.block_sizes, 64)?;
    let qd = d.stream.memcpy_stod(&p.q)?;
    let kd = d.stream.memcpy_stod(&p.k)?;
    let vd = d.stream.memcpy_stod(&p.v)?;
    let to16 =
        |x: &[f32]| -> Vec<half::bf16> { x.iter().map(|&v| half::bf16::from_f32(v)).collect() };
    let (q16, k16, v16) = (
        d.stream.memcpy_stod(&to16(&p.q))?,
        d.stream.memcpy_stod(&to16(&p.k))?,
        d.stream.memcpy_stod(&to16(&p.v))?,
    );
    report.note(
        "h3_fp4_problem",
        json!({"seq": seq, "heads": bh, "tiles": n, "prefix_tiles": p.plan.prefix_tiles, "video_tiles": p.plan.video_tiles}),
    );

    // 2. The quantizer, bit for bit (E2M1 -0 read as 0).
    let t32 = fp4::quantize(
        &Fp4In::F32 {
            q: &qd,
            k: &kd,
            v: &vd,
        },
        &plan,
        bh,
        seq,
    )?;
    let t16 = fp4::quantize(
        &Fp4In::Bf16 {
            q: &q16,
            k: &k16,
            v: &v16,
        },
        &plan,
        bh,
        seq,
    )?;
    let get = |s: &CudaSlice<u8>| -> anyhow::Result<Vec<u8>> { Ok(d.stream.memcpy_dtov(s)?) };
    let (qk4, qksf, v4, vsf) = (
        get(&t32.qk4)?,
        get(&t32.qksf)?,
        get(&t32.v4)?,
        get(&t32.vsf)?,
    );
    let same_inputs = qk4 == get(&t16.qk4)?
        && qksf == get(&t16.qksf)?
        && v4 == get(&t16.v4)?
        && vsf == get(&t16.vsf)?;
    let zero_neg = |b: u8| -> u8 {
        let lo = if b & 15 == 8 { b & 0xf0 } else { b };
        if lo >> 4 == 8 {
            lo & 15
        } else {
            lo
        }
    };
    let padded = n * 64;
    let mut mismatches = [0usize; 3];
    let row_of = |x: &[f32], h: usize, slot: usize| -> Vec<f32> {
        let s = p.plan.slot_src[slot];
        if s < 0 {
            vec![0.0; 128]
        } else {
            x[(h * seq + s as usize) * 128..(h * seq + s as usize + 1) * 128].to_vec()
        }
    };
    for h in 0..bh {
        for slot in 0..padded {
            for (which, x) in [(0usize, &p.q), (1, &p.k)] {
                let (c, sf) = quant_row(&row_of(x, h, slot));
                let stored = if which == 1 {
                    (slot / 64) * 64 + fp4::kperm(slot % 64)
                } else {
                    slot
                };
                let row = (which * bh + h) * padded + stored;
                let dc = &qk4[row * 64..row * 64 + 64];
                let ds = &qksf[row * 8..row * 8 + 8];
                if dc.iter().zip(&c).any(|(a, b)| zero_neg(*a) != zero_neg(*b)) || ds != sf {
                    mismatches[which] += 1;
                }
            }
        }
        for t in 0..n {
            for dim in 0..128 {
                let col: Vec<f32> = (0..64).map(|s| row_of(&p.v, h, t * 64 + s)[dim]).collect();
                let (c, sf) = quant_row(&{
                    let mut x = col.clone();
                    x.resize(128, 0.0);
                    x
                });
                let row = (h * n + t) * 128 + dim;
                let dc = &v4[row * 32..row * 32 + 32];
                let ds = &vsf[row * 4..row * 4 + 4];
                if dc
                    .iter()
                    .zip(&c[..32])
                    .any(|(a, b)| zero_neg(*a) != zero_neg(*b))
                    || ds != &sf[..4]
                {
                    mismatches[2] += 1;
                }
            }
        }
    }
    let quant_ok = mismatches == [0, 0, 0] && same_inputs;
    let quant_values = json!({"q_rows_differ": mismatches[0], "k_rows_differ": mismatches[1],
                              "v_columns_differ": mismatches[2], "f32_and_bf16_inputs_identical": same_inputs});

    // 3. The fine stage, sparse (VSA-H3 lists at 0.8) and with every tile listed.
    let scale = 1.0 / (128f32).sqrt();
    let k_vid = p.plan.k_vid(0.8);
    let mut values = serde_json::Map::new();
    let mut ok = true;
    for (case, kv) in [("sparse_0.8", Some(k_vid)), ("full_list", None)] {
        let (lists, topk) = select(&p, kv);
        let q_base = if kv.is_some() { p.plan.prefix_tiles } else { 0 };
        let sel = d.stream.memcpy_stod(&lists)?;
        let reference = live(&p, &reference(&p, &lists, topk, q_base), q_base);
        let mut row = serde_json::Map::new();
        for single in [false, true] {
            let out = fp4::fine(&t32, &sel, &plan, topk, scale, q_base, n - q_base, single)?;
            let got = live(&p, &d.stream.memcpy_dtov(&out)?, q_base);
            let tag = if single { "single_p" } else { "two_level_p" };
            let vs_ref = rel_l2(&got, &reference);
            row.insert(format!("{tag}_vs_f64"), json!(vs_ref));
            ok &= vs_ref < NOISE_FLOOR;
            if !single || kv.is_some() {
                let em = live(&p, &emulate(&p, &lists, topk, q_base, single), q_base);
                let vs_em = rel_l2(&got, &em);
                row.insert(format!("{tag}_vs_emulation"), json!(vs_em));
                row.insert(
                    format!("{tag}_emulation_vs_f64"),
                    json!(rel_l2(&em, &reference)),
                );
                ok &= vs_em < EMULATION_LIMIT;
            }
        }
        // Our bf16 VSA fine stage on the same lists, for scale.
        let bf = ops::vsa_mma_attn_range_device(
            &qd,
            &kd,
            &vd,
            &sel,
            &plan,
            bh,
            seq,
            128,
            topk,
            scale,
            q_base,
            n - q_base,
        )?;
        let bf = live(&p, &d.stream.memcpy_dtov(&bf)?, q_base);
        row.insert("bf16_vsa_vs_f64".into(), json!(rel_l2(&bf, &reference)));
        row.insert("topk".into(), json!(topk));
        values.insert(case.into(), serde_json::Value::Object(row));
    }
    report.check(
        "fp4_quantizer_bitwise",
        quant_ok,
        quant_values,
        json!({"differ": 0}),
    )?;
    report.check(
        "fp4_fine_stage",
        ok,
        serde_json::Value::Object(values),
        json!({"vs_f64": NOISE_FLOOR, "vs_emulation": EMULATION_LIMIT}),
    )
}

// ---- 4. timing ---------------------------------------------------------------------------------------------------------

fn timing(report: &mut Report, seed: u64) -> StageResult<()> {
    let d = dev()?;
    let heads =
        fastvideo_models::h3::config::H3TransformerConfig::fasth3_8step().num_attention_heads;
    for (name, h, w) in [("480p", 480usize, 832usize), ("768p", 768, 1344)] {
        let geo = H3Geometry::new(h, w, 124).map_err(anyhow::Error::msg)?;
        let layout = H3PackedLayout::from_geometry(&geo, 256).map_err(anyhow::Error::msg)?;
        let plan = H3TilePlan::new(&layout).map_err(|e| anyhow::anyhow!("{e}"))?;
        let (n, seq) = (plan.num_tiles(), plan.seq);
        let topk = plan.prefix_tiles + plan.k_vid(0.8);
        let pd = ops::vsa_plan_upload(&plan.slot_src, &plan.block_sizes, 64)?;
        let mut rng = Rng(seed ^ h as u64);
        // One head's values repeated over the heads: the timing does not depend on them.
        let rnd = |rng: &mut Rng| -> Vec<f32> {
            let one: Vec<f32> = (0..seq * 128).map(|_| bf16r(rng.normal())).collect();
            one.repeat(heads)
        };
        let (q, k, v) = (
            d.stream.memcpy_stod(&rnd(&mut rng))?,
            d.stream.memcpy_stod(&rnd(&mut rng))?,
            d.stream.memcpy_stod(&rnd(&mut rng))?,
        );
        // Random lists of the real length: prefix tiles, then distinct video tiles.
        let mut lists = Vec::with_capacity(heads * n * topk);
        for _ in 0..heads * n {
            lists.extend(0..plan.prefix_tiles as u32);
            let mut vids: Vec<u32> = (plan.prefix_tiles as u32..n as u32).collect();
            for i in 0..topk - plan.prefix_tiles {
                let j = i + (rng.next() as usize % (vids.len() - i));
                vids.swap(i, j);
            }
            lists.extend_from_slice(&vids[..topk - plan.prefix_tiles]);
        }
        let sel = d.stream.memcpy_stod(&lists)?;
        let scale = 1.0 / (128f32).sqrt();
        let (qb, qt) = (plan.prefix_tiles, n - plan.prefix_tiles);
        let time = |f: &mut dyn FnMut() -> anyhow::Result<()>| -> anyhow::Result<f64> {
            f()?;
            device::synchronize()?;
            let t = std::time::Instant::now();
            for _ in 0..5 {
                f()?;
            }
            device::synchronize()?;
            Ok(t.elapsed().as_secs_f64() * 1e3 / 5.0)
        };
        let bf16_ms = time(&mut || {
            ops::vsa_mma_attn_range_device(
                &q, &k, &v, &sel, &pd, heads, seq, 128, topk, scale, qb, qt,
            )?;
            Ok(())
        })?;
        let mut tiles = None;
        let quant_ms = time(&mut || {
            tiles = Some(fp4::quantize(
                &Fp4In::F32 {
                    q: &q,
                    k: &k,
                    v: &v,
                },
                &pd,
                heads,
                seq,
            )?);
            Ok(())
        })?;
        let tiles = tiles.ok_or_else(|| anyhow::anyhow!("no tiles"))?;
        let fine_ms = time(&mut || {
            fp4::fine(&tiles, &sel, &pd, topk, scale, qb, qt, false)?;
            Ok(())
        })?;
        let flop = 4.0 * (heads * qt * 64) as f64 * (topk * 64) as f64 * 128.0;
        report.note(
            format!("fp4_fine_timing_{name}"),
            json!({"seq": seq, "heads": heads, "tiles": n, "topk": topk,
                   "bf16_tile_and_mma_ms": bf16_ms, "fp4_quantize_ms": quant_ms, "fp4_fine_ms": fine_ms,
                   "fp4_total_ms": quant_ms + fine_ms, "speedup": bf16_ms / (quant_ms + fine_ms),
                   "fp4_fine_tflops": flop / (fine_ms * 1e-3) / 1e12}),
        );
    }
    Ok(())
}
