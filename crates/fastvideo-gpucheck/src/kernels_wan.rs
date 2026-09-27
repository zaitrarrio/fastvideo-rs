//! Wan DiT kernel checks.
//!
//! * **wan_causal_attn** — the block-causal flash kernel
//!   (`flash_mma_fwd2_causal_d*`, [`fastvideo_cudarc::wan::attn::device_mma_sdpa_causal`])
//!   against an f64 host SDPA under the same additive mask and against the
//!   device `sdpa_composed` path it replaces (masked `[S, S]` scores in f32),
//!   on synthetic bf16 inputs: odd sequence lengths, frame sizes that do not
//!   divide the 64-key tile, local windows and sinks, head dims 64 and 128.
//!   With one frame spanning the whole sequence it must equal the dense V2
//!   kernel bit for bit. Timed against dense flash and the composed path at
//!   the SF-Wan 1.3B 480x832 shape.
//! * **wan_fusion** — the Wan block's bf16 residual + norm kernels
//!   ([`fastvideo_cudarc::wan::fuse`]): fused (`FASTVIDEO_WAN_FUSE=1`)
//!   against unfused bit for bit, both against the host rounding-point twins,
//!   and the q/k norm + RoPE; fused / unfused timings at the FastWan 1.3B
//!   81-frame shape.

use fastvideo_cudarc::wan::attn::{self, BlockCausal, FlashKernel};
use fastvideo_cudarc::wan::fuse;
use fastvideo_cudarc::wan::fused::Rope;
use fastvideo_cudarc::wan::nn;
use fastvideo_cudarc::wan::tensor::with_bf16_act;
use serde_json::json;

use crate::kernels_fp8::{bf16v, check_ulps, host, rand, t16, t32, time_ms};
use crate::report::{Report, StageResult};

/// f64 SDPA of bf16-valued `[bh, s, d]` inputs under `mask`.
fn ref_masked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bh: usize,
    s: usize,
    d: usize,
    mask: &BlockCausal,
) -> Vec<f32> {
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0.0f32; bh * s * d];
    let mut p = vec![0.0f64; s];
    for b in 0..bh {
        for i in 0..s {
            let qi = &q[(b * s + i) * d..][..d];
            let mut mx = f64::NEG_INFINITY;
            for (j, pj) in p.iter_mut().enumerate() {
                *pj = if mask.allows(i, j) {
                    let kj = &k[(b * s + j) * d..][..d];
                    qi.iter()
                        .zip(kj)
                        .map(|(&a, &c)| f64::from(a) * f64::from(c))
                        .sum::<f64>()
                        * scale
                } else {
                    f64::NEG_INFINITY
                };
                mx = mx.max(*pj);
            }
            let mut sum = 0.0;
            for pj in p.iter_mut() {
                *pj = (*pj - mx).exp();
                sum += *pj;
            }
            for c in 0..d {
                let acc: f64 = (0..s)
                    .map(|j| p[j] * f64::from(v[(b * s + j) * d + c]))
                    .sum();
                out[(b * s + i) * d + c] = (acc / sum) as f32;
            }
        }
    }
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from((x - y).abs()))
        .fold(0.0, f64::max)
}

fn need<T>(v: Option<T>, what: &str) -> anyhow::Result<T> {
    v.ok_or_else(|| anyhow::anyhow!("{what}: the device path did not apply"))
}

/// Max abs error of the causal flash kernel against the f64 reference and
/// the device composed path (bf16 Q/K/V and P, f32 accumulation).
const CAUSAL_ABS_LIMIT: f64 = 2e-2;

pub fn wan_causal_attn(report: &mut Report, seed: &mut u64) -> StageResult<()> {
    let cases: [(usize, usize, usize, usize, BlockCausal); 7] = [
        // (b, h, s, d, mask)
        (
            1,
            2,
            300,
            128,
            BlockCausal {
                frame_tokens: 60,
                window: 0,
                sink: 0,
            },
        ),
        (
            2,
            3,
            257,
            128,
            BlockCausal {
                frame_tokens: 37,
                window: 2,
                sink: 0,
            },
        ),
        (
            1,
            2,
            511,
            128,
            BlockCausal {
                frame_tokens: 50,
                window: 3,
                sink: 1,
            },
        ),
        (
            1,
            4,
            190,
            64,
            BlockCausal {
                frame_tokens: 19,
                window: 0,
                sink: 2,
            },
        ),
        (
            1,
            1,
            129,
            128,
            BlockCausal {
                frame_tokens: 1,
                window: 0,
                sink: 0,
            },
        ),
        (
            1,
            2,
            640,
            128,
            BlockCausal {
                frame_tokens: 128,
                window: 1,
                sink: 0,
            },
        ),
        (
            1,
            2,
            96,
            64,
            BlockCausal {
                frame_tokens: 200,
                window: 0,
                sink: 0,
            },
        ),
    ];
    for (b, h, s, d, mask) in cases {
        let n = b * h * s * d;
        let (q, k, v) = (
            bf16v(rand(seed, n, 1.0)),
            bf16v(rand(seed, n, 1.0)),
            bf16v(rand(seed, n, 1.0)),
        );
        let shape = [b, h, s, d];
        let (q16, k16, v16) = (t16(&q, &shape)?, t16(&k, &shape)?, t16(&v, &shape)?);
        let got = need(
            attn::device_mma_sdpa_causal(&q16, &k16, &v16, None, false, mask)?,
            "causal flash",
        )?;
        let got = host(&got)?;
        let want = ref_masked(&q, &k, &v, b * h, s, d, &mask);
        let tag = format!(
            "causal_flash_b{b}h{h}s{s}d{d}_f{}w{}k{}",
            mask.frame_tokens, mask.window, mask.sink
        );
        let e_ref = max_abs(&got, &want);
        report.check(
            format!("{tag}_vs_f64"),
            e_ref <= CAUSAL_ABS_LIMIT,
            json!({"max_abs": e_ref}),
            json!({"max_abs": CAUSAL_ABS_LIMIT}),
        )?;
        // The path it replaces: the dense additive mask through sdpa_composed.
        let dense = t32(&mask.dense_mask(s, s), &[1, 1, s, s])?;
        let (q32, k32, v32) = (t32(&q, &shape)?, t32(&k, &shape)?, t32(&v, &shape)?);
        let composed = with_bf16_act(false, || {
            nn::scaled_dot_product_attention_masked(&q32, &k32, &v32, None, Some(&dense))
        })?;
        let e_comp = max_abs(&got, &host(&composed)?);
        report.check(
            format!("{tag}_vs_sdpa_composed"),
            e_comp <= CAUSAL_ABS_LIMIT,
            json!({"max_abs": e_comp, "composed_vs_f64": max_abs(&host(&composed)?, &want)}),
            json!({"max_abs": CAUSAL_ABS_LIMIT}),
        )?;
        // The routed entry point (bf16 activations) returns the kernel's bytes
        // where the fused kernels are the default (bf16 GEMM math, `--mode
        // fast`); in an exact-math context it takes the composed fallback.
        if attn::mma_sdpa_default() {
            let routed =
                with_bf16_act(true, || nn::sdpa_block_causal(&q16, &k16, &v16, None, mask))?;
            let got16 = need(
                attn::device_mma_sdpa_causal(&q16, &k16, &v16, None, true, mask)?,
                "causal flash bf16",
            )?;
            check_ulps(
                report,
                &format!("{tag}_routed_bf16"),
                &host(&routed)?,
                &host(&got16)?,
                true,
            )?;
        }
    }

    // One frame spanning the sequence: no score is masked and the walk is
    // every tile in order, so the causal kernel is the dense V2 kernel.
    for d in [64usize, 128] {
        let (b, h, s) = (1usize, 3usize, 333usize);
        let n = b * h * s * d;
        let shape = [b, h, s, d];
        let (q16, k16, v16) = (
            t16(&bf16v(rand(seed, n, 1.0)), &shape)?,
            t16(&bf16v(rand(seed, n, 1.0)), &shape)?,
            t16(&bf16v(rand(seed, n, 1.0)), &shape)?,
        );
        let all = BlockCausal {
            frame_tokens: s,
            window: 0,
            sink: 0,
        };
        let got = need(
            attn::device_mma_sdpa_causal(&q16, &k16, &v16, None, true, all)?,
            "causal flash",
        )?;
        let want = need(
            attn::device_mma_sdpa_with(&q16, &k16, &v16, None, true, FlashKernel::V2)?,
            "dense V2",
        )?;
        check_ulps(
            report,
            &format!("causal_flash_unmasked_is_dense_v2_d{d}"),
            &host(&got)?,
            &host(&want)?,
            true,
        )?;
    }

    // Timing at SF-Wan 1.3B 480x832: 1560 tokens per latent frame, 12 heads.
    let (h, d, ft) = (12usize, 128usize, 1560usize);
    for frames in [7usize, 21] {
        let s = frames * ft;
        let n = h * s * d;
        let shape = [1, h, s, d];
        let (q16, k16, v16) = (
            t16(&bf16v(rand(seed, n, 1.0)), &shape)?,
            t16(&bf16v(rand(seed, n, 1.0)), &shape)?,
            t16(&bf16v(rand(seed, n, 1.0)), &shape)?,
        );
        let mask = BlockCausal {
            frame_tokens: ft,
            window: 21,
            sink: 0,
        };
        let causal = time_ms(5, || {
            attn::device_mma_sdpa_causal(&q16, &k16, &v16, None, true, mask)?;
            Ok(())
        })?;
        let dense = time_ms(5, || {
            attn::device_mma_sdpa_with(&q16, &k16, &v16, None, true, FlashKernel::V2)?;
            Ok(())
        })?;
        // The composed path holds two f32 [H, S, S] buffers: 7 frames only.
        let composed = if frames == 7 {
            let dm = t32(&mask.dense_mask(s, s), &[1, 1, s, s])?;
            let (q32, k32, v32) = (q16.to_f32_act()?, k16.to_f32_act()?, v16.to_f32_act()?);
            Some(time_ms(3, || {
                with_bf16_act(false, || {
                    nn::scaled_dot_product_attention_masked(&q32, &k32, &v32, None, Some(&dm))
                })?;
                Ok(())
            })?)
        } else {
            None
        };
        report.note(
            format!("causal_flash_timing_{frames}f"),
            json!({"seq": s, "heads": h, "causal_ms": causal, "dense_flash_ms": dense,
                   "sdpa_composed_ms": composed,
                   "causal_over_dense": causal / dense.max(1e-9)}),
        );
    }
    Ok(())
}

pub fn wan_fusion(report: &mut Report, seed: &mut u64) -> StageResult<()> {
    let eps = 1e-6f32;
    for (b, s, dim) in [(2usize, 37usize, 1000usize), (1, 45, 1536)] {
        let n = b * s * dim;
        let shape = [b, s, dim];
        let h = t16(&bf16v(rand(seed, n, 1.5)), &shape)?;
        let a = t16(&bf16v(rand(seed, n, 1.5)), &shape)?;
        let e = t32(&rand(seed, b * 6 * dim, 0.4), &[b, 6, dim])?;
        let w = t32(
            &rand(seed, dim, 0.3)
                .iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            &[dim],
        )?;
        let bias = t32(&rand(seed, dim, 0.2), &[dim])?;
        let tag = format!("{b}x{s}x{dim}");

        let run = |fused: bool| {
            with_bf16_act(true, || {
                fuse::with_fuse(fused, || {
                    fuse::self_residual_norm(&h, &a, &e, 2, &w, &bias, eps)
                })
            })
        };
        let (fh, fnn) = need(run(true)?, "self residual fused")?;
        let (uh, un) = need(run(false)?, "self residual unfused")?;
        check_ulps(
            report,
            &format!("wan_self_res_hidden_fused_vs_unfused_{tag}"),
            &host(&fh)?,
            &host(&uh)?,
            true,
        )?;
        check_ulps(
            report,
            &format!("wan_self_res_norm_fused_vs_unfused_{tag}"),
            &host(&fnn)?,
            &host(&un)?,
            true,
        )?;
        let (th, tn) = need(
            with_bf16_act(true, || {
                fuse::with_host_twins(|| fuse::self_residual_norm(&h, &a, &e, 2, &w, &bias, eps))
            })?,
            "self residual host",
        )?;
        check_ulps(
            report,
            &format!("wan_self_res_hidden_vs_host_{tag}"),
            &host(&fh)?,
            &host(&th)?,
            true,
        )?;
        check_ulps(
            report,
            &format!("wan_self_res_norm_vs_host_{tag}"),
            &host(&fnn)?,
            &host(&tn)?,
            false,
        )?;

        let run = |fused: bool| {
            with_bf16_act(true, || {
                fuse::with_fuse(fused, || {
                    fuse::cross_residual_norm_mod(&h, &a, &e, 4, 3, eps)
                })
            })
        };
        let (fh, fnn) = need(run(true)?, "cross residual fused")?;
        let (uh, un) = need(run(false)?, "cross residual unfused")?;
        check_ulps(
            report,
            &format!("wan_cross_res_hidden_fused_vs_unfused_{tag}"),
            &host(&fh)?,
            &host(&uh)?,
            true,
        )?;
        check_ulps(
            report,
            &format!("wan_cross_res_norm_fused_vs_unfused_{tag}"),
            &host(&fnn)?,
            &host(&un)?,
            true,
        )?;
        let (th, tn) = need(
            with_bf16_act(true, || {
                fuse::with_host_twins(|| fuse::cross_residual_norm_mod(&h, &a, &e, 4, 3, eps))
            })?,
            "cross residual host",
        )?;
        check_ulps(
            report,
            &format!("wan_cross_res_hidden_vs_host_{tag}"),
            &host(&fh)?,
            &host(&th)?,
            true,
        )?;
        check_ulps(
            report,
            &format!("wan_cross_res_norm_vs_host_{tag}"),
            &host(&fnn)?,
            &host(&tn)?,
            false,
        )?;

        let got = need(
            with_bf16_act(true, || fuse::gate_residual(&h, &a, &e, 5))?,
            "gate residual",
        )?;
        let want = need(
            with_bf16_act(true, || {
                fuse::with_host_twins(|| fuse::gate_residual(&h, &a, &e, 5))
            })?,
            "gate residual host",
        )?;
        check_ulps(
            report,
            &format!("wan_gate_residual_vs_host_{tag}"),
            &host(&got)?,
            &host(&want)?,
            true,
        )?;
    }

    // q/k RMSNorm across heads + RoPE from a fused QKV projection.
    for (s, heads, d, rope) in [
        (33usize, 3usize, 64usize, true),
        (50, 12, 128, true),
        (50, 12, 128, false),
    ] {
        let width = 3 * heads * d;
        let x = t16(&bf16v(rand(seed, s * width, 2.0)), &[1, s, width])?;
        let w = t32(
            &bf16v(rand(seed, heads * d, 0.3).iter().map(|v| 1.0 + v).collect()),
            &[heads * d],
        )?;
        let ang = rand(seed, s * d, 2.0);
        let cos = t32(&ang.iter().map(|a| a.cos()).collect::<Vec<_>>(), &[s, d])?;
        let sin = t32(&ang.iter().map(|a| a.sin()).collect::<Vec<_>>(), &[s, d])?;
        let r = || {
            rope.then(|| Rope {
                cos: &cos,
                sin: &sin,
            })
        };
        let got = need(
            with_bf16_act(true, || {
                fuse::qk_norm_rope(&x, heads * d, heads, &w, r(), 1e-6)
            })?,
            "qk",
        )?;
        let want = need(
            with_bf16_act(true, || {
                fuse::with_host_twins(|| fuse::qk_norm_rope(&x, heads * d, heads, &w, r(), 1e-6))
            })?,
            "qk host",
        )?;
        let err = max_abs(&host(&got)?, &host(&want)?);
        // rsqrtf vs the host's 1/sqrt can move a value across a bf16
        // rounding boundary: one ulp of an O(1) value, 2^-7.
        report.check(
            format!("wan_qk_norm_rope_vs_host_s{s}h{heads}d{d}_rope{rope}"),
            got.is_bf16() && err <= 1.0 / 64.0,
            json!({"max_abs": err, "bf16": got.is_bf16()}),
            json!({"max_abs": 1.0 / 64.0}),
        )?;
    }

    // FastWan 1.3B, 480x832, 81 frames: 32 760 tokens x 1536.
    let (s, dim) = (32_760usize, 1536usize);
    let n = s * dim;
    let h = t16(&bf16v(rand(seed, n, 1.0)), &[1, s, dim])?;
    let a = t16(&bf16v(rand(seed, n, 1.0)), &[1, s, dim])?;
    let e = t32(&rand(seed, 6 * dim, 0.4), &[1, 6, dim])?;
    let w = t32(&vec![1.0; dim], &[dim])?;
    let bias = t32(&vec![0.0; dim], &[dim])?;
    let mut timing = serde_json::Map::new();
    for fused in [false, true] {
        let key = if fused { "fused" } else { "unfused" };
        let t_self = time_ms(20, || {
            with_bf16_act(true, || {
                fuse::with_fuse(fused, || {
                    fuse::self_residual_norm(&h, &a, &e, 2, &w, &bias, eps)
                })
            })?;
            Ok(())
        })?;
        let t_cross = time_ms(20, || {
            with_bf16_act(true, || {
                fuse::with_fuse(fused, || {
                    fuse::cross_residual_norm_mod(&h, &a, &e, 4, 3, eps)
                })
            })?;
            Ok(())
        })?;
        timing.insert(format!("{key}_self_ms"), json!(t_self));
        timing.insert(format!("{key}_cross_ms"), json!(t_cross));
    }
    // The chain the f32-activation block runs today, for scale.
    let (h32, a32) = (h.to_f32_act()?, a.to_f32_act()?);
    let t_f32 = time_ms(20, || {
        with_bf16_act(false, || -> anyhow::Result<()> {
            let x = h32.residual_gate_add_e(&a32, &e, 2)?;
            x.layer_norm(eps, Some(&w), Some(&bias))?;
            let y = x.add(&a32)?;
            y.ln_adaln_e(&e, 4, 3, eps)?;
            Ok(())
        })
    })?;
    timing.insert("f32_chain_ms".into(), json!(t_f32));
    report.note(
        "wan_fusion_timing_1p3b_81f",
        serde_json::Value::Object(timing),
    );
    Ok(())
}
