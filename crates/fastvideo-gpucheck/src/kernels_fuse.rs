//! `dit_fusion`: the Phase 3c DiT block fusions against the unfused op chains
//! they replace ([`fastvideo_cudarc::ltx2::fuse`]).
//!
//! Every fused kernel keeps its chain's rounding points, so each check is
//! **bit-exact** (0 bf16 ulps) against the same chain run through the public
//! tensor ops with the fusion off. The shapes are small odd ones (partial
//! reduction blocks, several heads); a second pass times fused vs unfused at
//! the LTX-2.5 4K stage-2 video width.

use fastvideo_cudarc::h3::fused16::{self, AdaRows, NormOut};
use fastvideo_cudarc::ltx2::attention::DeviceRope;
use fastvideo_cudarc::ltx2::fuse::{self, Mod, Norm, Residual};
use fastvideo_cudarc::wan::quant;
use fastvideo_cudarc::wan::tensor::with_bf16_act;
use fastvideo_cudarc::CudaTensor;
use fastvideo_models::ltx2::SplitRope;
use serde_json::json;

use crate::kernels_fp8::{bf16v, check_ulps, dev, host, rand, t16, t32, time_ms};
use crate::report::{Report, StageResult};

fn rope(seed: &mut u64, heads: usize, tokens: usize, d: usize) -> anyhow::Result<DeviceRope> {
    let half = d / 2;
    let ang = rand(seed, heads * tokens * half, 2.0);
    let table = SplitRope {
        heads,
        tokens,
        half,
        cos: ang.iter().map(|a| a.cos()).collect(),
        sin: ang.iter().map(|a| a.sin()).collect(),
    };
    Ok(DeviceRope::upload(&table)?)
}

/// `rms_norm(x, 1 + tab[scale]) + tab[shift]` as the block runs it unfused.
fn adaln_ops(
    x: &CudaTensor,
    tab: &CudaTensor,
    scale: usize,
    shift: usize,
    eps: f32,
) -> anyhow::Result<CudaTensor> {
    let s = tab.narrow(0, scale, 1)?;
    let h = tab.narrow(0, shift, 1)?;
    Ok(x.rms_norm(&s.try_add_scalar(1.0)?, eps)?.add(&h)?)
}

/// `rms_norm(x, w)` then `x · (1 + tab[scale]) + tab[shift]`.
fn then_mod_ops(
    x: &CudaTensor,
    w: &CudaTensor,
    tab: &CudaTensor,
    scale: usize,
    shift: usize,
    eps: f32,
) -> anyhow::Result<CudaTensor> {
    let s = tab.narrow(0, scale, 1)?;
    let h = tab.narrow(0, shift, 1)?;
    Ok(x.rms_norm(w, eps)?.mul(&s.try_add_scalar(1.0)?)?.add(&h)?)
}

fn qk_ops(
    x: &CudaTensor,
    w: &CudaTensor,
    eps: f32,
    heads: usize,
    d: usize,
    rope: Option<&DeviceRope>,
) -> anyhow::Result<CudaTensor> {
    let t = x.rms_norm(w, eps)?.split_heads_bhsd(0, heads, d)?;
    Ok(match rope {
        Some(r) => r.apply(&t)?,
        None => t,
    })
}

fn gate_merge_ops(out: &CudaTensor, logits: &CudaTensor) -> anyhow::Result<CudaTensor> {
    let [_, heads, seq, _] = out.shape[..] else {
        anyhow::bail!("out {:?}", out.shape);
    };
    let g = logits.try_sigmoid()?.try_mul_scalar(2.0)?;
    let g = g.permute(&[0, 2, 1])?.reshape(vec![1, heads, seq, 1])?;
    Ok(out.mul(&g)?.merge_heads()?)
}

fn need(t: Option<CudaTensor>, what: &str) -> anyhow::Result<CudaTensor> {
    t.ok_or_else(|| anyhow::anyhow!("{what}: the fused path did not apply"))
}

pub fn dit_fusion(report: &mut Report, seed: &mut u64) -> StageResult<()> {
    with_bf16_act(true, || parity(report, seed))?;
    with_bf16_act(true, || h3_parity(report, seed))?;
    with_bf16_act(true, || timing(report, seed))
}

/// `ltx_rope`: the LTX-2 rotary tables gathered on the device from their
/// factored form ([`fastvideo_cudarc::ltx2::transformer::Ropes`]) against the
/// direct host build ([`Ltx2RopeTables::with_conditioning`], CUDA division)
/// widened by [`SplitRope::rotate_half_tables`] — the tables the pipeline
/// uploaded before. Bit-exact, for T2V, I2V (first-frame keyframe), several
/// keyframes and an IC-LoRA reference, at the 1080p 6 s stage-2 geometry
/// (and a small one); also times both builds.
pub fn ltx_rope(report: &mut Report) -> StageResult<()> {
    use fastvideo_cudarc::ltx2::transformer::Ropes;
    use fastvideo_models::ltx2::config::Ltx2TransformerConfig;
    use fastvideo_models::ltx2::rope::{Ltx2RopeTables, ReferenceBlock, ScalarDivision};
    let cfg = Ltx2TransformerConfig::ltx2_5_22b();
    let reference = ReferenceBlock {
        grid: [19, 17, 30],
        downscale: 2,
    };
    type Case<'a> = (&'a str, [usize; 3], &'a [usize], Option<ReferenceBlock>);
    let cases: [Case; 5] = [
        ("t2v_1080p", [19, 34, 60], &[], None),
        ("i2v_1080p", [19, 34, 60], &[0], None),
        ("keyframes_540p", [19, 17, 30], &[0, 72, 144], None),
        ("ref2v_540p", [19, 17, 30], &[0], Some(reference)),
        ("t2v_small", [3, 2, 3], &[], None),
    ];
    for (name, grid, extra, reference) in cases {
        let audio_tokens = 151;
        let d = dev()?;
        d.synchronize()?;
        let timer = std::time::Instant::now();
        let ropes = Ropes::with_conditioning(&cfg, grid, extra, reference, audio_tokens, 24.0)?;
        d.synchronize()?;
        let device_s = timer.elapsed().as_secs_f64();
        let timer = std::time::Instant::now();
        let host = Ltx2RopeTables::with_conditioning(
            &cfg,
            grid,
            extra,
            reference,
            audio_tokens,
            24.0,
            ScalarDivision::Reciprocal,
        );
        let host_s = timer.elapsed().as_secs_f64();
        let mut differing = serde_json::Map::new();
        let mut ok = true;
        for (table, got, want) in [
            ("video", &ropes.video, &host.video),
            ("audio", &ropes.audio, &host.audio),
            ("cross_video", &ropes.cross_video, &host.cross_video),
            ("cross_audio", &ropes.cross_audio, &host.cross_audio),
        ] {
            let (gc, gs) = got.host_tables()?;
            let (wc, ws) = want.rotate_half_tables();
            let diff = |a: &[f32], b: &[f32]| {
                if a.len() != b.len() {
                    return usize::MAX;
                }
                a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
            };
            let n = diff(&gc, &wc).saturating_add(diff(&gs, &ws));
            ok &= n == 0;
            differing.insert(table.into(), json!(n));
        }
        report.check(
            format!("ltx_rope_{name}"),
            ok,
            json!({
                "differing_values": differing,
                "video_tokens": host.video.tokens,
                "device_build_s": device_s,
                "host_build_s": host_s,
            }),
            json!({"differing": 0}),
        )?;
    }
    Ok(())
}

/// The bytes of an MXFP8 activation: codes of the real rows, every scale.
fn mx_bytes(a: &quant::MxAct) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    let d = dev()?;
    let q = d.stream.memcpy_dtov(&a.q)?;
    Ok((q[..a.rows * a.k].to_vec(), d.stream.memcpy_dtov(&a.s)?))
}

fn check_bytes(
    report: &mut Report,
    name: &str,
    got: &quant::MxAct,
    want: &quant::MxAct,
) -> StageResult<()> {
    let (gq, gs) = mx_bytes(got)?;
    let (wq, ws) = mx_bytes(want)?;
    let dq = gq.iter().zip(&wq).filter(|(a, b)| a != b).count();
    let ds = gs.iter().zip(&ws).filter(|(a, b)| a != b).count();
    let ok = gq.len() == wq.len() && gs.len() == ws.len() && dq == 0 && ds == 0;
    report.check(
        name,
        ok,
        json!({"codes_differing": dq, "scales_differing": ds, "codes": gq.len(), "scales": gs.len()}),
        json!({"differing": 0}),
    )
}

fn h3_parity(report: &mut Report, seed: &mut u64) -> StageResult<()> {
    // merge_heads -> MXFP8 vs merge_heads then mxfp8_quantize.
    let (s, heads, d) = (77usize, 6usize, 64usize);
    let k = heads * d;
    let o = t16(&bf16v(rand(seed, heads * s * d, 1.5)), &[1, heads, s, d])?;
    let got = fused16::with_fuse(true, || fused16::merge_heads_mx(&o))?
        .ok_or_else(|| anyhow::anyhow!("merge_heads_mx: the fused path did not apply"))?;
    let merged = o.merge_heads()?;
    let mut want = quant::MxAct::alloc(s, k)?;
    let m16 = merged
        .device_slice_bf16()
        .ok_or_else(|| anyhow::anyhow!("merged heads not device bf16"))?;
    quant::mxfp8_quantize_raw(quant::ptr(m16), true, &mut want)?;
    check_bytes(report, "h3_merge_heads_mx", &got, &want)?;
    // An f32 attention output (VSA's) is rounded to bf16 inside the merge.
    let o32 = t32(&rand(seed, heads * s * d, 1.5), &[1, heads, s, d])?;
    let got = fused16::with_fuse(true, || fused16::merge_heads_mx(&o32))?
        .ok_or_else(|| anyhow::anyhow!("merge_heads_mx f32: the fused path did not apply"))?;
    let merged = o32.quantize_bf16()?.merge_heads()?;
    let mut want = quant::MxAct::alloc(s, k)?;
    let m16 = merged
        .device_slice_bf16()
        .ok_or_else(|| anyhow::anyhow!("merged heads not device bf16"))?;
    quant::mxfp8_quantize_raw(quant::ptr(m16), true, &mut want)?;
    check_bytes(report, "h3_merge_heads_mx_f32", &got, &want)?;

    // VSA feed: q/k (norm + partial RoPE) and head splits written as f32
    // holding exactly the bf16 path's values.
    let (s2, h2, d2, r2) = (83usize, 3usize, 128usize, 96usize);
    let inner = h2 * d2;
    let packed = t16(&bf16v(rand(seed, s2 * 4 * inner, 1.0)), &[1, s2, 4 * inner])?;
    let nq = t32(
        &bf16v(rand(seed, d2, 0.1).into_iter().map(|v| 1.0 + v).collect()),
        &[d2],
    )?;
    let ang: Vec<f32> = (0..s2 * r2)
        .map(|i| (i as f32 * 0.017).sin() * 3.0)
        .collect();
    let cos = t32(&ang.iter().map(|a| a.cos()).collect::<Vec<_>>(), &[s2, r2])?;
    let sin = t32(&ang.iter().map(|a| a.sin()).collect::<Vec<_>>(), &[s2, r2])?;
    for col in [0usize, inner] {
        let got = fused16::qk_norm_rope_f32(&packed, &nq, Some((&cos, &sin)), h2, d2, col, 1e-6)?
            .ok_or_else(|| anyhow::anyhow!("qk_norm_rope_f32 did not apply"))?;
        let want = fused16::qk_norm_rope(&packed, &nq, Some((&cos, &sin)), h2, d2, col, 1e-6)?;
        report.check(
            format!("h3_qk_norm_rope_f32_col{col}_dtype"),
            !got.is_bf16() && want.is_bf16() && got.shape == want.shape,
            json!({"got_bf16": got.is_bf16(), "shape": got.shape}),
            json!({"got_bf16": false}),
        )?;
        check_ulps(
            report,
            &format!("h3_qk_norm_rope_f32_col{col}"),
            &host(&got)?,
            &host(&want)?,
            true,
        )?;
    }
    // Head splits (fvf_split_heads_rows) against a host gather of the
    // packed values: bf16 -> bf16 and bf16 -> f32, batch 1 and 2.
    let host_split = |x: &[f32], b: usize, s: usize, w: usize, col: usize| -> Vec<f32> {
        let mut o = vec![0.0f32; b * h2 * s * d2];
        for bi in 0..b {
            for si in 0..s {
                for h in 0..h2 {
                    for p in 0..d2 {
                        o[((bi * h2 + h) * s + si) * d2 + p] =
                            x[(bi * s + si) * w + col + h * d2 + p];
                    }
                }
            }
        }
        o
    };
    let packed_h = host(&packed)?;
    for col in [2 * inner, 3 * inner] {
        let want = host_split(&packed_h, 1, s2, 4 * inner, col);
        let got = fused16::split_heads_f32(&packed, col, h2, d2)?
            .ok_or_else(|| anyhow::anyhow!("split_heads_f32 did not apply"))?;
        check_ulps(
            report,
            &format!("h3_split_heads_f32_col{col}"),
            &host(&got)?,
            &want,
            true,
        )?;
        let got16 = packed.split_heads_bhsd(col, h2, d2)?;
        check_ulps(
            report,
            &format!("split_heads_rows_bf16_col{col}"),
            &host(&got16)?,
            &want,
            true,
        )?;
    }
    let packed2 = t16(
        &bf16v(rand(seed, 2 * s2 * 4 * inner, 1.0)),
        &[2, s2, 4 * inner],
    )?;
    let want = host_split(&host(&packed2)?, 2, s2, 4 * inner, inner);
    let got = packed2.split_heads_bhsd(inner, h2, d2)?;
    check_ulps(
        report,
        "split_heads_rows_bf16_batch2",
        &host(&got)?,
        &want,
        true,
    )?;

    // Last residual + next block's norm vs gate_residual then norm_mod.
    let (s, dm) = (45usize, 640usize);
    let t = 12usize;
    let mut tab = bf16v(rand(seed, t * 6 * dm, 0.3));
    for r in 0..t {
        for slot in [1usize, 4] {
            for v in &mut tab[(r * 6 + slot) * dm..(r * 6 + slot + 1) * dm] {
                *v += 1.0;
            }
        }
    }
    let idx: Vec<u32> = (0..s).map(|r| (r % 3) as u32).collect();
    let ada = AdaRows {
        tab: t32(&tab, &[t, 6, dm])?,
        hidden: dm,
        idx: std::sync::Arc::new(idx.clone()),
        idx_dev: Some(std::sync::Arc::new(dev()?.stream.memcpy_stod(&idx)?)),
    };
    let res = t16(&bf16v(rand(seed, s * dm, 1.5)), &[1, s, dm])?;
    let f = t16(&bf16v(rand(seed, s * dm, 1.5)), &[1, s, dm])?;
    let w = t32(
        &bf16v(rand(seed, dm, 0.2).into_iter().map(|v| 1.0 + v).collect()),
        &[dm],
    )?;
    let (base_g, base_n) = (3usize, 6usize);
    let want_h = fused16::gate_residual(&res, &f, &ada, base_g, 5)?;
    for mx in [false, true] {
        let (h, n) = fused16::with_fuse(true, || {
            fused16::gate_res_norm_mod(&res, &f, &w, &ada, base_g, 5, base_n, (1, 0), 1e-6, mx)
        })?
        .ok_or_else(|| anyhow::anyhow!("gate_res_norm_mod: the fused path did not apply"))?;
        let want_n = fused16::norm_mod(&want_h, &w, &ada, base_n, 1, 0, 1e-6, mx)?;
        let tag = if mx { "mx" } else { "bf16" };
        check_ulps(
            report,
            &format!("h3_gate_res_hidden_{tag}"),
            &host(&h)?,
            &host(&want_h)?,
            true,
        )?;
        match (n, want_n) {
            (NormOut::T(a), NormOut::T(b)) => check_ulps(
                report,
                "h3_gate_res_norm_mod_bf16",
                &host(&a)?,
                &host(&b)?,
                true,
            )?,
            (NormOut::Mx(a), NormOut::Mx(b)) => {
                check_bytes(report, "h3_gate_res_norm_mod_mx", &a, &b)?
            }
            _ => {
                return Err(
                    anyhow::anyhow!("gate_res_norm_mod: output kinds differ (mx={mx})").into(),
                )
            }
        }
    }
    // VSA-H3: the dense prefix rows written into the output in place vs the
    // narrow + cat of the whole output (f32 q/k/v/gate, as the H3 block feeds it).
    {
        use fastvideo_cudarc::h3::vsa::{H3Vsa, H3VsaConfig};
        let layout =
            fastvideo_models::h3::packing::H3PackedLayout::new(70, (5, 8, 12), 33, [1, 2, 2])
                .map_err(anyhow::Error::msg)?;
        let (vh, vd) = (2usize, 128usize);
        let seq = layout.sequence_length();
        let shape = [1, vh, seq, vd];
        let n = vh * seq * vd;
        let parts: Vec<Vec<f32>> = (0..4).map(|_| bf16v(rand(seed, n, 1.0))).collect();
        let vsa = H3Vsa::new(
            &layout,
            vh,
            vd,
            H3VsaConfig {
                sparsity: 0.5,
                group: 2,
                tile_size: 64,
            },
        )?;
        let run = |on: bool| -> anyhow::Result<CudaTensor> {
            fused16::with_fuse(on, || {
                Ok(vsa.attend(
                    t32(&parts[0], &shape)?,
                    t32(&parts[1], &shape)?,
                    t32(&parts[2], &shape)?,
                    Some(t32(&parts[3], &shape)?),
                )?)
            })
        };
        let (got, want) = (run(true)?, run(false)?);
        report.check(
            "h3_vsa_prefix_in_place_dtype",
            got.is_bf16() == want.is_bf16() && got.shape == want.shape,
            json!({"got_bf16": got.is_bf16(), "want_bf16": want.is_bf16()}),
            json!({}),
        )?;
        check_ulps(
            report,
            "h3_vsa_prefix_in_place",
            &host(&got)?,
            &host(&want)?,
            true,
        )?;
    }

    let off = fused16::with_fuse(false, || -> anyhow::Result<bool> {
        Ok(fused16::merge_heads_mx(&o)?.is_none()
            && fused16::gate_res_norm_mod(
                &res,
                &f,
                &w,
                &ada,
                base_g,
                5,
                base_n,
                (1, 0),
                1e-6,
                false,
            )?
            .is_none())
    })?;
    report.check(
        "h3_fuse_off_declines",
        off,
        json!({"declined": off}),
        json!({"declined": true}),
    )?;
    Ok(())
}

fn parity(report: &mut Report, seed: &mut u64) -> StageResult<()> {
    let eps = 1e-6f32;
    // LTX q/k: 5 heads of 64 (inner 320, one partial reduction pass).
    let (s, heads, d) = (97usize, 5usize, 64usize);
    let inner = heads * d;
    let x = bf16v(rand(seed, s * inner, 2.0));
    let xt = t16(&x, &[1, s, inner])?;
    let w = bf16v(
        rand(seed, inner, 0.2)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect(),
    );
    let rp = rope(seed, heads, s, d)?;
    for (wname, wt) in [("w32", t32(&w, &[inner])?), ("w16", t16(&w, &[inner])?)] {
        for (rname, r) in [("rope", Some(&rp)), ("norope", None)] {
            let got = need(
                fuse::with_fuse(true, || fuse::qk_norm_rope(&xt, &wt, eps, heads, d, r))?,
                "qk",
            )?;
            let want = fuse::with_fuse(false, || qk_ops(&xt, &wt, eps, heads, d, r))?;
            report.check(
                format!("ltx_qk_norm_rope_{wname}_{rname}_shape"),
                got.shape == want.shape && got.is_bf16(),
                json!({"got": got.shape, "want": want.shape, "bf16": got.is_bf16()}),
                json!({}),
            )?;
            check_ulps(
                report,
                &format!("ltx_qk_norm_rope_{wname}_{rname}"),
                &host(&got)?,
                &host(&want)?,
                true,
            )?;
        }
    }

    // Modulated norms and residuals: width 1000 (not a multiple of the block).
    let (s, dm) = (61usize, 1000usize);
    let x = bf16v(rand(seed, s * dm, 1.5));
    let u = bf16v(rand(seed, s * dm, 1.5));
    let (xt, ut) = (t16(&x, &[1, s, dm])?, t16(&u, &[1, s, dm])?);
    let tab = t16(&bf16v(rand(seed, 9 * dm, 0.4)), &[9, dm])?;
    let tab3 = tab.reshape(vec![1, 9, dm])?;
    let ones = t32(&vec![1.0; dm], &[dm])?;
    let m = Mod {
        tab: &tab,
        scale: 4,
        shift: 3,
    };
    let got = need(
        fuse::with_fuse(true, || fuse::norm_mod(&xt, Norm::AdaLn(m), eps))?,
        "adaln",
    )?;
    let want = adaln_ops(&xt, &tab, 4, 3, eps)?;
    check_ulps(report, "ltx_adaln", &host(&got)?, &host(&want)?, true)?;
    let m7 = Mod {
        tab: &tab,
        scale: 7,
        shift: 6,
    };
    let got = need(
        fuse::with_fuse(true, || fuse::norm_mod(&xt, Norm::ThenMod(&ones, m7), eps))?,
        "then_mod",
    )?;
    let want = then_mod_ops(&xt, &ones, &tab, 7, 6, eps)?;
    check_ulps(
        report,
        "ltx_norm_then_mod",
        &host(&got)?,
        &host(&want)?,
        true,
    )?;

    // residual_gate_add_e (row 2 of [1, 9, D]) then the text-cross norm.
    let (h, n) = fuse::with_fuse(true, || {
        fuse::res_norm_mod(
            &xt,
            Residual::Gated {
                u: &ut,
                gates: &tab3,
                row: 2,
            },
            Norm::ThenMod(&ones, m7),
            eps,
        )
    })?
    .ok_or_else(|| anyhow::anyhow!("res_norm_mod gated: the fused path did not apply"))?;
    let want_h = xt.residual_gate_add_e(&ut, &tab3, 2)?;
    let want_n = then_mod_ops(&want_h, &ones, &tab, 7, 6, eps)?;
    check_ulps(
        report,
        "ltx_res_gate_hidden",
        &host(&h)?,
        &host(&want_h)?,
        true,
    )?;
    check_ulps(
        report,
        "ltx_res_gate_then_mod",
        &host(&n)?,
        &host(&want_n)?,
        true,
    )?;
    // Text-cross gate (u · tab[8], then add) then the a↔v query AdaLN.
    let cross = t16(&bf16v(rand(seed, 4 * dm, 0.4)), &[4, dm])?;
    let q = Mod {
        tab: &cross,
        scale: 0,
        shift: 1,
    };
    let (h, n) = fuse::with_fuse(true, || {
        fuse::res_norm_mod(
            &xt,
            Residual::Gated {
                u: &ut,
                gates: &tab,
                row: 8,
            },
            Norm::AdaLn(q),
            eps,
        )
    })?
    .ok_or_else(|| anyhow::anyhow!("res_norm_mod text gate: the fused path did not apply"))?;
    let gated = ut.mul(&tab.narrow(0, 8, 1)?.reshape(vec![1, 1, dm])?)?;
    let want_h = xt.add(&gated)?;
    let want_n = adaln_ops(&want_h, &cross, 0, 1, eps)?;
    check_ulps(
        report,
        "ltx_text_gate_hidden",
        &host(&h)?,
        &host(&want_h)?,
        true,
    )?;
    check_ulps(
        report,
        "ltx_text_gate_adaln",
        &host(&n)?,
        &host(&want_n)?,
        true,
    )?;
    // Plain add (LTX-2.0 text cross) and a [1, 1, D] gate (a↔v).
    let (h, _) = fuse::with_fuse(true, || {
        fuse::res_norm_mod(&xt, Residual::Plain(&ut), Norm::AdaLn(q), eps)
    })?
    .ok_or_else(|| anyhow::anyhow!("res_norm_mod plain: the fused path did not apply"))?;
    check_ulps(
        report,
        "ltx_plain_add_hidden",
        &host(&h)?,
        &host(&xt.add(&ut)?)?,
        true,
    )?;
    let g1 = t16(&bf16v(rand(seed, dm, 0.4)), &[1, 1, dm])?;
    let (h, n) = fuse::with_fuse(true, || {
        fuse::res_norm_mod(
            &xt,
            Residual::Gated {
                u: &ut,
                gates: &g1,
                row: 0,
            },
            Norm::AdaLn(m),
            eps,
        )
    })?
    .ok_or_else(|| anyhow::anyhow!("res_norm_mod av: the fused path did not apply"))?;
    let want_h = xt.residual_gate_add_e(&ut, &g1, 0)?;
    check_ulps(
        report,
        "ltx_av_gate_hidden",
        &host(&h)?,
        &host(&want_h)?,
        true,
    )?;
    check_ulps(
        report,
        "ltx_av_gate_adaln",
        &host(&n)?,
        &host(&adaln_ops(&want_h, &tab, 4, 3, eps)?)?,
        true,
    )?;

    // LTX-2.5 head gates + merge.
    let (s, heads, d) = (53usize, 6usize, 64usize);
    let o = t16(&bf16v(rand(seed, heads * s * d, 1.0)), &[1, heads, s, d])?;
    let l = t16(&bf16v(rand(seed, s * heads, 2.0)), &[1, s, heads])?;
    let got = need(
        fuse::with_fuse(true, || fuse::gate_merge(&o, &l))?,
        "gate_merge",
    )?;
    let want = gate_merge_ops(&o, &l)?;
    report.check(
        "ltx_gate_merge_shape",
        got.shape == want.shape,
        json!({"got": got.shape, "want": want.shape}),
        json!({}),
    )?;
    check_ulps(report, "ltx_gate_merge", &host(&got)?, &host(&want)?, true)?;

    // Off switch: with the fusion off every entry point declines.
    let off = fuse::with_fuse(false, || -> anyhow::Result<bool> {
        Ok(fuse::norm_mod(&xt, Norm::AdaLn(m), eps)?.is_none()
            && fuse::gate_merge(&o, &l)?.is_none())
    })?;
    report.check(
        "ltx_fuse_off_declines",
        off,
        json!({"declined": off}),
        json!({"declined": true}),
    )?;
    Ok(())
}

/// Fused vs unfused wall time at the LTX-2.5 video width (4096 = 32 x 128)
/// over 16k tokens (one FFN chunk of the 4K stage-2 stream).
fn timing(report: &mut Report, seed: &mut u64) -> StageResult<()> {
    let eps = 1e-6f32;
    let (s, heads, d) = (16_384usize, 32usize, 128usize);
    let dm = heads * d;
    let x = t16(&bf16v(rand(seed, s * dm, 1.0)), &[1, s, dm])?;
    let u = t16(&bf16v(rand(seed, s * dm, 1.0)), &[1, s, dm])?;
    let w = t32(&vec![1.0; dm], &[dm])?;
    let tab = t16(&bf16v(rand(seed, 9 * dm, 0.3)), &[9, dm])?;
    let tab3 = tab.reshape(vec![1, 9, dm])?;
    let rp = rope(seed, heads, s, d)?;
    let o = t16(&bf16v(rand(seed, s * dm, 1.0)), &[1, heads, s, d])?;
    let l = t16(&bf16v(rand(seed, s * heads, 1.0)), &[1, s, heads])?;
    let mut row = serde_json::Map::new();
    let mut pair = |name: &str,
                    fused: &mut dyn FnMut() -> anyhow::Result<()>,
                    ops: &mut dyn FnMut() -> anyhow::Result<()>|
     -> anyhow::Result<()> {
        let f = time_ms(10, &mut *fused)?;
        let u = time_ms(10, &mut *ops)?;
        row.insert(
            name.to_string(),
            json!({"fused_ms": f, "unfused_ms": u, "speedup": u / f.max(1e-9)}),
        );
        Ok(())
    };
    pair(
        "qk_norm_rope",
        &mut || {
            fuse::with_fuse(true, || {
                fuse::qk_norm_rope(&x, &w, eps, heads, d, Some(&rp))
            })?;
            Ok(())
        },
        &mut || {
            qk_ops(&x, &w, eps, heads, d, Some(&rp))?;
            Ok(())
        },
    )?;
    let m = Mod {
        tab: &tab,
        scale: 1,
        shift: 0,
    };
    pair(
        "adaln",
        &mut || {
            fuse::with_fuse(true, || fuse::norm_mod(&x, Norm::AdaLn(m), eps))?;
            Ok(())
        },
        &mut || {
            adaln_ops(&x, &tab, 1, 0, eps)?;
            Ok(())
        },
    )?;
    let m7 = Mod {
        tab: &tab,
        scale: 7,
        shift: 6,
    };
    pair(
        "res_gate_then_mod",
        &mut || {
            fuse::with_fuse(true, || {
                fuse::res_norm_mod(
                    &x,
                    Residual::Gated {
                        u: &u,
                        gates: &tab3,
                        row: 2,
                    },
                    Norm::ThenMod(&w, m7),
                    eps,
                )
            })?;
            Ok(())
        },
        &mut || {
            let h = x.residual_gate_add_e(&u, &tab3, 2)?;
            then_mod_ops(&h, &w, &tab, 7, 6, eps)?;
            Ok(())
        },
    )?;
    pair(
        "gate_merge",
        &mut || {
            fuse::with_fuse(true, || fuse::gate_merge(&o, &l))?;
            Ok(())
        },
        &mut || {
            gate_merge_ops(&o, &l)?;
            Ok(())
        },
    )?;
    report.note(
        "ltx_fusion_timing_16k_x_4096",
        serde_json::Value::Object(row),
    );
    Ok(())
}
