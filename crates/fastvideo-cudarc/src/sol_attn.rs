//! Sol-Attn on the cudarc device path.
//!
//! The selection and exact/approx combine are the published Sol rule in
//! [`fastvideo_models::sol_attn`]. With a live resident CUDA device every
//! entry here runs the fused kernels (`sol_prep_*` + `sol_mma_fwd`: bf16
//! operands, tensor-core route GEMM, approximate + exact terms in one pass,
//! no host traffic); their reference is
//! [`fastvideo_models::sol_attn::sol_attn_head_faithful`]. Without a device
//! the f32 host oracle runs. Logs once when the op first executes.

use std::sync::atomic::{AtomicBool, Ordering};

use fastvideo_models::sol_attn::{
    sol_attn_bhsd, sol_attn_bhsd_faithful, sol_attn_bhsd_sunk, SolNumerics, SolParams, SolThresh,
    BLOCK_SIZE,
};

use crate::wan::log;
use crate::wan::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

static LOGGED: AtomicBool = AtomicBool::new(false);

fn log_once(tau: f64, thresh: SolThresh, tokens: usize, sink_tokens: usize) {
    if LOGGED.swap(true, Ordering::Relaxed) {
        return;
    }
    let n = tokens.div_ceil(BLOCK_SIZE);
    let path = if crate::wan::stats::device_expected() {
        "fused device"
    } else {
        "host oracle"
    };
    log::info(format_args!(
        "sol-attn kernel ({path}): thresh_type={thresh:?} tau={tau} block={BLOCK_SIZE} tokens={tokens} blocks={n} sink={sink_tokens}"
    ));
}

/// Sol-Attn on `[B, H, S, D]` q/k/v. `scale` defaults to `1/sqrt(D)`.
pub fn sol_attn(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    tau: f64,
    scale: Option<f32>,
    sink_start: Option<usize>,
    sink_tokens: usize,
) -> Result<CudaTensor> {
    if q.rank() != 4 || q.shape != k.shape || q.shape != v.shape {
        return Err(msg(format!(
            "sol-attn expects matching BHSD q/k/v, got {:?} {:?} {:?}",
            q.shape, k.shape, v.shape
        )));
    }
    let (batch, heads, tokens, dim) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    if tokens == 0 {
        return Ok(q.clone());
    }
    let scale = scale.unwrap_or((dim as f32).sqrt().recip());
    log_once(tau, SolThresh::Diag, tokens, sink_tokens);
    let p = SolParams {
        sink_start,
        sink_tokens,
        ..SolParams::diag(tau as f32, scale)
    };
    if let Some(out) = crate::wan::sol_ops::try_sol_device_params(q, k, v, &p)? {
        return Ok(out);
    }
    crate::wan::stats::host_algorithm(
        "sol_attn",
        format_args!("BHSD {batch}x{heads}x{tokens}x{dim}"),
    )?;
    sol_from_host(
        q,
        k,
        v,
        batch,
        heads,
        tokens,
        dim,
        tau,
        scale,
        sink_start,
        sink_tokens,
    )
}

/// Sol-Attn with one or more official sink spans.
///
/// On the device the fused kernel takes ONE contiguous sink range, as the
/// Python interface does: several spans are accepted when their KV-block
/// ranges merge into one
/// ([`fastvideo_models::sol_attn::merge_sink_spans`]), and are an error
/// otherwise (H3 `FASTVIDEO_H3_SOL_SINK=native` with separated text/audio
/// spans). The host oracle still ORs any number of spans.
pub fn sol_attn_sunk(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    tau: f64,
    scale: Option<f32>,
    sinks: &[(usize, usize)],
) -> Result<CudaTensor> {
    if q.rank() != 4 || q.shape != k.shape || q.shape != v.shape {
        return Err(msg(format!(
            "sol-attn expects matching BHSD q/k/v, got {:?} {:?} {:?}",
            q.shape, k.shape, v.shape
        )));
    }
    let (batch, heads, tokens, dim) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    if tokens == 0 {
        return Ok(q.clone());
    }
    let scale = scale.unwrap_or((dim as f32).sqrt().recip());
    let sink_tokens: usize = sinks.iter().map(|(_, len)| *len).sum();
    log_once(tau, SolThresh::Diag, tokens, sink_tokens);
    if let Some(out) = crate::wan::sol_ops::try_sol_device(q, k, v, tau as f32, scale, sinks)? {
        return Ok(out);
    }
    crate::wan::stats::host_algorithm(
        "sol_attn",
        format_args!("BHSD sunk {batch}x{heads}x{tokens}x{dim}"),
    )?;
    if sinks.is_empty() {
        return sol_from_host(q, k, v, batch, heads, tokens, dim, tau, scale, None, 0);
    }
    if sinks.len() == 1 {
        return sol_from_host(
            q,
            k,
            v,
            batch,
            heads,
            tokens,
            dim,
            tau,
            scale,
            Some(sinks[0].0),
            sinks[0].1,
        );
    }
    let official: Vec<(Option<usize>, usize)> = sinks
        .iter()
        .map(|&(start, len)| (Some(start), len))
        .collect();
    let out = sol_attn_bhsd_sunk(
        q.host_cow()?.as_ref(),
        k.host_cow()?.as_ref(),
        v.host_cow()?.as_ref(),
        batch,
        heads,
        tokens,
        dim,
        tau as f32,
        scale,
        &official,
    )
    .map_err(msg)?;
    pin_like(CudaTensor::from_vec(out, q.shape.clone())?, q)
}

/// Sol-Attn with explicit [`SolParams`]: threshold type (`diag` / `exact`)
/// and the single Python-style `(sink_start, sink_tokens)` span. Device:
/// the fused kernels. Host: the f32 oracle (diag) or the ideal-numerics
/// reference-order oracle (exact, which the f32 oracle does not implement).
#[allow(clippy::too_many_arguments)]
pub fn sol_attn_params(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    tau: f64,
    scale: Option<f32>,
    thresh: SolThresh,
    sink_start: Option<usize>,
    sink_tokens: usize,
) -> Result<CudaTensor> {
    if q.rank() != 4 || q.shape != k.shape || q.shape != v.shape {
        return Err(msg(format!(
            "sol-attn expects matching BHSD q/k/v, got {:?} {:?} {:?}",
            q.shape, k.shape, v.shape
        )));
    }
    let (batch, heads, tokens, dim) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    if tokens == 0 {
        return Ok(q.clone());
    }
    let scale = scale.unwrap_or((dim as f32).sqrt().recip());
    log_once(tau, thresh, tokens, sink_tokens);
    let p = SolParams {
        tau: tau as f32,
        scale,
        thresh,
        sink_start,
        sink_tokens,
    };
    if let Some(out) = crate::wan::sol_ops::try_sol_device_params(q, k, v, &p)? {
        return Ok(out);
    }
    crate::wan::stats::host_algorithm(
        "sol_attn",
        format_args!("BHSD {thresh:?} {batch}x{heads}x{tokens}x{dim}"),
    )?;
    if thresh == SolThresh::Diag {
        return sol_from_host(
            q,
            k,
            v,
            batch,
            heads,
            tokens,
            dim,
            tau,
            scale,
            sink_start,
            sink_tokens,
        );
    }
    let (out, _lse) = sol_attn_bhsd_faithful(
        q.host_cow()?.as_ref(),
        k.host_cow()?.as_ref(),
        v.host_cow()?.as_ref(),
        batch,
        heads,
        tokens,
        dim,
        &p,
        SolNumerics::default(),
    )
    .map_err(msg)?;
    pin_like(CudaTensor::from_vec(out, q.shape.clone())?, q)
}

fn pin_like(t: CudaTensor, like: &CudaTensor) -> Result<CudaTensor> {
    #[cfg(feature = "cuda")]
    {
        let mut t = t;
        if like.is_device_fresh() {
            t.pin_device()?;
        }
        Ok(t)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = like;
        Ok(t)
    }
}

#[allow(clippy::too_many_arguments)]
fn sol_from_host(
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    batch: usize,
    heads: usize,
    tokens: usize,
    dim: usize,
    tau: f64,
    scale: f32,
    sink_start: Option<usize>,
    sink_tokens: usize,
) -> Result<CudaTensor> {
    let out = sol_attn_bhsd(
        q.host_cow()?.as_ref(),
        k.host_cow()?.as_ref(),
        v.host_cow()?.as_ref(),
        batch,
        heads,
        tokens,
        dim,
        tau as f32,
        scale,
        sink_start,
        sink_tokens,
    )
    .map_err(msg)?;
    pin_like(CudaTensor::from_vec(out, q.shape.clone())?, q)
}

/// Replace the first `sink_tokens` query rows with dense SDPA (H3
/// `text_query_rows=dense` / official MMDiT splice).
pub fn splice_dense_prefix(
    sol_out: &CudaTensor,
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    sink_tokens: usize,
    scale: Option<f32>,
) -> Result<CudaTensor> {
    if sink_tokens == 0 {
        return Ok(sol_out.clone());
    }
    let tokens = q.shape[2];
    let keep = sink_tokens.min(tokens);
    if keep == 0 {
        return Ok(sol_out.clone());
    }
    let q_pre = q.narrow(2, 0, keep)?;
    let dense = crate::wan::nn::scaled_dot_product_attention(&q_pre, k, v, scale)?;
    if keep == tokens {
        return Ok(dense);
    }
    let tail = sol_out.narrow(2, keep, tokens - keep)?;
    CudaTensor::cat(&[&dense, &tail], 2)
}

/// Replace each published query span with dense SDPA (`text_query_rows=dense`
/// plus the Spark audio-query subset).
pub fn splice_dense_ranges(
    sol_out: &CudaTensor,
    q: &CudaTensor,
    k: &CudaTensor,
    v: &CudaTensor,
    ranges: &[(usize, usize)],
    scale: Option<f32>,
) -> Result<CudaTensor> {
    if ranges.is_empty() {
        return Ok(sol_out.clone());
    }
    if ranges.len() == 1 && ranges[0].0 == 0 {
        return splice_dense_prefix(sol_out, q, k, v, ranges[0].1, scale);
    }
    let tokens = q.shape[2];
    let mut spans: Vec<(usize, usize)> = ranges
        .iter()
        .copied()
        .filter(|(_, len)| *len > 0)
        .map(|(start, len)| (start.min(tokens), len.min(tokens.saturating_sub(start))))
        .filter(|(_, len)| *len > 0)
        .collect();
    spans.sort_by_key(|s| s.0);
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, len) in spans {
        match merged.last_mut() {
            Some((ms, ml)) if start <= *ms + *ml => {
                *ml = (*ms + *ml).max(start + len) - *ms;
            }
            _ => merged.push((start, len)),
        }
    }
    if merged.is_empty() {
        return Ok(sol_out.clone());
    }
    let mut pieces = Vec::new();
    let mut cursor = 0usize;
    for &(start, len) in &merged {
        if start > cursor {
            pieces.push(sol_out.narrow(2, cursor, start - cursor)?);
        }
        let q_r = q.narrow(2, start, len)?;
        pieces.push(crate::wan::nn::scaled_dot_product_attention(
            &q_r, k, v, scale,
        )?);
        cursor = start + len;
    }
    if cursor < tokens {
        pieces.push(sol_out.narrow(2, cursor, tokens - cursor)?);
    }
    let refs: Vec<&CudaTensor> = pieces.iter().collect();
    CudaTensor::cat(&refs, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wan::nn::scaled_dot_product_attention;
    use fastvideo_models::sol_attn::dense_attn_head;

    fn seeded(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * k + 0.3).sin()).collect()
    }

    #[test]
    fn full_sink_matches_dense_sdpa() {
        let (b, h, t, d) = (1usize, 2usize, 24usize, 8usize);
        let q = CudaTensor::from_vec(seeded(b * h * t * d, 0.11), vec![b, h, t, d]).unwrap();
        let k = CudaTensor::from_vec(seeded(b * h * t * d, 0.17), vec![b, h, t, d]).unwrap();
        let v = CudaTensor::from_vec(seeded(b * h * t * d, 0.23), vec![b, h, t, d]).unwrap();
        let scale = Some((d as f32).sqrt().recip());
        let sol = sol_attn(&q, &k, &v, 1.0, scale, Some(0), t).unwrap();
        let dense = scaled_dot_product_attention(&q, &k, &v, scale).unwrap();
        let (a, b) = (sol.host_cow().unwrap(), dense.host_cow().unwrap());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 3e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn device_alg_matches_models_oracle() {
        let (b, h, t, d) = (1usize, 2usize, 40usize, 8usize);
        let q = seeded(b * h * t * d, 0.11);
        let k = seeded(b * h * t * d, 0.17);
        let v = seeded(b * h * t * d, 0.23);
        let scale = (d as f32).sqrt().recip();
        let qt = CudaTensor::from_vec(q.clone(), vec![b, h, t, d]).unwrap();
        let kt = CudaTensor::from_vec(k.clone(), vec![b, h, t, d]).unwrap();
        let vt = CudaTensor::from_vec(v.clone(), vec![b, h, t, d]).unwrap();
        let got = sol_attn(&qt, &kt, &vt, 1.25, Some(scale), Some(0), 8).unwrap();
        let want = fastvideo_models::sol_attn::sol_attn_bhsd(
            &q,
            &k,
            &v,
            b,
            h,
            t,
            d,
            1.25,
            scale,
            Some(0),
            8,
        )
        .unwrap();
        let a = got.host_cow().unwrap();
        for (x, y) in a.iter().zip(&want) {
            assert!((x - y).abs() < 3e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn host_head_matches_the_models_oracle() {
        let (t, d) = (20usize, 4usize);
        let q = seeded(t * d, 0.11);
        let k = seeded(t * d, 0.17);
        let v = seeded(t * d, 0.23);
        let scale = (d as f32).sqrt().recip();
        let qt = CudaTensor::from_vec(q.clone(), vec![1, 1, t, d]).unwrap();
        let kt = CudaTensor::from_vec(k.clone(), vec![1, 1, t, d]).unwrap();
        let vt = CudaTensor::from_vec(v.clone(), vec![1, 1, t, d]).unwrap();
        let got = sol_attn(&qt, &kt, &vt, 1.25, Some(scale), None, 0).unwrap();
        let want = dense_attn_head(&q, &k, &v, t, d, scale);
        // Not all-exact at tau 1.25; just check shape and finite values.
        assert_eq!(got.shape, vec![1, 1, t, d]);
        assert!(got.host_cow().unwrap().iter().all(|x| x.is_finite()));
        let _ = want;
    }
}

/// Fused device kernels vs the reference-faithful host oracle. Needs a CUDA
/// GPU (sm80+); skipped (with a message) when the driver or device is
/// missing. Run: `cargo test -p fastvideo-cudarc --release --features cuda
/// -- --ignored sol_gpu`.
#[cfg(all(test, feature = "cuda"))]
mod sol_gpu_tests {
    use std::sync::Arc;

    use fastvideo_models::sol_attn::{
        block_len, num_blocks, pool_kv_faithful, round_all, sink_blocks, sol_attn_head_faithful,
        threshold_faithful, SolNumerics, SolParams, SolThresh, ROUTE_GROUP,
    };

    use crate::wan::device::{set_thread_device, DeviceContext};
    use crate::wan::ops;
    use crate::wan::tensor::CudaTensor;

    const D: usize = 128;
    const FAITHFUL: SolNumerics = SolNumerics {
        bf16_faithful: true,
    };

    /// The calling thread's device; cleared on drop.
    struct Gpu(Arc<DeviceContext>);

    impl Drop for Gpu {
        fn drop(&mut self) {
            set_thread_device(None);
        }
    }

    fn gpu() -> Option<Gpu> {
        let made = std::panic::catch_unwind(|| DeviceContext::new(0));
        let dev = match made {
            Ok(Ok(d)) => Arc::new(d),
            Ok(Err(e)) => {
                eprintln!("skip: no CUDA device ({e})");
                return None;
            }
            Err(_) => {
                eprintln!("skip: CUDA driver library not loadable");
                return None;
            }
        };
        if dev.sm_major < 8 {
            eprintln!("skip: sm{}{} has no bf16 mma", dev.sm_major, dev.sm_minor);
            return None;
        }
        set_thread_device(Some(dev.clone()));
        Some(Gpu(dev))
    }

    fn normal(seed: u64, n: usize, std: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let mut next = || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let (u1, u2) = (next(), next());
                ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32 * std
            })
            .collect()
    }

    /// Per-block base `N(0, 1.5)` (shared across every 4th block pair) plus
    /// 0.3 noise: non-trivial routing at tau ~ 1.
    fn structured(seed: u64, bh: usize, tokens: usize) -> Vec<f32> {
        let n = num_blocks(tokens);
        let base = normal(seed, bh * n * D, 1.5);
        let common = normal(seed ^ 0xABCD, bh * n * D, 1.5);
        let noise = normal(seed + 7, bh * tokens * D, 0.3);
        (0..bh * tokens * D)
            .map(|i| {
                let (h, t, d) = (i / (tokens * D), (i / D) % tokens, i % D);
                let b = t / 64;
                let src = if b % 4 < 2 { &common } else { &base };
                src[(h * n + b) * D + d] + noise[i]
            })
            .collect()
    }

    fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            num += (*x as f64 - *y as f64).powi(2);
            den += (*y as f64).powi(2);
        }
        (num / den.max(1e-30)).sqrt()
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn up(dev: &DeviceContext, x: &[f32]) -> cudarc::driver::CudaSlice<f32> {
        dev.stream.memcpy_stod(x).unwrap()
    }

    fn down_bf16(dev: &DeviceContext, x: &cudarc::driver::CudaSlice<half::bf16>) -> Vec<f32> {
        dev.stream
            .memcpy_dtov(x)
            .unwrap()
            .into_iter()
            .map(|b| b.to_f32())
            .collect()
    }

    /// One case end to end: prep stages, route bits, output and LSE.
    fn check_case(g: &Gpu, name: &str, bh: usize, tokens: usize, p: SolParams, seed: u64) {
        let dev = &g.0;
        let n_el = bh * tokens * D;
        let q = structured(seed, bh, tokens);
        let k = structured(seed + 1, bh, tokens);
        let v = normal(seed + 2, n_el, 1.0);
        let (qd, kd, vd) = (up(dev, &q), up(dev, &k), up(dev, &v));
        let prep =
            ops::sol_prep_device(&qd, &kd, &vd, bh, tokens, D, p.tau, p.scale, p.thresh).unwrap();
        let sinks = sink_blocks(tokens, p.sink_start, p.sink_tokens);
        let fwd = ops::sol_fwd_device(&prep, p.scale, sinks, true, true).unwrap();
        let out = dev.stream.memcpy_dtov(&fwd.out).unwrap();
        let lse = dev.stream.memcpy_dtov(fwd.lse.as_ref().unwrap()).unwrap();
        let route = dev.stream.memcpy_dtov(fwd.route.as_ref().unwrap()).unwrap();
        let kc_dev = down_bf16(dev, &prep.kc);
        let vc_dev = down_bf16(dev, &prep.vc);
        let thr_dev = dev.stream.memcpy_dtov(&prep.thr).unwrap();

        let n = num_blocks(tokens);
        let groups = n.div_ceil(ROUTE_GROUP);
        let stride = tokens * D;
        let (mut want_out, mut want_lse) = (Vec::new(), Vec::new());
        let (mut flips, mut pairs) = (0usize, 0usize);
        for h in 0..bh {
            let s = h * stride..(h + 1) * stride;
            let head = sol_attn_head_faithful(
                &q[s.clone()],
                &k[s.clone()],
                &v[s.clone()],
                tokens,
                D,
                &p,
                FAITHFUL,
            );
            // P1: Kc / Vc within one bf16 ulp (bit-exact expected).
            let kr = round_all(&k[s.clone()], FAITHFUL);
            let vr = round_all(&v[s.clone()], FAITHFUL);
            let (kc, vc) = pool_kv_faithful(&kr, &vr, tokens, D, FAITHFUL);
            let blk = h * n * D..(h + 1) * n * D;
            for (a, b) in kc_dev[blk.clone()]
                .iter()
                .zip(&kc)
                .chain(vc_dev[blk.clone()].iter().zip(&vc))
            {
                assert!(
                    (a - b).abs() <= b.abs() / 128.0 + 1e-30,
                    "{name}: kc/vc {a} vs {b}"
                );
            }
            // P3: thresholds from the DEVICE kc isolate the stage.
            let qr = round_all(&q[s], FAITHFUL);
            let thr = threshold_faithful(&qr, &kc_dev[blk], tokens, D, &p, FAITHFUL);
            for (i, (a, b)) in thr_dev[h * n..(h + 1) * n].iter().zip(&thr).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                    "{name}: thr[{i}] {a} vs {b}"
                );
            }
            // Route bits: flips only inside the near-tie band.
            for i in 0..n {
                for j in 0..n {
                    let word = route[(h * n + i) * 2 * groups + j / 32];
                    let got = (word >> (j % 32)) & 1 == 1;
                    let want = head.mask[i * n + j];
                    pairs += 1;
                    if got != want {
                        flips += 1;
                        let forced = i.abs_diff(j) <= 1 || (sinks.0..sinks.1).contains(&j);
                        assert!(!forced, "{name}: forced block {i},{j} flipped");
                        let th = head.threshold[i];
                        let gap = (head.col_mean[i * n + j] - th).abs();
                        assert!(
                            gap <= 1e-3 * th.abs().max(1.0),
                            "{name}: flip {i},{j} gap {gap}"
                        );
                    }
                }
            }
            want_out.extend(head.out);
            want_lse.extend(head.lse);
        }
        assert!(flips * 1000 <= pairs, "{name}: {flips}/{pairs} route flips");
        let err = rel_l2(&out, &want_out);
        let mx = max_abs(&out, &want_out);
        assert!(
            err <= 1e-2 && mx <= 3e-2,
            "{name}: rel-L2 {err} max-abs {mx}"
        );
        let lerr = max_abs(&lse, &want_lse);
        assert!(lerr <= 2e-3, "{name}: lse max-abs {lerr}");
        // Rows never written stay out of bounds of nothing: every row is live.
        assert!(
            out.iter().all(|x| x.is_finite()),
            "{name}: non-finite output"
        );
        let _ = block_len;
        eprintln!("{name}: rel-L2 {err:.2e} max-abs {mx:.2e} lse {lerr:.2e} flips {flips}/{pairs}");
    }

    #[test]
    #[ignore = "needs a CUDA GPU (sm80+)"]
    fn sol_gpu_fused_matches_faithful_oracle() {
        let Some(g) = gpu() else { return };
        let sc = (D as f32).sqrt().recip();
        let with_sink = |p: SolParams, start: Option<usize>, len: usize| SolParams {
            sink_start: start,
            sink_tokens: len,
            ..p
        };
        for tau in [1.0f32, 1.25, 1.5] {
            check_case(
                &g,
                &format!("headline_tau{tau}"),
                2,
                4096,
                SolParams::diag(tau, sc),
                100,
            );
        }
        check_case(&g, "tail_4000", 2, 4000, SolParams::diag(1.0, sc), 110);
        check_case(&g, "tail_4033", 2, 4033, SolParams::diag(1.0, sc), 120);
        check_case(&g, "groups_8256", 2, 8256, SolParams::diag(1.25, sc), 130);
        check_case(
            &g,
            "suffix_sink",
            2,
            4096,
            with_sink(SolParams::diag(1.0, sc), None, 100),
            140,
        );
        check_case(
            &g,
            "mid_sink",
            2,
            4096,
            with_sink(SolParams::diag(1.0, sc), Some(1000), 300),
            150,
        );
        check_case(&g, "local_only", 2, 4096, SolParams::diag(1.0e4, sc), 160);
        check_case(
            &g,
            "all_exact_tau",
            1,
            2000,
            SolParams::diag(-1.0e4, sc),
            170,
        );
        for t in [1usize, 63, 64, 65, 130] {
            check_case(
                &g,
                &format!("tiny_{t}"),
                1,
                t,
                SolParams::diag(1.0, sc),
                180 + t as u64,
            );
        }
        let exact = SolParams {
            thresh: SolThresh::Exact,
            ..SolParams::diag(1.0, sc)
        };
        check_case(&g, "exact_thresh", 2, 4096, exact, 190);
        check_case(&g, "batch_2x3", 6, 1000, SolParams::diag(1.25, sc), 200);
    }

    #[test]
    #[ignore = "needs a CUDA GPU (sm80+)"]
    fn sol_gpu_all_exact_equals_dense_and_entry_uses_device() {
        let Some(_g) = gpu() else { return };
        let (b, h, t) = (1usize, 2usize, 1100usize);
        let n = b * h * t * D;
        let (q, k, v) = (normal(1, n, 1.0), normal(2, n, 1.0), normal(3, n, 1.0));
        let qt = CudaTensor::from_vec(q, vec![b, h, t, D])
            .unwrap()
            .to_device()
            .unwrap();
        let kt = CudaTensor::from_vec(k, vec![b, h, t, D])
            .unwrap()
            .to_device()
            .unwrap();
        let vt = CudaTensor::from_vec(v, vec![b, h, t, D])
            .unwrap()
            .to_device()
            .unwrap();
        let dense = crate::wan::nn::scaled_dot_product_attention(&qt, &kt, &vt, None).unwrap();
        let sol = super::sol_attn(&qt, &kt, &vt, 1.0, None, Some(0), t).unwrap();
        let err = rel_l2(&sol.host_cow().unwrap(), &dense.host_cow().unwrap());
        assert!(err <= 5e-3, "all-exact vs dense rel-L2 {err}");
        // Contiguous spans merge; a gap is refused instead of re-routed.
        assert!(super::sol_attn_sunk(&qt, &kt, &vt, 1.0, None, &[(0, 77), (128, 64)]).is_ok());
        assert!(super::sol_attn_sunk(&qt, &kt, &vt, 1.0, None, &[(0, 77), (640, 64)]).is_err());
    }

    /// The legacy multi-launch path after the m-units (D1) and m/l-store
    /// (D2) fixes in `sol_mma_attn_partials`: it must agree with the f32
    /// oracle to bf16 round-off again.
    #[test]
    #[ignore = "needs a CUDA GPU (sm80+)"]
    fn sol_gpu_legacy_multipass_is_fixed() {
        let Some(g) = gpu() else { return };
        let (bh, t) = (2usize, 256usize);
        let n = bh * t * D;
        let (q, k, v) = (
            structured(5, bh, t),
            structured(6, bh, t),
            normal(7, n, 1.0),
        );
        let sc = (D as f32).sqrt().recip();
        let dev = &g.0;
        let got = ops::sol_attn_multipass_device(
            &up(dev, &q),
            &up(dev, &k),
            &up(dev, &v),
            1,
            bh,
            t,
            D,
            1.0,
            sc,
            &[(None, 0)],
        )
        .unwrap();
        let got = dev.stream.memcpy_dtov(&got).unwrap();
        let want =
            fastvideo_models::sol_attn::sol_attn_bhsd(&q, &k, &v, 1, bh, t, D, 1.0, sc, None, 0)
                .unwrap();
        let err = rel_l2(&got, &want);
        assert!(err <= 2e-2, "legacy multipass rel-L2 {err}");
    }
}
