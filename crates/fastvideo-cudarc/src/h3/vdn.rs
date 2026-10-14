//! VDN-H3 hybrid attention on the H3 DiT (docs/ports/vdn-h3.md).
//!
//! Each of the 50 blocks keeps H3's attention projections (with the VDN LoRA
//! merged at load) and adds:
//!
//! * a **window softmax** in place of dense attention
//!   ([`crate::wan::nn::sdpa_vdn_window`], one flash launch,
//!   `flash_mma_fwd2_window_d128`), its output scaled per (row, head) by the
//!   softmax gate before `to_out`;
//! * the **linear branch** ([`VdnBranch`]) on the inner video frames, added
//!   through `to_out_linear` to the attention output's video rows.
//!
//! The branch runs as the kernels of `vdn.cu` on the device (bf16
//! activations, fp32 statistics / solve / scans) and as the host reference
//! of [`fastvideo_models::h3::vdn`] otherwise (CPU tests). Both round where
//! the reference's inference body rounds; the device kernels are checked
//! against the host functions stage by stage by `fv-gpucheck kernels
//! --groups vdn`.

use std::sync::Arc;

use fastvideo_models::h3::config::H3TransformerConfig;
use fastvideo_models::h3::packing::H3PackedLayout;
use fastvideo_models::h3::vdn::{
    self as vref, gather_plan, inner_bounds, VdnLayout, VdnTransform, VDN_TEXT_STATE_SCALE,
};

use crate::wan::attn::VdnWindow;
use crate::wan::nn::Linear;
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::WeightMap;

// `launch!` names `super::stats` / `super::device` from its call site (`dev`).
#[cfg(feature = "cuda")]
use crate::wan::{device, stats};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// The request's packed layout as the hybrid layers see it, built once per
/// generation and shared by every block of every step.
pub struct VdnPlan {
    pub layout: VdnLayout,
    pub transform: VdnTransform,
    pub window: VdnWindow,
    /// Every frame's window covers the clip: dense softmax, no branch.
    pub full_cover: bool,
    /// Inner frames (the branch's frames, `F - 2`; 0 when `F <= 2`).
    pub inner_frames: usize,
    /// `[inner, 4]`: before, after (-1 = the text state), before_from, after_to.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    plan_rows: Vec<i32>,
    #[cfg(feature = "cuda")]
    plan_dev: std::sync::OnceLock<Arc<cudarc::driver::CudaSlice<i32>>>,
}

impl VdnPlan {
    pub fn new(layout: &H3PackedLayout, transform: &VdnTransform) -> Result<Self> {
        let l = VdnLayout::from_packed(layout).map_err(msg)?;
        let bounds = transform.bounds(l.num_frames);
        let inner = inner_bounds(&bounds);
        let plan_rows = gather_plan(&inner)
            .iter()
            .flat_map(|r| {
                [
                    r.before.map_or(-1, |b| b as i32),
                    r.after.map_or(-1, |a| a as i32),
                    r.before_from as i32,
                    r.after_to as i32,
                ]
            })
            .collect();
        Ok(Self {
            window: VdnWindow {
                video_start: l.video_start,
                frame_tokens: l.tokens_per_frame,
                frames: l.num_frames,
                chunk: transform.chunk,
                radius: transform.radius,
            },
            full_cover: transform.full_cover(l.num_frames),
            inner_frames: l.num_frames.saturating_sub(2),
            layout: l,
            transform: transform.clone(),
            plan_rows,
            #[cfg(feature = "cuda")]
            plan_dev: std::sync::OnceLock::new(),
        })
    }

    /// Whether the linear branch runs (the window does not cover the clip
    /// and there are inner frames).
    pub fn branch_active(&self) -> bool {
        !self.full_cover && self.inner_frames > 0
    }

    pub fn inner_bounds(&self) -> Vec<(i64, i64)> {
        inner_bounds(&self.transform.bounds(self.layout.num_frames))
    }
}

/// The aux projection's column layout: `[beta (H) | output_gate.down (R) |
/// softmax_gate.up (H)]`, one GEMM over the attention input.
#[derive(Clone, Copy, Debug)]
pub struct AuxCols {
    pub beta: usize,
    pub og: usize,
    pub sg: usize,
    pub width: usize,
}

/// The small branch tensors: bf16 values (kept as f32 on the host), the
/// convolution weights transposed tap-major (`[25, C]`, `[5, C]`).
#[derive(Clone)]
pub(crate) struct Small {
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    alpha_down: Vec<f32>,
    alpha_up: Vec<f32>,
    norm: Vec<f32>,
    /// `[C, 25]` / `[C, 5]` as stored (host reference layout).
    k_sp: Vec<f32>,
    k_tm: Vec<f32>,
    v_sp: Vec<f32>,
    v_tm: Vec<f32>,
}

/// One block's linear branch and gates (`transformer_blocks.N.attn.*` of
/// `linear_branch/model.safetensors`, minus `orig.*`).
#[derive(Clone)]
pub struct VdnBranch {
    aux: Linear,
    og_up: Linear,
    pub(crate) to_out_linear: Linear,
    small: Arc<Small>,
    cols: AuxCols,
    heads: usize,
    hidden: usize,
    rank: usize,
    #[cfg(feature = "cuda")]
    dev: Option<Arc<dev::DevSmall>>,
}

impl VdnBranch {
    /// `prefix` = `transformer_blocks.N.attn`.
    pub fn load(map: &WeightMap, prefix: &str, cfg: &H3TransformerConfig) -> Result<Self> {
        let (hidden, heads, d) = (cfg.hidden_size, cfg.num_attention_heads, cfg.attention_head_dim);
        let c = heads * d;
        let la = format!("{prefix}.linear_attention");
        let get = |key: &str, shape: &[usize]| -> Result<Vec<f32>> {
            Ok(crate::wan::weights::cuda_tensor_shaped(map, key, shape)?
                .host_cow()?
                .into_owned())
        };
        // The output gate's and the decay's bottleneck is the head dim
        // (`OutputGate(bottleneck=head_dim)`, `FrameKDAAlpha` rank = head_dim).
        let rank = d;
        let beta = get(&format!("{la}.beta_proj.weight"), &[heads, hidden])?;
        let og_down = get(&format!("{la}.output_gate.down.weight"), &[rank, hidden])?;
        let sg = get(&format!("{prefix}.softmax_gate.up.weight"), &[heads, hidden])?;
        let sg_bias = get(&format!("{prefix}.softmax_gate.up.bias"), &[heads])?;
        let cols = AuxCols {
            beta: 0,
            og: heads,
            sg: heads + rank,
            width: 2 * heads + rank,
        };
        let mut w = Vec::with_capacity(cols.width * hidden);
        w.extend_from_slice(&beta);
        w.extend_from_slice(&og_down);
        w.extend_from_slice(&sg);
        let mut bias = vec![0.0f32; heads + rank];
        bias.extend_from_slice(&sg_bias);
        let aux = Linear::from_tensors(
            CudaTensor::from_vec(w, vec![cols.width, hidden])?,
            Some(CudaTensor::from_vec(bias, vec![cols.width])?),
        )?;
        let og_up = Linear::load(map, &format!("{la}.output_gate.up"), rank, c, true)?;
        let to_out_linear = Linear::load(map, &format!("{prefix}.to_out_linear"), c, hidden, false)?;
        let flat = |key: &str, shape: &[usize]| get(key, shape);
        let small = Small {
            a_log: flat(&format!("{la}.alpha.A_log"), &[heads])?,
            dt_bias: flat(&format!("{la}.alpha.dt_bias"), &[c])?,
            alpha_down: flat(&format!("{la}.alpha.down.weight"), &[d, hidden])?,
            alpha_up: flat(&format!("{la}.alpha.up.weight"), &[c, d])?,
            norm: flat(&format!("{la}.norm.weight"), &[d])?,
            k_sp: flat(&format!("{la}.short_conv.k_sp.weight"), &[c, 1, 5, 5])?,
            k_tm: flat(&format!("{la}.short_conv.k_tm.weight"), &[c, 1, 5])?,
            v_sp: flat(&format!("{la}.short_conv.v_sp.weight"), &[c, 1, 5, 5])?,
            v_tm: flat(&format!("{la}.short_conv.v_tm.weight"), &[c, 1, 5])?,
        };
        // The reference casts every hybrid parameter to bf16 after loading.
        let small = small.rounded();
        #[cfg(feature = "cuda")]
        let dev = if crate::wan::stats::device_expected() && d == vref::VDN_HEAD_DIM {
            Some(Arc::new(dev::DevSmall::upload(&small, heads)?))
        } else {
            None
        };
        Ok(Self {
            aux,
            og_up,
            to_out_linear,
            small: Arc::new(small),
            cols,
            heads,
            hidden,
            rank,
            #[cfg(feature = "cuda")]
            dev,
        })
    }

    pub fn cols(&self) -> AuxCols {
        self.cols
    }

    /// `[1, S, width]`: beta / output-gate down / softmax-gate logits of every row.
    pub fn aux(&self, n: &CudaTensor) -> Result<CudaTensor> {
        self.aux.forward(n)
    }

    /// The softmax gate on the window attention output `o` (`[1, H, S, D]`),
    /// in the reference's rounding: `bf16(o * bf16(sigmoid(logit)))`.
    pub fn gate_heads(&self, o: CudaTensor, aux: &CudaTensor) -> Result<CudaTensor> {
        #[cfg(feature = "cuda")]
        if let Some(out) = dev::gate_heads(&o, aux, self.cols, self.heads)? {
            return Ok(out);
        }
        let [1, h, s, d] = o.shape[..] else {
            return Err(msg(format!("vdn gate: attention output {:?}", o.shape)));
        };
        let bf = crate::wan::tensor::bf16_activations();
        let a = aux.host_cow()?;
        let mut v = o.host_cow()?.into_owned();
        for hh in 0..h {
            for ss in 0..s {
                let mut g = sigmoid(a[ss * self.cols.width + self.cols.sg + hh]);
                if bf {
                    g = round_bf16(g);
                }
                for e in &mut v[(hh * s + ss) * d..][..d] {
                    *e *= g;
                    if bf {
                        *e = round_bf16(*e);
                    }
                }
            }
        }
        CudaTensor::from_vec(v, o.shape.clone())
    }

    /// The branch readout of the inner frames, through `to_out_linear`:
    /// `[inner rows, hidden]`. `n`: the attention input `[1, S, hidden]`;
    /// `packed`: the raw projections `[1, S, 3C(+…)]` (q, k, v at columns 0,
    /// C, 2C); `aux`: [`Self::aux`] of `n`.
    pub fn forward(
        &self,
        n: &CudaTensor,
        packed: &CudaTensor,
        aux: &CudaTensor,
        plan: &VdnPlan,
    ) -> Result<CudaTensor> {
        let readout = {
            #[cfg(feature = "cuda")]
            {
                match &self.dev {
                    Some(d) => dev::branch(self, d, n, packed, aux, plan)?,
                    None => self.branch_host(n, packed, aux, plan)?,
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                self.branch_host(n, packed, aux, plan)?
            }
        };
        self.to_out_linear.forward(&readout)
    }

    /// The branch readout before `to_out_linear` (`[inner rows, C]`), on the
    /// device kernels or (`host`) the reference functions: the end-to-end
    /// check of `fv-gpucheck kernels --groups vdn`.
    pub fn readout(
        &self,
        n: &CudaTensor,
        packed: &CudaTensor,
        aux: &CudaTensor,
        plan: &VdnPlan,
        host: bool,
    ) -> Result<CudaTensor> {
        #[cfg(feature = "cuda")]
        if !host {
            if let Some(d) = &self.dev {
                return dev::branch(self, d, n, packed, aux, plan);
            }
        }
        let _ = host;
        self.branch_host(n, packed, aux, plan)
    }

    /// The branch on the host, with the reference functions (f32/f64; bf16
    /// roundings of the device path are not reproduced).
    fn branch_host(
        &self,
        n: &CudaTensor,
        packed: &CudaTensor,
        aux: &CudaTensor,
        plan: &VdnPlan,
    ) -> Result<CudaTensor> {
        let l = &plan.layout;
        let (heads, hidden) = (self.heads, self.hidden);
        let d = self.small.norm.len();
        let c = heads * d;
        let width = packed.shape[packed.rank() - 1];
        let (r0, rows) = l.inner_rows();
        let frames = plan.inner_frames;
        let s = l.tokens_per_frame;
        let x = n.host_cow()?;
        let p = packed.host_cow()?;
        let a = aux.host_cow()?;
        let cols = |row0: usize, rows: usize, off: usize, w: usize, stride: usize, src: &[f32]| -> Vec<f32> {
            (0..rows)
                .flat_map(|r| src[(row0 + r) * stride + off..][..w].iter().copied())
                .collect()
        };
        let q_raw = cols(r0, rows, 0, c, width, &p);
        let k_raw = cols(r0, rows, c, c, width, &p);
        let v_raw = cols(r0, rows, 2 * c, c, width, &p);
        let xin = cols(r0, rows, 0, hidden, hidden, &x);
        let (t0, tl) = (l.text_start, l.text_len);
        let k_text = cols(t0, tl, c, c, width, &p);
        let v_text = cols(t0, tl, 2 * c, c, width, &p);
        let w = &self.small;
        let q = vref::activate(&q_raw, heads, d, true);
        let k = vref::activate(
            &vref::temporal_conv(
                &vref::spatial_conv(&k_raw, frames, l.grid_h, l.grid_w, c, &w.k_sp),
                frames,
                s,
                c,
                &w.k_tm,
            ),
            heads,
            d,
            true,
        );
        let v = vref::activate(
            &vref::temporal_conv(
                &vref::spatial_conv(&v_raw, frames, l.grid_h, l.grid_w, c, &w.v_sp),
                frames,
                s,
                c,
                &w.v_tm,
            ),
            heads,
            d,
            false,
        );
        let beta_of = |row0: usize, rows: usize| -> Vec<f32> {
            cols(row0, rows, self.cols.beta, heads, self.cols.width, &a)
                .into_iter()
                .map(sigmoid)
                .collect()
        };
        let beta = beta_of(r0, rows);
        let (am, bm) = vref::frame_statistics(&k, &v, &beta, frames, s, heads, d);
        let mut mean = vec![0.0f32; frames * hidden];
        for f in 0..frames {
            for col in 0..hidden {
                let sum: f64 = (0..s).map(|si| f64::from(xin[(f * s + si) * hidden + col])).sum();
                mean[f * hidden + col] = (sum / s as f64) as f32;
            }
        }
        let alpha = vref::kda_alpha(
            &mean,
            frames,
            hidden,
            &w.alpha_down,
            &w.alpha_up,
            &w.dt_bias,
            &w.a_log,
            heads,
            d,
        );
        let kt = vref::activate(&k_text, heads, d, true);
        let vt = vref::activate(&v_text, heads, d, false);
        let (at, btm) = vref::frame_statistics(&kt, &vt, &beta_of(t0, tl), 1, tl, heads, d);
        let (_, jt) = vref::vdn_factors(&at, &btm, None, heads, d).map_err(msg)?;
        let start: Vec<f32> = jt.iter().map(|&v| VDN_TEXT_STATE_SCALE * v).collect();
        let (tf, jf) = vref::vdn_factors(&am, &bm, Some(&alpha), frames * heads, d).map_err(msg)?;
        let (pb, qb) = vref::scans(&tf, &jf, &start, frames, heads, d);
        let state = vref::gather(&pb, &qb, &alpha, &start, &gather_plan(&plan.inner_bounds()), heads, d);
        let og_mid = CudaTensor::from_vec(
            cols(r0, rows, self.cols.og, self.rank, self.cols.width, &a),
            vec![rows, self.rank],
        )?;
        let gate: Vec<f32> = self
            .og_up
            .forward(&og_mid)?
            .host_cow()?
            .iter()
            .map(|&g| sigmoid(g))
            .collect();
        let out = vref::readout(&state, &q, &gate, &w.norm, frames, s, heads, d);
        CudaTensor::from_vec(out, vec![rows, c])
    }
}

impl Small {
    fn rounded(mut self) -> Self {
        for v in [
            &mut self.a_log,
            &mut self.dt_bias,
            &mut self.alpha_down,
            &mut self.alpha_up,
            &mut self.norm,
            &mut self.k_sp,
            &mut self.k_tm,
            &mut self.v_sp,
            &mut self.v_tm,
        ] {
            for x in v.iter_mut() {
                *x = round_bf16(*x);
            }
        }
        self
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn round_bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

/// Add `add` (`[n, hidden]`) into rows `row0..row0 + n` of `out`
/// (`[1, S, hidden]`), rounding the sums to bf16 as the reference's bf16
/// `out[video] += ...` does under bf16 activations.
pub fn add_rows(out: CudaTensor, add: &CudaTensor, row0: usize) -> Result<CudaTensor> {
    #[cfg(feature = "cuda")]
    if let Some(t) = dev::add_rows(&out, add, row0)? {
        return Ok(t);
    }
    let hidden = out.shape[out.rank() - 1];
    let bf = crate::wan::tensor::bf16_activations();
    let mut v = out.host_cow()?.into_owned();
    let a = add.host_cow()?;
    for (o, &x) in v[row0 * hidden..][..a.len()].iter_mut().zip(a.iter()) {
        *o += x;
        if bf {
            *o = round_bf16(*o);
        }
    }
    CudaTensor::from_vec(v, out.shape.clone())
}

/// `fv-gpucheck kernels --groups vdn`: every `vdn.cu` stage against the host
/// reference on identical (bf16-valued) inputs, the window flash kernel
/// against an f64 masked softmax (and, for a clip the window covers, against
/// the dense kernel bit for bit), the whole device branch against the host
/// branch, and timings at the 480p and 768p 5 s shapes.
#[cfg(feature = "cuda")]
pub mod check {
    use super::dev;
    use super::*;
    use crate::wan::attn::{device_mma_sdpa_window, device_mma_sdpa_with, FlashKernel};
    use cudarc::driver::CudaSlice;
    use fastvideo_models::h3::vdn::VDN_HEAD_DIM as D;

    /// One check: `rel_l2` / `max_abs` of the device against the host.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct VdnCheck {
        pub name: String,
        pub pass: bool,
        pub rel_l2: f64,
        pub max_abs: f64,
        pub tol: f64,
        pub note: String,
    }

    /// Timings (ms) of one call at a real shape.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct VdnTiming {
        pub name: String,
        pub ms: f64,
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        }
        /// bf16-valued uniform `[-scale, scale)`.
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| bf(self.next() * scale)).collect()
        }
    }

    fn bf(x: f32) -> f32 {
        half::bf16::from_f32(x).to_f32()
    }

    fn sig(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn dev_ctx() -> Result<std::sync::Arc<crate::wan::device::DeviceContext>> {
        crate::wan::device::global_device().ok_or_else(|| msg("vdn check: no device"))
    }

    fn up16(v: &[f32]) -> Result<CudaSlice<half::bf16>> {
        let h: Vec<half::bf16> = v.iter().map(|&x| half::bf16::from_f32(x)).collect();
        dev_ctx()?.stream.memcpy_stod(&h).map_err(|e| msg(e.to_string()))
    }

    fn down16(s: &CudaSlice<half::bf16>) -> Result<Vec<f32>> {
        let h: Vec<half::bf16> = dev_ctx()?.stream.memcpy_dtov(s).map_err(|e| msg(e.to_string()))?;
        Ok(h.iter().map(|x| x.to_f32()).collect())
    }

    fn down32(s: &CudaSlice<f32>) -> Result<Vec<f32>> {
        dev_ctx()?.stream.memcpy_dtov(s).map_err(|e| msg(e.to_string()))
    }

    fn compare(name: &str, got: &[f32], want: &[f32], tol: f64, note: &str) -> VdnCheck {
        let (mut num, mut den, mut mx) = (0.0f64, 0.0f64, 0.0f64);
        let mut finite = got.len() == want.len();
        for (&g, &w) in got.iter().zip(want) {
            finite &= g.is_finite();
            let d = f64::from(g) - f64::from(w);
            num += d * d;
            den += f64::from(w) * f64::from(w);
            mx = mx.max(d.abs());
        }
        let rel = (num / den.max(1e-30)).sqrt();
        VdnCheck {
            name: name.to_string(),
            pass: finite && rel <= tol,
            rel_l2: rel,
            max_abs: mx,
            tol,
            note: note.to_string(),
        }
    }

    fn sync() -> Result<()> {
        dev_ctx()?.stream.synchronize().map_err(|e| msg(e.to_string()))
    }

    /// f64 softmax of `[h, s, d]` q / k / v under `w`'s mask.
    fn masked_ref(q: &[f32], k: &[f32], v: &[f32], h: usize, s: usize, w: &VdnWindow) -> Vec<f32> {
        let scale = 1.0 / (D as f64).sqrt();
        let mut out = vec![0.0f32; h * s * D];
        let mut p = vec![0.0f64; s];
        for hh in 0..h {
            for i in 0..s {
                let qi = &q[(hh * s + i) * D..][..D];
                let mut mx = f64::NEG_INFINITY;
                for (j, pj) in p.iter_mut().enumerate() {
                    *pj = if w.allows(i, j) {
                        let kj = &k[(hh * s + j) * D..][..D];
                        qi.iter().zip(kj).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum::<f64>() * scale
                    } else {
                        f64::NEG_INFINITY
                    };
                    mx = mx.max(*pj);
                }
                let mut z = 0.0;
                for pj in p.iter_mut() {
                    *pj = if pj.is_finite() { (*pj - mx).exp() } else { 0.0 };
                    z += *pj;
                }
                for e in 0..D {
                    let acc: f64 = (0..s).map(|j| p[j] * f64::from(v[(hh * s + j) * D + e])).sum();
                    out[(hh * s + i) * D + e] = (acc / z) as f32;
                }
            }
        }
        out
    }

    fn window_checks(rng: &mut Rng, out: &mut Vec<VdnCheck>) -> Result<()> {
        // [text 37 | audio 21 | video 14 frames x 40 tokens]: frames straddle
        // the 64-key tiles; the chunk-5 window leaves frames out of view.
        for (frames, name) in [(14usize, "window_attn_f64"), (8, "window_attn_full_cover_vs_dense")] {
            let (h, tpf, vs) = (2usize, 40usize, 58usize);
            let s = vs + frames * tpf;
            let w = VdnWindow { video_start: vs, frame_tokens: tpf, frames, chunk: 5, radius: 1 };
            let (q, k, v) = (rng.vec(h * s * D, 2.0), rng.vec(h * s * D, 2.0), rng.vec(h * s * D, 1.0));
            let t = |x: &[f32]| -> Result<CudaTensor> {
                CudaTensor::from_vec(x.to_vec(), vec![1, h, s, D])?.to_device()
            };
            let (qt, kt, vt) = (t(&q)?, t(&k)?, t(&v)?);
            let got = device_mma_sdpa_window(&qt, &kt, &vt, None, true, w)?
                .ok_or_else(|| msg("vdn check: the window kernel declined"))?;
            let got = got.host_cow()?.into_owned();
            if frames == 14 {
                out.push(compare(name, &got, &masked_ref(&q, &k, &v, h, s, &w), 1e-2, "bf16 output vs f64"));
            } else {
                let dense = device_mma_sdpa_with(&qt, &kt, &vt, None, true, FlashKernel::V2)?
                    .ok_or_else(|| msg("vdn check: the dense kernel declined"))?;
                let dense = dense.host_cow()?.into_owned();
                let same = got.iter().zip(&dense).all(|(a, b)| a.to_bits() == b.to_bits());
                let mut c = compare(name, &got, &dense, 0.0, "bitwise against the dense V2 kernel");
                c.pass = same;
                out.push(c);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn stage_checks(rng: &mut Rng, out: &mut Vec<VdnCheck>) -> Result<()> {
        let (heads, frames, gh, gw, hidden, tl) = (4usize, 6usize, 4usize, 6usize, 256usize, 10usize);
        let c = heads * D;
        let s = gh * gw;
        let n = frames * s;
        // a. q features
        let x = rng.vec(n * c, 2.0);
        let xd = up16(&x)?;
        let got = down16(&dev::act_rows(crate::wan::quant::ptr(&xd), c, 0, n, heads, true)?)?;
        out.push(compare("features_q", &got, &vref::activate(&x, heads, D, true), 1e-2, "bf16 out"));
        // b. spatial conv
        let wsp = rng.vec(c * 25, 0.4);
        let wsp_t: Vec<f32> = (0..25).flat_map(|t| (0..c).map(move |ch| (t, ch))).map(|(t, ch)| wsp[ch * 25 + t]).collect();
        let sp = dev::sconv5(crate::wan::quant::ptr(&xd), c, 0, frames, gh, gw, c, &up16(&wsp_t)?)?;
        let sp_h = down16(&sp)?;
        out.push(compare("sconv5", &sp_h, &vref::spatial_conv(&x, frames, gh, gw, c, &wsp), 1e-2, "bf16 out"));
        // c. temporal conv + SiLU + L2
        let wtm = rng.vec(c * 5, 0.5);
        let wtm_t: Vec<f32> = (0..5).flat_map(|t| (0..c).map(move |ch| (t, ch))).map(|(t, ch)| wtm[ch * 5 + t]).collect();
        let kf = dev::tconv5_act(&sp, frames, s, heads, &up16(&wtm_t)?, true)?;
        let kf_h = down16(&kf)?;
        let want = vref::activate(&vref::temporal_conv(&sp_h, frames, s, c, &wtm), heads, D, true);
        out.push(compare("tconv5_silu_l2", &kf_h, &want, 1e-2, "bf16 out"));
        // d. statistics (beta from the aux logits, bf16 rounded)
        let vf_h = rng.vec(n * c, 1.0);
        let vf = up16(&vf_h)?;
        let width = 2 * heads + D;
        let aux_h = rng.vec(n * width, 3.0);
        let auxd = up16(&aux_h)?;
        let (a, b) = dev::frame_stats(&kf, &vf, crate::wan::quant::ptr(&auxd), width, 0, frames, s, heads)?;
        let (a_h, b_h) = (down32(&a)?, down32(&b)?);
        let beta: Vec<f32> = (0..n).flat_map(|r| (0..heads).map(move |hh| (r, hh))).map(|(r, hh)| bf(sig(aux_h[r * width + hh]))).collect();
        let (a_want, _) = vref::frame_statistics(&kf_h, &vf_h, &beta, frames, s, heads, D);
        let vb: Vec<f32> = (0..n * c).map(|i| bf(vf_h[i] * beta[(i / c) * heads + (i % c) / D])).collect();
        let (_, b_want) = vref::frame_statistics(&kf_h, &vb, &vec![1.0; n * heads], frames, s, heads, D);
        let b_want: Vec<f32> = b_want.into_iter().map(bf).collect();
        let a_sym: Vec<f32> = (0..frames * heads)
            .flat_map(|m| (0..D * D).map(move |e| (m, e)))
            .map(|(m, e)| {
                let (i, j) = (e / D, e % D);
                0.5 * (a_h[m * D * D + i * D + j] + a_h[m * D * D + j * D + i])
            })
            .collect();
        out.push(compare("frame_stats_A", &a_sym, &a_want, 1e-5, "fp32"));
        out.push(compare("frame_stats_B", &b_h, &b_want, 1e-3, "bf16-rounded GEMM output"));
        // i. decay
        let xin = rng.vec(n * hidden, 1.0);
        let xind = up16(&xin)?;
        let small = Small {
            a_log: rng.vec(heads, 1.0).iter().map(|v| bf(v + 1.0)).collect(),
            dt_bias: rng.vec(c, 2.0),
            alpha_down: rng.vec(D * hidden, 0.1),
            alpha_up: rng.vec(c * D, 0.2),
            norm: rng.vec(D, 0.5).iter().map(|v| bf(v + 1.0)).collect(),
            k_sp: wsp.clone(),
            k_tm: wtm.clone(),
            v_sp: wsp.clone(),
            v_tm: wtm.clone(),
        };
        let sm = dev::DevSmall::upload(&small, heads)?;
        let alpha = dev::alpha(crate::wan::quant::ptr(&xind), 0, frames, s, hidden, heads, &sm)?;
        let alpha_h = down32(&alpha)?;
        let mut mean = vec![0.0f32; frames * hidden];
        for f in 0..frames {
            for col in 0..hidden {
                let sum: f64 = (0..s).map(|si| f64::from(xin[(f * s + si) * hidden + col])).sum();
                mean[f * hidden + col] = (sum / s as f64) as f32;
            }
        }
        let alpha_want = vref::kda_alpha(&mean, frames, hidden, &small.alpha_down, &small.alpha_up, &small.dt_bias, &small.a_log, heads, D);
        out.push(compare("kda_alpha", &alpha_h, &alpha_want, 1e-4, "fp32"));
        // e. solve (from the device statistics)
        let (t, j) = dev::solve(&a, &b, Some(&alpha), frames * heads, 1.0)?;
        let t = t.ok_or_else(|| msg("vdn check: no T"))?;
        let (t_want, j_want) = vref::vdn_factors(&a_sym, &b_h, Some(&alpha_h), frames * heads, D).map_err(msg)?;
        let (t_h, j_h) = (down32(&t)?, down32(&j)?);
        out.push(compare("solve_T", &t_h, &t_want, 1e-4, "Gauss-Jordan fp32 vs Cholesky f64"));
        out.push(compare("solve_J", &j_h, &j_want, 1e-4, "fp32 vs f64"));
        // text state: one chunk, alpha = 1, scaled by 1/2
        let ktx = rng.vec(tl * c, 1.0);
        let ktx = vref::activate(&ktx, heads, D, true).into_iter().map(bf).collect::<Vec<_>>();
        let vtx = rng.vec(tl * c, 1.0);
        let auxt = rng.vec(tl * width, 3.0);
        let (at, bt) = dev::frame_stats(&up16(&ktx)?, &up16(&vtx)?, crate::wan::quant::ptr(&up16(&auxt)?), width, 0, 1, tl, heads)?;
        let (_, start) = dev::solve(&at, &bt, None, heads, VDN_TEXT_STATE_SCALE)?;
        let (at_h, bt_h) = (down32(&at)?, down32(&bt)?);
        let at_sym: Vec<f32> = (0..heads)
            .flat_map(|m| (0..D * D).map(move |e| (m, e)))
            .map(|(m, e)| {
                let (i, jj) = (e / D, e % D);
                0.5 * (at_h[m * D * D + i * D + jj] + at_h[m * D * D + jj * D + i])
            })
            .collect();
        let (_, jt) = vref::vdn_factors(&at_sym, &bt_h, None, heads, D).map_err(msg)?;
        let start_want: Vec<f32> = jt.iter().map(|v| VDN_TEXT_STATE_SCALE * v).collect();
        let start_h = down32(&start)?;
        out.push(compare("text_state", &start_h, &start_want, 1e-4, "fp32 vs f64"));
        // f. scans
        let (p, q) = dev::scans(&t, &j, &start, frames, heads)?;
        let (p_want, q_want) = vref::scans(&t_h, &j_h, &start_h, frames, heads, D);
        out.push(compare("scan_forward", &down32(&p)?, &p_want, 1e-4, "fp32 vs f64"));
        out.push(compare("scan_reverse", &down32(&q)?, &q_want, 1e-4, "fp32 vs f64"));
        // g. gather on the chunk-5 windows of 8 frames (6 inner)
        let inner = inner_bounds(&VdnTransform::released().bounds(frames + 2));
        let plan_rows: Vec<i32> = gather_plan(&inner)
            .iter()
            .flat_map(|r| [r.before.map_or(-1, |x| x as i32), r.after.map_or(-1, |x| x as i32), r.before_from as i32, r.after_to as i32])
            .collect();
        let pl = dev_ctx()?.stream.memcpy_stod(&plan_rows).map_err(|e| msg(e.to_string()))?;
        let state = dev::gather(&p, &q, &start, &alpha, &pl, frames, heads)?;
        let state_h = down16(&state)?;
        let state_want = vref::gather(&down32(&p)?, &down32(&q)?, &alpha_h, &start_h, &gather_plan(&inner), heads, D);
        out.push(compare("gather_bridge", &state_h, &state_want, 1e-2, "bf16 out"));
        // h. readout + RMSNorm + gate
        let qf = rng.vec(n * c, 0.2);
        let og = rng.vec(n * c, 3.0);
        let ogd = up16(&og)?;
        let r = dev::readout(&state, &up16(&qf)?, crate::wan::quant::ptr(&ogd), &up16(&small.norm)?, frames, s, heads)?;
        let gate: Vec<f32> = og.iter().map(|&g| bf(sig(g))).collect();
        let r_want = vref::readout(&state_h, &qf, &gate, &small.norm, frames, s, heads, D);
        out.push(compare("readout_norm_gate", &down16(&r)?, &r_want, 1.5e-2, "bf16 GEMM output rounded"));
        Ok(())
    }

    fn generated(seed: u64) -> WeightMap {
        WeightMap::generated(move |key, shape| {
            let n: usize = shape.iter().product();
            let mut r = Rng(seed ^ key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |a, b| (a ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)));
            let scale = if key.ends_with("A_log") {
                return (0..n).map(|_| bf(1.0 + r.next())).collect();
            } else if key.ends_with("norm.weight") {
                return (0..n).map(|_| bf(1.0 + 0.2 * r.next())).collect();
            } else if key.contains("_sp.") {
                0.2
            } else if key.contains("_tm.") {
                0.45
            } else if key.ends_with("bias") {
                1.0
            } else {
                1.0 / (*shape.last().unwrap_or(&1) as f32).sqrt()
            };
            r.vec(n, scale)
        })
    }

    /// The whole branch (device kernels vs the host reference) on a small
    /// layout: heads 4, 14 frames of 4 x 6 tokens.
    fn branch_check(out: &mut Vec<VdnCheck>) -> Result<()> {
        let mut cfg = fastvideo_models::h3::config::H3TransformerConfig::fasth3_8step();
        cfg.num_attention_heads = 4;
        cfg.hidden_size = 256;
        let map = generated(5);
        let branch = VdnBranch::load(&map, "transformer_blocks.0.attn", &cfg)?;
        let layout = H3PackedLayout::new(10, (14, 8, 12), 3, cfg.patch_size).map_err(msg)?;
        let plan = VdnPlan::new(&layout, &VdnTransform::released())?;
        let sl = layout.sequence_length();
        let mut rng = Rng(77);
        let n = CudaTensor::from_vec(rng.vec(sl * cfg.hidden_size, 1.5), vec![1, sl, cfg.hidden_size])?.to_device()?;
        let c = cfg.num_attention_heads * D;
        let packed = CudaTensor::from_vec(rng.vec(sl * 3 * c, 2.0), vec![1, sl, 3 * c])?.to_device()?;
        let aux = branch.aux(&n)?;
        let dev_r = branch.readout(&n, &packed, &aux, &plan, false)?.host_cow()?.into_owned();
        let host_r = branch.readout(&n, &packed, &aux, &plan, true)?.host_cow()?.into_owned();
        out.push(compare("branch_end_to_end", &dev_r, &host_r, 3e-2, "device kernels vs host reference, 12 inner frames"));
        Ok(())
    }

    /// Branch and window timings at the real head count for one layer.
    fn timings(times: &mut Vec<VdnTiming>) -> Result<()> {
        let cfg = fastvideo_models::h3::config::H3TransformerConfig::fasth3_8step();
        let map = generated(9);
        let branch = VdnBranch::load(&map, "transformer_blocks.0.attn", &cfg)?;
        let c = cfg.num_attention_heads * D;
        for (label, latent) in [("480p_5s", (37usize, 30usize, 52usize)), ("768p_5s", (37, 48, 84))] {
            let layout = H3PackedLayout::new(256, latent, 207, cfg.patch_size).map_err(msg)?;
            let plan = VdnPlan::new(&layout, &VdnTransform::released())?;
            let sl = layout.sequence_length();
            let mut rng = Rng(3);
            let n = CudaTensor::from_vec(rng.vec(sl * cfg.hidden_size, 1.0), vec![1, sl, cfg.hidden_size])?.to_device()?.quantize_bf16()?;
            let packed = CudaTensor::from_vec(rng.vec(sl * 3 * c, 2.0), vec![1, sl, 3 * c])?.to_device()?.quantize_bf16()?;
            let aux = branch.aux(&n)?;
            let _ = branch.readout(&n, &packed, &aux, &plan, false)?;
            sync()?;
            let t0 = std::time::Instant::now();
            let _ = branch.readout(&n, &packed, &aux, &plan, false)?;
            sync()?;
            times.push(VdnTiming { name: format!("branch_{label}"), ms: t0.elapsed().as_secs_f64() * 1e3 });
            let hd = cfg.num_attention_heads;
            let qkv = |k: f32| -> Result<CudaTensor> {
                CudaTensor::from_vec(Rng(k as u64).vec(hd * sl * D, 1.0), vec![1, hd, sl, D])?.to_device()?.quantize_bf16()
            };
            let (q, k, v) = (qkv(1.0)?, qkv(2.0)?, qkv(3.0)?);
            for (name, window) in [("window", true), ("dense", false)] {
                let run = || -> Result<()> {
                    if window {
                        device_mma_sdpa_window(&q, &k, &v, None, true, plan.window)?;
                    } else {
                        device_mma_sdpa_with(&q, &k, &v, None, true, FlashKernel::V2)?;
                    }
                    Ok(())
                };
                run()?;
                sync()?;
                let t0 = std::time::Instant::now();
                run()?;
                sync()?;
                times.push(VdnTiming { name: format!("attn_{name}_{label}"), ms: t0.elapsed().as_secs_f64() * 1e3 });
            }
        }
        Ok(())
    }

    pub fn run(seed: u64) -> Result<(Vec<VdnCheck>, Vec<VdnTiming>)> {
        let mut out = Vec::new();
        let mut rng = Rng(seed | 1);
        window_checks(&mut rng, &mut out)?;
        stage_checks(&mut rng, &mut out)?;
        branch_check(&mut out)?;
        let mut times = Vec::new();
        timings(&mut times)?;
        Ok((out, times))
    }
}

#[cfg(feature = "cuda")]
pub(crate) mod dev {
    //! Launchers for `vdn.cu` (its own module, ahead-of-time cubins with
    //! IEEE math, NVRTC from the same source as a fallback).

    use std::sync::Arc;

    use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig};

    use super::{msg, AuxCols, Small, VdnBranch, VdnPlan};
    use crate::wan::device::DeviceContext;
    use crate::wan::kernels::launch;
    use crate::wan::quant::{ptr, ptr_mut};
    use crate::wan::tensor::{CudaTensor, Result};
    use fastvideo_models::h3::vdn::{VDN_HEAD_DIM as D, VDN_NORM_EPS, VDN_TEXT_STATE_SCALE};

    const SRC: &str = include_str!("vdn.cu");
    const SOLVE_SMEM: u32 = ((D * D + 2 * D) * 4) as u32;
    const SCAN_SMEM: u32 = ((D * D + 32 * D) * 4) as u32;
    const READOUT_SMEM: u32 = (D * D * 2 + 64 * D * 4) as u32;

    fn err(e: impl std::fmt::Display) -> crate::wan::tensor::TensorError {
        msg(format!("vdn: {e}"))
    }

    fn ctx() -> Result<Arc<DeviceContext>> {
        crate::wan::device::global_device().ok_or_else(|| msg("vdn: no global CUDA device"))
    }

    pub(crate) struct Kernels {
        _module: Arc<cudarc::driver::CudaModule>,
        act_rows: CudaFunction,
        sconv5: CudaFunction,
        tconv5_act: CudaFunction,
        frame_stats: CudaFunction,
        frame_mean: CudaFunction,
        alpha_down: CudaFunction,
        alpha_up: CudaFunction,
        solve: CudaFunction,
        scan: CudaFunction,
        logprefix: CudaFunction,
        gather: CudaFunction,
        copy_cols: CudaFunction,
        readout: CudaFunction,
        head_gate: CudaFunction,
        add_rows: CudaFunction,
    }

    mod aot {
        include!(concat!(env!("OUT_DIR"), "/aot_h3_vdn.rs"));
    }

    pub(crate) fn kernels() -> Result<std::rc::Rc<Kernels>> {
        thread_local! {
            static CELL: std::cell::RefCell<Option<std::rc::Rc<Kernels>>> =
                const { std::cell::RefCell::new(None) };
        }
        CELL.with(|c| {
            if let Some(k) = c.borrow().as_ref() {
                return Ok(k.clone());
            }
            let loaded = std::rc::Rc::new(load()?);
            *c.borrow_mut() = Some(loaded.clone());
            Ok(loaded)
        })
    }

    fn load_embedded(dev: &DeviceContext) -> Result<Option<(Arc<cudarc::driver::CudaModule>, String)>> {
        let want = (dev.sm_major * 10 + dev.sm_minor) as u32;
        if crate::wan::envflag::string_flag("FASTVIDEO_KERNELS", "auto") == "nvrtc" {
            return Ok(None);
        }
        if let Some((_, cubin, _)) = aot::AOT_H3_VDN.iter().find(|(sm, _, _)| *sm == want) {
            let path = std::env::temp_dir().join(format!("fv-h3-vdn-{}-sm{want}.cubin", std::process::id()));
            std::fs::write(&path, cubin).map_err(err)?;
            let loaded = dev.ctx.load_module(cudarc::nvrtc::Ptx::from_file(&path));
            let _ = std::fs::remove_file(&path);
            return Ok(Some((loaded.map_err(err)?, format!("cubin sm{want}"))));
        }
        if let Some((sm, _, ptx)) = aot::AOT_H3_VDN
            .iter()
            .filter(|(sm, _, _)| sm / 10 == want / 10 && *sm <= want)
            .max_by_key(|(sm, _, _)| *sm)
        {
            let module = dev.ctx.load_module(cudarc::nvrtc::Ptx::from_src(*ptx)).map_err(err)?;
            return Ok(Some((module, format!("ptx compute{sm}"))));
        }
        Ok(None)
    }

    fn load() -> Result<Kernels> {
        use cudarc::nvrtc::CompileOptions;
        let dev = ctx()?;
        let timer = std::time::Instant::now();
        let (module, origin) = if let Some(m) = load_embedded(&dev)? {
            m
        } else {
            let opts = CompileOptions {
                use_fast_math: Some(false),
                ftz: Some(false),
                prec_div: Some(true),
                prec_sqrt: Some(true),
                fmad: Some(true),
                options: vec!["-std=c++17".into()],
                ..Default::default()
            };
            let (module, origin) =
                crate::wan::nvrtc_sass::load_for_device(&dev.ctx, dev.sm_major, dev.sm_minor, SRC, &opts, "h3-vdn")
                    .map_err(err)?;
            (module, origin.to_string())
        };
        let f = |name: &str| module.load_function(name).map_err(err);
        let k = Kernels {
            act_rows: f("vdn_act_rows")?,
            sconv5: f("vdn_sconv5")?,
            tconv5_act: f("vdn_tconv5_act")?,
            frame_stats: f("vdn_frame_stats")?,
            frame_mean: f("vdn_frame_mean")?,
            alpha_down: f("vdn_alpha_down")?,
            alpha_up: f("vdn_alpha_up")?,
            solve: f("vdn_solve")?,
            scan: f("vdn_scan")?,
            logprefix: f("vdn_logprefix")?,
            gather: f("vdn_gather")?,
            copy_cols: f("vdn_copy_cols")?,
            readout: f("vdn_readout")?,
            head_gate: f("vdn_head_gate")?,
            add_rows: f("vdn_add_rows")?,
            _module: module,
        };
        crate::wan::ops::opt_in_dynamic_shared(&k.solve, SOLVE_SMEM)?;
        crate::wan::ops::opt_in_dynamic_shared(&k.scan, SCAN_SMEM)?;
        crate::wan::ops::opt_in_dynamic_shared(&k.readout, READOUT_SMEM)?;
        crate::wan::log::info(format_args!(
            "h3 vdn: kernels loaded ({origin}, {:.1}s, sm{}{})",
            timer.elapsed().as_secs_f64(),
            dev.sm_major,
            dev.sm_minor
        ));
        Ok(k)
    }

    fn alloc<T: cudarc::driver::DeviceRepr>(n: usize) -> Result<CudaSlice<T>> {
        let dev = ctx()?;
        unsafe { dev.stream.alloc::<T>(n.max(1)) }.map_err(err)
    }

    fn upload_bf16(v: &[f32]) -> Result<CudaSlice<half::bf16>> {
        let host: Vec<half::bf16> = v.iter().map(|&x| half::bf16::from_f32(x)).collect();
        ctx()?.stream.memcpy_stod(&host).map_err(err)
    }

    fn cfg(grid: (u32, u32, u32), smem: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: grid,
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        }
    }

    fn blocks(n: usize, per: usize) -> u32 {
        n.div_ceil(per).max(1) as u32
    }

    /// The small branch tensors as bf16 on the device; convolutions tap-major.
    pub(crate) struct DevSmall {
        a_log: CudaSlice<half::bf16>,
        dt_bias: CudaSlice<half::bf16>,
        alpha_down: CudaSlice<half::bf16>,
        alpha_up: CudaSlice<half::bf16>,
        norm: CudaSlice<half::bf16>,
        k_sp: CudaSlice<half::bf16>,
        k_tm: CudaSlice<half::bf16>,
        v_sp: CudaSlice<half::bf16>,
        v_tm: CudaSlice<half::bf16>,
    }

    fn tap_major(w: &[f32], c: usize, taps: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; w.len()];
        for ch in 0..c {
            for t in 0..taps {
                out[t * c + ch] = w[ch * taps + t];
            }
        }
        out
    }

    impl DevSmall {
        pub(crate) fn upload(s: &Small, heads: usize) -> Result<Self> {
            let c = heads * D;
            Ok(Self {
                a_log: upload_bf16(&s.a_log)?,
                dt_bias: upload_bf16(&s.dt_bias)?,
                alpha_down: upload_bf16(&s.alpha_down)?,
                alpha_up: upload_bf16(&s.alpha_up)?,
                norm: upload_bf16(&s.norm)?,
                k_sp: upload_bf16(&tap_major(&s.k_sp, c, 25))?,
                k_tm: upload_bf16(&tap_major(&s.k_tm, c, 5))?,
                v_sp: upload_bf16(&tap_major(&s.v_sp, c, 25))?,
                v_tm: upload_bf16(&tap_major(&s.v_tm, c, 5))?,
            })
        }
    }

    /// A bf16 device view of `t` (converted when stored as f32).
    fn bf16_of(t: &CudaTensor) -> Result<Arc<CudaSlice<half::bf16>>> {
        t.dev_bf16()?.ok_or_else(|| msg("vdn: tensor not on the device"))
    }

    fn tensor16(s: CudaSlice<half::bf16>, shape: Vec<usize>) -> Result<CudaTensor> {
        CudaTensor::from_device_slice_bf16(s, shape)
    }

    // ---- stage launchers (pub(crate): fv-gpucheck checks each one) --------

    /// SiLU (+ L2) of `rows` rows of `x` at `x + row * stride + off`.
    pub(crate) fn act_rows(
        x: u64,
        stride: usize,
        off: usize,
        rows: usize,
        heads: usize,
        l2: bool,
    ) -> Result<CudaSlice<half::bf16>> {
        let dev = ctx()?;
        let k = kernels()?;
        let mut out = alloc::<half::bf16>(rows * heads * D)?;
        let (o, l2i) = (ptr_mut(&mut out), i32::from(l2));
        let (st, of, r, h) = (stride as i64, off as i64, rows as i32, heads as i32);
        launch!(dev.stream, &k.act_rows, cfg((blocks(rows * heads, 8), 1, 1), 0);
            &x, &st, &of, &r, &h, &l2i, &o)
        .map_err(err)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sconv5(
        x: u64,
        stride: usize,
        off: usize,
        frames: usize,
        gh: usize,
        gw: usize,
        c: usize,
        wt: &CudaSlice<half::bf16>,
    ) -> Result<CudaSlice<half::bf16>> {
        let dev = ctx()?;
        let k = kernels()?;
        let n = frames * gh * gw;
        let mut out = alloc::<half::bf16>(n * c)?;
        let (o, w) = (ptr_mut(&mut out), ptr(wt));
        let (st, of, ghi, gwi, ci) = (stride as i64, off as i64, gh as i32, gw as i32, c as i32);
        launch!(dev.stream, &k.sconv5, cfg((n as u32, blocks(c / 2, 256), 1), 0);
            &x, &st, &of, &ghi, &gwi, &ci, &w, &o)
        .map_err(err)?;
        Ok(out)
    }

    pub(crate) fn tconv5_act(
        x: &CudaSlice<half::bf16>,
        frames: usize,
        s: usize,
        heads: usize,
        wt: &CudaSlice<half::bf16>,
        l2: bool,
    ) -> Result<CudaSlice<half::bf16>> {
        let dev = ctx()?;
        let k = kernels()?;
        let rows = frames * s;
        let mut out = alloc::<half::bf16>(rows * heads * D)?;
        let (xi, w, o) = (ptr(x), ptr(wt), ptr_mut(&mut out));
        let (f, si, h, l2i) = (frames as i32, s as i32, heads as i32, i32::from(l2));
        launch!(dev.stream, &k.tconv5_act, cfg((blocks(rows * heads, 8), 1, 1), 0);
            &xi, &f, &si, &h, &w, &l2i, &o)
        .map_err(err)?;
        Ok(out)
    }

    /// `(A, B)` `[chunks, H, D, D]` fp32 of `k`, `v` (`[chunks * s, C]`), beta
    /// from `aux` rows `aux_row0..` (logits at column `beta_off`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn frame_stats(
        k_feat: &CudaSlice<half::bf16>,
        v_feat: &CudaSlice<half::bf16>,
        aux: u64,
        aux_stride: usize,
        beta_off: usize,
        chunks: usize,
        s: usize,
        heads: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
        let dev = ctx()?;
        let kk = kernels()?;
        let n = chunks * heads * D * D;
        let mut a = alloc::<f32>(n)?;
        let mut b = alloc::<f32>(n)?;
        let (kp, vp, ap, bp) = (ptr(k_feat), ptr(v_feat), ptr_mut(&mut a), ptr_mut(&mut b));
        let (st, of, si, h) = (aux_stride as i64, beta_off as i64, s as i32, heads as i32);
        launch!(dev.stream, &kk.frame_stats, cfg(((chunks * heads) as u32, 1, 1), 0);
            &kp, &vp, &aux, &st, &of, &si, &h, &ap, &bp)
        .map_err(err)?;
        Ok((a, b))
    }

    /// `alpha [frames, C]` from the attention input rows `row0..`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn alpha(
        x: u64,
        row0: usize,
        frames: usize,
        s: usize,
        hidden: usize,
        heads: usize,
        sm: &DevSmall,
    ) -> Result<CudaSlice<f32>> {
        let dev = ctx()?;
        let k = kernels()?;
        let c = heads * D;
        let mut mean = alloc::<f32>(frames * hidden)?;
        let mp = ptr_mut(&mut mean);
        let (r0, si, hi) = (row0 as i64, s as i32, hidden as i32);
        launch!(dev.stream, &k.frame_mean, cfg((blocks(hidden, 256), frames as u32, 1), 0);
            &x, &r0, &si, &hi, &mp)
        .map_err(err)?;
        let mut z = alloc::<f32>(frames * D)?;
        let (zp, dp, ri) = (ptr_mut(&mut z), ptr(&sm.alpha_down), D as i32);
        let mp = ptr(&mean);
        launch!(dev.stream, &k.alpha_down, cfg((D as u32, frames as u32, 1), 0);
            &mp, &dp, &hi, &ri, &zp)
        .map_err(err)?;
        let mut alpha = alloc::<f32>(frames * c)?;
        let (zp, up, dt, al, ap) = (ptr(&z), ptr(&sm.alpha_up), ptr(&sm.dt_bias), ptr(&sm.a_log), ptr_mut(&mut alpha));
        let (ci, fi) = (c as i32, frames as i32);
        launch!(dev.stream, &k.alpha_up, cfg((blocks(frames * c, 256), 1, 1), 0);
            &zp, &up, &dt, &al, &ri, &ci, &fi, &ap)
        .map_err(err)?;
        Ok(alpha)
    }

    /// `(T, J)` of `n` matrices; `alpha = None` writes no `T` (the text chunk).
    pub(crate) fn solve(
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        alpha: Option<&CudaSlice<f32>>,
        n: usize,
        jscale: f32,
    ) -> Result<(Option<CudaSlice<f32>>, CudaSlice<f32>)> {
        let dev = ctx()?;
        let k = kernels()?;
        let mut j = alloc::<f32>(n * D * D)?;
        let mut t = match alpha {
            Some(_) => Some(alloc::<f32>(n * D * D)?),
            None => None,
        };
        let (ap, bp, jp) = (ptr(a), ptr(b), ptr_mut(&mut j));
        let alp = alpha.map_or(ap, ptr);
        let tp = match t.as_mut() {
            Some(t) => ptr_mut(t),
            None => jp,
        };
        let (has, wt) = (i32::from(alpha.is_some()), i32::from(alpha.is_some()));
        launch!(dev.stream, &k.solve, cfg((n as u32, 1, 1), SOLVE_SMEM);
            &ap, &bp, &alp, &has, &jscale, &wt, &tp, &jp)
        .map_err(err)?;
        Ok((t, j))
    }

    pub(crate) fn scans(
        t: &CudaSlice<f32>,
        j: &CudaSlice<f32>,
        start: &CudaSlice<f32>,
        frames: usize,
        heads: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
        let dev = ctx()?;
        let k = kernels()?;
        let n = frames * heads * D * D;
        let mut p = alloc::<f32>(n)?;
        let mut q = alloc::<f32>(n)?;
        let (tp, jp, sp, pp, qp) = (ptr(t), ptr(j), ptr(start), ptr_mut(&mut p), ptr_mut(&mut q));
        let (f, h) = (frames as i32, heads as i32);
        launch!(dev.stream, &k.scan, cfg((heads as u32, (D / 32) as u32, 2), SCAN_SMEM);
            &tp, &jp, &sp, &f, &h, &pp, &qp)
        .map_err(err)?;
        Ok((p, q))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gather(
        p: &CudaSlice<f32>,
        q: &CudaSlice<f32>,
        start: &CudaSlice<f32>,
        alpha: &CudaSlice<f32>,
        plan: &CudaSlice<i32>,
        frames: usize,
        heads: usize,
    ) -> Result<CudaSlice<half::bf16>> {
        let dev = ctx()?;
        let k = kernels()?;
        let c = heads * D;
        let mut lp = alloc::<f64>((frames + 1) * c)?;
        let (ap, lpp) = (ptr(alpha), ptr_mut(&mut lp));
        let (f, ci, h) = (frames as i32, c as i32, heads as i32);
        launch!(dev.stream, &k.logprefix, cfg((blocks(c, 256), 1, 1), 0); &ap, &f, &ci, &lpp)
            .map_err(err)?;
        let n = frames * heads * D * D;
        let mut out = alloc::<half::bf16>(n)?;
        let (pp, qp, sp, lpp, pl, op) = (ptr(p), ptr(q), ptr(start), ptr(&lp), ptr(plan), ptr_mut(&mut out));
        launch!(dev.stream, &k.gather, cfg((blocks(n / 2, 256), 1, 1), 0);
            &pp, &qp, &sp, &lpp, &pl, &f, &h, &op)
        .map_err(err)?;
        Ok(out)
    }

    pub(crate) fn copy_cols(src: u64, stride: usize, off: usize, rows: usize, w: usize) -> Result<CudaSlice<half::bf16>> {
        let dev = ctx()?;
        let k = kernels()?;
        let mut out = alloc::<half::bf16>(rows * w)?;
        let o = ptr_mut(&mut out);
        let (st, of, r, wi) = (stride as i64, off as i64, rows as i64, w as i32);
        launch!(dev.stream, &k.copy_cols, cfg((blocks(rows * w, 256), 1, 1), 0);
            &src, &st, &of, &r, &wi, &o)
        .map_err(err)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn readout(
        state: &CudaSlice<half::bf16>,
        q: &CudaSlice<half::bf16>,
        og: u64,
        norm: &CudaSlice<half::bf16>,
        frames: usize,
        s: usize,
        heads: usize,
    ) -> Result<CudaSlice<half::bf16>> {
        let dev = ctx()?;
        let k = kernels()?;
        let mut out = alloc::<half::bf16>(frames * s * heads * D)?;
        let (sp, qp, np, op) = (ptr(state), ptr(q), ptr(norm), ptr_mut(&mut out));
        let (si, h, eps) = (s as i32, heads as i32, VDN_NORM_EPS);
        launch!(dev.stream, &k.readout, cfg((blocks(s, 64), heads as u32, frames as u32), READOUT_SMEM);
            &sp, &qp, &og, &np, &si, &h, &eps, &op)
        .map_err(err)?;
        Ok(out)
    }

    pub(crate) fn gate_heads(o: &CudaTensor, aux: &CudaTensor, cols: AuxCols, heads: usize) -> Result<Option<CudaTensor>> {
        let [1, h, s, d] = o.shape[..] else {
            return Ok(None);
        };
        if h != heads || d != D || !crate::wan::stats::device_expected() {
            return Ok(None);
        }
        let dev = ctx()?;
        let k = kernels()?;
        let src = bf16_of(o)?;
        let mut out = alloc::<half::bf16>(h * s * d)?;
        dev.stream.memcpy_dtod(&*src, &mut out).map_err(err)?;
        let a = bf16_of(aux)?;
        let (op, ap) = (ptr_mut(&mut out), ptr(&*a));
        let (st, of, si, hi) = (cols.width as i64, cols.sg as i64, s as i32, h as i32);
        launch!(dev.stream, &k.head_gate, cfg((blocks(h * s * (d / 4), 256), 1, 1), 0);
            &op, &ap, &st, &of, &si, &hi)
        .map_err(err)?;
        Ok(Some(tensor16(out, o.shape.clone())?))
    }

    pub(crate) fn add_rows(out: &CudaTensor, add: &CudaTensor, row0: usize) -> Result<Option<CudaTensor>> {
        if !crate::wan::stats::device_expected() {
            return Ok(None);
        }
        let hidden = out.shape[out.rank() - 1];
        let dev = ctx()?;
        let k = kernels()?;
        let src = bf16_of(out)?;
        let mut o = alloc::<half::bf16>(src.len())?;
        dev.stream.memcpy_dtod(&*src, &mut o).map_err(err)?;
        let a = bf16_of(add)?;
        let n = a.len();
        let (op, ap) = (ptr_mut(&mut o), ptr(&*a));
        let (off, ni) = ((row0 * hidden) as i64, n as i64);
        launch!(dev.stream, &k.add_rows, cfg((blocks(n, 256), 1, 1), 0); &op, &off, &ap, &ni)
            .map_err(err)?;
        Ok(Some(tensor16(o, out.shape.clone())?))
    }

    /// The device branch: `[inner rows, C]` bf16, gated and normed.
    pub(crate) fn branch(
        b: &VdnBranch,
        sm: &DevSmall,
        n: &CudaTensor,
        packed: &CudaTensor,
        aux: &CudaTensor,
        plan: &VdnPlan,
    ) -> Result<CudaTensor> {
        let l = &plan.layout;
        let heads = b.heads;
        let c = heads * D;
        let width = packed.shape[packed.rank() - 1];
        let (r0, rows) = l.inner_rows();
        let frames = plan.inner_frames;
        let s = l.tokens_per_frame;
        let xs = bf16_of(n)?;
        let ps = bf16_of(packed)?;
        let as_ = bf16_of(aux)?;
        let (xp, pp, ap) = (ptr(&*xs), ptr(&*ps), ptr(&*as_));
        let el = 2usize; // bytes per bf16
        let row_ptr = |base: u64, row: usize, stride: usize| base + (row * stride * el) as u64;
        // 1. features
        let q = act_rows(row_ptr(pp, r0, width), width, 0, rows, heads, true)?;
        let k_sp = sconv5(row_ptr(pp, r0, width), width, c, frames, l.grid_h, l.grid_w, c, &sm.k_sp)?;
        let k = tconv5_act(&k_sp, frames, s, heads, &sm.k_tm, true)?;
        drop(k_sp);
        let v_sp = sconv5(row_ptr(pp, r0, width), width, 2 * c, frames, l.grid_h, l.grid_w, c, &sm.v_sp)?;
        let v = tconv5_act(&v_sp, frames, s, heads, &sm.v_tm, false)?;
        drop(v_sp);
        // 2-4. statistics, decay
        let aw = b.cols.width;
        let (a, bm) = frame_stats(&k, &v, row_ptr(ap, r0, aw), aw, b.cols.beta, frames, s, heads)?;
        drop((k, v));
        let alpha = alpha(xp, r0, frames, s, b.hidden, heads, sm)?;
        // 6. the text state (both scans start from half of it)
        let (t0, tl) = (l.text_start, l.text_len);
        let kt = act_rows(row_ptr(pp, t0, width), width, c, tl, heads, true)?;
        let vt = act_rows(row_ptr(pp, t0, width), width, 2 * c, tl, heads, false)?;
        let (at, bt) = frame_stats(&kt, &vt, row_ptr(ap, t0, aw), aw, b.cols.beta, 1, tl, heads)?;
        drop((kt, vt));
        let (_, start) = solve(&at, &bt, None, heads, VDN_TEXT_STATE_SCALE)?;
        drop((at, bt));
        // 5, 7. factors, scans
        let (t, j) = solve(&a, &bm, Some(&alpha), frames * heads, 1.0)?;
        drop((a, bm));
        let t = t.ok_or_else(|| msg("vdn: solve wrote no transitions"))?;
        let (pb, qb) = scans(&t, &j, &start, frames, heads)?;
        drop((t, j));
        // 8. gather
        let planv = plan_slice(plan)?;
        let state = gather(&pb, &qb, &start, &alpha, &planv, frames, heads)?;
        drop((pb, qb, alpha, start));
        // 9. output gate and readout
        let og_mid = tensor16(copy_cols(row_ptr(ap, r0, aw), aw, b.cols.og, rows, b.rank)?, vec![rows, b.rank])?;
        let og = b.og_up.forward(&og_mid)?;
        drop(og_mid);
        let ogs = bf16_of(&og)?;
        let out = readout(&state, &q, ptr(&*ogs), &sm.norm, frames, s, heads)?;
        tensor16(out, vec![rows, c])
    }

    fn plan_slice(plan: &VdnPlan) -> Result<Arc<CudaSlice<i32>>> {
        if let Some(p) = plan.plan_dev.get() {
            return Ok(p.clone());
        }
        let s = Arc::new(ctx()?.stream.memcpy_stod(&plan.plan_rows).map_err(err)?);
        let _ = plan.plan_dev.set(s.clone());
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h3::transformer::tests::tiny_cfg;
    use crate::h3::transformer::{AttnMode, DeviceLayout, H3Transformer};
    use fastvideo_models::h3::schedule::H3JointSchedule;

    /// The transformer tests' weights, with the softmax gate open (logit 30:
    /// sigmoid = 1.0 in f32) and, if `closed`, `to_out_linear` zero.
    fn vdn_weights(closed: bool) -> WeightMap {
        WeightMap::generated(move |key, shape| {
            let n: usize = shape.iter().product();
            if key.ends_with("softmax_gate.up.weight")
                || (closed && key.ends_with("to_out_linear.weight"))
            {
                return vec![0.0; n];
            }
            if key.ends_with("softmax_gate.up.bias") {
                return vec![30.0; n];
            }
            let seed = key
                .bytes()
                .fold(13u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
            (0..n)
                .map(|i| {
                    let u = (seed.wrapping_add(i as u32).wrapping_mul(2_654_435_761) >> 8) as f32
                        / (1u32 << 24) as f32;
                    if key.contains("norm") {
                        0.5 + u
                    } else {
                        (u - 0.5) * 0.8
                    }
                })
                .collect()
        })
    }

    fn seeded(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * k + 0.3).sin()).collect()
    }

    struct Run {
        video: Vec<f32>,
        block0: Vec<f32>,
    }

    fn run(frames: usize, closed: bool, mode_vdn: bool) -> (Run, H3PackedLayout) {
        let cfg = tiny_cfg();
        let map = vdn_weights(closed);
        let schedule = H3JointSchedule::fasth3_8step();
        let mut model = H3Transformer::load(cfg.clone(), &map, &schedule, false).unwrap();
        model.attach_vdn(&map).unwrap();
        assert!(model.has_vdn());
        let layout = H3PackedLayout::new(3, (frames, 2, 4), 1, cfg.patch_size).unwrap();
        let (nv, na, nt) = (layout.video.len, layout.audio.len, layout.text.len);
        let plan = VdnPlan::new(&layout, &VdnTransform::released()).unwrap();
        let dl = DeviceLayout::new(&cfg, layout.clone()).unwrap();
        let mut block0 = Vec::new();
        let mode = if mode_vdn {
            AttnMode::Vdn(&plan)
        } else {
            AttnMode::Dense
        };
        let t = |v: Vec<f32>, shape: Vec<usize>| CudaTensor::from_vec(v, shape).unwrap();
        let (v, _) = model
            .forward(
                2,
                &t(seeded(nv * cfg.video_patch_dim(), 0.41), vec![nv, cfg.video_patch_dim()]),
                &t(seeded(na * cfg.audio_in_channels, 0.23), vec![na, cfg.audio_in_channels]),
                &t(seeded(nt * cfg.hidden_size, 0.57), vec![1, nt, cfg.hidden_size]),
                &dl,
                mode,
                Some(&mut |name, t| {
                    if name == "block_0" {
                        block0 = t.host_cow()?.into_owned();
                    }
                    Ok(())
                }),
                None,
                None,
            )
            .unwrap();
        (
            Run {
                video: v.host_cow().unwrap().into_owned(),
                block0,
            },
            layout,
        )
    }

    /// A clip the window covers (6 latent frames, under 11) runs no branch;
    /// with the softmax gate at 1 the hybrid is the dense forward.
    #[test]
    fn a_covered_clip_with_an_open_gate_is_dense_attention() {
        let (vdn, layout) = run(6, false, true);
        let (dense, _) = run(6, false, false);
        assert!(VdnPlan::new(&layout, &VdnTransform::released()).unwrap().full_cover);
        let scale = dense.video.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(scale > 1e-3);
        for (i, (a, b)) in vdn.video.iter().zip(&dense.video).enumerate() {
            assert!((a - b).abs() <= 1e-5 * scale, "video[{i}]: {a} vs {b}");
        }
    }

    /// 12 frames (not covered): the branch writes the inner video rows of the
    /// first block's output only; text, audio and the anchor frames 0 / 11
    /// are untouched by `to_out_linear`, and everything stays finite.
    #[test]
    fn the_branch_writes_only_the_inner_video_rows() {
        let (open, layout) = run(12, false, true);
        let (closed, _) = run(12, true, true);
        let plan = VdnPlan::new(&layout, &VdnTransform::released()).unwrap();
        assert!(plan.branch_active());
        let hidden = tiny_cfg().hidden_size;
        let (r0, rows) = plan.layout.inner_rows();
        assert_eq!((r0, rows), (layout.video.start + 2, 10 * 2));
        assert!(open.block0.iter().chain(&open.video).all(|v| v.is_finite()));
        let mut inner_diff = 0.0f32;
        for row in 0..layout.sequence_length() {
            let (a, b) = (
                &open.block0[row * hidden..(row + 1) * hidden],
                &closed.block0[row * hidden..(row + 1) * hidden],
            );
            if (r0..r0 + rows).contains(&row) {
                inner_diff = a
                    .iter()
                    .zip(b)
                    .fold(inner_diff, |m, (x, y)| m.max((x - y).abs()));
            } else {
                assert_eq!(a, b, "row {row} is outside the inner frames");
            }
        }
        assert!(inner_diff > 1e-4, "the branch changed nothing ({inner_diff})");
    }

    /// The window SDPA's host path is the reference's masked softmax, and the
    /// kernel's mask description agrees with the model's.
    #[test]
    fn the_window_softmax_is_the_reference_mask() {
        let cfg = tiny_cfg();
        let layout = H3PackedLayout::new(3, (12, 2, 4), 1, cfg.patch_size).unwrap();
        let plan = VdnPlan::new(&layout, &VdnTransform::released()).unwrap();
        let bounds = plan.transform.bounds(plan.layout.num_frames);
        let s = layout.sequence_length();
        for q in 0..s {
            for k in 0..s {
                assert_eq!(
                    plan.window.allows(q, k),
                    plan.layout.allows(&bounds, q, k),
                    "{q} {k}"
                );
            }
        }
        let (h, d) = (2usize, 8usize);
        let q = seeded(s * h * d, 0.31);
        let k = seeded(s * h * d, 0.17);
        let v = seeded(s * h * d, 0.53);
        let want = vref::reference_window_attention(&q, &k, &v, &plan.layout, &bounds, h, d);
        let bhsd = |x: &[f32]| {
            let mut out = vec![0.0f32; x.len()];
            for si in 0..s {
                for hh in 0..h {
                    for e in 0..d {
                        out[(hh * s + si) * d + e] = x[(si * h + hh) * d + e];
                    }
                }
            }
            CudaTensor::from_vec(out, vec![1, h, s, d]).unwrap()
        };
        let got =
            crate::wan::nn::sdpa_vdn_window(&bhsd(&q), &bhsd(&k), &bhsd(&v), None, plan.window)
                .unwrap();
        let got = got.host_cow().unwrap();
        for si in 0..s {
            for hh in 0..h {
                for e in 0..d {
                    let (a, b) = (got[(hh * s + si) * d + e], want[(si * h + hh) * d + e]);
                    assert!((a - b).abs() < 1e-5, "row {si} head {hh}: {a} vs {b}");
                }
            }
        }
    }
}
