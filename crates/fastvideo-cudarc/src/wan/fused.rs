//! Fused Wan ops on [`CudaTensor`]: one kernel launch each on the device, a
//! plain-Rust twin on CPU runs (see [`super::ops::host`]).

use super::ops::host;
use super::tensor::{CudaTensor, Result, TensorError};

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// RoPE tables for [`CudaTensor::qk_norm_rope_bhsd`]: `[seq, head_dim]` each.
pub struct Rope<'a> {
    pub cos: &'a CudaTensor,
    pub sin: &'a CudaTensor,
}

impl CudaTensor {
    /// `LN(self) * (1 + e[:, scale_slot]) + e[:, shift_slot]` for `self`
    /// `[batch, seq, dim]` and the AdaLN table `e` `[batch, e_rows, dim]`.
    pub fn ln_adaln_e(
        &self,
        e: &CudaTensor,
        scale_slot: usize,
        shift_slot: usize,
        eps: f32,
    ) -> Result<CudaTensor> {
        let [batch, seq, dim] = self.shape[..] else {
            return Err(msg(format!(
                "ln_adaln_e expects [b, seq, dim], got {:?}",
                self.shape
            )));
        };
        let e_rows = e.shape.get(1).copied().unwrap_or(0);
        if e.shape != [batch, e_rows, dim] || scale_slot >= e_rows || shift_slot >= e_rows {
            return Err(msg(format!(
                "ln_adaln_e table {:?} for {:?}",
                e.shape, self.shape
            )));
        }
        #[cfg(feature = "cuda")]
        if self.act16_device() && self.is_bf16() {
            if let Some(t) = super::act16::ln_adaln_e(
                self, e, batch, seq, dim, e_rows, scale_slot, shift_slot, eps,
            )? {
                return Ok(t);
            }
        }
        #[cfg(feature = "cuda")]
        if let (Some(x), Some(ed)) = (self.dev()?, e.dev()?) {
            let out = super::ops::ln_adaln_e_device(
                &x, &ed, batch, seq, dim, e_rows, scale_slot, shift_slot, eps,
            )?;
            return self.keep_dtype(Self::from_dev_result(out, self.shape.clone())?);
        }
        let out = host::ln_adaln_e(
            &self.host_cow()?,
            &e.host_cow()?,
            seq,
            dim,
            e_rows,
            scale_slot,
            shift_slot,
            eps,
        );
        self.keep_dtype(Self::host_only(out, self.shape.clone()))
    }

    /// `rope_half(ln_adaln_e(x))` in one launch when `FASTVIDEO_NVFP4` is on
    /// and `self` is BHSD with AdaLN over the head width. Off keeps the two
    /// existing launches (caller should use [`Self::ln_adaln_e`] then
    /// [`Self::rope_half`]); this method still matches that composition.
    pub fn ln_adaln_e_rope_half(
        &self,
        e: &CudaTensor,
        scale_slot: usize,
        shift_slot: usize,
        eps: f32,
        cos: &CudaTensor,
        sin: &CudaTensor,
    ) -> Result<CudaTensor> {
        let [batch, heads, seq, dim] = self.shape[..] else {
            return Err(msg(format!(
                "ln_adaln_e_rope_half expects [b, h, s, d], got {:?}",
                self.shape
            )));
        };
        let e_rows = e.shape.get(1).copied().unwrap_or(0);
        if e.shape != [batch, e_rows, dim] || scale_slot >= e_rows || shift_slot >= e_rows {
            return Err(msg(format!(
                "ln_adaln_e_rope_half table {:?} for {:?}",
                e.shape, self.shape
            )));
        }
        let r = cos.shape.get(1).copied().unwrap_or(0);
        if cos.shape != [seq, r] || sin.shape != [seq, r] || r > dim || r % 2 != 0 {
            return Err(msg(format!(
                "ln_adaln_e_rope_half rope {:?}/{:?} for S={seq} D={dim}",
                cos.shape, sin.shape
            )));
        }
        #[cfg(feature = "cuda")]
        if fastvideo_models::nvfp4::from_env().is_some() {
            if let (Some(x), Some(ed), Some(c), Some(s)) =
                (self.dev()?, e.dev()?, cos.dev()?, sin.dev()?)
            {
                let out = super::ops::ln_adaln_e_rope_half_device(
                    &x, &ed, &c, &s, batch, heads, seq, dim, e_rows, scale_slot, shift_slot, r, eps,
                )?;
                return self.keep_dtype(Self::from_dev_result(out, self.shape.clone())?);
            }
        }
        let out = host::ln_adaln_e_rope_half(
            &self.host_cow()?,
            &e.host_cow()?,
            &cos.host_cow()?,
            &sin.host_cow()?,
            heads,
            seq,
            dim,
            e_rows,
            scale_slot,
            shift_slot,
            r,
            eps,
        );
        self.keep_dtype(Self::host_only(out, self.shape.clone()))
    }

    /// `self + update * e[:, slot]` (gated residual) for `[batch, seq, dim]`.
    /// With `FASTVIDEO_BF16_ACT` and a bf16 residual the result is bf16 with
    /// the eager rounding points: `bf16(h + bf16(update * gate))`.
    pub fn residual_gate_add_e(
        &self,
        update: &CudaTensor,
        e: &CudaTensor,
        slot: usize,
    ) -> Result<CudaTensor> {
        let [batch, seq, dim] = self.shape[..] else {
            return Err(msg(format!(
                "residual_gate_add_e expects [b, seq, dim], got {:?}",
                self.shape
            )));
        };
        let e_rows = e.shape.get(1).copied().unwrap_or(0);
        if update.shape != self.shape || e.shape != [batch, e_rows, dim] || slot >= e_rows {
            return Err(msg(format!(
                "residual_gate_add_e shapes {:?} {:?} {:?}",
                self.shape, update.shape, e.shape
            )));
        }
        let act16 = super::tensor::bf16_activations() && self.is_bf16();
        #[cfg(feature = "cuda")]
        if act16 && (self.act16_device() || update.act16_device()) {
            if let Some(t) =
                super::act16::residual_gate(self, update, e, (batch, seq, dim), e_rows, slot)?
            {
                return Ok(t);
            }
        }
        if act16 {
            let (h, a, g) = (self.host_cow()?, update.host_cow()?, e.host_cow()?);
            let out: Vec<f32> = (0..h.len())
                .map(|i| {
                    let (row, d) = (i / dim, i % dim);
                    let gate = g[((row / seq) * e_rows + slot) * dim + d];
                    super::quant::gate_residual_eager(h[i], gate, a[i])
                })
                .collect();
            return Ok(Self::host_only_dtype(
                out,
                self.shape.clone(),
                super::tensor::TensorDType::Bf16,
            ));
        }
        #[cfg(feature = "cuda")]
        if let (Some(h), Some(a), Some(ed)) = (self.dev()?, update.dev()?, e.dev()?) {
            let out =
                super::ops::residual_gate_add_e_device(&h, &a, &ed, batch, seq, dim, e_rows, slot)?;
            return Self::from_dev_result(out, self.shape.clone());
        }
        let out = host::residual_gate_add_e(
            &self.host_cow()?,
            &update.host_cow()?,
            &e.host_cow()?,
            seq,
            dim,
            e_rows,
            slot,
        );
        Ok(Self::host_only(out, self.shape.clone()))
    }

    /// Attention q/k from a (possibly fused) projection `self` `[b, seq, width]`:
    /// take columns `[col_off, col_off + heads*d)`, RMSNorm them with `weight`
    /// over the whole `heads*d`, apply RoPE per head, return BHSD.
    pub fn qk_norm_rope_bhsd(
        &self,
        col_off: usize,
        heads: usize,
        weight: &CudaTensor,
        rope: Option<Rope<'_>>,
        eps: f32,
    ) -> Result<CudaTensor> {
        let [batch, seq, width] = self.shape[..] else {
            return Err(msg(format!(
                "qk_norm_rope_bhsd expects [b, seq, w], got {:?}",
                self.shape
            )));
        };
        let proj = weight.numel();
        if heads == 0 || proj % heads != 0 || col_off + proj > width {
            return Err(msg(format!(
                "qk_norm_rope_bhsd: {heads} heads, weight {proj}, input {:?}",
                self.shape
            )));
        }
        let d = proj / heads;
        if let Some(r) = &rope {
            if r.cos.shape != [seq, d] || r.sin.shape != [seq, d] {
                return Err(msg(format!(
                    "rope tables {:?} for seq={seq} d={d}",
                    r.cos.shape
                )));
            }
        }
        let out_shape = vec![batch, heads, seq, d];
        #[cfg(feature = "cuda")]
        if let (Some(x), Some(w)) = (self.dev()?, weight.dev()?) {
            let tables = match &rope {
                Some(r) => Some((
                    r.cos.dev()?.ok_or_else(|| msg("rope cos upload"))?,
                    r.sin.dev()?.ok_or_else(|| msg("rope sin upload"))?,
                )),
                None => None,
            };
            let out = super::ops::qk_norm_rope_bhsd_device(
                &x,
                &w,
                tables.as_ref().map(|(c, s)| (&**c, &**s)),
                batch,
                seq,
                heads,
                d,
                width,
                col_off,
                eps,
            )?;
            return self.keep_dtype(Self::from_dev_result(out, out_shape)?);
        }
        let x = self.host_cow()?;
        let w = weight.host_cow()?;
        let tables = match &rope {
            Some(r) => Some((r.cos.host_cow()?, r.sin.host_cow()?)),
            None => None,
        };
        let out = host::qk_norm_rope_bhsd(
            &x,
            &w,
            tables.as_ref().map(|(c, s)| (&c[..], &s[..])),
            batch,
            seq,
            heads,
            d,
            width,
            col_off,
            eps,
        );
        self.keep_dtype(Self::host_only(out, out_shape))
    }

    /// Columns `[col_off, col_off + heads*d)` of `[b, seq, width]` as BHSD.
    pub fn split_heads_bhsd(&self, col_off: usize, heads: usize, d: usize) -> Result<CudaTensor> {
        let [batch, seq, width] = self.shape[..] else {
            return Err(msg(format!(
                "split_heads_bhsd expects [b, seq, w], got {:?}",
                self.shape
            )));
        };
        if col_off + heads * d > width {
            return Err(msg(format!(
                "split_heads_bhsd: {heads}x{d} at {col_off} beyond width {width}"
            )));
        }
        let out_shape = vec![batch, heads, seq, d];
        // A bf16 projection is split as bf16 (a copy, never widened).
        #[cfg(feature = "cuda")]
        if self.device_slice_bf16().is_some() {
            if let Some(t) =
                super::act16::split_heads(self, batch, seq, heads, d, width, col_off)?
            {
                return Ok(t);
            }
        }
        #[cfg(feature = "cuda")]
        if let Some(x) = self.dev()? {
            let out =
                super::ops::split_heads_bhsd_device(&x, batch, seq, heads, d, width, col_off)?;
            return self.keep_dtype(Self::from_dev_result(out, out_shape)?);
        }
        let out = host::split_heads_bhsd(&self.host_cow()?, batch, seq, heads, d, width, col_off);
        Ok(Self::host_only_dtype(out, out_shape, self.dtype()))
    }

    /// BHSD → `[b, seq, heads*d]`.
    pub fn merge_heads(&self) -> Result<CudaTensor> {
        let [batch, heads, seq, d] = self.shape[..] else {
            return Err(msg(format!(
                "merge_heads expects BHSD, got {:?}",
                self.shape
            )));
        };
        let out_shape = vec![batch, seq, heads * d];
        #[cfg(feature = "cuda")]
        if self.device_slice_bf16().is_some() {
            if let Some(t) = super::act16::merge_heads(self, batch, heads, seq, d)? {
                return Ok(t);
            }
        }
        #[cfg(feature = "cuda")]
        if let Some(x) = self.dev()? {
            return self.keep_dtype(Self::from_dev_result(
                super::ops::merge_heads_device(&x, batch, heads, seq, d)?,
                out_shape,
            )?);
        }
        Ok(Self::host_only_dtype(
            host::merge_heads(&self.host_cow()?, batch, heads, seq, d),
            out_shape,
            self.dtype(),
        ))
    }

    /// RMS over dim 1 of `[n, c, ...]` scaled by `gamma` `[c]` (Wan VAE norm).
    pub fn rms_norm_channels(&self, gamma: &CudaTensor, eps: f32) -> Result<CudaTensor> {
        self.rms_norm_channels_act(gamma, eps, false)
    }

    /// RMS norm over the channel axis, optionally with SiLU folded into the
    /// same pass. The VAE decoder always follows the norm with SiLU, and at
    /// decode resolution that separate pass is hundreds of MB read and written.
    pub fn rms_norm_channels_act(
        &self,
        gamma: &CudaTensor,
        eps: f32,
        silu: bool,
    ) -> Result<CudaTensor> {
        if self.rank() < 2 {
            return Err(msg("rms_norm_channels needs [n, c, ...]"));
        }
        let (n, c) = (self.shape[0], self.shape[1]);
        if gamma.numel() != c {
            return Err(msg(format!(
                "rms_norm_channels gamma {:?} for {:?}",
                gamma.shape, self.shape
            )));
        }
        let spatial: usize = self.shape[2..].iter().product();
        #[cfg(feature = "cuda")]
        if let (Some(x), Some(g)) = (self.dev()?, gamma.dev()?) {
            let out = super::ops::rms_norm_channels_device(&x, &g, n, c, spatial, eps, silu)?;
            return Self::from_dev_result(out, self.shape.clone());
        }
        let out = host::rms_norm_channels(
            &self.host_cow()?,
            &gamma.host_cow()?,
            n,
            c,
            spatial,
            eps,
            silu,
        );
        Ok(Self::host_only(out, self.shape.clone()))
    }

    /// Pair-rotate last dim of BSHD `[B,S,H,D]` with Flux2 tables `[S,D]`.
    /// Cos/sin use the even slot (Diffusers `repeat_interleave(2)`).
    pub fn apply_rotary_bshd(&self, cos: &CudaTensor, sin: &CudaTensor) -> Result<CudaTensor> {
        let [batch, seq, heads, d] = self.shape[..] else {
            return Err(msg(format!(
                "apply_rotary_bshd expects BSHD, got {:?}",
                self.shape
            )));
        };
        if d % 2 != 0 || cos.shape != [seq, d] || sin.shape != [seq, d] {
            return Err(msg(format!(
                "apply_rotary_bshd tables {:?} {:?} for {:?}",
                cos.shape, sin.shape, self.shape
            )));
        }
        #[cfg(feature = "cuda")]
        if let (Some(x), Some(c), Some(s)) = (self.dev()?, cos.dev()?, sin.dev()?) {
            let out = super::ops::apply_rotary_bshd_device(&x, &c, &s, batch, seq, heads, d)?;
            return Self::from_dev_result(out, self.shape.clone());
        }
        let out = host::apply_rotary_bshd(
            &self.host_cow()?,
            &cos.host_cow()?,
            &sin.host_cow()?,
            batch,
            seq,
            heads,
            d,
        );
        Ok(Self::host_only(out, self.shape.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: Vec<f32>, shape: &[usize]) -> CudaTensor {
        CudaTensor::from_vec(data, shape.to_vec()).unwrap()
    }

    fn vals(n: usize, k: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * k).sin() * 1.3) + 0.1).collect()
    }

    #[test]
    fn ln_adaln_e_matches_composed_ops() {
        let (b, s, d) = (2, 3, 5);
        let x = t(vals(b * s * d, 0.7), &[b, s, d]);
        let e = t(vals(b * 6 * d, 0.3), &[b, 6, d]);
        let got = x.ln_adaln_e(&e, 1, 0, 1e-6).unwrap();
        let ln = x.layer_norm(1e-6, None, None).unwrap();
        for i in 0..b * s * d {
            let (bi, j) = (i / (s * d), i % d);
            let want = ln.data[i] * (1.0 + e.data[(bi * 6 + 1) * d + j]) + e.data[bi * 6 * d + j];
            assert!((got.data[i] - want).abs() < 1e-5);
        }
        let upd = t(vals(b * s * d, 1.1), &[b, s, d]);
        let r = x.residual_gate_add_e(&upd, &e, 2).unwrap();
        for i in 0..b * s * d {
            let (bi, j) = (i / (s * d), i % d);
            assert!(
                (r.data[i] - (x.data[i] + upd.data[i] * e.data[(bi * 6 + 2) * d + j])).abs() < 1e-6
            );
        }
    }

    #[test]
    fn qk_norm_rope_matches_separate_steps() {
        let (b, s, heads, d) = (2usize, 4usize, 3usize, 4usize);
        let width = heads * d;
        // Fused projection: 3 columns wide, take the middle one.
        let x = t(vals(b * s * 3 * width, 0.37), &[b, s, 3 * width]);
        let w = t(
            vals(width, 0.9).iter().map(|v| 1.0 + 0.1 * v).collect(),
            &[width],
        );
        let angles = vals(s * d, 2.1);
        let cos = t(angles.iter().map(|a| a.cos()).collect(), &[s, d]);
        let sin = t(angles.iter().map(|a| a.sin()).collect(), &[s, d]);
        let got = x
            .qk_norm_rope_bhsd(
                width,
                heads,
                &w,
                Some(Rope {
                    cos: &cos,
                    sin: &sin,
                }),
                1e-6,
            )
            .unwrap();
        assert_eq!(got.shape, vec![b, heads, s, d]);
        let col = x.narrow(2, width, width).unwrap();
        let normed = col.rms_norm(&w, 1e-6).unwrap();
        for bi in 0..b {
            for si in 0..s {
                for h in 0..heads {
                    for p in (0..d).step_by(2) {
                        let base = (bi * s + si) * width + h * d + p;
                        let (x1, x2) = (normed.data[base], normed.data[base + 1]);
                        let (c, sn) = (cos.data[si * d + p], sin.data[si * d + p + 1]);
                        let o = ((bi * heads + h) * s + si) * d + p;
                        assert!((got.data[o] - (x1 * c - x2 * sn)).abs() < 1e-5);
                        assert!((got.data[o + 1] - (x1 * sn + x2 * c)).abs() < 1e-5);
                    }
                }
            }
        }
        let v = x.split_heads_bhsd(2 * width, heads, d).unwrap();
        let merged = v.merge_heads().unwrap();
        assert_eq!(merged.data, x.narrow(2, 2 * width, width).unwrap().data);
    }

    #[test]
    fn ln_adaln_e_rope_half_matches_the_two_ops() {
        let (b, h, s, d) = (2usize, 2usize, 3usize, 4usize);
        let x = t(vals(b * h * s * d, 0.7), &[b, h, s, d]);
        let e = t(vals(b * 6 * d, 0.3), &[b, 6, d]);
        let angles = vals(s * d, 2.1);
        let cos = t(angles.iter().map(|a| a.cos()).collect(), &[s, d]);
        let sin = t(angles.iter().map(|a| a.sin()).collect(), &[s, d]);
        let got = x.ln_adaln_e_rope_half(&e, 1, 0, 1e-6, &cos, &sin).unwrap();
        let flat = t(x.data.clone(), &[b, h * s, d]);
        let adaln = flat.ln_adaln_e(&e, 1, 0, 1e-6).unwrap();
        let want = t(adaln.data.clone(), &[b, h, s, d])
            .rope_half(&cos, &sin)
            .unwrap();
        assert_eq!(got.shape, want.shape);
        for i in 0..got.data.len() {
            assert!(
                (got.data[i] - want.data[i]).abs() < 1e-5,
                "elem {i}: {} vs {}",
                got.data[i],
                want.data[i]
            );
        }
    }

    #[test]
    fn apply_rotary_bshd_matches_pair_rotate() {
        let (b, s, h, d) = (1usize, 2usize, 2usize, 4usize);
        let xs = t((0..b * s * h * d).map(|i| i as f32 * 0.1).collect(), &[b, s, h, d]);
        let cs: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.2).cos()).collect();
        let sn: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.2).sin()).collect();
        let ct = t(cs.clone(), &[s, d]);
        let st = t(sn.clone(), &[s, d]);
        let got = xs.apply_rotary_bshd(&ct, &st).unwrap();
        let ones = t(vec![1.0; s * d], &[s, d]);
        let zeros = t(vec![0.0; s * d], &[s, d]);
        let id = xs.apply_rotary_bshd(&ones, &zeros).unwrap();
        assert_eq!(id.data, xs.data);
        assert_eq!(got.shape, xs.shape);
        assert_ne!(got.data, xs.data);
    }
}
