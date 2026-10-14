//! VDN-H3 (Video DeltaNet on MiniMax-H3): the hybrid attention's
//! configuration, window geometry, adapter key mapping, and a host reference
//! of every stage of the linear branch (f32 storage, f64 where the reference
//! accumulates in fp32 over long sums). See docs/ports/vdn-h3.md §1; `M:` cites
//! the remote code `modeling_vdn_h3.py` of `OpenVDN/vdn-minimax-h3`.
//!
//! Every function here is the arithmetic of the reference's eager (training)
//! body without bf16 rounding; the device kernels in
//! `fastvideo-cudarc::h3::vdn` round where the inference body rounds and are
//! checked against these stage by stage on identical inputs.

use std::path::{Path, PathBuf};

use super::packing::H3PackedLayout;

/// `linear_head_dim` of the released checkpoints (and H3's head dim).
pub const VDN_HEAD_DIM: usize = 128;
/// The separable short conv's taps per axis (`LinearAttentionSepConv.KERNEL`).
pub const VDN_CONV_TAPS: usize = 5;
/// `BidirectionalLinearBranch.TEXT_STATE_SCALE`: each scan starts from half the text state.
pub const VDN_TEXT_STATE_SCALE: f32 = 0.5;
/// The branch RMSNorm's eps (`RMSNorm(head_dim)` default).
pub const VDN_NORM_EPS: f32 = 1e-6;
/// `F.normalize(x, dim=-1, eps=1e-6)` on the features.
pub const VDN_L2_EPS: f32 = 1e-6;
/// `alpha.clamp_min(1e-12)` before the log-prefix of the bridge.
pub const VDN_ALPHA_FLOOR: f64 = 1e-12;
/// The DiT's attention heads (H3).
pub const VDN_HEADS: usize = 56;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// The `hybrid_attention` v2 transform (`linear_branch/config.json`,
/// `diffusers/config.json` `vdn.transform`). Only the released combination is
/// accepted: `vdn_solve`, `bridge alpha`, `a_fp32`, short conv on `k` and `v`,
/// softmax gate on, text state on, anchors `both`; chunk / radius / head dim
/// are read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VdnTransform {
    /// `softmax_attention.chunk`: frames per window chunk (0 = frame mode).
    pub chunk: usize,
    /// `softmax_attention.radius`: chunks (or frames) on each side.
    pub radius: usize,
    pub linear_head_dim: usize,
}

impl VdnTransform {
    /// The released transform: chunk 5, radius 1, head dim 128.
    pub fn released() -> Self {
        Self {
            chunk: 5,
            radius: 1,
            linear_head_dim: VDN_HEAD_DIM,
        }
    }

    /// Parse a transform object (the `config` of `linear_branch/config.json`
    /// or `vdn.transform` of `diffusers/config.json`). Anything but the
    /// released algorithm is refused rather than silently run differently.
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let get = |path: &[&str]| -> Option<&serde_json::Value> {
            let mut cur = v;
            for p in path {
                cur = cur.get(*p)?;
            }
            Some(cur)
        };
        let want_str = |path: &[&str], want: &str| -> Result<(), String> {
            match get(path).and_then(|x| x.as_str()) {
                Some(s) if s == want => Ok(()),
                other => Err(format!(
                    "vdn transform: {} = {other:?}, this port runs only {want:?}",
                    path.join(".")
                )),
            }
        };
        let want_bool = |path: &[&str], want: bool| -> Result<(), String> {
            match get(path).and_then(|x| x.as_bool()) {
                Some(b) if b == want => Ok(()),
                other => Err(format!(
                    "vdn transform: {} = {other:?}, this port runs only {want}",
                    path.join(".")
                )),
            }
        };
        want_str(&["anchor_frames"], "both")?;
        want_bool(&["enable_softmax_gate"], true)?;
        want_str(&["linear_attention", "delta_rule"], "vdn_solve")?;
        want_str(&["linear_attention", "bridge"], "alpha")?;
        want_bool(&["linear_attention", "a_fp32"], true)?;
        want_bool(&["linear_attention", "enable_text_state"], true)?;
        let targets: Vec<&str> = get(&["linear_attention", "short_conv", "targets"])
            .and_then(|t| t.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
            .unwrap_or_default();
        if targets != ["k", "v"] {
            return Err(format!(
                "vdn transform: short_conv.targets = {targets:?}, this port runs only [\"k\", \"v\"]"
            ));
        }
        let num = |path: &[&str]| -> Result<usize, String> {
            get(path)
                .and_then(|x| x.as_u64())
                .map(|n| n as usize)
                .ok_or_else(|| format!("vdn transform: {} missing", path.join(".")))
        };
        let out = Self {
            chunk: num(&["softmax_attention", "chunk"])?,
            radius: num(&["softmax_attention", "radius"])?,
            linear_head_dim: num(&["linear_attention", "linear_head_dim"])?,
        };
        if out.linear_head_dim != VDN_HEAD_DIM {
            return Err(format!(
                "vdn transform: linear_head_dim {} (the DiT's head dim is {VDN_HEAD_DIM})",
                out.linear_head_dim
            ));
        }
        Ok(out)
    }

    /// Per-frame inclusive window `[lo, hi]`, unclamped (`M:window_bounds`).
    pub fn bounds(&self, num_frames: usize) -> Vec<(i64, i64)> {
        window_bounds(num_frames, self.radius, self.chunk)
    }

    /// Whether every frame's window covers the whole clip: the softmax is
    /// then dense attention and the linear branch is skipped (it would count
    /// frames twice), `M:HybridAttention._hybrid_forward` `full_cover`.
    pub fn full_cover(&self, num_frames: usize) -> bool {
        self.bounds(num_frames)
            .iter()
            .all(|&(lo, hi)| lo <= 0 && hi >= num_frames as i64 - 1)
    }
}

/// `M:window_bounds`: frame mode (`chunk == 0`) `[t - r, t + r]`; chunk mode
/// `[(c - r) * K, (c + r + 1) * K - 1]` with `c = t / K`.
pub fn window_bounds(num_frames: usize, radius: usize, chunk: usize) -> Vec<(i64, i64)> {
    let r = radius as i64;
    (0..num_frames as i64)
        .map(|t| {
            if chunk == 0 {
                (t - r, t + r)
            } else {
                let k = chunk as i64;
                let c = t / k;
                ((c - r) * k, (c + r + 1) * k - 1)
            }
        })
        .collect()
}

/// A VDN checkpoint directory (`stage-b-step-2000/`, `stage-dmd-step-250/`):
/// what `diffusers/config.json`'s `vdn` block names, resolved to paths.
#[derive(Debug, Clone, PartialEq)]
pub struct VdnCheckpoint {
    pub dir: PathBuf,
    /// `linear_branch/model.safetensors`.
    pub branch: PathBuf,
    /// The LoRA adapters in merge order (`default`, then `turbo` on the DMD stage).
    pub adapters: Vec<PathBuf>,
    pub transform: VdnTransform,
}

/// The `diffusers/config.json` of a VDN checkpoint directory.
pub const VDN_CONFIG_REL: &str = "diffusers/config.json";

impl VdnCheckpoint {
    /// `dir` is a stage directory (holding `diffusers/config.json`), or a
    /// tree whose only stage directory is one level down
    /// (`vdn-h3-stage-b-2000/<rev>/stage-b-step-2000/`).
    pub fn open(dir: &Path) -> Result<Self, String> {
        let stage = resolve_stage_dir(dir)?;
        let path = stage.join(VDN_CONFIG_REL);
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_config_json(&stage, &text)
    }

    /// [`Self::open`] on a config's text (tests read the published files).
    pub fn from_config_json(stage: &Path, text: &str) -> Result<Self, String> {
        let doc: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("vdn config: {e}"))?;
        if doc.get("_class_name").and_then(|v| v.as_str()) != Some("VDNMiniMaxH3Transformer3DModel") {
            return Err("vdn config: _class_name is not VDNMiniMaxH3Transformer3DModel".into());
        }
        let vdn = doc.get("vdn").ok_or("vdn config: no `vdn` block")?;
        let rel = |v: &serde_json::Value| -> Result<PathBuf, String> {
            let s = v.as_str().ok_or("vdn config: a path is not a string")?;
            if s.starts_with('/') || s.split('/').any(|p| p == "..") {
                return Err(format!("vdn config: path {s:?} leaves the checkpoint directory"));
            }
            Ok(stage.join(s))
        };
        let branch = rel(vdn.get("branch").ok_or("vdn config: no branch")?)?;
        let adapters = vdn
            .get("adapters")
            .and_then(|a| a.as_array())
            .ok_or("vdn config: no adapters list")?
            .iter()
            .map(rel)
            .collect::<Result<Vec<_>, _>>()?;
        let base = vdn.get("base").ok_or("vdn config: no base")?;
        if base.get("subfolder").and_then(|s| s.as_str()) != Some("h3-base/transformer") {
            return Err("vdn config: base.subfolder is not h3-base/transformer".into());
        }
        let transform = VdnTransform::from_json(vdn.get("transform").ok_or("vdn config: no transform")?)?;
        Ok(Self {
            dir: stage.to_path_buf(),
            branch,
            adapters,
            transform,
        })
    }

    /// [`Self::open`], then every file the config names that is missing from
    /// `dir` taken from the same relative path under `shared`'s stage. The
    /// 8-step tree (`vdn-h3-dmd-250-turbo`) publishes only what differs from
    /// stage B (adapters/turbo and the JSON files); its linear branch and
    /// default adapter are stage B's byte-identical files. `shared` must carry
    /// the same transform.
    pub fn open_shared(dir: &Path, shared: Option<&Path>) -> Result<Self, String> {
        let mut ck = Self::open(dir)?;
        let Some(shared) = shared else {
            return Ok(ck);
        };
        let other = Self::open(shared)?;
        if other.transform != ck.transform {
            return Err(format!(
                "vdn: {} and {} have different transforms",
                ck.dir.display(),
                other.dir.display()
            ));
        }
        ck.relocate_missing(&other.dir, |p| p.is_file());
        Ok(ck)
    }

    /// Points each named file `exists` rejects at the same relative path
    /// under `shared` (tests pass their own `exists`).
    pub fn relocate_missing(&mut self, shared: &Path, exists: impl Fn(&Path) -> bool) {
        let stage = self.dir.clone();
        let fix = |p: &mut PathBuf| {
            if !exists(p) {
                if let Ok(rel) = p.strip_prefix(&stage) {
                    *p = shared.join(rel);
                }
            }
        };
        fix(&mut self.branch);
        self.adapters.iter_mut().for_each(fix);
    }

    /// Adapter names (`default`, `turbo`) in merge order.
    pub fn adapter_names(&self) -> Vec<String> {
        self.adapters
            .iter()
            .filter_map(|p| p.parent()?.file_name()?.to_str().map(str::to_string))
            .collect()
    }

    /// Whether this is the DMD-distilled 8-step model (it carries `turbo`).
    pub fn is_distilled(&self) -> bool {
        self.adapter_names().iter().any(|n| n == "turbo")
    }
}

fn resolve_stage_dir(dir: &Path) -> Result<PathBuf, String> {
    if dir.join(VDN_CONFIG_REL).is_file() {
        return Ok(dir.to_path_buf());
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join(VDN_CONFIG_REL).is_file())
        .collect();
    found.sort();
    match found.as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(format!(
            "{}: no VDN checkpoint ({VDN_CONFIG_REL} here or in one subdirectory)",
            dir.display()
        )),
        many => Err(format!(
            "{}: {} VDN checkpoints ({}); name the stage directory",
            dir.display(),
            many.len(),
            many.iter()
                .filter_map(|p| p.file_name()?.to_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// The base-model parameter a VDN adapter module names, and the adapter's
/// name: `transformer_blocks.3.attn.orig.to_q.lora_A.default.weight` ->
/// (`transformer_blocks.3.attn.to_q`, A, `default`). The hybrid wraps the
/// dense attention as `attn.orig` (`M:_resolve` re-roots `.attn.` the same
/// way). `None` for a key that is not a LoRA factor.
pub fn adapter_key(key: &str) -> Option<(String, LoraSideName, String)> {
    for (marker, side) in [(".lora_A.", LoraSideName::A), (".lora_B.", LoraSideName::B)] {
        if let Some((module, rest)) = key.split_once(marker) {
            let name = rest.strip_suffix(".weight")?;
            if name.is_empty() || name.contains('.') {
                return None;
            }
            let module = module.replacen(".attn.orig.", ".attn.", 1);
            return Some((module, side, name.to_string()));
        }
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraSideName {
    A,
    B,
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/// The packed sequence as the hybrid layers see it (`M:SequenceLayout`): the
/// target video rows (contiguous, frame-major) and the prompt rows. Every
/// other row (audio, FL2VA keyframe rows) is global.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VdnLayout {
    pub seq_len: usize,
    pub video_start: usize,
    pub num_frames: usize,
    pub tokens_per_frame: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    pub text_start: usize,
    pub text_len: usize,
}

impl VdnLayout {
    pub fn from_packed(l: &H3PackedLayout) -> Result<Self, String> {
        let (t, h, w) = l.token_grid;
        if t * h * w != l.video.len {
            return Err(format!(
                "vdn layout: token grid {t}x{h}x{w} != {} video rows",
                l.video.len
            ));
        }
        if !l.ref_segments.is_empty() {
            return Err("vdn layout: Ref2VA was not trained for VDN-H3".into());
        }
        if l.text.len == 0 {
            return Err("vdn layout: no prompt rows (the linear branch's text state reads them)".into());
        }
        Ok(Self {
            seq_len: l.sequence_length(),
            video_start: l.video.start,
            num_frames: t,
            tokens_per_frame: h * w,
            grid_h: h,
            grid_w: w,
            text_start: l.text.start,
            text_len: l.text.len,
        })
    }

    pub fn video_end(&self) -> usize {
        self.video_start + self.num_frames * self.tokens_per_frame
    }

    /// The rows the linear branch reads: frames `1..F-1` (the anchors are
    /// exact softmax), `(start, len)`; empty when `F <= 2`.
    pub fn inner_rows(&self) -> (usize, usize) {
        if self.num_frames <= 2 {
            return (self.video_start + self.tokens_per_frame, 0);
        }
        (
            self.video_start + self.tokens_per_frame,
            (self.num_frames - 2) * self.tokens_per_frame,
        )
    }

    /// Whether query row `q` attends to key row `k` in the window softmax
    /// (`M:build_window_block_mask`'s `mask_mod`, anchors `both`).
    pub fn allows(&self, bounds: &[(i64, i64)], q: usize, k: usize) -> bool {
        let (vs, ve, tpf) = (self.video_start, self.video_end(), self.tokens_per_frame);
        let qv = (vs..ve).contains(&q);
        let kv = (vs..ve).contains(&k);
        if !(qv && kv) {
            return true;
        }
        let last = self.num_frames - 1;
        let (qf, kf) = ((q - vs) / tpf, (k - vs) / tpf);
        if kf == 0 || kf == last || qf == 0 || qf == last {
            return true;
        }
        let (lo, hi) = bounds[qf];
        (lo..=hi).contains(&(kf as i64))
    }
}

/// The inner frames' windows rebased by one (`M:BidirectionalLinearBranch.forward`
/// `skip_ends`): inner frame `i` is frame `i + 1`, its window `(lo - 1, hi - 1)`.
pub fn inner_bounds(bounds: &[(i64, i64)]) -> Vec<(i64, i64)> {
    if bounds.len() <= 2 {
        return Vec::new();
    }
    bounds[1..bounds.len() - 1]
        .iter()
        .map(|&(lo, hi)| (lo - 1, hi - 1))
        .collect()
}

/// The gather's index plan per inner frame (`M:_gather_indices`): which
/// prefix / suffix state, and the alpha span bridged on each side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatherRow {
    /// `P[before]`, or the text state when `None`.
    pub before: Option<usize>,
    /// `Q[after]`, or the text state when `None`.
    pub after: Option<usize>,
    /// Frames `[before_from, t]` decay the left state into `t`.
    pub before_from: usize,
    /// Frames `[t, after_to]` decay the right state into `t`.
    pub after_to: usize,
}

pub fn gather_plan(inner: &[(i64, i64)]) -> Vec<GatherRow> {
    let n = inner.len() as i64;
    inner
        .iter()
        .enumerate()
        .map(|(t, &(lo, hi))| {
            let b = lo - 1;
            let a = hi + 1;
            GatherRow {
                before: (b >= 0).then_some(b as usize),
                after: (a < n).then_some(a as usize),
                before_from: lo.max(0) as usize,
                after_to: (hi.min(n - 1)).max(t as i64) as usize,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Host reference of the branch
// ---------------------------------------------------------------------------

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `F.softplus(x)` (beta 1, threshold 20).
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// `silu`, then (if `l2`) per-head L2 normalisation: `[rows, heads * d]`.
pub fn activate(x: &[f32], heads: usize, d: usize, l2: bool) -> Vec<f32> {
    let mut out: Vec<f32> = x.iter().map(|&v| silu(v)).collect();
    if l2 {
        for head in out.chunks_mut(d) {
            let n: f64 = head.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            let inv = 1.0 / n.sqrt().max(f64::from(VDN_L2_EPS));
            for v in head {
                *v = (f64::from(*v) * inv) as f32;
            }
        }
    }
    let _ = heads;
    out
}

/// Depthwise 5x5 spatial conv per frame, zero padding, no bias
/// (`F.conv2d(..., padding=2, groups=C)`): `x [frames, gh, gw, C]`,
/// `w [C, 5, 5]`.
pub fn spatial_conv(x: &[f32], frames: usize, gh: usize, gw: usize, c: usize, w: &[f32]) -> Vec<f32> {
    let k = VDN_CONV_TAPS as i64;
    let p = k / 2;
    let mut out = vec![0.0f32; frames * gh * gw * c];
    for f in 0..frames {
        for y in 0..gh as i64 {
            for xx in 0..gw as i64 {
                let o = ((f * gh + y as usize) * gw + xx as usize) * c;
                for ch in 0..c {
                    let mut acc = 0.0f64;
                    for dy in 0..k {
                        let yy = y + dy - p;
                        if yy < 0 || yy >= gh as i64 {
                            continue;
                        }
                        for dx in 0..k {
                            let xs = xx + dx - p;
                            if xs < 0 || xs >= gw as i64 {
                                continue;
                            }
                            let i = ((f * gh + yy as usize) * gw + xs as usize) * c + ch;
                            acc += f64::from(w[ch * 25 + (dy * k + dx) as usize]) * f64::from(x[i]);
                        }
                    }
                    out[o + ch] = acc as f32;
                }
            }
        }
    }
    out
}

/// Depthwise 5-tap temporal conv over frames, zero padding, no bias:
/// `out[t] = sum_dt w[dt] x[t + dt - 2]`; `x [frames, S, C]`, `w [C, 5]`.
pub fn temporal_conv(x: &[f32], frames: usize, s: usize, c: usize, w: &[f32]) -> Vec<f32> {
    let k = VDN_CONV_TAPS as i64;
    let p = k / 2;
    let mut out = vec![0.0f32; frames * s * c];
    for t in 0..frames as i64 {
        for si in 0..s {
            for ch in 0..c {
                let mut acc = 0.0f64;
                for dt in 0..k {
                    let tt = t + dt - p;
                    if tt < 0 || tt >= frames as i64 {
                        continue;
                    }
                    acc += f64::from(w[ch * 5 + dt as usize])
                        * f64::from(x[(tt as usize * s + si) * c + ch]);
                }
                out[(t as usize * s + si) * c + ch] = acc as f32;
            }
        }
    }
    out
}

/// Per-chunk statistics (`M:frame_statistics`): over each chunk's `s` rows,
/// `A[f,h] = sum beta k k^T` (symmetrized) and `B[f,h] = sum (beta v) k^T`
/// (`[d_v, d_k]`). `k`, `v`: `[chunks * s, H * d]`; `beta`: `[chunks * s, H]`.
pub fn frame_statistics(
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    chunks: usize,
    s: usize,
    heads: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mat = d * d;
    let mut a = vec![0.0f32; chunks * heads * mat];
    let mut b = vec![0.0f32; chunks * heads * mat];
    let c = heads * d;
    for f in 0..chunks {
        for h in 0..heads {
            let mut acc_a = vec![0.0f64; mat];
            let mut acc_b = vec![0.0f64; mat];
            for si in 0..s {
                let row = f * s + si;
                let be = f64::from(beta[row * heads + h]);
                let kr = &k[row * c + h * d..][..d];
                let vr = &v[row * c + h * d..][..d];
                for i in 0..d {
                    let bk = be * f64::from(kr[i]);
                    let bv = be * f64::from(vr[i]);
                    for j in 0..d {
                        acc_a[i * d + j] += bk * f64::from(kr[j]);
                        acc_b[i * d + j] += bv * f64::from(kr[j]);
                    }
                }
            }
            let base = (f * heads + h) * mat;
            for i in 0..d {
                for j in 0..d {
                    a[base + i * d + j] = (0.5 * (acc_a[i * d + j] + acc_a[j * d + i])) as f32;
                    b[base + i * d + j] = acc_b[i * d + j] as f32;
                }
            }
        }
    }
    (a, b)
}

/// `(I + A)^{-1}` of one SPD `d x d` matrix, by Cholesky in f64.
pub fn spd_inverse(a: &[f32], d: usize) -> Result<Vec<f64>, String> {
    let mut l = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = f64::from(a[i * d + j]) + if i == j { 1.0 } else { 0.0 };
            for p in 0..j {
                s -= l[i * d + p] * l[j * d + p];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(format!("vdn solve: I + A is not positive definite (pivot {i}: {s})"));
                }
                l[i * d + i] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    // L^{-1}, lower triangular.
    let mut li = vec![0.0f64; d * d];
    for j in 0..d {
        li[j * d + j] = 1.0 / l[j * d + j];
        for i in j + 1..d {
            let mut s = 0.0;
            for p in j..i {
                s += l[i * d + p] * li[p * d + j];
            }
            li[i * d + j] = -s / l[i * d + i];
        }
    }
    // (I + A)^{-1} = L^{-T} L^{-1}.
    let mut inv = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0;
            for p in i.max(j)..d {
                s += li[p * d + i] * li[p * d + j];
            }
            inv[i * d + j] = s;
        }
    }
    Ok(inv)
}

/// `vdn_solve`'s factors (`M:VdnDelta.factor_apply`): per matrix,
/// `T = Diag(alpha) (I + A)^{-1}` (row `i` scaled by `alpha[i]`) and
/// `J = B (I + A)^{-1}`. `alpha`: `[n, d]` (`None` = ones, the text chunk).
pub fn vdn_factors(
    a: &[f32],
    b: &[f32],
    alpha: Option<&[f32]>,
    n: usize,
    d: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let mat = d * d;
    let mut t = vec![0.0f32; n * mat];
    let mut jv = vec![0.0f32; n * mat];
    for m in 0..n {
        let inv = spd_inverse(&a[m * mat..][..mat], d)?;
        for i in 0..d {
            let al = alpha.map_or(1.0, |al| f64::from(al[m * d + i]));
            for j in 0..d {
                t[m * mat + i * d + j] = (al * inv[i * d + j]) as f32;
            }
        }
        for v in 0..d {
            for j in 0..d {
                let mut s = 0.0f64;
                for i in 0..d {
                    s += f64::from(b[m * mat + v * d + i]) * inv[i * d + j];
                }
                jv[m * mat + v * d + j] = s as f32;
            }
        }
    }
    Ok((t, jv))
}

/// KDA's decay (`M:FrameKDAAlpha`): `frame_mean [F, hidden]` ->
/// `alpha [F, H, d]` = `exp(-exp(A_log[h]) * softplus(up(down(m)) + dt_bias))`.
#[allow(clippy::too_many_arguments)]
pub fn kda_alpha(
    frame_mean: &[f32],
    frames: usize,
    hidden: usize,
    down: &[f32],
    up: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    heads: usize,
    d: usize,
) -> Vec<f32> {
    let rank = down.len() / hidden;
    let c = heads * d;
    let mut out = vec![0.0f32; frames * c];
    for f in 0..frames {
        let m = &frame_mean[f * hidden..][..hidden];
        let z: Vec<f64> = (0..rank)
            .map(|r| {
                down[r * hidden..][..hidden]
                    .iter()
                    .zip(m)
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum()
            })
            .collect();
        for ch in 0..c {
            let u: f64 = up[ch * rank..][..rank]
                .iter()
                .zip(&z)
                .map(|(&w, &x)| f64::from(w) * x)
                .sum::<f64>()
                + f64::from(dt_bias[ch]);
            let scale = f64::from(a_log[ch / d]).exp();
            out[f * c + ch] = (-scale * softplus(u)).exp() as f32;
        }
    }
    out
}

/// The two scans (`M:_run_scans`): `P[f] = P[f-1] T_f + J_f` from `P[-1] =
/// start`, and `Q[f] = Q[f+1] T_f + J_f` from `Q[F] = start`; states are
/// `[d_v, d_k]` per head, banks `[F, H, d, d]`.
pub fn scans(t: &[f32], j: &[f32], start: &[f32], frames: usize, heads: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let mat = d * d;
    let step = |state: &[f64], f: usize, h: usize| -> Vec<f64> {
        let base = (f * heads + h) * mat;
        let mut out = vec![0.0f64; mat];
        for v in 0..d {
            for c in 0..d {
                let mut s = f64::from(j[base + v * d + c]);
                for i in 0..d {
                    s += state[v * d + i] * f64::from(t[base + i * d + c]);
                }
                out[v * d + c] = s;
            }
        }
        out
    };
    let mut p = vec![0.0f32; frames * heads * mat];
    let mut q = vec![0.0f32; frames * heads * mat];
    for h in 0..heads {
        let s0: Vec<f64> = start[h * mat..][..mat].iter().map(|&x| f64::from(x)).collect();
        let mut s = s0.clone();
        for f in 0..frames {
            s = step(&s, f, h);
            for (o, &v) in p[(f * heads + h) * mat..][..mat].iter_mut().zip(&s) {
                *o = v as f32;
            }
        }
        let mut s = s0;
        for f in (0..frames).rev() {
            s = step(&s, f, h);
            for (o, &v) in q[(f * heads + h) * mat..][..mat].iter_mut().zip(&s) {
                *o = v as f32;
            }
        }
    }
    (p, q)
}

/// The boundary gather with the alpha bridge (`M:gather_linear_state`,
/// `text_state` given): per inner frame `t`, the state just outside its
/// window on both sides, decayed column-wise (over `d_k`) into `t`.
pub fn gather(
    p: &[f32],
    q: &[f32],
    alpha: &[f32],
    start: &[f32],
    plan: &[GatherRow],
    heads: usize,
    d: usize,
) -> Vec<f32> {
    let frames = plan.len();
    let mat = d * d;
    let c = heads * d;
    // log-prefix of alpha: lp[f] = sum_{u < f} log(max(alpha_u, 1e-12)), [F+1, H*d]
    let mut lp = vec![0.0f64; (frames + 1) * c];
    for f in 0..frames {
        for ch in 0..c {
            lp[(f + 1) * c + ch] = lp[f * c + ch] + f64::from(alpha[f * c + ch]).max(VDN_ALPHA_FLOOR).ln();
        }
    }
    let mut out = vec![0.0f32; frames * heads * mat];
    for (t, row) in plan.iter().enumerate() {
        for h in 0..heads {
            let before = match row.before {
                Some(b) => &p[(b * heads + h) * mat..][..mat],
                None => &start[h * mat..][..mat],
            };
            let after = match row.after {
                Some(a) => &q[(a * heads + h) * mat..][..mat],
                None => &start[h * mat..][..mat],
            };
            let o = &mut out[(t * heads + h) * mat..][..mat];
            for kk in 0..d {
                let ch = h * d + kk;
                let db = (lp[(t + 1) * c + ch] - lp[row.before_from * c + ch]).exp();
                let da = (lp[(row.after_to + 1) * c + ch] - lp[t * c + ch]).exp();
                for v in 0..d {
                    o[v * d + kk] = (f64::from(before[v * d + kk]) * db
                        + f64::from(after[v * d + kk]) * da) as f32;
                }
            }
        }
    }
    out
}

/// The readout, RMSNorm and output gate (`M:_readout` step 5): per inner
/// token `r = state_t q`, `RMSNorm(d)` with `weight`, times `gate`.
/// `q`, `gate`: `[frames * s, H * d]`; `state`: `[frames, H, d, d]`.
#[allow(clippy::too_many_arguments)]
pub fn readout(
    state: &[f32],
    q: &[f32],
    gate: &[f32],
    weight: &[f32],
    frames: usize,
    s: usize,
    heads: usize,
    d: usize,
) -> Vec<f32> {
    let c = heads * d;
    let mat = d * d;
    let mut out = vec![0.0f32; frames * s * c];
    for f in 0..frames {
        for si in 0..s {
            let row = f * s + si;
            for h in 0..heads {
                let st = &state[(f * heads + h) * mat..][..mat];
                let qr = &q[row * c + h * d..][..d];
                let r: Vec<f64> = (0..d)
                    .map(|v| {
                        st[v * d..][..d]
                            .iter()
                            .zip(qr)
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum()
                    })
                    .collect();
                let ms = r.iter().map(|x| x * x).sum::<f64>() / d as f64;
                let rs = 1.0 / (ms + f64::from(VDN_NORM_EPS)).sqrt();
                for v in 0..d {
                    out[row * c + h * d + v] = (r[v] * rs * f64::from(weight[v])
                        * f64::from(gate[row * c + h * d + v])) as f32;
                }
            }
        }
    }
    out
}

/// One branch's weights as host slices (the `linear_branch/model.safetensors`
/// tensors of one block, `transformer_blocks.N.attn.*`).
pub struct BranchWeightsRef<'a> {
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub alpha_down: &'a [f32],
    pub alpha_up: &'a [f32],
    pub beta_proj: &'a [f32],
    pub norm: &'a [f32],
    pub og_down: &'a [f32],
    pub og_up: &'a [f32],
    pub og_up_bias: &'a [f32],
    pub k_sp: &'a [f32],
    pub k_tm: &'a [f32],
    pub v_sp: &'a [f32],
    pub v_tm: &'a [f32],
}

fn linear(x: &[f32], w: &[f32], bias: Option<&[f32]>, rows: usize, inn: usize, out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out];
    for r in 0..rows {
        let xr = &x[r * inn..][..inn];
        for o in 0..out {
            let s: f64 = w[o * inn..][..inn]
                .iter()
                .zip(xr)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            y[r * out + o] = (s + bias.map_or(0.0, |b| f64::from(b[o]))) as f32;
        }
    }
    y
}

/// The branch inputs of one layer: the attention input rows and raw
/// (pre-norm, pre-RoPE) projections of the inner video frames and of the
/// prompt, all `[rows, width]` row-major.
pub struct BranchInputs<'a> {
    pub x_inner: &'a [f32],
    pub q_inner: &'a [f32],
    pub k_inner: &'a [f32],
    pub v_inner: &'a [f32],
    pub x_text: &'a [f32],
    pub k_text: &'a [f32],
    pub v_text: &'a [f32],
}

/// The whole linear branch for one layer (`M:BidirectionalLinearBranch.
/// _readout` on the inner frames): `[inner rows, H * d]`, gated and normed,
/// ready for `to_out_linear`. `inner` is [`inner_bounds`].
#[allow(clippy::too_many_arguments)]
pub fn reference_branch(
    w: &BranchWeightsRef<'_>,
    x: &BranchInputs<'_>,
    frames: usize,
    grid: (usize, usize),
    text_len: usize,
    hidden: usize,
    heads: usize,
    d: usize,
    inner: &[(i64, i64)],
) -> Result<Vec<f32>, String> {
    let c = heads * d;
    let s = grid.0 * grid.1;
    let n = frames * s;
    let rank = w.og_down.len() / hidden;
    // 1. features
    let q = activate(x.q_inner, heads, d, true);
    let k = activate(
        &temporal_conv(&spatial_conv(x.k_inner, frames, grid.0, grid.1, c, w.k_sp), frames, s, c, w.k_tm),
        heads,
        d,
        true,
    );
    let v = activate(
        &temporal_conv(&spatial_conv(x.v_inner, frames, grid.0, grid.1, c, w.v_sp), frames, s, c, w.v_tm),
        heads,
        d,
        false,
    );
    // 2. beta, the output gate
    let beta: Vec<f32> = linear(x.x_inner, w.beta_proj, None, n, hidden, heads)
        .into_iter()
        .map(sigmoid)
        .collect();
    let og_mid = linear(x.x_inner, w.og_down, None, n, hidden, rank);
    let gate: Vec<f32> = linear(&og_mid, w.og_up, Some(w.og_up_bias), n, rank, c)
        .into_iter()
        .map(sigmoid)
        .collect();
    // 3. statistics, 4. decay
    let (a, b) = frame_statistics(&k, &v, &beta, frames, s, heads, d);
    let mut mean = vec![0.0f32; frames * hidden];
    for f in 0..frames {
        for col in 0..hidden {
            let sum: f64 = (0..s).map(|si| f64::from(x.x_inner[(f * s + si) * hidden + col])).sum();
            mean[f * hidden + col] = (sum / s as f64) as f32;
        }
    }
    let alpha = kda_alpha(&mean, frames, hidden, w.alpha_down, w.alpha_up, w.dt_bias, w.a_log, heads, d);
    // 6. the text state
    let kt = activate(x.k_text, heads, d, true);
    let vt = activate(x.v_text, heads, d, false);
    let bt: Vec<f32> = linear(x.x_text, w.beta_proj, None, text_len, hidden, heads)
        .into_iter()
        .map(sigmoid)
        .collect();
    let (at, btm) = frame_statistics(&kt, &vt, &bt, 1, text_len, heads, d);
    let (_, jt) = vdn_factors(&at, &btm, None, heads, d)?;
    let start: Vec<f32> = jt.iter().map(|&v| VDN_TEXT_STATE_SCALE * v).collect();
    // 5, 7. factors and scans
    let (tf, jf) = vdn_factors(&a, &b, Some(&alpha), frames * heads, d)?;
    let (p, qb) = scans(&tf, &jf, &start, frames, heads, d);
    // 8. gather, 9. readout
    let state = gather(&p, &qb, &alpha, &start, &gather_plan(inner), heads, d);
    Ok(readout(&state, &q, &gate, w.norm, frames, s, heads, d))
}

/// Dense reference of the window softmax on `[S, H, d]` q / k / v
/// (already normed and roped): exact softmax over [`VdnLayout::allows`].
pub fn reference_window_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    layout: &VdnLayout,
    bounds: &[(i64, i64)],
    heads: usize,
    d: usize,
) -> Vec<f32> {
    let s = layout.seq_len;
    let c = heads * d;
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0.0f32; s * c];
    let mut p = vec![0.0f64; s];
    for h in 0..heads {
        for i in 0..s {
            let qi = &q[i * c + h * d..][..d];
            let mut mx = f64::NEG_INFINITY;
            for (jj, pj) in p.iter_mut().enumerate() {
                *pj = if layout.allows(bounds, i, jj) {
                    let kj = &k[jj * c + h * d..][..d];
                    qi.iter().zip(kj).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum::<f64>() * scale
                } else {
                    f64::NEG_INFINITY
                };
                mx = mx.max(*pj);
            }
            let mut sum = 0.0;
            for pj in p.iter_mut() {
                *pj = if pj.is_finite() { (*pj - mx).exp() } else { 0.0 };
                sum += *pj;
            }
            for e in 0..d {
                let acc: f64 = (0..s).map(|jj| p[jj] * f64::from(v[jj * c + h * d + e])).sum();
                out[i * c + h * d + e] = (acc / sum) as f32;
            }
        }
    }
    out
}

/// Recipe names of the VDN contracts: `vdn-h3-8step` (the 8-step model, 9
/// grid points), `vdn-h3-50step` (the stage-B model's 50 NFE) and
/// `vdn-h3-<N>step` in general; `vdn-h3` is the 50-step model. The forwards,
/// or `None` for another name.
pub fn vdn_recipe_forwards(name: &str) -> Option<usize> {
    if name == "vdn-h3" {
        return Some(50);
    }
    let n: usize = name.strip_prefix("vdn-h3-")?.strip_suffix("step")?.parse().ok()?;
    (1..=99).contains(&n).then_some(n)
}

pub fn is_vdn_recipe(name: &str) -> bool {
    vdn_recipe_forwards(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    fn rand(n: usize, seed: &mut u64, scale: f32) -> Vec<f32> {
        (0..n).map(|_| lcg(seed) * scale).collect()
    }

    #[test]
    fn chunk_windows_are_whole_chunks() {
        let b = window_bounds(12, 1, 5);
        // frames 0..4 are chunk 0: chunks -1..1 -> frames -5..9
        for t in 0..5 {
            assert_eq!(b[t], (-5, 9));
        }
        for t in 5..10 {
            assert_eq!(b[t], (0, 14));
        }
        for t in 10..12 {
            assert_eq!(b[t], (5, 19));
        }
        assert_eq!(window_bounds(4, 1, 0), vec![(-1, 1), (0, 2), (1, 3), (2, 4)]);
        let tr = VdnTransform::released();
        assert!(tr.full_cover(10) && !tr.full_cover(11));
        // the 14.4 s clip: 102 latent frames, chunk 20 holds 100 and 101
        let b = tr.bounds(102);
        assert_eq!(b[101], (95, 109));
    }

    #[test]
    fn the_published_transform_parses_and_others_are_refused() {
        let json: serde_json::Value = serde_json::from_str(
            r#"{"anchor_frames":"both","enable_softmax_gate":true,"linear_attention":{"a_fp32":true,
            "bridge":"alpha","delta_rule":"vdn_solve","enable_text_state":true,"linear_head_dim":128,
            "short_conv":{"targets":["k","v"]}},"softmax_attention":{"chunk":5,"radius":1}}"#,
        )
        .unwrap();
        assert_eq!(VdnTransform::from_json(&json).unwrap(), VdnTransform::released());
        for (path, bad) in [
            ("delta_rule", r#""sana_scaled""#),
            ("bridge", r#""none""#),
            ("enable_text_state", "false"),
        ] {
            let mut j = json.clone();
            j["linear_attention"][path] = serde_json::from_str(bad).unwrap();
            assert!(VdnTransform::from_json(&j).is_err(), "{path} = {bad}");
        }
        let mut j = json.clone();
        j["anchor_frames"] = "columns".into();
        assert!(VdnTransform::from_json(&j).is_err());
    }

    #[test]
    fn adapter_keys_reroot_the_hybrid_attention() {
        assert_eq!(
            adapter_key("transformer_blocks.3.attn.orig.to_q.lora_A.default.weight"),
            Some(("transformer_blocks.3.attn.to_q".into(), LoraSideName::A, "default".into()))
        );
        assert_eq!(
            adapter_key("token_refiner.refiner_blocks.1.attn.to_out.0.lora_B.turbo.weight"),
            Some((
                "token_refiner.refiner_blocks.1.attn.to_out.0".into(),
                LoraSideName::B,
                "turbo".into()
            ))
        );
        assert_eq!(
            adapter_key("transformer_blocks.0.adaln_proj.linear.lora_A.turbo.weight").unwrap().0,
            "transformer_blocks.0.adaln_proj.linear"
        );
        assert_eq!(adapter_key("transformer_blocks.0.attn.to_out_linear.weight"), None);
    }

    /// The 8-step tree holds only adapters/turbo (and JSON files): the branch
    /// and the default adapter come from stage B at the same relative paths.
    #[test]
    fn the_turbo_tree_reads_stage_b_files_it_does_not_carry() {
        let cfg = r#"{"_class_name":"VDNMiniMaxH3Transformer3DModel","vdn":{
            "adapters":["adapters/default/adapter_model.safetensors","adapters/turbo/adapter_model.safetensors"],
            "base":{"revision":null,"source":"OpenVDN/vdn-minimax-h3","subfolder":"h3-base/transformer"},
            "branch":"linear_branch/model.safetensors",
            "transform":{"anchor_frames":"both","enable_softmax_gate":true,"linear_attention":{"a_fp32":true,
            "bridge":"alpha","delta_rule":"vdn_solve","enable_text_state":true,"linear_head_dim":128,
            "short_conv":{"targets":["k","v"]}},"softmax_attention":{"chunk":5,"radius":1}}}}"#;
        let dmd = Path::new("/w/vdn-h3-dmd-250-turbo/stage-dmd-step-250");
        let b = Path::new("/w/vdn-h3-stage-b-2000/stage-b-step-2000");
        let mut ck = VdnCheckpoint::from_config_json(dmd, cfg).unwrap();
        assert!(ck.is_distilled());
        ck.relocate_missing(b, |p| p.to_str().unwrap().contains("/turbo/"));
        assert_eq!(ck.branch, b.join("linear_branch/model.safetensors"));
        assert_eq!(
            ck.adapters,
            [
                b.join("adapters/default/adapter_model.safetensors"),
                dmd.join("adapters/turbo/adapter_model.safetensors")
            ]
        );
        assert_eq!(ck.adapter_names(), ["default", "turbo"]);
    }

    #[test]
    fn recipes() {
        assert_eq!(vdn_recipe_forwards("vdn-h3-8step"), Some(8));
        assert_eq!(vdn_recipe_forwards("vdn-h3-50step"), Some(50));
        assert_eq!(vdn_recipe_forwards("vdn-h3"), Some(50));
        assert_eq!(vdn_recipe_forwards("vdn-h3-0step"), None);
        assert_eq!(vdn_recipe_forwards("8step"), None);
    }

    fn layout(frames: usize, tpf: usize) -> VdnLayout {
        VdnLayout {
            seq_len: 3 + 2 + frames * tpf,
            video_start: 5,
            num_frames: frames,
            tokens_per_frame: tpf,
            grid_h: 1,
            grid_w: tpf,
            text_start: 0,
            text_len: 3,
        }
    }

    #[test]
    fn the_mask_keeps_globals_anchors_and_the_window() {
        let l = layout(12, 2);
        let b = window_bounds(12, 1, 5);
        let row = |f: usize, i: usize| l.video_start + f * 2 + i;
        // global rows see and are seen by everything
        for k in 0..l.seq_len {
            assert!(l.allows(&b, 0, k) && l.allows(&b, k, 4));
        }
        // frame 2 (chunk 0, window -5..9) does not see frame 10, sees the anchors
        assert!(!l.allows(&b, row(2, 0), row(10, 1)));
        assert!(l.allows(&b, row(2, 0), row(9, 1)));
        assert!(l.allows(&b, row(2, 1), row(11, 0)) && l.allows(&b, row(2, 1), row(0, 0)));
        // the anchor frames' rows see everyone
        assert!(l.allows(&b, row(11, 0), row(2, 0)) && l.allows(&b, row(0, 1), row(10, 0)));
        // frame 10 (chunk 2, window 5..19) sees 5, not 4
        assert!(l.allows(&b, row(10, 0), row(5, 0)) && !l.allows(&b, row(10, 0), row(4, 1)));
    }

    #[test]
    fn the_solve_meets_the_normal_equation() {
        // S_new (I + A) = S Diag(alpha) + B, the frame-wise regularized delta rule.
        let d = 6;
        let mut seed = 7u64;
        let k = rand(9 * d, &mut seed, 1.0);
        let k = activate(&k, 1, d, true);
        let v = rand(9 * d, &mut seed, 1.0);
        let beta: Vec<f32> = rand(9, &mut seed, 1.0).iter().map(|x| 0.5 + 0.4 * x).collect();
        let (a, b) = frame_statistics(&k, &v, &beta, 1, 9, 1, d);
        let alpha: Vec<f32> = rand(d, &mut seed, 0.4).iter().map(|x| 0.5 + x).collect();
        let (t, j) = vdn_factors(&a, &b, Some(&alpha), 1, d).unwrap();
        let s = rand(d * d, &mut seed, 1.0);
        let (p, _) = scans(&t, &j, &s, 1, 1, d);
        for vrow in 0..d {
            for c in 0..d {
                let lhs: f64 = (0..d)
                    .map(|i| f64::from(p[vrow * d + i]) * (f64::from(a[i * d + c]) + if i == c { 1.0 } else { 0.0 }))
                    .sum();
                let rhs = f64::from(s[vrow * d + c]) * f64::from(alpha[c]) + f64::from(b[vrow * d + c]);
                assert!((lhs - rhs).abs() < 1e-4, "{lhs} vs {rhs}");
            }
        }
    }

    #[test]
    fn the_gather_reads_the_complement_of_the_window() {
        // alpha = 1, a zero text state: state_t = sum of the injections of every
        // frame outside the window, carried through the transitions in between.
        let (frames, heads, d) = (9usize, 1usize, 3usize);
        let mut seed = 11u64;
        let mat = d * d;
        let tm: Vec<f32> = (0..frames)
            .flat_map(|_| (0..mat).map(|i| if i % (d + 1) == 0 { 1.0 } else { 0.0 }).collect::<Vec<_>>())
            .collect();
        let jm = rand(frames * mat, &mut seed, 1.0);
        let start = vec![0.0f32; mat];
        let (p, q) = scans(&tm, &jm, &start, frames, heads, d);
        let inner = inner_bounds(&window_bounds(frames + 2, 1, 2));
        let alpha = vec![1.0f32; frames * d];
        let g = gather(&p, &q, &alpha, &start, &gather_plan(&inner), heads, d);
        for (t, &(lo, hi)) in inner.iter().enumerate() {
            for e in 0..mat {
                let want: f32 = (0..frames as i64)
                    .filter(|&f| f < lo || f > hi)
                    .map(|f| jm[f as usize * mat + e])
                    .sum();
                assert!((g[t * mat + e] - want).abs() < 1e-5, "t {t}: {} vs {want}", g[t * mat + e]);
            }
        }
    }

    #[test]
    fn the_bridge_decays_by_the_frames_in_between() {
        let (frames, d) = (6usize, 2usize);
        let mat = d * d;
        let mut seed = 5u64;
        let p = rand(frames * mat, &mut seed, 1.0);
        let q = rand(frames * mat, &mut seed, 1.0);
        let start = rand(mat, &mut seed, 1.0);
        let alpha: Vec<f32> = rand(frames * d, &mut seed, 0.3).iter().map(|x| 0.6 + x).collect();
        let inner = vec![(1i64, 3i64); frames];
        let plan = gather_plan(&inner);
        let g = gather(&p, &q, &alpha, &start, &plan, 1, d);
        let t = 2usize;
        for vrow in 0..d {
            for kk in 0..d {
                let db: f64 = (1..=t).map(|u| f64::from(alpha[u * d + kk])).product();
                let da: f64 = (t..=3).map(|u| f64::from(alpha[u * d + kk])).product();
                let want = f64::from(p[vrow * d + kk]) * db + f64::from(q[4 * mat + vrow * d + kk]) * da;
                assert!((f64::from(g[t * mat + vrow * d + kk]) - want).abs() < 1e-5);
            }
        }
        // a window touching the start reads the text state, decayed from frame 0
        let plan = gather_plan(&[(-2, 1), (-2, 1)]);
        assert_eq!(plan[0].before, None);
        assert_eq!(plan[0].before_from, 0);
        assert_eq!(plan[1].after, None);
        assert_eq!(plan[1].after_to, 1);
    }

    #[test]
    fn convolutions_are_cross_correlations_with_zero_padding() {
        // A one-hot kernel tap shifts the input; zero padding at the edges.
        let (frames, gh, gw, c) = (3usize, 4usize, 5usize, 2usize);
        let mut seed = 3u64;
        let x = rand(frames * gh * gw * c, &mut seed, 1.0);
        let mut w = vec![0.0f32; c * 25];
        w[2 * 5 + 3] = 1.0; // dy = 0, dx = +1 for channel 0
        w[25 + 12] = 2.0; // center, channel 1
        let y = spatial_conv(&x, frames, gh, gw, c, &w);
        let at = |f: usize, yy: usize, xx: usize, ch: usize| ((f * gh + yy) * gw + xx) * c + ch;
        assert_eq!(y[at(1, 2, 1, 0)], x[at(1, 2, 2, 0)]);
        assert_eq!(y[at(1, 2, 4, 0)], 0.0);
        assert_eq!(y[at(2, 3, 0, 1)], 2.0 * x[at(2, 3, 0, 1)]);
        let mut wt = vec![0.0f32; c * 5];
        wt[0] = 1.0; // dt = -2 for channel 0
        let s = gh * gw;
        let y = temporal_conv(&x, frames, s, c, &wt);
        assert_eq!(y[(2 * s + 3) * c], x[3 * c]);
        assert_eq!(y[3 * c], 0.0);
    }

    #[test]
    fn softplus_matches_torch_threshold() {
        assert_eq!(softplus(25.0), 25.0);
        assert!((softplus(0.0) - std::f64::consts::LN_2).abs() < 1e-15);
    }
}
