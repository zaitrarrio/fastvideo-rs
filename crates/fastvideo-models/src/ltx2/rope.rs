//! LTX-2 "split" rotary tables, host side.
//!
//! LTX-2 does not rotate by token index. Every token carries the physical
//! extent it covers — seconds on the time axis, pixels on the spatial axes —
//! and the rotary angle is a function of the extent's *midpoint* as a fraction
//! of a fixed maximum (20 s, 2048 px). That is what lets a video token and an
//! audio token at the same instant rotate identically in the audio↔video
//! cross-attention, and it is why this table is built from coordinates rather
//! than from `arange(S)`.
//!
//! Two more things differ from the usual rotary:
//!
//! * the frequency grid is `θ^(k/(n-1)) · π/2` — a `linspace` in float64 cast
//!   to float32 — multiplied by `2·frac - 1`, all per positional axis and laid
//!   out frequency-major (`k=0: t,h,w | k=1: t,h,w | …`);
//! * the flat vector of `dim/2` angles is left-padded with identity slots and
//!   then *chunked across heads*: head `j` owns slots `[j·r, (j+1)·r)`. Heads
//!   rotate with different frequencies, so the table is `[H, S, r]`.
//!
//! The references build the coordinates and the angles in float32
//! (`transformer_ltx2.py:906-1039`, `connectors.py:111-171`); the arithmetic
//! below is float32 in the same order so a table can be compared with theirs to
//! the last bit of `cos`/`sin`, not just to a tolerance that would hide a
//! layout mistake. See docs/ports/ltx2.md §e.

use super::config::Ltx2TransformerConfig;

/// How the reference divides a float32 tensor by a Python scalar — which
/// depends on where it runs. torch's CUDA kernel multiplies by the float32
/// reciprocal (`BinaryDivTrueKernel.cu`), its CPU kernel divides. The two
/// differ by an ulp in a coordinate, and at the top of the frequency grid
/// (angles up to 15 708 rad, where a float32 ulp is 2⁻¹⁰ rad) that is up to
/// 2⁻¹⁰…2⁻⁹ in a cos/sin — exactly the residue first measured against a CUDA
/// dump. Harmless, but free to remove: production mirrors CUDA, and the CPU
/// fixtures are compared under `Exact`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarDivision {
    /// `x / d`: torch on the CPU, numpy.
    Exact,
    /// `x * (1 / d)` with the reciprocal rounded to float32: torch on CUDA.
    Reciprocal,
}

impl ScalarDivision {
    fn div(self, x: f32, d: f32) -> f32 {
        match self {
            Self::Exact => x / d,
            Self::Reciprocal => x * (1.0 / d),
        }
    }
}

/// One rotary table in the reference's own layout: `[heads, tokens, half]`,
/// head-major, `half = dim / heads / 2` — one value per rotated pair.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitRope {
    pub heads: usize,
    pub tokens: usize,
    pub half: usize,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl SplitRope {
    /// `fractions` is `[tokens, axes]` row-major: each token's midpoint on each
    /// positional axis as a fraction of that axis' maximum. `dim` is the full
    /// attention width (`heads · head_dim`).
    ///
    /// # Panics
    /// If `dim` is not divisible by `2 · heads`, `axes` is zero, or `fractions`
    /// is not a whole number of tokens — all programming errors, not inputs.
    pub fn from_fractions(fractions: &[f32], axes: usize, dim: usize, heads: usize, theta: f64) -> Self {
        assert!(axes > 0 && heads > 0 && dim.is_multiple_of(2 * heads), "split rope: dim {dim}, {heads} heads, {axes} axes");
        assert!(fractions.len().is_multiple_of(axes), "split rope: {} fractions over {axes} axes", fractions.len());
        let tokens = fractions.len() / axes;
        let slots = dim / 2;
        let half = slots / heads;
        let n = dim / (2 * axes);
        let pad = slots - n * axes;
        // torch.linspace(0, 1, n, float64) → theta ** that → · pi/2 → float32.
        let freqs: Vec<f32> = (0..n)
            .map(|k| {
                let e = if n == 1 { 0.0 } else { k as f64 / (n - 1) as f64 };
                (theta.powf(e) * std::f64::consts::FRAC_PI_2) as f32
            })
            .collect();
        let (mut cos, mut sin) = (vec![1f32; heads * tokens * half], vec![0f32; heads * tokens * half]);
        for t in 0..tokens {
            for m in pad..slots {
                let (k, a) = ((m - pad) / axes, (m - pad) % axes);
                // (grid * 2 - 1) * freqs, each step rounded to float32 as torch does.
                let angle = (fractions[t * axes + a] * 2.0 - 1.0) * freqs[k];
                let (h, j) = (m / half, m % half);
                let at = (h * tokens + t) * half + j;
                // cos/sin of the float32 angle, correctly rounded: evaluate in f64.
                cos[at] = f64::from(angle).cos() as f32;
                sin[at] = f64::from(angle).sin() as f32;
            }
        }
        Self { heads, tokens, half, cos, sin }
    }

    /// The same table for a rotate_half kernel that takes `[rows, head_dim]`
    /// cos/sin and pairs channel `j` with `j + head_dim/2`: each value is
    /// duplicated across the two halves, and the head axis is folded into the
    /// row axis (`row = head · tokens + token`), which is the memory order of a
    /// `[B, H, S, D]` tensor viewed as `[B, 1, H·S, D]`.
    pub fn rotate_half_tables(&self) -> (Vec<f32>, Vec<f32>) {
        let widen = |v: &[f32]| -> Vec<f32> {
            v.chunks_exact(self.half).flat_map(|row| row.iter().chain(row.iter()).copied()).collect()
        };
        (widen(&self.cos), widen(&self.sin))
    }
}

/// `[tokens, 3]` midpoints of each latent cell's extent in (seconds, px, px).
/// Token order is frame-major, then row, then column — the packing order of
/// the latents.
fn video_midpoints(cfg: &Ltx2TransformerConfig, grid: [usize; 3], fps: f32, division: ScalarDivision) -> Vec<f32> {
    let [frames, height, width] = grid;
    let [st, sh, sw] = cfg.vae_scale_factors.map(|s| s as f32);
    let time = |f: usize| -> f32 {
        // The first latent frame covers one pixel frame, every later one eight:
        // shift by causal_offset - stride and clamp at zero, then seconds.
        let bound = |x: f32| division.div((x * st + cfg.causal_offset as f32 - st).max(0.0), fps);
        division.div(bound(f as f32) + bound(f as f32 + cfg.patch_size_t as f32), 2.0)
    };
    let space = |i: usize, scale: f32| -> f32 { (i as f32 * scale + (i as f32 + cfg.patch_size as f32) * scale) / 2.0 };
    let mut out = Vec::with_capacity(frames * height * width * 3);
    for f in 0..frames {
        for h in 0..height {
            for w in 0..width {
                out.extend([time(f), space(h, sh), space(w, sw)]);
            }
        }
    }
    out
}

/// `[tokens, 3]` fractions for the video self-attention table: the midpoints
/// over `(max_pos s, base_height, base_width)`.
pub fn video_fractions(cfg: &Ltx2TransformerConfig, grid: [usize; 3], fps: f32, division: ScalarDivision) -> Vec<f32> {
    let max = [cfg.pos_embed_max_pos as f32, cfg.base_height as f32, cfg.base_width as f32];
    video_midpoints(cfg, grid, fps, division)
        .chunks_exact(3)
        .flat_map(|c| [division.div(c[0], max[0]), division.div(c[1], max[1]), division.div(c[2], max[2])])
        .collect()
}

/// Midpoint in seconds of each video token's temporal extent, divided by
/// `max_seconds` — the single axis of the audio↔video cross-attention table.
pub fn video_time_fractions(cfg: &Ltx2TransformerConfig, grid: [usize; 3], fps: f32, max_seconds: f32, division: ScalarDivision) -> Vec<f32> {
    video_midpoints(cfg, grid, fps, division).chunks_exact(3).map(|c| division.div(c[0], max_seconds)).collect()
}

/// Midpoint in seconds of audio latent `i`, over `max_seconds`. One latent is
/// `audio_scale_factor` mel frames of `hop / rate` seconds, with the same
/// causal first-frame shift as the video.
pub fn audio_fractions(cfg: &Ltx2TransformerConfig, tokens: usize, max_seconds: f32, division: ScalarDivision) -> Vec<f32> {
    let scale = cfg.audio_scale_factor as f32;
    let (hop, rate) = (cfg.audio_hop_length as f32, cfg.audio_sampling_rate as f32);
    let bound = |x: f32| division.div((x * scale + cfg.causal_offset as f32 - scale).max(0.0) * hop, rate);
    (0..tokens)
        .map(|i| division.div(division.div(bound(i as f32) + bound(i as f32 + cfg.audio_patch_size_t as f32), 2.0), max_seconds))
        .collect()
}

/// The connectors' 1-D table: position `p` of `tokens` has fraction
/// `p / base_seq_len`.
pub fn connector_fractions(tokens: usize, base_seq_len: usize) -> Vec<f32> {
    (0..tokens).map(|p| p as f32 / base_seq_len as f32).collect()
}

/// The four tables one DiT forward needs. They depend only on the request's
/// geometry, so they are built once per run.
#[derive(Debug, Clone)]
pub struct Ltx2RopeTables {
    /// Video self-attention: 3 axes over the video width.
    pub video: SplitRope,
    /// Audio self-attention: time only, over the audio width.
    pub audio: SplitRope,
    /// Video side of audio↔video cross-attention: time only, audio head layout.
    pub cross_video: SplitRope,
    /// Audio side of audio↔video cross-attention.
    pub cross_audio: SplitRope,
}

impl Ltx2RopeTables {
    /// The tables as the product builds them: on CUDA.
    pub fn new(cfg: &Ltx2TransformerConfig, grid: [usize; 3], audio_tokens: usize, fps: f32) -> Self {
        Self::with_division(cfg, grid, audio_tokens, fps, ScalarDivision::Reciprocal)
    }

    pub fn with_division(cfg: &Ltx2TransformerConfig, grid: [usize; 3], audio_tokens: usize, fps: f32, division: ScalarDivision) -> Self {
        let theta = cfg.rope_theta;
        let cross_dim = cfg.audio_cross_attention_dim;
        // Both cross tables share one time base so equal instants rotate equally.
        let cross_max = cfg.pos_embed_max_pos.max(cfg.audio_pos_embed_max_pos) as f32;
        Self {
            video: SplitRope::from_fractions(&video_fractions(cfg, grid, fps, division), 3, cfg.inner_dim(), cfg.num_attention_heads, theta),
            audio: SplitRope::from_fractions(
                &audio_fractions(cfg, audio_tokens, cfg.audio_pos_embed_max_pos as f32, division),
                1,
                cfg.audio_inner_dim(),
                cfg.audio_num_attention_heads,
                theta,
            ),
            cross_video: SplitRope::from_fractions(
                &video_time_fractions(cfg, grid, fps, cross_max, division),
                1,
                cross_dim,
                cfg.num_attention_heads,
                theta,
            ),
            cross_audio: SplitRope::from_fractions(
                &audio_fractions(cfg, audio_tokens, cross_max, division),
                1,
                cross_dim,
                cfg.audio_num_attention_heads,
                theta,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Ltx2TransformerConfig {
        Ltx2TransformerConfig::ltx2_19b()
    }

    #[test]
    fn video_midpoints_follow_the_causal_first_frame() {
        let f = video_fractions(&cfg(), [3, 2, 2], 24.0, ScalarDivision::Exact);
        assert_eq!(f.len(), 3 * 2 * 2 * 3);
        // Latent 0 covers pixel frame [0, 1), latent 1 [1, 9), latent 2 [9, 17).
        let seconds: Vec<f32> = (0..3).map(|i| f[i * 4 * 3] * 20.0).collect();
        for (got, want) in seconds.iter().zip([0.5 / 24.0, 5.0 / 24.0, 13.0 / 24.0]) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
        // Row 1, column 1 of frame 0: pixels [32, 64) → midpoint 48.
        let last = &f[3 * 3..4 * 3];
        assert_eq!(last[1], 48.0 / 2048.0);
        assert_eq!(last[2], 48.0 / 2048.0);
        // Column varies fastest.
        assert_eq!(f[3 + 2], 48.0 / 2048.0);
        assert_eq!(f[3 + 1], 16.0 / 2048.0);
    }

    #[test]
    fn audio_midpoints_are_ten_milliseconds_per_mel_frame() {
        let f = audio_fractions(&cfg(), 4, 20.0, ScalarDivision::Exact);
        // Latent 0 covers mel [0, 1), latent 1 [1, 5), latent 2 [5, 9): 10 ms each.
        for (got, want) in f.iter().zip([0.005, 0.03, 0.07, 0.11]) {
            assert!((got * 20.0 - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn a_video_and_an_audio_token_at_the_same_instant_rotate_alike() {
        // Video latent 1 spans [1/24, 9/24) s, midpoint 5/24 s. An audio
        // fraction forced to the same instant must give the same angles.
        let c = cfg();
        let v = video_time_fractions(&c, [2, 1, 1], 24.0, 20.0, ScalarDivision::Reciprocal);
        let a = SplitRope::from_fractions(&[v[1]], 1, 2048, 32, c.rope_theta);
        let t = Ltx2RopeTables::new(&c, [2, 1, 1], 1, 24.0);
        let (s, r) = (t.cross_video.tokens, t.cross_video.half);
        for h in 0..32 {
            for j in 0..r {
                assert_eq!(t.cross_video.cos[(h * s + 1) * r + j], a.cos[h * r + j]);
            }
        }
    }

    /// Against the definition written out independently: slot `m` of the flat
    /// `dim/2` vector is identity below the pad, else frequency `(m-pad)/axes`
    /// on axis `(m-pad) % axes`; head `h` owns slots `[h·r, (h+1)·r)`.
    #[test]
    fn slots_are_frequency_major_left_padded_and_chunked_by_head() {
        let (dim, heads, axes) = (32usize, 2usize, 3usize);
        let fr = [0.1f32, 0.7, 0.4, 0.9, 0.2, 0.6];
        let t = SplitRope::from_fractions(&fr, axes, dim, heads, 10_000.0);
        assert_eq!((t.heads, t.tokens, t.half), (2, 2, 8));
        let n = dim / (2 * axes); // 5 frequencies per axis → 15 slots, 1 pad.
        let pad = dim / 2 - n * axes;
        assert_eq!((n, pad), (5, 1));
        for tok in 0..2 {
            for m in 0..dim / 2 {
                let (h, j) = (m / 8, m % 8);
                let got = (t.cos[(h * 2 + tok) * 8 + j], t.sin[(h * 2 + tok) * 8 + j]);
                if m < pad {
                    assert_eq!(got, (1.0, 0.0), "identity slot");
                    continue;
                }
                let (k, a) = ((m - pad) / axes, (m - pad) % axes);
                let freq = 10_000f64.powf(k as f64 / (n - 1) as f64) * std::f64::consts::FRAC_PI_2;
                let angle = (f64::from(fr[tok * axes + a]) * 2.0 - 1.0) * freq;
                assert!((f64::from(got.0) - angle.cos()).abs() < 2e-3, "slot {m}");
                assert!((f64::from(got.1) - angle.sin()).abs() < 2e-3, "slot {m}");
            }
        }
    }

    #[test]
    fn production_tables_have_the_documented_shapes() {
        let t = Ltx2RopeTables::new(&cfg(), [2, 2, 3], 5, 24.0);
        assert_eq!((t.video.heads, t.video.tokens, t.video.half), (32, 12, 64));
        assert_eq!((t.audio.heads, t.audio.tokens, t.audio.half), (32, 5, 32));
        assert_eq!((t.cross_video.heads, t.cross_video.tokens, t.cross_video.half), (32, 12, 32));
        assert_eq!((t.cross_audio.heads, t.cross_audio.tokens, t.cross_audio.half), (32, 5, 32));
        // Two identity slots in front of the video table: head 0, pairs 0 and 1.
        assert_eq!(&t.video.cos[..2], &[1.0, 1.0]);
        assert_eq!(&t.video.sin[..2], &[0.0, 0.0]);
        assert_ne!(t.video.sin[2], 0.0);
    }

    #[test]
    fn rotate_half_tables_duplicate_each_pair_value() {
        let t = SplitRope::from_fractions(&connector_fractions(3, 4096), 1, 16, 2, 10_000.0);
        let (cos, sin) = t.rotate_half_tables();
        assert_eq!(cos.len(), 2 * 3 * 8);
        for row in 0..6 {
            for j in 0..4 {
                assert_eq!(cos[row * 8 + j], t.cos[row * 4 + j]);
                assert_eq!(cos[row * 8 + 4 + j], t.cos[row * 4 + j]);
                assert_eq!(sin[row * 8 + 4 + j], t.sin[row * 4 + j]);
            }
        }
    }
    /// Golden values from a numpy transliteration of the reference
    /// (`transformer_ltx2.py:906-1076`, `connectors.py:111-171`): its meshgrid /
    /// stack / transpose / flatten / reshape / swapaxes sequence, float32 where
    /// torch is float32 — so the layout here is checked against the reference's
    /// own tensor plumbing, not against a second reading of it. Latent grid
    /// 3x2x3, 5 audio latents, 24 fps, production widths.
    #[allow(clippy::excessive_precision)] // the digits are numpy's repr, kept verbatim
    #[test]
    fn tables_match_a_numpy_transliteration_of_the_reference() {
        let t = Ltx2RopeTables::with_division(&cfg(), [3, 2, 3], 5, 24.0, ScalarDivision::Exact);
        let sums = |r: &SplitRope| (r.cos.iter().map(|&v| f64::from(v)).sum::<f64>(), r.sin.iter().map(|&v| f64::from(v)).sum::<f64>());
        for (name, table, want) in [
            ("video", &t.video, (-1456.788344584278, -526.246063605472)),
            ("audio", &t.audio, (-266.3353606130113, -141.95092843251768)),
            ("cross_video", &t.cross_video, (-962.3432892755955, -534.2277238606475)),
            ("cross_audio", &t.cross_audio, (-266.3353606130113, -141.95092843251768)),
        ] {
            let got = sums(table);
            assert!((got.0 - want.0).abs() < 1e-4 && (got.1 - want.1).abs() < 1e-4, "{name}: sums {got:?} vs {want:?}");
        }
        let at = |r: &SplitRope, h: usize, tok: usize, j: usize| {
            let i = (h * r.tokens + tok) * r.half + j;
            (r.cos[i], r.sin[i])
        };
        let close = |got: (f32, f32), want: (f32, f32), what: &str| {
            assert!((got.0 - want.0).abs() < 2e-7 && (got.1 - want.1).abs() < 2e-7, "{what}: {got:?} vs {want:?}");
        };
        close(at(&t.video, 0, 0, 0), (1.0, 0.0), "video identity pad");
        close(at(&t.video, 0, 0, 2), (0.003_272_483_8, -0.999_994_64), "video h0 t0 j2");
        close(at(&t.video, 0, 0, 3), (0.024_541_136, -0.999_698_8), "video h0 t0 j3");
        close(at(&t.video, 5, 7, 13), (0.846_158_03, -0.532_932_1), "video h5 t7 j13");
        close(at(&t.video, 31, 17, 63), (-0.382_976_6, 0.923_758_03), "video h31 t17 j63");
        close(at(&t.video, 17, 9, 40), (0.999_613_7, 0.027_792_243), "video h17 t9 j40");
        close(at(&t.cross_video, 0, 6, 0), (0.032_718_97, -0.999_464_57), "cross_video h0 t6 j0");
        close(at(&t.cross_video, 31, 17, 31), (-0.866_037, 0.499_979_88), "cross_video h31 t17 j31");
        close(at(&t.cross_video, 12, 11, 5), (0.707_948_03, -0.706_264_56), "cross_video h12 t11 j5");
        close(at(&t.audio, 0, 0, 0), (0.000_785_426_24, -0.999_999_7), "audio h0 t0 j0");
        close(at(&t.audio, 31, 4, 31), (-1.0, -6.892_973e-5), "audio h31 t4 j31");
        close(at(&t.audio, 9, 3, 17), (0.598_838_4, 0.800_869_94), "audio h9 t3 j17");

        let c = SplitRope::from_fractions(&connector_fractions(8, 4096), 1, 3840, 30, 10_000.0);
        let got = sums(&c);
        assert!((got.0 + 769.1431764554143).abs() < 1e-4 && (got.1 + 343.0537126405907).abs() < 1e-4, "connector sums {got:?}");
        close(at(&c, 0, 0, 0), (-4.371_139e-8, -1.0), "connector h0 p0 j0");
        close(at(&c, 29, 7, 63), (-0.960_290_3, -0.279_002_64), "connector h29 p7 j63");
        close(at(&c, 11, 3, 20), (0.925_963_9, -0.377_612_05), "connector h11 p3 j20");
    }
    /// The first hardware run found every table within cos 0.999999998 of the
    /// reference but off by *exactly* 2^-10 (video time) and 2^-9 (audio) at
    /// worst. That is not bfloat16: it is one float32 ulp of a 15 708 rad angle,
    /// from torch-on-CUDA dividing by a scalar as a multiply by its reciprocal.
    /// A numpy model of both conventions gives these same two numbers for the
    /// production geometry, which is what pins `Reciprocal` as the product's.
    #[allow(clippy::excessive_precision)]
    #[test]
    fn cuda_and_cpu_scalar_division_differ_by_the_residue_measured_on_hardware() {
        let c = cfg();
        let worst = |a: &SplitRope, b: &SplitRope| a.cos.iter().zip(&b.cos).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        // Time is the only axis of these two tables, so one spatial cell is enough.
        let exact = Ltx2RopeTables::with_division(&c, [16, 1, 1], 126, 24.0, ScalarDivision::Exact);
        let cuda = Ltx2RopeTables::new(&c, [16, 1, 1], 126, 24.0);
        assert_eq!(worst(&exact.cross_video, &cuda.cross_video), 0.000_976_562_44);
        assert_eq!(worst(&exact.audio, &cuda.audio), 0.001_952_932_2);
        // Halving and dividing by a power of two are exact either way.
        assert_eq!(ScalarDivision::Reciprocal.div(3.0, 2.0), 1.5);
        assert_eq!(ScalarDivision::Reciprocal.div(48.0, 2048.0), ScalarDivision::Exact.div(48.0, 2048.0));
    }
}
