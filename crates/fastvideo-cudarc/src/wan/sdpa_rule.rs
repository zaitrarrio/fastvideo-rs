//! The fixed rule `auto` uses to choose between cuDNN's fused SDPA and
//! `flash_mma_fwd2` for a bf16-output dense attention (sm_12x).
//!
//! Until 2026-09-29 `auto` timed both kernels on the first call of every
//! `(bh, sq, sk, d)` in a process and kept the faster. The two kernels do not
//! round alike (1-2 bf16 ulp apart), and at shapes where they run equally
//! fast (text / audio cross-attention: 0.63 vs 0.63 ms) the winner changed
//! from process to process, so the same request gave different bytes on
//! different boots (LTX 1080p: 33.6-34.3 dB between boots). The pick is now a
//! pure function of `(sm_major, sq, sk, d)`: [`RULES`] below, first match
//! wins, else `flash_mma_fwd2`.
//!
//! The table comes from timings, offline: `fv-gpucheck kernels --groups
//! sdpa_rule` times both kernels over the real shapes and the class
//! boundaries and prints, per shape, the faster kernel, the margin and what
//! [`pick`] says (docs/perf/determinism.md). `FASTVIDEO_SDPA_AUTO=timed`
//! restores the old per-process timing (not reproducible across processes;
//! for diagnosis only), and `FASTVIDEO_FLASH_KERNEL=v2|cudnn` still fixes the
//! kernel outright.
//!
//! SageAttention2 (`wan::attn_sage`) is decided before this rule, per recipe
//! and shape (`wan/nn.rs`): when it is on (default for `ltx-pro` on sm_120),
//! sequences of at least `FASTVIDEO_ATTN_SAGE_MIN_SEQ` on both sides run it,
//! and only the shorter ones (cross-attention) reach this rule.

/// The kernel the rule picks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DensePick {
    /// cuDNN's unified SDPA node (heuristic mode A, first deterministic config).
    Cudnn,
    /// `flash_mma_fwd2` (the 128-query double-buffered kernel).
    V2,
}

impl DensePick {
    pub fn name(self) -> &'static str {
        match self {
            DensePick::Cudnn => "cudnn",
            DensePick::V2 => "flash_mma_fwd2",
        }
    }
}

/// One row of the rule: a shape class on one architecture.
#[derive(Clone, Copy, Debug)]
pub struct DenseRule {
    /// Compute capability major the row applies to.
    pub sm_major: i32,
    /// Head dim.
    pub d: usize,
    /// Inclusive bounds on the query length.
    pub min_sq: usize,
    pub max_sq: usize,
    /// Inclusive bounds on the key length.
    pub min_sk: usize,
    pub max_sk: usize,
    pub pick: DensePick,
}

impl DenseRule {
    pub fn matches(&self, sm_major: i32, sq: usize, sk: usize, d: usize) -> bool {
        self.sm_major == sm_major
            && self.d == d
            && (self.min_sq..=self.max_sq).contains(&sq)
            && (self.min_sk..=self.max_sk).contains(&sk)
    }
}

/// The rule for sm_12x, RTX PRO 6000 Blackwell (cuDNN 9.26, heuristic mode A
/// config 0, engine 11), bf16 in and out, d = 128. Median of three
/// synchronized calls (`attn3_bench`, `sdpa_rule`; the timed picks logged by
/// earlier runs):
///
/// | shape (bh x sq x sk) | cuDNN | fwd2 | faster |
/// |---|---:|---:|---|
/// | 24 x 23 616 x 23 616 | 19.08 | 18.85 | fwd2 1 % |
/// | 48 x 23 616 x 23 616 | 37.16 | 37.42 | cuDNN 0.7 % |
/// | 24 x 27 280 x 27 280 | 22.9-23.8 | 24.7-25.2 | cuDNN 4-9 % |
/// | 24 / 48 x 36 080 x 36 080 | 38.6 / 80.8 | 42.6 / 87.4 | cuDNN 8-10 % |
/// | 56 x 37 710 x 37 710 (H3 768p) | 104.8-105.0 | 108.4-109.1 | cuDNN 3 % |
/// | 32 x 124 440 x 124 440 (LTX 1080p 20 s) | 658.8-659.8 | 682.2-682.7 | cuDNN 3.5 % |
/// | 32 x 130 560 x 130 560 (LTX 4K 5 s) | 766.6-769.0 | 757.2-759.9 | fwd2 1.2 % |
/// | 32 x 6 144 x 6 144 (LTX 512p) | 1.97-1.99 | 1.98-2.00 | tie |
/// | any sq x 512 (text cross-attention) | 0.58-1.59 | 0.56-1.60 | tie, either way by run |
///
/// So cuDNN for self-attention-sized keys from 24 576 to 128 000 tokens,
/// fwd2 everywhere else (short keys are a tie; fwd2 needs no plan build).
pub const RULES: &[DenseRule] = &[DenseRule {
    sm_major: 12,
    d: 128,
    min_sq: 24_576,
    max_sq: usize::MAX,
    min_sk: 24_576,
    max_sk: 128_000,
    pick: DensePick::Cudnn,
}];

/// The rule's kernel for a bf16-output dense SDPA of `sq` queries over `sk`
/// keys at head dim `d` on an `sm_major` device: the first matching row of
/// [`RULES`], else fwd2. Does not look at `bh`: the rows were measured at
/// 24-56 heads and the margins do not change sign across that range.
pub fn pick(sm_major: i32, sq: usize, sk: usize, d: usize) -> DensePick {
    RULES
        .iter()
        .find(|r| r.matches(sm_major, sq, sk, d))
        .map_or(DensePick::V2, |r| r.pick)
}

/// How `auto` chooses on sm_12x: `rule` (default, [`pick`]) or `timed` (the
/// old first-call timing, `FASTVIDEO_SDPA_AUTO=timed`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoMode {
    Rule,
    Timed,
}

pub fn auto_mode() -> AutoMode {
    static MODE: std::sync::OnceLock<AutoMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| parse_auto_mode(&super::envflag::string_flag("FASTVIDEO_SDPA_AUTO", "rule")))
}

pub fn parse_auto_mode(v: &str) -> AutoMode {
    match v.trim() {
        "timed" => AutoMode::Timed,
        _ => AutoMode::Rule,
    }
}

/// The table as the offline tool prints it: one line per row.
pub fn describe() -> String {
    RULES
        .iter()
        .map(|r| {
            format!(
                "sm_{}x d={} sq {}..={} sk {}..={} -> {}",
                r.sm_major,
                r.d,
                r.min_sq,
                if r.max_sq == usize::MAX { "inf".to_string() } else { r.max_sq.to_string() },
                r.min_sk,
                r.max_sk,
                r.pick.name()
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_measured_shapes_get_the_faster_kernel_or_a_tie() {
        // Self-attention where cuDNN measured faster.
        for n in [27_280, 36_080, 37_710, 124_440] {
            assert_eq!(pick(12, n, n, 128), DensePick::Cudnn, "{n}");
        }
        // fwd2 faster (4K 5 s, 23 616 at 24 heads) or a tie (512p, cross-attention).
        for (sq, sk) in [(130_560, 130_560), (23_616, 23_616), (6_144, 6_144), (27_280, 512), (38_760, 1_024), (124_440, 151)] {
            assert_eq!(pick(12, sq, sk, 128), DensePick::V2, "{sq} x {sk}");
        }
    }

    #[test]
    fn only_sm12x_d128_has_rows() {
        assert_eq!(pick(9, 37_710, 37_710, 128), DensePick::V2);
        assert_eq!(pick(10, 37_710, 37_710, 128), DensePick::V2);
        assert_eq!(pick(12, 37_710, 37_710, 64), DensePick::V2);
    }

    #[test]
    fn the_pick_is_a_pure_function_of_the_shape() {
        let a: Vec<DensePick> = (0..64).map(|i| pick(12, 20_000 + i * 2_000, 20_000 + i * 2_000, 128)).collect();
        let b: Vec<DensePick> = (0..64).map(|i| pick(12, 20_000 + i * 2_000, 20_000 + i * 2_000, 128)).collect();
        assert_eq!(a, b);
        assert!(describe().contains("cudnn"));
    }

    #[test]
    fn auto_mode_parses() {
        assert_eq!(parse_auto_mode("timed"), AutoMode::Timed);
        assert_eq!(parse_auto_mode("rule"), AutoMode::Rule);
        assert_eq!(parse_auto_mode(""), AutoMode::Rule);
    }
}
