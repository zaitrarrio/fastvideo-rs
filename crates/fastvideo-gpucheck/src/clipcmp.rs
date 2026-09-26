//! `compare-clips`: the paired-clip quality gate. A candidate clip (a
//! precision, cache or sparsity switch) is judged frame by frame against a
//! baseline clip of the same prompt, seed and geometry.
//!
//! Every metric is a port of sol-engine `scripts/collect_run.py`; the line
//! numbers cited below are that file's. Frames are compared as 8-bit RGB
//! (`image_array`, :784-787: PIL `convert("RGB")`), paired in sorted order and
//! truncated to the shorter clip (`paired_frames`, :579-581: a plain `zip`).
//!
//! Gates. sol-engine `evals/tiers.toml` sets no absolute pixel or LPIPS
//! threshold for lossy generative dimensions (`lpips_absolute_threshold =
//! "disabled_for_lossy_generative_dimensions"`); only exact/numeric dimensions
//! may enforce OFF identity as a hard gate. So the pixel metrics are recorded
//! as telemetry and the only hard check is `--off-identity`
//! (`max_abs_diff_uint8 == 0`).
//!
//! LPIPS (sol-engine `tools/vision/lpips_judge.py`: `lpips.LPIPS(net="alex")`)
//! is optional (`--lpips <weights dir>`, [`crate::lpips`]): scored on
//! sol-engine's frame selection (`select_stratified_and_worst_pairs`, 32
//! chronological pairs plus the 16 with the largest pixel difference, at most
//! 48) and reported as the `lpips` judge plus top-level `lpips_mean` /
//! `lpips_max`. Without the flag the report marks it `deferred`.

use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde_json::{json, Value};

use crate::metrics::jf;
use crate::report::{Report, StageError, StageResult};

/// `PATCH_BOUNDARY_SIZES` (:92).
pub const PATCH_BOUNDARY_SIZES: [usize; 3] = [8, 16, 32];

/// `ERROR_PATTERNS` (:38-49), matched case-insensitively against the run log.
const ERROR_PATTERNS: [&str; 10] = [
    "traceback (most recent call last)",
    "runtimeerror:",
    "cuda out of memory",
    "outofmemoryerror",
    "error: repository not found",
    "error:",
    "fatal:",
    "slurmstepd: error",
    "command not found",
    "no such file or directory",
];

/// One 8-bit RGB frame, row-major HWC.
#[derive(Clone, Debug, PartialEq)]
pub struct Rgb {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

impl Rgb {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let img = image::open(path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?
            .into_rgb8();
        let (w, h) = img.dimensions();
        Ok(Self {
            w: w as usize,
            h: h as usize,
            px: img.into_raw(),
        })
    }

    fn shape(&self) -> [usize; 3] {
        [self.h, self.w, 3]
    }
}

/// `mean_abs_gradient` (:790-795): mean |horizontal neighbour diff| plus mean
/// |vertical neighbour diff|, each over all channels; a term is 0 when its
/// dimension has one sample.
pub fn mean_abs_gradient(a: &Rgb) -> f64 {
    let (w, h) = (a.w, a.h);
    let row = w * 3;
    let dx = if w > 1 {
        let mut s = 0u64;
        for y in 0..h {
            let r = &a.px[y * row..(y + 1) * row];
            for i in 3..row {
                s += u64::from(r[i].abs_diff(r[i - 3]));
            }
        }
        s as f64 / (h * (w - 1) * 3) as f64
    } else {
        0.0
    };
    let dy = if h > 1 {
        let s: u64 = a.px[row..]
            .iter()
            .zip(&a.px[..(h - 1) * row])
            .map(|(p, q)| u64::from(p.abs_diff(*q)))
            .sum();
        s as f64 / ((h - 1) * row) as f64
    } else {
        0.0
    };
    dx + dy
}

/// `patch_boundary_score` (:798-806): for every patch-grid column x = p, 2p,
/// ... < W the mean |a[:, x] - a[:, x-1]| over rows and channels, likewise
/// for grid rows y; the score is the unweighted mean of those per-line means
/// (0 when the frame is no larger than one patch).
pub fn patch_boundary_score(a: &Rgb, patch: usize) -> f64 {
    let row = a.w * 3;
    let mut scores: Vec<f64> = Vec::new();
    for x in (patch..a.w).step_by(patch) {
        let mut s = 0u64;
        for y in 0..a.h {
            for c in 0..3 {
                let i = y * row + x * 3 + c;
                s += u64::from(a.px[i].abs_diff(a.px[i - 3]));
            }
        }
        scores.push(s as f64 / (a.h * 3) as f64);
    }
    for y in (patch..a.h).step_by(patch) {
        let cur = &a.px[y * row..(y + 1) * row];
        let prev = &a.px[(y - 1) * row..y * row];
        let s: u64 = cur
            .iter()
            .zip(prev)
            .map(|(p, q)| u64::from(p.abs_diff(*q)))
            .sum();
        scores.push(s as f64 / row as f64);
    }
    if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    }
}

/// Per-pair pixel error: `diff = ca - ba`, mse, mae and PSNR (:873-876), plus
/// the frame's max |diff| for OFF identity (:829-831).
#[derive(Clone, Copy, Debug)]
pub struct PairError {
    pub mse: f64,
    pub mae: f64,
    pub psnr: f64,
    pub max_abs: u8,
}

pub fn pair_error(b: &Rgb, c: &Rgb) -> PairError {
    let (mut sq, mut ab, mut mx) = (0u64, 0u64, 0u8);
    for (&p, &q) in c.px.iter().zip(&b.px) {
        let d = p.abs_diff(q);
        sq += u64::from(d) * u64::from(d);
        ab += u64::from(d);
        mx = mx.max(d);
    }
    let n = b.px.len().max(1) as f64;
    let mse = sq as f64 / n;
    // :876
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        20.0 * (255.0 / mse.sqrt()).log10()
    };
    PairError {
        mse,
        mae: ab as f64 / n,
        psnr,
        max_abs: mx,
    }
}

/// Temporal terms for pair i > 0 (:891-897): with base_delta = b_i - b_{i-1}
/// and cand_delta = c_i - c_{i-1}, the error is mean |cand_delta - base_delta|
/// and the jitter ratio mean |cand_delta| / max(mean |base_delta|, 1e-8).
pub fn temporal(bp: &Rgb, cp: &Rgb, b: &Rgb, c: &Rgb) -> (f64, f64) {
    let (mut err, mut bm, mut cm) = (0u64, 0u64, 0u64);
    for i in 0..b.px.len() {
        let bd = i32::from(b.px[i]) - i32::from(bp.px[i]);
        let cd = i32::from(c.px[i]) - i32::from(cp.px[i]);
        err += u64::from((cd - bd).unsigned_abs());
        bm += u64::from(bd.unsigned_abs());
        cm += u64::from(cd.unsigned_abs());
    }
    let n = b.px.len().max(1) as f64;
    let (bm, cm) = (bm as f64 / n, cm as f64 / n);
    (err as f64 / n, cm / bm.max(1e-8))
}

/// `--lpips`: where the weights are, which pairs, which device.
#[derive(Clone, Debug)]
pub struct LpipsOpts {
    pub weights: PathBuf,
    /// sol-engine's 48-pair budget, or every pair.
    pub all_pairs: bool,
    pub device: bool,
}

/// Everything measured on one (baseline, candidate) pair that does not need
/// the previous pair.
#[derive(Clone, Debug)]
struct PairStats {
    err: PairError,
    sharpness_ratio: f64,
    patch_ratio_by_size: [f64; PATCH_BOUNDARY_SIZES.len()],
}

fn pair_stats(b: &Rgb, c: &Rgb) -> PairStats {
    let err = pair_error(b, c);
    // :877-879, :887
    let sharpness_ratio = mean_abs_gradient(c) / mean_abs_gradient(b).max(1e-8);
    // :880-886
    let mut patch_ratio_by_size = [0.0; PATCH_BOUNDARY_SIZES.len()];
    for (k, &p) in PATCH_BOUNDARY_SIZES.iter().enumerate() {
        patch_ratio_by_size[k] = patch_boundary_score(c, p) / patch_boundary_score(b, p).max(1e-8);
    }
    PairStats {
        err,
        sharpness_ratio,
        patch_ratio_by_size,
    }
}

/// Accumulates `build_off_identity` (:813-843) and `build_pixel_metrics`
/// (:845-925) over pairs fed in frame order.
#[derive(Default)]
pub struct ClipComparison {
    shape_mismatch: Option<(usize, [usize; 3], [usize; 3])>,
    pairs: usize,
    nonidentical: usize,
    max_abs: u8,
    mse: Vec<f64>,
    mae: Vec<f64>,
    psnr: Vec<f64>,
    sharpness: Vec<f64>,
    patch: Vec<f64>,
    patch_by_size: [Vec<f64>; PATCH_BOUNDARY_SIZES.len()],
    temporal_err: Vec<f64>,
    jitter: Vec<f64>,
}

fn fmean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn fmax(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn fmin(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

impl ClipComparison {
    /// Feed a chunk of consecutive pairs; `prev` is the pair just before the
    /// chunk (None for the first). Stops at the first shape mismatch.
    fn push_chunk(&mut self, prev: Option<&(Rgb, Rgb)>, chunk: &[(Rgb, Rgb)]) {
        if self.shape_mismatch.is_some() {
            return;
        }
        if let Some(k) = chunk.iter().position(|(b, c)| b.shape() != c.shape()) {
            let (b, c) = &chunk[k];
            self.shape_mismatch = Some((self.pairs + k, b.shape(), c.shape()));
        }
        let usable = self
            .shape_mismatch
            .map_or(chunk.len(), |(i, _, _)| i - self.pairs);
        let chunk = &chunk[..usable];
        let stats: Vec<PairStats> = chunk.par_iter().map(|(b, c)| pair_stats(b, c)).collect();
        let temp: Vec<Option<(f64, f64)>> = (0..chunk.len())
            .into_par_iter()
            .map(|k| {
                let before = if k == 0 { prev } else { Some(&chunk[k - 1]) };
                let (b, c) = &chunk[k];
                // Consecutive frames of one clip must share a shape; a clip
                // that changes size mid-way gets no temporal term there.
                before
                    .filter(|(bp, cp)| bp.shape() == b.shape() && cp.shape() == c.shape())
                    .map(|(bp, cp)| temporal(bp, cp, b, c))
            })
            .collect();
        for (s, t) in stats.into_iter().zip(temp) {
            self.pairs += 1;
            // :829-833
            self.max_abs = self.max_abs.max(s.err.max_abs);
            if s.err.max_abs != 0 {
                self.nonidentical += 1;
            }
            self.mse.push(s.err.mse);
            self.mae.push(s.err.mae);
            self.psnr.push(s.err.psnr);
            self.sharpness.push(s.sharpness_ratio);
            // :887 the pair's patch ratio is the worst over patch sizes.
            self.patch.push(fmax(&s.patch_ratio_by_size));
            for (k, r) in s.patch_ratio_by_size.iter().enumerate() {
                self.patch_by_size[k].push(*r);
            }
            if let Some((e, j)) = t {
                self.temporal_err.push(e);
                self.jitter.push(j);
            }
        }
    }

    /// Compare in-memory clips (tests; the CLI streams from disk).
    #[cfg(test)]
    pub fn of(baseline: &[Rgb], candidate: &[Rgb]) -> Self {
        let pairs: Vec<(Rgb, Rgb)> = baseline
            .iter()
            .cloned()
            .zip(candidate.iter().cloned())
            .collect();
        let mut cmp = Self::default();
        cmp.push_chunk(None, &pairs);
        cmp
    }

    /// `build_off_identity` (:813-843). Status `ok` only when every paired
    /// frame is byte-identical; `failed` on a shape mismatch.
    pub fn off_identity(&self) -> Value {
        if let Some((i, b, c)) = self.shape_mismatch {
            return json!({"status": "failed", "reason": "shape_mismatch", "pair": i, "baseline_shape": b, "candidate_shape": c});
        }
        if self.pairs == 0 {
            return json!({"status": "blocked", "reason": "no_frame_pairs"});
        }
        json!({
            "status": if self.nonidentical == 0 { "ok" } else { "different" },
            "pairs": self.pairs,
            "nonidentical_frames": self.nonidentical,
            "max_abs_diff_uint8": self.max_abs,
        })
    }

    pub fn off_identity_ok(&self) -> bool {
        self.shape_mismatch.is_none() && self.pairs > 0 && self.max_abs == 0
    }

    /// `build_pixel_metrics` (:845-925), same keys and aggregates. PSNR means
    /// and minima skip identical (infinite) frames and are null when every
    /// frame is identical (:899, :904-905); `psnr_per_frame` keeps them.
    pub fn pixel_metrics(&self) -> Value {
        if let Some((i, b, c)) = self.shape_mismatch {
            return json!({"status": "blocked", "reason": "shape_mismatch", "pair": i, "baseline_shape": b, "candidate_shape": c});
        }
        if self.pairs == 0 {
            return json!({"status": "blocked", "reason": "no_frame_pairs"});
        }
        let finite: Vec<f64> = self
            .psnr
            .iter()
            .copied()
            .filter(|p| p.is_finite())
            .collect();
        let opt = |v: &[f64], f: fn(&[f64]) -> f64| -> Value {
            if v.is_empty() {
                Value::Null
            } else {
                json!(f(v))
            }
        };
        let or = |v: &[f64], f: fn(&[f64]) -> f64, d: f64| if v.is_empty() { d } else { f(v) };
        let by_size = |f: fn(&[f64]) -> f64| -> Value {
            PATCH_BOUNDARY_SIZES
                .iter()
                .zip(&self.patch_by_size)
                .map(|(p, v)| (p.to_string(), json!(or(v, f, 0.0))))
                .collect::<serde_json::Map<_, _>>()
                .into()
        };
        json!({
            "status": "ok",
            "pairs": self.pairs,
            "mse_mean": fmean(&self.mse),
            "mse_max": fmax(&self.mse),
            "mean_abs_pixel_diff": fmean(&self.mae),
            "psnr_mean": opt(&finite, fmean),
            "psnr_min": opt(&finite, fmin),
            "psnr_per_frame": self.psnr.iter().map(|&p| jf(p)).collect::<Vec<_>>(),
            "sharpness_ratio_mean": fmean(&self.sharpness),
            "patch_boundary_ratio_mean": fmean(&self.patch),
            "patch_boundary_ratio_max": fmax(&self.patch),
            "patch_boundary_ratio_by_size_mean": by_size(fmean),
            "patch_boundary_ratio_by_size_max": by_size(fmax),
            "temporal_delta_error_mean": or(&self.temporal_err, fmean, 0.0),
            "temporal_delta_error_max": or(&self.temporal_err, fmax, 0.0),
            "temporal_jitter_ratio_mean": or(&self.jitter, fmean, 1.0),
            "temporal_jitter_ratio_min": or(&self.jitter, fmin, 1.0),
            "temporal_jitter_ratio_max": or(&self.jitter, fmax, 1.0),
        })
    }
}

/// The directory holding a clip's frames: `<dir>/frames` when it exists, else
/// `<dir>` itself (our gen stages write `frame-NNN.png` straight into
/// `--clip-dir` / `--clip`).
pub fn frames_dir(dir: &Path) -> PathBuf {
    let sub = dir.join("frames");
    if sub.is_dir() {
        sub
    } else {
        dir.to_path_buf()
    }
}

/// Sorted frame PNGs of a clip. Ordered by the trailing frame number so
/// `frame-1000.png` follows `frame-999.png`; subdirectories (`warmup/`,
/// `cold/`) are not descended into.
pub fn list_frames(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let fd = frames_dir(dir);
    let mut v: Vec<PathBuf> = std::fs::read_dir(&fd)
        .map_err(|e| anyhow::anyhow!("{}: {e}", fd.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x.eq_ignore_ascii_case("png")))
        .collect();
    let key = |p: &PathBuf| {
        let stem = p
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let digits: String = stem
            .chars()
            .rev()
            .take_while(char::is_ascii_digit)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        (digits.parse::<u64>().ok(), stem)
    };
    v.sort_by_key(key);
    Ok(v)
}

fn nonempty(p: &Path) -> bool {
    std::fs::metadata(p).is_ok_and(|m| m.len() > 0)
}

/// `detect_log_errors` (:322-328).
fn log_errors(log: &Path) -> Vec<&'static str> {
    let Ok(bytes) = std::fs::read(log) else {
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&bytes).to_lowercase();
    ERROR_PATTERNS
        .iter()
        .copied()
        .filter(|p| text.contains(p))
        .collect()
}

/// `determine_status` (:513-542) for one clip: the video is `output.mp4` in
/// the frames directory, the log `run.log` there or the matrix cell's
/// `stderr.log` beside it. sol-engine's `prepared/submitted/running` branch
/// needs job metadata we do not have and is left out.
pub fn clip_status(dir: &Path) -> (String, Vec<String>, Option<PathBuf>) {
    let fd = frames_dir(dir);
    let video = fd.join("output.mp4");
    let log = [fd.join("run.log"), dir.join("run.log")]
        .into_iter()
        .chain(fd.parent().map(|p| p.join("stderr.log")))
        // A prompt set: <cell>/frames/<prompt>/, the log two levels up.
        .chain(
            fd.parent()
                .and_then(Path::parent)
                .map(|p| p.join("stderr.log")),
        )
        .find(|p| p.exists());
    let mut notes = Vec::new();
    let errors = log.as_deref().map(log_errors).unwrap_or_default();
    let status = if !errors.is_empty() {
        notes.push(format!(
            "log contains error patterns: {}",
            errors.join(", ")
        ));
        "failed"
    } else {
        match (nonempty(&video), log.as_deref().is_some_and(nonempty)) {
            (true, true) => "completed",
            (false, true) => {
                notes.push("run log exists but output.mp4 is missing or empty".into());
                "failed"
            }
            (true, false) if log.is_some() => {
                notes.push("output.mp4 exists but the run log is empty".into());
                "failed"
            }
            (true, false) => {
                notes.push("no run log beside the clip".into());
                "completed"
            }
            (false, false) => {
                notes.push("required artifacts are missing".into());
                "blocked"
            }
        }
    };
    (status.to_string(), notes, log)
}

/// Stream both clips from disk in chunks (decode in parallel, bounded memory).
pub fn compare_dirs(
    baseline: &Path,
    candidate: &Path,
) -> anyhow::Result<(ClipComparison, usize, usize)> {
    let (fb, fc) = (list_frames(baseline)?, list_frames(candidate)?);
    let n = fb.len().min(fc.len());
    let chunk = rayon::current_num_threads().clamp(1, 8);
    let mut cmp = ClipComparison::default();
    let mut prev: Option<(Rgb, Rgb)> = None;
    let mut i = 0;
    while i < n {
        let end = (i + chunk).min(n);
        let pairs: Vec<(Rgb, Rgb)> = (i..end)
            .into_par_iter()
            .map(|k| Ok((Rgb::load(&fb[k])?, Rgb::load(&fc[k])?)))
            .collect::<anyhow::Result<_>>()?;
        cmp.push_chunk(prev.as_ref(), &pairs);
        if cmp.shape_mismatch.is_some() {
            break;
        }
        prev = pairs.into_iter().last();
        i = end;
    }
    Ok((cmp, fb.len(), fc.len()))
}

/// What sol-engine's `evals/tiers.toml` says about thresholds, recorded so a
/// report states which gates it applied and why the rest are telemetry.
fn thresholds(off_identity: bool) -> Value {
    json!({
        "source": "sol-engine evals/tiers.toml",
        "speedup_targets": {"low": 1.5, "medium": 2.0, "high": 3.0},
        "selection": "best_quality_at_or_above_speed_target",
        "quality_ranking": {
            "primary": "aligned_pairwise_gemini_max_artifact_severity",
            "secondary": "aligned_lpips_max",
            "tie_breaker": "higher_speedup",
        },
        "lpips_absolute_threshold": "disabled_for_lossy_generative_dimensions",
        "gemini_absolute_threshold": "disabled_for_lossy_generative_dimensions",
        "pixel_metrics": "telemetry (no absolute threshold)",
        "off_identity": if off_identity { "hard gate: max_abs_diff_uint8 == 0" } else { "not requested" },
    })
}

fn clip_json(dir: &Path, frames: usize) -> Value {
    let (status, notes, log) = clip_status(dir);
    json!({
        "dir": dir,
        "frames_dir": frames_dir(dir),
        "frames": frames,
        "mp4": frames_dir(dir).join("output.mp4").exists().then(|| frames_dir(dir).join("output.mp4")),
        "log": log,
        "status": status,
        "notes": notes,
    })
}

/// The `lpips` judge block (collect_run.py `run_lpips_judge` shape).
fn lpips_judge(
    cmp: &ClipComparison,
    baseline: &Path,
    candidate: &Path,
    o: &LpipsOpts,
) -> anyhow::Result<Value> {
    use crate::lpips::{LPIPS_MAX_PAIRS, LPIPS_STRATIFIED_PAIRS, LPIPS_WORST_CASE_PAIRS};
    let n = cmp.pairs.min(cmp.mae.len());
    let pairs = if o.all_pairs {
        (0..n).collect()
    } else {
        crate::lpips::select_pairs(
            &cmp.mae[..n],
            LPIPS_STRATIFIED_PAIRS,
            LPIPS_WORST_CASE_PAIRS,
            LPIPS_MAX_PAIRS,
        )
    };
    let scorer = crate::lpips::Lpips::load(&o.weights, o.device)?;
    let (fb, fc) = (list_frames(baseline)?, list_frames(candidate)?);
    let result = crate::lpips::judge(&scorer, &fb, &fc, &pairs)?;
    Ok(
        json!({"status": "complete", "pairs_scored": pairs.len(), "selection": if o.all_pairs { "all" } else { "stratified_32_plus_worst_16_max_48" }, "result": result}),
    )
}

pub fn run(
    report: &mut Report,
    baseline: &Path,
    candidate: &Path,
    off_identity: bool,
    lpips: Option<&LpipsOpts>,
) -> StageResult<()> {
    let (cmp, nb, nc) = compare_dirs(baseline, candidate)?;
    report.set("baseline", clip_json(baseline, nb));
    report.set("candidate", clip_json(candidate, nc));
    report.set("thresholds", thresholds(off_identity));
    match lpips {
        None => report.set(
            "lpips",
            json!({"status": "deferred", "reason": "disabled (pass --lpips <weights dir>)"}),
        ),
        Some(_) if cmp.shape_mismatch.is_some() || cmp.pairs == 0 => report.set(
            "lpips",
            json!({"status": "blocked", "reason": "no comparable frame pairs"}),
        ),
        Some(o) => {
            // A scorer failure (weights, device) blocks the judge, as
            // collect_run.py's `lpips_judge_failed`, not the pixel metrics.
            let judge = lpips_judge(&cmp, baseline, candidate, o).unwrap_or_else(|e| {
                eprintln!("lpips: {e:#}");
                json!({"status": "blocked", "reason": "lpips_judge_failed", "error": format!("{e:#}")})
            });
            let r = &judge["result"];
            report.set("lpips_mean", &r["mean"]);
            report.set("lpips_max", &r["max"]);
            report.set("lpips_median", &r["median"]);
            report.note(
                "lpips",
                json!({"mean": r["mean"], "max": r["max"], "n": r["n"], "backend": r["backend"], "seconds": r["seconds"]}),
            );
            report.set("lpips", judge);
        }
    }
    let identity = cmp.off_identity();
    let pixels = cmp.pixel_metrics();
    report.set("off_identity", &identity);
    report.set("pixel_metrics", &pixels);
    if nb == 0 || nc == 0 {
        return Err(StageError::Check(format!(
            "frames missing: baseline {nb}, candidate {nc}"
        )));
    }
    report.note(
        "frame_counts",
        json!({"baseline": nb, "candidate": nc, "pairs": cmp.pairs, "match": nb == nc}),
    );
    report.check(
        "frame_shape",
        cmp.shape_mismatch.is_none(),
        json!({"shape_mismatch": cmp.shape_mismatch.map(|(i, b, c)| json!({"pair": i, "baseline": b, "candidate": c}))}),
        json!({}),
    )?;
    let mut summary = pixels.clone();
    if let Value::Object(m) = &mut summary {
        m.remove("psnr_per_frame");
    }
    report.note("pixel_metrics", summary);
    if off_identity {
        report.check(
            "off_identity",
            cmp.off_identity_ok(),
            identity,
            json!({"max_abs_diff_uint8": 0}),
        )?;
        report.check(
            "off_identity_frame_count",
            nb == nc,
            json!({"baseline": nb, "candidate": nc}),
            json!({"equal": true}),
        )?;
    } else {
        report.note("off_identity", identity);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, f: impl Fn(usize, usize, usize) -> u8) -> Rgb {
        let mut px = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    px.push(f(x, y, c));
                }
            }
        }
        Rgb { w, h, px }
    }

    /// A smooth ramp that moves one pixel per frame.
    fn moving_clip(n: usize) -> Vec<Rgb> {
        (0..n)
            .map(|t| frame(64, 48, move |x, y, c| (x + y + t + c * 5) as u8))
            .collect()
    }

    fn num(v: &Value, k: &str) -> f64 {
        v[k].as_f64()
            .unwrap_or_else(|| panic!("{k} missing in {v}"))
    }

    #[test]
    fn identical_clips_are_off_identical_with_infinite_psnr() {
        let a = moving_clip(4);
        let cmp = ClipComparison::of(&a, &a);
        assert!(cmp.off_identity_ok());
        let id = cmp.off_identity();
        assert_eq!(id["status"], "ok");
        assert_eq!(id["max_abs_diff_uint8"], 0);
        let m = cmp.pixel_metrics();
        assert_eq!(num(&m, "mse_mean"), 0.0);
        assert_eq!(num(&m, "mean_abs_pixel_diff"), 0.0);
        assert!(cmp.psnr.iter().all(|p| p.is_infinite()));
        assert!(m["psnr_mean"].is_null() && m["psnr_min"].is_null());
        assert_eq!(m["psnr_per_frame"][0], "inf");
        assert_eq!(num(&m, "sharpness_ratio_mean"), 1.0);
        assert_eq!(num(&m, "patch_boundary_ratio_max"), 1.0);
        assert_eq!(num(&m, "temporal_delta_error_max"), 0.0);
        assert_eq!(num(&m, "temporal_jitter_ratio_mean"), 1.0);
    }

    #[test]
    fn one_lsb_breaks_off_identity() {
        let a = moving_clip(3);
        let mut b = a.clone();
        b[1].px[100] += 1;
        let cmp = ClipComparison::of(&a, &b);
        assert!(!cmp.off_identity_ok());
        let id = cmp.off_identity();
        assert_eq!(id["status"], "different");
        assert_eq!(id["nonidentical_frames"], 1);
        assert_eq!(id["max_abs_diff_uint8"], 1);
        // mse = 1 / N on that frame → psnr = 20 log10(255 sqrt(N)).
        let n = (64 * 48 * 3) as f64;
        let want = 20.0 * (255.0 * n.sqrt()).log10();
        assert!((cmp.psnr[1] - want).abs() < 1e-9);
        assert!(cmp.psnr[0].is_infinite() && cmp.psnr[2].is_infinite());
        let m = cmp.pixel_metrics();
        assert!((num(&m, "psnr_mean") - want).abs() < 1e-9);
    }

    #[test]
    fn known_noise_gives_known_psnr() {
        // ±5 on every value (checkerboard of signs) around mid grey:
        // mse = 25, mae = 5, psnr = 20 log10(255 / 5) = 34.15 dB.
        let base = vec![frame(32, 32, |_, _, _| 128)];
        let noisy = vec![frame(
            32,
            32,
            |x, y, c| {
                if (x + y + c) % 2 == 0 {
                    133
                } else {
                    123
                }
            },
        )];
        let m = ClipComparison::of(&base, &noisy).pixel_metrics();
        assert_eq!(num(&m, "mse_mean"), 25.0);
        assert_eq!(num(&m, "mse_max"), 25.0);
        assert_eq!(num(&m, "mean_abs_pixel_diff"), 5.0);
        let want = 20.0 * (255.0f64 / 5.0).log10();
        assert!((num(&m, "psnr_mean") - want).abs() < 1e-12);
        assert!((want - 34.151).abs() < 1e-3);
    }

    #[test]
    fn gradient_and_boundary_formulas() {
        // Columns alternate 0/10: every horizontal step is 10, no vertical.
        let a = frame(4, 3, |x, _, _| if x % 2 == 0 { 0 } else { 10 });
        assert_eq!(mean_abs_gradient(&a), 10.0);
        // One pixel wide, one pixel tall: both terms are zero.
        assert_eq!(mean_abs_gradient(&frame(1, 1, |_, _, _| 7)), 0.0);
        // 16x16 frame, patch 8: one grid column (x=8) and one grid row (y=8).
        // Step of 20 across x=8, none across y=8 → mean(20, 0) = 10.
        let b = frame(16, 16, |x, _, _| if x < 8 { 0 } else { 20 });
        assert_eq!(patch_boundary_score(&b, 8), 10.0);
        assert_eq!(patch_boundary_score(&b, 16), 0.0);
    }

    #[test]
    fn blocky_candidate_raises_patch_boundary_ratio() {
        let base = vec![frame(64, 64, |x, y, _| (x + y) as u8)];
        // Each 16x16 block takes its mean: flat inside, jumps on the grid.
        let blocky = vec![frame(64, 64, |x, y, _| {
            let (bx, by) = (x / 16 * 16, y / 16 * 16);
            (bx + by + 15) as u8
        })];
        let cmp = ClipComparison::of(&base, &blocky);
        let m = cmp.pixel_metrics();
        assert!(num(&m, "patch_boundary_ratio_max") > 1.0, "{m}");
        let by = &m["patch_boundary_ratio_by_size_max"];
        // Base steps by 1 across every grid line, blocky by 16 across the
        // 16- and 32-pixel lines.
        assert!((by["16"].as_f64().unwrap() - 16.0).abs() < 1e-9, "{by}");
        assert!((by["32"].as_f64().unwrap() - 16.0).abs() < 1e-9, "{by}");
        assert!(
            num(&m, "sharpness_ratio_mean") < 1.0,
            "blocks are flat inside"
        );
    }

    #[test]
    fn jittery_candidate_raises_temporal_jitter_ratio() {
        let base = moving_clip(6);
        // Flicker ±8 on alternate frames on top of the same motion.
        let jitter: Vec<Rgb> = base
            .iter()
            .enumerate()
            .map(|(t, f)| Rgb {
                px: f
                    .px
                    .iter()
                    .map(|&v| if t % 2 == 0 { v + 8 } else { v })
                    .collect(),
                ..f.clone()
            })
            .collect();
        let m = ClipComparison::of(&base, &jitter).pixel_metrics();
        assert!(num(&m, "temporal_jitter_ratio_mean") > 1.0, "{m}");
        assert!(num(&m, "temporal_jitter_ratio_min") > 1.0, "{m}");
        // |cand_delta - base_delta| is exactly 8 on every step.
        assert_eq!(num(&m, "temporal_delta_error_mean"), 8.0);
        assert_eq!(num(&m, "temporal_delta_error_max"), 8.0);
    }

    #[test]
    fn shape_mismatch_fails_off_identity_and_blocks_metrics() {
        let a = vec![frame(8, 8, |_, _, _| 0)];
        let b = vec![frame(8, 4, |_, _, _| 0)];
        let cmp = ClipComparison::of(&a, &b);
        assert!(!cmp.off_identity_ok());
        assert_eq!(cmp.off_identity()["status"], "failed");
        assert_eq!(cmp.pixel_metrics()["status"], "blocked");
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fv-clipcmp-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn save(dir: &Path, frames: &[Rgb]) {
        for (i, f) in frames.iter().enumerate() {
            image::RgbImage::from_raw(f.w as u32, f.h as u32, f.px.clone())
                .unwrap()
                .save(dir.join(format!("frame-{i:03}.png")))
                .unwrap();
        }
    }

    #[test]
    fn dirs_round_trip_through_png_and_chunks() {
        let a = moving_clip(11);
        let mut b = a.clone();
        b[7].px[0] += 3;
        let (da, db) = (scratch("a"), scratch("b"));
        save(&da, &a);
        // The candidate keeps its frames under frames/, the other layout.
        std::fs::create_dir_all(db.join("frames")).unwrap();
        save(&db.join("frames"), &b);
        let (cmp, na, nb) = compare_dirs(&da, &db).unwrap();
        assert_eq!((na, nb, cmp.pairs), (11, 11, 11));
        let mem = ClipComparison::of(&a, &b);
        assert_eq!(cmp.pixel_metrics(), mem.pixel_metrics());
        assert_eq!(cmp.off_identity()["max_abs_diff_uint8"], 3);
        let _ = std::fs::remove_dir_all(&da);
        let _ = std::fs::remove_dir_all(&db);
    }

    #[test]
    fn frames_sort_numerically() {
        let d = scratch("sort");
        for i in [2usize, 1000, 999, 10] {
            std::fs::write(d.join(format!("frame-{i:03}.png")), b"x").unwrap();
        }
        std::fs::create_dir_all(d.join("warmup")).unwrap();
        let names: Vec<String> = list_frames(&d)
            .unwrap()
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            [
                "frame-002.png",
                "frame-010.png",
                "frame-999.png",
                "frame-1000.png"
            ]
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn status_follows_determine_status() {
        let cell = scratch("status");
        let fd = cell.join("frames");
        std::fs::create_dir_all(&fd).unwrap();
        assert_eq!(clip_status(&fd).0, "blocked");
        std::fs::write(cell.join("stderr.log"), "[PASS] h3/frames\n").unwrap();
        assert_eq!(clip_status(&fd).0, "failed", "log without mp4");
        std::fs::write(fd.join("output.mp4"), b"mp4").unwrap();
        assert_eq!(clip_status(&fd).0, "completed");
        std::fs::write(cell.join("stderr.log"), "CUDA out of memory\n").unwrap();
        let (s, notes, _) = clip_status(&fd);
        assert_eq!(s, "failed");
        assert!(notes[0].contains("cuda out of memory"));
        let _ = std::fs::remove_dir_all(&cell);
    }
}
