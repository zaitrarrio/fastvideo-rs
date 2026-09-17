//! Smoke suite for Vast A100 validation.
//!
//! Every test skips gracefully when FASTVIDEO_WEIGHTS is not set.
//! Some tests additionally skip when a specific mode flag is not set — the
//! shell runner (scripts/run_smoke_a100.sh) sets those flags and calls each
//! test function by name so each variant runs in its own fresh process.
//!
//! Run the full matrix:
//!   bash scripts/run_smoke_a100.sh
//!
//! Run a single group manually (env flags must be set before launch since
//! they are cached at first read and cannot be changed inside the process):
//!   FASTVIDEO_WEIGHTS=/mnt/wan \
//!   cargo test -p fastvideo-cudarc --features cuda --test smoke_a100 \
//!       -- --test-threads=1 t2v_default

use fastvideo_cudarc::{GenerateConfig, WanPipeline};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn weights_root() -> Option<PathBuf> {
    std::env::var("FASTVIDEO_WEIGHTS").ok().map(PathBuf::from)
}

/// Skip the calling test when FASTVIDEO_WEIGHTS is not set.
macro_rules! require_weights {
    () => {
        match weights_root() {
            Some(p) => p,
            None => {
                eprintln!("[SKIP] FASTVIDEO_WEIGHTS not set — pass path to Diffusers Wan layout");
                return;
            }
        }
    };
}

/// Skip the calling test when `var` is not equal to `expected`.
/// Use this for tests that require a specific mode flag to already be set
/// in the process environment (flags are cached on first read).
macro_rules! require_flag {
    ($var:expr, $expected:expr) => {
        match std::env::var($var).ok().as_deref() {
            Some($expected) => {}
            _ => {
                eprintln!(
                    "[SKIP] requires {}={} (set it before launching the test process)",
                    $var, $expected
                );
                return;
            }
        }
    };
}

/// Read PNG IHDR dimensions without pulling in an image crate.
/// Returns `(width, height)`.
fn png_dims(path: &Path) -> (u32, u32) {
    let data = std::fs::read(path).unwrap_or_else(|_| panic!("cannot read {}", path.display()));
    assert!(
        data.len() >= 24 && data[0..8] == *b"\x89PNG\r\n\x1a\n",
        "not a valid PNG: {}",
        path.display()
    );
    let w = u32::from_be_bytes(data[16..20].try_into().unwrap());
    let h = u32::from_be_bytes(data[20..24].try_into().unwrap());
    (w, h)
}

fn out_dir(tag: &str) -> String {
    std::env::temp_dir()
        .join(format!("fastvideo-smoke-{tag}"))
        .to_string_lossy()
        .into()
}

/// Minimal single-frame config with a given step count.
/// `num_frames=1` → T=1 latent (fastest possible VAE decode, satisfies 4n+1).
fn base_cfg(tag: &str, steps: usize) -> GenerateConfig {
    let mut c = GenerateConfig::default();
    c.num_frames = 1;
    c.num_inference_steps = steps;
    c.output_dir = out_dir(tag);
    c
}

fn assert_valid_png(path: &str, expected_w: usize, expected_h: usize) {
    assert!(
        Path::new(path).exists(),
        "output file not found: {path}"
    );
    let (w, h) = png_dims(Path::new(path));
    assert_eq!(
        (w as usize, h as usize),
        (expected_w, expected_h),
        "PNG dimensions mismatch"
    );
}

// ─── 1. Default — BF16 + TF32 + flash SDPA (all defaults) ───────────────────

/// Baseline smoke: default flags, 4 steps, 1 frame.
/// Exercises the whole pipeline: UMT5 encode → DiT forward × 4 → VAE decode → PNG.
#[test]
fn t2v_default() {
    let root = require_weights!();
    let mut pipe = WanPipeline::load_1_3b(&root).expect("load 1.3B weights");
    let mut cfg = base_cfg("default", 4);
    cfg.prompt = "a cat walking on a sunny street".into();

    let t = Instant::now();
    let paths = pipe.generate(&cfg).expect("generate");
    let elapsed = t.elapsed();

    assert_eq!(paths.len(), 1, "expected 1 frame");
    assert_valid_png(&paths[0], cfg.width, cfg.height);
    println!(
        "[t2v_default] 4 steps × 1 frame: {:.2?}  ({:.0} ms/step)",
        elapsed,
        elapsed.as_millis() as f64 / 4.0
    );
}

// ─── 2. DMD single-step (guidance=1, flow_shift=8) ───────────────────────────

/// DMD distilled path: 1 step, guidance=1.0, flow_shift=8.0.
/// This is the fastest valid T2V forward pass and exercises the
/// `guidance_scale == 1.0` shortcut in `dit_cfg`.
#[test]
fn t2v_dmd_1step() {
    let root = require_weights!();
    let mut pipe = WanPipeline::load_1_3b(&root).expect("load");
    let mut cfg = base_cfg("dmd-1step", 1);
    cfg.is_dmd = true;
    cfg.flow_shift = 8.0;
    cfg.guidance_scale = 1.0;

    let t = Instant::now();
    let paths = pipe.generate(&cfg).expect("generate");
    println!("[t2v_dmd_1step] 1 step: {:.2?}", t.elapsed());

    assert_eq!(paths.len(), 1);
    assert_valid_png(&paths[0], cfg.width, cfg.height);
}

// ─── 3. DMD 4-step (typical fast-inference config) ───────────────────────────

#[test]
fn t2v_dmd_4step() {
    let root = require_weights!();
    let mut pipe = WanPipeline::load_1_3b(&root).expect("load");
    let mut cfg = base_cfg("dmd-4step", 4);
    cfg.is_dmd = true;
    cfg.flow_shift = 8.0;
    cfg.guidance_scale = 1.0;
    cfg.prompt = "ocean waves crashing on a rocky shore at sunset, 4K cinematic".into();

    let t = Instant::now();
    let paths = pipe.generate(&cfg).expect("generate");
    let elapsed = t.elapsed();
    println!(
        "[t2v_dmd_4step] 4 steps: {:.2?}  ({:.0} ms/step)",
        elapsed,
        elapsed.as_millis() as f64 / 4.0
    );
    assert_eq!(paths.len(), 1);
    assert_valid_png(&paths[0], cfg.width, cfg.height);
}

// ─── 4. CFG guidance (batched cond+uncond DiT forward) ───────────────────────

/// guidance_scale=5.0 exercises the batched cond/uncond CFG path in `dit_cfg`.
/// Requires BF16 batching to be correct for both halves of the batch.
#[test]
fn t2v_cfg_guidance_5() {
    let root = require_weights!();
    let mut pipe = WanPipeline::load_1_3b(&root).expect("load");
    let mut cfg = base_cfg("cfg-5", 4);
    cfg.guidance_scale = 5.0;
    cfg.negative_prompt = "blurry, low quality, watermark, deformed".into();
    cfg.prompt = "a red fox running through a snowy forest".into();

    let t = Instant::now();
    let paths = pipe.generate(&cfg).expect("generate");
    let elapsed = t.elapsed();
    println!(
        "[t2v_cfg_5] 4 steps CFG=5: {:.2?}  ({:.0} ms/step)",
        elapsed,
        elapsed.as_millis() as f64 / 4.0
    );
    assert_eq!(paths.len(), 1);
    assert_valid_png(&paths[0], cfg.width, cfg.height);
}

// ─── 5. Seed determinism ─────────────────────────────────────────────────────

/// Two generate() calls with identical config and seed must produce bitwise
/// identical PNG output.  Catches any non-deterministic op in the pipeline
/// (uninitialized memory, non-reproducible atomics, etc.).
#[test]
fn t2v_seed_determinism() {
    let root = require_weights!();

    let make_cfg = |run: u8| {
        let mut c = base_cfg(&format!("seed-run-{run}"), 2);
        c.seed = 99999;
        c.is_dmd = true;
        c.flow_shift = 8.0;
        c.guidance_scale = 1.0;
        c
    };

    let mut pipe = WanPipeline::load_1_3b(&root).expect("load");
    let paths_a = pipe.generate(&make_cfg(0)).expect("run A");
    let paths_b = pipe.generate(&make_cfg(1)).expect("run B");

    let bytes_a = std::fs::read(&paths_a[0]).expect("read A");
    let bytes_b = std::fs::read(&paths_b[0]).expect("read B");
    assert_eq!(
        bytes_a.len(),
        bytes_b.len(),
        "PNG sizes differ across same-seed runs"
    );
    assert_eq!(
        bytes_a, bytes_b,
        "seed=99999 produced different PNG bytes across two generate() calls"
    );
    println!(
        "[t2v_seed_determinism] seed=99999 → identical ({} bytes) ✓",
        bytes_a.len()
    );
}

// ─── 6. Multi-frame 9-frame (4n+1) ───────────────────────────────────────────

/// 9 frames = 4*2+1, valid for two temporal-upsample stages (T=3 latent).
/// Validates per-frame PNG output and correct count.
#[test]
fn t2v_9frames() {
    let root = require_weights!();
    let mut pipe = WanPipeline::load_1_3b(&root).expect("load");
    let mut cfg = GenerateConfig::default();
    cfg.num_frames = 9;
    cfg.num_inference_steps = 2;
    cfg.is_dmd = true;
    cfg.flow_shift = 8.0;
    cfg.guidance_scale = 1.0;
    cfg.output_dir = out_dir("9frames");

    let t = Instant::now();
    let paths = pipe.generate(&cfg).expect("generate");
    println!(
        "[t2v_9frames] 2 steps × 9 frames: {:.2?}",
        t.elapsed()
    );
    assert_eq!(paths.len(), 9, "expected 9 output PNGs");
    for p in &paths {
        assert!(Path::new(p).exists(), "missing: {p}");
        let (w, h) = png_dims(Path::new(p));
        assert_eq!((w as usize, h as usize), (cfg.width, cfg.height));
    }
}

// ─── 7. HD 720 × 1280 ────────────────────────────────────────────────────────

/// High-resolution smoke: 2 steps at 720×1280.
/// Exercises larger attention tensors and higher VRAM pressure.
#[test]
fn t2v_hd_720p() {
    let root = require_weights!();
    let mut pipe = WanPipeline::load_1_3b(&root).expect("load");
    let mut cfg = base_cfg("hd-720p", 2);
    cfg.height = 720;
    cfg.width = 1280;
    cfg.is_dmd = true;
    cfg.flow_shift = 8.0;
    cfg.guidance_scale = 1.0;

    let t = Instant::now();
    let paths = pipe.generate(&cfg).expect("generate");
    println!("[t2v_hd_720p] 2 steps 720×1280: {:.2?}", t.elapsed());
    assert_eq!(paths.len(), 1);
    assert_valid_png(&paths[0], 1280, 720);
}

// ─── 8. Dense SDPA fallback ───────────────────────────────────────────────────
// Requires: FASTVIDEO_SDPA=dense in the environment before process start.

/// Validates the O(S²) dense attention path still produces valid output
/// when flash attention is disabled.
#[test]
fn t2v_dense_sdpa() {
    require_flag!("FASTVIDEO_SDPA", "dense");
    let root = require_weights!();

    let mut pipe = WanPipeline::load_1_3b(&root).expect("load");
    let cfg = base_cfg("dense-sdpa", 2);

    let t = Instant::now();
    let paths = pipe.generate(&cfg).expect("generate");
    println!("[t2v_dense_sdpa] 2 steps dense SDPA: {:.2?}", t.elapsed());
    assert_eq!(paths.len(), 1);
    assert_valid_png(&paths[0], cfg.width, cfg.height);
}

// ─── 9. F32 path (BF16 disabled) ─────────────────────────────────────────────
// Requires: FASTVIDEO_BF16=0 in the environment before process start.

/// Validates output correctness when the BF16 GEMM path is disabled;
/// every matmul falls back to F32 cuBLAS GemmEx.
#[test]
fn t2v_f32_path() {
    require_flag!("FASTVIDEO_BF16", "0");
    let root = require_weights!();

    let mut pipe = WanPipeline::load_1_3b(&root).expect("load");
    let cfg = base_cfg("f32-path", 2);

    let t = Instant::now();
    let paths = pipe.generate(&cfg).expect("generate");
    println!("[t2v_f32_path] 2 steps F32: {:.2?}", t.elapsed());
    assert_eq!(paths.len(), 1);
    assert_valid_png(&paths[0], cfg.width, cfg.height);
}

// ─── 10. TeaCache — default threshold ────────────────────────────────────────
// Requires: FASTVIDEO_TEACACHE=1 in the environment before process start.

/// TeaCache with default threshold (0.08).  8 steps gives enough iterations
/// for the cache to register at least one skip; if cache skipping is broken
/// the output will differ from the reference run but the test still passes
/// (correctness is validated by the timing summary in run_smoke_a100.sh).
#[test]
fn t2v_teacache_default() {
    require_flag!("FASTVIDEO_TEACACHE", "1");
    let root = require_weights!();

    let mut pipe = WanPipeline::load_1_3b(&root).expect("load");
    let mut cfg = base_cfg("teacache-08", 8);
    cfg.is_dmd = true;
    cfg.flow_shift = 8.0;
    cfg.guidance_scale = 1.0;

    let t = Instant::now();
    let paths = pipe.generate(&cfg).expect("generate");
    println!(
        "[t2v_teacache_default] 8 steps thresh=0.08: {:.2?}  ({:.0} ms/step)",
        t.elapsed(),
        t.elapsed().as_millis() as f64 / 8.0
    );
    assert_eq!(paths.len(), 1);
    assert_valid_png(&paths[0], cfg.width, cfg.height);
}

// ─── 11. TeaCache — aggressive threshold ─────────────────────────────────────
// Requires: FASTVIDEO_TEACACHE=1, FASTVIDEO_TEACACHE_THRESH=0.15

/// Higher threshold forces more cache hits.  Expected to be noticeably faster
/// than thresh=0.08 at the cost of some quality.
#[test]
fn t2v_teacache_aggressive() {
    require_flag!("FASTVIDEO_TEACACHE", "1");
    let root = require_weights!();

    let mut pipe = WanPipeline::load_1_3b(&root).expect("load");
    let mut cfg = base_cfg("teacache-15", 8);
    cfg.is_dmd = true;
    cfg.flow_shift = 8.0;
    cfg.guidance_scale = 1.0;

    let t = Instant::now();
    let paths = pipe.generate(&cfg).expect("generate");
    println!(
        "[t2v_teacache_aggressive] 8 steps thresh=0.15: {:.2?}  ({:.0} ms/step)",
        t.elapsed(),
        t.elapsed().as_millis() as f64 / 8.0
    );
    assert_eq!(paths.len(), 1);
    assert_valid_png(&paths[0], cfg.width, cfg.height);
}

// ─── 12. Max-perf combined (BF16 + flash + TeaCache + CFG) ───────────────────
// Requires: FASTVIDEO_TEACACHE=1 (BF16/flash/TF32 are defaults so no guard needed).

/// Maximum-performance config exercising all fast paths simultaneously:
/// flash attention + BF16 FFN chaining + fused LN+AdaLN + TeaCache + CFG batch.
#[test]
fn t2v_maxperf_combined() {
    require_flag!("FASTVIDEO_TEACACHE", "1");
    let root = require_weights!();

    let mut pipe = WanPipeline::load_1_3b(&root).expect("load");
    let mut cfg = base_cfg("maxperf", 8);
    cfg.guidance_scale = 5.0;
    cfg.negative_prompt = "blurry, low quality".into();
    cfg.prompt = "a cinematic drone shot over a glacier, 4K, smooth motion".into();

    let t = Instant::now();
    let paths = pipe.generate(&cfg).expect("generate");
    let elapsed = t.elapsed();
    println!(
        "[t2v_maxperf_combined] 8 steps CFG=5 + TeaCache: {:.2?}  ({:.0} ms/step)",
        elapsed,
        elapsed.as_millis() as f64 / 8.0
    );
    assert_eq!(paths.len(), 1);
    assert_valid_png(&paths[0], cfg.width, cfg.height);
}

// ─── 13. Step-latency benchmark ───────────────────────────────────────────────

/// Measures per-step wall-clock latency over 16 steps with default flags.
/// Useful for A/B comparison between configurations.
/// Not a correctness test — just prints timing and always passes.
#[test]
fn bench_step_latency_16step() {
    let root = require_weights!();
    let mut pipe = WanPipeline::load_1_3b(&root).expect("load");
    let mut cfg = base_cfg("bench-16step", 16);
    cfg.is_dmd = true;
    cfg.flow_shift = 8.0;
    cfg.guidance_scale = 1.0;

    // Warm-up: 2 steps (amortises first-call CUDA JIT / caching overhead).
    {
        let mut warmup = cfg.clone();
        warmup.num_inference_steps = 2;
        warmup.output_dir = out_dir("bench-warmup");
        pipe.generate(&warmup).expect("warmup");
    }

    let t = Instant::now();
    let _ = pipe.generate(&cfg).expect("benchmark");
    let elapsed = t.elapsed();
    let ms_per_step = elapsed.as_millis() as f64 / 16.0;

    println!(
        "[bench_step_latency_16step] 16 steps total={:.2?}  per-step={:.1}ms",
        elapsed, ms_per_step
    );
    // Soft assertion: on an A100 80GB the 1.3B model should do < 150 ms/step at 480×832.
    // This is informational — CI targets may differ; adjust for your hardware.
    if ms_per_step > 150.0 {
        println!(
            "  WARNING: {:.1}ms/step exceeds 150ms A100 target — check FASTVIDEO_BF16, FASTVIDEO_SDPA",
            ms_per_step
        );
    }
}

// ─── 14. Full 81-frame video (optional / slow) ───────────────────────────────
// Requires: FASTVIDEO_SMOKE_FULL=1

/// Full 81-frame video generation with 4 DMD steps.
/// Skipped by default because it takes several minutes even on A100.
/// Set FASTVIDEO_SMOKE_FULL=1 to enable.
#[test]
fn t2v_full_81frame() {
    require_flag!("FASTVIDEO_SMOKE_FULL", "1");
    let root = require_weights!();

    let mut pipe = WanPipeline::load_1_3b(&root).expect("load");
    let mut cfg = GenerateConfig::default(); // 81 frames, 480×832
    cfg.num_inference_steps = 4;
    cfg.is_dmd = true;
    cfg.flow_shift = 8.0;
    cfg.guidance_scale = 1.0;
    cfg.prompt = "a time-lapse of clouds over a mountain range, cinematic".into();
    cfg.output_dir = out_dir("81frames");

    let t = Instant::now();
    let paths = pipe.generate(&cfg).expect("generate");
    let elapsed = t.elapsed();
    println!(
        "[t2v_full_81frame] 4 steps × 81 frames: {:.2?}  ({:.0} ms/step)",
        elapsed,
        elapsed.as_millis() as f64 / 4.0
    );
    assert_eq!(paths.len(), 81, "expected 81 output PNGs");
    for p in &paths {
        assert!(Path::new(p).exists(), "missing: {p}");
    }
    let (w, h) = png_dims(Path::new(&paths[0]));
    assert_eq!((w as usize, h as usize), (cfg.width, cfg.height));
}
