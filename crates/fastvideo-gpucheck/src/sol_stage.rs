//! Sol-engine comparison stages for LingBot-Video MoE 30B-A3B and
//! Cosmos3-Super 64B (`fv-gpucheck sol …`).
//!
//! * `lingbot-router` — device group-limited top-k router vs the host
//!   reference on random logits (no weights; a cheap kernel check).
//! * `lingbot-gen` — the `models/lingbot_video.toml` two-stage run (base
//!   832×480×121 / 40 steps → 1920×1088 refiner / 8 steps) over the
//!   sol-engine validation prompts, baseline or fullopt arm, per-prompt
//!   `lingbot_timing.json` plus medians in the report.
//! * `cosmos3-gen` — the `models/cosmos3.toml` run (1280×720×189, 35 steps,
//!   guidance 6), baseline or TeaCache arm, optional warmup request.
//!
//! See docs/ports/lingbot.md and docs/ports/cosmos3.md ("Comparison").

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::json;

use crate::report::{Report, StageResult};

#[derive(clap::Subcommand, Debug)]
pub enum Stage {
    /// Device vs host LingBot router decisions on seeded random logits.
    LingbotRouter {
        #[arg(long, default_value_t = 65_536)]
        tokens: usize,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
    /// LingBot-Video MoE two-stage generation (base + 1080p refiner).
    LingbotGen {
        /// Diffusers tree of `robbyant/lingbot-video-moe-30b-a3b`.
        #[arg(long)]
        weights: PathBuf,
        /// Prompt set, one per line (`scripts/gpu/sol/lingbot-t2v-val3.txt`).
        #[arg(long)]
        prompts: Option<PathBuf>,
        /// A single prompt instead of `--prompts`.
        #[arg(long)]
        prompt: Option<String>,
        /// How many prompts of the set to run.
        #[arg(long, default_value_t = 3)]
        num_prompts: usize,
        /// `baseline`, `fullopt`, `easycache`, `pisa` (`FASTVIDEO_LINGBOT_SOL` syntax).
        #[arg(long, default_value = "baseline")]
        arm: String,
        /// `both` (B200: base and refiner resident) or `swap` (96 GB cards).
        #[arg(long, default_value = "swap")]
        residency: String,
        /// Stop after the base stage.
        #[arg(long)]
        no_refine: bool,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        num_steps: Option<usize>,
        #[arg(long)]
        refiner_steps: Option<usize>,
        #[arg(long, default_value = "gpucheck-out/lingbot-gen")]
        clip: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
    /// Cosmos3-Super text-to-video.
    Cosmos3Gen {
        /// Diffusers tree of `nvidia/Cosmos3-Super` (transformer, vae, text_tokenizer).
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "")]
        negative_prompt: String,
        /// `baseline` or `teacache`.
        #[arg(long, default_value = "baseline")]
        arm: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        num_steps: Option<usize>,
        #[arg(long)]
        num_frames: Option<usize>,
        #[arg(long)]
        height: Option<usize>,
        #[arg(long)]
        width: Option<usize>,
        /// Run one untimed request first (sol-engine `WARMUP=true`, 1 step).
        #[arg(long)]
        warm: bool,
        #[arg(long, default_value = "gpucheck-out/cosmos3-gen")]
        clip: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
}

pub fn run(report: &mut Report, stage: &Stage) -> StageResult<()> {
    match stage {
        Stage::LingbotRouter { tokens, device } => router_check(report, *tokens, device),
        Stage::LingbotGen {
            weights,
            prompts,
            prompt,
            num_prompts,
            arm,
            residency,
            no_refine,
            seed,
            num_steps,
            refiner_steps,
            clip,
            device,
        } => {
            report.set("device", crate::gpu::init(device)?);
            let texts = lingbot_prompts(prompts.as_deref(), prompt.as_deref(), *num_prompts)?;
            lingbot_gen(
                report,
                weights,
                &texts,
                arm,
                residency,
                !*no_refine,
                *seed,
                *num_steps,
                *refiner_steps,
                clip,
            )
        }
        Stage::Cosmos3Gen {
            weights,
            prompt,
            negative_prompt,
            arm,
            seed,
            num_steps,
            num_frames,
            height,
            width,
            warm,
            clip,
            device,
        } => {
            report.set("device", crate::gpu::init(device)?);
            let mut r = fastvideo_cudarc::cosmos3::Cosmos3Request::official(prompt.clone());
            r.negative_prompt = (!negative_prompt.is_empty()).then(|| negative_prompt.clone());
            r.seed = *seed;
            r.teacache = match arm.as_str() {
                "baseline" => false,
                "teacache" | "fullopt" => true,
                other => return Err(anyhow::anyhow!("--arm {other}: baseline | teacache").into()),
            };
            if let Some(v) = num_steps {
                r.num_steps = *v;
            }
            if let Some(v) = num_frames {
                r.num_frames = *v;
            }
            if let Some(v) = height {
                r.height = *v;
            }
            if let Some(v) = width {
                r.width = *v;
            }
            cosmos3_gen(report, weights, &r, *warm, clip)
        }
    }
}

/// Prompt lines: one prompt per line (`lingbot-t2v-val3.txt`, the captions
/// already rendered as the reference runner renders them).
fn lingbot_prompts(file: Option<&Path>, single: Option<&str>, n: usize) -> anyhow::Result<Vec<String>> {
    if let Some(p) = single {
        return Ok(vec![p.to_string()]);
    }
    let file = file.ok_or_else(|| anyhow::anyhow!("--prompts or --prompt"))?;
    let text = std::fs::read_to_string(file).with_context(|| format!("read {}", file.display()))?;
    Ok(text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .take(n)
        .map(str::to_string)
        .collect())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.is_empty() {
        return f64::NAN;
    }
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

#[allow(clippy::too_many_arguments)]
fn lingbot_gen(
    report: &mut Report,
    weights: &Path,
    prompts: &[String],
    arm: &str,
    residency: &str,
    refine: bool,
    seed: u64,
    num_steps: Option<usize>,
    refiner_steps: Option<usize>,
    clip: &Path,
) -> StageResult<()> {
    use fastvideo_cudarc::lingbot::pipeline::{LingBotPipeline, LingBotRequest, Residency};
    use fastvideo_models::lingbot::sol::SolArm;
    use fastvideo_models::lingbot::LingBotPreset;

    let sol = SolArm::parse(Some(arm)).map_err(|e| anyhow::anyhow!(e))?;
    let mut pipe = LingBotPipeline::open(weights, LingBotPreset::Moe30b)?;
    pipe.residency = match residency {
        "both" => Residency::Both,
        "swap" => Residency::Swap,
        other => return Err(anyhow::anyhow!("--residency {other}: both | swap").into()),
    };
    let peak = crate::gpu::PeakMem::start();
    let mut runs = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let mut r = LingBotRequest::official(prompt.clone());
        r.seed = seed;
        r.sol = sol;
        if !refine {
            r.refiner = None;
        }
        if let Some(s) = num_steps {
            r.num_steps = s;
        }
        if let (Some(s), Some(f)) = (refiner_steps, r.refiner.as_mut()) {
            f.steps = s;
        }
        let t = pipe.generate(&r, &clip.join(format!("prompt{i}")))?;
        runs.push(serde_json::to_value(&t)?);
        fastvideo_cudarc::wan::stats::phase_reset();
    }
    let req: Vec<f64> = runs.iter().filter_map(|r| r["request_s"].as_f64()).collect();
    let field = |k: &str| median(runs.iter().filter_map(|r| r[k].as_f64()).collect());
    report.note(
        "timings",
        json!({
            "arm": sol.label(),
            "residency": residency,
            "prompts": prompts.len(),
            "request_s_median": median(req.clone()),
            "request_s": req,
            "base_denoise_s_median": field("base_denoise_s"),
            "refiner_denoise_s_median": field("refiner_denoise_s"),
            "refiner_prepare_s_median": field("refiner_prepare_s"),
            "peak_mib": peak.stop(),
            "published_4xgb200_s": {
                "baseline": fastvideo_models::lingbot::sol::PUBLISHED_BASELINE_S,
                "fullopt": fastvideo_models::lingbot::sol::PUBLISHED_FULLOPT_S,
            },
            "runs": runs,
        }),
    );
    report.check("prompts_done", runs.len() == prompts.len(), json!({"done": runs.len()}), json!({"expected": prompts.len()}))?;
    Ok(())
}

fn cosmos3_gen(
    report: &mut Report,
    weights: &Path,
    r: &fastvideo_cudarc::cosmos3::Cosmos3Request,
    warm: bool,
    clip: &Path,
) -> StageResult<()> {
    let mut pipe = fastvideo_cudarc::cosmos3::Cosmos3Pipeline::open(weights)?;
    let peak = crate::gpu::PeakMem::start();
    if warm {
        let mut w = r.clone();
        w.num_steps = 1;
        let t = pipe.generate(&w, &clip.join("warmup"))?;
        report.note("warmup", serde_json::to_value(&t)?);
        fastvideo_cudarc::wan::stats::phase_reset();
    }
    let t = pipe.generate(r, clip)?;
    let mut v = serde_json::to_value(&t)?;
    v["peak_mib"] = json!(peak.stop());
    v["published_4xgb200"] = json!({
        "baseline_s": fastvideo_models::cosmos3::PUBLISHED_BASELINE_S,
        "baseline_denoise_s": fastvideo_models::cosmos3::PUBLISHED_BASELINE_DENOISE_S,
        "fullopt_speedup": fastvideo_models::cosmos3::PUBLISHED_FULLOPT_SPEEDUP,
    });
    report.note("timings", v);
    report.check("steps", t.steps_computed + t.steps_reused == t.steps, json!({"steps": t.steps}), json!({}))?;
    Ok(())
}

fn router_check(report: &mut Report, tokens: usize, device: &str) -> StageResult<()> {
    use fastvideo_models::lingbot::{route, LingBotTransformerConfig, RouterSpec};
    report.set("device", crate::gpu::init(device)?);
    let cfg = LingBotTransformerConfig::moe_30b();
    let spec = RouterSpec::from_config(&cfg);
    let e = spec.num_experts;
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 / (1u64 << 53) as f64) as f32
    };
    let logits: Vec<f32> = (0..tokens * e).map(|_| (next() - 0.5) * 8.0).collect();
    let bias: Vec<f32> = (0..e).map(|_| (next() - 0.5) * 0.1).collect();
    let (hi, hw) = route(&logits, &bias, &spec);
    #[cfg(feature = "cuda")]
    {
        use fastvideo_cudarc::wan::tensor::CudaTensor;
        let mut l = CudaTensor::from_vec(logits, vec![tokens, e])?;
        l.pin_device()?;
        let mut b = CudaTensor::from_vec(bias, vec![e])?;
        b.pin_device()?;
        let p = fastvideo_cudarc::wan::ops::GroupTopk {
            experts: e,
            top_k: spec.top_k,
            n_group: spec.n_group.unwrap_or(1),
            topk_group: spec.topk_group,
            softmax: false,
            norm: spec.norm_topk_prob,
            scale: spec.route_scale,
            round_bf16: spec.round_bf16,
        };
        let t = std::time::Instant::now();
        let (di, dw) = fastvideo_cudarc::wan::ops::moe_group_topk_device(
            l.device_slice().ok_or_else(|| anyhow::anyhow!("logits not on device"))?,
            b.device_slice().ok_or_else(|| anyhow::anyhow!("bias not on device"))?,
            p,
        )?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let k = spec.top_k;
        let (mut diff_sets, mut w_err) = (0usize, 0f32);
        for r in 0..tokens {
            let mut a = di[r * k..(r + 1) * k].to_vec();
            let mut c = hi[r * k..(r + 1) * k].to_vec();
            a.sort_unstable();
            c.sort_unstable();
            if a != c {
                diff_sets += 1;
                continue;
            }
            for (x, y) in dw[r * k..(r + 1) * k].iter().zip(&hw[r * k..(r + 1) * k]) {
                w_err = w_err.max((x - y).abs());
            }
        }
        report.note("router", json!({"tokens": tokens, "device_ms": ms, "different_sets": diff_sets, "max_w_err": w_err}));
        // Fast-math sigmoid may flip a near-tie: allow 0.1 % of rows.
        report.check(
            "router_sets",
            diff_sets * 1000 <= tokens,
            json!({"different_sets": diff_sets}),
            json!({"max": tokens / 1000}),
        )?;
        report.check("router_weights", w_err < 2e-2, json!({"max_w_err": w_err}), json!({"max": 2e-2}))?;
    }
    #[cfg(not(feature = "cuda"))]
    {
        report.note("router", json!({"tokens": tokens, "host_rows": hi.len() / spec.top_k, "weights": hw.len()}));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_lines_and_median() {
        let dir = std::env::temp_dir().join(format!("sol-prompts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("p.txt");
        std::fs::write(&f, "{\"a\":1}\n\nplain\nthird\n").unwrap();
        let p = lingbot_prompts(Some(&f), None, 2).unwrap();
        assert_eq!(p, vec![r#"{"a":1}"#.to_string(), "plain".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(median(vec![3.0, 1.0, 2.0]), 2.0);
    }

    #[test]
    fn vendored_val3_prompts_parse() {
        let f = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/gpu/sol/lingbot-t2v-val3.txt");
        let p = lingbot_prompts(Some(&f), None, 3).unwrap();
        assert_eq!(p.len(), 3);
        assert!(p.iter().all(|s| s.starts_with("{\"comprehensive_description\":")));
    }
}
