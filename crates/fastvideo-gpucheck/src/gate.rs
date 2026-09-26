//! `fv-gpucheck gate`: sol-engine's promotion rule over our artifacts.
//!
//! sol-engine `evals/README.md` ("Minimal Promotion Rule"): a config is
//! promoted only when `artifact`, `official_config`, `performance` and
//! `visual_artifact` pass, `off_identity` passes or is not applicable, and
//! `quantitative_quality` passes or is explicitly deferred. Thresholds come
//! from `evals/profiles/official_video_t2v.toml` (`[performance]`,
//! `[off_identity]`, `[quantitative_quality]`) and speed tiers from
//! `evals/tiers.toml`; `scripts/gpu/gate-policy.toml` carries them, with the
//! places we differ marked there.
//!
//! Inputs: the baseline and candidate cells' `benchmark.json` (a prompt set's
//! summary holds medians at the top level and each prompt under `prompts`),
//! the `compare-clips` report(s) of baseline vs candidate (one per prompt for
//! a prompt set), and optionally reports of baseline vs the candidate build
//! with the technique OFF (`--off-compare`, produced with `--off-identity`).
//!
//! Every check is recorded (keep-going); the stage fails when any required
//! gate fails. `context.verdict` holds the per-gate status and the verdict.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::report::{Report, StageResult};

fn load_json(p: &Path) -> anyhow::Result<Value> {
    let text = std::fs::read_to_string(p).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
    serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))
}

fn bench_path(p: &Path) -> PathBuf {
    if p.is_dir() {
        p.join("benchmark.json")
    } else {
        p.to_path_buf()
    }
}

pub fn load_policy(p: &Path) -> anyhow::Result<Value> {
    let text = std::fs::read_to_string(p).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
    let t: toml::Value =
        toml::from_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
    Ok(serde_json::to_value(t)?)
}

/// `(prompt name, document)` for each prompt of a benchmark.
fn prompts(b: &Value) -> Vec<(String, Value)> {
    match b.get("prompts").and_then(Value::as_object) {
        Some(m) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        None => vec![("default".into(), b.clone())],
    }
}

fn get<'a>(v: &'a Value, dotted: &str) -> &'a Value {
    dotted
        .split('.')
        .fold(v, |v, k| v.get(k).unwrap_or(&Value::Null))
}

fn num(v: &Value, dotted: &str) -> Option<f64> {
    get(v, dotted).as_f64()
}

/// The compare-clips reports for `baseline--candidate` in the matrix layout
/// (`<runs>/compare/compare-clips-<b>--<c>[-<prompt>].json`).
fn discover(baseline: &Path, candidate: &Path) -> Vec<PathBuf> {
    let cell = |p: &Path| {
        let d = if p.is_dir() {
            p.to_path_buf()
        } else {
            p.parent().map(Path::to_path_buf).unwrap_or_default()
        };
        d.file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let (b, c) = (cell(baseline), cell(candidate));
    let root = if baseline.is_dir() {
        baseline.parent()
    } else {
        baseline.parent().and_then(Path::parent)
    };
    let Some(dir) = root.map(|r| r.join("compare")) else {
        return Vec::new();
    };
    let prefix = format!("compare-clips-{b}--{c}");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name().is_some_and(|n| {
                let n = n.to_string_lossy();
                n.ends_with(".json")
                    && (n == format!("{prefix}.json") || n.starts_with(&format!("{prefix}-")))
            })
        })
        .collect();
    v.sort();
    v
}

#[derive(Default)]
struct Gates {
    status: serde_json::Map<String, Value>,
}

impl Gates {
    fn set(&mut self, gate: &str, status: &str, detail: Value) {
        self.status
            .insert(gate.into(), json!({"status": status, "detail": detail}));
    }
}

pub fn run(
    report: &mut Report,
    baseline: &Path,
    candidate: &Path,
    policy_path: &Path,
    compare: &[PathBuf],
    off_compare: &[PathBuf],
    kind: Option<&str>,
) -> StageResult<()> {
    report.set_keep_going(true);
    let policy = load_policy(policy_path)?;
    let kind = kind
        .map(str::to_string)
        .or_else(|| policy["kind"].as_str().map(str::to_string))
        .unwrap_or_else(|| "lossy".into());
    report.set(
        "policy",
        json!({"path": policy_path, "id": policy["id"], "source": policy["source"], "kind": kind}),
    );
    let mut gates = Gates::default();

    // ---- artifact
    let (bp, cp) = (bench_path(baseline), bench_path(candidate));
    let (b, c) = (load_json(&bp), load_json(&cp));
    let compare: Vec<PathBuf> = if compare.is_empty() {
        discover(baseline, candidate)
    } else {
        compare.to_vec()
    };
    let off_reports: Vec<PathBuf> = off_compare.to_vec();
    let reports: Vec<(PathBuf, anyhow::Result<Value>)> =
        compare.iter().map(|p| (p.clone(), load_json(p))).collect();
    let want_clip = policy["artifact"]["require_clip_status"]
        .as_str()
        .unwrap_or("completed");
    let mut artifact_ok = b.is_ok() && c.is_ok() && !reports.is_empty();
    report.check(
        "artifact/benchmark_json",
        b.is_ok() && c.is_ok(),
        json!({"baseline": bp, "candidate": cp, "baseline_error": b.as_ref().err().map(|e| e.to_string()), "candidate_error": c.as_ref().err().map(|e| e.to_string())}),
        json!({"required": ["benchmark.json"]}),
    )?;
    report.check(
        "artifact/compare_reports",
        !reports.is_empty(),
        json!({"reports": compare}),
        json!({"min": 1}),
    )?;
    for (p, r) in &reports {
        let name = p
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let (status, bs, cs) = match r {
            Ok(r) => (
                r["status"].as_str().unwrap_or("?").to_string(),
                get(r, "context.baseline.status")
                    .as_str()
                    .unwrap_or("?")
                    .to_string(),
                get(r, "context.candidate.status")
                    .as_str()
                    .unwrap_or("?")
                    .to_string(),
            ),
            Err(e) => (format!("unreadable: {e}"), "?".into(), "?".into()),
        };
        let ok = status == "pass" && bs == want_clip && cs == want_clip;
        artifact_ok &= ok;
        report.check(
            format!("artifact/{name}"),
            ok,
            json!({"report_status": status, "baseline_clip": bs, "candidate_clip": cs}),
            json!({"report_status": "pass", "clip_status": want_clip}),
        )?;
    }
    gates.set(
        "artifact",
        if artifact_ok { "pass" } else { "fail" },
        json!({"compare_reports": compare.len()}),
    );
    let (Ok(b), Ok(c)) = (b, c) else {
        report.set("verdict", verdict(&policy, &kind, gates, None));
        return Ok(());
    };

    // ---- official_config: the same workload, prompt by prompt
    let fields: Vec<String> = policy["official_config"]["same"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_else(|| {
            [
                "workload.height",
                "workload.width",
                "workload.num_frames",
                "workload.seed",
                "workload.prompt_sha256",
                "workload.measured_steps",
            ]
            .map(String::from)
            .to_vec()
        });
    let (pb, pc) = (prompts(&b), prompts(&c));
    let names_b: Vec<&String> = pb.iter().map(|(n, _)| n).collect();
    let names_c: Vec<&String> = pc.iter().map(|(n, _)| n).collect();
    let mut official_ok = names_b == names_c;
    report.check(
        "official_config/prompts",
        names_b == names_c,
        json!({"baseline": names_b, "candidate": names_c}),
        json!({"equal": true}),
    )?;
    let mut diffs = Vec::new();
    for ((name, db), (_, dc)) in pb.iter().zip(&pc) {
        for f in &fields {
            if get(db, f) != get(dc, f) {
                diffs.push(json!({"prompt": name, "field": f, "baseline": get(db, f), "candidate": get(dc, f)}));
            }
        }
    }
    official_ok &= diffs.is_empty();
    report.check(
        "official_config/workload",
        diffs.is_empty(),
        json!({"differences": diffs}),
        json!({"same": fields}),
    )?;
    gates.set(
        "official_config",
        if official_ok { "pass" } else { "fail" },
        json!({"fields": fields}),
    );

    // ---- performance
    let perf = &policy["performance"];
    let primary = perf["primary_metric"].as_str().unwrap_or("denoise_s");
    let secondary = perf["secondary_metric"].as_str().unwrap_or("total_s");
    let promo = perf["min_speedup_for_promotion"].as_f64().unwrap_or(1.10);
    let exper = perf["min_speedup_for_experimental"]
        .as_f64()
        .unwrap_or(1.03);
    let ratio = |k: &str| match (num(&b, k), num(&c, k)) {
        (Some(x), Some(y)) if y > 0.0 => Some(x / y),
        _ => None,
    };
    let speedup = ratio(primary);
    let speedup2 = ratio(secondary);
    let warm_needed = perf["warmup_required_for_promotion"]
        .as_bool()
        .unwrap_or(true);
    let warm = b["warm_steady_state"].as_bool() == Some(true)
        && c["warm_steady_state"].as_bool() == Some(true);
    let values = json!({
        "metric": primary, "baseline": num(&b, primary), "candidate": num(&c, primary), "speedup": speedup,
        "secondary_metric": secondary, "secondary_speedup": speedup2,
        "aggregate": b["aggregate"]["statistic"].as_str().unwrap_or("single"), "warm": warm,
    });
    report.check(
        "performance/speedup",
        speedup.is_some_and(|s| s >= promo),
        values.clone(),
        json!({"min_speedup_for_promotion": promo, "min_speedup_for_experimental": exper}),
    )?;
    if warm_needed {
        report.check(
            "performance/warm",
            warm,
            json!({"baseline": b["warm_steady_state"], "candidate": c["warm_steady_state"]}),
            json!({"warmup_required_for_promotion": true}),
        )?;
    }
    let perf_status = match speedup {
        Some(s) if s >= promo && (warm || !warm_needed) => "pass",
        Some(s) if s >= exper => "experimental",
        _ => "fail",
    };
    gates.set("performance", perf_status, values);

    // ---- off_identity
    let needs_off = policy["off_identity"]["required_for_kinds"]
        .as_array()
        .is_some_and(|a| a.iter().any(|k| k.as_str() == Some(kind.as_str())));
    if off_reports.is_empty() {
        report.check(
            "off_identity/report",
            !needs_off,
            json!({"kind": kind, "off_reports": 0}),
            json!({"required_for_kinds": policy["off_identity"]["required_for_kinds"]}),
        )?;
        gates.set(
            "off_identity",
            if needs_off { "fail" } else { "not_applicable" },
            json!({"reason": "no --off-compare report"}),
        );
    } else {
        let mut all = true;
        for p in &off_reports {
            let r = load_json(p);
            let st = r
                .as_ref()
                .map(|r| {
                    get(r, "context.off_identity.status")
                        .as_str()
                        .unwrap_or("?")
                        .to_string()
                })
                .unwrap_or_else(|e| e.to_string());
            let ok = st == "ok";
            all &= ok;
            report.check(
                format!("off_identity/{}", p.file_stem().unwrap_or_default().to_string_lossy()),
                ok,
                json!({"status": st, "detail": r.as_ref().ok().map(|r| get(r, "context.off_identity").clone())}),
                json!({"max_abs_diff_uint8": 0}),
            )?;
        }
        gates.set(
            "off_identity",
            if all { "pass" } else { "fail" },
            json!({"reports": off_reports}),
        );
    }

    // ---- quantitative_quality (per compare report, i.e. per prompt)
    let q = &policy["quantitative_quality"];
    let hard: Vec<String> = q["hard"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let is_hard = |k: &str| hard.iter().any(|h| h == k);
    let mut quality_ok = true;
    let mut rows = Vec::new();
    let mut lpips_means = Vec::new();
    let mut lpips_maxes = Vec::new();
    let mut psnrs = Vec::new();
    if q["enabled"].as_bool().unwrap_or(true) {
        for (p, r) in &reports {
            let Ok(r) = r else { continue };
            let tag = p
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let ctx = &r["context"];
            let pm = &ctx["pixel_metrics"];
            let mut checks: Vec<(&str, bool, Value, Value)> = Vec::new();
            if q["frame_count_equal"].as_bool().unwrap_or(true) {
                let (fb, fc) = (&ctx["baseline"]["frames"], &ctx["candidate"]["frames"]);
                checks.push((
                    "frame_count_equal",
                    fb == fc && fb.as_u64().is_some_and(|n| n > 0),
                    json!({"baseline": fb, "candidate": fc}),
                    json!({"equal": true}),
                ));
            }
            let range = |key: &str, metric: &str| -> Option<(bool, Value, Value)> {
                let (lo, hi) = (
                    q[format!("{key}_min")].as_f64(),
                    q[format!("{key}_max")].as_f64(),
                );
                if lo.is_none() && hi.is_none() {
                    return None;
                }
                let v = pm[metric].as_f64();
                let ok = v.is_some_and(|v| lo.is_none_or(|l| v >= l) && hi.is_none_or(|h| v <= h));
                Some((ok, json!({metric: v}), json!({"min": lo, "max": hi})))
            };
            if let Some((ok, v, l)) = range("sharpness_ratio", "sharpness_ratio_mean") {
                checks.push(("sharpness_ratio", ok, v, l));
            }
            if let Some((ok, v, l)) = range("temporal_jitter_ratio", "temporal_jitter_ratio_mean") {
                checks.push(("temporal_jitter_ratio", ok, v, l));
            }
            let psnr = pm["psnr_mean"].as_f64();
            if let Some(v) = psnr {
                psnrs.push(v);
            }
            if let Some(min) = q["psnr_mean_min_db"].as_f64() {
                // Identical clips have no finite PSNR (null): that passes.
                let ok = psnr.is_none_or(|v| v >= min) && pm["status"] == "ok";
                checks.push((
                    "psnr",
                    ok,
                    json!({"psnr_mean": psnr, "psnr_min": pm["psnr_min"]}),
                    json!({"psnr_mean_min_db": min}),
                ));
            }
            let lp = &ctx["lpips"];
            let lp_ok = lp["status"] == "complete";
            let (lm, lx) = (ctx["lpips_mean"].as_f64(), ctx["lpips_max"].as_f64());
            if let Some(v) = lm {
                lpips_means.push(v);
            }
            if let Some(v) = lx {
                lpips_maxes.push(v);
            }
            if q["require_lpips"].as_bool().unwrap_or(false) {
                checks.push((
                    "lpips_available",
                    lp_ok,
                    json!({"status": lp["status"], "reason": lp["reason"]}),
                    json!({"status": "complete"}),
                ));
            }
            if let Some(max) = q["lpips_mean_max"].as_f64() {
                checks.push((
                    "lpips_mean",
                    lp_ok && lm.is_some_and(|v| v <= max),
                    json!({"lpips_mean": lm}),
                    json!({"lpips_mean_max": max}),
                ));
            }
            if let Some(max) = q["lpips_max_max"].as_f64() {
                checks.push((
                    "lpips_max",
                    lp_ok && lx.is_some_and(|v| v <= max),
                    json!({"lpips_max": lx}),
                    json!({"lpips_max_max": max}),
                ));
            }
            for (name, ok, values, limits) in checks {
                let hard_check = is_hard(name);
                if hard_check {
                    quality_ok &= ok;
                    report.check(
                        format!("quality/{tag}/{name}"),
                        ok,
                        values.clone(),
                        limits.clone(),
                    )?;
                } else {
                    report.note(
                        format!("quality/{tag}/{name}"),
                        json!({"ok": ok, "values": values, "limits": limits, "telemetry": true}),
                    );
                }
                rows.push(json!({"report": tag, "check": name, "ok": ok, "hard": hard_check, "values": values, "limits": limits}));
            }
        }
    }
    let median = |v: &mut Vec<f64>| -> Option<f64> {
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        let n = v.len();
        Some(if n % 2 == 1 {
            v[n / 2]
        } else {
            0.5 * (v[n / 2 - 1] + v[n / 2])
        })
    };
    let quality_summary = json!({
        "reports": reports.len(),
        "psnr_mean_median": median(&mut psnrs),
        "lpips_mean_median": median(&mut lpips_means.clone()),
        "lpips_mean_max": lpips_means.iter().copied().fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v)))),
        "lpips_max_max": lpips_maxes.iter().copied().fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v)))),
        "checks": rows,
    });
    let quality_status = if !q["enabled"].as_bool().unwrap_or(true) {
        "explicitly_deferred"
    } else if quality_ok {
        "pass"
    } else {
        "fail"
    };
    gates.set("quantitative_quality", quality_status, quality_summary);

    // ---- visual_artifact
    let va = &policy["visual_artifact"];
    let va_required = va["required"].as_bool().unwrap_or(true);
    report.check(
        "visual_artifact/judge",
        !va_required,
        json!({"judge": "none in this runtime", "reason": va["reason"]}),
        json!({"required": va_required}),
    )?;
    gates.set(
        "visual_artifact",
        if va_required { "fail" } else { "deferred" },
        json!({"reason": va["reason"]}),
    );

    let v = verdict(&policy, &kind, gates, speedup);
    eprintln!("gate verdict: {}", v["verdict"]);
    report.set("verdict", v);
    Ok(())
}

fn tier(policy: &Value, speedup: Option<f64>) -> Value {
    let t = &policy["tiers"];
    let s = match speedup {
        Some(s) => s,
        None => return Value::Null,
    };
    let mut best = Value::Null;
    for (name, key) in [
        ("low", "low_speedup"),
        ("medium", "medium_speedup"),
        ("high", "high_speedup"),
    ] {
        if t[key].as_f64().is_some_and(|min| s >= min) {
            best = json!(name);
        }
    }
    best
}

fn verdict(policy: &Value, kind: &str, gates: Gates, speedup: Option<f64>) -> Value {
    let st = |g: &str| {
        gates
            .status
            .get(g)
            .and_then(|v| v["status"].as_str())
            .unwrap_or("missing")
            .to_string()
    };
    // evals/README.md Minimal Promotion Rule, with visual_artifact deferrable
    // only when the policy says so (no vision judge here).
    let va_required = policy["visual_artifact"]["required"]
        .as_bool()
        .unwrap_or(true);
    let blockers: Vec<String> = [
        ("artifact", vec!["pass"]),
        ("official_config", vec!["pass"]),
        ("performance", vec!["pass"]),
        ("off_identity", vec!["pass", "not_applicable"]),
        ("quantitative_quality", vec!["pass", "explicitly_deferred"]),
        (
            "visual_artifact",
            if va_required {
                vec!["pass"]
            } else {
                vec!["pass", "deferred"]
            },
        ),
    ]
    .into_iter()
    .filter(|(g, ok)| !ok.contains(&st(g).as_str()))
    .map(|(g, _)| format!("{g}:{}", st(g)))
    .collect();
    json!({
        "verdict": if blockers.is_empty() { "pass" } else { "fail" },
        "kind": kind,
        "promotion_blockers": blockers,
        "speedup": speedup,
        "tier": tier(policy, speedup),
        "gates": gates.status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> Value {
        load_policy(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/gpu/gate-policy.toml"),
        )
        .unwrap()
    }

    #[test]
    fn shipped_policy_mirrors_sol_engine_numbers() {
        let p = policy();
        assert_eq!(p["performance"]["primary_metric"], "denoise_s");
        assert_eq!(p["performance"]["min_speedup_for_promotion"], 1.10);
        assert_eq!(p["performance"]["min_speedup_for_experimental"], 1.03);
        assert_eq!(p["quantitative_quality"]["sharpness_ratio_min"], 0.95);
        assert_eq!(p["quantitative_quality"]["temporal_jitter_ratio_max"], 1.20);
        assert_eq!(p["tiers"]["high_speedup"], 3.0);
    }

    fn cell(dir: &Path, name: &str, denoise: f64, seed: u64) -> PathBuf {
        let d = dir.join(name);
        std::fs::create_dir_all(&d).unwrap();
        let b = json!({"denoise_s": denoise, "total_s": denoise + 5.0, "warm_steady_state": true,
            "workload": {"height": 480, "width": 832, "num_frames": 121, "seed": seed, "prompt_sha256": "x", "measured_steps": 8}});
        std::fs::write(d.join("benchmark.json"), b.to_string()).unwrap();
        d
    }

    fn compare(
        dir: &Path,
        b: &str,
        c: &str,
        sharp: f64,
        lpips: Option<f64>,
        identical: bool,
    ) -> PathBuf {
        let d = dir.join("compare");
        std::fs::create_dir_all(&d).unwrap();
        let r = json!({"status": "pass", "context": {
            "baseline": {"status": "completed", "frames": 121},
            "candidate": {"status": "completed", "frames": 121},
            "pixel_metrics": {"status": "ok", "psnr_mean": 30.0, "sharpness_ratio_mean": sharp, "temporal_jitter_ratio_mean": 1.0},
            "off_identity": {"status": if identical { "ok" } else { "different" }},
            "lpips": if lpips.is_some() { json!({"status": "complete"}) } else { json!({"status": "deferred"}) },
            "lpips_mean": lpips, "lpips_max": lpips.map(|v| v * 2.0),
        }});
        let p = d.join(format!("compare-clips-{b}--{c}.json"));
        std::fs::write(&p, r.to_string()).unwrap();
        p
    }

    fn run_gate(b: &Path, c: &Path, off: Vec<PathBuf>, kind: &str) -> Value {
        let out = b.parent().unwrap().join("gate-out");
        let mut report = Report::new(&out, "gate");
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/gpu/gate-policy.toml");
        run(&mut report, b, c, &p, &[], &off, Some(kind)).unwrap();
        let _ = report.finish(&Ok(()));
        let doc: Value =
            serde_json::from_str(&std::fs::read_to_string(out.join("gate.json")).unwrap()).unwrap();
        doc["context"]["verdict"].clone()
    }

    fn scratch(n: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fv-gate-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn faster_similar_candidate_passes_lossy() {
        let d = scratch("pass");
        let (b, c) = (cell(&d, "base", 20.0, 1), cell(&d, "cand", 10.0, 1));
        compare(&d, "base", "cand", 1.0, Some(0.05), false);
        let v = run_gate(&b, &c, vec![], "lossy");
        assert_eq!(v["verdict"], "pass", "{v}");
        assert_eq!(v["tier"], "medium");
        assert_eq!(v["gates"]["off_identity"]["status"], "not_applicable");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn slow_blurry_or_mismatched_candidates_fail() {
        let d = scratch("fail");
        let (b, c) = (cell(&d, "base", 20.0, 1), cell(&d, "slow", 19.5, 1));
        compare(&d, "base", "slow", 1.0, Some(0.05), false);
        let v = run_gate(&b, &c, vec![], "lossy");
        assert_eq!(v["verdict"], "fail");
        assert_eq!(v["gates"]["performance"]["status"], "fail");

        let c = cell(&d, "blurry", 10.0, 1);
        compare(&d, "base", "blurry", 0.80, Some(0.05), false);
        let v = run_gate(&b, &c, vec![], "lossy");
        assert_eq!(v["gates"]["quantitative_quality"]["status"], "fail", "{v}");

        let c = cell(&d, "otherseed", 10.0, 2);
        compare(&d, "base", "otherseed", 1.0, Some(0.05), false);
        let v = run_gate(&b, &c, vec![], "lossy");
        assert_eq!(v["gates"]["official_config"]["status"], "fail");

        let c = cell(&d, "nolpips", 10.0, 1);
        compare(&d, "base", "nolpips", 1.0, None, false);
        let v = run_gate(&b, &c, vec![], "lossy");
        assert_eq!(
            v["gates"]["quantitative_quality"]["status"], "fail",
            "lpips is required"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn exact_kind_needs_an_identical_off_arm() {
        let d = scratch("exact");
        let (b, c) = (cell(&d, "base", 20.0, 1), cell(&d, "cand", 10.0, 1));
        compare(&d, "base", "cand", 1.0, Some(0.01), false);
        let v = run_gate(&b, &c, vec![], "exact");
        assert_eq!(v["gates"]["off_identity"]["status"], "fail");
        let off_ok = compare(&d, "base", "off", 1.0, Some(0.0), true);
        let v = run_gate(&b, &c, vec![off_ok], "exact");
        assert_eq!(v["verdict"], "pass", "{v}");
        let off_bad = compare(&d, "base", "offbad", 1.0, Some(0.0), false);
        let v = run_gate(&b, &c, vec![off_bad], "exact");
        assert_eq!(v["gates"]["off_identity"]["status"], "fail");
        let _ = std::fs::remove_dir_all(&d);
    }
}
