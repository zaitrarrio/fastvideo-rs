//! `fv-lab report table`: tables over run directories (docs/adr/0019-fv-lab.md §13).
//!
//! - [`sol_bench`] (`--kind sol-bench`): `scripts/gpu/sol_bench_table.py`,
//!   our sol-bench cells next to sol-engine's published numbers, and its
//!   `results.json`;
//! - [`timings`] (`--kind timings`): `scripts/gpu/timings.sh`, a
//!   `validate.sh` run's stage reports (`<run>/remote/*.json`) as
//!   `timings.md`, with jq's semantics (`//` takes the right side when the
//!   left is `null`, `false` or an error; a failing expression drops its row)
//!   and jq's canonical number spelling.

use std::path::{Path, PathBuf};

use crate::pyfmt::{self, dumps_indent, fixed, jq_num, jq_text, py_str, Py};

type R<T> = Result<T, String>;

/// One `PUBLISHED` row: cell, model, config, their hardware, their seconds, note.
type Published = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    Option<f64>,
    &'static str,
);

/// `PUBLISHED`: cell -> (model, config, their hardware, their seconds, note).
const PUBLISHED: &[Published] = &[
    (
        "h3-768p-dense",
        "MiniMax-H3",
        "768p 124 f 50 st, dense (MXFP8 linears: our sm_100+ default)",
        "RTX 5090",
        Some(1045.4),
        "rtx5090_dense.toml is BF16",
    ),
    (
        "h3-768p-dense-bf16",
        "MiniMax-H3",
        "768p 124 f 50 st, dense, BF16",
        "RTX 5090",
        Some(1045.4),
        "",
    ),
    (
        "h3-768p-fullopt",
        "MiniMax-H3",
        "768p 124 f 50 st, fullopt (Sol + TeaCache) + MXFP8 linears",
        "RTX 5090",
        Some(231.2),
        "rtx5090_fullopt.toml is BF16",
    ),
    (
        "h3-768p-fullopt-bf16",
        "MiniMax-H3",
        "768p 124 f 50 st, fullopt (Sol + TeaCache), BF16",
        "RTX 5090",
        Some(231.2),
        "",
    ),
    (
        "ltx25-4k5s-sol-bf16",
        "LTX-2.5 distilled",
        "4K 5 s, Sol stage 2, BF16",
        "RTX 5090",
        Some(273.63),
        "",
    ),
    (
        "ltx25-1080p20s-sol-bf16",
        "LTX-2.5 distilled",
        "1080p 20 s, Sol stage 2, BF16",
        "RTX 5090",
        Some(261.65),
        "",
    ),
    (
        "ltx25-4k5s-sol-nvfp4",
        "LTX-2.5 distilled",
        "4K 5 s, Sol stage 2, NVFP4 video FFN",
        "RTX 5090",
        Some(171.82),
        "",
    ),
    (
        "ltx25-1080p20s-sol-nvfp4",
        "LTX-2.5 distilled",
        "1080p 20 s, Sol stage 2, NVFP4 video FFN",
        "RTX 5090",
        Some(164.40),
        "",
    ),
    (
        "wan5b-base",
        "Wan2.2 TI2V-5B",
        "704x1280x121, 50 st, CFG 5, base",
        "1x GB200",
        Some(70.25),
        "theirs: 5-prompt median",
    ),
    (
        "wan5b-easycache",
        "Wan2.2 TI2V-5B",
        "same, EasyCache 0.036",
        "1x GB200",
        Some(24.35),
        "theirs: fusion + EasyCache fullopt",
    ),
    (
        "wan5b-opt",
        "Wan2.2 TI2V-5B",
        "same, EasyCache 0.036 + PISA",
        "1x GB200",
        Some(28.69),
        "theirs: golden kernel+EasyCache+PISA run",
    ),
    (
        "wan14-720p-base-s15",
        "Wan2.1 T2V-14B",
        "1280x720x81, 50 st, CFG 5, base (15 st measured, x50/15)",
        "-",
        None,
        "absolute number withdrawn upstream",
    ),
    (
        "wan14-720p-fullstack",
        "Wan2.1 T2V-14B",
        "1280x720x81, 50 st, EasyCache + Sol-Attn",
        "-",
        None,
        "absolute number withdrawn upstream",
    ),
    (
        "sana-baseline",
        "SANA-Video 2B",
        "832x480x81, 50 st, cfg 6, baseline",
        "1x GB200",
        None,
        "theirs: ratio only (2.77x)",
    ),
    (
        "sana-full",
        "SANA-Video 2B",
        "same, EasyCache 0.1 + QKV merge + bf16 linear attn",
        "1x GB200",
        None,
        "theirs: ratio only (2.77x)",
    ),
    (
        "wan13-sol-base",
        "Wan2.1 T2V-1.3B",
        "832x480x81, 50 st, CFG 6, base (median of 5 prompts)",
        "-",
        None,
        "no published number",
    ),
    (
        "wan13-sol-fullstack",
        "Wan2.1 T2V-1.3B",
        "same, EasyCache 0.036 + Sol-Attn",
        "-",
        None,
        "no published number",
    ),
    (
        "a14b-sol-base",
        "Wan2.2 T2V-A14B",
        "1280x720x81, 40 st, CFG 4/3, base (expert swap)",
        "1x GB200",
        Some(449.67),
        "",
    ),
    (
        "a14b-sol-fullopt",
        "Wan2.2 T2V-A14B",
        "same, EasyCache + PISA",
        "1x GB200",
        Some(207.01),
        "theirs also: kernel fusion",
    ),
    (
        "ltx23-hq-base",
        "LTX-2.3 HQ",
        "1920x1088x241, res2s 15 + 3, dense stage 2",
        "1x GB200",
        None,
        "theirs: ratio only (2.40x)",
    ),
    (
        "ltx23-hq-fullopt",
        "LTX-2.3 HQ",
        "same, SCSP + PISA s2 + midpoint prune + NVFP4 FFN",
        "1x GB200",
        None,
        "theirs: ratio only (2.40x)",
    ),
    (
        "lingbot-baseline",
        "LingBot-Video MoE",
        "480p 121 f 40 st + 1080p refiner 8 st, 1 prompt",
        "4x GB200",
        Some(375.53),
        "theirs: 4 GPUs (CP4)",
    ),
    (
        "lingbot-baseline-rs3",
        "LingBot-Video MoE",
        "480p 121 f 40 st + 1080p refiner 8 st, 1 prompt (refiner: 3 of 8 st measured, x8/3)",
        "4x GB200",
        Some(375.53),
        "theirs: 4 GPUs (CP4)",
    ),
    (
        "lingbot-fullopt",
        "LingBot-Video MoE",
        "same, EasyCache + refiner PISA",
        "4x GB200",
        Some(144.36),
        "theirs: 4 GPUs (CP4)",
    ),
    (
        "cosmos3-baseline",
        "Cosmos3-Super 64B",
        "1280x720x189, 35 st, CFG 6",
        "4x GB200",
        Some(130.41),
        "theirs: 4 GPUs (SP)",
    ),
    (
        "cosmos3-teacache",
        "Cosmos3-Super 64B",
        "same, TeaCache 1.15/10/3 (BF16)",
        "4x GB200",
        None,
        "theirs: 2.26x incl. NVFP4",
    ),
    (
        "cosmos3-teacache-fp8",
        "Cosmos3-Super 64B",
        "same, TeaCache + W8A8 FP8",
        "4x GB200",
        None,
        "theirs: 2.26x incl. NVFP4",
    ),
    (
        "cosmos3-baseline-fp8",
        "Cosmos3-Super 64B",
        "same, no cache, W8A8 FP8",
        "4x GB200",
        Some(130.41),
        "theirs: BF16 on 4 GPUs (SP)",
    ),
];

/// `PAIRS`: (baseline cell, optimized cell, their ratio).
fn pairs() -> [(&'static str, &'static str, Option<f64>); 12] {
    [
        (
            "h3-768p-dense-bf16",
            "h3-768p-fullopt-bf16",
            Some(1045.4 / 231.2),
        ),
        ("wan5b-base", "wan5b-easycache", Some(70.25 / 24.35)),
        ("wan14-720p-base-s15", "wan14-720p-fullstack", None),
        ("sana-baseline", "sana-full", Some(2.77)),
        ("wan13-sol-base", "wan13-sol-fullstack", None),
        ("a14b-sol-base", "a14b-sol-fullopt", Some(449.67 / 207.01)),
        ("ltx23-hq-base", "ltx23-hq-fullopt", Some(2.40)),
        ("lingbot-baseline", "lingbot-fullopt", Some(375.53 / 144.36)),
        (
            "lingbot-baseline-rs3",
            "lingbot-fullopt",
            Some(375.53 / 144.36),
        ),
        ("cosmos3-baseline", "cosmos3-teacache", Some(2.26)),
        ("cosmos3-baseline", "cosmos3-teacache-fp8", Some(2.26)),
        ("cosmos3-baseline-fp8", "cosmos3-teacache-fp8", None),
    ]
}

/// sol-engine's RTX 5090 Sol stage-2 seconds (the DiT-only part of the E2E).
const STAGE2: [(&str, f64); 4] = [
    ("ltx25-4k5s-sol-bf16", 130.32),
    ("ltx25-1080p20s-sol-bf16", 122.27),
    ("ltx25-4k5s-sol-nvfp4", 72.82),
    ("ltx25-1080p20s-sol-nvfp4", 66.78),
];

fn get<'a>(v: &'a Py, k: &str) -> R<&'a Py> {
    match v {
        Py::Dict(_) => Ok(v.at(k)),
        other => Err(format!("{} has no .get (`{k}`)", pyfmt::repr(other))),
    }
}

fn get_or<'a>(v: &'a Py, k: &str, d: &'a Py) -> R<&'a Py> {
    match v {
        Py::Dict(_) => Ok(v.get(k).unwrap_or(d)),
        other => Err(format!("{} has no .get (`{k}`)", pyfmt::repr(other))),
    }
}

fn arith(v: Option<Py>, what: &str) -> R<Py> {
    v.ok_or_else(|| format!("{what}: not numbers"))
}

/// The directories of `d`, sorted.
fn subdirs(d: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(d)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect()
        })
        .unwrap_or_default();
    v.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    v
}

fn name_of(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// `sol_timings(cell)`: the `timings` check of a `fv-gpucheck sol` stage
/// report (LingBot / Cosmos3).
fn sol_timings(cell: &Path) -> Option<Py> {
    let dir = cell.join("gpucheck-out");
    if !dir.is_dir() {
        return None;
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json") && !name_of(p).starts_with('.'))
        .collect();
    files.sort();
    for f in files {
        let Ok(j) = pyfmt::load(&f) else { continue };
        let checks = j.at("checks").or(Py::List(Vec::new()));
        for c in checks.as_list() {
            if c.at("name").as_str() == Some("timings") {
                return Some(c.at("values").or(Py::dict()));
            }
        }
    }
    None
}

fn kw(row: &mut Py, items: Vec<(&str, Py)>) {
    for (k, v) in items {
        row.set(k, v);
    }
}

/// `load(run_dirs)`: one row per cell directory (a later run's cell of the
/// same name replaces the earlier one in its place).
fn load_rows(dirs: &[PathBuf]) -> R<Py> {
    let mut rows = Py::dict();
    for d in dirs {
        if !d.is_dir() {
            continue;
        }
        for c in subdirs(d) {
            let cname = name_of(&c);
            let mut b = c.join("benchmark.json");
            if !b.exists() && c.join("frames").join("benchmark.json").exists() {
                b = c.join("frames").join("benchmark.json");
            }
            let s = c.join("summary.json");
            let mut row = Py::Dict(vec![
                ("cell".into(), Py::Str(cname.clone())),
                ("run".into(), Py::Str(name_of(d))),
            ]);
            if s.exists() {
                row.set("summary", pyfmt::load(&s)?);
            }
            if b.exists() {
                let j = pyfmt::load(&b)?;
                let empty = Py::dict();
                let st = get_or(&j, "stage_seconds", &empty)?;
                let zero = Py::int(0);
                let s1 = get(st, "stage_1")?.or(zero.clone());
                let s2 = get(st, "stage_2")?.or(zero.clone());
                let den_default = arith(s1.add(&s2), "stage_1 + stage_2")?.or(Py::None);
                kw(
                    &mut row,
                    vec![
                        (
                            "total_s",
                            get(&j, "total_s")?.or(get(&j, "e2e_seconds")?.clone()),
                        ),
                        ("load_s", get(&j, "load_s")?.clone()),
                        ("text_s", get_or(&j, "text_s", get(st, "text")?)?.clone()),
                        ("denoise_s", get_or(&j, "denoise_s", &den_default)?.clone()),
                        ("stage_1_s", get(st, "stage_1")?.clone()),
                        ("stage_2_s", get(st, "stage_2")?.clone()),
                        (
                            "decode_s",
                            get_or(&j, "decode_s", get(st, "video_decode")?)?.clone(),
                        ),
                        ("warm", get(&j, "warm_steady_state")?.clone()),
                        ("text_cache", get(&j, "text_cache")?.clone()),
                        (
                            "peak_mib",
                            get(&j, "max_device_memory_used_mib")?
                                .or(get(&j, "peak_memory_mb")?.clone()),
                        ),
                        ("gpu", get(&j, "gpu")?.or(get(&j, "hardware")?.clone())),
                        ("env", get(&j, "env")?.clone()),
                    ],
                );
                if cname.ends_with("-s15") && row.at("denoise_s").truthy() {
                    let total = row.at("total_s").clone();
                    let den = row.at("denoise_s").clone();
                    let scaled = arith(
                        arith(den.mul(&Py::int(50)), "denoise * 50")?.div(&Py::int(15)),
                        "/ 15",
                    )?;
                    row.set("measured_total_s", total.clone());
                    let t = arith(arith(total.sub(&den), "total - denoise")?.add(&scaled), "+")?;
                    row.set("total_s", t);
                    row.set("denoise_s", scaled);
                    row.set("extrapolated", Py::Str("denoise x50/15".into()));
                }
            }
            let t = if b.exists() { None } else { sol_timings(&c) };
            if let Some(t) = t.filter(Py::truthy) {
                if t.get("request_s_median").is_some() {
                    let first = Py::List(vec![Py::dict()]);
                    let runs = get(&t, "runs")?.or(first);
                    let r0 = runs.as_list().first().cloned().ok_or("runs: empty")?;
                    let zero = Py::int(0);
                    let sum = |keys: &[&str]| -> R<Py> {
                        let mut acc = Py::int(0);
                        for k in keys {
                            acc = arith(acc.add(&get(&r0, k)?.or(zero.clone())), k)?;
                        }
                        Ok(acc)
                    };
                    let mut detail = Py::dict();
                    for k in [
                        "base_denoise_s",
                        "refiner_denoise_s",
                        "refiner_prepare_s",
                        "base_steps_reused",
                        "refiner_steps_computed",
                        "refiner_sparse_steps",
                    ] {
                        detail.set(k, get(&r0, k)?.clone());
                    }
                    kw(
                        &mut row,
                        vec![
                            ("total_s", t.at("request_s_median").clone()),
                            ("text_s", get(&r0, "text_encode_s")?.clone()),
                            ("denoise_s", sum(&["base_denoise_s", "refiner_denoise_s"])?),
                            ("decode_s", sum(&["base_decode_s", "refiner_decode_s"])?),
                            (
                                "load_s",
                                sum(&["load_text_s", "load_base_s", "load_refiner_s"])?,
                            ),
                            ("peak_mib", get(&t, "peak_mib")?.clone()),
                            ("gpu", Py::Str("RTX PRO 6000".into())),
                            ("warm", Py::Bool(false)),
                            ("detail", detail),
                        ],
                    );
                    let rsteps = get(&r0, "refiner_steps")?.or(zero.clone());
                    let rd = get(&r0, "refiner_denoise_s")?.clone();
                    if cname.ends_with("-rs3") && rsteps.truthy() && rd.truthy() {
                        let scaled = arith(
                            arith(rd.mul(&Py::int(8)), "rd * 8")?.div(&rsteps),
                            "/ refiner_steps",
                        )?;
                        let total = row.at("total_s").clone();
                        let den = row.at("denoise_s").clone();
                        row.set("measured_total_s", total.clone());
                        row.set(
                            "total_s",
                            arith(arith(total.sub(&rd), "total - rd")?.add(&scaled), "+")?,
                        );
                        row.set(
                            "denoise_s",
                            arith(arith(den.sub(&rd), "denoise - rd")?.add(&scaled), "+")?,
                        );
                        row.set(
                            "extrapolated",
                            Py::Str(format!("refiner denoise x8/{}", py_str(&rsteps))),
                        );
                    }
                } else {
                    let mut detail = Py::dict();
                    for k in ["steps", "steps_computed", "steps_reused", "fp8"] {
                        detail.set(k, get(&t, k)?.clone());
                    }
                    kw(
                        &mut row,
                        vec![
                            ("total_s", get(&t, "request_s")?.clone()),
                            ("text_s", get(&t, "text_tower_s")?.clone()),
                            ("denoise_s", get(&t, "denoise_s")?.clone()),
                            ("decode_s", get(&t, "decode_s")?.clone()),
                            ("load_s", get(&t, "load_s")?.clone()),
                            ("peak_mib", get(&t, "peak_mib")?.clone()),
                            ("gpu", Py::Str("RTX PRO 6000".into())),
                            ("warm", Py::Bool(true)),
                            ("detail", detail),
                        ],
                    );
                }
            }
            rows.set(&cname, row);
        }
    }
    Ok(rows)
}

/// `fmt(x)`: `—` for `None`, else two decimals.
fn fmt(v: &Py) -> R<String> {
    if v.is_none() {
        return Ok("—".into());
    }
    pyfmt::fixed_py(v, 2).ok_or_else(|| format!("cannot format {}", pyfmt::repr(v)))
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map_or("—".into(), |x| fixed(x, 2))
}

fn num(v: &Py) -> R<f64> {
    v.num()
        .ok_or_else(|| format!("{} is not a number", pyfmt::repr(v)))
}

/// The table and `results.json`.
pub struct SolBench {
    pub table: String,
    pub results_json: String,
    /// Where the script writes `results.json`: the first run directory's parent.
    pub results_path: PathBuf,
}

/// `sol_bench_table.py <run dir>...`.
pub fn sol_bench(args: &[PathBuf]) -> R<SolBench> {
    let rows = load_rows(args)?;
    let dirs: Vec<&PathBuf> = args.iter().filter(|a| a.is_dir()).collect();
    let mut out = vec![
        "| Model | Config | Their HW | Theirs (s) | Our HW | Ours (s) | Ours / theirs | Ours: load / text / denoise / decode (s) | Notes |".to_owned(),
        "|---|---|---|---:|---|---:|---:|---|---|".to_owned(),
    ];
    let empty = Py::dict();
    for (cell, model, cfg, thw, theirs, note) in PUBLISHED {
        let Some(r) = rows.get(cell) else { continue };
        let ours = r.at("total_s");
        let mut notes: Vec<String> = Vec::new();
        if !note.is_empty() {
            notes.push((*note).to_owned());
        }
        let ex = r.at("extrapolated");
        if ex.truthy() {
            notes.push(format!("extrapolated: {}", py_str(ex)));
        }
        if ours.is_none() {
            let summary = get_or(r, "summary", &empty)?;
            let skipped = get(summary, "skipped")?;
            let why = if skipped.truthy() {
                py_str(skipped)
            } else {
                format!("exit {}", py_str(get(summary, "exit")?))
            };
            notes.push(format!("not run: {why}"));
        }
        if *r.at("warm") == Py::Bool(false) {
            notes.push("first request after load".into());
        }
        if r.at("text_cache").as_str() == Some("hit") {
            notes.push("text: cache hit".into());
        }
        let ratio = match theirs {
            Some(t) if ours.truthy() && *t != 0.0 => format!("{}x", fixed(num(ours)? / t, 2)),
            _ => "—".into(),
        };
        let detail = ["load_s", "text_s", "denoise_s", "decode_s"]
            .iter()
            .map(|k| fmt(r.at(k)))
            .collect::<R<Vec<_>>>()?
            .join(" / ");
        let hw = if ours.is_none() {
            "—"
        } else {
            "RTX PRO 6000 (sm_120)"
        };
        out.push(format!(
            "| {model} | {cfg} | {thw} | {} | {hw} | {} | {ratio} | {detail} | {} |",
            fmt_opt(*theirs),
            fmt(ours)?,
            notes.join("; ")
        ));
    }
    let mut lines = Vec::new();
    for (cell, theirs) in STAGE2 {
        let r = rows.get(cell).unwrap_or(&empty);
        let s2 = r.at("stage_2_s");
        if s2.truthy() {
            let v = num(s2)?;
            lines.push(format!(
                "| {cell} | {} | {} | {}x | {} |",
                fixed(theirs, 2),
                fixed(v, 2),
                fixed(v / theirs, 2),
                fmt(r.at("stage_1_s"))?
            ));
        }
    }
    if !lines.is_empty() {
        out.extend(
            [
                "",
                "LTX-2.5 Sol stage 2 only (RTX 5090 published vs ours on RTX PRO 6000):",
                "",
                "| Cell | Theirs stage 2 (s) | Ours stage 2 (s) | Ours / theirs | Ours stage 1 (s) |",
                "|---|---:|---:|---:|---:|",
            ]
            .map(str::to_owned),
        );
        out.extend(lines);
    }
    let mut lines = Vec::new();
    for (base, opt, theirs) in pairs() {
        let b = rows.get(base).unwrap_or(&empty).at("total_s");
        let o = rows.get(opt).unwrap_or(&empty).at("total_s");
        if b.truthy() && o.truthy() {
            let theirs = match theirs {
                Some(t) if t != 0.0 => format!("{}x", fixed(t, 2)),
                _ => "—".into(),
            };
            lines.push(format!(
                "| {base} → {opt} | {}x | {theirs} |",
                fixed(num(b)? / num(o)?, 2)
            ));
        }
    }
    if !lines.is_empty() {
        out.extend(
            [
                "",
                "Optimized-arm speedup over our own baseline (same GPU) vs theirs:",
                "",
                "| Pair | Ours | Theirs |",
                "|---|---:|---:|",
            ]
            .map(str::to_owned),
        );
        out.extend(lines);
    }
    let first = dirs.first().ok_or("no run directory among the arguments")?;
    let parent = first
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let parent = if parent.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        parent
    };
    Ok(SolBench {
        table: format!("{}\n", out.join("\n")),
        results_json: format!("{}\n", dumps_indent(&rows, 1)),
        results_path: parent.join("results.json"),
    })
}

// ------------------------------------------------------------------ timings

/// A jq error (the expression stops; `2>/dev/null || true` drops its row).
#[derive(Debug)]
struct JqErr;

type J<T> = Result<T, JqErr>;

/// `.k` in jq: `null` on `null`, the value or `null` on an object, else an error.
fn jget<'a>(v: &'a Py, k: &str) -> J<&'a Py> {
    match v {
        Py::None => Ok(&Py::None),
        Py::Dict(_) => Ok(v.at(k)),
        _ => Err(JqErr),
    }
}

fn jpath<'a>(v: &'a Py, path: &[&str]) -> J<&'a Py> {
    let mut cur = v;
    for k in path {
        cur = jget(cur, k)?;
    }
    Ok(cur)
}

/// jq truthiness: everything but `null` and `false`.
fn jtrue(v: &Py) -> bool {
    !matches!(v, Py::None | Py::Bool(false))
}

/// `a // b`: `a` unless it is `null`, `false` or an error.
fn alt<'a>(a: J<&'a Py>, b: &'a Py) -> &'a Py {
    match a {
        Ok(v) if jtrue(v) => v,
        _ => b,
    }
}

/// `tostring | .[0:n]`.
fn tostring_head(v: &Py, n: usize) -> String {
    jq_text(v).chars().take(n).collect()
}

/// `x * m | round / m`.
fn round_to(v: &Py, m: f64) -> J<String> {
    match v {
        Py::Int(_) | Py::Float(_) => Ok(jq_num((v.num().ok_or(JqErr)? * m).round() / m)),
        _ => Err(JqErr),
    }
}

/// `split("/gpucheck-out/") | last`.
fn after_gpucheck_out(v: &Py) -> J<String> {
    let s = v.as_str().ok_or(JqErr)?;
    if s.is_empty() {
        return Ok("null".into());
    }
    Ok(s.rsplit("/gpucheck-out/").next().unwrap_or(s).to_owned())
}

fn stage_row(r: &Py) -> J<Option<String>> {
    let stage = jget(r, "stage")?;
    if stage.is_none() {
        return Ok(None);
    }
    Ok(Some(format!(
        "| {} | {} | {} |",
        jq_text(stage),
        jq_text(jget(r, "status")?),
        round_to(jget(r, "elapsed_s")?, 10.0)?
    )))
}

/// The rows of one stage report's "Generated outputs"; on an error the rows
/// before it stand (jq printed them).
fn output_rows(r: &Py, out: &mut Vec<String>) -> J<()> {
    let stage = jget(r, "stage")?;
    if stage.is_none() {
        return Ok(());
    }
    let s = stage.as_str().ok_or(JqErr)?;
    if !(s.starts_with("model-") || s.starts_with("parity-")) {
        return Ok(());
    }
    // `map` and `.[]` iterate an array's items or an object's values.
    let checks = jget(r, "checks")?;
    let list: Vec<&Py> = match checks {
        Py::List(v) => v.iter().collect(),
        Py::Dict(items) => items.iter().map(|(_, v)| v).collect(),
        _ => return Err(JqErr),
    };
    // `from_entries` of the `video/` checks: a later key wins.
    let mut videos: Vec<(String, Py)> = Vec::new();
    for c in &list {
        let name = jget(c, "name")?.as_str().ok_or(JqErr)?;
        if let Some(k) = name.strip_prefix("video/") {
            let v = jget(c, "values")?.clone();
            match videos.iter_mut().find(|(key, _)| key == k) {
                Some(slot) => slot.1 = v,
                None => videos.push((k.to_owned(), v)),
            }
        }
    }
    for c in &list {
        let values = jget(c, "values")?;
        let secs = jget(values, "seconds")?;
        if secs.is_none() {
            continue;
        }
        let raw = jget(c, "name")?.as_str().ok_or(JqErr)?;
        let name = raw.strip_suffix("/finite").unwrap_or(raw);
        let empty = Py::dict();
        let v = videos
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v)
            .filter(|v| jtrue(v))
            .unwrap_or(&empty);
        let rel = jget(values, "rel_l2")?;
        let psnr = jget(values, "psnr_db")?;
        let mp4 = if jtrue(jget(v, "mp4")?) {
            after_gpucheck_out(jget(v, "mp4")?)?
        } else if jtrue(jget(v, "error")?) {
            "encode failed".into()
        } else {
            "—".into()
        };
        out.push(format!(
            "| {s} | {name} | {} | {} | {} | {mp4} |",
            round_to(secs, 1000.0)?,
            if rel.is_none() {
                "—".into()
            } else {
                jq_text(rel)
            },
            if psnr.is_none() {
                "—".into()
            } else {
                tostring_head(psnr, 6)
            }
        ));
    }
    Ok(())
}

fn clip_row(r: &Py) -> J<String> {
    let c = jget(r, "context")?;
    let dash = Py::Str("—".into());
    let q = Py::Str("?".into());
    let t = |k: &str| jpath(c, &["timings", k]);
    let warm = if jtrue(jpath(c, &["timings", "warm"])?) {
        "yes"
    } else {
        "cold"
    };
    let generate = match t("generate_s") {
        Ok(v) if jtrue(v) => v,
        _ => alt(t("total_s"), &dash),
    };
    let mp4 = if jtrue(jpath(c, &["artifacts", "mp4"])?) {
        after_gpucheck_out(jpath(c, &["artifacts", "mp4"])?)?
    } else {
        "—".into()
    };
    Ok(format!(
        "| {} | {} | {}x{} | {} | {warm} | {} | {} | {} | {} | {} | {} | {} | {} | {mp4} |",
        jq_text(jget(r, "stage")?),
        jq_text(alt(jpath(c, &["spec", "frames"]), &dash)),
        jq_text(alt(jpath(c, &["spec", "width"]), &q)),
        jq_text(alt(jpath(c, &["spec", "height"]), &q)),
        jq_text(alt(jpath(c, &["spec", "steps"]), &dash)),
        tostring_head(alt(t("load_s"), &dash), 6),
        tostring_head(alt(t("denoise_s"), &dash), 7),
        tostring_head(alt(t("per_step_s"), &dash), 7),
        tostring_head(alt(t("vae_decode_s"), &dash), 7),
        tostring_head(alt(t("write_s"), &dash), 6),
        tostring_head(generate, 7),
        tostring_head(alt(t("seconds_per_video_second"), &dash), 6),
        jq_text(alt(jpath(c, &["timings", "peak_mib", "denoise"]), &dash)),
    ))
}

fn probe_row(r: &Py) -> J<String> {
    let q = Py::Str("?".into());
    let p = |k: &str| alt(jpath(r, &["context", "projection", k]), &q);
    Ok(format!(
        "- {}: {}; projected total {} s (DiT forward {} s at {} tokens), peak {} MiB",
        jq_text(jget(r, "stage")?),
        jq_text(jget(r, "status")?),
        tostring_head(p("total_s"), 7),
        tostring_head(p("dit_forward_s"), 7),
        jq_text(p("target_tokens")),
        tostring_head(p("peak_mib"), 7)
    ))
}

/// The JSON values of a file, as jq reads a stream; `None` when it does not
/// parse (jq's error, dropped by `2>/dev/null || true`).
fn jq_values(path: &Path) -> Option<Vec<Py>> {
    let text = std::fs::read_to_string(path).ok()?;
    let t = text.trim();
    if t.is_empty() {
        return Some(Vec::new());
    }
    pyfmt::loads(t).ok().map(|v| vec![v])
}

fn json_files(dir: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| {
                    let n = name_of(p);
                    n.starts_with(prefix) && n.ends_with(".json") && !n.starts_with('.')
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort_by_key(|p| name_of(p));
    v
}

/// `timings.sh <run dir>`: `timings.md`'s text.
pub fn timings(run: &Path) -> R<String> {
    let reports = run.join("remote");
    let files = json_files(&reports, "");
    if files.is_empty() {
        return Err(format!("no stage reports in {}", reports.display()));
    }
    let base = name_of(&run.components().collect::<PathBuf>());
    let mut out = format!("# Test timings — {base}\n\n");
    let strict = |p: &Path| -> R<Py> { pyfmt::load(p).map_err(|e| format!("jq: {e}")) };
    let s = run.join("summary.json");
    if s.is_file() {
        let v = strict(&s)?;
        let g = |k: &str| {
            jget(&v, k)
                .map(jq_text)
                .map_err(|_| format!("jq: {}: not an object", s.display()))
        };
        out.push_str(&format!(
            "Run **{}** in {} min, ~${} at ${}/hr (instance {}).\n\n",
            g("status")?,
            g("wall_minutes")?,
            g("est_cost_usd")?,
            g("dph")?,
            g("instance")?
        ));
    }
    let env = reports.join("env.json");
    if env.is_file() {
        let v = strict(&env)?;
        let g = |k: &str| {
            jget(&v, k)
                .map(jq_text)
                .map_err(|_| format!("jq: {}: not an object", env.display()))
        };
        out.push_str(&format!(
            "GPU: {} ({} MiB, sm {}, driver CUDA {}), {} cores, {} GB RAM.\n\n",
            g("gpu")?,
            g("vram_mib")?,
            g("compute_cap")?,
            g("driver_cuda")?,
            g("cores")?,
            g("ram_gb")?
        ));
    }
    out.push_str("## Stages\n\n| stage | status | wall s |\n|---|---|---:|\n");
    let cached: Vec<String> = std::fs::read_to_string(run.join("cached-refs"))
        .map(|t| t.split_whitespace().map(str::to_owned).collect())
        .unwrap_or_default();
    for f in &files {
        let Some(values) = jq_values(f) else { continue };
        let mut rows = Vec::new();
        for v in &values {
            match stage_row(v) {
                Ok(Some(r)) => rows.push(r),
                Ok(None) => {}
                Err(_) => break,
            }
        }
        let mut row = rows.join("\n");
        if row.is_empty() {
            continue;
        }
        for c in &cached {
            if row.starts_with(&format!("| {c}-{c}-cpu-ref |")) {
                row = row.replacen("| pass |", "| cached (earlier run) |", 1);
            }
        }
        out.push_str(&row);
        out.push('\n');
    }
    out.push_str("\n## Generated outputs\n\nTime is end-to-end generation on the stage's device (sampler videos: denoise + VAE decode). GPU rows compare against the CPU-path reference.\n\n| stage | output | seconds | rel_l2 | PSNR dB | mp4 |\n|---|---|---:|---:|---:|---|\n");
    for f in &files {
        let Some(values) = jq_values(f) else { continue };
        let mut rows = Vec::new();
        for v in &values {
            if output_rows(v, &mut rows).is_err() {
                break;
            }
        }
        for r in rows {
            out.push_str(&r);
            out.push('\n');
        }
    }
    let clips = json_files(&reports, "clip-");
    if !clips.is_empty() {
        out.push_str("\n## Clips\n\n| clip | frames | resolution | steps | warm | load s | denoise s | per step s | decode s | write s | generate s | s per video-second | peak MiB | mp4 |\n|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
        for f in &clips {
            for v in jq_values(f).unwrap_or_default() {
                match clip_row(&v) {
                    Ok(r) => {
                        out.push_str(&r);
                        out.push('\n');
                    }
                    Err(_) => break,
                }
            }
        }
    }
    let probes = json_files(&reports, "probe-");
    if !probes.is_empty() {
        out.push_str("\n## Projections\n\n");
        for f in &probes {
            for v in jq_values(f).unwrap_or_default() {
                match probe_row(&v) {
                    Ok(r) => {
                        out.push_str(&r);
                        out.push('\n');
                    }
                    Err(_) => break,
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jq_alternative_takes_the_right_side_on_null_false_and_errors() {
        let v = pyfmt::loads(r#"{"a": {"b": false}, "n": 3}"#).unwrap();
        let d = Py::Str("—".into());
        assert_eq!(alt(jpath(&v, &["a", "b"]), &d), &d);
        assert_eq!(alt(jpath(&v, &["n", "x"]), &d), &d);
        assert_eq!(alt(jpath(&v, &["n"]), &d), &Py::int(3));
        assert!(jpath(&v, &["n", "x"]).is_err());
        assert_eq!(round_to(&Py::Float(12.345678), 10.0).unwrap(), "12.3");
        assert_eq!(round_to(&Py::Float(0.25), 10.0).unwrap(), "0.3");
        assert_eq!(
            after_gpucheck_out(&Py::Str("/x/gpucheck-out/a/b.mp4".into())).unwrap(),
            "a/b.mp4"
        );
    }
}
