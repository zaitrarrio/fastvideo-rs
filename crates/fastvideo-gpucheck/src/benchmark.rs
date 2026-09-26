//! `benchmark.json`: one generation's timings, memory and compute/reuse
//! counters, written next to a matrix cell's outputs so the promotion gate
//! (`fv-gpucheck gate`) reads numbers, not logs.
//!
//! Key names follow sol-engine where the concept matches:
//!
//! - `scripts/collect_run.py` (`TIMING_FIELDS`, `PRESERVED_RUNNER_BENCHMARK_FIELDS`):
//!   `total_s`, `denoise_s`, `decode_s`, `stage_seconds`, `schema_version`,
//!   `timing_scope`, `warm_steady_state`, `warmup_requests`,
//!   `includes_model_load`, `wall_total_s`, `max_device_memory_used_mib`,
//!   `memory`, `aggregate`. `denoise_s` is `evals/profiles/*.toml
//!   [performance] primary_metric`, `total_s` the secondary.
//! - MiniMax-H3 RTX5090 runner (`benchmark.json` `measured` / `workload`):
//!   `inference_time_s`, `peak_memory_mb`, `workload.{height, width,
//!   duration_s, measured_steps, prompt_sha256, seed, task}`.
//! - LTX-2.5 RTX5090 runner (`gpu_infer.py` metrics): `stage_1_seconds`,
//!   `stage_2_seconds`, `video_vae_seconds`, `e2e_seconds`,
//!   `peak_allocated_gib`, `peak_reserved_gib`, `attention.{video_calls,
//!   sol_calls, dense_video_calls, tau_calls}`, `gpu`.
//! - TeaCache decisions (`teacache.py` `teacache_decision` /
//!   `teacache_generation_summary`): `calls`, `compute`, `reuse`,
//!   `reuse_rate`, per decision `step_index`, `action`, `reason`,
//!   `relative_l1`, `rescaled_l1`, `accumulator`.
//!
//! With several prompts (`--prompts`), the cell's `benchmark.json` holds each
//! prompt's document under `prompts` and, at the top level, the median of
//! every numeric leaf across them (`aggregate.statistic = "median"`), so a
//! reader of the single-prompt keys reads medians.

use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

pub const SCHEMA_VERSION: &str = "fastvideo-rs.benchmark.v1";

/// One prompt of a prompt set.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptSpec {
    pub name: String,
    pub prompt: String,
    pub seed: u64,
}

/// A prompt-set file: `{"name": .., "source": .., "prompts": [{"name",
/// "prompt", "seed"}]}` (`seed` defaults to `default_seed`). A bare JSON
/// list of such objects is accepted too.
pub fn load_prompts(path: &Path, default_seed: u64) -> anyhow::Result<Vec<PromptSpec>> {
    let text =
        std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let v: Value =
        serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let list = match &v {
        Value::Array(a) => a.clone(),
        Value::Object(o) => o
            .get("prompts")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("{}: no `prompts` list", path.display()))?,
        _ => anyhow::bail!("{}: expected an object or a list", path.display()),
    };
    let mut out = Vec::with_capacity(list.len());
    for (i, p) in list.iter().enumerate() {
        let prompt = p["prompt"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("{}: prompt {i} has no `prompt`", path.display()))?;
        let name = p["name"]
            .as_str()
            .map_or_else(|| format!("p{:02}", i + 1), str::to_string);
        if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') {
            anyhow::bail!(
                "{}: prompt name `{name}` is not a plain directory name",
                path.display()
            );
        }
        let seed = p["seed"].as_u64().unwrap_or(default_seed);
        out.push(PromptSpec {
            name,
            prompt: prompt.to_string(),
            seed,
        });
    }
    if out.is_empty() {
        anyhow::bail!("{}: empty prompt set", path.display());
    }
    let mut names: Vec<&str> = out.iter().map(|p| p.name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    if names.len() != out.len() {
        anyhow::bail!("{}: prompt names must be unique", path.display());
    }
    Ok(out)
}

/// `hashlib.sha256(prompt.encode()).hexdigest()`, as the H3 runner records it.
pub fn sha256_hex(s: &str) -> String {
    sha256(s.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// FIPS 180-4 SHA-256 (no dependency for one digest per prompt).
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, c) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

/// Every `FASTVIDEO_*` switch of this process: what the run was configured
/// with, for a reader deciding whether two benchmarks are comparable.
pub fn env_switches() -> Value {
    let mut m: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| k.starts_with("FASTVIDEO_") || k.starts_with("FASTH3_"))
        .collect();
    m.sort();
    Value::Object(m.into_iter().map(|(k, v)| (k, Value::String(v))).collect())
}

/// GPU name, as the LTX runner's `gpu` key.
pub fn gpu_name() -> Option<String> {
    #[cfg(feature = "cuda")]
    {
        let dev = fastvideo_cudarc::wan::device::global_device()?;
        dev.ctx.name().ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

/// Fields every benchmark document carries, merged under `doc`.
pub fn common(pipeline: &str, warm: bool) -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "pipeline": pipeline,
        "timing_scope": "one warm generate() call: text + denoise + decode + write, model load excluded",
        "timing_note": "total_s = inference_time_s = e2e_seconds; load_s is separate",
        "warm_steady_state": warm,
        "warmup_requests": u32::from(warm),
        "includes_model_load": false,
        "activations": if fastvideo_cudarc::wan::tensor::bf16_activations() { "bf16" } else { "f32" },
        "gpu": gpu_name(),
        "hardware": gpu_name().map(|g| format!("1x {g}")),
        "env": env_switches(),
    })
}

/// Shallow-merge `extra` into `base` (objects only).
pub fn merge(mut base: Value, extra: Value) -> Value {
    if let (Value::Object(b), Value::Object(e)) = (&mut base, extra) {
        for (k, v) in e {
            b.insert(k, v);
        }
    }
    base
}

fn median(v: &mut [f64]) -> Option<f64> {
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
}

/// The median of every numeric leaf present in all `docs` (objects walked
/// recursively; equal-length numeric arrays elementwise). Non-numeric values
/// shared by every doc are kept when equal.
pub fn median_of(docs: &[&Value]) -> Value {
    let Some(first) = docs.first() else {
        return Value::Null;
    };
    match *first {
        Value::Number(_) => {
            let mut xs: Vec<f64> = docs.iter().filter_map(|d| d.as_f64()).collect();
            if xs.len() != docs.len() {
                return Value::Null;
            }
            median(&mut xs).map_or(Value::Null, |m| json!(m))
        }
        Value::Object(o) => {
            let mut out = Map::new();
            for k in o.keys() {
                let children: Vec<&Value> = docs.iter().filter_map(|d| d.get(k)).collect();
                if children.len() != docs.len() {
                    continue;
                }
                let m = median_of(&children);
                if !m.is_null() {
                    out.insert(k.clone(), m);
                }
            }
            Value::Object(out)
        }
        Value::Array(a) => {
            let same = docs
                .iter()
                .all(|d| d.as_array().is_some_and(|b| b.len() == a.len()));
            if !same || !a.iter().all(Value::is_number) {
                return Value::Null;
            }
            (0..a.len())
                .map(|i| {
                    let col: Vec<&Value> = docs.iter().map(|d| &d[i]).collect();
                    median_of(&col)
                })
                .collect::<Vec<_>>()
                .into()
        }
        other => {
            if docs.iter().all(|d| *d == other) {
                other.clone()
            } else {
                Value::Null
            }
        }
    }
}

/// The multi-prompt summary: medians at the top level, each prompt's
/// document under `prompts`, the per-prompt primary metrics side by side.
pub fn summarize(prompts: &[(PromptSpec, Value)]) -> Value {
    let docs: Vec<&Value> = prompts.iter().map(|(_, d)| d).collect();
    let mut top = median_of(&docs);
    let per_prompt: Vec<Value> = prompts
        .iter()
        .map(|(p, d)| {
            json!({
                "name": p.name,
                "seed": p.seed,
                "prompt_sha256": sha256_hex(&p.prompt),
                "total_s": d.get("total_s"),
                "denoise_s": d.get("denoise_s"),
                "decode_s": d.get("decode_s"),
                "peak_memory_mb": d.get("peak_memory_mb"),
            })
        })
        .collect();
    if let Value::Object(m) = &mut top {
        m.insert(
            "aggregate".into(),
            json!({"statistic": "median", "n": prompts.len(), "per_prompt": per_prompt}),
        );
        m.insert(
            "prompts".into(),
            Value::Object(
                prompts
                    .iter()
                    .map(|(p, d)| (p.name.clone(), d.clone()))
                    .collect(),
            ),
        );
    }
    top
}

/// Where a cell's `benchmark.json` goes: beside the clip directory (the cell
/// directory in the matrix, whose clip dir is `<cell>/frames`).
pub fn path_beside(clip_dir: &Path) -> PathBuf {
    match clip_dir.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join("benchmark.json"),
        _ => PathBuf::from("benchmark.json"),
    }
}

pub fn write(path: &Path, doc: &Value) -> anyhow::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(doc)? + "\n")
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    eprintln!("benchmark.json → {}", path.display());
    Ok(())
}

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1u32 << 30)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_digests() {
        assert_eq!(
            sha256_hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // Two blocks.
        assert_eq!(
            sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn medians_walk_numeric_leaves() {
        let a =
            json!({"total_s": 1.0, "x": {"y": 10}, "steps": [1.0, 5.0], "name": "h3", "s": "a"});
        let b =
            json!({"total_s": 3.0, "x": {"y": 30}, "steps": [3.0, 1.0], "name": "h3", "s": "b"});
        let c =
            json!({"total_s": 2.0, "x": {"y": 20}, "steps": [2.0, 3.0], "name": "h3", "s": "c"});
        let m = median_of(&[&a, &b, &c]);
        assert_eq!(m["total_s"], 2.0);
        assert_eq!(m["x"]["y"], 20.0);
        assert_eq!(m["steps"], json!([2.0, 3.0]));
        assert_eq!(m["name"], "h3");
        assert!(m.get("s").is_none(), "differing strings are dropped");
        let even = median_of(&[&a, &b]);
        assert_eq!(even["total_s"], 2.0);
    }

    #[test]
    fn summary_keeps_prompts_and_medians() {
        let p = |n: &str| PromptSpec {
            name: n.into(),
            prompt: n.into(),
            seed: 1,
        };
        let s = summarize(&[
            (p("a"), json!({"total_s": 4.0, "denoise_s": 3.0})),
            (p("b"), json!({"total_s": 2.0, "denoise_s": 1.0})),
            (p("c"), json!({"total_s": 3.0, "denoise_s": 2.0})),
        ]);
        assert_eq!(s["total_s"], 3.0);
        assert_eq!(s["denoise_s"], 2.0);
        assert_eq!(s["aggregate"]["n"], 3);
        assert_eq!(s["prompts"]["b"]["total_s"], 2.0);
        assert_eq!(s["aggregate"]["per_prompt"][0]["name"], "a");
    }

    #[test]
    fn prompt_sets_parse_and_validate() {
        let dir = std::env::temp_dir().join(format!("fv-bench-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("p.json");
        std::fs::write(
            &f,
            r#"{"prompts": [{"name": "a", "prompt": "x", "seed": 7}, {"prompt": "y"}]}"#,
        )
        .unwrap();
        let ps = load_prompts(&f, 42).unwrap();
        assert_eq!(ps[0].seed, 7);
        assert_eq!(ps[1].name, "p02");
        assert_eq!(ps[1].seed, 42);
        std::fs::write(&f, r#"[{"name": "../x", "prompt": "x"}]"#).unwrap();
        assert!(load_prompts(&f, 1).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn repo_prompt_set_is_five_named_seeded_prompts() {
        let f = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/gpu/prompts-eval.json");
        let ps = load_prompts(&f, 0).unwrap();
        assert_eq!(ps.len(), 5);
        assert!(ps.iter().all(|p| p.prompt.len() > 40));
    }

    #[test]
    fn benchmark_path_is_the_cell_dir() {
        assert_eq!(
            path_beside(Path::new("/r/cell/frames")),
            Path::new("/r/cell/benchmark.json")
        );
        assert_eq!(
            path_beside(Path::new("frames")),
            Path::new("benchmark.json")
        );
    }
}
