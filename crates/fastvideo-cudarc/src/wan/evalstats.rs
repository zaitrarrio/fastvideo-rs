//! Compute / reuse counters for a generation's `benchmark.json`
//! (`fv-gpucheck`'s evaluation layer): what each denoise forward actually ran.
//!
//! - TeaCache: every decision (computed or reused step, the relative L1 of
//!   the modulated probe and the rescaled indicator it accumulated), the
//!   fields sol-engine's `teacache_decision` events carry
//!   (`models/minimax_h3/RTX5090/teacache.py`).
//! - Video self-attention routes per forward and layer: dense, Sol-Attn at a
//!   tau, PISA at a sparsity, or VSA; summarized with the keys of sol-engine's
//!   LTX-2.5 `attention.stats()` (`video_calls`, `sol_calls`,
//!   `dense_video_calls`, `tau_calls`; `models/ltx25/RTX5090/attention.py`).
//! - VSA density: key tiles attended over all (query tile, key tile) pairs.
//! - FFN row chunking: calls and the chunks they ran in.
//! - Quantized linears by recipe: GEMM calls per run and modules quantized
//!   at load (the latter survive [`reset`]: they describe the loaded model).
//!
//! Recording is a mutex-guarded push per block or linear call — thousands per
//! step, negligible against the GEMMs — and always on, so a benchmark never
//! depends on a debug switch.

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde_json::{json, Value};

/// One video self-attention call's kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AttnKind {
    Dense,
    Sol { tau: f64 },
    Pisa { sparsity: f64 },
    Vsa,
}

#[derive(Clone, Debug)]
struct TeaDecision {
    step: usize,
    compute: bool,
    reason: &'static str,
    relative_l1: Option<f64>,
    rescaled_l1: Option<f64>,
    accumulator: f64,
}

#[derive(Default)]
struct Forward {
    /// Denoise step when the caller knows it (H3), else the forward counter.
    step: Option<usize>,
    layers: Vec<(usize, AttnKind)>,
}

#[derive(Default)]
struct Run {
    teacache: Vec<TeaDecision>,
    forwards: Vec<Forward>,
    vsa_calls: u64,
    vsa_selected: u128,
    vsa_total: u128,
    vsa_video_selected: u128,
    vsa_video_total: u128,
    ffn_calls: u64,
    ffn_chunks: u64,
    ffn_chunked_calls: u64,
    quant_calls: BTreeMap<&'static str, u64>,
}

static RUN: Mutex<Option<Run>> = Mutex::new(None);
static QUANT_MODULES: Mutex<BTreeMap<&'static str, u64>> = Mutex::new(BTreeMap::new());

fn with_run(f: impl FnOnce(&mut Run)) {
    if let Ok(mut g) = RUN.lock() {
        f(g.get_or_insert_with(Run::default));
    }
}

/// Start a new measured generation: clears everything but the load-time
/// module counts.
pub fn reset() {
    if let Ok(mut g) = RUN.lock() {
        *g = Some(Run::default());
    }
}

/// A TeaCache decision at denoise `step`.
pub fn teacache_decision(
    step: usize,
    compute: bool,
    reason: &'static str,
    relative_l1: Option<f64>,
    rescaled_l1: Option<f64>,
    accumulator: f64,
) {
    with_run(|r| {
        r.teacache.push(TeaDecision {
            step,
            compute,
            reason,
            relative_l1,
            rescaled_l1,
            accumulator,
        })
    });
}

/// One video self-attention call of `layer`. `step` is the denoise step when
/// the caller has it; otherwise layer 0 opens the next forward.
pub fn attn(step: Option<usize>, layer: usize, kind: AttnKind) {
    with_run(|r| {
        let open_new = match r.forwards.last() {
            None => true,
            Some(f) => {
                layer == 0
                    || f.layers.last().is_some_and(|(l, _)| *l >= layer)
                    || (step.is_some() && f.step != step)
            }
        };
        if open_new {
            r.forwards.push(Forward {
                step,
                layers: Vec::new(),
            });
        }
        r.forwards.last_mut().unwrap().layers.push((layer, kind));
    });
}

/// One VSA call over `heads` heads: `selected` of `total` (query tile, key
/// tile) pairs attended per head; `video_selected` / `video_total` the same
/// restricted to video-to-video tiles (the part the budget applies to).
pub fn vsa(heads: usize, selected: u64, total: u64, video_selected: u64, video_total: u64) {
    let h = heads as u128;
    with_run(|r| {
        r.vsa_calls += 1;
        r.vsa_selected += h * u128::from(selected);
        r.vsa_total += h * u128::from(total);
        r.vsa_video_selected += h * u128::from(video_selected);
        r.vsa_video_total += h * u128::from(video_total);
    });
}

/// One feed-forward call that ran in `chunks` row chunks.
pub fn ffn(chunks: usize) {
    with_run(|r| {
        r.ffn_calls += 1;
        r.ffn_chunks += chunks as u64;
        if chunks > 1 {
            r.ffn_chunked_calls += 1;
        }
    });
}

/// One quantized-linear GEMM under `recipe` (`w8a8`, `mxfp8`, `nvfp4_w4a4`, ...).
pub fn quant_call(recipe: &'static str) {
    with_run(|r| *r.quant_calls.entry(recipe).or_default() += 1);
}

/// A linear quantized at load under `recipe`.
pub fn quant_module(recipe: &'static str) {
    if let Ok(mut g) = QUANT_MODULES.lock() {
        *g.entry(recipe).or_default() += 1;
    }
}

fn tau_key(t: f64) -> String {
    // sol-engine keys tau_calls by `str(float)`: 1.0, 1.25, 1.5.
    let s = format!("{t}");
    if s.contains('.') {
        s
    } else {
        format!("{s}.0")
    }
}

/// The counters as the `benchmark.json` blocks `teacache`, `attention`,
/// `vsa`, `ffn_chunking` and `quantized_linears`.
pub fn snapshot() -> Value {
    let g = RUN.lock().ok();
    let empty = Run::default();
    let r = g.as_ref().and_then(|g| g.as_ref()).unwrap_or(&empty);
    let modules = QUANT_MODULES.lock().map(|m| m.clone()).unwrap_or_default();

    let teacache = if r.teacache.is_empty() {
        json!({"enabled": false})
    } else {
        let compute = r.teacache.iter().filter(|d| d.compute).count();
        let calls = r.teacache.len();
        json!({
            "enabled": true,
            "calls": calls,
            "compute": compute,
            "reuse": calls - compute,
            "reuse_rate": (calls - compute) as f64 / calls as f64,
            "computed_steps": r.teacache.iter().filter(|d| d.compute).map(|d| d.step).collect::<Vec<_>>(),
            "reused_steps": r.teacache.iter().filter(|d| !d.compute).map(|d| d.step).collect::<Vec<_>>(),
            "decisions": r.teacache.iter().map(|d| json!({
                "step_index": d.step,
                "action": if d.compute { "compute" } else { "reuse" },
                "reason": d.reason,
                "relative_l1": d.relative_l1,
                "rescaled_l1": d.rescaled_l1,
                "accumulator": d.accumulator,
            })).collect::<Vec<_>>(),
        })
    };

    let mut tau_calls: BTreeMap<String, u64> = BTreeMap::new();
    let mut pisa_calls: BTreeMap<String, u64> = BTreeMap::new();
    let (mut video, mut sol, mut dense, mut pisa, mut vsa_calls) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut per_forward = Vec::with_capacity(r.forwards.len());
    for (i, f) in r.forwards.iter().enumerate() {
        let (mut fd, mut fs, mut fp, mut fv) = (0u64, 0u64, 0u64, 0u64);
        let mut ftau: BTreeMap<String, u64> = BTreeMap::new();
        let mut dense_layers = Vec::new();
        for &(layer, kind) in &f.layers {
            video += 1;
            match kind {
                AttnKind::Dense => {
                    dense += 1;
                    fd += 1;
                    dense_layers.push(layer);
                }
                AttnKind::Sol { tau } => {
                    sol += 1;
                    fs += 1;
                    *tau_calls.entry(tau_key(tau)).or_default() += 1;
                    *ftau.entry(tau_key(tau)).or_default() += 1;
                }
                AttnKind::Pisa { sparsity } => {
                    pisa += 1;
                    fp += 1;
                    *pisa_calls.entry(tau_key(sparsity)).or_default() += 1;
                }
                AttnKind::Vsa => {
                    vsa_calls += 1;
                    fv += 1;
                }
            }
        }
        per_forward.push(json!({
            "forward": i,
            "step": f.step,
            "layers": f.layers.len(),
            "dense": fd,
            "sol": fs,
            "pisa": fp,
            "vsa": fv,
            "tau_calls": ftau,
            "dense_layers": dense_layers,
        }));
    }
    let attention = json!({
        "forwards": r.forwards.len(),
        "video_calls": video,
        "sol_calls": sol,
        "dense_video_calls": dense,
        "pisa_calls": pisa,
        "vsa_calls": vsa_calls,
        "tau_calls": tau_calls,
        "pisa_sparsity_calls": pisa_calls,
        "per_forward": per_forward,
    });
    let ratio = |a: u128, b: u128| {
        if b == 0 {
            Value::Null
        } else {
            json!(a as f64 / b as f64)
        }
    };
    let vsa = json!({
        "calls": r.vsa_calls,
        "selected_tiles": r.vsa_selected as f64,
        "total_tiles": r.vsa_total as f64,
        "density": ratio(r.vsa_selected, r.vsa_total),
        "video_selected_tiles": r.vsa_video_selected as f64,
        "video_total_tiles": r.vsa_video_total as f64,
        "video_density": ratio(r.vsa_video_selected, r.vsa_video_total),
    });
    let ffn = json!({
        "calls": r.ffn_calls,
        "chunks": r.ffn_chunks,
        "chunked_calls": r.ffn_chunked_calls,
    });
    let quant = json!({
        "modules": modules,
        "calls": r.quant_calls,
    });
    json!({
        "teacache": teacache,
        "attention": attention,
        "vsa": vsa,
        "ffn_chunking": ffn,
        "quantized_linears": quant,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test: the counters are process-global.
    #[test]
    fn counters_summarize_like_sol_engine() {
        reset();
        for step in 0..2 {
            teacache_decision(
                step,
                step == 0,
                if step == 0 {
                    "warmup"
                } else {
                    "below_threshold"
                },
                None,
                None,
                0.0,
            );
            for layer in 0..3 {
                let kind = if layer == 0 {
                    AttnKind::Dense
                } else {
                    AttnKind::Sol { tau: 1.25 }
                };
                attn(None, layer, kind);
            }
        }
        attn(Some(5), 0, AttnKind::Vsa);
        vsa(2, 30, 100, 10, 50);
        ffn(1);
        ffn(3);
        quant_call("w8a8");
        quant_call("w8a8");
        let s = snapshot();
        assert_eq!(s["teacache"]["compute"], 1);
        assert_eq!(s["teacache"]["reuse"], 1);
        assert_eq!(s["teacache"]["reused_steps"], json!([1]));
        let a = &s["attention"];
        assert_eq!(a["forwards"], 3);
        assert_eq!(a["video_calls"], 7);
        assert_eq!(a["sol_calls"], 4);
        assert_eq!(a["dense_video_calls"], 2);
        assert_eq!(a["tau_calls"]["1.25"], 4);
        assert_eq!(a["per_forward"][2]["step"], 5);
        assert_eq!(s["vsa"]["density"], 0.3);
        assert_eq!(s["vsa"]["video_density"], 0.2);
        assert_eq!(s["ffn_chunking"]["chunks"], 4);
        assert_eq!(s["ffn_chunking"]["chunked_calls"], 1);
        assert_eq!(s["quantized_linears"]["calls"]["w8a8"], 2);
        assert_eq!(tau_key(1.0), "1.0");
        reset();
        assert_eq!(snapshot()["attention"]["video_calls"], 0);
    }
}
