//! LongLive 1.0 (NVlabs, arXiv 2509.22622; `NVlabs/LongLive` `LongLive1.0/`)
//! on the SF-Wan causal engine ([`super::stream`]). Opt-in: nothing here
//! runs unless a rollout is configured from [`LongLiveConfig`] and its
//! transformer is loaded with [`load_transformer_map`].
//!
//! LongLive-1.3B is Wan2.1-T2V-1.3B trained Self-Forcing style (4 DMD steps,
//! 3-latent-frame blocks), then tuned for long, interactive rollouts:
//!
//! * **frame sink + short window**: `local_attn_size = 12` latent frames of
//!   KV, the first `sink_size = 3` of which are the first block, kept for the
//!   whole rollout (`global_sink: true`). Queries read the sink and the
//!   newest 9 frames (their own block included). Our rolling cache with
//!   `local_attn_frames = 12`, `sink_frames = 3` is that window
//!   ([`super::causal::KvSpec::rolling`]).
//! * **RoPE**: the released interactive config ropes keys at their absolute
//!   frame (`causal_rope_apply(start_frame=current_start_frame)`, our
//!   [`RopePolicy::Absolute`]); the `longlive_inference_infinity.yaml`
//!   config (`use_infinite_attention`, `causal_model_infinity.py`,
//!   Infinity-RoPE) caches un-roped keys and ropes the window by slot
//!   (sink `0..3`, then the rolled frames, queries at the tail): FastVideo's
//!   `relativistic` policy, of which [`RopePolicy::RebasedSink`] gives the
//!   same query-key offsets (`causal::tests::rebased_sink_matches_relativistic`).
//! * **KV re-cache at a prompt switch** (`InteractiveCausalInferencePipeline
//!   ._recache_after_switch`): once per switch, before the next block, the
//!   last `min(local_attn_size, current_frame)` clean latent frames run one
//!   forward at `t = context_noise` (0) under the new prompt, starting at
//!   their own frame, and their keys and values overwrite the cache; with the
//!   global sink the sink slots are left as they were (the `is_recompute`
//!   guard, [`super::causal::CausalKvCache::set_sink_guard`]) unless the
//!   re-cache starts at frame 0. See [`recache_plan`].
//! * **weights**: `longlive_base.pt` (`{"generator": state_dict}`, original
//!   Wan names under `model.`) plus `lora.pt` (`{"generator_lora": ...}`,
//!   PEFT LoRA rank 256 / alpha 256 on every `nn.Linear` of every
//!   `CausalWanAttentionBlock`). The Hub files are torch pickles; the fetch
//!   converts them to safetensors with the same keys (no pickle reader here),
//!   and [`load_transformer_map`] renames to Diffusers and merges the LoRA
//!   (`W += alpha / rank · B @ A`).
//!
//! Schedule, shift and context pass are SF-Wan's: denoising steps
//! `[1000, 750, 500, 250]` warped onto `FlowMatchScheduler(shift=5,
//! sigma_min=0, extra_one_step=True)`, then the clean pass at `t = 0`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use fastvideo_loader::{LazyDType, LazyStore, RawDType, RawTensor};
use rayon::prelude::*;

use super::stream::{PromptSwitch, RolloutConfig, RopePolicy};
use super::tensor::{Result, TensorError};
use super::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// The inference settings of `LongLive1.0/configs/longlive_*inference*.yaml`.
#[derive(Debug, Clone, PartialEq)]
pub struct LongLiveConfig {
    /// `denoising_step_list` (warped: `warp_denoising_step: true`).
    pub denoising_steps: Vec<i32>,
    /// `num_frame_per_block` (latent frames per block).
    pub frames_per_block: usize,
    /// `model_kwargs.local_attn_size`: KV window in latent frames, sink
    /// included.
    pub local_attn_frames: usize,
    /// `model_kwargs.sink_size`.
    pub sink_frames: usize,
    /// `model_kwargs.timestep_shift`.
    pub timestep_shift: f64,
    /// `global_sink`: the sink keeps the first block for the whole rollout,
    /// a re-cache leaves it alone. `false` re-caches the sink too.
    pub global_sink: bool,
    /// `context_noise` (the clean pass's timestep).
    pub context_noise: i32,
    /// `adapter.rank` / `adapter.alpha` of `lora.pt`.
    pub lora_rank: usize,
    pub lora_alpha: f32,
    /// `use_infinite_attention` (the `_infinity` config): relative,
    /// window-slot RoPE instead of absolute frames.
    pub relative_rope: bool,
    /// Re-cache at a prompt switch (LongLive's default). Off: the cache is
    /// kept as it is, the new prompt only conditions the next blocks (our
    /// [`PromptSwitch::Keep`]), the "w/o KV re-cache" ablation.
    pub recache: bool,
}

impl LongLiveConfig {
    /// `longlive_interactive_inference.yaml` (and `longlive_inference.yaml`:
    /// the same model settings).
    pub fn interactive() -> Self {
        Self {
            denoising_steps: vec![1000, 750, 500, 250],
            frames_per_block: 3,
            local_attn_frames: 12,
            sink_frames: 3,
            timestep_shift: 5.0,
            global_sink: true,
            context_noise: 0,
            lora_rank: 256,
            lora_alpha: 256.0,
            relative_rope: false,
            recache: true,
        }
    }

    /// `longlive_inference_infinity.yaml`: the same, with relative RoPE for
    /// rollouts past the 1024-frame RoPE table.
    pub fn infinity() -> Self {
        Self {
            relative_rope: true,
            ..Self::interactive()
        }
    }

    /// `alpha / rank`, the PEFT LoRA scale.
    pub fn lora_scale(&self) -> f32 {
        self.lora_alpha / self.lora_rank.max(1) as f32
    }

    /// The rope policy of the cache. Relative RoPE maps to
    /// [`RopePolicy::RebasedSink`] (same offsets as Infinity-RoPE, the cost
    /// of absolute keys); `relativistic` picks FastVideo's literal policy.
    pub fn rope(&self, relativistic: bool) -> RopePolicy {
        match (self.relative_rope, relativistic) {
            (false, _) => RopePolicy::Absolute,
            (true, false) => RopePolicy::RebasedSink,
            (true, true) => RopePolicy::Relativistic,
        }
    }

    /// The prompt-switch behaviour.
    pub fn prompt_switch(&self) -> PromptSwitch {
        if self.recache {
            PromptSwitch::Recache {
                global_sink: self.global_sink,
            }
        } else {
            PromptSwitch::Keep
        }
    }

    /// `base` (prompt, canvas, seed, tokenizer, graphs ...) with LongLive's
    /// schedule, window, sink, RoPE and switch.
    pub fn rollout(&self, base: RolloutConfig) -> RolloutConfig {
        RolloutConfig {
            dmd_steps: self.denoising_steps.clone(),
            flow_shift: self.timestep_shift,
            local_attn_frames: self.local_attn_frames,
            sink_frames: self.sink_frames,
            rope: self.rope(false),
            prompt_switch: self.prompt_switch(),
            // LongLive has no periodic re-cache; the window restarts only at
            // a switch.
            recache: None,
            ..base
        }
    }
}

/// One KV re-cache (`_recache_after_switch`), in latent frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecachePlan {
    /// First frame recomputed (`recache_start_frame`): RoPE offset and cache
    /// position of the forward.
    pub start_frame: usize,
    /// Frames recomputed (`num_recache_frames`).
    pub frames: usize,
    /// The forward leaves the sink slots alone (`is_recompute` with the
    /// global sink).
    pub guard_sink: bool,
}

impl RecachePlan {
    /// The forward rewrites the sink slots (with new-prompt keys): no guard
    /// and the re-cache starts at frame 0 (the cache has not rolled), or
    /// `global_sink: false`.
    pub fn rewrites_sink(&self) -> bool {
        !self.guard_sink
    }
}

/// The re-cache before the block at `current_frame` (frames generated so
/// far, cache positions) with a `window_frames` KV window. `None` at frame 0
/// (nothing cached: LongLive only resets the cross-attention cache).
pub fn recache_plan(
    current_frame: usize,
    window_frames: usize,
    global_sink: bool,
) -> Option<RecachePlan> {
    if current_frame == 0 || window_frames == 0 {
        return None;
    }
    let frames = window_frames.min(current_frame);
    let start_frame = current_frame - frames;
    Some(RecachePlan {
        start_frame,
        frames,
        // `is_recompute` needs `current_start > 0`; `global_sink: false`
        // passes `sink_recache_after_switch` and writes the sink too.
        guard_sink: global_sink && start_frame > 0,
    })
}

/// Which cache slots (frames, `[0, window)`) a re-cache rewrites, given the
/// plan and the sink: what the next block reads under the new prompt.
pub fn recached_slots(plan: &RecachePlan, sink_frames: usize) -> std::ops::Range<usize> {
    let first = if plan.guard_sink {
        sink_frames.min(plan.frames)
    } else {
        0
    };
    first..plan.frames
}

/// Strip the wrappers a LongLive / Self-Forcing state dict can carry:
/// `WanDiffusionWrapper.model`, FSDP, PEFT.
fn strip_wrappers(key: &str) -> &str {
    let mut k = key;
    loop {
        let before = k;
        for p in [
            "base_model.model.",
            "_fsdp_wrapped_module.",
            "model.",
            "module.",
            "generator.",
        ] {
            if let Some(rest) = k.strip_prefix(p) {
                k = rest;
            }
        }
        if k == before {
            return k;
        }
    }
}

/// The Diffusers name of an original-Wan (`wan/modules/model.py`, as in
/// LongLive and Self-Forcing checkpoints) transformer key, the renames of
/// Diffusers' `convert_wan_to_diffusers.py`. `Ok(None)` for buffers the
/// Diffusers model does not keep (`freqs`); an error for anything unknown.
pub fn diffusers_key(original: &str) -> Result<Option<String>> {
    let k = strip_wrappers(original);
    if k == "freqs" {
        return Ok(None);
    }
    let unknown = || msg(format!("longlive: no Diffusers name for `{original}`"));
    let top = [
        ("patch_embedding.", "patch_embedding."),
        (
            "text_embedding.0.",
            "condition_embedder.text_embedder.linear_1.",
        ),
        (
            "text_embedding.2.",
            "condition_embedder.text_embedder.linear_2.",
        ),
        (
            "time_embedding.0.",
            "condition_embedder.time_embedder.linear_1.",
        ),
        (
            "time_embedding.2.",
            "condition_embedder.time_embedder.linear_2.",
        ),
        ("time_projection.1.", "condition_embedder.time_proj."),
        ("head.head.", "proj_out."),
    ];
    for (from, to) in top {
        if let Some(rest) = k.strip_prefix(from) {
            return Ok(Some(format!("{to}{rest}")));
        }
    }
    if k == "head.modulation" {
        return Ok(Some("scale_shift_table".into()));
    }
    let rest = k.strip_prefix("blocks.").ok_or_else(unknown)?;
    let (idx, tail) = rest.split_once('.').ok_or_else(unknown)?;
    idx.parse::<usize>().map_err(|_| unknown())?;
    let tail = if tail == "modulation" {
        "scale_shift_table".to_string()
    } else if let Some(t) = tail.strip_prefix("norm3.") {
        // Original norm3 is the (affine) cross-attention norm, Diffusers norm2.
        format!("norm2.{t}")
    } else if let Some(t) = tail.strip_prefix("ffn.0.") {
        format!("ffn.net.0.proj.{t}")
    } else if let Some(t) = tail.strip_prefix("ffn.2.") {
        format!("ffn.net.2.{t}")
    } else {
        let (attn, t) = if let Some(t) = tail.strip_prefix("self_attn.") {
            ("attn1", t)
        } else if let Some(t) = tail.strip_prefix("cross_attn.") {
            ("attn2", t)
        } else {
            return Err(unknown());
        };
        let (proj, p) = t.split_once('.').ok_or_else(unknown)?;
        let to = match proj {
            "q" => "to_q",
            "k" => "to_k",
            "v" => "to_v",
            "o" => "to_out.0",
            "norm_q" => "norm_q",
            "norm_k" => "norm_k",
            _ => return Err(unknown()),
        };
        format!("{attn}.{to}.{p}")
    };
    Ok(Some(format!("blocks.{idx}.{tail}")))
}

/// One LoRA pair of a PEFT state dict: the target module (original name)
/// and which half the tensor is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraKey {
    /// The module, wrappers stripped (`blocks.0.self_attn.q`).
    pub module: String,
    /// `true`: `lora_A` (`[rank, in]`); `false`: `lora_B` (`[out, rank]`).
    pub is_a: bool,
}

/// Parse `...<module>.lora_A[.default].weight` / `lora_B`.
pub fn lora_key(key: &str) -> Option<LoraKey> {
    let k = strip_wrappers(key);
    let k = k.strip_suffix(".weight")?;
    let k = k.strip_suffix(".default").unwrap_or(k);
    if let Some(m) = k.strip_suffix(".lora_A") {
        return Some(LoraKey {
            module: m.to_string(),
            is_a: true,
        });
    }
    if let Some(m) = k.strip_suffix(".lora_B") {
        return Some(LoraKey {
            module: m.to_string(),
            is_a: false,
        });
    }
    None
}

/// `w[out, in] += scale · b[out, rank] @ a[rank, in]` (row-parallel).
pub fn merge_lora(
    w: &mut [f32],
    out: usize,
    inp: usize,
    a: &[f32],
    b: &[f32],
    rank: usize,
    scale: f32,
) -> Result<()> {
    if w.len() != out * inp || a.len() != rank * inp || b.len() != out * rank {
        return Err(msg(format!(
            "lora merge: w {} (= {out}x{inp}?), A {} (= {rank}x{inp}?), B {} (= {out}x{rank}?)",
            w.len(),
            a.len(),
            b.len()
        )));
    }
    w.par_chunks_mut(inp).enumerate().for_each(|(o, row)| {
        for r in 0..rank {
            let c = scale * b[o * rank + r];
            if c == 0.0 {
                continue;
            }
            let ar = &a[r * inp..(r + 1) * inp];
            for (x, &y) in row.iter_mut().zip(ar) {
                *x += c * y;
            }
        }
    });
    Ok(())
}

/// Where the converted LongLive-1.3B checkpoint lives (the fetch's
/// safetensors conversion of `models/longlive_base.pt` and `models/lora.pt`,
/// same keys, the `generator` / `generator_lora` sub-dicts).
#[derive(Debug, Clone, PartialEq)]
pub struct LongLiveWeights {
    pub base: PathBuf,
    /// `None`: the base generator alone (`longlive_init` without the long
    /// tuning; not what LongLive serves).
    pub lora: Option<PathBuf>,
    /// LoRA scale (`alpha / rank`, 1.0 for the release).
    pub lora_scale: f32,
    /// Keep 2-D+ weights in bf16 (the DiT runs bf16 weights anyway; halves
    /// host memory). 1-D tensors and the modulation tables keep their dtype.
    pub bf16: bool,
}

impl LongLiveWeights {
    /// `<dir>/longlive_base.safetensors` and `<dir>/lora.safetensors`.
    pub fn in_dir(dir: &Path, cfg: &LongLiveConfig) -> Self {
        let lora = dir.join("lora.safetensors");
        Self {
            base: dir.join("longlive_base.safetensors"),
            lora: lora.is_file().then_some(lora),
            lora_scale: cfg.lora_scale(),
            bf16: true,
        }
    }
}

/// What [`build_transformer_tensors`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MergeReport {
    pub tensors: usize,
    /// Modules a LoRA pair was merged into.
    pub merged: usize,
    /// Keys dropped (`freqs`).
    pub skipped: Vec<String>,
}

/// Source tensors by name: `(shape, dtype, little-endian bytes)`.
pub trait TensorSource {
    fn keys(&self) -> Vec<String>;
    fn get(&self, key: &str) -> Result<(Vec<usize>, LazyDType, Vec<u8>)>;
}

impl TensorSource for LazyStore {
    fn keys(&self) -> Vec<String> {
        LazyStore::keys(self).map(str::to_string).collect()
    }
    fn get(&self, key: &str) -> Result<(Vec<usize>, LazyDType, Vec<u8>)> {
        let v = self.view(key).map_err(|e| msg(e.to_string()))?;
        Ok((v.shape.to_vec(), v.dtype.clone(), v.bytes.to_vec()))
    }
}

/// In-memory source (tests).
impl TensorSource for HashMap<String, (Vec<usize>, Vec<f32>)> {
    fn keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self.keys().cloned().collect();
        k.sort();
        k
    }
    fn get(&self, key: &str) -> Result<(Vec<usize>, LazyDType, Vec<u8>)> {
        let (s, v) = HashMap::get(self, key).ok_or_else(|| msg(format!("missing {key}")))?;
        Ok((
            s.clone(),
            LazyDType::F32,
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ))
    }
}

fn to_f32(dtype: &LazyDType, bytes: &[u8], key: &str) -> Result<Vec<f32>> {
    Ok(match dtype {
        LazyDType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        LazyDType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        LazyDType::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        other => {
            return Err(msg(format!(
                "longlive: {key} is {other:?}, not a float tensor"
            )))
        }
    })
}

fn raw_dtype(d: &LazyDType) -> Option<RawDType> {
    match d {
        LazyDType::F32 => Some(RawDType::F32),
        LazyDType::BF16 => Some(RawDType::BF16),
        LazyDType::F16 => Some(RawDType::F16),
        _ => None,
    }
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::bf16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

/// The Diffusers-named transformer tensors of a LongLive / Self-Forcing
/// generator `base`, with `lora` (a PEFT state dict) merged at `scale`.
/// Every LoRA pair must target a base weight, and every base key must have
/// a Diffusers name (or be a known buffer).
pub fn build_transformer_tensors(
    base: &dyn TensorSource,
    lora: Option<&dyn TensorSource>,
    scale: f32,
    bf16: bool,
) -> Result<(HashMap<String, RawTensor>, MergeReport)> {
    // LoRA pairs by the Diffusers name of their target weight.
    let mut pairs: HashMap<String, (Option<String>, Option<String>)> = HashMap::new();
    if let Some(l) = lora {
        for key in l.keys() {
            let lk = lora_key(&key)
                .ok_or_else(|| msg(format!("longlive lora: `{key}` is not lora_A/lora_B")))?;
            let target = diffusers_key(&format!("{}.weight", lk.module))?
                .ok_or_else(|| msg(format!("longlive lora: `{key}` targets a buffer")))?;
            let e = pairs.entry(target).or_default();
            let slot = if lk.is_a { &mut e.0 } else { &mut e.1 };
            if slot.replace(key.clone()).is_some() {
                return Err(msg(format!("longlive lora: `{key}` twice")));
            }
        }
    }
    let mut out = HashMap::new();
    let mut report = MergeReport::default();
    for key in base.keys() {
        let Some(name) = diffusers_key(&key)? else {
            report.skipped.push(key);
            continue;
        };
        let (shape, dtype, bytes) = base.get(&key)?;
        let pair = pairs.remove(&name);
        let small = shape.len() < 2 || name.ends_with("scale_shift_table");
        let t = match pair {
            Some((Some(ka), Some(kb))) => {
                let l = lora.expect("pairs come from the lora");
                let (sa, da, ba) = l.get(&ka)?;
                let (sb, db, bb) = l.get(&kb)?;
                if shape.len() != 2 || sa.len() != 2 || sb.len() != 2 || sa[0] != sb[1] {
                    return Err(msg(format!(
                        "longlive lora: {name} {shape:?} with A {sa:?} B {sb:?}"
                    )));
                }
                let mut w = to_f32(&dtype, &bytes, &key)?;
                let (a, b) = (to_f32(&da, &ba, &ka)?, to_f32(&db, &bb, &kb)?);
                merge_lora(&mut w, shape[0], shape[1], &a, &b, sa[0], scale)?;
                report.merged += 1;
                if bf16 {
                    RawTensor::from_le_bytes(shape, RawDType::BF16, bf16_bytes(&w))
                } else {
                    RawTensor::from_f32(shape, w)
                }
            }
            Some(_) => {
                return Err(msg(format!(
                    "longlive lora: {name} has only one of lora_A / lora_B"
                )))
            }
            None if bf16 && !small && dtype != LazyDType::BF16 => {
                let w = to_f32(&dtype, &bytes, &key)?;
                RawTensor::from_le_bytes(shape, RawDType::BF16, bf16_bytes(&w))
            }
            None => {
                let d = raw_dtype(&dtype)
                    .ok_or_else(|| msg(format!("longlive: {key} is {dtype:?}")))?;
                RawTensor::from_le_bytes(shape, d, bytes)
            }
        };
        if out.insert(name.clone(), t).is_some() {
            return Err(msg(format!("longlive: two keys map to {name}")));
        }
    }
    if let Some(name) = pairs.keys().next() {
        return Err(msg(format!(
            "longlive lora: no base weight {name} ({} unmatched)",
            pairs.len()
        )));
    }
    report.tensors = out.len();
    Ok((out, report))
}

/// The LongLive transformer as a Diffusers-named [`WeightMap`] (for
/// [`super::pipeline::WanPipeline::load_with_dit`]): `base` renamed, `lora`
/// merged.
pub fn load_transformer_map(w: &LongLiveWeights) -> Result<(WeightMap, MergeReport)> {
    let base = LazyStore::open_files(&[w.base.clone()])
        .map_err(|e| msg(format!("{}: {e}", w.base.display())))?;
    let lora = match &w.lora {
        Some(p) => Some(
            LazyStore::open_files(&[p.clone()])
                .map_err(|e| msg(format!("{}: {e}", p.display())))?,
        ),
        None => None,
    };
    let (tensors, report) = build_transformer_tensors(
        &base,
        lora.as_ref().map(|l| l as &dyn TensorSource),
        w.lora_scale,
        w.bf16,
    )?;
    super::log::info(format_args!(
        "longlive transformer: {} tensors, {} LoRA modules merged (scale {}), skipped {:?}",
        report.tensors, report.merged, w.lora_scale, report.skipped
    ));
    Ok((WeightMap::from_raw_tensors(tensors), report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configs_match_the_yaml() {
        let c = LongLiveConfig::interactive();
        assert_eq!(c.denoising_steps, [1000, 750, 500, 250]);
        assert_eq!(
            (c.frames_per_block, c.local_attn_frames, c.sink_frames),
            (3, 12, 3)
        );
        assert_eq!(
            (c.timestep_shift, c.context_noise, c.global_sink),
            (5.0, 0, true)
        );
        assert_eq!(c.lora_scale(), 1.0);
        assert_eq!(c.rope(false), RopePolicy::Absolute);
        let i = LongLiveConfig::infinity();
        assert_eq!(i.rope(false), RopePolicy::RebasedSink);
        assert_eq!(i.rope(true), RopePolicy::Relativistic);
        let r = c.rollout(RolloutConfig::default());
        assert_eq!(
            (r.local_attn_frames, r.sink_frames, r.flow_shift),
            (12, 3, 5.0)
        );
        assert_eq!(r.prompt_switch, PromptSwitch::Recache { global_sink: true });
        assert!(r.recache.is_none());
        let off = LongLiveConfig {
            recache: false,
            ..c
        };
        assert_eq!(off.prompt_switch(), PromptSwitch::Keep);
    }

    /// `_recache_after_switch` with `local_attn_size = 12`, `sink_size = 3`,
    /// 3-frame blocks: the cache fills over blocks 0..3, rolls from block 4.
    #[test]
    fn recache_plans_follow_longlive() {
        assert_eq!(recache_plan(0, 12, true), None);
        // Before the window is full the whole history is recomputed from
        // frame 0, sink included (current_start == 0: no is_recompute).
        for cur in [3, 6, 9, 12] {
            let p = recache_plan(cur, 12, true).unwrap();
            assert_eq!(
                (p.start_frame, p.frames, p.guard_sink),
                (0, cur, false),
                "{cur}"
            );
            assert_eq!(recached_slots(&p, 3), 0..cur);
        }
        // Rolled: the last 12 frames, the sink slots kept.
        let p = recache_plan(40, 12, true).unwrap();
        assert_eq!((p.start_frame, p.frames, p.guard_sink), (28, 12, true));
        assert_eq!(recached_slots(&p, 3), 3..12);
        assert!(!p.rewrites_sink());
        // global_sink: false rewrites every slot.
        let p = recache_plan(40, 12, false).unwrap();
        assert_eq!((p.start_frame, p.guard_sink), (28, false));
        assert_eq!(recached_slots(&p, 3), 0..12);
        assert!(p.rewrites_sink());
        // A window of -1 in LongLive is everything; ours is always bounded.
        assert_eq!(recache_plan(5, 21, true).unwrap().frames, 5);
    }

    #[test]
    fn original_wan_keys_map_to_diffusers() {
        let cases = [
            ("model.patch_embedding.weight", "patch_embedding.weight"),
            (
                "model.text_embedding.0.bias",
                "condition_embedder.text_embedder.linear_1.bias",
            ),
            (
                "model.text_embedding.2.weight",
                "condition_embedder.text_embedder.linear_2.weight",
            ),
            (
                "model.time_embedding.0.weight",
                "condition_embedder.time_embedder.linear_1.weight",
            ),
            (
                "model.time_embedding.2.bias",
                "condition_embedder.time_embedder.linear_2.bias",
            ),
            (
                "model.time_projection.1.weight",
                "condition_embedder.time_proj.weight",
            ),
            ("model.head.head.bias", "proj_out.bias"),
            ("model.head.modulation", "scale_shift_table"),
            ("model.blocks.7.modulation", "blocks.7.scale_shift_table"),
            (
                "model.blocks.0.self_attn.q.weight",
                "blocks.0.attn1.to_q.weight",
            ),
            (
                "model.blocks.0.self_attn.o.bias",
                "blocks.0.attn1.to_out.0.bias",
            ),
            (
                "model.blocks.0.self_attn.norm_k.weight",
                "blocks.0.attn1.norm_k.weight",
            ),
            (
                "model.blocks.29.cross_attn.v.weight",
                "blocks.29.attn2.to_v.weight",
            ),
            (
                "model.blocks.29.cross_attn.norm_q.weight",
                "blocks.29.attn2.norm_q.weight",
            ),
            ("model.blocks.3.norm3.weight", "blocks.3.norm2.weight"),
            (
                "model.blocks.3.ffn.0.weight",
                "blocks.3.ffn.net.0.proj.weight",
            ),
            ("model.blocks.3.ffn.2.bias", "blocks.3.ffn.net.2.bias"),
            // FSDP (EMA) and PEFT wrappers.
            (
                "model._fsdp_wrapped_module.blocks.1.self_attn.k.weight",
                "blocks.1.attn1.to_k.weight",
            ),
            (
                "base_model.model.blocks.1.cross_attn.o.weight",
                "blocks.1.attn2.to_out.0.weight",
            ),
        ];
        for (from, to) in cases {
            assert_eq!(diffusers_key(from).unwrap().as_deref(), Some(to), "{from}");
        }
        assert_eq!(diffusers_key("model.freqs").unwrap(), None);
        assert!(diffusers_key("model.blocks.0.self_attn.zz.weight").is_err());
        assert!(diffusers_key("model.img_emb.proj.0.weight").is_err());
    }

    #[test]
    fn lora_keys_parse() {
        assert_eq!(
            lora_key("base_model.model.blocks.0.self_attn.q.lora_A.weight"),
            Some(LoraKey {
                module: "blocks.0.self_attn.q".into(),
                is_a: true
            })
        );
        assert_eq!(
            lora_key("base_model.model.blocks.2.ffn.2.lora_B.default.weight"),
            Some(LoraKey {
                module: "blocks.2.ffn.2".into(),
                is_a: false
            })
        );
        assert_eq!(
            lora_key("base_model.model.blocks.0.self_attn.q.base_layer.weight"),
            None
        );
    }

    #[test]
    fn merge_is_w_plus_scaled_b_at_a() {
        // w 2x3, rank 2.
        let mut w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = vec![1.0, 0.0, 2.0, 0.0, 1.0, -1.0]; // [2, 3]
        let b = vec![1.0, 2.0, -1.0, 0.5]; // [2, 2]
        merge_lora(&mut w, 2, 3, &a, &b, 2, 0.5).unwrap();
        // B@A = [[1, 2, 0], [-1, 0.5, -2.5]]
        let want = [1.5, 3.0, 3.0, 3.5, 5.25, 4.75];
        for (x, y) in w.iter().zip(want) {
            assert!((x - y).abs() < 1e-6, "{w:?}");
        }
        assert!(merge_lora(&mut w, 3, 2, &a, &b, 2, 1.0).is_err());
    }

    fn src(items: &[(&str, Vec<usize>, Vec<f32>)]) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
        items
            .iter()
            .map(|(k, s, v)| (k.to_string(), (s.clone(), v.clone())))
            .collect()
    }

    #[test]
    fn transformer_tensors_rename_and_merge() {
        let base = src(&[
            (
                "model.blocks.0.self_attn.q.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            ("model.blocks.0.self_attn.q.bias", vec![2], vec![0.5, -0.5]),
            ("model.blocks.0.modulation", vec![1, 6, 2], vec![0.25; 12]),
            ("model.head.head.weight", vec![2, 2], vec![2.0; 4]),
            ("model.freqs", vec![4], vec![0.0; 4]),
        ]);
        let lora = src(&[
            (
                "base_model.model.blocks.0.self_attn.q.lora_A.weight",
                vec![1, 2],
                vec![1.0, 1.0],
            ),
            (
                "base_model.model.blocks.0.self_attn.q.lora_B.weight",
                vec![2, 1],
                vec![1.0, 2.0],
            ),
        ]);
        let (t, r) = build_transformer_tensors(&base, Some(&lora), 1.0, false).unwrap();
        assert_eq!((r.tensors, r.merged), (4, 1));
        assert_eq!(r.skipped, ["model.freqs"]);
        let q = t["blocks.0.attn1.to_q.weight"].to_f32_vec().unwrap();
        assert_eq!(q, [2.0, 1.0, 2.0, 3.0]);
        assert_eq!(
            t["blocks.0.attn1.to_q.bias"].to_f32_vec().unwrap(),
            [0.5, -0.5]
        );
        assert_eq!(t["blocks.0.scale_shift_table"].shape, [1, 6, 2]);
        // bf16: matrices narrowed, the modulation table and biases kept.
        let (t, _) = build_transformer_tensors(&base, Some(&lora), 1.0, true).unwrap();
        assert_eq!(t["blocks.0.attn1.to_q.weight"].dtype, RawDType::BF16);
        assert_eq!(t["proj_out.weight"].dtype, RawDType::BF16);
        assert_eq!(t["blocks.0.scale_shift_table"].dtype, RawDType::F32);
        assert_eq!(t["blocks.0.attn1.to_q.bias"].dtype, RawDType::F32);
        assert_eq!(
            t["blocks.0.attn1.to_q.weight"].to_f32_vec().unwrap(),
            [2.0, 1.0, 2.0, 3.0]
        );
        // Without the LoRA: the base weights.
        let (t, r) = build_transformer_tensors(&base, None, 1.0, false).unwrap();
        assert_eq!(r.merged, 0);
        assert_eq!(
            t["blocks.0.attn1.to_q.weight"].to_f32_vec().unwrap(),
            [1.0, 0.0, 0.0, 1.0]
        );
        // A LoRA pair without its base weight, or half a pair, is refused.
        let stray = src(&[
            (
                "base_model.model.blocks.9.self_attn.q.lora_A.weight",
                vec![1, 2],
                vec![1.0, 1.0],
            ),
            (
                "base_model.model.blocks.9.self_attn.q.lora_B.weight",
                vec![2, 1],
                vec![1.0, 2.0],
            ),
        ]);
        assert!(build_transformer_tensors(&base, Some(&stray), 1.0, false).is_err());
        let half = src(&[(
            "base_model.model.blocks.0.self_attn.q.lora_A.weight",
            vec![1, 2],
            vec![1.0, 1.0],
        )]);
        assert!(build_transformer_tensors(&base, Some(&half), 1.0, false).is_err());
    }

    /// A merged map loads into the Wan DiT: a tiny generator written with
    /// original names, renamed and merged, against the same weights loaded
    /// by their Diffusers names.
    #[test]
    fn renamed_map_loads_the_tiny_dit() {
        use fastvideo_models::wan::WanVideoArchConfig;
        let mut cfg = WanVideoArchConfig::tiny();
        cfg.causal = true;
        let gen = |key: &str, shape: &[usize]| -> Vec<f32> {
            let n: usize = shape.iter().product();
            let seed = key
                .bytes()
                .fold(7u64, |a, b| a.wrapping_mul(131).wrapping_add(u64::from(b)));
            (0..n)
                .map(|i| {
                    ((seed
                        .wrapping_add(i as u64)
                        .wrapping_mul(6364136223846793005)
                        >> 35)
                        % 2001) as f32
                        / 1000.0
                        - 1.0
                })
                .map(|x| x * 0.3)
                .collect()
        };
        // Record every key and shape the loader asks for (Diffusers names).
        let asked = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(String, Vec<usize>)>::new()));
        let rec = asked.clone();
        let probe = WeightMap::generated(move |k, s| {
            rec.lock().unwrap().push((k.to_string(), s.to_vec()));
            gen(k, s)
        });
        super::super::transformer::WanTransformer3D::load(cfg.clone(), &probe).unwrap();
        // Build the original-named source from the inverse of diffusers_key
        // (`None`: an optional Diffusers-only key the generated map offered,
        // e.g. the VSA gate; neither side gets it).
        let inverse = |d: &str| -> Option<String> {
            let orig = [
                "patch_embedding.weight",
                "patch_embedding.bias",
                "text_embedding.0.weight",
                "text_embedding.0.bias",
                "text_embedding.2.weight",
                "text_embedding.2.bias",
                "time_embedding.0.weight",
                "time_embedding.0.bias",
                "time_embedding.2.weight",
                "time_embedding.2.bias",
                "time_projection.1.weight",
                "time_projection.1.bias",
                "head.head.weight",
                "head.head.bias",
                "head.modulation",
            ];
            for o in orig {
                if diffusers_key(o).unwrap().as_deref() == Some(d) {
                    return Some(format!("model.{o}"));
                }
            }
            let blk = [
                "modulation",
                "norm3.weight",
                "norm3.bias",
                "ffn.0.weight",
                "ffn.0.bias",
                "ffn.2.weight",
                "ffn.2.bias",
            ];
            let (i, _) = d.strip_prefix("blocks.").unwrap().split_once('.').unwrap();
            for o in blk.iter().map(|s| s.to_string()).chain(
                ["self_attn", "cross_attn"].iter().flat_map(|a| {
                    ["q", "k", "v", "o", "norm_q", "norm_k"]
                        .iter()
                        .flat_map(move |p| ["weight", "bias"].map(|w| format!("{a}.{p}.{w}")))
                }),
            ) {
                let full = format!("blocks.{i}.{o}");
                if diffusers_key(&full).unwrap().as_deref() == Some(d) {
                    return Some(format!("model.{full}"));
                }
            }
            None
        };
        let asked = asked.lock().unwrap().clone();
        let (mut base, mut named) = (HashMap::new(), Vec::new());
        for (k, s) in &asked {
            if let Some(o) = inverse(k) {
                base.insert(o, (s.clone(), gen(k, s)));
                named.push((k.clone(), s.clone(), gen(k, s)));
            } else {
                assert!(k.contains("gate"), "{k} has no original name");
            }
        }
        assert!(named.len() > 20, "{} keys", named.len());
        let reference = super::super::transformer::WanTransformer3D::load(
            cfg.clone(),
            &WeightMap::from_f32_tensors(named.clone()),
        )
        .unwrap();
        let (tensors, r) = build_transformer_tensors(&base, None, 1.0, false).unwrap();
        assert_eq!(r.tensors, named.len());
        let loaded = super::super::transformer::WanTransformer3D::load(
            cfg.clone(),
            &WeightMap::from_raw_tensors(tensors),
        )
        .unwrap();
        // Same weights: the same output.
        let (c, t, h, w) = (cfg.in_channels, 3usize, 4usize, 6usize);
        let lat = super::super::tensor::CudaTensor::from_vec(
            (0..c * t * h * w)
                .map(|i| ((i as f32) * 0.37).sin())
                .collect(),
            vec![1, c, t, h, w],
        )
        .unwrap();
        let text = super::super::tensor::CudaTensor::from_vec(
            (0..cfg.text_len * cfg.text_dim)
                .map(|i| ((i as f32) * 0.11).cos())
                .collect(),
            vec![1, cfg.text_len, cfg.text_dim],
        )
        .unwrap();
        let ts = super::super::tensor::CudaTensor::from_vec(vec![500.0], vec![1]).unwrap();
        let a = reference.forward_ctx(&lat, &ts, &text, None).unwrap();
        let b = loaded.forward_ctx(&lat, &ts, &text, None).unwrap();
        assert_eq!(
            a.host_cow().unwrap().as_ref(),
            b.host_cow().unwrap().as_ref()
        );
    }
}
