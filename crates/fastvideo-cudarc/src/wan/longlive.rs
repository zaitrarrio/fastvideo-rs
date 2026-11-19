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
//!
//! **History Forcing** (ziplab, arXiv 2611.02782; Hub
//! `xzxxntxdy/History-Forcing` `dmd/model.pt`) runs on the same engine
//! through [`LongLiveConfig::history_forcing`]: the same architecture and
//! recipe (Wan2.1-T2V-1.3B, 4 DMD steps, 3-frame blocks, shift 5), the
//! generator trained with `local_attn_size 21, sink_size 3`, no LoRA and no
//! prompt-switch re-cache. Its checkpoint is a training `.pt` (`generator`
//! and `generator_ema`, f32, FSDP-wrapped original Wan names) that
//! [`LongLiveWeights`] reads directly ([`PthSource`]) or through the
//! torch-free `scripts/gpu/convert-history-forcing.py`
//! (`generator_ema.safetensors`, same keys). docs/ports/wan.md "History
//! Forcing".

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use fastvideo_loader::pth::{read_pth_nested, PthTensor};
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

    /// History Forcing's long-video generator (`github.com/ziplab/History-Forcing`
    /// `configs/long_video/dmd_ar10s_bi1p3b.yaml`, the config of the released
    /// `dmd/model.pt`, over `configs/long_video/default_config.yaml`; read
    /// 2026-11-18): `denoising_step_list: [1000, 750, 500, 250]`,
    /// `warp_denoising_step: true`, `timestep_shift: 5.0`,
    /// `num_frame_per_block: 3`, `model_kwargs.local_attn_size: 21`,
    /// `model_kwargs.sink_size: 3`, `context_noise: 0` (the default config).
    /// The generator is a full checkpoint (`generator_ema`, `inference.yaml`
    /// `weights: ema`): no LoRA. Its inference (`inference_memrope.py`,
    /// `CausalInferencePipeline`) rolls one prompt out with the sink kept for
    /// the whole run (`global_sink`) and has no prompt-switch re-cache
    /// (`recache: false`: our [`PromptSwitch::Keep`]; a family file may still
    /// set `recache = true` to borrow LongLive's). Absolute RoPE, as the
    /// pipeline ropes keys at their frame. MemRoPE (`inference.yaml`'s
    /// `memrope:` table, the optional demo cache; vendored code CC-BY-NC-SA)
    /// is not implemented: this is the plain 21 / 3 rolling cache the
    /// generator was trained with (docs/ports/wan.md "History Forcing").
    pub fn history_forcing() -> Self {
        Self {
            denoising_steps: vec![1000, 750, 500, 250],
            frames_per_block: 3,
            local_attn_frames: 21,
            sink_frames: 3,
            timestep_shift: 5.0,
            global_sink: true,
            context_noise: 0,
            lora_rank: 0,
            lora_alpha: 0.0,
            relative_rope: false,
            recache: false,
        }
    }

    /// Whether the checkpoint comes with a LoRA to merge (`lora.safetensors`):
    /// LongLive yes, History Forcing no.
    pub fn has_lora(&self) -> bool {
        self.lora_rank > 0
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

/// Where a LongLive / History Forcing generator lives: `base` is a
/// safetensors file with original Wan names (the fetch's conversion of
/// `models/longlive_base.pt`, `convert-history-forcing.py`'s
/// `generator_ema.safetensors`) or a torch `.pt` training checkpoint read
/// directly ([`PthSource`]); `lora` is LongLive's `lora.safetensors`
/// (`models/lora.pt`'s `generator_lora` sub-dict, same keys).
#[derive(Debug, Clone, PartialEq)]
pub struct LongLiveWeights {
    pub base: PathBuf,
    /// `None`: the base generator alone (LongLive's `longlive_init` without
    /// the long tuning, not what LongLive serves; History Forcing, which has
    /// no LoRA).
    pub lora: Option<PathBuf>,
    /// LoRA scale (`alpha / rank`, 1.0 for the release).
    pub lora_scale: f32,
    /// Keep 2-D+ weights in bf16 (the DiT runs bf16 weights anyway; halves
    /// host memory). 1-D tensors and the modulation tables keep their dtype.
    pub bf16: bool,
}

/// The generator files [`LongLiveWeights::in_dir`] looks for, in order:
/// LongLive's conversion, History Forcing's conversion (EMA, then raw), a
/// torch checkpoint at the root, the Hub's `dmd/model.pt` layout.
pub const BASE_FILES: [&str; 5] = [
    "longlive_base.safetensors",
    "generator_ema.safetensors",
    "generator.safetensors",
    "model.pt",
    "dmd/model.pt",
];

/// A torch `.pt` / `.pth` checkpoint (read with [`PthSource`]) rather than
/// safetensors.
pub fn is_torch_checkpoint(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("pt") | Some("pth")
    )
}

impl LongLiveWeights {
    /// The generator in `dir`: the first of [`BASE_FILES`] that exists
    /// (`longlive_base.safetensors` when none does, so the load names it),
    /// and `<dir>/lora.safetensors` when `cfg` has a LoRA
    /// ([`LongLiveConfig::has_lora`]) and the file exists. A LongLive dir
    /// resolves exactly as before History Forcing joined.
    pub fn in_dir(dir: &Path, cfg: &LongLiveConfig) -> Self {
        let base = BASE_FILES
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.is_file())
            .unwrap_or_else(|| dir.join(BASE_FILES[0]));
        Self::with_lora_beside(base, dir, cfg)
    }

    /// A bare generator file (`.pt`, `.pth` or safetensors); a LoRA, when
    /// `cfg` has one, is `lora.safetensors` beside it.
    pub fn from_file(path: &Path, cfg: &LongLiveConfig) -> Self {
        let dir = path.parent().unwrap_or(Path::new("."));
        Self::with_lora_beside(path.to_path_buf(), dir, cfg)
    }

    /// [`Self::from_file`] for a file, [`Self::in_dir`] for a directory (or
    /// anything else, so a missing dir fails at the load with its name).
    pub fn at(path: &Path, cfg: &LongLiveConfig) -> Self {
        if path.is_file() {
            Self::from_file(path, cfg)
        } else {
            Self::in_dir(path, cfg)
        }
    }

    fn with_lora_beside(base: PathBuf, dir: &Path, cfg: &LongLiveConfig) -> Self {
        let lora = dir.join("lora.safetensors");
        Self {
            base,
            lora: (cfg.has_lora() && lora.is_file()).then_some(lora),
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

/// A generator read whole from a torch `.pt` / `.pth` checkpoint
/// (`fastvideo_loader::pth`): the root's tensors, else `generator_ema`,
/// else `generator` (History Forcing's `dmd/model.pt`, Self-Forcing's
/// `self_forcing_dmd.pt`, LongLive's `longlive_base.pt`). Tensors come out
/// as f32 (bf16 / f16 storages widened), so a 1.3B generator holds 5.7 GB of
/// host memory until [`build_transformer_tensors`] has narrowed it; the
/// keys keep their wrappers (`model._fsdp_wrapped_module.`), which
/// [`diffusers_key`] strips.
#[derive(Debug)]
pub struct PthSource {
    tensors: BTreeMap<String, PthTensor>,
}

/// The sub-dicts a generator checkpoint keeps its state dict under, in
/// order of preference (the EMA weights are what the released inference
/// runs: History Forcing `inference.yaml` `weights: ema`).
pub const PTH_GENERATOR_KEYS: [&str; 2] = ["generator_ema", "generator"];

impl PthSource {
    pub fn open(path: &Path) -> Result<Self> {
        let tensors = read_pth_nested(path, &PTH_GENERATOR_KEYS, |_| true)
            .map_err(|e| msg(format!("longlive: {e}")))?;
        if tensors.is_empty() {
            return Err(msg(format!(
                "longlive: {} holds no tensors",
                path.display()
            )));
        }
        Ok(Self { tensors })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl TensorSource for PthSource {
    fn keys(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }
    fn get(&self, key: &str) -> Result<(Vec<usize>, LazyDType, Vec<u8>)> {
        let t = self
            .tensors
            .get(key)
            .ok_or_else(|| msg(format!("missing {key}")))?;
        Ok((
            t.shape.clone(),
            LazyDType::F32,
            t.data.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ))
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        LazyDType::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        LazyDType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
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

/// `base` opened by its kind: a torch checkpoint whole ([`PthSource`]), a
/// safetensors file lazily.
fn open_base(base: &Path) -> Result<Box<dyn TensorSource>> {
    if is_torch_checkpoint(base) {
        return Ok(Box::new(PthSource::open(base)?));
    }
    Ok(Box::new(
        LazyStore::open_files(&[base.to_path_buf()])
            .map_err(|e| msg(format!("{}: {e}", base.display())))?,
    ))
}

/// The LongLive / History Forcing transformer as a Diffusers-named
/// [`WeightMap`] (for [`super::pipeline::WanPipeline::load_with_dit`]):
/// `base` renamed (safetensors or a torch `.pt`), `lora` merged if any.
pub fn load_transformer_map(w: &LongLiveWeights) -> Result<(WeightMap, MergeReport)> {
    let base = open_base(&w.base)?;
    let lora = match &w.lora {
        Some(p) => Some(
            LazyStore::open_files(std::slice::from_ref(p))
                .map_err(|e| msg(format!("{}: {e}", p.display())))?,
        ),
        None => None,
    };
    let (tensors, report) = build_transformer_tensors(
        base.as_ref(),
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

    /// `configs/long_video/dmd_ar10s_bi1p3b.yaml` over `default_config.yaml`
    /// (ziplab/History-Forcing, 2026-11-18): window 21, sink 3, no LoRA, no
    /// switch re-cache, the Self-Forcing schedule.
    #[test]
    fn history_forcing_matches_its_config() {
        let h = LongLiveConfig::history_forcing();
        assert_eq!(h.denoising_steps, [1000, 750, 500, 250]);
        assert_eq!(
            (h.frames_per_block, h.local_attn_frames, h.sink_frames),
            (3, 21, 3)
        );
        assert_eq!(
            (h.timestep_shift, h.context_noise, h.global_sink),
            (5.0, 0, true)
        );
        assert!(!h.has_lora());
        assert_eq!((h.lora_rank, h.lora_alpha, h.lora_scale()), (0, 0.0, 0.0));
        assert!(!h.relative_rope && !h.recache);
        assert_eq!(h.rope(false), RopePolicy::Absolute);
        assert_eq!(h.prompt_switch(), PromptSwitch::Keep);
        let r = h.rollout(RolloutConfig::default());
        assert_eq!(
            (r.local_attn_frames, r.sink_frames, r.flow_shift),
            (21, 3, 5.0)
        );
        assert_eq!(r.dmd_steps, [1000, 750, 500, 250]);
        assert_eq!(
            (r.rope, r.prompt_switch),
            (RopePolicy::Absolute, PromptSwitch::Keep)
        );
        assert!(r.recache.is_none());
        // A family file may borrow LongLive's switch re-cache.
        let with = LongLiveConfig { recache: true, ..h };
        assert_eq!(
            with.prompt_switch(),
            PromptSwitch::Recache { global_sink: true }
        );
        assert!(LongLiveConfig::interactive().has_lora());
        assert!(LongLiveConfig::infinity().has_lora());
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fv-longlive-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// `in_dir` takes the first generator file of `BASE_FILES` and the LoRA
    /// only under a config that has one; `from_file` / `at` take a bare
    /// path. A LongLive dir resolves as it always did.
    #[test]
    fn weights_in_dir_pick_the_generator_and_the_lora_by_config() {
        let ll = LongLiveConfig::interactive();
        let hf = LongLiveConfig::history_forcing();
        let d = scratch("in-dir");
        // Empty dir: LongLive's names (the load names the missing file).
        let w = LongLiveWeights::in_dir(&d, &ll);
        assert_eq!(w.base, d.join("longlive_base.safetensors"));
        assert_eq!((w.lora, w.lora_scale, w.bf16), (None, 1.0, true));
        assert_eq!(
            LongLiveWeights::in_dir(&d, &hf).base,
            d.join("longlive_base.safetensors")
        );
        // History Forcing's conversion: the EMA generator first; no LoRA is
        // looked for, even with a lora.safetensors in the dir.
        std::fs::write(d.join("generator.safetensors"), b"").unwrap();
        std::fs::write(d.join("lora.safetensors"), b"").unwrap();
        let w = LongLiveWeights::in_dir(&d, &hf);
        assert_eq!(w.base, d.join("generator.safetensors"));
        assert_eq!((w.lora, w.lora_scale), (None, 0.0));
        std::fs::write(d.join("generator_ema.safetensors"), b"").unwrap();
        assert_eq!(
            LongLiveWeights::in_dir(&d, &hf).base,
            d.join("generator_ema.safetensors")
        );
        // The same dir under the LongLive config takes the LoRA (rank 256);
        // longlive_base.safetensors wins once it is there.
        let w = LongLiveWeights::in_dir(&d, &ll);
        assert_eq!(w.base, d.join("generator_ema.safetensors"));
        assert_eq!(w.lora, Some(d.join("lora.safetensors")));
        std::fs::write(d.join("longlive_base.safetensors"), b"").unwrap();
        let w = LongLiveWeights::in_dir(&d, &ll);
        assert_eq!(w.base, d.join("longlive_base.safetensors"));
        assert_eq!(w.lora, Some(d.join("lora.safetensors")));
        // The Hub layout of History Forcing (dmd/model.pt), and bare files.
        let h = scratch("hub");
        std::fs::create_dir_all(h.join("dmd")).unwrap();
        std::fs::write(h.join("dmd").join("model.pt"), b"").unwrap();
        let pt = h.join("dmd").join("model.pt");
        let w = LongLiveWeights::in_dir(&h, &hf);
        assert_eq!(w.base, pt);
        assert!(is_torch_checkpoint(&w.base));
        assert!(!is_torch_checkpoint(&d.join("generator_ema.safetensors")));
        let f = LongLiveWeights::from_file(&pt, &hf);
        assert_eq!((f.base.clone(), f.lora.clone()), (pt.clone(), None));
        assert_eq!(LongLiveWeights::at(&pt, &hf), f);
        assert_eq!(LongLiveWeights::at(&h, &hf), f);
        assert_eq!(
            LongLiveWeights::at(&d, &ll),
            LongLiveWeights::in_dir(&d, &ll)
        );
        // A LongLive dir handed as a file path: the LoRA beside it.
        let f = LongLiveWeights::from_file(&d.join("longlive_base.safetensors"), &ll);
        assert_eq!(f.lora, Some(d.join("lora.safetensors")));
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&h);
    }

    /// A stored zip of `files`, as torch writes a checkpoint.
    fn torch_zip(files: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let (mut out, mut cd) = (Vec::new(), Vec::new());
        for (name, data) in files {
            let off = out.len() as u32;
            out.extend([
                0x50, 0x4b, 0x03, 0x04, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]);
            out.extend((data.len() as u32).to_le_bytes());
            out.extend((data.len() as u32).to_le_bytes());
            out.extend((name.len() as u16).to_le_bytes());
            out.extend(0u16.to_le_bytes());
            out.extend(name.as_bytes());
            out.extend(data);
            cd.extend([
                0x50, 0x4b, 0x01, 0x02, 20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]);
            cd.extend((data.len() as u32).to_le_bytes());
            cd.extend((data.len() as u32).to_le_bytes());
            cd.extend((name.len() as u16).to_le_bytes());
            cd.extend([0u8; 12]);
            cd.extend(off.to_le_bytes());
            cd.extend(name.as_bytes());
        }
        let cd_off = out.len() as u32;
        out.extend(&cd);
        out.extend([0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0]);
        out.extend((files.len() as u16).to_le_bytes());
        out.extend((files.len() as u16).to_le_bytes());
        out.extend((cd.len() as u32).to_le_bytes());
        out.extend(cd_off.to_le_bytes());
        out.extend([0u8; 2]);
        out
    }

    fn pickle_str(s: &str) -> Vec<u8> {
        let mut p = vec![b'X'];
        p.extend((s.len() as u32).to_le_bytes());
        p.extend(s.as_bytes());
        p
    }

    /// A tensor as History Forcing's checkpoint saves it:
    /// `_rebuild_from_type_v2(_rebuild_tensor_v2, torch.Tensor,
    /// ((storage, FloatStorage, key, cpu, n), 0, shape, stride, False,
    /// OrderedDict()), {})`, row-major and contiguous.
    fn pickle_f32_tensor(key: &str, shape: &[u8]) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend(b"ctorch._tensor\n_rebuild_from_type_v2\n(");
        p.extend(b"ctorch._utils\n_rebuild_tensor_v2\nctorch\nTensor\n(");
        p.extend(b"(X\x07\x00\x00\x00storagectorch\nFloatStorage\n");
        p.extend(pickle_str(key));
        p.extend(b"X\x03\x00\x00\x00cpu");
        p.extend([b'K', shape.iter().product::<u8>(), b't', b'Q', b'K', 0]);
        let tuple = |items: &[u8], p: &mut Vec<u8>| {
            for &i in items {
                p.extend([b'K', i]);
            }
            p.push(match items.len() {
                1 => 0x85,
                2 => 0x86,
                3 => 0x87,
                _ => unreachable!(),
            });
        };
        tuple(shape, &mut p);
        let mut stride = vec![1u8; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        tuple(&stride, &mut p);
        p.extend(b"\x89ccollections\nOrderedDict\n)Rt}tR");
        p
    }

    /// `PthSource` reads the EMA generator of a training checkpoint (both
    /// sub-dicts present, FSDP wrappers on the keys) and feeds the same
    /// rename / narrowing as a safetensors source.
    #[test]
    fn pth_source_reads_the_ema_generator() {
        let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        // {"generator": {...raw...}, "generator_ema": {...ema...}}
        let mut p: Vec<u8> = vec![0x80, 2, b'}', b'('];
        p.extend(pickle_str("generator"));
        p.extend(b"}(");
        p.extend(pickle_str("model.blocks.0.self_attn.q.weight"));
        p.extend(pickle_f32_tensor("0", &[2, 2]));
        p.extend(pickle_str("model.head.modulation"));
        p.extend(pickle_f32_tensor("1", &[1, 2, 2]));
        p.extend(b"u");
        p.extend(pickle_str("generator_ema"));
        p.extend(b"}(");
        p.extend(pickle_str(
            "model._fsdp_wrapped_module.blocks.0.self_attn.q.weight",
        ));
        p.extend(pickle_f32_tensor("2", &[2, 2]));
        p.extend(pickle_str("model._fsdp_wrapped_module.head.modulation"));
        p.extend(pickle_f32_tensor("3", &[1, 2, 2]));
        p.extend(b"uu.");
        let zip = torch_zip(&[
            ("model/data.pkl", p),
            ("model/data/0", f32s(&[9.0, 9.0, 9.0, 9.0])),
            ("model/data/1", f32s(&[9.0, 9.0, 9.0, 9.0])),
            ("model/data/2", f32s(&[1.0, 0.5, -0.25, 2.0])),
            ("model/data/3", f32s(&[0.125, 0.25, 0.5, 1.0])),
        ]);
        let d = scratch("pth");
        std::fs::create_dir_all(d.join("dmd")).unwrap();
        let pt = d.join("dmd").join("model.pt");
        std::fs::write(&pt, zip).unwrap();
        let src = PthSource::open(&pt).unwrap();
        assert_eq!(src.len(), 2);
        assert_eq!(
            src.keys(),
            [
                "model._fsdp_wrapped_module.blocks.0.self_attn.q.weight",
                "model._fsdp_wrapped_module.head.modulation"
            ]
        );
        let (shape, dtype, bytes) = src
            .get("model._fsdp_wrapped_module.blocks.0.self_attn.q.weight")
            .unwrap();
        assert_eq!((shape, dtype), (vec![2, 2], LazyDType::F32));
        assert_eq!(
            to_f32(&LazyDType::F32, &bytes, "q").unwrap(),
            [1.0, 0.5, -0.25, 2.0]
        );
        // Through the loader: the EMA values, renamed, matrices in bf16.
        let w = LongLiveWeights::in_dir(&d, &LongLiveConfig::history_forcing());
        assert_eq!(w.base, pt);
        let (map, r) = load_transformer_map(&w).unwrap();
        assert_eq!((r.tensors, r.merged), (2, 0));
        assert!(r.skipped.is_empty());
        let base = open_base(&w.base).unwrap();
        let (t, _) = build_transformer_tensors(base.as_ref(), None, w.lora_scale, true).unwrap();
        assert_eq!(t["blocks.0.attn1.to_q.weight"].dtype, RawDType::BF16);
        assert_eq!(
            t["blocks.0.attn1.to_q.weight"].to_f32_vec().unwrap(),
            [1.0, 0.5, -0.25, 2.0]
        );
        assert_eq!(t["scale_shift_table"].dtype, RawDType::F32);
        assert_eq!(
            t["scale_shift_table"].to_f32_vec().unwrap(),
            [0.125, 0.25, 0.5, 1.0]
        );
        drop(map);
        // A file that is no checkpoint fails with its name.
        std::fs::write(d.join("model.pt"), b"not a checkpoint").unwrap();
        let e = PthSource::open(&d.join("model.pt"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("model.pt"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The real History Forcing checkpoint, `FV_HISTORY_FORCING_PT=<path to
    /// dmd/model.pt>` (11.35 GB on disk, about 6 GB of host memory while the
    /// EMA generator is read as f32):
    /// `cargo test -p fastvideo-cudarc --features cuda history_forcing_checkpoint_loads -- --ignored --nocapture`.
    /// Prints the tensor count, the dtypes, the time and the host high-water mark.
    #[test]
    #[ignore = "needs FV_HISTORY_FORCING_PT (the 11 GB checkpoint)"]
    fn history_forcing_checkpoint_loads() {
        let Some(p) = std::env::var_os("FV_HISTORY_FORCING_PT") else {
            eprintln!("FV_HISTORY_FORCING_PT unset: skipped");
            return;
        };
        let p = PathBuf::from(p);
        let hf = LongLiveConfig::history_forcing();
        let w = LongLiveWeights::at(&p, &hf);
        assert!(is_torch_checkpoint(&w.base), "{}", w.base.display());
        assert_eq!(w.lora, None);
        let t0 = std::time::Instant::now();
        let src = PthSource::open(&w.base).unwrap();
        let read_s = t0.elapsed().as_secs_f64();
        // Wan2.1-T2V-1.3B as Diffusers names it: 30 blocks x 27 tensors + 15
        // (patch, text / time embedders, time projection, head, its table).
        // The checkpoint keeps no `freqs` buffer.
        assert_eq!(src.len(), 825);
        let (t, r) = build_transformer_tensors(&src, None, w.lora_scale, w.bf16).unwrap();
        let build_s = t0.elapsed().as_secs_f64() - read_s;
        assert_eq!((r.tensors, r.merged), (825, 0));
        assert!(r.skipped.is_empty(), "{:?}", r.skipped);
        let mut dtypes: BTreeMap<String, usize> = BTreeMap::new();
        let mut bytes = 0usize;
        for v in t.values() {
            *dtypes.entry(format!("{:?}", v.dtype)).or_default() += 1;
            bytes += v.data.len();
        }
        assert_eq!(t["blocks.29.attn2.to_out.0.weight"].dtype, RawDType::BF16);
        assert_eq!(t["blocks.29.attn2.to_out.0.weight"].shape, [1536, 1536]);
        assert_eq!(t["scale_shift_table"].dtype, RawDType::F32);
        assert_eq!(t["scale_shift_table"].shape, [1, 2, 1536]);
        assert_eq!(t["proj_out.bias"].dtype, RawDType::F32);
        let hwm = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmHWM"))
                    .map(str::to_owned)
            });
        eprintln!(
            "history forcing {}: {} tensors (pth read {read_s:.1} s, rename + bf16 {build_s:.1} s), out dtypes {dtypes:?}, {bytes} bytes, {}",
            p.display(),
            r.tensors,
            hwm.unwrap_or_default()
        );
        let map = WeightMap::from_raw_tensors(t);
        drop(map);
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
