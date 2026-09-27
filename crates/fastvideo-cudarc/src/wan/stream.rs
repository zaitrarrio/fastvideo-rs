//! Open-ended causal SF-Wan: the Self-Forcing block loop of
//! [`super::pipeline`]'s `causal_dmd_denoise` turned into a generator that
//! runs until stopped (serve design §5.4, work package E6).
//!
//! Each [`CausalRollout::next_block`] denoises one block of
//! `num_frames_per_block` (3) latent frames through the per-layer KV cache
//! (every Self-Forcing timestep, then the clean-context pass at `t = 0` that
//! rewrites the block's cache slots), and decodes that block at once with
//! TAEHV carrying its temporal state across blocks
//! ([`super::taehv::TaeDecodeState`]), so the frames of a stream are the
//! frames a whole-clip decode of the same latents gives. Block 0 yields
//! `4·3 − 3 = 9` pixel frames, every later block 12.
//!
//! **Window.** The KV cache rolls (FastVideo `local_attn_size`, default 21
//! frames, the checkpoint's training window) and keeps `sink_frames` frames
//! at its head (FastVideo `sink_size`). RoPE ([`RopePolicy`], the cache's
//! [`super::causal::KvRope`]): [`RopePolicy::Absolute`] is the bounded
//! path's policy, which puts the sink ever further from the queries (the
//! temporal RoPE rows continue past the checkpoint's 1024-frame table);
//! [`RopePolicy::Relativistic`] is FastVideo's `relativistic` policy, which
//! caches un-roped keys and ropes the whole window from position 0 at every
//! forward, so every offset stays inside the trained window;
//! [`RopePolicy::RebasedSink`] (default) gives the relativistic offsets at
//! the absolute policy's cost by re-roping only the sink once per block.
//!
//! **Noise.** Latent frames `21·g .. 21·(g+1)` start from a `[1, C, 21, H, W]`
//! draw of `StdRng(seed_g)` (`seed_0 = seed`, the bounded path's own draw);
//! re-noise draw `k` is the bounded path's `causal_noise(seed, k)`. With
//! [`RopePolicy::Absolute`], a 21-frame window and no sink, the first seven
//! blocks are therefore exactly the bounded 81-frame generation.
//!
//! **Prompts.** [`CausalRollout::set_prompt`] encodes at once and applies from
//! the next block. The KV cache is kept ([`PromptSwitch::Keep`], the design's
//! default: the scene carries over and the new prompt steers it), or cleared
//! with the stream restarted at block 0 ([`PromptSwitch::Reset`], a hard
//! cut). [`CausalRollout::reset`] is the explicit restart.
//!
//! Only TAEHV decodes per block; the full Wan VAE clears its causal feature
//! cache per call, which is strobe's seam, so a pipeline without TAEHV
//! weights is refused.

use std::sync::mpsc::SyncSender;
use std::time::Instant;

use fastvideo_models::schedulers::{SelfForcingSchedule, SF_WAN_1_3B_DMD_STEPS};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use super::causal::{CausalKvCache, KvSpec};
use crate::hooks::{Hooks, Stage};
use super::pipeline::{
    causal_noise, frames_to_rgb8, GenerateConfig, PipelineError, Result, WanPipeline,
};
use super::taehv::TaeDecodeState;
use super::tensor::CudaTensor;

fn err(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// How the cached keys carry RoPE (see [`super::causal::KvRope`]).
pub use super::causal::KvRope as RopePolicy;

/// What a prompt change does to the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptSwitch {
    /// Keep the cache: the next block attends to frames made under the old
    /// prompt and conditions on the new one.
    Keep,
    /// Clear the cache and restart at block 0 (a hard cut).
    Reset,
}

#[derive(Debug, Clone)]
pub struct RolloutConfig {
    pub prompt: String,
    pub height: usize,
    pub width: usize,
    pub seed: u64,
    /// Self-Forcing DMD timesteps before warping (`[1000, 750, 500, 250]`).
    pub dmd_steps: Vec<i32>,
    pub flow_shift: f64,
    /// KV window in latent frames (FastVideo `local_attn_size`).
    pub local_attn_frames: usize,
    /// Frames kept at the head of the rolling cache (`sink_size`).
    pub sink_frames: usize,
    pub rope: RopePolicy,
    pub prompt_switch: PromptSwitch,
    /// Pack each block to 8-bit RGB on the host ([`StreamBlock::rgb`]).
    pub rgb8: bool,
    pub tokenizer_path: Option<String>,
    pub text_cache: Option<std::path::PathBuf>,
    /// Replay CUDA graphs of the block forwards ([`super::graph`]; default
    /// from `FASTVIDEO_WAN_GRAPH`, on). Off (or without a CUDA device): every
    /// block runs eagerly on the device's own stream.
    pub graphs: bool,
}

impl Default for RolloutConfig {
    fn default() -> Self {
        Self {
            prompt: "a cat walking".into(),
            height: 480,
            width: 832,
            seed: 1024,
            dmd_steps: SF_WAN_1_3B_DMD_STEPS.to_vec(),
            flow_shift: 5.0,
            local_attn_frames: 21,
            sink_frames: 3,
            rope: RopePolicy::RebasedSink,
            prompt_switch: PromptSwitch::Keep,
            rgb8: true,
            tokenizer_path: None,
            text_cache: None,
            graphs: super::graph::graphs_enabled(),
        }
    }
}

/// Wall time of one block (each phase ends with a device sync).
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockTimings {
    /// The Self-Forcing steps.
    pub denoise_s: f64,
    /// The clean-context KV pass.
    pub context_s: f64,
    /// TAEHV decode of the block.
    pub decode_s: f64,
    /// RGB8 packing and the copy down.
    pub rgb_s: f64,
    pub total_s: f64,
}

/// One generated block.
pub struct StreamBlock {
    /// Block index since the last reset.
    pub index: usize,
    /// Pixel frame index of `frames`' first frame since the last reset.
    pub first_frame: usize,
    /// Decoded frames `[n, 3, H, W]` in `[-1, 1]` (device).
    pub frames: CudaTensor,
    /// The block's clean latents `[1, C, 3, h, w]` (device).
    pub latents: CudaTensor,
    /// `[n, H, W, 3]` 8-bit RGB when [`RolloutConfig::rgb8`].
    pub rgb: Option<Vec<u8>>,
    /// Bumped by every [`CausalRollout::set_prompt`]; the version these
    /// frames were conditioned on.
    pub prompt_version: u64,
    pub timings: BlockTimings,
}

impl StreamBlock {
    pub fn num_frames(&self) -> usize {
        self.frames.shape[0]
    }

    /// The host part, for a consumer thread (a pacer or an encoder).
    pub fn into_host(self) -> HostBlock {
        HostBlock {
            index: self.index,
            first_frame: self.first_frame,
            frames: self.frames.shape[0],
            height: self.frames.shape[2],
            width: self.frames.shape[3],
            rgb: self.rgb.unwrap_or_default(),
            prompt_version: self.prompt_version,
            timings: self.timings,
        }
    }
}

/// [`StreamBlock`] without device tensors: what crosses to the media side.
#[derive(Debug, Clone)]
pub struct HostBlock {
    pub index: usize,
    pub first_frame: usize,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    /// `[frames, height, width, 3]` RGB8.
    pub rgb: Vec<u8>,
    pub prompt_version: u64,
    pub timings: BlockTimings,
}

/// What the block callback of [`CausalRollout::run`] asks for next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
}

/// Latent frames per noise group: the bounded clip's 21.
const NOISE_GROUP: usize = 21;

/// The open-ended causal SF-Wan generator over one resident pipeline.
pub struct CausalRollout<'p> {
    pipe: &'p WanPipeline,
    cfg: RolloutConfig,
    sched: SelfForcingSchedule,
    cache: CausalKvCache,
    tae_state: TaeDecodeState,
    cond: CudaTensor,
    prompt_version: u64,
    /// `[C, h, w]` of one latent frame.
    latent_chw: [usize; 3],
    fpb: usize,
    /// Blocks since the last reset.
    block: usize,
    /// Re-noise draws since the last reset.
    draw: usize,
    /// Seed of the current run (a reset may change it).
    seed: u64,
    /// Initial noise of the current 21-frame group: `(group, [1, C, 21, h, w])`.
    noise: Option<(usize, CudaTensor)>,
    /// Graph mode ([`RolloutConfig::graphs`] on a CUDA device).
    #[cfg(feature = "cuda")]
    graph: Option<GraphRun>,
}

/// What the graph mode of a rollout has done so far.
#[derive(Debug, Clone, Default)]
pub struct GraphReport {
    /// Blocks run eagerly on the graph stream (the first of each cache
    /// state: it warms every lazy cache before a capture).
    pub eager_blocks: usize,
    /// Blocks whose graphs were captured (then launched).
    pub captured_blocks: usize,
    /// Blocks run by replaying graphs captured before.
    pub replayed_blocks: usize,
    /// Distinct cache states (graph keys) seen.
    pub keys: usize,
    /// Kernel nodes of the last captured denoise / context graph.
    pub denoise_kernels: usize,
    pub context_kernels: usize,
    /// Why graph mode fell back to eager for good, if it did.
    pub failed: Option<String>,
}

/// One graph per cache state: the fill states of the window's first blocks,
/// then one steady state per roll.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GraphKey {
    kv: super::causal::KvBlockKey,
    rows: usize,
    sink: bool,
}

#[cfg(feature = "cuda")]
struct GraphEntry {
    /// The RoPE tables the graphs read, refilled per block.
    tables: super::transformer::KvTables,
    runs: usize,
    denoise: Option<super::graph::Graph>,
    context: Option<super::graph::Graph>,
}

/// Graph-mode state: the stream, and every persistent buffer a graph
/// reads (the inputs are refilled in place before each launch).
#[cfg(feature = "cuda")]
struct GraphRun {
    gs: super::graph::GraphStream,
    /// Time conditioning of each Self-Forcing step, then the context pass.
    times: Vec<super::transformer::KvTime>,
    /// The text conditioning slot and the prompt version it holds.
    text: Option<(super::transformer::KvText, u64)>,
    /// The block's initial noise and each re-noise draw.
    noise0: Option<CudaTensor>,
    renoise: Vec<CudaTensor>,
    /// The denoised latents: the denoise graph's output, the context
    /// graph's input.
    mid: Option<CudaTensor>,
    entries: std::collections::HashMap<GraphKey, GraphEntry>,
    report: GraphReport,
}

/// Graphs kept per rollout (distinct cache states; a 21-frame window with
/// 3-frame blocks has 9).
#[cfg(feature = "cuda")]
const MAX_GRAPH_KEYS: usize = 32;

#[cfg(feature = "cuda")]
impl Drop for GraphRun {
    fn drop(&mut self) {
        self.entries.clear();
        self.gs.trim();
    }
}

/// Graph mode falls back to eager blocks for good, saying why once.
#[cfg(feature = "cuda")]
fn give_up(failed: &mut Option<String>, why: String) {
    super::log::info(format_args!("causal rollout: {why}; running eagerly from here"));
    failed.get_or_insert(why);
}

/// Keep a captured (and launched) graph for replay, unless allocations
/// escape it: their memory is live (the launch made it), may be referenced,
/// and a replay would hand the same addresses out again, so the graph is
/// leaked (never replayed, never destroyed) and graph mode gives up.
#[cfg(feature = "cuda")]
fn keep_graph(
    graph: super::graph::Graph,
    slot: &mut Option<super::graph::Graph>,
    failed: &mut Option<String>,
    what: &str,
) {
    if graph.escaped() == 0 {
        *slot = Some(graph);
    } else {
        give_up(
            failed,
            format!("{what} graph: {} allocations outlive it ({:?})", graph.escaped(), graph.stats),
        );
        std::mem::forget(graph);
    }
}

/// Refill slot `slot` with `src` (allocating it the first time).
#[cfg(feature = "cuda")]
fn fill(slot: &mut Option<CudaTensor>, src: &CudaTensor) -> Result<()> {
    match slot {
        Some(s) if s.shape == src.shape && s.dtype() == src.dtype() => {
            super::graph::assign(s, src)?;
        }
        _ => *slot = Some(super::graph::duplicate(src)?),
    }
    Ok(())
}

impl<'p> CausalRollout<'p> {
    /// Encode the prompt and set up an empty cache. Nothing is generated yet.
    pub fn open(pipe: &'p WanPipeline, cfg: RolloutConfig) -> Result<Self> {
        let dit = pipe.transformer();
        if !dit.cfg.causal {
            return Err(err("causal rollout needs a causal (Self-Forcing) Wan preset"));
        }
        let tae = pipe.taehv().ok_or_else(|| {
            err("causal rollout decodes per block with TAEHV (taew2_1): none loaded \
                 (FASTVIDEO_TAEHV_WEIGHTS / FASTVIDEO_TAE_DIR; scripts/gpu/fetch_taehv.sh)")
        })?;
        let (z_c, _, z_h, z_w) = pipe.latent_shape(&GenerateConfig {
            height: cfg.height,
            width: cfg.width,
            num_frames: 1,
            ..GenerateConfig::default()
        });
        let p = dit.cfg.patch_size;
        let frame_tokens = (z_h / p[1].max(1)) * (z_w / p[2].max(1));
        let fpb = dit.cfg.num_frames_per_block.max(1);
        if cfg.local_attn_frames < fpb {
            return Err(err(format!(
                "local_attn_frames {} is below one block ({fpb} frames)",
                cfg.local_attn_frames
            )));
        }
        let spec = KvSpec::rolling(
            frame_tokens,
            cfg.local_attn_frames,
            cfg.sink_frames,
            cfg.rope,
        )?;
        let sched = SelfForcingSchedule::new(&cfg.dmd_steps, cfg.flow_shift, 1000, true);
        dit.begin_text_cache();
        #[cfg(feature = "cuda")]
        let graph = if cfg.graphs && super::device::has_live_device() {
            let gs = super::graph::GraphStream::new()?;
            // The time conditioning of every step (and of the context
            // pass), held for the rollout's life: graphs read it by address.
            let times = gs.scope(|| -> Result<Vec<_>> {
                let mut ts: Vec<f32> = sched.timesteps.clone();
                ts.push(0.0);
                let out = ts
                    .iter()
                    .map(|&t| dit.kv_time(&CudaTensor::from_vec(vec![t], vec![1])?, 1))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                gs.synchronize()?;
                Ok(out)
            })?;
            Some(GraphRun {
                gs,
                times,
                text: None,
                noise0: None,
                renoise: Vec::new(),
                mid: None,
                entries: Default::default(),
                report: GraphReport::default(),
            })
        } else {
            None
        };
        #[cfg(feature = "cuda")]
        let cache = if graph.is_some() {
            CausalKvCache::new_static(spec, dit.cfg.num_layers)
        } else {
            CausalKvCache::new(spec, dit.cfg.num_layers)
        };
        #[cfg(not(feature = "cuda"))]
        let cache = CausalKvCache::new(spec, dit.cfg.num_layers);
        let mut me = Self {
            pipe,
            sched,
            cache,
            tae_state: tae.decode_state(),
            cond: CudaTensor::zeros(&[1]),
            prompt_version: 0,
            latent_chw: [z_c, z_h, z_w],
            fpb,
            block: 0,
            draw: 0,
            seed: cfg.seed,
            noise: None,
            #[cfg(feature = "cuda")]
            graph,
            cfg,
        };
        let prompt = me.cfg.prompt.clone();
        me.cond = me.encode(&prompt)?;
        super::log::info(format_args!(
            "causal rollout {}x{} window={} sink={} rope={:?} switch={:?} timesteps {:?}",
            me.cfg.width,
            me.cfg.height,
            me.cfg.local_attn_frames,
            me.cfg.sink_frames,
            me.cfg.rope,
            me.cfg.prompt_switch,
            me.sched.timesteps
        ));
        Ok(me)
    }

    fn encode(&self, prompt: &str) -> Result<CudaTensor> {
        let req = GenerateConfig {
            prompt: prompt.to_string(),
            is_dmd: true,
            tokenizer_path: self.cfg.tokenizer_path.clone(),
            text_cache: self.cfg.text_cache.clone(),
            ..GenerateConfig::default()
        };
        Ok(self.pipe.encode_prompt(&req)?.to_device()?)
    }

    /// Condition the next block on `prompt` (encoded now). Returns the new
    /// prompt version.
    pub fn set_prompt(&mut self, prompt: &str) -> Result<u64> {
        self.set_prompt_embeds(self.encode(prompt)?)?;
        self.cfg.prompt = prompt.to_string();
        Ok(self.prompt_version)
    }

    /// [`Self::set_prompt`] from precomputed prompt embeddings
    /// (`[1, text_len, 4096]`).
    pub fn set_prompt_embeds(&mut self, embeds: CudaTensor) -> Result<u64> {
        if embeds.shape.len() != 3 || embeds.shape[0] != 1 {
            return Err(err(format!("prompt embeds {:?}", embeds.shape)));
        }
        self.cond = embeds.to_device()?;
        self.prompt_version += 1;
        if self.cfg.prompt_switch == PromptSwitch::Reset {
            self.restart(None);
        }
        Ok(self.prompt_version)
    }

    /// Clear the KV cache and the decoder state and restart at block 0,
    /// with `seed` when given (else the current one).
    pub fn reset(&mut self, seed: Option<u64>) {
        self.restart(seed);
    }

    fn restart(&mut self, seed: Option<u64>) {
        self.cache.reset();
        if let Some(tae) = self.pipe.taehv() {
            self.tae_state = tae.decode_state();
        }
        self.block = 0;
        self.draw = 0;
        self.noise = None;
        if let Some(s) = seed {
            self.seed = s;
        }
        self.pipe.transformer().forget_rotary_before(usize::MAX);
    }

    /// Blocks generated since the last reset.
    pub fn blocks(&self) -> usize {
        self.block
    }

    pub fn prompt_version(&self) -> u64 {
        self.prompt_version
    }

    pub fn config(&self) -> &RolloutConfig {
        &self.cfg
    }

    /// Bytes the KV cache holds now (bounded by the window).
    pub fn kv_bytes(&self) -> usize {
        self.cache.bytes()
    }

    /// The initial noise of latent frames `start .. start + fpb`.
    fn block_noise(&mut self, start: usize) -> Result<CudaTensor> {
        let group = start / NOISE_GROUP;
        let off = start % NOISE_GROUP;
        if off + self.fpb > NOISE_GROUP {
            // A block straddling two groups (fpb not dividing 21): its own draw.
            return self.draw_noise(group * 0x1_0000 + off + 1, self.fpb);
        }
        if self.noise.as_ref().map(|(g, _)| *g) != Some(group) {
            let t = self.draw_noise(group, NOISE_GROUP)?;
            self.noise = Some((group, t));
        }
        let (_, t) = self.noise.as_ref().expect("noise group");
        Ok(t.narrow(2, off, self.fpb)?.to_device()?)
    }

    /// `[1, C, frames, h, w]` from `StdRng(seed_g)`, in the order
    /// `WanPipeline::initial_latents` draws (group 0 is its draw).
    fn draw_noise(&self, group: usize, frames: usize) -> Result<CudaTensor> {
        let [c, h, w] = self.latent_chw;
        let seed = if group == 0 {
            self.seed
        } else {
            self.seed ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(group as u64)
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let v: Vec<f32> = (0..c * frames * h * w)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        Ok(CudaTensor::from_vec(v, vec![1, c, frames, h, w])?)
    }

    /// Generate, decode and return the next block.
    pub fn next_block(&mut self) -> Result<StreamBlock> {
        self.next_block_with_hooks(Hooks::NONE)
    }

    /// [`Self::next_block`] under the pipeline hooks (serve E1): a
    /// [`Stage::Denoise`] step event after each Self-Forcing step (with the
    /// block index; `total` is the steps of one block) and a
    /// [`Stage::VideoDecode`] frames event (cumulative since the last reset)
    /// after the decode, the cancel token checked at each. A cancelled block
    /// leaves the stream at the same block: the next call generates it again
    /// (its cache slots are overwritten in place), or [`Self::reset`].
    pub fn next_block_with_hooks(&mut self, hooks: Hooks<'_>) -> Result<StreamBlock> {
        hooks.check()?;
        #[cfg(feature = "cuda")]
        if self.graph.is_some() {
            return self.next_block_graph(hooks);
        }
        let t_block = Instant::now();
        let dit = self.pipe.transformer();
        if self.pipe.taehv().is_none() {
            return Err(err("causal rollout: TAEHV unloaded"));
        }
        let fpb = self.fpb;
        let start = self.block * fpb;
        if self.cfg.rope != RopePolicy::Relativistic {
            // One RoPE table per block offset (and per sink target): keep
            // only the current ones.
            dit.forget_rotary_before(start.saturating_sub(self.cfg.local_attn_frames));
        }
        let act16 = super::tensor::bf16_activations();
        let round = |x: CudaTensor| -> Result<CudaTensor> {
            Ok(if act16 { x.quantize_bf16()? } else { x })
        };
        let [c, h, w] = self.latent_chw;
        let steps = self.sched.num_steps();
        let mut cur = self.block_noise(start)?;
        for (i, &ts) in self.sched.timesteps.iter().enumerate() {
            let t1 = CudaTensor::from_vec(vec![ts], vec![1])?;
            let input = round(cur.clone())?;
            let flow = dit.forward_kv(&input, &t1, &self.cond, &self.cache, start)?;
            let sigma = self.sched.sigma(ts) as f32;
            let x0 = round(CudaTensor::lincomb(&[(1.0, &cur), (-sigma, &flow)])?)?;
            cur = if i + 1 < steps {
                let next = self.sched.sigma(self.sched.timesteps[i + 1]) as f32;
                let noise = causal_noise(self.seed, self.draw, [1, fpb, c, h, w])?;
                self.draw += 1;
                round(CudaTensor::lincomb(&[(1.0 - next, &x0), (next, &noise)])?)?
            } else {
                x0
            };
            if !hooks.is_none() {
                hooks.step(Stage::Denoise, i + 1, steps, Some(self.block))?;
            }
        }
        super::device::synchronize().map_err(|e| err(e.to_string()))?;
        let denoise_s = t_block.elapsed().as_secs_f64();
        // The clean context pass: the same block at t = context_noise (0).
        let t = Instant::now();
        let t0 = CudaTensor::from_vec(vec![0.0f32], vec![1])?;
        dit.forward_kv(&cur, &t0, &self.cond, &self.cache, start)?;
        super::device::synchronize().map_err(|e| err(e.to_string()))?;
        let context_s = t.elapsed().as_secs_f64();
        self.finish_block(cur, t_block, denoise_s, context_s, hooks)
    }

    /// Decode the block's latents `cur` (TAEHV, carried state), pack RGB8
    /// and deliver the block.
    fn finish_block(
        &mut self,
        cur: CudaTensor,
        t_block: Instant,
        denoise_s: f64,
        context_s: f64,
        hooks: Hooks<'_>,
    ) -> Result<StreamBlock> {
        let tae = self
            .pipe
            .taehv()
            .ok_or_else(|| err("causal rollout: TAEHV unloaded"))?;
        let t = Instant::now();
        let first_frame = self.tae_state.emitted();
        let frames = tae
            .decode_step(&mut self.tae_state, &cur)?
            .ok_or_else(|| err("causal rollout: a block decoded to no frames"))?;
        super::device::synchronize().map_err(|e| err(e.to_string()))?;
        let decode_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let rgb = if self.cfg.rgb8 {
            Some(frames_to_rgb8(&frames)?)
        } else {
            None
        };
        let rgb_s = t.elapsed().as_secs_f64();
        // The decoder state has moved on: this block is delivered whatever
        // the token says, and the next call's check sees a cancel.
        let _ = hooks.frames(self.tae_state.emitted());
        let index = self.block;
        self.block += 1;
        Ok(StreamBlock {
            index,
            first_frame,
            frames,
            latents: cur,
            rgb,
            prompt_version: self.prompt_version,
            timings: BlockTimings {
                denoise_s,
                context_s,
                decode_s,
                rgb_s,
                total_s: t_block.elapsed().as_secs_f64(),
            },
        })
    }

    /// What graph mode has done so far (`None`: the rollout runs eagerly).
    pub fn graph_report(&self) -> Option<GraphReport> {
        #[cfg(feature = "cuda")]
        if let Some(g) = &self.graph {
            let mut r = g.report.clone();
            r.keys = g.entries.len();
            return Some(r);
        }
        None
    }

    /// [`Self::next_block_with_hooks`] in graph mode: the denoise steps and
    /// the context pass run on the graph stream from persistent buffers, as
    /// one graph each per cache state. The first block of a state runs
    /// eagerly (warming every lazy cache, allocating the static KV buffers
    /// and the slots), the second captures and launches, later ones replay.
    /// The step events fire together once the denoise graph is done (a
    /// cancel is seen then: the block's cache slots are written, and the
    /// next call generates it again, as in eager mode).
    #[cfg(feature = "cuda")]
    fn next_block_graph(&mut self, hooks: Hooks<'_>) -> Result<StreamBlock> {
        let t_block = Instant::now();
        let gs = self.graph.as_ref().expect("graph mode").gs.clone();
        let (cur, denoise_s, context_s) = gs.scope(|| self.graph_block(&hooks, t_block))?;
        self.finish_block(cur, t_block, denoise_s, context_s, hooks)
    }

    /// The device part of a graph-mode block, inside the graph stream's
    /// scope: `(latents, denoise_s, context_s)`.
    #[cfg(feature = "cuda")]
    fn graph_block(&mut self, hooks: &Hooks<'_>, t_block: Instant) -> Result<(CudaTensor, f64, f64)> {
        use super::graph::{assign, duplicate};
        let dit = self.pipe.transformer();
        let fpb = self.fpb;
        let block = self.block;
        let start = block * fpb;
        let [c, h, w] = self.latent_chw;
        let steps = self.sched.num_steps();
        let p = dit.cfg.patch_size;
        let frame_tokens = (h / p[1].max(1)) * (w / p[2].max(1));
        let n = fpb * frame_tokens;
        // Inputs, refilled in place.
        let noise0 = self.block_noise(start)?;
        let mut renoise = Vec::with_capacity(steps.saturating_sub(1));
        for _ in 1..steps {
            renoise.push(causal_noise(self.seed, self.draw, [1, fpb, c, h, w])?);
            self.draw += 1;
        }
        let text_now = {
            let g = self.graph.as_ref().expect("graph mode");
            g.text.as_ref().map(|(_, v)| *v)
        };
        if text_now != Some(self.prompt_version) {
            let fresh = dit.kv_text(&self.cond)?;
            let g = self.graph.as_mut().expect("graph mode");
            match &g.text {
                Some((slot, _)) => slot.assign(&fresh)?,
                None => g.text = Some((fresh.duplicate()?, 0)),
            }
            g.text.as_mut().expect("text slot").1 = self.prompt_version;
        }
        let spec = dit.kv_table_spec(fpb, h, w, &self.cache, start)?;
        let target = spec.sink.map_or(0, |s| s.0);
        let host_tables = dit.kv_tables_host(&spec, h, w)?;
        let key = GraphKey {
            kv: self.cache.block_key(start * frame_tokens, n, target)?,
            rows: spec.rows,
            sink: spec.sink.is_some(),
        };
        let g = self.graph.as_mut().expect("graph mode");
        fill(&mut g.noise0, &noise0)?;
        if g.renoise.len() != renoise.len() {
            g.renoise.clear();
        }
        for (i, r) in renoise.iter().enumerate() {
            if let Some(slot) = g.renoise.get(i) {
                if slot.shape == r.shape && slot.dtype() == r.dtype() {
                    assign(slot, r)?;
                    continue;
                }
            }
            let d = duplicate(r)?;
            if i < g.renoise.len() {
                g.renoise[i] = d;
            } else {
                g.renoise.push(d);
            }
        }
        drop((noise0, renoise));
        let over_cap = !g.entries.contains_key(&key) && g.entries.len() >= MAX_GRAPH_KEYS;
        let entry = match g.entries.get_mut(&key) {
            Some(e) if e.tables.same_layout(&host_tables) => {
                for (d, s) in e.tables.tensors().into_iter().zip(host_tables.tensors()) {
                    assign(d, s)?;
                }
                e
            }
            _ => {
                let t = &host_tables;
                let tables = super::transformer::KvTables {
                    rope: (duplicate(&t.rope.0)?, duplicate(&t.rope.1)?),
                    sink: match &t.sink {
                        Some((tg, a, b)) => Some((
                            *tg,
                            (duplicate(&a.0)?, duplicate(&a.1)?),
                            (duplicate(&b.0)?, duplicate(&b.1)?),
                        )),
                        None => None,
                    },
                };
                g.entries.insert(
                    key,
                    GraphEntry {
                        tables,
                        runs: 0,
                        denoise: None,
                        context: None,
                    },
                );
                g.entries.get_mut(&key).expect("graph entry")
            }
        };
        // The target frame is the tables' (the slot keeps the first one's).
        if let (Some(s), Some(h)) = (entry.tables.sink.as_mut(), host_tables.sink.as_ref()) {
            s.0 = h.0;
        }
        let first = entry.runs == 0 || over_cap || g.report.failed.is_some();
        entry.runs += 1;

        let act16 = super::tensor::bf16_activations();
        let sched = &self.sched;
        let cache = &self.cache;
        let times = &g.times;
        let text = &g.text.as_ref().expect("text slot").0;
        let noise0 = g.noise0.as_ref().expect("noise slot");
        let renoise = &g.renoise;
        let tables = &entry.tables;
        let denoise = || -> super::tensor::Result<CudaTensor> {
            let round = |x: CudaTensor| -> super::tensor::Result<CudaTensor> {
                Ok(if act16 { x.quantize_bf16()? } else { x })
            };
            let mut cur = noise0.clone();
            for (i, &ts) in sched.timesteps.iter().enumerate() {
                let input = round(cur.clone())?;
                let flow = dit.forward_kv_cond(&input, &times[i], text, cache, start, tables)?;
                let sigma = sched.sigma(ts) as f32;
                let x0 = round(CudaTensor::lincomb(&[(1.0, &cur), (-sigma, &flow)])?)?;
                cur = if i + 1 < steps {
                    let next = sched.sigma(sched.timesteps[i + 1]) as f32;
                    round(CudaTensor::lincomb(&[(1.0 - next, &x0), (next, &renoise[i])])?)?
                } else {
                    x0
                };
            }
            Ok(cur)
        };
        let context = |mid: &CudaTensor| -> super::tensor::Result<()> {
            dit.forward_kv_cond(mid, &times[steps], text, cache, start, tables)?;
            Ok(())
        };
        let step_events = |hooks: &Hooks<'_>| -> Result<()> {
            if !hooks.is_none() {
                for i in 0..steps {
                    hooks.step(Stage::Denoise, i + 1, steps, Some(block))?;
                }
            }
            Ok(())
        };

        if first {
            // Eager: the same work on the same buffers.
            let cur = denoise()?;
            fill(&mut g.mid, &cur)?;
            drop(cur);
            g.gs.synchronize()?;
            let denoise_s = t_block.elapsed().as_secs_f64();
            step_events(hooks)?;
            let t = Instant::now();
            context(g.mid.as_ref().expect("mid slot"))?;
            let out = duplicate(g.mid.as_ref().expect("mid slot"))?;
            g.gs.synchronize()?;
            g.report.eager_blocks += 1;
            return Ok((out, denoise_s, t.elapsed().as_secs_f64()));
        }

        let mid = g.mid.as_ref().expect("mid slot");
        let replay = entry.denoise.is_some() && entry.context.is_some();
        let mut fell_back = false;
        if replay {
            entry.denoise.as_ref().expect("denoise graph").launch()?;
            for _ in 0..steps {
                cache.advance(start * frame_tokens, n, target)?;
            }
        } else {
            let before = cache.pointers();
            match g.gs.capture(|| {
                let cur = denoise()?;
                assign(mid, &cur)
            }) {
                Ok(graph) => {
                    // The launch is this block's denoise.
                    graph.launch()?;
                    g.report.denoise_kernels = graph.stats.kernels;
                    keep_graph(graph, &mut entry.denoise, &mut g.report.failed, "denoise");
                }
                Err(e) => {
                    // The capture moved the cache's pointers but ran nothing.
                    fell_back = true;
                    give_up(&mut g.report.failed, format!("denoise capture: {e}"));
                    cache.set_pointers(&before)?;
                    let cur = denoise()?;
                    assign(mid, &cur)?;
                }
            }
        }
        g.gs.synchronize()?;
        let denoise_s = t_block.elapsed().as_secs_f64();
        step_events(hooks)?;
        let t = Instant::now();
        if replay {
            entry.context.as_ref().expect("context graph").launch()?;
            cache.advance(start * frame_tokens, n, target)?;
        } else if g.report.failed.is_some() {
            fell_back = true;
            context(mid)?;
        } else {
            let before = cache.pointers();
            match g.gs.capture(|| context(mid)) {
                Ok(graph) => {
                    graph.launch()?;
                    g.report.context_kernels = graph.stats.kernels;
                    keep_graph(graph, &mut entry.context, &mut g.report.failed, "context");
                }
                Err(e) => {
                    fell_back = true;
                    give_up(&mut g.report.failed, format!("context capture: {e}"));
                    cache.set_pointers(&before)?;
                    context(mid)?;
                }
            }
        }
        let out = duplicate(mid)?;
        g.gs.synchronize()?;
        if replay {
            g.report.replayed_blocks += 1;
        } else if fell_back {
            g.report.eager_blocks += 1;
        } else {
            g.report.captured_blocks += 1;
        }
        Ok((out, denoise_s, t.elapsed().as_secs_f64()))
    }

    /// Generate until `sink` answers [`Flow::Stop`] (or `max_blocks`).
    /// Returns the number of blocks generated.
    pub fn run(
        &mut self,
        max_blocks: Option<usize>,
        mut sink: impl FnMut(StreamBlock) -> Result<Flow>,
    ) -> Result<usize> {
        let mut n = 0;
        while max_blocks.is_none_or(|m| n < m) {
            let block = self.next_block()?;
            n += 1;
            if sink(block)? == Flow::Stop {
                break;
            }
        }
        Ok(n)
    }

    /// [`Self::run`] into a bounded channel (the design's depth-4 hand-off
    /// to the pacer): blocks while the consumer is behind, stops when it
    /// hangs up.
    pub fn run_into(
        &mut self,
        max_blocks: Option<usize>,
        tx: &SyncSender<HostBlock>,
    ) -> Result<usize> {
        self.run(max_blocks, |b| {
            Ok(if tx.send(b.into_host()).is_ok() {
                Flow::Continue
            } else {
                Flow::Stop
            })
        })
    }
}

impl Drop for CausalRollout<'_> {
    fn drop(&mut self) {
        let dit = self.pipe.transformer();
        dit.end_text_cache();
        dit.forget_rotary_before(usize::MAX);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_window_is_the_training_window_with_a_block_sink() {
        let c = RolloutConfig::default();
        assert_eq!(c.local_attn_frames, 21);
        assert_eq!(c.sink_frames, 3);
        assert_eq!(c.rope, RopePolicy::RebasedSink);
        assert_eq!(c.dmd_steps, vec![1000, 750, 500, 250]);
    }
}
