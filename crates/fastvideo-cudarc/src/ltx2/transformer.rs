//! `LTX2VideoTransformer3DModel`: the 19B dual-stream DiT.
//!
//! Two token streams of different widths — video (4096) and audio (2048) — run
//! through the *same* 48 blocks. Each block, in order:
//!
//! 1. self-attention per stream (AdaLN in, gate out, the stream's own rotary);
//! 2. text cross-attention per stream — no modulation, no gate, no rotary;
//! 3. audio↔video cross-attention in both directions, both computed from the
//!    states as they were *before* either update, each side rotated by its own
//!    time-only table so tokens at the same instant line up;
//! 4. feed-forward per stream (AdaLN in, gate out).
//!
//! Everything that depends on the timestep is a per-step constant *vector*
//! (T2AV has one sigma per stream and the two are equal), so modulation is
//! broadcast over tokens. Every block norm is a weightless RMSNorm; the
//! `rms(x)·(1 + scale)` of AdaLN is therefore the RMSNorm kernel with
//! `1 + scale` as its weight, followed by one broadcast add.
//!
//! LTX-2.0 is the default path (6-row AdaLN, caption projections, ungated
//! attention). LTX-2.5 adds gated SDPA, 9-row AdaLN + prompt K/V tables,
//! optional global `prompt_adaln`, and `ff_bias=false` on the video FFN.
//! Perturbed/STG attention is not wired in the forward path (distilled CFG=1).
//! See docs/ports/ltx2.md §e and docs/ports/ltx25.md.

use std::sync::{Arc, Mutex};

use fastvideo_models::ltx2::config::Ltx2TransformerConfig;
use fastvideo_models::ltx2::fbcache::FbCache;
use fastvideo_models::ltx2::Ltx2RopeTables;

use crate::wan::nn::{sinusoidal_timesteps, Linear};
use crate::wan::tensor::{CudaTensor, Result};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

use super::attention::{Attention, AttentionDims, DeviceRope, FeedForward, VideoAttnKernel};
use super::keys::Keys;
use super::{msg, ones};

/// `LTX2AdaLayerNormSingle`: sinusoid → MLP → (`embedded`, `rows` modulation
/// vectors).
struct AdaLnSingle {
    linear_1: Linear,
    linear_2: Linear,
    linear: Linear,
    dim: usize,
    rows: usize,
    sinusoid: usize,
}

impl AdaLnSingle {
    fn load(
        map: &WeightMap,
        keys: &Keys,
        name: &str,
        dim: usize,
        rows: usize,
        sinusoid: usize,
    ) -> Result<Self> {
        let lin = |suffix: &str, i: usize, o: usize| {
            Linear::load(map, &keys.key(&format!("{name}.{suffix}")), i, o, true)
        };
        Ok(Self {
            linear_1: lin("emb.timestep_embedder.linear_1", sinusoid, dim)?,
            linear_2: lin("emb.timestep_embedder.linear_2", dim, dim)?,
            linear: lin("linear", dim, rows * dim)?,
            dim,
            rows,
            sinusoid,
        })
    }

    /// `(modulation [rows, dim], embedded [1, dim])` for one timestep
    /// (`1000·sigma`).
    fn forward(&self, timestep: f32) -> Result<(CudaTensor, CudaTensor)> {
        let s = sinusoidal_timesteps(
            &CudaTensor::from_vec(vec![timestep], vec![1])?,
            self.sinusoid,
        )?;
        let e = self.linear_2.forward(&self.linear_1.forward(&s)?.silu())?;
        let m = self
            .linear
            .forward(&e.silu())?
            .reshape(vec![self.rows, self.dim])?;
        Ok((m, e))
    }
}

/// `PixArtAlphaTextProjection`: `Linear → tanh-GELU → Linear`.
struct CaptionProjection {
    linear_1: Linear,
    linear_2: Linear,
}

impl CaptionProjection {
    fn load(map: &WeightMap, keys: &Keys, name: &str, caption: usize, dim: usize) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::load(
                map,
                &keys.key(&format!("{name}.linear_1")),
                caption,
                dim,
                true,
            )?,
            linear_2: Linear::load(map, &keys.key(&format!("{name}.linear_2")), dim, dim, true)?,
        })
    }

    fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        self.linear_2.forward(&self.linear_1.forward_gelu(x)?)
    }
}

/// Row `r` of a `[rows, dim]` table as `[1, dim]`.
fn row(t: &CudaTensor, r: usize) -> Result<CudaTensor> {
    t.narrow(0, r, 1)
}

/// `rms(x) · (1 + scale) + shift` with a weightless RMSNorm.
fn rms_adaln(
    x: &CudaTensor,
    scale: &CudaTensor,
    shift: &CudaTensor,
    eps: f32,
) -> Result<CudaTensor> {
    x.rms_norm(&scale.try_add_scalar(1.0)?, eps)?.add(shift)
}

/// `x · (1 + scale) + shift` with broadcast `[1, dim]` modulation rows.
fn scale_shift(x: &CudaTensor, scale: &CudaTensor, shift: &CudaTensor) -> Result<CudaTensor> {
    x.mul(&scale.try_add_scalar(1.0)?)?.add(shift)
}

/// Stage-2 video self-attention for one denoise loop. The step index is applied
/// inside the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ltx2Stage2Attn {
    Off,
    /// LTX-2.5 Sol: layer 0 dense, layers 1..=47 at that forward's tau.
    Sol,
    /// LTX-2.3 PISA: layers 0..=1 dense, later layers piecewise-sparse.
    Pisa,
}

/// One transformer forward's video self-attention plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ltx2VideoAttn {
    Off,
    Sol { step: usize },
    Pisa { step: usize },
}

impl Ltx2Stage2Attn {
    pub fn at(self, step: usize) -> Ltx2VideoAttn {
        match self {
            Self::Off => Ltx2VideoAttn::Off,
            Self::Sol => Ltx2VideoAttn::Sol { step },
            Self::Pisa => Ltx2VideoAttn::Pisa { step },
        }
    }
}

/// Video self-attention. Sol / PISA layers call the published kernels; the
/// dense prefix and `--sol-stage2`/`--pisa-stage2` off stay on today's SDPA.
fn video_self_attn(
    attn: &Attention,
    h: &CudaTensor,
    rope: &DeviceRope,
    route: Ltx2VideoAttn,
    layer: usize,
) -> Result<CudaTensor> {
    let kernel = match route {
        Ltx2VideoAttn::Sol { step } => {
            match fastvideo_models::ltx2::route(step, layer).map_err(msg)? {
                fastvideo_models::ltx2::Ltx25SolRoute::Dense => VideoAttnKernel::Dense,
                fastvideo_models::ltx2::Ltx25SolRoute::Sol { tau } => VideoAttnKernel::Sol { tau },
            }
        }
        Ltx2VideoAttn::Pisa { step } => {
            match fastvideo_models::ltx2::pisa_route(step, layer).map_err(msg)? {
                fastvideo_models::ltx2::Ltx23PisaRoute::Dense => VideoAttnKernel::Dense,
                fastvideo_models::ltx2::Ltx23PisaRoute::Pisa { sparsity, .. } => {
                    VideoAttnKernel::Pisa { sparsity }
                }
            }
        }
        Ltx2VideoAttn::Off => VideoAttnKernel::Dense,
    };
    attn.forward_kernel(h, None, Some(rope), None, kernel)
}

/// One stream's half of a block.
struct StreamBlock {
    attn1: Attention,
    attn2: Attention,
    ff: FeedForward,
    /// `[6 or 9, dim]`: MSA/MLP AdaLN rows; optional rows 6..8 modulate text Q.
    scale_shift_table: CudaTensor,
    /// `[5, dim]`: a2v_scale, a2v_shift, v2a_scale, v2a_shift, gate.
    cross_table: CudaTensor,
    /// `[2, dim]` prompt K/V shift/scale when `cross_attn_mod`.
    prompt_table: Option<CudaTensor>,
    cross_attn_mod: bool,
}

struct Block {
    video: StreamBlock,
    audio: StreamBlock,
    audio_to_video: Attention,
    video_to_audio: Attention,
}

/// The per-step modulation every block adds its own tables to.
struct StepModulation {
    /// `[6 or 9, dim]` from `time_embed` / `audio_time_embed`.
    main: CudaTensor,
    /// `[4, dim]` from `av_cross_attn_*_scale_shift`.
    cross: CudaTensor,
    /// `[1, dim]` from the stream's a↔v gate embedder.
    gate: CudaTensor,
}

/// The four rotary tables of one request geometry, on the device.
pub struct Ropes {
    pub video: DeviceRope,
    pub audio: DeviceRope,
    pub cross_video: DeviceRope,
    pub cross_audio: DeviceRope,
}

impl Ropes {
    pub fn new(
        cfg: &Ltx2TransformerConfig,
        grid: [usize; 3],
        audio_tokens: usize,
        fps: f32,
    ) -> Result<Self> {
        Self::upload(&Ltx2RopeTables::new(cfg, grid, audio_tokens, fps))
    }

    pub fn upload(t: &Ltx2RopeTables) -> Result<Self> {
        Ok(Self {
            video: DeviceRope::upload(&t.video)?,
            audio: DeviceRope::upload(&t.audio)?,
            cross_video: DeviceRope::upload(&t.cross_video)?,
            cross_audio: DeviceRope::upload(&t.cross_audio)?,
        })
    }
}

/// The connector outputs after the DiT's caption projections: `[1, T, 4096]`
/// and `[1, T, 2048]`. Timestep-independent, so computed once per prompt.
pub struct TextConditioning {
    pub video: CudaTensor,
    pub audio: CudaTensor,
}

/// Called after each block with `(index, video, audio)`.
pub type BlockObserver<'a> = &'a mut dyn FnMut(usize, &CudaTensor, &CudaTensor) -> Result<()>;

/// Called with named intermediates, in the oracle's vocabulary
/// (`ltx2_oracle.py`, `install_taps`): `blockNN.{video,audio}.{attn1,attn2,av,ff}_{in,out}`
/// — a sub-layer's modulated-norm input and its output before the gate —
/// `…_after` for the residual stream right after it, and
/// `head.{video,audio}.{norm,modulated}`. Diagnostic only: it costs a closure
/// call per tap and, for the head, one extra LayerNorm.
pub type Probe<'a> = &'a mut dyn FnMut(&str, &CudaTensor) -> Result<()>;

/// A probe that may be absent, with the block's name prefix filled in.
struct Tap<'p, 'a> {
    probe: Option<&'p mut Probe<'a>>,
    block: usize,
}

impl Tap<'_, '_> {
    fn emit(&mut self, stream: &str, what: &str, t: &CudaTensor) -> Result<()> {
        match self.probe.as_mut() {
            Some(p) => p(&format!("block{:02}.{stream}.{what}", self.block), t),
            None => Ok(()),
        }
    }
}

pub struct Ltx2Transformer {
    cfg: Ltx2TransformerConfig,
    proj_in: Linear,
    audio_proj_in: Linear,
    caption_projection: Option<CaptionProjection>,
    audio_caption_projection: Option<CaptionProjection>,
    time_embed: AdaLnSingle,
    audio_time_embed: AdaLnSingle,
    prompt_adaln: Option<AdaLnSingle>,
    audio_prompt_adaln: Option<AdaLnSingle>,
    cross_video_scale_shift: AdaLnSingle,
    cross_audio_scale_shift: AdaLnSingle,
    cross_video_gate: AdaLnSingle,
    cross_audio_gate: AdaLnSingle,
    /// `[2, dim]`: shift, scale of the output head.
    scale_shift_table: CudaTensor,
    audio_scale_shift_table: CudaTensor,
    proj_out: Linear,
    audio_proj_out: Linear,
    /// Loaded when `use_keyframes_abs_pos_embedding`; keyframe pipelines only.
    #[expect(dead_code, reason = "T2AV forward does not take a keyframe mask yet")]
    keyframes_abs_pos_embedding: Option<CudaTensor>,
    blocks: Vec<Block>,
    ones_video: CudaTensor,
    ones_audio: CudaTensor,
    fbcache: Arc<Mutex<Option<LtxFbRuntime>>>,
    prune: Arc<Mutex<Option<LtxPruneRuntime>>>,
    stage1: Arc<Mutex<Option<LtxStage1Runtime>>>,
}

struct LtxFbRuntime {
    state: FbCache,
    armed: bool,
    signals: Vec<Option<CudaTensor>>,
    res_v: Vec<Option<CudaTensor>>,
    res_a: Vec<Option<CudaTensor>>,
    pending_v: Option<CudaTensor>,
    pending_a: Option<CudaTensor>,
    active_pass: Option<usize>,
}

struct LtxPruneRuntime {
    active: bool,
    step: Option<usize>,
    prev: Option<CudaTensor>,
}

struct LtxStage1Runtime {
    last_v: Option<CudaTensor>,
    last_a: Option<CudaTensor>,
}

fn table(map: &WeightMap, key: &str, rows: usize, dim: usize) -> Result<CudaTensor> {
    let mut t = cuda_tensor_shaped(map, key, &[rows, dim])?;
    t.pin_device()?;
    Ok(t)
}

impl Ltx2Transformer {
    /// Load every block onto the device (37.8 GB as bf16). `keys` names the
    /// checkpoint's layout: the diffusers `transformer/` folder or the single
    /// `ltx-2-19b-*.safetensors`.
    pub fn load(map: &WeightMap, keys: &Keys, cfg: &Ltx2TransformerConfig) -> Result<Self> {
        Self::load_blocks(map, keys, cfg, &(0..cfg.num_layers).collect::<Vec<_>>())
    }

    /// [`Self::load`] with only the listed blocks (in that order) — the globals
    /// are always loaded. For the key-manifest tests, which check the loader at
    /// the production config without materialising 19B parameters; a model
    /// loaded this way is not the model.
    pub(crate) fn load_blocks(
        map: &WeightMap,
        keys: &Keys,
        cfg: &Ltx2TransformerConfig,
        which: &[usize],
    ) -> Result<Self> {
        if cfg.norm_elementwise_affine
            || cfg.patch_size != 1
            || cfg.patch_size_t != 1
            || !cfg.attention_bias
            || !cfg.attention_out_bias
        {
            return Err(msg(
                "ltx2 dit: expected weightless block norms, 1x1x1 patches and biased attention",
            ));
        }
        if cfg.cross_attention_dim != cfg.inner_dim()
            || cfg.audio_cross_attention_dim != cfg.audio_inner_dim()
        {
            return Err(msg("ltx2 dit: cross_attention_dim must match each stream width (projected text or connector output)"));
        }
        let (dv, da, eps) = (cfg.inner_dim(), cfg.audio_inner_dim(), cfg.norm_eps as f32);
        let (hv, ha) = (cfg.num_attention_heads, cfg.audio_num_attention_heads);
        let video_dims = AttentionDims {
            query_dim: dv,
            context_dim: dv,
            heads: hv,
            head_dim: cfg.attention_head_dim,
        };
        let audio_dims = AttentionDims {
            query_dim: da,
            context_dim: da,
            heads: ha,
            head_dim: cfg.audio_attention_head_dim,
        };
        // Both directions attend in the audio head layout.
        let a2v_dims = AttentionDims {
            query_dim: dv,
            context_dim: da,
            ..audio_dims
        };
        let v2a_dims = AttentionDims {
            query_dim: da,
            context_dim: dv,
            ..audio_dims
        };
        let video_mod_rows = if cfg.cross_attn_mod { 9 } else { 6 };
        let audio_mod_rows = if cfg.audio_cross_attn_mod { 9 } else { 6 };
        let prompt_mod = cfg.cross_attn_mod || cfg.audio_cross_attn_mod;

        let mut blocks = Vec::with_capacity(which.len());
        for &i in which {
            if i >= cfg.num_layers {
                return Err(msg(format!(
                    "ltx2 dit: block {i} of a {}-block model",
                    cfg.num_layers
                )));
            }
            let p = format!("transformer_blocks.{i}");
            let attn = |name: &str, dims: AttentionDims, gated: bool| {
                Attention::load(map, keys, &format!("{p}.{name}"), dims, eps, gated)
            };
            let video_gated = cfg.gated_attn;
            let audio_gated = cfg.audio_gated_attn;
            blocks.push(Block {
                video: StreamBlock {
                    attn1: attn("attn1", video_dims, video_gated)?,
                    attn2: attn("attn2", video_dims, video_gated)?,
                    ff: FeedForward::load(
                        map,
                        keys,
                        &format!("{p}.ff"),
                        dv,
                        cfg.ff_inner_dim(),
                        cfg.ff_bias,
                    )?,
                    scale_shift_table: table(
                        map,
                        &keys.key(&format!("{p}.scale_shift_table")),
                        video_mod_rows,
                        dv,
                    )?,
                    cross_table: table(
                        map,
                        &keys.key(&format!("{p}.video_a2v_cross_attn_scale_shift_table")),
                        5,
                        dv,
                    )?,
                    prompt_table: if prompt_mod {
                        Some(table(
                            map,
                            &keys.key(&format!("{p}.prompt_scale_shift_table")),
                            2,
                            dv,
                        )?)
                    } else {
                        None
                    },
                    cross_attn_mod: cfg.cross_attn_mod,
                },
                audio: StreamBlock {
                    attn1: attn("audio_attn1", audio_dims, audio_gated)?,
                    attn2: attn("audio_attn2", audio_dims, audio_gated)?,
                    ff: FeedForward::load(
                        map,
                        keys,
                        &format!("{p}.audio_ff"),
                        da,
                        cfg.audio_ff_inner_dim(),
                        cfg.audio_ff_bias,
                    )?,
                    scale_shift_table: table(
                        map,
                        &keys.key(&format!("{p}.audio_scale_shift_table")),
                        audio_mod_rows,
                        da,
                    )?,
                    cross_table: table(
                        map,
                        &keys.key(&format!("{p}.audio_a2v_cross_attn_scale_shift_table")),
                        5,
                        da,
                    )?,
                    prompt_table: if prompt_mod {
                        Some(table(
                            map,
                            &keys.key(&format!("{p}.audio_prompt_scale_shift_table")),
                            2,
                            da,
                        )?)
                    } else {
                        None
                    },
                    cross_attn_mod: cfg.audio_cross_attn_mod,
                },
                audio_to_video: attn("audio_to_video_attn", a2v_dims, video_gated)?,
                video_to_audio: attn("video_to_audio_attn", v2a_dims, audio_gated)?,
            });
            if (i + 1) % 8 == 0 || i + 1 == cfg.num_layers {
                let free = crate::wan::device::free_memory()
                    .map_or(-1.0, |(f, _)| f as f64 / f64::from(1u32 << 30));
                crate::wan::log::info(format_args!(
                    "ltx2 dit: loaded block {}/{} ({free:.1} GiB free)",
                    i + 1,
                    cfg.num_layers
                ));
            }
        }
        let ada = |name: &str, dim: usize, rows: usize| {
            AdaLnSingle::load(map, keys, name, dim, rows, cfg.timestep_proj_dim)
        };
        let caption =
            |name: &str, caption: usize, dim: usize| -> Result<Option<CaptionProjection>> {
                if cfg.use_prompt_embeddings {
                    Ok(Some(CaptionProjection::load(
                        map, keys, name, caption, dim,
                    )?))
                } else {
                    Ok(None)
                }
            };
        let prompt_ada = |name: &str, dim: usize| -> Result<Option<AdaLnSingle>> {
            if prompt_mod && cfg.use_prompt_adaln_single {
                Ok(Some(ada(name, dim, 2)?))
            } else {
                Ok(None)
            }
        };
        let keyframes = if cfg.use_keyframes_abs_pos_embedding {
            let mut t =
                cuda_tensor_shaped(map, &keys.key("keyframes_abs_pos_embedding"), &[1, dv])?;
            t.pin_device()?;
            Some(t)
        } else {
            None
        };
        Ok(Self {
            proj_in: Linear::load(map, &keys.key("proj_in"), cfg.in_channels, dv, true)?,
            audio_proj_in: Linear::load(
                map,
                &keys.key("audio_proj_in"),
                cfg.audio_in_channels,
                da,
                true,
            )?,
            caption_projection: caption("caption_projection", cfg.caption_channels, dv)?,
            audio_caption_projection: caption(
                "audio_caption_projection",
                cfg.caption_channels,
                da,
            )?,
            time_embed: ada("time_embed", dv, video_mod_rows)?,
            audio_time_embed: ada("audio_time_embed", da, audio_mod_rows)?,
            prompt_adaln: prompt_ada("prompt_adaln", dv)?,
            audio_prompt_adaln: prompt_ada("audio_prompt_adaln", da)?,
            cross_video_scale_shift: ada("av_cross_attn_video_scale_shift", dv, 4)?,
            cross_audio_scale_shift: ada("av_cross_attn_audio_scale_shift", da, 4)?,
            cross_video_gate: ada("av_cross_attn_video_a2v_gate", dv, 1)?,
            cross_audio_gate: ada("av_cross_attn_audio_v2a_gate", da, 1)?,
            scale_shift_table: table(map, &keys.key("scale_shift_table"), 2, dv)?,
            audio_scale_shift_table: table(map, &keys.key("audio_scale_shift_table"), 2, da)?,
            proj_out: Linear::load(map, &keys.key("proj_out"), dv, cfg.out_channels, true)?,
            audio_proj_out: Linear::load(
                map,
                &keys.key("audio_proj_out"),
                da,
                cfg.audio_out_channels,
                true,
            )?,
            keyframes_abs_pos_embedding: keyframes,
            blocks,
            ones_video: ones(dv)?,
            ones_audio: ones(da)?,
            fbcache: Default::default(),
            prune: Default::default(),
            stage1: Default::default(),
            cfg: cfg.clone(),
        })
    }

    pub fn enable_fbcache(&self) {
        *self.fbcache.lock().expect("ltx2 fbcache") = Some(LtxFbRuntime {
            state: FbCache::official(),
            armed: true,
            signals: Vec::new(),
            res_v: Vec::new(),
            res_a: Vec::new(),
            pending_v: None,
            pending_a: None,
            active_pass: None,
        });
        crate::wan::log::info(format_args!("{}", fastvideo_models::ltx2::fbcache::APPLIED));
    }

    pub fn begin_fbcache_step(&self, step: usize) {
        let mut slot = self.fbcache.lock().expect("ltx2 fbcache");
        if let Some(runtime) = slot.as_mut() {
            if runtime.armed {
                runtime.state.begin_step(step);
            }
            // Do not set armed=true. After [`Self::disarm_fbcache`] the
            // runtime stays off for stage 2 and the Spark refiner.
        }
    }

    pub fn disarm_fbcache(&self) {
        let mut slot = self.fbcache.lock().expect("ltx2 fbcache");
        if let Some(runtime) = slot.as_mut() {
            runtime.armed = false;
        }
    }

    pub fn enable_midpoint_prune(&self) {
        *self.prune.lock().expect("ltx2 prune") = Some(LtxPruneRuntime {
            active: false,
            step: None,
            prev: None,
        });
        crate::wan::log::info(format_args!(
            "{}",
            fastvideo_models::ltx2::pisa::PRUNE_APPLIED
        ));
    }

    pub fn arm_prune_step(&self, step: usize) {
        let mut slot = self.prune.lock().expect("ltx2 prune");
        if let Some(runtime) = slot.as_mut() {
            runtime.step = Some(step);
        }
    }

    pub fn set_prune_active(&self, active: bool) {
        let mut slot = self.prune.lock().expect("ltx2 prune");
        if let Some(runtime) = slot.as_mut() {
            runtime.active = active;
        }
    }

    pub fn enable_stage1_cache(&self) {
        *self.stage1.lock().expect("ltx2 stage1") = Some(LtxStage1Runtime {
            last_v: None,
            last_a: None,
        });
        crate::wan::log::info(format_args!(
            "{}",
            fastvideo_models::ltx2::pisa::STAGE1_CACHE_APPLIED
        ));
    }

    pub fn stage1_reuse(&self, step: usize) -> Option<(CudaTensor, CudaTensor)> {
        if !fastvideo_models::ltx2::stage1_skips_step(step) {
            return None;
        }
        let slot = self.stage1.lock().expect("ltx2 stage1");
        let runtime = slot.as_ref()?;
        Some((runtime.last_v.clone()?, runtime.last_a.clone()?))
    }

    pub fn stage1_store(&self, video: &CudaTensor, audio: &CudaTensor) {
        let mut slot = self.stage1.lock().expect("ltx2 stage1");
        if let Some(runtime) = slot.as_mut() {
            runtime.last_v = Some(video.clone());
            runtime.last_a = Some(audio.clone());
        }
    }

    fn index_video_ropes(&self, ropes: &Ropes, tokens: &[usize]) -> Result<Ropes> {
        Ok(Ropes {
            video: ropes.video.index_tokens(tokens)?,
            audio: ropes.audio.clone(),
            cross_video: ropes.cross_video.index_tokens(tokens)?,
            cross_audio: ropes.cross_audio.clone(),
        })
    }

    fn gather_prune(&self, xv: &CudaTensor) -> Result<Option<(Vec<usize>, CudaTensor)>> {
        let mut slot = self.prune.lock().expect("ltx2 prune");
        let Some(runtime) = slot.as_mut() else {
            return Ok(None);
        };
        if !runtime.active {
            return Ok(None);
        }
        let Some(step) = runtime.step.take() else {
            return Ok(None);
        };
        if !fastvideo_models::ltx2::prunes_step(step) || runtime.prev.is_none() {
            return Ok(None);
        }
        let seq = xv.shape[1];
        let dim = xv.shape[2];
        let flat = xv.reshape(vec![seq, dim])?;
        let host = flat.host_cow()?.into_owned();
        let idx = fastvideo_models::ltx2::feat_norm_keep_indices(
            &host,
            seq,
            dim,
            fastvideo_models::ltx2::pisa::PRUNE_RATIO,
        );
        if idx.len() >= seq {
            return Ok(None);
        }
        let kept = xv
            .reshape(vec![seq, dim])?
            .index_select_rows(&idx)?
            .reshape(vec![1, idx.len(), dim])?;
        Ok(Some((idx, kept)))
    }

    fn scatter_prune(&self, kept: CudaTensor, idx: &[usize]) -> Result<CudaTensor> {
        let mut slot = self.prune.lock().expect("ltx2 prune");
        let runtime = slot.as_mut().expect("ltx2 prune");
        let prev = runtime.prev.as_ref().expect("ltx2 prune prev");
        let (seq, dim) = (prev.shape[1], prev.shape[2]);
        let prev_flat = prev.reshape(vec![seq, dim])?;
        let kept_flat = kept.reshape(vec![idx.len(), dim])?;
        let prev_host = prev_flat.host_cow()?.into_owned();
        let kept_host = kept_flat.host_cow()?.into_owned();
        let full = fastvideo_models::ltx2::scatter_prev(&prev_host, seq, dim, idx, &kept_host);
        let out = CudaTensor::from_vec(full, vec![1, seq, dim])?;
        runtime.prev = Some(out.clone());
        Ok(out)
    }

    fn store_prune_full(&self, xv: &CudaTensor) -> Result<()> {
        let mut slot = self.prune.lock().expect("ltx2 prune");
        if let Some(runtime) = slot.as_mut() {
            if runtime.active {
                runtime.prev = Some(xv.clone());
            }
        }
        Ok(())
    }

    fn begin_fb_pass(&self, xv: &CudaTensor, xa: &CudaTensor) -> Result<bool> {
        let mut slot = self.fbcache.lock().expect("ltx2 fbcache");
        let Some(runtime) = slot.as_mut() else {
            return Ok(false);
        };
        if !runtime.armed {
            return Ok(false);
        }
        runtime.pending_v = Some(xv.clone());
        runtime.pending_a = Some(xa.clone());
        Ok(true)
    }

    fn decide_fb_after_block0(&self, xv: &CudaTensor, xa: &CudaTensor) -> Result<bool> {
        let mut slot = self.fbcache.lock().expect("ltx2 fbcache");
        let runtime = slot.as_mut().expect("ltx2 fbcache");
        let v_in = runtime.pending_v.as_ref().expect("ltx2 fb v_in");
        let signal = xv.sub(v_in)?;
        let distance = if runtime.state.needs_signal() {
            let prev = runtime.signals[runtime.state.current_pass()]
                .as_ref()
                .expect("ltx2 fb signal");
            let cur = signal.host_cow()?;
            let prev = prev.host_cow()?;
            fastvideo_models::ltx2::fbcache::relative_l1(&cur, &prev)
        } else {
            0.0
        };
        let decision = runtime.state.decide(distance);
        let pass = runtime.state.last_pass();
        while runtime.signals.len() <= pass {
            runtime.signals.push(None);
            runtime.res_v.push(None);
            runtime.res_a.push(None);
        }
        runtime.signals[pass] = Some(signal);
        runtime.active_pass = Some(pass);
        if decision.skip {
            runtime.state.note_reused(pass);
            crate::wan::log::debug(format_args!(
                "ltx2 fbcache reuse step pass {pass} reason {}",
                decision.reason
            ));
            Ok(true)
        } else {
            let _ = xa;
            Ok(false)
        }
    }

    fn reuse_fb_residual(&self) -> Result<(CudaTensor, CudaTensor)> {
        let slot = self.fbcache.lock().expect("ltx2 fbcache");
        let runtime = slot.as_ref().expect("ltx2 fbcache");
        let pass = runtime.active_pass.expect("ltx2 fb pass");
        let v_in = runtime.pending_v.as_ref().expect("ltx2 fb v_in");
        let a_in = runtime.pending_a.as_ref().expect("ltx2 fb a_in");
        let xv = v_in.add(runtime.res_v[pass].as_ref().expect("ltx2 fb res_v"))?;
        let xa = a_in.add(runtime.res_a[pass].as_ref().expect("ltx2 fb res_a"))?;
        Ok((xv, xa))
    }

    fn finish_fb(&self, xv: &CudaTensor, xa: &CudaTensor) -> Result<()> {
        let mut slot = self.fbcache.lock().expect("ltx2 fbcache");
        let runtime = slot.as_mut().expect("ltx2 fbcache");
        let pass = runtime.active_pass.take().expect("ltx2 fb pass");
        let v_in = runtime.pending_v.take().expect("ltx2 fb v_in");
        let a_in = runtime.pending_a.take().expect("ltx2 fb a_in");
        while runtime.res_v.len() <= pass {
            runtime.res_v.push(None);
            runtime.res_a.push(None);
        }
        runtime.res_v[pass] = Some(xv.sub(&v_in)?);
        runtime.res_a[pass] = Some(xa.sub(&a_in)?);
        runtime.state.note_computed(pass);
        Ok(())
    }

    pub fn config(&self) -> &Ltx2TransformerConfig {
        &self.cfg
    }

    /// Connector text at `[1, T, caption_channels]` or, when
    /// `use_prompt_embeddings` is false, already at each stream's width.
    pub fn project_text(&self, video: &CudaTensor, audio: &CudaTensor) -> Result<TextConditioning> {
        Ok(TextConditioning {
            video: match &self.caption_projection {
                Some(p) => p.forward(video)?,
                None => video.clone(),
            },
            audio: match &self.audio_caption_projection {
                Some(p) => p.forward(audio)?,
                None => audio.clone(),
            },
        })
    }

    /// One joint forward. `video`: packed latents `[1, S, 128]`; `audio`:
    /// `[1, L, 128]`; `timestep` = `1000·sigma`, shared by both streams.
    /// Returns the two velocities, same shapes.
    pub fn forward(
        &self,
        video: &CudaTensor,
        audio: &CudaTensor,
        text: &TextConditioning,
        timestep: f32,
        ropes: &Ropes,
        observer: Option<BlockObserver<'_>>,
    ) -> Result<(CudaTensor, CudaTensor)> {
        self.forward_sol(
            video,
            audio,
            text,
            timestep,
            ropes,
            observer,
            Ltx2VideoAttn::Off,
        )
    }

    /// [`Self::forward`] with a stage-2 video self-attention plan.
    pub fn forward_sol(
        &self,
        video: &CudaTensor,
        audio: &CudaTensor,
        text: &TextConditioning,
        timestep: f32,
        ropes: &Ropes,
        observer: Option<BlockObserver<'_>>,
        route: Ltx2VideoAttn,
    ) -> Result<(CudaTensor, CudaTensor)> {
        self.forward_probed(video, audio, text, timestep, ropes, observer, None, route)
    }

    /// [`Self::forward`] with sub-layer taps handed to `probe`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_probed(
        &self,
        video: &CudaTensor,
        audio: &CudaTensor,
        text: &TextConditioning,
        timestep: f32,
        ropes: &Ropes,
        mut observer: Option<BlockObserver<'_>>,
        mut probe: Option<Probe<'_>>,
        route: Ltx2VideoAttn,
    ) -> Result<(CudaTensor, CudaTensor)> {
        if video.rank() != 3 || audio.rank() != 3 || video.shape[0] != 1 || audio.shape[0] != 1 {
            return Err(msg(format!(
                "ltx2 dit expects [1, S, C] and [1, L, C], got {:?} and {:?}",
                video.shape, audio.shape
            )));
        }
        let eps = self.cfg.norm_eps as f32;
        // The a↔v gate embedders see the timestep rescaled by
        // cross_attn_timestep_scale_multiplier / timestep_scale_multiplier (= 1).
        let gate_t = timestep
            * (self.cfg.cross_attn_timestep_scale_multiplier / self.cfg.timestep_scale_multiplier)
                as f32;
        let (v_main, v_embedded) = self.time_embed.forward(timestep)?;
        let (a_main, a_embedded) = self.audio_time_embed.forward(timestep)?;
        let v_mod = StepModulation {
            main: v_main,
            cross: self.cross_video_scale_shift.forward(timestep)?.0,
            gate: self.cross_video_gate.forward(gate_t)?.0,
        };
        let a_mod = StepModulation {
            main: a_main,
            cross: self.cross_audio_scale_shift.forward(timestep)?.0,
            gate: self.cross_audio_gate.forward(gate_t)?.0,
        };
        let (v_prompt, a_prompt) = match (&self.prompt_adaln, &self.audio_prompt_adaln) {
            (Some(p), Some(a)) => (Some(p.forward(timestep)?.0), Some(a.forward(timestep)?.0)),
            _ => (None, None),
        };

        let mut xv = self.proj_in.forward(video)?;
        let mut xa = self.audio_proj_in.forward(audio)?;
        let pruned = self.gather_prune(&xv)?;
        let pruned_ropes = if let Some((idx, kept)) = &pruned {
            xv = kept.clone();
            Some(self.index_video_ropes(ropes, idx)?)
        } else {
            None
        };
        let ropes = pruned_ropes.as_ref().unwrap_or(ropes);
        let fb = self.begin_fb_pass(&xv, &xa)?;
        let mut skipped = false;
        for (i, block) in self.blocks.iter().enumerate() {
            let tap = Tap {
                probe: probe.as_mut(),
                block: i,
            };
            (xv, xa) = self.block(
                block,
                xv,
                xa,
                text,
                &v_mod,
                &a_mod,
                v_prompt.as_ref(),
                a_prompt.as_ref(),
                ropes,
                eps,
                tap,
                route,
            )?;
            if let Some(obs) = observer.as_mut() {
                obs(i, &xv, &xa)?;
            }
            if fb && i == 0 && self.decide_fb_after_block0(&xv, &xa)? {
                skipped = true;
                (xv, xa) = self.reuse_fb_residual()?;
                break;
            }
        }
        if fb && !skipped {
            self.finish_fb(&xv, &xa)?;
        }
        if let Some((idx, _)) = pruned {
            xv = self.scatter_prune(xv, &idx)?;
        } else {
            self.store_prune_full(&xv)?;
        }
        let mut head = |stream: &str,
                        x: &CudaTensor,
                        tab: &CudaTensor,
                        e: &CudaTensor,
                        out: &Linear|
         -> Result<CudaTensor> {
            // (shift, scale) = table[2, D] + embedded; LayerNorm here, not RMSNorm.
            let dim = e.shape[1];
            let mods = tab.add(e)?.reshape(vec![1, 2, dim])?;
            let modulated = x.ln_adaln_e(&mods, 1, 0, eps)?;
            if let Some(p) = probe.as_mut() {
                p(
                    &format!("head.{stream}.norm"),
                    &x.layer_norm(eps, None, None)?,
                )?;
                p(&format!("head.{stream}.modulated"), &modulated)?;
            }
            out.forward(&modulated)
        };
        Ok((
            head(
                "video",
                &xv,
                &self.scale_shift_table,
                &v_embedded,
                &self.proj_out,
            )?,
            head(
                "audio",
                &xa,
                &self.audio_scale_shift_table,
                &a_embedded,
                &self.audio_proj_out,
            )?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        b: &Block,
        xv: CudaTensor,
        xa: CudaTensor,
        text: &TextConditioning,
        v_mod: &StepModulation,
        a_mod: &StepModulation,
        v_prompt: Option<&CudaTensor>,
        a_prompt: Option<&CudaTensor>,
        ropes: &Ropes,
        eps: f32,
        mut tap: Tap<'_, '_>,
        route: Ltx2VideoAttn,
    ) -> Result<(CudaTensor, CudaTensor)> {
        let (dv, da) = (xv.shape[2], xa.shape[2]);
        // table + per-step modulation, kept as [1, rows, D] for the gated adds.
        // LTX-2.0: 6 rows; LTX-2.5 (`cross_attn_mod`): 9 rows (extra text-Q AdaLN + gate).
        let v_tab = b.video.scale_shift_table.add(&v_mod.main)?;
        let a_tab = b.audio.scale_shift_table.add(&a_mod.main)?;
        let (v_rows, a_rows) = (v_tab.shape[0], a_tab.shape[0]);
        let (v_gates, a_gates) = (
            v_tab.reshape(vec![1, v_rows, dv])?,
            a_tab.reshape(vec![1, a_rows, da])?,
        );

        // 1. self-attention.
        let h = rms_adaln(&xv, &row(&v_tab, 1)?, &row(&v_tab, 0)?, eps)?;
        let u = video_self_attn(&b.video.attn1, &h, &ropes.video, route, tap.block)?;
        let xv = xv.residual_gate_add_e(&u, &v_gates, 2)?;
        tap.emit("video", "attn1_in", &h)?;
        tap.emit("video", "attn1_out", &u)?;
        tap.emit("video", "attn1_after", &xv)?;
        let h = rms_adaln(&xa, &row(&a_tab, 1)?, &row(&a_tab, 0)?, eps)?;
        let u = b.audio.attn1.forward(&h, None, Some(&ropes.audio), None)?;
        let xa = xa.residual_gate_add_e(&u, &a_gates, 2)?;
        tap.emit("audio", "attn1_in", &h)?;
        tap.emit("audio", "attn1_out", &u)?;
        tap.emit("audio", "attn1_after", &xa)?;

        // 2. text cross-attention (optional Q/KV AdaLN + output gate, LTX-2.5).
        let text_cross = |stream: &StreamBlock,
                          x: &CudaTensor,
                          ctx: &CudaTensor,
                          tab: &CudaTensor,
                          ones: &CudaTensor,
                          prompt: Option<&CudaTensor>,
                          dim: usize|
         -> Result<(CudaTensor, CudaTensor, CudaTensor)> {
            let mut h = x.rms_norm(ones, eps)?;
            if stream.cross_attn_mod {
                h = scale_shift(&h, &row(tab, 7)?, &row(tab, 6)?)?;
            }
            let mut enc = ctx.clone();
            if let Some(pt) = &stream.prompt_table {
                let tab_p = match prompt {
                    Some(t) => pt.add(t)?,
                    None => pt.clone(),
                };
                enc = scale_shift(&enc, &row(&tab_p, 1)?, &row(&tab_p, 0)?)?;
            }
            let u = stream.attn2.forward(&h, Some(&enc), None, None)?;
            let u = if stream.cross_attn_mod {
                u.mul(&row(tab, 8)?.reshape(vec![1, 1, dim])?)?
            } else {
                u
            };
            let out = x.add(&u)?;
            Ok((h, u, out))
        };
        let (h, u, xv) = text_cross(
            &b.video,
            &xv,
            &text.video,
            &v_tab,
            &self.ones_video,
            v_prompt,
            dv,
        )?;
        tap.emit("video", "attn2_in", &h)?;
        tap.emit("video", "attn2_out", &u)?;
        tap.emit("video", "attn2_after", &xv)?;
        let (h, u, xa) = text_cross(
            &b.audio,
            &xa,
            &text.audio,
            &a_tab,
            &self.ones_audio,
            a_prompt,
            da,
        )?;
        tap.emit("audio", "attn2_in", &h)?;
        tap.emit("audio", "attn2_out", &u)?;
        tap.emit("audio", "attn2_after", &xa)?;

        // 3. audio↔video, both directions from the same pre-update states.
        // Rows 0..3 of each side's table: a2v_scale, a2v_shift, v2a_scale, v2a_shift
        // (scale first here); row 4 is the gate, modulated by its own embedder.
        let v_cross = b.video.cross_table.narrow(0, 0, 4)?.add(&v_mod.cross)?;
        let a_cross = b.audio.cross_table.narrow(0, 0, 4)?.add(&a_mod.cross)?;
        let a2v_gate = row(&b.video.cross_table, 4)?
            .add(&v_mod.gate)?
            .reshape(vec![1, 1, dv])?;
        let v2a_gate = row(&b.audio.cross_table, 4)?
            .add(&a_mod.gate)?
            .reshape(vec![1, 1, da])?;
        let side = |x: &CudaTensor, cross: &CudaTensor, first: usize| {
            rms_adaln(x, &row(cross, first)?, &row(cross, first + 1)?, eps)
        };
        let (a2v_q, v2a_q) = (side(&xv, &v_cross, 0)?, side(&xa, &a_cross, 2)?);
        let a2v = b.audio_to_video.forward(
            &a2v_q,
            Some(&side(&xa, &a_cross, 0)?),
            Some(&ropes.cross_video),
            Some(&ropes.cross_audio),
        )?;
        let v2a = b.video_to_audio.forward(
            &v2a_q,
            Some(&side(&xv, &v_cross, 2)?),
            Some(&ropes.cross_audio),
            Some(&ropes.cross_video),
        )?;
        let xv = xv.residual_gate_add_e(&a2v, &a2v_gate, 0)?;
        let xa = xa.residual_gate_add_e(&v2a, &v2a_gate, 0)?;
        tap.emit("video", "av_in", &a2v_q)?;
        tap.emit("video", "av_out", &a2v)?;
        tap.emit("video", "av_after", &xv)?;
        tap.emit("audio", "av_in", &v2a_q)?;
        tap.emit("audio", "av_out", &v2a)?;
        tap.emit("audio", "av_after", &xa)?;

        // 4. feed-forward.
        let h = rms_adaln(&xv, &row(&v_tab, 4)?, &row(&v_tab, 3)?, eps)?;
        let u = b.video.ff.forward(&h)?;
        let xv = xv.residual_gate_add_e(&u, &v_gates, 5)?;
        tap.emit("video", "ff_in", &h)?;
        tap.emit("video", "ff_out", &u)?;
        let h = rms_adaln(&xa, &row(&a_tab, 4)?, &row(&a_tab, 3)?, eps)?;
        let u = b.audio.ff.forward(&h)?;
        let xa = xa.residual_gate_add_e(&u, &a_gates, 5)?;
        tap.emit("audio", "ff_in", &h)?;
        tap.emit("audio", "ff_out", &u)?;
        Ok((xv, xa))
    }
}

/// `[1, C, F, H, W]` → `[1, F·H·W, C]`, tokens frame-major, then row, then column.
pub fn pack_video(latents: &CudaTensor) -> Result<CudaTensor> {
    let [b, c, f, h, w] = latents.shape[..] else {
        return Err(msg(format!(
            "pack_video expects [1, C, F, H, W], got {:?}",
            latents.shape
        )));
    };
    latents.reshape(vec![b, c, f * h * w])?.permute(&[0, 2, 1])
}

/// The inverse of [`pack_video`] for a `[frames, height, width]` grid.
pub fn unpack_video(tokens: &CudaTensor, grid: [usize; 3]) -> Result<CudaTensor> {
    let [b, s, c] = tokens.shape[..] else {
        return Err(msg(format!(
            "unpack_video expects [1, S, C], got {:?}",
            tokens.shape
        )));
    };
    let [f, h, w] = grid;
    if s != f * h * w {
        return Err(msg(format!(
            "unpack_video: {s} tokens for a {f}x{h}x{w} grid"
        )));
    }
    tokens.permute(&[0, 2, 1])?.reshape(vec![b, c, f, h, w])
}

#[cfg(test)]
mod tests {
    use super::super::attention::tests::{
        assert_close, attention_reference, get, linear, rms, rows, tensor, tokens, weights,
    };
    use super::super::keys::Layout;
    use super::*;

    fn tiny() -> Ltx2TransformerConfig {
        Ltx2TransformerConfig {
            in_channels: 6,
            out_channels: 6,
            num_attention_heads: 2,
            attention_head_dim: 8,
            cross_attention_dim: 16,
            audio_in_channels: 5,
            audio_out_channels: 5,
            audio_num_attention_heads: 2,
            audio_attention_head_dim: 4,
            audio_cross_attention_dim: 8,
            num_layers: 2,
            caption_channels: 12,
            timestep_proj_dim: 8,
            ..Ltx2TransformerConfig::ltx2_19b()
        }
    }

    fn add(a: &mut [f32], b: &[f32]) {
        a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
    }

    fn gelu(v: f32) -> f32 {
        0.5 * v * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044_715 * v * v * v)).tanh())
    }

    fn silu(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }

    fn lin(map: &WeightMap, prefix: &str, x: &[f32], o: usize) -> Vec<f32> {
        linear(
            x,
            &get(map, &format!("{prefix}.weight"), &[o, x.len()]),
            &get(map, &format!("{prefix}.bias"), &[o]),
        )
    }

    /// `(modulation rows, embedded)` of an AdaLN-single, from its definition.
    fn adaln(
        map: &WeightMap,
        name: &str,
        t: f32,
        dim: usize,
        rows_n: usize,
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        let half = 4;
        let s: Vec<f32> = (0..8)
            .map(|i| {
                let arg = t * (-(10000f32.ln()) * (i % half) as f32 / half as f32).exp();
                if i < half {
                    arg.cos()
                } else {
                    arg.sin()
                }
            })
            .collect();
        let h: Vec<f32> = lin(
            map,
            &format!("{name}.emb.timestep_embedder.linear_1"),
            &s,
            dim,
        )
        .into_iter()
        .map(silu)
        .collect();
        let e = lin(
            map,
            &format!("{name}.emb.timestep_embedder.linear_2"),
            &h,
            dim,
        );
        let m = lin(
            map,
            &format!("{name}.linear"),
            &e.iter().map(|v| silu(*v)).collect::<Vec<_>>(),
            rows_n * dim,
        );
        (m.chunks_exact(dim).map(<[f32]>::to_vec).collect(), e)
    }

    fn table_plus(
        map: &WeightMap,
        key: &str,
        rows_n: usize,
        dim: usize,
        m: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let t = get(map, key, &[rows_n, dim]);
        (0..m.len())
            .map(|r| {
                t[r * dim..(r + 1) * dim]
                    .iter()
                    .zip(&m[r])
                    .map(|(a, b)| a + b)
                    .collect()
            })
            .collect()
    }

    fn adaln_norm(x: &[Vec<f32>], scale: &[f32], shift: &[f32]) -> Vec<Vec<f32>> {
        x.iter()
            .map(|v| {
                rms(v, None, 1e-6)
                    .iter()
                    .enumerate()
                    .map(|(i, n)| n * (1.0 + scale[i]) + shift[i])
                    .collect()
            })
            .collect()
    }

    fn gated_add(x: &mut [Vec<f32>], update: &[Vec<f32>], gate: &[f32]) {
        for (v, u) in x.iter_mut().zip(update) {
            v.iter_mut()
                .enumerate()
                .for_each(|(i, a)| *a += u[i] * gate[i]);
        }
    }

    fn ff(map: &WeightMap, prefix: &str, x: &[Vec<f32>], dim: usize) -> Vec<Vec<f32>> {
        x.iter()
            .map(|v| {
                let up: Vec<f32> = lin(map, &format!("{prefix}.net.0.proj"), v, dim * 4)
                    .into_iter()
                    .map(gelu)
                    .collect();
                lin(map, &format!("{prefix}.net.2"), &up, dim)
            })
            .collect()
    }

    /// The whole model against the block order of docs/ports/ltx2.md §e written
    /// as loops: every modulation row, both a↔v directions from pre-update
    /// states, the four rotary tables, the LayerNorm heads.
    #[test]
    fn forward_matches_a_loop_reference() {
        let cfg = tiny();
        let map = weights();
        let model =
            Ltx2Transformer::load(&map, &Keys::transformer(Layout::Diffusers), &cfg).unwrap();
        let (grid, l, t_len, timestep) = ([2usize, 1, 3], 4usize, 5usize, 725.0f32);
        let s = 6;
        let tables = Ltx2RopeTables::new(&cfg, grid, l, 24.0);
        let ropes = Ropes::upload(&tables).unwrap();
        let (video, audio) = (tokens(s, 6, 0.41), tokens(l, 5, 0.83));
        let (ctx_v, ctx_a) = (tokens(t_len, 12, 0.29), tokens(t_len, 12, 0.57));
        let text = model
            .project_text(&tensor(&ctx_v), &tensor(&ctx_a))
            .unwrap();
        let mut seen = Vec::new();
        let mut obs = |i: usize, v: &CudaTensor, a: &CudaTensor| -> Result<()> {
            seen.push((i, rows(v, 16), rows(a, 8)));
            Ok(())
        };
        let (got_v, got_a) = model
            .forward(
                &tensor(&video),
                &tensor(&audio),
                &text,
                timestep,
                &ropes,
                Some(&mut obs),
            )
            .unwrap();
        assert_eq!(
            (got_v.shape.clone(), got_a.shape.clone()),
            (vec![1, s, 6], vec![1, l, 5])
        );
        assert_eq!(seen.len(), 2, "one observation per block");

        let (dv, da) = (16usize, 8usize);
        let caption = |name: &str, x: &[Vec<f32>], d: usize| -> Vec<Vec<f32>> {
            x.iter()
                .map(|v| {
                    lin(
                        &map,
                        &format!("{name}.linear_2"),
                        &lin(&map, &format!("{name}.linear_1"), v, d)
                            .into_iter()
                            .map(gelu)
                            .collect::<Vec<_>>(),
                        d,
                    )
                })
                .collect()
        };
        let (tv, ta) = (
            caption("caption_projection", &ctx_v, dv),
            caption("audio_caption_projection", &ctx_a, da),
        );
        let (v_main, v_emb) = adaln(&map, "time_embed", timestep, dv, 6);
        let (a_main, a_emb) = adaln(&map, "audio_time_embed", timestep, da, 6);
        let v_cross = adaln(&map, "av_cross_attn_video_scale_shift", timestep, dv, 4).0;
        let a_cross = adaln(&map, "av_cross_attn_audio_scale_shift", timestep, da, 4).0;
        let v_gate = adaln(&map, "av_cross_attn_video_a2v_gate", timestep, dv, 1).0;
        let a_gate = adaln(&map, "av_cross_attn_audio_v2a_gate", timestep, da, 1).0;

        let mut xv: Vec<Vec<f32>> = video.iter().map(|v| lin(&map, "proj_in", v, dv)).collect();
        let mut xa: Vec<Vec<f32>> = audio
            .iter()
            .map(|v| lin(&map, "audio_proj_in", v, da))
            .collect();
        let vd = AttentionDims {
            query_dim: dv,
            context_dim: dv,
            heads: 2,
            head_dim: 8,
        };
        let ad = AttentionDims {
            query_dim: da,
            context_dim: da,
            heads: 2,
            head_dim: 4,
        };
        for (i, tap) in seen.iter().enumerate() {
            let p = format!("transformer_blocks.{i}");
            let vt = table_plus(&map, &format!("{p}.scale_shift_table"), 6, dv, &v_main);
            let at = table_plus(
                &map,
                &format!("{p}.audio_scale_shift_table"),
                6,
                da,
                &a_main,
            );
            // 1. self-attention: rows shift, scale, gate.
            let h = adaln_norm(&xv, &vt[1], &vt[0]);
            let u = attention_reference(
                &map,
                &format!("{p}.attn1"),
                vd,
                &h,
                &h,
                Some(&tables.video),
                None,
                false,
            );
            gated_add(&mut xv, &u, &vt[2]);
            let h = adaln_norm(&xa, &at[1], &at[0]);
            let u = attention_reference(
                &map,
                &format!("{p}.audio_attn1"),
                ad,
                &h,
                &h,
                Some(&tables.audio),
                None,
                false,
            );
            gated_add(&mut xa, &u, &at[2]);
            // 2. text cross-attention.
            let h: Vec<Vec<f32>> = xv.iter().map(|v| rms(v, None, 1e-6)).collect();
            let u =
                attention_reference(&map, &format!("{p}.attn2"), vd, &h, &tv, None, None, false);
            xv.iter_mut().zip(&u).for_each(|(a, b)| add(a, b));
            let h: Vec<Vec<f32>> = xa.iter().map(|v| rms(v, None, 1e-6)).collect();
            let u = attention_reference(
                &map,
                &format!("{p}.audio_attn2"),
                ad,
                &h,
                &ta,
                None,
                None,
                false,
            );
            xa.iter_mut().zip(&u).for_each(|(a, b)| add(a, b));
            // 3. a↔v from the same pre-update states; rows scale, shift per direction.
            let vc = table_plus(
                &map,
                &format!("{p}.video_a2v_cross_attn_scale_shift_table"),
                5,
                dv,
                &v_cross,
            );
            let ac = table_plus(
                &map,
                &format!("{p}.audio_a2v_cross_attn_scale_shift_table"),
                5,
                da,
                &a_cross,
            );
            let g_v: Vec<f32> = get(
                &map,
                &format!("{p}.video_a2v_cross_attn_scale_shift_table"),
                &[5, dv],
            )[4 * dv..]
                .iter()
                .zip(&v_gate[0])
                .map(|(a, b)| a + b)
                .collect();
            let g_a: Vec<f32> = get(
                &map,
                &format!("{p}.audio_a2v_cross_attn_scale_shift_table"),
                &[5, da],
            )[4 * da..]
                .iter()
                .zip(&a_gate[0])
                .map(|(a, b)| a + b)
                .collect();
            let a2v = attention_reference(
                &map,
                &format!("{p}.audio_to_video_attn"),
                AttentionDims {
                    query_dim: dv,
                    context_dim: da,
                    ..ad
                },
                &adaln_norm(&xv, &vc[0], &vc[1]),
                &adaln_norm(&xa, &ac[0], &ac[1]),
                Some(&tables.cross_video),
                Some(&tables.cross_audio),
                false,
            );
            let v2a = attention_reference(
                &map,
                &format!("{p}.video_to_audio_attn"),
                AttentionDims {
                    query_dim: da,
                    context_dim: dv,
                    ..ad
                },
                &adaln_norm(&xa, &ac[2], &ac[3]),
                &adaln_norm(&xv, &vc[2], &vc[3]),
                Some(&tables.cross_audio),
                Some(&tables.cross_video),
                false,
            );
            gated_add(&mut xv, &a2v, &g_v);
            gated_add(&mut xa, &v2a, &g_a);
            // 4. feed-forward.
            let u = ff(
                &map,
                &format!("{p}.ff"),
                &adaln_norm(&xv, &vt[4], &vt[3]),
                dv,
            );
            gated_add(&mut xv, &u, &vt[5]);
            let u = ff(
                &map,
                &format!("{p}.audio_ff"),
                &adaln_norm(&xa, &at[4], &at[3]),
                da,
            );
            gated_add(&mut xa, &u, &at[5]);
            assert_eq!(tap.0, i);
            assert_close(&tap.1, &xv, 2e-4, &format!("block {i} video"));
            assert_close(&tap.2, &xa, 2e-4, &format!("block {i} audio"));
        }
        let head = |x: &[Vec<f32>],
                    tab: &str,
                    e: &[f32],
                    out: &str,
                    d: usize,
                    c: usize|
         -> Vec<Vec<f32>> {
            let t = get(&map, tab, &[2, d]);
            x.iter()
                .map(|v| {
                    let mean = v.iter().sum::<f32>() / d as f32;
                    let var = v.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / d as f32;
                    let n: Vec<f32> = v
                        .iter()
                        .enumerate()
                        .map(|(i, a)| {
                            (a - mean) / (var + 1e-6).sqrt() * (1.0 + t[d + i] + e[i]) + t[i] + e[i]
                        })
                        .collect();
                    lin(&map, out, &n, c)
                })
                .collect()
        };
        assert_close(
            &rows(&got_v, 6),
            &head(&xv, "scale_shift_table", &v_emb, "proj_out", dv, 6),
            2e-4,
            "video velocity",
        );
        assert_close(
            &rows(&got_a, 5),
            &head(
                &xa,
                "audio_scale_shift_table",
                &a_emb,
                "audio_proj_out",
                da,
                5,
            ),
            2e-4,
            "audio velocity",
        );
    }

    #[test]
    fn packing_is_frame_major_then_row_then_column() {
        let z =
            CudaTensor::from_vec((0..12).map(|i| i as f32).collect(), vec![1, 2, 2, 1, 3]).unwrap();
        let p = pack_video(&z).unwrap();
        assert_eq!(p.shape, vec![1, 6, 2]);
        // Token (f=1, h=0, w=2) = index 5: channel 0 → 5, channel 1 → 11.
        assert_eq!(&p.host_cow().unwrap()[10..12], &[5.0, 11.0]);
        let back = unpack_video(&p, [2, 1, 3]).unwrap();
        assert_eq!(&*back.host_cow().unwrap(), &*z.host_cow().unwrap());
        assert!(unpack_video(&p, [2, 2, 3]).is_err());
    }

    fn tiny_ltx25() -> Ltx2TransformerConfig {
        Ltx2TransformerConfig {
            in_channels: 6,
            out_channels: 6,
            num_attention_heads: 2,
            attention_head_dim: 8,
            cross_attention_dim: 16,
            audio_in_channels: 5,
            audio_out_channels: 5,
            audio_num_attention_heads: 2,
            audio_attention_head_dim: 4,
            audio_cross_attention_dim: 8,
            num_layers: 1,
            caption_channels: 12,
            timestep_proj_dim: 8,
            ff_bias: false,
            ..Ltx2TransformerConfig::ltx2_5_22b()
        }
    }

    #[test]
    fn load_blocks_accepts_ltx2_5_flags_on_one_block() {
        let cfg = tiny_ltx25();
        let model = Ltx2Transformer::load_blocks(
            &weights(),
            &Keys::transformer(Layout::Diffusers),
            &cfg,
            &[0],
        )
        .unwrap();
        assert_eq!(model.blocks.len(), 1);
        assert!(model.caption_projection.is_none());
        assert!(model.blocks[0].video.prompt_table.is_some());
        assert_eq!(model.blocks[0].video.scale_shift_table.shape, [9, 16]);
    }
}
