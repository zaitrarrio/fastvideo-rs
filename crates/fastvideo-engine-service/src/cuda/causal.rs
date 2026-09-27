//! SF-Wan causal sessions over the E6 open-ended rollout
//! (`fastvideo_cudarc::wan::stream::CausalRollout`, design §5.4).
//!
//! One session at a time per pipeline (the scheduler's exclusive causal
//! lease): the rollout keeps its KV window, TAEHV decode state and RoPE
//! cache on the shared transformer. `causal_open` encodes the prompt and
//! sets up an empty cache; each `causal_block` generates and decodes one
//! block (block 0: 9 frames, then 12) with a prompt change applied at the
//! block boundary (`PromptSwitch::Keep`) and `reset` restarting at block 0;
//! `causal_close` drops the rollout.
//!
//! The rollout borrows its pipeline, so the SF-Wan pipeline is loaded once
//! and kept for the process lifetime (`&'static`, never unloaded).

use std::path::PathBuf;

use fastvideo_cudarc::wan::stream::{CausalRollout, PromptSwitch, RolloutConfig, RopePolicy};
use fastvideo_cudarc::WanPipeline;
use fastvideo_protocol::{ApiError, RgbFrame};

use super::caps::SfWanRecipe;
use super::output::api_err;
use crate::backend::{BlockInput, BlockStats, CausalSpec, ClipSink, SessionId};
use crate::cancel::StepControl;

/// The resident SF-Wan pipeline and its (at most one) open session.
pub struct SfWanModel {
    pipe: &'static WanPipeline,
    recipe: SfWanRecipe,
    text_cache: Option<PathBuf>,
    open: Option<Session>,
}

struct Session {
    id: SessionId,
    rollout: CausalRollout<'static>,
    prompt: String,
}

impl SfWanModel {
    /// Wraps a pipeline that lives for the rest of the process.
    pub fn new(
        pipe: &'static WanPipeline,
        recipe: SfWanRecipe,
        text_cache: Option<PathBuf>,
    ) -> Self {
        Self {
            pipe,
            recipe,
            text_cache,
            open: None,
        }
    }

    pub fn pipeline(&self) -> &'static WanPipeline {
        self.pipe
    }

    /// Whether session `s` is the open one.
    pub fn is_open(&self, s: SessionId) -> bool {
        self.open.as_ref().is_some_and(|o| o.id == s)
    }

    pub fn recipe(&self) -> &SfWanRecipe {
        &self.recipe
    }

    fn rollout_config(&self, spec: &CausalSpec) -> RolloutConfig {
        let w = &self.recipe.wan;
        let dmd = super::wan::dmd_steps(w.sampler.steps() as usize);
        RolloutConfig {
            prompt: spec.prompt.clone(),
            height: spec.height as usize,
            width: spec.width as usize,
            seed: spec.seed,
            dmd_steps: if dmd.len() == 4 {
                fastvideo_models::schedulers::SF_WAN_1_3B_DMD_STEPS.to_vec()
            } else {
                dmd
            },
            flow_shift: w.flow_shift,
            local_attn_frames: self.recipe.local_attn_frames as usize,
            sink_frames: self.recipe.sink_frames as usize,
            rope: RopePolicy::Relativistic,
            prompt_switch: PromptSwitch::Keep,
            rgb8: true,
            tokenizer_path: Some(
                w.weights
                    .join("tokenizer")
                    .join("tokenizer.json")
                    .to_string_lossy()
                    .into_owned(),
            ),
            text_cache: self.text_cache.clone(),
        }
    }

    pub fn open(&mut self, s: SessionId, spec: &CausalSpec) -> Result<(), ApiError> {
        if let Some(o) = &self.open {
            if o.id != s {
                return Err(ApiError::conflict(format!(
                    "causal session {} is open on this GPU",
                    o.id
                )));
            }
            return Ok(());
        }
        let rollout = CausalRollout::open(self.pipe, self.rollout_config(spec))
            .map_err(|e| api_err("causal open", e))?;
        self.open = Some(Session {
            id: s,
            rollout,
            prompt: spec.prompt.clone(),
        });
        Ok(())
    }

    pub fn block(
        &mut self,
        s: SessionId,
        input: &BlockInput,
        out: &mut dyn ClipSink,
        ctl: &StepControl,
    ) -> Result<BlockStats, ApiError> {
        ctl.check()?;
        let sess = match &mut self.open {
            Some(o) if o.id == s => o,
            _ => return Err(ApiError::invalid(format!("causal session {s} is not open"))),
        };
        if input.reset {
            sess.rollout.reset(Some(input.seed));
        }
        if input.prompt != sess.prompt {
            ctl.stage("text");
            sess.rollout
                .set_prompt(&input.prompt)
                .map_err(|e| api_err("causal prompt", e))?;
            sess.prompt = input.prompt.clone();
        }
        ctl.stage("denoise");
        let block = sess
            .rollout
            .next_block()
            .map_err(|e| api_err("causal block", e))?;
        let ms = block.timings.total_s * 1e3;
        let host = block.into_host();
        let (w, h) = (host.width as u32, host.height as u32);
        let size = (w * h * 3) as usize;
        let frames: Vec<RgbFrame> = host
            .rgb
            .chunks_exact(size)
            .enumerate()
            .map(|(i, px)| RgbFrame {
                width: w,
                height: h,
                data: px.to_vec().into(),
                index: (host.first_frame + i) as u64,
            })
            .collect();
        out.frames(&frames);
        Ok(BlockStats {
            block_index: input.block_index,
            frames: frames.len() as u32,
            block_ms: ms,
        })
    }

    pub fn close(&mut self, s: SessionId) {
        if self.open.as_ref().is_some_and(|o| o.id == s) {
            self.open = None;
            let _ = fastvideo_cudarc::wan::device::trim_pool();
        }
    }
}
