//! Cancellation and progress hooks for the video pipelines (serve package E1).
//!
//! [`Hooks`] is what a caller such as `fastvideo-engine-service` hands to
//! `H3Pipeline::generate_with_hooks`, `Ltx2Pipeline::generate_with_hooks` and
//! `WanPipeline::generate_to_with_hooks`. The pipelines report a [`Progress`]
//! event at every stage boundary, after every denoise step (with the causal
//! block for the SF-Wan block rollout) and after every decoded video chunk,
//! and they check the [`CancelToken`] at the same points. A tripped token
//! unwinds the run with [`PipelineError::Cancelled`] at the next such point,
//! so a cancel lands within one denoise step (or one decode chunk).
//!
//! [`Hooks::NONE`] (what the plain `generate` entry points pass) does nothing:
//! no callback, no atomic load, and no device work, so the frames of an
//! un-hooked run are exactly those of a run before this module existed. The
//! hooks never touch the device either way; a hooked run's frames are
//! identical too.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::wan::pipeline::{PipelineError, Result};

/// A shared cancel flag. Clones share the flag; any holder may trip it from
/// any thread, and the pipeline sees it at its next check.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// A token over an existing flag (a caller's own cancel type can keep the
    /// `Arc` and hand the pipeline a view of it).
    pub fn from_flag(flag: Arc<AtomicBool>) -> Self {
        Self(flag)
    }

    pub fn flag(&self) -> &Arc<AtomicBool> {
        &self.0
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// The phase a [`Progress`] event belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Prompt encoding (and, for H3, the text refiner).
    Text,
    /// The denoise ladder. LTX single-stage runs and LTX stage 1 use this.
    Denoise,
    /// LTX two-stage: the latent upsampler between the stages.
    Upsample,
    /// LTX two-stage: the stage-2 refinement steps.
    Refine,
    /// Audio VAE / vocoder.
    AudioDecode,
    /// Video VAE decode (and the writer it feeds).
    VideoDecode,
}

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Denoise => "denoise",
            Self::Upsample => "upsample",
            Self::Refine => "refine",
            Self::AudioDecode => "audio_decode",
            Self::VideoDecode => "video_decode",
        }
    }
}

/// One progress event.
///
/// - At a stage boundary: `step == 0`, and `total` is the stage's step count
///   (0 when it has none).
/// - After a denoise step: `step` counts the steps done so far in this stage
///   (`1..=total`); for SF-Wan `block` is the causal block (0-based) that step
///   belongs to and `total` covers every block's steps.
/// - During the video decode: `frames` is the number of pixel frames decoded
///   so far (cumulative), `step`/`total` are 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    pub stage: Stage,
    pub step: usize,
    pub total: usize,
    pub block: Option<usize>,
    pub frames: usize,
}

/// A progress callback. It runs on the pipeline's thread between device
/// work; keep it short (push to a channel). Use interior mutability for state.
pub type ProgressFn<'a> = &'a (dyn Fn(&Progress) + 'a);

/// Cancellation plus progress for one generate call. `Copy`, so closures
/// inside a pipeline can each hold one.
#[derive(Clone, Copy, Default)]
pub struct Hooks<'a> {
    cancel: Option<&'a CancelToken>,
    progress: Option<ProgressFn<'a>>,
}

impl std::fmt::Debug for Hooks<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hooks")
            .field("cancel", &self.cancel)
            .field("progress", &self.progress.is_some())
            .finish()
    }
}

impl<'a> Hooks<'a> {
    /// No cancellation, no progress: exactly the un-hooked pipeline.
    pub const NONE: Hooks<'static> = Hooks {
        cancel: None,
        progress: None,
    };

    pub fn new(cancel: Option<&'a CancelToken>, progress: Option<ProgressFn<'a>>) -> Self {
        Self { cancel, progress }
    }

    pub fn with_cancel(mut self, cancel: &'a CancelToken) -> Self {
        self.cancel = Some(cancel);
        self
    }

    pub fn with_progress(mut self, progress: ProgressFn<'a>) -> Self {
        self.progress = Some(progress);
        self
    }

    pub fn is_none(&self) -> bool {
        self.cancel.is_none() && self.progress.is_none()
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_some_and(CancelToken::is_cancelled)
    }

    /// `Err(Cancelled)` once the token is tripped.
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(PipelineError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Report `p`, then [`Self::check`].
    pub fn report(&self, p: Progress) -> Result<()> {
        if let Some(progress) = self.progress {
            progress(&p);
        }
        self.check()
    }

    /// A stage begins (`total` steps, 0 when it has none).
    pub fn stage(&self, stage: Stage, total: usize) -> Result<()> {
        self.report(Progress {
            stage,
            step: 0,
            total,
            block: None,
            frames: 0,
        })
    }

    /// `done` of `total` steps of `stage` are finished.
    pub fn step(
        &self,
        stage: Stage,
        done: usize,
        total: usize,
        block: Option<usize>,
    ) -> Result<()> {
        self.report(Progress {
            stage,
            step: done,
            total,
            block,
            frames: 0,
        })
    }

    /// The video decode has produced `frames` pixel frames so far.
    pub fn frames(&self, frames: usize) -> Result<()> {
        self.report(Progress {
            stage: Stage::VideoDecode,
            step: 0,
            total: 0,
            block: None,
            frames,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn none_never_reports_and_never_cancels() {
        let h = Hooks::NONE;
        assert!(h.is_none());
        assert!(h.stage(Stage::Denoise, 4).is_ok());
        assert!(h.step(Stage::Denoise, 1, 4, None).is_ok());
        assert!(h.frames(17).is_ok());
    }

    #[test]
    fn a_tripped_token_is_a_distinct_cancelled_error_after_the_report() {
        let seen = RefCell::new(Vec::new());
        let progress = |p: &Progress| seen.borrow_mut().push(*p);
        let token = CancelToken::new();
        let h = Hooks::default()
            .with_cancel(&token)
            .with_progress(&progress);
        assert!(h.step(Stage::Denoise, 1, 2, Some(0)).is_ok());
        token.clone().cancel();
        let e = h.step(Stage::Denoise, 2, 2, Some(1)).unwrap_err();
        assert!(matches!(e, PipelineError::Cancelled));
        assert!(e.is_cancelled());
        assert_eq!(e.to_string(), "cancelled");
        let seen = seen.into_inner();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[1].block, Some(1));
        assert_eq!(Stage::VideoDecode.name(), "video_decode");
    }

    /// A seeded-random tiny Wan pipeline run through the real
    /// `generate_to` path (text, DMD denoise, streamed VAE decode, PNG
    /// writer) on whichever device the build has: hooks absent, hooks
    /// present, and a cancel tripped from the progress callback.
    mod wan_end_to_end {
        use super::*;
        use crate::wan::umt5::Umt5Encoder;
        use crate::wan::weights::WeightMap;
        use crate::wan::{AutoencoderKlWan, WanTransformer3D};
        use crate::{GenerateConfig, WanPipeline};
        use fastvideo_models::wan::{Umt5Config, WanVaeConfig, WanVideoArchConfig};
        use rand::{rngs::StdRng, Rng, SeedableRng};
        use rand_distr::StandardNormal;
        use sha2::{Digest, Sha256};
        use std::path::{Path, PathBuf};

        fn random_map(seed: u64) -> WeightMap {
            WeightMap::generated(move |key, shape| {
                let h = key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                    (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
                });
                let mut rng = StdRng::seed_from_u64(seed ^ h);
                let (mean, std) = if key.ends_with("bias") {
                    (0.0, 0.02)
                } else if key.contains("scale_shift_table") {
                    (0.0, 0.1)
                } else if key.contains("embed_tokens") || key.ends_with("shared.weight") {
                    (0.0, 1.0)
                } else if shape.len() <= 1 {
                    (1.0, 0.1)
                } else {
                    let fan_in: usize = shape[1..].iter().product::<usize>().max(1);
                    (0.0, 1.0 / (fan_in as f32).sqrt())
                };
                (0..shape.iter().product::<usize>())
                    .map(|_| mean + std * rng.sample::<f32, _>(StandardNormal))
                    .collect()
            })
        }

        fn pipeline() -> WanPipeline {
            let text = Umt5Config {
                vocab_size: 512,
                d_model: 64,
                d_kv: 32,
                d_ff: 128,
                num_heads: 2,
                num_layers: 2,
                relative_attention_num_buckets: 32,
                relative_attention_max_distance: 128,
                dropout: 0.0,
                eps: 1e-6,
            };
            let dit = WanVideoArchConfig {
                patch_size: [1, 2, 2],
                text_len: 16,
                num_attention_heads: 2,
                attention_head_dim: 128,
                in_channels: 16,
                out_channels: 16,
                text_dim: 64,
                freq_dim: 64,
                ffn_dim: 512,
                num_layers: 2,
                ..WanVideoArchConfig::wan_t2v_1_3b()
            };
            let vae = WanVaeConfig {
                base_dim: 32,
                z_dim: 16,
                dim_mult: vec![1, 2, 2],
                num_res_blocks: 1,
                temporal_upsample: vec![true, false],
                load_encoder: false,
                decoder_base_dim: 32,
                is_residual: false,
                patch_size: 1,
                latents_mean: vec![0.0; 16],
                latents_std: vec![1.0; 16],
            };
            WanPipeline::from_parts(
                Some(Umt5Encoder::load(text, &random_map(0x7e57)).unwrap()),
                WanTransformer3D::load(dit, &random_map(0xd17)).unwrap(),
                AutoencoderKlWan::load(vae, &random_map(0xfae)).unwrap(),
            )
        }

        fn scratch(tag: &str) -> PathBuf {
            let dir = std::env::temp_dir().join(format!(
                "fv-hooks-{}-{tag}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            dir
        }

        /// A word-level tokenizer: enough for `tokenize_prompt`.
        fn tokenizer(dir: &Path) -> String {
            let path = dir.join("tokenizer.json");
            let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
                "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,
                "decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"a":1,"cat":2,
                "walking":3,"blurry":4},"unk_token":"[UNK]"}}"#;
            std::fs::write(&path, json).unwrap();
            path.to_string_lossy().into_owned()
        }

        fn config(tokenizer: String) -> GenerateConfig {
            GenerateConfig {
                height: 48,
                width: 64,
                num_frames: 9,
                num_inference_steps: 3,
                guidance_scale: 1.0,
                flow_shift: 8.0,
                is_dmd: true,
                dmd_steps: Some(vec![1000, 757, 522]),
                seed: 11,
                tokenizer_path: Some(tokenizer),
                ..GenerateConfig::default()
            }
        }

        fn hash_frames(paths: &[String]) -> String {
            let mut h = Sha256::new();
            for p in paths {
                h.update(std::fs::read(p).unwrap());
            }
            format!("{:x}", h.finalize())
        }

        /// The default pool's live bytes after the stream drains (`None` on a
        /// CPU build, where there is no device allocator to check).
        fn live_bytes() -> Option<u64> {
            crate::wan::device::synchronize().unwrap();
            crate::wan::device::pool_usage().map(|u| u.used)
        }

        #[test]
        fn hooks_leave_the_frames_byte_identical_and_cancel_within_one_step() {
            let root = scratch("wan");
            let cfg = config(tokenizer(&root));
            let pipe = pipeline();

            let plain = pipe.generate_to(&cfg, &root.join("plain"), false).unwrap();
            assert_eq!(plain.frames, 9);

            let seen = RefCell::new(Vec::new());
            let progress = |p: &Progress| seen.borrow_mut().push(*p);
            let token = CancelToken::new();
            let hooks = Hooks::default()
                .with_cancel(&token)
                .with_progress(&progress);
            let hooked = pipe
                .generate_to_with_hooks(&cfg, &root.join("hooked"), false, hooks)
                .unwrap();
            assert_eq!(
                hash_frames(&plain.frame_paths),
                hash_frames(&hooked.frame_paths)
            );
            let seen = seen.into_inner();
            let steps: Vec<_> = seen
                .iter()
                .filter(|p| p.stage == Stage::Denoise && p.step > 0)
                .map(|p| (p.step, p.total))
                .collect();
            assert_eq!(steps, vec![(1, 3), (2, 3), (3, 3)]);
            assert_eq!(seen.first().map(|p| p.stage), Some(Stage::Text));
            assert_eq!(
                seen.last().map(|p| (p.stage, p.frames)),
                Some((Stage::VideoDecode, 9))
            );

            // Two warm runs have filled every lazy cache, so the pool's live
            // bytes are the resident model alone; a cancel must return to it.
            let resident = live_bytes();

            // Tripped by the observer after step 1: no step 2 runs.
            let token = CancelToken::new();
            let count = RefCell::new(0usize);
            let trip = |p: &Progress| {
                if p.stage == Stage::Denoise && p.step > 0 {
                    *count.borrow_mut() += 1;
                    token.cancel();
                }
            };
            let hooks = Hooks::default().with_cancel(&token).with_progress(&trip);
            let err = pipe
                .generate_to_with_hooks(&cfg, &root.join("cancelled"), false, hooks)
                .unwrap_err();
            assert!(err.is_cancelled(), "{err}");
            assert_eq!(count.into_inner(), 1);
            assert_eq!(
                live_bytes(),
                resident,
                "a cancelled run leaked device memory"
            );

            // A token tripped before the call stops before the text encoder.
            let hooks = Hooks::default().with_cancel(&token);
            assert!(pipe
                .generate_to_with_hooks(&cfg, &root.join("early"), false, hooks)
                .unwrap_err()
                .is_cancelled());
            assert_eq!(live_bytes(), resident);

            // And the pipeline still produces the same clip afterwards.
            let again = pipe.generate_to(&cfg, &root.join("again"), false).unwrap();
            assert_eq!(
                hash_frames(&plain.frame_paths),
                hash_frames(&again.frame_paths)
            );
            let _ = std::fs::remove_dir_all(&root);
        }
    }

    #[test]
    fn a_token_over_a_shared_flag_sees_the_owner_trip_it() {
        let flag = Arc::new(AtomicBool::new(false));
        let token = CancelToken::from_flag(flag.clone());
        assert!(Hooks::default().with_cancel(&token).check().is_ok());
        flag.store(true, Ordering::Release);
        assert!(Hooks::default().with_cancel(&token).check().is_err());
    }
}
