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
//!
//! [`Hooks::with_sink`] (serve package E2) also routes the decoded frames and
//! audio to a [`FrameSink`](crate::sink::FrameSink) in memory; see
//! [`crate::sink`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use std::path::Path;

use crate::sink::{AudioPcm, Port};
use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::writer::VideoWriter;

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
    sink: Option<&'a (dyn Port + 'a)>,
}

impl std::fmt::Debug for Hooks<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hooks")
            .field("cancel", &self.cancel)
            .field("progress", &self.progress.is_some())
            .field("sink", &self.sink.is_some())
            .finish()
    }
}

impl<'a> Hooks<'a> {
    /// No cancellation, no progress: exactly the un-hooked pipeline.
    pub const NONE: Hooks<'static> = Hooks {
        cancel: None,
        progress: None,
        sink: None,
    };

    pub fn new(cancel: Option<&'a CancelToken>, progress: Option<ProgressFn<'a>>) -> Self {
        Self {
            cancel,
            progress,
            sink: None,
        }
    }

    pub fn with_cancel(mut self, cancel: &'a CancelToken) -> Self {
        self.cancel = Some(cancel);
        self
    }

    pub fn with_progress(mut self, progress: ProgressFn<'a>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Deliver the decoded frames and audio to `port`'s sink in memory
    /// instead of writing `frame-NNN.png` files (serve E2; see [`crate::sink`]).
    pub fn with_sink<'s: 'a>(mut self, port: &'a crate::sink::SinkPort<'s>) -> Self {
        self.sink = Some(port);
        self
    }

    pub fn has_sink(&self) -> bool {
        self.sink.is_some()
    }

    pub fn is_none(&self) -> bool {
        self.cancel.is_none() && self.progress.is_none() && self.sink.is_none()
    }

    /// The clip's writer: exactly the batch writer
    /// ([`VideoWriter::spawn_with_audio`]) without a sink; with one, the same
    /// writer (mp4 as asked, no PNGs) tapped for the sink. `fps` is the
    /// clip's rate (the mp4 gets it rounded; 0 = no mp4).
    pub(crate) fn open_writer(
        &self,
        dir: &Path,
        fps: f64,
        mp4: bool,
        audio: Option<&Path>,
    ) -> Result<VideoWriter> {
        match self.sink {
            Some(port) => port.open_writer(dir, fps, mp4, audio),
            None => VideoWriter::spawn_with_audio(dir, crate::sink::mp4_fps(fps), mp4, audio),
        }
    }

    /// Hand the clip's audio to the sink (no-op without one).
    pub(crate) fn audio(&self, pcm: &AudioPcm<'_>) -> Result<()> {
        match self.sink {
            Some(port) => port.audio(pcm),
            None => Ok(()),
        }
    }

    /// After the writer's video half has finished: deliver every frame still
    /// pending. `Some(frames delivered)` with a sink.
    pub(crate) fn finish_sink(&self) -> Result<Option<usize>> {
        self.sink.map(|port| port.finish()).transpose()
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

    /// The video decode has produced `frames` pixel frames so far. Also
    /// delivers the chunks the writer has taken so far to the sink, if any.
    pub fn frames(&self, frames: usize) -> Result<()> {
        if let Some(port) = self.sink {
            port.pump()?;
        }
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

        /// Bind CUDA device 0 to this thread when the build and the box have
        /// one, so the whole run (weights included) takes the GPU path.
        /// `FV_HOOKS_REQUIRE_GPU=1` fails instead of falling back to the CPU.
        fn bind_gpu() -> bool {
            #[cfg(feature = "cuda")]
            if let Ok(dev) = crate::wan::device::device_for_index(0) {
                crate::wan::device::set_thread_device(Some(dev));
            }
            let live = crate::wan::device::has_live_device();
            if std::env::var_os("FV_HOOKS_REQUIRE_GPU").is_some() {
                assert!(
                    live,
                    "FV_HOOKS_REQUIRE_GPU is set but no CUDA device is live"
                );
            }
            live
        }

        #[test]
        fn hooks_leave_the_frames_byte_identical_and_cancel_within_one_step() {
            let root = scratch("wan");
            let on_gpu = bind_gpu();
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

            // Serve E2: the same clip through a frame sink is the PNG path's
            // frames byte for byte, with no PNG written.
            let mut sink = crate::sink::CollectFrames::default();
            let port = crate::sink::SinkPort::new(&mut sink);
            let sunk = pipe
                .generate_to_with_hooks(
                    &cfg,
                    &root.join("sink"),
                    false,
                    Hooks::default().with_sink(&port),
                )
                .unwrap();
            assert!(sunk.frame_paths.is_empty());
            assert_eq!(port.frames_delivered(), 9);
            drop(port);
            let pngs: Vec<Vec<u8>> = plain
                .frame_paths
                .iter()
                .map(|p| image::open(p).unwrap().to_rgb8().into_raw())
                .collect();
            let got: Vec<&[u8]> = sink.frames().collect();
            assert_eq!(got.len(), pngs.len());
            assert!(got.iter().zip(&pngs).all(|(a, b)| *a == b.as_slice()));
            assert!(sink.audio.is_none());
            assert!(std::fs::read_dir(root.join("sink"))
                .map(|d| d.count() == 0)
                .unwrap_or(true));

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
            if on_gpu {
                assert!(resident.is_some(), "GPU run without a pool to check");
            }
            eprintln!("hooks e2e: gpu={on_gpu} resident_pool_bytes={resident:?}");
            drop(pipe);
            #[cfg(feature = "cuda")]
            crate::wan::device::set_thread_device(None);
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
