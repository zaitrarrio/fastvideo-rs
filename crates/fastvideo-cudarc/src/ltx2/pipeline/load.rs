//! [`Ltx2Pipeline::load_with_settings`] in four stages
//! (`docs/adr/0017-loader-stages.md`):
//!
//! 1. **plan** ([`Ltx2LoadPlan::new`]): pure. From the paths, the config,
//!    the options, the settings snapshot, the files' names and headers (the
//!    distilled LoRA beside the weights, the IC-LoRA's metadata) and the
//!    [`DeviceFacts`]: which LoRAs, where the DiT, decoders and Gemma live,
//!    which video decoder. No device read, no tensor byte read.
//! 2. **open** ([`Ltx2LoadPlan::open`]): the DiT's map with its read-ahead
//!    queued (a pipeline without a LoRA; a LoRA pipeline loads its DiT on the
//!    first request).
//! 3. **place** ([`Ltx2LoadPlan::place`]): the DiT (the IC-LoRA's factors
//!    attached unfused), the decoders, the tiny autoencoder; then the
//!    read-ahead of what the first request places (Gemma, a LoRA pipeline's
//!    DiT), queued behind this placement as before.
//! 4. **fuse** ([`Ltx2LoadPlan::fuse`]): LTX-2 fuses its LoRAs per request
//!    (the distilled LoRA on a dev DiT at each stage's strength, the
//!    IC-LoRA in stage 1), so at load this stage records where they start
//!    (strength 0: the base) and builds the pipeline.
//!
//! The composition is the load it replaces, step for step: the same tensors
//! in the same order and precision, the same placement, the same `load/io`
//! line and ledger bookings.

use super::*;
use crate::rt::load_stages::DeviceFacts;
use fastvideo_models::techniques::settings::Active;

/// Everything an LTX-2 load decides before it touches the device or reads a
/// tensor ([`Self::new`]).
pub struct Ltx2LoadPlan {
    pub paths: Ltx2Paths,
    pub cfg: Ltx2Config,
    pub options: PipelineOptions,
    pub facts: DeviceFacts,
    /// When the load started (`load_s`, the `ltx2 load` I/O line).
    pub started: Instant,
    /// The distilled LoRA a dev (dynamic-shift) DiT fuses per request; a
    /// distilled DiT already carries it.
    pub lora: Option<PathBuf>,
    /// The IC-LoRA for reference-to-video and its header.
    pub ic_lora: Option<(PathBuf, super::super::lora::IcLoraInfo)>,
    /// The reference's `--offload` placement.
    pub offload: LtxOffload,
    /// Where Gemma and the connectors live.
    pub text_residency: TextResidency,
    /// The DiT placement policy under `offload`.
    pub dit_policy: DitOffload,
    /// The resident plan of the default workload ([`fastvideo_models::ltx2::Rtx5090Workload::DEFAULT`]).
    pub need: u64,
    /// Where the DiT blocks live (the policy against `need` and the facts'
    /// free memory). Streamed also keeps the decoders off the device between
    /// their calls.
    pub residency: Residency,
    /// A spatial upsampler is configured and present (two-stage runs).
    pub has_upsampler: bool,
    /// The tiny autoencoder decoding video instead of the conv VAE.
    pub tae: Option<PathBuf>,
    /// A LoRA DiT keeps its unfused base in pinned host memory.
    pub lora_base_host: bool,
    /// `FASTVIDEO_NVFP4`: the video FFN's NVFP4 rule (sol-engine's scope:
    /// everything else stays dense), as the snapshot sets it.
    pub video_ffn_nvfp4: Option<fastvideo_models::nvfp4::ScaleRule>,
    /// The runtime's DiT block fusions (`FASTVIDEO_LTX_FUSE`, applied under
    /// bf16 activations), as the snapshot sets them. The fuse stage installs
    /// it on the pipeline, which installs it on every DiT it holds
    /// (`crate::rt::fuse::installed`, ADR-0022).
    pub kernel_fusions: crate::rt::fuse::BlockFusion,
}

/// The DiT's map, opened ([`Ltx2LoadPlan::open`]).
pub struct Ltx2Opened {
    io_base: fastvideo_loader::PrefetchStats,
    dit: Option<WeightMap>,
}

impl Ltx2Opened {
    /// The DiT's map (`None`: a LoRA pipeline loads its DiT on the first request).
    pub fn dit_map(&self) -> Option<&WeightMap> {
        self.dit.as_ref()
    }
}

/// The placed models ([`Ltx2LoadPlan::place`]).
pub struct Ltx2Placed {
    model: Option<Ltx2Transformer>,
    decoders: Option<Decoders>,
    tae: Option<crate::rt::taehv::TaeHv>,
    booking: crate::rt::ledger::Booking,
    text: TextEncoder,
    dit_warm: Option<WeightMap>,
}

impl Ltx2Placed {
    /// The DiT is on the device (a pipeline without a LoRA).
    pub fn has_dit(&self) -> bool {
        self.model.is_some()
    }
    /// The decoders are on the device (a resident DiT).
    pub fn has_decoders(&self) -> bool {
        self.decoders.is_some()
    }
}

impl Ltx2LoadPlan {
    /// The plan of loading `paths` with `cfg` and `options` under `settings`
    /// (the load's snapshot), the process's `active` profile and the device
    /// as `facts` describe it; `started` is when the load began. Logs what it
    /// decides (the IC-LoRA, the profile, the offload and the DiT placement).
    pub fn new(
        paths: &Ltx2Paths,
        cfg: &Ltx2Config,
        options: &PipelineOptions,
        settings: &Snapshot,
        active: &Active,
        facts: DeviceFacts,
        started: Instant,
    ) -> Result<Self> {
        // The callees below read their settings through the scope (a2); a
        // nested scope of the same snapshot when the load already holds it.
        let _scope = settings.scope();
        // A distilled checkpoint already carries the adapter: the RTX5090
        // distilled two-stage (`run_ltx25_gpu.sh`) passes no LoRA. Only a dev
        // transformer fuses one (two-stage stage 2, the refiners).
        let lora = if cfg.scheduler.use_dynamic_shifting {
            super::super::lora::resolve(&paths.weights, &paths.dit, cfg.version)?
        } else {
            None
        };
        let ic_lora = match resolve_ic_lora(options.ic_lora.as_deref(), settings) {
            None => None,
            Some(path) => {
                if lora.is_some() || cfg.version != Ltx2ModelVersion::V25 {
                    return Err(err(
                        "ltx2 ic-lora: reference-to-video runs on the LTX-2.5 distilled DiT only",
                    ));
                }
                if !path.is_file() {
                    return Err(err(format!(
                        "ltx2 ic-lora: {} is not a file",
                        path.display()
                    )));
                }
                let info = super::super::lora::ic_lora_info(&path)?;
                if info.temporal_scale != 1 {
                    return Err(err(format!(
                        "ltx2 ic-lora: reference_temporal_scale_factor {} is not supported",
                        info.temporal_scale
                    )));
                }
                crate::rt::log::info(format_args!(
                    "ltx2 ic-lora: {} ({} pairs, reference downscale {}, stage 1 only)",
                    path.display(),
                    info.pairs,
                    info.downscale
                ));
                Some((path, info))
            }
        };
        if let Some(path) = active.path.as_ref() {
            crate::rt::log::info(format_args!("ltx2 technique profile: {}", path.display()));
            for (k, v, source) in active.settings.iter() {
                crate::rt::log::info(format_args!(
                    "ltx2 technique setting {k}={v} ({source}{})",
                    if settings.env_var_os(k).is_some() {
                        "; overridden by env"
                    } else {
                        ""
                    }
                ));
            }
            let has_taehv = active
                .profile
                .iter()
                .flat_map(|p| p.techniques.iter())
                .any(|t| {
                    t.name() == "taehv" && t.active_somewhere(fastvideo_models::techniques::HORIZON)
                });
            if has_taehv && resolve_tae(options.tae.as_deref(), settings).is_none() {
                return Err(err(
                    "techniques.taehv: no weights (pass --ltx-tae-weights, set FASTVIDEO_LTX2_TAE_WEIGHTS, or give techniques.taehv.weights)",
                ));
            }
        }
        let offload = LtxOffload::from_settings_or(settings, options.offload).map_err(err)?;
        let text_residency = resolve_text_for_offload(
            offload,
            options.text_residency,
            settings.env_var("FASTVIDEO_LTX2_TEXT").ok().as_deref(),
        )?;
        let dit_policy = dit_policy_for_offload(
            offload,
            DitOffload::from_settings_or(settings, options.dit_offload).map_err(err)?,
        )?;
        if offload.is_cpu() {
            crate::rt::log::info(format_args!(
                "ltx2 offload cpu: DiT blocks streamed from pinned host memory ({} device slots), Gemma streamed per layer, connectors / upsampler / video VAE / audio VAE on the device only around their calls",
                crate::rt::offload::lookahead() + 1
            ));
        }
        let w = fastvideo_models::ltx2::Rtx5090Workload::DEFAULT;
        let need = fastvideo_models::ltx2::memory::plan_distilled_two_stage(
            cfg,
            w.height(),
            w.width(),
            w.num_frames(),
            w.frame_rate(),
            fastvideo_models::ltx2::memory::PlanOptions::default(),
        )
        .peak();
        let free = facts.free;
        let residency = dit_policy.resolve(need, free);
        crate::rt::log::info(format_args!(
            "ltx2 dit offload: {} -> {} (resident {} plan {:.1} GiB, {} free before load)",
            dit_policy.as_str(),
            residency.as_str(),
            w.name(),
            need as f64 / f64::from(1u32 << 30),
            free.map_or("unknown".into(), |f| format!(
                "{:.1} GiB",
                f as f64 / f64::from(1u32 << 30)
            )),
        ));
        let has_upsampler =
            cfg.latent_upsampler.is_some() && upsampler_dir(&paths.weights).is_some();
        let tae = resolve_tae(options.tae.as_deref(), settings);
        if tae.is_some() {
            use crate::rt::taehv::TaeArch;
            if cfg.transformer.in_channels != TaeArch::LtxWide.latent_channels() {
                return Err(err(format!(
                    "ltx2: the LTX tiny autoencoder decodes {}-channel latents, this line has {}",
                    TaeArch::LtxWide.latent_channels(),
                    cfg.transformer.in_channels
                )));
            }
        }
        let lora_base_host = match settings
            .env_var("FASTVIDEO_LTX2_LORA_BASE")
            .ok()
            .as_deref()
            .map(str::trim)
        {
            Some("host") => true,
            Some("device") => false,
            Some(other) if !other.is_empty() => {
                return Err(err(format!(
                    "FASTVIDEO_LTX2_LORA_BASE={other}: expected host or device"
                )))
            }
            _ => options.lora_base_host,
        };
        if lora_base_host && lora.is_some() && residency.is_streamed() {
            return Err(err(
                "ltx2: a host-kept LoRA base needs a resident DiT (dit offload 'resident' or 'auto' with room)",
            ));
        }
        Ok(Self {
            paths: paths.clone(),
            cfg: cfg.clone(),
            options: options.clone(),
            facts,
            started,
            lora,
            ic_lora,
            offload,
            text_residency,
            dit_policy,
            need,
            residency,
            has_upsampler,
            tae,
            lora_base_host,
            video_ffn_nvfp4: fastvideo_models::nvfp4::from_settings(settings),
            kernel_fusions: crate::rt::fuse::BlockFusion::ltx(settings),
        })
    }

    /// The plan's decisions, one `key: value` line each (the golden of
    /// `docs/adr/0017-loader-stages.md`).
    pub fn describe(&self) -> String {
        let gib = |b: u64| format!("{:.2}GiB", b as f64 / f64::from(1u32 << 30));
        let rel = |p: &Path| {
            p.strip_prefix(&self.paths.weights).map_or_else(
                |_| p.display().to_string(),
                |r| format!("<weights>/{}", r.display()),
            )
        };
        let out = [
            format!("facts: {}", self.facts.describe()),
            format!(
                "model: {:?} {} layers={} in_channels={}",
                self.cfg.version,
                if self.cfg.scheduler.use_dynamic_shifting {
                    "dev"
                } else {
                    "distilled"
                },
                self.cfg.transformer.num_layers,
                self.cfg.transformer.in_channels
            ),
            format!(
                "offload: {:?} dit {} need={} -> {} (at the load-start free memory, as the load decides)",
                self.offload,
                self.dit_policy.as_str(),
                gib(self.need),
                self.residency.as_str()
            ),
            format!(
                "dit: {} decoders: {}",
                if self.lora.is_some() {
                    "first request (LoRA pipeline; read-ahead queued at load)"
                } else {
                    "at load"
                },
                if self.residency.is_streamed() {
                    "around their calls"
                } else {
                    "resident"
                }
            ),
            format!("text: {:?}", self.text_residency),
            format!(
                "precision: video ffn nvfp4={}",
                self.video_ffn_nvfp4
                    .map_or("off".to_owned(), |r| format!("{r:?}"))
            ),
            format!(
                "video decoder: {}",
                self.tae
                    .as_deref()
                    .map_or("conv vae".to_owned(), |p| format!("taehv {}", rel(p)))
            ),
            format!("upsampler: {}", self.has_upsampler),
            format!(
                "fuse lora: {}",
                self.lora
                    .as_deref()
                    .map_or("none (distilled DiT)".to_owned(), |p| format!(
                        "{} per request, unfused base on the {}",
                        rel(p),
                        if self.lora_base_host {
                            "host"
                        } else {
                            "device"
                        }
                    ))
            ),
            format!(
                "fuse ic-lora: {}",
                self.ic_lora
                    .as_ref()
                    .map_or("none".to_owned(), |(p, i)| format!(
                        "{} pairs={} downscale={} (attached unfused, stage 1)",
                        rel(p),
                        i.pairs,
                        i.downscale
                    ))
            ),
            format!(
                "fuse kernels: dit block fusions={}",
                self.kernel_fusions.on(crate::rt::fuse::Switch::Ltx)
            ),
            String::new(),
        ];
        out.join("\n")
    }

    /// Stage 2: the DiT's map with its read-ahead queued, for a pipeline
    /// without a LoRA (a LoRA pipeline loads its DiT on the first request).
    pub fn open(&self) -> Result<Ltx2Opened> {
        let io_base = fastvideo_loader::PrefetchStats::now();
        let dit = if self.lora.is_none() {
            let map = open_distilled(&self.paths.dit, "transformer")?;
            map.prefetch_all();
            Some(map)
        } else {
            None
        };
        Ok(Ltx2Opened { io_base, dit })
    }

    /// Stage 3: the DiT (with the IC-LoRA's factors attached at strength 0),
    /// the decoders unless the DiT streams, the tiny autoencoder; each booked
    /// in the device ledger. Then what the first request places is queued
    /// for read-ahead behind it: Gemma's shards, and a LoRA pipeline's DiT.
    pub fn place(&self, opened: Ltx2Opened) -> Result<Ltx2Placed> {
        let Ltx2Opened { io_base, dit } = opened;
        let (paths, cfg, options) = (&self.paths, &self.cfg, &self.options);
        let residency = self.residency;
        let mut booking = crate::rt::ledger::Booking::default();
        let model = match dit {
            Some(map) => Some(booking.track(
                crate::rt::ledger::DIT_NONLINEAR,
                || {
                    load_transformer_ic(
                        &map,
                        cfg,
                        residency,
                        self.ic_lora.as_ref().map(|(p, _)| p.as_path()),
                    )
                },
                |_| None,
            )?),
            None => None,
        };
        let decoders = if residency.is_streamed() {
            None
        } else {
            Some(booking.track(
                crate::rt::ledger::VAE,
                || Decoders::load_without_upsampler(&paths.weights, cfg),
                |_| None,
            )?)
        };
        let tae = match &self.tae {
            Some(path) => {
                use crate::rt::taehv::{TaeArch, TaeHv};
                let tae = TaeHv::load_from_path(path, TaeArch::LtxWide)
                    .map_err(|e| err(format!("ltx2 tae {}: {e}", path.display())))?;
                crate::rt::log::info(format_args!("ltx2 vae=taehv ({})", path.display()));
                Some(tae)
            }
            None => None,
        };
        sync()?;
        crate::rt::weights::log_load_io(
            "ltx2 load",
            &io_base,
            self.started.elapsed().as_secs_f64(),
        );
        let mut text = TextEncoder::new(
            paths,
            cfg,
            &PipelineOptions {
                text_residency: self.text_residency,
                ..options.clone()
            },
        );
        // The first request encodes the prompt (Gemma), then a LoRA pipeline
        // loads its DiT: queue both reads now, in that order.
        text.warm()?;
        let dit_warm = if model.is_none() {
            let map = open_distilled(&paths.dit, "transformer")?;
            map.prefetch_all();
            Some(map)
        } else {
            None
        };
        Ok(Ltx2Placed {
            model,
            decoders,
            tae,
            booking,
            text,
            dit_warm,
        })
    }

    /// Stage 4: the pipeline, its LoRAs at strength 0 (the base): the
    /// distilled LoRA of a dev DiT fuses at each stage's strength and the
    /// IC-LoRA in stage 1, per request.
    pub fn fuse(self, placed: Ltx2Placed) -> Result<Ltx2Pipeline> {
        let Ltx2Placed {
            model,
            decoders,
            tae,
            booking,
            text,
            dit_warm,
        } = placed;
        let loaded_strength = model.as_ref().map(|_| 0.0);
        // The block fusions this plan decided: every DiT the pipeline holds
        // runs its forwards under them (ADR-0017, ADR-0022).
        let mut model = model;
        if let Some(m) = model.as_mut() {
            m.install_kernel_fusion(self.kernel_fusions);
        }
        Ok(Ltx2Pipeline {
            kernel_fusion: self.kernel_fusions,
            lora_base_host: self.lora_base_host,
            ic_lora: self.ic_lora,
            ic_strength: 0.0,
            dit_warm: std::sync::Mutex::new(dit_warm),
            cfg: self.cfg,
            dit: self.paths.dit,
            weights: self.paths.weights,
            model,
            lora: self.lora,
            loaded_strength,
            decoders,
            residency: self.residency,
            offload: self.offload,
            has_upsampler: self.has_upsampler,
            text,
            load_s: self.started.elapsed().as_secs_f64(),
            booking,
            tae,
        })
    }
}
