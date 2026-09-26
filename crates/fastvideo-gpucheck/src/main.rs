//! `fv-gpucheck`: tiered, fail-fast validation of the cudarc Wan backend on GPU.
//!
//! Stages (cheapest first; each writes `<out>/<stage>.json` and exits non-zero
//! on failure — see `scripts/gpu/README.md` for the ladder):
//!
//! | stage   | needs                | proves |
//! |---------|----------------------|--------|
//! | nvrtc   | libnvrtc (no GPU)    | kernel source compiles for each SM |
//! | device  | GPU                  | context + cuBLAS/cuDNN + kernel load |
//! | kernels | GPU                  | every kernel/GEMM matches plain-Rust math |
//! | model   | GPU (+CPU-path dump) | random-weight UMT5/DiT/VAE/samplers: GPU == cudarc CPU path |
//! | parity  | GPU, 1.3B weights    | real-weight forward/decode/denoise: GPU == cudarc CPU path |
//! | embed   | GPU, UMT5-XXL        | real prompt embeddings for clip runs |
//! | probe   | GPU, 1.3B weights    | time/VRAM fit → projected clip cost within budget |
//! | clip    | GPU, 1.3B weights    | full clip, per-step NaN/time guards, quality gates |
//! | compare | two clip dirs        | fast path output stays close to exact path |
//! | compare-clips | two frame dirs (CPU) | paired-clip quality gate: OFF identity + pixel metrics vs a baseline |

mod benchmark;
mod clipcmp;
mod embed;
mod gate;
mod gpu;
mod h3_stage;
mod hunyuan15_stage;
#[cfg(feature = "cuda")]
mod kernels;
#[cfg(feature = "cuda")]
mod kernels_fp8;
mod llm_oracle;
mod lpips;
mod ltx2_stage;
#[cfg(feature = "cuda")]
mod mathprobe;
mod metrics;
mod mode;
mod model;
#[cfg(feature = "cuda")]
mod nvfp4_bench;
mod oracle;
mod parity;
mod perf;
mod quality;
mod rand_weights;
mod reference;
mod report;
mod serve;
mod st;
mod taehv;
mod writer_bench;

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use mode::Mode;
use report::{Report, StageError, StageResult};

#[derive(Parser)]
#[command(
    name = "fv-gpucheck",
    about = "Fail-fast numerical/perf validation for fastvideo-cudarc"
)]
struct Cli {
    /// Directory for `<stage>.json` reports and clip artifacts.
    #[arg(long, global = true, default_value = "gpucheck-out")]
    out: PathBuf,
    /// Precision mode (applied to FASTVIDEO_* env before any cudarc call).
    #[arg(long, global = true, value_enum, default_value_t = Mode::Exact)]
    mode: Mode,
    /// Report name suffix, for running a stage more than once per out dir.
    #[arg(long, global = true)]
    tag: Option<String>,
    /// Record failed checks and continue (stage still fails at the end).
    /// Meant for cheap diagnostic stages (kernels, tiny) so one run lists every bug.
    #[arg(long, global = true)]
    keep_going: bool,
    /// Run video sparse attention. A different attention algorithm, not a
    /// precision knob, so it is opt-in per stage: the model and parity stages
    /// compare against dense references and must not use it.
    #[arg(long, global = true)]
    vsa: bool,
    /// Run every DiT linear with FastVideo's W8A8 tensorwise FP8 recipe
    /// (`FASTVIDEO_FP8`). Only a checkpoint distilled against FP8
    /// (FastWan-QAD) should use it, so it is opt-in per stage.
    #[arg(long, global = true)]
    fp8: bool,
    /// H3 reference FP8 recipe (`FASTVIDEO_H3_QUANT`): `w8a8` (Sol-H3-Spark
    /// stage 1: all blocks + refiner) or `mxfp8` (Sol-H3: blocks 2..=46).
    /// Implies bf16 activations for the DiT. Quality is a clip A/B.
    #[arg(long, global = true, value_name = "RECIPE")]
    h3_quant: Option<String>,
    /// bf16 activations end to end (`FASTVIDEO_BF16_ACT`), as the reference.
    #[arg(long, global = true)]
    bf16_act: bool,
    /// MLX affine weight-only INT8/6/4 (group 64) on H3 attn/FFN. Own fused
    /// dequant-in-tile GEMM. Does not set process-wide `FASTVIDEO_FP8`.
    #[arg(long, global = true, value_name = "BITS")]
    h3_affine: Option<String>,
    /// Time each DiT block phase. Synchronizes per phase, so a profiled run
    /// measures *where* the time goes and must not be quoted for *how long*.
    #[arg(long, global = true)]
    profile: bool,
    /// Decode through TAEHV instead of the Wan VAE. Takes the directory
    /// holding taew2_1.safetensors, because those weights ship separately from
    /// the Wan checkpoint and there is nowhere sensible to guess.
    #[arg(long, global = true)]
    taehv_weights: Option<PathBuf>,
    /// Latent frames per VAE decode pass. Like --vsa this is per stage: exact
    /// parity already sits at the edge of a 24GB card, and a bigger chunk
    /// tips it over.
    #[arg(long, global = true)]
    vae_chunk: Option<usize>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct ClipOpts {
    #[arg(long, default_value_t = 480)]
    height: usize,
    #[arg(long, default_value_t = 832)]
    width: usize,
    /// 4n+1 frames; 129 = 8s at 16 fps.
    #[arg(long, default_value_t = 129)]
    frames: usize,
    #[arg(long, default_value_t = 3)]
    steps: usize,
    #[arg(long, default_value_t = 1.0)]
    guidance: f32,
    #[arg(long, default_value_t = 8.0)]
    flow_shift: f64,
    /// DMD (FastWan) Euler sampler instead of UniPC.
    #[arg(long)]
    dmd: bool,
    #[arg(long, default_value_t = 1024)]
    seed: u64,
    #[arg(long, default_value_t = 16)]
    fps: u32,
}

impl ClipOpts {
    fn spec(&self) -> perf::ClipSpec {
        perf::ClipSpec {
            height: self.height,
            width: self.width,
            frames: self.frames,
            steps: self.steps,
            guidance: self.guidance,
            flow_shift: self.flow_shift,
            dmd: self.dmd,
            seed: self.seed,
            fps: self.fps,
        }
    }
}

#[derive(Args, Clone)]
struct RefArgs {
    /// Write this run's outputs as the reference (use with --device cpu --mode exact).
    #[arg(long)]
    dump: Option<PathBuf>,
    /// Compare this run's outputs with a reference directory.
    #[arg(long)]
    reference: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve a directory read-only over HTTP (run logs for SSH-free drivers).
    Serve {
        #[arg(long)]
        dir: PathBuf,
        #[arg(long, default_value_t = 8000)]
        port: u16,
    },
    /// NVRTC-compile the kernel module for each compute capability (no GPU).
    #[cfg(feature = "cuda")]
    Nvrtc {
        #[arg(long, default_value = "7.5,8.0,8.6,8.9,9.0,10.0,12.0")]
        sm: String,
    },
    /// Create the CUDA context (cuBLAS, cuDNN, all kernels) and report the GPU.
    #[cfg(feature = "cuda")]
    Device,
    /// Kernel/GEMM/attention/conv parity on the live GPU.
    #[cfg(feature = "cuda")]
    Kernels {
        #[arg(long, default_value_t = 17)]
        seed: u64,
    },
    /// Time the DiT linears under each cuBLAS math option and report which
    /// ones this GPU + cuBLAS build actually honors.
    #[cfg(feature = "cuda")]
    GemmProbe,
    /// Random-weight UMT5/DiT/VAE/samplers (`--dump` on CPU, `--reference` on GPU).
    Model {
        #[arg(long, default_value = "cuda")]
        device: String,
        #[arg(long, default_value_t = 29)]
        seed: u64,
        #[command(flatten)]
        refs: RefArgs,
    },
    /// Real 1.3B weights, small case (`--dump` on CPU, `--reference` on GPU).
    Parity {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
        #[command(flatten)]
        refs: RefArgs,
    },
    /// Encode prompts with UMT5-XXL into `<embeds>/<name>.safetensors`.
    Embed {
        /// Diffusers root containing `text_encoder/` and `tokenizer/`.
        #[arg(long)]
        weights: PathBuf,
        /// JSON `{negative, prompts: [{name, prompt}]}`.
        #[arg(long)]
        prompts: PathBuf,
        #[arg(long)]
        embeds: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
    /// Fit DiT/VAE time and VRAM on short sequences; fail if the target clip won't fit the budget.
    Probe {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        embeds: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
        #[command(flatten)]
        clip: ClipOpts,
        /// Latent frame counts to time (T=1,3,9 → 1560/4680/14040 tokens at 480p).
        #[arg(long, default_value = "1,3,9")]
        probe_latent_frames: String,
        #[arg(long, default_value_t = 60.0)]
        budget_min: f64,
        #[arg(long, default_value_t = 0.92)]
        vram_headroom: f64,
    },
    /// Generate a clip with per-step fail-fast guards and quality gates.
    Clip {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        embeds: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
        #[command(flatten)]
        clip: ClipOpts,
        /// Clip name (artifacts under <out>/clips/<name>/).
        #[arg(long)]
        name: String,
        #[arg(long, default_value_t = 60.0)]
        budget_min: f64,
        #[arg(long)]
        no_mp4: bool,
        /// Run one untimed generation first, so the reported timings are a
        /// warm process (weights resident, allocator grown, first launches
        /// done) rather than the first clip after load.
        #[arg(long)]
        warm: bool,
    },
    /// MiniMax-H3 / FastH3 stages (see `h3_stage.rs`).
    H3 {
        #[command(subcommand)]
        stage: h3_stage::Stage,
    },
    /// LTX-2 stages (see `ltx2_stage.rs`).
    Ltx2 {
        #[command(subcommand)]
        stage: ltx2_stage::Stage,
    },
    /// HunyuanVideo 1.5 stages (see `hunyuan15_stage.rs`).
    Hunyuan {
        #[command(subcommand)]
        stage: hunyuan15_stage::Stage,
    },
    /// A decoder-only text encoder (Qwen3-VL for MiniMax-H3, Gemma-3 for
    /// LTX-2) against transformers' hidden states, on the oracle's own tokens.
    Llm {
        /// The text encoder's weight directory (sharded safetensors).
        #[arg(long)]
        weights: PathBuf,
        /// `qwen3-vl-32b` or `gemma3-12b`.
        #[arg(long)]
        family: String,
        /// Key of the layer list (e.g. `model.language_model.layers`); probed
        /// when omitted.
        #[arg(long)]
        layer_prefix: Option<String>,
        /// Reference written by the model's oracle script: input_ids,
        /// positions, attend and one hidden_<k> per tap.
        #[arg(long)]
        oracle: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// bf16 weights on both sides, through dozens of layers.
        #[arg(long, default_value_t = 0.05)]
        max_rel: f64,
    },
    /// Compare two clip runs (e.g. fast vs exact) of the same seed and prompt.
    Compare {
        #[arg(long)]
        a: PathBuf,
        #[arg(long)]
        b: PathBuf,
        /// Step-1 latents: one forward from identical noise.
        #[arg(long, default_value_t = 0.05)]
        max_step1_rel: f64,
        /// Final latents after the trajectories have drifted.
        #[arg(long, default_value_t = 0.35)]
        max_latent_rel: f64,
        #[arg(long, default_value_t = 20.0)]
        min_psnr: f64,
    },
    /// Two FASTVIDEO_DUMP_DIR directories (same seed, different paths),
    /// tensor by tensor: per-step latents / velocities and first-step block
    /// outputs, as rel-L2 (telemetry, no gate).
    CompareDumps {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
    },
    /// Paired-clip quality gate (CPU): a candidate clip's frames against a
    /// baseline clip's (sol-engine collect_run.py metrics). Each dir holds
    /// `frame-NNN.png` (+ `output.mp4`) directly or under `frames/`.
    CompareClips {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        /// Hard gate: every paired frame byte-identical (max_abs_diff_uint8 == 0).
        #[arg(long)]
        off_identity: bool,
        /// LPIPS(alex) on the frame pairs: the directory `fetch-lpips.sh`
        /// filled (torchvision AlexNet + LPIPS v0.1 heads, hash-pinned).
        #[arg(long)]
        lpips: Option<PathBuf>,
        /// `sol` (sol-engine's selection: 32 stratified + 16 worst, at most
        /// 48 pairs) or `all`.
        #[arg(long, default_value = "sol")]
        lpips_pairs: String,
        /// `cuda` (cuDNN convolutions) or `cpu`.
        #[arg(long, default_value = "cuda")]
        lpips_device: String,
    },
    /// LPIPS port check: the fixture pairs against the pinned official
    /// `lpips` numbers (CPU and, on a GPU, the device path), or one `--a` /
    /// `--b` pair scored.
    Lpips {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        fixtures: Option<PathBuf>,
        #[arg(long)]
        a: Option<PathBuf>,
        #[arg(long)]
        b: Option<PathBuf>,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
    /// Promotion gate: a candidate cell against a baseline cell, from their
    /// `benchmark.json` and the `compare-clips` report(s), under a policy
    /// (`scripts/gpu/gate-policy.toml`). Exit 0 = pass, 1 = fail.
    Gate {
        /// Baseline cell directory (holds `benchmark.json`) or the file.
        #[arg(long)]
        baseline: PathBuf,
        /// Candidate cell directory or its `benchmark.json`.
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long)]
        policy: PathBuf,
        /// compare-clips report(s) for this pair; several for a prompt set.
        /// Default: `<baseline>/../compare/compare-clips-<b>--<c>*.json`.
        #[arg(long)]
        compare: Vec<PathBuf>,
        /// compare-clips report(s), made with `--off-identity`, of the
        /// baseline against the candidate build with the technique OFF.
        #[arg(long)]
        off_compare: Vec<PathBuf>,
        /// Policy technique kind: `exact` (numeric / kernel switch: the OFF
        /// arm must be byte-identical) or `lossy` (generative: telemetry +
        /// limits). Default: the policy's `kind`.
        #[arg(long)]
        kind: Option<String>,
    },
    /// Diff our TAEHV decoder against madebyollin's own implementation.
    Taehv {
        /// Directory holding taew2_1.safetensors (the oracle fetches it there).
        #[arg(long)]
        weights: PathBuf,
        /// Written by scripts/gpu/taehv_oracle.py.
        #[arg(long)]
        oracle: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
        #[arg(long, default_value_t = 0.02)]
        max_rel: f64,
    },
    /// TAEHV device path (cuDNN convs, device kernels) against a plain-Rust
    /// transcription of `taehv.py` at the real channel widths: decode for
    /// every arch, encode too for the LTX wide checkpoint, plus the chunk
    /// seam. Generated weights unless `--weights` names the real file.
    TaehvDevice {
        /// `wan` (taew2_1), `h3` (taeh3) or `ltx` (taeltx2_3_wide).
        #[arg(long, default_value = "ltx")]
        arch: String,
        /// Weight file, or a directory holding it. Default: generated.
        #[arg(long)]
        weights: Option<PathBuf>,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// TF32 convolutions under `--mode fast` sit around 1e-3.
        #[arg(long, default_value_t = 5e-3)]
        max_rel: f64,
    },
    /// The frame writer (PNG frames + ffmpeg mp4) under a decode-paced
    /// producer of synthetic frames, per PNG mode (CPU only).
    WriterBench {
        #[arg(long, default_value_t = 3840)]
        width: usize,
        #[arg(long, default_value_t = 2176)]
        height: usize,
        #[arg(long, default_value_t = 121)]
        frames: usize,
        /// Frames per push (a decode chunk).
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 24)]
        fps: u32,
        /// Also feed ffmpeg (x264 veryfast, as gen).
        #[arg(long)]
        mp4: bool,
        /// Comma-separated PNG modes: inline, deferred, off.
        #[arg(long, default_value = "inline,deferred,off")]
        png: String,
        /// The paced "decode" length: batches are due evenly over it.
        #[arg(long, default_value_t = 13.0)]
        produce_s: f64,
        /// Scratch directory for the frames (removed after each mode).
        #[arg(long)]
        dir: PathBuf,
    },
    /// Diff our text encoder and one DiT step against an external reference.
    Oracle {
        #[arg(long)]
        weights: PathBuf,
        /// Written by scripts/gpu/upstream_oracle.py.
        #[arg(long)]
        oracle: PathBuf,
        /// Our own embeds for the same prompt (from the embed stage).
        #[arg(long)]
        embeds: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
        #[arg(long, default_value_t = 0.02)]
        max_text_rel: f64,
        #[arg(long, default_value_t = 0.05)]
        max_dit_rel: f64,
        #[arg(long, default_value_t = 0.05)]
        max_e2e_rel: f64,
    },
}

fn parse_list<T: std::str::FromStr>(s: &str) -> anyhow::Result<Vec<T>> {
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            p.trim()
                .parse::<T>()
                .map_err(|_| anyhow::anyhow!("bad list item `{p}`"))
        })
        .collect()
}

/// References must come from the exact-mode CPU path; comparisons from a GPU.
fn check_dump_mode(io: &reference::RefIo, device: &str, mode: Mode) -> StageResult<()> {
    let gpu = gpu::on_gpu(device);
    if io.is_dump() && (gpu || mode != Mode::Exact) {
        return Err(StageError::Error(anyhow::anyhow!(
            "--dump requires --device cpu --mode exact"
        )));
    }
    if !io.is_dump() && !gpu {
        return Err(StageError::Error(anyhow::anyhow!(
            "--reference requires a cuda device"
        )));
    }
    Ok(())
}

fn stage_name(cmd: &Cmd) -> &'static str {
    match cmd {
        Cmd::Serve { .. } => "serve",
        #[cfg(feature = "cuda")]
        Cmd::Nvrtc { .. } => "nvrtc",
        #[cfg(feature = "cuda")]
        Cmd::Device => "device",
        #[cfg(feature = "cuda")]
        Cmd::Kernels { .. } => "kernels",
        #[cfg(feature = "cuda")]
        Cmd::GemmProbe => "gemm-probe",
        Cmd::Model { .. } => "model",
        Cmd::Parity { .. } => "parity",
        Cmd::Embed { .. } => "embed",
        Cmd::Probe { .. } => "probe",
        Cmd::Clip { .. } => "clip",
        Cmd::H3 { .. } => "h3",
        Cmd::Ltx2 { .. } => "ltx2",
        Cmd::Hunyuan { .. } => "hunyuan",
        Cmd::Llm { .. } => "llm",
        Cmd::Compare { .. } => "compare",
        Cmd::CompareClips { .. } => "compare-clips",
        Cmd::Lpips { .. } => "lpips",
        Cmd::Gate { .. } => "gate",
        Cmd::CompareDumps { .. } => "compare-dumps",
        Cmd::Oracle { .. } => "oracle",
        Cmd::Taehv { .. } => "taehv",
        Cmd::WriterBench { .. } => "writer-bench",
        Cmd::TaehvDevice { .. } => "taehv-device",
    }
}

fn run(cli: &Cli, report: &mut Report) -> StageResult<()> {
    report.set("mode", cli.mode);
    report.set("cuda_feature", cfg!(feature = "cuda"));
    match &cli.cmd {
        Cmd::Serve { dir, port } => Ok(serve::run(dir, *port)?),
        #[cfg(feature = "cuda")]
        Cmd::Nvrtc { sm } => {
            // Which SMs this binary carries real SASS for. Empty means it was
            // built without nvcc and will NVRTC-compile on every box.
            report.set("aot_sms", fastvideo_cudarc::wan::kernels::aot_sms());
            // Tile-IR NVFP4 GEMM cubins (fv-oxide-aot) embedded beside them.
            report.set(
                "oxide_cubins",
                fastvideo_cudarc::wan::kernels::oxide_cubins(),
            );
            let archs = sm
                .split(',')
                .map(|s| {
                    let (a, b) = s.trim().split_once('.').unwrap_or((s.trim(), "0"));
                    Ok((a.parse::<i32>()?, b.parse::<i32>()?))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            kernels::nvrtc(report, &archs)
        }
        #[cfg(feature = "cuda")]
        Cmd::Device => {
            let info = gpu::init("cuda")?;
            report.set("device", &info);
            report.check(
                "cuda_context",
                info.compute_capability.is_some(),
                serde_json::to_value(&info)?,
                serde_json::json!({}),
            )
        }
        #[cfg(feature = "cuda")]
        Cmd::Kernels { seed } => kernels::run(report, mode::limits(cli.mode), *seed),
        #[cfg(feature = "cuda")]
        Cmd::GemmProbe => mathprobe::run(report),
        Cmd::Model { device, seed, refs } => {
            let mut io =
                reference::RefIo::new(refs.dump.as_deref(), refs.reference.as_deref(), "model")?
                    .with_videos(cli.out.join("videos").join(report.stage()));
            check_dump_mode(&io, device, cli.mode)?;
            model::run(report, &mut io, device, cli.mode, *seed)?;
            io.finish(report, cli.mode, device)?;
            Ok(())
        }
        Cmd::Parity {
            weights,
            device,
            refs,
        } => {
            let mut io =
                reference::RefIo::new(refs.dump.as_deref(), refs.reference.as_deref(), "parity")?
                    .with_videos(cli.out.join("videos").join(report.stage()));
            check_dump_mode(&io, device, cli.mode)?;
            parity::run(report, &mut io, weights, device, cli.mode)?;
            io.finish(report, cli.mode, device)?;
            Ok(())
        }
        Cmd::Embed {
            weights,
            prompts,
            embeds,
            device,
        } => embed::run(report, weights, prompts, embeds, device),
        Cmd::Probe {
            weights,
            embeds,
            device,
            clip,
            probe_latent_frames,
            budget_min,
            vram_headroom,
        } => perf::probe(
            report,
            perf::ProbeArgs {
                weights,
                embeds,
                device,
                spec: clip.spec(),
                probe_latent_frames: parse_list(probe_latent_frames)?,
                budget_min: *budget_min,
                vram_headroom: *vram_headroom,
            },
        ),
        Cmd::Clip {
            weights,
            embeds,
            device,
            clip,
            name,
            budget_min,
            no_mp4,
            warm,
        } => perf::clip(
            report,
            perf::ClipArgs {
                weights,
                embeds,
                device,
                spec: clip.spec(),
                out_dir: cli.out.join("clips").join(name),
                budget_min: *budget_min,
                gates: quality::QualityGates::default(),
                save_mp4: !no_mp4,
                warm: *warm,
            },
        ),
        Cmd::H3 { stage } => h3_stage::run(report, stage),
        Cmd::Ltx2 { stage } => ltx2_stage::run(report, stage),
        Cmd::Hunyuan { stage } => hunyuan15_stage::run(report, stage),
        Cmd::Llm {
            weights,
            family,
            layer_prefix,
            oracle,
            device,
            max_rel,
        } => llm_oracle::run(
            report,
            weights,
            family,
            layer_prefix.as_deref(),
            oracle,
            device,
            *max_rel,
        ),
        Cmd::Compare {
            a,
            b,
            max_step1_rel,
            max_latent_rel,
            min_psnr,
        } => perf::compare(
            report,
            a,
            b,
            perf::CompareGates {
                max_step1_rel: *max_step1_rel,
                max_latent_rel: *max_latent_rel,
                min_psnr: *min_psnr,
            },
        ),
        Cmd::CompareClips {
            baseline,
            candidate,
            off_identity,
            lpips,
            lpips_pairs,
            lpips_device,
        } => {
            let opts = match lpips {
                Some(weights) => {
                    let all_pairs = match lpips_pairs.as_str() {
                        "sol" => false,
                        "all" => true,
                        other => {
                            return Err(anyhow::anyhow!(
                                "--lpips-pairs {other}: expected sol or all"
                            )
                            .into())
                        }
                    };
                    Some(clipcmp::LpipsOpts {
                        weights: weights.clone(),
                        all_pairs,
                        device: gpu::on_gpu(lpips_device),
                    })
                }
                None => None,
            };
            clipcmp::run(report, baseline, candidate, *off_identity, opts.as_ref())
        }
        Cmd::Lpips {
            weights,
            fixtures,
            a,
            b,
            device,
        } => {
            let pair = match (a, b) {
                (Some(a), Some(b)) => Some((a.as_path(), b.as_path())),
                (None, None) => None,
                _ => return Err(anyhow::anyhow!("--a and --b go together").into()),
            };
            lpips::run(report, weights, fixtures.as_deref(), pair, device)
        }
        Cmd::Gate {
            baseline,
            candidate,
            policy,
            compare,
            off_compare,
            kind,
        } => gate::run(
            report,
            baseline,
            candidate,
            policy,
            compare,
            off_compare,
            kind.as_deref(),
        ),
        Cmd::CompareDumps {
            baseline,
            candidate,
        } => perf::compare_dumps(report, baseline, candidate),
        Cmd::Taehv {
            weights,
            oracle,
            device,
            max_rel,
        } => taehv::run(report, weights, oracle, device, *max_rel),
        Cmd::TaehvDevice {
            arch,
            weights,
            device,
            max_rel,
        } => taehv::run_device(report, arch, weights.as_deref(), device, *max_rel),
        Cmd::WriterBench {
            width,
            height,
            frames,
            batch,
            fps,
            mp4,
            png,
            produce_s,
            dir,
        } => writer_bench::run(
            report,
            &writer_bench::Args {
                out: dir,
                width: *width,
                height: *height,
                frames: *frames,
                batch: (*batch).max(1),
                fps: *fps,
                mp4: *mp4,
                modes: png,
                produce_s: *produce_s,
            },
        ),
        Cmd::Oracle {
            weights,
            oracle,
            embeds,
            device,
            max_text_rel,
            max_dit_rel,
            max_e2e_rel,
        } => oracle::run(
            report,
            weights,
            oracle,
            embeds,
            device,
            oracle::OracleGates {
                max_text_rel: *max_text_rel,
                max_dit_rel: *max_dit_rel,
                max_e2e_rel: *max_e2e_rel,
            },
        ),
    }
}

fn main() {
    let cli = Cli::parse();
    // The file server writes no report and touches no GPU.
    if let Cmd::Serve { dir, port } = &cli.cmd {
        if let Err(e) = serve::run(dir, *port) {
            eprintln!("serve: {e}");
            std::process::exit(1);
        }
        return;
    }
    // Must precede every cudarc call: its FASTVIDEO_* flags are cached on first read.
    cli.mode.apply_env();
    if cli.vsa {
        std::env::set_var("FASTVIDEO_VSA", "1");
    }
    if cli.fp8 {
        std::env::set_var("FASTVIDEO_FP8", "1");
    }
    if let Some(recipe) = &cli.h3_quant {
        std::env::set_var("FASTVIDEO_H3_QUANT", recipe);
    }
    if cli.bf16_act {
        std::env::set_var("FASTVIDEO_BF16_ACT", "1");
    }
    if let Some(bits) = &cli.h3_affine {
        std::env::set_var("FASTVIDEO_H3_AFFINE", bits);
    }
    if cli.profile {
        std::env::set_var("FASTVIDEO_PROFILE", "1");
    }
    if let Some(p) = &cli.taehv_weights {
        std::env::set_var("FASTVIDEO_TAEHV_WEIGHTS", p);
    }
    if let Some(n) = cli.vae_chunk {
        std::env::set_var("FASTVIDEO_VAE_CHUNK", n.to_string());
    }
    let name = match &cli.tag {
        Some(tag) => format!("{}-{tag}", stage_name(&cli.cmd)),
        None => stage_name(&cli.cmd).to_string(),
    };
    let mut report = Report::new(&cli.out, &name);
    report.set_keep_going(cli.keep_going);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&cli, &mut report)))
        .unwrap_or_else(|panic| {
            let msg = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".into());
            Err(StageError::Error(anyhow::anyhow!("panicked: {msg}")))
        });
    let result = report.deferred(result);
    std::process::exit(report.finish(&result));
}
