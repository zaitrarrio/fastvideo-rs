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

mod embed;
mod gpu;
#[cfg(feature = "cuda")]
mod kernels;
mod metrics;
mod mode;
mod model;
mod parity;
mod perf;
mod quality;
mod rand_weights;
mod reference;
mod report;
mod st;

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use mode::Mode;
use report::{Report, StageError, StageResult};

#[derive(Parser)]
#[command(name = "fv-gpucheck", about = "Fail-fast numerical/perf validation for fastvideo-cudarc")]
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
    /// NVRTC-compile the kernel module for each compute capability (no GPU).
    #[cfg(feature = "cuda")]
    Nvrtc {
        #[arg(long, default_value = "7.5,8.0,8.6,8.9,9.0")]
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
    },
    /// Compare two clip runs (e.g. fast vs exact) of the same seed and prompt.
    Compare {
        #[arg(long)]
        a: PathBuf,
        #[arg(long)]
        b: PathBuf,
        #[arg(long, default_value_t = 30.0)]
        min_psnr: f64,
        #[arg(long, default_value_t = 0.15)]
        max_latent_rel: f64,
    },
}

fn parse_list<T: std::str::FromStr>(s: &str) -> anyhow::Result<Vec<T>> {
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| p.trim().parse::<T>().map_err(|_| anyhow::anyhow!("bad list item `{p}`")))
        .collect()
}

/// References must come from the exact-mode CPU path; comparisons from a GPU.
fn check_dump_mode(io: &reference::RefIo, device: &str, mode: Mode) -> StageResult<()> {
    let gpu = gpu::on_gpu(device);
    if io.is_dump() && (gpu || mode != Mode::Exact) {
        return Err(StageError::Error(anyhow::anyhow!("--dump requires --device cpu --mode exact")));
    }
    if !io.is_dump() && !gpu {
        return Err(StageError::Error(anyhow::anyhow!("--reference requires a cuda device")));
    }
    Ok(())
}

fn stage_name(cmd: &Cmd) -> &'static str {
    match cmd {
        #[cfg(feature = "cuda")]
        Cmd::Nvrtc { .. } => "nvrtc",
        #[cfg(feature = "cuda")]
        Cmd::Device => "device",
        #[cfg(feature = "cuda")]
        Cmd::Kernels { .. } => "kernels",
        Cmd::Model { .. } => "model",
        Cmd::Parity { .. } => "parity",
        Cmd::Embed { .. } => "embed",
        Cmd::Probe { .. } => "probe",
        Cmd::Clip { .. } => "clip",
        Cmd::Compare { .. } => "compare",
    }
}

fn run(cli: &Cli, report: &mut Report) -> StageResult<()> {
    report.set("mode", cli.mode);
    report.set("cuda_feature", cfg!(feature = "cuda"));
    match &cli.cmd {
        #[cfg(feature = "cuda")]
        Cmd::Nvrtc { sm } => {
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
        Cmd::Model { device, seed, refs } => {
            let mut io = reference::RefIo::new(refs.dump.as_deref(), refs.reference.as_deref(), "model")?
                .with_videos(cli.out.join("videos").join(report.stage()));
            check_dump_mode(&io, device, cli.mode)?;
            model::run(report, &mut io, device, cli.mode, *seed)?;
            io.finish(report, cli.mode, device)?;
            Ok(())
        }
        Cmd::Parity { weights, device, refs } => {
            let mut io = reference::RefIo::new(refs.dump.as_deref(), refs.reference.as_deref(), "parity")?
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
            },
        ),
        Cmd::Compare {
            a,
            b,
            min_psnr,
            max_latent_rel,
        } => perf::compare(report, a, b, *min_psnr, *max_latent_rel),
    }
}

fn main() {
    let cli = Cli::parse();
    // Must precede every cudarc call: its FASTVIDEO_* flags are cached on first read.
    cli.mode.apply_env();
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
