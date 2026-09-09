use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use fastvideo_core::{BackendKind, LoadOptions, VideoGenerator, WAN_MODEL_DEFINITIONS};
use fastvideo_models::{DmdSchedule, FlowUniPCMultistepScheduler};

#[derive(Parser)]
#[command(
    name = "fastvideo",
    about = "Rust inference port of FastVideo (Wan/FastWan) for Burn, Candle, and Luminal."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List registered Wan/FastWan Hugging Face ids.
    ListModels,
    /// Resolve a model id and run generation (use --tiny for a zero-weight smoke test).
    Generate(GenerateArgs),
    /// Time CUDA generate (Vast GPU). Not a laptop CPU job.
    Bench(BenchArgs),
    /// Print the flow-match or DMD sigma table for a resolved model.
    Schedule(ScheduleArgs),
}

#[derive(clap::Args)]
struct GenerateArgs {
    /// Hugging Face repo id, e.g. Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    #[arg(long)]
    model: String,
    #[arg(long, default_value = "candle")]
    backend: CliBackend,
    #[arg(long, default_value = "A curious raccoon in a field of sunflowers.")]
    prompt: String,
    #[arg(long)]
    negative: Option<String>,
    #[arg(long, default_value_t = 1)]
    num_gpus: u32,
    /// Zero-weight tiny graph (no Hub download). Writes PNG frames.
    #[arg(long, default_value_t = false)]
    tiny: bool,
    /// Local Diffusers directory with transformer/, vae/, and text_encoder/.
    /// If omitted, uses FASTVIDEO_WEIGHTS or the cached Hugging Face snapshot.
    #[arg(long)]
    weights: Option<String>,
    /// Directory for decoded PNG frames.
    #[arg(long)]
    output: Option<String>,
    /// `cpu`, `cuda`, or `cuda:0`. CUDA binaries need `--features cuda`.
    #[arg(long, default_value = "cpu")]
    device: String,
    /// `f32`, `f16`, or `bf16`. Defaults to bf16 on CUDA, f32 on CPU.
    #[arg(long)]
    dtype: Option<String>,
    #[arg(long)]
    steps: Option<u32>,
    #[arg(long)]
    frames: Option<u32>,
    #[arg(long)]
    height: Option<u32>,
    #[arg(long)]
    width: Option<u32>,
    #[arg(long)]
    guidance: Option<f32>,
    #[arg(long)]
    guidance_2: Option<f32>,
    #[arg(long)]
    seed: Option<u64>,
    /// First-frame image for I2V (PNG or JPEG).
    #[arg(long)]
    image: Option<String>,
}

#[derive(clap::Args)]
struct BenchArgs {
    #[arg(long, default_value = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")]
    model: String,
    #[arg(long, default_value = "candle")]
    backend: CliBackend,
    #[arg(long, default_value = "A curious raccoon in a field of sunflowers.")]
    prompt: String,
    #[arg(long)]
    weights: Option<String>,
    #[arg(long)]
    output: Option<String>,
    #[arg(long, default_value = "cuda")]
    device: String,
    #[arg(long)]
    dtype: Option<String>,
    #[arg(long)]
    steps: Option<u32>,
    #[arg(long)]
    frames: Option<u32>,
    #[arg(long)]
    height: Option<u32>,
    #[arg(long)]
    width: Option<u32>,
    #[arg(long)]
    guidance: Option<f32>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    image: Option<String>,
    /// Only time CLIP ViT-H `image_encoder/` (no DiT).
    #[arg(long, default_value_t = false)]
    clip_only: bool,
}

#[derive(clap::Args)]
struct ScheduleArgs {
    #[arg(long)]
    model: String,
}

#[derive(Clone, Copy, ValueEnum)]
enum CliBackend {
    Host,
    Burn,
    Candle,
    Luminal,
}

impl From<CliBackend> for BackendKind {
    fn from(value: CliBackend) -> Self {
        match value {
            CliBackend::Host => Self::Host,
            CliBackend::Burn => Self::Burn,
            CliBackend::Candle => Self::Candle,
            CliBackend::Luminal => Self::Luminal,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::ListModels => {
            for def in WAN_MODEL_DEFINITIONS {
                for id in def.hf_model_paths {
                    println!(
                        "{id}\tpreset={}\tsampling={}",
                        def.preset,
                        def.sampling.as_str()
                    );
                }
            }
        }
        Commands::Generate(args) => {
            let gen = VideoGenerator::from_pretrained(
                &args.model,
                LoadOptions {
                    backend: args.backend.into(),
                    num_gpus: args.num_gpus,
                    tiny: args.tiny,
                    weights_path: args.weights,
                    output_path: args.output,
                    device: args.device,
                    dtype: args.dtype,
                    height: args.height,
                    width: args.width,
                    num_frames: args.frames,
                    num_inference_steps: args.steps,
                    guidance_scale: args.guidance,
                    guidance_scale_2: args.guidance_2,
                    seed: args.seed,
                    negative_prompt: args.negative,
                    image_path: args.image,
                },
            )?;
            println!("{}", gen.summary());
            match gen.generate_video(&args.prompt) {
                Ok(out) => {
                    if let Some(path) = out.frame_paths.first() {
                        println!("wrote {} frames, first={path}", out.frame_paths.len());
                    }
                }
                Err(err) => {
                    eprintln!("{err}");
                    std::process::exit(2);
                }
            }
        }
        Commands::Bench(args) => {
            if !args.device.to_ascii_lowercase().starts_with("cuda") {
                eprintln!("bench is a Vast CUDA job; got --device {}", args.device);
                std::process::exit(2);
            }
            let gen = VideoGenerator::from_pretrained(
                &args.model,
                LoadOptions {
                    backend: args.backend.into(),
                    weights_path: args.weights,
                    output_path: args.output.or_else(|| Some("/workspace/fastvideo-bench".into())),
                    device: args.device,
                    dtype: args.dtype,
                    height: args.height,
                    width: args.width,
                    num_frames: args.frames,
                    num_inference_steps: args.steps,
                    guidance_scale: args.guidance,
                    seed: args.seed,
                    image_path: args.image.clone(),
                    ..LoadOptions::default()
                },
            )?;
            println!("{}", gen.summary());
            if args.clip_only {
                let image = args.image.ok_or_else(|| {
                    anyhow::anyhow!("--clip-only needs --image <png|jpeg>")
                })?;
                match gen.bench_clip(&image) {
                    Ok(stats) => {
                        println!(
                            "{}",
                            serde_json::json!({
                                "clip_hidden": stats.hidden,
                                "clip_load_ms": stats.load_ms,
                                "clip_encode_ms": stats.encode_ms,
                                "image_encoder": stats.path,
                            })
                        );
                    }
                    Err(err) => {
                        eprintln!("{err}");
                        std::process::exit(2);
                    }
                }
            } else {
                match gen.bench_video(&args.prompt) {
                    Ok((out, stats)) => {
                        println!(
                            "{}",
                            serde_json::json!({
                                "model": stats.model,
                                "device": stats.device,
                                "dtype": stats.dtype,
                                "height": stats.height,
                                "width": stats.width,
                                "frames": stats.frames,
                                "steps": stats.steps,
                                "load_and_generate_ms": stats.load_and_generate_ms,
                                "frames_written": stats.frames_written,
                                "first_frame": out.frame_paths.first(),
                            })
                        );
                    }
                    Err(err) => {
                        eprintln!("{err}");
                        std::process::exit(2);
                    }
                }
            }
        }
        Commands::Schedule(args) => {
            let gen = VideoGenerator::from_pretrained(
                &args.model,
                LoadOptions::default(),
            )?;
            println!("{}", gen.summary());
            match gen.definition.sampling {
                fastvideo_core::SamplingAlgorithm::Dmd
                | fastvideo_core::SamplingAlgorithm::CausalDmd => {
                    let steps = gen
                        .pipeline
                        .dmd_steps
                        .unwrap_or(&fastvideo_models::schedulers::FAST_WAN_1_3B_DMD_STEPS);
                    let sched = DmdSchedule::new(
                        steps,
                        f64::from(gen.pipeline.flow_shift),
                        1000,
                    );
                    println!("dmd_timesteps={:?}", sched.train_timesteps);
                    println!("dmd_sigmas={:?}", sched.sigmas);
                }
                fastvideo_core::SamplingAlgorithm::UniPc => {
                    let mut sched = FlowUniPCMultistepScheduler::new(
                        1000,
                        f64::from(gen.pipeline.flow_shift),
                    );
                    sched.set_timesteps(gen.sampling.num_inference_steps as usize);
                    println!(
                        "unipc_first_timestep={:.6} last={:.6} n={} first_sigma={:.6} i64_t0={}",
                        sched.inference_timesteps()[0],
                        sched.inference_timesteps().last().copied().unwrap_or(0.0),
                        sched.inference_timesteps().len(),
                        sched.inference_sigmas()[0],
                        sched.inference_timesteps_i64()[0]
                    );
                }
            }
        }
    }
    Ok(())
}
