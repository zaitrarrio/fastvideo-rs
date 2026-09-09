use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use fastvideo_core::{BackendKind, LoadOptions, VideoGenerator, WAN_MODEL_DEFINITIONS};
use fastvideo_models::{DmdSchedule, FlowMatchEulerDiscreteScheduler};

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
    #[arg(long, default_value_t = 1)]
    num_gpus: u32,
    /// Zero-weight tiny graph (no Hub download). Writes PNG frames.
    #[arg(long, default_value_t = false)]
    tiny: bool,
    /// Local Diffusers directory with transformer/, vae/, and text_encoder/.
    #[arg(long)]
    weights: Option<String>,
    /// Directory for decoded PNG frames.
    #[arg(long)]
    output: Option<String>,
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
                    let mut sched = FlowMatchEulerDiscreteScheduler::new(
                        1000,
                        f64::from(gen.pipeline.flow_shift),
                    );
                    sched.set_timesteps(gen.sampling.num_inference_steps as usize);
                    println!(
                        "unipc_first_timestep={:.6} last={:.6} n={}",
                        sched.inference_timesteps()[0],
                        sched.inference_timesteps().last().copied().unwrap_or(0.0),
                        sched.inference_timesteps().len()
                    );
                }
            }
        }
    }
    Ok(())
}
