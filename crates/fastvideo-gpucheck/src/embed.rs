//! Real prompt embeddings from cudarc's UMT5-XXL on the GPU, in their own
//! process so the ~18GB of text-encoder weights are gone before the DiT loads.

use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use fastvideo_cudarc::wan::umt5::{pad_prompt_embeds, Umt5Encoder};
use fastvideo_cudarc::wan::weights::WeightMap;
use fastvideo_cudarc::CudaTensor;
use fastvideo_models::wan::Umt5Config;
use serde::Deserialize;
use serde_json::json;

use crate::metrics::{mean_std, non_finite};
use crate::report::{Report, StageResult};
use crate::st::{self, F32Tensor};

#[derive(Debug, Deserialize)]
pub struct PromptFile {
    pub negative: String,
    pub prompts: Vec<Prompt>,
}

#[derive(Debug, Deserialize)]
pub struct Prompt {
    pub name: String,
    pub prompt: String,
}

const TEXT_LEN: usize = 512;

pub fn run(report: &mut Report, weights: &Path, prompts: &Path, out: &Path, device: &str) -> StageResult<()> {
    report.set("device", crate::gpu::init(device)?);
    let file: PromptFile = serde_json::from_str(
        &std::fs::read_to_string(prompts).with_context(|| format!("read {}", prompts.display()))?,
    )?;
    let tokenizer = weights.join("tokenizer").join("tokenizer.json");
    let tokenizer = tokenizer.to_str().context("tokenizer path is not UTF-8")?.to_string();

    let timer = Instant::now();
    let text = {
        let map = WeightMap::from_dir(&weights.join("text_encoder"))?;
        Umt5Encoder::load(Umt5Config::xxl(), &map)?
    };
    report.note("load_text_encoder", json!({"seconds": timer.elapsed().as_secs_f64()}));

    let encode = |s: &str| -> anyhow::Result<CudaTensor> {
        let (ids, len) = fastvideo_models::tokenize_prompt(&tokenizer, s, TEXT_LEN)
            .map_err(anyhow::Error::msg)?;
        Ok(pad_prompt_embeds(&text.forward(&ids, 1, ids.len())?, &[len], TEXT_LEN)?)
    };
    let timer = Instant::now();
    let negative = encode(&file.negative)?;
    report.note("encode_negative", json!({"seconds": timer.elapsed().as_secs_f64()}));

    std::fs::create_dir_all(out)?;
    for p in &file.prompts {
        let timer = Instant::now();
        let embeds = CudaTensor::cat(&[&negative, &encode(&p.prompt)?], 0)?;
        let data = embeds.host_cow()?.into_owned();
        let secs = timer.elapsed().as_secs_f64();
        let bad = non_finite(&data);
        let (mean, std) = mean_std(&data);
        report.check(
            format!("{}/finite_nonzero", p.name),
            bad == 0 && std > 1e-4,
            json!({"non_finite": bad, "mean": mean, "std": std, "seconds": secs}),
            json!({"non_finite": 0, "std_min": 1e-4}),
        )?;
        st::save(
            &out.join(format!("{}.safetensors", p.name)),
            &[("embeds", &F32Tensor::new(embeds.shape.clone(), data)?)],
        )?;
    }
    Ok(())
}
