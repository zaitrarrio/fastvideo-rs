//! Kandinsky 5 generate entry (scaffold).

use std::path::{Path, PathBuf};

use fastvideo_models::kandinsky5::{Kandinsky5Preset, Kandinsky5TransformerConfig};

use crate::wan::pipeline::{PipelineError, Result};

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

#[derive(Debug, Clone)]
pub struct Kandinsky5Request {
    pub prompt: String,
    pub seed: u64,
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub num_steps: usize,
    pub preset: Kandinsky5Preset,
}

impl Kandinsky5Request {
    pub fn lite_5s(prompt: impl Into<String>, seed: u64) -> Self {
        Self {
            prompt: prompt.into(),
            seed,
            height: 512,
            width: 768,
            num_frames: 121,
            num_steps: 50,
            preset: Kandinsky5Preset::LiteT2v5s,
        }
    }
}

pub struct Kandinsky5Pipeline {
    pub root: PathBuf,
    pub dit_cfg: Kandinsky5TransformerConfig,
    pub preset: Kandinsky5Preset,
}

impl Kandinsky5Pipeline {
    pub fn open(root: impl Into<PathBuf>, preset: Kandinsky5Preset) -> Result<Self> {
        Ok(Self {
            root: root.into(),
            dit_cfg: match preset {
                Kandinsky5Preset::LiteT2v5s | Kandinsky5Preset::ProT2v5s => {
                    Kandinsky5TransformerConfig::lite()
                }
            },
            preset,
        })
    }

    pub fn generate(&self, request: &Kandinsky5Request, _out_dir: &Path) -> Result<()> {
        let _ = (&self.root, &self.dit_cfg, request);
        Err(msg(format!(
            "Kandinsky 5 generate not closed yet (preset={}; need DiT visual/text blocks, Qwen crop-129, CLIP, Hunyuan16 VAE). See docs/ports/kandinsky5.md",
            request.preset.as_str()
        )))
    }
}
