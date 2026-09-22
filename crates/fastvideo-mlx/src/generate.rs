//! MLX generate scaffold (host-side; Metal gated — see [`crate::metal`]).

use std::path::{Path, PathBuf};

use crate::config::{FastMetalPreset, MlxModelSpec};
use crate::metal::{MetalGate, MlxArrayStub};

#[derive(Debug, Clone)]
pub struct MlxGenerateRequest {
    pub prompt: String,
    pub seed: u64,
    pub height: u32,
    pub width: u32,
    pub num_frames: u32,
    pub num_steps: u32,
    pub spec: MlxModelSpec,
}

impl MlxGenerateRequest {
    pub fn fastmetal_1_3b(prompt: impl Into<String>, seed: u64) -> Self {
        let preset = FastMetalPreset::Qad1_3b;
        Self {
            prompt: prompt.into(),
            seed,
            height: preset.default_height(),
            width: preset.default_width(),
            num_frames: preset.default_frames(),
            num_steps: preset.default_steps(),
            spec: MlxModelSpec::FastMetal(preset),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MlxGenerateScaffold {
    pub weights: PathBuf,
    pub spec: MlxModelSpec,
}

impl MlxGenerateScaffold {
    pub fn open(weights: impl Into<PathBuf>, spec: MlxModelSpec) -> Self {
        Self {
            weights: weights.into(),
            spec,
        }
    }

    /// Returns whether this host can run the MLX path.
    pub fn apple_silicon_ready() -> bool {
        MetalGate::apple_silicon()
    }

    /// Validate expected FastMetal/FastH3 MLX pack layout (host-only).
    pub fn validate_weights_layout(&self) -> Result<Vec<String>, String> {
        let mut found = Vec::new();
        let candidates = [
            "transformer",
            "dit",
            "vae",
            "taehv",
            "text_encoder",
            "model.safetensors",
            "config.json",
        ];
        if !self.weights.exists() {
            return Err(format!(
                "mlx weights root missing: {} ({})",
                self.weights.display(),
                MetalGate::status_message()
            ));
        }
        for name in candidates {
            let p = self.weights.join(name);
            if p.exists() {
                found.push(name.to_string());
            }
        }
        if found.is_empty() && self.weights.is_dir() {
            // Empty dir is ok for scaffold; report gate status.
            found.push(format!("(empty dir; {})", MetalGate::status_message()));
        }
        Ok(found)
    }

    /// Host latent noise stub shaped like a tiny FastMetal latent.
    pub fn host_noise_latent(&self, request: &MlxGenerateRequest) -> MlxArrayStub {
        let lh = (request.height as usize / 8).max(1);
        let lw = (request.width as usize / 8).max(1);
        let lt = 1 + (request.num_frames as usize).saturating_sub(1) / 4;
        MlxArrayStub::zeros(&[1, 16, lt.min(4), lh.min(8), lw.min(8)])
    }

    /// Scaffold generate: writes a placeholder status JSON (no Metal yet).
    pub fn generate_scaffold(
        &self,
        request: &MlxGenerateRequest,
        out_dir: &Path,
    ) -> Result<PathBuf, String> {
        let layout = self.validate_weights_layout().unwrap_or_default();
        let noise = self.host_noise_latent(request);
        if !Self::apple_silicon_ready() {
            return Err(format!(
                "fastvideo-mlx requires Apple Silicon (macOS aarch64); hub {}; {}; layout={:?}",
                self.spec.hub_id(),
                MetalGate::status_message(),
                layout
            ));
        }
        #[cfg(not(feature = "mlx"))]
        {
            std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
            let status = out_dir.join("mlx-scaffold-status.json");
            let body = format!(
                "{{\n  \"status\": \"scaffold\",\n  \"hub\": \"{}\",\n  \"prompt\": \"{}\",\n  \"gate\": \"{}\",\n  \"noise_shape\": {:?},\n  \"layout\": {:?},\n  \"note\": \"enable --features mlx on aarch64 + add mlx-rs target dep for Metal\"\n}}\n",
                self.spec.hub_id(),
                request.prompt.replace('"', "'"),
                MetalGate::status_message(),
                noise.shape,
                layout
            );
            std::fs::write(&status, body).map_err(|e| e.to_string())?;
            Ok(status)
        }
        #[cfg(feature = "mlx")]
        {
            let _ = (&self.weights, request, out_dir, noise, layout);
            if MetalGate::metal_ready() {
                Err("mlx feature + Metal ready but DiT/TAEHV graph not wired yet".into())
            } else {
                Err(format!(
                    "mlx feature enabled but Metal not ready ({})",
                    MetalGate::status_message()
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_platform() {
        let ready = MlxGenerateScaffold::apple_silicon_ready();
        assert_eq!(ready, cfg!(all(target_os = "macos", target_arch = "aarch64")));
    }

    #[test]
    fn host_noise_shape() {
        let sc = MlxGenerateScaffold::open(
            "/tmp/mlx-missing",
            MlxModelSpec::FastMetal(FastMetalPreset::Qad1_3b),
        );
        let req = MlxGenerateRequest::fastmetal_1_3b("a cat", 0);
        let n = sc.host_noise_latent(&req);
        assert_eq!(n.shape[1], 16);
    }

    #[test]
    fn scaffold_errors_or_writes_on_host() {
        let sc = MlxGenerateScaffold::open(
            "/tmp/mlx-missing",
            MlxModelSpec::FastMetal(FastMetalPreset::Qad1_3b),
        );
        let req = MlxGenerateRequest::fastmetal_1_3b("a cat", 0);
        let dir = std::env::temp_dir().join("fastvideo-mlx-scaffold-test");
        let _ = std::fs::remove_dir_all(&dir);
        match sc.generate_scaffold(&req, &dir) {
            Ok(p) => {
                assert!(p.is_file());
                let _ = std::fs::remove_dir_all(&dir);
            }
            Err(e) => assert!(e.contains("Apple Silicon") || e.contains("mlx") || e.contains("metal")),
        }
    }
}
