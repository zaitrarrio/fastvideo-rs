//! MLX generate scaffold (host-side; no Metal kernels in this crate yet).

use std::path::{Path, PathBuf};

use crate::config::{FastMetalPreset, MlxModelSpec};

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
        cfg!(all(target_os = "macos", target_arch = "aarch64"))
    }

    /// Scaffold generate: writes a placeholder status JSON (no Metal yet).
    pub fn generate_scaffold(
        &self,
        request: &MlxGenerateRequest,
        out_dir: &Path,
    ) -> Result<PathBuf, String> {
        if !Self::apple_silicon_ready() {
            return Err(format!(
                "fastvideo-mlx requires Apple Silicon (macOS aarch64); hub {}",
                self.spec.hub_id()
            ));
        }
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (&self.weights, request);
            std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
            let status = out_dir.join("mlx-scaffold-status.json");
            let body = format!(
                "{{\n  \"status\": \"scaffold\",\n  \"hub\": \"{}\",\n  \"prompt\": \"{}\",\n  \"note\": \"enable --features mlx and wire mlx-rs kernels\"\n}}\n",
                self.spec.hub_id(),
                request.prompt.replace('"', "'")
            );
            std::fs::write(&status, body).map_err(|e| e.to_string())?;
            Ok(status)
        }
        #[cfg(feature = "mlx")]
        {
            let _ = (&self.weights, request, out_dir);
            Err("mlx feature enabled but Metal graph not wired yet".into())
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
    fn scaffold_errors_or_writes_on_host() {
        let sc = MlxGenerateScaffold::open("/tmp/mlx-missing", MlxModelSpec::FastMetal(FastMetalPreset::Qad1_3b));
        let req = MlxGenerateRequest::fastmetal_1_3b("a cat", 0);
        let dir = std::env::temp_dir().join("fastvideo-mlx-scaffold-test");
        let _ = std::fs::remove_dir_all(&dir);
        match sc.generate_scaffold(&req, &dir) {
            Ok(p) => {
                assert!(p.is_file());
                let _ = std::fs::remove_dir_all(&dir);
            }
            Err(e) => assert!(e.contains("Apple Silicon") || e.contains("mlx")),
        }
    }
}
