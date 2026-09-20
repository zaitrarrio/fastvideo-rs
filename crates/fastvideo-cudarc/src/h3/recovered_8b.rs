//! SearchingMan recovered Qwen3-VL-8B for MiniMax-H3: 24-layer BF16 base +
//! ARA LoRA (r=16, α=16 on layers 16..=23) + nonlinear 4096→5120 adapter.
//!
//! Output matches the official text_dim so the DiT token refiner is unchanged.
//! INT8 ConvRot is out of scope.

use std::path::{Path, PathBuf};

use crate::llm::{conditioning_adapter_host, DecoderConfig, ResidentDecoder};
use crate::wan::nn::Linear;
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::{cuda_tensor_shaped, WeightMap};

use super::text::HiddenStateEncoder;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// HF-style tap after the last loaded layer (un-normed residual after layer 23).
pub const RECOVERED_8B_TAP: usize = 24;
pub const RECOVERED_8B_HIDDEN: usize = 4096;
pub const RECOVERED_8B_OUT: usize = 5120;
pub const RECOVERED_8B_BOTTLENECK: usize = 256;
pub const ARA_LAYERS: [usize; 8] = [16, 17, 18, 19, 20, 21, 22, 23];
pub const ARA_SUFFIXES: [&str; 4] = [
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];
pub const ARA_RANK: usize = 16;
pub const ARA_ALPHA: f32 = 16.0;

/// `RMSNorm → proj + SiLU-bottleneck residual`, 4096→5120.
pub struct ConditioningAdapter {
    norm: CudaTensor,
    proj: Linear,
    down: Linear,
    up: Linear,
    eps: f32,
}

impl ConditioningAdapter {
    pub fn load(map: &WeightMap, eps: f32) -> Result<Self> {
        let norm = cuda_tensor_shaped(map, "norm.weight", &[RECOVERED_8B_HIDDEN])?;
        let mut norm = norm;
        norm.pin_device()?;
        Ok(Self {
            norm,
            proj: Linear::load(map, "proj", RECOVERED_8B_HIDDEN, RECOVERED_8B_OUT, false)?,
            down: Linear::load(map, "down", RECOVERED_8B_HIDDEN, RECOVERED_8B_BOTTLENECK, false)?,
            up: Linear::load(map, "up", RECOVERED_8B_BOTTLENECK, RECOVERED_8B_OUT, false)?,
            eps,
        })
    }

    /// `[1, S, 4096]` → `[1, S, 5120]`.
    pub fn forward(&self, x: &CudaTensor) -> Result<CudaTensor> {
        if x.shape.len() != 3 || x.shape[2] != RECOVERED_8B_HIDDEN {
            return Err(msg(format!(
                "recovered-8b adapter: expected [1, S, {RECOVERED_8B_HIDDEN}], got {:?}",
                x.shape
            )));
        }
        let xn = x.rms_norm(&self.norm, self.eps)?;
        let direct = self.proj.forward(&xn)?;
        let bottleneck = self.down.forward(&xn)?.silu();
        let residual = self.up.forward(&bottleneck)?;
        direct.add(&residual)
    }
}

/// Resident 8B decoder with ARA folded in, plus the conditioning adapter.
pub struct Recovered8bEncoder {
    decoder: ResidentDecoder,
    adapter: ConditioningAdapter,
}

impl Recovered8bEncoder {
    /// `root` is the SearchingMan `recovered_8b/` directory (or a parent that
    /// holds the four files).
    pub fn load(root: &Path) -> Result<Self> {
        let dir = resolve_recovered_dir(root)?;
        let base = WeightMap::open_files(&[dir.join("qwen3vl_8b_minimax_h3_recovered_bf16.safetensors")])?;
        let ara = WeightMap::open_files(&[dir.join("ara.safetensors")])?;
        let adapter_map = WeightMap::open_files(&[dir.join("conditioning_adapter.safetensors")])?;
        let cfg = DecoderConfig::qwen3_vl_8b_text().for_bf16_reference();
        let decoder = ResidentDecoder::load_with_comfy_lora(
            &base,
            &ara,
            &cfg,
            RECOVERED_8B_TAP,
            &ARA_LAYERS,
            &ARA_SUFFIXES,
            ARA_ALPHA,
            ARA_RANK,
        )?;
        let adapter = ConditioningAdapter::load(&adapter_map, cfg.rms_eps)?;
        Ok(Self { decoder, adapter })
    }

    pub fn device_bytes(&self) -> u64 {
        self.decoder.device_bytes()
            + (RECOVERED_8B_HIDDEN
                + RECOVERED_8B_OUT * RECOVERED_8B_HIDDEN
                + RECOVERED_8B_BOTTLENECK * RECOVERED_8B_HIDDEN
                + RECOVERED_8B_OUT * RECOVERED_8B_BOTTLENECK) as u64
                * 4
    }
}

impl HiddenStateEncoder for Recovered8bEncoder {
    fn hidden_state(&self, ids: &[u32], tap: usize) -> Result<CudaTensor> {
        if tap != RECOVERED_8B_TAP && tap != 0 {
            // Pipeline may pass the stock tap 50; we always emit after layer 23.
            crate::wan::log::info(format_args!(
                "recovered-8b: ignoring requested tap {tap}, using {RECOVERED_8B_TAP}"
            ));
        }
        let positions: Vec<u32> = (0..ids.len() as u32).collect();
        let attend = vec![true; ids.len()];
        let mut taps = self
            .decoder
            .hidden_states(ids, &positions, &attend, &[RECOVERED_8B_TAP])?;
        let hidden = taps.pop().ok_or_else(|| msg("recovered-8b: no hidden state"))?;
        self.adapter.forward(&hidden)
    }

    fn kind(&self) -> &'static str {
        "recovered-8b"
    }

    fn resident_bytes(&self) -> u64 {
        self.device_bytes()
    }
}

fn resolve_recovered_dir(root: &Path) -> Result<PathBuf> {
    let candidates = [
        root.to_path_buf(),
        root.join("recovered_8b"),
        root.join("text_encoders").join("recovered_8b"),
    ];
    for dir in candidates {
        let base = dir.join("qwen3vl_8b_minimax_h3_recovered_bf16.safetensors");
        if base.is_file() {
            return Ok(dir);
        }
    }
    Err(msg(format!(
        "recovered-8b: no qwen3vl_8b_minimax_h3_recovered_bf16.safetensors under {}",
        root.display()
    )))
}

/// Host-only adapter identity used by unit tests (no CUDA).
pub fn adapter_forward_host_test(
    x: &[f32],
    seq: usize,
    norm: &[f32],
    proj: &[f32],
    down: &[f32],
    up: &[f32],
) -> Vec<f32> {
    conditioning_adapter_host(
        x,
        seq,
        RECOVERED_8B_HIDDEN,
        RECOVERED_8B_OUT,
        RECOVERED_8B_BOTTLENECK,
        norm,
        proj,
        down,
        up,
        1e-6,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::merge_comfy_lora_host;

    #[test]
    fn ara_fold_is_scaled_outer_product() {
        // out=2, in=3, r=1: W += scale * up @ down
        let mut w = vec![1.0f32; 6];
        let down = vec![1.0f32, 0.0, 2.0]; // [1, 3]
        let up = vec![3.0f32, 4.0]; // [2, 1]
        merge_comfy_lora_host(&mut w, &down, &up, 2, 3, 1, 1.0);
        // row0 += 3 * [1,0,2] = [3,0,6]; row1 += 4 * [1,0,2] = [4,0,8]
        assert_eq!(w, vec![4.0, 1.0, 7.0, 5.0, 1.0, 9.0]);
        let mut w2 = vec![0.0f32; 6];
        merge_comfy_lora_host(&mut w2, &down, &up, 2, 3, 1, 0.5);
        assert_eq!(w2, vec![1.5, 0.0, 3.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn adapter_shape_and_zero_path() {
        let seq = 2;
        let x = vec![0.0f32; seq * RECOVERED_8B_HIDDEN];
        let norm = vec![1.0f32; RECOVERED_8B_HIDDEN];
        let proj = vec![0.0f32; RECOVERED_8B_OUT * RECOVERED_8B_HIDDEN];
        let down = vec![0.0f32; RECOVERED_8B_BOTTLENECK * RECOVERED_8B_HIDDEN];
        let up = vec![0.0f32; RECOVERED_8B_OUT * RECOVERED_8B_BOTTLENECK];
        // Identity-ish: proj[0,0]=1 so first out channel sees xn[0].
        let mut proj = proj;
        proj[0] = 1.0;
        let mut x = x;
        x[0] = 2.0;
        let y = adapter_forward_host_test(&x, seq, &norm, &proj, &down, &up);
        assert_eq!(y.len(), seq * RECOVERED_8B_OUT);
        // rms: mean(x^2)=4/4096, rsqrt≈32, xn[0]=2*32=64.
        let mean_sq = 4.0f32 / 4096.0;
        let expected = 2.0 * (1.0 / (mean_sq + 1e-6).sqrt());
        assert!((y[0] - expected).abs() < 1e-3, "{} vs {}", y[0], expected);
        assert_eq!(y[1], 0.0);
    }

    #[test]
    fn eight_b_config_matches_checkpoint() {
        let c = DecoderConfig::qwen3_vl_8b_text();
        assert_eq!(c.num_layers(), 24);
        assert_eq!(c.hidden, 4096);
        assert_eq!(c.heads * c.head_dim, 4096);
        assert_eq!(c.kv_heads * c.head_dim, 1024);
        assert_eq!(c.intermediate, 12288);
        assert_eq!(c.layer_prefix, "model.layers");
    }
}
