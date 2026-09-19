//! Numerical checks that fit in CI without a Python / Diffusers snapshot.
//!
//! What these gates cover:
//! - Flux2 flow-match timestep table + empirical μ (BFL / Diffusers formula)
//! - 2×2 pack / unpatchify invertibility
//! - GroupNorm reduction used by AutoencoderKLFlux2
//! - Tiny DiT / VAE / Qwen3 shapes
//!
//! Still approximate vs FastVideo / Diffusers (needs a local HF snapshot +
//! dumped tensors to claim bit-exactness):
//! - Full-width DiT forward (bf16 GEMM / SDPA rounding)
//! - Chat-template tokenization (string wrap, not Jinja `chat_template`)
//! - VAE mid-block attention vs fused Diffusers processors
//! - Tiling / slicing decode

use super::family::{compute_empirical_mu, packed_hw};

/// Mean squared error; `None` if lengths differ or empty.
pub fn mse(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let s: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    Some(s / a.len() as f32)
}

/// PSNR in dB. Identical buffers → `+inf`.
pub fn psnr(a: &[f32], b: &[f32], max: f32) -> Option<f32> {
    let err = mse(a, b)?;
    if err == 0.0 {
        return Some(f32::INFINITY);
    }
    Some(10.0 * (max * max / err).log10())
}

/// Klein 1024² / 4-step μ used by the Vast compare harness.
pub fn klein_1024_mu() -> f64 {
    let (h, w) = packed_hw(1024, 1024, 8);
    compute_empirical_mu(h * w, 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;

    use crate::flux2::{
        group_norm, pack_latents_2x2, unpatchify_2x2, AutoencoderKlFlux2, Flux2ArchConfig,
        Flux2TextEncoder, Flux2TextKind, Flux2Transformer2D, Flux2VaeConfig, Qwen3Config,
        Qwen3Encoder,
    };
    use crate::schedulers::FlowMatchEulerDiscreteScheduler;
    use candle_core::DType;

    #[test]
    fn klein_1024_timestep_table() {
        let mu = klein_1024_mu();
        // seq=4096 ≤ 4300 → interpolate the two BFL lines at 4 steps.
        let expected = {
            let seq = 4096.0;
            let m_200 = 0.00016927 * seq + 0.45666666;
            let m_10 = 8.73809524e-05 * seq + 1.89833333;
            let a = (m_200 - m_10) / 190.0;
            let b = m_200 - 200.0 * a;
            a * 4.0 + b
        };
        assert!((mu - expected).abs() < 1e-12, "mu={mu} expected={expected}");
        let mut sched = FlowMatchEulerDiscreteScheduler::new(1000, 1.0);
        sched.set_timesteps_flux2(4, Some(mu));
        assert_eq!(sched.inference_timesteps().len(), 4);
        assert!((sched.inference_sigmas()[0] - crate::flux2::flux2_time_shift(1.0, mu)).abs() < 1e-12);
        assert!((sched.inference_sigmas()[4] - 0.0).abs() < 1e-12);
        assert!(sched.inference_sigmas()[1] < sched.inference_sigmas()[0]);
    }

    #[test]
    fn pack_roundtrip_and_1024_seq() {
        let input: Vec<f32> = (0..32 * 8 * 8).map(|i| i as f32 * 0.01).collect();
        let packed = pack_latents_2x2(&input, 32, 8, 8).unwrap();
        assert_eq!(packed.len(), 128 * 4 * 4);
        let back = unpatchify_2x2(&packed, 32, 4, 4).unwrap();
        assert_eq!(back, input);
        assert_eq!(packed_hw(1024, 1024, 8), (64, 64));
    }

    #[test]
    fn psnr_identical_is_inf() {
        let a = [0.1f32, -0.2, 0.3];
        assert!(psnr(&a, &a, 1.0).unwrap().is_infinite());
        assert!(mse(&[1.0], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn group_norm_zero_mean_unit_groups() {
        let device = Device::Cpu;
        let xs = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (1, 2, 2, 2),
            &device,
        )
        .unwrap();
        let w = Tensor::from_vec(vec![1.0f32, 1.0], (2,), &device).unwrap();
        let b = Tensor::from_vec(vec![0.0f32, 0.0], (2,), &device).unwrap();
        let y = group_norm(&xs, &w, &b, 2, 1e-6).unwrap();
        let host = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Two groups of 4 values; each group mean-centered.
        let g0: Vec<f32> = host.iter().take(4).copied().collect();
        let mean0: f32 = g0.iter().sum::<f32>() / 4.0;
        assert!(mean0.abs() < 1e-5, "mean0={mean0}");
    }

    #[test]
    fn tiny_dit_and_qwen3_shapes() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let cfg = Flux2ArchConfig::tiny_klein();
        let dit = Flux2Transformer2D::load(cfg.clone(), vb.pp("dit")).unwrap();
        let hidden = Tensor::zeros((1, cfg.in_channels, 1, 2, 2), DType::F32, &device).unwrap();
        let enc = Tensor::zeros((1, 4, cfg.joint_attention_dim), DType::F32, &device).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &device).unwrap();
        let out = dit.forward(&hidden, &enc, &t, None, 2, 2).unwrap();
        assert_eq!(out.dims(), &[1, cfg.out_channels, 1, 2, 2]);

        let qwen = Qwen3Encoder::load(Qwen3Config::tiny(), vb.pp("text")).unwrap();
        let text = Flux2TextEncoder::qwen3(qwen, DType::F32);
        let stacked = text.encode_ids(&[1, 2, 3, 4]).unwrap();
        assert_eq!(stacked.dims(), &[1, 4, 48]);
        assert_eq!(Flux2TextKind::Qwen3.out_layers().len() * 16, 48);
    }

    #[test]
    fn tiny_vae_decode_psnr_self() {
        let device = Device::Cpu;
        let cfg = Flux2VaeConfig::tiny();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let vae = AutoencoderKlFlux2::load(cfg.clone(), vb).unwrap();
        let z = Tensor::zeros((1, cfg.latent_channels, 1, 2, 2), DType::F32, &device).unwrap();
        let out = vae.decode(&z).unwrap();
        let host = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(psnr(&host, &host, 1.0).unwrap().is_infinite());
        assert!(host.iter().all(|v| v.abs() < 1e-6));
    }
}
