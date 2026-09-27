//! MMAudio host configs and host algorithms. Spec: docs/ports/mmaudio.md.
//!
//! Everything here follows hkchengrex/MMAudio `974010a` (the commit strobe's
//! sidecar pins): `mmaudio/model/networks.py` (`large_44k_v2`),
//! `model/sequence_config.py` (`CONFIG_44K`), `eval_utils.py` (`load_video`,
//! `generate`) and `model/flow_matching.py` (Euler, `min_sigma = 0`).

pub mod frames;
pub mod tokenize;

/// Registry preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmAudioPreset {
    Large44kV2,
}

impl MmAudioPreset {
    pub fn as_str(self) -> &'static str {
        "mmaudio_large_44k_v2"
    }
    pub fn sample_rate(self) -> u32 {
        44_100
    }
    /// MMAudio is mono.
    pub fn audio_channels(self) -> usize {
        1
    }
    /// `demo.py --duration` default (the training length).
    pub fn duration_s(self) -> f32 {
        8.0
    }
    /// `demo.py --num_steps` default.
    pub fn default_steps(self) -> usize {
        25
    }
    /// `demo.py --cfg_strength` default (also strobe's sidecar).
    pub fn default_cfg(self) -> f32 {
        4.5
    }
    /// Waveform samples per latent frame: mel hop 512 x latent downsample 2.
    pub fn hop_length(self) -> usize {
        1024
    }
}

/// `SequenceConfig` (44k): every sequence length follows from the duration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SequenceConfig {
    pub duration: f64,
    pub sampling_rate: u32,
    pub spectrogram_frame_rate: u32,
    pub latent_downsample_rate: u32,
    pub clip_frame_rate: u32,
    pub sync_frame_rate: u32,
    pub sync_num_frames_per_segment: u32,
    pub sync_step_size: u32,
    pub sync_downsample_rate: u32,
}

impl SequenceConfig {
    pub fn config_44k(duration: f64) -> Self {
        Self {
            duration,
            sampling_rate: 44_100,
            spectrogram_frame_rate: 512,
            latent_downsample_rate: 2,
            clip_frame_rate: 8,
            sync_frame_rate: 25,
            sync_num_frames_per_segment: 16,
            sync_step_size: 8,
            sync_downsample_rate: 2,
        }
    }

    /// `ceil(duration * sr / hop / 2)`, in f64 like Python.
    pub fn latent_seq_len(&self) -> usize {
        (self.duration * f64::from(self.sampling_rate)
            / f64::from(self.spectrogram_frame_rate)
            / f64::from(self.latent_downsample_rate))
        .ceil() as usize
    }

    pub fn clip_seq_len(&self) -> usize {
        (self.duration * f64::from(self.clip_frame_rate)) as usize
    }

    /// Python: `num_frames = duration * 25` (a float), segments by float
    /// floor division, then `int(segments * 16 / 2)`.
    pub fn sync_seq_len(&self) -> usize {
        let num_frames = self.duration * f64::from(self.sync_frame_rate);
        let per = f64::from(self.sync_num_frames_per_segment);
        let segs = ((num_frames - per) / f64::from(self.sync_step_size)).floor() + 1.0;
        (segs * per / f64::from(self.sync_downsample_rate)) as usize
    }

    pub fn num_audio_samples(&self) -> usize {
        self.latent_seq_len()
            * (self.spectrogram_frame_rate * self.latent_downsample_rate) as usize
    }
}

/// `MMAudio(...)` constructor arguments of one variant.
#[derive(Debug, Clone, PartialEq)]
pub struct MmAudioDiTConfig {
    pub latent_dim: usize,
    pub clip_dim: usize,
    pub sync_dim: usize,
    pub text_dim: usize,
    pub hidden_dim: usize,
    /// Joint blocks = `depth - fused_depth`.
    pub depth: usize,
    pub fused_depth: usize,
    pub num_heads: usize,
    pub mlp_ratio: f32,
    pub text_seq_len: usize,
    /// `v2=True`: SiLU input projections, `t_embed` with
    /// `frequency_embedding_size = hidden_dim` and `max_period = 1`.
    pub v2: bool,
    /// `nn.RMSNorm(eps=None)` on q/k takes `finfo(dtype).eps`: 2^-7 for the
    /// bf16 network MMAudio runs (`2^-23` in f32).
    pub qk_norm_eps: f32,
}

impl MmAudioDiTConfig {
    pub fn large_44k_v2() -> Self {
        Self {
            latent_dim: 40,
            clip_dim: 1024,
            sync_dim: 768,
            text_dim: 1024,
            hidden_dim: 64 * 14,
            depth: 21,
            fused_depth: 14,
            num_heads: 14,
            mlp_ratio: 4.0,
            text_seq_len: 77,
            v2: true,
            qk_norm_eps: 1.0 / 128.0,
        }
    }

    /// Two blocks (one joint, one fused), for host tests.
    pub fn tiny() -> Self {
        Self {
            latent_dim: 4,
            clip_dim: 12,
            sync_dim: 10,
            text_dim: 12,
            hidden_dim: 16,
            depth: 2,
            fused_depth: 1,
            num_heads: 2,
            mlp_ratio: 4.0,
            text_seq_len: 5,
            v2: true,
            qk_norm_eps: 1.0 / 128.0,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }

    pub fn joint_depth(&self) -> usize {
        self.depth - self.fused_depth
    }

    /// `MLP` / `ConvMLP` hidden width: `int(2 * h / 3)` rounded up to 256.
    pub fn ffn_dim(&self) -> usize {
        ffn_hidden((self.hidden_dim as f32 * self.mlp_ratio) as usize)
    }

    /// Timestep frequency table width.
    pub fn t_freq_dim(&self) -> usize {
        if self.v2 {
            self.hidden_dim
        } else {
            256
        }
    }

    /// `TimestepEmbedder.freqs`: `(10000 / max_period) / 10000^(2i / F)`.
    pub fn t_freqs(&self) -> Vec<f32> {
        let f = self.t_freq_dim();
        let max_period = if self.v2 { 1.0f32 } else { 10000.0 };
        let scale = 10000.0f32 / max_period;
        (0..f / 2)
            .map(|i| scale * (1.0 / 10000f32.powf((2 * i) as f32 / f as f32)))
            .collect()
    }
}

/// `MLP.__init__` hidden width rule (`multiple_of = 256`).
pub fn ffn_hidden(hidden: usize) -> usize {
    let h = (2 * hidden) / 3;
    256 * h.div_ceil(256)
}

/// `compute_rope_rotations(length, dim, 10000, freq_scaling)`: per position
/// and pair `i`, the angle `pos * freq_scaling / 10000^(2i / dim)` (f32).
pub fn rope_angles(length: usize, dim: usize, freq_scaling: f32) -> Vec<f32> {
    let freqs: Vec<f32> = (0..dim / 2)
        .map(|i| freq_scaling * (1.0 / 10000f32.powf((2 * i) as f32 / dim as f32)))
        .collect();
    let mut out = Vec::with_capacity(length * dim / 2);
    for p in 0..length {
        for f in &freqs {
            out.push(p as f32 * f);
        }
    }
    out
}

/// `F.interpolate(mode='nearest-exact')` source index for each of `out`
/// positions over `input` (PyTorch: `min(floor((dst + 0.5) * in / out), in - 1)`
/// with the scale in f32).
pub fn nearest_exact_indices(input: usize, out: usize) -> Vec<usize> {
    let scale = input as f32 / out as f32;
    (0..out)
        .map(|d| (((d as f32 + 0.5) * scale).floor() as usize).min(input - 1))
        .collect()
}

/// `FlowMatching(min_sigma=0, 'euler', num_steps)`: `torch.linspace(0, 1, n + 1)`
/// in f32 (torch computes `start + i * step` for the first half and
/// `end - (n - i) * step` for the second).
pub fn euler_times(num_steps: usize) -> Vec<f32> {
    let n = num_steps + 1;
    if n == 1 {
        return vec![0.0];
    }
    let step = 1.0f32 / (n - 1) as f32;
    let half = n / 2;
    (0..n)
        .map(|i| {
            if i < half {
                i as f32 * step
            } else {
                1.0 - (n - 1 - i) as f32 * step
            }
        })
        .collect()
}

/// Round to bfloat16 (round to nearest even), returned as f32.
pub fn bf16_round(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    f32::from_bits(b.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

/// The two VAE normalisation tables of the 128-band 44k mel (`DATA_MEAN_128D`,
/// `DATA_STD_128D` in `ext/autoencoder/vae.py`).
pub const DATA_MEAN_128D: [f32; 128] = [
    -3.3462, -2.6723, -2.4893, -2.3143, -2.2664, -2.3317, -2.1802, -2.4006, -2.2357, -2.4597,
    -2.3717, -2.4690, -2.5142, -2.4919, -2.6610, -2.5047, -2.7483, -2.5926, -2.7462, -2.7033,
    -2.7386, -2.8112, -2.7502, -2.9594, -2.7473, -3.0035, -2.8891, -2.9922, -2.9856, -3.0157,
    -3.1191, -2.9893, -3.1718, -3.0745, -3.1879, -3.2310, -3.1424, -3.2296, -3.2791, -3.2782,
    -3.2756, -3.3134, -3.3509, -3.3750, -3.3951, -3.3698, -3.4505, -3.4509, -3.5089, -3.4647,
    -3.5536, -3.5788, -3.5867, -3.6036, -3.6400, -3.6747, -3.7072, -3.7279, -3.7283, -3.7795,
    -3.8259, -3.8447, -3.8663, -3.9182, -3.9605, -3.9861, -4.0105, -4.0373, -4.0762, -4.1121,
    -4.1488, -4.1874, -4.2461, -4.3170, -4.3639, -4.4452, -4.5282, -4.6297, -4.7019, -4.7960,
    -4.8700, -4.9507, -5.0303, -5.0866, -5.1634, -5.2342, -5.3242, -5.4053, -5.4927, -5.5712,
    -5.6464, -5.7052, -5.7619, -5.8410, -5.9188, -6.0103, -6.0955, -6.1673, -6.2362, -6.3120,
    -6.3926, -6.4797, -6.5565, -6.6511, -6.8130, -6.9961, -7.1275, -7.2457, -7.3576, -7.4663,
    -7.6136, -7.7469, -7.8815, -8.0132, -8.1515, -8.3071, -8.4722, -8.7418, -9.3975, -9.6628,
    -9.7671, -9.8863, -9.9992, -10.0860, -10.1709, -10.5418, -11.2795, -11.3861,
];

pub const DATA_STD_128D: [f32; 128] = [
    2.3804, 2.4368, 2.3772, 2.3145, 2.2803, 2.2510, 2.2316, 2.2083, 2.1996, 2.1835, 2.1769, 2.1659,
    2.1631, 2.1618, 2.1540, 2.1606, 2.1571, 2.1567, 2.1612, 2.1579, 2.1679, 2.1683, 2.1634, 2.1557,
    2.1668, 2.1518, 2.1415, 2.1449, 2.1406, 2.1350, 2.1313, 2.1415, 2.1281, 2.1352, 2.1219, 2.1182,
    2.1327, 2.1195, 2.1137, 2.1080, 2.1179, 2.1036, 2.1087, 2.1036, 2.1015, 2.1068, 2.0975, 2.0991,
    2.0902, 2.1015, 2.0857, 2.0920, 2.0893, 2.0897, 2.0910, 2.0881, 2.0925, 2.0873, 2.0960, 2.0900,
    2.0957, 2.0958, 2.0978, 2.0936, 2.0886, 2.0905, 2.0845, 2.0855, 2.0796, 2.0840, 2.0813, 2.0817,
    2.0838, 2.0840, 2.0917, 2.1061, 2.1431, 2.1976, 2.2482, 2.3055, 2.3700, 2.4088, 2.4372, 2.4609,
    2.4731, 2.4847, 2.5072, 2.5451, 2.5772, 2.6147, 2.6529, 2.6596, 2.6645, 2.6726, 2.6803, 2.6812,
    2.6899, 2.6916, 2.6931, 2.6998, 2.7062, 2.7262, 2.7222, 2.7158, 2.7041, 2.7485, 2.7491, 2.7451,
    2.7485, 2.7233, 2.7297, 2.7233, 2.7145, 2.6958, 2.6788, 2.6439, 2.6007, 2.4786, 2.2469, 2.1877,
    2.1392, 2.0717, 2.0107, 1.9676, 1.9140, 1.7102, 0.9101, 0.7164,
];

/// The 44k latent VAE (`VAE_44k`): 128-band mel, 40-d latent, width 512.
#[derive(Debug, Clone, PartialEq)]
pub struct MmAudioVaeConfig {
    pub data_dim: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub ch_mult: Vec<usize>,
    pub num_res_blocks: usize,
    /// Decoder levels (encoder `down_layers` + 1) followed by an upsample.
    pub up_levels: Vec<usize>,
    pub clip_act: f32,
}

impl MmAudioVaeConfig {
    pub fn vae_44k() -> Self {
        Self {
            data_dim: 128,
            embed_dim: 40,
            hidden_dim: 512,
            ch_mult: vec![1, 2, 4],
            num_res_blocks: 2,
            up_levels: vec![1],
            clip_act: 256.0,
        }
    }

    pub fn tiny() -> Self {
        Self {
            data_dim: 6,
            embed_dim: 4,
            hidden_dim: 8,
            ch_mult: vec![1, 2, 4],
            num_res_blocks: 2,
            up_levels: vec![1],
            clip_act: 256.0,
        }
    }
}

/// BigVGAN v2 generator hyper-parameters (`config.json` of
/// `nvidia/bigvgan_v2_44khz_128band_512x`).
#[derive(Debug, Clone, PartialEq)]
pub struct BigVganConfig {
    pub num_mels: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub use_tanh_at_final: bool,
    pub use_bias_at_final: bool,
}

impl BigVganConfig {
    pub fn v2_44k_128band_512x() -> Self {
        Self {
            num_mels: 128,
            upsample_rates: vec![8, 4, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![16, 8, 4, 4, 4, 4],
            upsample_initial_channel: 1536,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            use_tanh_at_final: false,
            use_bias_at_final: false,
        }
    }

    pub fn tiny() -> Self {
        Self {
            num_mels: 6,
            upsample_rates: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4],
            upsample_initial_channel: 16,
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1, 3]],
            use_tanh_at_final: false,
            use_bias_at_final: false,
        }
    }

    pub fn hop(&self) -> usize {
        self.upsample_rates.iter().product()
    }
}

/// Everything one MMAudio variant needs.
#[derive(Debug, Clone, PartialEq)]
pub struct MmAudioConfig {
    pub dit: MmAudioDiTConfig,
    pub vae: MmAudioVaeConfig,
    pub vocoder: BigVganConfig,
    pub sample_rate: u32,
}

impl MmAudioConfig {
    pub fn for_preset(preset: MmAudioPreset) -> Self {
        match preset {
            MmAudioPreset::Large44kV2 => Self {
                dit: MmAudioDiTConfig::large_44k_v2(),
                vae: MmAudioVaeConfig::vae_44k(),
                vocoder: BigVganConfig::v2_44k_128band_512x(),
                sample_rate: preset.sample_rate(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_config_matches_upstream_asserts() {
        // sequence_config.py __main__: CONFIG_44K at 8 s.
        let c = SequenceConfig::config_44k(8.0);
        assert_eq!(c.latent_seq_len(), 345);
        assert_eq!(c.clip_seq_len(), 64);
        assert_eq!(c.sync_seq_len(), 192);
        assert_eq!(c.num_audio_samples(), 353_280);
    }

    #[test]
    fn large_dims() {
        let c = MmAudioDiTConfig::large_44k_v2();
        assert_eq!(c.hidden_dim, 896);
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.joint_depth(), 7);
        assert_eq!(c.ffn_dim(), 2560);
    }

    #[test]
    fn linspace_matches_torch() {
        let t = euler_times(25);
        assert_eq!(t.len(), 26);
        assert_eq!(t[0], 0.0);
        assert_eq!(t[25], 1.0);
        assert!((t[1] - 0.04).abs() < 1e-7);
    }

    #[test]
    fn nearest_exact_upsample() {
        // 3 -> 7: floor((d + .5) * 3 / 7)
        assert_eq!(nearest_exact_indices(3, 7), vec![0, 0, 1, 1, 1, 2, 2]);
        assert_eq!(nearest_exact_indices(4, 8), vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }
}
