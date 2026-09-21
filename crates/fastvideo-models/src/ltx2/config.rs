//! LTX-2 19B component configs.
//!
//! One struct per `config.json` in the diffusers layout of `Lightricks/LTX-2`
//! (`transformer/`, `vae/`, `audio_vae/`, `connectors/`, `vocoder/`,
//! `scheduler/`, `text_encoder/`), field names kept identical to the JSON keys
//! so a serde derive can be dropped in once this crate takes the dependency.
//! `fastvideo-models` has no serde today, so the published values are
//! hand-written constructors, checked below against the numbers the weight
//! headers imply. See docs/ports/ltx2.md and docs/ports/ltx25.md.

/// Checkpoint family: LTX-2.0 dev/distilled vs LTX-2.5 distilled stage-1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ltx2ModelVersion {
    V20,
    V25,
}

/// How `LTX2Attention` rotates q/k. LTX-2.0 ships `"split"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ltx2RopeType {
    /// Adjacent pairs `(x[2i], x[2i+1])`, one table shared by every head.
    Interleaved,
    /// rotate_half within each head, with a *per-head* slice of the frequency
    /// vector (`transformer_ltx2.py:1054-1076`).
    Split,
}

/// `transformer/config.json` — `LTX2VideoTransformer3DModel`.
#[derive(Debug, Clone, PartialEq)]
pub struct Ltx2TransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub patch_size_t: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub cross_attention_dim: usize,
    pub vae_scale_factors: [usize; 3],
    pub pos_embed_max_pos: usize,
    pub base_height: usize,
    pub base_width: usize,
    pub audio_in_channels: usize,
    pub audio_out_channels: usize,
    pub audio_patch_size: usize,
    pub audio_patch_size_t: usize,
    pub audio_num_attention_heads: usize,
    pub audio_attention_head_dim: usize,
    pub audio_cross_attention_dim: usize,
    pub audio_scale_factor: usize,
    pub audio_pos_embed_max_pos: usize,
    pub audio_sampling_rate: usize,
    pub audio_hop_length: usize,
    pub num_layers: usize,
    /// `"gelu-approximate"`: `Linear(d, 4d)` → tanh-GELU → `Linear(4d, d)`.
    pub ff_mult: usize,
    pub norm_eps: f64,
    pub norm_elementwise_affine: bool,
    pub caption_channels: usize,
    pub attention_bias: bool,
    pub attention_out_bias: bool,
    pub rope_theta: f64,
    pub rope_double_precision: bool,
    pub rope_type: Ltx2RopeType,
    pub causal_offset: usize,
    pub timestep_scale_multiplier: f64,
    pub cross_attn_timestep_scale_multiplier: f64,
    /// Sinusoid width in front of every `LTX2AdaLayerNormSingle` MLP.
    pub timestep_proj_dim: usize,
    // Absent from the 2.0 JSON, so they take the class defaults — spelled out
    // because each one switches a code path off (LTX-2.3/2.5 features).
    pub gated_attn: bool,
    pub cross_attn_mod: bool,
    pub audio_gated_attn: bool,
    pub audio_cross_attn_mod: bool,
    pub use_prompt_embeddings: bool,
    pub perturbed_attn: bool,
    pub ff_bias: bool,
    pub audio_ff_bias: bool,
    pub use_prompt_adaln_single: bool,
    pub use_keyframes_abs_pos_embedding: bool,
}

impl Ltx2TransformerConfig {
    pub fn ltx2_19b() -> Self {
        Self {
            in_channels: 128,
            out_channels: 128,
            patch_size: 1,
            patch_size_t: 1,
            num_attention_heads: 32,
            attention_head_dim: 128,
            cross_attention_dim: 4096,
            vae_scale_factors: [8, 32, 32],
            pos_embed_max_pos: 20,
            base_height: 2048,
            base_width: 2048,
            audio_in_channels: 128,
            audio_out_channels: 128,
            audio_patch_size: 1,
            audio_patch_size_t: 1,
            audio_num_attention_heads: 32,
            audio_attention_head_dim: 64,
            audio_cross_attention_dim: 2048,
            audio_scale_factor: 4,
            audio_pos_embed_max_pos: 20,
            audio_sampling_rate: 16000,
            audio_hop_length: 160,
            num_layers: 48,
            ff_mult: 4,
            norm_eps: 1e-6,
            norm_elementwise_affine: false,
            caption_channels: 3840,
            attention_bias: true,
            attention_out_bias: true,
            rope_theta: 10000.0,
            rope_double_precision: true,
            rope_type: Ltx2RopeType::Split,
            causal_offset: 1,
            timestep_scale_multiplier: 1000.0,
            cross_attn_timestep_scale_multiplier: 1000.0,
            timestep_proj_dim: 256,
            gated_attn: false,
            cross_attn_mod: false,
            audio_gated_attn: false,
            audio_cross_attn_mod: false,
            use_prompt_embeddings: true,
            perturbed_attn: false,
            ff_bias: true,
            audio_ff_bias: true,
            use_prompt_adaln_single: true,
            use_keyframes_abs_pos_embedding: false,
        }
    }

    /// `Lightricks/LTX-2.5-Diffusers` distilled DiT (`transformer/config.json`).
    pub fn ltx2_5_22b() -> Self {
        Self {
            gated_attn: true,
            cross_attn_mod: true,
            audio_gated_attn: true,
            audio_cross_attn_mod: true,
            use_prompt_embeddings: false,
            perturbed_attn: true,
            ff_bias: false,
            audio_ff_bias: true,
            use_prompt_adaln_single: true,
            use_keyframes_abs_pos_embedding: true,
            ..Self::ltx2_19b()
        }
    }

    /// Video stream width: 32 × 128 = 4096.
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Audio stream width: 32 × 64 = 2048.
    pub fn audio_inner_dim(&self) -> usize {
        self.audio_num_attention_heads * self.audio_attention_head_dim
    }

    /// Width q/k/v live in for the audio↔video cross-attentions. Both
    /// directions use the *audio* head layout (`transformer_ltx2.py:529-557`).
    pub fn av_cross_inner_dim(&self) -> usize {
        self.audio_inner_dim()
    }

    pub fn ff_inner_dim(&self) -> usize {
        self.inner_dim() * self.ff_mult
    }

    pub fn audio_ff_inner_dim(&self) -> usize {
        self.audio_inner_dim() * self.ff_mult
    }

    /// Latent grid `[frames, height, width]` for a pixel-space request.
    /// `pipeline_ltx2.py:1261-1263`.
    pub fn latent_grid(&self, num_frames: usize, height: usize, width: usize) -> [usize; 3] {
        let [st, sh, sw] = self.vae_scale_factors;
        [(num_frames - 1) / st + 1, height / sh, width / sw]
    }

    /// Video tokens = latent cells, since both patch sizes are 1.
    pub fn video_tokens(&self, num_frames: usize, height: usize, width: usize) -> usize {
        let [f, h, w] = self.latent_grid(num_frames, height, width);
        (f / self.patch_size_t) * (h / self.patch_size) * (w / self.patch_size)
    }

    /// 16000 / 160 / 4 = 25 audio latent frames per second.
    pub fn audio_latents_per_second(&self) -> f64 {
        self.audio_sampling_rate as f64 / self.audio_hop_length as f64 / self.audio_scale_factor as f64
    }

    /// Audio tokens for a clip: `round(num_frames / fps * 25)` with Python's
    /// round-half-to-even (`pipeline_ltx2.py:1295-1299`).
    pub fn audio_tokens(&self, num_frames: usize, frame_rate: f64) -> usize {
        round_half_even(num_frames as f64 / frame_rate * self.audio_latents_per_second()) as usize
    }

    /// Rotary frequencies per positional axis: `dim // (2 * axes)`.
    pub fn rope_freqs_per_axis(dim: usize, axes: usize) -> usize {
        dim / (2 * axes)
    }

    /// Identity (cos 1, sin 0) slots prepended so the table reaches `dim / 2`.
    pub fn rope_pad(dim: usize, axes: usize) -> usize {
        dim / 2 - Self::rope_freqs_per_axis(dim, axes) * axes
    }
}

/// Python 3 `round()`: ties go to the even neighbour.
pub fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// `LTX2VideoUpsampler3d.upsample_type` in diffusers (`autoencoder_kl_ltx2.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ltx2VaeUpsampleKind {
    SpatioTemporal,
    Temporal,
    Spatial,
}

impl Ltx2VaeUpsampleKind {
    /// Depth-to-space stride `(T, H, W)` for this upsampler kind.
    pub fn stride(self) -> (usize, usize, usize) {
        match self {
            Self::SpatioTemporal => (2, 2, 2),
            Self::Temporal => (2, 1, 1),
            Self::Spatial => (1, 2, 2),
        }
    }

    pub fn stride_product(self) -> usize {
        let (t, h, w) = self.stride();
        t * h * w
    }
}

/// One decoder upsampler before a stage's resnets (`decoder_stages`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ltx2VaeDecoderUpsampler {
    pub in_channels: usize,
    pub conv_out_channels: usize,
    pub stride: (usize, usize, usize),
    pub residual: bool,
    /// Drop the first output frame when the temporal stride is greater than 1.
    pub drop_first_frame: bool,
}

/// One decoder stage: `mid_block` or an `up_blocks.*` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ltx2VaeDecoderStage {
    pub channels: usize,
    pub resnet_layers: usize,
    pub upsampler: Option<Ltx2VaeDecoderUpsampler>,
}

/// `vae/config.json` — `AutoencoderKLLTX2Video`.
#[derive(Debug, Clone, PartialEq)]
pub struct Ltx2VideoVaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub decoder_block_out_channels: Vec<usize>,
    pub layers_per_block: Vec<usize>,
    pub decoder_layers_per_block: Vec<usize>,
    pub decoder_spatio_temporal_scaling: Vec<bool>,
    pub decoder_inject_noise: Vec<bool>,
    pub upsample_residual: Vec<bool>,
    pub upsample_factor: Vec<usize>,
    pub upsample_type: Vec<Ltx2VaeUpsampleKind>,
    pub timestep_conditioning: bool,
    pub patch_size: usize,
    pub patch_size_t: usize,
    pub resnet_norm_eps: f64,
    /// `PerChannelRMSNorm` has its own eps, not `resnet_norm_eps`
    /// (`autoencoder_kl_ltx2.py:40`).
    pub pixel_norm_eps: f64,
    pub scaling_factor: f64,
    pub encoder_causal: bool,
    pub decoder_causal: bool,
    /// `"zeros"` for the encoder, `"reflect"` for the decoder.
    pub decoder_reflect_padding: bool,
    pub spatial_compression_ratio: usize,
    pub temporal_compression_ratio: usize,
}

impl Ltx2VideoVaeConfig {
    pub fn ltx2_19b() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 128,
            block_out_channels: vec![256, 512, 1024, 2048],
            decoder_block_out_channels: vec![256, 512, 1024],
            layers_per_block: vec![4, 6, 6, 2, 2],
            decoder_layers_per_block: vec![5, 5, 5, 5],
            decoder_spatio_temporal_scaling: vec![true, true, true],
            decoder_inject_noise: vec![false, false, false, false],
            upsample_residual: vec![true, true, true],
            upsample_factor: vec![2, 2, 2],
            upsample_type: vec![
                Ltx2VaeUpsampleKind::SpatioTemporal,
                Ltx2VaeUpsampleKind::SpatioTemporal,
                Ltx2VaeUpsampleKind::SpatioTemporal,
            ],
            timestep_conditioning: false,
            patch_size: 4,
            patch_size_t: 1,
            resnet_norm_eps: 1e-6,
            pixel_norm_eps: 1e-8,
            scaling_factor: 1.0,
            encoder_causal: true,
            decoder_causal: false,
            decoder_reflect_padding: true,
            spatial_compression_ratio: 32,
            temporal_compression_ratio: 8,
        }
    }

    /// Conv video VAE in `Lightricks/LTX-2.5-Diffusers` (`vae/config.json`).
    pub fn ltx2_5_22b() -> Self {
        Self {
            block_out_channels: vec![256, 512, 1024, 1024],
            decoder_block_out_channels: vec![256, 512, 512, 1024],
            layers_per_block: vec![4, 6, 4, 2, 2],
            decoder_layers_per_block: vec![4, 6, 4, 2, 2],
            decoder_spatio_temporal_scaling: vec![true, true, true, true],
            decoder_inject_noise: vec![false, false, false, false, false],
            upsample_residual: vec![false, false, false, false],
            upsample_factor: vec![2, 2, 1, 2],
            upsample_type: vec![
                Ltx2VaeUpsampleKind::SpatioTemporal,
                Ltx2VaeUpsampleKind::SpatioTemporal,
                Ltx2VaeUpsampleKind::Temporal,
                Ltx2VaeUpsampleKind::Spatial,
            ],
            decoder_reflect_padding: false,
            ..Self::ltx2_19b()
        }
    }

    /// Decoder stages in execution order. The first is `mid_block` (no upsampler);
    /// each later stage's upsampler runs *before* its resnets.
    pub fn decoder_stages(&self) -> Vec<Ltx2VaeDecoderStage> {
        let mut widths: Vec<usize> = self.decoder_block_out_channels.to_vec();
        widths.reverse();
        let mut layers: Vec<usize> = self.decoder_layers_per_block.to_vec();
        layers.reverse();
        let mut factors: Vec<usize> = self.upsample_factor.to_vec();
        factors.reverse();
        // `upsample_type` is listed in decode order (`up_blocks.0` …), same as the
        // reversed execution walk below — unlike `upsample_factor`, which is stored
        // in the checkpoint's encoder order.
        let kinds = self.upsample_type.clone();
        let mut residuals: Vec<bool> = self.upsample_residual.to_vec();
        residuals.reverse();
        let mut out = vec![Ltx2VaeDecoderStage {
            channels: widths[0],
            resnet_layers: layers[0],
            upsampler: None,
        }];
        for (i, &w) in widths.iter().enumerate() {
            let factor = factors[i];
            let ch = w / factor;
            let kind = kinds[i];
            let stride = kind.stride();
            let sp = kind.stride_product();
            let conv_out = (w * sp) / factor;
            debug_assert_eq!(out.last().map(|s| s.channels), Some(w));
            out.push(Ltx2VaeDecoderStage {
                channels: ch,
                resnet_layers: layers[i + 1],
                upsampler: Some(Ltx2VaeDecoderUpsampler {
                    in_channels: w,
                    conv_out_channels: conv_out,
                    stride,
                    residual: residuals[i],
                    drop_first_frame: stride.0 > 1,
                }),
            });
        }
        out
    }

    /// Frames out of the decoder: every ×2 temporal stage drops its first
    /// frame, so `F → 8(F-1)+1`.
    pub fn decoded_frames(&self, latent_frames: usize) -> usize {
        (latent_frames - 1) * self.temporal_compression_ratio + 1
    }
}

/// `audio_vae/config.json` — `AutoencoderKLLTX2Audio`.
#[derive(Debug, Clone, PartialEq)]
pub struct Ltx2AudioVaeConfig {
    pub base_channels: usize,
    pub output_channels: usize,
    pub ch_mult: [usize; 3],
    pub num_res_blocks: usize,
    pub in_channels: usize,
    pub resolution: usize,
    pub latent_channels: usize,
    /// `norm_type: "pixel"` — channel RMS with this eps, no affine.
    pub pixel_norm_eps: f64,
    /// `causality_axis: "height"`: the time axis (dim 2) is padded on the top only.
    pub causal_time_axis: bool,
    pub mid_block_add_attention: bool,
    pub sample_rate: usize,
    pub mel_hop_length: usize,
    pub mel_bins: usize,
    pub double_z: bool,
    /// `LATENT_DOWNSAMPLE_FACTOR`, used for both time and mel axes.
    pub latent_downsample_factor: usize,
}

impl Ltx2AudioVaeConfig {
    pub fn ltx2_19b() -> Self {
        Self {
            base_channels: 128,
            output_channels: 2,
            ch_mult: [1, 2, 4],
            num_res_blocks: 2,
            in_channels: 2,
            resolution: 256,
            latent_channels: 8,
            pixel_norm_eps: 1e-6,
            causal_time_axis: true,
            mid_block_add_attention: false,
            sample_rate: 16000,
            mel_hop_length: 160,
            mel_bins: 64,
            double_z: true,
            latent_downsample_factor: 4,
        }
    }

    /// 64 / 4 = 16 latent mel bins.
    pub fn latent_mel_bins(&self) -> usize {
        self.mel_bins / self.latent_downsample_factor
    }

    /// One DiT audio token = one latent frame: 8 channels × 16 bins = 128.
    pub fn token_channels(&self) -> usize {
        self.latent_channels * self.latent_mel_bins()
    }

    /// Mel frames out of the decoder: `max(4L - 3, 1)`
    /// (`autoencoder_kl_ltx2_audio.py:610-613`).
    pub fn mel_frames(&self, latent_frames: usize) -> usize {
        let f = self.latent_downsample_factor;
        (latent_frames * f).saturating_sub(f - 1).max(1)
    }
}

/// `connectors/config.json` — `LTX2TextConnectors`.
#[derive(Debug, Clone, PartialEq)]
pub struct Ltx2ConnectorsConfig {
    pub caption_channels: usize,
    /// Gemma hidden states stacked per token: embeddings + 48 layers.
    pub text_proj_in_factor: usize,
    pub video_connector_num_attention_heads: usize,
    pub video_connector_attention_head_dim: usize,
    pub video_connector_num_layers: usize,
    pub video_connector_num_learnable_registers: usize,
    pub audio_connector_num_attention_heads: usize,
    pub audio_connector_attention_head_dim: usize,
    pub audio_connector_num_layers: usize,
    pub audio_connector_num_learnable_registers: usize,
    pub connector_rope_base_seq_len: usize,
    pub rope_theta: f64,
    pub rope_double_precision: bool,
    pub causal_temporal_positioning: bool,
    pub rope_type: Ltx2RopeType,
    pub per_modality_projections: bool,
    pub proj_bias: bool,
    pub video_gated_attn: bool,
    pub audio_gated_attn: bool,
    pub video_hidden_dim: usize,
    pub audio_hidden_dim: usize,
    /// `per_layer_masked_mean_norm(scale_factor=8)` default, `connectors.py:18`.
    pub norm_scale_factor: f64,
    pub norm_eps: f64,
}

impl Ltx2ConnectorsConfig {
    pub fn ltx2_19b() -> Self {
        Self {
            caption_channels: 3840,
            text_proj_in_factor: 49,
            video_connector_num_attention_heads: 30,
            video_connector_attention_head_dim: 128,
            video_connector_num_layers: 2,
            video_connector_num_learnable_registers: 128,
            audio_connector_num_attention_heads: 30,
            audio_connector_attention_head_dim: 128,
            audio_connector_num_layers: 2,
            audio_connector_num_learnable_registers: 128,
            connector_rope_base_seq_len: 4096,
            rope_theta: 10000.0,
            rope_double_precision: true,
            causal_temporal_positioning: false,
            rope_type: Ltx2RopeType::Split,
            per_modality_projections: false,
            proj_bias: false,
            video_gated_attn: false,
            audio_gated_attn: false,
            video_hidden_dim: 3840,
            audio_hidden_dim: 3840,
            norm_scale_factor: 8.0,
            norm_eps: 1e-6,
        }
    }

    /// `Lightricks/LTX-2.5-Diffusers` (`connectors/config.json`).
    pub fn ltx2_5_22b() -> Self {
        Self {
            video_connector_num_attention_heads: 32,
            video_connector_attention_head_dim: 128,
            video_connector_num_layers: 8,
            audio_connector_num_attention_heads: 32,
            audio_connector_attention_head_dim: 64,
            audio_connector_num_layers: 8,
            per_modality_projections: true,
            proj_bias: true,
            video_gated_attn: true,
            audio_gated_attn: true,
            video_hidden_dim: 4096,
            audio_hidden_dim: 2048,
            ..Self::ltx2_19b()
        }
    }

    /// `text_proj_in` input width: 3840 × 49 = 188160.
    pub fn text_proj_in_features(&self) -> usize {
        self.caption_channels * self.text_proj_in_factor
    }

    /// Video connector stream width (`video_hidden_dim` on 2.5).
    pub fn inner_dim(&self) -> usize {
        self.video_hidden_dim
    }

    pub fn audio_inner_dim(&self) -> usize {
        self.audio_hidden_dim
    }
}

/// `vocoder/config.json` — `LTX2Vocoder` (HiFi-GAN generator).
#[derive(Debug, Clone, PartialEq)]
pub struct Ltx2VocoderConfig {
    pub in_channels: usize,
    pub hidden_channels: usize,
    pub out_channels: usize,
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_factors: Vec<usize>,
    pub resnet_kernel_sizes: [usize; 3],
    pub resnet_dilations: [[usize; 3]; 3],
    pub leaky_relu_negative_slope: f64,
    /// `act_out` is a bare `nn.LeakyReLU()`: slope 0.01, *not* the 0.1 above
    /// (`vocoder.py:367-369`).
    pub final_leaky_relu_negative_slope: f64,
    pub final_tanh: bool,
    pub output_sampling_rate: usize,
    /// `LTX2VocoderWithBWE` (LTX-2.5): SnakeBeta + band-width extension stack.
    pub with_bwe: bool,
}

impl Ltx2VocoderConfig {
    pub fn ltx2_19b() -> Self {
        Self {
            in_channels: 128,
            hidden_channels: 1024,
            out_channels: 2,
            upsample_kernel_sizes: vec![16, 15, 8, 4, 4],
            upsample_factors: vec![6, 5, 2, 2, 2],
            resnet_kernel_sizes: [3, 7, 11],
            resnet_dilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            leaky_relu_negative_slope: 0.1,
            final_leaky_relu_negative_slope: 0.01,
            final_tanh: true,
            output_sampling_rate: 24000,
            with_bwe: false,
        }
    }

    /// `LTX2VocoderWithBWE` in LTX-2.5 (`vocoder/config.json`); BWE sub-stack not
    /// modeled here beyond the flag and main upsampler geometry.
    pub fn ltx2_5_22b_bwe() -> Self {
        Self {
            in_channels: 128,
            hidden_channels: 1536,
            out_channels: 2,
            upsample_kernel_sizes: vec![11, 4, 4, 4, 4, 4],
            upsample_factors: vec![5, 2, 2, 2, 2, 2],
            output_sampling_rate: 48000,
            final_tanh: false,
            with_bwe: true,
            ..Self::ltx2_19b()
        }
    }

    /// 6·5·2·2·2 = 240 samples per mel frame.
    pub fn total_upsample_factor(&self) -> usize {
        self.upsample_factors.iter().product()
    }

    /// `ConvTranspose1d` padding per stage: `(kernel - stride) / 2`.
    pub fn upsample_padding(&self, stage: usize) -> usize {
        (self.upsample_kernel_sizes[stage] - self.upsample_factors[stage]) / 2
    }

    /// Channels after upsample stage `stage`: 1024 → 512 → 256 → 128 → 64 → 32.
    pub fn stage_channels(&self, stage: usize) -> usize {
        self.hidden_channels >> (stage + 1)
    }

    /// Exact waveform length: every stage satisfies `k - 2p = s`, so
    /// `(L-1)s - 2p + k = L·s` and the product telescopes.
    pub fn waveform_samples(&self, mel_frames: usize) -> usize {
        (0..self.upsample_factors.len()).fold(mel_frames, |len, i| {
            (len - 1) * self.upsample_factors[i] - 2 * self.upsample_padding(i) + self.upsample_kernel_sizes[i]
        })
    }
}

/// `scheduler/scheduler_config.json` — `FlowMatchEulerDiscreteScheduler`.
#[derive(Debug, Clone, PartialEq)]
pub struct Ltx2SchedulerConfig {
    pub num_train_timesteps: usize,
    pub shift: f64,
    pub use_dynamic_shifting: bool,
    pub base_shift: f64,
    pub max_shift: f64,
    pub base_image_seq_len: usize,
    pub max_image_seq_len: usize,
    pub shift_terminal: Option<f64>,
    /// `time_shift_type: "exponential"`.
    pub exponential_time_shift: bool,
    pub stochastic_sampling: bool,
}

impl Ltx2SchedulerConfig {
    /// `Lightricks/LTX-2` as published: the dev model's schedule.
    pub fn ltx2_19b() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 1.0,
            use_dynamic_shifting: true,
            base_shift: 0.95,
            max_shift: 2.05,
            base_image_seq_len: 1024,
            max_image_seq_len: 4096,
            shift_terminal: Some(0.1),
            exponential_time_shift: true,
            stochastic_sampling: false,
        }
    }

    /// The distilled recipe: the explicit sigma list is used as given, so both
    /// the dynamic shift and the terminal stretch are off
    /// (`rootonchair/LTX-2-19b-distilled` `scheduler_config.json`; model card).
    pub fn ltx2_19b_distilled() -> Self {
        Self { use_dynamic_shifting: false, shift_terminal: None, ..Self::ltx2_19b() }
    }
}

/// `text_encoder/config.json` → `text_config` (Gemma-3-12B language model).
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma3TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub query_pre_attn_scalar: f64,
    pub rms_norm_eps: f64,
    /// Global layers: θ = 1e6 with linear scaling (positions ÷ 8).
    pub rope_theta: f64,
    pub rope_scaling_factor: f64,
    /// Sliding layers: θ = 1e4, unscaled.
    pub rope_local_base_freq: f64,
    pub sliding_window: usize,
    /// Every 6th layer (index 5, 11, …, 47) is global, the rest slide.
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,
    pub attention_bias: bool,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub pad_token_id: u32,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
}

impl Gemma3TextConfig {
    pub fn ltx2_19b() -> Self {
        Self {
            vocab_size: 262_208,
            hidden_size: 3840,
            intermediate_size: 15360,
            num_hidden_layers: 48,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 256,
            query_pre_attn_scalar: 256.0,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_scaling_factor: 8.0,
            rope_local_base_freq: 10_000.0,
            sliding_window: 1024,
            sliding_window_pattern: 6,
            max_position_embeddings: 131_072,
            attention_bias: false,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 2,
        }
    }

    /// `layer_types[i] == "full_attention"`.
    pub fn is_global_layer(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.sliding_window_pattern)
    }

    /// Softmax scale: `query_pre_attn_scalar ** -0.5` = 1/16.
    pub fn attention_scale(&self) -> f64 {
        self.query_pre_attn_scalar.powf(-0.5)
    }

    /// q is 16 × 256 = 4096 wide — wider than the 3840 residual stream.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Embedding multiplier. transformers casts `sqrt(hidden)` to the weight
    /// dtype first (`modeling_gemma3.py:107`), and bf16 cannot hold 61.9677:
    /// a bf16 run multiplies by exactly 62.0.
    pub fn embed_scale(&self, bf16: bool) -> f32 {
        let s = (self.hidden_size as f32).sqrt();
        if bf16 {
            half_bf16_round(s)
        } else {
            s
        }
    }

    /// Hidden states the pipeline stacks per token: embeddings + every layer.
    pub fn num_hidden_states(&self) -> usize {
        self.num_hidden_layers + 1
    }
}

/// `text_encoder/config.json` → `text_config` (Gemma-4-12B unified text tower).
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub num_global_key_value_heads: usize,
    pub query_pre_attn_scalar: f64,
    pub rms_norm_eps: f64,
    pub rope_theta_full: f64,
    pub rope_theta_sliding: f64,
    pub rope_scaling_factor: f64,
    pub partial_rotary_factor_full: f64,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,
    pub attention_bias: bool,
    pub attention_k_eq_v: bool,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub pad_token_id: u32,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
}

impl Gemma4TextConfig {
    pub fn ltx2_5_22b() -> Self {
        Self {
            vocab_size: 262_144,
            hidden_size: 3840,
            intermediate_size: 15360,
            num_hidden_layers: 48,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 256,
            global_head_dim: 512,
            num_global_key_value_heads: 1,
            query_pre_attn_scalar: 256.0,
            rms_norm_eps: 1e-6,
            rope_theta_full: 1_000_000.0,
            rope_theta_sliding: 10_000.0,
            rope_scaling_factor: 8.0,
            partial_rotary_factor_full: 0.25,
            sliding_window: 1024,
            sliding_window_pattern: 6,
            max_position_embeddings: 262_144,
            attention_bias: false,
            attention_k_eq_v: true,
            attn_logit_softcapping: None,
            final_logit_softcapping: Some(30.0),
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 2,
        }
    }

    pub fn is_global_layer(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.sliding_window_pattern)
    }

    pub fn attention_scale(&self) -> f64 {
        self.query_pre_attn_scalar.powf(-0.5)
    }

    pub fn q_dim(&self, layer: usize) -> usize {
        if self.is_global_layer(layer) {
            self.num_attention_heads * self.global_head_dim
        } else {
            self.num_attention_heads * self.head_dim
        }
    }

    pub fn kv_dim(&self, layer: usize) -> usize {
        if self.is_global_layer(layer) {
            self.num_global_key_value_heads * self.global_head_dim
        } else {
            self.num_key_value_heads * self.head_dim
        }
    }

    pub fn embed_scale(&self, bf16: bool) -> f32 {
        let s = (self.hidden_size as f32).sqrt();
        if bf16 {
            half_bf16_round(s)
        } else {
            s
        }
    }

    pub fn num_hidden_states(&self) -> usize {
        self.num_hidden_layers + 1
    }
}

/// Round an f32 to the nearest bfloat16 (8 exponent bits, 7 mantissa bits),
/// ties to even. Only for normal, in-range values — all this module needs.
fn half_bf16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    let drop = 23 - 7;
    let half = 1u32 << (drop - 1);
    let mask = (1u32 << drop) - 1;
    let rem = bits & mask;
    let mut kept = bits & !mask;
    if rem > half || (rem == half && (kept >> drop) & 1 == 1) {
        kept += 1 << drop;
    }
    f32::from_bits(kept)
}

/// Pipeline-level defaults shared by diffusers `LTX2Pipeline.__call__` and the
/// Lightricks `PipelineParams` for LTX-2.0.
#[derive(Debug, Clone, PartialEq)]
pub struct Ltx2PipelineDefaults {
    pub height: usize,
    pub width: usize,
    pub num_frames: usize,
    pub frame_rate: f64,
    pub max_sequence_length: usize,
    /// Left padding: Gemma chat convention, `pipeline_ltx2.py:325-329`.
    pub pad_left: bool,
}

impl Ltx2PipelineDefaults {
    pub fn ltx2_19b() -> Self {
        Self { height: 512, width: 768, num_frames: 121, frame_rate: 24.0, max_sequence_length: 1024, pad_left: true }
    }
}

/// Every component config for one checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct Ltx2Config {
    pub version: Ltx2ModelVersion,
    pub transformer: Ltx2TransformerConfig,
    pub vae: Ltx2VideoVaeConfig,
    pub audio_vae: Ltx2AudioVaeConfig,
    pub connectors: Ltx2ConnectorsConfig,
    pub vocoder: Ltx2VocoderConfig,
    pub scheduler: Ltx2SchedulerConfig,
    pub text_encoder: Gemma3TextConfig,
    pub gemma4: Option<Gemma4TextConfig>,
    pub defaults: Ltx2PipelineDefaults,
}

/// `Lightricks/LTX-2` as published (the dev transformer and its scheduler).
pub fn ltx2_19b() -> Ltx2Config {
    Ltx2Config {
        version: Ltx2ModelVersion::V20,
        transformer: Ltx2TransformerConfig::ltx2_19b(),
        vae: Ltx2VideoVaeConfig::ltx2_19b(),
        audio_vae: Ltx2AudioVaeConfig::ltx2_19b(),
        connectors: Ltx2ConnectorsConfig::ltx2_19b(),
        vocoder: Ltx2VocoderConfig::ltx2_19b(),
        scheduler: Ltx2SchedulerConfig::ltx2_19b(),
        text_encoder: Gemma3TextConfig::ltx2_19b(),
        gemma4: None,
        defaults: Ltx2PipelineDefaults::ltx2_19b(),
    }
}

/// The distilled checkpoint: identical architecture, different transformer and
/// connector weights, and a scheduler that leaves the sigma list alone.
pub fn ltx2_19b_distilled() -> Ltx2Config {
    Ltx2Config { scheduler: Ltx2SchedulerConfig::ltx2_19b_distilled(), ..ltx2_19b() }
}

/// LTX-2.5 distilled stage-1 T2AV (Gemma 4 + 2.5 DiT/VAE/vocoder/connectors).
pub fn ltx2_5_22b_distilled() -> Ltx2Config {
    Ltx2Config {
        version: Ltx2ModelVersion::V25,
        transformer: Ltx2TransformerConfig::ltx2_5_22b(),
        vae: Ltx2VideoVaeConfig::ltx2_5_22b(),
        connectors: Ltx2ConnectorsConfig::ltx2_5_22b(),
        vocoder: Ltx2VocoderConfig::ltx2_5_22b_bwe(),
        scheduler: Ltx2SchedulerConfig::ltx2_19b_distilled(),
        gemma4: Some(Gemma4TextConfig::ltx2_5_22b()),
        ..ltx2_19b()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformer_widths() {
        let c = Ltx2TransformerConfig::ltx2_19b();
        assert_eq!(c.inner_dim(), 4096);
        assert_eq!(c.audio_inner_dim(), 2048);
        assert_eq!(c.av_cross_inner_dim(), 2048);
        assert_eq!(c.ff_inner_dim(), 16384);
        assert_eq!(c.audio_ff_inner_dim(), 8192);
        // The config's cross_attention_dim fields equal the stream widths: the
        // caption projections lift 3840 to them before any block runs.
        assert_eq!(c.cross_attention_dim, c.inner_dim());
        assert_eq!(c.audio_cross_attention_dim, c.audio_inner_dim());
    }

    #[test]
    fn default_clip_sequence_lengths() {
        let c = Ltx2TransformerConfig::ltx2_19b();
        // 768x512, 121 frames @ 24 fps: the diffusers default and distilled stage 1.
        assert_eq!(c.latent_grid(121, 512, 768), [16, 16, 24]);
        assert_eq!(c.video_tokens(121, 512, 768), 6144);
        // Stage 2 / full resolution.
        assert_eq!(c.latent_grid(121, 1024, 1536), [16, 32, 48]);
        assert_eq!(c.video_tokens(121, 1024, 1536), 24576);
        assert_eq!(c.audio_latents_per_second(), 25.0);
        // 121 / 24 * 25 = 126.04…
        assert_eq!(c.audio_tokens(121, 24.0), 126);
        assert_eq!(c.audio_tokens(241, 24.0), 251);
        assert_eq!(c.audio_tokens(121, 25.0), 121);
    }

    #[test]
    fn python_round_is_half_even() {
        assert_eq!(round_half_even(0.5), 0.0);
        assert_eq!(round_half_even(1.5), 2.0);
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(12.5), 12.0);
        assert_eq!(round_half_even(13.5), 14.0);
        assert_eq!(round_half_even(126.04), 126.0);
        assert_eq!(round_half_even(126.6), 127.0);
    }

    #[test]
    fn rope_table_layout() {
        // Video self-attention: 4096 wide over (t, h, w) → 682 freqs per axis,
        // 2046 slots, 2 identity pads, 64 per head.
        assert_eq!(Ltx2TransformerConfig::rope_freqs_per_axis(4096, 3), 682);
        assert_eq!(Ltx2TransformerConfig::rope_pad(4096, 3), 2);
        // Audio self-attention and both a↔v cross tables: 2048 wide, time only.
        assert_eq!(Ltx2TransformerConfig::rope_freqs_per_axis(2048, 1), 1024);
        assert_eq!(Ltx2TransformerConfig::rope_pad(2048, 1), 0);
        // Connectors: 3840 wide, 1-D.
        assert_eq!(Ltx2TransformerConfig::rope_freqs_per_axis(3840, 1), 1920);
        assert_eq!(Ltx2TransformerConfig::rope_pad(3840, 1), 0);
    }

    #[test]
    fn video_vae_decoder_shape() {
        let c = Ltx2VideoVaeConfig::ltx2_19b();
        // Matches the conv shapes in vae/diffusion_pytorch_model.safetensors:
        // upsamplers 1024→4096, 512→2048, 256→1024.
        assert_eq!(
            c.decoder_stages(),
            vec![
                Ltx2VaeDecoderStage { channels: 1024, resnet_layers: 5, upsampler: None },
                Ltx2VaeDecoderStage {
                    channels: 512,
                    resnet_layers: 5,
                    upsampler: Some(Ltx2VaeDecoderUpsampler {
                        in_channels: 1024,
                        conv_out_channels: 4096,
                        stride: (2, 2, 2),
                        residual: true,
                        drop_first_frame: true,
                    }),
                },
                Ltx2VaeDecoderStage {
                    channels: 256,
                    resnet_layers: 5,
                    upsampler: Some(Ltx2VaeDecoderUpsampler {
                        in_channels: 512,
                        conv_out_channels: 2048,
                        stride: (2, 2, 2),
                        residual: true,
                        drop_first_frame: true,
                    }),
                },
                Ltx2VaeDecoderStage {
                    channels: 128,
                    resnet_layers: 5,
                    upsampler: Some(Ltx2VaeDecoderUpsampler {
                        in_channels: 256,
                        conv_out_channels: 1024,
                        stride: (2, 2, 2),
                        residual: true,
                        drop_first_frame: true,
                    }),
                },
            ]
        );
        assert_eq!(c.decoded_frames(16), 121);
        assert_eq!(c.decoded_frames(1), 1);
        assert_eq!(c.out_channels * c.patch_size * c.patch_size, 48);
    }

    #[test]
    fn audio_shapes() {
        let a = Ltx2AudioVaeConfig::ltx2_19b();
        assert_eq!(a.latent_mel_bins(), 16);
        assert_eq!(a.token_channels(), 128);
        assert_eq!(a.token_channels(), Ltx2TransformerConfig::ltx2_19b().audio_in_channels);
        assert_eq!(a.mel_frames(126), 501);
        assert_eq!(a.mel_frames(1), 1);

        let v = Ltx2VocoderConfig::ltx2_19b();
        assert_eq!(v.in_channels, a.output_channels * a.mel_bins);
        assert_eq!(v.total_upsample_factor(), 240);
        assert_eq!((0..5).map(|i| v.upsample_padding(i)).collect::<Vec<_>>(), vec![5, 5, 3, 1, 1]);
        assert!(!v.with_bwe);
        assert_eq!((0..5).map(|i| v.stage_channels(i)).collect::<Vec<_>>(), vec![512, 256, 128, 64, 32]);
        assert_eq!(v.waveform_samples(501), 501 * 240);
        // One mel frame is 10 ms at either rate: 160 samples @ 16 kHz in, 240 @ 24 kHz out.
        assert_eq!(v.total_upsample_factor() * a.sample_rate, a.mel_hop_length * v.output_sampling_rate);
    }

    #[test]
    fn connectors_widths() {
        let c = Ltx2ConnectorsConfig::ltx2_19b();
        assert_eq!(c.text_proj_in_features(), 188_160);
        assert_eq!(c.inner_dim(), 3840);
        assert_eq!(c.text_proj_in_factor, Gemma3TextConfig::ltx2_19b().num_hidden_states());
        assert_eq!(1024 % c.video_connector_num_learnable_registers, 0);
    }

    #[test]
    fn gemma_layout() {
        let g = Gemma3TextConfig::ltx2_19b();
        assert_eq!(g.q_dim(), 4096);
        assert_eq!(g.kv_dim(), 2048);
        assert_eq!(g.attention_scale(), 0.0625);
        let globals: Vec<usize> = (0..g.num_hidden_layers).filter(|&i| g.is_global_layer(i)).collect();
        assert_eq!(globals, vec![5, 11, 17, 23, 29, 35, 41, 47]);
        assert!((g.embed_scale(false) - 61.967_735).abs() < 1e-5);
        assert_eq!(g.embed_scale(true), 62.0);
    }

    #[test]
    fn bf16_rounding() {
        assert_eq!(half_bf16_round(1.0), 1.0);
        assert_eq!(half_bf16_round(62.0), 62.0);
        // Step in [32, 64) is 0.25.
        assert_eq!(half_bf16_round(61.9), 62.0);
        assert_eq!(half_bf16_round(61.8), 61.75);
        // Tie at 61.875 goes to the even mantissa: 62.0 (…11000) over 61.75 (…10111).
        assert_eq!(half_bf16_round(61.875), 62.0);
    }

    #[test]
    fn distilled_differs_only_in_scheduler() {
        let (dev, dist) = (ltx2_19b(), ltx2_19b_distilled());
        assert_eq!(dev.transformer, dist.transformer);
        assert_eq!(dev.connectors, dist.connectors);
        assert!(dev.scheduler.use_dynamic_shifting && !dist.scheduler.use_dynamic_shifting);
        assert_eq!(dev.scheduler.shift_terminal, Some(0.1));
        assert_eq!(dist.scheduler.shift_terminal, None);
    }

    #[test]
    fn ltx25_transformer_flags() {
        let c = Ltx2TransformerConfig::ltx2_5_22b();
        assert!(c.gated_attn && c.audio_gated_attn && c.cross_attn_mod && c.audio_cross_attn_mod);
        assert!(c.perturbed_attn && c.use_keyframes_abs_pos_embedding);
        assert!(!c.ff_bias && c.audio_ff_bias);
        assert!(!c.use_prompt_embeddings && c.use_prompt_adaln_single);
        assert_eq!(c.inner_dim(), 4096);
        assert_eq!(c.num_layers, 48);
    }

    #[test]
    fn ltx25_connectors() {
        let c = Ltx2ConnectorsConfig::ltx2_5_22b();
        assert!(c.per_modality_projections && c.proj_bias);
        assert!(c.video_gated_attn && c.audio_gated_attn);
        assert_eq!(c.video_hidden_dim, 4096);
        assert_eq!(c.audio_hidden_dim, 2048);
        assert_eq!(c.video_connector_num_layers, 8);
        assert_eq!(c.text_proj_in_factor, Gemma4TextConfig::ltx2_5_22b().num_hidden_states());
    }

    #[test]
    fn ltx25_video_vae_decoder_stages() {
        let stages = Ltx2VideoVaeConfig::ltx2_5_22b().decoder_stages();
        assert_eq!(stages.len(), 5);
        assert_eq!(stages[0].channels, 1024);
        assert_eq!(stages[1].upsampler.as_ref().map(|u| u.conv_out_channels), Some(4096));
        assert_eq!(stages[2].upsampler.as_ref().map(|u| u.stride), Some((2, 2, 2)));
        assert_eq!(stages[3].upsampler.as_ref().map(|u| u.stride), Some((2, 1, 1)));
        assert!(stages.iter().all(|s| s.upsampler.as_ref().map(|u| u.residual) != Some(true)));
        assert_eq!(stages[4].upsampler.as_ref().map(|u| u.stride), Some((1, 2, 2)));
    }

    #[test]
    fn ltx25_vocoder_geometry() {
        let v = Ltx2VocoderConfig::ltx2_5_22b_bwe();
        assert_eq!(v.upsample_factors.len(), 6);
        assert_eq!(v.total_upsample_factor(), 160);
        assert_eq!(v.stage_channels(0), 768);
        assert_eq!(v.stage_channels(5), 24);
    }

    #[test]
    fn ltx25_bundle_version() {
        let cfg = ltx2_5_22b_distilled();
        assert_eq!(cfg.version, Ltx2ModelVersion::V25);
        assert!(cfg.gemma4.is_some());
        assert!(!cfg.scheduler.use_dynamic_shifting);
        assert!(cfg.vocoder.with_bwe);
        assert_eq!(cfg.vocoder.output_sampling_rate, 48000);
    }
}
