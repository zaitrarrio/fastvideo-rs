//! Configuration of the four FastH3 components, mirroring the `config.json`
//! files of `FastVideo/FastVideo-FastH3-8-Step-V2`, plus the request geometry
//! that follows from them (canvas, frame alignment, latent and audio counts).
//!
//! `fastvideo-models` does not depend on serde, so these are hand-written
//! constructors holding the exact checkpoint values rather than deserializers.
//! The values were read from the Hub on 2026-09-19; docs/ports/h3.md lists the
//! source file for each.
//!
//! Python references (FastVideo is what the checkpoint was trained with):
//! `fastvideo/pipelines/basic/minimax_h3/packing.py` for the geometry helpers,
//! diffusers `autoencoder_kl_minimax_h3.py` for the VAE chunk arithmetic.

/// Row modality tags of the packed sequence (`packing.py:17-19`). They index
/// the AdaLN table as `timestep_index * 3 + tag`.
pub const TAG_VIDEO: u8 = 0;
pub const TAG_TEXT: u8 = 1;
pub const TAG_AUDIO: u8 = 2;
/// `MINIMAX_H3_MODALITY_NUM`.
pub const MODALITY_NUM: usize = 3;

/// `transformer/config.json` (`MiniMaxH3Transformer3DModel`).
#[derive(Debug, Clone, PartialEq)]
pub struct H3TransformerConfig {
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    /// Residual width. Smaller than `heads * head_dim` (5376 < 7168).
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_refiner_layers: usize,
    /// SwiGLU inner width; `fc_in` is `hidden -> 2 * ffn_dim`, value half first.
    pub ffn_dim: usize,
    pub in_channels: usize,
    pub audio_in_channels: usize,
    pub patch_size: [usize; 3],
    pub text_dim: usize,
    pub freq_dim: usize,
    pub time_embed_hidden_dim: usize,
    pub time_embed_dim: usize,
    /// Rotary frequencies per axis; `2 * 3 * rope_freq_dim` channels rotate.
    pub rope_freq_dim: usize,
    pub rope_theta: f64,
    pub norm_eps: f64,
    pub qk_norm_eps: f64,
    pub final_norm_eps: f64,
}

impl H3TransformerConfig {
    pub fn fasth3_8step() -> Self {
        Self {
            num_attention_heads: 56,
            attention_head_dim: 128,
            hidden_size: 5376,
            num_layers: 50,
            num_refiner_layers: 2,
            ffn_dim: 14336,
            in_channels: 24,
            audio_in_channels: 32,
            patch_size: [1, 2, 2],
            text_dim: 5120,
            freq_dim: 256,
            time_embed_hidden_dim: 5376,
            time_embed_dim: 2688,
            rope_freq_dim: 16,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }

    /// `heads * head_dim`, the width of Q/K/V and `to_gate_compress`.
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Input width of `proj_in`: `in_channels * prod(patch_size)`.
    pub fn video_patch_dim(&self) -> usize {
        self.in_channels * self.patch_size.iter().product::<usize>()
    }

    /// Leading channels of every head that RoPE rotates (96 of 128).
    pub fn rotary_dim(&self) -> usize {
        2 * 3 * self.rope_freq_dim
    }

    /// Output width of one block's `adaln_proj.linear`: 6 parameters for each
    /// of the 3 modalities.
    pub fn adaln_out_dim(&self) -> usize {
        6 * self.hidden_size * MODALITY_NUM
    }

    /// `inv_freq[k] = theta^(-k / rope_freq_dim)`, shared by the three axes
    /// (`minimax_h3.py:79`: `theta ** (arange(0, 2F, 2) / (2F))`), in float32
    /// as the reference buffer is.
    pub fn rope_inv_freq(&self) -> Vec<f32> {
        let f = self.rope_freq_dim;
        (0..f)
            .map(|k| {
                let exponent = (2 * k) as f32 / (2 * f) as f32;
                1.0f32 / (self.rope_theta as f32).powf(exponent)
            })
            .collect()
    }
}

/// `vae/config.json` (`AutoencoderKLMiniMaxH3`, "f16t4d24").
#[derive(Debug, Clone, PartialEq)]
pub struct H3VideoVaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    /// Encoder only; T2AV never runs the encoder.
    pub block_out_channels: [usize; 6],
    pub layers_per_block: usize,
    pub spatial_downsample_factors: [usize; 6],
    pub temporal_downsample_factors: [usize; 6],
    pub norm_num_groups: usize,
    pub norm_eps: f64,
    pub decoder_num_layers: usize,
    pub decoder_num_attention_heads: usize,
    pub decoder_attention_head_dim: usize,
    pub decoder_num_register_tokens: usize,
    pub decoder_ffn_mult: usize,
    pub decoder_rope_theta: f64,
    pub decoder_rope_dim_ratio: f64,
    pub decoder_norm_eps: f64,
    /// Pixel frames per encoder chunk.
    pub clip_length: usize,
    /// Trailing latent frames dropped per encode.
    pub token_drop: usize,
    pub latents_mean: [f64; 24],
    pub latents_std: [f64; 24],
    /// Spatial tiling is on by default in both upstreams and changes the
    /// output (`autoencoder_kl_minimax_h3.py:608-615`). Pixels.
    pub tile_sample_min_size: usize,
    pub tile_sample_min_overlap: usize,
}

/// ImageNet statistics the VAE's pixel space is normalized with
/// (`minimax_h3_video.py:603-612`): `rgb01 = sample * std + mean`, clamped.
pub const H3_PIXEL_MEAN: [f64; 3] = [0.485, 0.456, 0.406];
pub const H3_PIXEL_STD: [f64; 3] = [0.229, 0.224, 0.225];

impl H3VideoVaeConfig {
    pub fn fasth3_8step() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 24,
            block_out_channels: [128, 256, 256, 512, 512, 1024],
            layers_per_block: 2,
            spatial_downsample_factors: [2, 2, 2, 2, 1, 1],
            temporal_downsample_factors: [1, 2, 2, 1, 1, 1],
            norm_num_groups: 32,
            norm_eps: 1e-6,
            decoder_num_layers: 36,
            decoder_num_attention_heads: 32,
            decoder_attention_head_dim: 64,
            decoder_num_register_tokens: 4,
            decoder_ffn_mult: 4,
            decoder_rope_theta: 100.0,
            decoder_rope_dim_ratio: 0.75,
            decoder_norm_eps: 1e-5,
            clip_length: 17,
            token_drop: 3,
            #[rustfmt::skip]
            latents_mean: [
                0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
                -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
                -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
                -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
                -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
                -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
            ],
            #[rustfmt::skip]
            latents_std: [
                1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
                1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
                0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
                0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
                2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
                3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
            ],
            tile_sample_min_size: 256,
            tile_sample_min_overlap: 64,
        }
    }

    /// 16: also the ViT decoder's spatial patch size.
    pub fn spatial_compression_ratio(&self) -> usize {
        self.spatial_downsample_factors.iter().product()
    }

    /// 4: also the ViT decoder's temporal patch size.
    pub fn temporal_compression_ratio(&self) -> usize {
        self.temporal_downsample_factors.iter().product()
    }

    /// ViT decoder width, `heads * head_dim` = 2048.
    pub fn decoder_dim(&self) -> usize {
        self.decoder_num_attention_heads * self.decoder_attention_head_dim
    }

    /// Rotated channels per decoder head: `int(head_dim * ratio)` = 48.
    pub fn decoder_rotary_dim(&self) -> usize {
        (self.decoder_attention_head_dim as f64 * self.decoder_rope_dim_ratio) as usize
    }

    /// Output width of the decoder's `proj_out`: `3 * 4 * 16 * 16` = 3072.
    pub fn decoder_patch_dim(&self) -> usize {
        let s = self.spatial_compression_ratio();
        self.out_channels * self.temporal_compression_ratio() * s * s
    }

    /// `frame_pre_padding = (-clip_length) % t_ratio` = 3
    /// (`autoencoder_kl_minimax_h3.py:596`).
    pub fn frame_pre_padding(&self) -> usize {
        let r = self.temporal_compression_ratio();
        (r - self.clip_length % r) % r
    }

    /// `tokens_chunk_size = ceil(clip_length / t_ratio)` = 5.
    pub fn tokens_chunk_size(&self) -> usize {
        self.clip_length.div_ceil(self.temporal_compression_ratio())
    }

    /// `token_overlap = (-token_drop) % tokens_chunk_size` = 2, so one decoder
    /// call sees `tokens_chunk_size + token_overlap` = 7 latent frames.
    pub fn token_overlap(&self) -> usize {
        let c = self.tokens_chunk_size();
        (c - self.token_drop % c) % c
    }

    /// `frame_overlap = max(token_overlap * t_ratio - frame_pre_padding, 0)` = 5
    /// pixel frames cross-faded between consecutive chunks.
    pub fn frame_overlap(&self) -> usize {
        (self.token_overlap() * self.temporal_compression_ratio())
            .saturating_sub(self.frame_pre_padding())
    }

    /// `(pad_tokens, num_chunks, decoded_frames)` for a latent clip, as
    /// FastVideo `_temporal_decode_plan` (`minimax_h3_video.py:890-917`).
    /// For the `5n + 2` latents a request produces this is `(0, n, 17n + 5)`.
    pub fn temporal_decode_plan(&self, latent_frames: usize) -> (usize, usize, usize) {
        let chunk = self.tokens_chunk_size();
        let ratio = self.temporal_compression_ratio();
        let has_drop = usize::from(self.token_drop > 0);
        let num_tokens = latent_frames + self.token_drop;
        let mut pad_tokens = (chunk - num_tokens % chunk) % chunk;
        let mut num_chunks = ((num_tokens + pad_tokens) / chunk).saturating_sub(has_drop);
        if num_chunks < 1 {
            pad_tokens += chunk;
            num_chunks = 1;
        }
        let mut frames = num_chunks * (chunk * ratio - self.frame_pre_padding());
        if self.token_drop > 0 {
            frames += self.frame_overlap();
        }
        let intra_tail = self.clip_length % ratio;
        let pad_frames: usize = (0..pad_tokens)
            .map(|offset| {
                // The last latent of a chunk covers `clip_length % ratio` frames, the rest `ratio`.
                let position_in_chunk = (latent_frames + offset) % chunk;
                if intra_tail != 0 && position_in_chunk == 0 {
                    intra_tail
                } else {
                    ratio
                }
            })
            .sum();
        (pad_tokens, num_chunks, frames.saturating_sub(pad_frames))
    }

    /// `_split_tiles` (`autoencoder_kl_minimax_h3.py:645-666`): tile starts and
    /// the overlap between each consecutive pair, in pixels. Every tile is
    /// `tile_sample_min_size` wide unless the axis fits in one tile.
    pub fn split_tiles(&self, length: usize) -> (Vec<usize>, Vec<usize>) {
        let tile = self.tile_sample_min_size;
        let min_overlap = self.tile_sample_min_overlap;
        if tile >= length {
            return (vec![0], Vec::new());
        }
        let mut num_tiles = length.div_ceil(tile);
        while tile * num_tiles < min_overlap * (num_tiles - 1) + length {
            num_tiles += 1;
        }
        let mut overlaps = vec![min_overlap; num_tiles - 1];
        let remaining = tile * num_tiles - min_overlap * (num_tiles - 1) - length;
        let step = self.spatial_compression_ratio();
        for i in 0..remaining / step {
            overlaps[i % (num_tiles - 1)] += step;
        }
        let mut starts = vec![0usize];
        for overlap in &overlaps {
            starts.push(starts[starts.len() - 1] + tile - overlap);
        }
        (starts, overlaps)
    }
}

/// `audio_vae/config.json` (`AutoencoderKLMiniMaxH3Audio`): DAC encoder,
/// BigVGAN decoder, mono; stereo is two batch items through the same weights.
#[derive(Debug, Clone, PartialEq)]
pub struct H3AudioVaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: [usize; 5],
    pub latent_dim: usize,
    pub latent_channels: usize,
    pub num_attention_heads: usize,
    pub decoder_dim: usize,
    pub decoder_rates: [usize; 7],
    pub decoder_kernel_sizes: [usize; 7],
    pub resblock_kernel_sizes: [usize; 3],
    pub resblock_dilation_sizes: [[usize; 3]; 3],
    pub sampling_rate: usize,
    pub latents_mean: [f64; 32],
    pub latents_std: [f64; 32],
}

impl H3AudioVaeConfig {
    pub fn fasth3_8step() -> Self {
        Self {
            encoder_dim: 64,
            encoder_rates: [2, 4, 4, 5, 5],
            latent_dim: 2048,
            latent_channels: 32,
            num_attention_heads: 8,
            decoder_dim: 1024,
            decoder_rates: [5, 5, 2, 2, 2, 2, 2],
            decoder_kernel_sizes: [9, 9, 4, 4, 4, 4, 4],
            resblock_kernel_sizes: [3, 7, 11],
            resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            sampling_rate: 32000,
            #[rustfmt::skip]
            latents_mean: [
                -0.020211687488382354, 0.3876466479950502, -0.04398279799186767, -0.28591514936373,
                0.08179686214561671, -0.35782641352446604, 0.040623809960919084, -0.01552534501956604,
                -0.223362481667332, 0.1821006842509091, 0.2941778783780663, -0.07901167601970885,
                -0.056815072777201, -0.3699028221860095, -0.31616315591624855, 0.5905951377425391,
                -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163,
                -0.3394662328788498, -0.30685677538541667, -0.24504598907458763, -0.034698524462007344,
                0.02868032184767538, -0.21217779266454084, -0.1678263169941987, 0.3221287889040614,
                -0.1223055851554907, 0.4356604928128464, -0.0502599202236253, 0.3979258376211797,
            ],
            #[rustfmt::skip]
            latents_std: [
                1.6895524230479284, 2.76263727217653, 1.7945344281264435, 1.6801681847309828,
                1.6390226546605453, 2.7788298348882177, 1.7659090095747236, 1.6199757612137327,
                2.6336525640336896, 1.8539356672817833, 2.5056497896915633, 1.811019237886178,
                1.9579657790720237, 1.6685498243529284, 1.4922469314453364, 3.298670198067373,
                1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093,
                1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912,
                1.8318420762504692, 1.5540637421583379, 1.9344930328968526, 1.599198216109855,
                1.718045989838149, 1.6307219190837705, 1.8661226051202384, 1.5613768203168363,
            ],
        }
    }

    /// Samples per latent frame: `prod(decoder_rates)` = 800, i.e. 40 Hz latents
    /// at 32 kHz. Decoding `n` latents yields exactly `800 n` samples.
    pub fn hop_length(&self) -> usize {
        self.decoder_rates.iter().product()
    }

    /// Latent frames per second per channel (40).
    pub fn latents_per_second(&self) -> usize {
        self.sampling_rate / self.hop_length()
    }

    /// `(in, out)` channels of upsampler `i`: `decoder_dim >> i` to
    /// `decoder_dim >> (i + 1)`.
    pub fn upsampler_channels(&self, i: usize) -> (usize, usize) {
        (self.decoder_dim >> i, self.decoder_dim >> (i + 1))
    }

    /// `ConvTranspose1d` padding of upsampler `i`: `(kernel - rate) / 2`.
    pub fn upsampler_padding(&self, i: usize) -> usize {
        (self.decoder_kernel_sizes[i] - self.decoder_rates[i]) / 2
    }

    /// Output length of upsampler `i`, `(L - 1) * stride - 2 * pad + kernel`.
    /// Equals `L * rate` for every stage of this checkpoint.
    pub fn upsampler_out_len(&self, i: usize, len: usize) -> usize {
        (len - 1) * self.decoder_rates[i] + self.decoder_kernel_sizes[i]
            - 2 * self.upsampler_padding(i)
    }
}

/// `text_encoder/config.json` `text_config` (Qwen3-VL-32B language model), plus
/// the H3 conditioning contract on top of it.
#[derive(Debug, Clone, PartialEq)]
pub struct H3TextEncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Interleaved mRoPE sections `(t, h, w)`; sums to `head_dim / 2`.
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
    pub attention_bias: bool,
    pub max_position_embeddings: usize,
    /// H3 conditions on HF `hidden_states[50]`: index 0 is the embedding, so
    /// this is the residual stream after decoder layer **49**, with no final
    /// norm (`minimax_h3_qwen3_vl.py:335`, diffusers `encoders.py:99`).
    pub output_hidden_state_index: usize,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
}

impl H3TextEncoderConfig {
    pub fn fasth3_8step() -> Self {
        Self {
            vocab_size: 151_936,
            hidden_size: 5120,
            intermediate_size: 25600,
            num_hidden_layers: 64,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            mrope_section: [24, 20, 20],
            mrope_interleaved: true,
            attention_bias: false,
            max_position_embeddings: 262_144,
            output_hidden_state_index: 50,
            vision_start_token_id: 151_652,
            vision_end_token_id: 151_653,
            image_token_id: 151_655,
            video_token_id: 151_656,
        }
    }

    /// Decoder layers that must run (and whose weights must be read): layers
    /// `0..layers_to_run()`, i.e. 0 through 49. Layers 50..63, the final norm,
    /// `lm_head` and the vision tower are never touched by a text-only prompt.
    pub fn layers_to_run(&self) -> usize {
        self.output_hidden_state_index
    }

    /// Query heads sharing one KV head (8).
    pub fn gqa_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// `inv_freq[i] = theta^(-2i / head_dim)`, `i < head_dim / 2`. For a
    /// text-only prompt all three mRoPE axes carry the same position, so the
    /// interleave is the identity and this is plain 1-D rotate-half RoPE.
    pub fn rope_inv_freq(&self) -> Vec<f32> {
        (0..self.head_dim / 2)
            .map(|i| {
                let exponent = (2 * i) as f32 / self.head_dim as f32;
                1.0f32 / (self.rope_theta as f32).powf(exponent)
            })
            .collect()
    }
}

/// `fastvideo_inference.json` plus the two `scheduler_config.json` shifts: the
/// recipe this distilled checkpoint was trained for.
#[derive(Debug, Clone, PartialEq)]
pub struct H3InferenceContract {
    /// DMD rungs; `rung / 1000` is the *unshifted* sigma of each forward.
    pub dmd_denoising_steps: Vec<u32>,
    /// Sigma-grid points, terminal zero included (`transformer_forwards + 1`).
    pub num_inference_steps: usize,
    pub transformer_forwards: usize,
    pub video_scheduler_shift: f64,
    pub audio_scheduler_shift: f64,
    pub guidance_scale: f64,
    pub vsa_sparsity: f64,
    /// Tokens per VSA tile; 64 is the `(4, 4, 4)` tile.
    pub vsa_tile_size: usize,
    /// Dense attention without `to_gate_compress` / VSA (Preview Dense).
    pub dense: bool,
}

impl H3InferenceContract {
    pub fn fasth3_8step() -> Self {
        Self {
            dmd_denoising_steps: vec![999, 874, 749, 624, 500, 375, 250, 125],
            num_inference_steps: 9,
            transformer_forwards: 8,
            video_scheduler_shift: 10.0,
            audio_scheduler_shift: 3.0,
            guidance_scale: 1.0,
            vsa_sparsity: 0.8,
            vsa_tile_size: 64,
            dense: false,
        }
    }

    /// FastH3 Preview VSA: 4 forwards, video shift 12, 90% sparsity.
    pub fn fasth3_4step_vsa() -> Self {
        Self {
            dmd_denoising_steps: vec![999, 749, 500, 250],
            num_inference_steps: 5,
            transformer_forwards: 4,
            video_scheduler_shift: 12.0,
            audio_scheduler_shift: 3.0,
            guidance_scale: 1.0,
            vsa_sparsity: 0.9,
            vsa_tile_size: 64,
            dense: false,
        }
    }

    /// FastH3 Preview Dense: same 4-rung ladder as VSA Preview, no VSA / gate.
    pub fn fasth3_4step_dense() -> Self {
        Self {
            dmd_denoising_steps: vec![999, 749, 500, 250],
            num_inference_steps: 5,
            transformer_forwards: 4,
            video_scheduler_shift: 12.0,
            audio_scheduler_shift: 3.0,
            guidance_scale: 1.0,
            vsa_sparsity: 0.0,
            vsa_tile_size: 64,
            dense: true,
        }
    }

    /// Named recipe: `8step` / `v2`, `4step-vsa` / `preview-vsa`, `4step-dense` / `preview-dense`.
    pub fn named(name: &str) -> Result<Self, String> {
        match name {
            "8step" | "v2" | "fasth3-8step" => Ok(Self::fasth3_8step()),
            "4step-vsa" | "preview-vsa" | "fasth3-4step-vsa" => Ok(Self::fasth3_4step_vsa()),
            "4step-dense" | "preview-dense" | "fasth3-4step-dense" => Ok(Self::fasth3_4step_dense()),
            other => Err(format!(
                "unknown H3 recipe '{other}' (8step|4step-vsa|4step-dense)"
            )),
        }
    }
}

// ---- request geometry (`packing.py`) ---------------------------------------

pub const H3_FPS: usize = 24;
pub const H3_SHORT_EDGE: usize = 768;
pub const H3_MAX_PIXELS: usize = 768 * 1344;
pub const H3_CANVAS_MULTIPLE: usize = 32;
pub const H3_MIN_DURATION_S: usize = 5;
pub const H3_MAX_DURATION_S: usize = 15;
pub const H3_FRAMES_PER_CHUNK: usize = 17;
pub const H3_LATENTS_PER_CHUNK: usize = 5;
pub const H3_AUDIO_LATENTS_PER_SECOND: usize = 40;
pub const H3_AUDIO_CHANNELS: usize = 2;

/// Python's `round()`: half to even. Only ever applied to non-negative values.
fn py_round(x: f64) -> usize {
    x.round_ties_even() as usize
}

/// `resolve_canvas_size` (`packing.py:94-110`): `(height, width)` in pixels for
/// an aspect ratio. 768 short edge, capped at `768 * 1344` pixels, snapped to
/// multiples of 32. `16:9` gives `(768, 1344)`.
pub fn resolve_canvas_size(
    aspect_width: f64,
    aspect_height: f64,
) -> Result<(usize, usize), String> {
    if aspect_width <= 0.0 || aspect_height <= 0.0 {
        return Err(format!(
            "aspect ratio must be positive, got {aspect_width}:{aspect_height}"
        ));
    }
    let ratio = aspect_width / aspect_height;
    if !(0.25..=4.0).contains(&ratio) {
        return Err(format!(
            "H3 supports 1:4 to 4:1, got {aspect_width}:{aspect_height}"
        ));
    }
    let short = H3_SHORT_EDGE as f64;
    let (mut width, mut height) = if ratio >= 1.0 {
        (short * ratio, short)
    } else {
        (short, short / ratio)
    };
    let area = width * height;
    if area > H3_MAX_PIXELS as f64 {
        let scale = (H3_MAX_PIXELS as f64 / area).powf(0.5);
        width *= scale;
        height *= scale;
    }
    let m = H3_CANVAS_MULTIPLE;
    let snap = |v: f64| (py_round(v / m as f64) * m).max(m);
    Ok((snap(height), snap(width)))
}

/// `align_num_frames`: round up to the next `17 n + 5`.
pub fn align_num_frames(num_frames: usize) -> usize {
    let mut n = num_frames.max(1);
    while n % H3_FRAMES_PER_CHUNK != H3_LATENTS_PER_CHUNK {
        n += 1;
    }
    n
}

/// `video_latent_num_frames`: `17 n + 5` pixel frames are `5 n + 2` latents.
pub fn video_latent_num_frames(aligned_frames: usize) -> Result<usize, String> {
    if aligned_frames % H3_FRAMES_PER_CHUNK != H3_LATENTS_PER_CHUNK {
        return Err(format!("num_frames must be 17 n + 5, got {aligned_frames}"));
    }
    Ok((aligned_frames - H3_LATENTS_PER_CHUNK) / H3_FRAMES_PER_CHUNK * H3_LATENTS_PER_CHUNK + 2)
}

/// `audio_latent_num_frames`: `round(frames / 24 * 40)` latents per channel.
pub fn audio_latent_num_frames(aligned_frames: usize) -> usize {
    py_round(aligned_frames as f64 / H3_FPS as f64 * H3_AUDIO_LATENTS_PER_SECOND as f64)
}

/// Everything about one T2AV request's shapes that does not depend on the
/// prompt. The packed sequence is `[text | audio | video]`; with no keyframes
/// there are no condition rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H3Geometry {
    pub height: usize,
    pub width: usize,
    /// Aligned pixel frames, `17 n + 5`.
    pub num_frames: usize,
    pub latent_frames: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    /// Audio latents per stereo channel.
    pub audio_latents: usize,
    /// Patch grid `(t, h, w)` of the video rows; also the VSA video grid.
    pub token_grid: (usize, usize, usize),
}

impl H3Geometry {
    /// `requested_frames` is aligned up and must land in the 5 to 15 second
    /// window (124 to 362 frames) both upstreams enforce.
    pub fn new(height: usize, width: usize, requested_frames: usize) -> Result<Self, String> {
        let vae = H3VideoVaeConfig::fasth3_8step();
        let dit = H3TransformerConfig::fasth3_8step();
        let m = H3_CANVAS_MULTIPLE;
        let (rem_h, rem_w) = (height % m, width % m);
        if height == 0 || width == 0 || rem_h != 0 || rem_w != 0 {
            return Err(format!(
                "height and width must be positive multiples of {m}, got {height}x{width}"
            ));
        }
        let num_frames = align_num_frames(requested_frames);
        let (lo, hi) = (
            align_num_frames(H3_MIN_DURATION_S * H3_FPS),
            align_num_frames(H3_MAX_DURATION_S * H3_FPS),
        );
        if num_frames < lo || num_frames > hi {
            return Err(format!(
                "aligned num_frames {num_frames} is outside {lo}..={hi}"
            ));
        }
        let ratio = vae.spatial_compression_ratio();
        let (latent_height, latent_width) = (height / ratio, width / ratio);
        let latent_frames = video_latent_num_frames(num_frames)?;
        let [pt, ph, pw] = dit.patch_size;
        if latent_frames % pt != 0 || latent_height % ph != 0 || latent_width % pw != 0 {
            return Err(format!(
                "latents {latent_frames}x{latent_height}x{latent_width} not divisible by the patch"
            ));
        }
        Ok(Self {
            height,
            width,
            num_frames,
            latent_frames,
            latent_height,
            latent_width,
            audio_latents: audio_latent_num_frames(num_frames),
            token_grid: (latent_frames / pt, latent_height / ph, latent_width / pw),
        })
    }

    /// The default 16:9 canvas for a duration in whole seconds.
    pub fn default_16x9(seconds: usize) -> Result<Self, String> {
        let (height, width) = resolve_canvas_size(16.0, 9.0)?;
        Self::new(height, width, seconds * H3_FPS)
    }

    pub fn video_rows(&self) -> usize {
        self.token_grid.0 * self.token_grid.1 * self.token_grid.2
    }

    /// Left channel rows then right channel rows.
    pub fn audio_rows(&self) -> usize {
        self.audio_latents * H3_AUDIO_CHANNELS
    }

    /// Packed length for a prompt of `text_tokens` tokens. No padding rows: the
    /// reference pads only inside VSA tiles, never the sequence.
    pub fn sequence_length(&self, text_tokens: usize) -> usize {
        text_tokens + self.audio_rows() + self.video_rows()
    }

    /// Output samples per channel: `800 * audio_latents`.
    pub fn audio_samples(&self) -> usize {
        self.audio_latents * H3AudioVaeConfig::fasth3_8step().hop_length()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transformer_numbers() {
        let c = H3TransformerConfig::fasth3_8step();
        assert_eq!(c.inner_dim(), 7168);
        assert_eq!(c.video_patch_dim(), 96);
        assert_eq!(c.rotary_dim(), 96);
        assert!(c.rotary_dim() <= c.attention_head_dim);
        // Checkpoint shape of transformer_blocks.N.adaln_proj.linear.weight is [96768, 2688].
        assert_eq!(c.adaln_out_dim(), 96_768);
        assert_eq!(c.time_embed_dim * 2, c.time_embed_hidden_dim);
        let inv = c.rope_inv_freq();
        assert_eq!(inv.len(), 16);
        assert_eq!(inv[0], 1.0);
        // theta^(-1/16) and theta^(-15/16) for theta = 1e4.
        assert!((inv[1] - 0.562_341_3).abs() < 1e-6);
        assert!((inv[15] - 1.778_279_4e-4).abs() < 1e-9);
    }

    #[test]
    fn video_vae_numbers() {
        let c = H3VideoVaeConfig::fasth3_8step();
        assert_eq!(c.spatial_compression_ratio(), 16);
        assert_eq!(c.temporal_compression_ratio(), 4);
        assert_eq!(c.decoder_dim(), 2048);
        assert_eq!(c.decoder_rotary_dim(), 48);
        assert_eq!(c.decoder_patch_dim(), 3072);
        assert_eq!(c.frame_pre_padding(), 3);
        assert_eq!(c.tokens_chunk_size(), 5);
        assert_eq!(c.token_overlap(), 2);
        assert_eq!(c.frame_overlap(), 5);
        assert_eq!(c.latents_mean.len(), c.latent_channels);
        assert!(c.latents_std.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn temporal_decode_plan_matches_reference() {
        // Values from running the Python `_temporal_decode_plan` arithmetic.
        let c = H3VideoVaeConfig::fasth3_8step();
        assert_eq!(c.temporal_decode_plan(37), (0, 7, 124));
        assert_eq!(c.temporal_decode_plan(72), (0, 14, 243));
        assert_eq!(c.temporal_decode_plan(107), (0, 21, 362));
        assert_eq!(c.temporal_decode_plan(7), (0, 1, 22));
        // A lone keyframe latent and the 2-latent minimum both pad.
        assert_eq!(c.temporal_decode_plan(1), (6, 1, 1));
        assert_eq!(c.temporal_decode_plan(2), (5, 1, 5));
        for n in 1..=21 {
            assert_eq!(c.temporal_decode_plan(5 * n + 2), (0, n, 17 * n + 5));
        }
    }

    #[test]
    fn spatial_tiles_at_768p() {
        let c = H3VideoVaeConfig::fasth3_8step();
        let (ys, yo) = c.split_tiles(768);
        assert_eq!(ys, vec![0, 160, 336, 512]);
        assert_eq!(yo, vec![96, 80, 80]);
        let (xs, xo) = c.split_tiles(1344);
        assert_eq!(xs, vec![0, 176, 352, 528, 704, 896, 1088]);
        assert_eq!(xo, vec![80, 80, 80, 80, 64, 64]);
        // The last tile ends exactly at the canvas edge; starts stay latent-aligned.
        assert_eq!(ys[ys.len() - 1] + 256, 768);
        assert_eq!(xs[xs.len() - 1] + 256, 1344);
        assert!(ys.iter().chain(&xs).all(|s| s % 16 == 0));
        assert_eq!(c.split_tiles(256), (vec![0], vec![]));
    }

    #[test]
    fn audio_vae_numbers() {
        let c = H3AudioVaeConfig::fasth3_8step();
        assert_eq!(c.hop_length(), 800);
        assert_eq!(c.encoder_rates.iter().product::<usize>(), 800);
        assert_eq!(c.latents_per_second(), 40);
        assert_eq!(c.upsampler_channels(0), (1024, 512));
        assert_eq!(c.upsampler_channels(6), (16, 8));
        let mut len = 207;
        for i in 0..7 {
            let next = c.upsampler_out_len(i, len);
            assert_eq!(next, len * c.decoder_rates[i], "stage {i}");
            len = next;
        }
        assert_eq!(len, 207 * 800);
    }

    #[test]
    fn text_encoder_numbers() {
        let c = H3TextEncoderConfig::fasth3_8step();
        assert_eq!(c.layers_to_run(), 50);
        assert_eq!(c.gqa_groups(), 8);
        assert_eq!(c.num_attention_heads * c.head_dim, 8192);
        assert_eq!(c.num_key_value_heads * c.head_dim, 1024);
        assert_eq!(c.mrope_section.iter().sum::<usize>() * 2, c.head_dim);
        assert_eq!(c.hidden_size, H3TransformerConfig::fasth3_8step().text_dim);
        let inv = c.rope_inv_freq();
        assert_eq!(inv.len(), 64);
        assert_eq!(inv[0], 1.0);
    }

    #[test]
    fn contract_is_consistent() {
        for c in [
            H3InferenceContract::fasth3_8step(),
            H3InferenceContract::fasth3_4step_vsa(),
            H3InferenceContract::fasth3_4step_dense(),
        ] {
            assert_eq!(c.dmd_denoising_steps.len(), c.transformer_forwards);
            assert_eq!(c.num_inference_steps, c.transformer_forwards + 1);
            assert!(c.dmd_denoising_steps.windows(2).all(|w| w[0] > w[1]));
            let j = super::super::schedule::H3JointSchedule::from_contract(&c).unwrap();
            assert_eq!(j.num_steps(), c.transformer_forwards);
        }
        assert!(!H3InferenceContract::fasth3_4step_vsa().dense);
        assert!(H3InferenceContract::fasth3_4step_dense().dense);
        assert_eq!(H3InferenceContract::fasth3_4step_vsa().vsa_sparsity, 0.9);
        assert_eq!(
            H3InferenceContract::named("preview-vsa").unwrap(),
            H3InferenceContract::fasth3_4step_vsa()
        );
        assert!(H3InferenceContract::named("nope").is_err());
    }

    #[test]
    fn canvas_resolution() {
        assert_eq!(resolve_canvas_size(16.0, 9.0).unwrap(), (768, 1344));
        assert_eq!(resolve_canvas_size(9.0, 16.0).unwrap(), (1344, 768));
        assert_eq!(resolve_canvas_size(1.0, 1.0).unwrap(), (768, 768));
        assert_eq!(resolve_canvas_size(4.0, 3.0).unwrap(), (768, 1024));
        assert_eq!(resolve_canvas_size(21.0, 9.0).unwrap(), (672, 1536));
        assert!(resolve_canvas_size(5.0, 1.0).is_err());
    }

    #[test]
    fn frame_alignment() {
        assert_eq!(align_num_frames(120), 124);
        assert_eq!(align_num_frames(124), 124);
        assert_eq!(align_num_frames(240), 243);
        assert_eq!(align_num_frames(360), 362);
        assert_eq!(video_latent_num_frames(124).unwrap(), 37);
        assert_eq!(video_latent_num_frames(362).unwrap(), 107);
        assert!(video_latent_num_frames(120).is_err());
        assert_eq!(audio_latent_num_frames(124), 207);
        assert_eq!(audio_latent_num_frames(243), 405);
        assert_eq!(audio_latent_num_frames(362), 603);
    }

    #[test]
    fn five_and_fifteen_second_sequences() {
        let g5 = H3Geometry::default_16x9(5).unwrap();
        assert_eq!((g5.height, g5.width, g5.num_frames), (768, 1344, 124));
        assert_eq!(
            (g5.latent_frames, g5.latent_height, g5.latent_width),
            (37, 48, 84)
        );
        assert_eq!(g5.token_grid, (37, 24, 42));
        assert_eq!(g5.video_rows(), 37_296);
        assert_eq!(g5.audio_rows(), 414);
        assert_eq!(g5.sequence_length(0), 37_710);
        assert_eq!(g5.sequence_length(256), 37_966);
        assert_eq!(g5.audio_samples(), 165_600);

        let g15 = H3Geometry::default_16x9(15).unwrap();
        assert_eq!(
            (g15.num_frames, g15.latent_frames, g15.audio_latents),
            (362, 107, 603)
        );
        assert_eq!(g15.video_rows(), 107_856);
        assert_eq!(g15.sequence_length(0), 109_062);
        assert_eq!(g15.sequence_length(256), 109_318);

        // Four seconds aligns to 107 frames, below the 124-frame floor.
        assert!(H3Geometry::default_16x9(4).is_err());
        assert!(H3Geometry::default_16x9(16).is_err());
        assert!(H3Geometry::new(770, 1344, 124).is_err());
    }
}
