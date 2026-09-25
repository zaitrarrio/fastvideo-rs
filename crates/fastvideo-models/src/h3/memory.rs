//! Device-memory plan of one H3 generation, priced from the configs and the
//! request geometry, so a placement that would not fit a card fails a host
//! test instead of a GPU run.
//!
//! The reference for the 32 GiB card is sol-engine `models/minimax_h3/
//! RTX5090`: "the 33B DiT, Qwen3-VL conditioner, and VAEs use layerwise
//! component offload so only the active component resides on the GPU", and
//! the full video VAE becomes resident only for decoding. In Rust terms
//! (`fastvideo_cudarc::wan::offload`):
//!
//! * the text encoder streams one layer at a time (its prefetcher keeps up to
//!   three on the device) and is gone before the DiT runs;
//! * the refiner's and the DiT's blocks stream through a ring of `slots`
//!   device buffers; the AdaLN table stays on the host except the running
//!   step's rows ([`adaln_step_bytes`]); the TeaCache rows and those AdaLN
//!   rows are freed when the denoise ends;
//! * the video and audio decoders are loaded for the decode and dropped.
//!
//! [`plan`] prices each stage with either placement. The activation terms
//! follow the allocations of `H3Transformer::forward`: **float32
//! activations** (residual, QKV(G), per-head Q/K/V, the SwiGLU hidden), bf16
//! only as GEMM staging, a chunked score buffer, and the SwiGLU FFN whole or
//! in 8192-row chunks. A resident FP8 text encoder is not part of the
//! streamed plan: `Auto` loads it only with 85 GB free (see
//! [`text_encoder_fp8_bytes`] for what it would add).
//!
//! Against the RTX PRO 6000 run of 1344x768x124 (sol-h3-rtx, TeaCache, DiT
//! streamed, FP8 encoder resident): the denoise grew the pool by 15.2 GiB
//! over the text phase (ring 1.4, the whole AdaLN ladder 0.9, TeaCache 1.5,
//! activations ~11.4); this plan prices the same stage at 17.3 GiB without
//! the whole ladder, so it errs on the safe side by ~2.5 GiB.

use super::config::{
    H3AudioVaeConfig, H3Geometry, H3TextEncoderConfig, H3TransformerConfig, H3VideoVaeConfig,
};

pub const GIB: u64 = 1 << 30;

/// cuBLAS/cuDNN workspaces, the CUDA context's own allocations, allocator
/// slack: added to every stage, as the LTX plan does.
pub const RUNTIME_ALLOWANCE: u64 = 2 * GIB;

/// Dense attention's score chunk: 1 GiB of float32 scores plus its bf16
/// probabilities (`fastvideo_cudarc::wan::attn::DENSE_SCORE_BUDGET`).
pub const SCORE_CHUNK_BYTES: u64 = GIB + GIB / 2;

/// Rows per FFN pass when the whole `[S, 2 ffn]` float32 buffer would exceed
/// [`FFN_WHOLE_BYTES`] (`h3/transformer.rs`).
pub const FFN_ROW_CHUNK: usize = 8192;
pub const FFN_WHOLE_BYTES: u64 = 6 << 30;

/// Text tokens a plan assumes (the H3 prompt template tops out well below).
pub const PLAN_TEXT_TOKENS: usize = 1024;

/// Where the DiT's (and the refiner's) blocks live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H3Placement {
    /// Every block on the device; the decoders loaded at start-up and kept.
    Resident,
    /// Blocks streamed through `slots` device buffers; the decoders loaded
    /// for the decode only.
    Streamed { slots: usize },
}

impl H3Placement {
    pub const STREAMED: Self = Self::Streamed { slots: 2 };

    pub fn is_streamed(self) -> bool {
        matches!(self, Self::Streamed { .. })
    }
}

/// Bytes of one DiT (or refiner) block's linears as bf16, with the gate
/// projection when `gate`. Norms are float32 `[hidden]` / `[head_dim]`.
pub fn block_bytes(cfg: &H3TransformerConfig, gate: bool) -> u64 {
    let (h, i, f) = (
        cfg.hidden_size as u64,
        cfg.inner_dim() as u64,
        cfg.ffn_dim as u64,
    );
    let qkvg = if gate { 4 } else { 3 } * i * h;
    let linears = qkvg + i * h + h * 2 * f + f * h;
    linears * 2 + block_norm_bytes(cfg)
}

/// The float32 norms a streamed block's skeleton keeps on the device.
pub fn block_norm_bytes(cfg: &H3TransformerConfig) -> u64 {
    (2 * cfg.hidden_size + 2 * cfg.attention_head_dim) as u64 * 4
}

/// Everything of the DiT outside the blocks: in/out projections (bf16 with
/// float32 biases), `norm_out`, and the device copy of the output modulation.
pub fn dit_global_bytes(cfg: &H3TransformerConfig, steps: usize) -> u64 {
    let h = cfg.hidden_size as u64;
    let (v, a) = (cfg.video_patch_dim() as u64, cfg.audio_in_channels as u64);
    let lin = |i: u64, o: u64| i * o * 2 + o * 4;
    lin(v, h) + lin(a, h) + lin(h, v) + lin(h, a) + h * 4 + steps as u64 * 4 * h * 4
}

/// `context_embedder` plus the refiner blocks.
pub fn refiner_bytes(cfg: &H3TransformerConfig, placement: H3Placement) -> u64 {
    let embed = (cfg.text_dim * cfg.hidden_size) as u64 * 2 + cfg.hidden_size as u64 * 4;
    embed + blocks_on_device(cfg, cfg.num_refiner_layers, false, placement)
}

/// Device bytes of `n` blocks under `placement`.
pub fn blocks_on_device(
    cfg: &H3TransformerConfig,
    n: usize,
    gate: bool,
    placement: H3Placement,
) -> u64 {
    match placement {
        H3Placement::Resident => n as u64 * block_bytes(cfg, gate),
        H3Placement::Streamed { slots } => {
            n as u64 * block_norm_bytes(cfg) + slots.min(n) as u64 * block_bytes(cfg, gate)
        }
    }
}

/// The ViT video decoder: 36 blocks of bf16 linears with float32 biases and
/// `[dim]` norms / LayerScales, `proj_in` / `proj_out` / `post_quant`.
pub fn video_vae_bytes(v: &H3VideoVaeConfig) -> u64 {
    let d = v.decoder_dim() as u64;
    let hid = d * v.decoder_ffn_mult as u64;
    let lin = |i: u64, o: u64| i * o * 2 + o * 4;
    let block = 4 * lin(d, d) + lin(d, 2 * hid) + lin(hid, d) + 4 * d * 4;
    let c = v.latent_channels as u64;
    v.decoder_num_layers as u64 * block
        + lin(c, c)
        + lin(c, d)
        + lin(d, v.decoder_patch_dim() as u64)
        + (v.decoder_num_register_tokens as u64 + 1) * d * 4
        + 2 * d * 4
}

/// BigVGAN decoder: float32 weight-normed convs (folded at load), every
/// stage's transposed conv and nine resblock convs of each kernel size.
pub fn audio_vae_bytes(a: &H3AudioVaeConfig) -> u64 {
    let mut params = (a.latent_dim * a.decoder_dim * 7) as u64;
    let kernels: usize = a.resblock_kernel_sizes.iter().sum();
    for i in 0..a.decoder_rates.len() {
        let (cin, cout) = a.upsampler_channels(i);
        params += (cin * cout * a.decoder_kernel_sizes[i]) as u64;
        params += (2 * 3 * cout * cout * kernels) as u64;
    }
    params * 4
}

/// The streamed text encoder beside nothing else: up to three bf16 decoder
/// layers (computing, staged, in the channel), the embedding table, and every
/// hidden state up to the tapped one.
pub fn text_stage_bytes(t: &H3TextEncoderConfig, tokens: usize) -> u64 {
    let (h, i) = (t.hidden_size as u64, t.intermediate_size as u64);
    let q = (t.num_attention_heads * t.head_dim) as u64;
    let kv = (t.num_key_value_heads * t.head_dim) as u64;
    let layer = (h * q + 2 * h * kv + q * h + 3 * h * i) * 2;
    let embed = (t.vocab_size * t.hidden_size) as u64 * 2;
    let states = (t.output_hidden_state_index + 1) as u64 * tokens as u64 * h * 4;
    let work = tokens as u64 * (4 * h + 2 * i) * 4;
    3 * layer + embed + states + work
}

/// Transient bytes of one DiT forward at `rows` packed rows on top of what the
/// denoise keeps, float32 activations with bf16 GEMM staging.
pub fn forward_transient_bytes(cfg: &H3TransformerConfig, rows: usize, gate: bool) -> u64 {
    let s = rows as u64;
    let (d, i, f) = (
        cfg.hidden_size as u64,
        cfg.inner_dim() as u64,
        cfg.ffn_dim as u64,
    );
    let g = if gate { 4 } else { 3 };
    // x (and the next x while a gated add runs), the normed input, the fused
    // QKV(G) output, Q/K/V heads plus one norm/RoPE temporary, their bf16
    // casts for attention, and one score chunk.
    let attn = s * 4 * (2 * d + g * i + 4 * i) + s * 2 * 3 * i + SCORE_CHUNK_BYTES;
    // The QKVG GEMM itself: bf16 input and output beside the float32 result.
    let qkvg = s * 4 * (2 * d + g * i) + s * 2 * (d + g * i);
    let whole = s * 2 * f * 4 <= FFN_WHOLE_BYTES;
    let c = if whole { s } else { FFN_ROW_CHUNK as u64 };
    // x, n, the finished output rows (and their concatenation when chunked),
    // one chunk's up-projection with bf16 staging, then its activation.
    let parts = if whole { 0 } else { 2 * s * d * 4 };
    let up = c * (2 * f * 4 + 2 * f * 2 + d * 2);
    let act = c * (2 * f * 4 + f * 4);
    let down = c * (f * 4 + f * 2 + d * 2 + d * 4);
    let ffn = s * 4 * 2 * d + parts + up.max(act).max(down);
    attn.max(qkvg).max(ffn)
}

/// What the denoise holds for its whole length beside the weights: the rotary
/// tables, the refined text, the latents and their scheduler state, the
/// TeaCache signal / residual / pending rows (RTX recipe), and a policy
/// allowance of two residual-sized buffers for VSA or Sol-Attn gathers.
pub fn denoise_resident_bytes(cfg: &H3TransformerConfig, rows: usize, text: usize) -> u64 {
    let s = rows as u64;
    let d = cfg.hidden_size as u64;
    let rope = 2 * s * cfg.rotary_dim() as u64 * 4;
    let text = text as u64 * d * 4;
    let latents = 4 * s * cfg.video_patch_dim() as u64 * 4;
    let teacache = 3 * s * d * 4;
    let policy = 2 * s * d * 4;
    rope + text + latents + teacache + policy + adaln_step_bytes(cfg)
}

/// The AdaLN rows on the device during one step: that step's
/// `[blocks, 3, 6, hidden]` ladder rows and the keyframe table of the same
/// shape (float32). The whole `[steps, ...]` ladder used to be uploaded and
/// kept for the life of the process (0.9 GiB at 50 forwards).
pub fn adaln_step_bytes(cfg: &H3TransformerConfig) -> u64 {
    2 * (cfg.num_layers * 3 * 6 * cfg.hidden_size) as u64 * 4
}

/// Device bytes of the resident FP8 text encoder (layers 0..=tap as E4M3
/// codes plus one float32 scale per output row, float32 norms): what `Auto`
/// adds on a card with 85 GB free. A bf16 copy of it would be twice this.
pub fn text_encoder_fp8_bytes(t: &H3TextEncoderConfig) -> u64 {
    let (h, i) = (t.hidden_size as u64, t.intermediate_size as u64);
    let q = (t.num_attention_heads * t.head_dim) as u64;
    let kv = (t.num_key_value_heads * t.head_dim) as u64;
    let params = h * q + 2 * h * kv + q * h + 3 * h * i;
    let rows = q + 2 * kv + h + 2 * i + h;
    let norms = 2 * h + 2 * t.head_dim as u64;
    (t.output_hidden_state_index as u64) * (params + rows * 4 + norms * 4)
}

/// The ViT decode of one temporal chunk: `tile_batch` tiles of
/// `(tokens_chunk + overlap)` latent frames x a 256 px tile, through one
/// block at a time, plus every decoded tile of the chunk and the stitched
/// frames.
pub fn video_decode_transient_bytes(
    v: &H3VideoVaeConfig,
    g: &H3Geometry,
    tile_batch: usize,
) -> u64 {
    let ratio = v.spatial_compression_ratio();
    let tile_latent = v.tile_sample_min_size / ratio;
    let frames = v.tokens_chunk_size() + v.token_overlap();
    let tile_tokens =
        (frames * tile_latent * tile_latent + v.decoder_num_register_tokens + 1) as u64;
    let n_h = v.split_tiles(g.height).0.len();
    let n_w = v.split_tiles(g.width).0.len();
    let tiles = (n_h * n_w) as u64;
    let batch = tiles.min(tile_batch.max(1) as u64);
    let t = batch * tile_tokens;
    let d = v.decoder_dim() as u64;
    let hid = d * v.decoder_ffn_mult as u64;
    let vit = t * 4 * (6 * d + 2 * hid + hid) + t * 2 * (d + 2 * hid);
    let chunk_frames = (frames * v.temporal_compression_ratio()) as u64;
    let tile_px = (v.tile_sample_min_size * v.tile_sample_min_size) as u64;
    let decoded = tiles * chunk_frames * 3 * tile_px * 4;
    let stitched = 2 * chunk_frames * 3 * (g.height * g.width) as u64 * 4;
    vit + decoded + stitched
}

/// One stage of the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageBytes {
    pub stage: &'static str,
    pub weights: u64,
    pub working: u64,
}

impl StageBytes {
    pub fn total(&self) -> u64 {
        self.weights + self.working + RUNTIME_ALLOWANCE
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H3MemoryPlan {
    pub stages: Vec<StageBytes>,
}

impl H3MemoryPlan {
    pub fn peak(&self) -> u64 {
        self.stages.iter().map(StageBytes::total).max().unwrap_or(0)
    }

    pub fn stage(&self, name: &str) -> Option<&StageBytes> {
        self.stages.iter().find(|s| s.stage == name)
    }
}

/// Options of [`plan`].
#[derive(Debug, Clone, Copy)]
pub struct H3PlanOptions {
    pub placement: H3Placement,
    /// `to_gate_compress` loaded (VSA recipes).
    pub gate: bool,
    pub text_tokens: usize,
    /// `FASTVIDEO_H3_VAE_TILE_BATCH` (default 8).
    pub vae_tile_batch: usize,
    /// Ladder steps (rows of the output modulation kept on the device).
    pub steps: usize,
}

impl Default for H3PlanOptions {
    fn default() -> Self {
        Self {
            placement: H3Placement::Resident,
            gate: false,
            text_tokens: PLAN_TEXT_TOKENS,
            vae_tile_batch: 8,
            steps: 50,
        }
    }
}

impl H3PlanOptions {
    pub fn streamed() -> Self {
        Self {
            placement: H3Placement::STREAMED,
            ..Self::default()
        }
    }
}

/// Every stage of a T2AV generation at `geometry`: text (streamed encoder),
/// refine, denoise, audio decode, video decode.
pub fn plan(geometry: &H3Geometry, opts: H3PlanOptions) -> H3MemoryPlan {
    let cfg = H3TransformerConfig::fasth3_8step();
    let vcfg = H3VideoVaeConfig::fasth3_8step();
    let acfg = H3AudioVaeConfig::fasth3_8step();
    let tcfg = H3TextEncoderConfig::fasth3_8step();
    let streamed = opts.placement.is_streamed();
    let rows = geometry.sequence_length(opts.text_tokens);
    let (video_vae, audio_vae) = (video_vae_bytes(&vcfg), audio_vae_bytes(&acfg));
    let decoders_held = if streamed { 0 } else { video_vae + audio_vae };
    // Skeletons (and, resident, every block) stay loaded for the whole run;
    // a streamed ring exists only while its model runs.
    let dit_skeleton = dit_global_bytes(&cfg, opts.steps)
        + match opts.placement {
            H3Placement::Resident => cfg.num_layers as u64 * block_bytes(&cfg, opts.gate),
            H3Placement::Streamed { .. } => cfg.num_layers as u64 * block_norm_bytes(&cfg),
        };
    let refiner_held = match opts.placement {
        H3Placement::Resident => refiner_bytes(&cfg, opts.placement),
        H3Placement::Streamed { .. } => refiner_bytes(&cfg, H3Placement::Streamed { slots: 0 }),
    };
    let dit_ring = match opts.placement {
        H3Placement::Resident => 0,
        H3Placement::Streamed { slots } => {
            slots.min(cfg.num_layers) as u64 * block_bytes(&cfg, opts.gate)
        }
    };
    let refiner_ring = match opts.placement {
        H3Placement::Resident => 0,
        H3Placement::Streamed { slots } => {
            slots.min(cfg.num_refiner_layers) as u64 * block_bytes(&cfg, false)
        }
    };
    let held = dit_skeleton + refiner_held + decoders_held;
    let t = opts.text_tokens as u64 * cfg.hidden_size as u64 * 4;
    H3MemoryPlan {
        stages: vec![
            StageBytes {
                stage: "text",
                weights: held,
                working: text_stage_bytes(&tcfg, opts.text_tokens),
            },
            StageBytes {
                stage: "refine",
                weights: held + refiner_ring,
                working: forward_transient_bytes(&cfg, opts.text_tokens, false)
                    + opts.text_tokens as u64 * cfg.text_dim as u64 * 4
                    + t,
            },
            StageBytes {
                stage: "denoise",
                weights: held + dit_ring,
                working: denoise_resident_bytes(&cfg, rows, opts.text_tokens)
                    + forward_transient_bytes(&cfg, rows, opts.gate),
            },
            StageBytes {
                stage: "audio_decode",
                weights: held + if streamed { audio_vae } else { 0 },
                working: 64 * (geometry.audio_samples() * 2) as u64 * 4
                    + (geometry.video_rows() * cfg.video_patch_dim()) as u64 * 4,
            },
            StageBytes {
                stage: "video_decode",
                weights: held + if streamed { video_vae } else { 0 },
                working: video_decode_transient_bytes(&vcfg, geometry, opts.vae_tile_batch)
                    + 2 * (geometry.video_rows() * cfg.video_patch_dim()) as u64 * 4,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gib(b: u64) -> f64 {
        b as f64 / GIB as f64
    }

    fn show(name: &str, p: &H3MemoryPlan) {
        for s in &p.stages {
            eprintln!(
                "{name} {:>12}: weights {:6.2} GiB, working {:6.2} GiB, total {:6.2} GiB",
                s.stage,
                gib(s.weights),
                gib(s.working),
                gib(s.total())
            );
        }
    }

    #[test]
    fn the_dit_is_about_thirty_seven_gib_of_blocks() {
        let cfg = H3TransformerConfig::fasth3_8step();
        let dit = cfg.num_layers as u64 * block_bytes(&cfg, false);
        assert!((35.0..39.0).contains(&gib(dit)), "{:.2}", gib(dit));
        let gated = cfg.num_layers as u64 * block_bytes(&cfg, true);
        assert!((39.0..42.0).contains(&gib(gated)), "{:.2}", gib(gated));
        let vae = video_vae_bytes(&H3VideoVaeConfig::fasth3_8step());
        assert!((4.0..5.5).contains(&gib(vae)), "{:.2}", gib(vae));
        assert!(audio_vae_bytes(&H3AudioVaeConfig::fasth3_8step()) < GIB);
    }

    /// The resident FP8 encoder is 22.7 GiB, what the PRO 6000 run's
    /// `device_gib` reported; the ~45.5 GiB live after its text phase was that
    /// encoder dequantized once and kept as bf16 (twice the size).
    #[test]
    fn the_fp8_text_encoder_is_half_the_live_bytes_the_run_measured() {
        let t = H3TextEncoderConfig::fasth3_8step();
        let fp8 = gib(text_encoder_fp8_bytes(&t));
        assert!((22.5..23.0).contains(&fp8), "{fp8:.2}");
        let measured_live_after_text = 45.47;
        assert!((2.0 * fp8 - measured_live_after_text).abs() < 0.3);
        // Only one step of the AdaLN ladder is on the device.
        assert!(adaln_step_bytes(&H3TransformerConfig::fasth3_8step()) < 64 << 20);
    }

    /// The RTX 5090 workloads (768p 5 s, the reference 1344x768 cell, and the
    /// GB10 480p cell, 832x480 x 124 frames) stay under 30 GiB streamed; the
    /// resident DiT alone would not fit a 32 GiB card.
    #[test]
    fn streamed_fits_a_32_gib_card() {
        let cells = [
            ("768p5s", H3Geometry::default_16x9(5).unwrap()),
            ("480p124f", H3Geometry::new(480, 832, 124).unwrap()),
        ];
        for (name, g) in cells {
            let streamed = plan(&g, H3PlanOptions::streamed());
            show(name, &streamed);
            assert!(
                streamed.peak() < 30 * GIB,
                "{name}: planned peak {:.2} GiB",
                gib(streamed.peak())
            );
            // With the VSA gate loaded too.
            let gated = plan(
                &g,
                H3PlanOptions {
                    gate: true,
                    ..H3PlanOptions::streamed()
                },
            );
            assert!(
                gated.peak() < 30 * GIB,
                "{name} gated {:.2}",
                gib(gated.peak())
            );
            let resident = plan(&g, H3PlanOptions::default());
            show(name, &resident);
            assert!(resident.peak() > 32 * GIB);
            // The denoise binds; the decoders never sit beside the ring.
            assert_eq!(streamed.peak(), streamed.stage("denoise").unwrap().total());
        }
    }
}
