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
//! in 8192-row chunks. The resident DiT is priced at the precision it loads
//! with ([`H3DitWeights`]: bf16, MXFP8, W8A8 or NVFP4 block linears).
//!
//! A resident FP8 text encoder is not part of these stages. Whether `Auto`
//! loads it is [`auto_text_encoder`]: the resident plan at the DiT's real
//! precision, the encoder ([`text_encoder_fp8_bytes`]) and the denoise reserve
//! of the largest request the process plans for ([`denoise_reserve_bytes`])
//! must fit the free memory measured before any load, and the encoder may
//! never cost the DiT its residency.
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

/// How the DiT's block linears are stored on the device: what the load
/// resolves (`FASTVIDEO_H3_QUANT`, the published NVFP4 tree). Everything
/// outside the block linears (norms, AdaLN, the refiner under MXFP8 / NVFP4,
/// the VSA gate rows under MXFP8 / W8A8) stays as bf16 / float32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum H3DitWeights {
    /// bf16 linears: `FASTVIDEO_H3_QUANT=off`, pre-sm_100 defaults, VDN-H3,
    /// an MLX tree (priced as bf16, the safe side).
    #[default]
    Bf16,
    /// Sol-H3 MXFP8 (the default on sm_100+): blocks
    /// [`MXFP8_FIRST_BLOCK`]..=[`MXFP8_LAST_BLOCK`] as one E4M3 byte per
    /// weight plus one E8M0 scale per 32; the others and the gate rows bf16.
    Mxfp8,
    /// Tensorwise W8A8: every block's linears one byte per weight, the gate
    /// rows bf16.
    W8a8,
    /// FastH3 Trim's published NVFP4 tree: every block linear (gate
    /// included) as E2M1 nibbles plus one E4M3 scale per 16 (rows padded to
    /// 128), resident only.
    Nvfp4,
}

/// The MXFP8 block range of `fastvideo_cudarc::wan::quant::H3QuantPlan`.
pub const MXFP8_FIRST_BLOCK: usize = 2;
pub const MXFP8_LAST_BLOCK: usize = 46;
/// bf16 blocks a pruned stack always keeps at its end.
pub const MXFP8_BF16_TAIL: usize = 3;

impl H3DitWeights {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Mxfp8 => "mxfp8",
            Self::W8a8 => "w8a8",
            Self::Nvfp4 => "nvfp4",
        }
    }
}

/// Device bytes of DiT block `i` resident as `weights` stores it.
pub fn block_bytes_as(cfg: &H3TransformerConfig, i: usize, gate: bool, weights: H3DitWeights) -> u64 {
    let (h, inner, f) = (
        cfg.hidden_size as u64,
        cfg.inner_dim() as u64,
        cfg.ffn_dim as u64,
    );
    // (rows, k) of qkv, out, ff_in, ff_out; the gate is (inner, h).
    let linears = [(3 * inner, h), (h, inner), (2 * f, h), (h, f)];
    let params: u64 = linears.iter().map(|(r, k)| r * k).sum();
    let gate_bf16 = if gate { inner * h * 2 } else { 0 };
    let last_mx = MXFP8_LAST_BLOCK.min(cfg.num_layers.saturating_sub(1 + MXFP8_BF16_TAIL));
    let linear_bytes = match weights {
        H3DitWeights::Bf16 => return block_bytes(cfg, gate),
        H3DitWeights::Mxfp8 if !(MXFP8_FIRST_BLOCK..=last_mx).contains(&i) => {
            return block_bytes(cfg, gate)
        }
        H3DitWeights::Mxfp8 => params + params / 32 + gate_bf16,
        H3DitWeights::W8a8 => params + gate_bf16,
        H3DitWeights::Nvfp4 => {
            let nv = |(r, k): (u64, u64)| r * k / 2 + r.next_multiple_of(128) * k / 16;
            let gate_nv = if gate { nv((inner, h)) } else { 0 };
            linears.iter().map(|&l| nv(l)).sum::<u64>() + gate_nv
        }
    };
    linear_bytes + block_norm_bytes(cfg)
}

/// Device bytes of every DiT block resident as `weights` stores them.
pub fn dit_blocks_bytes(cfg: &H3TransformerConfig, gate: bool, weights: H3DitWeights) -> u64 {
    (0..cfg.num_layers)
        .map(|i| block_bytes_as(cfg, i, gate, weights))
        .sum()
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

/// The MXFP8 copies of the video decoder's block linears
/// (`FASTVIDEO_H3_VAE_MXFP8`, default on for the 36- and 26-block decoders on
/// sm_100+):
/// one E4M3 byte per weight plus one E8M0 scale per 32, kept beside the bf16
/// weights (the unfused path and the check still read those). About 2.3 GiB
/// (measured: 768p peak 72.2 vs 69.8 GB, docs/perf/h3-vae-decode-speed.md).
/// The activations' MXFP8 staging is smaller than the float32 activations
/// [`video_decode_transient_bytes`] already prices for the fused ViT, which
/// keeps them in bf16.
pub fn video_vae_mxfp8_bytes(v: &H3VideoVaeConfig) -> u64 {
    let d = v.decoder_dim() as u64;
    let hid = d * v.decoder_ffn_mult as u64;
    let params = 4 * d * d + d * 2 * hid + hid * d;
    v.decoder_num_layers as u64 * (params + params / 32)
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
/// codes plus one float32 scale per output row, float32 norms): what a
/// resident `Auto` encoder adds ([`auto_text_encoder`]). A bf16 copy of it
/// would be twice this.
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

/// Device bytes one H3 generation's working set was measured to take per
/// packed row, on top of the loaded weights (RTX PRO 6000, h3-turbo,
/// docs/serve/h3-1080p-and-upscaler.md): 1344x768x124 grew the pool by
/// 11.2 GiB over the weights for about 38 k rows, 1920x1088x124 by 22.5 GiB
/// for about 76 k rows (FFN chunking on). That is about 300 KiB per row;
/// this rounds it up. [`plan`] prices the same denoise higher (it is the
/// planning bound: 74 GiB resident at 1920x1088x124 against 52.7 GiB
/// measured); this is the live admission estimate.
pub const MEASURED_WORKING_BYTES_PER_ROW: u64 = 320 << 10;

/// The measured-rate estimate of what one generation at `geometry` needs
/// free beside the loaded weights: [`MEASURED_WORKING_BYTES_PER_ROW`] per
/// packed row (with [`PLAN_TEXT_TOKENS`] text rows) plus
/// [`RUNTIME_ALLOWANCE`]. The serve engine checks a 1080P-tier job against
/// the device's free memory with it before the job starts.
pub fn measured_working_bytes(geometry: &H3Geometry) -> u64 {
    geometry.sequence_length(PLAN_TEXT_TOKENS) as u64 * MEASURED_WORKING_BYTES_PER_ROW
        + RUNTIME_ALLOWANCE
}

/// The device memory a GPU needs before a server offers the 1080P tier at
/// all: the resident plan's weights at the denoise ([`plan`], bf16 DiT with
/// the VSA gate, decoders held with the video decoder's MXFP8 copies; the
/// turbo recipe's MXFP8 linears are smaller) plus the measured-rate working
/// set of the tier's largest 5 s clip, 1920x1088x124
/// ([`measured_working_bytes`]). About 73.5 GiB (71.2 with a bf16-only
/// video decoder): 80 GB
/// and 96 GB cards yes, 48 GB cards no. The whole [`plan`] peak (about
/// 80 GiB) prices the activations well above the 52.7 GiB measured.
pub fn plan_1080p_min_device_bytes() -> u64 {
    let g = H3Geometry::new(
        super::config::H3_CANVAS_SHORT_1080P,
        1920,
        super::config::H3_FASTVIDEO_MIN_DURATION_S * super::config::H3_FPS,
    )
    .expect("the 1080P tier canvas is a valid geometry");
    let p = plan(
        &g,
        H3PlanOptions {
            gate: true,
            ..H3PlanOptions::default()
        },
    );
    let weights = p.stage("denoise").map_or(0, |s| s.weights);
    weights + measured_working_bytes(&g)
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
    /// The video decoder also holds MXFP8 copies of its block linears
    /// ([`video_vae_mxfp8_bytes`]): the default decode on sm_100+. The
    /// pipeline passes what `FASTVIDEO_H3_VAE_MXFP8` resolves to (off on
    /// older GPUs, with `=0` or with TAEH3).
    pub vae_mxfp8: bool,
    /// How the resident DiT's block linears are stored. Streamed blocks are
    /// priced as bf16 whatever this says.
    pub dit: H3DitWeights,
}

impl Default for H3PlanOptions {
    fn default() -> Self {
        Self {
            placement: H3Placement::Resident,
            gate: false,
            text_tokens: PLAN_TEXT_TOKENS,
            vae_tile_batch: 8,
            steps: 50,
            vae_mxfp8: true,
            dit: H3DitWeights::Bf16,
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
    let video_vae = video_vae_bytes(&vcfg)
        + if opts.vae_mxfp8 {
            video_vae_mxfp8_bytes(&vcfg)
        } else {
            0
        };
    let audio_vae = audio_vae_bytes(&acfg);
    let decoders_held = if streamed { 0 } else { video_vae + audio_vae };
    // Skeletons (and, resident, every block) stay loaded for the whole run;
    // a streamed ring exists only while its model runs.
    let dit_skeleton = dit_global_bytes(&cfg, opts.steps)
        + match opts.placement {
            H3Placement::Resident => dit_blocks_bytes(&cfg, opts.gate, opts.dit),
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

/// Device bytes a denoise needs beside the resident weights for `rows`
/// packed rows: 1 MiB per row under float32 activations (QKVG `[S, 4 x
/// 7168]`, BHSD Q/K/V, the SwiGLU hidden and the VSA gathers are all live
/// around one block), 640 KiB under bf16 activations (the GPU default), plus
/// 4 GiB of workspace. This is the measured bound, not [`plan`]'s: a 96 GB
/// card with the FP8 encoder, DiT and decoders resident ran out at the first
/// 5 s VSA step with ~27 GiB free (float32 activations), and the bf16 VSA
/// denoise at 1344x768x124 measured ~25 GiB over the live weights, where
/// [`plan`] prices about 11 GiB.
pub fn denoise_reserve_bytes(rows: usize, bf16_act: bool) -> u64 {
    let per_row: u64 = if bf16_act { 640 << 10 } else { 1 << 20 };
    (rows as u64) * per_row + (4u64 << 30)
}

/// What [`auto_text_encoder`] decides from.
#[derive(Debug, Clone)]
pub struct AutoTextEncoderInputs {
    /// Free device bytes measured before anything of this load is on the card.
    pub free: u64,
    /// The plan the load will run: placement (resident unless the DiT is
    /// configured streamed), VSA gate, VAE MXFP8 copies, DiT precision.
    pub plan: H3PlanOptions,
    /// The largest request the process plans for (its configured canvas and
    /// length; the default 768p 5 s when it names none). Larger requests
    /// (the 1080P tier, long clips) are the post-encode release's job.
    pub request: H3Geometry,
    /// bf16 activations (the GPU default): the denoise reserve's row rate.
    pub bf16_act: bool,
    /// Other pieces the load keeps beside the encoder: the image-to-video
    /// vision tower's reserve when it may load.
    pub extra: u64,
    /// The DiT placement's own need when the placement is decided from the
    /// free memory left after the encoder loads (`dit_offload = auto`, not
    /// the resident-only NVFP4 tree) and the card covers it without the
    /// encoder: the encoder then has to leave it, the DiT has priority.
    pub dit_need: Option<u64>,
}

/// The `Auto` text encoder decision with the numbers it was made from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoTextEncoder {
    /// Load the FP8 encoder resident (`false`: stream it per prompt).
    pub resident: bool,
    pub free: u64,
    /// Everything that has to fit `free`.
    pub need: u64,
    /// The resident FP8 encoder ([`text_encoder_fp8_bytes`]).
    pub encoder: u64,
    /// The plan's weights at the denoise, plus `extra`.
    pub weights: u64,
    /// [`denoise_reserve_bytes`] of the planned request.
    pub reserve: u64,
    /// The plan's own peak over every stage, plus `extra`.
    pub plan_peak: u64,
    /// What the DiT placement needs left after the encoder (0: none).
    pub dit_need: u64,
    /// Packed rows of the planned request.
    pub rows: usize,
}

impl AutoTextEncoder {
    /// One log line with the numbers.
    pub fn describe(&self, opts: &AutoTextEncoderInputs) -> String {
        let gib = |b: u64| b as f64 / GIB as f64;
        let g = &opts.request;
        format!(
            "{}: {:.1} GiB free before load {} need {:.1} GiB = fp8 encoder {:.1} + max(resident weights {:.1} [dit {} {}, vae mxfp8 {}, extra {:.1}] + denoise reserve {:.1} for {}x{}x{} ({} rows) + runtime {:.1}; plan peak {:.1}{})",
            if self.resident { "resident-fp8" } else { "streamed" },
            gib(self.free),
            if self.resident { ">=" } else { "<" },
            gib(self.need),
            gib(self.encoder),
            gib(self.weights),
            opts.plan.dit.as_str(),
            if opts.plan.placement.is_streamed() { "streamed" } else { "resident" },
            opts.plan.vae_mxfp8,
            gib(opts.extra),
            gib(self.reserve),
            g.width,
            g.height,
            g.num_frames,
            self.rows,
            gib(RUNTIME_ALLOWANCE),
            gib(self.plan_peak),
            if self.dit_need > 0 {
                format!("; the DiT placement's {:.1} + extra", gib(self.dit_need))
            } else {
                String::new()
            },
        )
    }
}

/// `TextEncoderChoice::Auto` at load: keep the FP8 encoder resident when the
/// encoder plus the larger of (a) the resident weights at the DiT's real
/// precision (decoders with their MXFP8 copies, refiner, `extra`) with the
/// measured denoise reserve of the planned request and [`RUNTIME_ALLOWANCE`],
/// (b) the plan's own peak, and (c) the DiT placement's need fit `free`.
/// Otherwise the encoder streams.
pub fn auto_text_encoder(opts: &AutoTextEncoderInputs) -> AutoTextEncoder {
    let p = plan(&opts.request, opts.plan);
    let weights = p.stage("denoise").map_or(0, |s| s.weights) + opts.extra;
    let plan_peak = p.peak() + opts.extra;
    let rows = opts.request.sequence_length(opts.plan.text_tokens);
    let reserve = denoise_reserve_bytes(rows, opts.bf16_act);
    let encoder = text_encoder_fp8_bytes(&H3TextEncoderConfig::fasth3_8step());
    let dit_need = opts.dit_need.unwrap_or(0);
    let beside = (weights + reserve + RUNTIME_ALLOWANCE)
        .max(plan_peak)
        .max(dit_need + opts.extra);
    let need = encoder + beside;
    AutoTextEncoder {
        resident: opts.free >= need,
        free: opts.free,
        need,
        encoder,
        weights,
        reserve,
        plan_peak,
        dit_need,
        rows,
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

    /// The video decoder's MXFP8 copies (the default decode on sm_100+) are
    /// about 2.3 GiB, the +2.4 GB of peak the 768p run measured (72.2 vs
    /// 69.8 GB), priced wherever the decoder is: for the whole run resident,
    /// for the decode only streamed.
    /// [`plan`] prices the 36-block decoder for every recipe, so FastH3
    /// Trim's 26-block decoder (MXFP8 by default since 2026-10-14) is
    /// covered: its bf16 weights and MXFP8 copies (1.68 GiB; measured +1.76
    /// GiB on the pod, docs/perf/h3-vae-decode-speed.md §6) both fit what
    /// the plan holds for the larger decoder.
    #[test]
    fn the_trim_decoders_mxfp8_copies_fit_the_planned_decoder() {
        let (big, trim) = (
            H3VideoVaeConfig::fasth3_8step(),
            H3VideoVaeConfig::fasth3_trim_8step(),
        );
        let mx = video_vae_mxfp8_bytes(&trim);
        assert!((1.6..1.75).contains(&gib(mx)), "{:.3}", gib(mx));
        assert!(video_vae_bytes(&trim) + mx <= video_vae_bytes(&big) + video_vae_mxfp8_bytes(&big));
        assert_eq!(mx * 36, video_vae_mxfp8_bytes(&big) * 26);
    }

    #[test]
    fn the_mxfp8_video_decoder_copies_are_priced() {
        let v = H3VideoVaeConfig::fasth3_8step();
        let mx = video_vae_mxfp8_bytes(&v);
        assert!((2.2..2.4).contains(&gib(mx)), "{:.3}", gib(mx));
        let g = H3Geometry::default_16x9(5).unwrap();
        let off = |o: H3PlanOptions| H3PlanOptions {
            vae_mxfp8: false,
            ..o
        };
        let (on_r, off_r) = (
            plan(&g, H3PlanOptions::default()),
            plan(&g, off(H3PlanOptions::default())),
        );
        for (a, b) in on_r.stages.iter().zip(&off_r.stages) {
            assert_eq!(a.total(), b.total() + mx, "resident {}", a.stage);
        }
        let (on_s, off_s) = (
            plan(&g, H3PlanOptions::streamed()),
            plan(&g, off(H3PlanOptions::streamed())),
        );
        for (a, b) in on_s.stages.iter().zip(&off_s.stages) {
            let extra = if a.stage == "video_decode" { mx } else { 0 };
            assert_eq!(a.total(), b.total() + extra, "streamed {}", a.stage);
        }
        show("768p5s mxfp8 vae streamed", &on_s);
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
            // The decoders never sit beside the ring: with bf16 decoder
            // linears the denoise binds. With the MXFP8 copies (the default
            // on sm_100+) the 480p video decode (11.9 GiB) passes the 480p
            // denoise (10.9 GiB), still far under the card.
            let bf16_vae = plan(
                &g,
                H3PlanOptions {
                    vae_mxfp8: false,
                    ..H3PlanOptions::streamed()
                },
            );
            assert_eq!(bf16_vae.peak(), bf16_vae.stage("denoise").unwrap().total());
            let decode = streamed.stage("video_decode").unwrap();
            assert!(decode.weights < 8 * GIB, "{name}: {:.2}", gib(decode.weights));
            assert!(decode.total() < 16 * GIB, "{name}: {:.2}", gib(decode.total()));
        }
    }

    /// The 1080P tier is offered on 80 GB-class cards (an H100 80 GB reports
    /// 79.6 GiB) and up, not on 48 GB ones. The live estimate covers the
    /// measured working sets, 22.5 GiB at 1920x1088x124 and 11.2 GiB at
    /// 1344x768x124, and grows with the clip length.
    #[test]
    fn the_1080p_tier_needs_an_80_gb_class_card() {
        let min = gib(plan_1080p_min_device_bytes());
        eprintln!("1080P tier: planned resident peak {min:.2} GiB");
        assert!((60.0..79.0).contains(&min), "{min:.2}");
        let at = |h: usize, w: usize, s: usize| {
            gib(measured_working_bytes(&H3Geometry::new(h, w, s * 24).unwrap()))
        };
        let (hd5, hd10, hd15, sd5, sd15) = (
            at(1088, 1920, 5),
            at(1088, 1920, 10),
            at(1088, 1920, 15),
            at(768, 1344, 5),
            at(768, 1344, 15),
        );
        eprintln!(
            "measured-rate working set: 1080p 5/10/15 s {hd5:.1}/{hd10:.1}/{hd15:.1} GiB, 768p 5/15 s {sd5:.1}/{sd15:.1} GiB"
        );
        assert!((22.5..28.0).contains(&hd5), "{hd5:.2}");
        assert!((11.2..16.0).contains(&sd5), "{sd5:.2}");
        assert!(hd10 > 1.8 * hd5 - 2.0 && hd15 > hd10);
    }


    /// The resident DiT at the precision it loads with. MXFP8 (45 of 50
    /// blocks, gate rows bf16) is the 23.87 GiB `dit` the RTX PRO 6000 ledger
    /// reported for the VSA turbo recipe (docs/perf/raw-inference.md); NVFP4
    /// every block linear at about 4.5 bits.
    #[test]
    fn the_resident_dit_is_priced_at_its_precision() {
        let cfg = H3TransformerConfig::fasth3_8step();
        let at = |w, gate| gib(dit_blocks_bytes(&cfg, gate, w));
        let bf16 = at(H3DitWeights::Bf16, true);
        assert_eq!(dit_blocks_bytes(&cfg, true, H3DitWeights::Bf16), cfg.num_layers as u64 * block_bytes(&cfg, true));
        let mx = at(H3DitWeights::Mxfp8, true);
        assert!((23.0..24.5).contains(&mx), "{mx:.2}");
        let w8 = at(H3DitWeights::W8a8, true);
        assert!(w8 < mx && w8 > bf16 / 2.0, "{w8:.2}");
        let nv = at(H3DitWeights::Nvfp4, true);
        assert!((10.5..11.5).contains(&nv), "{nv:.2}");
        eprintln!("dit blocks (gated): bf16 {bf16:.2}, mxfp8 {mx:.2}, w8a8 {w8:.2}, nvfp4 {nv:.2} GiB");
        // The plan's resident stages move by exactly the difference.
        let g = H3Geometry::default_16x9(5).unwrap();
        let p = |dit| {
            plan(
                &g,
                H3PlanOptions {
                    gate: true,
                    dit,
                    ..H3PlanOptions::default()
                },
            )
        };
        let (b, m) = (p(H3DitWeights::Bf16), p(H3DitWeights::Mxfp8));
        let diff = dit_blocks_bytes(&cfg, true, H3DitWeights::Bf16)
            - dit_blocks_bytes(&cfg, true, H3DitWeights::Mxfp8);
        for (x, y) in b.stages.iter().zip(&m.stages) {
            assert_eq!(x.total(), y.total() + diff, "{}", x.stage);
        }
        // Streamed blocks are priced as bf16 whatever the precision.
        let s = |dit| {
            plan(
                &g,
                H3PlanOptions {
                    gate: true,
                    dit,
                    ..H3PlanOptions::streamed()
                },
            )
            .peak()
        };
        assert_eq!(s(H3DitWeights::Bf16), s(H3DitWeights::Mxfp8));
    }

    /// The decision's arithmetic: everything it adds is in `need`, and the
    /// DiT placement's need is a floor.
    #[test]
    fn the_auto_text_encoder_need_adds_up() {
        let g = H3Geometry::default_16x9(5).unwrap();
        let inputs = AutoTextEncoderInputs {
            free: 94 * GIB,
            plan: H3PlanOptions {
                gate: true,
                dit: H3DitWeights::Mxfp8,
                ..H3PlanOptions::default()
            },
            request: g.clone(),
            bf16_act: true,
            extra: 2 * GIB,
            dit_need: None,
        };
        let d = auto_text_encoder(&inputs);
        let rows = g.sequence_length(PLAN_TEXT_TOKENS);
        assert_eq!(d.rows, rows);
        assert_eq!(d.reserve, denoise_reserve_bytes(rows, true));
        assert_eq!(d.encoder, text_encoder_fp8_bytes(&H3TextEncoderConfig::fasth3_8step()));
        assert_eq!(d.need, d.encoder + (d.weights + d.reserve + RUNTIME_ALLOWANCE).max(d.plan_peak));
        assert!(d.resident);
        let floor = AutoTextEncoderInputs {
            dit_need: Some(80 * GIB),
            ..inputs.clone()
        };
        let f = auto_text_encoder(&floor);
        assert_eq!(f.need, f.encoder + 82 * GIB);
        assert!(!f.resident);
        // A larger planned request needs more.
        let long = AutoTextEncoderInputs {
            request: H3Geometry::default_16x9(10).unwrap(),
            ..inputs
        };
        assert!(auto_text_encoder(&long).need > d.need);
    }
}
