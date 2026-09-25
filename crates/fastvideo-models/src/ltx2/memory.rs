//! Device-memory plan for the distilled two-stage run on one card.
//!
//! The reference is the sol-engine single-GPU profile
//! (`models/ltx25/RTX5090/`): workload `4k5s` = 3840×2176, 121 frames at
//! 24 fps (130 560 stage-2 video tokens), or `1080p20s` = 1920×1088, 481
//! frames. Its memory measures that leave the numbers unchanged are mirrored
//! here and in `fastvideo-cudarc::ltx2`:
//!
//! * [`FeedForwardChunking`] — `memory.py`: every `FeedForward` whose token
//!   axis holds at least 65 536 rows runs as `torch.split(x, 16384, dim=-2)`
//!   pieces, concatenated back. Rows are independent, so this is exact; using
//!   the same split keeps the GEMM shapes the reference ran.
//! * each pipeline block owns its model for one call (`ltx_pipelines/utils/
//!   blocks.py`, `AllocatorTrimStrategy.TRIM`): the text encoder is gone before
//!   the DiT runs, the upsampler lives only for the upsample, the DiT is gone
//!   before the VAE decodes, and the allocator is trimmed after each.
//!
//! [`plan_distilled_two_stage`] prices every phase of the Rust pipeline from
//! the config alone (weights from the layer shapes, activations from the token
//! counts), so a geometry that would not fit fails a host test instead of a
//! GPU run.

use super::config::{
    Ltx2Config, Ltx2LatentUpsamplerConfig, Ltx2TransformerConfig, Ltx2VideoVaeConfig,
};
use super::tiling::{DecodePlan, TileSizeConfig};

pub const GIB: u64 = 1 << 30;

/// Dense attention's query chunk is sized so one score buffer holds at most
/// this many float32 elements (1 GiB). Mirrors
/// `fastvideo_cudarc::wan::attn::DENSE_SCORE_BUDGET`, which a test there
/// pins to this value.
pub const DENSE_SCORE_BUDGET_ELEMS: usize = 256 * 1024 * 1024;

/// `models/ltx25/RTX5090/memory.py::FeedForwardChunking`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeedForwardChunking {
    pub chunk_tokens: usize,
    pub min_tokens: usize,
}

impl FeedForwardChunking {
    /// The BF16 RTX 5090 profile: `chunk_tokens=16384`, `min_tokens=65536`.
    pub const RTX5090: Self = Self {
        chunk_tokens: 16384,
        min_tokens: 65536,
    };

    /// Never chunk (for comparisons).
    pub const OFF: Self = Self {
        chunk_tokens: usize::MAX,
        min_tokens: usize::MAX,
    };

    /// `(start, len)` of each piece: the whole axis below `min_tokens`, else
    /// `torch.split(x, chunk_tokens)` — equal pieces and a shorter tail.
    pub fn spans(&self, tokens: usize) -> Vec<(usize, usize)> {
        if tokens < self.min_tokens || self.chunk_tokens == 0 {
            return vec![(0, tokens)];
        }
        (0..tokens)
            .step_by(self.chunk_tokens)
            .map(|start| (start, self.chunk_tokens.min(tokens - start)))
            .collect()
    }
}

/// The two workloads of `run_ltx25_gpu.sh`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rtx5090Workload {
    /// `4k5s`: 3840×2176, 121 frames.
    Uhd5s,
    /// `1080p20s`: 1920×1088, 481 frames.
    Fhd20s,
}

impl Rtx5090Workload {
    pub const DEFAULT: Self = Self::Uhd5s;

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "4k5s" => Some(Self::Uhd5s),
            "1080p20s" => Some(Self::Fhd20s),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Uhd5s => "4k5s",
            Self::Fhd20s => "1080p20s",
        }
    }

    pub fn width(self) -> usize {
        match self {
            Self::Uhd5s => 3840,
            Self::Fhd20s => 1920,
        }
    }

    pub fn height(self) -> usize {
        match self {
            Self::Uhd5s => 2176,
            Self::Fhd20s => 1088,
        }
    }

    pub fn num_frames(self) -> usize {
        match self {
            Self::Uhd5s => 121,
            Self::Fhd20s => 481,
        }
    }

    /// `--frame-rate 24` for both.
    pub fn frame_rate(self) -> f64 {
        24.0
    }

    /// The script's `TOKENS` (stage-2 video tokens; the 1080p20s value is the
    /// script's, which the exact-AdaLN table builder is keyed by).
    pub fn script_tokens(self) -> usize {
        match self {
            Self::Uhd5s => 130_560,
            Self::Fhd20s => 124_440,
        }
    }
}

/// Bytes of one bf16-weight linear as the Rust loader holds it: the weight
/// at `weight_bytes` per element, the bias in float32.
fn linear(i: usize, o: usize, bias: bool, weight_bytes: u64) -> u64 {
    (i * o) as u64 * weight_bytes + if bias { o as u64 * 4 } else { 0 }
}

/// `LTX2Attention`: q/k/v/out (always biased), optional per-head gate
/// logits, and the two float32 `[inner]` q/k norms.
fn attention(q: usize, ctx: usize, inner: usize, heads: usize, gated: bool, wb: u64) -> u64 {
    linear(q, inner, true, wb)
        + 2 * linear(ctx, inner, true, wb)
        + linear(inner, q, true, wb)
        + if gated { linear(q, heads, true, wb) } else { 0 }
        + 2 * inner as u64 * 4
}

/// `LTX2AdaLayerNormSingle`: sinusoid → d → d → rows·d.
fn adaln(sinusoid: usize, d: usize, rows: usize, wb: u64) -> u64 {
    linear(sinusoid, d, true, wb) + linear(d, d, true, wb) + linear(d, rows * d, true, wb)
}

/// Bytes of one transformer block: every linear at `weight_bytes` (2 for
/// the bf16 fast path), tables and norms in float32.
pub fn dit_block_bytes(t: &Ltx2TransformerConfig, weight_bytes: u64) -> u64 {
    let wb = weight_bytes;
    let (dv, da) = (t.inner_dim(), t.audio_inner_dim());
    let (hv, ha) = (t.num_attention_heads, t.audio_num_attention_heads);
    let cross = t.av_cross_inner_dim();
    let v_rows = if t.cross_attn_mod { 9 } else { 6 };
    let a_rows = if t.audio_cross_attn_mod { 9 } else { 6 };
    let prompt_mod = t.cross_attn_mod || t.audio_cross_attn_mod;
    let f32s = |n: usize| n as u64 * 4;

    let video = 2 * attention(dv, dv, dv, hv, t.gated_attn, wb)
        + linear(dv, t.ff_inner_dim(), t.ff_bias, wb)
        + linear(t.ff_inner_dim(), dv, t.ff_bias, wb)
        + f32s(v_rows * dv + 5 * dv + if prompt_mod { 2 * dv } else { 0 });
    let audio = 2 * attention(da, da, da, ha, t.audio_gated_attn, wb)
        + linear(da, t.audio_ff_inner_dim(), t.audio_ff_bias, wb)
        + linear(t.audio_ff_inner_dim(), da, t.audio_ff_bias, wb)
        + f32s(a_rows * da + 5 * da + if prompt_mod { 2 * da } else { 0 });
    let av = attention(dv, da, cross, ha, t.gated_attn, wb)
        + attention(da, dv, cross, ha, t.audio_gated_attn, wb);
    video + audio + av
}

/// Bytes of the modules outside the blocks (projections, AdaLN, heads).
pub fn dit_global_bytes(t: &Ltx2TransformerConfig, weight_bytes: u64) -> u64 {
    let wb = weight_bytes;
    let (dv, da) = (t.inner_dim(), t.audio_inner_dim());
    let v_rows = if t.cross_attn_mod { 9 } else { 6 };
    let a_rows = if t.audio_cross_attn_mod { 9 } else { 6 };
    let prompt_mod = t.cross_attn_mod || t.audio_cross_attn_mod;
    let f32s = |n: usize| n as u64 * 4;
    let s = t.timestep_proj_dim;
    let mut globals = linear(t.in_channels, dv, true, wb)
        + linear(t.audio_in_channels, da, true, wb)
        + adaln(s, dv, v_rows, wb)
        + adaln(s, da, a_rows, wb)
        + adaln(s, dv, 4, wb)
        + adaln(s, da, 4, wb)
        + adaln(s, dv, 1, wb)
        + adaln(s, da, 1, wb)
        + f32s(2 * dv + 2 * da + dv + da)
        + linear(dv, t.out_channels, true, wb)
        + linear(da, t.audio_out_channels, true, wb);
    if t.use_prompt_embeddings {
        globals += linear(t.caption_channels, dv, true, wb)
            + linear(dv, dv, true, wb)
            + linear(t.caption_channels, da, true, wb)
            + linear(da, da, true, wb);
    }
    if prompt_mod && t.use_prompt_adaln_single {
        globals += adaln(s, dv, 2, wb) + adaln(s, da, 2, wb);
    }
    if t.use_keyframes_abs_pos_embedding {
        globals += f32s(dv);
    }
    globals
}

/// DiT bytes on the device with every block resident. Walks the same
/// modules as `Ltx2Transformer::load`.
pub fn dit_weight_bytes(t: &Ltx2TransformerConfig, weight_bytes: u64) -> u64 {
    t.num_layers as u64 * dit_block_bytes(t, weight_bytes) + dit_global_bytes(t, weight_bytes)
}

/// Where the DiT's blocks live (`fastvideo_cudarc::wan::offload`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitPlacement {
    /// Every block on the device for the whole run.
    Resident,
    /// Blocks in pinned host memory, `slots` of them on the device at once
    /// (the ring: `lookahead + 1`); the globals and block tables stay.
    Streamed { slots: usize },
}

impl DitPlacement {
    /// The default streamed ring: one block computing, one arriving.
    pub const STREAMED: Self = Self::Streamed { slots: 2 };

    /// Device bytes of the DiT under this placement.
    pub fn dit_bytes(self, t: &Ltx2TransformerConfig, weight_bytes: u64) -> u64 {
        match self {
            Self::Resident => dit_weight_bytes(t, weight_bytes),
            Self::Streamed { slots } => {
                // The skeletons keep their float32 tables and biases.
                // `weight_bytes = 0` prices the biases and tables alone.
                let tables = t.num_layers as u64 * dit_block_bytes(t, 0);
                dit_global_bytes(t, weight_bytes)
                    + tables
                    + slots as u64 * dit_block_bytes(t, weight_bytes)
            }
        }
    }

    pub fn is_streamed(self) -> bool {
        matches!(self, Self::Streamed { .. })
    }
}

/// 3×3×3 conv, float32 weight and bias.
fn conv3(cin: usize, cout: usize) -> u64 {
    (27 * cin * cout + cout) as u64 * 4
}

/// Conv video VAE decoder, float32 on the device.
pub fn vae_decoder_bytes(v: &Ltx2VideoVaeConfig) -> u64 {
    let stages = v.decoder_stages();
    let mut total = conv3(v.latent_channels, stages[0].channels);
    for st in &stages {
        if let Some(up) = &st.upsampler {
            total += conv3(up.in_channels, up.conv_out_channels);
        }
        total += st.resnet_layers as u64 * 2 * conv3(st.channels, st.channels);
    }
    let last = stages.last().map_or(0, |s| s.channels);
    total
        + conv3(last, v.out_channels * v.patch_size * v.patch_size)
        + 2 * v.latent_channels as u64 * 4
}

/// Spatial ×2 latent upsampler, float32 on the device.
pub fn latent_upsampler_bytes(u: &Ltx2LatentUpsamplerConfig) -> u64 {
    let mid = u.mid_channels;
    let group_norm = 2 * mid as u64 * 4;
    let res_block = 2 * conv3(mid, mid) + 2 * group_norm;
    conv3(u.in_channels, mid)
        + group_norm
        + 2 * u.num_blocks_per_stage as u64 * res_block
        + ((9 * mid * 4 * mid + 4 * mid) as u64 * 4)
        + conv3(mid, u.in_channels)
}

/// Audio VAE, vocoder + BWE, cuBLAS/cuDNN workspaces and allocator slack.
/// Their shapes are small next to everything else here; this is a bound.
pub const RUNTIME_ALLOWANCE: u64 = 2 * GIB;

/// Peak transient bytes of one DiT forward's video sub-layers on top of
/// what the forward keeps for its whole length, float32 activations, as the
/// Rust forward allocates them (bf16 linears stage `x`/`y` in bf16 beside the
/// float32 result).
pub fn dit_forward_transient_bytes(
    t: &Ltx2TransformerConfig,
    video_tokens: usize,
    text_tokens: usize,
    ffn: FeedForwardChunking,
    score_budget_elems: usize,
) -> u64 {
    let s = video_tokens as u64;
    let dv = t.inner_dim() as u64;
    let x = s * dv * 4; // one [S, dv] float32 activation
    let ff = t.ff_inner_dim() as u64;
    // Dense SDPA holds one query chunk of float32 scores plus its bf16
    // probabilities; the chunk is at least one query row.
    let scores = |keys: u64, heads: u64| {
        let row = heads * keys;
        let rows = (score_budget_elems as u64 / row.max(1)).clamp(1, s.max(1));
        rows * row * 4 + rows * row * 2
    };
    let hv = t.num_attention_heads as u64;
    // Self-attention: xv, h, q, k, v, out, and V cast to bf16 (x/2) during SDPA.
    let self_attn = x * 13 / 2 + scores(s, hv);
    // Text cross-attention: xv, h (+ modulated copy), q, out, gated u, sum.
    let text_attn = 6 * x + scores(text_tokens as u64, hv);
    // FFN: xv, h, then per span the input copy, up (bf16 in/out + f32 act),
    // down (bf16 in/out + f32 out), beside the outputs of earlier spans and,
    // once there is more than one, the concatenation.
    let spans = ffn.spans(video_tokens);
    let span_peak = |len: u64| {
        let copy = if spans.len() > 1 { len * dv * 4 } else { 0 };
        let up = len * ff * 4 + len * ff * 2 + len * dv * 2;
        let down = len * ff * 4 + len * ff * 2 + len * dv * 2 + len * dv * 4;
        copy + up.max(down)
    };
    let mut done = 0u64;
    let mut ffn_peak = 0u64;
    for &(_, len) in &spans {
        let len = len as u64;
        ffn_peak = ffn_peak.max(done * dv * 4 + span_peak(len));
        done += len;
    }
    if spans.len() > 1 {
        ffn_peak = ffn_peak.max(2 * x);
    }
    let ffn_total = 2 * x + ffn_peak;
    self_attn.max(text_attn).max(ffn_total)
}

/// What one DiT forward keeps for its whole length: the rotary tables of the
/// geometry (cos and sin, `[heads·tokens, head_dim]` each, float32), the
/// projected text, and the Euler loop's latents.
pub fn dit_forward_resident_bytes(
    t: &Ltx2TransformerConfig,
    video_tokens: usize,
    audio_tokens: usize,
    text_tokens: usize,
) -> u64 {
    let (s, a, l) = (video_tokens as u64, audio_tokens as u64, text_tokens as u64);
    let (dv, da) = (t.inner_dim() as u64, t.audio_inner_dim() as u64);
    let cross = t.av_cross_inner_dim() as u64;
    let ropes = 2 * 4 * (s * dv + a * da + s * cross + a * cross);
    let text = 4 * l * (dv + da);
    let latents = 3 * 4 * (s * t.in_channels as u64 + a * t.audio_in_channels as u64);
    // The audio stream's own activations ride along every sub-layer.
    let audio_stream = 10 * a * (da * 4 + t.audio_ff_inner_dim() as u64 * 4);
    ropes + text + latents + audio_stream
}

/// Bytes of the streamed video-VAE decode's working set: one latent frame
/// fed at a time, every stage streamed (two carried frames per temporal conv),
/// about six activations of the widest stage alive at once.
pub fn vae_decode_transient_bytes(v: &Ltx2VideoVaeConfig, latent_h: usize, latent_w: usize) -> u64 {
    let mut worst = 0u64;
    let (mut frames, mut h, mut w) = (1usize, latent_h, latent_w);
    for st in v.decoder_stages() {
        if let Some(up) = &st.upsampler {
            // Conv output before depth-to-space, at the input resolution.
            let conv_out = (frames + 2) * h * w * up.conv_out_channels;
            worst = worst.max(conv_out as u64 * 4 * 2);
            frames *= up.stride.0;
            h *= up.stride.1;
            w *= up.stride.2;
        }
        let act = ((frames + 2) * h * w * st.channels) as u64 * 4;
        worst = worst.max(6 * act);
    }
    let out = (frames + 2) * h * w * v.out_channels * v.patch_size * v.patch_size;
    worst.max(out as u64 * 4 * 4)
}

/// Working set of the tiled decode LTX-2.5 runs (`AUTO_TILING`,
/// `VideoDecoder::decode_tiled`): every spatial tile of one temporal group is
/// decoded (each tile streamed as above, its frames then gathered and
/// concatenated) and held as float32 until the group is blended, beside the
/// overlap carried from the previous group and the blended chunk being
/// written. `None` when the plan is invalid for the grid.
pub fn vae_tiled_decode_bytes(
    v: &Ltx2VideoVaeConfig,
    grid: [usize; 3],
    tiles: &TileSizeConfig,
) -> Option<u64> {
    let scale = [
        v.temporal_compression_ratio,
        v.spatial_compression_ratio,
        v.spatial_compression_ratio,
    ];
    let plan = DecodePlan::new(grid, tiles, scale).ok()?;
    let rgb = |f: usize, h: usize, w: usize| (f * v.out_channels * h * w) as u64 * 4;
    let (max_h, max_w) = (
        plan.height.iter().map(|t| t.out.len()).max()?,
        plan.width.iter().map(|t| t.out.len()).max()?,
    );
    let (max_lh, max_lw) = (
        plan.height.iter().map(|t| t.latent.len()).max()?,
        plan.width.iter().map(|t| t.latent.len()).max()?,
    );
    let mut worst = 0u64;
    for (i, tt) in plan.time.iter().enumerate() {
        let frames = tt.out.len();
        let held: u64 = plan
            .height
            .iter()
            .flat_map(|h| plan.width.iter().map(move |w| (h.out.len(), w.out.len())))
            .map(|(h, w)| rgb(frames, h, w))
            .sum();
        let carry = plan
            .time
            .get(i + 1)
            .map_or(0, |n| tt.out.end.saturating_sub(n.out.start));
        // One more tile being decoded: its streamed working set plus its
        // gathered frames and their concatenation.
        let decoding =
            vae_decode_transient_bytes(v, max_lh, max_lw) + 2 * rgb(frames, max_h, max_w);
        let blending = rgb(8, plan.out[1], plan.out[2]) * 3;
        worst = worst.max(held + rgb(carry, plan.out[1], plan.out[2]) + decoding.max(blending));
    }
    Some(worst)
}

/// Transient bytes of the latent upsampler on the stage-1 latent grid.
pub fn upsample_transient_bytes(u: &Ltx2LatentUpsamplerConfig, grid: [usize; 3]) -> u64 {
    let [f, h, w] = grid;
    // After the pixel shuffle: [mid, F, 2H, 2W]; the padded conv input,
    // conv output, norm, residual and a few copies at that size.
    let act = (u.mid_channels * f * h * w * 4) as u64 * 4;
    10 * act
}

/// Gemma streamed one layer at a time (float32 shards) beside the DiT: the
/// embedding table, one layer, and every hidden state of the prompt.
pub fn text_transient_bytes(cfg: &Ltx2Config) -> u64 {
    let (hidden, inter, layers, vocab) = match &cfg.gemma4 {
        Some(g) => (
            g.hidden_size,
            g.intermediate_size,
            g.num_hidden_layers,
            g.vocab_size,
        ),
        None => {
            let g = &cfg.text_encoder;
            (
                g.hidden_size,
                g.intermediate_size,
                g.num_hidden_layers,
                g.vocab_size,
            )
        }
    };
    let layer = (3 * hidden * inter + 4 * hidden * 4096) as u64 * 4;
    let embed = (vocab * hidden) as u64 * 4;
    let states = ((layers + 1) * cfg.defaults.max_sequence_length * hidden) as u64 * 4;
    embed + layer + states
}

/// One phase of the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseBytes {
    pub phase: &'static str,
    /// Models held on the device during the phase.
    pub weights: u64,
    /// Activations, caches and buffers of the phase.
    pub working: u64,
}

impl PhaseBytes {
    pub fn total(&self) -> u64 {
        self.weights + self.working + RUNTIME_ALLOWANCE
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPlan {
    pub phases: Vec<PhaseBytes>,
}

impl MemoryPlan {
    pub fn peak(&self) -> u64 {
        self.phases.iter().map(PhaseBytes::total).max().unwrap_or(0)
    }

    pub fn phase(&self, name: &str) -> Option<&PhaseBytes> {
        self.phases.iter().find(|p| p.phase == name)
    }
}

/// Inputs of [`plan_distilled_two_stage`].
#[derive(Debug, Clone, Copy)]
pub struct PlanOptions {
    pub ffn: FeedForwardChunking,
    pub score_budget_elems: usize,
    /// 2 for bf16 linears (`--mode fast`).
    pub weight_bytes: u64,
    /// Resident blocks, or streamed ones with the video VAE loaded only
    /// around the calls that read it (the upsample's latent statistics and
    /// the decode), as the RTX 5090 profile runs (`--offload cpu`).
    pub dit: DitPlacement,
}

impl PlanOptions {
    /// The 32 GiB profile: streamed blocks, decoders only while decoding.
    pub fn streamed() -> Self {
        Self {
            dit: DitPlacement::STREAMED,
            ..Self::default()
        }
    }
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            ffn: FeedForwardChunking::RTX5090,
            score_budget_elems: DENSE_SCORE_BUDGET_ELEMS,
            weight_bytes: 2,
            dit: DitPlacement::Resident,
        }
    }
}

/// Every phase of the Rust distilled two-stage pipeline at `height`×`width`,
/// `num_frames` at `frame_rate`: text beside the loaded DiT, stage 1 at half
/// resolution, the upsampler loaded for its call only, stage 2, then the DiT
/// dropped before the VAE decodes (tiled, as the reference's `AUTO_TILING`).
pub fn plan_distilled_two_stage(
    cfg: &Ltx2Config,
    height: usize,
    width: usize,
    num_frames: usize,
    frame_rate: f64,
    opts: PlanOptions,
) -> MemoryPlan {
    let t = &cfg.transformer;
    let dit = opts.dit.dit_bytes(t, opts.weight_bytes);
    let vae = vae_decoder_bytes(&cfg.vae);
    // Streamed: the video VAE is loaded for the upsample and the decode only.
    let vae_held = if opts.dit.is_streamed() { 0 } else { vae };
    let upsampler = cfg.latent_upsampler.as_ref();
    let text_tokens = cfg.defaults.max_sequence_length;
    let audio = t.audio_tokens(num_frames, frame_rate);
    let grid1 = t.latent_grid(num_frames, height / 2, width / 2);
    let grid2 = t.latent_grid(num_frames, height, width);
    let tokens = |g: [usize; 3]| g[0] * g[1] * g[2];
    let forward = |s: usize| {
        dit_forward_resident_bytes(t, s, audio, text_tokens)
            + dit_forward_transient_bytes(t, s, text_tokens, opts.ffn, opts.score_budget_elems)
    };
    let mut phases = vec![PhaseBytes {
        phase: "text",
        weights: dit + vae_held,
        working: text_transient_bytes(cfg),
    }];
    phases.push(PhaseBytes {
        phase: "stage1",
        weights: dit + vae_held,
        working: forward(tokens(grid1)),
    });
    if let Some(u) = upsampler {
        phases.push(PhaseBytes {
            phase: "upsample",
            weights: dit + vae + latent_upsampler_bytes(u),
            working: upsample_transient_bytes(u, grid1),
        });
    }
    phases.push(PhaseBytes {
        phase: "stage2",
        weights: dit + vae_held,
        working: forward(tokens(grid2)),
    });
    phases.push(PhaseBytes {
        phase: "decode",
        weights: vae,
        working: TileSizeConfig::conv_auto(height, width)
            .ok()
            .and_then(|tiles| vae_tiled_decode_bytes(&cfg.vae, grid2, &tiles))
            .unwrap_or_else(|| vae_decode_transient_bytes(&cfg.vae, grid2[1], grid2[2]))
            + (tokens(grid2) * t.in_channels * 4 * 3) as u64,
    });
    MemoryPlan { phases }
}

#[cfg(test)]
mod tests {
    use super::super::config::ltx2_5_22b_distilled;
    use super::*;

    fn gib(b: u64) -> f64 {
        b as f64 / GIB as f64
    }

    #[test]
    fn ffn_spans_match_torch_split() {
        let c = FeedForwardChunking::RTX5090;
        // Below the threshold: one span, the module's own forward.
        assert_eq!(c.spans(65_535), vec![(0, 65_535)]);
        assert_eq!(c.spans(32_640), vec![(0, 32_640)]);
        // 130 560 = 7 · 16 384 + 15 872.
        let spans = c.spans(130_560);
        assert_eq!(spans.len(), 8);
        assert!(spans[..7].iter().all(|&(_, len)| len == 16_384));
        assert_eq!(spans[7], (7 * 16_384, 15_872));
        assert_eq!(spans.iter().map(|s| s.1).sum::<usize>(), 130_560);
        assert_eq!(c.spans(65_536).len(), 4);
        assert_eq!(FeedForwardChunking::OFF.spans(130_560), vec![(0, 130_560)]);
    }

    #[test]
    fn workloads_match_the_run_script() {
        let cfg = ltx2_5_22b_distilled();
        let w = Rtx5090Workload::DEFAULT;
        assert_eq!(w.name(), "4k5s");
        assert_eq!(
            cfg.transformer
                .video_tokens(w.num_frames(), w.height(), w.width()),
            w.script_tokens()
        );
        // Stage 1 runs at half resolution: below the FFN chunking threshold.
        let s1 = cfg
            .transformer
            .video_tokens(w.num_frames(), w.height() / 2, w.width() / 2);
        assert_eq!(s1, 32_640);
        assert!(s1 < FeedForwardChunking::RTX5090.min_tokens);
        let hd = Rtx5090Workload::from_name("1080p20s").unwrap();
        assert_eq!(
            (hd.width(), hd.height(), hd.num_frames()),
            (1920, 1088, 481)
        );
        assert!(Rtx5090Workload::from_name("8k").is_none());
        // Both workloads need the two-stage multiple of 64.
        for w in [Rtx5090Workload::Uhd5s, Rtx5090Workload::Fhd20s] {
            assert_eq!(w.width() % 64, 0);
            assert_eq!(w.height() % 64, 0);
            assert_eq!(w.num_frames() % 8, 1);
        }
    }

    #[test]
    fn dit_weights_are_about_twenty_billion_bf16_parameters() {
        let cfg = ltx2_5_22b_distilled();
        let bytes = dit_weight_bytes(&cfg.transformer, 2);
        // 48 blocks of ~0.39 B linear parameters plus the globals.
        assert!(
            (35.0..42.0).contains(&gib(bytes)),
            "dit {:.2} GiB",
            gib(bytes)
        );
    }

    #[test]
    fn distilled_4k5s_two_stage_fits_under_90_gib() {
        let cfg = ltx2_5_22b_distilled();
        let w = Rtx5090Workload::Uhd5s;
        let plan = plan_distilled_two_stage(
            &cfg,
            w.height(),
            w.width(),
            w.num_frames(),
            w.frame_rate(),
            PlanOptions::default(),
        );
        for p in &plan.phases {
            eprintln!(
                "{:>8}: weights {:6.2} GiB, working {:6.2} GiB, total {:6.2} GiB",
                p.phase,
                gib(p.weights),
                gib(p.working),
                gib(p.total())
            );
        }
        let peak = plan.peak();
        assert!(peak < 90 * GIB, "planned peak {:.2} GiB", gib(peak));
        // Stage 2 is the binding phase, and the decode runs without the DiT.
        let stage2 = plan.phase("stage2").unwrap().total();
        assert_eq!(peak, stage2);
        assert!(plan.phase("decode").unwrap().total() < stage2 / 2);
    }

    /// The RTX 5090 (32 GiB) profile: streamed blocks keep every phase of
    /// both reference workloads under 30 GiB, while the resident DiT alone
    /// would not fit.
    #[test]
    fn streamed_two_stage_fits_a_32_gib_card() {
        let cfg = ltx2_5_22b_distilled();
        for w in [Rtx5090Workload::Uhd5s, Rtx5090Workload::Fhd20s] {
            let plan = |opts| {
                plan_distilled_two_stage(
                    &cfg,
                    w.height(),
                    w.width(),
                    w.num_frames(),
                    w.frame_rate(),
                    opts,
                )
            };
            let streamed = plan(PlanOptions::streamed());
            for p in &streamed.phases {
                eprintln!(
                    "{} streamed {:>8}: weights {:6.2} GiB, working {:6.2} GiB, total {:6.2} GiB",
                    w.name(),
                    p.phase,
                    gib(p.weights),
                    gib(p.working),
                    gib(p.total())
                );
            }
            let peak = streamed.peak();
            assert!(
                peak < 30 * GIB,
                "{}: streamed peak {:.2} GiB",
                w.name(),
                gib(peak)
            );
            let resident = plan(PlanOptions::default());
            assert!(resident.peak() > 32 * GIB, "{}", w.name());
            // Two block slots instead of 48 blocks.
            let t = &cfg.transformer;
            assert!(
                DitPlacement::STREAMED.dit_bytes(t, 2) < 3 * GIB,
                "{:.2} GiB",
                gib(DitPlacement::STREAMED.dit_bytes(t, 2))
            );
        }
    }

    #[test]
    fn ffn_chunking_bounds_the_stage2_feed_forward() {
        let cfg = ltx2_5_22b_distilled();
        let t = &cfg.transformer;
        let s = 130_560;
        let x = (s * t.inner_dim() * 4) as u64;
        let b = DENSE_SCORE_BUDGET_ELEMS;
        let chunked = dit_forward_transient_bytes(t, s, 1024, FeedForwardChunking::RTX5090, b);
        let whole = dit_forward_transient_bytes(t, s, 1024, FeedForwardChunking::OFF, b);
        // Unchunked, the [S, 16384] float32 up-projection (4x) and its bf16
        // copy dominate the forward; chunked, attention does.
        assert!(whole >= 9 * x, "{:.2} GiB", gib(whole));
        assert!(chunked < whole);
    }

    #[test]
    fn score_chunk_is_bounded_at_130k_tokens() {
        let cfg = ltx2_5_22b_distilled();
        let t = &cfg.transformer;
        let s = 130_560;
        let x = (s * t.inner_dim() * 4) as u64;
        let bounded = dit_forward_transient_bytes(
            t,
            s,
            1024,
            FeedForwardChunking::RTX5090,
            DENSE_SCORE_BUDGET_ELEMS,
        );
        // Scores + bf16 probabilities of one chunk: at most 1.5 GiB over the
        // self-attention activations, never the 2 TB of a full 130k² matrix.
        assert!(bounded <= x * 13 / 2 + 3 * GIB / 2);
    }
}
