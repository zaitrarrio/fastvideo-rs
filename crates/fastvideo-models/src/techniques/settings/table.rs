//! The declarative list of every `FASTVIDEO_*` setting the workspace reads.
//!
//! One row per name: who reads it ([`Reader`]), its type and parse rule
//! ([`Parse`] plus the `rule` text), its default ([`DefaultValue`]), the
//! family and seam it belongs to, when it is read ([`Timing`]) and what it
//! does. `docs/scope.md`'s settings table is rendered from these rows
//! ([`super::render_markdown`]); `workspace_check` fails when a source file
//! reads or names a `FASTVIDEO_*` name that is not a row, or reads one outside
//! [`super`]. Adding a setting means adding its row here first.
//!
//! The rows describe the readers as they are; they do not drive them. A row
//! whose parse is one of the shared rules ([`Parse::Flag`], [`Parse::Usize`],
//! [`Parse::F32`], [`Parse::F64`], [`Parse::Keyword`]) is read through the
//! runtime's typed getters, which call [`super::parse`], and the off-identity
//! tests below check its documented default against that very code. A
//! [`Parse::Text`] row is parsed at its call site; `rule` says how.

/// Who reads a setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reader {
    /// [`super::var`] / [`super::is_set`]: the environment, then the technique
    /// profile's table (a technique, a `[kernels]` entry or a raw `[env]`
    /// entry), then the caller's default.
    Layered,
    /// [`super::env_var`] / [`super::env_var_os`]: the process environment only
    /// (a profile's `[env]` entry for it has no effect).
    Env,
    /// Declared (a parser and a `const` name exist) but nothing reads it yet.
    Reserved,
    /// No longer read: its reader was retired, and only a message that
    /// refuses the old usage still names it.
    Retired,
}

/// The value's type and parse rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Parse {
    /// A boolean flag, [`super::parse::flag`]: a default-on flag is off only
    /// for `0` / `false` / `off`, a default-off flag on only for `1` / `true` /
    /// `on` (trimmed, any case).
    Flag,
    /// [`super::parse::usize_or`]: not trimmed; unparsable = the default.
    Usize,
    /// [`super::parse::f32_or`]: not trimmed; unparsable = the default.
    F32,
    /// [`super::parse::f64_or`]: trimmed; unparsable = the default.
    F64,
    /// [`super::parse::lower_or`]: lower-cased, not trimmed.
    Keyword,
    /// The raw value, parsed at the call site as the row's `rule` says.
    Text,
}

/// What an unset setting means.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DefaultValue {
    /// No value: the setting's behaviour is off or absent.
    Unset,
    /// The value an unset setting behaves as. Setting it explicitly changes
    /// nothing (the off-identity invariant, tested below).
    Value(&'static str),
    /// Decided at run time (by the device, the model, the request or another
    /// setting); the text says how.
    Computed(&'static str),
}

/// The model family (or runtime layer) a setting belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Family {
    /// The shared CUDA runtime (`fastvideo-cudarc::rt`) and every family.
    Runtime,
    /// Weight loading (`fastvideo-loader`).
    Loader,
    /// Binaries and test harnesses (`fastvideo`, `fv-gpucheck`, tests).
    Harness,
    /// Shared LLM text encoders (`fastvideo-cudarc::llm`).
    TextEncoder,
    Wan,
    H3,
    Ltx2,
    Cosmos,
    Cosmos3,
    Hunyuan15,
    Lingbot,
    Flux,
    Flux2,
    SanaVideo,
    MmAudio,
}

impl Family {
    pub fn label(self) -> &'static str {
        match self {
            Family::Runtime => "runtime",
            Family::Loader => "loader",
            Family::Harness => "harness",
            Family::TextEncoder => "text encoders",
            Family::Wan => "Wan",
            Family::H3 => "H3",
            Family::Ltx2 => "LTX-2",
            Family::Cosmos => "Cosmos",
            Family::Cosmos3 => "Cosmos3",
            Family::Hunyuan15 => "HunyuanVideo 1.5",
            Family::Lingbot => "LingBot",
            Family::Flux => "FLUX.1",
            Family::Flux2 => "FLUX.2",
            Family::SanaVideo => "SANA-Video",
            Family::MmAudio => "MMAudio",
        }
    }
}

/// When a setting is read, which is when a change to it takes effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Timing {
    /// Once, by a binary at start-up.
    Start,
    /// On first use, then cached for the life of the process.
    Process,
    /// While a model loads (each load reads it again).
    Load,
    /// Once per generate (request).
    Request,
    /// On every call of its reader (per op, step or launch).
    Call,
}

impl Timing {
    pub fn label(self) -> &'static str {
        match self {
            Timing::Start => "at start-up",
            Timing::Process => "once per process",
            Timing::Load => "per load",
            Timing::Request => "per generate",
            Timing::Call => "per call",
        }
    }
}

/// One `FASTVIDEO_*` setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Setting {
    pub name: &'static str,
    pub reader: Reader,
    pub parse: Parse,
    /// The parse rule of a [`Parse::Text`] row; a note on the others.
    pub rule: &'static str,
    pub default: DefaultValue,
    pub family: Family,
    /// The technique seam (`docs/techniques.md`) or the subsystem it tunes.
    pub seam: &'static str,
    pub timing: Timing,
    pub doc: &'static str,
}

#[allow(clippy::too_many_arguments)]
const fn row(
    name: &'static str,
    reader: Reader,
    parse: Parse,
    rule: &'static str,
    default: DefaultValue,
    family: Family,
    seam: &'static str,
    timing: Timing,
    doc: &'static str,
) -> Setting {
    Setting {
        name,
        reader,
        parse,
        rule,
        default,
        family,
        seam,
        timing,
        doc,
    }
}

/// The row of `name`, if it is a declared setting.
pub fn lookup(name: &str) -> Option<&'static Setting> {
    SETTINGS
        .binary_search_by(|s| s.name.cmp(name))
        .ok()
        .map(|i| &SETTINGS[i])
}

/// Every `FASTVIDEO_*` setting, sorted by name.
pub const SETTINGS: &[Setting] = rows::ROWS;

mod rows {
    use super::DefaultValue::{Computed, Unset, Value};
    use super::Family::*;
    use super::Parse::*;
    use super::Reader::*;
    use super::Timing::*;
    use super::{row, Setting};

    #[rustfmt::skip]
    pub(super) const ROWS: &[Setting] = &[
        row("FASTVIDEO_ARTIFACT_DIR", Env, Text, "a directory", Computed("the system temp directory"), Harness, "test", Call,
            "Root of the artifact directories `fastvideo-core`'s GPU tests write."),
        row("FASTVIDEO_ATTENTION_BACKEND", Layered, Keyword, "", Value(""), Runtime, "attention_backend", Call,
            "`SLA_ATTN` / `sla` / `sagesla` / `SAGE_SLA_ATTN` (any case) select sparse-linear attention."),
        row("FASTVIDEO_ATTN_FP8", Layered, Text, "`0`, `1` / `all`, `dense` or `vsa` (H3 technique adapter)", Computed("the recipe's / profile's"), H3, "attention", Load,
            "FP8 attention ops (opt-in, lossy; H3)."),
        row("FASTVIDEO_ATTN_PROBS_BF16", Layered, Flag, "", Value("1"), Runtime, "activation_precision", Process,
            "Attention probabilities in bf16 (default on); `0` keeps them f32."),
        row("FASTVIDEO_ATTN_SAGE", Layered, Text, "`2` / `on` force, `0` / `off` disable, empty = unset (trimmed, any case); any other value turns it off with a one-time log line", Unset, Runtime, "attention_backend", Process,
            "SageAttention-2 (INT8 QK + FP8 PV) for dense attention; unset leaves the dispatch rule."),
        row("FASTVIDEO_ATTN_SAGE_MIN_SEQ", Layered, Usize, "", Value("6144"), Runtime, "attention_backend", Process,
            "Shortest sequence SageAttention runs on."),
        row("FASTVIDEO_BF16", Layered, Flag, "", Value("1"), Runtime, "ffn_precision", Process,
            "cuBLAS tf32/bf16 compute. On unless set to `0`."),
        row("FASTVIDEO_BF16_ACT", Layered, Flag, "set: a default-on flag", Computed("on for H3, LTX-2 and Wan on a GPU, f32 on the CPU"), Runtime, "activation_precision", Process,
            "DiT activations (and residual) as bf16, as the reference runs them. On by default for H3, LTX-2 and Wan on a GPU (Wan's UMT5 and VAE stay f32, FastVideo's fp32 precisions); `0` restores f32, `1` forces bf16 for every model. CPU runs keep f32. FastWan 1.3B 480x832x81 on RTX PRO 6000: denoise 2.50 → 1.96 s, LPIPS 0.107 vs f32 (see [ports/wan.md](ports/wan.md))."),
        row("FASTVIDEO_CACHE", Env, Text, "a directory (empty = unset)", Computed("`$XDG_CACHE_HOME/fastvideo`, else `~/.cache/fastvideo`"), Ltx2, "cache", Load,
            "Root of the LTX text-conditioning cache (`<root>/ltx2-text`)."),
        row("FASTVIDEO_CONV3D", Layered, Keyword, "", Value("auto"), Runtime, "kernel_choice", Process,
            "3D convolution kernel (profile `[kernels] conv3d`)."),
        row("FASTVIDEO_CONV3D_TIMED", Layered, Flag, "", Value("0"), Runtime, "debug", Process,
            "Time every 3D convolution and log it."),
        row("FASTVIDEO_CONV_BWD_DATA_NONDETERMINISTIC", Layered, Flag, "", Value("0"), Runtime, "debug", Call,
            "Allow cuDNN's non-deterministic backward-data convolution algorithms."),
        row("FASTVIDEO_COSMOS3_GEN", Env, Text, "`auto` / empty, `resident`, `parked` / `park` (trimmed); anything else is an error", Value("auto"), Cosmos3, "residency", Load,
            "Cosmos3 gen tower placement: `auto` parks it when the free memory does not cover it plus the decode headroom."),
        row("FASTVIDEO_COSMOS3_MLP_CHUNK", Layered, Usize, "at least 1", Value("8192"), Cosmos3, "memory", Call,
            "Cosmos3 MLP rows per chunk."),
        row("FASTVIDEO_COSMOS3_OFFICIAL", Env, Text, "`1` or `official` (trimmed, any case)", Value("0"), Cosmos, "canvas", Request,
            "Cosmos3 canvas: the published 1280×720 / 189f / 35-step / guidance-6 contract."),
        row("FASTVIDEO_COSMOS3_UND", Env, Text, "`resident` (trimmed) keeps it resident", Unset, Cosmos3, "residency", Load,
            "Cosmos3 understanding tower kept resident."),
        row("FASTVIDEO_COSMOS_SOL", Env, Text, "`1` or `teacache` (trimmed, any case)", Value("0"), Cosmos, "step_output", Request,
            "Cosmos TeaCache."),
        row("FASTVIDEO_CUDNN_SDPA_CFG", Layered, Usize, "", Value("0"), Runtime, "kernel_choice", Call,
            "cuDNN SDPA engine config index (0: the heuristic's first)."),
        row("FASTVIDEO_CUDNN_SDPA_GRAPH", Layered, Keyword, "", Value("auto"), Runtime, "kernel_choice", Call,
            "cuDNN SDPA graph capture: `auto`, or forced on / off."),
        row("FASTVIDEO_CUDNN_SDPA_NONDETERMINISTIC", Layered, Flag, "", Value("0"), Runtime, "debug", Process,
            "Allow cuDNN SDPA's non-deterministic algorithms."),
        row("FASTVIDEO_DC_SM103", Layered, Flag, "", Value("0"), Runtime, "kernel_choice", Call,
            "Build the data-center attention kernels for `compute_103a` on sm_103."),
        row("FASTVIDEO_DEVICE_BUDGET_GIB", Env, Text, "GiB, fractions allowed (trimmed); empty, `0` or `off` = no budget; an unparsable value is logged and ignored", Unset, Runtime, "memory", Process,
            "Plan and run as if the card had this many GiB. A phase that peaks above it fails the run."),
        row("FASTVIDEO_DEVICE_STATS", Layered, Flag, "", Value("0"), Wan, "trace", Request,
            "Log device memory statistics after a Wan generate."),
        row("FASTVIDEO_DIGEST", Layered, Flag, "", Value("0"), Runtime, "debug", Process,
            "Log a digest of every dumped tensor."),
        row("FASTVIDEO_DIGEST_WEIGHTS", Layered, Flag, "", Value("0"), Runtime, "debug", Process,
            "Log a digest of every loaded weight (with `FASTVIDEO_DIGEST`)."),
        row("FASTVIDEO_DIT_OFFLOAD", Layered, Text, "`auto`, `resident`, `streamed` (an unknown value is an error)", Value("auto"), Runtime, "residency", Load,
            "LTX-2 / H3 DiT blocks: `auto` (default), `resident`, or `streamed`. Wan DiTs: `streamed` streams the blocks layer by layer (any other value: resident, or the A14B plan of `FASTVIDEO_WAN_MOE`)."),
        row("FASTVIDEO_DIT_OFFLOAD_LOOKAHEAD", Layered, Usize, "clamped to 1..=8", Value("1"), Runtime, "residency", Load,
            "Streamed DiT blocks copied ahead of the one computing."),
        row("FASTVIDEO_DUMP_DIR", Env, Text, "a directory (empty = unset)", Unset, Runtime, "debug", Process,
            "Dump named intermediate tensors here (parity runs)."),
        row("FASTVIDEO_DUMP_OPS", Env, Text, "block indices whose ops are dumped", Value("0"), Runtime, "debug", Process,
            "Blocks whose per-op tensors are dumped."),
        row("FASTVIDEO_FFMPEG", Env, Text, "a program", Value("ffmpeg"), Ltx2, "io", Call,
            "ffmpeg binary for the LTX image-conditioning recompression."),
        row("FASTVIDEO_FLASH_KERNEL", Layered, Keyword, "", Value("auto"), Runtime, "kernel_choice", Call,
            "Dense attention kernel (profile `[kernels] dense_attention`)."),
        row("FASTVIDEO_FLUX1_DUMMY_TEXT", Env, Text, "`1`, `true` or `TRUE` (exact)", Value("0"), Flux, "text_encoder", Load,
            "FLUX.1: zero text embeddings instead of the T5/CLIP encoders (smoke runs)."),
        row("FASTVIDEO_FLUX1_TEXT_LEN", Env, Text, "a `usize` (not trimmed), at least 1; unparsable = default", Computed("the model's T5 length"), Flux, "text_encoder", Load,
            "FLUX.1 T5 sequence length."),
        row("FASTVIDEO_FLUX2_DUMMY_TEXT", Env, Text, "`1`, `true` or `TRUE` (exact)", Value("0"), Flux2, "text_encoder", Load,
            "FLUX.2: zero text embeddings instead of the text encoder (smoke runs)."),
        row("FASTVIDEO_FLUX2_TEXT_LEN", Env, Text, "a `usize` (not trimmed), at least 1; unparsable = default", Computed("the model's text length"), Flux2, "text_encoder", Load,
            "FLUX.2 text sequence length."),
        row("FASTVIDEO_FP8", Layered, Flag, "", Value("0"), Runtime, "ffn_precision", Process,
            "FastVideo W8A8 tensorwise FP8 recipe on every linear (Cosmos3 reads it per load and per generate)."),
        row("FASTVIDEO_GPU_SMOKE", Env, Text, "`1` runs the test", Unset, Harness, "test", Call,
            "Opt in to `fastvideo-core`'s 1.3B CUDA smoke test."),
        row("FASTVIDEO_GPU_TRACE", Layered, Flag, "", Value("0"), Runtime, "trace", Process,
            "CUPTI trace of one denoise step (kernel time by category, GPU idle)."),
        row("FASTVIDEO_GPU_TRACE_DECODE", Layered, Flag, "", Value("0"), Runtime, "trace", Process,
            "CUPTI trace of a whole video decode (kernel time by category, GPU idle)."),
        row("FASTVIDEO_GPU_TRACE_STEP", Layered, Usize, "", Value("1"), Runtime, "trace", Call,
            "Denoise step `FASTVIDEO_GPU_TRACE` traces."),
        row("FASTVIDEO_GPU_TRACE_TOP", Layered, Usize, "", Value("25"), Runtime, "trace", Call,
            "Kernels a `FASTVIDEO_GPU_TRACE` report lists (`top_kernels`, by time); the report only."),
        row("FASTVIDEO_H2D_TRACE_BYTES", Env, Text, "a `usize` (not trimmed); unset or unparsable = off", Unset, Runtime, "trace", Process,
            "Log host-to-device copies of at least this many bytes."),
        row("FASTVIDEO_H2D_TRACE_SKIP", Layered, Usize, "", Value("0"), Runtime, "trace", Call,
            "Host-to-device copies skipped before `FASTVIDEO_H2D_TRACE_BYTES` logs."),
        row("FASTVIDEO_H3_AFFINE", Layered, Keyword, "bit list", Value(""), H3, "kernel_fusion", Process,
            "H3 affine-fusion bits (`fv-gpucheck --h3-affine`)."),
        row("FASTVIDEO_H3_DIT_TREE", Env, Text, "`auto` / empty, `off` / `0` / `false` / `no`, `require` / `1` / `on` / `true` / `yes` (trimmed, any case); anything else is an error", Value("auto"), H3, "weights", Load,
            "Pre-quantized H3 DiT tree: use it when valid, never, or require it."),
        row("FASTVIDEO_H3_DIT_TREE_DIR", Env, Text, "a directory (trimmed; empty = default)", Computed("`<root>/transformer_prequant_<recipe>_<quant>`"), H3, "weights", Load,
            "Where the pre-quantized H3 DiT tree lives."),
        row("FASTVIDEO_H3_FFN_FP8", Layered, Flag, "", Value("0"), H3, "ffn_precision", Load,
            "Retired flag: set, the H3 load refuses and names `FASTVIDEO_H3_QUANT`."),
        row("FASTVIDEO_H3_FUSE", Layered, Flag, "", Value("1"), H3, "kernel_fusion", Process,
            "H3 fused MXFP8 activation and residual-plus-norm kernels (bit-identical); `0` runs the separate ops."),
        row("FASTVIDEO_H3_GLUE", Layered, Flag, "", Value("1"), H3, "kernel_fusion", Process,
            "H3 glue kernels between the fused ops (bit-identical); `0` runs the separate ops."),
        row("FASTVIDEO_H3_GRAPHS", Layered, Flag, "", Value("0"), H3, "cuda_graphs", Call,
            "CUDA-graph capture of the H3 denoise step."),
        row("FASTVIDEO_H3_LORA_DEVICE", Layered, Flag, "", Value("1"), H3, "lora", Process,
            "Fuse H3 LoRAs on the device."),
        row("FASTVIDEO_H3_LORA_VERIFY", Layered, Flag, "", Value("0"), H3, "debug", Process,
            "Check the device LoRA fusion against the host."),
        row("FASTVIDEO_H3_LTX_ADAPTER", Env, Text, "a path", Computed("found beside the H3 weights"), H3, "weights", Load,
            "Spark bridge LTX adapter checkpoint."),
        row("FASTVIDEO_H3_NVFP4_ACT", Layered, Text, "`static` / empty or `dynamic` (trimmed); anything else is an error", Value("static"), H3, "ffn_precision", Load,
            "ModelOpt NVFP4 DiT: activation scales from the checkpoint or measured per call."),
        row("FASTVIDEO_H3_NVFP4_ATTN", Layered, Text, "`1` / `on` / `true` / `nvfp4`, or `0` / `off` / `false` / `bf16` / empty (trimmed); anything else is an error", Value("0"), H3, "ffn_precision", Load,
            "ModelOpt NVFP4 DiT: attention linears in NVFP4 too."),
        row("FASTVIDEO_H3_QUANT", Layered, Text, "`mxfp8`, `w8a8` or `off` (an unknown value is an error)", Computed("`mxfp8` on sm_100+ GPUs, the recipe's scoped default (VDN-H3: bf16)"), H3, "ffn_precision", Load,
            "H3 FP8 linear recipe: `mxfp8` (Sol-H3, the default on sm_100+ GPUs), `w8a8` (FastVideo) or `off` (bf16)."),
        row("FASTVIDEO_H3_SOL_ATTN", Layered, Text, "`off`, `rtx`, `engine`, `spark` (H3 technique adapter)", Computed("the recipe's route"), H3, "attention_backend", Load,
            "H3 Sol-Attn route: `spark` or `rtx`."),
        row("FASTVIDEO_H3_SOL_CACHE", Layered, Text, "`teacache` or `0` (H3 technique adapter)", Unset, H3, "step_output", Load,
            "H3 RTX TeaCache."),
        row("FASTVIDEO_H3_STEP_SYNC", Layered, Flag, "", Value("0"), H3, "debug", Call,
            "Synchronize after every H3 denoise step (timing)."),
        row("FASTVIDEO_H3_UPSCALER", Env, Text, "a file", Computed("found beside the H3 weights"), H3, "weights", Load,
            "Spark bridge upscaler checkpoint."),
        row("FASTVIDEO_H3_VAE_AB", Layered, Flag, "", Value("0"), H3, "debug", Process,
            "Re-decode with the bf16 H3 VAE and log the difference."),
        row("FASTVIDEO_H3_VAE_CHECK", Layered, Flag, "", Value("0"), H3, "debug", Process,
            "Decode the first H3 VAE tile batch both ways and log the error."),
        row("FASTVIDEO_H3_VAE_FUSED", Layered, Flag, "", Value("1"), H3, "kernel_fusion", Process,
            "H3 VAE fused kernels."),
        row("FASTVIDEO_H3_VAE_GEMM", Layered, Keyword, "", Value("cublas"), H3, "kernel_choice", Process,
            "H3 VAE GEMM backend."),
        row("FASTVIDEO_H3_VAE_MXFP8", Layered, Text, "`0` or `1` (an unknown value is an error)", Computed("MXFP8 on sm_100+ for the measured decoder depths"), H3, "video_decoder", Load,
            "H3 VAE decoder linears as MXFP8."),
        row("FASTVIDEO_H3_VAE_MXFP8_ALGO", Layered, Usize, "", Value("0"), H3, "kernel_choice", Call,
            "H3 VAE MXFP8 GEMM algorithm index."),
        row("FASTVIDEO_H3_VAE_NVFP4", Layered, Text, "`0` or `1` (an unknown value is an error)", Computed("the tree's: NVFP4 when its `vae/` carries NVFP4 linears"), H3, "video_decoder", Load,
            "H3 VAE decoder linears as published ModelOpt NVFP4; `0` decodes with h3-base's bf16 `vae/`."),
        row("FASTVIDEO_H3_VAE_NVFP4_ACT", Layered, Text, "`static` / empty or `dynamic` (an unknown value is an error)", Value("static"), H3, "video_decoder", Load,
            "H3 NVFP4 VAE activation scales: calibrated `input_scale` or measured per call."),
        row("FASTVIDEO_H3_VAE_NVFP4_REF", Layered, Text, "a `vae/` directory", Unset, H3, "debug", Load,
            "With `FASTVIDEO_H3_VAE_CHECK`, also decode through this bf16 decoder."),
        row("FASTVIDEO_H3_VAE_SDPA", Layered, Keyword, "", Value("auto"), H3, "kernel_choice", Process,
            "H3 VAE attention kernel."),
        row("FASTVIDEO_H3_VAE_TILE_BATCH", Layered, Usize, "at least 1", Value("8"), H3, "memory", Load,
            "H3 VAE tiles decoded per batch."),
        row("FASTVIDEO_H3_VAE_VEC", Layered, Flag, "", Value("1"), H3, "kernel_choice", Process,
            "H3 VAE vectorized kernels."),
        row("FASTVIDEO_H3_VDN", Env, Text, "a path", Unset, H3, "weights", Load,
            "VDN-H3 checkpoint when `--vdn` is not given."),
        row("FASTVIDEO_H3_VDN_SHARED", Env, Text, "a path (empty = unset)", Unset, H3, "weights", Load,
            "VDN-H3 shared checkpoint when the option is not given."),
        row("FASTVIDEO_H3_VSA_FP4", Layered, Flag, "set: a default-on flag", Computed("on where the NVFP4 tier has it"), H3, "attention", Load,
            "H3 VSA attention in NVFP4."),
        row("FASTVIDEO_H3_VSA_FP4_P", Layered, Keyword, "", Value("two-level"), H3, "attention", Call,
            "`single`: upstream's single-level P quantization (default two-level)."),
        row("FASTVIDEO_HUNYUAN15_OFFICIAL", Env, Text, "`1` or `official` (trimmed, any case)", Value("0"), Hunyuan15, "canvas", Request,
            "720p / 129f / 50-step canvas."),
        row("FASTVIDEO_HUNYUAN15_SOL", Env, Text, "`1` or `teacache` (trimmed, any case)", Value("0"), Hunyuan15, "step_output", Request,
            "Logs the 13B TeaCache gap and stays dense."),
        row("FASTVIDEO_INJECT_COND", Env, Text, "H3: on only for `1`; LTX-2: on unless `0` (not trimmed)", Computed("off for H3, on for LTX-2"), Runtime, "debug", Call,
            "With `FASTVIDEO_INJECT_DIR`: inject the reference's condition tensors."),
        row("FASTVIDEO_INJECT_DIR", Env, Text, "a directory (empty = unset)", Unset, Runtime, "debug", Process,
            "Inject reference tensors from this directory (parity runs)."),
        row("FASTVIDEO_INJECT_PIXELS", Env, Text, "on unless `0` (not trimmed)", Value("1"), Ltx2, "debug", Call,
            "With `FASTVIDEO_INJECT_DIR`: inject the reference's conditioning pixels."),
        row("FASTVIDEO_INJECT_STAGE2", Env, Text, "on unless `0` (not trimmed)", Value("1"), Ltx2, "debug", Call,
            "With `FASTVIDEO_INJECT_DIR`: inject the reference's stage-2 entry latents."),
        row("FASTVIDEO_INJECT_TEXT", Env, Text, "on unless `0` (not trimmed)", Value("1"), Runtime, "debug", Call,
            "With `FASTVIDEO_INJECT_DIR`: inject the reference's text embeddings."),
        row("FASTVIDEO_KERNELS", Layered, Keyword, "", Value("auto"), Runtime, "kernel_choice", Load,
            "`nvrtc` compiles the kernels with NVRTC instead of the embedded cubins."),
        row("FASTVIDEO_LINGBOT_MOE_CHUNK", Layered, Usize, "at least 1", Value("32768"), Lingbot, "memory", Call,
            "LingBot MoE tokens per chunk."),
        row("FASTVIDEO_LINGBOT_OFFICIAL", Env, Text, "`1` or `official` (trimmed, any case)", Value("0"), Lingbot, "canvas", Request,
            "832×480 / 121f / 40-step canvas."),
        row("FASTVIDEO_LINGBOT_REFINE", Env, Text, "`0` or `off` (exact) stop after the base stage", Unset, Lingbot, "canvas", Request,
            "LingBot 1080p refiner stage."),
        row("FASTVIDEO_LINGBOT_RESIDENCY", Env, Text, "`both` or `resident` (trimmed); anything else swaps", Value("swap"), Lingbot, "residency", Load,
            "LingBot experts: both resident or swapped."),
        row("FASTVIDEO_LINGBOT_ROUTER", Layered, Keyword, "", Value("device"), Lingbot, "kernel_choice", Process,
            "LingBot MoE router on the device or the host."),
        row("FASTVIDEO_LINGBOT_ROUTER_CHECK", Layered, Flag, "", Value("0"), Lingbot, "debug", Call,
            "Check the device router against the host."),
        row("FASTVIDEO_LINGBOT_SOL", Env, Text, "`0` / `off` / `baseline`, `1` / `sol` / `fullopt`, `cache` / `easycache`, `pisa`, or a comma list (trimmed, any case)", Value("baseline"), Lingbot, "step_output", Request,
            "LingBot EasyCache and refiner PISA."),
        row("FASTVIDEO_LLM_PREFETCH", Env, Text, "on unless `0` (not trimmed)", Value("1"), TextEncoder, "loader", Call,
            "Prefetch the next text-encoder layer while one computes."),
        row("FASTVIDEO_LOG", Env, Text, "`0` / `off` / `false` / `quiet` / `error`, `2` / `debug` / `trace` / `verbose`, else info (trimmed, any case)", Value("info"), Runtime, "log", Process,
            "Runtime log level."),
        row("FASTVIDEO_LTX2_AUDIO_VAE", Env, Text, "a file or directory", Computed("`audio_vae/` or `ltx-2.5-audio-vae-bf16.safetensors`"), Ltx2, "weights", Load,
            "Audio encoder file or directory."),
        row("FASTVIDEO_LTX2_DIT", Env, Text, "a path", Computed("`$FASTVIDEO_LTX2_WEIGHTS/transformer`"), H3, "weights", Request,
            "Override the Spark refiner's DiT path."),
        row("FASTVIDEO_LTX2_FBCACHE", Env, Text, "`1` or `fbcache` (trimmed, any case)", Value("0"), Ltx2, "step_output", Request,
            "LTX-2.5 stage-1 first-block cache."),
        row("FASTVIDEO_LTX2_HOST_ROPE", Layered, Flag, "", Value("0"), Ltx2, "debug", Call,
            "Compute the LTX-2 RoPE tables on the host."),
        row("FASTVIDEO_LTX2_HQ", Env, Text, "`1`, `hq` or `official` (trimmed, any case)", Value("0"), Ltx2, "canvas", Request,
            "2.3-base 15+3 HQ contract (`fv-gpucheck ltx2 gen --model-version 2.3 --hq` runs it on the dev DiT)."),
        row("FASTVIDEO_LTX2_IC_LORA", Env, Text, "a file (empty = unset)", Unset, Ltx2, "lora", Load,
            "IC-LoRA file when the request names none."),
        row("FASTVIDEO_LTX2_LORA", Env, Text, "a file", Computed("a known filename beside the weights"), Ltx2, "lora", Load,
            "Distilled LoRA file."),
        row("FASTVIDEO_LTX2_LORA_BASE", Env, Text, "`host` or `device` (trimmed; another value is an error)", Computed("the load option"), Ltx2, "residency", Load,
            "Where the LoRA base weights stay during the fuse."),
        row("FASTVIDEO_LTX2_MIDPOINT_PRUNE", Env, Text, "`1`, `feat_norm` or `prune` (trimmed, any case)", Value("0"), Ltx2, "token_set", Request,
            "LTX-2.3 stage-2 feature-norm prune."),
        row("FASTVIDEO_LTX2_SAVE_LATENTS", Env, Text, "a path prefix (empty = unset)", Unset, Ltx2, "debug", Request,
            "Save the decoded video latents to `<prefix>.f32` / `.shape` (for `ltx2 vae-bench --latents`)."),
        row("FASTVIDEO_LTX2_STAGE1_CACHE", Env, Text, "`1`, the preset name or `scsp` (trimmed, any case)", Value("0"), Ltx2, "step_output", Request,
            "LTX-2.3 stage-1 SCSP (res2s calls 16–28 of 29)."),
        row("FASTVIDEO_LTX2_TAE_WEIGHTS", Layered, Text, "a path (empty = unset)", Unset, Ltx2, "video_decoder", Load,
            "LTX tiny decoder (TAEHV) weights."),
        row("FASTVIDEO_LTX2_TEXT", Env, Text, "`resident`, `streamed` or `auto` (trimmed, any case; empty = unset); anything else is an error", Computed("the load option's text residency"), Ltx2, "residency", Load,
            "Gemma text encoder placement."),
        row("FASTVIDEO_LTX2_TEXT_FP8", Env, Text, "on only for `1` (trimmed)", Value("0"), Ltx2, "text_encoder", Load,
            "Gemma from the pre-quantized FP8 tree."),
        row("FASTVIDEO_LTX2_TEXT_WARM", Env, Text, "on unless `0` (trimmed)", Value("1"), Ltx2, "text_encoder", Load,
            "Open the Gemma weights ahead of the first encode."),
        row("FASTVIDEO_LTX2_UPSAMPLER", Env, Text, "a file or directory", Computed("the pack's upsampler"), Ltx2, "weights", Load,
            "Spatial-upsampler file or folder in place of the pack's (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors` for the 2.3 HQ cell)."),
        row("FASTVIDEO_LTX2_VAE_CHUNK", Layered, Usize, "", Value("8"), Ltx2, "memory", Call,
            "LTX-2 streaming VAE frames per chunk."),
        row("FASTVIDEO_LTX2_VAE_LATENT_CHUNK", Layered, Usize, "at least 1", Value("1"), Ltx2, "memory", Call,
            "LTX-2 VAE latent frames per decode chunk."),
        row("FASTVIDEO_LTX2_WEIGHTS", Env, Text, "a directory", Unset, H3, "weights", Request,
            "LTX-2.5 root for the Spark refiner. DiT defaults to `weights/transformer`."),
        row("FASTVIDEO_LTX_FUSE", Layered, Flag, "", Value("1"), Ltx2, "kernel_fusion", Process,
            "LTX block fusions (bit-identical); `0` runs the separate ops."),
        row("FASTVIDEO_LTX_OFFLOAD", Layered, Text, "`none` / `off` / `0` / empty, `cpu` / `on` / `1` (trimmed, any case); `disk` and others are errors", Value("none"), Ltx2, "residency", Load,
            "`cpu` gives sol-engine's `--offload cpu` placement: the DiT blocks stream from pinned host memory through 2 device slots, Gemma streams per layer, the connectors, upsampler and VAEs are on the device only around their calls. Byte-identical frames. `ltx2 gen --offload` overrides it. See [ports/ltx25.md](ports/ltx25.md#offload-placement-fastvideo_ltx_offloadcpu)."),
        row("FASTVIDEO_LTX_VAE_CHECK", Layered, Flag, "", Value("0"), Ltx2, "debug", Process,
            "Decode the process's first LTX VAE tile both ways and log rel_l2 and both times."),
        row("FASTVIDEO_LTX_VAE_CHUNK_FRAMES", Layered, Usize, "at least 1", Value("16"), Ltx2, "memory", Call,
            "Output frames per channels-last conv launch."),
        row("FASTVIDEO_LTX_VAE_CHUNK_MB", Layered, Usize, "at least 1", Value("1024"), Ltx2, "memory", Call,
            "MB per channels-last conv launch."),
        row("FASTVIDEO_LTX_VAE_CONV_ALGO", Layered, Keyword, "", Value("heuristic"), Ltx2, "kernel_choice", Call,
            "`tune`: time every cuDNN forward algorithm per conv shape (default: cuDNN's heuristic)."),
        row("FASTVIDEO_LTX_VAE_FAST", Layered, Flag, "", Value("1"), Ltx2, "video_decoder", Process,
            "LTX conv VAE tiled decode on the channels-last bf16 decoder (default on); `0` restores the f32 streaming decoder."),
        row("FASTVIDEO_MMAUDIO_INJECT", Env, Text, "comma-separated stage names", Unset, MmAudio, "debug", Call,
            "MMAudio stages whose reference tensors are injected."),
        row("FASTVIDEO_MMAUDIO_WEIGHTS", Env, Text, "a directory", Computed("`MMAUDIO_MODEL_PATH`, else under `FV_WEIGHTS`"), MmAudio, "weights", Load,
            "MMAudio weights root."),
        row("FASTVIDEO_NVFP4", Layered, Text, "`0` / `off` / `false` / `none` / `no`, `1` / `static_6` / `te` / ..., `static_4`, `mse` / `4o6` (any case); unknown = off", Value("0"), Runtime, "ffn_precision", Load,
            "W4A4 dequant (`1` → `static_6`; `mse` for FourOverSix). LTX-2: the video FFN on NVFP4 tensor cores (`static_6` / `static_4`)."),
        row("FASTVIDEO_NVFP4_COSMOS_STEPS", Reserved, Text, "`1` / `true` / `on` / `yes` (trimmed, any case)", Value("0"), Cosmos, "ffn_precision", Load,
            "Reserved: restrict NVFP4 to the middle Cosmos steps (the parser exists, no reader yet)."),
        row("FASTVIDEO_NVFP4_LTX_VIDEO_FFN", Reserved, Text, "`1` / `true` / `on` / `yes` (trimmed, any case)", Value("0"), Ltx2, "ffn_precision", Load,
            "Reserved: restrict NVFP4 to the LTX video FFN (the parser exists, no reader yet)."),
        row("FASTVIDEO_NVFP4_OXIDE_GEMM", Layered, Text, "`1` / `true` / `on` / `yes` (trimmed, any case)", Value("0"), Runtime, "kernel_choice", Call,
            "Tile-IR W4A4 GEMM (profile `[kernels] nvfp4_gemm`); off until it beats cuBLAS bf16."),
        row("FASTVIDEO_NVRTC_PTX", Layered, Flag, "", Value("0"), Runtime, "kernel_choice", Load,
            "Have NVRTC emit PTX instead of SASS."),
        row("FASTVIDEO_PISA_FIRST_ORDER", Layered, Keyword, "", Value("1"), Runtime, "attention", Call,
            "`0` drops PISA's first-order term."),
        row("FASTVIDEO_PISA_KERNEL", Layered, Keyword, "", Value("fused"), Runtime, "kernel_choice", Call,
            "`legacy` runs the unfused PISA kernel."),
        row("FASTVIDEO_PLUG_ROOT", Env, Text, "a directory", Computed("`<base_root>/../longlive-plug`"), Wan, "lora", Load,
            "Where LongLive-Plug adapters live."),
        row("FASTVIDEO_PNG", Env, Text, "`inline`, `off` / `0` / `none`, else deferred (trimmed, any case)", Value("deferred"), Runtime, "io", Request,
            "PNG frame writing: deferred to a writer thread, inline, or off."),
        row("FASTVIDEO_PNG_BUFFER_GIB", Env, Text, "GiB as `f64` (trimmed), negative = 0", Computed("half the available host memory, else 8 GiB"), Runtime, "io", Request,
            "Host RAM the deferred PNG frames may hold."),
        row("FASTVIDEO_PNG_SYNC", Layered, Flag, "", Value("1"), Runtime, "io", Request,
            "Wait for the PNG writer before returning."),
        row("FASTVIDEO_PREFETCH", Env, Text, "on unless `0` (trimmed)", Value("1"), Loader, "loader", Load,
            "Weight read-ahead pool."),
        row("FASTVIDEO_PREFETCH_CHUNK_MB", Env, Text, "a `usize` (trimmed), clamped to 1..=1024", Value("16"), Loader, "loader", Load,
            "Read-ahead chunk size."),
        row("FASTVIDEO_PREFETCH_MODE", Env, Text, "`cache` (trimmed) forces page-cache mode", Unset, Loader, "loader", Process,
            "Read-ahead into owned memory while `MemAvailable` allows, else the page cache."),
        row("FASTVIDEO_PREFETCH_THREADS", Env, Text, "a `usize` (trimmed)", Computed("16; `FASTVIDEO_STREAMING_PREFETCH_THREADS` while the weights stream"), Loader, "loader", Load,
            "Read-ahead threads."),
        row("FASTVIDEO_PREFETCH_WINDOW_GB", Env, Text, "GB as `f64` (trimmed)", Computed("the smaller of 24 GB and 40% of `MemAvailable`"), Loader, "loader", Load,
            "Read-ahead window."),
        row("FASTVIDEO_PROFILE", Layered, Flag, "", Value("0"), Runtime, "trace", Process,
            "Per-op timing statistics."),
        row("FASTVIDEO_RESIDENT", Layered, Flag, "", Value("1"), Runtime, "residency", Process,
            "Keep tensors on the device between ops (default on); `0` is refused with a live CUDA device."),
        row("FASTVIDEO_SANA_EASYCACHE", Env, Text, "an EasyCache threshold (a number)", Unset, SanaVideo, "step_output", Load,
            "SANA-Video EasyCache threshold."),
        row("FASTVIDEO_SANA_LINATTN_BF16", Env, Text, "a SANA-Video flag", Computed("the `FASTVIDEO_SANA_OPT` arm's"), SanaVideo, "activation_precision", Load,
            "SANA-Video linear attention in bf16."),
        row("FASTVIDEO_SANA_OPT", Env, Text, "`baseline` or `full`", Value("baseline"), SanaVideo, "step_output", Load,
            "SANA-Video optimization arm."),
        row("FASTVIDEO_SANA_QKV_MERGE", Env, Text, "a SANA-Video flag", Computed("the `FASTVIDEO_SANA_OPT` arm's"), SanaVideo, "kernel_fusion", Load,
            "SANA-Video merged QKV projection."),
        row("FASTVIDEO_SAVE_MP4", Env, Text, "`1` or `true` (any case, not trimmed)", Value("0"), Harness, "io", Request,
            "Mux frames with ffmpeg."),
        row("FASTVIDEO_SDPA", Layered, Keyword, "`dense` (default), `cublas` or `flash`; `host` and `sparse` are retired and refused", Value("dense"), Runtime, "attention_backend", Process,
            "Dense attention backend for every family (`rt::sdpa_rule`): the dense rule's pick, or a forced `cublas` / `flash` route."),
        row("FASTVIDEO_SDPA_AUTO", Layered, Keyword, "", Value("rule"), Runtime, "kernel_choice", Process,
            "How the dense attention backend is chosen."),
        row("FASTVIDEO_SDPA_CHUNK", Env, Text, "a `usize` (not trimmed), at least 64", Computed("by device"), Runtime, "memory", Process,
            "Query rows per SDPA chunk."),
        row("FASTVIDEO_SLA_BLKK", Layered, Usize, "at least 1", Value("64"), Runtime, "attention", Call,
            "SLA key block."),
        row("FASTVIDEO_SLA_BLKQ", Layered, Usize, "at least 1", Value("128"), Runtime, "attention", Call,
            "SLA query block."),
        row("FASTVIDEO_SLA_TOPK", Layered, F32, "clamped to 0.01..=1", Value("0.1"), Runtime, "attention", Call,
            "SLA top-k ratio."),
        row("FASTVIDEO_SOL_KERNEL", Layered, Keyword, "", Value("auto"), Runtime, "kernel_choice", Call,
            "Sol attention kernel (profile `[kernels] sol_attention`)."),
        row("FASTVIDEO_SOL_SPLITS", Layered, Keyword, "", Value("auto"), Runtime, "kernel_choice", Call,
            "Sol attention split count."),
        row("FASTVIDEO_SPLIT_ROWS", Layered, Flag, "", Value("1"), Runtime, "kernel_fusion", Process,
            "Row-split activation kernels (profile `kernel_fusion.split_rows`)."),
        row("FASTVIDEO_SP_WORLD", Layered, Usize, "at least 1", Value("1"), Runtime, "parallel", Process,
            "Sequence-parallel world size (`fastvideo --num-gpus` sets it)."),
        row("FASTVIDEO_STAGED_UPLOAD", Layered, Flag, "", Value("1"), Runtime, "loader", Process,
            "Upload weights through pinned staging buffers."),
        row("FASTVIDEO_STREAMING_FETCH", Env, Text, "on only for `1` (trimmed)", Value("0"), Loader, "loader", Load,
            "The boot started this process while the weights were still being fetched."),
        row("FASTVIDEO_STREAMING_FETCH_STALE_S", Env, Text, "seconds as a positive finite `f64` (trimmed)", Value("120"), Loader, "loader", Load,
            "A fetch whose status file is older than this is dead."),
        row("FASTVIDEO_STREAMING_FETCH_STATUS", Env, Text, "a file (empty = unset)", Unset, Loader, "loader", Load,
            "The fetcher's status JSON."),
        row("FASTVIDEO_STREAMING_FETCH_TIMEOUT_S", Env, Text, "seconds as a positive finite `f64` (trimmed)", Value("3600"), Loader, "loader", Load,
            "The longest a single wait for streamed weights may take."),
        row("FASTVIDEO_STREAMING_PREFETCH_THREADS", Env, Text, "a `usize` (trimmed)", Value("4"), Loader, "loader", Load,
            "Read-ahead threads while the weights stream."),
        row("FASTVIDEO_TAEH3_WEIGHTS", Layered, Text, "a path (empty = unset)", Unset, H3, "video_decoder", Load,
            "H3 tiny decoder (TAEH3) weights."),
        row("FASTVIDEO_TAEHV_CHUNK", Layered, Usize, "at least 1", Computed("the TAE architecture's decode chunk"), Runtime, "memory", Call,
            "TAEHV decode frames per chunk."),
        row("FASTVIDEO_TAEHV_ENC_CHUNK", Layered, Usize, "", Computed("2 × the temporal downscale"), Runtime, "memory", Call,
            "TAEHV encode frames per chunk."),
        row("FASTVIDEO_TAEHV_WEIGHTS", Env, Text, "a path (empty = unset)", Unset, Wan, "video_decoder", Load,
            "Wan tiny decoder (TAEHV) weights."),
        row("FASTVIDEO_TAE_DIR", Env, Text, "a directory (empty = unset)", Unset, Wan, "video_decoder", Load,
            "Where the distilled presets look for `taew2_1.safetensors` (`scripts/gpu/fetch_taehv.sh`)."),
        row("FASTVIDEO_TAE_TEST_DIR", Env, Text, "a directory", Unset, Harness, "test", Call,
            "Real TAE checkpoints for the reference test."),
        row("FASTVIDEO_TEACACHE", Env, Text, "`1` or `true` (any case, not trimmed)", Value("0"), Wan, "step_output", Load,
            "Wan TeaCache (FastVideo's)."),
        row("FASTVIDEO_TEACACHE_RET", Env, Text, "`1` or `true` (any case, not trimmed)", Value("0"), Wan, "step_output", Load,
            "Wan TeaCache `use_ret_steps` coefficients."),
        row("FASTVIDEO_TEACACHE_THRESH", Env, Text, "an `f64` (not trimmed); unparsable = default", Value("0.08"), Wan, "step_output", Load,
            "Wan TeaCache threshold."),
        row("FASTVIDEO_TECHNIQUES", Env, Text, "a profile path or compiled-in name", Unset, Runtime, "techniques", Start,
            "Technique profile to install when no `--techniques` flag did (`docs/techniques.md`)."),
        row("FASTVIDEO_TEXT_FP8_TREE", Env, Text, "`0` (trimmed) skips the tree", Value("1"), TextEncoder, "weights", Load,
            "Text encoder from its pre-quantized FP8 tree when one is valid."),
        row("FASTVIDEO_TF32", Layered, Flag, "", Value("1"), Runtime, "ffn_precision", Process,
            "TF32 tensor-core math for f32 GEMMs."),
        row("FASTVIDEO_VAE_CHUNK", Layered, Usize, "at least 1", Value("2"), Wan, "memory", Call,
            "Latent frames per Wan VAE pass."),
        row("FASTVIDEO_VERIFY_UPLOAD", Layered, Flag, "", Value("0"), Runtime, "debug", Process,
            "Read back every staged upload and compare."),
        row("FASTVIDEO_VSA", Layered, Flag, "the `fastvideo` CLI's VSA-id check reads `1` / `true` (any case, not trimmed) from the environment only", Value("0"), Runtime, "attention_backend", Process,
            "Wan block-sparse video attention."),
        row("FASTVIDEO_VSA_BF16", Layered, Flag, "", Value("1"), H3, "activation_precision", Call,
            "H3 VSA in bf16."),
        row("FASTVIDEO_VSA_FUSED", Layered, Flag, "", Value("0"), Runtime, "kernel_choice", Call,
            "Fused scalar VSA fine kernel."),
        row("FASTVIDEO_VSA_GROUP", Layered, Usize, "H3 technique adapter: its own parse", Computed("8 for H3, 32 for Wan"), Runtime, "attention", Load,
            "VSA block group size."),
        row("FASTVIDEO_VSA_KERNEL", Layered, Keyword, "", Value("auto"), Runtime, "kernel_choice", Call,
            "VSA attention kernel (profile `[kernels] vsa_attention`)."),
        row("FASTVIDEO_VSA_SPARSITY", Layered, F64, "H3 technique adapter: a number in [0, 1)", Computed("0.8 for Wan, the recipe's for H3"), Runtime, "attention", Load,
            "VSA sparsity, one value everywhere."),
        row("FASTVIDEO_VSA_TMA", Layered, Keyword, "", Value("1"), Runtime, "kernel_choice", Call,
            "`0` turns the VSA TMA kernels off."),
        row("FASTVIDEO_VSA_TOPK", Layered, Keyword, "", Value("v2"), Runtime, "kernel_choice", Call,
            "`v1` runs the first VSA top-k kernel."),
        row("FASTVIDEO_VSA_UNION_LOG", Layered, Flag, "", Value("0"), H3, "debug", Call,
            "Log the VSA block-union statistics."),
        row("FASTVIDEO_VSA_WINDOW", Retired, Text, "not read", Unset, Runtime, "attention", Call,
            "Retired with the host block-sparse prototype (`FASTVIDEO_SDPA=sparse`, plan B5): not read. `FASTVIDEO_SDPA=sparse` is refused, and the refusal names this variable."),
        row("FASTVIDEO_WAN_AUDIO", Env, Text, "`mmaudio` or `1` (trimmed)", Unset, MmAudio, "audio", Request,
            "MMAudio soundtrack for Wan clips."),
        row("FASTVIDEO_WAN_AUDIO_CFG", Env, Text, "parsed as the preset's number (not trimmed); unparsable = default", Computed("the MMAudio preset's"), MmAudio, "audio", Request,
            "MMAudio guidance."),
        row("FASTVIDEO_WAN_AUDIO_PROMPT", Env, Text, "text", Computed("the video prompt"), MmAudio, "audio", Request,
            "MMAudio prompt."),
        row("FASTVIDEO_WAN_AUDIO_SEED", Env, Text, "an integer (not trimmed); unparsable = default", Computed("the request's seed"), MmAudio, "audio", Request,
            "MMAudio seed."),
        row("FASTVIDEO_WAN_AUDIO_STEPS", Env, Text, "a `usize` (not trimmed); unparsable = default", Computed("the MMAudio preset's"), MmAudio, "audio", Request,
            "MMAudio steps."),
        row("FASTVIDEO_WAN_CAUSAL_AR", Layered, Flag, "", Value("1"), Wan, "attention", Process,
            "SF-Wan autoregressive KV cache."),
        row("FASTVIDEO_WAN_CAUSAL_FLASH", Layered, Flag, "", Value("1"), Wan, "kernel_choice", Process,
            "SF-Wan block-causal self-attention on the masked flash kernel (default on); `0` materializes the `[S, S]` mask through `sdpa_composed`."),
        row("FASTVIDEO_WAN_CAUSAL_FPB", Layered, Usize, "", Computed("the config's frames per block"), Wan, "attention", Load,
            "SF-Wan frames per block."),
        row("FASTVIDEO_WAN_COND_CACHE", Layered, Flag, "`fv-gpucheck wan` reports the environment's value only", Value("1"), Wan, "cache", Process,
            "`0`: recompute text K/V, text and time embeddings every forward."),
        row("FASTVIDEO_WAN_DIT_LAZY", Layered, Flag, "", Computed("on for two experts or a streamed DiT"), Wan, "residency", Load,
            "Open the Wan DiT weights lazily."),
        row("FASTVIDEO_WAN_EASYCACHE_COOLDOWN", Env, Text, "a `usize` (not trimmed), else `WAN22_EASYCACHE_COOLDOWN_STEPS`", Computed("the EasyCache profile's"), Wan, "step_output", Request,
            "Wan EasyCache cooldown steps."),
        row("FASTVIDEO_WAN_EASYCACHE_PROFILE", Env, Text, "a profile name, else `WAN22_EASYCACHE_PROFILE`", Unset, Wan, "step_output", Request,
            "Wan EasyCache parameter profile."),
        row("FASTVIDEO_WAN_EASYCACHE_RETAIN", Env, Text, "a `usize` (not trimmed), else `WAN22_EASYCACHE_RETAIN_STEPS`", Computed("the EasyCache profile's"), Wan, "step_output", Request,
            "Wan EasyCache retained steps."),
        row("FASTVIDEO_WAN_EASYCACHE_THRESH", Env, Text, "an `f64` (not trimmed), else `WAN22_EASYCACHE_THRESHOLD`", Computed("the EasyCache profile's"), Wan, "step_output", Request,
            "Wan EasyCache threshold."),
        row("FASTVIDEO_WAN_FUSE", Layered, Flag, "", Value("1"), Wan, "kernel_fusion", Process,
            "Wan block residual + LayerNorm (+ AdaLN) as one kernel under bf16 activations (default on); `0` runs the byte-identical two-kernel form."),
        row("FASTVIDEO_WAN_GRAPH", Layered, Flag, "", Value("1"), Wan, "cuda_graphs", Process,
            "CUDA-graph capture of the Wan step."),
        row("FASTVIDEO_WAN_GRAPH_CUBLAS_WS_MIB", Layered, Usize, "", Value("32"), Wan, "memory", Call,
            "cuBLAS workspace for graph-captured GEMMs."),
        row("FASTVIDEO_WAN_MOE", Env, Text, "`auto` / empty, `both` / `resident`, `swap` / `parked` (trimmed, any case); anything else is an error", Value("auto"), Wan, "residency", Load,
            "Wan 2.2 A14B experts: `auto` (both resident when the free memory after UMT5 covers two bf16 experts plus `FASTVIDEO_WAN_MOE_HEADROOM_GIB`, else `swap`), `both`, or `swap` (both parked in pinned host memory, swapped at the 0.875 boundary). See [ports/sol-wan-ltx23.md](ports/sol-wan-ltx23.md)."),
        row("FASTVIDEO_WAN_MOE_HEADROOM_GIB", Env, Text, "GiB as a finite `f64` ≥ 0 (trimmed)", Value("36"), Wan, "residency", Load,
            "Headroom `FASTVIDEO_WAN_MOE=auto` keeps beside two experts."),
        row("FASTVIDEO_WAN_PISA", Env, Text, "`1`, `pisa`, `true` or `on` (trimmed), else `WAN22_PISA`", Value("0"), Wan, "attention_backend", Load,
            "Wan PISA sparse attention."),
        row("FASTVIDEO_WAN_QUANT", Layered, Text, "`mxfp8`, `w8a8` or `off` (an unknown value is an error)", Computed("`mxfp8` on sm_100, `off` elsewhere"), Wan, "ffn_precision", Load,
            "Wan DiT FP8 recipe on every block's attention + FFN linears (on sm_120 both were slower than bf16 and doubled LPIPS)."),
        row("FASTVIDEO_WAN_SOL_ATTN", Env, Text, "`1`, `sol`, `true`, `on` or `fullstack` (trimmed), else `WAN22_SOL_ATTN`", Value("0"), Wan, "attention_backend", Load,
            "Sol-Attn route: 14B (10 dense forwards, layer 0 dense) and 1.3B (layer 0 dense), Morton3D on the device. `fullstack` on the 1.3B: the 14B guards (sol-engine `wan21_fullstack_sol.toml`, the 50-step base model)."),
        row("FASTVIDEO_WAN_SOL_CACHE", Env, Text, "`easycache`, `teacache` or `taylorseer`; empty / `0` / `off` / `none` / `false` = off (trimmed, any case)", Unset, Wan, "step_output", Request,
            "Wan Sol step cache (lossy; off by default). `taylorseer` forecasts `proj_out` (interval 3, warmup 3, cooldown 2, order 1)."),
        row("FASTVIDEO_WAN_TAYLOR_COOLDOWN", Env, Text, "a `usize` (not trimmed), else `WAN22_TAYLOR_COOLDOWN_START`", Computed("steps − 2"), Wan, "step_output", Request,
            "TaylorSeer cooldown start."),
        row("FASTVIDEO_WAN_TAYLOR_INTERVAL", Env, Text, "a `usize` (not trimmed), else `WAN22_TAYLOR_INTERVAL`", Value("3"), Wan, "step_output", Request,
            "TaylorSeer interval."),
        row("FASTVIDEO_WAN_TAYLOR_LITE", Env, Text, "`0` / `false` / `off` (trimmed) are refused, else `WAN22_TAYLOR_LITE`", Value("1"), Wan, "step_output", Request,
            "TaylorSeer lite mode (the only mode implemented)."),
        row("FASTVIDEO_WAN_TAYLOR_ORDER", Env, Text, "a `usize` (not trimmed), else `WAN22_TAYLOR_ORDER`", Value("1"), Wan, "step_output", Request,
            "TaylorSeer order."),
        row("FASTVIDEO_WAN_TAYLOR_WARMUP", Env, Text, "a `usize` (not trimmed), else `WAN22_TAYLOR_WARMUP`", Value("3"), Wan, "step_output", Request,
            "TaylorSeer warmup."),
        row("FASTVIDEO_WAN_TEACACHE_COEFFS", Env, Text, "comma-separated `f64`s (unparsable parts dropped)", Value("1.0,0.0"), Wan, "step_output", Request,
            "Wan Sol TeaCache polynomial."),
        row("FASTVIDEO_WAN_TEACACHE_END", Env, Text, "a `usize` (not trimmed)", Computed("steps − 2"), Wan, "step_output", Request,
            "Wan Sol TeaCache last step."),
        row("FASTVIDEO_WAN_TEACACHE_MAX_HITS", Env, Text, "a `usize` (not trimmed)", Value("0"), Wan, "step_output", Request,
            "Wan Sol TeaCache consecutive-skip cap (0: none)."),
        row("FASTVIDEO_WAN_TEACACHE_PERIODIC", Env, Text, "a `usize` (not trimmed)", Value("0"), Wan, "step_output", Request,
            "Wan Sol TeaCache forced refresh period (0: none)."),
        row("FASTVIDEO_WAN_TEACACHE_START", Env, Text, "a `usize` (not trimmed)", Value("2"), Wan, "step_output", Request,
            "Wan Sol TeaCache first step."),
        row("FASTVIDEO_WAN_TEACACHE_THRESH", Env, Text, "an `f64` (not trimmed)", Value("0.12"), Wan, "step_output", Request,
            "Wan Sol TeaCache threshold."),
        row("FASTVIDEO_WAN_TEXT_CACHE", Env, Text, "a directory (trimmed), or `off` / `0` / `false` / `none`", Computed("`$XDG_CACHE_HOME/fastvideo/wan-text`, else `~/.cache/fastvideo/wan-text`"), Wan, "cache", Load,
            "UMT5 prompt cache directory, or `off`."),
        row("FASTVIDEO_WAN_UNIPC_SIGMAS", Env, Text, "`fastvideo` / `double` / empty, `diffusers` / `single` / `sol` (trimmed, any case); anything else is an error", Value("fastvideo"), Wan, "scheduler", Load,
            "`fastvideo` (FastVideo's double-shifted flow-UniPC sigmas) or `diffusers` (Diffusers' single shift, the sol-engine Wan baselines)."),
        row("FASTVIDEO_WAN_VAE", Env, Text, "`auto`, `full`, `taehv` (trimmed, any case)", Value("auto"), Wan, "video_decoder", Load,
            "`auto` (TAEHV for distilled presets when found), `full`, `taehv`."),
        row("FASTVIDEO_WEIGHTS", Env, Text, "a directory", Computed("`--weights`, else the Hugging Face snapshot"), Harness, "weights", Load,
            "Weights root when `--weights` is not given."),
        row("FASTVIDEO_WRITER_TRACE", Layered, Flag, "", Value("0"), Runtime, "trace", Request,
            "Log the frame writer's timing."),
    ];
}

#[cfg(test)]
mod tests {
    use super::super::parse;
    use super::super::{var_in, Settings};
    use super::*;

    #[test]
    fn rows_are_sorted_unique_and_complete() {
        for w in SETTINGS.windows(2) {
            assert!(w[0].name < w[1].name, "{} before {}", w[0].name, w[1].name);
        }
        for s in SETTINGS {
            let tail = s.name.strip_prefix("FASTVIDEO_").unwrap_or("");
            assert!(
                !tail.is_empty()
                    && tail
                        .bytes()
                        .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
                    && !tail.ends_with('_'),
                "{}: not a FASTVIDEO_* name",
                s.name
            );
            assert!(!s.doc.trim().is_empty(), "{}: no doc", s.name);
            assert!(!s.seam.trim().is_empty(), "{}: no seam", s.name);
            if s.parse == Parse::Text {
                assert!(
                    !s.rule.is_empty(),
                    "{}: a text row needs its parse rule",
                    s.name
                );
            }
            assert_eq!(lookup(s.name), Some(s));
        }
        assert_eq!(lookup("FASTVIDEO_NOT_A_SETTING"), None);
    }

    /// The parsed value of a typed row, as the runtime's getters compute it
    /// for `raw` with the row's documented default.
    fn typed(s: &Setting, raw: Option<&str>, default: &str) -> String {
        let num =
            |what: &str| -> String { panic!("{}: default {default:?} is not {what}", s.name) };
        match s.parse {
            Parse::Flag => {
                let d = match default {
                    "1" => true,
                    "0" => false,
                    _ => return num("`0` or `1`"),
                };
                parse::flag(raw, d).to_string()
            }
            Parse::Usize => {
                let d = default.parse().unwrap_or_else(|_| {
                    num("a usize");
                    0
                });
                parse::usize_or(raw, d).to_string()
            }
            Parse::F32 => {
                let d = default.parse().unwrap_or_else(|_| {
                    num("an f32");
                    0.0
                });
                parse::f32_or(raw, d).to_string()
            }
            Parse::F64 => {
                let d = default.parse().unwrap_or_else(|_| {
                    num("an f64");
                    0.0
                });
                parse::f64_or(raw, d).to_string()
            }
            Parse::Keyword => parse::lower_or(raw, default),
            Parse::Text => unreachable!("typed() is only called on typed rows"),
        }
    }

    /// Off-identity, typed rows: for every row with a documented default,
    /// unset and set-to-the-default parse to the same value, through the
    /// layered reader with and without a profile table.
    #[test]
    fn unset_and_the_documented_default_are_the_same_setting() {
        let mut checked = 0;
        for s in SETTINGS {
            let DefaultValue::Value(v) = s.default else {
                continue;
            };
            if s.parse == Parse::Text {
                continue;
            }
            let empty = Settings::default();
            let unset = |_: &str| None;
            let set = |_: &str| Some(v.to_string());
            let a = var_in(&empty, &unset, s.name);
            let b = var_in(&empty, &set, s.name);
            assert_eq!(
                typed(s, a.as_deref(), v),
                typed(s, b.as_deref(), v),
                "{}: unset and {v:?} differ",
                s.name
            );
            checked += 1;
        }
        assert!(checked > 80, "only {checked} typed rows with a default");
    }

    /// A text row's parsed value, through the parser its reader calls, for
    /// the rows whose parser lives in this crate.
    type Check = fn(Option<&str>) -> String;

    fn text_checks() -> Vec<(&'static str, Check)> {
        fn dbg<T: std::fmt::Debug>(v: T) -> String {
            format!("{v:?}")
        }
        vec![
            ("FASTVIDEO_COSMOS3_OFFICIAL", |v| {
                dbg(crate::cosmos::sol::official_requested(v))
            }),
            ("FASTVIDEO_COSMOS_SOL", |v| {
                dbg(crate::cosmos::sol::teacache_requested(v))
            }),
            ("FASTVIDEO_HUNYUAN15_OFFICIAL", |v| {
                dbg(crate::hunyuan15::sol::official_requested(v))
            }),
            ("FASTVIDEO_HUNYUAN15_SOL", |v| {
                dbg(crate::hunyuan15::sol::teacache_requested(v))
            }),
            ("FASTVIDEO_LINGBOT_OFFICIAL", |v| {
                dbg(crate::lingbot::sol::official_requested(v))
            }),
            ("FASTVIDEO_LINGBOT_SOL", |v| {
                dbg(crate::lingbot::sol::SolArm::parse(v))
            }),
            ("FASTVIDEO_LTX2_FBCACHE", |v| {
                dbg(crate::ltx2::fbcache::requested(v))
            }),
            ("FASTVIDEO_LTX2_HQ", |v| dbg(crate::ltx2::hq::requested(v))),
            ("FASTVIDEO_LTX2_MIDPOINT_PRUNE", |v| {
                dbg(crate::ltx2::pisa::midpoint_prune_requested(v))
            }),
            ("FASTVIDEO_LTX2_STAGE1_CACHE", |v| {
                dbg(crate::ltx2::pisa::stage1_cache_requested(v))
            }),
            ("FASTVIDEO_LTX_OFFLOAD", |v| {
                // `LtxOffload::from_env`: unset is the default arm.
                dbg(
                    v.map_or(Ok(crate::ltx2::memory::LtxOffload::default()), |v| {
                        crate::ltx2::memory::LtxOffload::parse(v)
                    }),
                )
            }),
            ("FASTVIDEO_NVFP4", |v| dbg(crate::nvfp4::requested(v))),
            ("FASTVIDEO_NVFP4_COSMOS_STEPS", |v| {
                dbg(crate::nvfp4::cosmos_steps_flag(v))
            }),
            ("FASTVIDEO_NVFP4_LTX_VIDEO_FFN", |v| {
                dbg(crate::nvfp4::ltx_video_ffn_flag(v))
            }),
            ("FASTVIDEO_NVFP4_OXIDE_GEMM", |v| {
                dbg(crate::nvfp4::oxide_gemm_flag(v))
            }),
            ("FASTVIDEO_SANA_OPT", |v| {
                dbg(crate::sana_video::sol::SanaOptimizations::parse(
                    v, None, None, None,
                ))
            }),
            ("FASTVIDEO_WAN_MOE", |v| {
                dbg(v.map_or(Ok(crate::wan::moe::MoeResidency::Auto), |v| {
                    crate::wan::moe::MoeResidency::parse(v)
                }))
            }),
            ("FASTVIDEO_WAN_PISA", |v| {
                dbg(crate::wan::sol::pisa_requested(v))
            }),
            ("FASTVIDEO_WAN_SOL_ATTN", |v| {
                dbg(crate::wan::sol::sol_attn_requested(v))
            }),
            ("FASTVIDEO_WAN_UNIPC_SIGMAS", |v| {
                dbg(
                    v.map_or(Ok(crate::schedulers::UniPcSigmas::FastVideo), |v| {
                        crate::schedulers::UniPcSigmas::parse(v)
                    }),
                )
            }),
        ]
    }

    /// Off-identity, text rows whose parser lives in this crate: unset and
    /// the documented default parse to the same value. The other text rows
    /// are parsed in `fastvideo-cudarc`, `-core` or `-loader`; their rows say
    /// how, and the routing change left those parsers untouched.
    #[test]
    fn text_rows_off_identity_through_their_parsers() {
        for (name, check) in text_checks() {
            let s = lookup(name).unwrap_or_else(|| panic!("{name}: not a row"));
            assert_eq!(s.parse, Parse::Text, "{name}");
            let DefaultValue::Value(v) = s.default else {
                panic!("{name}: a checked row documents a value default");
            };
            assert_eq!(
                check(None),
                check(Some(v)),
                "{name}: unset and {v:?} differ"
            );
        }
    }

    /// Every name a technique, a `[kernels]` entry or a compiled-in profile
    /// installs is read through the layered reader; an `Env` row there would
    /// make the profile entry silently do nothing.
    #[test]
    fn profile_settings_are_layered_rows() {
        let mut seen = 0;
        for name in crate::techniques::builtin::names() {
            let text = crate::techniques::builtin::get(name).unwrap_or_default();
            let profile = crate::techniques::Profile::parse(text)
                .unwrap_or_else(|e| panic!("profile {name}: {e}"));
            let settings = profile
                .settings()
                .unwrap_or_else(|e| panic!("profile {name}: {e}"));
            for (key, _, _) in settings.iter() {
                let row =
                    lookup(key).unwrap_or_else(|| panic!("profile {name}: {key} is not a row"));
                assert_eq!(row.reader, Reader::Layered, "profile {name}: {key}");
                seen += 1;
            }
        }
        for op in crate::techniques::kernels::KernelOp::ALL {
            let row = lookup(op.setting()).unwrap_or_else(|| panic!("{}: not a row", op.setting()));
            assert_eq!(row.reader, Reader::Layered, "{}", op.setting());
            assert_eq!(
                row.default,
                DefaultValue::Value(op.default_value()),
                "{}: the row's default is the dispatch site's",
                op.setting()
            );
        }
        assert!(seen > 0, "no compiled-in profile installed a setting");
    }

    /// Profile-vs-env precedence for H3 and LTX-2 keys: the environment
    /// wins, the profile fills an unset name, the default fills the rest.
    #[test]
    fn env_beats_profile_beats_default_for_h3_and_ltx_keys() {
        let h3 = crate::techniques::Profile::parse(
            "[id]\nname='x'\n[pipeline]\nmodel='h3'\n\
             [techniques.mxfp8]\n[techniques.bf16_activations]\n\
             [techniques.offload]\ndit='streamed'\nlookahead=2\n\
             [kernels]\ndense_attention='cudnn'\n",
        )
        .and_then(|p| p.settings())
        .unwrap_or_else(|e| panic!("{e}"));
        let ltx = crate::techniques::Profile::parse(
            "[id]\nname='x'\n[pipeline]\nmodel='ltx2'\n\
             [techniques.offload]\nplacement='cpu'\n",
        )
        .and_then(|p| p.settings())
        .unwrap_or_else(|e| panic!("{e}"));
        let cases: &[(&Settings, &str, &str, &str)] = &[
            (&h3, "FASTVIDEO_H3_QUANT", "mxfp8", "w8a8"),
            (&h3, "FASTVIDEO_BF16_ACT", "1", "0"),
            (&h3, "FASTVIDEO_DIT_OFFLOAD", "streamed", "resident"),
            (&h3, "FASTVIDEO_DIT_OFFLOAD_LOOKAHEAD", "2", "4"),
            (&h3, "FASTVIDEO_FLASH_KERNEL", "cudnn", "flash"),
            (&ltx, "FASTVIDEO_LTX_OFFLOAD", "cpu", "none"),
        ];
        for &(table, name, from_profile, from_env) in cases {
            let row = lookup(name).unwrap_or_else(|| panic!("{name}: not a row"));
            assert_eq!(row.reader, Reader::Layered, "{name}");
            let unset = |_: &str| None;
            let env = |_: &str| Some(from_env.to_string());
            assert_eq!(
                var_in(table, &unset, name).as_deref(),
                Some(from_profile),
                "{name}"
            );
            assert_eq!(
                var_in(table, &env, name).as_deref(),
                Some(from_env),
                "{name}"
            );
            let empty = Settings::default();
            assert_eq!(
                var_in(&empty, &unset, name),
                None,
                "{name}: the default applies"
            );
        }
        // The typed getters on top: a profile's `0` turns a default-on flag
        // off, an env `1` turns it back on.
        let mut off = Settings::default();
        off.set("FASTVIDEO_BF16_ACT", "0", "t")
            .unwrap_or_else(|e| panic!("{e}"));
        let unset = |_: &str| None;
        let on = |_: &str| Some("1".to_string());
        assert!(!parse::flag(
            var_in(&off, &unset, "FASTVIDEO_BF16_ACT").as_deref(),
            true
        ));
        assert!(parse::flag(
            var_in(&off, &on, "FASTVIDEO_BF16_ACT").as_deref(),
            true
        ));
    }

    /// The env-only reader does not see a model scope's (profile) table; the
    /// layered reader does.
    #[test]
    fn env_rows_ignore_the_profile_table() {
        let name = "FASTVIDEO_WAN_VAE";
        if std::env::var_os(name).is_some() {
            return; // the environment decides both readers
        }
        assert_eq!(lookup(name).map(|s| s.reader), Some(Reader::Env));
        let mut s = Settings::default();
        s.set(name, "full", "t").unwrap_or_else(|e| panic!("{e}"));
        let _scope = super::super::scope_model(s);
        assert_eq!(super::super::var(name).as_deref(), Some("full"));
        assert!(super::super::env_var(name).is_err());
        assert!(super::super::env_var_os(name).is_none());
    }
}
