//! `Technique`, its phase, its effect set (seams read / written) and the
//! capabilities a model must provide for it.
//!
//! Port of sol-engine `techniques/technique.py` (phases :27-38, seams
//! :41-62, capabilities :65-82, the base class :101-139) and
//! `techniques/transform.py` (build/load transforms, :32-77) folded into one
//! trait: a [`Technique`] is either a runtime technique that runs in a
//! forward [`Phase`] or a build/load transform that runs at a
//! [`TransformPhase`] ([`Kind`]).
//!
//! **Hooks live in the model adapter.** sol-engine's hooks (`before_blocks`,
//! `wrap_attention`, `on_step`, :124-139) are Python callables over torch
//! tensors. Here a technique is a typed, validated parameter set; the model
//! adapter (the H3 pipeline, the LTX-2 pipeline) implements the seam and
//! asks the composed [`super::compose::Plan`] what to do at each step and
//! layer. That is the same split sol-engine's transforms use: they
//! "delegate to the EXISTING mechanisms (they set the env/config the current
//! load/build code already reads)" (`transform.py:18-20`), and a sparse
//! attention transform leaves "translating latent/text shapes into backend
//! metadata" to the model runtime (`transforms/sparse_attention.py:3-6`).
//! [`Technique::settings`] is `set_env` (`transform.py:73-74`): the
//! process-wide settings a transform installs, read by the deep kernels.

use std::any::Any;
use std::fmt;

use super::schedule::Schedule;

/// Forward phases, in execution order (`technique.py:27-38`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    /// Swap or wrap the attention op (sparse attention).
    WrapAttention = 10,
    /// Before the block loop (token-prune gather).
    PreBlocks = 20,
    /// Per block (block-level cache decisions).
    InBlocks = 30,
    /// After the block loop (token-prune scatter).
    PostBlocks = 40,
    /// Whole-step decision (step-output cache replay: TeaCache).
    OnStep = 50,
}

/// When a transform is applied (`transform.py:32-36`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TransformPhase {
    /// At weight load (quantize linears, pick the decoder, place blocks).
    Load = 10,
    /// At module construction (install an attention backend, fusions, kernels).
    Build = 20,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    Runtime(Phase),
    Transform(TransformPhase),
}

impl Kind {
    /// Transforms first (load, build), then runtime phases (compose.py:177-178).
    pub fn order(self) -> u32 {
        match self {
            Kind::Transform(p) => p as u32,
            Kind::Runtime(p) => 100 + p as u32,
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Kind::Runtime(p) => write!(f, "runtime/{p:?}"),
            Kind::Transform(p) => write!(f, "transform/{p:?}"),
        }
    }
}

/// Named mutation points (`technique.py:41-54`), plus three this runtime
/// adds: which video decoder runs, where the DiT blocks live, and the
/// activation precision. Those three are process- or pipeline-wide choices
/// with one owner, so they are exclusive like sol-engine's four.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Seam {
    /// Attention output values (shared).
    Attention,
    /// Which attention kernel / algorithm (EXCLUSIVE).
    AttentionBackend,
    /// The set / count of tokens in the blocks (EXCLUSIVE).
    TokenSet,
    /// Block hidden-state values (shared).
    HiddenStates,
    /// Op fusion of attn / adaLN / FFN kernels (shared).
    KernelFusion,
    /// Cached block / step residuals (shared).
    ResidualCache,
    /// Denoiser output for the whole step (EXCLUSIVE).
    StepOutput,
    /// Numeric precision of the FFN / linear compute (EXCLUSIVE).
    FfnPrecision,
    /// Activation dtype between ops (EXCLUSIVE; this runtime).
    ActivationPrecision,
    /// Which video decoder turns latents into frames (EXCLUSIVE; this runtime).
    VideoDecoder,
    /// Where the DiT blocks live: resident or streamed (EXCLUSIVE; this runtime).
    Residency,
}

impl Seam {
    /// At most one active writer per plan (`EXCLUSIVE_SEAMS`, `technique.py:57-62`).
    pub fn is_exclusive(self) -> bool {
        matches!(
            self,
            Seam::AttentionBackend
                | Seam::TokenSet
                | Seam::StepOutput
                | Seam::FfnPrecision
                | Seam::ActivationPrecision
                | Seam::VideoDecoder
                | Seam::Residency
        )
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Seam::Attention => "attention",
            Seam::AttentionBackend => "attention_backend",
            Seam::TokenSet => "token_set",
            Seam::HiddenStates => "hidden_states",
            Seam::KernelFusion => "kernel_fusion",
            Seam::ResidualCache => "residual_cache",
            Seam::StepOutput => "step_output",
            Seam::FfnPrecision => "ffn_precision",
            Seam::ActivationPrecision => "activation_precision",
            Seam::VideoDecoder => "video_decoder",
            Seam::Residency => "residency",
        }
    }

    pub const ALL: [Seam; 11] = [
        Seam::Attention,
        Seam::AttentionBackend,
        Seam::TokenSet,
        Seam::HiddenStates,
        Seam::KernelFusion,
        Seam::ResidualCache,
        Seam::StepOutput,
        Seam::FfnPrecision,
        Seam::ActivationPrecision,
        Seam::VideoDecoder,
        Seam::Residency,
    ];
}

/// Structural seams a model provides (`technique.py:65-82`), plus the two
/// load-time ones this runtime needs (a swappable video decoder, streamable
/// DiT blocks).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    Blocks,
    PrunableTokens,
    ResidualTuple,
    SwappableAttention,
    HasDenoiseSteps,
    HasTransformerBlocks,
    HasAttentionLayers,
    HasAttentionBackendSwitch,
    HasFfnLinearModules,
    HasTokenSequenceAxis,
    HasSpatiotemporalTokenLayout,
    SupportsTokenGatherScatter,
    SupportsStepCache,
    SupportsCudaGraphProbe,
    SupportsNvfp4Linear,
    /// The video decoder can be replaced by a tiny autoencoder.
    SwappableVideoDecoder,
    /// DiT blocks can stream from pinned host memory.
    SupportsLayerOffload,
}

/// A model's declaration of the seams it exposes (`spec.py:20-55`).
#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub name: &'static str,
    pub capabilities: Vec<Capability>,
    /// Transformer blocks per forward (the layer axis of a route).
    pub layers: usize,
}

impl ModelSpec {
    pub fn has(&self, c: Capability) -> bool {
        self.capabilities.contains(&c)
    }

    pub fn missing(&self, required: &[Capability]) -> Vec<Capability> {
        required.iter().copied().filter(|c| !self.has(*c)).collect()
    }
}

/// An inference-acceleration technique: when it is active, where it runs,
/// what it touches, what it needs.
///
/// Implementations are plain parameter structs; OFF (`enabled = false`)
/// must leave the pipeline byte-identical to not listing the technique at
/// all (the off-identity invariant, `technique.py:104-107`).
pub trait Technique: fmt::Debug + Send + Sync + 'static {
    /// Registry name (the `[techniques.<name>]` table).
    fn name(&self) -> &'static str;
    fn kind(&self) -> Kind;
    fn reads(&self) -> &'static [Seam] {
        &[]
    }
    fn writes(&self) -> &'static [Seam];
    fn required_capabilities(&self) -> &'static [Capability] {
        &[]
    }
    /// Steps (and stages) at which the technique is active.
    fn enabled(&self) -> &Schedule<bool>;
    /// Process-wide settings this technique installs (`set_env`): the
    /// `FASTVIDEO_*` name the deep kernels read, and its value. An env var of
    /// the same name still wins ([`super::settings::var`]).
    fn settings(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
    /// One line for the pipeline log.
    fn describe(&self) -> String {
        format!("{} ({})", self.name(), self.kind())
    }
    fn as_any(&self) -> &dyn Any;
    fn clone_box(&self) -> Box<dyn Technique>;
}

impl Clone for Box<dyn Technique> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl dyn Technique {
    pub fn downcast_ref<T: Technique>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }

    /// Active at some step below `horizon` (`compose._always_on`, :48-53).
    pub fn active_somewhere(&self, horizon: usize) -> bool {
        !self.enabled().truthy_steps(horizon).is_empty()
    }
}
