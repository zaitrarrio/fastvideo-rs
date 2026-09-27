//! Kernel seam: which implementation runs an op, chosen by config, not by
//! the pipeline.
//!
//! Each [`KernelBackend`] is one provider of device code (our nvcc/NVRTC
//! kernels, the oxide Tile-IR cubins, cuDNN, cuBLAS) and lists the
//! implementations it has for each [`KernelOp`]. A profile's `[kernels]`
//! table names one per op (`dense_attention = "cudnn"`,
//! `sol_attention = "nvcc:x4f"`); [`resolve`] checks the choice against the
//! backends (unknown name, wrong op, too-old SM are config errors) and
//! [`KernelChoice::setting`] is the process-wide setting the dispatch site
//! reads ([`KernelOp::setting`], through [`super::settings::var`], so the
//! existing `FASTVIDEO_*_KERNEL` env vars still win). The dispatch sites in
//! `fastvideo-cudarc` (`wan::attn::flash_kernel_for`,
//! `wan::ops::sol_kernel_choice`, the VSA pickers, `conv3d`) are the only
//! readers: swapping a kernel touches neither a pipeline nor a model.

use std::fmt;

/// An op with more than one implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelOp {
    /// Dense softmax attention (flash).
    DenseAttention,
    /// Sol-Attn block-sparse attention.
    SolAttention,
    /// VSA fine (selected-tile) attention.
    VsaAttention,
    /// NVFP4 W4A4 GEMM.
    Nvfp4Gemm,
    /// 3-D convolution (VAE).
    Conv3d,
}

impl KernelOp {
    pub const ALL: [KernelOp; 5] = [
        KernelOp::DenseAttention,
        KernelOp::SolAttention,
        KernelOp::VsaAttention,
        KernelOp::Nvfp4Gemm,
        KernelOp::Conv3d,
    ];

    /// The `[kernels]` key.
    pub fn key(self) -> &'static str {
        match self {
            KernelOp::DenseAttention => "dense_attention",
            KernelOp::SolAttention => "sol_attention",
            KernelOp::VsaAttention => "vsa_attention",
            KernelOp::Nvfp4Gemm => "nvfp4_gemm",
            KernelOp::Conv3d => "conv3d",
        }
    }

    /// The setting the dispatch site reads (also the legacy env var).
    pub fn setting(self) -> &'static str {
        match self {
            KernelOp::DenseAttention => "FASTVIDEO_FLASH_KERNEL",
            KernelOp::SolAttention => "FASTVIDEO_SOL_KERNEL",
            KernelOp::VsaAttention => "FASTVIDEO_VSA_KERNEL",
            KernelOp::Nvfp4Gemm => "FASTVIDEO_NVFP4_OXIDE_GEMM",
            KernelOp::Conv3d => "FASTVIDEO_CONV3D",
        }
    }

    /// The dispatch site's value when nothing is set.
    pub fn default_value(self) -> &'static str {
        match self {
            KernelOp::Nvfp4Gemm => "0",
            _ => "auto",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|op| op.key() == key)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Provider {
    /// Our CUDA C++ kernels (NVRTC or embedded AOT cubins).
    Nvcc,
    /// Rust Tile-IR cubins (`fastvideo-oxide-kernels`, cutile-rs).
    Oxide,
    Cudnn,
    Cublas,
}

impl Provider {
    pub fn as_str(self) -> &'static str {
        match self {
            Provider::Nvcc => "nvcc",
            Provider::Oxide => "oxide",
            Provider::Cudnn => "cudnn",
            Provider::Cublas => "cublas",
        }
    }
}

/// One implementation of one op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelImpl {
    pub op: KernelOp,
    /// The name a profile uses (`"x4f"`, `"cudnn"`).
    pub id: &'static str,
    /// The setting value the dispatch site understands.
    pub value: &'static str,
    /// Lowest SM (major*10+minor) it runs on.
    pub min_sm: u32,
    pub note: &'static str,
}

/// A provider of device code.
pub trait KernelBackend: Sync {
    fn provider(&self) -> Provider;
    fn kernels(&self) -> &'static [KernelImpl];
    /// Whether `k` can run on `sm` (major*10+minor).
    fn available(&self, k: &KernelImpl, sm: u32) -> bool {
        sm >= k.min_sm
    }
}

const fn k(
    op: KernelOp,
    id: &'static str,
    value: &'static str,
    min_sm: u32,
    note: &'static str,
) -> KernelImpl {
    KernelImpl {
        op,
        id,
        value,
        min_sm,
        note,
    }
}

pub struct NvccBackend;
pub struct OxideBackend;
pub struct CudnnBackend;
pub struct CublasBackend;

static NVCC: [KernelImpl; 14] = [
    k(
        KernelOp::DenseAttention,
        "v1",
        "v1",
        80,
        "flash_mma_fwd: 64-query CTAs",
    ),
    k(
        KernelOp::DenseAttention,
        "v2",
        "v2",
        80,
        "flash_mma_fwd2: 128-query CTAs, double-buffered K/V (bit-identical to v1)",
    ),
    k(
        KernelOp::DenseAttention,
        "v3",
        "v3",
        80,
        "flash_mma_fwd3: v2 with S_{j+1} issued before softmax(S_j), 3-stage K/V ring (bit-identical to v2; d=128)",
    ),
    k(
        KernelOp::DenseAttention,
        "v3s",
        "v3s",
        80,
        "flash_mma_fwd3s: v3 that skips the O rescale when no row max rose (d=128)",
    ),
    k(KernelOp::SolAttention, "v1", "v1", 80, "sol_mma_fwd"),
    k(KernelOp::SolAttention, "x4", "x4", 80, "sol_mma_fwd_x4"),
    k(
        KernelOp::SolAttention,
        "x4f",
        "x4f",
        80,
        "sol_mma_fwd_x4f: ex2.approx exp2 (the auto kernel)",
    ),
    k(
        KernelOp::SolAttention,
        "ws",
        "ws",
        90,
        "sol_mma_fwd2: warp-specialised, KV splits",
    ),
    k(
        KernelOp::VsaAttention,
        "gather",
        "gather",
        0,
        "gathered tiles + dense SDPA",
    ),
    k(
        KernelOp::VsaAttention,
        "fused",
        "fused",
        0,
        "scalar f32 fused (measured negative result)",
    ),
    k(
        KernelOp::VsaAttention,
        "mma",
        "mma",
        80,
        "mma.sync bf16 online softmax",
    ),
    k(KernelOp::VsaAttention, "tma", "tma", 90, "TMA two-stage"),
    k(
        KernelOp::VsaAttention,
        "tma2",
        "tma2",
        90,
        "TMA three-slot ring (the auto kernel on sm90+)",
    ),
    k(
        KernelOp::Conv3d,
        "unfold",
        "unfold",
        0,
        "temporal unfold + GEMM",
    ),
];

static OXIDE: [KernelImpl; 1] = [k(
    KernelOp::Nvfp4Gemm,
    "oxide",
    "1",
    100,
    "Tile-IR W4A4 GEMM cubins (sm_100 / sm_120)",
)];

static CUDNN: [KernelImpl; 3] = [
    k(
        KernelOp::DenseAttention,
        "cudnn",
        "cudnn",
        80,
        "cuDNN fused SDPA, bf16 out (v2 when no engine)",
    ),
    k(KernelOp::Conv3d, "cudnn", "cudnn", 0, "cuDNN conv3d"),
    k(
        KernelOp::Conv3d,
        "cudnn-bf16",
        "cudnn-bf16",
        80,
        "cuDNN conv3d on bf16 (fast mode only)",
    ),
];

static CUBLAS: [KernelImpl; 1] = [k(
    KernelOp::Nvfp4Gemm,
    "cublas",
    "0",
    0,
    "dequantize, then the cuBLAS bf16 GEMM (the default)",
)];

impl KernelBackend for NvccBackend {
    fn provider(&self) -> Provider {
        Provider::Nvcc
    }
    fn kernels(&self) -> &'static [KernelImpl] {
        &NVCC
    }
}

impl KernelBackend for OxideBackend {
    fn provider(&self) -> Provider {
        Provider::Oxide
    }
    fn kernels(&self) -> &'static [KernelImpl] {
        &OXIDE
    }
    /// The cubins are compiled for sm_100 and sm_120 only.
    fn available(&self, k: &KernelImpl, sm: u32) -> bool {
        sm >= k.min_sm && (sm / 10 == 10 || sm / 10 == 12)
    }
}

impl KernelBackend for CudnnBackend {
    fn provider(&self) -> Provider {
        Provider::Cudnn
    }
    fn kernels(&self) -> &'static [KernelImpl] {
        &CUDNN
    }
}

impl KernelBackend for CublasBackend {
    fn provider(&self) -> Provider {
        Provider::Cublas
    }
    fn kernels(&self) -> &'static [KernelImpl] {
        &CUBLAS
    }
}

/// Every registered backend.
pub static BACKENDS: [&dyn KernelBackend; 4] =
    [&NvccBackend, &OxideBackend, &CudnnBackend, &CublasBackend];

/// A resolved `[kernels]` entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelChoice {
    pub op: KernelOp,
    pub provider: Provider,
    pub kernel: KernelImpl,
}

impl KernelChoice {
    pub fn setting(&self) -> (&'static str, &'static str) {
        (self.op.setting(), self.kernel.value)
    }
}

impl fmt::Display for KernelChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}={}:{}",
            self.op.key(),
            self.provider.as_str(),
            self.kernel.id
        )
    }
}

/// Resolve `choice` (`"auto"`, `"<id>"` or `"<provider>:<id>"`) for `op`.
/// `Ok(None)` is `auto` (the dispatch site's own default). `sm`, when
/// known, rejects a kernel the device cannot run.
pub fn resolve(
    op: KernelOp,
    choice: &str,
    sm: Option<u32>,
) -> Result<Option<KernelChoice>, String> {
    let choice = choice.trim().to_ascii_lowercase();
    if choice == "auto" || choice.is_empty() {
        return Ok(None);
    }
    let (want_provider, id) = match choice.split_once(':') {
        Some((p, id)) => (Some(p.to_string()), id.to_string()),
        None => (None, choice.clone()),
    };
    let mut found = Vec::new();
    for b in BACKENDS.iter() {
        if want_provider
            .as_deref()
            .is_some_and(|p| p != b.provider().as_str())
        {
            continue;
        }
        for kimpl in b.kernels().iter().filter(|x| x.op == op && x.id == id) {
            found.push((*b, *kimpl));
        }
    }
    let (backend, kimpl) = match found.as_slice() {
        [] => {
            let known: Vec<String> = BACKENDS
                .iter()
                .flat_map(|b| {
                    b.kernels()
                        .iter()
                        .filter(|x| x.op == op)
                        .map(|x| format!("{}:{}", b.provider().as_str(), x.id))
                })
                .collect();
            return Err(format!(
                "kernels.{} = {choice:?}: no such implementation (auto, {})",
                op.key(),
                known.join(", ")
            ));
        }
        [one] => *one,
        many => {
            return Err(format!(
                "kernels.{} = {choice:?} is ambiguous ({}); write provider:id",
                op.key(),
                many.iter()
                    .map(|(b, _)| b.provider().as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        }
    };
    if let Some(sm) = sm {
        if !backend.available(&kimpl, sm) {
            return Err(format!(
                "kernels.{} = {choice:?}: {}:{} needs sm_{}+ (device sm_{sm})",
                op.key(),
                backend.provider().as_str(),
                kimpl.id,
                kimpl.min_sm
            ));
        }
    }
    Ok(Some(KernelChoice {
        op,
        provider: backend.provider(),
        kernel: kimpl,
    }))
}

/// The dispatch site's read: the setting (env, else profile), lower-cased,
/// else the op's default. Identical to the `string_flag(name, "auto")` it
/// replaces when no profile is installed.
pub fn choice(op: KernelOp) -> String {
    super::settings::var(op.setting())
        .unwrap_or_else(|| op.default_value().to_string())
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choices_resolve_to_their_backend() {
        let c = resolve(KernelOp::DenseAttention, "cudnn", Some(120))
            .unwrap()
            .unwrap();
        assert_eq!(c.provider, Provider::Cudnn);
        assert_eq!(c.setting(), ("FASTVIDEO_FLASH_KERNEL", "cudnn"));
        let c = resolve(KernelOp::SolAttention, "nvcc:x4f", None)
            .unwrap()
            .unwrap();
        assert_eq!(c.setting(), ("FASTVIDEO_SOL_KERNEL", "x4f"));
        let c = resolve(KernelOp::Nvfp4Gemm, "oxide", Some(120))
            .unwrap()
            .unwrap();
        assert_eq!(c.setting(), ("FASTVIDEO_NVFP4_OXIDE_GEMM", "1"));
        assert_eq!(resolve(KernelOp::VsaAttention, "auto", None).unwrap(), None);
    }

    #[test]
    fn bad_choices_are_config_errors() {
        assert!(
            resolve(KernelOp::DenseAttention, "x4f", None).is_err(),
            "wrong op"
        );
        assert!(
            resolve(KernelOp::SolAttention, "ws", Some(86)).is_err(),
            "needs sm90"
        );
        assert!(
            resolve(KernelOp::Nvfp4Gemm, "oxide", Some(90)).is_err(),
            "no sm_90 cubin"
        );
        assert!(
            resolve(KernelOp::Conv3d, "cudnn:unfold", None).is_err(),
            "wrong provider"
        );
        // `cudnn` names a conv3d kernel in one backend only; unambiguous.
        assert!(resolve(KernelOp::Conv3d, "cudnn", None).unwrap().is_some());
    }

    #[test]
    fn every_op_has_a_setting_and_a_key() {
        for op in KernelOp::ALL {
            assert_eq!(KernelOp::from_key(op.key()), Some(op));
            assert!(op.setting().starts_with("FASTVIDEO_"));
        }
    }
}
