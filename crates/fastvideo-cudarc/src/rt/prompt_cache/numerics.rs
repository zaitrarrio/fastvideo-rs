//! Key fields for *how* an encoder computes, not what weights it reads
//! (ADR-0021 "B4c2"): the GEMM math, the GPU architecture and the activation
//! precision a run actually uses ([`Numerics`]), and which weights the
//! encoder's linears were read from at which precision ([`WeightsRead`]).
//!
//! The rule (B4c2): a setting that can change the encoded tensors is in the
//! key. A spurious miss costs one encode; a wrong hit is silent wrong output.
//! Every value here is the *resolved* one — the device context's GEMM math
//! (built once from `FASTVIDEO_BF16`, `FASTVIDEO_TF32` and the card), the
//! card's compute capability, and [`crate::rt::tensor::bf16_activations`]
//! (`FASTVIDEO_BF16_ACT`, else the pipeline default, else f32) — never an
//! environment variable read again at key time.

/// The cuBLAS math F32 GEMMs run with (`rt::device::GemmMath`), or `Host`
/// when ops run on the CPU (no live device, or residency off).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemm {
    Host,
    F32,
    Tf32,
    Bf16,
}

impl Gemm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::F32 => "f32",
            Self::Tf32 => "tf32",
            Self::Bf16 => "bf16",
        }
    }
}

/// What an encode's numbers depend on besides its weights and inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Numerics {
    /// The resolved GEMM math (`FASTVIDEO_BF16`, `FASTVIDEO_TF32` and whether
    /// the card has tensor cores).
    pub gemm: Gemm,
    /// The device's compute capability (`sm_major`, `sm_minor`): kernels,
    /// attention and GEMM paths are chosen by it. `None` on the host.
    pub arch: Option<(i32, i32)>,
    /// bf16 activations ([`crate::rt::tensor::bf16_activations`]).
    pub bf16_act: bool,
}

impl Numerics {
    /// The key field: `gemm=<host|f32|tf32|bf16>;arch=<sm_MAJOR.MINOR|none>;act=<bf16|f32>`.
    pub fn field(&self) -> Vec<u8> {
        let arch = match self.arch {
            Some((major, minor)) => format!("sm_{major}.{minor}"),
            None => "none".to_owned(),
        };
        let act = if self.bf16_act { "bf16" } else { "f32" };
        format!("gemm={};arch={arch};act={act}", self.gemm.as_str()).into_bytes()
    }

    /// The numerics ops on this thread run with *now*: the live device
    /// context's GEMM math and compute capability when ops are expected on
    /// the device, else the host; and the activation precision the ops read.
    /// Called right before the key is made, in the same call that encodes on
    /// a miss, so the key describes the encode it stands for.
    pub fn live() -> Self {
        let bf16_act = crate::rt::tensor::bf16_activations();
        #[cfg(feature = "cuda")]
        {
            use crate::rt::device::GemmMath;
            if crate::rt::stats::device_expected() {
                if let Some(dev) = crate::rt::device::global_device() {
                    return Self {
                        gemm: match dev.gemm_math {
                            GemmMath::F32 => Gemm::F32,
                            GemmMath::Tf32 => Gemm::Tf32,
                            GemmMath::Bf16 => Gemm::Bf16,
                        },
                        arch: Some((dev.sm_major, dev.sm_minor)),
                        bf16_act,
                    };
                }
            }
        }
        Self {
            gemm: Gemm::Host,
            arch: None,
            bf16_act,
        }
    }
}

/// Which weights an encoder's linears were read from, and at what precision
/// (gaps 1 and 6): the checkpoint at its own precision, weight-only FP8 rows
/// quantised from the checkpoint at load, or the pre-quantised FP8 tree
/// (`text_encoder_fp8/`, `llm::prequant`). The tree is meant to equal the
/// load-time quantisation; until that is proven bit for bit the two are
/// keyed apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightsRead {
    Checkpoint,
    Fp8RowsAtLoad,
    Fp8RowsTree,
    /// FP8 numbers from an encoder that does not say where its rows came
    /// from (never equal to a known source).
    Fp8RowsUnknown,
}

impl WeightsRead {
    /// The key field.
    pub fn field(self) -> &'static [u8] {
        match self {
            Self::Checkpoint => b"weights=checkpoint",
            Self::Fp8RowsAtLoad => b"weights=fp8-rows/at-load",
            Self::Fp8RowsTree => b"weights=fp8-rows/tree",
            Self::Fp8RowsUnknown => b"weights=fp8-rows/unknown",
        }
    }
}

/// Where the encoder's forward ran: a decoder kept on the device, or layer
/// by layer (streamed). Equal numbers are expected, not proven (LTX gap 5,
/// H3 resident vs streamed FP8), so the two are keyed apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextMode {
    Resident,
    Streamed,
}

impl TextMode {
    /// The key field.
    pub fn field(self) -> &'static [u8] {
        match self {
            Self::Resident => b"mode=resident",
            Self::Streamed => b"mode=streamed",
        }
    }
}

/// How an encode runs, beyond which encoder it is (ADR-0021 "B4c2"): every
/// value is what the run resolved, not what was asked for. H3 and LTX put
/// these four after their encoder identity, in this order.
#[derive(Debug, Clone, Copy)]
pub struct EncodeSettings<'a> {
    /// The encoder's linears: the checkpoint, FP8 rows quantised at load, or
    /// the pre-quantised `text_encoder_fp8/` tree (gaps 1 and 6).
    pub weights: WeightsRead,
    /// The tree's file identity when the tree is what was read (gap 1).
    pub tree: Option<&'a [u8; 32]>,
    pub mode: TextMode,
    /// GEMM math, card and activations (gaps 3 and 4).
    pub numerics: Numerics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_setting_moves_the_numerics_field() {
        let base = Numerics {
            gemm: Gemm::Bf16,
            arch: Some((9, 0)),
            bf16_act: true,
        };
        assert_eq!(base.field(), b"gemm=bf16;arch=sm_9.0;act=bf16");
        assert_eq!(base.field(), base.field());
        let host = Numerics {
            gemm: Gemm::Host,
            arch: None,
            bf16_act: false,
        };
        assert_eq!(host.field(), b"gemm=host;arch=none;act=f32");
        let others = [
            Numerics {
                gemm: Gemm::Tf32,
                ..base
            },
            Numerics {
                gemm: Gemm::F32,
                ..base
            },
            Numerics {
                gemm: Gemm::Host,
                ..base
            },
            Numerics {
                arch: Some((12, 0)),
                ..base
            },
            Numerics {
                arch: Some((9, 1)),
                ..base
            },
            Numerics { arch: None, ..base },
            Numerics {
                bf16_act: false,
                ..base
            },
        ];
        for (i, a) in others.iter().enumerate() {
            assert_ne!(a.field(), base.field(), "{a:?}");
            for b in &others[i + 1..] {
                assert_ne!(a.field(), b.field(), "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn every_weights_source_is_its_own_field() {
        let all = [
            WeightsRead::Checkpoint,
            WeightsRead::Fp8RowsAtLoad,
            WeightsRead::Fp8RowsTree,
            WeightsRead::Fp8RowsUnknown,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a.field(), b.field());
            }
        }
    }

    #[test]
    fn host_numerics_follow_the_activation_override() {
        // No device in unit tests: the host, and whatever activations the
        // ops on this thread would use.
        let on = crate::rt::tensor::with_bf16_act(true, Numerics::live);
        let off = crate::rt::tensor::with_bf16_act(false, Numerics::live);
        assert!(on.bf16_act && !off.bf16_act);
        assert_ne!(on.field(), off.field());
        if !crate::rt::stats::device_expected() {
            assert_eq!(off.gemm, Gemm::Host);
            assert_eq!(off.arch, None);
        }
    }
}
