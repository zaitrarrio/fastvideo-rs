//! Why a boot stopped. The message is what the `FAILED:` log line and the
//! `failed` report say; each variant keeps the wording of the shell boot it
//! replaces where fv-control or an operator reads it (ADR-0014 §5).

use super::sys::SysError;

#[derive(Debug, thiserror::Error)]
pub enum BootError {
    /// The host's NVIDIA driver is older than the image needs; the message
    /// starts with `driver_too_old: ` (fv-control's podlogs.ts matches it).
    #[error("{0}")]
    DriverTooOld(String),
    /// The worker config: missing, undecodable, or refused by `fv-serve
    /// --print-config`.
    #[error("{0}")]
    Config(String),
    /// A weights step (fetch, check, aux file).
    #[error("{0}")]
    Weights(String),
    /// A weights source this image cannot serve yet.
    #[error("weights_source {name}: not available yet: {why}")]
    SourceNotAvailable {
        name: &'static str,
        why: &'static str,
    },
    /// `FV_WEIGHTS_SOURCE` names no source.
    #[error("FV_WEIGHTS_SOURCE={0}: not one of volume, r2, hub, none, registry")]
    UnknownSource(String),
    /// The tunnel of a GMI / Brev pod.
    #[error("{0}")]
    Tunnel(String),
    /// A variable the step needs is missing or malformed.
    #[error("{0}")]
    Env(String),
    /// The machine (a program, a file, HTTP).
    #[error(transparent)]
    Sys(#[from] SysError),
}

impl BootError {
    /// The failure's own pause before the exit (`FV_DRIVER_FAIL_SLEEP_S` for
    /// the driver check, else the boot's fail sleep).
    pub fn is_driver(&self) -> bool {
        matches!(self, Self::DriverTooOld(_))
    }
}
