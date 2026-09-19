//! VSA-H3 sparse attention (placeholder until milestone 6 lands; the dense
//! path in [`super::transformer`] is what the diffusers oracle judges).

use crate::wan::tensor::{CudaTensor, Result, TensorError};

pub struct H3Vsa {
    _private: (),
}

impl H3Vsa {
    /// `q`, `k`, `v`, `gate`: `[1, H, S, D]` in packed order; returns `[1, H, S, D]`.
    pub fn attend(&self, _q: CudaTensor, _k: CudaTensor, _v: CudaTensor, _gate: Option<CudaTensor>) -> Result<CudaTensor> {
        Err(TensorError::Message("VSA-H3 is not implemented yet".into()))
    }
}
