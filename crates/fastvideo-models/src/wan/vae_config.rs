//! Wan VAE architecture constants (host-only; device graph lives in cudarc).

#[derive(Debug, Clone)]
pub struct WanVaeConfig {
    pub base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub temporal_upsample: Vec<bool>,
    pub load_encoder: bool,
}

impl WanVaeConfig {
    pub fn wan_2_1() -> Self {
        Self {
            base_dim: 96,
            z_dim: 16,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temporal_upsample: vec![true, true, false],
            load_encoder: true,
        }
    }

    pub fn tiny() -> Self {
        Self {
            base_dim: 8,
            z_dim: 4,
            dim_mult: vec![1, 2],
            num_res_blocks: 1,
            temporal_upsample: vec![false],
            load_encoder: false,
        }
    }
}
