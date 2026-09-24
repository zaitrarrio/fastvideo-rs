//! Tile-IR W4A4 GEMM for NVFP4 (cutile-rs `examples/nvfp4.rs` / cuda-oxide v0.2.1).
//!
//! Built only by `scripts/oxide.sh` via `cargo oxide build --arch sm_100,sm_120`.
//! Not a workspace member — `cargo test` on Mac/stable must not compile this.
//!
//! **Gate (default off):** do not load this cubin unless it beats cuBLAS bf16
//! on the H3 FFN shape (`K=5376`, `N=14336`) **and** PSNR ≥ 30 dB vs bf16.
//! Runtime flag: `FASTVIDEO_NVFP4_OXIDE_GEMM`.

use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::prelude::*;

#[cutile::module]
mod nvfp4_w4a4 {
    use cutile::core::*;

    /// `z[M, N] = x[M, K] @ y[N, K].T` with packed E2M1 + E4M3 block scales.
    ///
    /// Layout matches the host recipe (`fastvideo_models::nvfp4`):
    /// - `x` / `y`: `f4e2m1fnx2[M|N, K/2]`
    /// - scales: `f8e4m3fn[M|N, K/16]`
    #[cutile::entry()]
    fn nvfp4_oxide_w4a4_gemm<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const BK_PACKED: i32,
        const BK_SCALES: i32,
    >(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        y: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        x_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        y_scales: &Tensor<f8e4m3fn, { [-1, -1] }>,
        alpha: f32,
    ) {
        let pid = get_tile_block_id();
        let part_x = x.partition(shape![BM, BK_PACKED]);
        let part_y = y.partition(shape![BN, BK_PACKED]);
        let part_x_scales = x_scales.partition(shape![BM, BK_SCALES]);
        let part_y_scales = y_scales.partition(shape![BN, BK_SCALES]);
        let mut tile_z = constant(0.0f32, shape![BM, BN]);
        for k_tile in 0i32..num_tiles(&part_x, 1) {
            let tile_x = part_x.load([pid.0, k_tile]).unpack(shape![BM, BK]);
            let tile_y = part_y.load([pid.1, k_tile]).unpack(shape![BN, BK]).transpose();
            let tile_x_scales = part_x_scales.load([pid.0, k_tile]);
            let tile_y_scales = part_y_scales.load([pid.1, k_tile]).transpose();
            tile_z = mmaf_scaled(tile_x, tile_y, tile_z, tile_x_scales, tile_y_scales);
        }
        let alpha_tile = broadcast_scalar(alpha, z.shape());
        z.store(tile_z * alpha_tile);
    }
}

pub use nvfp4_w4a4::nvfp4_oxide_w4a4_gemm;
