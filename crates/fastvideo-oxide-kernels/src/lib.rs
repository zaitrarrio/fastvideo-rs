//! Tile-IR W4A4 GEMM for NVFP4 (cutile-rs v0.3.1 `examples/nvfp4.rs`).
//!
//! cutile-rs normally JIT-compiles a kernel's captured AST through CUDA Tile
//! IR and `tileiras` on first launch. We do that step ahead of time instead:
//! `fv-oxide-aot` (src/main.rs) runs cutile's compile-only
//! [`cutile::compile_api::KernelCompiler`] with fixed generics, strides and
//! divisibility facts for each target SM, hands the bytecode to `tileiras`,
//! and writes plain cubins that `fastvideo-cudarc/build.rs` embeds. The
//! runtime image therefore needs neither `tileiras` nor this crate: cudarc
//! loads the cubin and launches the entry with the ABI documented on
//! [`Variant`].
//!
//! Default off in model paths: `FASTVIDEO_NVFP4_OXIDE_GEMM`.

#[cutile::module]
pub mod nvfp4_w4a4 {
    use cutile::core::*;

    /// `z[M, N] = alpha * x[M, K] @ y[N, K]ᵀ` with packed E2M1 + E4M3 block
    /// scales, FP32 accumulate, `E` (f32 or bf16) out.
    ///
    /// Layout matches the host recipe (`fastvideo_models::nvfp4`):
    /// - `x` / `y`: `f4e2m1fnx2[M|N, K/2]`, low nibble first
    /// - scales: `f8e4m3fn[M|N, K/16]`, plain row-major (not swizzled)
    /// - `alpha`: product of the two tensor-level decode factors
    #[cutile::entry()]
    fn nvfp4_oxide_w4a4_gemm<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const BK_PACKED: i32,
        const BK_SCALES: i32,
    >(
        z: &mut Tensor<E, { [BM, BN] }>,
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
        let mut tile_z: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for k_tile in 0i32..num_tiles(&part_x, 1) {
            let tile_x = part_x.load([pid.0, k_tile]).unpack(shape![BM, BK]);
            let tile_y = part_y
                .load([pid.1, k_tile])
                .unpack(shape![BN, BK])
                .transpose();
            let tile_x_scales = part_x_scales.load([pid.0, k_tile]);
            let tile_y_scales = part_y_scales.load([pid.1, k_tile]).transpose();
            tile_z = mmaf_scaled(tile_x, tile_y, tile_z, tile_x_scales, tile_y_scales);
        }
        let alpha_tile: Tile<f32, { [BM, BN] }> = broadcast_scalar(alpha, shape![BM, BN]);
        let scaled: Tile<f32, { [BM, BN] }> = tile_z * alpha_tile;
        let out: Tile<E, { [BM, BN] }> = convert_tile(scaled);
        z.store(out);
    }
}

pub use nvfp4_w4a4::nvfp4_oxide_w4a4_gemm;

/// One compiled specialization. Every variant shares one launch ABI (cutile
/// v0.3.1 entry lowering, `kernel_entry_generator.rs` + `tile_kernel.rs`):
///
/// ```text
/// z:        u64 ptr, i32 dims[2], i32 strides[2], i32 part_dims[2], i32 part_strides[2]
/// x, y:     u64 ptr, i32 dims[2], i32 strides[2]          (packed, [rows, K/2])
/// x_s, y_s: u64 ptr, i32 dims[2], i32 strides[2]          (E4M3, [rows, K/16])
/// alpha:    f32
/// grid = (M/BM, N/BN, 1), block = (1, 1, 1), no dynamic shared memory
/// ```
///
/// Compiled with every dim and leading stride divisible by 16 and every base
/// pointer 16-byte aligned, so launches must satisfy `M % BM == 0`,
/// `N % BN == 0`, `K % 256 == 0` (K/16 scale columns divisible by 16).
#[derive(Debug, Clone, Copy)]
pub struct Variant {
    pub out: &'static str,
    pub bm: i32,
    pub bn: i32,
    pub bk: i32,
}

impl Variant {
    pub fn generics(&self) -> Vec<String> {
        vec![
            self.out.to_string(),
            self.bm.to_string(),
            self.bn.to_string(),
            self.bk.to_string(),
            (self.bk / 2).to_string(),
            (self.bk / 16).to_string(),
        ]
    }

    /// File stem, e.g. `nvfp4_w4a4_sm120_bf16_128x128x128`.
    pub fn stem(&self, sm: u32) -> String {
        format!(
            "nvfp4_w4a4_sm{sm}_{}_{}x{}x{}",
            self.out, self.bm, self.bn, self.bk
        )
    }
}

/// What CI compiles for each SM. bf16 out is the benchmark shape; one
/// f32-out variant backs `nvfp4_oxide_gemm_device` (f32 activations).
pub const VARIANTS: &[Variant] = &[
    Variant {
        out: "bf16",
        bm: 128,
        bn: 128,
        bk: 128,
    },
    Variant {
        out: "bf16",
        bm: 128,
        bn: 256,
        bk: 128,
    },
    Variant {
        out: "bf16",
        bm: 256,
        bn: 128,
        bk: 128,
    },
    Variant {
        out: "bf16",
        bm: 128,
        bn: 128,
        bk: 256,
    },
    Variant {
        out: "f32",
        bm: 128,
        bn: 128,
        bk: 128,
    },
];

/// Target SMs (datacenter and RTX/workstation Blackwell).
pub const SMS: &[u32] = &[100, 120];
