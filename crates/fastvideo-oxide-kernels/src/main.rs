//! Cubin-export stub. `cargo oxide build --arch sm_100,sm_120` compiles the
//! Tile-IR module in [`fastvideo_oxide_kernels`]. This binary is not run on
//! the host; oxide.sh copies the emitted cubins.

fn main() {
    let _ = fastvideo_oxide_kernels::nvfp4_oxide_w4a4_gemm;
}
