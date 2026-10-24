//! The NVRTC kernel source as one string, assembled by `build.rs` from the
//! region files (`rt/kernels/*.cu` and the family `kernels/*.cu`, listed in
//! its `KERNEL_PARTS`) in a fixed order. The concatenation is byte for byte
//! the former `src/wan/kernels.cu` (plan B1), which the tests below check on
//! every build, CUDA feature or not: the same bytes nvcc compiled ahead of
//! time are what NVRTC compiles at run time, as before the split.

include!(concat!(env!("OUT_DIR"), "/kernel_src.rs"));

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    /// sha256 of `crates/fastvideo-cudarc/src/wan/kernels.cu` at main
    /// `42da2570` (the single file before the split). Changing a kernel
    /// changes this on purpose; update it in the same commit.
    const KERNELS_CU_SHA256: &str =
        "c9b61cd2b2677292eba5a83b6ec93ba744c69f51de3bec06accbaae9087cc0d3";

    fn sha256_hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// The NVRTC string (the `concat!` of the region files) is the old file.
    #[test]
    fn nvrtc_source_is_the_former_kernels_cu() {
        assert_eq!(sha256_hex(super::KERNEL_SRC.as_bytes()), KERNELS_CU_SHA256);
    }

    /// nvcc's input (`$OUT_DIR/kernels.cu`, written by `build.rs` from the same
    /// list) is the same bytes, so the ahead-of-time path sees what NVRTC sees.
    #[test]
    fn nvcc_input_is_the_same_bytes() {
        let path = concat!(env!("OUT_DIR"), "/kernels.cu");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert_eq!(sha256_hex(&bytes), KERNELS_CU_SHA256);
        assert_eq!(bytes, super::KERNEL_SRC.as_bytes());
    }
}
