//! `reactor_wire.v1` codegen (feature `proto-codegen`).
//!
//! The default build compiles the committed bindings in
//! `src/pb/reactor_wire.v1.rs`. With `proto-codegen` the vendored protos in
//! `proto/` (Apache-2.0, see `proto/LICENSE` and `proto/NOTICE`) are compiled
//! with protox + prost-build into `OUT_DIR`, and the test
//! `pb::tests::committed_bindings_match_codegen` checks that the committed
//! file is byte-identical to the fresh output.

fn main() {
    #[cfg(feature = "proto-codegen")]
    codegen();
}

#[cfg(feature = "proto-codegen")]
fn codegen() {
    println!("cargo:rerun-if-changed=proto");
    let fds = protox::compile(
        [
            "reactor_wire/v1/common.proto",
            "reactor_wire/v1/control.proto",
            "reactor_wire/v1/data.proto",
            "reactor_wire/v1/model.proto",
            "reactor_wire/v1/platform.proto",
            "reactor_wire/v1/track.proto",
        ],
        ["proto"],
    )
    .expect("compiling the vendored reactor_wire protos");
    let mut cfg = prost_build::Config::new();
    // As the SDK's own build does (reactor §2): Struct is prost-types'.
    cfg.extern_path(".google.protobuf.Struct", "::prost_types::Struct");
    cfg.compile_fds(fds)
        .expect("generating reactor_wire.v1 bindings");
}
