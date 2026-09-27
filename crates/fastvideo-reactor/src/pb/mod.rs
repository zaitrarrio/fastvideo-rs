//! `reactor_wire.v1` prost bindings (reactor §2).
//!
//! Generated from `proto/reactor_wire/v1/*.proto` (copied verbatim from
//! reactor-team/reactor-runtime, Apache-2.0; see `proto/LICENSE` and
//! `proto/NOTICE`). The default build uses the committed file; the
//! `proto-codegen` feature regenerates it and checks the two agree.

#![allow(clippy::all, clippy::pedantic, missing_docs)]

#[cfg(not(feature = "proto-codegen"))]
include!("reactor_wire.v1.rs");

#[cfg(feature = "proto-codegen")]
include!(concat!(env!("OUT_DIR"), "/reactor_wire.v1.rs"));

#[cfg(all(test, feature = "proto-codegen"))]
mod tests {
    #[test]
    fn committed_bindings_match_codegen() {
        let fresh = include_str!(concat!(env!("OUT_DIR"), "/reactor_wire.v1.rs"));
        let committed = include_str!("reactor_wire.v1.rs");
        assert_eq!(
            fresh, committed,
            "src/pb/reactor_wire.v1.rs is stale: copy OUT_DIR/reactor_wire.v1.rs over it"
        );
    }
}
