//! `fv-serve`: one Rust server for the FastVideo, MiniMax, fal, LTX and Reactor
//! APIs over the fastvideo-rs engines (docs/serve/design.md).
//!
//! Owned by WP-10 (docs/serve/design.md §8). Scaffolded by WP-00: the binary
//! builds and exits with an error until WP-10 fills it in.

mod config;
mod health;
mod native;
mod router;
mod shutdown;

use std::process::ExitCode;

fn main() -> ExitCode {
    eprintln!(
        "fv-serve {}: not implemented yet (scaffold only; see docs/serve/design.md WP-10)",
        env!("CARGO_PKG_VERSION")
    );
    ExitCode::from(2)
}
