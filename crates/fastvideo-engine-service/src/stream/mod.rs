//! Streaming cores: `ClipSession` (clip-queue playout, design §5.5) and
//! `CausalSession` (SF-Wan block rollout, design §5.4).
//!
//! WP-02 provides the engine seams and their fake-backed behaviour; WP-12
//! fills `queue`/`rules` and the fast-h3 semantics of `clip`, and WP-15 the
//! pacer-facing parts of `causal`.

pub mod causal;
pub mod clip;
pub mod queue;
pub mod rules;
