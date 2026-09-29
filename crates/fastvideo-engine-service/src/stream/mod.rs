//! Streaming cores (design §5): the session API that the Reactor, fal
//! director and native `/fv/v1/streams` front-ends build on.
//!
//! | Module | What |
//! |---|---|
//! | [`clip`] | [`ClipSession`]: admission and clip builds (`Priority::Stream` jobs pinned to the session's executor) |
//! | [`player`] | [`ClipPlayer`]: fast-h3 queue-and-playout over a clip session: commands, events, lockstep slices, `Continuity` |
//! | [`queue`] | Generation / playout queues ([`ClipQueue`]) and the wire [`ClipInfo`] |
//! | [`rules`] | `valid_commands` (fast-h3 `fasth3_session_rules`) |
//! | [`causal`] | [`CausalSession`] / [`CausalControl`]: SF-Wan block rollout under an exclusive lease; the Reactor causal command set; TTFF |
//! | [`pace`] | Pacers: clip [`MediaItem`]s → `AvPacer` ticks; causal blocks → adaptive `FramePacer` ticks; the drop-oldest [`TickReceiver`] |
//!
//! Typical clip front-end (Reactor fast-h3 mode, fal director):
//!
//! ```text
//! let session = engine.open_clip_session(spec).await?;          // 429/409 busy, 503 not resident
//! let (player, out) = session.into_player(ClipPlayerConfig::default())?;
//! let paced = spawn_clip_pacer(out.media, ClipPacerConfig::for_spec(&spec))?;
//! // commands: player.command(ClipCommand::Enqueue{..}).await? -> Option<reply>
//! // events:   out.events.recv().await -> ClipEvent (broadcast to every client)
//! // media:    paced.ticks.recv().await -> Tick (one frame + 48000/fps samples)
//! ```
//!
//! Causal front-end (Reactor SF-Wan mode, native streams):
//!
//! ```text
//! let session = engine.open_causal_session(spec).await?;
//! let control = session.control();                               // set_prompt / apply(cmd) / ttff
//! control.set_prompt("...");
//! let paced = spawn_causal_pacer(session, CausalPacerConfig::for_spec(&spec))?;
//! ```

pub mod avatar;
pub mod causal;
pub mod clip;
pub mod pace;
pub mod player;
pub mod queue;
pub mod rules;

pub use avatar::{
    AvatarConfig, AvatarEvent, AvatarOutputs, AvatarPlan, AvatarPlayer, AvatarStatus, AvatarTake,
    AvatarWindow, PlanInput, WindowKind, WindowReport,
};
pub use causal::{
    CausalBlock, CausalCommand, CausalControl, CausalReply, CausalSession, CausalState,
    CausalStats, Ttff,
};
pub use clip::{ClipBuild, ClipSession};
pub use pace::{
    spawn_causal_pacer, spawn_clip_pacer, CausalPacerConfig, ClipPacerConfig, IdlePolicy,
    MediaItem, MediaSlice, PaceStats, PacedStream, PlayOutcome, Tick, TickReceiver, TickStart,
};
pub use player::{
    BuildReport, ClipCommand, ClipEvent, ClipOutputs, ClipPlayer, ClipPlayerConfig, ClipState,
};
pub use queue::{BuiltClip, ClipEntry, ClipInfo, ClipQueue, QueueName};
pub use rules::{valid_commands, CAUSAL_COMMANDS, CLIP_COMMANDS};
