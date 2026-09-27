//! Which commands a session accepts right now (design §5.5, §5.7).
//!
//! [`valid_commands`] is a port of fast-h3's `fasth3_session_rules.py`: the
//! clip-session command set is derived purely from four state fields, and
//! `state_update.valid_commands` carries the result so a front-end greys out
//! controls from the snapshot instead of re-deriving the rules.
//! [`CAUSAL_COMMANDS`] is the causal (SF-Wan) set, which has no state rules.

/// Always available: they only record a value or report one.
const ALWAYS: [&str; 6] = [
    "get_queue",
    "get_state",
    "reset",
    "set_autoplay",
    "set_clip_seconds",
    "set_seed",
];

/// Every clip-session command (fast-h3 verbatim), sorted.
pub const CLIP_COMMANDS: [&str; 12] = [
    "enqueue",
    "get_queue",
    "get_state",
    "move",
    "play",
    "pop",
    "reset",
    "set_autoplay",
    "set_canvas",
    "set_clip_seconds",
    "set_seed",
    "stop",
];

/// The causal command set (Waypoint-style `InputState` setters, design §5.7).
pub const CAUSAL_COMMANDS: [&str; 5] = ["get_state", "reset", "set_paused", "set_prompt", "set_seed"];

/// The clip commands the session accepts in this state, sorted.
///
/// - `playing`: a clip is streaming (or armed to).
/// - `generation_queued` / `generation_capacity`: the generation queue.
/// - `playout_queued`: built clips waiting.
pub fn valid_commands(
    playing: bool,
    generation_queued: usize,
    generation_capacity: usize,
    playout_queued: usize,
) -> Vec<&'static str> {
    let mut c: Vec<&'static str> = ALWAYS.to_vec();
    if generation_queued < generation_capacity {
        c.push("enqueue");
    }
    if generation_queued > 0 || playout_queued > 0 {
        c.push("pop");
        c.push("move");
    }
    if playing {
        c.push("stop");
    } else {
        if playout_queued > 0 {
            c.push("play");
        }
        // The canvas fixes the shape queued clips are built at.
        if generation_queued == 0 && playout_queued == 0 {
            c.push("set_canvas");
        }
    }
    c.sort_unstable();
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mirrors fast-h3 tests/test_fasth3.py (the session-rules block).
    #[test]
    fn an_empty_idle_session_can_only_enqueue_and_configure() {
        assert_eq!(
            valid_commands(false, 0, 20, 0),
            [
                "enqueue",
                "get_queue",
                "get_state",
                "reset",
                "set_autoplay",
                "set_canvas",
                "set_clip_seconds",
                "set_seed"
            ]
        );
    }

    #[test]
    fn a_built_clip_makes_play_valid() {
        let c = valid_commands(false, 0, 20, 1);
        assert!(c.contains(&"play"));
        assert!(c.contains(&"pop") && c.contains(&"move"));
        assert!(!c.contains(&"set_canvas"));
        assert!(!c.contains(&"stop"));
    }

    #[test]
    fn playing_offers_stop_and_locks_the_canvas() {
        let c = valid_commands(true, 0, 20, 0);
        assert!(c.contains(&"stop"));
        assert!(!c.contains(&"play"));
        assert!(!c.contains(&"set_canvas"));
    }

    #[test]
    fn a_full_generation_queue_refuses_enqueue() {
        assert!(!valid_commands(false, 20, 20, 0).contains(&"enqueue"));
    }

    #[test]
    fn conditions_and_reads_are_always_available() {
        for s in [
            valid_commands(true, 20, 20, 10),
            valid_commands(false, 0, 20, 0),
        ] {
            for a in ALWAYS {
                assert!(s.contains(&a));
            }
        }
    }

    #[test]
    fn every_valid_command_is_a_clip_command() {
        for s in [valid_commands(true, 3, 20, 2), valid_commands(false, 0, 20, 0)] {
            for c in s {
                assert!(CLIP_COMMANDS.contains(&c));
            }
        }
    }
}
