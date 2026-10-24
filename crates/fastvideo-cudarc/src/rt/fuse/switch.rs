//! The fusion switches of the three families as one implementation, and the
//! value a load plan installs (ADR-0017 "How B4 and a lazy cold start plug
//! in", `docs/adr/0022-block-fusion.md` "The switch").
//!
//! A switch resolves, innermost first:
//!
//! 1. a forced value ([`with_forced`]: the parity checks of `fv-gpucheck`,
//!    the old `with_fuse` / `with_glue`);
//! 2. the value the running pipeline's plan installed ([`installed`]: the
//!    H3 and LTX-2 DiT forwards run inside it);
//! 3. the setting itself, read once per process through the layered reader
//!    (`envflag::bool_flag`, default on), as the old cached getters did.
//!
//! (2) and (3) agree wherever a pipeline runs: the plan reads the same
//! setting from its load's snapshot, every process sets these names before
//! a load (CLI flags, `fv-gpucheck --mode`, fv-serve's process plan), and
//! fv-serve refuses to serve models whose process-wide settings differ
//! (`ProcessPlan::for_models`). `FASTVIDEO_H3_GLUE` applies only while
//! `FASTVIDEO_H3_FUSE` is on.

use std::cell::Cell;

use fastvideo_models::techniques::settings::Source;

use crate::rt::envflag::{bool_flag, CachedBool};

/// One fusion switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Switch {
    /// `FASTVIDEO_H3_FUSE`: H3's Phase 3c fusions (the head merge into
    /// `to_out`'s MXFP8 activation, a block's last residual with the next
    /// block's norm, the f32 attention output's merge).
    H3,
    /// `FASTVIDEO_H3_GLUE` (needs [`Switch::H3`]): H3's glue fusions around
    /// the GEMMs and attention.
    H3Glue,
    /// `FASTVIDEO_LTX_FUSE`: every LTX-2 block fusion.
    Ltx,
    /// `FASTVIDEO_WAN_FUSE`: Wan's residual + norm as one kernel (off: the
    /// same math as two).
    Wan,
}

impl Switch {
    pub const ALL: [Switch; 4] = [Switch::H3, Switch::H3Glue, Switch::Ltx, Switch::Wan];

    /// The setting (a `Layered` row of the settings table, default on).
    pub const fn setting(self) -> &'static str {
        match self {
            Switch::H3 => "FASTVIDEO_H3_FUSE",
            Switch::H3Glue => "FASTVIDEO_H3_GLUE",
            Switch::Ltx => "FASTVIDEO_LTX_FUSE",
            Switch::Wan => "FASTVIDEO_WAN_FUSE",
        }
    }

    const fn index(self) -> usize {
        match self {
            Switch::H3 => 0,
            Switch::H3Glue => 1,
            Switch::Ltx => 2,
            Switch::Wan => 3,
        }
    }

    /// The process value: the setting, read on first use and kept.
    fn process(self) -> bool {
        static CACHE: [CachedBool; 4] = [
            CachedBool::new(),
            CachedBool::new(),
            CachedBool::new(),
            CachedBool::new(),
        ];
        // The literal names keep each setting's readers greppable and let
        // the settings table's checks see the defaults (ADR-0010).
        CACHE[self.index()].get_or_init(|| match self {
            Switch::H3 => bool_flag("FASTVIDEO_H3_FUSE", true),
            Switch::H3Glue => bool_flag("FASTVIDEO_H3_GLUE", true),
            Switch::Ltx => bool_flag("FASTVIDEO_LTX_FUSE", true),
            Switch::Wan => bool_flag("FASTVIDEO_WAN_FUSE", true),
        })
    }
}

/// The fusion switches a load plan decided: `Some(on)` for each switch the
/// plan read from its settings snapshot, `None` where it leaves the process
/// value. The fuse stage installs it on the pipeline's DiT, whose forwards
/// run inside [`installed`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockFusion {
    values: [Option<bool>; 4],
}

impl BlockFusion {
    /// H3's switches from a settings source: `FASTVIDEO_H3_FUSE` and
    /// `FASTVIDEO_H3_GLUE` as set (the glue applies only with the fusions on).
    pub fn h3(settings: &dyn Source) -> Self {
        Self::default()
            .with(Switch::H3, settings.bool_flag("FASTVIDEO_H3_FUSE", true))
            .with(
                Switch::H3Glue,
                settings.bool_flag("FASTVIDEO_H3_GLUE", true),
            )
    }

    /// LTX-2's switch from a settings source (`FASTVIDEO_LTX_FUSE`).
    pub fn ltx(settings: &dyn Source) -> Self {
        Self::default().with(Switch::Ltx, settings.bool_flag("FASTVIDEO_LTX_FUSE", true))
    }

    /// This value with `switch` decided.
    pub const fn with(mut self, switch: Switch, on: bool) -> Self {
        self.values[switch.index()] = Some(on);
        self
    }

    /// What the plan decided for `switch` (not resolved against another).
    pub const fn get(&self, switch: Switch) -> Option<bool> {
        self.values[switch.index()]
    }

    /// Whether `switch` is on under this value alone (`H3Glue` needs `H3`;
    /// an undecided switch reads as its default, on). The load plans'
    /// descriptions print these.
    pub fn on(&self, switch: Switch) -> bool {
        let raw = |s: Switch| self.get(s).unwrap_or(true);
        match switch {
            Switch::H3Glue => raw(Switch::H3) && raw(Switch::H3Glue),
            s => raw(s),
        }
    }
}

thread_local! {
    static FORCED: Cell<[Option<bool>; 4]> = const { Cell::new([None; 4]) };
    static INSTALLED: Cell<BlockFusion> = const { Cell::new(BlockFusion { values: [None; 4] }) };
}

fn raw(switch: Switch) -> bool {
    let i = switch.index();
    if let Some(v) = FORCED.with(|c| c.get()[i]) {
        return v;
    }
    if let Some(v) = INSTALLED.with(|c| c.get().values[i]) {
        return v;
    }
    switch.process()
}

/// Whether `switch` is on for this call (forced, else installed, else the
/// process setting; [`Switch::H3Glue`] only while [`Switch::H3`] is on).
pub fn enabled(switch: Switch) -> bool {
    match switch {
        Switch::H3Glue => raw(Switch::H3) && raw(Switch::H3Glue),
        s => raw(s),
    }
}

/// Run `f` with `switch` forced on or off on this thread (parity checks).
pub fn with_forced<R>(switch: Switch, on: bool, f: impl FnOnce() -> R) -> R {
    let i = switch.index();
    let prev = FORCED.with(|c| {
        let old = c.get();
        let mut new = old;
        new[i] = Some(on);
        c.set(new);
        old[i]
    });
    let out = f();
    FORCED.with(|c| {
        let mut v = c.get();
        v[i] = prev;
        c.set(v);
    });
    out
}

/// Run `f` under a plan's switches on this thread (`None`: no plan, the
/// process values). Forced values still win.
pub fn installed<R>(value: Option<BlockFusion>, f: impl FnOnce() -> R) -> R {
    let Some(value) = value else { return f() };
    let prev = INSTALLED.with(|c| c.replace(value));
    let out = f();
    INSTALLED.with(|c| c.set(prev));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_beats_installed_and_restores() {
        let plan = BlockFusion::default().with(Switch::Ltx, false);
        installed(Some(plan), || {
            assert!(!enabled(Switch::Ltx));
            with_forced(Switch::Ltx, true, || assert!(enabled(Switch::Ltx)));
            assert!(!enabled(Switch::Ltx));
        });
    }

    #[test]
    fn glue_needs_the_fusions() {
        let plan = BlockFusion::default()
            .with(Switch::H3, false)
            .with(Switch::H3Glue, true);
        installed(Some(plan), || {
            assert!(!enabled(Switch::H3Glue));
            // The glue's own value stands once the fusions are forced on.
            with_forced(Switch::H3, true, || assert!(enabled(Switch::H3Glue)));
        });
        assert!(!plan.on(Switch::H3Glue));
        assert!(BlockFusion::default().on(Switch::H3Glue));
    }

    #[test]
    fn an_undecided_switch_keeps_the_process_value() {
        let plan = BlockFusion::default().with(Switch::H3, false);
        let wan = enabled(Switch::Wan);
        installed(Some(plan), || assert_eq!(enabled(Switch::Wan), wan));
    }
}
