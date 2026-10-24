//! The load-time settings snapshot (plan B3, `docs/adr/0010-settings-layer.md`
//! "The snapshot passed to `load`").
//!
//! A [`Snapshot`] holds, for every row of [`SETTINGS`], what the two readers
//! would see at the moment it was captured: the process environment's value
//! and the table's (the model scope's, else the active profile's). The H3,
//! LTX-2 and Wan loaders capture one when they start
//! ([`Snapshot::capture`]), take it as a parameter (`load_with_settings`) and
//! read their load-time settings from it; while the loader runs,
//! [`Snapshot::scope`] serves this thread's [`super::var`], [`super::is_set`],
//! [`super::env_var`] and [`super::env_var_os`] calls from it too, so the
//! shared runtime code a load calls (weight loading, quantization, the
//! decoders' constructors) reads the same values without a signature change.
//!
//! Identity with the readers it replaces, by construction:
//!
//! * the accessors ([`Source`]) are the readers' definitions over the captured
//!   pair: [`Source::var`] is the environment's value when it is valid UTF-8,
//!   else the table's (`std::env::var` then `table_get`), [`Source::is_set`]
//!   either present, [`Source::env_var`] / [`Source::env_var_os`] the
//!   environment only (`std::env::var` / `var_os`, including `NotUnicode`);
//!   the typed accessors are the runtime getters' parse rules
//!   ([`super::parse`]) over [`Source::var`];
//! * a value read through a snapshot is the value the reader returned at
//!   capture; it equals the value the reader would return later in the load
//!   unless the environment, the profile or the model scope changes in
//!   between. None of them does during a load: the profile is installed once
//!   per process, the model scope is entered by the caller around the whole
//!   load, and every `set_var` of a `FASTVIDEO_*` name in the workspace runs
//!   before a load starts (CLI flags, `fv-gpucheck --mode`, fv-serve's process
//!   plan and decoder guard) or in tests, except the MLX H3 tree's
//!   `FASTVIDEO_H3_AFFINE`, which goes through [`super::set_env`] and so
//!   updates the running load's scope too;
//! * the scope is per thread, like [`super::scope_model`]: a worker thread a
//!   loader spawns reads the live readers, as it did before.
//!
//! A name that is not a row (none is read, `workspace_check` enforces it) is
//! read live.

use std::cell::RefCell;
use std::env::VarError;
use std::ffi::OsString;
use std::sync::Arc;

use super::table::{Setting, SETTINGS};
use super::{parse, Settings};

/// One row's sources at capture time.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Captured {
    /// The process environment's value (`std::env::var_os`).
    pub env: Option<OsString>,
    /// The table's value: the model scope's, else the active profile's.
    pub table: Option<String>,
}

impl Captured {
    /// [`super::var`]: `std::env::var` (valid UTF-8), else the table.
    pub(super) fn var(&self) -> Option<String> {
        match self.env.as_ref().and_then(|v| v.to_str()) {
            Some(v) => Some(v.to_owned()),
            None => self.table.clone(),
        }
    }

    /// [`super::is_set`]: `var_os` present, or the table has it.
    pub(super) fn is_set(&self) -> bool {
        self.env.is_some() || self.table.is_some()
    }

    /// [`super::env_var`]: exactly `std::env::var`'s result.
    pub(super) fn env_var(&self) -> Result<String, VarError> {
        match &self.env {
            None => Err(VarError::NotPresent),
            Some(v) => v.clone().into_string().map_err(VarError::NotUnicode),
        }
    }

    /// [`super::env_var_os`]: exactly `std::env::var_os`.
    pub(super) fn env_var_os(&self) -> Option<OsString> {
        self.env.clone()
    }
}

/// Where a reader takes a `FASTVIDEO_*` setting from: the live readers
/// ([`Live`]) or a captured [`Snapshot`]. Load-time helpers take a
/// `&dyn Source`, so the same code serves the load (the snapshot) and the
/// callers outside a load (live).
pub trait Source {
    /// The layered read ([`super::var`]): env, else the table.
    fn var(&self, name: &str) -> Option<String>;
    /// [`super::is_set`].
    fn is_set(&self, name: &str) -> bool;
    /// The env-only read ([`super::env_var`]).
    fn env_var(&self, name: &str) -> Result<String, VarError>;
    /// [`super::env_var_os`].
    fn env_var_os(&self, name: &str) -> Option<OsString>;

    /// The runtime's `envflag::bool_flag` ([`parse::flag`] over [`Self::var`]).
    fn bool_flag(&self, name: &str, default: bool) -> bool {
        parse::flag(self.var(name).as_deref(), default)
    }
    /// `envflag::usize_flag` ([`parse::usize_or`]).
    fn usize_flag(&self, name: &str, default: usize) -> usize {
        parse::usize_or(self.var(name).as_deref(), default)
    }
    /// `envflag::f32_flag` ([`parse::f32_or`]).
    fn f32_flag(&self, name: &str, default: f32) -> f32 {
        parse::f32_or(self.var(name).as_deref(), default)
    }
    /// `envflag::f64_flag` ([`parse::f64_or`]).
    fn f64_flag(&self, name: &str, default: f64) -> f64 {
        parse::f64_or(self.var(name).as_deref(), default)
    }
    /// `envflag::string_flag` ([`parse::lower_or`]).
    fn string_flag(&self, name: &str, default: &str) -> String {
        parse::lower_or(self.var(name).as_deref(), default)
    }
}

/// The live readers: [`super::var`], [`super::is_set`], [`super::env_var`],
/// [`super::env_var_os`] at the time of the call (inside a
/// [`Snapshot::scope`], the scoped snapshot's values).
#[derive(Clone, Copy, Debug, Default)]
pub struct Live;

impl Source for Live {
    fn var(&self, name: &str) -> Option<String> {
        super::var(name)
    }
    fn is_set(&self, name: &str) -> bool {
        super::is_set(name)
    }
    fn env_var(&self, name: &str) -> Result<String, VarError> {
        super::env_var(name)
    }
    fn env_var_os(&self, name: &str) -> Option<OsString> {
        super::env_var_os(name)
    }
}

/// Every row's sources, captured once (the plan's `Settings` snapshot).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    /// Parallel to [`SETTINGS`] (sorted by name).
    values: Arc<Vec<Captured>>,
}

thread_local! {
    /// The snapshot this thread's readers serve while a [`Snapshot::scope`]
    /// guard lives.
    static LOAD_SCOPE: RefCell<Option<Snapshot>> = const { RefCell::new(None) };
}

/// Restores the previous [`Snapshot::scope`] when dropped.
#[must_use = "the scope ends when the guard drops"]
pub struct LoadScope(Option<Snapshot>);

impl Drop for LoadScope {
    fn drop(&mut self) {
        let prev = self.0.take();
        LOAD_SCOPE.with(|s| *s.borrow_mut() = prev);
    }
}

fn index(name: &str) -> Option<usize> {
    SETTINGS.binary_search_by(|s| s.name.cmp(name)).ok()
}

/// `f` over this thread's scoped value of `name`: `None` outside a scope or
/// for a name that is not a row (the caller then reads live).
pub(super) fn scoped<T>(name: &str, f: impl FnOnce(&Captured) -> T) -> Option<T> {
    LOAD_SCOPE.with(|s| {
        let s = s.borrow();
        let snap = s.as_ref()?;
        let i = index(name)?;
        Some(f(&snap.values[i]))
    })
}

/// The environment value of `name` in this thread's scoped snapshot, after
/// the process environment changed under a running load
/// ([`super::set_env`]); nothing outside a scope or for a name that is not a
/// row.
pub(super) fn set_scoped_env(name: &str, value: &str) {
    LOAD_SCOPE.with(|s| {
        if let (Some(snap), Some(i)) = (s.borrow_mut().as_mut(), index(name)) {
            Arc::make_mut(&mut snap.values)[i].env = Some(OsString::from(value));
        }
    });
}

impl Snapshot {
    /// What every reader sees now. Inside a [`Snapshot::scope`] (a load that
    /// starts while another one runs on this thread) that is the scoped
    /// snapshot, which is also what the live readers return there.
    pub fn capture() -> Self {
        if let Some(s) = LOAD_SCOPE.with(|s| s.borrow().clone()) {
            return s;
        }
        Self::capture_with(&|name| std::env::var_os(name), &super::table_get)
    }

    /// A snapshot of explicit sources: `env` stands for the process
    /// environment, `table` for the model scope's or profile's table. Tests,
    /// dry runs and a caller that builds a model's settings without touching
    /// the process (`Snapshot::from_parts(&|k| env.get(k).cloned(), &own)`).
    pub fn from_parts(env: &dyn Fn(&str) -> Option<OsString>, table: &Settings) -> Self {
        Self::capture_with(env, &|name| table.get(name).map(str::to_owned))
    }

    fn capture_with(
        env: &dyn Fn(&str) -> Option<OsString>,
        table: &dyn Fn(&str) -> Option<String>,
    ) -> Self {
        Self {
            values: Arc::new(
                SETTINGS
                    .iter()
                    .map(|s| Captured {
                        env: env(s.name),
                        table: table(s.name),
                    })
                    .collect(),
            ),
        }
    }

    /// While the guard lives, this thread's readers ([`super::var`],
    /// [`super::is_set`], [`super::env_var`], [`super::env_var_os`]) return
    /// this snapshot's values for every row. Nested scopes restore the outer
    /// one on drop.
    pub fn scope(&self) -> LoadScope {
        LoadScope(LOAD_SCOPE.with(|s| s.borrow_mut().replace(self.clone())))
    }

    /// The captured sources of `name`, if it is a row.
    pub fn get(&self, name: &str) -> Option<&Captured> {
        index(name).map(|i| &self.values[i])
    }

    /// Every row with its captured sources, in table order.
    pub fn rows(&self) -> impl Iterator<Item = (&'static Setting, &Captured)> {
        SETTINGS.iter().zip(self.values.iter())
    }

    /// The rows whose sources differ between two snapshots.
    pub fn diff(&self, other: &Snapshot) -> Vec<&'static str> {
        self.rows()
            .zip(other.values.iter())
            .filter(|((_, a), b)| a != b)
            .map(|((s, _), _)| s.name)
            .collect()
    }
}

impl Source for Snapshot {
    fn var(&self, name: &str) -> Option<String> {
        match self.get(name) {
            Some(c) => c.var(),
            None => super::var(name),
        }
    }
    fn is_set(&self, name: &str) -> bool {
        match self.get(name) {
            Some(c) => c.is_set(),
            None => super::is_set(name),
        }
    }
    fn env_var(&self, name: &str) -> Result<String, VarError> {
        match self.get(name) {
            Some(c) => c.env_var(),
            None => super::env_var(name),
        }
    }
    fn env_var_os(&self, name: &str) -> Option<OsString> {
        match self.get(name) {
            Some(c) => c.env_var_os(),
            None => super::env_var_os(name),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::super::table::{DefaultValue, Parse, Reader};
    use super::*;

    /// Two rows nothing in the workspace reads (`Reader::Reserved`), so a
    /// test can set them in the process environment without racing a reader.
    const FREE_A: &str = "FASTVIDEO_NVFP4_COSMOS_STEPS";
    const FREE_B: &str = "FASTVIDEO_NVFP4_LTX_VIDEO_FFN";

    /// Serializes the tests here that set [`FREE_A`] / [`FREE_B`].
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn empty() -> Snapshot {
        Snapshot::from_parts(&|_| None, &Settings::default())
    }

    fn of(env: &[(&str, &str)], table: &[(&str, &str)]) -> Snapshot {
        let env: BTreeMap<String, OsString> = env
            .iter()
            .map(|(k, v)| (k.to_string(), OsString::from(v)))
            .collect();
        let mut t = Settings::default();
        for (k, v) in table {
            t.set(k, v, "test").unwrap_or_else(|e| panic!("{e}"));
        }
        Snapshot::from_parts(&|k| env.get(k).cloned(), &t)
    }

    #[test]
    fn the_free_rows_are_reserved() {
        for name in [FREE_A, FREE_B] {
            let row = super::super::lookup(name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(row.reader, Reader::Reserved, "{name}");
        }
    }

    /// Off-identity: with no `FASTVIDEO_*` set and no profile, every row of
    /// the snapshot is empty and every reader gets what it gets today (no
    /// value; a typed row its documented default).
    #[test]
    fn an_empty_snapshot_is_todays_defaults_field_for_field() {
        let snap = empty();
        let mut typed = 0;
        for (row, c) in snap.rows() {
            assert_eq!(c, &Captured::default(), "{}", row.name);
            assert_eq!(snap.var(row.name), None, "{}", row.name);
            assert!(!snap.is_set(row.name), "{}", row.name);
            assert!(
                matches!(snap.env_var(row.name), Err(VarError::NotPresent)),
                "{}",
                row.name
            );
            assert_eq!(snap.env_var_os(row.name), None, "{}", row.name);
            let DefaultValue::Value(v) = row.default else {
                continue;
            };
            let set = of(&[(row.name, v)], &[]);
            let same = match row.parse {
                Parse::Flag => {
                    snap.bool_flag(row.name, v == "1") == set.bool_flag(row.name, v == "1")
                }
                Parse::Usize => {
                    let d = v.parse().unwrap_or(0);
                    snap.usize_flag(row.name, d) == set.usize_flag(row.name, d)
                }
                Parse::F32 => {
                    let d = v.parse().unwrap_or(0.0);
                    snap.f32_flag(row.name, d) == set.f32_flag(row.name, d)
                }
                Parse::F64 => {
                    let d = v.parse().unwrap_or(0.0);
                    snap.f64_flag(row.name, d) == set.f64_flag(row.name, d)
                }
                Parse::Keyword => snap.string_flag(row.name, v) == set.string_flag(row.name, v),
                Parse::Text => continue,
            };
            assert!(
                same,
                "{}: unset and {v:?} differ through the snapshot",
                row.name
            );
            typed += 1;
        }
        assert!(typed > 80, "only {typed} typed rows checked");
    }

    /// The accessors are the readers' definitions over the captured pair:
    /// env > table > nothing for the layered read, env only for `env_var`.
    #[test]
    fn accessors_follow_the_readers_precedence() {
        let n = "FASTVIDEO_H3_QUANT";
        let only_table = of(&[], &[(n, "mxfp8")]);
        assert_eq!(only_table.var(n).as_deref(), Some("mxfp8"));
        assert!(only_table.is_set(n));
        assert!(only_table.env_var(n).is_err());
        assert_eq!(only_table.env_var_os(n), None);
        let both = of(&[(n, "w8a8")], &[(n, "mxfp8")]);
        assert_eq!(both.var(n).as_deref(), Some("w8a8"));
        assert_eq!(both.env_var(n).as_deref().ok(), Some("w8a8"));
        // A default-on flag: the table's `0` turns it off, the env's `1` on.
        let f = "FASTVIDEO_BF16_ACT";
        assert!(!of(&[], &[(f, "0")]).bool_flag(f, true));
        assert!(of(&[(f, "1")], &[(f, "0")]).bool_flag(f, true));
        assert!(empty().bool_flag(f, true));
        assert_eq!(of(&[], &[]).diff(&of(&[(n, "x")], &[])), vec![n]);
    }

    #[cfg(unix)]
    #[test]
    fn a_non_utf8_environment_value_is_what_std_env_makes_of_it() {
        use std::os::unix::ffi::OsStringExt;
        let n = "FASTVIDEO_H3_QUANT";
        let bad = OsString::from_vec(vec![0x66, 0xff]);
        let mut t = Settings::default();
        t.set(n, "mxfp8", "test").unwrap_or_else(|e| panic!("{e}"));
        let snap = Snapshot::from_parts(&|k| (k == n).then(|| bad.clone()), &t);
        // `std::env::var` fails with NotUnicode, so `var` falls to the table.
        assert_eq!(snap.var(n).as_deref(), Some("mxfp8"));
        assert!(snap.is_set(n));
        assert!(matches!(snap.env_var(n), Err(VarError::NotUnicode(_))));
        assert_eq!(snap.env_var_os(n), Some(bad));
    }

    /// A snapshot of the live process equals the live readers row for row
    /// (the model scope standing in for a profile table).
    #[test]
    fn a_captured_snapshot_equals_the_live_readers() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var(FREE_A, "env-a");
        let mut t = Settings::default();
        t.set(FREE_A, "table-a", "test")
            .unwrap_or_else(|e| panic!("{e}"));
        t.set(FREE_B, "table-b", "test")
            .unwrap_or_else(|e| panic!("{e}"));
        let _model = super::super::scope_model(t);
        let snap = Snapshot::capture();
        for (row, _) in snap.rows() {
            // `nvfp4::with_env` sets FASTVIDEO_NVFP4 in this test binary.
            if row.name == "FASTVIDEO_NVFP4" {
                continue;
            }
            assert_eq!(snap.var(row.name), Live.var(row.name), "{}", row.name);
            assert_eq!(snap.is_set(row.name), Live.is_set(row.name), "{}", row.name);
            assert_eq!(
                snap.env_var(row.name).ok(),
                Live.env_var(row.name).ok(),
                "{}",
                row.name
            );
            assert_eq!(
                snap.env_var_os(row.name),
                Live.env_var_os(row.name),
                "{}",
                row.name
            );
        }
        assert_eq!(snap.var(FREE_A).as_deref(), Some("env-a"));
        assert_eq!(snap.var(FREE_B).as_deref(), Some("table-b"));
        assert!(snap.env_var(FREE_B).is_err());
        std::env::remove_var(FREE_A);
    }

    /// Inside a scope the readers return the capture-time value: a change
    /// made after the load started is not seen until the scope ends (the
    /// order dependence inside a load is gone; a process that sets its
    /// environment before start sees no difference).
    #[test]
    fn a_scope_serves_the_capture_not_a_later_change() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var(FREE_A, "before");
        let snap = Snapshot::capture();
        {
            let _scope = snap.scope();
            std::env::set_var(FREE_A, "after");
            assert_eq!(super::super::var(FREE_A).as_deref(), Some("before"));
            assert_eq!(
                super::super::env_var(FREE_A).ok().as_deref(),
                Some("before")
            );
            assert!(super::super::is_set(FREE_A));
            // A load nested in this one captures the same values.
            assert_eq!(Snapshot::capture(), snap);
            // Another thread reads live, as before the snapshot existed.
            let other = std::thread::spawn(|| super::super::var(FREE_A))
                .join()
                .unwrap_or_else(|_| panic!("thread"));
            assert_eq!(other.as_deref(), Some("after"));
        }
        assert_eq!(super::super::var(FREE_A).as_deref(), Some("after"));
        std::env::remove_var(FREE_A);
        assert_eq!(super::super::var(FREE_A), None);
    }

    /// The load-time helpers of this crate that take the snapshot resolve
    /// what their pre-snapshot code resolved from the same sources (the old
    /// bodies written out: `settings::var` / `env_var` of the name, then the
    /// same parse).
    #[test]
    fn model_load_helpers_resolve_the_same_from_a_snapshot() {
        use crate::ltx2::memory::LtxOffload;
        use crate::wan::moe::{self, MoeResidency};
        let ltx = LtxOffload::ENV;
        let moe_env = "FASTVIDEO_WAN_MOE";
        let headroom = "FASTVIDEO_WAN_MOE_HEADROOM_GIB";
        let nv = "FASTVIDEO_NVFP4";
        let values = [
            "", "none", "cpu", "disk", "auto", "both", "swap", "1", "2.5", "junk",
        ];
        for v in values.iter().map(Some).chain([None]) {
            for (in_env, in_table) in [(true, false), (false, true), (true, true)] {
                let pairs: Vec<(&str, &str)> = match v {
                    Some(v) => vec![(ltx, *v), (moe_env, *v), (headroom, *v), (nv, *v)],
                    None => vec![],
                };
                let env = if in_env { pairs.clone() } else { vec![] };
                let table: Vec<(&str, &str)> = if in_table {
                    pairs.iter().map(|(k, _)| (*k, "cpu")).collect()
                } else {
                    vec![]
                };
                let snap = of(&env, &table);
                let env_of = |k: &str| {
                    env.iter()
                        .find(|(n, _)| *n == k)
                        .map(|(_, v)| v.to_string())
                };
                let mut t = Settings::default();
                for (k, v) in &table {
                    t.set(k, v, "test").unwrap_or_else(|e| panic!("{e}"));
                }
                let layered = |k: &str| super::super::var_in(&t, &env_of, k);
                // LtxOffload::from_env_or before B3b.
                let legacy = |explicit: Option<LtxOffload>| -> Result<LtxOffload, String> {
                    if let Some(m) = explicit {
                        return Ok(m);
                    }
                    match layered(ltx) {
                        Some(v) => LtxOffload::parse(&v).map_err(|e| format!("{ltx}: {e}")),
                        None => Ok(LtxOffload::None),
                    }
                };
                for explicit in [None, Some(LtxOffload::Cpu)] {
                    assert_eq!(
                        legacy(explicit),
                        LtxOffload::from_settings_or(&snap, explicit),
                        "{v:?} {in_env} {in_table}"
                    );
                }
                // MoeResidency::from_env / headroom_bytes: the environment only.
                let legacy_moe =
                    env_of(moe_env).map_or(Ok(MoeResidency::Auto), |v| MoeResidency::parse(&v));
                assert_eq!(legacy_moe, MoeResidency::from_settings(&snap), "{v:?}");
                let legacy_gib = env_of(headroom)
                    .and_then(|v| v.trim().parse::<f64>().ok())
                    .filter(|g| g.is_finite() && *g >= 0.0)
                    .unwrap_or(moe::DEFAULT_HEADROOM_GIB);
                assert_eq!(
                    (legacy_gib * (1u64 << 30) as f64) as u64,
                    moe::headroom_bytes_from(&snap),
                    "{v:?}"
                );
                // nvfp4::from_env: layered.
                assert_eq!(
                    crate::nvfp4::requested(layered(nv).as_deref()),
                    crate::nvfp4::from_settings(&snap),
                    "{v:?}"
                );
            }
        }
    }

    /// A `FASTVIDEO_*` variable a load sets while it runs (the MLX H3 tree
    /// sets `FASTVIDEO_H3_AFFINE` for the linear loads that follow) is seen by
    /// the reads after it, as it was before the snapshot existed.
    #[test]
    fn set_env_inside_a_load_is_seen_by_the_later_reads() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var(FREE_B);
        let snap = Snapshot::capture();
        {
            let _scope = snap.scope();
            assert_eq!(super::super::var(FREE_B), None);
            super::super::set_env(FREE_B, "8");
            assert_eq!(super::super::var(FREE_B).as_deref(), Some("8"));
            assert_eq!(super::super::env_var(FREE_B).ok().as_deref(), Some("8"));
        }
        // The process environment has it too, and the snapshot passed to the
        // load is unchanged.
        assert_eq!(super::super::var(FREE_B).as_deref(), Some("8"));
        assert_eq!(snap.var(FREE_B), None);
        std::env::remove_var(FREE_B);
    }

    #[test]
    fn nested_scopes_restore_the_outer_one() {
        let outer = of(&[(FREE_B, "outer")], &[]);
        let inner = of(&[(FREE_B, "inner")], &[]);
        let _a = outer.scope();
        {
            let _b = inner.scope();
            assert_eq!(super::super::var(FREE_B).as_deref(), Some("inner"));
        }
        assert_eq!(super::super::var(FREE_B).as_deref(), Some("outer"));
        assert_eq!(Live.var(FREE_B).as_deref(), Some("outer"));
    }
}
