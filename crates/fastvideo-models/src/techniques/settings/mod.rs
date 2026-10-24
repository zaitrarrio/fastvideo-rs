//! Process-wide settings: the one place a `FASTVIDEO_*` knob is read.
//!
//! Precedence, highest first:
//!
//! 1. the environment variable (every existing flag keeps working, unchanged,
//!    and overrides the profile);
//! 2. the value the active technique profile installs for that name
//!    ([`super::Technique::settings`], sol-engine's `set_env`,
//!    `transform.py:73-74`, but written to this table instead of the
//!    process environment);
//! 3. the caller's built-in default.
//!
//! With no profile the table is empty and every read is exactly the old
//! `std::env::var` read, which is the off-identity of the whole layer.
//!
//! The profile is installed once, before the first read: by the binaries'
//! `--techniques <file>` flag ([`install_file`]), or lazily from
//! `FASTVIDEO_TECHNIQUES=<file>` on the first [`var`]. Getters cache their
//! parsed flags on first use, so a later install would be ignored by some
//! and seen by others; [`install`] refuses it instead.

//!
//! # The declarative list
//!
//! Every `FASTVIDEO_*` name the workspace reads is one row of [`SETTINGS`]
//! ([`table`]): its reader, type and parse rule, default, family, seam, read
//! timing and doc string. `docs/scope.md`'s settings table is rendered from
//! it ([`render_markdown`]) and a test fails when the committed doc drifts. A
//! second test scans the workspace sources and fails on any read of a
//! `FASTVIDEO_*` name outside this module (`std::env::var("FASTVIDEO_...")`,
//! a `const` name passed to `std::env::var`, a lookup closure) and on any
//! `"FASTVIDEO_..."` literal that is not a row (`docs/adr/0010-settings-layer.md`).
//!
//! Two readers, chosen per row ([`Reader`]):
//!
//! * [`var`] / [`is_set`] (`Reader::Layered`): the environment, then the
//!   technique profile's table, then the caller's default (above);
//! * [`env_var`] / [`env_var_os`] (`Reader::Env`): the process environment
//!   only, exactly `std::env::var` / `std::env::var_os`. These are the names
//!   that were read with `std::env::var` before the table existed; a profile's
//!   `[env]` entry for one of them has no effect, as before. Moving a name to
//!   the layered reader is a behaviour change (a profile could then set it)
//!   and gets its own byte-identity cell.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::profile::Profile;

mod doc;
pub mod parse;
pub mod table;
#[cfg(test)]
mod workspace_check;

pub use doc::{render_markdown, DOC_BEGIN, DOC_END};
pub use table::{lookup, DefaultValue, Family, Parse, Reader, Setting, Timing, SETTINGS};

/// Env var naming a technique profile to load when no binary installed one.
pub const PROFILE_ENV: &str = "FASTVIDEO_TECHNIQUES";

/// Name → value, with the technique that set each (for conflicts and logs).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Settings {
    values: BTreeMap<String, (String, String)>,
}

impl Settings {
    /// Set `key`; a different value from a different source is a conflict.
    pub fn set(&mut self, key: &str, value: &str, source: &str) -> Result<(), String> {
        match self.values.get(key) {
            Some((v, s)) if v != value && s != source => Err(format!(
                "setting {key}: '{s}' sets {v:?} and '{source}' sets {value:?}"
            )),
            _ => {
                self.values
                    .insert(key.to_string(), (value.to_string(), source.to_string()));
                Ok(())
            }
        }
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|(v, _)| v.as_str())
    }

    pub fn source(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|(_, s)| s.as_str())
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &str, &str)> {
        self.values
            .iter()
            .map(|(k, (v, s))| (k.as_str(), v.as_str(), s.as_str()))
    }
}

/// The installed profile and the settings it resolved to.
#[derive(Debug, Default)]
pub struct Active {
    pub profile: Option<Profile>,
    pub path: Option<PathBuf>,
    pub settings: Settings,
}

static ACTIVE: OnceLock<Active> = OnceLock::new();

fn load(path: &Path) -> Result<Active, String> {
    let profile = Profile::load(path)?;
    let settings = profile.settings()?;
    Ok(Active {
        profile: Some(profile),
        path: Some(path.to_path_buf()),
        settings,
    })
}

/// The active profile: installed, else `FASTVIDEO_TECHNIQUES`, else none.
///
/// # Panics
/// When `FASTVIDEO_TECHNIQUES` names a profile that does not load and no
/// binary installed one first (the binaries call [`install_from_env`] at
/// start-up and report the error there instead).
pub fn active() -> &'static Active {
    ACTIVE.get_or_init(|| match std::env::var_os(PROFILE_ENV) {
        Some(p) if !p.is_empty() => load(Path::new(&p))
            .unwrap_or_else(|e| panic!("{PROFILE_ENV}={}: {e}", Path::new(&p).display())),
        _ => Active::default(),
    })
}

/// Install `active`. Errors when a profile is already installed or a setting
/// was already read.
pub fn install(active: Active) -> Result<&'static Active, String> {
    let mut slot = Some(active);
    let got = ACTIVE.get_or_init(|| slot.take().expect("install"));
    if slot.is_some() {
        return Err(
            "technique profile: already initialised (install it before any FASTVIDEO_* setting is read)"
                .into(),
        );
    }
    Ok(got)
}

/// Load and install the profile at `path`.
pub fn install_file(path: &Path) -> Result<&'static Active, String> {
    install(load(path).map_err(|e| format!("{}: {e}", path.display()))?)
}

/// Binaries' start-up: `explicit` (a `--techniques` flag) wins over
/// `FASTVIDEO_TECHNIQUES`; errors are returned, not panicked.
pub fn install_from_env(explicit: Option<&Path>) -> Result<&'static Active, String> {
    let from_env = std::env::var_os(PROFILE_ENV).filter(|p| !p.is_empty());
    match explicit
        .map(Path::to_path_buf)
        .or(from_env.map(PathBuf::from))
    {
        Some(p) => install_file(&p),
        None => Ok(active()),
    }
}

thread_local! {
    /// One model's own settings while [`scope_model`]'s guard lives.
    static MODEL_SCOPE: std::cell::RefCell<Option<Settings>> = const { std::cell::RefCell::new(None) };
}

/// Restores the previous [`scope_model`] table when dropped.
#[must_use = "the scope ends when the guard drops"]
pub struct ModelScope(Option<Settings>);

impl Drop for ModelScope {
    fn drop(&mut self) {
        let prev = self.0.take();
        MODEL_SCOPE.with(|s| *s.borrow_mut() = prev);
    }
}

/// While the guard lives, this thread's reads ([`var`], [`is_set`]) see
/// `settings`, one model's own profile's (empty: a model without a profile),
/// in place of the process profile's; the environment still wins.
///
/// A process serving models of several technique profiles (fv-serve with
/// `[engine] swap = true`, one model resident at a time) loads and runs each
/// model under its own load-time settings, as it would alone in a process:
/// e.g. `FASTVIDEO_H3_QUANT` is `mxfp8` while an MXFP8 profile's model loads
/// and unset (the device's or the recipe's default) while an NVFP4 release
/// loads. Settings a getter caches on first read (`FASTVIDEO_BF16_ACT`) stay
/// process-wide: fastvideo-engine-service's `ProcessPlan` admits only sets
/// that agree on those.
pub fn scope_model(settings: Settings) -> ModelScope {
    ModelScope(MODEL_SCOPE.with(|s| s.borrow_mut().replace(settings)))
}

/// The table a read consults after the environment: the model scope's, else
/// the active profile's.
fn table_get(name: &str) -> Option<String> {
    match MODEL_SCOPE.with(|s| s.borrow().as_ref().map(|t| t.get(name).map(str::to_owned))) {
        Some(scoped) => scoped,
        None => active().settings.get(name).map(str::to_owned),
    }
}

/// A `FASTVIDEO_*` setting: the env var, else the active profile's value
/// (inside [`scope_model`]: the scoped model's).
pub fn var(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(v) => Some(v),
        Err(_) => table_get(name),
    }
}

/// Whether the setting is present (env or profile), as `var_os().is_some()`.
pub fn is_set(name: &str) -> bool {
    std::env::var_os(name).is_some() || table_get(name).is_some()
}

/// A `FASTVIDEO_*` setting read from the process environment only
/// (`Reader::Env` rows): exactly `std::env::var(name)`, read when called.
pub fn env_var(name: &str) -> Result<String, std::env::VarError> {
    std::env::var(name)
}

/// [`env_var`] as an `OsString`: exactly `std::env::var_os(name)`.
pub fn env_var_os(name: &str) -> Option<OsString> {
    std::env::var_os(name)
}

/// [`var`] against an explicit table instead of the global one (tests,
/// dry runs).
pub fn var_in(
    settings: &Settings,
    env: &dyn Fn(&str) -> Option<String>,
    name: &str,
) -> Option<String> {
    env(name).or_else(|| settings.get(name).map(str::to_owned))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_beats_profile_beats_default() {
        let mut s = Settings::default();
        s.set("FASTVIDEO_TEST_X", "profile", "t").unwrap();
        let none = |_: &str| None;
        let env = |_: &str| Some("env".to_string());
        assert_eq!(
            var_in(&s, &none, "FASTVIDEO_TEST_X").as_deref(),
            Some("profile")
        );
        assert_eq!(var_in(&s, &env, "FASTVIDEO_TEST_X").as_deref(), Some("env"));
        assert_eq!(var_in(&s, &none, "FASTVIDEO_TEST_Y"), None);
    }

    #[test]
    fn a_model_scope_replaces_the_profile_table_until_dropped() {
        // No profile is installed in this test binary: the table is empty.
        let name = "FASTVIDEO_TEST_MODEL_SCOPE_ONLY";
        assert_eq!(var(name), None);
        let mut own = Settings::default();
        own.set(name, "mxfp8", "mxfp8").unwrap();
        {
            let _outer = scope_model(own);
            assert_eq!(var(name).as_deref(), Some("mxfp8"));
            assert!(is_set(name));
            {
                // A model without a profile: nothing, whatever the outer scope had.
                let _inner = scope_model(Settings::default());
                assert_eq!(var(name), None);
                assert!(!is_set(name));
            }
            assert_eq!(var(name).as_deref(), Some("mxfp8"));
        }
        assert_eq!(var(name), None);
        // Another thread never sees this thread's scope.
        let mut s = Settings::default();
        s.set(name, "x", "t").unwrap();
        let _g = scope_model(s);
        assert_eq!(std::thread::spawn(move || var(name)).join().unwrap(), None);
    }

    #[test]
    fn two_techniques_cannot_set_one_name_differently() {
        let mut s = Settings::default();
        s.set("K", "a", "one").unwrap();
        s.set("K", "a", "two").unwrap();
        assert!(s.set("K", "b", "three").is_err());
        assert_eq!(s.source("K"), Some("two"));
    }
}
