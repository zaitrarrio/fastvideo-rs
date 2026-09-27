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

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::profile::Profile;

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

/// A `FASTVIDEO_*` setting: the env var, else the active profile's value.
pub fn var(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(v) => Some(v),
        Err(_) => active().settings.get(name).map(str::to_owned),
    }
}

/// Whether the setting is present (env or profile), as `var_os().is_some()`.
pub fn is_set(name: &str) -> bool {
    std::env::var_os(name).is_some() || active().settings.get(name).is_some()
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
        s.set("FASTVIDEO_X", "profile", "t").unwrap();
        let none = |_: &str| None;
        let env = |_: &str| Some("env".to_string());
        assert_eq!(var_in(&s, &none, "FASTVIDEO_X").as_deref(), Some("profile"));
        assert_eq!(var_in(&s, &env, "FASTVIDEO_X").as_deref(), Some("env"));
        assert_eq!(var_in(&s, &none, "FASTVIDEO_Y"), None);
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
