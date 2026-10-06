//! Build identity for `fv-serve --version` and `/health` (docs/serve/releases.md).
//!
//! Compile-time values, as `env!` variables of the crate:
//!
//! | variable | source (first that is set) |
//! |---|---|
//! | `FV_BUILD_GIT_SHA` | `FV_GIT_SHA`, `GITHUB_SHA`, `git rev-parse HEAD`, `unknown` |
//! | `FV_BUILD_TIME` | `FV_BUILD_TIME`, the commit time (`git log -1`, UTC), empty |
//! | `FV_BUILD_ID` | `FV_BUILD_ID`, `BUILD_ID`, empty |
//! | `FV_BUILD_VARIANT` | `FV_BUILD_VARIANT` (images set `FV_VARIANT` at run time instead: the CUDA variants share one binary) |
//! | `FV_BUILD_FEATURES` | the crate features this build enables |
//! | `FV_BUILD_PROFILE` | cargo's `PROFILE` |
//! | `FV_BUILD_VERSION` | `FV_RELEASE_VERSION` (the tools release being built, docs/dev/tools-releases.md), else the package version |
//!
//! The image build passes `FV_GIT_SHA` / `FV_BUILD_TIME` as build args
//! (`.git` is not in the Docker context). The commit time rather than the
//! wall clock keeps rebuilds of one commit identical and cacheable.

use std::process::Command;

fn env(name: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={name}");
    std::env::var(name).ok().map(|v| v.trim().to_owned()).filter(|v| !v.is_empty())
}

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).env("TZ", "UTC").output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_owned()).filter(|s| !s.is_empty())
}

/// Re-run when HEAD moves (a commit, a checkout), when git is there at all.
fn watch_git_head() {
    for p in ["HEAD", "packed-refs"] {
        if let Some(path) = git(&["rev-parse", "--git-path", p]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(r) = git(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git(&["rev-parse", "--git-path", &r]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let from_env = env("FV_GIT_SHA").or_else(|| env("GITHUB_SHA"));
    if from_env.is_none() {
        watch_git_head();
    }
    let sha = from_env
        .or_else(|| git(&["rev-parse", "HEAD"]))
        .filter(|s| s.chars().all(|c| c.is_ascii_hexdigit()))
        .unwrap_or_else(|| "unknown".into());
    let time = env("FV_BUILD_TIME")
        .or_else(|| git(&["log", "-1", "--format=%cd", "--date=format-local:%Y-%m-%dT%H:%M:%SZ"]))
        .unwrap_or_default();
    let build_id = env("FV_BUILD_ID").or_else(|| env("BUILD_ID")).unwrap_or_default();
    let variant = env("FV_BUILD_VARIANT").unwrap_or_default();
    let mut features: Vec<String> = std::env::vars()
        .filter_map(|(k, _)| k.strip_prefix("CARGO_FEATURE_").map(|f| f.to_ascii_lowercase().replace('_', "-")))
        .filter(|f| f != "default")
        .collect();
    features.sort();
    let profile = std::env::var("PROFILE").unwrap_or_default();
    let version = env("FV_RELEASE_VERSION").unwrap_or_else(|| std::env::var("CARGO_PKG_VERSION").unwrap_or_default());
    println!("cargo:rustc-env=FV_BUILD_GIT_SHA={sha}");
    println!("cargo:rustc-env=FV_BUILD_TIME={time}");
    println!("cargo:rustc-env=FV_BUILD_ID={build_id}");
    println!("cargo:rustc-env=FV_BUILD_VARIANT={variant}");
    println!("cargo:rustc-env=FV_BUILD_FEATURES={}", features.join(","));
    println!("cargo:rustc-env=FV_BUILD_PROFILE={profile}");
    println!("cargo:rustc-env=FV_BUILD_VERSION={version}");
}
