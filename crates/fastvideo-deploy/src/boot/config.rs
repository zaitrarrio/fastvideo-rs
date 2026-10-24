//! Config materialization: which TOML fv-serve runs, and the arguments it
//! gets. The three forms (ADR-0014 §3.2), in precedence order:
//!
//! 1. **inline**: `FV_WORKER_TOML_B64` (base64 TOML from fv-control) is
//!    decoded into `/fv-worker.toml`;
//! 2. **named**: an explicit `--config <path>` argument (the image entrypoint's
//!    arguments: GCP, CloudRift, Vast, a v2 load balancer's `args`), or, for a
//!    controller worker (cluster / GMI / Brev), `FV_WORKER_CONFIG`: a file in
//!    the image, copied to `/fv-worker.toml` (a controller worker with neither
//!    an inline config nor this fails, as the shell boots did);
//! 3. **baked**: the image's `FV_CONFIG`.
//!
//! A controller worker's config gets `register = false` in its `[gateway]`
//! table (the gateway is retired; the shell boots did the same edit).

use std::path::{Path, PathBuf};

use base64::Engine as _;

use super::env::{BootEnv, Kind};
use super::error::BootError;

/// Where the config came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// `FV_WORKER_TOML_B64`.
    Inline,
    /// `--config <path>` among the arguments.
    Args(String),
    /// `FV_WORKER_CONFIG` (a controller worker).
    WorkerConfig(String),
    /// `FV_CONFIG` (baked into the image).
    Baked(String),
    /// No config at all: fv-serve's own default.
    Default,
}

impl Source {
    pub fn describe(&self) -> String {
        match self {
            Self::Inline => "inline FV_WORKER_TOML_B64".into(),
            Self::Args(p) => format!("--config {p}"),
            Self::WorkerConfig(p) => format!("FV_WORKER_CONFIG {p}"),
            Self::Baked(p) => format!("FV_CONFIG {p}"),
            Self::Default => "fv-serve's default".into(),
        }
    }
}

/// The config fv-serve will run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Materialized {
    pub source: Source,
    /// The TOML file (`None`: fv-serve's default).
    pub path: Option<PathBuf>,
    /// fv-serve's arguments.
    pub serve_args: Vec<String>,
}

/// The `--config` value among fv-serve arguments (`--config X` or `--config=X`).
pub fn config_arg(args: &[String]) -> Option<String> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--config" {
            return it.next().cloned();
        }
        if let Some(v) = a.strip_prefix("--config=") {
            return Some(v.to_owned());
        }
    }
    None
}

/// The arguments without any `--config`.
pub fn without_config(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--config" {
            it.next();
            continue;
        }
        if a.starts_with("--config=") {
            continue;
        }
        out.push(a.clone());
    }
    out
}

/// `register = false` under `[gateway]` (appended as a new table when there
/// is none). Idempotent: a config that already starts its `[gateway]` table
/// with that line (a restarted container re-reading its own copy) is kept.
pub fn gateway_off(toml: &str) -> String {
    let mut out = String::with_capacity(toml.len() + 32);
    let mut done = false;
    let lines: Vec<&str> = toml.split_inclusive('\n').collect();
    for (i, line) in lines.iter().enumerate() {
        out.push_str(line);
        if !done && line.trim_end_matches(['\n', '\r']).starts_with("[gateway]") {
            done = true;
            if lines.get(i + 1).map(|l| l.trim()) == Some("register = false") {
                continue;
            }
            if !line.ends_with('\n') {
                out.push('\n');
            }
            out.push_str("register = false\n");
        }
    }
    if !done {
        out.push_str("\n[gateway]\nregister = false\n");
    }
    out
}

fn write(path: &Path, text: &str) -> Result<(), BootError> {
    std::fs::write(path, text)
        .map_err(|e| BootError::Config(format!("cannot write {}: {e}", path.display())))
}

/// Decides the config and writes `/fv-worker.toml` when the form needs it.
pub fn materialize(
    env: &BootEnv,
    kind: &Kind,
    args: &[String],
    worker_toml: &Path,
) -> Result<Materialized, BootError> {
    let rest = without_config(args);
    let with = |p: &Path| {
        let mut a = vec!["--config".to_owned(), p.display().to_string()];
        a.extend(rest.iter().cloned());
        a
    };
    if let Some(b64) = env.get("FV_WORKER_TOML_B64") {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64.trim())
            .map_err(|_| BootError::Config("FV_WORKER_TOML_B64 does not decode".into()))?;
        let text = String::from_utf8(bytes)
            .map_err(|_| BootError::Config("FV_WORKER_TOML_B64 is not UTF-8".into()))?;
        let text = if kind.controller_worker() {
            gateway_off(&text)
        } else {
            text
        };
        write(worker_toml, &text)?;
        return Ok(Materialized {
            source: Source::Inline,
            path: Some(worker_toml.to_path_buf()),
            serve_args: with(worker_toml),
        });
    }
    if let Some(p) = config_arg(args) {
        return Ok(Materialized {
            source: Source::Args(p.clone()),
            path: Some(PathBuf::from(&p)),
            serve_args: args.to_vec(),
        });
    }
    if kind.controller_worker() {
        // fv-control always sends one of the two; the shell boots died on the
        // unset variable (`set -u`) rather than serve a config without the edit.
        let p = env.get("FV_WORKER_CONFIG").ok_or_else(|| {
            BootError::Config(
                "no worker config: neither FV_WORKER_TOML_B64 nor FV_WORKER_CONFIG is set".into(),
            )
        })?;
        let src = Path::new(p);
        let text = std::fs::read_to_string(src)
            .map_err(|_| BootError::Config(format!("no worker config {p} in the image")))?;
        write(worker_toml, &gateway_off(&text))?;
        return Ok(Materialized {
            source: Source::WorkerConfig(p.to_owned()),
            path: Some(worker_toml.to_path_buf()),
            serve_args: with(worker_toml),
        });
    }
    if let Some(p) = env.get("FV_CONFIG") {
        return Ok(Materialized {
            source: Source::Baked(p.to_owned()),
            path: Some(PathBuf::from(p)),
            serve_args: with(Path::new(p)),
        });
    }
    Ok(Materialized {
        source: Source::Default,
        path: None,
        serve_args: rest,
    })
}

/// The weight trees a config names as `weights = "${FV_WEIGHTS}/<tree>…"`
/// (fv-entry.sh's preflight; the `[[models]] variant` form names none).
pub fn weight_trees(toml: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in toml.lines() {
        let Some(rest) = line.strip_prefix("weights") else {
            continue;
        };
        let Some(rest) = rest.trim_start_matches(' ').strip_prefix('=') else {
            continue;
        };
        let Some(rest) = rest
            .trim_start_matches(' ')
            .strip_prefix("\"${FV_WEIGHTS}/")
        else {
            continue;
        };
        let tree: String = rest
            .chars()
            .take_while(|c| *c != '"' && *c != '/')
            .collect();
        if !tree.is_empty() && !out.contains(&tree) {
            out.push(tree);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn gateway_edit() {
        assert_eq!(
            gateway_off("[server]\nbind = \"x\"\n"),
            "[server]\nbind = \"x\"\n\n[gateway]\nregister = false\n"
        );
        assert_eq!(
            gateway_off("[gateway]\npool = \"h3\"\n"),
            "[gateway]\nregister = false\npool = \"h3\"\n"
        );
        assert_eq!(
            gateway_off("a = 1\n[gateway]"),
            "a = 1\n[gateway]\nregister = false\n"
        );
        let once = gateway_off("[gateway]\npool = \"h3\"\n");
        assert_eq!(gateway_off(&once), once);
        let appended = gateway_off("a = 1\n");
        assert_eq!(gateway_off(&appended), appended);
    }

    #[test]
    fn config_args() {
        assert_eq!(
            config_arg(&s(&["--config", "/a.toml", "--x"])),
            Some("/a.toml".into())
        );
        assert_eq!(
            config_arg(&s(&["--config=/b.toml"])),
            Some("/b.toml".into())
        );
        assert_eq!(config_arg(&s(&["--x"])), None);
        assert_eq!(
            without_config(&s(&["--config", "/a", "--x", "--config=/b", "y"])),
            s(&["--x", "y"])
        );
    }

    #[test]
    fn trees_from_the_old_form() {
        let t = "[[models]]\nweights = \"${FV_WEIGHTS}/h3-base\"\n# weights = \"${FV_WEIGHTS}/nope\"\nweights  =  \"${FV_WEIGHTS}/ltx25/x\"\nweights = \"/abs\"\n";
        assert_eq!(
            weight_trees(t),
            vec!["h3-base".to_owned(), "ltx25".to_owned()]
        );
    }
}
