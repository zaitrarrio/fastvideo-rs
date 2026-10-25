//! Config materialization: which TOML fv-serve runs, and the arguments it
//! gets. Since the one-image decision (ADR-0011, owner decision 2026-10-25)
//! the serve images bake no config: every launch sends one, in precedence
//! order (ADR-0014 §3.2):
//!
//! 1. **inline**: `FV_WORKER_TOML_B64` (base64 TOML from fv-control,
//!    fv-deploy, the scripts) is decoded into `/fv-worker.toml`;
//! 2. **named**: an explicit `--config <path>` argument (the image
//!    entrypoint's arguments: GCP, CloudRift, Vast on the debug image), or
//!    `FV_WORKER_CONFIG`, a file in the image (CI's no-GPU smoke check names
//!    `/etc/fv/runpod-fake.toml`, the one config the serve images carry). A
//!    controller worker (cluster / GMI / Brev) gets a copy in
//!    `/fv-worker.toml`; any other kind runs the file as it is.
//!
//! Anything else **refuses to boot** ([`missing`]): no baked `FV_CONFIG`
//! (no longer read), no fv-serve default. A controller worker's config gets
//! `register = false` in its `[cluster]` table, or in `[gateway]`, the name
//! that table had before plan F3b and the one the shipped configs still use;
//! a config with neither gets `[gateway]` (the shell boots do the same edit;
//! [`register_off`]).

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
    /// `FV_WORKER_CONFIG` (a file in the image).
    WorkerConfig(String),
}

impl Source {
    pub fn describe(&self) -> String {
        match self {
            Self::Inline => "inline FV_WORKER_TOML_B64".into(),
            Self::Args(p) => format!("--config {p}"),
            Self::WorkerConfig(p) => format!("FV_WORKER_CONFIG {p}"),
        }
    }
}

/// The config fv-serve will run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Materialized {
    pub source: Source,
    /// The TOML file.
    pub path: PathBuf,
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

/// `register = false` under the config's `[cluster]` table, or under
/// `[gateway]`, its name before plan F3b (whichever the config has: adding
/// the other name would make fv-serve refuse a config with both). A config
/// with neither gets a new `[gateway]` table: every fv-serve reads that
/// name, and the fv-serve fv-boot runs need not be its own image's
/// (`FV_SERVE_BIN`), so nothing writes `[cluster]` until stage 2 of the
/// rename (ADR-0011, "Renaming a worker-config section"). Idempotent: a
/// config that already starts its table with that line (a restarted
/// container re-reading its own copy) is kept.
pub fn register_off(toml: &str) -> String {
    let mut out = String::with_capacity(toml.len() + 32);
    let mut done = false;
    let lines: Vec<&str> = toml.split_inclusive('\n').collect();
    for (i, line) in lines.iter().enumerate() {
        out.push_str(line);
        let head = line.trim_end_matches(['\n', '\r']);
        if !done && (head.starts_with("[cluster]") || head.starts_with("[gateway]")) {
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

/// The refusal of a boot without a config (`None` when one is given). The
/// boot checks it first, before the driver check, so an image started
/// without a config says so at once and exits non-zero.
pub fn missing(env: &BootEnv, args: &[String]) -> Option<BootError> {
    if env.get("FV_WORKER_TOML_B64").is_some()
        || config_arg(args).is_some()
        || env.get("FV_WORKER_CONFIG").is_some()
    {
        return None;
    }
    let baked = if env.get("FV_CONFIG").is_some() {
        " (FV_CONFIG, the old images' baked config, is no longer read)"
    } else {
        ""
    };
    Some(BootError::Config(format!(
        "no worker config: the image serves no default; send FV_WORKER_TOML_B64 (inline) or FV_WORKER_CONFIG (a file in the image), or pass --config <path>{baked}"
    )))
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
            register_off(&text)
        } else {
            text
        };
        write(worker_toml, &text)?;
        return Ok(Materialized {
            source: Source::Inline,
            path: worker_toml.to_path_buf(),
            serve_args: with(worker_toml),
        });
    }
    if let Some(p) = config_arg(args) {
        return Ok(Materialized {
            source: Source::Args(p.clone()),
            path: PathBuf::from(&p),
            serve_args: args.to_vec(),
        });
    }
    if let Some(p) = env.get("FV_WORKER_CONFIG") {
        let src = Path::new(p);
        if !kind.controller_worker() {
            // As a `--config` argument: fv-serve --print-config reads it (and
            // names a missing file) before anything else runs.
            return Ok(Materialized {
                source: Source::WorkerConfig(p.to_owned()),
                path: src.to_path_buf(),
                serve_args: with(src),
            });
        }
        let text = std::fs::read_to_string(src)
            .map_err(|_| BootError::Config(format!("no worker config {p} in the image")))?;
        write(worker_toml, &register_off(&text))?;
        return Ok(Materialized {
            source: Source::WorkerConfig(p.to_owned()),
            path: worker_toml.to_path_buf(),
            serve_args: with(worker_toml),
        });
    }
    Err(missing(env, args).unwrap_or_else(|| BootError::Config("no worker config".into())))
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
    fn register_edit() {
        assert_eq!(
            register_off("[server]\nbind = \"x\"\n"),
            "[server]\nbind = \"x\"\n\n[gateway]\nregister = false\n"
        );
        // Under the table the config has, by either name (never both).
        assert_eq!(
            register_off("[cluster]\npool = \"h3\"\n"),
            "[cluster]\nregister = false\npool = \"h3\"\n"
        );
        assert_eq!(
            register_off("[gateway.x]\n[cluster]\n"),
            "[gateway.x]\n[cluster]\nregister = false\n"
        );
        let once = register_off("[cluster]\npool = \"h3\"\n");
        assert_eq!(register_off(&once), once);
        assert_eq!(
            register_off("[gateway]\npool = \"h3\"\n"),
            "[gateway]\nregister = false\npool = \"h3\"\n"
        );
        assert_eq!(
            register_off("a = 1\n[gateway]"),
            "a = 1\n[gateway]\nregister = false\n"
        );
        let once = register_off("[gateway]\npool = \"h3\"\n");
        assert_eq!(register_off(&once), once);
        let appended = register_off("a = 1\n");
        assert_eq!(register_off(&appended), appended);
        // Stage 1 of the rename (ADR-0011): fv-boot never introduces
        // `[cluster]`; the fv-serve it runs may predate it (`FV_SERVE_BIN`).
        for toml in [
            "",
            "[server]\n",
            "a = 1\n[gateway]",
            "[gateway]\npool = \"h3\"\n",
        ] {
            assert!(!register_off(toml).contains("[cluster]"), "{toml:?}");
        }
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
    fn no_config_refuses_to_boot() {
        let env = |p: &[(&str, &str)]| BootEnv::from_pairs(p.iter().copied());
        let why = |e: &BootEnv, a: &[&str]| missing(e, &s(a)).map(|e| e.to_string());
        // Nothing, or only the old images' baked FV_CONFIG: refused, with the remedy.
        let none = why(&env(&[]), &[]).unwrap_or_default();
        assert!(
            none.starts_with("no worker config: the image serves no default"),
            "{none}"
        );
        assert!(
            none.contains("FV_WORKER_TOML_B64") && none.contains("FV_WORKER_CONFIG"),
            "{none}"
        );
        let baked = why(&env(&[("FV_CONFIG", "/etc/fv/runpod.toml")]), &[]).unwrap_or_default();
        assert!(
            baked.contains("FV_CONFIG, the old images' baked config, is no longer read"),
            "{baked}"
        );
        // An empty variable is no config either.
        assert!(why(&env(&[("FV_WORKER_TOML_B64", "")]), &[]).is_some());
        // Any one of the three forms is a config.
        assert_eq!(
            why(&env(&[("FV_WORKER_TOML_B64", "W3NlcnZlcl0K")]), &[]),
            None
        );
        assert_eq!(
            why(
                &env(&[("FV_WORKER_CONFIG", "/etc/fv/runpod-fake.toml")]),
                &[]
            ),
            None
        );
        assert_eq!(
            why(&env(&[]), &["--config", "/etc/fv-gcp/serve.toml"]),
            None
        );
        assert_eq!(why(&env(&[]), &["--config=/a.toml"]), None);
        // materialize refuses the same way.
        let dir = std::env::temp_dir().join(format!("fv-boot-cfg-{}", std::process::id()));
        let r = materialize(&env(&[]), &Kind::Image, &[], &dir.join("w.toml"));
        assert!(matches!(r, Err(BootError::Config(m)) if m.starts_with("no worker config")));
    }

    #[test]
    fn worker_config_outside_a_controller_runs_the_file_as_is() {
        let dir = std::env::temp_dir().join(format!("fv-boot-wc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("runpod-fake.toml");
        std::fs::write(&fake, "[server]\n").unwrap();
        let wt = dir.join("w.toml");
        let e = BootEnv::from_pairs([("FV_WORKER_CONFIG", fake.display().to_string())]);
        let m = materialize(&e, &Kind::Image, &s(&["--x"]), &wt).unwrap();
        assert_eq!(m.source, Source::WorkerConfig(fake.display().to_string()));
        assert_eq!(m.path, fake);
        assert_eq!(
            m.serve_args,
            s(&["--config", &fake.display().to_string(), "--x"])
        );
        assert!(!wt.exists());
        // A controller worker copies it (registration off); a file the image does not carry is refused.
        let m = materialize(&e, &Kind::ClusterWorker, &[], &wt).unwrap();
        assert_eq!(m.path, wt);
        assert_eq!(
            std::fs::read_to_string(&wt).unwrap(),
            "[server]\n\n[gateway]\nregister = false\n"
        );
        let e = BootEnv::from_pairs([("FV_WORKER_CONFIG", "/etc/fv/runpod-ltx.toml")]);
        let r = materialize(&e, &Kind::ClusterWorker, &[], &wt);
        assert!(
            matches!(r, Err(BootError::Config(m)) if m == "no worker config /etc/fv/runpod-ltx.toml in the image")
        );
        std::fs::remove_dir_all(&dir).unwrap();
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
