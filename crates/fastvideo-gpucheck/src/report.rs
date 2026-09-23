//! Stage reports: every check is recorded with its measured values and limits,
//! and the first failing check aborts the stage (fail-fast). The JSON report is
//! written on success *and* on failure so a run always leaves evidence behind.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use serde_json::{json, Map, Value};

/// Exit codes shared with `scripts/gpu/*.sh`.
pub const EXIT_OK: i32 = 0;
pub const EXIT_CHECK_FAILED: i32 = 1;
pub const EXIT_ERROR: i32 = 2;
pub const EXIT_BUDGET: i32 = 3;

#[derive(Debug)]
pub enum StageError {
    /// A numerical / quality check failed.
    Check(String),
    /// A projected or observed time/memory budget was exceeded.
    Budget(String),
    /// Anything else (I/O, load failure, device init, ...).
    Error(anyhow::Error),
}

impl std::fmt::Display for StageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageError::Check(m) => write!(f, "check failed: {m}"),
            StageError::Budget(m) => write!(f, "budget exceeded: {m}"),
            StageError::Error(e) => write!(f, "error: {e:#}"),
        }
    }
}

impl<E: Into<anyhow::Error>> From<E> for StageError {
    fn from(e: E) -> Self {
        StageError::Error(e.into())
    }
}

pub type StageResult<T> = std::result::Result<T, StageError>;

#[derive(Debug, Serialize)]
pub struct CheckRecord {
    pub name: String,
    pub ok: bool,
    pub values: Map<String, Value>,
    pub limits: Map<String, Value>,
}

pub struct Report {
    stage: String,
    path: PathBuf,
    started: Instant,
    checks: Vec<CheckRecord>,
    extra: Map<String, Value>,
    /// Record failed checks and continue; the stage still fails at the end.
    /// For cheap diagnostic stages where one run should surface every bug.
    keep_going: bool,
    failed: Vec<String>,
}

impl Report {
    pub fn new(out_dir: &Path, stage: &str) -> Self {
        Self {
            stage: stage.to_string(),
            path: out_dir.join(format!("{stage}.json")),
            started: Instant::now(),
            checks: Vec::new(),
            extra: Map::new(),
            keep_going: false,
            failed: Vec::new(),
        }
    }

    /// Report name (stage + tag), e.g. `model-model-exact`.
    pub fn stage(&self) -> &str {
        &self.stage
    }

    pub fn set_keep_going(&mut self, keep_going: bool) {
        self.keep_going = keep_going;
    }

    /// Fold failures deferred by `keep_going` into the stage result.
    pub fn deferred(&self, result: StageResult<()>) -> StageResult<()> {
        match result {
            Ok(()) if !self.failed.is_empty() => Err(StageError::Check(format!(
                "{} check(s) failed: {}",
                self.failed.len(),
                self.failed.join("; ")
            ))),
            other => other,
        }
    }

    /// Attach arbitrary context (config, timings, device info) to the report.
    pub fn set(&mut self, key: &str, value: impl Serialize) {
        self.extra.insert(
            key.to_string(),
            serde_json::to_value(value).unwrap_or(Value::Null),
        );
    }

    /// Record a check; return `Err` (fail-fast) when it did not pass.
    pub fn check(
        &mut self,
        name: impl Into<String>,
        ok: bool,
        values: Value,
        limits: Value,
    ) -> StageResult<()> {
        let name = name.into();
        let values = as_map(values);
        let limits = as_map(limits);
        let status = if ok { "PASS" } else { "FAIL" };
        eprintln!(
            "[{status}] {}/{name} {} (limits {})",
            self.stage,
            Value::Object(values.clone()),
            Value::Object(limits.clone())
        );
        let detail = format!(
            "{}/{name}: {} vs limits {}",
            self.stage,
            Value::Object(values.clone()),
            Value::Object(limits.clone())
        );
        self.checks.push(CheckRecord {
            name,
            ok,
            values,
            limits,
        });
        if ok {
            Ok(())
        } else if self.keep_going {
            self.failed.push(detail);
            Ok(())
        } else {
            Err(StageError::Check(detail))
        }
    }

    /// Informational measurement that never fails the stage.
    pub fn note(&mut self, name: impl Into<String>, values: Value) {
        let name = name.into();
        eprintln!("[INFO] {}/{name} {values}", self.stage);
        self.checks.push(CheckRecord {
            name,
            ok: true,
            values: as_map(values),
            limits: Map::new(),
        });
    }

    pub fn finish(self, result: &StageResult<()>) -> i32 {
        let (status, code, error) = match result {
            Ok(()) => ("pass", EXIT_OK, Value::Null),
            Err(e @ StageError::Check(_)) => ("fail", EXIT_CHECK_FAILED, json!(e.to_string())),
            Err(e @ StageError::Budget(_)) => ("budget", EXIT_BUDGET, json!(e.to_string())),
            Err(e @ StageError::Error(_)) => ("error", EXIT_ERROR, json!(e.to_string())),
        };
        let doc = json!({
            "stage": self.stage,
            "status": status,
            "error": error,
            "elapsed_s": self.started.elapsed().as_secs_f64(),
            "git_sha": std::env::var("FV_GIT_SHA").ok(),
            "env": fastvideo_env(),
            "checks": self.checks,
            "context": Value::Object(self.extra),
        });
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string_pretty(&doc) {
            Ok(text) => {
                if let Err(e) = std::fs::write(&self.path, text) {
                    eprintln!("warn: could not write {}: {e}", self.path.display());
                }
            }
            Err(e) => eprintln!("warn: could not serialize report: {e}"),
        }
        eprintln!(
            "[{}] stage {} → {} ({:.1}s)",
            status.to_uppercase(),
            self.stage,
            self.path.display(),
            self.started.elapsed().as_secs_f64()
        );
        if let Err(e) = result {
            eprintln!("{e}");
        }
        code
    }
}

fn as_map(v: Value) -> Map<String, Value> {
    match v {
        Value::Object(m) => m,
        Value::Null => Map::new(),
        other => {
            let mut m = Map::new();
            m.insert("value".into(), other);
            m
        }
    }
}

/// Every `FASTVIDEO_*` variable in effect, so a report is self-describing.
pub fn fastvideo_env() -> Map<String, Value> {
    let mut vars: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| k.starts_with("FASTVIDEO_"))
        .collect();
    vars.sort();
    vars.into_iter()
        .map(|(k, v)| (k, Value::String(v)))
        .collect()
}
