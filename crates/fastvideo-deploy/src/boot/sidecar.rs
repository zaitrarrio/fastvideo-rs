//! The prompt upsampler's local sidecar (`FV_UPSAMPLER_LOCAL=1`,
//! docs/serve/upsampler-local.md). The image's launcher
//! (`/opt/fastvideo-rs/bin/fv-upsampler`, deploy/runpod/fv-upsampler.sh) stays
//! the sidecar's supervisor; fv-boot drives it as the shell boots did
//! (control upsampler.ts `UPSAMPLER_START_SH` / `UPSAMPLER_WAIT_SH`):
//!
//! - [`start`] early, so its fetch and load overlap the weights;
//! - [`wait`] just before fv-serve: `fv-upsampler wait` prints `export K='V'`
//!   lines (FV_UPSAMPLER_BASE_URL, FV_UPSAMPLER_MODEL, FV_UPSAMPLER_VISION,
//!   FV_UPSAMPLER_LOCAL) that fv-serve gets; its outcome is reported (a
//!   serverless `step upsampler …`, a pod's `upsampler` phase). A failed
//!   sidecar never fails the boot: fv-serve starts and upsampling falls back.

use super::sys::is_executable;
use super::Ctx;

/// `export K='V'` lines -> pairs (`'\''` unquoted).
pub fn parse_exports(stdout: &str) -> Vec<(String, String)> {
    stdout
        .lines()
        .filter_map(|l| {
            let rest = l.trim().strip_prefix("export ")?;
            let (k, v) = rest.split_once('=')?;
            let v = v
                .strip_prefix('\'')
                .and_then(|v| v.strip_suffix('\''))
                .map(|v| v.replace("'\\''", "'"))
                .unwrap_or_else(|| v.to_owned());
            (!k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
                .then(|| (k.to_owned(), v))
        })
        .collect()
}

/// The wait's last line without its `[fv-boot] HH:MM:SS ` prefix, cleaned
/// (`tr -d '"\\' | cut -c1-280`).
pub fn outcome_line(log: &str) -> String {
    let last = log
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or_default();
    let rest = last
        .strip_prefix("[fv-boot] ")
        .map(|r| {
            r.trim_start_matches(|c: char| c.is_ascii_digit() || c == ':')
                .trim_start()
        })
        .unwrap_or(last);
    rest.chars()
        .filter(|c| *c != '"' && *c != '\\')
        .take(280)
        .collect()
}

/// The seconds of a `step upsampler: <s>s (…)` line.
pub fn outcome_secs(line: &str) -> Option<String> {
    let rest = line.strip_prefix("step upsampler: ")?;
    let n: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    (!n.is_empty() && rest[n.len()..].starts_with('s')).then_some(n)
}

/// Starts the sidecar (returns at once; a no-op unless `FV_UPSAMPLER_LOCAL=1`).
pub fn start(c: &Ctx<'_>) {
    if !c.env.flag("FV_UPSAMPLER_LOCAL") {
        return;
    }
    let ups = &c.paths.upsampler;
    if !is_executable(ups) {
        c.say(&format!(
            "upsampler: FAILED: this image has no llama-server sidecar ({}): fv-serve starts without it; upsampling falls back",
            ups.display()
        ));
        return;
    }
    if let Err(e) = c
        .deps
        .spawner
        .status(&c.cmd(ups.display().to_string()).arg("start"))
    {
        c.say(&format!(
            "upsampler: FAILED: {e}: fv-serve starts without it; upsampling falls back"
        ));
    }
}

/// Waits for the sidecar and exports what fv-serve needs.
pub fn wait(c: &mut Ctx<'_>) {
    if !c.env.flag("FV_UPSAMPLER_LOCAL") || !is_executable(&c.paths.upsampler) {
        return;
    }
    let log = c.state_dir().join("upsampler-wait.log");
    let cmd = c.cmd(c.paths.upsampler.display().to_string()).arg("wait");
    // Both streams captured: stdout holds the exports, stderr the outcome
    // (kept in upsampler-wait.log and copied to the container log).
    let out = match c.deps.spawner.output(&cmd) {
        Ok(o) => o,
        Err(e) => {
            c.say(&format!(
                "upsampler: FAILED: {e}: fv-serve starts without it; upsampling falls back"
            ));
            return;
        }
    };
    let err = out.stderr_text();
    let _ = std::fs::write(&log, &err);
    for l in err.lines() {
        c.log.raw(l);
    }
    for (k, v) in parse_exports(&out.stdout_text()) {
        c.export(&k, v);
    }
    let line = outcome_line(&err);
    match outcome_secs(&line) {
        Some(s) => {
            let rest = line.split_once("s (").map(|(_, r)| r).unwrap_or("");
            c.rep.step(
                c.deps.http,
                c.deps.clock,
                "upsampler",
                s.parse().unwrap_or(0.0),
                rest.trim_end_matches(')'),
            );
        }
        None => c.rep.sls_phase(
            c.deps.http,
            c.deps.clock,
            "step",
            &format!("upsampler_failed 0s ({line})"),
        ),
    }
    c.pod_report("upsampler", "", &line);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exports() {
        let o = "export FV_UPSAMPLER_BASE_URL='http://127.0.0.1:18080/v1'\nexport FV_UPSAMPLER_MODEL='qwen3-vl-4b-instruct'\nnoise\nexport BAD-KEY='x'\nexport Q='it'\\''s'\n";
        assert_eq!(
            parse_exports(o),
            vec![
                (
                    "FV_UPSAMPLER_BASE_URL".into(),
                    "http://127.0.0.1:18080/v1".into()
                ),
                ("FV_UPSAMPLER_MODEL".into(), "qwen3-vl-4b-instruct".into()),
                ("Q".into(), "it's".into()),
            ]
        );
    }

    #[test]
    fn outcome() {
        let l = outcome_line("x\n[fv-boot] 10:00:01 step upsampler: 41.2s (llama-server 127.0.0.1:18080, qwen3-vl-4b-instruct; load 30.1)\n");
        assert_eq!(
            l,
            "step upsampler: 41.2s (llama-server 127.0.0.1:18080, qwen3-vl-4b-instruct; load 30.1)"
        );
        assert_eq!(outcome_secs(&l).as_deref(), Some("41.2"));
        let f = outcome_line("[fv-boot] 10:00:01 upsampler: FAILED: no \"gguf\"\n");
        assert_eq!(f, "upsampler: FAILED: no gguf");
        assert_eq!(outcome_secs(&f), None);
    }
}
