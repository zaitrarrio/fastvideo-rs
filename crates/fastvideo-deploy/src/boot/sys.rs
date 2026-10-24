//! The seams between the boot sequence and the machine: processes
//! ([`Spawner`]), HTTP ([`Http`]), time ([`Clock`]) and the log
//! ([`LogSink`]). Production uses [`RealSpawner`], [`RealClock`],
//! [`StderrLog`] and (feature `boot`) `http::ReqwestHttp`; the tests use
//! fakes, so every step runs without a GPU, a network or a provider.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::fd::AsFd;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Where a child's stdout or stderr goes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Io {
    /// This process's own stream.
    #[default]
    Inherit,
    /// `/dev/null`.
    Null,
    /// A file, truncated first.
    Truncate(PathBuf),
    /// A file, appended to.
    Append(PathBuf),
    /// This process's stderr (stdout -> the container log).
    ToStderr,
}

/// One program to run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cmd {
    pub program: String,
    pub args: Vec<String>,
    /// Added to (or replacing in) the inherited environment.
    pub env: Vec<(String, String)>,
    /// Removed from the inherited environment.
    pub env_remove: Vec<String>,
    pub stdout: Io,
    pub stderr: Io,
}

impl Cmd {
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            ..Self::default()
        }
    }
    pub fn arg(mut self, a: impl Into<String>) -> Self {
        self.args.push(a.into());
        self
    }
    pub fn args<I, S>(mut self, a: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(a.into_iter().map(Into::into));
        self
    }
    pub fn env(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.env.push((k.into(), v.into()));
        self
    }
    pub fn env_remove(mut self, k: impl Into<String>) -> Self {
        self.env_remove.push(k.into());
        self
    }
    pub fn stdout(mut self, io: Io) -> Self {
        self.stdout = io;
        self
    }
    pub fn stderr(mut self, io: Io) -> Self {
        self.stderr = io;
        self
    }
    /// `program arg …` for logs.
    pub fn display(&self) -> String {
        std::iter::once(self.program.as_str())
            .chain(self.args.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// A finished program with its output captured.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Output {
    /// The exit code; `None` when a signal ended it.
    pub code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl Output {
    pub fn ok(&self) -> bool {
        self.code == Some(0)
    }
    pub fn stdout_text(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }
    pub fn stderr_text(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SysError {
    #[error("{program}: {source}")]
    Spawn {
        program: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{url}: {why}")]
    Http { url: String, why: String },
    #[error("{url}: HTTP {status}")]
    Status { url: String, status: u16 },
}

impl SysError {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// Runs programs.
pub trait Spawner {
    /// Runs to completion with stdout and stderr captured (the `Cmd`'s own
    /// stdio settings are ignored).
    fn output(&self, cmd: &Cmd) -> Result<Output, SysError>;
    /// Runs to completion with the `Cmd`'s stdio; the exit code (`None`: a
    /// signal ended it).
    fn status(&self, cmd: &Cmd) -> Result<Option<i32>, SysError>;
    /// Runs to completion with stdout and stderr both copied, line by line, to
    /// `log` (truncated first) and to this process's stderr (`… | tee log >&2`).
    fn tee(&self, cmd: &Cmd, log: &Path) -> Result<Option<i32>, SysError>;
    /// Starts it in the background (it outlives an `exec` of this process);
    /// its pid.
    fn spawn(&self, cmd: &Cmd) -> Result<u32, SysError>;
}

/// An HTTP method the boot (and fv-deploy, `crate::provider`) uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
    Delete,
    Patch,
}

impl Method {
    pub fn as_str(self) -> &'static str {
        match self {
            Method::Get => "GET",
            Method::Post => "POST",
            Method::Delete => "DELETE",
            Method::Patch => "PATCH",
        }
    }
}

/// One HTTP request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub method: Method,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
    pub timeout: Duration,
}

impl Request {
    pub fn new(method: Method, url: impl Into<String>, timeout: Duration) -> Self {
        Self {
            method,
            url: url.into(),
            headers: Vec::new(),
            body: None,
            timeout,
        }
    }
    pub fn bearer(self, token: &str) -> Self {
        self.header("authorization", &format!("Bearer {token}"))
    }
    pub fn header(mut self, k: &str, v: &str) -> Self {
        self.headers.push((k.to_owned(), v.to_owned()));
        self
    }
    pub fn json(mut self, body: String) -> Self {
        self.headers
            .push(("content-type".into(), "application/json".into()));
        self.body = Some(body.into_bytes());
        self
    }
}

/// An HTTP response (any status: the caller decides).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}

impl Response {
    pub fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// HTTP.
pub trait Http {
    fn send(&self, req: &Request) -> Result<Response, SysError>;
    /// GET `url` into `dest` (created or truncated); the bytes written. A
    /// non-2xx status is an error.
    fn download(&self, url: &str, dest: &Path, timeout: Duration) -> Result<u64, SysError>;
}

/// Time.
pub trait Clock {
    /// Unix time in seconds.
    fn now(&self) -> f64;
    fn sleep(&self, d: Duration);
}

/// Where the boot's log lines go (the container log: stderr).
pub trait LogSink {
    fn line(&self, s: &str);
}

// ------------------------------------------------------------------ real

fn stdio(io: &Io) -> Result<Stdio, SysError> {
    Ok(match io {
        Io::Inherit => Stdio::inherit(),
        Io::Null => Stdio::null(),
        Io::Truncate(p) => Stdio::from(
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(p)
                .map_err(|e| SysError::io(p, e))?,
        ),
        Io::Append(p) => Stdio::from(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .map_err(|e| SysError::io(p, e))?,
        ),
        Io::ToStderr => {
            let fd = std::io::stderr()
                .as_fd()
                .try_clone_to_owned()
                .map_err(|e| SysError::io("/dev/stderr", e))?;
            Stdio::from(fd)
        }
    })
}

fn command(cmd: &Cmd) -> Command {
    let mut c = Command::new(&cmd.program);
    c.args(&cmd.args);
    for k in &cmd.env_remove {
        c.env_remove(k);
    }
    for (k, v) in &cmd.env {
        c.env(k, v);
    }
    c
}

/// `std::process`.
#[derive(Clone, Copy, Debug, Default)]
pub struct RealSpawner;

impl Spawner for RealSpawner {
    fn output(&self, cmd: &Cmd) -> Result<Output, SysError> {
        let o = command(cmd)
            .stdin(Stdio::null())
            .output()
            .map_err(|source| SysError::Spawn {
                program: cmd.program.clone(),
                source,
            })?;
        Ok(Output {
            code: o.status.code(),
            stdout: o.stdout,
            stderr: o.stderr,
        })
    }

    fn status(&self, cmd: &Cmd) -> Result<Option<i32>, SysError> {
        let st = command(cmd)
            .stdin(Stdio::null())
            .stdout(stdio(&cmd.stdout)?)
            .stderr(stdio(&cmd.stderr)?)
            .status()
            .map_err(|source| SysError::Spawn {
                program: cmd.program.clone(),
                source,
            })?;
        Ok(st.code())
    }

    fn tee(&self, cmd: &Cmd, log: &Path) -> Result<Option<i32>, SysError> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(log)
            .map_err(|e| SysError::io(log, e))?;
        let file = Arc::new(Mutex::new(file));
        let mut child = command(cmd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| SysError::Spawn {
                program: cmd.program.clone(),
                source,
            })?;
        let mut pumps = Vec::new();
        let outs: Vec<Box<dyn Read + Send>> = [
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
        ]
        .into_iter()
        .flatten()
        .collect();
        for r in outs {
            let file = Arc::clone(&file);
            pumps.push(std::thread::spawn(move || {
                let mut rd = BufReader::new(r);
                let mut line = Vec::new();
                loop {
                    line.clear();
                    match rd.read_until(b'\n', &mut line) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {
                            if let Ok(mut f) = file.lock() {
                                let _ = f.write_all(&line);
                            }
                            let _ = std::io::stderr().write_all(&line);
                        }
                    }
                }
            }));
        }
        let st = child.wait().map_err(|source| SysError::Spawn {
            program: cmd.program.clone(),
            source,
        })?;
        for p in pumps {
            let _ = p.join();
        }
        Ok(st.code())
    }

    fn spawn(&self, cmd: &Cmd) -> Result<u32, SysError> {
        let child = command(cmd)
            .stdin(Stdio::null())
            .stdout(stdio(&cmd.stdout)?)
            .stderr(stdio(&cmd.stderr)?)
            .spawn()
            .map_err(|source| SysError::Spawn {
                program: cmd.program.clone(),
                source,
            })?;
        Ok(child.id())
    }
}

/// The system clock.
#[derive(Clone, Copy, Debug, Default)]
pub struct RealClock;

impl Clock for RealClock {
    fn now(&self) -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
    fn sleep(&self, d: Duration) {
        std::thread::sleep(d);
    }
}

/// The container log.
#[derive(Clone, Copy, Debug, Default)]
pub struct StderrLog;

impl LogSink for StderrLog {
    fn line(&self, s: &str) {
        let mut e = std::io::stderr().lock();
        let _ = writeln!(e, "{s}");
    }
}

/// A signal to a process through the shell's `kill` builtin: `sh -c 'kill -s
/// "$0" "$1"' <SIG> <pid>` (the signal and the pid are the script's `$0` and
/// `$1`, never spliced into it). The serve images have no procps, so no `kill`
/// / `pkill` binary can be assumed.
pub fn kill_cmd(sig: &str, pid: u32) -> Cmd {
    Cmd::new("sh").args(["-c", "kill -s \"$0\" \"$1\"", sig, &pid.to_string()])
}

/// `HH:MM:SS` (UTC) of a unix time.
pub fn utc_hms(unix: f64) -> String {
    let s = if unix.is_finite() && unix > 0.0 {
        unix as u64 % 86_400
    } else {
        0
    };
    format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// The first executable named `name` on `PATH` (`command -v`).
pub fn which(name: &str, path: Option<&str>) -> Option<PathBuf> {
    if name.contains('/') {
        let p = PathBuf::from(name);
        return is_executable(&p).then_some(p);
    }
    path?
        .split(':')
        .filter(|d| !d.is_empty())
        .map(|d| Path::new(d).join(name))
        .find(|p| is_executable(p))
}

/// A regular file with an execute bit.
pub fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(p)
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hms_is_utc_time_of_day() {
        assert_eq!(utc_hms(0.0), "00:00:00");
        assert_eq!(utc_hms(1_761_300_000.9), "10:00:00");
        assert_eq!(utc_hms(86_399.0), "23:59:59");
    }

    #[test]
    fn cmd_display_joins_program_and_args() {
        let c = Cmd::new("python3").args(["a", "b"]).arg("c");
        assert_eq!(c.display(), "python3 a b c");
    }
}
