//! The tunnel of a GMI / Brev pod (outbound only; control providers.ts
//! `TUNNEL_SH`): cloudflared, pinned and checked by SHA-256, to
//! `127.0.0.1:8000`. A named tunnel fv-control made for the pod
//! (`FV_TUNNEL_TOKEN`, handed to cloudflared as `TUNNEL_TOKEN` in its
//! environment, never argv, and kept from fv-serve's; its URL `FV_TUNNEL_URL`)
//! counts once a connection registered; else a quick tunnel
//! (`https://<name>.trycloudflare.com` from cloudflared's log). The URL is
//! logged as the untimed `[fv-boot] tunnel <url>` marker (the Brev host
//! watcher greps it) and reported (`tunnel` phase).

use std::path::Path;
use std::time::Duration;

use super::error::BootError;
use super::sys::{is_executable, Io};
use super::weights::sha256_file;
use super::Ctx;

/// cloudflared, pinned (control providers.ts `CLOUDFLARED`).
pub const CLOUDFLARED_URL: &str =
    "https://github.com/cloudflare/cloudflared/releases/download/2026.9.0/cloudflared-linux-amd64";
pub const CLOUDFLARED_SHA256: &str =
    "53b7a7a5420d188758d24341294acb0d1bca54296548ac05e38811a694ac6134";

/// What a registered named tunnel logs.
pub const REGISTERED: &str = "Registered tunnel connection";

/// The first `https://<a-z0-9->.trycloudflare.com` in a log.
pub fn quick_url(log: &str) -> Option<String> {
    let mut rest = log;
    while let Some(i) = rest.find("https://") {
        let after = &rest[i + 8..];
        let name: String = after
            .chars()
            .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '-')
            .collect();
        if !name.is_empty() && after[name.len()..].starts_with(".trycloudflare.com") {
            return Some(format!("https://{name}.trycloudflare.com"));
        }
        rest = after;
    }
    None
}

/// The log's last 3 lines, tunnel tokens redacted (`eyJhIjoi…`), one line.
pub fn tail(log: &str) -> String {
    let lines: Vec<&str> = log.lines().collect();
    let t = lines[lines.len().saturating_sub(3)..].join(" ");
    let mut out = String::with_capacity(t.len());
    let mut rest = t.as_str();
    while let Some(i) = rest.find("eyJhIjoi") {
        out.push_str(&rest[..i]);
        let tok = &rest[i..];
        let n = tok
            .find(|c: char| !(c.is_ascii_alphanumeric() || "+/_=-".contains(c)))
            .unwrap_or(tok.len());
        out.push_str("[redacted]");
        rest = &tok[n..];
    }
    out.push_str(rest);
    out.replace('"', " ")
}

fn install(c: &Ctx<'_>, dest: &Path) -> Result<(), BootError> {
    let want = c.paths.cloudflared_sha256.as_str();
    if is_executable(dest) && sha256_file(dest).map(|(h, _)| h == want).unwrap_or(false) {
        return Ok(());
    }
    if let Some(p) = dest.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    c.deps
        .http
        .download(&c.paths.cloudflared_url, dest, Duration::from_secs(120))
        .map_err(|_| BootError::Tunnel("cloudflared download".into()))?;
    if !sha256_file(dest).map(|(h, _)| h == want).unwrap_or(false) {
        return Err(BootError::Tunnel("cloudflared sha256".into()));
    }
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dest, std::fs::Permissions::from_mode(0o755))
        .map_err(|_| BootError::Tunnel("cloudflared chmod".into()))
}

/// Starts the tunnel and waits for its URL (180 s); reports it.
pub fn run(c: &mut Ctx<'_>) -> Result<(), BootError> {
    let bin = c.paths.cloudflared.clone();
    install(c, &bin)?;
    let log = c.state_dir().join("tunnel.log");
    let read = |p: &Path| std::fs::read_to_string(p).unwrap_or_default();
    let named = c.env.get("FV_TUNNEL_TOKEN").map(str::to_owned);
    let url = if let Some(tok) = named {
        let want = c.get_or("FV_TUNNEL_URL", "").to_owned();
        c.say(&format!("named tunnel: {want}"));
        let cmd = c
            .cmd(bin.display().to_string())
            .args(["tunnel", "--no-autoupdate", "run"])
            .env("TUNNEL_TOKEN", tok)
            .env_remove("FV_TUNNEL_TOKEN")
            .stdout(Io::Truncate(log.clone()))
            .stderr(Io::Append(log.clone()));
        c.deps.spawner.spawn(&cmd)?;
        c.unset.push("FV_TUNNEL_TOKEN".into());
        let mut ok = false;
        for _ in 0..90 {
            if read(&log).contains(REGISTERED) {
                ok = true;
                break;
            }
            c.deps.clock.sleep(Duration::from_secs(2));
        }
        if !ok {
            return Err(BootError::Tunnel(format!(
                "named tunnel did not connect: {}",
                tail(&read(&log))
            )));
        }
        want
    } else {
        let cmd = c
            .cmd(bin.display().to_string())
            .args([
                "tunnel",
                "--no-autoupdate",
                "--url",
                "http://127.0.0.1:8000",
            ])
            .stdout(Io::Truncate(log.clone()))
            .stderr(Io::Append(log.clone()));
        c.deps.spawner.spawn(&cmd)?;
        let mut u = None;
        for _ in 0..90 {
            u = quick_url(&read(&log));
            if u.is_some() {
                break;
            }
            c.deps.clock.sleep(Duration::from_secs(2));
        }
        u.ok_or_else(|| BootError::Tunnel(format!("no tunnel URL: {}", tail(&read(&log)))))?
    };
    c.log.marker(&format!("tunnel {url}"));
    if c.get("FV_DISPATCH_FRONT") == Some("1") {
        c.export("FV_DISPATCH_ENDPOINT", url.clone());
    }
    c.pod_report("tunnel", &url, "");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_urls() {
        let l = "2026 INF |  https://www.cloudflare.com/x  |\n2026 INF |  https://brave-fox-12.trycloudflare.com  |\n";
        assert_eq!(
            quick_url(l).as_deref(),
            Some("https://brave-fox-12.trycloudflare.com")
        );
        assert_eq!(quick_url("https://X.trycloudflare.com"), None);
        assert_eq!(quick_url(""), None);
    }

    #[test]
    fn tails_are_redacted() {
        let l = "a\nb\nERR token eyJhIjoiabc+/=_-xyz bad \"q\"\nc";
        assert_eq!(tail(l), "b ERR token [redacted] bad  q  c");
    }
}
