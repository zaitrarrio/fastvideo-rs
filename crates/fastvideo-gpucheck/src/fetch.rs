//! `fv-gpucheck fetch`: one pinned public file onto the box's disk.
//!
//! A lab arm (`lab/cells/*.toml`) can only run `fv-gpucheck`, and a tier's
//! weight trees come from the EU volume or the R2 store. A tree that is not
//! published there yet (a checkpoint under evaluation, e.g. History Forcing's
//! `dmd/model.pt` from the Hub) is staged by an ungated arm of this
//! subcommand, pinned by URL, SHA-256 and size, so the arms after it can
//! `requires` the file. Nothing here touches a volume or R2: the file lands
//! where `--out` says (the run's scratch), as `curl` writes it.
//!
//! `curl` is the box's own (the runtime image has it; the runner and the pod
//! start command use it the same way): `--proto =https` only, redirects
//! included, resumable (`-C -`), five retries. The download is checked
//! against the pinned SHA-256 and size before the `.part` file is renamed into
//! place; a file already there with the right hash and size is kept.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{anyhow, bail, Context};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::report::Report;

/// SHA-256 of `path`, streamed.
pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 16 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn pinned_ok(path: &Path, sha256: &str, size: u64) -> anyhow::Result<bool> {
    let Ok(meta) = std::fs::metadata(path) else {
        return Ok(false);
    };
    if !meta.is_file() || meta.len() != size {
        return Ok(false);
    }
    Ok(sha256_file(path)?.eq_ignore_ascii_case(sha256))
}

/// Download `url` to `out` unless `out` already holds the pinned bytes.
pub fn run(report: &mut Report, url: &str, sha256: &str, size: u64, out: &Path) -> anyhow::Result<()> {
    if !url.starts_with("https://") {
        bail!("fetch: {url}: only https:// URLs are fetched");
    }
    if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("fetch: --sha256 must be 64 hex digits");
    }
    let t0 = Instant::now();
    if pinned_ok(out, sha256, size)? {
        report.note(
            "fetch",
            json!({"url": url, "out": out.display().to_string(), "bytes": size, "cached": true}),
        );
        eprintln!("fetch ok (cached): {}", out.display());
        return Ok(());
    }
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
    }
    let part = out.with_extension(format!(
        "{}part",
        out.extension()
            .map(|e| format!("{}.", e.to_string_lossy()))
            .unwrap_or_default()
    ));
    let part_s = part.display().to_string();
    // The URL is an argument of --url, never an option; https only, redirects too.
    let status = Command::new("curl")
        .args([
            "-sS",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--retry-all-errors",
            "-C",
            "-",
            "-o",
            &part_s,
            "--url",
            url,
        ])
        .stdin(Stdio::null())
        .status()
        .with_context(|| "running curl")?;
    if !status.success() {
        bail!("fetch: curl {url} failed ({status})");
    }
    let got_size = std::fs::metadata(&part)?.len();
    if got_size != size {
        let _ = std::fs::remove_file(&part);
        bail!("fetch: {url}: {got_size} bytes, pinned {size}");
    }
    let got = sha256_file(&part)?;
    if !got.eq_ignore_ascii_case(sha256) {
        let _ = std::fs::remove_file(&part);
        bail!("fetch: {url}: sha256 {got}, pinned {sha256}");
    }
    std::fs::rename(&part, out).map_err(|e| anyhow!("rename {} -> {}: {e}", part_s, out.display()))?;
    let secs = t0.elapsed().as_secs_f64();
    report.note(
        "fetch",
        json!({"url": url, "out": out.display().to_string(), "bytes": size, "cached": false,
               "seconds": secs, "mb_per_s": size as f64 / 1e6 / secs.max(1e-9)}),
    );
    eprintln!(
        "fetch ok: {} ({size} bytes, {secs:.1}s)",
        out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_file_is_recognised_and_a_wrong_one_is_not() {
        let dir = std::env::temp_dir().join(format!("fv-fetch-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.bin");
        std::fs::write(&p, b"hello").unwrap();
        let sha = sha256_file(&p).unwrap();
        assert_eq!(
            sha,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        assert!(pinned_ok(&p, &sha, 5).unwrap());
        assert!(!pinned_ok(&p, &sha, 6).unwrap());
        assert!(!pinned_ok(&p, &"0".repeat(64), 5).unwrap());
        assert!(!pinned_ok(&dir.join("missing"), &sha, 5).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refuses_non_https_and_bad_hashes() {
        let dir = std::env::temp_dir().join(format!("fv-fetch-report-{}", std::process::id()));
        let mut r = Report::new(&dir, "fetch");
        let out = std::env::temp_dir().join("fv-fetch-never");
        assert!(run(&mut r, "http://example.com/x", &"a".repeat(64), 1, &out).is_err());
        assert!(run(&mut r, "https://example.com/x", "abc", 1, &out).is_err());
    }
}
