//! `fv-gpucheck serve`: a read-only static file server for a run directory.
//!
//! Rented boxes publish their logs and reports over an HTTP port (the Runpod
//! HTTPS proxy) so a driver without SSH can follow a run. This keeps that out
//! of the box's package manager: no Python, no apt, no network needed on the
//! box. GET/HEAD only; paths are confined to the served root; directories get
//! a plain listing whose links match what `python -m http.server` emits
//! (`href="name"` / `href="dir/"`).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};

pub fn run(root: &Path, port: u16) -> anyhow::Result<()> {
    let root = root.canonicalize()?;
    let listener = TcpListener::bind(("0.0.0.0", port))?;
    eprintln!("serving {} on :{port}", root.display());
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let root = root.clone();
        std::thread::spawn(move || {
            let _ = handle(&root, stream);
        });
    }
    Ok(())
}

fn handle(root: &Path, mut stream: TcpStream) -> std::io::Result<()> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    // Drain the headers; nothing in them changes the answer.
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
    }
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or("/"));
    if method != "GET" && method != "HEAD" {
        return respond(&mut stream, 405, "text/plain", b"method not allowed", true);
    }
    let head_only = method == "HEAD";
    let Some(path) = resolve(root, target) else {
        return respond(&mut stream, 404, "text/plain", b"not found", head_only);
    };
    if path.is_dir() {
        let body = listing(&path)?;
        return respond(&mut stream, 200, "text/html; charset=utf-8", body.as_bytes(), head_only);
    }
    let Ok(mut file) = std::fs::File::open(&path) else {
        return respond(&mut stream, 404, "text/plain", b"not found", head_only);
    };
    let len = file.metadata()?.len();
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n",
        content_type(&path)
    )?;
    if !head_only {
        let mut buf = vec![0u8; 1 << 16];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            stream.write_all(&buf[..n])?;
        }
    }
    stream.flush()
}

/// Map a request target onto a path under `root`, refusing anything that
/// would leave it (`..`, absolute components, symlinks pointing outside).
fn resolve(root: &Path, target: &str) -> Option<PathBuf> {
    let raw = target.split(['?', '#']).next().unwrap_or("/");
    let decoded = percent_decode(raw)?;
    let mut path = root.to_path_buf();
    for c in Path::new(decoded.trim_start_matches('/')).components() {
        match c {
            Component::Normal(p) => path.push(p),
            Component::CurDir => {}
            _ => return None,
        }
    }
    let real = path.canonicalize().ok()?;
    real.starts_with(root).then_some(real)
}

fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = std::str::from_utf8(bytes.get(i + 1..i + 3)?).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

fn listing(dir: &Path) -> std::io::Result<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if e.path().is_dir() {
                format!("{name}/")
            } else {
                name
            }
        })
        .collect();
    names.sort();
    let mut body = String::from("<!DOCTYPE html>\n<html><body><ul>\n");
    for n in names {
        body.push_str(&format!("<li><a href=\"{n}\">{n}</a></li>\n"));
    }
    body.push_str("</ul></body></html>\n");
    Ok(body)
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("json") => "application/json",
        Some("log" | "txt" | "out" | "tsv" | "csv") => "text/plain; charset=utf-8",
        Some("html") => "text/html; charset=utf-8",
        Some("mp4") => "video/mp4",
        Some("png") => "image/png",
        Some("wav") => "audio/wav",
        _ => "application/octet-stream",
    }
}

fn respond(
    stream: &mut TcpStream,
    code: u16,
    ctype: &str,
    body: &[u8],
    head_only: bool,
) -> std::io::Result<()> {
    let reason = match code {
        200 => "OK",
        404 => "Not Found",
        _ => "Method Not Allowed",
    };
    write!(
        stream,
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    if !head_only {
        stream.write_all(body)?;
    }
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_stays_inside_the_root() {
        let d = std::env::temp_dir().join(format!("fv-serve-{}", std::process::id()));
        std::fs::create_dir_all(d.join("a")).unwrap();
        std::fs::write(d.join("a/x.log"), "hi").unwrap();
        let root = d.canonicalize().unwrap();
        assert_eq!(resolve(&root, "/a/x.log"), Some(root.join("a/x.log")));
        assert_eq!(resolve(&root, "/a/x%2Elog?q=1"), Some(root.join("a/x.log")));
        assert_eq!(resolve(&root, "/"), Some(root.clone()));
        assert_eq!(resolve(&root, "/../etc/passwd"), None);
        assert_eq!(resolve(&root, "/a/../../etc"), None);
        assert_eq!(resolve(&root, "/missing"), None);
        assert!(listing(&root).unwrap().contains("href=\"a/\""));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
