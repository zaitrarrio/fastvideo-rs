//! `fv-lab pod fetch --fetch-tree` against `runpod-http.sh fetch_tree`
//! (`FV_FETCH_TREE=1`, docs/adr/0019-fv-lab.md §13): the script's own
//! function, cut from the script, runs against a fake `curl` serving a fake
//! pod's run directory; fv-lab's `fetch_tree` runs against the same files
//! through a fake HTTP client. The two trees are byte-identical, with and
//! without a skip pattern and a size limit.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use fastvideo_deploy::boot::sys::{Clock, Http, LogSink, Request, Response, SysError};
use fastvideo_lab::pod::{fetch_tree, FetchSkip, TreeFetch};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

const POD: &str = "pod123";

/// The pod's server: path (as requested, after the host) -> body. A path
/// ending in `/` is a directory listing.
fn site() -> BTreeMap<String, Vec<u8>> {
    let list = |entries: &[&str]| -> Vec<u8> {
        let mut s = String::from("<!DOCTYPE HTML>\n<html><body><ul>\n");
        for e in entries {
            s.push_str(&format!("<li><a href=\"{e}\">{e}</a></li>\n"));
        }
        s.push_str("</ul></body></html>\n");
        s.into_bytes()
    };
    let mut m = BTreeMap::new();
    m.insert(
        "fam/t1/".into(),
        list(&[
            "../",
            "/abs",
            "?C=N",
            "DONE",
            "live.log",
            "big.bin",
            "gone.json",
            "my%20file.txt",
            "fam.verdict.json",
            "arm-a/",
            "dump/",
            "sub%20dir/",
        ]),
    );
    m.insert("fam/t1/DONE".into(), b"done t1\n".to_vec());
    m.insert(
        "fam/t1/live.log".into(),
        b"[00:00:00] [fam] matrix start\n".to_vec(),
    );
    m.insert("fam/t1/big.bin".into(), vec![7u8; (1 << 20) + 1]);
    m.insert("fam/t1/my%20file.txt".into(), b"spaced\n".to_vec());
    m.insert(
        "fam/t1/fam.verdict.json".into(),
        b"{\"verdict\":\"pass\"}".to_vec(),
    );
    m.insert(
        "fam/t1/arm-a/".into(),
        list(&["summary.json", "out.mp4", "frames/", "x.cache"]),
    );
    m.insert("fam/t1/arm-a/summary.json".into(), b"{\"exit\":0}".to_vec());
    m.insert("fam/t1/arm-a/out.mp4".into(), b"mp4".to_vec());
    m.insert("fam/t1/arm-a/x.cache".into(), b"cache".to_vec());
    m.insert("fam/t1/arm-a/frames/".into(), list(&["frame-000.png"]));
    m.insert("fam/t1/arm-a/frames/frame-000.png".into(), b"png".to_vec());
    m.insert("fam/t1/dump/".into(), list(&["t.bin"]));
    m.insert("fam/t1/dump/t.bin".into(), b"t".to_vec());
    m.insert("fam/t1/sub%20dir/".into(), list(&["a.txt"]));
    m.insert("fam/t1/sub%20dir/a.txt".into(), b"a".to_vec());
    m
}

struct Fake(BTreeMap<String, Vec<u8>>);

impl Http for Fake {
    fn send(&self, req: &Request) -> Result<Response, SysError> {
        let base = format!("https://{POD}-8000.proxy.runpod.net/");
        let path = req.url.strip_prefix(&base).unwrap_or("");
        Ok(match self.0.get(path) {
            Some(b) => Response {
                status: 200,
                body: b.clone(),
            },
            None => Response {
                status: 404,
                body: Vec::new(),
            },
        })
    }
    fn download(&self, url: &str, dest: &Path, _: Duration) -> Result<u64, SysError> {
        let base = format!("https://{POD}-8000.proxy.runpod.net/");
        let path = url.strip_prefix(&base).unwrap_or("");
        match self.0.get(path) {
            Some(b) => {
                std::fs::write(dest, b).map_err(|e| SysError::io(dest, e))?;
                Ok(b.len() as u64)
            }
            None => Err(SysError::Status {
                url: url.into(),
                status: 404,
            }),
        }
    }
}

struct Still;
impl Clock for Still {
    fn now(&self) -> f64 {
        0.0
    }
    fn sleep(&self, _: Duration) {}
}

struct Quiet;
impl LogSink for Quiet {
    fn line(&self, _: &str) {}
}

/// Every file under `d`: relative path -> bytes.
fn tree(d: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(base: &Path, d: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let p = e.path();
            let rel = p.strip_prefix(base).unwrap().to_string_lossy().into_owned();
            if p.is_dir() {
                out.insert(format!("{rel}/"), Vec::new());
                walk(base, &p, out);
            } else {
                out.insert(rel, std::fs::read(&p).unwrap());
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(d, d, &mut out);
    out
}

/// The script's `proxy` and `fetch_tree`, cut from runpod-http.sh, against a
/// fake `curl` that serves `site/`.
fn script_tree(work: &Path, skip: Option<&str>, max_mb: Option<u64>) -> PathBuf {
    let http = std::fs::read_to_string(root().join("scripts/gpu/runpod-http.sh")).unwrap();
    let proxy = http
        .lines()
        .find(|l| l.starts_with("proxy() {"))
        .expect("runpod-http.sh proxy()")
        .to_owned();
    let start = http
        .find("\nfetch_tree() {\n")
        .expect("runpod-http.sh fetch_tree()")
        + 1;
    let end = start + http[start..].find("\n}\n").expect("end of fetch_tree") + 3;
    let func = &http[start..end];
    let site_dir = work.join("site");
    for (path, body) in site() {
        let p = if path.ends_with('/') {
            site_dir.join(&path).join("index.html")
        } else {
            site_dir.join(&path)
        };
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }
    let bin = work.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    // curl -sS --max-time N --fail [--max-filesize N] URL [-o FILE]
    std::fs::write(
        bin.join("curl"),
        format!(
            r#"#!/usr/bin/env bash
out="" max="" url=""
while (( $# )); do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    --max-filesize) max="$2"; shift 2 ;;
    --max-time) shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
path="${{url#https://{POD}-8000.proxy.runpod.net/}}"
f="{site}/$path"
case "$path" in */ | "") f="{site}/${{path}}index.html" ;; esac
[[ -f "$f" ]] || exit 22
if [[ -n "$max" ]] && (( max > 0 )) && (( $(stat -c %s "$f") > max )); then exit 63; fi
if [[ -n "$out" ]]; then cp "$f" "$out"; else cat "$f"; fi
"#,
            site = site_dir.display()
        ),
    )
    .unwrap();
    Command::new("chmod")
        .arg("+x")
        .arg(bin.join("curl"))
        .status()
        .unwrap();
    let dest = work.join("script");
    let mut env = String::new();
    if let Some(s) = skip {
        env.push_str(&format!("export FV_FETCH_SKIP='{s}'\n"));
    }
    if let Some(m) = max_mb {
        env.push_str(&format!("export FV_FETCH_MAX_MB={m}\n"));
    }
    let script = format!(
        "set -euo pipefail\n{proxy}\n{func}\n{env}fetch_tree {POD} fam/t1/ '{}'\n",
        dest.display()
    );
    let st = Command::new("bash")
        .arg("-c")
        .arg(&script)
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("LC_ALL", "C")
        .status()
        .unwrap();
    assert!(st.success(), "the script's fetch_tree");
    dest
}

fn case(name: &str, skip: Option<&str>, max_mb: Option<u64>) {
    let work = std::env::temp_dir().join(format!("fv-lab-tree-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    let theirs = tree(&script_tree(&work, skip, max_mb));
    let ours_dir = work.join("fv-lab");
    let opts = TreeFetch {
        skip: skip.map(|s| FetchSkip::parse(s).unwrap()),
        max_mb: max_mb.unwrap_or(400),
    };
    fetch_tree(
        &Fake(site()),
        &format!("https://{POD}-8000.proxy.runpod.net"),
        "fam",
        "t1",
        &ours_dir,
        &opts,
        &Still,
        &Quiet,
    )
    .unwrap();
    let ours = tree(&ours_dir);
    assert_eq!(
        ours.keys().collect::<Vec<_>>(),
        theirs.keys().collect::<Vec<_>>(),
        "{name}: the files"
    );
    assert!(ours == theirs, "{name}: the bytes");
    assert!(ours.contains_key("DONE"), "{name}");
    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn the_tree_fetch_is_the_scripts() {
    case("all", None, None);
    case("frames", Some("/frames/"), None);
    case("oracle", Some(r"oracle-dump\.tar|/dump/|\.mp4$"), Some(1));
    case("anchored", Some(r"^fam/t1/arm-./x\.cache$|my%20"), Some(0));
}
