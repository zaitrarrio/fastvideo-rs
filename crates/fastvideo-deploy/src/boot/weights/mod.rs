//! The weights plan: which source fills `FV_WEIGHTS` before fv-serve starts
//! (`FV_WEIGHTS_SOURCE`, ADR-0014 §3.3).
//!
//! - `volume` (the default): a network volume is mounted; nothing is fetched.
//! - `r2`: the R2 weights store, through the existing `scripts/gpu/r2-fetch.py`
//!   (presigned plan from fv-control, ranged GETs, every part checked), with
//!   streaming into a fv-serve that can load while the rest arrives ([`r2`]).
//!   It stays behind this source until the registry's `hydrate` replaces it.
//! - `hub`: the Hugging Face Hub at pinned revisions (owner-approved), through
//!   `fetch-hub-tree.py` ([`hub`]).
//! - `none`: the config loads no weight tree.
//! - `registry`: content-defined chunks from the model registry, hydrated in
//!   parallel (model-registry-rs `hydrate`, WO-002). Typed here and refused
//!   until that crate exists: a pod asked for it fails loudly instead of
//!   serving without weights.
//!
//! Each source is a [`WeightsSource`]; the `aux` rows (TAE decoders, LPIPS:
//! pinned URL + SHA-256) are shared ([`aux_files`]).

pub mod hub;
pub mod r2;

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::error::BootError;
use super::plan::{self, AuxRow, Row};
use super::sys::SysError;
use super::Ctx;

/// `FV_WEIGHTS_SOURCE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceKind {
    Volume,
    R2,
    Hub,
    None,
    Registry,
}

impl SourceKind {
    /// Unset means `volume` (every shell boot's default).
    pub fn parse(v: Option<&str>) -> Result<Self, BootError> {
        Ok(match v {
            None | Some("volume") => Self::Volume,
            Some("r2") => Self::R2,
            Some("hub") => Self::Hub,
            Some("none") => Self::None,
            Some("registry") => Self::Registry,
            Some(o) => return Err(BootError::UnknownSource(o.to_owned())),
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Volume => "volume",
            Self::R2 => "r2",
            Self::Hub => "hub",
            Self::None => "none",
            Self::Registry => "registry",
        }
    }
}

/// A streaming fetch still running when fv-serve starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Streaming {
    pub state: PathBuf,
}

/// What a source leaves for the rest of the boot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Prepared {
    /// Where the weights went (r2 / hub: the serverless `serve` report names it).
    pub root: Option<PathBuf>,
    /// A fetch that goes on while fv-serve loads.
    pub streaming: Option<Streaming>,
}

/// One weights source.
pub trait WeightsSource {
    fn kind(&self) -> SourceKind;
    /// Fills the weights root (or checks it), before fv-serve starts.
    fn prepare(&self, c: &mut Ctx<'_>) -> Result<Prepared, BootError>;
}

/// The mounted volume: nothing to fetch (the volume marker and the tree
/// preflight are the boot's own steps).
pub struct Volume;
impl WeightsSource for Volume {
    fn kind(&self) -> SourceKind {
        SourceKind::Volume
    }
    fn prepare(&self, _c: &mut Ctx<'_>) -> Result<Prepared, BootError> {
        Ok(Prepared::default())
    }
}

/// No weights.
pub struct NoWeights;
impl WeightsSource for NoWeights {
    fn kind(&self) -> SourceKind {
        SourceKind::None
    }
    fn prepare(&self, c: &mut Ctx<'_>) -> Result<Prepared, BootError> {
        c.say("weights: none (the config loads no weight tree)");
        Ok(Prepared::default())
    }
}

/// The registry's content-defined chunks (`hydrate`, model-registry-rs WO-002).
pub struct Registry;
/// Why [`Registry`] refuses.
pub const REGISTRY_PENDING: &str =
    "the registry's hydrate crate (model-registry-rs WO-002) is not in this image yet; use volume, r2 or hub";
impl WeightsSource for Registry {
    fn kind(&self) -> SourceKind {
        SourceKind::Registry
    }
    fn prepare(&self, _c: &mut Ctx<'_>) -> Result<Prepared, BootError> {
        Err(BootError::SourceNotAvailable {
            name: "registry",
            why: REGISTRY_PENDING,
        })
    }
}

/// The source of a kind.
pub fn source(k: SourceKind) -> Box<dyn WeightsSource> {
    match k {
        SourceKind::Volume => Box::new(Volume),
        SourceKind::R2 => Box::new(r2::R2),
        SourceKind::Hub => Box::new(hub::Hub),
        SourceKind::None => Box::new(NoWeights),
        SourceKind::Registry => Box::new(Registry),
    }
}

// ------------------------------------------------------------------ helpers

/// Lower-case hex.
pub fn hex(b: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push(H[(x >> 4) as usize] as char);
        s.push(H[(x & 15) as usize] as char);
    }
    s
}

/// A file's SHA-256 (hex) and size.
pub fn sha256_file(p: &Path) -> Result<(String, u64), SysError> {
    let mut f = File::open(p).map_err(|e| SysError::io(p, e))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = f.read(&mut buf).map_err(|e| SysError::io(p, e))?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((hex(&h.finalize()), n))
}

/// A tree complete at its revision: `.complete` and `.fv-revision` = `rev`.
pub fn tree_present(root: &Path, dest: &str, rev: &str) -> bool {
    let d = root.join(dest);
    d.join(".complete").is_file()
        && std::fs::read_to_string(d.join(".fv-revision"))
            .map(|r| r.trim_end_matches(['\n', '\r']) == rev)
            .unwrap_or(false)
}

/// Leftovers of an interrupted fetch in a root on the pod's own disk
/// (`.*.partial-*`, `.stale-*`; the shell's WEIGHTS_LOOP_SH). Never on an
/// add-only cache volume.
pub fn clean_leftovers(root: &Path) {
    if let Ok(rd) = std::fs::read_dir(root) {
        for e in rd.filter_map(Result::ok) {
            let n = e.file_name().to_string_lossy().into_owned();
            if n.starts_with(".stale-") || (n.starts_with('.') && n.contains(".partial-")) {
                let _ =
                    std::fs::remove_dir_all(e.path()).or_else(|_| std::fs::remove_file(e.path()));
            }
        }
    }
}

/// Whether `python3` is on `PATH`.
pub fn has_python(c: &Ctx<'_>) -> bool {
    super::sys::which("python3", c.get("PATH")).is_some()
}

/// The aux rows: each file at `root/<path>` must have its SHA-256; a file
/// missing or different comes from its pinned URL (checked by SHA-256 and
/// size before it is renamed into place). On an add-only root (a shared R2
/// cache volume) a different file is an error, never replaced. Returns
/// (rows checked, files fetched).
pub fn aux_files(
    c: &Ctx<'_>,
    root: &Path,
    rows: &[AuxRow],
    add_only: bool,
) -> Result<(usize, usize), BootError> {
    let mut got = 0;
    let suffix = c
        .get("RUNPOD_POD_ID")
        .or_else(|| c.get("FV_POD_ID"))
        .unwrap_or("w")
        .replace(['/', ':'], "_");
    for a in rows {
        let p = root.join(&a.path);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| SysError::io(parent, e))?;
        }
        if p.is_file() && sha256_file(&p).map(|(h, _)| h == a.sha256).unwrap_or(false) {
            continue;
        }
        if add_only && p.exists() {
            return Err(BootError::Weights(format!(
                "aux {} differs on the cache volume (add-only: not replaced)",
                a.path
            )));
        }
        got += 1;
        c.say(&format!("aux {} <- {}", a.path, a.url));
        let part = PathBuf::from(format!("{}.part-{suffix}", p.display()));
        let ok = c
            .deps
            .http
            .download(&a.url, &part, Duration::from_secs(600))
            .is_ok()
            && sha256_file(&part)
                .map(|(h, n)| h == a.sha256 && n == a.bytes)
                .unwrap_or(false);
        if !ok {
            let _ = std::fs::remove_file(&part);
            return Err(BootError::Weights(format!("aux {}", a.path)));
        }
        if add_only && p.exists() {
            let _ = std::fs::remove_file(&part);
        } else {
            std::fs::rename(&part, &p).map_err(|e| SysError::io(&p, e))?;
        }
    }
    Ok((rows.len(), got))
}

/// The aux rows of the environment's plan.
pub fn aux_rows(c: &Ctx<'_>) -> Result<Vec<AuxRow>, BootError> {
    Ok(plan::rows(c.get("FV_WEIGHTS_TREES_B64"))?
        .into_iter()
        .filter_map(|r| match r {
            Row::Aux(a) => Some(a),
            Row::Tree(_) => None,
        })
        .collect())
}

/// The installs a boot may need on a bare image (`apt-get install …`).
pub fn apt_install(c: &Ctx<'_>, pkgs: &[&str], log: &Path) -> bool {
    let opts = [
        "-o",
        "Acquire::http::Timeout=15",
        "-o",
        "Acquire::Retries=3",
    ];
    let upd = c
        .cmd("apt-get")
        .args(opts)
        .args(["update", "-qq"])
        .stdout(super::sys::Io::Append(log.into()))
        .stderr(super::sys::Io::Append(log.into()));
    if c.deps.spawner.status(&upd).ok().flatten() != Some(0) {
        return false;
    }
    let inst = c
        .cmd("apt-get")
        .env("DEBIAN_FRONTEND", "noninteractive")
        .args(opts)
        .args(["install", "-y", "-qq", "--no-install-recommends"])
        .args(pkgs.iter().copied())
        .stdout(super::sys::Io::Append(log.into()))
        .stderr(super::sys::Io::Append(log.into()));
    c.deps.spawner.status(&inst).ok().flatten() == Some(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_parse() {
        assert_eq!(SourceKind::parse(None).unwrap(), SourceKind::Volume);
        assert_eq!(SourceKind::parse(Some("r2")).unwrap(), SourceKind::R2);
        assert_eq!(
            SourceKind::parse(Some("registry")).unwrap(),
            SourceKind::Registry
        );
        assert!(matches!(
            SourceKind::parse(Some("s3")),
            Err(BootError::UnknownSource(_))
        ));
        for k in [
            SourceKind::Volume,
            SourceKind::R2,
            SourceKind::Hub,
            SourceKind::None,
            SourceKind::Registry,
        ] {
            assert_eq!(source(k).kind(), k);
            assert_eq!(SourceKind::parse(Some(k.name())).unwrap(), k);
        }
    }

    #[test]
    fn hex_and_sha() {
        assert_eq!(hex(&[0, 1, 0xab, 0xff]), "0001abff");
        let d = std::env::temp_dir().join(format!("fvb-sha-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let f = d.join("x");
        std::fs::write(&f, b"abc").unwrap();
        assert_eq!(
            sha256_file(&f).unwrap(),
            (
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
                3
            )
        );
        std::fs::remove_dir_all(&d).unwrap();
    }
}
