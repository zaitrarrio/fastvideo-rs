//! The admin token (`/fv/v1/admin/*`, `/fv/v1/gateway/pools`, key minting
//! in `/console/admin`): docs/serve/gateway.md §9.
//!
//! - `FV_ADMIN_TOKEN` (`auth.admin_token`) wins when set.
//! - Otherwise the token lives in `<state_dir>/admin_token` (mode 600): made
//!   once, on the first start, and reused by every later start with the same
//!   state dir. It is never logged: the log names the file and the token's
//!   first 4 characters. (A Runpod pod without a volume keeps its state dir
//!   on the container disk: it survives a restart, not a re-creation.)
//! - `FV_ADMIN_TOKEN_RECIPIENT` (`auth.admin_token_recipient`): an X25519
//!   public key (32 bytes, base64). The server then publishes the token
//!   sealed to that key at `GET /fv/v1/admin/token/sealed`, so an operator
//!   holding the private key (the cluster script) can fetch it over the
//!   public URL without the token ever being in a pod's environment, a log
//!   or a response anyone else can open.
//!
//! Sealing (`alg` `X25519-SHA512-AES256CTR-HMACSHA256`, all binary fields
//! base64): an ephemeral X25519 key `e`; `s = X25519(e, recipient)`;
//! `k = SHA-512("fv-admin-token-v1" ‖ s ‖ epk ‖ recipient)`; `ct` =
//! AES-256-CTR(`k[0..32]`, `iv`) of the token; `tag` =
//! HMAC-SHA256(`k[32..64]`, `iv ‖ ct`). Everything a stock `openssl` CLI can
//! open (`scripts/serve/runpod-cluster.sh admin-token`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use base64::Engine as _;
use fastvideo_serve_kit::AdminToken;
use serde_json::{json, Value};

use crate::config::Config;

/// The file name under `server.state_dir`.
pub const FILE_NAME: &str = "admin_token";
/// The sealing scheme (see the module docs).
pub const ALG: &str = "X25519-SHA512-AES256CTR-HMACSHA256";
const KDF_LABEL: &[u8] = b"fv-admin-token-v1";

/// Where the token came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// `FV_ADMIN_TOKEN` / `auth.admin_token`.
    Configured,
    /// `<state_dir>/admin_token`; `created` on this start.
    File { path: PathBuf, created: bool },
    /// Random for this process only (workers: the admin routes sit behind
    /// the internal token and nobody needs the value).
    Ephemeral,
}

/// The resolved admin token.
pub struct Resolved {
    pub token: AdminToken,
    pub source: Source,
    /// The sealed copy for `GET /fv/v1/admin/token/sealed`.
    pub sealed: Option<Value>,
}

/// Resolves the admin token (see the module docs); `persist = false` for a
/// worker (no file, no log line).
pub fn resolve(config: &Config, persist: bool) -> anyhow::Result<Resolved> {
    let (plain, source) = if !config.auth.admin_token.is_empty() {
        (config.auth.admin_token.expose().trim().to_owned(), Source::Configured)
    } else if !persist {
        (AdminToken::generate().1, Source::Ephemeral)
    } else {
        let path = config.server.state_dir.join(FILE_NAME);
        let (plain, created) = load_or_create(&path)?;
        (plain, Source::File { path, created })
    };
    match &source {
        Source::File { path, created } => tracing::info!(
            file = %path.display(),
            starts_with = %plain.chars().take(4).collect::<String>(),
            created,
            "admin token: {} (read it on the server; FV_ADMIN_TOKEN overrides)",
            if *created { "generated and stored" } else { "reused" }
        ),
        Source::Configured => tracing::info!("admin token: from FV_ADMIN_TOKEN"),
        Source::Ephemeral => {}
    }
    let sealed = match config.auth.admin_token_recipient.as_deref().map(str::trim).filter(|r| !r.is_empty()) {
        Some(r) => Some(seal(&plain, r).map_err(|e| anyhow::anyhow!("auth.admin_token_recipient: {e}"))?),
        None => None,
    };
    Ok(Resolved { token: AdminToken::from_secret(&plain), source, sealed })
}

/// Reads the token file, or creates it (mode 600, written under a temporary
/// name and renamed, so a crash never leaves half a token).
fn load_or_create(path: &Path) -> anyhow::Result<(String, bool)> {
    match std::fs::read_to_string(path) {
        Ok(s) if !s.trim().is_empty() => {
            restrict(path);
            return Ok((s.trim().to_owned(), false));
        }
        Ok(_) => tracing::warn!(file = %path.display(), "admin token file is empty; generating a new token"),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let (_, plain) = AdminToken::generate();
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    {
        use std::io::Write;
        let mut o = std::fs::OpenOptions::new();
        o.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            o.mode(0o600);
        }
        let mut f = o.open(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
        f.write_all(plain.as_bytes())?;
        f.write_all(b"\n")?;
        f.sync_all()?;
    }
    restrict(&tmp);
    std::fs::rename(&tmp, path).with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
    Ok((plain, true))
}

/// Mode 600 (an existing file copied in with looser rights is tightened).
fn restrict(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(m) = std::fs::metadata(path) {
            if m.permissions().mode() & 0o077 != 0 {
                let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
            }
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

fn b64(b: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(b)
}

/// Seals `plain` to the X25519 public key `recipient` (base64, standard or
/// URL-safe); see the module docs.
pub fn seal(plain: &str, recipient: &str) -> Result<Value, String> {
    use aes::cipher::{KeyIvInit, StreamCipher};
    use hmac::Mac;
    use sha2::Digest;
    let raw = base64::engine::general_purpose::STANDARD
        .decode(recipient)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(recipient.trim_end_matches('=')))
        .map_err(|_| "not base64".to_owned())?;
    let rpk: [u8; 32] = raw.try_into().map_err(|_| "an X25519 public key is 32 bytes".to_owned())?;
    let esk: [u8; 32] = rand::random();
    let epk = x25519_dalek::x25519(esk, x25519_dalek::X25519_BASEPOINT_BYTES);
    let shared = x25519_dalek::x25519(esk, rpk);
    if shared.iter().all(|b| *b == 0) {
        return Err("a low-order public key".into());
    }
    let mut h = sha2::Sha512::new();
    h.update(KDF_LABEL);
    h.update(shared);
    h.update(epk);
    h.update(rpk);
    let k = h.finalize();
    let iv: [u8; 16] = rand::random();
    let mut ct = plain.as_bytes().to_vec();
    let mut c = ctr::Ctr128BE::<aes::Aes256>::new((&k[..32]).into(), (&iv).into());
    c.apply_keystream(&mut ct);
    let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(&k[32..]).map_err(|e| e.to_string())?;
    mac.update(&iv);
    mac.update(&ct);
    let tag = mac.finalize().into_bytes();
    Ok(json!({
        "object": "fv.admin_token.sealed",
        "alg": ALG,
        "epk": b64(&epk),
        "iv": b64(&iv),
        "ct": b64(&ct),
        "tag": b64(&tag),
    }))
}

/// The X25519 public key of `secret` (tests and tools).
pub fn public_key(secret: [u8; 32]) -> [u8; 32] {
    x25519_dalek::x25519(secret, x25519_dalek::X25519_BASEPOINT_BYTES)
}

/// Opens a [`seal`]ed token with the recipient's private key (tests; the
/// cluster script does the same with `openssl`).
pub fn open(sealed: &Value, secret: [u8; 32]) -> Result<String, String> {
    use aes::cipher::{KeyIvInit, StreamCipher};
    use hmac::Mac;
    use sha2::Digest;
    let field = |k: &str| {
        sealed
            .get(k)
            .and_then(Value::as_str)
            .and_then(|s| base64::engine::general_purpose::STANDARD.decode(s).ok())
            .ok_or_else(|| format!("sealed token: `{k}` missing"))
    };
    let epk: [u8; 32] = field("epk")?.try_into().map_err(|_| "epk".to_owned())?;
    let iv: [u8; 16] = field("iv")?.try_into().map_err(|_| "iv".to_owned())?;
    let mut ct = field("ct")?;
    let tag = field("tag")?;
    let rpk = public_key(secret);
    let shared = x25519_dalek::x25519(secret, epk);
    let mut h = sha2::Sha512::new();
    h.update(KDF_LABEL);
    h.update(shared);
    h.update(epk);
    h.update(rpk);
    let k = h.finalize();
    let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(&k[32..]).map_err(|e| e.to_string())?;
    mac.update(&iv);
    mac.update(&ct);
    mac.verify_slice(&tag).map_err(|_| "sealed token: bad tag".to_owned())?;
    let mut c = ctr::Ctr128BE::<aes::Aes256>::new((&k[..32]).into(), (&iv).into());
    c.apply_keystream(&mut ct);
    String::from_utf8(ct).map_err(|_| "sealed token: not UTF-8".to_owned())
}

/// `GET /fv/v1/admin/token/sealed`: the sealed token, or 404 when no
/// recipient is configured. Public: only the private key's holder can open it.
pub fn routes<S: Clone + Send + Sync + 'static>(sealed: Option<Value>) -> Router<S> {
    let sealed = Arc::new(sealed);
    Router::new().route(
        "/fv/v1/admin/token/sealed",
        get(move || {
            let sealed = sealed.clone();
            async move {
                match sealed.as_ref() {
                    Some(v) => Json(v.clone()).into_response(),
                    None => not_configured(),
                }
            }
        }),
    )
}

fn not_configured() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({"error": {"kind": "not_found", "message": "no admin token recipient is configured (FV_ADMIN_TOKEN_RECIPIENT)"}})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(dir: &Path) -> Config {
        let mut c = Config::default();
        c.server.state_dir = dir.to_owned();
        c
    }

    #[test]
    fn stored_once_reused_and_overridden() {
        let dir = std::env::temp_dir().join(format!("fv-admin-{}", fastvideo_serve_kit::random_token()));
        let c = cfg(&dir);
        let a = resolve(&c, true).unwrap();
        let path = dir.join(FILE_NAME);
        assert_eq!(a.source, Source::File { path: path.clone(), created: true });
        let plain = std::fs::read_to_string(&path).unwrap().trim().to_owned();
        assert!(plain.starts_with(fastvideo_serve_kit::keys::ADMIN_PREFIX));
        assert!(a.token.check(&plain));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        // A restart reuses it.
        let b = resolve(&c, true).unwrap();
        assert_eq!(b.source, Source::File { path: path.clone(), created: false });
        assert!(b.token.check(&plain));
        // FV_ADMIN_TOKEN overrides and leaves the file alone.
        let mut o = cfg(&dir);
        o.auth.admin_token = crate::config::Secret("fvadm_chosen".into());
        let r = resolve(&o, true).unwrap();
        assert_eq!(r.source, Source::Configured);
        assert!(r.token.check("fvadm_chosen") && !r.token.check(&plain));
        assert_eq!(std::fs::read_to_string(&path).unwrap().trim(), plain);
        // Workers keep nothing.
        let w = resolve(&cfg(&dir.join("worker")), false).unwrap();
        assert_eq!(w.source, Source::Ephemeral);
        assert!(!dir.join("worker").join(FILE_NAME).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sealed_token_opens_with_the_private_key_only() {
        let sk: [u8; 32] = rand::random();
        let pk = x25519_dalek::x25519(sk, x25519_dalek::X25519_BASEPOINT_BYTES);
        let v = seal("fvadm_secret-value", &b64(&pk)).unwrap();
        assert_eq!(v["alg"], ALG);
        assert!(!v.to_string().contains("secret-value"));
        assert_eq!(open(&v, sk).unwrap(), "fvadm_secret-value");
        let other: [u8; 32] = rand::random();
        assert!(open(&v, other).is_err());
        let mut bad = v.clone();
        bad["ct"] = json!(b64(b"fvadm_other-value!"));
        assert!(open(&bad, sk).is_err(), "the tag covers the ciphertext");
        assert!(seal("x", "not base64!").is_err());
        assert!(seal("x", &b64(&[0u8; 32])).is_err(), "low-order key refused");
    }
}
