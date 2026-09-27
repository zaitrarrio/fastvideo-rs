//! Authentication (design §6.1): modes `none | keys | trust-gateway`, and a
//! per-API scheme policy.
//!
//! - fal and the fal director authenticate with `Authorization: Key <k>`
//!   (also `Key <id>:<secret>`, hashed as one token).
//! - MiniMax, LTX and the native `/fv/v1/*` API use `Authorization: Bearer <k>`.
//! - FastVideo `/v1/videos`, FastWan and the Reactor local runtime have no
//!   auth in their specs, so they are [`AuthPolicy::Open`] by default: a valid
//!   key (either scheme) still identifies the owner, a missing or wrong one is
//!   not an error. Each policy is configurable.
//!
//! Keys are configured as SHA-256 hashes (`FV_API_KEYS`, comma separated hex),
//! so the server never holds plaintext keys. The [`KeyId`] recorded on jobs is
//! a short prefix of the hash, never the key.

use std::collections::BTreeMap;

use axum::http::{header, HeaderMap};
use fastvideo_protocol::{ApiError, KeyId, ProtocolId};
use sha2::{Digest, Sha256};

/// Server-wide auth mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AuthMode {
    /// Everything is open (Reactor-local parity).
    None,
    /// API keys, per [`AuthPolicy`].
    #[default]
    Keys,
    /// An upstream gateway (Runpod LB) already authenticated the caller.
    TrustGateway,
}

/// An `Authorization` scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scheme {
    /// `Authorization: Bearer <key>`.
    Bearer,
    /// `Authorization: Key <key>` (fal).
    Key,
}

impl Scheme {
    fn prefix(&self) -> &'static str {
        match self {
            Scheme::Bearer => "bearer",
            Scheme::Key => "key",
        }
    }
}

/// What one API requires.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthPolicy {
    /// No key needed. A valid key still sets the owner.
    Open,
    /// One of these schemes with a configured key.
    Require(Vec<Scheme>),
}

/// Default per-API policy (see the module docs).
pub fn default_policy(p: ProtocolId) -> AuthPolicy {
    match p {
        ProtocolId::Fal | ProtocolId::FalDirector => AuthPolicy::Require(vec![Scheme::Key]),
        ProtocolId::MiniMaxV2 | ProtocolId::LtxV1 | ProtocolId::LtxV2 | ProtocolId::Native => {
            AuthPolicy::Require(vec![Scheme::Bearer])
        }
        ProtocolId::OpenAiVideos | ProtocolId::FastWan | ProtocolId::Reactor => AuthPolicy::Open,
    }
}

/// The configured key hashes.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct KeyRing {
    hashes: Vec<[u8; 32]>,
}

impl std::fmt::Debug for KeyRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KeyRing({} keys)", self.hashes.len())
    }
}

impl KeyRing {
    /// Parses `FV_API_KEYS`: comma/whitespace separated hex SHA-256 hashes.
    pub fn from_hash_list(list: &str) -> Result<Self, String> {
        let mut hashes = Vec::new();
        for item in list.split([',', ' ', '\n', '\t']).filter(|s| !s.is_empty()) {
            let bytes = crate::hex::decode(item.trim())
                .filter(|b| b.len() == 32)
                .ok_or_else(|| "FV_API_KEYS entries must be 64-hex SHA-256 hashes".to_owned())?;
            let mut h = [0u8; 32];
            h.copy_from_slice(&bytes);
            hashes.push(h);
        }
        Ok(Self { hashes })
    }

    /// A ring holding the hashes of plaintext keys (tests, dev configs).
    pub fn from_plain<'a>(keys: impl IntoIterator<Item = &'a str>) -> Self {
        Self {
            hashes: keys.into_iter().map(hash_key).collect(),
        }
    }

    /// The SHA-256 hex of `key`, as `FV_API_KEYS` expects it.
    pub fn hash_hex(key: &str) -> String {
        crate::hex::encode(&hash_key(key))
    }

    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    /// The owner id for `key` if it is configured.
    pub fn check(&self, key: &str) -> Option<KeyId> {
        let h = hash_key(key);
        // Compare every entry without early exit.
        let mut hit = false;
        for k in &self.hashes {
            hit |= ct_eq(k, &h);
        }
        hit.then(|| KeyId(format!("key_{}", &crate::hex::encode(&h)[..12])))
    }
}

fn hash_key(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
}

fn ct_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Keys minted at run time ([`crate::keys::KeyStore`]), checked after the
/// static ring.
#[derive(Clone)]
pub struct DynamicKeys(pub std::sync::Arc<dyn crate::keys::KeyCheck>);

impl std::fmt::Debug for DynamicKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DynamicKeys")
    }
}

/// Auth configuration and checks.
#[derive(Clone, Debug, Default)]
pub struct Auth {
    pub mode: AuthMode,
    pub keys: KeyRing,
    /// Minted keys (accepted for every API and scheme, like `keys`).
    pub dynamic: Option<DynamicKeys>,
    overrides: BTreeMap<ProtocolId, AuthPolicy>,
}

impl Auth {
    pub fn new(mode: AuthMode, keys: KeyRing) -> Self {
        Self {
            mode,
            keys,
            dynamic: None,
            overrides: BTreeMap::new(),
        }
    }
    /// Also accepts keys from a [`crate::keys::KeyStore`].
    pub fn with_key_store(mut self, store: std::sync::Arc<dyn crate::keys::KeyCheck>) -> Self {
        self.dynamic = Some(DynamicKeys(store));
        self
    }
    /// Static ring first, then minted keys.
    pub fn check_key(&self, key: &str) -> Option<KeyId> {
        self.keys.check(key).or_else(|| self.dynamic.as_ref().and_then(|d| d.0.check(key)))
    }
    /// Open everything.
    pub fn none() -> Self {
        Self::new(AuthMode::None, KeyRing::default())
    }
    /// Overrides the policy of one API.
    pub fn with_policy(mut self, p: ProtocolId, policy: AuthPolicy) -> Self {
        self.overrides.insert(p, policy);
        self
    }
    pub fn policy(&self, p: ProtocolId) -> AuthPolicy {
        self.overrides
            .get(&p)
            .cloned()
            .unwrap_or_else(|| default_policy(p))
    }

    /// Authenticates a request for API `p`. `Ok(None)` means anonymous but
    /// allowed; `Err` is `Unauthorized` for the adapter to render.
    pub fn authenticate(&self, p: ProtocolId, headers: &HeaderMap) -> Result<Option<KeyId>, ApiError> {
        match self.mode {
            AuthMode::None | AuthMode::TrustGateway => return Ok(None),
            AuthMode::Keys => {}
        }
        let presented = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_authorization);
        match self.policy(p) {
            AuthPolicy::Open => Ok(presented.and_then(|(_, k)| self.check_key(&k))),
            AuthPolicy::Require(schemes) => {
                let Some((scheme, key)) = presented else {
                    return Err(ApiError::unauthorized("missing credentials"));
                };
                if !schemes.contains(&scheme) {
                    let want: Vec<&str> = schemes.iter().map(|s| s.prefix()).collect();
                    return Err(ApiError::unauthorized(format!(
                        "unsupported authorization scheme (expected {})",
                        want.join(" or ")
                    )));
                }
                self.check_key(&key)
                    .map(Some)
                    .ok_or_else(|| ApiError::unauthorized("invalid credentials"))
            }
        }
    }
}

/// `Bearer <k>` / `Key <k>` (scheme case-insensitive) -> `(scheme, key)`.
pub fn parse_authorization(v: &str) -> Option<(Scheme, String)> {
    let v = v.trim();
    let (scheme, rest) = v.split_once(char::is_whitespace)?;
    let key = rest.trim();
    if key.is_empty() {
        return None;
    }
    let scheme = if scheme.eq_ignore_ascii_case("bearer") {
        Scheme::Bearer
    } else if scheme.eq_ignore_ascii_case("key") {
        Scheme::Key
    } else {
        return None;
    };
    Some((scheme, key.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn h(v: &str) -> HeaderMap {
        let mut m = HeaderMap::new();
        m.insert(header::AUTHORIZATION, HeaderValue::from_str(v).unwrap());
        m
    }

    fn auth() -> Auth {
        Auth::new(AuthMode::Keys, KeyRing::from_plain(["sk-good", "kid:secret"]))
    }

    #[test]
    fn hash_list_roundtrip() {
        let list = format!("{}, {}", KeyRing::hash_hex("a"), KeyRing::hash_hex("b"));
        let r = KeyRing::from_hash_list(&list).unwrap();
        assert!(r.check("a").is_some() && r.check("b").is_some() && r.check("c").is_none());
        assert!(KeyRing::from_hash_list("nothex").is_err());
        assert!(KeyRing::from_hash_list("").unwrap().is_empty());
    }

    #[test]
    fn fal_needs_key_scheme() {
        let a = auth();
        assert!(a.authenticate(ProtocolId::Fal, &h("Key sk-good")).unwrap().is_some());
        assert!(a.authenticate(ProtocolId::Fal, &h("key kid:secret")).unwrap().is_some());
        let e = a.authenticate(ProtocolId::Fal, &h("Bearer sk-good")).unwrap_err();
        assert_eq!(e.kind, fastvideo_protocol::ErrorKind::Unauthorized);
        assert!(a.authenticate(ProtocolId::Fal, &h("Key wrong")).is_err());
        assert!(a.authenticate(ProtocolId::Fal, &HeaderMap::new()).is_err());
    }

    #[test]
    fn minimax_ltx_need_bearer() {
        let a = auth();
        for p in [ProtocolId::MiniMaxV2, ProtocolId::LtxV1, ProtocolId::LtxV2] {
            assert!(a.authenticate(p, &h("Bearer sk-good")).unwrap().is_some());
            assert!(a.authenticate(p, &h("Key sk-good")).is_err());
            assert!(a.authenticate(p, &h("Bearer nope")).is_err());
            assert!(a.authenticate(p, &h("Bearer")).is_err());
        }
    }

    #[test]
    fn fastvideo_is_open_but_identifies() {
        let a = auth();
        assert_eq!(a.authenticate(ProtocolId::OpenAiVideos, &HeaderMap::new()).unwrap(), None);
        assert_eq!(a.authenticate(ProtocolId::OpenAiVideos, &h("Bearer bad")).unwrap(), None);
        let id = a.authenticate(ProtocolId::OpenAiVideos, &h("Bearer sk-good")).unwrap().unwrap();
        assert!(id.0.starts_with("key_") && !id.0.contains("sk-good"));
        // Configurable: require bearer.
        let a = a.with_policy(ProtocolId::OpenAiVideos, AuthPolicy::Require(vec![Scheme::Bearer]));
        assert!(a.authenticate(ProtocolId::OpenAiVideos, &HeaderMap::new()).is_err());
    }

    #[test]
    fn none_and_gateway_modes_open() {
        for mode in [AuthMode::None, AuthMode::TrustGateway] {
            let a = Auth::new(mode, KeyRing::default());
            assert_eq!(a.authenticate(ProtocolId::Fal, &HeaderMap::new()).unwrap(), None);
        }
    }

    #[tokio::test]
    async fn minted_keys_work_for_every_api() {
        let store = crate::keys::KeyStore::memory().await;
        let (k, rec) = store.mint("t").await.unwrap();
        let a = auth().with_key_store(store.clone());
        assert!(a.authenticate(ProtocolId::Fal, &h(&format!("Key {k}"))).unwrap().is_some());
        for p in [ProtocolId::MiniMaxV2, ProtocolId::LtxV1, ProtocolId::LtxV2, ProtocolId::Native] {
            let id = a.authenticate(p, &h(&format!("Bearer {k}"))).unwrap().unwrap();
            assert_eq!(id.0, rec.id);
        }
        assert_eq!(a.authenticate(ProtocolId::OpenAiVideos, &h(&format!("Bearer {k}"))).unwrap().unwrap().0, rec.id);
        // Static keys still work.
        assert!(a.authenticate(ProtocolId::Fal, &h("Key sk-good")).unwrap().is_some());
        store.revoke(&rec.id).await.unwrap();
        assert!(a.authenticate(ProtocolId::Fal, &h(&format!("Key {k}"))).is_err());
        assert!(a.authenticate(ProtocolId::Native, &h(&format!("Bearer {k}"))).is_err());
    }

    #[test]
    fn same_key_same_owner() {
        let a = auth();
        let x = a.authenticate(ProtocolId::MiniMaxV2, &h("Bearer sk-good")).unwrap();
        let y = a.authenticate(ProtocolId::Fal, &h("Key sk-good")).unwrap();
        assert_eq!(x, y);
    }
}
