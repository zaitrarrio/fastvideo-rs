//! Ed25519 signatures over a manifest's canonical digest (ADR-0007).
//!
//! The signed message is the ASCII domain separator [`DOMAIN`]
//! (`gia-manifest-v1\0`, 16 bytes) followed by the 32-byte digest, so a
//! signature over a manifest can never be replayed as a signature over
//! anything else. Signatures live in `Manifest::signatures`, which the digest
//! excludes; a manifest may carry several (one per `kid`).

use std::collections::BTreeMap;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use ed25519_dalek::{Signer as _, SigningKey, VerifyingKey};

use crate::manifest::{canonical_digest, Manifest, Signature};

/// The domain separator prefixed to the digest before signing.
pub const DOMAIN: &[u8; 16] = b"gia-manifest-v1\0";
/// The only signature algorithm of manifest v1.
pub const ALG_ED25519: &str = "ed25519";

/// Why a manifest's signatures do not verify.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum SigError {
    #[error("manifest carries no signatures")]
    Unsigned,
    #[error("unknown signing key id {kid:?}")]
    UnknownKid { kid: String },
    #[error("unsupported signature algorithm {alg:?} for kid {kid:?}")]
    UnsupportedAlg { kid: String, alg: String },
    #[error("malformed signature for kid {kid:?}: not base64url without padding, or not 64 bytes")]
    Malformed { kid: String },
    #[error("bad signature for kid {kid:?}")]
    BadSignature { kid: String },
}

/// Public keys by key id (`kid`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeyRing {
    keys: BTreeMap<String, VerifyingKey>,
}

impl KeyRing {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace a key; returns the key it replaced.
    pub fn insert(&mut self, kid: impl Into<String>, key: VerifyingKey) -> Option<VerifyingKey> {
        self.keys.insert(kid.into(), key)
    }

    pub fn get(&self, kid: &str) -> Option<&VerifyingKey> {
        self.keys.get(kid)
    }

    pub fn kids(&self) -> impl Iterator<Item = &str> {
        self.keys.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

impl<K: Into<String>> FromIterator<(K, VerifyingKey)> for KeyRing {
    fn from_iter<I: IntoIterator<Item = (K, VerifyingKey)>>(iter: I) -> Self {
        let mut ring = KeyRing::new();
        for (kid, key) in iter {
            ring.insert(kid, key);
        }
        ring
    }
}

/// The bytes that are signed: `DOMAIN || digest`.
pub fn signed_message(digest: &[u8; 32]) -> [u8; 48] {
    let mut msg = [0u8; 48];
    msg[..16].copy_from_slice(DOMAIN);
    msg[16..].copy_from_slice(digest);
    msg
}

/// Sign `m`'s canonical digest with `key` and append the signature as `kid`.
/// An existing signature with the same `kid` is replaced.
pub fn sign(m: &mut Manifest, kid: &str, key: &SigningKey) {
    let digest = canonical_digest(m);
    let sig = key.sign(&signed_message(&digest));
    m.signatures.retain(|s| s.kid != kid);
    m.signatures.push(Signature {
        kid: kid.to_string(),
        alg: ALG_ED25519.to_string(),
        sig: URL_SAFE_NO_PAD.encode(sig.to_bytes()),
    });
}

/// Verify every signature on `m` against `keys`. All must verify; a manifest
/// without signatures is [`SigError::Unsigned`].
pub fn verify(m: &Manifest, keys: &KeyRing) -> Result<(), SigError> {
    if m.signatures.is_empty() {
        return Err(SigError::Unsigned);
    }
    let digest = canonical_digest(m);
    let msg = signed_message(&digest);
    for s in &m.signatures {
        let key = keys
            .get(&s.kid)
            .ok_or_else(|| SigError::UnknownKid { kid: s.kid.clone() })?;
        if s.alg != ALG_ED25519 {
            return Err(SigError::UnsupportedAlg {
                kid: s.kid.clone(),
                alg: s.alg.clone(),
            });
        }
        let malformed = || SigError::Malformed { kid: s.kid.clone() };
        let raw = URL_SAFE_NO_PAD.decode(&s.sig).map_err(|_| malformed())?;
        let sig = ed25519_dalek::Signature::from_slice(&raw).map_err(|_| malformed())?;
        key.verify_strict(&msg, &sig)
            .map_err(|_| SigError::BadSignature { kid: s.kid.clone() })?;
    }
    Ok(())
}

/// Parse a PKCS#8 PEM private key (`-----BEGIN PRIVATE KEY-----`).
pub fn signing_key_from_pem(pem: &str) -> Result<SigningKey, ed25519_dalek::pkcs8::Error> {
    use ed25519_dalek::pkcs8::DecodePrivateKey as _;
    SigningKey::from_pkcs8_pem(pem)
}

/// Parse an SPKI PEM public key (`-----BEGIN PUBLIC KEY-----`).
pub fn verifying_key_from_pem(
    pem: &str,
) -> Result<VerifyingKey, ed25519_dalek::pkcs8::spki::Error> {
    use ed25519_dalek::pkcs8::DecodePublicKey as _;
    VerifyingKey::from_public_key_pem(pem)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_is_sixteen_bytes_with_a_nul() {
        assert_eq!(DOMAIN.len(), 16);
        assert_eq!(DOMAIN[15], 0);
        assert_eq!(&DOMAIN[..15], b"gia-manifest-v1");
    }
}
