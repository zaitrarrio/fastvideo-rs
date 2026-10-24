//! Lineage (ADR-0007): a chain of manifests from a vendor release to a fork
//! of a fork, each link a parent digest, each manifest signed.
//!
//! [`verify_chain`] is the pinned entry point over manifests alone:
//! structure, signatures, and the parent links. [`verify_lineage`] adds each
//! clone's `model.toml`, which is where the variant claim and the licence
//! live, and so is the only place [`LineageError::VariantMismatch`] and
//! [`LineageError::LicenceViolation`] can be found.

use crate::manifest::{blake3_hex, canonical_digest, Manifest, ManifestError};
use crate::model_toml::ModelToml;
use crate::name::ModelName;
use crate::sig::{verify, KeyRing, SigError};

/// Why a chain does not verify. `at` is the index into the chain
/// (0 = the vendor root).
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum LineageError {
    #[error("empty chain")]
    Empty,
    #[error("chain[{at}] is not a valid manifest: {source}")]
    Invalid { at: usize, source: ManifestError },
    #[error("chain[{at}] signature: {source}")]
    Signature { at: usize, source: SigError },
    #[error("chain[0] must be a vendor ref, not a fork")]
    RootIsFork,
    #[error("chain[{at}] must be a fork (only the root is a vendor ref)")]
    NotAFork { at: usize },
    #[error("chain[{at}] does not link to chain[{}]: parent digest or name differs", at - 1)]
    BrokenLink { at: usize },
    #[error("chain[{at}] forks a different model than its parent")]
    ModelMismatch { at: usize },
    #[error("chain[{at}]: model.toml claims a different variant than the manifest name")]
    VariantMismatch { at: usize },
    #[error("chain[{at}]: model.toml name differs from the manifest name")]
    NameMismatch { at: usize },
    #[error("chain[{at}]: model.toml bytes do not hash to model_toml_blake3")]
    ConfigDigestMismatch { at: usize },
    #[error("chain[{at}]: the parent's licence forbids redistribution; the fork mirrors its weights or claims redistribution")]
    LicenceViolation { at: usize },
}

/// A clone: its manifest, its `model.toml`, and (when the caller has them)
/// the exact `model.toml` bytes, to check `model_toml_blake3`.
#[derive(Debug, Clone, Copy)]
pub struct LineageEntry<'a> {
    pub manifest: &'a Manifest,
    pub model_toml: &'a ModelToml,
    pub model_toml_bytes: Option<&'a [u8]>,
}

/// Verify `chain[0]` (a vendor ref) through its forks: every manifest valid
/// and signed by a key in `keys`, every `parent` the digest and name of the
/// manifest before it, every non-root entry a fork of the same model.
pub fn verify_chain(chain: &[Manifest], keys: &KeyRing) -> Result<(), LineageError> {
    verify_manifests(chain.iter(), keys)
}

/// [`verify_chain`] plus the `model.toml` checks: name and variant agree
/// with the manifest, the parent pointer agrees with the link, the config
/// bytes hash to `model_toml_blake3`, and no fork redistributes weights
/// its parent's licence withholds.
pub fn verify_lineage(chain: &[LineageEntry<'_>], keys: &KeyRing) -> Result<(), LineageError> {
    verify_manifests(chain.iter().map(|e| e.manifest), keys)?;
    for (at, entry) in chain.iter().enumerate() {
        let manifest = entry.manifest;
        let toml = entry.model_toml;
        if let Some(bytes) = entry.model_toml_bytes {
            if blake3_hex(bytes) != manifest.model_toml_blake3 {
                return Err(LineageError::ConfigDigestMismatch { at });
            }
        }
        let mname = manifest
            .model_name()
            .map_err(|source| LineageError::Invalid { at, source })?;
        if toml.name != mname {
            let same_model =
                toml.name.namespace() == mname.namespace() && toml.name.model() == mname.model();
            if same_model && toml.name.variant() != mname.variant() {
                return Err(LineageError::VariantMismatch { at });
            }
            return Err(LineageError::NameMismatch { at });
        }
        match (&toml.parent, &manifest.parent) {
            (None, None) => {}
            (Some(tp), Some(mp)) => {
                let link_name: ModelName =
                    mp.name.parse().map_err(|source| LineageError::Invalid {
                        at,
                        source: ManifestError::ParentName(source),
                    })?;
                let rev_ok = tp.rev().is_some_and(|r| r.matches_digest_hex(&mp.digest));
                if !rev_ok || !tp.same_model(&link_name) {
                    return Err(LineageError::BrokenLink { at });
                }
            }
            _ => return Err(LineageError::BrokenLink { at }),
        }
        if at > 0 {
            let parent = &chain[at - 1];
            if !parent.model_toml.license.redistribution {
                let mirrors = manifest
                    .files
                    .iter()
                    .any(|f| parent.manifest.files.iter().any(|pf| pf.blake3 == f.blake3));
                if toml.license.redistribution || mirrors {
                    return Err(LineageError::LicenceViolation { at });
                }
            }
        }
    }
    Ok(())
}

fn verify_manifests<'a>(
    chain: impl Iterator<Item = &'a Manifest>,
    keys: &KeyRing,
) -> Result<(), LineageError> {
    let mut prev: Option<(ModelName, String)> = None;
    let mut len = 0usize;
    for (at, m) in chain.enumerate() {
        len += 1;
        m.validate()
            .map_err(|source| LineageError::Invalid { at, source })?;
        verify(m, keys).map_err(|source| LineageError::Signature { at, source })?;
        let name = m
            .model_name()
            .map_err(|source| LineageError::Invalid { at, source })?;
        match &prev {
            None => {
                if name.is_fork() {
                    return Err(LineageError::RootIsFork);
                }
            }
            Some((parent_name, parent_digest)) => {
                if !name.is_fork() {
                    return Err(LineageError::NotAFork { at });
                }
                let Some(link) = &m.parent else {
                    return Err(LineageError::Invalid {
                        at,
                        source: ManifestError::ForkWithoutParent,
                    });
                };
                let link_name: ModelName =
                    link.name.parse().map_err(|source| LineageError::Invalid {
                        at,
                        source: ManifestError::ParentName(source),
                    })?;
                if link.digest != *parent_digest || !link_name.same_model(parent_name) {
                    return Err(LineageError::BrokenLink { at });
                }
                if name.model() != parent_name.model() {
                    return Err(LineageError::ModelMismatch { at });
                }
            }
        }
        let digest = blake3::Hash::from_bytes(canonical_digest(m))
            .to_hex()
            .to_string();
        prev = Some((name, digest));
    }
    if len == 0 {
        return Err(LineageError::Empty);
    }
    Ok(())
}
