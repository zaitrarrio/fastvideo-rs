//! Manifest v1 (ADR-0007): `.gia/manifest.json`, the signed record of a
//! model tree: its files, their BLAKE3 digests and content-defined chunks,
//! the chunking parameters, the parent link, and the signatures.
//!
//! [`canonical_digest`] is `blake3(JCS(manifest without signatures))`: the
//! identity of a published model, whose first 12 hex digits are its `@rev`.
//! The canonical bytes are produced by [`canonical_body`], a hand-written
//! emitter over the struct (so it cannot fail); `tests/manifest.rs` proves
//! it equal to the generic [`crate::canonical::to_canonical_bytes`] of the
//! serde JSON with `signatures` removed.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::canonical::{write_string, MAX_SAFE_INTEGER};
use crate::name::{ModelName, NameError, Rev};
use crate::path::{check_relative_path, PathError};

/// The only manifest schema this crate reads and writes.
pub const MANIFEST_SCHEMA: u32 = 1;

/// A model tree's manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Always 1.
    pub schema: u32,
    /// Canonical [`ModelName`] without `@rev`.
    pub name: String,
    /// `None` for a vendor ref. Serialised as `null` when absent.
    #[serde(default)]
    pub parent: Option<ParentLink>,
    /// Lowercase hex BLAKE3 of the exact `model.toml` bytes.
    pub model_toml_blake3: String,
    /// Sorted by `path` (byte order), unique.
    pub files: Vec<FileEntry>,
    /// Recorded so readers never assume.
    pub chunking: ChunkingParams,
    /// Seconds since the Unix epoch.
    pub created_at_unix: i64,
    /// Excluded from the digest. May be empty before signing.
    #[serde(default)]
    pub signatures: Vec<Signature>,
}

/// The link from a fork to the manifest it was forked from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ParentLink {
    /// The parent's canonical name; an `@rev`, if present, must be the
    /// prefix of `digest`.
    pub name: String,
    /// Full lowercase hex BLAKE3 canonical digest of the parent manifest.
    pub digest: String,
}

/// One file of the tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FileEntry {
    /// Relative, `/` separators, no `..`.
    pub path: String,
    pub size: u64,
    /// Lowercase hex BLAKE3 of the whole file.
    pub blake3: String,
    /// Contiguous from offset 0 to `size`.
    pub chunks: Vec<ChunkRef>,
}

/// One content-defined chunk of a file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChunkRef {
    /// Lowercase hex BLAKE3 of the chunk bytes.
    pub id: String,
    pub offset: u64,
    pub len: u32,
}

/// How the files were chunked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChunkingParams {
    /// e.g. `fastcdc-2020`.
    pub algorithm: String,
    pub min: u32,
    pub avg: u32,
    pub max: u32,
}

/// One signature over the canonical digest (see [`crate::sig`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Signature {
    pub kid: String,
    /// `ed25519`.
    pub alg: String,
    /// base64url without padding.
    pub sig: String,
}

/// Why a manifest is not valid.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    #[error("not JSON, or not a manifest: {0}")]
    Json(String),
    #[error("unsupported manifest schema {0}; this crate reads schema {MANIFEST_SCHEMA}")]
    UnsupportedSchema(u32),
    #[error("name: {0}")]
    Name(NameError),
    #[error("name carries an @rev; a manifest's name never does")]
    NameHasRev,
    #[error("parent.name: {0}")]
    ParentName(NameError),
    #[error("parent.name carries an @rev that is not the prefix of parent.digest")]
    ParentRevMismatch,
    #[error("a vendor ref has no parent")]
    VendorWithParent,
    #[error("a fork needs a parent")]
    ForkWithoutParent,
    #[error("`{field}` is not 64 lowercase hex digits")]
    BadHex { field: String },
    #[error("files[{index}].path {path:?}: {source}")]
    Path {
        index: usize,
        path: String,
        source: PathError,
    },
    #[error("files[{index}] is out of order: files are sorted by path (byte order)")]
    FilesNotSorted { index: usize },
    #[error("files[{index}] repeats the path of an earlier entry")]
    DuplicatePath { index: usize },
    #[error("files[{index}].chunks do not tile the file contiguously from offset 0")]
    ChunksNotContiguous { index: usize },
    #[error("files[{index}].chunks cover {covered} bytes, size says {size}")]
    ChunkSizeMismatch {
        index: usize,
        covered: u64,
        size: u64,
    },
    #[error("chunking: {0}")]
    Chunking(String),
    #[error("`{field}` exceeds 2^53: canonical JSON integers must be exact")]
    IntegerTooLarge { field: String },
    #[error("created_at_unix is negative")]
    NegativeTimestamp,
    #[error("signatures[{index}]: {reason}")]
    Signature { index: usize, reason: String },
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl Manifest {
    /// Parse and validate a manifest from JSON text.
    pub fn from_json(text: &str) -> Result<Manifest, ManifestError> {
        let m: Manifest =
            serde_json::from_str(text).map_err(|e| ManifestError::Json(e.to_string()))?;
        m.validate()?;
        Ok(m)
    }

    /// Pretty JSON (not canonical; the digest does not depend on layout).
    pub fn to_json_pretty(&self) -> String {
        // A Manifest is plain data: this serialisation cannot fail.
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// The parsed name.
    pub fn model_name(&self) -> Result<ModelName, ManifestError> {
        self.name.parse().map_err(ManifestError::Name)
    }

    /// The canonical digest as lowercase hex.
    pub fn digest_hex(&self) -> String {
        blake3::Hash::from_bytes(canonical_digest(self))
            .to_hex()
            .to_string()
    }

    /// The `@rev` of this manifest.
    pub fn rev(&self) -> Rev {
        Rev::from_digest(&canonical_digest(self))
    }

    /// Structural validation (everything except signatures and lineage).
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.schema != MANIFEST_SCHEMA {
            return Err(ManifestError::UnsupportedSchema(self.schema));
        }
        let name = self.model_name()?;
        if name.rev().is_some() {
            return Err(ManifestError::NameHasRev);
        }
        match (&self.parent, name.is_fork()) {
            (Some(_), false) => return Err(ManifestError::VendorWithParent),
            (None, true) => return Err(ManifestError::ForkWithoutParent),
            _ => {}
        }
        if let Some(parent) = &self.parent {
            let pname: ModelName = parent.name.parse().map_err(ManifestError::ParentName)?;
            if !is_hex64(&parent.digest) {
                return Err(ManifestError::BadHex {
                    field: "parent.digest".to_string(),
                });
            }
            if let Some(rev) = pname.rev() {
                if !rev.matches_digest_hex(&parent.digest) {
                    return Err(ManifestError::ParentRevMismatch);
                }
            }
        }
        if !is_hex64(&self.model_toml_blake3) {
            return Err(ManifestError::BadHex {
                field: "model_toml_blake3".to_string(),
            });
        }
        if self.created_at_unix < 0 {
            return Err(ManifestError::NegativeTimestamp);
        }
        if self.created_at_unix.unsigned_abs() > MAX_SAFE_INTEGER {
            return Err(ManifestError::IntegerTooLarge {
                field: "created_at_unix".to_string(),
            });
        }
        let c = &self.chunking;
        if c.algorithm.is_empty() {
            return Err(ManifestError::Chunking("algorithm is empty".to_string()));
        }
        if c.min == 0 || !(c.min <= c.avg && c.avg <= c.max) {
            return Err(ManifestError::Chunking(format!(
                "need 0 < min <= avg <= max, got min={} avg={} max={}",
                c.min, c.avg, c.max
            )));
        }
        for (index, f) in self.files.iter().enumerate() {
            check_relative_path(&f.path).map_err(|source| ManifestError::Path {
                index,
                path: f.path.clone(),
                source,
            })?;
            if index > 0 {
                let prev = &self.files[index - 1].path;
                match f.path.as_bytes().cmp(prev.as_bytes()) {
                    std::cmp::Ordering::Less => {
                        return Err(ManifestError::FilesNotSorted { index })
                    }
                    std::cmp::Ordering::Equal => {
                        return Err(ManifestError::DuplicatePath { index })
                    }
                    std::cmp::Ordering::Greater => {}
                }
            }
            if !is_hex64(&f.blake3) {
                return Err(ManifestError::BadHex {
                    field: format!("files[{index}].blake3"),
                });
            }
            if f.size > MAX_SAFE_INTEGER {
                return Err(ManifestError::IntegerTooLarge {
                    field: format!("files[{index}].size"),
                });
            }
            let mut covered: u64 = 0;
            for (ci, chunk) in f.chunks.iter().enumerate() {
                if !is_hex64(&chunk.id) {
                    return Err(ManifestError::BadHex {
                        field: format!("files[{index}].chunks[{ci}].id"),
                    });
                }
                if chunk.offset != covered {
                    return Err(ManifestError::ChunksNotContiguous { index });
                }
                covered = covered
                    .checked_add(u64::from(chunk.len))
                    .ok_or(ManifestError::ChunksNotContiguous { index })?;
            }
            if covered != f.size {
                return Err(ManifestError::ChunkSizeMismatch {
                    index,
                    covered,
                    size: f.size,
                });
            }
        }
        for (index, s) in self.signatures.iter().enumerate() {
            if s.kid.is_empty() {
                return Err(ManifestError::Signature {
                    index,
                    reason: "empty kid".to_string(),
                });
            }
            if s.alg.is_empty() || s.sig.is_empty() {
                return Err(ManifestError::Signature {
                    index,
                    reason: "empty alg or sig".to_string(),
                });
            }
        }
        Ok(())
    }
}

/// `blake3(JCS(m without signatures))`.
pub fn canonical_digest(m: &Manifest) -> [u8; 32] {
    *blake3::hash(&canonical_body(m)).as_bytes()
}

/// The canonical JSON of `m` without `signatures`: the bytes that are
/// hashed. Members are emitted in JCS order (sorted keys), as the generic
/// canonicaliser would; this emitter exists so the digest cannot fail.
pub fn canonical_body(m: &Manifest) -> Vec<u8> {
    let mut out = Vec::with_capacity(256 + m.files.len() * 160);
    out.push(b'{');
    // chunking
    key(&mut out, "chunking");
    out.push(b'{');
    key(&mut out, "algorithm");
    write_string(&m.chunking.algorithm, &mut out);
    out.push(b',');
    key(&mut out, "avg");
    num(&mut out, u64::from(m.chunking.avg));
    out.push(b',');
    key(&mut out, "max");
    num(&mut out, u64::from(m.chunking.max));
    out.push(b',');
    key(&mut out, "min");
    num(&mut out, u64::from(m.chunking.min));
    out.push(b'}');
    out.push(b',');
    // created_at_unix
    key(&mut out, "created_at_unix");
    out.extend_from_slice(m.created_at_unix.to_string().as_bytes());
    out.push(b',');
    // files
    key(&mut out, "files");
    out.push(b'[');
    for (i, f) in m.files.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.push(b'{');
        key(&mut out, "blake3");
        write_string(&f.blake3, &mut out);
        out.push(b',');
        key(&mut out, "chunks");
        out.push(b'[');
        for (j, c) in f.chunks.iter().enumerate() {
            if j > 0 {
                out.push(b',');
            }
            out.push(b'{');
            key(&mut out, "id");
            write_string(&c.id, &mut out);
            out.push(b',');
            key(&mut out, "len");
            num(&mut out, u64::from(c.len));
            out.push(b',');
            key(&mut out, "offset");
            num(&mut out, c.offset);
            out.push(b'}');
        }
        out.push(b']');
        out.push(b',');
        key(&mut out, "path");
        write_string(&f.path, &mut out);
        out.push(b',');
        key(&mut out, "size");
        num(&mut out, f.size);
        out.push(b'}');
    }
    out.push(b']');
    out.push(b',');
    // model_toml_blake3
    key(&mut out, "model_toml_blake3");
    write_string(&m.model_toml_blake3, &mut out);
    out.push(b',');
    // name
    key(&mut out, "name");
    write_string(&m.name, &mut out);
    out.push(b',');
    // parent
    key(&mut out, "parent");
    match &m.parent {
        None => out.extend_from_slice(b"null"),
        Some(p) => {
            out.push(b'{');
            key(&mut out, "digest");
            write_string(&p.digest, &mut out);
            out.push(b',');
            key(&mut out, "name");
            write_string(&p.name, &mut out);
            out.push(b'}');
        }
    }
    out.push(b',');
    // schema
    key(&mut out, "schema");
    num(&mut out, u64::from(m.schema));
    out.push(b'}');
    out
}

fn key(out: &mut Vec<u8>, k: &str) {
    write_string(k, out);
    out.push(b':');
}

fn num(out: &mut Vec<u8>, n: u64) {
    out.extend_from_slice(n.to_string().as_bytes());
}

/// `blake3` of arbitrary bytes as lowercase hex (for `model_toml_blake3`,
/// file and chunk digests).
pub fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Manifest {
        Manifest {
            schema: 1,
            name: "wan-ai/wan2.2-t2v".to_string(),
            parent: None,
            model_toml_blake3: blake3_hex(b"model"),
            files: vec![FileEntry {
                path: "weights/a.safetensors".to_string(),
                size: 5,
                blake3: blake3_hex(b"hello"),
                chunks: vec![ChunkRef {
                    id: blake3_hex(b"hello"),
                    offset: 0,
                    len: 5,
                }],
            }],
            chunking: ChunkingParams {
                algorithm: "fastcdc-2020".to_string(),
                min: 1,
                avg: 2,
                max: 3,
            },
            created_at_unix: 1_700_000_000,
            signatures: vec![],
        }
    }

    #[test]
    fn hand_emitter_equals_generic_canonicaliser() {
        let m = sample();
        let mut v = serde_json::to_value(&m).unwrap();
        v.as_object_mut().unwrap().remove("signatures");
        let generic = crate::canonical::to_canonical_bytes(&v).unwrap();
        assert_eq!(canonical_body(&m), generic);
    }

    #[test]
    fn validates_sample_and_catches_errors() {
        let m = sample();
        assert_eq!(m.validate(), Ok(()));
        let mut bad = m.clone();
        bad.files[0].chunks[0].len = 4;
        assert!(matches!(
            bad.validate(),
            Err(ManifestError::ChunkSizeMismatch { .. })
        ));
        let mut bad = m.clone();
        bad.files.push(FileEntry {
            path: "a".to_string(),
            size: 0,
            blake3: blake3_hex(b""),
            chunks: vec![],
        });
        assert_eq!(
            bad.validate(),
            Err(ManifestError::FilesNotSorted { index: 1 })
        );
        let mut bad = m;
        bad.name = "zay/wan2.2-t2v/dmd:4:fp8".to_string();
        assert_eq!(bad.validate(), Err(ManifestError::ForkWithoutParent));
    }
}
