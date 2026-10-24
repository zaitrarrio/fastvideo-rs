//! The registry's public formats (`docs/adr/0005`, `0006`, `0007`): the one
//! part of this repository that is a pinned contract, because every
//! published model is written in them.
//!
//! - [`name`]: [`ModelName`], `namespace/model[@rev]` and
//!   `namespace/model/variant[@rev]`.
//! - [`model_toml`]: [`ModelToml`], the clone's config, v1 plus the
//!   `[variant]`, `[runtime].recipe/profile` and `[caps]` widenings.
//! - [`manifest`]: [`Manifest`], the signed file list, and
//!   [`canonical_digest`] = `blake3(JCS(manifest without signatures))`.
//! - [`canonical`]: the RFC 8785 subset (sorted keys, integers only).
//! - [`sig`]: [`KeyRing`], [`sign`], [`verify`] with the domain separator
//!   `gia-manifest-v1\0`.
//! - [`lineage`]: [`verify_chain`] over `[vendor, fork1, fork2, ...]` and
//!   [`verify_lineage`] with each clone's `model.toml`.
//! - [`schema`]: JSON Schema export, checked for drift against `schemas/`.
//! - [`family`]: layer 2 (ADR-0008), [`FamilyManifest`] and
//!   [`VariantManifest`]: `families/<family>.toml`, the runtime's superset
//!   the engine catalog and the deployment tooling are generated from. Not
//!   a public format: it is this repository's, versioned by `schema`.
//!
//! Pure: no I/O (callers read files), no CUDA, no other fastvideo crate.
//! The optional `fs` feature adds the one directory reader
//! (`family::load_dir`) the runtime consumers share; the registry never
//! enables it.
//! The registry (`zaitrarrio/model-registry-rs`) depends on this crate and
//! never re-implements a format; a change here is a change to every
//! published model, so widenings happen only before the first publish, each
//! with an ADR and fixtures (`docs/gaps/2026-10-22-architecture-review.md`
//! §7.2, §7.10).

pub mod canonical;
pub mod family;
pub mod lineage;
pub mod manifest;
pub mod model_toml;
pub mod name;
pub mod path;
pub mod schema;
pub mod sig;

pub use canonical::{to_canonical_bytes, to_canonical_string, CanonicalError};
pub use family::{
    FamilyError, FamilyHeader, FamilyId, FamilyManifest, FamilySet, GpuNeeds, Tier,
    VariantManifest, FAMILY_SCHEMA, TAE_VAR, WEIGHTS_VAR,
};
pub use lineage::{verify_chain, verify_lineage, LineageEntry, LineageError};
pub use manifest::{
    blake3_hex, canonical_body, canonical_digest, ChunkRef, ChunkingParams, FileEntry, Manifest,
    ManifestError, ParentLink, Signature,
};
pub use model_toml::{
    CanvasCaps, Caps, Change, Component, Defaults, FpsCaps, FrameCaps, KnobCaps, License,
    ModelToml, ModelTomlError, RefCaps, Runtime, VariantTable,
};
pub use name::{
    Distillation, EncoderPrecision, ModelId, ModelName, NameError, Namespace, Rev, Variant,
};
pub use path::{check_relative_path, PathError};
pub use sig::{
    sign, signing_key_from_pem, verify, verifying_key_from_pem, KeyRing, SigError, ALG_ED25519,
    DOMAIN,
};
