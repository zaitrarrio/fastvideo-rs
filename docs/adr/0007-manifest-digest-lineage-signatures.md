# ADR-0007: Manifest v1, canonical digest, lineage and signatures

- Status: Accepted
- Date: 2026-10-23
- Decision-id: FVID-2026-10-23-manifest-digest-lineage-signatures
- Crate: `fastvideo-manifest`, modules `manifest`, `canonical`, `sig`,
  `lineage`; schema `crates/fastvideo-manifest/schemas/manifest.v1.json`

## Context

A published model tree needs an identity that does not depend on where it
is stored or how its JSON is laid out, a verifiable link to its parent, and
a signature that cannot be replayed. The registry's WO-001 pinned the
manifest (`.gia/manifest.json`), the digest rule
`blake3(JCS(manifest without signatures))`, Ed25519 signatures over a
domain-separated digest, and `verify_chain` over a lineage; the owner's
decision of 2026-10-23 moved them here (ADR-0005 "Context"). WO-001 left
one implementation choice open: RFC 8785 (JCS) canonical JSON, "if no
well-maintained crate exists, implement the subset you need and document
it".

## Decision

**Manifest v1** is the pinned struct, verbatim:

```rust
pub struct Manifest {
    pub schema: u32,                 // 1
    pub name: String,                // canonical ModelName without @rev
    pub parent: Option<ParentLink>,  // None for vendor refs; serialised as null
    pub model_toml_blake3: String,   // hex of the exact model.toml bytes
    pub files: Vec<FileEntry>,       // sorted by path (byte order), unique
    pub chunking: ChunkingParams,    // { algorithm, min, avg, max }
    pub created_at_unix: i64,
    pub signatures: Vec<Signature>,  // { kid, alg: "ed25519", sig: base64url no pad }; EXCLUDED from the digest
}
pub struct ParentLink { pub name: String, pub digest: String /* full blake3 hex */ }
pub struct FileEntry  { pub path: String, pub size: u64, pub blake3: String, pub chunks: Vec<ChunkRef> }
pub struct ChunkRef   { pub id: String, pub offset: u64, pub len: u32 }
```

`Manifest::validate` enforces: `schema == 1`; `name` parses without
`@rev`; a vendor ref has no parent and a fork has one; `parent.digest`,
`model_toml_blake3`, every file and chunk digest are 64 lowercase hex; a
`parent.name` with `@rev` has the rev that prefixes `parent.digest`; file
paths are relative, `/`-separated, without `.`/`..`/empty components;
files are sorted by path (byte order) and unique; chunks tile each file
contiguously from offset 0 to `size`; `0 < min <= avg <= max`;
`created_at_unix >= 0`; every integer is at most 2^53 (see below). Unknown
JSON keys are rejected; a float anywhere fails to parse.

**Canonical JSON: the RFC 8785 subset, in-crate (no JCS crate).** The crate
`canonical` module implements exactly what manifests need:

- object members sorted by the UTF-16 code units of their keys (JCS §3.2.3;
  byte order for ASCII keys);
- no whitespace; `null`/`true`/`false` literals;
- strings escaped as `JSON.stringify` does (`"`, `\`, `\b \t \n \f \r`,
  other control characters as `\u00xx` lowercase; non-ASCII literal);
- **integers only.** A float anywhere is `CanonicalError::Float { path }`.
  Integers are written as plain decimal digits, which equals the ES6
  `Number::toString` JCS prescribes for magnitudes up to 2^53; `validate`
  bounds every manifest integer by 2^53 so a valid manifest's canonical
  form is exactly what any conforming JCS implementation produces.

Why no JCS crate: the dependency list is pinned (serde, serde_json, toml,
blake3, ed25519-dalek, base64, thiserror, semver, schemars; proptest for
tests), a JCS crate would be a new dependency for about sixty lines of
code, and the general JCS number rules (shortest round-trip float
formatting) are precisely what manifests exclude. Why integers only: float
formatting is the one part of JCS where implementations disagree, and a
manifest has no quantity that is not a count, a size, an offset or a
timestamp.

**Digest.** `canonical_digest(m) = blake3(canonical_body(m))` where
`canonical_body` is a hand-written emitter over the struct in JCS member
order (`chunking`, `created_at_unix`, `files`, `model_toml_blake3`,
`name`, `parent`, `schema`; `signatures` omitted; `parent` as `null` when
absent). It exists so the digest is infallible; a test proves it equal to
the generic canonicaliser applied to the serde JSON on every fixture. The
digest is the model's identity: `Rev::from_digest` (its first 12 hex
digits) is the `@rev` of ADR-0005, and `ParentLink.digest` is the full
hex.

**Signatures.** `sign(m, kid, key)` appends
`{ kid, alg: "ed25519", sig }` with `sig` the base64url (no padding)
Ed25519 signature over the message `DOMAIN || digest`, where `DOMAIN` is
the 16 ASCII bytes `gia-manifest-v1\0`. `verify(m, keys)` requires every
signature to verify against the `KeyRing` entry of its `kid`: a manifest
with none is `Unsigned`; an unknown `kid` is `UnknownKid`; an unlisted
algorithm `UnsupportedAlg`; a wrong encoding or length `Malformed`; a
wrong signature `BadSignature`. Keys are PKCS#8 / SPKI PEM
(`signing_key_from_pem`, `verifying_key_from_pem`); the library never
reads files.

**Lineage.** `verify_chain(&[vendor, fork1, fork2, ...], keys)` checks,
in order and naming the index `at`: every manifest valid (`Invalid`) and
signed (`Signature`); `chain[0]` a vendor ref (`RootIsFork`); every later
entry a fork (`NotAFork`) whose `parent.digest` is the canonical digest
of the previous manifest and whose `parent.name` is its name
(`BrokenLink`), of the same model id (`ModelMismatch`).

WO-001 also asks for `VariantMismatch` (a fork claims a variant other than
its manifest's) and `LicenceViolation` (a fork mirrored from a parent with
`redistribution = false`). Both facts live in `model.toml`, not in the
manifest, so they cannot be decided from manifests alone; the crate adds
`verify_lineage(&[LineageEntry { manifest, model_toml, model_toml_bytes }],
keys)`, which runs `verify_chain` and then, per entry: the config bytes
hash to `model_toml_blake3` (`ConfigDigestMismatch`); `model_toml.name`
equals the manifest name, with `VariantMismatch` when only the variant
differs and `NameMismatch` otherwise; `model_toml.parent` (with `@rev`)
agrees with the manifest link (`BrokenLink`); and, when the parent's
`license.redistribution` is false, the fork neither shares a file digest
with the parent nor claims `redistribution = true` (`LicenceViolation`).

## Consequences

- The digest is invariant to key order and whitespace and changes with
  every field; signatures do not change it, so a manifest can be
  countersigned without re-identifying the model.
- The domain separator makes a manifest signature unusable as a signature
  over any other registry object (future objects get their own prefix).
- The chunk-tiling and sort rules mean `hydrate` (the registry's WO-002)
  can trust offsets without re-deriving them; `fv-boot` can verify a tree
  from the manifest alone.
- The fixtures (`fixtures/manifests`, `fixtures/keys`, `fixtures/lineage`)
  were produced by an independent Python oracle (`fixtures/gen_fixtures.py`:
  `blake3`, `cryptography`, `json`), so agreement on every digest and
  every signature is a cross-implementation check, and the registry can
  run the same files as its acceptance tests.
- The 2^53 bound caps a single file at 9 PB; not a limit in practice, and
  what keeps the canonical form portable.
- Pinned after the first publish: the struct, the member order, the
  domain separator, the integer rule. A later change is `schema = 2`.

## Tests

- `crates/fastvideo-manifest/tests/manifest.rs`: `digests_match_the_oracle`
  (every `fixtures/manifests/*.json` against its `.digest`),
  `canonical_bytes_equal_the_oracle_and_a_hand_written_string` (the
  canonical bytes written out by hand, hashed independently),
  `hand_emitter_equals_generic_jcs_on_every_fixture`,
  `key_order_and_whitespace_do_not_change_the_digest`
  (`fork-shuffled.json`), `every_field_changes_the_digest_and_signatures_do_not`
  (18 mutations), `canonical_json_rejects_floats_and_sorts_by_utf16`,
  `invalid_manifests_are_rejected`, `structural_rules`,
  `pretty_json_round_trips`.
- `crates/fastvideo-manifest/tests/sig.rs`: `sign_then_verify`,
  `ed25519_is_deterministic_so_the_oracle_signature_is_reproduced`,
  `tampered_manifest_is_a_bad_signature`, `unknown_kid`,
  `wrong_key_alg_and_malformed_signatures`,
  `the_signed_message_is_domain_separated`.
- `crates/fastvideo-manifest/tests/lineage.rs`: `ok_chain_verifies`,
  `broken_parent_digest_is_a_broken_link_at_2`,
  `variant_mismatch_between_model_toml_and_manifest`,
  `licence_violation_when_a_fork_mirrors_non_redistributable_weights`,
  `config_digest_and_name_mismatches`,
  `structure_and_signature_errors_carry_their_index`.
- `crates/fastvideo-manifest/tests/schema.rs`:
  `manifest_schema_has_not_drifted`.
- Unit tests in `src/canonical.rs`, `src/manifest.rs`, `src/sig.rs`,
  `src/path.rs`.
