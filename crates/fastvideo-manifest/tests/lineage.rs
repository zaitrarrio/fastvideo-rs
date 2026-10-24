//! ADR-0007 acceptance (lineage): `verify_chain(&[vendor, fork1, fork2])`
//! OK; a broken parent digest -> `BrokenLink { at: 2 }`; a model.toml that
//! claims another variant -> `VariantMismatch`; a fork mirroring weights
//! whose parent forbids redistribution -> `LicenceViolation`.

mod common;

use fastvideo_manifest::lineage::{verify_chain, verify_lineage, LineageEntry, LineageError};
use fastvideo_manifest::manifest::Manifest;
use fastvideo_manifest::model_toml::ModelToml;
use fastvideo_manifest::sig::{KeyRing, SigError};

struct Loaded {
    manifest: Manifest,
    toml: ModelToml,
    toml_bytes: Vec<u8>,
}

fn chain(case: &str) -> Vec<Loaded> {
    ["vendor", "fork1", "fork2"]
        .into_iter()
        .map(|stem| {
            let manifest = Manifest::from_json(&common::read_fixture(&format!(
                "lineage/{case}/{stem}.json"
            )))
            .unwrap_or_else(|e| panic!("{case}/{stem}: {e}"));
            let expected = common::read_fixture(&format!("lineage/{case}/{stem}.digest"));
            assert_eq!(
                manifest.digest_hex(),
                expected.trim(),
                "{case}/{stem} digest"
            );
            let toml_bytes =
                common::read_fixture_bytes(&format!("lineage/{case}/{stem}.model.toml"));
            let toml = ModelToml::parse(std::str::from_utf8(&toml_bytes).unwrap())
                .unwrap_or_else(|e| panic!("{case}/{stem}.model.toml: {e}"));
            Loaded {
                manifest,
                toml,
                toml_bytes,
            }
        })
        .collect()
}

fn manifests(c: &[Loaded]) -> Vec<Manifest> {
    c.iter().map(|x| x.manifest.clone()).collect()
}

fn entries(c: &[Loaded]) -> Vec<LineageEntry<'_>> {
    c.iter()
        .map(|x| LineageEntry {
            manifest: &x.manifest,
            model_toml: &x.toml,
            model_toml_bytes: Some(&x.toml_bytes),
        })
        .collect()
}

#[test]
fn ok_chain_verifies() {
    let keys = common::keyring();
    let c = chain("ok");
    assert_eq!(verify_chain(&manifests(&c), &keys), Ok(()));
    assert_eq!(verify_lineage(&entries(&c), &keys), Ok(()));
    // Prefixes of a valid chain are valid chains.
    assert_eq!(verify_chain(&manifests(&c)[..1], &keys), Ok(()));
    assert_eq!(verify_chain(&manifests(&c)[..2], &keys), Ok(()));
    // The parent revs in the model.tomls are the parents' revs.
    assert_eq!(
        c[1].toml.parent.as_ref().unwrap().rev().unwrap(),
        &c[0].manifest.rev()
    );
    assert_eq!(
        c[2].toml.parent.as_ref().unwrap().rev().unwrap(),
        &c[1].manifest.rev()
    );
}

#[test]
fn broken_parent_digest_is_a_broken_link_at_2() {
    let keys = common::keyring();
    let c = chain("broken-link");
    assert_eq!(
        verify_chain(&manifests(&c), &keys),
        Err(LineageError::BrokenLink { at: 2 })
    );
    assert_eq!(
        verify_lineage(&entries(&c), &keys),
        Err(LineageError::BrokenLink { at: 2 })
    );
    // Programmatically: swap the parents' order.
    let ok = chain("ok");
    let mut ms = manifests(&ok);
    ms.swap(1, 2);
    assert!(matches!(
        verify_chain(&ms, &keys),
        Err(LineageError::BrokenLink { at: 1 })
    ));
}

#[test]
fn variant_mismatch_between_model_toml_and_manifest() {
    let keys = common::keyring();
    let c = chain("variant-mismatch");
    // The manifests alone are a consistent, signed chain ...
    assert_eq!(verify_chain(&manifests(&c), &keys), Ok(()));
    // ... but fork1's model.toml claims dmd:8:fp8 where the manifest says dmd:4:fp8.
    assert_eq!(c[1].toml.name.to_string(), "zay/wan2.2-t2v/dmd:8:fp8");
    assert_eq!(c[1].manifest.name, "zay/wan2.2-t2v/dmd:4:fp8");
    assert_eq!(
        verify_lineage(&entries(&c), &keys),
        Err(LineageError::VariantMismatch { at: 1 })
    );
}

#[test]
fn licence_violation_when_a_fork_mirrors_non_redistributable_weights() {
    let keys = common::keyring();
    let c = chain("licence-violation");
    assert!(!c[0].toml.license.redistribution);
    assert_eq!(verify_chain(&manifests(&c), &keys), Ok(()));
    assert_eq!(
        verify_lineage(&entries(&c), &keys),
        Err(LineageError::LicenceViolation { at: 1 })
    );
}

#[test]
fn config_digest_and_name_mismatches() {
    let keys = common::keyring();
    let c = chain("ok");
    let mut e = entries(&c);
    let other: &[u8] = b"schema = 1\n";
    e[0].model_toml_bytes = Some(other);
    assert_eq!(
        verify_lineage(&e, &keys),
        Err(LineageError::ConfigDigestMismatch { at: 0 })
    );
    // Without the bytes the digest is not checked.
    e[0].model_toml_bytes = None;
    assert_eq!(verify_lineage(&e, &keys), Ok(()));
    // A model.toml for another model entirely.
    let swapped = [
        LineageEntry {
            manifest: &c[0].manifest,
            model_toml: &c[0].toml,
            model_toml_bytes: None,
        },
        LineageEntry {
            manifest: &c[1].manifest,
            model_toml: &c[2].toml,
            model_toml_bytes: None,
        },
    ];
    assert_eq!(
        verify_lineage(&swapped, &keys),
        Err(LineageError::NameMismatch { at: 1 })
    );
}

#[test]
fn structure_and_signature_errors_carry_their_index() {
    let keys = common::keyring();
    assert_eq!(verify_chain(&[], &keys), Err(LineageError::Empty));
    let c = chain("ok");
    let mut ms = manifests(&c);
    // A fork at the root.
    assert_eq!(verify_chain(&ms[1..], &keys), Err(LineageError::RootIsFork));
    // A vendor ref after the root.
    let two = vec![ms[0].clone(), ms[0].clone()];
    assert_eq!(
        verify_chain(&two, &keys),
        Err(LineageError::NotAFork { at: 1 })
    );
    // Tampering breaks the signature before anything else is looked at.
    ms[2].created_at_unix += 1;
    assert_eq!(
        verify_chain(&ms, &keys),
        Err(LineageError::Signature {
            at: 2,
            source: SigError::BadSignature {
                kid: "test-key-2".into()
            }
        })
    );
    // An unknown key.
    let full = common::keyring();
    let ring: KeyRing = [("test-key-1", *full.get("test-key-1").unwrap())]
        .into_iter()
        .collect();
    assert_eq!(
        verify_chain(&manifests(&c), &ring),
        Err(LineageError::Signature {
            at: 2,
            source: SigError::UnknownKid {
                kid: "test-key-2".into()
            }
        })
    );
    // A fork of a different model.
    let mut ms = manifests(&c);
    ms[1].name = "zay/wan2.1-t2v/dmd:4:fp8".into();
    assert!(matches!(
        verify_chain(&ms, &keys),
        Err(LineageError::Signature { at: 1, .. })
    ));
}
