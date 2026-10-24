//! ADR-0007 acceptance (digest): every fixture's canonical digest equals its
//! `.digest` (computed by the Python oracle); key order and whitespace do
//! not change it; every field does; signatures do not; the canonical bytes
//! equal a hand-canonicalised string.

mod common;

use fastvideo_manifest::canonical::{to_canonical_bytes, to_canonical_string, CanonicalError};
use fastvideo_manifest::manifest::{
    blake3_hex, canonical_body, canonical_digest, ChunkRef, FileEntry, Manifest, ManifestError,
    ParentLink, Signature,
};

fn load(rel: &str) -> Manifest {
    Manifest::from_json(&common::read_fixture(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

fn digest_fixture(rel: &str) -> String {
    common::read_fixture(rel).trim().to_string()
}

#[test]
fn digests_match_the_oracle() {
    let files = common::fixture_files("manifests", "json");
    let mut n = 0;
    for p in files {
        if p.file_name()
            .is_some_and(|f| f.to_string_lossy().ends_with(".canonical.json"))
        {
            continue;
        }
        let text = common::read(&p);
        let m = Manifest::from_json(&text).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let expected = common::read(&p.with_extension("digest")).trim().to_string();
        assert_eq!(m.digest_hex(), expected, "{}", p.display());
        assert_eq!(m.rev().as_str(), &expected[..12]);
        n += 1;
    }
    assert!(n >= 3, "{n} manifest fixtures");
}

#[test]
fn canonical_bytes_equal_the_oracle_and_a_hand_written_string() {
    let m = load("manifests/vendor.json");
    let oracle = common::read_fixture("manifests/vendor.canonical.json");
    assert_eq!(
        String::from_utf8(canonical_body(&m)).unwrap(),
        oracle.trim_end()
    );
    // The same thing again, with the object written out by hand so the
    // digest is checked by a path that shares nothing with `canonical_body`.
    let f = &m.files;
    let chunk = |c: &ChunkRef| {
        format!(
            r#"{{"id":"{}","len":{},"offset":{}}}"#,
            c.id, c.len, c.offset
        )
    };
    let file = |f: &FileEntry| {
        format!(
            r#"{{"blake3":"{}","chunks":[{}],"path":"{}","size":{}}}"#,
            f.blake3,
            f.chunks.iter().map(chunk).collect::<Vec<_>>().join(","),
            f.path,
            f.size
        )
    };
    let hand = format!(
        r#"{{"chunking":{{"algorithm":"fastcdc-2020","avg":64,"max":256,"min":16}},"created_at_unix":1761264000,"files":[{}],"model_toml_blake3":"{}","name":"wan-ai/wan2.2-t2v","parent":null,"schema":1}}"#,
        f.iter().map(file).collect::<Vec<_>>().join(","),
        m.model_toml_blake3
    );
    assert_eq!(hand, oracle.trim_end());
    assert_eq!(
        blake3_hex(hand.as_bytes()),
        digest_fixture("manifests/vendor.digest")
    );
    assert_eq!(blake3_hex(hand.as_bytes()), m.digest_hex());
}

#[test]
fn hand_emitter_equals_generic_jcs_on_every_fixture() {
    for p in common::fixture_files("manifests", "json") {
        if p.to_string_lossy().ends_with(".canonical.json") {
            continue;
        }
        let m = Manifest::from_json(&common::read(&p)).unwrap();
        let mut v = serde_json::to_value(&m).unwrap();
        v.as_object_mut().unwrap().remove("signatures");
        assert_eq!(
            canonical_body(&m),
            to_canonical_bytes(&v).unwrap(),
            "{}",
            p.display()
        );
    }
}

#[test]
fn key_order_and_whitespace_do_not_change_the_digest() {
    // fork-shuffled.json is fork.json with every object's keys reversed and
    // four-space indentation.
    let a = load("manifests/fork.json");
    let b = load("manifests/fork-shuffled.json");
    assert_eq!(a, b);
    assert_eq!(canonical_digest(&a), canonical_digest(&b));
    let text = common::read_fixture("manifests/fork.json");
    assert_ne!(text, common::read_fixture("manifests/fork-shuffled.json"));
    // And a compact re-serialisation.
    let compact = serde_json::to_string(&a).unwrap();
    assert_eq!(
        Manifest::from_json(&compact).unwrap().digest_hex(),
        digest_fixture("manifests/fork.digest")
    );
}

#[test]
fn every_field_changes_the_digest_and_signatures_do_not() {
    let base = load("manifests/fork.json");
    let d0 = canonical_digest(&base);
    type Mutation = Box<dyn Fn(&mut Manifest)>;
    let mutations: Vec<(&str, Mutation)> = vec![
        ("schema", Box::new(|m| m.schema = 2)),
        (
            "name",
            Box::new(|m| m.name = "zay/wan2.2-t2v/dmd:8:fp8".into()),
        ),
        (
            "parent.name",
            Box::new(|m| m.parent.as_mut().unwrap().name = "wan-ai/wan2.1-t2v".into()),
        ),
        (
            "parent.digest",
            Box::new(|m| {
                let p = m.parent.as_mut().unwrap();
                p.digest = format!("0{}", &p.digest[1..]);
            }),
        ),
        ("parent -> null", Box::new(|m| m.parent = None)),
        (
            "model_toml_blake3",
            Box::new(|m| m.model_toml_blake3 = blake3_hex(b"other")),
        ),
        ("files[0].path", Box::new(|m| m.files[0].path = "a".into())),
        ("files[0].size", Box::new(|m| m.files[0].size += 1)),
        (
            "files[0].blake3",
            Box::new(|m| m.files[0].blake3 = blake3_hex(b"x")),
        ),
        (
            "files[0].chunks[0].id",
            Box::new(|m| m.files[0].chunks[0].id = blake3_hex(b"y")),
        ),
        (
            "files[0].chunks[0].offset",
            Box::new(|m| m.files[0].chunks[0].offset += 1),
        ),
        (
            "files[0].chunks[0].len",
            Box::new(|m| m.files[0].chunks[0].len += 1),
        ),
        (
            "files.pop",
            Box::new(|m| {
                m.files.pop();
            }),
        ),
        (
            "chunking.algorithm",
            Box::new(|m| m.chunking.algorithm = "rabin".into()),
        ),
        ("chunking.min", Box::new(|m| m.chunking.min += 1)),
        ("chunking.avg", Box::new(|m| m.chunking.avg += 1)),
        ("chunking.max", Box::new(|m| m.chunking.max += 1)),
        ("created_at_unix", Box::new(|m| m.created_at_unix += 1)),
    ];
    for (what, mutate) in mutations {
        let mut m = base.clone();
        mutate(&mut m);
        assert_ne!(canonical_digest(&m), d0, "{what} did not change the digest");
    }
    let mut m = base.clone();
    m.signatures.clear();
    assert_eq!(
        canonical_digest(&m),
        d0,
        "signatures are excluded from the digest"
    );
    m.signatures.push(Signature {
        kid: "k".into(),
        alg: "ed25519".into(),
        sig: "AA".into(),
    });
    assert_eq!(canonical_digest(&m), d0);
}

#[test]
fn canonical_json_rejects_floats_and_sorts_by_utf16() {
    let v: serde_json::Value = serde_json::from_str(r#"{"b": 1, "a": {"y": 2.5}}"#).unwrap();
    assert_eq!(
        to_canonical_bytes(&v).unwrap_err(),
        CanonicalError::Float {
            path: "$.a.y".to_string()
        }
    );
    let v: serde_json::Value =
        serde_json::from_str("{\"b\":1,\"a\":[true,null,\"x\\ty\"]}").unwrap();
    assert_eq!(
        to_canonical_string(&v).unwrap(),
        "{\"a\":[true,null,\"x\\ty\"],\"b\":1}"
    );
}

#[test]
fn invalid_manifests_are_rejected() {
    type Check = fn(&ManifestError) -> bool;
    let cases: &[(&str, Check)] = &[
        ("float-size.json", |e| matches!(e, ManifestError::Json(_))),
        (
            "unknown-key.json",
            |e| matches!(e, ManifestError::Json(m) if m.contains("unknown field")),
        ),
        ("unsorted-files.json", |e| {
            matches!(e, ManifestError::FilesNotSorted { index: 1 })
        }),
        ("chunks-gap.json", |e| {
            matches!(e, ManifestError::ChunksNotContiguous { index: 1 })
        }),
        ("dotdot-path.json", |e| {
            matches!(e, ManifestError::Path { index: 1, .. })
        }),
        ("name-with-rev.json", |e| {
            matches!(e, ManifestError::NameHasRev)
        }),
    ];
    let listed = common::fixture_files("manifests/invalid", "json");
    assert_eq!(
        listed.len(),
        cases.len(),
        "every invalid fixture has a case"
    );
    for (file, check) in cases {
        let text = common::read_fixture(&format!("manifests/invalid/{file}"));
        let err = Manifest::from_json(&text)
            .err()
            .unwrap_or_else(|| panic!("{file} parsed"));
        assert!(check(&err), "{file}: {err}");
    }
}

#[test]
fn structural_rules() {
    let mut m = load("manifests/vendor.json");
    assert_eq!(m.validate(), Ok(()));
    m.parent = Some(ParentLink {
        name: "wan-ai/wan2.2-t2v".into(),
        digest: "0".repeat(64),
    });
    assert_eq!(m.validate(), Err(ManifestError::VendorWithParent));

    let mut m = load("manifests/fork.json");
    m.parent = None;
    assert_eq!(m.validate(), Err(ManifestError::ForkWithoutParent));
    let mut m = load("manifests/fork.json");
    m.parent.as_mut().unwrap().name = format!("wan-ai/wan2.2-t2v@{}", "f".repeat(12));
    assert_eq!(m.validate(), Err(ManifestError::ParentRevMismatch));
    let mut m = load("manifests/fork.json");
    let rev = m.parent.as_ref().unwrap().digest[..12].to_string();
    m.parent.as_mut().unwrap().name = format!("wan-ai/wan2.2-t2v@{rev}");
    assert_eq!(m.validate(), Ok(()));
    let mut m = load("manifests/fork.json");
    m.model_toml_blake3 = "ABC".into();
    assert_eq!(
        m.validate(),
        Err(ManifestError::BadHex {
            field: "model_toml_blake3".into()
        })
    );
    let mut m = load("manifests/fork.json");
    m.files[1].size = (1u64 << 53) + 1;
    assert!(matches!(
        m.validate(),
        Err(ManifestError::IntegerTooLarge { .. })
    ));
    let mut m = load("manifests/fork.json");
    m.chunking.min = 0;
    assert!(matches!(m.validate(), Err(ManifestError::Chunking(_))));
    let mut m = load("manifests/fork.json");
    let dup = m.files[0].clone();
    m.files.insert(1, dup);
    assert_eq!(m.validate(), Err(ManifestError::DuplicatePath { index: 1 }));
}

#[test]
fn pretty_json_round_trips() {
    let m = load("manifests/fork.json");
    let text = m.to_json_pretty();
    assert_eq!(Manifest::from_json(&text).unwrap(), m);
}
