//! ADR-0007 acceptance (signatures): sign with the TEST key, verify OK;
//! tampered manifest -> BadSignature; unknown kid -> UnknownKid; and the
//! oracle's signatures verify here.

mod common;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use fastvideo_manifest::manifest::{canonical_digest, Manifest};
use fastvideo_manifest::sig::{
    sign, signed_message, verify, KeyRing, SigError, ALG_ED25519, DOMAIN,
};

fn vendor() -> Manifest {
    Manifest::from_json(&common::read_fixture("manifests/vendor.json")).unwrap()
}

#[test]
fn sign_then_verify() {
    let keys = common::keyring();
    let mut m = vendor();
    assert_eq!(verify(&m, &keys), Err(SigError::Unsigned));
    sign(&mut m, "test-key-1", &common::signing_key("test-key-1"));
    assert_eq!(m.signatures.len(), 1);
    assert_eq!(m.signatures[0].alg, ALG_ED25519);
    assert_eq!(verify(&m, &keys), Ok(()));
    // A second signer; both must verify.
    sign(&mut m, "test-key-2", &common::signing_key("test-key-2"));
    assert_eq!(m.signatures.len(), 2);
    assert_eq!(verify(&m, &keys), Ok(()));
    // Re-signing with the same kid replaces, not appends.
    sign(&mut m, "test-key-1", &common::signing_key("test-key-1"));
    assert_eq!(m.signatures.len(), 2);
}

#[test]
fn ed25519_is_deterministic_so_the_oracle_signature_is_reproduced() {
    // fork.json was signed by the Python oracle with test-key-1. Ed25519
    // signatures are deterministic (RFC 8032), so signing the same digest
    // with the same key here must produce the same bytes.
    let oracle = Manifest::from_json(&common::read_fixture("manifests/fork.json")).unwrap();
    assert_eq!(oracle.signatures.len(), 1);
    let mut ours = oracle.clone();
    ours.signatures.clear();
    sign(&mut ours, "test-key-1", &common::signing_key("test-key-1"));
    assert_eq!(ours.signatures, oracle.signatures);
    assert_eq!(verify(&oracle, &common::keyring()), Ok(()));
}

#[test]
fn tampered_manifest_is_a_bad_signature() {
    let keys = common::keyring();
    let mut m = Manifest::from_json(&common::read_fixture("manifests/fork.json")).unwrap();
    m.created_at_unix += 1;
    assert_eq!(
        verify(&m, &keys),
        Err(SigError::BadSignature {
            kid: "test-key-1".into()
        })
    );
    let mut m = Manifest::from_json(&common::read_fixture("manifests/fork.json")).unwrap();
    m.files[0].blake3 = m.files[1].blake3.clone();
    assert!(matches!(
        verify(&m, &keys),
        Err(SigError::BadSignature { .. })
    ));
}

#[test]
fn unknown_kid() {
    let mut m = vendor();
    sign(&mut m, "nobody", &common::signing_key("test-key-1"));
    assert_eq!(
        verify(&m, &common::keyring()),
        Err(SigError::UnknownKid {
            kid: "nobody".into()
        })
    );
    // The right key under the wrong kid is still unknown: kids are the
    // identity, not the key bytes.
    let mut ring = KeyRing::new();
    ring.insert("other", *common::keyring().get("test-key-1").unwrap());
    let mut m = vendor();
    sign(&mut m, "test-key-1", &common::signing_key("test-key-1"));
    assert_eq!(
        verify(&m, &ring),
        Err(SigError::UnknownKid {
            kid: "test-key-1".into()
        })
    );
}

#[test]
fn wrong_key_alg_and_malformed_signatures() {
    let keys = common::keyring();
    let mut m = vendor();
    // Signed with key 2 but labelled key 1.
    sign(&mut m, "test-key-1", &common::signing_key("test-key-2"));
    assert!(matches!(
        verify(&m, &keys),
        Err(SigError::BadSignature { .. })
    ));
    let mut m = vendor();
    sign(&mut m, "test-key-1", &common::signing_key("test-key-1"));
    m.signatures[0].alg = "rsa".into();
    assert!(matches!(
        verify(&m, &keys),
        Err(SigError::UnsupportedAlg { .. })
    ));
    let mut m = vendor();
    sign(&mut m, "test-key-1", &common::signing_key("test-key-1"));
    m.signatures[0].sig = "not base64url!".into();
    assert!(matches!(verify(&m, &keys), Err(SigError::Malformed { .. })));
    let mut m = vendor();
    sign(&mut m, "test-key-1", &common::signing_key("test-key-1"));
    m.signatures[0].sig = URL_SAFE_NO_PAD.encode([0u8; 63]);
    assert!(matches!(verify(&m, &keys), Err(SigError::Malformed { .. })));
    // Standard base64 with padding is not the pinned encoding.
    let mut m = vendor();
    sign(&mut m, "test-key-1", &common::signing_key("test-key-1"));
    m.signatures[0].sig.push('=');
    assert!(matches!(verify(&m, &keys), Err(SigError::Malformed { .. })));
}

#[test]
fn the_signed_message_is_domain_separated() {
    use ed25519_dalek::Signer as _;
    let m = vendor();
    let digest = canonical_digest(&m);
    let msg = signed_message(&digest);
    assert_eq!(&msg[..16], DOMAIN);
    assert_eq!(&msg[..15], b"gia-manifest-v1");
    assert_eq!(msg[15], 0);
    assert_eq!(&msg[16..], &digest);
    // A signature over the bare digest (no separator) must not verify.
    let key = common::signing_key("test-key-1");
    let bare = key.sign(&digest);
    let mut signed = m.clone();
    signed.signatures.push(fastvideo_manifest::Signature {
        kid: "test-key-1".into(),
        alg: ALG_ED25519.into(),
        sig: URL_SAFE_NO_PAD.encode(bare.to_bytes()),
    });
    assert!(matches!(
        verify(&signed, &common::keyring()),
        Err(SigError::BadSignature { .. })
    ));
}
