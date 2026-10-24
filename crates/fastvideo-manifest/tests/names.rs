//! ADR-0005 acceptance: every valid fixture round-trips byte for byte, every
//! invalid fixture fails with the named kind, and `parse(display(n)) == n`
//! for generated names.

mod common;

use fastvideo_manifest::name::{
    Distillation, EncoderPrecision, ModelName, NameError, Rev, Variant, RESERVED_NAMESPACES,
};
use proptest::prelude::*;

#[test]
fn valid_names_round_trip() {
    let text = common::read_fixture("names/valid.txt");
    let mut n = 0;
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let name: ModelName = line.parse().unwrap_or_else(|e| panic!("{line:?}: {e}"));
        assert_eq!(name.to_string(), line, "Display of {line:?}");
        assert_eq!(line.parse::<ModelName>().unwrap(), name);
        n += 1;
    }
    assert!(n >= 15, "{n} valid names");
}

#[test]
fn invalid_names_fail_with_their_kind() {
    let text = common::read_fixture("names/invalid.txt");
    let mut seen = std::collections::BTreeSet::new();
    let mut n = 0;
    for line in text.lines() {
        if line.starts_with("# Each") || line.trim().is_empty() {
            continue;
        }
        let (name, kind) = line
            .rsplit_once('#')
            .unwrap_or_else(|| panic!("no kind on {line:?}"));
        let name = name.strip_suffix(' ').unwrap_or(name);
        let kind = kind.trim();
        let err = name
            .parse::<ModelName>()
            .err()
            .unwrap_or_else(|| panic!("{name:?} parsed; expected {kind}"));
        assert_eq!(err.kind(), kind, "{name:?} -> {err}");
        seen.insert(kind.to_string());
        n += 1;
    }
    assert!(n >= 25, "{n} invalid names");
    let every_kind: [&str; 18] = [
        "Empty",
        "EmptySegment",
        "TrailingSlash",
        "TooFewSegments",
        "TooManySegments",
        "Uppercase",
        "InvalidChar",
        "BadHyphen",
        "DoubleDot",
        "BadDot",
        "NamespaceTooShort",
        "OverLength",
        "ReservedNamespace",
        "BadVariant",
        "UnknownDistillation",
        "BadSteps",
        "UnknownEncoder",
        "BadRev",
    ];
    for kind in every_kind {
        assert!(seen.contains(kind), "no fixture for {kind}");
    }
    assert_eq!(
        seen.len(),
        every_kind.len(),
        "fixture kinds not in the enum: {seen:?}"
    );
}

#[test]
fn reserved_namespaces_are_the_pinned_five() {
    assert_eq!(
        RESERVED_NAMESPACES,
        &["gia", "admin", "api", "registry", "official"]
    );
    for ns in RESERVED_NAMESPACES {
        assert_eq!(
            format!("{ns}/m").parse::<ModelName>(),
            Err(NameError::ReservedNamespace(ns.to_string()))
        );
    }
    // Reserved words are whole namespaces, not prefixes.
    assert!("gia-labs/m".parse::<ModelName>().is_ok());
}

#[test]
fn accessors_and_rev_helpers() {
    let n: ModelName = "zay/wan2.2-t2v/dmd:4:fp8@0123456789ab".parse().unwrap();
    assert!(n.is_fork());
    assert_eq!(n.namespace().as_str(), "zay");
    assert_eq!(n.model().as_str(), "wan2.2-t2v");
    assert_eq!(
        n.variant(),
        Some(&Variant::new(Distillation::Dmd, 4, EncoderPrecision::Fp8).unwrap())
    );
    assert_eq!(n.rev().map(Rev::as_str), Some("0123456789ab"));
    assert_eq!(n.without_rev().to_string(), "zay/wan2.2-t2v/dmd:4:fp8");
    assert!(n.same_model(&n.without_rev()));
    assert_eq!(
        Variant::new(Distillation::Dmd, 0, EncoderPrecision::Fp8),
        Err(NameError::BadSteps("0".into()))
    );
    assert_eq!(
        Variant::new(Distillation::Dmd, 1001, EncoderPrecision::Fp8),
        Err(NameError::BadSteps("1001".into()))
    );
}

#[test]
fn serde_as_string() {
    let n: ModelName = "minimax/h3@0123456789ab".parse().unwrap();
    let json = serde_json::to_string(&n).unwrap();
    assert_eq!(json, "\"minimax/h3@0123456789ab\"");
    assert_eq!(serde_json::from_str::<ModelName>(&json).unwrap(), n);
    assert!(serde_json::from_str::<ModelName>("\"GIA/x\"").is_err());
}

fn seg() -> impl Strategy<Value = String> {
    "[a-z0-9]([a-z0-9-]{0,37}[a-z0-9])?".prop_filter("no -- and not reserved", |s| {
        !s.contains("--") && s.len() >= 2 && !RESERVED_NAMESPACES.contains(&s.as_str())
    })
}

fn seg_dot() -> impl Strategy<Value = String> {
    "[a-z0-9]([a-z0-9.-]{0,62}[a-z0-9])?".prop_filter("no ..", |s| !s.contains(".."))
}

fn variant() -> impl Strategy<Value = Variant> {
    (
        proptest::sample::select(Distillation::ALL.to_vec()),
        1u16..=1000,
        proptest::sample::select(EncoderPrecision::ALL.to_vec()),
    )
        .prop_map(|(d, s, e)| Variant::new(d, s, e).expect("in range"))
}

fn rev() -> impl Strategy<Value = Option<Rev>> {
    proptest::option::of("[0-9a-f]{12}".prop_map(|s| s.parse::<Rev>().expect("12 lower hex")))
}

fn model_name() -> impl Strategy<Value = ModelName> {
    (seg(), seg_dot(), proptest::option::of(variant()), rev()).prop_map(|(ns, m, v, rev)| {
        let namespace = ns.parse().expect("namespace");
        let model = m.parse().expect("model");
        match v {
            None => ModelName::Vendor {
                namespace,
                model,
                rev,
            },
            Some(variant) => ModelName::Fork {
                namespace,
                model,
                variant,
                rev,
            },
        }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn parse_display_round_trip(name in model_name()) {
        let shown = name.to_string();
        let back: ModelName = shown.parse().unwrap();
        prop_assert_eq!(back, name);
    }

    #[test]
    fn display_is_canonical_and_stable(name in model_name()) {
        let shown = name.to_string();
        prop_assert_eq!(shown.parse::<ModelName>().unwrap().to_string(), shown);
    }
}
