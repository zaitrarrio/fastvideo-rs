//! Schema export (ADR-0006, ADR-0007, ADR-0008, ADR-0011): the committed `schemas/*.json` equal
//! what the crate generates. On drift the test fails and names the file;
//! regenerate with `FV_MANIFEST_WRITE=<dir> cargo test -p fastvideo-manifest
//! --test schema` and copy `<dir>/schemas/*.json` over.

mod common;

use fastvideo_manifest::schema::{
    deploy_schema, family_schema, manifest_schema, model_toml_schema, schema_text,
    DEPLOY_SCHEMA_PATH, FAMILY_SCHEMA_PATH, MANIFEST_SCHEMA_PATH, MODEL_TOML_SCHEMA_PATH,
};

fn check(rel: &str, generated: String) {
    common::write_generated(rel, &generated);
    if common::write_dir().is_some() {
        return;
    }
    let p = common::crate_root().join(rel);
    let committed = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("{}: {e}; regenerate (see tests/schema.rs)", p.display()));
    assert_eq!(
        committed,
        generated,
        "{} drifted from the crate; regenerate (see tests/schema.rs)",
        p.display()
    );
}

#[test]
fn model_toml_schema_has_not_drifted() {
    check(MODEL_TOML_SCHEMA_PATH, schema_text(&model_toml_schema()));
}

#[test]
fn manifest_schema_has_not_drifted() {
    check(MANIFEST_SCHEMA_PATH, schema_text(&manifest_schema()));
}

#[test]
fn family_schema_has_not_drifted() {
    check(FAMILY_SCHEMA_PATH, schema_text(&family_schema()));
}

#[test]
fn deploy_schema_has_not_drifted() {
    check(DEPLOY_SCHEMA_PATH, schema_text(&deploy_schema()));
}

#[test]
fn deploy_schema_is_strict_and_keeps_autoscale_pools_opaque() {
    let d: serde_json::Value = serde_json::from_str(&schema_text(&deploy_schema())).unwrap();
    assert_eq!(d["additionalProperties"], serde_json::Value::Bool(false));
    let props = d["properties"].as_object().unwrap();
    for key in [
        "schema",
        "dispatch",
        "images",
        "presets",
        "family_presets",
        "gcp",
        "autoscale",
    ] {
        assert!(props.contains_key(key), "deploy schema lacks `{key}`");
    }
    for def in ["Preset", "FamilyPreset", "Image", "PoolModel", "GcpRow"] {
        assert_eq!(
            d["$defs"][def]["additionalProperties"],
            serde_json::Value::Bool(false),
            "{def} is not strict"
        );
    }
    let pools = &d["$defs"]["AutoscaleSection"]["properties"]["pools"]["items"];
    assert_eq!(pools["type"], "object");
    assert_eq!(pools["additionalProperties"], serde_json::Value::Bool(true));
}

#[test]
fn family_schema_is_strict_and_keeps_engine_opaque() {
    let f: serde_json::Value = serde_json::from_str(&schema_text(&family_schema())).unwrap();
    assert_eq!(f["additionalProperties"], serde_json::Value::Bool(false));
    let props = f["properties"].as_object().unwrap();
    for key in ["schema", "family", "variants"] {
        assert!(props.contains_key(key), "family schema lacks `{key}`");
    }
    let variant = &f["$defs"]["VariantManifest"];
    assert_eq!(
        variant["additionalProperties"],
        serde_json::Value::Bool(false)
    );
    let vprops = variant["properties"].as_object().unwrap();
    for key in [
        "id",
        "registry_name",
        "tier",
        "served_names",
        "public_names",
        "resident_default",
        "recipe_name",
        "trees",
        "gpu",
        "caps",
        "serving",
        "engine",
    ] {
        assert!(vprops.contains_key(key), "variant schema lacks `{key}`");
    }
    // The engine table is opaque: any object.
    assert_eq!(vprops["engine"]["type"], "object");
    assert_eq!(
        vprops["engine"]["additionalProperties"],
        serde_json::Value::Bool(true)
    );
}

#[test]
fn schemas_are_strict_and_name_the_pinned_fields() {
    let m: serde_json::Value = serde_json::from_str(&schema_text(&model_toml_schema())).unwrap();
    assert_eq!(m["additionalProperties"], serde_json::Value::Bool(false));
    let props = m["properties"].as_object().unwrap();
    for key in [
        "schema",
        "name",
        "parent",
        "license",
        "runtime",
        "defaults",
        "components",
        "change",
        "variant",
        "caps",
    ] {
        assert!(props.contains_key(key), "model.toml schema lacks `{key}`");
    }
    let n: serde_json::Value = serde_json::from_str(&schema_text(&manifest_schema())).unwrap();
    assert_eq!(n["additionalProperties"], serde_json::Value::Bool(false));
    let props = n["properties"].as_object().unwrap();
    for key in [
        "schema",
        "name",
        "parent",
        "model_toml_blake3",
        "files",
        "chunking",
        "created_at_unix",
        "signatures",
    ] {
        assert!(props.contains_key(key), "manifest schema lacks `{key}`");
    }
}
