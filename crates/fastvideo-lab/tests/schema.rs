//! Schema export (ADR-0009): the committed `schemas/gold.v1.json` equals
//! what the crate generates. On drift the test fails and names the file;
//! regenerate with `FV_LAB_WRITE=<dir> cargo test -p fastvideo-lab --test
//! schema` and copy `<dir>/schemas/gold.v1.json` over.

use std::path::PathBuf;

use fastvideo_lab::schema::{gold_schema, schema_text, GOLD_SCHEMA_PATH};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn gold_schema_has_not_drifted() {
    let generated = schema_text(&gold_schema());
    if let Some(dir) = std::env::var_os("FV_LAB_WRITE") {
        let p = PathBuf::from(dir).join(GOLD_SCHEMA_PATH);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).expect("mkdir");
        }
        std::fs::write(&p, &generated).unwrap_or_else(|e| panic!("write {}: {e}", p.display()));
        return;
    }
    let p = crate_root().join(GOLD_SCHEMA_PATH);
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
fn gold_schema_is_strict_and_names_the_sections() {
    let g: serde_json::Value = serde_json::from_str(&schema_text(&gold_schema())).unwrap();
    assert_eq!(g["additionalProperties"], serde_json::Value::Bool(false));
    let props = g["properties"].as_object().unwrap();
    for key in [
        "schema",
        "model",
        "tier",
        "workload",
        "latency",
        "quality",
        "cost",
        "last_validated",
        "source",
        "gap",
    ] {
        assert!(props.contains_key(key), "gold schema lacks `{key}`");
    }
    for (def, keys) in [
        (
            "Model",
            vec!["variant", "family", "registry_name", "rev", "profile"],
        ),
        ("TierSpec", vec!["name", "gpu", "sm", "vram_gib", "offer"]),
        (
            "Latency",
            vec![
                "measures",
                "p50_s",
                "min_s",
                "max_s",
                "denoise_s",
                "decode_s",
                "load_s",
                "peak_gib",
                "fps",
                "ttff_s",
            ],
        ),
        (
            "Quality",
            vec![
                "oracle",
                "psnr_vs_upstream_min_db",
                "lpips_vs_upstream_max",
                "byte_identical",
                "gate",
            ],
        ),
        ("Cost", vec!["usd_per_hr", "usd_per_clip"]),
        ("LastValidated", vec!["commit", "run", "date"]),
        ("Source", vec!["fields", "path", "heading", "date", "note"]),
        ("Gap", vec!["field", "why"]),
    ] {
        let d = &g["$defs"][def];
        assert_eq!(
            d["additionalProperties"],
            serde_json::Value::Bool(false),
            "{def}"
        );
        let p = d["properties"]
            .as_object()
            .unwrap_or_else(|| panic!("{def} has no properties"));
        for k in keys {
            assert!(p.contains_key(k), "{def} lacks `{k}`");
        }
    }
}
