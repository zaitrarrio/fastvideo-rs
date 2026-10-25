//! The committed `schemas/*.v1.json` equal what the crate generates
//! (docs/adr/0018-fv-deploy.md §3, as `fastvideo-manifest`'s `family.v1.json`).
//! On drift the test fails, naming the file, and writes the generated text to
//! `$CARGO_TARGET_TMPDIR/fv-deploy-schemas/` (copy it over, or print it with
//! `fv-deploy schema <name>`).

use fastvideo_deploy::provider::schema;

#[test]
fn schemas_have_not_drifted() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(schema::SCHEMA_DIR);
    let out = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("fv-deploy-schemas");
    let _ = std::fs::create_dir_all(&out);
    let mut drift = Vec::new();
    for (file, s) in schema::all() {
        let generated = schema::text(&s);
        let _ = std::fs::write(out.join(file), &generated);
        let committed = std::fs::read_to_string(root.join(file)).unwrap_or_default();
        if committed != generated {
            drift.push(file);
        }
    }
    assert!(
        drift.is_empty(),
        "{drift:?} drifted from the crate; the generated files are in {}",
        out.display()
    );
}

#[test]
fn launch_schema_is_strict_and_names_its_parts() {
    let t = schema::text(&schema::by_name("launch").unwrap_or_else(|| panic!("no launch schema")));
    assert!(t.contains("\"title\": \"Launch\""));
    assert!(t.contains("\"additionalProperties\": false"));
    for k in [
        "\"runpod-pod\"",
        "\"runpod-serverless\"",
        "\"vast\"",
        "\"gcp\"",
        "\"ready_cycles\"",
        "\"min_balance_usd\"",
        "\"volume\"",
    ] {
        assert!(t.contains(k), "{k}");
    }
    assert!(schema::by_name("instance").is_some());
    assert!(schema::by_name("nope").is_none());
}
