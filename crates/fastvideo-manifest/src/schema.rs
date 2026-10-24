//! JSON Schema export (ADR-0006, ADR-0007, ADR-0008, ADR-0011): the schemas
//! of [`ModelToml`], [`Manifest`], [`FamilyManifest`] and [`DeployManifest`],
//! committed under `schemas/` and checked for drift by `tests/schema.rs`. The registry's
//! `task schema:export` compares against the first two.

use crate::deploy::DeployManifest;
use crate::family::FamilyManifest;
use crate::manifest::Manifest;
use crate::model_toml::ModelToml;

/// Where the committed `model.toml` schema lives, relative to the crate root.
pub const MODEL_TOML_SCHEMA_PATH: &str = "schemas/model-toml.v1.json";
/// Where the committed manifest schema lives, relative to the crate root.
pub const MANIFEST_SCHEMA_PATH: &str = "schemas/manifest.v1.json";
/// Where the committed family-file schema (layer 2) lives, relative to the crate root.
pub const FAMILY_SCHEMA_PATH: &str = "schemas/family.v1.json";

/// Where the committed deployment-file schema lives, relative to the crate root.
pub const DEPLOY_SCHEMA_PATH: &str = "schemas/deploy.v1.json";

/// JSON Schema (2020-12) of `model.toml` v1.
pub fn model_toml_schema() -> schemars::Schema {
    schemars::schema_for!(ModelToml)
}

/// JSON Schema (2020-12) of manifest v1.
pub fn manifest_schema() -> schemars::Schema {
    schemars::schema_for!(Manifest)
}

/// JSON Schema (2020-12) of `families/<family>.toml` v1 (layer 2).
pub fn family_schema() -> schemars::Schema {
    schemars::schema_for!(FamilyManifest)
}

/// JSON Schema (2020-12) of `deploy/deploy.toml` v1 (ADR-0011).
pub fn deploy_schema() -> schemars::Schema {
    schemars::schema_for!(DeployManifest)
}

/// A schema as the committed file holds it: pretty JSON, sorted keys, one
/// trailing newline.
pub fn schema_text(schema: &schemars::Schema) -> String {
    let mut s = serde_json::to_string_pretty(schema).unwrap_or_default();
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemas_name_their_types_and_deny_unknown_fields() {
        let m = schema_text(&model_toml_schema());
        assert!(m.contains("\"title\": \"ModelToml\""));
        assert!(m.contains("\"additionalProperties\": false"));
        let n = schema_text(&manifest_schema());
        assert!(n.contains("\"title\": \"Manifest\""));
        assert!(n.contains("\"model_toml_blake3\""));
        let f = schema_text(&family_schema());
        assert!(f.contains("\"title\": \"FamilyManifest\""));
        assert!(f.contains("\"engine\""));
        let d = schema_text(&deploy_schema());
        assert!(d.contains("\"title\": \"DeployManifest\""));
        assert!(d.contains("\"family_presets\""));
    }
}
