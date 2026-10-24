//! JSON Schema export (ADR-0009): the schema of [`Gold`], committed under
//! `schemas/gold.v1.json` and checked for drift by `tests/schema.rs`.

use crate::gold::Gold;

/// Where the committed gold-file schema lives, relative to the crate root.
pub const GOLD_SCHEMA_PATH: &str = "schemas/gold.v1.json";

/// JSON Schema (2020-12) of `lab/gold/<variant>.toml` v1.
pub fn gold_schema() -> schemars::Schema {
    schemars::schema_for!(Gold)
}

/// A schema as the committed file holds it: pretty JSON, one trailing newline.
pub fn schema_text(schema: &schemars::Schema) -> String {
    let mut s = serde_json::to_string_pretty(schema).unwrap_or_default();
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_names_its_type_and_denies_unknown_fields() {
        let g = schema_text(&gold_schema());
        assert!(g.contains("\"title\": \"Gold\""));
        assert!(g.contains("\"additionalProperties\": false"));
        for key in [
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
            assert!(g.contains(&format!("\"{key}\"")), "schema lacks `{key}`");
        }
    }
}
