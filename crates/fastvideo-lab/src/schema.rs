//! JSON Schema export (ADR-0009, ADR-0019): the schemas of [`Gold`],
//! [`Tiers`], [`CellFile`] and [`Stacks`] (`lab/upstream.toml`, §19), committed under `schemas/` and checked for
//! drift by `tests/schema.rs`.

use crate::gold::Gold;
use crate::oracle::Stacks;
use crate::plan::{CellFile, Tiers};

/// Where the committed gold-file schema lives, relative to the crate root.
pub const GOLD_SCHEMA_PATH: &str = "schemas/gold.v1.json";
/// `lab/tiers.toml` v1.
pub const TIERS_SCHEMA_PATH: &str = "schemas/tiers.v1.json";
/// `lab/cells/*.toml` v1.
pub const CELL_SCHEMA_PATH: &str = "schemas/cell.v1.json";
/// `lab/upstream.toml` v1 (ADR-0019 §19).
pub const UPSTREAM_SCHEMA_PATH: &str = "schemas/upstream.v1.json";

/// JSON Schema (2020-12) of `lab/gold/<variant>.toml` v1.
pub fn gold_schema() -> schemars::Schema {
    schemars::schema_for!(Gold)
}

/// JSON Schema (2020-12) of `lab/tiers.toml` v1.
pub fn tiers_schema() -> schemars::Schema {
    schemars::schema_for!(Tiers)
}

/// JSON Schema (2020-12) of `lab/cells/<cell>.toml` v1.
pub fn cell_schema() -> schemars::Schema {
    schemars::schema_for!(CellFile)
}

/// JSON Schema (2020-12) of `lab/upstream.toml` v1.
pub fn upstream_schema() -> schemars::Schema {
    schemars::schema_for!(Stacks)
}

/// A schema by name (`gold`, `tiers`, `cell`, `upstream`), for `fv-lab schema`.
pub fn by_name(name: &str) -> Option<schemars::Schema> {
    match name {
        "gold" => Some(gold_schema()),
        "tiers" => Some(tiers_schema()),
        "cell" => Some(cell_schema()),
        "upstream" => Some(upstream_schema()),
        _ => None,
    }
}

/// Every committed schema: (path relative to the crate root, schema).
pub fn all() -> [(&'static str, schemars::Schema); 4] {
    [
        (GOLD_SCHEMA_PATH, gold_schema()),
        (TIERS_SCHEMA_PATH, tiers_schema()),
        (CELL_SCHEMA_PATH, cell_schema()),
        (UPSTREAM_SCHEMA_PATH, upstream_schema()),
    ]
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
        for (s, title) in [
            (tiers_schema(), "Tiers"),
            (cell_schema(), "CellFile"),
            (upstream_schema(), "Stacks"),
        ] {
            let t = schema_text(&s);
            assert!(t.contains(&format!("\"title\": \"{title}\"")));
            assert!(t.contains("\"additionalProperties\": false"));
        }
        assert!(by_name("cell").is_some() && by_name("x").is_none());
        assert!(by_name("upstream").is_some());
    }
}
