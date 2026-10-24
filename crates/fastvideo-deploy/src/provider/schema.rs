//! JSON Schema export of the fv-deploy types (docs/adr/0018-fv-deploy.md §3),
//! committed under `crates/fastvideo-deploy/schemas/` and checked for drift by
//! `tests/schema.rs`, as `fastvideo-manifest` does for `family.v1.json`.
//! `fv-deploy schema <name>` prints one; fv-control's TypeScript types are
//! generated from them in plan C3c.

use super::types::{Account, Instance, Launch, Offer};

/// `(file name, schema)` of every exported type, in a fixed order.
pub fn all() -> Vec<(&'static str, schemars::Schema)> {
    vec![
        ("launch.v1.json", schemars::schema_for!(Launch)),
        ("instance.v1.json", schemars::schema_for!(Instance)),
        ("offer.v1.json", schemars::schema_for!(Offer)),
        ("account.v1.json", schemars::schema_for!(Account)),
    ]
}

/// The directory of the committed schemas, relative to the crate root.
pub const SCHEMA_DIR: &str = "schemas";

/// One schema by its short name (`launch`, `instance`, `offer`, `account`).
pub fn by_name(name: &str) -> Option<schemars::Schema> {
    let file = format!("{name}.v1.json");
    all().into_iter().find(|(f, _)| *f == file).map(|(_, s)| s)
}

/// A schema as the committed file holds it: pretty JSON, one trailing newline.
pub fn text(schema: &schemars::Schema) -> String {
    let mut s = serde_json::to_string_pretty(schema).unwrap_or_default();
    s.push('\n');
    s
}
