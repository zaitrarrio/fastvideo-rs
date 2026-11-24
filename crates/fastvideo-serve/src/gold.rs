//! The gold lab records (`lab/gold/<variant>.toml`, ADR-0009; filled by
//! `fv-lab run gold`, ADR-0023) served to the console: every served model's
//! gold standard (tier, workload, latency, quality gate, cost, when it was
//! last validated, the sources every number is copied from, the declared
//! gaps). The files are embedded at build time, as the family manifests are
//! (`fastvideo_engine_service::cuda::families::EMBEDDED_FAMILIES`): the
//! images carry no `lab/gold`.
//!
//! | Route | Body |
//! |---|---|
//! | `GET /fv/v1/gold` | `{object: "fv.gold", records: [{variant, family, tier, gpu, measures, p50_s, gate, usd_per_clip, last_validated}]}` |
//! | `GET /fv/v1/gold/{variant}` | The record as JSON (the TOML tables as objects) with `path` and `toml` (the file's text); 404 for a variant with no record |
//!
//! Auth as the native API (mounted with it, like `/fv/v1/params`).

use std::sync::OnceLock;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use fastvideo_protocol::{ApiError, ErrorCtx, ProtocolId};
use fastvideo_serve_kit::handlers::error_reply;
use fastvideo_serve_kit::ServeCtx;
use serde_json::{json, Value};

use crate::native::Native;

/// The gold files this binary was built with (`lab/gold/*.toml`, one per
/// variant; `embedded_list_is_the_directory` holds it to the directory).
pub const EMBEDDED_GOLD: [(&str, &str); 35] = [
    (
        "lab/gold/fasth3-4step-vsa-480p-taeh3.toml",
        include_str!("../../../lab/gold/fasth3-4step-vsa-480p-taeh3.toml"),
    ),
    (
        "lab/gold/fasth3-4step-vsa-nvfp4-taeh3.toml",
        include_str!("../../../lab/gold/fasth3-4step-vsa-nvfp4-taeh3.toml"),
    ),
    (
        "lab/gold/fasth3-4step-vsa-nvfp4.toml",
        include_str!("../../../lab/gold/fasth3-4step-vsa-nvfp4.toml"),
    ),
    (
        "lab/gold/fasth3-4step-vsa-taeh3.toml",
        include_str!("../../../lab/gold/fasth3-4step-vsa-taeh3.toml"),
    ),
    (
        "lab/gold/fasth3-4step-vsa.toml",
        include_str!("../../../lab/gold/fasth3-4step-vsa.toml"),
    ),
    (
        "lab/gold/fasth3-8step-dense.toml",
        include_str!("../../../lab/gold/fasth3-8step-dense.toml"),
    ),
    (
        "lab/gold/fasth3-trim-8step.toml",
        include_str!("../../../lab/gold/fasth3-trim-8step.toml"),
    ),
    (
        "lab/gold/fasth3-trim-nvfp4-fast.toml",
        include_str!("../../../lab/gold/fasth3-trim-nvfp4-fast.toml"),
    ),
    (
        "lab/gold/fasth3-trim-nvfp4.toml",
        include_str!("../../../lab/gold/fasth3-trim-nvfp4.toml"),
    ),
    (
        "lab/gold/fasth3-v2-nvfp4-taeh3.toml",
        include_str!("../../../lab/gold/fasth3-v2-nvfp4-taeh3.toml"),
    ),
    (
        "lab/gold/fasth3-v2-nvfp4.toml",
        include_str!("../../../lab/gold/fasth3-v2-nvfp4.toml"),
    ),
    (
        "lab/gold/fastwan21-1.3b-taehv.toml",
        include_str!("../../../lab/gold/fastwan21-1.3b-taehv.toml"),
    ),
    (
        "lab/gold/fastwan21-1.3b.toml",
        include_str!("../../../lab/gold/fastwan21-1.3b.toml"),
    ),
    (
        "lab/gold/fastwan22-ti2v-5b-taehv.toml",
        include_str!("../../../lab/gold/fastwan22-ti2v-5b-taehv.toml"),
    ),
    (
        "lab/gold/fastwan22-ti2v-5b.toml",
        include_str!("../../../lab/gold/fastwan22-ti2v-5b.toml"),
    ),
    (
        "lab/gold/h3-larry-turbo.toml",
        include_str!("../../../lab/gold/h3-larry-turbo.toml"),
    ),
    (
        "lab/gold/h3-lightx2v-turbo-v01.toml",
        include_str!("../../../lab/gold/h3-lightx2v-turbo-v01.toml"),
    ),
    (
        "lab/gold/h3-lightx2v-turbo.toml",
        include_str!("../../../lab/gold/h3-lightx2v-turbo.toml"),
    ),
    (
        "lab/gold/h3-plug-4step.toml",
        include_str!("../../../lab/gold/h3-plug-4step.toml"),
    ),
    (
        "lab/gold/h3-plug-cfg.toml",
        include_str!("../../../lab/gold/h3-plug-cfg.toml"),
    ),
    (
        "lab/gold/h3-ref2v-max.toml",
        include_str!("../../../lab/gold/h3-ref2v-max.toml"),
    ),
    (
        "lab/gold/h3-ref2v-turbo.toml",
        include_str!("../../../lab/gold/h3-ref2v-turbo.toml"),
    ),
    (
        "lab/gold/history-forcing-1.3b.toml",
        include_str!("../../../lab/gold/history-forcing-1.3b.toml"),
    ),
    (
        "lab/gold/longlive-1.3b.toml",
        include_str!("../../../lab/gold/longlive-1.3b.toml"),
    ),
    (
        "lab/gold/ltx25-a2v-guided.toml",
        include_str!("../../../lab/gold/ltx25-a2v-guided.toml"),
    ),
    (
        "lab/gold/ltx25-distill-dense.toml",
        include_str!("../../../lab/gold/ltx25-distill-dense.toml"),
    ),
    (
        "lab/gold/ltx25-distill-sol-nvfp4-taehv.toml",
        include_str!("../../../lab/gold/ltx25-distill-sol-nvfp4-taehv.toml"),
    ),
    (
        "lab/gold/ltx25-distill-sol.toml",
        include_str!("../../../lab/gold/ltx25-distill-sol.toml"),
    ),
    (
        "lab/gold/ltx25-ref2v.toml",
        include_str!("../../../lab/gold/ltx25-ref2v.toml"),
    ),
    (
        "lab/gold/sfwan21-1.3b.toml",
        include_str!("../../../lab/gold/sfwan21-1.3b.toml"),
    ),
    (
        "lab/gold/sol-h3.toml",
        include_str!("../../../lab/gold/sol-h3.toml"),
    ),
    (
        "lab/gold/vdn-h3-8step.toml",
        include_str!("../../../lab/gold/vdn-h3-8step.toml"),
    ),
    (
        "lab/gold/vdn-h3.toml",
        include_str!("../../../lab/gold/vdn-h3.toml"),
    ),
    (
        "lab/gold/wan14b-plug-4step.toml",
        include_str!("../../../lab/gold/wan14b-plug-4step.toml"),
    ),
    (
        "lab/gold/wan22-ti2v-5b.toml",
        include_str!("../../../lab/gold/wan22-ti2v-5b.toml"),
    ),
];

/// One embedded record, parsed.
pub struct Record {
    pub path: &'static str,
    pub text: &'static str,
    pub variant: String,
    pub value: Value,
}

fn parse(path: &'static str, text: &'static str) -> Result<Record, String> {
    let t: toml::Value = toml::from_str(text).map_err(|e| format!("{path}: {e}"))?;
    let value = serde_json::to_value(&t).map_err(|e| format!("{path}: {e}"))?;
    let variant = value["model"]["variant"]
        .as_str()
        .ok_or_else(|| format!("{path}: no model.variant"))?
        .to_owned();
    Ok(Record {
        path,
        text,
        variant,
        value,
    })
}

/// Every embedded record, parsed once (a file that does not parse is left
/// out and logged: the console must not lose the others).
pub fn records() -> &'static [Record] {
    static ALL: OnceLock<Vec<Record>> = OnceLock::new();
    ALL.get_or_init(|| {
        EMBEDDED_GOLD
            .iter()
            .filter_map(|(p, t)| match parse(p, t) {
                Ok(r) => Some(r),
                Err(e) => {
                    tracing::warn!(error = %e, "gold record skipped");
                    None
                }
            })
            .collect()
    })
}

/// The record of a variant (its `model.variant`), else the one whose file is
/// named after it.
pub fn record(variant: &str) -> Option<&'static Record> {
    records().iter().find(|r| r.variant == variant).or_else(|| {
        records().iter().find(|r| {
            r.path
                .strip_prefix("lab/gold/")
                .and_then(|p| p.strip_suffix(".toml"))
                == Some(variant)
        })
    })
}

/// The index row of a record: the headline numbers the home page lists.
fn index_row(r: &Record) -> Value {
    let v = &r.value;
    json!({
        "variant": r.variant,
        "family": v["model"]["family"],
        "tier": v["tier"]["name"],
        "gpu": v["tier"]["gpu"],
        "measures": v["latency"]["measures"],
        "p50_s": v["latency"]["p50_s"],
        "gate": v["quality"]["gate"],
        "usd_per_clip": v["cost"]["usd_per_clip"],
        "last_validated": v["last_validated"],
        "sources": v["source"].as_array().map_or(0, Vec::len),
        "gaps": v["gap"].as_array().map_or(0, Vec::len),
        "path": r.path,
    })
}

/// `GET /fv/v1/gold`.
pub fn index() -> Value {
    json!({"object": "fv.gold", "records": records().iter().map(index_row).collect::<Vec<_>>()})
}

/// `GET /fv/v1/gold/{variant}`.
pub fn detail(variant: &str) -> Option<Value> {
    let r = record(variant)?;
    let mut v = r.value.clone();
    v["object"] = "fv.gold.record".into();
    v["path"] = r.path.into();
    v["toml"] = r.text.into();
    Some(v)
}

fn reply_err(e: ApiError) -> Response {
    let r = error_reply(&Native, &e, &ErrorCtx::default());
    let body = r.json_body().cloned().unwrap_or(Value::Null);
    (
        axum::http::StatusCode::from_u16(r.status)
            .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
        Json(body),
    )
        .into_response()
}

/// The gold routes (auth as the native API).
pub fn routes() -> Router<ServeCtx> {
    Router::new()
        .route(
            "/fv/v1/gold",
            get(|State(ctx): State<ServeCtx>, headers: HeaderMap| async move {
                if let Err(e) = ctx.auth().authenticate(ProtocolId::Native, &headers) {
                    return reply_err(e);
                }
                Json(index()).into_response()
            }),
        )
        .route(
            "/fv/v1/gold/{variant}",
            get(
                |State(ctx): State<ServeCtx>, headers: HeaderMap, Path(variant): Path<String>| async move {
                    if let Err(e) = ctx.auth().authenticate(ProtocolId::Native, &headers) {
                        return reply_err(e);
                    }
                    match detail(&variant) {
                        Some(v) => Json(v).into_response(),
                        None => reply_err(ApiError::not_found(format!(
                            "no gold record for `{variant}` (lab/gold/<variant>.toml)"
                        ))),
                    }
                },
            ),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_embedded_record_parses_and_is_named_after_its_variant() {
        assert_eq!(records().len(), EMBEDDED_GOLD.len());
        for r in records() {
            assert_eq!(r.path, format!("lab/gold/{}.toml", r.variant), "{}", r.path);
            assert!(
                r.value["tier"].is_object() && r.value["workload"].is_object(),
                "{}",
                r.path
            );
        }
    }

    #[test]
    fn embedded_list_is_the_directory() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../lab/gold");
        let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
            .expect("lab/gold")
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".toml"))
            .map(|n| format!("lab/gold/{n}"))
            .collect();
        on_disk.sort();
        let embedded: Vec<String> = EMBEDDED_GOLD.iter().map(|(p, _)| (*p).to_owned()).collect();
        assert_eq!(
            embedded, on_disk,
            "EMBEDDED_GOLD must list every lab/gold/*.toml"
        );
    }

    #[test]
    fn index_and_detail() {
        let i = index();
        assert_eq!(i["object"], "fv.gold");
        let rows = i["records"].as_array().unwrap();
        assert_eq!(rows.len(), EMBEDDED_GOLD.len());
        let d = detail("fasth3-4step-vsa").expect("the H3 turbo record");
        assert_eq!(d["object"], "fv.gold.record");
        assert_eq!(d["model"]["variant"], "fasth3-4step-vsa");
        assert_eq!(d["path"], "lab/gold/fasth3-4step-vsa.toml");
        assert!(d["toml"].as_str().unwrap().contains("[last_validated]"));
        assert!(detail("no-such-variant").is_none());
    }
}
