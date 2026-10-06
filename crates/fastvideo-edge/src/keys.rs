//! API key administration at the edge (docs/serve/edge-control-plane.md
//! §3): the same routes and shapes as `fastvideo_serve_kit::keys`
//! (`admin_routes`), over the D1 `api_keys` table the workers share.
//!
//! - `POST /fv/v1/admin/keys` `{name}` → 201 `{api_key, key}` (the only
//!   time the key is shown; D1 keeps its SHA-256 digest).
//! - `GET /fv/v1/admin/keys` → `{keys, backend: "d1"}`, newest first.
//! - `DELETE /fv/v1/admin/keys/{id}` → `{key}` or 404.
//! - `POST /fv/v1/admin/keys/invalidate` → `{key_epoch}`.
//!
//! A revoke or an invalidate bumps the registry's key epoch, so every
//! isolate drops its key cache on its next registry read (≤ 2 s) instead of
//! waiting out the 15 s cache.

use base64::Engine as _;
use fastvideo_dispatch_proto::front::{self as f, EdgeRoute};
use serde::Deserialize;
use serde_json::{json, Value};
use wasm_bindgen::{JsCast, JsValue};
use worker::*;

use crate::edge::{json_err, now_ms, rust_err, D1_BINDING};
use crate::front::{drop_key_cache, registry_call};

const KEY_PREFIX: &str = "fv_";
const NAME_MAX_CHARS: usize = 64;

#[derive(Deserialize)]
struct Row {
    id: String,
    name: String,
    prefix: String,
    created_at: f64,
    last_used_at: Option<f64>,
    revoked_at: Option<f64>,
}

fn rfc(ms: Option<f64>) -> Value {
    match ms {
        Some(ms) => json!(js_sys::Date::new(&JsValue::from_f64(ms)).to_iso_string().as_string()),
        None => Value::Null,
    }
}

fn view(r: &Row) -> Value {
    json!({
        "id": r.id,
        "name": r.name,
        "prefix": r.prefix,
        "created_at": rfc(Some(r.created_at)),
        "last_used_at": rfc(r.last_used_at),
        "revoked_at": rfc(r.revoked_at),
        "revoked": r.revoked_at.is_some(),
    })
}

/// 32 bytes from the platform CSPRNG (`crypto.getRandomValues`).
fn random_bytes() -> Result<[u8; 32]> {
    let crypto = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("crypto"))?;
    let get = js_sys::Reflect::get(&crypto, &JsValue::from_str("getRandomValues"))?.dyn_into::<js_sys::Function>()?;
    let buf = js_sys::Uint8Array::new_with_length(32);
    get.call1(&crypto, &buf)?;
    let mut out = [0u8; 32];
    buf.copy_to(&mut out);
    Ok(out)
}

/// Bumps the key epoch and drops this isolate's cache.
async fn bump_epoch(env: &Env) -> Result<u64> {
    drop_key_cache();
    let mut r = registry_call(env, Method::Post, "/registry/epoch", None).await?;
    let v: Value = r.json().await?;
    Ok(v["key_epoch"].as_u64().unwrap_or(0))
}

/// The table as `fastvideo_serve_kit`'s D1 migration 2 makes it, for an
/// edge whose workers have not run their migrations yet.
const CREATE: &str = "CREATE TABLE IF NOT EXISTS api_keys (id TEXT PRIMARY KEY NOT NULL, name TEXT NOT NULL, prefix TEXT NOT NULL, \
     digest TEXT NOT NULL UNIQUE, created_at INTEGER NOT NULL, last_used_at INTEGER, revoked_at INTEGER)";

const COLS: &str = "id, name, prefix, created_at, last_used_at, revoked_at";

async fn one(db: &D1Database, id: &str) -> Result<Option<Row>> {
    db.prepare(format!("SELECT {COLS} FROM api_keys WHERE id = ?")).bind(&[JsValue::from_str(id)])?.first::<Row>(None).await
}

pub(crate) async fn admin(req: &mut Request, env: &Env, r: &EdgeRoute, method: Method) -> Result<Response> {
    let db = match env.d1(D1_BINDING) {
        Ok(db) => db,
        Err(_) => return json_err(503, "unavailable", "this edge has no key database (D1 binding `DB`)"),
    };
    db.exec(CREATE).await?;
    match (r, method) {
        (EdgeRoute::Keys, Method::Post) => {
            #[derive(Deserialize)]
            struct MintBody {
                name: String,
            }
            let b: MintBody = match req.json().await {
                Ok(b) => b,
                Err(e) => return json_err(400, "invalid_request", &format!("expected {{\"name\": \"...\"}}: {e}")),
            };
            let name = b.name.trim();
            if name.is_empty() {
                return json_err(400, "invalid_request", "`name` must not be empty");
            }
            if name.chars().count() > NAME_MAX_CHARS {
                return json_err(400, "invalid_request", &format!("`name` must be at most {NAME_MAX_CHARS} characters"));
            }
            if name.chars().any(char::is_control) {
                return json_err(400, "invalid_request", "`name` must not contain control characters");
            }
            let (key, digest, id) = loop {
                let k = format!("{KEY_PREFIX}{}", base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(random_bytes()?));
                let d = f::key_digest(&k);
                let id = f::key_id(&d);
                if one(&db, &id).await?.is_none() {
                    break (k, d, id);
                }
            };
            let row = Row { id, name: name.to_owned(), prefix: format!("{}…", &key[..KEY_PREFIX.len() + 6]), created_at: now_ms() as f64, last_used_at: None, revoked_at: None };
            db.prepare("INSERT INTO api_keys (id, name, prefix, digest, created_at, last_used_at, revoked_at) VALUES (?, ?, ?, ?, ?, NULL, NULL)")
                .bind(&[
                    JsValue::from_str(&row.id),
                    JsValue::from_str(&row.name),
                    JsValue::from_str(&row.prefix),
                    JsValue::from_str(&digest),
                    JsValue::from_f64(row.created_at),
                ])?
                .run()
                .await?;
            // A miss cached before the mint would hide the key for 5 s here.
            drop_key_cache();
            Ok(Response::from_json(&json!({"api_key": key, "key": view(&row)}))?.with_status(201))
        }
        (EdgeRoute::Keys, Method::Get) => {
            let rows: Vec<Row> = db.prepare(format!("SELECT {COLS} FROM api_keys ORDER BY created_at DESC, id")).all().await?.results()?;
            Response::from_json(&json!({"keys": rows.iter().map(view).collect::<Vec<_>>(), "backend": "d1"}))
        }
        (EdgeRoute::KeyRevoke(id), Method::Delete) => {
            db.prepare("UPDATE api_keys SET revoked_at = COALESCE(revoked_at, ?) WHERE id = ?")
                .bind(&[JsValue::from_f64(now_ms() as f64), JsValue::from_str(id)])?
                .run()
                .await?;
            match one(&db, id).await? {
                Some(row) => {
                    bump_epoch(env).await?;
                    Response::from_json(&json!({"key": view(&row)}))
                }
                None => json_err(404, "not_found", &format!("no key `{id}`")),
            }
        }
        (EdgeRoute::KeysInvalidate, Method::Post) => {
            let e = bump_epoch(env).await.map_err(rust_err)?;
            Response::from_json(&json!({"key_epoch": e}))
        }
        _ => json_err(405, "method_not_allowed", "method not allowed"),
    }
}
