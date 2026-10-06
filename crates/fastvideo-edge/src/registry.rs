//! The `Registry` Durable Object (one, named `registry`): what the public
//! front routes by (docs/serve/edge-control-plane.md §1, §2.3).
//!
//! - Each family object posts its status here when its workers or jobs
//!   change (`POST /registry/family/{f}`, throttled there); `GET /registry`
//!   answers every family's workers, load and owners in flight
//!   ([`Registry`]), which each Worker isolate caches for a few seconds.
//! - `key_epoch` (`POST /registry/epoch`): bumped on a key revoke or an
//!   invalidate; isolates drop their key caches when it moves.
//! - Session bindings (`/registry/bind/{alias}`): the id clients use for a
//!   session (the director's session id, the Reactor caller, a stream id) →
//!   its family object session and the worker's endpoint.
//!
//! State is a key-value table in the object's SQLite storage.

use std::cell::RefCell;
use std::collections::BTreeMap;

use fastvideo_dispatch_proto::front::{Registry as View, SessionBinding};
use fastvideo_dispatch_proto::PoolStatus;
use serde::Deserialize;
use worker::*;

use crate::edge::{json_err, now_ms, rust_err};

#[derive(Deserialize)]
struct Kv {
    k: String,
    v: String,
}

#[durable_object]
pub struct Registry {
    state: State,
    #[allow(dead_code)]
    env: Env,
    loaded: RefCell<bool>,
    families: RefCell<BTreeMap<String, PoolStatus>>,
    epoch: RefCell<u64>,
}

impl Registry {
    fn sql(&self) -> SqlStorage {
        self.state.storage().sql()
    }

    fn load(&self) -> Result<()> {
        if *self.loaded.borrow() {
            return Ok(());
        }
        let sql = self.sql();
        sql.exec("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)", None)?;
        let rows: Vec<Kv> = sql.exec("SELECT k, v FROM kv WHERE k LIKE 'fam:%' OR k = 'epoch'", None)?.to_array()?;
        let mut fams = BTreeMap::new();
        for r in rows {
            if let Some(f) = r.k.strip_prefix("fam:") {
                if let Ok(st) = serde_json::from_str::<PoolStatus>(&r.v) {
                    fams.insert(f.to_owned(), st);
                }
            } else if r.k == "epoch" {
                *self.epoch.borrow_mut() = r.v.parse().unwrap_or(0);
            }
        }
        *self.families.borrow_mut() = fams;
        *self.loaded.borrow_mut() = true;
        Ok(())
    }

    fn put(&self, k: &str, v: &str) -> Result<()> {
        self.sql().exec("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", vec![k.into(), v.into()])?;
        Ok(())
    }

    fn get(&self, k: &str) -> Result<Option<String>> {
        let rows: Vec<Kv> = self.sql().exec("SELECT k, v FROM kv WHERE k = ?", vec![k.into()])?.to_array()?;
        Ok(rows.into_iter().next().map(|r| r.v))
    }

    fn view(&self) -> View {
        let fams: Vec<(String, PoolStatus)> = self.families.borrow().iter().map(|(f, s)| (f.clone(), s.clone())).collect();
        View::from_statuses(fams, *self.epoch.borrow(), now_ms())
    }
}

impl DurableObject for Registry {
    fn new(state: State, env: Env) -> Self {
        Self { state, env, loaded: RefCell::new(false), families: RefCell::new(BTreeMap::new()), epoch: RefCell::new(0) }
    }

    async fn fetch(&self, mut req: Request) -> Result<Response> {
        self.load()?;
        let path = req.path();
        let segs: Vec<String> = path.trim_start_matches('/').split('/').map(str::to_owned).collect();
        match (req.method(), segs.iter().map(String::as_str).collect::<Vec<_>>().as_slice()) {
            (Method::Get, ["registry"]) => Response::from_json(&self.view()),
            (Method::Post, ["registry", "family", f]) => {
                let st: PoolStatus = match req.json().await {
                    Ok(s) => s,
                    Err(e) => return json_err(400, "invalid_request", &format!("status body: {e}")),
                };
                let text = serde_json::to_string(&st).map_err(rust_err)?;
                self.put(&format!("fam:{f}"), &text)?;
                self.families.borrow_mut().insert((*f).to_owned(), st);
                Response::from_json(&serde_json::json!({"ok": true}))
            }
            (Method::Post, ["registry", "epoch"]) => {
                let e = *self.epoch.borrow() + 1;
                *self.epoch.borrow_mut() = e;
                self.put("epoch", &e.to_string())?;
                Response::from_json(&serde_json::json!({"key_epoch": e}))
            }
            (Method::Put, ["registry", "bind", alias]) => {
                let b: SessionBinding = match req.json().await {
                    Ok(b) => b,
                    Err(e) => return json_err(400, "invalid_request", &format!("binding body: {e}")),
                };
                self.put(&format!("bind:{alias}"), &serde_json::to_string(&b).map_err(rust_err)?)?;
                Response::from_json(&serde_json::json!({"ok": true}))
            }
            (Method::Get, ["registry", "bind", alias]) => match self.get(&format!("bind:{alias}"))? {
                Some(v) => Response::ok(v).map(|r| {
                    let h = Headers::new();
                    let _ = h.set("content-type", "application/json");
                    r.with_headers(h)
                }),
                None => json_err(404, "not_found", "no such session"),
            },
            (Method::Delete, ["registry", "bind", alias]) => {
                self.sql().exec("DELETE FROM kv WHERE k = ?", vec![format!("bind:{alias}").into()])?;
                Response::from_json(&serde_json::json!({"ok": true}))
            }
            // Bindings of a family whose alias starts with `prefix` (the
            // director sessions to check before refusing a session).
            (Method::Get, ["registry", "binds", f, prefix]) => {
                let rows: Vec<Kv> = self.sql().exec("SELECT k, v FROM kv WHERE k LIKE ?", vec![format!("bind:{prefix}%").into()])?.to_array()?;
                let out: BTreeMap<String, SessionBinding> = rows
                    .into_iter()
                    .filter_map(|r| {
                        let b: SessionBinding = serde_json::from_str(&r.v).ok()?;
                        (b.family == *f).then(|| (r.k.trim_start_matches("bind:").to_owned(), b))
                    })
                    .collect();
                Response::from_json(&out)
            }
            _ => json_err(404, "not_found", "no such route"),
        }
    }
}
