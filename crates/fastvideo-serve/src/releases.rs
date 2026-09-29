//! Release channels and deployments in the gateway's admin API (docs/serve/releases.md),
//! behind the admin token, for the console's Deployments page:
//!
//! | Route | Behaviour |
//! |---|---|
//! | `GET /fv/v1/admin/releases?channel=&limit=` | the head of every channel and the history (D1 `releases`) |
//! | `GET /fv/v1/admin/deployments` | live deployments (D1 `deployments`), the gateway's and every pod worker's build, drift against the channel each follows |
//! | `POST /fv/v1/admin/releases/promote` `{target, channel, notes, dry_run}` | dispatches `.github/workflows/release.yml` (`promote`); `dry_run`: the plan only |
//! | `POST /fv/v1/admin/releases/rollback` `{channel, to, dry_run}` | picks the release to go back to (as `release.sh rollback`) and dispatches the workflow with it as `to`; `dry_run`: the plan only |
//!
//! The gateway never retags or edits templates itself: the workflow does
//! (GHCR retag, Runpod templates, D1 record), with `GITHUB_TOKEN` and the
//! repository's secrets. Dispatching needs `FV_GITHUB_TOKEN` (a token with
//! `actions:write` on the repository); without it the routes answer 503 and
//! dry runs still work. Pod pools of a standing cluster are rolled with
//! `release.sh redeploy`; serverless endpoints follow their template.
//!
//! Env: `FV_GITHUB_TOKEN`, `FV_GITHUB_REPO` (default
//! `zaitrarrio/fastvideo-rs`), `FV_GITHUB_API` (default
//! `https://api.github.com`), `FV_RELEASE_WORKFLOW` (default `release.yml`),
//! `FV_RELEASE_REF` (default `main`), `FV_TEMPLATE_CHANNEL` (default
//! `stable`). Tables: `deploy/d1/registry.sql`, applied (idempotently) on
//! first use.

// Handler helpers return a ready `Response` as their error (early return).
#![allow(clippy::result_large_err)]

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use fastvideo_serve_kit::d1::{D1Client, Stmt};
use fastvideo_serve_kit::AdminToken;
use serde_json::{json, Map, Value};

use crate::build_info::BuildInfo;
use crate::config::{Env, Secret};
use crate::gateway::{Gateway, WorkerBuild};

/// The D1 tables (shared with scripts/serve/lib/registry.sh).
pub const REGISTRY_SQL: &str = include_str!("../../../deploy/d1/registry.sql");

/// Statements of [`REGISTRY_SQL`] (comments dropped).
pub fn registry_statements() -> Vec<String> {
    let text: String = REGISTRY_SQL.lines().map(|l| l.split("--").next().unwrap_or("")).collect::<Vec<_>>().join("\n");
    text.split(';').map(|s| s.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|s| !s.is_empty()).collect()
}

/// Where promotions are dispatched.
#[derive(Clone, Debug)]
pub struct ReleasesCfg {
    pub github_token: Secret,
    pub github_repo: String,
    pub github_api: String,
    pub workflow: String,
    pub git_ref: String,
    pub template_channel: String,
}

impl ReleasesCfg {
    pub fn from_env(env: &dyn Env) -> Self {
        let v = |k: &str, d: &str| env.var(k).map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).unwrap_or_else(|| d.to_owned());
        Self {
            github_token: Secret(env.var("FV_GITHUB_TOKEN").unwrap_or_default().trim().to_owned()),
            github_repo: v("FV_GITHUB_REPO", "zaitrarrio/fastvideo-rs"),
            github_api: v("FV_GITHUB_API", "https://api.github.com").trim_end_matches('/').to_owned(),
            workflow: v("FV_RELEASE_WORKFLOW", "release.yml"),
            git_ref: v("FV_RELEASE_REF", "main"),
            template_channel: v("FV_TEMPLATE_CHANNEL", "stable"),
        }
    }

    fn dispatch_info(&self) -> Value {
        json!({
            "configured": !self.github_token.is_empty(),
            "repo": self.github_repo,
            "workflow": self.workflow,
            "ref": self.git_ref,
            "runs_url": format!("https://github.com/{}/actions/workflows/{}", self.github_repo, self.workflow),
        })
    }
}

struct Inner {
    gw: Arc<Gateway>,
    admin: Arc<AdminToken>,
    cfg: ReleasesCfg,
    schema: tokio::sync::OnceCell<()>,
}

type St = Arc<Inner>;

/// The admin release routes (gateway mode).
pub fn routes(gw: Arc<Gateway>, admin: Arc<AdminToken>, cfg: ReleasesCfg) -> Router {
    Router::new()
        .route("/fv/v1/admin/releases", get(list))
        .route("/fv/v1/admin/deployments", get(deployments))
        .route("/fv/v1/admin/releases/promote", post(promote))
        .route("/fv/v1/admin/releases/rollback", post(rollback))
        .with_state(Arc::new(Inner { gw, admin, cfg, schema: tokio::sync::OnceCell::new() }))
}

fn err(code: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (code, Json(json!({"error": {"kind": kind, "message": msg.into()}}))).into_response()
}

fn unauthorized() -> Response {
    err(StatusCode::UNAUTHORIZED, "unauthorized", "the admin token is required")
}

impl Inner {
    fn db(&self) -> &D1Client {
        &self.gw.db
    }

    async fn ensure(&self) -> Result<(), Response> {
        self.schema
            .get_or_try_init(|| async {
                self.db().batch(registry_statements().into_iter().map(Stmt::raw).collect()).await.map(|_| ())
            })
            .await
            .map(|_| ())
            .map_err(|e| err(StatusCode::BAD_GATEWAY, "d1", format!("release tables: {e}")))
    }

    async fn rows(&self, sql: &str, params: Vec<Value>) -> Result<Vec<Map<String, Value>>, Response> {
        self.ensure().await?;
        self.db().query(Stmt::new(sql, params)).await.map(|r| r.rows).map_err(|e| err(StatusCode::BAD_GATEWAY, "d1", e.to_string()))
    }

    async fn heads(&self) -> Result<Vec<Value>, Response> {
        let rows = self
            .rows(
                "SELECT r.* FROM releases r JOIN (SELECT channel, MAX(id) AS id FROM releases GROUP BY channel) h ON r.id = h.id ORDER BY r.channel",
                vec![],
            )
            .await?;
        Ok(rows.into_iter().map(release_json).collect())
    }

    async fn dispatch(&self, inputs: Value) -> Result<(), Response> {
        if self.cfg.github_token.is_empty() {
            return Err(err(
                StatusCode::SERVICE_UNAVAILABLE,
                "not_configured",
                "promotion is not configured on this gateway (FV_GITHUB_TOKEN); use scripts/serve/release.sh or the release workflow",
            ));
        }
        let url = format!("{}/repos/{}/actions/workflows/{}/dispatches", self.cfg.github_api, self.cfg.github_repo, self.cfg.workflow);
        let r = self
            .gw
            .http
            .post(url)
            .bearer_auth(self.cfg.github_token.expose())
            .header("accept", "application/vnd.github+json")
            .header("user-agent", "fv-serve")
            .json(&json!({"ref": self.cfg.git_ref, "inputs": inputs}))
            .send()
            .await
            .map_err(|e| err(StatusCode::BAD_GATEWAY, "github", e.without_url().to_string()))?;
        let code = r.status();
        if code == reqwest::StatusCode::NO_CONTENT || code.is_success() {
            return Ok(());
        }
        let body: Value = r.json().await.unwrap_or(Value::Null);
        let msg = body.get("message").and_then(Value::as_str).unwrap_or("").chars().take(200).collect::<String>();
        Err(err(StatusCode::BAD_GATEWAY, "github", format!("workflow dispatch answered {code}: {msg}")))
    }
}

/// A `releases` row with `digests` parsed.
fn release_json(mut r: Map<String, Value>) -> Value {
    if let Some(Value::String(d)) = r.get("digests") {
        let parsed: Value = serde_json::from_str(d).unwrap_or(Value::Null);
        r.insert("digests".into(), parsed);
    }
    Value::Object(r)
}

/// `true` for a channel name the release tooling accepts (`release.sh check_channel`).
pub fn valid_channel(c: &str) -> bool {
    let ok_chars = c.len() >= 2
        && c.len() <= 31
        && c.starts_with(|ch: char| ch.is_ascii_lowercase())
        && c.chars().all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-');
    let variants = ["h3-turbo", "h3-max", "ltx", "wan", "wan5b", "sfwan", "gateway"];
    ok_chars && !c.starts_with("sha-") && !c.starts_with("buildcache") && !c.contains("-sha-") && !variants.contains(&c)
}

/// `true` for a promote target: a git sha (7-40 hex), a digest
/// (`sha256:<64 hex>`, optionally `<repo>@…`) or a tag.
pub fn valid_target(t: &str) -> bool {
    let hex = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c));
    if (7..=40).contains(&t.len()) && hex(t) {
        return true;
    }
    if let Some(d) = t.rsplit_once("sha256:").map(|(_, d)| d) {
        let repo = &t[..t.len() - d.len() - "sha256:".len()];
        let repo_ok = repo.is_empty() || (repo.ends_with('@') && repo.chars().all(|c| c.is_ascii_alphanumeric() || "./-_@:".contains(c)));
        return d.len() == 64 && hex(d) && repo_ok;
    }
    !t.is_empty() && t.len() <= 128 && t.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c)) && !t.starts_with(['.', '-'])
}

/// Drift of an image against the channel it follows: `null` when it cannot
/// be told (no digest, no release for the channel or key), `false` when the
/// channel's current release has this digest for `key`, else the channel's
/// short sha (what it should run).
pub fn drift(heads: &[Value], channel: &str, key: &str, digest: Option<&str>) -> Value {
    let Some(d) = digest.filter(|d| d.starts_with("sha256:")) else { return Value::Null };
    let Some(h) = heads.iter().find(|h| h["channel"] == channel) else { return Value::Null };
    match h["digests"].get(key).and_then(Value::as_str) {
        None => Value::Null,
        Some(r) if r.ends_with(&format!("@{d}")) => Value::Bool(false),
        Some(_) => Value::String(h["git_sha"].as_str().unwrap_or("").chars().take(7).collect()),
    }
}

async fn list(State(st): State<St>, headers: HeaderMap, Query(q): Query<BTreeMap<String, String>>) -> Response {
    if !st.admin.check_headers(&headers) {
        return unauthorized();
    }
    let limit: i64 = q.get("limit").and_then(|l| l.parse().ok()).unwrap_or(30).clamp(1, 200);
    let heads = match st.heads().await {
        Ok(h) => h,
        Err(r) => return r,
    };
    let history = match q.get("channel").filter(|c| !c.is_empty()) {
        Some(c) => st.rows("SELECT * FROM releases WHERE channel = ? ORDER BY id DESC LIMIT ?", vec![json!(c), json!(limit)]).await,
        None => st.rows("SELECT * FROM releases ORDER BY id DESC LIMIT ?", vec![json!(limit)]).await,
    };
    let history = match history {
        Ok(r) => r.into_iter().map(release_json).collect::<Vec<_>>(),
        Err(r) => return r,
    };
    Json(json!({
        "object": "fv.releases",
        "template_channel": st.cfg.template_channel,
        "heads": heads,
        "history": history,
        "dispatch": st.cfg.dispatch_info(),
    }))
    .into_response()
}

fn build_view(heads: &[Value], template_channel: &str, b: Option<&WorkerBuild>) -> Value {
    let Some(b) = b else { return json!({"sha": "unknown", "drift": null}) };
    let channel = b.channel.clone().unwrap_or_else(|| template_channel.to_owned());
    let key = b.variant.clone().unwrap_or_else(|| "debug".into());
    json!({
        "sha": WorkerBuild::short_sha(Some(b)),
        "git_sha": b.git_sha,
        "version": b.version,
        "build_time": b.build_time,
        "variant": b.variant,
        "channel": b.channel,
        "follows": channel,
        "image_digest": b.image_digest,
        "image_tag": b.image_tag,
        "drift": drift(heads, &channel, &key, b.image_digest.as_deref()),
    })
}

async fn deployments(State(st): State<St>, headers: HeaderMap) -> Response {
    if !st.admin.check_headers(&headers) {
        return unauthorized();
    }
    let heads = match st.heads().await {
        Ok(h) => h,
        Err(r) => return r,
    };
    let rows = match st.rows("SELECT * FROM deployments WHERE deleted_at IS NULL ORDER BY created_at DESC LIMIT 500", vec![]).await {
        Ok(r) => r,
        Err(r) => return r,
    };
    let tc = st.cfg.template_channel.as_str();
    let deployments: Vec<Value> = rows
        .into_iter()
        .map(|mut r| {
            let channel = r.get("channel").and_then(Value::as_str).unwrap_or(tc).to_owned();
            let key = r.get("variant").and_then(Value::as_str).unwrap_or("debug").to_owned();
            let d = drift(&heads, &channel, &key, r.get("digest").and_then(Value::as_str));
            if let Some(Value::String(m)) = r.get("meta") {
                let parsed: Value = serde_json::from_str(m).unwrap_or(Value::Null);
                r.insert("meta".into(), parsed);
            }
            r.insert("follows".into(), json!(channel));
            r.insert("drift".into(), d);
            Value::Object(r)
        })
        .collect();
    let own = BuildInfo::current();
    let own_build = WorkerBuild {
        version: Some(own.version.clone()),
        git_sha: Some(own.git_sha.clone()),
        build_time: own.build_time.clone(),
        variant: own.variant.clone(),
        channel: own.channel.clone(),
        image_digest: own.image.digest.clone(),
        image_tag: own.image.tag.clone(),
    };
    let pools: Vec<Value> = st
        .gw
        .pools
        .iter()
        .map(|p| {
            let s = p.lock();
            let workers: Vec<Value> = s
                .workers
                .values()
                .map(|w| {
                    json!({"url": w.url, "id": w.id, "state": w.state(), "healthy": w.healthy,
                           "build": build_view(&heads, tc, w.build.as_ref())})
                })
                .collect();
            let (versions, mixed) = crate::status::versions(
                s.workers.values().filter(|w| w.healthy).map(|w| (WorkerBuild::short_sha(w.build.as_ref()), w.build.as_ref().and_then(|b| b.channel.clone()))),
            );
            json!({"id": p.id(), "kind": p.cfg.kind, "endpoint_id": p.cfg.endpoint_id, "workers": workers,
                   "versions": versions, "mixed_versions": mixed})
        })
        .collect();
    Json(json!({
        "object": "fv.deployments",
        "template_channel": tc,
        "gateway": build_view(&heads, tc, Some(&own_build)),
        "heads": heads,
        "pools": pools,
        "deployments": deployments,
        "dispatch": st.cfg.dispatch_info(),
    }))
    .into_response()
}

fn body_str(b: &Value, k: &str) -> Option<String> {
    b.get(k).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned)
}

async fn promote(State(st): State<St>, headers: HeaderMap, body: Option<Json<Value>>) -> Response {
    if !st.admin.check_headers(&headers) {
        return unauthorized();
    }
    let b = body.map(|Json(b)| b).unwrap_or(Value::Null);
    let Some(target) = body_str(&b, "target") else {
        return err(StatusCode::BAD_REQUEST, "invalid_request", "`target`: a git sha, an image digest or a tag");
    };
    if !valid_target(&target) {
        return err(StatusCode::BAD_REQUEST, "invalid_request", "`target` is not a git sha, digest or tag");
    }
    let channel = body_str(&b, "channel").unwrap_or_else(|| "stable".into());
    if !valid_channel(&channel) {
        return err(StatusCode::BAD_REQUEST, "invalid_request", format!("`channel` {channel:?} is not a channel name"));
    }
    let notes: String = body_str(&b, "notes").unwrap_or_default().chars().take(200).collect();
    let dry = b.get("dry_run").and_then(Value::as_bool).unwrap_or(false);
    let current = match st.heads().await {
        Ok(h) => h.into_iter().find(|h| h["channel"] == channel.as_str()).unwrap_or(Value::Null),
        Err(r) => return r,
    };
    let inputs = json!({"action": "promote", "target": target, "channel": channel, "notes": notes,
                        "to": "", "templates": "true", "allow_partial": "false", "dry_run": "false"});
    plan_or_dispatch(&st, dry, inputs, json!({"current": current, "templates": channel == st.cfg.template_channel})).await
}

async fn rollback(State(st): State<St>, headers: HeaderMap, body: Option<Json<Value>>) -> Response {
    if !st.admin.check_headers(&headers) {
        return unauthorized();
    }
    let b = body.map(|Json(b)| b).unwrap_or(Value::Null);
    let channel = body_str(&b, "channel").unwrap_or_else(|| "stable".into());
    if !valid_channel(&channel) {
        return err(StatusCode::BAD_REQUEST, "invalid_request", format!("`channel` {channel:?} is not a channel name"));
    }
    let to = b.get("to").and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok())));
    let dry = b.get("dry_run").and_then(Value::as_bool).unwrap_or(false);
    let current = match st.rows("SELECT * FROM releases WHERE channel = ? ORDER BY id DESC LIMIT 1", vec![json!(channel)]).await {
        Ok(mut r) if !r.is_empty() => release_json(r.remove(0)),
        Ok(_) => return err(StatusCode::CONFLICT, "no_history", format!("no release recorded for {channel}")),
        Err(r) => return r,
    };
    // The same choice as `release.sh rollback`: the newest earlier release
    // with another sha that was not itself rolled back.
    let target = match to {
        Some(id) => st.rows("SELECT * FROM releases WHERE id = ? AND channel = ?", vec![json!(id), json!(channel)]).await,
        None => {
            st.rows(
                "SELECT * FROM releases WHERE channel = ? AND id < ? AND git_sha != ? AND rolled_back_at IS NULL ORDER BY id DESC LIMIT 1",
                vec![json!(channel), current["id"].clone(), current["git_sha"].clone()],
            )
            .await
        }
    };
    let target = match target {
        Ok(mut r) if !r.is_empty() => release_json(r.remove(0)),
        Ok(_) => {
            return err(
                StatusCode::CONFLICT,
                "no_history",
                match to {
                    Some(id) => format!("no release {id} in {channel}"),
                    None => format!("{channel} has no earlier release to roll back to"),
                },
            )
        }
        Err(r) => return r,
    };
    let inputs = json!({"action": "rollback", "target": "", "channel": channel, "notes": "",
                        "to": target["id"].to_string(), "templates": "true", "allow_partial": "false", "dry_run": "false"});
    plan_or_dispatch(&st, dry, inputs, json!({"current": current, "target": target, "templates": channel == st.cfg.template_channel})).await
}

async fn plan_or_dispatch(st: &Inner, dry: bool, inputs: Value, extra: Value) -> Response {
    let mut out = json!({
        "object": "fv.release.request",
        "dry_run": dry,
        "workflow": st.cfg.workflow,
        "ref": st.cfg.git_ref,
        "inputs": inputs,
        "dispatch": st.cfg.dispatch_info(),
    });
    if let (Some(o), Some(e)) = (out.as_object_mut(), extra.as_object()) {
        o.extend(e.clone());
    }
    if dry {
        return Json(out).into_response();
    }
    if let Err(r) = st.dispatch(inputs.clone()).await {
        return r;
    }
    tracing::info!(inputs = %inputs, "admin: release workflow dispatched");
    out["dispatched"] = json!(true);
    (StatusCode::ACCEPTED, Json(out)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_splits_into_statements() {
        let s = registry_statements();
        assert_eq!(s.len(), 6, "{s:?}");
        assert!(s[0].starts_with("CREATE TABLE IF NOT EXISTS releases ("), "{}", s[0]);
        assert!(s.iter().all(|x| !x.contains("--") && x.starts_with("CREATE ")));
    }

    #[test]
    fn channel_and_target_names() {
        for c in ["stable", "latest", "canary", "rc-2"] {
            assert!(valid_channel(c), "{c}");
        }
        for c in ["", "s", "Stable", "sha-abc", "wan", "gateway", "x;y", "buildcache", "a-sha-b", "1abc"] {
            assert!(!valid_channel(c), "{c}");
        }
        let d = "c782eb378f3f5e41139096010c4942244793cde70b5e5bba97877a11ecfd6045";
        for t in ["2cd1ba0", "2cd1ba0e5531f3bec378d745668d040c63e36ba0", &format!("sha256:{d}"), &format!("ghcr.io/o/r@sha256:{d}"), "latest", "sha-2cd1ba0", "h3-turbo-sha-2cd1ba0"] {
            assert!(valid_target(t), "{t}");
        }
        for t in ["", "sha256:abc", "a b", "$(x)", "-x", "x;rm", &format!("ghcr.io/o r@sha256:{d}")] {
            assert!(!valid_target(t), "{t}");
        }
    }

    #[test]
    fn drift_against_the_channel_head() {
        let d = "sha256:aaaa";
        let heads = vec![json!({"channel": "stable", "git_sha": "bbbbbbb1234", "digests": {"h3-turbo": "r@sha256:aaaa", "wan": "r@sha256:cccc"}})];
        assert_eq!(drift(&heads, "stable", "h3-turbo", Some(d)), json!(false));
        assert_eq!(drift(&heads, "stable", "wan", Some(d)), json!("bbbbbbb"));
        assert_eq!(drift(&heads, "stable", "ltx", Some(d)), Value::Null);
        assert_eq!(drift(&heads, "latest", "h3-turbo", Some(d)), Value::Null);
        assert_eq!(drift(&heads, "stable", "h3-turbo", None), Value::Null);
    }

    #[test]
    fn config_from_env_and_no_token_in_debug() {
        let mut env = BTreeMap::new();
        env.insert("FV_GITHUB_TOKEN".to_owned(), "ghp_secret".to_owned());
        env.insert("FV_GITHUB_API".to_owned(), "http://127.0.0.1:1/".to_owned());
        let c = ReleasesCfg::from_env(&env);
        assert_eq!(c.github_api, "http://127.0.0.1:1");
        assert_eq!((c.github_repo.as_str(), c.workflow.as_str(), c.git_ref.as_str(), c.template_channel.as_str()), ("zaitrarrio/fastvideo-rs", "release.yml", "main", "stable"));
        assert!(!format!("{c:?}").contains("ghp_secret"));
        assert!(!c.dispatch_info().to_string().contains("ghp_secret"));
        assert_eq!(c.dispatch_info()["configured"], true);
        assert_eq!(ReleasesCfg::from_env(&BTreeMap::<String, String>::new()).dispatch_info()["configured"], false);
    }
}
