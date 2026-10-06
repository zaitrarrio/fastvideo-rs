//! The Reactor runtime mounted in fv-serve (design §5.7, §9; WP-13): the
//! local session routes and the signalling group on the assembled router,
//! over the fake engine's first resident stream model.
#![cfg(feature = "reactor")]

use std::collections::BTreeMap;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use fastvideo_serve::config::{Config, JobBackend};
use fastvideo_serve::router::{route_table, Owner};
use fastvideo_serve::{App, Overrides};
use serde_json::{json, Value};
use tower::ServiceExt;

fn config(tag: &str) -> Config {
    let dir = tempfile::Builder::new().prefix(&format!("fv-serve-reactor-{tag}-")).tempdir().unwrap().keep();
    let mut env = BTreeMap::new();
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "test-signing-key".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    env.insert("ORPHAN_TIMEOUT_SECONDS".to_owned(), "30".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::File;
    c.engine.fake.step_ms = 1;
    c.reactor.short_edge = Some(96);
    c.reactor.h264 = "off".into();
    c.validate().unwrap();
    c
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let b = Request::builder().method(method).uri(uri);
    let req = match body {
        Some(v) => b.header("content-type", "application/json").body(Body::from(v.to_string())).unwrap(),
        None => b.body(Body::empty()).unwrap(),
    };
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 24).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reactor_routes_are_mounted() {
    let c = config("mount");
    assert_eq!(c.reactor.orphan_timeout_s, 30);
    let app = App::build(c, Overrides::default()).await.unwrap();
    let rt = app.reactor.clone().expect("reactor mounted");
    app.gate.engine().wait_ready().await;
    let (s, d) = call(&app.router, "GET", "/session", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(d["state"], "ready");
    assert_eq!(d["session_id"], fastvideo_reactor::SESSION_ID);
    assert!(d["capabilities"]["tracks"].is_array(), "{d}");
    let (s, schema) = call(&app.router, "GET", "/schema", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(schema["openapi"], "3.1.0");
    let (s, d) = call(&app.router, "POST", "/start_session", Some(json!({}))).await;
    assert_eq!(s, StatusCode::OK, "{d}");
    assert_eq!(d["state"], "waiting");
    // Every §9 Reactor route (but the SSE journal) reaches a handler.
    for spec in route_table(&[]).into_iter().filter(|s| s.owner == Owner::Reactor && s.path != "/events") {
        if spec.path.ends_with("_session") {
            continue;
        }
        let uri = spec
            .path
            .replace("{sid}", fastvideo_reactor::SESSION_ID)
            .replace("{cid}", "1002");
        let body = matches!(spec.method, "POST" | "PUT").then(|| json!({}));
        let (s, v) = call(&app.router, spec.method, &uri, body).await;
        assert_ne!(s, StatusCode::METHOD_NOT_ALLOWED, "{} {uri}", spec.method);
        assert!(s != StatusCode::NOT_FOUND || !v.is_null(), "{} {uri} unrouted", spec.method);
    }
    let (s, v) = call(&app.router, "POST", "/sessions/00000000-0000-0000-0000-000000000000/transport/webrtc/connections", None).await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    let (s, _) = call(&app.router, "POST", "/stop_session", Some(json!({"reason": "done"}))).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(rt.state(), fastvideo_reactor::RtState::Ready);
}
