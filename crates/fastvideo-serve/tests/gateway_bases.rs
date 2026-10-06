//! The gateway configs in use: configs/serve/gateway.toml, gateway-pods.toml
//! and every config fv-control makes (control/test/fixtures/gateway-*.toml,
//! one per template plus overrides; its unit tests keep them current).
//!
//! - every one parses, validates, resolves its pools' static caps against
//!   the CUDA catalog, and starts a gateway (MockD1, no workers);
//! - a fal app or alias whose model no pool serves answers 404 and does not
//!   stop the gateway;
//! - gateway.toml and gateway-pods.toml agree except where they must differ;
//! - control/src/cluster/catalog.json's recipes agree with the CUDA catalog.

#![cfg(all(feature = "http-client", feature = "openai-videos", feature = "minimax", feature = "fal", feature = "ltxapi"))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use fastvideo_serve::config::{Config, KeyStoreBackend, ModelCfg, PoolKind};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::d1::mock::MockD1;
use fastvideo_serve_kit::{D1Client, KeyRing};
use serde_json::{json, Value};

const KEY: &str = "sk-bases-user";

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Every gateway config: (name, TOML).
fn bases() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for f in ["configs/serve/gateway.toml", "configs/serve/gateway-pods.toml"] {
        out.push((f.to_owned(), std::fs::read_to_string(repo().join(f)).unwrap()));
    }
    let dir = repo().join("control/test/fixtures");
    let mut names: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("gateway-") && p.extension().is_some_and(|x| x == "toml"))
        .collect();
    names.sort();
    assert!(names.len() >= 8, "control/test/fixtures has {} gateway configs", names.len());
    for p in names {
        out.push((p.strip_prefix(repo()).unwrap_or(&p).display().to_string(), std::fs::read_to_string(&p).unwrap()));
    }
    out
}

fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("fv-gwb-{tag}-{:x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()))
}

/// A base as the pod gets it: secrets and pool endpoints from the env.
fn load(name: &str, text: &str) -> Config {
    let mut c = Config::from_toml(text, name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let state = tmp("state");
    let arts = tmp("arts");
    std::fs::create_dir_all(&arts).unwrap();
    let mut env: BTreeMap<String, String> = [
        ("FV_INTERNAL_TOKEN", "t".to_owned()),
        ("FV_CF_ACCOUNT_ID", "a".to_owned()),
        ("FV_CF_API_TOKEN", "t".to_owned()),
        ("FV_D1_DATABASE_ID", "d".to_owned()),
        ("FV_URL_SIGNING_KEY", "k".to_owned()),
        ("FV_ADMIN_TOKEN", "fvadm_bases".to_owned()),
        ("FV_API_KEYS", KeyRing::hash_hex(KEY)),
        ("FV_STATE_DIR", state.display().to_string()),
        ("FV_ARTIFACTS_DIR", arts.display().to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect();
    for pool in &c.pools {
        if pool.kind != PoolKind::Pod {
            env.insert(format!("{}ENDPOINT", pool.env_prefix()), "ep".into());
        }
    }
    c.apply_env(&env).unwrap_or_else(|e| panic!("{name}: {e}"));
    c.auth.key_store = KeyStoreBackend::Memory;
    c.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
    c
}

/// Every base parses, its pools' static caps resolve against the CUDA
/// catalog, its fal apps are well-formed, and a gateway starts on it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_gateway_base_starts_a_gateway() {
    for (name, text) in bases() {
        let c = load(&name, &text);
        assert!(!c.pools.is_empty(), "{name}: no pools");
        for pool in &c.pools {
            let caps = fastvideo_serve::gateway::static_caps(pool).unwrap_or_else(|e| panic!("{name}: pool {}: {e}", pool.id));
            assert!(!caps.is_empty(), "{name}: pool {}", pool.id);
        }
        for app in &c.protocols.fal_apps {
            assert!(fastvideo_fal::FalApp::from_id(app).is_valid(), "{name}: fal app {app}");
        }
        let d1 = D1Client::new(Arc::new(MockD1::new()));
        let app =
            App::build(c, Overrides { d1: Some(d1), ..Overrides::default() }).await.unwrap_or_else(|e| panic!("{name}: the gateway does not start: {e:#}"));
        drop(app);
    }
}

async fn call(http: &reqwest::Client, method: &str, url: &str, body: Option<Value>, auth: &str) -> (u16, Value) {
    let mut r = http.request(reqwest::Method::from_bytes(method.as_bytes()).unwrap(), url).header("authorization", auth);
    if let Some(b) = body {
        r = r.json(&b);
    }
    let resp = r.send().await.unwrap();
    let s = resp.status().as_u16();
    let bytes = resp.bytes().await.unwrap();
    (s, serde_json::from_slice(&bytes).unwrap_or(Value::String(String::from_utf8_lossy(&bytes).into_owned())))
}

/// The `wan` template's gateway mounts every fal app, but only Wan pools
/// serve: the H3 and LTX apps answer 404, an alias to a model no pool serves
/// does not resolve, and the served app takes the job.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_app_without_a_pool_is_a_404_not_a_broken_gateway() {
    let name = "control/test/fixtures/gateway-wan.toml";
    let mut c = load(name, &std::fs::read_to_string(repo().join(name)).unwrap());
    assert!(c.protocols.fal_apps.iter().any(|a| a == "minimax/h3-turbo"));
    assert!(c.protocols.fal_apps.iter().any(|a| a == "fal-ai/ltx-2.3"));
    assert!(!c.pools.iter().any(|p| p.models.iter().any(|m| m.family == "h3" || m.family == "ltx2")));
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let g = format!("http://{}", l.local_addr().unwrap());
    c.server.public_base_url = Some(g.clone());
    let d1 = D1Client::new(Arc::new(MockD1::new()));
    let app = App::build(c, Overrides { d1: Some(d1), ..Overrides::default() }).await.unwrap();
    let router = app.router.clone();
    let task = tokio::spawn(async move {
        let _ = axum::serve(l, router).await;
    });
    let http = reqwest::Client::builder().no_proxy().build().unwrap();
    let fal = format!("Key {KEY}");
    // Mounted (a JSON error from the app, not the router's empty 404), but no pool serves the model.
    for (app_id, ep) in [
        ("minimax/h3-turbo", "text-to-video"),
        ("minimax/h3", "text-to-video"),
        ("minimax/h3-draft", "image-to-video"),
        ("lightricks/ltx-2.5", "text-to-video/fast"),
        ("lightricks/ltx-2.5", "text-to-video/pro"),
    ] {
        let body = if ep == "image-to-video" { json!({"prompt": "a kitten", "image_url": "https://example.com/a.png"}) } else { json!({"prompt": "a kitten"}) };
        let (s, v) = call(&http, "POST", &format!("{g}/{app_id}/{ep}"), Some(body), &fal).await;
        assert_eq!(s, 404, "{app_id}/{ep}: {v}");
        assert!(v.to_string().contains("not served") || v.to_string().contains("unknown model") || v.to_string().contains("not found"), "{app_id}/{ep}: {v}");
        assert_ne!(v, Value::String(String::new()), "{app_id}/{ep} is not mounted");
    }
    // The served app reaches its pool: 503 while the pool has no worker (not a 404).
    let (s, v) = call(&http, "POST", &format!("{g}/fal-ai/wan/v2.2-5b/text-to-video"), Some(json!({"prompt": "a kitten"})), &fal).await;
    assert_eq!(s, 503, "fal-ai/wan: {v}");
    assert!(v.to_string().contains("(`wan`)"), "{v}");
    // The gateway still answers.
    let (s, v) = call(&http, "GET", &format!("{g}/fv/v1/status"), None, &format!("Bearer {KEY}")).await;
    assert_eq!(s, 200, "{v}");
    task.abort();
}

/// gateway.toml (serverless pools) and gateway-pods.toml (the pod cluster,
/// fv-control's base) agree everywhere but the state dir, the Reactor model
/// and the pools.
#[test]
fn gateway_bases_are_in_sync() {
    let a = load("gateway.toml", &std::fs::read_to_string(repo().join("configs/serve/gateway.toml")).unwrap());
    let b = load("gateway-pods.toml", &std::fs::read_to_string(repo().join("configs/serve/gateway-pods.toml")).unwrap());
    let v = |x: &dyn erased::Ser| x.json();
    assert_eq!(v(&a.protocols), v(&b.protocols), "[protocols]");
    assert_eq!(a.aliases, b.aliases, "[aliases]");
    assert_eq!(v(&a.limits), v(&b.limits), "[limits]");
    assert_eq!(v(&a.auth), v(&b.auth), "[auth]");
    let mut ga = serde_json::to_value(&a.gateway).unwrap();
    let mut gb = serde_json::to_value(&b.gateway).unwrap();
    for g in [&mut ga, &mut gb] {
        g.as_object_mut().unwrap().remove("reactor_model");
    }
    assert_eq!(ga, gb, "[gateway] (all but reactor_model)");
    assert_eq!(a.server.shutdown_grace_s, b.server.shutdown_grace_s);
    assert_eq!(a.server.sync_timeout_s, b.server.sync_timeout_s);
    // The fal apps are every worker config's (none: fv-serve's default H3 apps).
    let mut want = std::collections::BTreeSet::new();
    for e in std::fs::read_dir(repo().join("configs/serve")).unwrap() {
        let p = e.unwrap().path();
        let n = p.file_name().unwrap().to_string_lossy().into_owned();
        if !n.starts_with("runpod") || n == "runpod-fake.toml" {
            continue;
        }
        let w = Config::from_toml(&std::fs::read_to_string(&p).unwrap(), &n).unwrap();
        if w.protocols.fal {
            want.extend(w.protocols.fal_apps.iter().cloned());
        }
    }
    for app in &want {
        assert!(b.protocols.fal_apps.contains(app), "gateway-pods.toml lacks the fal app {app}");
    }
}

/// control/src/cluster/catalog.json: the recipes it marks servable resolve
/// against the CUDA catalog (as the gateway resolves a pool's static caps),
/// the others do not, and every catalog model is listed.
#[test]
fn control_catalog_matches_the_cuda_catalog() {
    let cat: Value = serde_json::from_str(&std::fs::read_to_string(repo().join("control/src/cluster/catalog.json")).unwrap()).unwrap();
    let entry = |id: &str, family: &str, recipe: &str| -> ModelCfg { serde_json::from_value(json!({"id": id, "family": family, "recipe": recipe})).unwrap() };
    let mut listed = Vec::new();
    for r in cat["recipes"].as_array().unwrap() {
        let (id, fam, serve) = (r["id"].as_str().unwrap(), r["family"].as_str().unwrap(), r["serve"].as_bool().unwrap());
        let got = fastvideo_serve::app::catalog_models(&[entry("probe", fam, id)]);
        assert_eq!(got.is_ok(), serve, "recipe {id} ({fam}): serve = {serve} but {:?}", got.err());
        listed.push(id.to_owned());
    }
    for m in cat["models"].as_array().unwrap() {
        let (id, fam, rec) = (m["id"].as_str().unwrap(), m["family"].as_str().unwrap(), m["recipe"].as_str().unwrap());
        fastvideo_serve::app::catalog_models(&[entry(id, fam, rec)]).unwrap_or_else(|e| panic!("model {id}: {e}"));
    }
    for m in fastvideo_engine_service::cuda::catalog(&fastvideo_engine_service::cuda::WeightLayout::new("/w")) {
        assert!(listed.iter().any(|x| x == m.id.as_str()), "catalog model {} is not in control/src/cluster/catalog.json", m.id);
    }
}

/// serde_json for any Serialize (the config sections have no PartialEq).
mod erased {
    pub trait Ser {
        fn json(&self) -> serde_json::Value;
    }
    impl<T: serde::Serialize> Ser for T {
        fn json(&self) -> serde_json::Value {
            serde_json::to_value(self).unwrap()
        }
    }
}
