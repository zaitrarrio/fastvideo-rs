//! `[upsampler]` end to end (docs/serve/prompt-upsampling.md): fv-serve
//! configured from the environment against a local fake OpenAI-compatible
//! server; native jobs opt in with `prompt_expansion`, MiniMax with
//! `prompt_optimizer`; the key goes out as a bearer token and never into
//! logs or the config's Debug.
//!
//! `cargo test -p fastvideo-serve --features http-client --test upsample`
#![cfg(feature = "http-client")]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use fastvideo_protocol::ProtocolId;
use fastvideo_serve::config::{Config, JobBackend};
use fastvideo_serve::{App, Overrides};
use fastvideo_serve_kit::KeyRing;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::ServiceExt;

const KEY: &str = "sk-test-fv";

/// (authorization, request body) of each call.
type Seen = Arc<Mutex<Vec<(Option<String>, Value)>>>;

/// A fake `/v1/chat/completions`: answers `Rewritten: <user message>`.
async fn fake_llm() -> (String, Seen) {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", l.local_addr().unwrap());
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let s = seen.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = l.accept().await else { return };
            let s = s.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                loop {
                    let n = sock.read(&mut tmp).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") else { continue };
                    let head = String::from_utf8_lossy(&buf[..i]).to_ascii_lowercase();
                    let len: usize = head.lines().find_map(|l| l.strip_prefix("content-length:")).map_or(0, |v| v.trim().parse().unwrap());
                    if buf.len() < i + 4 + len {
                        continue;
                    }
                    let raw = String::from_utf8_lossy(&buf[..i]).to_string();
                    let auth = raw.lines().find_map(|l| l.split_once(':').filter(|(k, _)| k.eq_ignore_ascii_case("authorization")).map(|(_, v)| v.trim().to_owned()));
                    let body: Value = serde_json::from_slice(&buf[i + 4..i + 4 + len]).unwrap();
                    let user = body["messages"][1]["content"].as_str().unwrap_or_default().to_owned();
                    s.lock().unwrap().push((auth, body));
                    let answer = json!({"choices": [{"message": {"content": format!("Rewritten: {user}")}}]}).to_string();
                    let resp = format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{answer}", answer.len());
                    let _ = sock.write_all(resp.as_bytes()).await;
                    let _ = sock.shutdown().await;
                    return;
                }
            });
        }
    });
    (url, seen)
}

fn config(llm: &str) -> Config {
    let dir = tempfile::Builder::new().prefix("fv-serve-upsample-").tempdir().unwrap().keep();
    let mut env = BTreeMap::new();
    env.insert("FV_API_KEYS".to_owned(), KeyRing::hash_hex(KEY));
    env.insert("FV_URL_SIGNING_KEY".to_owned(), "test-signing-key".to_owned());
    env.insert("FV_STATE_DIR".to_owned(), dir.display().to_string());
    env.insert("FV_PUBLIC_BASE_URL".to_owned(), "http://fv.test".to_owned());
    env.insert("FV_UPSAMPLER_BASE_URL".to_owned(), llm.to_owned());
    env.insert("FV_UPSAMPLER_MODEL".to_owned(), "fake-llm".to_owned());
    env.insert("FV_UPSAMPLER_API_KEY".to_owned(), "llm-secret-key".to_owned());
    env.insert("FV_UPSAMPLER_HEDGE".to_owned(), "0".to_owned());
    let mut c = Config::default();
    c.apply_env(&env).unwrap();
    c.jobs.backend = JobBackend::File;
    c.engine.fake.step_ms = 1;
    c.validate().unwrap();
    c
}

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let b = Request::builder().method(method).uri(uri).header("authorization", format!("Bearer {KEY}"));
    let req = match body {
        Some(v) => b.header("content-type", "application/json").body(Body::from(v.to_string())).unwrap(),
        None => b.body(Body::empty()).unwrap(),
    };
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 24).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

async fn wait_done(app: &Router, id: &str) -> Value {
    for _ in 0..4000 {
        let (_, v) = call(app, "GET", &format!("/fv/v1/jobs/{id}"), None).await;
        if matches!(v["status"].as_str(), Some("succeeded" | "failed")) {
            return v;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("job {id} never finished");
}

#[tokio::test]
async fn native_and_minimax_flags_rewrite_through_the_configured_llm() {
    let (llm, seen) = fake_llm().await;
    let c = config(&llm);
    let u = c.upsampler.as_ref().expect("a base URL alone creates [upsampler]");
    assert_eq!((u.model.as_str(), u.hedge, u.default_on, u.timeout_ms), ("fake-llm", false, false, 10_000));
    assert!(!format!("{c:?}").contains("llm-secret-key"), "the key never prints");
    assert!(!serde_json::to_string(&c).unwrap().contains("llm-secret-key"));
    let app = App::build(c, Overrides::default()).await.unwrap();
    app.gate.engine().wait_ready().await;
    let r = app.router.clone();

    // Native: explicit `quality`.
    let body = json!({"model": "h3-turbo", "prompt": "a cat", "aspect_ratio": "16:9", "short_edge": 768, "prompt_expansion": "quality"});
    let (s, v) = call(&r, "POST", "/fv/v1/jobs", Some(body)).await;
    assert_eq!(s, StatusCode::ACCEPTED, "{v}");
    let done = wait_done(&r, v["id"].as_str().unwrap()).await;
    assert_eq!(done["status"], "succeeded", "{done}");
    assert_eq!(done["expanded_prompt"], "integrated_multimodal_description: Rewritten: a cat", "H3: Context-IR layout");
    {
        let calls = seen.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0.as_deref(), Some("Bearer llm-secret-key"));
        assert_eq!(calls[0].1["model"], "fake-llm");
        assert!(calls[0].1["messages"][0]["content"].as_str().unwrap().contains("120 to 220 words"));
    }
    // No flag: nothing rewritten, no call.
    let body = json!({"model": "h3-turbo", "prompt": "a dog", "aspect_ratio": "16:9", "short_edge": 768});
    let (_, v) = call(&r, "POST", "/fv/v1/jobs", Some(body)).await;
    let done = wait_done(&r, v["id"].as_str().unwrap()).await;
    assert!(done.get("expanded_prompt").is_none(), "{done}");
    assert_eq!(seen.lock().unwrap().len(), 1);
    // A bad value is refused up front.
    let body = json!({"model": "h3-turbo", "prompt": "a dog", "aspect_ratio": "16:9", "short_edge": 768, "prompt_expansion": "max"});
    let (s, v) = call(&r, "POST", "/fv/v1/jobs", Some(body)).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["param"], "prompt_expansion");

    // MiniMax: V1's `prompt_optimizer: true` opts in.
    let body = json!({"model": "MiniMax-H3-Turbo", "content": [{"type": "text", "text": "a fox"}], "resolution": "768P", "duration": 5, "ratio": "16:9", "prompt_optimizer": true});
    let (s, v) = call(&r, "POST", "/v2/video_generation", Some(body)).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let task = v["task_id"].as_str().unwrap().to_owned();
    let job = app.ctx.jobs().by_external(ProtocolId::MiniMaxV2, &task).await.unwrap();
    assert_eq!(job.expanded_prompt.as_deref(), Some("integrated_multimodal_description: Rewritten: a fox"));
    assert_eq!(job.resolved.prompt, "integrated_multimodal_description: Rewritten: a fox");
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[test]
fn config_validation() {
    let env = |kv: &[(&str, &str)]| kv.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect::<BTreeMap<_, _>>();
    let mut c = Config::default();
    c.apply_env(&env(&[])).unwrap();
    assert!(c.upsampler.is_none(), "off unless configured");
    // TOML, with the key from the named variable.
    let mut c = Config::from_toml(
        "[upsampler]\nbase_url = \"https://llm.example/v1\"\nmodel = \"m\"\napi_key_env = \"MY_LLM_KEY\"\ndefault_on = true\n[upsampler.max_chars]\nwan = 1500\n",
        "t",
    )
    .unwrap();
    c.apply_env(&env(&[("MY_LLM_KEY", "k1")])).unwrap();
    c.validate().unwrap();
    let u = c.upsampler.as_ref().unwrap();
    assert_eq!((u.api_key.expose(), u.default_on, u.max_chars.wan, u.max_chars.ltx), ("k1", true, 1500, 3600));
    let s = u.settings();
    assert_eq!((s.max_chars.wan, s.max_chars.h3, s.memory, s.timeout), (1500, 3200, 12, Duration::from_secs(10)));
    // Refusals.
    for (toml, want) in [
        ("[upsampler]\nbase_url = \"ftp://x\"\nmodel = \"m\"\n", "http(s)"),
        ("[upsampler]\nbase_url = \"https://x/v1\"\n", "model is required"),
        ("[upsampler]\nbase_url = \"https://x/v1\"\nmodel = \"m\"\nmemory = 0\n", "memory"),
        ("[upsampler]\nbase_url = \"https://x/v1\"\nmodel = \"m\"\ntimeout_ms = 5\n", "timeout_ms"),
        ("[upsampler]\nbase_url = \"https://x/v1\"\nmodel = \"m\"\n[upsampler.max_chars]\nh3 = 10\n", "max_chars"),
    ] {
        let c = Config::from_toml(toml, "t").unwrap();
        let e = c.validate().unwrap_err().to_string();
        assert!(e.contains(want), "{toml}: {e}");
    }
    assert!(Config::from_toml("[upsampler]\nbogus = 1\n", "t").is_err());
}
