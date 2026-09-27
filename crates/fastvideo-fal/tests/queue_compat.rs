//! Client compat (design §7.5): the real fal clients against the fal router
//! and the fake engine.
//!
//! - Python `fal-client` through `FAL_QUEUE_RUN_HOST` / `FAL_RUN_HOST`: needs
//!   a Python with `fal-client` named by `FV_FAL_PYTHON`, plus `openssl` for
//!   a throwaway certificate. `fal_client` is https-only, so the script
//!   terminates TLS locally and forwards to this plain-HTTP server; the
//!   server's `public_base` is the TLS address so the URLs it hands out are
//!   reachable by the client.
//! - JS `@fal-ai/client` with `requestMiddleware` and with `proxyUrl`
//!   (`/fal/proxy`): needs `node` (or `FV_NODE`) and `FV_FAL_JS_DIR`, a
//!   directory whose `node_modules` holds `@fal-ai/client`.
//!
//! Each test is skipped when its variable is unset. Setting FV_FFMPEG (or
//! having ffmpeg on PATH) makes the fake engine write real MP4s.
//!
//! ```text
//! python3 -m venv target/fal-venv
//! target/fal-venv/bin/pip install fal-client==1.0.3
//! (mkdir -p target/fal-js && cd target/fal-js && npm init -y && npm i @fal-ai/client@1.10.1)
//! FV_FAL_PYTHON=$PWD/target/fal-venv/bin/python FV_FAL_JS_DIR=$PWD/target/fal-js \
//!   cargo test -p fastvideo-fal --test queue_compat
//! ```

mod queue_common;

use std::time::Duration;

use queue_common::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn python_fal_client() {
    let Some(python) = std::env::var_os("FV_FAL_PYTHON").filter(|p| !p.is_empty()) else {
        eprintln!("skipped: set FV_FAL_PYTHON to a python with fal-client installed");
        return;
    };
    // A free port for the TLS side; the server must know it before it starts.
    let tls_port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let f = fixture(Opts {
        engine: Engine::Fake,
        public_base: format!("https://127.0.0.1:{tls_port}"),
        step: Duration::from_millis(60),
        mp4: true,
        ..Opts::default()
    })
    .await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = listener.local_addr().unwrap().port();
    let app = f.app.clone();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/queue_compat/fal_client_compat.py");
    let out = tokio::time::timeout(
        Duration::from_secs(600),
        tokio::process::Command::new(&python)
            .arg(&script)
            .args(["--upstream", &upstream.to_string(), "--tls-port", &tls_port.to_string(), "--key", KEY])
            .output(),
    )
    .await
    .expect("compat script timed out")
    .expect("python runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "fal-client compat failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
    eprintln!("fal-client compat: {stdout}");
    let summary: serde_json::Value = serde_json::from_str(stdout.lines().last().unwrap_or("{}")).unwrap();
    assert!(summary["checks"].as_array().is_some_and(|c| c.len() >= 10), "{summary}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn js_fal_client() {
    let Some(dir) = std::env::var_os("FV_FAL_JS_DIR").filter(|p| !p.is_empty()) else {
        eprintln!("skipped: set FV_FAL_JS_DIR to a directory with node_modules/@fal-ai/client");
        return;
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let base = format!("http://127.0.0.1:{port}");
    let f = fixture(Opts { engine: Engine::Fake, public_base: base.clone(), step: Duration::from_millis(60), ..Opts::default() }).await;
    let app = f.app.clone();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let png = f.dir.join("in.png");
    image::RgbImage::from_pixel(96, 128, image::Rgb([200, 40, 40])).save(&png).unwrap();

    let node = std::env::var_os("FV_NODE").unwrap_or_else(|| "node".into());
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/queue_compat/fal_js_compat.mjs");
    let out = tokio::time::timeout(
        Duration::from_secs(900),
        tokio::process::Command::new(node).arg(&script).arg(&dir).arg(&base).arg(KEY).arg(&png).output(),
    )
    .await
    .expect("compat script timed out")
    .expect("node runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "@fal-ai/client compat failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
    eprintln!("@fal-ai/client compat: {stdout}");
    let summary: serde_json::Value = serde_json::from_str(stdout.lines().last().unwrap_or("{}")).unwrap();
    assert_eq!(summary["checks"].as_array().map(Vec::len), Some(12), "{summary}");
}
