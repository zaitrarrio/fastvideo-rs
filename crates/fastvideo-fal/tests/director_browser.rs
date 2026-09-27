//! Browser compat (design §5.6, §7.5): the real `@fal-ai/client`
//! `fal.realtime.open(wma(...))` in headless Chromium against the director
//! and the fake engine, over real WebRTC.
//!
//! Needs:
//! - `FV_FAL_JS_DIR`: a directory whose `node_modules` holds
//!   `@fal-ai/client@alpha` (the realtime API ships there) and `esbuild`;
//! - Playwright (`FV_PLAYWRIGHT`, else the global
//!   `/opt/node22/lib/node_modules/playwright`) with Chromium under
//!   `PLAYWRIGHT_BROWSERS_PATH` (never `playwright install` here).
//!
//! Playwright's Chromium is built without H.264, so its offers carry
//! VP8/VP9/AV1 only and the director answers intra-only VP8 (libwebp,
//! `vp8_fallback`). Real Chrome offers H.264 and gets it.
//!
//! Skipped when `FV_FAL_JS_DIR` is unset.
//!
//! ```text
//! (mkdir -p target/fal-js-alpha && cd target/fal-js-alpha && npm init -y && npm i @fal-ai/client@alpha esbuild)
//! FV_FAL_JS_DIR=$PWD/target/fal-js-alpha cargo test -p fastvideo-fal --features director \
//!   --config 'profile.dev.package.libwebp-sys.opt-level=3' --test director_browser -- --nocapture
//! ```
#![cfg(feature = "director")]

mod director_common;

use std::time::Duration;

use director_common::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fal_realtime_open_in_chromium() {
    let Some(dir) = std::env::var_os("FV_FAL_JS_DIR").filter(|p| !p.is_empty()) else {
        eprintln!("skipped: set FV_FAL_JS_DIR to a directory with node_modules/@fal-ai/client@alpha and esbuild");
        return;
    };
    let h264 = h264_backend().unwrap_or(fastvideo_media::video::EncoderBackend::Nvenc);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let base = format!("http://127.0.0.1:{port}");
    let f = fixture(Opts { lan: true, public_base: base.clone(), h264, ..Opts::default() }).await;
    let app = f.app.clone();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let node = std::env::var_os("FV_NODE").unwrap_or_else(|| "node".into());
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/director_browser/wma_compat.mjs");
    let out = tokio::time::timeout(
        Duration::from_secs(600),
        tokio::process::Command::new(node).arg(&script).arg(&dir).args([&base, KEY, "fv/h3-silent/director"]).output(),
    )
    .await
    .expect("compat script timed out")
    .expect("node runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "fal.realtime.open compat failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
    eprintln!("fal.realtime.open compat: {stdout}");
    let summary: serde_json::Value = serde_json::from_str(stdout.lines().last().unwrap_or("{}")).unwrap();
    assert_eq!(summary["checks"].as_array().map(Vec::len), Some(3), "{summary}");
    let r = &summary["results"];
    // A/V: 24 fps video (VP8 in this Chromium) and 48 kHz stereo Opus.
    assert_eq!(r["middleware"]["video"]["codec"], "video/VP8", "{summary}");
    assert_eq!((r["middleware"]["video"]["width"].as_u64(), r["middleware"]["video"]["height"].as_u64()), (Some(832), Some(480)));
    assert_eq!(r["middleware"]["audio"]["channels"], 2, "{summary}");
    assert!(r["middleware"]["aliveAfterMs"].as_u64().unwrap() >= 17_000);
    for s in ["middleware", "proxy", "videoOnly"] {
        let fps = r[s]["video"]["decodedFps"].as_f64().unwrap();
        assert!((fps - 24.0).abs() < 2.5, "{s}: {fps} fps");
    }
    let sps = r["middleware"]["audio"]["samplesPerSecond"].as_f64().unwrap();
    assert!((sps - 48_000.0).abs() < 4_800.0, "{sps} samples/s");
}
