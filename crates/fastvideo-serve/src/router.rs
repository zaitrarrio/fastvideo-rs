//! Router assembly and the route table of design §9 (collision check owned
//! by WP-10).
//!
//! [`route_table`] lists every route each owner mounts (including the
//! adapters still being built, from design §4), with fal apps expanded as
//! static prefixes. [`check_route_table`] rejects two owners on one
//! (method, path) and any pair of templates the router cannot tell apart;
//! the tests also check that sample requests dispatch to the §9 owner.

use std::collections::BTreeMap;

use axum::Router;

/// Who mounts a route (design §9 "Owner").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Owner {
    Serve,
    ServeKit,
    Native,
    OpenAiVideos,
    MiniMax,
    Ltx,
    Fal,
    /// fal `minimax/h3-max/director` (WP-14).
    FalDirector,
    Reactor,
    /// The `/console` pages (WP-20).
    Console,
    /// Gateway mode and the worker role (docs/serve/gateway.md).
    Gateway,
}

impl Owner {
    pub fn as_str(&self) -> &'static str {
        match self {
            Owner::Serve => "serve",
            Owner::ServeKit => "serve-kit",
            Owner::Native => "native",
            Owner::OpenAiVideos => "openai-videos",
            Owner::MiniMax => "minimax",
            Owner::Ltx => "ltxapi",
            Owner::Fal => "fal",
            Owner::FalDirector => "fal-director",
            Owner::Reactor => "reactor",
            Owner::Console => "console",
            Owner::Gateway => "gateway",
        }
    }
}

/// One route: method, axum path template, owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteSpec {
    pub owner: Owner,
    pub method: &'static str,
    pub path: String,
}

fn r(owner: Owner, method: &'static str, path: impl Into<String>) -> RouteSpec {
    RouteSpec { owner, method, path: path.into() }
}

/// LTX endpoints with no engine path (design §4.5): 403 stubs.
pub const LTX_STUBS: &[&str] = &["retake", "extend", "video-to-video-hdr", "video-to-video-reframe"];
/// LTX generation endpoints (`/v1/*` sync, `/v2/*` async).
pub const LTX_GENERATE: &[&str] = &["text-to-video", "image-to-video", "audio-to-video"];
/// fal submit sub-paths of the H3 apps (design §4.4).
pub const FAL_SUBS: &[&str] = &["text-to-video", "image-to-video", "reference-to-video"];

/// An app's submit sub-paths (several segments on `lightricks/ltx-2.5` and
/// `fal-ai/wan`) and whether it has the director.
fn fal_app_subs(app: &str) -> (Vec<&'static str>, bool) {
    #[cfg(feature = "fal")]
    {
        let a = fastvideo_fal::FalApp::from_id(app);
        (a.endpoints().iter().map(|e| e.sub()).collect(), a.kind().director())
    }
    #[cfg(not(feature = "fal"))]
    {
        let _ = app;
        (FAL_SUBS.to_vec(), true)
    }
}

/// Every route of design §9, with `fal_apps` as static prefixes.
pub fn route_table(fal_apps: &[String]) -> Vec<RouteSpec> {
    use Owner::*;
    let mut v = vec![
        r(Serve, "GET", "/health"),
        r(Serve, "GET", "/"),
        r(Serve, "GET", "/healthz"),
        r(Serve, "GET", "/ping"),
        r(Serve, "GET", "/metrics"),
        // Public status view (single server and gateway, src/status.rs).
        r(Serve, "GET", "/fv/v1/status"),
        r(ServeKit, "GET", "/files/{artifact}/{name}"),
        r(ServeKit, "PUT", "/uploads/{token}"),
        // FastVideo /v1/videos family and FastWan (§4.1-4.2).
        r(OpenAiVideos, "POST", "/v1/videos"),
        r(OpenAiVideos, "GET", "/v1/videos"),
        r(OpenAiVideos, "POST", "/v1/videos/generations"),
        r(OpenAiVideos, "POST", "/v1/videos/sync"),
        r(OpenAiVideos, "GET", "/v1/videos/{id}"),
        r(OpenAiVideos, "DELETE", "/v1/videos/{id}"),
        r(OpenAiVideos, "GET", "/v1/videos/{id}/content"),
        r(OpenAiVideos, "GET", "/v1/models"),
        r(OpenAiVideos, "GET", "/v1/models/{model}"),
        r(OpenAiVideos, "GET", "/v1/model_info"),
        r(OpenAiVideos, "POST", "/generate"),
        r(OpenAiVideos, "GET", "/status/{prompt_id}"),
        r(OpenAiVideos, "GET", "/video/{prompt_id}"),
        r(OpenAiVideos, "DELETE", "/video/{prompt_id}"),
        // MiniMax V2 (§4.3).
        r(MiniMax, "POST", "/v2/video_generation"),
        r(MiniMax, "DELETE", "/v2/video_generation/{task_id}"),
        r(MiniMax, "GET", "/v2/query/video_generation"),
        r(MiniMax, "GET", "/v2/query/video_generation/{task_id}"),
        r(MiniMax, "POST", "/v2/h3_context_ir"),
        r(MiniMax, "POST", "/v2/video_regeneration"),
        // LTX (§4.5).
        r(Ltx, "POST", "/v1/upload"),
        // Native (§2.1).
        r(Native, "GET", "/fv/v1/capabilities"),
        r(Native, "POST", "/fv/v1/jobs"),
        r(Native, "GET", "/fv/v1/jobs"),
        r(Native, "GET", "/fv/v1/jobs/{id}"),
        r(Native, "DELETE", "/fv/v1/jobs/{id}"),
        r(Native, "GET", "/fv/v1/jobs/{id}/content"),
        r(Native, "GET", "/fv/v1/streams"),
        r(Native, "POST", "/fv/v1/streams"),
        // Minted API keys (serve-kit `admin_routes`, admin token).
        r(Native, "POST", "/fv/v1/admin/keys"),
        r(Native, "GET", "/fv/v1/admin/keys"),
        r(Native, "DELETE", "/fv/v1/admin/keys/{id}"),
        // Console pages (src/console.rs).
        r(Console, "GET", "/console"),
        r(Console, "GET", "/console/admin"),
        r(Console, "GET", "/console/deployments"),
        r(Console, "GET", "/console/models/{owner}/{alias}/{*task}"),
        r(Console, "GET", "/console/assets/{file}"),
        r(Native, "GET", "/fv/v1/streams/{id}"),
        r(Native, "DELETE", "/fv/v1/streams/{id}"),
        r(Native, "POST", "/fv/v1/streams/{id}/commands"),
        // Gateway pool metrics (admin token) and the worker role's internal
        // routes (docs/serve/gateway.md).
        r(Gateway, "GET", "/fv/v1/gateway/pools"),
        // Releases and deployments (admin token; src/releases.rs).
        r(Gateway, "GET", "/fv/v1/admin/releases"),
        r(Gateway, "GET", "/fv/v1/admin/deployments"),
        r(Gateway, "POST", "/fv/v1/admin/releases/promote"),
        r(Gateway, "POST", "/fv/v1/admin/releases/rollback"),
        r(Gateway, "POST", "/fv/v1/internal/jobs"),
        r(Gateway, "GET", "/fv/v1/internal/jobs/{id}"),
        r(Gateway, "DELETE", "/fv/v1/internal/jobs/{id}"),
        r(Gateway, "GET", "/fv/v1/internal/status"),
        r(Gateway, "POST", "/fv/v1/internal/drain"),
        r(Gateway, "POST", "/fv/v1/internal/undrain"),
        // fal shared routes (§4.4, §5.6).
        r(Fal, "POST", "/fal/proxy"),
        r(Fal, "GET", "/fal/proxy"),
        r(Fal, "POST", "/storage/upload/initiate"),
        r(Fal, "GET", "/fal/schema"),
        r(Fal, "GET", "/fal/schema/{owner}/{alias}/{*sub}"),
        r(FalDirector, "POST", "/wma/ice"),
        r(FalDirector, "POST", "/wma/session"),
        r(FalDirector, "POST", "/wma/session/heartbeat"),
        r(FalDirector, "POST", "/start-session"),
        r(FalDirector, "GET", "/info"),
        r(FalDirector, "POST", "/info"),
        r(Fal, "GET", "/.well-known/jwks.json"),
        // Reactor local runtime (§5.7).
        r(Reactor, "POST", "/start_session"),
        r(Reactor, "GET", "/session"),
        r(Reactor, "POST", "/stop_session"),
        r(Reactor, "GET", "/schema"),
        r(Reactor, "GET", "/events"),
        r(Reactor, "GET", "/sessions/{sid}/transport/webrtc/ice_servers"),
        r(Reactor, "POST", "/sessions/{sid}/transport/webrtc/connections"),
        r(Reactor, "POST", "/sessions/{sid}/transport/webrtc/connections/{cid}/sdp_params"),
        r(Reactor, "PUT", "/sessions/{sid}/transport/webrtc/connections/{cid}/sdp_params"),
        r(Reactor, "GET", "/sessions/{sid}/transport/webrtc/connections/{cid}/sdp_params"),
        r(Reactor, "POST", "/sessions/{sid}/transport/webrtc/connections/{cid}/ice_candidates"),
    ];
    for ep in LTX_GENERATE {
        v.push(r(Ltx, "POST", format!("/v2/{ep}")));
        v.push(r(Ltx, "GET", format!("/v2/{ep}/{{id}}")));
        v.push(r(Ltx, "POST", format!("/v1/{ep}")));
    }
    for ep in LTX_STUBS {
        v.push(r(Ltx, "POST", format!("/v1/{ep}")));
        v.push(r(Ltx, "POST", format!("/v2/{ep}")));
        v.push(r(Ltx, "GET", format!("/v2/{ep}/{{id}}")));
    }
    for app in fal_apps {
        let app = app.trim_matches('/');
        let (subs, director) = fal_app_subs(app);
        for sub in &subs {
            v.push(r(Fal, "POST", format!("/{app}/{sub}")));
            v.push(r(Fal, "POST", format!("/run/{app}/{sub}")));
        }
        // The director's app-local ICE fallback (`context.run`), direct and
        // through `/run` (what `/fal/proxy` maps `fal.run` to).
        if director {
            v.push(r(FalDirector, "POST", format!("/{app}/director/ice")));
            v.push(r(FalDirector, "POST", format!("/run/{app}/director/ice")));
        }
        let mut prefixes = vec![format!("/{app}")];
        prefixes.extend(subs.iter().map(|s| format!("/{app}/{s}")));
        for p in prefixes {
            v.push(r(Fal, "GET", format!("{p}/requests/{{id}}")));
            v.push(r(Fal, "GET", format!("{p}/requests/{{id}}/response")));
            v.push(r(Fal, "GET", format!("{p}/requests/{{id}}/status")));
            v.push(r(Fal, "GET", format!("{p}/requests/{{id}}/status/stream")));
            v.push(r(Fal, "PUT", format!("{p}/requests/{{id}}/cancel")));
        }
    }
    v
}

/// Refuses duplicates and templates the router cannot distinguish.
pub fn check_route_table(routes: &[RouteSpec]) -> Result<(), String> {
    let mut by_path: BTreeMap<&str, BTreeMap<&str, Owner>> = BTreeMap::new();
    for s in routes {
        let m = by_path.entry(s.path.as_str()).or_default();
        if let Some(prev) = m.insert(s.method, s.owner) {
            return Err(format!(
                "{} {} is claimed by both {} and {}",
                s.method,
                s.path,
                prev.as_str(),
                s.owner.as_str()
            ));
        }
    }
    // Two templates differing only in parameter names are one route to the
    // router: normalize the names and look for clashes.
    let mut shapes: BTreeMap<String, &str> = BTreeMap::new();
    for p in by_path.keys() {
        let shape: String = p
            .split('/')
            .map(|seg| if seg.starts_with("{*") { "{*}" } else if seg.starts_with('{') { "{}" } else { seg })
            .collect::<Vec<_>>()
            .join("/");
        if let Some(other) = shapes.insert(shape, p) {
            return Err(format!("`{p}` and `{other}` are the same route"));
        }
    }
    // Finally let axum (matchit) insert them all.
    let res = std::panic::catch_unwind(|| {
        let mut router: Router = Router::new();
        for (p, methods) in &by_path {
            let mut mr = axum::routing::MethodRouter::<()>::new();
            for m in methods.keys() {
                let f = axum::routing::MethodFilter::try_from(
                    axum::http::Method::from_bytes(m.as_bytes()).expect("method"),
                )
                .expect("filter");
                mr = mr.on(f, || async {});
            }
            router = router.route(p, mr);
        }
        router
    });
    res.map(|_| ()).map_err(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "route insertion panicked".into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    /// The default apps plus fal's LTX-2.5 and Wan apps (multi-segment subs).
    fn apps() -> Vec<String> {
        let mut v = crate::config::ProtocolsCfg::default().fal_apps;
        v.extend(["lightricks/ltx-2.5".to_owned(), "fal-ai/wan".to_owned(), "fastvideo/ltx-turbo".to_owned()]);
        v
    }

    #[test]
    fn design_route_table_has_no_collisions() {
        let t = route_table(&apps());
        check_route_table(&t).unwrap();
        assert!(t.len() > 120, "{}", t.len());
    }

    #[test]
    fn collisions_are_detected() {
        let mut t = route_table(&apps());
        t.push(r(Owner::Ltx, "GET", "/v1/videos/{id}"));
        assert!(check_route_table(&t).unwrap_err().contains("openai-videos"));
        let mut t = route_table(&apps());
        t.push(r(Owner::Fal, "GET", "/v1/videos/{video}"));
        assert!(check_route_table(&t).unwrap_err().contains("same route"));
        // A wildcard fal app (never allowed, §4.4) collides with /v1/videos.
        let mut t = route_table(&apps());
        t.push(r(Owner::Fal, "POST", "/{owner}/{app}"));
        t.push(r(Owner::Fal, "POST", "/{a}/{b}"));
        assert!(check_route_table(&t).is_err());
    }

    /// Requests land on the §9 owner: a router with one tagging handler per
    /// table entry.
    #[tokio::test]
    async fn samples_dispatch_to_their_owner() {
        let table = route_table(&apps());
        let mut by_path: BTreeMap<String, Vec<(&'static str, Owner)>> = BTreeMap::new();
        for s in &table {
            by_path.entry(s.path.clone()).or_default().push((s.method, s.owner));
        }
        let mut router: Router = Router::new();
        for (p, ms) in by_path {
            let mut mr = axum::routing::MethodRouter::<()>::new();
            for (m, o) in ms {
                let f = axum::routing::MethodFilter::try_from(axum::http::Method::from_bytes(m.as_bytes()).unwrap()).unwrap();
                mr = mr.on(f, move || async move { o.as_str() });
            }
            router = router.route(&p, mr);
        }
        let cases = [
            ("GET", "/health", "serve"),
            ("GET", "/ping", "serve"),
            ("POST", "/v1/videos/sync", "openai-videos"),
            ("GET", "/v1/videos/video_gen_abc/content", "openai-videos"),
            ("POST", "/v1/text-to-video", "ltxapi"),
            ("POST", "/v1/upload", "ltxapi"),
            ("GET", "/v2/text-to-video/0f1e", "ltxapi"),
            ("POST", "/v2/retake", "ltxapi"),
            ("GET", "/v2/query/video_generation", "minimax"),
            ("GET", "/v2/query/video_generation/123456789012345678", "minimax"),
            ("DELETE", "/v2/video_generation/123", "minimax"),
            ("POST", "/minimax/h3-max/text-to-video", "fal"),
            ("POST", "/minimax/h3-turbo/reference-to-video", "fal"),
            ("GET", "/minimax/h3-max/requests/abc/status", "fal"),
            ("GET", "/minimax/h3-max/image-to-video/requests/abc/response", "fal"),
            ("PUT", "/minimax/h3-draft/requests/abc/cancel", "fal"),
            ("POST", "/run/minimax/h3-max/text-to-video", "fal"),
            ("POST", "/wma/session", "fal-director"),
            ("POST", "/wma/ice", "fal-director"),
            ("POST", "/wma/session/heartbeat", "fal-director"),
            ("POST", "/minimax/h3-max/director/ice", "fal-director"),
            ("POST", "/run/minimax/h3-turbo/director/ice", "fal-director"),
            ("POST", "/start-session", "fal-director"),
            ("POST", "/info", "fal-director"),
            ("GET", "/session", "reactor"),
            ("POST", "/start_session", "reactor"),
            ("GET", "/sessions/00000000-0000-0000-0000-000000000000/transport/webrtc/ice_servers", "reactor"),
            ("PUT", "/sessions/s1/transport/webrtc/connections/1002/sdp_params", "reactor"),
            ("POST", "/sessions/s1/transport/webrtc/connections/1002/ice_candidates", "reactor"),
            ("GET", "/status/abc", "openai-videos"),
            ("GET", "/files/a/b.mp4", "serve-kit"),
            ("GET", "/fv/v1/jobs/fvjob_1", "native"),
            ("DELETE", "/fv/v1/admin/keys/key_abc", "native"),
            ("GET", "/console/models/minimax/h3-max/text-to-video", "console"),
            ("GET", "/console/assets/model.js", "console"),
            ("GET", "/fal/schema/minimax/h3-max/image-to-video", "fal"),
            ("POST", "/minimax/h3-max-turbo/text-to-video", "fal"),
            ("POST", "/minimax/h3/image-to-video", "fal"),
            ("GET", "/minimax/h3/requests/abc/status", "fal"),
            ("POST", "/minimax/h3/director/ice", "fal-director"),
            ("POST", "/lightricks/ltx-2.5/text-to-video/fast", "fal"),
            ("POST", "/run/lightricks/ltx-2.5/image-to-video/pro", "fal"),
            ("GET", "/lightricks/ltx-2.5/requests/abc", "fal"),
            ("GET", "/lightricks/ltx-2.5/text-to-video/fast/requests/abc/status", "fal"),
            ("POST", "/fal-ai/wan/v2.2-5b/text-to-video", "fal"),
            ("POST", "/fal-ai/wan/v2.2-5b/text-to-video/fast-wan", "fal"),
            ("POST", "/run/fal-ai/wan/v2.2-5b/image-to-video", "fal"),
            ("GET", "/fal-ai/wan/requests/abc/status/stream", "fal"),
            ("PUT", "/fal-ai/wan/v2.2-5b/text-to-video/fast-wan/requests/abc/cancel", "fal"),
            ("GET", "/fal/schema/fal-ai/wan/v2.2-5b/text-to-video/fast-wan", "fal"),
            ("GET", "/console/models/lightricks/ltx-2.5/text-to-video/fast", "console"),
        ];
        for (m, uri, want) in cases {
            let resp = router
                .clone()
                .oneshot(Request::builder().method(m).uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(resp.status(), 200, "{m} {uri}");
            let b = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
            assert_eq!(std::str::from_utf8(&b).unwrap(), want, "{m} {uri}");
        }
        // Unknown fal apps and LTX endpoint segments are not routed.
        for (m, uri) in [
            ("POST", "/minimax/h9/text-to-video"),
            ("GET", "/v2/text-to-image/abc"),
            ("POST", "/v2/text-to-image"),
            // The family apps have only their own subs, and no director.
            ("POST", "/lightricks/ltx-2.5/text-to-video"),
            ("POST", "/fal-ai/wan/reference-to-video"),
            ("POST", "/lightricks/ltx-2.5/director/ice"),
        ] {
            let resp = router
                .clone()
                .oneshot(Request::builder().method(m).uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert!(resp.status() == 404 || resp.status() == 405, "{m} {uri}: {}", resp.status());
        }
    }
}
