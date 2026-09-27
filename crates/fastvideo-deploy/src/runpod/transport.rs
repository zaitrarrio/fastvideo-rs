//! HTTP for the worker: a tiny request/response seam so the loop runs on
//! reqwest in production and on an in-process router (the simulator) in
//! tests.

use std::time::Duration;

use axum::body::Body;
use axum::Router;
use tower::ServiceExt;

/// The two methods the queue protocol uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
}

/// An outbound request.
#[derive(Clone, Debug)]
pub struct OutReq {
    pub method: Method,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
    pub timeout: Duration,
}

/// Its answer.
#[derive(Clone, Debug)]
pub struct InResp {
    pub status: u16,
    pub body: Vec<u8>,
}

/// Sends one request. `Err` is a transport failure (connect, timeout).
#[async_trait::async_trait]
pub trait Transport: Send + Sync + 'static {
    async fn send(&self, req: OutReq) -> Result<InResp, String>;
}

/// Drives an axum router in-process: the URL's host is ignored, its path and
/// query are routed. Used with [`super::sim::Sim`].
#[derive(Clone, Debug)]
pub struct RouterTransport(pub Router);

#[async_trait::async_trait]
impl Transport for RouterTransport {
    async fn send(&self, req: OutReq) -> Result<InResp, String> {
        let u = url::Url::parse(&req.url).map_err(|e| format!("bad url {}: {e}", req.url))?;
        let pq = match u.query() {
            Some(q) => format!("{}?{q}", u.path()),
            None => u.path().to_owned(),
        };
        let mut b = axum::http::Request::builder()
            .method(match req.method {
                Method::Get => axum::http::Method::GET,
                Method::Post => axum::http::Method::POST,
            })
            .uri(pq);
        for (k, v) in &req.headers {
            b = b.header(k.as_str(), v.as_str());
        }
        let r = b.body(req.body.map(Body::from).unwrap_or_else(Body::empty)).map_err(|e| e.to_string())?;
        let fut = async {
            let resp = self.0.clone().oneshot(r).await.map_err(|e| e.to_string())?;
            let status = resp.status().as_u16();
            let body = axum::body::to_bytes(resp.into_body(), 64 << 20).await.map_err(|e| e.to_string())?;
            Ok(InResp { status, body: body.to_vec() })
        };
        tokio::time::timeout(req.timeout, fut).await.map_err(|_| "timed out".to_owned())?
    }
}

/// Production transport (feature `runpod`).
#[cfg(feature = "runpod")]
#[derive(Clone, Debug)]
pub struct ReqwestTransport(pub reqwest::Client);

#[cfg(feature = "runpod")]
impl ReqwestTransport {
    pub fn new() -> Result<Self, String> {
        reqwest::Client::builder()
            .user_agent(super::version())
            .connect_timeout(Duration::from_secs(20))
            .build()
            .map(Self)
            .map_err(|e| e.to_string())
    }
}

#[cfg(feature = "runpod")]
#[async_trait::async_trait]
impl Transport for ReqwestTransport {
    async fn send(&self, req: OutReq) -> Result<InResp, String> {
        let mut b = match req.method {
            Method::Get => self.0.get(&req.url),
            Method::Post => self.0.post(&req.url),
        };
        for (k, v) in &req.headers {
            b = b.header(k.as_str(), v.as_str());
        }
        if let Some(body) = req.body {
            b = b.body(body);
        }
        let resp = b.timeout(req.timeout).send().await.map_err(|e| e.without_url().to_string())?;
        let status = resp.status().as_u16();
        let body = resp.bytes().await.map_err(|e| e.without_url().to_string())?;
        Ok(InResp { status, body: body.to_vec() })
    }
}
