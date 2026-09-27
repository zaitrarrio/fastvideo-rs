//! Output storage and download URLs (design §4 "Common conventions", §6.2).
//!
//! - [`LocalArtifactStore`]: files under `<state_dir>/artifacts/<id>/<name>`,
//!   served by [`files_router`] at `GET /files/{artifact_id}/{file_name}?exp=&sig=`
//!   where `sig = hex(HMAC-SHA256(key, "artifact_id|file_name|exp"))`. The
//!   route is unauthenticated and answers `Access-Control-Allow-Origin: *`;
//!   the unguessable id plus the signature and expiry are the access control.
//! - [`S3ArtifactStore`]: S3-compatible object storage (Runpod S3 API, R2) for
//!   `runpod-queue`, with SigV4 query-presigned URLs. Uploading needs the
//!   `fetch` feature.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{header, HeaderValue, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use fastvideo_protocol::{ApiError, Artifact, ArtifactId, ArtifactLocation, UrlSigner};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use time::OffsetDateTime;
use tower_http::services::ServeFile;
use url::Url;

use crate::hex;

type HmacSha256 = Hmac<Sha256>;

/// The URL signing key (`FV_URL_SIGNING_KEY`).
#[derive(Clone)]
pub struct UrlKey(Arc<[u8]>);

impl std::fmt::Debug for UrlKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("UrlKey(..)")
    }
}

impl UrlKey {
    pub fn new(key: impl AsRef<[u8]>) -> Self {
        Self(Arc::from(key.as_ref()))
    }
    /// A random key (URLs die with the process).
    pub fn random() -> Self {
        Self::new(format!("{}{}", crate::random_token(), crate::random_token()))
    }

    /// `hex(HMAC-SHA256(key, "id|name|exp"))`.
    pub fn sign(&self, id: &str, name: &str, exp: i64) -> String {
        let mut m = HmacSha256::new_from_slice(&self.0).expect("hmac accepts any key");
        m.update(format!("{id}|{name}|{exp}").as_bytes());
        hex::encode(&m.finalize().into_bytes())
    }

    /// Checks `sig` (constant time) and that `exp` is not before `now`.
    pub fn verify(&self, id: &str, name: &str, exp: i64, sig: &str, now: OffsetDateTime) -> bool {
        if exp < now.unix_timestamp() {
            return false;
        }
        let Some(sig) = hex::decode(sig) else {
            return false;
        };
        let mut m = HmacSha256::new_from_slice(&self.0).expect("hmac accepts any key");
        m.update(format!("{id}|{name}|{exp}").as_bytes());
        m.verify_slice(&sig).is_ok()
    }
}

/// Builds signed `/files/...` URLs on the public base.
#[derive(Clone, Debug)]
pub struct LocalUrls {
    pub public_base: Url,
    pub key: UrlKey,
}

impl LocalUrls {
    /// `<base>/files/{id}/{name}?exp=<unix>&sig=<hex>`.
    pub fn files_url(&self, id: &str, name: &str, exp: i64) -> Url {
        let mut base = self.public_base.clone();
        if !base.path().ends_with('/') {
            let p = format!("{}/", base.path());
            base.set_path(&p);
        }
        let mut u = base.join("files/").expect("static path");
        u.path_segments_mut()
            .expect("http base")
            .pop_if_empty()
            .push(id)
            .push(name);
        u.query_pairs_mut()
            .append_pair("exp", &exp.to_string())
            .append_pair("sig", &self.key.sign(id, name, exp));
        u
    }
    /// A URL valid for `ttl` from now.
    pub fn files_url_ttl(&self, id: &str, name: &str, ttl: Duration) -> Url {
        let exp = OffsetDateTime::now_utc().unix_timestamp() + ttl.as_secs() as i64;
        self.files_url(id, name, exp)
    }
}

/// Facts about an output file, besides its bytes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ArtifactMeta {
    pub file_name: String,
    pub mime: String,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub audio: Option<(u32, u8)>,
}

/// Where outputs go.
#[async_trait::async_trait]
pub trait ArtifactStore: UrlSigner + Send + Sync + 'static {
    /// Takes ownership of `src` (moved or uploaded, then removed).
    async fn put(&self, src: &Path, meta: ArtifactMeta) -> Result<Artifact, ApiError>;
    /// Best-effort removal.
    async fn delete(&self, a: &Artifact);
    /// `self` as a [`UrlSigner`] (no trait upcasting on the MSRV).
    fn signer(&self) -> &dyn UrlSigner;
}

/// A file name safe as one URL/path segment.
pub fn valid_file_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 200
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

async fn move_file(src: &Path, dst: &Path) -> std::io::Result<()> {
    if tokio::fs::rename(src, dst).await.is_ok() {
        return Ok(());
    }
    tokio::fs::copy(src, dst).await?;
    let _ = tokio::fs::remove_file(src).await;
    Ok(())
}

/// Local-disk artifacts with HMAC-signed URLs.
#[derive(Clone, Debug)]
pub struct LocalArtifactStore {
    root: PathBuf,
    urls: LocalUrls,
}

impl LocalArtifactStore {
    pub fn new(root: impl Into<PathBuf>, urls: LocalUrls) -> Self {
        Self {
            root: root.into(),
            urls,
        }
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn urls(&self) -> &LocalUrls {
        &self.urls
    }
}

impl UrlSigner for LocalArtifactStore {
    fn url_for(&self, a: &Artifact, ttl: Duration) -> Url {
        self.urls.files_url_ttl(&a.id.to_string(), &a.file_name, ttl)
    }
}

#[async_trait::async_trait]
impl ArtifactStore for LocalArtifactStore {
    async fn put(&self, src: &Path, meta: ArtifactMeta) -> Result<Artifact, ApiError> {
        if !valid_file_name(&meta.file_name) {
            return Err(ApiError::internal(format!("bad artifact name `{}`", meta.file_name)));
        }
        let id = ArtifactId::new();
        let dir = self.root.join(id.to_string());
        let dst = dir.join(&meta.file_name);
        let io = |e: std::io::Error| ApiError::internal(format!("storing artifact: {e}"));
        tokio::fs::create_dir_all(&dir).await.map_err(io)?;
        move_file(src, &dst).await.map_err(io)?;
        let bytes = tokio::fs::metadata(&dst).await.map_err(io)?.len();
        Ok(Artifact {
            id,
            mime: meta.mime,
            file_name: meta.file_name,
            bytes,
            location: ArtifactLocation::Local(dst),
            width: meta.width,
            height: meta.height,
            frames: meta.frames,
            fps: meta.fps,
            audio: meta.audio,
        })
    }

    async fn delete(&self, a: &Artifact) {
        let _ = tokio::fs::remove_dir_all(self.root.join(a.id.to_string())).await;
    }

    fn signer(&self) -> &dyn UrlSigner {
        self
    }
}

// ---------------------------------------------------------------- /files route

/// Serves `GET /files/{id}/{name}` from `<root>/<id>/<name>` in each root
/// (artifacts first, then uploads).
#[derive(Clone, Debug)]
pub struct FileServer {
    pub key: UrlKey,
    pub roots: Vec<PathBuf>,
}

#[derive(serde::Deserialize)]
struct SigQuery {
    exp: Option<i64>,
    sig: Option<String>,
}

/// The `/files/{artifact}/{name}` route (design §9, owned by serve-kit).
pub fn files_router<S: Clone + Send + Sync + 'static>(server: Arc<FileServer>) -> Router<S> {
    Router::new()
        .route("/files/{id}/{name}", get(serve_file).head(serve_file))
        .with_state(server)
}

async fn serve_file(
    State(fs): State<Arc<FileServer>>,
    UrlPath((id, name)): UrlPath<(String, String)>,
    Query(q): Query<SigQuery>,
    req: Request<axum::body::Body>,
) -> Response {
    let not_found = || {
        let mut r = (StatusCode::NOT_FOUND, "not found").into_response();
        r.headers_mut()
            .insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
        r
    };
    let (Some(exp), Some(sig)) = (q.exp, q.sig) else {
        return not_found();
    };
    if !valid_file_name(&name)
        || !valid_file_name(&id)
        || !fs.key.verify(&id, &name, exp, &sig, OffsetDateTime::now_utc())
    {
        return not_found();
    }
    for root in &fs.roots {
        let p = root.join(&id).join(&name);
        if tokio::fs::metadata(&p).await.is_ok_and(|m| m.is_file()) {
            let mut resp = match tower::ServiceExt::oneshot(ServeFile::new(&p), req).await {
                Ok(r) => r.map(axum::body::Body::new).into_response(),
                Err(e) => match e {},
            };
            let h = resp.headers_mut();
            h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
            h.insert(header::CACHE_CONTROL, HeaderValue::from_static("private, max-age=86400"));
            return resp;
        }
    }
    not_found()
}

// ---------------------------------------------------------------- S3

/// An S3-compatible bucket.
#[derive(Clone, PartialEq, Eq)]
pub struct S3Config {
    /// e.g. `https://s3api-eu-ro-1.runpod.io`, `https://<acct>.r2.cloudflarestorage.com`,
    /// `https://s3.amazonaws.com`.
    pub endpoint: Url,
    /// `us-east-1`, `auto` (R2), the Runpod data-center id, ...
    pub region: String,
    pub bucket: String,
    pub access_key: String,
    pub secret_key: String,
    /// `https://endpoint/bucket/key` (true) vs `https://bucket.endpoint/key`.
    pub path_style: bool,
    /// Key prefix, e.g. `fv-serve/outputs/`.
    pub prefix: String,
}

impl std::fmt::Debug for S3Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3Config")
            .field("endpoint", &self.endpoint.as_str())
            .field("region", &self.region)
            .field("bucket", &self.bucket)
            .field("path_style", &self.path_style)
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

/// S3 caps presigned URLs at 7 days.
pub const S3_MAX_PRESIGN: Duration = Duration::from_secs(7 * 24 * 3600);

fn uri_encode(s: &str, keep_slash: bool) -> String {
    let mut o = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => o.push(b as char),
            b'/' if keep_slash => o.push('/'),
            _ => o.push_str(&format!("%{b:02X}")),
        }
    }
    o
}

fn hmac(key: &[u8], msg: &str) -> Vec<u8> {
    let mut m = HmacSha256::new_from_slice(key).expect("hmac accepts any key");
    m.update(msg.as_bytes());
    m.finalize().into_bytes().to_vec()
}

impl S3Config {
    /// Scheme+host(+port) and path for `key`.
    fn locate(&self, key: &str) -> (Url, String) {
        let mut u = self.endpoint.clone();
        let ekey = uri_encode(key, true);
        let base_path = self.endpoint.path().trim_end_matches('/');
        let path = if self.path_style {
            format!("{base_path}/{}/{ekey}", uri_encode(&self.bucket, false))
        } else {
            let host = format!("{}.{}", self.bucket, self.endpoint.host_str().unwrap_or_default());
            let _ = u.set_host(Some(&host));
            format!("{base_path}/{ekey}")
        };
        (u, path)
    }

    /// A SigV4 query-presigned URL (`UNSIGNED-PAYLOAD`, signed header `host`).
    pub fn presign(&self, method: &str, key: &str, expires: Duration, now: OffsetDateTime) -> Url {
        let expires = expires.min(S3_MAX_PRESIGN).as_secs().max(1);
        let (mut u, path) = self.locate(key);
        let host = match u.port() {
            Some(p) => format!("{}:{p}", u.host_str().unwrap_or_default()),
            None => u.host_str().unwrap_or_default().to_owned(),
        };
        let fmt_d = time::macros::format_description!("[year][month][day]");
        let fmt_t = time::macros::format_description!("[year][month][day]T[hour][minute][second]Z");
        let now = now.to_offset(time::UtcOffset::UTC);
        let date = now.format(fmt_d).expect("date format");
        let ts = now.format(fmt_t).expect("time format");
        let scope = format!("{date}/{}/s3/aws4_request", self.region);
        let mut q: Vec<(String, String)> = vec![
            ("X-Amz-Algorithm".into(), "AWS4-HMAC-SHA256".into()),
            ("X-Amz-Credential".into(), format!("{}/{scope}", self.access_key)),
            ("X-Amz-Date".into(), ts.clone()),
            ("X-Amz-Expires".into(), expires.to_string()),
            ("X-Amz-SignedHeaders".into(), "host".into()),
        ];
        q.sort();
        let cq = q
            .iter()
            .map(|(k, v)| format!("{}={}", uri_encode(k, false), uri_encode(v, false)))
            .collect::<Vec<_>>()
            .join("&");
        let creq = format!("{method}\n{path}\n{cq}\nhost:{host}\n\nhost\nUNSIGNED-PAYLOAD");
        let sts = format!(
            "AWS4-HMAC-SHA256\n{ts}\n{scope}\n{}",
            hex::encode(&Sha256::digest(creq.as_bytes()))
        );
        let k = hmac(format!("AWS4{}", self.secret_key).as_bytes(), &date);
        let k = hmac(&k, &self.region);
        let k = hmac(&k, "s3");
        let k = hmac(&k, "aws4_request");
        let sig = hex::encode(&hmac(&k, &sts));
        u.set_path("");
        let full = format!(
            "{}://{host}{path}?{cq}&X-Amz-Signature={sig}",
            u.scheme()
        );
        Url::parse(&full).expect("well-formed presigned URL")
    }
}

/// S3-compatible artifacts with presigned GET URLs.
#[derive(Clone, Debug)]
pub struct S3ArtifactStore {
    cfg: S3Config,
    #[cfg(feature = "fetch")]
    http: reqwest::Client,
}

impl S3ArtifactStore {
    pub fn new(cfg: S3Config) -> Self {
        Self {
            cfg,
            #[cfg(feature = "fetch")]
            http: reqwest::Client::new(),
        }
    }
    pub fn config(&self) -> &S3Config {
        &self.cfg
    }
    fn key_for(&self, id: ArtifactId, name: &str) -> String {
        format!("{}{id}/{name}", self.cfg.prefix)
    }
}

impl UrlSigner for S3ArtifactStore {
    fn url_for(&self, a: &Artifact, ttl: Duration) -> Url {
        let key = match &a.location {
            ArtifactLocation::Object { key, .. } => key.clone(),
            ArtifactLocation::Local(_) => self.key_for(a.id, &a.file_name),
        };
        self.cfg.presign("GET", &key, ttl, OffsetDateTime::now_utc())
    }
}

#[async_trait::async_trait]
impl ArtifactStore for S3ArtifactStore {
    async fn put(&self, src: &Path, meta: ArtifactMeta) -> Result<Artifact, ApiError> {
        if !valid_file_name(&meta.file_name) {
            return Err(ApiError::internal(format!("bad artifact name `{}`", meta.file_name)));
        }
        let id = ArtifactId::new();
        let key = self.key_for(id, &meta.file_name);
        let bytes = self.upload(src, &key, &meta.mime).await?;
        let _ = tokio::fs::remove_file(src).await;
        Ok(Artifact {
            id,
            mime: meta.mime,
            file_name: meta.file_name,
            bytes,
            location: ArtifactLocation::Object {
                bucket: self.cfg.bucket.clone(),
                key,
            },
            width: meta.width,
            height: meta.height,
            frames: meta.frames,
            fps: meta.fps,
            audio: meta.audio,
        })
    }

    async fn delete(&self, a: &Artifact) {
        #[cfg(feature = "fetch")]
        if let ArtifactLocation::Object { key, .. } = &a.location {
            let u = self.cfg.presign("DELETE", key, Duration::from_secs(300), OffsetDateTime::now_utc());
            if let Err(e) = self.http.delete(u).send().await {
                tracing::warn!(error = %e, "S3 delete failed");
            }
        }
        #[cfg(not(feature = "fetch"))]
        let _ = a;
    }

    fn signer(&self) -> &dyn UrlSigner {
        self
    }
}

impl S3ArtifactStore {
    #[cfg(feature = "fetch")]
    async fn upload(&self, src: &Path, key: &str, mime: &str) -> Result<u64, ApiError> {
        let data = tokio::fs::read(src)
            .await
            .map_err(|e| ApiError::internal(format!("reading artifact: {e}")))?;
        let n = data.len() as u64;
        let u = self.cfg.presign("PUT", key, Duration::from_secs(900), OffsetDateTime::now_utc());
        let r = self
            .http
            .put(u)
            .header("content-type", mime)
            .body(data)
            .send()
            .await
            .map_err(|e| ApiError::internal(format!("S3 upload: {e}")))?;
        if !r.status().is_success() {
            return Err(ApiError::internal(format!("S3 upload: HTTP {}", r.status())));
        }
        Ok(n)
    }

    #[cfg(not(feature = "fetch"))]
    async fn upload(&self, _src: &Path, _key: &str, _mime: &str) -> Result<u64, ApiError> {
        Err(ApiError::internal(
            "S3 artifacts need fastvideo-serve-kit built with the `fetch` feature",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use tower::ServiceExt;

    fn t(s: &str) -> OffsetDateTime {
        OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339).unwrap()
    }

    #[test]
    fn hmac_sign_verify_expiry() {
        let k = UrlKey::new("secret");
        let now = t("2026-09-27T12:00:00Z");
        let exp = now.unix_timestamp() + 60;
        let sig = k.sign("id1", "a.mp4", exp);
        assert_eq!(sig.len(), 64);
        assert!(k.verify("id1", "a.mp4", exp, &sig, now));
        assert!(k.verify("id1", "a.mp4", exp, &sig, now + Duration::from_secs(60)));
        assert!(!k.verify("id1", "a.mp4", exp, &sig, now + Duration::from_secs(61)), "expired");
        assert!(!k.verify("id1", "b.mp4", exp, &sig, now), "other name");
        assert!(!k.verify("id2", "a.mp4", exp, &sig, now), "other id");
        assert!(!k.verify("id1", "a.mp4", exp + 1, &sig, now), "extended exp");
        assert!(!UrlKey::new("other").verify("id1", "a.mp4", exp, &sig, now), "other key");
        assert!(!k.verify("id1", "a.mp4", exp, "zz", now));
    }

    #[test]
    fn files_url_shape() {
        let urls = LocalUrls { public_base: Url::parse("https://h.example/base").unwrap(), key: UrlKey::new("k") };
        let u = urls.files_url("abc", "x.mp4", 100);
        assert_eq!(u.path(), "/base/files/abc/x.mp4");
        let q: Vec<(String, String)> = u.query_pairs().into_owned().collect();
        assert_eq!(q[0], ("exp".into(), "100".into()));
        assert_eq!(q[1].1, UrlKey::new("k").sign("abc", "x.mp4", 100));
    }

    #[test]
    fn names() {
        assert!(valid_file_name("V1StGXR8_Z5jdHi6B-myT_fasth3.mp4"));
        for bad in ["", "..", ".hidden", "a/b", "a\\b", "a b", "%2e"] {
            assert!(!valid_file_name(bad), "{bad}");
        }
    }

    /// The AWS SigV4 documentation example ("Example: presigned URL"):
    /// GET examplebucket/test.txt, 2013-05-24, 86400 s.
    #[test]
    fn sigv4_presign_matches_aws_example() {
        let cfg = S3Config {
            endpoint: Url::parse("https://s3.amazonaws.com").unwrap(),
            region: "us-east-1".into(),
            bucket: "examplebucket".into(),
            access_key: "AKIAIOSFODNN7EXAMPLE".into(),
            secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".into(),
            path_style: false,
            prefix: String::new(),
        };
        let u = cfg.presign("GET", "test.txt", Duration::from_secs(86400), t("2013-05-24T00:00:00Z"));
        assert_eq!(u.host_str(), Some("examplebucket.s3.amazonaws.com"));
        assert_eq!(u.path(), "/test.txt");
        let sig = u.query_pairs().find(|(k, _)| k == "X-Amz-Signature").unwrap().1.into_owned();
        assert_eq!(sig, "aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404");
    }

    #[test]
    fn s3_path_style_and_clamp() {
        let cfg = S3Config {
            endpoint: Url::parse("https://s3api-eu-ro-1.runpod.io").unwrap(),
            region: "eu-ro-1".into(),
            bucket: "vol123".into(),
            access_key: "AK".into(),
            secret_key: "SK".into(),
            path_style: true,
            prefix: "out/".into(),
        };
        let s = S3ArtifactStore::new(cfg);
        let a = Artifact {
            id: ArtifactId::new(),
            mime: "video/mp4".into(),
            file_name: "a b.mp4".into(),
            bytes: 1,
            location: ArtifactLocation::Object { bucket: "vol123".into(), key: "out/x/a b.mp4".into() },
            width: 1, height: 1, frames: 1, fps: 24, audio: None,
        };
        let u = s.url_for(&a, Duration::from_secs(30 * 24 * 3600));
        assert_eq!(u.host_str(), Some("s3api-eu-ro-1.runpod.io"));
        assert_eq!(u.path(), "/vol123/out/x/a%20b.mp4");
        let exp = u.query_pairs().find(|(k, _)| k == "X-Amz-Expires").unwrap().1.into_owned();
        assert_eq!(exp, "604800", "clamped to 7 days");
    }

    #[tokio::test]
    async fn local_put_and_serve() {
        let dir = std::env::temp_dir().join(format!("fvkit-art-{}", crate::random_token()));
        let key = UrlKey::new("k");
        let urls = LocalUrls { public_base: Url::parse("http://localhost:8000").unwrap(), key: key.clone() };
        let store = LocalArtifactStore::new(dir.join("artifacts"), urls);
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let src = dir.join("tmp.mp4");
        tokio::fs::write(&src, b"0123456789").await.unwrap();
        let meta = ArtifactMeta { file_name: "out.mp4".into(), mime: "video/mp4".into(), ..Default::default() };
        let a = store.put(&src, meta).await.unwrap();
        assert_eq!(a.bytes, 10);
        assert!(!src.exists());

        let app: Router = files_router(Arc::new(FileServer { key: key.clone(), roots: vec![dir.join("artifacts")] }));
        let url = store.url_for(&a, Duration::from_secs(60));
        let pq = format!("{}?{}", url.path(), url.query().unwrap());
        let r = app.clone().oneshot(Request::get(&pq).body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(r.status(), StatusCode::OK);
        assert_eq!(r.headers()["access-control-allow-origin"], "*");
        let b = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
        assert_eq!(&b[..], b"0123456789");

        // Range requests work (video players seek).
        let r = app
            .clone()
            .oneshot(Request::get(&pq).header("range", "bytes=2-4").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::PARTIAL_CONTENT);

        // Tampered, expired and unsigned requests are 404.
        let bad = pq.replace("sig=", "sig=00");
        let expired = key.sign(&a.id.to_string(), "out.mp4", 1);
        let expired = format!("{}?exp=1&sig={expired}", url.path());
        for p in [bad, expired, url.path().to_owned()] {
            let r = app.clone().oneshot(Request::get(&p).body(Body::empty()).unwrap()).await.unwrap();
            assert_eq!(r.status(), StatusCode::NOT_FOUND, "{p}");
        }
        store.delete(&a).await;
        let r = app.oneshot(Request::get(&pq).body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(r.status(), StatusCode::NOT_FOUND);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
