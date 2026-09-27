//! `UploadStore` and `PUT /uploads/{token}` (design §4.4 fal storage initiate,
//! §4.5 LTX `/v1/upload`, §9).
//!
//! An adapter issues a ticket ([`UploadStore::create`]); the client `PUT`s raw
//! bytes to the ticket's `upload_url` with no auth (the unguessable token is
//! the credential). A token takes exactly one upload ("creates a new object and
//! cannot overwrite an existing object", LTX); `x-goog-*` headers that LTX
//! clients copy from `required_headers` are accepted and ignored.
//!
//! Uploaded files live at `<root>/<token>/<file_name>`, the same layout as
//! local artifacts, so [`files_router`](crate::files_router) can serve them
//! with a signed URL (fal `file_url`), and ingestion resolves
//! `ltx://uploads/<token>` ([`MediaRef::Upload`](fastvideo_protocol::MediaRef)).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path as UrlPath, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::put;
use axum::Router;
use fastvideo_protocol::UploadId;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use tokio::io::AsyncWriteExt;
use url::Url;

use crate::artifacts::{valid_file_name, LocalUrls};

/// What a client gets back from an upload-initiate call.
#[derive(Clone, Debug, PartialEq)]
pub struct UploadTicket {
    pub token: String,
    /// `<public_base>/uploads/<token>`.
    pub upload_url: Url,
    /// The PUT must happen before this.
    pub expires_at: OffsetDateTime,
    pub file_name: String,
}

impl UploadTicket {
    /// `ltx://uploads/<token>`.
    pub fn ltx_storage_uri(&self) -> String {
        format!("ltx://uploads/{}", self.token)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Meta {
    token: String,
    file_name: String,
    declared_mime: Option<String>,
    #[serde(with = "time::serde::rfc3339")]
    put_deadline: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    file_expires_at: OffsetDateTime,
    /// Set once the PUT completed.
    mime: Option<String>,
    bytes: Option<u64>,
}

/// A completed upload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UploadedFile {
    pub path: PathBuf,
    /// The PUT `Content-Type` (or the declared one), if any.
    pub mime: Option<String>,
    pub bytes: u64,
    pub file_name: String,
}

/// Why a PUT was refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum UploadError {
    #[error("unknown or expired upload token")]
    NotFound,
    #[error("this upload URL was already used")]
    AlreadyUploaded,
    #[error("upload exceeds {0} bytes")]
    TooLarge(u64),
    #[error("upload failed: {0}")]
    Io(String),
}

impl UploadError {
    fn status(&self) -> StatusCode {
        match self {
            UploadError::NotFound => StatusCode::NOT_FOUND,
            UploadError::AlreadyUploaded => StatusCode::PRECONDITION_FAILED,
            UploadError::TooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            UploadError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

/// Upload tickets and files.
pub struct UploadStore {
    root: PathBuf,
    urls: LocalUrls,
    /// How long a ticket accepts its PUT (LTX: 1 h).
    pub put_ttl: Duration,
    /// How long an uploaded file lives (LTX: 24 h).
    pub file_ttl: Duration,
    /// Largest accepted upload (LTX: 200 MB).
    pub max_bytes: u64,
    meta: Mutex<HashMap<String, Meta>>,
}

impl std::fmt::Debug for UploadStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UploadStore").field("root", &self.root).finish_non_exhaustive()
    }
}

const META: &str = ".meta.json";

impl UploadStore {
    /// A store under `root`, reloading tickets left by a previous process.
    pub fn new(root: impl Into<PathBuf>, urls: LocalUrls) -> std::io::Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        let mut meta = HashMap::new();
        for e in std::fs::read_dir(&root)?.flatten() {
            let p = e.path().join(META);
            if let Ok(m) = std::fs::read(&p).map(|b| serde_json::from_slice::<Meta>(&b)) {
                match m {
                    Ok(m) => {
                        meta.insert(m.token.clone(), m);
                    }
                    Err(err) => tracing::warn!(path = %p.display(), %err, "bad upload meta"),
                }
            }
        }
        Ok(Self {
            root,
            urls,
            put_ttl: Duration::from_secs(3600),
            file_ttl: Duration::from_secs(24 * 3600),
            max_bytes: 200 * 1024 * 1024,
            meta: Mutex::new(meta),
        })
    }

    pub fn root(&self) -> &std::path::Path {
        &self.root
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Meta>> {
        self.meta.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Issues a ticket. `file_name` is sanitized (default `upload`);
    /// `content_type` is the declared type (fal initiate `content_type`).
    pub fn create(
        &self,
        file_name: Option<&str>,
        content_type: Option<&str>,
        now: OffsetDateTime,
    ) -> std::io::Result<UploadTicket> {
        let token = crate::random_token();
        let file_name = file_name
            .map(sanitize_name)
            .filter(|n| valid_file_name(n))
            .unwrap_or_else(|| "upload".to_owned());
        let m = Meta {
            token: token.clone(),
            file_name: file_name.clone(),
            declared_mime: content_type.map(str::to_owned),
            put_deadline: now + self.put_ttl,
            file_expires_at: now + self.put_ttl + self.file_ttl,
            mime: None,
            bytes: None,
        };
        let dir = self.root.join(&token);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join(META), serde_json::to_vec(&m).map_err(std::io::Error::other)?)?;
        self.lock().insert(token.clone(), m);
        let mut upload_url = self.urls.public_base.clone();
        if !upload_url.path().ends_with('/') {
            let p = format!("{}/", upload_url.path());
            upload_url.set_path(&p);
        }
        let upload_url = upload_url
            .join(&format!("uploads/{token}"))
            .expect("token is url-safe");
        Ok(UploadTicket {
            token,
            upload_url,
            expires_at: now + self.put_ttl,
            file_name,
        })
    }

    /// Stores the PUT body for `token`.
    pub async fn put(
        &self,
        token: &str,
        content_type: Option<&str>,
        body: Body,
        now: OffsetDateTime,
    ) -> Result<u64, UploadError> {
        let m = {
            let g = self.lock();
            let m = g.get(token).ok_or(UploadError::NotFound)?;
            if m.bytes.is_some() {
                return Err(UploadError::AlreadyUploaded);
            }
            if now > m.put_deadline {
                return Err(UploadError::NotFound);
            }
            m.clone()
        };
        let dir = self.root.join(token);
        let tmp = dir.join(format!(".part-{}", crate::random_token()));
        let io = |e: std::io::Error| UploadError::Io(e.to_string());
        let mut f = tokio::fs::File::create(&tmp).await.map_err(io)?;
        let mut n = 0u64;
        let mut stream = body.into_data_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    let _ = tokio::fs::remove_file(&tmp).await;
                    return Err(UploadError::Io(e.to_string()));
                }
            };
            n += chunk.len() as u64;
            if n > self.max_bytes {
                drop(f);
                let _ = tokio::fs::remove_file(&tmp).await;
                return Err(UploadError::TooLarge(self.max_bytes));
            }
            f.write_all(&chunk).await.map_err(io)?;
        }
        f.flush().await.map_err(io)?;
        drop(f);
        let mime = content_type
            .filter(|c| !c.is_empty() && *c != "application/octet-stream")
            .map(str::to_owned)
            .or(m.declared_mime.clone());
        let committed: Result<Vec<u8>, UploadError> = (|| {
            let mut g = self.lock();
            let e = g.get_mut(token).ok_or(UploadError::NotFound)?;
            if e.bytes.is_some() {
                return Err(UploadError::AlreadyUploaded);
            }
            e.bytes = Some(n);
            e.mime = mime;
            e.file_expires_at = now + self.file_ttl;
            serde_json::to_vec(&*e).map_err(|e| UploadError::Io(e.to_string()))
        })();
        let meta = match committed {
            Ok(m) => m,
            Err(e) => {
                let _ = tokio::fs::remove_file(&tmp).await;
                return Err(e);
            }
        };
        tokio::fs::rename(&tmp, dir.join(&m.file_name)).await.map_err(io)?;
        tokio::fs::write(dir.join(META), meta).await.map_err(io)?;
        Ok(n)
    }

    /// The uploaded file behind `id`, if the upload completed and is alive.
    pub fn resolve(&self, id: &UploadId, now: OffsetDateTime) -> Option<UploadedFile> {
        let g = self.lock();
        let m = g.get(&id.0)?;
        let bytes = m.bytes?;
        if now > m.file_expires_at {
            return None;
        }
        Some(UploadedFile {
            path: self.root.join(&m.token).join(&m.file_name),
            mime: m.mime.clone(),
            bytes,
            file_name: m.file_name.clone(),
        })
    }

    /// A signed `/files/<token>/<file_name>` URL for the upload (fal `file_url`).
    /// Valid whether or not the PUT happened yet.
    pub fn file_url(&self, token: &str, ttl: Duration) -> Option<Url> {
        let g = self.lock();
        let m = g.get(token)?;
        Some(self.urls.files_url_ttl(&m.token, &m.file_name, ttl))
    }

    /// Removes tickets and files past their lifetime; returns the count.
    pub async fn sweep(&self, now: OffsetDateTime) -> usize {
        let dead: Vec<String> = {
            let mut g = self.lock();
            let ids: Vec<String> = g
                .values()
                .filter(|m| now > m.file_expires_at || (m.bytes.is_none() && now > m.put_deadline))
                .map(|m| m.token.clone())
                .collect();
            for t in &ids {
                g.remove(t);
            }
            ids
        };
        for t in &dead {
            let _ = tokio::fs::remove_dir_all(self.root.join(t)).await;
        }
        dead.len()
    }
}

fn sanitize_name(n: &str) -> String {
    let base = n.rsplit(['/', '\\']).next().unwrap_or(n);
    let s: String = base
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' })
        .collect();
    s.trim_start_matches('.').chars().take(200).collect()
}

/// `PUT /uploads/{token}` (design §9, owned by serve-kit). Open route: the
/// token is the credential. Answers 200 `{}`; 404 unknown/expired token; 412
/// already used; 413 too large.
pub fn uploads_router<S: Clone + Send + Sync + 'static>(store: Arc<UploadStore>) -> Router<S> {
    Router::new()
        .route("/uploads/{token}", put(put_upload))
        .layer(DefaultBodyLimit::disable())
        .with_state(store)
}

async fn put_upload(
    State(store): State<Arc<UploadStore>>,
    UrlPath(token): UrlPath<String>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let ct = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(';').next().unwrap_or(s).trim().to_owned());
    let mut resp = match store
        .put(&token, ct.as_deref(), body, OffsetDateTime::now_utc())
        .await
    {
        Ok(_) => (StatusCode::OK, axum::Json(serde_json::json!({}))).into_response(),
        Err(e) => (e.status(), axum::Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
    };
    resp.headers_mut().insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        axum::http::HeaderValue::from_static("*"),
    );
    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::{files_router, FileServer, UrlKey};
    use axum::http::Request;
    use tower::ServiceExt;

    fn store(dir: &std::path::Path) -> Arc<UploadStore> {
        let urls = LocalUrls { public_base: Url::parse("http://h:1/api").unwrap(), key: UrlKey::new("k") };
        Arc::new(UploadStore::new(dir.join("uploads"), urls).unwrap())
    }

    fn put_req(tok: &str, body: &'static [u8]) -> Request<Body> {
        Request::put(format!("/uploads/{tok}"))
            .header("content-type", "image/png")
            .header("x-goog-content-length-range", "0,209715200")
            .header("x-goog-if-generation-match", "0")
            .body(Body::from(body))
            .unwrap()
    }

    #[tokio::test]
    async fn ticket_put_resolve_once() {
        let dir = std::env::temp_dir().join(format!("fvkit-up-{}", crate::random_token()));
        let s = store(&dir);
        let now = OffsetDateTime::now_utc();
        let t = s.create(Some("../../etc/a photo.png"), None, now).unwrap();
        assert_eq!(t.token.len(), 32);
        assert_eq!(t.file_name, "a_photo.png");
        assert_eq!(t.upload_url.as_str(), format!("http://h:1/api/uploads/{}", t.token));
        assert_eq!(t.ltx_storage_uri(), format!("ltx://uploads/{}", t.token));
        assert!(s.resolve(&UploadId(t.token.clone()), now).is_none(), "not uploaded yet");

        let app: Router = uploads_router(s.clone());
        let r = app.clone().oneshot(put_req(&t.token, b"PNGDATA")).await.unwrap();
        assert_eq!(r.status(), StatusCode::OK);
        let f = s.resolve(&UploadId(t.token.clone()), now).unwrap();
        assert_eq!((f.bytes, f.mime.as_deref()), (7, Some("image/png")));
        assert_eq!(std::fs::read(&f.path).unwrap(), b"PNGDATA");

        let r = app.clone().oneshot(put_req(&t.token, b"again")).await.unwrap();
        assert_eq!(r.status(), StatusCode::PRECONDITION_FAILED);
        let r = app.clone().oneshot(put_req("0123456789abcdef0123456789abcdef", b"x")).await.unwrap();
        assert_eq!(r.status(), StatusCode::NOT_FOUND);

        // Served by /files with a signed URL (fal file_url).
        let files: Router = files_router(Arc::new(FileServer { key: UrlKey::new("k"), roots: vec![s.root().to_owned()] }));
        let u = s.file_url(&t.token, Duration::from_secs(60)).unwrap();
        let r = files
            .oneshot(Request::get(format!("{}?{}", u.path().trim_start_matches("/api"), u.query().unwrap())).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::OK);

        // Reload from disk.
        let s2 = store(&dir);
        assert!(s2.resolve(&UploadId(t.token.clone()), now).is_some());
        // Expiry.
        let later = now + Duration::from_secs(3 * 24 * 3600);
        assert!(s2.resolve(&UploadId(t.token.clone()), later).is_none());
        assert_eq!(s2.sweep(later).await, 1);
        assert!(!s2.root().join(&t.token).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn size_limit_and_deadline() {
        let dir = std::env::temp_dir().join(format!("fvkit-up-{}", crate::random_token()));
        let urls = LocalUrls { public_base: Url::parse("http://h").unwrap(), key: UrlKey::new("k") };
        let mut s = UploadStore::new(dir.join("uploads"), urls).unwrap();
        s.max_bytes = 4;
        let s = Arc::new(s);
        let now = OffsetDateTime::now_utc();
        let t = s.create(None, Some("video/mp4"), now).unwrap();
        assert_eq!(t.file_name, "upload");
        let app: Router = uploads_router(s.clone());
        let r = app.clone().oneshot(put_req(&t.token, b"12345")).await.unwrap();
        assert_eq!(r.status(), StatusCode::PAYLOAD_TOO_LARGE);
        // A failed PUT does not burn the token.
        assert!(s.put(&t.token, None, Body::from("1234"), now).await.is_ok());
        assert_eq!(s.resolve(&UploadId(t.token.clone()), now).unwrap().mime.as_deref(), Some("video/mp4"));
        let t2 = s.create(None, None, now).unwrap();
        let late = now + Duration::from_secs(3601);
        assert_eq!(s.put(&t2.token, None, Body::from("1"), late).await, Err(UploadError::NotFound));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
