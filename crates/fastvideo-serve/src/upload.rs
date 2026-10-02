//! Direct output uploads (docs/serve/dispatch-do-family.md §7): a job taken
//! from a family Durable Object uploads its output straight to R2 through
//! part URLs the object mints for that job, so this host needs no R2
//! credentials, and the upload runs **while the engine is still writing**.
//!
//! - [`Uploads::start`] (at the ack) spawns one `std::thread` per job. It
//!   asks the object for a grant (`upload_init`), then follows the engine's
//!   output file (`<engine output dir>/<job>/output.mp4`) as it grows and
//!   PUTs every complete part (`dispatch.upload_part_mib`) at once.
//! - [`DirectStore`] wraps the artifact store: when the gate stores a job's
//!   finished file, the thread makes one local pass over it, re-PUTs the
//!   parts whose bytes changed since they were sent (a writer that patched
//!   its header) and the parts not sent yet, and reports `upload_done` with
//!   the whole file's SHA-256. The object completes the upload; only its
//!   `upload_committed` lets the job succeed (artifact `Object {bucket, key}`).
//! - Fragmented MP4 (`engine.mp4_fragmented`) makes the file append-only, so
//!   nothing is sent twice and what is left after the last frame is one part
//!   and the complete. Without it the pass still makes the result correct.
//! - A job that fails or is cancelled aborts its upload (`upload_abort`; the
//!   object also aborts uploads of jobs that moved on).

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fastvideo_dispatch_proto::{DoMsg, Part, PartUrl, WorkerMsg};
use fastvideo_protocol::{ApiError, Artifact, ArtifactId, ArtifactLocation, JobId, UrlSigner};
use fastvideo_serve_kit::artifacts::{ArtifactBody, ArtifactMeta};
use fastvideo_serve_kit::ArtifactStore;
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, oneshot};

use crate::edge_link::LinkHandle;

/// Part URLs asked for at a time.
const GRANT_PARTS: u16 = 64;
/// How often the thread looks at the growing file.
const POLL: Duration = Duration::from_millis(50);
/// A request to the object (grant, complete) gives up after this.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// What a job's upload is for.
#[derive(Clone)]
pub struct UploadSpec {
    pub job: JobId,
    pub attempt: u32,
    pub lease: u64,
    /// The file the engine writes while it runs (followed while it grows).
    pub tail: PathBuf,
    /// The object name (`output.mp4`).
    pub name: String,
    pub content_type: String,
    /// The family object's link.
    pub link: Arc<LinkHandle>,
}

/// A committed upload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Committed {
    pub bucket: String,
    pub key: String,
    pub bytes: u64,
    pub sha256: String,
    pub stats: UploadStats,
}

/// How much of the upload was hidden behind the encode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UploadStats {
    pub parts: u32,
    /// Bytes sent while the engine was still writing.
    pub bytes_early: u64,
    /// Bytes sent again because they changed after they were sent.
    pub bytes_resent: u64,
    /// Bytes sent after the engine finished (the tail).
    pub bytes_late: u64,
    /// Engine finished (the file handed over) → committed.
    pub finish_to_commit_ms: u64,
}

enum Ctl {
    Finish(PathBuf, oneshot::Sender<Result<Committed, String>>),
    Abort,
}

/// The per-process registry of running direct uploads.
pub struct Uploads {
    jobs: Mutex<HashMap<JobId, mpsc::UnboundedSender<Ctl>>>,
    part: u64,
    http: reqwest::Client,
}

impl std::fmt::Debug for Uploads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Uploads").field("part", &self.part).finish_non_exhaustive()
    }
}

impl Uploads {
    /// `part_bytes`: the part size (S3: ≥ 5 MiB except the last; tests use less).
    pub fn new(part_bytes: u64) -> Arc<Self> {
        let http = reqwest::Client::builder().connect_timeout(Duration::from_secs(10)).build().unwrap_or_default();
        Arc::new(Self { jobs: Mutex::default(), part: part_bytes.max(1), http })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<JobId, mpsc::UnboundedSender<Ctl>>> {
        self.jobs.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Starts the job's upload thread (a second start of a job is ignored).
    pub fn start(self: &Arc<Self>, spec: UploadSpec) {
        let (tx, rx) = mpsc::unbounded_channel();
        {
            let mut g = self.lock();
            if g.contains_key(&spec.job) {
                return;
            }
            g.insert(spec.job, tx);
        }
        let (part, http) = (self.part, self.http.clone());
        let job = spec.job;
        let r = std::thread::Builder::new().name(format!("fv-upload-{}", &job.to_string()[..job.to_string().len().min(8)])).spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                Ok(rt) => rt,
                Err(e) => {
                    tracing::warn!(job = %job, error = %e, "upload: no runtime for the upload thread");
                    return;
                }
            };
            rt.block_on(run(spec, part, http, rx));
        });
        if let Err(e) = r {
            tracing::warn!(job = %job, error = %e, "upload: could not start the upload thread");
            self.lock().remove(&job);
        }
    }

    /// Whether `job` has a direct upload.
    pub fn has(&self, job: JobId) -> bool {
        self.lock().contains_key(&job)
    }

    /// Hands the finished file to the job's thread and waits for the commit.
    /// `None` when the job has no direct upload.
    pub async fn finish(&self, job: JobId, file: &Path) -> Option<Result<Committed, String>> {
        let tx = self.lock().remove(&job)?;
        let (rtx, rrx) = oneshot::channel();
        if tx.send(Ctl::Finish(file.to_path_buf(), rtx)).is_err() {
            return Some(Err("the upload thread is gone".into()));
        }
        Some(rrx.await.unwrap_or_else(|_| Err("the upload thread ended without an answer".into())))
    }

    /// The job ended without an output to store: its upload is aborted.
    pub fn end(&self, job: JobId) {
        if let Some(tx) = self.lock().remove(&job) {
            let _ = tx.send(Ctl::Abort);
        }
    }
}

/// One sent part.
#[derive(Clone, Debug)]
struct Sent {
    len: u64,
    sha: [u8; 32],
    etag: String,
}

struct Job {
    spec: UploadSpec,
    part: u64,
    http: reqwest::Client,
    upload_id: String,
    key: String,
    bucket: String,
    urls: BTreeMap<u16, String>,
    expires_ms: i64,
    sent: BTreeMap<u16, Sent>,
    /// Bytes from the start of the file covered by sent parts.
    covered: u64,
    stats: UploadStats,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

async fn run(spec: UploadSpec, part: u64, http: reqwest::Client, mut rx: mpsc::UnboundedReceiver<Ctl>) {
    let t0 = Instant::now();
    let job_id = spec.job;
    let mut job = Job {
        spec,
        part,
        http,
        upload_id: String::new(),
        key: String::new(),
        bucket: String::new(),
        urls: BTreeMap::new(),
        expires_ms: 0,
        sent: BTreeMap::new(),
        covered: 0,
        stats: UploadStats::default(),
    };
    let granted = job.init().await;
    if let Err(e) = &granted {
        tracing::warn!(job = %job_id, error = %e, "upload: no grant from the family object (the output goes through the artifact store)");
    } else {
        tracing::debug!(job = %job_id, key = %job.key, grant_ms = t0.elapsed().as_millis() as u64, "upload: granted");
    }
    loop {
        tokio::select! {
            c = rx.recv() => {
                match c {
                    Some(Ctl::Finish(file, reply)) => {
                        let r = match &granted {
                            Ok(()) => job.finish(&file).await,
                            Err(e) => Err(e.clone()),
                        };
                        if r.is_err() && granted.is_ok() {
                            job.abort();
                        }
                        let _ = reply.send(r);
                        return;
                    }
                    Some(Ctl::Abort) | None => {
                        if granted.is_ok() {
                            job.abort();
                        }
                        return;
                    }
                }
            }
            _ = tokio::time::sleep(POLL) => {
                if granted.is_ok() {
                    if let Err(e) = job.follow().await {
                        tracing::debug!(job = %job_id, error = %e, "upload: early part not sent (sent again at the end)");
                    }
                }
            }
        }
    }
}

impl Job {
    async fn init(&mut self) -> Result<(), String> {
        let s = &self.spec;
        let (job_id, attempt, lease, name, ct) = (s.job.to_string(), s.attempt, s.lease, s.name.clone(), s.content_type.clone());
        let mut last = String::new();
        for i in 0..3u32 {
            if i > 0 {
                tokio::time::sleep(Duration::from_millis(500 << i)).await;
            }
            let (j, n, c) = (job_id.clone(), name.clone(), ct.clone());
            let r = self.spec.link.request(move |req| WorkerMsg::UploadInit { req, job_id: j, attempt, lease, name: n, content_type: c, parts: GRANT_PARTS }, REQUEST_TIMEOUT).await;
            match r {
                Ok(DoMsg::UploadGrant { error: Some(e), .. }) => return Err(e),
                Ok(DoMsg::UploadGrant { upload_id, key, bucket, part_urls, expires_ms, .. }) => {
                    self.upload_id = upload_id;
                    self.key = key;
                    self.bucket = bucket;
                    self.add_urls(part_urls, expires_ms);
                    return Ok(());
                }
                Ok(other) => last = format!("unexpected answer {other:?}"),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    fn add_urls(&mut self, urls: Vec<PartUrl>, expires_ms: i64) {
        for u in urls {
            self.urls.insert(u.n, u.url);
        }
        self.expires_ms = expires_ms;
    }

    /// A URL for part `n`, asking the object for more (or fresh) ones.
    async fn url(&mut self, n: u16) -> Result<String, String> {
        let fresh = self.expires_ms - now_ms() > 60_000;
        if let (Some(u), true) = (self.urls.get(&n), fresh) {
            return Ok(u.clone());
        }
        let (job_id, upload_id) = (self.spec.job.to_string(), self.upload_id.clone());
        let r = self.spec.link.request(move |req| WorkerMsg::UploadMore { req, job_id, upload_id, from: n, count: GRANT_PARTS }, REQUEST_TIMEOUT).await?;
        match r {
            DoMsg::UploadGrant { error: Some(e), .. } => Err(e),
            DoMsg::UploadGrant { part_urls, expires_ms, .. } => {
                if !fresh {
                    self.urls.clear();
                }
                self.add_urls(part_urls, expires_ms);
                self.urls.get(&n).cloned().ok_or_else(|| format!("the grant has no URL for part {n}"))
            }
            other => Err(format!("unexpected answer {other:?}")),
        }
    }

    /// PUTs part `n` (3 tries; a refused URL is refreshed once).
    async fn put(&mut self, n: u16, body: Vec<u8>) -> Result<String, String> {
        let mut last = String::new();
        let mut refreshed = false;
        for i in 0..4u32 {
            if i > 0 {
                tokio::time::sleep(Duration::from_millis(250 << i)).await;
            }
            let url = self.url(n).await?;
            let r = self.http.put(&url).header("content-length", body.len()).body(body.clone()).timeout(Duration::from_secs(120)).send().await;
            match r {
                Ok(resp) if resp.status().is_success() => {
                    let etag = resp.headers().get("etag").and_then(|v| v.to_str().ok()).unwrap_or_default().to_owned();
                    if etag.is_empty() {
                        return Err(format!("part {n}: no ETag in the answer"));
                    }
                    return Ok(etag);
                }
                Ok(resp) => {
                    let s = resp.status().as_u16();
                    last = format!("part {n}: HTTP {s}");
                    if matches!(s, 401 | 403) && !refreshed {
                        // Expired or refused URL: ask for fresh ones.
                        refreshed = true;
                        self.expires_ms = 0;
                    }
                }
                Err(e) => last = format!("part {n}: {}", e.without_url()),
            }
        }
        Err(last)
    }

    /// Sends the parts the growing file already holds in full.
    async fn follow(&mut self) -> Result<(), String> {
        let len = match std::fs::metadata(&self.spec.tail) {
            Ok(m) => m.len(),
            Err(_) => return Ok(()),
        };
        while len >= self.covered + self.part {
            let n = u16::try_from(self.covered / self.part + 1).map_err(|_| "too many parts".to_owned())?;
            let body = read_range(&self.spec.tail, self.covered, self.part).map_err(|e| format!("reading the output: {e}"))?;
            let sha: [u8; 32] = Sha256::digest(&body).into();
            let etag = self.put(n, body).await?;
            self.sent.insert(n, Sent { len: self.part, sha, etag });
            self.covered += self.part;
            self.stats.bytes_early += self.part;
            self.stats.parts += 1;
        }
        Ok(())
    }

    /// The pass over the finished file, then the commit.
    async fn finish(&mut self, file: &Path) -> Result<Committed, String> {
        let t0 = Instant::now();
        let len = std::fs::metadata(file).map_err(|e| format!("the output: {e}"))?.len();
        let count = len.div_ceil(self.part).max(1);
        let count = u16::try_from(count).map_err(|_| "too many parts for this part size".to_owned())?;
        let mut whole = Sha256::new();
        let mut f = std::fs::File::open(file).map_err(|e| format!("the output: {e}"))?;
        let mut parts = Vec::with_capacity(count as usize);
        for n in 1..=count {
            let want = self.part.min(len.saturating_sub(u64::from(n - 1) * self.part));
            let mut body = vec![0u8; want as usize];
            f.read_exact(&mut body).map_err(|e| format!("reading the output: {e}"))?;
            whole.update(&body);
            let sha: [u8; 32] = Sha256::digest(&body).into();
            let same = self.sent.get(&n).is_some_and(|s| s.len == want && s.sha == sha);
            if !same {
                if self.sent.contains_key(&n) {
                    self.stats.bytes_resent += want;
                } else {
                    self.stats.parts += 1;
                }
                self.stats.bytes_late += want;
                let etag = self.put(n, body).await?;
                self.sent.insert(n, Sent { len: want, sha, etag });
            }
            parts.push(Part { n, etag: self.sent[&n].etag.clone() });
        }
        let sha256 = fastvideo_dispatch_proto::presign::hex(&whole.finalize());
        let (job_id, attempt, lease, upload_id) = (self.spec.job.to_string(), self.spec.attempt, self.spec.lease, self.upload_id.clone());
        let sha2 = sha256.clone();
        let r = self
            .spec
            .link
            .request(move |req| WorkerMsg::UploadDone { req, job_id, attempt, lease, upload_id, parts, bytes: len, sha256: sha2 }, REQUEST_TIMEOUT)
            .await?;
        self.stats.finish_to_commit_ms = t0.elapsed().as_millis() as u64;
        match r {
            DoMsg::UploadCommitted { ok: true, key, bytes, .. } => {
                let st = self.stats;
                metrics::counter!("fv_worker_upload_bytes_early_total").increment(st.bytes_early);
                metrics::counter!("fv_worker_upload_bytes_late_total").increment(st.bytes_late);
                metrics::counter!("fv_worker_upload_bytes_resent_total").increment(st.bytes_resent);
                metrics::histogram!("fv_worker_upload_tail_seconds").record(st.finish_to_commit_ms as f64 / 1e3);
                tracing::info!(job = %self.spec.job, key = %key, bytes, early = st.bytes_early, late = st.bytes_late, resent = st.bytes_resent, tail_ms = st.finish_to_commit_ms, "upload: committed");
                Ok(Committed { bucket: self.bucket.clone(), key: if key.is_empty() { self.key.clone() } else { key }, bytes, sha256, stats: st })
            }
            DoMsg::UploadCommitted { error, .. } => Err(error.unwrap_or_else(|| "the object refused the upload".into())),
            other => Err(format!("unexpected answer {other:?}")),
        }
    }

    fn abort(&self) {
        if !self.upload_id.is_empty() {
            self.spec.link.send(WorkerMsg::UploadAbort { job_id: self.spec.job.to_string(), upload_id: self.upload_id.clone() });
        }
    }
}

fn read_range(p: &Path, offset: u64, len: u64) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(p)?;
    f.seek(SeekFrom::Start(offset))?;
    let mut b = vec![0u8; len as usize];
    f.read_exact(&mut b)?;
    Ok(b)
}

/// The job a staged output belongs to: outputs live in `…/<job id>/<file>`
/// (the engine's `engine-out/<job>/output.mp4`, the gate's
/// `outputs/<job>/final.mp4` or placeholder).
pub fn job_of(file: &Path) -> Option<JobId> {
    file.parent()?.file_name()?.to_str()?.parse().ok()
}

/// The artifact store with direct uploads in front: outputs of jobs with a
/// direct upload go through it, everything else to `inner`.
pub struct DirectStore {
    inner: Arc<dyn ArtifactStore>,
    uploads: Arc<Uploads>,
}

impl DirectStore {
    pub fn new(inner: Arc<dyn ArtifactStore>, uploads: Arc<Uploads>) -> Arc<Self> {
        Arc::new(Self { inner, uploads })
    }
}

impl UrlSigner for DirectStore {
    fn url_for(&self, a: &Artifact, ttl: Duration) -> url::Url {
        self.inner.signer().url_for(a, ttl)
    }
}

#[async_trait::async_trait]
impl ArtifactStore for DirectStore {
    async fn put(&self, src: &Path, meta: ArtifactMeta) -> Result<Artifact, ApiError> {
        if let Some(job) = job_of(src) {
            if let Some(r) = self.uploads.finish(job, src).await {
                match r {
                    Ok(c) => {
                        let _ = tokio::fs::remove_file(src).await;
                        return Ok(Artifact {
                            id: ArtifactId::new(),
                            mime: meta.mime,
                            file_name: meta.file_name,
                            bytes: c.bytes,
                            location: ArtifactLocation::Object { bucket: c.bucket, key: c.key },
                            width: meta.width,
                            height: meta.height,
                            frames: meta.frames,
                            fps: meta.fps,
                            audio: meta.audio,
                        });
                    }
                    Err(e) => {
                        metrics::counter!("fv_worker_upload_failed_total").increment(1);
                        tracing::warn!(job = %job, error = %e, "upload: the direct upload failed; storing through the artifact store");
                    }
                }
            }
        }
        self.inner.put(src, meta).await
    }

    async fn delete(&self, a: &Artifact) {
        self.inner.delete(a).await;
    }

    fn signer(&self) -> &dyn UrlSigner {
        self
    }

    async fn open(&self, a: &Artifact) -> Result<ArtifactBody, ApiError> {
        self.inner.open(a).await
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use fastvideo_dispatch_proto::presign::S3Presign;
    use fastvideo_serve_kit::artifacts::S3Config;

    use super::job_of;

    /// The family objects' SigV4 presigner (pure, wasm) signs exactly as
    /// serve-kit's `S3Config::presign` does.
    #[test]
    fn presigner_matches_serve_kit() {
        for (endpoint, path_style) in [("https://acct.r2.cloudflarestorage.com", true), ("https://s3.amazonaws.com", false), ("http://127.0.0.1:9000/base", true)] {
            let kit = S3Config {
                endpoint: url::Url::parse(endpoint).unwrap(),
                region: "auto".into(),
                bucket: "outs".into(),
                access_key: "AK".into(),
                secret_key: "SK/x+y".into(),
                path_style,
                prefix: String::new(),
            };
            let ours = S3Presign { endpoint: endpoint.into(), region: "auto".into(), bucket: "outs".into(), access_key: "AK".into(), secret_key: "SK/x+y".into(), path_style };
            let now = time::OffsetDateTime::from_unix_timestamp(1_790_000_000).unwrap();
            for key in ["outputs/wan/j1/1-1/out.mp4", "a b/ü.mp4"] {
                let a = kit.presign("PUT", key, Duration::from_secs(3600), now);
                let b = ours.presign("PUT", key, &[], 3600, 1_790_000_000);
                assert_eq!(a.as_str(), b, "{endpoint} {key}");
            }
        }
    }

    #[test]
    fn outputs_name_their_job() {
        let id = fastvideo_protocol::JobId::new();
        assert_eq!(job_of(&std::path::Path::new("/s/engine-out").join(id.to_string()).join("output.mp4")), Some(id));
        assert_eq!(job_of(std::path::Path::new("/s/outputs/x/final.mp4")), None);
    }
}
