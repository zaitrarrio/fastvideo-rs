//! The dispatch envelope and its inputs (docs/serve/gateway.md §3, §3.1),
//! shared by the gateway and the fronts behind the edge
//! (docs/serve/edge-control-plane.md §2.4).
//!
//! Envelope (**native**), the body of the worker's
//! `POST /fv/v1/internal/jobs` and of a family object's push:
//! `{"job": <Job>, "inputs": [{"path", "inline" | "source" | "url"+"artifact", …}], "attempt": n, "pool": id}`.
//!
//! Inputs travel the cheapest way that works: inline in the envelope
//! (base64) up to a byte budget per job; large video/audio the client gave
//! as a public URL are fetched by the executing worker from that URL; the
//! rest go through the artifact store (R2) with a signed URL.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use base64::Engine as _;
use fastvideo_protocol::{ApiError, Artifact, Job, JobId, MediaKind};
use fastvideo_serve_kit::artifacts::valid_file_name;
use fastvideo_serve_kit::d1::{D1Client, Stmt};
use fastvideo_serve_kit::{ArtifactMeta, ServeCtx};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// `param` of the error a dispatch returns when the worker could not fetch
/// a passed-through input (HTTP 424 from the worker).
pub const INPUT_FETCH_FAILED: &str = "dispatch.inputs";

/// Serverless pools inline at most this much: a Runpod `/run` body is
/// ≤ 10 MB and base64 adds a third.
pub(crate) const SERVERLESS_INLINE_MAX: u64 = 6 * 1024 * 1024;

/// One input shipped to the worker, by exactly one of `inline`, `source`,
/// or `url` (with `artifact`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InputRef {
    /// The path in the dispatched job's `resolved` (the gateway's).
    pub path: PathBuf,
    /// Signed URL the worker downloads it from (store path; empty otherwise).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub url: String,
    /// The artifact holding it (the worker deletes it when the job ends).
    #[serde(default)]
    pub artifact: Option<Artifact>,
    /// The bytes, base64 (standard alphabet), inside the envelope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline: Option<String>,
    /// A public URL the client gave: the worker fetches it itself (SSRF
    /// guard of ingestion) and checks `sha256`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// SHA-256 (hex) of the gateway's staged copy (`source` inputs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Media kind (limits of a `source` fetch).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<MediaKind>,
    #[serde(default)]
    pub bytes: u64,
}

impl InputRef {
    /// How it travels: `inline`, `source`, `store` (or `none`: not yet).
    pub fn via(&self) -> &'static str {
        if self.inline.is_some() {
            "inline"
        } else if self.source.is_some() {
            "source"
        } else if !self.url.is_empty() || self.artifact.is_some() {
            "store"
        } else {
            "none"
        }
    }
    /// The copy kept in `gw_dispatch` (D1 rows stay small: never the bytes).
    pub(crate) fn for_record(&self) -> Self {
        Self { inline: None, ..self.clone() }
    }
}

/// A job's input files with their kinds, in `resolved` order.
pub(crate) fn input_paths(job: &Job) -> Vec<(PathBuf, MediaKind)> {
    let r = &job.resolved;
    let mut v: Vec<(PathBuf, MediaKind)> = r.keyframes.iter().map(|(_, p)| (p.clone(), MediaKind::Image)).collect();
    v.extend(r.references.iter().map(|(k, p)| (p.clone(), *k)));
    v.extend(r.audio_in.iter().map(|(_, p)| (p.clone(), MediaKind::Audio)));
    v
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// SHA-256 (hex) of a file, off the async threads.
pub async fn sha256_file(path: &Path) -> Result<String, ApiError> {
    let p = path.to_owned();
    tokio::task::spawn_blocking(move || {
        use sha2::Digest;
        use std::io::Read;
        let mut f = std::fs::File::open(&p)?;
        let mut h = sha2::Sha256::new();
        let mut buf = vec![0u8; 1 << 16];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        Ok::<_, std::io::Error>(hex(&h.finalize()))
    })
    .await
    .map_err(|e| ApiError::internal(e.to_string()))?
    .map_err(|e| ApiError::internal(format!("hashing an input: {e}")))
}

/// Copies `input.path` into the artifact store and signs a URL for it.
pub(crate) async fn store_one(ctx: &ServeCtx, ttl: Duration, i: usize, input: &InputRef) -> Result<InputRef, ApiError> {
    let path = &input.path;
    let name = path.file_name().and_then(|n| n.to_str()).filter(|n| valid_file_name(n)).map(str::to_owned).unwrap_or_else(|| format!("input-{i}.bin"));
    // The store moves the file: keep the gateway copy (a re-dispatch or a
    // later copy still has it) by copying first.
    let tmp = path.with_extension(format!("gw{i}-{}.tmp", fastvideo_serve_kit::random_token()));
    tokio::fs::copy(path, &tmp).await.map_err(|e| ApiError::internal(format!("staging input for dispatch: {e}")))?;
    let meta = ArtifactMeta { file_name: name, mime: "application/octet-stream".into(), ..ArtifactMeta::default() };
    let art = match ctx.artifacts().put(&tmp, meta).await {
        Ok(a) => a,
        Err(e) => {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(e);
        }
    };
    let url = ctx.url_signer().url_for(&art, ttl).to_string();
    Ok(InputRef { url, artifact: Some(art), inline: None, source: None, sha256: None, ..input.clone() })
}

/// Stores the inputs at `idx` concurrently (one store round trip in wall
/// time, not one per input); on a failure the ones that made it are deleted.
pub(crate) async fn store_many(ctx: &ServeCtx, ttl: Duration, inputs: &mut [InputRef], idx: &[usize]) -> Result<(), ApiError> {
    let puts = idx.iter().map(|&i| {
        let input = inputs[i].clone();
        async move { store_one(ctx, ttl, i, &input).await.map(|r| (i, r)) }
    });
    let results = futures::future::join_all(puts).await;
    let mut failed = None;
    let mut done = Vec::new();
    for r in results {
        match r {
            Ok(x) => done.push(x),
            Err(e) => failed = failed.or(Some(e)),
        }
    }
    if let Some(e) = failed {
        for (_, r) in &done {
            if let Some(a) = &r.artifact {
                ctx.artifacts().delete(a).await;
            }
        }
        return Err(e);
    }
    for (i, r) in done {
        inputs[i] = r;
    }
    Ok(())
}

/// Background copy of the inputs that skipped the store (see the module
/// docs), then the `gw_dispatch` row points at them. Dropped again when
/// the row moved on meanwhile (job done, or already re-dispatched).
pub(crate) async fn stage_for_retry(db: D1Client, ctx: ServeCtx, ttl: Duration, job: JobId, attempt: u32, mut inputs: Vec<InputRef>) {
    let t0 = Instant::now();
    let idx: Vec<usize> = (0..inputs.len()).filter(|&i| inputs[i].artifact.is_none()).collect();
    if idx.is_empty() {
        return;
    }
    if let Err(e) = store_many(&ctx, ttl, &mut inputs, &idx).await {
        tracing::warn!(job = %job, error = %e.message, "gateway: copying inputs to the store for a re-dispatch failed");
        return;
    }
    let rec: Vec<InputRef> = inputs.iter().map(InputRef::for_record).collect();
    let body = serde_json::to_string(&rec).unwrap_or_else(|_| "[]".into());
    let changed = db
        .query(Stmt::new(
            "UPDATE gw_dispatch SET inputs = ? WHERE job_id = ? AND attempt = ? AND state = 'active'",
            vec![json!(body), json!(job.to_string()), json!(attempt)],
        ))
        .await
        .map(|r| r.changes)
        .unwrap_or(0);
    if changed == 0 {
        for i in &idx {
            if let Some(a) = &inputs[*i].artifact {
                ctx.artifacts().delete(a).await;
            }
        }
        return;
    }
    let s = t0.elapsed().as_secs_f64();
    metrics::histogram!("fv_gateway_retry_stage_seconds").record(s);
    tracing::debug!(job = %job, inputs = idx.len(), ms = (s * 1e3) as u64, "gateway: inputs copied to the store for a re-dispatch");
}

/// The dispatch envelope (see the module docs).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Envelope {
    pub job: Job,
    #[serde(default)]
    pub inputs: Vec<InputRef>,
    #[serde(default = "one")]
    pub attempt: u32,
    #[serde(default)]
    pub pool: Option<String>,
}

fn one() -> u32 {
    1
}


/// How each input travels: inline while the job's total fits `budget`,
/// then the client's URL for large video/audio (`passthrough`), else the
/// store (all store copies at once).
pub async fn plan_inputs(ctx: &ServeCtx, job: &Job, budget: u64, passthrough: bool, ttl: Duration) -> Result<Vec<InputRef>, ApiError> {
    let paths = input_paths(job);
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    let mut left = budget;
    let mut out = Vec::with_capacity(paths.len());
    let mut to_store = Vec::new();
    for (i, (path, kind)) in paths.into_iter().enumerate() {
        let bytes = tokio::fs::metadata(&path).await.map_err(|e| ApiError::internal(format!("staged input: {e}")))?.len();
        let mut r = InputRef { path, kind: Some(kind), bytes, ..InputRef::default() };
        let source = job.input_sources.iter().find(|(p, _)| *p == r.path).map(|(_, u)| u.clone());
        if bytes <= left {
            let data = tokio::fs::read(&r.path).await.map_err(|e| ApiError::internal(format!("staged input: {e}")))?;
            r.inline = Some(base64::engine::general_purpose::STANDARD.encode(data));
            left -= bytes;
        } else if let Some(u) = source.filter(|_| passthrough) {
            r.sha256 = Some(sha256_file(&r.path).await?);
            r.source = Some(u);
        } else {
            to_store.push(i);
        }
        out.push(r);
    }
    if !to_store.is_empty() {
        store_many(ctx, ttl, &mut out, &to_store).await?;
    }
    Ok(out)
}

/// Copies the inputs that skipped the store into it (`idx`: all of them
/// when `None`) and returns the inputs pointing at the copies.
pub async fn store_copies(ctx: &ServeCtx, ttl: Duration, inputs: &[InputRef]) -> Result<Vec<InputRef>, ApiError> {
    let mut out = inputs.to_vec();
    let idx: Vec<usize> = (0..out.len()).filter(|&i| out[i].artifact.is_none()).collect();
    if !idx.is_empty() {
        store_many(ctx, ttl, &mut out, &idx).await?;
    }
    Ok(out)
}
