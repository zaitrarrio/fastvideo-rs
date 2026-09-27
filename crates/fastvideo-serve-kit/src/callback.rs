//! Status callbacks (design §4.3 MiniMax `callback_url`, §4.4 fal webhooks).
//!
//! - **MiniMax** (INFERRED V1 form, minimax-fastvideo §1.6): before the first
//!   status POST, send `{"challenge": "<random>"}`; the receiver must answer
//!   2xx with `{"challenge": "<same>"}` within 3 s, else no callbacks are sent
//!   for that task. Then POST the adapter's body (`{"task": …}`) on **every
//!   status change**, in order.
//! - **fal** (fal §9.7): one POST on completion,
//!   `{request_id, gateway_request_id, status: "OK"|"ERROR", payload}`, signed
//!   with **our** Ed25519 key using fal's header scheme
//!   (`X-Fal-Webhook-{Request-Id,User-Id,Timestamp,Signature}`, signature =
//!   hex Ed25519 over `request_id\nuser_id\ntimestamp\nhex(sha256(body))`).
//!   Receivers verify against our JWKS ([`WebhookSigner::jwks`]).
//! - **LTX** and the others: no callbacks.
//!
//! Delivery: 2xx acknowledges; a 3xx or a non-public target is a permanent
//! failure; anything else is retried on the protocol's [`RetrySchedule`].
//! Every target passes the SSRF guard ([`crate::net`]).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine as _;
use bytes::Bytes;
use ed25519_dalek::{Signer, SigningKey};
use fastvideo_protocol::{CallbackSpec, Job, JobId, JobStatus, ViewCtx};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use url::Url;

use crate::hex;
use crate::net::TargetPolicy;

/// When to retry one callback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetrySchedule {
    /// Timeout of the first attempt.
    pub first_timeout: Duration,
    /// Timeout of every retry.
    pub retry_timeout: Duration,
    /// Wait before retry `i` (so `delays.len()` retries).
    pub delays: Vec<Duration>,
    /// Stop retrying once this long has passed since the first attempt.
    pub deadline: Option<Duration>,
}

impl RetrySchedule {
    /// fal §9.7: first attempt 15 s, retries 120 s, up to 31 retries backing
    /// off exponentially (1 s doubling, capped at 5 min), until the stored
    /// result expires (~1 h).
    pub fn fal() -> Self {
        Self {
            first_timeout: Duration::from_secs(15),
            retry_timeout: Duration::from_secs(120),
            delays: (0..31).map(|i| Duration::from_secs((1u64 << i.min(9)).min(300))).collect(),
            deadline: Some(Duration::from_secs(3600)),
        }
    }
    /// MiniMax (retries unspecified upstream): 10 s timeout, 5 retries after
    /// 1, 2, 4, 8, 16 s.
    pub fn minimax() -> Self {
        Self {
            first_timeout: Duration::from_secs(10),
            retry_timeout: Duration::from_secs(10),
            delays: [1, 2, 4, 8, 16].into_iter().map(Duration::from_secs).collect(),
            deadline: None,
        }
    }
    pub fn max_attempts(&self) -> usize {
        1 + self.delays.len()
    }
    fn timeout(&self, attempt: usize) -> Duration {
        if attempt == 0 {
            self.first_timeout
        } else {
            self.retry_timeout
        }
    }
}

/// A receiver's answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PostReply {
    pub status: u16,
    pub body: Bytes,
}

/// Why a POST did not get an answer.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PostError {
    /// The target is refused (SSRF guard): permanent.
    #[error("target refused: {0}")]
    Target(String),
    /// Connection, TLS, protocol: retryable.
    #[error("{0}")]
    Transport(String),
}

/// Sends one POST. The sender applies the timeout.
#[async_trait::async_trait]
pub trait CallbackTransport: Send + Sync + 'static {
    async fn post(
        &self,
        url: &Url,
        headers: &[(String, String)],
        body: Bytes,
        target: &TargetPolicy,
    ) -> Result<PostReply, PostError>;
}

/// The outcome of one callback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Delivery {
    Delivered { attempts: u32 },
    /// Every attempt failed (or the deadline passed).
    Exhausted { attempts: u32, last_error: String },
    /// 3xx or refused target: not retried.
    Permanent { attempts: u32, reason: String },
    /// No transport configured (built without `fetch`).
    Disabled,
}

/// Signs fal-style webhooks with our Ed25519 key (`FV_WEBHOOK_ED25519_KEY`).
#[derive(Clone)]
pub struct WebhookSigner {
    key: SigningKey,
    /// `X-Fal-Webhook-User-Id`.
    pub user_id: String,
}

impl std::fmt::Debug for WebhookSigner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebhookSigner").field("kid", &self.kid()).finish_non_exhaustive()
    }
}

impl WebhookSigner {
    pub fn from_seed(seed: [u8; 32], user_id: impl Into<String>) -> Self {
        Self { key: SigningKey::from_bytes(&seed), user_id: user_id.into() }
    }
    /// Parses a 32-byte seed given as 64 hex characters or base64(url).
    pub fn from_seed_str(s: &str, user_id: impl Into<String>) -> Result<Self, String> {
        let s = s.trim();
        let b = hex::decode(s)
            .or_else(|| base64::engine::general_purpose::STANDARD.decode(s).ok())
            .or_else(|| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(s).ok())
            .filter(|b| b.len() == 32)
            .ok_or_else(|| "FV_WEBHOOK_ED25519_KEY must be a 32-byte seed (hex or base64)".to_owned())?;
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&b);
        Ok(Self::from_seed(seed, user_id))
    }
    /// A random key (receivers must re-fetch the JWKS after a restart).
    pub fn random(user_id: impl Into<String>) -> Self {
        let a = uuid::Uuid::new_v4().into_bytes();
        let b = uuid::Uuid::new_v4().into_bytes();
        let mut seed = [0u8; 32];
        seed[..16].copy_from_slice(&a);
        seed[16..].copy_from_slice(&b);
        Self::from_seed(seed, user_id)
    }
    pub fn verifying_key(&self) -> ed25519_dalek::VerifyingKey {
        self.key.verifying_key()
    }
    /// Key id: the first 16 hex characters of SHA-256(public key).
    pub fn kid(&self) -> String {
        hex::encode(&Sha256::digest(self.key.verifying_key().as_bytes()))[..16].to_owned()
    }
    /// The signed message: `request_id\nuser_id\ntimestamp\nhex(sha256(body))`.
    pub fn message(&self, request_id: &str, timestamp: i64, body: &[u8]) -> String {
        format!(
            "{request_id}\n{}\n{timestamp}\n{}",
            self.user_id,
            hex::encode(&Sha256::digest(body))
        )
    }
    /// The four `X-Fal-Webhook-*` headers.
    pub fn headers(&self, request_id: &str, timestamp: i64, body: &[u8]) -> Vec<(String, String)> {
        let sig = self.key.sign(self.message(request_id, timestamp, body).as_bytes());
        vec![
            ("x-fal-webhook-request-id".into(), request_id.to_owned()),
            ("x-fal-webhook-user-id".into(), self.user_id.clone()),
            ("x-fal-webhook-timestamp".into(), timestamp.to_string()),
            ("x-fal-webhook-signature".into(), hex::encode(&sig.to_bytes())),
        ]
    }
    /// `/.well-known/jwks.json` body (OKP / Ed25519, `x` = base64url key).
    pub fn jwks(&self) -> serde_json::Value {
        let x = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(self.key.verifying_key().as_bytes());
        serde_json::json!({ "keys": [{
            "kty": "OKP", "crv": "Ed25519", "x": x, "kid": self.kid(), "use": "sig", "alg": "EdDSA"
        }]})
    }
}

/// The fal webhook body (fal §9.7). `error` is set for failures.
pub fn fal_webhook_body(
    request_id: &str,
    gateway_request_id: &str,
    ok: bool,
    payload: serde_json::Value,
    error: Option<&str>,
) -> serde_json::Value {
    let mut v = serde_json::json!({
        "request_id": request_id,
        "gateway_request_id": gateway_request_id,
        "status": if ok { "OK" } else { "ERROR" },
        "payload": payload,
    });
    if let Some(e) = error {
        v["error"] = e.into();
    }
    v
}

/// Renders the callback body for one API's jobs (MiniMax: `{"task": …}`; fal:
/// [`fal_webhook_body`]). `None` sends nothing for this change.
pub trait CallbackRender: Send + Sync + 'static {
    fn callback_body(&self, job: &Job, cx: &ViewCtx) -> Option<serde_json::Value>;
}

struct Item {
    job: Job,
    body: serde_json::Value,
}

/// Sends callbacks in order per job.
pub struct CallbackSender {
    transport: Option<Arc<dyn CallbackTransport>>,
    signer: Option<WebhookSigner>,
    pub target: TargetPolicy,
    pub fal: RetrySchedule,
    pub minimax: RetrySchedule,
    pub challenge_timeout: Duration,
    queues: Mutex<HashMap<JobId, mpsc::UnboundedSender<Item>>>,
    log: Mutex<Vec<(JobId, Delivery)>>,
}

impl std::fmt::Debug for CallbackSender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallbackSender")
            .field("enabled", &self.transport.is_some())
            .finish_non_exhaustive()
    }
}

impl CallbackSender {
    pub fn new(transport: Option<Arc<dyn CallbackTransport>>, signer: Option<WebhookSigner>) -> Self {
        Self {
            transport,
            signer,
            target: TargetPolicy::default(),
            fal: RetrySchedule::fal(),
            minimax: RetrySchedule::minimax(),
            challenge_timeout: Duration::from_secs(3),
            queues: Mutex::new(HashMap::new()),
            log: Mutex::new(Vec::new()),
        }
    }

    /// The reqwest transport when built with `fetch`, else none (callbacks
    /// are then dropped with a warning).
    pub fn default_transport() -> Option<Arc<dyn CallbackTransport>> {
        #[cfg(feature = "fetch")]
        {
            Some(Arc::new(HttpTransport))
        }
        #[cfg(not(feature = "fetch"))]
        {
            None
        }
    }

    pub fn signer(&self) -> Option<&WebhookSigner> {
        self.signer.as_ref()
    }

    /// Outcomes so far (most recent last; tests and metrics).
    pub fn deliveries(&self) -> Vec<(JobId, Delivery)> {
        self.log.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Queues the callback for `job`'s current state with `body` (from the
    /// adapter's [`CallbackRender`]). Jobs without a `callback` are ignored.
    pub fn dispatch(self: &Arc<Self>, job: Job, body: serde_json::Value) {
        if job.callback.is_none() {
            return;
        }
        let mut q = self.queues.lock().unwrap_or_else(|p| p.into_inner());
        let id = job.id;
        let item = Item { job, body };
        let item = match q.get(&id) {
            Some(tx) => match tx.send(item) {
                Ok(()) => return,
                Err(mpsc::error::SendError(item)) => item,
            },
            None => item,
        };
        let (tx, rx) = mpsc::unbounded_channel();
        let _ = tx.send(item);
        q.insert(id, tx);
        let me = self.clone();
        tokio::spawn(async move { me.worker(id, rx).await });
    }

    fn record(&self, id: JobId, d: Delivery) {
        self.log.lock().unwrap_or_else(|p| p.into_inner()).push((id, d));
    }

    async fn worker(self: Arc<Self>, id: JobId, mut rx: mpsc::UnboundedReceiver<Item>) {
        let mut verified: Option<bool> = None;
        let mut last: Option<JobStatus> = None;
        while let Some(Item { job, body }) = rx.recv().await {
            let status = job.status();
            let done = match job.callback.clone() {
                Some(CallbackSpec::MiniMax { url }) => {
                    if verified.is_none() {
                        let ok = self.minimax_challenge(&url).await;
                        if !ok {
                            tracing::warn!(job = %id, "MiniMax callback challenge failed; callbacks disabled");
                            self.record(id, Delivery::Permanent { attempts: 1, reason: "challenge failed".into() });
                        }
                        verified = Some(ok);
                    }
                    if verified == Some(true) && last != Some(status) {
                        last = Some(status);
                        let d = self.deliver(&url, Vec::new(), &body, &self.minimax).await;
                        let permanent = matches!(d, Delivery::Permanent { .. });
                        self.record(id, d);
                        if permanent {
                            verified = Some(false);
                        }
                    }
                    verified == Some(false) || status.is_terminal()
                }
                Some(CallbackSpec::FalWebhook { url }) => {
                    if status.is_terminal() {
                        let bytes = serde_json::to_vec(&body).unwrap_or_default();
                        let headers = match &self.signer {
                            Some(s) => s.headers(&job.external_id, time::OffsetDateTime::now_utc().unix_timestamp(), &bytes),
                            None => Vec::new(),
                        };
                        let d = self.deliver(&url, headers, &body, &self.fal).await;
                        self.record(id, d);
                        true
                    } else {
                        false
                    }
                }
                None => true,
            };
            if done {
                break;
            }
        }
        self.queues.lock().unwrap_or_else(|p| p.into_inner()).remove(&id);
    }

    /// POSTs `{"challenge": <random>}` and checks the echo within
    /// `challenge_timeout`.
    pub async fn minimax_challenge(&self, url: &Url) -> bool {
        let Some(t) = &self.transport else { return false };
        let challenge = crate::random_token();
        let body = Bytes::from(serde_json::to_vec(&serde_json::json!({ "challenge": challenge })).unwrap_or_default());
        let headers = vec![("content-type".to_owned(), "application/json".to_owned())];
        let r = tokio::time::timeout(self.challenge_timeout, t.post(url, &headers, body, &self.target)).await;
        match r {
            Ok(Ok(rep)) if (200..300).contains(&rep.status) => serde_json::from_slice::<serde_json::Value>(&rep.body)
                .ok()
                .and_then(|v| v.get("challenge").and_then(|c| c.as_str()).map(|c| c == challenge))
                .unwrap_or(false),
            _ => false,
        }
    }

    /// POSTs `body` as JSON on `schedule`.
    pub async fn deliver(
        &self,
        url: &Url,
        mut headers: Vec<(String, String)>,
        body: &serde_json::Value,
        schedule: &RetrySchedule,
    ) -> Delivery {
        let Some(t) = &self.transport else {
            tracing::warn!("callback dropped: no HTTP transport (build with `fetch`)");
            return Delivery::Disabled;
        };
        headers.push(("content-type".into(), "application/json".into()));
        let bytes = Bytes::from(serde_json::to_vec(body).unwrap_or_default());
        let start = tokio::time::Instant::now();
        let mut last_error = String::new();
        let mut attempts = 0u32;
        for attempt in 0..schedule.max_attempts() {
            if attempt > 0 {
                tokio::time::sleep(schedule.delays[attempt - 1]).await;
                if schedule.deadline.is_some_and(|d| start.elapsed() >= d) {
                    break;
                }
            }
            attempts += 1;
            let r = tokio::time::timeout(
                schedule.timeout(attempt),
                t.post(url, &headers, bytes.clone(), &self.target),
            )
            .await;
            match r {
                Ok(Ok(rep)) if (200..300).contains(&rep.status) => return Delivery::Delivered { attempts },
                Ok(Ok(rep)) if (300..400).contains(&rep.status) => {
                    return Delivery::Permanent { attempts, reason: format!("HTTP {}", rep.status) }
                }
                Ok(Ok(rep)) => last_error = format!("HTTP {}", rep.status),
                Ok(Err(PostError::Target(e))) => return Delivery::Permanent { attempts, reason: e },
                Ok(Err(PostError::Transport(e))) => last_error = e,
                Err(_) => last_error = "timed out".into(),
            }
        }
        Delivery::Exhausted { attempts, last_error }
    }
}

/// reqwest transport with the SSRF guard (pinned resolution, no redirects,
/// no proxy).
#[cfg(feature = "fetch")]
#[derive(Clone, Copy, Debug, Default)]
pub struct HttpTransport;

#[cfg(feature = "fetch")]
#[async_trait::async_trait]
impl CallbackTransport for HttpTransport {
    async fn post(
        &self,
        url: &Url,
        headers: &[(String, String)],
        body: Bytes,
        target: &TargetPolicy,
    ) -> Result<PostReply, PostError> {
        let addrs = crate::net::resolve_target(url, target)
            .await
            .map_err(|e| PostError::Target(e.to_string()))?;
        let mut b = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy();
        if let Some(url::Host::Domain(d)) = url.host() {
            b = b.resolve_to_addrs(d, &addrs);
        }
        let client = b.build().map_err(|e| PostError::Transport(e.to_string()))?;
        let mut req = client.post(url.clone()).body(body);
        for (k, v) in headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let resp = req.send().await.map_err(|e| PostError::Transport(e.to_string()))?;
        let status = resp.status().as_u16();
        let body = resp.bytes().await.unwrap_or_default();
        Ok(PostReply { status, body })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::tests::job;
    use ed25519_dalek::Verifier;
    use fastvideo_protocol::ProtocolId;
    use tokio::time::Instant;

    type Seen = (Instant, Vec<(String, String)>, serde_json::Value);

    /// Scripted receiver: pops one answer per POST and records the request.
    #[derive(Default)]
    struct Fake {
        script: Mutex<Vec<Result<PostReply, PostError>>>,
        seen: Mutex<Vec<Seen>>,
        echo: bool,
        hang: bool,
    }

    impl Fake {
        fn scripted(v: Vec<Result<PostReply, PostError>>) -> Arc<Self> {
            let mut v = v;
            v.reverse();
            Arc::new(Self { script: Mutex::new(v), ..Default::default() })
        }
        fn seen(&self) -> Vec<Seen> {
            self.seen.lock().unwrap().clone()
        }
    }

    fn rep(status: u16) -> Result<PostReply, PostError> {
        Ok(PostReply { status, body: Bytes::new() })
    }

    #[async_trait::async_trait]
    impl CallbackTransport for Fake {
        async fn post(&self, _u: &Url, h: &[(String, String)], body: Bytes, _t: &TargetPolicy) -> Result<PostReply, PostError> {
            let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
            self.seen.lock().unwrap().push((Instant::now(), h.to_vec(), v.clone()));
            if self.hang {
                tokio::time::sleep(Duration::from_secs(3600)).await;
            }
            if self.echo {
                if let Some(c) = v.get("challenge") {
                    return Ok(PostReply { status: 200, body: Bytes::from(serde_json::to_vec(&serde_json::json!({"challenge": c})).unwrap()) });
                }
            }
            self.script.lock().unwrap().pop().unwrap_or_else(|| rep(200))
        }
    }

    fn url() -> Url {
        Url::parse("https://hooks.example.com/cb").unwrap()
    }

    #[test]
    fn fal_schedule_shape() {
        let s = RetrySchedule::fal();
        assert_eq!(s.max_attempts(), 32);
        assert_eq!((s.first_timeout, s.retry_timeout), (Duration::from_secs(15), Duration::from_secs(120)));
        let secs: Vec<u64> = s.delays.iter().map(|d| d.as_secs()).collect();
        assert_eq!(&secs[..10], &[1, 2, 4, 8, 16, 32, 64, 128, 256, 300]);
        assert!(secs[10..].iter().all(|&d| d == 300));
    }

    #[tokio::test(start_paused = true)]
    async fn retry_schedule_is_followed() {
        let fake = Fake::scripted(vec![rep(500), Err(PostError::Transport("reset".into())), rep(503), rep(204)]);
        let s = CallbackSender::new(Some(fake.clone()), None);
        let d = s.deliver(&url(), vec![], &serde_json::json!({"a":1}), &RetrySchedule::minimax()).await;
        assert_eq!(d, Delivery::Delivered { attempts: 4 });
        let t: Vec<Instant> = fake.seen().iter().map(|x| x.0).collect();
        let gaps: Vec<u64> = t.windows(2).map(|w| (w[1] - w[0]).as_secs()).collect();
        assert_eq!(gaps, [1, 2, 4]);
    }

    #[tokio::test(start_paused = true)]
    async fn exhausted_permanent_timeout_deadline() {
        let s = CallbackSender::new(Some(Fake::scripted(vec![rep(500); 10])), None);
        let d = s.deliver(&url(), vec![], &serde_json::json!({}), &RetrySchedule::minimax()).await;
        assert_eq!(d, Delivery::Exhausted { attempts: 6, last_error: "HTTP 500".into() });

        let s = CallbackSender::new(Some(Fake::scripted(vec![rep(302)])), None);
        let d = s.deliver(&url(), vec![], &serde_json::json!({}), &RetrySchedule::fal()).await;
        assert_eq!(d, Delivery::Permanent { attempts: 1, reason: "HTTP 302".into() });

        let s = CallbackSender::new(Some(Fake::scripted(vec![Err(PostError::Target("private".into()))])), None);
        let d = s.deliver(&url(), vec![], &serde_json::json!({}), &RetrySchedule::fal()).await;
        assert!(matches!(d, Delivery::Permanent { attempts: 1, .. }));

        // Timeouts count as failures; the first attempt uses first_timeout.
        let hang = Arc::new(Fake { hang: true, ..Default::default() });
        let s = CallbackSender::new(Some(hang.clone()), None);
        let sched = RetrySchedule { first_timeout: Duration::from_secs(15), retry_timeout: Duration::from_secs(120), delays: vec![Duration::from_secs(1)], deadline: None };
        let t0 = Instant::now();
        let d = s.deliver(&url(), vec![], &serde_json::json!({}), &sched).await;
        assert_eq!(d, Delivery::Exhausted { attempts: 2, last_error: "timed out".into() });
        assert_eq!(t0.elapsed().as_secs(), 15 + 1 + 120);

        // The fal deadline (1 h) stops retries before 31 are spent.
        let fail = Fake::scripted(vec![rep(500); 40]);
        let s = CallbackSender::new(Some(fail.clone()), None);
        let d = s.deliver(&url(), vec![], &serde_json::json!({}), &RetrySchedule::fal()).await;
        let n = fail.seen().len();
        assert!(matches!(d, Delivery::Exhausted { .. }));
        assert!(n < 32 && n > 10, "{n}");
        // Delays sum to the deadline: 1+..+256 = 511, then 300 s steps.
        assert_eq!(n, 1 + 9 + ((3600 - 511) / 300) as usize);

        assert_eq!(CallbackSender::new(None, None).deliver(&url(), vec![], &serde_json::json!({}), &RetrySchedule::fal()).await, Delivery::Disabled);
    }

    #[test]
    fn webhook_signature_verifies() {
        let s = WebhookSigner::from_seed([7u8; 32], "fv-serve");
        let body = br#"{"request_id":"r1","status":"OK"}"#;
        let h = s.headers("r1", 1_790_000_000, body);
        let get = |k: &str| h.iter().find(|(n, _)| n == k).unwrap().1.clone();
        assert_eq!(get("x-fal-webhook-request-id"), "r1");
        assert_eq!(get("x-fal-webhook-user-id"), "fv-serve");
        assert_eq!(get("x-fal-webhook-timestamp"), "1790000000");
        let sig = hex::decode(&get("x-fal-webhook-signature")).unwrap();
        let sig = ed25519_dalek::Signature::from_slice(&sig).unwrap();
        // Verify with the key published in the JWKS.
        let jwks = s.jwks();
        let x = jwks["keys"][0]["x"].as_str().unwrap();
        let pk: [u8; 32] = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(x).unwrap().try_into().unwrap();
        let vk = ed25519_dalek::VerifyingKey::from_bytes(&pk).unwrap();
        let msg = format!("r1\nfv-serve\n1790000000\n{}", hex::encode(&Sha256::digest(body)));
        assert!(vk.verify(msg.as_bytes(), &sig).is_ok());
        assert!(vk.verify(b"tampered", &sig).is_err());
        assert_eq!(jwks["keys"][0]["crv"], "Ed25519");
        assert_eq!(jwks["keys"][0]["kid"], s.kid());
        // Seed parsing.
        let hexseed = hex::encode(&[7u8; 32]);
        assert_eq!(WebhookSigner::from_seed_str(&hexseed, "u").unwrap().kid(), s.kid());
        assert!(WebhookSigner::from_seed_str("short", "u").is_err());
    }

    async fn settle(s: &Arc<CallbackSender>) {
        for _ in 0..200 {
            if s.queues.lock().unwrap().is_empty() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("callback worker did not finish");
    }

    fn mm_job() -> Job {
        let mut j = job(ProtocolId::MiniMaxV2, "123456789012345678", time::OffsetDateTime::now_utc());
        j.callback = Some(CallbackSpec::MiniMax { url: url() });
        j
    }

    #[tokio::test(start_paused = true)]
    async fn minimax_challenge_then_each_status_change() {
        let fake = Arc::new(Fake { echo: true, ..Default::default() });
        let s = Arc::new(CallbackSender::new(Some(fake.clone()), None));
        let mut j = mm_job();
        s.dispatch(j.clone(), serde_json::json!({"task": {"status": "queued"}}));
        s.dispatch(j.clone(), serde_json::json!({"task": {"status": "queued", "dup": true}}));
        j.mark_running(time::OffsetDateTime::now_utc()).unwrap();
        s.dispatch(j.clone(), serde_json::json!({"task": {"status": "running"}}));
        j.mark_succeeded(time::OffsetDateTime::now_utc(), vec![], Default::default()).unwrap();
        s.dispatch(j.clone(), serde_json::json!({"task": {"status": "succeeded"}}));
        settle(&s).await;
        let seen: Vec<serde_json::Value> = fake.seen().into_iter().map(|x| x.2).collect();
        assert!(seen[0]["challenge"].as_str().unwrap().len() == 32);
        let statuses: Vec<&str> = seen[1..].iter().map(|v| v["task"]["status"].as_str().unwrap()).collect();
        assert_eq!(statuses, ["queued", "running", "succeeded"], "one POST per change, in order");
    }

    #[tokio::test(start_paused = true)]
    async fn minimax_wrong_or_slow_echo_disables() {
        // Wrong echo (the fake answers `{}`).
        let fake = Fake::scripted(vec![Ok(PostReply { status: 200, body: Bytes::from_static(b"{\"challenge\":\"nope\"}") })]);
        let s = Arc::new(CallbackSender::new(Some(fake.clone()), None));
        s.dispatch(mm_job(), serde_json::json!({"task": {}}));
        settle(&s).await;
        assert_eq!(fake.seen().len(), 1, "only the challenge");
        assert!(matches!(s.deliveries()[0].1, Delivery::Permanent { .. }));
        // No answer within 3 s.
        let hang = Arc::new(Fake { hang: true, ..Default::default() });
        let s = Arc::new(CallbackSender::new(Some(hang.clone()), None));
        let t0 = Instant::now();
        assert!(!s.minimax_challenge(&url()).await);
        assert_eq!(t0.elapsed(), Duration::from_secs(3));
    }

    #[tokio::test(start_paused = true)]
    async fn fal_webhook_only_on_completion_and_signed() {
        let fake = Fake::scripted(vec![]);
        let signer = WebhookSigner::from_seed([1u8; 32], "fv");
        let s = Arc::new(CallbackSender::new(Some(fake.clone()), Some(signer)));
        let mut j = job(ProtocolId::Fal, "req-1", time::OffsetDateTime::now_utc());
        j.callback = Some(CallbackSpec::FalWebhook { url: url() });
        s.dispatch(j.clone(), serde_json::json!({"ignored": true}));
        j.mark_cancelled(time::OffsetDateTime::now_utc()).unwrap();
        let body = fal_webhook_body("req-1", "req-1", false, serde_json::json!({"detail": "x"}), Some("cancelled"));
        s.dispatch(j, body.clone());
        settle(&s).await;
        let seen = fake.seen();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].2, body);
        assert!(seen[0].1.iter().any(|(k, v)| k == "x-fal-webhook-request-id" && v == "req-1"));
        assert_eq!(body["status"], "ERROR");

        // LTX (no callback spec): nothing at all.
        let before = fake.seen().len();
        s.dispatch(job(ProtocolId::LtxV2, "l", time::OffsetDateTime::now_utc()), serde_json::json!({}));
        settle(&s).await;
        assert_eq!(fake.seen().len(), before);
    }

    /// A real HTTP receiver (feature `fetch`): challenge echo + status POSTs.
    #[cfg(feature = "fetch")]
    #[tokio::test]
    async fn http_receiver_roundtrip() {
        use axum::routing::post;
        let got: Arc<Mutex<Vec<serde_json::Value>>> = Arc::default();
        let g2 = got.clone();
        let app = axum::Router::new().route("/cb", post(move |axum::Json(v): axum::Json<serde_json::Value>| {
            let g = g2.clone();
            async move {
                g.lock().unwrap().push(v.clone());
                match v.get("challenge") {
                    Some(c) => axum::Json(serde_json::json!({ "challenge": c })),
                    None => axum::Json(serde_json::json!({})),
                }
            }
        }));
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        let mut s = CallbackSender::new(CallbackSender::default_transport(), None);
        s.target.allow_private = true;
        let s = Arc::new(s);
        let mut j = mm_job();
        j.callback = Some(CallbackSpec::MiniMax { url: Url::parse(&format!("http://127.0.0.1:{port}/cb")).unwrap() });
        j.mark_cancelled(time::OffsetDateTime::now_utc()).unwrap();
        s.dispatch(j, serde_json::json!({"task": {"status": "cancelled"}}));
        settle(&s).await;
        let got = got.lock().unwrap().clone();
        assert_eq!(got.len(), 2);
        assert_eq!(got[1]["task"]["status"], "cancelled");

        // The guard refuses loopback by default: permanent, not retried.
        let strict = CallbackSender::new(CallbackSender::default_transport(), None);
        let d = strict.deliver(&Url::parse(&format!("http://127.0.0.1:{port}/cb")).unwrap(), vec![], &serde_json::json!({}), &RetrySchedule::fal()).await;
        assert!(matches!(d, Delivery::Permanent { attempts: 1, .. }), "{d:?}");
    }
}
