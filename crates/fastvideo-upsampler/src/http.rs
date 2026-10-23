//! [`OpenAiChat`]: `POST {base_url}/chat/completions` on any
//! OpenAI-compatible server (the spec's `callModel` transport).

use std::sync::atomic::{AtomicU8, Ordering};

use serde_json::{json, Value};

use crate::chat::{clip, ChatBackend, ChatRequest, UpsampleError};

/// An OpenAI-compatible chat endpoint. `Debug` never prints the key.
pub struct OpenAiChat {
    client: reqwest::Client,
    url: String,
    api_key: Option<String>,
    model: String,
    /// `POST {root}/tokenize` (llama-server, vLLM), `root` being the base
    /// URL without a trailing `/v1`.
    tokenize_url: String,
    /// 0 unknown, 1 supported, 2 unsupported (the first failure decides).
    tokenize: AtomicU8,
}

impl std::fmt::Debug for OpenAiChat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiChat")
            .field("url", &self.url)
            .field("model", &self.model)
            .field(
                "api_key",
                &if self.api_key.is_some() {
                    "<redacted>"
                } else {
                    "<unset>"
                },
            )
            .finish()
    }
}

impl OpenAiChat {
    /// `base_url` is the API root (`https://api.openai.com/v1`,
    /// `http://127.0.0.1:8000/v1`, …); `/chat/completions` is appended.
    /// No key: no `authorization` header (a local server without auth).
    pub fn new(
        base_url: &str,
        api_key: Option<String>,
        model: &str,
    ) -> Result<Self, UpsampleError> {
        let base = base_url.trim().trim_end_matches('/');
        if !(base.starts_with("http://") || base.starts_with("https://")) {
            return Err(UpsampleError::Unavailable(format!(
                "base_url must be an http(s) URL, got `{base}`"
            )));
        }
        if model.trim().is_empty() {
            return Err(UpsampleError::Unavailable("no model configured".into()));
        }
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| UpsampleError::Unavailable(format!("HTTP client: {e}")))?;
        let root = base.strip_suffix("/v1").unwrap_or(base);
        Ok(Self {
            client,
            tokenize_url: format!("{root}/tokenize"),
            tokenize: AtomicU8::new(0),
            url: format!("{base}/chat/completions"),
            api_key: api_key.filter(|k| !k.trim().is_empty()),
            model: model.trim().to_owned(),
        })
    }

    /// The request body (the spec's `JSON.stringify({...})`).
    pub fn body(&self, req: &ChatRequest) -> Value {
        // A vision request: the text, then one `image_url` part per image.
        let user = if req.images.is_empty() {
            json!(req.user)
        } else {
            let mut parts = vec![json!({"type": "text", "text": req.user})];
            parts.extend(
                req.images
                    .iter()
                    .map(|u| json!({"type": "image_url", "image_url": {"url": u}})),
            );
            Value::Array(parts)
        };
        let mut body = json!({
            "model": self.model,
            "temperature": req.temperature,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": user},
            ],
        });
        if let Some(rf) = &req.response_format {
            body["response_format"] = rf.clone();
        }
        body
    }
}

/// `choices[0].message.content` of a chat-completions answer.
pub fn content_of(v: &Value) -> Result<String, UpsampleError> {
    match v
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
    {
        Some(s) if !s.is_empty() => Ok(s.to_owned()),
        _ => Err(UpsampleError::NoContent),
    }
}

#[async_trait::async_trait]
impl ChatBackend for OpenAiChat {
    async fn complete(&self, req: &ChatRequest) -> Result<String, UpsampleError> {
        let mut rb = self.client.post(&self.url).json(&self.body(req));
        if let Some(k) = &self.api_key {
            rb = rb.bearer_auth(k);
        }
        let resp = rb
            .send()
            .await
            .map_err(|e| UpsampleError::Transport(e.without_url().to_string()))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| UpsampleError::Transport(e.without_url().to_string()))?;
        if !status.is_success() {
            return Err(UpsampleError::Status {
                status: status.as_u16(),
                body: clip(&text, 500),
            });
        }
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| UpsampleError::BadOutput(format!("the answer is not JSON: {e}")))?;
        content_of(&v)
    }

    fn describe(&self) -> String {
        format!("{} ({})", self.url, self.model)
    }

    /// llama-server answers `{"tokens": [...]}` to `{"content": text}`,
    /// vLLM `{"count": n, ...}` to `{"model", "prompt"}`; both fields are
    /// sent. Any failure marks the server unsupported for this process.
    async fn tokenize(&self, text: &str) -> Option<usize> {
        if self.tokenize.load(Ordering::Relaxed) == 2 {
            return None;
        }
        let n = self.tokenize_once(text).await;
        self.tokenize
            .store(if n.is_some() { 1 } else { 2 }, Ordering::Relaxed);
        if n.is_none() {
            tracing::info!(url = %self.tokenize_url, "upsampler: the LLM server has no usable /tokenize; token counts are estimated");
        }
        n
    }
}

impl OpenAiChat {
    async fn tokenize_once(&self, text: &str) -> Option<usize> {
        let mut rb = self
            .client
            .post(&self.tokenize_url)
            .timeout(std::time::Duration::from_secs(10))
            .json(&json!({"content": text, "prompt": text, "model": self.model, "add_special": false}));
        if let Some(k) = &self.api_key {
            rb = rb.bearer_auth(k);
        }
        let resp = rb.send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let v: Value = resp.json().await.ok()?;
        tokens_of(&v)
    }
}

/// The token count of a `/tokenize` answer (`count`, else `tokens`).
pub fn tokens_of(v: &Value) -> Option<usize> {
    if let Some(n) = v.get("count").and_then(Value::as_u64) {
        return Some(n as usize);
    }
    v.get("tokens").and_then(Value::as_array).map(Vec::len)
}
