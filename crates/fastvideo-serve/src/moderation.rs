//! `[moderation]`: the NCII prompt guard as an admission check (design §10
//! R8; docs/ports/h3-turbo-loras.md "NCII guard").
//!
//! [`NciiModerator`] runs `hfmlsoc/ncii-light-guard-v01` (MIT, the plain-Rust
//! CPU port in [`fastvideo_models::ncii`]) on the prompts its scope covers:
//! `keyframe` (requests with an image or video input, the MiniMax-H3-Turbo-Lora
//! Space's rule) or `all`. A prompt whose `ncii` probability reaches the
//! threshold is refused with `prompt refused (ncii <score>)` as
//! [`ApiError::content_filtered`], which every API renders as its own
//! content-policy error; nothing reaches the engine. Classification runs on
//! the blocking pool (at most `ncii_concurrency` at once), never on the async
//! workers, and not on the GPU.
//!
//! Logging: the prompt is never logged. A refusal logs its score, token count,
//! character count and the first 12 hex of the prompt's SHA-256 (to match
//! repeated attempts), as the rest of fv-serve logs no prompt text.

use std::path::Path;
use std::sync::Arc;

use fastvideo_models::ncii::NciiGuard;
use fastvideo_protocol::{ApiError, ModerationScope, PromptModerator};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

use crate::config::{Env, ModerationCfg};

/// The NCII guard behind the [`PromptModerator`] seam.
pub struct NciiModerator {
    guard: Arc<NciiGuard>,
    scope: ModerationScope,
    threshold: f32,
    permits: Arc<Semaphore>,
}

impl NciiModerator {
    pub fn new(guard: Arc<NciiGuard>, scope: ModerationScope, threshold: f32, concurrency: usize) -> Self {
        Self { guard, scope, threshold, permits: Arc::new(Semaphore::new(concurrency.max(1))) }
    }

    /// Loads the guard from `dir` (blocking: about 1.4 s and 0.75 GB of RAM).
    pub fn load(dir: &Path, scope: ModerationScope, threshold: f32, concurrency: usize) -> Result<Self, String> {
        let guard = NciiGuard::load(dir).map_err(|e| {
            format!(
                "[moderation] ncii = {}: the guard cannot be loaded from {} ({e}); fetch the R2 tree {} (or set ncii_dir)",
                scope.as_str(),
                dir.display(),
                fastvideo_models::ncii::NCII_TREE
            )
        })?;
        Ok(Self::new(Arc::new(guard), scope, threshold, concurrency))
    }
}

/// The first 12 hex of the prompt's SHA-256 (logged instead of the prompt).
pub fn prompt_digest(prompt: &str) -> String {
    let d = Sha256::digest(prompt.as_bytes());
    d.iter().take(6).map(|b| format!("{b:02x}")).collect()
}

/// The refusal every API renders (the Space's `prompt refused (ncii 0.87)`).
pub fn refusal(score: f32) -> ApiError {
    ApiError::content_filtered(format!(
        "prompt refused (ncii {score:.2}): this prompt was flagged by a content filter and was not run"
    ))
}

#[async_trait::async_trait]
impl PromptModerator for NciiModerator {
    fn scope(&self) -> ModerationScope {
        self.scope
    }

    async fn check(&self, prompt: &str) -> Result<(), ApiError> {
        let _permit = self
            .permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| ApiError::internal("ncii guard: closed"))?;
        let guard = self.guard.clone();
        let text = prompt.to_owned();
        let started = std::time::Instant::now();
        let verdict = tokio::task::spawn_blocking(move || guard.classify(&text))
            .await
            .map_err(|e| ApiError::internal(format!("ncii guard: {e}")))?
            // Fail closed: a prompt the guard cannot score is not run.
            .map_err(|e| ApiError::internal(format!("ncii guard: {e}")))?;
        let ms = started.elapsed().as_millis() as u64;
        if verdict.ncii >= self.threshold {
            metrics::counter!("fv_moderation_total", "check" => "ncii", "result" => "refused").increment(1);
            tracing::warn!(
                score = verdict.ncii,
                tokens = verdict.tokens,
                prompt_chars = prompt.chars().count(),
                prompt_sha256 = %prompt_digest(prompt),
                ms,
                "ncii guard: prompt refused"
            );
            return Err(refusal(verdict.ncii));
        }
        metrics::counter!("fv_moderation_total", "check" => "ncii", "result" => "passed").increment(1);
        tracing::debug!(score = verdict.ncii, tokens = verdict.tokens, ms, "ncii guard: passed");
        Ok(())
    }
}

/// The moderator `[moderation]` configures, or `None` when it is off.
pub fn from_config(c: &ModerationCfg, env: &dyn Env) -> anyhow::Result<Option<Arc<dyn PromptModerator>>> {
    let scope = c.scope()?;
    if scope == ModerationScope::Off {
        return Ok(None);
    }
    let dir = c.ncii_dir(env);
    let m = NciiModerator::load(&dir, scope, c.ncii_threshold, c.ncii_concurrency).map_err(anyhow::Error::msg)?;
    tracing::info!(
        scope = scope.as_str(),
        threshold = c.ncii_threshold,
        dir = %dir.display(),
        "ncii guard loaded ({} @ {}, MIT, CPU)",
        fastvideo_models::ncii::NCII_REPO,
        &fastvideo_models::ncii::NCII_REVISION[..7]
    );
    Ok(Some(Arc::new(m)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_protocol::ErrorKind;

    #[test]
    fn refusal_is_content_filtered_with_the_space_message() {
        let e = refusal(0.8749);
        assert_eq!(e.kind, ErrorKind::ContentFiltered);
        assert!(e.message.starts_with("prompt refused (ncii 0.87)"), "{}", e.message);
    }

    #[test]
    fn digest_is_short_and_stable() {
        assert_eq!(prompt_digest("abc"), "ba7816bf8f01");
        assert_eq!(prompt_digest("abc").len(), 12);
    }

    struct NoEnv;
    impl Env for NoEnv {
        fn var(&self, _: &str) -> Option<String> {
            None
        }
    }

    #[test]
    fn off_builds_nothing_and_a_missing_guard_fails_closed() {
        let off = ModerationCfg::default();
        assert!(from_config(&off, &NoEnv).unwrap().is_none());
        let on = ModerationCfg {
            ncii: "keyframe".into(),
            ncii_dir: Some("/nonexistent/ncii".into()),
            ..ModerationCfg::default()
        };
        let err = from_config(&on, &NoEnv).err().unwrap().to_string();
        assert!(err.contains("auxiliary/ncii-light-guard"), "{err}");
        let bad = ModerationCfg { ncii: "sometimes".into(), ..ModerationCfg::default() };
        assert!(bad.validate().is_err());
        let thr = ModerationCfg { ncii_threshold: 0.0, ..ModerationCfg::default() };
        assert!(thr.validate().is_err());
        assert_eq!(
            ModerationCfg::default().ncii_dir(&NoEnv),
            std::path::PathBuf::from("/workspace/weights/auxiliary/ncii-light-guard")
        );
    }
}
