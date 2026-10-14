//! Prompt moderation shared by every API (design §10 R8): an optional
//! admission check the server runs before any work. The deployed checker is
//! the NCII prompt guard (`fastvideo_models::ncii`, fv-serve
//! `[moderation] ncii`; docs/ports/h3-turbo-loras.md "NCII guard"). A refusal
//! is [`ApiError::content_filtered`], which each API renders as its own
//! content-policy error (fal `content_policy_violation`, MiniMax `1026`,
//! LTX `content_filtered_error`, OpenAI 400, native `content_filtered`); the
//! director answers `prompt_rejected{reason:"content_policy"}` (or the session
//! error `content_policy` at `configure`) and Reactor stops the session with
//! its `moderation` notice.

use serde::{Deserialize, Serialize};

use crate::{ApiError, GenerationRequest, MediaKind};

/// Which requests a moderator looks at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModerationScope {
    /// Nothing is checked (the default).
    #[default]
    Off,
    /// Requests that carry visual input a prompt could edit (a keyframe, an
    /// image or video reference, a source video): the edit-on-a-real-photo
    /// case the `MiniMax-H3-Turbo-Lora` Space guards.
    Keyframe,
    /// Every prompt.
    All,
}

impl ModerationScope {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "off" | "false" | "none" => Ok(Self::Off),
            "keyframe" | "keyframes" | "image" => Ok(Self::Keyframe),
            "all" | "on" | "true" => Ok(Self::All),
            other => Err(format!("moderation scope {other:?}: expected off | keyframe | all")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Keyframe => "keyframe",
            Self::All => "all",
        }
    }

    /// Whether a prompt with (or without) visual input is checked.
    pub fn applies(self, has_visual_input: bool) -> bool {
        match self {
            Self::Off => false,
            Self::Keyframe => has_visual_input,
            Self::All => true,
        }
    }
}

/// Whether a request carries visual input: keyframes, image or video
/// references, or a source video to edit.
pub fn has_visual_input(req: &GenerationRequest) -> bool {
    !req.keyframes.is_empty()
        || req
            .references
            .iter()
            .any(|r| matches!(r.kind, MediaKind::Image | MediaKind::Video))
        || req.edit.is_some()
}

/// A prompt checker. `check` may block on CPU work; implementations move it
/// off the async workers themselves.
#[async_trait::async_trait]
pub trait PromptModerator: Send + Sync + 'static {
    /// The requests this moderator looks at.
    fn scope(&self) -> ModerationScope;
    /// `Err(ApiError::content_filtered(..))` refuses the prompt.
    async fn check(&self, prompt: &str) -> Result<(), ApiError>;
}

/// The check every API runs: `Ok` when the moderator's scope does not cover
/// this input or the prompt is empty.
pub async fn moderate(
    moderator: &dyn PromptModerator,
    prompt: &str,
    has_visual_input: bool,
) -> Result<(), ApiError> {
    if !moderator.scope().applies(has_visual_input) || prompt.trim().is_empty() {
        return Ok(());
    }
    moderator.check(prompt).await
}

/// [`moderate`] over a whole request (the prompt; the negative prompt is
/// not an instruction and is not checked).
pub async fn moderate_request(moderator: &dyn PromptModerator, req: &GenerationRequest) -> Result<(), ApiError> {
    moderate(moderator, &req.prompt, has_visual_input(req)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Anchor, ErrorKind, Keyframe, MediaRef, ProtocolId};

    struct Refuse(ModerationScope);

    #[async_trait::async_trait]
    impl PromptModerator for Refuse {
        fn scope(&self) -> ModerationScope {
            self.0
        }
        async fn check(&self, _prompt: &str) -> Result<(), ApiError> {
            Err(ApiError::content_filtered("prompt refused (ncii 0.90)"))
        }
    }

    fn block<F: std::future::Future>(f: F) -> F::Output {
        // A ready future: the test moderator never awaits.
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        let mut f = std::pin::pin!(f);
        match f.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => v,
            std::task::Poll::Pending => panic!("pending"),
        }
    }

    #[test]
    fn scopes() {
        assert_eq!(ModerationScope::parse("keyframe").unwrap(), ModerationScope::Keyframe);
        assert_eq!(ModerationScope::parse(" OFF ").unwrap(), ModerationScope::Off);
        assert_eq!(ModerationScope::parse("all").unwrap(), ModerationScope::All);
        assert!(ModerationScope::parse("sometimes").is_err());
        assert!(!ModerationScope::Keyframe.applies(false));
        assert!(ModerationScope::Keyframe.applies(true));
        assert!(ModerationScope::All.applies(false));
        assert!(!ModerationScope::Off.applies(true));
    }

    #[test]
    fn keyframe_scope_checks_image_requests_only() {
        let mut req = GenerationRequest::text(ProtocolId::Fal, "h3-larry-turbo", "a prompt");
        let m = Refuse(ModerationScope::Keyframe);
        assert!(block(moderate_request(&m, &req)).is_ok());
        req.keyframes.push(Keyframe { at: Anchor::First, image: MediaRef::Http("https://x/y.png".parse().unwrap()) });
        assert!(has_visual_input(&req));
        let e = block(moderate_request(&m, &req)).unwrap_err();
        assert_eq!(e.kind, ErrorKind::ContentFiltered);
        assert!(block(moderate(&m, "  ", true)).is_ok());
        assert!(block(moderate(&Refuse(ModerationScope::All), "x", false)).is_err());
    }
}
