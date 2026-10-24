//! Batch prompt expansion (docs/serve/prompt-upsampling.md §Batch): the
//! submit pipelines call [`expand_request`] after model resolution and
//! before negotiation, so the engine generates with the rewritten prompt.
//!
//! It rewrites only when all hold: the request carries a flag
//! ([`GenerationRequest::prompt_expansion`]), the server has an
//! `[upsampler]`, and the flag asks for it (explicitly, or as the
//! protocol's default with `default_on`). Otherwise nothing changes: the
//! flag stays an accepted no-op, as before upsampling existed. A failed or
//! timed-out rewrite falls back to the client's prompt and says so in the
//! job log.

use fastvideo_protocol::{ExpansionMode, Family, GenerationRequest, MediaKind, ModelCaps};
use fastvideo_upsampler::{Effort, PromptFamily};

use crate::ctx::ServeCtx;

/// The prompt guide of a model family.
pub fn prompt_family(f: Family) -> PromptFamily {
    match f {
        Family::H3 => PromptFamily::H3,
        Family::Ltx2 => PromptFamily::Ltx,
        Family::Wan => PromptFamily::Wan,
        Family::MmAudio | Family::Loopback => PromptFamily::Generic,
    }
}

/// What [`expand_request`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Expansion {
    /// The rewritten prompt (now `req.prompt`), for `Job::expanded_prompt`.
    pub expanded: Option<String>,
    /// A job log line (the rewrite, or why it fell back); `None` when no
    /// rewrite was attempted.
    pub log: Option<String>,
}

impl Expansion {
    pub fn attempted(&self) -> bool {
        self.log.is_some()
    }
}

/// Rewrites `req.prompt` with the server's upsampler when the request asks
/// for it (module docs). Image-conditioned requests get a text-only
/// rewrite that is told an unseen image sets the first frame.
pub async fn expand_request(
    ctx: &ServeCtx,
    req: &mut GenerationRequest,
    caps: &ModelCaps,
) -> Expansion {
    let (Some(pe), Some(svc)) = (req.prompt_expansion, ctx.upsampler()) else {
        return Expansion::default();
    };
    if !pe.wanted(svc.settings().default_on) || req.prompt.trim().is_empty() || req.edit.is_some() {
        return Expansion::default();
    }
    let effort = if pe.mode == ExpansionMode::Quality {
        Effort::Quality
    } else {
        Effort::Balanced
    };
    let family = prompt_family(caps.family);
    let image =
        !req.keyframes.is_empty() || req.references.iter().any(|r| r.kind == MediaKind::Image);
    match svc.rewrite(family, effort, image, &req.prompt).await {
        Ok(p) => {
            req.unnote_noop(pe.field);
            let before = std::mem::replace(&mut req.prompt, p.clone());
            Expansion {
                log: Some(format!(
                    "prompt expanded by the upsampler ({} `{}`, {}): {} -> {} characters",
                    pe.field,
                    pe.mode.as_str(),
                    if pe.explicit {
                        "requested"
                    } else {
                        "protocol default"
                    },
                    before.chars().count(),
                    p.chars().count()
                )),
                expanded: Some(p),
            }
        }
        Err(e) => Expansion {
            expanded: None,
            log: Some(format!(
                "prompt expansion failed ({e}); generating with the original prompt"
            )),
        },
    }
}
