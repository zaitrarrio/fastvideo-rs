//! Callback challenge and status posts (design §4.3, research §1.6).
//!
//! With `callback_url` set, serve-kit's `CallbackSender` first POSTs
//! `{"challenge": "<random>"}`; the receiver must echo `{"challenge": <same>}`
//! within 3 s, otherwise no callbacks are sent for that task. After that it
//! POSTs `{"task": VideoTask}` (the query body, [`crate::task_json`]) on
//! every status change, in order: `queued`, `running`, then one of
//! `succeeded` / `failed` / `cancelled`. The target passes the SSRF guard
//! at create time (400) and again at delivery.
//!
//! The binary registers the renderer:
//!
//! ```ignore
//! ServeCtx::builder(cfg, engine)
//!     .renderer(ProtocolId::MiniMaxV2, fastvideo_minimax::callback_renderer(&mm))
//! ```

use std::sync::Arc;

use fastvideo_serve_kit::CallbackRender;

use crate::MiniMax;

/// The `CallbackRender` for `ProtocolId::MiniMaxV2` jobs.
pub fn callback_renderer(mm: &MiniMax) -> Arc<dyn CallbackRender> {
    Arc::new(mm.view())
}
