//! Server-sent events from an [`SseSpec`] (design §4.4 fal `/status/stream`).
//!
//! The stream sends `spec.initial`, then follows `spec.follow`: for
//! `JobStatus`, one event per job change whose `data` is the endpoint's
//! `JobView::status_reply` JSON (compact), closing after the first terminal
//! status when `close_on_terminal`. Idle streams get `: keepalive` comments.

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use fastvideo_protocol::{JobView, ReplyBody, SseEvent, SseFollow, SseSpec};
use futures::stream::{self, Stream, StreamExt};

use crate::ctx::ServeCtx;

fn to_event(e: SseEvent) -> Event {
    let mut ev = Event::default();
    if let Some(n) = e.event {
        ev = ev.event(n);
    }
    if let Some(i) = e.id {
        ev = ev.id(i);
    }
    ev.data(e.data)
}

/// The events of `spec`, following job changes through `ctx`'s store.
pub fn sse_events(
    ctx: ServeCtx,
    spec: SseSpec,
    view: Option<Arc<dyn JobView>>,
) -> impl Stream<Item = SseEvent> + Send + 'static {
    let initial = stream::iter(spec.initial);
    let follow = match (spec.follow, view) {
        (Some(SseFollow::JobStatus { job, close_on_terminal }), Some(view)) => {
            let rx = ctx.jobs().watch(job);
            stream::unfold((ctx, view, rx, false), move |(ctx, view, rx, done)| async move {
                if done {
                    return None;
                }
                let mut rx = rx?;
                rx.changed().await.ok()?;
                rx.borrow_and_update();
                let j = ctx.jobs().get(job).await?;
                let reply = view.status_reply(&j, &ctx.view_ctx(false));
                let data = match reply.body {
                    ReplyBody::Json(v) => v.to_string(),
                    _ => return None,
                };
                let done = close_on_terminal && j.is_terminal();
                Some((SseEvent::data(data), (ctx, view, Some(rx), done)))
            })
            .boxed()
        }
        _ => stream::empty().boxed(),
    };
    initial.chain(follow)
}

/// An axum SSE response for `spec`.
pub fn sse_response(ctx: ServeCtx, spec: SseSpec, view: Option<Arc<dyn JobView>>) -> Response {
    let keepalive = spec.keepalive;
    let s = sse_events(ctx, spec, view).map(|e| Ok::<_, Infallible>(to_event(e)));
    let sse = Sse::new(s);
    match keepalive {
        Some(d) => sse.keep_alive(KeepAlive::new().interval(d)).into_response(),
        None => sse.into_response(),
    }
}
