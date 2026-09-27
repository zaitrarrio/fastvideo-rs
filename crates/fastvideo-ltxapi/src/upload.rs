//! `POST /v1/upload` and `ltx://uploads/<token>` refs (design §4.5, ltx §2.10).
//!
//! The reply points the client at our own `PUT /uploads/{token}` (served by
//! serve-kit, which accepts and ignores the `x-goog-*` headers clients copy
//! from upstream examples) and hands out `ltx://uploads/<token>` for
//! `image_uri` / `last_frame_uri`. The signed URL lives 1 h, the file 24 h,
//! uploads are capped at 200 MB (serve-kit `UploadStore` defaults), and
//! generation applies the per-kind LTX limits (images 15 MB).
//! `required_headers` is empty: nothing has to be copied onto the `PUT`.

use fastvideo_protocol::HttpReply;
use fastvideo_serve_kit::UploadTicket;
use serde_json::{json, Value};

use crate::v2::fmt_ts;

/// The `UploadResponse` body.
pub fn upload_body(t: &UploadTicket) -> Value {
    json!({
        "upload_url": t.upload_url.as_str(),
        "storage_uri": t.ltx_storage_uri(),
        "expires_at": fmt_ts(t.expires_at),
        "required_headers": {},
    })
}

/// `200 UploadResponse`.
pub fn upload_reply(t: &UploadTicket) -> HttpReply {
    HttpReply::json(200, upload_body(t))
}
