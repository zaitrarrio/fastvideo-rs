//! Endpoints with no engine path: `403 permission_error` (design §1.2, §4.5).
//!
//! `audio-to-video`, `retake`, `extend`, `video-to-video-hdr` and
//! `video-to-video-reframe` answer `403 permission_error` ("endpoint not
//! available for the account", a documented LTX type, ltx §1.5) on both
//! `/v1/*` and `/v2/*` after authentication. `GET /v2/{endpoint}/{id}` for
//! them is `404 not_found_error`: no such job can exist.

use fastvideo_protocol::{ApiError, GapId};

/// Path segments of the refused endpoints.
pub const STUB_ENDPOINTS: [&str; 5] = [
    "audio-to-video",
    "retake",
    "extend",
    "video-to-video-hdr",
    "video-to-video-reframe",
];

/// The refusal every stub answers.
pub fn refused() -> ApiError {
    ApiError::forbidden(GapId::LtxEndpoint.default_message())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{render, Api};

    #[test]
    fn stub_is_permission_error() {
        let r = render(&refused(), Api::V2, &Default::default());
        assert_eq!(r.status, 403);
        assert_eq!(
            r.json_body().unwrap(),
            &serde_json::json!({"type": "error", "error": {"type": "permission_error", "message": "endpoint not available for the account"}})
        );
    }
}
