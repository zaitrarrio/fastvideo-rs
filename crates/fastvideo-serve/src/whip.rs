//! WHIP encode geometry at the integration point (design §0 decision 2).
//!
//! Every WHIP target gets the model's **native** canvas and fps
//! (docs/serve/stream-outputs.md). `fastvideo-webrtc`'s [`EncodeProfile`]
//! names the H.264 level the SDP offer advertises (`42e01f`, level 3.1, for
//! Cloudflare as its docs ask; 4.0 for MediaMTX), and `fastvideo-media`'s
//! [`H264Config::for_publish`] encodes the canvas at the smallest level that
//! fits, never below that one. [`whip_h264`] is the one place a WHIP
//! publisher takes its encoder config from; [`check_profile`] asserts the
//! two layers agree.
//!
//! The 1280x720 Cloudflare cap this module used to apply (padded frame,
//! level 3.1) was removed: Cloudflare documents no resolution cap for WHIP
//! (its FFmpeg example is 1920x1080@30 baseline).

use fastvideo_media::video::{H264Config, PublishTarget};
use fastvideo_webrtc::profile::EncodeProfile;
use fastvideo_webrtc::whip::WhipTarget;

/// The media-layer publish target of a WHIP endpoint kind.
pub fn publish_target(t: WhipTarget) -> PublishTarget {
    match t {
        WhipTarget::Cloudflare => PublishTarget::Cloudflare,
        WhipTarget::Mediamtx => PublishTarget::Mediamtx,
    }
}

/// The encoder config for publishing a `width`x`height` model canvas at
/// `fps` to a WHIP endpoint of kind `target`: native, at the smallest level
/// that fits and at least the offered one.
pub fn whip_h264(target: WhipTarget, width: u32, height: u32, fps: u32) -> H264Config {
    let mut c = H264Config::for_publish(publish_target(target), width, height, fps);
    // Cloudflare recommends a 1 s GOP for WebRTC (GOP 30 at 30 fps), as the
    // VP8/VP9 path does; the live run had H.264 at the 2 s default.
    if target == WhipTarget::Cloudflare {
        c.gop_seconds = 1.0;
    }
    debug_assert!(check_profile(target.profile(), &c).is_ok());
    c
}

/// Checks an encoder config against the WHIP profile the SDP advertises:
/// the native canvas (no scaling), and a level no lower than the offered
/// one that fits the canvas.
pub fn check_profile(p: EncodeProfile, c: &H264Config) -> Result<(), String> {
    let level = c.resolved_level().map_err(|e| e.to_string())?;
    if level.idc() < p.h264_level.idc() {
        return Err(format!(
            "encoder level {level} is below the offered {:?}",
            p.h264_level
        ));
    }
    let want = p.output_size(c.input_width, c.input_height);
    if (c.width, c.height) != want {
        return Err(format!(
            "frame {}x{} is not the native {}x{}",
            c.width, c.height, want.0, want.1
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use fastvideo_media::h264::H264Level;

    use super::*;

    #[test]
    fn cloudflare_gets_the_native_canvas() {
        for (w, h, fps, lvl) in [
            (1344, 768, 24, H264Level::L3_2),
            (1920, 1088, 24, H264Level::L4_0),
            (1280, 704, 24, H264Level::L3_1),
            (832, 480, 16, H264Level::L3_1),
            (768, 1344, 24, H264Level::L3_2),
        ] {
            let c = whip_h264(WhipTarget::Cloudflare, w, h, fps);
            assert_eq!((c.width, c.height), (w, h), "{w}x{h}");
            assert!(!c.scales());
            assert_eq!(c.resolved_level().unwrap(), lvl, "{w}x{h}@{fps}");
            check_profile(WhipTarget::Cloudflare.profile(), &c).unwrap();
            assert_eq!(c.gop_frames(), fps, "a 1 s GOP for Cloudflare");
        }
        assert_eq!(
            whip_h264(WhipTarget::Mediamtx, 1344, 768, 24).gop_frames(),
            48
        );
    }

    #[test]
    fn mediamtx_keeps_native_resolution_at_4_0() {
        let c = whip_h264(WhipTarget::Mediamtx, 1344, 768, 24);
        assert_eq!((c.width, c.height), (1344, 768));
        assert!(!c.scales());
        assert_eq!(c.resolved_level().unwrap(), H264Level::L4_0);
        check_profile(WhipTarget::Mediamtx.profile(), &c).unwrap();
        // A scaled or under-levelled config is the mismatch this prevents.
        let mut scaled = H264Config::new(1344, 768, 24);
        (scaled.width, scaled.height) = (1280, 720);
        assert!(check_profile(WhipTarget::Cloudflare.profile(), &scaled).is_err());
        let mut low = H264Config::new(832, 480, 16);
        low.level = Some(H264Level::L3_1);
        assert!(check_profile(WhipTarget::Mediamtx.profile(), &low).is_err());
    }
}
