//! WHIP encode geometry at the integration point (design §0 decision 2).
//!
//! `fastvideo-webrtc`'s [`EncodeProfile::output_size`] answers the **picture**
//! size that fits the profile's box (1344x768 under Cloudflare → 1260x720),
//! while `fastvideo-media`'s [`H264Config::for_publish`] encodes a **frame**
//! of the capped canvas with the picture letterboxed inside it
//! (1260x720 centred in a 1280x720 frame). Cloudflare Stream should always
//! receive the padded 1280x720 frame, so every WHIP publisher takes its
//! encoder config from [`whip_h264`], which derives the frame from the media
//! layer and only takes the H.264 level and the size bound from the WebRTC
//! profile. [`check_profile`] asserts that the two agree on everything else.

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

/// The encoder config for publishing a `width`x`height` model canvas to a
/// WHIP endpoint of kind `target`: Cloudflare gets a padded frame capped
/// at 1280x720 (orientation kept) at level 3.1; MediaMTX the native canvas
/// at level 4.0.
pub fn whip_h264(target: WhipTarget, width: u32, height: u32, fps: u32) -> H264Config {
    let c = H264Config::for_publish(publish_target(target), width, height, fps);
    debug_assert!(check_profile(target.profile(), &c).is_ok());
    c
}

/// Checks an encoder config against the WHIP profile the SDP advertises:
/// same H.264 level, frame inside the profile's box, picture no larger than
/// `output_size`.
pub fn check_profile(p: EncodeProfile, c: &H264Config) -> Result<(), String> {
    let level = c.resolved_level().map_err(|e| e.to_string())?;
    if level.idc() != p.h264_level.idc() {
        return Err(format!("encoder level {level} differs from the WHIP profile {:?}", p.h264_level));
    }
    if let Some((mw, mh)) = p.max_size {
        let (bw, bh) = if c.input_width >= c.input_height { (mw, mh) } else { (mh, mw) };
        if c.width > bw || c.height > bh {
            return Err(format!("frame {}x{} exceeds the profile box {bw}x{bh}", c.width, c.height));
        }
    }
    let (pw, ph) = p.output_size(c.input_width, c.input_height);
    let (fw, fh) = (c.width, c.height);
    if pw > fw || ph > fh {
        return Err(format!("picture {pw}x{ph} does not fit the frame {fw}x{fh}"));
    }
    if !p.h264_level.fits(fw, fh, c.fps) {
        return Err(format!("frame {fw}x{fh}@{} exceeds the profile level", c.fps));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloudflare_always_gets_a_padded_1280x720_frame() {
        for (w, h) in [(1344, 768), (1920, 1080), (1280, 720), (1360, 768)] {
            let c = whip_h264(WhipTarget::Cloudflare, w, h, 24);
            assert_eq!((c.width, c.height), (1280, 720), "{w}x{h}");
            check_profile(WhipTarget::Cloudflare.profile(), &c).unwrap();
        }
        // The WebRTC profile's picture (1260x720) sits inside that frame.
        assert_eq!(WhipTarget::Cloudflare.profile().output_size(1344, 768), (1260, 720));
        let c = whip_h264(WhipTarget::Cloudflare, 1344, 768, 24);
        assert!(c.scales());
        assert_eq!(c.profile_level_id().unwrap(), "42e01f");
        // Portrait keeps its orientation; small canvases are not upscaled.
        let p = whip_h264(WhipTarget::Cloudflare, 768, 1344, 24);
        assert_eq!((p.width, p.height), (720, 1280));
        check_profile(WhipTarget::Cloudflare.profile(), &p).unwrap();
        let s = whip_h264(WhipTarget::Cloudflare, 832, 480, 24);
        assert_eq!((s.width, s.height), (832, 480));
        check_profile(WhipTarget::Cloudflare.profile(), &s).unwrap();
    }

    #[test]
    fn mediamtx_keeps_native_resolution() {
        let c = whip_h264(WhipTarget::Mediamtx, 1344, 768, 24);
        assert_eq!((c.width, c.height), (1344, 768));
        assert!(!c.scales());
        check_profile(WhipTarget::Mediamtx.profile(), &c).unwrap();
        // A 1260x720 frame (the WebRTC picture size) would be the mismatch
        // this module prevents: it is not what the media layer encodes.
        let cf = whip_h264(WhipTarget::Cloudflare, 1344, 768, 24);
        assert_ne!((cf.width, cf.height), WhipTarget::Cloudflare.profile().output_size(1344, 768));
    }
}
