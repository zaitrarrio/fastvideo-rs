//! Canvas scaling before encode (design §0 decision 2).
//!
//! Cloudflare WHIP caps H3 streams at 1280×720 (level 3.1); MediaMTX and
//! peer WebRTC keep the native canvas. [`ScaleMode::Fit`] keeps the aspect
//! ratio and pads with black (1344×768 → 1260×720 centred in 1280×720);
//! [`ScaleMode::Stretch`] fills the target exactly.
//!
//! The ffmpeg-based encoders do this with an ffmpeg filter
//! ([`ffmpeg_filter`]); the in-process OpenH264 test backend uses
//! [`scale_rgb`], which computes the same geometry.

use bytes::Bytes;

use crate::av::RgbFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScaleMode {
    /// Keep the aspect ratio, pad the rest with black.
    #[default]
    Fit,
    /// Fill the target exactly (aspect may change slightly).
    Stretch,
}

/// Where the fitted picture sits inside the output: `(x, y, w, h)`, all even.
pub fn fit_rect(in_w: u32, in_h: u32, out_w: u32, out_h: u32, mode: ScaleMode) -> (u32, u32, u32, u32) {
    if mode == ScaleMode::Stretch || in_w == 0 || in_h == 0 {
        return (0, 0, out_w, out_h);
    }
    let s = (f64::from(out_w) / f64::from(in_w)).min(f64::from(out_h) / f64::from(in_h));
    let even = |v: f64, cap: u32| (((v.round() as u32) / 2) * 2).clamp(2, cap);
    let w = even(f64::from(in_w) * s, out_w);
    let h = even(f64::from(in_h) * s, out_h);
    (((out_w - w) / 2) & !1, ((out_h - h) / 2) & !1, w, h)
}

/// The largest canvas with the input's orientation that fits `cap`
/// (landscape cap `cw×ch`; portrait inputs get `ch×cw`). Inputs already
/// inside the cap are returned unchanged.
pub fn capped_canvas(in_w: u32, in_h: u32, cap: (u32, u32)) -> (u32, u32) {
    let (cw, ch) = if in_w >= in_h { cap } else { (cap.1, cap.0) };
    if in_w <= cw && in_h <= ch { (in_w, in_h) } else { (cw, ch) }
}

/// The ffmpeg `-vf` chain that maps `in` to `out` (None when equal).
pub fn ffmpeg_filter(in_w: u32, in_h: u32, out_w: u32, out_h: u32, mode: ScaleMode) -> Option<String> {
    if (in_w, in_h) == (out_w, out_h) {
        return None;
    }
    let (x, y, w, h) = fit_rect(in_w, in_h, out_w, out_h, mode);
    Some(if (w, h) == (out_w, out_h) {
        format!("scale={out_w}:{out_h}:flags=bicubic,setsar=1")
    } else {
        format!("scale={w}:{h}:flags=bicubic,pad={out_w}:{out_h}:{x}:{y}:black,setsar=1")
    })
}

/// Bilinear RGB24 scale into `out_w×out_h` (CPU; used by the OpenH264 test
/// backend and for thumbnails).
pub fn scale_rgb(src: &RgbFrame, out_w: u32, out_h: u32, mode: ScaleMode) -> RgbFrame {
    if (src.width, src.height) == (out_w, out_h) {
        return src.clone();
    }
    let (ox, oy, w, h) = fit_rect(src.width, src.height, out_w, out_h, mode);
    let mut out = vec![0u8; out_w as usize * out_h as usize * 3];
    let (sw, sh) = (src.width as usize, src.height as usize);
    let d = &src.data;
    let fx = sw as f32 / w as f32;
    let fy = sh as f32 / h as f32;
    for y in 0..h as usize {
        let syf = ((y as f32 + 0.5) * fy - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = syf.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = syf - y0 as f32;
        let row = (oy as usize + y) * out_w as usize;
        for x in 0..w as usize {
            let sxf = ((x as f32 + 0.5) * fx - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = sxf.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = sxf - x0 as f32;
            let o = (row + ox as usize + x) * 3;
            for c in 0..3 {
                let p = |xx: usize, yy: usize| f32::from(d[(yy * sw + xx) * 3 + c]);
                let top = p(x0, y0) * (1.0 - wx) + p(x1, y0) * wx;
                let bot = p(x0, y1) * (1.0 - wx) + p(x1, y1) * wx;
                out[o + c] = (top * (1.0 - wy) + bot * wy).round() as u8;
            }
        }
    }
    RgbFrame { width: out_w, height: out_h, data: Bytes::from(out), index: src.index }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h3_to_cloudflare_720p() {
        assert_eq!(capped_canvas(1344, 768, (1280, 720)), (1280, 720));
        assert_eq!(capped_canvas(768, 1344, (1280, 720)), (720, 1280));
        assert_eq!(capped_canvas(832, 480, (1280, 720)), (832, 480));
        assert_eq!(fit_rect(1344, 768, 1280, 720, ScaleMode::Fit), (10, 0, 1260, 720));
        assert_eq!(fit_rect(768, 1344, 720, 1280, ScaleMode::Fit), (0, 10, 720, 1260));
        assert_eq!(fit_rect(1344, 768, 1280, 720, ScaleMode::Stretch), (0, 0, 1280, 720));
        assert_eq!(
            ffmpeg_filter(1344, 768, 1280, 720, ScaleMode::Fit).unwrap(),
            "scale=1260:720:flags=bicubic,pad=1280:720:10:0:black,setsar=1"
        );
        assert_eq!(ffmpeg_filter(1344, 768, 1280, 720, ScaleMode::Stretch).unwrap(), "scale=1280:720:flags=bicubic,setsar=1");
        assert_eq!(ffmpeg_filter(1344, 768, 1344, 768, ScaleMode::Fit), None);
    }

    #[test]
    fn scale_rgb_fit_pads_black_and_keeps_colour() {
        let src = RgbFrame::solid(1344, 768, [200, 100, 50], 3);
        let out = scale_rgb(&src, 1280, 720, ScaleMode::Fit);
        assert_eq!((out.width, out.height, out.index), (1280, 720, 3));
        assert_eq!(out.pixel(0, 0), Some([0, 0, 0])); // pad
        assert_eq!(out.pixel(9, 360), Some([0, 0, 0]));
        assert_eq!(out.pixel(10, 0), Some([200, 100, 50]));
        assert_eq!(out.pixel(640, 360), Some([200, 100, 50]));
        assert_eq!(out.pixel(1269, 719), Some([200, 100, 50]));
        assert_eq!(out.pixel(1270, 719), Some([0, 0, 0]));
        let st = scale_rgb(&src, 1280, 720, ScaleMode::Stretch);
        assert_eq!(st.pixel(0, 0), Some([200, 100, 50]));
    }
}
