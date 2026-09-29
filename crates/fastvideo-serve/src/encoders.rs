//! H.264 encoder selection at startup (design §5.9).
//!
//! Every encoder setting — `[director] encoder`, `[reactor] h264`,
//! `[webrtc] encoder` (native `/fv/v1/streams` WHIP publishing,
//! `FV_STREAM_ENCODER`) and `[engine] post_encoder` (the crop re-encode) —
//! defaults to `auto`. [`resolve`] runs the NVENC encode probe
//! ([`fastvideo_media::video::auto_encoder`]) when any setting is `auto`,
//! logs the choice, and rewrites those settings to the concrete backend:
//! `nvenc` when the probe encoded, else the CPU fallback below. A probe failure that may
//! be transient (ffmpeg has `h264_nvenc` but opening it failed, as on a
//! serverless worker whose GPU is not ready) is retried once after 2 s; if
//! NVENC still fails, the fallback of each setting is logged at WARN
//! ([`Selection::describe`]). Explicit values are
//! kept as they are.
//!
//! Where OpenH264 cannot serve, the fallback is what that consumer can still
//! do: without OpenH264 in the build (feature `encoders`) the Reactor
//! runtime answers VP8 only (`off`), `/fv/v1/streams` keeps its CPU test
//! encoder (`x264-test`) and the director keeps `openh264`, which it then
//! knows it cannot encode: it prefers VP8 for every offer that has it, a
//! browser's H.264 + VP8 included (`vp8_fallback`; GPUs without NVENC such
//! as H100 and A100 land here). The crop
//! re-encode runs inside ffmpeg, which has no OpenH264, so its fallback is
//! ffmpeg's CPU test encoder (`cpu-test-x264`).

use fastvideo_media::video::{AutoEncoder, EncoderBackend};

use crate::config::Config;

/// What `auto` becomes for each setting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub director: String,
    pub reactor: String,
    pub streams: String,
    pub post: String,
}

impl Selection {
    /// The concrete values for `auto` given one probe outcome.
    pub fn new(auto: &AutoEncoder, openh264_compiled: bool) -> Self {
        let s = |d: &str, r: &str, st: &str, p: &str| Self {
            director: d.into(),
            reactor: r.into(),
            streams: st.into(),
            post: p.into(),
        };
        match auto.backend {
            EncoderBackend::Nvenc => s("nvenc", "nvenc", "nvenc", "nvenc"),
            _ if openh264_compiled => s("openh264", "openh264", "openh264", "cpu-test-x264"),
            _ => s("openh264", "off", "x264-test", "cpu-test-x264"),
        }
    }

    /// What each rewritten setting (`apply`'s names) now uses, for the log:
    /// `director.encoder=openh264, webrtc.encoder=x264-test (CPU test
    /// encoder), …`. Says what really runs, which is not OpenH264 for every
    /// consumer (see the module docs).
    pub fn describe(&self, settings: &[&str]) -> String {
        let note = |v: &str| match v {
            "off" => " (VP8 only)",
            "x264-test" | "cpu-test-x264" => " (CPU test encoder)",
            "openh264" if !EncoderBackend::OpenH264.compiled() => " (not in this build: answered with VP8 instead)",
            _ => "",
        };
        settings
            .iter()
            .filter_map(|name| {
                let v = match *name {
                    "director.encoder" => &self.director,
                    "reactor.h264" => &self.reactor,
                    "webrtc.encoder" => &self.streams,
                    "engine.post_encoder" => &self.post,
                    _ => return None,
                };
                Some(format!("{name}={v}{}", note(v)))
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Whether any encoder setting is `auto`.
pub fn wants_auto(c: &Config) -> bool {
    c.director.encoder == "auto"
        || c.reactor.h264 == "auto"
        || c.webrtc.encoder == "auto"
        || c.engine.post_encoder == "auto"
}

/// Rewrites every `auto` setting from `sel`; returns the names rewritten.
pub fn apply(c: &mut Config, sel: &Selection) -> Vec<&'static str> {
    let mut changed = Vec::new();
    for (name, slot, value) in [
        ("director.encoder", &mut c.director.encoder, &sel.director),
        ("reactor.h264", &mut c.reactor.h264, &sel.reactor),
        ("webrtc.encoder", &mut c.webrtc.encoder, &sel.streams),
        ("engine.post_encoder", &mut c.engine.post_encoder, &sel.post),
    ] {
        if slot == "auto" {
            slot.clone_from(value);
            changed.push(name);
        }
    }
    changed
}

/// The `auto` outcome for this process (probes once; blocking).
pub fn auto_selection() -> (AutoEncoder, Selection) {
    let auto = fastvideo_media::video::auto_encoder().clone();
    let sel = Selection::new(&auto, EncoderBackend::OpenH264.compiled());
    (auto, sel)
}

/// Resolves every `auto` encoder setting (see the module docs) and logs
/// the choice.
pub async fn resolve(c: &mut Config) {
    if !wants_auto(c) {
        return;
    }
    let (auto, sel) = match tokio::task::spawn_blocking(auto_selection).await {
        Ok(v) => v,
        Err(e) => {
            let auto = AutoEncoder::from_probe(Err(format!("probe task failed: {e}")));
            let sel = Selection::new(&auto, EncoderBackend::OpenH264.compiled());
            (auto, sel)
        }
    };
    let settings = apply(c, &sel);
    match &auto.nvenc_error {
        None => tracing::info!(
            ?settings,
            attempts = auto.attempts,
            "H.264 encoder: nvenc (auto: the NVENC encode probe succeeded)"
        ),
        // A GPU worker without a working NVENC (a serverless host whose
        // driver lacks the `video` capability, or no free NVENC session)
        // still serves, on the CPU fallback.
        Some(e) => {
            let fallback = sel.describe(&settings);
            tracing::warn!(
                ?settings,
                attempts = auto.attempts,
                director = %sel.director,
                reactor = %sel.reactor,
                streams = %sel.streams,
                post = %sel.post,
                nvenc_error = %e,
                "H.264 encoder: NVENC unavailable after the startup probe, falling back to: {fallback} (auto)"
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_follows_the_probe_and_keeps_explicit_values() {
        let ok = AutoEncoder::from_probe(Ok(()));
        let bad = AutoEncoder::from_probe(Err("no device".into()));
        assert_eq!(Selection::new(&ok, false).reactor, "nvenc");
        assert_eq!(Selection::new(&ok, false).post, "nvenc");
        let s = Selection::new(&bad, true);
        assert_eq!((s.director.as_str(), s.reactor.as_str(), s.streams.as_str()), ("openh264", "openh264", "openh264"));
        let s = Selection::new(&bad, false);
        assert_eq!((s.director.as_str(), s.reactor.as_str(), s.streams.as_str()), ("openh264", "off", "x264-test"));

        let mut c = Config::default();
        assert!(wants_auto(&c), "every encoder setting defaults to auto");
        c.reactor.h264 = "nvenc".into();
        let changed = apply(&mut c, &Selection::new(&bad, true));
        assert_eq!(changed, vec!["director.encoder", "webrtc.encoder", "engine.post_encoder"]);
        assert_eq!(c.director.encoder, "openh264");
        assert_eq!(c.reactor.h264, "nvenc", "explicit values are kept");
        assert_eq!(c.webrtc.encoder, "openh264");
        assert_eq!(c.engine.post_encoder, "cpu-test-x264");
        assert!(!wants_auto(&c));
    }

    /// The fallback log names what each rewritten setting really uses: a
    /// build without OpenH264 says x264-test / off, not "openh264".
    #[test]
    fn fallback_description_names_the_real_encoders() {
        let bad = AutoEncoder::from_probe(Err("no device".into()));
        let all = ["director.encoder", "reactor.h264", "webrtc.encoder", "engine.post_encoder"];
        let d = Selection::new(&bad, false).describe(&all);
        assert!(d.contains("reactor.h264=off (VP8 only)"), "{d}");
        assert!(d.contains("webrtc.encoder=x264-test (CPU test encoder)"), "{d}");
        assert!(d.contains("engine.post_encoder=cpu-test-x264 (CPU test encoder)"), "{d}");
        let d = Selection::new(&bad, true).describe(&["webrtc.encoder", "engine.post_encoder"]);
        assert!(d.starts_with("webrtc.encoder=openh264"), "{d}");
        assert!(!d.contains("director"), "only rewritten settings: {d}");
    }
}
