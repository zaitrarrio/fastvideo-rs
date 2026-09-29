//! Shared caps and request fixtures for the fastvideo-protocol tests.
#![allow(dead_code)]

use std::collections::BTreeSet;
use std::path::PathBuf;

use fastvideo_protocol::*;

/// FastH3 (fl2va resident, no ref2va).
pub fn h3() -> ModelCaps {
    ModelCaps::h3("fasth3", false)
}

/// FastH3 with ref2va co-resident.
pub fn h3_ref2va() -> ModelCaps {
    ModelCaps::h3("fasth3", true)
}

/// LTX-2.3 distilled two-stage: T2V + I2V, 24 fps only, 8k+1 over 6..20 s,
/// pad-and-crop to 64.
pub fn ltx23() -> ModelCaps {
    ModelCaps {
        id: ModelId::new("ltx2_distilled_23"),
        family: Family::Ltx2,
        served_names: vec!["ltx-2-3-fast".into()],
        tasks: [Task::T2V, Task::I2V].into_iter().collect(),
        audio: Some(AudioCaps {
            native_rate: 24_000,
            channels: 2,
            via_sidecar: false,
        }),
        fps: FpsCaps::fixed(24),
        frames: FrameGrid::new(8, 1, 145, 481, 145),
        canvas: CanvasCaps {
            multiple: 64,
            max_area: 3840 * 2176,
            aspect: (0.25, 4.0),
            short_edges: vec![1080, 720, 1440, 2160],
            pad_and_crop: true,
            hd: None,
        },
        refs: RefLimits::none(),
        stream: Some(StreamCaps::Clip {
            min_s: 6.0,
            max_s: 20.0,
        }),
        knobs: KnobCaps {
            seed: true,
            negative: true,
            steps: true,
            guidance: true,
            ..KnobCaps::default()
        },
        resident: true,
        tier: None,
        recipe: None,
    }
}

/// LTX-2.5: T2V only until E5.
pub fn ltx25() -> ModelCaps {
    let mut c = ltx23();
    c.id = ModelId::new("ltx2_distilled_25");
    c.served_names = vec!["ltx-2-5-fast".into()];
    c.tasks = [Task::T2V].into_iter().collect();
    c
}

/// FastWan: T2V only, 4k+1 over 49..=121, container-only fps, video-only.
pub fn fastwan() -> ModelCaps {
    ModelCaps {
        id: ModelId::new("fastwan"),
        family: Family::Wan,
        served_names: vec!["FastWan2.2-TI2V-5B".into()],
        tasks: BTreeSet::from([Task::T2V]),
        audio: None,
        fps: FpsCaps {
            allowed: vec![16, 24],
            default: 24,
            container_only: true,
        },
        frames: FrameGrid::new(4, 1, 49, 121, 81),
        canvas: CanvasCaps {
            multiple: 32,
            max_area: 1280 * 704,
            aspect: (0.25, 4.0),
            short_edges: vec![704, 480],
            pad_and_crop: false,
            hd: None,
        },
        refs: RefLimits::none(),
        stream: Some(StreamCaps::Clip {
            min_s: 2.0,
            max_s: 5.0,
        }),
        knobs: KnobCaps::all(),
        resident: true,
        tier: None,
        recipe: None,
    }
}

/// A video-only Wan model with the MMAudio sidecar available.
pub fn wan_sidecar() -> ModelCaps {
    let mut c = fastwan();
    c.id = ModelId::new("wan_mm");
    c.audio = Some(AudioCaps {
        native_rate: 44_100,
        channels: 2,
        via_sidecar: true,
    });
    c
}

pub fn t2v(model: &str, prompt: &str) -> GenerationRequest {
    GenerationRequest::text(ProtocolId::Native, model, prompt)
}

pub fn url(u: &str) -> MediaRef {
    MediaRef::Http(url::Url::parse(u).unwrap())
}

pub fn staged(mime: &str, dims: Option<(u32, u32)>, name: &str) -> StagedMedia {
    StagedMedia {
        path: PathBuf::from(format!("/stage/{name}")),
        mime: mime.into(),
        bytes: 1234,
        probe: MediaProbe {
            width: dims.map(|d| d.0),
            height: dims.map(|d| d.1),
            ..MediaProbe::default()
        },
    }
}

pub fn image(name: &str, w: u32, h: u32) -> StagedMedia {
    staged("image/png", Some((w, h)), name)
}

/// Stages every input of `req` with plausible media (images 1920x1080).
pub fn stage_all(req: &GenerationRequest) -> StagedInputs {
    StagedInputs {
        keyframes: req
            .keyframes
            .iter()
            .enumerate()
            .map(|(i, k)| (k.at, image(&format!("kf{i}.png"), 1920, 1080)))
            .collect(),
        references: req
            .references
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let m = match r.kind {
                    MediaKind::Image => image(&format!("ref{i}.png"), 1024, 1024),
                    MediaKind::Video => {
                        staged("video/mp4", Some((1280, 720)), &format!("ref{i}.mp4"))
                    }
                    MediaKind::Audio => staged("audio/wav", None, &format!("ref{i}.wav")),
                };
                (r.kind, m)
            })
            .collect(),
        audio_in: req
            .audio_in
            .as_ref()
            .map(|_| {
                // 8 s at 48 kHz: long enough for every audio-to-video default.
                let mut m = staged("audio/wav", None, "audio.wav");
                m.probe.duration_s = Some(8.0);
                m.probe.audio_rate = Some(48_000);
                m
            }),
        video_in: req.edit.as_ref().map(|_| source_video(768, 512, 24.0, 121, true)),
    }
}

/// A staged source video for retake / extend.
pub fn source_video(w: u32, h: u32, fps: f64, frames: u32, audio: bool) -> StagedMedia {
    let mut m = staged("video/mp4", Some((w, h)), "source.mp4");
    m.probe.fps = Some(fps);
    m.probe.frames = Some(frames);
    m.probe.duration_s = Some(f64::from(frames) / fps);
    m.probe.audio_rate = audio.then_some(44_100);
    m
}

/// Negotiates with auto-staged inputs.
pub fn nego(req: &GenerationRequest, caps: &ModelCaps) -> Result<ResolvedJob, ApiError> {
    negotiate(req, caps, &stage_all(req))
}

pub fn seconds(req: &mut GenerationRequest, s: f64) {
    req.timing.length = Length::Seconds {
        value: s,
        snap: Snap::AlignUp,
    };
}

pub fn frames(req: &mut GenerationRequest, n: u32, snap: Snap) {
    req.timing.length = Length::Frames { value: n, snap };
}

pub fn gap_of(r: Result<ResolvedJob, ApiError>) -> Option<GapId> {
    r.err().and_then(|e| e.gap())
}

pub fn err_of(r: Result<ResolvedJob, ApiError>) -> ApiError {
    r.expect_err("expected negotiation to fail")
}
