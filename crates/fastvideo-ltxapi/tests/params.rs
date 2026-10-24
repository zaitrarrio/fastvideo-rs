//! Plan E2: the LTX parser enforces the parse-time domains of its
//! `ParamSpec`s, for every endpoint and `model` id.

use fastvideo_ltxapi::error::Api;
use fastvideo_ltxapi::params::endpoint_spec;
use fastvideo_ltxapi::request::{normalize, Endpoint};
use fastvideo_ltxapi::LtxModels;
use fastvideo_protocol::params::parse_agreement;
use fastvideo_protocol::{
    AudioCaps, CanvasCaps, Family, FpsCaps, FrameGrid, KnobCaps, ModelCaps, ModelId, RefLimits,
    Task,
};
use serde_json::{json, Value};

fn ltx() -> ModelCaps {
    ModelCaps {
        id: ModelId("ltx25".into()),
        family: Family::Ltx2,
        served_names: vec!["ltx25".into()],
        tasks: [
            Task::T2V,
            Task::I2V,
            Task::Keyframes,
            Task::A2V,
            Task::Retake,
            Task::Extend,
        ]
        .into_iter()
        .collect(),
        audio: Some(AudioCaps {
            native_rate: 48_000,
            channels: 2,
            via_sidecar: false,
        }),
        fps: FpsCaps {
            allowed: vec![24, 25, 48, 50],
            default: 24,
            container_only: false,
        },
        frames: FrameGrid::new(8, 1, 9, 481, 121),
        canvas: CanvasCaps {
            multiple: 64,
            max_area: 1920 * 1088,
            aspect: (0.25, 4.0),
            short_edges: vec![1080, 720],
            pad_and_crop: true,
            hd: None,
        },
        refs: RefLimits::none(),
        stream: None,
        knobs: KnobCaps::default(),
        resident: true,
        tier: None,
        recipe: None,
    }
}

fn base(e: Endpoint, id: &str) -> Value {
    match e {
        Endpoint::TextToVideo => {
            json!({"prompt": "a cat", "model": id, "duration": 6, "resolution": "1920x1080"})
        }
        Endpoint::ImageToVideo => {
            json!({"prompt": "a cat", "model": id, "duration": 6, "resolution": "1920x1080", "image_uri": "https://example.com/i.png"})
        }
        Endpoint::AudioToVideo => {
            json!({"prompt": "a cat", "model": id, "audio_uri": "https://example.com/a.mp3"})
        }
        Endpoint::Retake => {
            json!({"model": id, "video_uri": "https://example.com/v.mp4", "start_time": 0, "duration": 3})
        }
        Endpoint::Extend => {
            json!({"model": id, "video_uri": "https://example.com/v.mp4", "duration": 3})
        }
    }
}

#[test]
fn ltx_parser_enforces_the_spec() {
    let table = LtxModels::default();
    let ids: Vec<&str> = table.ids().collect();
    let caps = ltx();
    for e in Endpoint::ALL {
        for id in &ids {
            if e == Endpoint::AudioToVideo && *id == "ltx-2-3-fast" {
                continue;
            }
            let m = table.get(id).unwrap();
            let spec = endpoint_spec(e, id, m, &ids, &caps);
            let Value::Object(b) = base(e, id) else {
                unreachable!()
            };
            let parse = |v: Value| normalize(e, Api::V2, &table, &v).map(|_| ());
            // `model` selects the matrix; `duration` is checked against the
            // matrix cell (resolution and fps); `camera_motion` values are a
            // gap (refused naming the field, as an Unsupported error);
            // the edit `duration` is clamped (retake) or bounded (extend).
            let skip = ["model", "duration", "camera_motion"];
            let bad = parse_agreement(&spec, &b, &parse, &skip);
            assert!(bad.is_empty(), "{e:?} {id}: {bad:#?}");
        }
    }
}
