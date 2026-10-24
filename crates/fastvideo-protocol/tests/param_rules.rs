//! Plan E2 (ADR-0012): the surface rules (`fastvideo_protocol::params::rules`)
//! say what the vendored upstream contracts under `docs/serve/specs/` say.
//! Where this server deliberately differs, the test pins the upstream
//! value next to ours, so a drift of either side is seen.

use std::path::Path;

use fastvideo_protocol::params::rules::{
    fal_director, fal_h3, fal_wan, h3, ltx, minimax, openai_videos,
};
use fastvideo_protocol::params::{IntField, IntRange, NumField, NumRange};
use fastvideo_protocol::{
    EDIT_DURATION_MAX_S, EDIT_DURATION_MIN_S, EXTEND_CONTEXT_MAX_S, EXTEND_CONTEXT_MIN_S,
};
use serde_json::Value;

fn spec(rel: &str) -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/serve/specs")
        .join(rel);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{rel}: {e}")))
        .unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// The input schema of a fal queue OpenAPI (the one `*Input` schema).
fn fal_input(rel: &str) -> Value {
    let doc = spec(&format!("fal/{rel}.openapi.json"));
    let schemas = doc["components"]["schemas"].as_object().unwrap();
    schemas
        .iter()
        .find(|(k, _)| k.ends_with("Input"))
        .map(|(_, v)| v["properties"].clone())
        .unwrap_or_else(|| panic!("{rel}: no Input schema"))
}

/// The non-null member of a property (`anyOf: [{..}, {"type": "null"}]`).
fn member(p: &Value) -> &Value {
    match p["anyOf"].as_array() {
        Some(a) => a.iter().find(|m| m["type"] != "null").unwrap_or(p),
        None => p,
    }
}

fn int_range(p: &Value) -> IntRange {
    let m = member(p);
    IntRange::new(
        m["minimum"].as_i64().unwrap(),
        m["maximum"].as_i64().unwrap(),
    )
}

fn num_range(p: &Value) -> NumRange {
    let m = member(p);
    NumRange::new(
        m["minimum"].as_f64().unwrap(),
        m["maximum"].as_f64().unwrap(),
    )
}

fn int_field(p: &Value) -> IntField {
    let r = int_range(p);
    IntField::new(r.min, r.max, p["default"].as_i64().unwrap())
}

fn num_field(p: &Value) -> NumField {
    let r = num_range(p);
    NumField::new(r.min, r.max, p["default"].as_f64().unwrap())
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .filter_map(|s| s.as_str().map(str::to_owned))
        .collect()
}

fn owned(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_owned()).collect()
}

#[test]
fn fastvideo_videos_rules() {
    let s = spec("fastvideo/videos.schema.json");
    let p = &s["properties"];
    assert_eq!(
        int_range(&p["num_inference_steps"]),
        openai_videos::NUM_INFERENCE_STEPS
    );
    assert_eq!(
        num_range(&p["guidance_scale"]),
        openai_videos::GUIDANCE_SCALE
    );
    assert_eq!(
        num_range(&p["guidance_scale_2"]),
        openai_videos::GUIDANCE_SCALE
    );
    assert_eq!(
        num_range(&p["boundary_ratio"]),
        openai_videos::BOUNDARY_RATIO
    );
    assert_eq!(int_range(&p["n"]), openai_videos::OUTPUTS);
    assert_eq!(
        int_range(&p["num_outputs_per_prompt"]),
        openai_videos::OUTPUTS
    );
    assert_eq!(
        strings(&p["quality"]["enum"]),
        owned(&openai_videos::QUALITIES)
    );
    // Known divergence: upstream takes any int64 seed; negative seeds are
    // refused here (a seed is a u64 inside).
    assert!(p["seed"]["minimum"].as_i64().unwrap() < 0);
}

#[test]
fn openai_list_limit() {
    let y = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/serve/specs/openai/videos.openapi.yaml"),
    )
    .unwrap();
    // `- name: limit ... minimum: 0 ... maximum: 100` (no YAML parser in
    // the workspace: the block is read as text).
    let at = y.find("- name: limit").unwrap();
    let block = &y[at..at + 400];
    let max = format!("maximum: {}", openai_videos::LIST_LIMIT.range.max);
    assert!(block.contains(&max), "{block}");
    // Known divergence: upstream lists `minimum: 0`; this server takes 1.
    assert!(block.contains("minimum: 0"), "{block}");
}

#[test]
fn minimax_rules() {
    let d = spec("minimax/v2-video-generation.json");
    let r = &d["components"]["schemas"]["VideoGenerationV2Req"]["properties"];
    let durations: Vec<i64> = r["duration"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let want: Vec<i64> = (minimax::DURATION.min..=minimax::DURATION.max).collect();
    assert_eq!(durations, want);
    assert_eq!(strings(&r["ratio"]["enum"]), owned(&minimax::RATIOS));
    let mut fixed = vec![h3::ADAPTIVE];
    fixed.extend(h3::ASPECT_RATIOS);
    assert_eq!(owned(&minimax::RATIOS), owned(&fixed));
}

#[test]
fn ltx_api_rules() {
    let d = spec("ltx/openapi.json");
    let s = &d["components"]["schemas"];
    for name in [
        "TextToVideoRequest",
        "ImageToVideoRequest",
        "AudioToVideoRequest",
        "EditVideoRequest",
        "ExtendVideoRequest",
    ] {
        assert_eq!(
            s[name]["properties"]["prompt"]["maxLength"].as_u64(),
            Some(ltx::PROMPT_MAX_CHARS as u64),
            "{name}"
        );
    }
    for name in [
        "TextToVideoRequest",
        "ImageToVideoRequest",
        "AudioToVideoRequest",
    ] {
        assert_eq!(
            s[name]["properties"]["fps"]["default"].as_u64(),
            Some(u64::from(ltx::API_DEFAULT_FPS)),
            "{name}"
        );
    }
    assert_eq!(
        strings(&s["TextToVideoRequestCameraMotion"]["enum"]),
        owned(&ltx::CAMERA_MOTIONS)
    );
    assert_eq!(
        strings(&s["EditVideoRequestResolution"]["enum"]),
        owned(&ltx::RETAKE_RESOLUTIONS)
    );
    let e = &s["ExtendVideoRequest"]["properties"];
    assert_eq!(
        num_range(&e["duration"]),
        NumRange::new(EDIT_DURATION_MIN_S, EDIT_DURATION_MAX_S)
    );
    assert_eq!(
        num_range(&e["context"]),
        NumRange::new(EXTEND_CONTEXT_MIN_S, EXTEND_CONTEXT_MAX_S)
    );
    let r = &s["EditVideoRequest"]["properties"];
    assert_eq!(r["duration"]["minimum"].as_f64(), Some(EDIT_DURATION_MIN_S));
    assert_eq!(r["start_time"]["minimum"].as_f64(), Some(0.0));
}

#[test]
fn fal_h3_rules() {
    for app in ["minimax/h3"] {
        for e in ["text-to-video", "image-to-video", "reference-to-video"] {
            let p = fal_input(&format!("{app}/{e}"));
            assert_eq!(int_range(&p["duration"]), fal_h3::DURATION, "{app}/{e}");
            assert_eq!(
                p["prompt"]["maxLength"].as_u64(),
                Some(fal_h3::PROMPT_MAX_CHARS as u64)
            );
        }
    }
    for app in ["minimax/h3", "minimax/h3-max"] {
        let r = fal_input(&format!("{app}/reference-to-video"));
        let refs = fal_h3::REFERENCES;
        assert_eq!(
            r["reference_image_urls"]["maxItems"].as_u64(),
            Some(u64::from(refs.images))
        );
        assert_eq!(
            r["reference_video_urls"]["maxItems"].as_u64(),
            Some(u64::from(refs.videos))
        );
        assert_eq!(
            r["reference_audio_urls"]["maxItems"].as_u64(),
            Some(u64::from(refs.audio))
        );
        let t = fal_input(&format!("{app}/text-to-video"));
        assert_eq!(
            strings(&t["aspect_ratio"]["enum"]),
            owned(&h3::ASPECT_RATIOS)
        );
    }
    let r = fal_input("minimax/h3-max/reference-to-video");
    assert_eq!(
        r["reference_conditions"]["maxItems"].as_u64(),
        Some(u64::from(fal_h3::REFERENCES.total))
    );
    // Known divergence (census, PR): the vendored h3-max schemas now give
    // `duration` as a number 0.92..=15; this server keeps integer 5..=15.
    let m = fal_input("minimax/h3-max/text-to-video");
    assert_eq!(m["duration"]["type"], "number");
    assert_eq!(
        m["duration"]["maximum"].as_i64(),
        Some(fal_h3::DURATION.max)
    );
}

#[test]
fn fal_ltx_rules() {
    for (sub, fast) in [
        ("text-to-video/fast", true),
        ("text-to-video/pro", false),
        ("image-to-video/fast", true),
        ("image-to-video/pro", false),
    ] {
        let p = fal_input(&format!("lightricks/ltx-2.5/{sub}"));
        let fps: Vec<u32> = p["fps"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            fps,
            if fast {
                ltx::FPS.to_vec()
            } else {
                ltx::FPS_FAL_PRO.to_vec()
            },
            "{sub}"
        );
        assert_eq!(
            p["fps"]["default"].as_u64(),
            Some(u64::from(ltx::FAL_DEFAULT_FPS))
        );
        let durations: Vec<u32> = p["duration"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_u64().map(|d| d as u32))
            .collect();
        let max = if fast {
            ltx::DURATION_MAX_LONG
        } else {
            ltx::DURATION_MAX_SHORT
        };
        assert_eq!(durations, ltx::durations(max), "{sub}");
        assert_eq!(
            strings(&member(&p["camera_motion"])["enum"]),
            owned(&ltx::CAMERA_MOTIONS)
        );
        assert_eq!(
            p["prompt"]["maxLength"].as_u64(),
            Some(ltx::PROMPT_MAX_CHARS as u64)
        );
    }
    let i = fal_input("fal-ai/ltx-2.3-quality/ingredient");
    assert_eq!(int_field(&i["num_frames"]), ltx::INGREDIENT_FRAMES);
    assert_eq!(int_field(&i["frames_per_second"]), ltx::INGREDIENT_FPS);
    assert_eq!(
        num_field(&i["ingredient_strength"]),
        ltx::INGREDIENT_STRENGTH
    );
    assert_eq!(
        num_field(&i["reference_strength"]),
        ltx::INGREDIENT_STRENGTH
    );
    let (w, h) = ltx::INGREDIENT_CANVAS;
    assert_eq!(
        i["resolution"]["default"]["width"].as_u64(),
        Some(u64::from(w))
    );
    assert_eq!(
        i["resolution"]["default"]["height"].as_u64(),
        Some(u64::from(h))
    );
    assert_eq!(ltx::FRAMES_MAX as i64, ltx::INGREDIENT_FRAMES.range.max);
    let r = fal_input("fal-ai/ltx-2.3/retake-video");
    assert_eq!(num_range(&r["start_time"]), ltx::FAL_RETAKE_START);
    assert_eq!(
        num_range(&r["duration"]),
        NumRange::new(EDIT_DURATION_MIN_S, EDIT_DURATION_MAX_S)
    );
    assert_eq!(
        r["duration"]["default"].as_f64(),
        Some(ltx::FAL_EDIT_DURATION_DEFAULT)
    );
    let x = fal_input("fal-ai/ltx-2.3/extend-video");
    assert_eq!(
        num_range(&x["context"]),
        NumRange::new(EXTEND_CONTEXT_MIN_S, EXTEND_CONTEXT_MAX_S)
    );
    assert_eq!(
        x["duration"]["default"].as_f64(),
        Some(ltx::FAL_EDIT_DURATION_DEFAULT)
    );
}

#[test]
fn fal_wan_rules() {
    for e in ["text-to-video", "image-to-video", "text-to-video/fast-wan"] {
        let p = fal_input(&format!("fal-ai/wan/v2.2-5b/{e}"));
        assert_eq!(int_field(&p["num_frames"]), fal_wan::NUM_FRAMES, "{e}");
        assert_eq!(
            int_field(&p["frames_per_second"]),
            fal_wan::FRAMES_PER_SECOND,
            "{e}"
        );
        assert_eq!(
            num_field(&p["guidance_scale"]),
            fal_wan::GUIDANCE_SCALE,
            "{e}"
        );
        assert_eq!(
            int_field(&p["num_interpolated_frames"]),
            fal_wan::INTERPOLATED_FRAMES,
            "{e}"
        );
        assert_eq!(
            strings(&p["interpolator_model"]["enum"]),
            owned(&fal_wan::INTERPOLATORS)
        );
        if !e.ends_with("fast-wan") {
            assert_eq!(
                int_field(&p["num_inference_steps"]),
                fal_wan::NUM_INFERENCE_STEPS,
                "{e}"
            );
            assert_eq!(num_field(&p["shift"]), fal_wan::SHIFT, "{e}");
        }
    }
}

#[test]
fn fal_director_rules() {
    let d = spec("fal/fal-ai/minimax-h3-max-director/asyncapi.json");
    let c = &d["components"]["schemas"]["MiniMaxH3MaxDirectorClientConfigureMessage"]["properties"];
    assert_eq!(int_range(&c["memory"]), fal_director::MEMORY);
    assert_eq!(
        c["prompt"]["maxLength"].as_u64(),
        Some(fal_director::PROMPT_MAX_CHARS as u64)
    );
    assert_eq!(
        c["protocol_version"]["const"].as_u64(),
        Some(fal_director::PROTOCOL_VERSION)
    );
    assert_eq!(c["prompt_version"]["minimum"].as_u64(), Some(1));
}
