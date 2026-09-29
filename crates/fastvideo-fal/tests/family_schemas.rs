//! The fal family schemas (docs/serve/fal-parity.md P0) agree with the CUDA
//! catalog's caps: every request the LTX, Wan and base-H3 parsers accept
//! negotiates on the model its endpoint runs (LTX durations past the
//! engine's frame ceiling run at it), and H3 2K/4K is a clean gap.

use fastvideo_engine_service::cuda::caps::{catalog, WeightLayout, WAN5B_FRAMES_MAX};
use fastvideo_fal::schema::ltx::{self, LtxClass, LtxResolution, LTX_FRAMES_MAX};
use fastvideo_fal::schema::wan::{WanVariant, WAN_FRAMES_MAX, WAN_FPS_MAX, WAN_FPS_MIN};
use fastvideo_fal::{AppKind, Endpoint, FalApp, FalInput};
use fastvideo_protocol::{
    negotiate, ApiError, ErrorKind, GapId, ModelCaps, NormalizeCtx, ResolvedJob, StagedInputs, Tier,
};
use serde_json::{json, Value};
use time::macros::datetime;

fn caps_of(tier: (fastvideo_protocol::Family, Tier)) -> ModelCaps {
    catalog(&WeightLayout::default())
        .into_iter()
        .find(|m| m.family() == tier.0 && m.tier == Some(tier.1))
        .unwrap_or_else(|| panic!("no {tier:?} model in the catalog"))
        .caps()
}

fn run(app: &str, e: Endpoint, body: Value) -> Result<ResolvedJob, ApiError> {
    let app = FalApp::from_id(app);
    let cx = NormalizeCtx::new(datetime!(2026-09-28 12:00 UTC));
    let (model, tier) = app.target(e);
    let req = FalInput::parse_for(app.kind(), e, &body)?.normalize(&model, &cx)?;
    negotiate(&req, &caps_of(tier.expect("tier")), &StagedInputs::default())
}

#[test]
fn ltx_matrix_matches_the_engine() {
    for (e, class) in [(Endpoint::LtxTextToVideoFast, LtxClass::Fast), (Endpoint::LtxTextToVideoPro, LtxClass::Pro)] {
        let caps = caps_of(e.target().unwrap());
        assert_eq!(caps.frames.max, LTX_FRAMES_MAX, "the schema's frame ceiling is the engine's");
        let mut clamped = Vec::new();
        for res in class.resolutions() {
            for &fps in class.fps() {
                assert!(caps.fps.allows(fps), "{fps}");
                for d in class.durations() {
                    // Every listed duration runs: past fal's matrix at the
                    // matrix's longest, past the engine's grid at 481 frames.
                    let body = json!({"prompt": "p", "resolution": res.as_str(), "fps": fps, "duration": d});
                    let j = run("lightricks/ltx-2.5", e, body).unwrap_or_else(|err| panic!("{e:?} {res:?} {fps} {d}: {err:?}"));
                    let want = ltx::frames_for(d.min(ltx::max_duration(class, *res, fps)), fps).min(LTX_FRAMES_MAX);
                    assert_eq!((j.fps, j.num_frames), (fps, want), "{e:?} {res:?} {fps} {d}");
                    let delivered = j.post.crop.unwrap_or((j.width, j.height));
                    assert_eq!(delivered, res.landscape(), "{e:?} {res:?}");
                    if d <= ltx::max_duration(class, *res, fps) && ltx::frames_for(d, fps) > LTX_FRAMES_MAX {
                        clamped.push((res.as_str(), fps, d));
                    }
                }
            }
        }
        // fal allows these; this server's LTX grid (481 frames) runs them at 481.
        let want: Vec<(&str, u32, u32)> = match class {
            LtxClass::Fast => vec![("720p", 25, 20), ("720p", 50, 10), ("1080p", 25, 20), ("1080p", 50, 10), ("1440p", 50, 10), ("2160p", 50, 10)],
            LtxClass::Pro => vec![("720p", 50, 10), ("1080p", 50, 10)],
        };
        assert_eq!(clamped, want, "{class:?}");
    }
    // Portrait, silent, `static` camera, auto duration.
    let j = run("lightricks/ltx-2.5", Endpoint::LtxTextToVideoFast, json!({"prompt": "p", "aspect_ratio": "9:16", "resolution": "720p", "generate_audio": false, "camera_motion": "static"})).unwrap();
    assert_eq!((j.post.crop.unwrap_or((j.width, j.height)), j.fps, j.num_frames), ((720, 1280), 25, 153));
    assert_eq!(j.audio, fastvideo_protocol::AudioPlan::Drop);
    let gap = |r: Result<ResolvedJob, ApiError>| r.unwrap_err().kind;
    assert_eq!(gap(run("lightricks/ltx-2.5", Endpoint::LtxTextToVideoFast, json!({"prompt": "p", "duration": "auto"}))), ErrorKind::Unsupported(GapId::LtxAutoDuration));
    assert_eq!(gap(run("lightricks/ltx-2.5", Endpoint::LtxTextToVideoPro, json!({"prompt": "p", "camera_motion": "dolly_in"}))), ErrorKind::Unsupported(GapId::LtxCameraMotion));
    // Pro has no 1440p/2160p, no 48 fps and no 12 s.
    for body in [json!({"prompt": "p", "resolution": "1440p"}), json!({"prompt": "p", "fps": 48}), json!({"prompt": "p", "duration": 12})] {
        assert!(FalInput::parse(Endpoint::LtxTextToVideoPro, &body).is_err(), "{body}");
    }
    // Image-to-video needs `image_url`.
    assert!(FalInput::parse(Endpoint::LtxImageToVideoFast, &json!({"prompt": "p"})).is_err());
}

#[test]
fn wan_schema_matches_the_engine() {
    assert_eq!(WAN_FRAMES_MAX as u32, WAN5B_FRAMES_MAX);
    for (e, v) in [(Endpoint::WanTextToVideo, WanVariant::TextToVideo), (Endpoint::WanFastWan, WanVariant::FastWan)] {
        let caps = caps_of(e.target().unwrap());
        for res in v.resolutions() {
            assert!(caps.canvas.short_edges.contains(&res.short_edge()), "{e:?} {res:?}");
            for aspect in v.aspects() {
                for frames in [17, 80, 81, 121, 161] {
                    for fps in [WAN_FPS_MIN, 16, 24, WAN_FPS_MAX] {
                        let body = json!({"prompt": "p", "resolution": res.as_str(), "aspect_ratio": aspect.as_str(), "num_frames": frames, "frames_per_second": fps});
                        let j = run("fal-ai/wan", e, body).unwrap_or_else(|err| panic!("{e:?} {res:?} {aspect:?} {frames} {fps}: {err:?}"));
                        assert_eq!(j.num_frames, (frames as u32 - 1).div_ceil(4) * 4 + 1);
                        assert_eq!(j.fps, fps as u32);
                        assert_eq!(j.width.min(j.height), res.short_edge(), "{e:?} {res:?} {aspect:?}");
                    }
                }
            }
        }
    }
    // fal's defaults: 720p 16:9, 81 frames at 24 fps; 40 steps, CFG 3.5, shift 5 on the 5B.
    let j = run("fal-ai/wan", Endpoint::WanTextToVideo, json!({"prompt": "p"})).unwrap();
    assert_eq!((j.width, j.height, j.num_frames, j.fps), (1280, 704, 81, 24));
    assert_eq!((j.sampling.steps, j.sampling.guidance, j.sampling.flow_shift), (Some(40), Some(3.5), Some(5.0)));
    assert_eq!(j.tier, Some(Tier::Max));
    // fast-wan: unguided DMD, guidance and negative prompt are no-ops.
    let j = run("fal-ai/wan", Endpoint::WanFastWan, json!({"prompt": "p", "guidance_scale": 5, "negative_prompt": "blur", "resolution": "480p"})).unwrap();
    assert_eq!((j.width, j.height), (832, 480));
    assert!(j.sampling.is_empty() && j.negative_prompt.is_empty());
    assert_eq!(j.tier, Some(Tier::Turbo));
    // 480p is fast-wan only; interpolation is refused; image-to-video needs an image.
    assert!(FalInput::parse(Endpoint::WanTextToVideo, &json!({"prompt": "p", "resolution": "480p"})).is_err());
    let e = FalInput::parse(Endpoint::WanTextToVideo, &json!({"prompt": "p", "interpolator_model": "film", "num_interpolated_frames": 1})).unwrap_err();
    assert_eq!(e.param.as_deref(), Some("interpolator_model"));
    assert!(FalInput::parse(Endpoint::WanImageToVideo, &json!({"prompt": "p"})).is_err());
}

#[test]
fn h3_app_ids() {
    // minimax/h3 (base): 480P and 768P run on the Max tier; 2K / 4K are a clean gap.
    for (res, ok) in [("480P", true), ("768P", true), ("2K", false), ("4K", false)] {
        let body = json!({"prompt": "p", "resolution": res});
        let app = FalApp::from_id("minimax/h3");
        let req = FalInput::parse_for(AppKind::H3Base, Endpoint::TextToVideo, &body)
            .unwrap()
            .normalize(&app.model, &NormalizeCtx::new(datetime!(2026-09-28 12:00 UTC)))
            .unwrap();
        let r = negotiate(&req, &caps_of((fastvideo_protocol::Family::H3, Tier::Max)), &StagedInputs::default());
        match ok {
            true => assert!(r.is_ok(), "{res}: {r:?}"),
            false => assert_eq!(r.unwrap_err().kind, ErrorKind::Unsupported(GapId::H3Resolution2K), "{res}"),
        }
    }
    // 1080P is not in the base enum; 2K is not in the Max enum.
    assert!(FalInput::parse_for(AppKind::H3Base, Endpoint::TextToVideo, &json!({"prompt": "p", "resolution": "1080P"})).is_err());
    assert!(FalInput::parse(Endpoint::TextToVideo, &json!({"prompt": "p", "resolution": "2K"})).is_err());
    let t = FalApp::from_id("minimax/h3-max-turbo");
    assert_eq!((t.model.as_str(), t.tier), ("h3-turbo", Some((fastvideo_protocol::Family::H3, Tier::Turbo))));
    let b = FalApp::from_id("minimax/h3");
    assert_eq!((b.model.as_str(), b.tier), ("h3-max", Some((fastvideo_protocol::Family::H3, Tier::Max))));
}

#[test]
fn endpoint_ids_and_slugs() {
    use fastvideo_fal::schema::{app_id, output_slug};
    assert_eq!(app_id("fal-ai/wan/v2.2-5b/text-to-video/fast-wan"), "fal-ai/wan");
    assert_eq!(app_id("fal-ai/wan/v2.2-5b/text-to-video"), "fal-ai/wan");
    assert_eq!(app_id("lightricks/ltx-2.5/image-to-video/pro"), "lightricks/ltx-2.5");
    assert_eq!(app_id("minimax/h3/text-to-video"), "minimax/h3");
    assert_eq!(app_id("fastvideo/ltx-turbo/text-to-video"), "fastvideo/ltx-turbo");
    assert_eq!(output_slug("minimax/h3-max-turbo"), "minimax-h3-max-turbo");
    assert_eq!(output_slug("lightricks/ltx-2.5"), "ltx-2.5");
    assert_eq!(output_slug("fal-ai/wan"), "wan");
    assert_eq!(output_slug("fastvideo/fastwan21-1.3b"), "fastwan21-1.3b");
    for e in Endpoint::EVERY {
        assert_eq!(Endpoint::from_sub(e.sub()), Some(e));
    }
    assert_eq!(
        fastvideo_fal::endpoint_ids("fal-ai/wan"),
        ["fal-ai/wan/v2.2-5b/text-to-video", "fal-ai/wan/v2.2-5b/image-to-video", "fal-ai/wan/v2.2-5b/text-to-video/fast-wan"]
    );
    let _ = LtxResolution::ALL;
}
