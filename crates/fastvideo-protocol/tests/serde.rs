//! Serde round-trips and pinned JSON shapes of the wire-facing types.

mod common;

use common::*;
use fastvideo_protocol::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::json;
use time::macros::datetime;

fn round_trip<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(
    v: &T,
) -> serde_json::Value {
    let j = serde_json::to_value(v).unwrap();
    let back: T = serde_json::from_value(j.clone()).unwrap();
    assert_eq!(&back, v, "round trip of {j}");
    j
}

pub fn full_request() -> GenerationRequest {
    let mut r = GenerationRequest::text(ProtocolId::MiniMaxV2, "MiniMax-H3", "a cat surfing");
    r.task = Task::Keyframes;
    r.negative_prompt = Some("blurry".into());
    r.seed = Some(7);
    r.canvas = CanvasSpec::Aspect {
        ratio: Ratio::R9_16,
        short_edge: 768,
    };
    r.timing = TimingSpec {
        length: Length::Seconds {
            value: 6.0,
            snap: Snap::AlignUp,
        },
        fps: Some(24),
    };
    r.keyframes = vec![
        Keyframe {
            at: Anchor::First,
            image: url("https://e.x/f.png"),
        },
        Keyframe {
            at: Anchor::Last,
            image: MediaRef::DataUri("data:image/png;base64,AA==".into()),
        },
    ];
    r.references = vec![Reference {
        kind: MediaKind::Video,
        media: MediaRef::Upload(UploadId("tok".into())),
    }];
    r.audio_in = Some(AudioInput {
        media: MediaRef::ProviderFile("mm_file://9".into()),
        role: AudioRole::TargetSoundtrack,
        max_s: None,
    });
    r.audio_out = AudioOut::Silent;
    r.sampling = SamplingOverrides {
        steps: Some(4),
        guidance: Some(1.0),
        ..Default::default()
    };
    r.output = OutputOptions {
        inline_data_uri: true,
    };
    r.callback = Some(CallbackSpec::MiniMax {
        url: url::Url::parse("https://cb.example/hook").unwrap(),
    });
    r
}

#[test]
fn generation_request_round_trip_and_shape() {
    let r = full_request();
    let j = round_trip(&r);
    assert_eq!(j["protocol"], "minimax_v2");
    assert_eq!(j["task"], "keyframes");
    assert_eq!(
        j["canvas"],
        json!({"aspect": {"ratio": {"w": 9, "h": 16}, "short_edge": 768}})
    );
    assert_eq!(
        j["timing"],
        json!({"length": {"seconds": {"value": 6.0, "snap": "align_up"}}, "fps": 24})
    );
    assert_eq!(
        j["keyframes"][0],
        json!({"at": "first", "image": {"http": "https://e.x/f.png"}})
    );
    assert_eq!(j["references"][0]["media"], json!({"upload": "tok"}));
    assert_eq!(j["audio_in"]["role"], "target_soundtrack");
    assert_eq!(j["audio_out"], "silent");
    assert_eq!(
        j["callback"],
        json!({"kind": "mini_max", "url": "https://cb.example/hook"})
    );

    let t = t2v("m", "p");
    let j = round_trip(&t);
    assert_eq!(j["canvas"], "model_default");
    assert_eq!(j["timing"]["length"], "model_default");
}

#[test]
fn accepted_noop_serializes_but_is_not_deserialized() {
    let mut r = t2v("m", "p");
    r.note_noop("prompt_expansion_mode");
    r.note_noop("prompt_expansion_mode");
    r.note_noop("enable_safety_checker");
    assert_eq!(
        r.accepted_noop,
        vec!["prompt_expansion_mode", "enable_safety_checker"]
    );
    let j = serde_json::to_value(&r).unwrap();
    assert_eq!(
        j["accepted_noop"],
        json!(["prompt_expansion_mode", "enable_safety_checker"])
    );
    let back: GenerationRequest = serde_json::from_value(j).unwrap();
    assert!(back.accepted_noop.is_empty());
    assert_eq!(
        GenerationRequest {
            accepted_noop: vec![],
            ..r
        },
        back
    );
}

#[test]
fn enum_names() {
    for t in [
        Task::T2V,
        Task::I2V,
        Task::Keyframes,
        Task::Ref2V,
        Task::A2V,
        Task::Extend,
        Task::Retake,
        Task::V2V,
    ] {
        let s = serde_json::to_value(t).unwrap();
        assert_eq!(s.as_str().unwrap(), s.as_str().unwrap().to_lowercase());
        assert!(!s.as_str().unwrap().contains('_'), "{s}");
        round_trip(&t);
    }
    for p in ProtocolId::ALL {
        assert_eq!(serde_json::to_value(p).unwrap(), p.as_str(), "{p:?}");
        assert_eq!(p.to_string(), p.as_str());
        round_trip(&p);
    }
    for g in GapId::ALL {
        assert_eq!(serde_json::to_value(g).unwrap(), g.code(), "{g:?}");
        round_trip(&g);
    }
    assert_eq!(
        serde_json::to_value(GapId::H3Resolution2K).unwrap(),
        "h3_resolution_2k"
    );
    assert_eq!(serde_json::to_value(GapId::Ltx25I2V).unwrap(), "ltx25_i2v");
    assert_eq!(serde_json::to_value(Family::MmAudio).unwrap(), "mm_audio");
    assert_eq!(serde_json::to_value(Family::Ltx2).unwrap(), "ltx2");
}

#[test]
fn error_kind_codes_match_serde() {
    let kinds = [
        ErrorKind::InvalidRequest,
        ErrorKind::Unauthorized,
        ErrorKind::Forbidden,
        ErrorKind::NotFound,
        ErrorKind::AlreadyCompleted,
        ErrorKind::Conflict,
        ErrorKind::PayloadTooLarge,
        ErrorKind::UnsupportedMedia,
        ErrorKind::ContentFiltered,
        ErrorKind::RateLimited,
        ErrorKind::QueueFull,
        ErrorKind::Loading,
        ErrorKind::Timeout,
        ErrorKind::Cancelled,
        ErrorKind::EngineFailed,
        ErrorKind::Internal,
    ];
    for k in kinds {
        assert_eq!(serde_json::to_value(k).unwrap(), k.code());
        round_trip(&k);
    }
    let u = ErrorKind::Unsupported(GapId::LtxFps);
    assert_eq!(
        serde_json::to_value(u).unwrap(),
        json!({"unsupported": "ltx_fps"})
    );
    assert_eq!(u.code(), "unsupported");
    round_trip(&u);
}

#[test]
fn api_error_shape() {
    let e = ApiError::unsupported(GapId::H3FourSeconds).with_param("duration");
    let j = round_trip(&e);
    assert_eq!(
        j,
        json!({"kind": {"unsupported": "h3_four_seconds"}, "message": GapId::H3FourSeconds.default_message(), "param": "duration"})
    );
    let j = round_trip(&ApiError::loading("warming up"));
    assert_eq!(
        j,
        json!({"kind": "loading", "message": "warming up", "retry_after_s": 1})
    );
    let e: ApiError = serde_json::from_value(json!({"kind": "internal", "message": "x"})).unwrap();
    assert_eq!(e, ApiError::internal("x"));
}

#[test]
fn caps_round_trip() {
    for c in [h3(), h3_ref2va(), ltx23(), fastwan(), wan_sidecar()] {
        round_trip(&c);
    }
    let j = serde_json::to_value(h3()).unwrap();
    assert_eq!(j["tasks"], json!(["t2v", "i2v", "keyframes"]));
    assert_eq!(
        j["frames"],
        json!({"step": 17, "offset": 5, "min": 107, "max": 362, "default": 124})
    );
    assert_eq!(
        j["audio"],
        json!({"native_rate": 32000, "channels": 2, "via_sidecar": false})
    );
    assert_eq!(j["canvas"]["aspect"], json!([0.25, 4.0]));
    let c = ModelCaps {
        stream: Some(StreamCaps::Causal {
            block_frames: 12,
            target_fps: 16,
        }),
        ..fastwan()
    };
    let j = round_trip(&c);
    assert_eq!(
        j["stream"],
        json!({"causal": {"block_frames": 12, "target_fps": 16}})
    );
}

#[test]
fn tier_and_recipe_shapes() {
    // Untiered caps and jobs omit both fields, and old JSON without them loads.
    let j = serde_json::to_value(h3()).unwrap();
    assert!(j.get("tier").is_none() && j.get("recipe").is_none());
    let back: ModelCaps = serde_json::from_value(j).unwrap();
    assert_eq!((back.tier, back.recipe), (None, None));

    let c = h3().with_tier(Tier::Turbo, "fasth3-4step-vsa");
    let j = round_trip(&c);
    assert_eq!(j["tier"], "turbo");
    assert_eq!(j["recipe"], "fasth3-4step-vsa");
    assert_eq!(serde_json::to_value(Tier::Max).unwrap(), "max");
    assert_eq!(Tier::Max.to_string(), "max");

    let mut r = t2v("fasth3", "a cat");
    r.seed = Some(1);
    let job = nego(&r, &c).unwrap();
    let j = round_trip(&job);
    assert_eq!(
        (j["tier"].clone(), j["recipe"].clone()),
        (json!("turbo"), json!("fasth3-4step-vsa"))
    );
    let j = serde_json::to_value(nego(&r, &h3()).unwrap()).unwrap();
    assert!(j.get("tier").is_none() && j.get("recipe").is_none());
}

#[test]
fn draft_tier_shape_and_order() {
    // Design §0.6: a third tier below the quality gate, serialized "draft".
    assert_eq!(serde_json::to_value(Tier::Draft).unwrap(), "draft");
    let back: Tier = serde_json::from_value(json!("draft")).unwrap();
    assert_eq!(back, Tier::Draft);
    assert_eq!(Tier::Draft.to_string(), "draft");
    assert!(Tier::Draft < Tier::Turbo && Tier::Turbo < Tier::Max);
    let mut all = vec![Tier::Max, Tier::Draft, Tier::Turbo];
    all.sort();
    assert_eq!(all, [Tier::Draft, Tier::Turbo, Tier::Max]);
    assert!(!Tier::Draft.passes_quality_gate());
    assert!(Tier::Turbo.passes_quality_gate() && Tier::Max.passes_quality_gate());

    // A draft model's resolved jobs say so.
    let c = h3().with_tier(Tier::Draft, "fasth3-4step-vsa-480p-taeh3");
    let j = round_trip(&c);
    assert_eq!(j["tier"], "draft");
    let mut r = t2v("fasth3", "a cat");
    r.seed = Some(1);
    let job = nego(&r, &c).unwrap();
    assert_eq!(job.tier, Some(Tier::Draft));
    assert_eq!(round_trip(&job)["tier"], "draft");
    assert_eq!(
        resolve_tier(Family::H3, Tier::Draft, [&c]).unwrap().id,
        c.id
    );
}

pub fn sample_job() -> Job {
    let req = full_request();
    let mut r = t2v("fasth3", "a cat");
    r.seed = Some(5);
    let resolved = nego(&r, &h3()).unwrap();
    let _ = req;
    let now = datetime!(2026-09-27 12:00:00 UTC);
    let mut job = Job::new(
        JobId::new(),
        ProtocolId::Fal,
        "ext-1",
        resolved,
        now,
        std::time::Duration::from_secs(3600),
    );
    job.owner = Some(KeyId("key-a".into()));
    job.request_echo = json!({"model": "minimax/h3-max", "prompt": "a cat"});
    job.logs.push(LogLine::info("started", now));
    job.callback = Some(CallbackSpec::FalWebhook {
        url: url::Url::parse("https://hook.example/x").unwrap(),
    });
    job
}

#[test]
fn job_round_trip_all_states() {
    let mut job = sample_job();
    let t0 = job.created_at;
    let j = round_trip(&job);
    assert_eq!(j["state"], json!({"status": "queued"}));
    assert_eq!(j["created_at"], "2026-09-27T12:00:00Z");
    assert_eq!(j["expires_at"], "2026-09-27T13:00:00Z");
    assert_eq!(j["started_at"], json!(null));
    assert_eq!(
        j["logs"][0],
        json!({"message": "started", "level": "info", "timestamp": "2026-09-27T12:00:00Z"})
    );
    assert_eq!(
        j["resolved"]["audio"],
        json!({"plan": "native", "rate": 32000, "channels": 2})
    );

    job.mark_running(t0 + time::Duration::seconds(1)).unwrap();
    job.set_progress(0.5);
    assert_eq!(round_trip(&job)["started_at"], "2026-09-27T12:00:01Z");

    let mut ok = job.clone();
    let artifact = Artifact {
        id: ArtifactId::new(),
        mime: "video/mp4".into(),
        file_name: "abc_minimax-h3.mp4".into(),
        bytes: 1_000_000,
        location: ArtifactLocation::Local("/state/a.mp4".into()),
        width: 1344,
        height: 768,
        frames: 124,
        fps: 24,
        audio: Some((32_000, 2)),
    };
    let mut metrics = JobMetrics {
        inference_s: Some(12.5),
        peak_memory_mb: Some(70_000.0),
        ..Default::default()
    };
    metrics.stage_durations.insert("denoise".into(), 12.5);
    ok.mark_succeeded(t0 + time::Duration::seconds(20), vec![artifact], metrics)
        .unwrap();
    let j = round_trip(&ok);
    assert_eq!(j["state"], json!({"status": "succeeded"}));
    assert_eq!(j["artifacts"][0]["audio"], json!([32000, 2]));
    assert_eq!(
        j["artifacts"][0]["location"],
        json!({"local": "/state/a.mp4"})
    );

    let mut failed = job.clone();
    failed
        .mark_failed(t0, ApiError::engine_failed("OOM"))
        .unwrap();
    assert_eq!(
        round_trip(&failed)["state"],
        json!({"status": "failed", "error": {"kind": "engine_failed", "message": "OOM"}})
    );

    let mut cancelled = job;
    cancelled.mark_cancelled(t0).unwrap();
    assert_eq!(
        round_trip(&cancelled)["state"],
        json!({"status": "cancelled"})
    );

    let obj = Artifact {
        location: ArtifactLocation::Object {
            bucket: "b".into(),
            key: "k/a.mp4".into(),
        },
        ..ok.artifacts[0].clone()
    };
    assert_eq!(
        round_trip(&obj)["location"],
        json!({"object": {"bucket": "b", "key": "k/a.mp4"}})
    );
}

#[test]
fn ids_are_plain_strings() {
    let id = JobId::new();
    assert_eq!(serde_json::to_value(id).unwrap(), id.to_string());
    assert_eq!(id.to_string().parse::<JobId>().unwrap(), id);
    let a = ArtifactId::new();
    assert_eq!(a.to_string().parse::<ArtifactId>().unwrap(), a);
    assert_eq!(serde_json::to_value(ModelId::new("m")).unwrap(), "m");
    assert_eq!(serde_json::to_value(KeyId("k".into())).unwrap(), "k");
    assert_eq!(serde_json::to_value(UploadId("u".into())).unwrap(), "u");
    assert!("nope".parse::<JobId>().is_err());
}

#[test]
fn staged_and_resolved_round_trip() {
    let mut r = t2v("fasth3", "x");
    r.task = Task::I2V;
    r.keyframes = vec![Keyframe {
        at: Anchor::First,
        image: url("https://e.x/f.png"),
    }];
    let st = stage_all(&r);
    round_trip(&st);
    let j = nego(&r, &h3()).unwrap();
    let v = round_trip(&j);
    assert_eq!(v["keyframes"], json!([["first", "/stage/kf0.png"]]));
    assert_eq!(v["post"], json!({"crop": null, "drop_audio": false}));
    for p in [AudioPlan::Drop, AudioPlan::Sidecar, AudioPlan::None] {
        round_trip(&p);
    }
}

#[test]
fn stream_types_round_trip() {
    let spec = SessionSpec {
        model: ModelId::new("fasth3"),
        tracks: TrackSet::for_model(
            &h3(),
            (1344, 768),
            24,
            ("main_video", "main_audio"),
            1,
            false,
        ),
        canvas: (1344, 768),
        fps: 24,
        continuity: Continuity::AnchorLastFrame { crossfade_ms: 20 },
        max_seconds: Some(600),
        seed: None,
    };
    let j = round_trip(&spec);
    assert_eq!(
        j["continuity"],
        json!({"mode": "anchor_last_frame", "crossfade_ms": 20})
    );
    assert_eq!(
        j["tracks"]["audio"],
        json!({"name": "main_audio", "rate": 48000, "channels": 1})
    );
    assert_eq!(
        serde_json::to_value(Continuity::HardCut).unwrap(),
        json!({"mode": "hard_cut"})
    );
    for s in [
        SessionState::Starting,
        SessionState::Streaming,
        SessionState::Closed(EndReason::Stopped),
        SessionState::Closed(EndReason::Error(ApiError::internal("x"))),
    ] {
        round_trip(&s);
    }
    assert_eq!(
        serde_json::to_value(SessionState::Closed(EndReason::ClientGone)).unwrap(),
        json!({"state": "closed", "reason": "client_gone"})
    );
}

#[test]
fn list_query_and_snapshot_round_trip() {
    let q = ListQuery {
        owner: Some(KeyId("k".into())),
        statuses: vec![JobStatus::Queued, JobStatus::Failed],
        after: Some("x".into()),
        ..ListQuery::default()
    };
    round_trip(&q);
    round_trip(&sample_job().snapshot(3));
    round_trip(&Page {
        items: vec![1, 2],
        total: 5,
        has_more: true,
    });
}
