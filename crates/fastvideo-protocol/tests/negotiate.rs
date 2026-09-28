//! Table-driven `negotiate()` tests (design §7.2): grids, canvases, tasks,
//! knobs, audio, and every §4.7 gap that negotiation detects.

mod common;

use std::collections::BTreeMap;

use common::*;
use fastvideo_models::h3::config as h3cfg;
use fastvideo_protocol::*;

// ---- frame grids ------------------------------------------------------------------

#[test]
fn h3_grid_matches_fastvideo_models() {
    let g = FrameGrid::h3();
    assert_eq!(
        (g.step, g.offset, g.min, g.max, g.default),
        (17, 5, 107, 362, 124),
        "4..15 s at 24 fps (MiniMax admits 4 s), default 5 s"
    );
    for n in 1..400 {
        let aligned = h3cfg::align_num_frames(n as usize) as u32;
        assert_eq!(g.next_on_grid(n), Some(aligned), "n={n}");
        let expect = (107..=362).contains(&aligned).then_some(aligned);
        assert_eq!(g.align_up(n), expect, "n={n}");
        assert_eq!(
            g.contains(n),
            n == aligned && (107..=362).contains(&n),
            "n={n}"
        );
    }
}

#[test]
fn frame_grid_edges() {
    let g = FrameGrid::new(4, 1, 49, 121, 81);
    assert!(g.on_grid(1) && g.on_grid(49) && g.on_grid(121) && !g.on_grid(0) && !g.on_grid(50));
    assert_eq!(g.next_on_grid(0), Some(1));
    assert_eq!(g.next_on_grid(50), Some(53));
    assert_eq!(g.align_up(10), None, "below min after snapping");
    assert_eq!(g.align_up(122), None, "above max");
    assert_eq!(g.align_up(121), Some(121));
    let g = FrameGrid::new(8, 1, 9, u32::MAX - 6, 9);
    assert_eq!(
        g.next_on_grid(u32::MAX),
        None,
        "overflow is None, not a panic"
    );
    let fixed = FrameGrid::new(0, 33, 33, 33, 33);
    assert!(fixed.contains(33) && !fixed.contains(34));
    assert_eq!(fixed.align_up(10), Some(33));
    assert_eq!(fixed.align_up(34), None);
}

#[test]
fn seconds_to_frames_per_design_7_2() {
    // H3 5 s -> 124 (fal: 5.167 s), 10 s -> 243, 15 s -> 362.
    for (s, n) in [(5.0, 124), (10.0, 243), (15.0, 362), (5.1, 124), (5.2, 141)] {
        let mut r = t2v("fasth3", "a cat");
        seconds(&mut r, s);
        assert_eq!(nego(&r, &h3()).unwrap().num_frames, n, "{s} s");
    }
    // LTX 6 s @ 24 -> 145; 20 s -> 481.
    for (s, n) in [(6.0, 145), (8.0, 193), (10.0, 241), (20.0, 481)] {
        let mut r = t2v("ltx2_distilled_23", "a cat");
        seconds(&mut r, s);
        assert_eq!(nego(&r, &ltx23()).unwrap().num_frames, n, "{s} s");
    }
    // FastWan 49..=121 on 4k+1.
    for n in (49..=121).step_by(4) {
        let mut r = t2v("fastwan", "a cat");
        frames(&mut r, n, Snap::Exact);
        assert_eq!(nego(&r, &fastwan()).unwrap().num_frames, n);
    }
    for n in [45, 50, 120, 125] {
        let mut r = t2v("fastwan", "a cat");
        frames(&mut r, n, Snap::Exact);
        let e = err_of(nego(&r, &fastwan()));
        assert_eq!(
            (e.kind, e.param.as_deref()),
            (ErrorKind::InvalidRequest, Some("num_frames")),
            "{n}"
        );
    }
}

#[test]
fn exact_frames_error_names_nearest() {
    let mut r = t2v("fasth3", "x");
    frames(&mut r, 125, Snap::Exact);
    let e = err_of(nego(&r, &h3()));
    assert_eq!(e.kind, ErrorKind::InvalidRequest);
    assert!(e.message.contains("nearest valid: 141"), "{}", e.message);
    frames(&mut r, 125, Snap::AlignUp);
    assert_eq!(nego(&r, &h3()).unwrap().num_frames, 141);
}

#[test]
fn length_errors_and_defaults() {
    let mut r = t2v("fasth3", "x");
    assert_eq!(nego(&r, &h3()).unwrap().num_frames, 124, "model default");
    seconds(&mut r, 16.0);
    assert_eq!(err_of(nego(&r, &h3())).param.as_deref(), Some("duration"));
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        seconds(&mut r, bad);
        assert_eq!(
            err_of(nego(&r, &h3())).kind,
            ErrorKind::InvalidRequest,
            "{bad}"
        );
    }
    r.timing.length = Length::Auto;
    assert_eq!(err_of(nego(&r, &h3())).kind, ErrorKind::InvalidRequest);
    let mut w = t2v("fastwan", "x");
    assert_eq!(nego(&w, &fastwan()).unwrap().num_frames, 81);
    w.timing.length = Length::Auto;
    assert_eq!(err_of(nego(&w, &fastwan())).kind, ErrorKind::InvalidRequest);
}

// ---- H3 4 s (E3) --------------------------------------------------------------------

#[test]
fn h3_four_seconds_negotiates_to_107_frames() {
    for exact in [false, true] {
        let mut r = t2v("fasth3", "x");
        if exact {
            frames(&mut r, 107, Snap::Exact);
        } else {
            seconds(&mut r, 4.0);
        }
        let j = negotiate(&r, &h3(), &StagedInputs::default()).expect("4 s is on the H3 grid");
        assert_eq!(j.num_frames, 107);
    }
}

// ---- §4.7 gap table ----------------------------------------------------------------

#[test]
fn gap_table() {
    struct Case {
        name: &'static str,
        caps: ModelCaps,
        req: GenerationRequest,
        gap: GapId,
    }
    let mut cases = Vec::new();

    let mut r = t2v("fasth3", "x");
    r.canvas = CanvasSpec::Aspect {
        ratio: Ratio::R16_9,
        short_edge: 1440,
    };
    cases.push(Case {
        name: "H3 2K",
        caps: h3(),
        req: r,
        gap: GapId::H3Resolution2K,
    });

    let mut r = t2v("fasth3", "x");
    r.canvas = CanvasSpec::Aspect {
        ratio: Ratio::R16_9,
        short_edge: 1080,
    };
    cases.push(Case {
        name: "H3 1080P",
        caps: h3(),
        req: r,
        gap: GapId::H3Refine1080P,
    });

    let mut r = t2v("fasth3", "x");
    r.audio_in = Some(AudioInput {
        media: url("https://e.x/a.wav"),
        role: AudioRole::TargetSoundtrack,
    });
    cases.push(Case {
        name: "H3 target audio",
        caps: h3(),
        req: r,
        gap: GapId::H3TargetAudio,
    });

    let mut r = t2v("fasth3", "x");
    r.task = Task::Ref2V;
    r.references = vec![Reference {
        kind: MediaKind::Image,
        media: url("https://e.x/r.png"),
    }];
    cases.push(Case {
        name: "H3 ref2va not loaded",
        caps: h3(),
        req: r,
        gap: GapId::H3Ref2vaNotLoaded,
    });

    let mut r = t2v("ltx", "x");
    r.task = Task::Keyframes;
    r.keyframes = vec![Keyframe {
        at: Anchor::Last,
        image: url("https://e.x/l.png"),
    }];
    cases.push(Case {
        name: "LTX last frame",
        caps: ltx23(),
        req: r,
        gap: GapId::LtxKeyframes,
    });

    let mut r = t2v("ltx", "x");
    r.task = Task::I2V;
    r.keyframes = vec![Keyframe {
        at: Anchor::First,
        image: url("https://e.x/f.png"),
    }];
    cases.push(Case {
        name: "LTX-2.5 I2V",
        caps: ltx25(),
        req: r,
        gap: GapId::Ltx25I2V,
    });

    let mut r = t2v("ltx", "x");
    r.timing.length = Length::Auto;
    cases.push(Case {
        name: "LTX auto duration",
        caps: ltx23(),
        req: r,
        gap: GapId::LtxAutoDuration,
    });

    for fps in [25, 48, 50] {
        let mut r = t2v("ltx", "x");
        r.timing.fps = Some(fps);
        cases.push(Case {
            name: "LTX fps",
            caps: ltx23(),
            req: r,
            gap: GapId::LtxFps,
        });
    }

    for task in [Task::A2V, Task::Extend, Task::Retake, Task::V2V] {
        let mut r = t2v("ltx", "x");
        r.task = task;
        cases.push(Case {
            name: "LTX endpoint",
            caps: ltx23(),
            req: r,
            gap: GapId::LtxEndpoint,
        });
    }

    let mut r = t2v("fasth3", "x");
    r.task = Task::I2V;
    r.keyframes = vec![Keyframe {
        at: Anchor::First,
        image: MediaRef::ProviderFile("mm_file://1".into()),
    }];
    cases.push(Case {
        name: "provider file",
        caps: h3(),
        req: r,
        gap: GapId::ProviderFiles,
    });

    let mut r = t2v("fasth3", "x");
    r.sampling.steps = Some(8);
    cases.push(Case {
        name: "H3 steps",
        caps: h3(),
        req: r,
        gap: GapId::PerRequestSteps,
    });

    for c in cases {
        let got = nego(&c.req, &c.caps);
        let e = got.expect_err(c.name);
        assert_eq!(e.kind, ErrorKind::Unsupported(c.gap), "{}: {e:?}", c.name);
        assert_eq!(e.http_status(), 400, "{}", c.name);
        // precheck refuses the same way, before any staging.
        assert_eq!(
            precheck(&c.req, &c.caps).unwrap_err().kind,
            e.kind,
            "{} precheck",
            c.name
        );
    }
}

// ---- canvas ---------------------------------------------------------------------------

#[test]
fn h3_aspect_canvases_match_resolve_canvas_size() {
    for (w, h) in [
        (16, 9),
        (9, 16),
        (1, 1),
        (4, 3),
        (3, 4),
        (21, 9),
        (4, 1),
        (1, 4),
        (3, 2),
        (2, 3),
    ] {
        let (eh, ew) = h3cfg::resolve_canvas_size(w as f64, h as f64).unwrap();
        let mut r = t2v("fasth3", "x");
        r.canvas = CanvasSpec::Aspect {
            ratio: Ratio::new(w, h),
            short_edge: 768,
        };
        let j = nego(&r, &h3()).unwrap();
        assert_eq!((j.width, j.height), (ew as u32, eh as u32), "{w}:{h}");
        // The generic short-edge generalization agrees at the 768 tier.
        assert_eq!(
            canvas_for_aspect(&CanvasCaps::h3(), w as f64 / h as f64, 768),
            (ew as u32, eh as u32),
            "{w}:{h}"
        );
    }
}

#[test]
fn h3_default_and_tiers() {
    let j = nego(&t2v("fasth3", "x"), &h3()).unwrap();
    assert_eq!(
        (j.width, j.height, j.num_frames, j.fps),
        (1344, 768, 124, 24)
    );
    assert_eq!(j.post, PostProcess::default());

    let mut r = t2v("fasth3", "x");
    r.canvas = CanvasSpec::Aspect {
        ratio: Ratio::R16_9,
        short_edge: 480,
    };
    let e = err_of(nego(&r, &h3()));
    assert_eq!(
        (e.kind, e.param.as_deref()),
        (ErrorKind::InvalidRequest, Some("resolution")),
        "480 before E3"
    );

    // Once a backend advertises the 480 tier (E3): 832x480 at 16:9 (fal §6).
    let mut caps = h3();
    caps.canvas.short_edges.push(480);
    let j = nego(&r, &caps).unwrap();
    assert_eq!((j.width, j.height), (832, 480));
    r.canvas = CanvasSpec::Aspect {
        ratio: Ratio::R9_16,
        short_edge: 480,
    };
    let j = nego(&r, &caps).unwrap();
    assert_eq!((j.width, j.height), (480, 832));
}

#[test]
fn h3_exact_canvas_uses_check_canvas() {
    let mut r = t2v("fasth3", "x");
    r.canvas = CanvasSpec::Exact {
        width: 1344,
        height: 768,
    };
    assert_eq!(nego(&r, &h3()).unwrap().width, 1344);
    for (w, h) in [(1000, 768), (1376, 768), (0, 768), (4096, 512)] {
        r.canvas = CanvasSpec::Exact {
            width: w,
            height: h,
        };
        let e = err_of(nego(&r, &h3()));
        assert_eq!(
            (e.kind, e.param.as_deref()),
            (ErrorKind::InvalidRequest, Some("size")),
            "{w}x{h}"
        );
    }
}

#[test]
fn aspect_out_of_range() {
    let mut r = t2v("fasth3", "x");
    r.canvas = CanvasSpec::Aspect {
        ratio: Ratio::new(5, 1),
        short_edge: 768,
    };
    assert_eq!(
        err_of(nego(&r, &h3())).param.as_deref(),
        Some("aspect_ratio")
    );
}

#[test]
fn ltx_pad_and_crop() {
    for ((w, h), (gw, gh)) in [
        ((1920, 1080), (1920, 1088)),
        ((1280, 720), (1280, 768)),
        ((3840, 2160), (3840, 2176)),
        ((1080, 1920), (1088, 1920)),
    ] {
        let mut r = t2v("ltx", "x");
        r.canvas = CanvasSpec::Exact {
            width: w,
            height: h,
        };
        let j = nego(&r, &ltx23()).unwrap();
        assert_eq!((j.width, j.height), (gw, gh), "{w}x{h}");
        assert_eq!(j.post.crop, Some((w, h)));
        assert_eq!(j.output_size(), (w, h));
    }
    // Already on the multiple: no crop.
    let mut r = t2v("ltx", "x");
    r.canvas = CanvasSpec::Exact {
        width: 1920,
        height: 1088,
    };
    assert_eq!(nego(&r, &ltx23()).unwrap().post.crop, None);
    // Aspect on a pad-and-crop model targets the exact size then pads.
    r.canvas = CanvasSpec::Aspect {
        ratio: Ratio::R16_9,
        short_edge: 1080,
    };
    let j = nego(&r, &ltx23()).unwrap();
    assert_eq!(
        ((j.width, j.height), j.post.crop),
        ((1920, 1088), Some((1920, 1080)))
    );
    // Default tier: 16:9 at 1080.
    let j = nego(&t2v("ltx", "x"), &ltx23()).unwrap();
    assert_eq!(j.output_size(), (1920, 1080));
    // Over the pixel budget.
    r.canvas = CanvasSpec::Exact {
        width: 7680,
        height: 4320,
    };
    assert_eq!(err_of(nego(&r, &ltx23())).param.as_deref(), Some("size"));
}

#[test]
fn wan_exact_canvas() {
    let mut r = t2v("fastwan", "x");
    r.canvas = CanvasSpec::Exact {
        width: 1280,
        height: 704,
    };
    let j = nego(&r, &fastwan()).unwrap();
    assert_eq!(((j.width, j.height), j.post.crop), ((1280, 704), None));
    r.canvas = CanvasSpec::Exact {
        width: 1280,
        height: 720,
    };
    assert_eq!(
        err_of(nego(&r, &fastwan())).param.as_deref(),
        Some("size"),
        "720 is not a multiple of 32"
    );
    let j = nego(&t2v("fastwan", "x"), &fastwan()).unwrap();
    assert_eq!(
        (j.width, j.height),
        (1248, 704),
        "16:9 at the 704 tier, capped by area and snapped"
    );
}

#[test]
fn aspect_canvas_stays_within_the_pixel_budget() {
    // FastWan 1.3B (832x480 budget, multiple 16): 16:9 at 480 once snapped
    // to 848x480, over the budget (seen through the fal 480P app).
    let c = CanvasCaps {
        aspect: (0.25, 4.0),
        max_area: 832 * 480,
        multiple: 16,
        pad_and_crop: false,
        short_edges: vec![480],
    };
    assert_eq!(canvas_for_aspect(&c, 16.0 / 9.0, 480), (832, 480));
    assert_eq!(canvas_for_aspect(&c, 9.0 / 16.0, 480), (480, 832));
    for (w, h) in [(16u32, 9u32), (9, 16), (1, 1), (4, 3), (3, 4), (21, 9), (4, 1), (1, 4), (3, 2), (2, 3)] {
        let (cw, ch) = canvas_for_aspect(&c, w as f64 / h as f64, 480);
        assert!(u64::from(cw) * u64::from(ch) <= c.max_area, "{w}:{h} -> {cw}x{ch}");
        assert!(cw % 16 == 0 && ch % 16 == 0, "{w}:{h} -> {cw}x{ch}");
    }
}

#[test]
fn follow_image() {
    let mut r = t2v("fasth3", "x");
    r.task = Task::I2V;
    r.keyframes = vec![Keyframe {
        at: Anchor::First,
        image: url("https://e.x/f.png"),
    }];
    r.canvas = CanvasSpec::FollowImage { short_edge: 768 };
    let mut st = stage_all(&r);

    st.keyframes[0].1 = image("f.png", 1920, 1080);
    let j = negotiate(&r, &h3(), &st).unwrap();
    assert_eq!((j.width, j.height), (1344, 768));
    st.keyframes[0].1 = image("f.png", 1080, 1920);
    let j = negotiate(&r, &h3(), &st).unwrap();
    assert_eq!((j.width, j.height), (768, 1344));

    st.keyframes[0].1 = staged("image/png", None, "f.png");
    let e = negotiate(&r, &h3(), &st).unwrap_err();
    assert_eq!(e.kind, ErrorKind::UnsupportedMedia);

    st.keyframes[0].1 = image("f.png", 5000, 1000);
    assert_eq!(
        negotiate(&r, &h3(), &st).unwrap_err().param.as_deref(),
        Some("image_url")
    );

    // FollowImage on T2V has no image.
    let mut t = t2v("fasth3", "x");
    t.canvas = CanvasSpec::FollowImage { short_edge: 768 };
    assert_eq!(
        err_of(nego(&t, &h3())).param.as_deref(),
        Some("aspect_ratio")
    );
    // precheck only checks the tier.
    assert!(precheck(&t, &h3()).is_ok());
    t.canvas = CanvasSpec::FollowImage { short_edge: 1440 };
    assert_eq!(
        precheck(&t, &h3()).unwrap_err().gap(),
        Some(GapId::H3Resolution2K)
    );
}

#[test]
fn follow_image_uses_first_image_reference_for_ref2v() {
    let mut r = t2v("fasth3", "x");
    r.task = Task::Ref2V;
    r.references = vec![
        Reference {
            kind: MediaKind::Audio,
            media: url("https://e.x/a.wav"),
        },
        Reference {
            kind: MediaKind::Image,
            media: url("https://e.x/a.png"),
        },
    ];
    r.canvas = CanvasSpec::FollowImage { short_edge: 768 };
    let mut st = stage_all(&r);
    st.references[1].1 = image("a.png", 720, 1280);
    let j = negotiate(&r, &h3_ref2va(), &st).unwrap();
    assert_eq!((j.width, j.height), (768, 1344));
}

// ---- tasks and inputs -------------------------------------------------------------------

#[test]
fn task_shapes() {
    let first = Keyframe {
        at: Anchor::First,
        image: url("https://e.x/f.png"),
    };
    let last = Keyframe {
        at: Anchor::Last,
        image: url("https://e.x/l.png"),
    };
    let img = Reference {
        kind: MediaKind::Image,
        media: url("https://e.x/r.png"),
    };
    let caps = h3_ref2va();
    let ok = |task, kfs: Vec<Keyframe>, refs: Vec<Reference>| {
        let mut r = t2v("fasth3", "x");
        r.task = task;
        r.keyframes = kfs;
        r.references = refs;
        nego(&r, &caps)
    };
    assert!(ok(Task::I2V, vec![first.clone()], vec![]).is_ok());
    assert!(ok(Task::Keyframes, vec![last.clone()], vec![]).is_ok());
    let j = ok(Task::Keyframes, vec![first.clone(), last.clone()], vec![]).unwrap();
    assert_eq!(
        j.keyframes.iter().map(|k| k.0).collect::<Vec<_>>(),
        vec![Anchor::First, Anchor::Last]
    );
    assert!(ok(Task::Ref2V, vec![], vec![img.clone()]).is_ok());

    for (task, kfs, refs) in [
        (Task::T2V, vec![first.clone()], vec![]),
        (Task::T2V, vec![], vec![img.clone()]),
        (Task::I2V, vec![], vec![]),
        (Task::I2V, vec![first.clone(), first.clone()], vec![]),
        (Task::I2V, vec![last.clone()], vec![]),
        (Task::Keyframes, vec![first.clone()], vec![]),
        (Task::Keyframes, vec![last.clone(), last.clone()], vec![]),
        (Task::Ref2V, vec![], vec![]),
        (Task::Ref2V, vec![first.clone()], vec![img.clone()]),
    ] {
        let e = ok(task, kfs, refs).unwrap_err();
        assert_eq!(
            (e.kind, e.param.as_deref()),
            (ErrorKind::InvalidRequest, Some("task")),
            "{task:?}"
        );
    }
}

#[test]
fn unsupported_task_without_gap_is_invalid() {
    let mut r = t2v("fastwan", "x");
    r.task = Task::I2V;
    r.keyframes = vec![Keyframe {
        at: Anchor::First,
        image: url("https://e.x/f.png"),
    }];
    let e = err_of(nego(&r, &fastwan()));
    assert_eq!(
        (e.kind, e.param.as_deref()),
        (ErrorKind::InvalidRequest, Some("task"))
    );
    assert!(e.message.contains("i2v"));
}

#[test]
fn empty_prompt_refused_for_t2v() {
    let e = err_of(nego(&t2v("fasth3", "  "), &h3()));
    assert_eq!(e.param.as_deref(), Some("prompt"));
}

#[test]
fn references_keep_order_and_limits() {
    let mk = |kind, i| Reference {
        kind,
        media: url(&format!("https://e.x/{i}")),
    };
    let mut r = t2v("fasth3", "x");
    r.task = Task::Ref2V;
    r.references = vec![
        mk(MediaKind::Audio, 0),
        mk(MediaKind::Image, 1),
        mk(MediaKind::Video, 2),
        mk(MediaKind::Image, 3),
    ];
    let j = nego(&r, &h3_ref2va()).unwrap();
    let kinds: Vec<_> = j.references.iter().map(|x| x.0).collect();
    assert_eq!(
        kinds,
        vec![
            MediaKind::Audio,
            MediaKind::Image,
            MediaKind::Video,
            MediaKind::Image
        ]
    );
    assert_eq!(j.references[2].1.to_str(), Some("/stage/ref2.mp4"));

    // 9 images + 3 videos = 12: ok. One more (audio) exceeds the total.
    r.references = (0..9)
        .map(|i| mk(MediaKind::Image, i))
        .chain((0..3).map(|i| mk(MediaKind::Video, 10 + i)))
        .collect();
    assert!(nego(&r, &h3_ref2va()).is_ok());
    r.references.push(mk(MediaKind::Audio, 20));
    let e = err_of(nego(&r, &h3_ref2va()));
    assert!(e.message.contains("in total"), "{}", e.message);
    r.references = (0..10).map(|i| mk(MediaKind::Image, i)).collect();
    assert!(err_of(nego(&r, &h3_ref2va()))
        .message
        .contains("reference images"));
    r.references = (0..4).map(|i| mk(MediaKind::Audio, i)).collect();
    assert_eq!(
        err_of(nego(&r, &h3_ref2va())).param.as_deref(),
        Some("references")
    );
}

#[test]
fn staged_inputs_must_match() {
    let mut r = t2v("fasth3", "x");
    r.task = Task::I2V;
    r.keyframes = vec![Keyframe {
        at: Anchor::First,
        image: url("https://e.x/f.png"),
    }];
    let e = negotiate(&r, &h3(), &StagedInputs::default()).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Internal);
    let mut st = stage_all(&r);
    st.keyframes[0].0 = Anchor::Last;
    assert_eq!(
        negotiate(&r, &h3(), &st).unwrap_err().kind,
        ErrorKind::Internal
    );
    let mut st = stage_all(&r);
    st.keyframes[0].1.mime = "video/mp4".into();
    let e = negotiate(&r, &h3(), &st).unwrap_err();
    assert_eq!(
        (e.kind, e.param.as_deref()),
        (ErrorKind::UnsupportedMedia, Some("image_url"))
    );
    // Unknown top-level types are left to ingestion.
    let mut st = stage_all(&r);
    st.keyframes[0].1.mime = "application/octet-stream".into();
    let j = negotiate(&r, &h3(), &st).unwrap();
    assert_eq!(j.keyframes, vec![(Anchor::First, "/stage/kf0.png".into())]);
}

// ---- knobs -------------------------------------------------------------------------------

#[test]
fn knobs_refused_not_dropped() {
    type Tweak = Box<dyn Fn(&mut GenerationRequest)>;
    let cases: Vec<(&str, Tweak)> = vec![
        (
            "guidance_scale",
            Box::new(|r| r.sampling.guidance = Some(1.0)),
        ),
        (
            "guidance_scale_2",
            Box::new(|r| r.sampling.guidance_2 = Some(1.0)),
        ),
        (
            "flow_shift",
            Box::new(|r| r.sampling.flow_shift = Some(3.0)),
        ),
        (
            "boundary_ratio",
            Box::new(|r| r.sampling.boundary_ratio = Some(0.9)),
        ),
        (
            "negative_prompt",
            Box::new(|r| r.negative_prompt = Some("blurry".into())),
        ),
    ];
    for (param, f) in &cases {
        let mut r = t2v("fasth3", "x");
        f(&mut r);
        let e = err_of(nego(&r, &h3()));
        assert_eq!(
            (e.kind, e.param.as_deref()),
            (ErrorKind::InvalidRequest, Some(*param))
        );
        // FastWan honours them all.
        let mut w = t2v("fastwan", "x");
        f(&mut w);
        assert!(nego(&w, &fastwan()).is_ok(), "{param}");
    }
    // An empty negative prompt is no request at all.
    let mut r = t2v("fasth3", "x");
    r.negative_prompt = Some(String::new());
    assert!(nego(&r, &h3()).is_ok());
    // LTX: steps honoured on a non-H3 family; boundary_ratio is not.
    let mut r = t2v("ltx", "x");
    r.sampling.steps = Some(30);
    assert_eq!(nego(&r, &ltx23()).unwrap().sampling.steps, Some(30));
    r.sampling.boundary_ratio = Some(0.5);
    assert_eq!(
        err_of(nego(&r, &ltx23())).param.as_deref(),
        Some("boundary_ratio")
    );
    // A non-H3 model without steps is InvalidRequest, not the H3 gap.
    let mut caps = fastwan();
    caps.knobs.steps = false;
    let mut w = t2v("fastwan", "x");
    w.sampling.steps = Some(4);
    assert_eq!(err_of(nego(&w, &caps)).kind, ErrorKind::InvalidRequest);
}

#[test]
fn knob_values_validated() {
    let mut w = t2v("fastwan", "x");
    w.sampling = SamplingOverrides {
        steps: Some(0),
        ..Default::default()
    };
    assert_eq!(
        err_of(nego(&w, &fastwan())).param.as_deref(),
        Some("num_inference_steps")
    );
    w.sampling = SamplingOverrides {
        guidance: Some(f32::NAN),
        ..Default::default()
    };
    assert_eq!(
        err_of(nego(&w, &fastwan())).param.as_deref(),
        Some("guidance_scale")
    );
    w.sampling = SamplingOverrides {
        flow_shift: Some(0.0),
        ..Default::default()
    };
    assert_eq!(
        err_of(nego(&w, &fastwan())).param.as_deref(),
        Some("flow_shift")
    );
    w.sampling = SamplingOverrides {
        boundary_ratio: Some(1.5),
        ..Default::default()
    };
    assert_eq!(
        err_of(nego(&w, &fastwan())).param.as_deref(),
        Some("boundary_ratio")
    );
    w.sampling = SamplingOverrides {
        steps: Some(8),
        guidance: Some(5.0),
        guidance_2: Some(3.0),
        flow_shift: Some(5.0),
        boundary_ratio: Some(0.875),
        ..Default::default()
    };
    assert_eq!(nego(&w, &fastwan()).unwrap().sampling, w.sampling);
    // The reference strengths belong to reference-to-video models only.
    for (s, param) in [
        (SamplingOverrides { reference_strength: Some(1.0), ..Default::default() }, "reference_strength"),
        (SamplingOverrides { reference_lora_strength: Some(1.0), ..Default::default() }, "reference_lora_strength"),
    ] {
        w.sampling = s;
        assert_eq!(err_of(nego(&w, &fastwan())).param.as_deref(), Some(param));
    }
}

#[test]
fn seed_handling() {
    let mut r = t2v("fasth3", "x");
    r.seed = Some(42);
    assert_eq!(nego(&r, &h3()).unwrap().seed, 42);
    r.seed = None;
    for _ in 0..32 {
        assert!(nego(&r, &h3()).unwrap().seed <= u32::MAX as u64);
    }
    let mut caps = h3();
    caps.knobs.seed = false;
    assert!(nego(&r, &caps).is_ok(), "a drawn seed is never refused");
    r.seed = Some(1);
    assert_eq!(err_of(nego(&r, &caps)).param.as_deref(), Some("seed"));
}

// ---- fps and audio --------------------------------------------------------------------

#[test]
fn fps_rules() {
    let mut r = t2v("fasth3", "x");
    r.timing.fps = Some(30);
    let e = err_of(nego(&r, &h3()));
    assert_eq!(
        (e.kind, e.param.as_deref()),
        (ErrorKind::InvalidRequest, Some("fps"))
    );
    let e = {
        let mut l = t2v("ltx", "x");
        l.timing.fps = Some(25);
        err_of(nego(&l, &ltx23()))
    };
    assert_eq!(
        (e.gap(), e.param.as_deref()),
        (Some(GapId::LtxFps), Some("fps"))
    );
    // Wan: container-only fps; frames from seconds use it.
    let mut w = t2v("fastwan", "x");
    w.timing.fps = Some(16);
    seconds(&mut w, 5.0);
    let j = nego(&w, &fastwan()).unwrap();
    assert_eq!((j.fps, j.num_frames), (16, 81));
}

#[test]
fn audio_plans() {
    let j = nego(&t2v("fasth3", "x"), &h3()).unwrap();
    assert_eq!(
        j.audio,
        AudioPlan::Native {
            rate: 32_000,
            channels: 2
        }
    );
    assert!(j.audio.has_audio() && !j.post.drop_audio);

    let mut l = t2v("ltx", "x");
    l.audio_out = AudioOut::Silent;
    let j = nego(&l, &ltx23()).unwrap();
    assert_eq!((j.audio, j.post.drop_audio), (AudioPlan::Drop, true));
    assert!(!j.audio.has_audio());

    let j = nego(&t2v("fastwan", "x"), &fastwan()).unwrap();
    assert_eq!(j.audio, AudioPlan::None);
    let mut w = t2v("fastwan", "x");
    w.audio_out = AudioOut::Silent;
    assert_eq!(nego(&w, &fastwan()).unwrap().audio, AudioPlan::None);
    w.audio_out = AudioOut::Sidecar;
    assert_eq!(err_of(nego(&w, &fastwan())).kind, ErrorKind::InvalidRequest);
    assert_eq!(nego(&w, &wan_sidecar()).unwrap().audio, AudioPlan::Sidecar);
    assert_eq!(
        nego(&t2v("wan_mm", "x"), &wan_sidecar()).unwrap().audio,
        AudioPlan::None
    );
    let mut h = t2v("fasth3", "x");
    h.audio_out = AudioOut::Sidecar;
    assert_eq!(
        err_of(nego(&h, &h3())).kind,
        ErrorKind::InvalidRequest,
        "native model has no sidecar"
    );

    // Input audio.
    let mut w = t2v("fastwan", "x");
    w.audio_in = Some(AudioInput {
        media: url("https://e.x/a.wav"),
        role: AudioRole::TargetSoundtrack,
    });
    assert_eq!(
        err_of(nego(&w, &fastwan())).param.as_deref(),
        Some("audio_url")
    );
    w.audio_in = Some(AudioInput {
        media: url("https://e.x/a.wav"),
        role: AudioRole::Drive,
    });
    assert_eq!(
        err_of(nego(&w, &fastwan())).param.as_deref(),
        Some("audio_url")
    );
    // A2V with driving audio passes once a model advertises A2V.
    let mut caps = ltx23();
    caps.tasks.insert(Task::A2V);
    let mut a = t2v("ltx", "x");
    a.task = Task::A2V;
    a.audio_in = Some(AudioInput {
        media: url("https://e.x/a.wav"),
        role: AudioRole::Drive,
    });
    let j = nego(&a, &caps).unwrap();
    assert_eq!(
        j.audio_in,
        Some((AudioRole::Drive, "/stage/audio.wav".into()))
    );
}

// ---- rule order, precheck, model resolution -----------------------------------------------

#[test]
fn rule_order_task_before_canvas_before_frames() {
    let mut r = t2v("ltx", "x");
    r.task = Task::Keyframes;
    r.keyframes = vec![Keyframe {
        at: Anchor::Last,
        image: url("https://e.x/l.png"),
    }];
    r.canvas = CanvasSpec::Exact {
        width: 100,
        height: 1000,
    };
    r.timing.length = Length::Auto;
    assert_eq!(gap_of(nego(&r, &ltx23())), Some(GapId::LtxKeyframes));
    r.task = Task::T2V;
    r.keyframes.clear();
    assert_eq!(err_of(nego(&r, &ltx23())).param.as_deref(), Some("size"));
    r.canvas = CanvasSpec::ModelDefault;
    assert_eq!(gap_of(nego(&r, &ltx23())), Some(GapId::LtxAutoDuration));
}

#[test]
fn precheck_matches_negotiate_on_success() {
    let mut r = t2v("fasth3", "x");
    seconds(&mut r, 10.0);
    r.seed = Some(3);
    assert!(precheck(&r, &h3()).is_ok());
    let j = nego(&r, &h3()).unwrap();
    assert_eq!(
        (
            j.model.as_str(),
            j.task,
            j.prompt.as_str(),
            j.negative_prompt.as_str(),
            j.seed
        ),
        ("fasth3", Task::T2V, "x", "", 3)
    );
    assert!((j.duration_s() - 243.0 / 24.0).abs() < 1e-9);
}

#[test]
fn resolve_model_through_aliases() {
    let models = [h3(), ltx23(), fastwan()];
    let aliases: BTreeMap<String, String> = [
        ("MiniMax-H3".to_string(), "fasth3".to_string()),
        ("MiniMax-H3-Max".to_string(), "fasth3".to_string()),
        ("ltx-2-3-fast".to_string(), "ltx2_distilled_23".to_string()),
    ]
    .into_iter()
    .collect();
    let look = |n: &str| aliases.get(n).cloned();
    assert_eq!(
        resolve_model("MiniMax-H3-Max", look, &models)
            .unwrap()
            .id
            .as_str(),
        "fasth3"
    );
    assert_eq!(
        resolve_model("ltx-2-3-fast", look, &models)
            .unwrap()
            .id
            .as_str(),
        "ltx2_distilled_23"
    );
    assert_eq!(
        resolve_model("fastwan", look, &models).unwrap().id.as_str(),
        "fastwan"
    );
    assert_eq!(
        resolve_model("FastWan2.2-TI2V-5B", look, &models)
            .unwrap()
            .id
            .as_str(),
        "fastwan"
    );
    let e = resolve_model("ltx-2-fast", look, &models).unwrap_err();
    assert_eq!(
        (e.kind, e.param.as_deref()),
        (ErrorKind::InvalidRequest, Some("model"))
    );
    // Works over a HashMap's values too (the engine's CapabilityTable).
    let table: std::collections::HashMap<ModelId, ModelCaps> =
        models.iter().map(|m| (m.id.clone(), m.clone())).collect();
    assert!(resolve_model("MiniMax-H3", look, table.values()).is_ok());
}

// ---- model tiers (design §0.3) ----------------------------------------------------

/// A max H3, a turbo FastH3, a max LTX-2.5 and an untiered LTX-2.3.
fn tiered() -> Vec<ModelCaps> {
    let mut h3max = ModelCaps::h3("h3_base", true).with_tier(Tier::Max, "h3-full");
    h3max.served_names = vec!["h3-max".into(), "MiniMax-H3-Max".into()];
    let mut h3turbo = ModelCaps::h3("fasth3", false).with_tier(Tier::Turbo, "fasth3-4step-vsa");
    h3turbo.served_names = vec!["h3-turbo".into(), "MiniMax-H3-Turbo".into()];
    let mut ltxmax = ltx25().with_tier(Tier::Max, "ltx25-full");
    ltxmax.served_names = vec!["ltx-2-5-pro".into()];
    vec![h3max, h3turbo, ltxmax, ltx23()]
}

#[test]
fn tiers_resolve_by_name_and_by_tier() {
    let models = tiered();
    let none = |_: &str| None;
    for (name, id) in [
        ("h3-max", "h3_base"),
        ("MiniMax-H3-Max", "h3_base"),
        ("h3-turbo", "fasth3"),
        ("MiniMax-H3-Turbo", "fasth3"),
        ("ltx-2-5-pro", "ltx2_distilled_25"),
    ] {
        assert_eq!(resolve_model(name, none, &models).unwrap().id.as_str(), id);
    }
    // A configured alias can route a public tier name to an engine id.
    let alias = |n: &str| (n == "ltx-turbo").then(|| "ltx2_distilled_23".to_owned());
    assert_eq!(
        resolve_model("ltx-turbo", alias, &models)
            .unwrap()
            .id
            .as_str(),
        "ltx2_distilled_23"
    );

    for (family, tier, id) in [
        (Family::H3, Tier::Max, "h3_base"),
        (Family::H3, Tier::Turbo, "fasth3"),
        (Family::Ltx2, Tier::Max, "ltx2_distilled_25"),
    ] {
        assert_eq!(resolve_tier(family, tier, &models).unwrap().id.as_str(), id);
    }
    // ltx23 is untiered, so no LTX turbo is served.
    let e = resolve_tier(Family::Ltx2, Tier::Turbo, &models).unwrap_err();
    assert_eq!(
        (e.kind, e.param.as_deref()),
        (ErrorKind::InvalidRequest, Some("model"))
    );
    assert!(e.message.contains("turbo"), "{}", e.message);
}

#[test]
fn resolved_job_carries_tier_and_recipe() {
    let models = tiered();
    let j = nego(&t2v("h3-turbo", "a cat"), &models[1]).unwrap();
    assert_eq!(j.tier, Some(Tier::Turbo));
    assert_eq!(j.recipe.as_deref(), Some("fasth3-4step-vsa"));
    let j = nego(&t2v("fasth3", "a cat"), &h3()).unwrap();
    assert_eq!((j.tier, j.recipe), (None, None));
}

// ---- H3 reference-to-video (docs/ports/h3-ref2v.md) --------------------------------

fn h3_tier(id: &str, ref2va_only: bool, tier: Tier) -> ModelCaps {
    let mut c = ModelCaps::h3(id, ref2va_only);
    if ref2va_only {
        c.tasks = [Task::Ref2V].into_iter().collect();
    }
    c.tier = Some(tier);
    c
}

fn image_ref(u: &str) -> Reference {
    Reference { kind: MediaKind::Image, media: url(u) }
}

#[test]
fn ref2v_routes_to_the_tier_companion() {
    let base = h3_tier("sol-h3", false, Tier::Max);
    let companion = h3_tier("h3-ref2v-max", true, Tier::Max);
    let turbo = h3_tier("fasth3", false, Tier::Turbo);
    let untiered = ModelCaps::h3("plain", false);
    let models = [companion.clone(), base.clone(), turbo.clone(), untiered.clone()];
    // The tier resolves to its text-to-video model even when the companion
    // comes first.
    assert_eq!(resolve_tier(Family::H3, Tier::Max, &models).unwrap().id, base.id);
    assert_eq!(route_task(&base, Task::Ref2V, &models).id, companion.id);
    assert_eq!(route_task(&base, Task::T2V, &models).id, base.id);
    // No companion at that tier, or an untiered model: unchanged, so
    // negotiate reports the gap.
    assert_eq!(route_task(&turbo, Task::Ref2V, &models).id, turbo.id);
    assert_eq!(route_task(&untiered, Task::Ref2V, &models).id, untiered.id);
    let mut r = t2v("h3-max", "x");
    r.task = Task::Ref2V;
    r.references = vec![image_ref("https://e.x/r.png")];
    assert_eq!(gap_of(nego(&r, &turbo)), Some(GapId::H3Ref2vaNotLoaded));
    assert!(nego(&r, &companion).is_ok());
}

#[test]
fn ref2v_clip_lengths_follow_the_minimax_limits() {
    let caps = h3_ref2va();
    let clip = |kind: MediaKind, secs: f64, i: usize| {
        let (mime, name) = match kind {
            MediaKind::Video => ("video/mp4", format!("v{i}.mp4")),
            _ => ("audio/wav", format!("a{i}.wav")),
        };
        let mut m = staged(mime, (kind == MediaKind::Video).then_some((1280, 720)), &name);
        m.probe.duration_s = Some(secs);
        (kind, m)
    };
    let run = |clips: Vec<(MediaKind, StagedMedia)>| {
        let mut r = t2v("fasth3", "x");
        r.task = Task::Ref2V;
        r.references = clips
            .iter()
            .map(|(k, _)| Reference { kind: *k, media: url("https://e.x/c") })
            .collect();
        let staged = StagedInputs { references: clips, ..StagedInputs::default() };
        negotiate(&r, &caps, &staged)
    };
    assert!(run(vec![clip(MediaKind::Video, 5.0, 0), clip(MediaKind::Audio, 15.02, 1)]).is_ok());
    assert!(run(vec![clip(MediaKind::Video, 7.5, 0), clip(MediaKind::Video, 7.5, 1)]).is_ok());
    for bad in [
        vec![clip(MediaKind::Video, 1.5, 0)],
        vec![clip(MediaKind::Video, 16.0, 0)],
        vec![clip(MediaKind::Video, 8.0, 0), clip(MediaKind::Video, 8.0, 1)],
        vec![clip(MediaKind::Video, 4.0, 0), clip(MediaKind::Audio, 9.0, 1), clip(MediaKind::Audio, 9.0, 2)],
    ] {
        let e = run(bad).unwrap_err();
        assert_eq!((e.kind, e.param.as_deref()), (ErrorKind::InvalidRequest, Some("references")));
    }
}

#[test]
fn ref2v_adaptive_canvas_follows_the_first_image_else_the_first_video() {
    let caps = h3_ref2va();
    let mut r = t2v("fasth3", "x");
    r.task = Task::Ref2V;
    r.canvas = CanvasSpec::FollowImage { short_edge: 768 };
    r.references = vec![
        Reference { kind: MediaKind::Video, media: url("https://e.x/v.mp4") },
        image_ref("https://e.x/i.png"),
    ];
    let mut staged = stage_all(&r);
    staged.references[1].1 = image("i.png", 768, 1344);
    let j = negotiate(&r, &caps, &staged).unwrap();
    assert!(j.height > j.width, "the portrait image wins over the landscape video");
    r.references.truncate(1);
    staged.references.truncate(1);
    let j = negotiate(&r, &caps, &staged).unwrap();
    assert!(j.width > j.height, "video only: the video's landscape canvas");
}
