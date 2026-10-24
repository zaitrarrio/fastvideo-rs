//! ADR-0006 acceptance: valid fixtures parse and round-trip through `toml`
//! serialisation; invalid ones fail with the kind and field path of their
//! `.expect` file; the two widenings parse; `[caps]` with `min > max` is
//! rejected.

mod common;

use fastvideo_manifest::model_toml::{ModelToml, ModelTomlError};

#[test]
fn valid_fixtures_parse_and_round_trip() {
    for p in common::fixture_files("model-toml/valid", "toml") {
        let text = common::read(&p);
        let mt = ModelToml::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        assert_eq!(mt.schema, 1);
        let out = mt
            .to_toml_string()
            .unwrap_or_else(|e| panic!("{}: serialise: {e}", p.display()));
        let back = ModelToml::parse(&out)
            .unwrap_or_else(|e| panic!("{}: re-parse: {e}\n{out}", p.display()));
        assert_eq!(back, mt, "{}: round trip", p.display());
        // And through JSON, which is how the registry API will carry it.
        let json = serde_json::to_string(&mt).unwrap();
        let from_json: ModelToml = serde_json::from_str(&json).unwrap();
        assert_eq!(from_json, mt);
    }
}

#[test]
fn invalid_fixtures_fail_with_kind_and_path() {
    let files = common::fixture_files("model-toml/invalid", "toml");
    assert!(files.len() >= 20, "{} invalid fixtures", files.len());
    for p in files {
        let text = common::read(&p);
        let expect = common::read(&p.with_extension("expect"));
        let (kind, path) = expect
            .trim()
            .split_once(' ')
            .unwrap_or_else(|| panic!("{}: bad .expect", p.display()));
        let err = ModelToml::parse(&text)
            .err()
            .unwrap_or_else(|| panic!("{} parsed; expected {kind} {path}", p.display()));
        assert_eq!(err.kind(), kind, "{}: {err}", p.display());
        if path != "-" {
            assert_eq!(err.path(), Some(path), "{}: {err}", p.display());
        }
    }
}

#[test]
fn unknown_keys_anywhere_name_their_path() {
    let text = common::read_fixture("model-toml/valid/fork-widened.toml");
    for (anchor, path) in [
        ("[runtime]\n", "runtime.bogus"),
        ("[variant]\n", "variant.bogus"),
        ("[caps]\n", "caps.bogus"),
        ("[change]\n", "change.bogus"),
    ] {
        let bad = text.replacen(anchor, &format!("{anchor}bogus = 1\n"), 1);
        assert_eq!(
            ModelToml::parse(&bad).unwrap_err(),
            ModelTomlError::UnknownKey {
                path: path.to_string()
            },
            "{anchor}"
        );
    }
}

#[test]
fn widenings_parse() {
    let text = common::read_fixture("model-toml/valid/fork-widened.toml");
    let mt = ModelToml::parse(&text).unwrap();
    assert_eq!(mt.runtime.recipe.as_deref(), Some("4step-vsa"));
    assert_eq!(mt.runtime.profile.as_deref(), Some("h3/fasth3_4step_vsa"));
    let v = mt.variant.as_ref().expect("[variant]");
    assert_eq!(v.attention_route.as_deref(), Some("vsa"));
    assert_eq!(v.dit_quant.as_deref(), Some("bf16"));
    assert_eq!(v.decoder.as_deref(), Some("tae"));
    assert_eq!(
        v.adapters,
        vec!["fasth3-4-step-preview-v1-lora".to_string()]
    );
    assert_eq!(v.stages, Some(1));
    assert_eq!(v.tiers, vec!["turbo".to_string()]);
    let caps = mt.caps.as_ref().expect("[caps]");
    let frames = caps.frames.as_ref().unwrap();
    assert_eq!(
        (frames.step, frames.offset, frames.min, frames.max),
        (17, 5, 107, 362)
    );
    let canvas = caps.canvas.as_ref().unwrap();
    assert_eq!(canvas.multiple, 32);
    assert_eq!(canvas.max_area, 1_032_192);
    assert_eq!(canvas.aspect, [0.25, 4.0]);
    assert_eq!(canvas.short_edges, vec![768, 1080]);
    let fps = caps.fps.as_ref().unwrap();
    assert_eq!((fps.allowed.clone(), fps.default), (vec![24], 24));
    let refs = caps.refs.as_ref().unwrap();
    assert_eq!(
        (refs.images, refs.videos, refs.audios, refs.total),
        (9, 3, 3, 12)
    );
    let knobs = caps.knobs.as_ref().unwrap();
    assert!(
        knobs.seed
            && !knobs.negative
            && !knobs.steps
            && !knobs.guidance
            && !knobs.guidance_2
            && !knobs.flow_shift
    );
}

#[test]
fn caps_min_greater_than_max_is_rejected() {
    let text = common::read_fixture("model-toml/valid/vendor-h3-caps.toml");
    let bad = text.replace("min = 107\nmax = 362", "min = 362\nmax = 107");
    assert_ne!(bad, text);
    let err = ModelToml::parse(&bad).unwrap_err();
    assert_eq!(err.kind(), "Field");
    assert_eq!(err.path(), Some("caps.frames"));
    // The same check via the typed API.
    let mut mt = ModelToml::parse(&text).unwrap();
    let frames = mt.caps.as_mut().unwrap().frames.as_mut().unwrap();
    std::mem::swap(&mut frames.min, &mut frames.max);
    assert_eq!(mt.validate().unwrap_err().path(), Some("caps.frames"));
}

#[test]
fn the_wo001_example_is_a_fork_with_a_pinned_parent() {
    let text = common::read_fixture("model-toml/valid/fork-dmd.toml");
    let mt = ModelToml::parse(&text).unwrap();
    assert!(mt.is_fork());
    assert_eq!(mt.name.to_string(), "zay/wan2.2-t2v/dmd:4:fp8");
    let parent = mt.parent.as_ref().unwrap();
    assert_eq!(parent.rev().unwrap().as_str(), "0123456789ab");
    assert_eq!(mt.defaults.guidance, Some(1.0));
    assert_eq!(mt.components.len(), 3);
    assert_eq!(mt.components["text_encoder"].dtype, "fp8");
    assert_eq!(mt.change.as_ref().unwrap().summary, "Distilled to 4 steps");
    assert!(mt.variant.is_none() && mt.caps.is_none());
}
