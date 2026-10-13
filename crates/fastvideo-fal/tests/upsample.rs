//! Prompt upsampling on the fal queue (docs/serve/prompt-upsampling.md):
//! how each fal flag normalizes, and what a job does with it when the
//! server has an upsampler (a fake chat backend here): explicit flags
//! rewrite, protocol defaults only with `default_on`, `disabled` never, a
//! failed call falls back to the client's prompt, and `expanded_prompt`
//! carries the rewrite.

mod queue_common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::http::StatusCode;
use fastvideo_fal::{Endpoint, FalApp, FalInput};
use fastvideo_protocol::{ExpansionMode, GenerationRequest, NormalizeCtx, PromptExpansion, ProtocolId};
use fastvideo_upsampler::{ChatBackend, ChatRequest, Service, Settings, UpsampleError};
use queue_common::*;
use serde_json::{json, Value};
use time::macros::datetime;

fn norm(app: &str, e: Endpoint, body: Value) -> GenerationRequest {
    let app = FalApp::from_id(app);
    let cx = NormalizeCtx::new(datetime!(2026-10-13 12:00 UTC));
    let (model, _) = app.target(e);
    FalInput::parse_for(app.kind(), e, &body).unwrap().normalize(&model, &cx).unwrap()
}

#[test]
fn fal_flags_normalize() {
    let pe = |b: Value| norm("minimax/h3-max", Endpoint::TextToVideo, b).prompt_expansion;
    const F: &str = "prompt_expansion_mode";
    assert_eq!(pe(json!({"prompt": "p"})), Some(PromptExpansion::default_for(ExpansionMode::Balanced, F)));
    assert_eq!(pe(json!({"prompt": "p", "prompt_expansion_mode": null})), Some(PromptExpansion::default_for(ExpansionMode::Balanced, F)));
    assert_eq!(pe(json!({"prompt": "p", "prompt_expansion_mode": "quality"})), Some(PromptExpansion::explicit(ExpansionMode::Quality, F)));
    assert_eq!(pe(json!({"prompt": "p", "prompt_expansion_mode": "balanced"})), Some(PromptExpansion::explicit(ExpansionMode::Balanced, F)));
    assert_eq!(pe(json!({"prompt": "p", "prompt_expansion_mode": "fast"})), Some(PromptExpansion::explicit(ExpansionMode::Balanced, F)));
    assert_eq!(pe(json!({"prompt": "p", "prompt_expansion_mode": "disabled"})), Some(PromptExpansion::explicit(ExpansionMode::Disabled, F)));
    // Any other string stays accepted, with no effect.
    assert_eq!(pe(json!({"prompt": "p", "prompt_expansion_mode": "turbo"})), None);
    // Still a no-op note at normalize time (the goldens are unchanged).
    assert!(norm("minimax/h3-max", Endpoint::TextToVideo, json!({"prompt": "p"})).accepted_noop.contains(&F));

    // Wan: `enable_prompt_expansion` (default false) opts in.
    const E: &str = "enable_prompt_expansion";
    let wan = |b: Value| norm("fal-ai/wan", Endpoint::WanTextToVideo, b).prompt_expansion;
    assert_eq!(wan(json!({"prompt": "p"})), None);
    assert_eq!(wan(json!({"prompt": "p", "enable_prompt_expansion": false})), None);
    assert_eq!(wan(json!({"prompt": "p", "enable_prompt_expansion": true})), Some(PromptExpansion::explicit(ExpansionMode::Balanced, E)));

    // Ingredient: fal's default is on (applied only with `default_on`);
    // read leniently (a non-boolean is ignored, as before).
    let ing = |b: Value| norm("fal-ai/ltx-2.3-quality", Endpoint::LtxIngredient, b).prompt_expansion;
    let img = "https://a.test/sheet.png";
    assert_eq!(ing(json!({"prompt": "p", "image_url": img})), Some(PromptExpansion::default_for(ExpansionMode::Balanced, E)));
    assert_eq!(ing(json!({"prompt": "p", "image_url": img, "enable_prompt_expansion": false})), Some(PromptExpansion::explicit(ExpansionMode::Disabled, E)));
    assert_eq!(ing(json!({"prompt": "p", "image_url": img, "enable_prompt_expansion": "yes"})), Some(PromptExpansion::default_for(ExpansionMode::Balanced, E)));

    // LTX partner endpoints: no fal switch; ours opts in explicitly.
    let ltx = |b: Value| norm("lightricks/ltx-2.5", Endpoint::LtxTextToVideoFast, b).prompt_expansion;
    assert_eq!(ltx(json!({"prompt": "p"})), None);
    assert_eq!(ltx(json!({"prompt": "p", "enable_prompt_expansion": true})), Some(PromptExpansion::explicit(ExpansionMode::Balanced, E)));
    assert_eq!(ltx(json!({"prompt": "p", "enable_prompt_expansion": 1})), None);
}

/// Rewrites to `Rewritten: <prompt>`, or fails; records system prompts.
struct FakeLlm {
    fail: bool,
    systems: Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl ChatBackend for FakeLlm {
    async fn complete(&self, req: &ChatRequest) -> Result<String, UpsampleError> {
        self.systems.lock().unwrap().push(req.system.clone());
        if self.fail {
            return Err(UpsampleError::Status { status: 503, body: "overloaded".into() });
        }
        Ok(format!("\"Rewritten: {}\"", req.user))
    }
}

async fn fixture_with(fail: bool, default_on: bool) -> (Fixture, Arc<FakeLlm>) {
    let llm = Arc::new(FakeLlm { fail, systems: Mutex::new(Vec::new()) });
    let settings = Settings { model: "fake".into(), hedge: false, timeout: Duration::from_secs(5), default_on, ..Settings::default() };
    let svc = Arc::new(Service::new(llm.clone(), settings));
    (fixture(Opts { upsampler: Some(svc), ..Opts::default() }).await, llm)
}

/// Submits a t2v, waits for it, returns (result body, job).
async fn run(f: &Fixture, body: Value) -> (Value, fastvideo_protocol::Job) {
    let r = call(&f.app, "POST", "/minimax/h3-max/text-to-video", Some(body)).await;
    assert_eq!(r.status, StatusCode::OK, "{:?}", r.json());
    let s = r.json();
    wait_completed(&f.app, &path_of(s["status_url"].as_str().unwrap())).await;
    let out = call(&f.app, "GET", &path_of(s["response_url"].as_str().unwrap()), None).await;
    assert_eq!(out.status, StatusCode::OK);
    let job = f.ctx.jobs().by_external(ProtocolId::Fal, s["request_id"].as_str().unwrap()).await.unwrap();
    (out.json(), job)
}

#[tokio::test]
async fn explicit_modes_rewrite_and_defaults_wait_for_default_on() {
    let (f, llm) = fixture_with(false, false).await;
    // Explicit `quality`: rewritten, returned, generated with.
    let (out, job) = run(&f, json!({"prompt": "a kitten", "prompt_expansion_mode": "quality"})).await;
    assert_eq!(out["expanded_prompt"], "integrated_multimodal_description: Rewritten: a kitten", "H3: Context-IR layout");
    assert_eq!(job.resolved.prompt, "integrated_multimodal_description: Rewritten: a kitten");
    assert_eq!(job.expanded_prompt.as_deref(), Some("integrated_multimodal_description: Rewritten: a kitten"));
    assert!(job.logs.iter().any(|l| l.message.contains("prompt expanded by the upsampler (prompt_expansion_mode `quality`, requested)")), "{:?}", job.logs);
    let sys = llm.systems.lock().unwrap().last().cloned().unwrap();
    assert!(sys.contains("camera language") && sys.contains("120 to 220 words") && sys.contains("[Shot 1]") && sys.contains("3200 characters"), "{sys}");
    // Omitted: fal's default `balanced` is not applied without default_on.
    let (out, job) = run(&f, json!({"prompt": "a puppy"})).await;
    assert_eq!(out["expanded_prompt"], Value::Null);
    assert_eq!(job.resolved.prompt, "a puppy");
    // `disabled`: never.
    let (out, _) = run(&f, json!({"prompt": "a puppy", "prompt_expansion_mode": "disabled"})).await;
    assert_eq!(out["expanded_prompt"], Value::Null);
    assert_eq!(llm.systems.lock().unwrap().len(), 1, "one LLM call in all");

    // default_on: the omitted flag follows fal's default (balanced).
    let (f, llm) = fixture_with(false, true).await;
    let (out, _) = run(&f, json!({"prompt": "a puppy"})).await;
    assert_eq!(out["expanded_prompt"], "integrated_multimodal_description: Rewritten: a puppy");
    assert!(llm.systems.lock().unwrap()[0].contains("60 to 110 words"));
    let (out, _) = run(&f, json!({"prompt": "a puppy", "prompt_expansion_mode": "disabled"})).await;
    assert_eq!(out["expanded_prompt"], Value::Null);
}

#[tokio::test]
async fn a_failed_rewrite_falls_back_to_the_prompt() {
    let (f, llm) = fixture_with(true, false).await;
    let (out, job) = run(&f, json!({"prompt": "a kitten", "prompt_expansion_mode": "balanced"})).await;
    assert_eq!(out["expanded_prompt"], Value::Null);
    assert_eq!(job.resolved.prompt, "a kitten");
    assert!(job.logs.iter().any(|l| l.message.contains("prompt expansion failed (upsampler LLM 503: overloaded); generating with the original prompt")), "{:?}", job.logs);
    assert_eq!(llm.systems.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn without_an_upsampler_nothing_changes() {
    let f = fixture(Opts::default()).await;
    let (out, job) = run(&f, json!({"prompt": "a kitten", "prompt_expansion_mode": "quality"})).await;
    assert_eq!(out["expanded_prompt"], Value::Null);
    assert_eq!(job.resolved.prompt, "a kitten");
    assert!(job.expanded_prompt.is_none());
    assert!(!job.logs.iter().any(|l| l.message.contains("prompt expan")));
}
