//! `EngineService::start_negotiated` (plan D1 follow-up 4, `fv-gpucheck
//! engine`): a request is negotiated against the caps the engine will serve
//! before any backend loads a model. The gold run at 7e580e5 loaded
//! `h3-ref2v-max`, `h3-ref2v-turbo`, `ltx25-ref2v` and `ltx25-a2v-guided` for
//! 5-199 s before `negotiate` refused its text-to-video request; here the
//! same refusal comes with no load, and what is accepted is unchanged.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fastvideo_engine_service::cuda::{catalog, WeightLayout};
use fastvideo_engine_service::{
    BlockInput, BlockStats, CausalSpec, ClipOutput, ClipSink, DeviceInfo, EngineBackend,
    EngineConfig, EngineService, FakeBackend, FakeConfig, FakeTiming, LoadEvent, Mp4Mode,
    Readiness, Recipe, SessionId, StartNegotiatedError, StepControl,
};
use fastvideo_protocol::{
    negotiate, ApiError, GenerationRequest, ModelCaps, ModelId, ProtocolId, ResolvedJob,
    StagedInputs,
};

/// A fake backend that counts its `load` calls.
struct Counting {
    inner: FakeBackend,
    loads: Arc<AtomicUsize>,
}

impl EngineBackend for Counting {
    fn device(&self) -> DeviceInfo {
        self.inner.device()
    }
    fn caps(&self) -> Vec<ModelCaps> {
        self.inner.caps()
    }
    fn recipe(&self, model: &ModelId) -> Recipe {
        self.inner.recipe(model)
    }
    fn public_names(&self, model: &ModelId) -> Vec<String> {
        self.inner.public_names(model)
    }
    fn load(&mut self, model: &ModelId, obs: &mut dyn FnMut(LoadEvent)) -> Result<(), ApiError> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.inner.load(model, obs)
    }
    fn generate(
        &mut self,
        job: &ResolvedJob,
        out: &mut dyn ClipSink,
        ctl: &StepControl,
    ) -> Result<ClipOutput, ApiError> {
        self.inner.generate(job, out, ctl)
    }
    fn causal_open(&mut self, s: SessionId, spec: &CausalSpec) -> Result<(), ApiError> {
        self.inner.causal_open(s, spec)
    }
    fn causal_block(
        &mut self,
        s: SessionId,
        input: &BlockInput,
        out: &mut dyn ClipSink,
        ctl: &StepControl,
    ) -> Result<BlockStats, ApiError> {
        self.inner.causal_block(s, input, out, ctl)
    }
    fn causal_close(&mut self, s: SessionId) {
        self.inner.causal_close(s)
    }
}

/// The CUDA catalog's models (their real caps, every one resident) on a fast
/// fake, `only` of them when given, and the backend's load counter.
fn catalog_backend(only: Option<&[&str]>) -> (Vec<Box<dyn EngineBackend>>, Arc<AtomicUsize>) {
    let layout = WeightLayout::new("/synthetic/weights").with_tae_dir("/synthetic/tae");
    let cat = catalog(&layout).expect("the family manifests build the catalog");
    let mut cfg = FakeConfig {
        timing: FakeTiming {
            step: Duration::from_millis(1),
            load: Duration::from_millis(4),
            ..FakeTiming::default()
        },
        mp4: Mp4Mode::Off,
        ..FakeConfig::cuda_catalog(&cat)
    };
    if let Some(ids) = only {
        cfg = cfg.with_models(ids);
    }
    let loads = Arc::new(AtomicUsize::new(0));
    let backend = Counting {
        inner: FakeBackend::new(cfg),
        loads: loads.clone(),
    };
    (vec![Box::new(backend)], loads)
}

fn t2v(model: &str) -> GenerationRequest {
    let mut req = GenerationRequest::text(ProtocolId::Native, model, "a cat on a windowsill");
    req.seed = Some(42);
    req
}

/// The four variants the gold run loaded before refusing t2v are refused
/// before any load, with the error `negotiate` gives.
#[test]
fn a_task_the_model_does_not_serve_is_refused_before_any_load() {
    for id in [
        "h3-ref2v-max",
        "h3-ref2v-turbo",
        "ltx25-ref2v",
        "ltx25-a2v-guided",
    ] {
        let (backends, loads) = catalog_backend(None);
        let model = ModelId::from(id);
        let err = match EngineService::start_negotiated(
            EngineConfig::default(),
            backends,
            &model,
            &t2v(id),
            &StagedInputs::default(),
        ) {
            Err(e) => e,
            Ok(_) => panic!("{id}: a t2v request was accepted"),
        };
        let StartNegotiatedError::Negotiate(api) = &err else {
            panic!("{id}: {err}");
        };
        assert!(
            api.message.contains("does not support task t2v"),
            "{id}: {api:?}"
        );
        assert!(err.to_string().starts_with("negotiate: "), "{err}");
        // Nothing started, so nothing loaded (and nothing will).
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(
            loads.load(Ordering::SeqCst),
            0,
            "{id}: loaded before refusing"
        );
    }
}

/// An unknown model is refused before any load too.
#[test]
fn an_unknown_model_is_refused_before_any_load() {
    let (backends, loads) = catalog_backend(None);
    let model = ModelId::from("no-such-model");
    let r = EngineService::start_negotiated(
        EngineConfig::default(),
        backends,
        &model,
        &t2v("no-such-model"),
        &StagedInputs::default(),
    );
    assert!(
        matches!(r, Err(StartNegotiatedError::UnknownModel(ref m)) if *m == model),
        "{:?}",
        r.err()
    );
    assert_eq!(loads.load(Ordering::SeqCst), 0);
}

/// An accepted request starts the engine, which then loads; the caps and
/// the job are what `negotiate` gives against the started engine's own
/// table, so nothing accepted before is refused now (or the other way).
#[tokio::test]
async fn an_accepted_request_starts_the_engine_with_the_same_caps_and_job() {
    let id = "fasth3-4step-vsa";
    let (backends, loads) = catalog_backend(Some(&[id, "h3-ref2v-turbo"]));
    let model = ModelId::from(id);
    let req = t2v(id);
    let (engine, caps, job) = EngineService::start_negotiated(
        EngineConfig::default(),
        backends,
        &model,
        &req,
        &StagedInputs::default(),
    )
    .expect("t2v on a t2v model is accepted");
    assert_eq!(engine.wait_ready().await, Readiness::Ready);
    assert_eq!(loads.load(Ordering::SeqCst), 2, "both resident models load");
    assert_eq!(engine.caps().get(&model), Some(&caps));
    assert_eq!(
        negotiate(&req, &caps, &StagedInputs::default()).ok(),
        Some(job)
    );
    drop(engine);
}

/// `caps_table` is the table `start` serves: every model's caps (tiers
/// rebound across models included) and the public aliases, for the whole
/// catalog on one backend.
#[tokio::test]
async fn caps_table_is_the_started_engines_table() {
    let (backends, _) = catalog_backend(None);
    let cfg = EngineConfig::default();
    let before = EngineService::caps_table(&cfg, &backends).expect("table");
    let engine = EngineService::start(cfg, backends).expect("start");
    let after = engine.caps();
    let all =
        |t: &fastvideo_engine_service::CapabilityTable| t.models().cloned().collect::<Vec<_>>();
    assert!(!all(&before).is_empty());
    assert_eq!(all(&before), all(after));
    assert_eq!(before.public_aliases(), after.public_aliases());
    drop(engine);
}
