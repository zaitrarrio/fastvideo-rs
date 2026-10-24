//! OpenTelemetry export of the request traces (cargo feature `otel`;
//! docs/serve/tracing.md "OpenTelemetry export").
//!
//! One OTLP exporter (http/protobuf, the SDK's batch processor on its own
//! thread) receives three kinds of spans:
//!
//! - the trace recorder's events (`fastvideo-trace`, docs/serve/tracing.md
//!   "Event list"): every drained event becomes a finished span under the
//!   request's W3C trace id ([`export_event`], called from the recorder's
//!   sink on the drain thread, never on a request's path) with the event's
//!   fields as `fv.*` attributes. The job's phases (`queue`, `engine`,
//!   `gpu`, `post`, `upload` and the terminal store write) are children of
//! - the `job` span of a traced job ([`JobSpan`]: the gate's pump opens it
//!   at submit and ends it at the terminal event) carrying the job id,
//!   model, api, status and the queue-wait, inference and total durations;
//!   so a collector shows one trace per job: the request spans, the `job`
//!   span and its phases, step by step;
//! - this process's `tracing` spans through a `tracing-opentelemetry`
//!   layer ([`init`] returns it for `main.rs`'s `init_tracing`); they carry
//!   their own trace ids (at the default `info` filter there are none).
//!
//! The recorder has no span ids (its hot path pushes an 80-byte record), so
//! the spans are built here as [`SpanData`] and handed to the batch
//! processor directly: the `job` span's id is derived from the trace id
//! ([`job_span_id`]) and the phase spans name it as their parent. Head
//! sampling is by trace id ([`sampled`] is `Sampler::TraceIdRatioBased`'s
//! rule, which the provider applies to the `tracing` spans), so a trace is
//! exported whole or not at all. Nothing here waits on the network.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fastvideo_trace::{Event, Trace, TraceId as FvTraceId};
use opentelemetry::trace::{SpanContext, SpanKind, Status, TraceState, TracerProvider as _};
use opentelemetry::{Context, InstrumentationScope, KeyValue, SpanId, TraceFlags, TraceId};
use opentelemetry_otlp::{Protocol, WithExportConfig as _};
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{
    BatchSpanProcessor, Sampler, SdkTracer, SdkTracerProvider, Span, SpanData, SpanEvents,
    SpanExporter, SpanLinks, SpanProcessor,
};
use opentelemetry_sdk::Resource;
use tracing_subscriber::registry::LookupSpan;

use crate::config::OtelCfg;

/// The `tracing` layer [`init`] returns.
pub type OtelLayer<S> = tracing_opentelemetry::OpenTelemetryLayer<S, SdkTracer>;

/// Why the exporter could not be set up.
#[derive(Debug, thiserror::Error)]
pub enum OtelError {
    #[error("{0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("[otel] exporter: {0}")]
    Exporter(#[from] opentelemetry_otlp::ExporterBuildError),
    #[error("[otel] initialised twice")]
    AlreadyInitialised,
}

/// The provider's span processor: the batch processor the recorder's spans
/// go to as well, so both leave through one exporter and one HTTP client.
/// The resource is set on the inner processor by [`Exporter::with_exporter`]
/// (the trait's `set_resource` takes `&mut self`; this one keeps the default
/// no-op).
#[derive(Debug)]
struct Shared(Arc<BatchSpanProcessor>);

impl SpanProcessor for Shared {
    fn on_start(&self, span: &mut Span, cx: &Context) {
        self.0.on_start(span, cx);
    }

    fn on_end(&self, span: SpanData) {
        self.0.on_end(span);
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.0.force_flush()
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.0.shutdown_with_timeout(timeout)
    }
}

/// The exporter: the SDK provider (for the `tracing` layer) and the batch
/// processor (for the spans built here).
pub struct Exporter {
    provider: SdkTracerProvider,
    processor: Arc<BatchSpanProcessor>,
    ratio: f64,
    scope: InstrumentationScope,
}

impl std::fmt::Debug for Exporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Exporter")
            .field("ratio", &self.ratio)
            .finish_non_exhaustive()
    }
}

/// A span to hand to the processor as finished.
struct Raw {
    trace: TraceId,
    span: SpanId,
    parent: SpanId,
    name: String,
    start: SystemTime,
    end: SystemTime,
    attributes: Vec<KeyValue>,
    status: Status,
}

impl Exporter {
    /// The OTLP/HTTP exporter for `cfg`. No network: the first export is
    /// the first batch, on the processor's thread.
    pub fn build(cfg: &OtelCfg) -> Result<Self, OtelError> {
        cfg.validate()?;
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(cfg.endpoint.clone())
            .with_timeout(cfg.timeout())
            .build()?;
        Ok(Self::with_exporter(cfg, exporter))
    }

    /// `cfg`'s sampling and service name over any exporter (tests keep the
    /// spans in memory).
    pub fn with_exporter<E: SpanExporter + 'static>(cfg: &OtelCfg, exporter: E) -> Self {
        let resource = Resource::builder()
            .with_service_name(cfg.service_name.clone())
            .build();
        let mut processor = BatchSpanProcessor::builder(exporter).build();
        processor.set_resource(&resource);
        let processor = Arc::new(processor);
        let provider = SdkTracerProvider::builder()
            .with_span_processor(Shared(Arc::clone(&processor)))
            .with_sampler(Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(
                cfg.sample_ratio,
            ))))
            .with_resource(resource)
            .build();
        Self {
            provider,
            processor,
            ratio: cfg.sample_ratio,
            scope: InstrumentationScope::builder("fastvideo-trace")
                .with_version(env!("CARGO_PKG_VERSION"))
                .build(),
        }
    }

    /// The `tracing` layer over this exporter's provider.
    pub fn layer<S>(&self) -> OtelLayer<S>
    where
        S: tracing::Subscriber + for<'span> LookupSpan<'span>,
    {
        tracing_opentelemetry::layer().with_tracer(self.provider.tracer("fv-serve"))
    }

    /// One recorder event as a finished span under its trace id (the
    /// recorder's sink calls this on the drain thread).
    pub fn export_event(&self, e: &Event) {
        let Some(fv) = FvTraceId::parse_hex(&e.trace) else {
            return;
        };
        if !sampled(self.ratio, fv) {
            return;
        }
        let start = wall(e.t_wall_ns);
        let mut attributes = vec![
            KeyValue::new("fv.trace_id", e.trace.clone()),
            KeyValue::new("fv.comp", e.comp.clone()),
            KeyValue::new("fv.stage", e.name.clone()),
            KeyValue::new("fv.clock", e.clock.clone()),
            KeyValue::new("fv.host", e.host.clone()),
            KeyValue::new("fv.dur_ms", e.dur_ns as f64 / 1e6),
        ];
        if let Some(a) = e.arg {
            attributes.push(KeyValue::new("fv.arg", a));
        }
        if let Some(t) = e.t_mono_ns {
            attributes.push(KeyValue::new("fv.t_mono_ns", t as i64));
        }
        if let Some(a) = &e.attrs {
            if let Ok(s) = serde_json::to_string(a) {
                attributes.push(KeyValue::new("fv.attrs", s));
            }
        }
        let parent = if is_job_phase(&e.comp, &e.name) {
            job_span_id(fv)
        } else {
            SpanId::INVALID
        };
        self.emit(Raw {
            trace: TraceId::from_bytes(fv.0),
            span: new_span_id(),
            parent,
            name: format!("{}.{}", e.comp, e.name),
            start,
            // `dur_ns` may come from the ingest endpoint: no overflow panic.
            end: start
                .checked_add(Duration::from_nanos(e.dur_ns))
                .unwrap_or(start),
            attributes,
            status: Status::Unset,
        });
    }

    /// The `job` span of a traced job, ended with its terminal `status`
    /// (`succeeded`, `failed`, `cancelled`).
    pub fn job_span(&self, j: JobSpan, status: &str) {
        if !sampled(self.ratio, j.trace.id) {
            return;
        }
        let end = SystemTime::now();
        let total = j.started.elapsed();
        let start = end.checked_sub(total).unwrap_or(end);
        let mut attributes = vec![
            KeyValue::new("fv.trace_id", j.trace.id.hex()),
            KeyValue::new("fv.job", j.job),
            KeyValue::new("fv.model", j.model),
            KeyValue::new("fv.api", j.api),
            KeyValue::new("fv.status", status.to_owned()),
            KeyValue::new("fv.duration_ms", ms(total)),
        ];
        if let Some(q) = j.queue_wait {
            attributes.push(KeyValue::new("fv.queue_wait_ms", ms(q)));
        }
        if let Some(i) = j.inference {
            attributes.push(KeyValue::new("fv.inference_ms", ms(i)));
        }
        let status = if status == "failed" {
            Status::error("failed")
        } else {
            Status::Unset
        };
        self.emit(Raw {
            trace: TraceId::from_bytes(j.trace.id.0),
            span: job_span_id(j.trace.id),
            parent: SpanId::INVALID,
            name: "job".to_owned(),
            start,
            end,
            attributes,
            status,
        });
    }

    /// Waits for the processor to export what it holds (tests, shutdown).
    pub fn force_flush(&self) -> bool {
        self.processor.force_flush().is_ok()
    }

    /// Flushes and stops the exporter.
    pub fn shutdown(&self) {
        let _ = self.provider.shutdown();
    }

    fn emit(&self, r: Raw) {
        self.processor.on_end(SpanData {
            span_context: SpanContext::new(
                r.trace,
                r.span,
                TraceFlags::SAMPLED,
                false,
                TraceState::default(),
            ),
            parent_span_id: r.parent,
            parent_span_is_remote: false,
            span_kind: SpanKind::Internal,
            name: r.name.into(),
            start_time: r.start,
            end_time: r.end,
            attributes: r.attributes,
            dropped_attributes_count: 0,
            events: SpanEvents::default(),
            links: SpanLinks::default(),
            status: r.status,
            instrumentation_scope: self.scope.clone(),
        });
    }
}

/// The `job` span of a traced job: the gate's pump opens it at submit,
/// notes the phases as the engine reports them and ends it at the terminal
/// event, or as `failed` when the engine drops the job without one
/// ([`JobSpan::finish`]).
#[derive(Debug)]
pub struct JobSpan {
    trace: Trace,
    job: String,
    model: String,
    api: &'static str,
    started: Instant,
    /// Submit → the engine's `Started`.
    pub queue_wait: Option<Duration>,
    /// `Started` → the engine's terminal event.
    pub inference: Option<Duration>,
}

impl JobSpan {
    /// Opens the span now, for `job` (its id) of `model` submitted on `api`.
    pub fn new(trace: Trace, job: String, model: String, api: &'static str) -> Self {
        Self {
            trace,
            job,
            model,
            api,
            started: Instant::now(),
            queue_wait: None,
            inference: None,
        }
    }

    /// Ends the span with the terminal `status`; nothing when no exporter
    /// is installed.
    pub fn finish(self, status: &str) {
        if let Some(x) = G.get() {
            x.job_span(self, status);
        }
    }
}

static G: OnceLock<Exporter> = OnceLock::new();

/// Builds the exporter for `cfg` and installs it as the process's; the
/// `tracing` layer to join in `init_tracing`. `Ok(None)` when `[otel]
/// enabled` is false: nothing is built and nothing exports.
pub fn init<S>(cfg: &OtelCfg) -> Result<Option<OtelLayer<S>>, OtelError>
where
    S: tracing::Subscriber + for<'span> LookupSpan<'span>,
{
    if !cfg.enabled {
        return Ok(None);
    }
    let x = Exporter::build(cfg)?;
    let layer = x.layer();
    G.set(x).map_err(|_| OtelError::AlreadyInitialised)?;
    Ok(Some(layer))
}

/// Whether [`init`] installed an exporter.
pub fn enabled() -> bool {
    G.get().is_some()
}

/// The recorder's sink: one event → one span (no-op without an exporter).
pub fn export_event(e: &Event) {
    if let Some(x) = G.get() {
        x.export_event(e);
    }
}

/// Flushes and stops the process's exporter (at exit): the recorder's drain
/// first (its events become spans there), then the batch processor, so the
/// last spans leave with the process. A process that traced nothing has no
/// drain to wait for.
pub fn shutdown() {
    if let Some(x) = G.get() {
        if fastvideo_trace::started() {
            fastvideo_trace::global().flush(Duration::from_secs(2));
        }
        x.shutdown();
    }
}

/// `Sampler::TraceIdRatioBased(ratio)`'s decision for `t`: the low 63 bits
/// of the trace id against `ratio * 2^63` (the SDK's rule, so the spans
/// built here and the `tracing` spans agree per trace).
pub fn sampled(ratio: f64, t: FvTraceId) -> bool {
    if ratio >= 1.0 {
        return true;
    }
    let upper = (ratio.max(0.0) * (1u64 << 63) as f64) as u64;
    (low_bits(t) >> 1) < upper
}

/// The `job` span's id in trace `t`: the recorder's events carry no span
/// ids, so the phases find their parent through the trace id alone.
pub fn job_span_id(t: FvTraceId) -> SpanId {
    const KEY: [u8; 8] = *b"fv.job\x5a\xa5";
    let mut b = low_bits(t).to_be_bytes();
    for (x, k) in b.iter_mut().zip(KEY) {
        *x ^= k;
    }
    if b == [0u8; 8] {
        b[7] = 1;
    }
    SpanId::from_bytes(b)
}

/// A job phase (a child of the `job` span) or a step of the request around
/// it (a root under the trace id).
fn is_job_phase(comp: &str, name: &str) -> bool {
    matches!(comp, "queue" | "engine" | "gpu" | "post" | "upload")
        || (comp == "store" && name.starts_with("terminal"))
}

fn low_bits(t: FvTraceId) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&t.0[8..16]);
    u64::from_be_bytes(b)
}

fn new_span_id() -> SpanId {
    SpanId::from_bytes(fastvideo_trace::new_span_id().to_be_bytes())
}

fn wall(ns: i64) -> SystemTime {
    UNIX_EPOCH + Duration::from_nanos(ns.max(0) as u64)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use opentelemetry_sdk::trace::{SamplingDecision, ShouldSample};
    use tracing_subscriber::prelude::*;

    use super::*;

    /// Keeps every exported span.
    #[derive(Debug, Clone, Default)]
    struct Keep(Arc<Mutex<Vec<SpanData>>>);

    impl Keep {
        fn spans(&self) -> Vec<SpanData> {
            self.0.lock().unwrap().clone()
        }
    }

    impl SpanExporter for Keep {
        fn export(
            &self,
            batch: Vec<SpanData>,
        ) -> impl std::future::Future<Output = OTelSdkResult> + Send {
            let keep = Arc::clone(&self.0);
            async move {
                keep.lock().unwrap().extend(batch);
                Ok(())
            }
        }
    }

    fn on() -> OtelCfg {
        OtelCfg {
            enabled: true,
            ..OtelCfg::default()
        }
    }

    fn event(trace: &Trace, comp: &str, name: &str, dur_ns: u64, arg: Option<i64>) -> Event {
        Event {
            trace: trace.id.hex(),
            host: "pod:test".into(),
            comp: comp.into(),
            name: name.into(),
            clock: if comp == "gpu" { "gpu" } else { "host" }.into(),
            t_wall_ns: 1_791_400_000_123_456_789,
            t_mono_ns: Some(42),
            dur_ns,
            arg,
            attrs: None,
        }
    }

    fn attr<'a>(s: &'a SpanData, key: &str) -> Option<&'a opentelemetry::Value> {
        s.attributes
            .iter()
            .find(|kv| kv.key.as_str() == key)
            .map(|kv| &kv.value)
    }

    #[test]
    fn the_otlp_exporter_and_the_layer_build_without_a_collector() {
        // Nothing listens on the default endpoint; building connects nowhere.
        let x = Exporter::build(&on()).expect("exporter");
        let layer = x.layer();
        let _subscriber = tracing_subscriber::registry().with(layer);
        // Nothing queued: the flush and the shutdown send nothing.
        assert!(x.force_flush());
        x.shutdown();
        // A bad endpoint is refused before anything is built.
        let bad = OtelCfg {
            endpoint: "not a url".into(),
            ..on()
        };
        assert!(matches!(Exporter::build(&bad), Err(OtelError::Config(_))));
    }

    #[test]
    fn recorder_events_and_the_job_span_make_one_trace() {
        let keep = Keep::default();
        let x = Exporter::with_exporter(&on(), keep.clone());
        let t = Trace::new_root();
        x.export_event(&event(&t, "http", "post", 2_000_000, Some(200)));
        x.export_event(&event(&t, "engine", "run", 5_000_000, None));
        x.export_event(&event(&t, "gpu", "denoise.step", 180_000, Some(3)));
        x.export_event(&event(&t, "store", "terminal", 1_000, None));
        x.export_event(&event(&t, "store", "lookup", 1_000, Some(2)));
        let mut j = JobSpan::new(t, "job_1".into(), "fv-fake".into(), "fal");
        j.queue_wait = Some(Duration::from_millis(7));
        j.inference = Some(Duration::from_millis(50));
        x.job_span(j, "succeeded");
        assert!(x.force_flush());
        let spans = keep.spans();
        assert_eq!(spans.len(), 6, "{spans:#?}");
        let want = TraceId::from_bytes(t.id.0);
        assert!(spans.iter().all(|s| s.span_context.trace_id() == want));
        assert!(spans.iter().all(|s| s.span_context.is_sampled()));
        let by_name = |n: &str| spans.iter().find(|s| s.name == n).expect(n);
        let job = by_name("job");
        assert_eq!(job.span_context.span_id(), job_span_id(t.id));
        assert_eq!(job.parent_span_id, SpanId::INVALID);
        assert_eq!(
            attr(job, "fv.job").map(|v| v.to_string()),
            Some("job_1".into())
        );
        assert_eq!(
            attr(job, "fv.model").map(|v| v.to_string()),
            Some("fv-fake".into())
        );
        assert_eq!(
            attr(job, "fv.api").map(|v| v.to_string()),
            Some("fal".into())
        );
        assert_eq!(
            attr(job, "fv.status").map(|v| v.to_string()),
            Some("succeeded".into())
        );
        assert_eq!(
            attr(job, "fv.queue_wait_ms").map(|v| v.to_string()),
            Some("7".into())
        );
        assert_eq!(
            attr(job, "fv.inference_ms").map(|v| v.to_string()),
            Some("50".into())
        );
        assert!(attr(job, "fv.duration_ms").is_some());
        // The phases hang off the job; the request's own steps are roots.
        for n in ["engine.run", "gpu.denoise.step", "store.terminal"] {
            assert_eq!(by_name(n).parent_span_id, job_span_id(t.id), "{n}");
        }
        for n in ["http.post", "store.lookup"] {
            assert_eq!(by_name(n).parent_span_id, SpanId::INVALID, "{n}");
        }
        let step = by_name("gpu.denoise.step");
        assert_eq!(
            attr(step, "fv.comp").map(|v| v.to_string()),
            Some("gpu".into())
        );
        assert_eq!(
            attr(step, "fv.stage").map(|v| v.to_string()),
            Some("denoise.step".into())
        );
        assert_eq!(
            attr(step, "fv.clock").map(|v| v.to_string()),
            Some("gpu".into())
        );
        assert_eq!(
            attr(step, "fv.arg").map(|v| v.to_string()),
            Some("3".into())
        );
        assert_eq!(
            attr(step, "fv.host").map(|v| v.to_string()),
            Some("pod:test".into())
        );
        assert_eq!(
            step.end_time.duration_since(step.start_time).unwrap(),
            Duration::from_nanos(180_000)
        );
        assert_eq!(
            step.start_time.duration_since(UNIX_EPOCH).unwrap(),
            Duration::from_nanos(1_791_400_000_123_456_789)
        );
        let post = by_name("http.post");
        assert_eq!(
            attr(post, "fv.arg").map(|v| v.to_string()),
            Some("200".into())
        );
        assert!(attr(post, "fv.t_mono_ns").is_some());
        // A failed job carries an error status.
        let j = JobSpan::new(t, "job_2".into(), "fv-fake".into(), "native");
        x.job_span(j, "failed");
        assert!(x.force_flush());
        let failed = keep
            .spans()
            .into_iter()
            .find(|s| attr(s, "fv.job").map(|v| v.to_string()) == Some("job_2".into()))
            .expect("job_2");
        assert!(matches!(failed.status, Status::Error { .. }));
        x.shutdown();
    }

    #[test]
    fn sampling_follows_the_sdk_ratio_rule() {
        for ratio in [0.0, 0.1, 0.25, 0.5, 0.9, 1.0] {
            let sdk = Sampler::TraceIdRatioBased(ratio);
            for _ in 0..200 {
                let t = FvTraceId::random();
                let want = sdk
                    .should_sample(
                        None,
                        TraceId::from_bytes(t.0),
                        "x",
                        &SpanKind::Internal,
                        &[],
                        &[],
                    )
                    .decision
                    == SamplingDecision::RecordAndSample;
                assert_eq!(sampled(ratio, t), want, "ratio {ratio} trace {t}");
            }
        }
        // Ratio 0: nothing leaves.
        let keep = Keep::default();
        let x = Exporter::with_exporter(
            &OtelCfg {
                sample_ratio: 0.0,
                ..on()
            },
            keep.clone(),
        );
        let t = Trace::new_root();
        x.export_event(&event(&t, "engine", "run", 1, None));
        x.job_span(JobSpan::new(t, "j".into(), "m".into(), "fal"), "succeeded");
        x.force_flush();
        assert!(keep.spans().is_empty());
        x.shutdown();
    }

    #[test]
    fn job_span_ids_are_stable_valid_and_distinct() {
        let a = FvTraceId::random();
        let b = FvTraceId::random();
        assert_eq!(job_span_id(a), job_span_id(a));
        assert_ne!(job_span_id(a), job_span_id(b));
        assert_ne!(job_span_id(a), SpanId::INVALID);
        let zero_low = FvTraceId(*b"\x01\x02\x03\x04\x05\x06\x07\x08fv.job\x5a\xa5");
        assert_ne!(job_span_id(zero_low), SpanId::INVALID, "the XOR hits zero");
        assert!(is_job_phase("queue", "wait"));
        assert!(is_job_phase("store", "terminal_write"));
        assert!(!is_job_phase("store", "insert"));
        assert!(!is_job_phase("http", "post"));
        assert!(!is_job_phase("adapter", "parse"));
    }

    #[test]
    fn without_an_installed_exporter_nothing_happens() {
        // The global is never set in this test binary; the hooks are no-ops.
        assert!(!enabled());
        let t = Trace::new_root();
        export_event(&event(&t, "engine", "run", 1, None));
        JobSpan::new(t, "j".into(), "m".into(), "fal").finish("succeeded");
        shutdown();
        let off: Result<Option<OtelLayer<tracing_subscriber::Registry>>, _> =
            init(&OtelCfg::default());
        assert!(matches!(off, Ok(None)));
        assert!(!enabled());
    }
}
