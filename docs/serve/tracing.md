# End-to-end request tracing

Status: **built 2026-10-07** (PR #59). Results of the warm-request bench
and the tracing-overhead A/B: [bench/e2e-warm-trace.md](bench/e2e-warm-trace.md).

One trace id per request, from the click on the console's **Run** button
(or the bench client) to the moment the video is playable, with a
timestamped event at every step on every host, recorded without
perturbing the request it measures.

```
browser ──► edge Worker ──► pod fv-serve (front) ──► family DO ──► pod (executor) ──► engine ──► GPU
 click        recv, auth,      http.post, adapter.*,    (enqueue,     worker.offer,      queue.wait,   gpu.<stage>,
 submit       registry,        store.insert,             offer)        worker.take        engine.*      gpu.denoise.step[k]
 poll…        body_read,       front.do_enqueue                                           post.*, upload.*, store.terminal
 result       forward ◄─sync─► x-fv-trace-t                                                  ▲
 video bytes  request          http.get (polls), store.lookup ───────────────────────────────┘
 canplay
```

## How to enable

Tracing is **opt-in per request** and off for everything else.

| where | how |
|---|---|
| a request | header `x-fv-trace: 1` (or query `?fv_trace=1`); `traceparent` (W3C) supplies the trace id, else the first hop makes one. A bare `traceparent` does **not** turn tracing on (instrumented SDKs send one on every call). |
| the pod | `FV_TRACE` = `opt-in` (default: honour the header), `off` (ignore it; `/fv/v1/traces/*` answer 404), `all` (trace every request; for experiments only) |
| the edge Worker | var `FV_TRACE` (`off` disables; default honours the header). The edge passes `x-fv-trace` on to the front (every other client `x-fv-*` header is still dropped). |
| the console | the **Trace** switch next to **Run** on a model page (remembered in `localStorage`, or `?trace=1`); the waterfall appears in the result's **Trace** tab once the video can play. |
| the bench | `node scripts/serve/trace-bench.mjs --base URL --endpoint <fal endpoint> …` (below) |

Other pod settings: `FV_TRACE_BUFFER` (channel capacity, records; default
65 536), `FV_TRACE_FILE` (append every event as a JSON line),
`FV_TRACE_LOG=0` (do not ship events through the log shipper; on by
default, so with `FV_LOG_SHIP_URL` set every event lands in fv-control's
log store, target `fv_trace`, field `trace_id`).

Reading a trace: `GET /fv/v1/traces/{id}` on any front (the edge routes it
like a native list call) answers every event this process recorded under
the id plus what the edge and the browser posted, and the recorder's
counters (`stats.sent`, `stats.dropped`, `truncated`).
`node scripts/serve/trace-report.mjs --base URL <id>` prints the waterfall.

## Non-perturbing by construction

| host | hot path | off the hot path |
|---|---|---|
| pod (Rust, `crates/fastvideo-trace`) | `Instant::now()` and a `try_send` of an 80-byte `Copy` record (static `&str` names, no allocation, no formatting, no lock, no I/O) into a bounded `std::sync::mpsc::sync_channel`. Full: the record is dropped and `stats.dropped` counts it; nothing waits. | one drain thread (`fv-trace-drain`, started by the first traced request: an untraced process never starts it): monotonic → wall time, the per-trace store (last 512 traces, 8 192 events each), the JSON-lines file, the log shipper |
| GPU | at every stage boundary and finished denoise step the engine enqueues `cuEventRecord` of a **preallocated** timing event on the compute stream (256 per traced job, created before the run). No `cudaDeviceSynchronize`, no stream sync, no host wait is added. | after the run (the host already waited for the last frames, so every event is complete) the marks are handed to the drain thread (`Recorder::defer`), which reads event-to-event times and frees the events |
| edge Worker | pushes small JSON objects into a `Vec` in the request's memory | adds `x-fv-edge-t` / `server-timing` to the answer, then ships the events to the front in `ctx.waitUntil`, after the response is sent |
| browser | `performance.now()` reads into an array; the Resource Timing entry of the video | after `canplay`, in `requestIdleCallback`: `navigator.sendBeacon` to `/fv/v1/traces/{id}/events`, then the waterfall |

**Zero cost when off:** an untraced request costs one header lookup in the
HTTP layer (`mode()` is an atomic load); every call site below it is
`if let Some(trace) = …` on a `None`; the engine creates no events. Tests:
`crates/fastvideo-trace/tests/off.rs` (10 M untraced call sites never
start the recorder or its thread), `recorder.rs`
(`a_full_channel_drops_and_counts_without_blocking`: 100 000 pushes into
a full 8-slot channel, 99 992 counted drops, no wait).

## Event format

One JSON object per event (the store, `FV_TRACE_FILE`, the beacon body):

```json
{"trace":"4bf92f3577b34da6a3ce929d0e0e4736","host":"pod:abc123","comp":"gpu","name":"denoise.step",
 "clock":"gpu","t_wall_ns":1791400000123456789,"t_mono_ns":81234567890,"dur_ns":182345678,"arg":3}
```

`host` is whose clock `t_wall_ns` is on (`client`, `edge`, `pod:<id>`);
`clock` is `host` or `gpu` (device time placed on the pod's clock, below);
`dur_ns` 0 is a point; `arg` a step number, an HTTP status, a byte count or
a job status; `attrs` carries clock-sync samples and details.

## Event list

| comp.name | host | what |
|---|---|---|
| `client.click` | browser | the Run click (origin of the waterfall) |
| `client.submit` (+`.body`) | browser | submit request: fetch start → answer headers (→ body read); `attrs.sync`: pod and edge clock samples |
| `edge.auth` | edge | API-key check (D1 / isolate cache) |
| `edge.registry` | edge | registry snapshot (Registry DO, cached 2 s) |
| `edge.body_read` | edge | reading the request body (model scan) |
| `edge.forward` | edge | the forward to the front until its answer headers; `attrs.sync`: edge↔pod sample |
| `edge.request` | edge | the edge's whole handling (receive → response headers) |
| `http.post` / `http.get` / `http.get_file` / `http.get_content` | pod | one per traced request the pod answers: receive → response headers; `arg` = status |
| `adapter.parse` | pod | auth + JSON parse + protocol normalize (`arg` = body bytes) |
| `adapter.validate` | pod | admission, safety, model resolution, precheck |
| `adapter.ingest_negotiate` | pod | input staging + negotiation |
| `store.insert` | pod | job record insert (memory / D1) |
| `queue.submit` | pod | the engine gate's submit (local queue, or the front's enqueue) |
| `front.stage_inputs`, `front.do_enqueue` | pod (front) | envelope inputs; the HTTP round trip to the family Durable Object |
| `worker.offer`, `worker.take` | pod (executor) | the DO's offer arriving on the worker socket; adopting the envelope |
| `store.get` | pod | re-read after submit |
| `queue.wait` | pod | engine queue: submit → executor dequeue (`arg` = executor) |
| `engine.text_encode` / `engine.text`, `engine.denoise.step` [k], `engine.denoise.tail`, `engine.audio_decode`, `engine.video_decode`, `engine.encode_tail` | pod | host-side stage segments (from stage boundary / step marks) |
| `gpu.*` (same names) | pod GPU | the same segments in **device time** from CUDA events |
| `engine.run` | pod | executor run (dequeue → output) |
| `post.encode_busy` | pod | NVENC feed busy time (overlaps the decode; placed ending at the decode's end) |
| `engine.finished` | pod | the engine's result reached the async side |
| `post.finalize` | pod | MP4 finalize / remux |
| `upload.artifact_put` | pod | output into the artifact store (R2 upload; edge mode: the direct upload's tail + commit) |
| `store.terminal_write` | pod | the terminal job write (D1 / memory) |
| `store.terminal` | pod | upload + terminal write (from here the job reads as done) |
| `store.lookup` | pod | each status / result lookup; `arg` = job status (0 queued, 1 running, 2 succeeded) |
| `client.poll` (+`.body`) | browser | each status poll; `attrs.job_status` |
| `client.result` | browser | the result request |
| `client.video_src`, `video_loadstart`, `video_metadata` | browser | player events |
| `client.video_fetch`, `video_ttfb`, `video_bytes` | browser | Resource Timing of the video (TTFB needs `Timing-Allow-Origin` on cross-origin storage, else only start/end) |
| `client.canplay` | browser | the video can play (end of the waterfall) |
| `client.video_error` | browser | the player cannot play the file (e.g. no H.264 in a bare Chromium): ends the waterfall instead, the trace still ships |
| `client.e2e` | browser | click → canplay |
| `engine.timeline.overflow` | pod | marks beyond the 256 preallocated (count) |

## GPU timing

`StepControl` (engine-service) owns a `Timeline` for a traced job; the
CUDA backend's `marks()` creates `EventMarks` (fastvideo-cudarc
`timing.rs`) on the global device's compute stream. Every
`ctl.stage(name)` / `ctl.step(k, n)` (already called by the pipelines'
progress hooks) records the next event; `deliver` adds `encode_tail` when
the pipeline returns. Segment *i* is mark *i* → mark *i+1*.

Device times are relative. They are placed on the pod's clock at
`h0 + lag + g_i` with `lag = max(0, max_i(h_i − h0 − g_i))` (`h_i`: host
enqueue time of mark *i*, `g_i`: device time since mark 0): a mark cannot
complete before the host enqueued it, and the end mark is enqueued after
the host waited for the device, which pins the placement to within that
wait. Host spans of the same segments are recorded too, so a step whose
host and device durations differ shows where the host ran ahead or
waited.

The fake engine's marks stand in for CUDA events (its clock), which is how
`crates/fastvideo-engine-service/tests/trace.rs` checks the device spans
(one per denoise step, the scripted step time) and that they are resolved
on the drain thread.

## Clock alignment

Every hop's durations come from its own monotonic clock (pod: `Instant`;
browser: `performance.now()`; edge: `Date.now()`, which on Workers only
advances across I/O, so edge spans measure the I/O they wait on with 1 ms
resolution and CPU time inside the isolate reads as 0).

Across hosts, each traced HTTP exchange gives an NTP sample: the client
sends at `t0` and reads the answer headers at `t3` (its clock); the pod
stamps `x-fv-trace-t: <t1>;<t2>` (receive, send; its wall clock, ns); the
edge stamps `x-fv-edge-t` (ms) and records the same sample for its forward
to the pod. `offset = ((t1 − t0) + (t2 − t3)) / 2`,
`delay = (t3 − t0) − (t2 − t1)`; per host pair the sample with the least
delay wins and the offset is known to **± delay / 2** (the true offset is
inside that bound whatever the path asymmetry). The waterfall uses the
client's clock (the pod's when there is no client), shows each host's
offset and bound, and cross-host gaps finer than the bound are not
meaningful: the report prints the bound next to each host. Over the
Runpod proxy the bound is typically a few ms (the proxy adds latency to
both directions); on the same host it is sub-ms. Durations measured on one
host (every span) carry no alignment error.

## Report

`analyze(events)` (`crates/fastvideo-serve/console/trace.js`, used by the
console, `trace-report.mjs` and `trace-bench.mjs`):

- **waterfall**: every event, start (ms from the click) and duration, host
  and clock;
- **critical path**: consecutive phases between anchors — click → submit
  sent → pod received → pod answered → engine dequeued → GPU first mark →
  GPU last mark → engine run end → job terminal → client saw COMPLETED →
  result answer → video first byte → last byte → playable;
- **unaccounted**: the parts of [click, playable] covered by no step
  (envelopes such as `engine.run`, `store.terminal`, `edge.request`,
  `client.e2e` excluded), with the steps on either side of each gap. The
  largest one in a polling client is the wait between the job turning
  terminal and the next poll.

## Bench

```bash
node scripts/serve/trace-bench.mjs --base https://<pod>-8000.proxy.runpod.net \
  --endpoint minimax/h3-turbo/text-to-video --key-file ~/.config/fv/bench-key \
  --input '{"resolution":"480P","duration":5}' --warmup 2 --n 20 --ab --label direct-480
```

`--ab` alternates traced and untraced runs (same prompt, same drift);
`--fresh-prompt` makes every prompt new (text encoder run vs the prompt
cache); the summary has per-metric median / p90 for both, the median
difference and its 95 % bootstrap interval (the noise estimate), the
per-step and per-phase median / p90 of the traced runs, the unaccounted
time and the largest `dropped` count seen.

The console's own click: `scripts/serve/trace-browser.mjs --base URL --page
<fal endpoint> --key-file F --set resolution=480P --n 3` (headless
Chromium with Playwright: ticks Trace, clicks Run, waits for the Trace
tab). `FV_BROWSER_RELAY=1` relays the page's requests through Node and
`FV_BROWSER_LOCAL_CONSOLE=1` serves this checkout's console scripts (both
for sandboxes; see the bench doc).

## Results

[bench/e2e-warm-trace.md](bench/e2e-warm-trace.md): H3 turbo 480P / 768P on
an RTX PRO 6000, pod direct. Overhead A/B: every traced − untraced median
difference is inside its 95 % bootstrap interval (denoise +6.5 ms of 7.1 s
at 480P, +14.5 ms of 19.2 s at 768P); 0 events dropped.

## OpenTelemetry export

Status: **built 2026-10-24** (F3c of `docs/plan/2026-10-23-execution-plan.md`),
validated on CPU with an in-memory exporter and the OTLP exporter built
against no collector; **verification against a live OTLP collector is
pending** (a GPU pod with a collector sidecar, or any `otel/opentelemetry-collector`
with the `otlp` receiver's HTTP port 4318).

The recorder above is the source of truth; this sends the same events to
an OTLP collector so they can be looked at next to other services' traces.
It is compiled in by the cargo feature `otel` of `fastvideo-serve`
(`cargo build -p fastvideo-serve --features otel,…`; the default image does
not carry it) and turned on by the `[otel]` section (`src/config.rs`); a
build without the feature accepts the section, warns when `enabled` and
exports nothing. Off by default either way; nothing changes without both.

```toml
[otel]
enabled = true                                   # FV_OTEL=1
endpoint = "http://127.0.0.1:4318/v1/traces"     # FV_OTEL_ENDPOINT: the full traces URL, used as given
protocol = "http/protobuf"                       # FV_OTEL_PROTOCOL: the only protocol built (no gRPC/tonic)
service_name = "fv-serve"                        # FV_OTEL_SERVICE_NAME: resource service.name
sample_ratio = 1.0                               # FV_OTEL_SAMPLE_RATIO: head sampling by trace id, 0..=1
timeout_ms = 10000                               # FV_OTEL_TIMEOUT_MS: one export request
```

What a collector shows for a traced job (`crates/fastvideo-serve/src/otel.rs`):

- one trace per request, under the request's W3C trace id (the same id as
  `GET /fv/v1/traces/{id}`, `traceparent` and the console's waterfall);
- a `job` span per job, opened at submit and ended at the terminal event,
  with the attributes `fv.job` (job id), `fv.model`, `fv.api`, `fv.status`
  (`succeeded` / `failed` / `cancelled`; `failed` sets the span's error
  status), `fv.queue_wait_ms` (submit → `Started`), `fv.inference_ms`
  (`Started` → the engine's terminal event, an on-demand load included, as
  `fv_job_inference_seconds`) and `fv.duration_ms`;
- every event of the [event list](#event-list) as a finished span named
  `<comp>.<name>` (`queue.wait`, `engine.run`, `gpu.denoise.step`,
  `post.finalize`, `upload.artifact_put`, `store.terminal`, …) with its
  start and duration as recorded and the attributes `fv.trace_id`,
  `fv.comp`, `fv.stage` (the event name), `fv.clock` (`host` / `gpu`),
  `fv.host`, `fv.dur_ms`, `fv.arg`, `fv.t_mono_ns` and `fv.attrs` (JSON):
  the same fields the log shipper's `fv_trace` lines carry. The job's
  phases (`queue`, `engine`, `gpu`, `post`, `upload`, `store.terminal*`)
  are children of the `job` span; the request's own steps (`http.*`,
  `adapter.*`, `store.insert` / `get` / `lookup`, `front.*`, `worker.*`)
  are roots in the same trace, since the recorder keeps no span ids
  (the `job` span's id is derived from the trace id).

Sampling is the SDK's `TraceIdRatioBased` rule applied per trace, so a
trace is exported whole or not at all; the events are converted on the
recorder's drain thread and sent by the SDK's batch processor on its own
thread (OTLP over HTTP with the blocking reqwest client), so a traced
request's path is unchanged: the hot path is still the recorder's
`try_send`. The process's own `tracing` spans go through a
`tracing-opentelemetry` layer in `main.rs`'s `init_tracing` as well (at the
default `info` filter there are none); they carry their own trace ids.

Edge and browser events posted to `/fv/v1/traces/{id}/events` are also
exported (they pass through the same drain), on their own hosts' clocks
(see [Clock alignment](#clock-alignment)). Crates (versions from
`Cargo.lock`): `opentelemetry` / `opentelemetry_sdk` / `opentelemetry-otlp`
0.33.1 (`trace`, `http-proto`, `reqwest-blocking-client`: the workspace's
reqwest 0.13, prost 0.14 and http 1 serve them, no second copy),
`tracing-opentelemetry` 0.34.0.

## Prometheus metrics

Traces answer "what happened to this request"; the series below answer
"how is the server doing", rendered by `GET /metrics` of every `fv-serve`
(`crates/fastvideo-serve/src/metrics.rs`). Durations are histograms with
the buckets 5 ms … 600 s. `model` is always the job's model id, one of
the configured `[[models]]` (the fake engine's ids in CPU tests), so the
label set stays bounded; `api` is the protocol the job came in on; `route`
is the matched template, never the raw path.

| Series | Labels | What |
|---|---|---|
| `fv_http_requests_total` | `method`, `route`, `status` | every request |
| `fv_http_request_duration_seconds` | `method`, `route` | request wall time |
| `fv_jobs_submitted_total` | `api` | jobs the engine accepted |
| `fv_jobs_finished_total` | `api`, `status` | terminal jobs: `succeeded`, `failed`, `cancelled` |
| `fv_job_duration_seconds` | `api`, `model` | submit to the terminal store write: queue wait, generation, post-processing and upload |
| `fv_job_queue_wait_seconds` | `model` | submit to the engine's `Started` (the scheduler's queue, including a swap-in wait behind the running job) |
| `fv_job_inference_seconds` | `model` | the engine's `Started` to its terminal event: an on-demand load (`[engine] swap`) and the generation itself |
| `fv_engine_queued` | `priority` | queued jobs (`batch`, `stream`), on scrape |
| `fv_engine_running`, `fv_engine_sessions` | | jobs running, streaming sessions open, on scrape |
| `fv_engine_resident_models` | | (executor, model) pairs resident now, on scrape |
| `fv_engine_model_loads_total` | | successful model loads since start (boot and on demand) |
| `fv_engine_model_swaps_total` | | on-demand loads in swap mode, each evicting the executor's other resident models |
| `fv_engine_model_load_seconds` | `model` | wall time of the model's last successful load |
| `fv_encoder_restart_duration_seconds`, `fv_encoder_restarts_total` | `codec`, `spare` | keyframe restarts of the pipe encoders |
| `fv_ready` | | 1 once every resident model loaded |

A job that outruns `[engine] job_deadline_s` (`FV_JOB_DEADLINE_S`; 0, the
default, is no limit) is cancelled by the engine at its next denoise step
and counts as `failed` with the error kind `timeout` (HTTP 504 on a sync
route); the deadline covers the generation only, from the start of
`generate` after any on-demand load, so a swap-in's cost does not eat
into it. Streaming sessions have their own limits (`[streams]`).
