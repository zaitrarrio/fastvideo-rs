# Deployment research: Runpod pods, Runpod Serverless, Vast.ai

Status: research, 2026-09-27. Nothing here is implemented yet.

Scope: how the planned Rust (Axum/Tokio) inference server should be deployed. It has batch HTTP APIs and WebRTC streaming, and runs on fastvideo-rs CUDA. There are three targets: (a) Runpod pods, (b) Runpod Serverless, (c) Vast.ai (instances and Vast Serverless).

Conventions:
- Every fact carries a source: a URL, a pinned commit of a source repo, or a path in this repo.
- **INFERRED** marks my own reasoning, which no source states.
- Pinned sources read for this document:
  - `runpod/runpod-python` @ `760aea2cc6c0a8f5739b2a8667f61ae11d49d37b` (2026-09-24). Links below abbreviate it as `rp@760aea2`, for example `rp@760aea2:runpod/serverless/modules/rp_job.py#L24` = https://github.com/runpod/runpod-python/blob/760aea2cc6c0a8f5739b2a8667f61ae11d49d37b/runpod/serverless/modules/rp_job.py#L24
  - `vast-ai/pyworker` @ `60cfeca889f979fd73cf9e00adcd7b0ebc016fdc` (2026-09-16), abbreviated `pw@60cfeca`.
  - `vast-ai/vast-cli` @ `92caa2dd203ebd89da8df99a8c7123130a90a2ba` (2026-09-25), abbreviated `vc@92caa2d`. It contains the PyWorker runtime `vastai/serverless/server/`.
  - `runpod-workers/worker-websocket` @ `67ca740`, `runpod-workers/worker-lb-websocket` @ `9f2f34f`.
  - Runpod docs were fetched as Markdown (`https://docs.runpod.io/<page>.md`) on 2026-09-27. Vast docs likewise (`https://docs.vast.ai/<page>.md`).

---

## 0. Summary and recommendation

| Target | HTTP batch API | WebRTC (inbound ICE to GPU box) | WebRTC via outbound WHIP to an SFU | Recommended mode |
|---|---|---|---|---|
| Runpod pod | Yes: `https://<pod>-<port>.proxy.runpod.net` (100 s Cloudflare cap), or a public TCP port | **No UDP** ("Pods do not support UDP connections"). ICE-TCP to a mapped public TCP port, or a TURN relay, only. | Yes, needs no inbound port. Outbound UDP is undocumented; strobe's evidence covers serverless only (§5.1). | Long-lived pods for the streaming service; the proxy URL for control and batch |
| Runpod Serverless, queue endpoint | Via `/run` `/runsync` `/status` `/stream` (job JSON, 10/20 MB caps). Needs a worker speaking the job-take protocol (§1.1). | No UDP. "Expose HTTP/TCP ports" gives public IP + TCP only. | Yes: a job = one stream session, `executionTimeout` ≥ stream length (§6) | Batch generation jobs; WHIP-out streaming sessions |
| Runpod Serverless, load balancer endpoint | **Yes, our Axum server runs unchanged**: `https://<endpoint>.api.runpod.ai/<path>`, `/ping` health, WebSockets supported. Limits: 5.5 min per request, 30 MB. | No (HTTP/WS only) | Signaling over HTTP/WS works; media must go outbound | Short synchronous APIs, WebRTC signaling |
| Vast instance | Public IP:random port (`VAST_TCP_PORT_<n>`) | **Yes**: `-p N:N/udp` and `VAST_UDP_PORT_<n>` (random external port, NAT on a shared IP) | Yes | Streaming service with direct ICE, cheap batch |
| Vast Serverless | Client → `route` → direct call to worker `https://PUBLIC_IPADDR:VAST_TCP_PORT_<WORKER_PORT>` with signed `auth_data`. The worker must implement the PyWorker protocol (§3.4). | Workers are ordinary Vast instances, so UDP ports are possible (INFERRED) | Yes | Autoscaled batch; later |

Main conclusions:

1. **Inbound UDP on Runpod is unavailable** on pods and on serverless. Runpod WebRTC must therefore use one of:
   - (i) outbound WHIP to an SFU (Cloudflare Stream or Realtime, or MediaMTX elsewhere);
   - (ii) a TURN relay reachable over TCP/TLS;
   - (iii) ICE-TCP on a public TCP port.

   Option (i) is the portable design: the same image works on all three targets.
2. **Runpod load-balancer endpoints** let the Axum server run as-is. Requirements: listen on `$PORT` (default 80), and answer `GET /ping` with 204 while loading and 200 when ready. They cannot host long (>5.5 min) HTTP requests, and they cannot carry media.
3. **The queue protocol is small** (§1.1): long-poll GET job-take, POST job-done, POST stream, GET ping every 10 s, plus a long-poll job-stop channel. A native Rust worker loop (`reqwest` + Tokio) is roughly 300 lines (INFERRED). It avoids shipping Python in the runtime image, which has no Python today (`docker/gpucheck.Dockerfile`: "No Python: weights come through hf-fm").
4. **Vast Serverless** is not a job queue. The engine hands clients a worker URL plus a signature, and the worker must:
   - verify an RSA signature against `REPORT_ADDR/pubkey/`;
   - POST `/worker_status/` metrics.

   This is reimplementable in Rust (§3.4), but the officially supported path is the Python `vastai` SDK's `Worker`.

---

## 1. Runpod Serverless

### 1.1 Queue-endpoint worker protocol (what runpod-python does, so Rust can do the same)

**Environment injected into a serverless worker** (names come from the SDK source; the values' shapes come from tests and ARCHITECTURE.md):

| Env var | Use | Source |
|---|---|---|
| `RUNPOD_WEBHOOK_GET_JOB` | Job-take URL. `$ID` is replaced by the worker id. Already carries a query string, since the SDK appends `&...` | `rp@760aea2:runpod/serverless/modules/rp_job.py#L24`, `#L116-L123` |
| `RUNPOD_WEBHOOK_POST_OUTPUT` | Job-done URL. `$RUNPOD_POD_ID` → worker id; `$ID` → job id | `rp_http.py#L16-L19`, `#L63` |
| `RUNPOD_WEBHOOK_POST_STREAM` | Stream-chunk URL, same templating | `rp_http.py#L21-L24` |
| `RUNPOD_WEBHOOK_PING` | Heartbeat URL. `$RUNPOD_POD_ID` → worker id | `rp_ping.py#L29-L30` (file lines 127-128 in the concatenated view) |
| `RUNPOD_PING_INTERVAL` | ms, default 10000 → 10 s | `rp_ping.py`: `int(os.environ.get("RUNPOD_PING_INTERVAL", 10000)) // 1000` |
| `RUNPOD_AI_API_KEY` | Sent as the raw `Authorization:` header value (no `Bearer`) on every call | `rp@760aea2:runpod/http_client.py#L18-L33`, `rp_ping.py` (`{"Authorization": os.environ.get("RUNPOD_AI_API_KEY", "")}`) |
| `RUNPOD_POD_ID` | Worker id. The SDK falls back to a uuid4 when absent | `worker_state.py#L18` |
| `RUNPOD_POD_HOSTNAME`, `RUNPOD_ENDPOINT_ID`, `RUNPOD_DEBUG_LEVEL` | Error reports and logging | `rp_job.py#L295`, `_logger.py` |
| `RUNPOD_REALTIME_PORT`, `RUNPOD_REALTIME_CONCURRENCY` | "Realtime" mode: the SDK serves FastAPI on that port instead of polling | `rp@760aea2:runpod/serverless/__init__.py` (`_get_realtime_port`) |

Local-vs-production switch: the SDK treats the process as local when `RUNPOD_WEBHOOK_GET_JOB` is unset (`worker_state.py#L24`, `worker.py#L18-L26`).

URL shapes, from the SDK's local simulator:
- `GET /v2/{endpoint_id}/job-take/{worker_id}`
- `GET /v2/{endpoint_id}/job-take-batch/{worker_id}?batch_size=N`
- `POST /v2/{endpoint_id}/job-done/{worker_id}`
- `GET /v2/{endpoint_id}/ping/{worker_id}`

Source: `rp@760aea2:tests/test_serverless/local_sim/localhost.py`. Stream: `.../stream/$ID`, per `rp@760aea2:ARCHITECTURE.md` L1258. Production host: `api.runpod.ai/v2/...` (tests reference `https://api.runpod.ai/job-take`; `rp_prestart` test L266). **INFERRED**: treat the env values as opaque templates and never construct them.

**Job take (long poll):**
- `GET {JOB_GET_URL}&job_in_progress={0|1}` (`rp_job.py#L105-L126`).
- Batch form: replace `/job-take/` with `/job-take-batch/` and append `&batch_size=N` (`#L116-L118`).
- Responses (`rp_job.py#L143-L193`):
  - `204` = no job.
  - `400` = "expected when FlashBoot is enabled", treat as no job.
  - `429` = back off 5 s (`rp_scale.py#L344-L348`).
  - `200 application/json` = single job `{"id":..., "input":..., ...}`. The SDK requires `id` and `input`; extra fields such as `webhook` and `batchId` may be present (`worker_state.py#L38-L50`, `rp_scale.py#L484`). The batch endpoint returns a list of jobs.
- Client-side timeout per poll: 90 s (`rp_scale.py#L81-L82`). HTTP session total timeout: 600 s (`http_client.py#L40-L46`).

**Job done:**
- `POST {JOB_DONE_URL with $ID→job_id}&isStream={true|false}`, with headers `Content-Type: application/x-www-form-urlencoded`, `charset: utf-8`, `X-Request-ID: <job id>`.
- The body is the JSON text of the result, despite the form content type (`rp_http.py#L29-L65`).
- Retried with Fibonacci backoff, 3 attempts (`#L33`).
- Body shapes:
  - Success: `{"output": <any>}`.
  - Failure: `{"error": "<string>"}`. The SDK puts a JSON-encoded `{error_type, error_message, error_traceback, hostname, worker_id, runpod_version, logs?}` into that string (`rp_job.py#L289-L304`).
  - `{"stopPod": true}` is added for refresh_worker (`#L222-L225`, `#L275-L276`).
  - A handler dict containing `error` produces `{"output":{...rest}, "error": msg}` (`#L268-L276`).
- Size: the SDK warns above 20 MB (`rp_tips.py`). The public caps are `/run` 10 MB and `/runsync` 20 MB (https://docs.runpod.io/serverless/endpoints/operation-reference). Large video must go to object storage, with URLs returned.

**Streaming outputs:**
- Each generator yield becomes `POST {JOB_STREAM_URL}&isStream=false` with body `{"output": <chunk>}` (`rp_job.py#L197-L217`, `rp_http.py#L92-L98`, where `stream_result` doesn't pass is_stream).
- The final job-done carries `isStream=true` (`rp_job.py#L240`).
- With `return_aggregate_stream`, the final body is `{"output": [all chunks]}`; otherwise `{"output": []}` (`rp_job.py#L202`, `#L214-L215`).
- An error chunk ends the stream and becomes the final `{"error":...}` (`#L206-L212`).
- Clients read chunks with `GET /stream/{id}`. The max chunk is 1 MB (operation-reference).

**Progress updates:** `progress_update(job, p)` sends `{"status":"IN_PROGRESS","output":p}` to the **job-done** URL, with `isStream=false` (`rp_progress.py#L230-L236`, `rp_http.py#L75-L80`). Clients see it in `/status`.
- Caution: `rp@760aea2:ARCHITECTURE.md` L965-L970 describes progress as a POST to the ping URL. The code does not do that, so the code wins.

**Heartbeat:**
- `GET {PING_URL}?job_id=<comma-separated in-progress ids or omitted>&runpod_version=<v>` every `RUNPOD_PING_INTERVAL` (10 s), with request timeout 2× the interval (`rp_ping.py#L197-L207`, `worker_state.py` `get_job_list`).
- Sent from a separate process, so a busy handler never starves it (`rp_ping.py#L165-L185`).
- Pings are skipped unless `RUNPOD_AI_API_KEY` and `RUNPOD_POD_ID` are set (`#L169-L175`).
- **INFERRED**: the platform uses the `job_id` list to detect orphaned jobs. A Rust worker should ping from its own Tokio task and never block it on GPU work.

**Cancellation / timeout stop channel (newer SDK):**
- `GET {JOB_GET_URL with /job-take/→/job-stop/}` is a long poll (client timeout 90 s).
- Replies: `204` = none; `200 {"jobsToStop":["id",...]}`; `429` = back off 5 s (`rp_job.py#L30-L102`, `rp_scale.py#L412-L458`).
- The worker cancels just that job's task (`rp_scale.py#L460-L478`).

**Concurrency modifier:**
- `concurrency_modifier(current) -> int` is called before each take.
- The worker asks for `concurrency - (queued + in_progress)` jobs and uses the batch endpoint when that is >1 (`rp_scale.py#L111-L126`, `#L292-L342`).
- The queue is resized only when idle (`#L118-L121`).
- **INFERRED**: for one GPU per model, keep concurrency 1. Raise it only if the server batches requests internally.

**refresh_worker:**
- `config["refresh_worker"]`, or a handler return of `{"refresh_worker": true}`, adds `stopPod: true` to the done body (`rp_job.py#L222-L225`, `#L270-L276`).
- The SDK then stops its loop (`rp_scale.py#L502-L503`). The platform replaces the worker.

**Graceful shutdown:**
- SIGTERM/SIGINT set a shutdown flag. The worker stops taking jobs and drains in-flight tasks before exiting (`rp_scale.py#L128-L160`, `#L373-L410`).
- A startup ("prestart") failure claims one queued job, fails it with the reason, then `os._exit(1)` (`rp_scale.py#L253-L290`, `_health/fitness.py#L28-L43`).
- The SIGTERM → SIGKILL grace period on serverless is **not documented** (searched docs.runpod.io; not found). **INFERRED**: finish or fail the current job quickly on SIGTERM, and return an error result so the job is not left IN_PROGRESS.

**Rust worker outline (INFERRED design, derived from the above):**

```
loop take:   GET job-take (&job_in_progress=0|1) → 204/400: continue; 429: sleep 5s; 200: spawn job
task ping:   every 10s GET ping?job_id=<ids>&runpod_version=rust-x
task stop:   GET job-stop long-poll → cancel matching job tokens
per job:     POST stream chunks (…&isStream=false, body {"output":chunk})
             POST job-done (…&isStream=true|false, body {"output":…}|{"error":…}); retry 3x
headers:     Authorization: $RUNPOD_AI_API_KEY ; X-Request-ID: <job id>
```

### 1.2 Public queue-endpoint API

Sources: https://docs.runpod.io/serverless/endpoints/send-requests and https://docs.runpod.io/serverless/endpoints/operation-reference.

- Base URL: `https://api.runpod.ai/v2/{ENDPOINT_ID}`, header `authorization: Bearer <RUNPOD_API_KEY>`.
- Operations:
  - `POST /run` (async, 10 MB, result kept 30 min).
  - `POST /runsync` (20 MB, result kept 1 min; `?wait=` 1000–300000 ms).
  - `GET /status/{id}`, `GET /stream/{id}` (chunks ≤1 MB), `POST /cancel/{id}`.
  - `POST /retry/{id}` (only FAILED or TIMED_OUT), `POST /purge-queue`, `GET /health` (worker and job counts).
- Body: `{"input":{...}, "webhook": "<url>", "policy": {...}, "s3Config": {...}}`.
  - `policy.executionTimeout` is in ms: default 600000, min 5 s, max 7 days.
  - `policy.ttl` is in ms: default 24 h, min 10 s, max 7 days.
  - `policy.lowPriority` is a boolean.
- Webhook: POSTed on completion. Retries up to 2 times with 10 s delays if it does not get a `200`.
- Job statuses: `IN_QUEUE, IN_PROGRESS, COMPLETED, FAILED, CANCELLED, TIMED_OUT`.
- Rate limits per endpoint: `/run` 1000 req/10 s, `/runsync` 2000/10 s, `/status` and `/stream` 2000/10 s, `/cancel` 100/10 s.
- The endpoint-create response lists all request URLs (`requestUrls.run|runSync|status|stream|cancel|retry|purgeQueue|health`), per https://docs.runpod.io/api-reference-v2/serverless/create-a-serverless-endpoint.

### 1.3 Load-balancing endpoints (run our HTTP server unchanged)

Sources: https://docs.runpod.io/serverless/load-balancing/overview, `/build-a-worker`, `/worker-affinity`.

- Endpoint type "Load Balancer" (REST `type: LOAD_BALANCER`). Requests go to `https://ENDPOINT_ID.api.runpod.ai/<any path>` with `Authorization: Bearer`. Any HTTP framework works.
- Env: `PORT` (default 80), `PORT_HEALTH` (default = PORT), `HEALTH_CHECK_PATH` (default `/ping`). The port must also be listed under "Expose HTTP Ports".
- Health semantics:
  - `200` = healthy.
  - `204` = initializing. Cold start is measured from the first 204 to the first 200.
  - Any other code = unhealthy, and the worker is removed from routing.
- Limits: 2 min wait for a worker, **5.5 min processing per request**, 30 MB request and response. No queue, no retries, and "drops requests when overloaded".
- A port misconfiguration leaves workers up for 8 min, returning 502.
- **WebSockets are supported** ("Load balancing endpoints also support WebSocket connections"). Clients need `open_timeout` ≈ 60 s to survive scale-from-zero.
  - HTTP and WS share `PORT` (worker-lb-websocket README).
  - Whether open WS connections count as load for autoscaling is untested; the README's `test_scaling.py` exists to probe it. Suggested mitigation: `min_workers=1` or an HTTP keepalive.
- **Worker affinity**: every response carries `X-Runpod-Worker-Id`.
  - Resend `X-Runpod-Worker-Id: <id>` for soft pinning.
  - `strict <id>`: waits ~5 min, then 404 `affinity_worker_gone` or 400 `worker_timeout`.
  - `strict-resume <id>`: also resumes a scaled-down worker.
  - **INFERRED**: this is what makes multi-request WebRTC signaling (offer, then trickle ICE, then teardown) work on an LB endpoint.
- **INFERRED fit for fastvideo-rs**: good for `/v1/generate` calls under 5.5 min and for WHIP/WHEP-style signaling. Anything longer should use a queue endpoint, or return a job id and poll a pod.

### 1.4 FlashBoot, cached models, network volumes

- FlashBoot "reduces cold starts by retaining worker state after spin-down" and is enabled by default in the console (https://docs.runpod.io/serverless/endpoints/endpoint-configurations).
  - In REST v2 it is an enum `OFF | FLASHBOOT | PRIORITY_FLASHBOOT`, default `'OFF'` on create (create-a-serverless-endpoint OpenAPI `FlashBoot`).
  - The SDK notes that job-take may return 400 under FlashBoot (`rp_job.py#L150-L152`).
- Cached models are **Hugging Face only**, one model per endpoint, mounted at `/runpod-volume/huggingface-cache/hub/models--{org}--{name}/snapshots/{hash}/` (https://docs.runpod.io/serverless/endpoints/model-caching).
  - **INFERRED**: usable only if our weights live in a single HF repo in HF layout. Our multi-repo weight sets (see `scripts/gpu/weights-manifest.tsv`) don't fit.
- Network volumes mount at **`/runpod-volume`** on serverless, versus `/workspace` on pods (https://docs.runpod.io/storage/network-volumes).
  - Each one pins the endpoint to that volume's data center.
  - Several volumes can be attached, at most one per data center, and they are not synced.
  - Concurrent writes can corrupt data.
- Global volumes are region-independent, object-storage backed, "best for read-heavy workloads such as loading model weights", and mutually exclusive with a network volume (https://docs.runpod.io/serverless/storage/overview).

### 1.5 Endpoint and template creation (REST v2)

Source: https://docs.runpod.io/api-reference-v2/serverless/create-a-serverless-endpoint. The SDK uses the same API: `rp@760aea2:runpod/api/ctl_commands.py` → `POST /v2/templates`, `/v2/serverless` on `https://api.runpod.io`, per `api/rest.py#L26-L27`.

`POST https://api.runpod.io/v2/serverless` (Bearer):

```json
{
  "name": "fv-serve", "type": "QUEUE" | "LOAD_BALANCER",
  "image": "ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-<7>",
  "args": "{\"entrypoint\":[\"/opt/fastvideo-rs/bin/fv-serve\"],\"cmd\":[\"--mode\",\"runpod-queue\"]}",
  "disk": 120, "ports": ["8000/http"], "env": {"PORT": "8000", "FV_WEIGHTS": "/runpod-volume/weights"},
  "gpu": {"pools": ["<pool id from GET /v2/catalog/gpus>"], "excludedTypes": [], "count": 1,
          "minCudaVersion": "13.0"},
  "workers": {"min": 0, "max": 3, "idleTimeout": 5},
  "scaling": {"type": "QUEUE_DELAY", "queueDelay": 4} | {"type": "REQUEST_COUNT", ...},
  "timeout": 300000, "networkVolumes": ["<id>"], "dataCenterIds": ["..."], "flashboot": "FLASHBOOT",
  "templateId": "<optional serverless template>"
}
```

Field notes:
- `gpu.pools` are serverless **pool** ids, not pod GPU type ids. `excludedTypes` subtracts specific cards.
- `timeout` defaults to 300000 in this schema. The console docs say the execution timeout default is 600 s. This conflicts; send it explicitly.
- `workers.idleTimeout` is 1–3600 s.
- LOAD_BALANCER endpoints must use request-count scaling.
- Templates: `POST /v2/templates` with `serverless: true`, image, disk, env, ports (`ctl_commands.py`).
- Private images need a registry credential id (`registry`).
- Our pod script pulls the ghcr image with no credential, so it is public (**INFERRED** from `scripts/gpu/runpod-http.sh` `create_pod`).

### 1.6 WebRTC / UDP on serverless

- "Expose HTTP/TCP ports: Exposes the worker's public IP and port for direct external communication. Required for persistent connections like WebSockets" (endpoint-configurations).
- The worker reads `RUNPOD_PUBLIC_IP` and `RUNPOD_TCP_PORT_<internal>` and publishes them to the client with `progress_update` (`runpod-workers/worker-websocket@67ca740:rp_handler.py#L52-L58`, README).
- Port protocol values are only `http` or `tcp` (https://docs.runpod.io/api-reference/pods/POST/pods.md, "Protocol can be either http or tcp"). The pods docs say "Pods do not support UDP connections" (https://docs.runpod.io/pods/configuration/expose-ports).
- **Conclusion (confirmed for inbound)**: no inbound UDP on Runpod serverless. WebRTC is possible only via ICE-TCP on an exposed TCP port, TURN over TCP/TLS, or outbound WHIP (§5).

---

## 2. Runpod pods

How we create pods today: `scripts/gpu/runpod-http.sh` `create_pod()`.
- REST v1 call: `POST https://rest.runpod.io/v1/pods` with `imageName`, `cloudType: SECURE`, `gpuTypeIds`, `containerDiskInGb`, `networkVolumeId`, `volumeMountPath: /workspace`, `dataCenterIds`, `ports: ["8000/http"]`, `dockerStartCmd`.
- It picks a stocked data center through GraphQL `dataCenters{gpuAvailability}`.
- Results come back over `https://<pod>-8000.proxy.runpod.net`.
- A price cap and a wall-clock cap guard spending, and the pod is deleted at the end.

The serve deployment can reuse this unchanged. Swap the start command for the server binary and add a TCP port for ICE-TCP or an admin port.

**HTTP proxy** (https://docs.runpod.io/pods/configuration/expose-ports):
- `https://[POD_ID]-[INTERNAL_PORT].proxy.runpod.net`, HTTPS always, "Expose HTTP Ports (Max 10)".
- **100-second Cloudflare timeout → 524.** A streaming response is fine if it starts within 100 s (https://www.runpod.io/blog/runpod-proxy-guide).
- The docs recommend TCP exposure for WebSocket apps "for persistent connections that might exceed timeout limits". WebSocket support through the proxy is not explicitly documented. **INFERRED**: WS likely works through Cloudflare, but idle connections should send pings well under 100 s.

**Public TCP:**
- "Expose TCP Ports" gives a public IP and a random external port, shown under Connect → Direct TCP Ports.
- REST fields: `supportPublicIp` (Community Cloud only; Secure Cloud "will always have a public IP"), and the response's `publicIp` + `portMappings` `{"22":10341}` (https://docs.runpod.io/api-reference/pods/POST/pods.md).
- v2: `runtime.ports[] {private, public, type, ip}` (https://docs.runpod.io/api-reference-v2/pods/create-a-pod).
- Symmetrical ports: request a port >70000 and read `$RUNPOD_TCP_PORT_70000` (expose-ports).
- Runtime env: `RUNPOD_PUBLIC_IP`, `RUNPOD_TCP_PORT_22`, `RUNPOD_POD_ID`, `RUNPOD_DC_ID`, `RUNPOD_VOLUME_ID`, `RUNPOD_API_KEY` (pod-scoped), `RUNPOD_GPU_COUNT` (https://docs.runpod.io/pods/templates/environment-variables).
- Mappings change whenever the pod resets. IPs can change on Community Cloud.

**UDP**: "Pods do not support UDP connections." For WebRTC this means:
- ICE-TCP: advertise a TCP host/srflx candidate as `RUNPOD_PUBLIC_IP:RUNPOD_TCP_PORT_<n>`. **INFERRED**: this requires symmetrical mapping or manually written candidates, because the internal port ≠ the external port.
- Or TURN over TCP/TLS, e.g. Cloudflare TURN `turn.cloudflare.com` 3478/tcp, 80/tcp, 5349/tcp, 443/tcp (https://developers.cloudflare.com/realtime/turn/).
- Or outbound WHIP (§5).

**Global networking**: pod-to-pod private network, `POD_ID.runpod.internal`, 100 Mbps, NVIDIA GPU pods in 17 DCs (https://docs.runpod.io/pods/networking). Too slow for video fan-out; fine for control traffic.

**Secrets**: `{{ RUNPOD_SECRET_<name> }}` in template env (https://docs.runpod.io/pods/templates/secrets). Changing env restarts the pod (environment-variables page).

---

## 3. Vast.ai

### 3.1 Instance creation (what we do today)

- `scripts/gpu/validate.sh` `create_instance`: `vastai create instance <offer> --image $IMAGE --disk N --ssh --direct --label fvgpu-... --cancel-unavail --raw` → `.new_contract`.
- `wait_ready` polls `vastai show instance --raw .actual_status` until `running`, and treats `exited|offline|error` as a bad host.
- `scripts/gpu/lib.sh`: `vast_ssh_target` and `vast_destroy` (3 tries).
- `scripts/vast-*.sh`: the legacy ssh+rsync flow.

CLI flags (https://docs.vast.ai/cli/reference/create-instance):
- `--image`, `--disk` (GB, default 10), `--env '<docker -e/-p/-h flags>'`, `--onstart-cmd`, `--entrypoint`, `--args` (must be last).
- `--ssh|--jupyter|--direct`, `--label`, `--login` (private registry), `--cancel-unavail`, `--bid_price` (interruptible).
- Volumes: `--create-volume/--link-volume/--volume-size/--mount-path`.
- "If you use args/entrypoint launch mode, we create a container from your image as is, without attempting to inject ssh and or jupyter". ssh/jupyter modes replace the image entrypoint, so use onstart (https://docs.vast.ai/guides/instances/docker-environment).

REST (https://docs.vast.ai/api-reference/instances/create-instance):
- `PUT https://console.vast.ai/api/v0/asks/{offer_id}/`, Bearer.
- Body fields: `image, disk, runtype (ssh|jupyter|args|ssh_direct|...), env (the docs show "-e K=V -p 8000:8000", but the API in practice requires a JSON object such as {"-p 8080:8080":"1","K":"V"}; see §5.1), onstart (≤4048 chars), args[], args_str, image_login, label, target_state, price, cancel_unavail, volume_info{create_new, volume_id, size, mount_path}`.
- Returns `{"success":true,"new_contract":<id>}`.
- If `actual_status` becomes `exited|unknown|offline`, it will never reach running.

### 3.2 Ports, public IP, UDP

Source: https://docs.vast.ai/guides/instances/connect/networking and `/docker-environment`.
- Shared public IPs. Each internal port maps to a *random* external port. Limit: 64 open ports per instance.
- `-p 8081:8081 -p 8082:8082/udp` is supported. Image `EXPOSE` ports are auto-mapped.
- Identity mapping: `-p 70000:70000` → read `$VAST_TCP_PORT_70000`.
- Env:
  - `VAST_TCP_PORT_<internal>` and **`VAST_UDP_PORT_<internal>`** give the external ports.
  - `PUBLIC_IPADDR` is the public IP. It is set at startup and not updated if the IP changes; refresh via `vastai show instance $CONTAINER_ID --api-key $CONTAINER_API_KEY`.
  - Also `CONTAINER_ID`, `CONTAINER_API_KEY` (per-instance key), `GPU_COUNT`, `VAST_CONTAINERLABEL`, `DATA_DIRECTORY`.
- Custom env is not visible in SSH sessions unless exported to `/etc/environment`.
- **WebRTC on Vast (INFERRED)**:
  - Open one UDP port, e.g. `-p 70010:70010/udp` for identity mapping, if identity mapping also applies to UDP (not stated explicitly).
  - Bind the ICE UDP socket to that internal port, and advertise a host candidate `PUBLIC_IPADDR:$VAST_UDP_PORT_<n>` (webrtc-rs: `SettingEngine::set_nat_1to1_ips` + a single-port UDP mux).
  - Multiplex all peers on one port. Keep TURN as a fallback for clients behind UDP-hostile networks.

### 3.3 Storage

- Volumes are **local only**: tied to one physical machine, fixed size, and deletable only after the instance is gone (https://docs.vast.ai/guides/instances/storage/volumes).
- **INFERRED**: there is no network volume equivalent. Weights either come baked into the image or are fetched at boot (the runtime image ships `hf-fm`), or a local volume is reused when re-renting the same machine.

### 3.4 Vast Serverless (endpoint → workergroup → PyWorker)

Architecture (https://docs.vast.ai/guides/serverless/architecture, `/overview`):
- An endpoint has scaling parameters: `max_workers`, `min_load`, `target_util`, `cold_mult`, `inactivity_timeout`, `max_queue_time`, `target_queue_time`.
- Each endpoint holds workergroups: a template + `search_params` + `launch_args` + `gpu_ram`.
- Workers are ordinary instances running a PyWorker.
- Flow:
  1. The client does `POST /route` `{"endpoint":name,"cost":N}`.
  2. It gets `{url, reqnum, signature, cost, endpoint, __request_id}` (https://docs.vast.ai/api-reference/serverless/route), or a no-worker status.
  3. It POSTs `{"auth_data":{signature,cost,endpoint,reqnum,url,request_idx},"payload":{...}}` **directly to the worker URL**.
- CLI: `vastai create endpoint --endpoint_name ... --max_workers ... --inactivity_timeout ...` and `vastai create workergroup --template_hash ... --endpoint_name ... --launch_args "..." --gpu_ram N` (https://docs.vast.ai/cli/reference/create-endpoint, `/create-workergroup`).

Worker-side protocol (from `vc@92caa2d:vastai/serverless/server/lib/*` and `pw@60cfeca:start_server.sh`):

| Item | Detail | Source |
|---|---|---|
| Bootstrap | Template onstart runs `start_server.sh`: clones `PYWORKER_REPO`@`PYWORKER_REF`, makes a uv venv, installs `vastai` SDK, runs `python -m worker` | `pw@60cfeca:start_server.sh#L173-L213`, `#L354-L358` |
| Env | `CONTAINER_ID` (required), `REPORT_ADDR` (default `https://run.vast.ai`, comma-list), `WORKER_PORT` (default 3000), `USE_SSL` (default true), `MASTER_TOKEN`, `UNSECURED`, `MODEL_LOG`, `BACKEND` | `start_server.sh#L19-L21`, `#L92`; `backend.py#L91-L100` |
| TLS | Generates a key+CSR and gets the cert signed by `POST https://console.vast.ai/api/v0/sign_cert/?instance_id=$CONTAINER_ID`, into `/etc/instance.{key,crt}` | `start_server.sh#L264-L316`; `server.py#L16-L26` |
| Worker URL | `http[s]://$PUBLIC_IPADDR:$VAST_TCP_PORT_$WORKER_PORT` | `metrics.py#L20-L25` |
| Auth | GET `{REPORT_ADDR}/pubkey/` (RSA PEM, up to 5 tries, backoff). Per request: verify PKCS#1 v1.5 SHA-256 signature over `json.dumps({"url": auth_data.url}, indent=4, sort_keys=True)`; 401 on failure; skipped if `UNSECURED=true` | `backend.py#L392-L427`, `#L751-L779` |
| Queue limit | If the estimated wait > handler `max_queue_time` → 429 | `backend.py#L513-L517` |
| Metrics | POST `{REPORT_ADDR}/worker_status/` JSON `WorkerStatusData{id, mtoken, version, loadtime, cur_load, rej_load, new_load, error_msg, max_perf, cur_perf, cur_capacity, max_capacity, num_requests_working, num_requests_recieved, additional_disk_usage, working_request_idxs, url}`. Sent when state changes (1 s tick) and at least every 10 s | `metrics.py#L14`, `#L145-L152`, `#L236-L309`; `data_types.py#L344-L363` |
| Completion | POST `{REPORT_ADDR}/delete_requests/` `{worker_id, mtoken, requests:[{request_idx, success, status, entered_queue_at, work_started_at, work_completed_at}]}` | `metrics.py#L173-L233` |
| Readiness | Tails the model log file for `on_load` prefixes, then runs a benchmark to set `max_perf`. "Loaded" is reported only after the pubkey is fetched | `backend.py#L781-L930`; https://docs.vast.ai/guides/serverless/creating-new-pyworkers |
| Fixed routes | `/session/create|end|get|health`, `/pyworker/update` (auth = mtoken), plus plain HTTP `/session/end` on `WORKER_PORT+1` | `server.py#L30-L60` |
| Errors at boot | POST `{REPORT_ADDR}/worker_status/` `{id, mtoken, version, error_msg, url}` | `start_server.sh#L31-L54` |

**INFERRED options for fastvideo-rs on Vast Serverless**:
- (a) Ship a tiny Python `worker.py` that proxies to the Rust server on localhost. That is the supported path, but it adds Python and uv to the image or installs them at boot.
- (b) Reimplement the table above in Rust: pubkey fetch, RSA verify, the status and delete loops, and the TLS cert fetch. It is not a documented contract and can drift with `vastai-sdk` releases.

Prefer (a) initially. Pin `SDK_VERSION` and `PYWORKER_REF`.

---

## 4. Packaging, health, weights, warm start, shutdown, secrets (per target)

Base: CI publishes `ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-<7>` (and `build-<id>`; `latest` on main) from `docker/gpucheck.Dockerfile` target `runtime` (`.github/workflows/gpucheck-runtime-image.yml`).
- It contains Ubuntu 22.04, CUDA 13.4 runtime libraries (needs driver ≥580), ffmpeg, openssh-server, `hf-fm`, and `/opt/fastvideo-rs/target/release/fv-gpucheck`.
- It sets **no ENTRYPOINT/CMD** and has no Python (`docker/gpucheck.Dockerfile`).
- **INFERRED**: add the server binary to this same `runtime` stage, or a `serve` target `FROM runtime`, with `ENTRYPOINT ["/opt/fastvideo-rs/bin/fv-serve"]`. Select the mode with an env var: `FV_SERVE_MODE=http|runpod-queue|runpod-lb|vast-pyworker`.
- Filter hosts for CUDA ≥13.0: Runpod `gpu.minCudaVersion: "13.0"` (endpoint API) or `allowedCudaVersions`; Vast `cuda_vers>=13.0` (already in validate.sh, per the Dockerfile comment).

| Concern | Runpod pod | Runpod Serverless (queue) | Runpod Serverless (LB) | Vast instance / serverless |
|---|---|---|---|---|
| Start | `dockerStartCmd` / REST `args` → server binary | ENTRYPOINT = server in `runpod-queue` mode (Rust job-take loop, §1.1) | ENTRYPOINT = server listening on `$PORT` | `--entrypoint`/`--args` (args runtype = image as is), or onstart in ssh mode. Serverless: template onstart → `start_server.sh` + our `worker.py`, server started by onstart |
| Health | Our `/healthz` via proxy; the driver polls like `wait_up` in runpod-http.sh | Heartbeat to `RUNPOD_WEBHOOK_PING`; fail fast and exit non-zero on CUDA init failure (the SDK uses `os._exit(1)`, `_health/fitness.py`) | `GET /ping`: 204 while weights load, 200 when ready (§1.3) | `/healthz`. Serverless: `on_load` log line + benchmark (§3.4) |
| Weights | Network volume at `/workspace` (current practice: `RUNPOD_VOLUME_NAME` volumes, read-only use; runpod-http.sh notes silently dropped writes) | Network volume at `/runpod-volume` (pins the DC; attach one per DC) or a global volume; baking 10s of GB into the image slows cold pulls (INFERRED) | Same as queue | Download at boot with `hf-fm` to the container disk, or a local volume; bake small models |
| Warm start | Keep the pod running | `workers.min ≥1` removes cold starts (billed); FlashBoot; keep weights resident across jobs in one process | Same; `strict-resume` affinity | Vast serverless `cold_mult`/`min_cold_load` keep stopped instances with a warm disk (INFERRED from parameter names; https://docs.vast.ai/guides/serverless/serverless-parameters) |
| Shutdown | `DELETE /pods/{id}`; container disk wiped on stop (https://docs.runpod.io/pods/manage-pods) | SIGTERM: stop taking, finish or fail in-flight, POST results; grace period undocumented | SIGTERM: stop accepting, drain ≤5.5 min | `vastai destroy instance`; serverless zero-downtime update: https://docs.vast.ai/guides/serverless/zero-downtime-worker-update |
| Secrets | `{{ RUNPOD_SECRET_x }}` in env | Endpoint env (console or REST `env`); secrets syntax also in templates | Same | `-e` in `--env`, `vastai create env-var` account env vars (https://docs.vast.ai/cli/reference/create-env-var); never in onstart logs |
| Large outputs | Serve files from the pod | Upload to S3/R2 and return a URL (10/20 MB caps; `s3Config`) | ≤30 MB response | Serve directly |

---

## 5. How outbound WHIP/WHEP changes the picture

Facts:
- WHIP ingest and WHEP playback are both plain HTTP(S) signaling. Media is WebRTC.
- Cloudflare Stream: `https://customer-<CODE>.cloudflarestream.com/<SECRET>/webRTC/publish` and `.../<INPUT_UID>/webRTC/play`.
  - Codecs: VP9, VP8, H.264 Constrained Baseline L3.1. B-frames must be 0.
  - WHIP and WHEP must be used together (no HLS or recording).
  - Source: https://developers.cloudflare.com/stream/webrtc-beta/.
- Cloudflare TURN: `turn.cloudflare.com` over 3478/udp, 443/udp, 3478/tcp, 80/tcp, and TLS 5349/tcp, 443/tcp. Free with the Realtime SFU; otherwise $0.05/GB (https://developers.cloudflare.com/realtime/turn/).

Implications (**INFERRED**):
1. The GPU worker becomes a WebRTC **client** (WHIP publisher), so it needs no inbound port. The "no UDP" limit on Runpod pods and serverless stops mattering *if outbound UDP egress works*.
   - Runpod documents only inbound exposure. Outbound UDP is neither documented nor tested here, and it must be verified on a real pod and serverless worker.
   - If egress UDP is blocked, ICE falls back to TURN over TCP/TLS 443 (Cloudflare supports it), at some latency cost.
2. On **Runpod Serverless**, a streaming session maps onto a **queue job**:
   - `/run` with `{"input":{"prompt":...,"whip_url":...,"duration_s":N}}` and `policy.executionTimeout` ≥ stream length + model load.
   - The worker publishes via WHIP for the job's duration, sends `progress_update`s (e.g. `{"state":"live","whep_url":...}`) readable from `/status`, and ends with a job-done carrying stats.
   - `/cancel` reaches the worker through the job-stop channel (§1.1) and should tear down the WHIP session (HTTP DELETE on the WHIP resource URL, per WHIP).
   - Execution timeout: 10 min by default, up to 7 days (§1.2).
3. **LB endpoints** can host the signaling API but not the media. Their 5.5 min per-request cap doesn't matter when the long-lived thing is the outbound WebRTC session and not an HTTP request.
   - This is dangerous, though: nothing ties worker lifetime to an active stream, so idle-timeout scale-down could kill a live stream. The queue model (a job in progress keeps the worker) is the safer fit.
4. One image runs everywhere: pods, serverless, Vast and plain VMs. Viewers use WHEP from the SFU. Local development can use MediaMTX (WHIP/WHEP-capable) in place of Cloudflare.

### 5.1 strobe: a working reference for this design

strobe is our near-real-time WebRTC video generation server. Its source was read from a local, git-excluded copy at `.refsrc/strobe/` (not committed; the paths below are relative to that copy). Everything in this section comes from that source unless marked INFERRED.

**Architecture.**
- The pipeline is "prompt → causal Wan DiT → VAE decode → FramePacer → aiortc H.264 track → WHIP → MediaMTX/Cloudflare → viewers". The GPU host needs "**no inbound port** — that's why it runs on a RunPod serverless worker unchanged" (`CLAUDE.md` §What this is).
- It "works end to end on a self-hosted MediaMTX relay". Throughput was measured "on the live RunPod endpoint", e.g. H100 SXM warm ~19.7 unique fps (`CLAUDE.md` §Current state).

**WHIP publisher** (`src/strobe/whip.py`):
- POSTs the full offer SDP with `Content-Type: application/sdp` and accepts 200 or 201. It reads the answer and the `Location` header, which may be relative and is resolved against the request URL (RFC 3986). Teardown is a best-effort `DELETE <resource URL>`.
- Auth: HTTP Basic (`user:token`) for MediaMTX internal auth, `Bearer <token>` for Cloudflare.
- ICE is non-trickle: aiortc gathers candidates during `setLocalDescription`, so the offer is complete.
- The POST times out after 30 s (`timeout_s=30.0`).
- H.264 is reordered to be offered first. Cloudflare Stream ingests H.264 only, and a VP8 negotiation "negotiates fine but produces a black stream".
- A short keyframe interval is forced (`KEYFRAME_SECONDS=2.0` per `CLAUDE.md` config list) so joining players don't wait on an IDR.
- No `RTCConfiguration` / ICE servers are set anywhere in `src/strobe/` (grep: no `iceServers`, `RTCIceServer` or `turn:`). The publisher therefore relies on aiortc's defaults, with no TURN.
  - **INFERRED**: aiortc has no ICE-TCP, so media from the live Runpod serverless endpoint must have flowed over outbound UDP. That is strong evidence that **outbound UDP egress works on Runpod serverless** (partly closes gap 1 in §7).

**Runpod serverless** (`src/strobe/runpod_handler.py`, `deploy/runpod/README.md`, `scripts/deploy/runpod-endpoint-create.sh`, `deploy/ui/src/server/providers/runpod.ts`):
- "One queued job = one stream session. The worker dials OUT to the WHIP endpoint."
- Job input: `{"input":{"prompt","whip_url","whip_token","duration_s","local_attn_size","sink_size","image_url"|"image_b64"}}`. The handler is an async `runpod.serverless.start({"handler": handler})` Python handler.
  - It returns `{"id","state","stats"}`, or `{"error", "stats"}` on failure.
  - It sends no progress updates. Viewers find the stream at the SFU, not via `/status`.
- Endpoint type: "Use a QUEUE-based endpoint, not load-balancing: streams are long jobs". "Set executionTimeout >= your max stream length (default is 600s)".
- Settings actually used. They go through REST **v1** `https://rest.runpod.io/v1` (`POST /templates`, then `POST /endpoints`), not the v2 API in §1.5:
  - Template: `isServerless: true`, `imageName: ghcr.io/<owner>/strobe:gpu` (or `@sha256:` digest pin), `containerDiskInGb` 50, `volumeInGb` 0, `dockerStartCmd: ["python3.12","-m","strobe.runpod_handler"]`, `env` = `STROBE_*` + HF/Torch caches on the volume.
  - Endpoint: `computeType GPU`, `gpuTypeIds` (default `NVIDIA H100 80GB HBM3, NVIDIA A100-SXM4-80GB, NVIDIA GeForce RTX 4090`), `workersMin` 1 in the script (0 in the UI), `workersMax` 3, `idleTimeout` 300, `scalerType QUEUE_DELAY`/`scalerValue 4`, `executionTimeoutMs = STROBE_MAX_STREAM_SECONDS×1000` (default 600 s), `flashboot: true`, and optional `networkVolumeId`.
  - Production guidance in `CLAUDE.md`: "`gpuTypeIds` = SXM/NVL only, `executionTimeoutMs=1800000` (cold start ~7 min), and `workersMin≥1`".
- Jobs are submitted with `POST https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run` and polled at `/status/<id>` (`scripts/deploy/runpod-stream.sh`).
- Weights and caches live on the network volume at `/runpod-volume`:
  - checkpoint, HF cache;
  - `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` under `<volume>/torch-compile`, "so only the first cold worker pays the compile";
  - TensorRT VAE plans at `/runpod-volume/trt-vae` (`scripts/deploy/runpod-configure-optimal.sh`).
  - Weights are uploaded through Runpod's S3-compatible API at `https://s3api-<dc>.runpod.io` with separate S3 keys, no pod needed (`runpod-common.sh` `runpod_s3_endpoint`).
- Image pull: the GHCR package is public, so no registry credential is needed. If private, the code recreates `containerregistryauth` on every deploy because "RunPod stores registry credentials immutably (there is no update endpoint)" and a rotated token fails silently.
  - Templates pin the image **digest** so FlashBoot or scale-up relaunches pull the exact tested image (`deploy/runpod/README.md`).
- The Runpod pod variant uses the same image with the default CMD (`strobe.server`), `ports: ["8080/http"]`, and health at `https://<pod>-8080.proxy.runpod.net/healthz` (`runpod.ts` `provisionPod`/`getPodStatus`).

**Vast provider flow** (`deploy/ui/src/server/providers/vast.ts`, `scripts/deploy/vast-common.sh`, `vast-create.sh`):
- Offer search: `POST https://console.vast.ai/api/v0/bundles` with filters `rentable`, `verified`, `disk_space>=`, `cuda_max_good>=12.4`, `reliability>=0.9`, and **`direct_port_count>=1`**. The last one is "mandatory or :8080 is unreachable — proxied ports only forward jupyter/ssh".
- Rent: `PUT https://console.vast.ai/api/v0/asks/<offer>/` with `{client_id:"me", image, label, disk, runtype:"ssh_direct", target_state:"running", onstart, env, image_login?, volume_info?, price?}`.
- **`env` must be a JSON object, not the docker flag string the docs show.** The API rejects a string with "invalid env type: env must be a dict". A published port is the key `"-p 8080:8080"` with value `"1"` (`vast-create.sh` comment; `vast.ts`).
  - This contradicts the REST reference cited in §3.1, which describes `env` as a flag string. Trust strobe's observed behaviour.
- Under every `ssh_*` runtype, Vast's sshd is PID 1 and the image CMD never runs. **onstart must start the server** (`nohup python3.12 -m strobe.server …&`).
  - onstart fetches weights to a `.part` file and renames it on success (Hugging Face directly, or `aws s3 sync` from the Runpod volume's S3 endpoint).
- Endpoint discovery: `GET /api/v0/instances/<id>/` → `http://<public_ipaddr>:<ports["8080/tcp"][0].HostPort>`.
  - Health: `/healthz` (the h3fast serve mode also tries `/health`, and accepts `ready|ok|status=="ok"`).
  - Destroy: `DELETE /api/v0/instances/<id>/`. Listing uses `/api/v1/instances/`.
- Discipline (`vast-common.sh`): ledger, a destroy-on-exit trap, dry-run mode, budget deadline, and a two-unmeasured-rentals stop rule. `CLAUDE.md` notes instances "bill until destroyed".
- `h3fast-deploy.ts` spawns `scripts/deploy/vast-create-h3fast.sh` for one persistent H100. It needs `VAST_API_KEY` plus the Runpod volume S3 keys, because weights are synced from the Runpod volume. Outside the US, `CLAUDE.md` recommends downloading from Hugging Face instead (45 GiB in 60 s vs 206–646 Mbit/s from the volume).

**Plain VMs and the SFU:**
- `deploy/cloud-init.yaml` (Vultr, Hetzner, Latitude, Lambda) installs docker + nvidia-container-toolkit and fetches the checkpoint. It then runs `docker run --gpus all -p 8080:8080 --env-file … -v /opt/strobe/weights:/weights`.
- MediaMTX (`deploy/mediamtx/mediamtx.yml`): WHIP and signalling on `:8889/tcp` (WHIP URL `http://<host>:8889/strobe/whip`), **media on `:8189/udp`**, LL-HLS on `:8888`. Auth is internal: user `strobe` may publish, anyone may read.
  - The SFU needs public UDP ("Railway can't host it (no public UDP)"; Fly needs a dedicated IPv4, per `CLAUDE.md`).
  - **INFERRED**: the SFU can therefore never be colocated on a Runpod pod. Put it on a VM, Fly, Cloudflare, or a Vast instance with a `/udp` port.

**Takeaways for fastvideo-rs (INFERRED):**
- Adopt the same split:
  - Runpod **queue** job = stream session. Set `executionTimeoutMs` ≥ max stream + cold start (strobe uses 1800000), `workersMin ≥1` for warm starts, and a digest-pinned image.
  - On pods and Vast: a long-running HTTP server with `/healthz`.
- A Rust WHIP publisher (webrtc-rs) must:
  - offer H.264 Constrained Baseline first, with B-frames 0;
  - send complete (non-trickle) SDP;
  - resolve a relative `Location`;
  - `DELETE` on teardown;
  - support both Bearer and Basic auth.
- On Vast, `ssh_direct` + onstart is the proven path. Our current `validate.sh` uses the CLI's `--ssh --direct` (§3.1).

---

## 6. Proposed concrete settings (INFERRED, to validate)

**Runpod queue endpoint, streaming or batch:**
- `type: QUEUE`, `workers {min: 0 (1 for warm), max: N, idleTimeout: 60}`, `scaling {QUEUE_DELAY, 4}`.
- `timeout` (execution) ≥ the maximum stream or batch length + load time, e.g. 1800000 ms.
- `flashboot: FLASHBOOT`, `gpu.minCudaVersion: "13.0"`, `networkVolumes: [one per DC holding weights]`.
- `env {FV_SERVE_MODE: runpod-queue, FV_WEIGHTS: /runpod-volume/weights}`.
- Clients may override per job via `policy.executionTimeout` and `ttl`.

**Runpod LB endpoint, sync API and signaling:**
- `type: LOAD_BALANCER`, `ports: ["8000/http"]`, `env {PORT: 8000, PORT_HEALTH: 8000, FV_SERVE_MODE: http}`.
- `/ping`: 204 until warm, then 200. Request-count scaling. Clients retry "no workers available", WS `open_timeout` 60 s.

**Runpod pod:**
- Reuse `runpod-http.sh create_pod`. Set `ports: ["8000/http", "70000/tcp"]` if ICE-TCP is wanted, and set the start command to the server.

**Vast instance:**
- `vastai create instance <offer> --image ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-<7> --disk 150 --env '-p 8000:8000 -p 70010:70010/udp -e FV_SERVE_MODE=http' --entrypoint /opt/fastvideo-rs/bin/fv-serve --args ...`.
- Discover the external addresses from `PUBLIC_IPADDR`, `VAST_TCP_PORT_8000` and `VAST_UDP_PORT_70010`.
- Over REST, follow strobe's proven form instead (§5.1):
  - filter offers on `direct_port_count>=1`;
  - `runtype: ssh_direct` with an onstart that launches the server;
  - `env` as a JSON object;
  - endpoint from `ports["8000/tcp"][0].HostPort`.

---

## 7. Gaps and open questions

1. **Outbound UDP egress** from Runpod pods and serverless workers is undocumented. strobe's aiortc WHIP publisher, which has no TURN and no ICE-TCP, streams from a live Runpod serverless endpoint (§5.1). That is strong evidence egress UDP works on serverless (INFERRED). It is still unconfirmed on pods; confirm with a STUN binding request to `stun.cloudflare.com:3478`.
2. The **SIGTERM grace period** on Runpod Serverless and the effect of scale-down on in-flight jobs are undocumented.
3. **WebSocket support through the Runpod pod HTTP proxy** is not explicitly documented, beyond the 100 s timeout.
4. The **execution-timeout default** conflicts: the console docs say 600 s, the REST v2 schema says `timeout` default 300000 ms. Always set it.
5. The internal job-take/job-done/job-stop protocol is **not a public contract**. It comes from SDK source, and the SDK's own ARCHITECTURE.md disagrees with its code on progress updates. Pin behaviour to `rp@760aea2` and re-check on SDK upgrades.
6. It is unconfirmed whether Vast identity mapping (`>70000`) applies to `/udp` ports, and whether serverless workergroups' `launch_args` allow `/udp` port options.
7. Vast `/route` host: the docs' OpenAPI server is `https://console.vast.ai` (`POST /route`), while worker `REPORT_ADDR` defaults to `https://run.vast.ai`. Confirm which host clients should call.
8. Vast REST `env`: the docs describe a docker flag string, but strobe observed the API reject it ("env must be a dict"). Use the JSON-object form (§5.1).
9. It is unknown whether LB endpoints count open WebSocket connections as load for scaling (probe with `worker-lb-websocket/test_scaling.py`).
