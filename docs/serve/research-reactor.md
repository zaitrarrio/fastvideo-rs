# Reactor Runtime: protocol and architecture research

Status: research note, 2026-09-27. Scope: the exact client-facing contract of the
Reactor Runtime, so that a Rust-native server (Tokio + Axum + a WebRTC stack)
driving the fastvideo-rs CUDA engine can stand in for the Python runtime without
changing any Reactor client SDK.

## 0. Sources and citation convention

The repository the task calls "reactr-team/reactor-runtim" is
**`reactor-team/reactor-runtime`** (public, Apache-2.0). It was found by web
search ("GitHub - reactor-team/reactor-runtime: The Runtime for Real-Time").
Two sibling repos from the same org are also public and were read, because they
pin down the client side of every exchange:

| Tag | Repo | Commit read | Notes |
|---|---|---|---|
| **RT** | https://github.com/reactor-team/reactor-runtime | `888db38a9a13fa19dee0e008aa76d21f2d59f53e` (2026-09-25), package version `3.6.0` | the Python runtime (the thing being replaced) |
| **SDK** | https://github.com/reactor-team/reactor-client-sdks | `f9df0e05181b5b7a46ce179f098ea478ba36e657` (2026-09-21) | Rust core (`crates/reactor-core`, `crates/reactor-protocol`) shared by the JS (wasm), Python (ctypes FFI), C++, Swift and Java SDKs |
| **RW** | https://github.com/reactor-team/reactor-webrtc | `bebf63e42624ee440e066b692494dd3229d29ae7` (2026-09-15), version `0.18.0` | Rust + PyO3 wrapper over an owned libwebrtc build; RT pins `reactor-webrtc==0.18.0` |

| **LS** | user-supplied `/home/user/refsrc/d5fc0bdf-infinite-livestream-source/` ("Reactor Infinite Livestream", Apache-2.0, `NOTICE`: "Copyright 2026 Reactor Technologies, Inc."; contains the fast-h3 model from reactor-cookbook) | as uploaded | a **working production Reactor model with video + audio** (`fast-h3/`), plus a Python-SDK client (`streaming-client/`) |
| **PyPI** | `reactor_runtime-3.2.6-py3-none-any.whl` and `reactor_sdk-1.6.0-…manylinux_2_34_x86_64.whl`, fetched with `pip download --no-deps` into `/tmp/claude-0/pypi/` | — | 3.2.6 is the runtime version that fast-h3 builds against (`LS:fast-h3/reactor.yaml` `build.runtime_version: "3.2.6"`) |

Citations look like `RT:src/reactor_runtime/transport/webrtc/router.py`. All
paths are relative to the repo root at the commit above. Anything I derived
rather than read is marked **INFERRED**. The clones used were at
`/tmp/claude-0/reactor-runtime`, `/tmp/claude-0/reactor-client-sdks`, and
`/tmp/claude-0/reactor-webrtc`. They were read only, never modified.

Public docs are linked from RT:README.md (https://docs.reactor.inc/deploy/overview).
I did not need them, because the source settled every question below.

---

## 1. Summary of the contract a replacement must honour

1. **HTTP (Axum)**: a fixed set of JSON routes. `POST /start_session`,
   `GET /session`, `POST /stop_session`, `GET /schema`, and the WebRTC
   signalling group under `/sessions/{sid}/transport/webrtc/…`. Uploads,
   clips, `/events` (SSE), `/health` and `/metrics` are auxiliary. The spec is
   committed at `RT:api/openapi.json`.
2. **Signalling**: HTTP only, with no WebSocket. The client POSTs an SDP offer
   and gets back **202**. It then polls GET for the answer (202 until ready,
   then 200 with the answer, which embeds all server candidates and
   `a=end-of-candidates`). The client trickles its candidates by POST. A
   reconnect is a PUT of a new offer on the same connection id.
3. **Data channels**: the client creates two channels, labelled **`"data"`**
   and **`"control"`**, with default options (reliable and ordered). There is
   no length prefix: one SCTP message is one protocol message.
4. **Two wire encodings share one vocabulary** (the `reactor_wire.v1`
   messages). **v1** is binary protobuf, with each message on its own channel.
   **v0** is legacy JSON text, where nearly everything rides the `"data"`
   channel under `{"scope": …}`. The server detects the encoding by sniffing
   the connection's first inbound frame: a JSON text frame means v0, anything
   else means v1. Current SDKs (reactor-core in Rust) speak **v1**. RT still
   treats v0 as the default until the first frame arrives.
5. **Media**: server-to-client video, with the codec negotiated from the
   preference list VP9 (profile 0), VP8, H264 (42e01f, packetization-mode 1),
   AV1, H265. Audio is optional and uses **Opus**. The runtime always pushes
   it as **48 kHz mono int16 PCM** in 10 ms frames. Optional client-to-server
   camera/mic tracks. **A session's tracks are exactly the fields of the
   model's `Output` subclass**, so a video-only model simply declares no
   `Audio` field (§4.5). Outbound tracks start **paused**; the client sends
   `ResumeTrack` for each one it wants. Frames are paced by a per-connection
   pacer at the model-reported fps. Audio rides in the same per-frame bundle
   as its video frame, which is how A/V sync works (§4.5). There is an
   optional per-frame metadata trailer (`RXMT`), negotiated in SDP by
   `a=x-reactor-frame-metadata:1`.
6. **Keepalive**: any inbound data-channel message counts as a ping. The
   watchdog closes a connection after 20 s of silence (polled every 2 s). SDK
   clients send `Ping` on the control channel every 10 s.
7. **Model plug-in**: the model is a `ReactorApp` subclass with `load()`,
   `generate()`, and optional `process_input()`/`process_output()`. Commands
   are `set_<field>` for each public field on the `InputState`, plus
   `@event`-decorated methods. The model emits `(H, W, 3)` or `(N, H, W, 3)`
   uint8 RGB **numpy** arrays. RT converts them to BGRA and pushes them into
   libwebrtc, which does the encoding (software VP9/VP8/AV1, or OpenH264 on
   Linux; **no NVENC**).
8. **License**: Apache-2.0 for all three repos. The `.proto` files are copied
   verbatim to `docs/serve/reactor-proto/`, together with RT's LICENSE and
   NOTICE.

---

## 2. The `reactor_wire.v1` protobuf definitions (verbatim)

Source of truth: `RT:proto/reactor_wire/v1/*.proto`. Copies are in
`docs/serve/reactor-proto/reactor_wire/v1/` (byte-identical; checked with
`diff -r`). `RT:proto/README.md` explains how they are versioned. Bindings are
not committed. Each push to `main` that touches `proto/` publishes a CalVer
release `wire/v1.YYYYMMDD.<run>` (wheel, TypeScript tarball, and
`reactor-wire-<ver>-protos.tar.gz`). Compatibility is enforced by
`buf breaking`, and any breaking change must go into a new `reactor_wire.v2`
package. RT pins `[tool.reactor-wire] version = "1.20260814.7"`
(`RT:pyproject.toml`). SDK pins `1.20260722.6`
(`SDK:crates/reactor-protocol/proto/WIRE_VERSION`) and downloads the protos at
build time with `prost-build` (`SDK:crates/reactor-protocol/build.rs`, which
uses `.extern_path(".google.protobuf.Struct", "::prost_types::Struct")`).

### common.proto
```proto
syntax = "proto3";

package reactor_wire.v1;

// Framing shared by every channel message. A request expects a correlated
// response (matched by request_id); a notification is fire-and-forget.
enum MessageKind {
  MESSAGE_KIND_UNSPECIFIED = 0;
  MESSAGE_KIND_REQUEST = 1;
  MESSAGE_KIND_RESPONSE = 2;
  MESSAGE_KIND_NOTIFICATION = 3;
}

// Payload of a failed response, correlated to its request by request_id.
message Error {
  string code = 1;
  string message = 2;
}
```

### control.proto
```proto
syntax = "proto3";

package reactor_wire.v1;

import "reactor_wire/v1/common.proto";
import "reactor_wire/v1/platform.proto";
import "reactor_wire/v1/track.proto";

// Control channel, client to runtime. Carries all platform and track traffic.
message ControlClientMessage {
  string request_id = 1;
  MessageKind kind = 2;
  oneof payload {
    Ping ping = 10;
    RequestSchema request_schema = 11;
    FileUploaded file_uploaded = 12;
    RequestClip request_clip = 13;
    RequestRecording request_recording = 14;
    PublishTrack publish_track = 15;
    PauseTrack pause_track = 16;
    ResumeTrack resume_track = 17;
    UnpublishTrack unpublish_track = 18;
    Error error = 40;
  }
}

// Control channel, runtime to client. Carries all platform and track traffic.
message ControlServerMessage {
  string request_id = 1;
  MessageKind kind = 2;
  oneof payload {
    ModelSchema model_schema = 10;
    ClipReady clip_ready = 11;
    ClipFailed clip_failed = 12;
    Moderation moderation = 13;
    PublishTrackResponse publish_track = 14;
    SessionEnded session_ended = 15;
    Error error = 40;
  }
}
```

### data.proto
```proto
syntax = "proto3";

package reactor_wire.v1;

import "reactor_wire/v1/common.proto";
import "reactor_wire/v1/model.proto";

// Data channel, client to runtime. Carries only the model's own command
// traffic; all platform and track messages travel on the control channel.
message DataClientMessage {
  string request_id = 1;
  MessageKind kind = 2;
  oneof payload {
    Command command = 10;
    Error error = 40;
  }
}

// Data channel, runtime to client. Carries only the model's own emitted
// messages.
message DataServerMessage {
  string request_id = 1;
  MessageKind kind = 2;
  oneof payload {
    ModelMessage message = 10;
    Error error = 40;
  }
}
```

### model.proto
```proto
syntax = "proto3";

package reactor_wire.v1;

import "google/protobuf/struct.proto";

// Reference to a file uploaded out of band via the presigned-URL protocol,
// injected into a Command argument by parameter name.
message UploadReference {
  string upload_id = 1;
  string name = 2;
  string mime_type = 3;
  int64 size = 4;
}

// A model-defined command from the client. type is the command name; the
// arguments are a dynamic object so the protocol carries any model's schema
// without knowing it.
message Command {
  string type = 1;
  google.protobuf.Struct data = 2;
  map<string, UploadReference> uploads = 3;
}

// A model-emitted message to the client. type is the message name; the body is
// a dynamic object.
message ModelMessage {
  string type = 1;
  google.protobuf.Struct data = 2;
}
```

### platform.proto
```proto
syntax = "proto3";

package reactor_wire.v1;

import "google/protobuf/struct.proto";

// Liveness heartbeat from the client.
message Ping {}

// Request for the model's schema.
message RequestSchema {}

// The model's schema, carried as an OpenAPI document.
message ModelSchema {
  google.protobuf.Struct openapi = 1;
}

// Notification that a client upload completed and can now be referenced.
message FileUploaded {
  string upload_id = 1;
  string name = 2;
  string mime_type = 3;
  int64 size = 4;
}

// Request a clip of the last duration_seconds of output.
message RequestClip {
  double duration_seconds = 1;
}

// Request a recording of the session so far.
message RequestRecording {}

// A clip or recording is being prepared.
message ClipReady {
  string session_id = 1;
  string kind = 2;
  double start_marker = 3;
  double end_marker = 4;
  double now_marker = 5;
  int64 predicted_ready_at_ms = 6;
  string playlist_url = 7;
}

// A clip or recording request failed.
message ClipFailed {
  string reason = 1;
}

// A content-moderation verdict delivered to the client. A "terminate" action
// means the session is about to end; "warn" is informational and the session
// continues. Fields the sender cannot attribute are left empty.
message Moderation {
  string action = 1;
  string input_kind = 2;
  string command = 3;
  repeated string categories = 4;
  string message = 5;
}

// The platform is ending the session, and this is why. Sent best-effort
// before the connections close, so the client sees the reason rather than a
// bare disconnect. `reason` is the platform-authored, human-readable
// description of the cause, delivered verbatim for the client to display.
message SessionEnded {
  string reason = 1;
}
```

### track.proto
```proto
syntax = "proto3";

package reactor_wire.v1;

// Request to claim the single publisher slot for an input track.
message PublishTrack {
  string name = 1;
}

// Stop sending an input track without releasing the publisher slot.
message PauseTrack {
  string name = 1;
}

// Resume a paused input track.
message ResumeTrack {
  string name = 1;
}

// Release the publisher slot for an input track.
message UnpublishTrack {
  string name = 1;
}

// Successful response to a PublishTrack request. A failure is carried as an
// Error in the same response slot instead.
message PublishTrackResponse {
  string name = 1;
}
```

### 2.1 Message table (direction, channel, kind, and how RT handles each)

The "channel (v1)" column is the physical data channel. In v1 it equals the
logical channel, as shown by `logical_channel()` in
`RT:src/reactor_runtime/protocol/base.py`: `Data*Message` goes on `"data"` and
everything else on `"control"`. `V1Codec.encode` is `SerializeToString()` and
`decode` is `ParseFromString()` into the type keyed by (channel, direction)
(`RT:src/reactor_runtime/protocol/v1/codec.py`).

| Message / oneof | Dir | Channel (v1) | `kind` RT expects/sends | RT handling (source) |
|---|---|---|---|---|
| `ControlClientMessage.ping` (10) | C→S | control | NOTIFICATION | `keepalive()`, but any frame already counts as a ping (§5.4). `RT:message_gateway.py:_route_control` |
| `.request_schema` (11) | C→S | control | REQUEST | Reply `ControlServerMessage.model_schema` with the same `request_id`. Dropped if the model is not loaded. `RT:runner/runner.py:schema_requested` |
| `.file_uploaded` (12) | C→S | control | NOTIFICATION | Fetch the upload (10 s wait) and dispatch `@file_uploaded` only if the model declares that hook. `runner.py:file_uploaded` |
| `.request_clip` (13) | C→S | control | REQUEST | Reply `clip_ready` or `clip_failed`. RT mints a request_id if it is empty. `runner.py:clip_requested` |
| `.request_recording` (14) | C→S | control | REQUEST | Same as above, for the full recording. |
| `.publish_track` (15) | C→S | control | REQUEST | First come, first served per track name. Reply `publish_track` (an empty `PublishTrackResponse`, so **`name` is not filled** by `encode_publish_response`) or `error{code:"publish_refused", message:"track already published"}`. `RT:protocol/base.py:encode_publish_response`, `runner/connection_manager.py:publish_track` |
| `.pause_track` (16) | C→S | control | NOTIFICATION | Pauses that **outbound** track for this connection only. `connection_manager.py:pause_track` |
| `.resume_track` (17) | C→S | control | NOTIFICATION | Resumes that outbound track for this connection. Outbound tracks start paused (§4.3). |
| `.unpublish_track` (18) | C→S | control | NOTIFICATION | Releases the publisher slot if this connection holds it. |
| `.error` (40) | C→S | control | — | Decodes, but the gateway does not route it (logged at debug). |
| `ControlServerMessage.model_schema` (10) | S→C | control | RESPONSE | OpenAPI document as a `Struct` (§6.3). |
| `.clip_ready` (11) / `.clip_failed` (12) | S→C | control | RESPONSE | `encode_clip_ready` / `encode_clip_failed` |
| `.moderation` (13) | S→C | control | NOTIFICATION, empty request_id | Sent as the session enters CLOSING on a moderated stop, with `action="terminate"` and `message="Session terminated due to policy violation."`. `runner.py:_broadcast_moderation_notice` |
| `.publish_track` (14) | S→C | control | RESPONSE | See above. |
| `.session_ended` (15) | S→C | control | NOTIFICATION | Sent as CLOSING is entered when the stop carried a `reason`. The drain uses `"Session ended: the server is shutting down."`. `runner.py:_broadcast_session_ended`, `_DRAIN_CLOSE_REASON` |
| `.error` (40) | S→C | control | RESPONSE | A publish refusal. |
| `DataClientMessage.command` (10) | C→S | data | REQUEST (SDK) | Validated against the model contract, then dispatched. RT mints a `request_id` if it is empty (`message_gateway.py:_new_request_id`, uuid4 hex). |
| `DataServerMessage.message` (10) | S→C | data | RESPONSE if it answers a command (request_id set), otherwise NOTIFICATION | `Codec.encode_model_message` (`RT:protocol/base.py`) |
| `DataServerMessage` with **no payload** | S→C | data | RESPONSE | Bodyless ack for a command whose handler returned `None`. `encode_command_ack`. **Sent to v1 connections only.** (`connection_manager.py:send_command_ack` skips v0.) |
| `DataServerMessage.error` (40) | S→C | data | RESPONSE | Command failure. `code` is one of `invalid_command`, `unresolved_upload`, `internal_error`, or an author's `CommandError.code`. v1 only. (`RT:src/reactor_runtime/codes.py`, `runner.py:_reject_command`) |

Correlation ids in the SDK are opaque strings of the form `ctrl_<n>` and
`data_<n>` (`SDK:crates/reactor-core/src/control.rs`, `data.rs`). The SDK
treats an inbound `DataServerMessage` with an empty `request_id` as a broadcast
and a matching one as the reply. A correlated reply with a `message` payload is
also surfaced as an event (`data.rs` module docs).

`Struct` conversion: RT uses `json_format.MessageToDict` for Struct to dict,
and coerces non-string dict keys the way `json.dumps` does
(`RT:protocol/common.py`). **INFERRED:** all numbers therefore travel as
protobuf `double`, so large integers lose precision above 2^53.

### 2.2 The legacy v0 JSON encoding (must also be supported)

Source: `RT:src/reactor_runtime/protocol/v0/codec.py`. This is text JSON. It
differs from v1 in two ways. First, **channel placement**: every platform
message rides the **`"data"`** channel under `{"scope":"runtime"}`, and the
`"control"` channel carries only the track verbs. Second, **no correlation**
for commands: they are fire-and-forget.

Data channel envelopes:

```jsonc
// C→S command   (DataClientMessage.command)
{"scope":"application","data":{"type":"<cmd>","data":{...},"uploads":{"<param>":{"upload_id":"","name":"","mime_type":"","size":0}}}}
// S→C model msg (DataServerMessage.message)   — request_id is dropped in v0
{"scope":"application","data":{"type":"<msg>","data":{...}}}
// C→S platform  {"scope":"runtime","data":{"type":T,"data":{...}}}
//   T ∈ "ping" | "requestSchema" | "fileUploaded"{upload_id,name,mime_type,size}
//       | "requestClip"{duration_seconds} | "requestRecording"
// S→C platform  {"scope":"runtime","data":{"type":T,"data":{...}}}
//   T ∈ "modelSchema"(data = the OpenAPI doc itself) | "clipReady"{session_id,kind,start_marker,
//       end_marker,now_marker,predicted_ready_at_ms,playlist_url} | "clipFailed"{reason}
//       | "moderation"{action,input_kind,command,categories,message} | "sessionEnded"{reason}
```

Control channel (v0):

```jsonc
// C→S
{"type":"request","method":"publish_track","request_id":"…","data":{"name":"…"}}
{"type":"notification","event":"pause_track"|"resume_track"|"unpublish_track","data":{"name":"…"}}
// S→C
{"type":"response","method":"publish_track","request_id":"…","data":{}}
{"type":"response","method":"publish_track","request_id":"…","error":{"code":"…","message":"…"}}
```

Encoding selection (`RT:protocol/base.py:sniff`,
`RT:transport/webrtc/peer.py:_make_message_sink`,
`RT:transport/webrtc/version.py`). The `Reactor-WebRTC-Version` header on the
offer POST maps to a codec: `"1.0"`, absent, and unknown values all map to
**V0**. The peer is seeded with that codec. **The first inbound frame on
either channel is then sniffed and latched for the life of the peer.** A text
frame, or a binary frame whose first non-space byte is `{` or `[`, means v0;
anything else means v1. Outbound frames are sent as text if encoded to a
`str` (v0) and as binary if encoded to `bytes` (v1)
(`peer.py:_send_on`). **INFERRED consequence:** before a client's first
message the runtime would encode any unsolicited server message as v0. The
SDK's first action after the channels open is a v1 `Ping` or `ResumeTrack`
on `"control"`, which latches v1.

RT's docstring in `transport/webrtc/version.py` says "Every shipped client
speaks v0". The current SDKs contradict this: the Rust core used by all
language SDKs encodes v1 protobuf (`SDK:crates/reactor-core/src/control.rs`,
`data.rs`, `reactor.rs`). The JS SDK `3.0.2` loads that core as wasm
(`SDK:sdks/js/src/reactor.ts` imports `./internal/wasm`). The Python SDK
wraps `libreactor_ffi` through ctypes (`SDK:sdks/python/reactor_sdk/_ffi.py`).
v0 therefore matters only for older clients. **Recommendation:** implement v1
first and v0 as a compatibility codec behind the same sniff.

---

## 3. Session lifecycle (HTTP)

### 3.1 Two deployment shapes

- **Local / standalone runtime** (what RT serves, and what we replace):
  `POST /start_session`, `GET /session`, `POST /stop_session`. A process hosts
  exactly **one** session, whose id is fixed at
  `"00000000-0000-0000-0000-000000000000"` (`RT:runner/runner.py:SESSION_ID`).
  **There is no auth.** CORS is `allow_origins=["*"]` with all methods and
  headers (`RT:http/server.py:build_app`, whose comment reads "Auth rides the
  Authorization header, never a cookie"). The SDK's local mode sends no
  `Authorization` header (`SDK:crates/reactor-core/src/coordinator.rs:local_headers`).
- **Reactor cloud**: a coordinator at `https://api.reactor.inc` (from the
  default and doc strings in `SDK:crates/reactor-core/src/coordinator.rs` and
  `auth.rs`) sits in front of the runtimes. It exposes:
  - `POST /sessions` (body `CreateSessionRequest{model{name,version?}, client_info{sdk_version,sdk_type}, supported_transports[{protocol:"webrtc",version:"1.0"}], extra_args?, extra_configs?}`)
  - `GET /sessions/{id}`, polled until `capabilities` and `selected_transport` are present
  - `DELETE /sessions/{id}`
  - `POST /sessions/{id}/uploads`

  Auth is `Authorization: Bearer <jwt>`. The JWT is minted by
  `POST /tokens` with the header `Reactor-API-Key: <key>`. The body is
  `null`, or `{authorization_details:[{type:"session", resources:{models:{match:[…]}}, constraints:{max_sessions, max_session_duration_seconds}}], expires_after}`,
  and the response is `{jwt}` (`SDK:crates/reactor-core/src/auth.rs`). Every
  request carries `Reactor-API-Version: 1` and
  `Reactor-API-Accept-Version: 1` (`SDK:crates/reactor-protocol/src/lib.rs`).
  Cloud session states are
  `CREATED|PENDING|SUSPENDED|WAITING|ACTIVE|INACTIVE|CLOSED`, with
  INACTIVE and CLOSED terminal (`SDK:crates/reactor-protocol/src/session.rs`).

  **INFERRED:** in the cloud the WebRTC signalling URLs are
  `{api_url}/sessions/{id}/transport/webrtc/...`
  (`coordinator.rs:transport_base_url`). The coordinator ("director", in RT's
  comments) therefore proxies them to the runtime and presumably calls
  `/start_session` itself. RT also provides `/events` (SSE) and the
  `connection_answered` journal event, "so a consumer driving the runtime
  without polling (a director) can relay it back"
  (`RT:transport/webrtc/acceptor.py`). The coordinator itself is not open
  source.

### 3.2 Local session routes (`RT:src/reactor_runtime/http/routes.py`, `SessionRoutes`)

| Route | Request | Success | Errors |
|---|---|---|---|
| `POST /start_session` | optional JSON object, stored as the session `params` (the SDK sends `{"extra_args": …}`, `coordinator.rs:local_start_session`). A `session_id` key is adopted as the recording/log id (`runner.py:_recording_id_from`). | 200, descriptor (below) | 503 + `Retry-After: 1` while the model is loading (CREATED); 503 if TERMINATED; 409 from any other non-READY state. Body `{"detail":"cannot start session while <state>[: …]"}` |
| `GET /session` | — | 200, descriptor | — |
| `GET /schema` | — | 200, model OpenAPI (`{}` before load) | — |
| `POST /stop_session` | optional `{"moderate":bool=false,"reason":str(max 64)=""}` | 200, empty | 409/503 as above |

The descriptor (`runner.py:descriptor`) looks like this:

```json
{"session_id":"00000000-0000-0000-0000-000000000000",
 "state":"waiting",                      // lower-cased SessionState name (§3.4)
 "cluster":"local",
 "model":{"name":"<reactor.yaml model.name or class-derived>"},
 "server_info":{"server_version":"<reactor-runtime package version>"},
 "selected_transport":{"protocol":"webrtc","version":"1.0"},
 "recording":{"enabled":false,"chunk_seconds":4},
 "capabilities":{                        // only once the model is loaded
   "protocol_version":"v0",
   "tracks":[{"name":"main_video","kind":"video","direction":"recvonly"}],  // CLIENT perspective
   "commands":[]}}                       // always empty; commands come from /schema
```

The SDK considers a session ready when `capabilities` and
`selected_transport` are both present (`SDK:.../session.rs:is_ready`). SDK
`Capabilities` also has an optional `emission_fps`, which RT never sends.
**INFERRED:** RT's lower-case `state` deserialises to the SDK's
`SessionState::Unknown` through `#[serde(other)]`, which is harmless.

### 3.3 WebRTC signalling routes (`RT:src/reactor_runtime/transport/webrtc/router.py`)

Prefix: `/sessions/{sid}/transport/webrtc`. `{sid}` must equal the fixed
session id. Every route first calls `require_session_running`, which answers
**400** `{"detail":"No session running"}` unless the state is WAITING,
STREAMING or ORPHANED, and **404** `{"detail":"Unknown session"}` for the
wrong sid.

| # | Route | Body | Response |
|---|---|---|---|
| 1 | `GET /ice_servers` | — | `{"ice_servers":[{"uris":["stun:…"]},{"uris":["turn:…"],"credentials":{"username":"…","password":"…"}}]}` |
| 2 | `POST /connections` | — | **201** `{"connection_id":<int 1002..9999, random, unique per session>,"track_map":{"<name>":{"kind":"video","direction":"out","rate":0.0}}}` (the track_map is in **model** perspective). 503 `"No connection ids left"` |
| 3 | `POST /connections/{cid}/sdp_params` (first offer), `PUT` (re-offer/reconnect) | `SdpParamsRequest` (below). Header `Reactor-WebRTC-Version: 1.0` | **202** `{"connection_id":cid}`. 503 `"Connection limit reached"` (past `max_connections`=64, but re-offers are always admitted). 409 when a pinned port is taken. 422 on validation. |
| 4 | `GET /connections/{cid}/sdp_params` | — | **202** empty while negotiating; **200** `{"sdp_answer":"<sdp>","connection_id":cid}` once. The answer is *taken*, so a second GET returns 202 again (`acceptor.py:take_answer` pops it). |
| 5 | `POST /connections/{cid}/ice_candidates` | `{"candidates":[{"candidate":"candidate:…","sdp_mid":"0","sdp_mline_index":0}],"is_final":false}` | **202**. Candidates that arrive before the offer is negotiated are buffered (at most 128 connections × 256 candidates, `acceptor.py`). An empty `candidate` is end-of-candidates. `is_final` is accepted but ignored by RT. |

`SdpParamsRequest`:
```jsonc
{"sdp_offer":"v=0…",
 "track_mapping":[{"mid":"0","name":"main_video","kind":"video","direction":"recvonly"}],  // CLIENT perspective: recvonly = model OUT, sendonly = model IN
 "ice_servers":[{"uris":[…],"credentials":{…}}],   // optional; if present (even []) it overrides the server config for this connection
 "ice_credentials":{"ufrag":"4..256 ice-char","pwd":"22..256 ice-char"},  // optional, for relaying front-ends
 "port_range":[min,max]}                                                   // optional, inclusive; a single port pins
```

The SDK also sends `client_info` in the offer and ICE bodies
(`SDK:crates/reactor-protocol/src/webrtc.rs`). RT's pydantic models silently
ignore it. **INFERRED:** that is pydantic's default `extra="ignore"`.

Client flow (`SDK:crates/reactor-core/src/reactor.rs:finish_transport`,
`signaling.rs`, `backoff.rs`):

1. Register a connection, or reuse the existing id on reconnect.
2. Flush any locally gathered ICE candidates.
3. POST the offer (PUT when replacing).
4. Poll GET `sdp_params` with exponential backoff: 200 ms initial, ×2,
   capped at 15 s, 6 attempts (`PollConfig::sdp`).
5. `setRemoteDescription(answer)`.
6. Wait for "ready", meaning the peer is connected and both channels are open.

Server answer production (`RT:transport/webrtc/peer.py:_negotiate`):

1. `deduplicate_bundle_pts(offer)`, which removes Chrome RTX payload-type
   collisions (`RT:transport/webrtc/sdp.py`).
2. `set_remote_description`.
3. Attach a sender track to each OUT mid, set it SendOnly, and set codec
   preferences on every video transceiver.
4. `create_answer`, optionally substituting ICE credentials.
5. `set_local_description`.
6. Wait up to `ice_gathering_timeout_ms`=3000 for gathering to complete.
7. Embed all gathered candidates plus `a=end-of-candidates` in the answer
   (non-trickle on the server side, `sdp.py:embed_ice_candidates`).
8. Pause every OUT track.

**The server never trickles its own candidates.**

Timeouts. A connection must reach the connected wire within
`negotiation_timeout`=30 s of its offer, or it is closed and its slot freed
(`config.py`, `acceptor.py:_enforce_deadline`). A fresh offer on the same cid
cancels the in-flight negotiation (`acceptor.start_offer`). A wire that
connects after its session ended is refused: an epoch is stamped at offer
admission (`runner.py:connection_opened`, `runner/offer_epochs.py`).

ICE and TURN configuration come from the environment (`RT:serve.py`):
- `STUN_SERVERS`: comma-separated URLs.
- `TURN_SERVERS`: `user;cred;url`, comma-separated.
- With neither set, the default is `stun:stun.l.google.com:19302`.
- `WEBRTC_PORT_RANGE`: `min:max`.
- `ICE_TRANSPORT_POLICY`: `all|relay`.
- `WEBRTC_VIDEO_CODECS`.
- `WEBRTC_BWE_{MIN,INITIAL,MAX}_KBPS`: defaults 500, 4000, 10000.
- `WEBRTC_SENDER_{MIN,MAX}_KBPS`: defaults 0, 10000.
- `WEBRTC_CLIENT_PING_TIMEOUT_SECONDS`: default 20.
- `HOST`/`PORT`: defaults 0.0.0.0 and 8080.
- `ORPHAN_TIMEOUT_SECONDS`: default 60.
- `SIGTERM_GRACE_PERIOD`: default 30.
- `REACTOR_RECORDINGS_DIR`, `REACTOR_LOG_LEVEL`, `REACTOR_LOG_FORMAT`.

Other WebRtcConfig defaults (`RT:transport/webrtc/config.py`):
- `rtp_payload_mtu=1200`
- `rtx_max_size_packets=512` and `rtx_max_size_time_ms=200`
- transport-wide-cc header extension mirrored
- `ice_tcp=False`, `upnp=False`
- `warp=True`: SNAP (SCTP INIT mirrored, draft-hancke-tsvwg-snap) and SPED (DTLS in STUN, draft-hancke-webrtc-sped). These are optional accelerations that only apply when the peer offers them.

### 3.4 Session state machine and teardown

States (`RT:core/session.py`, `RT:runner/state_machine.py`):

- `CREATED` (model loading) → `READY` (on `initialization_success`)
- `READY` → `WAITING` (on `start_session`)
- `WAITING`/`ORPHANED` → `STREAMING` on the first `connection_opened`
- `STREAMING` → `ORPHANED` when the last connection closes
- `WAITING`/`ORPHANED` → `CLOSING` after `ORPHAN_TIMEOUT_SECONDS` (60 s)
- any running state → `CLOSING` on `stop_session`
- `CLOSING` → `READY` (`cleanup_complete`, after every connection is closed)
- any state → `TERMINATED` on `eviction` (model crash). This asks the process to exit.

`/health` maps these to `LOADING|AVAILABLE|SERVING|TERMINATED` and returns
503 only when TERMINATED. `EndReason` values: `stopped`, `timed_out`,
`evicted`, `moderated`, `error` (`RT:core/model.py`).

On entering CLOSING (`runner.py:_dispatch_transition`), RT:
1. broadcasts `moderation` or `session_ended` (these are queued on the ordered
   channels before the close);
2. clears uploads;
3. stops the recorder;
4. closes every connection;
5. sends `CLEANUP_COMPLETE`.

A connection lost involuntarily (peer state Disconnected, Failed or Closed,
or the ping watchdog firing) raises `CONNECTION_CLOSED`, which the model sees
as a `@disconnected` hook. The session stays up, and the client may
reconnect by PUT on the same cid (`router.py` module doc).

`GET /events` is a Server-Sent Events journal of every transition. Each event
is `id: <seq>\ndata: {"type":"transition","event":"…","from":"…","to":"…","ts":<ms>,"detail":{…}}`,
resumable through `?since=` or `Last-Event-ID` (`RT:http/events.py`,
`routes.py`). Journal-only events (`chunk_ready`, `clip_ready`, `command`,
`error`, `metric`) self-loop, with their payload in `detail`. This is a
director-facing surface; SDK clients do not consume it.

### 3.5 Uploads and clips (auxiliary, and used by image-seeded models)

- `POST /sessions/{sid}/uploads` `{name,size,mime_type,upload_id?}` returns
  **201** `{presigned_id, presigned_url:"<base>/uploads/<id>", path}`.
- `PUT /uploads/{id}` sends the raw bytes (`application/octet-stream`; the
  exact size is enforced).
- The client then either sends `FileUploaded` or references the upload in
  `Command.uploads[param]` (or inline as `{upload_id,…}` in an argument). RT
  waits up to 10 s for the bytes (`runner.py:_UPLOAD_RESOLVE_TIMEOUT_SECONDS`).
- `GET /clips?session_id&start&end` returns an HLS fMP4 manifest (200),
  "not ready" (202 with `Retry-After`), or "gone" (410). The segments are
  served at `GET /clips/chunks/{session_id}/{filename}`. Recording uses PyAV
  and libx264 (`yuv420p`, preset veryfast, CRF 23), is off by default, and is
  configured in `reactor.yaml` (`RT:recording/chunk_encoder.py`,
  `core/service.py:RecordingConfig`).

---

## 4. Media

### 4.1 Tracks and negotiation

- Tracks are declared by the model: fields annotated `Video`/`Audio` on an
  `Output` subclass become OUT tracks, and fields on a `MediaInput` subclass
  become IN tracks. The field name is the track name, for example
  `main_video` (`RT:interface/tracks/*.py`). `Audio.sample_rate` defaults to
  48 000.
- The client builds its transceivers from the descriptor's `capabilities.tracks`
  and sends the `mid`↔name mapping in `track_mapping`
  (`SDK:crates/reactor-core/src/peer.rs`). RT matches inbound tracks by
  arrival order within each kind (`peer.py:_on_track`/`_inbound_name`).
- Video codec preference, where the first one present in the offer wins
  (`RT:transport/webrtc/config.py:_DEFAULT_VIDEO_CODECS`): **VP9
  (`profile-id=0`) → VP8 → H264 (`profile-level-id=42e01f`,
  `packetization-mode=1`) → AV1 (`profile=0`) → H265**. Audio uses **Opus**.
  Encoding is done by libwebrtc inside `reactor-webrtc`. On Linux, H.264 is
  Cisco's prebuilt **OpenH264** (software), and hardware codecs are
  VideoToolbox on Apple only
  (`RW:crates/reactor-webrtc-sys/src/openh264.rs`). RT's
  `WebRtcConfig.hw_codecs_enabled` exists but is never read (a grep finds only
  its definition). **There is no NVENC path in RT.**
- Resolution and fps are **not negotiated**. They follow whatever the model
  emits: the pacer adopts the frame shape (a black-frame placeholder of
  720×1280 is used until the first frame arrives; `pacer.py:DEFAULT_FRAME_DIMENSIONS`).
  libwebrtc's congestion control adapts bitrate within the BWE and per-sender
  bounds. The per-sender max defaults to 10 Mbps specifically to lift
  libwebrtc's 2.5 Mbps default for frames above 960×540 (`config.py`
  docstring).

### 4.2 From model frame to wire (`RT:interface/internal/reactor_core.py`, `runner.py:_emit_media`, `transport/webrtc/pacer.py`, `peer.py`, `frames.py`)

1. The model returns an `Output`, with one payload per track. A video payload
   is a numpy uint8 array of shape `(H,W,3)` RGB, or a batch `(N,H,W,3)`. The
   Waypoint example emits `(4,720,1280,3)` per step, moved to the CPU with
   `.cpu().numpy()` (`RT:examples/waypoint/waypoint_model.py`). Audio is
   int16 or float in [-1,1], `(M,)`/`(1,M)`/`(C,M)`, mixed down to mono
   (`frames.py:to_int16_mono`).
2. `emit(output, compute_time=…)` wraps the payload in a
   `MediaChunk(bundle, fps, n_frames, wait)`. `fps` is
   `n_frames / generate_elapsed`, unless the app pins a class attribute `fps`
   (default 30). By default `wait=True`, which gives backpressure
   (`reactor_core.py:OutputStream.emit`; `reactor_app.py:run`).
3. The runner fans the chunk out to every connection's `MediaPacer`, and also
   to the recorder.
4. The pacer splits batches into single frames and queues them. The queue
   bound is `buffer_size` (default 10 frames, never less than one chunk). If
   the chunk has `wait`, the producer blocks; otherwise the tail is dropped.
   A dedicated thread emits one frame per `1/fps` tick. When the queue is
   empty, nothing is sent (the client holds its last frame), except that
   metadata-carrying video is repeated. A single **black frame** is sent at
   connection start and after `output.flush()`.
5. The peer's drain thread (queue of 10 bundles, overflow dropped) converts
   RGB to **BGRA** (`frames.py:rgb_to_bgra`; only the last frame of a 4-D
   array is used) and calls `track.push_video_frame(bgra, w, h, user_data=metadata)`.
   libwebrtc timestamps the frame on push and encodes it.
6. Audio goes through a separate feeder thread. It pushes exactly one 10 ms
   frame (480 samples at 48 kHz mono) per tick for the life of the track,
   substituting silence when the model has not supplied samples. The buffer
   caps at 200 ms. The long rationale on A/V clock integrity is in the
   `RT:transport/webrtc/peer.py` module docstring.

Per-frame metadata trailer (optional). This lives in
`RW:crates/reactor-webrtc/src/metadata.rs` and
`RW:docs/frame-metadata.md`. The trailer is appended to each **encoded**
video frame:

```
[encoded payload][FrameMetadata protobuf][u32 LE proto_len]["RXMT"]
```

```proto
message FrameMetadata { uint64 frame_id = 1; uint64 capture_time_us = 2; bytes user_data = 3; }
```

It is enabled only when the offer carries the session-level
`a=x-reactor-frame-metadata:1` and the answer mirrors it. RT fills
`user_data` with the JSON of `TrackPayload(metadata=…)`; Waypoint, for
example, sends `{"step": n}`. A replacement server can simply **not mirror
the attribute**, and SDKs will then receive no metadata but otherwise work.
**INFERRED:** this follows from the negotiation rule, since the gate stays
closed.

### 4.3 Subscription (the pause gate)

After negotiation every OUT track is paused. Its m-section is re-applied as
`inactive` by a local renegotiation that is never signalled to the client
(`peer.py:_apply_pauses`, which calls `set_media_direction` and
`bump_session_version`). The client's `ResumeTrack{name}` turns sending on
for that connection. The SDK does this automatically for every recvonly track
(`ReactorOptions.auto_resume_tracks = true`,
`SDK:crates/reactor-core/src/reactor.rs`). The SDK also flips its own
transceiver direction locally before sending Pause or Resume
(`reactor.rs:pause_track`/`resume_track`). **A replacement server must start
sending only after `ResumeTrack`**, or it must at least accept that message.
Sending immediately is not detectable as an error by the client. **INFERRED.**

### 4.4 Client input tracks

These are `sendonly` from the client, and cover things like a webcam or
microphone (see `RT:examples/echo`). Decoded frames arrive through libwebrtc
as BGRA and are converted to RGB `(H,W,3)` `InputFrame`s, with `metadata` and
`capture_time_us` taken from the trailer (`peer.py:_make_video_sink`). Audio
arrives as `(1,M)` int16 mono. The frames are pushed into per-track ring
buffers, which the model reads with `try_read()` or `read()`
(`RT:interface/internal/input_buffer.py`). Publishing is arbitrated per track
name through `PublishTrack`/`UnpublishTrack`, with one publisher at a time.

### 4.5 How a session declares its tracks, and how audio is carried and synced

**Declaring tracks: video+audio vs video-only.** The track set is whatever
the model's `Output` subclass declares. For example:

```python
class FastH3Output(Output):      # video + audio   (LS:fast-h3/fasth3_types.py)
    main_video: Video
    main_audio: Audio
class WaypointOutput(Output):    # video only      (RT:examples/waypoint/waypoint.py)
    main_video: Video
```

This declaration drives three things:

- The `x-reactor.tracks` entry in `/schema`. fast-h3's test pins this to
  `[("main_video","video","out"),("main_audio","audio","out")]`
  (`LS:fast-h3/tests/test_fasth3.py:test_the_model_publishes_two_outbound_tracks`).
- `capabilities.tracks` in `/start_session` and `/session`, with directions
  given from the client's side as `recvonly`.
- `track_map` in `POST /connections`.

The client creates one transceiver per advertised track and sends the
`mid`→name mapping with its offer. RT attaches a sender only to the mids that
map to OUT tracks (`peer.py:_attach_out_tracks`). A video-only session
therefore has no audio m-line in the answer. The audio feeder is a no-op in
that case, because `_push_audio_frame` returns unless the track is audio. The
track **names** are the model's field names, and clients subscribe by name:
the fast-h3 streaming client registers `reactor.track("main_video")` and
`reactor.track("main_audio")` before connecting
(`LS:streaming-client/reactor_link.py`). A Rust server must let each engine
declare its own list, for example:

- `[main_video, main_audio]` for H3 and LTX-2;
- `[main_video]` for Wan, FastWan and SF-Wan.

It must publish that list identically in all three places. **INFERRED:**
reusing the names `main_video`/`main_audio` keeps existing frontends
working.

**Audio format on the model side.** The model emits a numpy array, either
int16 or float in [-1,1], shaped `(M,)`, `(1,M)` or `(C,M)`. Multi-channel
audio is **mean-downmixed to mono** (`RT:transport/webrtc/frames.py:to_int16_mono`).
`Audio.sample_rate` defaults to 48 000, and a subclass may declare another
value (`RT:interface/tracks/descriptors.py`). However, **the outbound path
never resamples**. The peer hands every buffered sample to libwebrtc as
48 kHz mono (`track.push_pcm(payload, _AUDIO_SAMPLE_RATE=48_000, 1)` in
`RT:transport/webrtc/peer.py:_push_audio_frame`). The declared rate only
reaches the `rate` field of `track_map`. **INFERRED:** a model must therefore
emit 48 kHz itself. fast-h3 does exactly that: its checkpoint's audio decoder
produces 32 kHz stereo, which the backend resamples to 48 kHz with
`torchaudio.functional.resample`, averages to mono in float, and scales to
int16 `[1, samples]`. The backend comment explains the choice: "the transport
mean-downmixes before the wire anyway, and the runtime recorder flattens two
channels by concatenation, so a stereo emit only corrupts recordings"
(`LS:fast-h3/fasth3_backend.py:_to_wire_audio`, `OUTPUT_SAMPLE_RATE = 48_000`,
`NATIVE_SAMPLE_RATE = 32_000`).

**Codec.** libwebrtc negotiates **Opus**
(`RT:transport/webrtc/config.py:_DEFAULT_AUDIO_CODECS = ({"codec":"Opus"},)`).
RT sets no Opus fmtp or bitrate. The per-sender bitrate bounds apply only to
video, and a peer.py comment mentions "a 64 kbps Opus stream". Each outbound
audio track gets its own `LocalPush` source, so audio never crosses between
peers (`peer.py` module docstring, "Per-peer audio isolation"). On the
receiving side the Python SDK delivers audio frames as interleaved int16 plus
`(num_samples, sample_rate, channels)`
(`PyPI:reactor_sdk/client.py`, `_media.py:_pcm_to_array`). The fast-h3 client
warns if `sample_rate != 48000`
(`LS:streaming-client/reactor_link.py:_on_audio_frame`).

**A/V sync mechanism.** Neither stream carries a capture timestamp:
libwebrtc leaves `abs-capture-time` unoffered, and RT deliberately stamps
neither track (`peer.py` module docstring, "Audio/video sync"). Sync is
achieved **by co-transport and wall-clock feeding**:

1. The model emits video and audio for the same span of time in one `Output`.
   fast-h3 slices each clip into 3-frame emits, taking audio samples
   `[round(lo*2000), round(hi*2000))` (48000/24 = 2000 samples per frame).
   A test asserts `audio_samples == video_frames * 48000 / 24` for every emit
   (`LS:fast-h3/fasth3.py:_emit_clip`,
   `tests/test_fasth3.py:test_video_and_audio_stay_locked_slice_for_slice`).
2. `split_batch` splits a batched chunk into per-video-frame bundles and
   divides the audio **proportionally across the frames** with
   `np.array_split(audio, n_frames, axis=1)`. Each video frame therefore
   travels with its own slice of audio (`RT:core/values.py:split_batch`).
3. The pacer releases one bundle per `1/fps` tick. The peer's drain thread
   pushes the video frame to libwebrtc, which timestamps it on push, and
   appends that bundle's audio to a ≤200 ms per-track buffer.
4. A separate feeder thread pushes exactly one 10 ms / 480-sample frame per
   tick to each audio track. It uses the model's samples when a full frame is
   buffered and silence otherwise, so the audio RTP clock (a sample counter)
   stays locked to wall time. Silence is counted as under-production for up
   to 3 s (`_AUDIO_GRACE_TICKS=300`). A stall is repaid in at most 5 catch-up
   frames. Overflow beyond 200 ms drops the oldest samples.
5. Gap-fill video repeats (during underrun) never replay audio
   (`pacer.py:_video_only`). `output.flush()` drops queued bundles, audio
   included, and emits one black frame.

A pinned `fps` matters for audio models. fast-h3 pins `fps = 24` and never
passes `compute_time`. Its comment explains why: "Measuring instead
re-estimates the rate from observed timing, whose wobble both drops chunks
while converging and drifts video against the sample-clocked audio"
(`LS:fast-h3/fasth3.py`). It also sets `buffer_size = 48`, which is 2 s of
transport slack. **INFERRED:** in a Rust server, audio-bearing engines (H3,
LTX-2) should pace at the model's native fps and deliver exactly
`sr/fps` samples per frame. Video-only engines can use measured pacing.

---

## 4bis. Reference implementation with audio: fast-h3 (the queue-and-playout contract)

`LS:fast-h3/` is a production Reactor model with video and audio: it is
MiniMax-H3 distilled by FastVideo, served on 8×B200. Its client contract
lives entirely in `LS:fast-h3/fasth3_types.py`, and its handlers are in
`LS:fast-h3/fasth3.py`. Its reactor.yaml declares
`runtime.import: fasth3:FastH3` and `build.runtime_version: "3.2.6"`. That
version is wire-identical to 3.6.0 in every `reactor_wire.v1` field. I
checked by diffing the `*_FIELD_NUMBER` sets in the 3.2.6 wheel's
`reactor_wire/v1/*_pb2.pyi` against the protos. `protocol/base.py`,
`protocol/v0/codec.py`, `transport/webrtc/version.py` and `frames.py` are
byte-identical. 3.6.0 adds only the optional `ice_credentials`/`port_range`
offer fields and the 409 (`diff` of `transport/webrtc/router.py`,
`config.py`).

**Model shape.** fast-h3 subclasses **`ReactorModel`**, the 3.2.x name. In
3.6.0 this is a deprecated alias of `ReactorApp`
(`RT:interface/model/reactor_model.py`). It overrides `run()` itself instead
of using the step hooks, because its unit of work is a whole clip. The loop:

```python
async def run(self):
    while True:
        await self.connected.wait()   # generation gated on an audience
        await self._serve()           # pump builds (worker thread), play armed clips
```

The model's surfaces are:

- **Lifecycle hooks.** `@session_started` resets session state.
  `@session_ended` cancels builds and clears the queues. `@connected(client)`
  greets the joining client with `state_update` and `queue_update` via
  `client.send(...)`, which is addressed to that client and not correlated.
- **Replies and broadcasts.** `await self.send(msg)` broadcasts to everyone.
  A handler's return value is the correlated reply. Returning `None` gives a
  bodyless ack.
- **Media.** `self.emit(FastH3Output(main_video=uint8[N,H,W,3], main_audio=int16[1,S]))`
  sends a slice. `self.output.flush()` runs after each clip and on
  `stop`/`reset`, and holds the stream on black.
- **Weights.** `reactor_runtime.get_weights_path()` is also present in 3.6.0
  (`RT:paths.py`).

**Commands** (each is `@event`; the wire name is `Command.type`, and the args
travel in `Command.data`):

| command | args (InputField constraints) | correlated reply (`ModelMessage.type`) | also broadcasts |
|---|---|---|---|
| `enqueue` | `prompt: str` (≤800, moderate), `metadata: str` (≤2000, moderate, echoed opaque), `seed: int\|null` (≥0), `seconds: float\|null` (5.167–14.375, snapped to 17n+5 frames @24 fps), `position: int\|null` (≥0) | `clip_queued{clip}` | `queue_update`, `state_update` |
| `play` | `clip_id: str` (blank = playout front) | bodyless | `queue_update`, `state_update`, then `clip_started`, and later `clip_finished` or `clip_stopped` |
| `pop` | `clip_id` | `clip_popped{clip}` | `queue_update`, `state_update` |
| `move` | `clip_id`, `position: int` (0 = front) | `clip_moved{clip,queue,position}` | `queue_update` |
| `stop` | — | bodyless | `clip_stopped`, `state_update` |
| `get_queue` / `get_state` | — | `queue_update` / `state_update` | — |
| `set_clip_seconds` | `seconds: float` | `clip_length_accepted{clip_seconds,frames}` | `state_update` |
| `set_seed` | `seed: int ≥0` | `seed_accepted{seed}` | `state_update` |
| `set_autoplay` | `enabled: bool` | `autoplay_accepted{enabled}` | `state_update` |
| `set_canvas` | `aspect ∈ {16:9,1:1,9:16,4:3}` (the 16:9 canvas is 1344×768) | `canvas_accepted{aspect,width,height}` | `state_update` |
| `reset` | — | `session_reset{cleared_clips,was_playing}` | `queue_update`, `state_update` |

Message `type` strings are the snake_case of the `ModelMessage` class name
(`RT:interface/events/messages.py`: `cls.name = pascal_to_snake(cls.__name__)`).
So `StateUpdate` becomes `state_update` and `CommandError` becomes
`command_error`.

`ClipInfo` is embedded whole in every clip message as
`{clip_id(uuid), prompt, metadata, frames, seconds, seed, ready}`.
`state_update` carries these fields:

- `clip_seconds`, `clip_seconds_min`, `clip_seconds_max`
- `seed`, `autoplay`, `aspect`, `width`, `height`
- `playing`, `playing_clip_id|null`
- `generation_queued`, `generation_capacity`, `playout_queued`, `playout_capacity`
- `clips_played`, `seconds_sent`
- `valid_commands[]`, computed by `LS:fast-h3/fasth3_session_rules.py`

The **queue and playout semantics** work as follows (`LS:fast-h3/fasth3.py`,
`fasth3_queue.py`, `LS:skills/reactor-fast-h3-model/SKILL.md` §3):

- A clip moves generation queue (cap 20) → build (one at a time, front
  first, continuing while a clip plays) → playout queue (cap 10; builds pause
  while it is full) → playing. The playing clip is in neither queue.
- Nothing plays unless `play` is sent or autoplay is on. After each clip the
  model flushes to black and holds.
- **Refusals are broadcast, never raised.** A refused command broadcasts
  `command_error{command,reason}` and returns bodyless, "because a raised
  `CommandError`'s failure frame is withheld from older SDK generations",
  meaning v0 (§2.2; `connection_manager.py:send_command_ack` skips v0).

The Python SDK's `reactor.send_command(cmd, data)` returns the reply as a
`{"type","data"}` envelope, or `None` for a bodyless ack
(`LS:streaming-client/reactor_link.py:payload`, `model_link.py`).

Operational facts from LS:

- A local runtime hosts one session. A second client must join with
  `connect(session_id=…)`, which maps to `GET /session`.
- A plain connect is answered **409** while a session is streaming or
  orphaned. This is `/start_session` from a non-READY state (§3.2).
- A dead client leaves the session ORPHANED until the 60 s orphan timeout.
  The streaming client clears it with a bare `POST /stop_session`
  (`LS:skills/reactor-streaming-client/SKILL.md` §6).
- Frame handlers are registered by wire track name before connecting,
  because "querying the track list after connect races the session's track
  declaration".

**Why this matters for the Rust server.** Reproducing fast-h3's command set,
message names and track names on top of fastvideo-rs H3 or LTX-2 would make
the existing `streaming-client` (its `ReactorLink`) work unchanged. For
video-only Wan, the same contract applies with `generates_audio = False` and
no `main_audio` track. LS's `FastWanLink` already emulates this contract
client-side over FastVideo's HTTP job API (`LS:streaming-client/model_link.py`,
`fastwan_link.py`).

---

## 5. Commands, replies and telemetry

### 5.1 How commands are defined

Commands are model-defined. There is no fixed "prompt/seed/pause" set in the
protocol. They come from two places:
- **`InputState` fields**. Every public field `f` generates the command
  `set_f` with data `{"f": value}`, which assigns `self.state.f`
  (`RT:interface/app/reactor_app.py:_stamp_auto_setters`/`_make_setter`).
  Constraints come from `InputField(default, ge, le, choices, max_length,
  moderate, description)`.
- **`@event(name=…, description=…)` methods**. Their parameters become the
  command fields. The return value is either a `ModelMessage` (sent as the
  correlated reply), or `None`, which gives the bodyless ack on v1.
  `raise CommandError(code, message)` produces a correlated `Error`
  (`RT:interface/events/decorators.py`, `errors.py`).

Waypoint (`RT:examples/waypoint/waypoint.py`) is the closest analogue to a
Wan world model:

- `set_paused(bool)`
- `set_action(str ∈ idle|forward|back|left|right|forward_left|…)`
- `set_buttons(str)`
- `set_mouse_x(float)`, `set_mouse_y(float)`
- `set_scroll_wheel(int −1..1)`
- `set_image(UploadedFile)` (the seed image)
- `reset()`
- It broadcasts a `WaypointStatus{has_image,paused,step_index}` message.

The README's minimal app exposes `set_prompt` and `set_paused`
(`RT:README.md`).

### 5.2 Routing

1. The data-channel frame is decoded by `MessageGateway`
   (`RT:message_gateway.py`). Undecodable frames are dropped with a warning.
2. `Runner._submit_command` resolves uploads, then calls
   `ModelBridge.submit_command`, which validates against the contract. A
   rejection produces a journal `error` and, on v1, the reply
   `Error{code:"invalid_command"}`.
3. An accepted command is queued to the model's own asyncio loop (a separate
   thread). Handlers run **between steps, under the step lock**
   (`reactor_app.py:run` docs).
4. The model reads the new state on its next `process_input()`.

### 5.3 Server-to-client messages

- `ModelMessage{type,data}` broadcasts come from `await self.send(msg)`. They
  are sent **before** the step's media is emitted.
- Correlated replies, acks and errors are described in §2.1.
- Platform notices: `Moderation`, `SessionEnded`, `ClipReady`, `ClipFailed`,
  `ModelSchema`, `PublishTrackResponse`.
- **There is no server-sent latency or frame-timing telemetry in the
  protocol.** Every 2 s RT samples libwebrtc stats (packets, bytes,
  frames_sent, NACK, PLI/FIR, RTT, available outgoing bitrate) and exports
  them only as Prometheus metrics at `GET /metrics`
  (`RT:transport/webrtc/connection.py:_stats_loop`,
  `transport/webrtc/metrics.py`, `metrics/*.py`). SDKs compute client-side
  stats from the peer connection themselves (`SDK:crates/reactor-core/src/stats.rs`).

### 5.4 Heartbeat

The runtime fires `keepalive` on **every** inbound data-channel message, not
only on `Ping` (`peer.py:_make_message_sink` calls `_fire(self._cb_ping)`
after each message). The watchdog starts on connect, polls every 2 s, and
declares the connection lost after `ping_timeout` (20 s) without a message
(`connection.py:_watchdog_loop`). The SDK sends a `Ping` notification on
`"control"` immediately and then every `heartbeat_interval` = 10 s
(`SDK:crates/reactor-core/src/reactor.rs:run_heartbeat`).

---

## 6. Python runtime architecture (what is generic vs model-specific)

### 6.1 Layers

| Layer | Files | Model-agnostic? |
|---|---|---|
| HTTP ingress (FastAPI + uvicorn) | `http/server.py`, `http/routes.py`, `http/events.py` | yes |
| Transport router, acceptor, connection, peer, pacer, SDP helpers | `transport/router.py`, `transport/webrtc/*` | yes. SDP and ICE never leave `acceptor.py` |
| Wire codecs | `protocol/{base,common}.py`, `protocol/v0`, `protocol/v1` | yes |
| Runner: state machine, connection manager, gateway, uploads, recorder, metrics | `runner/*`, `message_gateway.py`, `upload_store.py`, `recording/*`, `metrics/*` | yes |
| Bridge | `interface/internal/bridge.py` (`ModelBridge.submit_command/dispatch_reactor_event/push_media/bind_outbound/start/stop`) | yes. It is the seam |
| Authoring surface | `interface/app/reactor_app.py` (`ReactorApp`), `interface/internal/reactor_core.py` (`ReactorCore`: own thread and asyncio loop, `emit`, `send`, `output.fps`, `output.flush()`, `buffer_size`), `interface/model/contract.py`/`schema.py` (contract to OpenAPI) | generic framework, subclassed per model |
| Model | e.g. `examples/waypoint/waypoint.py` + `waypoint_model.py`, named in `reactor.yaml` `runtime.import: module:Class` | model-specific |
| Multi-GPU helper | `distributed/*` (`DistributedRunner`: one process per GPU, lockstep, shared memory) | optional, standalone |

### 6.2 The model interface (`RT:interface/app/reactor_app.py`, `reactor_core.py`)

```python
class MyApp(ReactorApp):
    state: MyState                      # InputState -> set_<field> commands
    fps = 16.0                          # optional: pin playout; else measured from generate()
    buffer_size = 10                    # optional: frames queued per wire
    def load(self, config_path): ...    # once, off the event loop (asyncio.to_thread)
    async def process_input(self): ...  # under step lock; raise ApplicationError to skip
    def generate(self, input): ...      # sync, blocking GPU work; returns Output or author type
    async def process_output(self, outcome: StepOutcome): ...  # -> Output | None; re-raise = crash
    @event(name="reset") async def reset(self): ...
    @session_started / @session_ended / @connected / @disconnected / @file_uploaded hooks
    # or override `async def run(self)` and call self.emit()/self.send() yourself
```

There are two plug-in styles. The **step style** is `ReactorApp` with
`generate()`, as in Waypoint. The **own-loop style** overrides `run()` and
calls `emit`/`send`/`output.flush()` directly; fast-h3 uses this, under the
3.2.x name `ReactorModel`, which is now an alias of `ReactorApp` (§4bis). In
the own-loop style, command handlers run as concurrent coroutines on the
model's asyncio loop. `self.connected` (an `asyncio.Event`) gates work on
having an audience. Blocking GPU work must sit on a worker thread, as
`LS:fast-h3/fasth3_backend.py` does with `submit()` → `ClipJob`. The older
generator-based `ReactorPipeline` (one `yield` per chunk) is deprecated in
3.6.0 (`RT:src/reactor_runtime/__init__.py`).

In the step style, the loop runs steps only while a session is live **and**
at least one client is connected. A refused step (`ApplicationError`) sleeps 5 ms
(`_REFUSED_SLEEP`). A raise out of `process_output` evicts the session to
TERMINATED and the process exits (`runner.py:_on_model_failure`).

### 6.3 Model schema

`GET /schema` and `ModelSchema.openapi` carry an OpenAPI 3.1 document
(`RT:interface/model/schema.py:to_openapi`):

- each command is `paths["/events/<name>"].post` with a JSON body schema, and
  responds 200 with a `$ref` to its reply message, or 202;
- messages are listed under `webhooks`;
- `components.schemas` holds the message schemas plus `ReactorUploadReference`;
- tracks are in `x-reactor.tracks[{name,kind,direction:"out"|"in"}]` (model
  perspective);
- each field carries `x-reactor-moderate`.

A golden example is `RT:tests/unit/interface/app/golden/brightness_openapi.json`.

### 6.4 Media engine

`reactor_webrtc` (`RW`) is a PyO3 binding over a **Rust crate
`reactor-webrtc`** that wraps an owned libwebrtc build (milestone
`branch-heads/7907`, `RW:WEBRTC_VERSION`). **It is not aiortc, and not
GStreamer.** A comment in the SDK wasm peer mentions an earlier GStreamer
runtime (`SDK:crates/reactor-wasm/src/peer.rs` ~l.363). There is one
process-wide `PeerConnectionFactory` (`peer.py:_get_factory`).

---

## 7. Licensing

- **reactor-runtime: Apache-2.0**. `RT:LICENSE`; `RT:NOTICE` reads
  "Reactor Runtime / Copyright 2026 Reactor Technologies, Inc."; and
  `pyproject.toml` declares `license = "Apache-2.0"`. Copying the `.proto`
  files is permitted, provided the license and NOTICE travel with them. Both
  are copied to `docs/serve/reactor-proto/LICENSE` and `docs/serve/reactor-proto/NOTICE`.
- **reactor-client-sdks: Apache-2.0** (`SDK:LICENSE`).
- **reactor-webrtc: Apache-2.0** (`RW:LICENSE`, `RW:NOTICE.md`). The OpenH264
  binary it downloads has Cisco's binary license, which requires an
  attribution notice in the integrating app (`RW:crates/reactor-webrtc-sys/src/openh264.rs`).

---

## 8. Implications for the Rust server (fastvideo-rs `serve`)

These are recommendations. **All of this section is INFERRED.**

1. **Reuse, don't re-derive.**
   - Generate `reactor_wire.v1` with `prost-build` from the vendored protos,
     exactly as `SDK:crates/reactor-protocol/build.rs` does.
   - `reactor-protocol` (Apache-2.0) already defines serde types for the
     signalling JSON (`webrtc.rs`, `session.rs`, `upload.rs`). It is
     client-oriented, so for example `RegisterConnectionResponse` lacks
     `track_map`, but it can serve as the reference or be depended on.
   - The SDK's Rust core makes an ideal conformance-test client against our
     server.
2. **WebRTC stack choice.**
   - **`reactor-webrtc`** (a Rust crate on crates.io, the same libwebrtc RT
     uses) gives byte-level behavioural parity: the codecs, the
     `RXMT` metadata trailer, SNAP/SPED, and BWE. It accepts BGRA pushes, so
     it encodes in software. It also accepts **pre-encoded** frames through
     `EncodedVideoTrack` (`RW:docs/frame-metadata.md` mentions
     `EncodedVideoTrack::push_frame_with_metadata`), which would let us feed
     **NVENC H.264/AV1 bitstreams** from the CUDA engine and skip the
     GPU→CPU→BGRA round trip. This needs verification: check
     `RW:crates/reactor-webrtc/src/encoded.rs`.
   - **str0m** or **webrtc-rs** require our own encoder (NVENC via
     `nvidia-video-codec-sdk` or ffmpeg) and our own packetization. Only
     H.264 or VP8/VP9/AV1 payloaders that the client's browser accepts can be
     offered. The metadata trailer can be omitted by not mirroring
     `a=x-reactor-frame-metadata`.
3. **Minimum compatible surface** for current SDK clients (v1):
   - `POST /start_session`, `GET /session` (the descriptor with
     `capabilities` and `selected_transport`), `POST /stop_session`,
     `GET /schema`.
   - The 5 signalling routes, with async answer via 202 then 200, and the
     answer carrying embedded candidates.
   - Accept the `"data"` and `"control"` channels the client creates; do not
     create them server-side.
   - Sniff the encoding from the first frame.
   - Handle `Ping`, `ResumeTrack`/`PauseTrack`, `RequestSchema`, `Command`.
   - Reply with an ack or error on the data channel for each correlated
     command.
   - Enforce the 20 s liveness watchdog.
   - Pace at the engine's measured fps, keeping a black-frame boundary on
     reset.
4. **Map the fastvideo-rs engines to the command model.**
   - For clip models (H3, LTX-2, bidirectional Wan/FastWan), reproduce the
     fast-h3 queue-and-playout contract of §4bis verbatim, so the existing
     `streaming-client` works unchanged.
   - For streaming causal models (SF-Wan), use `InputState`-style setters
     (`set_prompt`, `set_paused`, action/camera fields) plus `reset`, as in
     Waypoint.
   - Publish all of them in the OpenAPI shape of §6.3.
   - Track list per engine: `[main_video, main_audio]` when audio is
     generated, and `[main_video]` otherwise (§4.5).
   - Audio: 48 kHz mono int16, exactly `48000/fps` samples per video frame,
     carried in the same emit, with a pinned fps.
   - Video-only causal engines may pace at `N / step_time`, which is RT's
     default.

---

## 9. Gaps and open questions

- **The cloud coordinator** (`api.reactor.inc`) is closed source. Its exact
  proxying of `/sessions/{id}/transport/webrtc/*`, how it calls
  `/start_session`, and its `SUSPENDED/PENDING` semantics are known only from
  the client's side (§3.1). A drop-in replacement *inside* Reactor's cloud
  would need their director contract. A standalone server needs only the
  local routes.
- The RT docstring claims shipped clients speak v0, but the current SDK core
  speaks v1. It is unclear which deployed clients (the Sandbox, older npm
  versions) still use v0. Support both.
- `PublishTrackResponse.name` is never filled by RT, and clients don't rely
  on it (`SDK reactor.rs` only matches the variant).
- `emission_fps` exists in SDK `Capabilities` but RT never sends it.
- The GStreamer-era runtime and its SDP quirks (referenced in
  `SDK:crates/reactor-wasm/src/peer.rs`) were not researched.
- I did not verify `reactor-webrtc`'s encoded-frame input path for H.264 or
  AV1 from NVENC, or its Linux hardware-codec status beyond the OpenH264
  comment.
- `docs.reactor.inc` was not consulted, because the source was sufficient.
- **Audio at a declared rate other than 48 kHz.** I found no resampling in
  RT's outbound path. **INFERRED:** a non-48k emit would play at the wrong
  speed. I did not test this.
- **The Opus parameters** (bitrate, stereo, DTX, FEC) are libwebrtc defaults
  and were not inspected. The LS client only observes 48 kHz.
- **The Python SDK wheel** (`reactor_sdk 1.6.0`) wraps the same Rust core
  (`libreactor_ffi.so`). Its on-wire behaviour is therefore the SDK repo's,
  and I did not re-verify it separately.
