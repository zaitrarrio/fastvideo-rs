# Stream outputs: Cloudflare Stream by default, WHIP, RTMPS, SRT

**Status (2026-10-13).** Built on `claude/stream-outputs` (PR #93) and
tested on the fake engine against a fake Cloudflare Stream API, a local
MediaMTX and, since the owner enabled Stream for the API token, **the real
Cloudflare Stream** (branch `claude/stream-outputs-live`): every kind plays
there ([results](#results-cloudflare-live-2026-10-13)). WHIP H.264, VP8 and
VP9 come back over WHEP at the native 1344x768; RTMPS and SRT come back as
HLS/DASH, whose top rendition Cloudflare transcodes to 1260x720. RTMPS now
verifies the server's certificate and name ([RTMPS TLS](#rtmps-tls)).

Owner request (2026-10-13): fix WHIP for Cloudflare; RTMPS and SRT outputs;
an optional output copy of director and Reactor sessions; view the stream in
the console by its output format; let the caller name a destination and its
credentials; Cloudflare as the default.

## What a stream can publish to

| `output.kind` | destination | transport | encode | playback |
|---|---|---|---|---|
| `cloudflare` (**default**) | a Cloudflare Stream live input the server creates per stream (`fv-<stream id>`) | WHIP (`webRTC.url`) | H.264 Constrained Baseline, or VP8 / VP9 by `codec`; Opus 48 kHz stereo | WHEP (`webRTCPlayback.url`) |
| `cloudflare-rtmps` | the same, RTMPS ingest | ffmpeg FLV, TLS by fv-serve ([verified](#rtmps-tls)) | H.264 + AAC-LC (ffmpeg sink) | HLS / DASH (the input records) |
| `cloudflare-srt` | the same, SRT ingest | ffmpeg MPEG-TS over `srt://` | H.264 + AAC-LC | HLS / DASH |
| `whip` | any WHIP endpoint (`url`, `token` or `user`) | WHIP | as `cloudflare` | none known: the client gives its WHEP URL |
| `rtmps` (alias `rtmp`) | any `rtmp://` / `rtmps://` ingest (`url` + `stream_key`) | ffmpeg FLV (`rtmps://`: TLS by fv-serve, verified) | H.264 + AAC-LC | none known: the client gives an HLS URL |
| `srt` | any `srt://host:port` listener (`stream_id`, `passphrase`) | ffmpeg MPEG-TS | H.264 + AAC-LC | none known |

Every kind takes the model's **native canvas and fps** (H3 1344x768 and its
1080p tier, LTX, Wan sizes). Native streams (`POST /fv/v1/streams`) publish
to one output; director and Reactor sessions can add one output *copy*
([below](#director-and-reactor-output-copies)).

Code: `crates/fastvideo-serve/src/stream_outputs/` (`spec.rs` request model,
redaction, URL policy; `cloudflare.rs` Live Inputs client; `publish.rs` the
publisher; `rtmps.rs` the verifying TLS relay), `crates/fastvideo-webrtc/src/whip.rs` (WHIP client, codec
choice), `crates/fastvideo-media/src/sink.rs` (RTMP(S) / SRT sink),
`crates/fastvideo-media/src/output.rs` (the session tap).

## WHIP for Cloudflare (what changed)

- **Native resolution.** The old profile capped Cloudflare at 1280x720
  (1344x768 scaled to 1260x720 and padded) because the offer says level 3.1.
  Cloudflare documents no resolution cap for WHIP (its FFmpeg example is
  1920x1080@30 Constrained Baseline), and an offer's level is what the
  offerer can *receive*: browsers offer `42e01f` and send 1080p. We now offer
  `profile-level-id=42e01f` (Cloudflare's documented value) and encode the
  native canvas at the level it needs (1344x768@24: 3.2, 1920x1088@30: 4.0;
  `H264Config::for_publish`, never below the offered level). MediaMTX keeps
  `42e028`. The live run confirmed it: Cloudflare accepted the 1344x768
  level-3.2 stream and delivered it unscaled over WHEP.
- **GOP.** 1 s for the Cloudflare target (Cloudflare recommends GOP 30 at
  30 fps for WebRTC), 2 s elsewhere; no B-frames; CBR. (The live run found
  the H.264 path still at 2 s, unlike VP8/VP9; `whip_h264` now sets 1 s.)
- **Codec choice.** `output.codec`: `h264` (default), `vp8` or `vp9`. Exactly
  one codec is offered (Cloudflare's WHIP takes all three). VP8 and VP9 run
  ffmpeg `libvpx` / `libvpx-vp9` in real-time mode through the same pipe
  encoder (`Vp8Config::vp9`, profile 0, row-mt, no alt-ref and no lag, so no
  superframes); str0m packetizes VP9 in non-flexible mode. The old claim
  that "Cloudflare plays VP8 black" is false: the live run decoded every
  VP8 frame Cloudflare delivered at 1344x768 with the same picture as H.264
  (mean luma 131.8, deviation 53.8, no black frame); VP9 too.
- **Trickle ICE: not needed.** The offer is complete (host candidates and a
  STUN srflx candidate, `a=end-of-candidates`), which Cloudflare and
  MediaMTX accept; no PATCH is sent.
- **Errors never carry the URL.** A Cloudflare WHIP URL holds the publish
  key in its path; reqwest's error text names the URL, so the WHIP client
  drops it (`without_url`) and the publisher scrubs every transport error.

## API

### `POST /fv/v1/streams`

```json
{
  "model": "sf-wan",
  "prompt": "a coastal road at golden hour",
  "output": {
    "kind": "cloudflare" | "cloudflare-rtmps" | "cloudflare-srt" | "whip" | "rtmps" | "srt",
    "url": "…",            // whip / rtmps / srt only
    "token": "…",          // whip: Bearer (Basic password with "user")
    "user": "…",           // whip: Basic auth user (MediaMTX)
    "stream_key": "…",     // rtmps: appended as the URL's last path segment
    "stream_id": "…",      // srt: streamid (MediaMTX: "publish:<path>")
    "passphrase": "…",     // srt: 10-79 characters
    "codec": "h264" | "vp8" | "vp9",          // WHIP kinds only
    "target": "cloudflare" | "mediamtx",      // whip: the offered level (default: from the URL)
    "insecure_skip_verify": false             // rtmps:// only, tests only (needs allow_private)
  }
}
```

- **No `output`:** the server's default, `{"kind": "cloudflare"}`, when it
  has a Cloudflare account; otherwise 400 `param: "output"`.
- **Legacy fields** `whip_url`, `whip_token`, `whip_user`, `whip_target` are
  an `output` of kind `whip` (errors name the legacy field). `output` and
  `whip_url` together are refused. The server default WHIP token
  (`FV_WHIP_TOKEN`) still applies to kind `whip` without a `token`.
- **Validation** (400 with `param`): the kind's URL scheme (`http(s)` for
  WHIP, `rtmp`/`rtmps`, `srt` with a port); fields a kind does not take
  (`stream_key` on an SRT output, a `url` on a Cloudflare kind, `codec` other
  than `h264` on RTMPS/SRT); an RTMP URL needs a stream key (field or last
  path segment); SRT passphrase 10-79 characters, and SRT values without
  spaces, `&`, `#`, `?` or `%` (ffmpeg takes them verbatim from the URL
  query: its libsrt does not percent-decode); `insecure_skip_verify` other
  than on an `rtmps://` URL, or without `allow_private`; unknown fields.
  Messages name fields and rules, never values.
- **Private and loopback destinations are refused** (loopback, RFC 1918,
  link-local, CGNAT, ULA, documentation ranges, `localhost`, `*.local`,
  `*.internal`; IP literals in `srt://` / `rtmp://` too), also when the name
  resolves to such an address at start, unless `[outputs] allow_private`
  (`FV_OUTPUTS_ALLOW_PRIVATE=1`: MediaMTX tests, LAN relays). The name is
  resolved **once**: RTMP, RTMPS and SRT connect to the addresses that were
  checked (ffmpeg gets the IP, and RTMP announces the real `tcUrl`), so a
  DNS change after the check cannot redirect the stream. Cloudflare's
  ingest hosts come from its API: resolved and pinned the same way, but not
  held to the private-address policy (an ingest that does not resolve fails
  the request with a 500 and the live input is released).

The **201 stream object** adds:

```json
"output": {"kind": "rtmps", "url": "rtmps://live.example.com:443/live", "credentials": "set", "playback_format": "hls"},
"playback": {"provider": "cloudflare", "live_input_id": "66be4b…", "format": "whep",
             "whep_url": "https://customer-<code>.cloudflarestream.com/<playback id>/webRTC/play",
             "iframe_url": "https://customer-<code>.cloudflarestream.com/<uid>/iframe"}
```

`playback` for RTMPS/SRT inputs has `hls_url` (`…/<uid>/manifest/video.m3u8`)
and `dash_url` (`…/video.mpd`) instead of `whep_url`; for your own
destinations it is `{}`. `status` carries the publisher's counters next to
the stream's: `output_state` (`provisioning`, `waiting`, `connecting`,
`publishing`, `ended`, `failed`), `output_error` (scrubbed), `video_codec`,
`encoder`, `width`, `height`, `frames_sent`, `keyframes_sent`,
`audio_packets_sent`, `forced_idrs`, `keyframe_requests(_covered)`,
`ticks_dropped`, `sink_restarts`, and `whip_resource` (redacted).

A destination that fails **while** streaming (an RTMP server down, a bad
key) is reported in `output_error` and retried (the ffmpeg sink restarts
every 2 s); the stream keeps running. A WHIP peer that closes ends the
stream with `end_reason: "error"`.

### `GET /fv/v1/outputs`

What the server publishes to: `{"default": "cloudflare" | null, "kinds":
[…], "codecs": {"whip": ["h264", "vp8", "vp9"], "rtmps": ["h264"], "srt":
["h264"]}, "cloudflare": {"recording", "on_stop"} | null, "allow_private",
"publisher"}`. The console reads it to grey out the Cloudflare kinds on a
server without an account.

## The Cloudflare default

Configuration (`[outputs.cloudflare]`, environment wins):

| setting | env | default | |
|---|---|---|---|
| `account_id` | `FV_CF_ACCOUNT_ID` (also D1's) | – | both id and token make Cloudflare the default |
| `api_token` | `FV_CF_STREAM_API_TOKEN` (**secret**) | – | needs **Account > Stream > Edit** |
| `api_base` | `FV_CF_STREAM_API_BASE` | `https://api.cloudflare.com/client/v4` | tests point it at a fake |
| `name_prefix` | `FV_CF_STREAM_NAME_PREFIX` | `fv-` | tests use `fv-test-` |
| `recording` | – | `auto` | `auto`: off for WebRTC inputs, `automatic` for RTMPS/SRT (Cloudflare serves HLS/DASH of a live input only while it records); `off`; `automatic` |
| `on_stop` | – | `auto` | `auto`: delete the input, or disable it (`PUT {"enabled": false}`) when it recorded so the recording stays reachable; `delete`; `disable`; `keep` |
| `delete_recording_after_days` | – | none | `deleteRecordingAfterDays` of recording inputs (Cloudflare's minimum is 30) |
| `customer_domain` | `FV_CF_STREAM_CUSTOMER_DOMAIN` | from the API | `customer-<code>.cloudflarestream.com`; the API's WebRTC URLs name it |

Per stream: `POST /accounts/{id}/stream/live_inputs` `{"meta": {"name":
"fv-<stream id>"}, "recording": {"mode": …}}` before the 201 (a refusal is a
500 that names the missing permission, and the engine session is released);
publish to the input's WebRTC, RTMPS (`url` + `streamKey`) or SRT (`url`,
`streamId`, `passphrase`) ingest; on stop, release it per `on_stop`.

Cost (Cloudflare pricing page, 2026-10): ingest and encoding are free;
delivery is $1 per 1,000 minutes viewed (WebRTC delivery billed from
2026-10-15); recordings count as stored minutes, prepaid at $5 per 1,000
minutes a month. A WebRTC stream with no viewers costs nothing; an RTMPS/SRT
stream records (for HLS) and its recording is stored until deleted.

### Cloudflare account (owner setup: done 2026-10-13)

First checked 2026-10-13: the token managed Workers but every Stream call
answered `10000 Authentication error`. The owner then did steps 1-3:

1. ~~Dashboard → **Stream**: subscribe if it offers a plan~~ **done.**
   Stream bills $5 per 1,000 minutes of storage a month (prepaid, in $5
   steps) and $1 per 1,000 minutes delivered; WebRTC outputs store nothing.
2. ~~**My Profile → API Tokens**: add **Account → Stream → Edit** for
   account 0cd06fd37ee4e07de370821cb3852a8a~~ **done** (the token in
   `/root/.config/fv/cf_api_token` lists live inputs).
3. ~~Put the token in fv-serve's environment as `FV_CF_STREAM_API_TOKEN`~~
   **done:** the Runpod secret `fv_cf_stream_api_token` exists (owner
   approval 2026-10-13). Pods and endpoints take it as
   `"FV_CF_STREAM_API_TOKEN": "{{ RUNPOD_SECRET_fv_cf_stream_api_token }}"`
   ([fv-control](#fv-control)); `FV_CF_ACCOUNT_ID` is already set on every
   cluster and endpoint (`fv_cf_account_id`).
4. ~~Run the live test~~ **done** ([results](#results-cloudflare-live-2026-10-13)).

The account has one live input that is not fastvideo's
(`divine-firefly-3665`); tests only create and delete `fv-test-` inputs.

## Secrets

- Credentials (`token`, `stream_key`, `stream_id`, `passphrase`, a URL's
  userinfo, query and RTMP key segment, the Cloudflare API token and ingest
  keys) live only in `OutputSpec` / `Provisioned`, whose `Debug` print the
  redacted view. Responses, events, logs and status show `url` without
  credentials (userinfo, query and fragment dropped; an RTMP key segment and
  any long opaque path segment such as Cloudflare's WHIP key become
  `<key>`) and `"credentials": "set"`.
- Errors: validation messages name fields, never values; ffmpeg's stderr is
  scrubbed of every secret part before it is kept (`SinkTarget::secrets`,
  `scrub`), and the publisher scrubs transport errors once more.
- Nothing is stored: a stream's entry holds the redacted view and the
  counters; the credentials live in the running task and go with it.
  Metrics carry no output field.
- The console keeps credentials in password inputs only (never
  localStorage); the director page's log shows `x_fv_output` redacted.
- fv-control seals `FV_CF_STREAM_API_TOKEN` (refused as a plain value).

## RTMPS and SRT sinks

`fastvideo_media::sink::FfmpegSink` (one ffmpeg, RGB frames on stdin, PCM
on a loopback socket; a video-only session gets silent AAC):

- `SinkTarget::Rtmp { url }` covers `rtmp://` and `rtmps://` (FLV,
  `-flvflags no_duration_filesize`); `SinkTarget::Srt { url }` is MPEG-TS
  (`-mpegts_flags +resend_headers`) over `srt://host:port?streamid=…&
  passphrase=…&latency=<µs>&mode=caller&pkt_size=1316` (`srt_url`;
  `[outputs] srt_latency_ms`, default 1000).
- Cloudflare Stream Live's input rules: H.264 Constrained Baseline (NVENC;
  libx264 on CPU builds), **CBR clamped to 2-13 Mb/s** video (+128 kb/s AAC
  is within 2-14 Mb/s; libx264 `nal-hrd=cbr`), **closed 2 s GOP** (IDR
  only, `-flags +cgop`, no B-frames, no scene-cut keyframes), **AAC-LC
  48 kHz stereo** (ADTS in MPEG-TS).
- A dead ffmpeg is restarted at most every 2 s; `output_error` is its exit
  status and last stderr lines (scrubbed), read after its stderr reached
  EOF (`FfmpegSink::exit_error`; before, a race left only "ffmpeg exited").
- RTMP and SRT connect to the address checked at start (above); RTMP passes
  the server's own URL as `-rtmp_tcurl`.

### RTMPS TLS

RTMPS **verifies the server's certificate chain and host name** and never
falls back to an unverified connection (`stream_outputs/rtmps.rs`):

- ffmpeg's `rtmps://` opens TLS as a nested protocol that verifies nothing
  by default. Whether FFmpeg 4.4 passes `-tls_verify` / `-ca_file` down to
  it, and whether its OpenSSL backend then checks the host name, varies by
  version and could not be tested here (the build pod's ffmpeg is a
  different build), so fastvideo does the TLS itself. Options weighed: (a)
  bump the image's ffmpeg to a release whose rtmps passes TLS options
  through: a new ffmpeg build and size audit, and still ffmpeg's TLS
  (hostname checking differs by backend); (b) **chosen**: a rustls client
  in fv-serve. (b) works with every ffmpeg (the image's 4.4, the build
  pod's), needs no image change, checks the name for certain, and reuses
  crates already in the tree (rustls 0.23 with aws-lc-rs, tokio-rustls,
  rustls-native-certs).
- How: the publisher starts an `RtmpsRelay` on `127.0.0.1:<port>`; ffmpeg
  publishes plain RTMP to `rtmp://127.0.0.1:<port>/<app>/<key>` with
  `-rtmp_tcurl rtmps://<host>:<port>/<app>`; each connection the relay
  accepts (loopback only) is carried over a rustls session to the
  addresses resolved and checked at start, with SNI and the certificate
  checked against the URL's host. The stream key never leaves the TLS
  session.
- Trust: the system's CA certificates (`ca-certificates` in every runtime
  image, checked in `base-os`; `SSL_CERT_FILE` / `SSL_CERT_DIR` honoured)
  plus `[outputs] rtmps_ca_file` (`FV_OUTPUTS_RTMPS_CA_FILE`, PEM: a
  private CA). No CA certificates at all fails the output.
- A verification failure (unknown issuer, expired, wrong name, no
  certificate) is **fatal**: the relay refuses every later connection, the
  publisher ends the output (`output_state: "failed"`, `output_error`:
  "RTMPS: the TLS certificate of <host>:<port> failed verification
  (UnknownIssuer); the output stops rather than connect unverified"), and
  a director/Reactor copy reports `output_error{fatal: true}`. Network
  errors (refused, timed out) stay retryable like any sink failure.
- `insecure_skip_verify: true` skips the check, only on an `rtmps://` URL
  and only with `[outputs] allow_private` (tests against a self-signed
  MediaMTX); the output view shows `"tls_verify": false`.

### The runtime ffmpeg (docker/gpucheck.Dockerfile)

(fv-serve no longer uses ffmpeg's `rtmps` protocol: [RTMPS TLS](#rtmps-tls).
`base-os` also checks that the CA bundle the relay trusts is there.)

FFmpeg 4.4.5 now has `--enable-openssl --enable-version3 --enable-libsrt`:
TLS through Ubuntu's shared libssl3 (OpenSSL 3, Apache-2.0, so the build is
GPLv3) and SRT 1.5.4 built from source as a static library (its crypto is
the same libcrypto). The `base-os` stage fails unless `ffmpeg -protocols`
lists `rtmps` and `srt`, `libvpx-vp9` encodes, and `ldd` resolves every
library of ffmpeg and ffprobe. Measured on GHCR (compressed): base-os
52.3 → **55.4 MB** (libssl3 +2.3 MB, ffmpeg +0.8 MB), so every runtime
image grows by 3.1 MB; the `cpu` image 69.5 → 74.5 MB (also fv-serve +2.0
MB). Linking OpenSSL statically would put it into both ffmpeg and ffprobe.
docs/serve/images.md has the table.

## Director and Reactor output copies

A live session can publish a **second encode** of its paced frames and
audio to an output while its WebRTC peer keeps playing
(`fastvideo_media::output`): the session pushes every tick into an
`OutputTap` (non-blocking, ~2 s deep, drops the newest when full and asks
for a keyframe after a gap); the copy runs fv-serve's publisher on its own
task. A refused request refuses the session's start or configure; anything
after that is an event, and **a copy's failure never stops the session**.
The copy ends with the session (its Cloudflare live input released).

**fal director.** `configure.x_fv_output` takes the `output` object above.
fal's schema forbids extra fields, so `configure.output` is accepted only
with `[director] allow_output = true` (`FV_DIRECTOR_ALLOW_OUTPUT=1`);
otherwise it is refused as `invalid_message`, like any unknown field.
`x_fv_output` always works when the server can publish. Messages (fv
extensions, sent only to sessions that asked):

| message | when | fields |
|---|---|---|
| `session_info.x_fv_output` | channel open | `{available, fields, server_message_types, destinations}` (`GET /fv/v1/outputs`) |
| `configured.x_fv_output` | configure accepted | the redacted output |
| `output_ready` | publishing | `{output, playback}` |
| `output_error` | a failure | `{output, message, fatal}`; `fatal`: the copy ended |
| `output_ended` | the copy ended | `{output, reason, frames_sent}` |

A bad output (`x_fv_output: …`) answers `error{code: "invalid_input"}` and
the session stays unconfigured.

**Reactor.** `POST /start_session` takes `output` or `x_fv_output` (400 when
refused). The session descriptor (`GET /session`, the `start_session` reply)
carries `x_fv_output`: the redacted output, then the copy's latest event;
every event is also broadcast to the connections as a model message of its
type (`output_ready`, …). Reactor audio is mono: the copy duplicates it to
stereo for Opus, and the sink up-mixes for AAC.

## Console

The Live stream page's native transport has a **destination form**
(`console/outputs.js`): Cloudflare (default; greyed out when the server has
no account), Cloudflare RTMPS / SRT, your WHIP endpoint (URL, token, codec,
endpoint kind), your RTMP(S) ingest (URL, stream key), your SRT listener
(URL, stream id, passphrase). Credentials are password inputs and are
never stored. The player follows the output:

| output | player |
|---|---|
| `cloudflare`, or any `whep_url` | WHEP (`rtc.js` `whepPlay`) on `playback.whep_url` |
| `cloudflare-rtmps`, `cloudflare-srt` | HLS on `playback.hls_url`: native HLS (Safari) or the vendored hls.js 1.5.20 light build (`console/vendor/`, Apache-2.0, loaded on first use, no worker under the console's CSP); a not-yet-live playlist is retried for 2 min |
| your own destination | the "Playback URL" the user pastes: HLS when it ends in `.m3u8`, else WHEP |

The director page has an optional **Output copy** section (the same form
with "No copy" first): it adds `x_fv_output` to `configure`, shows
`output_ready` / `output_error` / `output_ended`, and "Watch the copy" plays
it with the same rules.

## fv-control

- **Clusters and pods:** set `FV_CF_STREAM_API_TOKEN` as an account,
  cluster or pool variable with `secret: true` (sealed in D1, masked
  everywhere) whose value is the reference
  `{{ RUNPOD_SECRET_fv_cf_stream_api_token }}` (the Runpod secret exists
  since 2026-10-13), then roll the pods. A plain value is refused ("is a
  secret").
  `FV_CF_ACCOUNT_ID` is already in every pod's env (the Runpod secret
  `fv_cf_account_id`). `FV_OUTPUTS_ALLOW_PRIVATE` and
  `FV_DIRECTOR_ALLOW_OUTPUT` are switches (0/1).
- **Serverless endpoints:** the spec's `env` takes plain values only, so put
  the token in Runpod as a secret and reference it:
  `"FV_CF_STREAM_API_TOKEN": "{{ RUNPOD_SECRET_fv_cf_stream_api_token }}"`
  (a plain value is refused).
- **Brev / GMI pods:** `{{ RUNPOD_SECRET_… }}` references are replaced from
  `FV_PROVIDER_SECRET_ENV` (a reference with no value there is dropped).
- The cluster editor lists the new keys (`catalog.json engine_env`).

## Limits

- One output per native stream, one copy per director / Reactor session.
- WHIP outputs need outbound UDP (no TURN client in str0m, risk R2).
- Cloudflare's HLS/DASH ladder for a 1344x768 RTMPS/SRT input tops out at
  1260x720 (then 840x480, 630x360, 420x240); only WHIP/WHEP keeps the
  native canvas.
- Cloudflare HLS from SRT starts late: the input shows connected within
  10-15 s, but the first HLS segment took 68-72 s in all four runs (RTMPS:
  16-19 s). Prefer RTMPS (or WHIP) when start-up time matters.
- The HLS playlists carry no `EXT-X-PROGRAM-DATE-TIME`, so live-edge latency
  cannot be read from them; segments are 3 s (target duration).
- The build pod's ffmpeg 6.0.1 is a fully static build whose resolver
  crashes on host names (SIGSEGV in `getaddrinfo`); pinning the address
  sidesteps it for RTMP/SRT, and the live test fetches HLS over reqwest
  instead of handing ffprobe an `https://` URL.
- SRT values with URL delimiters are refused (ffmpeg takes them verbatim).
- VP9 over RTSP from MediaMTX does not decode in ffmpeg (an ffmpeg RTSP VP9
  depacketizer issue: MediaMTX's WebRTC read-back of the same stream decodes
  fine); not a Cloudflare path.
- Cloudflare: WHIP inputs have no HLS and no recording, RTMPS/SRT inputs no
  WHEP (Cloudflare's pairing rules); HLS appears some seconds after the
  broadcast starts.

## Testing

| suite | what |
|---|---|
| `cargo test -p fastvideo-serve --features full --test stream_outputs` | the Cloudflare default against a fake Stream API: live input `fv-test-<id>`, WHIP with `42e01f` offered and a 1344x768 level-3.2 SPS received, Opus, WHEP playback in the stream object, nothing secret in any response, the input deleted on stop; VP8 by request; refusals (no account, private targets, shapes) without echo; a 403 `10000` names the permission and frees the executor; RTMPS/SRT inputs record and return HLS/DASH, an unreachable ingest is `output_error` without ending the stream, recorded inputs are disabled not deleted |
| `--test output_copies` | a director session's `x_fv_output` over WHIP to an in-process endpoint: `session_info.x_fv_output`, `output` refused without `allow_output`, a bad output refused, `configured` echo, `output_ready`, 1344x768 H.264 + audio copied, WHIP DELETE after stop |
| `cargo test -p fastvideo-reactor --test runtime output_copy` | `start_session` `x_fv_output`: refusals, every paced tick tapped, events broadcast and in the descriptor, the tap closed with the session |
| unit tests | request parsing and kinds, redaction (`spec.rs`), URL policy, legacy aliases, `insecure_skip_verify` gating, the Cloudflare client's parsing / playback / stop policy, sink args (FLV, MPEG-TS, CBR clamp, AAC-LC, `+cgop`, `-rtmp_tcurl`), `srt_url`, stderr scrubbing, `exit_error` (ffmpeg's reason, no key; `fastvideo-media/tests/ffmpeg_io.rs`), VP9 args and keyframe detection, WHIP codec offers (`fastvideo-webrtc/tests/whip.rs`), the 1 s H.264 GOP for Cloudflare; the RTMPS relay (`rtmps.rs`): a test CA and name accepted, an unknown CA and a wrong name refused and fatal with no later connection, a refused TCP connection not fatal, a bad CA file an error; address pinning and `tcUrl` (`publish.rs`) |
| `FV_TEST_OUTPUTS_MEDIAMTX=1 … --test outputs_live` | real MediaMTX v1.15.1 (downloaded): see below |
| `FV_TEST_OUTPUTS_CLOUDFLARE=1 … --test outputs_live` | real Cloudflare Stream: one stream per kind (WHIP H.264 / VP8 / VP9, RTMPS, SRT; `FV_TEST_CF_CASES` picks some) for `FV_TEST_CF_SECONDS` (60). WHEP: a str0m viewer records what Cloudflare delivers, ffmpeg decodes it (codec, resolution, luma: a black picture shows). RTMPS/SRT: the test fetches the HLS playlists and segments over HTTPS and decodes them, reads the DASH manifest, and polls the input's `status` for when Cloudflare saw the ingest. Every `fv-test-` input and its recordings are deleted (a recording still processing answers 409 and is retried). The token comes from the environment only |
| `tests/console/stream_outputs.cjs` | the destination form → request shape for Cloudflare, Cloudflare SRT, your RTMPS with a key, your WHIP with a token and VP8; WHEP player on `webRTCPlayback`, hls.js on `hls_url`, a pasted playlist that really plays (VP9 fMP4: Playwright's Chromium has no H.264 for MSE), `playerFor` for every shape; no credential in storage, the page or the server log; the director's `x_fv_output` (redacted in the log) |

### Results: MediaMTX (build pod, fake engine, 2026-10-13)

`fake-h3-turbo` (1344x768 @ 24 fps, with audio), CPU encoders (libx264
test path, libvpx), MediaMTX v1.15.1 on loopback; read back by decoding 4 s
over RTSP with ffprobe:

| kind | MediaMTX tracks | ready after | decoded (frames / keyframes in 4 s) | resolution |
|---|---|---|---|---|
| WHIP H.264 | H264, Opus | 2.7 s | 53 / 2 | 1344x768 |
| WHIP VP8 | VP8, Opus | 3.2 s | 96 / 2 | 1344x768 |
| WHIP VP9 | VP9, Opus | 3.5 s | 53 / 2 (WebRTC read-back from MediaMTX; RTSP: see Limits) | 1344x768 |
| RTMP | H264, MPEG-4 Audio | 4.6 s | 51 / 2 | 1344x768 |
| RTMPS (test CA via `rtmps_ca_file`; and `insecure_skip_verify`) | H264, MPEG-4 Audio | 5.0 s; 4.9 s | 49 / 2; 50 / 2 | 1344x768 |
| SRT (passphrase required by the path) | H264, MPEG-4 Audio | 4.6 s | 74 / 2 | 1344x768 |

"Ready after" is from `POST /fv/v1/streams` to MediaMTX reporting the path
ready with its tracks (it includes the fake engine's first frame, ~2.4 s).
The RTMPS rows are the second run (RTMPS through the relay; the other rows
matched the first run within 0.5 s). In the same run, RTMPS to the same
MediaMTX **without** the test CA failed after 5.3 s with "RTMPS: the TLS
certificate of 127.0.0.1:<port> failed verification (UnknownIssuer); the
output stops rather than connect unverified" (key scrubbed), and nothing
reached MediaMTX.

### Results: Cloudflare from a GPU-served model (2026-10-14)

h3-turbo on an RTX PRO 6000 (NVENC), every kind against the real Cloudflare Stream, plus the director
`x_fv_output` copy: WHIP H.264 / VP8 / VP9 come back over WHEP at 1344x768 (29.7-30.8 s from the POST, the first
clip's build included), RTMPS / SRT as HLS topped at 1260x720 (41 s / 64-87 s), and every kind, the copy included,
carries the model's audio (decoded). Cloudflare serves HLS audio as a separate rendition (`#EXT-X-MEDIA TYPE=AUDIO`),
which a read of the top video variant alone does not see. docs/serve/e2e/h3-test-round-2026-10-14.md.

### Results: Cloudflare (live, 2026-10-13)

Build pod (EU-RO-1), fake engine `fake-h3-turbo` (1344x768 @ 24 fps with
audio), CPU encoders, one `fv-test-` live input per case, 60 s each; the
token reached the test process through its environment only. Final run,
on this branch's code (RTMPS/SRT timings: the range over all runs that
reached Cloudflare):

| case | ingest → playable | delivered | decoded | picture |
|---|---|---|---|---|
| WHIP H.264 | 5.2 s to the first WHEP frame (1.9 s after the WHEP POST) | WHEP, H.264 **1344x768** | 1360 frames / 57 keyframes in ~57 s (1 s GOP) | mean luma 131.7, deviation 53.8, 0 black frames |
| WHIP VP8 | 7.6 s (1.1 s after the POST) | WHEP, VP8 **1344x768** | 1273 / 54 | 131.8 / 53.8, 0 black: **VP8 plays correctly** |
| WHIP VP9 | 4.9 s (1.7 s after the POST) | WHEP, VP9 **1344x768** | 1358 / 57 | 131.7 / 53.8, 0 black |
| RTMPS (verified TLS) | 15.8 s to the first HLS segment that decodes (3 runs: 15.8-18.7 s); Cloudflare showed the input connected after 9.6 s | HLS H.264 **1260x720** top of 4 renditions (840x480, 630x360, 420x240); DASH the same ladder | 144 frames / 3 keyframes in 3 segments (6 s) | 131.8 / 53.9, 0 black |
| SRT (passphrase) | 72.4 s (4 runs: 67.6-72.4 s); connected after 9.9-14.4 s | HLS H.264 1260x720 top; DASH the same | 144 / 3 | 131.3 / 53.9, 0 black |

- The WHEP "ingest → playable" includes the fake engine's first frame
  (~2.4 s) and the test's WHEP retry (every 2 s from 2 s); WHEP itself
  answered and played within 1.1-2.8 s over all runs. Glass-to-glass latency was not
  measurable (no timestamped frames; Cloudflare's playlists carry no
  program date-time). HLS segments are 3 s.
- Audio arrived on every case (Opus over WHEP: ~2,800 packets per case).
- What the first run found and this branch fixed: (1) RTMPS and SRT to
  Cloudflare failed with only "ffmpeg exited": the sink read ffmpeg's
  stderr before its reader had drained it (now `exit_error`: status and
  stderr), which then showed **SIGSEGV**: the build pod's static ffmpeg
  crashes resolving host names (MediaMTX on 127.0.0.1 never resolved one).
  Pinning the checked address fixed both; the image's dynamically linked
  ffmpeg was not affected. A restart could also hide an exit (the sink
  restarted ffmpeg before the once-a-second check saw it dead): the sink
  now keeps the reaped ffmpeg's reason until it is read. (2) The H.264
  WHIP GOP was 2 s (27 keyframes a minute), not the documented 1 s; now 57.
  (3) RTMPS now verifies TLS.
- Cleanup: every `fv-test-` input deleted (WHIP inputs on stop; recorded
  RTMPS/SRT inputs disabled on stop, then deleted by the test with their
  recordings). A recording still processing answered 409 once; it was
  deleted by hand and the test now retries. Afterwards the account held only
  its pre-existing input and no videos.
- Cost (six runs, two of them RTMPS/SRT-only, two that failed before
  Cloudflare received RTMPS/SRT): about 9 minutes delivered (WHEP viewing
  ~8 min, HLS segments fetched) ≈ $0.009, and about 8 minutes of
  recordings, each stored for a few minutes before deletion (≤ $0.04 even
  if billed as a full month's storage): **under 5 cents** in total.
