# GPU E2E: h3-max (Sol-H3 tau ladder) on a Runpod pod

WP-18 run, pod B, 2026-09-28. Design refs: [design.md](../design.md) §7.6, §8.
Raw results: [`artifacts/serve/e2e/h3-max/`](../../../artifacts/serve/e2e/h3-max/).

## Setup

| Item | Value |
|---|---|
| Image | `ghcr.io/zaitrarrio/fastvideo-rs-serve:sha-9c42844` (`@sha256:3beb71e4…bc330`) |
| Config | `/etc/fv/runpod-h3-max.toml` (`configs/serve/runpod-h3-max.toml`): `sol-h3`, recipe `h3-max` = `sol-h3-4step-engine-ladder`, resident |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB, driver 595.91.07, EUR-IS-1, $2.09/hr |
| Weights | EU network volume `jg48s6o1w0` at `/workspace`, read only |
| Stores | jobs D1, artifacts R2 (s3), webhook key configured; NVENC probe OK |
| Pod | `9hfkyhux5frj5j` (`fv-e2e-b-0928164248`), 16:42:48 to 16:59:07 UTC, deleted and checked (API 404) |

Stock: no RTX PRO 6000 in EUR-IS-1 for 10 min (`create pod: There are no
instances currently available`). The US H100 types in US-CA-2 then failed
with `could not find any pods with required specifications`, but the
`allowedCudaVersions: ["13.0"]` filter was still set at that point. Once that
filter was dropped (the image checks for driver ≥ 580 itself), the first EU
attempt succeeded. This run never landed on an H100, so the Hopper MXFP8 →
W8A8 fallback (c054278) is **still untested for h3-max**.

## Boot

| Phase | Time after create |
|---|---|
| First `/ping` answer (204, container up, model loading) | 242 s |
| `/ping` 200 (Sol-H3 resident, ready) | 422 s (7.0 min) |

The model load took about 3 min (204 → 200). WP-11 measured about 19 min on
the same volume, so the earlier load was probably slowed by a cold volume
cache or a different host.

## Results

All generations are 5 s at 768p: 1344x768, 124 frames at 24 fps, H.264 plus
AAC stereo at 32 kHz. Every MP4 was downloaded and checked with ffprobe.
The `/v1/videos/sync` call downloaded only its 302, not the MP4 (see the
notes). Wall time runs from the client's submit to the terminal status,
measured from this sandbox through the Runpod HTTPS proxy with polls every
3 s.

| Endpoint | Result | Wall | Denoise | Notes |
|---|---|---|---|---|
| fal queue `minimax/h3-max/text-to-video` | PASS | 31.8 s | 21.38 s | 8.3 MB MP4 from R2 |
| fal sync `/run/minimax/h3-max/text-to-video` | PASS | 31.1 s | 21.37 s | 200, output JSON on the connection |
| MiniMax V2 `MiniMax-H3-Max` create → query → download | PASS | 33.6 s | n/a | `succeeded`; `usage.output_seconds 5`; content.url download 200 with no key |
| FastVideo `POST /v1/videos` → poll → `/content` | PASS | 34.1 s | 21.35 s | `completed`; peak 56.4 GB |
| FastVideo `POST /v1/videos/sync` | PASS (302) | 30.2 s | 21.09 s | Answers a 302 to R2 with the `X-*` headers (see the notes) |
| Native `/fv/v1/jobs` → poll → `/content` | PASS | 30.2 s | n/a | queue 0.3 s; run 28.8 s (created → completed) |
| fal queue `minimax/h3-max/image-to-video` | PASS (see notes) | 125.2 s | 23.65 s | FL2VA text stage streams Qwen-VL from the volume, as in WP-11 |
| fal director `fal_director.mjs` (browser) | NOT RUN | n/a | n/a | Sandbox Chromium does not trust the egress proxy's CA (`ERR_CERT_AUTHORITY_INVALID`) |
| fal director signalling (aiortc offer → `/wma/session`) | PASS (signalling) | 0.67 s answer | n/a | `x-fv-model sol-h3`, tier max; heartbeat `{alive:true}`; media not connected (below) |

Per-stage split (`X-Stage-Durations`, `/v1/videos/sync`):

| Stage | Seconds |
|---|---|
| text | 0.41 |
| denoise (4 steps, tau ladder) | 21.09 |
| video_decode (full VAE) | 6.51 |
| audio_decode | 0.05 |
| encode (NVENC MP4) | 0.61 |
| **engine total** | **28.7** |

The engine total matches the known 28.7 s RTX PRO 6000 figure. On top of
that, finalize, the R2 upload and the D1 writes add about 1.5 s: the native
job ran 28.8 s from created to completed, and the fal and MiniMax walls add
the 3 s poll interval and the proxy round trips. The denoise time, 21.1 to
21.4 s, matches the profile header (21.80 s median).

## Quality

Frames were extracted and inspected (`frames-grid.jpg`, `fal-t2v-frame60.jpg`).
The T2V outputs (fox in snow, hummingbird, ocean waves, coffee cup) are
sharp and coherent and follow their prompts. There is no noise, colour cast
or banding.

I2V (`fal-i2v-strip.jpg`): frame 0 is the beach fixture, then the clip
**hard-cuts** to the fox-in-snow scene. `fal-queue-smoke.sh` sends the
default fox prompt with the beach image, and the model follows the text
after the first frame. The endpoint works mechanically, but this run does
not show that I2V keeps the scene. Rerun it with a prompt that describes
the image.

## Notes and findings

- **`/v1/videos/sync` on R2 answers 302, not `video/mp4` bytes.** This is
  intentional in `sync_reply`: `ArtifactLocation::Local` returns bytes, and
  anything else returns `artifact_reply` with a redirect. The `X-*` headers
  are on the 302. `requests` follows the redirect (as a GET) and exposes the
  final response's headers, not the 302's, so a FastVideo client that reads
  `X-Inference-Time-S` from `r.headers` gets nothing. §4.1 says "Returns
  `video/mp4`". Recorded here, not changed: either stream the object bytes
  (`ArtifactStore::open`, as the LTX v1 sync path does) or document the
  redirect.
- The `/v1/videos` retrieve body of a completed job has `"url": null`. The
  content is served from `/content`.
- fal output `file_name` is `<nanoid>_minimax-h3.mp4` for the `h3-max` app.
  The slug drops the tier. This is cosmetic.
- **Director app id**: the WMA routes want `app_id: "minimax/h3-max/director"`.
  `minimax/h3-max` answers 404 `unknown app`. This matches what `wma(app)`
  sends, so it is not a bug.
- **Director media**: the answer carries only ICE-TCP passive candidates
  (`157.157.221.177:10897`, public, and the container IP). aiortc has no
  ICE-TCP, so its peer connection ended `failed`, which is expected. The
  real browser suite could not run here because the sandbox's headless
  Chromium rejects the egress proxy's CA for `*.proxy.runpod.net`. That
  happens both with `chromium-headless-shell` and with the full
  `channel: "chromium"` build. Run the director suite from a normal network:
  `node tests/compat/suites/fal_director.mjs <node dir> <pod url> <key>`.
- The Runpod proxy (Cloudflare) answers 403 `error code: 1010` to
  Python-urllib's default User-Agent. Clients need their own UA; `requests`
  and the fal clients are fine.
- `/metrics` after the run: 7 jobs submitted, 7 succeeded
  (fal 3, openai_videos 2, minimax_v2 1, native 1).

## Spend

Pod lifetime 16 min 19 s at $2.09/hr ≈ **$0.57**. Balance before $39.78, after
$36.76; the difference includes the other agents' pods. No volume was
written. The detached backstop (4500 s) was stopped after the pod was
deleted.
