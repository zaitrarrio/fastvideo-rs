# Vendored upstream API specifications

Machine-readable copies of the upstream contracts our serving adapters mirror,
with a drift check. Plan E1 (`docs/plan/2026-10-23-execution-plan.md`), from
`docs/gaps/2026-10-22-architecture-review.md` §4.4 and §7.6 item 1. Nothing
here is executed: these are reference documents (and two reference Python
sources) that tests and generators read.

## Rule

**The prose research docs cite these files from now on.** `research-fal.md`,
`research-ltx-api.md`, `research-minimax-fastvideo.md` and `fal-parity.md`
each start with a "Machine-readable source" pointer. When a claim in a
research doc and a vendored file disagree, the vendored file (at its
`fetched_at`) wins and the prose gets fixed. When the drift check reports a
change, the PR that accepts it (`--update`) also revisits the prose that
cites the changed file.

## What is vendored

One directory per upstream; each has a `SOURCES.toml` with one `[[source]]`
per file: `url` (pinned where the upstream offers an immutable URL),
`drift_url` (the moving URL re-fetched by the check, when different),
`fetched_at` (UTC), `sha256` of the file as stored, `normalize` and a `note`
saying what was trimmed or why a document is unavailable.

| Directory | Files | Source | Normalisation |
|---|---|---|---|
| `ltx/` | `openapi.json` | `https://docs.ltx.io/openapi.json` (OpenAPI 3.1.0 of api.ltx.io) | JSON, pretty-printed, sorted keys; complete |
| `fal/` | `<endpoint id>.openapi.json`, 21 files | `https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=<id>`, one per fal endpoint this server mirrors: `minimax/h3-max/{text,image,reference}-to-video`, `minimax/h3-max-turbo/{text,image}-to-video`, `minimax/h3/{text,image,reference}-to-video` (`crates/fastvideo-fal/src/schema.rs`); `lightricks/ltx-2.5/{text,image}-to-video/{fast,pro}` (`schema/ltx.rs`); `lightricks/ltx-2.5/audio-to-video/{fast,pro}` (`schema/a2v.rs`); `fal-ai/ltx-2.3/{extend,retake}-video` (`schema/edit.rs`); `fal-ai/ltx-2.3-quality/ingredient` (`schema/ingredient.rs`); `fal-ai/wan/v2.2-5b/{text-to-video,image-to-video,text-to-video/fast-wan}` (`schema/wan.rs`); and `fal-ai/ltx-2.3-22b/text-to-video` (the `video_size` enum source cited by `fal-parity.md` §0) | JSON, sorted keys; complete |
| `fal/` | `fal-ai/minimax-h3-max-director/asyncapi.json` | `https://fal.ai/api/apps/fal-ai/minimax-h3-max-director/asyncapi.json` (AsyncAPI 3.1.0, the director's WebRTC control channel) | JSON, sorted keys; complete |
| `minimax/` | `v2-video-generation.json` | `https://platform.minimax.io/docs/api-reference/video/generation/api/v2-video-generation.json` (OpenAPI 3.1.0 of `POST /v2/video_generation` and `GET /v2/query/video_generation/{task_id}`) | JSON, sorted keys; complete |
| `openai/` | `videos.openapi.yaml` | `github.com/openai/openai-openapi`, `openapi.yaml` at `main` commit `0938b0f0…` (4.2 MB) | **TRIMMED** to the `/videos*` paths, the `components.schemas` closure they reference, `components.securitySchemes` and the `openapi`/`info`/`servers`/`security` preamble; blocks copied as text in upstream order (`trim_openai_videos` in `scripts/serve/lib/specs-drift.py`) |
| `fastvideo/` | `protocol.py`, `api_server.py` | `github.com/hao-ai-lab/FastVideo`, `fastvideo/entrypoints/openai/` at `main` commit `8db2d71e…` | verbatim bytes |
| `fastvideo/` | `videos.schema.json` | **DERIVED** by hand from `protocol.py`'s `VideoGenerationRequest` (every pydantic field: name, type, default, bounds, required; `extra="forbid"`) | JSON Schema 2020-12; `$comment` names the commit it was derived from |

Unavailable, recorded in `fal/SOURCES.toml` with `status = "unavailable"` and
re-probed on every run (a document that starts answering is reported so a
human vendors it):

- `minimax/h3-max/director` queue OpenAPI: the URL answers 404 with `null`
  (the director is a WebRTC session, not a queue endpoint; `research-fal.md`
  §1, §8).
- the director app's own OpenAPI (`…/minimax-h3-max-director/openapi.json`,
  with `/start-session`, `/info`, `/health`): 405 without a fal API key;
  paraphrased in `research-fal.md` §8 from the model page.

Not vendored on purpose: fal's `llms.txt` model pages (prose, not a spec; the
queue OpenAPI carries the same schema), the LTX docs pages (the OpenAPI
carries the contract), and FastVideo files other than the two the HTTP
contract lives in. `protocol.py` moved on from the research pin `e90be598`
(`scripts/gpu/upstream/setup.sh` `FV_REV`): it adds `audio_start_in_s`,
`audio_end_in_s` and `audio/wav` as a `VideoResponse.media_type`; the prose
in `research-minimax-fastvideo.md` §2 predates those.

## The drift check

```
bash scripts/serve/specs-drift.sh            # exit 1 on any drift
bash scripts/serve/specs-drift.sh --only fal # one upstream
bash scripts/serve/specs-drift.sh --update   # accept upstream changes
```

`scripts/serve/specs-drift.sh` runs `scripts/serve/lib/specs-drift.py` under
`python3 -I` (bash, curl and the Python standard library only; TLS
verification stays on and the environment's proxy applies). For every
`[[source]]`:

1. the stored file's sha256 is compared with `SOURCES.toml` (a local edit
   without `--update` fails the check);
2. entries with a URL are re-fetched from `drift_url` (else `url`) with
   `curl --fail`, normalised the same way as the stored copy (`json`:
   pretty-print with sorted keys; `verbatim`; `openai-videos`: the same
   trim) and compared byte for byte;
3. `derived` entries are integrity-checked only; `unavailable` entries are
   probed and reported if they start serving a parsable document.

The report has one line per spec (`ok`, `derived`, `unavailable`, `DRIFT`,
`UNREACHABLE`, `UNPARSABLE`, `AVAILABLE`, `SHA256`, `MISSING`), the head of a
unified diff for every drift, and a summary; the exit code is 1 when anything
other than `ok`/`derived`/`unavailable` was found. `--update` rewrites the
drifted files, their `sha256` and `fetched_at`, and `SOURCES.toml` (its header
comment is kept). Re-derive `fastvideo/videos.schema.json` by hand whenever
`protocol.py` changes.

`.github/workflows/specs-drift.yml` runs the check every Monday (and on
`workflow_dispatch`); on drift it opens or updates one issue titled
"Upstream API spec drift" with the report, closes it when the check is green
again, and never fails the run. It does not run on pull requests.

## Later work packages

E2 derives `ParamSpec` per (surface, model) and generates the native and
director schemas from the Rust types with `schemars`; E3 builds the
conformance generator over these files (boundary-valid bodies and one invalid
body per constraint, replies validated against the response schemas).
