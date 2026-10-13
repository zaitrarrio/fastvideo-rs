# Prompt upsampling

**Status (2026-10-13):** implemented, off by default. The owner asked
for "optional prompt upsampling for all models. if the model already
supports prompt upsampling make sure it is working. for the director models
implement the following [`src/upsampler.ts`] prompt upsampling technique".
Before this, no model rewrote prompts: every expansion flag was an accepted
no-op ([below](#what-was-a-no-op-before)). Now fv-serve can call any
OpenAI-compatible `/chat/completions` endpoint the operator configures:

- **Batch jobs** (fal, MiniMax, LTX API, native): a one-shot rewrite before
  generation when the request asks for it, with a system prompt per model
  family ([Batch rewrites](#batch-rewrites)).
- **Director sessions** (fal WMA `…/director`, clip and causal models): the
  owner's showrunner upsampler, ported 1:1 from
  [upsampler-spec.ts](upsampler-spec.ts) to `crates/fastvideo-upsampler`
  ([Director](#director)).

Nothing changes for a server without `[upsampler]`, or for a client that
does not ask (unless the operator sets `default_on`). No live LLM was
called in testing: every test runs against fake servers
([Tests](#tests)).

## Code

| Where | What |
|---|---|
| `crates/fastvideo-upsampler` | `chat`: the `ChatBackend` seam and the hedge; `http` (feature `http`): `OpenAiChat`; `director`: the spec port (`Upsampler`, `build_user_message`, `assemble_prompt`, `SYSTEM_PROMPT`, `output_schema`); `rewrite`: one-shot rewrites; `Service`: backend + settings, metrics, logs |
| `crates/fastvideo-protocol` | `GenerationRequest::prompt_expansion` (`PromptExpansion {mode, explicit, field}`, not serialized), `Job::expanded_prompt` |
| `crates/fastvideo-serve-kit/src/upsample.rs` | `expand_request`: the batch rewrite, called by `submit_request` (and MiniMax's own submit) after model resolution, before negotiation |
| adapters | each protocol's flag → `prompt_expansion` ([table](#per-protocol-mapping)); fal returns `expanded_prompt`, the native job view `expanded_prompt` |
| `crates/fastvideo-fal/src/director/upsample.rs` | `SegmentUpsampler`: which segment takes which expansion, expanded ahead, fallbacks |
| `crates/fastvideo-fal/src/director/session.rs` | the session wiring (clip chunks held for their prompt, causal segments) |
| `crates/fastvideo-serve/src/config.rs`, `app.rs` | `[upsampler]`, `FV_UPSAMPLER_*`, the service on `ServeCtx` |
| `crates/fastvideo-serve/console/director.js` | the director page: toggle, style, entities, the segment list |

## Configuration

```toml
[upsampler]
base_url = "https://llm.example.com/v1"   # any OpenAI-compatible API root; /chat/completions is appended
model = "<chat model>"
api_key_env = "FV_UPSAMPLER_API_KEY"      # the variable holding the key (default); unset: no auth header
timeout_ms = 10000                        # one call, both hedged attempts included (100..120000)
hedge = true                              # two concurrent attempts, first success wins
memory = 12                               # director: prior segments as context when configure.memory is absent (1..50)
default_on = false                        # see Defaults
[upsampler.max_chars]                     # prompt budgets (Limits)
wan = 1800
ltx = 3600
h3 = 3200
other = 1800
```

Environment (they override the file; a base URL alone creates the
section): `FV_UPSAMPLER_BASE_URL`, `FV_UPSAMPLER_MODEL`,
`FV_UPSAMPLER_TIMEOUT_MS`, `FV_UPSAMPLER_HEDGE`, `FV_UPSAMPLER_DEFAULT_ON`,
`FV_UPSAMPLER_API_KEY_ENV`, and the key in `FV_UPSAMPLER_API_KEY` (or the
variable `api_key_env` names, also in its lower-case Runpod-secret
spelling). The key is held as a `Secret`: never in logs, `Debug`, the
serialized config or error messages (the startup line says `api_key=set`).
The HTTP client needs fv-serve's `http-client` feature (every release
image has it); without it, or with an unusable URL, the server logs why
and runs without upsampling.

**Local sidecar (owner decision, 2026-10-13).** The planned production LLM
is a small local instruct model served by llama.cpp's `llama-server` on the
same GPU as fv-serve (built separately: image, weights, boot wiring). It is
an OpenAI-compatible `/v1/chat/completions` that takes the director's
`response_format: {type: "json_schema", json_schema: {name, strict,
schema}}` as sent (grammar-constrained). Point fv-serve at it with the
environment alone, no TOML edit and no key:

```bash
FV_UPSAMPLER_BASE_URL=http://127.0.0.1:<port>/v1
FV_UPSAMPLER_MODEL=<the name llama-server serves>
FV_UPSAMPLER_HEDGE=0      # one request per call: the hedge doubles the load on a local server
```

With no key (`FV_UPSAMPLER_API_KEY` unset or empty) no `authorization`
header is sent. For a small model the system prompts of the batch rewrites
are short and concrete, a leading `<think>…</think>` block (or a lone
`</think>`) is stripped before parsing or using an answer, a Markdown code
fence around the director's JSON is dropped, and an answer that still does
not parse or misses a field counts as a failure and falls back.

**fv-control.** No code change was needed. Controller clusters: set
`FV_UPSAMPLER_BASE_URL` and `FV_UPSAMPLER_MODEL` as account, cluster or
pool variables, and `FV_UPSAMPLER_API_KEY` as a **secret** variable
(sealed in D1, masked everywhere; docs/control/README.md §5), then roll
the pods. Serverless endpoints: put the base URL and model in the spec's
`env`, and the key as a Runpod secret reference
(`FV_UPSAMPLER_API_KEY = "{{ RUNPOD_SECRET_fv_upsampler_api_key }}"`): the
spec's `env` takes plain values only, never a secret itself.

**Cost.** One call per batch rewrite; with `hedge` every call is two
requests. A director session makes one call per segment: per chunk on a
clip model (5 or 10 s of video), per director chunk on a causal model (3 s
of video at the defaults), so a long causal session makes a call every few
seconds. Turn `hedge` off to halve the cost.

## Defaults

A request's flag is either **explicit** (the client sent it) or the
protocol's **documented default** (the client omitted it):

| `[upsampler]` | explicit `disabled` / `false` | explicit on (`balanced`, `quality`, `true`, …) | omitted |
|---|---|---|---|
| absent | no-op | no-op (accepted, as before) | no-op |
| present, `default_on = false` (default) | off | **rewritten** | off |
| present, `default_on = true` | off | **rewritten** | the protocol's documented default (fal and MiniMax H3-Max `balanced`, fal ingredient and the LTX API on, director on; Wan and native off) |

So an operator who adds `[upsampler]` changes nothing for existing
clients until they opt in; `default_on` makes the servers behave like the
hosted APIs, which expand by default.

## Batch rewrites

`expand_request` runs in the submit pipeline after admission, safety,
model resolution and the precheck, and before staging and negotiation: the
engine generates with the rewritten prompt, and the job records it
(`Job::expanded_prompt`, a job log line, `upsample` trace span). It is
synchronous: the submit answers after the rewrite (at most `timeout_ms`;
a `balanced` rewrite is usually a second or two).

One call: system prompt per family and effort, the client's prompt as the
user message, temperature 0.6, plain text out. The answer is cleaned
(code fences, a `Prompt:` label and wrapping quotes go; lines join into one
paragraph) and cut to the family budget at the last sentence end that fits
(else the last whole word).

| Family (`ModelCaps.family`) | Guide followed | Sound | Words: `balanced` / `quality` |
|---|---|---|---|
| Wan (Wan 2.x, FastWan, SF-Wan, LongLive) | Wan's prompt-extend: subject and action first, appearance, setting and spatial layout, style, shot scale and camera motion, lighting; motion emphasized; a style chosen when none is given | none (the model is silent) | 60-100 / 100-160 |
| LTX-2.x | LTX's prompt guide: one detailed paragraph as a cinematographer describes a shot, starting with the main action; movements, appearance, background, camera, lighting and colour, changes in the shot | ambience and effects; dialogue quoted after the speaker | 80-130 / 140-200 |
| H3 | subject and appearance, scene, action through the shot, camera language (shot size, angle, push / pull / pan / tilt / tracking), aesthetic; dialogue kept verbatim | ambience, music, effects | 80-130 / 140-200 |
| other | subject, appearance, action, setting, camera, lighting, mood | – | 80-130 / 140-200 |

Every system prompt also says: keep the user's intent, subjects, quoted
text and explicit style or camera instructions; add only plausible
concrete detail; one continuous shot in the present tense; English; no
meta commentary; the length bound and the character budget.

**Image-to-video** (keyframes, image references): text only. The LLM does
not see the image; the system prompt says an unseen image sets the first
frame, so it describes motion and camera from there and does not restate
the image's appearance. Retake / extend edits are never rewritten.

**Fallback.** A failed or timed-out call (non-2xx, no content, transport
error, timeout) leaves the client's prompt; the job log says
`prompt expansion failed (<error>); generating with the original prompt`
and `expanded_prompt` stays null. The job never fails because of the LLM.

### Per-protocol mapping

| Protocol | Flag | Mapping | Output |
|---|---|---|---|
| fal H3 (`minimax/h3*`: t2v, i2v, r2v) | `prompt_expansion_mode` (any string, default `balanced`) | `disabled` off; `balanced` and H3 base's `fast` → balanced; `quality` → quality; other strings: no effect. Omitted: fal's default (with `default_on`) | `expanded_prompt` (was always null) |
| fal Wan (`fal-ai/wan/v2.2-5b/*`) | `enable_prompt_expansion` (default `false`) | `true` → balanced | `expanded_prompt` |
| fal LTX ingredient (`fal-ai/ltx-2.3-quality/ingredient`) | `enable_prompt_expansion` (fal default `true`; not in our schema before, ignored) | `true` / `false` explicit; omitted: on with `default_on`; a non-boolean is ignored | `expanded_prompt` |
| fal LTX partner endpoints (`lightricks/ltx-2.5/*` t2v / i2v) | `enable_prompt_expansion` (ours: fal has no switch there) | `true` → balanced; absent or non-boolean: nothing | `expanded_prompt` |
| LTX API (`/v1`, `/v2` t2v / i2v) | `enhance_prompt` (ours, after the LTX CLI's `--enhance-prompt`; the hosted API enhances server-side with no switch) | `true` / `false` explicit; omitted: on with `default_on`; a non-boolean is ignored | the job (no output field in the LTX API) |
| MiniMax V2 | `extra.prompt_expansion_mode` (H3-Max and our tiers; default `balanced`) and V1's `prompt_optimizer` (boolean, any model) | as fal; `extra` wins over `prompt_optimizer`; omitted on an `extra` model: `balanced` with `default_on` | the job (the query has no prompt field) |
| native `/fv/v1/jobs` | `prompt_expansion`: `disabled` (default), `balanced`, `quality`; other values are a 400 | explicit only | `expanded_prompt` in the job object |

Not covered: OpenAI-compatible `/v1/videos` (its request model is
`extra="forbid"` upstream, so it has no sensible place for a flag; use the
native API), fal and LTX audio-to-video / retake / extend (their prompts
describe edits of a given clip), Reactor and native live streams
(`/fv/v1/streams`), whose prompts change per block and are the client's.

## Director

**Opt-in.** `configure` takes three fields of ours (strict like the rest):

| Field | Meaning |
|---|---|
| `upsample` (bool) | Rewrite every segment with the server's upsampler. Omitted: the server's `default_on` |
| `style` (string, ≤ 2000) | The world bible's style, appended to every segment |
| `entities` (array, ≤ 32) | The opening cast: `{id (1..64), kind: character | location | prop, name (1..200), descriptor (1..1000)}`, ids unique; unknown fields or kinds are `invalid_message` |

The world bible's premise is `configure.prompt`; `configure.memory` (1..50,
default 12, already in fal's schema: "prior segment prompts retained as
context for future prompt expansion") is the spec's `memory`.
`maxPromptChars` is the family budget ([Limits](#limits)). `configured`
echoes `upsample: true` when it is on; `upsample: false` with
`upsample_error` when the client asked but the server has no upsampler
(the session runs on the raw prompts). `session_info` on a server with an
upsampler says `prompt_expander: "upsampler"` (else `"none"`, as before)
and adds `upsample: {available, default, max_prompt_chars,
server_message_types: ["segment_expanded"]}`. fal's 16
`server_message_types` are unchanged.

**Segments.** Each segment the director needs takes its prompt from
`Upsampler.next(direction)`:

- clip models (H3, LTX): each chunk is a segment;
- causal models (LongLive, SF-Wan): the first block, every director chunk
  (`causal_chunk_blocks` blocks), and every new direction start a segment;
  the prompt goes to the engine with `set_prompt` at the next block
  boundary (one KV re-cache per switch on LongLive).

The **direction** is the control's current direction for the chunk
(`ChunkPlan::direction`): a new one (a `prompt` message, a script beat's
text; id = its `prompt_version`) is passed once as the spec's `Direction`;
a chunk whose direction was already consumed passes `null` (advance one
beat). Replan and append keep their meaning: they decide which chunk
carries the direction.

**Ahead of time.** After a segment is committed the next beat is expanded
at once, while the current chunk generates (only when no planned chunk is
waiting in the deck; a planned direction is expanded when it comes up). A
replanning `prompt` starts its expansion the moment it arrives and drops a
speculative beat. A segment whose expansion is not ready waits for it: a
clip chunk is held (not yet dispatched, its `prompt_applied` held with it);
a causal rollout keeps its current prompt, or, for the first segment, has
not started. The wait is bounded by `timeout_ms`. To make this possible the
port splits `next` into `user_message` → `expand_with_hedge` → `commit`
([Deviations](#director-port)); a speculative expansion is never committed,
so the ledger only holds segments that played.

**Fallback.** A failed or timed-out call uses the raw text the director
would have used without upsampling (premise + direction), records nothing
in the ledger (as `next` throws in the spec), and is reported with
`fallback: true`; the direction counts as consumed. The stream never stops
for the LLM.

**`segment_expanded`** (ours; only to sessions with upsampling on), sent
when a segment's prompt goes to the engine:

```json
{"type": "segment_expanded", "chunk_index": 3, "first_block": null, "prompt_version": 2,
 "segment_index": 3, "direction_id": "2", "summary": "The storm breaks over the tower.",
 "prompt": "a white lighthouse on black rocks Mara: … Sound: … 35mm film.",
 "new_entities": [{"id": "ship", "kind": "prop", "name": "The ship", "descriptor": "…"}],
 "fallback": false, "error": null, "latency_ms": 1840.2}
```

`chunk_index` on clip models, `first_block` (blocks received so far) on
causal ones; `segment_index`, `direction_id`, `summary` are the ledger
record (null on a fallback); `prompt` is what the engine got.

**Console.** The director page has a "Prompt upsampling" section (toggle,
style, entities as JSON) and lists every `segment_expanded` under the
timeline (where, version or `fallback`, summary, prompt, new entities);
session info shows whether the server has an upsampler.

### Director port

The spec ([upsampler-spec.ts](upsampler-spec.ts), committed verbatim) maps
1:1 onto `fastvideo_upsampler::director`: `Entity`, `EntityKind`,
`WorldBible`, `Direction`, `SegmentRecord`, `ExpandedSegment`,
`DialogueLine`, `ExpansionOutput`, `UpsamplerConfig`, `SYSTEM_PROMPT`
(verbatim), `OUTPUT_SCHEMA` (`output_schema()`, the same strict
`segment` json_schema), `buildUserMessage`, `callModel` (`call_model`:
temperature 0.8, `response_format: {type: "json_schema", json_schema:
segment}`, bearer key), `expandWithHedge` (`expand_with_hedge`: two
concurrent attempts with the hedge, the first success wins, a failure waits
for the other, both are dropped (HTTP requests aborted) at the timeout),
`assemblePrompt`, `Upsampler.next` (registry accepts new ids only,
`entity_ids` filtered to the registry, ledger `slice(-memory)`) and
`snapshot()`. Where Rust needed a choice the TS does not make:

- **Characters, not UTF-16 units:** lengths and the cut count Unicode
  scalar values (JS `length` / `slice` count UTF-16 code units); the cut
  never splits a character.
- **Strict parsing:** `JSON.parse(...) as ExpansionOutput` checks nothing;
  serde requires every schema field with its type, so a malformed answer
  fails the call (the TS would fail later on `undefined`). An entity `kind`
  outside the three parses as `other` (treated like a non-location, as the
  TS treats any string but `"location"`).
- **Small-model tolerance:** before parsing, a leading `<think>…</think>`
  block and a code fence around the JSON are removed (the TS parses the raw
  content).
- **`next` split** into `user_message` (read-only), `expand_with_hedge` and
  `commit`, for the director's look-ahead; `next` is the three in a row.
- **Kept as in the TS** (noted, not changed): over `maxPromptChars` the
  *last* part is dropped before the cut, which is the style when there is
  one and the `Sound:` line when the style is empty; `memory = 0` keeps the
  whole ledger (`slice(-0)`), which the director never passes (1..50); a
  dialogue speaker not in the registry is "a voice"; `baseUrl` gets
  `/chat/completions` appended (a trailing `/` is trimmed first); without a
  key no `authorization` header is sent (the TS always sends
  `Bearer ${apiKey}`).
- Error messages keep the spec's (`upsampler LLM <status>: <body>`, the
  body cut to 500 characters; `upsampler LLM returned no content`); all
  hedged attempts failing joins their messages (`AggregateError`).

## Limits

Prompt budgets in characters, from each text encoder's token limit at about
3.5 to 4 characters per English token, with headroom. `[upsampler.max_chars]`
overrides them.

| Family | Text encoder, limit | Budget |
|---|---|---|
| Wan, FastWan, SF-Wan, LongLive | umT5-XXL, `text_len` 512 tokens (`wan/config.rs`) | 1800 (the spec's figure) |
| LTX-2.x | Gemma 3, `max_sequence_length` 1024 (`ltx2/config.rs`) | 3600 |
| H3 | the Qwen3-VL text tower, `H3_MAX_PROMPT_TOKENS` 1024, **refused beyond, never truncated** (`h3/tokenizer.rs`) | 3200 (a 20% margin: an over-long prompt fails the job) |
| other | – | 1800 |

The client-facing prompt limits (fal 50 000, MiniMax 7000, LTX 5000
characters) still apply to what the client sends.

## Metrics and logs

- `fv_upsampler_calls_total{kind="rewrite"|"director", outcome}`:
  `ok`, or the failure (`status`, `no_content`, `bad_output`, `transport`,
  `timeout`, `all_failed`); every non-`ok` outcome is a fallback.
- `fv_upsampler_latency_seconds{kind}`: per call, both hedged attempts
  included.
- Logs: `prompt upsampler configured` at startup (URL, model, timeout,
  hedge, default_on, `api_key=set|unset`); `upsampler: prompt rewritten` /
  `rewrite failed, the original prompt is used` per batch call (family,
  effort, latency, lengths); `director: prompt upsampling on` per session;
  `director: segment upsampled` / `upsampling failed, the raw prompt is
  used` per segment (chunk or block, version, latency). Prompts are not
  logged; the job log carries the rewrite note.

## What was a no-op before

| Flag | Before | Now |
|---|---|---|
| fal `prompt_expansion_mode` | accepted, `expanded_prompt: null` | [mapped](#per-protocol-mapping) |
| fal Wan `enable_prompt_expansion` | accepted, no effect | mapped |
| fal ingredient `enable_prompt_expansion` | ignored (not in our schema) | mapped (fal's default `true`) |
| MiniMax `extra.prompt_expansion_mode` | accepted, no effect | mapped |
| MiniMax V1 `prompt_optimizer` | ignored as unknown | mapped |
| LTX API | no switch | `enhance_prompt` (ours) |
| director `memory` | accepted and echoed | the upsampler's history window |
| director `prompt_expander` | `"none"` | `"upsampler"` when available |

Without `[upsampler]` all of these behave exactly as before (the request
goldens, `accepted_noop` notes and `expanded_prompt: null` are unchanged).

## Tests

- `cargo test -p fastvideo-upsampler --features http`: the spec port
  against hand-computed expectations (user message layout, prompt assembly
  order, dialogue, the style drop and character cut, registry and ledger,
  memory window, failed calls record nothing), the hedge (first success,
  a failure waits for the other, both fail, the timeout drops both), the
  rewrite cleaning and cut, and the HTTP path against a local fake server
  (request body and bearer key, hedge with one 500, timeout aborting both
  connections, non-2xx, empty content, no choices, non-JSON content and
  body, connection refused, no key no header, `Debug` without the key).
- `cargo test -p fastvideo-fal --test upsample`: every fal flag's
  normalization; on the queue with a fake LLM, explicit modes rewrite and
  return `expanded_prompt`, omitted waits for `default_on`, `disabled`
  never, a failed call falls back with a job log line, and nothing changes
  without an upsampler.
- `cargo test -p fastvideo-fal --features director --test director_e2e upsample`:
  over WebRTC with a fake LLM: a clip session (segments per chunk with the
  cast, location and style spliced in, a direction once, a failed
  direction falling back to premise + direction, upsampling resuming), a
  causal session (the rollout starting on the first segment, a segment per
  director chunk, a direction applied with `prompt_applied`, `memory`), and
  `upsample: true` without an upsampler.
- `crates/fastvideo-fal/src/director/upsample.rs` unit tests: keys,
  speculative beats replaced by a direction, fallbacks.
- `cargo test -p fastvideo-serve --features http-client --test upsample`:
  fv-serve from `FV_UPSAMPLER_*` against a fake HTTP LLM (native
  `prompt_expansion`, MiniMax `prompt_optimizer`, the key as bearer and
  never in `Debug` or the serialized config), and the `[upsampler]`
  validation.
- MiniMax `validation.rs` and LTX API `request.rs` unit tests for their
  flags; the fal, MiniMax, LTX and protocol goldens are unchanged.
- `tests/console/director_upsample.cjs` (headless Chromium, in
  `tests/console/run.sh`): the director page's toggle, style and entities
  against fv-serve's fake engine and a fake LLM; the segments shown, the
  fallback, and the spec's request on the wire.

Untested: a real LLM (none was available while building this; the request
format is the spec's and the common OpenAI-compatible one, but a provider
without `response_format: json_schema` support would fail every director
call and fall back), the quality of the family system prompts on real
generations, and GPU director sessions with upsampling.
