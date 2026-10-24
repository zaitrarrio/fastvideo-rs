# ADR-0012: `ParamSpec`: one source for every request limit

- Status: Accepted (plan E2; the director and Reactor control channels and
  the certification matrix follow in E3, see "Not in this change")
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-param-spec
- Code: `crates/fastvideo-protocol/src/params/` (`mod.rs` the types and the
  agreement harness, `rules.rs` the surface rules, `json_schema.rs` the fal
  reader); `params.rs` in `fastvideo-openai-videos`, `fastvideo-minimax`,
  `fastvideo-ltxapi`, `fastvideo-fal` and `fastvideo-serve` (the builders
  and `GET /fv/v1/params`); `crates/fastvideo-ltxapi/src/body.rs` (the typed
  LTX bodies)
- Doc: `docs/serve/params.md`

## Context

The architecture review (`docs/gaps/2026-10-22-architecture-review.md`
§4.4, §7.6 item 2) found request limits duplicated across the adapters and
disagreeing between surfaces: the H3 5-15 s rule in three crates, the H3
reference counts in three, the LTX duration matrix in two, the LTX camera
motions, prompt length and rates in two, the fast-h3 prompt limit in three,
and knob ranges that differ by surface. Three validation styles coexisted
(serde strict, serde permissive, hand-rolled from `Value`), and the LTX
request had no typed struct. Plan E1 vendored the upstream contracts under
`docs/serve/specs/`; E3 will build a parameter x surface x model
certification matrix and needs a machine-readable list of every parameter.

## Decision

1. **Types in `fastvideo-protocol::params`** (no cycle: every adapter and
   `fv-serve` already depend on the protocol crate). A `ParamSpec` is one
   submit endpoint of one surface for one served model: `surface`,
   `endpoint`, the client-side `model` name, the `engine_model` it resolves
   to, the surface's unknown-field policy, and one `Param` per field: wire
   name, JSON type, required, nullable, default, `domain` (what this server
   accepts for this model), `wire` (the wider domain the parser enforces
   when the model narrows it; the rest is negotiation's), `support`
   (`honoured`, `no_op`, `refused`, `gap` with its `GapId`), `maps_to` (the
   `GenerationRequest` field), `upstream` (the vendored contract) and a
   note. `Domain` is a closed set (int / number bounds, a set, the frame
   grid, a text length, an item count, the canvas, an aspect range, a media
   URL) with `contains`, `valid_example` and `invalid_examples`, so E3 can
   generate a valid and an invalid case for every parameter.
2. **One constant per upstream limit** in `params::rules`, one module per
   contract (`h3`, `openai_videos`, `minimax`, `ltx`, `fal_h3`, `fal_wan`,
   `fal_director`, `fast_h3`, `reactor`), each citing its vendored file.
   Limits that are facts of the model (the H3 4 / 5 / 15 s range, the H3
   reference counts) are derived from `fastvideo_models` instead of
   restated. Adapters validate with these constants and keep their own
   error wording; the messages are formatted from the constants, so the
   text is unchanged.
3. **Specs are derived**, never written per model: each adapter builds its
   surface's specs from the rules, its own wire enums and the served
   model's `ModelCaps` (the family manifests' layer 1 and
   `[variants.serving]`, projected by the engine). fal's specs are read from
   the JSON Schema the endpoint is served with (`/fal/schema`, already
   narrowed to the caps and tested against the parser).
4. **Served at `GET /fv/v1/params`** (`{"object": "fv.param_specs",
   "specs": [...]}`, native auth). `/fv/v1/capabilities` is unchanged (its
   golden is byte-identical).
5. **The LTX bodies are typed** (`TextToVideoRequest`,
   `ImageToVideoRequest`, `AudioToVideoRequest`, `EditVideoRequest`,
   `ExtendVideoRequest`), field for field from `ltx/openapi.json`; the
   parser reads the JSON into them in the order the errors are reported
   (so every refusal keeps its message), and the request is built from the
   typed value.

Enforced by tests:

- `fastvideo-protocol/tests/param_scan.rs`: no numeric limit literal in the
  adapters' request-validation sources (a `const` with a numeric
  initializer, a range or comparison with a literal of 2 or more), with a
  five-entry allow-list of service settings and JSON-precision constants;
- `fastvideo-protocol/tests/param_rules.rs`: the rules equal the vendored
  specs (FastVideo, OpenAI, MiniMax, LTX, 14 fal endpoints, the director
  AsyncAPI); where this server differs on purpose, the upstream value is
  pinned next to ours;
- `params.rs` agreement tests in each adapter (`parse_agreement`): the
  parser accepts each parse-time domain's valid example and refuses every
  invalid one naming the field;
- `ltxapi::body` tests: the typed bodies have exactly the OpenAPI
  properties and `required` lists;
- `fastvideo-serve/tests/param_spec_golden.rs`: the specs of every shipped
  config (`goldens/param_specs.json`, each distinct spec once) and their
  examples' consistency.

## Consequences

- A new limit, or a changed one, is a one-line change in `params::rules`;
  the spec, the parser, the fal schema and the golden follow. The scan
  refuses a literal added to an adapter.
- Surfaces still differ where their upstreams do (fal pro rates, MiniMax
  4 s, OpenAI 5 s); that is now visible in one file.
- One behaviour change, where two places disagreed: `/v1/videos`
  `guidance_scale_2` is bounded to `0..=20` as FastVideo's schema says (it
  was only non-negative). Every other valid request and every error keep
  their status, code and message.
- The golden is 1.1 MB (182 distinct specs over 36 configs); a change shows
  as line diffs, one parameter per line.

## Not in this change

- The director `configure` / `prompt` and Reactor command specs (the
  director's limits moved to `rules::fal_director`; Reactor's command tables
  were already declarative), the certification matrix and its TSV: E3.
- `schemars` schemas for native and director (deferred from E1): the native
  request is described by its `ParamSpec`; generating JSON Schema from the
  Rust types adds a dependency to `fastvideo-protocol` / `fastvideo-serve`
  and is left to E3 with the conformance generator.
- Model-side ranges for knobs: `KnobCaps` says only whether a knob is
  honoured, so step and guidance ranges are surface rules. Proposed
  manifest fields (a follow-up to `fastvideo-manifest::family`):
  `caps.knobs.steps = {min, max}`, `caps.knobs.guidance = {min, max}`, and
  `[variants.serving] seconds_min` for the FastVideo-compatible 5 s H3
  floor.
