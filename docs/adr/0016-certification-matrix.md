# ADR-0016: The parameter certification matrix

- Status: Accepted (plan E3)
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-cert-matrix
- Code: `crates/fastvideo-serve/tests/cert_matrix.rs` (the harness),
  `crates/fastvideo-serve/tests/common/mod.rs` (the shipped configs, shared
  with `param_spec_golden`), `fastvideo_fal::params::director_specs`,
  `fastvideo_reactor::commands::CommandTable::param_specs`
- Output: `docs/serve/certification.tsv`
- Doc: `docs/serve/params.md` "Certification matrix"

## Context

ADR-0012 gave every request parameter of every served (surface, model) a
`ParamSpec` with a `Domain` that can produce a valid and an invalid value.
The review (`docs/gaps/2026-10-22-architecture-review.md` §7.6 item 3)
asks for a parameter x surface x model matrix over every surface, driven
through the real server, emitted as a TSV, with a CI assertion that every
parameter has a valid and an invalid case. E2 left four pieces to it: the
caps-level checks of MiniMax, LTX and fal (E2 checked them at parse time),
the pairs of fields that are valid alone and refused together, the director
and Reactor control-channel messages, and the census findings
(docs/serve/params.md).

## Decision

1. **A test in `fastvideo-serve`, not a new crate.** The matrix needs the
   assembled router (`App::build`), the shipped configs, the fake engine
   serving their real caps with the CUDA pipelines' job checks, and the
   adapters' parsers; `fastvideo-serve`'s tests already build exactly that
   (`param_spec_golden`, `dimension_sweep`). A crate would duplicate the
   dev-dependencies and the config loader, and `scripts/serve/coverage.sh`
   (the adapter-tests workflow) and `scripts/serve/check.sh` already run
   every `fastvideo-serve` test, so the CI wiring is free.
2. **The specs are the served ones.** For each shipped config the harness
   reads `GET /fv/v1/params` from the running server; each distinct spec
   (the golden's key, `surface endpoint model @ engine_model`) is driven
   once, on the first config that serves it, and the row lists how many
   configs serve it. The Reactor tables of the configs that mount Reactor
   and the director's `configure` / `prompt` specs join them. These two are
   built by their crates (`CommandTable::param_specs`, `director_specs`)
   but are not served at `/fv/v1/params` (that endpoint lists submit
   endpoints; its golden is unchanged).
3. **Cases come from the domains.** A parameter's valid case is its
   domain's valid example (or the parser's wider domain's), with media as
   data URIs (the server stages inputs at submit), canvas sides with their
   partner, lists as one item; when the surface refuses it with the other
   fields of the base, the rest of a set (or a range's upper bound) is
   tried and the cell names the value accepted. Its invalid cases are every
   invalid example of the model's domain and of the parser's, a malformed
   media URL, and a wrong JSON type where the domain has no invalid value.
   Composite fields (MiniMax `content[]`, LTX `audio_uri(duration)`, the
   director's `script` / `entities`, `model`, MiniMax `callback_url`,
   native `end_s`) have explicit plans. A refused or gap parameter has no
   valid case: any value is its invalid case. A few fields need others to
   be read at all (an edit field its source video, an aspect its tier):
   these companions are applied with the value.
4. **The base.** Every case is applied to a base request: the spec's
   required fields plus the surface's minimal task inputs, the first
   candidate the server accepts on the spec's own model (a candidate the
   server routes to a tier companion is used only when none stays). The
   base is a row of its own.
5. **Drivers.** HTTP for the submit endpoints (in-process `oneshot` on the
   router, rotating API keys so per-key limits never refuse a case; an
   accepted job is checked against the engine's job checks and cancelled);
   the channel's parser for the director (`messages::parse`) and Reactor
   (`CommandTable::validate`), whose transport is a WebRTC data channel
   with nothing else to certify per parameter.
6. **Judgement, not enforcement.** A valid case must be accepted (and pass
   the engine's checks, on the spec's model); an invalid one must be
   refused with a 4xx naming the parameter: the wire's `param` (native,
   `/v1/videos`), `loc` (fal), the parse error's field (director, Reactor),
   or, where the surface has no such field (FastWan, MiniMax, LTX), the
   message. A disagreement is written to the row's `finding`; the test
   does not fail on it. Wire behaviour is changed by separate decisions
   (docs/serve/params.md lists the candidates). The test fails on a 5xx,
   a panic, an empty cell or drift from the committed TSV.
7. **The CI assertion** (`every_param_has_a_valid_and_an_invalid_case`,
   no server): the plan of every parameter of `goldens/param_specs.json`,
   the director and every Reactor mode has a valid case (or is refused, or
   has an empty domain on the model) and an invalid case. The plans are
   made without the exemption table; a parameter whose plan has no invalid
   case (it takes any JSON value) carries an exemption with its reason
   (`EXEMPT`). An exempt parameter that has an invalid case, or an
   exemption matching no parameter, is stale and fails
   (`a_stale_exemption_fails_the_assertion`). Every served spec must have
   a driver (`every_golden_spec_has_a_driver`, and the matrix asserts it
   on what each config serves); the matrix has a 10-minute wall limit.
8. **Output**: one row per (section, surface, endpoint, model,
   engine model, parameter), sections `base`, `param`, `unknown_field` (the
   surface's policy) and `pair`; no empty cell (`n/a (...)` with the
   reason). `FV_CERT_WRITE=1` regenerates it.

## Consequences

- A new parameter, a new served model or a changed limit shows as TSV
  rows in the PR that causes it; a parameter whose domain cannot produce an
  invalid value needs a reasoned exemption before CI passes.
- The matrix runs about 8,000 requests in about 10-25 s on the build pod
  (debug build); it adds no workflow, only test time to adapter-tests and
  `scripts/serve/check.sh`.
- The findings are visible per row; fixing one is a wire change with its
  own decision, and its TSV rows change with it.
- Not covered: LTX `/v1/*` (the same schemas as `/v2`, synchronous),
  `/v1/segments`, streams and WHIP ingest (their limits are in
  `params::rules` but they have no `ParamSpec` yet), real transports of the
  data channels, and upstream captures (plan E5).
