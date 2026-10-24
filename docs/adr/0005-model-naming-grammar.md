# ADR-0005: Model naming grammar

- Status: Accepted
- Date: 2026-10-23
- Decision-id: FVID-2026-10-23-model-naming-grammar
- Crate: `fastvideo-manifest`, module `name`

## Context

The Gia Labs Registry (`zaitrarrio/model-registry-rs`) names every published
model with one string that says what it is and how it differs from its
parent: a vendor release is `vendor/model`, a fork is `user/model/variant`
with `variant = distillation:steps:encoder`. The registry's work order
WO-001 pinned the grammar; the owner's decision of 2026-10-23
(`docs/gaps/2026-10-22-architecture-review.md` §7.2 layer 1, §7.11; the
registry's `decision-log.md`) moved its implementation into this repository
as the pure crate `fastvideo-manifest`, because the runtime is the first
consumer of a published model and must agree with the registry on every
byte. Once a model is published its name is public: the grammar is a pinned
contract (review §7.10).

## Decision

The grammar is implemented verbatim as WO-001 pins it:

```text
name        = vendor-ref / fork-ref
vendor-ref  = namespace "/" model [ "@" rev ]
fork-ref    = namespace "/" model "/" variant [ "@" rev ]
namespace   = seg            ; 2..39 chars; reserved: gia, admin, api, registry, official
model       = seg-dot        ; 1..64 chars, may contain "." (wan2.2-t2v)
seg         = lower-alnum *( lower-alnum / "-" )        ; no leading/trailing "-", no "--"
seg-dot     = lower-alnum *( lower-alnum / "-" / "." )  ; no "..", no leading/trailing "." or "-"
variant     = distillation ":" steps ":" encoder
distillation= "base" / "dmd" / "lcm" / "cm" / "sf" / "adv"      ; vocabulary v1
steps       = 1*5DIGIT       ; 1..=1000, no leading zeros
encoder     = "bf16" / "fp16" / "fp8" / "int8" / "nf4"          ; vocabulary v1
rev         = 12LOWERHEX     ; first 12 hex digits of the manifest's canonical digest (ADR-0007)
```

- `ModelName` is an enum `Vendor { namespace, model, rev }` /
  `Fork { namespace, model, variant, rev }` with `FromStr` and `Display`;
  `Display` is the canonical form and `parse(display(n)) == n` for every
  valid name. It serialises as its string in serde (so `model.toml` and the
  registry API carry the string, not a struct).
- `NameError` has one variant per failure kind, and `NameError::kind()`
  names it; the fixtures use those names. The kinds: `Empty`,
  `EmptySegment`, `TrailingSlash`, `TooFewSegments`, `TooManySegments`,
  `Uppercase`, `InvalidChar`, `BadHyphen`, `DoubleDot`, `BadDot`,
  `NamespaceTooShort`, `OverLength`, `ReservedNamespace`, `BadVariant`,
  `UnknownDistillation`, `BadSteps`, `UnknownEncoder`, `BadRev`.
- Reserved namespaces are whole words (`gia` is reserved, `gia-labs` is
  not). Uppercase anywhere is `Uppercase`, never silently lowercased.
- `seg-dot` is read literally: it forbids `..` and a leading or trailing `.`
  or `-`, and (unlike `seg`) does not forbid `--`. If the registry owner
  intended `--` to be forbidden in model ids too, that is a one-line change
  plus a fixture line, made before the first publish.
- The vocabularies (`Distillation`, `EncoderPrecision`) are closed enums;
  extending them is an ADR amendment, never a code-only change. The `rev`
  is a `Rev` newtype with `Rev::from_digest` and `matches_digest_hex`.

What the name does **not** carry (attention route, DiT quantisation,
decoder, adapters, stage count, resolution tiers) lives in the `[variant]`
table of `model.toml` (ADR-0006): the grammar stays the human-facing
identifier and is unchanged by the widenings.

## Consequences

- The registry depends on this crate for `gia name check` and never
  re-implements the grammar; a disagreement is a bug here.
- Every valid fixture name prints back byte-identical, so names can be used
  as map keys and file names without normalisation.
- Changing the grammar after the first publish is a breaking change for
  every published model; the fixtures are the regression suite that makes
  such a change visible.

## Tests

- `crates/fastvideo-manifest/tests/names.rs`:
  `valid_names_round_trip` (every line of `fixtures/names/valid.txt`),
  `invalid_names_fail_with_their_kind` (every line of
  `fixtures/names/invalid.txt`, 46 lines, every kind at least once, and no
  kind outside the enum), `reserved_namespaces_are_the_pinned_five`,
  `accessors_and_rev_helpers`, `serde_as_string`, and the property tests
  `parse_display_round_trip` and `display_is_canonical_and_stable`
  (proptest, 512 cases each, over generated namespaces, model ids,
  variants and revs).
- `crates/fastvideo-manifest/src/name.rs` unit tests: `kinds`,
  `vendor_and_fork_round_trip`, `rev_from_digest`.
