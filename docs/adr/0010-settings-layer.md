# ADR-0010: The settings layer: one declarative list of `FASTVIDEO_*` settings

- Status: Accepted (first half of plan B3; the `Settings` snapshot passed to
  `load` and the `model_spec`s of Wan and the small families follow in a
  second PR, see "Not in this change")
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-settings-layer
- Code: `crates/fastvideo-models/src/techniques/settings/` (`mod.rs` the
  readers, `table.rs` the list, `parse.rs` the shared parse rules, `doc.rs`
  the renderer, `workspace_check.rs` the acceptance tests); re-exported as
  `fastvideo_models::settings`
- Doc: `docs/scope.md` "Configurations" (generated region)

## Context

The architecture review (`docs/gaps/2026-10-22-architecture-review.md` §4.2,
§7.3 item 3) counted 233 `FASTVIDEO_*` names in the crates, two readers
(`techniques::settings::var`, the env > profile > default reader, and the
runtime's `envflag` getters on top of it) and 111 direct `std::env::var`
reads that bypass both. `docs/scope.md` documented about fifty of them by
hand. A name could be added, renamed or given a second parse anywhere, and
nothing listed what exists.

The non-negotiable for this change is byte identity: every variable keeps
its name, its parse (truthy set, number format, trimming, default when unset
or unparsable), its read time (once per process, per load, per generate, per
call) and its precedence against technique profiles.

The census (the PR description has it per name, with every read site):
222 real names (the other 11 of the review's 233 are test scaffolding,
prefixes used in `starts_with`, and Rust identifiers that contain the
string). 112 direct literal reads in 42 files, plus 27 indirect ones: a
`const` name passed to `std::env::var` (`BUDGET_ENV`, `dit_tree::ENV`,
`PLUG_ROOT_ENV`, `UPSAMPLER_ENV`, `MoeResidency`'s `ENV`, ...), helpers that
take a name (`env_usize`, `env_secs`, `env_or`, `env_f64`), and lookup
closures (`|k| std::env::var(k).ok()` in SANA-Video and `fv-gpucheck`). No
`option_env!` and no `format!`-built names.

## Decision

### Where the list lives

In `fastvideo-models` (`techniques::settings`), not in a new crate.
`fastvideo-models` depends only on `fastvideo-ops`, and every crate that
reads a setting (`fastvideo-cudarc`, `-core`, `-loader`, `-gpucheck`,
`-engine-service`) already depends on it, so the list and both readers sit
below all of them without a cycle. A separate dependency-free crate would add
a workspace member for no gain; it becomes worth it only if `fastvideo-ops`
ever needs a setting.

### One row per name

`SETTINGS` is a sorted `const` array of `Setting { name, reader, parse,
rule, default, family, seam, timing, doc }`:

- `reader`: `Layered` (`settings::var` / `is_set`: environment, then the
  profile table, then the caller's default), `Env` (`settings::env_var` /
  `env_var_os`: exactly `std::env::var` / `var_os`), or `Reserved` (a parser
  and a `const` exist, nothing reads it: `FASTVIDEO_NVFP4_COSMOS_STEPS`,
  `FASTVIDEO_NVFP4_LTX_VIDEO_FFN`);
- `parse`: `Flag`, `Usize`, `F32`, `F64`, `Keyword` (the shared rules of the
  runtime's typed getters) or `Text` (parsed at the call site, as `rule`
  says);
- `default`: `Unset`, `Value(v)` (setting `v` explicitly changes nothing) or
  `Computed(how)` (device, model, request or another setting decides);
- `family`, `seam` (a technique seam from `docs/techniques.md`, or the
  subsystem), `timing` (`Start`, `Process`, `Load`, `Request`, `Call`), and
  the doc string `docs/scope.md` shows.

The rows describe the readers; they do not drive them. Call sites keep
their string literals (`bool_flag("FASTVIDEO_WAN_FUSE", true)`,
`settings::env_var("FASTVIDEO_WAN_VAE")`): searching for a name still finds
its readers, and moving every read behind a typed key would have rewritten
about 400 sites in one structure change. The tests below make the list
authoritative instead.

### Routing: two readers, each name keeps its own

Every direct read became `settings::env_var(name)` / `env_var_os(name)` at
the same call site; the indirect forms call the same functions inside their
helpers and closures. These functions are `std::env::var` / `var_os`, so the
value, the parse that follows, the call time and the precedence are the ones
the code had. None of these names was moved to the layered reader: that
would let a profile's `[env]` entry (or a technique) set it, a behaviour
change that gets its own byte-identity cell when someone wants it.

The layered reads were already `settings::var` / `is_set` and the runtime's
`envflag` getters; they are unchanged except that `bool_flag`, `usize_flag`,
`f32_flag`, `f64_flag` and `string_flag` now call `settings::parse::{flag,
usize_or, f32_or, f64_or, lower_or}`, the same expressions moved into
`fastvideo-models` so the table's tests check the code the getters run.

### What the tests enforce (the acceptance criterion)

`workspace_check.rs` scans every `.rs` file under `crates/` (comments
stripped, string literals located) and fails on:

1. `std::env::var("FASTVIDEO_...")` / `var_os` anywhere outside the settings
   module;
2. in the inference crates, a `std::env::var` / `var_os` whose argument is
   not a string literal, except eight reviewed sites, none of which reads a
   `FASTVIDEO_*` setting (`INDIRECT_ALLOWED`: an env save/restore guard, the
   process-plan check that compares before it sets, legacy `WAN22_*`
   aliases, `FV_*` and `XDG`/`HOME` names);
3. a `FASTVIDEO_*` name in any string literal that is not a row
   (`FASTVIDEO_TEST_*` exempt), and a row no source names;
4. a typed getter whose parse, literal default or reader disagrees with its
   row, `settings::var` on an `Env` row, `env_var` on a `Layered` row
   (except six reviewed sites: tests that save and restore a variable, and
   two readers that have always read the environment only);
5. `docs/scope.md`'s generated region differing from `render_markdown()`
   (the failure prints the region to paste).

Off-identity tests (`table.rs`): for every typed row with a `Value`
default, unset and set-to-the-default parse equal through the layered reader;
the same for the twenty text rows whose parser lives in `fastvideo-models`
(`ltx2::hq::requested`, `nvfp4::requested`, `MoeResidency::parse`, ...);
truth tables of `flag` (both polarities), the number parsers and `lower_or`;
env > profile > default for `FASTVIDEO_H3_QUANT`, `BF16_ACT`, `DIT_OFFLOAD`,
`DIT_OFFLOAD_LOOKAHEAD`, `FLASH_KERNEL` (H3 profile) and `LTX_OFFLOAD` (LTX
profile); every name a compiled-in profile or a `[kernels]` op installs is a
`Layered` row (an `Env` row there would make the profile entry silently do
nothing); an `Env` row does not see a model scope's table.

### Read timing

Unchanged by construction: no read moved out of its function, no cached
value was introduced or removed, and `env_var` does not touch the profile
table (so it never initializes the lazily loaded profile, which a
`settings::var` call would). The `timing` column records what the code does.

## Consequences

- A new setting is a row first; a reader of an undeclared name fails
  `cargo test -p fastvideo-models`. `docs/scope.md` lists all 222.
- The census surfaced three quirks, kept as they are and now documented in
  the rows: `FASTVIDEO_INJECT_COND` is on only for `1` in H3 and on unless
  `0` in LTX-2; `FASTVIDEO_VSA` is a layered default-off flag in the runtime
  but the `fastvideo` CLI's VSA-id check reads `1` / `true` from the
  environment only; `fv-gpucheck wan` reports `FASTVIDEO_WAN_COND_CACHE` from
  the environment even when a profile set it. Two names are reserved
  (declared, never read).
- 108 rows are `Env` (112 `Layered`, 2 `Reserved`): a technique profile
  cannot set them. Promoting one is a reviewed behaviour change.

## Not in this change (plan B3, second PR)

- **A `Settings` snapshot passed to the H3 / LTX-2 / Wan `load` paths.**
  Every load-time read would have to move from the readers' own functions
  into a struct captured at load start and threaded through `load` and the
  helpers it calls (the H3 loader alone is about 830 lines, plan B6 splits
  it). The snapshot is identical in timing only for reads that happen inside
  `load`; reads cached once per process and reads per step or per generate
  must stay where they are. That is a loader-signature change per family
  with its own GPU cell, not part of making the list.
- **`model_spec` for Wan and the small families.** `registry::model_spec`
  also decides which `[pipeline].model` a profile may name; adding `wan`
  makes Wan profiles loadable, which is a feature with its own capability
  list and settings mapping (today the precision techniques map to H3 or
  LTX names only).

## Status of the evidence

CPU: the tests above on the build pod (numbers in the PR). GPU: the
coordinator's `techniques` byte-identity cell (H3 and LTX profiles, env and
profile arms); Wan has no trees on the EU volume and relies on the routing
being `std::env::var` at the same call sites.
