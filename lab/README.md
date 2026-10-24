# lab: gold standards per served variant

`lab/gold/<variant>.toml` is the gold standard of one served model: what it
is (the family-file id, and once published the registry name and `@rev` it
certifies), the tier and the GPU class the tier is defined on, the reference
cell its numbers refer to, the measured latency, quality and cost, when
`fv-lab` last validated it, and where every number came from. The design is
in `docs/gaps/2026-10-22-architecture-review.md` §7.5 and
`docs/adr/0009-gold-standard-files.md`. Types, validation, the directory
reader and the JSON Schema are in `crates/fastvideo-lab` (`gold.rs`, schema
`crates/fastvideo-lab/schemas/gold.v1.json`).

There is one file per variant id of `families/*.toml`, 30 in all. A test
fails when a variant has no gold file or a gold file has no variant
(`crates/fastvideo-lab/tests/gold_files.rs`).

## The rule: never invent a number

Every numeric field is optional, and the format enforces two things:

- **A number is cited.** A numeric field that is present is named in the
  `fields` of a `[[source]]` (a repository path plus the heading or table it
  is under, and the measurement date). Without one, the file does not parse.
- **A missing number is declared.** A numeric field that is absent is named
  by a `[[gap]]` with the reason (`no oracle run of this variant`, `one timed
  generation`, `measured with another decoder`). Without one, the file does
  not parse.

So "not yet measured" and "forgotten" can be told apart, and a reader can
always trace a value back to the run it came from. Where sources disagree,
the file keeps the most recent dated one and cites both: the older one goes
in a `[[source]]` with empty `fields` and a `note`. Measurements of the same
recipe on another GPU, canvas or configuration are cited the same way (with
a note) and never copied into the variant's fields.

The first files (plan D2) were filled from `docs/perf/`, `docs/oracle.md`,
`docs/techniques.md`, `artifacts/perf/` and the numbers that used to be in
the profile comments. They are medians **as the sources report them**: no
margin was added and nothing was rescaled. `cost.usd_per_clip` is the only
derived value: `usd_per_hr * latency.p50_s / 3600`, checked to 1 %.

## Shape

```toml
schema = 1

[model]
variant = "fasth3-4step-vsa"          # families/h3.toml id; also the file stem
family = "h3"
# registry_name = "minimax/h3/dmd:4:bf16"   # once published (no @rev here)
# rev = "0123456789ab"                       # the manifest rev this file certifies
profile = "h3/fasth3_4step_vsa"       # profiles/<profile>.toml, as the family file says

[tier]
name = "turbo"                         # draft | turbo | max (absent: untiered)
gpu = "RTX PRO 6000 Blackwell"
sm = 120
vram_gib = 96
offer = "runpod: ..."

[workload]                             # the reference cell the numbers refer to
width = 832
height = 480
frames = 124                           # or `seconds` for a stream
fps = 24
steps = 4
prompt_set = "..."
seeds = [7]
note = "..."
upstream = { target = "fasth3-4step-vsa", image = "...", rev = "e90be59" }

[latency]
measures = "gen"                       # denoise | gen | end_to_end | serve_job | stream
p50_s = 9.84
min_s = 9.75
max_s = 9.85
denoise_s = 7.07
decode_s = 2.71
load_s = 138.8
peak_gib = 60.3
# fps, ttff_s: streams

[quality]
cell = "..."                           # when the quality cell is not [workload]
oracle = "..."                         # the docs/oracle.md verdict
psnr_vs_upstream_min_db = 19.73        # the measured value, the floor until D3 sets a tolerance
lpips_vs_upstream_max = 0.236          # the measured value, the ceiling until D3 sets one
# sharpness_ratio, jitter_ratio: the gate's quality section against the baseline `gate` names
# byte_identical: docs/perf/determinism.md, when measured for this variant
gate = "..."

[cost]
usd_per_hr = 2.49
usd_per_clip = 0.00681

[last_validated]                       # empty until fv-lab has run
# commit = "fb1bce3"
# run = "artifacts/gold/fasth3-4step-vsa/fb1bce3.json"
# date = "2026-10-14"

[[source]]
fields = ["latency.p50_s", "..."]
path = "docs/perf/raw-inference.md"
heading = "RTX PRO 6000 (sm_120)"
date = "2026-09-30"
note = "..."

[[gap]]
field = "quality.sharpness_ratio"
why = "..."
```

The numeric fields are `crates/fastvideo-lab/src/gold.rs` `NUMERIC_FIELDS`;
the non-numeric fields a source may also cite are `CITABLE_FIELDS`
(`workload`, `quality.gate`, ...). `deny_unknown_fields` everywhere: a typo
is an error, not a silently ignored key.

## How `fv-lab` will update a file (plan D1, D3, D4)

- `fv-lab run gold <variant>` (D1) runs the file's `[workload]` on its
  `[tier]` GPU class, writes the run record (a superset of the registry's
  benchmark record) under `artifacts/gold/<variant>/<commit>.json`, and
  compares it with the file's numbers using the gate's tolerances
  (`scripts/gpu/gate-policy.toml`).
- On a pass it writes `[last_validated]` (`commit`, `run`, `date`) and
  replaces each measured field with the run's value, its `[[source]]` with
  the run record, and closes the `[[gap]]` entries the run measured. It
  never writes a number the run did not measure.
- `fv-lab report tiers` (D3) joins the gold files with the run records into
  `docs/perf/tiers.md`; most of today's reference cells were measured at the
  smallest canvas (480p / 720p), and D3's gold runs move them to the serving
  defaults.
- The CI bridge (D4) checks that every file parses and that
  `last_validated.commit` is an ancestor of HEAD no older than N days.
- When a variant is published, `model.registry_name` and `model.rev` are
  set: the file then certifies that one manifest digest, and the registry's
  Verified badge points at its `last_validated.run`.

## Profiles carry no numbers

The comments of `profiles/**/*.toml` say what a technique does and why it is
on. Measurements of a profile live in its variant's gold file or in
`docs/perf` / `docs/techniques.md`, and the comment points there.
`crates/fastvideo-lab/tests/profiles.rs` fails when a profile comment
carries a measurement-looking number (seconds, ms, dB, GiB, %, `1.5x`,
`LPIPS 0.5`, `$2.09`); settings (`tau 1.0`, `sparsity 0.925`), sizes
(`832x480`), card classes (`a 96 GB card`) and dates pass.

## Not here yet

- `lab/tiers.toml` (per tier and family: GPU class, offer query, `$/hr` cap,
  wall cap, disk, image flavour, stages, cells) and `lab/cells/*.toml`:
  the wall caps, stages and cells are `fv-lab`'s run plan, which no
  document defines today. Plan D1 writes them with the runner; D3 derives
  the tier table from one run set.
