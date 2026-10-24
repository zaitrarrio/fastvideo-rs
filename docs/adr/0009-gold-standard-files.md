# ADR-0009: Gold standard files per served variant

- Status: Accepted
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-gold-standard-files
- Crate: `fastvideo-lab`, module `gold` (and `profiles`, the comment lint);
  schema `crates/fastvideo-lab/schemas/gold.v1.json`; files `lab/gold/*.toml`

## Context

The architecture review (`docs/gaps/2026-10-22-architecture-review.md` §7.5)
proposes one validation driver, `fv-lab`, and a checked-in gold standard per
served model: the bounds a variant must meet on its GPU class, the reference
cell they refer to, and the last time they were validated. The family
manifest (ADR-0008, §7.2) names it per variant (`gold =
"lab/gold/<variant>.toml"`). Plan D2 writes the format and the first files
from the numbers the repository already holds; D1 writes `fv-lab`, D3 the
tier table, D4 the CI bridge.

Today those numbers are scattered: `docs/perf/*.md`, `docs/oracle.md`,
`docs/techniques.md`, `artifacts/perf/` and the first lines of about twenty
`profiles/**/*.toml` comments, which go stale with every run. Most of them
were measured at the smallest canvas, many for configurations close to but
not equal to a served variant (another decoder, another DiT precision, one
prompt), and several variants have no number at all.

## Decision

### Format

One TOML file per variant id of `families/*.toml`, named
`lab/gold/<variant>.toml`, with `schema = 1` and the tables `[model]`
(variant, family, optional `registry_name` without `@rev`, optional `rev` of
12 hex digits, the technique profile), `[tier]` (tier name as the family
file says, GPU class, `sm`, `vram_gib`, offer), `[workload]` (the reference
cell: size, frames or seconds, fps, steps, prompt set, seeds, the upstream
reference), `[latency]` (`measures` says what the times are: `denoise`,
`gen`, `end_to_end`, `serve_job`, `stream`; `p50_s`, `min_s`, `max_s`,
`denoise_s`, `decode_s`, `load_s`, `peak_gib`, `fps`, `ttff_s`),
`[quality]` (the `docs/oracle.md` verdict, `psnr_vs_upstream_min_db`,
`lpips_vs_upstream_max`, the gate's `sharpness_ratio` / `jitter_ratio`,
`byte_identical`, the gate verdict in words), `[cost]` (`usd_per_hr`,
`usd_per_clip`), `[last_validated]` (commit, run record, date) and the
arrays `[[source]]` and `[[gap]]`. The types live in a new library crate,
`fastvideo-lab`, which plan D1 extends with the `fv-lab` binary; it depends
on `fastvideo-manifest` for `FamilyId`, `Tier` and `ModelName`, so a gold
file and a family file cannot disagree on what a tier or a name is.

### Choice 1: never invent, enforced by the parser

Every numeric field is optional. A present number must be named in the
`fields` of a `[[source]]` (repository path, heading, date); an absent one
must be named by a `[[gap]]` with a reason. `Gold::validate` rejects a file
that breaks either rule, and a `[[source]]` that cites a number the file
does not carry. The first files hold the sources' values as reported
(medians where the sources give medians): no margins, no rescaling, no
extrapolation across canvases or GPUs. Where sources disagree, the most
recent dated one is kept and both are cited; a measurement of a nearby
configuration (another decoder, another precision, another GPU) is cited in
a note and not copied into the variant's fields.

### Choice 2: measured values, not invented bounds

The review's sketch has `*_max` bounds with margins. No document defines
those margins, so the first files carry the measured values, and the field
names say which side is the bound (`psnr_vs_upstream_min_db`,
`lpips_vs_upstream_max`). `fv-lab` compares a new run against them with the
gate's tolerances (`scripts/gpu/gate-policy.toml`), which are already the
repository's rule; an explicit margin per field is a D3 decision with its
own data.

### Choice 3: profiles carry no numbers

The measurement lines of the profile comments move into the gold files (for
served variants) or stay where they already are in `docs/techniques.md` (for
matrix arms no variant serves); each comment now points there and keeps the
prose that says why a technique is on. `profiles::measurement_claims` finds
measurement-looking numbers in TOML comments (a number with a unit of time,
memory, dB or %, a speed-up `1.5x`, a metric name followed by a number, a
price) and lets settings, sizes, ranges, dates and card classes pass. The
profiles' keys and values do not change: the GPU cells byte-compare them.

### Choice 4: the cost is the class's price, dated

`cost.usd_per_hr` is the most recent dated price of the tier's GPU class
(RTX PRO 6000 at $2.49/hr in EU-RO-1, 2026-10-14), with the older prices
cited; `usd_per_clip = usd_per_hr * p50_s / 3600` is the one derived value
and is checked to 1 %. A price is a property of the class at a date, not of
the run that measured the latency, so it is not rescaled per source.

### Choice 5: the tier's GPU, and the served configuration or its exact setting (review of #151)

Two rules decide whether a measured number may sit in a variant's fields.
A number measured on a GPU other than the tier's is a gap; its values stay
in a cited `[[source]]` note and in the gap's reason ("measured on H100
only; D3 re-measures on the tier GPU"). A number measured under a setting
that is not the variant's served default stays in the fields only when
`workload.note` names the exact setting that reproduces it; an environment
setting also goes in `workload.env`, which the parser checks the note names,
so `fv-lab` reruns the cell exactly. When the setting cannot be named
exactly, the number is a gap. In the first files this applies to the H3
runs from before 2026-10-14 (`FASTVIDEO_H3_VAE_MXFP8=0`, the bf16 video
decode that was the default then) and to the Wan TAEHV arms
(`FASTVIDEO_WAN_QUANT=mxfp8`). Every `[[source]]` names its heading (a
Markdown heading or a top-level JSON key, checked to exist) and a date (the
measurement date, or the cited document's last commit where the source
states none).

### Not decided here

`lab/tiers.toml` and `lab/cells/*.toml` (§7.5): their wall caps, stages and
cells are `fv-lab`'s run plan, which no document defines today. Plan D1
writes them with the runner.

## Consequences

- Every served variant has a gold file, and the gaps are explicit: 17 of 30
  have a p50 time on the tier GPU, 1 a PSNR against upstream (none an
  LPIPS), none a `[last_validated]`; 11 carry only the class price. The gaps are D3's GPU work list.
- A new variant in a family file fails CI until it has a gold file, at
  least an all-gaps one with its workload and sources.
- `fastvideo-lab` is CPU only and adds no package to the lockfile beyond
  itself.
- Enforced by: `crates/fastvideo-lab/src/gold.rs` unit tests (every number
  cited, every gap declared, sources carry a heading and a date, `workload.env`
  named in the note, paths named in errors, cost derivation, stream fields),
  `tests/gold_files.rs` (every file parses and validates; the set of gold
  files equals the set of family variant ids, both directions; family, tier
  and profile agree with the family file; the tier's GPU meets the variant's
  floor; every cited path exists; every cited heading exists in its file),
  `tests/profiles.rs` (no
  measurement in a profile comment), `tests/schema.rs`
  (`schemas/gold.v1.json` drift).
