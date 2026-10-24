# ADR-0021: One disk prompt cache

- Status: Accepted (plan B4c, `docs/plan/2026-10-23-execution-plan.md`)
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-prompt-cache
- Source: `docs/gaps/2026-10-22-architecture-review.md` §3 ("three separate
  disk prompt caches"), §7.3 item 4 ("one disk prompt cache")
- Code: `crates/fastvideo-cudarc/src/rt/prompt_cache/` (`mod.rs`: format,
  store, load, protocol; `key.rs`: the key; `identity.rs`: file
  identities; `tokens_width.rs`: the `[tokens, width]` entry H3 and Wan
  store); the family adapters `h3/text_cache.rs`, `ltx2/text_cache.rs` and
  Wan's `text_cache_key` / `encode_one` in `wan/pipeline.rs`
- Tests: `rt/prompt_cache/equality.rs` (old against new) with the pre-B4c
  caches verbatim in `rt/prompt_cache/legacy.rs`; the families' existing
  cache tests (`h3::text_cache`, `ltx2::text_cache`, `h3::slim`,
  `ltx2::pipeline` `the_cache_computes_once_per_prompt...`,
  `tests/determinism.rs` `a_cached_conditioning_reads_back_bit_for_bit`);
  GPU: `scripts/gpu/runpod-matrix.sh b4c`

Line references to "pre-B4c" code are to main `e18ec010`, where the copies
lived; `legacy.rs` holds them verbatim (formatted as nested modules; Wan's
key method as a free function of what it read from `self`; see its header).

## Context

Three families cached their text conditioning on disk, each its own way:

| copy (pre-B4c) | key | store | used by |
|---|---|---|---|
| H3 `h3/text_cache.rs:35-278` | `cache_key` `:141-154` | `.h3text`, `serialize` / `deserialize` `:172-213`, `load` `:217-221`, `store` `:226-249`, `get_or_compute` `:252-278` | `h3/text.rs:278-349` (`encode_prompt_with_at`), `:360-405` (recovered 8B), `h3/slim.rs` tests |
| Wan `wan/pipeline.rs:728-799` | `text_cache_key` (its own derivation and file identity) | H3's `get_or_compute` at width 4096 (`encode_one`, `:701-726`) | `encode_text` `:803-849` |
| LTX-2 `ltx2/text_cache.rs:36-255` | `cache_key` `:116-134`, `weights_identity` `:61-113` | `.ltx2text`, `TextCache::store` `:165-200`, `load` `:204-254`; protocol `cached_or` `ltx2/pipeline.rs:2667-2684` | `TextEncoder::encode` `ltx2/pipeline.rs:2599-2643` |

They are one idea — hash what the conditioning depends on, store the result
under that name, check it on the way back, never fail a run because of the
cache — but what is on disk differs (two file layouts, three key
derivations, two file identities). Caches written by today's code exist on
pods, in serve state directories and on developers' machines. The plan's
first rule is byte identity (§1): unifying them may not move an output bit,
nor a key, nor a byte of a written entry.

## Decision

### Where it lives: `fastvideo_cudarc::rt::prompt_cache`

Not in `fastvideo-models` (beside B4a's `techniques::step_cache`), because:

- every key and seal is SHA-256 (`sha2`), a dependency of fastvideo-cudarc
  only; fastvideo-models has none, and B4c adds no dependency;
- the identities read weight files (LTX, Wan) or a lazily opened
  checkpoint (`fastvideo_loader::LazyStore`, H3), and the cached values are
  device tensors read back to the host: both cudarc's business;
- every caller is a cudarc family pipeline, and `rt` is where B1 put what
  every family shares (`rt/mod.rs`). It needs no device: it compiles and is
  tested without the `cuda` feature.

What stays with each family is only what is that family's: the tensors its
identity reads (`h3::text_cache::tap_keys`, `encoder_identity`), its
default directory (`ltx2::text_cache::default_dir`,
`fastvideo-core` `wan_text_cache_dir`), its value type and the arguments of
its key.

### Key

One derivation, `KeyRecipe::key` (`key.rs`): SHA-256 over length-prefixed
fields (`len as u64 LE | bytes`):

```text
domain | normalise(prompt) | sha256(tokenizer.json) | length as u64 | identity fields...
```

All three pre-B4c keys are exactly this (recomputed independently with
Python `hashlib` for the pinned values below). What differs is the recipe
and the arguments:

| | H3 (`H3_KEY`) | Wan (`WAN_KEY`) | LTX-2 (`LTX2_KEY`) |
|---|---|---|---|
| domain | `H3TEXT01` (the entry magic, `h3/text_cache.rs:36, :148`) | `WANTEXT1` (`wan/pipeline.rs:788`) | `fastvideo ltx2 text conditioning v1` (`ltx2/text_cache.rs:125`) |
| prompt | verbatim (`:149`) | verbatim (`:789`) | `trim()` (`:127`; the pipeline strips before tokenising) |
| tokenizer | `sha256(<root>/tokenizer/tokenizer.json)` (`h3/text.rs:322-333`) | `sha256(cfg.tokenizer_path)`, empty bytes for no path, memoised per path per process (`wan/pipeline.rs:761-785`) | `sha256(<text root>/tokenizer/tokenizer.json)` (`:128`, `ltx2/pipeline.rs:2402-2403`) |
| length | the tap, 50 (24 for recovered 8B) (`:151`) | the DiT's `text_len` (`:791`) | `max_sequence_length` (`:129`) |
| identity fields | one: `encoder_identity` — every tap tensor's name, dtype, shape, plus 4 KiB of the embedding and the first and last input norm (`h3/text_cache.rs:110-138`); FP8 rows: `sha256(identity ‖ "/fp8-rows")` (`h3/text.rs:328-331`); recovered 8B: `sha256("recovered-8b-tap{tap}")` (`:388`) | one: `identity::files_head(text_encoder/)` — names, sizes and one `read` of up to 256 KiB per `.safetensors`, *unframed*, never failing (`wan/pipeline.rs:730-758`) | three: the encoder kind (`gemma3-12b`, `gemma4-12b-hf`, each optionally `/per-token-rms`: encoder and feature normalisation, `ltx2/pipeline.rs:2358-2370`), then `identity::files_head_tail` of Gemma (`model-` shards when present, `gemma/` on 2.3 trees, `:2400-2414`) and of the connectors (`:2415-2442`) — names, sizes, first and last MiB, framed (`ltx2/text_cache.rs:61-113`) |
| key form | `[u8; 32]`, file stem in hex | `[u8; 32]`, hex | hex `String` (also reported as `text.key`) |

Precision is in the key only where it was: H3's FP8-rows suffix, LTX's
kind string. Two pre-existing gaps are carried over unchanged, because
closing them changes keys (decision log, follow-up):

- LTX `FASTVIDEO_LTX2_TEXT_FP8=1` (weight-only FP8 Gemma, "changes the
  conditioning numerics") is not in the LTX key: an entry written by an FP8
  run is served to a bf16 run of the same prompt and weights.
- Wan's UMT5 precision is not in the Wan key.

### On-disk format

Directory layout (unchanged): one flat directory of `<hex key>.<ext>`
entries. Defaults, all unchanged: gpucheck H3 and Wan
`<clip dir>/../text-cache` (`gpucheck/src/h3_stage.rs:448-456`,
`wan_stage.rs:304-310`; `--text-cache <dir>`, `--no-text-cache`);
gpucheck LTX `--text-cache`, else `$FASTVIDEO_CACHE`, else
`$XDG_CACHE_HOME/fastvideo`, else `~/.cache/fastvideo`, under
`ltx2-text` (`ltx2/text_cache.rs` `default_dir`, also H3's joint LTX
refine, `h3/pipeline.rs:2791`); `fastvideo-core` Wan
`FASTVIDEO_WAN_TEXT_CACHE` (`off` / `0` / `false` / `none`), else
`$XDG_CACHE_HOME` or `~/.cache` + `fastvideo/wan-text`
(`core/src/generator.rs:402-413`); engine-service `<text_cache>/<h3 | ltx2
| wan>` (`engine-service/src/cuda/backend.rs:566-572, :698`), serve
`<state_dir>/text-cache` (`serve/src/app.rs:353`).

Every entry is `magic | header | ids | tensors | sha256(everything
before)`. The `Format` names each way the two files differ:

| parameter | `H3` (H3, Wan) | `LTX2` |
|---|---|---|
| magic, extension | `H3TEXT01`, `.h3text` | `FVLTX2T1`, `.ltx2text` |
| `Layout` | `TokensWidth`: `tokens u64 | width u64 | ids u32[] | data f32[tokens·width]`; tokens and width non-zero | `LenRealShaped { tensors: 2, max_rank: 8 }`: `len u32 | real u32 | ids u32[len] | 2 × (rank u32 | shape u64[rank] | data f32[∏ shape])`, video then audio |
| `Header` checked on a hit | `Width(width)` | `Real(real)` (and the padded ids) |
| `Finite` | `Reject`: any NaN / inf is a miss | `Accept` |
| `TempName` | `<key>.tmp<pid>` (`with_extension`) | `<key>.<pid>.part` |
| `Durability` | `fs::write`, no fsync | create, write, `sync_all` |
| `TempCleanup` on a failed write or rename | `Remove` | `Leave` |
| `ErrorText` | `<path>: <error>` (the directory, or the entry path) | `text cache <dir>: <error>` |
| `StoreLog` | `h3 text cache: not written (<error>)` (Wan's too: it stored through H3's code) | `ltx2 text cache: not stored: <error>` |

Writes are atomic: the whole entry goes to the temporary name and is
renamed onto the entry name, so a reader never sees half an entry under the
final name (a killed writer leaves only its temporary file, which no
lookup reads).

Versioning: the magic and the key domain. A changed layout or key gets a
new magic *and* a new domain: old entries then have other names (never
looked up) and, should a name match, the magic does not (a miss). B4c
changes neither, so no version moves.

### Eviction and size

None, as before: entries live until someone deletes them, and deleting any
of them (or the directory) at any time is safe (the next request misses).
Sizes: H3 `S × 5120 × 4` bytes (~700 KB at 35 tokens), Wan `S × 4096 × 4`,
LTX two padded `[1, 1024, 3840]` f32 contexts (~31 MB) per prompt. A
size cap or LRU is a behaviour change left out of B4c.

### Concurrency (two processes, one directory)

- Two processes have different temporary names (the pid) and `rename` is
  atomic within one file system: a reader sees no entry, the old one or a
  new whole one; the last rename wins. Encoders are deterministic for one
  key, so the writers normally write the same bytes.
- Two threads of one process share the temporary name. A store can then
  fail (the other thread renamed the file away; logged, the run continues)
  or rename a file another thread is still writing; the seal makes that
  entry a miss. Never a misread.
- Both are tested (`concurrent_writers_never_produce_a_misread`: eight
  writers with distinct or one shared temporary name, four readers, both
  formats). Unchanged from before; the per-thread race is not fixed here
  (temporary names are pinned by a fixture).

### How each family's encoder output maps in and out

| family | stored | a hit returns | a miss returns |
|---|---|---|---|
| H3 | the encoder's `[1, S, 5120]` hidden state read to the host (`hidden_state(..).host_cow()`), `Check { ids, Width(5120) }` | `CachedText` → `CudaTensor::from_vec(.., [1, S, 5120]).to_device()` (`h3/text.rs`) | the same host vector the store wrote, through the same `from_vec(..).to_device()` |
| Wan | UMT5 `[1, S, 4096]` read to the host, `Check { ids, Width(4096) }` | as H3, then `pad_prompt_embeds` to `text_len` | as H3 |
| LTX-2 | the two connector outputs read to the host (`host_cow`: bf16 storage widens exactly to f32), their shapes, `Check { padded ids, Real(real) }` | two host f32 `CudaTensor`s from the entry | the connectors' own device tensors |

LTX: the pipeline puts both through `canonical_contexts`
(`ltx2/pipeline.rs:2653`: host f32, uploaded as an f32 device tensor)
before the DiT. The techniques `ltx25-512p-sol` comment
(`runpod-matrix.sh`, "a conditioning-cache hit feeds the DiT the stored
f32 connector output instead of the freshly encoded device tensor, which
is not the same clip") was measured on 2026-09-27 (`e7acf3f5`), two days
before `canonical_contexts` (`63848af2`) was added to give a miss and a
hit one form. B4c changes neither path: a hit still returns exactly the
stored f32 values and a miss exactly the computed tensors, and both still
go through `canonical_contexts`. The techniques cell keeps
`--no-text-cache` and its `-cache` arm; the `b4c` LTX cell records
`pr-miss` against `pr-hit` (not gating), so whether a hit is the same
clip as a miss today is measured, not assumed.

### Compatibility rule

**Every existing cache directory is read exactly as before, per family**:
the same key (same file name), the same format (same magic, layout and
seal), the same acceptance checks. Nothing needed versioning, so nothing
is ignored by version. Files that are damaged, foreign (the other format
under this name, another magic), from another key, or made from other
token ids or another width / real count are a miss, as before. H3 and Wan
share a format and may share a directory (gpucheck's default
`text-cache`): their domains differ, so their names do; were a name to
match, the width check (5120 / 4096) misses.

Tested (`equality.rs`): legacy-written files read by the new code, new
files read by the legacy code, and new files byte-equal to legacy files
under the same names, for every family; every truncation, every single-bit
flip, trailing bytes and a set of resealed header lies judged identically
by both readers.

### What did not change

Every CLI flag (`--text-cache`, `--no-text-cache`), setting
(`FASTVIDEO_CACHE`, `FASTVIDEO_WAN_TEXT_CACHE`, `FASTVIDEO_LTX2_TEXT*`),
default path, report field (`text_cache` / `text.cache` `off` / `hit` /
`miss` / `disabled`, `text.key`, the `warm_text_is_a_cache_hit` check) and
log line; H3's `CacheStatus` (`disabled`) and LTX's `CacheOutcome` (now the
shared `Outcome`, `off`) keep their strings.

## Evidence

- **CPU** (`rt/prompt_cache/equality.rs`, 15 tests, against `legacy.rs`):
  keys equal for every family over prompts (empty, whitespace, unicode,
  5000 bytes), tokenizers, lengths and identities, with the pinned keys
  (H3 `86ae3a4c…`, LTX `b944238d…`, Wan `032905ee…`, all three recomputed
  in Python); `files_head_tail` around the 1 MiB probe (0, PROBE−1, PROBE,
  PROBE+1, 2·PROBE+5, 3·PROBE), prefixes and errors; Wan's identity over
  short, 256 KiB and longer files, a `.json`, a directory named
  `*.safetensors`, a missing directory; H3's `encoder_identity` over a
  lazily opened checkpoint; byte-equal files and cross reads; identical
  accept / reject of every mutation; the same hit / miss sequence, compute
  calls and files for each protocol (including a wrong-length compute and
  an empty prompt); the same error text and the same files left after a
  failed rename, a squatted temporary name and an uncreatable directory; a
  stale temporary file never read; concurrent writers.
- **Mutations** (each applied alone, fixtures rerun, all 18 caught): LTX
  prompt not trimmed; the length field as `u32`; tokenizer and length
  fields swapped; Wan's domain; identity fields unframed;
  `files_head_tail` without the tail below 2 MiB; `files_head` framed; LTX
  `Finite::Reject`; H3 `Finite::Accept`; `max_rank` 9; zero-token H3
  entries accepted; the seal not checked; LTX `real` as `u64`; LTX
  `TempCleanup::Remove`; LTX `ErrorText::EntryPath`; H3 `TempName::PidPart`;
  trailing bytes accepted; the header not checked on a hit.
- **GPU** (`scripts/gpu/runpod-matrix.sh b4c`, `FV_BASELINE_SHA` = the
  commit before B4c): per cell, the baseline binary and this build with the
  same argv and env, each over its own new cache directory, pass `-miss`
  (cold, writes) then `-hit` (reads); `compare-clips --off-identity` base
  against pr for each pass, each pass's `frames.sha256.json`; the cell
  fails unless every miss pass reports `"text_cache": "miss"` and every hit
  pass `"hit"` (`benchmark.json`) and the two arms' cache directories hold
  the same entries byte for byte. Cells: `b4c-h3-4stepvsa` (FastH3 4-step
  VSA, 480p / 5 s, auto encoder: resident FP8 on an RTX PRO 6000) and
  `b4c-ltx25-512p` (LTX-2.5 two-stage, 512p / 49 frames, streamed Gemma).
  Wan has no tree on the EU volume; its cache is the H3 format with its own
  key, both covered by the CPU fixtures.

## Consequences

- A new family's prompt cache is a `KeyRecipe` (a new domain), a `Format`
  (or an existing one) and a `Cached` value type; the protocol, the file
  handling and the checks are not written again. A new on-disk variant is a
  new named parameter with a fixture, never an edit of an existing format.
- `legacy.rs` is test-only and frozen.
- Follow-ups, each a key or behaviour change and so its own decision: the
  LTX FP8-Gemma and Wan UMT5-precision key gaps; a size cap; per-thread
  temporary names.
