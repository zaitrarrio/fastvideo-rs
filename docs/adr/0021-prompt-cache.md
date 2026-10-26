# ADR-0021: One disk prompt cache

- Status: Accepted (plan B4c, `docs/plan/2026-10-23-execution-plan.md`);
  amended by B4c2 (the key gaps closed, section "B4c2"). The GPU gate
  families `runpod-matrix.sh b4c` / `b4c2` and their verdict script
  (`b4-verdict.sh`, its test) are retired with no tier since their gates
  passed (ADR-0019 §18; `git show b019ccb2:scripts/gpu/b4-verdict.sh`)
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-prompt-cache
- Source: `docs/gaps/2026-10-22-architecture-review.md` §3 ("three separate
  disk prompt caches"), §7.3 item 4 ("one disk prompt cache")
- Code: `crates/fastvideo-cudarc/src/rt/prompt_cache/` (`mod.rs`: format,
  store, load, protocol; `key.rs`: the key; `identity.rs`: file
  identities; `tokens_width.rs`: the `[tokens, width]` entry H3 and Wan
  store; since B4c2 `numerics.rs`: the numerics and weights-read key
  fields); the family adapters `h3/text_cache.rs`, `ltx2/text_cache.rs` and
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
kind string. The keys leave out several things the conditioning depends
on. All of these gaps predate B4c and are carried over unchanged: closing
any of them changes keys, so they are the follow-up B4c2 (decision log
`FVID-2026-10-24-prompt-cache-gaps-kept`), not part of this unification.
Line references are to this branch (B4c, `17e4a1dc`). **B4c2 closed all
six** (section "B4c2"); the recipes above are now `H3_KEY_V1`,
`WAN_KEY_V1` and `LTX2_KEY_V1`, kept for the tests.

#### Follow-ups: key gaps carried over

| # | gap | where | effect | severity |
|---|---|---|---|---|
| 1 | LTX FP8 Gemma (`FASTVIDEO_LTX2_TEXT_FP8=1`, weight-only FP8 rows, "changes the conditioning numerics") is not in the key: neither the kind string nor the Gemma identity, which is taken from `text_encoder/`, not from the `text_encoder_fp8/` tree that is read | read at `ltx2/pipeline.rs:2274`; key `:2385-2441` (`TextEncoder::key`), kind `:2343-2355` (`text_encoder_kind`); FP8 tree `:2282` (`open_gemma`) | an entry written by an FP8 run is served to a bf16 run of the same prompt and weights (and the reverse): **silent wrong output** | medium **closed (B4c2)** |
| 2 | H3 recovered-8B identity is a constant, `sha256("recovered-8b-tap24")`: the 8B base weights, the `ara.safetensors` LoRA and `conditioning_adapter.safetensors` are not fingerprinted | `h3/text_cache.rs:120` (`recovered_identity`), used at `h3/text.rs:388`; weights read at `h3/recovered_8b.rs:89-107` | a changed LoRA or adapter (or base) under the same tokenizer serves stale conditioning: **silent wrong output** | medium **closed (B4c2)** |
| 3 | Wan UMT5 GEMM compute type is not in the key: `FASTVIDEO_BF16` (`rt/bf16_gemm.rs:14`), `FASTVIDEO_TF32` (`rt/hopper.rs:16`) and the GPU architecture choose the GEMM path | key `wan/pipeline.rs:732` (`text_cache_key`) | entries from one compute type served to another: small numeric differences | low **closed (B4c2)** |
| 4 | H3 and LTX: `FASTVIDEO_BF16_ACT` (`rt/tensor.rs:66-67`), the global GEMM compute type (`FASTVIDEO_BF16`, `FASTVIDEO_TF32`) and the GPU architecture are not in the keys | H3 key `h3/text.rs:333`, LTX key `ltx2/pipeline.rs:2385` | small numeric differences between a hit and a fresh encode under other settings or on another card (a cache directory shared across cards) | low **closed (B4c2)** |
| 5 | LTX resident vs streamed Gemma share a key | `ltx2/pipeline.rs:2509-2524` (`compute`: `HiddenStack::encode_resident` vs `HiddenStack::encode`) | expected to be equal, **not proven**: if they differ, a hit replays whichever ran first | low until measured **closed (B4c2)** |
| 6 | The pre-quantised FP8 tree and quantise-at-load share a key (H3: the `/fp8-rows` identity is computed from `text_encoder/` whether the rows come from the tree or are quantised at load; LTX: as row 1) | H3 `h3/text.rs:291` (`open_tree`) and `:331`; `llm/prequant.rs:150` (`open_tree`) | correct only while the two paths are bit-equal (the tree is meant to be the load-time quantisation, written once) | low while bit-equal **closed (B4c2)** |

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

Versioning: the magic and the key domain. A changed layout gets a new
magic *and* a new domain: old entries then have other names (never looked
up) and, should a name match, the magic does not (a miss). A changed key
derivation alone gets a new domain (B4c2): every name moves, and the
layout, which the magic versions, is unchanged. B4c changes neither, so no
version moves.

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
  the same entries byte for byte; on H3 also unless the miss passes'
  `"text_encoder"` contains `fp8` and the hit passes' is `cache` (so a card
  where `auto` streams bf16 cannot pass without the `/fp8-rows` identity);
  and unless both base-vs-PR compares passed `--off-identity` (not skipped),
  every pass exited 0 and no temporary file is left. A cell named in
  `FV_B4C_CELLS` without a passing verdict fails the family
  (`b4c.verdict.json`). The verdict is `scripts/gpu/b4-verdict.sh`, held to
  stub run trees by `scripts/gpu/tests/b4-verdict.test.sh`.
  Cells: `b4c-h3-4stepvsa` (FastH3 4-step
  VSA, 480p / 5 s, auto encoder: resident FP8 on an RTX PRO 6000) and
  `b4c-ltx25-512p` (LTX-2.5 two-stage, 512p / 49 frames, streamed Gemma).
  Wan has no tree on the EU volume; its cache is the H3 format with its own
  key, both covered by the CPU fixtures.

## B4c2: the key gaps closed

Plan B4c2 (decision log `FVID-2026-10-24-prompt-cache-b4c2-*`). Rule: a
setting that can change the encoded tensors is in the key, as the value the
run actually resolved (a spurious miss costs one encode; a wrong hit is
silent wrong output). The derivation (`KeyRecipe::key`), the entry layout
and the magics are unchanged; each family has a new domain and more
identity fields (`rt/prompt_cache/numerics.rs` holds the shared ones:
`Numerics`, `WeightsRead`, `TextMode` and `EncodeSettings`, the four
fields H3 and LTX put after their encoder identity, in that order).

| # | gap | field added (key field text) | resolved where | test |
|---|---|---|---|---|
| 1 | LTX FP8 Gemma | `WeightsRead`: `weights=checkpoint` / `weights=fp8-rows/at-load` / `weights=fp8-rows/tree`, then the tree's `files_head_tail` (an empty field unless the tree is read) | the tree: `TextEncoder::gemma_tree` (`ltx2/pipeline.rs`, the pure `gemma_tree`), decided once per encoder from `FASTVIDEO_LTX2_TEXT_FP8` and `llm::prequant::resolve_tree`; `open_gemma` reads the tree it names, so the key and every open agree. The field: `gemma_weights_read(fp8, tree, mode)` (see "LTX FP8 with an explicit streamed residency") | `ltx2::text_cache` `the_key_is_stable_and_moves_with_every_input`, `ltx2::pipeline` `gemma_is_keyed_as_what_it_reads`; GPU `b4c2-ltx25-fp8` |
| 2 | H3 recovered-8B | `recovered_weights_identity`: `recovered-8b-tap{tap}`, then `files_head_tail` (name, size, first and last MiB, as LTX) of the 8B base, `ara.safetensors` and `conditioning_adapter.safetensors`, in that order. Not a full hash: a multi-GB read per prompt for no case the ends miss (a retrained LoRA or adapter changes the tensor bytes at both ends; a fine-tune of the base changes its tail) | `encode_prompt_recovered` (`h3/text.rs`) from the root the encoder was loaded from (`recovered_8b::weight_files`, the resolution `Recovered8bEncoder::load` uses), through `identity::files_head_tail_cached` (read once per process for the same files: re-read when a file's size or modification time, or the set of files, changes) | `h3::text_cache` `the_recovered_identity_follows_each_weight_file`; CPU only: the recovered-8B weights are not on the EU volume |
| 3 | Wan UMT5 GEMM | `Numerics`: `gemm=<host\|f32\|tf32\|bf16>;arch=sm_<major>.<minor>\|none;act=<bf16\|f32>` | `umt5_numerics()` in `text_cache_key` (`wan/pipeline.rs`, now `wan_text_key`): `Numerics::live()` read inside `with_bf16_act(false, ..)`, the scope `Umt5Encoder::forward` runs in, so `act` is UMT5's f32, not the DiT's default | `wan::pipeline` `the_wan_key_is_pinned_and_follows_the_numerics`, `the_umt5_numerics_are_its_f32_activations_whatever_the_default`; CPU only (no Wan tree on the EU volume) |
| 4 | H3 / LTX numerics | `Numerics`, as row 3 | `Numerics::live()` in `encode_prompt_with_at`, `encode_prompt_recovered` and `TextEncoder::key` | `numerics` `every_setting_moves_the_numerics_field`, the families' key tests; GPU `b4c2-h3-numerics` |
| 5 | LTX resident vs streamed (and H3's, the same unproven equality) | `TextMode`: `mode=resident` / `mode=streamed` | LTX: `TextEncoder::planned_mode` (the pure `plan_mode`), the decision `ensure_backend` made, now taken before the lookup (no load) and passed to `compute`, which runs exactly it. H3: `EncoderRun::mode` (`h3/text.rs`): resident when a resident encoder ran, streamed otherwise (the `auto` encoder writes resident FP8, and a process that released it, or another process, streams FP8: keyed apart); recovered 8B is always resident | `ltx2::text_cache` and `h3::text_cache` key tests, `ltx2::pipeline` `the_planned_mode_is_the_backend_decision`, `h3::text` `every_encoder_run_is_keyed_as_what_it_read_and_where_it_ran`; GPU `b4c2-ltx25-mode` (also measures whether the LTX modes are bit-equal) |
| 6 | FP8 tree vs quantise-at-load | `WeightsRead` and the tree identity, as row 1 (H3 too) | H3: `EncoderRun::weights`, a pure function of what ran: a resident decoder records it at load (`ResidentDecoder::prequantized`: FP8 rows and the map holds `weight_scale_rows`, `llm::prequant::holds_fp8_rows`); the streamed FP8 encoder read the tree iff `open_tree` returned one; another resident encoder with FP8 numbers is `fp8-rows/unknown`. The tree identity through `files_head_tail_cached` (row 2). LTX: row 1 | `llm` `a_prequantized_fp8_tree_is_the_load_time_quantization` (the flag), `h3::text` `every_encoder_run_is_keyed_as_what_it_read_and_where_it_ran` (the whole table) and `the_encoder_run_is_the_encoder_that_was_chosen`, the key tests; GPU `b4c2-h3-tree` (resident: `auto` on the RTX PRO 6000; also measures tree vs at-load on the device). The streamed-FP8 tree decision has no GPU cell: no flag makes H3 stream FP8; `auto` does only when its memory plan rejects the resident encoder for the planned request, which on a 96 GB card needs a larger request (well over the ~10 GPU-minute budget for five runs) at an unmeasured threshold |

Where the numerics are resolved: the device context resolves the GEMM math
once, from `FASTVIDEO_BF16`, `FASTVIDEO_TF32` and whether the card has
tensor cores (`rt/device.rs`, `GemmMath`), and keeps the compute
capability; `rt::tensor::bf16_activations` resolves the activations
(`FASTVIDEO_BF16_ACT`, else the H3 / LTX pipeline default on a live device,
else f32; a thread's `with_bf16_act` override included). `Numerics::live`
reads those resolved values — `global_device()`'s `gemm_math`, `sm_major`,
`sm_minor` when ops are expected on the device (`stats::device_expected`),
else `host` — and never an environment variable. It is called in the same
call, on the same thread, right before the lookup whose miss encodes, so no
signature had to change to thread the device through: the context it reads
is the one the encode runs on. The arch is the full compute capability:
coarser classes (tensor-core or not) would be enough for the GEMM math, but
kernels and attention paths are chosen per arch too, and a cache directory
shared across cards is exactly where a hit would replay another card's
numbers.

Domains (new; the old ones are never computed by the pipelines):

| recipe | before B4c2 (`*_V1`) | since B4c2 | identity fields since B4c2 |
|---|---|---|---|
| H3 | `H3TEXT01` | `fastvideo h3 text conditioning v2` | encoder identity (recovered 8B: row 2), weights read, tree identity, mode, numerics |
| Wan | `WANTEXT1` | `WANTEXT2` | `files_head(text_encoder/)`, numerics |
| LTX-2 | `fastvideo ltx2 text conditioning v1` | `fastvideo ltx2 text conditioning v2` | kind, Gemma, connectors, weights read, tree identity, mode, numerics |

The H3 domain is no longer the magic: the magic versions the layout,
which did not change (`H3_MAGIC` keeps `H3TEXT01`).

Pinned keys (recomputed independently with Python `hashlib`; numerics
`gemm=bf16;arch=sm_9.0;act=bf16` unless said): H3 `"a dog <d>hi</d>"`,
tokenizer `[1; 32]`, tap 50, identity `[2; 32]`, checkpoint resident
`29d517fd…`, FP8 tree `[5; 32]` resident `9556f05a…`, streamed
`59ddcb14…`; recovered identity of base / LoRA / adapter `[6]`, `[7]`,
`[8]` `21417c18…`, its key at tap 24 (checkpoint, resident) `1af19d5f…`;
LTX `"a red fox"`, kind `gemma3-12b`, Gemma `[1; 32]`, connectors
`[2; 32]`, bf16 streamed `f961d9c7…`, FP8 tree `[5; 32]` resident
`d3ec427e…`; Wan `"a cat"`, no tokenizer, 512, identity `[3; 32]`, numerics
`act=f32` (UMT5's) `1f09144a…`. The pre-B4c2 pins
(`86ae3a4c…`, `b944238d…`, `032905ee…`) stay, on the `*_V1` recipes.

One-time consequence: every existing H3, Wan and LTX cache entry misses
once after B4c2 (its name is a v1 name, never looked up again) and is
re-encoded and stored under the new name; old entries are not deleted (no
eviction, as before) and may be removed by hand at any time. Keys now also
differ per card and per numerics setting, so a cache directory shared
across card types holds one entry per (prompt, card, settings). If an old
name did match a new one (a SHA-256 collision; the domain makes the two
derivations disjoint otherwise), the entry would be read: the magic is
unchanged, so it is not what separates them. That is the price of not
bumping the magic, which the layout does not need.

Behaviour changes besides keys (LTX): the resident / streamed decision
for `auto` is made at each encode before the lookup (from the free memory
then, as `ensure_backend` made it at the miss), so a hit now also queries
free memory, and an invalid `FASTVIDEO_LTX2_TEXT` fails a hit as it failed
a miss. In `auto` the key therefore depends on the free memory at the
lookup: with less free memory than a resident Gemma needs, a prompt
cached by a resident encode is looked up under the streamed key (and the
reverse). That gives spurious misses only, never a wrong hit: the key
always names the mode the miss then runs. The FP8 tree decision
(`FASTVIDEO_LTX2_TEXT_FP8`) is made once per `TextEncoder` instead of at
every open of Gemma, so its log line prints once.

LTX FP8 with an explicit streamed residency (`--text streamed` or
`FASTVIDEO_LTX2_TEXT=streamed` with `FASTVIDEO_LTX2_TEXT_FP8=1`), behaviour
unchanged here and out of scope: FP8 only makes `ensure_backend` choose
resident, and an explicit streamed residency returns before that, so the
streamed path runs (`HiddenStack::encode`, `llm::hidden_states`: Native
precision over whatever map it is handed). Without a tree that map is
`text_encoder/`, so the encode is bf16 over the checkpoint, and the key
says so (`gemma_weights_read`: `weights=checkpoint;mode=streamed`, not
FP8). With a valid `text_encoder_fp8/` tree, `open_gemma` hands the
streamed path the *tree's* map; Native loading reads its `weight` tensors
as floats, and the tree stores them as `F8_E4M3` codes, which
`LazyStore::to_f32` refuses ("has no F32 conversion"): by reading the code
that run fails before producing numbers, so no entry is ever stored (not
run on a device). The key names it `weights=fp8-rows/tree` with the tree's
identity, the map it is handed. LTX-2.5 on the EU volume has no FP8 tree.

Tests kept meaningful: `equality.rs` compares the *pre-B4c2* derivations
(`H3_KEY_V1`, `WAN_KEY_V1`, `LTX2_KEY_V1`, `cache_key_v1`) with `legacy.rs`,
which pins the names entries on disk have; the format, protocol and
failure tests compare the one implementation with legacy as before (the
format did not change). `b4c2_keys_never_name_a_legacy_entry` shows the
new keys never equal a legacy key over the prompts, weights sources,
numerics and modes, that the domain alone separates them (the v2 recipe
over the v1 fields), and, on disk, that an entry the legacy code wrote is
not read by the new code for the same prompt and weights (a miss that
computes, leaves the old file and stores its own).

Mutations (each applied alone, the tests rerun, all 31 caught): each new
field dropped from the H3, LTX and Wan keys (weights read, tree identity,
mode, numerics); each recovered-8B file dropped from its identity; each
domain reverted to its v1 value; the arch, the activations or the GEMM
math dropped from `Numerics`; `ResidentDecoder::prequantized` never set;
the tree keyed as quantise-at-load; one field for both modes; H3
`EncoderRun` ignoring the streamed tree, keying streamed as resident, or
ignoring whether a resident encoder ran; LTX `plan_mode` with FP8 not
resident, an explicit streamed residency losing to FP8, or no device
counted as room; `gemma_tree` looking for a tree without FP8;
`gemma_weights_read` keying FP8 + streamed as FP8; Wan's numerics read
outside UMT5's f32 scope; the identity memo ignoring a file's size. What
remains device-only — the live values `Numerics::live` reads, and that the
pipelines pass the runs these functions describe — is held by the `b4c2`
GPU cells: each fails when this build's read pass hits.

GPU (`scripts/gpu/runpod-matrix.sh b4c2`, `FV_BASELINE_SHA` = main before
B4c2): switch cells write the cache under one setting and read it under
another, per arm over its own new cache, plus a fresh encode of the read
setting over another new cache; the verdict (`b4-verdict.sh`
`b4c2_switch_verdict`) requires main's read to *hit* (the gap reproduced),
this build's to miss with frames byte-identical to the fresh encode and the
same entry (name and bytes) the fresh encode stored. Cells
`b4c2-ltx25-fp8` (gap 1: FP8 Gemma at load, then bf16, both resident),
`b4c2-ltx25-mode` (gap 5), `b4c2-h3-numerics` (gap 4:
`FASTVIDEO_BF16_ACT=0`, then the default), `b4c2-h3-tree` (gap 6:
`FASTVIDEO_TEXT_FP8_TREE=0`, then the tree). `b4c2-h3-noop` (no change):
default settings, base and PR byte-identical on the miss and on the hit,
the PR's hit equal to its miss, and the two caches holding the same bytes
under different names. The write-against-read compares of the mode and
tree cells record whether those paths are bit-equal on the device.

Found while closing these, not in the table and not closed (each its own
decision, recorded as follow-ups):

| # | gap | where | effect |
|---|---|---|---|
| 7 | `FASTVIDEO_FP8=1` (per-tensor E4M3 `Linear`s, `rt/nn.rs` `fp8_linears`) is consulted when a `Linear` loads; Wan's UMT5 loads `Linear`s (`wan/umt5.rs`), and LTX's connectors (`ltx2/text.rs` `TextConnectors::load`) may; whether those loads take the FP8 path is not verified; it is in no key | `rt/nn.rs:219-222, :339` | if they do: an entry from an FP8-linears run served to a run without it (and the reverse): wrong conditioning, under an opt-in flag |
| 8 | `FASTVIDEO_SDPA` (`rt/sdpa_rule.rs` `backend_setting`) picks the attention backend; whether the text encoders' attention routes through it is not verified | `rt/sdpa_rule.rs:212-219` | if it does: small numeric differences, as row 4 |

## Consequences

- A new family's prompt cache is a `KeyRecipe` (a new domain), a `Format`
  (or an existing one) and a `Cached` value type; the protocol, the file
  handling and the checks are not written again. A new on-disk variant is a
  new named parameter with a fixture, never an edit of an existing format.
- `legacy.rs` is test-only and frozen.
- Follow-ups, each a key or behaviour change and so its own decision: the
  six key gaps in "Follow-ups: key gaps carried over" (closed by B4c2), the
  two found then (rows 7 and 8 of "B4c2"), a size cap, per-thread
  temporary names.
