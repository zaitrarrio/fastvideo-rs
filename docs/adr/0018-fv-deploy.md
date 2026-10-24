# ADR-0018: `fv-deploy`, the `Provider` trait and the shared money guards

- Status: Accepted for C3a (Runpod pods and serverless queue endpoints, the shared guards, the `Launch` schema); the
  live gate (one `fv-deploy up` on Runpod EU) is the coordinator's; Vast and GCP (C3b), fv-control on the generated
  `Launch` types (C3c) and templates / release / the D1 ledger (C3d) follow
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-fv-deploy
- Code: `crates/fastvideo-deploy/src/{provider,launch.rs,guard.rs,lifecycle.rs}`, `src/bin/fv-deploy.rs` (feature
  `deploy`); schemas `crates/fastvideo-deploy/schemas/{launch,instance,offer,account}.v1.json`; goldens
  `crates/fastvideo-deploy/fixtures/deploy/*.json` (written by `scripts/serve/tests/fv-deploy-goldens.sh`)
- Usage: `docs/ops/fv-deploy.md`
- Plan: C3 (`docs/plan/2026-10-23-execution-plan.md`); review `docs/gaps/2026-10-22-architecture-review.md` §7.4
  (C3, the trait sketch and "Collapses"), §4.5 ("`fastvideo-autoscale` is orphaned")

## Context

Every provider launch is written several times: `scripts/serve/runpod-pod.sh`, `runpod-endpoint.sh`,
`scripts/gpu/runpod-http.sh` (bash + jq + curl), fv-control's `runpod.ts`, `buildpods.ts`, `providers.ts`,
`cluster/payloads.ts` (TypeScript), `fastvideo-autoscale`'s `provider/runpod.rs` (Rust, orphaned: no crate depends on
it). The money guards are repeated in each: the price check before a create (`runpod-price.sh`, sourced by the
scripts), the balance floor (`check_balance` twice in bash, `myself { clientBalance }` in TS and Rust), the wall-clock
backstop (`nohup bash -c 'sleep $CAP; curl -X DELETE …' &` three times, the in-pod `WATCHDOG`, fv-control's cron), the
readiness wait (`FV_BOOT_WAIT_S` 1200 s, one cycle in `runpod-pod.sh`, three in `runpod-http.sh`), the EU-only volume
rule (`volumes.sh fv_check_volume`, `deploy.rs check_volumes`, `regions.ts`). The review's C3 asks for one Rust CLI
with one `Provider` trait, promoted from the autoscaler's, and the guards written once.

C3 as a whole is weeks of work. This ADR decides the shared shape and lands **C3a**: Runpod only (pods and
serverless queue endpoints), everything shared, and the binary. Later slices add a provider at a time against the same
trait and schema.

## Decision

### 1. Scope of C3a

`fv-deploy plan | up | status | logs | smoke | down` for `runpod-pod` and `runpod-serverless` (queue) launches of a
`deploy/deploy.toml` preset on the EU volume; `fv-deploy schema` and the internal `fv-deploy backstop`. Not in C3a:
Vast and GCP (C3b), fv-control on the generated types (C3c), `templates sync`, `release promote|rollback`,
`reconcile` and the D1 `deployments` ledger (C3d), weights sources other than `volume` (the region guard refuses
them), Runpod load-balancer endpoints (an open owner decision, plan §1).

### 2. Where the trait lives, and why blocking

**The trait lives in `fastvideo-deploy` (`provider` module); `fastvideo-autoscale` depends on `fastvideo-deploy`**,
not the reverse:

- The instance lifecycle is the lower layer. Its consumers are the `fv-deploy` CLI, `fv-lab pod` (plan D1, "on the
  `Provider` trait of C3") and fv-control's schema; the autoscaler is one more consumer (a controller that steers
  pools). Putting the trait in `fastvideo-autoscale` would make `fv-deploy` and `fv-lab` depend on the scaling policy,
  the gateway interface, the D1 lease and the traffic simulator.
- `fastvideo-deploy` already owns everything provider-facing that runs on a host or in a container (fv-boot, the
  Runpod queue worker, Vast's env and onstart, the `RUNPOD_*` / `VAST_*` discovery), and its seams (`boot/sys.rs`:
  `Http`, `Clock`, `Spawner`, `LogSink`) make a provider testable without a network. `fastvideo-autoscale` had no
  dependents; it now has a dependency that gives it the shared Runpod constants and helpers (`REST`, `QUEUE`,
  `GRAPHQL`, `NO_CAPACITY`, `parse_runpod_time`, re-exported where its API named them).
- The autoscaler's own `Provider` (`observe` / `apply` a `PoolDecision`) stays: it is the pool-steering interface, not
  an instance lifecycle. Its `RunpodPods` / `RunpodServerless` move onto the new trait in a later slice (they create
  from templates and patch endpoint scaling, which C3a's trait does not need yet).

**The trait is blocking.** Every caller is a sequential command-line flow (create, wait, smoke, delete), as fv-boot
is; the `Http` seam it runs on is blocking (`boot::http::ReqwestHttp` blocks on a current-thread runtime), so the whole
lifecycle runs in tests on a fake HTTP server and a fake clock in milliseconds (a 3 × 600 s replace cycle included).
An async caller (the autoscaler's controller) calls it through `spawn_blocking`.

```rust
pub trait Provider {
    fn kind(&self) -> ProviderKind;
    fn offers(&self, q: &OfferQuery) -> Result<Vec<Offer>, ProviderError>;   // quotes, for the price cap
    fn payload(&self, launch: &Launch) -> Result<Vec<ApiCall>, ProviderError>; // what create sends (plan)
    fn create(&self, launch: &Launch) -> Result<Instance, ProviderError>;
    fn status(&self, id: &InstanceId) -> Result<InstanceState, ProviderError>;
    fn logs(&self, id: &InstanceId, since: Option<&Cursor>) -> Result<LogPage, ProviderError>;
    fn destroy(&self, id: &InstanceId) -> Result<(), ProviderError>;          // gone is not an error
    fn account(&self) -> Result<Account, ProviderError>;                       // balance, for the floor
    fn reach<'a>(&'a self, i: &Instance, opts: &ReachOptions) -> Result<Box<dyn Reach + 'a>, ProviderError>;
    fn sweep(&self, name: &str, keep: Option<&InstanceId>) -> Result<Vec<InstanceId>, ProviderError>; // lost answers, every exit
    fn bills_at_create(&self) -> bool;                                          // pods: the billed price is checked
}
pub trait Reach {
    fn ping(&mut self) -> Probe;
    fn call(&mut self, m: Method, path: &str, body: Option<&Value>, stop: &dyn Fn() -> bool) -> Result<(u16, Value), ProviderError>;
}
```

`ProviderError::is_definite_refusal` tells a create the provider certainly refused (4xx, no capacity) from one whose
answer was lost (transport, 5xx, 2xx without an id: §5.6). Additions to the review's sketch: `sweep`, `bills_at_create`,
`payload` (so `plan` prints the provider's exact requests from the provider's
own builder, with no network) and `reach` (how the readiness poll and the smoke reach the instance: a pod through
Runpod's HTTPS proxy, a serverless endpoint through its queue, one native job per call, so the smoke is the same
sequence on both).

### 3. The types and the schema

`provider::types`: `Launch` (schema 1), `Instance`, `InstanceId`, `InstanceState` (`pending`, `running`, `exited`,
`terminated`, `unknown`: the provider's view; readiness is the instance's own `/ping`), `Offer`, `OfferQuery`,
`Account`, `LogPage`, `Cursor`, `ApiCall`; serde + schemars, `deny_unknown_fields`. Their JSON Schemas are committed
under `crates/fastvideo-deploy/schemas/` and `tests/schema.rs` fails on drift (as `family.v1.json`); `fv-deploy schema
<name>` prints one. fv-control's TypeScript types are generated from them in C3c (control is untouched here).

A `Launch` never carries a secret value: `env` holds plain settings and Runpod secret references
(`{{ RUNPOD_SECRET_… }}`); `FV_API_KEYS` holds the SHA-256 of the run's key. The provider key lives only in the client
(an `Authorization` header; `Debug` redacts it; error bodies are scrubbed of it) and the env guard refuses a launch
that contains it.

#### 3.1 Errors

`thiserror` enums per layer: `ProviderError` (`Api`, `NoCapacity`, `Transport`, `Decode`, `Unsupported`,
`MissingKey`), `LaunchError` (`Io`, `Deploy`, `Preset`, `Region`, `Image`), `GuardError` (`UsGone`, `Region`,
`BalanceBelowFloor`, `NoQuote`, `OverCap`, `BilledOverCap`, `SecretInLaunch`, `Invalid`, `Backstop`), `DeployError`
(the lifecycle: wraps the others plus `NotReady`, `NoCapacity`, `Interrupted`, `Smoke`). No `unwrap` / `expect` /
`panic` outside tests; the binary maps errors to a `FAILED:` line and exit 1.

#### 3.2 How a `Launch` is built

`launch::from_preset(repo, request, stamp)`: the **preset** (`deploy/deploy.toml` `[[presets]]` or
`[[family_presets]]`, read with `fastvideo-manifest` and checked against `families/*.toml`) gives the image variant,
the worker config, the container disk and the variants (hence the family and the weight trees); the **provider
section** (the region: `eu` → `EU-RO-1`, volume `bz0ofkfx45`, GPU types `FV_EU_GPUS`; the kind; the channel; the
guards; optional GPU types and image) gives the rest:

| Field | From |
|---|---|
| `image` | `ghcr.io/zaitrarrio/fastvideo-rs-serve:<variant>-<channel>`, pinned to its digest by `up` (the scripts' `resolve_digest`: the digest is the SHA-256 of the manifest bytes); `--image` replaces it |
| `entrypoint` | `/opt/fastvideo-rs/bin/fv-boot` (ADR-0014: one boot; it links the serverless volume itself) |
| `args` / config | the image carries the config → `--config /etc/fv/<config>`; else the committed `configs/serve/<config>` inline as `FV_WORKER_TOML_B64` (ADR-0014 §3.2) |
| `env` | the Runpod secret references, `FV_IMAGE_REF` / `FV_IMAGE_DIGEST` / `FV_RELEASE_CHANNEL`, `FV_WEIGHTS_SOURCE=volume`, and per kind what `runpod-pod.sh` / `runpod-endpoint.sh` send (`FV_SERVE_MODE`, `FV_STATE_DIR`, `FV_WEIGHTS`, `FV_API_KEYS`, `FV_SERVE_FORWARD`, `FV_AUTH_MODE`, `FV_CACHE_DIR`, `RUST_LOG`) |
| `placement` | the region's data centre and volume (at `/workspace` on a pod), Secure cloud |
| `disk` | the preset's `container_disk_gb`, else 40 GB for a pod (fv-control's default with the volume), 20 GB for a template (the script's) |
| `gpu` | `--gpu` types or the region's, one GPU; no `allowedCudaVersions` (REST cannot name 13.1 / 13.2, `runpod-cuda.sh`); serverless `minCudaVersion` 13.0 |
| `serverless` | the script's scaling: workers 0..1, idle 5 s, FlashBoot off, 30 min execution timeout, `QUEUE_DELAY` 1 |

The preset's trees must all be on the EU volume (`scripts/gpu/volumes.sh` `FV_EU_VOLUME_TREES`), else `plan` / `up`
refuse it (r2 / hub weights are not in C3a): H3 and LTX presets build, Wan presets are refused.

### 4. Runpod payloads equal the scripts'

`runpod::pod_create_body`, `template_body` and `endpoint_body` build field for field what the scripts send today.
Evidence: `scripts/serve/tests/fv-deploy-goldens.sh` runs `runpod-pod.sh plan` and `runpod-endpoint.sh plan`
themselves (a fake `curl` on `PATH` answers the one volume lookup, nothing else) and writes
`fixtures/deploy/{runpod-pod,runpod-template,runpod-endpoint}.json`; `tests/payload_goldens.rs` asserts the Rust
bodies for the same inputs equal them (pod with the network volume, `EU-RO-1`, ports, env, `dockerEntrypoint` /
`dockerStartCmd`; template; queue endpoint), and that a preset launch differs from the script pod **only** in
`name`, `dockerEntrypoint` (fv-boot), `containerDiskInGb` (40) and two env keys (`FV_WEIGHTS_SOURCE`,
`FV_RELEASE_CHANNEL`). CI (`control.yml`) runs the script with `--check`, so a script change that is not mirrored in
Rust fails.

The price query is the script's text (`price_query`), the quote the script's rule (the cloud's price and the data
centre's lowest uninterruptible price, the larger; no positive number = no quote). Readiness: `/ping` 502 / 404 while
the container starts, 204 loading, 200 ready (`runpod-pod.sh`); a queue endpoint is ready when a cold `info` job
completes (`runpod-endpoint.sh`'s first job).

### 5. The shared guards (`guard.rs`, `lifecycle.rs`)

Written once, tested with a provider-neutral fake (`lifecycle/tests.rs`) and through the Runpod code on a fake Runpod
(`provider/tests.rs`). Amended after the PR #169 review (leak and bypass paths closed, each with a test).

1. **Region guard** (no network): a Runpod GPU instance mounts the EU weights volume `bz0ofkfx45` in `EU-RO-1` and
   nothing else; no volume, another volume, another data centre or a weights source other than `volume` is refused;
   the deleted US volume (`s2k01690bi`, by id or by name) and region `us` are refused with `volumes.sh`'s `FV_US_GONE`
   text (CLAUDE.md; plan §1, owner instruction 2026-10-23).
2. **Balance floor, over the whole cap**: `account()` before every create (a replacement included); the balance must be
   at least `min_balance_usd` (default and minimum $8, CLAUDE.md) **and** stay there over the cap:
   `balance − (quote + currentSpendPerHr) × cap_s / 3600 ≥ floor` (CLAUDE.md: stop before the balance *would* drop
   below $8). A GPU type whose projection fails is skipped; when none passes, `up` refuses before any create.
3. **Price cap**: quotes before any create; no quote or a quote over `max_usd_per_hr` (default $2.50 for one GPU,
   `--max-usd-per-hr`) skips the GPU type, never creating at an unknown price (`runpod-price.sh`). A pod's billed
   `costPerHr` (a number or a numeric string: Runpod's REST docs show `'0.74'`) over the cap, **or missing /
   unreadable**, fails the check and the pod is deleted. A pod tries the GPU types in order; a serverless endpoint gets
   every type whose quote passes.
   **Known gap (serverless):** Runpod reports no price for a serverless endpoint at create, and fv-deploy reads no
   serverless quote: the cap and the projection use the GPU's *pod* quote for one worker, and a serverless worker's
   per-second price is not checked against it. The guard therefore refuses a serverless launch with `workers_max > 1`
   (the projection multiplies the quote by `workers_max` regardless). A serverless quote is a C3b/C3d follow-up.
4. **Detached wall-clock backstop, armed before anything else**: right after **every** successful create (before the
   billed-price check), `up` starts this binary again as `fv-deploy backstop --kind K --id ID --deadline T
   --armed-file F` (`T` = creation + `cap_s`, default 1800 s) in a **process group of its own** (`process_group(0)`: a
   Ctrl-C to the terminal's foreground group does not reach it), stdin closed, output appended to
   `artifacts/fv-deploy/backstop-<stamp>.log`, ignoring SIGHUP and SIGINT (as `nohup`); it inherits the environment,
   so `RUNPOD_API_KEY` is never on a command line. **Handshake:** the child writes `armed` to `F` once its API client
   exists, the key is there and its signal handlers are installed; `up` waits up to 15 s for it, and a child that
   exits or stays silent (it is then killed) means **no backstop**. It sleeps to the deadline and deletes the instance
   (3 attempts, 5 s apart; a serverless endpoint as `runpod-endpoint.sh down`: workers to 0, the endpoint, then its
   template). It stands down early only after **two "gone" readings at least 60 s apart** (only a 404 is "gone": a 400
   or an answer without an id is not). It is the scripts' `nohup bash -c 'sleep $CAP; curl -X DELETE …' &` plus the
   handshake and the early exit. **It dies with its host**: run `up` from a machine that outlives `cap_s` (not a
   laptop about to sleep, not a container about to be recycled). Not used: fv-boot's in-pod watchdog (it needs the
   account key in the pod's env, `FV_BACKSTOP_API_KEY`, which Runpod's `GET /pods` returns in clear; open decision
   below) and fv-control's cron (fv-deploy runs without fv-control).
5. **No backstop, no instance**: when arming fails, the instance is deleted at once (3 attempts, 5 s apart) and `up`
   stops with `NoBackstop { id, deleted }`; when the delete failed too, the error says to delete that id by hand.
6. **A create whose answer was lost**: any create error other than a 4xx answer or "no capacity" (a transport error
   after the request may have gone out, a 5xx, a 2xx without an id) stops the GPU-type fallback and `up`
   (`CreateUncertain`): the provider's `sweep` lists pods (or endpoints and templates) and deletes those named
   `launch.name` (unique per `up`: `fv-deploy-<preset>-<MMDDHHMMSS>`). **Every exit of `up`** sweeps the name the
   same way (all but an instance kept with `--keep`).
7. **Readiness and replace** (the owner's rule, plan §1): `/ping` every 5 s; not 200 within `ready_timeout_s` (600)
   of the creation, or the provider reports the instance exited or gone → delete it (3 attempts) and create a new
   one, balance floor first; **if that delete fails, `up` stops** (`DeleteFailed`) instead of creating another. At
   most `ready_cycles` (3) instances per `up`: **every create counts**, a GPU-type fallback after a billed-over-cap
   delete included. Then `up` stops with `NotReady` and the last reason.
8. **Upper bounds**: `cap_s ≤ 4 h`, `ready_cycles ≤ 3`, `ready_timeout_s ≤ 600` unless `--force-guards`; the $8 floor is
   never lifted.
9. **Env guard**: a launch whose env, command, image or name contains the provider key is refused before any request.
10. **Delete on exit, and interrupts**: any failure after a create, a failed smoke, or SIGINT / SIGTERM deletes the
    instance (with retries; the ledger says "deleted" only when it was, "delete-failed" otherwise); `--keep` leaves it
    to the backstop (an interrupt deletes it even then). The interrupt is polled in the readiness wait, in the
    smoke's waits and inside a serverless queue call (whose execution timeout is the smoke's `job_wait_s`, not a fixed
    30 min); a **second** SIGINT / SIGTERM ends the process at once (exit 130; the backstops still delete at their
    caps). A local ledger (`artifacts/fv-deploy/ledger.tsv`, the scripts' format) records every create, ready, delete
    and failed delete.
11. **Serverless template leftovers**: `destroy` deletes the endpoint, then its template; a failed template delete
    leaves a template behind (no cost; the exit sweep retries it by name).

### 6. The smoke and the commands

The smoke is the C2 entry smoke (`scripts/gpu/fv-boot-smoke-pod.sh`), run from outside: `/ping` 200, `/healthz`
(`state`, `loaded`, `loading`), the first model of `/fv/v1/capabilities` (or `--model`), one job `{"prompt": "a red
fox trotting through fresh snow", "seed": 1, "short_edge": 480, "aspect_ratio": "16:9"}` (+ `--params`) polled to
`succeeded` / `failed` / `cancelled` within 600 s. A pod is called with the run's key (its SHA-256 is the launch's
`FV_API_KEYS`, as `runpod-pod.sh`); a queue endpoint's calls are native `http` jobs.

`up` = guards → create → backstop → ready (replace rule) → smoke → delete (unless `--keep`); `up` without `--yes` is
`plan`.

What each subcommand replaces, and when the script goes (**nothing is deleted in C3a**; each script is deleted after
the coordinator's live gate shows the subcommand equal on Runpod):

| fv-deploy | Replaces | Deleted when |
|---|---|---|
| `plan --kind pod` | `runpod-pod.sh plan` | after the pod live gate |
| `up --kind pod` (with the smoke) | `runpod-pod.sh smoke` / `up`; the create / wait-up / cap half of `runpod-http.sh` | `runpod-pod.sh` after the pod live gate; `runpod-http.sh` with D1 (`fv-lab pod` / `run` on this trait) |
| `plan` / `up --kind serverless` | `runpod-endpoint.sh plan` / `smoke` / `up` (queue) | after a serverless live gate; `smoke-lb` / `up-lb` stay until the owner's Runpod LB decision |
| `down` | `runpod-pod.sh down`, `runpod-endpoint.sh down`, `runpod-http.sh down` | with their scripts |
| `status`, `logs` | `runpod-http.sh status`; fv-control's log tail for a script pod | with `runpod-http.sh` |
| `smoke <id>` | the smoke half of `runpod-pod.sh smoke` | with `runpod-pod.sh` |
| price cap | `runpod-price.sh` (`fv_runpod_price_ok`) | when its last caller (`runpod-http.sh`, `runpod-matrix.sh`, `family-pool-pod.sh`, …) has moved (D1) |
| backstop | the `nohup … curl -X DELETE` lines of the three scripts | with them |
| region guard | `volumes.sh fv_check_volume` for these paths | kept: sourced by every GPU script |

Kept: `runpod-cuda.sh` (constants), `runpod-templates.sh` and `release.sh` (C3d), `volumes.sh`.

## Open decisions

- **The default price cap** ($2.50/hr for one GPU): a guard choice, not a price fact (the RTX PRO 6000 Secure quote
  is UNVERIFIED here: no live call was made); the quote is checked live before every create, so a cap below the quote
  refuses rather than overspends (`--max-usd-per-hr` raises it).
- **Serverless price**: no serverless quote is read (§5.3); `workers_max > 1` is refused until one is.
- **In-pod backstop**: fv-boot's watchdog could repeat the deadline inside the pod (as fv-control's standalone pods
  do) if fv-deploy sent `FV_CLUSTER_DEADLINE` and the account key as `FV_BACKSTOP_API_KEY`; that puts the account key
  in the pod's env, readable through `GET /pods`. C3a does not; the owner decides.
- **Which images carry fv-boot**: the preset launch's entrypoint is fv-boot. The live gate must use an image built
  after #159 (`--channel latest`, or `--image <digest>`), until `stable` is promoted (ADR-0014 §6 step 4).
- The autoscaler's pod and endpoint providers onto this trait, Vast and GCP (C3b), fv-control on the generated types
  (C3c), templates / release / the D1 ledger (C3d).

## Consequences

- One Rust path for a Runpod launch, with the guards in one place and testable without a network; the scripts stay
  until the live gate and are then deleted one by one.
- `fastvideo-autoscale` is no longer orphaned: it depends on the provider layer it will run on.
- Four more schemas under drift check; fv-control can generate its types from them (C3c).
- Review fixes (PR #169) are pinned by: `a_lost_create_answer_stops_and_sweeps_by_name`,
  `a_lost_endpoint_answer_sweeps_the_endpoint_and_its_template`, `the_exit_sweep_removes_a_stray_instance_of_the_run`,
  `billed_price_as_a_string_passes_and_unreadable_fails`, `over_cap_pod_whose_delete_fails_stops_with_its_backstop_armed`,
  `not_ready_pod_whose_delete_fails_is_not_replaced`, `no_backstop_deletes_with_retries_and_says_so`,
  `the_floor_holds_over_the_whole_cap`, `serverless_with_more_than_one_worker_is_refused`,
  `prices_parse_from_numbers_and_strings`, `create_refusals_are_told_from_lost_answers` (`provider/tests.rs`);
  `guard_upper_bounds_unless_forced`, `every_create_counts_against_the_cycles`, `unknown_billed_price_fails_the_cap`,
  `the_backstop_does_not_stand_down_on_a_flapping_gone`, `an_interrupt_during_the_smoke_deletes_the_instance`
  (`lifecycle/tests.rs`); `detached_backstop_deletes_the_pod_on_its_own` (the armed handshake, and a child without a
  key fails `arm`), `down_needs_a_kind_for_a_bare_id_and_infers_serverless_from_a_slash` (`tests/fv_deploy_bin.rs`).
- Enforced by: `provider/tests.rs` (Runpod over a fake Runpod: the create body is what goes out, no stock, the key
  never in a URL / body / error / `Debug`, serverless create and delete order, `up` end to end on both kinds, 3
  replace cycles then give up, the floor before any create, over-cap and unquoted types never created, billed over
  the cap deleted, GPU-type fallback, `--keep` then the backstop deletes, smoke failure still deletes, interrupt
  deletes, the key in the env refused), `lifecycle/tests.rs` (the same rules on a provider-neutral fake, the region
  guard table, the backstop command's arguments), `tests/payload_goldens.rs`, `tests/schema.rs`,
  `tests/fv_deploy_bin.rs` (feature `deploy`: the real detached backstop process deletes a pod on a local fake Runpod
  by itself; `up --yes` below the floor makes no create; `plan` and a dry `up` make no request and never print the
  key; the US volume is refused), `scripts/serve/tests/fv-deploy-goldens.sh --check` in `control.yml`.

## Validation

CPU, on the build pod: `cargo test --locked -p fastvideo-deploy --all-features`, `cargo test --locked -p
fastvideo-autoscale` (and `--all-features`), `cargo clippy --locked --workspace --all-targets -- -D warnings`, `cargo
fmt --all --check`, `bash scripts/gpu/lint.sh`, `bash scripts/serve/tests/fv-deploy-goldens.sh --check`.

GPU (the coordinator's gate; never run by the implementing agent): build `fv-deploy` on the build pod
(`cargo build --release --locked -p fastvideo-deploy --features deploy --bin fv-deploy`, fetch
`release/fv-deploy`), then from the repository root

```sh
RUNPOD_API_KEY=… ./fv-deploy up --family h3 --preset h3-turbo --region eu --channel latest --yes --cap-s 1800
```

Expected: balance ≥ $8 checked, the RTX PRO 6000 quote ≤ $2.50/hr, one pod on `bz0ofkfx45` in `EU-RO-1` with fv-boot,
the backstop armed (pid in the log), ready within 600 s (else replaced, at most 3), the smoke's job `succeeded`, the
pod deleted, `artifacts/fv-deploy/{ledger.tsv,up-<stamp>.json}` written. Worst case at the cap: three pods × 600 s
of readiness, then on the third a smoke of at most `PING_WAIT` (120 s) plus the job wait (600 s; on a serverless
endpoint the job's POST runs as one queue job bounded by the same 600 s), about 42 GPU-minutes, about $1.75 at
$2.50/hr; the projection guard needs `balance − (quote + current spend) × 0.5 h ≥ $8` first. Run it from a host that
outlives `--cap-s` (the backstop runs there). Recommended: `--cap-s 1800`, the default.
