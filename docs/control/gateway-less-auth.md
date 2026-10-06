# Admin token and API keys without a gateway

**Status (2026-10-02):** implemented on `wip/gwless-auth`, covered by tests
(below). A live run on Runpod is listed at the end.

> **Since 2026-10-06** the gateway is retired
> ([edge-control-plane.md](../serve/edge-control-plane.md) §9, "Stage 4 as
> built"). This mode is now fv-control's `control_plane: "direct"` (the
> spec's `gateway.enabled: false` migrates to it), with the client auth in
> the spec's top-level `auth`; the other mode is `edge`. The gateway-side
> notes below are history.

A cluster spec with `gateway.enabled: false` launches only workers. Clients
call each worker at `https://<podId>-8000.proxy.runpod.net`. Before this
change those workers ran as gateway workers (`FV_SERVE_ROLE=worker`): every
API route wanted the internal token, there was no admin token to show, and
nothing could mint keys.

## What was there

- **fv-serve worker role** (`crates/fastvideo-serve/src/worker.rs`,
  `app.rs`): `server.role = worker` forces `auth.mode = trust-gateway`, puts
  every route except health, metrics and signed files behind
  `x-fv-internal-token`, and makes a random admin token for this process only.
- **Keys on a gateway** (`fastvideo-serve-kit/src/keys.rs`): the gateway is a
  standalone fv-serve. `auth.mode = keys` and `/fv/v1/admin/keys` mint
  (`fv_…`, shown once), list and revoke keys. Only SHA-256 digests are kept,
  in the D1 table `api_keys` (`auth.key_store = auto` resolves to D1 when D1
  is configured). Every process caches the table and reloads it every 30 s.
- **Admin token on a gateway**: fv-serve makes it, publishes it sealed to the
  cluster's X25519 key at `/fv/v1/admin/token/sealed`, and fv-control opens it
  (`adminToken` in `control/src/cluster/ops.ts`).

## Design

The workers authenticate clients themselves, with the same code and the same
key table the gateway uses. fv-control makes the admin token.

1. **fv-serve: `gateway.direct`** (`FV_WORKER_DIRECT=1`, only with
   `server.role = worker`):
   - `auth.mode` is no longer forced to `trust-gateway`. It must be `keys` or
     `none`.
   - The internal token guards only `/fv/v1/internal/*`, which fv-control
     still uses to drain workers and read their status for scale-down and
     roll. Every other route answers as a standalone server would.
   - `FV_ADMIN_TOKEN` is required. Config validation fails without it, so
     every worker of the cluster shares one admin token.
   - The worker runs the MiniMax callback challenge itself.
2. **fv-control** (`isDirect(spec)` = `control_plane: "direct"`):
   - `start` makes `fvadm_<48 hex>` and keeps it in the cluster's sealed
     secrets as `admin_token`. Each launch gets a new one.
   - The worker env adds `FV_WORKER_DIRECT=1`, `FV_AUTH_MODE=<auth>`,
     `FV_KEY_STORE=d1` and `FV_ADMIN_TOKEN`. The admin token is masked in
     every view, and `FV_WORKER_DIRECT` is a reserved key.
   - Restarts (env apply) and scale-ups render the same env, so new pods get
     the same token and read the same keys from D1.
   - `POST …/admin-token` returns the token, `direct: true`, the per-worker
     URLs and the first worker's `/console/admin`.
   - `POST …/mint-key` calls `/fv/v1/admin/keys` on the first worker that
     answers. That worker accepts the key at once; the others accept it after
     their next D1 reload (≤ 30 s, `propagation_s` in the reply).
   - `GET …/keys` lists the keys. It works on edge clusters too (at the edge).
   - `DELETE …/keys/<key_id>` revokes. On a direct cluster it sends the
     DELETE to every worker, so the key is refused everywhere at once. A
     worker that misses the call reads the revocation from D1 within 30 s.
     The reply lists `applied` and `failed` pods, and the action is audited
     as `cluster.revoke-key`.
   - `GET …/front` (alias `…/gateway`) returns `{direct: true, workers:
     [{pod, pool, url, health}]}`.
   - The dashboard card is "Workers (direct)". It lists the worker
     URLs and consoles and has buttons for the workers view, revealing the
     admin token, minting a key, and listing and revoking keys.
   - A gateway-less cluster launched before this change has no token. The
     first `admin-token` call makes one and answers 409 "restart the workers
     (Env: apply)". The env view then shows the workers as needing a restart.

### Why D1, and not keys kept in fv-control and pushed to the workers

- **Persistence:** the table already exists, already holds only digests, and
  already survives restarts and new pods. Every worker image reads it with no
  new code.
- **Restarts:** pushing keys through the env (`FV_API_KEYS`) would restart
  every GPU worker on each mint or revoke, which means minutes of model load.
  A push route would be new code on both sides plus a resync on every boot.
- **Revocation:** fanning the DELETE out from fv-control makes it immediate.
  The 30 s reload covers a worker that misses the call.

## Requirements and limits

- **Image:** the workers need an fv-serve image that reads
  `FV_WORKER_DIRECT` (this branch or later). An older image ignores it and
  keeps answering 401 "this is a gateway worker" without the internal token,
  which is the behaviour before this change. No new keys are written to the
  worker TOML, so older images still start.
- **Key scope:** keys are account-wide, as they already are for gateways.
  `api_keys` has no cluster column, and every cluster (and every gateway)
  that uses the same serve D1 (`FV_D1_DATABASE_ID`) accepts the same minted
  keys. A key minted on a gateway-less cluster also works on a gateway
  cluster, and the reverse. Scoping keys per cluster would need a column in
  `api_keys` and a filter in the key store. That is a separate change.
- **New keys:** after a mint, the other workers accept the key within 30 s
  (their D1 reload). Clients that need it at once can use the worker named in
  `minted_on`.
- **Collector:** the per-minute collector reads jobs only from the edge's
  families view. Direct workers show Runpod's pod data and the ready
  flag from `start`.

## Tests

- `cargo test -p fastvideo-serve --lib direct_worker`: config validation
  (worker role only, no `trust-gateway`, `FV_ADMIN_TOKEN` required).
- `cargo test -p fastvideo-serve --features http-client,minimax --test direct_workers`
  runs two, then four, direct workers on one mock D1 and checks:
  - without a key the APIs answer 401 and `/health` stays open;
  - the internal routes still need the internal token;
  - the admin routes need the admin token;
  - a minted key runs a job on the worker that minted it, and works on
    another worker after its reload and on a worker started later;
  - once revoked on every worker it is refused everywhere, also on a new
    worker;
  - the admin token is not accepted as a client key.
- `control/test/unit/payloads.test.ts`: the direct worker env (client
  auth, the controller's admin token, the D1 key store).
- `control/test/integration/run.mjs`, step "direct cluster" runs
  start → one pod per worker and the env checked → admin-token reveal → env view
  masked → scale to 2 (same token) → workers view → mint → list → revoke on
  both workers → stop.

## Live run

See the results section of the change's report (staging cluster, one GPU
worker, a small model): an unauthenticated request is rejected, a minted key
works, a revoked key fails, and the admin token works.
