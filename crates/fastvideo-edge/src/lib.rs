//! `fv-edge`: a Cloudflare Worker and one `PoolScheduler` Durable Object per
//! GPU pool (docs/serve/gateway-cloudflare.md, phases 0-1).
//!
//! | Route (Worker) | Auth | What |
//! |---|---|---|
//! | `GET /`, `/healthz` | none | `{"object": "fv.edge", "version"}` |
//! | `GET /pools/{pool}/connect` (WebSocket) | internal token, `x-fv-worker-id` | a GPU worker's socket (`fastvideo-dispatch-proto`) |
//! | `POST /pools/{pool}/enqueue` | internal token | `EnqueueReq` → `EnqueueResp` (the fv-serve gateway, `dispatch = "durable-object"`) |
//! | `POST /pools/{pool}/cancel/{job}` | internal token | client cancel |
//! | `GET /pools/{pool}/status` | internal or admin token | `PoolStatus` |
//!
//! Tokens are Worker secrets: `FV_INTERNAL_TOKEN` (the gateway's and the
//! workers' shared secret) and `FV_ADMIN_TOKEN` (the gateway's admin
//! token), sent as `x-fv-internal-token` or `Authorization: Bearer`. A
//! missing secret fails closed (503).
//!
//! The Durable Object keeps its queue and worker registry in its SQLite
//! storage (`jobs`, `envelopes`, `workers`, `meta`), rebuilds the pure
//! scheduler ([`fastvideo_dispatch_proto::sched`]) from it after every
//! eviction (hibernation, deploys), accepts worker sockets with the
//! hibernation API, and writes a record of each job to D1 (`edge_jobs`,
//! binding `DB`) behind the critical path (`waitUntil`). Its alarm runs
//! the scheduler's timers: ack deadlines, lost workers (re-dispatch once,
//! then fail), backoffs and cleanup.
//!
//! On the host this crate is empty: build it with
//! `scripts/serve/cf-edge.sh build` (wasm32-unknown-unknown).

#[cfg(target_arch = "wasm32")]
mod edge;
