//! `D1JobStore` against the SQLite mock of the D1 HTTP API.

use std::sync::Arc;
use std::time::Duration;

use fastvideo_protocol::{
    ErrorKind, JobMetrics, JobStatus, JobStore, KeyId, ListQuery, ProtocolId, SortOrder, StoreError,
    Task,
};
use serde_json::json;
use time::OffsetDateTime;

use super::mock::MockD1;
use super::*;
use crate::store::tests::job;
use crate::store::MemJobStore;

fn t0() -> OffsetDateTime {
    OffsetDateTime::now_utc()
}

fn client(m: &MockD1) -> D1Client {
    D1Client::new(Arc::new(m.clone())).with_retry(RetryPolicy::immediate(4))
}

fn opts(worker: &str) -> D1Options {
    let mut o = D1Options::new(worker);
    o.flush_tick = Duration::from_millis(50);
    o
}

async fn open(m: &MockD1, worker: &str) -> Arc<D1JobStore> {
    D1JobStore::new(client(m), opts(worker)).open(t0()).await.unwrap()
}

fn row(m: &MockD1, id: fastvideo_protocol::JobId) -> serde_json::Map<String, serde_json::Value> {
    m.sql("SELECT * FROM jobs WHERE id = ?", &[json!(id.to_string())]).unwrap().remove(0)
}

fn upserts_for(m: &MockD1) -> usize {
    m.statements().iter().filter(|s| s.contains("ON CONFLICT(id)")).count()
}

#[tokio::test]
async fn schema_migrates_idempotently() {
    let m = MockD1::new();
    let db = client(&m);
    assert_eq!(schema::migrate(&db).await.unwrap(), schema::latest());
    assert_eq!(schema::migrate(&db).await.unwrap(), schema::latest());
    let idx: Vec<String> = m
        .sql("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'jobs' AND name LIKE 'jobs_%' ORDER BY name", &[])
        .unwrap()
        .into_iter()
        .map(|r| r["name"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        idx,
        ["jobs_expires", "jobs_owner_created", "jobs_protocol_created", "jobs_status_created", "jobs_worker_status"]
    );
    let v = m.sql("SELECT COUNT(*) AS n FROM schema_migrations", &[]).unwrap();
    assert_eq!(v[0]["n"], schema::latest());
    assert_eq!(m.sql("SELECT COUNT(*) AS n FROM api_keys", &[]).unwrap()[0]["n"], 0);
}

#[tokio::test]
async fn insert_is_durable_and_shared_between_workers() {
    let m = MockD1::new();
    let a = open(&m, "w1").await;
    let b = open(&m, "w2").await;
    let mut j = job(ProtocolId::MiniMaxV2, "123456789012345678", t0());
    j.owner = Some(KeyId("key_abc".into()));
    a.insert(j.clone()).await.unwrap();
    let r = row(&m, j.id);
    assert_eq!(r["status"], "queued");
    assert_eq!(r["protocol"], "minimax_v2");
    assert_eq!(r["owner"], "key_abc");
    assert_eq!(r["worker"], "w1");
    assert_eq!(r["task"], "t2v");
    // Another worker reads it from D1.
    assert_eq!(b.get(j.id).await.unwrap(), j);
    assert_eq!(b.by_external(ProtocolId::MiniMaxV2, "123456789012345678").await.unwrap().id, j.id);
    assert!(b.by_external(ProtocolId::Fal, "123456789012345678").await.is_none());
    assert!(b.watch(j.id).is_none(), "only the owning worker can watch");
    assert!(a.watch(j.id).is_some());
    // Duplicate wire ids are refused across workers.
    let dup = job(ProtocolId::MiniMaxV2, "123456789012345678", t0());
    assert!(matches!(b.insert(dup).await, Err(StoreError::DuplicateExternal(..))));
    assert_eq!(a.insert(j.clone()).await, Err(StoreError::AlreadyExists(j.id)));
}

#[tokio::test]
async fn progress_writes_are_throttled_and_state_changes_are_immediate() {
    let m = MockD1::new();
    let s = open(&m, "w1").await;
    let j = job(ProtocolId::Fal, "p", t0());
    let id = j.id;
    s.insert(j).await.unwrap();
    s.update(id, Box::new(|j| j.mark_running(OffsetDateTime::now_utc()).unwrap())).await.unwrap();
    assert_eq!(row(&m, id)["status"], "running", "state change written at once");
    let base = upserts_for(&m);
    for step in 1..=40 {
        s.update(id, Box::new(move |j| j.set_step(step, 50))).await.unwrap();
    }
    assert_eq!(upserts_for(&m), base, "progress is not written inline");
    tokio::time::sleep(Duration::from_millis(1400)).await;
    let n = upserts_for(&m) - base;
    assert!((1..=2).contains(&n), "{n} progress writes in ~1.4 s");
    assert!((row(&m, id)["progress"].as_f64().unwrap() - 0.8).abs() < 1e-6);
    // Terminal state is written immediately, even right after a progress write.
    s.update(id, Box::new(|j| j.mark_succeeded(OffsetDateTime::now_utc(), vec![], JobMetrics::default()).unwrap()))
        .await
        .unwrap();
    let r = row(&m, id);
    assert_eq!(r["status"], "succeeded");
    assert!(r["completed_at"].as_f64().is_some());
    let stored: fastvideo_protocol::Job = serde_json::from_str(r["job"].as_str().unwrap()).unwrap();
    assert_eq!(stored.progress, 1.0);
}

#[tokio::test]
async fn watchers_see_updates() {
    let m = MockD1::new();
    let s = open(&m, "w1").await;
    let j = job(ProtocolId::Fal, "w", t0());
    s.insert(j.clone()).await.unwrap();
    let mut rx = s.watch(j.id).unwrap();
    s.update(j.id, Box::new(|j| j.mark_running(OffsetDateTime::now_utc()).unwrap())).await.unwrap();
    assert!(rx.has_changed().unwrap());
    assert_eq!(rx.borrow_and_update().state, fastvideo_protocol::JobState::Running);
}

#[tokio::test]
async fn restart_recovery_and_lost_workers() {
    let m = MockD1::new();
    let (mine, theirs, done);
    {
        let a = open(&m, "w1").await;
        let b = open(&m, "w2").await;
        let x = job(ProtocolId::Fal, "mine", t0());
        let y = job(ProtocolId::Fal, "theirs", t0());
        let z = job(ProtocolId::Fal, "done", t0());
        mine = x.id;
        theirs = y.id;
        done = z.id;
        a.insert(x).await.unwrap();
        a.insert(z).await.unwrap();
        b.insert(y).await.unwrap();
        a.update(mine, Box::new(|j| j.mark_running(OffsetDateTime::now_utc()).unwrap())).await.unwrap();
        a.update(done, Box::new(|j| j.mark_cancelled(OffsetDateTime::now_utc()).unwrap())).await.unwrap();
    }
    // w1 restarts: its unfinished job fails, its finished one is untouched.
    let a = open(&m, "w1").await;
    let j = a.get(mine).await.unwrap();
    assert_eq!(j.state.error().unwrap().message, "interrupted by restart");
    assert_eq!(row(&m, mine)["status"], "failed");
    assert_eq!(a.get(done).await.unwrap().status(), JobStatus::Cancelled);
    assert_eq!(a.get(theirs).await.unwrap().status(), JobStatus::Queued, "other workers' jobs are not recovered");
    // w2 goes silent: its heartbeat ages past `stale_after`.
    m.sql("UPDATE jobs SET updated_at = updated_at - 3600000 WHERE worker = 'w2'", &[]).unwrap();
    a.sweep_expired(OffsetDateTime::now_utc()).await;
    let e = a.get(theirs).await.unwrap();
    assert_eq!(e.state.error().unwrap().kind, ErrorKind::Internal);
    assert!(e.state.error().unwrap().message.contains("lost"));
}

#[tokio::test]
async fn cancel_from_another_worker_uses_a_version_check() {
    let m = MockD1::new();
    let a = open(&m, "w1").await;
    let b = open(&m, "w2").await;
    let j = job(ProtocolId::MiniMaxV2, "1", t0());
    a.insert(j.clone()).await.unwrap();
    let out = b.update(j.id, Box::new(|j| j.mark_cancelled(OffsetDateTime::now_utc()).unwrap())).await.unwrap();
    assert_eq!(out.status(), JobStatus::Cancelled);
    assert_eq!(row(&m, j.id)["status"], "cancelled");
    assert_eq!(row(&m, j.id)["version"], 1);
    // No-op updates write nothing; unknown ids are NotFound.
    let before = m.statements().len();
    b.update(j.id, Box::new(|_| {})).await.unwrap();
    assert_eq!(m.statements().iter().skip(before).filter(|s| s.starts_with("UPDATE")).count(), 0);
    let missing = fastvideo_protocol::JobId::new();
    assert_eq!(b.update(missing, Box::new(|_| {})).await.unwrap_err(), StoreError::NotFound(missing));
}

#[tokio::test]
async fn transient_failures_are_retried_and_outages_leave_jobs_dirty() {
    let m = MockD1::new();
    let s = open(&m, "w1").await;
    m.fail_next(2, 503);
    let j = job(ProtocolId::Fal, "r", t0());
    s.insert(j.clone()).await.unwrap();
    assert_eq!(row(&m, j.id)["status"], "queued");
    // A state write during an outage: memory stays authoritative, the
    // flusher writes it once D1 is back.
    m.fail_next(4, 500);
    let out = s.update(j.id, Box::new(|j| j.mark_running(OffsetDateTime::now_utc()).unwrap())).await.unwrap();
    assert_eq!(out.status(), JobStatus::Running);
    assert_eq!(s.get(j.id).await.unwrap().status(), JobStatus::Running);
    assert_eq!(row(&m, j.id)["status"], "queued");
    assert!(s.stats().failed_writes >= 1);
    tokio::time::sleep(Duration::from_millis(1300)).await;
    assert_eq!(row(&m, j.id)["status"], "running");
    // SQL errors surface as I/O errors on insert without retries.
    let bad = MockD1::new();
    let st = D1JobStore::new(client(&bad), opts("w")).open(t0()).await.unwrap();
    bad.sql("DROP TABLE jobs", &[]).unwrap();
    assert!(matches!(st.insert(job(ProtocolId::Fal, "x", t0())).await, Err(StoreError::Io(_))));
}

#[tokio::test]
async fn remove_sweep_and_eviction() {
    let m = MockD1::new();
    let mut o = opts("w1");
    o.terminal_ttl = Duration::ZERO;
    let s = D1JobStore::new(client(&m), o).open(t0()).await.unwrap();
    let a = job(ProtocolId::Fal, "a", t0());
    let b = job(ProtocolId::Fal, "b", t0() - Duration::from_secs(7200));
    s.insert(a.clone()).await.unwrap();
    s.insert(b.clone()).await.unwrap();
    assert_eq!(s.remove(a.id).await.unwrap().id, a.id);
    assert!(s.get(a.id).await.is_none());
    assert_eq!(s.sweep_expired(OffsetDateTime::now_utc()).await, 1, "b expired an hour ago");
    assert!(m.sql("SELECT id FROM jobs", &[]).unwrap().is_empty());
    // Finished jobs leave the cache after `terminal_ttl` but stay readable.
    let c = job(ProtocolId::Fal, "c", t0());
    s.insert(c.clone()).await.unwrap();
    s.update(c.id, Box::new(|j| j.mark_cancelled(OffsetDateTime::now_utc()).unwrap())).await.unwrap();
    s.tick().await;
    assert_eq!(s.cached(), 0);
    assert_eq!(s.get(c.id).await.unwrap().status(), JobStatus::Cancelled);
}

#[tokio::test]
async fn list_matches_list_query_semantics() {
    let m = MockD1::new();
    let s = open(&m, "w1").await;
    let mem = MemJobStore::memory();
    let base = t0();
    for i in 0..12u64 {
        let proto = if i % 4 == 3 { ProtocolId::OpenAiVideos } else { ProtocolId::MiniMaxV2 };
        let mut j = job(proto, &format!("{:018}", 100 + i), base + Duration::from_secs(i / 2));
        j.owner = Some(KeyId(if i % 3 == 0 { "key_a".into() } else { "key_b".into() }));
        j.request_echo = json!({"model": if i % 2 == 0 { "MiniMax-H3" } else { "MiniMax-H3-Max" }});
        if i % 5 == 0 {
            j.resolved.task = Task::I2V;
        }
        if i % 4 == 1 {
            j.mark_cancelled(base).unwrap();
        }
        s.insert(j.clone()).await.unwrap();
        mem.insert(j).await.unwrap();
    }
    let queries = vec![
        ListQuery::default(),
        ListQuery { protocol: Some(ProtocolId::MiniMaxV2), limit: 3, ..Default::default() },
        ListQuery { protocol: Some(ProtocolId::MiniMaxV2), limit: 3, offset: 2, ..Default::default() },
        ListQuery { owner: Some(KeyId("key_a".into())), order: SortOrder::Asc, ..Default::default() },
        ListQuery { statuses: vec![JobStatus::Cancelled], ..Default::default() },
        ListQuery { statuses: vec![JobStatus::Queued, JobStatus::Cancelled], limit: 4, ..Default::default() },
        ListQuery { model: Some("MiniMax-H3-Max".into()), ..Default::default() },
        ListQuery { model: Some("fasth3".into()), limit: 2, ..Default::default() },
        ListQuery { task: Some(Task::I2V), ..Default::default() },
        ListQuery { external_ids: vec![format!("{:018}", 101), format!("{:018}", 105), "nope".into()], ..Default::default() },
        ListQuery { after: Some(format!("{:018}", 106)), limit: 4, ..Default::default() },
        ListQuery { after: Some(format!("{:018}", 106)), order: SortOrder::Asc, limit: 4, ..Default::default() },
        ListQuery { protocol: Some(ProtocolId::MiniMaxV2), after: Some(format!("{:018}", 104)), ..Default::default() },
        ListQuery { after: Some("unknown".into()), ..Default::default() },
        ListQuery {
            external_ids: (0..80).map(|i| format!("{:018}", 90 + i)).collect(),
            limit: 5,
            ..Default::default()
        },
    ];
    for q in queries {
        let a = s.list(q.clone()).await;
        let b = mem.list(q.clone()).await;
        let ids = |p: &fastvideo_protocol::Page<fastvideo_protocol::Job>| {
            p.items.iter().map(|j| j.external_id.clone()).collect::<Vec<_>>()
        };
        assert_eq!(ids(&a), ids(&b), "{q:?}");
        assert_eq!((a.total, a.has_more), (b.total, b.has_more), "{q:?}");
    }
}

#[cfg(feature = "fetch")]
#[tokio::test]
async fn over_http_with_bearer_auth() {
    let m = MockD1::new().with_token("tok");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = m.router();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let mut cfg = D1Config::new("acct", "tok", "db");
    cfg.api_base = format!("http://{addr}/client/v4");
    let s = D1JobStore::new(D1Client::http(cfg.clone()).unwrap(), opts("w1")).open(t0()).await.unwrap();
    let j = job(ProtocolId::LtxV2, "l", t0());
    s.insert(j.clone()).await.unwrap();
    assert_eq!(row(&m, j.id)["protocol"], "ltx_v2");
    m.fail_next(1, 503);
    let other = D1JobStore::new(D1Client::http(cfg.clone()).unwrap(), opts("w2")).open(t0()).await.unwrap();
    assert_eq!(other.get(j.id).await.unwrap().id, j.id);
    cfg.api_token = "wrong".into();
    let e = schema::migrate(&D1Client::http(cfg).unwrap()).await.unwrap_err();
    assert!(matches!(e, D1Error::Api { status: 401, .. }), "{e}");
}

/// A worker adopts a row a gateway inserted in one D1 round trip: the row
/// keeps its fields, takes the worker's input paths and `dispatched_at`,
/// and the conditions (finished, cancel requested, held by another live
/// worker) are checked inside the statement.
#[tokio::test]
async fn adopt_is_one_conditional_upsert() {
    let m = MockD1::new();
    let mut go = opts("gateway");
    go.hold_inserts = false;
    let gw = D1JobStore::new(client(&m), go).open(t0()).await.unwrap();
    let w1 = open(&m, "w1").await;
    let w2 = open(&m, "w2").await;

    let mut j = job(ProtocolId::Fal, "adopt-1", t0());
    j.resolved.keyframes = vec![(fastvideo_protocol::Anchor::First, "/gw/inputs/a.png".into())];
    gw.insert(j.clone()).await.unwrap();
    assert!(row(&m, j.id)["worker"].is_null());
    // The gateway logged something on the row meanwhile (a re-dispatch note):
    // the row's fields win over the dispatched copy.
    gw.update(j.id, Box::new(|j: &mut fastvideo_protocol::Job| {
        j.logs.push(fastvideo_protocol::LogLine::info("dispatching again", t0()));
    }))
    .await
    .unwrap();

    let mut mine = j.clone();
    mine.resolved.keyframes[0].1 = "/w1/inputs/0-a.png".into();
    mine.dispatched_at = Some(t0());
    let before = m.statements().len();
    let got = w1.adopt(mine.clone()).await.unwrap();
    assert_eq!(m.statements().len() - before, 1, "one statement: {:?}", &m.statements()[before..]);
    assert_eq!(got.resolved.keyframes[0].1, std::path::PathBuf::from("/w1/inputs/0-a.png"));
    assert_eq!(got.dispatched_at.map(|t| t.unix_timestamp()), Some(t0().unix_timestamp()));
    assert_eq!(got.logs.len(), 1, "the row's logs are kept");
    let r = row(&m, j.id);
    assert_eq!(r["worker"], "w1");
    let stored: fastvideo_protocol::Job = serde_json::from_str(r["job"].as_str().unwrap()).unwrap();
    assert_eq!(stored.resolved.keyframes, got.resolved.keyframes);
    assert_eq!(stored.dispatched_at, got.dispatched_at);
    // Held here: idempotent, no D1 call.
    let before = m.statements().len();
    assert_eq!(w1.adopt(mine.clone()).await.unwrap().id, j.id);
    assert_eq!(m.statements().len(), before);
    // Another worker while w1's heartbeat is fresh: refused, row unchanged.
    assert_eq!(w2.adopt(mine.clone()).await, Err(StoreError::AlreadyExists(j.id)));
    assert_eq!(row(&m, j.id)["worker"], "w1");
    // Stale heartbeat: w2 may take it over.
    m.sql("UPDATE jobs SET updated_at = 0 WHERE id = ?", &[json!(j.id.to_string())]).unwrap();
    assert_eq!(w2.adopt(mine.clone()).await.unwrap().id, j.id);
    assert_eq!(row(&m, j.id)["worker"], "w2");

    // Cancel requested or finished rows are refused.
    let c = job(ProtocolId::Fal, "adopt-2", t0());
    gw.insert(c.clone()).await.unwrap();
    gw.update(c.id, Box::new(|j: &mut fastvideo_protocol::Job| {
        j.cancel_requested = true;
    }))
    .await
    .unwrap();
    assert_eq!(w1.adopt(c.clone()).await, Err(StoreError::AlreadyExists(c.id)));
    let f = job(ProtocolId::Fal, "adopt-3", t0());
    gw.insert(f.clone()).await.unwrap();
    gw.update(f.id, Box::new(|j: &mut fastvideo_protocol::Job| {
        let _ = j.mark_failed(t0(), fastvideo_protocol::ApiError::internal("x"));
    }))
    .await
    .unwrap();
    assert_eq!(w1.adopt(f.clone()).await, Err(StoreError::AlreadyExists(f.id)));
    assert!(row(&m, f.id)["worker"].is_null());
    // No row yet (the gateway's insert was lost): the dispatched copy is inserted.
    let n = job(ProtocolId::Fal, "adopt-4", t0());
    assert_eq!(w1.adopt(n.clone()).await.unwrap().id, n.id);
    assert_eq!(row(&m, n.id)["worker"], "w1");
}
