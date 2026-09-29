//! ONE live smoke test of `D1JobStore` against the real Cloudflare D1
//! database `fv-jobs` (design §0 decision 7). Needs the `fetch` feature and
//! a token in `FV_CF_API_TOKEN` or `CLOUDFLARE_API_KEY`; without one it
//! prints a skip note and passes. The account and database ids come from
//! `FV_CF_ACCOUNT_ID` / `FV_D1_DATABASE_ID` or are looked up through the
//! Cloudflare API (first account; database named `fv-jobs`).
//!
//! The database is shared (production workers, other test runs at the same
//! time), so every write and assertion is scoped to this run: the jobs carry
//! a per-run id (`fvtest-smoke-<uuid>`) as owner, external-id prefix and
//! worker name; lists filter on that owner; cleanup deletes exactly the rows
//! with that owner or those two worker names and checks it deleted no more
//! than it wrote. It never sweeps or reaps (which touch every row), never
//! drops or alters a table, and accepts a schema newer than this build's
//! (another branch may have migrated the shared database further).

#![cfg(feature = "fetch")]

use std::time::Duration;

use fastvideo_protocol::{
    AudioPlan, Job, JobId, JobMetrics, JobStatus, JobStore, KeyId, ListQuery, ModelId, PostProcess, ProtocolId,
    ResolvedJob, SamplingOverrides, Task,
};
use fastvideo_serve_kit::d1::{schema, Stmt};
use fastvideo_serve_kit::{D1Client, D1Config, D1JobStore, D1Options};
use serde_json::{json, Value};
use time::OffsetDateTime;

fn var(n: &str) -> Option<String> {
    std::env::var(n).ok().filter(|v| !v.trim().is_empty())
}

async fn cf_get(http: &reqwest::Client, token: &str, path: &str) -> Value {
    http.get(format!("https://api.cloudflare.com/client/v4{path}"))
        .bearer_auth(token)
        .send()
        .await
        .expect("Cloudflare API")
        .json()
        .await
        .expect("Cloudflare API JSON")
}

fn resolved() -> ResolvedJob {
    ResolvedJob {
        model: ModelId::new("fasth3"),
        task: Task::T2V,
        prompt: "fv-serve live D1 smoke test".into(),
        negative_prompt: String::new(),
        seed: 1,
        width: 1344,
        height: 768,
        num_frames: 124,
        fps: 24,
        keyframes: vec![],
        references: vec![],
        audio_in: None,
        audio: AudioPlan::Native { rate: 32000, channels: 2 },
        post: PostProcess::default(),
        sampling: SamplingOverrides::default(),
        tier: None,
        recipe: None,
        edit: None,
    }
}

#[tokio::test]
async fn live_d1_smoke() {
    let Some(token) = var("FV_CF_API_TOKEN").or_else(|| var("CLOUDFLARE_API_KEY")) else {
        eprintln!("live_d1_smoke: skipped (no FV_CF_API_TOKEN / CLOUDFLARE_API_KEY)");
        return;
    };
    let http = reqwest::Client::new();
    let account = match var("FV_CF_ACCOUNT_ID") {
        Some(a) => a,
        None => {
            let v = cf_get(&http, &token, "/accounts").await;
            v["result"][0]["id"].as_str().expect("an account").to_owned()
        }
    };
    let database = match var("FV_D1_DATABASE_ID") {
        Some(d) => d,
        None => {
            let v = cf_get(&http, &token, &format!("/accounts/{account}/d1/database?name=fv-jobs")).await;
            v["result"]
                .as_array()
                .and_then(|a| a.iter().find(|d| d["name"] == "fv-jobs"))
                .and_then(|d| d["uuid"].as_str())
                .expect("D1 database fv-jobs")
                .to_owned()
        }
    };
    let tag = format!("fvtest-smoke-{}", uuid::Uuid::new_v4().simple());
    let cfg = D1Config::new(&account, &token, &database);
    let db = D1Client::http(cfg.clone()).unwrap();
    let version = schema::migrate(&db).await.unwrap();
    assert!(version >= schema::latest(), "schema {version} < {}", schema::latest());

    let mut opts = D1Options::new(&tag);
    opts.stale_after = None;
    let store = D1JobStore::new(D1Client::http(cfg.clone()).unwrap(), opts.clone())
        .open(OffsetDateTime::now_utc())
        .await
        .unwrap();

    let owner = KeyId(tag.clone());
    let now = OffsetDateTime::now_utc();
    let mut ids = Vec::new();
    for i in 0..3u32 {
        let mut j = Job::new(
            JobId::new(),
            ProtocolId::MiniMaxV2,
            format!("{tag}-{i}"),
            resolved(),
            now + Duration::from_millis(i as u64),
            Duration::from_secs(3600),
        );
        j.owner = Some(owner.clone());
        j.request_echo = json!({"model": "MiniMax-H3"});
        ids.push(j.id);
        store.insert(j).await.unwrap();
    }
    let (st, ids2, cfg2, tag2, owner2, opts2) = (store.clone(), ids.clone(), cfg.clone(), tag.clone(), owner.clone(), opts.clone());
    let result = tokio::spawn(async move {
        let (store, ids, cfg, tag, owner, opts) = (st, ids2, cfg2, tag2, owner2, opts2);
        // Running + throttled progress + success on job 0; cancel job 1.
        store
            .update(ids[0], Box::new(|j| j.mark_running(OffsetDateTime::now_utc()).unwrap()))
            .await
            .unwrap();
        for s in 1..=10 {
            store.update(ids[0], Box::new(move |j| j.set_step(s, 10))).await.unwrap();
        }
        store
            .update(
                ids[0],
                Box::new(|j| j.mark_succeeded(OffsetDateTime::now_utc(), vec![], JobMetrics::default()).unwrap()),
            )
            .await
            .unwrap();
        store
            .update(ids[1], Box::new(|j| j.mark_cancelled(OffsetDateTime::now_utc()).unwrap()))
            .await
            .unwrap();

        // Another worker sees the durable state.
        let mut o2 = opts.clone();
        o2.worker_id = format!("{tag}-b");
        let other = D1JobStore::new(D1Client::http(cfg.clone()).unwrap(), o2)
            .open(OffsetDateTime::now_utc())
            .await
            .unwrap();
        let j0 = other.get(ids[0]).await.expect("job 0 in D1");
        assert_eq!(j0.status(), JobStatus::Succeeded);
        assert_eq!(j0.progress, 1.0);
        assert_eq!(
            other.by_external(ProtocolId::MiniMaxV2, &format!("{tag}-1")).await.unwrap().status(),
            JobStatus::Cancelled
        );
        // Owner-scoped list with a status filter, newest first.
        let page = other
            .list(ListQuery { owner: Some(owner.clone()), ..Default::default() })
            .await;
        assert_eq!(page.total, 3);
        assert_eq!(page.items[0].external_id, format!("{tag}-2"));
        let queued = other
            .list(ListQuery { owner: Some(owner.clone()), statuses: vec![JobStatus::Queued], ..Default::default() })
            .await;
        assert_eq!(queued.items.iter().map(|j| j.id).collect::<Vec<_>>(), vec![ids[2]]);
        // Removal deletes the row.
        assert!(store.remove(ids[2]).await.is_some());
        assert!(other.get(ids[2]).await.is_none());
    })
    .await;
    // No deferred write may land after the cleanup below.
    store.flush_all().await;
    drop(store);
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Clean up every row of the test owner, whatever happened above.
    // Exact matches only (no LIKE pattern), and at most the rows written.
    let workers = (tag.clone(), format!("{tag}-b"));
    let mine = "owner = ? OR worker = ? OR worker = ?";
    let params = || vec![json!(tag), json!(workers.0), json!(workers.1)];
    let r = db.query(Stmt::new(format!("DELETE FROM jobs WHERE {mine}"), params())).await.unwrap();
    eprintln!("live_d1_smoke: removed {} test rows", r.changes);
    assert!(r.changes <= ids.len() as u64, "cleanup removed {} rows, more than the {} written", r.changes, ids.len());
    let left = db.query(Stmt::new(format!("SELECT COUNT(*) AS n FROM jobs WHERE {mine}"), params())).await.unwrap();
    assert_eq!(left.rows[0]["n"].as_f64(), Some(0.0));
    if let Err(e) = result {
        std::panic::resume_unwind(e.into_panic());
    }
}
