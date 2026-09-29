//! `GET /fv/v1/status`: the public, non-sensitive state of the server and
//! its pools, for the console's status strip (docs/serve/console.md).
//!
//! No credentials are needed in any auth mode, so the body carries only
//! labels, states, ages and counts: never worker URLs, pod or endpoint ids,
//! IPs, tokens or probe error texts. Workers are labelled `w1`, `w2`, … per
//! pool (single-server mode: one `local` pool with one `local` worker).
//!
//! ```json
//! {
//!   "object": "fv.status",
//!   "gateway": true,
//!   "state": "ready",
//!   "pools": [{
//!     "id": "h3", "kind": "pod", "state": "busy", "available": true,
//!     "models": ["h3-turbo"], "queued": 2, "running": 1, "last_seen_s": 1.4,
//!     "workers": [{"label": "w1", "state": "busy", "last_seen_s": 1.4, "running": 1, "queued": 0, "sessions": 0}]
//!   }],
//!   "models": {"h3-turbo": {"state": "busy", "pools": ["h3"]}},
//!   "names": {"h3-turbo": "h3-turbo", "turbo": "h3-turbo"}
//! }
//! ```
//!
//! States, best first ([`State`]): `ready`, `busy` (working, would queue),
//! `loading` (models loading, or a worker still starting: `/ping` 204),
//! `scaled_to_zero` (no worker; a serverless pool starts one on demand),
//! `draining`, `unhealthy` (answers with an error), `down` (unreachable),
//! `failed` (model loading failed, or the startup capability check found the
//! GPU cannot run it: the model then carries a `reason` and its pool lists
//! it in `failed_models`). A pool is its best worker; a model is
//! its best pool; `state` is the best pool. Serverless pools also carry
//! Runpod's `worker_counts` (idle, running, initializing, …).
//!
//! Versions (docs/serve/releases.md): each pod pool lists the builds its
//! answering workers run as `versions: [{sha, channel, workers}]` (the
//! 7-character git sha and the release channel only: no digests, image
//! names or ids) and sets `mixed_versions` when they run more than one sha;
//! the top level has this server's own `version` and `mixed_versions` when
//! any pool is mixed. The admin route `/fv/v1/gateway/pools` has the full
//! builds.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{json, Value};

/// A worker's, pool's or model's state; `Ord` is best first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Ready,
    Busy,
    Loading,
    ScaledToZero,
    Draining,
    Unhealthy,
    Down,
    Failed,
}

/// One worker as the status view shows it.
#[derive(Clone, Debug, Serialize)]
pub struct WorkerStatus {
    pub label: String,
    pub state: State,
    /// Seconds since it last answered a probe (`null`: never).
    pub last_seen_s: Option<f64>,
    pub running: u32,
    pub queued: u32,
    pub sessions: u32,
}

/// One build a pool runs, as the public views show it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct VersionCount {
    /// The 7-character git sha (`unknown` for a worker that does not say).
    pub sha: String,
    /// The release channel it was deployed from, when known.
    pub channel: Option<String>,
    /// How many of the pool's answering workers run it.
    pub workers: u32,
}

/// Groups `(short sha, channel)` per worker; `true` when there is more than
/// one sha.
pub fn versions(it: impl IntoIterator<Item = (String, Option<String>)>) -> (Vec<VersionCount>, bool) {
    let mut m: BTreeMap<(String, Option<String>), u32> = BTreeMap::new();
    for k in it {
        *m.entry(k).or_default() += 1;
    }
    let shas: std::collections::BTreeSet<&String> = m.keys().map(|(s, _)| s).collect();
    let mixed = shas.len() > 1;
    (m.into_iter().map(|((sha, channel), workers)| VersionCount { sha, channel, workers }).collect(), mixed)
}

/// This server's own `{sha, channel}` (see [`crate::build_info`]).
pub fn own_version() -> Value {
    let b = crate::build_info::BuildInfo::current();
    json!({"sha": b.git_sha_short, "channel": b.channel})
}

/// One pool as the status view shows it.
#[derive(Clone, Debug, Serialize)]
pub struct PoolStatus {
    pub id: String,
    /// `local`, `pod` or `runpod-serverless`.
    pub kind: String,
    pub state: State,
    /// A job would be dispatched now.
    pub available: bool,
    pub models: Vec<String>,
    pub queued: u32,
    pub running: u32,
    /// Seconds since any worker (or Runpod `/health`) last answered.
    pub last_seen_s: Option<f64>,
    pub workers: Vec<WorkerStatus>,
    /// Model loading progress (`{done, total}`), single-server mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loading: Option<Value>,
    /// Serverless: Runpod's worker counts (`idle`, `running`, `initializing`, …).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_counts: Option<BTreeMap<String, u64>>,
    /// The builds the answering workers run (short sha and channel).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub versions: Vec<VersionCount>,
    /// More than one sha among them.
    pub mixed_versions: bool,
    /// Models that failed on this pool's workers, with the reason (the
    /// startup capability check: "cannot run on this GPU (…, sm80): it needs
    /// FP8 tensor cores …", or a failed load). Messages only: no URLs or ids.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub failed_models: BTreeMap<String, String>,
}

/// A pod pool's state from its workers' (none: scaled to zero).
pub fn pool_state(workers: &[WorkerStatus]) -> State {
    workers.iter().map(|w| w.state).min().unwrap_or(State::ScaledToZero)
}

/// A serverless pool's state from Runpod's `/health` worker counts.
pub fn serverless_state(counts: &BTreeMap<String, u64>) -> State {
    let n = |k: &str| counts.get(k).copied().unwrap_or(0);
    if n("idle") + n("ready") > 0 {
        State::Ready
    } else if n("running") > 0 {
        State::Busy
    } else if n("initializing") > 0 {
        State::Loading
    } else if n("unhealthy") > 0 {
        State::Unhealthy
    } else {
        State::ScaledToZero
    }
}

/// Runpod `/health` `workers` counts (numbers only).
pub fn worker_counts(health: &Value) -> BTreeMap<String, u64> {
    health
        .get("workers")
        .and_then(Value::as_object)
        .map(|o| o.iter().filter_map(|(k, v)| v.as_u64().map(|n| (k.clone(), n))).collect())
        .unwrap_or_default()
}

/// The `/fv/v1/status` body. `names`: every served name, alias and id →
/// model id, so a client can find the model behind a fal app's `model`.
pub fn body(gateway: bool, pools: Vec<PoolStatus>, names: BTreeMap<String, String>) -> Value {
    let mut models: BTreeMap<String, (State, Vec<String>, Option<String>)> = BTreeMap::new();
    for p in &pools {
        for m in &p.models {
            // A model failed on this pool is `failed` there, whatever the pool's state.
            let (st, why) = match p.failed_models.get(m) {
                Some(why) => (State::Failed, Some(why.clone())),
                None => (p.state, None),
            };
            let e = models.entry(m.clone()).or_insert((st, Vec::new(), None));
            e.0 = e.0.min(st);
            if e.2.is_none() {
                e.2 = why;
            }
            e.1.push(p.id.clone());
        }
    }
    let state = pools.iter().map(|p| p.state).min().unwrap_or(State::Down);
    let models: BTreeMap<String, Value> = models
        .into_iter()
        .map(|(m, (s, p, why))| {
            let mut v = json!({"state": s, "pools": p});
            if let (State::Failed, Some(why)) = (s, why) {
                v["reason"] = Value::String(why);
            }
            (m, v)
        })
        .collect();
    let mixed = pools.iter().any(|p| p.mixed_versions);
    json!({
        "object": "fv.status",
        "gateway": gateway,
        "state": state,
        "version": own_version(),
        "mixed_versions": mixed,
        "pools": pools,
        "models": models,
        "names": names,
    })
}

/// `names` for [`body`]: ids and served names of `models`, then `aliases`.
pub fn names<'a>(models: impl Iterator<Item = &'a fastvideo_protocol::ModelCaps>, aliases: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for m in models {
        out.insert(m.id.0.clone(), m.id.0.clone());
        for n in &m.served_names {
            out.entry(n.clone()).or_insert_with(|| m.id.0.clone());
        }
    }
    for (a, m) in aliases {
        out.entry(a.clone()).or_insert_with(|| m.clone());
    }
    out
}

/// Seconds since `t`, rounded to 0.1 s.
pub fn age_s(t: Option<std::time::Instant>) -> Option<f64> {
    t.map(|t| (t.elapsed().as_secs_f64() * 10.0).round() / 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(state: State) -> WorkerStatus {
        WorkerStatus { label: "w1".into(), state, last_seen_s: Some(1.0), running: 0, queued: 0, sessions: 0 }
    }

    #[test]
    fn a_pool_is_its_best_worker_and_empty_is_scaled_to_zero() {
        assert_eq!(pool_state(&[]), State::ScaledToZero);
        assert_eq!(pool_state(&[w(State::Down), w(State::Loading)]), State::Loading);
        assert_eq!(pool_state(&[w(State::Busy), w(State::Ready), w(State::Down)]), State::Ready);
        assert_eq!(pool_state(&[w(State::Down), w(State::Unhealthy)]), State::Unhealthy);
    }

    #[test]
    fn serverless_counts_map_to_states() {
        let c = |pairs: &[(&str, u64)]| pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect::<BTreeMap<_, _>>();
        assert_eq!(serverless_state(&c(&[])), State::ScaledToZero);
        assert_eq!(serverless_state(&c(&[("idle", 1), ("running", 2)])), State::Ready);
        assert_eq!(serverless_state(&c(&[("running", 2)])), State::Busy);
        assert_eq!(serverless_state(&c(&[("initializing", 1)])), State::Loading);
        assert_eq!(worker_counts(&json!({"workers": {"idle": 2, "x": "y"}})), c(&[("idle", 2)]));
    }

    #[test]
    fn models_take_their_best_pool() {
        let pool = |id: &str, state: State, models: &[&str]| PoolStatus {
            id: id.into(),
            kind: "pod".into(),
            state,
            available: state <= State::Loading,
            models: models.iter().map(|m| m.to_string()).collect(),
            queued: 0,
            running: 0,
            last_seen_s: None,
            workers: vec![],
            loading: None,
            worker_counts: None,
            versions: vec![],
            mixed_versions: false,
            failed_models: BTreeMap::new(),
        };
        let b = body(true, vec![pool("a", State::Down, &["m1", "m2"]), pool("b", State::Busy, &["m2"])], BTreeMap::new());
        assert_eq!(b["state"], "busy");
        assert_eq!(b["models"]["m1"], json!({"state": "down", "pools": ["a"]}));
        assert_eq!(b["models"]["m2"], json!({"state": "busy", "pools": ["a", "b"]}));
        assert_eq!(body(false, vec![], BTreeMap::new())["state"], "down");
        assert_eq!(b["mixed_versions"], false);
        assert_eq!(b["version"]["sha"], crate::build_info::BuildInfo::current().git_sha_short);
    }

    #[test]
    fn a_failed_model_carries_its_reason() {
        let mut failed = BTreeMap::new();
        failed.insert("h3-turbo".to_owned(), "model `h3-turbo` cannot run on this GPU (NVIDIA A100 80GB, sm80)".to_owned());
        let p = PoolStatus {
            id: "local".into(),
            kind: "local".into(),
            state: State::Failed,
            available: false,
            models: vec!["h3-turbo".into(), "wan".into()],
            queued: 0,
            running: 0,
            last_seen_s: None,
            workers: vec![],
            loading: None,
            worker_counts: None,
            versions: vec![],
            mixed_versions: false,
            failed_models: failed,
        };
        let b = body(false, vec![p], BTreeMap::new());
        assert_eq!(b["models"]["h3-turbo"]["state"], "failed");
        assert!(b["models"]["h3-turbo"]["reason"].as_str().unwrap().contains("sm80"));
        assert!(b["models"]["wan"].get("reason").is_none());
        assert!(b["pools"][0]["failed_models"]["h3-turbo"].is_string());
    }

    #[test]
    fn versions_group_workers_and_flag_a_mix() {
        let k = |s: &str, c: Option<&str>| (s.to_owned(), c.map(str::to_owned));
        let (v, mixed) = versions([k("aaaaaaa", Some("stable")), k("aaaaaaa", Some("stable"))]);
        assert!(!mixed);
        assert_eq!(v, vec![VersionCount { sha: "aaaaaaa".into(), channel: Some("stable".into()), workers: 2 }]);
        let (v, mixed) = versions([k("bbbbbbb", Some("stable")), k("aaaaaaa", Some("stable")), k("aaaaaaa", None)]);
        assert!(mixed);
        assert_eq!(v.len(), 3);
        assert_eq!(serde_json::to_value(&v[0]).unwrap(), json!({"sha": "aaaaaaa", "channel": null, "workers": 1}));
        // One sha under two channel names is not a mix.
        assert!(!versions([k("aaaaaaa", Some("stable")), k("aaaaaaa", Some("latest"))]).1);
        assert_eq!(versions(Vec::new()), (vec![], false));
    }
}
