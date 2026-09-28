//! The controller against the simulator: serverless and pod pools, leader
//! election, dry run, the balance floor, the admin route, and the harness.

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use fastvideo_autoscale::admin::{admin_router, bearer, ADMIN_PATH};
use fastvideo_autoscale::config::{AutoscaleConfig, PodConfig, PoolConfig, PoolKind, ServerlessConfig};
use fastvideo_autoscale::controller::Controller;
use fastvideo_autoscale::lease::MemoryLease;
use fastvideo_autoscale::provider::Providers;
use fastvideo_autoscale::sim::harness::{self, pool_for, Scenario, Strategy, SIM_EPOCH};
use fastvideo_autoscale::sim::trace::{self, Arrival, Rng};
use fastvideo_autoscale::sim::{FamilyProfile, SharedWorld, SimProvider, SimSignals, SimWorld};
use fastvideo_autoscale::Action;
use tower::ServiceExt;

fn profile() -> FamilyProfile {
    // wan-like, no uncached pulls: deterministic cold starts of 120 s.
    FamilyProfile { uncached_share: 0.0, ..FamilyProfile::wan() }
}

fn serverless_pool() -> PoolConfig {
    PoolConfig {
        name: "wan".into(),
        family: "wan".into(),
        kind: PoolKind::Serverless,
        max_workers: 3,
        slo_queue_wait_s: 60.0,
        default_job_s: 6.4,
        cold_start_s: 120.0,
        idle_timeout_s: 300.0,
        serverless: ServerlessConfig { endpoint_id: "sim".into(), scaler_value: 30, idle_timeout_s: 60, ..ServerlessConfig::default() },
        ..PoolConfig::default()
    }
}

fn pod_pool() -> PoolConfig {
    PoolConfig {
        kind: PoolKind::Pod,
        pod: PodConfig {
            template_id: "tpl".into(),
            gpu_types: vec!["NVIDIA H100 80GB HBM3".into()],
            max_lifetime_s: 3.0 * 3600.0,
            ..PodConfig::default()
        },
        ..serverless_pool()
    }
}

fn setup(pool: PoolConfig, arrivals: Vec<Arrival>, dry_run: bool) -> (SharedWorld, Controller) {
    let world = Arc::new(Mutex::new(SimWorld::new(SIM_EPOCH, 1, 30.0)));
    {
        let mut w = world.lock().unwrap();
        w.add_pool(pool.clone(), profile(), 4.18, None);
        w.load_trace(&pool.name, arrivals);
    }
    let ctrl = controller(pool, &world, Arc::new(MemoryLease::default()), "a", dry_run);
    (world, ctrl)
}

fn controller(pool: PoolConfig, world: &SharedWorld, lease: Arc<MemoryLease>, holder: &str, dry_run: bool) -> Controller {
    let cfg = AutoscaleConfig {
        enabled: true,
        dry_run,
        interval_s: 15.0,
        balance_interval_s: 15.0,
        pools: vec![pool],
        ..AutoscaleConfig::default()
    };
    let sim = Arc::new(SimProvider(world.clone()));
    let providers = Providers::new(sim.clone()).with(PoolKind::Serverless, sim.clone()).with(PoolKind::Pod, sim);
    Controller::new(cfg, providers, Arc::new(SimSignals(world.clone())), lease, holder)
}

async fn run_for(world: &SharedWorld, ctrls: &[&Controller], from: f64, secs: f64) {
    let mut t = from;
    while t < from + secs {
        for c in ctrls {
            c.tick(t).await;
        }
        t += 15.0;
        world.lock().unwrap().advance_to(t);
    }
}

fn burst(at: f64, n: usize) -> Vec<Arrival> {
    (0..n).map(|i| Arrival::job(SIM_EPOCH + at + i as f64)).collect()
}

#[tokio::test]
async fn serverless_burst_raises_min_then_returns_to_zero() {
    let (world, ctrl) = setup(serverless_pool(), burst(60.0, 30), false);
    run_for(&world, &[&ctrl], SIM_EPOCH, 90.0).await;
    let ep = world.lock().unwrap().endpoint("wan").unwrap();
    assert!(ep.workers_min >= 2, "min raised ahead of Runpod's scaler: {ep:?}");
    assert_eq!(ep.workers_max, 3);
    run_for(&world, &[&ctrl], SIM_EPOCH + 90.0, 1800.0).await;
    let w = world.lock().unwrap();
    assert_eq!(w.endpoint("wan").unwrap().workers_min, 0, "back to zero after idle");
    assert_eq!(w.worker_count("wan"), 0, "Runpod reaped the idle workers");
    let st = w.stats("wan");
    assert_eq!(st.jobs, 30);
    assert_eq!(st.busy_delete_attempts, 0);
    drop(w);
    let status = ctrl.status();
    assert!(status.leader);
    assert!(status.history.iter().any(|d| d.action == Action::ScaleUp));
    assert!(status.history.iter().any(|d| d.action == Action::ScaleDown));
}

#[tokio::test]
async fn pod_pool_creates_drains_and_deletes_never_busy() {
    let mut arrivals = burst(30.0, 40);
    // A 20-minute stream that must survive every scale-down.
    arrivals.push(Arrival::stream(SIM_EPOCH + 45.0, 1200.0));
    let (world, ctrl) = setup(pod_pool(), arrivals, false);
    run_for(&world, &[&ctrl], SIM_EPOCH, 3600.0).await;
    let w = world.lock().unwrap();
    let st = w.stats("wan");
    assert_eq!(st.jobs, 40);
    assert_eq!(st.streams, 1);
    assert_eq!(st.busy_delete_attempts, 0, "the policy never asked to delete a busy pod");
    assert!(st.max_workers >= 2);
    assert_eq!(w.worker_count("wan"), 0, "all pods deleted after the stream ended and the idle timeout");
    assert_eq!(w.outstanding("wan"), 0);
}

#[tokio::test]
async fn pods_past_max_lifetime_are_replaced() {
    let mut pool = pod_pool();
    pool.warm_min = 1;
    pool.pod.max_lifetime_s = 1800.0;
    let (world, ctrl) = setup(pool, vec![], false);
    run_for(&world, &[&ctrl], SIM_EPOCH, 3.0 * 3600.0).await;
    let w = world.lock().unwrap();
    let st = w.stats("wan");
    // One warm pod at a time, each replaced about every 30 min.
    assert!((5..=8).contains(&st.cold_starts), "{}", st.cold_starts);
    assert!(w.worker_count("wan") >= 1);
    assert!(st.max_workers <= 2, "replacement overlaps by one pod at most");
}

#[tokio::test]
async fn only_the_lease_holder_applies() {
    let world = Arc::new(Mutex::new(SimWorld::new(SIM_EPOCH, 1, 30.0)));
    {
        let mut w = world.lock().unwrap();
        w.add_pool(serverless_pool(), profile(), 4.18, None);
        w.load_trace("wan", burst(0.0, 20));
    }
    world.lock().unwrap().advance_to(SIM_EPOCH + 1.0);
    let lease = Arc::new(MemoryLease::default());
    let a = controller(serverless_pool(), &world, lease.clone(), "replica-a", false);
    let b = controller(serverless_pool(), &world, lease, "replica-b", false);
    let ra = a.tick(SIM_EPOCH + 1.0).await;
    let rb = b.tick(SIM_EPOCH + 1.0).await;
    assert!(ra.leader && !rb.leader);
    assert!(ra.pools[0].applied.is_some());
    assert!(rb.pools[0].applied.is_none(), "the follower decides but does not apply");
    assert_eq!(world.lock().unwrap().stats("wan").endpoint_patches, 1);
    // The leader stops renewing: after the TTL (60 s) the other replica takes over.
    let rb = b.tick(SIM_EPOCH + 30.0).await;
    assert!(!rb.leader);
    let rb = b.tick(SIM_EPOCH + 62.0).await;
    assert!(rb.leader);
}

#[tokio::test]
async fn dry_run_decides_without_applying() {
    let (world, ctrl) = setup(serverless_pool(), burst(0.0, 20), true);
    let r = ctrl.tick(SIM_EPOCH + 5.0).await;
    world.lock().unwrap().advance_to(SIM_EPOCH + 5.0);
    let r = if r.pools[0].decision.target > 0 { r } else { ctrl.tick(SIM_EPOCH + 20.0).await };
    assert!(r.dry_run && r.leader);
    assert_eq!(r.pools[0].decision.action, Action::ScaleUp);
    assert!(r.pools[0].applied.is_none());
    assert_eq!(world.lock().unwrap().endpoint("wan").unwrap().workers_min, 0, "nothing written");
    // Switching dry run off applies the next decision.
    ctrl.set_dry_run(false);
    let r = ctrl.tick(SIM_EPOCH + 35.0).await;
    assert!(r.pools[0].applied.is_some());
    assert!(world.lock().unwrap().endpoint("wan").unwrap().workers_min > 0);
}

#[tokio::test]
async fn balance_floor_stops_everything() {
    let (world, ctrl) = setup(serverless_pool(), burst(0.0, 20), false);
    run_for(&world, &[&ctrl], SIM_EPOCH, 60.0).await;
    assert!(world.lock().unwrap().endpoint("wan").unwrap().workers_min > 0);
    world.lock().unwrap().balance_usd = 7.5;
    let r = ctrl.tick(SIM_EPOCH + 75.0).await;
    let d = &r.pools[0].decision;
    assert_eq!(d.action, Action::HardStop);
    let ep = world.lock().unwrap().endpoint("wan").unwrap();
    assert_eq!(ep.workers_min, 0);
    assert_eq!(ep.workers_max, d.busy, "only busy workers may stay");
}

#[tokio::test]
async fn admin_route_needs_the_token() {
    let (_world, ctrl) = setup(serverless_pool(), vec![], true);
    let ctrl = Arc::new(ctrl);
    ctrl.tick(SIM_EPOCH).await;
    let app = admin_router(ctrl.clone(), bearer("s3cret"));
    let r = app.clone().oneshot(Request::get(ADMIN_PATH).body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(r.status(), StatusCode::UNAUTHORIZED);
    let r = app
        .clone()
        .oneshot(Request::get(ADMIN_PATH).header("authorization", "Bearer wrong!").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::UNAUTHORIZED);
    let r = app
        .clone()
        .oneshot(Request::get(ADMIN_PATH).header("authorization", "Bearer s3cret").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    let body = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["dry_run"], true);
    assert_eq!(v["last"]["pools"][0]["decision"]["pool"], "wan");
    let r = app
        .oneshot(
            Request::post(ADMIN_PATH)
                .header("authorization", "Bearer s3cret")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"dry_run": false}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    assert!(!ctrl.dry_run());
}

#[test]
fn harness_autoscaler_beats_runpod_only_on_steady_h3() {
    let p = FamilyProfile::h3();
    let mut rng = Rng::new(42);
    let mut arrivals = trace::steady(&mut rng, SIM_EPOCH, 3.0, 60.0);
    trace::jitter(&mut rng, &mut arrivals, 0.2);
    let mk = |strategy| Scenario {
        trace: "steady".into(),
        profile: p.clone(),
        pool: pool_for(&p, 4),
        arrivals: arrivals.clone(),
        hours: 3.0,
        strategy,
        seed: 42,
        price_usd_per_hr: 4.18,
    };
    let a = harness::run(&mk(Strategy::Autoscaler));
    let b = harness::run(&mk(Strategy::RunpodOnly));
    assert_eq!(a.jobs, b.jobs);
    assert_eq!(a.unfinished + b.unfinished, 0);
    assert!(a.wait_p95_s < b.wait_p95_s, "p95 {} vs {}", a.wait_p95_s, b.wait_p95_s);
    assert!(a.cold_starts < b.cold_starts);
    assert!(a.within_slo >= 0.95, "{}", a.within_slo);
    // Deterministic: the same scenario gives the same report.
    let a2 = harness::run(&mk(Strategy::Autoscaler));
    assert_eq!((a.wait_p95_s, a.gpu_hours, a.cold_starts), (a2.wait_p95_s, a2.gpu_hours, a2.cold_starts));
}
