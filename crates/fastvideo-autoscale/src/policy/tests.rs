//! Policy tests on a simulated clock (plain `f64` seconds).

use super::*;
use crate::config::{PodConfig, ScheduleRule, ServerlessConfig};
use crate::types::Worker;

const T0: f64 = 1_790_553_600.0; // Monday 00:00 UTC

fn serverless(name: &str) -> PoolConfig {
    PoolConfig {
        name: name.into(),
        family: "h3".into(),
        kind: PoolKind::Serverless,
        max_workers: 6,
        slo_queue_wait_s: 60.0,
        default_job_s: 20.0,
        cold_start_s: 130.0,
        scale_up_step: 2,
        scale_up_cooldown_s: 30.0,
        idle_timeout_s: 300.0,
        scale_down_cooldown_s: 120.0,
        serverless: ServerlessConfig { endpoint_id: "ep".into(), ..ServerlessConfig::default() },
        ..PoolConfig::default()
    }
}

fn pod(name: &str) -> PoolConfig {
    PoolConfig {
        kind: PoolKind::Pod,
        pod: PodConfig {
            template_id: "tpl".into(),
            gpu_types: vec!["NVIDIA H200".into()],
            max_lifetime_s: 3600.0,
            boot_timeout_s: 600.0,
            ..PodConfig::default()
        },
        ..serverless(name)
    }
}

fn cfg(pools: Vec<PoolConfig>) -> AutoscaleConfig {
    AutoscaleConfig { enabled: true, dry_run: false, pools, ..AutoscaleConfig::default() }
}

fn sig(pool: &str, queued: u32, oldest: f64, running: u32) -> PoolSignals {
    PoolSignals { pool: pool.into(), queued, oldest_queued_s: oldest, running_jobs: running, ..Default::default() }
}

fn worker(id: &str, state: WorkerState, busy: u32, created: f64) -> Worker {
    Worker {
        id: id.into(),
        state,
        busy,
        created_at_s: created,
        ready_at_s: (state != WorkerState::Starting).then_some(created + 100.0),
        url: None,
        gpu_type: None,
        usd_per_hr: None,
    }
}

fn endpoint_obs(min: u32, workers: Vec<Worker>) -> PoolObservation {
    PoolObservation {
        workers,
        endpoint: Some(EndpointSettings {
            workers_min: min,
            workers_max: 6,
            idle_timeout_s: 60,
            scaler_type: "QUEUE_DELAY".into(),
            scaler_value: 2,
        }),
    }
}

fn inputs(now: f64, sigs: Vec<PoolSignals>, obs: Vec<(&str, PoolObservation)>) -> Inputs {
    Inputs {
        now_s: now,
        signals: sigs.into_iter().map(|s| (s.pool.clone(), s)).collect(),
        observations: obs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect(),
        balance_usd: Some(30.0),
    }
}

fn one(p: &mut Policy, c: &AutoscaleConfig, i: Inputs) -> PoolDecision {
    p.decide(c, &i).remove(0)
}

#[test]
fn idle_pool_stays_at_zero() {
    let c = cfg(vec![serverless("a")]);
    let mut p = Policy::new();
    let d = one(&mut p, &c, inputs(T0, vec![sig("a", 0, 0.0, 0)], vec![("a", endpoint_obs(0, vec![]))]));
    assert_eq!(d.action, Action::Hold);
    assert_eq!(d.target, 0);
    let e = d.endpoint.unwrap();
    assert_eq!((e.workers_min, e.workers_max), (0, 6));
}

#[test]
fn warm_min_and_schedule_floors() {
    let mut pool = serverless("a");
    pool.warm_min = 1;
    pool.schedule = vec![ScheduleRule { days: vec![0], start_hour: 9, end_hour: 17, min_workers: 3 }];
    let c = cfg(vec![pool]);
    let mut p = Policy::new();
    let d = one(&mut p, &c, inputs(T0, vec![], vec![("a", endpoint_obs(0, vec![]))]));
    assert_eq!((d.floor, d.target), (1, 1), "warm minimum at night");
    // Monday 10:00: the schedule floor jumps straight to 3 (floors ignore the step).
    let d = one(&mut p, &c, inputs(T0 + 10.0 * 3600.0, vec![], vec![("a", endpoint_obs(1, vec![]))]));
    assert_eq!((d.floor, d.target, d.action), (3, 3, Action::ScaleUp));
    assert_eq!(d.endpoint.unwrap().workers_min, 3);
}

#[test]
fn queued_jobs_get_workers_at_once() {
    let c = cfg(vec![serverless("a")]);
    let mut p = Policy::new();
    // 20 queued 20 s jobs, SLO 60 s: each worker starts 4 within the SLO
    // -> 5 workers, at once (the step limits only rate-driven growth).
    let d = one(&mut p, &c, inputs(T0, vec![sig("a", 20, 1.0, 0)], vec![("a", endpoint_obs(0, vec![]))]));
    assert_eq!(d.demand, 5.0);
    assert_eq!((d.action, d.target), (Action::ScaleUp, 5));
    assert_eq!(d.endpoint.unwrap().workers_min, 5);
}

#[test]
fn rate_driven_growth_steps_with_cooldown() {
    let c = cfg(vec![serverless("a")]);
    let mut p = Policy::new();
    let ready = |n: usize| (0..n).map(|i| worker(&format!("w{i}"), WorkerState::Ready, 0, T0 - 900.0)).collect::<Vec<_>>();
    // 12 jobs/min of 20 s at 80 % utilization: 5 workers. Nothing queued.
    let mut s = sig("a", 0, 0.0, 0);
    let mut min = 1;
    let mut seen = Vec::new();
    for k in 0..=8u32 {
        s.arrivals_total = u64::from(k) * 3;
        let d = one(&mut p, &c, inputs(T0 + f64::from(k) * 15.0, vec![s.clone()], vec![("a", endpoint_obs(min, ready(min as usize)))]));
        if d.action == Action::ScaleUp {
            seen.push((k, d.target));
        } else if k > 1 {
            assert!(d.target == min && (d.reasons.iter().any(|r| r.contains("cooldown")) || d.target == 5), "{d:?}");
        }
        min = d.target;
    }
    // +2 per 30 s cooldown (ticks every 15 s) up to 5.
    assert_eq!(seen, vec![(1, 3), (3, 5)]);
}

#[test]
fn queue_age_breach_bypasses_cooldown() {
    let c = cfg(vec![serverless("a")]);
    let mut p = Policy::new();
    let one_busy = || vec![worker("w1", WorkerState::Ready, 1, T0 - 500.0)];
    // Two 20 s jobs behind one busy worker start within 40 s: no scale-up.
    let d = one(&mut p, &c, inputs(T0, vec![sig("a", 2, 1.0, 1)], vec![("a", endpoint_obs(1, one_busy()))]));
    assert_eq!((d.action, d.target), (Action::Hold, 1));
    // The busy worker's job overran: 5 s later the oldest job is 40 s old
    // (> 0.5 × 60 s) and nobody idle or booting takes it -> +1 despite the cooldown.
    let d = one(&mut p, &c, inputs(T0 + 5.0, vec![sig("a", 2, 40.0, 1)], vec![("a", endpoint_obs(1, one_busy()))]));
    assert_eq!((d.action, d.target), (Action::ScaleUp, 2));
    assert!(d.reasons.iter().any(|r| r.starts_with("queue age")));
    let d = one(&mut p, &c, inputs(T0 + 10.0, vec![sig("a", 2, 45.0, 1)], vec![("a", endpoint_obs(2, one_busy()))]));
    assert_eq!(d.target, 3, "still nobody to take it: another one");
    // Not when a booting worker will take the job.
    let mut p = Policy::new();
    let ws = vec![worker("w1", WorkerState::Ready, 1, T0 - 500.0), worker("w2", WorkerState::Starting, 0, T0 - 10.0)];
    let d = one(&mut p, &c, inputs(T0, vec![sig("a", 1, 40.0, 1)], vec![("a", endpoint_obs(2, ws))]));
    assert!(!d.reasons.iter().any(|r| r.starts_with("queue age")), "{:?}", d.reasons);
}

#[test]
fn scale_down_waits_for_idle_timeout_and_steps_by_one() {
    let c = cfg(vec![serverless("a")]);
    let mut p = Policy::new();
    let idle = |n: usize| (0..n).map(|i| worker(&format!("w{i}"), WorkerState::Ready, 0, T0 - 1000.0)).collect::<Vec<_>>();
    let mut t = T0;
    let mut min = 3;
    let d = one(&mut p, &c, inputs(t, vec![sig("a", 0, 0.0, 0)], vec![("a", endpoint_obs(min, idle(3)))]));
    assert_eq!(d.action, Action::Hold, "just went idle");
    let mut downs = Vec::new();
    while t < T0 + 1200.0 {
        t += 15.0;
        let d = one(&mut p, &c, inputs(t, vec![sig("a", 0, 0.0, 0)], vec![("a", endpoint_obs(min, idle(min as usize)))]));
        if d.action == Action::ScaleDown {
            assert_eq!(d.target, min - 1, "one at a time");
            downs.push(t - T0);
            min = d.target;
        }
    }
    // First after the 300 s idle timeout, then one per 120 s cooldown.
    assert_eq!(downs, vec![300.0, 420.0, 540.0]);
    assert_eq!(min, 0);
}

#[test]
fn hysteresis_blocks_scale_down_near_capacity() {
    let c = cfg(vec![serverless("a")]);
    let mut p = Policy::new();
    // 2 workers, 1 busy, arrivals steady at 1.8 jobs/min x 20 s = 0.6 / 0.8 = 0.75 workers: need 1,
    // steady 0.75 > (2-1) x 0.8 = 0.8? No -> 0.75 <= 0.8: may scale down after idle.
    // At 2.4/min: 1.0 > 0.8, blocked.
    let ws = || vec![worker("w1", WorkerState::Ready, 1, T0 - 900.0), worker("w2", WorkerState::Ready, 0, T0 - 900.0)];
    let mut arrivals = 0;
    let mut last = Action::Hold;
    for k in 0..60 {
        let t = T0 + f64::from(k) * 15.0;
        arrivals += if k % 5 == 4 { 3 } else { 0 }; // 3 per 75 s = 2.4/min
        let mut s = sig("a", 0, 0.0, 1);
        s.arrivals_total = arrivals;
        last = one(&mut p, &c, inputs(t, vec![s], vec![("a", endpoint_obs(2, ws()))])).action;
        assert_ne!(last, Action::ScaleDown, "at {k}");
    }
    assert_eq!(last, Action::Hold);
}

#[test]
fn never_below_busy_and_pods_drain_only_idle() {
    let c = cfg(vec![pod("p")]);
    let mut p = Policy::new();
    let ws = vec![
        worker("a", WorkerState::Ready, 1, T0 - 900.0),
        worker("b", WorkerState::Ready, 1, T0 - 800.0),
        worker("c", WorkerState::Ready, 0, T0 - 700.0),
        worker("d", WorkerState::Ready, 0, T0 - 600.0),
    ];
    let obs = PoolObservation { workers: ws, endpoint: None };
    let mut last = None;
    for k in 0..100 {
        let d = one(&mut p, &c, inputs(T0 + f64::from(k) * 15.0, vec![sig("p", 0, 0.0, 2)], vec![("p", obs.clone())]));
        assert!(d.target >= 2);
        assert!(!d.drain.iter().any(|id| id == "a" || id == "b"), "busy worker drained: {:?}", d.drain);
        assert!(!d.delete.iter().any(|id| id == "a" || id == "b"));
        if d.action == Action::ScaleDown {
            last = Some(d);
            break;
        }
    }
    let d = last.expect("scaled down");
    assert_eq!(d.target, 3);
    assert_eq!(d.drain, vec!["c".to_string()], "oldest idle worker first");
    assert!(d.delete.is_empty(), "drain before delete");
}

#[test]
fn pods_create_delete_boot_timeout_and_draining() {
    let c = cfg(vec![pod("p")]);
    let mut p = Policy::new();
    let ws = vec![
        worker("slow", WorkerState::Starting, 0, T0 - 700.0), // past the 600 s boot timeout
        worker("gone", WorkerState::Draining, 0, T0 - 900.0), // idle draining: delete
        worker("hold", WorkerState::Draining, 1, T0 - 900.0), // busy draining: keep
    ];
    let d = one(
        &mut p,
        &c,
        inputs(T0, vec![sig("p", 4, 5.0, 1)], vec![("p", PoolObservation { workers: ws, endpoint: None })]),
    );
    assert!(d.delete.contains(&"slow".to_string()) && d.delete.contains(&"gone".to_string()));
    assert!(!d.delete.contains(&"hold".to_string()));
    // Demand: 1 busy + 4 queued (20 s jobs, 4 per worker per SLO) = 2; the
    // busy draining worker is taken back instead of a new pod.
    assert_eq!(d.target, 2);
    assert_eq!(d.undrain, vec!["hold".to_string()]);
    assert_eq!(d.create, 1, "one new pod next to the undrained one: {d:?}");
}

#[test]
fn pod_max_lifetime_replaces_then_drains() {
    let mut pool = pod("p");
    pool.warm_min = 1;
    let c = cfg(vec![pool]);
    let mut p = Policy::new();
    let old = worker("old", WorkerState::Ready, 0, T0 - 4000.0);
    let d = one(&mut p, &c, inputs(T0, vec![], vec![("p", PoolObservation { workers: vec![old.clone()], endpoint: None })]));
    assert_eq!(d.create, 1, "replacement first");
    assert!(d.drain.is_empty(), "keeps serving until the replacement is ready");
    let fresh = worker("new", WorkerState::Ready, 0, T0 - 10.0);
    let d = one(
        &mut p,
        &c,
        inputs(T0 + 200.0, vec![], vec![("p", PoolObservation { workers: vec![old, fresh], endpoint: None })]),
    );
    assert_eq!(d.create, 0);
    assert_eq!(d.drain, vec!["old".to_string()]);
}

#[test]
fn pool_budget_caps_and_global_budget_by_priority() {
    let mut a = serverless("a");
    a.budget_usd_per_hr = 9.0; // 2 × $4.18
    let c = cfg(vec![a]);
    let mut p = Policy::new();
    let d = one(&mut p, &c, inputs(T0, vec![sig("a", 40, 1.0, 0)], vec![("a", endpoint_obs(0, vec![]))]));
    assert_eq!(d.cap, 2);
    assert_eq!(d.endpoint.unwrap().workers_max, 2);

    let mut hi = serverless("hi");
    hi.priority = 1;
    hi.scale_up_step = 0;
    let mut lo = serverless("lo");
    lo.scale_up_step = 0;
    lo.warm_min = 1;
    let mut c = cfg(vec![lo, hi]);
    c.budget_usd_per_hr = 4.18 * 4.0 + 0.01;
    let mut p = Policy::new();
    let ds = p.decide(
        &c,
        &inputs(
            T0,
            vec![sig("hi", 40, 1.0, 0), sig("lo", 40, 1.0, 0)],
            vec![("hi", endpoint_obs(0, vec![])), ("lo", endpoint_obs(0, vec![]))],
        ),
    );
    // Floors first (lo's warm 1), then priority: hi gets the other 3.
    let (lo, hi) = (&ds[0], &ds[1]);
    assert_eq!((lo.target, hi.target), (1, 3), "{:?} / {:?}", lo.reasons, hi.reasons);
    assert!(hi.reasons.iter().any(|r| r.starts_with("global budget")));
}

#[test]
fn balance_floor_is_a_hard_stop() {
    let c = cfg(vec![serverless("s"), pod("p")]);
    let mut p = Policy::new();
    let mut i = inputs(
        T0,
        vec![sig("s", 10, 100.0, 1), sig("p", 10, 100.0, 1)],
        vec![
            ("s", endpoint_obs(3, vec![worker("s1", WorkerState::Ready, 1, T0 - 99.0)])),
            (
                "p",
                PoolObservation {
                    workers: vec![
                        worker("p1", WorkerState::Ready, 1, T0 - 99.0),
                        worker("p2", WorkerState::Ready, 0, T0 - 99.0),
                        worker("p3", WorkerState::Starting, 0, T0 - 9.0),
                    ],
                    endpoint: None,
                },
            ),
        ],
    );
    i.balance_usd = Some(7.99);
    let ds = p.decide(&c, &i);
    assert!(ds.iter().all(|d| d.action == Action::HardStop));
    let e = ds[0].endpoint.clone().unwrap();
    assert_eq!((e.workers_min, e.workers_max), (0, 1), "max = busy workers");
    assert_eq!(ds[1].drain, vec!["p2".to_string()]);
    assert_eq!(ds[1].delete, vec!["p3".to_string()]);
    assert_eq!(ds[1].create, 0);
}

#[test]
fn unknown_balance_blocks_scale_up() {
    let c = cfg(vec![serverless("a")]);
    let mut p = Policy::new();
    let mut i = inputs(T0, vec![sig("a", 5, 1.0, 0)], vec![("a", endpoint_obs(0, vec![]))]);
    i.balance_usd = None;
    let d = one(&mut p, &c, i);
    assert_eq!(d.target, 0);
    assert!(d.reasons.iter().any(|r| r.contains("balance unknown")));
}

#[test]
fn predictive_scale_up_covers_the_cold_start_backlog() {
    let mut pool = serverless("a");
    pool.scale_up_step = 0;
    let c = cfg(vec![pool]);
    let mut p = Policy::new();
    // One busy worker; arrivals ramp to 6/min of 20 s jobs (μ = 3/min for one worker).
    let ws = || vec![worker("w1", WorkerState::Ready, 1, T0 - 900.0)];
    let mut ups = Vec::new();
    for k in 0..=8u32 {
        let t = T0 + f64::from(k) * 15.0;
        let mut s = sig("a", k / 2, 5.0, 1);
        s.arrivals_total = u64::from(k) * 3 / 2; // 6 per minute
        let d = one(&mut p, &c, inputs(t, vec![s], vec![("a", endpoint_obs(1, ws()))]));
        if d.action == Action::ScaleUp {
            ups.push(d);
        }
    }
    let d = ups.pop().expect("scaled up");
    assert!(d.reasons.iter().any(|r| r.starts_with("predictive") || r.starts_with("steady")), "{:?}", d.reasons);
    assert!(d.estimates.projected_queue > 5.0, "{:?}", d.estimates);
    assert!(d.target >= 3, "{d:?}");
}

#[test]
fn serverless_max_never_below_busy_and_decisions_repeat() {
    let mut a = serverless("a");
    a.max_workers = 1;
    let c = cfg(vec![a]);
    let ws = vec![worker("w1", WorkerState::Ready, 1, T0), worker("w2", WorkerState::Ready, 1, T0)];
    let i = inputs(T0, vec![sig("a", 0, 0.0, 2)], vec![("a", endpoint_obs(2, ws))]);
    let d1 = one(&mut Policy::new(), &c, i.clone());
    let d2 = one(&mut Policy::new(), &c, i);
    assert_eq!(d1, d2, "deterministic");
    assert_eq!(d1.endpoint.unwrap().workers_max, 2);
}

#[test]
fn measured_cold_starts_and_job_durations_feed_the_estimates() {
    let c = cfg(vec![pod("p")]);
    let mut p = Policy::new();
    let mut w = worker("x", WorkerState::Ready, 0, T0);
    w.ready_at_s = Some(T0 + 200.0);
    let mut s = sig("p", 0, 0.0, 0);
    s.recent_job_s = Some(40.0);
    let d = one(&mut p, &c, inputs(T0 + 300.0, vec![s], vec![("p", PoolObservation { workers: vec![w], endpoint: None })]));
    assert_eq!(d.estimates.cold_start_s, 200.0);
    assert_eq!(d.estimates.job_s, 40.0);
    assert_eq!(p.states["p"].cold_starts, 1);
}
