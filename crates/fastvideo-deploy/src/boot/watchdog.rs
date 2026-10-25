//! The in-container backstop (control cluster/payloads.ts `WATCHDOG`,
//! providers.ts PROVIDER_BOOT's deadline loop). fv-control's cron enforces
//! the same deadline; this one works when fv-control cannot reach the pod.
//! It runs as `fv-boot watchdog`, a process of its own started before the
//! exec (it outlives it).
//!
//! - **Runpod** (an edge front or a standalone pod: `FV_CLUSTER_DEADLINE`,
//!   `FV_BACKSTOP_API_KEY`, `RUNPOD_POD_ID`): at the deadline, or when the
//!   account balance (checked every 60 s) is below `FV_MIN_BALANCE`, the pod
//!   deletes itself through the Runpod REST API. The key is fv-control's
//!   account key (`envCtx` `runpodApiKey` = its `RUNPOD_API_KEY`, the key it
//!   creates and deletes pods with over REST), not the pod-scoped
//!   `RUNPOD_API_KEY` Runpod injects (which REST refuses with a 403,
//!   docs/adr/0023-lab-report-and-ci-bridge.md §4). Only a 2xx, or a 404 (the
//!   pod is already gone), ends the attempts; anything else is retried.
//! - **GMI** (`FV_BACKSTOP_API_KEY` = the GMI key, `FV_BACKSTOP_API`): at the
//!   deadline the container deletes itself (found by `FV_POD_NAME`), and
//!   fv-serve is stopped.
//! - **Brev / other providers**: at the deadline fv-serve is stopped
//!   (fv-control parks or deletes the VM).

use std::time::Duration;

use serde_json::Value;

use super::env::{BootEnv, Kind, Provider};
use super::sys::{kill_cmd, Method, Request, Response};
use super::Ctx;

pub const RUNPOD_REST: &str = "https://rest.runpod.io/v1";
pub const RUNPOD_GRAPHQL: &str = "https://api.runpod.io/graphql";

/// Secrets only the watchdog needs: unset for fv-serve and the boot's other
/// children (as `FV_TUNNEL_TOKEN` is), handed to the watchdog alone.
pub const BACKSTOP_SECRETS: [&str; 1] = ["FV_BACKSTOP_API_KEY"];

/// What the watchdog guards.
#[derive(Clone, Debug, PartialEq)]
pub enum Mode {
    Runpod {
        pod: String,
        key: String,
        deadline: f64,
        min_balance: f64,
    },
    Gmi {
        api: String,
        key: String,
        name: String,
        deadline: f64,
    },
    Stop {
        deadline: f64,
    },
}

/// The watchdog this environment asks for, if any.
pub fn mode(env: &BootEnv) -> Option<Mode> {
    let deadline = env.get("FV_CLUSTER_DEADLINE")?.trim().parse::<f64>().ok()?;
    match env.kind() {
        Kind::Provider(p) => {
            if let (Provider::Gmi, Some(key)) = (&p, env.get("FV_BACKSTOP_API_KEY")) {
                return Some(Mode::Gmi {
                    api: env
                        .get_or("FV_BACKSTOP_API", "https://console.gmicloud.ai/api")
                        .trim_end_matches('/')
                        .to_owned(),
                    key: key.to_owned(),
                    name: env.get_or("FV_POD_NAME", "").to_owned(),
                    deadline,
                });
            }
            Some(Mode::Stop { deadline })
        }
        Kind::ClusterWorker => Some(Mode::Runpod {
            pod: env.get("RUNPOD_POD_ID")?.to_owned(),
            key: env.get("FV_BACKSTOP_API_KEY")?.to_owned(),
            deadline,
            min_balance: env
                .get("FV_MIN_BALANCE")
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0.0),
        }),
        _ => None,
    }
}

/// Starts `fv-boot watchdog` when the environment asks for one.
pub fn start(c: &Ctx<'_>) {
    let Some(m) = mode(&c.env) else { return };
    let what = match &m {
        Mode::Runpod { .. } => "deadline and balance floor: the pod deletes itself",
        Mode::Gmi { .. } => "deadline: the container deletes itself",
        Mode::Stop { .. } => "deadline: fv-serve stops",
    };
    // The backstop key is kept from fv-serve and every other child
    // (`Ctx::unset`); only the watchdog gets it back.
    let mut cmd = c.cmd(c.paths.self_exe.display().to_string()).args([
        "watchdog",
        "--serve-pid",
        &c.pid.to_string(),
    ]);
    for k in BACKSTOP_SECRETS {
        if let Some(v) = c.env.get(k) {
            cmd = cmd.env(k, v);
        }
    }
    match c.deps.spawner.spawn(&cmd) {
        Ok(pid) => c.say(&format!("watchdog: pid {pid} ({what})")),
        Err(e) => c.say(&format!(
            "watchdog: not started: {e} (fv-control's cron still enforces the deadline)"
        )),
    }
}

/// The balance in a GraphQL `{ myself { clientBalance } }` answer.
pub fn balance(body: &[u8]) -> Option<f64> {
    let v: Value = serde_json::from_slice(body).ok()?;
    v.pointer("/data/myself/clientBalance")?.as_f64()
}

/// The id of the object named `name` anywhere in a JSON answer.
pub fn container_id(body: &[u8], name: &str) -> Option<String> {
    fn walk(v: &Value, name: &str) -> Option<String> {
        match v {
            Value::Object(m) => {
                if m.get("name").and_then(Value::as_str) == Some(name) {
                    if let Some(id) = m.get("id").and_then(Value::as_str) {
                        return Some(id.to_owned());
                    }
                }
                m.values().find_map(|x| walk(x, name))
            }
            Value::Array(a) => a.iter().find_map(|x| walk(x, name)),
            _ => None,
        }
    }
    if name.is_empty() {
        return None;
    }
    walk(&serde_json::from_slice(body).ok()?, name)
}

/// Whether a REST `DELETE /pods/{id}` answer means the pod is going or gone:
/// a 2xx, or a 404 (already deleted). A 403 (a key REST refuses), a 5xx or
/// any other status is not, and neither is a transport error.
pub fn runpod_deleted(r: &Response) -> bool {
    r.ok() || r.status == 404
}

fn delete_runpod(c: &Ctx<'_>, pod: &str, key: &str, why: &str) {
    c.say(&format!("[watchdog] {why}: deleting {pod}"));
    for i in 0..3 {
        let req = Request::new(
            Method::Delete,
            format!("{RUNPOD_REST}/pods/{pod}"),
            Duration::from_secs(30),
        )
        .bearer(key);
        match c.deps.http.send(&req) {
            Ok(r) if runpod_deleted(&r) => return,
            Ok(r) => c.say(&format!("[watchdog] delete {pod}: HTTP {}", r.status)),
            Err(e) => c.say(&format!("[watchdog] delete {pod}: {e}")),
        }
        if i < 2 {
            c.deps.clock.sleep(Duration::from_secs(5));
        }
    }
}

/// fv-serve is the boot's own pid (the exec kept it): TERM through the shell's
/// `kill` builtin (no procps in the images; the shell's `pkill -f fv-serve`
/// needed it).
fn stop_serve(c: &Ctx<'_>, serve_pid: Option<u32>) {
    match serve_pid {
        Some(pid) => {
            let _ = c.deps.spawner.output(&kill_cmd("TERM", pid));
        }
        None => c.say("[watchdog] no fv-serve pid (--serve-pid): nothing to stop"),
    }
}

/// The loop (`serve_pid`: fv-serve's pid; `max_rounds`: tests stop it).
pub fn run(c: &Ctx<'_>, m: &Mode, serve_pid: Option<u32>, max_rounds: Option<u64>) {
    let mut n: u64 = 0;
    loop {
        if max_rounds.is_some_and(|k| n >= k) {
            return;
        }
        let now = c.now();
        match m {
            Mode::Runpod {
                pod,
                key,
                deadline,
                min_balance,
            } => {
                if now >= *deadline {
                    delete_runpod(c, pod, key, "deadline");
                    c.deps.clock.sleep(Duration::from_secs(60));
                    n += 1;
                    continue;
                }
                if n.is_multiple_of(2) {
                    let req = Request::new(Method::Post, RUNPOD_GRAPHQL, Duration::from_secs(20))
                        .bearer(key)
                        .json(r#"{"query":"{ myself { clientBalance } }"}"#.into());
                    if let Some(b) = c.deps.http.send(&req).ok().and_then(|r| balance(&r.body)) {
                        if b < *min_balance {
                            delete_runpod(c, pod, key, &format!("balance {b} below {min_balance}"));
                        }
                    }
                }
            }
            Mode::Gmi {
                api,
                key,
                name,
                deadline,
            } => {
                if now >= *deadline {
                    c.say(&format!("[watchdog] deadline: {name}"));
                    let list = Request::new(
                        Method::Get,
                        format!("{api}/v1/containers"),
                        Duration::from_secs(30),
                    )
                    .bearer(key);
                    if let Some(id) = c
                        .deps
                        .http
                        .send(&list)
                        .ok()
                        .and_then(|r| container_id(&r.body, name))
                    {
                        let del = Request::new(
                            Method::Delete,
                            format!("{api}/v1/containers/{id}"),
                            Duration::from_secs(30),
                        )
                        .bearer(key);
                        let _ = c.deps.http.send(&del);
                    }
                    stop_serve(c, serve_pid);
                    c.deps.clock.sleep(Duration::from_secs(60));
                }
            }
            Mode::Stop { deadline } => {
                if now >= *deadline {
                    c.say("[watchdog] deadline: stopping fv-serve");
                    stop_serve(c, serve_pid);
                    c.deps.clock.sleep(Duration::from_secs(60));
                }
            }
        }
        n += 1;
        c.deps.clock.sleep(Duration::from_secs(30));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes() {
        let m = |p: &[(&str, &str)]| mode(&BootEnv::from_pairs(p.iter().copied()));
        assert_eq!(
            m(&[
                ("FV_SERVE_ROLE", "worker"),
                ("FV_CLUSTER_DEADLINE", "100"),
                ("FV_BACKSTOP_API_KEY", "k"),
                ("RUNPOD_POD_ID", "p"),
                ("FV_MIN_BALANCE", "8")
            ]),
            Some(Mode::Runpod {
                pod: "p".into(),
                key: "k".into(),
                deadline: 100.0,
                min_balance: 8.0
            })
        );
        assert_eq!(
            m(&[("FV_SERVE_ROLE", "worker"), ("FV_CLUSTER_DEADLINE", "100")]),
            None
        );
        assert_eq!(
            m(&[
                ("FV_SERVE_ROLE", "worker"),
                ("FV_BACKSTOP_API_KEY", "k"),
                ("RUNPOD_POD_ID", "p")
            ]),
            None
        );
        assert_eq!(
            m(&[
                ("FV_PROVIDER", "gmi"),
                ("FV_CLUSTER_DEADLINE", "5"),
                ("FV_BACKSTOP_API_KEY", "g"),
                ("FV_BACKSTOP_API", "https://gmi/api/"),
                ("FV_POD_NAME", "fv-a")
            ]),
            Some(Mode::Gmi {
                api: "https://gmi/api".into(),
                key: "g".into(),
                name: "fv-a".into(),
                deadline: 5.0
            })
        );
        assert_eq!(
            m(&[("FV_PROVIDER", "brev"), ("FV_CLUSTER_DEADLINE", "5")]),
            Some(Mode::Stop { deadline: 5.0 })
        );
        assert_eq!(m(&[("FV_CLUSTER_DEADLINE", "5")]), None);
    }

    #[test]
    fn answers() {
        assert_eq!(
            balance(br#"{"data":{"myself":{"clientBalance":21.6}}}"#),
            Some(21.6)
        );
        assert_eq!(balance(b"{}"), None);
        let list = br#"{"items":[{"id":"aa-1","name":"fv-b"},{"id":"bb-2","name":"fv-a","x":{"id":"no"}}]}"#;
        assert_eq!(container_id(list, "fv-a").as_deref(), Some("bb-2"));
        assert_eq!(container_id(list, "fv-c"), None);
        assert_eq!(container_id(list, ""), None);
    }
}
