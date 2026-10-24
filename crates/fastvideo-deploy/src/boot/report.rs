//! The boot's log lines and its phase reports to fv-control (ADR-0014 §4).
//!
//! Two report protocols exist and fv-control parses both; the boot speaks the
//! one its environment names, byte for byte as the shell boots did:
//!
//! - **pod** (GMI / Brev pods, `POST $FV_ENDPOINT_REPORT_URL` with
//!   `Bearer $FV_ENDPOINT_REPORT_TOKEN`, providers.ts `endpointReport`):
//!   `{"pod":"$FV_POD_ID","phase":…,"url":…,"detail":…}`, phases `weights`,
//!   `tunnel`, `failed`, `upsampler`;
//! - **serverless** (`POST $FV_SLS_REPORT_URL` with `Bearer $FV_WEIGHTS_TOKEN`,
//!   serverless/ingest.ts `slsReport`): the same body with
//!   `pod = "<worker>@<container start, unix s>"` and `url = ""`, phases
//!   `boot`, `cache`, `weights`, `step`, `serve`, `failed`; a `step` detail is
//!   `"<name> <seconds>s (<note>)"` (serverless/weights.ts `applyReport`).
//!
//! The details are cleaned as the shell did (`tr -d '"\\' | tr '\n' ' ' | cut
//! -c1-280`); the body is serialized as JSON with the four keys in that order.
//! `crates/fastvideo-deploy/fixtures/boot-protocol.json` holds the bodies and
//! log markers fv-control parses; a Rust test and a control vitest both read it.

use std::time::Duration;

use serde::Serialize;

use super::env::BootEnv;
use super::sys::{utc_hms, Clock, Http, LogSink, Method, Request};

/// The log line prefix fv-control's boot timeline keys on (boottime.ts).
pub const LOG_PREFIX: &str = "[fv-boot]";

/// The boot's log: `[fv-boot] HH:MM:SS +E.Es <msg>` lines on the container log,
/// and the few untimed markers fv-control and the Brev host watcher match.
pub struct Log<'a> {
    sink: &'a dyn LogSink,
    clock: &'a dyn Clock,
    t0: f64,
}

impl<'a> Log<'a> {
    pub fn new(sink: &'a dyn LogSink, clock: &'a dyn Clock, t0: f64) -> Self {
        Self { sink, clock, t0 }
    }

    pub fn t0(&self) -> f64 {
        self.t0
    }

    /// Seconds since the boot started.
    pub fn elapsed(&self) -> f64 {
        (self.clock.now() - self.t0).max(0.0)
    }

    /// A timed line.
    pub fn say(&self, msg: &str) {
        let now = self.clock.now();
        self.sink.line(&format!(
            "{LOG_PREFIX} {} +{:.1}s {msg}",
            utc_hms(now),
            (now - self.t0).max(0.0)
        ));
    }

    /// An untimed marker: `[fv-boot] <rest>` (`start`, `volume: …`,
    /// `tunnel <url>`), exactly as the shell boots printed it.
    pub fn marker(&self, rest: &str) {
        self.sink.line(&format!("{LOG_PREFIX} {rest}"));
    }

    /// A line as it is (a sidecar's own output).
    pub fn raw(&self, line: &str) {
        self.sink.line(line);
    }

    /// `step <name>: <secs>s (<note>)`.
    pub fn step(&self, name: &str, secs: f64, note: &str) {
        if note.is_empty() {
            self.say(&format!("step {name}: {secs:.1}s"));
        } else {
            self.say(&format!("step {name}: {secs:.1}s ({note})"));
        }
    }
}

/// Which report protocol the environment names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    Pod,
    Serverless,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Target {
    protocol: Protocol,
    url: String,
    token: String,
    pod: String,
}

/// The report sink: fv-control's ingest endpoint, or nothing (a pod without
/// a report URL only logs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reporter {
    target: Option<Target>,
}

#[derive(Serialize)]
struct Body<'a> {
    pod: &'a str,
    phase: &'a str,
    url: &'a str,
    detail: &'a str,
}

/// The cleaned detail: no `"` or `\`, newlines as spaces, at most 280 chars.
pub fn clean_detail(s: &str) -> String {
    s.chars()
        .filter(|c| *c != '"' && *c != '\\')
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .take(280)
        .collect()
}

/// The JSON body of one report.
pub fn encode(pod: &str, phase: &str, url: &str, detail: &str) -> String {
    serde_json::to_string(&Body {
        pod,
        phase,
        url,
        detail,
    })
    .unwrap_or_default()
}

/// A serverless step name as the shell made it (`tr ' /' '__'`).
pub fn step_name(s: &str) -> String {
    s.chars()
        .map(|c| if c == ' ' || c == '/' { '_' } else { c })
        .collect()
}

impl Reporter {
    pub fn none() -> Self {
        Self { target: None }
    }

    /// From the environment: serverless when `FV_SLS_REPORT_URL` is set, else
    /// pod when `FV_ENDPOINT_REPORT_URL` is, else none. `t0` is the boot's
    /// start (unix s): a serverless boot's id is `<worker>@<t0>`.
    pub fn from_env(env: &BootEnv, t0: f64) -> Self {
        if let Some(url) = env.get("FV_SLS_REPORT_URL") {
            let worker = env
                .get("RUNPOD_POD_ID")
                .or_else(|| env.get("FV_POD_ID"))
                .unwrap_or("worker");
            return Self {
                target: Some(Target {
                    protocol: Protocol::Serverless,
                    url: url.to_owned(),
                    token: env.get_or("FV_WEIGHTS_TOKEN", "").to_owned(),
                    pod: format!("{worker}@{}", t0.max(0.0).floor() as u64),
                }),
            };
        }
        if let Some(url) = env.get("FV_ENDPOINT_REPORT_URL") {
            let pod = env
                .get("FV_POD_ID")
                .or_else(|| env.get("RUNPOD_POD_ID"))
                .unwrap_or("worker");
            return Self {
                target: Some(Target {
                    protocol: Protocol::Pod,
                    url: url.to_owned(),
                    token: env.get_or("FV_ENDPOINT_REPORT_TOKEN", "").to_owned(),
                    pod: pod.to_owned(),
                }),
            };
        }
        Self::none()
    }

    pub fn protocol(&self) -> Option<Protocol> {
        self.target.as_ref().map(|t| t.protocol)
    }

    /// The id the reports (and the R2 fetcher's `--pod`) carry.
    pub fn pod(&self) -> Option<&str> {
        self.target.as_ref().map(|t| t.pod.as_str())
    }

    /// The body this sink would send for one report; `None` without a sink.
    pub fn body(&self, phase: &str, url: &str, detail: &str) -> Option<String> {
        let t = self.target.as_ref()?;
        let url = match t.protocol {
            Protocol::Pod => url,
            Protocol::Serverless => "",
        };
        Some(encode(&t.pod, phase, url, &clean_detail(detail)))
    }

    fn send(&self, http: &dyn Http, clock: &dyn Clock, phase: &str, url: &str, detail: &str) {
        let (Some(t), Some(body)) = (self.target.as_ref(), self.body(phase, url, detail)) else {
            return;
        };
        // Pod reports are retried as PROVIDER_BOOT's `report` did (5 x 20 s),
        // and like `curl -sS` there only when no HTTP answer came (a 401 / 404 /
        // 429 is an answer: retrying it would only cost the pauses); serverless
        // ones are best effort once (the shell's `rep`).
        let (tries, timeout) = match t.protocol {
            Protocol::Pod => (5, Duration::from_secs(20)),
            Protocol::Serverless => (1, Duration::from_secs(10)),
        };
        for i in 0..tries {
            let req = Request::new(Method::Post, &t.url, timeout)
                .bearer(&t.token)
                .json(body.clone());
            if http.send(&req).is_ok() {
                return;
            }
            if i + 1 < tries {
                clock.sleep(Duration::from_secs(5));
            }
        }
    }

    /// A pod-protocol report (GMI / Brev): sent only to a pod sink.
    pub fn pod_phase(
        &self,
        http: &dyn Http,
        clock: &dyn Clock,
        phase: &str,
        url: &str,
        detail: &str,
    ) {
        if self.protocol() == Some(Protocol::Pod) {
            self.send(http, clock, phase, url, detail);
        }
    }

    /// A serverless-protocol report: sent only to a serverless sink.
    pub fn sls_phase(&self, http: &dyn Http, clock: &dyn Clock, phase: &str, detail: &str) {
        if self.protocol() == Some(Protocol::Serverless) {
            self.send(http, clock, phase, "", detail);
        }
    }

    /// A timed step (serverless `step` phase: `<name> <secs>s (<note>)`).
    pub fn step(&self, http: &dyn Http, clock: &dyn Clock, name: &str, secs: f64, note: &str) {
        let d = if note.is_empty() {
            format!("{name} {secs:.1}s")
        } else {
            format!("{name} {secs:.1}s ({note})")
        };
        self.sls_phase(http, clock, "step", &d);
    }

    /// The boot failed (both protocols).
    pub fn failed(&self, http: &dyn Http, clock: &dyn Clock, detail: &str) {
        self.send(http, clock, "failed", "", detail);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detail_is_cleaned_like_the_shell() {
        assert_eq!(clean_detail("a \"b\" \\c\nd"), "a b c d");
        assert_eq!(clean_detail(&"x".repeat(400)).len(), 280);
    }

    #[test]
    fn body_has_the_four_keys_in_order() {
        assert_eq!(
            encode("gmi:fv-a-1", "tunnel", "https://x.trycloudflare.com", ""),
            r#"{"pod":"gmi:fv-a-1","phase":"tunnel","url":"https://x.trycloudflare.com","detail":""}"#
        );
    }

    #[test]
    fn sink_from_env() {
        let sls = Reporter::from_env(
            &BootEnv::from_pairs([
                ("FV_SLS_REPORT_URL", "https://c/ingest/v1/sls-report"),
                ("FV_WEIGHTS_TOKEN", "fvw_x"),
                ("RUNPOD_POD_ID", "w1"),
                ("FV_ENDPOINT_REPORT_URL", "https://c/e"),
            ]),
            1_761_300_000.7,
        );
        assert_eq!(sls.protocol(), Some(Protocol::Serverless));
        assert_eq!(sls.pod(), Some("w1@1761300000"));
        assert_eq!(
            sls.body("serve", "u", "x").as_deref(),
            Some(r#"{"pod":"w1@1761300000","phase":"serve","url":"","detail":"x"}"#)
        );
        let pod = Reporter::from_env(
            &BootEnv::from_pairs([
                ("FV_ENDPOINT_REPORT_URL", "https://c/e"),
                ("FV_POD_ID", "brev:fv-a-1"),
            ]),
            1.0,
        );
        assert_eq!(pod.protocol(), Some(Protocol::Pod));
        assert_eq!(pod.pod(), Some("brev:fv-a-1"));
        assert_eq!(
            Reporter::from_env(&BootEnv::default(), 1.0).protocol(),
            None
        );
    }

    #[test]
    fn step_names_like_tr() {
        assert_eq!(step_name("hub h3-base/x"), "hub_h3-base_x");
    }
}
