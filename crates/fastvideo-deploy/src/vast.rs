//! Vast assets (design §6.2; research-deploy §3, §5.1).
//!
//! - [`env_dict`]: the REST `env` **object** Vast requires (`"env must be a
//!   dict"`): published ports as `{"-p 8000:8000": "1"}` keys plus plain
//!   `KEY: value` pairs. Secrets never go here: they are Vast account env
//!   vars (the same upper-case `FV_*` names), injected by Vast.
//! - [`onstart`]: the `ssh_direct` onstart script that launches `fv-serve`
//!   (sshd is PID 1 under every `ssh_*` runtype; the image entrypoint never
//!   runs), logging to [`MODEL_LOG`] so the PyWorker can tail it.
//! - [`WORKER_PY`]: the Vast serverless PyWorker forwarder
//!   (`deploy/vast/worker.py`), baked into the `serve` image.

use serde_json::{Map, Value};

/// fv-serve's log on Vast (the PyWorker's `MODEL_LOG`).
pub const MODEL_LOG: &str = "/var/log/fv-serve.log";

/// The readiness line fv-serve prints (design §6.1).
pub const READY_LINE: &str = "FV-SERVE READY";

/// The PyWorker forwarder.
pub const WORKER_PY: &str = include_str!("../../../deploy/vast/worker.py");

/// A published port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Port {
    pub port: u32,
    pub udp: bool,
}

impl Port {
    pub const fn tcp(port: u32) -> Self {
        Self { port, udp: false }
    }
    pub const fn udp(port: u32) -> Self {
        Self { port, udp: true }
    }
    /// The docker flag Vast parses: `-p 8000:8000` / `-p 70010:70010/udp`.
    pub fn flag(&self) -> String {
        format!("-p {0}:{0}{1}", self.port, if self.udp { "/udp" } else { "" })
    }
}

/// Default fv-serve ports on Vast: HTTP, ICE-UDP (symmetric), ICE-TCP (symmetric).
pub const DEFAULT_PORTS: [Port; 3] = [Port::tcp(8000), Port::udp(70010), Port::tcp(70000)];

/// Names that must never be placed in the env dict (they are account env
/// vars on Vast).
pub const SECRET_NAMES: &[&str] = &[
    "FV_CF_ACCOUNT_ID",
    "FV_CF_API_TOKEN",
    "FV_D1_DATABASE_ID",
    "FV_R2_BUCKET",
    "FV_R2_ENDPOINT",
    "FV_R2_ACCESS_KEY_ID",
    "FV_R2_SECRET_ACCESS_KEY",
    "FV_WEBHOOK_ED25519_KEY",
    "FV_URL_SIGNING_KEY",
    "FV_WHIP_TOKEN",
    "HF_TOKEN",
];

/// Builds the `env` object. `Err` if a secret name is passed.
pub fn env_dict(ports: &[Port], vars: &[(&str, &str)]) -> Result<Value, String> {
    let mut m = Map::new();
    for p in ports {
        m.insert(p.flag(), Value::String("1".into()));
    }
    for (k, v) in vars {
        if SECRET_NAMES.contains(k) {
            return Err(format!("{k} is a secret: set it as a Vast account env var, not in the instance env"));
        }
        m.insert((*k).to_owned(), Value::String((*v).to_owned()));
    }
    Ok(Value::Object(m))
}

/// The onstart script: exports the env to `/etc/environment` (ssh sessions
/// see it), then starts fv-serve detached with its log at [`MODEL_LOG`].
/// `args` are fv-serve's arguments (e.g. `--config /etc/fv/vast.toml`).
pub fn onstart(args: &str) -> String {
    format!(
        "#!/bin/bash\n\
         env | grep -E '^(FV_|VAST_|PUBLIC_IPADDR|CONTAINER_ID|RUST_LOG)' >> /etc/environment\n\
         mkdir -p /var/log /fvstate\n\
         nohup /opt/fastvideo-rs/bin/fv-serve {args} >> {MODEL_LOG} 2>&1 &\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_is_a_dict_with_port_keys() {
        let v = env_dict(&DEFAULT_PORTS, &[("FV_SERVE_MODE", "http"), ("FV_ENGINE", "fake")]).unwrap();
        let o = v.as_object().unwrap();
        assert_eq!(o["-p 8000:8000"], "1");
        assert_eq!(o["-p 70010:70010/udp"], "1");
        assert_eq!(o["-p 70000:70000"], "1");
        assert_eq!(o["FV_ENGINE"], "fake");
        assert!(env_dict(&[], &[("FV_CF_API_TOKEN", "x")]).is_err());
    }

    #[test]
    fn onstart_fits_and_starts_the_server() {
        let s = onstart("--config /etc/fv/vast.toml");
        assert!(s.len() < 4048, "Vast caps onstart at 4048 chars");
        assert!(s.contains("nohup /opt/fastvideo-rs/bin/fv-serve --config /etc/fv/vast.toml >> /var/log/fv-serve.log"));
    }

    #[test]
    fn worker_py_forwards_to_fv_serve() {
        assert!(WORKER_PY.contains("/fv/v1/forward"));
        assert!(WORKER_PY.contains(READY_LINE));
        assert!(WORKER_PY.contains(MODEL_LOG));
    }
}
