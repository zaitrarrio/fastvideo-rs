//! `fv-runpod-sim [addr]`: serves the Runpod queue simulator
//! (`fastvideo_deploy::runpod::sim`) and prints the `RUNPOD_*` exports a
//! worker needs, for local end-to-end runs:
//!
//! ```text
//! cargo run -p fastvideo-deploy --bin fv-runpod-sim -- 127.0.0.1:8765 > /tmp/sim.env &
//! set -a; . /tmp/sim.env; set +a
//! FV_SERVE_MODE=runpod-queue cargo run -p fastvideo-serve --features http-client -- --config configs/serve/fake.toml &
//! curl -s localhost:8765/v2/sim-ep/run -d '{"input":{"kind":"http","method":"GET","path":"/fv/v1/capabilities"}}'
//! ```

use fastvideo_deploy::runpod::sim::Sim;

fn main() {
    let addr = std::env::args().nth(1).unwrap_or_else(|| "127.0.0.1:8765".into());
    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("fv-runpod-sim: {e}");
            std::process::exit(1);
        }
    };
    rt.block_on(async move {
        let key = format!("sim-{}", std::process::id());
        let sim = Sim::new(key).with_long_poll(std::time::Duration::from_secs(5));
        let (base, handle) = match sim.serve(&addr).await {
            Ok(x) => x,
            Err(e) => {
                eprintln!("fv-runpod-sim: bind {addr}: {e}");
                std::process::exit(1);
            }
        };
        for (k, v) in sim.env(&base, "local-worker", 10_000) {
            println!("{k}='{v}'");
        }
        eprintln!("fv-runpod-sim: listening on {base} (client API: {base}/v2/sim-ep/run, /status/<id>)");
        let _ = handle.await;
    });
}
