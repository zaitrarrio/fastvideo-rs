//! `fv-autoscale-sim [--seed N] [--seeds K] [--json]`: the simulation suite
//! over K seeds from N (waits pooled, totals per run) as a Markdown table
//! (docs/serve/gateway.md "Simulation") or JSON lines.
//! `--timeline <trace prefix> <family>` prints one run minute by minute.

use fastvideo_autoscale::sim::harness::{markdown, merge, run, standard_suite};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seed = args
        .iter()
        .position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_260_928u64);
    let json = args.iter().any(|a| a == "--json");
    // --timeline <trace prefix> <family>: per-minute queue,
    // workers and endpoint min for each strategy.
    if let Some(i) = args.iter().position(|a| a == "--timeline") {
        let (tr, fam) = (args.get(i + 1).cloned().unwrap_or_default(), args.get(i + 2).cloned().unwrap_or_default());
        for s in standard_suite(seed).iter().filter(|s| s.trace.starts_with(&tr) && s.profile.family == fam) {
            let r = run(s);
            println!("# {} {} {}", r.trace, r.family, r.strategy);
            for (m, q, w, min) in &r.timeline {
                println!("{m:>6.0} queued={q:<4} workers={w:<3} min={min}");
            }
        }
        return;
    }
    let seeds: u64 = args
        .iter()
        .position(|a| a == "--seeds")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let reports = merge((seed..seed + seeds).flat_map(|s| standard_suite(s).iter().map(run).collect::<Vec<_>>()).collect());
    if json {
        for r in &reports {
            println!("{}", serde_json::to_string(r).unwrap_or_default());
        }
    } else {
        print!("{}", markdown(&reports));
    }
}
