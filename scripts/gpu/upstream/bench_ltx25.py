#!/usr/bin/env python3
"""sol-engine LTX-2.5 RTX5090 cell (models/ltx25/RTX5090/gpu_infer.py), dense or Sol stage 2.

Runs the reference's own driver in-process (same CLI run_ltx25_gpu.sh builds)
so its timings (stage_1/stage_2/video VAE/e2e, peak allocated/reserved) are
the reference methodology: one cold request timed end to end after the
pipeline object is built. Model weights load lazily inside the stages (the
BF16 profile uses --offload cpu), so e2e includes them, as in the reference.

--arm dense keeps the reference driver and its instrumentation but routes every
stage-2 video self-attention call to the dense kernel: the upstream README's
"Dense Stage 2" column. --arm sol is the unmodified driver.
We add: pipeline construction time (build_s) and the process wall time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sol-engine", required=True)
    ap.add_argument("--arm", choices=("dense", "sol"), required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    sys.path.insert(0, a.sol_engine)
    t_proc = time.perf_counter()
    import models.ltx25.RTX5090.gpu_infer as gi
    from models.ltx25.RTX5090 import attention as att

    if a.arm == "dense":
        from contextlib import contextmanager

        @contextmanager
        def stage2_dense(self, enabled):  # noqa: ARG001
            prev = self.enabled
            self.enabled = False
            try:
                yield
            finally:
                self.enabled = prev

        att.LTX25Stage2SolAttention.stage2 = stage2_dense
    orig_build = gi.build_pipeline
    info: dict = {}

    def timed_build(args):
        t = time.perf_counter()
        p = orig_build(args)
        info["build_s"] = time.perf_counter() - t
        return p

    gi.build_pipeline = timed_build
    sys.argv = ["gpu_infer", *rest]
    rc = 0
    try:
        gi.main()
    except SystemExit as e:
        rc = int(e.code or 0)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        info["error"] = f"{type(e).__name__}: {e}"
        rc = 1
    info["process_s"] = time.perf_counter() - t_proc
    info["arm"] = a.arm
    metrics_path = None
    for i, x in enumerate(rest):
        if x == "--metrics" and i + 1 < len(rest):
            metrics_path = Path(rest[i + 1])
    if metrics_path and metrics_path.exists():
        info["reference_metrics"] = json.loads(metrics_path.read_text())
    Path(a.result).write_text(json.dumps(info, indent=2, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())
