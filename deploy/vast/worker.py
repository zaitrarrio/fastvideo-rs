"""Vast serverless PyWorker forwarder for fv-serve (docs/serve/design.md §6.2).

The PyWorker (``start_server.sh`` at a pinned ``PYWORKER_REF`` with the
``vastai`` SDK at a pinned ``SDK_VERSION``) runs this file as ``python -m
worker``. It owns the Vast side of the protocol (signature check against
``REPORT_ADDR/pubkey/``, ``/worker_status/`` metrics, TLS) and forwards each
request's ``payload`` to fv-serve on localhost:

    client -> POST https://<worker>/fv/v1/forward
              {"auth_data": {...}, "payload": <job envelope>}
    worker -> POST http://127.0.0.1:8000/fv/v1/forward  <job envelope>

The envelope is fv-serve's native Runpod job input (``kind: http`` or
``kind: info``), e.g. ``{"kind": "http", "method": "POST", "path":
"/v2/video_generation", "headers": {...}, "body": {...}, "wait": true}``;
fv-serve dispatches it to the same in-process router its HTTP port serves.
Batch only: ``kind: stream`` is refused.

Readiness: the PyWorker tails fv-serve's log (``MODEL_LOG``) for
``FV-SERVE READY``; its benchmark calls ``/fv/v1/capabilities`` (plus one
tiny generation when ``FV_VAST_BENCH_JOB`` names a model).

No third-party imports beyond the ``vastai`` SDK, so ``python -m worker``
works in the venv ``start_server.sh`` creates.
"""

import os

from vastai.serverless.server.worker import (  # type: ignore[import-not-found]
    BenchmarkConfig,
    HandlerConfig,
    LogActionConfig,
    Worker,
    WorkerConfig,
)

MODEL_SERVER_URL = os.environ.get("FV_MODEL_SERVER_URL", "http://127.0.0.1")
MODEL_SERVER_PORT = int(os.environ.get("FV_MODEL_SERVER_PORT", "8000"))
MODEL_LOG = os.environ.get("MODEL_LOG", "/var/log/fv-serve.log")
FORWARD_ROUTE = "/fv/v1/forward"
READY_LINE = "FV-SERVE READY"


def _api_headers():
    """fv-serve's own API key (raw) for forwarded calls, if configured."""
    key = os.environ.get("FV_VAST_API_KEY", "")
    return {"authorization": f"Bearer {key}"} if key else {}


def benchmark_dataset():
    """Envelopes the PyWorker benchmark sends (sets the worker's max_perf)."""
    data = [
        {"kind": "http", "method": "GET", "path": "/fv/v1/capabilities", "headers": _api_headers()},
    ]
    model = os.environ.get("FV_VAST_BENCH_JOB", "")
    if model:
        data.append(
            {
                "kind": "http",
                "method": "POST",
                "path": "/fv/v1/jobs",
                "headers": _api_headers(),
                "body": {"model": model, "prompt": "a red fox in fresh snow", "seed": 1, "size": "480x832", "num_frames": 17},
                "wait": True,
                "timeout_s": 900,
            }
        )
    return data


def workload(payload):
    """Relative cost: generations dominate, metadata calls are cheap."""
    if payload.get("kind") == "http" and str(payload.get("method", "POST")).upper() == "POST":
        return 100.0
    return 1.0


def request_parser(msg):
    """Rejects anything that is not a batch envelope before it is queued."""
    kind = msg.get("kind", "http" if "path" in msg else None)
    if kind not in ("http", "info"):
        raise ValueError("payload must be a fv-serve job envelope with kind http or info")
    return msg


def build_config():
    return WorkerConfig(
        model_server_url=MODEL_SERVER_URL,
        model_server_port=MODEL_SERVER_PORT,
        model_log_file=MODEL_LOG,
        model_healthcheck_url="/healthz",
        handlers=[
            HandlerConfig(
                route=FORWARD_ROUTE,
                allow_parallel_requests=False,
                max_queue_time=600.0,
                request_parser=request_parser,
                workload_calculator=workload,
                benchmark_config=BenchmarkConfig(dataset=benchmark_dataset(), runs=2, concurrency=1),
            )
        ],
        log_action_config=LogActionConfig(
            on_load=[READY_LINE],
            on_error=["fv-serve: ", "fv-serve failed", "model loading failed"],
            on_info=["listening", "runpod worker"],
        ),
        max_sessions=1,
    )


if __name__ == "__main__":
    Worker(build_config()).run()
